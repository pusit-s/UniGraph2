"""Paper-guided UniGraph2 core, implemented with PyTorch (no DGL dependency)."""
import torch
from torch import nn
from torch.nn import functional as F


class AttentionHead(nn.Module):
    """Single-head scaled dot-product self-attention over every node of the (sub)graph.
    Q/K/V each map input_dim -> hidden, matching the MLP expert's parameter count."""
    def __init__(self, input_dim, hidden):
        super().__init__()
        self.q, self.k, self.v = (nn.Linear(input_dim, hidden) for _ in range(3))

    def forward(self, queries, context):
        scores = self.q(queries) @ self.k(context).T / self.k.out_features ** .5
        return scores.softmax(-1) @ self.v(context)


class SharedSelfAttention(nn.Module):
    """Shared multihead node attention before routing, scoped to one graph call.

    Project back to the input feature width so the residual and existing MLP MoE
    retain their dimensions. No position encodings or attention dropout are used.
    """
    def __init__(self, input_dim, heads=4):
        super().__init__()
        if input_dim < 1 or heads < 1 or input_dim % heads:
            raise ValueError('shared attention input_dim must be divisible by positive attention_heads')
        self.input_dim, self.heads, self.width = input_dim, heads, input_dim // heads
        self.q, self.k, self.v = (nn.Linear(input_dim, input_dim) for _ in range(3))
        self.out = nn.Linear(input_dim, input_dim)
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, x):
        if x.ndim != 2 or x.shape[1] != self.input_dim:
            raise ValueError('shared attention expects one graph with shape [nodes, input_dim]')
        projections = [projection(x).reshape(len(x), self.heads, self.width).transpose(0, 1).unsqueeze(0)
                       for projection in (self.q, self.k, self.v)]
        attended = F.scaled_dot_product_attention(*projections, dropout_p=0.)
        attended = attended.squeeze(0).transpose(0, 1).reshape(len(x), self.input_dim)
        return self.norm(x + self.out(attended))


class MoE(nn.Module):
    def __init__(self, input_dim, hidden, experts=8, top_k=0, expert='mlp'):
        super().__init__()
        if not 0 <= top_k <= experts:
            raise ValueError('top_k must be between zero (dense) and num_experts')
        if expert not in ('mlp', 'attention'):
            raise ValueError("expert must be 'mlp' or 'attention'")
        self.top_k, self.hidden, self.attention = top_k or experts, hidden, expert == 'attention'
        self.gate = nn.Sequential(nn.Linear(input_dim, hidden), nn.GELU(), nn.Linear(hidden, experts))
        self.experts = nn.ModuleList([
            AttentionHead(input_dim, hidden) if self.attention else
            nn.Sequential(nn.Linear(input_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden))
            for _ in range(experts)])

    def routing_weights(self, x):
        """Actual post-top-k probabilities, with zeros for inactive experts."""
        scores, indices = self.gate(x).topk(self.top_k, dim=-1)
        weights = scores.softmax(-1)
        return scores.new_zeros(len(x), len(self.experts)).scatter(1, indices, weights)

    def forward(self, x):
        scores, indices = self.gate(x).topk(self.top_k, dim=-1)
        weights = scores.softmax(-1)
        out = x.new_zeros(len(x), self.hidden)
        for k, expert in enumerate(self.experts):
            rows, slots = (indices == k).nonzero(as_tuple=True)
            if rows.numel():
                # Attention experts: routed nodes query, all nodes act as keys/values.
                y = expert(x[rows], x) if self.attention else expert(x[rows])
                out.index_add_(0, rows, y * weights[rows, slots, None])
        return out


class GATLayer(nn.Module):
    """Additive multihead GAT with destination softmax and self loops."""
    def __init__(self, hidden, heads, dropout):
        super().__init__()
        if hidden % heads:
            raise ValueError('hidden must be divisible by heads')
        self.heads, self.width, self.dropout = heads, hidden // heads, dropout
        self.linear = nn.Linear(hidden, hidden, bias=False)
        self.a_src = nn.Parameter(torch.empty(heads, self.width))
        self.a_dst = nn.Parameter(torch.empty(heads, self.width))
        self.bias = nn.Parameter(torch.zeros(hidden))
        self.norm = nn.LayerNorm(hidden)
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)

    def forward(self, x, edge_index):
        n = len(x)
        h = self.linear(F.dropout(x, self.dropout, self.training)).view(n, self.heads, self.width)
        src, dst = edge_index
        logits = F.leaky_relu((h * self.a_src).sum(-1)[src] + (h * self.a_dst).sum(-1)[dst], .2)
        index = dst[:, None].expand(-1, self.heads)
        maximum = logits.new_full((n, self.heads), -torch.inf)
        maximum.scatter_reduce_(0, index, logits.detach(), reduce='amax', include_self=True)
        exp = (logits - maximum[dst]).exp()
        denom = exp.new_zeros(n, self.heads).scatter_add_(0, index, exp)
        alpha = F.dropout(exp / denom[dst].clamp_min(1e-12), self.dropout, self.training)
        out = h.new_zeros(n, self.heads, self.width)
        out.index_add_(0, dst, h[src] * alpha[..., None])
        return F.elu(self.norm(out.reshape(n, -1) + self.bias + x))


class UniGraph2(nn.Module):
    def __init__(self, input_dim, domains, hidden=1024, experts=8, layers=4,
                 heads=4, dropout=.4, mask_rate=.8, gamma=2., lambda_spd=.1, top_k=0, expert='mlp',
                 shared_attention=False, attention_heads=4):
        super().__init__()
        if not 0 < mask_rate < 1 or layers < 1 or gamma < 1 or lambda_spd < 0:
            raise ValueError('invalid model configuration')
        self.mask_rate, self.gamma, self.lambda_spd = mask_rate, gamma, lambda_spd
        # Frozen per-dimension input standardization, set from pre-training domains only.
        # Raw CLIP rows have norm ~0.8, so without it bias terms swamp node-specific signal.
        self.register_buffer('feature_mean', torch.zeros(input_dim))
        self.register_buffer('feature_std', torch.ones(input_dim))
        self.mask_token = nn.Parameter(torch.zeros(input_dim))
        if shared_attention and expert != 'mlp':
            raise ValueError('shared pre-router attention requires MLP experts; use expert=mlp')
        self.shared_attention_enabled = bool(shared_attention)
        # Identity adds no parameters and consumes no RNG, preserving defaults.
        self.shared_attention = SharedSelfAttention(input_dim, attention_heads) if shared_attention else nn.Identity()
        self.moe = MoE(input_dim, hidden, experts, top_k, expert)
        self.gnn = nn.ModuleList([GATLayer(hidden, heads, dropout) for _ in range(layers)])
        self.decoders = nn.ModuleDict({d: nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(),
                                                       nn.Linear(hidden, input_dim)) for d in domains})
        self.spd = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def preprocess(self, x, masked=None):
        """Standardize, optionally mask, then attend within this graph/subgraph."""
        x = (x - self.feature_mean) / self.feature_std
        if masked is not None:
            x[masked] = self.mask_token
        return self.shared_attention(x)

    def encode(self, x, edges, masked=None):
        h = self.moe(self.preprocess(x, masked))
        for layer in self.gnn:
            h = layer(h, edges)
        return h

    def routing_weights(self, x):
        """Unmasked routing using the same graph context and preprocessing as encode."""
        return self.moe.routing_weights(self.preprocess(x))

    def reconstruction_losses(self, x, edges, domain, masked):
        h = self.encode(x, edges, masked)
        prediction = self.decoders[domain](h[masked])
        return (1 - F.cosine_similarity(prediction, x[masked], dim=-1)).clamp_min(0).pow(self.gamma)

    def objective(self, x, edges, domain, pairs, distances):
        if len(x) < 2:
            raise ValueError('need at least two nodes')
        count = max(1, min(len(x) - 1, int(len(x) * self.mask_rate)))
        masked = torch.randperm(len(x), device=x.device)[:count]
        h = self.encode(x, edges, masked)
        prediction = self.decoders[domain](h[masked])
        feature = (1 - F.cosine_similarity(prediction, x[masked], dim=-1)).clamp_min(0).pow(self.gamma).mean()
        if distances.numel():
            estimate = self.spd(torch.cat([h[pairs[0]], h[pairs[1]]], -1)).flatten()
            structural = F.mse_loss(estimate, distances)
        else:
            structural = h.sum() * 0
        total = feature + self.lambda_spd * structural
        return total, {'feature': feature.detach().item(), 'spd': structural.detach().item(), 'total': total.detach().item()}
