"""MM-Graph data adapters and bounded subgraph sampling."""
from collections import deque
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
from scipy import sparse
from scipy.sparse.csgraph import shortest_path


def load(path):
    return torch.load(path, map_location='cpu', weights_only=False)


def edge_tensor(value):
    value = torch.as_tensor(value, dtype=torch.long)
    if value.ndim != 2:
        raise ValueError(f'edges must be a matrix, got {value.shape}')
    if value.shape[1] == 2:
        value = value.T
    elif value.shape[0] != 2:
        raise ValueError(f'edges must have shape [E,2] or [2,E], got {value.shape}')
    return value.contiguous()


def fuse_features(value):
    """Average only available, nonzero finite modalities in a shared CLIP space."""
    if isinstance(value, dict):
        tensors = list(value.values())
    elif isinstance(value, (list, tuple)):
        tensors = list(value)
    elif isinstance(value, torch.Tensor) and value.ndim == 3:
        # MM-Graph release stores [nodes, modalities, dimension].
        tensors = list(value.unbind(1))
    else:
        tensors = [value]
    if any(t.ndim != 2 or t.shape != tensors[0].shape for t in tensors):
        raise ValueError('CLIP modality tensors must have identical [N,D] shape')
    stacked = torch.stack([t.float() for t in tensors], 1)
    present = torch.isfinite(stacked).all(-1) & (stacked.nan_to_num().abs().sum(-1) > 0)
    if not present.any(1).all():
        raise ValueError('some nodes have no available finite modality')
    return (stacked.nan_to_num() * present[..., None]).sum(1) / present.sum(1)[:, None]


@dataclass
class Graph:
    name: str
    x: torch.Tensor
    adjacency: sparse.csr_matrix
    y: torch.Tensor = None
    splits: dict = None

    def subgraph(self, ids):
        ids = np.asarray(ids, dtype=np.int64)
        adj = self.adjacency[ids][:, ids].tocsr()
        row, col = adj.nonzero()
        loops = np.arange(len(ids))
        edges = torch.from_numpy(np.stack([np.r_[row, loops], np.r_[col, loops]])).long()
        return self.x[torch.from_numpy(ids)], edges, adj


def read_graph(root, name, edge_split='random', limit=0):
    folder = Path(root) / name
    cached = load(folder / 'clip_feat.pt')
    if isinstance(cached, torch.Tensor) and cached.ndim == 2:
        if cached.shape[1] != 1024:
            raise ValueError('expected concatenated 512d text + 512d image CLIP features')
        cached = list(cached.split(512, dim=1))
    x = fuse_features(cached)
    y = splits = None
    if (folder / 'nc_edges-nodeid.pt').exists():
        edges = edge_tensor(load(folder / 'nc_edges-nodeid.pt'))
        y = torch.as_tensor(load(folder / 'labels-w-missing.pt'), dtype=torch.long).flatten()
        splits = load(folder / 'split.pt')
        splits = {k: torch.as_tensor(v).flatten() for k,v in splits.items()}
    else:
        data = load(folder / 'lp-edge-split.pt')
        if edge_split in data:
            data = data[edge_split]
        train = data['train']
        if isinstance(train, dict):
            train = train['edge'] if 'edge' in train else torch.stack([train['source_node'],train['target_node']],1)
        edges = edge_tensor(train)
    n = len(x)
    if edges.numel() and (edges.min() < 0 or edges.max() >= n):
        raise ValueError('graph edges do not align with feature rows')
    if limit:
        n = min(limit, n)
        x = x[:n]
        keep = (edges < n).all(0)
        edges = edges[:, keep]
        if y is not None:
            y = y[:n]
            splits = {k: torch.as_tensor(v).flatten() for k, v in splits.items()}
            splits = {k: v[:n].nonzero().flatten() if v.dtype == torch.bool else v[v < n] for k, v in splits.items()}
    row, col = edges.numpy()
    adj = sparse.coo_matrix((np.ones(len(row)), (row, col)), shape=(n, n)).tocsr()
    adj = adj.maximum(adj.T)
    adj.setdiag(0)
    adj.eliminate_zeros()
    adj.data[:] = 1.
    return Graph(name, x, adj, y, splits)


def ppr_nodes(graph, seed, topk=256, alpha=.15, epsilon=1e-4):
    """Approximate PPR by local residual push; bounded to 50,000 pushes."""
    adj = graph.adjacency
    residual, score = {int(seed): 1.}, {}
    queue, queued = deque([int(seed)]), {int(seed)}
    pushes = 0
    while queue and pushes < 50000:
        i = queue.popleft()
        queued.remove(i)
        r = residual.get(i, 0.)
        degree = adj.indptr[i+1] - adj.indptr[i]
        residual[i] = 0.
        score[i] = score.get(i, 0.) + alpha * r
        if degree == 0:
            score[i] += (1-alpha) * r
            continue
        amount = (1-alpha) * r / degree
        for j in adj.indices[adj.indptr[i]:adj.indptr[i+1]]:
            j = int(j)
            residual[j] = residual.get(j, 0.) + amount
            deg_j = max(1, adj.indptr[j+1] - adj.indptr[j])
            if residual[j] / deg_j > epsilon and j not in queued:
                queue.append(j)
                queued.add(j)
        pushes += 1
    ranked = sorted(score, key=lambda i: (-score[i], i))[:topk]
    if seed not in ranked:
        ranked = [int(seed)] + ranked[:topk-1]
    return np.asarray(ranked, dtype=np.int64)


def spd_targets(adj, pair_limit, rng):
    # Sample targets inside the induced PPR subgraph. Disconnected pairs are omitted.
    distances = shortest_path(adj, directed=False, unweighted=True)
    src, dst = np.where(np.isfinite(distances))
    if pair_limit and len(src) > pair_limit:
        ids = rng.choice(len(src), pair_limit, replace=False)
        src, dst = src[ids], dst[ids]
    return torch.from_numpy(np.stack([src, dst])).long(), torch.from_numpy(distances[src, dst]).float()
