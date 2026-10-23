"""Label-free graph adaptation of RoMA (arXiv:2511.07419, Eqs. 5--10).

RoMA is router-only post-training, not a replacement expert architecture. Here
frozen CLIP features replace task-text embeddings, and low masked reconstruction
error replaces answer correctness. This is an explicit surrogate for the paper's
supervised successful set, not a claim to reproduce its LLM experiments.
"""
import math
import numpy as np
import torch
from torch.nn import functional as F


def freeze_for_router(model):
    """Freeze every parameter except the gate; keep frozen modules in eval mode."""
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    for parameter in model.moe.gate.parameters():
        parameter.requires_grad_(True)
    return list(model.moe.gate.parameters())


@torch.no_grad()
def build_bank(model, graphs, pools, fraction=.5, seed=0):
    """Deduplicate masked-node errors across a fixed, pre-training-only pool.

    Select the best fraction separately in each domain, so an easy domain cannot
    displace all other domains. Reference routes are unmasked and frozen. Masked
    inputs share one token, so applying semantic alignment to their identical
    routes would create a degenerate training signal.
    """
    if not 0 < fraction <= 1:
        raise ValueError('success fraction must be in (0, 1]')
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    features, routes, identities, errors = [], [], [], []
    contextual = getattr(model, 'shared_attention_enabled', False)
    # Do not let bank construction change post-training masks or dropout RNG.
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        for domain_index, (graph, pool) in enumerate(zip(graphs, pools)):
            sums, counts, route_sums = {}, {}, {}
            for x, edges, _, _, ids in pool:
                x, edges = x.to(device), edges.to(device)
                count = max(1, min(len(x)-1, int(len(x)*model.mask_rate)))
                masked = torch.randperm(len(x), device=device)[:count]
                losses = model.reconstruction_losses(x, edges, graph.name, masked)
                # Shared attention makes the gate depend on the complete source
                # subgraph. Freeze routes in that original context, not in an
                # artificial batch containing only the eventual anchor nodes.
                pool_routes = model.routing_weights(x)[masked] if contextual else None
                for index, (node, loss) in enumerate(zip(np.asarray(ids)[masked.cpu().numpy()], losses.cpu().tolist())):
                    node = int(node)
                    sums[node] = sums.get(node, 0.) + loss
                    counts[node] = counts.get(node, 0) + 1
                    if contextual:
                        route_sums[node] = route_sums.get(node, torch.zeros_like(pool_routes[index])) + pool_routes[index]
            ranked = sorted(sums, key=lambda node: (sums[node]/counts[node], node))
            selected = ranked[:max(2, math.ceil(len(ranked)*fraction))]
            if len(selected) < 2:
                raise ValueError('RoMA needs at least two distinct scored nodes per domain')
            x = graph.x[selected].to(device)
            features.append(F.normalize(x, dim=-1))
            routes.append(torch.stack([route_sums[node]/counts[node] for node in selected]) if contextual else model.routing_weights(x))
            identities.extend((domain_index, node) for node in selected)
            errors.extend(sums[node]/counts[node] for node in selected)
    model.train(was_training)
    return {'features': torch.cat(features).detach(), 'routes': torch.cat(routes).detach(),
            'identities': torch.tensor(identities, device=device),
            'errors': torch.tensor(errors, device=device),
            'success_definition': 'lowest mean masked reconstruction error fraction per pre-training domain',
            'routing_context': ('mean of complete original unmasked PPR-pool routes over scored occurrences'
                                if contextual else 'pointwise unmasked routing'),
            'fraction': fraction}


def neighbor_targets(x, identities, bank, k=5, sigma=.4):
    """Gaussian kNN on unit CLIP features (paper Eq. 5), excluding self IDs."""
    if k < 1 or sigma <= 0:
        raise ValueError('k and sigma must be positive')
    with torch.no_grad():
        distance2 = (2 - 2*F.normalize(x.detach(), dim=-1) @ bank['features'].T).clamp_min(0)
        same = (identities[:, None] == bank['identities'][None]).all(-1)
        distance2.masked_fill_(same, torch.inf)
        nearest, indices = distance2.topk(min(k, distance2.shape[1]), largest=False)
        valid = torch.isfinite(nearest)
        if not valid.any(1).all():
            raise ValueError('each query must have a non-self successful neighbor')
        weights = (-nearest/(2*sigma*sigma)).softmax(-1)
        return bank['routes'][indices].detach(), weights.detach()


def manifold_loss(routes, neighbors, weights):
    """Weighted squared L2 routing distance, summed over experts, mean over nodes."""
    return ((routes[:, None] - neighbors.detach()).square().sum(-1) * weights.detach()).sum(-1).mean()


def posttrain(model, graphs, pools, args, output):
    """Identical router task-loss control and RoMA branches from one checkpoint."""
    parameters = freeze_for_router(model)
    device = next(model.parameters()).device
    bank = None
    if args.router_posttrain == 'roma':
        bank = build_bank(model, graphs, pools, args.roma_success_fraction, args.seed + 7000)
        torch.save({k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in bank.items()}, output/'roma_bank.pt')
    # A separate stream makes task and RoMA use identical masks and subgraph order.
    torch.manual_seed(args.seed + 10000)
    np_rng = np.random.default_rng(args.seed + 10000)
    optimizer = torch.optim.AdamW(parameters, lr=args.router_lr, weight_decay=args.weight_decay)
    total = args.router_epochs * sum(map(len, pools))
    warm = max(1, int(args.warmup * total))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min(1., (step+1)/warm) * .5*(1+np.cos(np.pi*step/total)) if args.warmup else 1.)
    history, step = [], 0
    for epoch in range(1, args.router_epochs+1):
        schedule = [(i, j) for i, pool in enumerate(pools) for j in range(len(pool))]
        np_rng.shuffle(schedule)
        metrics = []
        for i, j in schedule:
            x, edges, pairs, distances, ids = pools[i][j]
            x, edges = x.to(device), edges.to(device)
            optimizer.zero_grad(set_to_none=True)
            task, detail = model.objective(x, edges, graphs[i].name, pairs.to(device), distances.to(device))
            alignment = task.new_zeros(())
            if bank is not None:
                identities = torch.stack([torch.full((len(x),), i, device=device), torch.as_tensor(ids, device=device)], -1)
                neighbors, weights = neighbor_targets(x, identities, bank, args.roma_k, args.roma_sigma)
                alignment = manifold_loss(model.routing_weights(x), neighbors, weights)
            coefficient = args.roma_lambda * min(1., (step+1)/warm) if bank is not None else 0.
            loss = task + coefficient * alignment
            if not torch.isfinite(loss):
                raise RuntimeError('nonfinite router post-training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.)
            optimizer.step()
            scheduler.step()
            metrics.append({**detail, 'manifold': alignment.item(), 'lambda': coefficient, 'combined': loss.item()})
            step += 1
        row = {'epoch': epoch, **{key: float(np.mean([m[key] for m in metrics])) for key in metrics[0]}}
        history.append(row)
        print('ROUTER', args.router_posttrain, row, flush=True)
    return history
