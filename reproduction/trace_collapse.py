"""Trace where and when embeddings collapse during pre-training.

Pre-training uses no labels. The final linear probe reports validation accuracy only.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from graph_data import read_graph, ppr_nodes, spd_targets
from model import UniGraph2
from run import seed_all, probe


def spread(h):
    """Cosine to centroid and effective rank, as in diagnose.py."""
    z = F.normalize(h.float(), dim=-1)
    power = torch.linalg.svdvals(z - z.mean(0)).square()
    p = power / power.sum().clamp_min(1e-20)
    return {'cos': F.cosine_similarity(z, z.mean(0)[None]).mean().item(),
            'rank': (-(p * p.clamp_min(1e-20).log()).sum()).exp().item()}


@torch.no_grad()
def stages(model, x, edges):
    model.eval()
    h = model.moe(x)
    out = {'moe': spread(h)}
    for i, layer in enumerate(model.gnn):
        h = layer(h, edges)
        out[f'gat{i}'] = spread(h)
    model.train()
    return out


def update_shape(w, w0):
    """Spectral size of the accumulated update relative to init, and its stable rank."""
    s = torch.linalg.svdvals((w - w0).float())
    return {'ratio': (s[0] / torch.linalg.svdvals(w0.float())[0]).item(),
            'stable_rank': (s.square().sum() / s[0].square().clamp_min(1e-20)).item()}


def constant_loss(x, center, gamma=2.):
    """Feature loss of predicting one fixed vector for every node."""
    return (1 - F.cosine_similarity(x, center[None])).clamp_min(0).pow(gamma).mean().item()


def objective(model, x, target, edges, domain, pairs, distances, corrupt=True, loss='sce'):
    """model.objective with a separate reconstruction target; corrupt=False is plain autoencoding."""
    count = max(1, min(len(x) - 1, int(len(x) * model.mask_rate)))
    masked = torch.randperm(len(x), device=x.device)[:count]
    corrupted = x.clone()
    if corrupt:
        corrupted[masked] = model.mask_token
    h = model.encode(corrupted, edges)
    prediction = model.decoders[domain](h[masked])
    if loss == 'mse':
        feature = F.mse_loss(prediction, target[masked])
    else:
        feature = (1 - F.cosine_similarity(prediction, target[masked], dim=-1)).clamp_min(0).pow(model.gamma).mean()
    estimate = model.spd(torch.cat([h[pairs[0]], h[pairs[1]]], -1)).flatten()
    structural = F.mse_loss(estimate, distances)
    total = feature + model.lambda_spd * structural
    return total, {'feature': feature.item(), 'spd': structural.item(), 'total': total.item()}


# A variant is '+'-joined tokens, e.g. 'ae+z+sgd'.
TOKENS = {
    'base': {},
    'ae': {'corrupt': False},
    'z': {'zscore': True},
    'zin': {'zscore': True, 'raw_target': True},
    'meaninit': {'meaninit': True},
    'warmup': {'warmup': .1},
    'lr1e-4': {'lr': 1e-4},
    'sgd': {'opt': 'sgd'},
    'mse': {'loss': 'mse'},
    'nogat': {'nogat': True},
    'noln': {'noln': True},
    'nomoe': {'nomoe': True},
    'nodrop': {'dropout': 0.},
    'spd0': {'lambda_spd': 0.},
    'mask05': {'mask_rate': .5},
}


def config(name):
    cfg = {}
    for token in name.split('+'):
        cfg.update(TOKENS[token])
    return cfg


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, default=Path('/home/pusit.see/dataset/raw/paper_2602_00067/mm_graph'))
    p.add_argument('--variants', nargs='+', default=['base'])
    p.add_argument('--steps', type=int, default=160)
    p.add_argument('--every', type=int, default=20)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(4)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed_all(0)
    source = read_graph(args.data_root, 'cloth-copurchase')
    target = read_graph(args.data_root, 'ele-fashion', limit=4096)
    rng = np.random.default_rng(0)
    eligible = np.flatnonzero(np.diff(source.adjacency.indptr) > 0)
    pool = []
    for seed in rng.choice(eligible, 32, replace=False):
        x, edges, adj = source.subgraph(ppr_nodes(source, int(seed), 256))
        pairs, distances = spd_targets(adj, 4096, rng)
        pool.append((x, edges, pairs, distances))
    mu, sigma = source.x.mean(0), source.x.std(0)
    mean_distance = torch.cat([d for *_, d in pool]).mean().item()
    tx, te, _ = target.subgraph(np.arange(len(target.x)))
    y = target.y.to(device)
    report = {'inputs': {
        'feature_norm': source.x.norm(dim=1).mean().item(),
        'constant_loss_global_mean': float(np.mean([constant_loss(x, mu) for x, *_ in pool])),
        'constant_loss_subgraph_mean': float(np.mean([constant_loss(x, x.mean(0)) for x, *_ in pool])),
        'constant_spd_loss': float(np.mean([d.var(unbiased=False).item() for *_, d in pool])),
        'mean_distance': mean_distance}}
    print('INPUTS', report['inputs'], flush=True)
    for name in args.variants:
        cfg = config(name)
        lr, warmup = cfg.pop('lr', 1e-3), cfg.pop('warmup', 0)
        corrupt, meaninit, zscore = cfg.pop('corrupt', True), cfg.pop('meaninit', False), cfg.pop('zscore', False)
        opt_name, loss_name = cfg.pop('opt', 'adamw'), cfg.pop('loss', 'sce')
        nogat, noln, nomoe = cfg.pop('nogat', False), cfg.pop('noln', False), cfg.pop('nomoe', False)
        raw_target = cfg.pop('raw_target', False)
        prep = (lambda v: (v - mu) / sigma) if zscore else (lambda v: v)
        seed_all(0)
        model = UniGraph2(512, ['cloth-copurchase'], **cfg)
        if nomoe:
            model.moe = nn.Sequential(nn.Linear(512, model.decoders['cloth-copurchase'][0].in_features))
        if nogat:
            model.gnn = nn.ModuleList()
        if noln:
            for layer in model.gnn:
                layer.norm = nn.Identity()
        model = model.to(device)
        if meaninit:
            with torch.no_grad():
                last = model.decoders['cloth-copurchase'][-1]
                last.weight.zero_()
                last.bias.copy_(mu)
                model.spd[-1].weight.zero_()
                model.spd[-1].bias.fill_(mean_distance)
        opt = (torch.optim.SGD(model.parameters(), lr=.1, momentum=.9) if opt_name == 'sgd'
               else torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=.01))
        w = max(1, int(warmup * args.steps))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda t: min(1., (t + 1) / w) * .5 * (1 + np.cos(np.pi * min(t, args.steps) / args.steps)) if warmup else 1.)
        first = model.moe[0].weight if nomoe else model.moe.experts[0][0].weight
        first0 = first.detach().clone()
        ev = (prep(tx).to(device), te.to(device))
        xi, ei, *_ = max(pool, key=lambda item: len(item[0]))
        ein = (prep(xi).to(device), ei.to(device))
        order = np.random.default_rng(0)
        trace = []
        for step in range(args.steps + 1):
            if step % args.every == 0 or step == args.steps:
                row = {'step': step, 'target': stages(model, *ev), 'source': list(stages(model, *ein).values())[-1],
                       'moe_update': update_shape(first.detach(), first0) if step else None}
                trace.append(row)
            if step == args.steps:
                break
            x, edges, pairs, distances = pool[order.integers(len(pool))]
            raw, x = x.to(device), prep(x).to(device)
            loss, detail = objective(model, x, raw if raw_target else x, edges.to(device), 'cloth-copurchase',
                                     pairs.to(device), distances.to(device), corrupt, loss_name)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            opt.step()
            sched.step()
            if step % args.every == 0:
                row.update(detail)
                print(name, step, {a: round(b, 4) for a, b in detail.items()},
                      {a: (round(b['cos'], 5), round(b['rank'], 1)) for a, b in row['target'].items() if a in ('moe', 'gat3')},
                      'src', round(row['source']['rank'], 1),
                      'dW', row['moe_update'] and {a: round(b, 2) for a, b in row['moe_update'].items()}, flush=True)
        model.eval()
        with torch.no_grad():
            emb = model.encode(*ev)
        metrics, _ = probe(emb, y, target.splits, 0)
        row['valid_accuracy'] = metrics['valid_accuracy']
        print(name, 'FINAL', {a: (round(b['cos'], 5), round(b['rank'], 1)) for a, b in row['target'].items() if a in ('moe', 'gat3')},
              'valid_accuracy', round(metrics['valid_accuracy'], 4), flush=True)
        report[name] = trace
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
