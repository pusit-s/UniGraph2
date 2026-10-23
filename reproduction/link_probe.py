"""Frozen, symmetric linear edge probing on bounded MM-Graph target graphs.

Official positive split membership is preserved. Within a bounded node cohort we
draw fixed, distinct non-neighbors instead of silently discarding out-of-cohort
official negative candidates. These sampled-MRR results are not full benchmark
MRR. All methods/seeds share the same positive rows and negative candidates.
"""
import copy
import hashlib
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy import sparse


def _unique_pairs(edges, nodes):
    pairs = torch.as_tensor(edges, dtype=torch.long).cpu().numpy()
    keep = (pairs >= 0).all(0) & (pairs < nodes).all(0) & (pairs[0] != pairs[1])
    pairs = pairs[:, keep]
    keys = np.minimum(*pairs)*nodes + np.maximum(*pairs)
    _, first = np.unique(keys, return_index=True)
    first.sort()
    return pairs[:, first], keys[first]


def prepare_link_splits(edge_splits, nodes, seed=1729, negatives=100,
                        train_limit=4096, eval_limit=1024):
    """Deduplicate undirected positives, remove observed held-out edges, fix negatives."""
    if nodes < negatives+2 or min(negatives, train_limit, eval_limit) < 1:
        raise ValueError('invalid link-probe limits')
    rng = np.random.default_rng(seed)
    unique = {name: _unique_pairs(edge_splits[name], nodes) for name in ['train', 'valid', 'test']}
    all_edges = np.concatenate([value[0] for value in unique.values()], axis=1)
    forbidden = sparse.coo_matrix((np.ones(all_edges.shape[1]), tuple(all_edges)), shape=(nodes, nodes)).tocsr()
    forbidden = forbidden.maximum(forbidden.T)
    forbidden.setdiag(1)
    forbidden.eliminate_zeros()
    used, prepared, counts = set(), {}, {}
    for name in ['train', 'valid', 'test']:
        pairs, keys = unique[name]
        keep = np.asarray([int(key) not in used for key in keys], dtype=bool)
        used.update(map(int, keys))
        pairs = pairs[:, keep]
        degree = np.diff(forbidden.indptr)
        required = 1 if name == 'train' else negatives
        eligible = nodes-degree[pairs[0]] >= required
        pairs = pairs[:, eligible]
        available = pairs.shape[1]
        cap = train_limit if name == 'train' else eval_limit
        if available > cap:
            pairs = pairs[:, np.sort(rng.choice(available, cap, replace=False))]
        if not pairs.shape[1]:
            raise ValueError(f'no non-overlapping {name} positives inside the node cohort')
        candidates = []
        for source in pairs[0]:
            allowed = np.ones(nodes, dtype=bool)
            allowed[forbidden.indices[forbidden.indptr[source]:forbidden.indptr[source+1]]] = False
            candidates.append(rng.choice(np.flatnonzero(allowed), required, replace=False))
        prepared[name] = {'positive': torch.from_numpy(pairs.copy()),
                          'negative_targets': torch.from_numpy(np.stack(candidates)).long()}
        counts[name] = {'unique_within_cohort': len(keys), 'overlap_with_earlier_splits_removed': int((~keep).sum()),
                        'insufficient_non_neighbors_removed': int((~eligible).sum()),
                        'eligible_positives': available, 'selected_positives': pairs.shape[1],
                        'negatives_per_positive': required}
    # Hash logical arrays rather than serialization internals.
    digest = hashlib.sha256()
    for name, split in prepared.items():
        for key, value in split.items():
            digest.update(f'{name}/{key}/{list(value.shape)}'.encode())
            digest.update(value.numpy().tobytes())
    protocol = {'task': 'link_prediction', 'primary_metric': 'mrr', 'nodes': nodes, 'split_seed': seed,
                'candidate_protocol': 'fixed sampled non-neighbors within the node cohort; all known positives and self loops excluded',
                'official_positive_splits': True, 'official_negative_candidates': False,
                'undirected': True, 'cohort_counts': counts, 'split_sha256': digest.hexdigest(),
                'head': 'linear classifier on Hadamard products of unit-normalized frozen embeddings',
                'ranking_ties': 'average optimistic and pessimistic rank, as in the MM-Graph evaluator'}
    return prepared, protocol


def ranking_metrics(scores):
    """Column zero is positive; remaining columns are distinct fixed negatives."""
    if scores.ndim != 2 or scores.shape[1] < 2 or not torch.isfinite(scores).all():
        raise ValueError('ranking scores must be a finite matrix with positive and negative columns')
    positive, negative = scores[:, :1], scores[:, 1:]
    rank = 1 + .5*((negative > positive).sum(-1) + (negative >= positive).sum(-1))
    return {'mrr': rank.float().reciprocal().mean().item(),
            **{f'hits@{k}': (rank <= k).float().mean().item() for k in [1, 3, 10]}}


def _candidate_features(x, split):
    positive = split['positive'].to(x.device)
    candidates = torch.cat([positive[1, :, None], split['negative_targets'].to(x.device)], 1)
    return x[positive[0], None] * x[candidates]


def lp_probe(embeddings, splits, seed, epochs=5000, patience=100, validation_every=5):
    """Select by validation MRR; evaluate the selected head on test exactly once."""
    if min(epochs, patience, validation_every) < 1:
        raise ValueError('probe limits must be positive')
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    x = F.normalize(embeddings.detach(), dim=-1)
    train = _candidate_features(x, splits['train'])
    train = train.reshape(-1, x.shape[-1])
    labels = torch.zeros(len(splits['train']['positive'][0]), 2, device=x.device)
    labels[:, 0] = 1
    labels = labels.flatten()
    validation_features = _candidate_features(x, splits['valid'])
    head = nn.Linear(x.shape[-1], 1).to(x.device)
    optimizer = torch.optim.Adam(head.parameters(), lr=.01)
    best, state, selected, stale, last_checked = -1., None, 0, 0, 0
    for epoch in range(1, epochs+1):
        loss = F.binary_cross_entropy_with_logits(head(train).flatten(), labels)
        if not torch.isfinite(loss):
            raise RuntimeError('nonfinite edge-probe objective')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if epoch % validation_every and epoch != epochs:
            continue
        with torch.no_grad():
            validation = ranking_metrics(head(validation_features).squeeze(-1))['mrr']
        if validation > best:
            best, state, selected, stale = validation, copy.deepcopy(head.state_dict()), epoch, 0
        else:
            stale += epoch-last_checked
        last_checked = epoch
        if stale >= patience:
            break
    head.load_state_dict(state)
    # Test candidates and scores are only materialized after validation selection.
    with torch.no_grad():
        test = ranking_metrics(head(_candidate_features(x, splits['test'])).squeeze(-1))
    return {'valid_mrr': best, **{f'test_{key}': value for key, value in test.items()},
            'selected_epoch': selected, 'epochs_run': epoch, 'validation_every': validation_every,
            'split_sizes': {name: len(split['positive'][0]) for name, split in splits.items()}}, head.state_dict()
