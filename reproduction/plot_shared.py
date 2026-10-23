"""Validate nine architecture/stage cases and export actual routing manifolds.

Run on a compute node. Raw CLIP stays in its source directory; plotted node IDs,
learned routes, coordinates and all probe evidence are exported for inspection.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from compare_shared import CORE_SOURCES
from plot_roma import COLORS, MARKERS, metric_of, neighborhood_metrics, probe_scores


ARCHITECTURE_LABELS = {'mlp': 'MLP experts', 'attention': 'Per-expert attention', 'shared': 'Shared attention + MLP'}
MODES = ['none', 'task', 'roma']
CONDITIONS = [(architecture, mode) for architecture in ARCHITECTURE_LABELS for mode in MODES]


def checkpoint_digest(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8*1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def case_label(architecture, mode):
    return ARCHITECTURE_LABELS[architecture] + {'none': '', 'task': ' / router task', 'roma': ' / RoMA'}[mode]


def review_shared(root, seeds):
    """Require complete matching inputs/protocols and exact own-baseline lineage."""
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('comparison requires distinct encoder seeds')
    records, reference = {}, None
    ignored = {'output', 'seed', 'expert', 'shared_attention', 'router_posttrain', 'resume_pretrain'}
    for seed in seeds:
        baselines = {}
        for architecture, mode in CONDITIONS:
            path = root/f'seed-{seed}'/f'{architecture}-{mode}'
            report = json.loads((path/'results.json').read_text())
            if report.get('status') != 'completed' or set(report['results']) != {'CLIP', 'NoPretrain', 'UniGraph2'}:
                raise ValueError(f'incomplete architecture case: {path}')
            cfg = report['args']
            expected_expert = 'attention' if architecture == 'attention' else 'mlp'
            if (cfg['seed'], cfg['expert'], bool(cfg.get('shared_attention', False)), cfg['router_posttrain']) != (seed, expected_expert, architecture == 'shared', mode):
                raise ValueError(f'architecture/stage mismatch: {path}')
            if cfg['evaluate'] in cfg['pretrain']:
                raise ValueError(f'target leaked into encoder pre-training: {path}')
            kwargs = report['model_kwargs']
            if kwargs['expert'] != expected_expert or bool(kwargs.get('shared_attention', False)) != (architecture == 'shared'):
                raise ValueError(f'model kwargs disagree with architecture: {path}')
            if kwargs.get('attention_heads', 4) != cfg.get('attention_heads', 4):
                raise ValueError(f'attention head metadata mismatch: {path}')
            comparable = {key: value for key, value in cfg.items() if key not in ignored}
            controlled = (comparable, report['graphs'], report['data_files'], report.get('evaluation_protocol'), metric_of(report))
            if reference is None:
                reference = (controlled, report['source_sha256'])
                for filename, metadata in report['data_files'].items():
                    stat = Path(filename).stat()
                    if (stat.st_size, stat.st_mtime_ns) != (metadata['bytes'], metadata['mtime_ns']):
                        raise ValueError(f'source data changed since training: {filename}')
            if controlled != reference[0]:
                raise ValueError(f'unmatched data, protocol or training settings: {path}')
            if any(report['source_sha256'].get(name) != reference[1].get(name) or name not in report['source_sha256'] for name in CORE_SOURCES):
                raise ValueError(f'mixed core implementation: {path}')
            for representation, result in report['results'].items():
                probes = result['runs']
                if [probe['seed'] for probe in probes] != cfg['probe_seeds']:
                    raise ValueError(f'missing or repeated probe seeds: {path}/{representation}')
                scores = probe_scores(report, representation)
                if not scores or not all(0 <= score <= 100 for score in scores):
                    raise ValueError(f'invalid metric values: {path}/{representation}')
                if not np.isclose(np.mean(scores), result[f'mean_test_{metric_of(report)}_percent']):
                    raise ValueError(f'cached mean disagrees with probe evidence: {path}')
                std_key = f'std_test_{metric_of(report)}_percent'
                if std_key in result and not np.isclose(np.std(scores), result[std_key]):
                    raise ValueError(f'cached probe spread disagrees with evidence: {path}')
                if any(not 0 <= probe[f'valid_{metric_of(report)}'] <= 1 or not 1 <= probe['selected_epoch'] <= probe['epochs_run'] for probe in probes):
                    raise ValueError(f'invalid validation selection evidence: {path}')
            if mode == 'none':
                baselines[architecture] = checkpoint_digest(path/'checkpoint.pt')
            else:
                lineage = report.get('pretrain_lineage')
                baseline = root/f'seed-{seed}'/f'{architecture}-none'
                if not lineage or Path(lineage['run']).resolve() != baseline.resolve() or lineage['checkpoint_sha256'] != baselines[architecture]:
                    raise ValueError(f'case did not branch from its own matching baseline: {path}')
            records[(seed, architecture, mode)] = report
        clip_reference = records[(seed, 'mlp', 'none')]['results']['CLIP']
        for architecture in ARCHITECTURE_LABELS:
            baseline = records[(seed, architecture, 'none')]
            if baseline['results']['CLIP'] != clip_reference:
                raise ValueError(f'unmatched raw CLIP reference: seed {seed}, {architecture}')
            for mode in ['task', 'roma']:
                candidate = records[(seed, architecture, mode)]
                if any(candidate['results'][representation] != baseline['results'][representation] for representation in ['CLIP', 'NoPretrain']):
                    raise ValueError(f'unmatched frozen references: seed {seed}, {architecture}, {mode}')
    return records


def summarize_shared(records, seeds):
    rows = []
    for architecture, mode in CONDITIONS:
        means = np.asarray([np.mean(probe_scores(records[(seed, architecture, mode)])) for seed in seeds])
        row = {'architecture': architecture, 'method': mode, 'label': case_label(architecture, mode),
               'primary_metric': metric_of(records[(seeds[0], architecture, mode)]),
               'mean_test_score_percent': float(means.mean()), 'std_across_encoder_seeds': float(means.std()),
               'seed_means': means.tolist(),
               'job_ids': [records[(seed, architecture, mode)]['environment']['slurm_job_id'] for seed in seeds],
               'parameter_counts': records[(seeds[0], architecture, mode)].get('parameter_counts'),
               'attention_heads': records[(seeds[0], architecture, mode)]['args'].get('attention_heads', 4) if architecture == 'shared' else None}
        for comparison_mode, key in [('none', 'paired_delta_vs_baseline'), ('task', 'paired_delta_vs_router_task')]:
            if comparison_mode == 'task' and mode != 'roma':
                continue
            reference = [np.mean(probe_scores(records[(seed, architecture, comparison_mode)])) for seed in seeds]
            delta = means-reference
            row[key] = {'mean': float(delta.mean()), 'population_std': float(delta.std()), 'per_seed': delta.tolist()}
        rows.append(row)
    return rows


def write_scores(path, records, dataset=None):
    with path.open('w') as handle:
        writer = csv.writer(handle)
        writer.writerow(['dataset', 'primary_metric', 'encoder_seed', 'architecture', 'posttrain', 'representation', 'probe_seed', 'valid_score', 'test_score', 'selected_epoch', 'slurm_job_id'])
        for (seed, architecture, mode), report in records.items():
            metric = metric_of(report)
            for representation, result in report['results'].items():
                for probe in result['runs']:
                    writer.writerow([dataset or report['args']['evaluate'], metric, seed, architecture, mode, representation, probe['seed'], probe[f'valid_{metric}'], probe[f'test_{metric}'], probe['selected_epoch'], report['environment']['slurm_job_id']])


def export_figures(root, records, seeds, rows, n_nodes=1500, plot_seed=42):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from scipy.stats import gaussian_kde
    from sklearn.cluster import KMeans
    from sklearn.neighbors import NearestNeighbors
    import sklearn
    import torch
    import umap
    from importlib.metadata import version
    from graph_data import read_graph
    destination = root/'figures'
    destination.mkdir(exist_ok=True)
    seed = seeds[0]
    first = records[(seed, 'mlp', 'none')]
    cfg, dataset = first['args'], first['args']['evaluate']
    artifacts = {f'{architecture}-{mode}': torch.load(root/f'seed-{seed}'/f'{architecture}-{mode}'/'embeddings.pt', map_location='cpu', weights_only=False) for architecture, mode in CONDITIONS}
    base = artifacts['mlp-none']
    for artifact in artifacts.values():
        torch.testing.assert_close(artifact['node_ids'], base['node_ids'], rtol=0, atol=0)
    target = read_graph(cfg['data_root'], dataset, limit=cfg['eval_limit'])
    selected = np.sort(np.random.default_rng(plot_seed).choice(len(base['node_ids']), min(n_nodes, len(base['node_ids'])), replace=False))
    node_ids = base['node_ids'][selected].numpy()
    clip = target.x[node_ids].numpy()
    clip = clip/np.linalg.norm(clip, axis=1, keepdims=True).clip(1e-20)
    groups = KMeans(n_clusters=4, n_init=10, random_state=plot_seed).fit_predict(clip)
    order = sorted(range(4), key=lambda group: node_ids[groups == group].min())
    groups = np.asarray([order.index(group) for group in groups])
    features = {'CLIP': clip, **{name: artifact['routing'][selected].numpy() for name, artifact in artifacts.items()}}
    umap_config = {'n_neighbors': 15, 'min_dist': .1, 'metric': 'euclidean', 'random_state': plot_seed, 'n_jobs': 1, 'n_components': 2}
    coordinates = {name: umap.UMAP(**umap_config).fit_transform(value) for name, value in features.items()}
    nearest = NearestNeighbors(n_neighbors=16).fit(clip).kneighbors(clip, return_distance=False)
    semantic_neighbors = np.asarray([[other for other in candidates if other != index][:15] for index, candidates in enumerate(nearest)])
    metrics = {name: neighborhood_metrics(value, semantic_neighbors, groups) for name, value in features.items() if name != 'CLIP'}
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'pdf.fonttype': 42, 'svg.fonttype': 'none', 'axes.spines.top': False, 'axes.spines.right': False})
    legend = [Line2D([], [], marker=MARKERS[group], color='none', markerfacecolor=COLORS[group], markeredgecolor='none', markersize=6, label=f'CLIP cluster {group+1}') for group in range(4)]
    panels = [('CLIP', '(a) CLIP embedding'), ('mlp-none', '(b) MLP routing'), ('attention-none', '(c) Per-expert attention'), ('shared-none', '(d) Shared attention + MLP'), ('shared-roma', '(e) Shared attention + RoMA')]
    fig = plt.figure(figsize=(19, 5), facecolor='white')
    grid = fig.add_gridspec(2, 10, width_ratios=[1, .14]*5, height_ratios=[.16, 1], left=.02, right=.985, bottom=.23, top=.80, hspace=.03, wspace=.13)
    for column, (name, title) in enumerate(panels):
        ax = fig.add_subplot(grid[1, 2*column])
        top = fig.add_subplot(grid[0, 2*column], sharex=ax)
        right = fig.add_subplot(grid[1, 2*column+1], sharey=ax)
        for group in range(4):
            points = coordinates[name][groups == group]
            ax.scatter(*points.T, s=7, c=COLORS[group], marker=MARKERS[group], alpha=.68, linewidths=0, rasterized=True)
            for marginal, values, vertical in [(top, points[:, 0], False), (right, points[:, 1], True)]:
                if len(values) > 2 and values.std() > 1e-10:
                    ticks = np.linspace(values.min()-values.std(), values.max()+values.std(), 160)
                    density = gaussian_kde(values)(ticks)
                    marginal.plot(density if vertical else ticks, ticks if vertical else density, color=COLORS[group], lw=.9)
        ax.set_xlabel(title, labelpad=9)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color('#777777')
            spine.set_linewidth(.65)
        top.axis('off')
        right.axis('off')
    fig.suptitle(f'Routing architectures in UniGraph2: {dataset}', y=.98, fontsize=16)
    fig.legend(handles=legend, loc='upper center', bbox_to_anchor=(.5, .925), ncol=4, frameon=False)
    fig.text(.5, .105, f"{len(selected):,} fixed nodes; encoder seed {seed}; routing top-k {cfg['top_k']} (0 = dense). Shared attention has {cfg.get('attention_heads', 4)} heads.", ha='center')
    fig.text(.5, .055, 'Fixed unsupervised CLIP colors; independent UMAP axes. Real routing weights, no label-optimized oracle.', ha='center', fontsize=9)
    fig.text(.5, .015, 'Visual separation alone does not establish manifold alignment or task improvement.', ha='center', fontsize=9)
    for extension in ['png', 'pdf', 'svg']:
        fig.savefig(destination/f'figure3-routing.{extension}', dpi=200)
    plt.close(fig)
    fig, axes = plt.subplots(3, 3, figsize=(12, 11))
    for ax, (architecture, mode) in zip(axes.flat, CONDITIONS):
        name = f'{architecture}-{mode}'
        for group in range(4):
            points = coordinates[name][groups == group]
            ax.scatter(*points.T, s=7, c=COLORS[group], marker=MARKERS[group], alpha=.65, linewidths=0, rasterized=True)
        ax.set_title(case_label(architecture, mode), fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(f"CLIP kNN overlap@15: {metrics[name]['semantic_knn_overlap_at_15']:.3f}")
    fig.suptitle('All architecture and router-tuning conditions', y=.995, fontsize=15)
    fig.legend(handles=legend, loc='upper center', bbox_to_anchor=(.5, .96), ncol=4, frameon=False)
    fig.text(.5, .015, 'Independent UMAP layouts. kNN overlap is measured before UMAP in the original spaces.', ha='center', fontsize=10)
    fig.subplots_adjust(top=.90, bottom=.075, hspace=.35)
    fig.savefig(destination/'routing-controls.png', dpi=180)
    fig.savefig(destination/'routing-controls.pdf')
    plt.close(fig)
    colors = ['#A7BBD2', '#6F9CC8', '#3675B8', '#D5BA94', '#E4B260', '#D58C27', '#D5AAC0', '#D987AA', '#C85C87']
    fig, ax = plt.subplots(figsize=(12, 7))
    means = [row['mean_test_score_percent'] for row in rows]
    spreads = [row['std_across_encoder_seeds'] for row in rows]
    positions = np.arange(len(rows))
    ax.barh(positions, means, xerr=spreads, capsize=4, color=colors)
    for index, (mean, spread) in enumerate(zip(means, spreads)):
        ax.text(mean+spread+.7, index, f'{mean:.2f} ± {spread:.2f}', va='center', fontsize=9)
    ax.set_yticks(positions, [row['label'] for row in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel('Test accuracy (%)' if metric_of(first) == 'accuracy' else 'Test sampled MRR (%)')
    ax.set_title(f'Architecture and RoMA comparison: {dataset}')
    ax.grid(axis='x', color='#E5E5E5', linewidth=.7)
    ax.set_axisbelow(True)
    fig.text(.5, .015, f"Mean ± population SD over {len(seeds)} encoder seeds, each averaging {len(cfg['probe_seeds'])} probe seeds. Additional attention parameters are not matched away.", ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, 1))
    fig.savefig(destination/'performance.png', dpi=180)
    fig.savefig(destination/'performance.pdf')
    plt.close(fig)
    np.savez_compressed(destination/'manifold-data.npz', node_ids=node_ids, clip_clusters=groups,
                        **{f'features_{name}': value for name, value in features.items() if name != 'CLIP'},
                        **{f'umap_{name}': value for name, value in coordinates.items()})
    provenance = {'encoder_seed': seed, 'sampling_seed': plot_seed, 'nodes': len(selected), 'top_k': cfg['top_k'],
                  'clip_source': str(Path(cfg['data_root'])/dataset/'clip_feat.pt'),
                  'cluster_method': 'KMeans on unit-normalized raw CLIP, four clusters, n_init=10; no labels',
                  'cluster_sizes': np.bincount(groups).tolist(), 'umap': umap_config, 'metrics': metrics,
                  'versions': {'numpy': np.__version__, 'torch': torch.__version__, 'sklearn': sklearn.__version__, 'matplotlib': matplotlib.__version__, 'umap-learn': version('umap-learn')},
                  'note': 'Independent layouts; alignment diagnostics use original spaces. Selected nodes and seed are never chosen for test performance or apparent cluster separation.'}
    (destination/'visualization.json').write_text(json.dumps(provenance, indent=2))
    return provenance


def write_dataset_report(root, records, seeds, rows, provenance):
    first = next(iter(records.values()))
    cfg, metric = first['args'], metric_of(first)
    evidence = {'status': 'completed', 'encoder_seeds': seeds, 'probe_seeds': cfg['probe_seeds'],
                'primary_metric': metric, 'summary': rows, 'evaluation_protocol': first.get('evaluation_protocol'),
                'visualization': provenance,
                'runs': [{'encoder_seed': seed, 'architecture': architecture, 'method': mode, 'report': report} for (seed, architecture, mode), report in records.items()]}
    (root/'comparison.json').write_text(json.dumps(evidence, indent=2))
    write_scores(root/'scores.csv', records)
    metric_label = 'accuracy' if metric == 'accuracy' else 'sampled MRR'
    lines = [f"# UniGraph2 architecture comparison: {cfg['evaluate']}", '',
             f"Completed nine-condition comparison. Mean ± population SD across {len(seeds)} encoder seeds, each averaging {len(cfg['probe_seeds'])} probe seeds.", '',
             f'| Architecture / stage | Test {metric_label} (%) | Seed means (%) | Slurm jobs |', '| --- | ---: | --- | --- |']
    for row in rows:
        lines.append(f"| {row['label']} | {row['mean_test_score_percent']:.2f} ± {row['std_across_encoder_seeds']:.2f} | {', '.join(f'{value:.2f}' for value in row['seed_means'])} | {', '.join(row['job_ids'])} |")
    lines += ['', '![Performance](figures/performance.png)', '', '![Routing manifolds](figures/figure3-routing.png)', '',
              '## Architectures and matched protocol', '',
              'MLP is the original router → selected feed-forward experts → GAT. Per-expert attention keeps the earlier attention expert implementation. Shared attention applies learned multihead Q/K/V scaled dot-product attention, output projection, residual and LayerNorm before one router and its MLP experts; its GAT backbone is unchanged. Attention operates only on the current graph/subgraph. All layers are unmasked during frozen evaluation.', '',
              f"All cases use source {', '.join(cfg['pretrain'])}, {cfg['epochs']} × {cfg['steps']} source updates per domain, lr {cfg['lr']}, source-only standardization {cfg['standardize']}, width {cfg['hidden']}, {cfg['experts']} experts, and routing top-k {cfg['top_k']} (zero means dense). Shared attention has {cfg.get('attention_heads', 4)} heads. Source and target graphs are disjoint. Architecture comparisons fix these hyperparameters and the budgets; the extra shared-attention parameters are not matched away.", '',
              f"Task-only and RoMA add {cfg['router_epochs']} × {cfg['steps']} gate-only updates per domain at lr {cfg['router_lr']}, each from its own exact baseline checkpoint. Shared attention, experts, GAT, mask token and decoders remain frozen during these stages. RoMA uses source-only reconstruction-quality anchors, λ={cfg['roma_lambda']}, k={cfg['roma_k']}, σ={cfg['roma_sigma']}. Contextual shared-attention teacher routes are averaged over the complete original PPR-pool contexts in which a scored anchor occurs; selected anchors are not treated as a new attention graph.", '',
              '## Paired effects', '', '| Architecture | RoMA − baseline (points) | RoMA − task-only (points) |', '| --- | ---: | ---: | ---: |']
    for row in rows:
        if row['method'] == 'roma':
            a, b = row['paired_delta_vs_baseline'], row['paired_delta_vs_router_task']
            lines.append(f"| {ARCHITECTURE_LABELS[row['architecture']]} | {a['mean']:+.3f} ± {a['population_std']:.3f} | {b['mean']:+.3f} ± {b['population_std']:.3f} |")
    lines += ['', '## Model parameter counts', '', '| Architecture | Total parameters | Router parameters | Shared heads |', '| --- | ---: | ---: | ---: |']
    for row in rows:
        if row['method'] == 'none':
            counts = row.get('parameter_counts') or {}
            lines.append(f"| {ARCHITECTURE_LABELS[row['architecture']]} | {counts.get('total', 'unrecorded')} | {counts.get('router', 'unrecorded')} | {row['attention_heads'] or '—'} |")
    if metric == 'mrr':
        lines += ['', f"LP ranks each positive against {cfg['lp_negatives']} fixed sampled non-neighbors in the bounded target cohort. Training-edge-only message passing, removal of held-out undirected edges visible in earlier splits, and validation-only head selection prevent direct held-out-edge leakage. Candidates differ from the official full-graph benchmark.", '', f"Protocol: `{json.dumps(first['evaluation_protocol'], sort_keys=True)}`."]
    lines += ['', '## Original-space routing diagnostics', '',
              f"Same {provenance['nodes']} nodes and four fixed unsupervised CLIP clusters in every view, from the first declared encoder seed ({provenance['encoder_seed']}). Independent UMAP axes do not establish alignment; kNN overlap is measured before dimensionality reduction. No evaluation labels or label-optimized oracle are used in these plots.", '',
              '| Case | CLIP kNN overlap@15 | CLIP-cluster silhouette | Mean routing-probability SD | Effective rank |', '| --- | ---: | ---: | ---: | ---: |']
    for name, values in provenance['metrics'].items():
        lines.append(f"| {name} | {values['semantic_knn_overlap_at_15']:.4f} | {values['routing_silhouette_of_clip_clusters']:.4f} | {values['mean_expert_probability_std']:.5f} | {values['effective_rank']:.2f} |")
    lines += ['', 'These bounded transfer experiments do not establish full-dataset or full-paper benchmark performance. Population SD across encoder seeds is observed spread, not a confidence interval or significance test. The RoMA success criterion is a label-free reconstruction surrogate for the paper’s correct-answer criterion.', '',
              '[All nine routing views](figures/routing-controls.png), [per-probe fractional scores](scores.csv), [complete results and provenance](comparison.json), [plotted routing weights/node IDs/coordinates](figures/manifold-data.npz), [plot settings](figures/visualization.json). PNG, PDF and SVG exports are beside the main figure.', '']
    (root/'COMPARISON.md').write_text('\n'.join(lines))
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    parser.add_argument('--nodes', type=int, default=1500)
    parser.add_argument('--plot-seed', type=int, default=42)
    args = parser.parse_args()
    if args.nodes < 32:
        parser.error('plot at least 32 nodes')
    records = review_shared(args.root, args.seeds)
    rows = summarize_shared(records, args.seeds)
    provenance = export_figures(args.root, records, args.seeds, rows, args.nodes, args.plot_seed)
    write_dataset_report(args.root, records, args.seeds, rows, provenance)
    print('COMPLETE', args.root/'COMPARISON.md', flush=True)


if __name__ == '__main__':
    main()
