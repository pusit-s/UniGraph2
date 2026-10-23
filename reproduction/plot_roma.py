"""Validate a complete paired comparison and export Figure-3-style UMAP figures.

Run on a Slurm compute node. No labels are used to select or color plotted nodes.
The fourth paper-style panel shows attention routing instead of a label oracle.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np


CONDITIONS = [('mlp', 'none', 'UniGraph2 / MLP'),
              ('attention', 'none', 'Attention experts'),
              ('mlp', 'task', 'MLP + router task'),
              ('mlp', 'roma', 'MLP + RoMA'),
              ('attention', 'task', 'Attention + router task'),
              ('attention', 'roma', 'Attention + RoMA')]
COLORS = ['#3675B8', '#D58C27', '#858B35', '#C85C87']
MARKERS = ['o', 's', '^', 'D']


def review_runs(root, seeds):
    """Refuse partial, unpaired, mixed-configuration or stale-source comparisons."""
    records, reference = {}, None
    allowed = {'output', 'seed', 'expert', 'router_posttrain', 'resume_pretrain'}
    for seed in seeds:
        baseline_hashes = {}
        for expert, method, _ in CONDITIONS:
            path = root/f'seed-{seed}'/f'{expert}-{method}'
            report = json.loads((path/'results.json').read_text())
            if report.get('status') != 'completed' or set(report['results']) != {'CLIP', 'NoPretrain', 'UniGraph2'}:
                raise ValueError(f'incomplete run: {path}')
            cfg = report['args']
            if (cfg['seed'], cfg['expert'], cfg['router_posttrain']) != (seed, expert, method):
                raise ValueError(f'condition mismatch: {path}')
            if cfg['evaluate'] in cfg['pretrain']:
                raise ValueError('evaluation-domain leakage')
            comparable = {k: v for k, v in cfg.items() if k not in allowed}
            if reference is None:
                reference = (comparable, report['graphs'], report['data_files'], report['source_sha256'])
            if (comparable, report['graphs'], report['data_files']) != reference[:3]:
                raise ValueError(f'unmatched data or training configuration: {path}')
            for name in ['model.py', 'graph_data.py', 'run.py', 'roma.py']:
                if report['source_sha256'][name] != reference[3][name]:
                    raise ValueError(f'mixed implementation: {path}/{name}')
            for representation, result in report['results'].items():
                probes = result['runs']
                if [p['seed'] for p in probes] != cfg['probe_seeds']:
                    raise ValueError(f'missing probe seeds: {path}/{representation}')
                scores = [p['test_accuracy']*100 for p in probes]
                if not all(0 <= score <= 100 for score in scores):
                    raise ValueError('invalid accuracy')
                if not np.isclose(np.mean(scores), result['mean_test_accuracy_percent']):
                    raise ValueError('reported mean disagrees with probe rows')
            if method == 'none':
                baseline_hashes[expert] = hashlib.sha256((path/'checkpoint.pt').read_bytes()).hexdigest()
            else:
                lineage = report['pretrain_lineage']
                expected = root/f'seed-{seed}'/f'{expert}-none'
                if Path(lineage['run']).resolve() != expected.resolve() or lineage['checkpoint_sha256'] != baseline_hashes[expert]:
                    raise ValueError(f'unpaired checkpoint: {path}')
            records[(seed, expert, method)] = report
        for expert in ['mlp', 'attention']:
            baseline = records[(seed, expert, 'none')]
            for method in ['task', 'roma']:
                candidate = records[(seed, expert, method)]
                for representation in ['CLIP', 'NoPretrain']:
                    if candidate['results'][representation] != baseline['results'][representation]:
                        raise ValueError(f'unpaired reference probes: seed {seed}, {expert}, {method}')
    return records


def summarize(records, seeds):
    rows = []
    for expert, method, label in CONDITIONS:
        seed_means = [np.mean([r['test_accuracy']*100 for r in records[(seed, expert, method)]['results']['UniGraph2']['runs']]) for seed in seeds]
        row = {'expert': expert, 'method': method, 'label': label,
               'mean_test_accuracy_percent': float(np.mean(seed_means)),
               'std_across_encoder_seeds': float(np.std(seed_means)),
               'seed_means': [float(v) for v in seed_means],
               'job_ids': [records[(seed, expert, method)]['environment']['slurm_job_id'] for seed in seeds]}
        if method == 'roma':
            controls = [records[(seed, expert, 'task')]['results']['UniGraph2']['mean_test_accuracy_percent'] for seed in seeds]
            deltas = np.asarray(seed_means)-controls
            row['paired_delta_vs_router_task'] = {'mean': float(deltas.mean()), 'population_std': float(deltas.std()), 'per_seed': deltas.tolist()}
        rows.append(row)
    return rows


def neighborhood_metrics(features, semantic_neighbors, groups):
    from sklearn.neighbors import NearestNeighbors
    from sklearn.metrics import silhouette_score
    indices = NearestNeighbors(n_neighbors=16).fit(features).kneighbors(features, return_distance=False)
    neighbors = np.asarray([[j for j in row if j != i][:15] for i, row in enumerate(indices)])
    overlap = np.mean([len(set(a) & set(b))/15 for a, b in zip(neighbors, semantic_neighbors)])
    centered = features-features.mean(0)
    power = np.linalg.svd(centered, compute_uv=False)**2
    probability = power/max(power.sum(), 1e-30)
    return {'semantic_knn_overlap_at_15': float(overlap),
            'routing_silhouette_of_clip_clusters': float(silhouette_score(features, groups)),
            'mean_expert_probability_std': float(features.std(0).mean()),
            'effective_rank': float(np.exp(-np.sum(probability*np.log(np.maximum(probability, 1e-30))))) }


def export_figures(root, records, seeds, rows, n_nodes, plot_seed):
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

    destination = root/'figures'
    destination.mkdir(exist_ok=True)
    # Use the first declared encoder seed, never choose the most attractive run.
    seed = seeds[0]
    configuration = records[(seed, 'mlp', 'none')]['args']
    evaluation = configuration['evaluate']
    evaluation_nodes = records[(seed, 'mlp', 'none')]['graphs'][evaluation]['nodes']
    artifacts = {}
    for expert, method, _ in CONDITIONS:
        artifacts[f'{expert}-{method}'] = torch.load(root/f'seed-{seed}'/f'{expert}-{method}'/'embeddings.pt', map_location='cpu', weights_only=False)
    base = artifacts['mlp-none']
    for artifact in artifacts.values():
        torch.testing.assert_close(artifact['clip'], base['clip'], rtol=0, atol=0)
        torch.testing.assert_close(artifact['node_ids'], base['node_ids'], rtol=0, atol=0)
    selected = np.sort(np.random.default_rng(plot_seed).choice(len(base['clip']), min(n_nodes, len(base['clip'])), replace=False))
    clip = base['clip'][selected].numpy()
    clip = clip/np.linalg.norm(clip, axis=1, keepdims=True).clip(1e-20)
    groups = KMeans(n_clusters=4, n_init=10, random_state=plot_seed).fit_predict(clip)
    # Stable naming by smallest sampled node ID, rather than arbitrary k-means IDs.
    order = sorted(range(4), key=lambda group: selected[groups == group].min())
    groups = np.asarray([order.index(group) for group in groups])
    features = {'CLIP': clip, **{name: artifact['routing'][selected].numpy() for name, artifact in artifacts.items()}}
    config = {'n_neighbors': 15, 'min_dist': .1, 'metric': 'euclidean', 'random_state': plot_seed, 'n_jobs': 1, 'n_components': 2}
    coordinates = {name: umap.UMAP(**config).fit_transform(x) for name, x in features.items()}
    nearest = NearestNeighbors(n_neighbors=16).fit(clip).kneighbors(clip, return_distance=False)
    semantic_neighbors = np.asarray([[j for j in row if j != i][:15] for i, row in enumerate(nearest)])
    metrics = {name: neighborhood_metrics(x, semantic_neighbors, groups) for name, x in features.items() if name != 'CLIP'}
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.labelcolor': '#262626', 'text.color': '#262626',
                         'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    legend = [Line2D([], [], marker=MARKERS[i], color='none', markerfacecolor=COLORS[i], markeredgecolor='none',
                     markersize=6, label=f'CLIP cluster {i+1}') for i in range(4)]
    panels = [('CLIP', '(a) CLIP semantic embedding'), ('mlp-none', '(b) MLP routing / baseline'),
              ('mlp-roma', '(c) MLP routing / RoMA'), ('attention-none', '(d) Attention routing / baseline')]
    fig = plt.figure(figsize=(15.6, 4.8), facecolor='white')
    grid = fig.add_gridspec(2, 8, width_ratios=[1, .15]*4, height_ratios=[.16, 1],
                           left=.025, right=.985, bottom=.24, top=.80, hspace=.03, wspace=.13)
    for column, (name, title) in enumerate(panels):
        ax = fig.add_subplot(grid[1, 2*column])
        top = fig.add_subplot(grid[0, 2*column], sharex=ax)
        right = fig.add_subplot(grid[1, 2*column+1], sharey=ax)
        xy = coordinates[name]
        for group in range(4):
            points = xy[groups == group]
            ax.scatter(*points.T, s=7, c=COLORS[group], marker=MARKERS[group], alpha=.68, linewidths=0, rasterized=True)
            for axis, values, vertical in [(top, points[:, 0], False), (right, points[:, 1], True)]:
                if len(values) > 2 and values.std() > 1e-10:
                    ticks = np.linspace(values.min()-values.std(), values.max()+values.std(), 160)
                    density = gaussian_kde(values)(ticks)
                    axis.plot(density if vertical else ticks, ticks if vertical else density, color=COLORS[group], lw=.9)
        ax.set_xlabel(title, labelpad=9)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color('#777777')
            spine.set_linewidth(.65)
        top.axis('off')
        right.axis('off')
    fig.suptitle('Routing manifolds in UniGraph2', y=.98, fontsize=16)
    fig.legend(handles=legend, loc='upper center', bbox_to_anchor=(.5, .925), ncol=4, frameon=False)
    fig.text(.5, .105, f"{evaluation}: {len(selected):,} fixed nodes from the {evaluation_nodes:,}-node evaluation subset; encoder seed {seed}.", ha='center')
    fig.text(.5, .055, 'Colors are fixed unsupervised CLIP clusters. Independent UMAP axes; visual separation alone does not establish alignment.', ha='center', fontsize=9)
    fig.text(.5, .015, 'Adapted from RoMA Fig. 3: attention replaces the label-optimized oracle. No evaluation labels are used in this figure.', ha='center', fontsize=9)
    for extension in ['png', 'pdf', 'svg']:
        fig.savefig(destination/f'figure3-routing.{extension}', dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for ax, (name, title) in zip(axes.flat, [(f'{e}-{m}', f'{e.upper()} / {m}') for e in ['mlp', 'attention'] for m in ['none', 'task', 'roma']]):
        for group in range(4):
            points = coordinates[name][groups == group]
            ax.scatter(*points.T, s=8, c=COLORS[group], marker=MARKERS[group], alpha=.65, linewidths=0, rasterized=True)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(f"CLIP kNN overlap@15: {metrics[name]['semantic_knn_overlap_at_15']:.3f}")
    fig.legend(handles=legend, loc='upper center', ncol=4, frameon=False)
    fig.suptitle('Routing conditions on the same sampled nodes', y=.995)
    fig.text(.5, .02, 'Independent UMAP layouts. kNN overlap is measured in the original feature and routing spaces.', ha='center')
    fig.subplots_adjust(top=.88, bottom=.10, hspace=.30)
    fig.savefig(destination/'routing-controls.png', dpi=180)
    fig.savefig(destination/'routing-controls.pdf')
    plt.close(fig)

    labels = [row['label'] for row in rows]
    means = [row['mean_test_accuracy_percent'] for row in rows]
    stds = [row['std_across_encoder_seeds'] for row in rows]
    fig, ax = plt.subplots(figsize=(11, 5))
    y = np.arange(len(rows))
    ax.barh(y, means, xerr=stds, capsize=4, color=['#87909A', '#87909A', '#9BBADA', '#3675B8', '#DDBA83', '#D58C27'])
    for i, (mean, std) in enumerate(zip(means, stds)):
        ax.text(mean+std+1, i, f'{mean:.2f} ± {std:.2f}', va='center', fontsize=10)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel('Frozen linear-probe test accuracy (%)')
    ax.set_title('UniGraph2 transfer: MLP, attention and RoMA')
    ax.grid(axis='x', color='#E5E5E5', linewidth=.7)
    ax.set_axisbelow(True)
    updates = configuration['router_epochs'] * configuration['steps'] * len(configuration['pretrain'])
    fig.text(.5, .015, f"Mean ± population SD over {len(seeds)} encoder seeds; each averages {len(configuration['probe_seeds'])} probe seeds. Task and RoMA add {updates} router updates.", ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, 1))
    fig.savefig(destination/'performance.png', dpi=180)
    fig.savefig(destination/'performance.pdf')
    plt.close(fig)
    np.savez_compressed(destination/'manifold-data.npz', node_ids=selected, clip_clusters=groups,
                        **{f'features_{name}': x for name, x in features.items()},
                        **{f'umap_{name}': x for name, x in coordinates.items()})
    provenance = {'encoder_seed': seed, 'sampling_seed': plot_seed, 'nodes': len(selected),
                  'cluster_method': 'KMeans on unit-normalized raw CLIP, 4 clusters, n_init=10; no labels',
                  'cluster_sizes': np.bincount(groups).tolist(), 'umap': config, 'metrics': metrics,
                  'versions': {'numpy': np.__version__, 'torch': torch.__version__, 'sklearn': sklearn.__version__,
                               'matplotlib': matplotlib.__version__, 'umap-learn': version('umap-learn')},
                  'note': 'Each panel is fit independently. Original-space metrics, not apparent UMAP clustering, assess alignment.'}
    (destination/'visualization.json').write_text(json.dumps(provenance, indent=2))
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('comparison', type=Path)
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    parser.add_argument('--nodes', type=int, default=1500)
    parser.add_argument('--plot-seed', type=int, default=42)
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or args.nodes < 32:
        parser.error('use distinct seeds and at least 32 plotted nodes')
    records = review_runs(args.comparison, args.seeds)
    rows = summarize(records, args.seeds)
    provenance = export_figures(args.comparison, records, args.seeds, rows, args.nodes, args.plot_seed)
    cfg = next(iter(records.values()))['args']
    evidence = {'status': 'completed', 'summary': rows, 'encoder_seeds': args.seeds,
                'probe_seeds': cfg['probe_seeds'], 'visualization': provenance,
                'scope': 'bounded cloth-copurchase to ele-fashion transfer, not a full benchmark',
                'runs': [{'encoder_seed': seed, 'expert': expert, 'method': method, 'report': report}
                         for (seed, expert, method), report in records.items()]}
    (args.comparison/'comparison.json').write_text(json.dumps(evidence, indent=2))
    with (args.comparison/'scores.csv').open('w') as handle:
        writer = csv.writer(handle)
        writer.writerow(['encoder_seed', 'expert', 'posttrain', 'representation', 'probe_seed', 'valid_accuracy', 'test_accuracy', 'selected_epoch', 'slurm_job_id'])
        for (seed, expert, method), report in records.items():
            for representation, result in report['results'].items():
                for probe in result['runs']:
                    writer.writerow([seed, expert, method, representation, probe['seed'], probe['valid_accuracy'], probe['test_accuracy'], probe['selected_epoch'], report['environment']['slurm_job_id']])
    lines = ['# UniGraph2, attention and RoMA comparison', '',
             'Completed real-data transfer experiment. Mean ± population standard deviation across encoder seeds, each averaged over probe seeds.', '',
             '| Condition | Test accuracy (%) | Encoder-seed means (%) | Slurm jobs |', '| --- | ---: | --- | --- |']
    for row in rows:
        values = ', '.join(f'{v:.2f}' for v in row['seed_means'])
        lines.append(f"| {row['label']} | {row['mean_test_accuracy_percent']:.2f} ± {row['std_across_encoder_seeds']:.2f} | {values} | {', '.join(row['job_ids'])} |")
    lines += ['', '![Performance](figures/performance.png)', '', '![Figure 3 adaptation](figures/figure3-routing.png)', '',
              'The fourth panel compares the earlier attention expert method. No label-optimized oracle is constructed. Four semantic groups are inferred once from raw CLIP features, and every panel uses the same nodes and group assignments. UMAP layouts are independently fit with fixed settings; their apparent geometry is not a quantitative alignment test.', '',
              '## Adaptation and controls', '',
              '[RoMA](https://arxiv.org/html/2511.07419v2) fine-tunes routers against successful semantic neighbors with experts frozen (Eqs. 5–10). This graph adaptation uses normalized frozen CLIP features, Gaussian kNN weights, and fixed unmasked teacher routes. Success means the lowest half of mean masked reconstruction errors within each pre-training domain, rather than correct labeled answers. This surrogate is an experimental design choice.', '',
              f"All conditions use {cfg['epochs']} passes × {cfg['steps']} Amazon-Cloth subgraphs, lr {cfg['lr']}, warmup {cfg['warmup']}, standardization {cfg['standardize']}, width {cfg['hidden']}, {cfg['experts']} experts, and top-k {cfg['top_k']} (zero means dense). Evaluation uses the first {cfg['eval_limit']} Ele-Fashion nodes and original intersected splits. Encoder seeds: {args.seeds}; probe seeds: {cfg['probe_seeds']}.", '',
              f"Task-only and RoMA each add {cfg['router_epochs']} × {cfg['steps']} gate-only updates at lr {cfg['router_lr']}, branching from the identical pretrained checkpoint within each expert/seed. Frozen modules run with dropout disabled. RoMA uses λ={cfg['roma_lambda']}, k={cfg['roma_k']}, σ={cfg['roma_sigma']}; λ warms up with the learning rate. Defaults were fixed before examining results; no test-based selection was performed.", '',
              '## Paired effect of manifold regularization', '']
    for row in rows:
        if 'paired_delta_vs_router_task' in row:
            delta = row['paired_delta_vs_router_task']
            lines.append(f"- {row['expert'].upper()}: RoMA − task-only = {delta['mean']:+.3f} ± {delta['population_std']:.3f} percentage points across matched encoder seeds.")
    lines += ['', '## Original-space routing diagnostics', '',
              f"These diagnostics use the {provenance['nodes']} plotted nodes from encoder seed {provenance['encoder_seed']}. kNN overlap@15 compares raw-CLIP semantic neighbors with routing neighbors before UMAP. Chance is approximately 15/(N−1); higher overlap alone does not prove better task performance.", '',
              '| Routing | CLIP kNN overlap@15 | CLIP-cluster silhouette | Mean expert probability SD | Effective rank |', '| --- | ---: | ---: | ---: | ---: |']
    for name, metric in provenance['metrics'].items():
        lines.append(f"| {name} | {metric['semantic_knn_overlap_at_15']:.4f} | {metric['routing_silhouette_of_clip_clusters']:.4f} | {metric['mean_expert_probability_std']:.5f} | {metric['effective_rank']:.2f} |")
    lines += ['', '## Limits and evidence', '',
              'This is a bounded transfer experiment using one source domain and a target subset. It does not reproduce either paper\'s full benchmark. Three encoder seeds measure observed spread, not a confidence interval or a significance test. Additional router optimization is controlled by the task-only branches; baseline and attention results have fewer total optimizer steps.', '',
              'Per-probe validation/test scores, selected epochs and job IDs: [scores.csv](scores.csv). Complete results and provenance: [comparison.json](comparison.json). Plotted coordinates, raw routing weights and node IDs: [manifold-data.npz](figures/manifold-data.npz). Plot settings and original-space metrics: [visualization.json](figures/visualization.json). All routing conditions: [routing-controls.png](figures/routing-controls.png). Vector exports are beside each PNG.', '']
    (args.comparison/'COMPARISON.md').write_text('\n'.join(lines))
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
