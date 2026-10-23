"""Validate, visualize and archive the five-dataset shared-attention experiment."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import numpy as np
from all_datasets import DATASETS
from compare_shared import CORE_SOURCES
from plot_roma import metric_of, probe_scores
from plot_shared import (ARCHITECTURE_LABELS, CONDITIONS, export_figures, review_shared,
                         summarize_shared, write_dataset_report)


def collect_shared(root, seeds):
    datasets, all_records, global_settings, global_sources = [], {}, None, None
    ignored = {'evaluate', 'pretrain', 'output', 'seed', 'expert', 'shared_attention', 'router_posttrain', 'resume_pretrain'}
    for domain in DATASETS:
        records = review_shared(root/domain, seeds)
        first = next(iter(records.values()))
        cfg = first['args']
        source = 'sports-copurchase' if domain == 'cloth-copurchase' else 'cloth-copurchase'
        if cfg['evaluate'] != domain or cfg['pretrain'] != [source]:
            raise ValueError(f'incorrect held-out source/target condition: {domain}')
        settings = {key: value for key, value in cfg.items() if key not in ignored}
        sources = {name: first['source_sha256'][name] for name in CORE_SOURCES}
        if global_settings is None:
            global_settings, global_sources = settings, sources
        elif settings != global_settings or sources != global_sources:
            raise ValueError('dataset comparisons use different budgets or core implementations')
        references = []
        for architecture in ARCHITECTURE_LABELS:
            for representation in ['CLIP', 'NoPretrain']:
                means = [float(np.mean(probe_scores(records[(seed, architecture, 'none')], representation))) for seed in seeds]
                references.append({'architecture': architecture, 'representation': representation,
                                   'mean': float(np.mean(means)), 'population_std': float(np.std(means)),
                                   'encoder_seed_means': means})
        rows = summarize_shared(records, seeds)
        architecture_deltas = []
        for reference in ['mlp', 'attention']:
            for mode in ['none', 'roma']:
                candidate = [np.mean(probe_scores(records[(seed, 'shared', mode)])) for seed in seeds]
                comparator = [np.mean(probe_scores(records[(seed, reference, mode)])) for seed in seeds]
                delta = np.asarray(candidate)-comparator
                architecture_deltas.append({'candidate': 'shared', 'reference': reference, 'method': mode,
                                            'mean': float(delta.mean()), 'population_std': float(delta.std()),
                                            'per_seed': delta.tolist()})
        datasets.append({'dataset': domain, 'pretrain': cfg['pretrain'], 'primary_metric': metric_of(first),
                         'evaluation_protocol': first.get('evaluation_protocol'), 'summary': rows,
                         'references': references, 'architecture_deltas': architecture_deltas,
                         'split_sizes': first['results']['UniGraph2']['runs'][0]['split_sizes']})
        all_records[domain] = records
    return datasets, all_records


def performance_figure(datasets, destination, seeds, probe_seeds, configuration):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    cases = [('mlp', 'none', 'MLP', '#65717C', 'o'),
             ('attention', 'none', 'Per-expert attention', '#D58C27', 's'),
             ('shared', 'none', 'Shared attention', '#C85C87', '^'),
             ('mlp', 'roma', 'MLP + RoMA', '#3675B8', 'o'),
             ('attention', 'roma', 'Per-expert + RoMA', '#D58C27', 's'),
             ('shared', 'roma', 'Shared + RoMA', '#C85C87', '^')]
    for ax, dataset in zip(axes.flat, datasets):
        for index, (architecture, mode, label, color, marker) in enumerate(cases):
            row = next(row for row in dataset['summary'] if (row['architecture'], row['method']) == (architecture, mode))
            mean, spread = row['mean_test_score_percent'], row['std_across_encoder_seeds']
            ax.errorbar(mean, index, xerr=spread, fmt=marker, color=color, capsize=3, markersize=6,
                        markerfacecolor='white' if mode == 'none' else color)
            ax.annotate(f'{mean:.2f} ± {spread:.2f}', (mean+spread, index), xytext=(7, 0),
                        textcoords='offset points', va='center', fontsize=8.5)
        ax.set_yticks(range(len(cases)), [case[2] for case in cases], fontsize=9)
        ax.invert_yaxis()
        ax.set_xlim(0, 100)
        ax.set_title(dataset['dataset'], fontsize=12)
        ax.set_xlabel('Accuracy (%)' if dataset['primary_metric'] == 'accuracy' else 'Sampled MRR (%)')
        ax.grid(axis='x', color='#E4E4E4', linewidth=.7)
        ax.spines[['top', 'right']].set_visible(False)
    axes.flat[-1].axis('off')
    axes.flat[-1].text(.02, .95,
        f"Mean over {len(seeds)} encoder seeds\nEach encoder averages {len(probe_seeds)} probe seeds\nError bars: population SD across encoders\n\n{configuration['eval_limit']:,}-node target cohorts\nLP: {configuration['lp_negatives']} fixed negatives/query\nShared attention: {configuration.get('attention_heads', 4)} heads\nRouting top-k: {configuration['top_k']} (0 = dense)\n\nTask-only controls are in the dataset reports.\nExtra attention parameters are not matched away.",
        transform=axes.flat[-1].transAxes, va='top', fontsize=10, linespacing=1.6)
    fig.suptitle('UniGraph2: original MLP, per-expert attention and shared pre-router attention', fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, .95), w_pad=2, h_pad=2)
    fig.savefig(destination/'performance.png', dpi=180)
    fig.savefig(destination/'performance.pdf')
    plt.close(fig)


def format_estimate(row, mean='mean_test_score_percent', spread='std_across_encoder_seeds'):
    return f"{row[mean]:.2f} ± {row[spread]:.2f}"


def write_root_report(root, datasets, all_records, manifest, seeds):
    first = next(iter(all_records[DATASETS[0]].values()))
    cfg = first['args']
    probe_seeds = cfg['probe_seeds']
    aggregate = {'status': 'completed', 'encoder_seeds': seeds, 'probe_seeds': probe_seeds,
                 'completed_runs': len(DATASETS)*len(seeds)*len(CONDITIONS), 'datasets': datasets,
                 'download_manifest': manifest,
                 'scope': 'five CLIP-ready multimodal graphs; bounded cohorts; accuracy and sampled MRR are not pooled',
                 'architecture_sequence': 'node features -> shared multihead self-attention -> router -> selected MLP experts -> GAT',
                 'attention_heads': cfg.get('attention_heads', 4), 'top_k': cfg['top_k']}
    (root/'summary.json').write_text(json.dumps(aggregate, indent=2))
    (root/'download-manifest.json').write_text(json.dumps(manifest, indent=2))
    with (root/'scores.csv').open('w') as handle:
        writer = csv.writer(handle)
        writer.writerow(['dataset', 'primary_metric', 'encoder_seed', 'architecture', 'posttrain', 'representation', 'probe_seed', 'valid_score', 'test_score', 'selected_epoch', 'slurm_job_id'])
        for domain, records in all_records.items():
            for (seed, architecture, mode), report in records.items():
                metric = metric_of(report)
                for representation, result in report['results'].items():
                    for probe in result['runs']:
                        writer.writerow([domain, metric, seed, architecture, mode, representation, probe['seed'], probe[f'valid_{metric}'], probe[f'test_{metric}'], probe['selected_epoch'], report['environment']['slurm_job_id']])
    lines = ['# Shared pre-router attention: five-dataset comparison', '',
             f"All {aggregate['completed_runs']} runs completed. Values are mean ± population standard deviation over {len(seeds)} encoder seeds, each averaging {len(probe_seeds)} probe seeds.", '',
             '| Target | Metric (%) | Original MLP | Old per-expert attention | New shared attention + MLP |', '| --- | --- | ---: | ---: | ---: |']
    for dataset in datasets:
        values = [format_estimate(next(row for row in dataset['summary'] if row['architecture'] == architecture and row['method'] == 'none')) for architecture in ARCHITECTURE_LABELS]
        metric_label = 'accuracy' if dataset['primary_metric'] == 'accuracy' else 'sampled MRR'
        lines.append(f"| [{dataset['dataset']}]({dataset['dataset']}/COMPARISON.md) | {metric_label} | "+' | '.join(values)+' |')
    lines += ['', '| Target | MLP + RoMA | Old attention + RoMA | Shared attention + RoMA |', '| --- | ---: | ---: | ---: |']
    for dataset in datasets:
        values = [format_estimate(next(row for row in dataset['summary'] if row['architecture'] == architecture and row['method'] == 'roma')) for architecture in ARCHITECTURE_LABELS]
        lines.append(f"| {dataset['dataset']} | "+' | '.join(values)+' |')
    lines += ['', '![Architecture performance](performance.png)', '', '## Shared-attention differences', '',
              'These are differences over matched encoder-seed labels, with population SD across seeds. Architectures have different parameter counts and initial states; within each architecture, task-only/RoMA branches share the exact pretrained checkpoint.', '',
              '| Target | Stage | Shared − MLP (points) | Shared − old attention (points) |', '| --- | --- | ---: | ---: |']
    for dataset in datasets:
        for mode in ['none', 'roma']:
            values = [format_estimate(next(row for row in dataset['architecture_deltas'] if row['reference'] == reference and row['method'] == mode), 'mean', 'population_std') for reference in ['mlp', 'attention']]
            lines.append(f"| {dataset['dataset']} | {mode} | "+' | '.join(values)+' |')
    lines += ['', '## RoMA and equal-budget router controls', '',
              '| Target | Architecture | RoMA − own baseline (points) | RoMA − task-only (points) |', '| --- | --- | ---: | ---: |']
    for dataset in datasets:
        for row in dataset['summary']:
            if row['method'] == 'roma':
                a, b = row['paired_delta_vs_baseline'], row['paired_delta_vs_router_task']
                lines.append(f"| {dataset['dataset']} | {ARCHITECTURE_LABELS[row['architecture']]} | {a['mean']:+.3f} ± {a['population_std']:.3f} | {b['mean']:+.3f} ± {b['population_std']:.3f} |")
    lines += ['', '## Frozen references', '', '| Target | CLIP | Untrained MLP | Untrained per-expert attention | Untrained shared attention |', '| --- | ---: | ---: | ---: | ---: |']
    for dataset in datasets:
        cases = [('mlp', 'CLIP'), ('mlp', 'NoPretrain'), ('attention', 'NoPretrain'), ('shared', 'NoPretrain')]
        values = [format_estimate(next(row for row in dataset['references'] if (row['architecture'], row['representation']) == case), 'mean', 'population_std') for case in cases]
        lines.append(f"| {dataset['dataset']} | "+' | '.join(values)+' |')
    lines += ['', '## Architecture and protocol', '',
              'The new optional model applies shared self-attention before routing: node features → learned multihead Q/K/V scaled dot-product attention → output projection + residual + LayerNorm → router → selected MLP experts → existing GAT. The older attention option gives each expert its own attention and remains a separate comparison. Every attention call uses only the current graph/subgraph. Both dense and top-k 2 routing are supported; the routing setting used in this sweep is recorded below.', '',
              f"Shared heads: {cfg.get('attention_heads', 4)}; routing top-k: {cfg['top_k']} (0 means dense); width: {cfg['hidden']}; experts: {cfg['experts']}. Encoder pre-training uses {cfg['epochs']*cfg['steps']} updates per source domain, lr {cfg['lr']}, warmup {cfg['warmup']}, source-only standardization {cfg['standardize']}, and the same seeded subgraph pools. Task-only and RoMA each add {cfg['router_epochs']*cfg['steps']} gate-only updates per source domain at lr {cfg['router_lr']}. Shared attention and every non-router parameter remain frozen during those stages. The extra shared-attention parameters are not matched away, so this fixes hyperparameters and update budgets rather than total parameter counts.", '',
              'The RoMA graph adaptation uses source-only low masked-reconstruction-error anchors instead of correct labeled answers. Shared-attention reference routes are averaged over scored anchor occurrences in their original complete PPR contexts, never by constructing an attention graph from selected anchors. No target graph or target labels enter pre-training or the teacher bank.', '',
              f"The required files for all five CLIP-ready datasets were downloaded or checksum-verified directly in the canonical source directory against [MM-Graph release](https://huggingface.co/datasets/mm-graph-org/mm-graph/tree/{manifest['revision']}) `{manifest['revision']}`. CoDEx-S/M release folders supply relational triples without cached CLIP node features and are recorded as outside this feature/task contract. Only required feature/task files were downloaded; raw images and unused feature families were not copied.", '',
              'Every target is held out from pre-training. Amazon-Cloth is the source for four targets; Amazon-Sports is the source for Amazon-Cloth evaluation. Node classification uses the first 4,096 nodes and intersected official splits. Link prediction also uses a 4,096-node cohort, training-edge-only message passing, official positive membership after deduplication and removal of held-out undirected edges seen in earlier splits, and fixed sampled negatives. At most 4,096 training and 1,024 validation/test positives are selected; each held-out query has 100 distinct non-neighbors, excluding self loops and every known positive. All conditions and seeds share identical candidate arrays and logical split hashes. These sampled-MRR candidate populations differ from the official full-graph benchmark.', '',
              'NC head epochs are selected by validation accuracy; LP heads use validation MRR every five epochs. Test metrics are computed after selection. LP uses a linear probe on Hadamard products of unit-normalized frozen embeddings, one training negative per positive, and average optimistic/pessimistic ranks for ties. MRR and Hits@1/3/10 are preserved in per-run evidence. Accuracy and MRR are not averaged together.', '',
              '## Model parameter counts', '', '| Architecture | Total | Router | Shared heads |', '| --- | ---: | ---: | ---: |']
    for row in datasets[0]['summary']:
        if row['method'] == 'none':
            counts = row.get('parameter_counts') or {}
            lines.append(f"| {ARCHITECTURE_LABELS[row['architecture']]} | {counts.get('total', 'unrecorded')} | {counts.get('router', 'unrecorded')} | {row['attention_heads'] or '—'} |")
    lines += ['', '## Cohorts and jobs', '', '| Target | Pre-training source | Train | Validation | Test | Slurm jobs (encoder seeds in declared order) |', '| --- | --- | ---: | ---: | ---: | --- |']
    for dataset in datasets:
        sizes = dataset['split_sizes']
        lines.append(f"| {dataset['dataset']} | {', '.join(dataset['pretrain'])} | {sizes['train']} | {sizes['valid']} | {sizes['test']} | {', '.join(dataset['summary'][0]['job_ids'])} |")
    lines += ['', 'Each linked dataset report includes all nine conditions, actual routing Figure-3-style PNG/PDF/SVG exports, fixed semantic colors, original-space neighborhood diagnostics, per-probe scores, source hashes and input metadata. No label-optimized oracle or visually selected seed is used. These are bounded transfer experiments; neither the observed seed spread nor apparent UMAP separation establishes statistical significance or full-benchmark superiority.', '',
              '[Per-probe fractional scores](scores.csv), [structured summary](summary.json), [download checksums](download-manifest.json), and [performance PDF](performance.pdf).', '']
    (root/'REPORT.md').write_text('\n'.join(lines))
    return aggregate


def archive_report(root, destination):
    destination.mkdir(parents=True, exist_ok=True)
    for name in ['REPORT.md', 'summary.json', 'scores.csv', 'download-manifest.json', 'performance.png', 'performance.pdf']:
        shutil.copyfile(root/name, destination/name)
    for domain in DATASETS:
        target = destination/domain
        target.mkdir(exist_ok=True)
        for name in ['COMPARISON.md', 'comparison.json', 'scores.csv']:
            shutil.copyfile(root/domain/name, target/name)
        shutil.copytree(root/domain/'figures', target/'figures', dirs_exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--download-manifest', type=Path, required=True)
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    parser.add_argument('--nodes', type=int, default=1500)
    parser.add_argument('--plot-seed', type=int, default=42)
    parser.add_argument('--archive', type=Path)
    args = parser.parse_args()
    if args.nodes < 32:
        parser.error('plot at least 32 nodes')
    manifest = json.loads(args.download_manifest.read_text())
    if manifest.get('status') != 'completed' or set(manifest['domains']) != set(DATASETS):
        raise ValueError('need the completed checksum-verified manifest for all five datasets')
    datasets, all_records = collect_shared(args.root, args.seeds)
    for dataset in datasets:
        domain = dataset['dataset']
        records = all_records[domain]
        provenance = export_figures(args.root/domain, records, args.seeds, dataset['summary'], args.nodes, args.plot_seed)
        write_dataset_report(args.root/domain, records, args.seeds, dataset['summary'], provenance)
    cfg = next(iter(all_records[DATASETS[0]].values()))['args']
    performance_figure(datasets, args.root, args.seeds, cfg['probe_seeds'], cfg)
    write_root_report(args.root, datasets, all_records, manifest, args.seeds)
    if args.archive:
        archive_report(args.root, args.archive)
    print('COMPLETE', args.root/'REPORT.md', flush=True)


if __name__ == '__main__':
    main()
