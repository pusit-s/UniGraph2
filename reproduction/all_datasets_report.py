"""Build and archive the complete five-dataset comparison and Figure 3 exports."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
from all_datasets import DATASETS
from plot_roma import CONDITIONS, metric_of, probe_scores, review_runs, summarize


def collect(root, seeds):
    datasets=[]
    for domain in DATASETS:
        records=review_runs(root/domain,seeds)
        first=next(iter(records.values()))
        if first['args']['evaluate']!=domain:
            raise ValueError('dataset folder does not match evaluation target')
        summary=summarize(records,seeds)
        references=[]
        for expert in ['mlp','attention']:
            for representation in ['CLIP','NoPretrain']:
                means=[float(np.mean(probe_scores(records[(seed,expert,'none')],representation))) for seed in seeds]
                references.append({'expert':expert,'representation':representation,'mean':float(np.mean(means)),
                                   'population_std':float(np.std(means)),'encoder_seed_means':means})
        for row in summary:
            baseline=next(candidate for candidate in summary if candidate['expert']==row['expert'] and candidate['method']=='none')
            delta=np.asarray(row['seed_means'])-baseline['seed_means']
            row['paired_delta_vs_baseline']={'mean':float(delta.mean()),'population_std':float(delta.std()),'per_seed':delta.tolist()}
        datasets.append({'dataset':domain,'pretrain':first['args']['pretrain'],'primary_metric':metric_of(first),
                         'evaluation_protocol':first.get('evaluation_protocol'),'summary':summary,'references':references})
    return datasets


def performance_figure(datasets, destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(16,8))
    methods=[('mlp','none','MLP baseline','#65717C','o'),('attention','none','Attention baseline','#8E98A1','s'),
             ('mlp','roma','MLP + RoMA','#3675B8','o'),('attention','roma','Attention + RoMA','#D58C27','s')]
    for ax,dataset in zip(axes.flat,datasets):
        for index,(expert,method,label,color,marker) in enumerate(methods):
            row=next(row for row in dataset['summary'] if (row['expert'],row['method'])==(expert,method))
            score,std=row['mean_test_score_percent'],row['std_across_encoder_seeds']
            ax.errorbar(score,index,xerr=std,fmt=marker,color=color,capsize=3,markersize=6)
            ax.annotate(f'{score:.2f} ± {std:.2f}',(score+std,index),xytext=(7,0),textcoords='offset points',va='center',fontsize=9)
        ax.set_yticks(range(4),[method[2] for method in methods],fontsize=9)
        ax.invert_yaxis()
        ax.set_xlim(0,100)
        ax.set_title(dataset['dataset'],fontsize=12)
        ax.set_xlabel('Accuracy (%)' if dataset['primary_metric']=='accuracy' else 'Sampled MRR (%)')
        ax.grid(axis='x',color='#E4E4E4',linewidth=.7)
        ax.spines[['top','right']].set_visible(False)
    axes.flat[-1].axis('off')
    axes.flat[-1].text(.02,.8,'Each point: mean over 3 encoder seeds\nEach encoder: mean over 3 probe seeds\nError bars: population SD across encoders\n\n4,096-node target cohorts\nLP: 100 fixed non-neighbors/query\n\nTask-only controls and Figure 3 plots\nare in each dataset report.',transform=axes.flat[-1].transAxes,va='top',fontsize=11,linespacing=1.6)
    fig.suptitle('UniGraph2 and RoMA across all five CLIP-ready MM-Graph datasets',fontsize=16)
    fig.tight_layout(rect=(0,0,1,.95),w_pad=2.0,h_pad=2.0)
    fig.savefig(destination/'performance.png',dpi=180)
    fig.savefig(destination/'performance.pdf')
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    parser.add_argument('--download-manifest',type=Path,required=True)
    parser.add_argument('--seeds',nargs='+',type=int,default=[0,1,2])
    parser.add_argument('--archive',type=Path)
    args=parser.parse_args()
    manifest=json.loads(args.download_manifest.read_text())
    if manifest.get('status')!='completed' or set(manifest['domains'])!=set(DATASETS):
        raise ValueError('need a completed checksum-verified download manifest for all five datasets')
    datasets=collect(args.root,args.seeds)
    for domain in DATASETS:
        subprocess.run([sys.executable,'-u','plot_roma.py',str(args.root/domain),'--seeds',*map(str,args.seeds)],check=True)
    performance_figure(datasets,args.root)
    aggregate={'status':'completed','encoder_seeds':args.seeds,'probe_seeds':[0,1,2],
               'completed_runs':len(DATASETS)*len(args.seeds)*len(CONDITIONS),'datasets':datasets,
               'download_manifest':manifest,
               'scope':'five CLIP-ready datasets; bounded target cohorts; no aggregate across accuracy and MRR'}
    (args.root/'summary.json').write_text(json.dumps(aggregate,indent=2))
    shutil.copyfile(args.download_manifest,args.root/'download-manifest.json')
    with (args.root/'scores.csv').open('w') as handle:
        writer=csv.writer(handle)
        writer.writerow(['dataset','primary_metric','encoder_seed','expert','posttrain','representation','probe_seed','valid_score','test_score','selected_epoch','slurm_job_id'])
        for dataset in datasets:
            domain,metric=dataset['dataset'],dataset['primary_metric']
            records=review_runs(args.root/domain,args.seeds)
            for (seed,expert,method),report in records.items():
                for representation,result in report['results'].items():
                    for probe in result['runs']:
                        writer.writerow([domain,metric,seed,expert,method,representation,probe['seed'],probe[f'valid_{metric}'],probe[f'test_{metric}'],probe['selected_epoch'],report['environment']['slurm_job_id']])
    lines=['# UniGraph2 / attention / RoMA: all five CLIP-ready datasets','',
           f"All {aggregate['completed_runs']} runs completed. Values are mean ± population standard deviation across three encoder seeds, each averaged over three probe seeds.",'',
           '| Target | Pre-training source | Metric (%) | MLP baseline | Attention baseline | MLP + RoMA | Attention + RoMA |',
           '| --- | --- | --- | ---: | ---: | ---: | ---: |']
    for dataset in datasets:
        values=[]
        for expert,method in [('mlp','none'),('attention','none'),('mlp','roma'),('attention','roma')]:
            row=next(row for row in dataset['summary'] if (row['expert'],row['method'])==(expert,method))
            values.append(f"{row['mean_test_score_percent']:.2f} ± {row['std_across_encoder_seeds']:.2f}")
        metric='accuracy' if dataset['primary_metric']=='accuracy' else 'sampled MRR'
        lines.append(f"| [{dataset['dataset']}]({dataset['dataset']}/COMPARISON.md) | {', '.join(dataset['pretrain'])} | {metric} | "+' | '.join(values)+' |')
    lines+=['','![Performance](performance.png)','','## Paired RoMA effects','',
            'Deltas below use matched encoder seeds. Task-only controls have the same 96 additional gate updates as RoMA. No hypothesis test or confidence interval is claimed.','',
            '| Target | Expert | RoMA − baseline (points) | RoMA − task-only (points) |',
            '| --- | --- | ---: | ---: |']
    for dataset in datasets:
        for row in dataset['summary']:
            if row['method']=='roma':
                a,b=row['paired_delta_vs_baseline'],row['paired_delta_vs_router_task']
                lines.append(f"| {dataset['dataset']} | {row['expert']} | {a['mean']:+.3f} ± {a['population_std']:.3f} | {b['mean']:+.3f} ± {b['population_std']:.3f} |")
    lines+=['','## Frozen references','',
            '| Target | CLIP | Untrained MLP encoder | Untrained attention encoder |','| --- | ---: | ---: | ---: |']
    for dataset in datasets:
        values=[]
        for expert,representation in [('mlp','CLIP'),('mlp','NoPretrain'),('attention','NoPretrain')]:
            row=next(row for row in dataset['references'] if (row['expert'],row['representation'])==(expert,representation))
            values.append(f"{row['mean']:.2f} ± {row['population_std']:.2f}")
        lines.append(f"| {dataset['dataset']} | "+' | '.join(values)+' |')
    lines+=['','## Coverage and evaluation protocol','',
            f"Required files were downloaded directly into the canonical data directory or verified in place against release `{manifest['revision']}` from [MM-Graph](https://huggingface.co/datasets/mm-graph-org/mm-graph). Only cached CLIP features and task files were needed; raw images and other feature families were not downloaded. The publisher also lists mm-codex-s/m, but their folders do not contain cached CLIP node features and use relational knowledge-graph triples. They are outside this runnable feature/task contract.",'',
            'Every target is excluded from pre-training. Amazon-Cloth is the source for four targets; Amazon-Sports is the source when evaluating Amazon-Cloth. Pre-training uses 160 updates on 32 PPR subgraphs, width 1024, 8 dense experts, lr 1e-4, warmup 0.1, and source-only standardization. RoMA and task-only controls add 96 gate-only updates at lr 7e-5; λ=0.8, k=5, σ=0.4, success fraction 0.5. These settings were fixed before this sweep. The label-free RoMA success criterion is masked reconstruction quality, rather than the paper\'s correct-answer criterion.','',
            'Node-classification targets use the first 4,096 nodes and intersected official train/validation/test splits. Link targets use the same node limit and training-edge-only message passing. Held-out undirected positives already visible in earlier splits, including reversals, are removed. At most 4,096 training, 1,024 validation, and 1,024 test positive queries are selected with fixed seed 1729. Each held-out query is ranked against 100 fixed distinct non-neighbors from that cohort, excluding self loops and all known positives. Training uses one fixed negative per positive. All encoder/probe seeds and methods share exactly the same cohorts/candidates.','',
            'The frozen link probe is linear on Hadamard products of unit-normalized embeddings. Epoch selection uses validation MRR every five epochs; test MRR and Hits@1/3/10 are computed after selection. Ties use the average optimistic/pessimistic rank, as in MM-Graph\'s evaluator. These sampled-MRR scores have a different candidate population from the official full-graph benchmark. Accuracy and MRR are not averaged together.','',
            'Each dataset report includes Figure-3-style UMAP panels using 1,500 fixed nodes, real routing weights and fixed CLIP clusters, plus original-space diagnostics. The fourth panel shows attention instead of a label-optimized oracle. UMAP appearance does not establish alignment or task improvement.','',
            '## Cohort sizes and jobs','',
            '| Target | Train positives / nodes | Validation positives / nodes | Test positives / nodes | Slurm jobs (encoder seeds 0/1/2) |',
            '| --- | ---: | ---: | ---: | --- |']
    for dataset in datasets:
        report=json.loads((args.root/dataset['dataset']/'seed-0'/'mlp-none'/'results.json').read_text())
        sizes=report['results']['UniGraph2']['runs'][0]['split_sizes']
        jobs=dataset['summary'][0]['job_ids']
        lines.append(f"| {dataset['dataset']} | {sizes['train']} | {sizes['valid']} | {sizes['test']} | {', '.join(jobs)} |")
    lines+=['','[Per-probe scores](scores.csv) use fractional units. [Structured summary](summary.json) includes per-seed means, paired deltas, exact protocols, checksums and coverage exclusions. Each dataset folder retains all three representation results, source hashes, data metadata and plot provenance. These bounded results do not establish full-dataset or full-paper benchmark performance.','']
    (args.root/'REPORT.md').write_text('\n'.join(lines))
    if args.archive:
        args.archive.mkdir(parents=True,exist_ok=True)
        for name in ['REPORT.md','summary.json','scores.csv','download-manifest.json','performance.png','performance.pdf']:
            shutil.copyfile(args.root/name,args.archive/name)
        for domain in DATASETS:
            destination=args.archive/domain
            destination.mkdir(exist_ok=True)
            for name in ['COMPARISON.md','comparison.json','scores.csv']:
                shutil.copyfile(args.root/domain/name,destination/name)
            shutil.copytree(args.root/domain/'figures',destination/'figures',dirs_exist_ok=True)
    print('COMPLETE',args.root/'REPORT.md',flush=True)


if __name__=='__main__':
    main()
