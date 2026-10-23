"""Create a concise Markdown report from a completed experiment artifact."""
import argparse
import json
from pathlib import Path

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    args=parser.parse_args()
    r=json.loads((args.run/'results.json').read_text())
    if r['status']!='completed':
        raise ValueError('run is incomplete')
    cfg=r['args']
    metric=r.get('primary_metric','accuracy')
    label='accuracy' if metric=='accuracy' else 'sampled MRR'
    mean_key,std_key=f'mean_test_{metric}_percent',f'std_test_{metric}_percent'
    lines=['# UniGraph2 bounded reproduction results','',
           'This run validates a paper-guided training and transfer pipeline on real MM-Graph data. It does not reproduce the full paper benchmark.','',
           f"Pre-training: {', '.join(cfg['pretrain'])}. Evaluation: {cfg['evaluate']}, first {r['graphs'][cfg['evaluate']]['nodes']:,} nodes and their induced graph.",'',
           f'| Frozen representation | Test {label} (%) | Standard deviation (percentage points) |',
           '| --- | ---: | ---: |']
    for name,res in r['results'].items():
        lines.append(f"| {name} | {res[mean_key]:.2f} | {res[std_key]:.2f} |")
    history=r['history']
    lines += ['',f"Probe seeds: {cfg['probe_seeds']}; encoder seed: {cfg['seed']}. Standard deviations are population standard deviations across probe initializations, not uncertainty across independently pre-trained encoders.",'',
              f"Pre-training: {cfg['epochs']} passes over {cfg['steps']} sampled subgraphs per domain, at most {cfg['subgraph_nodes']} nodes each. Mean total objective: {history[0]['total']:.4f} in pass 1 to {history[-1]['total']:.4f} in pass {cfg['epochs']}.",'',
              f"Runtime: {r['seconds']/60:.1f} minutes. GPU: {r['environment']['gpu']}. Slurm job: {r['environment']['slurm_job_id']}.",'',
              'Model, optimizer, probe weights, embeddings and sampled node IDs are saved beside this report. Full parameters, environment, input file metadata and source hashes are in `results.json`.','',
              'Differences from the paper: reduced pre-training corpus, a bounded subgraph pool, induced-subgraph SPD targets, one encoder seed, three probe seeds, a restricted downstream graph, and a PyTorch GAT reconstruction. See the reproduction README for all assumptions.','']
    delta=r['results']['UniGraph2'][mean_key]-r['results']['NoPretrain'][mean_key]
    lines += [f'The trained encoder changes test {label} by {delta:+.2f} percentage points relative to the untrained encoder in this limited run. This result alone does not establish the paper\'s generalization claims.','']
    if metric=='mrr':
        lines += [f"Link evaluation uses {cfg['lp_negatives']} fixed sampled non-neighbors per held-out query within the node cohort. Official positive split membership is preserved; reversed training/validation edges and duplicates are excluded. Encoder message passing uses training edges only. A linear edge probe on Hadamard products is selected by validation MRR every {cfg['lp_validation_every']} epochs, then evaluated on test once. These scores use a different candidate population from the full benchmark.", '',
                  f"Protocol and split hash: `{json.dumps(r['evaluation_protocol'],sort_keys=True)}`.", '']
    if cfg.get('router_posttrain', 'none') != 'none':
        lines += [f"Router post-training: {cfg['router_posttrain']}, {cfg['router_epochs']} additional pool passes at lr {cfg['router_lr']}. Only gate parameters change; all experts, GNN layers, mask token and decoders are frozen and dropout is disabled.", '',
                  'RoMA uses frozen unmasked routing targets from low-reconstruction-error pre-training nodes and Gaussian CLIP kNN. This label-free success surrogate differs from the paper\'s correct-answer criterion. No evaluation data enters the adaptation.', '']
        if r.get('pretrain_lineage'):
            lines += [f"Baseline checkpoint: {r['pretrain_lineage']['run']}; SHA256 `{r['pretrain_lineage']['checkpoint_sha256']}`.", '']
    diagnostic_path=args.run/'diagnostics.json'
    if diagnostic_path.exists():
        diag=json.loads(diagnostic_path.read_text())['embedding_diagnostics']
        before,after=diag['untrained'],diag['pretrained']
        lines += [f"Saved-embedding diagnostics (untrained → pre-trained): mean cosine to centroid {before['mean_cosine_to_centroid']:.5f} → {after['mean_cosine_to_centroid']:.5f}; effective rank {before['effective_rank']:.2f} → {after['effective_rank']:.2f}. Cosine near 1 with rank near 1 indicates collapsed, near-constant embeddings.",'']
    (args.run/'REPORT.md').write_text('\n'.join(lines))
    print('\n'.join(lines))

if __name__=='__main__':
    main()
