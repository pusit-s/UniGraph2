"""Run one held-out dataset/seed with the three attention/MLP architectures."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from all_datasets import DATASETS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--index', type=int, required=True, help='dataset index times three plus encoder seed')
    args, overrides = parser.parse_known_args()
    if not 0 <= args.index < len(DATASETS)*3:
        parser.error('index must be 0..14')
    if any(argument.split('=')[0] in {'--evaluate', '--pretrain', '--output', '--seed', '--index'} for argument in overrides):
        parser.error('dataset, source and seed are controlled by this sweep')
    target, seed = DATASETS[args.index//3], args.index % 3
    source = 'sports-copurchase' if target == 'cloth-copurchase' else 'cloth-copurchase'
    output = args.output/target/f'seed-{seed}'
    subprocess.run([sys.executable, '-u', 'compare_shared.py', '--output', str(output), '--seed', str(seed),
                    '--evaluate', target, '--pretrain', source, *overrides], check=True)
    (output/'dataset_manifest.json').write_text(json.dumps(
        {'status': 'completed', 'target': target, 'source': source, 'seed': seed,
         'architectures': ['mlp', 'attention', 'shared'], 'slurm_job_id': os.environ.get('SLURM_JOB_ID')}, indent=2))


if __name__ == '__main__':
    main()
