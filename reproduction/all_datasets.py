"""Run all five CLIP-ready MM-Graph datasets, keeping every target held out."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


DATASETS=['ele-fashion','books-nc','sports-copurchase','cloth-copurchase','books-lp']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--index',type=int,required=True,help='dataset index times 3 plus encoder seed')
    args,overrides=parser.parse_known_args()
    if not 0<=args.index<len(DATASETS)*3:
        parser.error('index must be 0..14')
    target=DATASETS[args.index//3]
    seed=args.index%3
    source='sports-copurchase' if target=='cloth-copurchase' else 'cloth-copurchase'
    output=args.output/target/f'seed-{seed}'
    subprocess.run([sys.executable,'-u','compare_roma.py','--output',str(output),'--seed',str(seed),
                    '--evaluate',target,'--pretrain',source,*overrides],check=True)
    (output/'dataset_manifest.json').write_text(json.dumps(
        {'status':'completed','target':target,'source':source,'seed':seed,
         'slurm_job_id':os.environ.get('SLURM_JOB_ID')},indent=2))


if __name__=='__main__':
    main()
