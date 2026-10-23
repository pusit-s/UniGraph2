"""Run paired MLP/attention x none/task-only/RoMA conditions for one encoder seed."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    args, overrides = parser.parse_known_args()
    if any(arg.split('=')[0] in ['--output', '--seed', '--expert', '--router-posttrain', '--resume-pretrain'] for arg in overrides):
        parser.error('comparison condition arguments cannot be overridden')
    args.output.mkdir(parents=True, exist_ok=True)
    runs = []
    for expert in ['mlp', 'attention']:
        baseline = args.output/f'{expert}-none'
        for method in ['none', 'task', 'roma']:
            output = args.output/f'{expert}-{method}'
            command = [sys.executable, '-u', 'run.py', '--output', str(output),
                       '--seed', str(args.seed), '--expert', expert, '--router-posttrain', method, *overrides]
            if method != 'none':
                command += ['--resume-pretrain', str(baseline)]
            subprocess.run(command, check=True)
            subprocess.run([sys.executable, 'summarize.py', str(output)], check=True)
            runs.append(str(output))
    (args.output/'manifest.json').write_text(json.dumps(
        {'status': 'completed', 'encoder_seed': args.seed, 'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
         'runs': runs, 'overrides': overrides}, indent=2))


if __name__ == '__main__':
    main()
