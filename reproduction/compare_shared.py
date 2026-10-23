"""Compare MLP, attention experts and shared pre-router attention for one seed."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


ARCHITECTURES = {
    'mlp': ['--expert', 'mlp'],
    'attention': ['--expert', 'attention'],
    'shared': ['--expert', 'mlp', '--shared-attention', '--attention-heads', '4'],
}
MODES = ['none', 'task', 'roma']
CONTROLLED = {'--output', '--seed', '--expert', '--router-posttrain', '--resume-pretrain',
              '--shared-attention', '--no-shared-attention', '--attention-heads'}
CORE_SOURCES = ['model.py', 'graph_data.py', 'run.py', 'roma.py', 'link_probe.py']


def _run_cases(args, overrides):
    """Run/reuse all nine cases while the caller holds the output-directory lock."""
    source_hashes = {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in CORE_SOURCES}
    runs = []
    for architecture, architecture_args in ARCHITECTURES.items():
        baseline = args.output/f'{architecture}-none'
        for mode in MODES:
            output = args.output/f'{architecture}-{mode}'
            command = [sys.executable, '-u', 'run.py', '--output', str(output), '--seed', str(args.seed),
                       *architecture_args, '--router-posttrain', mode, *overrides]
            if mode != 'none':
                command += ['--resume-pretrain', str(baseline)]
            signature = {'command': command, 'core_source_sha256': source_hashes}
            invocation = output/'invocation.json'
            if (output/'results.json').exists() and not invocation.exists():
                raise ValueError(f'cannot reuse existing results without a pre-existing invocation: {output}')
            if invocation.exists() and json.loads(invocation.read_text()) != signature:
                raise ValueError(f'cannot reuse a case with a different command or implementation: {output}')
            output.mkdir(parents=True, exist_ok=True)
            invocation.write_text(json.dumps(signature, indent=2))
            if (output/'results.json').exists():
                report = json.loads((output/'results.json').read_text())
                if report.get('status') != 'completed' or set(report['results']) != {'CLIP', 'NoPretrain', 'UniGraph2'}:
                    raise ValueError(f'existing results are incomplete: {output}')
                if any(report['source_sha256'][name] != digest for name, digest in source_hashes.items()):
                    raise ValueError(f'existing results have a different implementation: {output}')
                for filename, metadata in report['data_files'].items():
                    stat = Path(filename).stat()
                    if (stat.st_size, stat.st_mtime_ns) != (metadata['bytes'], metadata['mtime_ns']):
                        raise ValueError(f'input data changed: {filename}')
                print('REUSE_COMPLETED', output, flush=True)
            else:
                subprocess.run(command, check=True)
            subprocess.run([sys.executable, 'summarize.py', str(output)], check=True)
            runs.append(str(output))
    (args.output/'manifest.json').write_text(json.dumps(
        {'status': 'completed', 'encoder_seed': args.seed, 'architectures': list(ARCHITECTURES),
         'slurm_job_id': os.environ.get('SLURM_JOB_ID'), 'runs': runs, 'overrides': overrides}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    args, overrides = parser.parse_known_args()
    if any(argument.split('=')[0] in CONTROLLED for argument in overrides):
        parser.error('architecture, seed, stage and output arguments are controlled by this comparison')
    args.output.mkdir(parents=True, exist_ok=True)
    lock_path = args.output/'.comparison.lock'
    with lock_path.open('a') as lock:
        print('WAIT_COMPARISON_LOCK', lock_path, flush=True)
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        print('ACQUIRED_COMPARISON_LOCK', lock_path, flush=True)
        try:
            _run_cases(args, overrides)
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


if __name__ == '__main__':
    main()
