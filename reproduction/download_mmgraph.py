"""Download only missing CLIP/task files from a pinned public MM-Graph release.

Run through Slurm. Writes directly to the canonical data root, verifies LFS SHA256,
and never installs packages or duplicates datasets into the repository.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
from urllib.request import urlopen


REPOSITORY = 'mm-graph-org/mm-graph'
DOMAINS = ['books-nc', 'books-lp', 'ele-fashion', 'sports-copurchase', 'cloth-copurchase']
REVISION = 'c707a3203b77a153f11370a05d2a9be6571f7670'


def digest(path):
    sha = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8*1024*1024), b''):
            sha.update(chunk)
    return sha.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=Path('/home/pusit.see/dataset/raw/paper_2602_00067/mm_graph'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    with urlopen(f'https://huggingface.co/api/datasets/{REPOSITORY}/tree/{REVISION}?recursive=true&expand=false', timeout=60) as response:
        tree = json.load(response)
    required = {f'{domain}/{name}' for domain in DOMAINS for name in
                (['clip_feat.pt', 'nc_edges-nodeid.pt', 'labels-w-missing.pt', 'split.pt']
                 if domain in ['books-nc', 'ele-fashion'] else ['clip_feat.pt', 'lp-edge-split.pt'])}
    entries = [entry for entry in tree if entry['path'] in required]
    if {entry['path'] for entry in entries} != required:
        raise RuntimeError('the pinned release does not contain all requested files')

    def fetch(entry):
        path = args.data_root/entry['path']
        expected = entry.get('lfs', {}).get('oid')
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            if path.stat().st_size != entry['size']:
                raise RuntimeError(f'existing source file has a different size; preserving it: {path}')
            actual = digest(path)
            if expected and actual != expected:
                raise RuntimeError(f'existing source checksum mismatch; preserving it: {path}')
            print('VERIFIED', entry['path'], flush=True)
            return {'path': str(path), 'bytes': entry['size'], 'sha256': actual, 'action': 'verified existing'}
        partial = path.with_suffix('.pt.download')
        for attempt in range(4):
            try:
                sha = hashlib.sha256()
                with urlopen(f'https://huggingface.co/datasets/{REPOSITORY}/resolve/{REVISION}/{entry["path"]}?download=true', timeout=90) as response, partial.open('wb') as handle:
                    for chunk in iter(lambda: response.read(8*1024*1024), b''):
                        handle.write(chunk)
                        sha.update(chunk)
                if partial.stat().st_size != entry['size'] or (expected and sha.hexdigest() != expected):
                    raise RuntimeError(f'download integrity check failed: {path}')
                partial.replace(path)
                print('DOWNLOADED', entry['path'], entry['size'], flush=True)
                return {'path': str(path), 'bytes': entry['size'], 'sha256': sha.hexdigest(), 'action': 'downloaded'}
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(2**attempt)

    with ThreadPoolExecutor(max_workers=2) as pool:
        files = list(pool.map(fetch, entries))
    report = {'status': 'completed', 'repository': REPOSITORY, 'revision': REVISION,
              'domains': DOMAINS, 'files': files,
              'unsupported_release_folders': {'mm-codex-s': 'Knowledge-graph triples are supplied without cached CLIP node features.',
                                              'mm-codex-m': 'Knowledge-graph triples are supplied without cached CLIP node features.'}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print('COMPLETE', args.output, flush=True)


if __name__ == '__main__':
    main()
