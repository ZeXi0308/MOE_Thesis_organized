#!/usr/bin/env python3
"""Fetch pinned checkpoint metadata and optionally its first expert shard."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT', '60')


def main():
    from huggingface_hub import hf_hub_download, hf_hub_url, get_hf_file_metadata
    p = argparse.ArgumentParser()
    p.add_argument('--repo', required=True)
    p.add_argument('--revision', required=True)
    p.add_argument('--dest', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--first-expert-shard', action='store_true')
    p.add_argument('--reserve-gib', type=float, default=6)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    a.dest.mkdir(parents=True, exist_ok=True)
    manifest = {'repository': a.repo, 'revision': a.revision,
                'endpoint': os.environ['HF_ENDPOINT'], 'state': 'RUNNING',
                'started_utc_s': time.time(), 'files': []}
    def save():
        (a.out / 'download.json').write_text(json.dumps(manifest, indent=2) + '\n')
    def fetch(name):
        meta = get_hf_file_metadata(hf_hub_url(a.repo, name, revision=a.revision))
        dest = a.dest / name
        if not dest.exists() and (meta.size is None or shutil.disk_usage(a.dest).free < meta.size + a.reserve_gib * 2**30):
            raise RuntimeError(f'Insufficient reserved disk space for {name}: {meta.size}')
        print('Downloading', name, meta.size, flush=True)
        dest = Path(hf_hub_download(a.repo, name, revision=a.revision, local_dir=a.dest))
        digest = hashlib.sha256()
        with dest.open('rb') as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                digest.update(block)
        actual = digest.hexdigest()
        etag = (meta.etag or '').strip('"')
        if len(etag) == 64 and actual != etag:
            raise ValueError(f'SHA256 differs from source ETag: {name}')
        manifest['files'].append({'name': name, 'bytes': dest.stat().st_size,
                                  'sha256': actual, 'source_etag': etag})
        save()
    save()
    try:
        fetch('config.json')
        fetch('model.safetensors.index.json')
        index = json.loads((a.dest / 'model.safetensors.index.json').read_text())['weight_map']
        shards = sorted({v for k, v in index.items() if k.startswith('model.layers.0.mlp.experts.')})
        manifest['first_layer_expert_shards'] = shards
        save()
        if a.first_expert_shard:
            if len(shards) != 1:
                raise ValueError(f'First-layer experts span {len(shards)} shards; explicit plan required')
            fetch(shards[0])
        manifest['state'] = 'COMPLETE'
    except BaseException as e:
        manifest.update(state='FAILED', error=f'{type(e).__name__}: {e}')
        raise
    finally:
        manifest['ended_utc_s'] = time.time()
        save()


if __name__ == '__main__':
    main()
