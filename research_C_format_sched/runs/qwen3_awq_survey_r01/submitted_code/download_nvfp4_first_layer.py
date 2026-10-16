#!/usr/bin/env python3
"""Download one publicly calibrated Qwen NVFP4 shard at an immutable revision."""
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT', '60')
from huggingface_hub import hf_hub_download

root = Path('/root/autodl-tmp/research_C_format_sched')
dest = root / 'checkpoints' / 'qwen3_nvfp4_2538ded2'
out = root / 'runs' / 'download_nvfp4_first_layer'
out.mkdir(parents=True, exist_ok=True)
repo = 'nvidia/Qwen3-30B-A3B-NVFP4'
revision = '2538ded2a4edb247b4d2b4a8ba24e44bd4c017c3'
files = ['config.json', 'model.safetensors.index.json', 'model-00001-of-00004.safetensors']
manifest = {'repository': repo, 'revision': revision, 'endpoint': os.environ['HF_ENDPOINT'],
            'started_utc_s': time.time(), 'files': [], 'state': 'RUNNING'}
path = out / 'download.json'
path.write_text(json.dumps(manifest, indent=2) + '\n')
try:
    for filename in files:
        print('Downloading', filename, flush=True)
        file = Path(hf_hub_download(repo, filename, revision=revision, local_dir=dest))
        digest = hashlib.sha256()
        with file.open('rb') as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                digest.update(block)
        manifest['files'].append({'name': filename, 'bytes': file.stat().st_size,
                                  'sha256': digest.hexdigest()})
        path.write_text(json.dumps(manifest, indent=2) + '\n')
        print('Saved', filename, file.stat().st_size, digest.hexdigest(), flush=True)
    manifest['state'] = 'COMPLETE'
except BaseException as e:
    manifest['state'] = 'FAILED'
    manifest['error_type'] = type(e).__name__
    raise
finally:
    manifest['ended_utc_s'] = time.time()
    path.write_text(json.dumps(manifest, indent=2) + '\n')
