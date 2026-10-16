#!/usr/bin/env python3
"""Download three pinned Qwen checkpoints, reusing existing first shards."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT', '60')


def main():
    from huggingface_hub import HfApi
    p = argparse.ArgumentParser()
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--reserve-gib', type=float, default=20)
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    a.out_dir.mkdir(parents=True, exist_ok=False)
    catalog = json.loads((root / 'configs/models.json').read_text())['models']['qwen3']['formats']
    dests = {'bf16': 'qwen3_bf16_ad44e777', 'awq': 'qwen3_awq_c3b14f75', 'nvfp4': 'qwen3_nvfp4_2538ded2'}
    record = {'state': 'PREPARING', 'controller_pid': os.getpid(), 'started_utc_s': time.time(),
              'scope': 'public model downloads only; no GPU', 'models': {}}
    mutex = threading.Lock()
    def save():
        tmp = a.out_dir / 'status.tmp'
        tmp.write_text(json.dumps(record, indent=2) + '\n')
        tmp.replace(a.out_dir / 'status.json')
    save()
    api = HfApi()
    total_missing = 0
    for fmt, name in dests.items():
        spec = catalog[fmt]
        info = api.model_info(spec['repo_id'], revision=spec['revision'], files_metadata=True)
        dest = root / 'checkpoints' / name
        dest.mkdir(parents=True, exist_ok=True)
        files = [x for x in info.siblings if x.rfilename.endswith(('.safetensors', '.json', '.yaml', '.jinja', '.txt', '.model', '.md'))]
        if any(x.size is None for x in files):
            raise ValueError('Missing source file size')
        missing = sum(x.size for x in files if not (dest / x.rfilename).exists()
                      or (dest / x.rfilename).stat().st_size != x.size)
        total_missing += missing
        record['models'][fmt] = {'repo': info.id, 'revision': info.sha, 'directory': str(dest),
                                 'planned_missing_bytes': missing, 'state': 'PENDING'}
    free = shutil.disk_usage(root).free
    record.update(free_bytes_at_start=free, planned_missing_bytes=total_missing, reserve_gib=a.reserve_gib)
    if free < total_missing + a.reserve_gib * 2**30:
        record.update(state='FAILED', error='Insufficient space for the aggregate download plan')
        save()
        raise RuntimeError(record['error'])
    record['state'] = 'RUNNING'
    save()
    def download(fmt):
        item = record['models'][fmt]
        cmd = [sys.executable, str(root / 'code/prepare_quantization.py'), 'fetch', '--model', 'qwen3',
               '--format', fmt, '--output', item['directory'], '--reserve-gib', str(a.reserve_gib)]
        with (a.out_dir / f'{fmt}.log').open('w') as log:
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
            with mutex:
                item.update(state='RUNNING', pid=proc.pid, command=cmd)
                save()
            rc = proc.wait()
        with mutex:
            item.update(state='COMPLETE' if rc == 0 else 'FAILED', exit_code=rc, ended_utc_s=time.time())
            save()
        return rc
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = [f.result() for f in as_completed([pool.submit(download, fmt) for fmt in dests])]
    record.update(state='COMPLETE' if all(rc == 0 for rc in results) else 'FAILED', ended_utc_s=time.time())
    save()
    print(json.dumps(record, indent=2))
    raise SystemExit(0 if record['state'] == 'COMPLETE' else 1)


if __name__ == '__main__':
    main()
