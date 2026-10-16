#!/usr/bin/env python3
"""Quarantine three proven orphan modules from a prior torch install.

No wheel-owned file is changed. Backups and hashes make this reversible.
"""
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import shutil
import time

root = Path('/root/autodl-tmp/research_C_format_sched/runs/torch_repair_20261016')
root.mkdir(parents=True, exist_ok=False)
dist = metadata.distribution('torch')
owned = {str(p) for p in dist.files}
names = ('flex_attention.py', 'flex_decoding.py', 'mm_scaled_grouped.py')
records = []
for name in names:
    rel = 'torch/_inductor/kernel/' + name
    if rel in owned:
        raise RuntimeError(f'Refusing to alter wheel-owned file: {rel}')
    source = Path(dist.locate_file(rel))
    data = source.read_bytes()
    dest = root / name
    shutil.copy2(source, dest)
    if dest.read_bytes() != data:
        raise RuntimeError('Backup differs from source')
    records.append({'source': str(source), 'backup': str(dest),
                    'sha256': hashlib.sha256(data).hexdigest(), 'wheel_owned': False})
(root / 'repair.json').write_text(json.dumps({'time_utc_s': time.time(), 'torch_version': dist.version,
    'reason': 'orphaned prior-version modules cause duplicate TritonTemplate registration',
    'files': records}, indent=2) + '\n')
for r in records:
    source = Path(r['source'])
    if hashlib.sha256(source.read_bytes()).hexdigest() != r['sha256']:
        raise RuntimeError('Source changed after backup; refusing to remove')
    source.unlink()
print(json.dumps(records, indent=2))
