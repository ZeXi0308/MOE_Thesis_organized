"""Verify the completed environment copy without initializing CUDA."""
import hashlib
import importlib.metadata as metadata
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path('/root/autodl-tmp/moe-transfer-weste-26862-20260915-r01')
OUT = ROOT / 'verification.json'
assert not OUT.exists(), 'Retain prior verification; do not overwrite'
source = json.loads((ROOT / 'source_environment.json').read_text())
packages = {d.metadata['Name']: d.version for d in metadata.distributions()}
assert packages == source['packages'], 'Installed distribution list differs'
assert sys.version == source['python'], 'Python build differs'
vllm = Path(importlib.util.find_spec('vllm').origin).parent
hashes = {name: hashlib.sha256((vllm / name).read_bytes()).hexdigest()
          for name in source['vllm_source_sha256']}
assert hashes == source['vllm_source_sha256'], 'Runtime source differs'
repo = Path('/root/autodl-tmp/hf-cache/hub/models--allenai--OLMoE-1B-7B-0924')
snapshot = repo / 'snapshots' / source['model_snapshot']
assert snapshot.is_dir()
blobs = {}
for path in sorted((repo / 'blobs').iterdir()):
    assert path.is_file() and re.fullmatch(r'[0-9a-f]{40}|[0-9a-f]{64}', path.name), path
    size = path.stat().st_size
    digest = hashlib.sha256() if len(path.name) == 64 else hashlib.sha1()
    if len(path.name) == 40:
        digest.update(f'blob {size}\0'.encode())
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    assert digest.hexdigest() == path.name, f'Blob content mismatch: {path.name}'
    blobs[path.name] = size
files = sorted(str(p.relative_to(snapshot)) for p in snapshot.rglob('*') if p.is_file())
assert files and 'config.json' in files
for path in snapshot.rglob('*'):
    if path.is_symlink():
        assert path.resolve().is_file() and path.resolve().is_relative_to(repo / 'blobs'), path
for index in snapshot.glob('*.index.json'):
    for name in set(json.loads(index.read_text())['weight_map'].values()):
        assert (snapshot / name).is_file(), name
gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name,memory.total,driver_version',
                               '--format=csv,noheader'], text=True)
processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,used_gpu_memory',
                                     '--format=csv,noheader'], text=True).strip()
assert not processes, 'Another compute process is present'
result = dict(status='PASS_ENVIRONMENT_AND_MODEL_CPU_VERIFICATION', checked_unix_s=time.time(),
              source_receipt_sha256=hashlib.sha256((ROOT / 'source_environment.json').read_bytes()).hexdigest(),
              python=sys.version, package_count=len(packages),
              versions={k: packages[k] for k in ['torch', 'vllm', 'transformers']},
              runtime_source_sha256=hashes, model_snapshot=source['model_snapshot'],
              blobs=blobs, total_blob_bytes=sum(blobs.values()), snapshot_files=files,
              gpu_snapshot=gpu, compute_processes=processes,
              gpu_initialized=False, performance_result=False)
with OUT.open('x') as handle:
    handle.write(json.dumps(result, indent=2) + '\n')
print(json.dumps({k: v for k, v in result.items()
                  if k not in ['runtime_source_sha256', 'blobs', 'snapshot_files']}))
