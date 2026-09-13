import os
os.sched_setaffinity(0,{8})
import gzip,hashlib,json,os,shutil,time
from pathlib import Path
source=Path('/root/autodl-tmp/qwen3-new-gpu-localized-static-v026-r01/shard_workspace/serial-shards-hqv2zn1f/model-00003-of-00016.safetensors')
assert source.stat().st_size==752877568
rows=[]
with source.open('rb') as f:
 for offset in [0,256*1024**2,512*1024**2]:
  f.seek(offset);data=f.read(8*1024**2);start=time.monotonic();compressed=gzip.compress(data,compresslevel=1,mtime=0)
  assert gzip.decompress(compressed)==data
  rows.append(dict(offset=offset,bytes=len(data),gzip_bytes=len(compressed),ratio=len(compressed)/len(data),wall_s=time.monotonic()-start,source_sample_sha256=hashlib.sha256(data).hexdigest()))
print(json.dumps(dict(scope='Read-only three8MiB samples from retained r01 partial shard; compression estimate only, not whole-model ratio or full-shard integrity.',rows=rows,disks={str(p):shutil.disk_usage(p)._asdict() for p in [Path('/root'),Path('/root/autodl-tmp')]},affinity=sorted(os.sched_getaffinity(0)))))
