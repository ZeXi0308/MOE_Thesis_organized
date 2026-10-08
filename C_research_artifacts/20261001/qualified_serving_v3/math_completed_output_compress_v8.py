#!/usr/bin/env python3
"""Losslessly compact only finished raw logs with independently verified local copies."""
import hashlib,json,lzma,os,time
from pathlib import Path
BASE=Path(__file__).resolve().parent
manifest=json.loads((BASE/'math_completed_output_compression_v8.json').read_text())
receipt=BASE/'math_completed_output_compression_receipt_v8.json'
allowed=('/root/c-qualified-serving-20261001/qwen7b-docqa-concurrency64-v1/native/', '/root/c-qualified-serving-20261001/qwen7b-native-docqa128-v1/native/', '/root/c-qualified-serving-20261001/qwen7b-math-native128-v1/native/', '/root/c-qualified-serving-20261001/qwen7b-math-fixed64-v1/native/')
def digest(p, compressed=False):
 h=hashlib.sha256();size=0
 with (lzma.open(p,'rb') if compressed else p.open('rb')) as f:
  while True:
   b=f.read(1024*1024)
   if not b:break
   size+=len(b);h.update(b)
 return size,h.hexdigest()
def mem():
 b=Path('/sys/fs/cgroup');m=(b/'memory.max').read_text().strip();return None if m=='max' else int(m)-int((b/'memory.current').read_text())
import fcntl
lock=os.open('/root/autodl-tmp/moe-research-gpu.lock',os.O_RDWR|os.O_NOFOLLOW)
assert (os.fstat(lock).st_dev,os.fstat(lock).st_ino)==(2304,25841682495)
deadline=time.monotonic()+180
while True:
 try:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
 except BlockingIOError:
  assert time.monotonic()<deadline,'shared lock wait expired without mutation'
  time.sleep(2)
assert (os.stat('/root/autodl-tmp/moe-research-gpu.lock').st_dev,os.stat('/root/autodl-tmp/moe-research-gpu.lock').st_ino)==(2304,25841682495)
r=dict(status='RUNNING',start_unix_s=time.time(),memory_headroom_before=mem(),files=[])
with receipt.open('x') as f:json.dump(r,f,indent=2)
try:
 for item in manifest['files']:
  p=Path(item['remote']);c=Path(str(p)+'.xz')
  assert str(p).startswith(allowed) and p.name in ('measured-outputs.json','rendered-inputs.json','measured-pressure.json')
  assert p.is_file() and not p.is_symlink() and p.resolve()==p
  assert json.loads((p.parent/'status.json').read_text())['status']=='COMPLETE'
  assert json.loads((p.parent.parent/'launcher-receipt.json').read_text())['status']=='COMPLETE'
  assert digest(p)==(item['size'],item['sha256'])
  assert not c.exists() and not c.is_symlink()
  with p.open('rb') as src,lzma.open(c,'xb',preset=1) as dst:
   while True:
    chunk=src.read(1024*1024)
    if not chunk:break
    dst.write(chunk)
  assert digest(c,True)==(item['size'],item['sha256'])
  saved=item['size']-c.stat().st_size
  p.unlink()
  r['files'].append(dict(item,compressed_path=str(c),compressed_bytes=c.stat().st_size,bytes_released=saved,decompressed_verified=True))
  receipt.write_text(json.dumps(r,indent=2)+'\n')
  print(json.dumps(dict(completed=len(r['files']),bytes_released=sum(x['bytes_released'] for x in r['files']))),flush=True)
 r['status']='COMPLETE'
except BaseException as e:
 r.update(status='INCOMPLETE',error=str(e));raise
finally:
 r.update(end_unix_s=time.time(),memory_headroom_after=mem());receipt.write_text(json.dumps(r,indent=2)+'\n')
