"""Add one verified immutable public blob; never alter another download's partial."""
import concurrent.futures, hashlib, json, os, pathlib, re, subprocess, time
root=pathlib.Path('/root/autodl-tmp/bootstrap-first-swap-20260913-r01/shard2-assist')
root.mkdir(exist_ok=False)
blobs=pathlib.Path('/root/autodl-tmp/hf-cache/hub/models--allenai--OLMoE-1B-7B-0924/blobs')
digest='15ef5c730ee3cfed7199498788cd2faf337203fc74b529625e7502cdd759f4a7'
size=4997235176
chunk=64*1024*1024
target=blobs/digest
state=dict(status='STARTING',pid=os.getpid(),expected_sha256=digest,expected_bytes=size,started_unix_s=time.time())
def save():
 (root/'status.json').write_text(json.dumps(state,indent=2)+'\n')
def sha(path):
 with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
if target.exists():
 assert target.stat().st_size==size and sha(target)==digest
 state['status']='ALREADY_READY';save();print(json.dumps(state),flush=True);raise SystemExit(0)
partials=list(blobs.glob(digest+'*.incomplete'))
assert len(partials)==1
prefix=(partials[0].stat().st_size//chunk)*chunk
assembled=root/'assembled.partial'
with partials[0].open('rb') as src, assembled.open('xb') as dst:
 remaining=prefix
 while remaining:
  block=src.read(min(8*1024*1024,remaining));assert block
  dst.write(block);remaining-=len(block)
state.update(status='DOWNLOADING_RANGES',readonly_prefix_bytes=prefix,total_ranges=(size-prefix+chunk-1)//chunk,completed_ranges=0);save();print(json.dumps(state),flush=True)
url='https://hf-mirror.com/allenai/OLMoE-1B-7B-0924/resolve/6d84c48581ece794365f2b8e9cfb043c68ade9c5/model-00002-of-00003.safetensors'
def fetch(start):
 end=min(size-1,start+chunk-1); p=root/f'range-{start}';hdr=p.with_suffix('.headers')
 subprocess.run(['curl','-L','--fail','--silent','--show-error','--range',f'{start}-{end}','--max-time','150','--retry','2','--dump-header',str(hdr),'-o',str(p),url],check=True,stderr=subprocess.DEVNULL)
 content=hdr.read_text(errors='replace');ranges=re.findall(r'(?im)^content-range:\s*bytes\s+(\d+)-(\d+)/(\d+)',content)
 assert ranges and tuple(map(int,ranges[-1]))==(start,end,size),('range mismatch',start)
 assert p.stat().st_size==end-start+1,('size mismatch',start)
 return p
starts=list(range(prefix,size,chunk))
try:
 with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
  jobs={pool.submit(fetch,s):s for s in starts}
  for f in concurrent.futures.as_completed(jobs):
   f.result();state['completed_ranges']+=1;save();print(json.dumps(dict(completed_ranges=state['completed_ranges'],total_ranges=len(starts))),flush=True)
 with assembled.open('ab') as dst:
  for start in starts:
   with (root/f'range-{start}').open('rb') as src:
    while block:=src.read(8*1024*1024):dst.write(block)
 assert assembled.stat().st_size==size and sha(assembled)==digest,'full content hash mismatch'
 try:os.link(assembled,target)
 except FileExistsError:assert target.stat().st_size==size and sha(target)==digest
 state.update(status='VERIFIED_IMMUTABLE_BLOB_READY',sha256=digest,finished_unix_s=time.time());save();print(json.dumps(state),flush=True)
except BaseException as exc:
 state.update(status='FAILED',error=str(exc),finished_unix_s=time.time());save();raise
