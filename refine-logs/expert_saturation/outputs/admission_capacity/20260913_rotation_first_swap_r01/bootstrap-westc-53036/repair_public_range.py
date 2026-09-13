import concurrent.futures,hashlib,json,os,pathlib,re,subprocess,time
root=pathlib.Path('/root/autodl-tmp/bootstrap-first-swap-20260913-r01/shard2-assist')
state=json.loads((root/'status.json').read_text());assert state['status']=='FAILED'
repair=root/'repair-01';repair.mkdir(exist_ok=False)
digest=state['expected_sha256'];size=state['expected_bytes'];prefix=state['readonly_prefix_bytes'];chunk=64*1024*1024
url='https://hf-mirror.com/allenai/OLMoE-1B-7B-0924/resolve/6d84c48581ece794365f2b8e9cfb043c68ade9c5/model-00002-of-00003.safetensors'
missing=3758096384
assert (root/'assembled.partial').stat().st_size==prefix
for start in range(prefix,size,chunk):
 if start==missing:continue
 p=root/f'range-{start}';expected=(start,min(size-1,start+chunk-1),size)
 ranges=re.findall(r'(?im)^content-range:\s*bytes\s+(\d+)-(\d+)/(\d+)',p.with_suffix('.headers').read_text())
 assert ranges and tuple(map(int,ranges[-1]))==expected and p.stat().st_size==expected[1]-start+1

def fetch(start):
 end=start+16*1024*1024-1;p=repair/str(start);header=p.with_suffix('.headers')
 subprocess.run(['curl','-L','--fail','--silent','--show-error','--range',f'{start}-{end}','--max-time','90','--retry','1','--dump-header',str(header),'-o',str(p),url+f'?download=true&part={start}'],check=True,stderr=subprocess.DEVNULL)
 ranges=re.findall(r'(?im)^content-range:\s*bytes\s+(\d+)-(\d+)/(\d+)',header.read_text())
 assert ranges and tuple(map(int,ranges[-1]))==(start,end,size) and p.stat().st_size==end-start+1
 return p
starts=list(range(missing,missing+chunk,16*1024*1024))
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
 for p in pool.map(fetch,starts):print('verified repaired subrange',p.name,flush=True)
fixed=repair/'fixed-range'
with fixed.open('xb') as dst:
 for start in starts:dst.write((repair/str(start)).read_bytes())
assembled=root/'assembled.partial'
with assembled.open('ab') as dst:
 for start in range(prefix,size,chunk):
  p=fixed if start==missing else root/f'range-{start}'
  with p.open('rb') as src:
   while block:=src.read(8*1024*1024):dst.write(block)
assert assembled.stat().st_size==size
with assembled.open('rb') as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
assert actual==digest,('full hash mismatch',actual)
model=pathlib.Path('/root/autodl-tmp/hf-cache/hub/models--allenai--OLMoE-1B-7B-0924')
target=model/'blobs'/digest
try:os.link(assembled,target)
except FileExistsError:
 with target.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==digest
link=model/'snapshots/6d84c48581ece794365f2b8e9cfb043c68ade9c5/model-00002-of-00003.safetensors'
try:link.symlink_to('../../blobs/'+digest)
except FileExistsError:assert link.resolve()==target
result=dict(status='VERIFIED_IMMUTABLE_BLOB_READY',sha256=actual,bytes=size,finished_unix_s=time.time(),original_download_untouched=True,prior_attempt='../status.json')
(repair/'status.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
