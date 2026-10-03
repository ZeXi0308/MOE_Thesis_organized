"""Read-only pinned-cache/process/public-metadata query; never starts a download."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

HERE = Path(__file__).resolve().parent
QUERY = r'''
import hashlib,json,os,pathlib,time,urllib.parse,urllib.request
repo=pathlib.Path('/root/autodl-tmp/hf-cache/hub/models--allenai--OLMoE-1B-7B-0924')
revision='6d84c48581ece794365f2b8e9cfb043c68ade9c5'; snapshot=repo/'snapshots'/revision
index=json.loads((snapshot/'model.safetensors.index.json').read_text())
tree=json.loads((repo/'trees'/f'{revision}.json').read_text())['files']
names=sorted(set(index['weight_map'].values()))
shards=[]
for name in names:
 expected=tree[name]; path=snapshot/name
 partial=[dict(name=p.name,bytes=p.stat().st_size,mtime=p.stat().st_mtime) for p in (repo/'blobs').glob(expected['lfs_sha256']+'*.incomplete')]
 shards.append(dict(name=name,expected_bytes=expected['size'],expected_sha256=expected['lfs_sha256'],resolved=path.exists(),bytes=path.stat().st_size if path.exists() else None,incomplete=partial))
processes=[]
for pid in [2025,2498,2975]:
 proc=pathlib.Path('/proc')/str(pid)
 if not proc.exists():processes.append(dict(pid=pid,exists=False));continue
 status={r.split(':',1)[0]:r.split(':',1)[1].strip() for r in (proc/'status').read_text().splitlines() if ':' in r}
 fields=[b'HF_ENDPOINT',b'HF_HUB_DISABLE_XET',b'HF_XET_HIGH_PERFORMANCE',b'HF_HUB_DOWNLOAD_TIMEOUT',b'HF_HUB_ETAG_TIMEOUT',b'HF_XET_NUM_CONCURRENT_RANGE_GETS']
 selected={}
 # Only expose the explicitly requested non-secret download configuration.
 for entry in (proc/'environ').read_bytes().split(b'\0'):
  key,sep,value=entry.partition(b'=')
  if key in fields:
   value=value.decode(errors='replace')
   selected[key.decode()]=urllib.parse.urlparse(value).hostname if key==b'HF_ENDPOINT' else value
 maps=(proc/'maps').read_text()
 cache_fds=[]
 for fd in (proc/'fd').iterdir():
  try:
   target=str(fd.readlink())
   if str(repo) in target or 'hub/.locks/models--allenai--OLMoE-1B-7B-0924/' in target:cache_fds.append(dict(fd=fd.name,path=target))
  except OSError:pass
 processes.append(dict(pid=pid,exists=True,name=status['Name'],state=status['State'],ppid=int(status['PPid']),nonsecret_download_configuration=selected,hf_xet_library_mapped='hf_xet' in maps,cache_fds=cache_fds))
url='https://huggingface.co/api/models/allenai/OLMoE-1B-7B-0924/revision/'+revision+'?blobs=true'
started=time.time()
try:
 with urllib.request.urlopen(url,timeout=20) as response:public=json.load(response)
 public_metadata=dict(status='READ',revision=public.get('sha'),shards=[s for s in public.get('siblings',[]) if s.get('rfilename') in names],elapsed_s=time.time()-started)
except Exception as error:public_metadata=dict(status='FAILED',error_type=type(error).__name__,error=str(error),elapsed_s=time.time()-started)
print(json.dumps(dict(event='read_only_model_progress',observed_unix_s=time.time(),revision=revision,shards=shards,matching_processes=processes,public_metadata_url=url,public_metadata=public_metadata)))
'''
command = ['ssh', '-o', 'ConnectTimeout=15', '-o', 'NumberOfPasswordPrompts=1',
           '-o', 'StrictHostKeyChecking=yes', '-p', '11155',
           'root@connect.weste.seetacloud.com', '/root/miniconda3/bin/python', '-']
env = dict(os.environ, SSH_ASKPASS='/tmp/moe-new-endpoint-askpass.py',
           SSH_ASKPASS_REQUIRE='force', DISPLAY=':0')
record = dict(command=command, query_sha256=hashlib.sha256(QUERY.encode()).hexdigest(),
              local_observed_unix_s=time.time())
try:
    result = subprocess.run(command, input=QUERY, text=True, capture_output=True,
                            env=env, timeout=50, check=True)
    record.update(json.loads(result.stdout))
except Exception as error:
    record.update(event='READINESS_QUERY_FAILED', error_type=type(error).__name__, error=str(error))
with (HERE / 'model-readiness-20260910-agent.jsonl').open('a') as stream:
    stream.write(json.dumps(record) + '\n')
print(json.dumps(record))
