from pathlib import Path
import os,json,subprocess,time
root=Path('/root/autodl-tmp/moe-prefill-share-01a07d4b-20260911-r01')
gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True).strip()
assert not gpu, 'GPU became occupied; no launch'
others=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit() or int(p.name)==os.getpid(): continue
 try:
  args=(p/'cmdline').read_bytes().split(b'\0')
  if any(b'moe-' in a for a in args): others.append(int(p.name))
 except OSError: pass
assert not others, 'another MoE launcher remains active: '+str(others)
cmd=['/root/autodl-tmp/expert-saturation/vllm-0.26/bin/python','-u','launch_block.py','forward']
env=dict(os.environ,HF_HUB_CACHE='/root/autodl-tmp/moe-model-cache-01a07d4b/hub',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
with (root/'forward-supervisor.json').open('x') as meta, (root/'forward-supervisor.stdout.log').open('x') as out, (root/'forward-supervisor.stderr.log').open('x') as err:
 p=subprocess.Popen(cmd,cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
 state={'pid':p.pid,'command':cmd,'started_unix_s':time.time(),'proc_start_ticks':Path('/proc',str(p.pid),'stat').read_text().split()[21]}
 json.dump(state,meta,indent=2)
print(json.dumps(state,indent=2))
