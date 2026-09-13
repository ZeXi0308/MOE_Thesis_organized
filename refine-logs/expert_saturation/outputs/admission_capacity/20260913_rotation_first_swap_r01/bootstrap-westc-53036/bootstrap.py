import json,os,pathlib,subprocess,time
root=pathlib.Path('/root/autodl-tmp/bootstrap-first-swap-20260913-r01')
root.mkdir(exist_ok=False)
py='/root/miniconda3/bin/python'
venv=pathlib.Path('/root/autodl-tmp/expert-saturation/vllm-0.26')
venv.parent.mkdir(parents=True,exist_ok=True)
assert not venv.exists()
state=dict(status='INITIALIZING',pid=os.getpid(),started_unix_s=time.time())
def save(): (root/'status.json').write_text(json.dumps(state,indent=2)+'\n')
save()
subprocess.run([py,'-m','venv',str(venv)],check=True)
env=dict(os.environ,HF_HUB_DISABLE_IMPLICIT_TOKEN='1',HF_HUB_DISABLE_TELEMETRY='1',HF_HUB_DISABLE_XET='1',HF_HUB_CACHE='/root/autodl-tmp/hf-cache/hub',HF_HUB_DOWNLOAD_TIMEOUT='120')
pip=[str(venv/'bin/python'),'-m','pip','install','--no-cache-dir','--disable-pip-version-check','--index-url','https://pypi.org/simple','--report',str(root/'pip-report.json'),'vllm==0.26.0','transformers==5.15.1','torch==2.11.0']
download="from huggingface_hub import snapshot_download; p=snapshot_download('allenai/OLMoE-1B-7B-0924',revision='6d84c48581ece794365f2b8e9cfb043c68ade9c5',allow_patterns=['*.json','*.safetensors','*.model','*.txt'],max_workers=4,token=False); print(p)"
logs=[(root/'install.log').open('w'),(root/'model.log').open('w')]
children=[subprocess.Popen(pip,stdout=logs[0],stderr=subprocess.STDOUT,env=env),subprocess.Popen([py,'-u','-c',download],stdout=logs[1],stderr=subprocess.STDOUT,env=env)]
state.update(status='RUNNING',pip_pid=children[0].pid,model_pid=children[1].pid);save();print(json.dumps(state),flush=True)
for name,process in zip(['pip','model'],children):
 state[name+'_returncode']=process.wait();save();print(name,state[name+'_returncode'],flush=True)
state.update(status='COMPLETE' if all(p.returncode==0 for p in children) else 'FAILED',finished_unix_s=time.time());save()
for log in logs:log.close()
print(json.dumps(state),flush=True)
