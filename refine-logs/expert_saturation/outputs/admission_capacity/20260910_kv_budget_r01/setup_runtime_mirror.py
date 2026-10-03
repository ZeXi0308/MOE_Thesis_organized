"""Install the frozen runtime into its dedicated venv; no GPU experiment."""
import json, pathlib, subprocess, sys, time
root=pathlib.Path(__file__).resolve().parent
state_path=root/'setup-runtime-mirror-state.json'
command=['/root/autodl-tmp/expert-saturation/vllm-0.26/bin/python','-m','pip','install','--index-url','https://mirrors.aliyun.com/pypi/simple','--timeout','90','--retries','3','vllm==0.26.0','transformers==5.15.1','torch==2.11.0']
state=dict(status='STARTING',command=command,started_unix_s=time.time(),launcher_pid=__import__('os').getpid())
with state_path.open('x') as f:json.dump(state,f,indent=2)
with (root/'setup-runtime-mirror.stdout.log').open('x') as stdout,(root/'setup-runtime-mirror.stderr.log').open('x') as stderr:
 p=subprocess.Popen(command,stdout=stdout,stderr=stderr)
 state.update(status='RUNNING',child_pid=p.pid);state_path.write_text(json.dumps(state,indent=2))
 state.update(status='EXITED',returncode=p.wait(),finished_unix_s=time.time());state_path.write_text(json.dumps(state,indent=2))
