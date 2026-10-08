"""Fresh-host setup wait, then one existing six-cell group; no automatic reruns."""
import json
import os
from pathlib import Path
import struct
import subprocess
import time
root=Path(__file__).resolve().parent
os.chdir(root)
status=root/'setup_and_baseline_status.json'
def write(state, **kw):
    status.write_text(json.dumps(dict(status=state,unix_s=time.time(),pid=os.getpid(),**kw),indent=2)+'\n')
def model_ready():
    paths=sorted((root/'model').glob('model-*-of-00003.safetensors'))
    if len(paths)!=3:return False
    for path in paths:
        try:
            with path.open('rb') as f:
                n=struct.unpack('<Q',f.read(8))[0]
                header=json.loads(f.read(n))
            end=max(v['data_offsets'][1] for k,v in header.items() if k!='__metadata__')
            if path.stat().st_size!=8+n+end:return False
        except (OSError,ValueError,KeyError,struct.error):return False
    return True
for attempt in range(180):
    log=(root/'install_uv_resume.log').read_text() if (root/'install_uv_resume.log').exists() else ''
    installed='Installed ' in log and 'vllm==' in log
    ready=model_ready()
    write('WAITING_SETUP',install_complete=installed,model_complete=ready)
    if installed and ready:break
    time.sleep(10)
else:
    write('SETUP_TIMEOUT');raise SystemExit(2)
check=subprocess.run(['python','-c','import torch,vllm; assert vllm.__version__=="0.26.0"; print(torch.__version__, vllm.__version__)'],check=False)
if check.returncode:
    write('IMPORT_FAILED');raise SystemExit(check.returncode)
write('RUNNING_BASELINE')
run=subprocess.run(['python','launch_ngram_short_41307.py','--static-cap','16'],check=False)
write('BASELINE_EXITED',returncode=run.returncode)
raise SystemExit(run.returncode)
