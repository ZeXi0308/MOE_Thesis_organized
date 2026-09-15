"""Retain the local driver log and its actual OS exit code for this campaign."""
import json
from pathlib import Path
import subprocess
import sys
import time

root=Path(__file__).resolve().parents[1]
record=root/'local-driver-process.json'
command=[sys.executable,'-u',str(root/'operations/run_prepared_local.py')]
with record.open('x') as meta,(root/'local-driver.stdout.log').open('x') as out,(root/'local-driver.stderr.log').open('x') as err:
    child=subprocess.Popen(command,cwd=root,stdout=subprocess.PIPE,stderr=err,text=True)
    state={'pid':child.pid,'command':command,'started_unix_s':time.time(),'exit_code':None}
    json.dump(state,meta,indent=2);meta.flush()
    print(json.dumps({'stage':'LOCAL_DRIVER_STARTED',**state}),flush=True)
    for line in child.stdout:
        out.write(line);out.flush();print(line,end='',flush=True)
    state['exit_code']=child.wait()
state['finished_unix_s']=time.time()
record.write_text(json.dumps(state,indent=2)+'\n')
print(json.dumps({'stage':'LOCAL_DRIVER_EXIT',**state}),flush=True)
raise SystemExit(state['exit_code'])
