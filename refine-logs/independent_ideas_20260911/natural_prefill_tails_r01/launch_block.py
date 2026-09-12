"""Launch one engine and retain its OS exit status, including startup failures."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

if len(sys.argv) != 2 or sys.argv[1] not in ('forward', 'reverse'):
    raise SystemExit('usage: python launch_block.py forward|reverse')
root = Path(__file__).resolve().parent
block = sys.argv[1]
results = root/'results'
results.mkdir(exist_ok=True)
metadata = results/f'{block}-process.json'
cmd = [sys.executable, '-u', str(root/'run_natural_prefill.py'), '--block', block,
       '--output-dir', str(results/block)]
state = dict(command=cmd, block=block, started_unix_s=time.time(), pid=None, exit_code=None)
with metadata.open('x') as f:
    json.dump(state, f, indent=2)
env = dict(os.environ, CUDA_VISIBLE_DEVICES='0',
           HF_HUB_CACHE=os.environ.get('HF_HUB_CACHE', '/root/autodl-tmp/hf-cache/hub'))
with (results/f'{block}-launch.stdout.log').open('x') as out, \
     (results/f'{block}-launch.stderr.log').open('x') as err:
    # Record this attempt before acquiring the lock; failed starts must archive too.
    try:
        lease = Path('/root/autodl-tmp/moe-gpu0.lock').open('a+')
        fcntl.flock(lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        state.update(status='BLOCKED', exit_code=75, finished_unix_s=time.time(),
                     blocked_reason='GPU_LOCK_HELD', gpu_initialized=False,
                     gpu_processes_started=0, cells_completed=0, warmups_completed=0)
        (results/block).mkdir()
        (results/block/'status.json').write_text(json.dumps(state, indent=2)+'\n')
        metadata.write_text(json.dumps(state, indent=2)+'\n')
        err.write('GPU_LOCK_HELD: shared GPU lock is occupied; no experiment process started.\n')
        raise SystemExit(state['exit_code'])
    child = subprocess.Popen(cmd, cwd=root, env=env, stdout=out, stderr=err)
    state['pid'] = child.pid
    metadata.write_text(json.dumps(state, indent=2)+'\n')
    state['exit_code'] = child.wait()
state['finished_unix_s'] = time.time()
metadata.write_text(json.dumps(state, indent=2)+'\n')
raise SystemExit(state['exit_code'])
