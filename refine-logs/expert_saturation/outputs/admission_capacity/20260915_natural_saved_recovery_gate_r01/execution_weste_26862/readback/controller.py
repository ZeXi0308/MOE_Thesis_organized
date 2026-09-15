"""Foreground execution of one accepted, immutable, serial experiment package."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

root = Path(__file__).resolve().parent
for name, expected in json.loads((root/'manifest.json').read_text()).items():
    if hashlib.sha256((root/name).read_bytes()).hexdigest() != expected:
        raise RuntimeError(f'Package drift before launch: {name}')
with (root/'launch-once').open('x') as f:
    f.write(str(os.getpid()))
state = dict(status='STARTING', pid=os.getpid(), started=time.time())


def save():
    p = root/'group-status.json.tmp'
    p.write_text(json.dumps(state, indent=2)+'\n')
    p.replace(root/'group-status.json')


save()
try:
    with (root/'campaign.log').open('x') as log:
        child = subprocess.Popen(['bash', 'pkg/run.sh'], cwd=root,
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        state.update(status='RUNNING', shell_pid=child.pid)
        save()
        rc = child.wait()
    cells = {}
    for name in ('diagnostic-current',):
        p = root/'results'/name/'status.json'
        cells[name] = json.loads(p.read_text()) if p.exists() else dict(status='UNRUN')
    state.update(returncode=rc, cells=cells, status='COMPLETE' if rc == 0
        and all(c['status'] == 'COMPLETE' for c in cells.values()) else 'STOPPED')
except BaseException as exc:
    state.update(status='STOPPED', error=repr(exc))
    raise
finally:
    state['finished'] = time.time()
    save()
