"""Transfer and verify the frozen package; no engine import, GPU probe or launch."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess
import time

base = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('funding_driver', base/'execute.py')
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)
control = '/private/tmp/moe-funding-01a0953f.sock'
driver.SSH[driver.SSH.index('-S')+1] = control
driver.SCP = [('ControlPath='+control) if s.startswith('ControlPath=') else s for s in driver.SCP]
metadata = json.loads((base/'preparation/preparation.json').read_text())
digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
archive = base/'preparation/execution.tar.gz'
assert digest(archive) == metadata['archive_sha256']
assert all(digest(base/'preparation/pkg'/n) == h for n,h in metadata['files_sha256'].items())
out = base/'execution'
out.mkdir(exist_ok=False)
state = dict(status='STAGING', archive_sha256=metadata['archive_sha256'], remote_dir=driver.REMOTE,
             host=driver.HOST, started_unix_s=time.time(), driver_sha256=digest(base/'execute.py'),
             staging_only=True, gpu_execution='UNRUN', existing_queue='Qwen r03 lifecycle first')
try:
    subprocess.run(driver.SSH+[shlex.join(['mkdir',driver.REMOTE])],check=True)
    subprocess.run(driver.SCP+[str(archive),driver.HOST+':'+driver.REMOTE+'/execution.tar.gz'],check=True)
    code = f'''import hashlib,json,pathlib,tarfile
r=pathlib.Path({driver.REMOTE!r}); a=r/'execution.tar.gz'
assert hashlib.sha256(a.read_bytes()).hexdigest()=={metadata['archive_sha256']!r}
assert not (r/'pkg').exists() and not (r/'launch-once').exists() and not (r/'results').exists()
with tarfile.open(a) as t:t.extractall(r,filter='data')
expected={metadata['files_sha256']!r}
actual={{n:hashlib.sha256((r/'pkg'/n).read_bytes()).hexdigest() for n in expected}}
assert actual==expected
print(json.dumps(dict(files_verified=len(actual),archive_sha256=hashlib.sha256(a.read_bytes()).hexdigest(),gpu_execution='UNRUN')))
'''
    receipt = driver.query(code)
    state.update(status='STAGED',staging_receipt=receipt)
except BaseException as exc:
    state.update(status='STAGING_INCOMPLETE',error=f'{type(exc).__name__}: {exc}')
    raise
finally:
    state['updated_unix_s'] = time.time()
    with (out/'execution.json').open('x') as f:json.dump(state,f,indent=2);f.write('\n')
    print(json.dumps(state),flush=True)
