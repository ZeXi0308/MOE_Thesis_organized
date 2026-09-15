"""Read back the completed original group after SSH loss; never launch jobs."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import tarfile

path=Path(__file__).with_name('execute.py')
spec=importlib.util.spec_from_file_location('context_original_driver',path)
driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)
driver.SSH[driver.SSH.index('-S')+1]='/private/tmp/moe-context-readback.sock'
driver.SCP=[('ControlPath=/private/tmp/moe-context-readback.sock') if s.startswith('ControlPath=') else s for s in driver.SCP]
receipt=driver.query(f'''import hashlib,json,pathlib,tarfile
r=pathlib.Path({driver.REMOTE!r});s=json.loads((r/'group-status.json').read_text())
assert s['status']=='COMPLETE' and s['returncode']==0 and all(c['terminal']['status']=='COMPLETE' for c in s['cells'])
assert not pathlib.Path('/proc/'+str(s['pid'])).exists() and not pathlib.Path('/proc/'+str(s['shell_pid'])).exists()
p=r/'readback.tar.gz'
if not p.exists():
 with tarfile.open(p,'w:gz') as t:
  for n in ['pkg','results','campaign.log','group-status.json','execute_group.py']:t.add(r/n,arcname=n)
print(json.dumps(dict(status='COMPLETE',sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size,group=s)))''')
out=path.parent/'execution';archive=out/'readback.tar.gz'
with (out/'remote_completion.json').open('x') as f:json.dump(receipt,f,indent=2)
assert not archive.exists() and not (out/'readback').exists()
subprocess.run(driver.SCP+[driver.HOST+':'+driver.REMOTE+'/readback.tar.gz',str(archive)],check=True)
assert hashlib.sha256(archive.read_bytes()).hexdigest()==receipt['sha256']
with tarfile.open(archive) as t:t.extractall(out/'readback',filter='data')
with (out/'recovery.json').open('x') as f:
 json.dump(dict(**receipt,scope='Original complete group recovered after SSH loss; no cell restarted, prior UNKNOWN_REMOTE receipt preserved'),f,indent=2)
print(json.dumps(dict(status='COMPLETE_READBACK',bytes=receipt['bytes'],sha256=receipt['sha256'])),flush=True)
