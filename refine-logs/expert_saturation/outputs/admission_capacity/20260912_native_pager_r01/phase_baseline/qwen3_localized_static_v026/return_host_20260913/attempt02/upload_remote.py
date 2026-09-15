import hashlib,io,json,sys,tarfile
from pathlib import Path
expected='7f54419f53dccf9a4fab0abecb04d4dc8d225199b3aeef42ee4fdd436ead2633'
payload=sys.stdin.buffer.read();assert hashlib.sha256(payload).hexdigest()==expected
p=Path('/root/autodl-tmp/qwen3-localized-static-v026-launch-r02');p.mkdir(exist_ok=False)
with tarfile.open(fileobj=io.BytesIO(payload),mode='r:gz') as a:
 for m in a.getmembers():
  assert m.isfile() and not m.name.startswith('/') and '..' not in Path(m.name).parts
  f=p/m.name;f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes(a.extractfile(m).read())
r=dict(status='UPLOADED_NOT_STARTED',sha256=expected,bytes=len(payload),stage=str(p))
(p/'upload.json').write_text(json.dumps(r)+'\n');print(json.dumps(r))
