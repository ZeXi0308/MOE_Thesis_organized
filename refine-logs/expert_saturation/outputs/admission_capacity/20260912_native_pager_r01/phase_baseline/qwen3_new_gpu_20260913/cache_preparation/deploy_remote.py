import hashlib,io,json,subprocess,sys,tarfile
from pathlib import Path
payload=sys.stdin.buffer.read()
expected='ad9aa121038986167882e2b79ed87979815582ae4d674a82be11b7a1477816a9'
assert hashlib.sha256(payload).hexdigest()==expected
assert b'/root/autodl-tmp/qwen3-new-gpu-localized-static-v026-launch-r02/run_native_pager.py' in Path('/proc/5127/cmdline').read_bytes().split(b'\0')
stage=Path('/root/qwen3-cache-collector-launch-r01');stage.mkdir(exist_ok=False)
assert not Path('/root/qwen3-compressed-cache-ad44e777').exists()
with tarfile.open(fileobj=io.BytesIO(payload),mode='r:gz') as a:
 for m in a.getmembers():
  assert m.isfile() and m.name in ['collect_shards.py','run_remote.py','protocol.json']
  (stage/m.name).write_bytes(a.extractfile(m).read())
receipts=stage/'detached_receipts';receipts.mkdir()
cmd=['/root/miniconda3/bin/python','/root/autodl-tmp/qwen3-new-gpu-localized-static-v026-launch-r02/detached_launch.py','--package-dir',str(stage),'--python','/root/miniconda3/bin/python','--entry-sha256','32af5523df808a4820e7a6a59968e6ad55b1e0cf424976b3cf92474d8d985f47','--protocol-sha256','5dd2e23eeceb1e921fe11a77a19b4a90df4299d07f9350d1755bd7fb82504ccf','--receipt-root',str(receipts)]
r=subprocess.run(cmd,capture_output=True,text=True)
print(json.dumps(dict(bundle_sha256=expected,stage=str(stage),exit_code=r.returncode,stdout=r.stdout,stderr=r.stderr)))
raise SystemExit(r.returncode)
