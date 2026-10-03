from pathlib import Path
import datetime
import hashlib
import json
import subprocess

root=Path('/root/autodl-tmp/moe-prefill-share-01a07d4b-20260911-r01')
archive=root.parent/'moe-prefill-share-01a07d4b-20260911-execution.tar.gz'
result=dict(time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    root_exists=root.exists(), upload_exists=archive.exists(),
    upload_sha256=hashlib.sha256(archive.read_bytes()).hexdigest() if archive.exists() else None,
    stage=json.loads((root/'stage-preflight.json').read_text()) if (root/'stage-preflight.json').exists() else None,
    gpu_processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True))
print(json.dumps(result,indent=2))
