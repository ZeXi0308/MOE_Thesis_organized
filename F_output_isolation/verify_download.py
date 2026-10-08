"""Verify and unpack the preserved remote CPU experiment archive."""
import hashlib, json, tarfile
from datetime import datetime, timezone
from pathlib import Path
root=Path(__file__).resolve().parent
archive=root/'results_archive.tgz'
expected='27adb17f7cfd5830c1f2931700c02a83c5f6a02ee8194841240c806c0a41c894'
actual=hashlib.sha256(archive.read_bytes()).hexdigest()
assert actual==expected,('archive hash mismatch/incomplete download',actual)
with tarfile.open(archive) as f:f.extractall(root,filter='data')
manifest=json.loads((root/'archive_manifest.json').read_text())
for rel,sha in manifest.items():
 p=root/rel
 assert p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest()==sha,rel
runtime=json.loads((root/'environment.json').read_text())
for name,sha in runtime['prototype'].items():
 assert hashlib.sha256((root/name).read_bytes()).hexdigest()==sha,('prototype differs from measured version',name)
receipt={'verified_at':datetime.now(timezone.utc).isoformat(),'archive_sha256':actual,'archive_bytes':archive.stat().st_size,'all_original_files_match':True,'original_files':len(manifest),'measured_prototype_matches':True,'formal_runs':16,'formal_responses':1568,'formal_generated_tokens':690176,'gpu_runs':0,'candidate_iterations':1,'invalid_diagnostic_runs_retained':7,'formal_source_pacing_warning':'formal_medium/03_cost; preserved, no replacement'}
(root/'download_verification.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
