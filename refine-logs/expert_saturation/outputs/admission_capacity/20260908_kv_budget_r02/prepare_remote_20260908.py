"""Stage the retained r02 bundle once; uses the caller's SSH authentication."""
import os,json,pathlib,shlex,subprocess,hashlib
root=pathlib.Path(__file__).resolve().parent
remote='/root/autodl-tmp/moe-kv-budget-20260908-r02'
python='/root/autodl-tmp/expert-saturation/vllm-0.26/bin/python'
host='root@connect.westd.seetacloud.com'
ssh=['ssh','-o','ConnectTimeout=15','-o','NumberOfPasswordPrompts=1','-o','StrictHostKeyChecking=yes','-p','37116',host]
def call(code):return json.loads(subprocess.check_output([*ssh,shlex.join([python,'-c',code])],text=True,timeout=45))
info=call(f'''import json,pathlib,subprocess
root=pathlib.Path({remote!r}); processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True).strip()
assert not processes, processes
root.mkdir(exist_ok=False)
print(json.dumps(dict(remote_root=str(root),compute_processes=processes)))''')
archive=root/'execution.tar.gz'
subprocess.run(['scp','-q','-o','ConnectTimeout=15','-o','NumberOfPasswordPrompts=1','-o','StrictHostKeyChecking=yes','-P','37116',str(archive),f'{host}:{remote}/execution.tar.gz'],check=True,timeout=45)
sha=hashlib.sha256(archive.read_bytes()).hexdigest()
checked=call(f'''import json,pathlib,hashlib,tarfile
root=pathlib.Path({remote!r}); archive=root/'execution.tar.gz'
sha=hashlib.sha256(archive.read_bytes()).hexdigest(); assert sha=={sha!r}
with tarfile.open(archive) as tar:
 for member in tar.getmembers():
  assert member.isfile() and (root/member.name).resolve().is_relative_to(root.resolve())
  assert not (root/member.name).exists()
 tar.extractall(root)
print(json.dumps(dict(sha256=sha,bytes=archive.stat().st_size,files=len(list(root.rglob('*'))))))''')
record=dict(**info,archive_verification=checked)
(root/'UPLOAD_20260908.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
