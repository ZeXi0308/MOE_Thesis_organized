from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import tarfile

root = Path('/root/autodl-tmp/moe-prefill-share-01a07d4b-20260911-r01')
uploaded = root.parent / 'moe-prefill-share-01a07d4b-20260911-execution.tar.gz'
expected = '27c7c26b7709ee5376b84447dfc01481f22401014f7367068d411c171983e799'
assert hashlib.sha256(uploaded.read_bytes()).hexdigest() == expected
root.mkdir()
with tarfile.open(uploaded) as archive:
    members = archive.getmembers()
    assert len(members) == 16
    for member in members:
        assert member.isfile() and not member.name.startswith('/') and '..' not in Path(member.name).parts
        target = root / member.name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as handle:
            handle.write(archive.extractfile(member).read())
shutil.copyfile(uploaded, root / 'execution.tar.gz')
checks = []
for block in ('forward', 'reverse'):
    command = ['/root/autodl-tmp/expert-saturation/vllm-0.26/bin/python',
        str(root/'run_prefill_share.py'), '--block', block, '--prepare-only',
        '--output-dir', str(root/'prepared-plans'/block)]
    result = subprocess.run(command, capture_output=True, text=True)
    checks.append(dict(block=block, command=command, exit_code=result.returncode,
        stdout=result.stdout, stderr=result.stderr))
record = dict(root=str(root), sha256=expected, bytes=uploaded.stat().st_size,
    files=len(members), prepare_only=checks, gpu_executions=0)
(root/'stage-preflight.json').write_text(json.dumps(record, indent=2)+'\n')
print(json.dumps(record, indent=2))
assert all(c['exit_code'] == 0 for c in checks)
