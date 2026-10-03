from pathlib import Path
import hashlib,json,tarfile
block='reverse'
root=Path('/root/autodl-tmp/moe-waiting-bypass-01a07d4b-20260910-r01')
meta=json.loads((root/f'results/{block}-process.json').read_text())
assert meta['exit_code'] is not None, 'engine still live'
selected=set(p for p in (root/'results'/block).rglob('*') if p.is_file())
selected.update(p for p in (root/'results').glob(f'{block}-*') if p.is_file())
selected.update(p for p in root.glob(f'{block}-supervisor.*') if p.is_file())
selected.update([root/'stage-preflight.json',root/'execution.tar.gz',root/'execution-initial.tar.gz',root/'execution-update.json'])
selected.update([root/'execution-before-independent-cache.tar.gz',root/'execution-cache-update.json'])
selected.update(root.glob('model-cache-verification.json'))
archive=root.parent/f'moe-waiting-bypass-01a07d4b-{block}.tar.gz'
with tarfile.open(archive,'x:gz') as t:
 for p in sorted(selected):t.add(p,arcname=str(p.relative_to(root)),recursive=False)
record={'block':block,'archive':str(archive),'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'bytes':archive.stat().st_size,'files':len(selected),'process':meta,'status':json.loads((root/f'results/{block}/status.json').read_text())}
(root/f'{block}-archive.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
