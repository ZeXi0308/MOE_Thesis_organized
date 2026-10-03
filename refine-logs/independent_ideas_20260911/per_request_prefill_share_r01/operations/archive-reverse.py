from pathlib import Path
import hashlib,json,tarfile
block='reverse'
root=Path('/root/autodl-tmp/moe-prefill-share-01a07d4b-20260911-r01')
meta=json.loads((root/f'results/{block}-process.json').read_text())
assert meta['exit_code'] is not None, 'engine still live'
selected=set(p for p in (root/'results'/block).rglob('*') if p.is_file())
selected.update(p for p in (root/'results').glob(f'{block}-*') if p.is_file())
selected.update(p for p in root.glob(f'{block}-supervisor.*') if p.is_file())
selected.update([root/'stage-preflight.json',root/'execution.tar.gz'])
selected.update(root.glob('model-cache-verification.json'))
archive=root.parent/f'moe-prefill-share-01a07d4b-20260911-{block}.tar.gz'
with tarfile.open(archive,'x:gz') as t:
 for p in sorted(selected):t.add(p,arcname=str(p.relative_to(root)),recursive=False)
record={'block':block,'archive':str(archive),'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'bytes':archive.stat().st_size,'files':len(selected),'process':meta,'status':json.loads((root/f'results/{block}/status.json').read_text())}
(root/f'{block}-archive.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
