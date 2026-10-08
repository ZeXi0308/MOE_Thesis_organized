"""Verify the isolated H1 candidate payload without importing vLLM or CUDA."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
manifest = json.loads((root / 'manifest.json').read_text())
pkg = root / 'pkg'
actual = list(pkg.rglob('*'))
if any(path.is_symlink() for path in actual):
    raise RuntimeError('Symlink in candidate pkg')
actual_files = {str(path.relative_to(root)) for path in actual if path.is_file()}
if set(manifest) != actual_files | {'verify_package.py'}:
    raise RuntimeError('Candidate manifest does not cover the exact pkg file set')
for name, expected in manifest.items():
    path = root / name
    if (not (name.startswith('pkg/') or name == 'verify_package.py')
            or '..' in Path(name).parts
            or not path.is_file() or path.is_symlink()):
        raise RuntimeError(f'Invalid candidate path: {name}')
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise RuntimeError(f'Candidate payload drift: {name}')
print(f'Candidate payload verified: {len(manifest)} files')
