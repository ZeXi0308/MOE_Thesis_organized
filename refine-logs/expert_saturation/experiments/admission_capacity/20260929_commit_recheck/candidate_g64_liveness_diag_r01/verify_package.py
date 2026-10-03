"""Verify every executable or input byte in the isolated G64 admission diagnostic package."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
manifest = json.loads((root / 'manifest.json').read_text())
pkg = root / 'pkg'
entries = list(pkg.rglob('*'))
if any(path.is_symlink() for path in entries):
    raise RuntimeError('Symlink in candidate pkg')
actual = {str(path.relative_to(root)) for path in entries if path.is_file()}
if set(manifest) != actual | {'verify_package.py'}:
    raise RuntimeError('Candidate manifest does not cover the exact pkg file set')
if len(manifest) != 29:
    raise RuntimeError('Unexpected candidate payload count')
for name, expected in manifest.items():
    path = root / name
    if (not (name.startswith('pkg/') or name == 'verify_package.py')
            or '..' in Path(name).parts or not path.is_file() or path.is_symlink()):
        raise RuntimeError(f'Invalid candidate path: {name}')
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise RuntimeError(f'Candidate payload drift: {name}')
print(f'G64 admission diagnostic payload verified: {len(manifest)} files')
