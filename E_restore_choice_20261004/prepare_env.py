"""Private RECORD-only torch view; omit stale files from an older installation.

No package implementation is modified and shared site-packages remain read-only.
"""
import importlib.metadata
import json
from pathlib import Path

root = Path(__file__).resolve().parent
overlay = root / 'package_view'
dist = importlib.metadata.distribution('torch')
included = set()
for entry in dist.files:
    relative = Path(str(entry))
    if not relative.parts or relative.parts[0] != 'torch' or '..' in relative.parts:
        continue
    source = Path(dist.locate_file(entry)).resolve()
    if not source.is_file():
        raise RuntimeError(f'installed RECORD path missing: {entry}')
    target = overlay / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink():
        assert target.resolve() == source
    else:
        assert not target.exists()
        target.symlink_to(source)
    included.add(str(relative))
source_root = Path(dist.locate_file('torch'))
omitted = sorted(str(p.relative_to(source_root.parent)) for p in source_root.rglob('*.py')
                 if str(p.relative_to(source_root.parent)) not in included)
(root / 'environment_repair.json').write_text(json.dumps(dict(
    torch_version=dist.version, source=str(source_root), private_view=str(overlay),
    recorded_files=len(included), omitted_unrecorded_python_files=omitted,
    shared_files_modified=False), indent=2) + '\n')
print(json.dumps(dict(recorded_files=len(included), omitted_python_files=len(omitted))))
