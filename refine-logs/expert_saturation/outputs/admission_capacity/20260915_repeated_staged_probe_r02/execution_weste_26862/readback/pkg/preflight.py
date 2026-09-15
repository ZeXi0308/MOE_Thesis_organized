import hashlib, importlib.util, json
from pathlib import Path
root=Path(importlib.util.find_spec('vllm').origin).parent
expected=json.loads(Path(__file__).with_name('runtime_source_hashes.json').read_text())
for name,sha in expected.items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest()==sha, name
print('Installed source hashes match; no GPU initialized')
