#!/usr/bin/env python3
"""Check downloaded terminal-run files against their remote SHA-256 values."""
import hashlib
import json
from pathlib import Path
import time

root = Path(__file__).resolve().parents[1] / 'runs'
checked = []
for manifest in sorted(root.glob('*/SHA256.json')):
    entries = json.loads(manifest.read_text())
    total = 0
    for rel, expected in entries.items():
        path = manifest.parent / rel
        h = hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
                h.update(block)
                total += len(block)
        if h.hexdigest() != expected:
            raise RuntimeError(f'Checksum differs: {path}')
    status = json.loads((manifest.parent / 'status.json').read_text())
    if status['state'] not in ('COMPLETE', 'FAILED'):
        raise RuntimeError(f'Nonterminal run manifest: {manifest}')
    checked.append({'run': manifest.parent.name, 'state': status['state'],
                    'files': len(entries), 'bytes': total})
receipt = {'verified_utc_s': time.time(), 'runs': checked,
           'total_files': sum(r['files'] for r in checked),
           'total_bytes': sum(r['bytes'] for r in checked)}
(root / 'local_backup_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt, indent=2))
