#!/usr/bin/env python3
"""Restore and verify losslessly compressed research JSON, preserving existing files."""
import gzip
import hashlib
import json
from pathlib import Path


def sha256(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def main():
    script = Path(__file__).resolve()
    root = script.parents[5]
    manifest = json.loads(script.with_name('GITHUB_LARGE_DATA_20261004.json').read_text())
    for entry in manifest['files']:
        target = root / entry['path']
        compressed = root / entry['gzip_path']
        for path in (target, compressed):
            if not path.resolve().is_relative_to(root):
                raise ValueError('Manifest path escapes repository')
        created = False
        try:
            if not target.exists():
                with target.open('xb') as output:
                    created = True
                    with gzip.open(compressed, 'rb') as source:
                        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b''):
                            output.write(chunk)
            if target.stat().st_size != entry['bytes'] or sha256(target) != entry['sha256']:
                raise RuntimeError('Original hash mismatch: ' + entry['path'])
        except BaseException:
            if created:
                target.unlink(missing_ok=True)
            raise
        print('Verified:', entry['path'])
    print('Verified', len(manifest['files']), 'original research files')


if __name__ == '__main__':
    main()
