#!/usr/bin/env python3
"""Merge unchanged private-cache copies of one completed A block under its lock."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

LOCK = Path('/root/autodl-tmp/moe-research-gpu.lock')
SESSION = Path('/root/moe-a-native-residency-ablation-primary-session-r01-20261002')
SEED = Path('/root/moe-a-native-backfill-only-primary-session-r01-20261001/runtime-cache-first')
OUT = Path('/root/moe-a-completed-residency-cache-dedup-r01-20261002.json')

def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def main():
    assert not OUT.exists()
    with LOCK.open('r+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        info = os.fstat(lock.fileno())
        assert f'{info.st_dev}:{info.st_ino}' == '2304:29005388732'
        r = json.loads((SESSION / 'receipt.json').read_text())
        assert r['status'] == 'CELLS_COMPLETE'
        assert all(c['archive_status'] == 'VERIFIED' for c in r['cells'])
        assert json.loads((SEED.parent / 'receipt.json').read_text())['status'] == 'CELLS_COMPLETE'
        gpu = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], check=True, capture_output=True, text=True)
        assert not gpu.stdout.strip(), gpu.stdout
        before = shutil.disk_usage('/root').free
        rows = []
        for label in ('first', 'second', 'third'):
            directory = SESSION / ('runtime-cache-' + label)
            assert directory.is_dir() and not directory.is_symlink()
            for duplicate in directory.rglob('*'):
                if not duplicate.is_file() or duplicate.is_symlink():
                    continue
                original = SEED / duplicate.relative_to(directory)
                if not original.is_file() or original.is_symlink():
                    continue
                old, new = original.stat(), duplicate.stat()
                if (old.st_dev != new.st_dev or old.st_ino == new.st_ino or
                    old.st_size != new.st_size or old.st_mtime_ns != new.st_mtime_ns):
                    continue
                digest = sha(original)
                if sha(duplicate) != digest:
                    continue
                temp = duplicate.with_name(duplicate.name + '.a-dedup-temp')
                assert not temp.exists()
                os.link(original, temp)
                os.replace(temp, duplicate)
                rows.append(dict(path=str(duplicate), source=str(original), bytes=old.st_size, sha256=digest))
        receipt = dict(status='COMPLETE', completed_unix_s=time.time(),
                       free_before=before, free_after=shutil.disk_usage('/root').free,
                       note='Only completed immutable A cache copies; same bytes, size, mtime and filesystem; all paths retained. Never rerun completed controllers.',
                       files=rows)
        OUT.write_text(json.dumps(receipt, indent=2) + '\n')
        print(json.dumps({k: v for k, v in receipt.items() if k != 'files'}), 'files', len(rows), flush=True)

if __name__ == '__main__':
    main()
