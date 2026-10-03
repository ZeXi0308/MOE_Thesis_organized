#!/usr/bin/env python3
"""Authorized backup fallback; identical guard policy and per-arm private warm caches."""
import json
from pathlib import Path
import shutil
import signal
import sys
import time
import run_native_current_guard_triplet_r01 as prior

group,base=prior.group,prior.base
group.BASE='/root/moe-a-native-current-guard-backup-stage-r01-20261002'
group.SESSION='/root/moe-a-native-current-guard-backup-session-r01-20261002'
group.SOURCE_CACHE=Path('/root/moe-a-native-backfill-lazy-perf-session-r01-20261002/runtime-cache-first')
group.OUTPUTS=tuple('/root/moe-a-native-current-guard-backup-output-'+label+'-r01-20261002' for label in group.LABELS)
DISK_RESERVE_BYTES=1536*1024**2

def prepare_runtime_cache(session_dir):
    group.original_prepare_cache(session_dir)
    base.require(session_dir==Path(group.SESSION),'unexpected current-guard retry session')
    source=group.SOURCE_CACHE
    base.require(source.is_dir() and not source.is_symlink(),'completed seed cache missing')
    receipt=json.loads((source.parent/'receipt.json').read_text())
    base.require(receipt.get('status')=='CELLS_COMPLETE' and
                 tuple(c.get('arm') for c in receipt.get('cells',[]))==('native_full_ordinary_only','native_full_reference'),
                 'source cache receipt differs')
    source_files=group.cache_inventory(source)
    base.require(bool(source_files),'empty source cache')
    source_bytes=sum(size for size,_ in source_files.values())
    base.require(shutil.disk_usage('/root').free>=DISK_RESERVE_BYTES+3*source_bytes,
                 'need 1.5 GiB free after three private warm-cache copies')
    for label in group.LABELS:
        destination=session_dir/f'runtime-cache-{label}'
        base.require(not destination.exists(),'private cache already exists')
        shutil.copytree(source,destination,copy_function=shutil.copy2)
        base.require(group.cache_inventory(destination)==source_files,'cache metadata differs')
    group._seed_files=source_files
    base.write_json(session_dir/'runtime-cache-seed-receipt.json',dict(
        status='THREE_PRIVATE_COPIES_PREPARED_UNDER_LOCK',source=str(source),
        source_files=len(source_files),source_bytes=source_bytes,
        destinations=[str(session_dir/f'runtime-cache-{label}') for label in group.LABELS],
        copied_unix_s=time.time(),disk_reserve_after_copies_bytes=DISK_RESERVE_BYTES,
        reserve_basis='Three preceding complete triplets each produced under 0.38 GB total raw output plus retained archive; reserve remains over four times that measured total.',
        identity_check='relative path, byte size and preserved mtime_ns'))

base.prepare_runtime_cache=prepare_runtime_cache

if __name__=='__main__':
    signal.signal(signal.SIGTERM,base.stop_requested)
    try:raise SystemExit(base.main())
    except (OSError,ValueError,base.subprocess.SubprocessError) as error:
        print('ABORT_NATIVE_CURRENT_GUARD_BACKUP:',error,file=sys.stderr)
        raise SystemExit(75)
