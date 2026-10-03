#!/usr/bin/env python3
"""Fresh confirmation block1: native tail, BidKV break/continue, current guard."""
from pathlib import Path
import signal
import sys
import run_native_current_guard_triplet_r02 as inherited

group,base=inherited.group,inherited.base
guard=inherited.prior
base.PACKAGE_NAME='candidate_native_current_guard_fresh_r01'
base.PACKAGE_SHA256='1dc54ab2bf41522a56f53cc96ab697f25dce24f9742b40196f201532ac616dc2'
group.PACKAGE_SHA256=base.PACKAGE_SHA256
group.LABELS=('first','second','third','fourth')
group.ARMS=('tail_break','bidkv_break','bidkv_continue','bidkv_current_guard')
group.GATES=('ordinary',)*4
group.VARIANTS=('native_full_ordinary_only',)*4
guard.prior.RULES=('tail','bidkv_score','bidkv_score','bidkv_score')
guard.inherited.MODES=('off','off','on','off')
guard.GUARDS=('off','off','off','on')
group.BASE='/root/moe-a-native-current-guard-fresh-b1-stage-r01-20261002'
group.SESSION='/root/moe-a-native-current-guard-fresh-b1-session-r01-20261002'
group.OUTPUTS=tuple('/root/moe-a-native-current-guard-fresh-b1-output-'+label+'-r01-20261002' for label in group.LABELS)

def validate(plan):
    base.require(plan.get('session_dir')==group.SESSION and
                 plan.get('approved_total_wall_seconds')==4800 and
                 plan.get('warm_runtime_cache_source')==str(group.SOURCE_CACHE),
                 'four-arm confirmation session/cache/wall differs')
    cells=plan.get('cells',[])
    base.require(tuple(c.get('arm') for c in cells)==group.ARMS and len(cells)==4,
                 'four-arm confirmation order differs')
    for i,cell in enumerate(cells):
        base.require(cell.get('package_dir')==group.BASE+'/'+group.LABELS[i]+'/'+base.PACKAGE_NAME and
                     cell.get('output_dir')==group.OUTPUTS[i] and cell.get('max_wall_seconds')==900,
                     'confirmation cell differs')
        base.FROZEN_ARMS=(group.ARMS[i],)
        group.original_validate(dict(plan,cells=[cell]))
    base.FROZEN_ARMS=group.ARMS
    base.require(plan['approved_total_wall_seconds']>=sum(c['max_wall_seconds'] for c in cells)+1055,
                 'insufficient full-group timeout')
    return plan

base.validate_plan=validate

def prepare_runtime_cache(session_dir):
    group.original_prepare_cache(session_dir)
    base.require(session_dir==Path(group.SESSION),'unexpected current-guard retry session')
    source=group.SOURCE_CACHE
    base.require(source.is_dir() and not source.is_symlink(),'completed seed cache missing')
    receipt=inherited.json.loads((source.parent/'receipt.json').read_text())
    base.require(receipt.get('status')=='CELLS_COMPLETE' and
                 tuple(c.get('arm') for c in receipt.get('cells',[]))==('native_full_reference','native_full_ordinary_only'),
                 'source cache receipt differs')
    source_files=group.cache_inventory(source)
    base.require(bool(source_files),'empty source cache')
    source_bytes=sum(size for size,_ in source_files.values())
    base.require(inherited.shutil.disk_usage('/root').free>=inherited.DISK_RESERVE_BYTES+len(group.LABELS)*source_bytes,
                 'need 1.5 GiB free after four private warm-cache copies')
    for label in group.LABELS:
        destination=session_dir/f'runtime-cache-{label}'
        base.require(not destination.exists(),'private cache already exists')
        inherited.shutil.copytree(source,destination,copy_function=inherited.shutil.copy2)
        base.require(group.cache_inventory(destination)==source_files,'cache metadata differs')
    group._seed_files=source_files
    base.write_json(session_dir/'runtime-cache-seed-receipt.json',dict(
        status='FOUR_PRIVATE_COPIES_PREPARED_UNDER_LOCK',source=str(source),
        source_files=len(source_files),source_bytes=source_bytes,
        destinations=[str(session_dir/f'runtime-cache-{label}') for label in group.LABELS],
        copied_unix_s=inherited.time.time(),disk_reserve_after_copies_bytes=inherited.DISK_RESERVE_BYTES,
        reserve_basis='Previous complete triplets were below0.38GB; four arms budget below0.51GB by per-cell scaling, with1.5GiB reserve after four independentcachecopies.',
        identity_check='relative path, byte size and preserved mtime_ns'))


base.prepare_runtime_cache=prepare_runtime_cache

if __name__=='__main__':
    signal.signal(signal.SIGTERM,base.stop_requested)
    try:raise SystemExit(base.main())
    except (OSError,ValueError,base.subprocess.SubprocessError) as error:
        print('ABORT_NATIVE_CURRENT_GUARD_FRESH_B1:',error,file=sys.stderr)
        raise SystemExit(75)
