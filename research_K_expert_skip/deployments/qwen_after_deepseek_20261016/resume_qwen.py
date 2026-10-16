#!/usr/bin/env python3
"""K-owned Qwen download: complete C shards are read-only links; prefixes are copies.

No DeepSeek cleanup, signals to C, GPU calls, reflinks, or hardlinks. --plan-only
uses current free space; --execute requires that space already be available.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import threading
import time

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
PRIOR = HERE.parent / 'qwen_partial_resume_20261016'
PRIOR_SHA = '1ffa0f610c2531ecd1fd5c5f88067bcc846759cffbec344e71070c4b2c39d010'
GIB = 2**30
C_APPEND_FLOOR = 17424291144
ROOTFS_NAMES = {'model-00009-of-00016.safetensors', 'model-00016-of-00016.safetensors'}


def support():
    path = PRIOR / 'resume_qwen.py'
    if hashlib.sha256(path.read_bytes()).hexdigest() != PRIOR_SHA:
        raise ValueError('Pinned prior read-only parsing helpers changed')
    spec = importlib.util.spec_from_file_location('qwen_source_support', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def idle(root):
    """CPU process checks only; the owner separately approves any DS RAM release."""
    boot = next(int(line.split()[1]) for line in Path('/proc/stat').read_text().splitlines() if line.startswith('btime '))
    ticks = os.sysconf('SC_CLK_TCK')
    for path in (root/'runs').glob('*/status.json'):
        row = json.loads(path.read_text())
        if row.get('state') in ('PREPARING', 'WAITING', 'RUNNING'):
            raise RuntimeError(f'K run still active: {path.parent.name}')
        for key in ('controller_pid', 'worker_pid'):
            if not row.get(key):
                continue
            try:
                fields = Path(f"/proc/{int(row[key])}/stat").read_text().rsplit(')', 1)[1].split()
            except FileNotFoundError:
                continue
            if boot+int(fields[19])/ticks <= row.get('ended_unix', float('inf'))+1:
                raise RuntimeError(f'K process has not exited: {path.parent.name} {key}')


def payload_paths(row):
    directory = Path(row['storage_directory'])
    return directory/row['path'], directory/'.modelscope-partials'/(row['sha256']+'.partial')


def budget(plan, helpers):
    """Count every remaining private payload byte, including copied C prefixes."""
    out = {}
    for key, store in plan['filesystems'].items():
        remaining = 0
        for row in plan['files']:
            if row['source_complete'] or row['storage'] != key:
                continue
            target, partial = payload_paths(row)
            present = target if target.exists() else partial
            size = present.stat().st_size if present.exists() else 0
            if not 0 <= size <= row['size']:
                raise ValueError(f'Private payload exceeds pinned size: {present}')
            remaining += row['size']-size
        if key == 'persistent':
            remaining += sum(row['size'] for row in plan['small_assets']
                             if not (Path(plan['destination'])/row['path']).exists())
        actual = helpers.filesystem_state(store['directory'])
        if actual['device_id'] != store['device_id']:
            raise RuntimeError('Storage filesystem identity changed')
        after = actual['free_bytes']-remaining-store['c_reserved_bytes']-store['other_reserved_bytes']-store['overhead_bytes']
        out[key] = {**actual, 'remaining_private_bytes': remaining,
                    'predicted_free_after_reserved_bytes': after,
                    'margin_above_reserve_bytes': after-store['reserve_bytes'],
                    'space_ok': after >= store['reserve_bytes']}
    return out


def make_plan(args, helpers, base):
    root, source, destination = args.root.resolve(), args.source.resolve(), args.destination.resolve()
    if not destination.is_relative_to(root/'models') or destination == root/'models':
        raise ValueError('Destination must be inside K/models')
    mapping = json.loads((PRIOR/'pinned_files.json').read_text())
    if helpers.digest(PRIOR/'pinned_files.json') != base.MAPPING_SHA256:
        raise ValueError('Fixed official mapping changed')
    assets = base.assets(root, source, mapping, helpers.digest)
    index = next(row for row in assets if row['path'] == 'model.safetensors.index.json')
    weight_map = json.loads(Path(index['source']).read_text())['weight_map']
    stores = {
        'persistent': {'directory': str(destination), 'reserve_bytes': 20*GIB, 'other_reserved_bytes': 2*GIB},
        'rootfs': {'directory': '/tmp/research_K_expert_skip/qwen3_bf16_ad44e777', 'reserve_bytes': 10*GIB, 'other_reserved_bytes': 0},
        'shm': {'directory': '/dev/shm/research_K_expert_skip/qwen3_bf16_ad44e777', 'reserve_bytes': 20*GIB, 'other_reserved_bytes': 0}}
    for key, store in stores.items():
        if Path(store['directory']).resolve() != Path(store['directory']):
            raise ValueError('Storage directory must not resolve through a symlink')
        store.update(helpers.filesystem_state(store['directory']), overhead_bytes=256*2**20, c_reserved_bytes=0)
        if key == 'shm' and (store.get('filesystem_type'), store.get('mount_point')) != ('tmpfs', '/dev/shm'):
            raise ValueError('RAM payloads require the /dev/shm tmpfs mount')
    if len({s['device_id'] for s in stores.values()}) != 3 or stores['persistent']['device_id'] != source.stat().st_dev:
        raise ValueError('Expected independent persistent/rootfs/shm budgets and C on persistent filesystem')
    process, contract, source_stats, rows, c_remaining = None, None, {}, [], 0
    candidates = list((source/'.cache/huggingface/download').glob('*.incomplete'))
    for original in mapping['files']:
        row = {**original, 'lfs_sha256': original['sha256']}
        complete = source/row['path']
        row['source_complete'] = complete.exists()
        row['source_prefix'] = None
        if row['source_complete']:
            before = base.fingerprint(complete)
            if before['size'] != row['size'] or (args.execute and helpers.digest(complete) != row['sha256']) or base.fingerprint(complete) != before:
                raise ValueError(f'Complete C shard failed pinned identity: {complete}')
            source_stats[str(complete)] = before
            row.update(storage='source_link', source_complete_path=str(complete),
                       source_sha_verified=bool(args.execute), new_private_bytes=0,
                       private_existing_bytes=0, source_prefix_bytes=0)
            expected_link = complete
        else:
            selected = []
            for path in candidates:
                if row['sha256'] in path.name.split('.'):
                    selected.append({'path': str(path), 'stat': base.prefix_layout(path, row, weight_map)})
            prefix = max(selected, key=lambda x:x['stat']['size']) if selected else None
            if prefix:
                if process is None:
                    process = base.source_process(args.source_pid, source)
                    observed = json.loads((PRIOR/'source_inventory.json').read_text())
                    if process['pid'] != observed['source_pid'] or process['boot_id'] != observed['boot_id'] or process['start_ticks'] != int(observed['source_start_ticks']):
                        raise RuntimeError('C prefix writer identity differs from verified inventory')
                    contract = base.http_contract(helpers.digest)
                if prefix['path'] not in process['open_partials'] or process['open_partial_positions'].get(prefix['path']) != prefix['stat']['size']:
                    raise RuntimeError('C prefix must remain at the stopped sequential HTTP writer end offset')
                source_stats[prefix['path']] = prefix['stat']
            prefix_bytes = prefix['stat']['size'] if prefix else 0
            c_remaining += row['size']-prefix_bytes
            row.update(storage='rootfs' if row['path'] in ROOTFS_NAMES else 'shm', source_prefix=prefix,
                       source_prefix_bytes=prefix_bytes)
            row['storage_directory'] = stores[row['storage']]['directory']
            target, partial = payload_paths(row)
            if target.is_symlink() or partial.is_symlink():
                raise ValueError('Private payloads must not be symlinks')
            present = target if target.exists() else partial if partial.exists() else None
            size = 0
            if present is not None:
                before = base.fingerprint(present)
                if present.stat().st_nlink != 1:
                    raise ValueError('Private payload must not be hardlinked')
                size = before['size']
                if present == target:
                    if size != row['size'] or (args.execute and helpers.digest(present) != row['sha256']):
                        raise ValueError('Existing K complete shard failed pinned SHA')
                elif size:
                    base.prefix_layout(present, row, weight_map)
                if base.fingerprint(present) != before:
                    raise RuntimeError('Private payload changed during planning')
            row.update(private_existing_bytes=size, new_private_bytes=row['size']-size,
                       prefix_copy_bytes=max(0,prefix_bytes-size),
                       network_tail_bytes=row['size']-max(size,prefix_bytes))
            expected_link = target
        link = destination/row['path']
        if (link.exists() or link.is_symlink()) and (not link.is_symlink() or link.resolve() != expected_link):
            raise ValueError(f'Existing model entry conflicts with explicit source/storage: {link}')
        row['model_link_target'] = str(expected_link)
        rows.append(row)
    stores['persistent']['c_reserved_bytes'] = max(C_APPEND_FLOOR, c_remaining)
    plan = {'status': 'PLAN', 'repo': mapping['repo'], 'revision': mapping['hf_revision'],
            'root': str(root), 'source': str(source), 'destination': str(destination),
            'source_process': process, 'source_files': source_stats, 'http_contract': contract,
            'files': rows, 'small_assets': assets, 'filesystems': stores,
            'weight_sha_verification_deferred': not args.execute,
            'c_actual_future_append_bytes': c_remaining, 'c_reserved_floor_bytes': C_APPEND_FLOOR,
            'linked_complete_shards': sum(row['source_complete'] for row in rows),
            'private_payload_full_bytes': sum(row['size'] for row in rows if not row['source_complete']),
            'new_private_payload_bytes': sum(row['new_private_bytes'] for row in rows),
            'scope': 'Complete C shards are only read through symlinks; missing shards consume full private bytes. No DS cleanup or reflinks.'}
    check_sources(plan, base)
    # Refresh free space after potentially long source SHA scans.
    plan['current_budget'] = budget(plan, helpers)
    plan['space_ok'] = all(row['space_ok'] for row in plan['current_budget'].values())
    return plan


def check_sources(plan, base):
    if plan['source_process'] and base.source_process(plan['source_process']['pid'], Path(plan['source'])) != plan['source_process']:
        raise RuntimeError('C prefix writer changed; stopping without modifying or signaling it')
    if any(base.fingerprint(Path(path)) != expected for path,expected in plan['source_files'].items()):
        raise RuntimeError('C source files changed')


def copy_prefix(row, base, guard):
    """Ordinary sequential isolated copy; interrupted K prefix remains resumable."""
    if not row['source_prefix']:
        return
    target, partial = payload_paths(row)
    if target.exists():
        return
    source = Path(row['source_prefix']['path'])
    expected = row['source_prefix']['stat']
    offset = partial.stat().st_size if partial.exists() else 0
    if offset >= expected['size']:
        return
    if base.fingerprint(source) != expected:
        raise RuntimeError('C prefix changed before copy')
    with source.open('rb') as old, partial.open('ab') as new:
        old.seek(offset)
        try:
            while offset < expected['size']:
                if guard.is_set():
                    raise RuntimeError('Prefix copy cancelled; K partial preserved')
                block = old.read(min(8*2**20, expected['size']-offset))
                if not block:
                    raise IOError('C prefix unexpectedly truncated')
                new.write(block)
                offset += len(block)
        finally:
            new.flush()
            os.fsync(new.fileno())
    if base.fingerprint(source) != expected:
        raise RuntimeError('C prefix changed during copy; K partial preserved')


def execute(args, plan, helpers, base):
    root = Path(plan['root'])
    run = root/'runs'/(time.strftime('qwen_after_deepseek_%Y%m%d_%H%M%S_')+str(os.getpid()))
    run.mkdir(parents=True, exist_ok=False)
    receipt = {**plan, 'status': 'STARTING', 'pid': os.getpid(), 'started_unix': time.time(), 'completed': []}
    stop, mutex, checked = threading.Event(), threading.Lock(), [0.]
    def save():
        helpers.atomic_json(run/'source.json', receipt)
    def interrupted(signum, frame):
        stop.set()
        raise InterruptedError(f'Signal {signum}; private partials retained')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    class Guard:
        def is_set(self):
            with mutex:
                if not stop.is_set() and time.monotonic()-checked[0] >= 2:
                    check_sources(plan, base)
                    current = budget(plan, helpers)
                    receipt['last_budget'] = current
                    if not all(row['space_ok'] for row in current.values()):
                        raise RuntimeError('Actual free space no longer preserves all reserves; see last_budget')
                    checked[0] = time.monotonic()
                return stop.is_set()
        def wait(self, seconds):
            return stop.wait(seconds)
    guard = Guard()
    save()
    try:
        idle(root)
        if not plan['space_ok']:
            raise RuntimeError('Actual capacity insufficient; no DS cleanup is performed')
        for store in plan['filesystems'].values():
            directory = Path(store['directory'])
            directory.mkdir(parents=True, exist_ok=True)
            if (directory/'.modelscope-partials').is_symlink():
                raise ValueError('Private partial directory must not be a symlink')
            (directory/'.modelscope-partials').mkdir(exist_ok=True)
        guard.is_set()
        destination = Path(plan['destination'])
        for row in plan['small_assets']:
            target = destination/row['path']
            if not target.exists():
                with target.open('xb') as output:
                    output.write(Path(row['source']).read_bytes())
            if target.is_symlink() or helpers.digest(target) != row['sha256']:
                raise ValueError('K small asset failed pinned SHA')
        receipt['status'] = 'COPYING_PREFIXES'
        save()
        for row in plan['files']:
            if not row['source_complete']:
                copy_prefix(row, base, guard)
        receipt['status'] = 'DOWNLOADING'
        save()
        with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
            pending = [pool.submit(helpers.download_shard, row, Path(row['storage_directory']),
                       args.chunk_mib*2**20, args.retries, guard) for row in plan['files'] if not row['source_complete']]
            try:
                for future in as_completed(pending):
                    receipt['completed'].append(future.result())
                    save()
            except BaseException:
                stop.set()
                for future in pending:
                    future.cancel()
                raise
        check_sources(plan, base)
        for row in plan['files']:
            link, target = destination/row['path'], Path(row['model_link_target'])
            if not link.exists() and not link.is_symlink():
                link.symlink_to(target)
            if not link.is_symlink() or link.resolve() != target:
                raise ValueError('Final model link differs from verified target')
        checked[0] = 0.
        if guard.is_set():
            raise RuntimeError('Cancelled before READY')
        receipt.update(status='READY', scope='16 pinned full-file SHA256 identities verified; C links read-only, K shards private and volatile; model load not tested')
    except BaseException as error:
        stop.set()
        receipt.update(status='FAILED', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        receipt['ended_unix'] = time.time()
        save()
    print(json.dumps({'status': receipt['status'], 'receipt': str(run/'source.json'), 'destination': plan['destination']}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--plan-only', action='store_true')
    mode.add_argument('--execute', action='store_true')
    parser.add_argument('--root', type=Path, default=Path('/root/autodl-tmp/research_K_expert_skip'))
    parser.add_argument('--source', type=Path, default=Path('/root/autodl-tmp/research_C_format_sched/checkpoints/qwen3_bf16_ad44e777'))
    parser.add_argument('--destination', type=Path)
    parser.add_argument('--source-pid', type=int, default=21898)
    parser.add_argument('--max-workers', type=int, default=2)
    parser.add_argument('--chunk-mib', type=int, default=64)
    parser.add_argument('--retries', type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.max_workers <= 2 or not 1 <= args.chunk_mib <= 128 or not 0 <= args.retries <= 5:
        parser.error('workers1..2, chunk1..128MiB, retries0..5 required')
    args.root = args.root.resolve()
    args.destination = args.destination or args.root/'models/qwen3_bf16_ad44e777'
    sys.path.insert(0, str(args.root/'code'))
    import prepare_deepseek_mirror as helpers
    base = support()
    if args.plan_only:
        print(json.dumps(make_plan(args, helpers, base), indent=2))
        return
    with (args.root/'qwen_partial_download.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        idle(args.root)
        execute(args, make_plan(args, helpers, base), helpers, base)


if __name__ == '__main__':
    main()
