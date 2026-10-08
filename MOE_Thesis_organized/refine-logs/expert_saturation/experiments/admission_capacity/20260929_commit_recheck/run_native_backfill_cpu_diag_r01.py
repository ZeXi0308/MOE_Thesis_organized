#!/usr/bin/env python3
"""Single same-policy native-full cell with adapter host-time diagnostics."""
from pathlib import Path
import json
import shutil
import signal
import sys
import time
import run_native_backfill_only_pair_r01 as prior

group = prior.group
base = prior.base
base.PACKAGE_NAME = 'candidate_native_backfill_cpu_r01'
base.PACKAGE_SHA256 = '5050566b57250692d24976932d1f4fbc31b8dc53f37904fdffb5f2ab1ddc6727'
group.PACKAGE_SHA256 = base.PACKAGE_SHA256
group.LABELS = ('first',)
group.ARMS = ('native_full_ordinary_cpu_diagnostic',)
group.GATES = ('ordinary',)
group.VARIANTS = ('native_full_ordinary_only',)
group.BASE = '/root/moe-a-native-backfill-cpu-stage-r01-20261001'
group.SESSION = '/root/moe-a-native-backfill-cpu-session-r01-20261001'
group.SOURCE_CACHE = Path('/root/moe-a-waiter-backfill-backup-session-r01-20261001/runtime-cache-first')
group.OUTPUTS = ('/root/moe-a-native-backfill-cpu-output-r01-20261001',)
SOURCE_ARMS = ('waiter_reference_q1', 'ordinary_backfill', 'primary_first_followup')


def validate(plan):
    base.require(plan.get('session_dir') == group.SESSION and
                 plan.get('approved_total_wall_seconds') == 2100 and
                 plan.get('warm_runtime_cache_source') == str(group.SOURCE_CACHE),
                 'CPU diagnostic identity, cache or wall bound changed')
    cells = plan.get('cells', [])
    base.require(len(cells) == 1 and cells[0].get('arm') == group.ARMS[0] and
                 cells[0].get('package_dir') == group.BASE + '/first/' + base.PACKAGE_NAME and
                 cells[0].get('output_dir') == group.OUTPUTS[0] and
                 cells[0].get('max_wall_seconds') == 900,
                 'CPU diagnostic cell differs')
    base.FROZEN_ARMS = group.ARMS
    return group.original_validate(plan)


def prepare_runtime_cache(session_dir):
    group.original_prepare_cache(session_dir)
    base.require(session_dir == Path(group.SESSION), 'unexpected CPU diagnostic session')
    source = group.SOURCE_CACHE
    receipt = json.loads((source.parent / 'receipt.json').read_text())
    base.require(receipt.get('status') == 'CELLS_COMPLETE' and
                 tuple(c.get('arm') for c in receipt.get('cells', [])) == SOURCE_ARMS,
                 'CPU seed lacks completed source receipt')
    seed = group.cache_inventory(source)
    size = sum(row[0] for row in seed.values())
    base.require(seed and shutil.disk_usage('/root').free >= 2 * 1024**3 + size,
                 'empty seed or insufficient CPU diagnostic storage')
    destination = session_dir / 'runtime-cache-first'
    base.require(not destination.exists(), 'CPU diagnostic cache already exists')
    shutil.copytree(source, destination, copy_function=shutil.copy2)
    base.require(group.cache_inventory(destination) == seed, 'CPU diagnostic seed differs')
    group._seed_files = seed
    base.write_json(session_dir / 'runtime-cache-seed-receipt.json', {
        'status': 'ONE_PRIVATE_COPY_PREPARED_UNDER_LOCK',
        'source': str(source), 'source_receipt_arms': list(SOURCE_ARMS),
        'source_files': len(seed), 'source_bytes': size,
        'destinations': [str(destination)], 'copied_unix_s': time.time(),
        'identity_check': 'relative path, byte size and preserved mtime_ns',
    })


base.validate_plan = validate
base.prepare_runtime_cache = prepare_runtime_cache

if __name__ == '__main__':
    signal.signal(signal.SIGTERM, base.stop_requested)
    try:
        raise SystemExit(base.main())
    except (OSError, ValueError, base.subprocess.SubprocessError) as error:
        print('ABORT_NATIVE_BACKFILL_CPU_DIAGNOSTIC:', error, file=sys.stderr)
        raise SystemExit(75)
