#!/usr/bin/env python3
"""Two native full-save cells; ordinary backfill adds no forced rotation."""
from pathlib import Path
import json
import shutil
import signal
import sys
import time

import run_waiter_backfill_backup_triplet_r01 as prior

group = prior.fresh
base = group.base
base.PACKAGE_NAME = 'candidate_native_backfill_only_r01'
base.PACKAGE_SHA256 = 'ffa9dc592ec9c3c14aa4db2d9d8978b745695196dcb591e1f03f4185738392bc'
group.PACKAGE_SHA256 = base.PACKAGE_SHA256
group.LABELS = ('first', 'second')
group.ARMS = ('native_full_reference', 'native_full_ordinary_only')
group.GATES = ('off', 'ordinary')
group.VARIANTS = ('native_full_native', 'native_full_ordinary_only')
group.BASE = '/root/moe-a-native-backfill-only-stage-r01-20261001'
group.SESSION = '/root/moe-a-native-backfill-only-session-r01-20261001'
group.SOURCE_CACHE = Path('/root/moe-a-waiter-backfill-backup-session-r01-20261001/runtime-cache-first')
group.OUTPUTS = tuple('/root/moe-a-native-backfill-only-output-' + label + '-r01-20261001'
                      for label in group.LABELS)
SOURCE_ARMS = ('waiter_reference_q1', 'ordinary_backfill', 'primary_first_followup')


def validate(plan):
    base.require(plan.get('session_dir') == group.SESSION and
                 plan.get('approved_total_wall_seconds') == 3000 and
                 plan.get('warm_runtime_cache_source') == str(group.SOURCE_CACHE),
                 'native backfill pair identity, cache or wall bound changed')
    cells = plan.get('cells', [])
    base.require(tuple(c.get('arm') for c in cells) == group.ARMS,
                 'native backfill pair order changed')
    for index, cell in enumerate(cells):
        base.require(cell.get('package_dir') == group.BASE + '/' + group.LABELS[index] + '/' + base.PACKAGE_NAME
                     and cell.get('output_dir') == group.OUTPUTS[index]
                     and cell.get('max_wall_seconds') == 900,
                     'native backfill pair cell changed')
        base.FROZEN_ARMS = (group.ARMS[index],)
        group.original_validate(dict(plan, cells=[cell]))
    base.FROZEN_ARMS = group.ARMS
    base.require(plan['approved_total_wall_seconds'] >=
                 sum(c['max_wall_seconds'] for c in cells) + 360 + 480 + 60 + 90 + 20,
                 'insufficient pair time')
    return plan


def prepare_runtime_cache(session_dir):
    group.original_prepare_cache(session_dir)
    base.require(session_dir == Path(group.SESSION), 'unexpected pair session')
    source = group.SOURCE_CACHE
    receipt = json.loads((source.parent / 'receipt.json').read_text())
    base.require(receipt.get('status') == 'CELLS_COMPLETE' and
                 tuple(c.get('arm') for c in receipt.get('cells', [])) == SOURCE_ARMS,
                 'seed lacks completed three-arm receipt')
    seed = group.cache_inventory(source)
    size = sum(row[0] for row in seed.values())
    base.require(seed and shutil.disk_usage('/root').free >= 2 * 1024**3 + 2 * size,
                 'empty seed or insufficient storage for two copies')
    for label in group.LABELS:
        destination = session_dir / ('runtime-cache-' + label)
        base.require(not destination.exists(), 'pair cache already exists')
        shutil.copytree(source, destination, copy_function=shutil.copy2)
        base.require(group.cache_inventory(destination) == seed, 'pair seed differs')
    group._seed_files = seed
    base.write_json(session_dir / 'runtime-cache-seed-receipt.json', {
        'status': 'TWO_PRIVATE_COPIES_PREPARED_UNDER_LOCK',
        'source': str(source), 'source_receipt_arms': list(SOURCE_ARMS),
        'source_files': len(seed), 'source_bytes': size,
        'destinations': [str(session_dir / ('runtime-cache-' + label)) for label in group.LABELS],
        'copied_unix_s': time.time(),
        'identity_check': 'relative path, byte size and preserved mtime_ns',
    })


base.validate_plan = validate
base.prepare_runtime_cache = prepare_runtime_cache

if __name__ == '__main__':
    signal.signal(signal.SIGTERM, base.stop_requested)
    try:
        raise SystemExit(base.main())
    except (OSError, ValueError, base.subprocess.SubprocessError) as error:
        print('ABORT_NATIVE_BACKFILL_ONLY_PAIR:', error, file=sys.stderr)
        raise SystemExit(75)
