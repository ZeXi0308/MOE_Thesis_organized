#!/usr/bin/env python3
"""Compare original and demand-driven running views with identical host timers."""
from pathlib import Path
import json
import shutil
import signal
import sys
import time
import run_native_backfill_cpu_diag_r01 as prior

group, base = prior.group, prior.base
PACKAGES = ('candidate_native_backfill_cpu_r01', 'candidate_native_backfill_lazy_r01')
HASHES = ('5050566b57250692d24976932d1f4fbc31b8dc53f37904fdffb5f2ab1ddc6727',
          '275b25e8cc1db9d642d862a7f66f03a49eee53b528606ea92128b864a3196ad1')
group.LABELS = ('first', 'second')
group.ARMS = ('ordinary_timed_original', 'ordinary_timed_lazy')
group.GATES = ('ordinary', 'ordinary')
group.VARIANTS = ('native_full_ordinary_only', 'native_full_ordinary_only')
group.BASE = '/root/moe-a-native-backfill-lazy-stage-r01-20261002'
group.SESSION = '/root/moe-a-native-backfill-lazy-session-r01-20261002'
group.SOURCE_CACHE = Path('/root/moe-a-native-backfill-cpu-session-r01-20261001/runtime-cache-first')
group.OUTPUTS = tuple('/root/moe-a-native-backfill-lazy-output-' + label + '-r01-20261002'
                      for label in group.LABELS)


def select_package(index):
    base.PACKAGE_NAME = PACKAGES[index]
    base.PACKAGE_SHA256 = HASHES[index]
    group.PACKAGE_SHA256 = HASHES[index]


def validate(plan):
    base.require(plan.get('session_dir') == group.SESSION and
                 plan.get('approved_total_wall_seconds') == 3000 and
                 plan.get('warm_runtime_cache_source') == str(group.SOURCE_CACHE),
                 'lazy pair session, cache or wall bound differs')
    cells = plan.get('cells', [])
    base.require(tuple(c.get('arm') for c in cells) == group.ARMS,
                 'lazy pair order differs')
    for index, cell in enumerate(cells):
        select_package(index)
        base.require(cell.get('package_dir') == group.BASE + '/' + group.LABELS[index] + '/' + PACKAGES[index]
                     and cell.get('output_dir') == group.OUTPUTS[index]
                     and cell.get('max_wall_seconds') == 900,
                     'lazy pair cell differs')
        base.FROZEN_ARMS = (group.ARMS[index],)
        group.original_validate(dict(plan, cells=[cell]))
    select_package(0)
    base.FROZEN_ARMS = group.ARMS
    return plan


def private_model_env(plan, session_dir):
    # Base run_session creates the current cell directory before this call,
    # then sets H1_EXPECTED_MANIFEST_SHA256 from the selected package below.
    found = [i for i, arm in enumerate(group.ARMS)
             if (session_dir / f'cell-{i:02d}-{arm}').exists()]
    select_package(found[-1] if found else 0)
    return group.private_model_env(plan, session_dir)


def prepare_runtime_cache(session_dir):
    group.original_prepare_cache(session_dir)
    base.require(session_dir == Path(group.SESSION), 'unexpected lazy pair session')
    source = group.SOURCE_CACHE
    receipt = json.loads((source.parent / 'receipt.json').read_text())
    base.require(receipt.get('status') == 'CELLS_COMPLETE' and
                 tuple(c.get('arm') for c in receipt.get('cells', [])) ==
                 ('native_full_ordinary_cpu_diagnostic',), 'seed diagnostic incomplete')
    seed = group.cache_inventory(source)
    size = sum(row[0] for row in seed.values())
    base.require(seed and shutil.disk_usage('/root').free >= 2 * 1024**3 + 2 * size,
                 'empty seed or insufficient pair storage')
    for label in group.LABELS:
        destination = session_dir / ('runtime-cache-' + label)
        base.require(not destination.exists(), 'private pair cache exists')
        shutil.copytree(source, destination, copy_function=shutil.copy2)
        base.require(group.cache_inventory(destination) == seed, 'pair cache differs')
    group._seed_files = seed
    base.write_json(session_dir / 'runtime-cache-seed-receipt.json', dict(
        status='TWO_PRIVATE_COPIES_PREPARED_UNDER_LOCK', source=str(source),
        source_files=len(seed), source_bytes=size,
        destinations=[str(session_dir / ('runtime-cache-' + label)) for label in group.LABELS],
        copied_unix_s=time.time(), identity_check='relative path, size and preserved mtime_ns'))


base.validate_plan = validate
base.private_model_env = private_model_env
base.prepare_runtime_cache = prepare_runtime_cache
select_package(0)

if __name__ == '__main__':
    signal.signal(signal.SIGTERM, base.stop_requested)
    try:
        raise SystemExit(base.main())
    except (OSError, ValueError, base.subprocess.SubprocessError) as error:
        print('ABORT_NATIVE_BACKFILL_LAZY_PAIR:', error, file=sys.stderr)
        raise SystemExit(75)
