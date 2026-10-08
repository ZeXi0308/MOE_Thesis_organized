#!/usr/bin/env python3
"""One unchanged-Q10 native observation of the actual extended-protection gates."""
import json
from pathlib import Path
import shutil
import signal
import sys
import serial_group_h1_guard_qual_r02 as base

base.PACKAGE_NAME = 'candidate_protection_scope_diag_r01'
base.PACKAGE_SHA256 = '3030cc4b0af46a3cfa642dbaed6039e0d5e451a177843fc2444d889786e342ec'
base.FROZEN_ARMS = ('protection_scope_q10_diagnostic',)
STAGE = '/root/moe-a-protection-scope-diag-stage-r01-20261001'
SESSION = '/root/moe-a-protection-scope-diag-session-r01-20261001'
OUTPUT = '/root/moe-a-protection-scope-diag-output-r01-20261001'
SOURCE_CACHE = Path('/root/moe-a-capacity-protection-session-r03-20261001/runtime-cache-second')
original_validate = base.validate_plan
original_prepare = base.prepare_runtime_cache


def validate(plan):
    base.require(plan['session_dir'] == SESSION and plan['approved_total_wall_seconds'] == 2200,
                 'diagnostic identity or finite wall limit changed')
    base.require(plan['cells'] == [dict(arm=base.FROZEN_ARMS[0],
        package_dir=STAGE+'/'+base.PACKAGE_NAME, output_dir=OUTPUT, max_wall_seconds=900)],
        'diagnostic cell changed')
    return original_validate(plan)


def prepare_cache(session):
    receipt = json.loads((SOURCE_CACHE.parent/'receipt.json').read_text())
    base.require(receipt['status'] == 'CELLS_COMPLETE', 'seed experiment is incomplete')
    base.require(SOURCE_CACHE.is_dir() and not SOURCE_CACHE.is_symlink(), 'missing seed cache')
    base.require(shutil.disk_usage('/root').free > 2*1024**3 + base.regular_tree_bytes(SOURCE_CACHE),
                 'need 2 GiB free after cache copy')
    original_prepare(session)
    shutil.copytree(SOURCE_CACHE, session/'runtime-cache', dirs_exist_ok=True, copy_function=shutil.copy2)
    base.write_json(session/'runtime-cache-seed-receipt.json',dict(source=str(SOURCE_CACHE),
        role='Independent private copy for source localization, not a performance comparison'))


def argv(package, output):
    return [str(package/'pkg/run.sh'), 'eager', 'performance', 'q10', str(output)]


base.validate_plan = validate
base.prepare_runtime_cache = prepare_cache
base.qualification_argv = argv
if __name__ == '__main__':
    signal.signal(signal.SIGTERM, base.stop_requested)
    try:
        raise SystemExit(base.main())
    except (OSError, ValueError, base.subprocess.SubprocessError) as exc:
        print('ABORT_PROTECTION_SCOPE_DIAGNOSTIC:', exc, file=sys.stderr)
        raise SystemExit(75)
