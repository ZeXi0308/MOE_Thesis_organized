#!/usr/bin/env python3
"""Opposite-order second confirmation block; same1.25GiB postcache reserve."""
import signal
import sys
import run_native_current_guard_fresh_b1_r02 as first

group,base,guard=first.group,first.base,first.first.guard
group.ARMS=tuple(reversed(group.ARMS))
guard.prior.RULES=tuple(reversed(guard.prior.RULES))
guard.inherited.MODES=tuple(reversed(guard.inherited.MODES))
guard.GUARDS=tuple(reversed(guard.GUARDS))
group.BASE='/root/moe-a-native-current-guard-fresh-b2-stage-r02-20261002'
group.SESSION='/root/moe-a-native-current-guard-fresh-b2-session-r02-20261002'
group.OUTPUTS=tuple('/root/moe-a-native-current-guard-fresh-b2-output-'+label+'-r02-20261002' for label in group.LABELS)
original_prepare=base.prepare_runtime_cache

def prepare_runtime_cache(session_dir):
    original_prepare(session_dir)
    receipt=first.first.inherited.json.loads((session_dir/'runtime-cache-seed-receipt.json').read_text())
    receipt['reserve_basis']='Same1.25GiB after4x228492707Bseed as B1R02. B1R01 diskabort informedresourcechange; B2R01 neverexecuted. Both originalfrozenorders/input/18criteria remainunchanged.'
    base.write_json(session_dir/'runtime-cache-seed-receipt.json',receipt)

base.prepare_runtime_cache=prepare_runtime_cache

if __name__=='__main__':
    signal.signal(signal.SIGTERM,base.stop_requested)
    try:raise SystemExit(base.main())
    except (OSError,ValueError,base.subprocess.SubprocessError) as error:
        print('ABORT_NATIVE_CURRENT_GUARD_FRESH_B2_R02:',error,file=sys.stderr)
        raise SystemExit(75)
