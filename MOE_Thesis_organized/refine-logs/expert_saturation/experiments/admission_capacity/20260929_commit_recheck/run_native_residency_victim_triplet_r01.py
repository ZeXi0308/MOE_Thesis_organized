#!/usr/bin/env python3
"""One direct native-full victim-selection triplet, all other policy equal."""
from pathlib import Path
import signal
import sys
import run_waiter_backfill_backup_triplet_r01 as prior

group,base=prior.fresh,prior.base
base.PACKAGE_NAME='candidate_native_residency_victim_r01'
base.PACKAGE_SHA256='9b97653fe7b022160b1dd2143fbc1f717db5d32e39cbcb80b9318eb4704cb6c8'
group.PACKAGE_SHA256=base.PACKAGE_SHA256
group.ARMS=('ordinary_tail','ordinary_arrival','ordinary_service_density')
group.LABELS=('first','second','third')
group.GATES=('ordinary',)*3
group.VARIANTS=('native_full_ordinary_only',)*3
group.BASE='/root/moe-a-native-residency-victim-stage-r01-20261002'
group.SESSION='/root/moe-a-native-residency-victim-session-r01-20261002'
group.SOURCE_CACHE=Path('/root/moe-a-native-backfill-lazy-perf-session-r01-20261002/runtime-cache-first')
group.OUTPUTS=tuple('/root/moe-a-native-residency-victim-output-'+label+'-r01-20261002' for label in group.LABELS)
prior.SOURCE_ARMS=('native_full_ordinary_only','native_full_reference')
RULES=('tail','arrival','service_density')


def private_model_env(plan,session_dir):
    env=group.private_model_env(plan,session_dir)
    found=[i for i,arm in enumerate(group.ARMS) if (session_dir/f'cell-{i:02d}-{arm}').exists()]
    index=found[-1] if found else 0
    env['A_NATIVE_VICTIM_RULE']=RULES[index]
    if found:
        base.write_json(session_dir/f'cell-{index:02d}-{group.ARMS[index]}'/'victim-rule.json',
                        dict(arm=group.ARMS[index],rule=RULES[index]))
    return env

base.private_model_env=private_model_env

if __name__=='__main__':
    signal.signal(signal.SIGTERM,base.stop_requested)
    try:raise SystemExit(base.main())
    except (OSError,ValueError,base.subprocess.SubprocessError) as error:
        print('ABORT_NATIVE_RESIDENCY_VICTIM_TRIPLET:',error,file=sys.stderr)
        raise SystemExit(75)
