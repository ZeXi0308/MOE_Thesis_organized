#!/usr/bin/env python3
"""Compare current-request retention with both completed BidKV controls."""
import signal
import sys
import run_native_self_preempt_continue_triplet_r01 as inherited

group,base,prior=inherited.group,inherited.base,inherited.prior
base.PACKAGE_NAME='candidate_native_current_guard_r01'
base.PACKAGE_SHA256='a9d0f1d18428149955bf6678a767a25e9dbfb9567ac774cb60812d0dea5ec5ef'
group.PACKAGE_SHA256=base.PACKAGE_SHA256
prior.RULES=('bidkv_score',)*3
inherited.MODES=('off','on','off')
GUARDS=('off','off','on')
group.ARMS=('bidkv_break','bidkv_continue','bidkv_current_guard')
group.BASE='/root/moe-a-native-current-guard-stage-r01-20261002'
group.SESSION='/root/moe-a-native-current-guard-session-r01-20261002'
group.OUTPUTS=tuple('/root/moe-a-native-current-guard-output-'+label+'-r01-20261002' for label in group.LABELS)
original_env=base.private_model_env

def private_model_env(plan,session_dir):
    env=original_env(plan,session_dir)
    found=[i for i,arm in enumerate(group.ARMS) if (session_dir/f'cell-{i:02d}-{arm}').exists()]
    index=found[-1] if found else 0
    env['A_NATIVE_VICTIM_CURRENT_GUARD']=GUARDS[index]
    if found:
        base.write_json(session_dir/f'cell-{index:02d}-{group.ARMS[index]}'/'current-guard-mode.json',
                        dict(arm=group.ARMS[index],mode=GUARDS[index]))
    return env

base.private_model_env=private_model_env

if __name__=='__main__':
    signal.signal(signal.SIGTERM,base.stop_requested)
    try:raise SystemExit(base.main())
    except (OSError,ValueError,base.subprocess.SubprocessError) as error:
        print('ABORT_NATIVE_CURRENT_GUARD:',error,file=sys.stderr)
        raise SystemExit(75)
