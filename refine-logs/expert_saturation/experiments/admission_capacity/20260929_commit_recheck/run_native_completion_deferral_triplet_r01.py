#!/usr/bin/env python3
"""Three-arm capacity-deferral pilot; freeze manifest and plan before launch."""
import signal
import sys
import run_native_bidkv_full_running_triplet_r01 as prior

group, base, current = prior.group, prior.base, prior.current
base.PACKAGE_NAME = 'candidate_native_completion_deferral_r01'
base.PACKAGE_SHA256 = 'ed0361a4c5e562664dc121fe28b3b9159f124fd55c8fccef2e9599daf6f2bf47'
group.PACKAGE_SHA256 = base.PACKAGE_SHA256
current.prior.RULES = ('tail', 'tail', 'tail')
current.inherited.MODES = ('off', 'off', 'off')
current.GUARDS = ('off', 'off', 'off')
prior.FULL_MODES = ('off', 'off', 'off')
DEFERRAL_MODES = ('off', 'prefix_work', 'prefix_finish')
group.ARMS = ('tail_break', 'prefix_work', 'prefix_finish')
group.BASE = '/root/moe-a-native-completion-deferral-stage-r01-20261002'
group.SESSION = '/root/moe-a-native-completion-deferral-session-r01-20261002'
group.OUTPUTS = tuple('/root/moe-a-native-completion-deferral-output-' + label + '-r01-20261002'
                      for label in group.LABELS)
previous_private_env = base.private_model_env

def private_model_env(plan, session_dir):
    env = previous_private_env(plan, session_dir)
    found = [i for i, arm in enumerate(group.ARMS)
             if (session_dir / f'cell-{i:02d}-{arm}').exists()]
    index = found[-1] if found else 0
    env['A_NATIVE_CAPACITY_DEFERRAL'] = DEFERRAL_MODES[index]
    if found:
        base.write_json(session_dir / f'cell-{index:02d}-{group.ARMS[index]}' / 'capacity-deferral-mode.json',
                        dict(arm=group.ARMS[index], mode=DEFERRAL_MODES[index],
                             victim_rule='tail', full_running='off', current_guard='off',
                             self_preempt_continue='off'))
    return env

base.private_model_env = private_model_env

if __name__ == '__main__':
    signal.signal(signal.SIGTERM, base.stop_requested)
    try:
        raise SystemExit(base.main())
    except (OSError, ValueError, base.subprocess.SubprocessError) as error:
        print('ABORT_NATIVE_COMPLETION_DEFERRAL:', error, file=sys.stderr)
        raise SystemExit(75)
