#!/usr/bin/env python3
"""Direct probe: native tail, density with break, density with continuation."""
import signal
import sys
import run_native_residency_fresh_b1_r01 as inherited

group, base, prior = inherited.group, inherited.base, inherited.prior
base.PACKAGE_NAME = 'candidate_native_self_preempt_continue_r01'
base.PACKAGE_SHA256 = '2770b1a25075fd153d18e89c25a4f166db9dbd993343b5ce652912f2a85a6376'
group.PACKAGE_SHA256 = base.PACKAGE_SHA256
prior.RULES = ('tail', 'service_density', 'service_density')
MODES = ('off', 'off', 'on')
group.ARMS = ('tail_break', 'density_break', 'density_continue')
group.BASE = '/root/moe-a-native-self-preempt-continue-stage-r01-20261002'
group.SESSION = '/root/moe-a-native-self-preempt-continue-session-r01-20261002'
group.OUTPUTS = tuple('/root/moe-a-native-self-preempt-continue-output-' + label + '-r01-20261002' for label in group.LABELS)
original_env = base.private_model_env

def private_model_env(plan, session_dir):
    env = original_env(plan, session_dir)
    found = [i for i, arm in enumerate(group.ARMS) if (session_dir / f'cell-{i:02d}-{arm}').exists()]
    index = found[-1] if found else 0
    env['A_SELF_PREEMPT_CONTINUE'] = MODES[index]
    if found:
        base.write_json(session_dir / f'cell-{index:02d}-{group.ARMS[index]}' / 'continuation-mode.json',
                        dict(arm=group.ARMS[index], rule=prior.RULES[index], mode=MODES[index]))
    return env

base.private_model_env = private_model_env

if __name__ == '__main__':
    signal.signal(signal.SIGTERM, base.stop_requested)
    try:
        raise SystemExit(base.main())
    except (OSError, ValueError, base.subprocess.SubprocessError) as error:
        print('ABORT_NATIVE_SELF_PREEMPT_CONTINUE:', error, file=sys.stderr)
        raise SystemExit(75)
