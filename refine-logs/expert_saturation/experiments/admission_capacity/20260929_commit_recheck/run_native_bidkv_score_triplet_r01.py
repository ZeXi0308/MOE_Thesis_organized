#!/usr/bin/env python3
"""Conditional next baseline probe; no launch without its own frozen plan."""
import signal
import sys
import run_native_self_preempt_continue_triplet_r01 as inherited

group, base, prior = inherited.group, inherited.base, inherited.prior
base.PACKAGE_NAME = 'candidate_native_bidkv_score_r01'
base.PACKAGE_SHA256 = '24f1e31a0f116fe96b0e26696e59bcb6b43e85658e5d95a82d6835e86941166e'
group.PACKAGE_SHA256 = base.PACKAGE_SHA256
prior.RULES = ('tail', 'bidkv_score', 'bidkv_score')
inherited.MODES = ('off', 'off', 'on')
group.ARMS = ('tail_break', 'bidkv_break', 'bidkv_continue')
group.BASE = '/root/moe-a-native-bidkv-score-stage-r01-20261002'
group.SESSION = '/root/moe-a-native-bidkv-score-session-r01-20261002'
group.OUTPUTS = tuple('/root/moe-a-native-bidkv-score-output-' + label + '-r01-20261002' for label in group.LABELS)

if __name__ == '__main__':
    signal.signal(signal.SIGTERM, base.stop_requested)
    try:
        raise SystemExit(base.main())
    except (OSError, ValueError, base.subprocess.SubprocessError) as error:
        print('ABORT_NATIVE_BIDKV_SCORE:', error, file=sys.stderr)
        raise SystemExit(75)
