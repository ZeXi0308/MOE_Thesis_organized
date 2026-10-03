#!/usr/bin/env python3
"""Frozen opposite-order block2; retain both blocks independently."""
import signal
import sys
import run_native_current_guard_fresh_b1_r01 as first

group,base,guard=first.group,first.base,first.guard
group.ARMS=tuple(reversed(group.ARMS))
guard.prior.RULES=tuple(reversed(guard.prior.RULES))
guard.inherited.MODES=tuple(reversed(guard.inherited.MODES))
guard.GUARDS=tuple(reversed(guard.GUARDS))
group.BASE='/root/moe-a-native-current-guard-fresh-b2-stage-r01-20261002'
group.SESSION='/root/moe-a-native-current-guard-fresh-b2-session-r01-20261002'
group.OUTPUTS=tuple('/root/moe-a-native-current-guard-fresh-b2-output-'+label+'-r01-20261002' for label in group.LABELS)

if __name__=='__main__':
    signal.signal(signal.SIGTERM,base.stop_requested)
    try:raise SystemExit(base.main())
    except (OSError,ValueError,base.subprocess.SubprocessError) as error:
        print('ABORT_NATIVE_CURRENT_GUARD_FRESH_B2:',error,file=sys.stderr)
        raise SystemExit(75)
