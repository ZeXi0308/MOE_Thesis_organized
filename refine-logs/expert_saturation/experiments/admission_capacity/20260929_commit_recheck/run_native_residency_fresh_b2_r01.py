#!/usr/bin/env python3
"""Second frozen fresh-input block: tail, arrival, density; retain both."""
import signal
import sys
import run_native_residency_fresh_b1_r01 as first

group, base, prior = first.group, first.base, first.prior
prior.RULES = ('tail', 'arrival', 'service_density')
group.ARMS = tuple('ordinary_' + rule for rule in prior.RULES)
group.BASE = '/root/moe-a-native-residency-fresh-b2-stage-r01-20261002'
group.SESSION = '/root/moe-a-native-residency-fresh-b2-session-r01-20261002'
group.OUTPUTS = tuple('/root/moe-a-native-residency-fresh-b2-output-' + label + '-r01-20261002' for label in group.LABELS)

if __name__ == '__main__':
    signal.signal(signal.SIGTERM, base.stop_requested)
    try:
        raise SystemExit(base.main())
    except (OSError, ValueError, base.subprocess.SubprocessError) as error:
        print('ABORT_NATIVE_RESIDENCY_FRESH_B2:', error, file=sys.stderr)
        raise SystemExit(75)
