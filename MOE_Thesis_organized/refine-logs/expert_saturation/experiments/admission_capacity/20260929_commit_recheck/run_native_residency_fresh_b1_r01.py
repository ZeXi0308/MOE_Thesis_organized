#!/usr/bin/env python3
"""First frozen fresh-input block: density, arrival, native tail."""
from pathlib import Path
import signal
import sys
import run_native_residency_victim_triplet_r01 as prior

group, base = prior.group, prior.base
base.PACKAGE_NAME = 'candidate_native_residency_fresh_r01'
base.PACKAGE_SHA256 = 'd4ce24415f7e9c120f0af0fcf3397aa6b0b2dbb307a7a1e1c8313cf4a7ee4667'
group.PACKAGE_SHA256 = base.PACKAGE_SHA256
prior.RULES = ('service_density', 'arrival', 'tail')
group.ARMS = tuple('ordinary_' + rule for rule in prior.RULES)
group.BASE = '/root/moe-a-native-residency-fresh-b1-stage-r01-20261002'
group.SESSION = '/root/moe-a-native-residency-fresh-b1-session-r01-20261002'
group.SOURCE_CACHE = Path('/root/moe-a-native-backfill-only-primary-session-r01-20261001/runtime-cache-first')
group.OUTPUTS = tuple('/root/moe-a-native-residency-fresh-b1-output-' + label + '-r01-20261002' for label in group.LABELS)
prior.prior.SOURCE_ARMS = ('native_full_reference', 'native_full_ordinary_only')

if __name__ == '__main__':
    signal.signal(signal.SIGTERM, base.stop_requested)
    try:
        raise SystemExit(base.main())
    except (OSError, ValueError, base.subprocess.SubprocessError) as error:
        print('ABORT_NATIVE_RESIDENCY_FRESH_B1:', error, file=sys.stderr)
        raise SystemExit(75)
