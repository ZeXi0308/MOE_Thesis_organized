#!/usr/bin/env python3
"""One timer-free lazy-ordinary/native reference pair on the backup GPU."""
from pathlib import Path
import signal
import sys
import run_native_backfill_only_pair_r01 as prior

group, base = prior.group, prior.base
base.PACKAGE_NAME = 'candidate_native_backfill_lazy_perf_r01'
base.PACKAGE_SHA256 = 'a0a942dbf490d51630f2ab4175846c83f5d475cb3a6827658fe8c27edc72c046'
group.PACKAGE_SHA256 = base.PACKAGE_SHA256
group.ARMS = ('native_full_ordinary_only', 'native_full_reference')
group.GATES = ('ordinary', 'off')
group.VARIANTS = ('native_full_ordinary_only', 'native_full_native')
group.BASE = '/root/moe-a-native-backfill-lazy-perf-stage-r01-20261002'
group.SESSION = '/root/moe-a-native-backfill-lazy-perf-session-r01-20261002'
group.SOURCE_CACHE = Path('/root/moe-a-native-backfill-lazy-session-r01-20261002/runtime-cache-first')
group.OUTPUTS = tuple('/root/moe-a-native-backfill-lazy-perf-output-' + label + '-r01-20261002'
                      for label in group.LABELS)
prior.SOURCE_ARMS = ('ordinary_timed_original', 'ordinary_timed_lazy')

if __name__ == '__main__':
    signal.signal(signal.SIGTERM, base.stop_requested)
    try:
        raise SystemExit(base.main())
    except (OSError, ValueError, base.subprocess.SubprocessError) as error:
        print('ABORT_NATIVE_BACKFILL_LAZY_PERF_PAIR:', error, file=sys.stderr)
        raise SystemExit(75)
