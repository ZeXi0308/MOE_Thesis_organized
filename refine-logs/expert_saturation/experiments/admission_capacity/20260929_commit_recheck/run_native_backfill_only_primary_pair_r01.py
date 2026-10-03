#!/usr/bin/env python3
"""Run the unchanged two-arm package on the original authorized GPU."""
from pathlib import Path
import signal
import sys
import run_native_backfill_only_pair_r01 as pair

group = pair.group
base = pair.base
group.BASE = '/root/moe-a-native-backfill-only-primary-stage-r01-20261001'
group.SESSION = '/root/moe-a-native-backfill-only-primary-session-r01-20261001'
group.SOURCE_CACHE = Path('/root/moe-a-spare-followup-fresh-session-r01-20261001/runtime-cache-first')
group.OUTPUTS = tuple('/root/moe-a-native-backfill-only-primary-output-' + label + '-r01-20261001'
                      for label in group.LABELS)
pair.SOURCE_ARMS = ('spare_followup_off', 'spare_followup_on', 'native_full_native')

if __name__ == '__main__':
    signal.signal(signal.SIGTERM, base.stop_requested)
    try:
        raise SystemExit(base.main())
    except (OSError, ValueError, base.subprocess.SubprocessError) as error:
        print('ABORT_NATIVE_BACKFILL_ONLY_PRIMARY_PAIR:', error, file=sys.stderr)
        raise SystemExit(75)
