"""Compare Q1, ordinary eligible backfill, and primary-output-gated follow-up."""
from pathlib import Path
import signal
import sys

import run_spare_followup_fresh_triplet_r01 as group

group.PACKAGE_SHA256 = '7504ca82cb0be53056e6aae135a07cf75cdc382e87f2077435415ecadbd277cc'
group.base.PACKAGE_SHA256 = group.PACKAGE_SHA256
group.base.PACKAGE_NAME = 'candidate_waiter_backfill_r01'
group.ARMS = ('waiter_reference_q1', 'ordinary_backfill', 'primary_first_followup')
group.GATES = ('off', 'ordinary', 'on')
group.VARIANTS = ('eager', 'eager', 'eager')
group.BASE = '/root/moe-a-waiter-backfill-stage-r01-20261001'
group.SESSION = '/root/moe-a-waiter-backfill-session-r01-20261001'
group.OUTPUTS = tuple('/root/moe-a-waiter-backfill-output-' + label + '-r01-20261001'
                      for label in group.LABELS)

if __name__ == '__main__':
    signal.signal(signal.SIGTERM, group.base.stop_requested)
    try:
        raise SystemExit(group.base.main())
    except (OSError, ValueError, group.base.subprocess.SubprocessError) as exc:
        print('ABORT_WAITER_BACKFILL_TRIPLET:', exc, file=sys.stderr)
        raise SystemExit(75)
