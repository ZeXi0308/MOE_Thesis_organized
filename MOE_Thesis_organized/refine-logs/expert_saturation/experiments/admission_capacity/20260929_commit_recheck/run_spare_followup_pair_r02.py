#!/usr/bin/env python3
"""Reverse-order replication of the unchanged spare-followup native pair."""
import signal
import sys
import run_spare_followup_pair_r01 as pair

pair.ARMS = ("spare_followup_on", "spare_followup_off")
pair.GATES = ("on", "off")
pair.BASE = "/root/moe-a-spare-followup-stage-r02-20261001"
pair.SESSION = "/root/moe-a-spare-followup-session-r02-20261001"
pair.OUTPUTS = tuple("/root/moe-a-spare-followup-output-" + label + "-r02-20261001"
                     for label in pair.LABELS)

if __name__ == "__main__":
    signal.signal(signal.SIGTERM, pair.base.stop_requested)
    try:
        raise SystemExit(pair.base.main())
    except (OSError, ValueError, pair.base.subprocess.SubprocessError) as exc:
        print("ABORT_SPARE_FOLLOWUP_PAIR_R02:", exc, file=sys.stderr)
        raise SystemExit(75)
