#!/usr/bin/env python3
"""Resource-only retry using unchanged R01 packages after zero-cell disk abort."""
import signal
import sys
import run_native_victim_size_triplet_r01 as prior

group,base=prior.group,prior.base
group.SESSION="/root/moe-a-native-victim-size-session-r02-20261002"
group.OUTPUTS=tuple("/root/moe-a-native-victim-size-output-"+label+"-r02-20261002"
                    for label in group.LABELS)

if __name__=="__main__":
    signal.signal(signal.SIGTERM,base.stop_requested)
    try:
        raise SystemExit(base.main())
    except (OSError,ValueError,base.subprocess.SubprocessError) as error:
        print("ABORT_NATIVE_VICTIM_SIZE_R02:",error,file=sys.stderr)
        raise SystemExit(75)

