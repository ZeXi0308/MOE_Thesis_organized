#!/usr/bin/env python3
"""Resource-only retry of frozen oldest-admission R02; reuse its package copies."""
import signal
import sys
import run_native_oldest_admission_triplet_r02 as prior
base,group=prior.base,prior.group
group.SESSION="/root/moe-a-native-oldest-admission-session-r03-20261002"
group.OUTPUTS=tuple("/root/moe-a-native-oldest-admission-output-"+label+"-r03-20261002" for label in group.LABELS)
if __name__=="__main__":
    signal.signal(signal.SIGTERM,base.stop_requested)
    try:raise SystemExit(base.main())
    except (OSError,ValueError,base.subprocess.SubprocessError) as error:
        print("ABORT_OLDEST_ADMISSION_RESOURCE_RETRY:",error,file=sys.stderr)
        raise SystemExit(75)
