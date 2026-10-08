#!/usr/bin/env python3
"""Same frozen ablation, relocated as a whole to the authorized primary GPU."""
from pathlib import Path
import signal
import sys
import run_native_residency_ablation_triplet_r01 as triple

group,base=triple.group,triple.base
group.BASE='/root/moe-a-native-residency-ablation-primary-stage-r01-20261002'
group.SESSION='/root/moe-a-native-residency-ablation-primary-session-r01-20261002'
group.SOURCE_CACHE=Path('/root/moe-a-native-backfill-only-primary-session-r01-20261001/runtime-cache-first')
group.OUTPUTS=tuple('/root/moe-a-native-residency-ablation-primary-output-'+label+'-r01-20261002' for label in group.LABELS)
triple.prior.SOURCE_ARMS=('native_full_reference','native_full_ordinary_only')

if __name__=='__main__':
    signal.signal(signal.SIGTERM,base.stop_requested)
    try:raise SystemExit(base.main())
    except (OSError,ValueError,base.subprocess.SubprocessError) as error:
        print('ABORT_NATIVE_RESIDENCY_ABLATION_PRIMARY:',error,file=sys.stderr)
        raise SystemExit(75)
