#!/usr/bin/env python3
"""Resource-only retry after zero-cell disk gate; unchanged frozen four-arm test."""
import signal
import sys
import run_native_current_guard_fresh_b1_r01 as first

group,base=first.group,first.base
first.inherited.DISK_RESERVE_BYTES=1280*1024**2
group.SESSION='/root/moe-a-native-current-guard-fresh-b1-session-r02-20261002'
group.OUTPUTS=tuple('/root/moe-a-native-current-guard-fresh-b1-output-'+label+'-r02-20261002' for label in group.LABELS)
original_prepare=base.prepare_runtime_cache

def prepare_runtime_cache(session_dir):
    original_prepare(session_dir)
    receipt=first.inherited.json.loads((session_dir/'runtime-cache-seed-receipt.json').read_text())
    receipt['reserve_basis']='Measured seed228492707B times4;1.25GiB after copies exceeds2.5times the four-arm output estimate0.51GB from completedtriplets. R01 aborted before anycell. Policy/input/order/analysiscriteria unchanged.'
    base.write_json(session_dir/'runtime-cache-seed-receipt.json',receipt)

base.prepare_runtime_cache=prepare_runtime_cache

if __name__=='__main__':
    signal.signal(signal.SIGTERM,base.stop_requested)
    try:raise SystemExit(base.main())
    except (OSError,ValueError,base.subprocess.SubprocessError) as error:
        print('ABORT_NATIVE_CURRENT_GUARD_FRESH_B1_R02:',error,file=sys.stderr)
        raise SystemExit(75)
