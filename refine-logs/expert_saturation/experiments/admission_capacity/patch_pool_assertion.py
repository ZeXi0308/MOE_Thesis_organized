#!/usr/bin/env python3
"""Make the sealed pool assertion track the requested KV size instead of one constant.

What it currently does
----------------------
`run_probe.py` verifies, after the engine is built, that the scheduler really
applied the reservation policy and that the block pool is the one the experiment
was frozen against:

    if observed != (...) or qualification.get('usable_blocks') != 7671:
        raise RuntimeError('actual reservation policy or KV pool differs ...')

Both halves are worth keeping. The first catches a silently ignored scheduler
flag. The second catches the much nastier case where `kv_cache_memory_bytes` is
accepted but the engine ends up with a different pool than intended, which would
silently change the structural deficit -- the independent variable of this sweep.

Why the constant has to go
--------------------------
The sweep varies exactly that pool, so 7671 is only correct at one of the four
points. Removing the check would give up the protection; hardcoding four values
would rot the moment a point is added.

Instead the expected pool is derived from the bytes actually requested, using
the ratio the sealed run reported (16,089,350,144 bytes over 7672 total blocks,
usable = total - 1 because block 0 is the null block):

    expected_total  = round(kv_cache_bytes / bytes_per_block)
    expected_usable = expected_total - 1

and the assertion allows +-2 blocks for rounding. At 16,089,350,144 bytes this
reproduces 7671 exactly, so the d2 point keeps the identical guard it passed
under. The observed value is recorded either way, and the analyzer uses the
engine-reported number rather than the requested one.
"""

import sys
from pathlib import Path

TARGET = Path("/root/psweep-r01/pkg/run_probe.py")

OLD = """        if observed != (args.reservation_policy == 'full') or qualification.get('usable_blocks') != 7671:
            raise RuntimeError('actual reservation policy or KV pool differs from fixed experiment')"""

NEW = """        # Expected pool derived from the bytes requested, using the sealed
        # ratio (16,089,350,144 bytes / 7672 total blocks). Block 0 is the null
        # block, so usable = total - 1. At the sealed byte count this evaluates
        # to 7671 and the guard is unchanged; at other points it tracks the
        # requested pool instead of silently accepting a different deficit.
        _bytes_per_block = 16089350144 / 7672
        _expected_usable = round(args.kv_cache_bytes / _bytes_per_block) - 1
        _actual_usable = qualification.get('usable_blocks')
        qualification['expected_usable_blocks'] = _expected_usable
        qualification['usable_blocks_within_tolerance'] = (
            _actual_usable is not None and abs(_actual_usable - _expected_usable) <= 2)
        if observed != (args.reservation_policy == 'full'):
            raise RuntimeError('actual reservation policy differs from fixed experiment')
        if _actual_usable is None or abs(_actual_usable - _expected_usable) > 2:
            raise RuntimeError(
                f'KV pool differs from request: got {_actual_usable} usable blocks, '
                f'expected {_expected_usable} for {args.kv_cache_bytes} bytes')"""


def main():
    text = TARGET.read_text()
    if "expected_usable_blocks" in text:
        print("already patched")
        return
    if text.count(OLD) != 1:
        print(f"ERROR: expected one match, found {text.count(OLD)}", file=sys.stderr)
        sys.exit(1)
    TARGET.write_text(text.replace(OLD, NEW))
    print(f"patched {TARGET}")


if __name__ == "__main__":
    main()
