"""Reuse the unchanged native driver for the frozen two-fixed-cap arrival probe."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
import run_confirmation as base


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--wait-lock', type=float, default=0)
    args = parser.parse_args()
    design = json.loads((ROOT/'design.json').read_text())
    assert design['status_at_freeze'] == 'PRE_GPU_FROZEN'
    for relative, digest in design['source_sha256'].items():
        assert sha(ROOT/relative) == digest, relative
    workload = (ROOT/design['workload']).resolve()
    assert sha(workload) == design['workload_sha256']
    original_dump = base.dump
    def dump(path, value):
        if path.name == 'protocol.json':
            value = dict(value, experiment_kind='EXTENDED_MIXED_ARRIVAL_FIXED_CAP_DEVELOPMENT',
                frozen_design=design, frozen_design_sha256=sha(ROOT/'design.json'),
                legal_fixed_cap_choices=[384, 2048],
                action_record='The fixed policy chooses its named cap on every step. budget records that choice; prefill_tokens records actual native execution. No dynamic/shadow controller.',
                decision_time_note='Original start_s is engine.step entry; policy is a constant and does not consume a wall-clock state signal.')
        original_dump(path, value)
    base.dump = dump
    sys.argv = [__file__, '--output', args.output, '--wait-lock', str(args.wait_lock),
        '--workload', str(workload), '--policies', ','.join(design['policies']),
        '--warm-policies', ','.join(design['warm_policies']), '--target-ms', '24']
    raise SystemExit(base.main())
