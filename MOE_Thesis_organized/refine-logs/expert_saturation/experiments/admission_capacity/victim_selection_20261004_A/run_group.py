"""One nonblocking common lock for the complete ABBA victim-only block."""
import argparse
import json
from pathlib import Path
import signal

import run_group_base as base


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    if plan['kind'] != 'FUNDING_VICTIM_ONLY_ABBA':
        raise ValueError('Wrong experiment')
    base.MANIFEST = plan['package_manifest_sha256']
    base.ARMS = ('tail', 'host_missing', 'host_missing', 'tail')
    original_env = base.private_env

    def env(p, cache, arm, memory_file):
        result = original_env(p, cache, 'queue_fund', memory_file)
        result['A_RECOVERY_LEASE_MODE'] = 'q1'
        result['A_FUNDING_VICTIM_RULE'] = arm if arm in ('tail', 'host_missing') else 'tail'
        result['PYTHONPATH'] = p['runtime_overlay']
        return result

    base.private_env = env
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(InterruptedError('SIGTERM')))
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('Group budget')))
    return base.run(plan, args.plan)


if __name__ == '__main__':
    raise SystemExit(main())
