"""Reuse the common-lock runner for one normal-capacity characterization group."""
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
    base.require(plan['kind'] == 'PRO6000_NORMAL_CAPACITY', 'Wrong experiment')
    cells = plan['cells']
    base.require(len({c['label'] for c in cells}) == len(cells), 'Duplicate cell label')
    base.require(plan['arms'] == [c['label'] for c in cells], 'Cell order differs')
    base.require(all(c['input_case'] in ('low', 'knee', 'high') and
                     c['victim_rule'] in ('tail', 'arrival', 'host_missing') for c in cells),
                 'Unsupported cell')
    profiles = [i for i, c in enumerate(cells) if c.get('profile_only')]
    base.require(profiles == [0] if profiles else isinstance(plan.get('pinned_gpu_kv_bytes'), int),
                 'Profile must be first and unique, or supply measured pinned capacity')
    base.MANIFEST = plan['package_manifest_sha256']
    base.ARMS = tuple(plan['arms'])
    original_env = base.private_env

    def env(p, cache, arm, memory_file):
        result = original_env(p, cache, 'queue_fund', memory_file)
        # Remove inherited resource selectors, then use only this frozen plan.
        for key in ('A_PROFILE_ONLY', 'A_GPU_KV_BYTES', 'A_INPUT_CASE',
                    'A_MAX_NUM_SEQS', 'A_MAX_SECONDS', 'A_FUNDING_VICTIM_RULE',
                    'A_RECOVERY_LEASE_MODE'):
            result.pop(key, None)
        result.update(A_RECOVERY_LEASE_MODE='q1', A_FUNDING_VICTIM_RULE='tail',
                      PYTHONPATH=p['runtime_overlay'])
        if arm == 'ordinary':  # CUDA-free startup preflight.
            return result
        cell = next(c for c in cells if c['label'] == arm)
        result.update(A_INPUT_CASE=cell['input_case'],
                      A_FUNDING_VICTIM_RULE=cell['victim_rule'])
        if cell.get('profile_only'):
            result['A_PROFILE_ONLY'] = '1'
        else:
            if profiles:
                profile_path = (Path(p['session_dir']) / f"cell-00-{cells[0]['label']}" /
                                'output' / 'normal_capacity_profile.json')
                profile = json.loads(profile_path.read_text())
                base.require(profile['status'] == 'PROFILE_COMPLETE', 'Profile incomplete')
                capacity = profile['pin_kv_cache_memory_bytes']
            else:
                capacity = p['pinned_gpu_kv_bytes']
            base.require(isinstance(capacity, int) and capacity > 0, 'Invalid measured KV capacity')
            result['A_GPU_KV_BYTES'] = str(capacity)
        return result

    base.private_env = env
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(InterruptedError('SIGTERM')))
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('Group budget')))
    return base.run(plan, args.plan)


if __name__ == '__main__':
    raise SystemExit(main())
