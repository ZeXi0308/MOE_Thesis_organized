"""Run a frozen equal-physical-release host-prefix single-intervention victim probe under the existing whole-group GPU lock."""
import argparse
import json
from pathlib import Path
import re
import signal

import run_native_group_base as base


LOCK_PATH = '/root/autodl-tmp/moe-research-gpu.lock'
SOURCE_ROOT = Path('/root/autodl-tmp/moe-a-victim-20261004')
OUTPUT_ROOT = Path('/root/moe-a-victim-20261007')


def configure(plan):
    base.require(plan['kind'] == 'PRO6000_NATIVE_EQUAL_RELEASE_HOST_ONCE', 'Wrong experiment')
    base.require(re.fullmatch(r'[0-9a-f]{64}', plan.get('package_manifest_sha256', '')) is not None,
                 'Package manifest is UNFROZEN or invalid')
    base.require(plan['lock_path'] == LOCK_PATH, 'Must use the existing shared GPU lock')
    base.require(Path(plan['source_package']).parent == SOURCE_ROOT,
                 'Source package must remain in the existing A directory')
    base.require(Path(plan['runtime_overlay']) == SOURCE_ROOT / 'runtime-overlay',
                 'Runtime overlay must remain in the existing A directory')
    base.require(Path(plan['session_dir']).parent == OUTPUT_ROOT,
                 'New session must be under the authorized root-filesystem output directory')
    config = plan['configuration']
    base.require(type(config['max_num_seqs']) is int and config['max_num_seqs'] == 384 and
                 type(config['max_seconds']) is int and config['max_seconds'] == 600,
                 'Explicit fixed max_num_seqs=384 and max_seconds=600 required')
    cells = plan['cells']
    base.require(plan.get('experiment_role') == 'EQUAL_RELEASE_HOST_ONCE_INTERVENTION', 'Wrong role')
    base.require([c['native_victim_rule'] for c in cells] ==
                 ['tail', 'equal_release_host_once', 'equal_release_host_once', 'tail'] and
                 all(c['input_case'] == 'high' and c['profile_only'] is False for c in cells),
                 'This bounded development probe is one fixed ABBA on shared high inputs')
    base.require(cells and len({c['label'] for c in cells}) == len(cells),
                 'Empty group or duplicate cell label')
    base.require(all(isinstance(c['label'], str) and c['label'] != 'ordinary' and
                     re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', c['label']) for c in cells),
                 'Invalid cell label')
    base.require(plan['arms'] == [c['label'] for c in cells], 'Cell order differs')
    base.require(all(c['input_case'] in ('low', 'knee', 'high') and
                     c['native_victim_rule'] in ('tail', 'equal_release_host_once') and
                     c.get('funding_victim_rule', 'tail') == 'tail' and
                     c.get('victim_rule', 'tail') == 'tail' and
                     isinstance(c['profile_only'], bool) for c in cells),
                 'Unsupported cell or non-tail funding rule')
    profiles = [i for i, c in enumerate(cells) if c['profile_only']]
    measured = [c for c in cells if not c['profile_only']]
    base.require(measured and len({(c['input_case'], c['requests']) for c in measured}) == 1,
                 'Comparison cells must use the same workload and request count')
    request_counts = {'low': 128, 'knee': 256, 'high': 320}
    base.require(all(c['requests'] == request_counts[c['input_case']] for c in measured),
                 'Request count differs from the frozen normal-capacity input case')
    base.require(profiles == [0] if profiles else
                 type(plan.get('pinned_gpu_kv_bytes')) is int and plan['pinned_gpu_kv_bytes'] > 0,
                 'Profile must be first and unique, or supply positive measured pinned capacity')
    base.require(not profiles or cells[0]['native_victim_rule'] == 'tail',
                 'Capacity profile must use native tail')
    base.MANIFEST = plan['package_manifest_sha256']
    base.ARMS = tuple(plan['arms'])
    original_env = base.private_env

    def env(p, cache, arm, memory_file):
        result = original_env(p, cache, 'queue_fund', memory_file)
        # The dedicated base removes all inherited A_* values. Explicitly pin
        # the shared mechanisms here; only native victim selection varies.
        result.update(A_NATIVE_OLDEST_ADMISSION='queue_fund', A_NATIVE_OLDEST_REPEAT='1',
                      A_NATIVE_VICTIM_RULE='tail', A_NATIVE_VICTIM_FULL_RUNNING='off',
                      A_NATIVE_VICTIM_CURRENT_GUARD='off', A_SELF_PREEMPT_CONTINUE='off',
                      A_NATIVE_CAPACITY_DEFERRAL='off', A_RECOVERY_LEASE_MODE='q1',
                      A_FUNDING_VICTIM_RULE='tail', A_PROFILE_ONLY='0',
                      A_INPUT_CASE=cells[0]['input_case'],
                      A_MAX_NUM_SEQS=str(config['max_num_seqs']),
                      A_MAX_SECONDS=str(config['max_seconds']), PYTHONPATH=p['runtime_overlay'])
        if arm == 'ordinary':  # CUDA-free source/model preflight only.
            return result
        cell = next(c for c in cells if c['label'] == arm)
        result.update(A_INPUT_CASE=cell['input_case'],
                      A_NATIVE_VICTIM_RULE=cell['native_victim_rule'])
        if cell['profile_only']:
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
            base.require(type(capacity) is int and capacity > 0, 'Invalid measured KV capacity')
            result['A_GPU_KV_BYTES'] = str(capacity)
        return result

    base.private_env = env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    configure(plan)
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(InterruptedError('SIGTERM')))
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('Group budget')))
    return base.run(plan, args.plan)


if __name__ == '__main__':
    raise SystemExit(main())
