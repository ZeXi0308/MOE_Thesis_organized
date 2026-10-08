"""Replacement-device entrypoint for the unchanged existing Niyama component."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
import run_confirmation as base
from niyama_component.run_component import ComponentController, derived_episode
from device_refresh_53005.run_refresh import resource_check


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--wait-lock', type=float, default=7200)
    parser.add_argument('--check-only', action='store_true', help='Verify execution files only; no device/resource queries')
    args = parser.parse_args()
    design = json.loads((ROOT/'design.json').read_text())
    assert design['status_at_freeze'] == 'PRE_GPU_FROZEN'
    assert design['policies'] == ['fixed1024', 'niyama_component', 'niyama_component', 'fixed1024']
    assert design['warm_policies'] == ['fixed1024', 'niyama_component']
    assert design['primary_slo'] == dict(ttft_s=4, gap_s=.1, completion_s=20)
    assert design['batch_token_deadline_s'] == (20-4)/512
    assert design['normal_total_tiers'] == [128]+list(range(256,2049,256))+[2552]
    assert design['low_memory_total_tiers'] == list(range(128,513,128))
    assert design['prediction_margin'] == 1.2
    assert design['timeout_s'] == 180 and design['formal_attempt_limit'] == 4
    assert design['warm_attempt_limit'] == 2
    assert design['max_wait_lock_s'] == 7200 and 0 <= args.wait_lock <= 7200
    assert design['shared_lock'] == '/root/autodl-tmp/moe-research-gpu.lock'
    assert design['root_start_free_bytes_required'] == 2147483648
    assert design['root_between_episode_free_bytes_required'] == 1610612736
    assert design['after_lock_idle_check'] == dict(max_utilization_percent=5,
        max_memory_used_mib=16, consecutive_readings=2, sample_separation_s=.5, grace_s=5)
    assert args.output.name == design['intended_output_name'], 'This design authorizes one named group'
    required_sources = {'run_component.py', 'launch.sh', '../niyama_component/run_component.py',
        '../device_refresh_53005/run_refresh.py', '../run_confirmation.py',
        '../prefill_policy.py', '../runtime_bootstrap.py'}
    assert set(design['source_sha256']) == required_sources, 'Only actual execution dependencies belong in this set'
    for relative, digest in design['source_sha256'].items():
        assert sha(ROOT/relative) == digest, relative
    workload = (ROOT/design['workload']).resolve()
    assert sha(workload) == design['workload_sha256']
    work = json.loads(workload.read_text())
    contract = dict(request_count=len(work), prompt_tokens=sum(len(q['prompt_token_ids']) for q in work),
                    declared_output_tokens=sum(q['max_tokens'] for q in work))
    assert contract == design['input_totals'], 'Frozen input contract changed'
    calibration_path = (ROOT/design['calibration_file']).resolve()
    assert sha(calibration_path) == design['calibration_sha256']
    calibration = json.loads(calibration_path.read_text())
    assert not calibration['fit_errors']
    assert set(calibration['runtime_models']) == {'le512', 'gt512'}
    ComponentController.design = design
    ComponentController.calibration = calibration
    component_episode, source = derived_episode()
    if args.check_only:
        print(json.dumps(dict(status='FROZEN_GPU_EXECUTION_INPUTS_VERIFIED_NO_GPU',
            resource_check='Not performed in check-only mode', design_sha256=sha(ROOT/'design.json'),
            analysis_files='Local metadata only; not required on execution host',
            calibration_sha256=sha(calibration_path), input_totals=contract)))
        return 0
    assert not args.output.exists(), 'No overwrite or retry of an existing group'
    before_queue = resource_check(design, 'before_queue')
    after_lock = None
    original_dump, original_episode = base.dump, base.episode
    expected_cells = [('warm_'+policy, policy) for policy in design['warm_policies']]
    expected_cells += [(f'{i:02d}_{policy}', policy) for i, policy in enumerate(design['policies'])]
    episode_count = 0
    def guarded_episode(engine, work, policy_name, path, target_ms):
        nonlocal episode_count
        assert episode_count < len(expected_cells), 'No extra episode or automatic retry permitted'
        assert (path.name, policy_name) == expected_cells[episode_count]
        check = resource_check(design, 'before_episode')
        print('EPISODE_RESOURCE_CHECK '+json.dumps(dict(cell=path.name, **check)), flush=True)
        episode_count += 1
        # This guard precedes the exact imported component episode's clock.
        return component_episode(engine, work, policy_name, path, target_ms)
    def dump(path, value):
        nonlocal after_lock
        if path.name == 'status.json' and value.get('status') == 'INITIALIZING':
            try:
                after_lock = resource_check(design, 'after_lock')
            except Exception as exc:
                original_dump(path, dict(status='RESOURCE_BLOCKED_NO_GPU_INITIALIZED',
                    pid=os.getpid(), reason=repr(exc)))
                raise
        if path.name == 'engine_args.json':
            assert value == design['engine_args'], 'Normal-capacity engine configuration changed'
        if path.name == 'protocol.json':
            assert after_lock is not None
            assert value['policies'] == design['policies'] and value['warm_policies'] == design['warm_policies']
            assert value['slo'] == design['primary_slo'] and value['workload_sha256'] == design['workload_sha256']
            value = dict(value, experiment_kind='NIYAMA_CHUNK_COMPONENT_DEVELOPMENT',
                frozen_component_design=design, frozen_component_design_sha256=sha(ROOT/'design.json'),
                startup_resource_checks=dict(before_queue=before_queue, after_lock=after_lock),
                episode_scope='Exact imported original component episode for fixed and dynamic policies; resource guards run outside its clock.')
            (path.parent/'derived_episode.py').write_text(source)
            original_dump(path.parent/'calibration.json', calibration)
        original_dump(path, value)
    base.dump, base.episode = dump, guarded_episode
    sys.argv = [str(ROOT.parent/'run_confirmation.py'), '--output', str(args.output.resolve()),
        '--workload', str(workload), '--policies', ','.join(design['policies']),
        '--warm-policies', ','.join(design['warm_policies']), '--target-ms', '24',
        '--wait-lock', str(args.wait_lock)]
    try:
        result = base.main()
        if result == 0:
            assert episode_count == len(expected_cells)
        return result
    finally:
        base.dump, base.episode = original_dump, original_episode


if __name__ == '__main__':
    raise SystemExit(main())
