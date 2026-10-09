"""One bounded head-prefill action probe; no online rollout or model fitting."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import traceback

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
import run_confirmation as base
import niyama_component.run_component as original_component
from cohort_trace_53005.run_shm import resource_check, own_output_log_bytes
from action_rollout.probe_controller import ProbeController


class GroupTimeout(TimeoutError):
    pass


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def derived_episode():
    previous = original_component.ComponentController
    original_component.ComponentController = ProbeController
    try:
        return original_component.derived_episode()
    finally:
        original_component.ComponentController = previous


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--wait-lock', type=float, default=1800)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    design_path = ROOT/'probe_design.json'
    design = json.loads(design_path.read_text())
    assert design['status_at_freeze'] == 'PRE_GPU_FROZEN'
    assert design['policies'] == ['fixed1024', 'head2048', 'head2048', 'fixed1024']
    assert design['warm_policies'] == ['fixed1024', 'fixed2048']
    assert design['legal_caps'] == [1024, 2048]
    assert design['primary_slo'] == dict(ttft_s=4, gap_s=.1, completion_s=20)
    assert design['timeout_s'] == 180 and design['group_timeout_s'] == 900
    assert design['formal_attempt_limit'] == 4 and design['warm_attempt_limit'] == 2
    assert design['max_intervention_steps'] == 4
    assert design['max_wait_lock_s'] == 1800 and 0 <= args.wait_lock <= 1800
    exact = '/dev/shm/moe-d-prefill-continuation-20261007/rollout53005_01'
    assert str(args.output) == design['intended_output_path'] == exact
    assert design['intended_output_name'] == 'rollout53005_01'
    assert design['intended_log_path'] == exact+'.log'
    assert design['shared_lock'] == '/root/autodl-tmp/moe-research-gpu.lock'
    assert design['root_start_free_bytes_required'] == design['root_between_episode_free_bytes_required'] == 1610612736
    assert design['shm_free_bytes_required'] == 536870912
    assert design['own_output_log_bytes_limit'] == 524288000
    assert design['effective_host_headroom_bytes_required'] == 2147483648
    assert design['after_lock_idle_check'] == dict(max_utilization_percent=5,
        max_memory_used_mib=16, consecutive_readings=2, sample_separation_s=.5, grace_s=5)
    required = {'run_probe.py', 'launch.sh', 'probe_controller.py', '../prefill_demand/controller.py',
        '../cohort_trace_53005/run_shm.py', '../cohort_control/controller.py', '../cohort_control/cohort_cost.py',
        '../niyama_component/run_component.py', '../device_refresh_53005/run_refresh.py',
        '../run_confirmation.py', '../prefill_policy.py', '../runtime_bootstrap.py'}
    assert set(design['source_sha256']) == required
    for relative, digest in design['source_sha256'].items():
        assert sha(ROOT/relative) == digest, relative
    workload = (ROOT/design['workload']).resolve()
    assert sha(workload) == design['workload_sha256']
    work = json.loads(workload.read_text())
    contract = dict(request_count=len(work), prompt_tokens=sum(len(q['prompt_token_ids']) for q in work),
        declared_output_tokens=sum(q['max_tokens'] for q in work))
    assert contract == design['input_totals']
    assert design['calibration_file'] == '../niyama_component/calibration.json'
    calibration_path = ROOT/design['calibration_file']
    assert sha(calibration_path) == design['calibration_sha256']
    calibration = json.loads(calibration_path.read_text())
    assert not calibration['fit_errors'] and set(calibration['runtime_models']) == {'le512', 'gt512'}
    ProbeController.design, ProbeController.calibration = design, calibration
    component_episode, source = derived_episode()
    if args.check_only:
        print(json.dumps(dict(status='FROZEN_GPU_EXECUTION_INPUTS_VERIFIED_NO_GPU',
            design_sha256=sha(design_path), input_totals=contract,
            episode='Existing component episode; only controller class replaced; no online rollout',
            resources='Not queried; original compiler/cache/TMPDIR paths unchanged')))
        return 0
    assert base.UUID == design['gpu_uuid'], 'Explicit authorized physical GPU required'
    assert not args.output.exists(), 'No overwrite or automatic retry'
    before_queue = resource_check(design, 'before_queue')
    after_lock = None
    original_dump, original_episode, original_argv = base.dump, base.episode, sys.argv
    expected = [('warm_'+p, p) for p in design['warm_policies']]
    expected += [(f'{i:02d}_{p}', p) for i, p in enumerate(design['policies'])]
    episode_count, coverage = 0, {}
    previous_alarm, previous_timer, timer_started = None, None, False

    def group_timeout(signum, frame):
        raise GroupTimeout('900-second group budget exhausted after INITIALIZING; no retry')

    def guarded_episode(engine, work, policy, path, target_ms):
        nonlocal episode_count
        assert episode_count < len(expected) and (path.name, policy) == expected[episode_count]
        if episode_count >= 2:
            assert set(coverage) == {'fixed1024', 'fixed2048'}
            assert all(v['actual_P_equals_cap_steps'] > 0 for v in coverage.values()), 'Warm cap never saturated; no extra warm retry'
        check = resource_check(design, 'before_episode')
        print('EPISODE_RESOURCE_CHECK '+json.dumps(dict(cell=path.name, **check)), flush=True)
        episode_count += 1
        result = component_episode(engine, work, policy, path, target_ms)
        if path.name.startswith('warm_fixed'):
            raw = json.loads((path/'raw.json').read_text())
            cap = int(policy[5:])
            saturated = [s for s in raw['steps'] if s['end_s'] is not None
                and s['component_decision']['forward_completed'] and s['prefill_tokens'] == cap]
            coverage[policy] = dict(cap=cap, actual_P_equals_cap_steps=len(saturated),
                mixed_actual_P_equals_cap_steps=sum(s['decode_tokens'] > 0 for s in saturated))
            protocol_path = path.parent/'protocol.json'
            protocol = json.loads(protocol_path.read_text())
            protocol['warm_actual_saturation'] = dict(coverage)
            original_dump(protocol_path, protocol)
            print('WARM_ACTUAL_SATURATION '+json.dumps(coverage[policy]), flush=True)
        if policy == 'head2048':
            raw = json.loads((path/'raw.json').read_text())
            summary = raw['component']
            actions = [s for s in raw['steps'] if s['component_decision']['probe_action_requested']]
            assert len(actions) == summary['probe_action_decisions_requested'] <= 4
            assert all(s['component_decision']['requested_cap'] == 2048 for s in actions)
            assert all((s['component_decision']['requested_cap'] > 1024)
                == bool(s['component_decision']['probe_action_requested']) for s in raw['steps'])
            actual = sum(s['end_s'] is not None and s['component_decision']['forward_completed']
                         and s['prefill_tokens'] > 1024 for s in actions)
            assert actual == summary['probe_actual_P_above_1024_action_forwards']
            if actions:
                assert summary['probe_triggered'] and summary['probe_guard_passed']
                assert sum(bool(s['component_decision']['probe_trigger_this_step']) for s in raw['steps']) == 1
            print('PROBE_ACTUAL_INTERVENTIONS '+json.dumps(dict(cell=path.name,
                triggered=summary['probe_triggered'], requested_steps=len(actions), actual_P_above_1024=actual)), flush=True)
            if path.name == '01_head2048':
                assert summary['probe_triggered'] and summary['probe_guard_passed'] and actual > 0, \
                    'ZERO_ACTUAL_ACTION_STOP_AFTER_FIRST_PAIR; preserve completed raw, no reverse retry'
        return result

    def dump(path, value):
        nonlocal after_lock, previous_alarm, previous_timer, timer_started
        if path.name == 'status.json' and value.get('status') == 'INITIALIZING':
            assert not timer_started
            previous_timer = signal.getitimer(signal.ITIMER_REAL)
            previous_alarm = signal.signal(signal.SIGALRM, group_timeout)
            timer_started = True
            signal.alarm(design['group_timeout_s'])
            try:
                after_lock = resource_check(design, 'after_lock')
            except GroupTimeout:
                raise
            except Exception as exc:
                original_dump(path, dict(status='RESOURCE_BLOCKED_NO_GPU_INITIALIZED', pid=os.getpid(), reason=repr(exc)))
                raise
        if path.name == 'engine_args.json':
            assert value == design['engine_args'], 'Normal-capacity engine configuration changed'
        if path.name == 'protocol.json':
            assert after_lock is not None
            assert value['policies'] == design['policies'] and value['warm_policies'] == design['warm_policies']
            assert value['slo'] == design['primary_slo'] and value['workload_sha256'] == design['workload_sha256']
            value = dict(value, experiment_kind='HEAD_PREFILL_ACTION_PROBE_DEVELOPMENT',
                frozen_component_design=design, frozen_component_design_sha256=sha(design_path),
                startup_resource_checks=dict(before_queue=before_queue, after_lock=after_lock),
                group_timeout_s=design['group_timeout_s'], group_timeout_origin='INITIALIZING after common lock; waiting excluded',
                warm_actual_saturation_requirement='Both fixed warms must actually saturate before formal; no extra warm.',
                episode_scope='Original component episode; only controller class replaced. No online rollout/cost fitting.')
            (path.parent/'derived_episode.py').write_text(source)
            original_dump(path.parent/'calibration.json', calibration)
        original_dump(path, value)

    base.dump, base.episode = dump, guarded_episode
    sys.argv = [str(ROOT.parent/'run_confirmation.py'), '--output', str(args.output),
        '--workload', str(workload), '--policies', ','.join(design['policies']),
        '--warm-policies', ','.join(design['warm_policies']), '--target-ms', '24', '--wait-lock', str(args.wait_lock)]
    try:
        result = base.main()
        if result == 0:
            assert episode_count == 6
            try:
                own_output_log_bytes(design)
            except Exception as exc:
                original_dump(args.output/'status.json', dict(status='RESOURCE_BUDGET_FAILED_AFTER_EPISODES',
                    pid=os.getpid(), completed_episodes=episode_count, reason=repr(exc)))
                raise
        return result
    except GroupTimeout:
        original_dump(args.output/'status.json', dict(status='GROUP_TIMEOUT', pid=os.getpid(),
            group_timeout_s=900, attempted_episodes=episode_count, traceback=traceback.format_exc()))
        raise
    finally:
        if timer_started:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous_alarm)
            signal.setitimer(signal.ITIMER_REAL, *previous_timer)
        base.dump, base.episode, sys.argv = original_dump, original_episode, original_argv


if __name__ == '__main__':
    raise SystemExit(main())
