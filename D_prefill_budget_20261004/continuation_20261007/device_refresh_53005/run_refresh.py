"""One fixed-cap baseline group on the replacement device; unchanged episodes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
import run_confirmation as base


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def wait_for_idle(design):
    rule = design['after_lock_idle_check']
    began = time.monotonic()
    deadline = began + rule['grace_s']
    first_good = last = None
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeError(f'GPU not confirmed idle within startup grace: {last}')
        value = subprocess.check_output(['nvidia-smi', '--id='+design['gpu_uuid'],
            '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader,nounits'],
            text=True, timeout=remaining).strip()
        utilization, memory = [float(field.strip()) for field in value.split(',')]
        sampled = time.monotonic()
        last = dict(elapsed_s=sampled-began, utilization_percent=utilization, memory_used_mib=memory)
        if sampled >= deadline:
            raise RuntimeError(f'GPU idle query exhausted startup grace: {last}')
        healthy = (0 <= utilization <= rule['max_utilization_percent']
                   and 0 <= memory <= rule['max_memory_used_mib'])
        if healthy:
            if first_good is not None and sampled-began-first_good['elapsed_s'] >= rule['sample_separation_s']:
                return dict(status='IDLE_CONFIRMED', samples=[first_good, last])
            if first_good is None:
                first_good = last
        else:
            first_good = None
        remaining = deadline - time.monotonic()
        if remaining > 0:
            time.sleep(min(rule['sample_separation_s'], remaining))


def resource_check(design, phase):
    assert os.environ.get('D_GPU_UUID') == design['gpu_uuid'], 'Explicit new physical GPU required'
    assert base.UUID == design['gpu_uuid'], 'Parent GPU selection differs from frozen device'
    lock = Path(design['shared_lock'])
    assert lock.stat().st_ino == design['shared_lock_inode'], 'Shared lock inode changed'
    free = shutil.disk_usage('/tmp').free
    required = design['root_between_episode_free_bytes_required' if phase == 'before_episode'
                      else 'root_start_free_bytes_required']
    assert free >= required, ('Root filesystem below frozen reserve', phase, free, required)
    device = subprocess.check_output(['nvidia-smi', '--id='+design['gpu_uuid'],
        '--query-gpu=uuid,driver_version', '--format=csv,noheader'], text=True, timeout=5).strip()
    fields = [value.strip() for value in device.split(',')]
    assert fields == [design['gpu_uuid'], design['expected_driver_version']], ('Device identity changed', device)
    result = dict(phase=phase, gpu_uuid=fields[0], driver_version=fields[1], shared_lock_inode=lock.stat().st_ino,
                  root_free_bytes=free, root_free_bytes_required=required)
    if phase == 'after_lock':
        result['gpu_idle_check'] = wait_for_idle(design)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--wait-lock', type=float, default=7200)
    parser.add_argument('--check-only', action='store_true', help='Verify frozen files only; no device/resource queries')
    args = parser.parse_args()
    design = json.loads((ROOT/'design.json').read_text())
    assert design['status_at_freeze'] == 'PRE_GPU_FROZEN'
    assert design['policies'] == ['fixed1024', 'fixed2048', 'fixed2048', 'fixed1024']
    assert design['warm_policies'] == ['fixed1024', 'fixed2048']
    assert design['legal_caps'] == [1024, 2048]
    assert design['primary_slo'] == dict(ttft_s=4, gap_s=.1, completion_s=20)
    assert design['timeout_s'] == 180 and design['formal_attempt_limit'] == 4
    assert design['max_wait_lock_s'] == 7200 and 0 <= args.wait_lock <= 7200
    assert design['shared_lock'] == '/root/autodl-tmp/moe-research-gpu.lock'
    assert design['root_start_free_bytes_required'] == 2147483648
    assert design['root_between_episode_free_bytes_required'] == 1610612736
    assert design['after_lock_idle_check'] == dict(max_utilization_percent=5,
        max_memory_used_mib=16, consecutive_readings=2, sample_separation_s=.5, grace_s=5)
    assert args.output.name == design['intended_output_name'], 'This design authorizes one named group'
    # Analysis runs locally after readback; those metadata hashes are not GPU
    # execution dependencies and need not be present on the execution host.
    for relative, digest in design['source_sha256'].items():
        assert sha(ROOT/relative) == digest, relative
    assert {'run_refresh.py', 'launch.sh', '../run_confirmation.py', '../prefill_policy.py',
            '../runtime_bootstrap.py'} <= set(design['source_sha256'])
    workload = (ROOT/design['workload']).resolve()
    assert sha(workload) == design['workload_sha256']
    work = json.loads(workload.read_text())
    contract = dict(request_count=len(work), prompt_tokens=sum(len(q['prompt_token_ids']) for q in work),
                    declared_output_tokens=sum(q['max_tokens'] for q in work))
    assert contract == design['input_totals'], 'Frozen input contract changed'
    if args.check_only:
        print(json.dumps(dict(status='FROZEN_GPU_EXECUTION_INPUTS_VERIFIED_NO_GPU',
            resource_check='Not performed in check-only mode', design_sha256=sha(ROOT/'design.json'),
            analysis_files='Hashes retained as metadata; not verified or required by GPU entrypoint',
            input_totals=contract)))
        return 0
    assert not args.output.exists(), 'No overwrite or retry of an existing group'
    before_queue = resource_check(design, 'before_queue')
    after_lock = None
    original_dump = base.dump
    original_episode = base.episode
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
        # The unmodified parent establishes its episode clock only after entry.
        return original_episode(engine, work, policy_name, path, target_ms)
    def dump(path, value):
        nonlocal after_lock
        if path.name == 'status.json' and value.get('status') == 'INITIALIZING':
            # Parent has acquired the shared lock and completed its CUDA cleanup
            # grace; this remains before bootstrap, engine initialization or timing.
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
            value = dict(value, experiment_kind='REPLACEMENT_DEVICE_FIXED_BASELINE_REFRESH',
                frozen_device_refresh_design=design, frozen_device_refresh_design_sha256=sha(ROOT/'design.json'),
                startup_resource_checks=dict(before_queue=before_queue, after_lock=after_lock),
                episode_scope='Exact parent episode and normal arrival loop; no new scheduling/timing instrumentation.')
        original_dump(path, value)
    base.dump = dump
    base.episode = guarded_episode
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
        base.dump = original_dump
        base.episode = original_episode


if __name__ == '__main__':
    raise SystemExit(main())
