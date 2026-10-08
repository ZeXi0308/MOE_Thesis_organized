"""Same frozen cohort comparison; raw output uses an explicitly budgeted tmpfs."""
import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
import run_confirmation as base
import niyama_component.run_component as original_component
from cohort_control.controller import CohortController
from device_refresh_53005.run_refresh import resource_check as original_resource_check


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def derived_episode():
    previous = original_component.ComponentController
    original_component.ComponentController = CohortController
    try:
        return original_component.derived_episode()
    finally:
        original_component.ComponentController = previous



def parse_host_headroom(meminfo, maximum_text, current_text):
    rows = [line.split() for line in meminfo.splitlines() if line.startswith('MemAvailable:')]
    assert len(rows) == 1 and len(rows[0]) == 3 and rows[0][2] == 'kB', 'Malformed/missing MemAvailable'
    def positive_integer(text, name):
        text = text.strip()
        assert text.isascii() and text.isdecimal() and int(text) > 0, ('Malformed/nonpositive host-memory value', name, text)
        return int(text)
    available = positive_integer(rows[0][1], 'MemAvailable')*1024
    current = positive_integer(current_text, 'memory.current')
    maximum_text = maximum_text.strip()
    if maximum_text == 'max':
        maximum, cgroup_free, effective = None, None, available
    else:
        maximum = positive_integer(maximum_text, 'memory.max')
        cgroup_free = maximum-current
        assert cgroup_free > 0, ('Nonpositive cgroup headroom', maximum, current)
        effective = min(available,cgroup_free)
    return dict(mem_available_bytes=available,cgroup_memory_max_bytes=maximum,
        cgroup_memory_current_bytes=current,cgroup_headroom_bytes=cgroup_free,
        cgroup_limit_unbounded=maximum is None,effective_host_headroom_bytes=effective)


def validate_output_path(output, design):
    exact = '/dev/shm/moe-d-prefill-continuation-20261007/cohort53005_03'
    assert design['intended_output_path'] == exact and design['intended_output_name'] == 'cohort53005_03'
    assert design['intended_log_path'] == exact+'.log'
    assert output.is_absolute() and str(output) == exact, 'Only the frozen absolute tmpfs output path is authorized'


def own_output_log_bytes(design):
    # Only this run's subtree and its one designated log are inspected.
    output, log = Path(design['intended_output_path']),Path(design['intended_log_path'])
    total = 0
    assert not output.is_symlink() and not log.is_symlink(), 'Own output/log must not be symlinks'
    if output.exists():
        assert output.is_dir(), 'Own output is not a directory'
        for directory, dirs, files in os.walk(output,followlinks=False):
            for name in dirs:
                assert not (Path(directory)/name).is_symlink(), 'Output subtree contains a symlink'
            for name in files:
                path = Path(directory)/name
                assert path.is_file() and not path.is_symlink(), 'Output subtree contains a nonregular file'
                total += path.stat().st_size
    if log.exists():
        assert log.is_file(), 'Own log is not a regular file'
        total += log.stat().st_size
    assert total <= design['own_output_log_bytes_limit'], ('Own output/log exceeds frozen allowance',total,design['own_output_log_bytes_limit'])
    return total


def resource_check(design, phase):
    # Same GPU/driver/shared-lock and after-lock idle checks, with the explicitly
    # changed root reserve in this new design. Never called inside an episode.
    result = original_resource_check(design,phase)
    shm, root = Path('/dev/shm'),Path('/')
    assert shm.is_dir() and not shm.is_symlink(), 'Existing /dev/shm directory required'
    mounts = [line.split() for line in Path('/proc/mounts').read_text().splitlines()]
    assert any(len(row)>=3 and row[1]=='/dev/shm' and row[2]=='tmpfs' for row in mounts), '/dev/shm is not a mounted tmpfs'
    assert shm.stat().st_dev != root.stat().st_dev, 'tmpfs output must be off the root filesystem'
    output = Path(design['intended_output_path'])
    assert output.parent.is_dir() and output.parent.resolve()==output.parent, 'Frozen tmpfs output parent must exist without symlinks'
    assert output.parent.stat().st_dev == shm.stat().st_dev, 'Output parent is not on /dev/shm'
    free = shutil.disk_usage(shm).free
    assert free >= design['shm_free_bytes_required'], ('Insufficient tmpfs reservation',free,design['shm_free_bytes_required'])
    host = parse_host_headroom(Path('/proc/meminfo').read_text(),
        Path('/sys/fs/cgroup/memory.max').read_text(),Path('/sys/fs/cgroup/memory.current').read_text())
    assert host['effective_host_headroom_bytes'] >= design['effective_host_headroom_bytes_required'], ('Insufficient effective Host headroom',host)
    own_bytes = own_output_log_bytes(design)
    result.update(shm_free_bytes=free,shm_free_bytes_required=design['shm_free_bytes_required'],
        effective_host_headroom_requirement=design['effective_host_headroom_bytes_required'],
        host_memory=host,own_output_log_prior_bytes=own_bytes,own_output_log_bytes_limit=design['own_output_log_bytes_limit'],
        split_storage_plan=dict(root_reserve_bytes=design['root_start_free_bytes_required'],
            tmpfs_reserve_bytes=design['shm_free_bytes_required'],startup_total_reserve_bytes=design['root_start_free_bytes_required']+design['shm_free_bytes_required'],
            scope='Explicit new 1.5GiB root plus0.5GiB tmpfs plan; not the old2GiB-root requirement. No quota or guarantee against other processes.'))
    return result

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--wait-lock', type=float, default=7200)
    parser.add_argument('--check-only', action='store_true', help='Verify execution files only; no device/resource queries')
    args = parser.parse_args()
    design = json.loads((ROOT/'design.json').read_text())
    assert design['status_at_freeze'] == 'PRE_GPU_FROZEN'
    assert design['policies'] == ['fixed1024', 'progress_floor', 'cohort_protection', 'cohort_protection', 'progress_floor', 'fixed1024']
    assert design['warm_policies'] == ['fixed1024', 'progress_floor', 'cohort_protection']
    assert design['primary_slo'] == dict(ttft_s=4, gap_s=.1, completion_s=20)
    assert design['batch_token_deadline_s'] == (20-4)/512
    assert design['normal_total_tiers'] == [128]+list(range(256,2049,256))+[2552]
    assert design['low_memory_total_tiers'] == list(range(128,513,128))
    assert design['prediction_margin'] == 1.2
    assert design['progress_floor_cap'] == 1024
    assert design['cohort_cost_model_regime'] == 'le512'
    assert design['cohort_cost_margin'] == 1.0 and design['cohort_protection_headroom_s'] == 0.0
    assert design['timeout_s'] == 180 and design['formal_attempt_limit'] == 6
    assert design['warm_attempt_limit'] == 3
    assert design['max_wait_lock_s'] == 7200 and 0 <= args.wait_lock <= 7200
    assert design['shared_lock'] == '/root/autodl-tmp/moe-research-gpu.lock'
    assert design['root_start_free_bytes_required'] == 1610612736
    assert design['root_between_episode_free_bytes_required'] == 1610612736
    assert design['after_lock_idle_check'] == dict(max_utilization_percent=5,
        max_memory_used_mib=16, consecutive_readings=2, sample_separation_s=.5, grace_s=5)
    assert design['shm_free_bytes_required'] == 536870912
    assert design['own_output_log_bytes_limit'] == 524288000
    assert design['effective_host_headroom_bytes_required'] == 2147483648
    validate_output_path(args.output,design)
    required_sources = {'run_shm.py', 'launch.sh', '../cohort_control/controller.py', '../cohort_control/cohort_cost.py', '../niyama_component/run_component.py',
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
    CohortController.design = design
    CohortController.calibration = calibration
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
            value = dict(value, experiment_kind='COHORT_PREFILL_CAP_DEVELOPMENT',
                frozen_component_design=design, frozen_component_design_sha256=sha(ROOT/'design.json'),
                startup_resource_checks=dict(before_queue=before_queue, after_lock=after_lock),
                episode_scope='Exact original component derived episode with only controller class replaced for all three policies; resource guards run outside its clock.')
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
            # Includes the final raw, final status and log already written by base.main.
            try:
                own_output_log_bytes(design)
            except Exception as exc:
                original_dump(args.output/'status.json', dict(
                    status='RESOURCE_BUDGET_FAILED_AFTER_EPISODES', pid=os.getpid(),
                    completed_episodes=episode_count, reason=repr(exc)))
                raise
        return result
    finally:
        base.dump, base.episode = original_dump, original_episode


if __name__ == '__main__':
    raise SystemExit(main())
