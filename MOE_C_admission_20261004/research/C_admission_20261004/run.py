#!/usr/bin/env python3
"""One serial, host-locked development + reverse-order admission experiment."""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import time
import traceback

ROOT = Path(__file__).resolve().parent
LOCK = '/root/autodl-tmp/moe-research-gpu.lock'
OPPORTUNITY_LOCK_IDENTITY = (2304, 4312099778)  # Observed common westd:53005 lock; never create a replacement.
UUID = 'GPU-51b8e4bb-27b8-4b82-5254-7317aae7298c'
KV_BYTES = 4097 * 2097152


def dump(p, value):
    p.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def read(p):
    return json.loads(p.read_text())


class GPUProcessesPresent(RuntimeError):
    def __init__(self, state):
        self.state = state
        super().__init__('Other GPU process: '+state['processes'])


class GPUIdentityMismatch(RuntimeError):
    def __init__(self, device):
        self.state = dict(device=device, processes=None, unix=time.time())
        super().__init__('Expected one GPU '+UUID+', observed: '+device)


def gpu_state(allow_own=False, *, deadline=None):
    def query(fields):
        kwargs = dict(text=True)
        if deadline is not None:
            remaining = deadline-time.monotonic()
            if remaining <= 0:
                raise TimeoutError('GPU handoff query deadline reached')
            kwargs['timeout'] = remaining
        return subprocess.check_output(['nvidia-smi', fields, '--format=csv,noheader'], **kwargs).strip()

    # Identity is checked on every poll, including polls that find busy processes.
    device = query('--query-gpu=uuid,name,memory.total,memory.used')
    if not device.startswith(UUID+',') or len(device.splitlines()) != 1:
        raise GPUIdentityMismatch(device)
    rows = query('--query-compute-apps=pid,process_name,used_memory')
    pids = [int(x.split(',')[0]) for x in rows.splitlines() if x]
    state = dict(device=device, processes=rows, unix=time.time())
    if any(pid != os.getpid() or not allow_own for pid in pids):
        raise GPUProcessesPresent(state)
    return state


def wait_idle_gpu(out, timeout_s=15., poll_s=.5):
    """Bound a prior lock-holder's process teardown; never allow our own PID."""
    assert 0 < timeout_s <= 15. and 0 < poll_s <= 1.
    started = time.monotonic()
    deadline = started+timeout_s
    report = dict(status='WAITING', expected_uuid=UUID, pid=os.getpid(),
        started_unix=time.time(), timeout_s=timeout_s, poll_interval_s=poll_s, polls=[])
    try:
        while True:
            if time.monotonic() >= deadline:
                raise TimeoutError('GPU compute processes did not clear within the handoff deadline')
            poll_started = time.monotonic()
            row = dict(start_elapsed_s=poll_started-started, unix=time.time())
            try:
                state = gpu_state(allow_own=False, deadline=deadline)
                row.update(status='IDLE', state=state)
            except GPUProcessesPresent as exc:
                row.update(status='BUSY', state=exc.state)
            except BaseException as exc:
                row.update(status='ERROR', error=repr(exc), state=getattr(exc, 'state', None))
                raise
            finally:
                row['duration_s'] = time.monotonic()-poll_started
                report['polls'].append(row)
                report['elapsed_s'] = time.monotonic()-started
                dump(out/'gpu-handoff.json', report)
            if row['status'] == 'IDLE':
                report['status'] = 'IDLE'
                return state
            remaining = deadline-time.monotonic()
            if remaining > 0:
                time.sleep(min(poll_s, remaining))
    except BaseException as exc:
        report.update(status='FAILED', error=repr(exc))
        raise
    finally:
        report.update(elapsed_s=time.monotonic()-started, finished_unix=time.time())
        dump(out/'gpu-handoff.json', report)


def subset(work, start, count, gap=.2):
    return dict(work, source_requests=work['source_requests'][start:start+count],
                actual_prompt_token_ids=work['actual_prompt_token_ids'][start:start+count],
                arrival_traces_s={'steady': [i*gap for i in range(count)]})


def pressure_population(dev, test):
    selected = dict(dev,
        schema='olmoe-natural-cadence-pressure-reuse-v1',
        source_requests=[row for pair in zip(dev['source_requests'], test['source_requests']) for row in pair],
        actual_prompt_token_ids=[tokens for pair in zip(dev['actual_prompt_token_ids'],
            test['actual_prompt_token_ids']) for tokens in pair],
        arrival_traces_s={'steady': [.1*i for i in range(384)]},
        arrival_rule='384 requests, dev[i] then test[i]; external arrival at 0.1*i seconds. '
            'Same population and trace used for fixed-cap selection and exploratory comparison.')
    assert len(selected['source_requests']) == len(selected['actual_prompt_token_ids']) == 384
    assert len({r['request_id'] for r in selected['source_requests']}) == 384
    lengths = list(map(len, selected['actual_prompt_token_ids']))
    assert all(0 < n and n+1024 <= 4096 for n in lengths)
    selected['input_stats'] = dict(requests=384, minimum=min(lengths), maximum=max(lengths),
        mean=statistics.mean(lengths), total=sum(lengths), short_lt1536=sum(n<1536 for n in lengths),
        medium_1536_2559=sum(1536<=n<2560 for n in lengths), long_ge2560=sum(n>=2560 for n in lengths),
        prompt_rounded_pages=sum(math.ceil(n/16) for n in lengths),
        output_limit_rounded_pages=sum(math.ceil((n+1024)/16) for n in lengths))
    return selected


def pressure_burst_population(steady):
    return dict(steady, schema='olmoe-natural-cadence-pressure-burst-reuse-v1',
        arrival_traces_s={'steady': [.1*i if i < 256 else 55. for i in range(384)]},
        arrival_rule='Same ordered 384 requests as pressure development: first 256 arrive at '
            '0.1*i seconds; final 128 all arrive at 55.0 seconds. Fixed external trace shared by '
            'all policies, independent of runtime state. Fixed cap transferred from steady development.')


def pressure_cadence_population(steady):
    return dict(steady, schema='olmoe-natural-cadence-pressure-02s-observation-v1',
        arrival_traces_s={'steady': [.2*i for i in range(384)]},
        arrival_rule='Same ordered 384 requests as pressure exploration; external arrival at '
            '0.2*i seconds, a 76.6s window. One observation-only KV256 baseline at half the '
            'previous offered rate; fixed trace independent of runtime state. No cap selection.')


def validate_burst_selection(accepted, status, pro_input_sha, steady):
    assert accepted['profile'] == 'pro-pressure-dev'
    assert accepted['pro_inputs_file_sha256'] == pro_input_sha
    assert accepted['population'] == 384
    assert accepted['dev_caps'] == [128, 192, 256]
    assert sorted(row['cap'] for row in accepted['scores']) == accepted['dev_caps']
    assert accepted['selected_cap'] in accepted['dev_caps']
    assert accepted['selected_pressure_gap_s'] == .1
    steady_trace = dict(request_ids=[r['request_id'] for r in steady['source_requests']],
        actual_prompt_token_ids=steady['actual_prompt_token_ids'], arrival_traces_s=steady['arrival_traces_s'])
    assert accepted['trace_sha256'] == hashlib.sha256(json.dumps(steady_trace, sort_keys=True).encode()).hexdigest()
    assert accepted['workload_sha256'] == hashlib.sha256(json.dumps(steady, sort_keys=True).encode()).hexdigest()
    assert status['status'] == 'COMPLETE' and status['selected_cap'] == accepted['selected_cap']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--profile', choices=('controlled', 'pro-dev', 'pro-test',
        'pro-pressure-dev', 'pro-pressure-test', 'pro-pressure-burst-test',
        'pro-opportunity-scan', 'pro-opportunity-probe', 'pro-opportunity-progress',
        'pro-opportunity-reservation-scan', 'pro-opportunity-reservation-probe',
        'pro-opportunity-headroom-probe', 'pro-opportunity-gc-liveness',
        'pro-opportunity-simple-recheck', 'pro-opportunity-native-cap-recheck',
        'pro-opportunity-cadence-scan', 'pro-opportunity-declared-budget',
        'pro-opportunity-declared-budget-matched', 'pro-opportunity-mc-budget',
        'pro-opportunity-mc-budget-ablation', 'pro-opportunity-mc-budget-phase'), default='controlled')
    parser.add_argument('--selection', type=Path)
    parser.add_argument('--liveness-history', type=Path)
    parser.add_argument('--wait-lock-seconds', type=int, default=0)
    args = parser.parse_args()
    if not 0 <= args.wait_lock_seconds <= 3600:
        parser.error('A single bounded lock wait must be between 0 and 3600 seconds')
    if args.profile in ('pro-test', 'pro-pressure-test', 'pro-pressure-burst-test') and args.selection is None:
        parser.error(args.profile+' requires the matching completed dev selection.json')
    pro = args.profile != 'controlled'
    opportunity = args.profile.startswith('pro-opportunity-')
    progress_probe = args.profile == 'pro-opportunity-progress'
    reservation_scan = args.profile == 'pro-opportunity-reservation-scan'
    reservation_probe = args.profile == 'pro-opportunity-reservation-probe'
    headroom_probe = args.profile == 'pro-opportunity-headroom-probe'
    gc_liveness = args.profile == 'pro-opportunity-gc-liveness'
    native_cap_recheck = args.profile == 'pro-opportunity-native-cap-recheck'
    cadence_scan = args.profile == 'pro-opportunity-cadence-scan'
    declared_budget_matched = args.profile == 'pro-opportunity-declared-budget-matched'
    declared_budget_comparison = args.profile == 'pro-opportunity-declared-budget' or declared_budget_matched
    mc_budget_ablation = args.profile == 'pro-opportunity-mc-budget-ablation'
    mc_budget_phase = args.profile == 'pro-opportunity-mc-budget-phase'
    mc_budget = args.profile == 'pro-opportunity-mc-budget' or mc_budget_ablation or mc_budget_phase
    budget_experiment = declared_budget_comparison or mc_budget
    native_guard_matches_cap = native_cap_recheck or cadence_scan or budget_experiment
    simple_recheck = args.profile in ('pro-opportunity-simple-recheck',
                                     'pro-opportunity-native-cap-recheck')
    observe_gc = gc_liveness or simple_recheck or cadence_scan or budget_experiment
    if gc_liveness and args.liveness_history is None:
        parser.error('pro-opportunity-gc-liveness requires --liveness-history')
    reservation = reservation_scan or reservation_probe or headroom_probe
    pressure = args.profile.startswith('pro-pressure-') or opportunity
    burst = args.profile == 'pro-pressure-burst-test'
    usable_pages, engine_max = (32768, 256) if pro else (4096, 32)
    kv_bytes, kv_floor = (usable_pages+1)*2097152, math.ceil(usable_pages*.10)
    args.output.mkdir(parents=True, exist_ok=False)
    out = args.output
    lock = open(LOCK, 'r+' if opportunity else 'a+')
    if opportunity:
        identity = os.fstat(lock.fileno())
        assert (identity.st_dev, identity.st_ino) == OPPORTUNITY_LOCK_IDENTITY
    wait_start = time.monotonic()
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        if args.wait_lock_seconds == 0:
            dump(out/'status.json', dict(status='LOCK_BUSY_NO_GPU_NO_WAITER', pid=os.getpid()))
            lock.close()
            return 75
        deadline = wait_start + args.wait_lock_seconds
        dump(out/'status.json', dict(status='WAITING_COMMON_LOCK_NO_GPU', pid=os.getpid(),
            deadline_unix=time.time()+max(0., deadline-time.monotonic())))

        class LockWaitTimeout(Exception):
            pass

        def lock_wait_timeout(signum, frame):
            raise LockWaitTimeout

        previous_alarm_handler = signal.signal(signal.SIGALRM, lock_wait_timeout)
        try:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise LockWaitTimeout
            signal.setitimer(signal.ITIMER_REAL, remaining)
            # Join the kernel's wait queue once; SIGALRM bounds this same wait.
            fcntl.flock(lock, fcntl.LOCK_EX)
        except LockWaitTimeout:
            dump(out/'status.json', dict(status='LOCK_WAIT_TIMEOUT_NO_GPU', pid=os.getpid()))
            lock.close()
            return 75
        except BaseException:
            lock.close()
            raise
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous_alarm_handler)
    lock_identity = os.fstat(lock.fileno())
    assert (lock_identity.st_dev, lock_identity.st_ino) == (os.stat(LOCK).st_dev, os.stat(LOCK).st_ino)
    dump(out/'lock.json', dict(pid=os.getpid(), path=LOCK,
                              device=lock_identity.st_dev, inode=lock_identity.st_ino,
                              acquired_unix=time.time(),
                              waited_seconds=time.monotonic()-wait_start))
    dump(out/'status.json', dict(status='RUNNING', pid=os.getpid()))
    engine = None
    try:
        if headroom_probe or observe_gc:
            disk = os.statvfs(out)
            free_output_bytes = disk.f_bavail * disk.f_frsize
            if free_output_bytes < 1536 * 1024**2:
                raise RuntimeError('Insufficient output disk space before GPU initialization: '
                    + str(free_output_bytes) + ' bytes available; previous four-arm raw data used767MB.')
        dump(out/'gpu-before.json', wait_idle_gpu(out))
        dump(out/'source.json', {p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in ROOT.glob('*.py')})
        os.environ.update(CUDA_VISIBLE_DEVICES=UUID, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                          VLLM_USE_FLASHINFER_SAMPLER='0', OMP_NUM_THREADS='8',
                          VLLM_ENABLE_V1_MULTIPROCESSING='0', VLLM_BATCH_INVARIANT='0',
                          VLLM_USE_SIMPLE_KV_OFFLOAD='0',
                          HF_HOME='/root/autodl-tmp/moe-a-20261002/hf',
                          HF_HUB_CACHE='/root/autodl-tmp/moe-a-20261002/hf/hub')
        # The native installed libraries require the existing CUDA 13 library directory.
        import torch
        from runtime_bootstrap import apply
        dump(out/'runtime-bootstrap.json', dict(ignored_orphan_kernel_modules=apply(),
            scope='Process-local import filter by installed Torch RECORD; installed files unchanged.'))
        import vllm
        from vllm.engine.arg_utils import EngineArgs
        from vllm.v1.engine.llm_engine import LLMEngine
        from request_measurement import measure_episode
        from native_capture import set_empty_admission_cap
        from native_offload_observer import drain
        from native_host_snapshot import snapshot as host_snapshot
        from memory_telemetry import memory_snapshot
        from admission import install
        if opportunity:
            from admission_probe import install as install_opportunity
        assert vllm.__version__ == '0.26.0'
        torch.set_num_threads(8)
        config, work = read(ROOT/'inputs/config.json'), read(ROOT/'inputs/workload.json')
        assert hashlib.sha256(json.dumps(work, sort_keys=True).encode()).hexdigest() == config['workload_sha256']
        pro_input_sha = None
        if pro:
            pro_input_sha = hashlib.sha256((ROOT/'pro_inputs.json').read_bytes()).hexdigest()
            pro_inputs = read(ROOT/'pro_inputs.json')
            dev, test = pro_inputs['dev'], pro_inputs['test']
            for selected in (dev, test):
                assert len(selected['source_requests']) == len(selected['actual_prompt_token_ids']) == 192
                assert len({r['request_id'] for r in selected['source_requests']}) == 192
                assert all(len(tokens)+1024 <= 4096 for tokens in selected['actual_prompt_token_ids'])
        else:
            dev, test = subset(work,0,32), subset(work,32,64)
        pressure_trace_sha = None
        if pressure:
            dev = test = pressure_population(dev, test)
            if cadence_scan:
                dev = test = pressure_cadence_population(dev)
            pressure_trace_sha = hashlib.sha256(json.dumps(dict(
                request_ids=[r['request_id'] for r in dev['source_requests']],
                actual_prompt_token_ids=dev['actual_prompt_token_ids'],
                arrival_traces_s=dev['arrival_traces_s']), sort_keys=True).encode()).hexdigest()
            if burst:
                test = pressure_burst_population(dev)
        if declared_budget_matched:
            # Check only known declarations, before engine initialization or measurement.
            # Never inspect realized output or adjust the cap during service.
            assert config['output_tokens'] == 1024 and not config.get('output_tokens_by_request')
            declared_count = len(test['actual_prompt_token_ids'])
            declared_pages = sum((len(tokens)+config['output_tokens']+15)//16
                                 for tokens in test['actual_prompt_token_ids'])
            declared_mean_ceiling = (declared_pages+declared_count-1)//declared_count
            assert (declared_count, declared_pages, declared_mean_ceiling) == (384, 70702, 185)
            assert usable_pages == 32768 and usable_pages//declared_mean_ceiling == 177
            declared_budget_matched_calibration = dict(known_page_sum=declared_pages,
                requests=declared_count, mean_pages=declared_pages/declared_count,
                uniform_charge_pages=declared_mean_ceiling,
                derived_fixed_cap=usable_pages//declared_mean_ceiling,
                source='known_prompt_plus_declared_max')
        # Declared before any outcomes. SLO grid is exploratory, never selected post hoc.
        protocol = dict(base_commit='5593b5f', dev_indices=[0,32], test_indices=[32,96],
            native_dev_reference=32, dev_caps=[32,24,16], test_order=['fixed','kv','recovery','recovery','kv','fixed'],
            selection='Among complete dev arms minimize mean external-arrival-to-completion time; ties use higher cap.',
            fixed_baseline_claim_scope='The selected cap is the mean-flow choice, not necessarily the best '
                'complete-service or joint-SLO baseline. All three fixed-cap raw distributions and the full '
                'exploratory SLO surface remain comparators; any benefit claim must inspect them, including '
                'TTFT tails and natural output amounts, rather than only the selected mean-flow arm.',
            kv_floor=kv_floor, recovery_high=2, recovery_clear=0, sample_interval_s=.1,
            max_signal_wait_s=10, run_timeout_s=240, prompt_arrival_gap_s=.2,
            output_mode='natural EOS or max 1024; no future output length read',
            slo_grid=dict(ttft=[2,5,10,20,40], max_gap=[.25,.5,1,2], completion=120),
            slo_role='Exploratory full 20-point surface, not an application SLO or chosen headline.',
            invariants='Native running scheduling, victim tail, recovery path, transfer and quantum unmodified.',
            reset='Same engine/compiled shapes; identical warmup per cell then connector reset; fresh policy state.',
            lock=LOCK, host_budget_bytes=int(Path('/sys/fs/cgroup/memory.max').read_text()),
            model=config['model'], input_sha256=config['workload_sha256'])
        protocol.update(profile=args.profile, usable_kv_pages=usable_pages,
            native_max_seqs=engine_max, pro_inputs_file_sha256=pro_input_sha,
            evidence_role='NORMAL_CAPACITY_PRO6000' if pro else 'CONTROLLED_8GIB_MECHANISM_ONLY')
        if pro:
            protocol.update(dev_indices='pro_inputs.dev 192 distinct articles',
                test_indices='pro_inputs.test 192 disjoint articles', native_dev_reference=256,
                dev_caps=[96,128,192], prompt_arrival_gap_s=0.,
                pressure_probes=[dict(label='low',count=64,gap=.5),
                    dict(label='near',count=192,gap=.1), dict(label='high',count=192,gap=.02)],
                pressure_selection='First near if actual preemptions>0; else high. '
                    'No preemptions even in high is a valid normal-capacity boundary, not a failed experiment.',
                host_kv_gib=16, max_model_len=4096, precision='bfloat16; no rope extension')
        if pressure:
            protocol.pop('pressure_probes')
            protocol.pop('pressure_selection')
            protocol.update(dev_indices='pro_inputs.dev[i], pro_inputs.test[i] interleaved; 384 unique IDs',
                test_indices='Same 384-request population and arrival trace as fixed-cap development',
                native_dev_reference=None, dev_caps=[128,192,256], prompt_arrival_gap_s=.1,
                population=384, trace_sha256=pressure_trace_sha,
                trace_hash_contents='Ordered request IDs, actual prompt token IDs, and external arrival traces',
                workload_sha256=hashlib.sha256(json.dumps(dev, sort_keys=True).encode()).hexdigest(),
                evidence_role='EXPLORATORY_SAME_POPULATION_SAME_TRACE_NOT_INDEPENDENT_CONFIRMATION',
                concurrency_comparison='Native maximum stays 256; fixed uses selected cap, KV/recovery use 256. '
                    'These policies do not have identical admission concurrency limits. '
                    'KV and recovery differ only by the recovery signal.',
                test_admission_caps=dict(fixed='selected from 128/192/256', kv=256, recovery=256))
        if burst:
            protocol.update(test_indices='Same ordered 384-request population as fixed-cap development; '
                    'different fixed external arrival trace', prompt_arrival_gap_s=None,
                arrival_definition=dict(first_count=256, first_gap_s=.1, first_arrival_s=0.,
                    final_count=128, final_simultaneous_arrival_s=55., runtime_state_dependent=False),
                selection_development=dict(profile='pro-pressure-dev', population=384, dev_caps=[128,192,256],
                    arrival_gap_s=.1, arrival_last_s=max(dev['arrival_traces_s']['steady']), trace_sha256=pressure_trace_sha,
                    workload_sha256=protocol['workload_sha256']),
                trace_sha256=hashlib.sha256(json.dumps(dict(
                    request_ids=[r['request_id'] for r in test['source_requests']],
                    actual_prompt_token_ids=test['actual_prompt_token_ids'],
                    arrival_traces_s=test['arrival_traces_s']), sort_keys=True).encode()).hexdigest(),
                workload_sha256=hashlib.sha256(json.dumps(test, sort_keys=True).encode()).hexdigest(),
                evidence_role='EXPLORATORY_ARRIVAL_WINDOW_DIAGNOSTIC_STEADY_CAP_TRANSFER_NOT_INDEPENDENT_TEST',
                selection_transfer='Use the completed steady development selection unchanged; no burst cap tuning. '
                    'This cap is a transferred baseline, not demonstrated optimal on the burst trace.',
                comparison_scope='Compare policies only within this identical burst trace. Do not interpret '
                    'steady-versus-burst timing differences as causal speedups or natural production benefits.')
        accepted = read(args.selection) if args.profile in ('pro-test', 'pro-pressure-test',
            'pro-pressure-burst-test') else None
        if accepted is not None:
            assert accepted['pro_inputs_file_sha256'] == pro_input_sha
            assert accepted['selected_cap'] in protocol['dev_caps']
            if pressure:
                assert accepted['profile'] == 'pro-pressure-dev'
                assert accepted['population'] == 384
                assert accepted['trace_sha256'] == pressure_trace_sha
                assert accepted['workload_sha256'] == hashlib.sha256(json.dumps(dev, sort_keys=True).encode()).hexdigest()
                assert accepted['selected_pressure_gap_s'] == .1
                protocol['test_admission_caps']['fixed'] = accepted['selected_cap']
                if burst:
                    selection_status_path = args.selection.parent/'status.json'
                    selection_status = read(selection_status_path)
                    validate_burst_selection(accepted, selection_status, pro_input_sha, dev)
                    protocol['selection_completion_source'] = dict(path=str(selection_status_path),
                        sha256=hashlib.sha256(selection_status_path.read_bytes()).hexdigest(), contents=selection_status)
            else:
                assert accepted['selected_pressure_gap_s'] in (.02, .1)
                test = subset(test,0,192,accepted['selected_pressure_gap_s'])
            protocol['selection_source'] = dict(path=str(args.selection),
                sha256=hashlib.sha256(args.selection.read_bytes()).hexdigest(), contents=accepted)
        if opportunity:
            protocol.update(
                evidence_role='EXPLORATORY_NATIVE_FIT_SINGLE_DELAY_PROBE_NOT_INDEPENDENT_CONFIRMATION',
                test_order=(['baseline'] if args.profile == 'pro-opportunity-scan' or reservation_scan
                            else ['baseline', 'headroom32', 'headroom32', 'baseline'] if headroom_probe
                            else ['baseline', 'reservation', 'reservation', 'baseline'] if reservation_probe
                            else ['timer', 'progress', 'progress', 'timer'] if progress_probe
                            else ['baseline', 'delay', 'delay', 'baseline']),
                dev_caps=[], dev_indices='No new development sweep in this diagnostic',
                test_indices='Same 384 interleaved articles and 0.1s trace as pressure exploration',
                test_admission_caps=dict(baseline=256, delay=256),
                concurrency_comparison='Both arms use native maximum256, complete-inflight cap256 and KV floor3277. '
                    'Only the once-per-run bounded extra admission delay differs.',
                recovery_high=None, recovery_clear=None, sample_interval_s=None,
                max_signal_wait_s=None,
                admission_count='Unique successfully admitted but unfinished requests, including paused/recovery.',
                diagnostic_cap=256,
                selection='No new tuning. Reuse the previously competitive KV256 reference, with corrected '
                    'complete-lifecycle counting shared by both arms. Not a claim of optimal fixed cap.',
                fixed_baseline_claim_scope='Prior fixed128/192/256 tuning informs this diagnostic; old running+remote '
                    'caps are not complete-inflight caps. This group tests one bounded action, not a method claim.',
                intervention=dict(events_per_run=1, delay_s=.1, hard_gate_deadline_s=.25,
                    trigger='First actually native-fit, cap/KV-allowed never-started request with contemporaneous '
                        'unfinished recovery; baseline records the same first opportunity without delaying.',
                    expiry='Measured from first controller block; beyond deadline no extra probe refusal. '
                        'Native capacity or step duration may still delay actual execution.',
                    original_kv_age_bypass_s=10,
                    original_age_rule_role='Prototype anti-starvation bypass for KV only; not an application deadline.'),
                prefix_comparison='Same frozen inputs and policy prefix; compare recorded first-opportunity states. '
                    'Separate executions are not asserted to be identical GPU-state forks.',
                candidate_status='Diagnostic only; no recovery-controller efficacy claim.')
            if progress_probe:
                protocol.update(
                    evidence_role='EXPLORATORY_SINGLE_RELEASE_CONDITION_CORRECTION_NOT_CONFIRMATION',
                    test_admission_caps=dict(timer=256, progress=256),
                    concurrency_comparison='Both arms execute one identical first-opportunity delay. '
                        'Only release differs: timer100ms versus minimum100ms then observed output '
                        'progress of the trigger-time recovery set, with the same250ms hard deadline.',
                    intervention=dict(events_per_run=1, minimum_delay_s=.1, hard_gate_deadline_s=.25,
                        timer_release='Release after100ms from first controller block.',
                        progress_release='After100ms, release when every frozen pending recovery request '
                            'has produced another output token or finished; unconditionally remove extra '
                            'gate at250ms. Native capacity can continue to delay first prefill.',
                        trigger='Unchanged first actually native-fit, cap/KV-allowed never-started '
                            'request with action-time unfinished recovery.', original_kv_age_bypass_s=10),
                    evidence='Prior B2 scheduled726 restored tokens and98 fresh prefill tokens together '
                        'after100ms; B1 was already native-capacity blocked. This diagnoses the residual '
                        'same-batch admission opportunity, not a claim that prior timer failed to '
                        'protect native recovery scheduling.',
                    stopping_rule='One timer/progress/progress/timer group. If no actual refusal differs '
                        'from timer, or no coherent recovery and complete-service improvement appears, '
                        'stop this release-condition attempt; do not sweep durations. A positive signal '
                        'requires a matched signal-free control before any innovation claim.')
            if reservation_scan:
                protocol.update(evidence_role='OBSERVATION_ONLY_KNOWN_PREFILL_RESERVATION_OPPORTUNITY_SCAN',
                    intervention=None,
                    reservation_observation='At the existing actual new allocation hook, read the native '
                        'unfinished-prefill unallocated-block reservation. New local prefill uses native '
                        'reserved_blocks=0; do not subtract already held blocks again. Record full and '
                        'chunk footprint plus other unfinished reservations, not future output length.',
                    stopping_rule='One baseline only. If no native/cap/KV-allowed combined-footprint '
                        'excess exists, do not launch a candidate matrix. If an opportunity exists, '
                        'first test its bounded admission action; accounting alone is no novelty claim.')
            if reservation_probe:
                protocol.update(evidence_role='EXPLORATORY_FIRST_KNOWN_PREFILL_RESERVATION_ACTION_ABBA',
                    test_admission_caps=dict(baseline=256, reservation=256),
                    intervention=dict(events_per_run=1, hard_gate_deadline_s=.25,
                        trigger='First actual new allocation with native/cap/KV allowance, positive existing '
                            'native unfinished-prefill unallocated blocks, and new full known sequence '
                            'plus existing unallocated blocks plus native watermark exceeding actual free.',
                        release='At a target allocation hook, release if the combined footprint fits or '
                            'existing unallocated blocks become zero; otherwise release extra gating at '
                            '250 ms from first actual refusal. No minimum hold; native constraints remain.',
                        scope='Only the first target new request; FIFO followers are held within the same '
                            'scheduler call. Existing requests and recovery always bypass this extra gate.',
                        original_kv_age_bypass_s=10),
                    stopping_rule='Initial ABBA group, prescribed pairs 01/00 and 02/03. If no actual '
                        'extra refusals or no coherent full-service signal, do not tune neighboring durations '
                        'or reservation margins. If positive, immediately compare matched fixed headroom '
                        'and complete-inflight fixed caps before claiming value of the recovery information.',
                    repeat_budget='At most one extra frozen ABBA group for sign-flipped or noisy service '
                        'results, and only if both candidate runs actually intervene and target pre-output '
                        'preemption is consistently reduced. Otherwise close this action attempt. '
                        'No repeat-until-significant and no parameter changes.',
                    motivation='The prior observation-only scan found two full-footprint excesses of 1/2 blocks; '
                        'both new targets were preempted before their first output. Chunk-footprint excesses '
                        'were zero. This is a correlation motivating an action probe, not proof of harm.')
            if headroom_probe:
                protocol.update(evidence_role='EXPLORATORY_FIRST_FIXED_HEADROOM_ACTION_ABBA',
                    test_admission_caps=dict(baseline=256, headroom32=256),
                    reservation_signal_kind='fixed_headroom', fixed_headroom_blocks=32,
                    intervention=dict(events_per_run=1, hard_gate_deadline_s=.25,
                        minimum_delay_s=0,
                        trigger='First actually native-fit, cap/KV-allowed never-started request with '
                            'positive new compute and free minus full known prefill footprint minus native '
                            'watermark below32 pages. Recovery state does not control this probe.',
                        release='At a target allocation hook release when the same full-fit margin reaches32 '
                            'pages; otherwise unconditionally remove extra gating at250ms from first actual '
                            'refusal. No minimum hold. Native constraints may still delay execution.',
                        scope='One first target per run; retain FIFO followers within the scheduler call. '
                            'Existing requests and recovery always bypass extra gating.',
                        original_kv_age_bypass_s=10),
                    motivation='Four prior baseline runs had35 pre-output preemptions. Shadow extra32-page '
                        'headroom covered33 and selected23 other first allocations, while known-prefill '
                        'reservation excess covered3. Each baseline first32-page opportunity had R=0 and '
                        'later pre-output preemption. These are retrospective labels, not causal savings.',
                    margin_selection='Reuse the prior small1/16/32-page shadow comparison;32 is an action '
                        'probe selected for observed coverage, not a baseline tuned on online service outcomes.',
                    selection='No new online cap or service-metric tuning. Keep cap256 and the existing KV '
                        'reference;32-page probe comes from the stated three-value shadow comparison.',
                    stopping_rule='One baseline/headroom32/headroom32/baseline group with prescribed pairs '
                        '01/00 and02/03. No changed actions or inconsistent full-service distributions and '
                        'the full20-point joint-SLO surface closes this probe; no nearby page/time tuning '
                        'and no further repetition. Only consistent positive execution and service evidence '
                        'permits comparison with strong complete-inflight fixed128/192 controls.',
                    repeat_budget='Exactly one ABBA group; no additional repetition.',
                    candidate_status='Strong simple explanation and action-value probe only. Ordinary fixed '
                        'headroom is not a novelty claim or independent confirmation.')
        if gc_liveness:
            history_sha = hashlib.sha256(args.liveness_history.read_bytes()).hexdigest()
            assert history_sha == '8c59e1c65eafca2285768fc0cac101a1fe76a671fa25266ba20ad35fe5b7249a'
            protocol.update(evidence_role='HARNESS_HISTORY_LIVENESS_DIAGNOSTIC_NOT_ADMISSION_METHOD',
                test_order=['retained', 'released', 'released', 'retained'],
                test_admission_caps=dict(retained=192, released=192), diagnostic_cap=192,
                kv_floor=0, fixed_baseline_claim_scope='Same fixed192 admission rule in every cell; '
                    'implemented by complete-inflight cap192 and KV floor0. No extra admission gate.',
                concurrency_comparison='All cells use complete-inflight cap192, native maximum256, '
                    'KV floor0; running/recovery scheduling and1024 quantum unchanged.',
                selection='No policy or cap tuning. Fixed192 was the previously competitive simple baseline '
                    'whose strict-gap results contained four common host stalls without preemption.',
                intervention=dict(kind='history_raw_reference_liveness', admission_rule_changed=False,
                    history_path=str(args.liveness_history), history_sha256=history_sha,
                    retained='Keep the identical completed dev-cap128 raw object strongly referenced '
                        'through current measurement; it is not submitted to the engine.',
                    released='Load and validate the same history, then release it before common heap cleanup.',
                    common_setup='After identical warmup and before current arrivals, release warmup raw '
                        'and call gc.collect() in all cells. No forced collection or threshold changes '
                        'during measurement. Passive callbacks record actual collection intervals.',
                    measured_work='Same384 requests,0.1s arrivals, naturalEOS/max1024 and all-arrival timing.'),
                stopping_rule='Exactly one retained/released/released/retained group, pairs01/00 and02/03. '
                    'Report all service outcomes and every>250ms common host interval with GC overlap. '
                    'Only aligned collection intervals support GC attribution; a service delta alone does not. '
                    'If retained-history stalls align with GC and release consistently reduces them, fix '
                    'common measurement lifecycle and re-establish affected strong baselines. Otherwise '
                    'retain the uncertainty, do not subtract old delays or repeat until significant.',
                repeat_budget='One group; no additional repeats or GC-threshold tuning.',
                candidate_status='Measurement credibility diagnosis only; no recovery/admission efficacy claim.')
        if simple_recheck:
            protocol.update(evidence_role='REPAIRED_MEASUREMENT_STRONG_SIMPLE_BASELINE_COMPARISON',
                test_order=['fixed192', 'kv256', 'kv256', 'fixed192'],
                test_admission_caps=dict(fixed192=192, kv256=256), diagnostic_cap=None,
                selection='No retuning: fixed192 and KV256 reuse prior competing simple configurations. '
                    'This is a same-trace baseline re-establishment, not an independent test.',
                fixed_baseline_claim_scope='Compare the previously competitive fixed192 against KV256. '
                    'Fixed128 remains a prior mean-flow comparator; this group does not establish a '
                    'globally optimal cap or an application SLO.',
                concurrency_comparison='Native maximum256 and resources are shared. Fixed192 uses complete '
                    'inflight cap192 and KV floor0; KV256 uses complete cap256, floor3277 and the existing '
                    '10s age bypass. Their cap and gate differ explicitly; all old requests bypass.',
                kv_floor=dict(fixed192=0, kv256=3277),
                intervention=dict(kind='strong_simple_admission_rules', extra_probe_enabled=False,
                    common_setup='Release every warmup raw and cross-cell scored raw after use. After '
                        'warmup, collect unreachable objects before current external arrivals in every '
                        'cell. Passive GC callbacks enabled equally; no collection/settings changes '
                        'during service and no historical raw injected.',
                    measured_work='Same384 inputs,0.1s arrivals,naturalEOS/max1024; all-arrival timing.'),
                motivation='The one frozen liveness ABBA had4/0/0/4 common>250ms stalls, with GC '
                    'covering77-90% of each retained-history stall. Historical strict-gap comparisons '
                    'therefore need a fair lifecycle before claiming residual admission value.',
                stopping_rule='One fixed192/KV256/KV256/fixed192 group, pairs01/00 and02/03. Report '
                    'all20 exploratory joint-SLO points and raw distributions, admission differences, '
                    'preemptions, GC overlaps, output changes and all outcomes. If fixed192 removes '
                    'the previously reported KV advantage, drop that advantage. If KV retains a '
                    'consistent tradeoff, document its residual without attributing it to recovery. '
                    'No tuning, revival of closed probes, or further repeats in this diagnostic.',
                repeat_budget='Exactly one ABBA group; no additional repeats.',
                candidate_status='Baseline credibility repair, not a new admission method.')
        if native_cap_recheck:
            protocol.update(evidence_role='STRONG_SIMPLE_BASELINE_WITH_EQUIVALENT_NATIVE_ADMISSION_GUARD',
                native_admission_guards=dict(fixed192=192, kv256=256),
                concurrency_comparison='Engine compiled maximum remains256. Native waiting-loop '
                    'guards are192/256, matching the complete-inflight caps192/256. Fixed uses '
                    'KV floor0; KV256 uses floor3277 and the unchanged10s age bypass. Running '
                    'scheduling is unchanged; these admission configurations differ explicitly.',
                motivation='The repaired quartet recorded76992/77250 FIFO-cap holds and3.65/3.73s '
                    'controller-plus-recording cost in fixed192, versus0.80s in KV256. That cost '
                    'difference is material relative to the1.58/2.19s mean-flow difference; no time '
                    'may be subtracted from either result.',
                equivalent_guard_conditions='Install on a drained engine, no streaming, every running '
                    'or recovery request belongs to the complete admitted set, and its size<=cap. '
                    'Then running==cap means all admitted requests are already running, so no old '
                    'recovery can be excluded by the native waiting-loop guard. Keep the complete '
                    'Gate: running191+remote1 must still prevent a193rd unique admission.',
                native_guard_scope='Only scheduler.max_num_running_reqs changes at the drained cell '
                    'boundary. Engine compiled maximum256, running loop, victim, recovery, transfer '
                    'and quantum stay unchanged. Both arms retain the same complete-inflight Gate; '
                    'the native guard skips redundant new-request scans at a full cap.',
                stopping_rule='One fixed192/KV256/KV256/fixed192 group, pairs01/00 and02/03. '
                    'Re-establish the strong simple comparison after eliminating the identified '
                    'equivalent FIFO scan. Record full results and total remaining observation cost. '
                    'If fixed192 explains the prior KV advantage, delete that advantage; otherwise '
                    'retain only the measured simple-policy tradeoff. No new recovery mechanism '
                    'claim, threshold tuning, or additional repetition.')
        if cadence_scan:
            protocol.update(evidence_role='OBSERVATION_ONLY_HALVED_OFFERED_RATE_DOMAIN_DIAGNOSIS',
                test_order=['baseline'], test_admission_caps=dict(baseline=256), diagnostic_cap=256,
                test_indices='Same ordered 384 interleaved articles; new fixed 0.2s external arrival trace',
                prompt_arrival_gap_s=.2,
                arrival_definition=dict(count=384, first_arrival_s=0., gap_s=.2,
                    last_arrival_s=max(test['arrival_traces_s']['steady']),
                    runtime_state_dependent=False),
                offered_rate_requests_per_s=5., previous_offered_rate_requests_per_s=10.,
                kv_floor=3277, max_signal_wait_s=10, sample_interval_s=.1,
                native_admission_guards=dict(baseline=256),
                concurrency_comparison='One baseline: compiled maximum256, native waiting guard256, '
                    'complete-inflight cap256, KV floor3277 and unchanged10s external-age bypass. '
                    'No probe or reservation intervention; running/recovery execution is unchanged.',
                selection='No tuning or cap selection. Reuse the KV256 baseline with the latest '
                    'measurement lifecycle and native guard; only external arrival cadence changes.',
                fixed_baseline_claim_scope='Operating-domain observation only; this single arm does '
                    'not select a strongest policy or establish a recovery-aware method.',
                intervention=dict(kind='external_arrival_cadence_only', observation_only=True,
                    extra_probe_enabled=False, reservation_probe_enabled=False,
                    common_setup='Release each warmup raw and any cross-cell scored raw after use; '
                        'collect before current external arrivals and passively record GC. No forced '
                        'collection/settings changes during service and no historical raw injected.',
                    measured_work='Same384 ordered prompts, naturalEOS/max1024, context4096, BF16; '
                        '64GiB usable GPU KV,16GiB Host KV and1024 service quantum. Arrival0.2*i only.'),
                motivation='All32 eligible pending-recovery new allocations in the preceding '
                    'native-cap comparison had external age>=10s. This single halved-rate trace '
                    'checks for fresh admission opportunities; the mixed-output turnover explanation '
                    'was rejected and output budgets remain unchanged.',
                native_guard_scope='Set scheduler.max_num_running_reqs=256 only at the drained '
                    'cell boundary; compiled maximum256 and complete admitted-set Gate remain. '
                    'Require no streaming, admitted<=cap and running contained in admitted.',
                prefix_comparison='A single baseline observation; no paired intervention or claim '
                    'that executions on different arrival traces share a policy prefix.',
                comparison_scope='Classify fresh external-age<10s pending-recovery/native-fit/'
                    'base-allowed new prefill opportunities, versus no recovery or ordinary KV '
                    'coverage. No cross-trace speedup, independent-test, or novelty claim.',
                stopping_rule='Exactly one baseline at0.2s cadence. Report all384 arrivals, full '
                    'service outcomes, recovery, native fit, cap/KV coverage, age and observation '
                    'costs. No opportunity is a valid operating-domain result. No adjacent-rate '
                    'sweep, repeats, output-budget changes or intervention within this profile.',
                repeat_budget='One operating point, one baseline run; no repeats or nearby rate sweep.',
                candidate_status='Observation-only operating-domain diagnosis, not a new admission mechanism.')
        if declared_budget_comparison:
            fixed_cap = declared_budget_matched_calibration['derived_fixed_cap'] if declared_budget_matched else 128
            fixed_label = f'fixed{fixed_cap}'
            protocol.update(evidence_role='STRONG_SIMPLE_DECLARED_TOKEN_BUDGET_BASELINE',
                test_order=[fixed_label, 'declaredbudget', 'declaredbudget', fixed_label],
                test_admission_caps={fixed_label:fixed_cap, 'declaredbudget':256}, diagnostic_cap=None,
                native_admission_guards={fixed_label:fixed_cap, 'declaredbudget':256},
                kv_floor={fixed_label:0, 'declaredbudget':0},
                declared_budget_blocks=usable_pages, declared_budget_block_size=16,
                declared_budget_rule='For every admitted unfinished request charge '
                    'ceil((prompt_tokens+declared_max_tokens)/16) once. Admit a new FIFO request '
                    'only if total charge including it<=32768 usable blocks. Release charge on '
                    'native completion; retain it through pause/recovery. Do not additionally '
                    'charge actual GPU usage or infer future realized output length.',
                selection='No tuning: fixed128 is the previous completed mean-flow development '
                    'choice; the declared-budget threshold is the whole usable physical pool. '
                    'One simple baseline comparison on the existing development trace.',
                fixed_baseline_claim_scope='Fixed128 is a prior mean-flow choice, not a globally '
                    'optimal joint-SLO policy. This comparison is not independent confirmation.',
                concurrency_comparison='Same compiled maximum256, model, inputs, arrivals, '
                    f'64GiB GPU KV and16GiB host KV. Complete caps/native waiting guards{fixed_cap}/256. '
                    'Both use KV floor0. Only declaredbudget additionally gates total declared '
                    'lifecycle KV commitment. Running/victim/recovery/transfer/quantum unchanged.',
                admission_wait_rule='Fixed cap or declared budget waits inside the scheduler '
                    'in unchanged FIFO order until its ordinary capacity predicate permits. '
                    'No age bypass of the declared budget. Existing requests always bypass '
                    'the new gate. Same external-arrival240s observation timeout and full '
                    'population accounting; no claim of a per-request waiting/SLO bound.',
                max_signal_wait_s=None, recovery_high=None, recovery_clear=None,
                intervention=dict(kind='ordinary_declared_budget_vs_fixed_concurrency',
                    extra_probe_enabled=False, reservation_probe_enabled=False,
                    measured_work='Same384 ordered prompts,0.1s external trace,naturalEOS/max1024.',
                    common_setup='Release warmup/cross-cell raw, collect before external '
                        'arrivals, observe GC equally; no in-service GC change.'),
                motivation='The18 observed>1s gaps ended before the first legal pending-recovery '
                    'new-admission opportunity. Their initial admissions hadR=0 and95-1512 '
                    'full-fit headroom pages. Test ordinary declared lifecycle capacity '
                    'before attributing an earlier decision principle to a new signal.',
                prefix_comparison='Separate online runs of frozen inputs; actual action prefixes '
                    'and output changes are recorded, not asserted identical GPU states.',
                stopping_rule='Exactly one ABBA group, pairs01/00 and02/03. Report all arrivals, '
                    'all20 exploratory SLO points, raw distributions, action differences, '
                    'output changes and total control cost. If this simple rule explains '
                    'benefits, retain it as a simple baseline, not recovery innovation. '
                    'No cap/budget sweep, hold-duration tuning or additional repeats.',
                repeat_budget='One ABBA group; no additional repetition.',
                candidate_status='Ordinary conservative token-budget baseline, no novelty claim.')
            if declared_budget_matched:
                protocol.update(evidence_role='ORDINARY_DECLARED_LENGTH_ACCOUNTING_VS_MEAN_MATCHED_FIXED_CAP',
                    declared_budget_matched_calibration=declared_budget_matched_calibration,
                    selection='No outcome-based selection: known384 prompt-plus-declared-max page '
                        'bounds sum70702; ceil(70702/384)=185;32768//185=177. Verify these input '
                        'declarations before engine initialization. No online cap adjustment.',
                    fixed_baseline_claim_scope='Fixed177 is the mean-declared-page corresponding '
                        'concurrency baseline, not a tuned or globally optimal SLO policy. Ordinary '
                        'length accounting versus this fixed cap on the same reused trace; not '
                        'independent confirmation or a new controller.',
                    motivation='Test the simple fixed-concurrency explanation at the cap implied '
                        'by the known mean declared page bound. An observed action representation '
                        'difference does not establish a full-service benefit.',
                    intervention=dict(protocol['intervention'],
                        kind='ordinary_declared_budget_vs_mean_declared_fixed_concurrency'))
        if mc_budget:
            protocol.update(evidence_role='ADAPTED_MC_BENCHMARK_VS_FULL_DECLARED_RESERVATION',
                test_order=['declaredbudget', 'mcbudget', 'mcbudget', 'declaredbudget'],
                test_admission_caps={'declaredbudget':256, 'mcbudget':256}, diagnostic_cap=None,
                native_admission_guards={'declaredbudget':256, 'mcbudget':256},
                kv_floor={'declaredbudget':0, 'mcbudget':0},
                declared_budget_blocks=usable_pages, declared_budget_block_size=16,
                selection='No parameter tuning; same full32768-page ordinary budget in both arms. '
                    'MC peak feasibility is a production-guarded adaptation of arXiv2502.07115v5 '
                    'section4 equation5 and AppendixC Algorithm2, not a new time-reuse algorithm.',
                candidate_status='Nearest-method adaptation and action-value comparison; no novelty claim.',
                fixed_baseline_claim_scope='No fixed-cap comparison in this group; ordinary full '
                    'declared budget is the comparator. The prior matched fixed177 comparison is closed.',
                prefix_comparison='Same frozen inputs in separate online executions; no claim of '
                    'identical GPU state, natural output, or complete pre-intervention trajectory.',
                max_signal_wait_s=None, recovery_high=None, recovery_clear=None,
                admission_wait_rule='Wait in unchanged scheduler FIFO. No age override; old requests '
                    'always bypass. Common240s external-arrival observation cutoff reports unfinished '
                    'requests. No TTFT, completion, or wall-time wait guarantee.',
                concurrency_comparison='Both complete cap/native guard256, compiled256, KVfloor0, '
                    'same model/BF16,64GiB GPU KV/16GiB host, FCFS/quantum1024 and native recovery.',
                intervention=dict(kind='full_declared_sum_vs_guarded_future_peak',
                    measured_work='Same384 ordered prompts,0.1s arrivals,naturalEOS/max1024.',
                    rule='Only relax an otherwise rejected declared-budget decision when all existing '
                        'admitted requests are resident single-token decode, their current native '
                        'slots cover exactly their present known tokens, and all used pages are accounted. '
                        'Old future occupancy ends after its declared remaining decode steps; candidate '
                        'starts at full prompt and grows by at most one token per round, retaining its '
                        'full plateau even if prefill delays completion. Otherwise keep ordinary budget.',
                    common_setup='Release old raw, identical warmup, collect before arrivals, passive GC.'),
                motivation='Determine whether temporally non-overlapping request capacity provides '
                    'full-service value beyond ordinary full-lifecycle reservations. Recovery feedback '
                    'was late and fixed177 already closely matched ordinary budget; neither establishes '
                    'this stronger published baseline or a novel residual.',
                stopping_rule='Exactly one ABBA group. Count proposed and actually allocated extra '
                    'admissions, preemptions, all arrivals/outcomes, full20 exploratory SLO grid, raw '
                    'distributions, natural output changes and control cost. No threshold/cap sweep. '
                    'No actions closes this guarded domain; no complete benefit retains simple budget; '
                    'benefit retains MC adaptation as a stronger baseline, not a new algorithm claim.',
                repeat_budget='One ABBA group, pairs01/00 and02/03; not independent confirmation.')
        if mc_budget_ablation:
            protocol.update(evidence_role='STRUCTURE_MATCHED_FUTURE_PEAK_ABLATION',
                test_order=['mcstatic', 'mccapped', 'mccapped', 'mcstatic'],
                test_admission_caps={'mcstatic':256, 'mccapped':256},
                native_admission_guards={'mcstatic':256, 'mccapped':256},
                kv_floor={'mcstatic':0, 'mccapped':0},
                maximum_declared_blocks=39138,
                use_future_peak={'mcstatic':False, 'mccapped':True},
                ceiling_calibration=dict(source_group='westd-mc-budget-r01',
                    source_runtime_commit='e10356596c6fad39a7334219a30a0d60a8bde12a',
                    observed_candidate_declared_peaks=[39138,39078],
                    rule='Maximum observed full-declaration charge in the two completed MC development arms; no search.'),
                selection='Both arms share the fixed39138 declaration ceiling calibrated on seen '
                    'development MC charge peaks. It denied none of the recorded202/203 successful '
                    'MC relaxations. This is a new common restriction relative to the uncapped v18 '
                    'MC implementation and is frozen before this group. Physical KV stays32768 pages.',
                candidate_status='Ablation of an existing published admission principle; no novel-method claim.',
                fixed_baseline_claim_scope='Static overbooking control with matching guard, ordinary '
                    'budget, ceiling and lifecycle. Not a fixed-concurrency comparison or independent confirmation.',
                intervention=dict(kind='guarded_static_ceiling_vs_guarded_capped_future_peak',
                    measured_work='Same384 ordered prompts,0.1s arrivals,naturalEOS/max1024.',
                    rule='Ordinary sum<=32768 permits unchanged. Otherwise both arms require native '
                        'fit, complete cap256, identical resident synchronous-decode guards and '
                        'sum<=39138. Only mccapped additionally requires the future peak<=32768. '
                        'mcstatic computes no future peak and makes no predictive safety claim. '
                        'Old requests always bypass. No victim/recovery/quantum changes.',
                    common_setup='Same warmup, raw lifecycle, prearrival GC and passive measurement.'),
                motivation='On seen MC states, static39138 permits92/106 evaluations over15/16 IDs '
                    'that the peak check denied; all202/203 MC permits satisfy the ceiling. This is '
                    'legal action disagreement, not an executed static trajectory or service benefit. '
                    'Observed MC output progress did not fail, so test ordinary relaxed-credit explanation.',
                stopping_rule='Exactly one ABBA, pairs01/00 and02/03, no sweep or extra repetitions. '
                    'Report all arrivals, actual actions, preemptions, full20 fixed exploratory SLO '
                    'grid, raw distributions, output differences and complete cost. If static matches '
                    'the useful tradeoff, do not attribute it to future timing; if peak check helps, '
                    'retain MC as a stronger known baseline, not an original algorithm or progress-failure claim.',
                repeat_budget='One new structure-matched ABBA, not a repeat of MC versus full reservation.')
        if mc_budget_phase:
            protocol.update(evidence_role='MC_ADAPTATION_SCHEDULED_PREFILL_COMPARISON',
                test_order=['mccapped', 'mcphase', 'mcphase', 'mccapped'],
                test_admission_caps={'mccapped':256, 'mcphase':256},
                native_admission_guards={'mccapped':256, 'mcphase':256},
                kv_floor={'mccapped':0, 'mcphase':0}, maximum_declared_blocks=39138,
                use_future_peak={'mccapped':True, 'mcphase':True},
                allow_scheduled_prefill={'mccapped':False, 'mcphase':True},
                ceiling_calibration=dict(source_group='westd-mc-budget-r01',
                    source_runtime_commit='e10356596c6fad39a7334219a30a0d60a8bde12a',
                    observed_candidate_declared_peaks=[39138,39078],
                    rule='Unchanged maximum calibrated before completed v19; no new tuning.'),
                selection='Same32768 physical and39138 declaration limits, cap256 and equivalent '
                    'optimized peak arithmetic from0fd6537. Only the phase guard differs; no search.',
                candidate_status='Nearest-method runtime adaptation improvement; no new-method claim.',
                fixed_baseline_claim_scope='Strict versus scheduled-prefill MC adaptation. '
                    'Not an independent workload, fixed-cap or static-overbooking comparison.',
                intervention=dict(kind='strict_old_decode_vs_current_step_scheduled_prefill',
                    measured_work='Same384 ordered prompts,0.1s arrivals,naturalEOS/max1024.',
                    rule='Both require native fit, cap256, complete physical accounting and future '
                        'peak<=32768 with declaration sum<=39138. Candidate also accepts an old '
                        'zero-output request when its entire remaining prompt is already scheduled '
                        'this step: computed+scheduled==num_tokens and scheduled>0. Its first '
                        'output is due at h=0; it is not considered GPU-completed. Truly incomplete '
                        'prefills and all prior execution guards still fail. No native running, '
                        'victim, recovery, quantum or FIFO changes.',
                    common_setup='Same warmup, raw lifecycle, prearrival GC and passive measurement.'),
                motivation='Existing native-fit phase refusals have positive prefill token budget. '
                    'Strict adaptation permits at most one borrowed admission per schedule call '
                    'because computed advances at schedule end. Existing-state envelopes diagnose '
                    'opportunity, not a counterfactual full-serving gain.',
                stopping_rule='One ABBA, pairs01/00 and02/03; no cap/ceiling sweep or extra repeats. '
                    'Report actual phase allocations/starts, all arrivals, full20 exploratory grid, '
                    'output changes and full cost. Zero actual phase relaxations closes this '
                    'interface/domain; no complete benefit retains strict baseline. Benefit '
                    'strengthens known MC, not an original progress-failure claim.',
                repeat_budget='One adaptation comparison, not independent confirmation.')
        dump(out/'protocol.json', protocol)
        kwargs = dict(model=config['model']['id'], revision=config['model']['revision'],
            tokenizer_revision=config['model']['tokenizer_revision'], dtype='bfloat16', seed=config['seed'],
            max_model_len=4096, max_num_seqs=engine_max, max_num_batched_tokens=1024,
            gpu_memory_utilization=.90, enable_chunked_prefill=True, enable_prefix_caching=False,
            scheduling_policy='fcfs', async_scheduling=False, kv_cache_memory_bytes=kv_bytes,
            scheduler_reserve_full_isl=True, stream_interval=1, enforce_eager=False,
            enable_return_routed_experts=False, kv_offloading_size=16, kv_offloading_backend='native',
            kv_transfer_config={'kv_connector_extra_config': {'offload_prompt_only': False}})
        dump(out/'engine_args.json', kwargs)
        init_start = time.perf_counter()
        engine = LLMEngine.from_engine_args(EngineArgs(**kwargs), enable_multiprocessing=False)
        dump(out/'environment.json', dict(torch=torch.__version__, vllm=vllm.__version__, cuda=torch.version.cuda,
            cpu_threads=torch.get_num_threads(), init_seconds=time.perf_counter()-init_start,
            gpu=gpu_state(True), memory=memory_snapshot(engine, torch), host=host_snapshot(engine)))
        scheduler = engine.engine_core.engine_core.scheduler
        assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == usable_pages
        assert host_snapshot(engine)['cpu_kv']['unique_storage_bytes'] == 16*1024**3
        assert memory_snapshot(engine, torch)['kv_storage_bytes'] == kv_bytes

        def cell(label, mode, cap, selected, probe_enabled=False, release_mode='timer', reservation_enabled=False,
                 retain_history=False):
            path = out/label
            path.mkdir()
            dump(path/'gpu-before.json', gpu_state(True))
            warm_started = time.perf_counter()
            warm_specs = [('short',32,16), ('short',32,32), ('long',2,2)]
            if pro:
                warm_specs += [('short',256,256)]
            for domain, count, warm_cap in warm_specs:
                wc = read(ROOT/f'warmups/{domain}/config.json')
                ww = subset(read(ROOT/f'warmups/{domain}/workload.json'), 0, count, 0)
                if len(ww['source_requests']) < count:
                    base_rows, base_tokens = ww['source_requests'], ww['actual_prompt_token_ids']
                    ww['source_requests'] = [dict(base_rows[i%len(base_rows)], request_id=f'warm{i}') for i in range(count)]
                    ww['actual_prompt_token_ids'] = [base_tokens[i%len(base_tokens)] for i in range(count)]
                wc.update(cap=warm_cap, output_tokens=16, output_tokens_by_request={}, ignore_eos=True, min_tokens=16)
                set_empty_admission_cap(engine, warm_cap)
                wr = measure_episode(engine, ww, wc, 'steady', 1., label+'warm'+str(warm_cap), max_seconds=120)
                dump(path/f'warmup-{domain}-{warm_cap}.json', wr)
                assert wr['status'] == 'COMPLETE'
                del wr
            dump(path/'warmup-drain.json', drain(engine))
            assert engine.reset_prefix_cache(reset_connector=True)
            native_running_cap = cap if native_guard_matches_cap else engine_max
            set_empty_admission_cap(engine, native_running_cap)
            assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == usable_pages
            cell_kv_floor = 0 if (gc_liveness or budget_experiment or
                                 (simple_recheck and mode == 'fixed')) else kv_floor
            if opportunity:
                if mc_budget and mode in ('mc_budget', 'mc_budget_capped', 'mc_budget_static', 'mc_budget_phase'):
                    from mc_budget import install as install_mc_budget
                    gate, uninstall = install_mc_budget(scheduler, cap=cap,
                        budget_blocks=usable_pages,
                        **(dict(maximum_declared_blocks=39138,
                            use_future_peak=mode != 'mc_budget_static') if mc_budget_ablation or mc_budget_phase else {}),
                        **(dict(allow_scheduled_prefill=mode == 'mc_budget_phase') if mc_budget_phase else {}))
                elif budget_experiment and mode == 'declared_budget':
                    from declared_budget import install as install_declared_budget
                    gate, uninstall = install_declared_budget(scheduler, cap=cap,
                        budget_blocks=usable_pages)
                else:
                    gate, uninstall = install_opportunity(scheduler, mode='kv', cap=cap,
                        kv_floor=cell_kv_floor, probe_enabled=probe_enabled, delay_s=.1, max_extra_s=.25,
                        release_mode=release_mode)
                if native_guard_matches_cap:
                    original_begin = gate.begin

                    def guarded_begin():
                        original_begin()
                        assert scheduler.num_waiting_for_streaming_input == 0
                        assert len(gate.admitted_ids) <= cap
                        assert all(r.request_id in gate.admitted_ids for r in scheduler.running)

                    gate.begin = guarded_begin
                if reservation:
                    from reservation_observer import install as observe_reservation
                    observe_reservation(gate, probe_enabled=reservation_enabled, max_extra_s=.25,
                        fixed_headroom_blocks=32 if headroom_probe else None)
            else:
                gate, uninstall = install(scheduler, mode, cap, kv_floor=kv_floor)
            cc = dict(config, requests=len(selected['source_requests']), cap=cap,
                engine_max_num_seqs=engine_max, target_usable_kv_blocks=usable_pages,
                fixed_kv_cache_memory_bytes=kv_bytes,
                intended_usable_kv_bytes=usable_pages*2097152,
                prompt_tokens=max(map(len,selected['actual_prompt_token_ids'])),
                prompt_tokens_semantics='Full unpadded article lengths; see prompt_tokens_by_request.',
                workload_sha256=hashlib.sha256(json.dumps(selected, sort_keys=True).encode()).hexdigest(),
                prompt_tokens_by_request=list(map(len,selected['actual_prompt_token_ids'])),
                arrival_span_s=max(selected['arrival_traces_s']['steady']),
                arrival_gap_s=.2 if cadence_scan else None, arrival_process='explicit_frozen_trace',
                input_preparation='See selected workload.json; source population and split documented in protocol.json.',
                input_independence=('Exploratory reuse: same ordered 384-request population, new fixed '
                    '0.2s external arrival trace. One observation-only baseline; no development '
                    'selection, independent test, or cross-trace speedup claim.' if cadence_scan else
                    'Exploratory reuse: same ordered 384-request population, new fixed burst '
                    'trace, and cap transferred unchanged from completed steady development; not independent '
                    'confirmation or evidence that this cap is burst-optimal.' if burst else
                    'Exploratory reuse: same 384-request population and trace for fixed-cap '
                    'selection and comparison; not independent confirmation.' if pressure else
                    'Historically seen source corpus; normal-profile dev/test are document-disjoint within C.'),
                source='See frozen workload.json / pro_inputs provenance' if pro else config['source'])
            dump(path/'config.json', dict(cc, admission_mode=mode, native_running_limit=native_running_cap,
                                        admission_cap=cap, kv_floor=cell_kv_floor,
                                        native_admission_guard_matches_complete_cap=native_guard_matches_cap,
                                        **(dict(record_gc=True, history_retained=retain_history,
                                            admission_rule='complete cap192; KV floor0, no extra delay')
                                           if gc_liveness else {}),
                                        **(dict(record_gc=True, history_retained=False,
                                            admission_rule=('complete cap192; KV floor0, no extra delay'
                                                if mode == 'fixed' else
                                                'complete cap256; KV floor3277;10s age bypass;no extra delay'))
                                           if simple_recheck or cadence_scan else {}),
                                        **(dict(record_gc=True, history_retained=False,
                                            declared_budget_blocks=usable_pages if mode == 'declared_budget' else None,
                                            declared_budget_block_size=16,
                                            admission_rule=(f'complete cap{cap}; KV floor0' if mode == 'fixed' else
                                                'complete cap256; sum ceil((prompt+max_tokens)/16)<=32768; '
                                                'KV floor0; no age bypass of declared budget'),
                                            admission_wait_rule=protocol['admission_wait_rule'])
                                           if declared_budget_comparison else {}),
                                        **(dict(record_gc=True, history_retained=False,
                                            declared_budget_blocks=usable_pages,
                                            declared_budget_block_size=16,
                                            admission_rule=('guarded static declaration ceiling, no future peak'
                                                if mode == 'mc_budget_static' else
                                                'guarded future peak admitting fully scheduled old prefills'
                                                if mode == 'mc_budget_phase' else
                                                'guarded future peak with common declaration ceiling'
                                                if mode == 'mc_budget_capped' else
                                                'ordinary full declared sum' if mode == 'declared_budget'
                                                else 'production-guarded MC-Benchmark future peak adaptation'),
                                            admission_wait_rule=protocol['admission_wait_rule'])
                                           if mc_budget else {}),
                                        **(dict(maximum_declared_blocks=39138,
                                            use_future_peak=mode != 'mc_budget_static',
                                            ceiling_calibration=protocol['ceiling_calibration'])
                                           if mc_budget_ablation or mc_budget_phase else {}),
                                        **(dict(allow_scheduled_prefill=mode == 'mc_budget_phase')
                                           if mc_budget_phase else {}),
                                        **(dict(declared_budget_matched_calibration=declared_budget_matched_calibration)
                                           if declared_budget_matched else {}),
                                        **(dict(probe_enabled=probe_enabled,
                                            release_mode=release_mode,
                                            reservation_observation=reservation,
                                            reservation_probe_enabled=reservation_enabled,
                                            reservation_signal_kind=(None if observe_gc else
                                                'fixed_headroom' if headroom_probe else 'known_prefill'),
                                            fixed_headroom_blocks=32 if headroom_probe else None,
                                            admission_count='complete_unique_unfinished',
                                            delay_s=None if reservation or observe_gc else .1,
                                            reservation_min_hold_s=0 if reservation else None,
                                            hard_gate_deadline_s=None if observe_gc else .25) if opportunity else {})))
            dump(path/'workload.json', selected)
            if gc_liveness:
                history_holder = read(args.liveness_history)
                assert history_holder['status'] == 'COMPLETE'
                assert len(history_holder['requests']) == 384
                assert len(history_holder['output_events']) == 381367
                if not retain_history:
                    history_holder = None
            if observe_gc:
                import gc
                # Equal setup before this cell's external arrival origin. Never collect
                # or change GC settings inside the measured service window.
                cleanup_started = time.perf_counter()
                collected = gc.collect()
                dump(path/'liveness-setup.json', dict(history_retained=retain_history if gc_liveness else False,
                    history_sha256=history_sha if gc_liveness else None,
                    history_events=381367 if gc_liveness else 0,
                    common_collect_seconds=time.perf_counter()-cleanup_started,
                    common_collected_objects=collected, gc_enabled=gc.isenabled(),
                    gc_thresholds=gc.get_threshold(), setup_before_current_external_arrivals=True))
            print('CELL_BEGIN', label, flush=True)
            started = time.perf_counter()
            try:
                raw = measure_episode(engine, selected, cc, 'steady', 1., label, max_seconds=240,
                                      record_preemptions=True, **(dict(record_gc=True) if observe_gc else {}))
            finally:
                report = uninstall()
                dump(path/'admission.json', report)
            if gc_liveness:
                assert (history_holder is not None) == retain_history
                if retain_history:
                    assert len(history_holder['output_events']) == 381367
                del history_holder
            dump(path/'raw.json', raw)
            dump(path/'post-request-drain.json', drain(engine))
            dump(path/'timing.json', dict(warmup_s=started-warm_started,
                service_s=raw['observation_end_s'], drain_s=time.perf_counter()-started-raw['observation_end_s']))
            print('CELL_END', label, raw['status'], raw['observation_end_s'],
                  'preempt',raw['actual_preemption_count'], flush=True)
            if raw['status'] != 'COMPLETE':
                raise RuntimeError('Incomplete cell '+label+': '+str(raw['error']))
            return raw

        if opportunity:
            best = 192 if gc_liveness else 256
            for i, variant in enumerate(protocol['test_order']):
                mode = ('fixed' if gc_liveness or variant in ('fixed128', 'fixed177', 'fixed192') else
                        'declared_budget' if variant == 'declaredbudget' else
                        'mc_budget' if variant == 'mcbudget' else
                        'mc_budget_capped' if variant == 'mccapped' else
                        'mc_budget_phase' if variant == 'mcphase' else
                        'mc_budget_static' if variant == 'mcstatic' else 'kv')
                cap = (128 if variant == 'fixed128' else 177 if variant == 'fixed177' else
                       192 if variant == 'fixed192' else best)
                cell(f'probe-{i:02d}-{variant}', mode, cap, test,
                    probe_enabled=variant in ('delay', 'timer', 'progress'),
                    release_mode='output_progress' if variant == 'progress' else 'timer',
                    reservation_enabled=variant in ('reservation', 'headroom32'),
                    retain_history=gc_liveness and variant == 'retained')
        elif accepted is None:
            if not pressure:
                cell('dev-native'+str(engine_max), 'native', engine_max, dev)
            scores=[]
            for cap in protocol['dev_caps']:
                raw = cell('dev-cap'+str(cap), 'fixed', cap, dev)
                score=statistics.mean(r['completion_s']-r['arrival_s'] for r in raw['requests'])
                scores.append(dict(cap=cap, mean_flow_s=score, preemptions=raw['actual_preemption_count']))
                del raw
            best = min(scores, key=lambda s:(s['mean_flow_s'], -s['cap']))['cap']
            selection = dict(scores=scores, selected_cap=best, pro_inputs_file_sha256=pro_input_sha)
            if pressure:
                selection.update(profile=args.profile, population=384, trace_sha256=pressure_trace_sha,
                    workload_sha256=protocol['workload_sha256'], selected_pressure_gap_s=.1,
                    dev_caps=protocol['dev_caps'], selection=protocol['selection'],
                    fixed_baseline_claim_scope=protocol['fixed_baseline_claim_scope'],
                    evidence_role=protocol['evidence_role'],
                    test_admission_caps=dict(fixed=best, kv=256, recovery=256))
            if args.profile == 'pro-dev':
                probes=[]
                for probe in protocol['pressure_probes']:
                    raw = cell('probe-'+probe['label'], 'native', engine_max,
                               subset(test,0,probe['count'],probe['gap']))
                    probes.append(dict(probe, actual_preemptions=raw['actual_preemption_count']))
                    del raw
                chosen = probes[1] if probes[1]['actual_preemptions'] > 0 else probes[2]
                selection.update(pressure_probes=probes, selected_pressure=chosen['label'],
                                 selected_pressure_gap_s=chosen['gap'])
            dump(out/'selection.json', selection)
        else:
            best = accepted['selected_cap']
        if not opportunity and args.profile not in ('pro-dev', 'pro-pressure-dev'):
            for i, mode in enumerate(protocol['test_order']):
                cap = engine_max if pressure and mode in ('kv', 'recovery') else best
                cell(f'test-{i:02d}-{mode}', mode, cap, test)
        dump(out/'status.json', dict(status='COMPLETE',
            selected_cap=None if simple_recheck or cadence_scan else best, finished_unix=time.time()))
    except BaseException as exc:
        dump(out/'status.json', dict(status='FAILED', error=repr(exc), traceback=traceback.format_exc()))
        raise
    finally:
        if engine is not None:
            engine.engine_core.shutdown()
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
