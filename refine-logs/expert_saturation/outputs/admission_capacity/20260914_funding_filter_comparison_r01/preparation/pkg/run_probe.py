#!/usr/bin/env python3
"""One context-calibration most/least/funding-filter component comparison."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from metrics import summarize_episode_requests
from native_capture import capture_episode, set_empty_admission_cap
from memory_telemetry import capture_with_memory, memory_snapshot
from safe_static import qualify_safe_cap
from completion_headroom import install
from absence_rotation import RotationConfig
from rotation_native import install as install_rotation
from ltr_recompute_native import install as install_component


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def read(path):
    return json.loads(path.read_text())


def gpu_state():
    processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory',
        '--format=csv,noheader'], text=True).strip()
    if any(row and int(row[0]) != os.getpid() for row in csv.reader(processes.splitlines())):
        raise RuntimeError(f'GPU has another compute process; compute_processes={processes!r}')
    return dict(compute_processes=processes, device=subprocess.check_output(['nvidia-smi',
        '--query-gpu=name,uuid,memory.total,memory.used,temperature.gpu,power.draw,clocks.sm',
        '--format=csv,noheader'], text=True).strip())


def load_inputs(root, domain):
    folder = root / 'inputs_preparation/prepared' / domain
    config, workload = read(folder/'config.json'), read(folder/'workload.json')
    if hashlib.sha256(json.dumps(workload, sort_keys=True).encode()).hexdigest() != config['workload_sha256']:
        raise ValueError('workload changed')
    rows, tokens = workload['source_requests'], workload['actual_prompt_token_ids']
    if len(rows) != 32 or len(tokens) != 32 or config['output_tokens'] != 1024:
        raise ValueError('frozen workload dimensions differ')
    lengths = config.get('prompt_lengths', [128 if domain == 'short' else 3072]*32)
    if len(lengths) != 32 or any(type(v) is not int or v < 1 or v + config['output_tokens'] > 4096 for v in lengths):
        raise ValueError('invalid native-context bounds')
    for row, ids, expected in zip(rows, tokens, lengths):
        if len(ids) != expected or hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest() != row['prompt_token_ids_sha256']:
            raise ValueError('prompt identity mismatch')
    return config, workload


def arm_spec(name):
    specs = {
        "native": dict(adapter="native", policy_family="native", completion_policy="native",
            victim_order="least_progress", boost=False, packing="native",
            complete_restores=False, reserve_ready_tokens=False),
        "most_output": dict(adapter="rotation", policy_family="strong_simple_rotation",
            completion_policy="rotate", victim_order="most_output", boost=False, packing="native",
            complete_restores=False, reserve_ready_tokens=False),
        "fit_scan": dict(adapter="component", policy_family="ltr_component",
            completion_policy="restore_token_reservation_fit_scan", victim_order="least_progress",
            boost=True, packing="fit_scan", complete_restores=False, reserve_ready_tokens=False),
        "guard_residual": dict(adapter="component", policy_family="ltr_component",
            completion_policy="restore_token_reservation_guard_residual", victim_order="least_progress",
            boost=True, packing="rank_prefix", complete_restores=True, reserve_ready_tokens=True),
    }
    specs['least_progress'] = dict(specs['most_output'], victim_order='least_progress')
    for value in specs.values():
        value['filter_victims_by_funding'] = False
    specs['least_feasible'] = dict(specs['least_progress'], filter_victims_by_funding=True)
    return dict(specs[name])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=['most_output','least_progress','least_feasible'], required=True)
    parser.add_argument('--domain', choices=['heterogeneous'], required=True)
    parser.add_argument('--reservation-policy', choices=['full'], required=True)
    parser.add_argument('--cap', type=int, choices=[32], required=True)
    parser.add_argument('--gpu-memory-utilization', type=float, choices=[0.90], required=True)
    parser.add_argument('--kv-cache-bytes', type=int, choices=[13960740864], required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    spec = arm_spec(args.variant)
    root, out = Path(__file__).resolve().parent, args.output_dir
    inputs = {d: load_inputs(root, d) for d in ['short', 'long', 'heterogeneous']}
    config, workload = inputs[args.domain]
    config = dict(config, cap=args.cap, domain=args.domain, engine_max_num_seqs=32,
        metric_role='continuous request metrics primary; SLO 5s/.2s reference only',
        evidence_ceiling='NATIVE_INPROCESS_CAPACITY_QUALIFICATION', max_seconds=120)
    config.update(requested_arm='budget90', reservation_policy=args.reservation_policy,
        completion_policy=spec['completion_policy'], policy_family=spec['policy_family'],
        variant=args.variant, headroom_observer='fast', fixed_kv_cache_memory_bytes=args.kv_cache_bytes,
        gpu_memory_utilization=args.gpu_memory_utilization, rotation_victim_order=spec['victim_order'],
        preemption_mode='native_recompute', rotation_config=vars(RotationConfig()),
        filter_victims_by_funding=spec['filter_victims_by_funding'])
    if spec['adapter'] == 'component':
        config['component'] = dict(boost=spec['boost'], threshold=200, quantum=10,
            base_order='FCFS', backend='priority packing / current-history reservation / native recompute',
            packing=spec['packing'], complete_restores=spec['complete_restores'],
            reserve_ready_tokens=spec['reserve_ready_tokens'])
    out.mkdir(parents=True, exist_ok=False)
    dump(out/'config.json', config)
    (out/'commands.txt').write_text(shlex.join([sys.executable, *sys.argv])+'\n')
    dump(out/'status.json', dict(status='INITIALIZING'))
    engine = None
    try:
        before = gpu_state()
        os.environ.update(VLLM_ENABLE_V1_MULTIPROCESSING='0', VLLM_BATCH_INVARIANT='0')
        import torch
        import vllm
        from importlib.metadata import version
        from vllm.engine.arg_utils import EngineArgs
        from vllm.v1.engine.llm_engine import LLMEngine
        if vllm.__version__ != '0.26.0' or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError('requires the existing vLLM0.26 one-GPU environment')
        dump(out/'environment.json', dict(python=sys.version, torch=torch.__version__,
            cuda=torch.version.cuda, vllm=vllm.__version__, transformers=version('transformers'),
            gpu_before=before, cpu_threads=torch.get_num_threads(),
            source_sha256={n: hashlib.sha256((root/n).read_bytes()).hexdigest() for n in
                ['run_probe.py', 'memory_telemetry.py', 'native_capture.py', 'metrics.py', 'safe_static.py', 'completion_headroom.py', 'absence_rotation.py', 'rotation_native.py', 'ltr_recompute_native.py', 'recovery_service_components.py', 'restore_obligation.py']},
            vllm_source_sha256={n: hashlib.sha256((Path(vllm.__file__).parent/n).read_bytes()).hexdigest()
                for n in ['v1/core/sched/scheduler.py', 'v1/core/kv_cache_manager.py', 'v1/core/block_pool.py',
                          'v1/worker/gpu_model_runner.py', 'v1/core/kv_cache_coordinator.py',
                          'v1/core/single_type_kv_cache_manager.py', 'v1/core/kv_cache_utils.py']}))
        expected_runtime_sources = {'v1/core/sched/scheduler.py': '2ed2a550b6558b2495eda845a97ae38bcf0225027b9e25fbf00fc3880c1d3941', 'v1/core/kv_cache_manager.py': '3f4af8d247f3fe9570b0132818b832b66ae6a2ac12942588828f899f6ff77ccf'}
        for name, digest in expected_runtime_sources.items():
            if hashlib.sha256((Path(vllm.__file__).parent/name).read_bytes()).hexdigest() != digest:
                raise RuntimeError('pinned runtime source changed: ' + name)
        model = config['model']
        kwargs = dict(model=model['id'], revision=model['revision'], tokenizer_revision=model['tokenizer_revision'],
            dtype='bfloat16', seed=config['seed'], max_model_len=4096, max_num_seqs=32,
            max_num_batched_tokens=1024, gpu_memory_utilization=args.gpu_memory_utilization, enable_chunked_prefill=True,
            enable_prefix_caching=False, scheduling_policy='fcfs', async_scheduling=False,
            kv_cache_memory_bytes=args.kv_cache_bytes, scheduler_reserve_full_isl=args.reservation_policy == 'full',
            stream_interval=1, enforce_eager=False, enable_return_routed_experts=False)
        dump(out/'engine_args.json', kwargs)
        engine = LLMEngine.from_engine_args(EngineArgs(**kwargs), enable_multiprocessing=False)
        dump(out/'memory-after-init.json', memory_snapshot(engine, torch))
        qualification = qualify_safe_cap(engine, dict(config, prompt_tokens=config['prompt_tokens_max']))
        qualification.update(reservation_basis='worst request upper bound, not equal request sizes or time-varying demand',
            prompt_lengths=config['prompt_lengths'],
            summed_full_length_blocks=sum((v + config['output_tokens'] + qualification.get('block_size',16)-1)//qualification.get('block_size',16) for v in config['prompt_lengths']))
        observed = engine.engine_core.engine_core.scheduler.scheduler_reserve_full_isl
        qualification['observed_scheduler_reserve_full_isl'] = observed
        if observed != (args.reservation_policy == 'full') or qualification.get('usable_blocks') != {17181966336: 8192, 16089350144: 7671, 15034482688: 7168, 13960740864: 6656}[args.kv_cache_bytes]:
            raise RuntimeError('actual reservation policy or KV pool differs from fixed experiment')
        qualification.update(gpu_memory_utilization=args.gpu_memory_utilization,
            full_reservation_required=args.gpu_memory_utilization == 0.95,
            full_reservation_role='sufficient for full upper-bound residency; not necessary for native no-preemption')
        if qualification['status'] == 'QUALIFIED':
            required = args.cap * qualification['per_request_reserved_blocks']
            sufficient = qualification['usable_blocks'] >= required
            qualification.update(full_cap_reserved_blocks=required, full_reservation_sufficient=sufficient)
            if args.gpu_memory_utilization == 0.95 and not sufficient:
                qualification.update(status='QUALIFICATION_FAILED', measurement_status='UNRUN',
                    error='budget95 usable blocks below the full cap32 reservation requirement')
        dump(out/'safe-cap-qualification.json', qualification)
        if qualification['status'] != 'QUALIFIED':
            dump(out/'status.json', dict(status='UNRUN', qualification_status='QUALIFICATION_FAILED',
                 error=qualification.get('error'), measurement_executed=False))
            raise SystemExit(2)
        print('PHASE APPLICATION_WARMUP_BEGIN', flush=True)
        # Identical warmups in every process; do not run the pressure case as warmup.
        for index, (domain, count, cap) in enumerate([('short', 32, 16), ('short', 32, 32), ('long', 2, 2)]):
            wc, ww = inputs[domain]
            warm_workload = dict(ww, source_requests=ww['source_requests'][:count],
                actual_prompt_token_ids=ww['actual_prompt_token_ids'][:count],
                arrival_traces_s={'steady': [0.0]*count})
            set_empty_admission_cap(engine, cap)
            raw = capture_with_memory(engine, capture_episode, warm_workload, dict(wc, cap=cap, requests=count, output_tokens=16),
                regime='steady', arrival_scale=1.0, run_id=f'warmup{index}', max_seconds=120)
            dump(out/f'warmup-{index}.json', raw)
            if raw['status'] != 'COMPLETE':
                raise RuntimeError('warmup did not complete; no primary episode run')
        print('PHASE APPLICATION_WARMUP_END', flush=True)
        set_empty_admission_cap(engine, args.cap)
        before = gpu_state()
        torch.cuda.reset_peak_memory_stats()
        dump(out/'memory-before.json', memory_snapshot(engine, torch))
        scheduler = engine.engine_core.engine_core.scheduler
        dump(out/'resolved-scheduler-config.json', dict(long_prefill_token_threshold=scheduler.scheduler_config.long_prefill_token_threshold))
        obligations = None
        if spec['adapter'] == 'component':
            decisions, counters, obligations, uninstall = install_component(scheduler,
                vllm_config=engine.vllm_config, block_size=qualification['block_size'],
                boost=True, threshold=200, quantum=10, packing=spec['packing'],
                complete_restores=spec['complete_restores'],
                reserve_ready_tokens=spec['reserve_ready_tokens'])
            decision_name = 'component-decisions.json'
        elif spec['adapter'] == 'rotation':
            decisions, uninstall = install_rotation(scheduler, vllm_config=engine.vllm_config,
                block_size=qualification['block_size'], expected_requests=32,
                rotation_config=RotationConfig(**config['rotation_config']), victim_order=spec['victim_order'],
                filter_victims_by_funding=spec['filter_victims_by_funding'])
            decision_name = 'headroom-decisions.json'
        else:
            decisions, uninstall = install(scheduler, vllm_config=engine.vllm_config,
                block_size=qualification['block_size'], expected_requests=32,
                mode='native', observer='fast')
            decision_name = 'headroom-decisions.json'
        try:
            print('PHASE MEASUREMENT_BEGIN', flush=True)
            raw = capture_with_memory(engine, capture_episode, workload, config,
                allow_preemption=True,
                regime='steady', arrival_scale=1.0, run_id='measured', max_seconds=120)
            print('PHASE MEASUREMENT_END', flush=True)
            if raw['status'] == 'COMPLETE' and obligations is not None:
                obligations.finalize(len(decisions), {}, [r['internal_request_id'] for r in raw['requests'] if r['status']=='completed'])
        finally:
            uninstall()
            dump(out/decision_name, decisions)
            if obligations is not None:
                dump(out/'restore-obligations.json', obligations.snapshot())
        raw['rotation_exposed'] = any(d.get('forced_preempted') for d in decisions)
        raw['calibration_role'] = 'retain complete no-action episodes; pressure and policy effects are separate'
        raw['gpu_before'] = before
        dump(out/'raw.json', raw)
        dump(out/'memory-after.json', memory_snapshot(engine, torch))
        dump(out/'gpu-after.json', gpu_state())
        metrics = summarize_episode_requests(raw['requests'], observation_end_s=raw['observation_end_s'],
            ttft_slo_s=5.0, tpot_slo_s=0.2)
        metrics.update(complete_episode_comparison_eligible=raw['status'] == 'COMPLETE',
            actual_preemption_count=raw['actual_preemption_count'], preemption_mode=raw['preemption_mode'],
            role='partial completion retained; incomplete cells are not full-horizon throughput comparisons')
        dump(out/'metrics.json', metrics)
        status = 'CAPACITY_BOUNDARY_STOP' if raw['capacity_boundary'] else raw['status']
        dump(out/'status.json', dict(status=status, requests_completed=sum(r['status']=='completed' for r in raw['requests']),
            actual_preemption_count=raw['actual_preemption_count'], preemption_mode=raw['preemption_mode'],
            scientific_verdict='MEASUREMENT_ONLY', error=raw['error']))
        if status not in ['COMPLETE', 'CAPACITY_BOUNDARY_STOP']:
            raise RuntimeError(raw['error'])
    except Exception as exc:
        previous = read(out/'status.json')
        dump(out/'status.json', dict(previous, status='INCOMPLETE', error=f'{type(exc).__name__}: {exc}'))
        raise
    finally:
        if engine is not None:
            engine.engine_core.shutdown()


if __name__ == '__main__':
    main()
