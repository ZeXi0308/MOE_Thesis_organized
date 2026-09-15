#!/usr/bin/env python3
"""One natural-length open-population native-save diagnostic; no performance comparison."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

KV_BYTES = (4096 + 1) * 2097152
RUNTIME_SOURCES = ['v1/core/sched/scheduler.py', 'v1/core/kv_cache_manager.py',
    'v1/core/block_pool.py', 'v1/worker/gpu_model_runner.py', 'v1/core/kv_cache_coordinator.py',
    'v1/core/single_type_kv_cache_manager.py', 'v1/core/kv_cache_utils.py', 'config/model.py']


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_inputs(folder):
    config, workload = read(folder/'config.json'), read(folder/'workload.json')
    if hashlib.sha256(json.dumps(workload, sort_keys=True).encode()).hexdigest() != config['workload_sha256']:
        raise ValueError('workload identity changed')
    rows, prompts = workload['source_requests'], workload['actual_prompt_token_ids']
    if len(rows) != len(prompts) or len({r['request_id'] for r in rows}) != len(rows):
        raise ValueError('request identities do not align')
    for row, ids in zip(rows, prompts):
        if not ids or hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest() != row['prompt_token_ids_sha256']:
            raise ValueError('prompt token identity mismatch')
    return config, workload


def gpu_state():
    processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory',
        '--format=csv,noheader'], text=True).strip()
    if any(row and int(row[0]) != os.getpid() for row in csv.reader(processes.splitlines())):
        raise RuntimeError(f'GPU has another compute process: {processes!r}')
    return dict(compute_processes=processes, device=subprocess.check_output(['nvidia-smi',
        '--query-gpu=name,uuid,memory.total,memory.used,temperature.gpu,power.draw,clocks.sm',
        '--format=csv,noheader'], text=True).strip())


def eos_metadata(engine):
    """Read the installed ModelConfig API; never modify generation defaults."""
    try:
        model = engine.vllm_config.model_config
        generation = model.try_get_generation_config()
        return dict(status='READ', hf_eos_token_id=model.hf_config.to_dict().get('eos_token_id'),
            generation_config_source=model.generation_config,
            generation_config=generation, generation_overrides=model.override_generation_config,
            effective_sampling_defaults=model.get_diff_sampling_param(),
            scope='ModelConfig metadata; actual request finish/stop reasons come from capture.')
    except Exception as error:
        return dict(status='UNAVAILABLE', error=f'{type(error).__name__}: {error}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--warmup-inputs', type=Path, required=True)
    parser.add_argument('--variant', choices=['current'], default='current')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--commit-recheck', action='store_true', help='Recheck direct target funding at commit; default retains existing swap')
    args = parser.parse_args()
    root, out, engine = Path(__file__).resolve().parent, args.output_dir, None
    timing = dict(process_start_unix_s=time.time(), process_start_perf_s=time.perf_counter())
    out.mkdir(parents=True, exist_ok=False)
    dump(out/'status.json', dict(status='INITIALIZING'))
    (out/'commands.txt').write_text(shlex.join([sys.executable, *sys.argv]) + '\n')
    try:
        from metrics import summarize_episode_requests
        from native_capture import capture_episode, set_empty_admission_cap
        from memory_telemetry import capture_with_memory, memory_snapshot
        from safe_static import qualify_safe_cap
        from absence_rotation import RotationConfig
        from staged_store_rotation import install as install_rotation
        from native_offload_observer import install as observe_offload, drain as drain_offload
        from native_host_snapshot import snapshot as host_snapshot
        config, workload = load_inputs(args.inputs)
        warmups = {d: load_inputs(args.warmup_inputs/d) for d in ('short', 'long')}
        lengths = [len(ids) for ids in workload['actual_prompt_token_ids']]
        if (len(lengths), min(lengths), max(lengths)) != (64, 334, 3011) or workload['arrival_traces_s']['steady'] != [i * .2 for i in range(64)]:
            raise ValueError('requires the frozen 64 natural prompts and steady 0.2 s arrivals')
        if any(wc['model'] != config['model'] for wc, _ in warmups.values()):
            raise ValueError('warmup and measurement model identities differ')
        config.update(requests=64, prompt_tokens=max(lengths), output_tokens=1024, output_mode='eos',
            output_tokens_by_request={}, cap=32, engine_max_num_seqs=32, policy='static', max_seconds=180,
            variant=args.variant, recheck_commit_funding=args.commit_recheck, population_mode='open', preemption_mode='native_recompute_or_external_kv',
            measurement_mode='diagnostic', selective_save='on', store_scope='native_full', offload_gib=16,
            fixed_kv_cache_memory_bytes=KV_BYTES, gpu_memory_utilization=.90, reservation_policy='full',
            rotation_config=vars(RotationConfig()), rotation_victim_order='most_output',
            prompt_tokens_role='maximum observed input length; output_tokens is requested cap',
            evidence_ceiling='NATIVE_INPROCESS_DIAGNOSTIC',
            metric_role='action existence / natural EOS qualification; native save open adapter and capture costs included; not a performance comparison')
        dump(out/'config.json', config)
        before = gpu_state()
        os.environ.update(VLLM_ENABLE_V1_MULTIPROCESSING='0', VLLM_BATCH_INVARIANT='0', VLLM_USE_SIMPLE_KV_OFFLOAD='0')
        import torch
        import vllm
        from importlib.metadata import version
        from vllm.engine.arg_utils import EngineArgs
        from vllm.v1.engine.llm_engine import LLMEngine
        if vllm.__version__ != '0.26.0' or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError('requires existing vLLM0.26 single-GPU environment')
        runtime_hashes = {name: sha(Path(vllm.__file__).parent/name) for name in RUNTIME_SOURCES}
        dump(out/'environment.json', dict(python=sys.version, torch=torch.__version__, cuda=torch.version.cuda,
            vllm=vllm.__version__, transformers=version('transformers'), gpu_before=before,
            cpu_threads=torch.get_num_threads(), vllm_source_sha256=runtime_hashes,
            source_sha256={name: sha(root/name) for name in ['run_streaming_recovery.py', 'native_capture.py',
                'memory_telemetry.py', 'metrics.py', 'safe_static.py', 'rotation_native.py', 'absence_rotation.py',
                'staged_store_rotation.py', 'staged_save_contract.py', 'native_offload_observer.py',
                'native_host_snapshot.py', 'native_store_delta.py', 'native_full_store_evidence.py', 'commit_disposition.py']}))
        if runtime_hashes['v1/core/sched/scheduler.py'] != '2ed2a550b6558b2495eda845a97ae38bcf0225027b9e25fbf00fc3880c1d3941' or runtime_hashes['v1/core/kv_cache_manager.py'] != '3f4af8d247f3fe9570b0132818b832b66ae6a2ac12942588828f899f6ff77ccf':
            raise RuntimeError('pinned runtime source changed')
        model = config['model']
        kwargs = dict(model=model['id'], revision=model['revision'], tokenizer_revision=model['tokenizer_revision'],
            dtype='bfloat16', seed=config['seed'], max_model_len=4096, max_num_seqs=32,
            max_num_batched_tokens=1024, gpu_memory_utilization=.90, enable_chunked_prefill=True,
            enable_prefix_caching=False, scheduling_policy='fcfs', async_scheduling=False,
            kv_cache_memory_bytes=KV_BYTES, scheduler_reserve_full_isl=True, stream_interval=1,
            enforce_eager=False, enable_return_routed_experts=False)
        kwargs.update(kv_offloading_size=16, kv_offloading_backend='native',
            kv_transfer_config={'kv_connector_extra_config': {'offload_prompt_only': False}})
        dump(out/'engine_args.json', kwargs)
        timing['engine_init_start_perf_s']=time.perf_counter()
        engine = LLMEngine.from_engine_args(EngineArgs(**kwargs), enable_multiprocessing=False)
        timing['engine_init_end_perf_s']=time.perf_counter()
        dump(out/'resolved-eos.json', eos_metadata(engine))
        host_init=host_snapshot(engine)
        dump(out/'host-after-init.json', host_init)
        if ((host_init.get('cpu_kv') or {}).get('unique_storage_bytes') != 16*1024**3
                or (host_init.get('manager') or {}).get('capacity_blocks') != 8192):
            raise RuntimeError('Actual unique host KV allocation/capacity differs from16GiB/8192blocks')
        memory = memory_snapshot(engine, torch)
        dump(out/'memory-after-init.json', memory)
        qualification = qualify_safe_cap(engine, config)
        qualification['observed_scheduler_reserve_full_isl'] = engine.engine_core.engine_core.scheduler.scheduler_reserve_full_isl
        dump(out/'safe-cap-qualification.json', qualification)
        if (qualification['status'] != 'QUALIFIED' or qualification['usable_blocks'] != 4096
                or not qualification['observed_scheduler_reserve_full_isl'] or memory['kv_storage_bytes'] != KV_BYTES):
            raise RuntimeError('actual KV pool/storage or initial-history reservation differs from fixed budget')
        timing['warmup_start_perf_s']=time.perf_counter()
        print('PHASE APPLICATION_WARMUP_BEGIN', flush=True)
        for index, (domain, count, cap) in enumerate([('short', 32, 16), ('short', 32, 32), ('long', 2, 2)]):
            wc, ww = warmups[domain]
            work = dict(ww, source_requests=ww['source_requests'][:count],
                actual_prompt_token_ids=ww['actual_prompt_token_ids'][:count], arrival_traces_s={'steady': [0.] * count})
            warm_config = dict(wc, cap=cap, requests=count, output_tokens=16, output_tokens_by_request={})
            warm_config.pop('output_mode', None)
            set_empty_admission_cap(engine, cap)
            raw = capture_with_memory(engine, capture_episode, work, warm_config,
                regime='steady', arrival_scale=1., run_id=f'warmup{index}', max_seconds=120)
            dump(out/f'warmup-{index}.json', raw)
            if raw['status'] != 'COMPLETE':
                raise RuntimeError('warmup incomplete; measurement unrun')
        dump(out/'warmup-offload-drain.json', drain_offload(engine))
        if not engine.reset_prefix_cache(reset_connector=True):
            raise RuntimeError('Failed to clear warmup native connector cache')
        dump(out/'warmup-cache-reset.json', dict(success=True))
        timing['warmup_end_perf_s']=time.perf_counter()
        print('PHASE APPLICATION_WARMUP_END', flush=True)
        set_empty_admission_cap(engine, 32)  # Verifies drain before installing either adapter arm.
        before = gpu_state()
        torch.cuda.reset_peak_memory_stats()
        dump(out/'memory-before.json', memory_snapshot(engine, torch))
        dump(out/'host-before.json', host_snapshot(engine))
        scheduler = engine.engine_core.engine_core.scheduler
        dump(out/'resolved-scheduler-config.json', dict(max_num_running_reqs=scheduler.max_num_running_reqs,
            long_prefill_token_threshold=scheduler.scheduler_config.long_prefill_token_threshold))
        offload_data, uninstall_offload = observe_offload(scheduler.connector, diagnostic=True,
            host_snapshot=lambda: host_snapshot(engine))
        selective, uninstall = install_rotation(scheduler, vllm_config=engine.vllm_config,
            block_size=qualification['block_size'], population_mode='open', save=True,
            global_cooldown_steps=20, diagnostic=True, store_scope='native_full',
            recheck_commit_funding=args.commit_recheck)
        if selective['recheck_commit_funding'] != config['recheck_commit_funding']:
            raise RuntimeError('Applied commit recheck switch differs from recorded config')
        if selective['store_scope']!='native_full' or selective['native_calc_overridden']:
            raise RuntimeError('Full native saving must retain the original calculation method')
        if selective['rotation_config'] != config['rotation_config']:
            raise RuntimeError('Applied rotation configuration differs from current baseline')
        raw = None
        try:
            try:
                timing['measurement_start_perf_s']=time.perf_counter()
                print('PHASE MEASUREMENT_BEGIN', flush=True)
                raw = capture_with_memory(engine, capture_episode, workload, config, allow_preemption=True,
                    regime='steady', arrival_scale=1., run_id='measured', max_seconds=180)
                timing['measurement_return_perf_s']=time.perf_counter()
                print('PHASE MEASUREMENT_END', flush=True)
                dump(out/'host-request-end.json', host_snapshot(engine))
                dump(out/'post-request-drain.json', drain_offload(engine))
                timing['post_request_drain_end_perf_s']=time.perf_counter()
            finally:
                dump(out/'selective-store.json', uninstall())
                dump(out/'offload-events.json', offload_data)
                uninstall_offload()
        finally:
            if raw is not None:
                raw['gpu_before'] = before
                raw['capture_preemption_mode_label'] = raw.get('preemption_mode')
                raw['preemption_mode'] = config['preemption_mode']
                dump(out/'raw.json', raw)
        dump(out/'memory-after.json', memory_snapshot(engine, torch))
        dump(out/'host-after.json', host_snapshot(engine))
        dump(out/'gpu-after.json', gpu_state())
        metrics = summarize_episode_requests(raw['requests'], observation_end_s=raw['observation_end_s'], ttft_slo_s=5., tpot_slo_s=.2)
        metrics.update(complete_episode_comparison_eligible=raw['status'] == 'COMPLETE',
            actual_preemption_count=raw['actual_preemption_count'],
            forced_rotations=selective['applied_rotations'],
            role=config['metric_role'])
        dump(out/'metrics.json', metrics)
        reasons = [str(r.get('finish_reason', r.get('stop_reason')) or 'unfinished') for r in raw['requests']]
        dump(out/'status.json', dict(status=raw['status'], capture_status=raw['status'],
            requests_completed=sum(r['status'] == 'completed' for r in raw['requests']),
            finish_reason_counts={reason: reasons.count(reason) for reason in sorted(set(reasons))},
            length_capped_requests=reasons.count('length'),
            forced_rotations=metrics['forced_rotations'], scientific_verdict='MEASUREMENT_ONLY', error=raw['error']))
        if raw['status'] != 'COMPLETE':
            raise RuntimeError(raw['error'] or raw['status'])
    except Exception as error:
        dump(out/'status.json', dict(read(out/'status.json'), status='INCOMPLETE', error=f'{type(error).__name__}: {error}'))
        raise
    finally:
        timing['shutdown_start_perf_s']=time.perf_counter()
        try:
            if engine is not None:
                engine.engine_core.shutdown()
        finally:
            timing['process_end_perf_s']=time.perf_counter()
            timing['scope']='Initialization, warmups, measured capture, later drain and shutdown are distinct host intervals; diagnostic capture is not a lightweight performance claim.'
            dump(out/'timing.json', timing)


if __name__ == '__main__':
    main()
