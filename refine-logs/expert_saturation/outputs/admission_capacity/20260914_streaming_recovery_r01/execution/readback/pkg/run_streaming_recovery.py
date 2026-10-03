#!/usr/bin/env python3
"""Matched native/most open-adapter diagnostic; includes common capture costs."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

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
    parser.add_argument('--variant', choices=['native', 'most_output'], required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    root, out, engine = Path(__file__).resolve().parent, args.output_dir, None
    out.mkdir(parents=True, exist_ok=False)
    dump(out/'status.json', dict(status='INITIALIZING'))
    (out/'commands.txt').write_text(shlex.join([sys.executable, *sys.argv]) + '\n')
    try:
        from metrics import summarize_episode_requests
        from native_capture import capture_episode, set_empty_admission_cap
        from memory_telemetry import capture_with_memory, memory_snapshot
        from safe_static import qualify_safe_cap
        from absence_rotation import RotationConfig
        from rotation_native import install as install_rotation
        config, workload = load_inputs(args.inputs)
        warmups = {d: load_inputs(args.warmup_inputs/d) for d in ('short', 'long')}
        lengths = [len(ids) for ids in workload['actual_prompt_token_ids']]
        if (len(lengths), min(lengths), max(lengths)) != (64, 334, 3011) or workload['arrival_traces_s']['steady'] != [i * .5 for i in range(64)]:
            raise ValueError('requires the frozen 64 natural prompts and steady 0.5 s arrivals')
        if any(wc['model'] != config['model'] for wc, _ in warmups.values()):
            raise ValueError('warmup and measurement model identities differ')
        config.update(requests=64, prompt_tokens=max(lengths), output_tokens=1024, output_mode='eos',
            output_tokens_by_request={}, cap=32, engine_max_num_seqs=32, policy='static', max_seconds=180,
            variant=args.variant, population_mode='open', preemption_mode='native_recompute',
            fixed_kv_cache_memory_bytes=KV_BYTES, gpu_memory_utilization=.90, reservation_policy='full',
            rotation_config=vars(RotationConfig()), rotation_victim_order='most_output',
            prompt_tokens_role='maximum observed input length; output_tokens is requested cap',
            evidence_ceiling='NATIVE_INPROCESS_DIAGNOSTIC',
            metric_role='action existence / natural EOS qualification; shared open adapter and capture costs included')
        dump(out/'config.json', config)
        before = gpu_state()
        os.environ.update(VLLM_ENABLE_V1_MULTIPROCESSING='0', VLLM_BATCH_INVARIANT='0')
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
                'memory_telemetry.py', 'metrics.py', 'safe_static.py', 'rotation_native.py', 'absence_rotation.py']}))
        if runtime_hashes['v1/core/sched/scheduler.py'] != '2ed2a550b6558b2495eda845a97ae38bcf0225027b9e25fbf00fc3880c1d3941' or runtime_hashes['v1/core/kv_cache_manager.py'] != '3f4af8d247f3fe9570b0132818b832b66ae6a2ac12942588828f899f6ff77ccf':
            raise RuntimeError('pinned runtime source changed')
        model = config['model']
        kwargs = dict(model=model['id'], revision=model['revision'], tokenizer_revision=model['tokenizer_revision'],
            dtype='bfloat16', seed=config['seed'], max_model_len=4096, max_num_seqs=32,
            max_num_batched_tokens=1024, gpu_memory_utilization=.90, enable_chunked_prefill=True,
            enable_prefix_caching=False, scheduling_policy='fcfs', async_scheduling=False,
            kv_cache_memory_bytes=KV_BYTES, scheduler_reserve_full_isl=True, stream_interval=1,
            enforce_eager=False, enable_return_routed_experts=False)
        dump(out/'engine_args.json', kwargs)
        engine = LLMEngine.from_engine_args(EngineArgs(**kwargs), enable_multiprocessing=False)
        dump(out/'resolved-eos.json', eos_metadata(engine))
        memory = memory_snapshot(engine, torch)
        dump(out/'memory-after-init.json', memory)
        qualification = qualify_safe_cap(engine, config)
        qualification['observed_scheduler_reserve_full_isl'] = engine.engine_core.engine_core.scheduler.scheduler_reserve_full_isl
        dump(out/'safe-cap-qualification.json', qualification)
        if (qualification['status'] != 'QUALIFIED' or qualification['usable_blocks'] != 4096
                or not qualification['observed_scheduler_reserve_full_isl'] or memory['kv_storage_bytes'] != KV_BYTES):
            raise RuntimeError('actual KV pool/storage or initial-history reservation differs from fixed budget')
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
        print('PHASE APPLICATION_WARMUP_END', flush=True)
        set_empty_admission_cap(engine, 32)  # Verifies drain before installing either adapter arm.
        before = gpu_state()
        torch.cuda.reset_peak_memory_stats()
        dump(out/'memory-before.json', memory_snapshot(engine, torch))
        scheduler = engine.engine_core.engine_core.scheduler
        dump(out/'resolved-scheduler-config.json', dict(max_num_running_reqs=scheduler.max_num_running_reqs,
            long_prefill_token_threshold=scheduler.scheduler_config.long_prefill_token_threshold))
        decisions, uninstall = install_rotation(scheduler, vllm_config=engine.vllm_config,
            block_size=qualification['block_size'], population_mode='open',
            mode='native' if args.variant == 'native' else 'rotate', victim_order='most_output',
            rotation_config=RotationConfig(**config['rotation_config']))
        try:
            print('PHASE MEASUREMENT_BEGIN', flush=True)
            raw = capture_with_memory(engine, capture_episode, workload, config, allow_preemption=True,
                regime='steady', arrival_scale=1., run_id='measured', max_seconds=180)
            print('PHASE MEASUREMENT_END', flush=True)
        finally:
            uninstall()
            dump(out/'headroom-decisions.json', decisions)
        raw['gpu_before'] = before
        dump(out/'raw.json', raw)
        dump(out/'memory-after.json', memory_snapshot(engine, torch))
        dump(out/'gpu-after.json', gpu_state())
        metrics = summarize_episode_requests(raw['requests'], observation_end_s=raw['observation_end_s'], ttft_slo_s=5., tpot_slo_s=.2)
        metrics.update(complete_episode_comparison_eligible=raw['status'] == 'COMPLETE',
            actual_preemption_count=raw['actual_preemption_count'],
            forced_rotations=sum(bool(d.get('forced_preempted')) for d in decisions),
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
        if engine is not None:
            engine.engine_core.shutdown()


if __name__ == '__main__':
    main()
