#!/usr/bin/env python3
"""One full native-preemption or guarded safe-cap episode."""
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


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def read(path):
    return json.loads(path.read_text())


def gpu_state():
    processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory',
        '--format=csv,noheader'], text=True).strip()
    if any(row and int(row[0]) != os.getpid() for row in csv.reader(processes.splitlines())):
        raise RuntimeError('GPU has another compute process')
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
    expected = 128 if domain == 'short' else 3072
    for row, ids in zip(rows, tokens):
        if len(ids) != expected or hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest() != row['prompt_token_ids_sha256']:
            raise ValueError('prompt identity mismatch')
    return config, workload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--domain', choices=['short', 'long'], required=True)
    parser.add_argument('--cap', type=int, choices=[32], required=True)
    arm = parser.add_mutually_exclusive_group(required=True)
    arm.add_argument('--safe-static', action='store_true')
    arm.add_argument('--native-preemption', action='store_true')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    root, out = Path(__file__).resolve().parent, args.output_dir
    inputs = {d: load_inputs(root, d) for d in ['short', 'long']}
    config, workload = inputs[args.domain]
    config = dict(config, cap=args.cap, domain=args.domain, engine_max_num_seqs=32,
        metric_role='continuous request metrics primary; SLO 5s/.2s reference only',
        evidence_ceiling='NATIVE_INPROCESS_CAPACITY_QUALIFICATION', max_seconds=120)
    config.update(requested_arm='safe' if args.safe_static else 'native32',
                  cap=None if args.safe_static else 32,
                  preemption_mode='stop_before_preemption' if args.safe_static else 'native_recompute')
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
                ['run_probe.py', 'memory_telemetry.py', 'native_capture.py', 'metrics.py', 'safe_static.py']},
            vllm_source_sha256={n: hashlib.sha256((Path(vllm.__file__).parent/n).read_bytes()).hexdigest()
                for n in ['v1/core/sched/scheduler.py', 'v1/core/kv_cache_manager.py', 'v1/core/block_pool.py',
                          'v1/worker/gpu_model_runner.py']}))
        model = config['model']
        kwargs = dict(model=model['id'], revision=model['revision'], tokenizer_revision=model['tokenizer_revision'],
            dtype='bfloat16', seed=config['seed'], max_model_len=4096, max_num_seqs=32,
            max_num_batched_tokens=1024, gpu_memory_utilization=0.90, enable_chunked_prefill=True,
            enable_prefix_caching=False, scheduling_policy='fcfs', async_scheduling=False,
            stream_interval=1, enforce_eager=False, enable_return_routed_experts=False)
        dump(out/'engine_args.json', kwargs)
        engine = LLMEngine.from_engine_args(EngineArgs(**kwargs), enable_multiprocessing=False)
        dump(out/'memory-after-init.json', memory_snapshot(engine, torch))
        qualification = qualify_safe_cap(engine, config)
        if qualification['status'] == 'QUALIFIED' and qualification['safe_cap'] != 29:
            qualification.update(status='QUALIFICATION_FAILED', measurement_status='UNRUN',
                                 error='live safe cap differs from the frozen safe29 comparison')
        dump(out/'safe-cap-qualification.json', qualification)
        if qualification['status'] != 'QUALIFIED':
            dump(out/'status.json', dict(status='UNRUN', qualification_status='QUALIFICATION_FAILED',
                 error=qualification.get('error'), measurement_executed=False))
            raise SystemExit(2)
        args.cap = qualification['safe_cap'] if args.safe_static else 32
        config.update(cap=args.cap)
        dump(out/'config.json', config)
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
        set_empty_admission_cap(engine, args.cap)
        before = gpu_state()
        torch.cuda.reset_peak_memory_stats()
        dump(out/'memory-before.json', memory_snapshot(engine, torch))
        raw = capture_with_memory(engine, capture_episode, workload, config,
            allow_preemption=args.native_preemption,
            regime='steady', arrival_scale=1.0, run_id='measured', max_seconds=120)
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
