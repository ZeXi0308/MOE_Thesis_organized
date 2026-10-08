#!/usr/bin/env python3
"""One fresh-process native vLLM startup probe; no requests, search or KV override.

GPU path is untested. --help/--dry-run import neither torch nor vLLM.
Use one invocation per set, under the existing shared research GPU lock.
"""
from __future__ import annotations

import argparse
import fcntl
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parent
LOCK_PATH = Path('/root/autodl-tmp/moe-research-gpu.lock')
MODEL = '/root/autodl-tmp/moe-research-20261002/model'
GRAPH_SETS = {
    'compact': [1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
    # Archived CompilationConfig's native rule for max_num_seqs=256.
    'native': [1, 2, 4] + list(range(8, 256, 8)) + list(range(256, 513, 16)),
    'dense': [1, 2, 4] + list(range(8, 513, 8)),
}
ENVIRONMENT = {
    'VLLM_ENABLE_STARTUP_PLAN': '0',
    'VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS': '1',
    'VLLM_ENABLE_V1_MULTIPROCESSING': '0',
    'VLLM_USE_V2_MODEL_RUNNER': '0',
    'VLLM_USE_FLASHINFER_SAMPLER': '0',
    'HF_HUB_OFFLINE': '1',
    'TRANSFORMERS_OFFLINE': '1',
    'TOKENIZERS_PARALLELISM': 'false',
    'WISP_PLUGIN_DISABLE': '1',
    'OMP_NUM_THREADS': '8',
}
LIMITATIONS = [
    'Prototype untested on GPU; CPU validation does not establish backend correctness.',
    'Native profile estimates graph memory from representative captures before KV allocation.',
    'The complete live graph set is captured after KV allocation; no exact capture-first reallocation is implemented.',
    'capture_model returns its native device-free-memory delta, not an attribution of graph metadata alone.',
    'Allocator pool snapshots exclude non-PyTorch allocations; inactive reserved bytes are not necessarily releasable to KV.',
    'Startup-only probe does not establish SLO goodput, a speed-memory tradeoff, or service capacity pressure.',
    'Wall times include instrumentation; native allocator peaks may be reset by vLLM. An explicit cache cleanup is reported separately after startup warmup and never reallocates KV.',
]
_LOCK_HANDLE = None  # Held through interpreter exit, including engine teardown.


def engine_kwargs(plan: str, backend: str, model: str) -> dict:
    return dict(
        model=model, dtype='bfloat16', tensor_parallel_size=1,
        gpu_memory_utilization=0.9, max_model_len=4096,
        max_num_batched_tokens=2048, max_num_seqs=256, seed=20261008,
        moe_backend=backend, worker_cls='g_worker.GWorker',
        distributed_executor_backend='uni', enforce_eager=False,
        enable_chunked_prefill=True, enable_prefix_caching=False,
        async_scheduling=False, scheduling_policy='fcfs',
        scheduler_reserve_full_isl=True, long_prefill_token_threshold=0,
        enable_return_routed_experts=False,
        compilation_config=dict(cudagraph_mode='FULL_AND_PIECEWISE',
                                cudagraph_capture_sizes=GRAPH_SETS[plan],
                                max_cudagraph_capture_size=512),
    )


def write_result(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    print(json.dumps(dict(status=value['status'], output=str(path)), ensure_ascii=False), flush=True)


def record_only_torch_imports() -> list[str]:
    """Same narrow process-local orphan-module guard used in existing D runs."""
    dist = importlib.metadata.distribution('torch')
    if dist.files is None:
        raise RuntimeError('Torch RECORD unavailable; cannot validate orphan-module guard')
    recorded = {str(p) for p in dist.files}
    root = Path(dist.locate_file('torch'))
    import torch._dynamo.utils as dynamo_utils
    original = dynamo_utils.import_submodule

    def recorded_import(mod):
        if mod.__name__ != 'torch._inductor.kernel':
            return original(mod)
        # A clean RECORD-only package_view may be active through PYTHONPATH.
        # Its module paths are outside dist.locate_file('torch'); RECORD keys
        # identify modules by package-relative name, not that physical path.
        module_path = Path(*mod.__name__.split('.'))
        for path in sorted(Path(mod.__file__).parent.glob('*.py')):
            if not path.name.startswith('_') and str(module_path / path.name) in recorded:
                importlib.import_module(f'{mod.__name__}.{path.stem}')

    dynamo_utils.import_submodule = recorded_import
    return sorted(p.name for p in (root / '_inductor/kernel').glob('*.py')
                  if not p.name.startswith('_') and str(p.relative_to(root.parent)) not in recorded)


def smi(*args: str) -> str:
    return subprocess.check_output(['nvidia-smi', *args], text=True, timeout=15).strip()


def main() -> int:
    global _LOCK_HANDLE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', choices=tuple(GRAPH_SETS), required=True)
    parser.add_argument('--moe-backend', choices=('triton', 'flashinfer_cutlass'), default='triton')
    parser.add_argument('--model', default=MODEL)
    parser.add_argument('--output', type=Path, required=True, help='New JSON file; sibling .stages.jsonl contains startup snapshots')
    parser.add_argument('--dry-run', action='store_true', help='CPU-only argument manifest; no lock, nvidia-smi, torch or vLLM')
    args = parser.parse_args()
    output = args.output.resolve()
    stages = output.with_suffix('.stages.jsonl')
    if output.exists() or stages.exists():
        parser.error('Output or stage file exists; use a new output for each fresh process')
    output.parent.mkdir(parents=True, exist_ok=True)
    kwargs = engine_kwargs(args.plan, args.moe_backend, args.model)
    result = dict(status='DRY_RUN' if args.dry_run else 'PRECHECK', plan=args.plan,
                  engine_args=kwargs, environment=ENVIRONMENT, lock_path=str(LOCK_PATH),
                  stage_file=str(stages), pid=os.getpid(), limitations=LIMITATIONS,
                  gpu_experiment_launched=False)
    if args.dry_run:
        result['torch_imported'] = 'torch' in sys.modules
        result['vllm_imported'] = 'vllm' in sys.modules
        write_result(output, result)
        return 0

    engine = None
    try:
        _LOCK_HANDLE = LOCK_PATH.open('a+')
        try:
            fcntl.flock(_LOCK_HANDLE, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            result.update(status='LOCK_BUSY_NO_GPU_INITIALIZED')
            write_result(output, result)
            return 75
        processes = smi('--query-compute-apps=pid,process_name,used_gpu_memory', '--format=csv,noheader')
        result['compute_processes_precheck'] = processes
        if processes:
            result.update(status='GPU_BUSY_NO_GPU_INITIALIZED')
            write_result(output, result)
            return 75
        gpu_uuids = smi('--query-gpu=uuid', '--format=csv,noheader').splitlines()
        if len(gpu_uuids) != 1:
            raise RuntimeError('Probe requires the audited single-GPU environment')
        result['gpu_before'] = smi('--query-gpu=name,uuid,driver_version,memory.total,memory.used,memory.free,compute_cap', '--format=csv')
        if not Path(args.model).is_dir():
            raise FileNotFoundError(f'Existing local model missing: {args.model}')
        os.environ.update(ENVIRONMENT)
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_uuids[0].strip()
        os.environ['G_STARTUP_STAGE_FILE'] = str(stages)
        stages.touch(exist_ok=False)
        sys.path.insert(0, str(ROOT))
        result['orphan_torch_modules_skipped'] = record_only_torch_imports()
        import torch
        import vllm
        from vllm.engine.arg_utils import EngineArgs
        from vllm.v1.engine.llm_engine import LLMEngine
        torch.set_num_threads(8)
        result['versions'] = dict(torch=torch.__version__, vllm=vllm.__version__, cuda=torch.version.cuda,
                                  torch_module_file=torch.__file__, vllm_module_file=vllm.__file__)
        result.update(status='INITIALIZING', gpu_experiment_launched=True)
        write_result(output, result)
        start = time.perf_counter()
        engine = LLMEngine.from_engine_args(EngineArgs(**kwargs), enable_multiprocessing=False)
        result['engine_startup_wall_s'] = time.perf_counter() - start
        scheduler = engine.engine_core.engine_core.scheduler
        result['scheduler_kv_blocks_including_reserved_block'] = scheduler.kv_cache_manager.block_pool.num_gpu_blocks
        result['scheduler_free_kv_blocks'] = scheduler.kv_cache_manager.block_pool.get_num_free_blocks()
        result['cache_block_size_tokens'] = engine.vllm_config.cache_config.block_size
        config = engine.vllm_config
        manager = scheduler.kv_cache_manager
        result['observed_capacity_domain'] = dict(
            num_lookahead_tokens=scheduler.num_lookahead_tokens,
            watermark_blocks=manager.watermark_blocks,
            enable_prefix_caching=manager.enable_caching,
            async_scheduling=config.scheduler_config.async_scheduling,
            has_kv_transfer_config=config.kv_transfer_config is not None,
            has_speculative_config=config.speculative_config is not None,
            num_kv_cache_groups=manager.num_kv_cache_groups,
            scheduler_block_size_tokens=scheduler.block_size,
            kv_cache_spec_types=[type(g.kv_cache_spec).__name__
                                 for g in manager.kv_cache_config.kv_cache_groups],
            model_type=config.model_config.hf_config.model_type,
            scheduler_reserve_full_isl=scheduler.scheduler_reserve_full_isl,
        )
        result['resolved_capture_sizes'] = engine.vllm_config.compilation_config.cudagraph_capture_sizes
        result['resolved_cudagraph_mode'] = str(engine.vllm_config.compilation_config.cudagraph_mode)
        result['stage_events'] = [json.loads(line) for line in stages.read_text().splitlines() if line]
        result['gpu_after_startup'] = smi('--query-gpu=name,uuid,memory.used,memory.free', '--format=csv')
        result['status'] = 'STARTUP_COMPLETE_NO_SERVICE_RUN'
        write_result(output, result)
        return 0
    except Exception:
        result.update(status='ERROR', error=traceback.format_exc())
        if stages.exists():
            result['stage_events'] = [json.loads(line) for line in stages.read_text().splitlines() if line]
        write_result(output, result)
        return 1
    finally:
        if engine is not None:
            shutdown = getattr(engine.engine_core, 'shutdown', None)
            if callable(shutdown):
                shutdown()
        # Do not unlock early: process-global CUDA objects may still be alive.


if __name__ == '__main__':
    raise SystemExit(main())
