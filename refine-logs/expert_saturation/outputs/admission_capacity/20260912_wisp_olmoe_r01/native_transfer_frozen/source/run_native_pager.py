"""Bounded OLMoE/vLLM 0.26 integration probe using the upstream WiSP pager.

This is a runtime/cost probe, not a scheduler result. Both full-cap and paged
arms use the same BF16 TRITON wrapper and explicit KV bytes. Original native
request capture is reused; scheduler counts do not establish token row order.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import traceback


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def gpu_state():
    def query(*args):
        return subprocess.check_output(["nvidia-smi", *args], text=True).strip()
    return dict(gpu=query("--query-gpu=uuid,name,memory.total,memory.used,temperature.gpu,power.draw",
                          "--format=csv,noheader"),
                compute_pids=query("--query-compute-apps=pid", "--format=csv,noheader"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--prepared", type=Path, required=True)
    p.add_argument("--model", required=True, help="Existing pinned local snapshot only")
    p.add_argument("--expert-cap", type=int, choices=[24, 64], required=True)
    p.add_argument("--prompt-tokens", type=int, default=64)
    p.add_argument("--output-tokens", type=int, default=8)
    p.add_argument("--requests", type=int, default=4)
    p.add_argument("--token-budget", type=int, default=128)
    p.add_argument("--kv-bytes", type=int, default=1073741824)
    p.add_argument("--arrival-interval", type=float, default=0.05)
    p.add_argument("--max-seconds", type=float, default=180)
    p.add_argument("--verify-kernel", action="store_true")
    p.add_argument("--prefill-limit", type=int, default=0)
    p.add_argument("--warmup-full", action="store_true")
    args = p.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    dump(out / "config.json", {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    (out / "commands.txt").write_text(shlex.join([sys.executable, *sys.argv]) + "\n")
    dump(out / "status.json", dict(status="INITIALIZING", started_unix_s=time.time()))
    engine, pager = None, None
    try:
        before = gpu_state()
        if before["compute_pids"]:
            raise RuntimeError("GPU has another compute process; this attempt did not initialize CUDA")
        if not Path(args.model).is_dir():
            raise ValueError("local model snapshot missing")
        frozen = json.loads((args.prepared / "workload.json").read_text())
        if not 1 <= args.requests <= 16 or not 2 <= args.output_tokens <= 128:
            raise ValueError("bounded pilot requires 1..16 requests and 2..128 output tokens")
        if len(frozen["source_requests"]) < args.requests:
            raise ValueError("insufficient prepared requests")
        prompts = frozen["actual_prompt_token_ids"][:args.requests]
        if any(len(ids) < args.prompt_tokens for ids in prompts):
            raise ValueError("requested prompt exceeds prepared input")
        workload = dict(source_requests=frozen["source_requests"][:args.requests],
                        actual_prompt_token_ids=[ids[:args.prompt_tokens] for ids in prompts],
                        arrival_traces_s={"steady": [i * args.arrival_interval for i in range(args.requests)]})
        dump(out / "workload.json", workload)
        os.environ.update(VLLM_ENABLE_V1_MULTIPROCESSING="0", VLLM_BATCH_INVARIANT="0",
                          VLLM_USE_FLASHINFER_SAMPLER="0", WISP_PLUGIN_DISABLE="1",
                          HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        import torch
        import vllm
        from vllm.engine.arg_utils import EngineArgs
        from vllm.v1.engine.llm_engine import LLMEngine
        import wisp_v026_adapter as pager
        from native_pager_context import install_row_context
        from native_capture import capture_episode, set_empty_admission_cap
        assert vllm.__version__ == "0.26.0" and torch.cuda.device_count() == 1
        root = Path(__file__).parent
        dump(out / "environment.json", dict(python=sys.version, torch=torch.__version__,
            cuda=torch.version.cuda, vllm=vllm.__version__, gpu_before=before,
            cgroup_memory_max=Path("/sys/fs/cgroup/memory.max").read_text().strip(),
            sources={n: hashlib.sha256((root / n).read_bytes()).hexdigest() for n in
                     ["run_native_pager.py", "wisp_v026_adapter.py", "native_pager_context.py",
                      "native_capture.py", "admission_feedback.py"]}))
        pager.install(args.expert_cap, out / "pager")
        if args.verify_kernel:
            pager.enable_validation()
        kwargs = dict(model=args.model, tokenizer=args.model, dtype="bfloat16", seed=20260912,
            max_model_len=max(512, args.prompt_tokens + args.output_tokens),
            max_num_seqs=16, max_num_batched_tokens=args.token_budget,
            long_prefill_token_threshold=0 if args.warmup_full else args.prefill_limit,
            kv_cache_memory_bytes=args.kv_bytes, gpu_memory_utilization=0.9,
            enable_chunked_prefill=True, enable_prefix_caching=False,
            scheduling_policy="fcfs", async_scheduling=False, stream_interval=1,
            enforce_eager=True, enable_return_routed_experts=False,
            kernel_config={"moe_backend": "triton"})
        dump(out / "engine_args.json", kwargs)
        engine = LLMEngine.from_engine_args(EngineArgs(**kwargs), enable_multiprocessing=False)
        core = engine.engine_core.engine_core
        runner = core.model_executor.driver_worker.worker.model_runner
        install_row_context(runner, pager)
        cfg = runner.model.config
        shape = dict(layers=cfg.num_hidden_layers, experts=cfg.num_experts,
                     top_k=cfg.num_experts_per_tok, hidden=cfg.hidden_size,
                     intermediate=cfg.intermediate_size)
        dump(out / "model_shape.json", shape)
        assert shape["layers"] == 16 and shape["experts"] == 64 and shape["top_k"] == 8
        cache = engine.vllm_config.cache_config
        pool = core.scheduler.kv_cache_manager.block_pool
        dump(out / "resources.json", dict(kv_cache_memory_bytes=cache.kv_cache_memory_bytes,
            num_gpu_blocks=cache.num_gpu_blocks, block_size=cache.block_size,
            pool_num_gpu_blocks=pool.num_gpu_blocks,
            free_blocks_after_init=pool.get_num_free_blocks(),
            allocated_cuda_bytes=torch.cuda.memory_allocated(),
            peak_allocated_cuda_bytes=torch.cuda.max_memory_allocated(),
            expert_cap=args.expert_cap))
        set_empty_admission_cap(engine, args.requests)
        scheduler = core.scheduler
        original = scheduler.schedule
        phase, step = "warmup", 0

        def scheduled_context(*a, **kw):
            nonlocal step
            result = original(*a, **kw)
            pager.set_context(phase=phase, step_id=step,
                scheduled_tokens=dict(result.num_scheduled_tokens),
                expected_rows=sum(result.num_scheduled_tokens.values()),
                row_request_order_verified=False)
            step += 1
            return result

        scheduler.schedule = scheduled_context
        config = dict(output_tokens=args.output_tokens, cap=args.requests,
                      policy="static", allow_preemption=True)
        warmup = dict(workload, source_requests=workload["source_requests"][:1],
            actual_prompt_token_ids=workload["actual_prompt_token_ids"][:1],
            arrival_traces_s={"steady": [0.0]})
        raw = capture_episode(engine, warmup, dict(config, output_tokens=2),
            regime="steady", arrival_scale=1, run_id="warmup", max_seconds=args.max_seconds)
        dump(out / "warmup.json", raw)
        if raw["status"] != "COMPLETE":
            raise RuntimeError("warmup incomplete: " + str(raw["error"]))
        if args.warmup_full:
            raw = capture_episode(engine, workload, config, regime="steady", arrival_scale=1,
                run_id="warmup-full", max_seconds=args.max_seconds)
            dump(out / "warmup_full.json", raw)
            if raw["status"] != "COMPLETE":
                raise RuntimeError("full warmup incomplete: " + str(raw["error"]))
        assert not engine.has_unfinished_requests() and not scheduler.requests
        prior_limit = scheduler.scheduler_config.long_prefill_token_threshold
        scheduler.scheduler_config.long_prefill_token_threshold = args.prefill_limit
        dump(out / "policy_application.json", dict(prior_prefill_limit=prior_limit,
            measured_prefill_limit=args.prefill_limit, applied_on_drained_engine=True,
            common_full_warmup_prefill_limit=0 if args.warmup_full else None))
        phase, step = "measurement", 0
        pager.begin_measurement()
        dump(out / "status.json", dict(status="MEASURING", started_unix_s=time.time()))
        raw = capture_episode(engine, workload, config, regime="steady", arrival_scale=1,
            run_id="measurement", max_seconds=args.max_seconds)
        dump(out / "raw.json", raw)
        summary = pager.finalize()
        dump(out / "pager_summary.json", summary)
        if raw["status"] != "COMPLETE":
            raise RuntimeError("measurement incomplete: " + str(raw["error"]))
        assert len(raw["requests"]) == args.requests
        assert all(len(r["output_token_ids"]) == args.output_tokens for r in raw["requests"])
        dump(out / "status.json", dict(status="COMPLETE", finished_unix_s=time.time(),
            evidence_type=("MATCHED_LAYER_KERNEL_VALIDATION_ONLY" if args.verify_kernel
                           else "NATIVE_EAGER_PAGER_INTEGRATION_ONLY"), gpu_after=gpu_state(),
            requests_completed=args.requests, generated_tokens=args.requests * args.output_tokens,
            observation_end_s=raw["observation_end_s"]))
    except BaseException as exc:
        (out / "exception.txt").write_text(traceback.format_exc())
        if pager is not None and pager._runtime is not None:
            try:
                dump(out / "pager_failure_summary.json", pager.finalize(status="failed"))
            except Exception:
                (out / "pager_finalize_error.txt").write_text(traceback.format_exc())
        dump(out / "status.json", dict(status="FAILED", error=f"{type(exc).__name__}: {exc}",
                                      finished_unix_s=time.time()))
        raise
    finally:
        if engine is not None:
            engine.engine_core.shutdown()


if __name__ == "__main__":
    main()
