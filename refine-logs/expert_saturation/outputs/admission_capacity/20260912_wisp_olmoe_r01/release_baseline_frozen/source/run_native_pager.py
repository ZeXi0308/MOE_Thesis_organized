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
    return dict(gpu=query("--query-gpu=uuid,name,memory.total,memory.used,temperature.gpu,power.draw,clocks.current.sm,clocks.current.memory,pstate",
                          "--format=csv,noheader"),
                compute_pids=query("--query-compute-apps=pid", "--format=csv,noheader"))


def cache_metadata(runtime):
    return {e["layer_name"]: dict(slot_to_expert=list(e["state"].slot_to_expert),
        expert_to_slot=dict(e["state"].expert_to_slot), lru_tick=list(e["state"].lru_tick),
        lru_clock=e["state"].lru_clock, expert_map_device=e["state"].expert_map_device.cpu().tolist())
        for e in runtime.layers.values()}


def allocation_metadata(runtime, runner):
    tensors = [getattr(e["state"], name) for e in runtime.layers.values()
               for name in ("cpu_w13", "cpu_w2", "scratch_w13", "scratch_w2", "expert_map_device")]
    return [dict(device=str(t.device), pointer=t.data_ptr(), storage_pointer=t.untyped_storage().data_ptr(),
                 storage_bytes=t.untyped_storage().nbytes(), shape=list(t.shape)) for t in tensors + runner.kv_caches]


def reset_drained_pager(engine, runner, runtime):
    scheduler, torch = engine.engine_core.engine_core.scheduler, runtime.torch
    assert not engine.has_unfinished_requests() and not scheduler.requests and not scheduler.running and not scheduler.waiting
    torch.cuda.synchronize()
    pool = scheduler.kv_cache_manager.block_pool
    blocks = [b for b in pool.blocks if not b.is_null]
    free = pool.free_block_queue.get_all_free_blocks()
    assert (pool.get_num_free_blocks() == len(blocks) == len(free)
            and {b.block_id for b in blocks} == {b.block_id for b in free}
            and all(b.ref_cnt == 0 and b.block_hash is None for b in blocks))
    before = allocation_metadata(runtime, runner)
    result = dict(cache_before=cache_metadata(runtime), allocations=before,
        free_kv_block_ids_in_order=[b.block_id for b in free],
        kv_blocks=[dict(block_id=b.block_id, ref_count=b.ref_cnt, is_null=b.is_null,
                        block_hash=None if b.block_hash is None else str(b.block_hash)) for b in pool.blocks],
        pending_finished_request_ids=sorted(scheduler.finished_req_ids), kv_free_queue_modified=False)
    runtime.measurement = False
    scheduler.scheduler_config.long_prefill_token_threshold = 32
    for entry in runtime.layers.values():
        state = entry["state"]
        state.slot_to_expert[:] = [-1] * state.cap_experts
        state.expert_to_slot.clear()
        state.expert_map_device.fill_(-1)
        state.lru_tick[:] = [0] * state.cap_experts
        state.lru_clock, state.last_topk_ids_cpu = 0, None
        state.last_unique_experts.clear()
    torch.cuda.synchronize()
    result["cache_after"] = cache_metadata(runtime)
    assert allocation_metadata(runtime, runner) == before
    assert all(all(v == -1 for v in c["expert_map_device"]) for c in result["cache_after"].values())
    result["allocations_unchanged"] = True
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--prepared", type=Path, required=True)
    p.add_argument("--model", required=True, help="Existing pinned local snapshot only")
    p.add_argument("--expert-cap", type=int, choices=[24, 64], required=True)
    p.add_argument("--execution", choices=["token", "expert"], default="token")
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
    p.add_argument("--warmup-executions", action="store_true",
                   help="Common token-short, expert-full, token-full warmup; requires --warmup-full")
    p.add_argument("--injection-chunk", type=int, choices=[0, 8, 128], default=None,
                   help="Expert P32/P32/P128 event probe; 0 holds the third request")
    p.add_argument("--same-engine-diagnostic", action="store_true",
                   help="Four chunk128 episodes in one engine, with common reset/warmup and CPU observations")
    p.add_argument("--same-engine-release-baseline", action="store_true",
                   help="One-engine fixed8/release8/release8/fixed8 ABBA; release after both old requests finish")
    args = p.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    dump(out / "config.json", {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    (out / "commands.txt").write_text(shlex.join([sys.executable, *sys.argv]) + "\n")
    dump(out / "status.json", dict(status="INITIALIZING", started_unix_s=time.time()))
    engine, pager = None, None
    try:
        injection = args.injection_chunk is not None
        same_engine = args.same_engine_diagnostic or args.same_engine_release_baseline
        if args.same_engine_diagnostic and args.same_engine_release_baseline:
            raise ValueError("same-engine diagnostic and release baseline are mutually exclusive")
        if args.same_engine_diagnostic and args.injection_chunk != 128:
            raise ValueError("same-engine diagnostic requires --injection-chunk 128")
        if args.same_engine_release_baseline and args.injection_chunk != 8:
            raise ValueError("same-engine release baseline requires --injection-chunk 8")
        if injection and (args.execution != "expert" or args.expert_cap != 24
                          or args.requests != 3 or args.token_budget != 160 or args.verify_kernel):
            raise ValueError("injection requires expert execution, cap24, requests3, token-budget160, no validation")
        if args.warmup_executions and not args.warmup_full:
            raise ValueError("--warmup-executions requires --warmup-full")
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
        if injection:
            lengths = [32, 32, 128]
            if any(len(ids) < n for ids, n in zip(prompts, lengths)):
                raise ValueError("injection needs prepared prefixes of 32/32/128 tokens")
            workload.update(actual_prompt_token_ids=[ids[:n] for ids, n in zip(prompts, lengths)],
                            arrival_traces_s={"steady": [0.0] * 3},
                            scope="new v0.26 prepared three-document event probe; third clock arrival overridden by event")
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
        def cuda_memory():
            return dict(allocated_bytes=torch.cuda.memory_allocated(),
                        reserved_bytes=torch.cuda.memory_reserved(),
                        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                        peak_reserved_bytes=torch.cuda.max_memory_reserved())
        def cpu_environment():
            paths = ("/sys/fs/cgroup/cpu.stat", "/sys/fs/cgroup/cpu.max", "/proc/sys/kernel/sched_schedstats")
            return dict(unix_s=time.time(), affinity=sorted(os.sched_getaffinity(0)),
                torch_threads=torch.get_num_threads(), torch_interop_threads=torch.get_num_interop_threads(),
                proc={p: Path(p).read_text().strip() if Path(p).exists() else None for p in paths})
        root = Path(__file__).parent
        dump(out / "environment.json", dict(python=sys.version, torch=torch.__version__,
            cuda=torch.version.cuda, vllm=vllm.__version__, gpu_before=before,
            cgroup_memory_max=Path("/sys/fs/cgroup/memory.max").read_text().strip(),
            sources={n: hashlib.sha256((root / n).read_bytes()).hexdigest() for n in
                     ["run_native_pager.py", "wisp_v026_adapter.py", "native_pager_context.py",
                      "native_capture.py", "admission_feedback.py"]
                     + (["wisp_expert_groups.py"] if args.execution == "expert"
                        or args.warmup_executions else [])}))
        runtime = pager.install(args.expert_cap, out / "pager")
        execution_apply = {"token": runtime.apply}
        if args.execution == "expert" or args.warmup_executions:
            from wisp_expert_groups import install_expert_groups
            execution_apply["expert"] = install_expert_groups(runtime)
        runtime.apply = execution_apply["token" if args.warmup_executions and not injection else args.execution]
        if args.verify_kernel:
            pager.enable_validation()
        kwargs = dict(model=args.model, tokenizer=args.model, dtype="bfloat16", seed=20260912,
            max_model_len=max(512, args.prompt_tokens + args.output_tokens),
            max_num_seqs=16, max_num_batched_tokens=args.token_budget,
            long_prefill_token_threshold=32 if injection else (0 if args.warmup_full else args.prefill_limit),
            kv_cache_memory_bytes=args.kv_bytes, gpu_memory_utilization=0.9,
            enable_chunked_prefill=True, enable_prefix_caching=False,
            scheduling_policy="fcfs", async_scheduling=False, stream_interval=1,
            enforce_eager=True, enable_return_routed_experts=False,
            kernel_config={"moe_backend": "triton"})
        dump(out / "engine_args.json", kwargs)
        engine = LLMEngine.from_engine_args(EngineArgs(**kwargs), enable_multiprocessing=False)
        memory = dict(after_engine_initialization=cuda_memory())
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
        if injection:
            source_ids = [s["request_id"] for s in workload["source_requests"]]
            config.update(output_tokens_by_request=dict(zip(source_ids, [16, 16, 8])), allow_preemption=False)

        def injection_episode(chunk, run_id, release_after_old=False):
            assert not engine.has_unfinished_requests() and not scheduler.requests
            scheduler.scheduler_config.long_prefill_token_threshold = 32
            def before_add(scheduler, action, rows, now):
                action["snapshot_start_s"] = now()
                storages = {(str(t.device), t.untyped_storage().data_ptr()): t.untyped_storage().nbytes()
                            for t in runner.kv_caches}
                action["before"] = dict(
                    cache={e["layer_name"]: dict(slot_to_expert=list(e["state"].slot_to_expert),
                        expert_to_slot=dict(e["state"].expert_to_slot), lru_tick=list(e["state"].lru_tick),
                        lru_clock=e["state"].lru_clock, cache_epoch="UNAVAILABLE") for e in runtime.layers.values()},
                    old_output_tokens={rid: list(rows[rid]["output_token_ids"]) for rid in source_ids[:2]},
                    running=[r.request_id for r in scheduler.running], waiting=[r.request_id for r in scheduler.waiting],
                    requests={rid: dict(computed=r.num_computed_tokens, prompt=r.num_prompt_tokens,
                        status=r.status.name, num_preemptions=r.num_preemptions,
                        kv_block_ids=tuple(tuple(g) for g in scheduler.kv_cache_manager.get_block_ids(rid)))
                        for rid, r in scheduler.requests.items()},
                    total_kv_blocks=pool.num_gpu_blocks, free_kv_blocks=pool.get_num_free_blocks(),
                    actual_unique_kv_storage_bytes=sum(storages.values()))
                action["snapshot_end_s"] = now()
                action["threshold_before"] = scheduler.scheduler_config.long_prefill_token_threshold
                assert action["threshold_before"] == 32
                scheduler.scheduler_config.long_prefill_token_threshold = chunk or 32
                action.update(threshold_after=scheduler.scheduler_config.long_prefill_token_threshold,
                              threshold_applied_s=now())
            return capture_episode(engine, workload, config, regime="steady", arrival_scale=1,
                run_id=run_id, max_seconds=args.max_seconds, before_event_add=before_add,
                cpu_diagnostics=same_engine, release_prefill_after_old_complete=release_after_old,
                event_arrival=dict(request_id=source_ids[2], after_output_tokens={rid: 4 for rid in source_ids[:2]},
                                   inject=bool(chunk), expected_first_tokens=2 + chunk))
        episodes, reference = [], None
        count = 4 if same_engine else 1
        for repeat in range(count):
            variant = ("fixed8", "release8", "release8", "fixed8")[repeat] if args.same_engine_release_baseline else None
            episode_name = f"repeat_{repeat}" + ("_" + variant if variant else "")
            prefix = episode_name + "/" if same_engine else ""
            episode_out = out / episode_name if same_engine else out
            if same_engine:
                episode_out.mkdir(exist_ok=False)
                reset = reset_drained_pager(engine, runner, runtime)
                reset.update(environment=cpu_environment(), gpu=gpu_state())
                dump(episode_out / "reset.json", reset)
            runtime.measurement = False
            phase, step = prefix + "warmup", 0
            warmup = dict(workload, source_requests=workload["source_requests"][:1],
                actual_prompt_token_ids=workload["actual_prompt_token_ids"][:1],
                arrival_traces_s={"steady": [0.0]})
            raw = capture_episode(engine, warmup, dict(config, output_tokens=2, output_tokens_by_request={}),
                regime="steady", arrival_scale=1, run_id=phase, max_seconds=args.max_seconds,
                cpu_diagnostics=same_engine)
            dump(episode_out / "warmup.json", raw)
            if raw["status"] != "COMPLETE":
                raise RuntimeError("warmup incomplete: " + str(raw["error"]))
            if injection:
                warmups = ([(8, False, "fixed8"), (8, True, "release8")] if args.same_engine_release_baseline
                           else [(8, False, "8"), (128, False, "128")])
                for chunk, release_after_old, label in warmups:
                    name = "warmup_injection_" + label
                    phase, step = prefix + name, 0
                    raw = injection_episode(chunk, phase, release_after_old)
                    dump(episode_out / (name + ".json"), raw)
                    if raw["status"] != "COMPLETE":
                        raise RuntimeError("injection warmup incomplete: " + str(raw["error"]))
            elif args.warmup_full:
                for execution in (["expert", "token"] if args.warmup_executions else [args.execution]):
                    runtime.apply = execution_apply[execution]
                    if args.warmup_executions:
                        phase = "warmup_full_" + execution
                    raw = capture_episode(engine, workload, config, regime="steady", arrival_scale=1,
                        run_id=phase if args.warmup_executions else "warmup-full", max_seconds=args.max_seconds)
                    name = phase + ".json" if args.warmup_executions else "warmup_full.json"
                    dump(episode_out / name, raw)
                    if raw["status"] != "COMPLETE":
                        raise RuntimeError("full warmup incomplete: " + str(raw["error"]))
            assert not engine.has_unfinished_requests() and not scheduler.requests
            prior_limit = scheduler.scheduler_config.long_prefill_token_threshold
            scheduler.scheduler_config.long_prefill_token_threshold = 32 if injection else args.prefill_limit
            dump(episode_out / "policy_application.json", dict(prior_prefill_limit=prior_limit,
                measured_prefill_limit=scheduler.scheduler_config.long_prefill_token_threshold, applied_on_drained_engine=True,
                common_full_warmup_prefill_limit=32 if injection else (0 if args.warmup_full else None),
                injection_warmup_sequence=["expert:" + label for _, _, label in warmups] if injection else None,
                old_completion_prefill_release=variant == "release8"))
            phase, step = prefix + "measurement", 0
            runtime.apply = execution_apply[args.execution]
            pager.begin_measurement(phase)
            kv_storages = {(str(t.device), t.untyped_storage().data_ptr()): t.untyped_storage().nbytes()
                           for t in runner.kv_caches}
            resources = dict(actual_unique_kv_storage_bytes=sum(kv_storages.values()),
                kv_storage_count=len(kv_storages), kv_storage_devices=sorted({device for device, _ in kv_storages}),
                pool_num_gpu_blocks=pool.num_gpu_blocks, free_blocks=pool.get_num_free_blocks(),
                scheduler_requests=len(scheduler.requests), expert_cap=args.expert_cap,
                measured_execution=args.execution, cpu_affinity=sorted(os.sched_getaffinity(0)))
            if same_engine:
                resources.update(cache=cache_metadata(runtime), allocations=allocation_metadata(runtime, runner),
                                 environment_before=cpu_environment(), phase=phase)
                dump(episode_out / "measurement_initial_cache.json", resources["cache"])
            dump(episode_out / "measurement_resources.json", resources)
            memory = dict(after_engine_initialization=memory["after_engine_initialization"], before_measurement=cuda_memory())
            torch.cuda.reset_peak_memory_stats()
            memory["measurement_peak_scope"] = "since drained warmup; includes validation reference if enabled"
            dump(episode_out / "cuda_memory.json", memory)
            dump(out / "status.json", dict(status="MEASURING", phase=phase, started_unix_s=time.time()))
            raw = (injection_episode(args.injection_chunk, phase, variant == "release8") if injection else
                   capture_episode(engine, workload, config, regime="steady", arrival_scale=1,
                                   run_id="measurement", max_seconds=args.max_seconds))
            memory["after_measurement"] = cuda_memory()
            dump(episode_out / "cuda_memory.json", memory)
            dump(episode_out / "raw.json", raw)
            episodes.append(dict(phase=phase, status=raw["status"], raw_path=str(episode_out / "raw.json"),
                                 observation_end_s=raw["observation_end_s"]))
            if variant:
                episodes[-1]["variant"] = variant
            if same_engine:
                dump(episode_out / "environment_after.json", dict(cpu=cpu_environment(), gpu=gpu_state()))
                if raw["event_actions"] and "before" in raw["event_actions"][0]:
                    before = raw["event_actions"][0]["before"]
                    logical = {k: v for k, v in before.items() if k not in ("requests", "running", "waiting")}
                    logical["requests"] = {raw["internal_to_source"][rid]: {k: v for k, v in r.items() if k != "kv_block_ids"}
                                           for rid, r in before["requests"].items()}
                    logical.update({k: [raw["internal_to_source"][rid] for rid in before[k]]
                                    for k in ("running", "waiting")})
                    block_ids = {raw["internal_to_source"][rid]: r["kv_block_ids"] for rid, r in before["requests"].items()}
                    reference = reference or dict(logical=logical, blocks=block_ids, resources=resources)
                    episodes[-1].update(logical_preaction_metadata_equal=logical == reference["logical"],
                        physical_kv_block_ids_equal=block_ids == reference["blocks"], kv_tensor_contents_compared=False,
                        initial_cache_equal=resources["cache"] == reference["resources"]["cache"],
                        allocations_equal=resources["allocations"] == reference["resources"]["allocations"])
                dump(out / "episodes.json", episodes)
            if raw["status"] != "COMPLETE":
                raise RuntimeError("measurement incomplete: " + str(raw["error"]))
            assert len(raw["requests"]) == args.requests
            assert all(len(r["output_token_ids"]) == (0 if r["status"] == "not_injected" else r["max_output_tokens"])
                       for r in raw["requests"])
        summary = pager.finalize()
        dump(out / "pager_summary.json", summary)
        dump(out / "status.json", dict(status="COMPLETE", finished_unix_s=time.time(),
            evidence_type=("MATCHED_LAYER_KERNEL_VALIDATION_ONLY" if args.verify_kernel
                           else "NATIVE_EAGER_PAGER_INTEGRATION_ONLY"), gpu_after=gpu_state(),
            requests_completed=sum(r["status"] == "completed" for r in raw["requests"]),
            generated_tokens=sum(len(r["output_token_ids"]) for r in raw["requests"]),
            episodes_completed=len(episodes), request_count_scope="last episode; each episode retained separately",
            hold_scope="two-request observation control; not same-task throughput" if args.injection_chunk == 0 else None,
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
