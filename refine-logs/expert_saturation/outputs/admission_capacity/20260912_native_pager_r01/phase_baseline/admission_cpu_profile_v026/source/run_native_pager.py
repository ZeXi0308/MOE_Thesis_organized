"""Bounded OLMoE/vLLM 0.26 integration probe using the upstream WiSP pager.

This is a runtime/cost probe, not a scheduler result. Both full-cap and paged
arms use the same BF16 TRITON wrapper and explicit KV bytes. Original native
request capture is reused; scheduler counts do not establish token row order.
"""
import argparse
import csv
from datetime import datetime, timezone
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
        command = ["nvidia-smi", *args]
        try:
            r = subprocess.run(command, text=True, capture_output=True, timeout=10)
            return dict(command=command, returncode=r.returncode, stdout=r.stdout.strip(), stderr=r.stderr.strip())
        except (OSError, subprocess.SubprocessError) as exc:
            return dict(command=command, returncode=None, stdout="", stderr=f"{type(exc).__name__}: {exc}")
    queries = dict(gpu=query("--query-gpu=uuid,name,memory.total,memory.used,temperature.gpu,power.draw,clocks.current.sm,clocks.current.memory,pstate",
                             "--format=csv,noheader"),
                   processes=query("--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory", "--format=csv,noheader"))
    processes, errors = [], []
    for name, q in queries.items():
        if q["returncode"] != 0:
            errors.append(name + " query failed")
    if not queries["gpu"]["stdout"]:
        errors.append("empty GPU state")
    for row in csv.reader(queries["processes"]["stdout"].splitlines(), skipinitialspace=True):
        if len(row) != 4 or not row[1].strip().isdigit() or int(row[1]) <= 0:
            errors.append("invalid compute process row: " + repr(row))
        else:
            processes.append(dict(gpu_uuid=row[0].strip(), pid=int(row[1]), process_name=row[2].strip(), used_memory=row[3].strip()))
    return dict(utc=datetime.now(timezone.utc).isoformat(), unix_s=time.time(), caller_pid=os.getpid(),
                gpu=queries["gpu"]["stdout"], compute_pids="\n".join(str(p["pid"]) for p in processes),
                processes=processes, queries=queries, query_errors=errors)


def check_gpu(out, stage, *, allow_self=False):
    snapshot = gpu_state()
    allowed = {os.getpid()} if allow_self else set()
    foreign = sorted({p["pid"] for p in snapshot["processes"]} - allowed)
    decision = "ABORT_QUERY" if snapshot["query_errors"] else "ABORT_BUSY" if foreign else "PASS"
    snapshot.update(stage=stage, allowed_pids=sorted(allowed), foreign_pids=foreign, decision=decision)
    with (out / "gpu_checks.jsonl").open("a") as log:
        log.write(json.dumps(snapshot, allow_nan=False) + "\n")
    if decision != "PASS":
        raise RuntimeError(f"{decision} at {stage}; see retained gpu_checks.jsonl")
    return snapshot


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
    p.add_argument("--expert-cap", type=int, choices=[16, 24, 64], required=True)
    p.add_argument("--execution", choices=["token", "expert"], default="token")
    p.add_argument("--group-retention", choices=["none", "frequency", "decode", "matched_hash"], default="none")
    p.add_argument("--group-retention-order", choices=["early", "late"], default="early")
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
    p.add_argument("--injection-chunk", type=int, choices=[0, 8, 16, 128], default=None,
                   help="Expert P32/P32/P128 event probe; 0 holds the third request")
    p.add_argument("--same-engine-diagnostic", action="store_true",
                   help="Four chunk128 episodes in one engine, with common reset/warmup and CPU observations")
    p.add_argument("--same-engine-release-baseline", action="store_true",
                   help="One-engine fixed8/release8/release8/fixed8 ABBA; release after both old requests finish")
    p.add_argument("--same-engine-retention-baseline", action="store_true",
                   help="Eight release8 episodes: none/frequency/decode/matched_hash and reverse; common reset/warmup")
    p.add_argument("--same-engine-retention-order-comparison", action="store_true",
                   help="Compare none, early frequency, late frequency, late decode and reverse in one engine")
    p.add_argument("--same-engine-runtime-diagnostic", action="store_true",
                   help="Eight identical none/early release8 episodes with passive GC/CPU observations")
    p.add_argument("--same-engine-retention-lifecycle-comparison", action="store_true",
                   help="Eight none/early versus frequency/late episodes, with bounded trace retention and complete cycle timing")
    p.add_argument("--retention-lifecycle-reverse", action="store_true",
                   help="Swap A/B in the predeclared ABBA BAAB lifecycle comparison")
    p.add_argument("--trace-retention", choices=["memory", "episode"], default="memory",
                   help="Observed runtime matrices: retain trace until finalization or write/release at drained episode boundaries")
    p.add_argument("--retention-order-offset", type=int, choices=[0, 2], default=0,
                   help="Predeclared rotation of the four order-comparison arms; only with that matrix")
    p.add_argument("--same-engine-admission-cpu-profile", action="store_true", required=True,
                   help="Six identical admit2_32 episodes, middle four with measurement-only current-thread CPU profiling")
    args = p.parse_args()
    if args.same_engine_admission_cpu_profile:
        args.admission_width_order = ('admit2_32',) * 6
        args.cpu_profile_tags = ('plain_before', 'cpu_profile_1', 'cpu_profile_2', 'cpu_profile_3', 'cpu_profile_4', 'plain_after')
        args.admission_width_ordering = "identical_D_plain_profile4_plain"
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    dump(out / "config.json", {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    (out / "commands.txt").write_text(shlex.join([sys.executable, *sys.argv]) + "\n")
    dump(out / "status.json", dict(status="INITIALIZING", started_unix_s=time.time()))
    engine, pager, observer, cycle = None, None, None, None
    try:
        injection = args.injection_chunk is not None
        matrices = (args.same_engine_admission_cpu_profile, args.same_engine_diagnostic, args.same_engine_release_baseline,
                    args.same_engine_retention_baseline, args.same_engine_retention_order_comparison,
                    args.same_engine_runtime_diagnostic, args.same_engine_retention_lifecycle_comparison)
        same_engine = any(matrices)
        observed_run = args.same_engine_admission_cpu_profile or args.same_engine_runtime_diagnostic or args.same_engine_retention_lifecycle_comparison
        retention_run = (args.same_engine_retention_baseline or args.same_engine_retention_order_comparison
                         or args.same_engine_retention_lifecycle_comparison or args.group_retention != "none")
        if sum(matrices) > 1 or (same_engine and (args.group_retention != "none" or args.group_retention_order != "early")):
            raise ValueError("same-engine matrices are mutually exclusive and select their own retention modes")
        if args.retention_order_offset and not args.same_engine_retention_order_comparison:
            raise ValueError("retention order offset requires the order comparison")
        if args.retention_lifecycle_reverse and not args.same_engine_retention_lifecycle_comparison:
            raise ValueError("lifecycle reverse requires the retention lifecycle comparison")
        if args.same_engine_retention_lifecycle_comparison and args.trace_retention != "episode":
            raise ValueError("retention lifecycle comparison requires episode trace retention")
        if args.group_retention_order == "late" and args.group_retention == "none":
            raise ValueError("standalone late order requires a protected selection")
        if args.same_engine_diagnostic and args.injection_chunk != 128:
            raise ValueError("same-engine diagnostic requires --injection-chunk 128")
        if args.same_engine_release_baseline and args.injection_chunk != 8:
            raise ValueError("same-engine release baseline requires --injection-chunk 8")
        if args.same_engine_runtime_diagnostic and args.injection_chunk != 8:
            raise ValueError("same-engine runtime diagnostic requires --injection-chunk 8")
        if args.trace_retention != "memory" and not observed_run:
            raise ValueError("episode trace retention requires an observed runtime matrix")
        if retention_run and args.injection_chunk != 8:
            raise ValueError("retention probe requires --injection-chunk 8")
        if args.same_engine_admission_cpu_profile and (
                args.expert_cap != 16 or args.requests != 3 or args.token_budget != 64
                or args.kv_bytes != 536870912 or args.injection_chunk != 16
                or args.group_retention != "none" or args.group_retention_order != "early"):
            raise ValueError("expert curve requires cap16/requests3/tokens64/KV512MiB/injection-placeholder16/none/early")
        if injection and not args.same_engine_admission_cpu_profile and (args.execution != "expert" or args.expert_cap != 24
                          or args.requests != 3 or args.token_budget != 160):
            raise ValueError("injection requires expert execution, cap24, requests3, token-budget160")
        if injection and args.verify_kernel and (not retention_run or same_engine):
            raise ValueError("injection validation is limited to a single non-none retention qualification")
        if args.warmup_executions and not args.warmup_full:
            raise ValueError("--warmup-executions requires --warmup-full")
        before = check_gpu(out, "before_initialization")
        if not Path(args.model).is_dir():
            raise ValueError("local model snapshot missing")
        frozen = json.loads((args.prepared / "workload.json").read_text())
        if not 1 <= args.requests <= 16 or not 2 <= args.output_tokens <= 128:
            raise ValueError("bounded pilot requires 1..16 requests and 2..128 output tokens")
        if args.same_engine_admission_cpu_profile:
            indices = [4, 5, 10]
            frozen = dict(frozen,
                source_requests=[frozen["source_requests"][i] for i in indices],
                actual_prompt_token_ids=[frozen["actual_prompt_token_ids"][i] for i in indices])
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
        if args.same_engine_admission_cpu_profile:
            workload.update(source_indices=[4, 5, 10],
                scope="same-source c10 native admission baseline; all three requests enter at the unchanged event time")
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
        from admission_width import apply_admission
        from cpu_capture_profile import profile_capture
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
                     + (["wisp_expert_groups.py"] if args.same_engine_admission_cpu_profile or args.execution == "expert"
                        or args.warmup_executions else [])
                     + (["runtime_variation_observer.py"] if observed_run else [])}))
        import batched_expert_map as common_map
        common_map.set_mode("batched")
        dump(out / "map_selftest.json", common_map.gpu_selftest())
        runtime = pager.install(args.expert_cap, out / "pager")
        execution_apply = {"token": runtime.apply}
        if args.same_engine_admission_cpu_profile or args.execution == "expert" or args.warmup_executions:
            from wisp_expert_groups import install_expert_groups
            execution_apply["expert"] = install_expert_groups(runtime)
        runtime.group_retention = args.group_retention
        runtime.group_retention_order = args.group_retention_order
        runtime.apply = execution_apply["token" if args.same_engine_admission_cpu_profile or (args.warmup_executions and not injection) else args.execution]
        if args.verify_kernel:
            pager.enable_validation()
        kwargs = dict(model=args.model, tokenizer=args.model, dtype="bfloat16", seed=20260912,
            max_model_len=max(512, args.prompt_tokens + args.output_tokens),
            max_num_seqs=3 if args.same_engine_admission_cpu_profile else 16, max_num_batched_tokens=args.token_budget,
            long_prefill_token_threshold=32 if injection else (0 if args.warmup_full else args.prefill_limit),
            kv_cache_memory_bytes=args.kv_bytes, gpu_memory_utilization=0.9,
            enable_chunked_prefill=True, enable_prefix_caching=False,
            scheduling_policy="fcfs", async_scheduling=False, stream_interval=1,
            enforce_eager=True, enable_return_routed_experts=False,
            kernel_config={"moe_backend": "triton"})
        dump(out / "engine_args.json", kwargs)
        check_gpu(out, "before_model_load", allow_self=True)
        engine = LLMEngine.from_engine_args(EngineArgs(**kwargs), enable_multiprocessing=False)
        if observed_run:
            from runtime_variation_observer import RuntimeVariationObserver
            observer = RuntimeVariationObserver()
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

        comparison_prestate = None
        def injection_episode(chunk, run_id, release_after_old=False):
            admission_mode = cell_arm if run_id.endswith("/measurement") else next(
                (m for m in ("immediate32", "admit2_32", "admit2_release64")
                 if run_id.endswith("/warmup_injection_" + m)), "immediate32")
            episode_raw = None
            assert scheduler.max_num_running_reqs == 3
            if args.same_engine_admission_cpu_profile:
                runtime.apply = execution_apply["token"]
            assert not engine.has_unfinished_requests() and not scheduler.requests
            scheduler.scheduler_config.long_prefill_token_threshold = 32
            def before_add(scheduler, action, rows, now):
                nonlocal comparison_prestate
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
                if args.same_engine_admission_cpu_profile and run_id.endswith("/measurement"):
                    assert runtime.apply == execution_apply["token"]
                    ids = {r["internal_request_id"]: rid for rid, r in rows.items()
                           if "internal_request_id" in r}
                    logical = dict(action["before"])
                    logical["requests"] = {ids[rid]: dict(
                        {k: v for k, v in r.items() if k != "kv_block_ids"},
                        kv_block_counts=[len(g) for g in r["kv_block_ids"]])
                        for rid, r in logical["requests"].items()}
                    for key in ("running", "waiting"):
                        logical[key] = [ids[rid] for rid in logical[key]]
                    digest = hashlib.sha256(json.dumps(logical, sort_keys=True,
                        separators=(",", ":")).encode()).hexdigest()
                    action.update(prestate_sha256=digest,
                        prestate_scope="Source-ID-normalized tokens/request/KV counts/pager metadata; physical KV IDs retained separately; no KV tensor comparison")
                    if comparison_prestate is not None and digest != comparison_prestate:
                        raise RuntimeError("common token pre-injection metadata differs")
                    comparison_prestate = digest
                    runtime.apply = execution_apply[cell_execution]
                    action.update(execution_before="token", execution_after=cell_execution,
                        execution_applied_s=now(), execution_boundary="before event add and next native schedule/KV allocation")
                elif args.same_engine_admission_cpu_profile:
                    assert runtime.apply == execution_apply["token"]
                    execution = "expert" if run_id.endswith(("/warmup_injection_immediate32", "/warmup_injection_admit2_32", "/warmup_injection_admit2_release64")) else "token"
                    runtime.apply = execution_apply[execution]
                    action.update(execution_before="token", execution_after=execution,
                        execution_applied_s=now(), execution_boundary="before event add and next native schedule/KV allocation")
                apply_admission(scheduler, action, admission_mode, now)
                action["threshold_before"] = scheduler.scheduler_config.long_prefill_token_threshold
                assert action["threshold_before"] == 32
                scheduler.scheduler_config.long_prefill_token_threshold = chunk or 32
                action.update(threshold_after=scheduler.scheduler_config.long_prefill_token_threshold,
                              threshold_applied_s=now())
            try:
                episode_raw = capture_episode(engine, workload, dict(config, admission_width_mode=admission_mode), regime="steady", arrival_scale=1,
                    run_id=run_id, max_seconds=args.max_seconds, before_event_add=before_add,
                    cpu_diagnostics=same_engine, runtime_observer=observer,
                    release_prefill_after_old_complete=release_after_old,
                    event_arrival=dict(request_id=source_ids[2], after_output_tokens={rid: 4 for rid in source_ids[:2]},
                                       inject=True, expected_first_tokens=2 + chunk if admission_mode == "immediate32" else 2))
                return episode_raw
            finally:
                restoration = dict(drained=not engine.has_unfinished_requests(),
                    cap_before=scheduler.max_num_running_reqs,
                    threshold_before=scheduler.scheduler_config.long_prefill_token_threshold)
                scheduler.max_num_running_reqs = 3
                scheduler.scheduler_config.long_prefill_token_threshold = 32
                restoration.update(cap_after=3, threshold_after=32)
                if episode_raw is not None:
                    episode_raw["admission_restore"] = restoration
                if args.same_engine_admission_cpu_profile:
                    runtime.apply = execution_apply["token"]
        episodes, reference = [], None
        retention_modes = ("none", "frequency", "decode", "matched_hash", "matched_hash", "decode", "frequency", "none")
        order_arms = [("none", "early"), ("frequency", "early"), ("frequency", "late"), ("decode", "late")]
        order_arms = order_arms[args.retention_order_offset:] + order_arms[:args.retention_order_offset]
        order_sequence = order_arms + list(reversed(order_arms))
        lifecycle_arms = [("none", "early"), ("frequency", "late")]
        if args.retention_lifecycle_reverse:
            lifecycle_arms.reverse()
        lifecycle_sequence = [lifecycle_arms[i] for i in (0, 1, 1, 0, 1, 0, 0, 1)]
        count = 8 if args.same_engine_retention_order_comparison or observed_run else len(retention_modes) if args.same_engine_retention_baseline else 4 if same_engine else 1
        if args.same_engine_admission_cpu_profile:
            count = 6
        if observed_run:
            cycle = dict(start_perf_ns=time.perf_counter_ns(), start_unix_s=time.time(),
                trace_retention=args.trace_retention, repeats=[],
                scope="After engine setup through all resets/warmups/measurements/IO/flushes/finalize/observer export/shutdown; excludes initial setup and writing this final marker. Parent process wall includes those too.")
        for repeat in range(count):
            repeat_start_ns = time.perf_counter_ns() if cycle is not None else None
            variant = ("fixed8", "release8", "release8", "fixed8")[repeat] if args.same_engine_release_baseline else None
            retention_mode = retention_modes[repeat] if args.same_engine_retention_baseline else args.group_retention
            retention_order = args.group_retention_order
            if args.same_engine_retention_baseline:
                variant = "retention_" + retention_mode
            if args.same_engine_retention_order_comparison:
                retention_mode, retention_order = order_sequence[repeat]
                variant = f"retention_{retention_mode}_{retention_order}"
            if args.same_engine_runtime_diagnostic:
                variant = "retention_none_early"
            if args.same_engine_retention_lifecycle_comparison:
                retention_mode, retention_order = lifecycle_sequence[repeat]
                variant = f"retention_{retention_mode}_{retention_order}"
            cell_execution = args.execution
            cell_chunk = args.injection_chunk
            cell_arm = "immediate32"
            if args.same_engine_admission_cpu_profile:
                cell_execution = "expert"
                cell_arm = args.admission_width_order[repeat]
                cell_chunk = 32
                variant = args.cpu_profile_tags[repeat]
                runtime.apply = execution_apply["token"]
            release_after_old = cell_arm == "admit2_release64" if args.same_engine_admission_cpu_profile else retention_run or variant == "release8" or args.same_engine_runtime_diagnostic
            episode_name = f"repeat_{repeat}" + ("_" + variant if variant else "")
            prefix = episode_name + "/" if same_engine else ""
            episode_out = out / episode_name if same_engine else out
            if same_engine:
                episode_out.mkdir(exist_ok=False)
                boundary_gpu = check_gpu(out, prefix + "before_reset", allow_self=True)
                reset = reset_drained_pager(engine, runner, runtime)
                reset.update(environment=cpu_environment(), gpu=boundary_gpu)
                dump(episode_out / "reset.json", reset)
            runtime.measurement = False
            runtime.group_retention = "none"
            runtime.group_retention_order = "early"
            phase, step = prefix + "warmup", 0
            warmup = dict(workload, source_requests=workload["source_requests"][:1],
                actual_prompt_token_ids=workload["actual_prompt_token_ids"][:1],
                arrival_traces_s={"steady": [0.0]})
            raw = capture_episode(engine, warmup, dict(config, output_tokens=2, output_tokens_by_request={}),
                regime="steady", arrival_scale=1, run_id=phase, max_seconds=args.max_seconds,
                cpu_diagnostics=same_engine, runtime_observer=observer)
            dump(episode_out / "warmup.json", raw)
            if raw["status"] != "COMPLETE":
                raise RuntimeError("warmup incomplete: " + str(raw["error"]))
            if injection:
                warmups = ([(8, True, m + "_release8", m) for m in ("none", "frequency", "decode", "matched_hash")]
                           if retention_run else [(8, False, "fixed8", "none"), (8, True, "release8", "none")]
                           if args.same_engine_release_baseline else [(8, False, "8", "none"), (128, False, "128", "none")])
                warmup_orders = {}
                if (args.same_engine_retention_order_comparison or (observed_run and not args.same_engine_admission_cpu_profile)
                        or args.group_retention_order == "late"):
                    warmups = [(8, True, f"{m}_{o}_release8", m) for m, o in
                               [("none", "early"), ("frequency", "early"), ("frequency", "late"), ("decode", "late")]]
                    warmup_orders = {f"{m}_{o}_release8": o for m, o in
                                     [("none", "early"), ("frequency", "early"), ("frequency", "late"), ("decode", "late")]}
                if args.same_engine_admission_cpu_profile:
                    warmups = [(chunk, label == "admit2_release64", label, "none") for chunk, label in
                               [(32, "token32_before"), (32, "immediate32"), (32, "admit2_32"),
                                (32, "admit2_release64"), (32, "token32_after")]]
                for chunk, warmup_release, label, mode in warmups:
                    name = "warmup_injection_" + label
                    phase, step = prefix + name, 0
                    runtime.group_retention = mode
                    runtime.group_retention_order = warmup_orders.get(label, "early")
                    raw = injection_episode(chunk, phase, warmup_release)
                    dump(episode_out / (name + ".json"), raw)
                    if raw["status"] != "COMPLETE":
                        raise RuntimeError("injection warmup incomplete: " + str(raw["error"]))
            elif args.warmup_full:
                for execution in (["expert", "token"] if args.warmup_executions else [args.execution]):
                    runtime.apply = execution_apply[execution]
                    if args.warmup_executions:
                        phase = "warmup_full_" + execution
                    raw = capture_episode(engine, workload, config, regime="steady", arrival_scale=1,
                        run_id=phase if args.warmup_executions else "warmup-full", max_seconds=args.max_seconds,
                        runtime_observer=observer)
                    name = phase + ".json" if args.warmup_executions else "warmup_full.json"
                    dump(episode_out / name, raw)
                    if raw["status"] != "COMPLETE":
                        raise RuntimeError("full warmup incomplete: " + str(raw["error"]))
            assert not engine.has_unfinished_requests() and not scheduler.requests
            prior_limit = scheduler.scheduler_config.long_prefill_token_threshold
            scheduler.scheduler_config.long_prefill_token_threshold = 32 if injection else args.prefill_limit
            runtime.group_retention = retention_mode
            runtime.group_retention_order = retention_order
            dump(episode_out / "policy_application.json", dict(prior_prefill_limit=prior_limit,
                measured_prefill_limit=scheduler.scheduler_config.long_prefill_token_threshold, applied_on_drained_engine=True,
                common_full_warmup_prefill_limit=32 if injection else (0 if args.warmup_full else None),
                injection_warmup_sequence=[(("expert:" if label in ("immediate32", "admit2_32", "admit2_release64") else "token:") if args.same_engine_admission_cpu_profile else "expert:") + label for _, _, label, _ in warmups] if injection else None,
                group_retention=retention_mode, group_retention_order=retention_order,
                old_completion_prefill_release=release_after_old))
            phase, step = prefix + "measurement", 0
            runtime.apply = execution_apply["token" if args.same_engine_admission_cpu_profile else args.execution]
            for entry in runtime.layers.values():
                common_map._entry(entry["state"])
            common_map.stats(reset=True)
            pager.begin_measurement(phase)
            kv_storages = {(str(t.device), t.untyped_storage().data_ptr()): t.untyped_storage().nbytes()
                           for t in runner.kv_caches}
            resources = dict(actual_unique_kv_storage_bytes=sum(kv_storages.values()),
                kv_storage_count=len(kv_storages), kv_storage_devices=sorted({device for device, _ in kv_storages}),
                pool_num_gpu_blocks=pool.num_gpu_blocks, free_blocks=pool.get_num_free_blocks(),
                scheduler_requests=len(scheduler.requests), expert_cap=args.expert_cap,
                measured_execution=cell_execution, measured_chunk=cell_chunk, arm=cell_arm, group_retention=retention_mode,
                group_retention_order=retention_order, cpu_affinity=sorted(os.sched_getaffinity(0)))
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
            if observer is not None:
                observation = dict(schema="runtime_episode_observation_v1", phase=phase, after=None,
                    before=dict(runtime_records_count=len(runtime.records),
                        flushed_records_count=runtime.flushed_calls,
                        total_records_count=runtime.flushed_calls+len(runtime.records),
                        snapshot=observer.snapshot()))
                dump(episode_out / "runtime_observation.json", observation)
            try:
                raw = profile_capture(
                    lambda: injection_episode(cell_chunk, phase, release_after_old), episode_out,
                    enabled=1 <= repeat <= 4, tag=args.cpu_profile_tags[repeat])
            finally:
                if observer is not None:
                    observation["after"] = dict(runtime_records_count=len(runtime.records),
                        flushed_records_count=runtime.flushed_calls,
                        total_records_count=runtime.flushed_calls+len(runtime.records),
                        snapshot=observer.snapshot())
                    dump(episode_out / "runtime_observation.json", observation)
            memory["after_measurement"] = cuda_memory()
            dump(episode_out / "cuda_memory.json", memory)
            dump(episode_out / "raw.json", raw)
            dump(episode_out / "map_optimization.json", dict(
                statistics=common_map.stats(), validation=common_map.validate(),
                scope="Same batched full resident map update in token and expert arms; expert kernel still receives a separate masked group map. Validation runs after capture wall."))
            episodes.append(dict(phase=phase, execution=cell_execution, status=raw["status"], raw_path=str(episode_out / "raw.json"),
                                 observation_end_s=raw["observation_end_s"]))
            if args.same_engine_admission_cpu_profile:
                episodes[-1].update(arm=cell_arm, chunk=cell_chunk, block=repeat // 3, order=repeat,
                    order_in_block=repeat % 3, ordering=args.admission_width_ordering,
                    profile_tag=args.cpu_profile_tags[repeat], cpu_profile_enabled=1 <= repeat <= 4)
            if variant:
                episodes[-1]["variant"] = variant
            if same_engine:
                dump(episode_out / "environment_after.json", dict(cpu=cpu_environment(),
                     gpu=check_gpu(out, phase + "/after_measurement", allow_self=True)))
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
            if cycle is not None:
                assert not engine.has_unfinished_requests() and not scheduler.requests
                flush = runtime.flush_records(episode_name) if args.trace_retention == "episode" else None
                cycle["repeats"].append(dict(name=episode_name, start_perf_ns=repeat_start_ns,
                    end_perf_ns=time.perf_counter_ns(), capture_wall_s=raw["observation_end_s"],
                    trace_flush=flush, held_records=len(runtime.records), held_events=len(runtime.events),
                    flushed_records=runtime.flushed_calls))
                dump(out / "cycle_progress.json", cycle)
        if args.verify_kernel and retention_run and not runtime.retention_validated:
            raise RuntimeError("retention qualification never reached an active mixed layer call")
        summary = pager.finalize()
        dump(out / "pager_summary.json", summary)
        dump(out / "status.json", dict(status="COMPLETE", finished_unix_s=time.time(),
            evidence_type=("MATCHED_LAYER_KERNEL_VALIDATION_ONLY" if args.verify_kernel
                           else "NATIVE_EAGER_PAGER_INTEGRATION_ONLY"), gpu_after=check_gpu(out, "after_run", allow_self=True),
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
        try:
            if observer is not None:
                try:
                    observer.close()
                    dump(out / "runtime_events.json", dict(schema="runtime_variation_events_v1",
                        events=observer.events, final_snapshot=observer.snapshot(),
                        runtime_records_count=runtime.flushed_calls+len(runtime.records),
                        held_records_count=len(runtime.records), flushed_records_count=runtime.flushed_calls,
                        final_snapshot_after_close=True))
                except Exception:
                    (out / "runtime_observer_error.txt").write_text(traceback.format_exc())
        finally:
            try:
                if engine is not None:
                    engine.engine_core.shutdown()
            finally:
                if cycle is not None:
                    cycle.update(end_perf_ns=time.perf_counter_ns(), end_unix_s=time.time(),
                        trace_flushes=runtime.flushes)
                    cycle["post_init_cycle_wall_s"] = (cycle["end_perf_ns"]-cycle["start_perf_ns"])/1e9
                    dump(out / "cycle_complete.json", cycle)


if __name__ == "__main__":
    main()
