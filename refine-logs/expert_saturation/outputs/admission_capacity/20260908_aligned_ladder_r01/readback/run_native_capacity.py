#!/usr/bin/env python3
"""Small native vLLM transfer probe with actual scheduler and host-output timing."""
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

from metrics import summarize_episode_requests
from native_capture import capture_episode, set_empty_admission_cap


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def gpu_state():
    query = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                                     "--format=csv,noheader"], text=True).strip()
    others = [r for r in csv.reader(query.splitlines()) if r and int(r[0]) != os.getpid()]
    if others:
        raise RuntimeError(f"other GPU compute processes: {others}")
    return dict(compute_processes=query, gpu=subprocess.check_output(["nvidia-smi",
        "--query-gpu=name,uuid,memory.total,memory.used,temperature.gpu,power.draw,clocks.sm",
        "--format=csv,noheader"], text=True).strip())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    cap_option = parser.add_mutually_exclusive_group(required=True)
    cap_option.add_argument("--cap", type=int)
    cap_option.add_argument("--caps", help="ordered static caps, changed only between drained episodes")
    parser.add_argument("--engine-max-seqs", type=int, help="keep engine/graph capacity fixed; cap controls empty-episode admission")
    parser.add_argument("--arrival-scales", default="1,0.02")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--reverse-conditions", action="store_true")
    feedback_option = parser.add_mutually_exclusive_group()
    feedback_option.add_argument("--include-feedback", action="store_true", help="append shadow and ordinary-ITL feedback arms at max cap")
    feedback_option.add_argument("--single-action-probe", action="store_true", help="hold32 / one-time down16 / static16 with unchanged warmup caps")
    parser.add_argument("--ttft-slo-s", type=float)
    parser.add_argument("--tpot-slo-s", type=float)
    parser.add_argument("--warmup-condition-rounds", type=int, default=0)
    parser.add_argument("--underload-scale", type=float, help="optional steady endpoint-cap negative control")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--max-cell-seconds", type=float, default=120)
    args = parser.parse_args()
    scales = [float(v) for v in args.arrival_scales.split(",")]
    import math
    caps = [int(v) for v in args.caps.split(",")] if args.caps else [args.cap]
    if not caps or len(set(caps)) != len(caps) or min(caps) < 1:
        parser.error("caps must be distinct positive integers")
    args.cap = caps[0]
    if args.cap < 1 or args.repeats < 1 or not scales or any(not math.isfinite(v) or v <= 0 for v in scales):
        parser.error("positive cap, repeats and finite arrival scales required")
    if not math.isfinite(args.max_cell_seconds) or args.max_cell_seconds <= 0:
        parser.error("finite positive time budget required")
    engine_max_seqs = max(caps) if args.engine_max_seqs is None else args.engine_max_seqs
    if engine_max_seqs < max(caps) or args.warmup_condition_rounds < 0:
        parser.error("engine capacity must cover admission cap; warmup rounds must be nonnegative")
    if args.single_action_probe and (sorted(caps) != [8, 12, 16, 32] or engine_max_seqs != 32):
        parser.error("single-action probe freezes engine32 and warmup caps8/12/16/32")
    if any(v is not None and (not math.isfinite(v) or v <= 0) for v in (args.ttft_slo_s, args.tpot_slo_s)):
        parser.error("SLO overrides must be finite and positive")
    if args.underload_scale is not None and (not math.isfinite(args.underload_scale) or args.underload_scale <= max(scales)):
        parser.error("underload scale must be finite and exceed primary arrival scales")
    original = json.loads((args.prepared_dir / "config.json").read_text())
    workload = json.loads((args.prepared_dir / "workload.json").read_text())
    rows, tokens = workload["source_requests"], workload["actual_prompt_token_ids"]
    if len(rows) != original["requests"] or len(tokens) != len(rows) or max(caps) > len(rows):
        raise ValueError("cohort/cap mismatch")
    if hashlib.sha256(json.dumps(workload, sort_keys=True).encode()).hexdigest() != original["workload_sha256"]:
        raise ValueError("prepared workload changed")
    for r, ids in zip(rows, tokens):
        if len(ids) != original["prompt_tokens"] or hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest() != r["prompt_token_ids_sha256"]:
            raise ValueError("prompt token identity mismatch")
    maximum_scale = max(scales + ([args.underload_scale] if args.underload_scale else []))
    if maximum_scale * max(max(t) for t in workload["arrival_traces_s"].values()) >= args.max_cell_seconds:
        raise ValueError("time budget ends before arrivals")
    plans = []
    for repeat in range(args.repeats):
        conditions = [(scale, regime) for scale in scales for regime in ("steady", "bursty")]
        if args.reverse_conditions:
            conditions.reverse()
        arms = [(cap, "static") for cap in caps]
        if args.single_action_probe:
            arms = [(32, "single_shadow"), (32, "single_down"), (16, "static")]
            if args.reverse_conditions:
                arms.reverse()
        if args.include_feedback:
            extra = [(max(caps), "shadow"), (max(caps), "feedback")]
            arms = list(reversed(extra)) + arms if args.reverse_conditions else arms + extra
        for cap, policy in (arms if repeat % 2 == 0 else list(reversed(arms))):
            for scale, regime in (conditions if repeat % 2 == 0 else list(reversed(conditions))):
                plan = dict(repeat=repeat, arrival_scale=scale, regime=regime, cap=cap)
                if args.include_feedback or args.single_action_probe:
                    plan["policy"] = policy
                plans.append(plan)
        if args.underload_scale is not None:
            for cap in (caps[0], caps[-1]):
                plans.append(dict(repeat=repeat, arrival_scale=args.underload_scale, regime="steady", cap=cap))
    config = dict(model=original["model"], output_tokens=original["output_tokens"],
        prompt_tokens=original["prompt_tokens"], requests=len(rows), cap=args.cap,
        ttft_slo_s=original["ttft_slo_s"] if args.ttft_slo_s is None else args.ttft_slo_s,
        tpot_slo_s=original["tpot_slo_s"] if args.tpot_slo_s is None else args.tpot_slo_s, plans=plans,
        reference_slo=dict(ttft_slo_s=original["ttft_slo_s"], tpot_slo_s=original["tpot_slo_s"]),
        engine_max_num_seqs=engine_max_seqs, warmup_condition_rounds=args.warmup_condition_rounds,
        queue_policy="native_fcfs_all_due_requests_submitted", seed=original["seed"],
        evidence_ceiling="NATIVE_VLLM_SYNCHRONOUS_INPROC_REQUEST_MEASUREMENT",
        scheduler_trace="in_memory_schedule_wrapper; cost included in host serving time",
        request_metrics="external host delivery; core timestamps retained separately",
        enforce_eager=args.enforce_eager, max_cell_seconds=args.max_cell_seconds,
        workload_sha256=original["workload_sha256"])
    config.update(caps=caps, underload_scale=args.underload_scale, reverse_conditions=args.reverse_conditions,
                  include_feedback=args.include_feedback, feedback_caps=sorted(caps),
                  single_action_probe=args.single_action_probe, single_down_target=16)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    dump(out / "config.json", config)
    dump(out / "workload.json", workload)
    (out / "commands.txt").write_text(shlex.join([sys.executable, *sys.argv]) + "\n")
    dump(out / "status.json", dict(status="INITIALIZING", cells_completed=0))
    engine = None
    completed = 0
    executed = 0
    try:
        before = gpu_state()
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        os.environ["VLLM_BATCH_INVARIANT"] = "0"
        import torch
        import vllm
        from importlib.metadata import version
        from vllm.engine.arg_utils import EngineArgs
        from vllm.v1.engine.llm_engine import LLMEngine
        if vllm.__version__ != "0.26.0":
            raise RuntimeError("this small scheduler capture is verified against vLLM 0.26.0")
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("one visible CUDA GPU required")
        package = Path(vllm.__file__).parent
        sources = [Path(__file__), Path(__file__).with_name("native_capture.py"),
                   Path(__file__).with_name("metrics.py")]
        if args.include_feedback or args.single_action_probe:
            sources.append(Path(__file__).with_name("admission_feedback.py"))
        environment = dict(python=sys.version, torch=torch.__version__, cuda=torch.version.cuda,
            vllm=vllm.__version__, transformers=version("transformers"), gpu_before_init=before,
            cpu_threads=torch.get_num_threads(),
            source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
            vllm_source_sha256={str(p): hashlib.sha256((package / p).read_bytes()).hexdigest()
                for p in ("v1/core/sched/scheduler.py", "v1/engine/llm_engine.py", "v1/engine/output_processor.py",
                          "v1/worker/gpu_model_runner.py")},
            execution_env={k: os.environ.get(k) for k in ("VLLM_ENABLE_V1_MULTIPROCESSING",
                "VLLM_BATCH_INVARIANT", "VLLM_USE_FLASHINFER_SAMPLER", "OMP_NUM_THREADS",
                "MKL_NUM_THREADS", "CUDA_LAUNCH_BLOCKING")})
        dump(out / "environment.json", environment)
        engine_kwargs = dict(model=original["model"]["id"], revision=original["model"]["revision"],
            tokenizer_revision=original["model"]["tokenizer_revision"], dtype="bfloat16", seed=original["seed"],
            enforce_eager=args.enforce_eager, enable_return_routed_experts=False,
            max_model_len=256, max_num_seqs=engine_max_seqs, max_num_batched_tokens=1024,
            gpu_memory_utilization=0.70, enable_prefix_caching=False, scheduling_policy="fcfs",
            async_scheduling=False, stream_interval=1, disable_log_stats=False)
        dump(out / "engine_args.json", engine_kwargs)
        engine = LLMEngine.from_engine_args(EngineArgs(**engine_kwargs), enable_multiprocessing=False)
        warmups = [(cap, "steady", 0) for cap in sorted(caps)] + [(cap, regime, scale)
            for _ in range(args.warmup_condition_rounds) for cap in sorted(caps)
            for scale in scales for regime in ("steady", "bursty")]
        for index, (cap, regime, scale) in enumerate(warmups):
            cap_state = set_empty_admission_cap(engine, cap)
            print(json.dumps(dict(phase="WARMUP_START", index=index, regime=regime, arrival_scale=scale,
                                  monotonic_s=time.perf_counter())), flush=True)
            warm = capture_episode(engine, workload, dict(config, cap=cap), regime=regime, arrival_scale=scale,
                                   run_id=f"warmup{index}", max_seconds=args.max_cell_seconds)
            dump(out / f"warmup-{index:03d}-summary.json", dict(status=warm["status"], error=warm["error"],
                regime=regime, arrival_scale=scale, cap_state=cap_state, observation_end_s=warm["observation_end_s"],
                requests_completed=sum(r["status"] == "completed" for r in warm["requests"]),
                decode_widths=sorted({s["decode_requests"] for s in warm["scheduler_steps"]})))
            print(json.dumps(dict(phase="WARMUP_END", index=index, status=warm["status"],
                                  monotonic_s=time.perf_counter())), flush=True)
            if warm["status"] != "COMPLETE":
                dump(out / f"warmup-{index:03d}-failure.json", warm)
                raise RuntimeError(f"warmup failed: {warm['error']}")
            del warm
        for index, plan in enumerate(plans):
            before = gpu_state()
            cap_state = set_empty_admission_cap(engine, plan["cap"])
            print(json.dumps(dict(phase="CELL_START", index=index, plan=plan,
                                  monotonic_s=time.perf_counter())), flush=True)
            raw = capture_episode(engine, workload, dict(config, cap=plan["cap"], policy=plan.get("policy", "static")), regime=plan["regime"],
                arrival_scale=plan["arrival_scale"], run_id=f"cell{index}", max_seconds=args.max_cell_seconds)
            raw.update(plan=plan, gpu_before=before, admission_cap_state=cap_state)
            print(json.dumps(dict(phase="CELL_END", index=index, status=raw["status"],
                                  monotonic_s=time.perf_counter())), flush=True)
            dump(out / f"cell-{index:03d}.json", raw)
            executed += 1
            try:
                after = gpu_state()
                dump(out / f"checks-{index:03d}.json", dict(status="PASS", gpu_after=after))
            except Exception as exc:
                dump(out / f"checks-{index:03d}.json", dict(status="FAILED", error=str(exc)))
                raise
            metrics = summarize_episode_requests(raw["requests"], observation_end_s=raw["observation_end_s"],
                ttft_slo_s=config["ttft_slo_s"], tpot_slo_s=config["tpot_slo_s"])
            metrics["tpot_claim_eligible"] = raw["host_chunk_diagnostics"]["token_level_itl_resolved"]
            metrics["timing_semantics"] = "host delivery; do not interpret unresolved chunks as token-generation ITL"
            metrics["reference_slo_metrics"] = summarize_episode_requests(raw["requests"],
                observation_end_s=raw["observation_end_s"], **config["reference_slo"])
            dump(out / f"metrics-{index:03d}.json", metrics)
            completed += 1
            print(json.dumps(dict(cell=index, **plan, status=raw["status"],
                goodput=metrics["goodput_rps"], attainment=metrics["slo_attainment"],
                ttft_p50=metrics["latency_s"]["ttft"]["p50"],
                tpot_p50=metrics["latency_s"]["tpot"]["p50"])), flush=True)
            dump(out / "status.json", dict(status="RUNNING", cells_completed=completed))
            if raw["status"] != "COMPLETE":
                raise RuntimeError(raw["error"])
        dump(out / "status.json", dict(status="COMPLETE", cells_completed=completed,
            scientific_verdict="MEASUREMENT_ONLY_PENDING_NATIVE_RESPONSE_ANALYSIS"))
    except Exception as exc:
        dump(out / "status.json", dict(status="INCOMPLETE" if executed else "BLOCKED",
            cells_executed=executed, cells_completed=completed, error=f"{type(exc).__name__}: {exc}"))
        raise
    finally:
        if engine is not None:
            engine.engine_core.shutdown()


if __name__ == "__main__":
    main()
