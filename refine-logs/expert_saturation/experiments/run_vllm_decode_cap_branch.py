#!/usr/bin/env python3
"""Run one native vLLM initial-cohort branch at a frozen active-sequence cap.

Each invocation constructs a fresh engine, enqueues the same pre-tokenized
cohort in the same order, and drains every request.  Run this script once for
each budget arm and once with route telemetry OFF/ON per arm.  This is an
initial same-start request-level experiment, not a later-epoch KV snapshot
replay and not an online controller.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from run_vllm_route_shape_probe import (
    _environment,
    _json_hash,
    _load_workload,
    _quantile,
    build_exact_prompts,
    summarize_routes,
)


SCHEMA = "vllm-native-initial-decode-cap-branch-v1"
CLAIM_CEILING = "REQUEST_LEVEL_INITIAL_SAME_START_BRANCH_EXPLORATORY"
HELPER_SOURCE_ARTIFACT = "run_vllm_route_shape_probe.py"
TIMING_FIELDS = (
    "ttft_ms",
    "queue_ms",
    "prefill_ms",
    "decode_span_ms",
    "tpot_ms",
    "e2e_ms",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def producer_source_bundle() -> dict[str, bytes]:
    """Return every local Python source required to execute this producer."""
    return {
        "producer_source.py": Path(__file__).read_bytes(),
        HELPER_SOURCE_ARTIFACT: Path(__file__).with_name(
            HELPER_SOURCE_ARTIFACT
        ).read_bytes(),
    }


def summarize_request_outputs(
    outputs: Sequence[Any],
    prompt_ids: Sequence[Sequence[int]],
    expected_output_tokens: int,
    *,
    branch_started_s: float,
    branch_finished_s: float,
) -> list[dict[str, Any]]:
    """Validate the frozen denominator and retain every request metric."""
    if (
        not math.isfinite(branch_started_s)
        or not math.isfinite(branch_finished_s)
        or branch_finished_s <= branch_started_s
    ):
        raise ValueError("branch wall-clock interval must be finite and positive")
    if len(outputs) != len(prompt_ids):
        raise ValueError(
            f"request output count mismatch: expected {len(prompt_ids)}, got {len(outputs)}"
        )
    rows: list[dict[str, Any]] = []
    for index, (output, expected_prompt) in enumerate(zip(outputs, prompt_ids)):
        if len(output.outputs) != 1:
            raise ValueError("exactly one completion per request is required")
        completion = output.outputs[0]
        metrics = output.metrics
        if metrics is None:
            raise ValueError("vLLM request metrics are missing")
        observed_prompt = list(output.prompt_token_ids or [])
        if observed_prompt != list(expected_prompt):
            raise ValueError(f"prompt identity/order mismatch at cohort index {index}")
        generated = int(metrics.num_generation_tokens)
        token_ids = list(completion.token_ids)
        if generated != expected_output_tokens or len(token_ids) != generated:
            raise ValueError("generated-token count drifted from the frozen denominator")
        if completion.finish_reason != "length":
            raise ValueError(f"unexpected finish reason: {completion.finish_reason!r}")
        raw_timing_s = {
            "branch_started_perf_counter": float(branch_started_s),
            "branch_finished_perf_counter": float(branch_finished_s),
            "queued_ts": float(metrics.queued_ts),
            "scheduled_ts": float(metrics.scheduled_ts),
            "first_token_ts": float(metrics.first_token_ts),
            "last_token_ts": float(metrics.last_token_ts),
            "first_token_latency": float(metrics.first_token_latency),
        }
        if any(not math.isfinite(value) for value in raw_timing_s.values()):
            raise ValueError("request timing contains a non-finite value")
        if generated < 2:
            raise ValueError("at least two generated tokens are required")
        if not (
            raw_timing_s["queued_ts"] <= raw_timing_s["scheduled_ts"]
            < raw_timing_s["first_token_ts"]
            < raw_timing_s["last_token_ts"]
        ):
            raise ValueError("request timing timestamps are not ordered")
        if raw_timing_s["first_token_latency"] <= 0:
            raise ValueError("request TTFT must be positive")
        ttft_s = raw_timing_s["first_token_latency"]
        queue_s = raw_timing_s["scheduled_ts"] - raw_timing_s["queued_ts"]
        prefill_s = raw_timing_s["first_token_ts"] - raw_timing_s["scheduled_ts"]
        decode_s = raw_timing_s["last_token_ts"] - raw_timing_s["first_token_ts"]
        rows.append(
            {
                "cohort_index": index,
                "request_id": str(output.request_id),
                "prompt_token_ids_sha256": _json_hash(list(expected_prompt)),
                "generated_tokens": generated,
                "decode_intervals": generated - 1,
                "finish_reason": completion.finish_reason,
                "raw_timing_s": raw_timing_s,
                "ttft_ms": ttft_s * 1000.0,
                "queue_ms": queue_s * 1000.0,
                "prefill_ms": prefill_s * 1000.0,
                "decode_span_ms": decode_s * 1000.0,
                "tpot_ms": decode_s * 1000.0 / (generated - 1),
                "e2e_ms": (ttft_s + decode_s) * 1000.0,
                "token_ids": token_ids,
            }
        )
    return rows


def summarize_branch(
    rows: Sequence[dict[str, Any]], wall_s: float, output_tokens: int
) -> dict[str, Any]:
    if not rows or wall_s <= 0:
        raise ValueError("non-empty rows and positive wall time are required")
    result: dict[str, Any] = {
        "request_count": len(rows),
        "completed_request_count": len(rows),
        "expected_output_tokens_per_request": output_tokens,
        "total_generated_tokens": sum(int(r["generated_tokens"]) for r in rows),
        "total_decode_intervals": sum(int(r["decode_intervals"]) for r in rows),
        "wall_ms": wall_s * 1000.0,
        "throughput_output_tokens_per_s": sum(
            int(r["generated_tokens"]) for r in rows
        )
        / wall_s,
    }
    for field in TIMING_FIELDS:
        values = [float(row[field]) for row in rows]
        result[f"request_{field.removesuffix('_ms')}_p50_ms"] = _quantile(
            values, 0.50
        )
        result[f"request_{field.removesuffix('_ms')}_p95_ms"] = _quantile(
            values, 0.95
        )
        result[f"request_{field.removesuffix('_ms')}_max_ms"] = max(values)
    return result


def summarize_fcfs_waves(
    decode_routes: np.ndarray, decode_cap: int, num_experts: int
) -> dict[str, Any]:
    """Summarize inferred equal-length FCFS waves; no scheduler trace claim."""
    if decode_routes.ndim != 4:
        raise ValueError(f"expected route tensor [N,T,L,K], got {decode_routes.shape}")
    if len(decode_routes) % decode_cap:
        raise ValueError("cohort size must be divisible by decode cap")
    waves: list[dict[str, Any]] = []
    for wave_index, start in enumerate(range(0, len(decode_routes), decode_cap)):
        metrics = summarize_routes(
            [route for route in decode_routes[start : start + decode_cap]],
            num_experts,
        )
        waves.append(
            {
                "wave_index": wave_index,
                "cohort_index_start": start,
                "cohort_index_end_exclusive": start + decode_cap,
                "max_layer_step_load": metrics["max_layer_step_load"],
                "p95_layer_step_max_load": metrics["p95_layer_step_max_load"],
                "max_layer_step_concentration": metrics[
                    "max_layer_step_concentration"
                ],
                "mean_active_expert_fraction": metrics[
                    "mean_active_expert_fraction"
                ],
                "mean_load_cv": metrics["mean_load_cv"],
                "mean_layer_working_set_fraction": metrics[
                    "mean_layer_working_set_fraction"
                ],
            }
        )
    loads = [float(w["max_layer_step_load"]) for w in waves]
    return {
        "scope": "INFERRED_EQUAL_LENGTH_FCFS_CAP_WAVES",
        "scheduler_trace_captured": False,
        "wave_count": len(waves),
        "max_expert_load_across_waves": max(loads),
        "p95_wave_max_expert_load": _quantile(loads, 0.95),
        "waves": waves,
    }


def run(args: argparse.Namespace) -> None:
    from vllm import LLM, SamplingParams

    environment = _environment()
    processes = environment["compute_processes_before_engine_init"]
    if args.require_exclusive_gpu and processes is None:
        raise RuntimeError("cannot verify GPU process isolation")
    if args.require_exclusive_gpu and processes:
        raise RuntimeError(f"GPU is not isolated: {processes}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    source_bundle = producer_source_bundle()
    producer_bytes = source_bundle["producer_source.py"]
    producer_sha256 = hashlib.sha256(producer_bytes).hexdigest()
    (output_dir / "producer_source.py").write_bytes(producer_bytes)
    helper_bytes = source_bundle[HELPER_SOURCE_ARTIFACT]
    helper_sha256 = hashlib.sha256(helper_bytes).hexdigest()
    (output_dir / HELPER_SOURCE_ARTIFACT).write_bytes(helper_bytes)
    workload_path = Path(args.workload_manifest)
    workload_bytes = workload_path.read_bytes()
    texts = _load_workload(workload_path)

    runtime_identity = {
        key: environment[key]
        for key in (
            "python",
            "platform",
            "torch",
            "torch_cuda",
            "vllm",
            "gpu",
            "vllm_batch_invariant",
            "vllm_use_flashinfer_sampler",
            "vllm_runtime_sources",
            "vllm_actuator_sources",
            "compute_processes_before_engine_init",
        )
    }
    _write_json(output_dir / "environment.json", environment)
    (output_dir / "workload_manifest.json").write_bytes(workload_bytes)

    llm = LLM(
        model=args.model,
        revision=args.revision,
        tokenizer_revision=args.revision,
        dtype=args.dtype,
        seed=args.seed,
        enforce_eager=args.enforce_eager,
        enable_return_routed_experts=args.capture_routes,
        disable_log_stats=False,
        enable_prefix_caching=False,
        max_model_len=args.max_model_len,
        max_num_seqs=args.decode_cap,
        max_num_batched_tokens=args.max_num_batched_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        scheduling_policy="fcfs",
    )
    tokenizer = llm.get_tokenizer()
    prompt_ids = build_exact_prompts(
        tokenizer, texts, args.prompt_length, args.cohort_size, args.prompt_offset
    )
    prompt_array = np.asarray(prompt_ids, dtype=np.int32)
    np.savez_compressed(output_dir / "input_cohort.npz", prompt_token_ids=prompt_array)

    hf_config = llm.llm_engine.vllm_config.model_config.hf_text_config
    route_shape = {
        "num_experts": int(getattr(hf_config, "num_experts")),
        "num_layers": int(getattr(hf_config, "num_hidden_layers")),
        "top_k": int(getattr(hf_config, "num_experts_per_tok")),
        "decode_route_steps": args.output_tokens - 1,
    }
    if (
        route_shape["num_experts"] < 2
        or route_shape["num_layers"] < 1
        or route_shape["top_k"] < 1
        or route_shape["top_k"] > route_shape["num_experts"]
    ):
        raise ValueError(f"invalid frozen route shape: {route_shape}")
    script_sha256 = producer_sha256
    experiment_identity = {
        "model": args.model,
        "revision": args.revision,
        "dtype": args.dtype,
        "seed": args.seed,
        "replicate_id": args.replicate_id,
        "workload_manifest_sha256": hashlib.sha256(workload_bytes).hexdigest(),
        "prompt_token_ids_sha256": _json_hash(prompt_ids),
        "prompt_length": args.prompt_length,
        "prompt_offset": args.prompt_offset,
        "cohort_size": args.cohort_size,
        "output_tokens": args.output_tokens,
        "warmup_output_tokens": args.warmup_output_tokens,
        "max_model_len": args.max_model_len,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "enforce_eager": args.enforce_eager,
        "require_exclusive_gpu": args.require_exclusive_gpu,
        "compute_processes_before_engine_init": processes,
        "route_shape": route_shape,
        "runner_sha256": script_sha256,
        "helper_source_sha256": helper_sha256,
    }
    config = vars(args) | {
        "schema": SCHEMA,
        "claim_ceiling": CLAIM_CEILING,
        "scheduler_policy": "fcfs",
        "experiment_identity": experiment_identity,
        "experiment_identity_sha256": _json_hash(experiment_identity),
        "runtime_identity": runtime_identity,
        "runtime_identity_sha256": _json_hash(runtime_identity),
        "probe_script_sha256": script_sha256,
        "producer_source_sha256": script_sha256,
        "producer_source_artifact": "producer_source.py",
        "helper_source_sha256": helper_sha256,
        "helper_source_artifact": HELPER_SOURCE_ARTIFACT,
        "route_shape": route_shape,
    }
    _write_json(output_dir / "config.json", config)

    warm_params = [
        SamplingParams(
            temperature=0.0,
            max_tokens=args.warmup_output_tokens,
            min_tokens=args.warmup_output_tokens,
            ignore_eos=True,
            seed=args.seed,
            routed_experts_prompt_start=args.prompt_length - 1,
        )
        for _ in prompt_ids
    ]
    llm.generate(
        [{"prompt_token_ids": ids} for ids in prompt_ids],
        warm_params,
        use_tqdm=False,
    )

    params = [
        SamplingParams(
            temperature=0.0,
            max_tokens=args.output_tokens,
            min_tokens=args.output_tokens,
            ignore_eos=True,
            seed=args.seed,
            routed_experts_prompt_start=args.prompt_length - 1,
        )
        for _ in prompt_ids
    ]
    started = time.perf_counter()
    outputs = llm.generate(
        [{"prompt_token_ids": ids} for ids in prompt_ids], params, use_tqdm=False
    )
    finished = time.perf_counter()
    wall_s = finished - started
    rows = summarize_request_outputs(
        outputs,
        prompt_ids,
        args.output_tokens,
        branch_started_s=started,
        branch_finished_s=finished,
    )
    with (output_dir / "requests.jsonl").open("x") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")

    summary: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "COMPLETE",
        "claim_ceiling": CLAIM_CEILING,
        "budget_arm": args.budget_arm,
        "decode_cap": args.decode_cap,
        "capture_routes": args.capture_routes,
        "timing": summarize_branch(rows, wall_s, args.output_tokens),
        "anti_claims": [
            "max_num_seqs also bounds initial prefill admission; this is not a pure later-epoch decode-only actuator",
            "fresh-engine initial cohorts are not KV snapshot replay",
            "fixed offline arrivals are not an online controller",
            "inferred FCFS waves are not native scheduler-event traces",
            "single-GPU evidence is not Expert Parallel evidence",
        ],
    }
    if args.capture_routes:
        num_experts = route_shape["num_experts"]
        num_layers = route_shape["num_layers"]
        top_k = route_shape["top_k"]
        full_routes = [np.asarray(output.outputs[0].routed_experts) for output in outputs]
        expected = (args.output_tokens, num_layers, top_k)
        if any(tuple(route.shape) != expected for route in full_routes):
            raise ValueError(f"routed-expert shape mismatch; expected {expected}")
        decode_routes = np.stack([route[1:] for route in full_routes])
        if not np.issubdtype(decode_routes.dtype, np.integer):
            raise ValueError(f"routed-expert dtype is not integral: {decode_routes.dtype}")
        if np.any(decode_routes < 0) or np.any(decode_routes >= num_experts):
            raise ValueError("routed-expert ID is outside the frozen expert range")
        if top_k > 1 and np.any(np.diff(np.sort(decode_routes, axis=-1), axis=-1) == 0):
            raise ValueError("routed-expert top-k contains duplicate expert IDs")
        np.savez_compressed(output_dir / "routes.npz", routes=decode_routes)
        summary["route_shape"] = route_shape
        summary["route_pressure"] = summarize_fcfs_waves(
            decode_routes, args.decode_cap, num_experts
        )
    _write_json(output_dir / "summary.json", summary)

    artifacts = [
        "config.json",
        "environment.json",
        "workload_manifest.json",
        "input_cohort.npz",
        "requests.jsonl",
        "summary.json",
        "producer_source.py",
        HELPER_SOURCE_ARTIFACT,
    ] + (["routes.npz"] if args.capture_routes else [])
    artifact_hashes = {name: _sha256(output_dir / name) for name in artifacts}
    _write_json(output_dir / "ARTIFACT_HASHES.json", artifact_hashes)
    seal = {
        "status": "RUN_COMPLETE",
        "schema": SCHEMA,
        "config_sha256": artifact_hashes["config.json"],
        "requests_sha256": artifact_hashes["requests.jsonl"],
        "summary_sha256": artifact_hashes["summary.json"],
        "producer_source_sha256": artifact_hashes["producer_source.py"],
        "helper_source_sha256": artifact_hashes[HELPER_SOURCE_ARTIFACT],
        "artifact_hashes": artifact_hashes,
        "artifact_hashes_sha256": _sha256(output_dir / "ARTIFACT_HASHES.json"),
    }
    _write_json(output_dir / "RUN_COMPLETE.json", seal)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workload-manifest", required=True)
    parser.add_argument("--budget-arm", choices=("low", "mid", "high"), required=True)
    parser.add_argument("--decode-cap", type=int, required=True)
    parser.add_argument("--replicate-id", required=True)
    parser.add_argument("--model", default="allenai/OLMoE-1B-7B-0924")
    parser.add_argument("--revision", default="6d84c48581ece794365f2b8e9cfb043c68ade9c5")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--prompt-length", type=int, default=512)
    parser.add_argument("--prompt-offset", type=int, default=0)
    parser.add_argument("--cohort-size", type=int, default=48)
    parser.add_argument("--output-tokens", type=int, default=32)
    parser.add_argument("--warmup-output-tokens", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    parser.add_argument("--runtime-patch-id", required=True)
    parser.add_argument("--capture-routes", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enforce-eager", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--require-exclusive-gpu", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()
    if args.decode_cap < 1 or args.decode_cap >= args.cohort_size:
        parser.error("decode-cap must be positive and smaller than cohort-size")
    if args.cohort_size % args.decode_cap:
        parser.error("cohort-size must be divisible by decode-cap")
    if args.prompt_length + args.output_tokens > args.max_model_len:
        parser.error("max-model-len is smaller than prompt + output")
    if args.max_num_batched_tokens < args.decode_cap * args.prompt_length:
        parser.error("token budget must fit a full equal-length cap wave prefill")
    if args.output_tokens < 3 or args.warmup_output_tokens < 1:
        parser.error("output-tokens must be >=3 and warmup-output-tokens >=1")
    if not args.runtime_patch_id.strip() or not args.replicate_id.strip():
        parser.error("runtime-patch-id and replicate-id must be non-empty")
    return args


if __name__ == "__main__":
    run(parse_args())
