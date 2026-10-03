#!/usr/bin/env python3
"""Native vLLM fixed-batch probe for MoE route shape and request latency.

This is an existence probe, not an admission-policy experiment.  Within each
cell, model, prompt length, batch size, output length, and engine settings are
fixed; only the request composition changes.  The returned routed-expert IDs
are observational outcomes and must not be presented as an action Oracle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np


CLAIM_CEILING = "NATIVE_OFFLINE_FIXED_BATCH_MEASUREMENT_ONLY"


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def _quantile(values: Sequence[float], q: float) -> float:
    return float(np.quantile(values, q)) if values else float("nan")


def _json_hash(value: Any) -> str:
    blob = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def summarize_routes(
    routes_by_request: Sequence[np.ndarray], num_experts: int
) -> dict[str, Any]:
    """Aggregate [request, decode_step, layer, top_k] expert IDs."""
    if not routes_by_request:
        raise ValueError("routes_by_request must be non-empty")
    if num_experts < 2:
        raise ValueError("num_experts must be >= 2")
    shapes = {tuple(np.asarray(x).shape) for x in routes_by_request}
    if len(shapes) != 1:
        raise ValueError(f"route shapes differ across requests: {sorted(shapes)}")
    routes = np.stack(routes_by_request).astype(np.int64, copy=False)
    if routes.ndim != 4:
        raise ValueError(f"expected [B,T,L,K], got {routes.shape}")
    if routes.shape[1] < 1:
        raise ValueError("at least one decode route step is required")
    if np.any(routes < 0) or np.any(routes >= num_experts):
        raise ValueError("expert ID outside configured range")

    batch, steps, layers, top_k = routes.shape
    max_loads: list[float] = []
    concentrations: list[float] = []
    active_counts: list[float] = []
    entropies: list[float] = []
    hhis: list[float] = []
    load_cvs: list[float] = []
    temporal_jaccards: list[float] = []
    per_request_temporal_jaccards: list[float] = []
    per_request_exact_route_matches: list[float] = []
    load_vector_cosines: list[float] = []
    load_vectors = np.zeros((steps, layers, num_experts), dtype=np.float64)

    for step in range(steps):
        for layer in range(layers):
            ids = routes[:, step, layer, :].reshape(-1)
            counts = np.bincount(ids, minlength=num_experts).astype(np.float64)
            load_vectors[step, layer] = counts
            probs = counts / counts.sum()
            nonzero = probs[probs > 0]
            max_loads.append(float(counts.max()))
            concentrations.append(float(probs.max()))
            active_counts.append(float(np.count_nonzero(counts)))
            entropies.append(float(-(nonzero * np.log(nonzero)).sum() / math.log(num_experts)))
            hhis.append(float(np.square(probs).sum()))
            load_cvs.append(float(counts.std() / counts.mean()))

    working_sets: list[float] = []
    for layer in range(layers):
        working_sets.append(float(np.unique(routes[:, :, layer, :]).size))
        for step in range(1, steps):
            previous = set(routes[:, step - 1, layer, :].reshape(-1).tolist())
            current = set(routes[:, step, layer, :].reshape(-1).tolist())
            temporal_jaccards.append(len(previous & current) / len(previous | current))
            previous_load = load_vectors[step - 1, layer]
            current_load = load_vectors[step, layer]
            load_vector_cosines.append(
                float(
                    np.dot(previous_load, current_load)
                    / (np.linalg.norm(previous_load) * np.linalg.norm(current_load))
                )
            )
            for request in range(batch):
                previous_request = set(routes[request, step - 1, layer].tolist())
                current_request = set(routes[request, step, layer].tolist())
                per_request_temporal_jaccards.append(
                    len(previous_request & current_request)
                    / len(previous_request | current_request)
                )
                per_request_exact_route_matches.append(
                    float(previous_request == current_request)
                )

    return {
        "batch_size": batch,
        "decode_route_steps": steps,
        "num_layers": layers,
        "top_k": top_k,
        "num_experts": num_experts,
        "total_assignments": int(routes.size),
        "max_layer_step_load": max(max_loads),
        "mean_layer_step_max_load": _mean(max_loads),
        "p95_layer_step_max_load": _quantile(max_loads, 0.95),
        "max_layer_step_concentration": max(concentrations),
        "mean_layer_step_concentration": _mean(concentrations),
        "p95_layer_step_concentration": _quantile(concentrations, 0.95),
        "mean_active_experts": _mean(active_counts),
        "mean_active_expert_fraction": _mean(active_counts) / num_experts,
        "mean_normalized_entropy": _mean(entropies),
        "mean_hhi": _mean(hhis),
        "mean_load_cv": _mean(load_cvs),
        "mean_layer_working_set": _mean(working_sets),
        "mean_layer_working_set_fraction": _mean(working_sets) / num_experts,
        "mean_temporal_jaccard": _mean(temporal_jaccards),
        "mean_temporal_churn": 1.0 - _mean(temporal_jaccards),
        "mean_per_request_temporal_jaccard": _mean(
            per_request_temporal_jaccards
        ),
        "per_request_exact_route_match_fraction": _mean(
            per_request_exact_route_matches
        ),
        "mean_temporal_load_vector_cosine": _mean(load_vector_cosines),
    }


def summarize_timings(
    outputs: Sequence[Any], wall_s: float, expected_output_tokens: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for output in outputs:
        if len(output.outputs) != 1:
            raise ValueError("probe requires exactly one completion per request")
        completion = output.outputs[0]
        metrics = output.metrics
        if metrics is None:
            raise ValueError("vLLM request metrics missing; disable_log_stats must be false")
        generated = int(metrics.num_generation_tokens)
        if generated != expected_output_tokens or len(completion.token_ids) != generated:
            raise ValueError("generated-token count drifted from the frozen denominator")
        if completion.finish_reason != "length":
            raise ValueError(f"unexpected finish reason: {completion.finish_reason!r}")
        if generated < 2 or metrics.last_token_ts < metrics.first_token_ts:
            raise ValueError("invalid generation timing interval")
        rows.append(
            {
                "request_id": output.request_id,
                "generated_tokens": generated,
                "ttft_ms": float(metrics.first_token_latency * 1000.0),
                "queue_ms": float(max(0.0, metrics.scheduled_ts - metrics.queued_ts) * 1000.0),
                "decode_span_ms": float((metrics.last_token_ts - metrics.first_token_ts) * 1000.0),
                "tpot_ms": float(
                    (metrics.last_token_ts - metrics.first_token_ts)
                    * 1000.0
                    / (generated - 1)
                ),
                "token_ids": list(completion.token_ids),
                "finish_reason": completion.finish_reason,
            }
        )
    tpots = [row["tpot_ms"] for row in rows]
    ttfts = [row["ttft_ms"] for row in rows]
    queues = [row["queue_ms"] for row in rows]
    total_tokens = sum(row["generated_tokens"] for row in rows)
    return (
        {
            "wall_ms": wall_s * 1000.0,
            "throughput_tokens_per_s": total_tokens / wall_s,
            "request_tpot_p50_ms": _quantile(tpots, 0.50),
            "request_tpot_p95_ms": _quantile(tpots, 0.95),
            "request_tpot_max_ms": max(tpots),
            "request_ttft_p50_ms": _quantile(ttfts, 0.50),
            "request_ttft_p95_ms": _quantile(ttfts, 0.95),
            "request_queue_p95_ms": _quantile(queues, 0.95),
        },
        rows,
    )


def build_exact_prompts(
    tokenizer: Any, texts: Sequence[str], prompt_length: int, count: int, offset: int
) -> list[list[int]]:
    if not texts or prompt_length < 2 or count < 1:
        raise ValueError("invalid prompt construction input")
    separator = tokenizer.encode("\n", add_special_tokens=False) or [1]
    prompts: list[list[int]] = []
    for sample in range(count):
        ids: list[int] = []
        cursor = offset + sample
        while len(ids) < prompt_length:
            text = texts[cursor % len(texts)]
            token_ids = tokenizer.encode(text, add_special_tokens=False)
            if token_ids:
                ids.extend(token_ids)
                ids.extend(separator)
            cursor += 1
            if cursor - (offset + sample) > len(texts) * 4 and not ids:
                raise ValueError("workload contains no tokenizable text")
        prompts.append(ids[:prompt_length])
    return prompts


def _pearson(x: Sequence[float], y: Sequence[float]) -> float | None:
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def build_cell_summaries(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    cells: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        cells[(record["prompt_length"], record["batch_size"])].append(record)
    summaries: list[dict[str, Any]] = []
    for (prompt_length, batch_size), rows in sorted(cells.items()):
        item: dict[str, Any] = {
            "prompt_length": prompt_length,
            "batch_size": batch_size,
            "samples": len(rows),
            "tpot_p95_ms_mean": _mean([r["timing"]["request_tpot_p95_ms"] for r in rows]),
            "tpot_p95_ms_cv": float(
                np.std([r["timing"]["request_tpot_p95_ms"] for r in rows])
                / np.mean([r["timing"]["request_tpot_p95_ms"] for r in rows])
            ),
        }
        if "route" in rows[0]:
            pressure = [r["route"]["max_layer_step_concentration"] for r in rows]
            working_set = [r["route"]["mean_layer_working_set_fraction"] for r in rows]
            latency = [r["timing"]["request_tpot_p95_ms"] for r in rows]
            ordered = sorted(rows, key=lambda r: r["route"]["max_layer_step_concentration"])
            low, high = ordered[0], ordered[-1]
            item.update(
                {
                    "concentration_to_tpot_pearson": _pearson(pressure, latency),
                    "working_set_to_tpot_pearson": _pearson(working_set, latency),
                    "concentration_range": [min(pressure), max(pressure)],
                    "working_set_fraction_range": [min(working_set), max(working_set)],
                    "observational_high_minus_low_tpot_pct": 100.0
                    * (
                        high["timing"]["request_tpot_p95_ms"]
                        / low["timing"]["request_tpot_p95_ms"]
                        - 1.0
                    ),
                    "low_concentration_batch_id": low["batch_id"],
                    "high_concentration_batch_id": high["batch_id"],
                }
            )
        summaries.append(item)
    return summaries


def _environment() -> dict[str, Any]:
    import torch
    import vllm

    package_root = Path(vllm.__file__).resolve().parent
    runtime_sources: dict[str, dict[str, Any]] = {}
    for relative in (
        "model_executor/layers/fused_moe/routed_experts_capturer.py",
        "v1/worker/gpu_model_runner.py",
    ):
        source = package_root / relative
        if not source.is_file():
            raise RuntimeError(f"missing vLLM runtime source: {source}")
        runtime_sources[relative] = {
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "size_bytes": source.stat().st_size,
        }

    # Keep actuator provenance separate from the route-telemetry patch map.
    # The telemetry implementation comparator intentionally validates the two
    # patched files above, while decode-cap experiments additionally depend on
    # these stock scheduler/offline-entrypoint semantics.
    actuator_sources: dict[str, dict[str, Any]] = {}
    for relative in (
        "entrypoints/offline_utils.py",
        "v1/core/sched/scheduler.py",
        "v1/core/sched/request_queue.py",
        "config/scheduler.py",
    ):
        source = package_root / relative
        if not source.is_file():
            raise RuntimeError(f"missing vLLM actuator source: {source}")
        actuator_sources[relative] = {
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "size_bytes": source.stat().st_size,
        }

    try:
        gpu = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            text=True,
        ).strip()
    except Exception as exc:  # pragma: no cover - environment evidence only
        gpu = f"unavailable:{type(exc).__name__}"
    try:
        process_text = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip()
        compute_processes: list[str] | None = (
            [line for line in process_text.splitlines() if line.strip()]
            if process_text
            else []
        )
    except Exception:  # pragma: no cover - environment evidence only
        compute_processes = None
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "vllm": vllm.__version__,
        "vllm_runtime_sources": runtime_sources,
        "vllm_actuator_sources": actuator_sources,
        "gpu": gpu,
        "compute_processes_before_engine_init": compute_processes,
        "vllm_batch_invariant": os.environ.get("VLLM_BATCH_INVARIANT", "0"),
        "vllm_use_flashinfer_sampler": os.environ.get(
            "VLLM_USE_FLASHINFER_SAMPLER", "1"
        ),
    }


def _load_workload(path: Path) -> list[str]:
    payload = json.loads(path.read_text())
    texts = [str(row["prompt"]) for row in payload.get("requests", []) if row.get("prompt")]
    if len(texts) < 16:
        raise ValueError("workload must contain at least 16 non-empty prompts")
    return texts


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
    route_dir = output_dir / "routes"
    route_dir.mkdir()
    input_dir = output_dir / "inputs"
    input_dir.mkdir()
    workload_path = Path(args.workload_manifest)
    workload_bytes = workload_path.read_bytes()
    producer_source_bytes = Path(__file__).read_bytes()
    producer_source_sha256 = hashlib.sha256(producer_source_bytes).hexdigest()
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
        )
    }
    config = vars(args) | {
        "claim_ceiling": CLAIM_CEILING,
        "workload_manifest_sha256": hashlib.sha256(workload_bytes).hexdigest(),
        "probe_script_sha256": producer_source_sha256,
        "producer_source_artifact": "producer_source.py",
        "producer_source_artifact_sha256": producer_source_sha256,
        "runtime_identity": runtime_identity,
    }
    (output_dir / "environment.json").write_text(
        json.dumps(environment, indent=2, sort_keys=True)
    )
    (output_dir / "workload_manifest.json").write_bytes(workload_bytes)
    # Embed the exact producer bytes in every new bundle. A script hash that no
    # longer resolves to source is insufficient provenance for a formal rerun.
    (output_dir / "producer_source.py").write_bytes(producer_source_bytes)

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
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    tokenizer = llm.get_tokenizer()
    hf_config = llm.llm_engine.vllm_config.model_config.hf_text_config
    num_experts = int(getattr(hf_config, "num_experts"))
    num_layers = int(getattr(hf_config, "num_hidden_layers"))
    top_k = int(getattr(hf_config, "num_experts_per_tok"))
    model_shape = {"num_experts": num_experts, "num_layers": num_layers, "top_k": top_k}
    (output_dir / "model_shape.json").write_text(json.dumps(model_shape, indent=2, sort_keys=True))
    config["model_shape"] = model_shape
    (output_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True))

    # Warm every physical Token/KV cell once. A single B=1 warmup does not
    # cover the larger prefill/decode kernel shapes and would contaminate the
    # first recorded composition in those cells with compilation/cold caches.
    for prompt_length in sorted(set(args.prompt_lengths)):
        for batch_size in sorted(set(args.batch_sizes)):
            warm_prompts = build_exact_prompts(
                tokenizer, texts, prompt_length, batch_size, 0
            )
            warm_params = [
                SamplingParams(
                    temperature=0.0,
                    max_tokens=4,
                    min_tokens=4,
                    ignore_eos=True,
                    seed=args.seed,
                    routed_experts_prompt_start=prompt_length - 1,
                )
                for _ in warm_prompts
            ]
            llm.generate(
                [{"prompt_token_ids": ids} for ids in warm_prompts],
                warm_params,
                use_tqdm=False,
            )

    specs = [
        (prompt_length, batch_size, group, repeat)
        for prompt_length in args.prompt_lengths
        for batch_size in args.batch_sizes
        for group in range(args.groups)
        for repeat in range(args.within_process_repeats)
    ]
    random.Random(args.order_seed).shuffle(specs)
    records: list[dict[str, Any]] = []
    artifact_hashes: dict[str, str] = {}
    for fixed_artifact in (
        "environment.json",
        "model_shape.json",
        "producer_source.py",
        "workload_manifest.json",
    ):
        artifact_hashes[fixed_artifact] = hashlib.sha256(
            (output_dir / fixed_artifact).read_bytes()
        ).hexdigest()
    max_batch = max(args.batch_sizes)
    raw_path = output_dir / "batches.jsonl"
    with raw_path.open("x") as raw_file:
        for order, (prompt_length, batch_size, group, repeat) in enumerate(specs):
            prompt_ids = build_exact_prompts(
                tokenizer, texts, prompt_length, batch_size, group * max_batch
            )
            params = [
                SamplingParams(
                    temperature=0.0,
                    max_tokens=args.output_tokens,
                    min_tokens=args.output_tokens,
                    ignore_eos=True,
                    seed=args.seed,
                    routed_experts_prompt_start=prompt_length - 1,
                )
                for _ in prompt_ids
            ]
            started = time.perf_counter()
            outputs = llm.generate(
                [{"prompt_token_ids": ids} for ids in prompt_ids], params, use_tqdm=False
            )
            wall_s = time.perf_counter() - started
            if len(outputs) != batch_size:
                raise ValueError(
                    f"request output count mismatch: expected {batch_size}, got {len(outputs)}"
                )
            timing, request_rows = summarize_timings(
                outputs, wall_s, args.output_tokens
            )
            batch_id = f"r{args.process_repeat:02d}-p{prompt_length}-b{batch_size}-g{group:02d}-w{repeat:02d}"
            input_path = input_dir / f"{batch_id}.npz"
            np.savez_compressed(
                input_path, prompt_token_ids=np.asarray(prompt_ids, dtype=np.int32)
            )
            input_relative = str(input_path.relative_to(output_dir))
            input_sha256 = hashlib.sha256(input_path.read_bytes()).hexdigest()
            artifact_hashes[input_relative] = input_sha256
            record: dict[str, Any] = {
                "batch_id": batch_id,
                "execution_order": order,
                "process_repeat": args.process_repeat,
                "within_process_repeat": repeat,
                "prompt_length": prompt_length,
                "batch_size": batch_size,
                "group": group,
                "prompt_token_ids_sha256": _json_hash(prompt_ids),
                "input_artifact": input_relative,
                "input_artifact_sha256": input_sha256,
                "request_metrics": request_rows,
                "timing": timing,
            }
            if args.capture_routes:
                full_routes = [np.asarray(output.outputs[0].routed_experts) for output in outputs]
                expected = (args.output_tokens, num_layers, top_k)
                if any(tuple(route.shape) != expected for route in full_routes):
                    raise ValueError(
                        f"routed-expert shape mismatch: expected {expected}, "
                        f"got {[tuple(route.shape) for route in full_routes]}"
                    )
                # Element zero was computed as part of prefill. Remaining rows
                # are the output-token decode forward passes.
                decode_routes = [route[1:] for route in full_routes]
                route_path = route_dir / f"{batch_id}.npz"
                np.savez_compressed(route_path, routes=np.stack(decode_routes))
                route_relative = str(route_path.relative_to(output_dir))
                route_sha256 = hashlib.sha256(route_path.read_bytes()).hexdigest()
                artifact_hashes[route_relative] = route_sha256
                record["route_artifact"] = route_relative
                record["route_artifact_sha256"] = route_sha256
                record["route"] = summarize_routes(decode_routes, num_experts)
            raw_file.write(json.dumps(record, sort_keys=True) + "\n")
            raw_file.flush()
            records.append(record)

    summary = {
        "schema": "vllm-native-route-shape-probe-v1",
        "status": "COMPLETE",
        "claim_ceiling": CLAIM_CEILING,
        "capture_routes": args.capture_routes,
        "record_count": len(records),
        "cell_summaries": build_cell_summaries(records),
        "anti_claims": [
            "route metrics are outcomes, not pre-action signals",
            "fixed offline batches are not online arrival or queue evidence",
            "single-GPU evidence is not Expert Parallel evidence",
            "correlation does not authorize an admission controller",
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    hash_manifest = output_dir / "ARTIFACT_HASHES.json"
    hash_manifest.write_text(json.dumps(artifact_hashes, indent=2, sort_keys=True))
    seal = {
        "status": "RUN_COMPLETE",
        "config_sha256": hashlib.sha256((output_dir / "config.json").read_bytes()).hexdigest(),
        "raw_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        "summary_sha256": hashlib.sha256((output_dir / "summary.json").read_bytes()).hexdigest(),
        "artifact_hashes_sha256": hashlib.sha256(hash_manifest.read_bytes()).hexdigest(),
    }
    (output_dir / "RUN_COMPLETE.json").write_text(json.dumps(seal, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workload-manifest", required=True)
    parser.add_argument("--model", default="allenai/OLMoE-1B-7B-0924")
    parser.add_argument("--revision", default="6d84c48581ece794365f2b8e9cfb043c68ade9c5")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[4, 8, 16])
    parser.add_argument("--prompt-lengths", type=int, nargs="+", default=[128, 512])
    parser.add_argument("--output-tokens", type=int, default=16)
    parser.add_argument("--groups", type=int, default=6)
    parser.add_argument("--within-process-repeats", type=int, default=1)
    parser.add_argument("--process-repeat", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--order-seed", type=int, default=20260823)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    parser.add_argument(
        "--runtime-patch-id",
        default="stock-vllm-0.26.0",
        help="Provenance label for an experiment-only vLLM source patch.",
    )
    parser.add_argument("--capture-routes", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enforce-eager", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--require-exclusive-gpu", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()
    if max(args.prompt_lengths) + args.output_tokens > args.max_model_len:
        parser.error("max-model-len is smaller than prompt + output")
    if max(args.batch_sizes) > args.max_num_seqs:
        parser.error("max-num-seqs is smaller than requested batch size")
    if args.output_tokens < 3:
        parser.error("output-tokens must be >= 3 for temporal route metrics")
    if not args.runtime_patch_id.strip():
        parser.error("runtime-patch-id must be non-empty")
    return args


if __name__ == "__main__":
    run(parse_args())
