#!/usr/bin/env python3
"""Paired small-sample check for native router-output telemetry overhead.

This is not a model benchmark.  It replays the same fixed request sample,
arrival trace, seed, decode steps, and frozen maximum batch width with
``output_router_logits`` OFF and ON.  It checks token/logit parity and reports
both CUDA-synchronized model-call time and loop wall time including route
extraction.  No Python forward hook is installed by the producer.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Sequence


class ProtocolError(RuntimeError):
    pass


HERE = Path(__file__).resolve().parent
PRODUCER_PATH = HERE.parents[2] / "bcrd" / "experiments" / "capture_continuous_decode.py"


def _load_producer():
    name = "rce_continuous_decode_producer"
    producer_dir = PRODUCER_PATH.parent
    sys.path.insert(0, str(producer_dir))
    spec = importlib.util.spec_from_file_location(name, PRODUCER_PATH)
    if spec is None or spec.loader is None:
        raise ProtocolError(f"cannot import producer: {PRODUCER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


PRODUCER = _load_producer()


def _timed_model_call(model: Any, *, output_router_logits: bool, **kwargs: Any):
    """Mirror the producer's CUDA-synchronized whole-model timing boundary."""

    PRODUCER._sync_model(model)
    start_ns = time.perf_counter_ns()
    base_model = getattr(model, "model", None)
    lm_head = getattr(model, "lm_head", None)
    if base_model is None or lm_head is None:
        raise ProtocolError("model does not expose the producer's base-model/lm-head boundary")
    base_output = base_model(
        return_dict=True,
        output_router_logits=output_router_logits,
        **kwargs,
    )
    logits = lm_head(base_output.last_hidden_state)
    PRODUCER._sync_model(model)
    elapsed_us = max((time.perf_counter_ns() - start_ns) / 1000.0, 1e-3)
    return base_output, logits, elapsed_us


def _new_states(model: Any, requests: Sequence[Any], *, router_on: bool):
    import torch

    states: list[Any] = []
    with torch.inference_mode():
        for request in requests:
            output, logits, _ = _timed_model_call(
                model,
                output_router_logits=router_on,
                input_ids=request.input_ids,
                attention_mask=request.attention_mask,
                use_cache=True,
            )
            cache = getattr(output, "past_key_values", None)
            if cache is None:
                raise ProtocolError("prefill returned no cache")
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            states.append(
                PRODUCER._ActiveRequest(
                    spec=request,
                    cache=cache,
                    attention_mask=request.attention_mask,
                    next_token=next_token,
                    prompt_length=int(request.input_ids.shape[1]),
                )
            )
    return states


def run_mode(
    model: Any,
    requests: Sequence[Any],
    *,
    router_on: bool,
    decode_steps: int,
    max_batch_size: int,
) -> dict[str, Any]:
    import torch

    states = _new_states(model, requests, router_on=router_on)
    model_call_ms: list[float] = []
    loop_wall_ms: list[float] = []
    token_trace: list[list[int]] = []
    logit_trace: list[str] = []
    route_trace: list[str] = []
    route_layers: list[int] = []
    for _ in range(decode_steps):
        for offset in range(0, len(states), max_batch_size):
            scheduled = states[offset : offset + max_batch_size]
            (
                input_ids,
                attention_mask,
                position_ids,
                cache,
                prior_lengths,
                prior_max,
            ) = PRODUCER._pad_decode_inputs(scheduled)
            PRODUCER._sync_model(model)
            wall_start = time.perf_counter_ns()
            with torch.inference_mode():
                output, logits, elapsed_us = _timed_model_call(
                    model,
                    output_router_logits=router_on,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    cache_position=torch.tensor(
                        [prior_max], dtype=torch.long, device=input_ids.device
                    ),
                    past_key_values=cache,
                    use_cache=True,
                )
                if router_on:
                    batches = PRODUCER._native_route_batches(
                        output,
                        expected_rows=len(scheduled),
                        config=getattr(model, "config"),
                    )
                    route_layers.append(len(batches))
                    route_digest = hashlib.sha256()
                    for route_batch in batches:
                        route_digest.update(
                            route_batch["selected_experts"]
                            .detach()
                            .contiguous()
                            .cpu()
                            .numpy()
                            .tobytes()
                        )
                        route_digest.update(
                            route_batch["routing_weights"]
                            .detach()
                            .contiguous()
                            .view(torch.uint8)
                            .cpu()
                            .numpy()
                            .tobytes()
                        )
                    route_trace.append(route_digest.hexdigest())
                elif getattr(output, "router_logits", None) not in (None, (), []):
                    raise ProtocolError("router OFF unexpectedly returned router logits")
            PRODUCER._sync_model(model)
            wall_us = (time.perf_counter_ns() - wall_start) / 1000.0
            output_cache = getattr(output, "past_key_values", None)
            if output_cache is None:
                raise ProtocolError("decode returned no cache")
            split = PRODUCER.split_left_padded_cache(
                output_cache,
                prior_lengths=prior_lengths,
                prior_max_length=prior_max,
            )
            predicted = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            logit_trace.append(
                hashlib.sha256(
                    logits[:, -1, :]
                    .detach()
                    .contiguous()
                    .view(torch.uint8)
                    .cpu()
                    .numpy()
                    .tobytes()
                ).hexdigest()
            )
            token_trace.append([int(value) for value in predicted.flatten().tolist()])
            for index, state in enumerate(scheduled):
                state.cache = split[index]
                state.attention_mask = torch.cat(
                    (state.attention_mask, state.attention_mask.new_ones((1, 1))), dim=1
                )
                state.next_token = predicted[index : index + 1]
                state.decode_step += 1
            model_call_ms.append(elapsed_us / 1000.0)
            loop_wall_ms.append(wall_us / 1000.0)
    return {
        "model_call_ms": model_call_ms,
        "loop_wall_ms": loop_wall_ms,
        "token_trace": token_trace,
        "logit_trace": logit_trace,
        "route_trace": route_trace,
        "route_layers": route_layers,
    }


def _relative_overhead(off: float, on: float) -> float:
    if off <= 0:
        raise ProtocolError("OFF timing is not positive")
    return (on - off) / off


def measure(args: argparse.Namespace) -> dict[str, Any]:
    try:
        import torch
        import transformers
    except ImportError as exc:
        raise ProtocolError("telemetry check requires PyTorch and Transformers") from exc
    if not torch.cuda.is_available():
        raise ProtocolError("telemetry check requires CUDA")
    manifest = PRODUCER.load_workload_manifest(Path(args.workload_manifest).resolve())
    if manifest.get("run_class") != "development":
        raise ProtocolError("telemetry check accepts only a development workload")
    seed = int(manifest["seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    repo_root = next(parent for parent in HERE.parents if (parent / ".git").exists())
    sys.path.insert(0, str(repo_root / "experiments" / "shared"))
    from modeling import load_model, load_tokenizer

    model_spec = manifest["model"]
    tokenizer = load_tokenizer(
        str(model_spec["id"]),
        local_files_only=args.offline,
        revision=str(model_spec["tokenizer_revision"]),
    )
    model, load_seconds = load_model(
        str(model_spec["id"]),
        dtype_name=str(model_spec["dtype"]),
        local_files_only=args.offline,
        revision=str(model_spec["revision"]),
    )
    model.eval()
    all_requests = PRODUCER._prepare_requests(manifest, tokenizer, model.device)
    requests = sorted(
        all_requests,
        key=lambda request: (request.arrival_us, request.request_id),
    )[: args.requests]
    if len(requests) != args.requests:
        raise ProtocolError("workload does not contain the requested overhead sample")

    max_batch_size = int(manifest["scheduler"]["max_batch_size"])
    if max_batch_size <= 0:
        raise ProtocolError("workload max_batch_size must be positive")
    recorded: dict[str, list[dict[str, Any]]] = {"off": [], "on": []}
    total_rounds = args.warmup_repeats + args.repeats
    for round_index in range(total_rounds):
        order = (False, True) if round_index % 2 == 0 else (True, False)
        for router_on in order:
            result = run_mode(
                model,
                requests,
                router_on=router_on,
                decode_steps=args.decode_steps,
                max_batch_size=max_batch_size,
            )
            if round_index >= args.warmup_repeats:
                recorded["on" if router_on else "off"].append(result)

    off_tokens = [row["token_trace"] for row in recorded["off"]]
    on_tokens = [row["token_trace"] for row in recorded["on"]]
    token_match = bool(off_tokens and on_tokens and off_tokens == on_tokens)
    off_logits = [row["logit_trace"] for row in recorded["off"]]
    on_logits = [row["logit_trace"] for row in recorded["on"]]
    logit_match = bool(off_logits and on_logits and off_logits == on_logits)
    on_routes = [row["route_trace"] for row in recorded["on"]]
    route_stable = bool(
        on_routes and all(value == on_routes[0] for value in on_routes)
    )
    off_model = [value for row in recorded["off"] for value in row["model_call_ms"]]
    on_model = [value for row in recorded["on"] for value in row["model_call_ms"]]
    off_wall = [value for row in recorded["off"] for value in row["loop_wall_ms"]]
    on_wall = [value for row in recorded["on"] for value in row["loop_wall_ms"]]
    medians = {
        "off_model_call_ms": statistics.median(off_model),
        "on_model_call_ms": statistics.median(on_model),
        "off_loop_wall_ms": statistics.median(off_wall),
        "on_loop_wall_ms": statistics.median(on_wall),
    }
    model_overhead = _relative_overhead(
        medians["off_model_call_ms"], medians["on_model_call_ms"]
    )
    wall_overhead = _relative_overhead(
        medians["off_loop_wall_ms"], medians["on_loop_wall_ms"]
    )
    threshold = float(args.max_relative_overhead)
    status = (
        "TELEMETRY_OVERHEAD_OK"
        if token_match
        and logit_match
        and route_stable
        and model_overhead <= threshold
        and wall_overhead <= threshold
        else "BLOCKED_HOOK_DISTORTION"
    )
    return {
        "schema": "route-capacity-envelope-telemetry-overhead-v1",
        "status": status,
        "model": {
            "id": model_spec["id"],
            "revision": model_spec["revision"],
            "dtype": model_spec["dtype"],
        },
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "model_load_seconds": load_seconds,
        "request_ids": [request.request_id for request in requests],
        "same_requests": True,
        "same_arrival_trace": False,
        "same_seed": True,
        "same_batch_schedule": True,
        "same_decode_steps": True,
        "same_dtype": True,
        "same_timing_boundary": "cuda_synchronized_whole_model_call",
        "route_hook_installed": False,
        "telemetry_mechanism": "native output_router_logits plus post-call extraction",
        "sample_request_count": args.requests,
        "max_batch_size": max_batch_size,
        "scheduled_batch_sizes": [
            min(max_batch_size, args.requests - offset)
            for offset in range(0, args.requests, max_batch_size)
        ],
        "arrival_policy_applied_to_timing": False,
        "arrival_scope": (
            "fixed preselected batches only; manifest arrival timing is not replayed"
        ),
        "decode_steps": args.decode_steps,
        "measured_repeats": args.repeats,
        "token_output_match": token_match,
        "logit_output_match": logit_match,
        "on_route_trace_stable": route_stable,
        "completion_trace_match": token_match,
        "median": medians,
        "model_call_relative_overhead": model_overhead,
        "loop_wall_relative_overhead": wall_overhead,
        "max_relative_overhead": threshold,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--requests", type=int, default=8)
    parser.add_argument("--decode-steps", type=int, default=8)
    parser.add_argument("--warmup-repeats", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-relative-overhead", type=float, default=0.02)
    parser.add_argument("--offline", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (
        args.requests <= 0
        or args.decode_steps <= 0
        or args.repeats <= 0
        or args.warmup_repeats < 0
        or args.max_relative_overhead < 0
    ):
        raise SystemExit("invalid positive overhead-check scale")
    output = Path(args.output).resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    result = measure(args)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": result["status"]}, sort_keys=True))
    if result["status"] != "TELEMETRY_OVERHEAD_OK":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
