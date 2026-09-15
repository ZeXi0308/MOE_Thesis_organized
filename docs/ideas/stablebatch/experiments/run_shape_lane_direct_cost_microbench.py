#!/usr/bin/env python3
"""Measure only the direct GPU cost of one sparse fixed-C8 expert action.

This is a post-selector cost microbenchmark, not a serving or quality test.  It
reconstructs two existing calibration hidden rows that route to the same
layer/expert, then times native, singleton-C8, split-and-pad, and two-row
coalesced-C8 expert execution on those exact rows.  It does not read any fresh
selector outcome.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any, Callable


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] + (position - low) * (ordered[high] - ordered[low])


def tensor_sha256(tensor: Any) -> str:
    # BF16 has no portable NumPy dtype; hash the underlying uint16 storage.
    import torch

    data = tensor.detach().contiguous().view(torch.uint16).cpu().numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def capture_hidden(model: Any, cell: dict[str, Any]) -> Any:
    import torch

    layer = int(cell["layer"])
    token_index = int(cell["flat_token_idx"])
    captured: list[Any] = []

    def hook(_module: Any, inputs: tuple[Any, ...]) -> None:
        hidden = inputs[0]
        flat = hidden.reshape(-1, hidden.shape[-1])
        captured.append(flat[token_index].detach().clone())

    mlp = model.model.layers[layer].mlp
    handle = mlp.register_forward_pre_hook(hook)
    try:
        input_ids = torch.tensor(
            [cell["window_token_ids"]], dtype=torch.long, device="cuda"
        )
        with torch.inference_mode():
            model(input_ids=input_ids, use_cache=False, return_dict=True)
    finally:
        handle.remove()
    if len(captured) != 1:
        raise RuntimeError(f"captured {len(captured)} hidden rows for {cell['cell_id']}")
    focal = captured[0]
    router_logits = mlp.gate(focal[None, :])
    selected = torch.topk(torch.softmax(router_logits, dim=-1), mlp.top_k, dim=-1).indices[0]
    rank = int(cell["frozen_m1_rank"])
    expected_expert = int(cell["expert_ids"][rank])
    if int(selected[rank].item()) != expected_expert:
        raise RuntimeError(f"route identity changed for {cell['cell_id']}")
    return focal


def timed_samples(
    arms: dict[str, Callable[[], Any]], warmup: int, repeats: int
) -> dict[str, list[float]]:
    import torch

    names = list(arms)
    with torch.inference_mode():
        for _ in range(warmup):
            for name in names:
                arms[name]()
        torch.cuda.synchronize()
        samples = {name: [] for name in names}
        for repeat in range(repeats):
            shift = repeat % len(names)
            for name in names[shift:] + names[:shift]:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                output = arms[name]()
                end.record()
                end.synchronize()
                if isinstance(output, tuple):
                    finite = all(bool(torch.isfinite(item).all().item()) for item in output)
                else:
                    finite = bool(torch.isfinite(output).all().item())
                if not finite:
                    raise RuntimeError(f"{name} produced NaN/Inf")
                samples[name].append(float(start.elapsed_time(end)))
    return samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--cell-ids", nargs=2, default=("cell-005", "cell-012"))
    parser.add_argument("--layer", type=int, default=7)
    parser.add_argument("--expert", type=int, default=43)
    parser.add_argument("--canonical-m", type=int, default=8)
    parser.add_argument("--focal-slot", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=1000)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.canonical_m != 8 or not 0 <= args.focal_slot < args.canonical_m:
        raise ValueError("this benchmark is frozen to a valid canonical C8 slot")

    import torch
    from transformers import AutoModelForCausalLM

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model_path),
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="eager",
    ).eval().to("cuda")

    by_id = {row["cell_id"]: row for row in read_jsonl(args.ledger)}
    cells = [by_id[cell_id] for cell_id in args.cell_ids]
    for cell in cells:
        rank = int(cell["frozen_m1_rank"])
        if int(cell["layer"]) != args.layer or int(cell["expert_ids"][rank]) != args.expert:
            raise ValueError(f"{cell['cell_id']} is not layer={args.layer}, expert={args.expert}")

    hidden = torch.stack([capture_hidden(model, cell) for cell in cells], dim=0)
    if hidden.dtype != torch.bfloat16 or int(hidden.shape[0]) != 2:
        raise RuntimeError(f"unexpected hidden shape/dtype: {hidden.shape}, {hidden.dtype}")
    expert = model.model.layers[args.layer].mlp.experts[args.expert]

    singleton_lane = hidden.new_zeros((args.canonical_m, hidden.shape[1]))
    singleton_lane[args.focal_slot] = hidden[0]
    coalesced_lane = singleton_lane.clone()
    companion_slot = 0 if args.focal_slot != 0 else 1
    coalesced_lane[companion_slot] = hidden[1]

    arms: dict[str, Callable[[], Any]] = {
        # One natural expert group containing both compatible rows.
        "default_native_m2": lambda: expert(hidden),
        # The isolated protected expert call used by the offline action contract.
        "fixed_c8_direct_singleton": lambda: expert(singleton_lane),
        # Full sparse split: native remainder M1 plus one padded C8 lane.
        "fixed_c8_padded_split": lambda: (expert(hidden[1:2]), expert(singleton_lane)),
        # Both compatible rows protected in one C8 lane; six dummy rows.
        "fixed_c8_coalesced_2": lambda: expert(coalesced_lane),
    }
    samples = timed_samples(arms, args.warmup, args.repeats)
    useful_protected = {
        "default_native_m2": 0,
        "fixed_c8_direct_singleton": 1,
        "fixed_c8_padded_split": 1,
        "fixed_c8_coalesced_2": 2,
    }
    logical_calls = {
        "default_native_m2": 1,
        "fixed_c8_direct_singleton": 1,
        "fixed_c8_padded_split": 2,
        "fixed_c8_coalesced_2": 1,
    }
    metrics: dict[str, Any] = {}
    for name, values in samples.items():
        median_ms = statistics.median(values)
        metrics[name] = {
            "samples": len(values),
            "median_gpu_ms": median_ms,
            "p10_gpu_ms": percentile(values, 0.10),
            "p90_gpu_ms": percentile(values, 0.90),
            "p99_gpu_ms": percentile(values, 0.99),
            "logical_expert_calls": logical_calls[name],
            "useful_protected_rows": useful_protected[name],
            "median_gpu_ms_per_protected_row": (
                median_ms / useful_protected[name] if useful_protected[name] else None
            ),
        }
    baseline = metrics["default_native_m2"]["median_gpu_ms"]
    for row in metrics.values():
        row["median_ratio_vs_default_native_m2"] = row["median_gpu_ms"] / baseline
    singleton_full = metrics["fixed_c8_padded_split"]["median_gpu_ms"]
    coalesced_full = metrics["fixed_c8_coalesced_2"]["median_gpu_ms"]
    derived_deltas = {
        "singleton_full_delta_ms_vs_native_m2": singleton_full - baseline,
        "coalesced_two_action_delta_ms_vs_native_m2": coalesced_full - baseline,
        "coalesced_delta_ms_per_protected_action": (coalesced_full - baseline) / 2.0,
        "singleton_to_coalesced_delta_amortization_ratio": (
            ((coalesced_full - baseline) / 2.0) / (singleton_full - baseline)
            if singleton_full != baseline
            else None
        ),
    }

    result = {
        "schema_version": "stablebatch-shape-lane-direct-cost-microbench-v1",
        "scope": "isolated_single_expert_direct_gpu_cost_not_serving_not_quality",
        "fresh_selector_outcomes_read": False,
        "model_path": str(args.model_path),
        "cell_ids": list(args.cell_ids),
        "layer": args.layer,
        "expert": args.expert,
        "canonical_m": args.canonical_m,
        "focal_slot": args.focal_slot,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "input_shape": list(hidden.shape),
        "input_dtype": str(hidden.dtype),
        "input_sha256": tensor_sha256(hidden),
        "metrics": metrics,
        "derived_deltas": derived_deltas,
        "interpretation": (
            "Calibrates only direct expert execution and split/coalescing deltas. "
            "It does not measure queueing, scheduler/controller cost, TTFT, TPOT, or P99."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
