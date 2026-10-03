#!/usr/bin/env python3
"""Quick native-FP8 route-row crossover screen on one real MoE expert.

This measures a real model expert and the exact DualResidentExpertMLP FP8 path,
but uses controlled random BF16 activations. It is a mechanism screen, not a
continuous-serving, route-mass, end-to-end quality, or energy result.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random
import re
import statistics
from typing import Any, Mapping, Sequence

import torch

from route_row_policy import (
    ACTION_BF16,
    ACTION_FP8,
    DualResidentExpertMLP,
    RuntimeCounters,
    _fp8_quantize_per_tensor,
    require_cuda_fp8,
)


MODEL_SPECS = {
    "olmoe": {
        "model_id": "allenai/OLMoE-1B-7B-0924",
        "revision": "6d84c48581ece794365f2b8e9cfb043c68ade9c5",
        "num_experts": 64,
    },
    "llmjp": {
        "model_id": "llm-jp/optimal-sparsity-math-d512-E32-k16-920M-A520M",
        "revision": "1d5983076dfc67aee4a77ec06a27027f5bab6055",
        "num_experts": 32,
    },
}
EXPERT_PATH = re.compile(r"(?:^|\.)layers\.(\d+)\..*experts\.(\d+)$")


def parse_ints(value: str) -> tuple[int, ...]:
    result = tuple(int(item) for item in value.split(",") if item.strip())
    if not result or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError("expected positive comma-separated integers")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-key", choices=tuple(MODEL_SPECS), required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--expert", type=int, default=0)
    parser.add_argument("--rows", type=parse_ints, default=(1, 2, 4, 8, 16, 32, 64, 128, 256))
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--blocks", type=int, default=30)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fp8-mode", choices=("three_quant", "shared_gate_up_quant"), default="three_quant")
    parser.add_argument("--compile-fp8", action="store_true")
    return parser.parse_args()


def find_expert(model: Any, layer: int, expert_id: int) -> Any:
    found = []
    for name, module in model.named_modules():
        match = EXPERT_PATH.search(name)
        if match and (int(match.group(1)), int(match.group(2))) == (layer, expert_id):
            if all(isinstance(getattr(module, key, None), torch.nn.Linear) for key in ("gate_proj", "up_proj", "down_proj")) or all(
                isinstance(getattr(module, key, None), torch.nn.Linear) for key in ("w1", "w3", "w2")
            ):
                found.append(module)
    if len(found) != 1:
        raise RuntimeError(f"expected one expert ({layer},{expert_id}), found {len(found)}")
    return found[0]


def time_us(operation: Any) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    output = operation()
    end.record()
    end.synchronize()
    if not isinstance(output, torch.Tensor) or output.numel() == 0:
        raise RuntimeError("measured expert returned invalid output")
    value = float(start.elapsed_time(end)) * 1000.0
    if value <= 0:
        raise RuntimeError("non-positive CUDA timing")
    return value


def percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    fraction = position - low
    return ordered[low] * (1 - fraction) + ordered[high] * fraction


def bootstrap_speedup(pairs: Sequence[tuple[float, float]], samples: int, seed: int) -> tuple[float, float, float]:
    def metric(selected: Sequence[tuple[float, float]]) -> float:
        bf16 = statistics.mean(item[0] for item in selected)
        fp8 = statistics.mean(item[1] for item in selected)
        return (bf16 - fp8) / bf16

    point = metric(pairs)
    rng = random.Random(seed)
    boot = [metric([pairs[rng.randrange(len(pairs))] for _ in pairs]) for _ in range(samples)]
    return point, percentile(boot, 0.025), percentile(boot, 0.975)


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.blocks < 10 or args.bootstrap < 100 or args.warmups < 1:
        raise ValueError("minimums: blocks=10, bootstrap=100, warmups=1")
    if args.output_dir.exists():
        raise RuntimeError("refusing to overwrite output directory")
    device = require_cuda_fp8("cuda:0", probe_kernel=True)
    spec = MODEL_SPECS[args.model_key]
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        str(args.model_path), torch_dtype=torch.bfloat16, device_map="cuda:0", local_files_only=True
    ).eval()
    original = find_expert(model, args.layer, args.expert)
    counters = RuntimeCounters()
    expert = DualResidentExpertMLP(original, counters).eval()
    if counters.weight_casts != 3:
        raise AssertionError("one expert must create exactly three resident FP8 weights")
    hidden = expert.input_features
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)
    inputs = {
        rows: torch.randn(rows, hidden, generator=generator, device=device, dtype=torch.bfloat16)
        for rows in args.rows
    }
    raw: list[dict[str, object]] = []
    rng = random.Random(args.seed + 1)

    def fp8_forward_impl(value: torch.Tensor) -> torch.Tensor:
        if args.fp8_mode == "three_quant":
            return expert(value, ACTION_FP8)
        # Gate and up consume the same expert input. Quantize it once and reuse
        # the exact FP8 tensor/scale for both GEMMs; down still quantizes the
        # post-activation intermediate. This is implementable without assuming
        # free/pre-existing FP8 activations.
        quantized, scale = _fp8_quantize_per_tensor(value)
        gate = torch._scaled_mm(
            quantized,
            expert.gate.weight_fp8_t,
            scale_a=scale,
            scale_b=expert.gate.weight_scale,
            out_dtype=torch.bfloat16,
            use_fast_accum=True,
        )
        up = torch._scaled_mm(
            quantized,
            expert.up.weight_fp8_t,
            scale_a=scale,
            scale_b=expert.up.weight_scale,
            out_dtype=torch.bfloat16,
            use_fast_accum=True,
        )
        intermediate = expert.original_expert.act_fn(gate) * up
        down_quantized, down_scale = _fp8_quantize_per_tensor(intermediate)
        return torch._scaled_mm(
            down_quantized,
            expert.down.weight_fp8_t,
            scale_a=down_scale,
            scale_b=expert.down.weight_scale,
            out_dtype=torch.bfloat16,
            use_fast_accum=True,
        )

    fp8_forward = (
        torch.compile(fp8_forward_impl, fullgraph=True, dynamic=True)
        if args.compile_fp8
        else fp8_forward_impl
    )

    with torch.inference_mode():
        for rows in args.rows:
            for _ in range(args.warmups):
                expert(inputs[rows], ACTION_BF16)
                fp8_forward(inputs[rows])
        torch.cuda.synchronize()
        for block in range(args.blocks):
            row_order = list(args.rows)
            rng.shuffle(row_order)
            for rows in row_order:
                arm_order = [ACTION_BF16, ACTION_FP8]
                if (block + rows) % 2:
                    arm_order.reverse()
                measured: dict[str, float] = {}
                for action in arm_order:
                    measured[action] = time_us(
                        lambda action=action, rows=rows: (
                            expert(inputs[rows], ACTION_BF16)
                            if action == ACTION_BF16
                            else fp8_forward(inputs[rows])
                        )
                    )
                raw.append(
                    {
                        "block": block,
                        "rows": rows,
                        "first_arm": arm_order[0],
                        "bf16_us": measured[ACTION_BF16],
                        "fp8_us": measured[ACTION_FP8],
                    }
                )
        quality = {}
        for rows in args.rows:
            bf16 = expert(inputs[rows], ACTION_BF16).float()
            fp8 = fp8_forward(inputs[rows]).float()
            quality[str(rows)] = float(((fp8 - bf16).square().sum() / bf16.square().sum().clamp_min(1e-12)).item())

    cells = []
    for rows in args.rows:
        pairs = [(float(item["bf16_us"]), float(item["fp8_us"])) for item in raw if int(item["rows"]) == rows]
        point, low, high = bootstrap_speedup(pairs, args.bootstrap, args.seed + rows)
        cells.append({"rows": rows, "speedup": point, "ci95_low": low, "ci95_high": high, "relative_mse": quality[str(rows)]})
    bf16_regions = [cell for cell in cells if float(cell["ci95_high"]) < 0.0]
    fp8_regions = [cell for cell in cells if float(cell["ci95_low"]) > 0.05]
    if bf16_regions and fp8_regions:
        verdict = "CROSSOVER_EXISTS_PROCEED_TO_ROUTE_MASS"
    elif fp8_regions and not bf16_regions:
        verdict = "NO_DYNAMIC_NEED_FP8_DOMINATES_TESTED_GRID"
    else:
        verdict = "NO_GO_FP8_HAS_NO_ROBUST_FAST_REGION"
    summary = {
        "verdict": verdict,
        "evidence_boundary": "REAL_EXPERT_NATIVE_FP8_CONTROLLED_ACTIVATIONS_NOT_SERVING_NOT_ENERGY_NOT_END_TO_END_QUALITY",
        "model": f"{spec['model_id']}@{spec['revision']}",
        "model_path": str(args.model_path),
        "layer": args.layer,
        "expert": args.expert,
        "gpu": torch.cuda.get_device_name(device),
        "fp8_mode": args.fp8_mode,
        "compile_fp8": args.compile_fp8,
        "cells": cells,
        "counters": counters.snapshot(),
        "decision_rule": "crossover iff some row CI high < 0 and another row speedup CI low > 0.05",
    }
    args.output_dir.mkdir(parents=True)
    write_csv(args.output_dir / "paired_timings.csv", raw)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
