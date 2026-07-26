#!/usr/bin/env python3
"""Fast RouteShare Gate-0 on real model experts and one CUDA GPU.

Evidence boundary: real BF16 expert weights/shapes and CUDA execution, but
synthetic controlled route histograms. This is a mechanism-necessity screen,
not continuous-serving, tenant-fairness, EP-network, or production evidence.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import random
import re
import statistics
import sys
from typing import Any, Callable, Iterable, Mapping, Sequence

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from route_share_core import RoutePlan, build_plans, simple_features


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
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-key", choices=tuple(MODEL_SPECS), default="olmoe")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--rows", type=parse_ints, default=(32, 64, 128, 256))
    parser.add_argument("--active", type=parse_ints, default=(1, 2, 4, 8, 16))
    parser.add_argument("--replicas", type=int, default=3)
    parser.add_argument("--blocks", type=int, default=12)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def source_sha256() -> str:
    digest = hashlib.sha256()
    for path in (Path(__file__), HERE / "route_share_core.py"):
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def is_expert(module: Any, torch: Any) -> bool:
    triplets = (("gate_proj", "up_proj", "down_proj"), ("w1", "w3", "w2"))
    return any(
        all(isinstance(getattr(module, name, None), torch.nn.Linear) for name in names)
        for names in triplets
    )


def find_experts(model: Any, torch: Any, layer: int) -> dict[int, Any]:
    experts: dict[int, Any] = {}
    for name, module in model.named_modules():
        match = EXPERT_PATH.search(name)
        if match and int(match.group(1)) == layer and is_expert(module, torch):
            expert_id = int(match.group(2))
            if expert_id in experts:
                raise RuntimeError(f"duplicate expert id {expert_id} in layer {layer}")
            experts[expert_id] = module
    if not experts:
        raise RuntimeError(f"no auditable expert modules found in layer {layer}")
    return experts


def load_model(args: argparse.Namespace, spec: Mapping[str, Any], torch: Any) -> Any:
    from transformers import AutoModelForCausalLM

    source = str(args.model_path) if args.model_path else spec["model_id"]
    common = {"local_files_only": not args.allow_download}
    if args.model_path is None:
        common.update(
            revision=spec["revision"],
            cache_dir=str(args.cache_dir) if args.cache_dir else None,
        )
    model = AutoModelForCausalLM.from_pretrained(
        source, torch_dtype=torch.bfloat16, device_map="cuda:0", **common
    )
    model.eval()
    return model


def expert_hidden_size(expert: Any) -> int:
    for name in ("gate_proj", "w1"):
        layer = getattr(expert, name, None)
        if layer is not None and hasattr(layer, "in_features"):
            return int(layer.in_features)
    raise RuntimeError("cannot infer expert hidden size")


def prepare_operation(
    plan: RoutePlan,
    *,
    experts: Mapping[int, Any],
    activation_pool: Any,
    torch: Any,
    seed: int,
) -> Callable[[], Any]:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    permutation = torch.randperm(plan.total_rows, generator=generator, device="cuda")
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(plan.total_rows, device="cuda")
    source = activation_pool[: plan.total_rows]

    def operation() -> Any:
        # Routes are assumed known. This includes dispatch gather, every active
        # expert MLP, concatenation, and restore/combine permutation. Router,
        # argsort construction, attention, and KV are intentionally excluded.
        packed = torch.index_select(source, 0, permutation)
        outputs = []
        offset = 0
        for expert_id, count in zip(plan.active_experts, plan.counts):
            outputs.append(experts[expert_id](packed[offset : offset + count]))
            offset += count
        if offset != plan.total_rows:
            raise AssertionError("plan did not consume all rows")
        grouped = torch.cat(outputs, dim=0)
        return torch.index_select(grouped, 0, inverse)

    return operation


def time_cuda_us(torch: Any, operation: Callable[[], Any]) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    output = operation()
    end.record()
    end.synchronize()
    if tuple(output.shape[:1]) == (0,):
        raise AssertionError("empty measured output")
    value = float(start.elapsed_time(end)) * 1000.0
    if value <= 0:
        raise RuntimeError("non-positive CUDA timing")
    return value


def aggregate(rows: Sequence[Mapping[str, object]], blocks: Sequence[int]) -> dict[str, float]:
    selected = Counter(blocks)
    values: dict[str, list[float]] = {}
    for row in rows:
        multiplicity = selected.get(int(row["block"]), 0)
        if multiplicity:
            values.setdefault(str(row["plan_id"]), []).extend(
                [float(row["latency_us"])] * multiplicity
            )
    return {key: statistics.mean(sample) for key, sample in values.items()}


def fit_and_score(
    torch: Any,
    plans: Sequence[RoutePlan],
    means: Mapping[str, float],
    num_experts: int,
    *,
    strong: bool,
) -> float:
    train = [plan for plan in plans if plan.replica < max(p.replica for p in plans)]
    test = [plan for plan in plans if plan.replica == max(p.replica for p in plans)]

    def features(plan: RoutePlan) -> tuple[float, ...]:
        all_features = simple_features(plan, num_experts)
        return all_features if strong else all_features[:2]

    x_train = torch.tensor([features(plan) for plan in train], dtype=torch.float64)
    y_train = torch.tensor([means[plan.plan_id] for plan in train], dtype=torch.float64)
    x_test = torch.tensor([features(plan) for plan in test], dtype=torch.float64)
    y_test = torch.tensor([means[plan.plan_id] for plan in test], dtype=torch.float64)
    coefficient = torch.linalg.lstsq(x_train, y_train).solution
    prediction = x_test @ coefficient
    residual = float(((y_test - prediction) ** 2).sum().item())
    centered = float(((y_test - y_test.mean()) ** 2).sum().item())
    return 1.0 - residual / centered if centered > 0 else float("nan")


def shape_range_ratio(plans: Sequence[RoutePlan], means: Mapping[str, float]) -> float:
    cells: dict[tuple[int, int, int], dict[tuple[int, ...], list[float]]] = {}
    for plan in plans:
        cell = cells.setdefault((plan.total_rows, plan.active_count, plan.replica), {})
        cell.setdefault(plan.counts, []).append(means[plan.plan_id])
    ratios = []
    for histograms in cells.values():
        # Some named shapes collapse to the same integer row histogram (always
        # true for active_count=1). They are repeated measurements, not a route
        # contrast, and must not dilute the matched effect.
        values = [statistics.mean(sample) for sample in histograms.values()]
        if len(values) >= 2 and statistics.median(values) > 0:
            ratios.append((max(values) - min(values)) / statistics.median(values))
    if not ratios:
        raise RuntimeError("no matched cells contain two distinct histograms")
    return statistics.median(ratios)


def percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    fraction = position - low
    return ordered[low] * (1.0 - fraction) + ordered[high] * fraction


def bootstrap_metrics(
    torch: Any,
    rows: Sequence[Mapping[str, object]],
    plans: Sequence[RoutePlan],
    num_experts: int,
    blocks: int,
    samples: int,
    seed: int,
) -> dict[str, list[float]]:
    rng = random.Random(seed)
    result = {"row_r2": [], "strong_r2": [], "shape_range_ratio": []}
    for _ in range(samples):
        selected = [rng.randrange(blocks) for _ in range(blocks)]
        means = aggregate(rows, selected)
        result["row_r2"].append(fit_and_score(torch, plans, means, num_experts, strong=False))
        result["strong_r2"].append(fit_and_score(torch, plans, means, num_experts, strong=True))
        result["shape_range_ratio"].append(shape_range_ratio(plans, means))
    return result


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU/proxy timing is forbidden")
    args = parse_args()
    if args.replicas < 3 or args.blocks < 5 or args.warmups < 1 or args.bootstrap < 100:
        raise ValueError("minimums: replicas=3, blocks=5, warmups=1, bootstrap=100")
    if args.output_dir.exists():
        raise RuntimeError("refusing to overwrite output directory")
    spec = MODEL_SPECS[args.model_key]
    model = load_model(args, spec, torch)
    experts = find_experts(model, torch, args.layer)
    if len(experts) != int(spec["num_experts"]):
        raise RuntimeError(f"expected {spec['num_experts']} experts, found {len(experts)}")
    plans = build_plans(
        num_experts=len(experts),
        row_grid=args.rows,
        active_grid=args.active,
        shapes=("uniform", "linear_skew", "zipf", "hot50"),
        replicas=args.replicas,
        seed=args.seed,
    )
    hidden = expert_hidden_size(experts[min(experts)])
    generator = torch.Generator(device="cuda")
    generator.manual_seed(args.seed + 1)
    activation_pool = torch.randn(
        max(args.rows), hidden, generator=generator, device="cuda", dtype=torch.bfloat16
    )
    operations = {
        plan.plan_id: prepare_operation(
            plan,
            experts=experts,
            activation_pool=activation_pool,
            torch=torch,
            # Same dispatch permutation within a matched
            # (rows, active-count, replica) cell; only row histogram changes.
            seed=(
                args.seed
                + plan.total_rows * 1009
                + plan.active_count * 97
                + plan.replica * 7
            ),
        )
        for plan in plans
    }
    with torch.inference_mode():
        for plan in plans:
            for _ in range(args.warmups):
                operations[plan.plan_id]()
        torch.cuda.synchronize()
        rows_out: list[dict[str, object]] = []
        rng = random.Random(args.seed + 3)
        for block in range(args.blocks):
            order = list(plans)
            rng.shuffle(order)
            for ordinal, plan in enumerate(order):
                latency = time_cuda_us(torch, operations[plan.plan_id])
                rows_out.append(
                    {
                        "block": block,
                        "ordinal": ordinal,
                        "plan_id": plan.plan_id,
                        "total_rows": plan.total_rows,
                        "active_count": plan.active_count,
                        "shape": plan.shape,
                        "replica": plan.replica,
                        "max_fraction": plan.max_fraction,
                        "cv": plan.cv,
                        "expert_ids": ";".join(map(str, plan.active_experts)),
                        "counts": ";".join(map(str, plan.counts)),
                        "latency_us": latency,
                    }
                )

    all_blocks = list(range(args.blocks))
    means = aggregate(rows_out, all_blocks)
    point = {
        "row_r2": fit_and_score(torch, plans, means, len(experts), strong=False),
        "strong_r2": fit_and_score(torch, plans, means, len(experts), strong=True),
        "shape_range_ratio": shape_range_ratio(plans, means),
    }
    boot = bootstrap_metrics(
        torch, rows_out, plans, len(experts), args.blocks, args.bootstrap, args.seed + 4
    )
    intervals = {
        name: {"p025": percentile(values, 0.025), "p975": percentile(values, 0.975)}
        for name, values in boot.items()
    }
    # Conservative proceed gate: unexplained cost remains AND controlled
    # histogram effects are at least 10% at their lower confidence bound.
    proceed = intervals["strong_r2"]["p975"] < 0.90 and intervals["shape_range_ratio"]["p025"] >= 0.10
    verdict = "PROCEED_TO_ROUTE_REAL_SCHEDULER_GATE" if proceed else "KILL_OR_REFORMULATE_ROUTESHARE"
    summary = {
        "verdict": verdict,
        "scientific_result": True,
        "evidence_boundary": "REAL_MODEL_EXPERTS_SYNTHETIC_CONTROLLED_ROUTES_SINGLE_GPU_NOT_SERVING_NOT_NETWORK",
        "decision_rule": "proceed iff strong_r2_95pct_UCB < 0.90 and shape_range_ratio_95pct_LCB >= 0.10",
        "point": point,
        "intervals": intervals,
        "model": f"{spec['model_id']}@{spec['revision']}",
        "model_load_path": str(args.model_path) if args.model_path else None,
        "layer": args.layer,
        "gpu_name": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "source_sha256": source_sha256(),
        "config": {
            "rows": list(args.rows),
            "active": list(args.active),
            "replicas": args.replicas,
            "blocks": args.blocks,
            "warmups": args.warmups,
            "bootstrap": args.bootstrap,
            "seed": args.seed,
        },
        "scope_note": "A positive result authorizes route-real scheduling validation only; it is not a fairness or topology claim.",
    }
    args.output_dir.mkdir(parents=True)
    write_csv(args.output_dir / "timings.csv", rows_out)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
