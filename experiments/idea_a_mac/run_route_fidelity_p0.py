"""P0 for correlation-preserving MoE expert-parallel workload synthesis.

The script compares real routed-token hypergraphs with progressively weaker
synthetic abstractions, then lowers each workload into rank-deduplicated EP
records.  It measures whether the abstraction changes receiver-tail and buffer
provisioning decisions.  No latency or actual-wire claim is made.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


VARIANTS = ("real", "hyperedge_shuffle", "degree_shuffle", "uniform_unique")


@dataclass(frozen=True)
class RouteMatrix:
    experts: np.ndarray
    sample: np.ndarray
    layer: np.ndarray
    position: np.ndarray
    num_experts: int
    top_k: int


def load_routes(path: Path) -> RouteMatrix:
    data = pd.read_csv(
        path,
        usecols=["sample_id", "layer", "token_position", "rank", "expert_id"],
    ).sort_values(["sample_id", "layer", "token_position", "rank"])
    top_k = int(data["rank"].max())
    counts = data.groupby(["sample_id", "layer", "token_position"]).size()
    if not bool((counts == top_k).all()):
        raise ValueError("incomplete token routes")
    meta = data.drop_duplicates(["sample_id", "layer", "token_position"])
    experts = data["expert_id"].to_numpy(np.int16).reshape(-1, top_k)
    if np.any(np.sort(experts, axis=1)[:, 1:] == np.sort(experts, axis=1)[:, :-1]):
        raise ValueError("real top-k route contains duplicate experts")
    return RouteMatrix(
        experts=experts,
        sample=meta["sample_id"].to_numpy(np.int32),
        layer=meta["layer"].to_numpy(np.int16),
        position=meta["token_position"].to_numpy(np.int32),
        num_experts=int(data["expert_id"].max()) + 1,
        top_k=top_k,
    )


def _weighted_unique(
    rows: int, probabilities: np.ndarray, top_k: int, rng: np.random.Generator
) -> np.ndarray:
    """Vectorized weighted sampling without replacement via Gumbel top-k."""
    safe = np.maximum(probabilities.astype(np.float64), 1e-12)
    scores = np.log(safe)[None, :] + rng.gumbel(size=(rows, safe.size))
    chosen = np.argpartition(scores, -top_k, axis=1)[:, -top_k:]
    # Put the selected experts in score order to retain a deterministic rank.
    selected_scores = np.take_along_axis(scores, chosen, axis=1)
    order = np.argsort(-selected_scores, axis=1)
    return np.take_along_axis(chosen, order, axis=1).astype(np.int16)


def _degree_preserving_shuffle(
    layer_routes: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Randomize token hyperedges while preserving exact expert degrees.

    Starting from a valid simple bipartite token--expert graph, perform
    degree-preserving double-edge swaps.  Every accepted swap keeps both token
    degree (top-k) and layer-level expert occurrence counts unchanged, while
    progressively destroying the original expert co-activation structure.
    """
    out = layer_routes.copy()
    row_sets = [set(int(value) for value in row) for row in out]
    attempts = max(1, 4 * out.shape[0] * out.shape[1])
    accepted = 0
    for _ in range(attempts):
        first, second = rng.integers(0, out.shape[0], size=2)
        if first == second:
            continue
        first_rank = int(rng.integers(0, out.shape[1]))
        second_rank = int(rng.integers(0, out.shape[1]))
        first_expert = int(out[first, first_rank])
        second_expert = int(out[second, second_rank])
        if (
            first_expert == second_expert
            or second_expert in row_sets[first]
            or first_expert in row_sets[second]
        ):
            continue
        out[first, first_rank] = second_expert
        out[second, second_rank] = first_expert
        row_sets[first].remove(first_expert)
        row_sets[first].add(second_expert)
        row_sets[second].remove(second_expert)
        row_sets[second].add(first_expert)
        accepted += 1
    if accepted < out.shape[0]:
        raise RuntimeError("degree-preserving shuffle did not mix sufficiently")
    before = np.bincount(layer_routes.reshape(-1))
    after = np.bincount(out.reshape(-1), minlength=before.size)
    if not np.array_equal(before, after):
        raise AssertionError("degree-preserving shuffle changed expert counts")
    return out


def synthesize(routes: RouteMatrix, variant: str, seed: int) -> np.ndarray:
    if variant == "real":
        return routes.experts.copy()
    rng = np.random.default_rng(seed)
    out = np.empty_like(routes.experts)
    for layer_id in np.unique(routes.layer):
        idx = np.flatnonzero(routes.layer == layer_id)
        layer_routes = routes.experts[idx]
        if variant == "hyperedge_shuffle":
            out[idx] = layer_routes[rng.permutation(len(idx))]
        elif variant == "degree_shuffle":
            out[idx] = _degree_preserving_shuffle(layer_routes, rng)
        elif variant == "uniform_unique":
            probability = np.full(routes.num_experts, 1.0 / routes.num_experts)
            out[idx] = _weighted_unique(len(idx), probability, routes.top_k, rng)
        else:
            raise ValueError(variant)
    return out


def placements(num_experts: int, ep_size: int, seeds: int) -> dict[str, np.ndarray]:
    if num_experts % ep_size != 0:
        raise ValueError("P0 requires balanced expert-to-rank placement")
    per_rank = num_experts // ep_size
    result = {
        "contiguous": (np.arange(num_experts) // per_rank).astype(np.int16),
        "round_robin": (np.arange(num_experts) % ep_size).astype(np.int16),
    }
    for seed in range(seeds):
        rng = np.random.default_rng(7100 + seed)
        permutation = rng.permutation(num_experts)
        mapping = np.empty(num_experts, dtype=np.int16)
        mapping[permutation] = np.repeat(np.arange(ep_size), per_rank)
        result[f"random_{seed:03d}"] = mapping
    return result


def _rank_deduplicated_owners(experts: np.ndarray, mapping: np.ndarray):
    owners = np.sort(mapping[experts], axis=1)
    unique = np.ones_like(owners, dtype=bool)
    unique[:, 1:] = owners[:, 1:] != owners[:, :-1]
    return owners, unique


def lower_metrics(
    experts: np.ndarray,
    routes: RouteMatrix,
    mapping: np.ndarray,
    ep_size: int,
    batch_tokens: int,
) -> tuple[dict[str, float], np.ndarray]:
    owners, unique = _rank_deduplicated_owners(experts, mapping)
    fanout = unique.sum(axis=1)

    block = routes.position // batch_tokens
    keys = np.stack([routes.sample, routes.layer.astype(np.int32), block], axis=1)
    _, window_id = np.unique(keys, axis=0, return_inverse=True)
    repeated_window = np.broadcast_to(window_id[:, None], owners.shape)[unique]
    repeated_owner = owners[unique]
    encoded = repeated_window.astype(np.int64) * ep_size + repeated_owner
    counts = np.bincount(
        encoded, minlength=(int(window_id.max()) + 1) * ep_size
    ).reshape(-1, ep_size)
    receiver_max = counts.max(axis=1)
    metrics = {
        "mean_fanout": float(fanout.mean()),
        "p95_fanout": float(np.quantile(fanout, 0.95)),
        "physical_records": float(fanout.sum()),
        "receiver_max_mean": float(receiver_max.mean()),
        "receiver_max_p95": float(np.quantile(receiver_max, 0.95)),
        "receiver_max_p99": float(np.quantile(receiver_max, 0.99)),
        "window_count": int(receiver_max.size),
    }
    return metrics, receiver_max


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--route",
        action="append",
        nargs=2,
        metavar=("MODEL", "CSV"),
        required=True,
        help="repeat for every model",
    )
    parser.add_argument("--ep-sizes", default="8,16")
    parser.add_argument("--batch-tokens", default="8,32,128")
    parser.add_argument("--random-placements", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    ep_sizes = [int(value) for value in args.ep_sizes.split(",")]
    batch_sizes = [int(value) for value in args.batch_tokens.split(",")]
    rows: list[dict[str, object]] = []
    overflow_rows: list[dict[str, object]] = []
    decisions: list[dict[str, object]] = []

    manifests = []
    for model_name, route_name in args.route:
        route_path = Path(route_name)
        routes = load_routes(route_path)
        manifests.append(
            {
                "model": model_name,
                "path": str(route_path),
                "sha256": sha256(route_path.read_bytes()).hexdigest(),
                "tokens": int(routes.experts.shape[0]),
                "top_k": routes.top_k,
                "num_experts": routes.num_experts,
            }
        )
        variants = {
            variant: synthesize(routes, variant, args.seed + idx * 101)
            for idx, variant in enumerate(VARIANTS)
        }
        for ep_size in ep_sizes:
            if routes.num_experts % ep_size:
                continue
            mapping_set = placements(
                routes.num_experts, ep_size, args.random_placements
            )
            for batch_tokens in batch_sizes:
                receiver_by_key: dict[tuple[str, str], np.ndarray] = {}
                metric_by_key: dict[tuple[str, str], dict[str, float]] = {}
                for placement_name, mapping in mapping_set.items():
                    for variant, expert_matrix in variants.items():
                        metrics, receiver_max = lower_metrics(
                            expert_matrix,
                            routes,
                            mapping,
                            ep_size,
                            batch_tokens,
                        )
                        key = (placement_name, variant)
                        receiver_by_key[key] = receiver_max
                        metric_by_key[key] = metrics
                        rows.append(
                            {
                                "model": model_name,
                                "ep_size": ep_size,
                                "batch_tokens": batch_tokens,
                                "placement": placement_name,
                                "variant": variant,
                                **metrics,
                            }
                        )

                for placement_name in mapping_set:
                    real = receiver_by_key[(placement_name, "real")]
                    real_p99 = metric_by_key[(placement_name, "real")][
                        "receiver_max_p99"
                    ]
                    for variant in VARIANTS[1:]:
                        synthetic_p99 = metric_by_key[(placement_name, variant)][
                            "receiver_max_p99"
                        ]
                        capacity = int(math.ceil(synthetic_p99))
                        overflow_rows.append(
                            {
                                "model": model_name,
                                "ep_size": ep_size,
                                "batch_tokens": batch_tokens,
                                "placement": placement_name,
                                "variant": variant,
                                "real_p99": real_p99,
                                "synthetic_p99": synthetic_p99,
                                "relative_p99_error": (synthetic_p99 - real_p99)
                                / max(real_p99, 1.0),
                                "capacity_from_synthetic": capacity,
                                "real_overflow_rate": float((real > capacity).mean()),
                            }
                        )

                for objective in ("receiver_max_p99", "physical_records"):
                    candidates = list(mapping_set)
                    real_values = {
                        placement: metric_by_key[(placement, "real")][objective]
                        for placement in candidates
                    }
                    real_best = min(real_values, key=real_values.get)
                    for variant in VARIANTS[1:]:
                        selected = min(
                            candidates,
                            key=lambda placement: metric_by_key[
                                (placement, variant)
                            ][objective],
                        )
                        regret = real_values[selected] / max(
                            real_values[real_best], 1e-30
                        ) - 1.0
                        decisions.append(
                            {
                                "model": model_name,
                                "ep_size": ep_size,
                                "batch_tokens": batch_tokens,
                                "variant": variant,
                                "objective": objective,
                                "selected_placement": selected,
                                "real_best_placement": real_best,
                                "real_regret": regret,
                            }
                        )
        print(f"processed {model_name}", flush=True)

    metrics_df = pd.DataFrame(rows)
    overflow_df = pd.DataFrame(overflow_rows)
    decisions_df = pd.DataFrame(decisions)
    metrics_df.to_csv(output / "lowering_metrics.csv", index=False)
    overflow_df.to_csv(output / "buffer_provisioning.csv", index=False)
    decisions_df.to_csv(output / "placement_decisions.csv", index=False)
    (output / "manifest.json").write_text(
        json.dumps(manifests, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    under = overflow_df.assign(under=-overflow_df["relative_p99_error"])
    per_model = []
    per_variant = []
    for model, group in under.groupby("model"):
        model_decisions = decisions_df[decisions_df["model"] == model]
        per_model.append(
            {
                "model": model,
                "max_p99_underestimate": float(group["under"].max()),
                "max_real_overflow_rate": float(group["real_overflow_rate"].max()),
                "max_placement_regret": float(
                    model_decisions["real_regret"].max()
                ),
            }
        )
        for variant, variant_group in group.groupby("variant"):
            consequential = variant_group[variant_group["under"] >= 0.05]
            variant_decisions = model_decisions[model_decisions["variant"] == variant]
            per_variant.append(
                {
                    "model": model,
                    "variant": variant,
                    "max_p99_underestimate": float(variant_group["under"].max()),
                    "max_consequential_overflow_rate": float(
                        consequential["real_overflow_rate"].max()
                        if not consequential.empty
                        else 0.0
                    ),
                    "max_placement_regret": float(variant_decisions["real_regret"].max()),
                }
            )

    def variant_rows(name: str) -> list[dict[str, object]]:
        return [row for row in per_variant if row["variant"] == name]

    architecture_gap = all(
        row["max_p99_underestimate"] >= 0.10
        for row in variant_rows("uniform_unique")
    )
    exact_degree_gap = any(
        row["max_p99_underestimate"] >= 0.10
        for row in variant_rows("degree_shuffle")
    )
    temporal_gap = any(
        row["max_p99_underestimate"] >= 0.05
        for row in variant_rows("hyperedge_shuffle")
    )
    logical_capacity_gap = any(
        row["max_consequential_overflow_rate"] >= 0.05
        for row in per_variant
        if row["variant"] in ("degree_shuffle", "hyperedge_shuffle")
    )
    placement_gate = any(
        row["max_placement_regret"] >= 0.05 for row in per_variant
    )
    exploratory_signal = (
        architecture_gap
        and (exact_degree_gap or temporal_gap)
        and logical_capacity_gap
    )
    # This script scans a development grid and reports maxima.  It can identify
    # a signal worth sealing, but cannot legitimately issue a confirmatory PASS.
    verdict = "EXPLORATORY_SIGNAL" if exploratory_signal else "NO_SIGNAL"
    summary = {
        "verdict": verdict,
        "analysis_status": "development maximum scan; not a sealed hypothesis test",
        "gates": {
            "architecture_only_ge_10pct_gap_on_both_models": architecture_gap,
            "exact_degree_shuffle_ge_10pct_gap_on_any_model": exact_degree_gap,
            "temporal_shuffle_ge_5pct_gap_on_any_model": temporal_gap,
            "stronger_surrogate_causes_ge_5pct_capacity_overflow": logical_capacity_gap,
            "any_ge_5pct_placement_regret": placement_gate,
        },
        "per_model": per_model,
        "per_model_variant": per_variant,
        "evidence_boundary": (
            "route-hypergraph and logical rank-record analysis only; no backend "
            "latency, NIC bytes, TPOT, or P99 serving evidence"
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    table = "\n".join(
        f"| {row['model']} | `{row['variant']}` | "
        f"{row['max_p99_underestimate']:.1%} | "
        f"{row['max_consequential_overflow_rate']:.1%} | "
        f"{row['max_placement_regret']:.1%} |"
        for row in per_variant
    )
    report = f"""# RouteFidelity P0-A

> Boundary: route hypergraph and logical rank-deduplicated records only.  This
> is not an actual backend, wire, or latency experiment.

| model | surrogate | max receiver-P99 underestimate | hypothetical exceedance when underestimate >=5% | max placement regret |
|---|---|---:|---:|---:|
{table}

Verdict: **{verdict}**

Compared abstractions:

- `uniform_unique`: architecture-only B/E/k style synthesis;
- `degree_shuffle`: preserves exact layer-level expert occurrence counts while
  destroying token-level co-activation and ordering;
- `hyperedge_shuffle`: preserves every routed top-k set but destroys temporal bursts;
- `real`: preserves the complete token-expert hypergraph and ordering.

This is a development-grid maximum scan, so `EXPLORATORY_SIGNAL` is not a
confirmatory pass. The exceedance column applies a hypothetical zero-headroom
logical capacity and is neither a real buffer overflow nor an OOM/drop claim.
Placement regret remained below the 5% gate. CCF-C promotion requires fresh,
pre-registered serving batches, two real backend adapters, and at least one
reproducible >=5% latency or backend-configuration ranking inversion whose
confidence interval excludes zero. Without that, this remains a
benchmark/workshop seed.
"""
    (output / "report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
