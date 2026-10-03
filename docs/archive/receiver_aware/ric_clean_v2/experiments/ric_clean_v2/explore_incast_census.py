#!/usr/bin/env python3
"""Route-real census for a bounded-incast admission gate.

This deliberately does not simulate a network.  It asks only whether native
MoE routes contain the structural precondition for a later multi-sender,
receiver-bounded experiment: several live joins and several sender ranks
converging on the same token-owner receiver in one causal token wave.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from statistics import median
from typing import Any, Iterable

try:
    from .explore_receiver_matched_milp import load_verified_joins
except ImportError:  # pragma: no cover
    from explore_receiver_matched_milp import load_verified_joins  # type: ignore


MODELS = ("olmoe", "llmjp")


def quantile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("empty quantile input")
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    low = int(index)
    high = min(low + 1, len(ordered) - 1)
    fraction = index - low
    return ordered[low] * (1.0 - fraction) + ordered[high] * fraction


def summarize(values: Iterable[int]) -> dict[str, float]:
    materialized = [float(value) for value in values]
    return {
        "min": min(materialized),
        "median": median(materialized),
        "p95": quantile(materialized, 0.95),
        "max": max(materialized),
    }


def census_model(route_root: Path, model: str) -> dict[str, Any]:
    joins, metadata = load_verified_joins(route_root, model)
    by_wave_receiver: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    join_sender_fanin: list[int] = []
    join_sender_multiplicity: list[int] = []
    for join in joins:
        senders = [int(row["sender_rank"]) for row in join["siblings"]]
        join_sender_fanin.append(len(set(senders)))
        join_sender_multiplicity.append(max(Counter(senders).values()))
        by_wave_receiver[(int(join["token_position"]), int(join["receiver_rank"]))].append(join)

    wave_join_count: list[int] = []
    wave_sender_count: list[int] = []
    wave_contribution_count: list[int] = []
    wave_max_sender_burst: list[int] = []
    qualifying = 0
    for wave_joins in by_wave_receiver.values():
        sender_counts: Counter[int] = Counter()
        for join in wave_joins:
            sender_counts.update(int(row["sender_rank"]) for row in join["siblings"])
        wave_join_count.append(len(wave_joins))
        wave_sender_count.append(len(sender_counts))
        wave_contribution_count.append(sum(sender_counts.values()))
        wave_max_sender_burst.append(max(sender_counts.values()))
        if len(wave_joins) >= 2 and len(sender_counts) >= 2:
            qualifying += 1

    expected_waves = 128 * 8
    if len(by_wave_receiver) != expected_waves:
        raise RuntimeError(f"expected {expected_waves} receiver waves, got {len(by_wave_receiver)}")
    return {
        "model": model,
        "evidence_boundary": "native_routes_only_not_network_or_serving",
        "top_k": int(metadata["top_k"]),
        "num_joins": len(joins),
        "num_receiver_waves": len(by_wave_receiver),
        "qualifying_multi_join_multi_sender_fraction": qualifying / len(by_wave_receiver),
        "per_join_distinct_sender_fanin": summarize(join_sender_fanin),
        "per_join_max_contributions_from_one_sender": summarize(join_sender_multiplicity),
        "per_receiver_wave_live_joins": summarize(wave_join_count),
        "per_receiver_wave_distinct_senders": summarize(wave_sender_count),
        "per_receiver_wave_contributions": summarize(wave_contribution_count),
        "per_receiver_wave_max_sender_burst": summarize(wave_max_sender_burst),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--route-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {
        "schema_version": "incast-route-census-v1",
        "scientific_result": False,
        "models": [census_model(args.route_root, model) for model in MODELS],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
