#!/usr/bin/env python3
"""Estimate token inclusion q_e and test distinct-expert predictions on traces.

Accepts capture_routes_v2.py CSV (sample_id,layer,token_position,rank,expert_id),
or JSON/JSONL records with layer,num_tokens,expert_counts. The latter needs an
explicit --top-k-without-replacement assertion, since histograms alone cannot
detect repeated expert IDs within one token. Counter histograms are sufficient
for q_e, but cannot reconstruct token order or hypothetical speculative routes.

Exact-route replay groups B separate samples and adjacent gamma+1 positions.
This preserves within-sample route correlation. It is a trace-window diagnostic,
NOT an executed speculative-decoding experiment: drafted tokens can route
differently, and workload/context position can change the marginal distribution.
Probability fitting and correction use fit samples; held-out samples are never
used to fit q_e or correction ratios. Samples are split deterministically by ID.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Mapping

from cost_model import expected_distinct, expected_distinct_slot_approx


def load_records(path: str | Path) -> list[dict]:
    path = Path(path)
    if path.suffix.lower() == ".csv":
        with path.open(newline="") as f:
            return list(csv.DictReader(f))
    if path.suffix.lower() in (".jsonl", ".ndjson"):
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    obj = json.loads(path.read_text())
    if isinstance(obj, list):
        return obj
    if "episodes" in obj:
        return [row for episode in obj["episodes"] for row in episode.get("plugin", {}).get("routes", [])]
    if "plugin" in obj:
        return obj["plugin"].get("routes", [])
    return obj.get("records", obj.get("routes", []))


def parse_exact_routes(records: Iterable[Mapping], num_experts: int) -> dict:
    routes = defaultdict(dict)
    rank_sets = defaultdict(set)
    for row in records:
        layer = str(row["layer"])
        sample = str(row["sample_id"])
        position, expert = int(row["token_position"]), int(row["expert_id"])
        if not 0 <= expert < num_experts or position < 0:
            raise ValueError("expert or token position out of bounds")
        key = (sample, position)
        experts = routes[layer].setdefault(key, set())
        if expert in experts:
            raise ValueError(f"duplicate expert within token: layer={layer}, token={key}")
        experts.add(expert)
        if row.get("rank") not in (None, ""):
            rank = int(row["rank"])
            rank_key = (layer, sample, position)
            if rank in rank_sets[rank_key]:
                raise ValueError(f"duplicate rank: {rank_key}")
            rank_sets[rank_key].add(rank)
    if not routes:
        raise ValueError("empty route trace")
    for layer, tokens in routes.items():
        topks = {len(ids) for ids in tokens.values()}
        if len(topks) != 1:
            raise ValueError(f"layer {layer}: variable/incomplete top-k rows {topks}")
    return dict(routes)


def probability_estimate(tokens: Mapping, num_experts: int) -> dict:
    if not tokens:
        raise ValueError("no tokens in probability fitting partition")
    counts = Counter(e for experts in tokens.values() for e in experts)
    total_routes = sum(counts.values())
    q = [counts[e] / len(tokens) for e in range(num_experts)]
    p = [counts[e] / total_routes for e in range(num_experts)]
    return {"q_e": q, "p_e": p, "token_count": len(tokens), "route_count": total_routes,
            "top_k": total_routes / len(tokens), "q_sum": math.fsum(q), "p_sum": math.fsum(p)}


def split_samples(samples: Iterable[str], fit_fraction: float) -> tuple[set, set]:
    if not 0 < fit_fraction <= 1:
        raise ValueError("fit_fraction must be in (0,1]")
    ordered = sorted(set(samples), key=lambda s: hashlib.sha256(str(s).encode()).hexdigest())
    if len(ordered) < 2 or fit_fraction == 1:
        return set(ordered), set()
    split = min(len(ordered) - 1, max(1, int(len(ordered) * fit_fraction)))
    return set(ordered[:split]), set(ordered[split:])


def replay_windows(tokens: Mapping, batch: int, gamma: int) -> list[int]:
    """Nonoverlapping B-sample groups and gamma+1 token windows, no resampling.

    Incomplete groups/windows are excluded and their absence is explicit in
    window_count. Window counts are diagnostic observations, not independent
    runtime repetitions or confidence-interval sample sizes.
    """
    if batch < 1 or gamma < 0:
        raise ValueError("batch/gamma out of range")
    sequences = defaultdict(dict)
    for (sample, position), expert_set in tokens.items():
        sequences[sample][position] = expert_set
    sample_ids = sorted(sequences)
    width, observed = gamma + 1, []
    for group_start in range(0, len(sample_ids) - batch + 1, batch):
        group = [sequences[s] for s in sample_ids[group_start:group_start + batch]]
        common_positions = sorted(set.intersection(*(set(s) for s in group)))
        cursor = 0
        while cursor + width <= len(common_positions):
            positions = common_positions[cursor:cursor + width]
            if positions[-1] - positions[0] != width - 1:
                cursor += 1
                continue
            union = set()
            for sequence in group:
                for position in positions:
                    union.update(sequence[position])
            observed.append(len(union))
            cursor += width
    return observed


def analyze_exact(records: Iterable[Mapping], num_experts: int, batches: list[int],
                  gammas: list[int], fit_fraction: float = 0.7) -> tuple[dict, list[dict]]:
    routes = parse_exact_routes(records, num_experts)
    all_samples = {sample for tokens in routes.values() for sample, _ in tokens}
    fit_ids, holdout_ids = split_samples(all_samples, fit_fraction)
    layers, comparisons, corrections = [], [], []
    for layer, tokens in sorted(routes.items()):
        fit = {key: value for key, value in tokens.items() if key[0] in fit_ids}
        held = {key: value for key, value in tokens.items() if key[0] in holdout_ids}
        estimate = probability_estimate(fit, num_experts)
        layers.append({"layer": layer, **estimate})
        for batch in batches:
            for gamma in gammas:
                predicted = expected_distinct(estimate["q_e"], batch, gamma)
                slot_predicted = expected_distinct_slot_approx(
                    estimate["p_e"], batch, gamma, int(estimate["top_k"]))
                fit_values = replay_windows(fit, batch, gamma)
                correction = statistics.mean(fit_values) / predicted if fit_values else None
                if correction is not None:
                    corrections.append({"layer": layer, "batch": batch, "gamma": gamma,
                                        "fit_ratio": correction, "fit_windows": len(fit_values)})
                for partition, subset, values in (
                        ("fit", fit, fit_values), ("holdout", held, replay_windows(held, batch, gamma))):
                    actual = statistics.mean(values) if values else None
                    corrected = min(num_experts, max(estimate["top_k"], predicted * correction)) \
                        if correction is not None else None
                    comparisons.append({"layer": layer, "batch": batch, "gamma": gamma,
                        "partition": partition, "sample_count": len({k[0] for k in subset}),
                        "window_count": len(values), "predicted_distinct": predicted,
                        "slot_approx_distinct": slot_predicted, "observed_window_distinct_mean": actual,
                        "observed_window_distinct_min": min(values) if values else None,
                        "observed_window_distinct_max": max(values) if values else None,
                        "fit_corrected_distinct": corrected,
                        "prediction_error": predicted - actual if actual is not None else None,
                        "corrected_error": corrected - actual if actual is not None and corrected is not None else None,
                        "evidence_kind": "route_window_replay_not_executed_speculation"})
    result = {"schema_version": 1, "num_experts": num_experts,
        "probability_definition": {"q_e": "P(expert included in token top-k), sum(q_e)=k",
                                   "p_e": "fraction of individual route slots, sum(p_e)=1"},
        "fit_samples": sorted(fit_ids), "holdout_samples": sorted(holdout_ids),
        "layers": layers, "trace_correction": corrections,
        "evidence_kind": "measured_route_trace_with_offline_replay",
        "boundaries": ["B=1,gamma=0 expected distinct equals k",
            "tokens independent in analytic q prediction; top-k slots not independent",
            "window replay does not simulate draft-token route changes",
            "fit ratio is an offline lookup correction, not a transferable bandwidth model",
            "windows from shared documents are not independent system repetitions"]}
    return result, comparisons


def analyze_histograms(records: Iterable[Mapping], num_experts: int,
                       top_k_without_replacement: bool) -> tuple[dict, list[dict]]:
    if not top_k_without_replacement:
        raise ValueError("histograms require --top-k-without-replacement; cannot infer token uniqueness")
    per_layer = defaultdict(list)
    for record in records:
        if record.get("model_phase", "target") != "target" or record.get("phase", "decode") != "decode":
            continue
        if int(record.get("invalid_expert_assignments", 0)):
            raise ValueError("route event has invalid expert assignments")
        counts = record.get("expert_counts", record.get("expert_token_counts", record.get("tokens_per_expert")))
        if counts is None:
            continue  # non-route events may share the JSONL stream
        if isinstance(counts, dict):
            sparse = {int(k): int(v) for k, v in counts.items()}
            if any(k < 0 or k >= num_experts for k in sparse):
                raise ValueError("expert id outside --experts range")
            counts = [sparse.get(e, 0) for e in range(num_experts)]
        else:
            counts = [int(v) for v in counts]
        token_count = int(record.get("num_tokens", record.get("tokens", record.get("token_count", 0))))
        if len(counts) != num_experts or token_count <= 0 or any(c < 0 or c > token_count for c in counts):
            raise ValueError("invalid per-expert histogram or num_tokens")
        layer = str(record["layer"])
        per_layer[layer].append((token_count, counts, record))
    if not per_layer:
        raise ValueError("no expert histogram events found")
    layers, comparisons = [], []
    for layer, observations in sorted(per_layer.items()):
        total_tokens = sum(n for n, _, _ in observations)
        counts = [sum(c[e] for _, c, _ in observations) for e in range(num_experts)]
        total_routes = sum(counts)
        if total_routes == 0:
            raise ValueError("histograms contain no expert routes")
        q = [c / total_tokens for c in counts]
        layers.append({"layer": layer, "q_e": q, "p_e": [c / total_routes for c in counts],
                       "token_count": total_tokens, "route_count": total_routes,
                       "top_k": total_routes / total_tokens, "q_sum": sum(q), "p_sum": 1.0})
        for n, histogram, record in observations:
            distinct = sum(c > 0 for c in histogram)
            # B*gamma is often unavailable/variable, so use ACTUAL token count.
            predicted = expected_distinct(q, n, 0)
            comparisons.append({"layer": layer, "event_id": record.get("event_id", record.get("step_id", record.get("step"))),
                "batch": record.get("batch"), "gamma": record.get("gamma"), "num_tokens": n,
                "observed_distinct": distinct, "predicted_distinct": predicted,
                "prediction_error": predicted - distinct, "partition": "in_sample",
                "evidence_kind": "measured_histogram_in_sample_probability_fit"})
    return {"schema_version": 1, "num_experts": num_experts, "layers": layers,
            "probability_definition": {"q_e": "expert token count / total tokens; sum(q_e)=mean k",
                                       "p_e": "expert route count / total routes; sum(p_e)=1"},
            "evidence_kind": "measured_route_histograms",
            "boundaries": ["in-sample descriptive check, not holdout calibration",
                "token-level top-k uniqueness asserted by caller",
                "histograms cannot replay token adjacency or cross-step correlations",
                "logical expert working-set bytes do not measure HBM traffic"]}, comparisons


def write_csv(path: str | Path, rows: list[dict]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace")
    parser.add_argument("--experts", type=int, required=True, help="global expert count per routed layer")
    parser.add_argument("--output", required=True, help="probability and correction JSON")
    parser.add_argument("--comparison", help="comparison CSV, default OUTPUT_stem_comparison.csv")
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32, 64])
    parser.add_argument("--gammas", nargs="+", type=int, default=[0, 1, 2, 3, 4, 5, 7])
    parser.add_argument("--fit-fraction", type=float, default=0.7)
    parser.add_argument("--top-k-without-replacement", action="store_true")
    args = parser.parse_args()
    if args.experts < 1:
        parser.error("--experts must be positive")
    records = load_records(args.trace)
    if not records:
        parser.error("empty trace")
    if "expert_id" in records[0]:
        result, comparison = analyze_exact(records, args.experts, args.batches, args.gammas, args.fit_fraction)
    else:
        result, comparison = analyze_histograms(records, args.experts, args.top_k_without_replacement)
    result["source"] = str(Path(args.trace).resolve())
    result["source_sha256"] = hashlib.sha256(Path(args.trace).read_bytes()).hexdigest()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    comparison_path = args.comparison or output.with_name(output.stem + "_comparison.csv")
    write_csv(comparison_path, comparison)
    print(json.dumps({"probabilities": str(output), "comparison": str(comparison_path),
                      "layers": len(result["layers"]), "evidence_kind": result["evidence_kind"]}))


if __name__ == "__main__":
    main()
