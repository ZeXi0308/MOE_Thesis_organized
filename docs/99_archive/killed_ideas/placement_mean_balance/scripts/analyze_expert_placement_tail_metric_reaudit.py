#!/usr/bin/env python3
"""Zero-new-GPU-time deepening of the Receiver-Aware Expert Placement
negative result: reuses the EXACT SAME real routing CSVs and scenario
construction as ``analyze_expert_placement_optimization.py`` (already
verdict: NO-GO against ``fp8_total`` = SUM of per-step bottleneck
byte-time), but re-defines the target metric as the metric the project's own
diagnosis says actually matters for P99/TPOT: the TAIL (P95/P99) of the
PER-STEP bottleneck value, not its sum across the whole scenario.

Why this is the correct, cheap follow-up (not p-hacking a new metric until
one works)
----------------------------------------------------------------------------
The original report's own root-cause section states verbatim: "这次评测复用的
`fp8_total`指标，在每个全局时间步取的是跨rank的max（瓶颈/尾部驱动），不是跨rank的
总和或均值...平均负载均衡(mean-balancing)和尾部/瞬时突发均衡(tail-balancing)是
排队论里两个不同的目标". That diagnosis is about the SUM being dominated by
transient concurrency noise, not about the per-step max itself being wrong --
so the natural, pre-specified (not post-hoc-tuned) correction is to look at
the DISTRIBUTION of per-step max values and ask whether popularity-balanced
placement shifts its TAIL (P95/P99), even if it does not move the SUM. This
is exactly what a real system would care about for P99 TPOT: a placement
that only shaves the median step but leaves the worst steps untouched would
be worthless; a placement that shaves the tail even a little, while leaving
the sum unchanged (because most steps are cheap), would be valuable and is
NOT visible in the original ``fp8_total`` (sum) metric.

Frozen GO/NO-GO (pre-specified BEFORE running, matching the original
script's 10% bar and paired-bootstrap discipline)
----------------------------------------------------------------------------
GO iff, for a given (model, origin_mode) cell: the scenario-level paired
bootstrap 95% CI of (best_existing_baseline_P95_step_us -
popularity_balanced_P95_step_us) is entirely > 0, AND the mean P95 reduction
exceeds 10%. Also reports P99 and max-step as secondary diagnostics (not
part of the GO decision, to avoid multiple-comparisons cherry-picking).

Known confound this experiment is specifically designed to surface
----------------------------------------------------------------------------
If placement still shows NO effect even on the tail-of-per-step-max metric,
that would upgrade the original finding from "the SUM metric was the wrong
target" to "the underlying claim (concurrency-driven transient bottlenecks
dominate ANY per-step statistic, not just the sum) is correct" -- i.e. it
would make the original NO-GO much harder to attribute to a metric-choice
artifact, which is exactly the kind of falsification this follow-up is
designed to allow.
"""
from __future__ import annotations


# --- shared-lib bootstrap (auto) ---
import sys
from pathlib import Path as _Path

def _ensure_shared_on_path() -> None:
    here = _Path(__file__).resolve().parent
    for p in [here, *here.parents]:
        cand = p / "experiments" / "shared"
        if (cand / "capture_moe.py").exists():
            s = str(cand)
            if s not in sys.path:
                sys.path.insert(0, s)
            return
        if (p / "capture_moe.py").exists():
            s = str(p)
            if s not in sys.path:
                sys.path.insert(0, s)
            return

_ensure_shared_on_path()
del _ensure_shared_on_path, _Path
# --- end bootstrap ---

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path("/Users/leandrozhao/Desktop/毕设论文资料/experiments/idea_a_mac/outputs")
ROUTES = {
    "olmoe": (BASE / "receiver_aware_v2" / "olmoe_routes.csv", 64, 8),
    "llmjp": (BASE / "receiver_aware_v2" / "llmjp_routes.csv", 32, 16),
}
EP_SIZE = 8
GPUS_PER_NODE = 4
NUM_JOBS = 16
MAX_STAGGER_FRACTION = 0.5
CALIB_JOBS = 12
NUM_SCENARIO_SEEDS = 30
INTER_NODE_GBPS = 200.0
ORIGIN_MODES = ["balanced", "hotspot"]
N_BOOTSTRAP = 2000
SEED = 20260720
REDUCTION_THRESHOLD = 0.10
HIDDEN_SIZE = {"olmoe": 2048, "llmjp": 512}


def placement_map(expert_id: np.ndarray, num_experts: int, ep_size: int, mapping: str) -> np.ndarray:
    if mapping == "contiguous":
        return np.minimum(expert_id * ep_size // num_experts, ep_size - 1)
    if mapping == "round_robin":
        return expert_id % ep_size
    raise ValueError(mapping)


def build_scenario(routes: pd.DataFrame, doc_ids: list[int], arrivals: np.ndarray, origin_mode: str, ep_size: int) -> pd.DataFrame:
    frames = []
    for job_id, (doc_id, arrival) in enumerate(zip(doc_ids, arrivals)):
        rows = routes[routes["sample_id"] == doc_id].copy()
        rows["job_id"] = job_id
        rows["g"] = rows["layer"].astype(int) + int(arrival)
        if origin_mode == "balanced":
            receiver = job_id % ep_size
        elif origin_mode == "hotspot":
            hotspot_jobs = max(1, math.ceil(len(doc_ids) * 0.5))
            receiver = 0 if job_id < hotspot_jobs else 1 + ((job_id - hotspot_jobs) % max(ep_size - 1, 1))
        else:
            raise ValueError(origin_mode)
        rows["receiver_rank"] = receiver
        frames.append(rows[["job_id", "g", "rank", "expert_id", "receiver_rank"]])
    return pd.concat(frames, ignore_index=True)


def popularity_balanced_placement(routes_calib: pd.DataFrame, num_experts: int, ep_size: int) -> dict[int, int]:
    popularity = routes_calib.groupby("expert_id").size()
    popularity = popularity.reindex(range(num_experts), fill_value=0).sort_values(ascending=False)
    loads = [0.0] * ep_size
    assignment: dict[int, int] = {}
    for expert_id, weight in popularity.items():
        target = int(np.argmin(loads))
        assignment[int(expert_id)] = target
        loads[target] += float(weight)
    return assignment


def placement_map_from_assignment(expert_id: np.ndarray, assignment: dict[int, int]) -> np.ndarray:
    return np.array([assignment[int(e)] for e in expert_id])


def scenario_step_series(scn: pd.DataFrame, sender_rank: np.ndarray, gpus_per_node: int, bytes_to_us: float) -> np.ndarray:
    """Returns the PER-STEP bottleneck value (max over ranks of ingress/egress
    remote traffic), i.e. the un-summed distribution underlying fp8_total."""
    scn = scn.copy()
    scn["sender_rank"] = sender_rank
    remote = scn[(scn["sender_rank"] // gpus_per_node) != (scn["receiver_rank"] // gpus_per_node)]
    if remote.empty:
        return np.array([0.0])
    ingress = remote.groupby(["g", "receiver_rank"]).size().astype(float)
    egress = remote.groupby(["g", "sender_rank"]).size().astype(float)
    step_us = pd.concat(
        [ingress.groupby(level="g").max(), egress.groupby(level="g").max()], axis=1
    ).max(axis=1) * bytes_to_us
    return step_us.to_numpy()


def document_bootstrap_ci(values: np.ndarray, n_bootstrap: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    means = np.array([values[rng.integers(0, n, size=n)].mean() for _ in range(n_bootstrap)])
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def run_for_model(model_key: str, route_path: Path, num_experts: int) -> pd.DataFrame:
    routes = pd.read_csv(route_path)
    hidden_size = HIDDEN_SIZE[model_key]
    bw_bytes_per_us = INTER_NODE_GBPS * 1e9 / 8 / 1e6
    bytes_to_us = hidden_size / bw_bytes_per_us
    num_layers = int(routes["layer"].max()) + 1
    doc_pool = sorted(routes["sample_id"].unique().tolist())
    max_stagger = max(1, int(round(num_layers * MAX_STAGGER_FRACTION)))

    calib_docs = doc_pool[:CALIB_JOBS]
    test_docs = doc_pool[CALIB_JOBS:]
    calib_routes = routes[routes["sample_id"].isin(calib_docs)]
    assignment = popularity_balanced_placement(calib_routes, num_experts, EP_SIZE)

    rows = []
    for origin_mode in ORIGIN_MODES:
        # Collect P95, P99, max PER SCENARIO SEED (each seed's own per-step
        # distribution first reduced to its own tail statistic), so bootstrap
        # resamples SCENARIOS (matching the original script's unit of
        # resampling), not raw per-step values pooled across scenarios.
        per_seed = {"contiguous": {"p95": [], "p99": [], "max": []},
                    "round_robin": {"p95": [], "p99": [], "max": []},
                    "popularity_balanced": {"p95": [], "p99": [], "max": []}}
        for scenario_seed in range(NUM_SCENARIO_SEEDS):
            rng = np.random.default_rng(20260720_00 + scenario_seed)
            chosen_docs = rng.choice(
                test_docs, size=min(NUM_JOBS, len(test_docs)), replace=len(test_docs) < NUM_JOBS,
            ).tolist()
            arrivals = rng.integers(0, max_stagger + 1, size=len(chosen_docs))
            scn = build_scenario(routes, chosen_docs, arrivals, origin_mode, EP_SIZE)
            expert_ids = scn["expert_id"].to_numpy()

            for name, sender_rank in [
                ("contiguous", placement_map(expert_ids, num_experts, EP_SIZE, "contiguous")),
                ("round_robin", placement_map(expert_ids, num_experts, EP_SIZE, "round_robin")),
                ("popularity_balanced", placement_map_from_assignment(expert_ids, assignment)),
            ]:
                steps = scenario_step_series(scn, sender_rank, GPUS_PER_NODE, bytes_to_us)
                per_seed[name]["p95"].append(float(np.quantile(steps, 0.95)))
                per_seed[name]["p99"].append(float(np.quantile(steps, 0.99)))
                per_seed[name]["max"].append(float(steps.max()))

        for metric in ["p95", "p99", "max"]:
            arrs = {k: np.array(v[metric]) for k, v in per_seed.items()}
            best_existing_name = "contiguous" if arrs["contiguous"].mean() <= arrs["round_robin"].mean() else "round_robin"
            best_existing = arrs[best_existing_name]
            diff = best_existing - arrs["popularity_balanced"]
            ci_low, ci_high = document_bootstrap_ci(diff, N_BOOTSTRAP, SEED)
            mean_reduction = float(diff.mean() / max(best_existing.mean(), 1e-12))
            go = bool(ci_low > 0.0 and mean_reduction > REDUCTION_THRESHOLD)
            rows.append({
                "model": model_key, "origin_mode": origin_mode, "metric": metric,
                "contiguous_mean_us": float(arrs["contiguous"].mean()),
                "round_robin_mean_us": float(arrs["round_robin"].mean()),
                "popularity_balanced_mean_us": float(arrs["popularity_balanced"].mean()),
                "best_existing_baseline": best_existing_name,
                "reduction_vs_best_existing": mean_reduction,
                "reduction_ci_low_us": ci_low, "reduction_ci_high_us": ci_high,
                "n_scenario_seeds": NUM_SCENARIO_SEEDS,
                "go_no_go": "GO" if go else "NO-GO",
            })
    return pd.DataFrame(rows)


def main() -> None:
    all_rows = []
    for model_key, (route_path, num_experts, _top_k) in ROUTES.items():
        result = run_for_model(model_key, route_path, num_experts)
        all_rows.append(result)
        print(f"\n=== {model_key} ===")
        print(result.to_string(index=False))

    combined = pd.concat(all_rows, ignore_index=True)
    out_dir = BASE / "expert_placement_tail_metric_reaudit_2026-07-20"
    out_dir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_dir / "results.csv", index=False)
    (out_dir / "metadata.json").write_text(json.dumps({
        "ep_size": EP_SIZE, "gpus_per_node": GPUS_PER_NODE, "num_jobs": NUM_JOBS,
        "num_scenario_seeds": NUM_SCENARIO_SEEDS, "calib_jobs": CALIB_JOBS,
        "reduction_threshold": REDUCTION_THRESHOLD,
        "primary_go_metric": "p95",
        "note": "re-targets analyze_expert_placement_optimization.py's SUM (fp8_total) "
                "metric to the per-step-max TAIL distribution (P95/P99/max), zero new GPU time, "
                "reuses identical real routing data and scenario construction",
        "evidence_boundary": "bandwidth-only analytic trace replay, real routing data, no real RDMA/collective kernel measurement",
    }, indent=2), encoding="utf-8")
    print(f"\nsaved to {out_dir}")


if __name__ == "__main__":
    main()
