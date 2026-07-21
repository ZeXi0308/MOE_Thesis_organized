#!/usr/bin/env python3
"""Communication-aware expert placement optimization via popularity-balanced
partitioning (LPT greedy) -- a genuinely NEW lever on the receiver-aware
line that NO prior experiment in this project has touched.

READ THIS BEFORE RUNNING OR CITING RESULTS.

Why this is a different, and arguably more natural, lever than anything
tried today
----------------------------------------------------------------------------
Every receiver-aware experiment in this project so far (v1, v2 systematic,
v3 adaptive, direct-benefit controller) held EXPERT PLACEMENT fixed
(``contiguous`` = expert_id // ep_size, or ``round_robin`` = expert_id %
ep_size -- both completely blind to any real data) and only ever optimized
the ONLINE precision-degradation decision GIVEN that placement. But
receiver-side congestion in MoE expert-parallel serving has a well-known
STRUCTURAL cause that placement -- not online control -- is the classical
tool for: if popular ("hot") experts happen to cluster on the same physical
node under a data-blind placement, that node's egress becomes a bottleneck
for every job whose receiver lives elsewhere, no matter how clever the
runtime controller is. This is exactly a MULTIWAY NUMBER PARTITIONING /
balanced bin-packing problem (assign weighted items to bins minimizing the
max bin load), for which greedy LPT (Longest-Processing-Time-first: sort
items by weight descending, always place the next item in the currently
least-loaded bin) is a classical, well-understood, easy-to-verify heuristic
with a known worst-case approximation ratio (<= 4/3 - 1/(3*ep_size) of
optimal makespan). Applying it here, using REAL per-expert selection
frequency from real routing traces as the item weights, is the natural
"topology+congestion, established-technique" lever the receiver-aware line
has been missing.

This needs ZERO new GPU time: it is pure pandas/numpy analysis of ALREADY
COLLECTED real routing CSVs
(``outputs/receiver_aware_v2/olmoe_routes.csv``/``llmjp_routes.csv``,
already used by today's direct-benefit-controller experiment), reusing the
EXACT SAME scenario-building and bottleneck-byte-time metric
(``build_scenario``/``fp8_total``, i.e. the sum over global steps of
max(receiver-side ingress, sender-side egress) traffic count converted to
microseconds at a fixed inter-node bandwidth) as
``run_receiver_aware_v2_systematic.py``, so results are directly comparable
to every number already reported for that line. ``build_scenario`` and
``placement_map`` below are copied VERBATIM from that script (not
reimplemented) purely so this script has zero dependency on ``transformers``
and can run on a machine with no GPU/CUDA stack at all.

Hypothesis and frozen GO/NO-GO
----------------------------------------------------------------------------
H: expert popularity (real per-token top-k selection frequency) is skewed
enough, and uncorrelated enough with expert_id, that a data-driven
popularity-balanced placement reduces the all-FP8 baseline bottleneck
byte-time (``fp8_total``) relative to the best of the two existing
data-blind baselines (``contiguous``, ``round_robin``), on HELD-OUT test
scenarios never used to fit the placement.

GO iff, for a given (model, origin_mode) cell, across >= 20 held-out
scenario seeds: the scenario-level paired bootstrap 95% CI of
(best_existing_baseline_fp8_total - popularity_balanced_fp8_total) is
entirely > 0 (i.e., strictly less bottleneck byte-time, not just on
average) AND the mean reduction exceeds 10% (matching this project's usual
"non-trivial, not noise-level" bar for structural placement claims).

The placement is fit ONLY on calibration documents (the same calib/test
document split boundary used throughout today's other analyses) -- it never
sees the held-out test scenarios' documents, so there is no leakage between
"fitting the placement" and "evaluating it".

Known confounds most likely to overturn a GO result
--------------------------------------------------------
  1. This uses TOTAL selection count as the popularity weight, which
     matches ``fp8_total``'s bottleneck definition (raw traffic count) but
     ignores RANK (a tail-rank hit and a rank-1 hit cost the analytic model
     the same 1 unit here) -- the project's own rank-tail finding (see the
     rank-lut re-verification report) suggests a rank-weighted popularity
     signal might do even better; this script deliberately tests the
     SIMPLEST version first.
  2. Under ``hotspot`` origin_mode, the receiver side is concentrated at a
     single rank by construction (independent of placement) -- placement
     can only ever rebalance the SENDER (egress) side, so this mechanism is
     expected to help LESS (or not at all) under hotspot than under
     balanced origin_mode; a NO-GO under hotspot alone should not be read
     as invalidating a GO under balanced.
  3. This is still bandwidth-only analytic trace replay (identical
     evidence boundary to ``run_receiver_aware_v2_systematic.py``) -- no
     real collective/kernel/RDMA measurement. A placement change also has a
     real-world cost this script does NOT model: physically moving expert
     weights between GPUs/nodes at deployment/rebalance time.
  4. Real production EP placement is also constrained by memory capacity
     per node (equal expert COUNT per node, not just equal popularity) --
     this script's LPT variant balances popularity-weighted load but does
     NOT explicitly cap experts-per-node; the ``max_experts_per_node``
     check below reports whether the resulting assignment happens to stay
     memory-balanced too, but does not enforce it as a hard constraint.
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


# ---------------------------------------------------------------------------
# Copied verbatim from run_receiver_aware_v2_systematic.py (see module
# docstring for why: avoids a hard ``transformers`` dependency here).
# ---------------------------------------------------------------------------

def placement_map(expert_id: np.ndarray, num_experts: int, ep_size: int, mapping: str) -> np.ndarray:
    if mapping == "contiguous":
        return np.minimum(expert_id * ep_size // num_experts, ep_size - 1)
    if mapping == "round_robin":
        return expert_id % ep_size
    raise ValueError(mapping)


def build_scenario(
    routes: pd.DataFrame, doc_ids: list[int], arrivals: np.ndarray, origin_mode: str, ep_size: int,
) -> pd.DataFrame:
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


# ---------------------------------------------------------------------------
# NEW: popularity-balanced placement via greedy LPT.
# ---------------------------------------------------------------------------

def popularity_balanced_placement(routes_calib: pd.DataFrame, num_experts: int, ep_size: int) -> dict[int, int]:
    popularity = routes_calib.groupby("expert_id").size()
    popularity = popularity.reindex(range(num_experts), fill_value=0).sort_values(ascending=False)
    loads = [0.0] * ep_size
    counts = [0] * ep_size
    assignment: dict[int, int] = {}
    for expert_id, weight in popularity.items():
        target = int(np.argmin(loads))
        assignment[int(expert_id)] = target
        loads[target] += float(weight)
        counts[target] += 1
    return assignment, loads, counts


def placement_map_from_assignment(expert_id: np.ndarray, assignment: dict[int, int]) -> np.ndarray:
    return np.array([assignment[int(e)] for e in expert_id])


def scenario_fp8_total(scn: pd.DataFrame, sender_rank: np.ndarray, gpus_per_node: int, bytes_to_us: float) -> float:
    scn = scn.copy()
    scn["sender_rank"] = sender_rank
    remote = scn[(scn["sender_rank"] // gpus_per_node) != (scn["receiver_rank"] // gpus_per_node)]
    if remote.empty:
        return 0.0
    ingress = remote.groupby(["g", "receiver_rank"]).size().astype(float)
    egress = remote.groupby(["g", "sender_rank"]).size().astype(float)
    step_us = pd.concat(
        [ingress.groupby(level="g").max(), egress.groupby(level="g").max()], axis=1
    ).max(axis=1) * bytes_to_us
    return float(step_us.sum())


def document_bootstrap_ci(values: np.ndarray, n_bootstrap: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    means = np.array([values[rng.integers(0, n, size=n)].mean() for _ in range(n_bootstrap)])
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


HIDDEN_SIZE = {"olmoe": 2048, "llmjp": 512}


def run_for_model(model_key: str, route_path: Path, num_experts: int, top_k: int) -> pd.DataFrame:
    routes = pd.read_csv(route_path)
    hidden_size = HIDDEN_SIZE[model_key]
    bw_bytes_per_us = INTER_NODE_GBPS * 1e9 / 8 / 1e6
    bytes_to_us = hidden_size / bw_bytes_per_us
    num_layers = int(routes["layer"].max()) + 1
    doc_pool = sorted(routes["sample_id"].unique().tolist())
    max_stagger = max(1, int(round(num_layers * MAX_STAGGER_FRACTION)))

    calib_docs = doc_pool[:CALIB_JOBS]
    test_docs = doc_pool[CALIB_JOBS:]
    if len(test_docs) < 4:
        raise RuntimeError(f"{model_key}: not enough documents left for test after calibration split")

    calib_routes = routes[routes["sample_id"].isin(calib_docs)]
    assignment, loads, counts = popularity_balanced_placement(calib_routes, num_experts, EP_SIZE)
    print(f"[{model_key}] popularity-balanced placement: per-rank calib load={[round(v) for v in loads]}, "
          f"experts-per-rank={counts}")

    rows = []
    for origin_mode in ORIGIN_MODES:
        per_seed = {"contiguous": [], "round_robin": [], "random": [], "popularity_balanced": []}
        for scenario_seed in range(NUM_SCENARIO_SEEDS):
            rng = np.random.default_rng(20260720_00 + scenario_seed)
            chosen_docs = rng.choice(
                test_docs, size=min(NUM_JOBS, len(test_docs)), replace=len(test_docs) < NUM_JOBS,
            ).tolist()
            arrivals = rng.integers(0, max_stagger + 1, size=len(chosen_docs))
            scn = build_scenario(routes, chosen_docs, arrivals, origin_mode, EP_SIZE)
            expert_ids = scn["expert_id"].to_numpy()

            per_seed["contiguous"].append(
                scenario_fp8_total(scn, placement_map(expert_ids, num_experts, EP_SIZE, "contiguous"),
                                    GPUS_PER_NODE, bytes_to_us))
            per_seed["round_robin"].append(
                scenario_fp8_total(scn, placement_map(expert_ids, num_experts, EP_SIZE, "round_robin"),
                                    GPUS_PER_NODE, bytes_to_us))
            random_rng = np.random.default_rng(SEED + scenario_seed)
            random_assignment = {e: int(x) for e, x in zip(range(num_experts), random_rng.integers(0, EP_SIZE, num_experts))}
            per_seed["random"].append(
                scenario_fp8_total(scn, placement_map_from_assignment(expert_ids, random_assignment),
                                    GPUS_PER_NODE, bytes_to_us))
            per_seed["popularity_balanced"].append(
                scenario_fp8_total(scn, placement_map_from_assignment(expert_ids, assignment),
                                    GPUS_PER_NODE, bytes_to_us))

        arrs = {k: np.array(v) for k, v in per_seed.items()}
        best_existing_name = "contiguous" if arrs["contiguous"].mean() <= arrs["round_robin"].mean() else "round_robin"
        best_existing = arrs[best_existing_name]
        diff = best_existing - arrs["popularity_balanced"]
        ci_low, ci_high = document_bootstrap_ci(diff, N_BOOTSTRAP, SEED)
        mean_reduction = float(diff.mean() / max(best_existing.mean(), 1e-12))
        go = bool(ci_low > 0.0 and mean_reduction > REDUCTION_THRESHOLD)

        rows.append({
            "model": model_key, "origin_mode": origin_mode,
            "contiguous_mean_us": float(arrs["contiguous"].mean()),
            "round_robin_mean_us": float(arrs["round_robin"].mean()),
            "random_mean_us": float(arrs["random"].mean()),
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
    for model_key, (route_path, num_experts, top_k) in ROUTES.items():
        result = run_for_model(model_key, route_path, num_experts, top_k)
        all_rows.append(result)
        print(f"\n=== {model_key} ===")
        print(result.to_string(index=False))

    combined = pd.concat(all_rows, ignore_index=True)
    out_dir = BASE / "expert_placement_optimization_2026-07-20"
    out_dir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_dir / "results.csv", index=False)
    (out_dir / "metadata.json").write_text(json.dumps({
        "ep_size": EP_SIZE, "gpus_per_node": GPUS_PER_NODE, "num_jobs": NUM_JOBS,
        "num_scenario_seeds": NUM_SCENARIO_SEEDS, "calib_jobs": CALIB_JOBS,
        "reduction_threshold": REDUCTION_THRESHOLD,
        "evidence_boundary": "bandwidth-only analytic trace replay, real routing data, no real RDMA/collective kernel measurement",
    }, indent=2), encoding="utf-8")
    print(f"\nsaved to {out_dir}")


if __name__ == "__main__":
    main()
