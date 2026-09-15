#!/usr/bin/env python3
"""Adaptive-period (AIMD-style) shadow-verification controller: a genuinely
NEW mechanism refinement over the FIXED-period controller tested on real GPU
earlier today (``run_expert_precision_persistence_shadow_verify_p0.py``).

READ THIS BEFORE RUNNING OR CITING RESULTS.

Why this exists
----------------
Today's real-GPU fixed-period controller simulation (period in {4, 8, 16})
found: OLMoE GO at period=4 (50.1% KL reduction, CI[42.1%,56.6%]); LLM-jp
NO-GO at period=4 by a narrow margin (47.0% reduction, CI[40.2%,53.4%] --
the CI upper bound already exceeds the 50% bar, only the point estimate
falls short). A FIXED period cannot adapt: it verifies exactly as often
during a long safe streak as during a risky one. This script tests whether
an AIMD-style (Additive-Increase-Multiplicative-Decrease, the same principle
behind TCP congestion control) ADAPTIVE verify interval -- widen the window
after a safe verify, sharply narrow it after a risky one -- captures the
SAME underlying persistence signal more efficiently, which could plausibly
close LLM-jp's narrow gap and/or improve OLMoE's margin further.

This is a genuinely NEW mechanism, not a re-verification: today's script
only ever tested a FIXED grid of periods. It costs ZERO new GPU time -- it
re-analyzes the SAME already-collected ``per_step_samples.csv`` trajectories
from today's persistence experiment, purely offline (no torch/CUDA needed).

Mechanism
---------
State: current window size w (steps until next verify), initialized to
``w_init``. At each verify step t: observe the REALIZED kl_trajectory[t]
(this step is served bf16/high-precision, exactly as in the fixed-period
controller). If verified_kl > tau (risky): MULTIPLICATIVE DECREASE --
w <- max(w_min, w // shrink_factor); escalate to high precision for the
(now-shrunk) window, then re-verify sooner. If verified_kl <= tau (safe):
ADDITIVE INCREASE -- w <- min(w_max, w + grow_step); serve LOW precision
for that (now-widened) window, then re-verify later. This only ever uses
information realized strictly before the current decision point (the last
verify's own outcome) -- same causality discipline as the fixed-period
controller.

Statistical discipline (unchanged from today)
----------------------------------------------------------------------------
Hyperparameters (w_min, w_max, w_init, grow_step, shrink_factor, escalate
quantile) are grid-searched ONLY on calibration documents (first 12 of 32,
same split boundary as today, for direct comparability), selecting the
config that maximizes a bootstrap-LCB-robust reduction score (mean - std,
NOT the raw mean) to avoid overfitting to calibration noise. The SINGLE
selected config is then evaluated ONCE on the held-out test documents (the
same 20 test documents used today), with document-level paired bootstrap
CIs. GO/NO-GO thresholds are IDENTICAL to today's frozen criteria (>=50%
reduction vs always-low, <=50% high-precision fraction, <=2x the oracle
upper bound at the same threshold) so the comparison to the fixed-period
result is apples-to-apples.

Known confounds
--------------------
  1. The calibration split has only 12 documents -- the same small-sample
     risk flagged today applies to hyperparameter selection here too; a
     borderline win should be treated as suggestive, not final, until
     re-run with more documents.
  2. This is still an OFFLINE REPLAY of already-collected trajectories, not
     a real in-loop runtime decision (see the separate real-time-controller
     script for that). The "high precision" / "low precision" labels here
     are the SAME abstract cost units as today (verify-step count), not
     wall-clock time.
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

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path("/Users/leandrozhao/Desktop/毕设论文资料/experiments/idea_a_mac/outputs")
MODELS = {
    "olmoe": BASE / "expert_precision_persistence_2026-07-20_olmoe" / "per_step_samples.csv",
    "llmjp": BASE / "expert_precision_persistence_2026-07-20_llmjp" / "per_step_samples.csv",
}
CALIB_SAMPLES = 12
N_BOOTSTRAP = 2000
SEED = 20260720

REDUCTION_THRESHOLD = 0.50
HIGH_FRAC_THRESHOLD = 0.50
ORACLE_RATIO_THRESHOLD = 2.0

# Grid search space for the adaptive controller (calibration-only).
W_MIN_GRID = [1, 2]
W_MAX_GRID = [8, 12, 16, 24]
W_INIT_GRID = [2, 4]
GROW_STEP_GRID = [1, 2]
SHRINK_FACTOR_GRID = [2, 3, 4]
QUANTILE_GRID = [0.60, 0.70, 0.75, 0.80, 0.90]


def simulate_adaptive(kl_trajectory: np.ndarray, threshold: float, w_min: int, w_max: int,
                       w_init: int, grow_step: int, shrink_factor: int) -> tuple[float, float]:
    T = len(kl_trajectory)
    high = np.zeros(T, dtype=bool)
    t = 0
    w = w_init
    while t < T:
        high[t] = True  # verify point, served high precision (bf16)
        verified_kl = kl_trajectory[t]
        if verified_kl > threshold:
            w = max(w_min, w // shrink_factor)
            end = min(t + w, T)
            high[t:end] = True  # escalate: whole shrunk window served high
        else:
            end = min(t + w, T)
            w = min(w_max, w + grow_step)
        t = end
    kl = float(kl_trajectory[~high].sum())
    high_frac = float(high.mean())
    return kl, high_frac


def simulate_oracle(kl_trajectory: np.ndarray, threshold: float) -> tuple[float, float]:
    high = kl_trajectory > threshold
    return float(kl_trajectory[~high].sum()), float(high.mean())


def aggregate(per_doc: dict[int, dict[str, float]], keys: list[int], count_keys: list[str]) -> dict[str, float]:
    agg = {name: 0.0 for name in per_doc[keys[0]]}
    for d in keys:
        for name, value in per_doc[d].items():
            if name in count_keys:
                agg[name] += value / len(keys)
            else:
                agg[name] += value
    return agg


def calibrate(df: pd.DataFrame, calib_docs: list[int]) -> dict:
    calib_traj = {d: df[df.doc_id == d].sort_values("step")["kl"].to_numpy() for d in calib_docs}
    calib_kl_pool = np.concatenate(list(calib_traj.values()))

    best = None
    for quantile in QUANTILE_GRID:
        tau = float(np.quantile(calib_kl_pool, quantile))
        for w_min, w_max, w_init, grow_step, shrink_factor in itertools.product(
            W_MIN_GRID, W_MAX_GRID, W_INIT_GRID, GROW_STEP_GRID, SHRINK_FACTOR_GRID,
        ):
            if w_init > w_max or w_min > w_init:
                continue
            reductions = []
            for d, traj in calib_traj.items():
                if len(traj) < 2:
                    continue
                always_low_kl = float(traj.sum())
                kl, _ = simulate_adaptive(traj, tau, w_min, w_max, w_init, grow_step, shrink_factor)
                reductions.append(1.0 - kl / max(always_low_kl, 1e-12))
            if not reductions:
                continue
            score = float(np.mean(reductions) - np.std(reductions))  # LCB-style robust score
            config = {
                "quantile": quantile, "tau": tau, "w_min": w_min, "w_max": w_max,
                "w_init": w_init, "grow_step": grow_step, "shrink_factor": shrink_factor,
                "calib_score": score, "calib_mean_reduction": float(np.mean(reductions)),
            }
            if best is None or score > best["calib_score"]:
                best = config
    return best


def evaluate_on_test(df: pd.DataFrame, test_docs: list[int], config: dict) -> dict:
    per_doc: dict[int, dict[str, float]] = {}
    for d in test_docs:
        traj = df[df.doc_id == d].sort_values("step")["kl"].to_numpy()
        if len(traj) < 2:
            continue
        always_low_kl = float(traj.sum())
        adaptive_kl, adaptive_high_frac = simulate_adaptive(
            traj, config["tau"], config["w_min"], config["w_max"],
            config["w_init"], config["grow_step"], config["shrink_factor"],
        )
        oracle_kl, oracle_high_frac = simulate_oracle(traj, config["tau"])
        per_doc[d] = {
            "always_low_kl": always_low_kl, "adaptive_kl": adaptive_kl,
            "adaptive_high_frac": adaptive_high_frac, "oracle_kl": oracle_kl, "oracle_high_frac": oracle_high_frac,
        }
    doc_ids = list(per_doc.keys())
    point = aggregate(per_doc, doc_ids, count_keys=["adaptive_high_frac", "oracle_high_frac"])
    reduction = 1.0 - point["adaptive_kl"] / max(point["always_low_kl"], 1e-12)
    oracle_ratio = point["adaptive_kl"] / max(point["oracle_kl"], 1e-9)

    rng = np.random.default_rng(SEED)
    boot_reductions = []
    for _ in range(N_BOOTSTRAP):
        chosen = rng.choice(doc_ids, size=len(doc_ids), replace=True)
        agg_b = aggregate(per_doc, list(chosen), count_keys=["adaptive_high_frac", "oracle_high_frac"])
        boot_reductions.append(1.0 - agg_b["adaptive_kl"] / max(agg_b["always_low_kl"], 1e-12))
    ci_low, ci_high = np.quantile(boot_reductions, [0.025, 0.975])

    go = bool(
        reduction >= REDUCTION_THRESHOLD
        and point["adaptive_high_frac"] <= HIGH_FRAC_THRESHOLD
        and oracle_ratio <= ORACLE_RATIO_THRESHOLD
        and ci_low > 0.0
    )
    return {
        "n_documents": len(doc_ids),
        "reduction_vs_always_low": reduction,
        "reduction_ci_low": float(ci_low),
        "reduction_ci_high": float(ci_high),
        "high_frac": point["adaptive_high_frac"],
        "oracle_ratio": oracle_ratio,
        "go_no_go": "GO" if go else "NO-GO",
    }


def main() -> None:
    all_results = {}
    for model_key, csv_path in MODELS.items():
        df = pd.read_csv(csv_path)
        doc_ids_sorted = sorted(df.doc_id.unique())
        calib_docs = doc_ids_sorted[:CALIB_SAMPLES]
        test_docs = doc_ids_sorted[CALIB_SAMPLES:]

        config = calibrate(df, calib_docs)
        test_result = evaluate_on_test(df, test_docs, config)

        print(f"\n=== {model_key} ===")
        print(f"calibrated config: {config}")
        print(f"test result: {test_result}")
        all_results[model_key] = {"config": config, "test_result": test_result}

    out_path = BASE / "adaptive_shadow_verify_controller_2026-07-20.json"
    out_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"\nsaved to {out_path}")

    print("\n=== Comparison to today's fixed-period=4 results ===")
    print("olmoe  fixed period=4: reduction=50.1% CI[42.1%,56.6%] high_frac=43.4% oracle_ratio=1.66x GO")
    print("llmjp  fixed period=4: reduction=47.0% CI[40.2%,53.4%] high_frac=40.3% oracle_ratio=1.26x NO-GO")
    for model_key, res in all_results.items():
        r = res["test_result"]
        print(f"{model_key}  adaptive: reduction={r['reduction_vs_always_low']*100:.1f}% "
              f"CI[{r['reduction_ci_low']*100:.1f}%,{r['reduction_ci_high']*100:.1f}%] "
              f"high_frac={r['high_frac']*100:.1f}% oracle_ratio={r['oracle_ratio']:.2f}x {r['go_no_go']}")


if __name__ == "__main__":
    main()
