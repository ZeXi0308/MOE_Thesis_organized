#!/usr/bin/env python3
"""Zero-new-GPU-time deepening of Idea B (precision-sufficiency shadow
verification controller): reuses the ALREADY-COLLECTED
``per_step_samples.csv`` from
``outputs/expert_precision_persistence_2026-07-20_{olmoe,llmjp}`` (produced by
``run_expert_precision_persistence_shadow_verify_p0.py`` on real RTX 5090 GPU
data) and re-runs the controller simulation with:

1. A finer verify-period grid {1,2,3,4,6,8,16} (original grid was {4,8,16}).
2. An escalate-quantile sensitivity sweep {0.6,0.7,0.75,0.8,0.9} (original
   was fixed at 0.75).
3. A GO-threshold sensitivity analysis (45%/50%/55% reduction bar) at the
   original period=4, quantile=0.75 operating point.

This is the exact "Task 1" cheap follow-up flagged in
``三个新创新方向_诚实验证结果_2026-07-20.md`` and ``Expert权重精度轴...md``:
"把 verify period 网格加细... 并把 escalate-quantile... 扫描... 使最终论文表述
不依赖单一任意阈值". No new model forward passes are run -- this is pure
pandas/numpy re-analysis of already-frozen per-step KL trajectories, so it
cannot leak (thresholds are still fit on calib docs, applied to test docs,
exactly as in the original script) and cannot introduce new GPU-measurement
confounds. It CAN, however, introduce a new multiple-comparisons confound
(scanning many (period, quantile) cells and reporting the best one would be
p-hacking) -- this script's output explicitly reports the FULL grid, not just
the best cell, and flags which original H2 threshold (50% reduction, <=50%
high_frac, <=2x oracle) each cell would have passed under.
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
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path("/Users/leandrozhao/Desktop/毕设论文资料/experiments/idea_a_mac/outputs")
MODELS = {
    "olmoe": BASE / "expert_precision_persistence_2026-07-20_olmoe",
    "llmjp": BASE / "expert_precision_persistence_2026-07-20_llmjp",
}
CALIB_SAMPLES = 12
N_BOOTSTRAP = 500
SEED = 20260720
PERIOD_GRID = [1, 2, 3, 4, 6, 8, 16]
QUANTILE_GRID = [0.6, 0.7, 0.75, 0.8, 0.9]
HIGH_FRAC_THRESHOLD = 0.50
ORACLE_RATIO_THRESHOLD = 2.0
REDUCTION_THRESHOLDS_FOR_SENSITIVITY = [0.45, 0.50, 0.55]


# ---------------------------------------------------------------------------
# Copied verbatim (pure numpy/pandas, no torch) from
# run_expert_precision_persistence_shadow_verify_p0.py so this script has
# zero dependency on the model/GPU stack.
# ---------------------------------------------------------------------------

def simulate_policies(kl_trajectory: np.ndarray, threshold: float, period: int) -> dict[str, float]:
    T = len(kl_trajectory)
    always_low_kl = float(kl_trajectory.sum())

    verify_mask = np.zeros(T, dtype=bool)
    verify_mask[::period] = True

    high_reactive = np.zeros(T, dtype=bool)
    t = 0
    while t < T:
        high_reactive[t] = True
        verified_kl = kl_trajectory[t]
        end = min(t + period, T)
        if verified_kl > threshold:
            high_reactive[t:end] = True
        t = end
    reactive_kl = float(kl_trajectory[~high_reactive].sum())
    reactive_high_frac = float(high_reactive.mean())

    high_oracle = kl_trajectory > threshold
    oracle_kl = float(kl_trajectory[~high_oracle].sum())
    oracle_high_frac = float(high_oracle.mean())

    return {
        "always_low_kl": always_low_kl,
        "reactive_kl": reactive_kl, "reactive_high_frac": reactive_high_frac,
        "oracle_kl": oracle_kl, "oracle_high_frac": oracle_high_frac,
    }


def controller_cell(
    df: pd.DataFrame, calib_docs: list[int], test_docs: list[int], period: int, quantile: float,
) -> dict[str, float]:
    calib_kl = df[df.doc_id.isin(calib_docs)]["kl"].to_numpy()
    threshold = float(np.quantile(calib_kl, quantile))

    per_doc: dict[int, dict[str, float]] = {}
    for doc_id in test_docs:
        traj = df[df.doc_id == doc_id].sort_values("step")["kl"].to_numpy()
        if len(traj) < period:
            continue
        per_doc[doc_id] = simulate_policies(traj, threshold, period)
    if not per_doc:
        return {}
    doc_ids = list(per_doc.keys())

    def aggregate(keys: list[int]) -> dict[str, float]:
        agg = {name: 0.0 for name in per_doc[doc_ids[0]]}
        for d in keys:
            for name, value in per_doc[d].items():
                if name.endswith("_high_frac"):
                    agg[name] += value / len(keys)
                else:
                    agg[name] += value
        return agg

    point = aggregate(doc_ids)
    reduction = 1.0 - point["reactive_kl"] / max(point["always_low_kl"], 1e-12)
    oracle_ratio = point["reactive_kl"] / max(point["oracle_kl"], 1e-9)

    rng = np.random.default_rng(SEED + period + int(quantile * 1000))
    boot = []
    for _ in range(N_BOOTSTRAP):
        chosen = rng.choice(doc_ids, size=len(doc_ids), replace=True)
        agg_b = aggregate(list(chosen))
        boot.append(1.0 - agg_b["reactive_kl"] / max(agg_b["always_low_kl"], 1e-12))
    ci_low, ci_high = np.quantile(boot, [0.025, 0.975])

    return {
        "period": period, "escalate_quantile": quantile, "threshold_tau": threshold,
        "n_documents": len(doc_ids),
        "reduction": reduction, "reduction_ci_low": float(ci_low), "reduction_ci_high": float(ci_high),
        "reactive_high_frac": point["reactive_high_frac"], "oracle_ratio": oracle_ratio,
    }


def run_for_model(model_key: str, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(out_dir / "per_step_samples.csv")
    doc_ids_ordered = sorted(df["doc_id"].unique().tolist())
    calib_docs = doc_ids_ordered[:CALIB_SAMPLES]
    test_docs = doc_ids_ordered[CALIB_SAMPLES:]

    grid_rows = []
    for period in PERIOD_GRID:
        for quantile in QUANTILE_GRID:
            cell = controller_cell(df, calib_docs, test_docs, period, quantile)
            if not cell:
                continue
            cell["model"] = model_key
            for r_thr in REDUCTION_THRESHOLDS_FOR_SENSITIVITY:
                cell[f"go_at_{int(r_thr*100)}pct"] = bool(
                    cell["reduction"] >= r_thr
                    and cell["reactive_high_frac"] <= HIGH_FRAC_THRESHOLD
                    and cell["oracle_ratio"] <= ORACLE_RATIO_THRESHOLD
                    and cell["reduction_ci_low"] > 0.0
                )
            grid_rows.append(cell)
    grid_df = pd.DataFrame(grid_rows)

    # Threshold-sensitivity table restricted to the ORIGINAL operating point
    # (period=4, quantile=0.75), varying only the reduction bar.
    orig_cell = grid_df[(grid_df.period == 4) & (grid_df.escalate_quantile == 0.75)]
    sens_rows = []
    for r_thr in REDUCTION_THRESHOLDS_FOR_SENSITIVITY:
        if orig_cell.empty:
            continue
        row = orig_cell.iloc[0]
        sens_rows.append({
            "model": model_key, "period": 4, "escalate_quantile": 0.75,
            "reduction_point_estimate": row["reduction"], "reduction_ci_low": row["reduction_ci_low"],
            "reduction_ci_high": row["reduction_ci_high"], "reduction_bar": r_thr,
            "go_no_go": "GO" if row[f"go_at_{int(r_thr*100)}pct"] else "NO-GO",
        })
    sens_df = pd.DataFrame(sens_rows)
    return grid_df, sens_df


def main() -> None:
    all_grid, all_sens = [], []
    for model_key, out_dir in MODELS.items():
        grid_df, sens_df = run_for_model(model_key, out_dir)
        all_grid.append(grid_df)
        all_sens.append(sens_df)

    grid = pd.concat(all_grid, ignore_index=True)
    sens = pd.concat(all_sens, ignore_index=True)

    out = BASE / "idea_b_finegrid_threshold_sensitivity_2026-07-20"
    out.mkdir(parents=True, exist_ok=True)
    grid.to_csv(out / "finegrid_results.csv", index=False)
    sens.to_csv(out / "threshold_sensitivity_results.csv", index=False)

    # Console summary: best period per model at the original quantile=0.75,
    # and a joint-robustness check (does the SAME period pass GO on BOTH
    # models at ANY quantile in the grid?).
    print("=== Full grid (period x quantile), quantile=0.75 slice ===")
    print(grid[grid.escalate_quantile == 0.75][
        ["model", "period", "reduction", "reduction_ci_low", "reactive_high_frac", "oracle_ratio"]
    ].to_string(index=False))

    print("\n=== Threshold sensitivity at period=4, quantile=0.75 ===")
    print(sens.to_string(index=False))

    print("\n=== Joint-robust cells: BOTH models GO at 45% bar, same (period, quantile) ===")
    piv = grid.pivot_table(index=["period", "escalate_quantile"], columns="model", values="go_at_45pct")
    joint = piv[(piv.get("olmoe") == True) & (piv.get("llmjp") == True)]
    print(joint if not joint.empty else "NONE")

    with (out / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump({
            "period_grid": PERIOD_GRID, "quantile_grid": QUANTILE_GRID,
            "reduction_thresholds_tested": REDUCTION_THRESHOLDS_FOR_SENSITIVITY,
            "high_frac_threshold": HIGH_FRAC_THRESHOLD, "oracle_ratio_threshold": ORACLE_RATIO_THRESHOLD,
            "note": "pure offline re-analysis of already-collected per_step_samples.csv, zero new GPU time",
        }, f, indent=2)
    print(f"\nsaved to {out}")


if __name__ == "__main__":
    main()
