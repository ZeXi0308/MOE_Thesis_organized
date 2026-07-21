#!/usr/bin/env python3
"""Per-request Quality Isolation, reconstructed as PREDICTOR-FREE quality-debt
fairness scheduling (VTC-like), replacing the fragility-PREDICTION mechanism
that the 2026-07-20 GPU rounds progressively falsified.

READ THIS BEFORE RUNNING OR CITING RESULTS.

Why prediction is retired here, not reused
---------------------------------------------
Round 2 (calibration/validation/test-split proxy audit) showed the
prefill-based fragility proxy only transfers within one model (LLM-jp) and is
already fragile to resampling (validation ρ often does not replicate on the
sealed test split). Round 4 (Prefill->DecodeFragility) then showed that EVEN
on LLM-jp, using the frozen proxy to predict FUTURE decode-time harm from the
SAME degradation family is NO-GO (all four actions' CI cross zero), and that
different actions' harm rankings correlate only 0.129-0.865 with each other --
i.e. there is no single stable "this request is fragile" label independent of
which degradation mechanism is applied. Any predictor built on this project's
existing features would inherit that instability.

This script does not predict anything. It only uses REALIZED, PAST harm
(exactly what a real deployed system can causally observe about its own
history) to decide who gets protected next -- a scheduling/fairness property,
not a forecasting property. Its correctness does not depend on any of the
falsified predictive claims above.

Data source and zero-new-GPU-time property
----------------------------------------------
Uses the ALREADY-COLLECTED ``decode_fragility_samples.csv`` (48 LLM-jp
documents x 4 real degradation actions: ``fp8top8_rest_int4``,
``rankk_drop_renorm``, ``keep12_drop_renorm``, ``keep8_drop_renorm``; real
GPU-measured per-step KL from ``run_decode_fragility_strict_gpu.py``). No new
model forward passes are run. A synthetic multi-round multi-tenant STREAM is
built by bootstrap-resampling (with replacement) these already-measured
per-document harm values -- the realism claim is scoped to "these are real
measured harm values for real documents under a real degradation mechanism";
the MULTI-ROUND ARRIVAL PROCESS itself is synthetic (the project has never
run a real multi-tenant streaming workload; see confound #1 below).

Policies compared (same total per-round degradation budget K = round(N *
budget_fraction) for every policy, so none of them can "win" by simply doing
less degradation):
  - random: K tenants chosen uniformly at random each round.
  - round_robin: a fixed rotating window of K tenants, no adaptivity to harm
    (matches COUNT exactly, not realized magnitude).
  - static_once: K tenants fixed once at round 0 and degraded EVERY round --
    the project's earlier static one-time-quota baseline (Part A/C style).
  - quality_debt (main proposal): every round, degrade the K tenants with the
    LOWEST cumulative realized harm so far (ties broken randomly). This is a
    VTC-like debt scheduler: it only ever reads its own past, never predicts,
    and actively equalizes accumulated harm rather than accumulated count.

Go/No-Go criterion (frozen before looking at results)
----------------------------------------------------------
For a given action, quality_debt is a GO if, across bootstrap trials, the
mean relative reduction in worst-tenant cumulative harm vs random is >= 20%
AND the 95% CI of that reduction excludes 0, AND the relative change in
TOTAL system harm (summed over all tenants) is within +/-3% of random's
(sanity check: fairness should not be bought by silently doing less total
degradation, which would be a scheduling artifact, not a fairness gain).

Known confounds most likely to make this look better than it would be in a
real deployment:
  1. The multi-round arrival process is synthetic (i.i.d. resampling with
     replacement from a 48-document pool every round). If real request
     streams have strong temporal correlation (the same physical
     tenant/session repeatedly hitting similarly-hard documents in a burst),
     the debt scheduler's advantage could be larger or smaller than reported
     here -- this script cannot tell which.
  2. Only 48 documents back the resampling pool; with replacement, this
     understates real-world harm-value diversity. Results should be treated
     as a shape/mechanism check, not a calibrated production number.
  3. round_robin's exact-count fairness is a strong baseline by construction
     for LOW-VARIANCE harm distributions; the debt scheduler's edge should
     mainly show up when the per-document harm distribution is heavy-tailed
     (check ``harm_skewness`` in metadata.json -- if it is close to a normal
     distribution's, a small edge over round_robin is expected and not a bug).
  4. Cross-action results in this script are run independently per action
     (matching the round-4 finding that fragility is action-specific); do not
     average across actions when reporting a single number.

Usage
-----
  python run_quality_debt_fairness_p0.py \\
      --samples-csv outputs/decode_fragility_strict_llmjp_2026-07-20/decode_fragility_samples.csv \\
      --output-dir outputs/quality_debt_fairness_2026-07-20
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

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--samples-csv", required=True)
    p.add_argument("--split", default="test", help="split label to evaluate on; falls back to all rows if too few")
    p.add_argument("--min-split-rows", type=int, default=8)
    p.add_argument("--num-tenants", type=int, default=12)
    p.add_argument("--num-rounds", type=int, default=200)
    p.add_argument("--budget-fraction", type=float, default=0.5)
    p.add_argument("--num-trials", type=int, default=500)
    p.add_argument("--cvar-fraction", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=20260720)
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


def resolve_pool(df: pd.DataFrame, split: str, min_rows: int) -> pd.DataFrame:
    if "split" not in df.columns:
        return df
    sub = df[df["split"] == split]
    if len(sub) >= min_rows:
        return sub
    print(f"WARNING: split={split!r} has only {len(sub)} rows (< {min_rows}); "
          f"falling back to ALL {len(df)} rows for this run.")
    return df


def find_actions(df: pd.DataFrame) -> list[str]:
    suffix = "__decode_mean_kl"
    return sorted(col[: -len(suffix)] for col in df.columns if col.endswith(suffix))


def simulate_trial(
    harm_values: np.ndarray, num_tenants: int, num_rounds: int, budget: int, rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """One bootstrap trial: draw a (num_rounds, num_tenants) harm matrix by
    resampling `harm_values` with replacement (this is the realized harm a
    tenant WOULD suffer IF degraded that round), then run all four policies
    against the SAME draw matrix so their comparison is paired, not noisier
    than necessary."""
    draw = rng.choice(harm_values, size=(num_rounds, num_tenants), replace=True)

    cum_random = np.zeros(num_tenants)
    cum_round_robin = np.zeros(num_tenants)
    cum_static = np.zeros(num_tenants)
    cum_debt = np.zeros(num_tenants)

    static_selected = rng.choice(num_tenants, size=budget, replace=False)
    rr_cursor = 0

    for t in range(num_rounds):
        round_harm = draw[t]

        sel_random = rng.choice(num_tenants, size=budget, replace=False)
        cum_random[sel_random] += round_harm[sel_random]

        rr_idx = (np.arange(rr_cursor, rr_cursor + budget)) % num_tenants
        cum_round_robin[rr_idx] += round_harm[rr_idx]
        rr_cursor = (rr_cursor + budget) % num_tenants

        cum_static[static_selected] += round_harm[static_selected]

        order = np.argsort(cum_debt + rng.uniform(0, 1e-9, size=num_tenants))
        sel_debt = order[:budget]
        cum_debt[sel_debt] += round_harm[sel_debt]

    return {
        "random": cum_random,
        "round_robin": cum_round_robin,
        "static_once": cum_static,
        "quality_debt": cum_debt,
    }


def summarize(cum: dict[str, np.ndarray], cvar_fraction: float) -> dict[str, float]:
    out: dict[str, float] = {}
    n = len(cum["random"])
    k_tail = max(1, int(round(n * cvar_fraction)))
    for policy, values in cum.items():
        sorted_desc = np.sort(values)[::-1]
        out[f"{policy}_worst"] = float(sorted_desc[0])
        out[f"{policy}_cvar"] = float(sorted_desc[:k_tail].mean())
        out[f"{policy}_total"] = float(values.sum())
    return out


def run_action(action: str, harm_values: np.ndarray, args: argparse.Namespace) -> dict[str, object]:
    budget = max(1, int(round(args.num_tenants * args.budget_fraction)))
    rng = np.random.default_rng(args.seed)
    rows = []
    for trial in range(args.num_trials):
        cum = simulate_trial(harm_values, args.num_tenants, args.num_rounds, budget, rng)
        rows.append(summarize(cum, args.cvar_fraction))
    df = pd.DataFrame(rows)

    def improvement(metric: str) -> tuple[float, float, float]:
        base = df[f"random_{metric}"].to_numpy()
        alt = df[f"quality_debt_{metric}"].to_numpy()
        rel = np.divide(base - alt, np.clip(base, 1e-12, None))
        return float(rel.mean()), float(np.quantile(rel, 0.025)), float(np.quantile(rel, 0.975))

    worst_mean, worst_lo, worst_hi = improvement("worst")
    cvar_mean, cvar_lo, cvar_hi = improvement("cvar")
    total_rel = float(((df["quality_debt_total"] - df["random_total"]) / df["random_total"].clip(lower=1e-12)).mean())

    go = bool(worst_mean >= 0.20 and worst_lo > 0.0 and abs(total_rel) <= 0.03)

    return {
        "action": action,
        "n_documents": int(len(harm_values)),
        "harm_mean": float(np.mean(harm_values)),
        "harm_skewness": float(pd.Series(harm_values).skew()),
        "policy_means": {
            col: float(df[col].mean()) for col in df.columns
        },
        "worst_tenant_relative_improvement_mean": worst_mean,
        "worst_tenant_relative_improvement_ci": [worst_lo, worst_hi],
        "cvar_relative_improvement_mean": cvar_mean,
        "cvar_relative_improvement_ci": [cvar_lo, cvar_hi],
        "total_system_harm_relative_change": total_rel,
        "go_no_go": "GO" if go else "NO-GO",
    }


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.samples_csv)
    pool = resolve_pool(df, args.split, args.min_split_rows)
    actions = find_actions(pool)
    if not actions:
        raise RuntimeError(f"no `<action>__decode_mean_kl` columns found in {args.samples_csv}")

    results = []
    for action in actions:
        col = f"{action}__decode_mean_kl"
        harm_values = pool[col].dropna().to_numpy(dtype=float)
        harm_values = np.clip(harm_values, 0.0, None)
        if len(harm_values) < 4:
            print(f"skip action={action}: only {len(harm_values)} usable rows")
            continue
        result = run_action(action, harm_values, args)
        results.append(result)
        print(f"[{action}] n={result['n_documents']} worst_improve={result['worst_tenant_relative_improvement_mean']:.3f} "
              f"CI={result['worst_tenant_relative_improvement_ci']} "
              f"cvar_improve={result['cvar_relative_improvement_mean']:.3f} "
              f"total_change={result['total_system_harm_relative_change']:.4f} "
              f"-> {result['go_no_go']}")

    (out / "metadata.json").write_text(json.dumps({
        "config": vars(args),
        "results": results,
        "evidence_boundary": (
            "Predictor-free debt scheduler over real measured per-document decode "
            "KL, resampled into a SYNTHETIC multi-round multi-tenant stream (no real "
            "streaming workload has been run in this project). Zero new GPU time."
        ),
    }, indent=2, default=str), encoding="utf-8")

    lines = ["# Predictor-Free Quality-Debt Fairness (VTC-like) -- Per-Action Results", ""]
    cols = ["action", "n_documents", "worst_tenant_relative_improvement_mean", "cvar_relative_improvement_mean",
            "total_system_harm_relative_change", "go_no_go"]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    for r in results:
        vals = [f"{r[c]:.4f}" if isinstance(r[c], float) else str(r[c]) for c in cols]
        lines.append("| " + " | ".join(vals) + " |")
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
