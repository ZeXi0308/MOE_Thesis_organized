#!/usr/bin/env python3
"""Per-request Quality Isolation P0: does giving repeatedly-degraded requests
priority access to a limited full-precision quota reduce worst-case (P95/max)
per-document KL, at a MATCHED total byte budget, relative to allocating that
same quota randomly/uniformly?

This candidate has no independent host mechanism of its own (it was designed
as "the second contribution" riding on top of Graceful-EP, which is KILLED).
Rather than inventing a brand-new degradation mechanism from scratch, this P0
reuses the project's existing, already-validated uniform degradation policy
(`fixed_tail4`/`fixed_tail8` from the layer_budget experiments) as the host
"base policy", and asks the isolation-specific question on top of it: given a
fixed quota of M documents (out of N) that can be spared full precision, does
picking WHICH M to spare based on quality-risk beat picking them randomly?

Two variants:
  (A) Oracle ceiling -- pick the M highest-KL-under-base-policy documents
      using the TEST LABELS DIRECTLY (an upper bound on what a perfect
      "quality debt" credit signal could achieve, same style as MassCover-EP's
      oracle_cvar ceiling analysis).
  (B) Causally valid transfer signal -- since a real system cannot see the
      test-time KL before deciding, check whether a document's KL-riskiness
      under one degradation mechanism (e.g. kl_profile_3_5) is correlated
      with its KL-riskiness under the base mechanism (fixed_tail). If Spearman
      correlation is high, that other mechanism's cheap diagnostic could serve
      as the real-world credit signal; if near zero, quality isolation has no
      feasible signal on this data (same class of negative result as
      TokenRace-EP's adaptive-trigger free-signal test).

Evidence tag: [Observed], pure re-analysis of already-collected
sample_metrics.csv from the pre-registered layer_budget experiments -- zero
new GPU time, zero new statistical assumptions beyond what MassCover-EP /
TokenRace-EP already used in this project.
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
from scipy import stats


def paired_bootstrap_ci(values: np.ndarray, n_boot: int, seed: int, alpha: float = 0.05):
    rng = np.random.default_rng(seed)
    n = len(values)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot[b] = np.percentile(values[idx], 95)
    return float(np.quantile(boot, alpha / 2)), float(np.quantile(boot, 1 - alpha / 2))


def oracle_vs_random_isolation(
    base_kl: pd.Series, quota_fracs: list[float], n_boot: int, seed: int
) -> pd.DataFrame:
    n = len(base_kl)
    rows = []
    baseline_p95 = float(np.percentile(base_kl.to_numpy(), 95))
    baseline_max = float(base_kl.max())
    baseline_mean = float(base_kl.mean())
    for frac in quota_fracs:
        m = max(1, int(round(n * frac)))
        sorted_ids = base_kl.sort_values(ascending=False).index.tolist()
        oracle_upgrade = set(sorted_ids[:m])
        oracle_kl = base_kl.copy()
        oracle_kl.loc[list(oracle_upgrade)] = 0.0

        rng = np.random.default_rng(seed)
        random_p95s, random_means = [], []
        for _ in range(500):
            upgrade = set(rng.choice(base_kl.index.to_numpy(), size=m, replace=False))
            kl = base_kl.copy()
            kl.loc[list(upgrade)] = 0.0
            random_p95s.append(float(np.percentile(kl.to_numpy(), 95)))
            random_means.append(float(kl.mean()))

        rows.append({
            "quota_frac": frac,
            "quota_m": m,
            "baseline_p95_kl": baseline_p95,
            "baseline_max_kl": baseline_max,
            "baseline_mean_kl": baseline_mean,
            "oracle_p95_kl": float(np.percentile(oracle_kl.to_numpy(), 95)),
            "oracle_mean_kl": float(oracle_kl.mean()),
            "random_p95_kl_mean": float(np.mean(random_p95s)),
            "random_p95_kl_std": float(np.std(random_p95s)),
            "random_mean_kl_mean": float(np.mean(random_means)),
            "oracle_p95_reduction_pct": 100 * (1 - np.percentile(oracle_kl.to_numpy(), 95) / max(baseline_p95, 1e-12)),
            "random_p95_reduction_pct": 100 * (1 - np.mean(random_p95s) / max(baseline_p95, 1e-12)),
            "oracle_advantage_over_random_pp": 100 * (
                (1 - np.percentile(oracle_kl.to_numpy(), 95) / max(baseline_p95, 1e-12))
                - (1 - np.mean(random_p95s) / max(baseline_p95, 1e-12))
            ),
        })
    return pd.DataFrame(rows)


def proxy_realizable_isolation(
    df: pd.DataFrame, base_strategy: str, proxy_strategy: str, quota_fracs: list[float]
) -> pd.DataFrame:
    """Realistic (non-oracle) variant: rank documents by their KL under a
    DIFFERENT, already-computed diagnostic mechanism (`proxy_strategy`), then
    spend the upgrade quota on the top-ranked documents by that proxy. This is
    causally valid (the proxy signal does not use the base-policy's own
    test-time outcome) and directly tests how much of the oracle ceiling in
    Part A survives once the credit signal is realistic."""
    base_kl = df[df["strategy"] == base_strategy].set_index("sample_id")["mean_token_kl"]
    proxy_kl = df[df["strategy"] == proxy_strategy].set_index("sample_id")["mean_token_kl"]
    common = base_kl.index.intersection(proxy_kl.index)
    base_kl = base_kl.loc[common]
    proxy_rank = proxy_kl.loc[common].sort_values(ascending=False).index.tolist()
    n = len(base_kl)
    baseline_p95 = float(np.percentile(base_kl.to_numpy(), 95))
    rows = []
    for frac in quota_fracs:
        m = max(1, int(round(n * frac)))
        upgrade = set(proxy_rank[:m])
        kl = base_kl.copy()
        kl.loc[list(upgrade)] = 0.0
        rows.append({
            "quota_frac": frac,
            "quota_m": m,
            "proxy_signal_strategy": proxy_strategy,
            "proxy_realized_p95_kl": float(np.percentile(kl.to_numpy(), 95)),
            "proxy_realized_p95_reduction_pct": 100 * (1 - np.percentile(kl.to_numpy(), 95) / max(baseline_p95, 1e-12)),
        })
    return pd.DataFrame(rows)


def cross_mechanism_transfer(df: pd.DataFrame, base_strategy: str, other_strategies: list[str]) -> pd.DataFrame:
    base = df[df["strategy"] == base_strategy].set_index("sample_id")["mean_token_kl"]
    rows = []
    for other in other_strategies:
        other_kl = df[df["strategy"] == other].set_index("sample_id")["mean_token_kl"]
        common = base.index.intersection(other_kl.index)
        if len(common) < 5:
            continue
        rho, p = stats.spearmanr(base.loc[common], other_kl.loc[common])
        rows.append({
            "base_strategy": base_strategy,
            "candidate_signal_strategy": other,
            "n_docs": len(common),
            "spearman_rho": float(rho),
            "p_value": float(p),
        })
    return pd.DataFrame(rows)


def run_model(model_key: str, csv_path: Path, base_strategy: str, quota_fracs: list[float],
              n_boot: int, seed: int) -> dict:
    df = pd.read_csv(csv_path)
    strategies = df["strategy"].unique().tolist()
    base_kl = df[df["strategy"] == base_strategy].set_index("sample_id")["mean_token_kl"]
    iso = oracle_vs_random_isolation(base_kl, quota_fracs, n_boot, seed)
    iso.insert(0, "model", model_key)
    iso.insert(1, "base_strategy", base_strategy)

    other_strategies = [s for s in strategies if s not in (base_strategy, "full")]
    transfer = cross_mechanism_transfer(df, base_strategy, other_strategies)
    transfer.insert(0, "model", model_key)

    # use the strongest-correlated proxy from Part B as the realistic signal
    best_proxy = transfer.sort_values("spearman_rho", ascending=False).iloc[0]["candidate_signal_strategy"]
    proxy_df = proxy_realizable_isolation(df, base_strategy, best_proxy, quota_fracs)
    proxy_df.insert(0, "model", model_key)
    proxy_df.insert(1, "base_strategy", base_strategy)
    iso = iso.merge(proxy_df[["quota_frac", "proxy_signal_strategy", "proxy_realized_p95_kl",
                               "proxy_realized_p95_reduction_pct"]], on="quota_frac", how="left")

    return {"isolation": iso, "transfer": transfer}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--olmoe-csv", required=True)
    ap.add_argument("--olmoe-base-strategy", default="fixed_tail4")
    ap.add_argument("--llmjp-csv", required=True)
    ap.add_argument("--llmjp-base-strategy", default="fixed_tail8")
    ap.add_argument("--quota-fracs", type=float, nargs="+", default=[0.1, 0.15, 0.25, 0.35, 0.5])
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260720)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    r_o = run_model("olmoe", Path(args.olmoe_csv), args.olmoe_base_strategy, args.quota_fracs, args.n_boot, args.seed)
    r_l = run_model("llmjp", Path(args.llmjp_csv), args.llmjp_base_strategy, args.quota_fracs, args.n_boot, args.seed + 1)

    iso_all = pd.concat([r_o["isolation"], r_l["isolation"]], ignore_index=True)
    transfer_all = pd.concat([r_o["transfer"], r_l["transfer"]], ignore_index=True)
    iso_all.to_csv(out / "isolation_oracle_vs_random.csv", index=False)
    transfer_all.to_csv(out / "cross_mechanism_transfer.csv", index=False)

    lines = ["# Per-request Quality Isolation P0", "",
              "## Part A: oracle-vs-random upgrade quota ceiling (worst-case KL reduction)", ""]
    cols_a = ["model", "base_strategy", "quota_frac", "quota_m", "baseline_p95_kl",
              "oracle_p95_kl", "random_p95_kl_mean", "oracle_p95_reduction_pct",
              "random_p95_reduction_pct", "oracle_advantage_over_random_pp",
              "proxy_signal_strategy", "proxy_realized_p95_kl", "proxy_realized_p95_reduction_pct"]
    lines.append("| " + " | ".join(cols_a) + " |")
    lines.append("|" + "|".join(["---"] * len(cols_a)) + "|")
    for _, row in iso_all.iterrows():
        vals = [f"{row[c]:.4f}" if isinstance(row[c], float) else str(row[c]) for c in cols_a]
        lines.append("| " + " | ".join(vals) + " |")

    lines.append("")
    lines.append("## Part B: is document KL-riskiness transferable across degradation mechanisms "
                 "(the causally-valid signal a real credit system would need)?")
    lines.append("")
    cols_b = ["model", "base_strategy", "candidate_signal_strategy", "n_docs", "spearman_rho", "p_value"]
    lines.append("| " + " | ".join(cols_b) + " |")
    lines.append("|" + "|".join(["---"] * len(cols_b)) + "|")
    for _, row in transfer_all.iterrows():
        vals = [f"{row[c]:.4f}" if isinstance(row[c], float) else str(row[c]) for c in cols_b]
        lines.append("| " + " | ".join(vals) + " |")

    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nsaved to {out}")


if __name__ == "__main__":
    main()
