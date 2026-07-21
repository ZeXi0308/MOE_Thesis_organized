"""Paired document-bootstrap survival tests for fixed-rate combine policies.

This analysis deliberately treats a WikiText article/request as the sampling
unit.  Tokens from one article are not bootstrapped as independent evidence.
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
import math
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True)
    p.add_argument(
        "--comparison",
        action="append",
        default=[],
        help="Paired test encoded as candidate:reference; may be repeated.",
    )
    p.add_argument(
        "--rank-gate-recovery",
        action="append",
        default=[],
        help="Recovery encoded as candidate:rank_reference:gate_reference.",
    )
    p.add_argument(
        "--oracle-recovery",
        action="append",
        default=[],
        help="Recovery encoded as deployable:baseline:oracle.",
    )
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--seed", type=int, default=20260714)
    return p.parse_args()


def _parse_spec(spec: str, expected: int) -> tuple[str, ...]:
    values = tuple(value.strip() for value in spec.split(":"))
    if len(values) != expected or any(not value for value in values):
        raise ValueError(f"expected {expected} colon-separated names, got {spec!r}")
    return values


def _aligned_rows(df: pd.DataFrame, strategies: tuple[str, ...]) -> list[pd.DataFrame]:
    rows = []
    sample_ids: set[int] | None = None
    for strategy in strategies:
        sub = df[df["strategy"] == strategy].sort_values("sample_id").reset_index(drop=True)
        if sub.empty:
            raise ValueError(f"strategy not found: {strategy}")
        current = set(int(v) for v in sub["sample_id"])
        sample_ids = current if sample_ids is None else sample_ids & current
        rows.append(sub)
    assert sample_ids is not None
    ordered = sorted(sample_ids)
    if not ordered:
        raise ValueError(f"no aligned samples for {strategies}")
    aligned = [
        sub[sub["sample_id"].isin(ordered)].sort_values("sample_id").reset_index(drop=True)
        for sub in rows
    ]
    if any(list(sub["sample_id"]) != ordered for sub in aligned):
        raise RuntimeError(f"sample alignment failed for {strategies}")
    return aligned


def _corpus_metrics(rows: pd.DataFrame, indices: np.ndarray) -> tuple[float, float]:
    chosen = rows.iloc[indices]
    tokens = max(float(chosen["token_count"].sum()), 1.0)
    return float(chosen["kl_sum"].sum() / tokens), math.exp(
        float(chosen["nll_sum"].sum() / tokens)
    )


def paired_bootstrap(
    candidate: pd.DataFrame,
    reference: pd.DataFrame,
    sampled_indices: np.ndarray,
) -> dict[str, float]:
    all_idx = np.arange(len(candidate))
    cand_kl, cand_ppl = _corpus_metrics(candidate, all_idx)
    ref_kl, ref_ppl = _corpus_metrics(reference, all_idx)
    kl_delta = np.empty(len(sampled_indices), dtype=np.float64)
    ppl_delta = np.empty(len(sampled_indices), dtype=np.float64)
    for boot_idx, indices in enumerate(sampled_indices):
        c_kl, c_ppl = _corpus_metrics(candidate, indices)
        r_kl, r_ppl = _corpus_metrics(reference, indices)
        kl_delta[boot_idx] = c_kl - r_kl
        ppl_delta[boot_idx] = c_ppl - r_ppl
    # Two-sided bootstrap sign p-value.  Holm correction is applied across all
    # explicitly requested comparisons below.
    nonpositive = int(np.count_nonzero(kl_delta <= 0))
    nonnegative = int(np.count_nonzero(kl_delta >= 0))
    # Add-one correction prevents impossible p=0 claims from a finite Monte
    # Carlo bootstrap.  Very small document counts are separately marked as
    # inference-invalid in the output.
    p_value = min(
        1.0,
        2.0 * (min(nonpositive, nonnegative) + 1) / (len(kl_delta) + 1),
    )
    return {
        "candidate_kl": cand_kl,
        "reference_kl": ref_kl,
        "kl_delta": cand_kl - ref_kl,
        "kl_delta_ci_low": float(np.quantile(kl_delta, 0.025)),
        "kl_delta_ci_high": float(np.quantile(kl_delta, 0.975)),
        "relative_kl_change": (cand_kl - ref_kl) / max(ref_kl, 1e-15),
        "ppl_delta": cand_ppl - ref_ppl,
        "ppl_delta_ci_low": float(np.quantile(ppl_delta, 0.025)),
        "ppl_delta_ci_high": float(np.quantile(ppl_delta, 0.975)),
        "bootstrap_sign_p": p_value,
        "inference_valid_n_ge_5": len(candidate) >= 5,
    }


def recovery_bootstrap(
    candidate: pd.DataFrame,
    baseline: pd.DataFrame,
    target: pd.DataFrame,
    sampled_indices: np.ndarray,
) -> dict[str, float]:
    all_idx = np.arange(len(candidate))

    def recovery(indices: np.ndarray) -> float:
        candidate_kl, _ = _corpus_metrics(candidate, indices)
        baseline_kl, _ = _corpus_metrics(baseline, indices)
        target_kl, _ = _corpus_metrics(target, indices)
        gap = baseline_kl - target_kl
        if gap <= 0:
            return float("nan")
        return (baseline_kl - candidate_kl) / gap

    point = recovery(all_idx)
    values = np.array([recovery(indices) for indices in sampled_indices])
    finite = values[np.isfinite(values)]
    return {
        "recovery": point,
        "recovery_ci_low": float(np.quantile(finite, 0.025)) if len(finite) else float("nan"),
        "recovery_ci_high": float(np.quantile(finite, 0.975)) if len(finite) else float("nan"),
        "valid_bootstrap_fraction": float(len(finite) / max(len(values), 1)),
    }


def _holm_adjust(rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    order = sorted(range(len(rows)), key=lambda idx: float(rows[idx]["bootstrap_sign_p"]))
    running = 0.0
    total = len(rows)
    for rank, idx in enumerate(order):
        adjusted = min(1.0, (total - rank) * float(rows[idx]["bootstrap_sign_p"]))
        running = max(running, adjusted)
        rows[idx]["holm_adjusted_p"] = running


def _markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "(none)"
    columns = list(df.columns)
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for _, row in df.iterrows():
        values = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                values.append(f"{value:.6g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    df = pd.read_csv(run_dir / "sample_metrics.csv")
    sample_count = int(df.groupby("strategy")["sample_id"].nunique().min())
    rng = np.random.default_rng(args.seed)
    sampled_indices = rng.integers(0, sample_count, size=(args.bootstrap, sample_count))

    comparison_rows: list[dict[str, object]] = []
    for spec in args.comparison:
        candidate_name, reference_name = _parse_spec(spec, 2)
        candidate, reference = _aligned_rows(df, (candidate_name, reference_name))
        row: dict[str, object] = {
            "candidate": candidate_name,
            "reference": reference_name,
        }
        row.update(paired_bootstrap(candidate, reference, sampled_indices))
        comparison_rows.append(row)
    _holm_adjust(comparison_rows)

    recovery_rows: list[dict[str, object]] = []
    for recovery_type, specs in (
        ("rank_to_gate", args.rank_gate_recovery),
        ("baseline_to_oracle", args.oracle_recovery),
    ):
        for spec in specs:
            candidate_name, baseline_name, target_name = _parse_spec(spec, 3)
            candidate, baseline, target = _aligned_rows(
                df, (candidate_name, baseline_name, target_name)
            )
            row = {
                "type": recovery_type,
                "candidate": candidate_name,
                "baseline": baseline_name,
                "target": target_name,
            }
            row.update(recovery_bootstrap(candidate, baseline, target, sampled_indices))
            recovery_rows.append(row)

    comparisons = pd.DataFrame(comparison_rows)
    recoveries = pd.DataFrame(recovery_rows)
    comparisons.to_csv(run_dir / "fixed_rate_survival_comparisons.csv", index=False)
    recoveries.to_csv(run_dir / "fixed_rate_survival_recovery.csv", index=False)
    report = f"""# Fixed-Rate Survival Analysis

- inference unit: document/request cluster
- aligned documents per strategy: {sample_count}
- paired bootstrap replicates: {args.bootstrap}
- multiple-comparison control: Holm adjustment over requested paired KL tests
- boundary: fake-quant quality analysis; no native codec, EP network, or latency result

## Paired comparisons

{_markdown(comparisons)}

## Gap recovery

Recovery is only defined in bootstrap replicates where the target KL is below the
baseline KL. A low `valid_bootstrap_fraction` means the denominator is unstable
and the recovery ratio must not be used as evidence.

{_markdown(recoveries)}
"""
    (run_dir / "fixed_rate_survival_report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
