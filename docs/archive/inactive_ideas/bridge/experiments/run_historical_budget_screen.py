#!/usr/bin/env python3
"""Historical, non-sealed BRIDGE budget-allocation screen.

The runner fits only frozen configurations on the old exploratory train and
validation rows, then evaluates two deployable observation loci on fixed
historical targets. It never selects a feature group or ridge alpha on target
data. Results are a development screen, not formal scientific evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats


FEATURE_GROUPS = {
    "arrival_lexical": [
        "char_count",
        "token_count_input",
        "unique_token_ratio",
        "token_id_entropy",
        "adjacent_repeat_rate",
    ],
    "full_router_plus_lexical": [
        "char_count",
        "token_count_input",
        "unique_token_ratio",
        "token_id_entropy",
        "adjacent_repeat_rate",
        "full_route_top1_weight_mean",
        "full_route_top1_weight_std",
        "full_route_top1_top2_margin_mean",
        "full_route_tail_mass_mean",
        "full_route_routing_entropy_mean",
        "full_route_rank1_hhi_mean",
        "full_route_active_expert_fraction_mean",
        "full_route_same_id_adjacent_layer_rate",
    ],
    "prefill_nll_only": ["full_mean_nll"],
    "post_prefill_all": [
        "char_count",
        "token_count_input",
        "unique_token_ratio",
        "token_id_entropy",
        "adjacent_repeat_rate",
        "full_route_top1_weight_mean",
        "full_route_top1_weight_std",
        "full_route_top1_top2_margin_mean",
        "full_route_tail_mass_mean",
        "full_route_routing_entropy_mean",
        "full_route_rank1_hhi_mean",
        "full_route_active_expert_fraction_mean",
        "full_route_same_id_adjacent_layer_rate",
        "full_mean_nll",
    ],
}

PRIMARY = {
    "arrival_same_prompt": "arrival_lexical",
    "postprefill_future_decode": "post_prefill_all",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{name} missing columns: {missing}")


def validate_frame(frame: pd.DataFrame, feature_columns: list[str], label: str, name: str) -> None:
    require_columns(frame, ["sample_id", label, *feature_columns], name)
    if frame.empty:
        raise ValueError(f"{name} is empty")
    if frame["sample_id"].duplicated().any():
        raise ValueError(f"{name} has duplicate sample_id")
    values = frame[[label, *feature_columns]].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains non-finite values")
    if (frame[label].to_numpy(dtype=float) < 0).any():
        raise ValueError(f"{name} contains negative harm")


def ridge_predict(
    fit_x: np.ndarray,
    fit_y: np.ndarray,
    target_x: np.ndarray,
    alpha: float,
) -> np.ndarray:
    mean = fit_x.mean(axis=0)
    std = fit_x.std(axis=0)
    std[std < 1e-12] = 1.0
    fit_z = (fit_x - mean) / std
    target_z = (target_x - mean) / std
    fit_design = np.column_stack([np.ones(len(fit_z)), fit_z])
    target_design = np.column_stack([np.ones(len(target_z)), target_z])
    if alpha < 0:
        raise ValueError("ridge alpha must be non-negative")
    # Solve the augmented least-squares system directly instead of forming
    # X.T @ X. Several frozen feature groups contain constant/collinear
    # columns; normal equations squared the condition number and emitted
    # overflow/invalid warnings under Accelerate even when inputs were finite.
    ridge_rows = np.eye(fit_design.shape[1], dtype=float) * math.sqrt(float(alpha))
    ridge_rows[0, 0] = 0.0
    augmented_x = np.vstack([fit_design, ridge_rows])
    augmented_y = np.concatenate([fit_y, np.zeros(fit_design.shape[1], dtype=float)])
    beta, _, _, _ = np.linalg.lstsq(augmented_x, augmented_y, rcond=None)
    prediction = target_design @ beta
    if not np.isfinite(prediction).all():
        raise ValueError("ridge solver produced non-finite prediction")
    return prediction


def frozen_prediction(
    fit: pd.DataFrame,
    target: pd.DataFrame,
    frozen: pd.DataFrame,
    group: str,
) -> tuple[np.ndarray, float]:
    if group not in FEATURE_GROUPS:
        raise ValueError(f"unknown feature group: {group}")
    row = frozen[frozen["feature_group"] == group]
    if len(row) != 1:
        raise ValueError(f"expected one frozen row for {group}, found {len(row)}")
    alpha = float(row.iloc[0]["selected_alpha_on_validation"])
    columns = FEATURE_GROUPS[group]
    validate_frame(fit, columns, "label_mean_token_kl", "fit")
    require_columns(target, columns, "target")
    fit_x = fit[columns].to_numpy(dtype=float)
    fit_y = np.log10(fit["label_mean_token_kl"].to_numpy(dtype=float) + 1e-12)
    target_x = target[columns].to_numpy(dtype=float)
    prediction = ridge_predict(fit_x, fit_y, target_x, alpha)
    if not np.isfinite(prediction).all():
        raise ValueError(f"non-finite prediction for {group}")
    return prediction, alpha


def upper_cvar(values: np.ndarray, tail_fraction: float = 0.10) -> float:
    if len(values) == 0:
        raise ValueError("cannot compute CVaR of empty values")
    count = max(1, int(math.ceil(len(values) * tail_fraction)))
    return float(np.sort(values)[-count:].mean())


def metric_value(values: np.ndarray, metric: str) -> float:
    if metric == "cvar90":
        return upper_cvar(values, 0.10)
    if metric == "p95":
        return float(np.quantile(values, 0.95))
    if metric == "mean":
        return float(np.mean(values))
    raise ValueError(f"unknown metric: {metric}")


def protected_values(harm: np.ndarray, selected: np.ndarray) -> np.ndarray:
    result = harm.copy()
    result[selected] = 0.0
    return result


def selected_count(n: int, budget: float) -> int:
    if not 0 < budget < 1:
        raise ValueError("budget must be strictly between 0 and 1")
    return max(1, min(n - 1, int(math.floor(n * budget))))


def top_indices(scores: np.ndarray, count: int) -> np.ndarray:
    # Stable tie handling makes the decision reproducible by sample order.
    return np.argsort(-scores, kind="mergesort")[:count]


def random_metric_mean(
    harm: np.ndarray,
    count: int,
    metric: str,
    trials: int,
    rng: np.random.Generator,
) -> float:
    values = []
    for _ in range(trials):
        selected = rng.choice(len(harm), size=count, replace=False)
        values.append(metric_value(protected_values(harm, selected), metric))
    return float(np.mean(values))


def bootstrap_relative_reduction(
    harm: np.ndarray,
    scores: np.ndarray,
    budget: float,
    metric: str,
    repeats: int,
    random_trials_per_repeat: int,
    seed: int,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    reductions = []
    for _ in range(repeats):
        sampled = rng.integers(0, len(harm), size=len(harm))
        sample_harm = harm[sampled]
        sample_scores = scores[sampled]
        count = selected_count(len(sample_harm), budget)
        predicted = metric_value(
            protected_values(sample_harm, top_indices(sample_scores, count)), metric
        )
        random_mean = random_metric_mean(
            sample_harm,
            count,
            metric,
            random_trials_per_repeat,
            rng,
        )
        reduction = (random_mean - predicted) / max(random_mean, 1e-12)
        reductions.append(float(reduction))
    return (
        float(np.mean(reductions)),
        float(np.quantile(reductions, 0.025)),
        float(np.quantile(reductions, 0.975)),
    )


def evaluate_policy(
    locus: str,
    group: str,
    harm: np.ndarray,
    scores: np.ndarray,
    budgets: list[float],
    point_random_trials: int,
    bootstrap: int,
    bootstrap_random_trials: int,
    seed: int,
) -> list[dict]:
    rows = []
    rho = float(stats.spearmanr(harm, scores).statistic)
    if not np.isfinite(rho):
        rho = 0.0
    for budget_index, budget in enumerate(budgets):
        count = selected_count(len(harm), budget)
        predicted_indices = top_indices(scores, count)
        oracle_indices = top_indices(harm, count)
        for metric_index, metric in enumerate(["cvar90", "p95", "mean"]):
            rng = np.random.default_rng(seed + budget_index * 100 + metric_index)
            baseline = metric_value(harm, metric)
            predicted = metric_value(protected_values(harm, predicted_indices), metric)
            oracle = metric_value(protected_values(harm, oracle_indices), metric)
            random_mean = random_metric_mean(
                harm, count, metric, point_random_trials, rng
            )
            numerator = random_mean - predicted
            denominator = random_mean - oracle
            recovery = numerator / denominator if denominator > 1e-12 else 0.0
            boot_mean, boot_low, boot_high = bootstrap_relative_reduction(
                harm,
                scores,
                budget,
                metric,
                bootstrap,
                bootstrap_random_trials,
                seed + budget_index * 1000 + metric_index * 10000,
            )
            rows.append(
                {
                    "locus": locus,
                    "feature_group": group,
                    "is_primary": bool(PRIMARY.get(locus) == group),
                    "n_documents": len(harm),
                    "budget_fraction": budget,
                    "protected_count": count,
                    "metric": metric,
                    "spearman": rho,
                    "unprotected_metric": baseline,
                    "predicted_metric": predicted,
                    "random_metric_mean": random_mean,
                    "oracle_metric": oracle,
                    "relative_reduction_vs_random": numerator / max(random_mean, 1e-12),
                    "relative_reduction_bootstrap_mean": boot_mean,
                    "relative_reduction_ci_low": boot_low,
                    "relative_reduction_ci_high": boot_high,
                    "oracle_headroom_recovery": recovery,
                }
            )
    return rows


def locus_pass(frame: pd.DataFrame, locus: str) -> tuple[bool, list[float]]:
    primary_group = PRIMARY[locus]
    rows = frame[
        (frame["locus"] == locus)
        & (frame["feature_group"] == primary_group)
        & (frame["metric"] == "cvar90")
    ].sort_values("budget_fraction")
    passing = rows[
        (rows["relative_reduction_ci_low"] >= 0.10)
        & (rows["oracle_headroom_recovery"] >= 0.30)
    ]["budget_fraction"].tolist()
    adjacent = any(
        math.isclose(right - left, 0.15, abs_tol=1e-9)
        or math.isclose(right - left, 0.25, abs_tol=1e-9)
        for left, right in zip(passing[:-1], passing[1:])
    )
    predicted = rows["predicted_metric"].to_numpy(dtype=float)
    monotone = bool(np.all(np.diff(predicted) <= 1e-12))
    # With the frozen {0.10, 0.25, 0.50} grid, any two consecutive passing
    # entries in sorted order are the predeclared adjacent budget points.
    return bool(len(passing) >= 2 and adjacent and monotone), [float(x) for x in passing]


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def render_report(decision: dict, primary_rows: pd.DataFrame) -> str:
    lines = [
        "# BRIDGE Historical Budget Screen",
        "",
        f"Verdict: **{decision['verdict']}**",
        "",
        "> Historical, already-viewed data. This is not sealed scientific evidence.",
        "",
        "## Primary CVaR90 results",
        "",
        "| locus | budget | LCB vs random | oracle recovery | pass point |",
        "|---|---:|---:|---:|---|",
    ]
    for row in primary_rows.itertuples(index=False):
        point_pass = (
            row.relative_reduction_ci_low >= 0.10
            and row.oracle_headroom_recovery >= 0.30
        )
        lines.append(
            f"| {row.locus} | {row.budget_fraction:.2f} | "
            f"{row.relative_reduction_ci_low:.3f} | "
            f"{row.oracle_headroom_recovery:.3f} | "
            f"{'yes' if point_pass else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## Gate interpretation",
            "",
            f"- arrival_same_prompt: {decision['loci']['arrival_same_prompt']['status']}",
            f"- postprefill_future_decode: {decision['loci']['postprefill_future_decode']['status']}",
            "- A positive historical screen only authorizes a fresh temporal-tail protocol.",
            "- It does not authorize online, latency, energy, topology, or network claims.",
            "",
        ]
    )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict:
    paths = {
        "exploratory": Path(args.exploratory_csv),
        "frozen": Path(args.frozen_proxy_csv),
        "replication": Path(args.replication_csv),
        "decode": Path(args.decode_csv),
    }
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"{name} input not found: {path}")

    exploratory = pd.read_csv(paths["exploratory"])
    frozen = pd.read_csv(paths["frozen"])
    replication = pd.read_csv(paths["replication"])
    decode = pd.read_csv(paths["decode"])
    require_columns(exploratory, ["split", "sample_id", "label_mean_token_kl"], "exploratory")
    fit = exploratory[exploratory["split"].isin(["train", "validation"])].copy()
    if len(fit) == len(exploratory):
        raise ValueError("exploratory input has no excluded test rows")
    source_ids = set(fit["sample_id"].astype(int))
    for target_name, target in [("replication", replication), ("decode", decode)]:
        target_ids = set(target["sample_id"].astype(int))
        overlap = sorted(source_ids & target_ids)
        if overlap:
            raise ValueError(f"fit/target sample_id overlap in {target_name}: {overlap[:5]}")

    targets = {
        "arrival_same_prompt": (replication, "label_mean_token_kl"),
        "postprefill_future_decode": (
            decode,
            "fp8top8_rest_int4__decode_mean_kl",
        ),
    }
    rows = []
    selected_alphas = {}
    for locus, (target, label) in targets.items():
        groups = list(FEATURE_GROUPS)
        for group_index, group in enumerate(groups):
            validate_frame(target, FEATURE_GROUPS[group], label, locus)
            prediction, alpha = frozen_prediction(fit, target, frozen, group)
            selected_alphas[f"{locus}:{group}"] = alpha
            rows.extend(
                evaluate_policy(
                    locus,
                    group,
                    target[label].to_numpy(dtype=float),
                    prediction,
                    args.budgets,
                    args.random_trials,
                    args.bootstrap,
                    args.bootstrap_random_trials,
                    args.seed + len(rows) * 37 + group_index,
                )
            )

    results = pd.DataFrame(rows)
    locus_decisions = {}
    for locus in PRIMARY:
        passed, passing_budgets = locus_pass(results, locus)
        locus_decisions[locus] = {
            "status": "PASS" if passed else "FAIL",
            "passing_budgets": passing_budgets,
        }
    passed_count = sum(value["status"] == "PASS" for value in locus_decisions.values())
    if passed_count == 2:
        verdict = "HISTORICAL_SCREEN_GO_NEEDS_FRESH_SEALED"
    elif passed_count == 1:
        verdict = "PARTIAL_NEEDS_TEMPORAL_FIRST_CHUNK_LABEL"
    else:
        verdict = "NO_GO_FOR_CURRENT_FROZEN_SIGNALS"
    decision = {
        "verdict": verdict,
        "formal_scientific_result": False,
        "evidence_boundary": "historical already-viewed development screen",
        "loci": locus_decisions,
        "budgets": args.budgets,
        "primary_metric": "cvar90",
        "pass_lcb": 0.10,
        "pass_oracle_recovery": 0.30,
        "seed": args.seed,
        "selected_alphas": selected_alphas,
    }

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    results_csv = results.to_csv(index=False)
    primary_rows = results[
        results["is_primary"] & (results["metric"] == "cvar90")
    ].sort_values(["locus", "budget_fraction"])
    manifest = {
        "inputs": {
            name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for name, path in paths.items()
        },
        "runner_sha256": sha256_file(Path(__file__)),
    }
    atomic_write_text(output / "allocation_results.csv", results_csv)
    atomic_write_text(output / "decision.json", json.dumps(decision, indent=2) + "\n")
    atomic_write_text(output / "source_manifest.json", json.dumps(manifest, indent=2) + "\n")
    atomic_write_text(output / "report.md", render_report(decision, primary_rows))
    return decision


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exploratory-csv", required=True)
    parser.add_argument("--frozen-proxy-csv", required=True)
    parser.add_argument("--replication-csv", required=True)
    parser.add_argument("--decode-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--budgets", nargs="+", type=float, default=[0.10, 0.25, 0.50])
    parser.add_argument("--random-trials", type=int, default=10000)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--bootstrap-random-trials", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2026072302)
    return parser.parse_args()


if __name__ == "__main__":
    decision = run(parse_args())
    print(json.dumps(decision, indent=2))
