#!/usr/bin/env python3
"""Evaluate a frozen Quality Isolation proxy on untouched documents.

All feature groups and ridge alphas come from the exploratory run's validation
set. The exploratory test rows are excluded from fitting. The replication CSV
is treated entirely as one sealed test set; no model or threshold is selected
on it. ``post_prefill_all`` is the pre-declared primary proxy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_quality_isolation_proxy_strict import feature_groups
from run_quality_isolation_proxy_gpu_strict import (
    allocation_utility,
    auc_score,
    bootstrap_spearman_ci,
    recall_at_count,
    ridge_fit_predict,
    spearman,
    worst_fraction_labels,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exploratory-dir", required=True)
    parser.add_argument("--replication-dir", required=True)
    parser.add_argument("--primary-group", default="post_prefill_all")
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--permutations", type=int, default=10000)
    parser.add_argument("--random-trials", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260720)
    args = parser.parse_args()

    exploratory_dir = Path(args.exploratory_dir)
    replication_dir = Path(args.replication_dir)
    exploratory = pd.read_csv(exploratory_dir / "sample_features_labels.csv")
    replication = pd.read_csv(replication_dir / "sample_features_labels.csv")
    frozen_results = pd.read_csv(exploratory_dir / "proxy_results_augmented.csv")
    fit = exploratory[exploratory["split"].isin(["train", "validation"])].copy()
    if len(fit) == len(exploratory):
        raise ValueError("exploratory CSV has no held-out test rows to exclude")

    fit_y = np.log10(fit["label_mean_token_kl"].to_numpy(dtype=float) + 1e-12)
    test_y = replication["label_mean_token_kl"].to_numpy(dtype=float)
    worst_labels = worst_fraction_labels(test_y, 0.1)
    groups = feature_groups(exploratory)
    rows: list[dict] = []
    predictions: dict[str, np.ndarray] = {}
    for group_name, columns in groups.items():
        frozen_row = frozen_results[frozen_results["feature_group"] == group_name]
        if len(frozen_row) != 1:
            raise ValueError(f"missing unique frozen configuration for {group_name}")
        alpha = float(frozen_row.iloc[0]["selected_alpha_on_validation"])
        fit_x = fit[columns].to_numpy(dtype=float)
        test_x = replication[columns].to_numpy(dtype=float)
        _, prediction = ridge_fit_predict(
            fit_x,
            fit_y,
            fit_x[:1],
            test_x,
            alpha,
        )
        predictions[group_name] = prediction
        rho = spearman(test_y, prediction)
        ci_low, ci_high = bootstrap_spearman_ci(
            test_y,
            prediction,
            args.bootstrap,
            args.seed + len(rows),
        )
        rows.append(
            {
                "feature_group": group_name,
                "role": "primary" if group_name == args.primary_group else "control",
                "frozen_alpha": alpha,
                "fit_samples_exploratory_train_plus_validation": len(fit),
                "replication_test_samples": len(replication),
                "replication_spearman": rho,
                "replication_spearman_ci_low": ci_low,
                "replication_spearman_ci_high": ci_high,
                "replication_worst_decile_auc": auc_score(
                    worst_labels, prediction
                ),
                "replication_worst_decile_recall_at_10pct": recall_at_count(
                    test_y, prediction, 0.1
                ),
            }
        )

    if args.primary_group not in predictions:
        raise ValueError(f"unknown primary group: {args.primary_group}")
    primary_prediction = predictions[args.primary_group]
    observed = abs(spearman(test_y, primary_prediction))
    rng = np.random.default_rng(args.seed)
    exceedances = 0
    for _ in range(args.permutations):
        permuted = rng.permutation(test_y)
        if abs(spearman(permuted, primary_prediction)) >= observed:
            exceedances += 1
    permutation_p = (exceedances + 1) / (args.permutations + 1)

    allocation = allocation_utility(
        test_y,
        primary_prediction,
        replication["label_token_count"].to_numpy(dtype=int),
        [0.1, 0.25, 0.5],
        args.random_trials,
        args.seed,
    )
    allocation.insert(0, "primary_feature_group", args.primary_group)
    results = pd.DataFrame(rows)
    results.to_csv(replication_dir / "frozen_proxy_replication.csv", index=False)
    allocation.to_csv(
        replication_dir / "frozen_proxy_replication_allocation.csv", index=False
    )
    primary_row = results[results["feature_group"] == args.primary_group].iloc[0]
    summary = {
        "protocol": (
            "feature groups and alphas frozen from exploratory validation; "
            "fit excludes exploratory test; replication is test-only"
        ),
        "primary_feature_group": args.primary_group,
        "primary_replication_spearman": float(primary_row["replication_spearman"]),
        "primary_replication_spearman_ci_low": float(
            primary_row["replication_spearman_ci_low"]
        ),
        "primary_replication_spearman_ci_high": float(
            primary_row["replication_spearman_ci_high"]
        ),
        "primary_two_sided_permutation_p": permutation_p,
        "fit_samples": len(fit),
        "replication_test_samples": len(replication),
    }
    (replication_dir / "frozen_proxy_replication_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(results.to_string(index=False))
    print("\nPrimary allocation:")
    print(allocation.to_string(index=False))
    print("\nSummary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
