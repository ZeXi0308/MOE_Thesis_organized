#!/usr/bin/env python3
"""Add proxy ablations and cross-model transfer to strict Quality Isolation runs."""
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
from types import SimpleNamespace

import numpy as np
import pandas as pd

from run_quality_isolation_proxy_gpu_strict import (
    auc_score,
    bootstrap_spearman_ci,
    evaluate_proxies,
    recall_at_count,
    ridge_fit_predict,
    spearman,
    worst_fraction_labels,
)


def feature_groups(samples: pd.DataFrame) -> dict[str, list[str]]:
    lexical = [
        "char_count",
        "token_count_input",
        "unique_token_ratio",
        "token_id_entropy",
        "adjacent_repeat_rate",
    ]
    full_route = [
        column for column in samples.columns if column.startswith("full_route_")
    ]
    return {
        "arrival_lexical": lexical,
        "full_router_plus_lexical": lexical + full_route,
        "prefill_nll_only": ["full_mean_nll"],
        "prefill_nll_plus_lexical": lexical + ["full_mean_nll"],
        "post_prefill_all": lexical + full_route + ["full_mean_nll"],
    }


def cross_model_transfer(
    source: pd.DataFrame,
    target: pd.DataFrame,
    source_name: str,
    target_name: str,
    train_samples: int,
    validation_samples: int,
    bootstrap: int,
    seed: int,
) -> list[dict]:
    train_end = train_samples
    validation_end = train_samples + validation_samples
    source_y = np.log10(source["label_mean_token_kl"].to_numpy(dtype=float) + 1e-12)
    target_test_y = target["label_mean_token_kl"].to_numpy(dtype=float)[validation_end:]
    rows: list[dict] = []
    for group_name, columns in feature_groups(source).items():
        source_matrix = source[columns].to_numpy(dtype=float)
        target_matrix = target[columns].to_numpy(dtype=float)
        best_alpha = None
        best_validation = -float("inf")
        for alpha in [0.0, 0.1, 1.0, 10.0, 100.0]:
            validation_prediction, _ = ridge_fit_predict(
                source_matrix[:train_end],
                source_y[:train_end],
                source_matrix[train_end:validation_end],
                target_matrix[validation_end:],
                alpha,
            )
            value = spearman(
                source_y[train_end:validation_end],
                validation_prediction,
            )
            if value > best_validation:
                best_validation = value
                best_alpha = alpha
        assert best_alpha is not None
        source_fit_x = source_matrix[:validation_end]
        source_fit_y = source_y[:validation_end]
        _, target_prediction = ridge_fit_predict(
            source_fit_x,
            source_fit_y,
            source_fit_x[:1],
            target_matrix[validation_end:],
            best_alpha,
        )
        target_rho = spearman(target_test_y, target_prediction)
        ci_low, ci_high = bootstrap_spearman_ci(
            target_test_y,
            target_prediction,
            bootstrap,
            seed + len(rows),
        )
        worst_labels = worst_fraction_labels(target_test_y, 0.1)
        rows.append(
            {
                "source_model": source_name,
                "target_model": target_name,
                "feature_group": group_name,
                "selected_alpha_on_source_validation": best_alpha,
                "source_validation_spearman": best_validation,
                "target_test_spearman": target_rho,
                "target_test_spearman_ci_low": ci_low,
                "target_test_spearman_ci_high": ci_high,
                "target_worst_decile_auc": auc_score(
                    worst_labels, target_prediction
                ),
                "target_worst_decile_recall_at_10pct": recall_at_count(
                    target_test_y, target_prediction, 0.1
                ),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--olmoe-dir", required=True)
    parser.add_argument("--llmjp-dir", required=True)
    parser.add_argument("--train-samples", type=int, default=48)
    parser.add_argument("--validation-samples", type=int, default=16)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--random-trials", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260720)
    args = parser.parse_args()

    directories = {
        "olmoe": Path(args.olmoe_dir),
        "llmjp": Path(args.llmjp_dir),
    }
    samples_by_model: dict[str, pd.DataFrame] = {}
    selections: dict[str, dict] = {}
    analysis_args = SimpleNamespace(
        train_samples=args.train_samples,
        validation_samples=args.validation_samples,
        bootstrap=args.bootstrap,
        random_trials=args.random_trials,
        seed=args.seed,
    )
    for model, directory in directories.items():
        samples = pd.read_csv(directory / "sample_features_labels.csv")
        samples_by_model[model] = samples
        proxies, allocation, selection = evaluate_proxies(samples, analysis_args)
        proxies.to_csv(directory / "proxy_results_augmented.csv", index=False)
        allocation.to_csv(
            directory / "selected_proxy_allocation_augmented.csv", index=False
        )
        selections[model] = selection
        print(f"\n{model} augmented proxies")
        print(proxies.to_string(index=False))
        print(allocation.to_string(index=False))

    transfer_rows = cross_model_transfer(
        samples_by_model["olmoe"],
        samples_by_model["llmjp"],
        "olmoe",
        "llmjp",
        args.train_samples,
        args.validation_samples,
        args.bootstrap,
        args.seed,
    )
    transfer_rows.extend(
        cross_model_transfer(
            samples_by_model["llmjp"],
            samples_by_model["olmoe"],
            "llmjp",
            "olmoe",
            args.train_samples,
            args.validation_samples,
            args.bootstrap,
            args.seed + 100,
        )
    )
    transfer = pd.DataFrame(transfer_rows)
    shared_output = directories["olmoe"].parent
    transfer.to_csv(
        shared_output / "quality_isolation_cross_model_transfer_2026-07-20.csv",
        index=False,
    )
    validation_end = args.train_samples + args.validation_samples
    paired_harm_rho = spearman(
        samples_by_model["olmoe"]["label_mean_token_kl"].to_numpy()[validation_end:],
        samples_by_model["llmjp"]["label_mean_token_kl"].to_numpy()[validation_end:],
    )
    summary = {
        "selections": selections,
        "paired_test_harm_spearman_olmoe_vs_llmjp": paired_harm_rho,
        "paired_test_documents": len(samples_by_model["olmoe"]) - validation_end,
    }
    (shared_output / "quality_isolation_proxy_summary_2026-07-20.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print("\nCross-model transfer")
    print(transfer.to_string(index=False))
    print(f"\nPaired test harm Spearman: {paired_harm_rho:.6f}")


if __name__ == "__main__":
    main()
