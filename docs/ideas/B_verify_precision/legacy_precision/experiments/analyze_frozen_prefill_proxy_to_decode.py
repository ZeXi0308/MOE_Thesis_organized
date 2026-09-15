#!/usr/bin/env python3
"""Transfer the previously frozen same-prompt proxy to future decode harm."""
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
from pathlib import Path

import numpy as np
import pandas as pd

from run_quality_isolation_proxy_gpu_strict import (
    auc_score,
    bootstrap_spearman_ci,
    recall_at_count,
    ridge_fit_predict,
    spearman,
    worst_fraction_labels,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--decode-dir", required=True)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260720)
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    decode_dir = Path(args.decode_dir)
    source = pd.read_csv(source_dir / "sample_features_labels.csv")
    decode = pd.read_csv(decode_dir / "decode_fragility_samples.csv")
    frozen = pd.read_csv(source_dir / "proxy_results_augmented.csv")
    source_fit = source[source["split"].isin(["train", "validation"])]
    decode_test = decode[decode["split"] == "test"]

    lexical = [
        "char_count",
        "token_count_input",
        "unique_token_ratio",
        "token_id_entropy",
        "adjacent_repeat_rate",
    ]
    full_route = [
        column for column in source.columns if column.startswith("full_route_")
    ]
    columns = lexical + full_route + ["full_mean_nll"]
    frozen_row = frozen[frozen["feature_group"] == "post_prefill_all"].iloc[0]
    alpha = float(frozen_row["selected_alpha_on_validation"])
    fit_x = source_fit[columns].to_numpy(dtype=float)
    fit_y = np.log10(
        source_fit["label_mean_token_kl"].to_numpy(dtype=float) + 1e-12
    )
    test_x = decode_test[columns].to_numpy(dtype=float)
    _, prediction = ridge_fit_predict(
        fit_x,
        fit_y,
        fit_x[:1],
        test_x,
        alpha,
    )

    action_columns = [
        column
        for column in decode.columns
        if column.endswith("__decode_mean_kl")
    ]
    rows: list[dict] = []
    for index, column in enumerate(action_columns):
        action = column.removesuffix("__decode_mean_kl")
        target = decode_test[column].to_numpy(dtype=float)
        ci_low, ci_high = bootstrap_spearman_ci(
            target,
            prediction,
            args.bootstrap,
            args.seed + index,
        )
        worst = worst_fraction_labels(target, 0.1)
        rows.append(
            {
                "frozen_source_label": "same_prompt_fixed_tail_int4",
                "frozen_feature_group": "post_prefill_all",
                "frozen_alpha": alpha,
                "decode_action": action,
                "decode_test_samples": len(decode_test),
                "decode_test_spearman": spearman(target, prediction),
                "decode_test_spearman_ci_low": ci_low,
                "decode_test_spearman_ci_high": ci_high,
                "decode_worst_decile_auc": auc_score(worst, prediction),
                "decode_worst_decile_recall_at_10pct": recall_at_count(
                    target, prediction, 0.1
                ),
            }
        )
    results = pd.DataFrame(rows)
    results.to_csv(decode_dir / "frozen_prefill_proxy_to_decode.csv", index=False)
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
