#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from routeshare_core import (  # noqa: E402
    fit_linear_cost,
    predict_linear_cost,
    squared_error_gap_recovery,
)


def interval(values: np.ndarray, repeats: int, seed: int, statistic=np.median) -> dict:
    if len(values) == 0:
        raise ValueError("cannot bootstrap an empty sample")
    rng = np.random.default_rng(seed)
    boot = np.empty(repeats, dtype=np.float64)
    for index in range(repeats):
        sample = values[rng.integers(0, len(values), size=len(values))]
        boot[index] = statistic(sample)
    return {
        "point": float(statistic(values)),
        "lcb": float(np.quantile(boot, 0.025)),
        "ucb": float(np.quantile(boot, 0.975)),
        "n": int(len(values)),
    }


def recovery_interval(
    truth: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    repeats: int,
    seed: int,
) -> dict:
    if not (len(truth) == len(baseline) == len(candidate)) or len(truth) == 0:
        raise ValueError("invalid recovery vectors")
    rng = np.random.default_rng(seed)
    boot = np.empty(repeats, dtype=np.float64)
    for index in range(repeats):
        choice = rng.integers(0, len(truth), size=len(truth))
        boot[index] = squared_error_gap_recovery(
            truth[choice], baseline[choice], candidate[choice]
        )
    return {
        "point": float(squared_error_gap_recovery(truth, baseline, candidate)),
        "lcb": float(np.quantile(boot, 0.025)),
        "ucb": float(np.quantile(boot, 0.975)),
        "n": int(len(truth)),
    }


def matched_histogram_contrasts(rows: pd.DataFrame) -> np.ndarray:
    keys = [
        "model_key",
        "layer_id",
        "tokens_per_tenant",
        "overlap_fraction",
        "seed",
        "total_rows",
        "active_experts",
    ]
    contrasts = []
    for _, group in rows.groupby(keys, sort=False):
        if set(group["histogram_regime"]) != {"balanced", "skewed"}:
            continue
        balanced = group[group["histogram_regime"] == "balanced"].iloc[0]
        skewed = group[group["histogram_regime"] == "skewed"].iloc[0]
        histogram_columns = [
            "max_rows_per_expert",
            "row_count_cv",
            *[column for column in rows.columns if column.startswith("experts_bin_")],
        ]
        # Expert order inside top-k can alter route SHA while leaving the
        # executable row histogram identical (notably k=pool_size). Such a
        # pair is a null cost intervention and must not enter the contrast.
        if all(
            np.isclose(float(balanced[column]), float(skewed[column]))
            for column in histogram_columns
        ):
            continue
        a = float(balanced["coalition_latency_ms"])
        b = float(skewed["coalition_latency_ms"])
        contrasts.append(abs(a - b) / max((a + b) / 2.0, 1e-12))
    return np.asarray(contrasts, dtype=np.float64)


def analyze_cell(calibration: pd.DataFrame, sealed: pd.DataFrame, config: dict, seed: int) -> dict:
    models = ("m0_rows", "m1_rows_active", "m2_row_bins")
    predictions = {}
    coefficients = {}
    for model in models:
        beta = fit_linear_cost(calibration.to_dict("records"), model)
        coefficients[model] = beta.tolist()
        predictions[model] = predict_linear_cost(sealed.to_dict("records"), model, beta)
    truth = sealed["coalition_latency_ms"].to_numpy(dtype=np.float64)
    m1_explained = recovery_interval(
        truth, predictions["m0_rows"], predictions["m1_rows_active"],
        config["bootstrap_resamples"], seed,
    )
    m2_recovery = recovery_interval(
        truth, predictions["m1_rows_active"], predictions["m2_row_bins"],
        config["bootstrap_resamples"], seed + 1,
    )
    contrasts = matched_histogram_contrasts(sealed)
    sham = np.abs(
        sealed["sham_latency_ms"].to_numpy(dtype=np.float64)
        - sealed["tenant_separate_latency_ms"].to_numpy(dtype=np.float64)
    ) / np.maximum(sealed["tenant_separate_latency_ms"].to_numpy(dtype=np.float64), 1e-12)
    relative_error = np.abs(truth - predictions["m1_rows_active"]) / np.maximum(truth, 1e-12)
    return {
        "coefficients": coefficients,
        "m1_absolute_relative_error": interval(
            relative_error, config["bootstrap_resamples"], seed + 2
        ),
        "matched_histogram_contrast": interval(
            contrasts, config["bootstrap_resamples"], seed + 3
        ),
        "m1_gap_explained_over_rows_only": m1_explained,
        "m2_gap_recovery_over_m1": m2_recovery,
        "sham_relative_difference": interval(
            sham, config["bootstrap_resamples"], seed + 4
        ),
        "coalition_over_separate": interval(
            sealed["coalition_over_separate"].to_numpy(dtype=np.float64),
            config["bootstrap_resamples"], seed + 5,
        ),
        "closure_max_abs": float(sealed["closure_max_abs"].max()),
        "closure_max_rel": float(sealed["closure_max_rel"].max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    gate = config["gate"]
    results = {}
    approvals = []
    for model_index, model_key in enumerate(config["models"]):
        calibration = pd.read_csv(
            args.input_root / model_key / "calibration" / "scenario_summary.csv"
        )
        sealed = pd.read_csv(args.input_root / model_key / "sealed" / "scenario_summary.csv")
        if set(calibration["split"]) != {"calibration"} or set(sealed["split"]) != {"sealed"}:
            raise RuntimeError("split contamination detected")
        if set(calibration["seed"]) & set(sealed["seed"]):
            raise RuntimeError("calibration/sealed seed leakage detected")
        for layer_id in config["layers"]:
            calibration_cell = calibration[calibration["layer_id"] == layer_id].copy()
            sealed_cell = sealed[sealed["layer_id"] == layer_id].copy()
            expected = (
                len(config["tokens_per_tenant"])
                * len(config["overlap_fractions"])
                * len(config["histogram_regimes"])
                * len(config["sealed_seeds"])
            )
            if len(sealed_cell) != expected:
                raise RuntimeError(f"incomplete sealed cell {model_key}/L{layer_id}")
            cell = analyze_cell(
                calibration_cell,
                sealed_cell,
                config,
                config["bootstrap_seed"] + model_index * 100 + layer_id,
            )
            checks = {
                "matched_contrast": cell["matched_histogram_contrast"]["lcb"]
                >= gate["matched_contrast_lcb_min"],
                "m1_not_sufficient": cell["m1_gap_explained_over_rows_only"]["point"]
                < gate["m1_oracle_gap_explained_max"],
                "m2_recovery": cell["m2_gap_recovery_over_m1"]["lcb"]
                >= gate["m2_gap_recovery_lcb_min"],
                "sham": cell["sham_relative_difference"]["ucb"]
                <= gate["sham_relative_difference_ucb_max"],
                "closure": cell["closure_max_abs"] <= gate["output_max_abs"]
                and cell["closure_max_rel"] <= gate["output_max_rel"],
            }
            cell["checks"] = checks
            cell["pass"] = all(checks.values())
            approvals.append(cell["pass"])
            results[f"{model_key}_layer_{layer_id}"] = cell
    decision = "GO_ROUTE_COALITION_COST" if all(approvals) else "NO_GO_ROUTE_COALITION_COST"
    payload = {
        "decision": decision,
        "all_model_layer_cells_must_pass": True,
        "results": results,
        "evidence_boundary": "single-GPU single-layer BF16 executable oracle; not serving/network",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
