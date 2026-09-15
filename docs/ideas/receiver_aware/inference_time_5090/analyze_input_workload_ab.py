#!/usr/bin/env python3
"""Analyze the interleaved natural versus synthetic single-GPU A/B."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import analyze_multi_moe_inference as common


PAIR_KEYS = ["phase", "batch_size", "prompt_len", "repeat", "seed", "decode_step"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir).resolve()
    timings = pd.read_csv(input_dir / "timings_raw.csv")
    routes = pd.read_csv(input_dir / "route_census_untimed.csv")
    manifest = json.loads((input_dir / "run_manifest.json").read_text(encoding="utf-8"))
    natural = timings[timings["arm"] == "natural"]
    synthetic = timings[timings["arm"] == "synthetic"]
    paired = natural.merge(
        synthetic,
        on=PAIR_KEYS,
        suffixes=("_natural", "_synthetic"),
        validate="one_to_one",
    )
    if len(paired) != len(natural) or len(paired) != len(synthetic):
        raise RuntimeError("natural/synthetic timing arms are not exactly paired")
    paired["natural_over_synthetic"] = paired["latency_ms_natural"] / paired["latency_ms_synthetic"]
    common.write_stable_csv(paired, input_dir / "paired_forward.csv")

    sequence = (
        paired.groupby(["phase", "batch_size", "repeat"], as_index=False)
        .agg(
            natural_median_ms=("latency_ms_natural", "median"),
            synthetic_median_ms=("latency_ms_synthetic", "median"),
        )
    )
    sequence["natural_over_synthetic"] = sequence["natural_median_ms"] / sequence["synthetic_median_ms"]
    common.write_stable_csv(sequence, input_dir / "paired_sequence.csv")

    rows = []
    for (phase, batch_size), group in sequence.groupby(["phase", "batch_size"]):
        ratios = group["natural_over_synthetic"].to_numpy(dtype=float)
        rows.append(
            {
                "phase": phase,
                "batch_size": int(batch_size),
                "independent_sequence_pairs": len(group),
                "synthetic_latency_median_ms": float(group["synthetic_median_ms"].median()),
                "natural_latency_median_ms": float(group["natural_median_ms"].median()),
                "natural_over_synthetic_median": float(np.median(ratios)),
                "natural_over_synthetic_min": float(np.min(ratios)),
                "natural_over_synthetic_max": float(np.max(ratios)),
                "natural_delta_pct_median": float((np.median(ratios) - 1.0) * 100.0),
            }
        )
    summary = pd.DataFrame(rows).sort_values(["phase", "batch_size"])
    common.write_stable_csv(summary, input_dir / "summary.csv")

    route_summary = (
        routes.groupby(["workload", "phase", "batch_size"], as_index=False)
        .agg(
            route_events=("module_name", "count"),
            max_to_mean_mean=("max_to_mean", "mean"),
            load_cv_mean=("load_cv", "mean"),
            active_expert_fraction_mean=("active_expert_fraction", "mean"),
            normalized_entropy_mean=("normalized_entropy", "mean"),
        )
    )
    common.write_stable_csv(route_summary, input_dir / "route_summary.csv")
    decode = summary[summary["phase"] == "decode"]
    decision = {
        "status": "SINGLE_GPU_INTERLEAVED_INPUT_AB_ONLY_NOT_RECEIVER_CONGESTION",
        "decode_natural_delta_pct_median_range": [
            float(decode["natural_delta_pct_median"].min()),
            float(decode["natural_delta_pct_median"].max()),
        ],
        "independent_sequence_pairs_per_cell": int(decode["independent_sequence_pairs"].min()),
        "receiver_congestion_status": "NOT_TESTED_REQUIRES_REAL_MULTI_GPU_EP",
        "interpretation": (
            "Input-dependent local latency is measured by an interleaved one-GPU A/B. "
            "It is not receiver congestion and has only five independent sequence pairs per cell."
        ),
    }
    (input_dir / "decision.json").write_text(
        json.dumps(decision, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    lines = [
        "# Interleaved Natural versus Synthetic Input A/B on RTX 5090",
        "",
        f"- model: `{manifest['model']['name']}`",
        f"- revision: `{manifest['model']['config_commit_hash']}`",
        "- order: `AB/BA alternating by repeat`",
        "- receiver congestion: `NOT_TESTED_REQUIRES_REAL_MULTI_GPU_EP`",
        "",
        common.dataframe_to_markdown(summary),
        "",
        "Ratios use the median forward latency within each independent sequence, then the median across",
        "sequence pairs. This controls run-order drift better than comparing two standalone runs, but the",
        "sample size remains descriptive. It measures local input sensitivity, not receiver congestion.",
        "",
        "## Untimed route census",
        "",
        common.dataframe_to_markdown(route_summary),
        "",
    ]
    (input_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")

    analyzer_path = Path(__file__).resolve()
    output_names = (
        "paired_forward.csv",
        "paired_sequence.csv",
        "summary.csv",
        "route_summary.csv",
        "decision.json",
        "report.md",
    )
    analysis_manifest = {
        "status": decision["status"],
        "analyzer": {"path": str(analyzer_path), "sha256": common.sha256_file(analyzer_path)},
        "inputs": {
            name: common.sha256_file(input_dir / name)
            for name in ("run_manifest.json", "timings_raw.csv", "route_census_untimed.csv")
        },
        "outputs": {name: common.sha256_file(input_dir / name) for name in output_names},
    }
    (input_dir / "analysis_manifest.json").write_text(
        json.dumps(analysis_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
