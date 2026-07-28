#!/usr/bin/env python3
"""Analyze the single-GPU multi-MoE inference-time characterization."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    return parser.parse_args()


def percentile(values: pd.Series, q: float) -> float:
    return float(np.percentile(values.to_numpy(dtype=float), q))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def layer_index(module_name: str) -> int:
    match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", module_name)
    return int(match.group(1)) if match else -1


def write_stable_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, float_format="%.12g")


def markdown_scalar(value: object) -> str:
    if pd.isna(value):
        return "nan"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.6g}"
    return str(value).replace("|", "\\|")


def dataframe_to_markdown(frame: pd.DataFrame) -> str:
    """Render a compact deterministic table without pandas' tabulate extra."""
    columns = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(markdown_scalar(value) for value in row) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir).resolve()
    manifest = json.loads((input_dir / "run_manifest.json").read_text(encoding="utf-8"))
    timings = pd.read_csv(input_dir / "timings_raw.csv")
    layers = pd.read_csv(input_dir / "moe_layers_raw.csv")
    routes = pd.read_csv(input_dir / "route_census_untimed.csv")
    required_timing = {
        "arm",
        "phase",
        "batch_size",
        "repeat",
        "seed",
        "decode_step",
        "latency_ms",
    }
    if not required_timing.issubset(timings.columns):
        raise RuntimeError(f"timings schema missing {sorted(required_timing - set(timings.columns))}")
    if timings.empty or layers.empty:
        raise RuntimeError("timing or layer evidence is empty")

    layer_sum = (
        layers.groupby(
            ["arm", "phase", "batch_size", "prompt_len", "repeat", "seed", "decode_step"],
            as_index=False,
        )
        .agg(moe_sum_ms=("latency_ms", "sum"), moe_calls=("module_name", "count"))
    )
    profiled = timings[timings["arm"] == "profiled"].merge(
        layer_sum,
        on=["arm", "phase", "batch_size", "prompt_len", "repeat", "seed", "decode_step"],
        how="left",
        validate="one_to_one",
    )
    if profiled[["moe_sum_ms", "moe_calls"]].isna().any().any():
        raise RuntimeError("some profiled forwards have no matching per-layer records")
    expected_moe_blocks = int(manifest["model"]["moe_block_count"])
    if not (profiled["moe_calls"] == expected_moe_blocks).all():
        bad = profiled[profiled["moe_calls"] != expected_moe_blocks]
        raise RuntimeError(f"MoE layer census mismatch in {len(bad)} forwards")
    profiled["moe_fraction"] = profiled["moe_sum_ms"] / profiled["latency_ms"]

    layer_summary = (
        layers.groupby(["phase", "batch_size", "module_name"], as_index=False)
        .agg(
            samples=("latency_ms", "count"),
            latency_median_ms=("latency_ms", "median"),
            latency_p95_ms=("latency_ms", lambda values: percentile(values, 95)),
        )
    )
    layer_summary["layer_index"] = layer_summary["module_name"].map(layer_index)
    layer_summary["share_of_sum_of_layer_medians"] = layer_summary.groupby(
        ["phase", "batch_size"]
    )["latency_median_ms"].transform(lambda values: values / values.sum())
    layer_summary = layer_summary.sort_values(["phase", "batch_size", "layer_index"])
    write_stable_csv(layer_summary, input_dir / "layer_summary.csv")

    concentration_rows: list[dict[str, object]] = []
    for (phase, batch_size), group in layer_summary.groupby(["phase", "batch_size"]):
        shares = group["share_of_sum_of_layer_medians"].sort_values(ascending=False)
        concentration_rows.append(
            {
                "phase": phase,
                "batch_size": int(batch_size),
                "num_moe_layers": len(group),
                "max_single_layer_share": float(shares.iloc[0]),
                "top4_layer_share": float(shares.iloc[: min(4, len(shares))].sum()),
                "layer_latency_cv": float(
                    group["latency_median_ms"].std(ddof=0)
                    / group["latency_median_ms"].mean()
                ),
            }
        )
    concentration = pd.DataFrame(concentration_rows).sort_values(["phase", "batch_size"])
    write_stable_csv(concentration, input_dir / "layer_concentration.csv")

    summary_rows: list[dict[str, object]] = []
    for (phase, batch_size), group in timings.groupby(["phase", "batch_size"]):
        unprofiled = group[group["arm"] == "unprofiled"]
        prof = profiled[(profiled["phase"] == phase) & (profiled["batch_size"] == batch_size)]
        if unprofiled.empty or prof.empty:
            raise RuntimeError(f"missing arm for phase={phase}, batch={batch_size}")
        summary_rows.append(
            {
                "phase": phase,
                "batch_size": int(batch_size),
                "unprofiled_n": len(unprofiled),
                "unprofiled_latency_median_ms": float(unprofiled["latency_ms"].median()),
                "unprofiled_latency_p95_ms": percentile(unprofiled["latency_ms"], 95),
                "unprofiled_goodput_median_tokens_per_s": float(
                    unprofiled["goodput_tokens_per_s"].median()
                ),
                "profiled_latency_median_ms": float(prof["latency_ms"].median()),
                "instrumentation_ratio_median": float(
                    prof["latency_ms"].median() / unprofiled["latency_ms"].median()
                ),
                "moe_sum_median_ms": float(prof["moe_sum_ms"].median()),
                "moe_sum_p95_ms": percentile(prof["moe_sum_ms"], 95),
                "moe_fraction_median": float(prof["moe_fraction"].median()),
                "moe_fraction_p95": percentile(prof["moe_fraction"], 95),
                "moe_calls_per_forward": expected_moe_blocks,
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values(["phase", "batch_size"])
    write_stable_csv(summary, input_dir / "summary.csv")

    correlation_rows: list[dict[str, object]] = []
    if not routes.empty:
        route_step = (
            routes[routes["phase"] == "decode"]
            .groupby(["batch_size", "seed", "decode_step"], as_index=False)
            .agg(
                route_max_to_mean_max=("max_to_mean", "max"),
                route_max_to_mean_mean=("max_to_mean", "mean"),
                route_load_cv_mean=("load_cv", "mean"),
                route_entropy_mean=("normalized_entropy", "mean"),
                route_layers=("module_name", "count"),
            )
        )
        primary = timings[
            (timings["arm"] == "unprofiled")
            & (timings["phase"] == "decode")
            & (timings["repeat"] == 0)
        ]
        joined = primary.merge(
            route_step,
            on=["batch_size", "seed", "decode_step"],
            how="inner",
            validate="one_to_one",
        )
        for batch_size, group in joined.groupby("batch_size"):
            correlations_for_metric: dict[str, tuple[float, float]] = {}
            for metric in ("route_max_to_mean_mean", "route_load_cv_mean"):
                if len(group) < 4 or group[metric].nunique() < 2:
                    correlations_for_metric[metric] = (float("nan"), float("nan"))
                    continue
                result = spearmanr(group["latency_ms"], group[metric])
                correlations_for_metric[metric] = (
                    float(result.statistic),
                    float(result.pvalue),
                )
            correlation_rows.append(
                {
                    "batch_size": int(batch_size),
                    "n_decode_steps": len(group),
                    "spearman_latency_vs_route_max_to_mean_mean": correlations_for_metric[
                        "route_max_to_mean_mean"
                    ][0],
                    "pvalue_max_to_mean_descriptive_only": correlations_for_metric[
                        "route_max_to_mean_mean"
                    ][1],
                    "spearman_latency_vs_route_load_cv_mean": correlations_for_metric[
                        "route_load_cv_mean"
                    ][0],
                    "pvalue_load_cv_descriptive_only": correlations_for_metric[
                        "route_load_cv_mean"
                    ][1],
                    "route_max_to_mean_max_median": float(
                        group["route_max_to_mean_max"].median()
                    ),
                    "route_max_to_mean_mean": float(group["route_max_to_mean_mean"].mean()),
                    "route_load_cv_mean": float(group["route_load_cv_mean"].mean()),
                    "evidence_boundary": "single-GPU route imbalance correlation; not receiver congestion",
                }
            )
    correlations = pd.DataFrame(correlation_rows)
    write_stable_csv(correlations, input_dir / "route_latency_correlations.csv")

    decode_summary = summary[summary["phase"] == "decode"]
    high_share_cells = int((decode_summary["moe_fraction_median"] >= 0.30).sum())
    status = "SINGLE_GPU_MULTI_LAYER_MOE_COST_CHARACTERIZED"
    decision = {
        "status": status,
        "receiver_congestion_status": "NOT_TESTED_REQUIRES_REAL_MULTI_GPU_EP",
        "ranklane_status": "UNCHANGED_NO_GO_UNDER_FROZEN_P_RETURN_MAX_0_20",
        "moe_block_count": expected_moe_blocks,
        "decode_cells_with_median_moe_fraction_ge_0_30": high_share_cells,
        "decode_max_single_layer_share_range": [
            float(concentration[concentration["phase"] == "decode"]["max_single_layer_share"].min()),
            float(concentration[concentration["phase"] == "decode"]["max_single_layer_share"].max()),
        ],
        "decode_top4_layer_share_range": [
            float(concentration[concentration["phase"] == "decode"]["top4_layer_share"].min()),
            float(concentration[concentration["phase"] == "decode"]["top4_layer_share"].max()),
        ],
        "interpretation": (
            "A high cumulative local MoE-block share establishes only that repeated MoE execution is material "
            "to single-GPU inference time. It does not isolate return all-to-all or receiver queueing."
        ),
        "next_gate": (
            "Use real EP ranks and one timeline to separate expert compute, return A2A, receiver-visible, "
            "unpack/combine, and token completion."
        ),
    }
    (input_dir / "decision.json").write_text(
        json.dumps(decision, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    lines = [
        "# Receiver Multi-MoE Inference-Time 5090 Result",
        "",
        f"- model: `{manifest['model']['name']}`",
        f"- model revision: `{manifest['model']['config_commit_hash']}`",
        f"- discovered MoE blocks: `{expected_moe_blocks}`",
        f"- status: `{status}`",
        "- receiver congestion: `NOT_TESTED_REQUIRES_REAL_MULTI_GPU_EP`",
        "",
        "## Timing summary",
        "",
        dataframe_to_markdown(summary),
        "",
        "## Layer-time concentration",
        "",
        dataframe_to_markdown(concentration),
        "",
        "## Route imbalance correlation (descriptive only)",
        "",
        dataframe_to_markdown(correlations) if not correlations.empty else "No compatible route rows.",
        "",
        "## Interpretation",
        "",
        "The unprofiled arm is the primary full-inference timing. The profiled arm adds per-MoE-block CUDA events;",
        "`instrumentation_ratio_median` quantifies this observer tax. `moe_fraction` uses the profiled denominator",
        "and is a local cumulative MoE-block characterization, not an EP return-path fraction.",
        "",
        "This run has one GPU, no EP ranks, no NCCL/NVLink return collective, no receiver queue, and no natural",
        "continuous arrivals. Multiple sequential MoE blocks can accumulate inference cost, but this cannot be",
        "called receiver congestion. The formal Receiver existence Gate remains 8xA100 real EP serving.",
        "",
    ]
    (input_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    analyzer_path = Path(__file__).resolve()
    analysis_manifest = {
        "status": status,
        "analyzer": {"path": str(analyzer_path), "sha256": sha256_file(analyzer_path)},
        "inputs": {
            name: sha256_file(input_dir / name)
            for name in (
                "run_manifest.json",
                "timings_raw.csv",
                "moe_layers_raw.csv",
                "route_census_untimed.csv",
            )
        },
        "outputs": {
            name: sha256_file(input_dir / name)
            for name in (
                "summary.csv",
                "layer_summary.csv",
                "layer_concentration.csv",
                "route_latency_correlations.csv",
                "decision.json",
                "report.md",
            )
        },
        "evidence_boundary": "single-GPU characterization; not receiver congestion",
    }
    (input_dir / "analysis_manifest.json").write_text(
        json.dumps(analysis_manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
