#!/usr/bin/env python3
"""Combine the component, context-length, and natural-input 5090 evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

import analyze_multi_moe_inference as common


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--component-dir", required=True)
    parser.add_argument("--context-dirs", required=True, nargs="+")
    parser.add_argument("--input-ab-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_analysis(directory: Path) -> None:
    analysis = load_json(directory / "analysis_manifest.json")
    for section in ("inputs", "outputs"):
        for name, expected in analysis[section].items():
            actual = common.sha256_file(directory / name)
            if actual != expected:
                raise RuntimeError(f"hash mismatch: {directory / name}")


def main() -> None:
    args = parse_args()
    component_dir = Path(args.component_dir).resolve()
    context_dirs = [Path(item).resolve() for item in args.context_dirs]
    input_ab_dir = Path(args.input_ab_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs = [component_dir, *context_dirs, input_ab_dir]
    for directory in inputs:
        validate_analysis(directory)

    component_summary = pd.read_csv(component_dir / "summary.csv")
    component_decision = load_json(component_dir / "decision.json")
    common.write_stable_csv(component_summary, output_dir / "component_summary.csv")

    context_frames = []
    for directory in context_dirs:
        manifest = load_json(directory / "run_manifest.json")
        frame = pd.read_csv(directory / "summary.csv")
        frame.insert(0, "prompt_len", int(manifest["args"]["prompt_len"]))
        context_frames.append(frame)
    context = pd.concat(context_frames, ignore_index=True).sort_values(
        ["phase", "batch_size", "prompt_len"]
    )
    common.write_stable_csv(context, output_dir / "context_summary.csv")

    decode_context = context[context["phase"] == "decode"].copy()
    decode_range = (
        decode_context.groupby("batch_size", as_index=False)["unprofiled_latency_median_ms"]
        .agg(["min", "max"])
        .reset_index()
    )
    decode_range["max_over_min"] = decode_range["max"] / decode_range["min"]
    common.write_stable_csv(decode_range, output_dir / "decode_context_range.csv")

    comparison = pd.read_csv(input_ab_dir / "summary.csv")
    common.write_stable_csv(comparison, output_dir / "natural_vs_synthetic_interleaved.csv")

    decode_natural = comparison[comparison["phase"] == "decode"]
    decision = {
        "status": "SINGLE_GPU_EXTENSIONS_COMPLETE_NOT_RECEIVER_GATE",
        "component_breakdown_valid": component_decision["quantitative_breakdown_valid"],
        "component_max_decode_observer_tax_fraction": component_decision[
            "max_decode_observer_tax_fraction"
        ],
        "decode_context_max_over_min_max": float(decode_range["max_over_min"].max()),
        "natural_decode_delta_pct_range": [
            float(decode_natural["natural_delta_pct_median"].min()),
            float(decode_natural["natural_delta_pct_median"].max()),
        ],
        "receiver_congestion_status": "NOT_TESTED_REQUIRES_REAL_MULTI_GPU_EP",
        "interpretation": (
            "Single-GPU evidence characterizes local MoE cost and workload sensitivity only. "
            "It cannot establish return-path receiver congestion or RankLane benefit."
        ),
    }
    (output_dir / "decision.json").write_text(
        json.dumps(decision, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    decode_component = component_summary[component_summary["phase"] == "decode"]
    context_compact = context[
        [
            "prompt_len",
            "phase",
            "batch_size",
            "unprofiled_latency_median_ms",
            "unprofiled_latency_p95_ms",
            "moe_fraction_median",
        ]
    ]
    component_compact = decode_component[
        [
            "batch_size",
            "observer_ratio_median",
            "moe_fraction_median",
            "gate_fraction_median",
            "routing_setup_fraction_median",
            "expert_loop_fraction_median",
            "unattributed_tail_fraction_median",
        ]
    ]
    comparison_compact = comparison[
        [
            "phase",
            "batch_size",
            "independent_sequence_pairs",
            "synthetic_latency_median_ms",
            "natural_latency_median_ms",
            "natural_delta_pct_median",
        ]
    ]
    lines = [
        "# Receiver Single-GPU Extensions on RTX 5090",
        "",
        f"- status: `{decision['status']}`",
        f"- component breakdown valid: `{str(decision['component_breakdown_valid']).upper()}`",
        "- receiver congestion: `NOT_TESTED_REQUIRES_REAL_MULTI_GPU_EP`",
        "",
        "## Coarse local MoE components (decode)",
        "",
        common.dataframe_to_markdown(component_compact),
        "",
        "`expert_loop` includes gather, expert compute, weighting, and local index_add. It is not pure",
        "expert GEMM and is not return-path time.",
        "",
        "## Context-length sweep",
        "",
        common.dataframe_to_markdown(context_compact),
        "",
        "## Interleaved frozen natural text versus synthetic token sequence",
        "",
        common.dataframe_to_markdown(comparison_compact),
        "",
        "These experiments use one GPU and eager local MoE execution. They contain no EP ranks, NCCL",
        "return collective, receiver queue, or RankLane implementation, so they do not answer the formal",
        "Receiver existence or benefit Gate.",
        "",
    ]
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")

    analyzer_path = Path(__file__).resolve()
    input_hashes = {}
    for directory in inputs:
        for name in ("run_manifest.json", "summary.csv", "decision.json", "analysis_manifest.json"):
            path = directory / name
            if path.exists():
                input_hashes[f"{directory.name}/{name}"] = common.sha256_file(path)
    output_names = (
        "component_summary.csv",
        "context_summary.csv",
        "decode_context_range.csv",
        "natural_vs_synthetic_interleaved.csv",
        "decision.json",
        "report.md",
    )
    analysis_manifest = {
        "status": decision["status"],
        "analyzer": {"path": str(analyzer_path), "sha256": common.sha256_file(analyzer_path)},
        "inputs": input_hashes,
        "outputs": {name: common.sha256_file(output_dir / name) for name in output_names},
    }
    (output_dir / "analysis_manifest.json").write_text(
        json.dumps(analysis_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
