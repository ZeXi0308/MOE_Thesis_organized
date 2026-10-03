#!/usr/bin/env python3
"""Analyze coarse single-GPU local MoE component timing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import analyze_multi_moe_inference as common


KEYS = ["arm", "phase", "batch_size", "prompt_len", "repeat", "seed", "decode_step"]
OBSERVER_TAX_LIMIT = 0.10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    return parser.parse_args()


def p95(values: pd.Series) -> float:
    return float(np.percentile(values.to_numpy(dtype=float), 95))


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir).resolve()
    manifest = json.loads((input_dir / "run_manifest.json").read_text(encoding="utf-8"))
    timings = pd.read_csv(input_dir / "timings_raw.csv")
    components = pd.read_csv(input_dir / "components_raw.csv")
    if timings.empty or components.empty:
        raise RuntimeError("empty timing evidence")
    expected_blocks = int(manifest["model"]["moe_block_count"])
    breakdown_events = components[components["arm"] == "breakdown"]

    grouped = (
        breakdown_events.groupby(KEYS + ["component"], as_index=False)
        .agg(component_sum_ms=("latency_ms", "sum"), calls=("module_name", "count"))
    )
    sums = grouped.pivot(index=KEYS, columns="component", values="component_sum_ms").reset_index()
    calls = grouped.pivot(index=KEYS, columns="component", values="calls").reset_index()
    required_components = ("moe_total", "gate", "routing_setup", "expert_loop")
    for required in required_components:
        if required not in sums.columns:
            raise RuntimeError(f"missing component {required}")
        component_calls = calls.set_index(KEYS)[required]
        if not (component_calls == expected_blocks).all():
            raise RuntimeError(f"not every profiled forward covered every MoE block for {required}")

    sums["unattributed_tail_ms"] = (
        sums["moe_total"] - sums["gate"] - sums["routing_setup"] - sums["expert_loop"]
    )
    if (sums["unattributed_tail_ms"] < -1e-3).any():
        raise RuntimeError("component nesting produced a materially negative residual")
    sums["unattributed_tail_ms"] = sums["unattributed_tail_ms"].clip(lower=0.0)
    breakdown_timing = timings[timings["arm"] == "breakdown"].merge(
        sums, on=KEYS, how="left", validate="one_to_one"
    )
    if breakdown_timing[list(required_components)].isna().any().any():
        raise RuntimeError("some breakdown forwards lack component rows")
    for component in ("moe_total", "gate", "routing_setup", "expert_loop", "unattributed_tail_ms"):
        breakdown_timing[f"{component}_fraction"] = (
            breakdown_timing[component] / breakdown_timing["latency_ms"]
        )
    common.write_stable_csv(breakdown_timing, input_dir / "breakdown_per_forward.csv")

    rows: list[dict[str, object]] = []
    for (phase, batch_size), group in timings.groupby(["phase", "batch_size"]):
        primary = group[group["arm"] == "unprofiled"]
        detailed = breakdown_timing[
            (breakdown_timing["phase"] == phase) & (breakdown_timing["batch_size"] == batch_size)
        ]
        observer_ratio = float(detailed["latency_ms"].median() / primary["latency_ms"].median())
        rows.append(
            {
                "phase": phase,
                "batch_size": int(batch_size),
                "unprofiled_n": len(primary),
                "unprofiled_latency_median_ms": float(primary["latency_ms"].median()),
                "unprofiled_latency_p95_ms": p95(primary["latency_ms"]),
                "breakdown_latency_median_ms": float(detailed["latency_ms"].median()),
                "observer_ratio_median": observer_ratio,
                "observer_tax_acceptable": observer_ratio <= 1.0 + OBSERVER_TAX_LIMIT,
                "moe_total_median_ms": float(detailed["moe_total"].median()),
                "gate_median_ms": float(detailed["gate"].median()),
                "routing_setup_median_ms": float(detailed["routing_setup"].median()),
                "expert_loop_median_ms": float(detailed["expert_loop"].median()),
                "unattributed_tail_median_ms": float(detailed["unattributed_tail_ms"].median()),
                "moe_fraction_median": float(detailed["moe_total_fraction"].median()),
                "gate_fraction_median": float(detailed["gate_fraction"].median()),
                "routing_setup_fraction_median": float(detailed["routing_setup_fraction"].median()),
                "expert_loop_fraction_median": float(detailed["expert_loop_fraction"].median()),
                "unattributed_tail_fraction_median": float(
                    detailed["unattributed_tail_ms_fraction"].median()
                ),
            }
        )
    summary = pd.DataFrame(rows).sort_values(["phase", "batch_size"])
    common.write_stable_csv(summary, input_dir / "summary.csv")

    decode = summary[summary["phase"] == "decode"]
    max_observer_tax = float((decode["observer_ratio_median"] - 1.0).max())
    quantitative_valid = bool((decode["observer_tax_acceptable"] == True).all())  # noqa: E712
    decision = {
        "status": (
            "SINGLE_GPU_LOCAL_MOE_COMPONENT_BREAKDOWN_ONLY"
            if quantitative_valid
            else "INVALID_FOR_QUANTITATIVE_BREAKDOWN_OBSERVER_TAX"
        ),
        "receiver_congestion_status": "NOT_TESTED_NO_EP_TRAFFIC",
        "observer_tax_limit_fraction": OBSERVER_TAX_LIMIT,
        "max_decode_observer_tax_fraction": max_observer_tax,
        "quantitative_breakdown_valid": quantitative_valid,
        "interpretation": (
            "The expert_loop includes gather, expert compute, routing-weight multiplication, and index_add. "
            "It is not pure expert GEMM or return-path time."
        ),
    }
    (input_dir / "decision.json").write_text(
        json.dumps(decision, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    lines = [
        "# Coarse Local MoE Component Breakdown on RTX 5090",
        "",
        f"- model: `{manifest['model']['name']}`",
        f"- revision: `{manifest['model']['config_commit_hash']}`",
        f"- MoE blocks: `{expected_blocks}`",
        f"- quantitative breakdown valid: `{str(quantitative_valid).upper()}`",
        f"- maximum decode observer tax: `{max_observer_tax:.2%}` (limit `{OBSERVER_TAX_LIMIT:.0%}`)",
        "- receiver congestion: `NOT_TESTED_NO_EP_TRAFFIC`",
        "",
        common.dataframe_to_markdown(summary),
        "",
        "`routing_setup` contains softmax/top-k/normalization/allocation/one-hot/active-expert discovery.",
        "`expert_loop` contains gather, expert compute, weighting, and local index_add. It is not a pure",
        "expert-GEMM or return-all-to-all measurement. Rows fail closed when decode observer tax exceeds 10%.",
        "",
    ]
    (input_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")

    analyzer_path = Path(__file__).resolve()
    analysis_manifest = {
        "status": decision["status"],
        "analyzer": {"path": str(analyzer_path), "sha256": common.sha256_file(analyzer_path)},
        "inputs": {
            name: common.sha256_file(input_dir / name)
            for name in ("run_manifest.json", "timings_raw.csv", "components_raw.csv")
        },
        "outputs": {
            name: common.sha256_file(input_dir / name)
            for name in ("breakdown_per_forward.csv", "summary.csv", "decision.json", "report.md")
        },
    }
    (input_dir / "analysis_manifest.json").write_text(
        json.dumps(analysis_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
