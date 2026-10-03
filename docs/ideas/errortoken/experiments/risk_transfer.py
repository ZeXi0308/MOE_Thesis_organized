#!/usr/bin/env python3
"""Retrospective calibration-to-fresh transfer probe for ErrorToken-MoE.

This analysis deliberately does not import the SemanticFence runner.  It reads
the sealed contract, fresh call plan, and row-level result table as data.  The
result is exploratory because the aggregate evaluation verdict was known before
this program was written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable


Key = tuple[int, int, int]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            rows.append(value)
    return rows


def calibration_risk(entry: dict[str, Any]) -> float:
    exact = int(entry["exact_checks"])
    total = int(entry["total_checks"])
    if total <= 0 or exact < 0 or exact > total:
        raise ValueError(f"invalid exact/total checks: {exact}/{total}")
    return 1.0 - exact / total


def aggregate_risks(
    entries: Iterable[dict[str, Any]], fields: tuple[str, ...]
) -> dict[tuple[int, ...], float]:
    counts: dict[tuple[int, ...], list[int]] = {}
    for entry in entries:
        key = tuple(int(entry[field]) for field in fields)
        pair = counts.setdefault(key, [0, 0])
        pair[0] += int(entry["exact_checks"])
        pair[1] += int(entry["total_checks"])
    return {key: 1.0 - exact / total for key, (exact, total) in counts.items() if total > 0}


def roc_auc(scores: list[float], labels: list[int]) -> float | None:
    if len(scores) != len(labels):
        raise ValueError("score and label lengths differ")
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None

    ordered = sorted(zip(scores, labels), key=lambda item: item[0])
    positive_rank_sum = 0.0
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        average_rank = ((index + 1) + end) / 2.0
        positive_rank_sum += average_rank * sum(label for _, label in ordered[index:end])
        index = end
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def policy_point(calls: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    total_rows = sum(len(call["row_ids"]) for call in calls)
    natural_mismatch_rows = sum(call["mismatch_rows"] for call in calls)
    admitted = [
        call
        for call in calls
        if call.get("key_risk") is not None and float(call["key_risk"]) <= threshold
    ]
    admitted_rows = sum(len(call["row_ids"]) for call in admitted)
    mismatches = sum(call["mismatch_rows"] for call in admitted)
    launches = len(admitted) + (total_rows - admitted_rows)
    return {
        "threshold": threshold,
        "admitted_calls": len(admitted),
        "admitted_rows": admitted_rows,
        "batched_row_fraction": admitted_rows / total_rows if total_rows else 0.0,
        "mismatch_rows_after_policy": mismatches,
        "mismatch_row_fraction": mismatches / total_rows if total_rows else 0.0,
        "avoided_natural_mismatch_fraction": (
            1.0 - mismatches / natural_mismatch_rows if natural_mismatch_rows else 1.0
        ),
        "launch_count_proxy": launches,
        "launch_reduction_fraction": 1.0 - launches / total_rows if total_rows else 0.0,
    }


def build_analysis(config: dict[str, Any], root: Path) -> dict[str, Any]:
    contract_path = root / "CONTRACT.json"
    calls_path = root / "evaluation_calls.jsonl"
    results_path = root / "evaluation_arm_results.json"

    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    entries = contract["entries"]
    key_risks: dict[Key, float] = {
        (int(entry["layer"]), int(entry["expert_id"]), int(entry["m"])): calibration_risk(entry)
        for entry in entries
    }
    if len(key_risks) != len(entries):
        raise ValueError("duplicate (layer, expert, M) contract entries")

    m_risks = aggregate_risks(entries, ("m",))
    layer_m_risks = aggregate_risks(entries, ("layer", "m"))

    arm_results = json.loads(results_path.read_text(encoding="utf-8"))
    b_arms = [arm for arm in arm_results["arms"] if arm["arm"] == "B_native_unrestricted"]
    if len(b_arms) != 1:
        raise ValueError("expected exactly one B_native_unrestricted arm")
    row_results: dict[str, dict[str, Any]] = {}
    for row in b_arms[0]["rows"]:
        row_id = str(row["row_id"])
        if row_id in row_results:
            raise ValueError(f"duplicate result row_id: {row_id}")
        if not bool(row["bitwise_stable"]):
            raise ValueError(f"unstable evaluation row: {row_id}")
        row_results[row_id] = row

    eligible_m = {int(value) for value in config["eligible_m"]}
    calls: list[dict[str, Any]] = []
    seen_rows: set[str] = set()
    for call in load_jsonl(calls_path):
        if call.get("arm") != "B_native_unrestricted":
            continue
        m = int(call["m"])
        if m <= 1:
            continue
        row_ids = [str(value) for value in call["row_ids"]]
        if len(row_ids) != m:
            raise ValueError(f"call M/row count mismatch at {call['call_index']}")
        overlap = seen_rows.intersection(row_ids)
        if overlap:
            raise ValueError(f"rows occur in multiple B calls: {sorted(overlap)[:3]}")
        seen_rows.update(row_ids)
        missing = [row_id for row_id in row_ids if row_id not in row_results]
        if missing:
            raise ValueError(f"missing result rows: {missing[:3]}")
        mismatch_rows = sum(
            0 if bool(row_results[row_id]["all_exact_to_reference"]) else 1 for row_id in row_ids
        )
        key = (int(call["layer"]), int(call["expert_id"]), m)
        calls.append(
            {
                "call_index": int(call["call_index"]),
                "layer": key[0],
                "expert_id": key[1],
                "m": m,
                "row_ids": row_ids,
                "mismatch_rows": mismatch_rows,
                "eligible_grid_m": m in eligible_m,
                "key_risk": key_risks.get(key),
                "m_only_risk": m_risks.get((m,)),
                "layer_m_risk": layer_m_risks.get((key[0], m)),
            }
        )

    eligible_calls = [call for call in calls if call["eligible_grid_m"]]
    matched_calls = [call for call in eligible_calls if call["key_risk"] is not None]
    eligible_rows = sum(len(call["row_ids"]) for call in eligible_calls)
    matched_rows = sum(len(call["row_ids"]) for call in matched_calls)

    row_labels: list[int] = []
    key_scores: list[float] = []
    m_scores: list[float] = []
    layer_m_scores: list[float] = []
    for call in matched_calls:
        for row_id in call["row_ids"]:
            label = 0 if bool(row_results[row_id]["all_exact_to_reference"]) else 1
            row_labels.append(label)
            key_scores.append(float(call["key_risk"]))
            m_scores.append(float(call["m_only_risk"]))
            layer_m_scores.append(float(call["layer_m_risk"]))

    auc_key = roc_auc(key_scores, row_labels)
    auc_m = roc_auc(m_scores, row_labels)
    auc_layer_m = roc_auc(layer_m_scores, row_labels)
    curves = [policy_point(eligible_calls, float(value)) for value in config["thresholds"]]
    primary = next(
        point for point in curves if math.isclose(point["threshold"], float(config["primary_threshold"]))
    )

    gate = config["gate"]
    coverage = matched_rows / eligible_rows if eligible_rows else 0.0
    class_supported = bool(row_labels) and 0 < sum(row_labels) < len(row_labels)
    if coverage < float(gate["minimum_matched_row_fraction"]) or not class_supported or auc_key is None:
        verdict = config["interpretation"]["inconclusive"]
        reason = "matched coverage or positive/negative class support is insufficient"
    elif (
        auc_key >= float(gate["support_min_key_auc"])
        and auc_m is not None
        and auc_key - auc_m >= float(gate["support_min_auc_gain_over_m_only"])
        and primary["launch_reduction_fraction"]
        >= float(gate["support_min_primary_launch_reduction_fraction"])
        and primary["mismatch_row_fraction"]
        <= float(gate["support_max_primary_mismatch_row_fraction"])
    ):
        verdict = config["interpretation"]["support"]
        reason = "keyed calibration risk transfers beyond M-only and retains a bounded proxy tradeoff"
    elif auc_key <= float(gate["weaken_max_key_auc"]):
        verdict = config["interpretation"]["weaken"]
        reason = "keyed calibration risk does not discriminate fresh mismatch rows"
    else:
        verdict = config["interpretation"]["inconclusive"]
        reason = "some ranking signal exists but the frozen support conditions are not jointly met"

    return {
        "schema_version": "errortoken-risk-transfer-summary-v1",
        "status": "COMPLETE_RETROSPECTIVE_CPU_ANALYSIS",
        "verdict": verdict,
        "reason": reason,
        "claim_boundary": config["interpretation"]["claim_boundary"],
        "unblinding_status": config["status"],
        "inputs": {
            "contract_sha256": sha256_file(contract_path),
            "evaluation_calls_sha256": sha256_file(calls_path),
            "evaluation_arm_results_sha256": sha256_file(results_path),
        },
        "denominators": {
            "all_natural_m_gt_1_calls": len(calls),
            "eligible_grid_calls": len(eligible_calls),
            "eligible_grid_rows": eligible_rows,
            "matched_key_calls": len(matched_calls),
            "matched_key_rows": matched_rows,
            "matched_row_fraction": coverage,
            "matched_positive_rows": sum(row_labels),
            "matched_negative_rows": len(row_labels) - sum(row_labels),
        },
        "discrimination": {
            "key_layer_expert_m_row_auc": auc_key,
            "layer_m_row_auc": auc_layer_m,
            "m_only_row_auc": auc_m,
            "key_auc_gain_over_m_only": (
                auc_key - auc_m if auc_key is not None and auc_m is not None else None
            ),
            "note": "rows from one call share a score and are not independent samples",
        },
        "primary_policy_point": primary,
        "policy_curve": curves,
        "proxy_warning": "launch_count_proxy is not measured GPU latency",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"refusing to reuse output directory: {args.output_dir}")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    summary = build_analysis(config, args.artifact_root)
    summary["inputs"]["config_sha256"] = sha256_file(args.config)
    summary["inputs"]["runner_sha256"] = sha256_file(Path(__file__))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    output_path = args.output_dir / "summary.json"
    output_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
