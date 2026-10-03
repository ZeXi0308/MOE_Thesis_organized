#!/usr/bin/env python3
"""Open and analyze sealed TriageAudit document metrics exactly once."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Mapping

import numpy as np

from triage_artifacts import write_json_no_overwrite
from triage_policy import AuditState, FrozenConfidenceGuard, common_audit_phase, cvar
from triage_statistics import analyze_model, cross_model_decision


class AnalysisError(RuntimeError):
    pass


ALL_POLICIES = (
    "always_bf16",
    "always_low",
    "triage_2_4_8",
    "hash_budget_matched_2_4_8",
    "fixed_2",
    "fixed_4",
    "fixed_8",
    "full_shadow",
)


def _read_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise AnalysisError(f"line {line_number} is not an object")
            rows.append(value)
    return rows


def _close(actual: float, expected: float) -> bool:
    return math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-12)


def validate_complete_raw_results(
    rows: list[Mapping[str, object]],
    config: Mapping[str, object],
    calibration_lock: Mapping[str, object],
) -> None:
    expected_documents = int(config["dataset"]["sealed_documents"])
    decode_steps = int(config["dataset"]["decode_steps"])
    lock_schema = calibration_lock.get("schema_version")
    if lock_schema not in {
        "triage-calibration-lock-v2",
        "confidence-guard-calibration-lock-v3",
    }:
        raise AnalysisError("wrong calibration lock schema")
    if {str(row.get("model_key")) for row in rows} != {"olmoe", "llmjp"}:
        raise AnalysisError("raw results must contain exactly both frozen models")
    for model_key in ("olmoe", "llmjp"):
        threshold = float(calibration_lock["models"][model_key]["audit_threshold"])
        confidence_guard = None
        if lock_schema == "confidence-guard-calibration-lock-v3":
            confidence_guard = FrozenConfidenceGuard.from_dict(
                calibration_lock["models"][model_key]["confidence_guard"]["frozen_guard"]
            )
        selected = [row for row in rows if row.get("model_key") == model_key]
        by_policy: dict[str, dict[str, Mapping[str, object]]] = {policy: {} for policy in ALL_POLICIES}
        for row in selected:
            policy = str(row.get("policy"))
            digest = str(row.get("text_sha256"))
            if policy not in by_policy or row.get("split") != "sealed" or len(digest) != 64:
                raise AnalysisError(f"{model_key}: unexpected policy/split/document")
            if digest in by_policy[policy]:
                raise AnalysisError(f"{model_key}: duplicate {policy}/{digest}")
            by_policy[policy][digest] = row
        document_sets = {tuple(sorted(values)) for values in by_policy.values()}
        if len(document_sets) != 1 or len(next(iter(document_sets))) != expected_documents:
            raise AnalysisError(f"{model_key}: eight-arm document sets do not close")
        documents = list(next(iter(document_sets)))
        triage_periods: list[int] = []
        hash_periods: list[int] = []
        for policy in ALL_POLICIES:
            for digest in documents:
                row = by_policy[policy][digest]
                period = row.get("period")
                phase = row.get("phase")
                if policy == "always_bf16":
                    if period is not None or phase is not None or row.get("steps") != []:
                        raise AnalysisError("always_bf16 must have no period, phase, or diagnostic steps")
                    if float(row.get("document_cvar90_kl", -1)) != 0.0 or int(row.get("physical_low_forward_calls", -1)) != 0:
                        raise AnalysisError("always_bf16 baseline is not exact")
                    continue
                if policy == "always_low":
                    if period is not None or phase is not None:
                        raise AnalysisError("always_low must have no period/phase")
                else:
                    expected_period = {
                        "fixed_2": 2,
                        "fixed_4": 4,
                        "fixed_8": 8,
                        "full_shadow": 1,
                    }.get(policy)
                    if expected_period is not None and period != expected_period:
                        raise AnalysisError(f"{policy}: fixed period changed")
                    if type(period) is not int or period not in {1, 2, 4, 8}:
                        raise AnalysisError(f"{policy}: invalid period")
                    if phase != common_audit_phase(digest, period):
                        raise AnalysisError(f"{policy}: phase rule mismatch")
                    if policy == "triage_2_4_8":
                        triage_periods.append(period)
                        if confidence_guard is not None:
                            features = row.get("features")
                            if not isinstance(features, Mapping):
                                raise AnalysisError("ConfidenceGuard row lacks predictor features")
                            expected_probability = confidence_guard.safe_probability(features)
                            if not _close(float(row.get("safe_probability", -1)), expected_probability):
                                raise AnalysisError("ConfidenceGuard safe probability mismatch")
                            if period != confidence_guard.period(features):
                                raise AnalysisError("ConfidenceGuard period mismatch")
                    elif policy == "hash_budget_matched_2_4_8":
                        hash_periods.append(period)
                steps = row.get("steps")
                if not isinstance(steps, list) or len(steps) != decode_steps:
                    raise AnalysisError(f"{policy}: step count mismatch")
                dangerous_count = 0
                protected_count = 0
                candidate_calls = 0
                diagnostic_high = 0
                diagnostic_low = 0
                audit_events = 0
                candidate_clones = 0
                diagnostic_clones = 0
                quality_values: list[float] = []
                served_high = 0
                replay_state = None if policy == "always_low" else AuditState(
                    period=int(period),
                    phase=int(phase),
                    max_unaudited_steps=int(config["controller"]["max_unaudited_steps"]),
                    lockout_following_steps=(
                        0 if policy == "full_shadow" else int(config["controller"]["lockout_following_steps"])
                    ),
                )
                for step_index, step in enumerate(steps):
                    if not isinstance(step, Mapping) or step.get("step") != step_index:
                        raise AnalysisError(f"{policy}: invalid step order")
                    discrepancy = float(step["same_state_discrepancy"])
                    dangerous = discrepancy > threshold
                    if not math.isfinite(discrepancy) or discrepancy < 0 or step.get("dangerous") is not dangerous:
                        raise AnalysisError(f"{policy}: dangerous label mismatch")
                    expected_decision = "low" if replay_state is None else replay_state.decision(step_index)
                    if step.get("decision") != expected_decision:
                        raise AnalysisError(f"{policy}: policy state replay decision mismatch")
                    if replay_state is None:
                        expected_action = "low"
                    elif expected_decision == "audit":
                        expected_action = replay_state.record_audit(discrepancy, threshold)
                    else:
                        replay_state.record_single(expected_decision)
                        expected_action = "high" if expected_decision == "lockout_high" else "low"
                    if step.get("served_action") != expected_action:
                        raise AnalysisError(f"{policy}: policy state replay action mismatch")
                    candidate = int(step["candidate_forward_calls"])
                    diagnostic = int(step["diagnostic_forward_calls"])
                    if candidate + diagnostic != 2:
                        raise AnalysisError(f"{policy}: physical work does not close to two forwards")
                    if int(step["diagnostic_high_forward_calls"]) + int(step["diagnostic_low_forward_calls"]) != diagnostic:
                        raise AnalysisError(f"{policy}: diagnostic high/low split mismatch")
                    if (step.get("decision") == "audit") != (candidate == 2):
                        raise AnalysisError(f"{policy}: audit candidate cost mismatch")
                    served_action = step.get("served_action")
                    if served_action not in {"high", "low"}:
                        raise AnalysisError(f"{policy}: invalid served action")
                    dangerous_count += int(dangerous)
                    protected_count += int(dangerous and served_action == "high")
                    served_high += int(served_action == "high")
                    candidate_calls += candidate
                    candidate_clones += int(step["candidate_clone_events"])
                    diagnostic_clones += int(step["diagnostic_clone_events"])
                    diagnostic_high += int(step["diagnostic_high_forward_calls"])
                    diagnostic_low += int(step["diagnostic_low_forward_calls"])
                    audit_events += int(step.get("decision") == "audit")
                    quality_values.append(float(step["served_quality_kl"]))
                expected_recall = protected_count / dangerous_count if dangerous_count else 1.0
                expected_violation = (dangerous_count - protected_count) / decode_steps
                checks = (
                    int(row["total_candidate_forward_calls"]) == candidate_calls,
                    int(row["high_forward_calls"]) + int(row["low_forward_calls"]) == candidate_calls,
                    int(row["audit_events"]) == audit_events,
                    int(row["cache_clone_events"]) == candidate_clones,
                    int(row["diagnostic_clone_events"]) == diagnostic_clones,
                    int(row["diagnostic_high_forward_calls"]) == diagnostic_high,
                    int(row["diagnostic_low_forward_calls"]) == diagnostic_low,
                    int(row["dangerous_steps"]) == dangerous_count,
                    int(row["served_high_steps"]) == served_high,
                    int(row["served_low_steps"]) == decode_steps - served_high,
                    int(row["physical_high_forward_calls"]) == int(row["high_forward_calls"]) + diagnostic_high,
                    int(row["physical_low_forward_calls"]) == int(row["low_forward_calls"]) + diagnostic_low,
                    int(row["physical_high_forward_calls"]) + int(row["physical_low_forward_calls"])
                    == 2 * decode_steps,
                    _close(float(row["dangerous_step_recall"]), expected_recall),
                    _close(float(row["threshold_violation_fraction"]), expected_violation),
                    _close(float(row["document_mean_kl"]), sum(quality_values) / decode_steps),
                    _close(float(row["document_cvar90_kl"]), cvar(quality_values, 0.1)),
                    _close(float(row["document_p95_kl"]), float(np.quantile(quality_values, 0.95))),
                )
                if not all(checks):
                    raise AnalysisError(f"{policy}: document counters or metrics do not close")
                if policy == "full_shadow" and audit_events != decode_steps:
                    raise AnalysisError("full_shadow did not audit every step")
        if sorted(triage_periods) != sorted(hash_periods):
            raise AnalysisError(f"{model_key}: triage/hash period histograms differ")


def analyze(rows: list[Mapping[str, object]], config: Mapping[str, object]) -> dict[str, object]:
    statistics = config.get("statistics")
    dataset = config.get("dataset")
    if not isinstance(statistics, Mapping) or not isinstance(dataset, Mapping):
        raise AnalysisError("config lacks statistics or dataset")
    if statistics.get("effect_estimator") != "median_of_document_level_paired_effects":
        raise AnalysisError("unexpected or missing frozen effect estimator")
    if statistics.get("pareto_effect_estimator") != "policy_level_mean_points_rebuilt_per_document_bootstrap":
        raise AnalysisError("unexpected or missing frozen Pareto estimator")
    model_results: dict[str, object] = {}
    for offset, model_key in enumerate(("olmoe", "llmjp")):
        selected = [row for row in rows if row.get("model_key") == model_key]
        if any(row.get("split") != "sealed" for row in selected):
            raise AnalysisError(f"{model_key}: non-sealed row in sealed analysis")
        result = analyze_model(
            selected,
            bootstrap_repeats=int(statistics["bootstrap_replicates"]),
            seed=int(config["seed"]) + 100 + offset,
        )
        if result["documents"] != int(dataset["sealed_documents"]):
            raise AnalysisError(f"{model_key}: sealed document count mismatch")
        model_results[model_key] = result
    decision = cross_model_decision(model_results)
    return {
        "schema_version": "triage-sealed-decision-v2",
        "evidence_boundary": config.get("evidence_boundary"),
        "models": model_results,
        "cross_model": decision,
        "status": "GO" if decision["go"] else "NO_GO",
    }


def write_document_csv(path: Path, rows: list[Mapping[str, object]]) -> None:
    if path.exists():
        raise AnalysisError(f"refusing to overwrite {path}")
    fields = [
        "model_key", "split", "text_sha256", "policy", "period",
        "document_mean_kl", "document_cvar90_kl", "document_p95_kl",
        "total_candidate_forward_calls", "audit_events", "dangerous_steps",
        "dangerous_step_recall", "threshold_violation_fraction",
        "diagnostic_high_forward_calls", "diagnostic_low_forward_calls",
        "physical_high_forward_calls", "physical_low_forward_calls",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--raw-results", type=Path, required=True)
    parser.add_argument("--calibration-lock", type=Path, required=True)
    parser.add_argument("--decision-output", type=Path, required=True)
    parser.add_argument("--paired-bootstrap-output", type=Path, required=True)
    parser.add_argument("--document-csv-output", type=Path, required=True)
    parser.add_argument("--status-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    rows = _read_rows(args.raw_results)
    calibration_lock = json.loads(args.calibration_lock.read_text(encoding="utf-8"))
    validate_complete_raw_results(rows, config, calibration_lock)
    result = analyze(rows, config)
    write_document_csv(args.document_csv_output, rows)
    write_json_no_overwrite(args.paired_bootstrap_output, {
        "schema_version": "triage-paired-bootstrap-v2",
        "independent_unit": "document",
        "effect_estimator": config["statistics"]["effect_estimator"],
        "pareto_effect_estimator": config["statistics"]["pareto_effect_estimator"],
        "models": result["models"],
    })
    write_json_no_overwrite(args.decision_output, result)
    write_json_no_overwrite(args.status_output, {
        "schema_version": "triage-analysis-status-v2",
        "status": result["status"],
        "verdict": result["cross_model"]["verdict"],
        "evidence_boundary": result["evidence_boundary"],
    })
    if args.summary_output.exists():
        raise AnalysisError(f"refusing to overwrite {args.summary_output}")
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    with args.summary_output.open("x", encoding="utf-8") as handle:
        handle.write(
            "# TriageAudit Gate M\n\n"
            f"Status: {result['status']}\n\n"
            f"Verdict: {result['cross_model']['verdict']}\n\n"
            f"Evidence boundary: {result['evidence_boundary']}\n"
        )


if __name__ == "__main__":
    main()
