#!/usr/bin/env python3
"""Independent artifact-ledger recompute for the completed SFV2-O1 run."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Mapping, Sequence


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import run_semantic_online_observability_5090 as gate


class AuditError(RuntimeError):
    pass


COMPLETE_KEYS = {
    "schema_version",
    "status",
    "verdict",
    "completion_last",
    "pre_outcome_lock_sha256",
    "validation_threshold_sha256",
    "test_admission_plan_sha256",
    "summary_sha256",
    "artifact_sha256",
    "paper_result",
}

FEATURE_ROW_KEYS = {
    "row_id",
    "logical_split",
    "document_sha256",
    "row_record",
    "window_id",
    "cell",
    "feature_vector",
    "feature_vector_sha256",
}

COST_KEYS = {
    "schema_version",
    "test_expert_microcost",
    "natural_oracle",
    "frozen_certificate",
    "feature_lookup_and_greedy_overhead",
    "exact_baseline_projected_saving",
    "boundary",
}

NUMERIC_ABS_TOLERANCE = 1e-12
NUMERIC_REL_TOLERANCE = 0.0


def require_equal(left: Any, right: Any, label: str) -> None:
    if gate.canonical_sha256(left) != gate.canonical_sha256(right):
        raise AuditError(f"{label} differs")


def require_numeric_close(left: Any, right: Any, label: str) -> None:
    """Compare recomputed numeric artifacts without weakening discrete bindings.

    Python releases can differ by a few ULPs in floating-point reductions.  JSON
    container shape, scalar types, and every non-float value remain exact; only
    finite float leaves receive the narrowly bounded comparison used elsewhere
    by the experiment harness.
    """

    def compare(actual: Any, expected: Any, path: str) -> None:
        if type(actual) is not type(expected):
            raise AuditError(
                f"{label} differs at {path}: types "
                f"{type(actual).__name__} != {type(expected).__name__}"
            )
        if isinstance(actual, dict):
            if set(actual) != set(expected):
                missing = sorted(set(expected) - set(actual))
                unexpected = sorted(set(actual) - set(expected))
                raise AuditError(
                    f"{label} differs at {path}: "
                    f"missing={missing}, unexpected={unexpected}"
                )
            for key in sorted(actual):
                compare(actual[key], expected[key], f"{path}.{key}")
            return
        if isinstance(actual, list):
            if len(actual) != len(expected):
                raise AuditError(
                    f"{label} differs at {path}: lengths "
                    f"{len(actual)} != {len(expected)}"
                )
            for index, (actual_item, expected_item) in enumerate(
                zip(actual, expected)
            ):
                compare(actual_item, expected_item, f"{path}[{index}]")
            return
        if isinstance(actual, float):
            if not (
                math.isfinite(actual)
                and math.isfinite(expected)
                and math.isclose(
                    actual,
                    expected,
                    rel_tol=NUMERIC_REL_TOLERANCE,
                    abs_tol=NUMERIC_ABS_TOLERANCE,
                )
            ):
                raise AuditError(
                    f"{label} differs at {path}: actual={actual!r}, "
                    f"expected={expected!r}, absolute_delta={abs(actual - expected)!r}"
                )
            return
        if actual != expected:
            raise AuditError(
                f"{label} differs at {path}: actual={actual!r}, expected={expected!r}"
            )

    compare(left, right, "$")


def compare_oracle_recompute(
    actual: Mapping[str, Any], expected: Mapping[str, Any]
) -> tuple[str, str]:
    """Compare oracle results while separating producer/auditor provenance."""

    actual_matching = actual.get("matching")
    expected_matching = expected.get("matching")
    if not isinstance(actual_matching, dict) or not isinstance(expected_matching, dict):
        raise AuditError("natural oracle matching is absent or malformed")
    actual_version = actual_matching.get("networkx_version")
    expected_version = expected_matching.get("networkx_version")
    if not isinstance(actual_version, str) or not actual_version:
        raise AuditError("auditor networkx provenance is absent")
    if not isinstance(expected_version, str) or not expected_version:
        raise AuditError("producer networkx provenance is absent")

    actual_result = dict(actual)
    actual_result["matching"] = dict(actual_matching)
    actual_result["matching"].pop("networkx_version")
    expected_result = dict(expected)
    expected_result["matching"] = dict(expected_matching)
    expected_result["matching"].pop("networkx_version")
    require_numeric_close(actual_result, expected_result, "natural oracle")
    return expected_version, actual_version


def require_exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        missing = sorted(expected - set(value))
        unexpected = sorted(set(value) - expected)
        raise AuditError(f"{label} keys differ: missing={missing}, unexpected={unexpected}")


def _actual_artifact_hashes(root: Path) -> dict[str, str]:
    return {
        path.name: gate.sha256_file(path)
        for path in sorted(root.iterdir(), key=lambda value: value.name)
        if path.is_file() and path.name != "COMPLETE.json"
    }


def ensure_output_outside_formal_root(root: Path, output: Path) -> None:
    formal_root = root.resolve()
    audit_output = output.resolve()
    try:
        audit_output.relative_to(formal_root)
    except ValueError:
        return
    raise AuditError("audit output must be outside the sealed formal output directory")


def validate_complete(root: Path) -> dict[str, Any]:
    complete_path = root / "COMPLETE.json"
    complete = gate.load_json(complete_path)
    if not isinstance(complete, dict):
        raise AuditError("formal COMPLETE is not an object")
    require_exact_keys(complete, COMPLETE_KEYS, "formal COMPLETE")
    if (
        complete["schema_version"] != gate.COMPLETE_SCHEMA
        or complete["status"] != "SUCCESS_COMPLETE"
        or complete["completion_last"] is not True
        or complete["paper_result"] is not False
    ):
        raise AuditError("formal COMPLETE is absent or non-success")
    declared = complete["artifact_sha256"]
    if not isinstance(declared, dict) or not all(
        isinstance(name, str) and isinstance(digest, str)
        for name, digest in declared.items()
    ):
        raise AuditError("formal COMPLETE artifact map is malformed")
    actual = _actual_artifact_hashes(root)
    require_equal(actual, declared, "formal COMPLETE exact artifact closure")
    direct = {
        "pre_outcome_lock_sha256": "PRE_OUTCOME_LOCK.json",
        "validation_threshold_sha256": "VALIDATION_THRESHOLD.json",
        "test_admission_plan_sha256": "TEST_ADMISSION_PLAN.json",
        "summary_sha256": "SUMMARY.json",
    }
    for key, name in direct.items():
        if complete[key] != actual.get(name):
            raise AuditError(f"formal COMPLETE direct digest differs: {key}")
    complete_mtime = complete_path.stat().st_mtime_ns
    if any(
        path.stat().st_mtime_ns > complete_mtime
        for path in root.iterdir()
        if path.is_file() and path.name != "COMPLETE.json"
    ):
        raise AuditError("artifact was modified after COMPLETE")
    return complete


def load_schedule(root: Path) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    schedule = gate.load_jsonl(root / "CANDIDATE_SCHEDULE.jsonl")
    by_split = {
        split: [dict(row) for row in schedule if row["logical_split"] == split]
        for split in ("train", "validation", "test")
    }
    return schedule, by_split


def load_results(root: Path) -> dict[str, list[dict[str, Any]]]:
    return {
        "train": gate.load_jsonl(root / "TRAIN_EDGE_RESULTS.jsonl"),
        "validation": gate.load_jsonl(root / "VALIDATION_EDGE_RESULTS.jsonl"),
        "test": gate.load_jsonl(root / "TEST_EDGE_RESULTS.jsonl"),
    }


def validate_edge_ledgers(
    schedule: Mapping[str, Sequence[Mapping[str, Any]]],
    results: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for split in ("train", "validation", "test"):
        scheduled = [str(row["edge_id"]) for row in schedule[split]]
        observed = [str(row["edge_id"]) for row in results[split]]
        if scheduled != observed or len(observed) != len(set(observed)):
            raise AuditError(f"{split} schedule/result edge closure differs")
        for planned, row in zip(schedule[split], results[split]):
            identity_keys = (
                "schedule_index",
                "edge_id",
                "logical_split",
                "document_sha256",
                "window_id",
                "layer",
                "expert_id",
                "abi",
                "row_ids",
            )
            if any(row.get(key) != planned.get(key) for key in identity_keys):
                raise AuditError(f"{split} result/schedule identity differs: {row['edge_id']}")
            endpoints = list(row["endpoints"])
            if len(endpoints) != 2 or [str(value.get("row_id")) for value in endpoints] != list(
                map(str, planned["row_ids"])
            ):
                raise AuditError(f"{split} endpoint/schedule identity differs: {row['edge_id']}")
            expected = all(bool(endpoint["semantic_safe"]) for endpoint in endpoints)
            if bool(row["pair_safe"]) != expected:
                raise AuditError(f"{split} pair AND differs: {row['edge_id']}")
            for endpoint_index, endpoint in enumerate(endpoints):
                if (
                    endpoint.get("row_record") != planned["row_records"][endpoint_index]
                    or endpoint.get("window_id")
                    != planned["endpoint_context"][endpoint_index]["window_id"]
                ):
                    raise AuditError(f"{split} endpoint context differs: {row['edge_id']}")
                route_delta = endpoint["route_delta"]
                ordered_changed = list(route_delta["ordered_topk_changed_layers"])
                membership_changed = list(route_delta["membership_changed_layers"])
                if (
                    bool(route_delta["any_ordered_topk_change"]) != bool(ordered_changed)
                    or bool(route_delta["any_membership_change"]) != bool(membership_changed)
                ):
                    raise AuditError(f"{split} route-delta aggregate differs")
                expected_safe = not bool(ordered_changed)
                if bool(endpoint["semantic_safe"]) != expected_safe:
                    raise AuditError(f"{split} route/top-k label differs")
                if not endpoint["m1_injected_full_forward_stable_2_of_2"]:
                    raise AuditError(f"{split} M1 repeat is unstable")
                if not endpoint["m2_injected_full_forward_stable_2_of_2"]:
                    raise AuditError(f"{split} M2 repeat is unstable")
        report[split] = {
            "edges": len(observed),
            "safe_edges": sum(bool(row["pair_safe"]) for row in results[split]),
            "endpoint_observations": 2 * len(observed),
        }
    return report


def reconstruct_bank(
    root: Path,
    schedule: Sequence[Mapping[str, Any]],
    train_results: Sequence[Mapping[str, Any]],
) -> tuple[
    dict[str, list[float]],
    dict[str, tuple[int, int, str]],
    dict[tuple[int, int, str], dict[str, list[str]]],
]:
    feature_rows = gate.load_jsonl(root / "PRE_OUTCOME_FEATURES.jsonl")
    expected_identity: dict[str, dict[str, Any]] = {}
    for edge in schedule:
        if not (
            len(edge["row_ids"]) == len(edge["row_records"]) == len(edge["endpoint_context"]) == 2
        ):
            raise AuditError("candidate endpoint identity denominator differs")
        for row_id, record, context in zip(
            edge["row_ids"], edge["row_records"], edge["endpoint_context"]
        ):
            identity = {
                "logical_split": edge["logical_split"],
                "document_sha256": edge["document_sha256"],
                "row_record": record,
                "window_id": context["window_id"],
                "cell": [int(edge["layer"]), int(edge["expert_id"]), str(edge["abi"])],
            }
            existing = expected_identity.setdefault(str(row_id), identity)
            if existing != identity:
                raise AuditError("candidate endpoint identity changes across edges")

    features: dict[str, list[float]] = {}
    row_cells: dict[str, tuple[int, int, str]] = {}
    observed_identity: dict[str, dict[str, Any]] = {}
    dimensions: set[int] = set()
    for row in feature_rows:
        require_exact_keys(row, FEATURE_ROW_KEYS, "pre-outcome feature row")
        row_id = str(row["row_id"])
        if row_id in features:
            raise AuditError("pre-outcome feature row ID repeats")
        vector = list(map(float, row["feature_vector"]))
        if not vector or not all(math.isfinite(value) for value in vector):
            raise AuditError("pre-outcome feature vector is empty or nonfinite")
        if row["feature_vector_sha256"] != gate.canonical_sha256(vector):
            raise AuditError("pre-outcome feature vector hash differs")
        cell = (int(row["cell"][0]), int(row["cell"][1]), str(row["cell"][2]))
        features[row_id] = vector
        row_cells[row_id] = cell
        dimensions.add(len(vector))
        observed_identity[row_id] = {
            key: row[key]
            for key in ("logical_split", "document_sha256", "row_record", "window_id", "cell")
        }
    if dimensions != {76}:
        raise AuditError("pre-outcome feature dimension differs")
    require_equal(observed_identity, expected_identity, "candidate/pre-outcome feature identity")

    observations: dict[str, list[bool]] = defaultdict(list)
    for edge in train_results:
        for endpoint in edge["endpoints"]:
            observations[str(endpoint["row_id"])].append(bool(endpoint["semantic_safe"]))
    labels = {row_id: all(values) for row_id, values in observations.items()}
    train_rows = {
        str(row_id)
        for edge in schedule
        if edge["logical_split"] == "train"
        for row_id in edge["row_ids"]
    }
    if set(labels) != train_rows or any(not values for values in observations.values()):
        raise AuditError("train witness label/vertex closure differs")

    banks_accumulator: dict[tuple[int, int, str], dict[str, list[str]]] = defaultdict(
        lambda: {"safe": [], "unsafe": []}
    )
    for row_id in sorted(labels):
        bucket = "safe" if labels[row_id] else "unsafe"
        banks_accumulator[row_cells[row_id]][bucket].append(row_id)
    banks = {
        cell: {"safe": sorted(value["safe"]), "unsafe": sorted(value["unsafe"])}
        for cell, value in sorted(banks_accumulator.items())
    }
    cells = [
        {
            "cell": list(cell),
            "safe_row_ids": bank["safe"],
            "unsafe_row_ids": bank["unsafe"],
            "safe_count": len(bank["safe"]),
            "unsafe_count": len(bank["unsafe"]),
            "safe_feature_digest": gate.canonical_sha256(
                [features[row_id] for row_id in bank["safe"]]
            ),
            "unsafe_feature_digest": gate.canonical_sha256(
                [features[row_id] for row_id in bank["unsafe"]]
            ),
        }
        for cell, bank in banks.items()
    ]
    expected_manifest = {
        "schema_version": "semanticfence-online-witness-bank-v1",
        "label_aggregation": "unique_row_safe_iff_all_incident_train_endpoint_observations_safe",
        "cell_key": ["layer", "expert_id", "abi"],
        "safe_rows": sum(labels.values()),
        "unsafe_rows": len(labels) - sum(labels.values()),
        "unique_rows": len(labels),
        "cells": cells,
        "pre_outcome_features_sha256": gate.sha256_file(root / "PRE_OUTCOME_FEATURES.jsonl"),
        "train_edge_results_sha256": gate.sha256_file(root / "TRAIN_EDGE_RESULTS.jsonl"),
    }
    manifest = gate.load_json(root / "TRAIN_WITNESS_BANK.json")
    require_equal(expected_manifest, manifest, "reconstructed train witness bank")
    return features, row_cells, banks


def _positive_finite_vector(value: Any, label: str) -> list[float]:
    if not isinstance(value, list) or not value:
        raise AuditError(f"{label} is empty or not a list")
    result = [float(item) for item in value]
    if any(not math.isfinite(item) or item <= 0 for item in result):
        raise AuditError(f"{label} contains a non-positive or nonfinite value")
    return result


def recompute_microcost(
    value: Mapping[str, Any], *, expected_edges: int, label: str
) -> tuple[float, float]:
    if int(value.get("candidate_edges", -1)) != expected_edges or expected_edges <= 0:
        raise AuditError(f"{label} candidate-edge denominator differs")
    raw = value.get("raw_aggregate")
    if not isinstance(raw, dict):
        raise AuditError(f"{label} raw aggregate is absent")
    m1 = _positive_finite_vector(raw.get("m1_two_single_calls_ms"), f"{label} raw M1")
    m2 = _positive_finite_vector(raw.get("m2_one_pair_call_ms"), f"{label} raw M2")
    repeats = int(raw.get("repeats", -1))
    if (
        int(raw.get("warmups", -1)) != 3
        or repeats != 10
        or repeats != len(m1)
        or repeats != len(m2)
    ):
        raise AuditError(f"{label} timing repeat denominator differs")
    m1_median = statistics.median(m1)
    m2_median = statistics.median(m2)
    require_equal(m1_median, raw.get("m1_median_ms"), f"{label} raw M1 median")
    require_equal(m2_median, raw.get("m2_median_ms"), f"{label} raw M2 median")
    require_equal(
        m1_median / m2_median,
        raw.get("ratio_of_medians_m1_over_m2"),
        f"{label} raw ratio of medians",
    )
    c1_ms = m1_median / (2 * expected_edges)
    c2_ms = m2_median / expected_edges
    require_equal(c1_ms, value.get("estimated_single_m1_ms"), f"{label} estimated M1")
    require_equal(c2_ms, value.get("estimated_pair_m2_ms"), f"{label} estimated M2")
    return c1_ms, c2_ms


def recompute_online_overhead(
    value: Mapping[str, Any], *, schedule: Sequence[Mapping[str, Any]], label: str
) -> float:
    elapsed = _positive_finite_vector(value.get("elapsed_ms"), f"{label} elapsed timing")
    if int(value.get("repeats", -1)) != 5 or len(elapsed) != 5:
        raise AuditError(f"{label} repeat denominator differs")
    unique_rows = {str(row_id) for edge in schedule for row_id in edge["row_ids"]}
    if (
        int(value.get("candidate_edges", -1)) != len(schedule)
        or int(value.get("unique_endpoints", -1)) != len(unique_rows)
    ):
        raise AuditError(f"{label} schedule denominator differs")
    median = statistics.median(elapsed)
    require_equal(median, value.get("median_total_ms"), f"{label} median")
    return median


def run_audit(args: argparse.Namespace) -> int:
    root = Path(args.output_dir).resolve()
    output = Path(args.output).resolve()
    ensure_output_outside_formal_root(root, output)
    complete = validate_complete(root)

    pre_lock = gate.load_json(root / "PRE_OUTCOME_LOCK.json")
    if pre_lock.get("status") != "FROZEN_BEFORE_ANY_M1_M2_SEMANTIC_OUTCOME":
        raise AuditError("pre-outcome lock status differs")
    for name, digest in pre_lock["pre_outcome_artifact_sha256"].items():
        if gate.sha256_file(root / name) != digest:
            raise AuditError(f"pre-outcome artifact drift: {name}")
    threshold_artifact = gate.load_json(root / "VALIDATION_THRESHOLD.json")
    admission_plan = gate.load_json(root / "TEST_ADMISSION_PLAN.json")
    if (
        threshold_artifact.get("test_semantic_outcome_count_at_threshold_freeze") != 0
        or admission_plan.get("test_semantic_outcome_count_at_plan_freeze") != 0
        or admission_plan.get("validation_threshold_sha256")
        != gate.sha256_file(root / "VALIDATION_THRESHOLD.json")
    ):
        raise AuditError("threshold/test predecision ordering evidence differs")

    full_schedule, schedule = load_schedule(root)
    results = load_results(root)
    edge_report = validate_edge_ledgers(schedule, results)
    features, row_cells, banks = reconstruct_bank(
        root, full_schedule, results["train"]
    )

    validation_scores = gate._score_split(
        schedule["validation"], features=features, row_cells=row_cells, banks=banks
    )
    validation_labels = gate.conservative_row_labels(results["validation"])
    validation_c1_ms, validation_c2_ms = recompute_microcost(
        threshold_artifact["validation_microcost"],
        expected_edges=len(schedule["validation"]),
        label="validation microcost",
    )
    validation_overhead_ms = recompute_online_overhead(
        threshold_artifact["validation_online_overhead"],
        schedule=schedule["validation"],
        label="validation online overhead",
    )
    recomputed_cost_inputs = {
        "c1_ms": validation_c1_ms,
        "c2_ms": validation_c2_ms,
        "fixed_online_overhead_ms": validation_overhead_ms,
        "overhead_is_constant_across_thresholds": True,
    }
    require_equal(
        recomputed_cost_inputs,
        threshold_artifact["cost_inputs"],
        "validation threshold cost inputs",
    )
    recomputed_threshold = gate.select_validation_threshold(
        schedule["validation"],
        validation_scores,
        validation_labels,
        c1_ms=validation_c1_ms,
        c2_ms=validation_c2_ms,
        fixed_online_overhead_ms=validation_overhead_ms,
    )
    for key, value in recomputed_threshold.items():
        require_numeric_close(
            value, threshold_artifact.get(key), f"validation threshold {key}"
        )
    threshold_bindings = {
        "pre_outcome_lock_sha256": "PRE_OUTCOME_LOCK.json",
        "train_witness_bank_sha256": "TRAIN_WITNESS_BANK.json",
        "validation_edge_results_sha256": "VALIDATION_EDGE_RESULTS.jsonl",
    }
    for key, name in threshold_bindings.items():
        if threshold_artifact.get(key) != gate.sha256_file(root / name):
            raise AuditError(f"validation threshold binding differs: {key}")

    threshold = gate.threshold_value(threshold_artifact)
    test_scores = gate._score_split(
        schedule["test"], features=features, row_cells=row_cells, banks=banks
    )
    require_numeric_close(
        [test_scores[row_id] for row_id in sorted(test_scores)],
        admission_plan["scores"],
        "pre-test scores",
    )
    test_admitted = gate._threshold_admitted(test_scores, threshold)
    if sorted(test_admitted) != sorted(map(str, admission_plan["admitted_row_ids"])):
        raise AuditError("pre-test admitted row IDs differ")
    admission_bindings = {
        "pre_outcome_features_sha256": "PRE_OUTCOME_FEATURES.jsonl",
        "train_witness_bank_sha256": "TRAIN_WITNESS_BANK.json",
    }
    for key, name in admission_bindings.items():
        if admission_plan.get(key) != gate.sha256_file(root / name):
            raise AuditError(f"test admission-plan binding differs: {key}")
    require_equal(
        threshold_artifact["threshold"],
        admission_plan.get("threshold"),
        "test admission-plan threshold",
    )
    require_equal(
        [
            str(edge["edge_id"])
            for edge in schedule["test"]
            if set(map(str, edge["row_ids"])).issubset(test_admitted)
        ],
        admission_plan.get("candidate_admissible_edge_ids"),
        "test admission-plan candidate edges",
    )
    require_equal(
        gate.rolling_greedy_matching(schedule["test"], test_admitted),
        admission_plan.get("rolling_greedy_matching"),
        "test admission-plan greedy matching",
    )

    cost = gate.load_json(root / "COST_PROJECTION.json")
    require_exact_keys(cost, COST_KEYS, "cost projection")
    if cost.get("schema_version") != "semanticfence-online-cost-projection-v1":
        raise AuditError("cost projection schema differs")
    micro = cost["test_expert_microcost"]
    c1_ms, c2_ms = recompute_microcost(
        micro, expected_edges=len(schedule["test"]), label="test microcost"
    )
    test_overhead_ms = recompute_online_overhead(
        admission_plan["online_overhead"],
        schedule=schedule["test"],
        label="test online overhead",
    )
    oracle = gate._oracle_result(
        schedule["test"], results["test"], c1_ms=c1_ms, c2_ms=c2_ms
    )
    stored_oracle = gate.load_json(root / "ORACLE_MATCHING.json")
    producer_networkx_version, auditor_networkx_version = compare_oracle_recompute(
        oracle, stored_oracle
    )
    certificate = gate._certificate_result(
        schedule["test"],
        results["test"],
        test_scores,
        threshold,
        c1_ms=c1_ms,
        c2_ms=c2_ms,
        online_overhead_ms=test_overhead_ms,
    )
    stored_certificate = gate.load_json(root / "CERTIFICATE_RESULTS.json")
    for key, value in certificate.items():
        require_numeric_close(value, stored_certificate.get(key), f"certificate {key}")
    require_numeric_close(
        oracle["cost_projection"],
        cost.get("natural_oracle"),
        "cost projection natural oracle",
    )
    require_numeric_close(
        certificate["cost_projection"],
        cost.get("frozen_certificate"),
        "cost projection frozen certificate",
    )
    require_equal(
        admission_plan["online_overhead"],
        cost.get("feature_lookup_and_greedy_overhead"),
        "cost projection online overhead",
    )
    verdict = gate.decide_verdict(oracle, certificate)
    summary = gate.load_json(root / "SUMMARY.json")
    if verdict != summary.get("verdict") or verdict != complete.get("verdict"):
        raise AuditError("recomputed verdict differs")

    report = {
        "schema_version": "semanticfence-online-independent-recompute-v1",
        "status": "PASS",
        "formal_output": str(root),
        "complete_sha256": gate.sha256_file(root / "COMPLETE.json"),
        "pre_outcome_lock_sha256": gate.sha256_file(root / "PRE_OUTCOME_LOCK.json"),
        "numeric_comparison": {
            "absolute_tolerance": NUMERIC_ABS_TOLERANCE,
            "relative_tolerance": NUMERIC_REL_TOLERANCE,
            "container_shape_scalar_types_and_non_float_values": "exact",
        },
        "networkx_provenance": {
            "producer": producer_networkx_version,
            "auditor": auditor_networkx_version,
        },
        "edge_recompute": edge_report,
        "natural_oracle": {
            "safe_edge_density": oracle["safe_edge_density"],
            "matching_edges": oracle["matching"]["matching_edges"],
            "row_coverage": oracle["matching"]["row_coverage"],
            "projected_saving": oracle["cost_projection"]["gross_saved_fraction"],
        },
        "certificate": {
            "admitted_endpoints": certificate["admitted_endpoints"],
            "unsafe_admissible_candidate_edges": certificate[
                "unsafe_admissible_candidate_edges"
            ],
            "greedy_executed_pairs": certificate["greedy_executed_pairs"],
            "net_projected_saving": certificate["cost_projection"][
                "net_saved_fraction"
            ],
        },
        "verdict": verdict,
        "summary_not_trusted_as_input": True,
    }
    gate.write_json_exclusive(output, report)
    print(gate.canonical_json_bytes(report).decode("utf-8"), end="")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    return run_audit(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AuditError, gate.GateError) as error:
        print(f"SFV2_O1_AUDIT_FAILED: {error}", file=sys.stderr)
        raise SystemExit(2)
