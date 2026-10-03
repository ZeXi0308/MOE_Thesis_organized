#!/usr/bin/env python3
"""Independent raw-ledger verifier for the SemanticFence SFV2-O1 gate.

This verifier deliberately does not import the experiment runner (or any other
project module).  Its decision inputs are the sealed per-edge ledgers, the
pre-outcome locks, the frozen test admission plan, and the raw timing samples.
Derived result artifacts and SUMMARY.json are opened only after the independent
recomputation has finished, and are used solely as consistency comparisons.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


class VerificationError(RuntimeError):
    """Raised when a sealed artifact or independently recomputed invariant fails."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise VerificationError(f"blank JSONL record: {path.name}:{line_number}")
            value = json.loads(line)
            _require(isinstance(value, dict), f"non-object JSONL record: {path.name}:{line_number}")
            rows.append(value)
    return rows


def _as_pair(value: Sequence[Any], context: str) -> tuple[str, str]:
    _require(len(value) == 2, f"{context}: expected exactly two row IDs")
    left, right = map(str, value)
    _require(left != right, f"{context}: self-loop")
    return left, right


def _float_close(left: Any, right: Any, *, context: str) -> None:
    a, b = float(left), float(right)
    _require(
        math.isfinite(a) and math.isfinite(b),
        f"{context}: non-finite value",
    )
    _require(
        math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12),
        f"{context}: {a!r} != {b!r}",
    )


def _is_within(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _verify_complete_closure(formal_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    complete_path = formal_dir / "COMPLETE.json"
    complete = _load_json(complete_path)
    _require(complete.get("status") == "SUCCESS_COMPLETE", "COMPLETE status is not SUCCESS_COMPLETE")
    _require(complete.get("completion_last") is True, "COMPLETE does not claim completion_last")

    sealed = complete.get("artifact_sha256")
    _require(isinstance(sealed, dict) and sealed, "COMPLETE artifact_sha256 is empty")
    directory_files = {path.name for path in formal_dir.iterdir() if path.is_file()}
    _require(
        set(sealed) == directory_files - {"COMPLETE.json"},
        "COMPLETE artifact map is not an exact formal-directory closure",
    )

    observed: dict[str, str] = {}
    mtimes: dict[str, int] = {}
    for name, expected in sorted(sealed.items()):
        path = formal_dir / name
        _require(path.is_file(), f"sealed artifact missing: {name}")
        actual = _sha256(path)
        _require(actual == str(expected), f"sealed artifact hash differs: {name}")
        observed[name] = actual
        mtimes[name] = path.stat().st_mtime_ns

    alias_bindings = {
        "PRE_OUTCOME_LOCK.json": "pre_outcome_lock_sha256",
        "SUMMARY.json": "summary_sha256",
        "TEST_ADMISSION_PLAN.json": "test_admission_plan_sha256",
        "VALIDATION_THRESHOLD.json": "validation_threshold_sha256",
    }
    for artifact, alias in alias_bindings.items():
        _require(complete.get(alias) == observed[artifact], f"COMPLETE alias differs: {alias}")

    complete_mtime = complete_path.stat().st_mtime_ns
    latest_artifact_mtime = max(mtimes.values())
    _require(
        complete_mtime >= latest_artifact_mtime,
        "COMPLETE mtime predates a sealed artifact",
    )
    return complete, {
        "sealed_artifact_count": len(sealed),
        "exact_directory_closure": True,
        "all_sha256_match": True,
        "complete_mtime_ns": complete_mtime,
        "latest_sealed_artifact_mtime_ns": latest_artifact_mtime,
        "complete_mtime_not_older": True,
        "strict_mtime_order_observable": complete_mtime > latest_artifact_mtime,
        "artifact_sha256": observed,
    }


def _verify_pre_outcome_bindings(
    formal_dir: Path,
    complete: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    prelock_path = formal_dir / "PRE_OUTCOME_LOCK.json"
    validation_path = formal_dir / "VALIDATION_THRESHOLD.json"
    plan_path = formal_dir / "TEST_ADMISSION_PLAN.json"
    prelock = _load_json(prelock_path)
    validation = _load_json(validation_path)
    plan = _load_json(plan_path)

    _require(
        prelock.get("status") == "FROZEN_BEFORE_ANY_M1_M2_SEMANTIC_OUTCOME",
        "PRE_OUTCOME_LOCK status differs",
    )
    _require(prelock.get("test_outcome_count_at_lock") == 0, "PRE_OUTCOME_LOCK test outcome count is nonzero")
    _require(
        plan.get("status") == "FROZEN_BEFORE_FIRST_TEST_SEMANTIC_OUTCOME",
        "TEST_ADMISSION_PLAN status differs",
    )
    _require(
        plan.get("test_semantic_outcome_count_at_plan_freeze") == 0,
        "TEST_ADMISSION_PLAN test outcome count is nonzero",
    )
    _require(
        validation.get("test_semantic_outcome_count_at_threshold_freeze") == 0,
        "VALIDATION_THRESHOLD test outcome count is nonzero",
    )

    prelock_hash = _sha256(prelock_path)
    validation_hash = _sha256(validation_path)
    plan_hash = _sha256(plan_path)
    _require(validation.get("pre_outcome_lock_sha256") == prelock_hash, "validation/prelock hash binding differs")
    _require(plan.get("validation_threshold_sha256") == validation_hash, "plan/validation hash binding differs")
    _require(complete.get("pre_outcome_lock_sha256") == prelock_hash, "COMPLETE/prelock hash binding differs")
    _require(complete.get("validation_threshold_sha256") == validation_hash, "COMPLETE/validation hash binding differs")
    _require(complete.get("test_admission_plan_sha256") == plan_hash, "COMPLETE/plan hash binding differs")

    for name, expected in prelock.get("pre_outcome_artifact_sha256", {}).items():
        path = formal_dir / str(name)
        _require(path.is_file(), f"pre-outcome artifact missing: {name}")
        _require(_sha256(path) == str(expected), f"pre-outcome artifact hash differs: {name}")
    _require(
        prelock.get("candidate_schedule_sha256") == _sha256(formal_dir / "CANDIDATE_SCHEDULE.jsonl"),
        "prelock candidate schedule hash differs",
    )
    _require(
        plan.get("pre_outcome_features_sha256") == _sha256(formal_dir / "PRE_OUTCOME_FEATURES.jsonl"),
        "plan pre-outcome feature hash differs",
    )
    _require(
        plan.get("train_witness_bank_sha256") == _sha256(formal_dir / "TRAIN_WITNESS_BANK.json"),
        "plan train witness bank hash differs",
    )

    validation_threshold = validation.get("threshold")
    plan_threshold = plan.get("threshold")
    _require(plan_threshold == validation_threshold, "frozen plan threshold differs from validation threshold")
    _require(isinstance(plan_threshold, dict), "frozen threshold is not an object")
    _require(plan_threshold.get("mode") == "FINITE", "only a finite frozen threshold is expected")
    _require(plan_threshold.get("comparison") == "score_greater_than_or_equal", "threshold comparator differs")
    threshold = float(plan_threshold.get("value"))
    _require(math.isfinite(threshold), "frozen threshold is non-finite")

    _require(
        prelock_path.stat().st_mtime_ns <= (formal_dir / "TRAIN_EDGE_RESULTS.jsonl").stat().st_mtime_ns,
        "PRE_OUTCOME_LOCK mtime follows train outcomes",
    )
    _require(
        plan_path.stat().st_mtime_ns <= (formal_dir / "TEST_EDGE_RESULTS.jsonl").stat().st_mtime_ns,
        "TEST_ADMISSION_PLAN mtime follows test outcomes",
    )
    return prelock, validation, plan


def _verify_schedule_and_ledgers(
    schedule: Sequence[Mapping[str, Any]],
    results_by_split: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[dict[str, list[Mapping[str, Any]]], dict[str, Any]]:
    _require(schedule, "candidate schedule is empty")
    edge_ids = [str(edge.get("edge_id")) for edge in schedule]
    _require(len(edge_ids) == len(set(edge_ids)), "candidate edge IDs repeat")
    _require(
        [int(edge.get("schedule_index")) for edge in schedule] == list(range(len(schedule))),
        "candidate schedule indices are not contiguous/in order",
    )

    schedule_by_split: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    pairs_seen: set[frozenset[str]] = set()
    for edge in schedule:
        split = str(edge.get("logical_split"))
        _require(split in results_by_split, f"unknown logical split: {split}")
        left, right = _as_pair(edge.get("row_ids", []), f"schedule edge {edge.get('edge_id')}")
        pair = frozenset((left, right))
        _require(pair not in pairs_seen, f"duplicate undirected schedule edge: {edge.get('edge_id')}")
        pairs_seen.add(pair)
        _require(len(edge.get("endpoint_context", [])) == 2, f"schedule endpoint context differs: {edge.get('edge_id')}")
        _require(len(edge.get("row_records", [])) == 2, f"schedule row records differ: {edge.get('edge_id')}")
        _require(int(edge.get("dispatch_order")) == int(edge.get("schedule_index")), "schedule dispatch order differs")
        opening, closing = map(int, edge.get("compatible_arrival_indices", []))
        _require(1 <= closing - opening <= 8, f"schedule W=8 deadline differs: {edge.get('edge_id')}")
        schedule_by_split[split].append(edge)

    identity_fields = (
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
    split_report: dict[str, Any] = {}
    for split in ("train", "validation", "test"):
        expected = schedule_by_split.get(split, [])
        observed = list(results_by_split[split])
        _require(len(observed) == len(expected), f"{split} result count differs from schedule")
        _require(
            [str(row.get("edge_id")) for row in observed]
            == [str(row.get("edge_id")) for row in expected],
            f"{split} result/schedule edge order differs",
        )
        for schedule_edge, result in zip(expected, observed):
            edge_id = str(schedule_edge["edge_id"])
            for field in identity_fields:
                _require(result.get(field) == schedule_edge.get(field), f"{edge_id}: result/schedule {field} differs")
            endpoints = result.get("endpoints", [])
            _require(len(endpoints) == 2, f"{edge_id}: result does not contain two endpoints")
            for index, endpoint in enumerate(endpoints):
                _require(str(endpoint.get("row_id")) == str(schedule_edge["row_ids"][index]), f"{edge_id}: endpoint row identity differs")
                _require(endpoint.get("row_record") == schedule_edge["row_records"][index], f"{edge_id}: endpoint row record differs")
                _require(str(endpoint.get("window_id")) == str(schedule_edge["endpoint_context"][index]["window_id"]), f"{edge_id}: endpoint window differs")
        split_report[split] = {
            "schedule_edges": len(expected),
            "result_edges": len(observed),
            "ordered_identity_exact": True,
        }
    return dict(schedule_by_split), split_report


def _verify_semantic_labels(
    results_by_split: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[dict[str, dict[str, bool]], dict[str, Any]]:
    labels_by_split: dict[str, dict[str, bool]] = {}
    report: dict[str, Any] = {}
    for split in ("train", "validation", "test"):
        incidents: dict[str, list[bool]] = defaultdict(list)
        endpoint_count = 0
        safe_edge_count = 0
        for edge in results_by_split[split]:
            edge_id = str(edge["edge_id"])
            endpoint_labels: list[bool] = []
            for endpoint in edge["endpoints"]:
                endpoint_count += 1
                row_id = str(endpoint["row_id"])
                noop = endpoint.get("native_noop")
                _require(isinstance(noop, dict), f"{edge_id}/{row_id}: native_noop missing")
                _require(noop.get("native_noop_exact") is True, f"{edge_id}/{row_id}: native self-noop is not exact")
                _require(noop.get("native_full_forward_stable_2_of_2") is True, f"{edge_id}/{row_id}: native forward unstable")
                _require(noop.get("noop_full_forward_stable_2_of_2") is True, f"{edge_id}/{row_id}: self-noop forward unstable")
                _require(noop.get("native_observation") == noop.get("noop_observation"), f"{edge_id}/{row_id}: native/self-noop observations differ")
                noop_trace = noop.get("noop_trace", {})
                _require(
                    noop_trace.get("target_applied_raw_sha256") == noop_trace.get("target_native_raw_sha256"),
                    f"{edge_id}/{row_id}: self-noop raw contribution differs",
                )
                _require(endpoint.get("m1_injected_full_forward_stable_2_of_2") is True, f"{edge_id}/{row_id}: M1 injection unstable")
                _require(endpoint.get("m2_injected_full_forward_stable_2_of_2") is True, f"{edge_id}/{row_id}: M2 injection unstable")
                _require(
                    endpoint.get("m1_injected_baseline", {}).get("trace", {}).get("target_applied_raw_sha256")
                    == endpoint.get("independent_m1_output_sha256"),
                    f"{edge_id}/{row_id}: injected M1 hash differs",
                )
                _require(
                    endpoint.get("m2_injected_treatment", {}).get("trace", {}).get("target_applied_raw_sha256")
                    == endpoint.get("paired_m2_output_sha256"),
                    f"{edge_id}/{row_id}: injected M2 hash differs",
                )
                route_delta = endpoint.get("route_delta")
                _require(isinstance(route_delta, dict), f"{edge_id}/{row_id}: route delta missing")
                route_changed = bool(route_delta.get("any_ordered_topk_change"))
                _require(bool(endpoint.get("route_topk_changed")) == route_changed, f"{edge_id}/{row_id}: route change field differs")
                safe = not route_changed
                _require(bool(endpoint.get("semantic_safe")) == safe, f"{edge_id}/{row_id}: endpoint semantic label differs")
                endpoint_labels.append(safe)
                incidents[row_id].append(safe)
            pair_safe = all(endpoint_labels)
            _require(bool(edge.get("pair_safe")) == pair_safe, f"{edge_id}: pair-safe is not endpoint AND")
            safe_edge_count += int(pair_safe)
        labels_by_split[split] = {row_id: all(values) for row_id, values in incidents.items()}
        report[split] = {
            "endpoint_incidents_checked": endpoint_count,
            "unique_endpoints": len(incidents),
            "safe_edges": safe_edge_count,
            "total_edges": len(results_by_split[split]),
            "native_self_noop_exact": True,
            "native_m1_m2_stability_exact": True,
            "endpoint_label_recomputed_as_not_route_topk_changed": True,
            "pair_safe_recomputed_as_endpoint_and": True,
        }
    return labels_by_split, report


def _maximum_safe_matching(
    test_schedule: Sequence[Mapping[str, Any]],
    test_results: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], str]:
    try:
        import networkx as nx
    except ImportError as exc:  # pragma: no cover - the formal environment pins networkx
        raise VerificationError("networkx is required for independent blossom matching") from exc

    result_by_edge = {str(row["edge_id"]): row for row in test_results}
    _require(len(result_by_edge) == len(test_results), "test result edge IDs repeat")
    vertices = sorted({str(row_id) for edge in test_schedule for row_id in edge["row_ids"]})
    graph = nx.Graph()
    graph.add_nodes_from(vertices)
    edge_by_pair: dict[frozenset[str], str] = {}
    safe_edge_ids: list[str] = []
    for edge in sorted(test_schedule, key=lambda value: str(value["edge_id"])):
        edge_id = str(edge["edge_id"])
        pair = frozenset(_as_pair(edge["row_ids"], f"test edge {edge_id}"))
        _require(pair not in edge_by_pair, f"test matching graph has duplicate pair: {edge_id}")
        edge_by_pair[pair] = edge_id
        if bool(result_by_edge[edge_id]["pair_safe"]):
            graph.add_edge(*tuple(pair))
            safe_edge_ids.append(edge_id)

    raw_matching = nx.max_weight_matching(graph, maxcardinality=True)
    pairs = sorted(tuple(sorted(map(str, pair))) for pair in raw_matching)
    matching = [
        {"edge_id": edge_by_pair[frozenset(pair)], "row_ids": list(pair)}
        for pair in pairs
    ]
    covered = 2 * len(matching)
    return {
        "algorithm": "networkx_general_graph_blossom_max_weight_matching_maxcardinality",
        "networkx_version": str(nx.__version__),
        "candidate_edges": len(test_schedule),
        "safe_edges": len(safe_edge_ids),
        "safe_edge_ids": sorted(safe_edge_ids),
        "safe_edge_density": len(safe_edge_ids) / len(test_schedule) if test_schedule else 0.0,
        "unique_vertices": len(vertices),
        "matching_edges": len(matching),
        "covered_vertices": covered,
        "row_coverage": covered / len(vertices) if vertices else 0.0,
        "pair_slot_coverage": len(matching) / (len(vertices) // 2) if len(vertices) >= 2 else 0.0,
        "matching": matching,
    }, str(nx.__version__)


def _rolling_greedy_matching(
    schedule: Sequence[Mapping[str, Any]], admitted_rows: set[str]
) -> list[dict[str, Any]]:
    matched: set[str] = set()
    selected: list[dict[str, Any]] = []
    ordered = sorted(
        schedule,
        key=lambda edge: (
            int(edge["row_records"][0]["document_index"]),
            int(edge["global_arrival_indices"][1]),
            int(edge["global_arrival_indices"][0]),
            str(edge["edge_id"]),
        ),
    )
    for edge in ordered:
        left, right = _as_pair(edge["row_ids"], f"greedy edge {edge['edge_id']}")
        if left not in admitted_rows or right not in admitted_rows:
            continue
        if left in matched or right in matched:
            continue
        matched.update((left, right))
        selected.append({"edge_id": str(edge["edge_id"]), "row_ids": [left, right]})
    return selected


def _recompute_certificate(
    test_schedule: Sequence[Mapping[str, Any]],
    test_results: Sequence[Mapping[str, Any]],
    test_labels: Mapping[str, bool],
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    threshold = float(plan["threshold"]["value"])
    score_rows = plan.get("scores")
    _require(isinstance(score_rows, list), "test admission scores missing")
    score_by_row = {str(row["row_id"]): row for row in score_rows}
    _require(len(score_by_row) == len(score_rows), "test score row IDs repeat")
    _require(set(score_by_row) == set(test_labels), "test score vertex set differs from test outcomes")
    admitted = {
        row_id
        for row_id, row in score_by_row.items()
        if bool(row.get("eligible")) and float(row["score"]) >= threshold
    }
    _require(sorted(admitted) == sorted(map(str, plan.get("admitted_row_ids", []))), "frozen admitted row IDs do not follow bound threshold")

    result_by_edge = {str(row["edge_id"]): row for row in test_results}
    admissible = [
        edge for edge in test_schedule if set(map(str, edge["row_ids"])).issubset(admitted)
    ]
    admissible_ids = [str(edge["edge_id"]) for edge in admissible]
    _require(admissible_ids == list(map(str, plan.get("candidate_admissible_edge_ids", []))), "frozen admissible edge IDs differ")
    greedy = _rolling_greedy_matching(test_schedule, admitted)
    _require(greedy == plan.get("rolling_greedy_matching"), "frozen rolling greedy IDs differ")

    unsafe_endpoints = sorted(row_id for row_id in admitted if not bool(test_labels[row_id]))
    unsafe_admissible = [edge_id for edge_id in admissible_ids if not bool(result_by_edge[edge_id]["pair_safe"])]
    unsafe_greedy = [str(edge["edge_id"]) for edge in greedy if not bool(result_by_edge[str(edge["edge_id"])]["pair_safe"])]
    vertices = len(test_labels)
    document_by_edge = {str(edge["edge_id"]): str(edge["document_sha256"]) for edge in test_schedule}
    all_documents = set(document_by_edge.values())
    positive_documents = {document_by_edge[str(edge["edge_id"])] for edge in greedy}
    return {
        "threshold": threshold,
        "eligible_endpoint_ids": sorted(row_id for row_id, row in score_by_row.items() if bool(row.get("eligible"))),
        "admitted_endpoint_ids": sorted(admitted),
        "unsafe_admitted_endpoint_ids": unsafe_endpoints,
        "admissible_candidate_edge_ids": admissible_ids,
        "unsafe_admissible_edge_ids": unsafe_admissible,
        "greedy_matching": greedy,
        "greedy_edge_ids": [str(edge["edge_id"]) for edge in greedy],
        "unsafe_greedy_edge_ids": unsafe_greedy,
        "total_unique_endpoints": vertices,
        "eligible_endpoints": sum(bool(row.get("eligible")) for row in score_rows),
        "admitted_endpoints": len(admitted),
        "unsafe_admitted_endpoints": len(unsafe_endpoints),
        "admissible_candidate_edges": len(admissible_ids),
        "unsafe_admissible_candidate_edges": len(unsafe_admissible),
        "greedy_executed_pairs": len(greedy),
        "unsafe_greedy_executed_pairs": len(unsafe_greedy),
        "admitted_row_coverage": 2 * len(greedy) / vertices if vertices else 0.0,
        "admitted_pair_slot_coverage": len(greedy) / (vertices // 2) if vertices >= 2 else 0.0,
        "positive_action_document_ids": sorted(positive_documents),
        "positive_action_documents": len(positive_documents),
        "document_coverage": len(positive_documents) / len(all_documents) if all_documents else 0.0,
    }


def _projected_cost(
    *, vertices: int, pairs: int, c1_ms: float, c2_ms: float, overhead_ms: float
) -> dict[str, float | int]:
    _require(vertices >= 2 * pairs, "cost projection has more matched endpoints than vertices")
    _require(c1_ms > 0 and c2_ms > 0 and overhead_ms >= 0, "cost projection inputs are invalid")
    baseline = vertices * c1_ms
    gross = pairs * c2_ms + (vertices - 2 * pairs) * c1_ms
    net = gross + overhead_ms
    return {
        "vertices": vertices,
        "pairs": pairs,
        "c1_ms": c1_ms,
        "c2_ms": c2_ms,
        "all_m1_ms": baseline,
        "gross_runtime_ms": gross,
        "online_overhead_ms": overhead_ms,
        "net_runtime_ms": net,
        "gross_saved_fraction": (baseline - gross) / baseline,
        "net_saved_fraction": (baseline - net) / baseline,
    }


def _recompute_costs(
    raw_cost_artifact: Mapping[str, Any],
    plan: Mapping[str, Any],
    *,
    vertices: int,
    expected_candidate_edges: int,
    oracle_pairs: int,
    certificate_pairs: int,
) -> dict[str, Any]:
    microcost = raw_cost_artifact.get("test_expert_microcost", {})
    raw = microcost.get("raw_aggregate", {})
    m1_samples = list(map(float, raw.get("m1_two_single_calls_ms", [])))
    m2_samples = list(map(float, raw.get("m2_one_pair_call_ms", [])))
    _require(m1_samples and m2_samples, "raw test microcost samples are empty")
    _require(int(raw.get("repeats")) == len(m1_samples) == len(m2_samples), "raw test microcost repeat count differs")
    candidate_edges = int(microcost.get("candidate_edges"))
    _require(candidate_edges > 0, "test microcost candidate edge count is invalid")
    _require(candidate_edges == expected_candidate_edges, "test microcost candidate edge count differs from test schedule")
    m1_median = statistics.median(m1_samples)
    m2_median = statistics.median(m2_samples)
    _float_close(raw.get("m1_median_ms"), m1_median, context="stored raw M1 median")
    _float_close(raw.get("m2_median_ms"), m2_median, context="stored raw M2 median")
    c1_ms = m1_median / (2 * candidate_edges)
    c2_ms = m2_median / candidate_edges

    overhead = plan.get("online_overhead", {})
    _require(int(overhead.get("candidate_edges")) == expected_candidate_edges, "online-overhead candidate edge count differs from test schedule")
    _require(int(overhead.get("unique_endpoints")) == vertices, "online-overhead vertex count differs from test schedule")
    overhead_samples = list(map(float, overhead.get("elapsed_ms", [])))
    _require(overhead_samples, "raw test online-overhead samples are empty")
    _require(int(overhead.get("repeats")) == len(overhead_samples), "online-overhead repeat count differs")
    overhead_median = statistics.median(overhead_samples)
    _float_close(overhead.get("median_total_ms"), overhead_median, context="stored online-overhead median")

    return {
        "raw_timing_recompute": {
            "candidate_edges": candidate_edges,
            "m1_samples_ms": m1_samples,
            "m2_samples_ms": m2_samples,
            "m1_median_ms": m1_median,
            "m2_median_ms": m2_median,
            "c1_ms": c1_ms,
            "c2_ms": c2_ms,
            "online_overhead_samples_ms": overhead_samples,
            "online_overhead_median_ms": overhead_median,
        },
        "natural_oracle": _projected_cost(
            vertices=vertices,
            pairs=oracle_pairs,
            c1_ms=c1_ms,
            c2_ms=c2_ms,
            overhead_ms=0.0,
        ),
        "frozen_certificate": _projected_cost(
            vertices=vertices,
            pairs=certificate_pairs,
            c1_ms=c1_ms,
            c2_ms=c2_ms,
            overhead_ms=overhead_median,
        ),
    }


def _mechanical_verdict(oracle: Mapping[str, Any], certificate: Mapping[str, Any]) -> str:
    natural_cost = oracle["cost_projection"]
    if (
        float(natural_cost["gross_saved_fraction"]) < 0.05
        or float(oracle["matching"]["row_coverage"]) < 0.05
        or int(oracle["matching"]["matching_edges"]) == 0
    ):
        return "NO_GO_NATURAL_SEMANTIC_HEADLINE"
    go = (
        int(certificate["unsafe_admissible_candidate_edges"]) == 0
        and int(certificate["unsafe_greedy_executed_pairs"]) == 0
        and int(certificate["greedy_executed_pairs"]) >= 16
        and float(certificate["admitted_row_coverage"]) >= 0.05
        and int(certificate["positive_action_documents"]) >= 2
        and float(certificate["cost_projection"]["net_saved_fraction"]) > 0.0
    )
    return "GO_SEMANTIC_WITNESS_GATE" if go else "PIVOT_TO_SHADOW_VERIFY"


def _compare_cost(actual: Mapping[str, Any], expected: Mapping[str, Any], context: str) -> None:
    for key in (
        "vertices",
        "pairs",
        "c1_ms",
        "c2_ms",
        "all_m1_ms",
        "gross_runtime_ms",
        "online_overhead_ms",
        "net_runtime_ms",
        "gross_saved_fraction",
        "net_saved_fraction",
    ):
        if key in ("vertices", "pairs"):
            _require(int(actual.get(key)) == int(expected.get(key)), f"{context}.{key} differs")
        else:
            _float_close(actual.get(key), expected.get(key), context=f"{context}.{key}")


def _compare_derived_artifacts(
    formal_dir: Path,
    *,
    matching: Mapping[str, Any],
    certificate: Mapping[str, Any],
    costs: Mapping[str, Any],
    verdict: str,
) -> dict[str, Any]:
    # These derived artifacts are deliberately loaded only after recomputation.
    oracle_artifact = _load_json(formal_dir / "ORACLE_MATCHING.json")
    certificate_artifact = _load_json(formal_dir / "CERTIFICATE_RESULTS.json")
    cost_artifact = _load_json(formal_dir / "COST_PROJECTION.json")

    stored_matching = oracle_artifact.get("matching", {})
    for key in (
        "candidate_edges",
        "safe_edges",
        "unique_vertices",
        "matching_edges",
        "covered_vertices",
    ):
        _require(int(stored_matching.get(key)) == int(matching.get(key)), f"ORACLE_MATCHING.matching.{key} differs")
    for key in ("row_coverage", "pair_slot_coverage"):
        _float_close(stored_matching.get(key), matching.get(key), context=f"ORACLE_MATCHING.matching.{key}")
    _require(stored_matching.get("matching") == matching.get("matching"), "ORACLE_MATCHING matching IDs differ")
    _float_close(oracle_artifact.get("safe_edge_density"), matching.get("safe_edge_density"), context="ORACLE_MATCHING.safe_edge_density")
    _compare_cost(oracle_artifact.get("cost_projection", {}), costs["natural_oracle"], "ORACLE_MATCHING.cost_projection")

    integer_certificate_fields = (
        "total_unique_endpoints",
        "eligible_endpoints",
        "admitted_endpoints",
        "unsafe_admitted_endpoints",
        "admissible_candidate_edges",
        "unsafe_admissible_candidate_edges",
        "greedy_executed_pairs",
        "unsafe_greedy_executed_pairs",
        "positive_action_documents",
    )
    for key in integer_certificate_fields:
        _require(int(certificate_artifact.get(key)) == int(certificate.get(key)), f"CERTIFICATE_RESULTS.{key} differs")
    for key in ("admitted_row_coverage", "admitted_pair_slot_coverage", "document_coverage"):
        _float_close(certificate_artifact.get(key), certificate.get(key), context=f"CERTIFICATE_RESULTS.{key}")
    exact_certificate_fields = (
        "unsafe_admitted_endpoint_ids",
        "unsafe_admissible_edge_ids",
        "unsafe_greedy_edge_ids",
        "greedy_matching",
    )
    for key in exact_certificate_fields:
        _require(certificate_artifact.get(key) == certificate.get(key), f"CERTIFICATE_RESULTS.{key} differs")
    _compare_cost(certificate_artifact.get("cost_projection", {}), costs["frozen_certificate"], "CERTIFICATE_RESULTS.cost_projection")
    _compare_cost(cost_artifact.get("natural_oracle", {}), costs["natural_oracle"], "COST_PROJECTION.natural_oracle")
    _compare_cost(cost_artifact.get("frozen_certificate", {}), costs["frozen_certificate"], "COST_PROJECTION.frozen_certificate")

    # SUMMARY is comparison-only and is the final input opened by this function.
    summary = _load_json(formal_dir / "SUMMARY.json")
    summary_oracle = summary.get("natural_oracle", {})
    summary_certificate = summary.get("frozen_certificate", {})
    _require(summary.get("verdict") == verdict, "SUMMARY verdict differs")
    _require(int(summary.get("test_candidate_edges")) == int(matching["candidate_edges"]), "SUMMARY test candidate edge count differs")
    _float_close(summary_oracle.get("safe_edge_density"), matching["safe_edge_density"], context="SUMMARY natural safe density")
    _require(int(summary_oracle.get("matching_edges")) == int(matching["matching_edges"]), "SUMMARY natural matching count differs")
    _float_close(summary_oracle.get("row_coverage"), matching["row_coverage"], context="SUMMARY natural row coverage")
    _float_close(summary_oracle.get("projected_saving"), costs["natural_oracle"]["gross_saved_fraction"], context="SUMMARY natural saving")
    _require(int(summary_certificate.get("admitted_endpoints")) == int(certificate["admitted_endpoints"]), "SUMMARY certificate admitted endpoints differ")
    _require(int(summary_certificate.get("greedy_executed_pairs")) == int(certificate["greedy_executed_pairs"]), "SUMMARY certificate greedy count differs")
    _require(int(summary_certificate.get("unsafe_greedy_executed_pairs")) == int(certificate["unsafe_greedy_executed_pairs"]), "SUMMARY certificate unsafe greedy count differs")
    _float_close(summary_certificate.get("admitted_row_coverage"), certificate["admitted_row_coverage"], context="SUMMARY certificate row coverage")
    _float_close(summary_certificate.get("net_projected_saving"), costs["frozen_certificate"]["net_saved_fraction"], context="SUMMARY certificate net saving")
    return {
        "ORACLE_MATCHING.json": "MATCH",
        "CERTIFICATE_RESULTS.json": "MATCH",
        "COST_PROJECTION.json": "MATCH",
        "SUMMARY.json": "MATCH_COMPARISON_ONLY",
    }


def verify(formal_output_dir: Path) -> dict[str, Any]:
    formal_dir = formal_output_dir.resolve()
    _require(formal_dir.is_dir(), f"formal output directory missing: {formal_dir}")

    complete, closure = _verify_complete_closure(formal_dir)
    prelock, validation, plan = _verify_pre_outcome_bindings(formal_dir, complete)
    schedule = _load_jsonl(formal_dir / "CANDIDATE_SCHEDULE.jsonl")
    results_by_split = {
        "train": _load_jsonl(formal_dir / "TRAIN_EDGE_RESULTS.jsonl"),
        "validation": _load_jsonl(formal_dir / "VALIDATION_EDGE_RESULTS.jsonl"),
        "test": _load_jsonl(formal_dir / "TEST_EDGE_RESULTS.jsonl"),
    }
    schedule_by_split, identity = _verify_schedule_and_ledgers(schedule, results_by_split)
    labels_by_split, semantic_checks = _verify_semantic_labels(results_by_split)
    matching, networkx_version = _maximum_safe_matching(
        schedule_by_split["test"], results_by_split["test"]
    )
    certificate = _recompute_certificate(
        schedule_by_split["test"],
        results_by_split["test"],
        labels_by_split["test"],
        plan,
    )

    # COST_PROJECTION is read here only for its raw timing samples.  No stored
    # median, c1/c2, projected cost, saving, or verdict is used as an input.
    raw_cost_artifact = _load_json(formal_dir / "COST_PROJECTION.json")
    costs = _recompute_costs(
        raw_cost_artifact,
        plan,
        vertices=int(matching["unique_vertices"]),
        expected_candidate_edges=len(schedule_by_split["test"]),
        oracle_pairs=int(matching["matching_edges"]),
        certificate_pairs=int(certificate["greedy_executed_pairs"]),
    )
    oracle = {
        "safe_edge_density": matching["safe_edge_density"],
        "matching": matching,
        "cost_projection": costs["natural_oracle"],
    }
    certificate["cost_projection"] = costs["frozen_certificate"]
    verdict = _mechanical_verdict(oracle, certificate)

    comparisons = _compare_derived_artifacts(
        formal_dir,
        matching=matching,
        certificate=certificate,
        costs=costs,
        verdict=verdict,
    )
    _require(complete.get("verdict") == verdict, "COMPLETE verdict differs")

    return {
        "schema_version": "semanticfence-sfv2-o1-independent-raw-verification-v1",
        "status": "PASS_INDEPENDENT_RAW_RECOMPUTE",
        "formal_output_dir": str(formal_dir),
        "primary_module_imported": False,
        "summary_not_trusted_as_input": True,
        "dependency_boundary": {
            "stdlib_only_except_networkx": True,
            "networkx_version": networkx_version,
            "networkx_role": "general-graph maximum-cardinality blossom matching only",
        },
        "complete_hash_mtime_closure": closure,
        "pre_outcome_freeze": {
            "pre_outcome_lock_status": prelock["status"],
            "pre_outcome_lock_test_outcome_count": prelock["test_outcome_count_at_lock"],
            "validation_threshold_test_outcome_count": validation["test_semantic_outcome_count_at_threshold_freeze"],
            "test_admission_plan_status": plan["status"],
            "test_admission_plan_test_outcome_count": plan["test_semantic_outcome_count_at_plan_freeze"],
            "threshold": plan["threshold"],
            "threshold_hash_binding_exact": True,
            "prelock_mtime_not_after_train_outcomes": True,
            "plan_mtime_not_after_test_outcomes": True,
        },
        "schedule_result_identity": identity,
        "semantic_integrity": semantic_checks,
        "natural_oracle_recomputed": oracle,
        "certificate_recomputed": certificate,
        "cost_recomputed_from_raw_samples": costs,
        "mechanical_verdict": verdict,
        "derived_artifact_comparisons_after_recompute": comparisons,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-output-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    formal_dir = args.formal_output_dir.resolve()
    output_path = args.output.resolve()
    if _is_within(output_path, formal_dir):
        raise VerificationError("audit output must not be inside the sealed formal output directory")
    report = verify(formal_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "verdict": report["mechanical_verdict"],
        "output": str(output_path),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError, VerificationError) as exc:
        print(f"VERIFICATION_FAILED: {exc}", file=sys.stderr)
        raise SystemExit(2)
