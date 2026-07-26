#!/usr/bin/env python3
"""Outcome-blind instance construction for the PhaseMap oracle gate.

This module owns route selection, structural pairing, and causal pre-t0 FIFO
manifests only.  It deliberately contains no policy optimizer, metric gate, or
scientific verdict.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from .explore_receiver_matched_milp import load_verified_joins
except ImportError:  # pragma: no cover
    from explore_receiver_matched_milp import load_verified_joins  # type: ignore


SCHEMA_VERSION = "phasemap-v1-instance-manifest"
SPLIT_NAMES = ("selection", "holdout")
Q_BITS = ("q0", "q1")
J_BITS = ("j0", "j1")
DEFAULT_DEPTHS = (8, 16)
MODELS = ("olmoe", "llmjp")
FROZEN_HIDDEN_BY_MODEL = {"olmoe": 2048, "llmjp": 512}
SELECTION_CERTIFICATE_SCHEMA = "phasemap-v1-route-selection-certificate"
SPLIT_BUNDLE_SCHEMA = "phasemap-split-instance-bundle-v1"
DEFAULT_CANDIDATE_JOINS_PER_REQUEST = 128
ROUTE_PROVENANCE_FIELDS = (
    "manifest_sha256",
    "route_trace_file_sha256",
    "route_phase4_signoff_sha256",
    "producer_signoff_file_sha256",
    "data_manifest_sha256",
    "placement_manifest_sha256",
    "placement_file_sha256",
)


class PhaseMapInstanceError(RuntimeError):
    """A frozen route, pairing, identity, or reachability invariant failed."""


def canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise PhaseMapInstanceError("value is not strict canonical JSON") from exc


def object_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _require_int(row: Mapping[str, Any], field: str) -> int:
    value = row.get(field)
    if type(value) is not int:
        raise PhaseMapInstanceError(f"{field} is not an integer")
    return value


def _require_text(row: Mapping[str, Any], field: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value:
        raise PhaseMapInstanceError(f"{field} is not a non-empty string")
    return value


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _model_identity(metadata: Mapping[str, Any], model: str) -> dict[str, Any]:
    if metadata.get("model_key", model) != model:
        raise PhaseMapInstanceError("route metadata model identity drift")
    revision = metadata.get("model_revision")
    top_k = metadata.get("top_k")
    expected_candidates = metadata.get(
        "expected_join_candidates_per_request", DEFAULT_CANDIDATE_JOINS_PER_REQUEST
    )
    if (
        not isinstance(revision, str)
        or not revision
        or type(top_k) is not int
        or top_k <= 0
        or type(expected_candidates) is not int
        or expected_candidates <= 0
        or not _is_sha256(metadata.get("data_manifest_sha256"))
        or not _is_sha256(metadata.get("placement_manifest_sha256"))
    ):
        raise PhaseMapInstanceError("route metadata lacks frozen model/data/placement/top-k identity")
    return {
        "model_key": model,
        "model_revision": revision,
        "data_manifest_sha256": metadata["data_manifest_sha256"],
        "placement_manifest_sha256": metadata["placement_manifest_sha256"],
        "top_k": top_k,
        "expected_join_candidates_per_request": expected_candidates,
    }


def _route_provenance(metadata: Mapping[str, Any], model: str) -> dict[str, Any]:
    identity = _model_identity(metadata, model)
    missing = [field for field in ROUTE_PROVENANCE_FIELDS if not _is_sha256(metadata.get(field))]
    if missing:
        raise PhaseMapInstanceError(f"reviewed route provenance is incomplete: {missing}")
    return {
        "model_identity": identity,
        **{field: metadata[field] for field in ROUTE_PROVENANCE_FIELDS},
    }


def full_sibling_identity(
    row: Mapping[str, Any], *, model: str, metadata: Mapping[str, Any]
) -> dict[str, Any]:
    """Return the frozen contribution identity without timing or outcomes."""

    model_identity = _model_identity(metadata, model)
    row_model = row.get("model_key", model)
    if row_model != model:
        raise PhaseMapInstanceError("sibling model identity drift")
    for field in (
        "model_revision",
        "data_manifest_sha256",
        "placement_manifest_sha256",
    ):
        if row.get(field, model_identity[field]) != model_identity[field]:
            raise PhaseMapInstanceError(f"sibling {field} disagrees with route metadata")
    receiver = _require_int(row, "receiver_rank")
    identity = {
        "model_key": model,
        "model_revision": model_identity["model_revision"],
        "data_manifest_sha256": row.get(
            "data_manifest_sha256", model_identity["data_manifest_sha256"]
        ),
        "placement_manifest_sha256": row.get(
            "placement_manifest_sha256", model_identity["placement_manifest_sha256"]
        ),
        "request_id": _require_text(row, "request_id"),
        "forward_id": _require_text(row, "forward_id"),
        "layer_id": _require_int(row, "layer_id"),
        "token_position": _require_int(row, "token_position"),
        "epoch": _require_int(row, "epoch"),
        "topk_slot": _require_int(row, "topk_slot"),
        "expert_id": _require_int(row, "expert_id"),
        "sender_rank": _require_int(row, "sender_rank"),
        "receiver_rank": receiver,
    }
    canonical_json_bytes(identity)
    return identity


def _normalize_join(
    join: Mapping[str, Any], *, model: str, metadata: Mapping[str, Any]
) -> dict[str, Any]:
    siblings_value = join.get("siblings")
    if not isinstance(siblings_value, Sequence) or isinstance(siblings_value, (str, bytes)):
        raise PhaseMapInstanceError("join siblings are missing")
    siblings = [dict(row) for row in siblings_value if isinstance(row, Mapping)]
    if len(siblings) != len(siblings_value) or not siblings:
        raise PhaseMapInstanceError("join has a malformed or empty sibling set")
    model_identity = _model_identity(metadata, model)
    identities = [full_sibling_identity(row, model=model, metadata=metadata) for row in siblings]
    identities.sort(key=lambda value: (int(value["topk_slot"]), object_sha256(value)))
    slots = [int(value["topk_slot"]) for value in identities]
    if len(identities) != model_identity["top_k"] or slots != list(
        range(model_identity["top_k"])
    ):
        raise PhaseMapInstanceError("join does not retain the full canonical top-k sibling set")
    if len({object_sha256(value) for value in identities}) != len(identities):
        raise PhaseMapInstanceError("duplicate sibling identity")
    requests = {str(value["request_id"]) for value in identities}
    layers = {int(value["layer_id"]) for value in identities}
    positions = {int(value["token_position"]) for value in identities}
    receivers = {int(value["receiver_rank"]) for value in identities}
    forwards = {str(value["forward_id"]) for value in identities}
    epochs = {int(value["epoch"]) for value in identities}
    if any(len(values) != 1 for values in (requests, layers, positions, receivers, forwards, epochs)):
        raise PhaseMapInstanceError("sibling identities do not close one native join")
    identity_tuples = {
        (
            value["model_key"],
            value["model_revision"],
            value["data_manifest_sha256"],
            value["placement_manifest_sha256"],
        )
        for value in identities
    }
    if identity_tuples != {
        (
            model_identity["model_key"],
            model_identity["model_revision"],
            model_identity["data_manifest_sha256"],
            model_identity["placement_manifest_sha256"],
        )
    }:
        raise PhaseMapInstanceError("sibling-wide model/data/placement identity drift")
    request_id = next(iter(requests))
    receiver_rank = next(iter(receivers))
    if (
        str(join.get("request_id")) != request_id
        or _require_int(join, "layer_id") != next(iter(layers))
        or _require_int(join, "token_position") != next(iter(positions))
        or _require_int(join, "receiver_rank") != receiver_rank
    ):
        raise PhaseMapInstanceError("join wrapper disagrees with sibling identity")
    join_identity = {
        "model_key": model,
        "model_revision": identities[0]["model_revision"],
        "data_manifest_sha256": identities[0]["data_manifest_sha256"],
        "placement_manifest_sha256": identities[0]["placement_manifest_sha256"],
        "request_id": request_id,
        "forward_id": next(iter(forwards)),
        "layer_id": next(iter(layers)),
        "token_position": next(iter(positions)),
        "epoch": next(iter(epochs)),
        "receiver_rank": receiver_rank,
        "topk_siblings": [
            {
                "topk_slot": value["topk_slot"],
                "expert_id": value["expert_id"],
                "sender_rank": value["sender_rank"],
                "receiver_rank": value["receiver_rank"],
            }
            for value in identities
        ],
    }
    sibling_records = [
        {"identity": value, "full_sibling_key": object_sha256(value)} for value in identities
    ]
    return {
        "request_id": request_id,
        "receiver_rank": receiver_rank,
        "layer_id": next(iter(layers)),
        "token_position": next(iter(positions)),
        "full_join_identity": join_identity,
        "full_join_key": object_sha256(join_identity),
        "siblings": sibling_records,
    }


def normalize_join_universe(
    joins: Sequence[Mapping[str, Any]], metadata: Mapping[str, Any], model: str
) -> list[dict[str, Any]]:
    """Normalize and census the complete reviewed candidate join universe."""

    model_identity = _model_identity(metadata, model)
    normalized = [_normalize_join(join, model=model, metadata=metadata) for join in joins]
    if not normalized:
        raise PhaseMapInstanceError("normalized join universe is empty")
    normalized.sort(key=lambda row: (str(row["request_id"]), str(row["full_join_key"])))
    by_request: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in normalized:
        by_request[str(row["request_id"])].append(row)
    expected = int(model_identity["expected_join_candidates_per_request"])
    if len(by_request) != 64 or any(len(rows) != expected for rows in by_request.values()):
        raise PhaseMapInstanceError("complete normalized join universe census mismatch")
    if len({str(row["full_join_key"]) for row in normalized}) != len(normalized):
        raise PhaseMapInstanceError("normalized join universe contains duplicate join identity")
    return normalized


def _canonical_split_from_normalized(
    normalized: Sequence[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    by_request: dict[str, list[dict[str, Any]]] = defaultdict(list)
    receiver_by_request: dict[str, int] = {}
    for join in normalized:
        request = str(join["request_id"])
        receiver = int(join["receiver_rank"])
        previous = receiver_by_request.setdefault(request, receiver)
        if previous != receiver:
            raise PhaseMapInstanceError("request output receiver identity drift")
        by_request[request].append(join)
    by_receiver: dict[int, list[str]] = defaultdict(list)
    for request, receiver in receiver_by_request.items():
        by_receiver[receiver].append(request)
    if set(by_receiver) != set(range(8)) or any(len(values) != 8 for values in by_receiver.values()):
        raise PhaseMapInstanceError("expected exactly eight calibration requests per receiver")

    request_splits: dict[str, set[str]] = {name: set() for name in SPLIT_NAMES}
    for receiver in range(8):
        ordered = sorted(
            by_receiver[receiver],
            key=lambda request: object_sha256(["phasemap-v1-split", request]),
        )
        request_splits["selection"].update(ordered[:4])
        request_splits["holdout"].update(ordered[4:])

    result: dict[str, list[dict[str, Any]]] = {}
    for split_name in SPLIT_NAMES:
        selected: list[dict[str, Any]] = []
        for request in sorted(request_splits[split_name]):
            candidates = by_request[request]
            chosen = min(
                candidates,
                key=lambda value: object_sha256(
                    ["phasemap-v1-join", value["full_join_identity"]]
                ),
            )
            selected.append(dict(chosen))
        if len(selected) != 32 or len({row["request_id"] for row in selected}) != 32:
            raise PhaseMapInstanceError("split does not contain 32 distinct native requests")
        result[split_name] = selected
    if {row["request_id"] for row in result["selection"]} & {
        row["request_id"] for row in result["holdout"]
    }:
        raise PhaseMapInstanceError("selection and holdout requests overlap")
    return result


def canonical_split_and_select(
    joins: Sequence[Mapping[str, Any]], metadata: Mapping[str, Any], model: str
) -> dict[str, list[dict[str, Any]]]:
    """Split 8 requests/receiver and select one join/request without outcomes."""

    return _canonical_split_from_normalized(
        normalize_join_universe(joins, metadata, model)
    )


def _pair_spec(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any] | None:
    request_ids = sorted((str(left["request_id"]), str(right["request_id"])))
    if request_ids[0] == request_ids[1]:
        return None
    records = {str(left["request_id"]): left, str(right["request_id"]): right}
    a, b = (records[request_ids[0]], records[request_ids[1]])
    if int(a["receiver_rank"]) == int(b["receiver_rank"]):
        return None
    pair_key = object_sha256(["phasemap-v1-pair", request_ids[0], request_ids[1]])

    def by_sender(record: Mapping[str, Any]) -> dict[int, list[dict[str, Any]]]:
        result: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for sibling in record["siblings"]:
            result[int(sibling["identity"]["sender_rank"])].append(dict(sibling))
        return result

    a_senders, b_senders = by_sender(a), by_sender(b)
    common = set(a_senders) & set(b_senders)
    if len(common) < 2:
        return None
    decision_senders = sorted(
        common,
        key=lambda sender: object_sha256(
            ["phasemap-v1-decision-sender", pair_key, sender]
        ),
    )[:2]
    decision: dict[str, dict[str, str]] = {request_ids[0]: {}, request_ids[1]: {}}
    carriers: dict[str, list[dict[str, Any]]] = {}
    for request in request_ids:
        record = records[request]
        sender_map = by_sender(record)
        selected_keys: set[str] = set()
        for sender in decision_senders:
            chosen = min(
                sender_map[sender],
                key=lambda sibling: object_sha256(
                    [
                        "phasemap-v1-decision-contribution",
                        pair_key,
                        request,
                        sibling["full_sibling_key"],
                    ]
                ),
            )
            selected_keys.add(str(chosen["full_sibling_key"]))
            decision[request][str(sender)] = str(chosen["full_sibling_key"])
        remaining = [
            dict(sibling)
            for sibling in record["siblings"]
            if str(sibling["full_sibling_key"]) not in selected_keys
        ]
        if len(remaining) < 4:
            return None
        remaining.sort(
            key=lambda sibling: object_sha256(
                ["phasemap-v1-carrier", sibling["identity"]]
            )
        )
        carriers[request] = remaining
    edge_key = object_sha256(
        ["phasemap-v1-edge", request_ids[0], request_ids[1]]
    )
    return {
        "pair_key": pair_key,
        "edge_key": edge_key,
        "request_a": request_ids[0],
        "request_b": request_ids[1],
        "receiver_a": int(a["receiver_rank"]),
        "receiver_b": int(b["receiver_rank"]),
        "decision_senders": decision_senders,
        "decision_contributions": decision,
        "phase_carriers": carriers,
        "joins": {request_ids[0]: dict(a), request_ids[1]: dict(b)},
    }


def _perfect_matching_exists(vertices: frozenset[str], edges: Sequence[Mapping[str, Any]]) -> bool:
    if len(vertices) % 2:
        return False
    adjacency: dict[str, list[tuple[str, str]]] = {value: [] for value in vertices}
    for edge in edges:
        left, right = str(edge["request_a"]), str(edge["request_b"])
        if left in vertices and right in vertices:
            adjacency[left].append((str(edge["edge_key"]), right))
            adjacency[right].append((str(edge["edge_key"]), left))
    memo: dict[frozenset[str], bool] = {}

    def solve(remaining: frozenset[str]) -> bool:
        if not remaining:
            return True
        if remaining in memo:
            return memo[remaining]
        candidates = []
        for vertex in remaining:
            neighbors = sorted(
                (edge_key, other)
                for edge_key, other in adjacency[vertex]
                if other in remaining
            )
            candidates.append((len(neighbors), vertex, neighbors))
        _degree, vertex, neighbors = min(candidates, key=lambda item: (item[0], item[1]))
        for _edge_key, other in neighbors:
            if solve(remaining - {vertex, other}):
                memo[remaining] = True
                return True
        memo[remaining] = False
        return False

    return solve(vertices)


def canonical_perfect_matching(selected: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return the lexicographically canonical route-structural perfect matching."""

    by_request = {str(row["request_id"]): row for row in selected}
    if len(by_request) != len(selected) or len(by_request) % 2:
        raise PhaseMapInstanceError("matching input is not an even set of distinct requests")
    edges = []
    request_ids = sorted(by_request)
    for left_index, left in enumerate(request_ids):
        for right in request_ids[left_index + 1 :]:
            edge = _pair_spec(by_request[left], by_request[right])
            if edge is not None:
                edges.append(edge)
    edges.sort(key=lambda edge: str(edge["edge_key"]))
    remaining = frozenset(request_ids)
    if not _perfect_matching_exists(remaining, edges):
        raise PhaseMapInstanceError("BLOCKED_ROUTE_SUPPORT: no perfect matching")
    chosen: list[dict[str, Any]] = []
    while remaining:
        feasible_edge = None
        for edge in edges:
            left, right = str(edge["request_a"]), str(edge["request_b"])
            if left not in remaining or right not in remaining:
                continue
            after = remaining - {left, right}
            if _perfect_matching_exists(after, edges):
                feasible_edge = edge
                break
        if feasible_edge is None:
            raise PhaseMapInstanceError("canonical matching construction lost feasibility")
        chosen.append(dict(feasible_edge))
        remaining -= {str(feasible_edge["request_a"]), str(feasible_edge["request_b"])}
    if len(chosen) * 2 != len(selected):
        raise PhaseMapInstanceError("perfect matching cardinality drift")
    return chosen


def _fifo_replay(jobs: list[dict[str, Any]], service_us: float) -> list[dict[str, Any]]:
    available = -math.inf
    ledger = []
    for job in sorted(jobs, key=lambda value: (float(value["arrival_us"]), str(value["job_key"]))):
        start = max(float(job["arrival_us"]), available)
        end = start + service_us
        ledger.append({**job, "start_us": start, "end_us": end})
        available = end
    return ledger


def _world(
    pair: Mapping[str, Any], *, q_bit: str, j_bit: str, service_us: float,
    low_depth: int, high_depth: int,
) -> dict[str, Any]:
    request_a, request_b = str(pair["request_a"]), str(pair["request_b"])
    receiver_a, receiver_b = int(pair["receiver_a"]), int(pair["receiver_b"])
    pair_models = {
        str(row["identity"]["model_key"])
        for request in (request_a, request_b)
        for row in pair["joins"][request]["siblings"]
    }
    if len(pair_models) != 1:
        raise PhaseMapInstanceError("pair siblings do not share one model identity")
    model = next(iter(pair_models))
    depth_by_receiver = (
        {receiver_a: low_depth, receiver_b: high_depth}
        if q_bit == "q0"
        else {receiver_a: high_depth, receiver_b: low_depth}
    )
    near_request, far_request = (
        (request_a, request_b) if j_bit == "j0" else (request_b, request_a)
    )
    unfinished_count = {near_request: 1, far_request: 4}
    receiver_by_request = {request_a: receiver_a, request_b: receiver_b}
    unfinished_by_request: dict[str, list[dict[str, Any]]] = {}
    committed_by_request: dict[str, list[dict[str, Any]]] = {}
    for request in (request_a, request_b):
        carriers = list(pair["phase_carriers"][request])
        count = unfinished_count[request]
        unfinished_by_request[request] = carriers[:count]
        committed_by_request[request] = carriers[count:]

    jobs_by_receiver: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for request in (request_a, request_b):
        receiver = receiver_by_request[request]
        committed = committed_by_request[request]
        for ordinal, sibling in enumerate(committed):
            jobs_by_receiver[receiver].append(
                {
                    "job_key": str(sibling["full_sibling_key"]),
                    "kind": "foreground_committed_carrier",
                    "request_id": request,
                    "arrival_us": -service_us * (64 + len(committed) - ordinal),
                }
            )

    for request in (request_a, request_b):
        receiver = receiver_by_request[request]
        foreground = unfinished_by_request[request]
        depth = depth_by_receiver[receiver]
        if len(foreground) > depth:
            raise PhaseMapInstanceError("phase carriers exceed frozen receiver depth")
        unfinished_jobs = [
            {
                "job_key": str(sibling["full_sibling_key"]),
                "kind": "foreground_unfinished_carrier",
                "request_id": request,
            }
            for sibling in foreground
        ]
        for ordinal in range(depth - len(foreground)):
            key_value = [
                "phasemap-v1-background",
                model,
                pair["pair_key"],
                receiver,
                q_bit,
                j_bit,
                ordinal,
            ]
            unfinished_jobs.append(
                {
                    "job_key": object_sha256(key_value),
                    "kind": "background_unfinished",
                    "request_id": None,
                    "background_identity": key_value,
                }
            )
        unfinished_jobs.sort(
            key=lambda job: object_sha256(["phasemap-v1-fifo", job["job_key"]])
        )
        eps = service_us / (4 * (depth + 1))
        for index, job in enumerate(unfinished_jobs):
            job["arrival_us"] = -service_us / 2 - (depth - 1 - index) * eps
            jobs_by_receiver[receiver].append(job)

    fifo_ledgers: dict[str, list[dict[str, Any]]] = {}
    q_rows = []
    for receiver in sorted(depth_by_receiver):
        ledger = _fifo_replay(jobs_by_receiver[receiver], service_us)
        fifo_ledgers[str(receiver)] = ledger
        unfinished = [row for row in ledger if "unfinished" in row["kind"]]
        committed = [row for row in ledger if row["kind"] == "foreground_committed_carrier"]
        if len(unfinished) != depth_by_receiver[receiver] or any(row["end_us"] <= 0 for row in unfinished):
            raise PhaseMapInstanceError("unfinished FIFO depth/reachability mismatch")
        earliest_unfinished = min(float(row["arrival_us"]) for row in unfinished)
        if any(float(row["end_us"]) >= earliest_unfinished for row in committed):
            raise PhaseMapInstanceError("committed carrier did not drain before unfinished arrivals")
        availability = max(float(row["end_us"]) for row in unfinished)
        q_rows.append(
            {
                "receiver_rank": receiver,
                "unfinished_depth": len(unfinished),
                "unfinished_work_us": availability,
                "availability_us": availability,
            }
        )

    j_rows = []
    for request in (request_a, request_b):
        join = pair["joins"][request]
        expected = sorted(str(row["full_sibling_key"]) for row in join["siblings"])
        committed = sorted(
            str(row["full_sibling_key"]) for row in committed_by_request[request]
        )
        queued = sorted(
            str(row["full_sibling_key"]) for row in unfinished_by_request[request]
        )
        ready_unsent = sorted(pair["decision_contributions"][request].values())
        if set(committed) & (set(queued) | set(ready_unsent)):
            raise PhaseMapInstanceError("join phase sets overlap")
        if set(committed) | set(queued) | set(ready_unsent) != set(expected):
            raise PhaseMapInstanceError("join phase census does not cover full native siblings")
        j_rows.append(
            {
                "request_id": request,
                "full_join_key": join["full_join_key"],
                "expected_siblings": expected,
                "committed_siblings": committed,
                "queued_siblings": queued,
                "ready_unsent_siblings": ready_unsent,
                "phase_carrier_unfinished": len(queued),
                "deficit": len(expected) - len(committed),
            }
        )
    q_rows.sort(key=lambda row: int(row["receiver_rank"]))
    j_rows.sort(key=lambda row: str(row["request_id"]))
    q_observation = {"receiver_state": q_rows}
    j_observation = {"join_state": j_rows}
    b0_observation = {
        "q_multiset": sorted(
            (row["unfinished_depth"], row["unfinished_work_us"], row["availability_us"])
            for row in q_rows
        ),
        "j_multiset": sorted(
            (row["phase_carrier_unfinished"], row["deficit"]) for row in j_rows
        ),
    }
    decision_keys = sorted(
        value
        for request in (request_a, request_b)
        for value in pair["decision_contributions"][request].values()
    )
    carrier_keys = sorted(
        str(row["full_sibling_key"])
        for request in (request_a, request_b)
        for row in pair["phase_carriers"][request]
    )
    sender_history = [
        {
            "full_sibling_key": key,
            "event": "decision_ready_unsent",
            "timestamp_us": 0.0,
        }
        for key in decision_keys
    ] + [
        {
            "full_sibling_key": key,
            "event": "send_complete_no_commit_ack",
            "timestamp_us": -service_us * (256 + ordinal + 1),
        }
        for ordinal, key in enumerate(carrier_keys)
    ]
    sender_history.sort(
        key=lambda row: (str(row["full_sibling_key"]), str(row["event"]))
    )
    send_complete_by_key = {
        str(row["full_sibling_key"]): float(row["timestamp_us"])
        for row in sender_history
        if row["event"] == "send_complete_no_commit_ack"
    }
    receiver_arrival_by_key = {
        str(job["job_key"]): float(job["arrival_us"])
        for ledger in fifo_ledgers.values()
        for job in ledger
        if job["kind"] in {
            "foreground_committed_carrier",
            "foreground_unfinished_carrier",
        }
    }
    if set(send_complete_by_key) != set(carrier_keys) or set(receiver_arrival_by_key) != set(
        carrier_keys
    ):
        raise PhaseMapInstanceError("sender/receiver carrier transit census mismatch")
    receiver_transit_ledger = [
        {
            "full_sibling_key": key,
            "sender_send_complete_us": send_complete_by_key[key],
            "receiver_arrival_us": receiver_arrival_by_key[key],
            "hidden_transit_us": receiver_arrival_by_key[key] - send_complete_by_key[key],
        }
        for key in carrier_keys
    ]
    if any(float(row["hidden_transit_us"]) < 0 for row in receiver_transit_ledger):
        raise PhaseMapInstanceError("negative hidden sender-to-receiver transit")
    return {
        "world_id": f"{q_bit}{j_bit}",
        "q_bit": q_bit,
        "j_bit": j_bit,
        "depth_by_receiver": {str(key): value for key, value in sorted(depth_by_receiver.items())},
        "fifo_ledgers": fifo_ledgers,
        "q_observation": q_observation,
        "j_observation": j_observation,
        "observation_hashes": {
            "B0": object_sha256(b0_observation),
            "Q": object_sha256(q_observation),
            "J": object_sha256(j_observation),
            "R": object_sha256({"Q": q_observation, "J": j_observation}),
        },
        "sender_history": sender_history,
        "sender_history_hash": object_sha256(sender_history),
        "receiver_transit_ledger": receiver_transit_ledger,
        "commit_unfinished_census": j_rows,
    }


def build_world_manifest(
    pair: Mapping[str, Any],
    unpack_service_us: float,
    depths: tuple[int, int] = DEFAULT_DEPTHS,
    *,
    control_mode: str = "primary",
) -> dict[str, Any]:
    """Construct the frozen 2x2 reachable worlds for one structural pair."""

    if not math.isfinite(unpack_service_us) or unpack_service_us <= 0:
        raise PhaseMapInstanceError("unpack service must be positive and finite")
    low, high = depths
    if type(low) is not int or type(high) is not int or low <= 0 or high < low or high > 16:
        raise PhaseMapInstanceError("invalid frozen receiver depth pair")
    if control_mode not in {"primary", "equal_q"}:
        raise PhaseMapInstanceError("unknown PhaseMap world construction mode")
    if control_mode == "equal_q" and low != high:
        raise PhaseMapInstanceError("equal-Q control requires equal depths")
    if control_mode == "primary" and low == high:
        raise PhaseMapInstanceError("primary PhaseMap worlds require unequal depths")
    worlds = [
        _world(
            pair,
            q_bit=q_bit,
            j_bit=j_bit,
            service_us=float(unpack_service_us),
            low_depth=low,
            high_depth=high,
        )
        for q_bit in Q_BITS
        for j_bit in J_BITS
    ]
    by_id = {row["world_id"]: row for row in worlds}
    fixed_q = all(
        by_id[f"{q_bit}j0"]["observation_hashes"]["Q"]
        == by_id[f"{q_bit}j1"]["observation_hashes"]["Q"]
        for q_bit in Q_BITS
    )
    fixed_j = all(
        by_id[f"q0{j_bit}"]["observation_hashes"]["J"]
        == by_id[f"q1{j_bit}"]["observation_hashes"]["J"]
        for j_bit in J_BITS
    )
    class_counts = {
        arm: len({row["observation_hashes"][arm] for row in worlds})
        for arm in ("B0", "Q", "J", "R")
    }
    sender_hashes = {row["sender_history_hash"] for row in worlds}
    sender_ledgers = {canonical_json_bytes(row["sender_history"]) for row in worlds}
    certificate = {
        "fixed_q_flip_j_q_observation_byte_identical": fixed_q,
        "fixed_j_flip_q_j_observation_byte_identical": fixed_j,
        "sender_history_byte_identical": len(sender_hashes) == 1
        and len(sender_ledgers) == 1,
        "all_hidden_transit_nonnegative": all(
            float(row["hidden_transit_us"]) >= 0
            for world in worlds
            for row in world["receiver_transit_ledger"]
        ),
        "observation_class_counts": class_counts,
        "all_worlds_replayed_from_empty_fifo": True,
        "all_pre_t0_arrivals_negative": all(
            float(job["arrival_us"]) < 0
            for world in worlds
            for ledger in world["fifo_ledgers"].values()
            for job in ledger
        ),
    }
    expected_classes = (
        {"B0": 1, "Q": 2, "J": 2, "R": 4}
        if control_mode == "primary"
        else {"B0": 1, "Q": 1, "J": 2, "R": 2}
    )
    if not all(
        (
            certificate["fixed_q_flip_j_q_observation_byte_identical"],
            certificate["fixed_j_flip_q_j_observation_byte_identical"],
            certificate["sender_history_byte_identical"],
            certificate["all_hidden_transit_nonnegative"],
            certificate["all_pre_t0_arrivals_negative"],
            class_counts == expected_classes,
        )
    ):
        raise PhaseMapInstanceError("BLOCKED_PHASE_NOT_RECEIVER_PRIVATE")
    request_ids = (str(pair["request_a"]), str(pair["request_b"]))
    decision_records: dict[str, dict[str, dict[str, Any]]] = {}
    for request in request_ids:
        siblings_by_key = {
            str(row["full_sibling_key"]): row for row in pair["joins"][request]["siblings"]
        }
        decision_records[request] = {}
        for sender, sibling_key in pair["decision_contributions"][request].items():
            record = siblings_by_key.get(str(sibling_key))
            if record is None:
                raise PhaseMapInstanceError("decision contribution is absent from the native join")
            identity = record["identity"]
            if (
                str(identity["request_id"]) != request
                or int(identity["sender_rank"]) != int(sender)
                or int(identity["receiver_rank"]) != int(pair["joins"][request]["receiver_rank"])
            ):
                raise PhaseMapInstanceError("decision contribution identity mapping drift")
            decision_records[request][str(sender)] = dict(record)
    pair_identity = {
        "pair_key": pair["pair_key"],
        "edge_key": pair["edge_key"],
        "request_a": pair["request_a"],
        "request_b": pair["request_b"],
        "receiver_a": pair["receiver_a"],
        "receiver_b": pair["receiver_b"],
        "joins": {
            request: dict(pair["joins"][request]) for request in request_ids
        },
        "decision_senders": list(pair["decision_senders"]),
        "decision_contributions": decision_records,
        "phase_carriers": {
            request: [dict(row) for row in pair["phase_carriers"][request]]
            for request in request_ids
        },
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "control_mode": control_mode,
        "pair_key": pair["pair_key"],
        "request_ids": [pair["request_a"], pair["request_b"]],
        "receivers": [pair["receiver_a"], pair["receiver_b"]],
        "decision_senders": list(pair["decision_senders"]),
        "decision_contributions": pair["decision_contributions"],
        "pair_identity": pair_identity,
        "unpack_service_us": float(unpack_service_us),
        "depths": {"low": low, "high": high},
        "worlds": worlds,
        "reachability_certificate": certificate,
    }
    return {**payload, "manifest_sha256": object_sha256(payload)}


def rebuild_from_pair_identity(
    pair_identity: Mapping[str, Any],
    unpack_service_us: float,
    depths: tuple[int, int] = DEFAULT_DEPTHS,
    *,
    mode: str = "primary",
) -> dict[str, Any]:
    """Rebuild a primary/equal-Q world manifest from its self-contained identity."""

    requests = (str(pair_identity.get("request_a")), str(pair_identity.get("request_b")))
    joins_value = pair_identity.get("joins")
    decisions_value = pair_identity.get("decision_contributions")
    carriers_value = pair_identity.get("phase_carriers")
    if not all(isinstance(value, Mapping) for value in (joins_value, decisions_value, carriers_value)):
        raise PhaseMapInstanceError("pair_identity is missing builder inputs")
    decision_keys: dict[str, dict[str, str]] = {}
    for request in requests:
        request_decisions = decisions_value.get(request)
        if not isinstance(request_decisions, Mapping):
            raise PhaseMapInstanceError("pair_identity decision mapping is malformed")
        decision_keys[request] = {}
        for sender, record in request_decisions.items():
            if not isinstance(record, Mapping) or not isinstance(record.get("full_sibling_key"), str):
                raise PhaseMapInstanceError("pair_identity decision record is malformed")
            decision_keys[request][str(sender)] = str(record["full_sibling_key"])
    pair = {
        "pair_key": pair_identity.get("pair_key"),
        "edge_key": pair_identity.get("edge_key"),
        "request_a": requests[0],
        "request_b": requests[1],
        "receiver_a": pair_identity.get("receiver_a"),
        "receiver_b": pair_identity.get("receiver_b"),
        "joins": {request: dict(joins_value[request]) for request in requests},
        "decision_senders": list(pair_identity.get("decision_senders", [])),
        "decision_contributions": decision_keys,
        "phase_carriers": {
            request: [dict(row) for row in carriers_value[request]] for request in requests
        },
    }
    return build_world_manifest(
        pair,
        unpack_service_us,
        depths,
        control_mode=mode,
    )


def load_model_route_support(route_root: Path, model: str) -> dict[str, Any]:
    """Reuse the reviewed clean-v2 route loader and build both split matchings."""

    joins, metadata = load_verified_joins(route_root, model)
    provenance = _route_provenance(metadata, model)
    normalized = normalize_join_universe(joins, metadata, model)
    split = _canonical_split_from_normalized(normalized)
    pairings = {
        name: canonical_perfect_matching(split[name]) for name in SPLIT_NAMES
    }
    return {
        "model": model,
        "metadata": metadata,
        "model_identity": provenance["model_identity"],
        "route_provenance": provenance,
        "normalized_join_universe": normalized,
        "selected_joins": split,
        "pairings": pairings,
    }


def _build_selection_certificate(support: Mapping[str, Any]) -> dict[str, Any]:
    normalized = list(support["normalized_join_universe"])
    candidates = [
        {
            "request_id": row["request_id"],
            "receiver_rank": row["receiver_rank"],
            "full_join_key": row["full_join_key"],
            "full_join_identity": row["full_join_identity"],
        }
        for row in normalized
    ]
    candidates.sort(key=lambda row: (str(row["request_id"]), str(row["full_join_key"])))
    split_rows = {}
    for split_name in SPLIT_NAMES:
        chosen = list(support["selected_joins"][split_name])
        pairs = list(support["pairings"][split_name])
        split_rows[split_name] = {
            "request_ids": sorted(str(row["request_id"]) for row in chosen),
            "selected_join_keys": sorted(
                [str(row["request_id"]), str(row["full_join_key"])] for row in chosen
            ),
            "pair_edge_keys": [str(pair["edge_key"]) for pair in pairs],
            "pair_keys": [str(pair["pair_key"]) for pair in pairs],
        }
    payload = {
        "schema_version": SELECTION_CERTIFICATE_SCHEMA,
        "model_identity": dict(support["model_identity"]),
        "route_provenance": dict(support["route_provenance"]),
        "normalized_join_count": len(candidates),
        "normalized_join_universe_sha256": object_sha256(candidates),
        "candidate_universe": candidates,
        "splits": split_rows,
    }
    return {**payload, "certificate_sha256": object_sha256(payload)}


def _validate_pair_manifest(pair: Mapping[str, Any]) -> None:
    payload = dict(pair)
    recorded = payload.pop("manifest_sha256", None)
    if (
        pair.get("schema_version") != SCHEMA_VERSION
        or not _is_sha256(recorded)
        or recorded != object_sha256(payload)
        or not isinstance(pair.get("pair_identity"), Mapping)
    ):
        raise PhaseMapInstanceError("pair instance manifest hash/schema mismatch")


def _selected_joins_from_pairs(pairs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    joins: dict[str, dict[str, Any]] = {}
    for pair in pairs:
        _validate_pair_manifest(pair)
        pair_identity = pair["pair_identity"]
        pair_joins = pair_identity.get("joins")
        if not isinstance(pair_joins, Mapping):
            raise PhaseMapInstanceError("pair identity lacks selected joins")
        for request, join in pair_joins.items():
            if not isinstance(join, Mapping) or str(join.get("request_id")) != str(request):
                raise PhaseMapInstanceError("pair selected join identity mismatch")
            if request in joins:
                raise PhaseMapInstanceError("selected request appears in multiple pairs")
            joins[str(request)] = dict(join)
    return [joins[request] for request in sorted(joins)]


def validate_model_manifest(value: Mapping[str, Any]) -> None:
    """Recompute route split, join selection, and pairing from the certificate."""

    payload = dict(value)
    recorded = payload.pop("manifest_sha256", None)
    if (
        value.get("schema_version") != SCHEMA_VERSION
        or value.get("scientific_result") is not False
        or not _is_sha256(recorded)
        or recorded != object_sha256(payload)
    ):
        raise PhaseMapInstanceError("model manifest hash/schema mismatch")
    model = value.get("model")
    identity = value.get("model_identity")
    provenance = value.get("route_provenance")
    service_provenance = value.get("service_provenance")
    certificate = value.get("selection_certificate")
    splits = value.get("splits")
    if (
        not isinstance(model, str)
        or not isinstance(identity, Mapping)
        or not isinstance(provenance, Mapping)
        or not isinstance(service_provenance, Mapping)
        or not isinstance(certificate, Mapping)
        or not isinstance(splits, Mapping)
        or set(splits) != set(SPLIT_NAMES)
        or provenance.get("model_identity") != identity
    ):
        raise PhaseMapInstanceError("model manifest provenance/certificate surface is incomplete")
    unpack_service_us = service_provenance.get("unpack_service_us")
    if (
        set(service_provenance) != {
            "lut_artifact_sha256", "component", "statistic", "unpack_service_us",
            "lut_model_identity",
        }
        or not _is_sha256(service_provenance.get("lut_artifact_sha256"))
        or service_provenance.get("component") != "receiver_unpack"
        or service_provenance.get("statistic") != "median_cuda_event_us"
        or isinstance(unpack_service_us, bool)
        or not isinstance(unpack_service_us, (int, float))
        or not math.isfinite(float(unpack_service_us))
        or float(unpack_service_us) <= 0
        or not isinstance(service_provenance.get("lut_model_identity"), Mapping)
        or service_provenance["lut_model_identity"].get("model_revision")
        != identity["model_revision"]
        or service_provenance["lut_model_identity"].get("top_k") != identity["top_k"]
        or type(service_provenance["lut_model_identity"].get("hidden")) is not int
        or service_provenance["lut_model_identity"]["hidden"] <= 0
        or (
            model in FROZEN_HIDDEN_BY_MODEL
            and service_provenance["lut_model_identity"]["hidden"]
            != FROZEN_HIDDEN_BY_MODEL[model]
        )
    ):
        raise PhaseMapInstanceError("model manifest receiver-unpack service provenance is invalid")
    _model_identity(identity, model)
    for field in ROUTE_PROVENANCE_FIELDS:
        if not _is_sha256(provenance.get(field)):
            raise PhaseMapInstanceError("model manifest reviewed route provenance is incomplete")
    certificate_payload = dict(certificate)
    certificate_hash = certificate_payload.pop("certificate_sha256", None)
    candidates = certificate.get("candidate_universe")
    if (
        certificate.get("schema_version") != SELECTION_CERTIFICATE_SCHEMA
        or certificate.get("model_identity") != identity
        or certificate.get("route_provenance") != provenance
        or not _is_sha256(certificate_hash)
        or certificate_hash != object_sha256(certificate_payload)
        or not isinstance(candidates, list)
    ):
        raise PhaseMapInstanceError("route selection certificate closure mismatch")
    normalized_candidates = []
    for row in candidates:
        if not isinstance(row, Mapping) or set(row) != {
            "request_id", "receiver_rank", "full_join_key", "full_join_identity"
        }:
            raise PhaseMapInstanceError("candidate selection certificate schema drift")
        join_identity = row["full_join_identity"]
        if (
            not isinstance(join_identity, Mapping)
            or row["full_join_key"] != object_sha256(join_identity)
            or row["request_id"] != join_identity.get("request_id")
            or row["receiver_rank"] != join_identity.get("receiver_rank")
            or join_identity.get("model_key") != model
            or join_identity.get("model_revision") != identity["model_revision"]
            or join_identity.get("data_manifest_sha256")
            != identity["data_manifest_sha256"]
            or join_identity.get("placement_manifest_sha256")
            != identity["placement_manifest_sha256"]
            or len(join_identity.get("topk_siblings", [])) != identity["top_k"]
        ):
            raise PhaseMapInstanceError("candidate full join identity mismatch")
        normalized_candidates.append(dict(row))
    normalized_candidates.sort(
        key=lambda row: (str(row["request_id"]), str(row["full_join_key"]))
    )
    if (
        len(normalized_candidates) != certificate.get("normalized_join_count")
        or object_sha256(normalized_candidates)
        != certificate.get("normalized_join_universe_sha256")
    ):
        raise PhaseMapInstanceError("normalized candidate universe hash/census mismatch")
    by_request: dict[str, list[dict[str, Any]]] = defaultdict(list)
    receiver_by_request: dict[str, int] = {}
    for row in normalized_candidates:
        request = str(row["request_id"])
        receiver = int(row["receiver_rank"])
        if receiver_by_request.setdefault(request, receiver) != receiver:
            raise PhaseMapInstanceError("certificate request receiver identity drift")
        by_request[request].append(row)
    expected_candidates = int(identity["expected_join_candidates_per_request"])
    if len(by_request) != 64 or any(
        len(rows) != expected_candidates for rows in by_request.values()
    ):
        raise PhaseMapInstanceError("certificate candidate universe is incomplete")
    by_receiver: dict[int, list[str]] = defaultdict(list)
    for request, receiver in receiver_by_request.items():
        by_receiver[receiver].append(request)
    if set(by_receiver) != set(range(8)) or any(len(rows) != 8 for rows in by_receiver.values()):
        raise PhaseMapInstanceError("certificate receiver/request census mismatch")
    expected_request_splits = {name: set() for name in SPLIT_NAMES}
    for receiver in range(8):
        ordered = sorted(
            by_receiver[receiver],
            key=lambda request: object_sha256(["phasemap-v1-split", request]),
        )
        expected_request_splits["selection"].update(ordered[:4])
        expected_request_splits["holdout"].update(ordered[4:])

    certificate_splits = certificate.get("splits")
    if not isinstance(certificate_splits, Mapping) or set(certificate_splits) != set(
        SPLIT_NAMES
    ):
        raise PhaseMapInstanceError("certificate split surface mismatch")
    for split_name in SPLIT_NAMES:
        split = splits[split_name]
        certificate_split = certificate_splits[split_name]
        if (
            not isinstance(split, Mapping)
            or not isinstance(certificate_split, Mapping)
            or not isinstance(split.get("pairs"), list)
            or len(split["pairs"]) != 16
        ):
            raise PhaseMapInstanceError("model split manifest is malformed")
        embedded_joins = _selected_joins_from_pairs(split["pairs"])
        for pair in split["pairs"]:
            depths = pair.get("depths")
            control_mode = pair.get("control_mode")
            if (
                not isinstance(depths, Mapping)
                or set(depths) != {"low", "high"}
                or type(depths.get("low")) is not int
                or type(depths.get("high")) is not int
                or (depths.get("low"), depths.get("high")) != DEFAULT_DEPTHS
                or control_mode != "primary"
            ):
                raise PhaseMapInstanceError("pair world reconstruction parameters are malformed")
            rebuilt = rebuild_from_pair_identity(
                pair["pair_identity"],
                float(unpack_service_us),
                (int(depths["low"]), int(depths["high"])),
                mode=str(control_mode),
            )
            if canonical_json_bytes(pair) != canonical_json_bytes(rebuilt):
                raise PhaseMapInstanceError(
                    "pair worlds do not match canonical reconstruction from pair identity"
                )
        embedded_by_request = {
            str(row["request_id"]): str(row["full_join_key"]) for row in embedded_joins
        }
        expected_requests = expected_request_splits[split_name]
        expected_chosen = {
            request: str(
                min(
                    by_request[request],
                    key=lambda row: object_sha256(
                        ["phasemap-v1-join", row["full_join_identity"]]
                    ),
                )["full_join_key"]
            )
            for request in expected_requests
        }
        candidate_by_identity = {
            (str(row["request_id"]), str(row["full_join_key"])): row
            for row in normalized_candidates
        }
        for embedded in embedded_joins:
            request = str(embedded["request_id"])
            key = str(embedded["full_join_key"])
            candidate = candidate_by_identity.get((request, key))
            if candidate is None:
                raise PhaseMapInstanceError(
                    "embedded selected join is absent from the certified candidate universe"
                )
            join_identity = candidate["full_join_identity"]
            expected_siblings = []
            for sibling in join_identity["topk_siblings"]:
                sibling_identity = {
                    "model_key": join_identity["model_key"],
                    "model_revision": join_identity["model_revision"],
                    "data_manifest_sha256": join_identity["data_manifest_sha256"],
                    "placement_manifest_sha256": join_identity[
                        "placement_manifest_sha256"
                    ],
                    "request_id": join_identity["request_id"],
                    "forward_id": join_identity["forward_id"],
                    "layer_id": join_identity["layer_id"],
                    "token_position": join_identity["token_position"],
                    "epoch": join_identity["epoch"],
                    "topk_slot": sibling["topk_slot"],
                    "expert_id": sibling["expert_id"],
                    "sender_rank": sibling["sender_rank"],
                    "receiver_rank": sibling["receiver_rank"],
                }
                expected_siblings.append(
                    {
                        "identity": sibling_identity,
                        "full_sibling_key": object_sha256(sibling_identity),
                    }
                )
            expected_embedded = {
                "request_id": candidate["request_id"],
                "receiver_rank": candidate["receiver_rank"],
                "layer_id": join_identity["layer_id"],
                "token_position": join_identity["token_position"],
                "full_join_identity": join_identity,
                "full_join_key": candidate["full_join_key"],
                "siblings": expected_siblings,
            }
            if canonical_json_bytes(embedded) != canonical_json_bytes(expected_embedded):
                raise PhaseMapInstanceError(
                    "embedded selected join differs from its certified candidate identity"
                )
        if (
            set(embedded_by_request) != expected_requests
            or embedded_by_request != expected_chosen
            or certificate_split.get("request_ids") != sorted(expected_requests)
            or certificate_split.get("selected_join_keys")
            != sorted([request, key] for request, key in expected_chosen.items())
            or split.get("selected_request_count") != 32
            or split.get("pair_count") != 16
        ):
            raise PhaseMapInstanceError("certificate split/join selection cannot be reproduced")
        recomputed_pairs = canonical_perfect_matching(embedded_joins)
        if (
            certificate_split.get("pair_edge_keys")
            != [str(pair["edge_key"]) for pair in recomputed_pairs]
            or certificate_split.get("pair_keys")
            != [str(pair["pair_key"]) for pair in recomputed_pairs]
            or [str(pair["pair_key"]) for pair in split["pairs"]]
            != [str(pair["pair_key"]) for pair in recomputed_pairs]
        ):
            raise PhaseMapInstanceError("canonical pair selection cannot be reproduced")


def build_model_manifests(
    route_root: Path,
    model: str,
    unpack_service_us: float,
    depths: tuple[int, int] = DEFAULT_DEPTHS,
    *,
    lut_artifact_sha256: str,
    lut_model_identity: Mapping[str, Any],
) -> dict[str, Any]:
    if depths != DEFAULT_DEPTHS:
        raise PhaseMapInstanceError("formal PhaseMap depths are frozen to (8,16)")
    if not _is_sha256(lut_artifact_sha256):
        raise PhaseMapInstanceError("LUT artifact SHA-256 is malformed")
    support = load_model_route_support(route_root, model)
    expected_lut_identity = {
        "model_revision": support["model_identity"]["model_revision"],
        "top_k": support["model_identity"]["top_k"],
    }
    if (
        not isinstance(lut_model_identity, Mapping)
        or lut_model_identity.get("model_revision") != expected_lut_identity["model_revision"]
        or lut_model_identity.get("top_k") != expected_lut_identity["top_k"]
        or type(lut_model_identity.get("hidden")) is not int
        or int(lut_model_identity["hidden"]) <= 0
        or (
            model in FROZEN_HIDDEN_BY_MODEL
            and int(lut_model_identity["hidden"]) != FROZEN_HIDDEN_BY_MODEL[model]
        )
    ):
        raise PhaseMapInstanceError("LUT model revision/top-k/hidden differs from route identity")
    frozen_lut_identity = {
        "model_revision": lut_model_identity["model_revision"],
        "top_k": lut_model_identity["top_k"],
        "hidden": lut_model_identity["hidden"],
    }
    certificate = _build_selection_certificate(support)
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "model": model,
        "scientific_result": False,
        "model_identity": support["model_identity"],
        "route_provenance": support["route_provenance"],
        "service_provenance": {
            "lut_artifact_sha256": lut_artifact_sha256,
            "component": "receiver_unpack",
            "statistic": "median_cuda_event_us",
            "unpack_service_us": float(unpack_service_us),
            "lut_model_identity": frozen_lut_identity,
        },
        "selection_certificate": certificate,
        "splits": {},
    }
    for split_name in SPLIT_NAMES:
        pairs = support["pairings"][split_name]
        result["splits"][split_name] = {
            "selected_request_count": len(support["selected_joins"][split_name]),
            "pair_count": len(pairs),
            "pairs": [
                build_world_manifest(pair, unpack_service_us, depths) for pair in pairs
            ],
        }
    payload = dict(result)
    manifest = {**result, "manifest_sha256": object_sha256(payload)}
    validate_model_manifest(manifest)
    return manifest


def make_split_bundle(model_manifest: Mapping[str, Any], split: str) -> dict[str, Any]:
    validate_model_manifest(model_manifest)
    if split not in SPLIT_NAMES:
        raise PhaseMapInstanceError("unknown split bundle")
    payload = {
        "schema_version": SPLIT_BUNDLE_SCHEMA,
        "model": model_manifest["model"],
        "split": split,
        "scientific_result": False,
        "source_model_manifest_sha256": model_manifest["manifest_sha256"],
        "model_identity": model_manifest["model_identity"],
        "route_provenance": model_manifest["route_provenance"],
        "service_provenance": model_manifest["service_provenance"],
        "selection_certificate_sha256": model_manifest["selection_certificate"][
            "certificate_sha256"
        ],
        "pairs": model_manifest["splits"][split]["pairs"],
    }
    return {**payload, "artifact_sha256": object_sha256(payload)}


def _write_json_atomic_no_overwrite(path: Path, value: Mapping[str, Any]) -> None:
    path = path.absolute()
    if path.exists() or path.is_symlink():
        raise PhaseMapInstanceError(f"refusing to overwrite {path}")
    encoded = json.dumps(
        value, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True
    ).encode("utf-8") + b"\n"
    temporary = path.with_name(f".{path.name}.partial-{os.getpid()}")
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o444,
    )
    try:
        try:
            offset = 0
            while offset < len(encoded):
                written = os.write(descriptor, encoded[offset:])
                if written <= 0:
                    raise PhaseMapInstanceError("atomic instance write made no progress")
                offset += written
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise PhaseMapInstanceError(f"refusing to overwrite {path}") from exc
    finally:
        temporary.unlink(missing_ok=True)
    directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    expected = hashlib.sha256(encoded).hexdigest()
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise PhaseMapInstanceError("published instance file failed content verification")


def produce_formal_instance_artifacts(
    route_root: Path,
    lut_artifact: Mapping[str, Any],
    output_dir: Path,
    depths: tuple[int, int] = DEFAULT_DEPTHS,
) -> dict[str, str]:
    """Produce both models' closed model manifests and split bundles."""

    if depths != DEFAULT_DEPTHS:
        raise PhaseMapInstanceError("formal PhaseMap depths are frozen to (8,16)")

    try:
        from . import capture_phasemap_lut_gpu as lut_module
    except ImportError:  # pragma: no cover
        import capture_phasemap_lut_gpu as lut_module  # type: ignore

    lut_module.validate_artifact(lut_artifact)
    lut_sha = lut_artifact.get("artifact_sha256")
    if not _is_sha256(lut_sha):
        raise PhaseMapInstanceError("validated LUT lacks artifact SHA-256")
    unpack_by_model = {
        str(row["model_key"]): float(row["median_cuda_event_us"])
        for row in lut_artifact["summary"]
        if row.get("component") == "receiver_unpack"
    }
    if set(unpack_by_model) != set(MODELS) or any(
        not math.isfinite(value) or value <= 0 for value in unpack_by_model.values()
    ):
        raise PhaseMapInstanceError("LUT receiver-unpack service surface is incomplete")
    lut_inputs = lut_artifact.get("model_inputs")
    if not isinstance(lut_inputs, Mapping) or set(lut_inputs) != set(MODELS):
        raise PhaseMapInstanceError("LUT model input identity surface is incomplete")
    if output_dir.exists() or output_dir.is_symlink():
        raise PhaseMapInstanceError("refusing to overwrite formal instance output directory")
    output_dir.mkdir(parents=True, exist_ok=False)
    incomplete = output_dir / ".INCOMPLETE"
    _write_json_atomic_no_overwrite(
        incomplete,
        {
            "schema_version": "phasemap-v1-incomplete-publication-marker",
            "scientific_result": False,
        },
    )
    outputs: dict[str, str] = {}
    manifests = {}
    for model in MODELS:
        manifest = build_model_manifests(
            route_root,
            model,
            unpack_by_model[model],
            depths,
            lut_artifact_sha256=str(lut_sha),
            lut_model_identity=lut_inputs[model],
        )
        manifests[model] = manifest
        filename = f"{model}_model_manifest.json"
        _write_json_atomic_no_overwrite(output_dir / filename, manifest)
        outputs[f"{model}_model_manifest"] = filename
        for split in SPLIT_NAMES:
            bundle = make_split_bundle(manifest, split)
            filename = f"{model}_{split}_instances.json"
            _write_json_atomic_no_overwrite(output_dir / filename, bundle)
            outputs[f"{model}_{split}"] = filename
    published_files = {
        key: {
            "path": filename,
            "sha256": hashlib.sha256((output_dir / filename).read_bytes()).hexdigest(),
        }
        for key, filename in outputs.items()
    }
    source_payload = {
        "schema_version": "phasemap-v1-formal-instance-source-manifest",
        "scientific_result": False,
        "producer_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "lut_artifact_sha256": lut_sha,
        "route_provenance": {
            model: manifests[model]["route_provenance"] for model in MODELS
        },
        "published_files": published_files,
    }
    source_manifest = {
        **source_payload,
        "manifest_sha256": object_sha256(source_payload),
    }
    _write_json_atomic_no_overwrite(output_dir / "source_manifest.json", source_manifest)
    outputs["source_manifest"] = "source_manifest.json"
    incomplete.unlink()
    os.chmod(output_dir, 0o555)
    directory_fd = os.open(output_dir, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    parent_fd = os.open(output_dir.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)
    return outputs


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PhaseMapInstanceError(f"cannot load JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise PhaseMapInstanceError("JSON artifact root is not an object")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route-root", required=True, type=Path)
    parser.add_argument("--lut", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    outputs = produce_formal_instance_artifacts(
        args.route_root,
        _load_json_object(args.lut),
        args.output_dir,
    )
    print(json.dumps({"output_dir": str(args.output_dir), "outputs": outputs}, sort_keys=True))


if __name__ == "__main__":
    main()
