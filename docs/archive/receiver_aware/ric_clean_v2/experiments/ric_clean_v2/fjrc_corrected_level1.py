#!/usr/bin/env python3
"""Native-route pair producer for corrected FJRC Level 1.

The module selects request-disjoint matched pairs without reading outcomes and
materializes Level-0 scenarios from validated clean-v2 joins plus a validated
RTX-5090 primitive LUT.  It does not run a formal campaign by itself.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from . import capture_fjrc_lut_gpu as lut_core
    from .explore_receiver_matched_milp import load_verified_joins
    from .fjrc_corrected_level0 import Join, PriorCompletion, Scenario, Task, World, validate_scenario
except ImportError:  # pragma: no cover
    import capture_fjrc_lut_gpu as lut_core  # type: ignore
    from explore_receiver_matched_milp import load_verified_joins  # type: ignore
    from fjrc_corrected_level0 import Join, PriorCompletion, Scenario, Task, World, validate_scenario  # type: ignore


class Level1Error(RuntimeError):
    pass


def _canonical(value: Any) -> bytes:
    return json.dumps(value, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


@dataclass(frozen=True)
class ServiceLUT:
    model: str
    pack_us: float
    cut_us: float
    unpack_us: float
    combine_us: float
    total_contribution_us: float
    source_artifact_sha256: str


def _summary_point(
    rows: Sequence[Mapping[str, Any]], model: str, component: str, depth: int | None
) -> Mapping[str, Any]:
    matches = [
        row
        for row in rows
        if row.get("model_key") == model
        and row.get("component") == component
        and row.get("queue_depth") == depth
    ]
    if len(matches) != 1:
        raise Level1Error(f"LUT point missing or duplicated: {model}/{component}/{depth}")
    return matches[0]


def extract_service_lut(value: Mapping[str, Any], model: str) -> ServiceLUT:
    if model not in lut_core.MODEL_SHAPES:
        raise Level1Error("unknown model")
    try:
        lut_core.validate_self_hash(value)
        recomputed = lut_core.validate_and_summarize(value.get("raw_trials", []))
    except (lut_core.FJRCLUTError, TypeError) as exc:
        raise Level1Error("invalid FJRC LUT artifact") from exc
    if _canonical(recomputed) != _canonical(value.get("summary")):
        raise Level1Error("LUT summary replay mismatch")
    if (
        value.get("schema_version") != "fjrc-primitive-lut-v1"
        or value.get("status") != "EXPLORATORY_CALIBRATION_INPUT_ONLY"
        or value.get("scientific_result") is not False
        or value.get("environment", {}).get("gpu_name") != "NVIDIA GeForce RTX 5090"
        or value.get("shared_cut", {}).get("source") != "ANALYTIC_NETWORK_L2_PROXY_NOT_RDMA"
    ):
        raise Level1Error("LUT evidence envelope mismatch")
    rows = value["summary"]
    pack = float(_summary_point(rows, model, "sender_pack", None)["median_cuda_event_us"])
    unpack = float(_summary_point(rows, model, "receiver_unpack", 1)["median_cuda_event_us"])
    combine = float(_summary_point(rows, model, "canonical_combine", None)["median_cuda_event_us"])
    hidden = int(lut_core.MODEL_SHAPES[model]["hidden"])
    transport_bytes = ((hidden * 2 + 16 + 15) // 16) * 16
    cut = transport_bytes * 8 / (200 * 1000)
    values = (pack, cut, unpack, combine)
    if any(not math.isfinite(item) or item <= 0 for item in values):
        raise Level1Error("nonpositive service point")
    return ServiceLUT(model, pack, cut, unpack, combine, pack + cut + unpack, str(value["artifact_sha256"]))


def load_service_lut(path: Path, model: str) -> ServiceLUT:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Level1Error("cannot read FJRC LUT") from exc
    if not isinstance(value, Mapping):
        raise Level1Error("FJRC LUT root is not an object")
    return extract_service_lut(value, model)


def request_split(joins: Sequence[Mapping[str, Any]]) -> tuple[set[str], set[str]]:
    by_receiver: dict[int, set[str]] = defaultdict(set)
    for join in joins:
        by_receiver[int(join["receiver_rank"])].add(str(join["request_id"]))
    if set(by_receiver) != set(range(8)):
        raise Level1Error("expected receiver ranks 0..7")
    selection: set[str] = set()
    holdout: set[str] = set()
    for receiver in range(8):
        requests = sorted(by_receiver[receiver], key=lambda value: _sha({"fjrc-corrected-split": value}))
        if len(requests) != 8:
            raise Level1Error("expected exactly eight requests per receiver")
        selection.update(requests[:4])
        holdout.update(requests[4:])
    if selection & holdout or len(selection) != 32 or len(holdout) != 32:
        raise Level1Error("request split overlap or denominator drift")
    return selection, holdout


def _slot_by_sender(join: Mapping[str, Any]) -> dict[int, list[Mapping[str, Any]]]:
    result: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for sibling in join["siblings"]:
        result[int(sibling["sender_rank"])].append(sibling)
    return result


def _choose_roles(
    join_a: Mapping[str, Any], join_b: Mapping[str, Any]
) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]] | None:
    by_sender_a, by_sender_b = _slot_by_sender(join_a), _slot_by_sender(join_b)
    common = sorted(set(by_sender_a) & set(by_sender_b))
    candidates = []
    for prior_sender in common:
        pa = min(by_sender_a[prior_sender], key=lambda row: int(row["topk_slot"]))
        pb = min(by_sender_b[prior_sender], key=lambda row: int(row["topk_slot"]))
        remaining_a = [row for row in join_a["siblings"] if row is not pa]
        remaining_b = [row for row in join_b["siblings"] if row is not pb]
        for ca in remaining_a:
            for cb in remaining_b:
                if int(ca["sender_rank"]) == int(cb["sender_rank"]):
                    continue
                key = _sha(
                    {
                        "prior_sender": prior_sender,
                        "pa": int(pa["topk_slot"]),
                        "pb": int(pb["topk_slot"]),
                        "ca": int(ca["topk_slot"]),
                        "cb": int(cb["topk_slot"]),
                    }
                )
                candidates.append((key, pa, pb, ca, cb))
    if not candidates:
        return None
    _key, pa, pb, ca, cb = min(candidates, key=lambda row: row[0])
    return pa, pb, ca, cb


def _task_id(join: Mapping[str, Any], sibling: Mapping[str, Any]) -> str:
    return f"{join['join_id']}:slot:{int(sibling['topk_slot']):02d}"


def materialize_pair(
    model: str,
    join_a: Mapping[str, Any],
    join_b: Mapping[str, Any],
    service: ServiceLUT,
    *,
    future_release_factor: float = 4.0,
    deadline_factor: float = 2.0,
) -> Scenario:
    if model != service.model:
        raise Level1Error("model/service mismatch")
    if join_a["request_id"] == join_b["request_id"] or int(join_a["receiver_rank"]) != int(join_b["receiver_rank"]):
        raise Level1Error("pair must contain distinct requests at one receiver")
    roles = _choose_roles(join_a, join_b)
    if roles is None:
        raise Level1Error("pair lacks swappable prior and distinct-sender candidates")
    pa, pb, ca, cb = roles
    selected = {_task_id(join_a, ca), _task_id(join_b, cb)}
    future_ready = future_release_factor * service.total_contribution_us
    tasks = []
    for join in (join_a, join_b):
        for sibling in join["siblings"]:
            task_id = _task_id(join, sibling)
            tasks.append(
                Task(
                    task_id,
                    str(join["join_id"]),
                    str(join["request_id"]),
                    int(sibling["sender_rank"]),
                    int(join["receiver_rank"]),
                    0.0 if task_id in selected else future_ready,
                    service.total_contribution_us,
                )
            )
    deadline = deadline_factor * service.total_contribution_us * max(
        len(join_a["siblings"]), len(join_b["siblings"])
    )
    joins = tuple(
        Join(
            str(join["join_id"]),
            str(join["request_id"]),
            int(join["receiver_rank"]),
            deadline,
            service.combine_us,
            tuple(_task_id(join, sibling) for sibling in join["siblings"]),
        )
        for join in (join_a, join_b)
    )
    pa_id, pb_id = _task_id(join_a, pa), _task_id(join_b, pb)
    prior_start = -service.total_contribution_us
    worlds = (
        World("world0", (PriorCompletion(pa_id, prior_start, 0.0),)),
        World("world1", (PriorCompletion(pb_id, prior_start, 0.0),)),
    )
    scenario = Scenario(
        _sha({"model": model, "a": join_a["join_id"], "b": join_b["join_id"]}),
        0.0,
        ((int(join_a["receiver_rank"]), 0.0),),
        tuple(tasks),
        joins,
        tuple(sorted(selected)),
        worlds,
    )
    validate_scenario(scenario)
    return scenario


def select_split_scenarios(
    model: str,
    joins: Sequence[Mapping[str, Any]],
    service: ServiceLUT,
    *,
    split: str,
) -> list[Scenario]:
    selection, holdout = request_split(joins)
    if split not in {"selection", "holdout"}:
        raise Level1Error("split must be selection or holdout")
    allowed = selection if split == "selection" else holdout
    by_request: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    receiver_by_request: dict[str, int] = {}
    for join in joins:
        request = str(join["request_id"])
        if request in allowed:
            by_request[request].append(join)
            receiver_by_request[request] = int(join["receiver_rank"])
    scenarios = []
    used: set[str] = set()
    for receiver in range(8):
        requests = sorted(
            (request for request in allowed if receiver_by_request.get(request) == receiver),
            key=lambda value: _sha({"fjrc-corrected-pair": split, "request": value}),
        )
        if len(requests) != 4:
            raise Level1Error("holdout receiver denominator drift")
        for request_a, request_b in zip(requests[::2], requests[1::2]):
            pair_candidates = []
            for join_a in by_request[request_a]:
                for join_b in by_request[request_b]:
                    if int(join_a["token_position"]) != int(join_b["token_position"]):
                        continue
                    roles = _choose_roles(join_a, join_b)
                    if roles is not None:
                        pair_candidates.append((_sha({"a": join_a["join_id"], "b": join_b["join_id"]}), join_a, join_b))
            if not pair_candidates:
                raise Level1Error("BLOCKED_INSUFFICIENT_MATCHED_SUPPORT")
            _key, join_a, join_b = min(pair_candidates, key=lambda row: row[0])
            scenarios.append(materialize_pair(model, join_a, join_b, service))
            used.update((request_a, request_b))
    if len(scenarios) != 16 or len(used) != 32:
        raise Level1Error("expected 16 disjoint holdout scenarios")
    return scenarios


def select_holdout_scenarios(
    model: str, joins: Sequence[Mapping[str, Any]], service: ServiceLUT
) -> list[Scenario]:
    return select_split_scenarios(model, joins, service, split="holdout")


def load_native_scenarios(route_root: Path, model: str, lut_path: Path) -> list[Scenario]:
    joins, _metadata = load_verified_joins(route_root, model)
    service = load_service_lut(lut_path, model)
    return select_holdout_scenarios(model, joins, service)
