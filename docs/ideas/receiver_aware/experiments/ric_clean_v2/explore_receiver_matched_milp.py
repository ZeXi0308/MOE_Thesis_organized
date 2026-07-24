#!/usr/bin/env python3
"""Matched-world receiver-information headroom on clean-v2 native routes.

Evidence boundary: exploratory L2 single-shared-cut proxy with normalized
service.  B uses one schedule across sender-indistinguishable worlds; R0 sees
current keyed receiver residual work.  MILP optima are checked by exhaustive
8! enumeration.  This is not RDMA, serving, or a scientific result.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import random
from statistics import NormalDist, median
from typing import Any, Mapping, Sequence

try:
    from . import native_route_core as native
    from . import prepare_clean_v2_data as data
    from . import capture_clean_v2_routes_gpu as route_capture
except ImportError:  # pragma: no cover
    import native_route_core as native  # type: ignore
    import prepare_clean_v2_data as data  # type: ignore
    import capture_clean_v2_routes_gpu as route_capture  # type: ignore


MODELS = ("olmoe", "llmjp")
LOADS = (0.60, 0.85, 0.95)
SERVICE_CVS = (0.0, 0.25, 0.50)
PRIMARY_RESIDUAL = 2.0
BASELINES = ("fcfs", "spt", "receiver_qdepth", "sender_drr", "topology_finish", "sender_join_remaining")
TASKS_PER_WINDOW = 8


class MatchedError(RuntimeError):
    pass


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(value: object) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class Candidate:
    task_id: str
    join_id: str
    request_id: str
    release: float
    service: float
    sender_rank: int
    receiver_rank: int
    expert_id: int
    token_position: int


@dataclass(frozen=True)
class MatchedWindow:
    model: str
    split: str
    receiver_rank: int
    window_id: int
    tasks: tuple[Candidate, ...]
    residual_by_world: tuple[Mapping[str, float], Mapping[str, float]]
    target_rho: float
    requested_service_cv: float
    realized_service_cv: float
    residual_scale: float
    fingerprint: str


def _load_self(path: Path, field: str = "manifest_sha256") -> dict[str, Any]:
    value = data.load_mapping(path, label=path.name)
    data.validate_self_hash(value, field)
    return value


def load_verified_joins(route_root: Path, model: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if route_root.resolve(strict=True) != route_capture.ROUTE_OUTPUTS[model].parent.resolve(strict=True):
        raise MatchedError("route root is not the reviewed clean-v2 bundle")
    model_dir = route_root / model
    metadata_path = model_dir / "capture_metadata.json"
    placement_path = model_dir / "placement.json"
    parity_path = model_dir / "route_parity.json"
    signoff_path = model_dir / "producer_signoff.json"
    trace_path = model_dir / "route_trace.jsonl"
    metadata = _load_self(metadata_path)
    placement = _load_self(placement_path)
    parity = _load_self(parity_path)
    signoff = _load_self(signoff_path, "signoff_sha256")
    if (
        metadata.get("status") != "CALIBRATION_INPUT_ONLY"
        or metadata.get("scientific_result") is not False
        or metadata.get("model_key") != model
        or parity.get("status") != "CALIBRATION_INPUT_ONLY"
        or parity.get("scientific_result") is not False
        or signoff.get("status") != "SIGNED-OFF"
        or signoff.get("open_p0") != 0
        or metadata.get("placement_manifest_sha256") != placement.get("manifest_sha256")
        or metadata.get("route_parity_sha256") != parity.get("manifest_sha256")
        or metadata.get("placement_file_sha256") != file_sha(placement_path)
        or metadata.get("route_parity_file_sha256") != file_sha(parity_path)
        or metadata.get("producer_signoff_file_sha256") != file_sha(signoff_path)
    ):
        raise MatchedError("route artifact closure mismatch")
    selected_layers = [int(value) for value in metadata["selected_replay_layers"]]
    all_layers = [int(value) for value in metadata["all_moe_layers_captured"]]
    if selected_layers != [int(value) for value in parity.get("selected_replay_layers", [])] or len(selected_layers) != 4:
        raise MatchedError("selected layer closure mismatch")
    clean_root = route_root.parents[1]
    manifest = data.validate_calibration_manifest(clean_root / "data/calibration/manifest.json")
    if manifest.get("manifest_sha256") != metadata.get("data_manifest_sha256"):
        raise MatchedError("route/data manifest mismatch")
    reviewed_signoff = route_capture.validate_route_signoff(model, manifest)
    if reviewed_signoff.get("signoff_sha256") != signoff.get("signoff_sha256"):
        raise MatchedError("embedded route signoff differs from reviewed signoff closure")
    request_ids = {str(row["request_id"]) for row in manifest["requests"]}
    receivers = placement["request_to_receiver"]
    expert_to_sender = placement["expert_to_sender"]
    top_k = int(metadata["top_k"])
    num_experts = int(metadata["num_experts"])
    masks: dict[tuple[str, int], int] = defaultdict(int)
    tuple_hashes: dict[tuple[str, int], str] = {}
    selected: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    digest = hashlib.sha256()
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(trace_path, flags)
    before = os.fstat(descriptor)
    count = 0
    try:
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            for raw in handle:
                digest.update(raw)
                count += 1
                try:
                    row = json.loads(raw)
                except (UnicodeError, json.JSONDecodeError) as exc:
                    raise MatchedError("invalid route JSONL") from exc
                request = row.get("request_id")
                layer = row.get("layer_id")
                position = row.get("token_position")
                slot = row.get("topk_slot")
                expert = row.get("expert_id")
                if (
                    row.get("schema_version") != "ric-clean-v2-route-row-v1"
                    or row.get("model_key") != model
                    or request not in request_ids
                    or type(layer) is not int or layer not in all_layers
                    or type(position) is not int or not 0 <= position < 128
                    or type(slot) is not int or not 0 <= slot < top_k
                    or type(expert) is not int or not 0 <= expert < num_experts
                    or row.get("token_id") != f"{request}:token:{position:03d}"
                    or row.get("token_block_id") != row.get("token_id")
                    or row.get("sender_rank") != native.expert_sender(expert, num_experts, int(placement["ep_size"]))
                    or expert_to_sender.get(str(expert)) != row.get("sender_rank")
                    or receivers.get(request) != row.get("receiver_rank")
                    or row.get("epoch") != 1 or row.get("valid") is not True
                    or not math.isfinite(float(row.get("route_weight")))
                    or row.get("route_weight_dtype") != "torch.bfloat16"
                ):
                    raise MatchedError("route identity/accounting mismatch")
                bit = 1 << (position * top_k + slot)
                key = (str(request), layer)
                if masks[key] & bit:
                    raise MatchedError("duplicate route Cartesian key")
                masks[key] |= bit
                tuple_value = str(row.get("native_route_tuple_sha256"))
                if len(tuple_value) != 64 or tuple_hashes.setdefault(key, tuple_value) != tuple_value:
                    raise MatchedError("route tuple hash drift")
                expected_selected = layer == native.assigned_layer(str(request), selected_layers)
                if row.get("selected_for_replay") is not expected_selected:
                    raise MatchedError("selected_for_replay mismatch")
                if expected_selected:
                    selected[(str(request), layer, position)].append(dict(row))
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise MatchedError("route trace changed during streaming validation")
    expected_rows = len(request_ids) * len(all_layers) * 128 * top_k
    full_mask = (1 << (128 * top_k)) - 1
    if (
        count != expected_rows
        or digest.hexdigest() != metadata.get("route_trace_file_sha256")
        or set(masks) != {(request, layer) for request in request_ids for layer in all_layers}
        or any(mask != full_mask for mask in masks.values())
    ):
        raise MatchedError("route trace hash/census mismatch")
    joins = []
    for (request, layer, position), siblings in selected.items():
        siblings.sort(key=lambda row: int(row["topk_slot"]))
        if len(siblings) != top_k or [row["topk_slot"] for row in siblings] != list(range(top_k)) or len({row["expert_id"] for row in siblings}) != top_k:
            raise MatchedError("selected join sibling mismatch")
        joins.append({
            "join_id": sha({"model": model, "request": request, "layer": layer, "position": position}),
            "request_id": request,
            "layer_id": layer,
            "token_position": position,
            "receiver_rank": int(siblings[0]["receiver_rank"]),
            "siblings": siblings,
        })
    if len(joins) != len(request_ids) * 128:
        raise MatchedError("selected join census mismatch")
    return joins, metadata


def outcome_blind_windows(joins: Sequence[Mapping[str, Any]]) -> dict[str, list[list[Mapping[str, Any]]]]:
    request_receiver = {str(row["request_id"]): int(row["receiver_rank"]) for row in joins}
    split = {"selection_half": set(), "holdout_half": set()}
    for receiver in range(8):
        requests = sorted((request for request, rank in request_receiver.items() if rank == receiver), key=sha)
        if len(requests) != 8:
            raise MatchedError("expected eight calibration requests per receiver")
        split["selection_half"].update(requests[:4])
        split["holdout_half"].update(requests[4:])
    result: dict[str, list[list[Mapping[str, Any]]]] = {}
    for split_name, allowed in split.items():
        windows = []
        for receiver in range(8):
            pool = sorted(
                (row for row in joins if row["request_id"] in allowed and int(row["receiver_rank"]) == receiver),
                key=lambda row: sha({"split": split_name, "join": row["join_id"]}),
            )
            if len(pool) < TASKS_PER_WINDOW:
                raise MatchedError("insufficient receiver-local joins")
            windows.append(pool[:TASKS_PER_WINDOW])
        used_requests = [{row["request_id"] for row in window} for window in windows]
        if any(used_requests[left] & used_requests[right] for left in range(8) for right in range(left + 1, 8)):
            raise MatchedError("request cluster reused across windows")
        result[split_name] = windows
    return result


def _factor(key: object, cv: float) -> float:
    if cv == 0:
        return 1.0
    uniform = (int(sha(key)[:16], 16) + 0.5) / float(16**16)
    uniform = min(1 - 1e-12, max(1e-12, uniform))
    sigma = math.sqrt(math.log1p(cv * cv))
    return math.exp(sigma * NormalDist().inv_cdf(uniform) - sigma * sigma / 2)


def materialize(model: str, split: str, window_id: int, joins: Sequence[Mapping[str, Any]], rho: float, cv: float, residual: float) -> MatchedWindow:
    rng = random.Random(int(sha({"model": model, "split": split, "window": window_id, "arrival_base": True})[:16], 16))
    now = 0.0
    raw = []
    for index, join in enumerate(joins):
        if index:
            now += rng.expovariate(rho)
        slot = int(sha({"candidate": join["join_id"]})[:8], 16) % len(join["siblings"])
        sibling = join["siblings"][slot]
        service = _factor({"model": model, "join": join["join_id"], "expert": sibling["expert_id"]}, cv)
        raw.append((join, sibling, now, service))
    mean_service = sum(item[3] for item in raw) / len(raw)
    tasks = []
    for join, sibling, release, service in raw:
        tasks.append(Candidate(
            task_id=sha({"join": join["join_id"], "slot": sibling["topk_slot"]}),
            join_id=str(join["join_id"]), request_id=str(join["request_id"]), release=release,
            service=service / mean_service, sender_rank=int(sibling["sender_rank"]),
            receiver_rank=int(join["receiver_rank"]), expert_id=int(sibling["expert_id"]),
            token_position=int(join["token_position"]),
        ))
    values = [task.service for task in tasks]
    realized_cv = math.sqrt(sum((value - 1.0) ** 2 for value in values) / len(values))
    ordered = sorted(tasks, key=lambda task: sha({"pairing": task.task_id}))
    world0: dict[str, float] = {}
    world1: dict[str, float] = {}
    for index in range(0, len(ordered), 2):
        first, second = ordered[index:index+2]
        world0[first.join_id], world0[second.join_id] = residual, 0.0
        world1[first.join_id], world1[second.join_id] = 0.0, residual
    if (
        sorted(world0.values()) != sorted(world1.values())
        or world0 == world1
        or set(world0) != {task.join_id for task in tasks}
    ):
        raise MatchedError("matched receiver aggregate state mismatch")
    payload = {
        "model": model, "split": split, "receiver": tasks[0].receiver_rank, "window": window_id,
        "rho": rho, "cv": cv, "residual": residual,
        "tasks": [task.__dict__ for task in tasks], "world0": world0, "world1": world1,
    }
    return MatchedWindow(model, split, tasks[0].receiver_rank, window_id, tuple(tasks), (world0, world1), rho, cv, realized_cv, residual, sha(payload))


def empirical_cvar(values: Sequence[float], alpha: float = 0.99) -> float:
    if not values or not 0 <= alpha < 1:
        raise MatchedError("invalid CVaR input")
    ordered = sorted(float(value) for value in values)
    best = math.inf
    for z in ordered:
        score = z + sum(max(0.0, value - z) for value in ordered) / ((1 - alpha) * len(ordered))
        best = min(best, score)
    return best


def evaluate_order(window: MatchedWindow, order: Sequence[int], worlds: Sequence[int]) -> tuple[float, float, list[float]]:
    if sorted(order) != list(range(len(window.tasks))):
        raise MatchedError("order is not a task permutation")
    finishes: dict[int, float] = {}
    now = 0.0
    for index in order:
        task = window.tasks[index]
        now = max(now, task.release) + task.service
        finishes[index] = now
    flows = []
    for world in worlds:
        residuals = window.residual_by_world[world]
        flows.extend(finishes[index] + residuals[task.join_id] - task.release for index, task in enumerate(window.tasks))
    return empirical_cvar(flows), sum(flows) / len(flows), flows


def exhaustive_optimum(window: MatchedWindow, worlds: Sequence[int]) -> dict[str, Any]:
    best = (math.inf, math.inf)
    actions: set[str] = set()
    best_order: tuple[int, ...] | None = None
    for order in itertools.permutations(range(len(window.tasks))):
        cvar, mean, _flows = evaluate_order(window, order, worlds)
        objective = (cvar, mean)
        if cvar < best[0] - 1e-10 or (abs(cvar - best[0]) <= 1e-10 and mean < best[1] - 1e-10):
            best, actions, best_order = objective, {window.tasks[order[0]].task_id}, order
        elif abs(cvar - best[0]) <= 1e-10 and abs(mean - best[1]) <= 1e-10:
            actions.add(window.tasks[order[0]].task_id)
    if best_order is None:
        raise MatchedError("enumeration found no order")
    return {"cvar99": best[0], "mean": best[1], "first_action_set": sorted(actions), "order": list(best_order)}


def exhaustive_joint_r0(window: MatchedWindow) -> dict[str, Any]:
    """Exact R0 with world-specific schedules and one pooled CVaR then mean."""

    options: list[list[tuple[float, float, tuple[int, ...], list[float]]]] = [[], []]
    for world in (0, 1):
        for order in itertools.permutations(range(len(window.tasks))):
            cvar, mean, flows = evaluate_order(window, order, (world,))
            options[world].append((cvar, mean, order, flows))
    global_z = max(min(item[0] for item in options[0]), min(item[0] for item in options[1]))
    chosen = []
    action_sets = []
    for world in (0, 1):
        feasible = [item for item in options[world] if item[0] <= global_z + 1e-10]
        best_mean = min(item[1] for item in feasible)
        optimal = [item for item in feasible if abs(item[1] - best_mean) <= 1e-10]
        chosen.append(min(optimal, key=lambda item: tuple(window.tasks[index].task_id for index in item[2])))
        action_sets.append(sorted({window.tasks[item[2][0]].task_id for item in optimal}))
    flows = chosen[0][3] + chosen[1][3]
    return {
        "cvar99": empirical_cvar(flows),
        "mean": sum(flows) / len(flows),
        "orders": [list(chosen[0][2]), list(chosen[1][2])],
        "first_action_sets": action_sets,
        "global_cvar_threshold": global_z,
    }


def milp_optimum(window: MatchedWindow, worlds: Sequence[int]) -> dict[str, Any]:
    """Exact permutation-selection MILP for the frozen eight-task horizon.

    Every binary variable selects one release-feasible serial permutation.
    This avoids the SciPy-1.18/HiGHS numerical failure observed in the
    equivalent pairwise big-M formulation, while retaining an explicit MILP
    and an independent exhaustive replay check.
    """

    import numpy as np
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import csr_matrix, vstack

    if len(window.tasks) != TASKS_PER_WINDOW and len(window.tasks) != 4:
        raise MatchedError("permutation MILP only supports reviewed 8-task horizon or 4-task test")
    orders = list(itertools.permutations(range(len(window.tasks))))
    cvars = np.empty(len(orders), dtype=float)
    means = np.empty(len(orders), dtype=float)
    for index, order in enumerate(orders):
        cvars[index], means[index], _flows = evaluate_order(window, order, worlds)
    choose_one = csr_matrix(np.ones((1, len(orders)), dtype=float))
    stage1_constraint = LinearConstraint(choose_one, np.array([1.0]), np.array([1.0]))
    integrality = np.ones(len(orders), dtype=int)
    bounds = Bounds(np.zeros(len(orders)), np.ones(len(orders)))
    stage1 = milp(cvars, integrality=integrality, bounds=bounds, constraints=stage1_constraint,
                  options={"mip_rel_gap": 1e-9, "time_limit": 60, "presolve": True})
    if not stage1.success or stage1.x is None or stage1.status != 0 or stage1.mip_gap > 1e-7:
        raise MatchedError(f"MILP stage1 not proven optimal: {stage1.message}")
    stage1_z = float(stage1.fun)
    stage2_matrix = vstack((choose_one, csr_matrix(cvars.reshape(1, -1))), format="csr")
    stage2_constraint = LinearConstraint(
        stage2_matrix,
        np.array([1.0, -np.inf]),
        np.array([1.0, stage1_z + 1e-7]),
    )
    stage2 = milp(means, integrality=integrality, bounds=bounds, constraints=stage2_constraint,
                  options={"mip_rel_gap": 1e-9, "time_limit": 60, "presolve": True})
    if not stage2.success or stage2.x is None or stage2.status != 0 or stage2.mip_gap > 1e-7:
        raise MatchedError(f"MILP stage2 not proven optimal: {stage2.message}")
    selected = np.flatnonzero(stage2.x > 0.5)
    if len(selected) != 1:
        raise MatchedError("permutation MILP did not select exactly one order")
    order = list(orders[int(selected[0])])
    replay_cvar, replay_mean, replay_flows = evaluate_order(window, order, worlds)
    enumeration = exhaustive_optimum(window, worlds)
    if abs(replay_cvar - stage1_z) > 2e-6 or abs(replay_mean - float(stage2.fun)) > 2e-6:
        raise MatchedError("MILP objective/replay mismatch")
    if abs(replay_cvar - enumeration["cvar99"]) > 2e-6 or abs(replay_mean - enumeration["mean"]) > 2e-6:
        raise MatchedError("MILP differs from independent exhaustive optimum")
    return {
        "cvar99": replay_cvar, "mean": replay_mean, "flows": replay_flows, "order": order,
        "first_action_set": enumeration["first_action_set"],
        "stage1_status": int(stage1.status), "stage1_mip_gap": float(stage1.mip_gap), "stage1_objective": stage1_z,
        "stage2_status": int(stage2.status), "stage2_mip_gap": float(stage2.mip_gap), "stage2_objective": float(stage2.fun),
    }


def policy_order(window: MatchedWindow, policy: str, world: int | None) -> list[int]:
    pending = set(range(len(window.tasks)))
    order = []
    now = 0.0
    last_sender = -1
    while pending:
        ready = [index for index in pending if window.tasks[index].release <= now + 1e-12]
        if not ready:
            now = min(window.tasks[index].release for index in pending)
            ready = [index for index in pending if window.tasks[index].release <= now + 1e-12]
        qdepth = Counter(window.tasks[index].receiver_rank for index in ready)
        if policy == "fcfs":
            key = lambda index: (window.tasks[index].release, window.tasks[index].task_id)
        elif policy == "spt":
            key = lambda index: (window.tasks[index].service, window.tasks[index].release, window.tasks[index].task_id)
        elif policy == "receiver_qdepth":
            key = lambda index: (-qdepth[window.tasks[index].receiver_rank], window.tasks[index].release, window.tasks[index].task_id)
        elif policy == "sender_drr":
            senders = sorted({window.tasks[index].sender_rank for index in ready})
            target = next((sender for sender in senders if sender > last_sender), senders[0])
            key = lambda index: (window.tasks[index].sender_rank != target, window.tasks[index].release, window.tasks[index].task_id)
        elif policy == "topology_finish":
            key = lambda index: (window.tasks[index].service + 0.1 * qdepth[window.tasks[index].receiver_rank], window.tasks[index].release, window.tasks[index].task_id)
        elif policy == "sender_join_remaining":
            key = lambda index: (1, window.tasks[index].release, window.tasks[index].service, window.tasks[index].task_id)
        elif policy == "receiver_shadow_price":
            if world is None:
                raise MatchedError("receiver policy requires keyed world")
            key = lambda index: (-window.residual_by_world[world][window.tasks[index].join_id], window.tasks[index].release, window.tasks[index].service, window.tasks[index].task_id)
        else:
            raise MatchedError("unknown policy")
        chosen = min(ready, key=key)
        pending.remove(chosen)
        order.append(chosen)
        now = max(now, window.tasks[chosen].release) + window.tasks[chosen].service
        last_sender = window.tasks[chosen].sender_rank
    return order


def rel(base: float, candidate: float) -> float:
    if base <= 0:
        raise MatchedError("non-positive denominator")
    return (base - candidate) / base


def evaluate_cell(model: str, pools: Mapping[str, Sequence[Sequence[Mapping[str, Any]]]], rho: float, cv: float, residual: float) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selection_scores = {}
    for policy in BASELINES:
        values = []
        for index, joins in enumerate(pools["selection_half"]):
            window = materialize(model, "selection_half", index, joins, rho, cv, residual)
            order = policy_order(window, policy, None)
            cvar, _mean, _ = evaluate_order(window, order, (0, 1))
            values.append(cvar)
        selection_scores[policy] = sum(values) / len(values)
    selected = min(BASELINES, key=lambda policy: (selection_scores[policy], policy))
    rows = []
    for index, joins in enumerate(pools["holdout_half"]):
        window = materialize(model, "holdout_half", index, joins, rho, cv, residual)
        baseline_order = policy_order(window, selected, None)
        baseline_cvar, baseline_mean, _ = evaluate_order(window, baseline_order, (0, 1))
        receiver_metrics = []
        for world in (0, 1):
            order = policy_order(window, "receiver_shadow_price", world)
            receiver_metrics.append(evaluate_order(window, order, (world,)))
        receiver_flows = receiver_metrics[0][2] + receiver_metrics[1][2]
        receiver_cvar = empirical_cvar(receiver_flows)
        receiver_mean = sum(receiver_flows) / len(receiver_flows)
        b_opt = milp_optimum(window, (0, 1))
        r0 = exhaustive_joint_r0(window)
        r0_cvar = r0["cvar99"]
        r0_mean = r0["mean"]
        if r0_cvar > b_opt["cvar99"] + 2e-6:
            raise MatchedError("receiver-information optimum is worse than B")
        flip = (
            len(r0["first_action_sets"][0]) == 1
            and len(r0["first_action_sets"][1]) == 1
            and r0["first_action_sets"][0] != r0["first_action_sets"][1]
        )
        rows.append({
            "model": model, "rho": rho, "requested_service_cv": cv, "realized_service_cv": window.realized_service_cv,
            "residual_scale": residual, "receiver_rank": window.receiver_rank, "window": index,
            "request_cluster": sorted({task.request_id for task in window.tasks}), "window_fingerprint": window.fingerprint,
            "instance": {
                "tasks": [task.__dict__ for task in window.tasks],
                "residual_world0": dict(window.residual_by_world[0]),
                "residual_world1": dict(window.residual_by_world[1]),
                "total_service": sum(task.service for task in window.tasks),
                "release_span": max(task.release for task in window.tasks) - min(task.release for task in window.tasks),
            },
            "selected_simple_baseline": selected, "baseline_cvar99": baseline_cvar, "baseline_mean": baseline_mean,
            "receiver_shadow_cvar99": receiver_cvar, "receiver_shadow_mean": receiver_mean,
            "receiver_shadow_gain_vs_simple": rel(baseline_cvar, receiver_cvar),
            "B_joint_milp_cvar99": b_opt["cvar99"], "B_joint_milp_mean": b_opt["mean"],
            "R0_joint_exhaustive_cvar99": r0_cvar, "R0_joint_exhaustive_mean": r0_mean,
            "exact_information_cvar_gap": rel(b_opt["cvar99"], r0_cvar),
            "exact_information_mean_gap": rel(b_opt["mean"], r0_mean),
            "unique_optimal_first_action_flip": flip,
            "B_order_task_ids": [window.tasks[position].task_id for position in b_opt["order"]],
            "B_first_action_set": b_opt["first_action_set"],
            "R0_world0_order_task_ids": [window.tasks[position].task_id for position in r0["orders"][0]],
            "R0_world1_order_task_ids": [window.tasks[position].task_id for position in r0["orders"][1]],
            "R0_world0_first_action_set": r0["first_action_sets"][0],
            "R0_world1_first_action_set": r0["first_action_sets"][1],
            "B_solver": {key: b_opt[key] for key in b_opt if key.startswith("stage")},
            "R0_joint_exact": {"method": "independent_8_factorial_enumeration_with_shared_global_cvar", "global_cvar_threshold": r0["global_cvar_threshold"]},
        })
    return rows, {"selected": selected, "scores": selection_scores}


def run(route_root: Path) -> dict[str, Any]:
    all_rows = []
    selections = {}
    route_evidence = {}
    cells = [(rho, cv, PRIMARY_RESIDUAL) for rho in LOADS for cv in SERVICE_CVS]
    cells += [(0.85, 0.25, 0.5), (0.85, 0.25, 4.0)]
    for model in MODELS:
        joins, metadata = load_verified_joins(route_root, model)
        pools = outcome_blind_windows(joins)
        route_evidence[model] = {
            "route_manifest_sha256": metadata["manifest_sha256"],
            "route_trace_file_sha256": metadata["route_trace_file_sha256"],
            "selected_replay_layers": metadata["selected_replay_layers"],
        }
        for rho, cv, residual in cells:
            rows, selection = evaluate_cell(model, pools, rho, cv, residual)
            all_rows.extend(rows)
            selections[f"{model}:rho={rho}:cv={cv}:residual={residual}"] = selection
    summaries = []
    keys = sorted({(row["model"], row["rho"], row["requested_service_cv"], row["residual_scale"]) for row in all_rows})
    for key in keys:
        rows = [row for row in all_rows if (row["model"], row["rho"], row["requested_service_cv"], row["residual_scale"]) == key]
        summaries.append({
            "model": key[0], "rho": key[1], "requested_service_cv": key[2], "residual_scale": key[3],
            "selected_simple_baseline": rows[0]["selected_simple_baseline"],
            "median_exact_information_cvar_gap": median(row["exact_information_cvar_gap"] for row in rows),
            "min_exact_information_cvar_gap": min(row["exact_information_cvar_gap"] for row in rows),
            "median_exact_information_mean_gap": median(row["exact_information_mean_gap"] for row in rows),
            "first_action_flip_rate": sum(row["unique_optimal_first_action_flip"] for row in rows) / len(rows),
            "median_receiver_shadow_gain_vs_simple": median(row["receiver_shadow_gain_vs_simple"] for row in rows),
            "request_cluster_count": len({tuple(row["request_cluster"]) for row in rows}),
        })
    return {
        "schema_version": "ric-clean-v2-matched-receiver-headroom-v1",
        "status": "EXPLORATORY_NOT_SCIENTIFIC_RESULT", "scientific_result": False,
        "evidence_boundary": "ROUTE_REAL_SYNTHETIC_KEYED_POST_ARRIVAL_DAG_TAIL_MATCHED_WORLD_NORMALIZED_SERVICE_SINGLE_SHARED_CUT_OPEN_LOOP_B_CLAIRVOYANT_MILP_NOT_RDMA_NOT_SERVING",
        "sender_view_equality": "same_current_tasks_route_release_service_and_aggregate_receiver_residual_sum",
        "world_difference": "pairwise_swap_of_nondecaying_keyed_receiver_DAG_tail_locked_until_candidate_arrival_only",
        "producer_source_sha256": file_sha(Path(__file__)),
        "route_evidence": route_evidence, "baseline_selection": selections,
        "summaries": summaries, "per_window": all_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise MatchedError("refusing to overwrite output")
    result = run(args.route_root)
    encoded = json.dumps(result, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.partial-{os.getpid()}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(temporary, flags, 0o444)
    try:
        offset = 0
        while offset < len(encoded):
            written = os.write(descriptor, encoded[offset:])
            if written <= 0:
                raise MatchedError("atomic output write made no progress")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        os.link(temporary, args.output)
    except FileExistsError as exc:
        raise MatchedError("output appeared during atomic publish") from exc
    finally:
        temporary.unlink(missing_ok=True)
    directory_fd = os.open(args.output.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    print(json.dumps({"output": str(args.output), "cells": len(result["summaries"])}, sort_keys=True))


if __name__ == "__main__":
    main()
