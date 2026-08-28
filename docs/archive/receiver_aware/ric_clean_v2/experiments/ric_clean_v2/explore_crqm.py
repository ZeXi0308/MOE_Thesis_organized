#!/usr/bin/env python3
"""CRQM early receiver queue-map headroom pilot.

Native route identities are real calibration inputs. Receiver queue work is
calibrated from a single RTX 5090 serial CUDA-stream primitive benchmark and
replayed as virtual receiver state. This is not RDMA, serving, or a scientific
result.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Sequence

try:
    from . import capture_crqm_queue_calibration_gpu as calibration_core
    from .explore_receiver_matched_milp import (
        canonical,
        empirical_cvar,
        file_sha,
        load_verified_joins,
        sha,
    )
except ImportError:  # pragma: no cover
    import capture_crqm_queue_calibration_gpu as calibration_core  # type: ignore
    from explore_receiver_matched_milp import (  # type: ignore
        canonical,
        empirical_cvar,
        file_sha,
        load_verified_joins,
        sha,
    )


MODELS = ("olmoe", "llmjp")
TASKS = 6
WINDOWS = 8
PRIMARY_DEPTHS = (2, 8)
DEPTH_PAIRS = ((0, 0), PRIMARY_DEPTHS, (1, 4), (4, 16))


class CRQMError(RuntimeError):
    pass


@dataclass(frozen=True)
class Task:
    task_id: str
    join_id: str
    request_id: str
    sender_rank: int
    receiver_rank: int
    layer_id: int
    expert_id: int
    topk_slot: int
    cut_service_us: float
    receiver_service_us: float


@dataclass(frozen=True)
class Window:
    model: str
    sender_rank: int
    tasks: tuple[Task, ...]
    queue_maps: tuple[Mapping[int, float], Mapping[int, float]]
    queue_depth_maps: tuple[Mapping[int, int], Mapping[int, int]]
    queue_histories: tuple[Mapping[int, tuple[Mapping[str, Any], ...]], Mapping[int, tuple[Mapping[str, Any], ...]]]
    depth_pair: tuple[int, int]
    fingerprint: str


def load_calibration(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    calibration_core.validate_self_hash(value)
    if (
        value.get("schema_version") != "crqm-queue-drain-calibration-v1"
        or value.get("status") != "EXPLORATORY_CALIBRATION_INPUT_ONLY"
        or value.get("scientific_result") is not False
        or value.get("producer_source_sha256") != file_sha(Path(calibration_core.__file__))
        or value.get("summary_consumer_field") != "backlog_only_queue_work_us"
        or value.get("measurement_semantics") != calibration_core.MEASUREMENT_SEMANTICS
        or value.get("protocol_sha256") != file_sha(calibration_core.PROTOCOL)
        or value.get("environment", {}).get("gpu_name") != "NVIDIA GeForce RTX 5090"
    ):
        raise CRQMError("invalid CRQM calibration envelope")
    recomputed = calibration_core.validate_and_summarize(value.get("raw_trials", []))
    if canonical(recomputed) != canonical(value.get("summary")):
        raise CRQMError("calibration summary replay mismatch")
    points = {(str(row["model_key"]), int(row["queue_depth"])): row for row in recomputed}
    if set(points) != {(model, depth) for model in MODELS for depth in (0, 1, 2, 4, 8, 16)}:
        raise CRQMError("calibration point census mismatch")
    for model in MODELS:
        previous = -math.inf
        for depth in (0, 1, 2, 4, 8, 16):
            work = float(points[(model, depth)]["backlog_only_queue_work_us"])
            if not math.isfinite(work) or work < 0 or (depth == 0 and work != 0):
                raise CRQMError("invalid backlog-only queue work")
            if work + 1e-9 < previous:
                raise CRQMError("queue work calibration is non-monotonic")
            previous = work
    value["consumer_points"] = points
    return value


def analytic_cut_service_us(model: str) -> float:
    hidden = int(calibration_core.MODEL_SHAPES[model]["hidden"])
    payload_bytes = hidden * 2
    descriptor_bytes = 16
    transport_bytes = ((payload_bytes + descriptor_bytes + 15) // 16) * 16
    return transport_bytes * 8 / (200 * 1000)


def _request_split(joins: Sequence[Mapping[str, Any]]) -> tuple[set[str], set[str]]:
    by_receiver: dict[int, set[str]] = defaultdict(set)
    for join in joins:
        by_receiver[int(join["receiver_rank"])].add(str(join["request_id"]))
    selection: set[str] = set()
    holdout: set[str] = set()
    if set(by_receiver) != set(range(8)):
        raise CRQMError("receiver census mismatch")
    for receiver in range(8):
        requests = sorted(by_receiver[receiver], key=lambda value: sha({"crqm-request": value}))
        if len(requests) != 8:
            raise CRQMError("expected eight requests per receiver")
        selection.update(requests[:4])
        holdout.update(requests[4:])
    if selection & holdout:
        raise CRQMError("request split overlap")
    return selection, holdout


def select_windows(
    model: str,
    joins: Sequence[Mapping[str, Any]],
    cut_service_us: float,
    receiver_service_us: float,
) -> list[tuple[Task, ...]]:
    _selection, holdout = _request_split(joins)
    by_sender: dict[int, list[Task]] = defaultdict(list)
    for join in joins:
        if str(join["request_id"]) not in holdout:
            continue
        for sibling in join["siblings"]:
            sender = int(sibling["sender_rank"])
            task = Task(
                task_id=sha({"crqm": join["join_id"], "slot": sibling["topk_slot"]}),
                join_id=str(join["join_id"]),
                request_id=str(join["request_id"]),
                sender_rank=sender,
                receiver_rank=int(join["receiver_rank"]),
                layer_id=int(join["layer_id"]),
                expert_id=int(sibling["expert_id"]),
                topk_slot=int(sibling["topk_slot"]),
                cut_service_us=cut_service_us,
                receiver_service_us=receiver_service_us,
            )
            by_sender[sender].append(task)
    windows = []
    for sender in range(WINDOWS):
        pool = sorted(by_sender[sender], key=lambda task: sha({"crqm-window": sender, "task": task.task_id}))
        chosen: list[Task] = []
        used_requests: set[str] = set()
        for task in pool:
            if task.request_id in used_requests:
                continue
            chosen.append(task)
            used_requests.add(task.request_id)
            if len(chosen) == TASKS and len({item.receiver_rank for item in chosen}) >= 3:
                break
        if len(chosen) != TASKS or len({item.receiver_rank for item in chosen}) < 3:
            raise CRQMError("insufficient same-sender cross-receiver window")
        if any(item.sender_rank != sender for item in chosen):
            raise CRQMError("cross-sender window")
        windows.append(tuple(chosen))
    return windows


def materialize(
    model: str,
    sender_rank: int,
    tasks: tuple[Task, ...],
    depth_pair: tuple[int, int],
    depth_us: Mapping[int, float],
) -> Window:
    receivers = sorted({task.receiver_rank for task in tasks})
    if len(receivers) < 3:
        raise CRQMError("window lacks receiver diversity")
    ordered = sorted(receivers, key=lambda value: sha({"crqm-map": model, "sender": sender_rank, "receiver": value}))
    first, second = ordered[:2]
    low, high = depth_pair
    neutral = low if low == high else min(depth_pair)
    d0 = {receiver: neutral for receiver in receivers}
    d1 = dict(d0)
    d0[first], d0[second] = low, high
    d1[first], d1[second] = high, low
    q0 = {receiver: float(depth_us[d0[receiver]]) for receiver in receivers}
    q1 = {receiver: float(depth_us[d1[receiver]]) for receiver in receivers}
    if sorted(d0.values()) != sorted(d1.values()) or sorted(q0.values()) != sorted(q1.values()):
        raise CRQMError("matched queue multiset mismatch")
    def histories(depths: Mapping[int, int], work: Mapping[int, float], world: int):
        result = {}
        for receiver in receivers:
            depth = depths[receiver]
            each = 0.0 if depth == 0 else work[receiver] / depth
            events = tuple(
                {
                    "event_id": sha({"window": sender_rank, "world": world, "receiver": receiver, "prior": index}),
                    "receiver_rank": receiver,
                    "enqueue_time_us": 0.0,
                    "service_us": each,
                    "source": "measured_5090_backlog_only_queue_work_evenly_replayed",
                }
                for index in range(depth)
            )
            if not math.isclose(sum(event["service_us"] for event in events), work[receiver], rel_tol=1e-12, abs_tol=1e-12):
                raise CRQMError("queue history replay mismatch")
            result[receiver] = events
        return result

    h0 = histories(d0, q0, 0)
    h1 = histories(d1, q1, 1)
    payload = {
        "model": model,
        "sender": sender_rank,
        "tasks": [task.__dict__ for task in tasks],
        "depth_pair": depth_pair,
        "depth_world0": d0,
        "depth_world1": d1,
        "queue_world0": q0,
        "queue_world1": q1,
        "history_world0": h0,
        "history_world1": h1,
    }
    return Window(model, sender_rank, tasks, (q0, q1), (d0, d1), (h0, h1), depth_pair, sha(payload))


def evaluate_order(window: Window, order: Sequence[int], worlds: Sequence[int]) -> tuple[float, float, list[float]]:
    if sorted(order) != list(range(len(window.tasks))):
        raise CRQMError("order is not a permutation")
    flows = []
    for world in worlds:
        available = dict(window.queue_maps[world])
        sender_now = 0.0
        completions: dict[int, float] = {}
        for index in order:
            task = window.tasks[index]
            sender_now += task.cut_service_us
            completion = max(sender_now, available[task.receiver_rank]) + task.receiver_service_us
            available[task.receiver_rank] = completion
            completions[index] = completion
        flows.extend(completions[index] for index in range(len(window.tasks)))
    return empirical_cvar(flows), sum(flows) / len(flows), flows


def exhaustive(window: Window, worlds: Sequence[int]) -> dict[str, Any]:
    best = (math.inf, math.inf)
    best_order: tuple[int, ...] | None = None
    actions: set[str] = set()
    for order in itertools.permutations(range(len(window.tasks))):
        cvar, mean_value, _ = evaluate_order(window, order, worlds)
        if cvar < best[0] - 1e-10 or (abs(cvar - best[0]) <= 1e-10 and mean_value < best[1] - 1e-10):
            best = (cvar, mean_value)
            best_order = order
            actions = {window.tasks[order[0]].task_id}
        elif abs(cvar - best[0]) <= 1e-10 and abs(mean_value - best[1]) <= 1e-10:
            actions.add(window.tasks[order[0]].task_id)
    if best_order is None:
        raise CRQMError("no exhaustive optimum")
    return {"cvar99": best[0], "mean": best[1], "order": list(best_order), "first_action_set": sorted(actions)}


def exact_r0(window: Window) -> dict[str, Any]:
    options = [[], []]
    for world in (0, 1):
        for order in itertools.permutations(range(len(window.tasks))):
            cvar, mean_value, flows = evaluate_order(window, order, (world,))
            options[world].append((cvar, mean_value, order, flows))
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


def milp_b(window: Window) -> dict[str, Any]:
    import numpy as np
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import csr_matrix, vstack

    orders = list(itertools.permutations(range(len(window.tasks))))
    cvars = np.empty(len(orders), dtype=float)
    means = np.empty(len(orders), dtype=float)
    for index, order in enumerate(orders):
        cvars[index], means[index], _ = evaluate_order(window, order, (0, 1))
    choose = csr_matrix(np.ones((1, len(orders))))
    bounds = Bounds(np.zeros(len(orders)), np.ones(len(orders)))
    integrality = np.ones(len(orders), dtype=int)
    stage1 = milp(cvars, integrality=integrality, bounds=bounds,
                  constraints=LinearConstraint(choose, [1.0], [1.0]),
                  options={"mip_rel_gap": 1e-9, "time_limit": 30, "presolve": True})
    if not stage1.success or stage1.status != 0 or stage1.x is None or stage1.mip_gap > 1e-7:
        raise CRQMError("B MILP stage1 not proven optimal")
    matrix = vstack((choose, csr_matrix(cvars.reshape(1, -1))), format="csr")
    stage2 = milp(means, integrality=integrality, bounds=bounds,
                  constraints=LinearConstraint(matrix, [1.0, -np.inf], [1.0, float(stage1.fun) + 1e-7]),
                  options={"mip_rel_gap": 1e-9, "time_limit": 30, "presolve": True})
    if not stage2.success or stage2.status != 0 or stage2.x is None or stage2.mip_gap > 1e-7:
        raise CRQMError("B MILP stage2 not proven optimal")
    selected = np.flatnonzero(stage2.x > 0.5)
    if len(selected) != 1:
        raise CRQMError("B MILP selection is not unique")
    order = list(orders[int(selected[0])])
    cvar, mean_value, _ = evaluate_order(window, order, (0, 1))
    check = exhaustive(window, (0, 1))
    if abs(cvar - check["cvar99"]) > 2e-6 or abs(mean_value - check["mean"]) > 2e-6:
        raise CRQMError("MILP/enumeration mismatch")
    return {
        "cvar99": cvar,
        "mean": mean_value,
        "order": order,
        "first_action_set": check["first_action_set"],
        "stage1_status": int(stage1.status),
        "stage1_mip_gap": float(stage1.mip_gap),
        "stage2_status": int(stage2.status),
        "stage2_mip_gap": float(stage2.mip_gap),
    }


def relative(base: float, candidate: float) -> float:
    if base <= 0:
        raise CRQMError("non-positive baseline")
    return (base - candidate) / base


def run(route_root: Path, calibration_path: Path) -> dict[str, Any]:
    calibration = load_calibration(calibration_path)
    rows = []
    route_evidence = {}
    for model in MODELS:
        joins, metadata = load_verified_joins(route_root, model)
        points = calibration["consumer_points"]
        depth_us = {
            depth: float(points[(model, depth)]["backlog_only_queue_work_us"])
            for depth in (0, 1, 2, 4, 8, 16)
        }
        cut_service_us = analytic_cut_service_us(model)
        receiver_service_us = depth_us[1]
        if receiver_service_us <= 0:
            raise CRQMError("non-positive candidate receiver service")
        windows = select_windows(model, joins, cut_service_us, receiver_service_us)
        route_evidence[model] = {
            "route_trace_file_sha256": metadata["route_trace_file_sha256"],
            "selected_replay_layers": metadata["selected_replay_layers"],
        }
        for depth_pair in DEPTH_PAIRS:
            for window_id, tasks in enumerate(windows):
                window = materialize(model, window_id, tasks, depth_pair, depth_us)
                b = milp_b(window)
                r0 = exact_r0(window)
                if r0["cvar99"] > b["cvar99"] + 2e-6:
                    raise CRQMError("receiver information is worse than B")
                flip = (
                    len(r0["first_action_sets"][0]) == 1
                    and len(r0["first_action_sets"][1]) == 1
                    and r0["first_action_sets"][0] != r0["first_action_sets"][1]
                )
                rows.append({
                    "model": model,
                    "window": window_id,
                    "sender_rank": window.sender_rank,
                    "depth_pair": list(depth_pair),
                    "window_fingerprint": window.fingerprint,
                    "tasks": [task.__dict__ for task in window.tasks],
                    "queue_depth_world0": dict(window.queue_depth_maps[0]),
                    "queue_depth_world1": dict(window.queue_depth_maps[1]),
                    "queue_work_us_world0": dict(window.queue_maps[0]),
                    "queue_work_us_world1": dict(window.queue_maps[1]),
                    "queue_history_world0": {str(key): list(value) for key, value in window.queue_histories[0].items()},
                    "queue_history_world1": {str(key): list(value) for key, value in window.queue_histories[1].items()},
                    "defer_dominance": "ALL_TASKS_READY_NO_FUTURE_INFORMATION_POSITIVE_SERVICE_DEFER_STRICTLY_WEAKLY_DOMINATED",
                    "B_joint_cvar99_us": b["cvar99"],
                    "B_joint_mean_us": b["mean"],
                    "R0_joint_cvar99_us": r0["cvar99"],
                    "R0_joint_mean_us": r0["mean"],
                    "exact_information_cvar_gap": relative(b["cvar99"], r0["cvar99"]),
                    "unique_optimal_first_action_flip": flip,
                    "B_order_task_ids": [window.tasks[index].task_id for index in b["order"]],
                    "B_first_action_set": b["first_action_set"],
                    "R0_world0_order_task_ids": [window.tasks[index].task_id for index in r0["orders"][0]],
                    "R0_world1_order_task_ids": [window.tasks[index].task_id for index in r0["orders"][1]],
                    "R0_world0_first_action_set": r0["first_action_sets"][0],
                    "R0_world1_first_action_set": r0["first_action_sets"][1],
                    "B_solver": {key: b[key] for key in b if key.startswith("stage")},
                })
    summaries = []
    for model in MODELS:
        for depth_pair in DEPTH_PAIRS:
            selected = [row for row in rows if row["model"] == model and tuple(row["depth_pair"]) == depth_pair]
            summaries.append({
                "model": model,
                "depth_pair": list(depth_pair),
                "windows": len(selected),
                "median_exact_information_cvar_gap": median(row["exact_information_cvar_gap"] for row in selected),
                "min_exact_information_cvar_gap": min(row["exact_information_cvar_gap"] for row in selected),
                "first_action_flip_rate": sum(row["unique_optimal_first_action_flip"] for row in selected) / len(selected),
            })
    indexed = {(row["model"], tuple(row["depth_pair"])): row for row in summaries}
    primary_pass = all(
        indexed[(model, PRIMARY_DEPTHS)]["median_exact_information_cvar_gap"] >= 0.05
        and indexed[(model, PRIMARY_DEPTHS)]["first_action_flip_rate"] >= 0.25
        for model in MODELS
    )
    zero_pass = all(
        abs(indexed[(model, (0, 0))]["median_exact_information_cvar_gap"]) <= 1e-12
        and indexed[(model, (0, 0))]["first_action_flip_rate"] == 0.0
        for model in MODELS
    )
    sensitivity_pass = all(
        indexed[(model, pair)]["median_exact_information_cvar_gap"] >= -1e-12
        for model in MODELS for pair in ((1, 4), (4, 16))
    )
    decision = (
        "PROMISING_CRQM_L2_HEADROOM"
        if primary_pass and zero_pass and sensitivity_pass
        else "NO_GO_CRQM_L2_EARLY_RECEIVER_CONFLICT"
    )
    return {
        "schema_version": "crqm-route-real-5090-calibrated-l2-v1",
        "status": "EXPLORATORY_NOT_SCIENTIFIC_RESULT",
        "scientific_result": False,
        "evidence_boundary": "NATIVE_ROUTE_RTX5090_UNPACK_QUEUE_DRAIN_VIRTUAL_RECEIVER_APPLY_COMPLETION_ANALYTIC_200GBPS_CUT_NOT_RDMA_NOT_SERVING",
        "metric_semantics": "PER_CONTRIBUTION_RECEIVER_UNPACK_APPLY_COMPLETION_NOT_JOIN_CLOSURE_NOT_COMBINE",
        "cut_accounting": {
            "payload_rule": "hidden_times_2_bf16_row1_bytes",
            "descriptor_bytes": 16,
            "alignment_bytes": 16,
            "bandwidth_gbps": 200,
            "source": "ANALYTIC_NETWORK_L2_PROXY_NOT_RDMA",
        },
        "producer_source_sha256": file_sha(Path(__file__)),
        "calibration_file_sha256": file_sha(calibration_path),
        "route_evidence": route_evidence,
        "frozen_gate": {
            "model_and_primary_pass": primary_pass,
            "depth0_negative_control_pass": zero_pass,
            "sensitivity_nonnegative_pass": sensitivity_pass,
            "decision": decision,
        },
        "summaries": summaries,
        "per_window": rows,
    }


def atomic_write(path: Path, value: Mapping[str, Any]) -> None:
    path = path.absolute()
    if path.exists() or path.is_symlink():
        raise CRQMError("output already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.is_symlink():
        raise CRQMError("output parent identity mismatch")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    temporary = path.parent / f".{path.name}.{os.getpid()}.tmp"
    descriptor = os.open(temporary, flags, 0o600)
    try:
        payload = canonical(value) + b"\n"
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise CRQMError("output appeared during atomic publish") from exc
    finally:
        os.close(descriptor)
        if temporary.exists():
            temporary.unlink()
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route-root", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.route_root, args.calibration)
    atomic_write(args.output, result)
    print(json.dumps({"output": str(args.output), "rows": len(result["per_window"])}, sort_keys=True))


if __name__ == "__main__":
    main()
