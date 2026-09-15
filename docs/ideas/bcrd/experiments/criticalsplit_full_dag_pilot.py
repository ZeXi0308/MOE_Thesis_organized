from __future__ import annotations

"""Development-only CriticalSplit action-space qualification.

This module extends the frozen FrontierCredit simulator without modifying it.
It asks one narrow question: does a join-closing critical/bulk proper-subset
action enlarge the exact whole-ready-queue action space?  The result is a
deterministic CPU simulation qualification, never a serving or paper result.
"""

import argparse
from collections import Counter
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import platform
from statistics import median
import sys
import traceback
from typing import Mapping, Sequence

try:
    from . import frontiercredit_full_dag_pilot as frontier
except ImportError:
    import frontiercredit_full_dag_pilot as frontier  # type: ignore


EPS = frontier.EPS
MODEL = frontier.MODEL
ACTION_KINDS = ("bulk", "critical", "hold", "whole")
DECISION_THRESHOLDS = {
    "minimum_actual_eligible_cells": 2,
    "maximum_whole_oracle_median_capture": 0.90,
    "minimum_sham_applicable_eligible_cells": 2,
    "minimum_identity_gap": 0.10,
    "deadline_miss_delta_must_be_nonpositive": True,
}
FROZEN_MAX_ORACLE_STATES = 500_000
LOCK_SCHEMA = "criticalsplit-p0-run-lock-v1"
FROZEN_ACTION_SPACE = (
    "WHOLE + revealed join-closing CRITICAL/BULK proper subsets + bounded HOLD"
)
CLAIM_CEILING = (
    "This deterministic CPU simulation can only qualify or weaken the frozen "
    "CriticalSplit action-space formulation. It cannot establish natural prevalence, "
    "an online policy, GPU serving latency, EP/NCCL/RDMA behavior, production gains, "
    "or a paper result."
)
REQUIRED_LOCK_SOURCES = (
    "docs/ideas/bcrd/experiments/__init__.py",
    "docs/ideas/bcrd/experiments/core.py",
    "docs/ideas/bcrd/experiments/frontiercredit_full_dag_pilot.py",
    "docs/ideas/bcrd/experiments/test_frontiercredit_full_dag_pilot.py",
    "docs/ideas/bcrd/experiments/criticalsplit_full_dag_pilot.py",
    "docs/ideas/bcrd/experiments/test_criticalsplit_full_dag_pilot.py",
    "refine-logs/EXPERIMENT_PLAN_20260810_164938.md",
    "artifacts/frontiercredit_pilot/20260809_230235/pilot_protocol.json",
    "artifacts/frontiercredit_pilot/20260809_230235/pilot_results.json",
    "artifacts/frontiercredit_pilot/20260809_230235/pilot_summary.md",
)
FROZEN_EXPECTED_SHA256 = {
    "docs/ideas/bcrd/experiments/frontiercredit_full_dag_pilot.py": (
        "76f1db024e09a797b6235f1affd7c48026feb6e68557ae97eeb484f6d9ea2df7"
    ),
    "docs/ideas/bcrd/experiments/test_frontiercredit_full_dag_pilot.py": (
        "0012fb27671ca53fa76ce920e2002117718ef71ac21438fbadac34a94b95a016"
    ),
    "refine-logs/EXPERIMENT_PLAN_20260810_164938.md": (
        "6648e2a341904ea7bfa282ab8f794d43f98484a50b0e73d1dfa8811640afd474"
    ),
    "artifacts/frontiercredit_pilot/20260809_230235/pilot_protocol.json": (
        "1f7e0d554035ee0dc0916ef7a7c94a9e54915a50caf7c444290762a90706b495"
    ),
    "artifacts/frontiercredit_pilot/20260809_230235/pilot_results.json": (
        "a1c17ebbd848005b3def7a6bb1ec9bb4900c2052544261992bdbc709121ea179"
    ),
    "artifacts/frontiercredit_pilot/20260809_230235/pilot_summary.md": (
        "21732780067ec35c995e75096455b07447eb1f9c35e57d7d3cd0b3bb1862746a"
    ),
}

NodeId = frontier.NodeId
QueueKey = frontier.QueueKey
Episode = frontier.Episode
SimulationState = frontier.SimulationState


@dataclass(frozen=True, order=True)
class Action:
    """A canonical physical decision action.

    Exact node ids are part of the token so Oracle replay cannot silently
    re-evaluate a changed subset under the same symbolic action.
    """

    kind: str
    queue_key: QueueKey | None = None
    node_ids: tuple[NodeId, ...] = ()

    def __post_init__(self) -> None:
        if self.kind not in ACTION_KINDS:
            raise ValueError(f"unsupported action kind: {self.kind}")
        if self.kind == "hold":
            if self.queue_key is not None or self.node_ids:
                raise ValueError("hold cannot bind a queue or node ids")
            return
        if self.queue_key is None:
            raise ValueError("launch action requires a queue key")
        if not self.node_ids:
            raise ValueError("launch action requires at least one node")
        if len(set(self.node_ids)) != len(self.node_ids):
            raise ValueError("launch action contains duplicate node ids")
        if self.node_ids != tuple(sorted(self.node_ids)):
            raise ValueError("launch action node ids must be canonical-sorted")

    def token(self) -> str:
        return json.dumps(
            {
                "kind": self.kind,
                "node_ids": list(self.node_ids),
                "queue_key": None if self.queue_key is None else list(self.queue_key),
            },
            sort_keys=True,
            separators=(",", ":"),
        )

    @classmethod
    def from_token(cls, token: str) -> "Action":
        try:
            payload = json.loads(token)
        except json.JSONDecodeError as error:
            raise frontier.ProtocolError(f"invalid action token JSON: {error}") from error
        if not isinstance(payload, dict) or set(payload) != {
            "kind",
            "node_ids",
            "queue_key",
        }:
            raise frontier.ProtocolError("action token fields are not canonical")
        queue_raw = payload["queue_key"]
        if queue_raw is None:
            queue_key = None
        elif (
            isinstance(queue_raw, list)
            and len(queue_raw) == 3
            and all(type(value) is int for value in queue_raw)
        ):
            queue_key = tuple(queue_raw)
        else:
            raise frontier.ProtocolError("action token queue key is invalid")
        node_raw = payload["node_ids"]
        if not isinstance(node_raw, list) or not all(
            isinstance(value, str) for value in node_raw
        ):
            raise frontier.ProtocolError("action token node ids are invalid")
        kind_raw = payload["kind"]
        if not isinstance(kind_raw, str):
            raise frontier.ProtocolError("action token kind is invalid")
        try:
            action = cls(kind_raw, queue_key, tuple(node_raw))
        except ValueError as error:
            raise frontier.ProtocolError(str(error)) from error
        if action.token() != token:
            raise frontier.ProtocolError("action token is not canonical JSON")
        return action


@dataclass(frozen=True)
class SplitOracleTail:
    flow_us: float
    total_tardiness_us: float
    deadline_misses: int
    launches: int
    total_service_us: float
    request_completion_us: tuple[tuple[str, float], ...]
    actions: tuple[str, ...]

    @property
    def objective(self) -> tuple[object, ...]:
        return (
            self.flow_us,
            self.total_tardiness_us,
            self.deadline_misses,
            self.launches,
            self.total_service_us,
            self.actions,
        )


def _launch_ready_subset(
    episode: Episode,
    state: SimulationState,
    catalog: frontier.ServiceCatalog,
    queue_key: QueueKey,
    node_ids: Sequence[NodeId],
) -> tuple[SimulationState, float, tuple[NodeId, ...]]:
    """Launch one validated subset and preserve every unselected ReadyNode."""

    selected = tuple(node_ids)
    if not selected:
        raise frontier.ProtocolError("subset launch cannot be empty")
    if len(set(selected)) != len(selected):
        raise frontier.ProtocolError("subset launch contains duplicate node ids")
    if selected != tuple(sorted(selected)):
        raise frontier.ProtocolError("subset launch node ids are not canonical-sorted")
    eligible = frontier._eligible_queues(episode, state)
    if queue_key not in eligible:
        raise frontier.ProtocolError(f"queue {queue_key} is not currently eligible")
    eligible_ids = {item.node_id for item in eligible[queue_key]}
    if not set(selected) <= eligible_ids:
        raise frontier.ProtocolError("subset launch selected a non-ready or cross-queue node")
    replica, layer, _expert = queue_key
    if state.running[replica] is not None:
        raise frontier.ProtocolError("subset launch selected a busy executor")
    service_us = catalog.estimate_us(MODEL, layer, len(selected))
    selected_set = set(selected)
    ready = tuple(item for item in state.ready if item.node_id not in selected_set)
    running = list(state.running)
    start = state.now_us + episode.launch_cost_us
    running[replica] = frontier.RunningBatch(
        queue_key=queue_key,
        node_ids=selected,
        start_us=start,
        finish_us=start + service_us,
        service_us=service_us,
    )
    return replace(state, ready=ready, running=tuple(running)), service_us, selected


def _critical_bulk_partitions(
    episode: Episode,
    state: SimulationState,
    *,
    sham: bool,
) -> dict[QueueKey, tuple[tuple[NodeId, ...], tuple[NodeId, ...]]]:
    """Return proper, revealed-prefix join-closing partitions per queue."""

    eligible = frontier._decision_queues(episode, state)
    if not eligible:
        return {}
    node_map = episode.node_map
    joined = set(state.joined_groups)
    revealed = set(dict(state.revealed_at))
    ready_ids = {item.node_id for item in state.ready}
    observed = frontier._observation_map(episode, state, sham=sham)
    grouped: dict[frontier.GroupKey, list[NodeId]] = {}
    for node_id in sorted(revealed):
        if node_map[node_id].group_key in joined:
            continue
        grouped.setdefault(observed[node_id], []).append(node_id)

    critical_by_queue: dict[QueueKey, set[NodeId]] = {}
    for members in grouped.values():
        if len(members) != episode.top_k:
            continue
        blockers = [node_id for node_id in members if node_id in ready_ids]
        if not blockers:
            continue
        queue_keys = {node_map[node_id].queue_key for node_id in blockers}
        if len(queue_keys) != 1:
            continue
        queue_key = next(iter(queue_keys))
        eligible_ids = {item.node_id for item in eligible.get(queue_key, ())}
        if not set(blockers) <= eligible_ids:
            continue
        critical_by_queue.setdefault(queue_key, set()).update(blockers)

    partitions: dict[QueueKey, tuple[tuple[NodeId, ...], tuple[NodeId, ...]]] = {}
    for queue_key, items in eligible.items():
        all_ids = {item.node_id for item in items}
        critical = all_ids & critical_by_queue.get(queue_key, set())
        bulk = all_ids - critical
        if critical and bulk:
            partitions[queue_key] = (tuple(sorted(critical)), tuple(sorted(bulk)))
    return partitions


def canonical_actions(
    episode: Episode,
    state: SimulationState,
    *,
    sham: bool,
) -> tuple[Action, ...]:
    eligible = frontier._decision_queues(episode, state)
    partitions = _critical_bulk_partitions(episode, state, sham=sham)
    actions: set[Action] = set()
    for queue_key, items in eligible.items():
        whole = tuple(sorted(item.node_id for item in items))
        actions.add(Action("whole", queue_key, whole))
        if queue_key in partitions:
            critical, bulk = partitions[queue_key]
            actions.add(Action("critical", queue_key, critical))
            actions.add(Action("bulk", queue_key, bulk))
    if frontier._hold_allowed(episode, state):
        actions.add(Action("hold"))
    return tuple(sorted(actions))


def _physical_action_signature(
    actions: Sequence[Action],
) -> frozenset[tuple[str, QueueKey | None, tuple[NodeId, ...]]]:
    """Erase critical/bulk labels while retaining every physical transition."""

    return frozenset(
        ("hold", None, ())
        if action.kind == "hold"
        else ("launch", action.queue_key, action.node_ids)
        for action in actions
    )


def _apply_action(
    episode: Episode,
    state: SimulationState,
    catalog: frontier.ServiceCatalog,
    action: Action,
    *,
    sham: bool,
) -> tuple[SimulationState, float, tuple[NodeId, ...]]:
    if action not in canonical_actions(episode, state, sham=sham):
        raise frontier.ProtocolError("action is stale, ineligible, or noncanonical")
    if action.kind == "hold":
        return frontier._hold_once(episode, state), 0.0, ()
    assert action.queue_key is not None
    return _launch_ready_subset(
        episode,
        state,
        catalog,
        action.queue_key,
        action.node_ids,
    )


def _replay_actions(
    episode: Episode,
    catalog: frontier.ServiceCatalog,
    action_tokens: Sequence[str],
    *,
    sham: bool,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    state = frontier._settle(episode, frontier._initial_state(episode))
    ledger: list[dict[str, object]] = []
    total_service = 0.0
    launches = 0
    sham_applicable_decisions = 0
    sham_partition_changed_decisions = 0
    for token in action_tokens:
        while not frontier._terminal(episode, state) and not canonical_actions(
            episode, state, sham=sham
        ):
            state = frontier._advance_without_action(episode, state)
        if frontier._terminal(episode, state):
            raise frontier.ProtocolError("action trace continues after terminal completion")
        actual_actions = canonical_actions(episode, state, sham=False)
        sham_actions = canonical_actions(episode, state, sham=True)
        if set(actual_actions) != set(sham_actions):
            sham_partition_changed_decisions += 1
        if _physical_action_signature(actual_actions) != _physical_action_signature(
            sham_actions
        ):
            sham_applicable_decisions += 1
        action = Action.from_token(token)
        before = state.now_us
        state, service, selected = _apply_action(
            episode,
            state,
            catalog,
            action,
            sham=sham,
        )
        if action.kind == "hold":
            ledger.append(
                {
                    "kind": "hold",
                    "start_us": before,
                    "end_us": state.now_us,
                    "token": token,
                }
            )
            continue
        launches += 1
        total_service += service
        ledger.append(
            {
                "kind": action.kind,
                "time_us": before,
                "queue_key": list(action.queue_key or ()),
                "node_ids": list(selected),
                "rows": len(selected),
                "service_us": service,
                "token": token,
            }
        )
        state = frontier._settle(episode, state)

    while not frontier._terminal(episode, state):
        if canonical_actions(episode, state, sham=sham):
            raise frontier.ProtocolError("action trace ended while a decision remained")
        state = frontier._advance_without_action(episode, state)
    metrics = frontier._metrics(episode, state)
    launched_ids = [
        node_id
        for row in ledger
        if row["kind"] != "hold"
        for node_id in row["node_ids"]
    ]
    launch_counts = Counter(launched_ids)
    expected_nodes = set(episode.node_map)
    if set(launch_counts) != expected_nodes or any(
        count != 1 for count in launch_counts.values()
    ):
        raise frontier.ProtocolError("action replay did not launch every node exactly once")
    if (
        state.ready
        or any(batch is not None for batch in state.running)
        or state.pending_releases
        or set(state.joined_groups) != set(episode.group_map)
    ):
        raise frontier.ProtocolError("terminal action replay left nonterminal DAG state")
    return ledger, {
        **metrics,
        "launches": launches,
        "total_service_us": total_service,
        "sham_applicable_decisions": sham_applicable_decisions,
        "sham_partition_changed_decisions": sham_partition_changed_decisions,
        "node_conservation": {
            "expected_nodes": len(expected_nodes),
            "launched_nodes": len(launched_ids),
            "unique_launched_nodes": len(launch_counts),
            "exactly_once": True,
        },
    }


def solve_split_oracle(
    episode: Episode,
    catalog: frontier.ServiceCatalog,
    *,
    sham: bool = False,
    max_states: int = 500_000,
) -> dict[str, object]:
    if max_states <= 0:
        raise ValueError("max_states must be positive")
    evaluated = 0

    @lru_cache(maxsize=None)
    def solve(raw_state: SimulationState) -> SplitOracleTail:
        nonlocal evaluated
        state = frontier._settle(episode, raw_state)
        if state != raw_state:
            return solve(state)
        evaluated += 1
        if evaluated > max_states:
            raise frontier.ProtocolError("UNSOLVED_EXACT_STATE_LIMIT")
        if frontier._terminal(episode, state):
            metrics = frontier._metrics(episode, state)
            return SplitOracleTail(
                flow_us=float(metrics["flow_us"]),
                total_tardiness_us=float(metrics["total_tardiness_us"]),
                deadline_misses=int(metrics["deadline_misses"]),
                launches=0,
                total_service_us=0.0,
                request_completion_us=tuple(
                    sorted(dict(metrics["request_completion_us"]).items())
                ),
                actions=(),
            )
        actions = canonical_actions(episode, state, sham=sham)
        if not actions:
            return solve(frontier._advance_without_action(episode, state))
        candidates: list[SplitOracleTail] = []
        for action in actions:
            next_state, service, _selected = _apply_action(
                episode,
                state,
                catalog,
                action,
                sham=sham,
            )
            tail = solve(next_state)
            candidates.append(
                replace(
                    tail,
                    launches=tail.launches + int(action.kind != "hold"),
                    total_service_us=tail.total_service_us + service,
                    actions=(action.token(),) + tail.actions,
                )
            )
        return min(candidates, key=lambda result: result.objective)

    initial = frontier._settle(episode, frontier._initial_state(episode))
    result = solve(initial)
    ledger, replay = _replay_actions(
        episode,
        catalog,
        result.actions,
        sham=sham,
    )
    scalar_checks = (
        math.isclose(float(replay["flow_us"]), result.flow_us, abs_tol=EPS),
        math.isclose(
            float(replay["total_tardiness_us"]),
            result.total_tardiness_us,
            abs_tol=EPS,
        ),
        int(replay["deadline_misses"]) == result.deadline_misses,
        int(replay["launches"]) == result.launches,
        math.isclose(
            float(replay["total_service_us"]), result.total_service_us, abs_tol=EPS
        ),
        dict(replay["request_completion_us"]) == dict(result.request_completion_us),
    )
    if not all(scalar_checks):
        raise frontier.ProtocolError("split Oracle replay does not reproduce solved objective")
    return {
        "policy": "sham_split_exact_oracle" if sham else "actual_split_exact_oracle",
        "identity_mode": "revealed_prefix_sham" if sham else "revealed_physical",
        "status": "SOLVED_EXACT_SIMULATION_ONLY",
        "exact": True,
        "evaluation_type": "simulation_only",
        "scientific_result_eligible": False,
        "flow_us": result.flow_us,
        "total_tardiness_us": result.total_tardiness_us,
        "deadline_misses": result.deadline_misses,
        "request_completion_us": dict(result.request_completion_us),
        "launches": result.launches,
        "total_service_us": result.total_service_us,
        "action_tokens": list(result.actions),
        "actions": ledger,
        "critical_launches": sum(row["kind"] == "critical" for row in ledger),
        "bulk_launches": sum(row["kind"] == "bulk" for row in ledger),
        "proper_split_launches": sum(
            row["kind"] in {"critical", "bulk"} for row in ledger
        ),
        "sham_applicable_decisions": int(replay["sham_applicable_decisions"]),
        "sham_partition_changed_decisions": int(
            replay["sham_partition_changed_decisions"]
        ),
        "node_conservation": dict(replay["node_conservation"]),
        "states_evaluated": evaluated,
    }


def _objective_prefix(result: Mapping[str, object]) -> tuple[object, ...]:
    return (
        float(result["flow_us"]),
        float(result["total_tardiness_us"]),
        int(result["deadline_misses"]),
        int(result["launches"]),
        float(result["total_service_us"]),
    )


def _capture(immediate: float, candidate: float, expanded_oracle: float) -> float | None:
    headroom = immediate - expanded_oracle
    if headroom <= EPS:
        return None
    return (immediate - candidate) / headroom


def decide(cell_summaries: Sequence[Mapping[str, object]]) -> dict[str, object]:
    eligible = [row for row in cell_summaries if bool(row["actual_split_improves_whole"])]
    captures = [float(row["whole_capture"]) for row in eligible if row["whole_capture"] is not None]
    gaps = [float(row["identity_gap"]) for row in eligible if row["identity_gap"] is not None]
    median_capture = median(captures) if captures else None
    median_gap = median(gaps) if gaps else None
    sham_applicable = sum(int(row["sham_applicable_decisions"]) > 0 for row in eligible)
    conditions = {
        "minimum_actual_eligible_cells": len(eligible)
        >= DECISION_THRESHOLDS["minimum_actual_eligible_cells"],
        "critical_action_used_in_every_eligible_cell": bool(eligible)
        and all(int(row["critical_launches"]) > 0 for row in eligible),
        "maximum_whole_oracle_median_capture": median_capture is not None
        and median_capture
        < DECISION_THRESHOLDS["maximum_whole_oracle_median_capture"],
        "minimum_sham_applicable_eligible_cells": sham_applicable
        >= DECISION_THRESHOLDS["minimum_sham_applicable_eligible_cells"],
        "minimum_identity_gap": median_gap is not None
        and median_gap >= DECISION_THRESHOLDS["minimum_identity_gap"],
        "deadline_miss_delta_nonpositive": bool(cell_summaries)
        and all(
            int(row["deadline_miss_delta_vs_whole"]) <= 0
            for row in cell_summaries
        ),
    }
    supported = all(conditions.values())
    return {
        "verdict": "SUPPORT_ACTION_SPACE" if supported else "WEAKEN_ACTION_SPACE",
        "supports_action_space": supported,
        "paper_result": False,
        "eligible_cells": len(eligible),
        "sham_applicable_eligible_cells": sham_applicable,
        "median_whole_capture": median_capture,
        "median_identity_gap": median_gap,
        "conditions": conditions,
        "thresholds": dict(DECISION_THRESHOLDS),
        "reason": (
            "Actual join-closing proper subsets clear all frozen action-space gates."
            if supported
            else "At least one frozen CriticalSplit action-space gate is not satisfied."
        ),
    }


def run_pilot(*, max_oracle_states: int = 500_000) -> dict[str, object]:
    catalog = frontier.make_service_catalog()
    cells: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for episode in frontier.generate_eight_cells():
        immediate = frontier.simulate_policy(
            episode, catalog, frontier.ImmediatePolicy()
        )
        whole = frontier.solve_exact_oracle(
            episode, catalog, max_states=max_oracle_states
        )
        actual = solve_split_oracle(
            episode, catalog, sham=False, max_states=max_oracle_states
        )
        sham = solve_split_oracle(
            episode, catalog, sham=True, max_states=max_oracle_states
        )
        if _objective_prefix(actual) > _objective_prefix(whole):
            raise frontier.ProtocolError("expanded actual Oracle is worse than whole-ready Oracle")
        if _objective_prefix(actual) > _objective_prefix(immediate):
            raise frontier.ProtocolError("online whole-ready policy beat expanded exact Oracle")
        immediate_flow = float(immediate["flow_us"])
        whole_flow = float(whole["flow_us"])
        actual_flow = float(actual["flow_us"])
        sham_flow = float(sham["flow_us"])
        whole_capture = _capture(immediate_flow, whole_flow, actual_flow)
        sham_capture = _capture(immediate_flow, sham_flow, actual_flow)
        summary = {
            "episode_id": episode.episode_id,
            "immediate_flow_us": immediate_flow,
            "whole_oracle_flow_us": whole_flow,
            "actual_split_flow_us": actual_flow,
            "sham_split_flow_us": sham_flow,
            "expanded_headroom_us": immediate_flow - actual_flow,
            "split_increment_vs_whole_us": whole_flow - actual_flow,
            "actual_split_improves_whole": actual_flow < whole_flow - EPS,
            "whole_capture": whole_capture,
            "sham_capture": sham_capture,
            "identity_gap": None if sham_capture is None else 1.0 - sham_capture,
            "critical_launches": int(actual["critical_launches"]),
            "bulk_launches": int(actual["bulk_launches"]),
            "proper_split_launches": int(actual["proper_split_launches"]),
            "sham_applicable_decisions": int(actual["sham_applicable_decisions"]),
            "sham_partition_changed_decisions": int(
                actual["sham_partition_changed_decisions"]
            ),
            "deadline_miss_delta_vs_whole": int(actual["deadline_misses"])
            - int(whole["deadline_misses"]),
        }
        summaries.append(summary)
        cells.append(
            {
                "episode_id": episode.episode_id,
                "episode": {
                    "requests": [asdict(request) for request in episode.requests],
                    "contributions": [
                        asdict(contribution) for contribution in episode.contributions
                    ],
                    "replicas": episode.replicas,
                    "decode_steps": episode.decode_steps,
                    "layers": episode.layers,
                    "top_k": episode.top_k,
                    "tick_us": episode.tick_us,
                    "max_hold_us": episode.max_hold_us,
                    "combine_us": episode.combine_us,
                    "launch_cost_us": episode.launch_cost_us,
                    "nodes": [asdict(node) for node in episode.nodes],
                },
                "results": {
                    "immediate": immediate,
                    "whole_ready_exact_oracle": whole,
                    "actual_split_exact_oracle": actual,
                    "sham_split_exact_oracle": sham,
                },
                "summary": summary,
            }
        )
    decision = decide(summaries)
    return {
        "schema": "criticalsplit-full-dag-pilot-v1",
        "status": "COMPUTED_SIMULATION_ONLY",
        "evaluation_type": "simulation_only",
        "scientific_result_eligible": False,
        "paper_result": False,
        "protocol": {
            "cells": "frozen FrontierCredit overlap(2) x arrival(2) x deadline(2)",
            "action_space": FROZEN_ACTION_SPACE,
            "partition_visibility": "revealed physical identity or revealed-prefix identity sham",
            "oracle_visibility": "future-known exact search over identical physical transitions",
            "max_oracle_states": max_oracle_states,
            "decision_thresholds": dict(DECISION_THRESHOLDS),
        },
        "cells": cells,
        "decision": decision,
        "claim_ceiling": CLAIM_CEILING,
    }


def _workspace_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mapping_digest(values: Mapping[str, str]) -> str:
    encoded = json.dumps(dict(values), sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_json_atomic_exclusive(path: Path, payload: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    with temporary.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _source_path(relative: str) -> Path:
    root = _workspace_root()
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise frontier.ProtocolError(f"lock source path is unsafe: {relative}")
    resolved = (root / candidate).resolve()
    if resolved != root and root not in resolved.parents:
        raise frontier.ProtocolError(f"lock source escapes workspace: {relative}")
    if not resolved.is_file():
        raise frontier.ProtocolError(f"locked source is missing: {relative}")
    return resolved


def _validate_preflight(path: Path, source_files: Mapping[str, str]) -> None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        raise frontier.ProtocolError(f"preflight is unreadable: {error}") from error
    expected_fields = {
        "checks",
        "created_at_utc",
        "schema",
        "source_files_tested",
        "source_set_sha256",
        "status",
    }
    if not isinstance(payload, dict) or set(payload) != expected_fields:
        raise frontier.ProtocolError("preflight fields are not canonical")
    if payload["schema"] != "criticalsplit-preflight-v1" or payload["status"] != "PASS":
        raise frontier.ProtocolError("preflight status is not PASS")
    try:
        created_at = datetime.fromisoformat(payload["created_at_utc"])
    except (TypeError, ValueError) as error:
        raise frontier.ProtocolError("preflight timestamp is invalid") from error
    if created_at.tzinfo is None:
        raise frontier.ProtocolError("preflight timestamp must include a timezone")
    if payload["source_files_tested"] != dict(source_files):
        raise frontier.ProtocolError("preflight source map does not match the run lock")
    if payload["source_set_sha256"] != _mapping_digest(source_files):
        raise frontier.ProtocolError("preflight source-set digest is invalid")
    checks = payload["checks"]
    expected_checks = {
        "criticalsplit_contract_tests",
        "frontier_tests",
        "old_artifact_replay",
        "py_compile",
    }
    if not isinstance(checks, dict) or set(checks) != expected_checks:
        raise frontier.ProtocolError("preflight check set is incomplete")
    for name, check in checks.items():
        if not isinstance(check, dict) or check.get("exit_code") != 0:
            raise frontier.ProtocolError(f"preflight check did not pass: {name}")
        if not isinstance(check.get("command"), str) or not check["command"]:
            raise frontier.ProtocolError(f"preflight command is missing: {name}")
    if checks["frontier_tests"].get("tests_run") != 13:
        raise frontier.ProtocolError("preflight did not run all 13 Frontier tests")
    critical_tests = checks["criticalsplit_contract_tests"].get("tests_run")
    if type(critical_tests) is not int or critical_tests < 11:
        raise frontier.ProtocolError("preflight CriticalSplit test count is incomplete")
    replay_hashes = checks["old_artifact_replay"].get("artifact_sha256")
    expected_replay_hashes = {
        relative: digest
        for relative, digest in FROZEN_EXPECTED_SHA256.items()
        if relative.startswith("artifacts/frontiercredit_pilot/")
    }
    if replay_hashes != expected_replay_hashes:
        raise frontier.ProtocolError("preflight historical artifact replay hashes are invalid")


def _validate_lock(
    lock_path: Path,
    expected_lock_sha256: str,
    *,
    max_oracle_states: int,
) -> dict[str, object]:
    if (
        len(expected_lock_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_lock_sha256)
    ):
        raise frontier.ProtocolError("expected lock SHA256 must be 64 lowercase hex characters")
    if not lock_path.is_file():
        raise frontier.ProtocolError(f"run lock is missing: {lock_path}")
    actual_lock_sha256 = _sha256(lock_path)
    if actual_lock_sha256 != expected_lock_sha256:
        raise frontier.ProtocolError("run lock SHA256 mismatch")
    try:
        payload = json.loads(lock_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        raise frontier.ProtocolError(f"run lock is unreadable: {error}") from error
    expected_fields = {
        "action_space",
        "claim_ceiling",
        "created_at_utc",
        "decision_thresholds",
        "episode_ids",
        "evaluation_type",
        "experiment_plan",
        "max_oracle_states",
        "paper_result",
        "preflight",
        "preflight_sha256",
        "schema",
        "scientific_result_eligible",
        "source_files",
        "source_set_sha256",
        "status",
    }
    if not isinstance(payload, dict) or set(payload) != expected_fields:
        raise frontier.ProtocolError("run lock fields are not canonical")
    if payload["schema"] != LOCK_SCHEMA or payload["status"] != "LOCKED_BEFORE_COMPUTE":
        raise frontier.ProtocolError("run lock schema or status is invalid")
    try:
        created_at = datetime.fromisoformat(payload["created_at_utc"])
    except (TypeError, ValueError) as error:
        raise frontier.ProtocolError("run lock timestamp is invalid") from error
    if created_at.tzinfo is None:
        raise frontier.ProtocolError("run lock timestamp must include a timezone")
    if (
        payload["action_space"] != FROZEN_ACTION_SPACE
        or payload["claim_ceiling"] != CLAIM_CEILING
        or payload["evaluation_type"] != "simulation_only"
        or payload["scientific_result_eligible"] is not False
        or payload["paper_result"] is not False
    ):
        raise frontier.ProtocolError("run lock exceeds the frozen evidence boundary")
    if max_oracle_states != FROZEN_MAX_ORACLE_STATES:
        raise frontier.ProtocolError("formal run must use the frozen 500000-state cap")
    if payload["max_oracle_states"] != max_oracle_states:
        raise frontier.ProtocolError("run lock state cap does not match invocation")
    if payload["decision_thresholds"] != DECISION_THRESHOLDS:
        raise frontier.ProtocolError("run lock decision thresholds do not match source")
    expected_episode_ids = [
        episode.episode_id for episode in frontier.generate_eight_cells()
    ]
    if payload["episode_ids"] != expected_episode_ids:
        raise frontier.ProtocolError("run lock episode ids do not match frozen cells")
    if payload["experiment_plan"] != "refine-logs/EXPERIMENT_PLAN_20260810_164938.md":
        raise frontier.ProtocolError("run lock points to the wrong experiment plan")
    locked_sources = payload["source_files"]
    if not isinstance(locked_sources, dict) or set(locked_sources) != set(
        REQUIRED_LOCK_SOURCES
    ):
        raise frontier.ProtocolError("run lock source set is incomplete or unexpected")
    observed_sources: dict[str, str] = {}
    for relative in REQUIRED_LOCK_SOURCES:
        locked_digest = locked_sources[relative]
        if (
            not isinstance(locked_digest, str)
            or len(locked_digest) != 64
            or any(character not in "0123456789abcdef" for character in locked_digest)
        ):
            raise frontier.ProtocolError(f"run lock source hash is invalid: {relative}")
        observed_digest = _sha256(_source_path(relative))
        if observed_digest != locked_digest:
            raise frontier.ProtocolError(f"locked source hash mismatch: {relative}")
        frozen_digest = FROZEN_EXPECTED_SHA256.get(relative)
        if frozen_digest is not None and observed_digest != frozen_digest:
            raise frontier.ProtocolError(f"plan-frozen source hash mismatch: {relative}")
        observed_sources[relative] = observed_digest
    if payload["source_set_sha256"] != _mapping_digest(observed_sources):
        raise frontier.ProtocolError("run lock source-set digest is invalid")
    preflight_relative = payload["preflight"]
    preflight_digest = payload["preflight_sha256"]
    if not isinstance(preflight_relative, str) or not isinstance(preflight_digest, str):
        raise frontier.ProtocolError("run lock preflight binding is invalid")
    preflight_component = Path(preflight_relative)
    if preflight_component.is_absolute() or ".." in preflight_component.parts:
        raise frontier.ProtocolError("run lock preflight path is unsafe")
    lock_parent = lock_path.resolve().parent
    preflight_path = (lock_parent / preflight_component).resolve()
    if lock_parent not in preflight_path.parents or not preflight_path.is_file():
        raise frontier.ProtocolError("run lock preflight file is missing or escapes lock directory")
    if _sha256(preflight_path) != preflight_digest:
        raise frontier.ProtocolError("run lock preflight SHA256 mismatch")
    _validate_preflight(preflight_path, observed_sources)
    return {
        "lock_path": str(lock_path.resolve()),
        "lock_sha256": actual_lock_sha256,
        "source_files": observed_sources,
        "source_set_sha256": _mapping_digest(observed_sources),
        "preflight_path": str(preflight_path),
        "preflight_sha256": preflight_digest,
    }


def _validate_payload(payload: Mapping[str, object], *, max_oracle_states: int) -> None:
    if payload.get("schema") != "criticalsplit-full-dag-pilot-v1":
        raise frontier.ProtocolError("result schema is invalid")
    if payload.get("status") != "COMPUTED_SIMULATION_ONLY":
        raise frontier.ProtocolError("result status is invalid")
    if payload.get("evaluation_type") != "simulation_only":
        raise frontier.ProtocolError("result evaluation type is invalid")
    if payload.get("scientific_result_eligible") is not False:
        raise frontier.ProtocolError("simulation result cannot be scientific-result eligible")
    if payload.get("paper_result") is not False:
        raise frontier.ProtocolError("CriticalSplit P0 cannot be a paper result")
    if payload.get("claim_ceiling") != CLAIM_CEILING:
        raise frontier.ProtocolError("result claim ceiling is invalid")
    protocol = payload.get("protocol")
    expected_protocol = {
        "cells": "frozen FrontierCredit overlap(2) x arrival(2) x deadline(2)",
        "action_space": FROZEN_ACTION_SPACE,
        "partition_visibility": (
            "revealed physical identity or revealed-prefix identity sham"
        ),
        "oracle_visibility": (
            "future-known exact search over identical physical transitions"
        ),
        "max_oracle_states": max_oracle_states,
        "decision_thresholds": dict(DECISION_THRESHOLDS),
    }
    if protocol != expected_protocol:
        raise frontier.ProtocolError("result protocol does not match the frozen protocol")
    cells = payload.get("cells")
    if not isinstance(cells, list):
        raise frontier.ProtocolError("result cells are invalid")
    expected_ids = [episode.episode_id for episode in frontier.generate_eight_cells()]
    observed_ids = [cell.get("episode_id") for cell in cells if isinstance(cell, dict)]
    if observed_ids != expected_ids or len(observed_ids) != len(cells):
        raise frontier.ProtocolError("result cells do not match the frozen eight-cell order")
    summaries: list[Mapping[str, object]] = []
    for cell in cells:
        assert isinstance(cell, dict)
        summary = cell.get("summary")
        results = cell.get("results")
        if not isinstance(summary, dict) or not isinstance(results, dict):
            raise frontier.ProtocolError("result cell is missing summary or raw results")
        summaries.append(summary)
        for name in ("actual_split_exact_oracle", "sham_split_exact_oracle"):
            result = results.get(name)
            if not isinstance(result, dict):
                raise frontier.ProtocolError(f"result cell is missing {name}")
            conservation = result.get("node_conservation")
            if not isinstance(conservation, dict) or conservation.get("exactly_once") is not True:
                raise frontier.ProtocolError(f"{name} node conservation is not closed")
    recomputed = decide(summaries)
    if payload.get("decision") != recomputed:
        raise frontier.ProtocolError("parent decision recompute does not match payload")


def _summary_markdown(payload: Mapping[str, object]) -> str:
    decision = dict(payload["decision"])

    def fmt(value: object) -> str:
        return "NA" if value is None else f"{float(value):.6f}"

    lines = [
        "# CriticalSplit Full Request-DAG Action-Space Pilot",
        "",
        f"- Status: `{payload['status']}`",
        f"- Evaluation type: `{payload['evaluation_type']}`",
        f"- Scientific-result eligible: `{str(payload['scientific_result_eligible']).lower()}`",
        f"- Paper result: `{str(payload['paper_result']).lower()}`",
        f"- Verdict: `{decision['verdict']}`",
        f"- Reason: {decision['reason']}",
        "",
        "## Frozen gate recompute",
        "",
    ]
    for name, passed in dict(decision["conditions"]).items():
        lines.append(f"- `{name}`: `{str(bool(passed)).lower()}`")
    lines.extend(
        [
            "",
            "## Per-cell summary",
            "",
            "| Cell | Immediate flow | Whole flow | Split flow | Whole capture | Identity gap | Critical/Bulk launches | Sham physical/partition decisions | Miss delta |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for cell in payload["cells"]:
        summary = cell["summary"]
        lines.append(
            "| {cell} | {immediate:.6f} | {whole:.6f} | {split:.6f} | {capture} | {gap} | {critical}/{bulk} | {physical}/{partition} | {miss} |".format(
                cell=summary["episode_id"],
                immediate=float(summary["immediate_flow_us"]),
                whole=float(summary["whole_oracle_flow_us"]),
                split=float(summary["actual_split_flow_us"]),
                capture=fmt(summary["whole_capture"]),
                gap=fmt(summary["identity_gap"]),
                critical=int(summary["critical_launches"]),
                bulk=int(summary["bulk_launches"]),
                physical=int(summary["sham_applicable_decisions"]),
                partition=int(summary["sham_partition_changed_decisions"]),
                miss=int(summary["deadline_miss_delta_vs_whole"]),
            )
        )
    lines.extend(["", "## Evidence boundary", "", str(payload["claim_ceiling"]), ""])
    return "\n".join(lines)


def write_outputs(
    output_dir: Path,
    payload: Mapping[str, object],
    lock_report: Mapping[str, object],
    *,
    lock_path: Path,
    expected_lock_sha256: str,
    max_oracle_states: int,
) -> None:
    run_lock_path = output_dir / "RUN_LOCK.json"
    preflight_path = output_dir / "PREFLIGHT.json"
    if (
        not output_dir.is_dir()
        or {path.name for path in output_dir.iterdir()}
        != {run_lock_path.name, preflight_path.name}
        or _sha256(run_lock_path) != expected_lock_sha256
        or _sha256(preflight_path) != lock_report["preflight_sha256"]
    ):
        raise frontier.ProtocolError(
            "output writer requires only validated RUN_LOCK.json and PREFLIGHT.json"
        )
    protocol = dict(payload["protocol"])
    protocol.update(
        {
            "schema": payload["schema"],
            "evaluation_type": payload["evaluation_type"],
            "scientific_result_eligible": payload["scientific_result_eligible"],
            "paper_result": payload["paper_result"],
        }
    )
    protocol_path = output_dir / "pilot_protocol.json"
    results_path = output_dir / "pilot_results.json"
    decision_path = output_dir / "decision.json"
    summary_path = output_dir / "pilot_summary.md"
    status_path = output_dir / "RUN_STATUS.json"
    source_pre_path = output_dir / "SOURCE_PRE.json"
    source_post_path = output_dir / "SOURCE_POST.json"
    _write_json(protocol_path, protocol)
    _write_json(results_path, payload)
    _write_json(decision_path, payload["decision"])
    summary_path.write_text(_summary_markdown(payload), encoding="utf-8")
    _write_json(
        status_path,
        {
            "status": payload["status"],
            "evaluation_type": payload["evaluation_type"],
            "scientific_result_eligible": payload["scientific_result_eligible"],
            "paper_result": payload["paper_result"],
            "verdict": payload["decision"]["verdict"],
            "completion_authority": "COMPLETE.json only",
        },
    )
    _write_json(
        source_pre_path,
        {
            "schema": "criticalsplit-source-pre-v1",
            "lock_path": lock_report["lock_path"],
            "lock_sha256": lock_report["lock_sha256"],
            "source_files_pre": lock_report["source_files"],
            "source_set_sha256_pre": lock_report["source_set_sha256"],
            "git_commit": frontier._git_commit(_workspace_root()),
            "git_commit_note": "Context only; source hashes and run lock are authoritative.",
            "runtime": {
                "python_executable": sys.executable,
                "python_implementation": platform.python_implementation(),
                "python_version": platform.python_version(),
                "platform": platform.platform(),
            },
            "parent_recompute": "PASS",
        },
    )
    post_report = _validate_lock(
        lock_path,
        expected_lock_sha256,
        max_oracle_states=max_oracle_states,
    )
    mismatches = sorted(
        relative
        for relative in REQUIRED_LOCK_SOURCES
        if post_report["source_files"].get(relative)
        != lock_report["source_files"].get(relative)
    )
    _write_json(
        source_post_path,
        {
            "schema": "criticalsplit-source-post-v1",
            "source_files_pre": lock_report["source_files"],
            "source_files_post": post_report["source_files"],
            "source_set_sha256_pre": lock_report["source_set_sha256"],
            "source_set_sha256_post": post_report["source_set_sha256"],
            "mismatches": mismatches,
            "all_match": not mismatches,
        },
    )
    if mismatches:
        raise frontier.ProtocolError("locked sources changed before manifest seal")
    evidence_paths = (
        run_lock_path,
        preflight_path,
        protocol_path,
        results_path,
        decision_path,
        summary_path,
        status_path,
        source_pre_path,
        source_post_path,
    )
    manifest_path = output_dir / "MANIFEST.json"
    manifest_payload = {
        "schema": "criticalsplit-output-manifest-v1",
        "files": {path.name: _sha256(path) for path in evidence_paths},
    }
    _write_json(manifest_path, manifest_payload)
    persisted_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if persisted_manifest != manifest_payload:
        raise frontier.ProtocolError("persisted output manifest does not round-trip")
    for path in evidence_paths:
        if persisted_manifest["files"].get(path.name) != _sha256(path):
            raise frontier.ProtocolError(f"output hash mismatch before completion: {path.name}")
    seal_lock_report = _validate_lock(
        lock_path,
        expected_lock_sha256,
        max_oracle_states=max_oracle_states,
    )
    if seal_lock_report["source_files"] != lock_report["source_files"]:
        raise frontier.ProtocolError("locked sources changed before completion seal")
    expected_pre_complete_names = {path.name for path in evidence_paths} | {
        manifest_path.name
    }
    observed_pre_complete_names = {path.name for path in output_dir.iterdir()}
    if observed_pre_complete_names != expected_pre_complete_names:
        raise frontier.ProtocolError("output directory closure failed before COMPLETE")
    pre_complete_paths = tuple(sorted(output_dir.iterdir(), key=lambda path: path.name))
    complete_path = output_dir / "COMPLETE.json"
    if complete_path.exists():
        raise frontier.ProtocolError("COMPLETE.json existed before completion seal")
    _write_json_atomic_exclusive(
        complete_path,
        {
            "schema": "criticalsplit-complete-v1",
            "status": "SUCCESS_COMPLETE",
            "result_status": payload["status"],
            "authority_scope": "artifact_completeness_only",
            "authority_rule": (
                "Absence, parse failure, hash mismatch, or directory-closure mismatch makes "
                "the run invalid or incomplete."
            ),
            "verdict": payload["decision"]["verdict"],
            "evaluation_type": "simulation_only",
            "scientific_result_eligible": False,
            "paper_result": False,
            "source_binding_verified": True,
            "manifest_sha256": _sha256(manifest_path),
            "claim_ceiling": CLAIM_CEILING,
            "completed_at_utc": _utc_now(),
            "files": {path.name: _sha256(path) for path in pre_complete_paths},
        },
    )


def verify_complete(output_dir: Path) -> dict[str, object]:
    complete_path = output_dir / "COMPLETE.json"
    if not complete_path.is_file() or (output_dir / "failure.json").exists():
        raise frontier.ProtocolError("completion authority is absent or conflicts with failure")
    try:
        complete = json.loads(complete_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        raise frontier.ProtocolError(f"COMPLETE.json is unreadable: {error}") from error
    if (
        not isinstance(complete, dict)
        or complete.get("schema") != "criticalsplit-complete-v1"
        or complete.get("status") != "SUCCESS_COMPLETE"
        or complete.get("authority_scope") != "artifact_completeness_only"
        or complete.get("evaluation_type") != "simulation_only"
        or complete.get("scientific_result_eligible") is not False
        or complete.get("paper_result") is not False
        or complete.get("source_binding_verified") is not True
        or complete.get("claim_ceiling") != CLAIM_CEILING
    ):
        raise frontier.ProtocolError("COMPLETE.json evidence boundary is invalid")
    files = complete.get("files")
    if not isinstance(files, dict):
        raise frontier.ProtocolError("COMPLETE.json file map is invalid")
    observed_names = {path.name for path in output_dir.iterdir()}
    if observed_names != set(files) | {complete_path.name}:
        raise frontier.ProtocolError("completed output directory has missing or extra files")
    for name, expected_digest in files.items():
        path = output_dir / name
        if not path.is_file() or _sha256(path) != expected_digest:
            raise frontier.ProtocolError(f"completed output hash mismatch: {name}")
    manifest_path = output_dir / "MANIFEST.json"
    if complete.get("manifest_sha256") != _sha256(manifest_path):
        raise frontier.ProtocolError("COMPLETE.json does not bind MANIFEST.json")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        raise frontier.ProtocolError(f"MANIFEST.json is unreadable: {error}") from error
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema") != "criticalsplit-output-manifest-v1"
    ):
        raise frontier.ProtocolError("output manifest schema is invalid")
    manifest_files = manifest.get("files")
    if not isinstance(manifest_files, dict) or set(manifest_files) != set(files) - {
        manifest_path.name
    }:
        raise frontier.ProtocolError("output manifest file closure is invalid")
    for name, expected_digest in manifest_files.items():
        if _sha256(output_dir / name) != expected_digest:
            raise frontier.ProtocolError(f"manifest hash mismatch: {name}")
    return complete


def execute_once(
    output_dir: Path,
    lock_path: Path,
    expected_lock_sha256: str,
    *,
    max_oracle_states: int = FROZEN_MAX_ORACLE_STATES,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=False)
    stage = "VALIDATE_RUN_LOCK"
    lock_report: dict[str, object] | None = None
    try:
        lock_report = _validate_lock(
            lock_path,
            expected_lock_sha256,
            max_oracle_states=max_oracle_states,
        )
        stage = "COPY_LOCK_AND_PREFLIGHT"
        run_lock_path = output_dir / "RUN_LOCK.json"
        run_lock_path.write_bytes(lock_path.read_bytes())
        if _sha256(run_lock_path) != expected_lock_sha256:
            raise frontier.ProtocolError("output RUN_LOCK.json copy does not match expected SHA256")
        preflight_path = output_dir / "PREFLIGHT.json"
        preflight_path.write_bytes(Path(str(lock_report["preflight_path"])).read_bytes())
        if _sha256(preflight_path) != lock_report["preflight_sha256"]:
            raise frontier.ProtocolError("output PREFLIGHT.json copy does not match run lock")
        stage = "EXACT_SEARCH"
        payload = run_pilot(max_oracle_states=max_oracle_states)
        stage = "PARENT_RECOMPUTE"
        _validate_payload(payload, max_oracle_states=max_oracle_states)
        stage = "SOURCE_POST_COMPUTE"
        post_compute_report = _validate_lock(
            lock_path,
            expected_lock_sha256,
            max_oracle_states=max_oracle_states,
        )
        if post_compute_report["source_files"] != lock_report["source_files"]:
            raise frontier.ProtocolError("locked sources changed during exact search")
        stage = "WRITE_AND_SEAL_OUTPUTS"
        write_outputs(
            output_dir,
            payload,
            lock_report,
            lock_path=lock_path,
            expected_lock_sha256=expected_lock_sha256,
            max_oracle_states=max_oracle_states,
        )
        return payload
    except BaseException as error:
        complete_path = output_dir / "COMPLETE.json"
        if not complete_path.exists():
            try:
                message = str(error)
                error_code = (
                    "UNSOLVED_EXACT_STATE_LIMIT"
                    if "UNSOLVED_EXACT_STATE_LIMIT" in message
                    else type(error).__name__
                )
                _write_json(
                    output_dir / "failure.json",
                    {
                        "schema": "criticalsplit-failure-v1",
                        "status": "FAILED_INCOMPLETE",
                        "stage": stage,
                        "error_code": error_code,
                        "failed_at_utc": _utc_now(),
                        "error_type": type(error).__name__,
                        "error": message,
                        "traceback": traceback.format_exc(),
                        "expected_lock_sha256": expected_lock_sha256,
                        "validated_lock_sha256": (
                            None if lock_report is None else lock_report["lock_sha256"]
                        ),
                        "max_oracle_states": max_oracle_states,
                        "source_files_pre": (
                            None if lock_report is None else lock_report["source_files"]
                        ),
                        "complete_written": False,
                        "verdict_authorized": False,
                        "evaluation_type": "simulation_only",
                        "scientific_result_eligible": False,
                        "paper_result": False,
                        "authority_rule": "No COMPLETE.json means invalid or incomplete.",
                    },
                )
            except Exception:
                pass
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--lock", required=True)
    parser.add_argument("--expected-lock-sha256", required=True)
    parser.add_argument(
        "--max-oracle-states",
        type=int,
        default=FROZEN_MAX_ORACLE_STATES,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = execute_once(
        Path(args.output_dir),
        Path(args.lock),
        args.expected_lock_sha256,
        max_oracle_states=args.max_oracle_states,
    )
    print(json.dumps(payload["decision"], indent=2, sort_keys=True))
    print(f"wrote CriticalSplit pilot artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
