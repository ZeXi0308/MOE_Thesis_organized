#!/usr/bin/env python3
"""Sealed PhaseMap selection/holdout runner.

Selection and holdout are deliberately separate entry points.  Holdout only
consumes a frozen selection manifest; it never invokes kappa fitting.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

try:
    from . import capture_phasemap_lut_gpu as lutmod
    from . import phasemap_instances as inst
    from . import phasemap_baselines as baselines
    from . import phasemap_milp_crosscheck as milpcheck
    from . import phasemap_oracle_core as core
except ImportError:  # pragma: no cover
    import capture_phasemap_lut_gpu as lutmod  # type: ignore
    import phasemap_instances as inst  # type: ignore
    import phasemap_baselines as baselines  # type: ignore
    import phasemap_milp_crosscheck as milpcheck  # type: ignore
    import phasemap_oracle_core as core  # type: ignore


KAPPAS = (1.25, 1.5, 2.0, 3.0)
MODELS = ("olmoe", "llmjp")
SCHEMA = "phasemap-oracle-runner-v1"
SPLIT_SCHEMA = "phasemap-split-instance-bundle-v1"
SELECTION_RULE = "pooled_B0_miss_closest_to_0.5_tie_smaller_kappa"
PROTOCOL = Path(lutmod.PROTOCOL)
REQUIRED_HOLDOUT_ARTIFACTS = (
    "selection_manifest.json",
    "holdout_instance_manifest.json",
    "lut.json",
    "per_pair.jsonl",
    "per_request.jsonl",
    "baseline_results.json",
    "controls.json",
    "milp_crosscheck.json",
    "decision.json",
    "environment.json",
    "source_manifest.json",
    "summary.md",
)


class PhaseMapRunnerError(RuntimeError):
    pass


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _plain(value: Any) -> Any:
    if is_dataclass(value):
        return _plain(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _self_hashed(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = _plain(payload)
    if not isinstance(result, dict):
        raise PhaseMapRunnerError("self-hashed artifact payload must be a mapping")
    result["artifact_sha256"] = inst.object_sha256(result)
    return result


def _validate_self(value: Mapping[str, Any], field: str = "artifact_sha256") -> None:
    expected = value.get(field)
    payload = {key: item for key, item in value.items() if key != field}
    if not isinstance(expected, str) or expected != inst.object_sha256(payload):
        raise PhaseMapRunnerError(f"{field} mismatch")


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def frozen_config() -> dict[str, Any]:
    """Return every runner-side frozen choice not already owned by the protocol."""

    return {
        "schema_version": SCHEMA,
        "models": list(MODELS),
        "kappa_grid": list(KAPPAS),
        "selection_rule": SELECTION_RULE,
        "primary_depths": [8, 16],
        "depth_sensitivity": [8, 12],
        "equal_q_depths": [8, 8],
        "linear_grid_denominator": baselines.LINEAR_GRID_DENOMINATOR,
        "baseline_names": list(baselines.BASELINE_NAMES),
    }


def current_source_hashes() -> dict[str, str]:
    return {
        "runner": file_sha256(Path(__file__)),
        "core": file_sha256(Path(core.__file__)),
        "instances": file_sha256(Path(inst.__file__)),
        "lut_producer": file_sha256(Path(lutmod.__file__)),
        "baselines": file_sha256(Path(baselines.__file__)),
        "milp_crosscheck": file_sha256(Path(milpcheck.__file__)),
    }


def runtime_solver_provenance() -> dict[str, Any]:
    """Record the numeric runtime without making optional metadata a hidden gate."""

    try:
        import numpy as np
    except ImportError:
        numpy_version = "UNAVAILABLE"
    else:
        numpy_version = str(np.__version__)
    try:
        import scipy
        from scipy.optimize import milp
    except ImportError:
        return {
            "numpy_version": numpy_version,
            "scipy_version": "UNAVAILABLE",
            "milp_callable": "UNAVAILABLE",
            "highs_version": "UNAVAILABLE",
        }
    try:
        from scipy.optimize._highspy import _core as highs_core  # type: ignore[attr-defined]

        highs_version = ".".join(str(int(getattr(highs_core, field))) for field in (
            "HIGHS_VERSION_MAJOR", "HIGHS_VERSION_MINOR", "HIGHS_VERSION_PATCH",
        ))
    except (ImportError, AttributeError, TypeError, ValueError):
        highs_version = "UNAVAILABLE"
    return {
        "numpy_version": numpy_version,
        "scipy_version": str(scipy.__version__),
        "milp_callable": f"{milp.__module__}.{milp.__qualname__}",
        "highs_version": highs_version,
    }


def git_provenance() -> dict[str, Any]:
    """Record Git metadata when available; source hashes remain authoritative."""

    anchor = Path(__file__).resolve().parent

    try:
        head = subprocess.run(
            ["git", "-C", str(anchor), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(anchor), "status", "--porcelain=v1", "--untracked-files=all"],
            check=True, capture_output=True, text=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return {
            "git_available": False,
            "git_head": "UNAVAILABLE",
            "git_dirty": False,
            "git_status_porcelain_format": "UNAVAILABLE",
            "git_status_porcelain_sha256": hashlib.sha256(b"").hexdigest(),
        }
    if len(head) not in {40, 64} or any(character not in "0123456789abcdef" for character in head):
        return {
            "git_available": False,
            "git_head": "UNAVAILABLE",
            "git_dirty": False,
            "git_status_porcelain_format": "UNAVAILABLE",
            "git_status_porcelain_sha256": hashlib.sha256(b"").hexdigest(),
        }
    return {
        "git_available": True,
        "git_head": head,
        "git_dirty": bool(status),
        "git_status_porcelain_format": "v1-untracked-files-all",
        "git_status_porcelain_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
    }


def services_from_lut(value: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    lutmod.validate_artifact(value)
    result: dict[str, dict[str, float]] = {model: {} for model in MODELS}
    for row in value["summary"]:
        result[str(row["model_key"])][str(row["component"])] = float(row["median_cuda_event_us"])
    if any(set(result[model]) != set(lutmod.COMPONENTS) for model in MODELS):
        raise PhaseMapRunnerError("LUT primitive surface is incomplete")
    cut = value["shared_cut"]
    for model in MODELS:
        hidden = int(value["model_inputs"][model]["hidden"])
        raw_bytes = hidden * 2 + int(cut["descriptor_bytes"])
        aligned = math.ceil(raw_bytes / int(cut["alignment_bytes"])) * int(cut["alignment_bytes"])
        result[model]["analytic_cut"] = aligned * 8 / (float(cut["bandwidth_gbps"]) * 1_000.0)
    return result


def _validate_pair_manifest(pair: Mapping[str, Any]) -> None:
    expected = pair.get("manifest_sha256")
    payload = {key: item for key, item in pair.items() if key != "manifest_sha256"}
    if pair.get("schema_version") != inst.SCHEMA_VERSION or expected != inst.object_sha256(payload):
        raise PhaseMapRunnerError("instance pair self-hash/schema mismatch")
    if not isinstance(pair.get("pair_identity"), Mapping):
        raise PhaseMapRunnerError("instance pair lacks full pair_identity")


def isolated_critical_path_us(pair: Mapping[str, Any], request: str, service: Mapping[str, float]) -> float:
    identity = pair["pair_identity"]
    join = identity["joins"][request]
    by_sender: dict[int, list[tuple[bytes, str]]] = {}
    for row in join["siblings"]:
        key = str(row["full_sibling_key"])
        sender = int(row["identity"]["sender_rank"])
        by_sender.setdefault(sender, []).append((inst.canonical_json_bytes(row["identity"]), key))
    arrivals: list[tuple[float, str]] = []
    for identities in by_sender.values():
        elapsed = 0.0
        for _identity, key in sorted(identities):
            elapsed += service["sender_pack"] + service["analytic_cut"]
            arrivals.append((elapsed, key))
    arrivals.sort(key=lambda value: (value[0], value[1]))
    available = -math.inf
    completions = []
    for arrival, _full_sibling_key in arrivals:
        available = max(arrival, available) + service["receiver_unpack"]
        completions.append(available)
    return max(completions) + service["canonical_combine"]


def scenario_from_pair(
    pair: Mapping[str, Any], model: str, service: Mapping[str, float], kappa: float,
    *, kind: core.ScenarioKind = "primary",
) -> core.Scenario:
    _validate_pair_manifest(pair)
    if not math.isclose(float(pair["unpack_service_us"]), service["receiver_unpack"], rel_tol=1e-12):
        raise PhaseMapRunnerError("instance/LUT unpack service mismatch")
    identity = pair["pair_identity"]
    decision_keys = {
        str(record["full_sibling_key"])
        for request_rows in identity["decision_contributions"].values()
        for record in request_rows.values()
    }
    contributions = []
    joins = []
    carrier_records = []
    for request in (str(identity["request_a"]), str(identity["request_b"])):
        join = identity["joins"][request]
        sibling_ids = []
        for row in join["siblings"]:
            key = str(row["full_sibling_key"])
            sibling_ids.append(key)
            raw = row["identity"]
            contributions.append(core.Contribution(
                key, str(join["full_join_key"]), request, int(raw["sender_rank"]),
                int(raw["receiver_rank"]), service["sender_pack"], service["analytic_cut"],
                service["receiver_unpack"], key in decision_keys,
            ))
            if key not in decision_keys:
                carrier_records.append((key, int(raw["sender_rank"])))
        critical = isolated_critical_path_us(pair, request, service)
        joins.append(core.Join(
            str(join["full_join_key"]), request, int(join["receiver_rank"]), -critical,
            (kappa - 1.0) * critical, service["canonical_combine"], tuple(sibling_ids),
        ))
    task_by_key = {task.task_id: task for task in contributions}
    carrier_keys = {key for key, _sender in carrier_records}
    worlds = []
    for raw_world in sorted(pair["worlds"], key=lambda row: (row["q_bit"], row["j_bit"])):
        sender_rows = raw_world.get("sender_history")
        transit_rows = raw_world.get("receiver_transit_ledger")
        fifo_ledgers = raw_world.get("fifo_ledgers")
        if (
            not isinstance(sender_rows, list)
            or not isinstance(transit_rows, list)
            or not isinstance(fifo_ledgers, Mapping)
        ):
            raise PhaseMapRunnerError("world lacks explicit sender/receiver causal ledgers")
        sender_by_key: dict[str, Mapping[str, Any]] = {}
        for row in sender_rows:
            if not isinstance(row, Mapping):
                raise PhaseMapRunnerError("sender history row is malformed")
            key = str(row.get("full_sibling_key"))
            if key in sender_by_key:
                raise PhaseMapRunnerError("sender history key is duplicated")
            sender_by_key[key] = row
        if set(sender_by_key) != decision_keys | carrier_keys:
            raise PhaseMapRunnerError("sender history does not cover the exact sibling census")
        for key in decision_keys:
            row = sender_by_key[key]
            if row.get("event") != "decision_ready_unsent" or float(row.get("timestamp_us", math.nan)) != 0.0:
                raise PhaseMapRunnerError("decision sender history is not ready-unsent at t0")
        history_rows = []
        for key in sorted(carrier_keys):
            row = sender_by_key[key]
            timestamp = float(row.get("timestamp_us", math.nan))
            if (
                row.get("event") != "send_complete_no_commit_ack"
                or not math.isfinite(timestamp)
                or timestamp >= 0.0
            ):
                raise PhaseMapRunnerError("carrier sender history is not causal send-complete evidence")
            history_rows.append(core.SenderHistoryEvent(
                key, task_by_key[key].sender_rank, timestamp, False,
            ))
        history = tuple(history_rows)

        foreground_fifo: dict[str, Mapping[str, Any]] = {}
        jobs = []
        for receiver_key, receiver_rows in fifo_ledgers.items():
            if not isinstance(receiver_rows, list):
                raise PhaseMapRunnerError("receiver FIFO ledger is malformed")
            for row in receiver_rows:
                if not isinstance(row, Mapping):
                    raise PhaseMapRunnerError("receiver FIFO row is malformed")
                key = str(row.get("job_key"))
                if key in foreground_fifo and str(row.get("kind", "")).startswith("foreground_"):
                    raise PhaseMapRunnerError("foreground FIFO carrier is duplicated")
                foreground = str(row.get("kind", "")).startswith("foreground_")
                if foreground:
                    foreground_fifo[key] = row
                jobs.append(core.ReceiverJob(
                    key, int(receiver_key), float(row["arrival_us"]), service["receiver_unpack"],
                    key if foreground else None,
                ))
        if set(foreground_fifo) != carrier_keys:
            raise PhaseMapRunnerError("receiver FIFO does not cover the exact carrier census")
        transit_by_key: dict[str, Mapping[str, Any]] = {}
        for row in transit_rows:
            if not isinstance(row, Mapping):
                raise PhaseMapRunnerError("receiver transit row is malformed")
            key = str(row.get("full_sibling_key"))
            if key in transit_by_key:
                raise PhaseMapRunnerError("receiver transit key is duplicated")
            transit_by_key[key] = row
        if set(transit_by_key) != carrier_keys:
            raise PhaseMapRunnerError("receiver transit ledger does not cover the exact carrier census")
        for key in carrier_keys:
            row = transit_by_key[key]
            sender_us = float(row.get("sender_send_complete_us", math.nan))
            arrival_us = float(row.get("receiver_arrival_us", math.nan))
            hidden_us = float(row.get("hidden_transit_us", math.nan))
            if (
                not all(math.isfinite(value) for value in (sender_us, arrival_us, hidden_us))
                or sender_us != float(sender_by_key[key]["timestamp_us"])
                or arrival_us != float(foreground_fifo[key]["arrival_us"])
                or not math.isclose(hidden_us, arrival_us - sender_us, rel_tol=0.0, abs_tol=1e-12)
                or hidden_us < 0.0
            ):
                raise PhaseMapRunnerError("sender-to-receiver transit accounting mismatch")
        worlds.append(core.World(
            str(raw_world["world_id"]), int(str(raw_world["q_bit"])[1]),
            int(str(raw_world["j_bit"])[1]), tuple(jobs), history,
            "UNINFORMATIVE-J" if kind == "shuffled_key" else None,
        ))
    depths = pair["depths"]
    return core.Scenario(
        f"{model}:{pair['pair_key']}:{kind}", tuple(contributions), tuple(joins), tuple(worlds),
        kind, int(depths["low"]), int(depths["high"]),
    )


def _equal_j_scenario(scenario: core.Scenario) -> core.Scenario:
    source = {(world.q_bit, world.j_bit): world for world in scenario.worlds}
    joins = sorted(scenario.joins, key=lambda value: value.request_id)
    receiver_a, receiver_b = joins[0].receiver_rank, joins[1].receiver_rank
    worlds = []
    for q in (0, 1):
        jobs = tuple(
            job
            for job in source[(q, 0)].receiver_jobs
            if job.receiver_rank == receiver_a
        ) + tuple(
            job
            for job in source[(q, 1)].receiver_jobs
            if job.receiver_rank == receiver_b
        )
        for j in (0, 1):
            worlds.append(replace(
                source[(q, 0)], world_id=f"equal-j:q{q}j{j}", j_bit=j, receiver_jobs=jobs
            ))
    return replace(scenario, scenario_id=f"{scenario.scenario_id}:equal-j", worlds=tuple(worlds), kind="equal_j")


def _shuffled_scenario(scenario: core.Scenario) -> core.Scenario:
    worlds = tuple(replace(world, j_observation_override="UNINFORMATIVE-J") for world in scenario.worlds)
    return replace(scenario, scenario_id=f"{scenario.scenario_id}:shuffled", worlds=worlds, kind="shuffled_key")


def _fanout1_scenario(scenario: core.Scenario) -> core.Scenario:
    sender = min(task.sender_rank for task in scenario.contributions if task.is_decision)
    kept = tuple(task for task in scenario.contributions if task.is_decision and task.sender_rank == sender)
    joins = tuple(
        replace(join, sibling_task_ids=(next(task.task_id for task in kept if task.request_id == join.request_id),))
        for join in scenario.joins
    )
    worlds = tuple(
        replace(
            world,
            receiver_jobs=tuple(replace(job, task_id=None) for job in world.receiver_jobs),
            sender_history=(),
            j_observation_override=None,
        )
        for world in scenario.worlds
    )
    return core.Scenario(
        f"{scenario.scenario_id}:fanout1", kept, joins, worlds, "fanout1",
        scenario.low_depth, scenario.high_depth,
    )


def _no_conflict_scenario(scenario: core.Scenario) -> core.Scenario:
    tasks = tuple(replace(task, is_decision=False) for task in scenario.contributions)
    history = tuple(
        core.SenderHistoryEvent(task.task_id, task.sender_rank, -2_000.0 - index)
        for index, task in enumerate(sorted(tasks, key=lambda value: value.task_id))
    )
    decision_ids = {task.task_id for task in scenario.contributions if task.is_decision}
    worlds = []
    for world in scenario.worlds:
        jobs = list(world.receiver_jobs)
        for index, task in enumerate(sorted(tasks, key=lambda value: value.task_id)):
            if task.task_id in decision_ids:
                jobs.append(core.ReceiverJob(
                    f"no-conflict:{task.task_id}", task.receiver_rank, -1_000.0 + index * 2.0,
                    task.unpack_us, task.task_id,
                ))
        worlds.append(replace(world, receiver_jobs=tuple(jobs), sender_history=history))
    return core.Scenario(
        f"{scenario.scenario_id}:no-conflict", tasks, scenario.joins, tuple(worlds), "no_conflict",
        scenario.low_depth, scenario.high_depth,
    )


def control_scenario(scenario: core.Scenario, name: str) -> core.Scenario:
    if name == "equal_j": return _equal_j_scenario(scenario)
    if name == "fanout1": return _fanout1_scenario(scenario)
    if name == "no_conflict": return _no_conflict_scenario(scenario)
    if name == "shuffled_key": return _shuffled_scenario(scenario)
    raise PhaseMapRunnerError("unknown Scenario-level control")


def baseline_observation(scenario: core.Scenario, world: core.World) -> dict[str, Any]:
    state = core.causal_receiver_state(scenario, world)
    joins = {join.request_id: join for join in scenario.joins}
    tasks = []
    for task in sorted((row for row in scenario.contributions if row.is_decision), key=lambda row: row.task_id):
        join = joins[task.request_id]
        q = state["q_state"][str(task.receiver_rank)]
        j = state["j_state"][join.join_id]
        deficit = int(j["deficit"])
        tasks.append({
            "task_id": task.task_id, "request_id": task.request_id, "full_join_key": join.join_id,
            "sender_rank": task.sender_rank, "receiver_rank": task.receiver_rank,
            "request_arrival_us": join.request_arrival_us, "deadline_us": join.deadline_us,
            "ready_us": 0.0, "service_us": task.pack_us + task.cut_us + task.unpack_us,
            "receiver_service_us": task.unpack_us, "combine_service_us": join.combine_us,
            "receiver_work_us": float(q["unfinished_work_us"]),
            "receiver_availability_us": float(q["availability_us"]),
            "remaining_siblings": deficit,
        })
    return {
        "observation_id": inst.object_sha256([scenario.scenario_id, world.world_id, "baseline"]),
        "pair_key": scenario.scenario_id, "world_id": world.world_id, "now_us": 0.0, "tasks": tasks,
    }


def _world_objective(result: Mapping[str, Any]) -> list[float]:
    rows = result["requests"]
    return [
        float(sum(bool(row["miss"]) for row in rows)),
        float(sum(float(row["normalized_tardiness"]) for row in rows)),
        float(sum(float(row["join_close_us"]) for row in rows)),
    ]


def baseline_selection_examples(scenario: core.Scenario) -> list[dict[str, Any]]:
    examples = []
    for index, world in enumerate(scenario.worlds):
        observation = baseline_observation(scenario, world)
        objectives = {
            baselines.action_key(action): _world_objective(core.simulate(scenario, index, action))
            for action in core.enumerate_actions(scenario)
        }
        examples.append({"observation": observation, "action_objectives": objectives})
    return examples


def evaluate_baselines(
    scenarios: Sequence[core.Scenario], linear_artifact: Mapping[str, Any]
) -> dict[str, core.AggregateMetrics]:
    metrics, _ledgers = evaluate_baselines_with_ledgers(scenarios, linear_artifact)
    return metrics


def evaluate_baselines_with_ledgers(
    scenarios: Sequence[core.Scenario], linear_artifact: Mapping[str, Any]
) -> tuple[dict[str, core.AggregateMetrics], list[dict[str, Any]]]:
    rows: dict[str, list[core.PairMetrics]] = {name: [] for name in baselines.BASELINE_NAMES}
    pair_ledgers: list[dict[str, Any]] = []
    for scenario in scenarios:
        results: dict[str, list[dict[str, Any]]] = {name: [] for name in baselines.BASELINE_NAMES}
        pair_worlds = []
        for index, world in enumerate(scenario.worlds):
            observation = baseline_observation(scenario, world)
            actions = baselines.run_all_baselines(observation, linear_artifact)
            world_rows = {}
            for name, action in actions.items():
                replay = core.simulate(scenario, index, action)
                results[name].append(replay)
                world_rows[name] = {
                    "action": action,
                    "replay": replay,
                }
            pair_worlds.append({
                "world_id": world.world_id,
                "observation": observation,
                "baselines": world_rows,
            })
        for name in baselines.BASELINE_NAMES:
            rows[name].append(core.fold_four_world_results(results[name]))
        pair_ledgers.append({"scenario_id": scenario.scenario_id, "worlds": pair_worlds})
    return (
        {name: core.aggregate_pair_metrics(values) for name, values in rows.items()},
        pair_ledgers,
    )


def _split_certificate(pairs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(pairs) != 16:
        raise PhaseMapRunnerError("split certificate requires exactly 16 pairs")
    request_rows: dict[str, dict[str, Any]] = {}
    pair_rows = []
    for pair in pairs:
        _validate_pair_manifest(pair)
        pair_identity = pair["pair_identity"]
        pair_key = str(pair["pair_key"])
        request_ids = [str(value) for value in pair.get("request_ids", [])]
        if len(request_ids) != 2 or len(set(request_ids)) != 2:
            raise PhaseMapRunnerError("pair does not carry two distinct request identities")
        joins = pair_identity.get("joins")
        if not isinstance(joins, Mapping) or set(joins) != set(request_ids):
            raise PhaseMapRunnerError("pair identity does not retain both full native joins")
        pair_rows.append({
            "pair_key": pair_key,
            "pair_manifest_sha256": str(pair["manifest_sha256"]),
            "request_ids": sorted(request_ids),
        })
        for request in request_ids:
            if request in request_rows:
                raise PhaseMapRunnerError("split request identity is duplicated")
            join = joins[request]
            siblings = join.get("siblings")
            if not isinstance(siblings, list) or not siblings:
                raise PhaseMapRunnerError("full native top-k sibling identity is absent")
            sibling_keys = [str(row["full_sibling_key"]) for row in siblings]
            if len(sibling_keys) != len(set(sibling_keys)):
                raise PhaseMapRunnerError("full native top-k sibling identity is duplicated")
            request_rows[request] = {
                "pair_key": pair_key,
                "full_join_key": str(join["full_join_key"]),
                "full_join_identity_sha256": inst.object_sha256(join["full_join_identity"]),
                "full_sibling_keys": sorted(sibling_keys),
                "top_k": len(sibling_keys),
            }
    if len(request_rows) != 32 or len({row["pair_key"] for row in pair_rows}) != 16:
        raise PhaseMapRunnerError("split must retain 32 distinct requests in 16 distinct pairs")
    pair_rows.sort(key=lambda row: row["pair_key"])
    request_ids = sorted(request_rows)
    payload = {
        "pair_count": 16,
        "selected_request_count": 32,
        "request_ids": request_ids,
        "request_identity_sha256": inst.object_sha256(request_rows),
        "pairs": pair_rows,
    }
    return {**payload, "certificate_sha256": inst.object_sha256(payload)}


def _validate_split_certificate(
    certificate: Mapping[str, Any], pairs: Sequence[Mapping[str, Any]]
) -> None:
    expected = _split_certificate(pairs)
    if dict(certificate) != expected:
        raise PhaseMapRunnerError("split identity certificate mismatch")


def _validate_model_manifest_for_run(
    value: Mapping[str, Any], model: str, service: Mapping[str, float],
    lut_artifact: Mapping[str, Any],
) -> None:
    try:
        inst.validate_model_manifest(value)
    except Exception as exc:
        raise PhaseMapRunnerError("full model manifest validation failed") from exc
    identity = value.get("model_identity")
    service_provenance = value.get("service_provenance")
    lut_model_identity = (
        service_provenance.get("lut_model_identity")
        if isinstance(service_provenance, Mapping) else None
    )
    lut_inputs = lut_artifact.get("model_inputs")
    if (
        value.get("model") != model
        or not isinstance(identity, Mapping)
        or not isinstance(service_provenance, Mapping)
        or not isinstance(lut_model_identity, Mapping)
        or not isinstance(lut_inputs, Mapping)
        or not isinstance(lut_inputs.get(model), Mapping)
        or service_provenance.get("lut_artifact_sha256") != lut_artifact.get("artifact_sha256")
        or not math.isclose(
            float(service_provenance.get("unpack_service_us", math.nan)),
            float(service["receiver_unpack"]), rel_tol=1e-12, abs_tol=1e-12,
        )
        or identity.get("model_revision") != lut_inputs[model].get("model_revision")
        or identity.get("top_k") != lut_inputs[model].get("top_k")
        or lut_model_identity.get("model_revision") != lut_inputs[model].get("model_revision")
        or lut_model_identity.get("top_k") != lut_inputs[model].get("top_k")
        or lut_model_identity.get("hidden") != lut_inputs[model].get("hidden")
    ):
        raise PhaseMapRunnerError("model manifest differs from the frozen LUT/model identity")


def _validate_pair_reconstruction(pair: Mapping[str, Any], unpack_service_us: float) -> None:
    depths = pair.get("depths")
    mode = pair.get("control_mode")
    if (
        not isinstance(depths, Mapping)
        or set(depths) != {"low", "high"}
        or type(depths.get("low")) is not int
        or type(depths.get("high")) is not int
        or mode not in {"primary", "equal_q"}
        or (mode == "primary" and (depths.get("low"), depths.get("high")) != (8, 16))
        or (mode == "equal_q" and (depths.get("low"), depths.get("high")) != (8, 8))
    ):
        raise PhaseMapRunnerError("pair reconstruction parameters are malformed")
    try:
        rebuilt = inst.rebuild_from_pair_identity(
            pair["pair_identity"], unpack_service_us,
            (int(depths["low"]), int(depths["high"])), mode=str(mode),
        )
    except Exception as exc:
        raise PhaseMapRunnerError("pair canonical reconstruction failed") from exc
    if inst.canonical_json_bytes(pair) != inst.canonical_json_bytes(rebuilt):
        raise PhaseMapRunnerError("pair worlds differ from canonical pair_identity reconstruction")


def validate_split_bundle(
    value: Mapping[str, Any], model: str, split: str, *,
    model_manifest: Mapping[str, Any], lut_artifact: Mapping[str, Any],
) -> None:
    services = services_from_lut(lut_artifact)
    _validate_model_manifest_for_run(model_manifest, model, services[model], lut_artifact)
    _validate_self(value)
    if (
        value.get("schema_version") != SPLIT_SCHEMA
        or value.get("model") != model or value.get("split") != split
        or value.get("scientific_result") is not False
        or not _is_sha256(value.get("source_model_manifest_sha256"))
        or not _is_sha256(value.get("selection_certificate_sha256"))
        or not isinstance(value.get("model_identity"), Mapping)
        or not isinstance(value.get("route_provenance"), Mapping)
        or not isinstance(value.get("pairs"), list) or len(value["pairs"]) != 16
    ):
        raise PhaseMapRunnerError("split bundle contract mismatch")
    if (
        value["model_identity"].get("model_key") != model
        or value["route_provenance"].get("model_identity") != value["model_identity"]
        or any(
            not _is_sha256(value["route_provenance"].get(field))
            for field in inst.ROUTE_PROVENANCE_FIELDS
        )
    ):
        raise PhaseMapRunnerError("split bundle loses model/data/placement provenance")
    expected_pairs = model_manifest["splits"][split]["pairs"]
    if (
        value.get("source_model_manifest_sha256") != model_manifest.get("manifest_sha256")
        or value.get("selection_certificate_sha256")
        != model_manifest["selection_certificate"].get("certificate_sha256")
        or value.get("model_identity") != model_manifest.get("model_identity")
        or value.get("route_provenance") != model_manifest.get("route_provenance")
        or inst.canonical_json_bytes(value["pairs"]) != inst.canonical_json_bytes(expected_pairs)
    ):
        raise PhaseMapRunnerError("split bundle is not the exact split of its full model manifest")
    for pair in value["pairs"]:
        _validate_pair_reconstruction(pair, services[model]["receiver_unpack"])
    _split_certificate(value["pairs"])


def make_split_bundle(
    model_manifest: Mapping[str, Any], split: str, *, lut_artifact: Mapping[str, Any]
) -> dict[str, Any]:
    try:
        result = inst.make_split_bundle(model_manifest, split)
    except Exception as exc:
        raise PhaseMapRunnerError("model instance manifest hash/split mismatch") from exc
    validate_split_bundle(
        result, str(model_manifest["model"]), split,
        model_manifest=model_manifest, lut_artifact=lut_artifact,
    )
    return result


def validate_selection_manifest(
    value: Mapping[str, Any], *, expected_source_hashes: Mapping[str, str] | None = None,
    expected_lut_artifact_sha256: str | None = None,
) -> None:
    _validate_self(value)
    required = {
        "schema_version", "artifact", "scientific_result", "kappa_grid",
        "selection_rows", "selected_kappa", "selection_rule", "source_hashes",
        "protocol_sha256", "frozen_config", "config_sha256", "lut_artifact_sha256",
        "instance_hashes", "model_lineage", "linear_baseline_artifact", "artifact_sha256",
    }
    if (
        not required <= set(value)
        or value.get("schema_version") != SCHEMA
        or value.get("artifact") != "selection_manifest"
        or value.get("scientific_result") is not False
        or value.get("kappa_grid") != list(KAPPAS)
        or value.get("selection_rule") != SELECTION_RULE
        or value.get("frozen_config") != frozen_config()
        or value.get("config_sha256") != inst.object_sha256(frozen_config())
        or value.get("protocol_sha256") != file_sha256(PROTOCOL)
        or not _is_sha256(value.get("lut_artifact_sha256"))
    ):
        raise PhaseMapRunnerError("selection manifest frozen contract mismatch")
    source_hashes = value.get("source_hashes")
    if (
        not isinstance(source_hashes, Mapping)
        or set(source_hashes) != set(current_source_hashes())
        or any(not _is_sha256(item) for item in source_hashes.values())
        or (expected_source_hashes is not None and dict(source_hashes) != dict(expected_source_hashes))
    ):
        raise PhaseMapRunnerError("selection source hash closure mismatch")
    if (
        expected_lut_artifact_sha256 is not None
        and value.get("lut_artifact_sha256") != expected_lut_artifact_sha256
    ):
        raise PhaseMapRunnerError("selection LUT closure mismatch")
    rows = value.get("selection_rows")
    if not isinstance(rows, list) or len(rows) != len(KAPPAS):
        raise PhaseMapRunnerError("selection rows do not cover the frozen kappa grid")
    normalized_rows = []
    for expected_kappa, row in zip(KAPPAS, rows):
        if not isinstance(row, Mapping) or set(row) != {"kappa", "B0_miss_by_model", "pooled_B0_miss"}:
            raise PhaseMapRunnerError("selection row schema mismatch")
        by_model = row["B0_miss_by_model"]
        if row["kappa"] != expected_kappa or not isinstance(by_model, Mapping) or set(by_model) != set(MODELS):
            raise PhaseMapRunnerError("selection row model/kappa identity mismatch")
        values = [float(by_model[model]) for model in MODELS]
        pooled = float(row["pooled_B0_miss"])
        if (
            any(not math.isfinite(item) or not 0.0 <= item <= 1.0 for item in values)
            or not math.isclose(pooled, sum(values) / len(values), rel_tol=0.0, abs_tol=1e-12)
        ):
            raise PhaseMapRunnerError("selection miss accounting mismatch")
        normalized_rows.append((abs(pooled - 0.5), float(expected_kappa), values))
    chosen = min(normalized_rows, key=lambda row: (row[0], row[1]))
    if value.get("selected_kappa") != chosen[1] or any(not 0.2 <= item <= 0.8 for item in chosen[2]):
        raise PhaseMapRunnerError("selection kappa rule or informativeness gate mismatch")
    lineage = value.get("model_lineage")
    instance_hashes = value.get("instance_hashes")
    if not isinstance(lineage, Mapping) or set(lineage) != set(MODELS) or not isinstance(instance_hashes, Mapping):
        raise PhaseMapRunnerError("selection model lineage is incomplete")
    for model in MODELS:
        row = lineage[model]
        if not isinstance(row, Mapping) or set(row) != {
            "source_model_manifest_sha256", "selection_certificate_sha256",
            "model_identity", "route_provenance", "selection_bundle_sha256",
            "selection_split_certificate",
        }:
            raise PhaseMapRunnerError("selection model lineage schema mismatch")
        if (
            not _is_sha256(row["source_model_manifest_sha256"])
            or not _is_sha256(row["selection_certificate_sha256"])
            or not isinstance(row["model_identity"], Mapping)
            or row["model_identity"].get("model_key") != model
            or not isinstance(row["route_provenance"], Mapping)
            or row["route_provenance"].get("model_identity") != row["model_identity"]
            or not isinstance(row["selection_split_certificate"], Mapping)
            or instance_hashes.get(model) != row["selection_bundle_sha256"]
        ):
            raise PhaseMapRunnerError("selection model lineage certificate mismatch")
        certificate_payload = dict(row["selection_split_certificate"])
        certificate_hash = certificate_payload.pop("certificate_sha256", None)
        if certificate_hash != inst.object_sha256(certificate_payload):
            raise PhaseMapRunnerError("selection split certificate self-hash mismatch")
    expected_linear_source = inst.object_sha256(
        {model: instance_hashes[model] for model in MODELS}
    )
    try:
        baselines.validate_linear_artifact(value["linear_baseline_artifact"])
    except Exception as exc:
        raise PhaseMapRunnerError("selection linear artifact validation failed") from exc
    if value["linear_baseline_artifact"].get("selection_source_sha256") != expected_linear_source:
        raise PhaseMapRunnerError("selection linear artifact source closure mismatch")


def validate_holdout_lineage(
    bundles: Mapping[str, Mapping[str, Any]],
    selection_bundles: Mapping[str, Mapping[str, Any]],
    model_manifests: Mapping[str, Mapping[str, Any]],
    lut_artifact: Mapping[str, Any],
    selection_manifest: Mapping[str, Any],
) -> None:
    validate_selection_manifest(
        selection_manifest,
        expected_source_hashes=current_source_hashes(),
        expected_lut_artifact_sha256=str(lut_artifact.get("artifact_sha256")),
    )
    evidence = _selection_replay_evidence(
        selection_bundles, model_manifests, lut_artifact
    )
    if any(
        inst.canonical_json_bytes(selection_manifest[field])
        != inst.canonical_json_bytes(evidence[field])
        for field in ("selection_rows", "selected_kappa", "instance_hashes", "model_lineage")
    ):
        raise PhaseMapRunnerError("selection manifest differs from recomputed frozen selection evidence")
    try:
        baselines.validate_linear_fit_against_examples(
            selection_manifest["linear_baseline_artifact"],
            evidence["examples"],
            selection_source_sha256=evidence["selection_source_sha256"],
        )
    except Exception as exc:
        raise PhaseMapRunnerError(
            "selection manifest differs from recomputed frozen selection evidence"
        ) from exc
    lineage = selection_manifest["model_lineage"]
    for model in MODELS:
        bundle = bundles[model]
        validate_split_bundle(
            selection_bundles[model], model, "selection",
            model_manifest=model_manifests[model], lut_artifact=lut_artifact,
        )
        validate_split_bundle(
            bundle, model, "holdout",
            model_manifest=model_manifests[model], lut_artifact=lut_artifact,
        )
        selected = lineage[model]
        selection_ids = set(selected["selection_split_certificate"]["request_ids"])
        holdout_certificate = _split_certificate(bundle["pairs"])
        holdout_ids = set(holdout_certificate["request_ids"])
        if (
            bundle["source_model_manifest_sha256"] != selected["source_model_manifest_sha256"]
            or bundle["selection_certificate_sha256"] != selected["selection_certificate_sha256"]
            or bundle["model_identity"] != selected["model_identity"]
            or bundle["route_provenance"] != selected["route_provenance"]
            or selection_ids & holdout_ids
            or len(selection_ids | holdout_ids) != 64
        ):
            raise PhaseMapRunnerError("holdout is foreign to the frozen selection mother manifest")


def _selection_replay_evidence(
    bundles: Mapping[str, Mapping[str, Any]],
    model_manifests: Mapping[str, Mapping[str, Any]],
    lut_artifact: Mapping[str, Any],
) -> dict[str, Any]:
    services = services_from_lut(lut_artifact)
    rows = []
    for kappa in KAPPAS:
        by_model = {}
        for model in MODELS:
            validate_split_bundle(
                bundles[model], model, "selection",
                model_manifest=model_manifests[model], lut_artifact=lut_artifact,
            )
            pair_metrics = [
                core.optimize_arm(scenario_from_pair(pair, model, services[model], kappa), "B0")["metrics"]
                for pair in bundles[model]["pairs"]
            ]
            by_model[model] = core.aggregate_pair_metrics(pair_metrics).miss_rate
        pooled = sum(by_model.values()) / len(MODELS)
        rows.append({"kappa": kappa, "B0_miss_by_model": by_model, "pooled_B0_miss": pooled})
    chosen = min(rows, key=lambda row: (abs(row["pooled_B0_miss"] - 0.5), row["kappa"]))
    if any(not 0.2 <= value <= 0.8 for value in chosen["B0_miss_by_model"].values()):
        raise PhaseMapRunnerError("BLOCKED_UNINFORMATIVE_DEADLINE_GRID")
    selection_source_sha = inst.object_sha256({model: bundles[model]["artifact_sha256"] for model in MODELS})
    examples = []
    for model in MODELS:
        examples.extend(
            example
            for pair in bundles[model]["pairs"]
            for example in baseline_selection_examples(
                scenario_from_pair(pair, model, services[model], float(chosen["kappa"]))
            )
        )
    lineage = {
        model: {
            "source_model_manifest_sha256": bundles[model]["source_model_manifest_sha256"],
            "selection_certificate_sha256": bundles[model]["selection_certificate_sha256"],
            "model_identity": bundles[model]["model_identity"],
            "route_provenance": bundles[model]["route_provenance"],
            "selection_bundle_sha256": bundles[model]["artifact_sha256"],
            "selection_split_certificate": _split_certificate(bundles[model]["pairs"]),
        }
        for model in MODELS
    }
    return {
        "selection_rows": rows,
        "selected_kappa": chosen["kappa"],
        "selection_source_sha256": selection_source_sha,
        "examples": examples,
        "instance_hashes": {model: bundles[model]["artifact_sha256"] for model in MODELS},
        "model_lineage": lineage,
    }


def _recompute_selection_manifest(
    bundles: Mapping[str, Mapping[str, Any]],
    model_manifests: Mapping[str, Mapping[str, Any]],
    lut_artifact: Mapping[str, Any],
    source_hashes: Mapping[str, str],
) -> dict[str, Any]:
    evidence = _selection_replay_evidence(bundles, model_manifests, lut_artifact)
    linear_artifact = baselines.fit_separable_linear(
        evidence["examples"],
        selection_source_sha256=evidence["selection_source_sha256"],
    )
    if set(source_hashes) != set(current_source_hashes()) or any(
        not _is_sha256(value) for value in source_hashes.values()
    ):
        raise PhaseMapRunnerError("selection source hashes are incomplete")
    result = _self_hashed({
        "schema_version": SCHEMA, "artifact": "selection_manifest", "scientific_result": False,
        "kappa_grid": list(KAPPAS), "selection_rows": evidence["selection_rows"],
        "selected_kappa": evidence["selected_kappa"],
        "selection_rule": SELECTION_RULE,
        "source_hashes": dict(source_hashes),
        "protocol_sha256": file_sha256(PROTOCOL),
        "frozen_config": frozen_config(),
        "config_sha256": inst.object_sha256(frozen_config()),
        "lut_artifact_sha256": lut_artifact["artifact_sha256"],
        "instance_hashes": evidence["instance_hashes"],
        "model_lineage": evidence["model_lineage"],
        "linear_baseline_artifact": linear_artifact,
    })
    validate_selection_manifest(
        result, expected_source_hashes=source_hashes,
        expected_lut_artifact_sha256=str(lut_artifact["artifact_sha256"]),
    )
    return result


def freeze_selection(
    bundles: Mapping[str, Mapping[str, Any]],
    model_manifests: Mapping[str, Mapping[str, Any]],
    lut_artifact: Mapping[str, Any],
    source_hashes: Mapping[str, str],
) -> dict[str, Any]:
    return _recompute_selection_manifest(
        bundles, model_manifests, lut_artifact, source_hashes
    )


def evaluate_holdout_primary(
    bundles: Mapping[str, Mapping[str, Any]],
    selection_bundles: Mapping[str, Mapping[str, Any]],
    model_manifests: Mapping[str, Mapping[str, Any]],
    lut_artifact: Mapping[str, Any],
    selection_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    services = services_from_lut(lut_artifact)
    validate_holdout_lineage(
        bundles, selection_bundles, model_manifests, lut_artifact,
        selection_manifest,
    )
    kappa = float(selection_manifest["selected_kappa"])
    models = {}
    for model in MODELS:
        validate_split_bundle(
            bundles[model], model, "holdout",
            model_manifest=model_manifests[model], lut_artifact=lut_artifact,
        )
        scenarios = [scenario_from_pair(pair, model, services[model], kappa) for pair in bundles[model]["pairs"]]
        reports = [core.optimize_information_lattice(scenario) for scenario in scenarios]
        baseline_metrics, baseline_pair_ledgers = evaluate_baselines_with_ledgers(
            scenarios, selection_manifest["linear_baseline_artifact"]
        )
        primary_aggregate = core.aggregate_16_pair_reports(reports)
        capture = baselines.compute_capture(
            primary_aggregate["arms"][primary_aggregate["best_single_arm"]].miss_rate,
            primary_aggregate["arms"]["R"].miss_rate,
            {name: metric.miss_rate for name, metric in baseline_metrics.items()},
        )
        milp = [
            milpcheck.crosscheck_information_lattice(scenario, report)
            for scenario, report in zip(scenarios, reports)
        ]
        sensitivity_manifest = [
            inst.rebuild_from_pair_identity(pair["pair_identity"], services[model]["receiver_unpack"], (8, 12))
            for pair in bundles[model]["pairs"]
        ]
        sensitivity_reports = [
            core.optimize_information_lattice(scenario_from_pair(pair, model, services[model], kappa))
            for pair in sensitivity_manifest
        ]
        equal_q_manifest = [
            inst.rebuild_from_pair_identity(
                pair["pair_identity"], services[model]["receiver_unpack"], (8, 8), mode="equal_q"
            )
            for pair in bundles[model]["pairs"]
        ]
        control_reports: dict[str, list[dict[str, Any]]] = {
            "equal_q": [
                core.optimize_information_lattice(
                    scenario_from_pair(pair, model, services[model], kappa, kind="equal_q")
                ) for pair in equal_q_manifest
            ]
        }
        for name in ("equal_j", "fanout1", "no_conflict", "shuffled_key"):
            control_reports[name] = [
                core.optimize_information_lattice(control_scenario(scenario, name)) for scenario in scenarios
            ]
        controls = {}
        for name, values in control_reports.items():
            aggregate = core.aggregate_16_pair_reports(values)
            controls[name] = {
                "aggregate": aggregate,
                "validation": core.validate_control(name, aggregate),
            }
        adjacent = [value for value in KAPPAS if abs(KAPPAS.index(value) - KAPPAS.index(kappa)) == 1]
        robustness = {}
        robustness_reports = {}
        for value in adjacent:
            rows = [
                core.optimize_information_lattice(scenario_from_pair(pair, model, services[model], value))
                for pair in bundles[model]["pairs"]
            ]
            robustness[str(value)] = core.aggregate_16_pair_reports(rows)
            robustness_reports[str(value)] = rows
        models[model] = {
            "pair_reports": reports,
            "aggregate": primary_aggregate,
            "baseline_metrics": baseline_metrics,
            "baseline_pair_ledgers": baseline_pair_ledgers,
            "baseline_capture": capture,
            "depth_sensitivity_8_12": core.aggregate_16_pair_reports(sensitivity_reports),
            "depth_sensitivity_pair_reports": sensitivity_reports,
            "adjacent_kappa_robustness": robustness,
            "adjacent_kappa_pair_reports": robustness_reports,
            "controls": controls,
            "control_pair_reports": control_reports,
            "milp_crosscheck": milp,
        }
    return {"kappa": kappa, "models": models}


def decision_from_holdout(result: Mapping[str, Any]) -> dict[str, Any]:
    model_rows = {}
    for model in MODELS:
        row = result["models"][model]
        aggregate = row["aggregate"]
        capture = row["baseline_capture"]
        controls_pass = all(value["validation"]["passed"] for value in row["controls"].values())
        robustness_pass = all(
            value["absolute_miss_reduction"] >= -1e-12
            for value in row["adjacent_kappa_robustness"].values()
        )
        gates = {
            "relative_miss_reduction": aggregate["relative_miss_reduction"] is not None
            and aggregate["relative_miss_reduction"] >= 0.10,
            "absolute_miss_reduction": aggregate["absolute_miss_reduction"] >= 0.02,
            "relative_cvar90_reduction": aggregate["relative_cvar90_reduction"] is not None
            and aggregate["relative_cvar90_reduction"] >= 0.05,
            "actionable_rate": aggregate["actionable_rate"] >= 0.50,
            "strict_interaction_flip_rate": aggregate["strict_interaction_flip_rate"] >= 0.25,
            "simple_baseline_capture": capture["gate_eligible"]
            and capture["strongest_capture"] < 0.90,
            "adjacent_kappa_nonnegative": robustness_pass,
            "negative_controls": controls_pass,
            "milp_crosscheck": all(value["passed"] for value in row["milp_crosscheck"]),
        }
        adjacent_values = [
            float(value["absolute_miss_reduction"])
            for value in row["adjacent_kappa_robustness"].values()
        ]
        control_values = [value["validation"]["passed"] for value in row["controls"].values()]
        milp_values = [value["passed"] for value in row["milp_crosscheck"]]
        evidence = {
            "relative_miss_reduction": {
                "value": aggregate["relative_miss_reduction"], "comparator": ">=", "threshold": 0.10,
                "passed": gates["relative_miss_reduction"],
            },
            "absolute_miss_reduction": {
                "value": aggregate["absolute_miss_reduction"], "comparator": ">=", "threshold": 0.02,
                "passed": gates["absolute_miss_reduction"],
            },
            "relative_cvar90_reduction": {
                "value": aggregate["relative_cvar90_reduction"], "comparator": ">=", "threshold": 0.05,
                "passed": gates["relative_cvar90_reduction"],
            },
            "actionable_rate": {
                "value": aggregate["actionable_rate"], "comparator": ">=", "threshold": 0.50,
                "passed": gates["actionable_rate"],
            },
            "strict_interaction_flip_rate": {
                "value": aggregate["strict_interaction_flip_rate"], "comparator": ">=", "threshold": 0.25,
                "passed": gates["strict_interaction_flip_rate"],
            },
            "simple_baseline_capture": {
                "value": capture["strongest_capture"], "comparator": "<", "threshold": 0.90,
                "gate_eligible": capture["gate_eligible"], "passed": gates["simple_baseline_capture"],
            },
            "adjacent_kappa_nonnegative": {
                "value": min(adjacent_values) if adjacent_values else None,
                "comparator": ">=", "threshold": 0.0,
                "passed": gates["adjacent_kappa_nonnegative"],
            },
            "negative_controls": {
                "value": sum(control_values), "denominator": len(control_values),
                "comparator": "==", "threshold": len(control_values),
                "passed": gates["negative_controls"],
            },
            "milp_crosscheck": {
                "value": sum(milp_values), "denominator": len(milp_values),
                "comparator": "==", "threshold": len(milp_values),
                "passed": gates["milp_crosscheck"],
            },
        }
        model_rows[model] = {
            "gates": gates, "gate_evidence": evidence, "passed": all(gates.values())
        }
    passed = all(value["passed"] for value in model_rows.values())
    return {
        "two_model_AND": passed,
        "decision": "PROMISING_L2_QUEUE_JOIN_INTERACTION_HEADROOM" if passed
        else "NO_GO_PHASEMAP_QUEUE_JOIN_INTERACTION",
        "models": model_rows,
    }


def _raw_four_world_ledger(
    scenario: core.Scenario, report: Mapping[str, Any]
) -> dict[str, list[dict[str, Any]]]:
    ledgers: dict[str, list[dict[str, Any]]] = {}
    for arm in ("B0", "Q", "J", "R", "C"):
        arm_report = report["ceiling"] if arm == "C" else report["arms"][arm]
        selected_policy = arm_report["selected_canonical_policy"]
        policy = dict(selected_policy)
        rows = []
        for index, world in enumerate(scenario.worlds):
            observation = f"C:{world.world_id}" if arm == "C" else core.observation_key(scenario, world, arm)
            rows.append(core.simulate(scenario, index, policy[observation]))
        ledgers[arm] = rows
    return ledgers


def _holdout_rows(
    bundles: Mapping[str, Mapping[str, Any]], services: Mapping[str, Mapping[str, float]],
    holdout: Mapping[str, Any], selection: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    per_pair = []
    per_request = []
    kappa = float(selection["selected_kappa"])
    for model in MODELS:
        pairs = bundles[model]["pairs"]
        reports = holdout["models"][model]["pair_reports"]
        if len(pairs) != 16 or len(reports) != 16:
            raise PhaseMapRunnerError("holdout artifact rows require exactly 16 pairs per model")
        for ordinal, (pair, report) in enumerate(zip(pairs, reports)):
            scenario = scenario_from_pair(pair, model, services[model], kappa)
            raw = _raw_four_world_ledger(scenario, report)
            per_pair.append({
                "model": model,
                "pair_ordinal": ordinal,
                "pair_key": pair["pair_key"],
                "pair_manifest_sha256": pair["manifest_sha256"],
                "source_model_manifest_sha256": bundles[model]["source_model_manifest_sha256"],
                "request_ids": pair["request_ids"],
                "pair_identity": pair["pair_identity"],
                "reachability_certificate": pair["reachability_certificate"],
                "oracle_report": report,
                "raw_four_world_ledger": raw,
                "baseline_ledger": holdout["models"][model]["baseline_pair_ledgers"][ordinal],
                "depth_sensitivity_8_12_report": holdout["models"][model][
                    "depth_sensitivity_pair_reports"
                ][ordinal],
                "adjacent_kappa_reports": {
                    adjacent: rows[ordinal]
                    for adjacent, rows in holdout["models"][model][
                        "adjacent_kappa_pair_reports"
                    ].items()
                },
                "control_reports": {
                    name: rows[ordinal]
                    for name, rows in holdout["models"][model]["control_pair_reports"].items()
                },
                "milp_crosscheck": holdout["models"][model]["milp_crosscheck"][ordinal],
            })
            for request in pair["request_ids"]:
                arms = {}
                for arm in ("B0", "Q", "J", "R", "C"):
                    metrics = (report["ceiling"] if arm == "C" else report["arms"][arm])["metrics"]
                    misses = dict(metrics.expected_miss_by_request)
                    tardiness = dict(metrics.expected_tardiness_by_request)
                    closes = dict(metrics.expected_join_close_by_request)
                    arms[arm] = {
                        "folded_expected_miss": misses[request],
                        "folded_expected_normalized_tardiness": tardiness[request],
                        "folded_expected_join_close_us": closes[request],
                        "world_rows": [
                            request_row
                            for world_row in raw[arm]
                            for request_row in world_row["requests"]
                            if request_row["request_id"] == request
                        ],
                    }
                per_request.append({
                    "model": model,
                    "pair_key": pair["pair_key"],
                    "request_id": request,
                    "full_join_identity": pair["pair_identity"]["joins"][request]["full_join_identity"],
                    "arms": arms,
                })
    if len(per_pair) != 32 or len(per_request) != 64:
        raise PhaseMapRunnerError("published holdout denominator drift")
    return per_pair, per_request


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_bytes_no_overwrite(path: Path, encoded: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise PhaseMapRunnerError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.is_symlink():
        raise PhaseMapRunnerError("refusing to publish through a symlink directory")
    temporary = path.with_name(f".{path.name}.partial-{os.getpid()}-{time.time_ns()}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(temporary, flags, 0o444)
    try:
        offset = 0
        view = memoryview(encoded)
        while offset < len(encoded):
            written = os.write(descriptor, view[offset:])
            if written <= 0:
                raise PhaseMapRunnerError("artifact writer made no forward progress")
            offset += written
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        os.link(temporary, path, follow_symlinks=False)
        os.unlink(temporary)
        _fsync_directory(path.parent)
        if file_sha256(path) != hashlib.sha256(encoded).hexdigest():
            raise PhaseMapRunnerError("published artifact failed post-write SHA-256 verification")
    except BaseException:
        if descriptor >= 0:
            try: os.close(descriptor)
            except OSError: pass
        try: os.unlink(temporary)
        except OSError: pass
        raise


def write_json_no_overwrite(path: Path, value: Any) -> None:
    encoded = json.dumps(
        _plain(value), ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True
    ).encode() + b"\n"
    _write_bytes_no_overwrite(path, encoded)


def write_jsonl_no_overwrite(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    encoded = b"".join(
        json.dumps(
            _plain(row), ensure_ascii=False, allow_nan=False,
            separators=(",", ":"), sort_keys=True,
        ).encode() + b"\n"
        for row in rows
    )
    _write_bytes_no_overwrite(path, encoded)


def publish_holdout_artifacts(
    output_dir: Path, *, bundles: Mapping[str, Mapping[str, Any]],
    selection_bundles: Mapping[str, Mapping[str, Any]],
    model_manifests: Mapping[str, Mapping[str, Any]],
    services: Mapping[str, Mapping[str, float]], selection: Mapping[str, Any],
    lut: Mapping[str, Any], holdout: Mapping[str, Any], decision: Mapping[str, Any],
    source_hashes: Mapping[str, str],
) -> None:
    if output_dir.exists() or output_dir.is_symlink():
        raise PhaseMapRunnerError(f"refusing to overwrite historical output directory {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    os.mkdir(output_dir, 0o755)
    incomplete = output_dir / ".INCOMPLETE"
    try:
        _write_bytes_no_overwrite(incomplete, b"PhaseMap holdout publication is incomplete.\n")
        per_pair, per_request = _holdout_rows(bundles, services, holdout, selection)
        environment = _self_hashed({
            "schema_version": SCHEMA,
            "artifact": "environment",
            "scientific_result": False,
            "python": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "argv": list(sys.argv),
            "numeric_runtime": runtime_solver_provenance(),
            "git": git_provenance(),
            "lut_environment": lut.get("environment"),
        })
        holdout_instances = _self_hashed({
            "schema_version": SCHEMA,
            "artifact": "holdout_instance_manifest",
            "scientific_result": False,
            "model_manifests": {model: model_manifests[model] for model in MODELS},
            "selection_bundles": {model: selection_bundles[model] for model in MODELS},
            "holdout_bundles": {model: bundles[model] for model in MODELS},
        })
        baseline_results = _self_hashed({
            "schema_version": SCHEMA,
            "artifact": "baseline_results",
            "scientific_result": False,
            "models": {
                model: {
                    "metrics": holdout["models"][model]["baseline_metrics"],
                    "capture": holdout["models"][model]["baseline_capture"],
                    "per_pair_ledgers": holdout["models"][model]["baseline_pair_ledgers"],
                } for model in MODELS
            },
        })
        controls = _self_hashed({
            "schema_version": SCHEMA,
            "artifact": "controls",
            "scientific_result": False,
            "models": {
                model: {
                    "aggregates": holdout["models"][model]["controls"],
                    "per_pair_reports": holdout["models"][model]["control_pair_reports"],
                    "depth_sensitivity_8_12": holdout["models"][model]["depth_sensitivity_8_12"],
                    "depth_sensitivity_pair_reports": holdout["models"][model][
                        "depth_sensitivity_pair_reports"
                    ],
                    "adjacent_kappa_robustness": holdout["models"][model][
                        "adjacent_kappa_robustness"
                    ],
                    "adjacent_kappa_pair_reports": holdout["models"][model][
                        "adjacent_kappa_pair_reports"
                    ],
                } for model in MODELS
            },
        })
        milp = _self_hashed({
            "schema_version": SCHEMA,
            "artifact": "milp_crosscheck",
            "scientific_result": False,
            "claim_boundary": "INDEPENDENT_OPTIMIZER_USING_SHARED_REPLAY_COST_TABLE",
            "models": {model: holdout["models"][model]["milp_crosscheck"] for model in MODELS},
        })
        decision_artifact = _self_hashed({
            "schema_version": SCHEMA,
            "artifact": "decision",
            "scientific_result": False,
            "selection_artifact_sha256": selection["artifact_sha256"],
            "lut_artifact_sha256": lut["artifact_sha256"],
            "holdout_bundle_sha256": {
                model: bundles[model]["artifact_sha256"] for model in MODELS
            },
            "selection_bundle_sha256": {
                model: selection_bundles[model]["artifact_sha256"] for model in MODELS
            },
            "model_manifest_sha256": {
                model: model_manifests[model]["manifest_sha256"] for model in MODELS
            },
            "decision": decision,
        })
        summary = (
            "# PhaseMap-MILP holdout gate\n\n"
            "Status: executed frozen L2 oracle gate; this is not a serving or RDMA result.\n\n"
            f"Decision: `{decision['decision']}`\n\n"
            f"Two-model AND: `{decision['two_model_AND']}`\n\n"
            "The exact gate rows are in `decision.json`; raw four-world event ledgers are in "
            "`per_pair.jsonl`.\n"
        )
        write_json_no_overwrite(output_dir / "selection_manifest.json", selection)
        write_json_no_overwrite(output_dir / "holdout_instance_manifest.json", holdout_instances)
        write_json_no_overwrite(output_dir / "lut.json", lut)
        write_jsonl_no_overwrite(output_dir / "per_pair.jsonl", per_pair)
        write_jsonl_no_overwrite(output_dir / "per_request.jsonl", per_request)
        write_json_no_overwrite(output_dir / "baseline_results.json", baseline_results)
        write_json_no_overwrite(output_dir / "controls.json", controls)
        write_json_no_overwrite(output_dir / "milp_crosscheck.json", milp)
        write_json_no_overwrite(output_dir / "decision.json", decision_artifact)
        write_json_no_overwrite(output_dir / "environment.json", environment)
        _write_bytes_no_overwrite(output_dir / "summary.md", summary.encode("utf-8"))
        already_written = [name for name in REQUIRED_HOLDOUT_ARTIFACTS if name != "source_manifest.json"]
        source_manifest = _self_hashed({
            "schema_version": SCHEMA,
            "artifact": "source_manifest",
            "scientific_result": False,
            "protocol": {"path": str(PROTOCOL), "sha256": file_sha256(PROTOCOL)},
            "frozen_config": frozen_config(),
            "config_sha256": inst.object_sha256(frozen_config()),
            "source_hashes": dict(source_hashes),
            "selection_artifact_sha256": selection["artifact_sha256"],
            "lut_artifact_sha256": lut["artifact_sha256"],
            "holdout_bundle_sha256": {
                model: bundles[model]["artifact_sha256"] for model in MODELS
            },
            "selection_bundle_sha256": {
                model: selection_bundles[model]["artifact_sha256"] for model in MODELS
            },
            "model_manifest_sha256": {
                model: model_manifests[model]["manifest_sha256"] for model in MODELS
            },
            "published_file_sha256": {
                name: file_sha256(output_dir / name) for name in already_written
            },
        })
        write_json_no_overwrite(output_dir / "source_manifest.json", source_manifest)
        actual = {path.name for path in output_dir.iterdir() if path.name != ".INCOMPLETE"}
        if actual != set(REQUIRED_HOLDOUT_ARTIFACTS):
            raise PhaseMapRunnerError("published holdout artifact set is incomplete or contains drift")
        os.unlink(incomplete)
        os.chmod(output_dir, 0o555)
        _fsync_directory(output_dir)
        _fsync_directory(output_dir.parent)
    except BaseException:
        # Keep the fail-closed directory and marker for forensic inspection; never replace it.
        raise


def _load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise PhaseMapRunnerError("JSON root must be an object")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("selection", "holdout"))
    parser.add_argument("--olmoe-instances", type=Path, required=True)
    parser.add_argument("--llmjp-instances", type=Path, required=True)
    parser.add_argument("--olmoe-model-manifest", type=Path, required=True)
    parser.add_argument("--llmjp-model-manifest", type=Path, required=True)
    parser.add_argument("--olmoe-selection-instances", type=Path)
    parser.add_argument("--llmjp-selection-instances", type=Path)
    parser.add_argument("--lut", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path)
    parser.add_argument(
        "--output", type=Path, required=True,
        help="selection JSON file, or a new holdout artifact directory",
    )
    args = parser.parse_args()
    bundles = {"olmoe": _load(args.olmoe_instances), "llmjp": _load(args.llmjp_instances)}
    model_manifests = {
        "olmoe": _load(args.olmoe_model_manifest),
        "llmjp": _load(args.llmjp_model_manifest),
    }
    lut = _load(args.lut); services = services_from_lut(lut)
    if args.mode == "selection":
        if (
            args.selection_manifest is not None
            or args.olmoe_selection_instances is not None
            or args.llmjp_selection_instances is not None
        ):
            raise PhaseMapRunnerError("selection mode may not consume prior selection artifacts")
        value = freeze_selection(
            bundles, model_manifests, lut, current_source_hashes(),
        )
        write_json_no_overwrite(args.output, value)
    else:
        if (
            args.selection_manifest is None
            or args.olmoe_selection_instances is None
            or args.llmjp_selection_instances is None
        ):
            raise PhaseMapRunnerError(
                "holdout mode requires --selection-manifest and both selection bundles"
            )
        selection_bundles = {
            "olmoe": _load(args.olmoe_selection_instances),
            "llmjp": _load(args.llmjp_selection_instances),
        }
        selection = _load(args.selection_manifest)
        current_sources = current_source_hashes()
        validate_selection_manifest(
            selection,
            expected_source_hashes=current_sources,
            expected_lut_artifact_sha256=str(lut["artifact_sha256"]),
        )
        validate_holdout_lineage(
            bundles, selection_bundles, model_manifests, lut, selection
        )
        holdout = evaluate_holdout_primary(
            bundles, selection_bundles, model_manifests, lut, selection
        )
        decision = decision_from_holdout(holdout)
        publish_holdout_artifacts(
            args.output, bundles=bundles, selection_bundles=selection_bundles,
            model_manifests=model_manifests, services=services, selection=selection,
            lut=lut, holdout=holdout, decision=decision, source_hashes=current_sources,
        )


if __name__ == "__main__":
    main()
