#!/usr/bin/env python3
"""Frozen causal simple baselines and capture accounting for PhaseMap-v1."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import itertools
import json
import math
from typing import Any, Mapping, Sequence


BASELINE_SCHEMA = "phasemap-v1-causal-baselines"
LINEAR_SCHEMA = "phasemap-v1-selection-linear-baseline"
BASELINE_NAMES = (
    "request_fcfs",
    "edf",
    "qwork_first",
    "remaining_siblings_last_missing_first",
    "least_laxity",
    "lexicographic_slack_deficit_qwork",
    "separable_linear_slack_qwork_deficit",
    "myopic_predicted_join_close",
)
LINEAR_FEATURES = ("slack", "qwork", "deficit")
LINEAR_GRID_DENOMINATOR = 4
JointAction = tuple[tuple[int, str], ...]


class PhaseMapBaselineError(RuntimeError):
    """A causal-observation, frozen-fit, or capture invariant failed."""


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
        raise PhaseMapBaselineError("value is not strict canonical JSON") from exc


def object_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _finite(value: object, field: str) -> float:
    if isinstance(value, bool):
        raise PhaseMapBaselineError(f"{field} is not numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise PhaseMapBaselineError(f"{field} is not numeric") from exc
    if not math.isfinite(result):
        raise PhaseMapBaselineError(f"{field} is not finite")
    return result


def _integer(value: object, field: str) -> int:
    if type(value) is not int:
        raise PhaseMapBaselineError(f"{field} is not an integer")
    return value


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise PhaseMapBaselineError(f"{field} is not a non-empty string")
    return value


@dataclass(frozen=True)
class CausalTask:
    task_id: str
    request_id: str
    full_join_key: str
    sender_rank: int
    receiver_rank: int
    request_arrival_us: float
    deadline_us: float
    ready_us: float
    service_us: float
    receiver_service_us: float
    combine_service_us: float
    receiver_work_us: float
    receiver_availability_us: float
    remaining_siblings: int

    @property
    def sender_service_us(self) -> float:
        """Causal sender pack+cut component of the frozen per-task service."""

        return self.service_us - self.receiver_service_us

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> "CausalTask":
        allowed = {
            "task_id", "request_id", "full_join_key", "sender_rank", "receiver_rank",
            "request_arrival_us", "deadline_us", "ready_us", "service_us",
            "receiver_service_us", "combine_service_us",
            "receiver_work_us", "receiver_availability_us", "remaining_siblings",
        }
        unknown = set(row) - allowed
        if unknown:
            raise PhaseMapBaselineError(f"task observation contains forbidden fields: {sorted(unknown)}")
        task = cls(
            task_id=_text(row.get("task_id"), "task_id"),
            request_id=_text(row.get("request_id"), "request_id"),
            full_join_key=_text(row.get("full_join_key"), "full_join_key"),
            sender_rank=_integer(row.get("sender_rank"), "sender_rank"),
            receiver_rank=_integer(row.get("receiver_rank"), "receiver_rank"),
            request_arrival_us=_finite(row.get("request_arrival_us"), "request_arrival_us"),
            deadline_us=_finite(row.get("deadline_us"), "deadline_us"),
            ready_us=_finite(row.get("ready_us"), "ready_us"),
            service_us=_finite(row.get("service_us"), "service_us"),
            receiver_service_us=_finite(
                row.get("receiver_service_us"), "receiver_service_us"
            ),
            combine_service_us=_finite(
                row.get("combine_service_us"), "combine_service_us"
            ),
            receiver_work_us=_finite(row.get("receiver_work_us"), "receiver_work_us"),
            receiver_availability_us=_finite(
                row.get("receiver_availability_us"), "receiver_availability_us"
            ),
            remaining_siblings=_integer(row.get("remaining_siblings"), "remaining_siblings"),
        )
        if (
            task.sender_rank < 0
            or task.receiver_rank < 0
            or task.service_us <= 0
            or task.receiver_service_us <= 0
            or task.sender_service_us <= 0
            or task.combine_service_us <= 0
            or task.receiver_work_us < 0
            or task.receiver_availability_us < 0
            or task.remaining_siblings <= 0
            or task.deadline_us <= task.request_arrival_us
        ):
            raise PhaseMapBaselineError("causal task invariant failed")
        return task


@dataclass(frozen=True)
class DecisionObservation:
    observation_id: str
    pair_key: str
    world_id: str
    now_us: float
    tasks: tuple[CausalTask, ...]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "DecisionObservation":
        allowed = {"observation_id", "pair_key", "world_id", "now_us", "tasks"}
        unknown = set(value) - allowed
        if unknown:
            raise PhaseMapBaselineError(
                f"observation contains forbidden/noncausal fields: {sorted(unknown)}"
            )
        tasks_value = value.get("tasks")
        if not isinstance(tasks_value, Sequence) or isinstance(tasks_value, (str, bytes)):
            raise PhaseMapBaselineError("observation tasks are missing")
        tasks = tuple(
            CausalTask.from_mapping(row)
            for row in tasks_value
            if isinstance(row, Mapping)
        )
        if len(tasks) != len(tasks_value):
            raise PhaseMapBaselineError("observation contains a malformed task")
        observation = cls(
            observation_id=_text(value.get("observation_id"), "observation_id"),
            pair_key=_text(value.get("pair_key"), "pair_key"),
            world_id=_text(value.get("world_id"), "world_id"),
            now_us=_finite(value.get("now_us"), "now_us"),
            tasks=tasks,
        )
        observation.validate()
        return observation

    def validate(self) -> None:
        if len(self.tasks) != 4 or len({task.task_id for task in self.tasks}) != 4:
            raise PhaseMapBaselineError("PhaseMap decision requires four distinct ready tasks")
        senders = sorted({task.sender_rank for task in self.tasks})
        requests = sorted({task.request_id for task in self.tasks})
        receivers = sorted({task.receiver_rank for task in self.tasks})
        if len(senders) != 2 or len(requests) != 2 or len(receivers) != 2:
            raise PhaseMapBaselineError(
                "PhaseMap decision requires two senders, requests, and output receivers"
            )
        for sender in senders:
            sender_tasks = [task for task in self.tasks if task.sender_rank == sender]
            if {task.request_id for task in sender_tasks} != set(requests) or len(sender_tasks) != 2:
                raise PhaseMapBaselineError("each sender must expose one task per request")
        if any(task.ready_us > self.now_us for task in self.tasks):
            raise PhaseMapBaselineError("baseline observation includes a future-unready task")
        for request in requests:
            rows = [task for task in self.tasks if task.request_id == request]
            signatures = {
                (
                    task.full_join_key,
                    task.receiver_rank,
                    task.request_arrival_us,
                    task.deadline_us,
                    task.receiver_work_us,
                    task.receiver_availability_us,
                    task.remaining_siblings,
                    task.receiver_service_us,
                    task.combine_service_us,
                )
                for task in rows
            }
            if len(signatures) != 1:
                raise PhaseMapBaselineError("request-level causal state disagrees across senders")


def coerce_observation(value: DecisionObservation | Mapping[str, Any]) -> DecisionObservation:
    if isinstance(value, DecisionObservation):
        value.validate()
        return value
    if not isinstance(value, Mapping):
        raise PhaseMapBaselineError("baseline observation is malformed")
    return DecisionObservation.from_mapping(value)


def enumerate_joint_actions(value: DecisionObservation | Mapping[str, Any]) -> tuple[JointAction, ...]:
    observation = coerce_observation(value)
    by_sender: dict[int, list[str]] = {}
    for sender in sorted({task.sender_rank for task in observation.tasks}):
        by_sender[sender] = sorted(
            task.task_id for task in observation.tasks if task.sender_rank == sender
        )
    return tuple(
        tuple((sender, task_id) for sender, task_id in zip(sorted(by_sender), choice))
        for choice in itertools.product(*(by_sender[sender] for sender in sorted(by_sender)))
    )


def action_key(action: JointAction) -> str:
    normalized = tuple(sorted((int(sender), str(task)) for sender, task in action))
    if normalized != action or len(normalized) != 2:
        raise PhaseMapBaselineError("joint action is not canonical")
    return object_sha256(["phasemap-v1-joint-action", normalized])


def _normalized(values: Mapping[str, float]) -> dict[str, float]:
    low, high = min(values.values()), max(values.values())
    if high <= low:
        return {key: 0.0 for key in values}
    return {key: (value - low) / (high - low) for key, value in values.items()}


def _linear_weights(artifact: Mapping[str, Any]) -> dict[str, float]:
    validate_linear_artifact(artifact)
    return {name: float(artifact["weights"][name]) for name in LINEAR_FEATURES}


def _task_keys(
    observation: DecisionObservation,
    baseline_name: str,
    linear_artifact: Mapping[str, Any] | None,
) -> dict[str, tuple[Any, ...]]:
    slack = {task.task_id: task.deadline_us - observation.now_us for task in observation.tasks}
    if baseline_name == "request_fcfs":
        return {
            task.task_id: (task.request_arrival_us, task.request_id, task.task_id)
            for task in observation.tasks
        }
    if baseline_name == "edf":
        return {task.task_id: (task.deadline_us, task.task_id) for task in observation.tasks}
    if baseline_name == "qwork_first":
        return {
            task.task_id: (
                task.receiver_work_us,
                task.receiver_availability_us,
                task.task_id,
            )
            for task in observation.tasks
        }
    if baseline_name == "remaining_siblings_last_missing_first":
        return {
            task.task_id: (task.remaining_siblings, task.task_id)
            for task in observation.tasks
        }
    if baseline_name == "least_laxity":
        remaining = _causal_remaining_work_by_request(observation)
        return {
            task.task_id: (slack[task.task_id] - remaining[task.request_id], task.task_id)
            for task in observation.tasks
        }
    if baseline_name == "lexicographic_slack_deficit_qwork":
        return {
            task.task_id: (
                slack[task.task_id],
                task.remaining_siblings,
                task.receiver_work_us,
                task.task_id,
            )
            for task in observation.tasks
        }
    if baseline_name == "separable_linear_slack_qwork_deficit":
        if linear_artifact is None:
            raise PhaseMapBaselineError("separable linear baseline lacks frozen selection artifact")
        weights = _linear_weights(linear_artifact)
        normalized_slack = _normalized(slack)
        normalized_qwork = _normalized(
            {task.task_id: task.receiver_work_us for task in observation.tasks}
        )
        normalized_deficit = _normalized(
            {task.task_id: float(task.remaining_siblings) for task in observation.tasks}
        )
        return {
            task.task_id: (
                weights["slack"] * normalized_slack[task.task_id]
                + weights["qwork"] * normalized_qwork[task.task_id]
                + weights["deficit"] * normalized_deficit[task.task_id],
                task.task_id,
            )
            for task in observation.tasks
        }
    raise PhaseMapBaselineError(f"unknown baseline: {baseline_name}")


def _causal_remaining_work_by_request(
    observation: DecisionObservation,
) -> dict[str, float]:
    """Recompute least-laxity work from primitive pre-t0 state.

    ``receiver_availability_us`` already closes all work queued at the receiver,
    including foreground phase carriers.  Only the two ready decision
    contributions and the once-per-join combine are added here; queued receiver
    work is therefore never counted again through the join deficit.
    """

    result: dict[str, float] = {}
    for request in sorted({task.request_id for task in observation.tasks}):
        rows = [task for task in observation.tasks if task.request_id == request]
        receiver_wait = max(0.0, rows[0].receiver_availability_us - observation.now_us)
        sender_tail = max(task.sender_service_us for task in rows)
        receiver_decisions = sum(task.receiver_service_us for task in rows)
        result[request] = (
            receiver_wait + sender_tail + receiver_decisions + rows[0].combine_service_us
        )
    return result


def _myopic_action_objective(
    observation: DecisionObservation,
    action: JointAction,
) -> tuple[float, float, float, tuple[tuple[int, str], ...]]:
    """Causally replay one complete joint action in the lightweight predictor."""

    tasks = {task.task_id: task for task in observation.tasks}
    first_by_sender = {sender: tasks[task_id] for sender, task_id in action}
    arrivals: list[tuple[float, str, CausalTask]] = []
    for sender in sorted(first_by_sender):
        sender_rows = sorted(
            (task for task in observation.tasks if task.sender_rank == sender),
            key=lambda task: task.task_id,
        )
        first = first_by_sender[sender]
        second = next(task for task in sender_rows if task.task_id != first.task_id)
        available = observation.now_us
        for task in (first, second):
            available = max(available, task.ready_us) + task.sender_service_us
            arrivals.append((available, task.task_id, task))

    completion_by_task: dict[str, float] = {}
    for receiver in sorted({task.receiver_rank for task in observation.tasks}):
        receiver_rows = [item for item in arrivals if item[2].receiver_rank == receiver]
        base = next(task.receiver_availability_us for _arrival, _key, task in receiver_rows)
        available = max(observation.now_us, base)
        for arrival, task_id, task in sorted(receiver_rows, key=lambda item: (item[0], item[1])):
            available = max(available, arrival) + task.receiver_service_us
            completion_by_task[task_id] = available

    miss = 0.0
    tardiness = 0.0
    close_sum = 0.0
    for request in sorted({task.request_id for task in observation.tasks}):
        rows = [task for task in observation.tasks if task.request_id == request]
        close = max(completion_by_task[task.task_id] for task in rows) + rows[0].combine_service_us
        miss += float(close > rows[0].deadline_us)
        tardiness += max(0.0, close - rows[0].deadline_us) / (
            rows[0].deadline_us - rows[0].request_arrival_us
        )
        close_sum += close
    canonical_identity = tuple(
        sorted((int(sender), str(task_id)) for sender, task_id in action)
    )
    return miss, tardiness, close_sum, canonical_identity


def baseline_action(
    value: DecisionObservation | Mapping[str, Any],
    baseline_name: str,
    linear_artifact: Mapping[str, Any] | None = None,
) -> JointAction:
    observation = coerce_observation(value)
    if baseline_name not in BASELINE_NAMES:
        raise PhaseMapBaselineError(f"unknown baseline: {baseline_name}")
    if baseline_name == "myopic_predicted_join_close":
        return min(
            enumerate_joint_actions(observation),
            key=lambda action: _myopic_action_objective(observation, action),
        )
    task_keys = _task_keys(observation, baseline_name, linear_artifact)
    action = tuple(
        (
            sender,
            min(
                (task for task in observation.tasks if task.sender_rank == sender),
                key=lambda task: task_keys[task.task_id],
            ).task_id,
        )
        for sender in sorted({task.sender_rank for task in observation.tasks})
    )
    if action not in enumerate_joint_actions(observation):
        raise PhaseMapBaselineError("baseline produced an action outside the frozen domain")
    return action


def run_all_baselines(
    value: DecisionObservation | Mapping[str, Any],
    linear_artifact: Mapping[str, Any],
) -> dict[str, JointAction]:
    observation = coerce_observation(value)
    validate_linear_artifact(linear_artifact)
    return {
        name: baseline_action(observation, name, linear_artifact)
        for name in BASELINE_NAMES
    }


def _linear_grid() -> tuple[dict[str, float], ...]:
    denominator = LINEAR_GRID_DENOMINATOR
    return tuple(
        {
            "slack": slack / denominator,
            "qwork": qwork / denominator,
            "deficit": deficit / denominator,
        }
        for slack in range(denominator + 1)
        for qwork in range(denominator + 1 - slack)
        for deficit in (denominator - slack - qwork,)
    )


def _objective_tuple(value: object) -> tuple[float, float, float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 3:
        raise PhaseMapBaselineError("selection action objective must have three stages")
    result = tuple(_finite(item, "selection objective") for item in value)
    if any(item < 0 for item in result):
        raise PhaseMapBaselineError("selection objective cannot be negative")
    return result  # type: ignore[return-value]


def fit_separable_linear(
    selection_examples: Sequence[Mapping[str, Any]],
    *,
    selection_source_sha256: str,
    split: str = "selection",
) -> dict[str, Any]:
    """Fit the frozen finite-grid separable score on selection examples only."""

    if split != "selection":
        raise PhaseMapBaselineError("linear weights may only be fitted on selection")
    if (
        not isinstance(selection_source_sha256, str)
        or len(selection_source_sha256) != 64
        or any(character not in "0123456789abcdef" for character in selection_source_sha256)
    ):
        raise PhaseMapBaselineError("selection source SHA-256 is malformed")
    if not selection_examples:
        raise PhaseMapBaselineError("selection fit has no examples")
    parsed = []
    normalized_examples = []
    seen_ids = set()
    for example in selection_examples:
        if set(example) != {"observation", "action_objectives"}:
            raise PhaseMapBaselineError("selection example schema drift")
        observation = coerce_observation(example["observation"])
        if observation.observation_id in seen_ids:
            raise PhaseMapBaselineError("duplicate selection observation")
        seen_ids.add(observation.observation_id)
        objectives_value = example["action_objectives"]
        if not isinstance(objectives_value, Mapping):
            raise PhaseMapBaselineError("selection action objective map is malformed")
        expected_keys = {action_key(action) for action in enumerate_joint_actions(observation)}
        if set(objectives_value) != expected_keys:
            raise PhaseMapBaselineError("selection objectives do not cover the frozen action domain")
        objectives = {key: _objective_tuple(value) for key, value in objectives_value.items()}
        parsed.append((observation, objectives))
        normalized_examples.append(
            {
                "observation": {
                    "observation_id": observation.observation_id,
                    "pair_key": observation.pair_key,
                    "world_id": observation.world_id,
                    "now_us": observation.now_us,
                    "tasks": [
                        {
                            field: getattr(task, field)
                            for field in CausalTask.__dataclass_fields__
                        }
                        for task in sorted(observation.tasks, key=lambda row: row.task_id)
                    ],
                },
                "action_objectives": {
                    key: list(objectives[key]) for key in sorted(objectives)
                },
            }
        )
    normalized_examples.sort(
        key=lambda example: str(example["observation"]["observation_id"])
    )
    selection_examples_sha256 = object_sha256(normalized_examples)
    best_key: tuple[float, float, float, tuple[float, float, float]] | None = None
    best_weights: dict[str, float] | None = None
    for weights in _linear_grid():
        temporary_artifact = {
            "schema_version": LINEAR_SCHEMA,
            "split": "selection",
            "weights": weights,
            "selection_source_sha256": selection_source_sha256,
            "grid_sha256": object_sha256(_linear_grid()),
            "grid_denominator": LINEAR_GRID_DENOMINATOR,
            "example_count": len(parsed),
            "selection_examples_sha256": selection_examples_sha256,
            "selection_observation_ids_sha256": object_sha256(sorted(seen_ids)),
            "selection_objective_sum": [0.0, 0.0, 0.0],
        }
        temporary_artifact["manifest_sha256"] = object_sha256(temporary_artifact)
        total = [0.0, 0.0, 0.0]
        for observation, objectives in parsed:
            selected = baseline_action(
                observation,
                "separable_linear_slack_qwork_deficit",
                temporary_artifact,
            )
            objective = objectives[action_key(selected)]
            total = [left + right for left, right in zip(total, objective)]
        weight_tuple = tuple(weights[name] for name in LINEAR_FEATURES)
        key = (total[0], total[1], total[2], weight_tuple)
        if best_key is None or key < best_key:
            best_key, best_weights = key, weights
    if best_key is None or best_weights is None:
        raise PhaseMapBaselineError("selection fit produced no weight vector")
    payload = {
        "schema_version": LINEAR_SCHEMA,
        "split": "selection",
        "weights": best_weights,
        "selection_source_sha256": selection_source_sha256,
        "grid_sha256": object_sha256(_linear_grid()),
        "grid_denominator": LINEAR_GRID_DENOMINATOR,
        "example_count": len(parsed),
        "selection_examples_sha256": selection_examples_sha256,
        "selection_observation_ids_sha256": object_sha256(sorted(seen_ids)),
        "selection_objective_sum": list(best_key[:3]),
    }
    artifact = {**payload, "manifest_sha256": object_sha256(payload)}
    validate_linear_artifact(artifact)
    return artifact


def validate_linear_artifact(artifact: Mapping[str, Any]) -> None:
    required = {
        "schema_version", "split", "weights", "selection_source_sha256",
        "grid_sha256", "grid_denominator", "example_count",
        "selection_examples_sha256", "selection_observation_ids_sha256",
        "selection_objective_sum", "manifest_sha256",
    }
    if not required <= set(artifact):
        raise PhaseMapBaselineError("frozen linear artifact is incomplete")
    payload = dict(artifact)
    recorded = payload.pop("manifest_sha256")
    if recorded != object_sha256(payload):
        raise PhaseMapBaselineError("frozen linear artifact self-hash mismatch")
    if artifact["schema_version"] != LINEAR_SCHEMA or artifact["split"] != "selection":
        raise PhaseMapBaselineError("linear artifact was not frozen from selection")
    weights = artifact["weights"]
    if not isinstance(weights, Mapping) or set(weights) != set(LINEAR_FEATURES):
        raise PhaseMapBaselineError("linear weight schema drift")
    values = [_finite(weights[name], f"weight {name}") for name in LINEAR_FEATURES]
    if any(value < 0 for value in values) or abs(sum(values) - 1.0) > 1e-12:
        raise PhaseMapBaselineError("linear weights are not a nonnegative simplex")
    if artifact["grid_sha256"] != object_sha256(_linear_grid()):
        raise PhaseMapBaselineError("linear candidate grid identity drift")
    if artifact["grid_denominator"] != LINEAR_GRID_DENOMINATOR:
        raise PhaseMapBaselineError("linear candidate grid denominator drift")
    for field in (
        "selection_source_sha256",
        "selection_examples_sha256",
        "selection_observation_ids_sha256",
    ):
        value = artifact[field]
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise PhaseMapBaselineError(f"{field} is malformed")
    _objective_tuple(artifact["selection_objective_sum"])


def validate_linear_fit_against_examples(
    artifact: Mapping[str, Any],
    selection_examples: Sequence[Mapping[str, Any]],
    *,
    selection_source_sha256: str,
) -> None:
    """Replay the frozen grid against exact selection evidence without fitting.

    This integrity check deliberately does not call ``fit_separable_linear`` and
    never returns replacement weights.  It proves that the already-frozen
    artifact is still the lexicographic optimum of the complete frozen grid.
    """

    validate_linear_artifact(artifact)
    if artifact.get("selection_source_sha256") != selection_source_sha256:
        raise PhaseMapBaselineError("linear artifact selection source mismatch")
    if not selection_examples:
        raise PhaseMapBaselineError("selection validation has no examples")
    parsed = []
    normalized_examples = []
    seen_ids = set()
    for example in selection_examples:
        if set(example) != {"observation", "action_objectives"}:
            raise PhaseMapBaselineError("selection example schema drift")
        observation = coerce_observation(example["observation"])
        if observation.observation_id in seen_ids:
            raise PhaseMapBaselineError("duplicate selection observation")
        seen_ids.add(observation.observation_id)
        objectives_value = example["action_objectives"]
        if not isinstance(objectives_value, Mapping):
            raise PhaseMapBaselineError("selection action objective map is malformed")
        expected_keys = {action_key(action) for action in enumerate_joint_actions(observation)}
        if set(objectives_value) != expected_keys:
            raise PhaseMapBaselineError("selection objectives do not cover the frozen action domain")
        objectives = {key: _objective_tuple(value) for key, value in objectives_value.items()}
        parsed.append((observation, objectives))
        normalized_examples.append({
            "observation": {
                "observation_id": observation.observation_id,
                "pair_key": observation.pair_key,
                "world_id": observation.world_id,
                "now_us": observation.now_us,
                "tasks": [
                    {field: getattr(task, field) for field in CausalTask.__dataclass_fields__}
                    for task in sorted(observation.tasks, key=lambda row: row.task_id)
                ],
            },
            "action_objectives": {
                key: list(objectives[key]) for key in sorted(objectives)
            },
        })
    normalized_examples.sort(key=lambda example: str(example["observation"]["observation_id"]))
    examples_sha = object_sha256(normalized_examples)
    observation_ids_sha = object_sha256(sorted(seen_ids))
    if (
        artifact.get("example_count") != len(parsed)
        or artifact.get("selection_examples_sha256") != examples_sha
        or artifact.get("selection_observation_ids_sha256") != observation_ids_sha
    ):
        raise PhaseMapBaselineError("linear artifact selection evidence mismatch")

    best_key: tuple[float, float, float, tuple[float, float, float]] | None = None
    best_weights: Mapping[str, float] | None = None
    for weights in _linear_grid():
        temporary_payload = {
            "schema_version": LINEAR_SCHEMA,
            "split": "selection",
            "weights": weights,
            "selection_source_sha256": selection_source_sha256,
            "grid_sha256": object_sha256(_linear_grid()),
            "grid_denominator": LINEAR_GRID_DENOMINATOR,
            "example_count": len(parsed),
            "selection_examples_sha256": examples_sha,
            "selection_observation_ids_sha256": observation_ids_sha,
            "selection_objective_sum": [0.0, 0.0, 0.0],
        }
        temporary_artifact = {
            **temporary_payload,
            "manifest_sha256": object_sha256(temporary_payload),
        }
        total = [0.0, 0.0, 0.0]
        for observation, objectives in parsed:
            selected = baseline_action(
                observation, "separable_linear_slack_qwork_deficit", temporary_artifact
            )
            objective = objectives[action_key(selected)]
            total = [left + right for left, right in zip(total, objective)]
        weight_tuple = tuple(weights[name] for name in LINEAR_FEATURES)
        key = (total[0], total[1], total[2], weight_tuple)
        if best_key is None or key < best_key:
            best_key, best_weights = key, weights
    if best_key is None or best_weights is None:
        raise PhaseMapBaselineError("selection validation produced no grid optimum")
    if (
        dict(artifact["weights"]) != dict(best_weights)
        or list(artifact["selection_objective_sum"]) != list(best_key[:3])
    ):
        raise PhaseMapBaselineError("frozen linear artifact is not the exact selection optimum")


def compute_capture(
    best_single_miss: float,
    r_miss: float,
    baseline_miss: Mapping[str, float],
) -> dict[str, Any]:
    """Compute frozen best_single->R capture, preserving raw negative values."""

    best = _finite(best_single_miss, "best_single_miss")
    exact_r = _finite(r_miss, "r_miss")
    if not 0 <= best <= 1 or not 0 <= exact_r <= 1:
        raise PhaseMapBaselineError("miss rate is outside [0,1]")
    denominator = best - exact_r
    if denominator <= 0:
        return {
            "gate_eligible": False,
            "status": "FAILED_NONPOSITIVE_BEST_SINGLE_TO_R_GAIN",
            "denominator": denominator,
            "per_baseline": {},
            "strongest_baseline": None,
            "strongest_capture": None,
        }
    rows = {}
    for name in sorted(baseline_miss):
        if name not in BASELINE_NAMES:
            raise PhaseMapBaselineError(f"capture contains unknown baseline: {name}")
        miss = _finite(baseline_miss[name], f"{name} miss")
        if not 0 <= miss <= 1:
            raise PhaseMapBaselineError("baseline miss rate is outside [0,1]")
        if miss < exact_r - 1e-12:
            raise PhaseMapBaselineError("simple baseline beats the exact R oracle")
        raw = (best - miss) / denominator
        rows[name] = {"miss_rate": miss, "raw_capture": raw, "capture": max(0.0, raw)}
    if set(rows) != set(BASELINE_NAMES):
        raise PhaseMapBaselineError("capture requires all and only the eight frozen baselines")
    strongest = min(rows, key=lambda name: (-rows[name]["capture"], name))
    return {
        "gate_eligible": True,
        "status": "CAPTURE_COMPUTED",
        "denominator": denominator,
        "per_baseline": rows,
        "strongest_baseline": strongest,
        "strongest_capture": rows[strongest]["capture"],
        "capture_lt_90_percent": rows[strongest]["capture"] < 0.90,
    }
