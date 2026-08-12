#!/usr/bin/env python3
"""Run the lightweight M0--M4 Route Capacity Envelope comparison.

M0--M3 use only fields available by the end of window t.  M4 alone reads the
observed route of t+1 and is reported only as a future-route latency
diagnostic.  With two episodes, evaluation is bidirectional leave-one-episode-
out; adjacent windows are never randomly shuffled.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


class AnalysisError(RuntimeError):
    pass


RAW_WORKLOAD_FEATURES = (
    "custom_waiting_count",
    "running_sequences",
    "arrived_active_sequences",
    "active_tokens",
    "batch_tokens",
    "mean_logical_kv",
    "max_logical_kv",
    "mean_physical_kv",
    "max_physical_kv",
    "left_padding_ratio",
    "arrival_count",
    "arrival_rate_per_s",
    "decode_step",
)

DERIVED_WORKLOAD_FEATURES = (
    "recent_step_service_ms",
    "recent_throughput_tokens_s",
)

WORKLOAD_FEATURES = RAW_WORKLOAD_FEATURES + DERIVED_WORKLOAD_FEATURES

CURRENT_EXPERT_LOAD_FEATURES = (
    "route_max_expert_load",
    "route_max_mean",
    "route_cv",
    "route_hhi",
    "active_experts",
    "top1_share",
)

HISTORICAL_ROUTE_FEATURES = (
    "hotspot_persistence",
    "cross_layer_max_pressure",
    "cross_layer_mean_pressure",
    "route_shape_ewma",
    "route_shape_delta",
    "expert_identity_turnover",
    "top1_share_persistence",
)

METHODS = {
    "M0_constant": (),
    "M1_workload_only": WORKLOAD_FEATURES,
    "M2_workload_plus_expert_load": WORKLOAD_FEATURES + CURRENT_EXPERT_LOAD_FEATURES,
    "M3_workload_expert_load_plus_historical_route": (
        WORKLOAD_FEATURES + CURRENT_EXPERT_LOAD_FEATURES + HISTORICAL_ROUTE_FEATURES
    ),
    "M4_future_route_latency_diagnostic": (
        WORKLOAD_FEATURES
        + CURRENT_EXPERT_LOAD_FEATURES
        + HISTORICAL_ROUTE_FEATURES
        + tuple(
            f"future_{name}"
            for name in CURRENT_EXPERT_LOAD_FEATURES + HISTORICAL_ROUTE_FEATURES
        )
    ),
}

REQUIRED_COLUMNS = {
    "model",
    "model_revision",
    "episode_id",
    "arrival_regime",
    "window_id",
    "window_start_us",
    "window_end_us",
    "feature_available_at_us",
    "request_ids",
    "document_ids",
    "step_service_ms",
    "serial_route_conformance",
    "serial_route_identity_match_fraction",
    "batch_dependent_route_observed",
    *RAW_WORKLOAD_FEATURES,
    *CURRENT_EXPERT_LOAD_FEATURES,
    *HISTORICAL_ROUTE_FEATURES,
}


def _float(row: Mapping[str, str], key: str) -> float:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise AnalysisError(f"invalid numeric field {key}") from exc
    if not math.isfinite(value):
        raise AnalysisError(f"non-finite numeric field {key}")
    return value


def _identities(row: Mapping[str, str], key: str) -> tuple[str, ...]:
    try:
        values = json.loads(row[key])
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise AnalysisError(f"{key} must be a JSON list") from exc
    if (
        not isinstance(values, list)
        or not values
        or any(not isinstance(value, str) or not value for value in values)
    ):
        raise AnalysisError(f"{key} must contain non-empty string identities")
    return tuple(sorted(set(values)))


def read_windows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or ())
        if missing:
            raise AnalysisError(f"windows.csv lacks fields: {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise AnalysisError("windows.csv is empty")
    return rows


def align_next_window(rows: Sequence[Mapping[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, str]]] = {}
    identities: set[tuple[str, str]] = set()
    for row in rows:
        identity = (str(row["episode_id"]), str(row["window_id"]))
        if identity in identities:
            raise AnalysisError(f"duplicate window identity {identity}")
        identities.add(identity)
        grouped.setdefault(str(row["episode_id"]), []).append(row)
    if len(grouped) < 2:
        raise AnalysisError("lightweight comparison requires at least two complete episodes")

    aligned: list[dict[str, Any]] = []
    for episode_id, values in sorted(grouped.items()):
        ordered = sorted(values, key=lambda row: (_float(row, "window_start_us"), row["window_id"]))
        for current, future in zip(ordered, ordered[1:]):
            current_end = _float(current, "window_end_us")
            future_start = _float(future, "window_start_us")
            if _float(current, "feature_available_at_us") > current_end:
                raise AnalysisError("M0-M3 feature becomes available after window t")
            if current_end > future_start:
                raise AnalysisError("window t overlaps t+1")
            for invariant in ("model", "model_revision", "episode_id", "arrival_regime"):
                if current[invariant] != future[invariant]:
                    raise AnalysisError(f"{invariant} changes inside an episode")
            step_ms = _float(current, "step_service_ms")
            completed = _float(current, "active_tokens")
            record: dict[str, Any] = {
                "model": str(current["model"]),
                "model_revision": str(current["model_revision"]),
                "episode_id": episode_id,
                "arrival_regime": str(current["arrival_regime"]),
                "window_id": str(current["window_id"]),
                "target_window_id": str(future["window_id"]),
                "request_ids": tuple(
                    sorted(set(_identities(current, "request_ids")) | set(_identities(future, "request_ids")))
                ),
                "document_ids": tuple(
                    sorted(set(_identities(current, "document_ids")) | set(_identities(future, "document_ids")))
                ),
                "target_step_service_ms": _float(future, "step_service_ms"),
                "recent_step_service_ms": step_ms,
                "recent_throughput_tokens_s": completed * 1000.0 / step_ms,
                "serial_route_conformance": str(current["serial_route_conformance"]),
                "serial_route_identity_match_fraction": _float(
                    current, "serial_route_identity_match_fraction"
                ),
                "batch_dependent_route_observed": str(
                    current["batch_dependent_route_observed"]
                ).lower()
                == "true",
            }
            for name in RAW_WORKLOAD_FEATURES + CURRENT_EXPERT_LOAD_FEATURES + HISTORICAL_ROUTE_FEATURES:
                record[name] = _float(current, name)
            for name in CURRENT_EXPERT_LOAD_FEATURES + HISTORICAL_ROUTE_FEATURES:
                record[f"future_{name}"] = _float(future, name)
            aligned.append(record)
    if not aligned:
        raise AnalysisError("episodes do not contain adjacent windows")
    return aligned


def validate_episode_split(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    owners: dict[str, dict[str, set[str]]] = {
        "request": {},
        "document": {},
    }
    for row in rows:
        episode = str(row["episode_id"])
        for kind, field in (("request", "request_ids"), ("document", "document_ids")):
            for identity in row[field]:
                owners[kind].setdefault(str(identity), set()).add(episode)
    overlap = {
        kind: {identity: sorted(groups) for identity, groups in values.items() if len(groups) > 1}
        for kind, values in owners.items()
    }
    if overlap["request"] or overlap["document"]:
        raise AnalysisError("request/document identity overlaps across episode splits")
    return {
        "episodes": sorted({str(row["episode_id"]) for row in rows}),
        "request_overlap": 0,
        "document_overlap": 0,
    }


def _matrix(rows: Sequence[Mapping[str, Any]], features: Sequence[str]) -> np.ndarray:
    if not features:
        return np.empty((len(rows), 0), dtype=float)
    return np.asarray([[float(row[name]) for name in features] for row in rows], dtype=float)


def _targets(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    return np.asarray([float(row["target_step_service_ms"]) for row in rows], dtype=float)


def _standardize(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if train.shape[1] == 0:
        return train, test
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    scale[scale < 1e-12] = 1.0
    return (train - mean) / scale, (test - mean) / scale


def _higher_quantile(values: np.ndarray, quantile: float) -> float:
    return float(np.quantile(values, quantile, method="higher"))


def fit_quantile_linear(
    x: np.ndarray,
    y: np.ndarray,
    quantile: float,
    l2_alpha: float = 0.0,
) -> np.ndarray:
    """Fit a deterministic NumPy-only linear quantile model.

    A linear quantile-regression optimum has a basic solution whose residual is
    zero on a full-rank set of design rows. Small unregularized problems use
    exact basic-solution enumeration. The actual pilot uses one frozen L2
    value and a convex ADMM solver so feature-rich M2--M4 cannot interpolate a
    single training episode merely because columns outnumber windows.
    """

    if l2_alpha < 0:
        raise AnalysisError("quantile L2 alpha must be non-negative")
    if x.shape[1] == 0:
        return np.asarray([_higher_quantile(y, quantile)])
    design = np.column_stack((np.ones(len(x)), x))
    sample_count, parameter_count = design.shape

    if l2_alpha > 0:
        rho = 1.0
        penalty = np.eye(parameter_count) * l2_alpha
        penalty[0, 0] = 0.0
        system = rho * (design.T @ design) + penalty
        coefficients = np.linalg.solve(system, rho * design.T @ y)
        residual_variable = y - design @ coefficients
        dual = np.zeros(sample_count, dtype=float)
        tolerance = 1e-9 * math.sqrt(sample_count)
        for _ in range(50_000):
            target = y - residual_variable + dual
            coefficients = np.linalg.solve(
                system, rho * design.T @ target
            )
            shifted = y - design @ coefficients + dual
            previous_residual = residual_variable
            residual_variable = np.where(
                shifted > quantile / rho,
                shifted - quantile / rho,
                np.where(
                    shifted < -(1.0 - quantile) / rho,
                    shifted + (1.0 - quantile) / rho,
                    0.0,
                ),
            )
            primal = y - design @ coefficients - residual_variable
            dual = dual + primal
            if (
                float(np.linalg.norm(primal)) <= tolerance
                and float(
                    rho
                    * np.linalg.norm(
                        design.T @ (residual_variable - previous_residual)
                    )
                )
                <= tolerance
            ):
                break
        else:
            raise AnalysisError("regularized P95 quantile ADMM did not converge")
        if not bool(np.isfinite(coefficients).all()):
            raise AnalysisError("P95 quantile regression produced non-finite coefficients")
        return coefficients

    basis_columns: list[int] = []
    rank = 0
    for column in range(parameter_count):
        candidate_rank = int(
            np.linalg.matrix_rank(design[:, basis_columns + [column]])
        )
        if candidate_rank > rank:
            basis_columns.append(column)
            rank = candidate_rank
    if rank == 0:
        raise AnalysisError("quantile design has zero rank")

    combination_count = math.comb(sample_count, rank)
    if combination_count > 200_000:
        raise AnalysisError(
            "frozen pilot quantile fit exceeds the exact enumeration budget"
        )

    reduced = design[:, basis_columns]
    best_loss = math.inf
    best_reduced: np.ndarray | None = None
    for row_indices in itertools.combinations(range(sample_count), rank):
        indices = np.asarray(row_indices)
        square = reduced[indices, :]
        if int(np.linalg.matrix_rank(square)) != rank:
            continue
        coefficients = np.linalg.solve(square, y[indices])
        residual = y - reduced @ coefficients
        loss = float(
            np.where(
                residual >= 0,
                quantile * residual,
                (1.0 - quantile) * -residual,
            ).sum()
        )
        if loss < best_loss:
            best_loss = loss
            best_reduced = coefficients

    if best_reduced is None:
        raise AnalysisError("P95 quantile regression found no full-rank basis")
    result = np.zeros(parameter_count, dtype=float)
    result[np.asarray(basis_columns)] = best_reduced
    return result


def predict_linear(coefficients: np.ndarray, x: np.ndarray) -> np.ndarray:
    if len(coefficients) == 1 and x.shape[1] == 0:
        return np.full(len(x), coefficients[0])
    return np.column_stack((np.ones(len(x)), x)) @ coefficients


def fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    design = np.column_stack((np.ones(len(x)), x))
    penalty = np.eye(design.shape[1]) * alpha
    penalty[0, 0] = 0.0
    return np.linalg.lstsq(design.T @ design + penalty, design.T @ y, rcond=None)[0]


@dataclass
class TreeNode:
    value: float
    feature: int | None = None
    threshold: float | None = None
    left: "TreeNode | None" = None
    right: "TreeNode | None" = None


def fit_tree(x: np.ndarray, y: np.ndarray, depth: int = 0, max_depth: int = 4) -> TreeNode:
    node = TreeNode(float(y.mean()))
    if depth >= max_depth or len(y) < 8 or x.shape[1] == 0:
        return node
    parent_loss = float(((y - y.mean()) ** 2).sum())
    best: tuple[float, int, float, np.ndarray] | None = None
    for feature in range(x.shape[1]):
        values = np.unique(x[:, feature])
        if len(values) < 2:
            continue
        thresholds = (values[:-1] + values[1:]) / 2.0
        if len(thresholds) > 16:
            thresholds = np.quantile(thresholds, np.linspace(0.05, 0.95, 16))
        for threshold in thresholds:
            left = x[:, feature] <= threshold
            if left.sum() < 3 or (~left).sum() < 3:
                continue
            loss = float(((y[left] - y[left].mean()) ** 2).sum() + ((y[~left] - y[~left].mean()) ** 2).sum())
            gain = parent_loss - loss
            if best is None or gain > best[0]:
                best = (gain, feature, float(threshold), left)
    if best is None or best[0] <= 1e-12:
        return node
    _, feature, threshold, left = best
    node.feature = feature
    node.threshold = threshold
    node.left = fit_tree(x[left], y[left], depth + 1, max_depth)
    node.right = fit_tree(x[~left], y[~left], depth + 1, max_depth)
    return node


def predict_tree(node: TreeNode, x: np.ndarray) -> np.ndarray:
    values: list[float] = []
    for row in x:
        current = node
        while current.feature is not None:
            branch = current.left if row[current.feature] <= float(current.threshold) else current.right
            if branch is None:
                break
            current = branch
        values.append(current.value)
    return np.asarray(values)


def metric(y: np.ndarray, prediction: np.ndarray, quantile: float, margin: float, slo: np.ndarray) -> dict[str, float | int]:
    residual = y - prediction
    loss = np.where(residual >= 0, quantile * residual, (1.0 - quantile) * -residual)
    dangerous = y > prediction * (1.0 + margin)
    actual_risk = y > slo
    predicted_risk = prediction > slo
    false_negative = actual_risk & ~predicted_risk
    return {
        "n": int(len(y)),
        "p95_pinball_loss": float(loss.mean()),
        "dangerous_underprediction_rate": float(dangerous.mean()),
        "slo_risk_false_negative_rate": float(false_negative.sum() / max(1, actual_risk.sum())),
        "mae_ms": float(np.abs(y - prediction).mean()),
    }


def relative_reduction(baseline: float, candidate: float) -> float:
    if baseline > 1e-12:
        return (baseline - candidate) / baseline
    return 0.0 if candidate <= 1e-12 else -1.0


def analyze(
    rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    overhead: Mapping[str, Any],
) -> dict[str, Any]:
    contract = config.get("analysis_contract")
    action = config.get("action")
    if not isinstance(contract, Mapping) or not isinstance(action, Mapping):
        raise AnalysisError("capture config lacks analysis/action contract")
    quantile = float(contract.get("target_quantile", 0.95))
    margin = float(
        contract.get("dangerous_underprediction_margin_fraction", 0.05)
    )
    ridge_alpha = float(contract.get("ridge_alpha", 1.0))
    quantile_l2_alpha = float(contract.get("quantile_l2_alpha", 1.0))
    if (
        not 0.5 < quantile < 1.0
        or margin < 0
        or ridge_alpha < 0
        or quantile_l2_alpha <= 0
    ):
        raise AnalysisError("invalid frozen analysis parameters")
    if (
        action.get("candidate_if_stage_d_is_authorized") != "running_set_budget"
        or action.get("stage_d_authorized") is not False
    ):
        raise AnalysisError(
            "P1 must retain only running_set_budget candidate semantics without Stage D"
        )
    if (
        overhead.get("schema") != "route-capacity-envelope-telemetry-overhead-v1"
        or overhead.get("status") != "TELEMETRY_OVERHEAD_OK"
        or overhead.get("token_output_match") is not True
        or overhead.get("logit_output_match") is not True
        or overhead.get("on_route_trace_stable") is not True
        or overhead.get("completion_trace_match") is not True
        or overhead.get("same_requests") is not True
        or overhead.get("same_seed") is not True
        or overhead.get("same_batch_schedule") is not True
        or overhead.get("same_decode_steps") is not True
        or overhead.get("same_dtype") is not True
    ):
        raise AnalysisError(
            "BLOCKED_HOOK_DISTORTION: paired telemetry ON/OFF check is missing or failed"
        )
    expected_model = contract.get("model")
    observed_model = overhead.get("model")
    telemetry_contract = config.get("telemetry_overhead_contract")
    if (
        not isinstance(expected_model, Mapping)
        or not isinstance(observed_model, Mapping)
        or observed_model.get("id") != expected_model.get("id")
        or observed_model.get("revision") != expected_model.get("revision")
    ):
        raise AnalysisError("telemetry check and frozen analysis model do not match")
    if not isinstance(telemetry_contract, Mapping):
        raise AnalysisError("capture config lacks telemetry overhead contract")
    allowed_overhead = float(telemetry_contract.get("max_relative_overhead", -1.0))
    if (
        allowed_overhead < 0
        or float(overhead.get("max_relative_overhead", math.inf)) > allowed_overhead
        or float(overhead.get("model_call_relative_overhead", math.inf)) > allowed_overhead
        or float(overhead.get("loop_wall_relative_overhead", math.inf)) > allowed_overhead
    ):
        raise AnalysisError("BLOCKED_HOOK_DISTORTION: telemetry overhead exceeds frozen limit")
    episodes = sorted({str(row["episode_id"]) for row in rows})
    split = validate_episode_split(rows)
    observed_windows = {
        (str(row["model"]), str(row["model_revision"])) for row in rows
    }
    frozen_identity = {
        (str(expected_model.get("id", "")), str(expected_model.get("revision", "")))
    }
    if observed_windows != frozen_identity:
        raise AnalysisError("window model identity differs from the frozen analysis model")
    route_conformance = {
        str(row["serial_route_conformance"]) for row in rows
    }
    batch_dependent_route = any(
        bool(row["batch_dependent_route_observed"]) for row in rows
    )
    route_match_fractions = [
        float(row["serial_route_identity_match_fraction"]) for row in rows
    ]
    if batch_dependent_route != ("BATCH_DEPENDENT" in route_conformance):
        raise AnalysisError("serial route-conformance fields are inconsistent")
    predictions: dict[str, list[float]] = {method: [] for method in METHODS}
    targets: list[float] = []
    slos: list[float] = []
    fold_metrics: dict[str, Any] = {}
    auxiliary: dict[str, Any] = {}

    for held_out in episodes:
        train = [row for row in rows if row["episode_id"] != held_out]
        test = [row for row in rows if row["episode_id"] == held_out]
        if not train or not test:
            raise AnalysisError("each leave-one-episode-out fold needs train and test rows")
        train_models = {(row["model"], row["model_revision"]) for row in train}
        test_models = {(row["model"], row["model_revision"]) for row in test}
        if train_models != test_models or len(train_models) != 1:
            raise AnalysisError("the two episodes must use the same single frozen model")
        y_train, y_test = _targets(train), _targets(test)
        calibration_slo = _higher_quantile(y_train, quantile) * float(
            contract.get("slo_multiplier", 1.10)
        )
        fold_metrics[held_out] = {}
        for method, features in METHODS.items():
            x_train, x_test = _matrix(train, features), _matrix(test, features)
            x_train, x_test = _standardize(x_train, x_test)
            coefficients = fit_quantile_linear(
                x_train, y_train, quantile, quantile_l2_alpha
            )
            prediction = predict_linear(coefficients, x_test)
            values = metric(
                y_test,
                prediction,
                quantile,
                margin,
                np.full(len(y_test), calibration_slo),
            )
            fold_metrics[held_out][method] = values
            predictions[method].extend(float(value) for value in prediction)

        for method in (
            "M2_workload_plus_expert_load",
            "M3_workload_expert_load_plus_historical_route",
        ):
            features = METHODS[method]
            x_train, x_test = _standardize(_matrix(train, features), _matrix(test, features))
            ridge_coefficients = fit_ridge(x_train, y_train, ridge_alpha)
            ridge_prediction = predict_linear(ridge_coefficients, x_test)
            tree = fit_tree(x_train, y_train, max_depth=4)
            tree_train = predict_tree(tree, x_train)
            tree_prediction = predict_tree(tree, x_test) + _higher_quantile(y_train - tree_train, quantile)
            auxiliary.setdefault(held_out, {})[method] = {
                "ridge_mae_ms": float(np.abs(y_test - ridge_prediction).mean()),
                "ridge_rmse_ms": float(
                    np.sqrt(np.mean((y_test - ridge_prediction) ** 2))
                ),
                "tree_max_depth": 4,
                "tree_p95_pinball_loss": metric(
                    y_test,
                    tree_prediction,
                    quantile,
                    margin,
                    np.full(len(y_test), calibration_slo),
                )["p95_pinball_loss"],
            }
        targets.extend(float(value) for value in y_test)
        slos.extend([calibration_slo] * len(y_test))

    target_array = np.asarray(targets)
    slo_array = np.asarray(slos)
    aggregate = {
        method: metric(target_array, np.asarray(values), quantile, margin, slo_array)
        for method, values in predictions.items()
    }
    m2 = aggregate["M2_workload_plus_expert_load"]
    m3 = aggregate["M3_workload_expert_load_plus_historical_route"]
    m4 = aggregate["M4_future_route_latency_diagnostic"]
    improvement = relative_reduction(float(m2["p95_pinball_loss"]), float(m3["p95_pinball_loss"]))
    under_reduction = relative_reduction(
        float(m2["dangerous_underprediction_rate"]),
        float(m3["dangerous_underprediction_rate"]),
    )
    pinball_fold_directions = [
        relative_reduction(
            float(values["M2_workload_plus_expert_load"]["p95_pinball_loss"]),
            float(values["M3_workload_expert_load_plus_historical_route"]["p95_pinball_loss"]),
        )
        for values in fold_metrics.values()
    ]
    underprediction_fold_directions = [
        relative_reduction(
            float(values["M2_workload_plus_expert_load"]["dangerous_underprediction_rate"]),
            float(values["M3_workload_expert_load_plus_historical_route"]["dangerous_underprediction_rate"]),
        )
        for values in fold_metrics.values()
    ]
    pinball_direction_consistent = all(
        value >= 0 for value in pinball_fold_directions
    )
    underprediction_direction_consistent = all(
        value >= 0 for value in underprediction_fold_directions
    )
    future_stronger = float(m4["p95_pinball_loss"]) < float(m3["p95_pinball_loss"])
    thresholds = contract.get("signal_thresholds")
    if not isinstance(thresholds, Mapping):
        raise AnalysisError("analysis contract lacks signal thresholds")
    promising_pinball = float(thresholds["promising_pinball_improvement"])
    promising_under = float(thresholds["promising_dangerous_reduction"])
    weak_pinball = float(thresholds["weak_pinball_improvement"])
    if (
        pinball_direction_consistent and improvement >= promising_pinball
    ) or (
        underprediction_direction_consistent and under_reduction >= promising_under
    ):
        signal_verdict = "PROMISING_SINGLE_MODEL"
    elif (
        pinball_direction_consistent
        and weak_pinball <= improvement < promising_pinball
        and future_stronger
    ):
        signal_verdict = "WEAK_SIGNAL_FOLD_INTO_DEPA"
    else:
        signal_verdict = "USE_TOKEN_OR_EXPERT_LOAD_CONTROLLER"
    verdict = (
        "PIVOT_TO_EXECUTION_CONFORMANCE"
        if batch_dependent_route
        else signal_verdict
    )
    return {
        "schema": "route-capacity-envelope-lightweight-p1-v1",
        "status": "M0_M4_COMPLETE",
        "verdict": verdict,
        "route_signal_diagnostic_verdict": signal_verdict,
        "evidence_scope": "development custom continuous runtime; not native serving",
        "method_contract": {name: list(features) for name, features in METHODS.items()},
        "primary_estimator": {
            "kind": "linear_quantile_regression",
            "target_quantile": quantile,
            "l2_alpha": quantile_l2_alpha,
            "solver": "deterministic_numpy_admm",
        },
        "information_boundary": "M0-M3 use only completed window t; M4 alone reads t+1 route",
        "serial_route_conformance": {
            "statuses": sorted(route_conformance),
            "minimum_expert_assignment_match_fraction": min(
                route_match_fractions
            ),
            "batch_dependent_route_observed": batch_dependent_route,
            "interpretation": (
                "serial-vs-batched route differs despite exact token parity; "
                "capacity interpretation is not authorized"
                if batch_dependent_route
                else "serial-vs-batched expert assignment closed"
            ),
        },
        "action_space": "none executed in P1; all methods use identical windows and targets",
        "telemetry_overhead": dict(overhead),
        "split": split,
        "aggregate": aggregate,
        "folds": fold_metrics,
        "auxiliary_ridge_and_depth4_tree": auxiliary,
        "comparison_m3_vs_m2": {
            "p95_pinball_relative_improvement": improvement,
            "dangerous_underprediction_relative_reduction": under_reduction,
            "pinball_fold_relative_improvements": pinball_fold_directions,
            "dangerous_underprediction_fold_relative_reductions": (
                underprediction_fold_directions
            ),
            "pinball_direction_consistent": pinball_direction_consistent,
            "dangerous_underprediction_direction_consistent": (
                underprediction_direction_consistent
            ),
            "future_route_diagnostic_stronger_than_historical": future_stronger,
        },
        "capacity_claim_authorized": False,
        "action_oracle_authorized": (
            not batch_dependent_route
            and verdict
            in {"PROMISING_SINGLE_MODEL", "WEAK_SIGNAL_FOLD_INTO_DEPA"}
        ),
    }


def render_report(result: Mapping[str, Any]) -> str:
    rows = [
        "# Route-Conditioned Capacity Envelope P1 development report",
        "",
        f"Verdict: `{result['verdict']}`.",
        "",
        "| Method | P95 pinball loss | Dangerous underprediction | SLO-risk false negative |",
        "|---|---:|---:|---:|",
    ]
    for method in METHODS:
        values = result["aggregate"][method]
        rows.append(
            f"| {method} | {float(values['p95_pinball_loss']):.8f} | "
            f"{float(values['dangerous_underprediction_rate']):.6f} | "
            f"{float(values['slo_risk_false_negative_rate']):.6f} |"
        )
    comparison = result["comparison_m3_vs_m2"]
    rows.extend(
        [
            "",
            "M3 vs M2 P95 pinball relative improvement: "
            f"`{float(comparison['p95_pinball_relative_improvement']):+.4%}`.",
            "",
            "M3 vs M2 dangerous-underprediction relative reduction: "
            f"`{float(comparison['dangerous_underprediction_relative_reduction']):+.4%}`.",
            "",
            "M4 reads the observed t+1 route and is diagnostic only. No action was "
            "executed; this report does not establish safe capacity, action headroom, "
            "controller gain, native serving behavior, or a two-model claim.",
            "",
            "Serial-vs-batched route conformance: "
            f"`{result['serial_route_conformance']}`. A batch-dependent route "
            "finding overrides the route-signal diagnostic verdict and blocks a "
            "capacity interpretation.",
            "",
        ]
    )
    return "\n".join(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--overhead", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--report", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    overhead = json.loads(Path(args.overhead).read_text(encoding="utf-8"))
    rows = align_next_window(read_windows(Path(args.windows)))
    result = analyze(rows, config, overhead)
    metrics = Path(args.metrics)
    report = Path(args.report)
    if metrics.exists() or report.exists():
        raise SystemExit("refusing to overwrite metrics/report")
    metrics.parent.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    metrics.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report.write_text(render_report(result), encoding="utf-8")
    print(json.dumps({"verdict": result["verdict"], "metrics": str(metrics)}, sort_keys=True))


if __name__ == "__main__":
    main()
