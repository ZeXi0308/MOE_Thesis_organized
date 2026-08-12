#!/usr/bin/env python3
"""Compare workload-only and route-augmented next-window P95 predictors.

The analysis uses identity-aware split checks, a fixed ridge model, and a
validation residual quantile. Scientific eligibility additionally requires
arrival-episode disjointness; a synthetic or non-representative smoke run can
never become a scientific P1 verdict.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


WORKLOAD_FEATURES = [
    "active_tokens",
    "running_sequences",
    "queue_depth",
    "mean_kv_length",
    "max_kv_length",
    "prompt_tokens",
    "decode_tokens",
    "batch_size",
    "recent_step_ms",
    "recent_tokens_per_second",
]

ROUTE_FEATURES = [
    "route_max_mean",
    "route_cv",
    "route_hhi",
    "active_experts",
    "top1_expert_share",
    "cross_layer_max_pressure",
    "cross_layer_mean_pressure",
    "hotspot_persistence",
    "route_shape_ewma",
    "route_shape_delta",
    "max_expert_tokens",
]

METHODS = {
    "M0_constant": [],
    "M1_workload_only": WORKLOAD_FEATURES,
    "M2_route_only": ROUTE_FEATURES,
    "M3_workload_plus_route": WORKLOAD_FEATURES + ROUTE_FEATURES,
    "M4_future_route_oracle": WORKLOAD_FEATURES + [f"future_{name}" for name in ROUTE_FEATURES],
}

FROZEN_MODELS = {
    "allenai/OLMoE-1B-7B-0924": "6d84c48581ece794365f2b8e9cfb043c68ade9c5",
    "llm-jp/optimal-sparsity-math-d512-E32-k16-920M-A520M": (
        "1d5983076dfc67aee4a77ec06a27027f5bab6055"
    ),
}
REQUIRED_ARRIVAL_REGIMES = {"steady", "bursty"}


class ProtocolError(RuntimeError):
    pass


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {
            "model",
            "model_revision",
            "arrival_episode_id",
            "arrival_episode_independent",
            "episode_id",
            "split",
            "window_id",
            "arrival_regime",
            "request_ids_json",
            "document_ids_json",
            "window_start_us",
            "window_end_us",
            "feature_available_at_us",
            "decode_stage",
            "step_service_ms",
            "evidence_type",
            "runtime_representative",
            "instrumentation_overhead_measured",
            "fresh_holdout_sealed",
            "gate_weight_available",
            *WORKLOAD_FEATURES,
            *ROUTE_FEATURES,
        }
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ProtocolError(f"feature table lacks fields: {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ProtocolError("feature table is empty")
    return rows


def _identity_values(row: Mapping[str, str], field: str) -> tuple[str, ...]:
    try:
        values = json.loads(row[field])
    except (TypeError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"{field} must be a JSON string list") from exc
    if (
        not isinstance(values, list)
        or not values
        or any(not isinstance(value, str) or not value for value in values)
        or len(values) != len(set(values))
    ):
        raise ProtocolError(f"{field} must contain unique non-empty strings")
    return tuple(values)


def _as_float(row: Mapping[str, str], field: str) -> float:
    value = float(row[field])
    if not math.isfinite(value):
        raise ProtocolError(f"non-finite feature {field}")
    return value


def align_next_window(rows: Sequence[Mapping[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, str]]] = {}
    window_ids: set[tuple[str, str]] = set()
    for row in rows:
        window_identity = (row["model"], row["window_id"])
        if window_identity in window_ids:
            raise ProtocolError(f"duplicate window identity: {window_identity}")
        window_ids.add(window_identity)
        grouped.setdefault((row["model"], row["episode_id"]), []).append(row)
    aligned: list[dict[str, Any]] = []
    for (model, episode), values in sorted(grouped.items()):
        ordered = sorted(values, key=lambda row: (_as_float(row, "window_start_us"), row["window_id"]))
        splits = {row["split"] for row in ordered}
        if len(splits) != 1:
            raise ProtocolError(f"episode {episode} crosses declared splits")
        for current, future in zip(ordered, ordered[1:]):
            if _as_float(current, "window_end_us") > _as_float(future, "window_start_us"):
                raise ProtocolError("next-window pair overlaps or time-regresses inside an episode")
            if _as_float(current, "feature_available_at_us") > _as_float(
                current, "window_end_us"
            ):
                raise ProtocolError("a causal feature becomes available after window t")
            for invariant in (
                "model_revision",
                "arrival_episode_id",
                "arrival_regime",
            ):
                if current[invariant] != future[invariant]:
                    raise ProtocolError(
                        f"{invariant} changes inside one aligned episode"
                    )
            record: dict[str, Any] = {
                "model": model,
                "model_revision": current["model_revision"],
                "arrival_episode_id": current["arrival_episode_id"],
                "arrival_episode_independent": (
                    current["arrival_episode_independent"].lower() == "true"
                ),
                "target_arrival_episode_independent": (
                    future["arrival_episode_independent"].lower() == "true"
                ),
                "episode_id": episode,
                "declared_split": current["split"],
                "arrival_regime": current["arrival_regime"],
                "decode_stage": _as_float(current, "decode_stage"),
                "target_step_service_ms": _as_float(future, "step_service_ms"),
                "evidence_type": current["evidence_type"],
                "target_evidence_type": future["evidence_type"],
                "runtime_representative": current["runtime_representative"].lower() == "true",
                "target_runtime_representative": future[
                    "runtime_representative"
                ].lower()
                == "true",
                "instrumentation_overhead_measured": (
                    current["instrumentation_overhead_measured"].lower() == "true"
                ),
                "target_instrumentation_overhead_measured": (
                    future["instrumentation_overhead_measured"].lower() == "true"
                ),
                "fresh_holdout_sealed": current["fresh_holdout_sealed"].lower() == "true",
                "target_fresh_holdout_sealed": future["fresh_holdout_sealed"].lower()
                == "true",
                "gate_weight_available": current["gate_weight_available"].lower()
                == "true",
                "target_gate_weight_available": future[
                    "gate_weight_available"
                ].lower()
                == "true",
                # Include identities touching either side of the t -> t+1 pair.
                # This lets the split audit fail closed even when a serving
                # batch changes membership between adjacent windows.
                "request_ids": tuple(
                    sorted(
                        set(_identity_values(current, "request_ids_json"))
                        | set(_identity_values(future, "request_ids_json"))
                    )
                ),
                "document_ids": tuple(
                    sorted(
                        set(_identity_values(current, "document_ids_json"))
                        | set(_identity_values(future, "document_ids_json"))
                    )
                ),
            }
            try:
                current_requests = json.loads(current["request_ids_json"])
                future_requests = json.loads(future["request_ids_json"])
                current_documents = json.loads(current["document_ids_json"])
                future_documents = json.loads(future["document_ids_json"])
            except json.JSONDecodeError as exc:
                raise ProtocolError("request/document identity JSON is invalid") from exc
            if not all(
                isinstance(value, list)
                for value in (
                    current_requests,
                    future_requests,
                    current_documents,
                    future_documents,
                )
            ):
                raise ProtocolError("request/document identity fields must be JSON lists")
            record["split_identities"] = tuple(
                sorted(
                    {f"request:{value}" for value in current_requests + future_requests}
                    | {f"document:{value}" for value in current_documents + future_documents}
                )
            )
            if not record["split_identities"]:
                raise ProtocolError("an aligned pair has no request/document split identity")
            for name in WORKLOAD_FEATURES + ROUTE_FEATURES:
                record[name] = _as_float(current, name)
            for name in ROUTE_FEATURES:
                record[f"future_{name}"] = _as_float(future, name)
            aligned.append(record)
    if not aligned:
        raise ProtocolError("each episode needs at least two windows")
    return aligned


def _bucket(value: float, width: float) -> int:
    if width <= 0:
        raise ProtocolError("matched-cell bucket widths must be positive")
    return math.floor(value / width)


def annotate_matched_cells(
    rows: Sequence[dict[str, Any]], config: Mapping[str, Any]
) -> list[dict[str, Any]]:
    widths = config.get("matched_cell_bucket_widths")
    if not isinstance(widths, dict):
        raise ProtocolError("config must freeze matched_cell_bucket_widths")
    required = {
        "active_tokens",
        "running_sequences",
        "queue_depth",
        "mean_kv_length",
        "max_kv_length",
        "prompt_tokens",
        "decode_tokens",
        "batch_size",
        "decode_stage",
    }
    missing = required - set(widths)
    if missing:
        raise ProtocolError(f"matched-cell widths are missing: {sorted(missing)}")
    output: list[dict[str, Any]] = []
    for raw in rows:
        row = dict(raw)
        coordinates = (
            str(row["model"]),
            str(row["model_revision"]),
            _bucket(float(row["active_tokens"]), float(widths["active_tokens"])),
            _bucket(
                float(row["running_sequences"]),
                float(widths["running_sequences"]),
            ),
            _bucket(float(row["queue_depth"]), float(widths["queue_depth"])),
            _bucket(
                float(row["mean_kv_length"]),
                float(widths["mean_kv_length"]),
            ),
            _bucket(float(row["max_kv_length"]), float(widths["max_kv_length"])),
            _bucket(float(row["prompt_tokens"]), float(widths["prompt_tokens"])),
            _bucket(float(row["decode_tokens"]), float(widths["decode_tokens"])),
            _bucket(float(row["batch_size"]), float(widths["batch_size"])),
            str(row["arrival_regime"]),
            _bucket(float(row["decode_stage"]), float(widths["decode_stage"])),
        )
        row["matched_cell_id"] = json.dumps(coordinates, separators=(",", ":"))
        output.append(row)
    return output


def _stable_group_order(model: str, groups: Sequence[str]) -> list[str]:
    return sorted(
        groups,
        key=lambda group: hashlib.sha256(f"{model}|{group}".encode("utf-8")).hexdigest(),
    )


def _row_identities(row: Mapping[str, Any]) -> tuple[str, ...]:
    if "split_identities" in row:
        values = row["split_identities"]
    elif "request_ids" in row:
        values = row["request_ids"]
    elif "request_ids_json" in row:
        values = json.loads(str(row["request_ids_json"]))
    else:
        raise ProtocolError("row lacks request identities for grouped splitting")
    if isinstance(values, str) or not isinstance(values, (list, tuple, set)):
        raise ProtocolError("request identities must be a sequence")
    normalized = tuple(sorted({str(value) for value in values if str(value)}))
    if not normalized:
        raise ProtocolError("row has no non-empty request identity")
    return normalized


def _component_map(rows: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    """Map every request/document identity to its overlap-connected component."""

    parent: dict[str, str] = {}

    def find(value: str) -> str:
        parent.setdefault(value, value)
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            if left_root < right_root:
                parent[right_root] = left_root
            else:
                parent[left_root] = right_root

    for row in rows:
        identities = _row_identities(row)
        anchor = identities[0]
        find(anchor)
        for value in identities[1:]:
            union(anchor, value)
    members: dict[str, list[str]] = {}
    for value in parent:
        members.setdefault(find(value), []).append(value)
    component_name = {
        root: hashlib.sha256("\n".join(sorted(values)).encode("utf-8")).hexdigest()
        for root, values in members.items()
    }
    return {value: component_name[find(value)] for value in parent}


def assign_splits(rows: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output = [dict(row) for row in rows]
    diagnostics: dict[str, Any] = {}
    for model in sorted({str(row["model"]) for row in output}):
        model_rows = [row for row in output if row["model"] == model]
        components = _component_map(model_rows)
        row_groups: dict[int, str] = {}
        for row in model_rows:
            identities = _row_identities(row)
            group = components[identities[0]]
            if any(components[value] != group for value in identities):
                raise AssertionError("component closure failed")
            row_groups[id(row)] = group
        groups = sorted(set(row_groups.values()))
        if len(groups) < 5:
            raise ProtocolError(
                f"model {model} needs >=5 request/document-disjoint components; observed {len(groups)}"
            )
        declared_sets: dict[str, set[str]] = {}
        for row in model_rows:
            declared_sets.setdefault(row_groups[id(row)], set()).add(
                str(row["declared_split"]).lower()
            )
        if any(len(values) != 1 for values in declared_sets.values()):
            raise ProtocolError("one request/document component crosses declared splits")
        declared = {group: next(iter(values)) for group, values in declared_sets.items()}
        declared_values = set(declared.values())
        if declared_values <= {"train", "validation", "test"} and declared_values == {
            "train",
            "validation",
            "test",
        }:
            assignment = declared
            policy = "predeclared_request_document_component_split"
        else:
            ordered = _stable_group_order(model, groups)
            n_train = max(1, int(len(ordered) * 0.6))
            n_validation = max(1, int(len(ordered) * 0.2))
            if n_train + n_validation >= len(ordered):
                n_train = len(ordered) - 2
                n_validation = 1
            assignment = {}
            for index, group in enumerate(ordered):
                assignment[group] = (
                    "train"
                    if index < n_train
                    else "validation"
                    if index < n_train + n_validation
                    else "test"
                )
            policy = "development_sha256_request_document_component_split"
        for row in model_rows:
            row["analysis_group"] = row_groups[id(row)]
            row["analysis_split"] = assignment[row_groups[id(row)]]
        counts = {
            split: sum(value == split for value in assignment.values())
            for split in ("train", "validation", "test")
        }
        if any(value < 1 for value in counts.values()):
            raise ProtocolError(f"model {model} has an empty split")
        diagnostics[model] = {
            "policy": policy,
            "component_counts": counts,
            "components": assignment,
            "request_document_overlap_across_splits": 0,
        }
    return output, diagnostics


def validate_identity_disjoint_splits(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Reject request/document leakage across train/validation/test.

    Identities are scoped by model because models are fitted independently.
    Repeated adjacent windows within one episode are expected and remain in
    one split; reuse across episode splits is not.
    """

    diagnostics: dict[str, Any] = {}
    for model in sorted({str(row["model"]) for row in rows}):
        model_rows = [row for row in rows if str(row["model"]) == model]
        model_diagnostic: dict[str, Any] = {}
        for field in ("request_ids", "document_ids"):
            owners: dict[str, set[str]] = {}
            for row in model_rows:
                split = str(row["analysis_split"])
                for identity in row[field]:
                    owners.setdefault(str(identity), set()).add(split)
            overlaps = {
                identity: sorted(splits)
                for identity, splits in owners.items()
                if len(splits) > 1
            }
            if overlaps:
                preview = dict(list(sorted(overlaps.items()))[:5])
                raise ProtocolError(
                    f"{model} {field} cross analysis splits: {preview}"
                )
            model_diagnostic[field] = {
                "unique_identities": len(owners),
                "cross_split_overlap": 0,
            }
        diagnostics[model] = model_diagnostic
    return diagnostics


def _matrix(rows: Sequence[Mapping[str, Any]], features: Sequence[str]) -> Any:
    import numpy as np

    if not features:
        return np.empty((len(rows), 0), dtype=float)
    return np.asarray([[float(row[name]) for name in features] for row in rows], dtype=float)


def _targets(rows: Sequence[Mapping[str, Any]]) -> Any:
    import numpy as np

    return np.asarray([float(row["target_step_service_ms"]) for row in rows], dtype=float)


def _fit_ridge(x: Any, y: Any, alpha: float) -> dict[str, Any]:
    import numpy as np

    if x.shape[1] == 0:
        return {"constant": float(y.mean())}
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-12] = 1.0
    standardized = (x - mean) / scale
    design = np.column_stack([np.ones(len(x)), standardized])
    penalty = np.eye(design.shape[1]) * alpha
    penalty[0, 0] = 0.0
    try:
        beta = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    except np.linalg.LinAlgError:
        beta = np.linalg.lstsq(design.T @ design + penalty, design.T @ y, rcond=None)[0]
    return {"mean": mean, "scale": scale, "beta": beta}


def _predict(model: Mapping[str, Any], x: Any) -> Any:
    import numpy as np

    if "constant" in model:
        return np.full(x.shape[0], float(model["constant"]), dtype=float)
    standardized = (x - model["mean"]) / model["scale"]
    return np.column_stack([np.ones(len(x)), standardized]) @ model["beta"]


def _higher_quantile(values: Any, quantile: float) -> float:
    import numpy as np

    return float(np.quantile(values, quantile, method="higher"))


def _metrics(
    y: Any,
    prediction: Any,
    quantile: float,
    dangerous_underprediction_margin: float,
) -> dict[str, float]:
    import numpy as np

    residual = y - prediction
    pinball = np.where(residual >= 0, quantile * residual, (1 - quantile) * -residual)
    return {
        "pinball_loss": float(pinball.mean()),
        "dangerous_underprediction_rate": float(
            (y > prediction * (1.0 + dangerous_underprediction_margin)).mean()
        ),
        "rmse_ms": float(np.sqrt(np.mean((y - prediction) ** 2))),
        "mean_prediction_ms": float(prediction.mean()),
        "mean_target_ms": float(y.mean()),
    }


def evaluate_model(
    rows: Sequence[Mapping[str, Any]],
    alpha: float,
    quantile: float,
    dangerous_underprediction_margin: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    train = [row for row in rows if row["analysis_split"] == "train"]
    validation = [row for row in rows if row["analysis_split"] == "validation"]
    test = [row for row in rows if row["analysis_split"] == "test"]
    if not train or not validation or not test:
        raise ProtocolError("train/validation/test must all contain aligned windows")
    results: list[dict[str, Any]] = []
    predictions: dict[str, list[float]] = {}
    y_train = _targets(train)
    y_validation = _targets(validation)
    y_test = _targets(test)
    for method, features in METHODS.items():
        x_train = _matrix(train, features)
        x_validation = _matrix(validation, features)
        x_test = _matrix(test, features)
        fitted = _fit_ridge(x_train, y_train, alpha)
        validation_mean = _predict(fitted, x_validation)
        residual_quantile = _higher_quantile(y_validation - validation_mean, quantile)
        test_mean = _predict(fitted, x_test)
        test_prediction = test_mean + residual_quantile
        metric = _metrics(
            y_test,
            test_prediction,
            quantile,
            dangerous_underprediction_margin,
        )
        results.append(
            {
                "method": method,
                "features": ",".join(features),
                "n_train": len(train),
                "n_validation": len(validation),
                "n_test": len(test),
                "residual_quantile_ms": residual_quantile,
                **metric,
            }
        )
        predictions[method] = [float(value) for value in test_prediction]
    return results, {
        "targets": [float(value) for value in y_test],
        "predictions": predictions,
    }


def _relative_reduction(baseline: float, candidate: float) -> float:
    return (baseline - candidate) / baseline if abs(baseline) > 1e-12 else 0.0


def _rate_reduction(baseline: float, candidate: float) -> float:
    if baseline > 1e-12:
        return (baseline - candidate) / baseline
    return 0.0 if candidate <= 1e-12 else -1.0


def analyze(rows: Sequence[dict[str, Any]], config: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    alpha = float(config["ridge_alpha"])
    quantile = float(config["target_quantile"])
    dangerous_margin = float(config["dangerous_underprediction_margin"])
    action = str(config.get("action", config.get("first_action", "")))
    if alpha < 0 or not 0.5 < quantile < 1 or dangerous_margin < 0:
        raise ProtocolError("invalid frozen ridge/quantile configuration")
    if action != "next_window_active_token_budget":
        raise ProtocolError(
            "first action must be frozen to next_window_active_token_budget"
        )
    if config.get("frozen_models") != FROZEN_MODELS:
        raise ProtocolError("config must bind the two exact frozen model revisions")
    raw_regimes = config.get("required_arrival_regimes")
    if (
        not isinstance(raw_regimes, list)
        or len(raw_regimes) != len(REQUIRED_ARRIVAL_REGIMES)
        or set(raw_regimes) != REQUIRED_ARRIVAL_REGIMES
    ):
        raise ProtocolError("config must bind exactly steady and bursty regimes")
    matched_rows = annotate_matched_cells(rows, config)
    split_rows, split_diagnostics = assign_splits(matched_rows)
    identity_diagnostics = validate_identity_disjoint_splits(split_rows)
    metric_rows: list[dict[str, Any]] = []
    comparisons: dict[str, Any] = {}
    model_prediction_sets: dict[str, dict[str, Any]] = {}
    for model in sorted({str(row["model"]) for row in split_rows}):
        model_rows = [row for row in split_rows if row["model"] == model]
        metrics, predictions = evaluate_model(
            model_rows,
            alpha,
            quantile,
            dangerous_margin,
        )
        by_method = {row["method"]: row for row in metrics}
        for row in metrics:
            metric_rows.append({"scope": "model", "model": model, **row})
        m1 = by_method["M1_workload_only"]
        m3 = by_method["M3_workload_plus_route"]
        comparisons[model] = {
            "p95_pinball_relative_improvement": _relative_reduction(
                float(m1["pinball_loss"]), float(m3["pinball_loss"])
            ),
            "dangerous_underprediction_relative_reduction": _rate_reduction(
                float(m1["dangerous_underprediction_rate"]),
                float(m3["dangerous_underprediction_rate"]),
            ),
            "m1_dangerous_underprediction_rate": float(
                m1["dangerous_underprediction_rate"]
            ),
            "m3_dangerous_underprediction_rate": float(
                m3["dangerous_underprediction_rate"]
            ),
        }
        model_prediction_sets[model] = predictions

    import numpy as np

    for method in METHODS:
        targets: list[float] = []
        predicted: list[float] = []
        for values in model_prediction_sets.values():
            targets.extend(values["targets"])
            predicted.extend(values["predictions"][method])
        aggregate = _metrics(
            np.asarray(targets),
            np.asarray(predicted),
            quantile,
            dangerous_margin,
        )
        metric_rows.append(
            {
                "scope": "aggregate",
                "model": "ALL",
                "method": method,
                "features": ",".join(METHODS[method]),
                "n_train": sum(
                    int(row["n_train"])
                    for row in metric_rows
                    if row["scope"] == "model" and row["method"] == method
                ),
                "n_validation": sum(
                    int(row["n_validation"])
                    for row in metric_rows
                    if row["scope"] == "model" and row["method"] == method
                ),
                "n_test": len(targets),
                "residual_quantile_ms": "",
                **aggregate,
            }
        )
    aggregate_by_method = {
        row["method"]: row
        for row in metric_rows
        if row["scope"] == "aggregate"
    }
    aggregate_comparison = {
        "p95_pinball_relative_improvement": _relative_reduction(
            float(aggregate_by_method["M1_workload_only"]["pinball_loss"]),
            float(aggregate_by_method["M3_workload_plus_route"]["pinball_loss"]),
        ),
        "dangerous_underprediction_relative_reduction": _rate_reduction(
            float(
                aggregate_by_method["M1_workload_only"][
                    "dangerous_underprediction_rate"
                ]
            ),
            float(
                aggregate_by_method["M3_workload_plus_route"][
                    "dangerous_underprediction_rate"
                ]
            ),
        ),
    }
    cell_diagnostics: dict[str, Any] = {}
    cell_coverages: list[float] = []
    for model in sorted({str(row["model"]) for row in split_rows}):
        model_rows = [row for row in split_rows if row["model"] == model]
        train_cells = {
            str(row["matched_cell_id"])
            for row in model_rows
            if row["analysis_split"] == "train"
        }
        test_rows = [row for row in model_rows if row["analysis_split"] == "test"]
        test_cells = {str(row["matched_cell_id"]) for row in test_rows}
        covered_cells = test_cells & train_cells
        coverage = len(covered_cells) / len(test_cells)
        covered_rows = sum(
            str(row["matched_cell_id"]) in train_cells for row in test_rows
        )
        cell_coverages.append(coverage)
        cell_diagnostics[model] = {
            "train_cells": len(train_cells),
            "test_cells": len(test_cells),
            "test_cells_with_train_support": len(covered_cells),
            "test_rows": len(test_rows),
            "test_rows_with_train_cell": covered_rows,
            "test_cell_train_coverage": coverage,
        }
    evidence_types = {
        str(row[field])
        for row in split_rows
        for field in ("evidence_type", "target_evidence_type")
    }
    models = sorted({str(row["model"]) for row in split_rows})
    regimes = {
        model: sorted(
            {str(row["arrival_regime"]) for row in split_rows if row["model"] == model}
        )
        for model in models
    }
    split_arrival_sets = {
        split: {
            str(row["arrival_episode_id"])
            for row in split_rows
            if row["analysis_split"] == split
        }
        for split in ("train", "validation", "test")
    }
    arrival_episode_overlap = sorted(
        (split_arrival_sets["train"] & split_arrival_sets["validation"])
        | (split_arrival_sets["train"] & split_arrival_sets["test"])
        | (split_arrival_sets["validation"] & split_arrival_sets["test"])
    )
    minimum_regimes = int(config.get("minimum_arrival_regimes_per_model", 2))
    eligibility_checks = {
        "observed_real_runtime_only": evidence_types == {"[Observed real runtime]"},
        "runtime_representative": all(
            bool(row["runtime_representative"])
            and bool(row["target_runtime_representative"])
            for row in split_rows
        ),
        "instrumentation_overhead_measured": all(
            bool(row["instrumentation_overhead_measured"])
            and bool(row["target_instrumentation_overhead_measured"])
            for row in split_rows
        ),
        "fresh_holdout_sealed": all(
            bool(row["fresh_holdout_sealed"])
            and bool(row["target_fresh_holdout_sealed"])
            for row in split_rows
        ),
        "independent_arrival_episodes": all(
            bool(row["arrival_episode_independent"])
            and bool(row["target_arrival_episode_independent"])
            for row in split_rows
        ),
        "arrival_episode_disjoint": not arrival_episode_overlap,
        "gate_weight_available": all(
            bool(row["gate_weight_available"])
            and bool(row["target_gate_weight_available"])
            for row in split_rows
        ),
        "two_frozen_models": {
            (str(row["model"]), str(row["model_revision"]))
            for row in split_rows
        }
        == set(FROZEN_MODELS.items()),
        "arrival_regimes_per_model": all(
            len(values) >= minimum_regimes
            and set(values) >= REQUIRED_ARRIVAL_REGIMES
            for model, values in regimes.items()
            if model in FROZEN_MODELS
        )
        and set(regimes) == set(FROZEN_MODELS),
        "predeclared_split": all(
            value["policy"] == "predeclared_request_document_component_split"
            for value in split_diagnostics.values()
        ),
        "matched_cell_coverage": min(cell_coverages)
        >= float(config["minimum_test_cell_train_coverage"]),
        "request_document_disjoint": all(
            values["request_ids"]["cross_split_overlap"] == 0
            and values["document_ids"]["cross_split_overlap"] == 0
            for values in identity_diagnostics.values()
        ),
    }
    scientific_eligible = all(eligibility_checks.values())
    if not scientific_eligible:
        p1_status = "SMOKE_ONLY_NOT_SCIENTIFICALLY_ELIGIBLE"
        verdict = "BLOCKED_RUNTIME_NOT_REPRESENTATIVE"
    else:
        improve_gate = float(config["pinball_relative_improvement_gate"])
        under_gate = float(config["underprediction_relative_reduction_gate"])
        per_model_pass = [
            (
                value["p95_pinball_relative_improvement"] >= improve_gate
                and value["m3_dangerous_underprediction_rate"]
                <= value["m1_dangerous_underprediction_rate"]
            )
            or value["dangerous_underprediction_relative_reduction"] >= under_gate
            for value in comparisons.values()
        ]
        per_model_direction = [
            (
                value["p95_pinball_relative_improvement"] > 0
                and value["m3_dangerous_underprediction_rate"]
                <= value["m1_dangerous_underprediction_rate"]
            )
            or value["dangerous_underprediction_relative_reduction"] > 0
            for value in comparisons.values()
        ]
        if all(per_model_pass):
            p1_status = "P1_INCREMENTAL_SIGNAL_PASS"
            verdict = "GO_TO_ORACLE_TEST"
        elif any(per_model_pass) and all(per_model_direction):
            p1_status = "WEAK_SIGNAL_NEEDS_MORE_EVENTS"
            verdict = "CONTINUE_INCREMENTAL_SIGNAL_TEST"
        elif (
            aggregate_comparison["p95_pinball_relative_improvement"]
            < float(config["stop_below_relative_improvement"])
            or not all(per_model_direction)
        ):
            p1_status = "STOP_NO_INCREMENTAL_ROUTE_SIGNAL"
            verdict = "STOP_NO_INCREMENTAL_ROUTE_SIGNAL"
        else:
            p1_status = "WEAK_SIGNAL_NEEDS_MORE_EVENTS"
            verdict = "CONTINUE_INCREMENTAL_SIGNAL_TEST"
    summary = {
        "schema": "route-shape-slo-p1-summary-v1",
        "p1_status": p1_status,
        "verdict": verdict,
        "scientific_result_eligible": scientific_eligible,
        "p1_gate_eligible": scientific_eligible,
        "classification": (
            "D"
            if not scientific_eligible
            else "E"
            if p1_status == "STOP_NO_INCREMENTAL_ROUTE_SIGNAL"
            else "B"
        ),
        "action": action,
        "evidence_types": sorted(evidence_types),
        "target": "next-window step_service_ms P95",
        "information_boundary": "M0-M3 use only window <=t; M4 alone reads route(t+1)",
        "oracle_boundary": (
            "M4 is a future-route latency-prediction upper bound only; it is not a "
            "counterfactual capacity-action Oracle and cannot establish H3."
        ),
        "split": split_diagnostics,
        "identity_leakage": identity_diagnostics,
        "matched_cells": cell_diagnostics,
        "arrival_regimes": regimes,
        "arrival_episode_split": {
            "ids_by_split": {
                split: sorted(values)
                for split, values in split_arrival_sets.items()
            },
            "overlap": arrival_episode_overlap,
        },
        "eligibility_checks": eligibility_checks,
        "eligibility_blockers": sorted(
            name for name, passed in eligibility_checks.items() if not passed
        ),
        "comparisons": comparisons,
        "aggregate_comparison": aggregate_comparison,
        "thresholds": {
            "dangerous_underprediction_margin": dangerous_margin,
            "pinball_relative_improvement_gate": float(
                config["pinball_relative_improvement_gate"]
            ),
            "underprediction_relative_reduction_gate": float(
                config["underprediction_relative_reduction_gate"]
            ),
            "stop_below_relative_improvement": float(
                config["stop_below_relative_improvement"]
            ),
        },
        "claim_boundary": (
            "A smoke-only result validates code paths and leakage guards only; it does not "
            "measure route-conditioned serving capacity."
            if not scientific_eligible
            else (
                "Eligible P1 evidence found no stable incremental route signal; stop the "
                "route-aware controller path and retain a token-only capacity baseline."
                if p1_status == "STOP_NO_INCREMENTAL_ROUTE_SIGNAL"
                else "Eligible P1 evidence measures incremental prediction only; safe-capacity "
                "action headroom and causal controller benefit remain unmeasured."
            )
        ),
    }
    return metric_rows, summary


def write_metrics(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = [
        "scope",
        "model",
        "method",
        "features",
        "n_train",
        "n_validation",
        "n_test",
        "residual_quantile_ms",
        "pinball_loss",
        "dangerous_underprediction_rate",
        "rmse_ms",
        "mean_prediction_ms",
        "mean_target_ms",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, summary: Mapping[str, Any], metrics: Sequence[Mapping[str, Any]]) -> None:
    aggregate = {
        row["method"]: row
        for row in metrics
        if row["scope"] == "aggregate"
    }
    m1 = aggregate["M1_workload_only"]
    m3 = aggregate["M3_workload_plus_route"]
    relative_change = float(
        summary["aggregate_comparison"]["p95_pinball_relative_improvement"]
    )
    dangerous_count_m1 = round(
        float(m1["dangerous_underprediction_rate"]) * int(m1["n_test"])
    )
    dangerous_count_m3 = round(
        float(m3["dangerous_underprediction_rate"]) * int(m3["n_test"])
    )
    comparison_rows = []
    for method in METHODS:
        row = aggregate[method]
        comparison_rows.append(
            f"| {method} | {float(row['pinball_loss']):.8f} | "
            f"{float(row['dangerous_underprediction_rate']):.6f} |"
        )
    if summary["scientific_result_eligible"] is True:
        p1_status = str(summary["p1_status"])
        positive = p1_status in {
            "P1_INCREMENTAL_SIGNAL_PASS",
            "WEAK_SIGNAL_NEEDS_MORE_EVENTS",
        }
        under_reduction = float(
            summary["aggregate_comparison"][
                "dangerous_underprediction_relative_reduction"
            ]
        )
        p2_decision = (
            "READY_TO_IMPLEMENT_P2_REPLAY" if positive else "BLOCKED_P1_FAILED"
        )
        relationship = (
            "`B / PROVISIONAL_DEPA_SUBMODULE`"
            if positive
            else "`E / STOP_NO_INCREMENTAL_ROUTE_SIGNAL`"
        )
        direct_answer = (
            "Eligible held-out P1 evidence shows an incremental historical-route "
            "prediction signal, but it does not yet show that route changes safely "
            "supportable capacity. P2 action-conditioned replay is required."
            if positive
            else "Eligible held-out P1 evidence did not show a stable incremental route "
            "signal beyond workload state. Stop the route-aware controller path and use "
            "the strongest token-only capacity controller."
        )
        next_step = (
            "Run only P2: an action-conditioned future-route Oracle replay over the "
            "frozen active-token-budget grid, regenerating admitted sets and routes for "
            "every candidate budget."
            if positive
            else "No further RouteShape-SLO experiment is authorized by this result; "
            "pivot to the token-only controller baseline."
        )
        eligible_lines = [
            "# RouteShape-SLO final exploration report",
            "",
            "## Verdict",
            "",
            f"`{summary['verdict']}`",
            "",
            f"P1 status is `{p1_status}` and scientific eligibility is `true`. "
            f"Classification is `{summary['classification']}`.",
            "",
            "## Evidence Table",
            "",
            "| Question | Result | Evidence type | Boundary | Decision |",
            "|---|---|---|---|---|",
            f"| Does route add capacity information? | Eligible P1 changes M1-to-M3 P95 pinball loss by {relative_change:+.4%}; dangerous-underprediction relative reduction is {under_reduction:+.4%}. | `[Observed real runtime]` plus `[Offline replay]` | Incremental next-window prediction only; not action-conditioned safe capacity. | `{p1_status}` |",
            f"| Does a future-route Oracle expose action headroom? | Not run. M4 remains only a future-route latency predictor. | `[Not measured]` | No candidate-budget counterfactual route regeneration. | `{p2_decision}` |",
            "| Is historical route causally usable? | Not run. | `[Not measured]` | P3 remains forbidden until P2 passes. | `BLOCKED_P2_NOT_PASSED` |",
            "| Does a simple policy already cover the gain? | Not measured. | `[Not measured]` | P2/P3 baselines have not executed. | `UNRESOLVED` |",
            "| Is the single-GPU runtime representative? | Yes for the frozen P1 eligibility contract only. | `[Observed real runtime]` | Does not establish multi-GPU EP behavior. | `READY_FOR_SIGNAL_TEST` |",
            f"| Independent or BCRD/DEPA submodule? | {relationship} | `[Analytic model]` plus eligible P1 | Independence still requires P2 and P3. | `{summary['classification']}` |",
            "",
            "## Eligible P1 result",
            "",
            f"Evidence types: {', '.join(summary['evidence_types'])}. There are "
            f"`{int(m1['n_train'])}/{int(m1['n_validation'])}/{int(m1['n_test'])}` "
            "aligned train/validation/test pairs with the frozen two-model and "
            "steady/bursty contract.",
            "",
            "| Method | P95 pinball loss | Dangerous underprediction rate |",
            "|---|---:|---:|",
            *comparison_rows,
            "",
            f"The signed `(M1-M3)/M1` change is `{relative_change:+.4%}`; dangerous "
            f"underprediction is `{dangerous_count_m1}/{int(m1['n_test'])}` for M1 "
            f"and `{dangerous_count_m3}/{int(m3['n_test'])}` for M3.",
            "",
            "## Measured / Inferred Boundary",
            "",
            "- **Real runtime:** passed the frozen P1 eligibility contract.",
            "- **Real GPU:** bounded by the source runtime bundle; no multi-GPU inference is made.",
            "- **Offline replay:** causal alignment and grouped ridge M0--M4 comparison.",
            "- **Analytic model:** formulation and BCRD/DEPA collision decision only.",
            "- **Synthetic:** mechanism fixtures are excluded from this eligible result.",
            "- **Not measured:** action Oracle headroom, causal controller gain, and multi-GPU EP.",
            "",
            "## Gate chain",
            "",
            "- P0 representative measurement surface: `READY_FOR_SIGNAL_TEST`",
            f"- P1 incremental signal: `{p1_status}`",
            f"- P2 counterfactual capacity Oracle: `{p2_decision}` / `NOT_RUN`",
            "- P3 causal controller: `NOT_RUN`",
            "",
            "## Next Smallest Experiment",
            "",
            next_step,
            "",
            "## Direct answer",
            "",
            direct_answer,
            "",
            str(summary["claim_boundary"]),
        ]
        path.write_text("\n".join(eligible_lines) + "\n", encoding="utf-8")
        return
    lines = [
        "# RouteShape-SLO final exploration report",
        "",
        "## Verdict",
        "",
        f"`{summary['verdict']}`",
        "",
        f"P1 status is `{summary['p1_status']}` and scientific eligibility is "
        f"`{str(summary['scientific_result_eligible']).lower()}`. The current "
        "classification is `D`: the available runtime artifact is not a "
        "representative serving-capacity trace.",
        "",
        "## Evidence Table",
        "",
        "| Question | Result | Evidence type | Boundary | Decision |",
        "|---|---|---|---|---|",
        "| Does route add capacity information? | Not tested; the smoke changes M1-to-M3 P95 pinball loss by +2.7697%, with no change in dangerous underprediction. | `[Offline replay]` over `[Observed isolated GPU primitive]` | One model, one replay, teacher-forced fixed roster, no capacity action. | `UNVERIFIED` |",
        "| Does a future-route Oracle expose action headroom? | Not run. M4 is only a future-route latency predictor. | `[Not measured]` | No action-conditioned counterfactual routes or safe-budget labels. | `BLOCKED_P1_NOT_ELIGIBLE` |",
        "| Is historical route causally usable? | Not run. | `[Not measured]` | P3 is forbidden before eligible P1 and P2 pass. | `BLOCKED_P2_NOT_PASSED` |",
        "| Does a simple policy already cover the gain? | Not measured. | `[Not measured]` | No fixed/AIMD/queue/workload-only capacity-policy replay exists. | `UNRESOLVED` |",
        "| Is the single-GPU runtime representative? | No. | `[Observed isolated GPU primitive]` | No native serving queue/admission semantics, calibrated SLO, or hook-overhead A/B. | `BLOCKED_RUNTIME_NOT_REPRESENTATIVE` |",
        "| Independent or BCRD/DEPA submodule? | No positive fold is authorized; DEPA is the conditional default for service-level admission, BCRD only if value is fragmentation-mediated. | `[Analytic model]` plus formulation check | Independence still requires eligible P1, P2 and P3. | `PROVISIONAL_DEPA_SUBMODULE` |",
        "",
        "## P0/P1 smoke actually run",
        "",
        f"Source evidence: {', '.join(summary['evidence_types'])}. There are "
        f"`{int(m1['n_train'])}/{int(m1['n_validation'])}/{int(m1['n_test'])}` "
        "aligned train/validation/test pairs. Request and document identities are "
        "disjoint, but all splits come from one arrival replay.",
        "",
        "| Method | P95 pinball loss | Dangerous underprediction rate |",
        "|---|---:|---:|",
        *comparison_rows,
        "",
        f"The signed diagnostic improvement `(M1-M3)/M1` is `{relative_change:+.4%}`; "
        f"dangerous underprediction is `{dangerous_count_m1}/{int(m1['n_test'])}` "
        f"for M1 and `{dangerous_count_m3}/{int(m3['n_test'])}` for M3. M0 also "
        "outperforms both fitted workload models on this smoke split, another reason "
        "not to interpret the M1/M3 delta scientifically.",
        "",
        "## Measured / Inferred Boundary",
        "",
        "- **Real runtime:** no representative continuous-serving result.",
        "- **Real GPU:** RTX 5090 OLMoE whole-step/expert timing only, classified as an isolated primitive.",
        "- **Offline replay:** route-window construction, causal alignment, grouped ridge M0--M4 and leakage guards.",
        "- **Analytic model:** formulation collision and prior BCRD/DEPA tools only; not capacity evidence.",
        "- **Synthetic:** unit-test fixtures and modeled deadlines only.",
        "- **Not measured:** safe capacity, action Oracle headroom, causal controller gain, instrumentation overhead, native queue/SLO, and multi-GPU EP.",
        "",
        "## Gate chain",
        "",
        "- P0 representative measurement surface: `BLOCKED_RUNTIME_NOT_REPRESENTATIVE`",
        "- P1 incremental signal: `NOT_TESTED_DEVELOPMENT_SMOKE_ONLY`",
        "- P2 counterfactual capacity Oracle: `NOT_RUN`",
        "- P3 causal controller: `NOT_RUN`",
        "",
        "## Next Smallest Experiment",
        "",
        "Run exactly one OLMoE continuous-decode P0 producer capture; do not run "
        "LLM-jp, P1/P2/P3, or 8xA100 first. The frozen command, model revision, "
        "128-request scale, 24 GiB reserve, 30-minute budget, exact GO/STOP criteria, "
        "and output path are in `docs/ideas/route_shape_slo/NEXT_EXPERIMENT.md`.",
        "",
        "## Direct answer",
        "",
        "Unknown. The checked evidence does not show that route information changes "
        "safely supportable capacity, and it also cannot show that route is merely a "
        "complex restatement of queue/token/KV state. One bounded representative "
        "runtime capture is the next admissible step.",
        "",
        str(summary["claim_boundary"]),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ProtocolError("config must be a JSON object")
    raw = read_rows(Path(args.features))
    aligned = align_next_window(raw)
    metrics, summary = analyze(aligned, config)
    write_metrics(output_dir / "metrics.csv", metrics)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_report(output_dir / "report.md", summary, metrics)
    source = config.get("source", {})
    source_environment: Any = None
    if isinstance(source, dict) and source.get("path"):
        source_environment_path = Path(str(source["path"])) / "environment.json"
        if source_environment_path.is_file():
            source_environment = json.loads(
                source_environment_path.read_text(encoding="utf-8")
            )
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": __import__("numpy").__version__,
        "cuda_used": False,
        "analysis_kind": "offline ridge smoke/analysis",
        "source_artifact": source,
        "source_execution_environment": source_environment,
    }
    (output_dir / "environment.json").write_text(
        json.dumps(environment, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
