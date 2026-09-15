#!/usr/bin/env python3
"""Document-disjoint CPU probe for pre-call M2 row-safety prediction.

This exploratory analysis reuses the sealed run03 calibration captures.  It
does not execute a model or GPU kernel.  The primary gate is deliberately
fail-closed: a score threshold is selected only from validation unsafe rows,
then any unsafe admission on the held-out document kills this predictor.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import dataclasses
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


SCHEMA = "semanticfence-row-safety-predictability-analysis-v1"
MODEL_SHAPE = "M0_shape_control"
MODEL_INPUT = "M1_input_value"
LAYER_COUNT = 16
EXPERT_COUNT = 64
ROUTE_RANK_COUNT = 8
PROJECTION_COUNT = 32
PROJECTION_SALT = "semanticfence-row-safety-predictor-v1"


class AnalysisError(RuntimeError):
    """The CPU probe cannot produce an interpretable result."""


@dataclass(frozen=True)
class LabelRow:
    row_id: str
    record: Mapping[str, Any]
    safe: bool


@dataclass(frozen=True)
class FittedLogistic:
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    intercept: float
    optimizer: Mapping[str, Any]

    def score(self, values: np.ndarray) -> np.ndarray:
        standardized = (values - self.mean) / self.scale
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            result = standardized @ self.weights + self.intercept
        if not np.isfinite(result).all():
            raise AnalysisError("non-finite model score")
        return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AnalysisError(f"expected a JSON object: {path}")
    return value


def _validate_config(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != "semanticfence-row-safety-predictor-config-v1":
        raise AnalysisError("unexpected predictor config schema")
    if config.get("status") != "FROZEN_PRE_RUN":
        raise AnalysisError("predictor config was not frozen before the run")
    source = config.get("source", {})
    expected = {
        "required_m": 2,
        "expected_rows": 32234,
        "expected_safe_rows": 2768,
        "expected_unsafe_rows": 29466,
        "expected_documents": 8,
        "expected_hidden_size": 2048,
        "expected_repeats": 10,
    }
    if any(source.get(key) != value for key, value in expected.items()):
        raise AnalysisError("frozen source denominators changed")
    model = config.get("model", {})
    if (
        model.get("family") != "l2_logistic_regression"
        or float(model.get("l2", -1)) != 0.01
        or model.get("class_weight") != "balanced_from_current_train_fold"
        or model.get("optimizer") != "scipy_lbfgsb"
        or int(model.get("maximum_iterations", -1)) != 200
        or float(model.get("gradient_tolerance", -1)) != 1e-8
        or model.get("hyperparameter_search") is not False
    ):
        raise AnalysisError("frozen model protocol changed")


def load_m2_labels(path: Path, *, expected_repeats: int) -> tuple[LabelRow, ...]:
    result: list[LabelRow] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if int(value.get("m", -1)) != 2:
                continue
            row_ids = value.get("row_ids")
            records = value.get("row_records")
            repeats = value.get("repeat_row_exact")
            if (
                not isinstance(row_ids, list)
                or len(row_ids) != 2
                or not isinstance(records, list)
                or len(records) != 2
                or not isinstance(repeats, list)
                or len(repeats) != expected_repeats
                or any(not isinstance(repeat, list) or len(repeat) != 2 for repeat in repeats)
            ):
                raise AnalysisError(f"malformed M2 label at line {line_number}")
            for slot, row_id in enumerate(row_ids):
                if not isinstance(row_id, str) or row_id in seen:
                    raise AnalysisError("M2 row identities are missing or duplicated")
                outcomes = [bool(repeat[slot]) for repeat in repeats]
                if any(outcome != outcomes[0] for outcome in outcomes[1:]):
                    raise AnalysisError("M2 row label is not stable across repeats")
                record = records[slot]
                if not isinstance(record, dict):
                    raise AnalysisError("M2 row record is not an object")
                seen.add(row_id)
                result.append(LabelRow(row_id=row_id, record=record, safe=all(outcomes)))
    return tuple(sorted(result, key=lambda value: value.row_id))


def _load_gpu_module(experiment_dir: Path) -> Any:
    name = "semanticfence_gpu_execution"
    cached = sys.modules.get(name)
    if cached is not None:
        return cached
    path = experiment_dir / "gpu_execution.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise AnalysisError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    # The project venv is Python 3.9 while the sealed capture writer used a
    # newer Python and ``dataclass(slots=True)``.  Ignoring only that syntax at
    # import time preserves field definitions; the explicit state restorer
    # then accepts the list state emitted by the frozen slotted dataclass.
    original_dataclass = dataclasses.dataclass
    if sys.version_info < (3, 10):
        def compatible_dataclass(*args: Any, **kwargs: Any) -> Any:
            kwargs.pop("slots", None)
            return original_dataclass(*args, **kwargs)

        dataclasses.dataclass = compatible_dataclass
    try:
        spec.loader.exec_module(module)
    finally:
        dataclasses.dataclass = original_dataclass
    if sys.version_info < (3, 10):
        def restore_capture_state(instance: Any, state: Sequence[Any]) -> None:
            fields = dataclasses.fields(instance)
            if len(state) != len(fields):
                raise AnalysisError("captured-window pickle state width changed")
            for field, value in zip(fields, state):
                object.__setattr__(instance, field.name, value)

        module.CapturedWindow.__setstate__ = restore_capture_state
    return module


def projection_matrix(hidden_size: int, numeric_sha256: str) -> np.ndarray:
    seed_digest = hashlib.sha256(
        (numeric_sha256 + PROJECTION_SALT).encode("utf-8")
    ).digest()
    seed = int.from_bytes(seed_digest[:8], byteorder="big", signed=False)
    generator = np.random.default_rng(seed)
    signs = generator.integers(
        0, 2, size=(hidden_size, PROJECTION_COUNT), dtype=np.int8
    )
    return ((signs.astype(np.float64) * 2.0 - 1.0) / math.sqrt(hidden_size)).astype(
        np.float64
    )


def hidden_features(hidden: Any, projection: np.ndarray) -> np.ndarray:
    import torch

    if hidden.dtype != torch.bfloat16 or hidden.ndim != 1:
        raise AnalysisError("hidden input must be one BF16 row")
    values = hidden.detach().cpu().float().numpy().astype(np.float64, copy=False)
    raw = (
        hidden.detach()
        .contiguous()
        .view(torch.uint16)
        .cpu()
        .numpy()
        .astype(np.uint16, copy=False)
    )
    exponent = ((raw >> 7) & 0xFF).astype(np.int64, copy=False)
    mantissa = raw & 0x7F
    exponent_hist = np.bincount(exponent >> 4, minlength=16).astype(np.float64)
    exponent_hist /= float(values.size)
    absolute = np.abs(values)
    mean_abs = float(absolute.mean())
    summary = np.asarray(
        [
            float(values.mean()),
            mean_abs,
            float(np.sqrt(np.mean(values * values))),
            float(values.std()),
            float(absolute.max()),
            float(np.mean(values > 0)),
            float(np.mean(values == 0)),
            float(np.mean((exponent == 0) & (mantissa != 0))),
            float(abs(values.sum()) / max(absolute.sum(), np.finfo(np.float64).tiny)),
        ],
        dtype=np.float64,
    )
    log_abs = np.log2(np.maximum(absolute, np.finfo(np.float32).tiny))
    quantiles = np.quantile(log_abs, [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
    # NumPy 2.0 on this macOS Accelerate build can leave spurious floating
    # status flags after a finite matmul.  Suppress those flags and validate
    # the actual values explicitly instead of accepting or hiding NaNs/Infs.
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        projected = values @ projection
    result = np.concatenate((summary, quantiles, exponent_hist, projected)).astype(
        np.float64, copy=False
    )
    if not np.isfinite(result).all():
        raise AnalysisError("hidden feature vector contains a non-finite value")
    return result


def feature_names() -> tuple[tuple[str, ...], tuple[str, ...]]:
    shape = (
        tuple(f"layer_{index}" for index in range(LAYER_COUNT))
        + tuple(f"expert_{index}" for index in range(EXPERT_COUNT))
        + tuple(f"route_rank_{index + 1}" for index in range(ROUTE_RANK_COUNT))
        + ("routing_weight",)
    )
    hidden = (
        "hidden_mean",
        "hidden_mean_abs",
        "hidden_rms",
        "hidden_std",
        "hidden_max_abs",
        "hidden_positive_fraction",
        "hidden_zero_fraction",
        "hidden_subnormal_fraction",
        "hidden_cancellation_ratio",
        "hidden_log2_abs_q0",
        "hidden_log2_abs_q10",
        "hidden_log2_abs_q25",
        "hidden_log2_abs_q50",
        "hidden_log2_abs_q75",
        "hidden_log2_abs_q90",
        "hidden_log2_abs_q100",
    ) + tuple(f"hidden_exponent_bin_{index}" for index in range(16)) + tuple(
        f"hidden_projection_{index}" for index in range(PROJECTION_COUNT)
    )
    return shape, shape + hidden


def build_feature_matrices(
    *,
    labels: Sequence[LabelRow],
    capture_path: Path,
    numeric_sha256: str,
    experiment_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[tuple[int, int]], dict[str, Any]]:
    import torch

    gpu = _load_gpu_module(experiment_dir)
    captures = torch.load(capture_path, map_location="cpu", weights_only=False)
    materialized = {row.row_id: row for row in gpu.materialize_routed_rows(captures)}
    if len(materialized) != 32768:
        raise AnalysisError("capture materialization denominator changed")

    projection = projection_matrix(2048, numeric_sha256)
    hidden_cache: dict[str, np.ndarray] = {}
    shape_rows: list[np.ndarray] = []
    input_rows: list[np.ndarray] = []
    outcomes: list[bool] = []
    documents: list[int] = []
    cells: list[tuple[int, int]] = []
    document_sha_by_index: dict[int, str] = {}

    for label in labels:
        row = materialized.get(label.row_id)
        if row is None:
            raise AnalysisError("labeled row is absent from captures")
        record = label.record
        if row.record.identity_payload() != record:
            raise AnalysisError("capture row identity differs from numeric ledger")
        if row.record.hidden_sha256 != gpu.tensor_storage_sha256(row.tensor):
            raise AnalysisError("capture hidden hash does not close")
        layer = int(record["layer"])
        expert = int(record["expert_id"])
        route_rank = int(record["route_rank"])
        document = int(record["document_index"])
        document_sha = str(record["document_sha256"])
        if (
            layer < 0
            or layer >= LAYER_COUNT
            or expert < 0
            or expert >= EXPERT_COUNT
            or route_rank < 1
            or route_rank > ROUTE_RANK_COUNT
        ):
            raise AnalysisError("row categorical feature is outside frozen bounds")
        previous_sha = document_sha_by_index.setdefault(document, document_sha)
        if previous_sha != document_sha:
            raise AnalysisError("document index maps to multiple document hashes")

        shape = np.zeros(LAYER_COUNT + EXPERT_COUNT + ROUTE_RANK_COUNT + 1)
        shape[layer] = 1.0
        shape[LAYER_COUNT + expert] = 1.0
        shape[LAYER_COUNT + EXPERT_COUNT + route_rank - 1] = 1.0
        shape[-1] = float(row.context.routing_weight)
        cached = hidden_cache.get(row.record.hidden_sha256)
        if cached is None:
            cached = hidden_features(row.tensor, projection)
            hidden_cache[row.record.hidden_sha256] = cached
        shape_rows.append(shape)
        input_rows.append(np.concatenate((shape, cached)))
        outcomes.append(bool(label.safe))
        documents.append(document)
        cells.append((layer, expert))

    if sorted(document_sha_by_index) != list(range(8)):
        raise AnalysisError("expected document indices 0 through 7")
    shape_matrix = np.vstack(shape_rows)
    input_matrix = np.vstack(input_rows)
    if not np.isfinite(shape_matrix).all() or not np.isfinite(input_matrix).all():
        raise AnalysisError("feature matrix contains a non-finite value")
    return (
        shape_matrix,
        input_matrix,
        np.asarray(outcomes, dtype=bool),
        np.asarray(documents, dtype=np.int64),
        cells,
        {
            "capture_count": len(captures),
            "materialized_row_count": len(materialized),
            "labeled_row_count": len(labels),
            "unique_hidden_count": len(hidden_cache),
            "document_sha256_by_index": {
                str(key): value for key, value in sorted(document_sha_by_index.items())
            },
        },
    )


def rotating_document_split(documents: np.ndarray, fold: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if fold < 0 or fold >= 8:
        raise ValueError("fold must be in [0, 7]")
    test_document = fold
    validation_document = (fold + 1) % 8
    test = documents == test_document
    validation = documents == validation_document
    train = ~(test | validation)
    if np.any(train & validation) or np.any(train & test) or np.any(validation & test):
        raise AssertionError("document split overlaps")
    return train, validation, test


def fit_logistic(
    values: np.ndarray,
    outcomes: np.ndarray,
    *,
    l2: float,
    maximum_iterations: int,
    gradient_tolerance: float,
) -> FittedLogistic:
    x = np.asarray(values, dtype=np.float64)
    y = np.asarray(outcomes, dtype=np.float64)
    if x.ndim != 2 or y.shape != (x.shape[0],) or set(np.unique(y)) != {0.0, 1.0}:
        raise AnalysisError("training fold must contain both binary classes")
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-12] = 1.0
    zvalues = (x - mean) / scale
    positives = float(y.sum())
    negatives = float(y.size - positives)
    sample_weight = np.where(y == 1, y.size / (2.0 * positives), y.size / (2.0 * negatives))

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        weights = parameters[:-1]
        intercept = parameters[-1]
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            score = zvalues @ weights + intercept
            losses = np.logaddexp(0.0, score) - y * score
            probability = expit(score)
            residual = sample_weight * (probability - y)
            loss = float(
                np.mean(sample_weight * losses)
                + 0.5 * l2 * np.dot(weights, weights)
            )
            gradient = np.empty_like(parameters)
            gradient[:-1] = (zvalues.T @ residual) / y.size + l2 * weights
            gradient[-1] = float(residual.mean())
        if not math.isfinite(loss) or not np.isfinite(gradient).all():
            raise AnalysisError("non-finite logistic objective or gradient")
        return loss, gradient

    initial = np.zeros(x.shape[1] + 1, dtype=np.float64)
    optimized = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": maximum_iterations, "gtol": gradient_tolerance, "ftol": 1e-12},
    )
    if not optimized.success:
        raise AnalysisError(f"logistic optimizer failed: {optimized.message}")
    return FittedLogistic(
        mean=mean,
        scale=scale,
        weights=np.asarray(optimized.x[:-1], dtype=np.float64),
        intercept=float(optimized.x[-1]),
        optimizer={
            "success": bool(optimized.success),
            "status": int(optimized.status),
            "message": str(optimized.message),
            "iterations": int(optimized.nit),
            "function_evaluations": int(optimized.nfev),
            "final_objective": float(optimized.fun),
        },
    )


def fail_closed_threshold(validation_scores: np.ndarray, validation_outcomes: np.ndarray) -> float:
    unsafe = np.asarray(validation_scores)[~np.asarray(validation_outcomes, dtype=bool)]
    if unsafe.size == 0:
        raise AnalysisError("validation fold has no unsafe rows")
    return float(np.nextafter(float(unsafe.max()), math.inf))


def _fold_metrics(
    *,
    fold: int,
    model_name: str,
    values: np.ndarray,
    outcomes: np.ndarray,
    documents: np.ndarray,
    cells: Sequence[tuple[int, int]],
    l2: float,
    maximum_iterations: int,
    gradient_tolerance: float,
) -> dict[str, Any]:
    train, validation, test = rotating_document_split(documents, fold)
    fitted = fit_logistic(
        values[train],
        outcomes[train],
        l2=l2,
        maximum_iterations=maximum_iterations,
        gradient_tolerance=gradient_tolerance,
    )
    validation_scores = fitted.score(values[validation])
    threshold = fail_closed_threshold(validation_scores, outcomes[validation])
    validation_admitted = validation_scores > threshold
    if int(np.sum(validation_admitted & ~outcomes[validation])) != 0:
        raise AssertionError("validation threshold did not fail closed")
    test_scores = fitted.score(values[test])
    test_admitted = test_scores > threshold
    test_indices = np.flatnonzero(test)
    admitted_indices = test_indices[test_admitted]
    true_admissions = int(np.sum(test_admitted & outcomes[test]))
    false_admissions = int(np.sum(test_admitted & ~outcomes[test]))
    safe_rows = int(np.sum(outcomes[test]))
    return {
        "fold": fold,
        "model": model_name,
        "test_document_index": fold,
        "validation_document_index": (fold + 1) % 8,
        "train_document_indices": [index for index in range(8) if index not in {fold, (fold + 1) % 8}],
        "train_rows": int(train.sum()),
        "validation_rows": int(validation.sum()),
        "test_rows": int(test.sum()),
        "validation_threshold": threshold,
        "validation_true_admissions": int(np.sum(validation_admitted & outcomes[validation])),
        "validation_false_admissions": 0,
        "test_safe_rows": safe_rows,
        "test_unsafe_rows": int(test.sum()) - safe_rows,
        "test_true_admissions": true_admissions,
        "test_false_admissions": false_admissions,
        "test_safe_coverage": true_admissions / safe_rows if safe_rows else 0.0,
        "admitted_layer_expert_cells": sorted(
            {f"{cells[index][0]}:{cells[index][1]}" for index in admitted_indices}
        ),
        "optimizer": dict(fitted.optimizer),
    }


def aggregate_model(folds: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(folds) != 8:
        raise AnalysisError("each model must have eight held-out folds")
    true_admissions = sum(int(value["test_true_admissions"]) for value in folds)
    false_admissions = sum(int(value["test_false_admissions"]) for value in folds)
    safe_rows = sum(int(value["test_safe_rows"]) for value in folds)
    unsafe_rows = sum(int(value["test_unsafe_rows"]) for value in folds)
    cells = sorted(
        {
            cell
            for value in folds
            for cell in value.get("admitted_layer_expert_cells", [])
        }
    )
    admitted_documents = [
        int(value["test_document_index"])
        for value in folds
        if int(value["test_true_admissions"]) + int(value["test_false_admissions"]) > 0
    ]
    return {
        "held_out_rows": safe_rows + unsafe_rows,
        "held_out_safe_rows": safe_rows,
        "held_out_unsafe_rows": unsafe_rows,
        "held_out_true_admissions": true_admissions,
        "held_out_false_admissions": false_admissions,
        "held_out_admitted_rows": true_admissions + false_admissions,
        "held_out_safe_coverage": true_admissions / safe_rows if safe_rows else 0.0,
        "admitted_document_indices": admitted_documents,
        "admitted_document_count": len(admitted_documents),
        "admitted_layer_expert_cells": cells,
        "admitted_layer_expert_cell_count": len(cells),
    }


def basic_gate(metrics: Mapping[str, Any], gate: Mapping[str, Any]) -> bool:
    return (
        int(metrics["held_out_false_admissions"]) == 0
        and int(metrics["held_out_true_admissions"]) >= int(gate["minimum_true_admissions"])
        and float(metrics["held_out_safe_coverage"]) >= float(gate["minimum_safe_coverage"])
        and int(metrics["admitted_document_count"]) >= int(gate["minimum_admitted_documents"])
        and int(metrics["admitted_layer_expert_cell_count"])
        >= int(gate["minimum_admitted_layer_expert_cells"])
    )


def decide(
    shape: Mapping[str, Any], input_value: Mapping[str, Any], gate: Mapping[str, Any]
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if int(input_value["held_out_false_admissions"]) > 0:
        reasons.append("input-value model admitted at least one held-out unsafe row")
        return str(gate["any_test_false_admission"]), reasons
    input_basic = basic_gate(input_value, gate)
    shape_basic = basic_gate(shape, gate)
    coverage_gain = float(input_value["held_out_safe_coverage"]) - float(
        shape["held_out_safe_coverage"]
    )
    if not input_basic:
        reasons.append("input-value model did not meet frozen coverage or span gate")
        return str(gate["zero_false_but_below_coverage_or_span"]), reasons
    if coverage_gain < float(gate["minimum_input_over_shape_coverage_gain"]):
        reasons.append("input-value model did not beat shape control by the frozen margin")
        if shape_basic:
            return str(gate["shape_control_meets_gate_without_input_gain"]), reasons
        return str(gate["zero_false_but_below_coverage_or_span"]), reasons
    reasons.append("input-value model passed the exploratory frozen calibration gate")
    return str(gate["input_model_meets_all_gates"]), reasons


def render_markdown(result: Mapping[str, Any]) -> str:
    models = result["models"]
    lines = [
        "# SemanticFence row-safety predictability probe",
        "",
        f"- Decision: `{result['decision']}`",
        f"- Paper result: `{str(result['paper_result']).lower()}`",
        f"- Evidence boundary: `{result['evidence_boundary']}`",
        "",
        "| Model | Held-out FP | Held-out TP | Safe coverage | Documents | Layer/expert cells |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in (MODEL_SHAPE, MODEL_INPUT):
        value = models[name]["aggregate"]
        lines.append(
            "| {name} | {fp} | {tp} | {coverage:.6%} | {docs} | {cells} |".format(
                name=name,
                fp=value["held_out_false_admissions"],
                tp=value["held_out_true_admissions"],
                coverage=value["held_out_safe_coverage"],
                docs=value["admitted_document_count"],
                cells=value["admitted_layer_expert_cell_count"],
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            *[f"- {value}" for value in result["decision_reasons"]],
            "- Thresholds were selected only from validation unsafe scores; each document was tested exactly once.",
            "- This is reused calibration evidence, not fresh generalization or a sound certificate.",
            "- No latency, serving, EP, multi-GPU, or paper claim is authorized.",
            "",
        ]
    )
    return "\n".join(lines)


def analyze(config_path: Path, source_dir: Path) -> dict[str, Any]:
    config = load_json(config_path)
    _validate_config(config)
    source = config["source"]
    complete_path = source_dir / "COMPLETE.json"
    numeric_path = source_dir / "calibration_numeric.jsonl"
    capture_path = source_dir / "calibration_captures.pt"
    observed_hashes = {
        "complete_sha256": sha256_file(complete_path),
        "calibration_numeric_sha256": sha256_file(numeric_path),
        "calibration_captures_sha256": sha256_file(capture_path),
    }
    if any(observed_hashes[key] != source[key] for key in observed_hashes):
        raise AnalysisError("source artifact hash changed after protocol freeze")
    complete = load_json(complete_path)
    if complete.get("status") != "SUCCESS_COMPLETE":
        raise AnalysisError("source run lacks completion authority")

    labels = load_m2_labels(numeric_path, expected_repeats=int(source["expected_repeats"]))
    safe_count = sum(value.safe for value in labels)
    if (
        len(labels) != int(source["expected_rows"])
        or safe_count != int(source["expected_safe_rows"])
        or len(labels) - safe_count != int(source["expected_unsafe_rows"])
    ):
        raise AnalysisError("M2 label denominators changed")
    shape_values, input_values, outcomes, documents, cells, mapping = build_feature_matrices(
        labels=labels,
        capture_path=capture_path,
        numeric_sha256=observed_hashes["calibration_numeric_sha256"],
        experiment_dir=Path(__file__).resolve().parent,
    )
    shape_names, input_names = feature_names()
    if shape_values.shape[1] != len(shape_names) or input_values.shape[1] != len(input_names):
        raise AssertionError("feature name/matrix dimensions disagree")

    model_config = config["model"]
    folds_by_model: dict[str, list[dict[str, Any]]] = {MODEL_SHAPE: [], MODEL_INPUT: []}
    for model_name, values in ((MODEL_SHAPE, shape_values), (MODEL_INPUT, input_values)):
        for fold in range(8):
            folds_by_model[model_name].append(
                _fold_metrics(
                    fold=fold,
                    model_name=model_name,
                    values=values,
                    outcomes=outcomes,
                    documents=documents,
                    cells=cells,
                    l2=float(model_config["l2"]),
                    maximum_iterations=int(model_config["maximum_iterations"]),
                    gradient_tolerance=float(model_config["gradient_tolerance"]),
                )
            )
    models = {
        MODEL_SHAPE: {
            "feature_count": len(shape_names),
            "features": list(shape_names),
            "folds": folds_by_model[MODEL_SHAPE],
            "aggregate": aggregate_model(folds_by_model[MODEL_SHAPE]),
        },
        MODEL_INPUT: {
            "feature_count": len(input_names),
            "features": list(input_names),
            "folds": folds_by_model[MODEL_INPUT],
            "aggregate": aggregate_model(folds_by_model[MODEL_INPUT]),
        },
    }
    decision, reasons = decide(
        models[MODEL_SHAPE]["aggregate"],
        models[MODEL_INPUT]["aggregate"],
        config["decision"],
    )
    return {
        "schema_version": SCHEMA,
        "decision": decision,
        "decision_reasons": reasons,
        "paper_result": False,
        "evidence_boundary": config["evidence_boundary"],
        "source": {
            "run": source["source_run"],
            **observed_hashes,
            "config_sha256": sha256_file(config_path),
            "analysis_script_sha256": sha256_file(Path(__file__).resolve()),
        },
        "integrity": {
            "all_m2_labels_repeat_stable": True,
            "numeric_finiteness_checks_passed": True,
            "duplicate_m2_row_ids": 0,
            "capture_record_identity_mismatches": 0,
            "capture_hidden_hash_mismatches": 0,
            **mapping,
        },
        "protocol": {
            "split": config["split"],
            "model": config["model"],
            "admission": config["admission"],
            "decision_gate": config["decision"],
            "forbidden_features": config["features"]["forbidden"],
        },
        "models": models,
        "claim_boundary": [
            "The probe predicts reused run03 calibration M2 labels with document-disjoint folds.",
            "Partner and slot invariance were qualified only on the separate sampled calibration probes.",
            "A positive result would remain an empirical predictor, not a sound certificate.",
            "No fresh-row, latency, serving, EP, multi-GPU, or paper claim is authorized.",
        ],
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.output_dir.exists():
        raise AnalysisError("output directory already exists")
    result = analyze(args.config.resolve(), args.source_dir.resolve())
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "ANALYSIS.json").write_text(
        json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "ANALYSIS.md").write_text(
        render_markdown(result), encoding="utf-8"
    )
    print(json.dumps({"decision": result["decision"], "output": str(args.output_dir)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
