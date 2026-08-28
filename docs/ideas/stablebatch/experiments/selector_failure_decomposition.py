#!/usr/bin/env python3
"""Pure-CPU retrospective selector failure decomposition for Sparse-C8.

This module deliberately separates action-pre feature transforms from route
outcome labels.  Both input surfaces are already known, so its outputs are
exploratory diagnostics rather than a fresh confirmation.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from docs.ideas.stablebatch.experiments import (
    selectability_policy as static_policy,
)
from docs.ideas.stablebatch.experiments import (
    sparse_c8_stability_budget_policy as action_policy,
)


CONFIG_RELATIVE = Path(
    "docs/ideas/stablebatch/experiments/configs/selector_failure_decomposition_v1.json"
)
LOCK_RELATIVE = Path(
    "docs/ideas/stablebatch/experiments/configs/FROZEN_SELECTOR_FAILURE_DECOMPOSITION_LOCK_V2.json"
)
TEST_RELATIVE = Path(
    "docs/ideas/stablebatch/experiments/test_selector_failure_decomposition.py"
)
SPARSE_POLICY_RELATIVE = Path(
    "docs/ideas/stablebatch/experiments/sparse_c8_stability_budget_policy.py"
)
STATIC_POLICY_RELATIVE = Path(
    "docs/ideas/stablebatch/experiments/selectability_policy.py"
)
TOP_K = 8
NUM_LAYERS = 15
NUM_EXPERTS = 64
RIDGE_ALPHA = 1.0


class DecompositionError(RuntimeError):
    """The frozen retrospective decomposition contract was violated."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise DecompositionError(f"non-object JSONL row at {path}:{line_number}")
            rows.append(value)
    return rows


def write_json_new(path: Path, value: Any) -> None:
    if path.exists():
        raise DecompositionError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(payload)


def write_text_new(path: Path, value: str) -> None:
    if path.exists():
        raise DecompositionError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(value)


def write_jsonl_new(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    if path.exists():
        raise DecompositionError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def fraction_payload(value: Fraction | int) -> dict[str, int | float]:
    exact = value if isinstance(value, Fraction) else Fraction(value, 1)
    return {
        "numerator": exact.numerator,
        "denominator": exact.denominator,
        "float": float(exact),
    }


def exact_from_payload(value: Mapping[str, Any]) -> Fraction:
    return Fraction(int(value["numerator"]), int(value["denominator"]))


def surface_actions(
    cells: Sequence[Mapping[str, Any]], *, top_k: int = TOP_K
) -> list[dict[str, Any]]:
    """Convert raw Sparse-C8 ledgers and derive exact cell/rank labels."""

    actions: list[dict[str, Any]] = []
    seen: set[str] = set()
    for cell in cells:
        identity = str(cell["cell_identity"])
        if identity in seen:
            raise DecompositionError(f"duplicate cell_identity {identity}")
        seen.add(identity)
        experts = list(map(int, cell["expert_ids"]))
        weights = list(map(float, cell["gate_weights"]))
        if len(experts) != top_k or len(weights) != top_k:
            raise DecompositionError(f"cell {identity} is not top-{top_k}")
        raw_actions = cell.get("c8_actions")
        if not isinstance(raw_actions, Mapping) or set(map(str, raw_actions)) != {
            str(rank) for rank in range(top_k)
        }:
            raise DecompositionError(f"cell {identity} lacks exactly ranks 0..{top_k - 1}")
        cell_rows: list[dict[str, Any]] = []
        for rank in range(top_k):
            raw = raw_actions[str(rank)]
            if not isinstance(raw, Mapping):
                raise DecompositionError(f"cell {identity} rank {rank} is not an object")
            recovered = int(raw["route_recovered_count"])
            harmed = int(raw["route_harmed_count"])
            utility = recovered - harmed
            if min(recovered, harmed) < 0:
                raise DecompositionError("route recovered/harmed must be nonnegative")
            if int(raw["route_net_reward"]) != utility:
                raise DecompositionError("route net disagrees with recovered-harmed")
            if "utility" in raw and int(raw["utility"]) != utility:
                raise DecompositionError("stored utility disagrees with route net")
            if "expert_id" in raw and int(raw["expert_id"]) != experts[rank]:
                raise DecompositionError("action expert_id disagrees with cell rank")
            cell_rows.append(
                {
                    "cell_identity": identity,
                    "document_id": str(cell["document_text_sha256"]),
                    "document_index": int(cell.get("document_index", 0)),
                    "layer": int(cell["layer"]),
                    "rank": rank,
                    "expert_id": experts[rank],
                    "gate_weights": weights,
                    "current_layer_topk_cutoff_margin": float(
                        cell["current_layer_topk_cutoff_margin"]
                    ),
                    "recovered": recovered,
                    "harmed": harmed,
                    "utility": utility,
                    "net": utility,
                }
            )
        utility_sum = sum(int(row["utility"]) for row in cell_rows)
        for row in cell_rows:
            row["cell_utility_sum"] = utility_sum
            row["cell_opportunity"] = float(Fraction(utility_sum, top_k))
            row["residual8"] = top_k * int(row["utility"]) - utility_sum
        if sum(int(row["residual8"]) for row in cell_rows) != 0:
            raise DecompositionError(f"residual8 closure failed for {identity}")
        actions.extend(cell_rows)
    return sorted(actions, key=lambda row: (str(row["cell_identity"]), int(row["rank"])))


def fit_action_feature_schema(
    train_actions: Sequence[Mapping[str, Any]],
    *,
    num_layers: int = NUM_LAYERS,
    num_experts: int = NUM_EXPERTS,
    top_k: int = TOP_K,
) -> dict[str, Any]:
    """Fit only action-pre continuous normalizers on training actions."""

    if not train_actions:
        raise DecompositionError("empty feature training set")
    continuous = [action_policy._continuous_values(row, top_k) for row in train_actions]
    means: dict[str, float] = {}
    stds: dict[str, float] = {}
    for name in action_policy.CONTINUOUS_FEATURES:
        values = np.asarray([row[name] for row in continuous], dtype=np.float64)
        means[name] = float(np.mean(values))
        observed = float(np.std(values, ddof=0))
        stds[name] = observed if observed > 0.0 else 1.0
    return {
        "schema_version": "selector-failure-action-pre-feature-schema-v1",
        "num_layers": int(num_layers),
        "num_experts": int(num_experts),
        "top_k": int(top_k),
        "feature_names": action_policy._feature_names(num_layers, num_experts, top_k),
        "continuous_mean": means,
        "continuous_std": stds,
        "training_action_count": len(train_actions),
        "outcome_derived_features": [],
    }


def transform_action_features(
    actions: Sequence[Mapping[str, Any]], schema: Mapping[str, Any]
) -> np.ndarray:
    """Apply a frozen action-pre schema without mutating or refitting it."""

    if not actions:
        return np.empty((0, len(schema["feature_names"])), dtype=np.float64)
    matrix = np.vstack([action_policy._feature_vector(row, schema) for row in actions])
    if not np.all(np.isfinite(matrix)):
        raise DecompositionError("non-finite transformed action feature")
    return matrix


def fit_ridge(
    matrix: np.ndarray,
    target: np.ndarray,
    *,
    alpha: float = RIDGE_ALPHA,
    fit_intercept: bool,
) -> dict[str, Any]:
    """Fit fixed-alpha ridge with an explicitly unpenalized optional intercept."""

    x = np.asarray(matrix, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    if x.ndim != 2 or y.shape != (x.shape[0],) or x.shape[0] == 0:
        raise DecompositionError("invalid ridge matrix/target shape")
    if alpha <= 0 or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise DecompositionError("invalid ridge values")
    design = np.column_stack([np.ones(x.shape[0]), x]) if fit_intercept else x
    penalty = np.eye(design.shape[1], dtype=np.float64) * float(alpha)
    if fit_intercept:
        penalty[0, 0] = 0.0
    # NumPy/Accelerate on macOS can leave stale floating-point status flags and
    # emit spurious matmul warnings even when every operand/result is finite.
    # Suppress backend flags here, then enforce explicit finite-result checks.
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        gram = design.T @ design
        right_hand_side = design.T @ y
        coefficients = np.linalg.solve(gram + penalty, right_hand_side)
    if not np.all(np.isfinite(gram)) or not np.all(np.isfinite(right_hand_side)):
        raise DecompositionError("non-finite ridge normal equations")
    if not np.all(np.isfinite(coefficients)):
        raise DecompositionError("non-finite ridge coefficients")
    intercept = float(coefficients[0]) if fit_intercept else 0.0
    weights = coefficients[1:] if fit_intercept else coefficients
    return {
        "schema_version": "selector-failure-ridge-v1",
        "alpha": float(alpha),
        "fit_intercept": bool(fit_intercept),
        "intercept_penalized": False if fit_intercept else None,
        "intercept": intercept,
        "coefficients": weights.tolist(),
        "feature_count": x.shape[1],
        "training_row_count": x.shape[0],
    }


def predict_ridge(matrix: np.ndarray, model: Mapping[str, Any]) -> np.ndarray:
    x = np.asarray(matrix, dtype=np.float64)
    weights = np.asarray(model["coefficients"], dtype=np.float64)
    if x.ndim != 2 or weights.shape != (x.shape[1],):
        raise DecompositionError("ridge prediction shape mismatch")
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        result = x @ weights + float(model["intercept"])
    if not np.all(np.isfinite(result)):
        raise DecompositionError("non-finite ridge prediction")
    return result


def fit_hierarchical_profile(
    actions: Sequence[Mapping[str, Any]], *, shrinkage_lambda: float = 4.0
) -> dict[str, Any]:
    """Fit the pre-existing hierarchy to integer residual8 labels."""

    if any(not isinstance(row.get("residual8"), int) for row in actions):
        raise DecompositionError("profile requires exact integer residual8 labels")
    candidates = [{**dict(row), "reward": int(row["residual8"])} for row in actions]
    model = static_policy.fit_static_map(candidates, shrinkage_lambda)
    model = dict(model)
    model.pop("parameter_content_sha256", None)
    model["schema_version"] = "hierarchical-static-rank-profile-v1"
    model["label"] = "residual8=8*utility-sum_rank_utility"
    model["claim_boundary"] = "profile-aware rank selection"
    model["parameter_content_sha256"] = static_policy.content_sha256(model)
    return model


def profile_score(row: Mapping[str, Any], model: Mapping[str, Any]) -> dict[str, Any]:
    return static_policy.static_score(row, model)


def select_ranks(
    scored_actions: Sequence[Mapping[str, Any]],
    *,
    score_key: str,
    maximize: bool = True,
    cell_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Choose exactly one rank per requested cell, tie by smallest rank."""

    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in scored_actions:
        identity = str(row["cell_identity"])
        if cell_ids is None or identity in cell_ids:
            grouped[identity].append(row)
    if cell_ids is not None and set(grouped) != set(cell_ids):
        raise DecompositionError("requested cells are missing from rank scores")
    selected: list[dict[str, Any]] = []
    for identity in sorted(grouped):
        rows = grouped[identity]
        if len(rows) != TOP_K or {int(row["rank"]) for row in rows} != set(range(TOP_K)):
            raise DecompositionError(f"cell {identity} does not contain exactly eight ranks")
        for row in rows:
            if not math.isfinite(float(row[score_key])):
                raise DecompositionError(f"non-finite {score_key}")
        if maximize:
            winner = min(rows, key=lambda row: (-float(row[score_key]), int(row["rank"])))
        else:
            winner = min(rows, key=lambda row: (float(row[score_key]), int(row["rank"])))
        selected.append(dict(winner))
    return selected


def _action_lookup(
    actions: Sequence[Mapping[str, Any]],
) -> tuple[dict[tuple[str, int], dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    lookup: dict[tuple[str, int], dict[str, Any]] = {}
    by_cell: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for raw in actions:
        row = dict(raw)
        key = (str(row["cell_identity"]), int(row["rank"]))
        if key in lookup:
            raise DecompositionError(f"duplicate action {key}")
        lookup[key] = row
        by_cell[key[0]].append(row)
    for identity, rows in by_cell.items():
        if len(rows) != TOP_K or {int(row["rank"]) for row in rows} != set(range(TOP_K)):
            raise DecompositionError(f"cell {identity} action surface is incomplete")
    return lookup, by_cell


def _aggregate_exact(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    recovered = sum(int(row["recovered"]) for row in rows)
    harmed = sum(int(row["harmed"]) for row in rows)
    return {
        "actions": len(rows),
        "recovered": recovered,
        "harmed": harmed,
        "net": recovered - harmed,
    }


def _expectation(rows: Sequence[Mapping[str, Any]], denominator: int) -> dict[str, Any]:
    if denominator <= 0 or len(rows) % denominator:
        raise DecompositionError("expected action count is not integral")
    recovered = Fraction(sum(int(row["recovered"]) for row in rows), denominator)
    harmed = Fraction(sum(int(row["harmed"]) for row in rows), denominator)
    return {
        "actions": len(rows) // denominator,
        "recovered": fraction_payload(recovered),
        "harmed": fraction_payload(harmed),
        "net": fraction_payload(recovered - harmed),
    }


def evaluate_exact_budget(
    actions: Sequence[Mapping[str, Any]],
    selected: Sequence[Mapping[str, Any]],
    *,
    budget: int,
) -> dict[str, Any]:
    """Compute both exact random baselines for one exact-B rank policy."""

    lookup, by_cell = _action_lookup(actions)
    if len(selected) != budget:
        raise DecompositionError("selector does not contain exact budget")
    selected_cells = [str(row["cell_identity"]) for row in selected]
    if len(set(selected_cells)) != budget:
        raise DecompositionError("selector must choose distinct cells")
    chosen: list[dict[str, Any]] = []
    for row in selected:
        key = (str(row["cell_identity"]), int(row["rank"]))
        if key not in lookup:
            raise DecompositionError(f"selected action absent from surface: {key}")
        chosen.append(lookup[key])
    all_rows = [row for rows in by_cell.values() for row in rows]
    global_scale = Fraction(budget, len(by_cell) * TOP_K)
    global_recovered = global_scale * sum(int(row["recovered"]) for row in all_rows)
    global_harmed = global_scale * sum(int(row["harmed"]) for row in all_rows)
    matched_rows = [row for identity in selected_cells for row in by_cell[identity]]
    matched = _expectation(matched_rows, TOP_K)
    selector = _aggregate_exact(chosen)
    return {
        "budget": budget,
        "cell_count": len(by_cell),
        "global_matched_random": {
            "actions": budget,
            "recovered": fraction_payload(global_recovered),
            "harmed": fraction_payload(global_harmed),
            "net": fraction_payload(global_recovered - global_harmed),
        },
        "cell_matched_random_rank": matched,
        "selector": selector,
        "selector_selected": [
            {"cell_identity": row["cell_identity"], "rank": int(row["rank"])}
            for row in chosen
        ],
        "cell_selection_gain": fraction_payload(
            exact_from_payload(matched["net"])
            - (global_recovered - global_harmed)
        ),
        "rank_selection_gain": fraction_payload(
            Fraction(int(selector["net"]), 1) - exact_from_payload(matched["net"])
        ),
    }


def _group_indices(actions: Sequence[Mapping[str, Any]]) -> dict[str, list[int]]:
    result: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(actions):
        result[str(row["cell_identity"])].append(index)
    for identity, indices in result.items():
        if len(indices) != TOP_K:
            raise DecompositionError(f"cell {identity} does not have eight feature rows")
    return dict(result)


def _design_matrices(
    actions: Sequence[Mapping[str, Any]], schema: Mapping[str, Any]
) -> dict[str, Any]:
    full = transform_action_features(actions, schema)
    raw = full[:, 1:]
    raw_names = list(schema["feature_names"])[1:]
    by_cell = _group_indices(actions)
    cell_ids = sorted(by_cell)
    means: dict[str, np.ndarray] = {
        identity: np.mean(raw[indices], axis=0) for identity, indices in by_cell.items()
    }
    rank_centered = np.vstack(
        [raw[index] - means[str(row["cell_identity"])] for index, row in enumerate(actions)]
    )
    cell_mask = np.asarray([not name.startswith("rank_") for name in raw_names])
    cell_matrix = np.vstack([means[identity][cell_mask] for identity in cell_ids])
    return {
        "raw_action_matrix": raw,
        "rank_centered_matrix": rank_centered,
        "cell_matrix": cell_matrix,
        "cell_ids": cell_ids,
        "cell_feature_names": [name for name, keep in zip(raw_names, cell_mask) if keep],
        "action_feature_names": raw_names,
    }


def fit_models(train_actions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    schema = fit_action_feature_schema(train_actions)
    design = _design_matrices(train_actions, schema)
    by_cell = _action_lookup(train_actions)[1]
    cell_target = np.asarray(
        [float(Fraction(sum(int(row["utility"]) for row in by_cell[identity]), TOP_K)) for identity in design["cell_ids"]],
        dtype=np.float64,
    )
    rank_target = np.asarray([int(row["residual8"]) for row in train_actions], dtype=np.float64)
    harm_target = np.asarray([int(row["harmed"]) for row in train_actions], dtype=np.float64)
    return {
        "schema_version": "selector-failure-model-bundle-v1",
        "feature_schema": schema,
        "cell_feature_names": design["cell_feature_names"],
        "action_feature_names": design["action_feature_names"],
        "cell_head": fit_ridge(
            design["cell_matrix"], cell_target, alpha=RIDGE_ALPHA, fit_intercept=True
        ),
        "rank_residual_ridge": fit_ridge(
            design["rank_centered_matrix"],
            rank_target,
            alpha=RIDGE_ALPHA,
            fit_intercept=False,
        ),
        "harm_head": fit_ridge(
            design["raw_action_matrix"], harm_target, alpha=RIDGE_ALPHA, fit_intercept=True
        ),
        "hierarchical_static_rank_profile": fit_hierarchical_profile(
            train_actions, shrinkage_lambda=4.0
        ),
        "training_cell_opportunity_mean": float(np.mean(cell_target)),
        "training_harm_mean": float(np.mean(harm_target)),
        "training_document_count": len({str(row["document_id"]) for row in train_actions}),
        "training_cell_count": len(by_cell),
        "training_action_count": len(train_actions),
    }


def score_models(
    actions: Sequence[Mapping[str, Any]], bundle: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Score from action-pre fields, then attach outcomes only for evaluation."""

    design = _design_matrices(actions, bundle["feature_schema"])
    cell_prediction_values = predict_ridge(design["cell_matrix"], bundle["cell_head"])
    cell_predictions = dict(zip(design["cell_ids"], map(float, cell_prediction_values)))
    rank_predictions = predict_ridge(
        design["rank_centered_matrix"], bundle["rank_residual_ridge"]
    )
    harm_predictions = predict_ridge(design["raw_action_matrix"], bundle["harm_head"])
    scored: list[dict[str, Any]] = []
    for index, raw in enumerate(actions):
        row = dict(raw)
        profile = profile_score(row, bundle["hierarchical_static_rank_profile"])
        scored.append(
            {
                **row,
                "predicted_cell_opportunity": cell_predictions[str(row["cell_identity"])],
                "predicted_residual8": float(rank_predictions[index]),
                "predicted_rank_residual": float(rank_predictions[index] / TOP_K),
                "predicted_harm": float(harm_predictions[index]),
                "profile_residual8_score": float(profile["score"]),
                "profile_rank_score": float(profile["score"] / TOP_K),
                "profile_exact_support": int(profile["exact_support"]),
                "cell_baseline_prediction": float(bundle["training_cell_opportunity_mean"]),
                "harm_baseline_prediction": float(bundle["training_harm_mean"]),
            }
        )
    return scored


def lodo_predictions(actions: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    documents = sorted({str(row["document_id"]) for row in actions})
    if len(documents) != 16:
        raise DecompositionError("broad LODO requires exactly 16 documents")
    result: list[dict[str, Any]] = []
    for held_document in documents:
        train = [row for row in actions if str(row["document_id"]) != held_document]
        held = [row for row in actions if str(row["document_id"]) == held_document]
        bundle = fit_models(train)
        scored = score_models(held, bundle)
        for row in scored:
            row["fold_held_document_id"] = held_document
            row["fold_training_document_count"] = 15
        result.extend(scored)
    return sorted(result, key=lambda row: (str(row["cell_identity"]), int(row["rank"])))


def regression_metrics(
    actual: Sequence[float], predicted: Sequence[float], baseline: Sequence[float]
) -> dict[str, Any]:
    y = np.asarray(actual, dtype=np.float64)
    p = np.asarray(predicted, dtype=np.float64)
    b = np.asarray(baseline, dtype=np.float64)
    if y.shape != p.shape or y.shape != b.shape or y.size == 0:
        raise DecompositionError("invalid regression metric vectors")
    error = p - y
    baseline_error = b - y
    mse = float(np.mean(error**2))
    baseline_mse = float(np.mean(baseline_error**2))
    if np.std(y) > 0.0 and np.std(p) > 0.0:
        pearson = float(np.corrcoef(y, p)[0, 1])
    else:
        pearson = None
    return {
        "count": int(y.size),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(math.sqrt(mse)),
        "mse": mse,
        "baseline_mae": float(np.mean(np.abs(baseline_error))),
        "baseline_rmse": float(math.sqrt(baseline_mse)),
        "baseline_mse": baseline_mse,
        "mse_skill": float(1.0 - mse / baseline_mse) if baseline_mse > 0.0 else None,
        "pearson": pearson,
    }


def _median_fraction(values: Sequence[Fraction]) -> Fraction:
    ordered = sorted(values)
    if not ordered:
        return Fraction(0, 1)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def _strategy_document_gain(
    by_cell: Mapping[str, Sequence[Mapping[str, Any]]],
    chosen: Sequence[Mapping[str, Any]],
    all_documents: Sequence[str],
) -> dict[str, Any]:
    chosen_by_cell = {str(row["cell_identity"]): row for row in chosen}
    values: dict[str, Fraction] = {document: Fraction(0, 1) for document in all_documents}
    for identity, picked in chosen_by_cell.items():
        rows = by_cell[identity]
        document = str(rows[0]["document_id"])
        uniform = Fraction(sum(int(row["utility"]) for row in rows), TOP_K)
        values[document] += Fraction(int(picked["utility"]), 1) - uniform
    ordered_values = [values[document] for document in all_documents]
    return {
        "positive_document_count": sum(value > 0 for value in ordered_values),
        "zero_document_count": sum(value == 0 for value in ordered_values),
        "negative_document_count": sum(value < 0 for value in ordered_values),
        "median_gain": fraction_payload(_median_fraction(ordered_values)),
        "per_document": {
            document: fraction_payload(values[document]) for document in all_documents
        },
    }


def evaluate_scored_surface(
    scored_actions: Sequence[Mapping[str, Any]], *, budget: int
) -> dict[str, Any]:
    lookup, by_cell = _action_lookup(scored_actions)
    cell_scores: dict[str, float] = {}
    for identity, rows in by_cell.items():
        values = {float(row["predicted_cell_opportunity"]) for row in rows}
        if len(values) != 1:
            raise DecompositionError("cell-head prediction differs within a cell")
        cell_scores[identity] = values.pop()
    if budget <= 0 or budget > len(by_cell):
        raise DecompositionError("budget outside cell population")
    selected_cells = [
        identity
        for identity, _ in sorted(cell_scores.items(), key=lambda item: (-item[1], item[0]))[:budget]
    ]
    selected_set = set(selected_cells)
    ridge = select_ranks(
        scored_actions,
        score_key="predicted_residual8",
        maximize=True,
        cell_ids=selected_set,
    )
    profile = select_ranks(
        scored_actions,
        score_key="profile_residual8_score",
        maximize=True,
        cell_ids=selected_set,
    )
    min_harm = select_ranks(
        scored_actions,
        score_key="predicted_harm",
        maximize=False,
        cell_ids=selected_set,
    )
    ridge_all = select_ranks(scored_actions, score_key="predicted_residual8", maximize=True)
    profile_all = select_ranks(
        scored_actions, score_key="profile_residual8_score", maximize=True
    )
    harm_all = select_ranks(scored_actions, score_key="predicted_harm", maximize=False)

    exact_ridge = evaluate_exact_budget(scored_actions, ridge, budget=budget)
    exact_profile = evaluate_exact_budget(scored_actions, profile, budget=budget)
    exact_harm = evaluate_exact_budget(scored_actions, min_harm, budget=budget)
    cell_random_net = exact_from_payload(exact_ridge["cell_matched_random_rank"]["net"])
    cell_random_harm = exact_from_payload(exact_ridge["cell_matched_random_rank"]["harmed"])
    selected_oracle_net = sum(
        max(int(row["utility"]) for row in by_cell[identity]) for identity in selected_cells
    )
    headroom = Fraction(selected_oracle_net, 1) - cell_random_net

    all_uniform_net = Fraction(
        sum(int(row["utility"]) for rows in by_cell.values() for row in rows), TOP_K
    )
    all_uniform_harm = Fraction(
        sum(int(row["harmed"]) for rows in by_cell.values() for row in rows), TOP_K
    )

    def rank_summary(name: str, rows: Sequence[Mapping[str, Any]], exact: Mapping[str, Any]) -> dict[str, Any]:
        gain = Fraction(int(exact["selector"]["net"]), 1) - cell_random_net
        return {
            "name": name,
            "outcome": exact["selector"],
            "rank_gain": fraction_payload(gain),
            "rank_headroom": fraction_payload(headroom),
            "rank_headroom_capture": fraction_payload(gain / headroom) if headroom > 0 else None,
            "selected_rank_histogram": [
                sum(int(int(row["rank"]) == rank) for row in rows) for rank in range(TOP_K)
            ],
        }

    documents = sorted({str(row["document_id"]) for row in scored_actions})
    cell_rows = [by_cell[identity][0] for identity in sorted(by_cell)]
    cell_actual = [float(Fraction(sum(int(row["utility"]) for row in by_cell[str(cell["cell_identity"])]), TOP_K)) for cell in cell_rows]
    cell_predicted = [float(cell["predicted_cell_opportunity"]) for cell in cell_rows]
    cell_baseline = [float(cell["cell_baseline_prediction"]) for cell in cell_rows]
    residual_actual = [float(Fraction(int(row["residual8"]), TOP_K)) for row in scored_actions]
    residual_predicted = [float(row["predicted_rank_residual"]) for row in scored_actions]
    profile_predicted = [float(row["profile_rank_score"]) for row in scored_actions]
    harm_actual = [float(row["harmed"]) for row in scored_actions]
    harm_predicted = [float(row["predicted_harm"]) for row in scored_actions]
    harm_baseline = [float(row["harm_baseline_prediction"]) for row in scored_actions]

    harm_avoidance = cell_random_harm - Fraction(int(exact_harm["selector"]["harmed"]), 1)
    all_policy: dict[str, Any] = {}
    for name, rows in (
        ("rank_residual_ridge", ridge_all),
        ("hierarchical_static_rank_profile", profile_all),
    ):
        actual = sum(int(row["utility"]) for row in rows)
        all_policy[name] = {
            "actions": len(rows),
            "net": actual,
            "rank_gain": fraction_payload(Fraction(actual, 1) - all_uniform_net),
            "document_gain": _strategy_document_gain(by_cell, rows, documents),
        }
    all_harm_value = sum(int(row["harmed"]) for row in harm_all)
    all_policy["harm_head"] = {
        "actions": len(harm_all),
        "harmed": all_harm_value,
        "harm_avoidance": fraction_payload(all_uniform_harm - all_harm_value),
    }

    result = {
        "cell_count": len(by_cell),
        "action_count": len(scored_actions),
        "document_count": len(documents),
        "budget": budget,
        "selected_cells": selected_cells,
        "baselines": {
            "global_matched_random": exact_ridge["global_matched_random"],
            "cell_matched_random_rank": exact_ridge["cell_matched_random_rank"],
        },
        "cell_head": {
            "cell_selection_gain": exact_ridge["cell_selection_gain"],
            "prediction": regression_metrics(cell_actual, cell_predicted, cell_baseline),
        },
        "rank_residual_ridge": {
            **rank_summary("RankResidualRidge-v1", ridge, exact_ridge),
            "prediction": regression_metrics(
                residual_actual, residual_predicted, [0.0] * len(residual_actual)
            ),
            "document_gain": _strategy_document_gain(by_cell, ridge, documents),
        },
        "hierarchical_static_rank_profile": {
            **rank_summary("HierarchicalStaticRankProfile-v1", profile, exact_profile),
            "prediction": regression_metrics(
                residual_actual, profile_predicted, [0.0] * len(residual_actual)
            ),
            "document_gain": _strategy_document_gain(by_cell, profile, documents),
        },
        "harm_head": {
            "name": "HarmHead-v1",
            "outcome": exact_harm["selector"],
            "prediction": regression_metrics(harm_actual, harm_predicted, harm_baseline),
            "harm_random": exact_ridge["cell_matched_random_rank"]["harmed"],
            "harm_avoidance": fraction_payload(harm_avoidance),
            "net_vs_cell_random": exact_harm["rank_selection_gain"],
            "document_gain": _strategy_document_gain(by_cell, min_harm, documents),
        },
        "all_cell_rank_views": {
            "uniform_rank_net": fraction_payload(all_uniform_net),
            "uniform_rank_harmed": fraction_payload(all_uniform_harm),
            **all_policy,
        },
        "selected_actions": {
            "rank_residual_ridge": [
                {"cell_identity": row["cell_identity"], "rank": int(row["rank"])} for row in ridge
            ],
            "hierarchical_static_rank_profile": [
                {"cell_identity": row["cell_identity"], "rank": int(row["rank"])} for row in profile
            ],
            "harm_head": [
                {"cell_identity": row["cell_identity"], "rank": int(row["rank"])} for row in min_harm
            ],
        },
    }
    harm_skill = result["harm_head"]["prediction"]["mse_skill"]
    result["harm_head"]["effective"] = bool(
        harm_skill is not None and float(harm_skill) > 0.0 and harm_avoidance > 0
    )
    return result


def validate_surface(
    rows: Sequence[Mapping[str, Any]], actions: Sequence[Mapping[str, Any]], spec: Mapping[str, Any]
) -> dict[str, Any]:
    expected_cells = int(spec["expected_cells"])
    expected_actions = int(spec["expected_actions"])
    expected_documents = int(spec["expected_documents"])
    if len(rows) != expected_cells or len(actions) != expected_actions:
        raise DecompositionError("surface cardinality differs from frozen config")
    documents = {str(row["document_text_sha256"]) for row in rows}
    if len(documents) != expected_documents:
        raise DecompositionError("surface document count differs from frozen config")
    for document in documents:
        layers = sorted(
            int(row["layer"]) for row in rows if str(row["document_text_sha256"]) == document
        )
        if layers != list(range(NUM_LAYERS)):
            raise DecompositionError("document does not contain exactly layers 0..14")
    if any(str(row.get("integrity_status", "PASS")) != "PASS" for row in rows):
        raise DecompositionError("surface contains non-PASS cell")
    return {
        "documents": sorted(documents),
        "document_count": len(documents),
        "cell_count": len(rows),
        "action_count": len(actions),
        "positive_action_count": sum(int(int(row["utility"]) > 0) for row in actions),
        "zero_action_count": sum(int(int(row["utility"]) == 0) for row in actions),
        "negative_action_count": sum(int(int(row["utility"]) < 0) for row in actions),
        "residual8_closure": all(
            sum(int(row["residual8"]) for row in actions if row["cell_identity"] == identity) == 0
            for identity in {str(row["cell_identity"]) for row in actions}
        ),
    }


def validate_config(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != "stablebatch-selector-failure-decomposition-v1":
        raise DecompositionError("wrong config schema")
    if config.get("status") != "FROZEN_BEFORE_NEW_DECOMPOSITION_COMPUTATION":
        raise DecompositionError("config is not frozen")
    if config.get("evidence_class") != "EXPLORATORY_POST_HOC_DECOMPOSITION":
        raise DecompositionError("wrong evidence class")
    if int(config["action_space"]["budget_B"]) != 33:
        raise DecompositionError("budget must be 33")
    if list(map(int, config["action_space"]["candidate_ranks"])) != list(range(TOP_K)):
        raise DecompositionError("candidate ranks must be 0..7")
    if float(config["heads"]["cell_head"]["alpha"]) != RIDGE_ALPHA:
        raise DecompositionError("cell ridge alpha drifted")
    if float(config["heads"]["rank_residual_ridge"]["alpha"]) != RIDGE_ALPHA:
        raise DecompositionError("rank ridge alpha drifted")
    if float(config["heads"]["harm_head"]["alpha"]) != RIDGE_ALPHA:
        raise DecompositionError("harm ridge alpha drifted")
    if float(config["heads"]["hierarchical_static_rank_profile"]["shrinkage_lambda"]) != 4.0:
        raise DecompositionError("profile lambda drifted")
    if not bool(config["execution"]["cpu_only"]) or bool(config["execution"]["gpu_allowed"]):
        raise DecompositionError("execution must remain CPU-only")


def frozen_file_paths(repo_root: Path, config: Mapping[str, Any]) -> list[Path]:
    return [
        CONFIG_RELATIVE,
        Path("docs/ideas/stablebatch/experiments/selector_failure_decomposition.py"),
        TEST_RELATIVE,
        SPARSE_POLICY_RELATIVE,
        STATIC_POLICY_RELATIVE,
        Path(config["inputs"]["train_surface"]["path"]),
        Path(config["inputs"]["evaluation_surface"]["path"]),
    ]


def freeze_lock(repo_root: Path, config_path: Path, lock_path: Path) -> dict[str, Any]:
    config = load_json(config_path)
    validate_config(config)
    if lock_path.exists():
        raise DecompositionError(f"refusing to overwrite frozen lock {lock_path}")
    files: dict[str, str] = {}
    for relative in frozen_file_paths(repo_root, config):
        path = repo_root / relative
        if not path.is_file():
            raise DecompositionError(f"missing frozen file {relative}")
        files[str(relative)] = sha256_file(path)
    for role in ("train_surface", "evaluation_surface"):
        expected = str(config["inputs"][role]["sha256"])
        observed = files[str(config["inputs"][role]["path"])]
        if expected != observed:
            raise DecompositionError(f"{role} hash differs from frozen config")
    lock = {
        "schema_version": "frozen-selector-failure-decomposition-lock-v1",
        "status": "FROZEN_PRE_COMPUTE_RETROSPECTIVE_INPUTS_ALREADY_OBSERVED",
        "created_at": utc_now(),
        "evidence_class": config["evidence_class"],
        "config_sha256": sha256_file(config_path),
        "frozen_semantics_sha256": hashlib.sha256(canonical_json_bytes(config)).hexdigest(),
        "files": files,
    }
    write_json_new(lock_path, lock)
    return lock


def verify_lock(repo_root: Path, config: Mapping[str, Any], lock_path: Path) -> dict[str, Any]:
    lock = load_json(lock_path)
    if lock.get("schema_version") != "frozen-selector-failure-decomposition-lock-v1":
        raise DecompositionError("wrong lock schema")
    if lock.get("status") != "FROZEN_PRE_COMPUTE_RETROSPECTIVE_INPUTS_ALREADY_OBSERVED":
        raise DecompositionError("lock status is not frozen")
    if lock.get("frozen_semantics_sha256") != hashlib.sha256(canonical_json_bytes(config)).hexdigest():
        raise DecompositionError("frozen config semantics drifted")
    expected_paths = {str(path) for path in frozen_file_paths(repo_root, config)}
    if set(lock["files"]) != expected_paths:
        raise DecompositionError("lock file set drifted")
    for relative, expected in lock["files"].items():
        if sha256_file(repo_root / relative) != str(expected):
            raise DecompositionError(f"locked file hash mismatch: {relative}")
    return lock


def _effect_sign(value: Fraction) -> int:
    return int(value > 0) - int(value < 0)


def transfer_summary(broad: Mapping[str, Any], fresh: Mapping[str, Any]) -> dict[str, Any]:
    paths = {
        "cell_selection_gain": ("cell_head", "cell_selection_gain"),
        "rank_residual_ridge_gain": ("rank_residual_ridge", "rank_gain"),
        "profile_rank_gain": ("hierarchical_static_rank_profile", "rank_gain"),
        "harm_avoidance": ("harm_head", "harm_avoidance"),
    }
    effects: dict[str, Any] = {}
    for name, (section, metric) in paths.items():
        broad_value = exact_from_payload(broad[section][metric])
        fresh_value = exact_from_payload(fresh[section][metric])
        effects[name] = {
            "broad_lodo": fraction_payload(broad_value),
            "fresh": fraction_payload(fresh_value),
            "sign_retained": _effect_sign(broad_value) == _effect_sign(fresh_value),
        }
    return {"effects": effects}


def render_result(summary: Mapping[str, Any]) -> str:
    broad = summary["broad_lodo"]
    fresh = summary["fresh"]
    decision = summary["decision"]

    def f(payload: Mapping[str, Any]) -> str:
        return f"{float(payload['float']):+.6g}"

    profile_gain = fresh["hierarchical_static_rank_profile"]["rank_gain"]
    lines = [
        "# Selector Failure Decomposition — exploratory result",
        "",
        f"## Decision: `{decision['selected_policy']}`",
        "",
        "本轮复用两套已经看过 outcome 的完整 C8 surface，仅做纯 CPU 回顾性 failure decomposition；它不能确认新 policy。",
        "",
        "## Exact decomposition",
        "",
        "`u=recovered-harmed`, `mu_c=mean_rank(u)`, `delta_c,r=u_c,r-mu_c`。实现用整数 `residual8=8*u-sum_rank(u)`，所有 cell 均通过零和闭合检查。",
        "",
        "| Effect | broad 16-fold LODO | fresh transfer |",
        "|---|---:|---:|",
        f"| Cell-head selection gain | {f(broad['cell_head']['cell_selection_gain'])} | {f(fresh['cell_head']['cell_selection_gain'])} |",
        f"| Rank-residual ridge gain | {f(broad['rank_residual_ridge']['rank_gain'])} | {f(fresh['rank_residual_ridge']['rank_gain'])} |",
        f"| Hierarchical profile rank gain | {f(broad['hierarchical_static_rank_profile']['rank_gain'])} | {f(profile_gain)} |",
        f"| Harm-head exact harm avoidance | {f(broad['harm_head']['harm_avoidance'])} | {f(fresh['harm_head']['harm_avoidance'])} |",
        "",
        "## Fresh fixed-B outcomes",
        "",
        "| Policy | Recovered | Harmed | Net |",
        "|---|---:|---:|---:|",
    ]
    for label, value in (
        ("Global matched random, exact", fresh["baselines"]["global_matched_random"]),
        ("Cell-head cells + uniform rank, exact", fresh["baselines"]["cell_matched_random_rank"]),
    ):
        lines.append(
            f"| {label} | {float(value['recovered']['float']):.6g} | {float(value['harmed']['float']):.6g} | {float(value['net']['float']):.6g} |"
        )
    for label, key in (
        ("CellGate + RankResidualRidge", "rank_residual_ridge"),
        ("CellGate + HierarchicalProfile", "hierarchical_static_rank_profile"),
        ("CellGate + MinPredictedHarm", "harm_head"),
    ):
        value = fresh[key]["outcome"]
        lines.append(
            f"| {label} | {value['recovered']} | {value['harmed']} | {value['net']} |"
        )
    lines.extend(
        [
            "",
            "## Head diagnostics",
            "",
            f"- Cell head fresh MSE skill: `{fresh['cell_head']['prediction']['mse_skill']}`; exact cell gain `{f(fresh['cell_head']['cell_selection_gain'])}`.",
            f"- Rank-residual ridge fresh MSE skill: `{fresh['rank_residual_ridge']['prediction']['mse_skill']}`; rank gain `{f(fresh['rank_residual_ridge']['rank_gain'])}`.",
            f"- Profile fresh MSE skill: `{fresh['hierarchical_static_rank_profile']['prediction']['mse_skill']}`; rank gain `{f(profile_gain)}`; positive documents `{fresh['hierarchical_static_rank_profile']['document_gain']['positive_document_count']}/16`.",
            f"- Harm head fresh MSE skill: `{fresh['harm_head']['prediction']['mse_skill']}`; exact harm avoidance `{f(fresh['harm_head']['harm_avoidance'])}`; effective=`{str(fresh['harm_head']['effective']).lower()}`.",
            "",
            "## Interpretation boundary",
            "",
            decision["reason"],
            "",
            "任何 Hybrid 选择只代表下一套全新 document-disjoint cohort 的预注册候选；不得把本结果称为 online dynamic observability、模型质量、serving SLO 或生产证据。",
            "",
        ]
    )
    return "\n".join(lines)


def build_manifest(output: Path, names: Sequence[str]) -> dict[str, Any]:
    return {
        "schema_version": "selector-failure-decomposition-output-manifest-v1",
        "files": {
            name: {"size_bytes": (output / name).stat().st_size, "sha256": sha256_file(output / name)}
            for name in names
        },
    }


def run_decomposition(repo_root: Path, config_path: Path, lock_path: Path, output: Path) -> dict[str, Any]:
    started = time.time()
    config = load_json(config_path)
    validate_config(config)
    lock = verify_lock(repo_root, config, lock_path)
    if output.exists():
        raise DecompositionError(f"refusing to reuse output directory {output}")

    train_spec = config["inputs"]["train_surface"]
    fresh_spec = config["inputs"]["evaluation_surface"]
    train_path = repo_root / str(train_spec["path"])
    fresh_path = repo_root / str(fresh_spec["path"])
    train_rows = load_jsonl(train_path)
    fresh_rows = load_jsonl(fresh_path)
    train_actions = surface_actions(train_rows)
    fresh_actions = surface_actions(fresh_rows)
    train_validation = validate_surface(train_rows, train_actions, train_spec)
    fresh_validation = validate_surface(fresh_rows, fresh_actions, fresh_spec)
    overlap = set(train_validation["documents"]) & set(fresh_validation["documents"])
    if len(overlap) != int(config["inputs"]["required_document_hash_overlap"]):
        raise DecompositionError("train/fresh document overlap differs from frozen requirement")

    broad_predictions = lodo_predictions(train_actions)
    full_models = fit_models(train_actions)
    fresh_predictions = score_models(fresh_actions, full_models)
    budget = int(config["action_space"]["budget_B"])
    broad_metrics = evaluate_scored_surface(broad_predictions, budget=budget)
    fresh_metrics = evaluate_scored_surface(fresh_predictions, budget=budget)
    profile_gain = exact_from_payload(
        fresh_metrics["hierarchical_static_rank_profile"]["rank_gain"]
    )
    if profile_gain > 0:
        selected_policy = "PRE_REGISTER_HYBRID_CELLGATE_PROFILEDRANK_V1"
        reason = (
            "Fresh primary profile rank gain is positive. Freeze Hybrid CellGate + "
            "ProfiledRank-v1 before collecting C8 outcomes on a third fully new "
            "document-disjoint cohort; this retrospective result is not confirmation."
        )
    else:
        selected_policy = "STOP_SUPERVISED_SELECTOR_TO_WITNESSPATCH_BUDGETED_PROBING"
        reason = (
            "Fresh primary profile rank gain is non-positive. Under the pre-result "
            "two-branch rule, do not create a harm-only rescue policy; stop supervised "
            "selector iteration and move to WitnessPatch / budgeted probing."
        )

    summary = {
        "schema_version": "selector-failure-decomposition-summary-v1",
        "status": "COMPLETE",
        "completed_at": utc_now(),
        "evidence_class": config["evidence_class"],
        "confirmatory_claim_allowed": False,
        "gpu_used": False,
        "model_inference_used": False,
        "runtime_seconds": time.time() - started,
        "inputs": {
            "train": {**train_validation, "path": str(train_spec["path"]), "sha256": sha256_file(train_path)},
            "fresh": {**fresh_validation, "path": str(fresh_spec["path"]), "sha256": sha256_file(fresh_path)},
            "document_hash_overlap": len(overlap),
        },
        "broad_lodo": broad_metrics,
        "fresh": fresh_metrics,
        "transfer": transfer_summary(broad_metrics, fresh_metrics),
        "decision": {
            "selected_policy": selected_policy,
            "fresh_profile_rank_gain_B": fraction_payload(profile_gain),
            "harm_effective_diagnostic": bool(fresh_metrics["harm_head"]["effective"]),
            "reason": reason,
            "policy_count": 1,
        },
        "integrity": {
            "lock_status": lock["status"],
            "locked_file_count": len(lock["files"]),
            "residual8_closure_train": train_validation["residual8_closure"],
            "residual8_closure_fresh": fresh_validation["residual8_closure"],
            "exact_budget": budget,
            "max_actions_per_cell": 1,
            "fresh_refit": False,
        },
    }
    bindings = {
        "schema_version": "selector-failure-decomposition-input-bindings-v1",
        "config_sha256": sha256_file(config_path),
        "lock_sha256": sha256_file(lock_path),
        "locked_files": lock["files"],
        "document_hash_overlap": len(overlap),
    }
    environment = {
        "schema_version": "selector-failure-decomposition-environment-v1",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "gpu_used": False,
        "torch_imported": False,
        "model_inference_used": False,
    }
    write_json_new(output / "config_snapshot.json", config)
    write_json_new(output / "INPUT_BINDINGS.json", bindings)
    write_json_new(output / "environment.json", environment)
    write_json_new(output / "models.json", full_models)
    write_jsonl_new(output / "broad_lodo_predictions.jsonl", broad_predictions)
    write_jsonl_new(output / "fresh_predictions.jsonl", fresh_predictions)
    write_json_new(output / "summary.json", summary)
    write_text_new(output / "PILOT_RESULT.md", render_result(summary))
    names = [
        "config_snapshot.json",
        "INPUT_BINDINGS.json",
        "environment.json",
        "models.json",
        "broad_lodo_predictions.jsonl",
        "fresh_predictions.jsonl",
        "summary.json",
        "PILOT_RESULT.md",
    ]
    write_json_new(output / "MANIFEST.json", build_manifest(output, names))
    write_json_new(
        output / "RUN_STATUS.json",
        {
            "status": "COMPLETE",
            "evidence_class": config["evidence_class"],
            "confirmatory_claim_allowed": False,
            "selected_policy": selected_policy,
            "completed_at": summary["completed_at"],
        },
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--config", type=Path, default=CONFIG_RELATIVE)
    parser.add_argument("--lock", type=Path, default=LOCK_RELATIVE)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("freeze-lock")
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    config_path = args.config if args.config.is_absolute() else repo_root / args.config
    lock_path = args.lock if args.lock.is_absolute() else repo_root / args.lock
    if args.command == "freeze-lock":
        lock = freeze_lock(repo_root, config_path, lock_path)
        print(json.dumps({"status": lock["status"], "lock": str(lock_path)}, ensure_ascii=False))
        return
    config = load_json(config_path)
    configured_output = Path(config["execution"]["output_directory"])
    output = args.output or configured_output
    output_path = output if output.is_absolute() else repo_root / output
    summary = run_decomposition(repo_root, config_path, lock_path, output_path)
    print(json.dumps(summary["decision"], ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
