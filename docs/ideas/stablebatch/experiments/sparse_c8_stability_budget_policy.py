#!/usr/bin/env python3
"""Outcome-naive ridge selection and exact Sparse-C8 budget decomposition.

The feature path deliberately reads only action-pre router fields.  Outcome
fields are consumed separately as training labels or after a selector plan has
already been frozen.
"""

from __future__ import annotations

from collections import defaultdict
from fractions import Fraction
import math
from typing import Any, Mapping, Sequence

import numpy as np


RIDGE_ALPHA = 1.0
CONTINUOUS_FEATURES = (
    "gate_weight",
    "gate_share",
    "gate_gap_to_min",
    "topk_mass",
    "normalized_entropy",
    "cutoff_margin",
)


class PolicyError(RuntimeError):
    """The frozen Sparse-C8 policy contract was violated."""


def fraction_payload(value: Fraction | int) -> dict[str, int | float]:
    """Return an exact, JSON-serializable fraction plus its display value."""

    exact = value if isinstance(value, Fraction) else Fraction(value, 1)
    return {
        "numerator": exact.numerator,
        "denominator": exact.denominator,
        "float": float(exact),
    }


def _document_id(row: Mapping[str, Any]) -> str:
    for key in ("document_id", "victim_id", "document_text_sha256"):
        if key in row:
            return str(row[key])
    if "document_index" in row:
        return f"document_index={int(row['document_index'])}"
    raise PolicyError("cell has no document identity")


def _cell_fields(cell: Mapping[str, Any], top_k: int) -> tuple[list[int], list[float]]:
    experts = list(map(int, cell["expert_ids"]))
    weights = list(map(float, cell["gate_weights"]))
    if len(experts) != top_k or len(weights) != top_k:
        raise PolicyError(f"cell is not top-{top_k}")
    if len(set(experts)) != top_k:
        raise PolicyError("cell expert_ids are not unique")
    if any(not math.isfinite(value) or value < 0.0 for value in weights):
        raise PolicyError("gate_weights must be finite and nonnegative")
    if sum(weights) <= 0.0:
        raise PolicyError("top-k gate mass must be positive")
    return experts, weights


def flatten_preaction_cells(
    cells: Sequence[Mapping[str, Any]], *, top_k: int = 8
) -> list[dict[str, Any]]:
    """Expand frozen cells into rank actions without reading any outcome field."""

    actions: list[dict[str, Any]] = []
    seen: set[str] = set()
    for cell in cells:
        identity = str(cell["cell_identity"])
        if identity in seen:
            raise PolicyError(f"duplicate cell {identity}")
        seen.add(identity)
        experts, weights = _cell_fields(cell, top_k)
        cutoff = float(cell["current_layer_topk_cutoff_margin"])
        if not math.isfinite(cutoff):
            raise PolicyError("cutoff margin must be finite")
        for rank in range(top_k):
            actions.append(
                {
                    "cell_identity": identity,
                    "document_id": _document_id(cell),
                    "layer": int(cell["layer"]),
                    "rank": rank,
                    "expert_id": experts[rank],
                    "gate_weights": list(weights),
                    "current_layer_topk_cutoff_margin": cutoff,
                }
            )
    return actions


def flatten_outcome_cells(
    cells: Sequence[Mapping[str, Any]], *, top_k: int = 8
) -> list[dict[str, Any]]:
    """Expand cell ledgers and derive the only training label: recovered-harmed."""

    preaction = flatten_preaction_cells(cells, top_k=top_k)
    by_identity = {str(cell["cell_identity"]): cell for cell in cells}
    result: list[dict[str, Any]] = []
    for row in preaction:
        cell = by_identity[str(row["cell_identity"])]
        outcomes = cell.get("actions")
        if not isinstance(outcomes, Mapping):
            raise PolicyError("outcome cell has no actions mapping")
        raw = outcomes.get(str(row["rank"]), outcomes.get(int(row["rank"])))
        if not isinstance(raw, Mapping):
            raise PolicyError("outcome cell is missing a rank action")
        recovered = int(raw["recovered"])
        harmed = int(raw["harmed"])
        if recovered < 0 or harmed < 0:
            raise PolicyError("recovered/harmed counts must be nonnegative")
        net = recovered - harmed
        for optional in ("net", "reward"):
            if optional in raw and int(raw[optional]) != net:
                raise PolicyError(f"stored {optional} disagrees with recovered-harmed")
        result.append(
            {
                **row,
                "recovered": recovered,
                "harmed": harmed,
                "net": net,
            }
        )
    return result


def _continuous_values(row: Mapping[str, Any], top_k: int) -> dict[str, float]:
    weights = list(map(float, row["gate_weights"]))
    if len(weights) != top_k:
        raise PolicyError(f"action is not top-{top_k}")
    rank = int(row["rank"])
    if rank < 0 or rank >= top_k:
        raise PolicyError("rank is outside the action space")
    mass = float(sum(weights))
    if mass <= 0.0 or not math.isfinite(mass):
        raise PolicyError("top-k gate mass must be finite and positive")
    probabilities = [value / mass for value in weights]
    entropy = -sum(value * math.log(value) for value in probabilities if value > 0.0)
    normalized_entropy = entropy / math.log(top_k) if top_k > 1 else 0.0
    values = {
        "gate_weight": weights[rank],
        "gate_share": probabilities[rank],
        "gate_gap_to_min": weights[rank] - min(weights),
        "topk_mass": mass,
        "normalized_entropy": normalized_entropy,
        "cutoff_margin": float(row["current_layer_topk_cutoff_margin"]),
    }
    if any(not math.isfinite(value) for value in values.values()):
        raise PolicyError("continuous action-pre feature is not finite")
    return values


def _feature_names(num_layers: int, num_experts: int, top_k: int) -> list[str]:
    return (
        ["intercept"]
        + [f"layer_{index}" for index in range(num_layers)]
        + [f"expert_{index}" for index in range(num_experts)]
        + [f"rank_{index}" for index in range(top_k)]
        + [f"z_{name}" for name in CONTINUOUS_FEATURES]
    )


def _validate_dimensions(
    row: Mapping[str, Any], *, num_layers: int, num_experts: int, top_k: int
) -> None:
    layer = int(row["layer"])
    expert = int(row["expert_id"])
    rank = int(row["rank"])
    if not 0 <= layer < num_layers:
        raise PolicyError(f"layer {layer} is outside [0,{num_layers})")
    if not 0 <= expert < num_experts:
        raise PolicyError(f"expert {expert} is outside [0,{num_experts})")
    if not 0 <= rank < top_k:
        raise PolicyError(f"rank {rank} is outside [0,{top_k})")


def _feature_vector(
    row: Mapping[str, Any], model: Mapping[str, Any]
) -> np.ndarray:
    num_layers = int(model["num_layers"])
    num_experts = int(model["num_experts"])
    top_k = int(model["top_k"])
    _validate_dimensions(
        row, num_layers=num_layers, num_experts=num_experts, top_k=top_k
    )
    vector = np.zeros(len(model["feature_names"]), dtype=np.float64)
    vector[0] = 1.0
    layer = int(row["layer"])
    expert = int(row["expert_id"])
    rank = int(row["rank"])
    vector[1 + layer] = 1.0
    expert_start = 1 + num_layers
    vector[expert_start + expert] = 1.0
    rank_start = expert_start + num_experts
    vector[rank_start + rank] = 1.0
    continuous_start = rank_start + top_k
    raw = _continuous_values(row, top_k)
    for offset, name in enumerate(CONTINUOUS_FEATURES):
        mean = float(model["continuous_mean"][name])
        std = float(model["continuous_std"][name])
        vector[continuous_start + offset] = (raw[name] - mean) / std
    return vector


def fit_action_pre_ridge(
    training_actions: Sequence[Mapping[str, Any]],
    *,
    num_layers: int,
    num_experts: int,
    top_k: int = 8,
) -> dict[str, Any]:
    """Fit the fixed alpha=1 ridge on net=recovered-harmed labels."""

    if not training_actions:
        raise PolicyError("ridge training set is empty")
    if min(num_layers, num_experts, top_k) <= 0:
        raise PolicyError("model dimensions must be positive")
    raw_continuous: list[dict[str, float]] = []
    labels: list[float] = []
    for row in training_actions:
        _validate_dimensions(
            row, num_layers=num_layers, num_experts=num_experts, top_k=top_k
        )
        recovered = int(row["recovered"])
        harmed = int(row["harmed"])
        net = recovered - harmed
        if "net" in row and int(row["net"]) != net:
            raise PolicyError("training net disagrees with recovered-harmed")
        raw_continuous.append(_continuous_values(row, top_k))
        labels.append(float(net))

    means: dict[str, float] = {}
    stds: dict[str, float] = {}
    for name in CONTINUOUS_FEATURES:
        values = np.asarray([row[name] for row in raw_continuous], dtype=np.float64)
        means[name] = float(np.mean(values))
        observed = float(np.std(values, ddof=0))
        stds[name] = observed if observed > 0.0 else 1.0

    feature_names = _feature_names(num_layers, num_experts, top_k)
    skeleton: dict[str, Any] = {
        "schema_version": "sparse-c8-action-pre-ridge-v1",
        "alpha": RIDGE_ALPHA,
        "intercept_penalized": False,
        "label": "net=recovered-harmed",
        "num_layers": int(num_layers),
        "num_experts": int(num_experts),
        "top_k": int(top_k),
        "feature_names": feature_names,
        "continuous_mean": means,
        "continuous_std": stds,
        "training_action_count": len(training_actions),
        "outcome_derived_features": [],
    }
    matrix = np.vstack([_feature_vector(row, skeleton) for row in training_actions])
    target = np.asarray(labels, dtype=np.float64)
    penalty = np.eye(matrix.shape[1], dtype=np.float64) * RIDGE_ALPHA
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(matrix.T @ matrix + penalty, matrix.T @ target)
    skeleton["coefficients"] = coefficients.tolist()
    return skeleton


def predict_action_scores(
    actions: Sequence[Mapping[str, Any]], model: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Score actions using only fields consumed by ``_feature_vector``."""

    if float(model.get("alpha", -1.0)) != RIDGE_ALPHA:
        raise PolicyError("ridge alpha is not the frozen value 1.0")
    coefficients = np.asarray(model["coefficients"], dtype=np.float64)
    if coefficients.shape != (len(model["feature_names"]),):
        raise PolicyError("ridge coefficient cardinality mismatch")
    result: list[dict[str, Any]] = []
    for row in actions:
        score = float(_feature_vector(row, model) @ coefficients)
        if not math.isfinite(score):
            raise PolicyError("predicted utility is not finite")
        result.append(
            {
                "cell_identity": str(row["cell_identity"]),
                "document_id": _document_id(row),
                "layer": int(row["layer"]),
                "rank": int(row["rank"]),
                "expert_id": int(row["expert_id"]),
                "predicted_utility": score,
            }
        )
    return result


def select_global_exact_b(
    scored_actions: Sequence[Mapping[str, Any]], *, budget: int, top_k: int = 8
) -> dict[str, Any]:
    """Choose one best rank per cell, then the globally best exact-B cells."""

    by_cell: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in scored_actions:
        score = float(row["predicted_utility"])
        if not math.isfinite(score):
            raise PolicyError("predicted utility is not finite")
        by_cell[str(row["cell_identity"])].append(row)
    if budget <= 0 or budget > len(by_cell):
        raise PolicyError("budget is outside the unique-cell population")

    winners: list[dict[str, Any]] = []
    for identity, rows in by_cell.items():
        ranks = {int(row["rank"]) for row in rows}
        if len(rows) != top_k or ranks != set(range(top_k)):
            raise PolicyError(f"cell {identity} does not have exactly ranks 0..{top_k - 1}")
        winner = min(
            rows,
            key=lambda row: (-float(row["predicted_utility"]), int(row["rank"])),
        )
        winners.append(dict(winner))
    ranking = sorted(
        winners,
        key=lambda row: (-float(row["predicted_utility"]), str(row["cell_identity"])),
    )
    for order, row in enumerate(ranking):
        row["policy_order"] = order
        row["selected"] = order < budget
    return {
        "schema_version": "sparse-c8-global-exact-budget-plan-v1",
        "budget": budget,
        "cell_count": len(by_cell),
        "ranking": ranking,
        "selected": ranking[:budget],
    }


def _outcome_lookup(
    actions: Sequence[Mapping[str, Any]], top_k: int
) -> tuple[dict[tuple[str, int], dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    lookup: dict[tuple[str, int], dict[str, Any]] = {}
    by_cell: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for raw in actions:
        row = dict(raw)
        identity = str(row["cell_identity"])
        rank = int(row["rank"])
        recovered = int(row["recovered"])
        harmed = int(row["harmed"])
        if recovered < 0 or harmed < 0:
            raise PolicyError("outcome counts must be nonnegative")
        net = recovered - harmed
        if "net" in row and int(row["net"]) != net:
            raise PolicyError("outcome net disagrees with recovered-harmed")
        row.update(
            {
                "cell_identity": identity,
                "document_id": _document_id(row),
                "rank": rank,
                "recovered": recovered,
                "harmed": harmed,
                "net": net,
            }
        )
        key = (identity, rank)
        if key in lookup:
            raise PolicyError(f"duplicate outcome action {key}")
        lookup[key] = row
        by_cell[identity].append(row)
    for identity, rows in by_cell.items():
        ranks = {int(row["rank"]) for row in rows}
        if len(rows) != top_k or ranks != set(range(top_k)):
            raise PolicyError(f"cell {identity} outcome surface is incomplete")
        if len({str(row["document_id"]) for row in rows}) != 1:
            raise PolicyError(f"cell {identity} spans multiple documents")
    return lookup, by_cell


def _aggregate(actions: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return {
        "actions": len(actions),
        "recovered": sum(int(row["recovered"]) for row in actions),
        "harmed": sum(int(row["harmed"]) for row in actions),
        "net": sum(int(row["net"]) for row in actions),
    }


def _fraction_fields(values: Mapping[str, Fraction]) -> dict[str, Any]:
    return {name: fraction_payload(value) for name, value in values.items()}


def _expected_global(
    actions: Sequence[Mapping[str, Any]], *, budget: int, cell_count: int, top_k: int
) -> dict[str, Fraction]:
    return {
        field: Fraction(
            budget * sum(int(row[field]) for row in actions), cell_count * top_k
        )
        for field in ("recovered", "harmed", "net")
    }


def _expected_on_cells(
    by_cell: Mapping[str, Sequence[Mapping[str, Any]]], selected_cells: set[str], top_k: int
) -> dict[str, Fraction]:
    return {
        field: sum(
            (
                Fraction(sum(int(row[field]) for row in by_cell[identity]), top_k)
                for identity in sorted(selected_cells)
            ),
            Fraction(0),
        )
        for field in ("recovered", "harmed", "net")
    }


def _decomposition_payload(
    selector_net: Fraction,
    global_random_net: Fraction,
    cell_random_net: Fraction,
    oracle_exact_net: Fraction,
) -> dict[str, Any]:
    cell_gain = cell_random_net - global_random_net
    rank_gain = selector_net - cell_random_net
    denominator = oracle_exact_net - cell_random_net
    capture = rank_gain / denominator if denominator > 0 else None
    return {
        "cell_selection_gain": fraction_payload(cell_gain),
        "rank_selection_gain": fraction_payload(rank_gain),
        "rank_headroom_denominator": fraction_payload(denominator),
        "rank_headroom_capture": fraction_payload(capture) if capture is not None else None,
    }


def evaluate_budget_decomposition(
    outcome_actions: Sequence[Mapping[str, Any]],
    selector_plan: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    *,
    budget: int,
    top_k: int = 8,
) -> dict[str, Any]:
    """Compute exact random baselines, both Oracles, gains, and per-document results."""

    lookup, by_cell = _outcome_lookup(outcome_actions, top_k)
    if budget <= 0 or budget > len(by_cell):
        raise PolicyError("budget is outside the outcome cell population")
    selected_raw = (
        selector_plan["selected"]
        if isinstance(selector_plan, Mapping)
        else selector_plan
    )
    selected = [lookup[(str(row["cell_identity"]), int(row["rank"]))] for row in selected_raw]
    selected_cells = {str(row["cell_identity"]) for row in selected}
    if len(selected) != budget or len(selected_cells) != budget:
        raise PolicyError("selector must use exact B distinct cells")

    best_per_cell: list[dict[str, Any]] = []
    for identity, rows in by_cell.items():
        best_per_cell.append(
            dict(
                min(
                    rows,
                    key=lambda row: (
                        -int(row["net"]),
                        int(row["harmed"]),
                        -int(row["recovered"]),
                        int(row["rank"]),
                    ),
                )
            )
        )
    oracle_ranking = sorted(
        best_per_cell,
        key=lambda row: (
            -int(row["net"]),
            int(row["harmed"]),
            -int(row["recovered"]),
            str(row["cell_identity"]),
            int(row["rank"]),
        ),
    )
    oracle_exact = oracle_ranking[:budget]
    oracle_at_most = [row for row in oracle_ranking if int(row["net"]) > 0][:budget]

    global_random = _expected_global(
        outcome_actions, budget=budget, cell_count=len(by_cell), top_k=top_k
    )
    cell_random = _expected_on_cells(by_cell, selected_cells, top_k)
    selector_aggregate = _aggregate(selected)
    oracle_exact_aggregate = _aggregate(oracle_exact)
    oracle_at_most_aggregate = _aggregate(oracle_at_most)
    decomposition = _decomposition_payload(
        Fraction(selector_aggregate["net"]),
        global_random["net"],
        cell_random["net"],
        Fraction(oracle_exact_aggregate["net"]),
    )

    documents = sorted({str(row["document_id"]) for row in outcome_actions})
    per_document: dict[str, Any] = {}
    for document in documents:
        doc_actions = [row for row in outcome_actions if str(row["document_id"]) == document]
        doc_selector = [row for row in selected if str(row["document_id"]) == document]
        doc_oracle_exact = [row for row in oracle_exact if str(row["document_id"]) == document]
        doc_oracle_at_most = [row for row in oracle_at_most if str(row["document_id"]) == document]
        doc_selected_cells = {str(row["cell_identity"]) for row in doc_selector}
        doc_global = _expected_global(
            doc_actions, budget=budget, cell_count=len(by_cell), top_k=top_k
        )
        doc_cell = _expected_on_cells(by_cell, doc_selected_cells, top_k)
        doc_selector_aggregate = _aggregate(doc_selector)
        doc_oracle_exact_aggregate = _aggregate(doc_oracle_exact)
        per_document[document] = {
            "cell_count": len({str(row["cell_identity"]) for row in doc_actions}),
            "global_matched_random": _fraction_fields(doc_global),
            "cell_matched_random_rank": _fraction_fields(doc_cell),
            "selector": doc_selector_aggregate,
            "oracle_exact_B": doc_oracle_exact_aggregate,
            "oracle_at_most_B": _aggregate(doc_oracle_at_most),
            "decomposition": _decomposition_payload(
                Fraction(doc_selector_aggregate["net"]),
                doc_global["net"],
                doc_cell["net"],
                Fraction(doc_oracle_exact_aggregate["net"]),
            ),
        }

    return {
        "schema_version": "sparse-c8-stability-budget-decomposition-v1",
        "cell_count": len(by_cell),
        "action_count": len(outcome_actions),
        "budget": budget,
        "global_matched_random": _fraction_fields(global_random),
        "cell_matched_random_rank": _fraction_fields(cell_random),
        "selector": selector_aggregate,
        "oracle_exact_B": oracle_exact_aggregate,
        "oracle_at_most_B": oracle_at_most_aggregate,
        "decomposition": decomposition,
        "selector_selected": [
            {"cell_identity": row["cell_identity"], "rank": row["rank"]}
            for row in selected
        ],
        "oracle_exact_B_selected": [
            {"cell_identity": row["cell_identity"], "rank": row["rank"]}
            for row in oracle_exact
        ],
        "oracle_at_most_B_selected": [
            {"cell_identity": row["cell_identity"], "rank": row["rank"]}
            for row in oracle_at_most
        ],
        "per_document": per_document,
    }
