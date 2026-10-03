#!/usr/bin/env python3
"""Pure CPU policy and aggregation logic for the Selectability Gate."""

from __future__ import annotations

from collections import defaultdict
from fractions import Fraction
import hashlib
import json
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


class PolicyError(RuntimeError):
    """The frozen selectability policy contract was violated."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def content_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def selectability_cell_identity(
    row: Mapping[str, Any], manifest_sha256: str
) -> str:
    text_hash = str(row["document_text_sha256"])
    window_hash = str(row["window_token_ids_sha256"])
    return (
        f"manifest={manifest_sha256}|text={text_hash}|doc={int(row['document_index']):03d}"
        f"|offset={int(row['token_offset']):04d}|width={int(row['window_tokens']):02d}"
        f"|window={window_hash}"
        f"|layer={int(row['layer']):02d}"
    )


def route_decomposition(
    unprotected_layers: Sequence[int], action_layers: Sequence[int]
) -> dict[str, Any]:
    unprotected = set(map(int, unprotected_layers))
    action = set(map(int, action_layers))
    recovered = sorted(unprotected - action)
    harmed = sorted(action - unprotected)
    persistent = sorted(unprotected & action)
    return {
        "recovered_layers": recovered,
        "harmed_layers": harmed,
        "persistent_layers": persistent,
        "recovered": len(recovered),
        "harmed": len(harmed),
        "persistent": len(persistent),
        "reward": len(recovered) - len(harmed),
    }


def calibration_candidates(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    seen_cells: set[str] = set()
    for row in rows:
        cell = f"calibration|{row['victim_id']}|layer={int(row['layer']):02d}"
        if cell in seen_cells:
            raise PolicyError(f"duplicate calibration cell {cell}")
        seen_cells.add(cell)
        weights = list(map(float, row["gate_weights"]))
        experts = list(map(int, row["expert_ids"]))
        if len(weights) != 8 or len(experts) != 8:
            raise PolicyError("calibration cell is not top-8")
        for rank in range(8):
            action = row["actions"][str(rank)]
            decomposition = route_decomposition(
                row["unprotected_changed_layers_vs_R"],
                action["changed_layers_vs_R"],
            )
            if int(action["reward"]) != int(decomposition["reward"]):
                raise PolicyError("stored calibration reward disagrees with route sets")
            candidates.append(
                {
                    "cell_identity": cell,
                    "victim_id": str(row["victim_id"]),
                    "layer": int(row["layer"]),
                    "rank": rank,
                    "expert_id": experts[rank],
                    "gate_weights": weights,
                    "current_layer_topk_cutoff_margin": float(
                        row["current_layer_topk_cutoff_margin"]
                    ),
                    "reward": int(decomposition["reward"]),
                    "recovered": int(decomposition["recovered"]),
                    "harmed": int(decomposition["harmed"]),
                }
            )
    if len(candidates) != len(rows) * 8:
        raise PolicyError("calibration action cardinality mismatch")
    return candidates


def heldout_candidates(cells: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    identities: set[str] = set()
    for cell in cells:
        identity = str(cell["cell_identity"])
        if identity in identities:
            raise PolicyError(f"duplicate held-out cell {identity}")
        identities.add(identity)
        weights = list(map(float, cell["gate_weights"]))
        experts = list(map(int, cell["expert_ids"]))
        if len(weights) != 8 or len(experts) != 8:
            raise PolicyError("held-out cell is not top-8")
        for rank in range(8):
            candidates.append(
                {
                    "cell_identity": identity,
                    "victim_id": str(cell["victim_id"]),
                    "document_index": int(cell["document_index"]),
                    "layer": int(cell["layer"]),
                    "rank": rank,
                    "expert_id": experts[rank],
                    "gate_weights": weights,
                    "current_layer_topk_cutoff_margin": float(
                        cell["current_layer_topk_cutoff_margin"]
                    ),
                }
            )
    return candidates


def _key(values: Iterable[int]) -> str:
    return "|".join(map(str, values))


def _group_stats(
    candidates: Sequence[Mapping[str, Any]], fields: Sequence[str]
) -> dict[str, dict[str, int]]:
    sums: dict[str, int] = defaultdict(int)
    counts: dict[str, int] = defaultdict(int)
    for row in candidates:
        key = _key(int(row[field]) for field in fields)
        sums[key] += int(row["reward"])
        counts[key] += 1
    return {
        key: {"sum_reward": int(sums[key]), "count": int(counts[key])}
        for key in sorted(counts)
    }


def fit_static_map(
    candidates: Sequence[Mapping[str, Any]], shrinkage_lambda: float = 4.0
) -> dict[str, Any]:
    if shrinkage_lambda <= 0:
        raise PolicyError("static shrinkage lambda must be positive")
    if not candidates:
        raise PolicyError("cannot fit static map without calibration actions")
    total = sum(int(row["reward"]) for row in candidates)
    global_mean = float(total / len(candidates))
    model = {
        "schema_version": "stablebatch-static-compatibility-map-v1",
        "algorithm": "three_pair_posteriors_mean_then_exact_tuple_posterior",
        "shrinkage_lambda": float(shrinkage_lambda),
        "global_sum_reward": int(total),
        "global_count": len(candidates),
        "global_mean": global_mean,
        "action_signature": "one_M1_rank_plus_seven_M64_BF16_eager",
        "tables": {
            "layer_expert": _group_stats(candidates, ("layer", "expert_id")),
            "expert_rank": _group_stats(candidates, ("expert_id", "rank")),
            "layer_rank": _group_stats(candidates, ("layer", "rank")),
            "layer_expert_rank": _group_stats(
                candidates, ("layer", "expert_id", "rank")
            ),
        },
    }
    model["parameter_content_sha256"] = content_sha256(model)
    return model


def _posterior(
    table: Mapping[str, Mapping[str, int]], key: str, prior: float, lam: float
) -> tuple[float, int]:
    value = table.get(key)
    if value is None:
        return float(prior), 0
    count = int(value["count"])
    return (
        (float(value["sum_reward"]) + lam * float(prior)) / (count + lam),
        count,
    )


def static_score(row: Mapping[str, Any], model: Mapping[str, Any]) -> dict[str, Any]:
    layer = int(row["layer"])
    expert = int(row["expert_id"])
    rank = int(row["rank"])
    lam = float(model["shrinkage_lambda"])
    global_mean = float(model["global_mean"])
    tables = model["tables"]
    pair_values: list[float] = []
    pair_support: dict[str, int] = {}
    for name, values in (
        ("layer_expert", (layer, expert)),
        ("expert_rank", (expert, rank)),
        ("layer_rank", (layer, rank)),
    ):
        value, support = _posterior(
            tables[name], _key(values), global_mean, lam
        )
        pair_values.append(value)
        pair_support[name] = support
    tuple_prior = sum(pair_values) / len(pair_values)
    score, exact_support = _posterior(
        tables["layer_expert_rank"],
        _key((layer, expert, rank)),
        tuple_prior,
        lam,
    )
    return {
        "score": float(score),
        "exact_support": exact_support,
        "pair_support": pair_support,
        "tuple_prior": float(tuple_prior),
    }


CONTINUOUS_FEATURES = (
    "static_score",
    "gate_weight",
    "gate_share",
    "gate_gap_to_min",
    "topk_mass",
    "topk_entropy",
    "cutoff_margin",
)


def _continuous_values(
    row: Mapping[str, Any], static_model: Mapping[str, Any]
) -> dict[str, float]:
    weights = np.asarray(list(map(float, row["gate_weights"])), dtype=np.float64)
    if weights.shape != (8,) or not np.all(np.isfinite(weights)):
        raise PolicyError("online gate weights are invalid")
    mass = float(weights.sum())
    if mass <= 0:
        raise PolicyError("online top-8 gate mass is non-positive")
    normalized = weights / mass
    entropy = float(-np.sum(normalized * np.log(np.maximum(normalized, 1e-300))))
    rank = int(row["rank"])
    return {
        "static_score": float(static_score(row, static_model)["score"]),
        "gate_weight": float(weights[rank]),
        "gate_share": float(normalized[rank]),
        "gate_gap_to_min": float(weights[rank] - weights[-1]),
        "topk_mass": mass,
        "topk_entropy": entropy,
        "cutoff_margin": float(row["current_layer_topk_cutoff_margin"]),
    }


def _feature_names(num_layers: int, num_experts: int, top_k: int) -> list[str]:
    return (
        ["intercept"]
        + [f"layer_{index}" for index in range(num_layers)]
        + [f"expert_{index}" for index in range(num_experts)]
        + [f"rank_{index}" for index in range(top_k)]
        + [f"z_{name}" for name in CONTINUOUS_FEATURES]
    )


def fit_online_ridge(
    candidates: Sequence[Mapping[str, Any]],
    static_model: Mapping[str, Any],
    *,
    alpha: float = 1.0,
    num_layers: int = 15,
    num_experts: int = 64,
    top_k: int = 8,
) -> dict[str, Any]:
    if alpha <= 0 or not candidates:
        raise PolicyError("ridge alpha and calibration data must be positive")
    raw = [_continuous_values(row, static_model) for row in candidates]
    means = {
        name: float(np.mean([value[name] for value in raw]))
        for name in CONTINUOUS_FEATURES
    }
    stds = {
        name: float(np.std([value[name] for value in raw], ddof=0))
        for name in CONTINUOUS_FEATURES
    }
    stds = {name: (value if value > 0 else 1.0) for name, value in stds.items()}
    feature_names = _feature_names(num_layers, num_experts, top_k)
    matrix = np.zeros((len(candidates), len(feature_names)), dtype=np.float64)
    labels = np.asarray([float(row["reward"]) for row in candidates], dtype=np.float64)
    for index, (row, continuous) in enumerate(zip(candidates, raw)):
        layer = int(row["layer"])
        expert = int(row["expert_id"])
        rank = int(row["rank"])
        if layer not in range(num_layers) or expert not in range(num_experts) or rank not in range(top_k):
            raise PolicyError("ridge categorical feature is out of range")
        matrix[index, 0] = 1.0
        matrix[index, 1 + layer] = 1.0
        matrix[index, 1 + num_layers + expert] = 1.0
        matrix[index, 1 + num_layers + num_experts + rank] = 1.0
        start = 1 + num_layers + num_experts + top_k
        for offset, name in enumerate(CONTINUOUS_FEATURES):
            matrix[index, start + offset] = (
                continuous[name] - means[name]
            ) / stds[name]
    penalty = np.eye(matrix.shape[1], dtype=np.float64) * float(alpha)
    penalty[0, 0] = 0.0
    # Some Accelerate-backed NumPy builds emit spurious matmul overflow warnings
    # for this small, finite matrix.  Fail closed on the resulting arrays instead.
    with np.errstate(all="ignore"):
        lhs = matrix.T @ matrix + penalty
        rhs = matrix.T @ labels
    if not np.all(np.isfinite(lhs)) or not np.all(np.isfinite(rhs)):
        raise PolicyError("ridge normal equations are non-finite")
    coefficients = np.linalg.solve(lhs, rhs)
    if not np.all(np.isfinite(coefficients)):
        raise PolicyError("ridge produced non-finite coefficients")
    model = {
        "schema_version": "stablebatch-online-observable-ridge-v1",
        "alpha": float(alpha),
        "intercept_penalized": False,
        "label": "signed_route_reward",
        "feature_names": feature_names,
        "continuous_feature_names": list(CONTINUOUS_FEATURES),
        "continuous_mean": means,
        "continuous_std": stds,
        "coefficients": [float(value) for value in coefficients.tolist()],
        "num_layers": num_layers,
        "num_experts": num_experts,
        "top_k": top_k,
    }
    model["parameter_content_sha256"] = content_sha256(model)
    return model


def online_score(
    row: Mapping[str, Any],
    static_model: Mapping[str, Any],
    ridge_model: Mapping[str, Any],
) -> float:
    num_layers = int(ridge_model["num_layers"])
    num_experts = int(ridge_model["num_experts"])
    top_k = int(ridge_model["top_k"])
    vector = np.zeros(len(ridge_model["feature_names"]), dtype=np.float64)
    vector[0] = 1.0
    vector[1 + int(row["layer"])] = 1.0
    vector[1 + num_layers + int(row["expert_id"])] = 1.0
    vector[1 + num_layers + num_experts + int(row["rank"])] = 1.0
    continuous = _continuous_values(row, static_model)
    start = 1 + num_layers + num_experts + top_k
    for offset, name in enumerate(CONTINUOUS_FEATURES):
        vector[start + offset] = (
            continuous[name] - float(ridge_model["continuous_mean"][name])
        ) / float(ridge_model["continuous_std"][name])
    return float(vector @ np.asarray(ridge_model["coefficients"], dtype=np.float64))


def rank_scored_candidates(
    candidates: Sequence[Mapping[str, Any]],
    scores: Sequence[float],
    *,
    name: str,
    tie_seed: str,
    budget: int,
) -> dict[str, Any]:
    if len(candidates) != len(scores):
        raise PolicyError("candidate/score cardinality mismatch")
    by_cell: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row, score in zip(candidates, scores):
        value = {key: row[key] for key in (
            "cell_identity", "victim_id", "document_index", "layer", "rank", "expert_id"
        )}
        value["score"] = float(score)
        by_cell[str(row["cell_identity"])].append(value)
    winners: list[dict[str, Any]] = []
    for identity, rows in by_cell.items():
        winner = min(rows, key=lambda row: (-float(row["score"]), int(row["rank"])))
        winner = dict(winner)
        winner["tie_hash"] = sha256_text(
            f"{tie_seed}|{identity}|rank={int(winner['rank'])}"
        )
        winners.append(winner)
    ranking = sorted(
        winners,
        key=lambda row: (
            -float(row["score"]), str(row["tie_hash"]), str(row["cell_identity"]), int(row["rank"])
        ),
    )
    if budget <= 0 or budget > len(ranking):
        raise PolicyError("invalid policy budget")
    for index, row in enumerate(ranking):
        row["policy_order"] = index
        row["selected"] = index < budget
    return {
        "name": name,
        "budget": budget,
        "ranking": ranking,
        "selected": ranking[:budget],
    }


def build_matched_shuffle(
    candidates: Sequence[Mapping[str, Any]], *, seed: str, budget: int, top_k: int = 8
) -> dict[str, Any]:
    cells: dict[str, Mapping[str, Any]] = {}
    for row in candidates:
        cells.setdefault(str(row["cell_identity"]), row)
    ranking_ids = sorted(
        cells,
        key=lambda identity: (sha256_text(f"{seed}|cell|{identity}"), identity),
    )
    rank_offset = int(sha256_text(f"{seed}|rank-offset")[:8], 16) % top_k
    rank_by_identity = {
        identity: (index + rank_offset) % top_k
        for index, identity in enumerate(ranking_ids)
    }
    ranking: list[dict[str, Any]] = []
    for index, identity in enumerate(ranking_ids):
        cell = cells[identity]
        rank = int(rank_by_identity[identity])
        experts = list(map(int, cell["expert_ids"])) if "expert_ids" in cell else None
        expert_id = experts[rank] if experts is not None else next(
            int(row["expert_id"])
            for row in candidates
            if str(row["cell_identity"]) == identity and int(row["rank"]) == rank
        )
        ranking.append(
            {
                "cell_identity": identity,
                "victim_id": str(cell["victim_id"]),
                "document_index": int(cell["document_index"]),
                "layer": int(cell["layer"]),
                "rank": rank,
                "expert_id": expert_id,
                "shuffle_cell_hash": sha256_text(f"{seed}|cell|{identity}"),
                "policy_order": index,
                "selected": index < budget,
            }
        )
    selected_counts = [
        sum(int(row["rank"]) == rank for row in ranking[:budget])
        for rank in range(top_k)
    ]
    if max(selected_counts) - min(selected_counts) > 1:
        raise PolicyError(f"shuffle rank balance failed: {selected_counts}")
    return {
        "name": "ActionBudgetMatchedHashShuffle-v1",
        "seed": seed,
        "budget": budget,
        "selected_rank_counts": selected_counts,
        "ranking": ranking,
        "selected": ranking[:budget],
    }


def build_preoutcome_policy_lock(
    calibration_rows: Sequence[Mapping[str, Any]],
    heldout_cells: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    calibration = calibration_candidates(calibration_rows)
    heldout = heldout_candidates(heldout_cells)
    static_model = fit_static_map(
        calibration, float(config["selectors"]["static_map"]["shrinkage_lambda"])
    )
    ridge_model = fit_online_ridge(
        calibration,
        static_model,
        alpha=float(config["selectors"]["online_ridge"]["alpha"]),
        num_layers=int(config["model"]["num_hidden_layers"]) - 1,
        num_experts=int(config["model"]["num_experts"]),
        top_k=int(config["model"]["num_experts_per_tok"]),
    )
    budget = int(config["selection"]["action_budget_cells"])
    static_scores = [static_score(row, static_model)["score"] for row in heldout]
    online_scores = [online_score(row, static_model, ridge_model) for row in heldout]
    static_plan = rank_scored_candidates(
        heldout,
        static_scores,
        name="StaticCompatibilityMap-v1",
        tie_seed=str(config["selectors"]["static_map"]["tie_seed"]),
        budget=budget,
    )
    online_plan = rank_scored_candidates(
        heldout,
        online_scores,
        name="OnlineObservableRidge-v1",
        tie_seed=str(config["selectors"]["online_ridge"]["tie_seed"]),
        budget=budget,
    )
    shuffle = build_matched_shuffle(
        heldout,
        seed=str(config["selectors"]["shuffle"]["seed"]),
        budget=budget,
        top_k=int(config["model"]["num_experts_per_tok"]),
    )
    for plan in (static_plan, online_plan, shuffle):
        selected = plan["selected"]
        if len(selected) != budget or len({row["cell_identity"] for row in selected}) != budget:
            raise PolicyError(f"{plan['name']} does not use exact unique-cell budget")
    value = {
        "schema_version": "stablebatch-selectability-preoutcome-policy-v1",
        "status": "SEALED_BEFORE_HELDOUT_ACTION_OUTCOMES",
        "action_budget_cells": budget,
        "action_signature": config["action_space"]["candidate_action"],
        "calibration_action_count": len(calibration),
        "heldout_cell_count": len(heldout_cells),
        "heldout_candidate_count": len(heldout),
        "static_model": static_model,
        "online_ridge_model": ridge_model,
        "static_plan": static_plan,
        "online_plan": online_plan,
        "shuffle_plan": shuffle,
        "forbidden_feature_fields": list(config["selectors"]["forbidden_fields"]),
        "outcome_rows_existed_at_seal": False,
    }
    value["deterministic_content_sha256"] = content_sha256(value)
    return value


def _outcome_lookup(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
    lookup: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        identity = str(row["cell_identity"])
        for rank in range(8):
            action = row["actions"][str(rank)]
            decomposition = route_decomposition(
                row["unprotected_changed_layers_vs_R"], action["changed_layers_vs_R"]
            )
            if int(action["reward"]) != int(decomposition["reward"]):
                raise PolicyError("held-out stored reward disagrees with route sets")
            lookup[(identity, rank)] = {
                **decomposition,
                "cell_identity": identity,
                "victim_id": str(row["victim_id"]),
                "rank": rank,
            }
    return lookup


def oracle_ranking(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    lookup = _outcome_lookup(rows)
    winners: list[dict[str, Any]] = []
    for row in rows:
        identity = str(row["cell_identity"])
        options = [lookup[(identity, rank)] for rank in range(8)]
        winner = min(
            options,
            key=lambda value: (
                -int(value["reward"]),
                int(value["harmed"]),
                int(value["rank"]),
            ),
        )
        winners.append(dict(winner))
    return sorted(
        winners,
        key=lambda value: (
            -int(value["reward"]),
            int(value["harmed"]),
            int(value["rank"]),
            str(value["cell_identity"]),
        ),
    )


def _select_from_ranking(
    ranking: Sequence[Mapping[str, Any]], budget: int, excluded_victim: str | None = None
) -> list[Mapping[str, Any]]:
    selected = [
        row for row in ranking if excluded_victim is None or str(row["victim_id"]) != excluded_victim
    ][:budget]
    if len(selected) != budget:
        raise PolicyError("ranking cannot refill exact budget")
    return selected


def aggregate_plan(
    plan_rows: Sequence[Mapping[str, Any]],
    lookup: Mapping[tuple[str, int], Mapping[str, Any]],
) -> dict[str, Any]:
    outcomes = [lookup[(str(row["cell_identity"]), int(row["rank"]))] for row in plan_rows]
    per_victim: dict[str, dict[str, int]] = defaultdict(
        lambda: {"reward": 0, "recovered": 0, "harmed": 0, "actions": 0}
    )
    for outcome in outcomes:
        victim = str(outcome["victim_id"])
        for field in ("reward", "recovered", "harmed"):
            per_victim[victim][field] += int(outcome[field])
        per_victim[victim]["actions"] += 1
    return {
        "actions": len(outcomes),
        "reward": sum(int(row["reward"]) for row in outcomes),
        "recovered": sum(int(row["recovered"]) for row in outcomes),
        "harmed": sum(int(row["harmed"]) for row in outcomes),
        "positive_net_victims": sorted(
            victim for victim, value in per_victim.items() if int(value["reward"]) > 0
        ),
        "per_victim": {key: dict(value) for key, value in sorted(per_victim.items())},
    }


def classify_selectability(
    rows: Sequence[Mapping[str, Any]],
    policy_lock: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    expected_cells = int(config["selection"]["cell_count"])
    budget = int(config["selection"]["action_budget_cells"])
    if len(rows) != expected_cells or any(row.get("integrity_status") != "PASS" for row in rows):
        raise PolicyError("held-out result rows fail cardinality or integrity")
    lookup = _outcome_lookup(rows)
    oracle_full = oracle_ranking(rows)
    rankings = {
        "oracle": oracle_full,
        "static": policy_lock["static_plan"]["ranking"],
        "online": policy_lock["online_plan"]["ranking"],
        "shuffle": policy_lock["shuffle_plan"]["ranking"],
    }
    aggregates = {
        name: aggregate_plan(_select_from_ranking(ranking, budget), lookup)
        for name, ranking in rankings.items()
    }
    denominator = int(aggregates["oracle"]["reward"]) - int(
        aggregates["shuffle"]["reward"]
    )
    recovered_gap: dict[str, float | None] = {}
    for name in ("static", "online"):
        recovered_gap[name] = (
            (int(aggregates[name]["reward"]) - int(aggregates["shuffle"]["reward"]))
            / denominator
            if denominator > 0
            else None
        )

    all_rewards = [int(value["reward"]) for value in lookup.values()]
    uniform = Fraction(budget * sum(all_rewards), expected_cells * 8)
    total_unprotected = sum(int(row["unprotected_distance_vs_R"]) for row in rows)
    oracle_selected = _select_from_ranking(oracle_full, budget)
    positive_oracle_victims = sorted(
        {str(row["victim_id"]) for row in oracle_selected if int(row["reward"]) > 0}
    )
    opportunity_cfg = config["gate"]["oracle_opportunity"]
    oracle_recovery_fraction = (
        float(aggregates["oracle"]["recovered"] / total_unprotected)
        if total_unprotected
        else 0.0
    )
    oracle_checks = {
        "reward_positive": int(aggregates["oracle"]["reward"]) > 0,
        "above_shuffle": denominator > 0,
        "min_recovery_fraction": oracle_recovery_fraction
        >= float(opportunity_cfg["min_recovery_fraction"]),
        "min_positive_victims": len(positive_oracle_victims)
        >= int(opportunity_cfg["min_positive_victims"]),
    }
    victims = sorted({str(row["victim_id"]) for row in rows})
    lodo: dict[str, Any] = {}
    for victim in victims:
        values = {
            name: aggregate_plan(
                _select_from_ranking(ranking, budget, excluded_victim=victim), lookup
            )
            for name, ranking in rankings.items()
        }
        denom = int(values["oracle"]["reward"]) - int(values["shuffle"]["reward"])
        lodo[victim] = {"aggregates": values, "oracle_minus_shuffle": denom}
        for name in ("static", "online"):
            gap = (
                (int(values[name]["reward"]) - int(values["shuffle"]["reward"])) / denom
                if denom > 0
                else None
            )
            lodo[victim][f"{name}_recovered_oracle_gap"] = gap

    selector_results: dict[str, Any] = {}
    min_gap = float(config["gate"]["selector"]["min_recovered_oracle_gap"])
    min_victims = int(config["gate"]["selector"]["min_positive_net_victims"])
    for name in ("static", "online"):
        full_checks = {
            "reward_positive": int(aggregates[name]["reward"]) > 0,
            "above_shuffle": int(aggregates[name]["reward"])
            > int(aggregates["shuffle"]["reward"]),
            "min_recovered_oracle_gap": recovered_gap[name] is not None
            and float(recovered_gap[name]) >= min_gap,
            "min_positive_net_victims": len(aggregates[name]["positive_net_victims"])
            >= min_victims,
        }
        lodo_checks = []
        for victim in victims:
            values = lodo[victim]["aggregates"]
            gap = lodo[victim][f"{name}_recovered_oracle_gap"]
            lodo_checks.append(
                bool(
                    gap is not None
                    and int(values[name]["reward"]) > 0
                    and int(values[name]["reward"]) > int(values["shuffle"]["reward"])
                    and float(gap) >= min_gap
                )
            )
        selector_results[name] = {
            "recovered_oracle_gap": recovered_gap[name],
            "full_checks": full_checks,
            "lodo_pass_count": sum(lodo_checks),
            "lodo_total": len(lodo_checks),
            "go": all(oracle_checks.values()) and all(full_checks.values()) and all(lodo_checks),
        }

    if not all(oracle_checks.values()):
        verdict = "STOP_NO_FRESH_ORACLE_OPPORTUNITY"
    elif selector_results["static"]["go"]:
        verdict = "GO_STATIC_COMPATIBILITY"
    elif selector_results["online"]["go"]:
        verdict = "GO_ROW_CONDITIONED"
    else:
        verdict = "STOP_PREACTION_STABLEBATCH"
    return {
        "cell_count": len(rows),
        "action_budget_cells": budget,
        "total_unprotected_route_distance": total_unprotected,
        "aggregates": aggregates,
        "oracle_minus_shuffle": denominator,
        "oracle_recovery_fraction": oracle_recovery_fraction,
        "positive_oracle_victims": positive_oracle_victims,
        "oracle_opportunity_checks": oracle_checks,
        "selector_results": selector_results,
        "uniform_random_expected_reward": {
            "numerator": uniform.numerator,
            "denominator": uniform.denominator,
            "float": float(uniform),
        },
        "leave_one_victim_out": lodo,
        "verdict": verdict,
    }
