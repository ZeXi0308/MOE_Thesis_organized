from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np


ROW_BINS = ("1", "2", "3-4", "5-8", "9-16", "17-32", "33-64", "65-128", ">=129")


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    split: str
    tokens_per_tenant: int
    top_k: int
    num_experts: int
    overlap_fraction: float
    histogram_regime: str
    seed: int
    selected_experts: np.ndarray
    tenant_ids: np.ndarray


def row_bin(count: int) -> str:
    if count <= 0:
        raise ValueError("row count must be positive")
    if count == 1:
        return "1"
    if count == 2:
        return "2"
    for low, high in ((3, 4), (5, 8), (9, 16), (17, 32), (33, 64), (65, 128)):
        if low <= count <= high:
            return f"{low}-{high}"
    return ">=129"


def _pool_pair(num_experts: int, pool_size: int, overlap_fraction: float) -> tuple[list[int], list[int]]:
    if not 0.0 <= overlap_fraction <= 1.0:
        raise ValueError("overlap_fraction must be in [0,1]")
    overlap = int(round(pool_size * overlap_fraction))
    overlap = min(pool_size, max(0, overlap))
    required = overlap + 2 * (pool_size - overlap)
    if required > num_experts:
        raise ValueError("expert pool construction exceeds num_experts")
    shared = list(range(overlap))
    cursor = overlap
    unique_a = list(range(cursor, cursor + pool_size - overlap))
    cursor += pool_size - overlap
    unique_b = list(range(cursor, cursor + pool_size - overlap))
    return shared + unique_a, shared + unique_b


def _sample_routes(
    rng: np.random.Generator,
    tokens: int,
    top_k: int,
    pool: Sequence[int],
    regime: str,
) -> np.ndarray:
    if len(pool) < top_k:
        raise ValueError("expert pool must contain at least top_k experts")
    if regime not in {"balanced", "skewed"}:
        raise ValueError(f"unknown histogram regime: {regime}")
    pool_array = np.asarray(pool, dtype=np.int64)
    if regime == "balanced":
        probabilities = np.ones(len(pool_array), dtype=np.float64)
    else:
        probabilities = 1.0 / np.power(np.arange(1, len(pool_array) + 1), 1.25)
    probabilities /= probabilities.sum()
    routes = np.empty((tokens, top_k), dtype=np.int64)
    for token in range(tokens):
        routes[token] = rng.choice(pool_array, size=top_k, replace=False, p=probabilities)
    # Force exact pool coverage while preserving per-token uniqueness. Replace
    # only experts with global multiplicity >1, so fixing one missing expert
    # cannot make an already-covered expert disappear.
    counts = {int(expert): int(np.count_nonzero(routes == expert)) for expert in pool_array}
    for expert in pool_array:
        expert = int(expert)
        if counts[expert] > 0:
            continue
        replaced = False
        for token in range(tokens):
            if expert in routes[token]:
                continue
            for slot in range(top_k):
                old = int(routes[token, slot])
                if counts[old] > 1:
                    routes[token, slot] = expert
                    counts[old] -= 1
                    counts[expert] += 1
                    replaced = True
                    break
            if replaced:
                break
        if not replaced:
            raise AssertionError("cannot enforce exact pool coverage")
    if any(len(np.unique(row)) != top_k for row in routes):
        raise AssertionError("generated route contains duplicate expert within token")
    if set(np.unique(routes)) != set(int(expert) for expert in pool_array):
        raise AssertionError("generated route does not cover the frozen expert pool")
    return routes


def make_scenario(
    *,
    split: str,
    tokens_per_tenant: int,
    top_k: int,
    num_experts: int,
    overlap_fraction: float,
    histogram_regime: str,
    seed: int,
) -> Scenario:
    if split not in {"calibration", "sealed"}:
        raise ValueError("split must be calibration or sealed")
    # At overlap=0, two disjoint pools constrain each pool to E/2. As overlap
    # grows, use the freed expert budget to make pool_size > top_k. This is
    # essential for k=E/2 models: a pool of exactly k makes balanced and skewed
    # routes identical because every token selects the whole pool.
    denominator = 2.0 - overlap_fraction
    pool_size = min(2 * top_k, int(math.floor(num_experts / denominator)))
    pool_a, pool_b = _pool_pair(num_experts, pool_size, overlap_fraction)
    rng_a = np.random.default_rng(seed * 2 + 17)
    rng_b = np.random.default_rng(seed * 2 + 18)
    route_a = _sample_routes(rng_a, tokens_per_tenant, top_k, pool_a, histogram_regime)
    route_b = _sample_routes(rng_b, tokens_per_tenant, top_k, pool_b, histogram_regime)
    routes = np.concatenate([route_a, route_b], axis=0)
    tenants = np.concatenate(
        [np.zeros(tokens_per_tenant, dtype=np.int64), np.ones(tokens_per_tenant, dtype=np.int64)]
    )
    identity = {
        "split": split,
        "tokens": tokens_per_tenant,
        "top_k": top_k,
        "num_experts": num_experts,
        "overlap": overlap_fraction,
        "regime": histogram_regime,
        "seed": seed,
        "routes_sha256": hashlib.sha256(routes.tobytes()).hexdigest(),
    }
    scenario_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode("utf-8")
    ).hexdigest()[:24]
    scenario = Scenario(
        scenario_id=scenario_id,
        split=split,
        tokens_per_tenant=tokens_per_tenant,
        top_k=top_k,
        num_experts=num_experts,
        overlap_fraction=overlap_fraction,
        histogram_regime=histogram_regime,
        seed=seed,
        selected_experts=routes,
        tenant_ids=tenants,
    )
    validate_scenario(scenario)
    return scenario


def validate_scenario(scenario: Scenario) -> None:
    expected_shape = (2 * scenario.tokens_per_tenant, scenario.top_k)
    if scenario.selected_experts.shape != expected_shape:
        raise ValueError(f"route shape {scenario.selected_experts.shape} != {expected_shape}")
    if scenario.tenant_ids.shape != (expected_shape[0],):
        raise ValueError("tenant identity shape mismatch")
    if scenario.selected_experts.min() < 0 or scenario.selected_experts.max() >= scenario.num_experts:
        raise ValueError("expert id out of range")
    if any(len(np.unique(row)) != scenario.top_k for row in scenario.selected_experts):
        raise ValueError("duplicate expert inside token top-k")
    if set(np.unique(scenario.tenant_ids)) != {0, 1}:
        raise ValueError("scenario must contain exactly tenants 0 and 1")


def scenario_features(scenario: Scenario) -> dict[str, float | int | str]:
    counts = np.bincount(
        scenario.selected_experts.reshape(-1), minlength=scenario.num_experts
    )
    active = counts[counts > 0]
    result: dict[str, float | int | str] = {
        "scenario_id": scenario.scenario_id,
        "split": scenario.split,
        "tokens_per_tenant": scenario.tokens_per_tenant,
        "top_k": scenario.top_k,
        "total_rows": int(active.sum()),
        "active_experts": int(len(active)),
        "max_rows_per_expert": int(active.max()),
        "row_count_cv": float(active.std() / max(active.mean(), 1e-12)),
        "overlap_fraction": scenario.overlap_fraction,
        "histogram_regime": scenario.histogram_regime,
        "seed": scenario.seed,
        "route_sha256": hashlib.sha256(scenario.selected_experts.tobytes()).hexdigest(),
    }
    bins = {label: 0 for label in ROW_BINS}
    for count in active:
        bins[row_bin(int(count))] += 1
    result.update({f"experts_bin_{label}": value for label, value in bins.items()})
    return result


def design_matrix(rows: Sequence[Mapping[str, float]], model: str) -> np.ndarray:
    if model == "m0_rows":
        names = ["total_rows"]
    elif model == "m1_rows_active":
        names = ["total_rows", "active_experts"]
    elif model == "m2_row_bins":
        names = ["total_rows", "active_experts", *[f"experts_bin_{x}" for x in ROW_BINS]]
    else:
        raise ValueError(f"unknown cost model: {model}")
    matrix = np.asarray([[float(row[name]) for name in names] for row in rows], dtype=np.float64)
    return np.column_stack([np.ones(len(matrix)), matrix])


def fit_linear_cost(rows: Sequence[Mapping[str, float]], model: str) -> np.ndarray:
    x = design_matrix(rows, model)
    y = np.asarray([float(row["coalition_latency_ms"]) for row in rows], dtype=np.float64)
    beta, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
    return beta


def predict_linear_cost(rows: Sequence[Mapping[str, float]], model: str, beta: np.ndarray) -> np.ndarray:
    return design_matrix(rows, model) @ beta


def squared_error_gap_recovery(y: np.ndarray, baseline: np.ndarray, candidate: np.ndarray) -> float:
    baseline_sse = float(np.square(y - baseline).sum())
    candidate_sse = float(np.square(y - candidate).sum())
    if baseline_sse <= 1e-18:
        return 0.0
    return 1.0 - candidate_sse / baseline_sse


def bootstrap_gap_recovery(
    y: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    repeats: int,
    seed: int,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(repeats):
        indices = rng.integers(0, len(y), size=len(y))
        values.append(squared_error_gap_recovery(y[indices], baseline[indices], candidate[indices]))
    point = squared_error_gap_recovery(y, baseline, candidate)
    return point, float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))
