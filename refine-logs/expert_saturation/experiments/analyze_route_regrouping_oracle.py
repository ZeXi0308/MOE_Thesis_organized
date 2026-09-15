#!/usr/bin/env python3
"""Fixed-trace request-regrouping diagnostics for the native OLMoE probe.

The replay deliberately keeps each captured request route fixed while changing
only its group membership.  Because rerunning a different group can change the
route itself in either direction, every result is explicitly a fixed-trace
structural diagnostic rather than a mathematical bound on an executed action.
No timing, TPOT, queue, SLO, or controller claim is made here.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import random
from itertools import combinations
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np


SCHEMA = "moe-fixed-route-regrouping-diagnostic-v3"
CLAIM_CEILING = "STRUCTURAL_FIXED_TRACE_DIAGNOSTIC_ONLY"
FIXED_BATCH_SIZE = 16
FIXED_GROUPS = 6
MINIMUM_PROCESS_REPEATS = 2
RANDOM_SEED = 20260823
ROUTE_BLIND_SHUFFLE_SEEDS = tuple(
    RANDOM_SEED + 104729 * index for index in range(8)
)
MAX_SWAP_PASSES = 12
ROUTE_BLIND_POLICIES = (
    "original",
    "route_blind_round_robin",
    "route_blind_hash",
    *(f"route_blind_shuffle_s{index:02d}" for index in range(len(ROUTE_BLIND_SHUFFLE_SEEDS))),
)
BALANCE_POLICIES = ROUTE_BLIND_POLICIES + (
    "simple_greedy",
    "history_greedy_tminus1",
)
ALL_POLICIES = BALANCE_POLICIES + (
    "future_route_local_search",
    "working_set_coalesce",
)
THRESHOLDS = {
    "material_trajectory_mean_max_load_reduction_pct": 3.0,
    "material_trajectory_active_expert_reduction_pct": 3.0,
    "minimum_positive_trajectory_fraction": 0.75,
    "maximum_secondary_hhi_degradation_pct": 0.0,
}
COMPATIBILITY_FIELDS = (
    "model",
    "revision",
    "dtype",
    "batch_sizes",
    "prompt_lengths",
    "output_tokens",
    "groups",
    "within_process_repeats",
    "seed",
    "max_model_len",
    "max_num_seqs",
    "max_num_batched_tokens",
    "gpu_memory_utilization",
    "enforce_eager",
    "workload_manifest_sha256",
    "probe_script_sha256",
    "runtime_identity",
    "model_shape",
)


def _load_comparator() -> Any:
    path = Path(__file__).with_name("compare_vllm_route_probe_runs.py")
    spec = importlib.util.spec_from_file_location("route_probe_comparator", path)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


COMPARATOR = _load_comparator()


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _distribution(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "median": None, "p10": None, "p90": None,
                "minimum": None, "maximum": None}
    data = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "median": float(np.median(data)),
        "p10": float(np.quantile(data, 0.10)),
        "p90": float(np.quantile(data, 0.90)),
        "minimum": float(np.min(data)),
        "maximum": float(np.max(data)),
    }


def _pct_reduction(before: float, after: float) -> float:
    if before <= 0:
        raise ValueError("percentage denominator must be positive")
    return 100.0 * (before - after) / before


def _pct_increase(before: float, after: float) -> float:
    if before <= 0:
        raise ValueError("percentage denominator must be positive")
    return 100.0 * (after - before) / before


def _validate_partition(
    partition: Sequence[Sequence[int]], request_count: int
) -> dict[str, Any]:
    flat = [int(request) for batch in partition for request in batch]
    valid = bool(
        len(partition) == FIXED_GROUPS
        and all(len(batch) == FIXED_BATCH_SIZE for batch in partition)
        and len(flat) == request_count
        and sorted(flat) == list(range(request_count))
    )
    if not valid:
        raise ValueError("partition does not preserve every request exactly once")
    return {
        "request_count": len(flat),
        "unique_request_count": len(set(flat)),
        "batch_sizes": [len(batch) for batch in partition],
        "preserves_every_request_exactly_once": True,
    }


def _loads(request_counts: np.ndarray, partition: Sequence[Sequence[int]]) -> np.ndarray:
    _validate_partition(partition, len(request_counts))
    return np.stack(
        [request_counts[np.asarray(batch, dtype=np.int64)].sum(axis=0) for batch in partition]
    ).astype(np.int16, copy=False)


def _balance_objective(loads: np.ndarray) -> tuple[int, int, int, int]:
    """Lexicographic pressure objective over every batch-layer pair.

    1) minimize the worst per-layer maximum expert load;
    2) minimize the sum of per-layer maximum loads;
    3) minimize sum(load^2), equivalent to mean HHI at fixed B/top-k;
    4) maximize the number of active experts.
    """
    layer_max = loads.max(axis=-1)
    return (
        int(layer_max.max()),
        int(layer_max.sum()),
        int(np.square(loads.astype(np.int64)).sum()),
        -int(np.count_nonzero(loads)),
    )


def _coalesce_objective(loads: np.ndarray) -> tuple[int, int, int]:
    """Separate resident-weight-reuse bound; never combined with balance.

    It minimizes total per-layer active experts, then the worst active-expert
    count, then maximizes pairwise assignment overlap via sum(load^2).
    """
    active = np.count_nonzero(loads, axis=-1)
    return (
        int(active.sum()),
        int(active.max()),
        -int(np.square(loads.astype(np.int64)).sum()),
    )


def _partition_metrics(loads: np.ndarray, top_k: int, num_experts: int) -> dict[str, Any]:
    assignments = FIXED_BATCH_SIZE * top_k
    if not np.all(loads.sum(axis=-1) == assignments):
        raise ValueError("per-layer assignment conservation failed")
    layer_max = loads.max(axis=-1).astype(np.float64)
    active = np.count_nonzero(loads, axis=-1).astype(np.float64)
    hhi = np.square(loads.astype(np.float64) / assignments).sum(axis=-1)
    overlap_denominator = math.comb(FIXED_BATCH_SIZE, 2) * top_k
    pair_overlap = (
        (loads.astype(np.int64) * (loads.astype(np.int64) - 1) // 2).sum(axis=-1)
        / overlap_denominator
    )
    return {
        "worst_layer_max_load": int(layer_max.max()),
        "p95_layer_max_load": float(np.quantile(layer_max, 0.95)),
        "mean_layer_max_load": float(layer_max.mean()),
        "mean_layer_max_concentration": float(layer_max.mean() / assignments),
        "mean_layer_hhi": float(hhi.mean()),
        "mean_layer_active_experts": float(active.mean()),
        "mean_layer_active_expert_fraction": float(active.mean() / num_experts),
        "mean_pairwise_route_overlap_fraction": float(pair_overlap.mean()),
    }


def _partition_signature(partition: Sequence[Sequence[int]]) -> tuple[tuple[int, ...], ...]:
    return tuple(sorted(tuple(sorted(batch)) for batch in partition))


def _seeded_random_partition(request_count: int, seed: int) -> list[list[int]]:
    order = list(range(request_count))
    random.Random(seed).shuffle(order)
    partition = [order[start:start + FIXED_BATCH_SIZE]
                 for start in range(0, request_count, FIXED_BATCH_SIZE)]
    _validate_partition(partition, request_count)
    return partition


def _greedy_partition(
    request_counts: np.ndarray,
    objective: Callable[[np.ndarray], tuple[int, ...]],
) -> list[list[int]]:
    request_count, num_layers, num_experts = request_counts.shape
    pool_load = request_counts.sum(axis=0)
    hotness = np.einsum("nle,le->n", request_counts, pool_load, dtype=np.int64)
    order = sorted(range(request_count), key=lambda index: (-int(hotness[index]), index))
    partition: list[list[int]] = [[] for _ in range(FIXED_GROUPS)]
    loads = np.zeros((FIXED_GROUPS, num_layers, num_experts), dtype=np.int16)
    for request in order:
        best: tuple[tuple[int, ...], int, int] | None = None
        for batch in range(FIXED_GROUPS):
            if len(partition[batch]) >= FIXED_BATCH_SIZE:
                continue
            loads[batch] += request_counts[request]
            candidate = (objective(loads), len(partition[batch]), batch)
            loads[batch] -= request_counts[request]
            if best is None or candidate < best:
                best = candidate
        if best is None:  # pragma: no cover
            raise RuntimeError("no batch capacity remains")
        batch = best[2]
        partition[batch].append(request)
        loads[batch] += request_counts[request]
    _validate_partition(partition, request_count)
    return partition


def _swap_descent(
    request_counts: np.ndarray,
    start: Sequence[Sequence[int]],
    objective: Callable[[np.ndarray], tuple[int, ...]],
) -> tuple[list[list[int]], dict[str, Any]]:
    """Deterministic bounded pair-exchange descent; not an exact solver."""
    partition = [list(batch) for batch in start]
    loads = _loads(request_counts, partition)
    current = objective(loads)
    swaps = 0
    converged = False
    for pass_index in range(MAX_SWAP_PASSES):
        changed = False
        for left_batch in range(FIXED_GROUPS):
            for right_batch in range(left_batch + 1, FIXED_GROUPS):
                best_objective = current
                best_swap: tuple[int, int, np.ndarray, np.ndarray] | None = None
                for left_pos, left_request in enumerate(partition[left_batch]):
                    for right_pos, right_request in enumerate(partition[right_batch]):
                        left_load = (
                            loads[left_batch]
                            - request_counts[left_request]
                            + request_counts[right_request]
                        )
                        right_load = (
                            loads[right_batch]
                            - request_counts[right_request]
                            + request_counts[left_request]
                        )
                        saved_left = loads[left_batch].copy()
                        saved_right = loads[right_batch].copy()
                        loads[left_batch] = left_load
                        loads[right_batch] = right_load
                        candidate = objective(loads)
                        loads[left_batch] = saved_left
                        loads[right_batch] = saved_right
                        if candidate < best_objective:
                            best_objective = candidate
                            best_swap = (left_pos, right_pos, left_load, right_load)
                if best_swap is not None:
                    left_pos, right_pos, left_load, right_load = best_swap
                    partition[left_batch][left_pos], partition[right_batch][right_pos] = (
                        partition[right_batch][right_pos], partition[left_batch][left_pos]
                    )
                    loads[left_batch], loads[right_batch] = left_load, right_load
                    current = best_objective
                    swaps += 1
                    changed = True
        if not changed:
            converged = True
            break
    _validate_partition(partition, len(request_counts))
    if objective(_loads(request_counts, partition)) != current:
        raise AssertionError("incremental swap accounting drift")
    return partition, {
        "algorithm": "deterministic_pair_exchange_descent",
        "maximum_passes": MAX_SWAP_PASSES,
        "passes_executed": pass_index + 1,
        "accepted_swaps": swaps,
        "converged_no_improving_pair": converged,
        "global_optimality_claimed": False,
    }


def _stable_seed(base_seed: int, prompt_length: int) -> int:
    # Each predeclared v3 route-blind partition is static across decode steps and
    # process repeats.  Repeat comparisons therefore replay the same action.
    text = f"{base_seed}:{prompt_length}".encode()
    return int.from_bytes(hashlib.sha256(text).digest()[:8], "big")


def _round_robin_partition(request_count: int) -> list[list[int]]:
    partition = [list(range(batch, request_count, FIXED_GROUPS))
                 for batch in range(FIXED_GROUPS)]
    _validate_partition(partition, request_count)
    return partition


def _hash_partition(request_keys: Sequence[str], prompt_length: int) -> list[list[int]]:
    order = sorted(
        range(len(request_keys)),
        key=lambda index: hashlib.sha256(
            f"{RANDOM_SEED}:{prompt_length}:{request_keys[index]}".encode()
        ).digest(),
    )
    partition = [order[start:start + FIXED_BATCH_SIZE]
                 for start in range(0, len(order), FIXED_BATCH_SIZE)]
    _validate_partition(partition, len(request_keys))
    return partition


def _policy_partitions(
    all_steps: np.ndarray,
    step: int,
    process_repeat: int,
    prompt_length: int,
    request_keys: Sequence[str] | None = None,
) -> tuple[dict[str, list[list[int]]], dict[str, Any]]:
    request_count = len(all_steps)
    keys = list(request_keys) if request_keys is not None else [str(index) for index in range(request_count)]
    if len(keys) != request_count:
        raise ValueError("request key count mismatch")
    original = [list(range(group * FIXED_BATCH_SIZE, (group + 1) * FIXED_BATCH_SIZE))
                for group in range(FIXED_GROUPS)]
    route_blind = {
        "original": original,
        "route_blind_round_robin": _round_robin_partition(request_count),
        "route_blind_hash": _hash_partition(keys, prompt_length),
    }
    for index, base_seed in enumerate(ROUTE_BLIND_SHUFFLE_SEEDS):
        route_blind[f"route_blind_shuffle_s{index:02d}"] = _seeded_random_partition(
            request_count, _stable_seed(base_seed, prompt_length)
        )
    current = all_steps[:, step]
    simple = _greedy_partition(current, _balance_objective)
    history = original if step == 0 else _greedy_partition(
        all_steps[:, step - 1], _balance_objective
    )
    simple_candidates = {
        **route_blind,
        "simple_greedy": simple,
        "history_greedy_tminus1": history,
    }
    strongest = min(
        simple_candidates,
        key=lambda name: (
            _balance_objective(_loads(current, simple_candidates[name])),
            _partition_signature(simple_candidates[name]),
            name,
        ),
    )
    local_search, balance_search = _swap_descent(
        current, simple_candidates[strongest], _balance_objective
    )
    coalesce_start = min(
        (*route_blind.values(), _greedy_partition(current, _coalesce_objective)),
        key=lambda part: (_coalesce_objective(_loads(current, part)), _partition_signature(part)),
    )
    coalesce, coalesce_search = _swap_descent(
        current, coalesce_start, _coalesce_objective
    )
    return {
        **simple_candidates,
        "future_route_local_search": local_search,
        "working_set_coalesce": coalesce,
    }, {
        "strongest_simple_balance_baseline": strongest,
        "balance_local_search": balance_search,
        "working_set_coalesce_search": coalesce_search,
        "history_signal_step": None if step == 0 else step - 1,
        "history_step_zero_fallback": "original" if step == 0 else None,
    }


def _load_bundle(path: Path) -> tuple[dict[str, Any] | None, list[str]]:
    integrity = COMPARATOR.verify_bundle(path)
    errors = [f"{path}:{error}" for error in integrity["errors"]]
    if not integrity["valid"]:
        return None, errors
    config = _json(path / "config.json")
    rows = _jsonl(path / "batches.jsonl")
    if not config.get("capture_routes"):
        errors.append(f"{path}:not_a_route_ON_bundle")
    if FIXED_BATCH_SIZE not in list(map(int, config.get("batch_sizes", []))):
        errors.append(f"{path}:missing_B16")
    if int(config.get("groups", -1)) != FIXED_GROUPS:
        errors.append(f"{path}:expected_exactly_{FIXED_GROUPS}_groups")
    if int(config.get("within_process_repeats", -1)) != 1:
        errors.append(f"{path}:expected_one_within_process_repeat")
    repeat = int(config.get("process_repeat", -1))
    if repeat < 0:
        errors.append(f"{path}:invalid_process_repeat")
    selected = [row for row in rows if int(row.get("batch_size", -1)) == FIXED_BATCH_SIZE]
    for row in selected:
        prompt = int(row.get("prompt_length", -1))
        group = int(row.get("group", -1))
        within = int(row.get("within_process_repeat", -1))
        row_repeat = int(row.get("process_repeat", -1))
        expected_id = f"r{repeat:02d}-p{prompt}-b{FIXED_BATCH_SIZE}-g{group:02d}-w00"
        if (
            prompt not in list(map(int, config.get("prompt_lengths", [])))
            or group not in range(FIXED_GROUPS)
            or within != 0
            or row_repeat != repeat
            or int(row.get("batch_size", -1)) != FIXED_BATCH_SIZE
            or row.get("batch_id") != expected_id
        ):
            errors.append(f"{path}:B16_row_provenance:{row.get('batch_id')}")
    row_map = {
        (int(row["prompt_length"]), int(row["group"])): row for row in selected
    }
    expected = {
        (int(prompt), group)
        for prompt in config.get("prompt_lengths", [])
        for group in range(FIXED_GROUPS)
    }
    if len(row_map) != len(selected) or set(row_map) != expected:
        errors.append(f"{path}:incomplete_or_duplicate_B16_coverage")
    return {
        "path": path,
        "config": config,
        "repeat": repeat,
        "rows": row_map,
        "integrity": integrity,
    }, errors


def _load_pool(bundle: dict[str, Any], prompt_length: int) -> tuple[np.ndarray, list[dict[str, Any]]]:
    config, root = bundle["config"], bundle["path"]
    shape = config["model_shape"]
    layers, top_k, experts = (int(shape["num_layers"]), int(shape["top_k"]),
                              int(shape["num_experts"]))
    expected_route = (FIXED_BATCH_SIZE, int(config["output_tokens"]) - 1, layers, top_k)
    routes, catalog = [], []
    for group in range(FIXED_GROUPS):
        row = bundle["rows"][(prompt_length, group)]
        route_path, input_path = root / row["route_artifact"], root / row["input_artifact"]
        if _sha256(route_path) != row["route_artifact_sha256"]:
            raise ValueError(f"route row hash mismatch:{route_path}")
        if _sha256(input_path) != row["input_artifact_sha256"]:
            raise ValueError(f"input row hash mismatch:{input_path}")
        with np.load(route_path, allow_pickle=False) as payload:
            route = np.asarray(payload["routes"])
        with np.load(input_path, allow_pickle=False) as payload:
            prompts = np.asarray(payload["prompt_token_ids"])
        if route.shape != expected_route:
            raise ValueError(f"route shape {route.shape}, expected {expected_route}")
        if prompts.shape != (FIXED_BATCH_SIZE, prompt_length):
            raise ValueError(f"prompt shape {prompts.shape}")
        if route.min() < 0 or route.max() >= experts:
            raise ValueError("expert id out of model range")
        if np.any(np.diff(np.sort(route, axis=-1), axis=-1) == 0):
            raise ValueError("top-k contains a duplicate expert; overlap accounting invalid")
        routes.append(route)
        for index, prompt in enumerate(prompts):
            catalog.append({
                "pool_index": group * FIXED_BATCH_SIZE + index,
                "source_group": group,
                "source_request_index": index,
                "prompt_token_ids_sha256": hashlib.sha256(prompt.tobytes()).hexdigest(),
            })
    stacked = np.concatenate(routes, axis=0)
    one_hot = np.eye(experts, dtype=np.int8)[stacked]
    counts = one_hot.sum(axis=-2, dtype=np.int16)
    return counts, catalog


def _compatibility(bundles: Sequence[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    repeats = [bundle["repeat"] for bundle in bundles]
    if len(repeats) != len(set(repeats)):
        errors.append(f"duplicate_process_repeats:{repeats}")
    if bundles:
        reference = bundles[0]
        for bundle in bundles[1:]:
            drift = {
                field: [reference["config"].get(field), bundle["config"].get(field)]
                for field in COMPATIBILITY_FIELDS
                if reference["config"].get(field) != bundle["config"].get(field)
            }
            if drift:
                errors.append(f"{bundle['path']}:config_drift:{json.dumps(drift, sort_keys=True)}")
    return errors


def _effect(before: dict[str, Any], after: dict[str, Any]) -> dict[str, float]:
    return {
        "worst_layer_max_load_reduction_pct": _pct_reduction(
            before["worst_layer_max_load"], after["worst_layer_max_load"]
        ),
        "mean_layer_max_load_reduction_pct": _pct_reduction(
            before["mean_layer_max_load"], after["mean_layer_max_load"]
        ),
        "hhi_reduction_pct": _pct_reduction(
            before["mean_layer_hhi"], after["mean_layer_hhi"]
        ),
        "active_expert_increase_pct": _pct_increase(
            before["mean_layer_active_experts"], after["mean_layer_active_experts"]
        ),
        "active_expert_reduction_pct": _pct_reduction(
            before["mean_layer_active_experts"], after["mean_layer_active_experts"]
        ),
        "route_overlap_increase_pct": _pct_increase(
            before["mean_pairwise_route_overlap_fraction"],
            after["mean_pairwise_route_overlap_fraction"],
        ),
    }


def _coassignment_jaccard(
    left: Sequence[Sequence[int]], right: Sequence[Sequence[int]]
) -> float:
    def pairs(partition: Sequence[Sequence[int]]) -> set[tuple[int, int]]:
        return {pair for batch in partition for pair in combinations(sorted(batch), 2)}
    left_pairs, right_pairs = pairs(left), pairs(right)
    return len(left_pairs & right_pairs) / len(left_pairs | right_pairs)


def _canonical_transfer(
    cells: Sequence[dict[str, Any]],
    counts_by_pool: dict[tuple[int, int], np.ndarray],
    repeats: Sequence[int],
    top_k: int,
    num_experts: int,
) -> dict[str, Any]:
    """Freeze the first repeat's partitions and score them on later routes."""
    source_repeat = min(repeats)
    cell_map = {
        (cell["process_repeat"], cell["prompt_length"], cell["decode_route_step"]): cell
        for cell in cells
    }
    policies = ("simple_greedy", "future_route_local_search", "working_set_coalesce")
    rows = []
    for target_repeat in repeats:
        for prompt in sorted({cell["prompt_length"] for cell in cells}):
            all_steps = counts_by_pool[(target_repeat, prompt)]
            for step in range(all_steps.shape[1]):
                source = cell_map[(source_repeat, prompt, step)]
                target = cell_map[(target_repeat, prompt, step)]
                current = all_steps[:, step]
                original = target["policies"]["original"]["metrics"]
                for policy in policies:
                    partition = source["policies"][policy]["partition"]
                    transferred = _partition_metrics(
                        _loads(current, partition), top_k, num_experts
                    )
                    rows.append({
                        "source_process_repeat": source_repeat,
                        "target_process_repeat": target_repeat,
                        "prompt_length": prompt,
                        "decode_route_step": step,
                        "policy": policy,
                        "source_partition_sha256": source["policies"][policy]["partition_sha256"],
                        "transfer_vs_target_original": _effect(original, transferred),
                        "target_refit_vs_transfer": _effect(
                            transferred, target["policies"][policy]["metrics"]
                        ),
                    })
    holdout = [row for row in rows if row["target_process_repeat"] != source_repeat]
    summaries: dict[str, Any] = {}
    for policy in policies:
        selected = [row for row in holdout if row["policy"] == policy]
        primary = (
            "active_expert_reduction_pct"
            if policy == "working_set_coalesce"
            else "mean_layer_max_load_reduction_pct"
        )
        summaries[policy] = {
            "primary_metric": primary,
            "transfer_vs_target_original": _distribution([
                row["transfer_vs_target_original"][primary] for row in selected
            ]),
            "target_refit_residual_over_transfer": _distribution([
                row["target_refit_vs_transfer"][primary] for row in selected
            ]),
            "positive_holdout_cell_fraction": (
                sum(row["transfer_vs_target_original"][primary] > 0 for row in selected)
                / len(selected)
            ),
        }
    return {
        "canonical_process_repeat": source_repeat,
        "holdout_process_repeats": [repeat for repeat in repeats if repeat != source_repeat],
        "note": "The partition is frozen from the canonical repeat; routes are still fixed replay outcomes, not action-conditioned reruns.",
        "policy_summaries": summaries,
        "cells": rows,
    }


def _trajectory_policy(cells: Sequence[dict[str, Any]], policy: str) -> dict[str, Any]:
    metric_names = tuple(cells[0]["policies"][policy]["metrics"])
    metrics = {
        name: (
            max(cell["policies"][policy]["metrics"][name] for cell in cells)
            if name == "worst_layer_max_load"
            else float(np.mean([cell["policies"][policy]["metrics"][name] for cell in cells]))
        )
        for name in metric_names
    }
    balance = [cell["policies"][policy]["balance_objective"] for cell in cells]
    coalesce = [cell["policies"][policy]["working_set_objective"] for cell in cells]
    return {
        "metrics": metrics,
        "balance_lex_objective": [
            max(row[0] for row in balance),
            sum(row[1] for row in balance),
            sum(row[2] for row in balance),
            sum(row[3] for row in balance),
        ],
        "working_set_lex_objective": [
            sum(row[0] for row in coalesce),
            max(row[1] for row in coalesce),
            sum(row[2] for row in coalesce),
        ],
    }


def _trajectory_analysis(
    cells: Sequence[dict[str, Any]], repeats: Sequence[int]
) -> dict[str, Any]:
    trajectories = []
    prompts = sorted({cell["prompt_length"] for cell in cells})
    for repeat in repeats:
        for prompt in prompts:
            selected = sorted(
                [cell for cell in cells if cell["process_repeat"] == repeat
                 and cell["prompt_length"] == prompt],
                key=lambda cell: cell["decode_route_step"],
            )
            if not selected:
                raise ValueError(f"missing trajectory:r{repeat}:p{prompt}")
            policies = {policy: _trajectory_policy(selected, policy) for policy in ALL_POLICIES}
            balance_blind = min(
                ROUTE_BLIND_POLICIES,
                key=lambda policy: (policies[policy]["balance_lex_objective"], policy),
            )
            working_blind = min(
                ROUTE_BLIND_POLICIES,
                key=lambda policy: (policies[policy]["working_set_lex_objective"], policy),
            )
            balance_reference = policies[balance_blind]["metrics"]
            working_reference = policies[working_blind]["metrics"]
            trajectories.append({
                "process_repeat": repeat,
                "prompt_length": prompt,
                "decode_route_steps": len(selected),
                "strongest_route_blind_balance_baseline": balance_blind,
                "strongest_route_blind_working_set_baseline": working_blind,
                "policies": policies,
                "effects_vs_strongest_route_blind": {
                    "history_greedy_tminus1": _effect(
                        balance_reference, policies["history_greedy_tminus1"]["metrics"]
                    ),
                    "simple_greedy_hindsight": _effect(
                        balance_reference, policies["simple_greedy"]["metrics"]
                    ),
                    "future_route_local_search": _effect(
                        balance_reference, policies["future_route_local_search"]["metrics"]
                    ),
                    "working_set_coalesce": _effect(
                        working_reference, policies["working_set_coalesce"]["metrics"]
                    ),
                },
            })

    def summarize(policy: str, metric: str) -> dict[str, Any]:
        values = [row["effects_vs_strongest_route_blind"][policy][metric] for row in trajectories]
        return {
            "distribution": _distribution(values),
            "positive_trajectory_fraction": sum(value > 0 for value in values) / len(values),
        }

    history_primary = summarize("history_greedy_tminus1", "mean_layer_max_load_reduction_pct")
    history_hhi = summarize("history_greedy_tminus1", "hhi_reduction_pct")
    local_primary = summarize("future_route_local_search", "mean_layer_max_load_reduction_pct")
    coalesce_primary = summarize("working_set_coalesce", "active_expert_reduction_pct")

    def median(summary: dict[str, Any]) -> float:
        value = summary["distribution"]["median"]
        return float(value) if value is not None else -math.inf

    history_material = bool(
        median(history_primary) >= THRESHOLDS["material_trajectory_mean_max_load_reduction_pct"]
        and history_primary["positive_trajectory_fraction"]
        >= THRESHOLDS["minimum_positive_trajectory_fraction"]
        and median(history_hhi) >= THRESHOLDS["maximum_secondary_hhi_degradation_pct"]
    )
    local_material = bool(
        median(local_primary) >= THRESHOLDS["material_trajectory_mean_max_load_reduction_pct"]
        and local_primary["positive_trajectory_fraction"]
        >= THRESHOLDS["minimum_positive_trajectory_fraction"]
    )
    coalesce_material = bool(
        median(coalesce_primary) >= THRESHOLDS["material_trajectory_active_expert_reduction_pct"]
        and coalesce_primary["positive_trajectory_fraction"]
        >= THRESHOLDS["minimum_positive_trajectory_fraction"]
    )
    balance_wins = {
        policy: sum(row["strongest_route_blind_balance_baseline"] == policy for row in trajectories)
        for policy in ROUTE_BLIND_POLICIES
    }
    working_wins = {
        policy: sum(row["strongest_route_blind_working_set_baseline"] == policy for row in trajectories)
        for policy in ROUTE_BLIND_POLICIES
    }
    return {
        "verdict": "BUDGET_GATE_COMPOSITION_DIAGNOSTIC_ONLY",
        "trajectory_count": len(trajectories),
        "evidence_unit": "process_repeat_x_prompt_length_full_decode_trajectory",
        "independence_warning": (
            "The six summaries are descriptive, not six independent workload samples: "
            "adjacent steps are correlated, P128 is a prefix-related view of P512, and all use one model/pool."
        ),
        "strongest_route_blind_baselines": {
            "balance_trajectory_win_counts": balance_wins,
            "working_set_trajectory_win_counts": working_wins,
            "selection_scope": "one fixed policy per full 15-step trajectory; never a per-step best-of baseline",
        },
        "history_vs_strongest_route_blind": {
            "verdict": (
                "MATERIAL_FIXED_TRACE_HISTORY_RESIDUAL"
                if history_material else "NO_MATERIAL_FIXED_TRACE_HISTORY_RESIDUAL"
            ),
            "mean_layer_max_load_reduction_pct": history_primary,
            "hhi_reduction_pct": history_hhi,
            "note": "This is the only t-1 pre-action-style route comparison; it is still fixed-trace replay.",
        },
        "hindsight_local_search_vs_strongest_route_blind": {
            "verdict": (
                "MATERIAL_FIXED_TRACE_REGROUPING_POTENTIAL"
                if local_material else "NO_MATERIAL_FIXED_TRACE_REGROUPING_POTENTIAL"
            ),
            "descriptive_mean_layer_max_load_reduction_pct": local_primary,
            "note": (
                "The search optimizes the reported lexicographic objective components. "
                "Mean layer max load is a descriptive post-search metric, not a mathematical bound."
            ),
        },
        "working_set_coalesce_vs_strongest_route_blind": {
            "verdict": (
                "MATERIAL_FIXED_TRACE_REGROUPING_POTENTIAL"
                if coalesce_material else "NO_MATERIAL_FIXED_TRACE_REGROUPING_POTENTIAL"
            ),
            "active_expert_reduction_pct": coalesce_primary,
            "route_overlap_increase_pct": summarize(
                "working_set_coalesce", "route_overlap_increase_pct"
            ),
            "hhi_reduction_pct": summarize("working_set_coalesce", "hhi_reduction_pct"),
        },
        "trajectories": trajectories,
    }


def _step_diagnostics(cells: Sequence[dict[str, Any]], repeats: Sequence[int]) -> dict[str, Any]:
    partition_jaccards: dict[str, list[float]] = {name: [] for name in ALL_POLICIES}
    by_key = {
        (cell["process_repeat"], cell["prompt_length"], cell["decode_route_step"]): cell
        for cell in cells
    }
    for prompt in sorted({cell["prompt_length"] for cell in cells}):
        for step in sorted({cell["decode_route_step"] for cell in cells}):
            for left_repeat, right_repeat in combinations(repeats, 2):
                left, right = by_key[(left_repeat, prompt, step)], by_key[(right_repeat, prompt, step)]
                for policy in ALL_POLICIES:
                    partition_jaccards[policy].append(_coassignment_jaccard(
                        left["policies"][policy]["partition"],
                        right["policies"][policy]["partition"],
                    ))
    return {
        "role": "CORRELATED_STEP_LEVEL_DESCRIPTIVE_DIAGNOSTIC_ONLY",
        "step_cell_count": len(cells),
        "not_an_evidence_sample_count": True,
        "partition_coassignment_jaccard": {
            policy: _distribution(values) for policy, values in partition_jaccards.items()
        },
        "search_convergence": {
            "balance_local_search_converged_cells": sum(
                cell["balance_local_search"]["converged_no_improving_pair"] for cell in cells
            ),
            "working_set_coalesce_converged_cells": sum(
                cell["working_set_coalesce_search"]["converged_no_improving_pair"] for cell in cells
            ),
            "total_cells": len(cells),
        },
    }


def analyze_bundles(paths: Sequence[Path]) -> dict[str, Any]:
    resolved = [Path(path).resolve() for path in paths]
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "claim_ceiling": CLAIM_CEILING,
        "claim_ceiling_semantics": (
            "fixed-trace hindsight diagnostic only; neither a mathematical bound "
            "nor evidence of action-conditioned scheduling gain"
        ),
        "verdict": "BUDGET_GATE_COMPOSITION_DIAGNOSTIC_ONLY",
        "input_bundles": list(map(str, resolved)),
        "thresholds": THRESHOLDS,
        "fixed_design": {
            "batch_size": FIXED_BATCH_SIZE,
            "groups_per_pool": FIXED_GROUPS,
            "requests_per_pool": FIXED_BATCH_SIZE * FIXED_GROUPS,
            "random_seed": RANDOM_SEED,
            "predeclared_for_v3_reanalysis_static_shuffle_seeds": list(ROUTE_BLIND_SHUFFLE_SEEDS),
            "route_blind_partition_scope": "static across all decode steps and process repeats",
            "balance_objective_lexicographic": [
                "worst per-layer max expert load",
                "sum of per-layer max expert loads",
                "sum of squared expert loads (HHI numerator)",
                "negative active-expert count",
            ],
            "working_set_objective_lexicographic": [
                "sum of per-layer active experts",
                "worst per-layer active experts",
                "negative sum of squared loads (route overlap proxy)",
            ],
        },
        "anti_claims": [
            "the 96-request pool is synthesized from six separately executed B16 generate calls, not an observed simultaneous continuous-batching ready set",
            "the v3 shuffle ladder was declared before this reanalysis but after the sealed data existed; it is deterministic, not an experiment preregistration",
            "regrouping reuses captured routes instead of rerunning the model",
            "regrouping can change hidden states, output tokens, and future routes",
            "fixed-trace replay is neither a mathematical upper bound nor lower bound on an action-conditioned rerun",
            "future_route_local_search and simple_greedy consume current-step future route information",
            "history_greedy_tminus1 is evaluated on fixed current-step routes, not an action-conditioned rerun",
            "the pair-exchange search is deterministic but not globally optimal",
            "no TPOT, latency, throughput, SLO-goodput, or controller headroom is claimed",
            "90 adjacent step cells are correlated diagnostics, not 90 independent evidence samples",
            "P128 and P512 are prefix-related views of the same underlying pool",
            "single-GPU structural results are not Expert Parallel rank-pressure evidence",
            "balance and working-set coalescing are opposite objectives and are not combined into one optimum",
        ],
    }
    bundles, errors = [], []
    try:
        for path in resolved:
            bundle, found = _load_bundle(path)
            errors.extend(found)
            if bundle is not None:
                bundles.append(bundle)
        errors.extend(_compatibility(bundles))
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        errors.append(str(exc))
    if errors:
        report.update({"status": "INVALID_INPUT", "validation_errors": errors})
        return report
    if len(bundles) < MINIMUM_PROCESS_REPEATS:
        report.update({
            "status": "INSUFFICIENT_EVIDENCE",
            "validation_errors": [f"need at least {MINIMUM_PROCESS_REPEATS} process repeats"],
        })
        return report

    repeats = sorted(bundle["repeat"] for bundle in bundles)
    prompts = sorted(map(int, bundles[0]["config"]["prompt_lengths"]))
    cells, catalogs, reference_catalogs, counts_by_pool = [], {}, {}, {}
    try:
        for bundle in sorted(bundles, key=lambda item: item["repeat"]):
            for prompt in prompts:
                all_steps, catalog = _load_pool(bundle, prompt)
                digests = [row["prompt_token_ids_sha256"] for row in catalog]
                if prompt in reference_catalogs and reference_catalogs[prompt] != digests:
                    raise ValueError(f"prompt identity drift across process repeats:p{prompt}")
                reference_catalogs[prompt] = digests
                catalogs[str(prompt)] = catalog
                counts_by_pool[(bundle["repeat"], prompt)] = all_steps
                request_keys = [
                    f"g{row['source_group']:02d}-i{row['source_request_index']:02d}-{row['prompt_token_ids_sha256']}"
                    for row in catalog
                ]
                for step in range(all_steps.shape[1]):
                    partitions, search = _policy_partitions(
                        all_steps, step, bundle["repeat"], prompt, request_keys
                    )
                    current = all_steps[:, step]
                    policies: dict[str, Any] = {}
                    for name, partition in partitions.items():
                        loads = _loads(current, partition)
                        policies[name] = {
                            "partition": partition,
                            "partition_sha256": hashlib.sha256(
                                json.dumps(_partition_signature(partition)).encode()
                            ).hexdigest(),
                            "validation": _validate_partition(partition, len(current)),
                            "balance_objective": list(_balance_objective(loads)),
                            "working_set_objective": list(_coalesce_objective(loads)),
                            "metrics": _partition_metrics(
                                loads,
                                int(bundle["config"]["model_shape"]["top_k"]),
                                int(bundle["config"]["model_shape"]["num_experts"]),
                            ),
                        }
                    strongest = search["strongest_simple_balance_baseline"]
                    effects = {
                        "future_route_local_search_vs_original_diagnostic": _effect(
                            policies["original"]["metrics"], policies["future_route_local_search"]["metrics"]
                        ),
                        "future_route_local_search_vs_strongest_simple_diagnostic": _effect(
                            policies[strongest]["metrics"], policies["future_route_local_search"]["metrics"]
                        ),
                        "history_vs_original_diagnostic": _effect(
                            policies["original"]["metrics"], policies["history_greedy_tminus1"]["metrics"]
                        ),
                        "future_greedy_vs_history_diagnostic": _effect(
                            policies["history_greedy_tminus1"]["metrics"], policies["simple_greedy"]["metrics"]
                        ),
                        "working_set_coalesce_vs_original_diagnostic": _effect(
                            policies["original"]["metrics"], policies["working_set_coalesce"]["metrics"]
                        ),
                    }
                    cells.append({
                        "process_repeat": bundle["repeat"],
                        "prompt_length": prompt,
                        "decode_route_step": step,
                        "pool_request_count": len(current),
                        "policies": policies,
                        "effects": effects,
                        **search,
                    })
    except (KeyError, OSError, TypeError, ValueError, AssertionError) as exc:
        report.update({"status": "INVALID_INPUT", "validation_errors": [str(exc)]})
        return report

    trajectory_analysis = _trajectory_analysis(cells, repeats)
    step_diagnostics = _step_diagnostics(cells, repeats)
    step_diagnostics["canonical_partition_transfer_vs_original_diagnostic"] = _canonical_transfer(
        cells,
        counts_by_pool,
        repeats,
        int(bundles[0]["config"]["model_shape"]["top_k"]),
        int(bundles[0]["config"]["model_shape"]["num_experts"]),
    )
    report.update({
        "status": "COMPLETE",
        "process_repeats": repeats,
        "prompt_lengths": prompts,
        "decode_route_steps": int(bundles[0]["config"]["output_tokens"]) - 1,
        "trajectory_count": trajectory_analysis["trajectory_count"],
        "diagnostic_step_cell_count": len(cells),
        "request_catalogs": catalogs,
        "bundle_integrity": [
            {"path": str(bundle["path"]), "process_repeat": bundle["repeat"], **bundle["integrity"]}
            for bundle in bundles
        ],
        "policy_information": {
            "original": "captured six-group membership",
            "route_blind_round_robin": "static pool-index interleave; no route information",
            "route_blind_hash": "static stable-request-key hash grouping; no route information",
            "route_blind_shuffles": "eight static seeds declared for v3 reanalysis; no route information and no pre-data preregistration claim",
            "simple_greedy": "one-pass greedy using current-step future routes",
            "history_greedy_tminus1": "t-1 routes; original partition fallback at t=0",
            "future_route_local_search": "best simple start plus bounded current-step pair-exchange descent; no global optimality claim",
            "working_set_coalesce": "current-step future-route search under the separate coalescing objective",
        },
        "trajectory_analysis": trajectory_analysis,
        "step_level_diagnostics": step_diagnostics,
        "diagnostic_step_cells": cells,
        "one_next_gate": (
            "Use this only to control/stratify composition in the action-conditioned budget Gate; "
            "do not implement a route-aware regrouping controller from fixed-trace replay."
        ),
    })
    report["analysis_script_sha256"] = _sha256(Path(__file__))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route-on", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; structural diagnostic results are write-once")
    report = analyze_bundles(args.route_on)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        output.write(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    print(json.dumps({
        "status": report["status"],
        "claim_ceiling": report["claim_ceiling"],
        "verdict": report.get("verdict"),
        "history_residual": report.get("trajectory_analysis", {}).get("history_vs_strongest_route_blind", {}).get("verdict"),
    }))
    if report["status"] == "INVALID_INPUT":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
