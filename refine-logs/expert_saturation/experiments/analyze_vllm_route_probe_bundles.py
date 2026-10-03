#!/usr/bin/env python3
"""Select the next research pivot from sealed native route-probe bundles.

Route-ON timing is always treated as instrumented diagnostic timing. A route
signal may be associated with request timing for the action screen only when a
sealed, token-identical route-OFF bundle exists for every process repeat. Even
then, the result selects a causal experiment; it is not a controller gain.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import zipfile
from collections import defaultdict
from itertools import combinations, product
from pathlib import Path
from typing import Any, Sequence

import numpy as np
SCHEMA = "vllm-native-route-pivot-analysis-v1"
CLAIM_CEILING = "NATIVE_OFFLINE_FIXED_BATCH_OBSERVATIONAL_PIVOT_ONLY"
PIVOTS = {"STOP_ROUTE_CONTROL", "WORKING_SET_MEASUREMENT_ONLY",
          "TEST_MARGINAL_PRESSURE_ACTION"}
# Fixed before results are analyzed; passing only selects the next A/B test.
THRESHOLDS_V1: dict[str, float | int] = {
    "minimum_process_repeats": 2,
    "minimum_groups_per_cell_per_repeat": 6,
    "minimum_tested_cells": 2,
    "minimum_supportive_cells": 2,
    "minimum_supportive_cell_fraction": 0.50,
    "minimum_pressure_p90_p10_relative": 0.10,
    "minimum_pressure_p90_p10_absolute": 0.01,
    "minimum_tpot_p90_p10_pct": 5.0,
    "minimum_pressure_tpot_spearman": 0.50,
    "minimum_high_pressure_tpot_effect_pct": 3.0,
    "minimum_positive_repeat_fraction": 0.75,
    "minimum_temporal_load_cosine_median": 0.90,
    "minimum_temporal_load_cosine_p10": 0.80,
    "minimum_repeat_route_set_jaccard_median": 0.98,
    "minimum_repeat_route_set_jaccard_p10": 0.95,
    "maximum_repeat_pressure_cv_p90": 0.05,
    "maximum_repeat_tpot_cv_p90": 0.10,
    "maximum_repeat_working_set_cv_p90": 0.05,
    "maximum_repeat_active_fraction_cv_p90": 0.05,
    "minimum_cross_batch_active_fraction_delta": 0.10,
    "minimum_cross_batch_working_set_fraction_delta": 0.05,
    "minimum_cross_batch_material_fraction": 0.75,
    "cross_batch_monotonic_tolerance": 0.02,
}

PRIMARY_PRESSURE = "mean_layer_step_concentration"
COMPATIBILITY_FIELDS = (
    "model", "revision", "dtype", "batch_sizes", "prompt_lengths",
    "output_tokens", "groups", "within_process_repeats", "seed",
    "max_model_len", "max_num_seqs", "max_num_batched_tokens",
    "gpu_memory_utilization", "enforce_eager", "workload_manifest_sha256",
    "runtime_patch_id", "probe_script_sha256", "runtime_identity", "model_shape",
    "producer_source_artifact", "producer_source_artifact_sha256",
    "require_exclusive_gpu",
)
REQUIRED_ROUTE_METRICS = (
    PRIMARY_PRESSURE, "p95_layer_step_concentration", "mean_load_cv",
    "mean_active_expert_fraction", "mean_layer_working_set_fraction",
    "mean_temporal_load_vector_cosine", "mean_temporal_jaccard",
    "mean_per_request_temporal_jaccard",
    "per_request_exact_route_match_fraction",
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


def _load_producer() -> Any:
    path = Path(__file__).with_name("run_vllm_route_shape_probe.py")
    spec = importlib.util.spec_from_file_location("route_probe_producer", path)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PRODUCER = _load_producer()


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _number(value: Any, label: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"non-finite metric:{label}")
    return value


def _q(values: Sequence[float], quantile: float) -> float | None:
    return float(np.quantile(values, quantile)) if values else None


def _median(values: Sequence[float]) -> float | None:
    return float(np.median(values)) if values else None


def _cv(values: Sequence[float]) -> float | None:
    if not values:
        return None
    mean = float(np.mean(values))
    return float(np.std(values) / abs(mean)) if mean else 0.0


def _distribution(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "median": None, "p10": None, "p90": None,
                "p90_p10_absolute": None, "p90_p10_relative": None, "cv": None}
    p10, p90, median = _q(values, 0.10), _q(values, 0.90), _median(values)
    assert p10 is not None and p90 is not None and median is not None
    spread = p90 - p10
    return {"count": len(values), "median": median, "p10": p10, "p90": p90,
            "p90_p10_absolute": spread,
            "p90_p10_relative": spread / abs(median) if median else None,
            "cv": _cv(values)}


def _ranks(values: Sequence[float]) -> np.ndarray:
    data = np.asarray(values, dtype=np.float64)
    order, ranks = np.argsort(data, kind="mergesort"), np.empty(len(data))
    start = 0
    while start < len(data):
        stop = start + 1
        while stop < len(data) and data[order[stop]] == data[order[start]]:
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2.0
        start = stop
    return ranks


def _spearman(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) < 3 or len(left) != len(right):
        return None
    left_ranks, right_ranks = _ranks(left), _ranks(right)
    if np.std(left_ranks) == 0 or np.std(right_ranks) == 0:
        return None
    return float(np.corrcoef(left_ranks, right_ranks)[0, 1])


def _key(row: dict[str, Any]) -> tuple[int, int, int, int]:
    return (int(row["prompt_length"]), int(row["batch_size"]),
            int(row["group"]), int(row["within_process_repeat"]))


def _expected(config: dict[str, Any]) -> set[tuple[int, int, int, int]]:
    return set(product(map(int, config.get("prompt_lengths", [])),
                       map(int, config.get("batch_sizes", [])),
                       range(int(config.get("groups", 0))),
                       range(int(config.get("within_process_repeats", 0)))))


def _metric(row: dict[str, Any], name: str) -> float:
    source = row["timing"] if name == "request_tpot_p95_ms" else row["route"]
    return _number(source[name], name)


def _load_and_recompute_routes(
    path: Path, config: dict[str, Any], row: dict[str, Any]
) -> tuple[np.ndarray, dict[str, Any]]:
    relative = Path(str(row["route_artifact"]))
    route_path = (path / relative).resolve()
    try:
        route_path.relative_to(path.resolve())
    except ValueError as exc:
        raise ValueError(f"route artifact escapes bundle:{relative}") from exc
    if hashlib.sha256(route_path.read_bytes()).hexdigest() != row["route_artifact_sha256"]:
        raise ValueError(f"route record hash mismatch:{route_path}")
    try:
        with np.load(route_path, allow_pickle=False) as payload:
            if set(payload.files) != {"routes"}:
                raise ValueError(f"invalid route NPZ keys:{payload.files}")
            routes = np.asarray(payload["routes"])
    except (EOFError, OSError, ValueError, zipfile.BadZipFile) as exc:
        raise ValueError(
            f"invalid route NPZ:{type(exc).__name__}:{exc}"
        ) from exc
    shape = config["model_shape"]
    expected = (
        int(row["batch_size"]),
        int(config["output_tokens"]) - 1,
        int(shape["num_layers"]),
        int(shape["top_k"]),
    )
    if routes.shape != expected:
        raise ValueError(f"invalid route shape:{routes.shape}:expected:{expected}")
    if not np.issubdtype(routes.dtype, np.integer):
        raise ValueError(f"route dtype must be integer:{routes.dtype}")
    num_experts = int(shape["num_experts"])
    if np.any(routes < 0) or np.any(routes >= num_experts):
        raise ValueError("expert ID outside configured range")
    if routes.shape[-1] > 1 and np.any(np.diff(np.sort(routes, axis=-1), axis=-1) == 0):
        raise ValueError("duplicate expert IDs inside a top-k assignment")
    metrics = PRODUCER.summarize_routes(
        [routes[index] for index in range(routes.shape[0])], num_experts
    )
    reported = row.get("route")
    if not isinstance(reported, dict):
        raise ValueError("missing reported route metrics")
    for name in REQUIRED_ROUTE_METRICS:
        actual = _number(metrics[name], f"recomputed:{name}")
        claimed = _number(reported[name], f"reported:{name}")
        if not math.isclose(actual, claimed, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(
                f"route metric mismatch:{name}:reported={claimed}:recomputed={actual}"
            )
    return routes, metrics


def _load_bundle(path: Path, route_on: bool) -> tuple[dict[str, Any] | None, list[str]]:
    integrity = COMPARATOR.verify_bundle(path)
    errors = [f"{path}:{error}" for error in integrity["errors"]]
    if not integrity["valid"]:
        return None, errors
    config, summary = _json(path / "config.json"), _json(path / "summary.json")
    environment = _json(path / "environment.json")
    rows = _jsonl(path / "batches.jsonl")
    if bool(config.get("capture_routes")) != route_on or bool(summary.get("capture_routes")) != route_on:
        errors.append(f"{path}:capture_routes_mismatch")
    row_map = {_key(row): row for row in rows}
    if len(row_map) != len(rows) or set(row_map) != _expected(config):
        errors.append(f"{path}:incomplete_or_duplicate_cartesian_coverage")
    if int(summary.get("record_count", -1)) != len(rows):
        errors.append(f"{path}:record_count_mismatch")
    repeat = int(config.get("process_repeat", -1))
    route_arrays: dict[tuple[int, int, int, int], np.ndarray] = {}
    for key, row in row_map.items():
        if int(row.get("process_repeat", -2)) != repeat:
            errors.append(f"{path}:row_process_repeat:{list(key)}")
        try:
            if _metric(row, "request_tpot_p95_ms") <= 0 or _number(row["timing"]["wall_ms"], "wall_ms") <= 0:
                raise ValueError("timing denominator must be positive")
            if route_on:
                for name in REQUIRED_ROUTE_METRICS:
                    _metric(row, name)
                if not row.get("route_artifact") or not row.get("route_artifact_sha256"):
                    raise ValueError("missing route artifact reference")
                routes, recomputed = _load_and_recompute_routes(path, config, row)
                route_arrays[key] = routes
                # Every downstream decision consumes recomputed raw-artifact
                # metrics, never a trusted precomputed JSON summary.
                row["route"] = recomputed
        except (
            EOFError,
            KeyError,
            OSError,
            TypeError,
            ValueError,
            zipfile.BadZipFile,
        ) as exc:
            errors.append(f"{path}:metric:{list(key)}:{exc}")
    return {"path": path, "config": config, "rows": rows, "map": row_map,
            "route_arrays": route_arrays, "repeat": repeat,
            "environment": environment, "integrity": integrity}, errors


def _load_sets(paths: Sequence[Path], route_on: bool) -> tuple[list[dict[str, Any]], list[str]]:
    bundles, errors = [], []
    for path in paths:
        bundle, found = _load_bundle(path, route_on)
        errors.extend(found)
        if bundle is not None:
            bundles.append(bundle)
    repeats = [bundle["repeat"] for bundle in bundles]
    if len(repeats) != len(set(repeats)) or any(repeat < 0 for repeat in repeats):
        errors.append(f"duplicate_or_invalid_process_repeats:{repeats}")
    if bundles:
        reference = bundles[0]
        for bundle in bundles[1:]:
            drift = {field: [reference["config"].get(field), bundle["config"].get(field)]
                     for field in COMPATIBILITY_FIELDS
                     if reference["config"].get(field) != bundle["config"].get(field)}
            if drift:
                errors.append(f"{bundle['path']}:config_drift:{json.dumps(drift, sort_keys=True)}")
            for key in set(reference["map"]) & set(bundle["map"]):
                if reference["map"][key].get("prompt_token_ids_sha256") != bundle["map"][key].get("prompt_token_ids_sha256"):
                    errors.append(f"{bundle['path']}:prompt_digest_drift:{list(key)}")
    return bundles, errors


def _pair_off(on: Sequence[dict[str, Any]], off: Sequence[dict[str, Any]]) -> tuple[dict[int, dict[str, Any]], list[dict[str, Any]], list[str]]:
    available, qualified, reports, errors = {bundle["repeat"]: bundle for bundle in off}, {}, [], []
    if off and set(available) != {bundle["repeat"] for bundle in on}:
        errors.append("route_OFF_process_repeats_do_not_match_route_ON")
    for on_bundle in on:
        off_bundle = available.get(on_bundle["repeat"])
        if off_bundle is None:
            continue
        comparison = COMPARATOR.compare_runs(on_bundle["config"], off_bundle["config"],
                                             on_bundle["rows"], off_bundle["rows"], 5.0)
        pair_structure_invalid = bool(
            comparison["config_drift"] or comparison["duplicate_keys"]
            or comparison["missing_on"] or comparison["missing_off"]
            or comparison["incomplete_on"] or comparison["incomplete_off"]
            or comparison["unexpected_on"] or comparison["unexpected_off"]
            or comparison["prompt_digest_mismatches"]
        )
        token_drift = bool(comparison["token_mismatches"])
        status = "TELEMETRY_TOKEN_DRIFT" if token_drift and not pair_structure_invalid else comparison["status"]
        reports.append({"process_repeat": on_bundle["repeat"], "status": status,
                        "pair_count": comparison["pair_count"], "token_parity": comparison["token_parity"],
                        "token_mismatch_count": len(comparison["token_mismatches"]),
                        "token_mismatch_keys": comparison["token_mismatches"],
                        "wall_overhead_p95_pct": comparison["wall_overhead_p95_pct"],
                        "tpot_p95_overhead_p95_pct": comparison["tpot_p95_overhead_p95_pct"]})
        if pair_structure_invalid or (comparison["status"] == "INVALID_TELEMETRY_PAIR" and not token_drift):
            errors.append(f"invalid_ON_OFF_pair:process_repeat={on_bundle['repeat']}")
        elif comparison["status"] == "TELEMETRY_OVERHEAD_QUALIFIED":
            qualified[on_bundle["repeat"]] = off_bundle
    return qualified, reports, errors


def _group_means(rows: Sequence[dict[str, Any]], name: str) -> dict[int, float]:
    values: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        values[int(row["group"])].append(_metric(row, name))
    return {group: float(np.mean(items)) for group, items in values.items()}


def _cell_repeat(route_rows: Sequence[dict[str, Any]], timing_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    pressure = _group_means(route_rows, PRIMARY_PRESSURE)
    tpot = _group_means(timing_rows, "request_tpot_p95_ms")
    groups = sorted(pressure)
    if set(groups) != set(tpot):
        raise ValueError("route/timing group mismatch")
    split = max(1, len(groups) // 3)
    ordered = sorted(groups, key=lambda group: (pressure[group], group))
    low, high = ordered[:split], ordered[-split:]
    low_tpot = np.mean([tpot[group] for group in low])
    high_tpot = np.mean([tpot[group] for group in high])
    return {"groups": len(groups), "pressure": _distribution([pressure[group] for group in groups]),
            "tpot_p95_ms": _distribution([tpot[group] for group in groups]),
            "pressure_to_tpot_spearman": _spearman([pressure[group] for group in groups], [tpot[group] for group in groups]),
            "high_pressure_minus_low_pressure_tpot_pct": 100.0 * (high_tpot / low_tpot - 1.0) if low_tpot else None}


def _fixed_cells(on: Sequence[dict[str, Any]], timing: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    cells: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for bundle in on:
        timing_bundle = timing.get(bundle["repeat"], bundle)
        for prompt, batch in sorted({key[:2] for key in bundle["map"]}):
            route_rows = [row for key, row in bundle["map"].items() if key[:2] == (prompt, batch)]
            timing_rows = [row for key, row in timing_bundle["map"].items() if key[:2] == (prompt, batch)]
            item = _cell_repeat(route_rows, timing_rows)
            item["process_repeat"] = bundle["repeat"]
            item["timing_source"] = "ROUTE_OFF" if bundle["repeat"] in timing else "ROUTE_ON_DIAGNOSTIC"
            cells[(prompt, batch)].append(item)
    result, t = [], THRESHOLDS_V1
    for (prompt, batch), repeats in sorted(cells.items()):
        rel = [row["pressure"]["p90_p10_relative"] for row in repeats if row["pressure"]["p90_p10_relative"] is not None]
        absolute = [row["pressure"]["p90_p10_absolute"] for row in repeats if row["pressure"]["p90_p10_absolute"] is not None]
        off_repeats = [row for row in repeats if row["timing_source"] == "ROUTE_OFF"]
        spread = [100.0 * row["tpot_p95_ms"]["p90_p10_relative"] for row in off_repeats if row["tpot_p95_ms"]["p90_p10_relative"] is not None]
        corr = [row["pressure_to_tpot_spearman"] for row in off_repeats if row["pressure_to_tpot_spearman"] is not None]
        effect = [row["high_pressure_minus_low_pressure_tpot_pct"] for row in off_repeats if row["high_pressure_minus_low_pressure_tpot_pct"] is not None]
        positive_corr = sum(value > 0 for value in corr) / len(off_repeats) if off_repeats else 0.0
        positive_effect = sum(value > 0 for value in effect) / len(off_repeats) if off_repeats else 0.0
        enough = all(row["groups"] >= t["minimum_groups_per_cell_per_repeat"] for row in repeats)
        association = bool(enough and len(off_repeats) >= t["minimum_process_repeats"]
                       and len(corr) == len(off_repeats) and len(effect) == len(off_repeats)
                       and (_median(rel) or 0) >= t["minimum_pressure_p90_p10_relative"]
                       and (_median(absolute) or 0) >= t["minimum_pressure_p90_p10_absolute"]
                       and (_median(spread) or 0) >= t["minimum_tpot_p90_p10_pct"]
                       and (_median(corr) if corr else -1) >= t["minimum_pressure_tpot_spearman"]
                       and (_median(effect) if effect else -math.inf) >= t["minimum_high_pressure_tpot_effect_pct"]
                       and min(positive_corr, positive_effect) >= t["minimum_positive_repeat_fraction"])
        result.append({"prompt_length": prompt, "batch_size": batch, "process_repeats": len(repeats),
                       "route_OFF_process_repeats": len(off_repeats),
                       "enough_groups": enough, "median_pressure_p90_p10_relative": _median(rel),
                       "median_pressure_p90_p10_absolute": _median(absolute), "median_tpot_p90_p10_pct": _median(spread),
                       "median_pressure_to_tpot_spearman": _median(corr),
                       "median_high_pressure_minus_low_pressure_tpot_pct": _median(effect),
                       "positive_correlation_repeat_fraction": positive_corr,
                       "positive_effect_repeat_fraction": positive_effect,
                       "supports_composition_association": association,
                       "per_repeat": repeats})
    return result


def _routes(bundle: dict[str, Any], row: dict[str, Any]) -> np.ndarray:
    return bundle["route_arrays"][_key(row)]


def _set_jaccard(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        raise ValueError(f"route shape drift:{left.shape}:{right.shape}")
    scores = []
    for lrow, rrow in zip(left.reshape(-1, left.shape[-1]), right.reshape(-1, right.shape[-1])):
        lset, rset = set(map(int, lrow)), set(map(int, rrow))
        scores.append(len(lset & rset) / len(lset | rset))
    return float(np.mean(scores))


def _stability(on: Sequence[dict[str, Any]], timing: dict[int, dict[str, Any]]) -> dict[str, Any]:
    jaccard, pressure_cv, tpot_cv, working_cv, active_cv = [], [], [], [], []
    keys = sorted(set.intersection(*(set(bundle["map"]) for bundle in on)))
    for key in keys:
        on_rows = [(bundle, bundle["map"][key]) for bundle in on]
        for left, right in combinations(on_rows, 2):
            jaccard.append(_set_jaccard(_routes(*left), _routes(*right)))
        pressure_cv.append(_cv([_metric(row, PRIMARY_PRESSURE) for _, row in on_rows]) or 0.0)
        working_cv.append(_cv([_metric(row, "mean_layer_working_set_fraction") for _, row in on_rows]) or 0.0)
        active_cv.append(_cv([_metric(row, "mean_active_expert_fraction") for _, row in on_rows]) or 0.0)
        timed = [bundle for bundle in on if bundle["repeat"] in timing]
        if len(timed) >= THRESHOLDS_V1["minimum_process_repeats"]:
            tpot_cv.append(_cv([_metric(timing[bundle["repeat"]]["map"][key], "request_tpot_p95_ms") for bundle in timed]) or 0.0)
    values = {"route_set_jaccard_median": _median(jaccard), "route_set_jaccard_p10": _q(jaccard, 0.10),
              "pressure_cv_p90": _q(pressure_cv, 0.90), "route_OFF_tpot_cv_p90": _q(tpot_cv, 0.90),
              "working_set_fraction_cv_p90": _q(working_cv, 0.90),
              "active_expert_fraction_cv_p90": _q(active_cv, 0.90)}
    t = THRESHOLDS_V1
    values["qualified_route_OFF_process_repeats"] = len(timing)
    values["stable_for_action_screen"] = bool(len(timing) >= t["minimum_process_repeats"]
        and (values["route_set_jaccard_median"] or 0) >= t["minimum_repeat_route_set_jaccard_median"]
        and (values["route_set_jaccard_p10"] or 0) >= t["minimum_repeat_route_set_jaccard_p10"]
        and values["pressure_cv_p90"] is not None and values["pressure_cv_p90"] <= t["maximum_repeat_pressure_cv_p90"]
        and values["route_OFF_tpot_cv_p90"] is not None and values["route_OFF_tpot_cv_p90"] <= t["maximum_repeat_tpot_cv_p90"])
    values["stable_for_structural_measurement"] = bool(len(on) >= t["minimum_process_repeats"]
        and values["pressure_cv_p90"] is not None
        and values["pressure_cv_p90"] <= t["maximum_repeat_pressure_cv_p90"]
        and values["working_set_fraction_cv_p90"] is not None
        and values["working_set_fraction_cv_p90"] <= t["maximum_repeat_working_set_cv_p90"]
        and values["active_expert_fraction_cv_p90"] is not None
        and values["active_expert_fraction_cv_p90"] <= t["maximum_repeat_active_fraction_cv_p90"])
    return values


def _temporal(on: Sequence[dict[str, Any]]) -> dict[str, Any]:
    names = REQUIRED_ROUTE_METRICS[-4:]
    distributions = {name: _distribution([_metric(row, name) for bundle in on for row in bundle["rows"]]) for name in names}
    load = distributions["mean_temporal_load_vector_cosine"]
    qualified = bool((load["median"] or 0) >= THRESHOLDS_V1["minimum_temporal_load_cosine_median"]
                     and (load["p10"] or 0) >= THRESHOLDS_V1["minimum_temporal_load_cosine_p10"])
    return {"primary_metric": "mean_temporal_load_vector_cosine", "qualified": qualified, "metrics": distributions}


def _cross_batch(on: Sequence[dict[str, Any]]) -> dict[str, Any]:
    series = []
    for bundle in on:
        prompts = sorted({key[0] for key in bundle["map"]})
        for prompt in prompts:
            points = []
            for batch in sorted({key[1] for key in bundle["map"] if key[0] == prompt}):
                rows = [row for key, row in bundle["map"].items() if key[:2] == (prompt, batch)]
                points.append({"batch_size": batch,
                    "mean_active_expert_fraction": float(np.mean([_metric(row, "mean_active_expert_fraction") for row in rows])),
                    "mean_layer_working_set_fraction": float(np.mean([_metric(row, "mean_layer_working_set_fraction") for row in rows]))})
            active = [point["mean_active_expert_fraction"] for point in points]
            working = [point["mean_layer_working_set_fraction"] for point in points]
            tolerance = THRESHOLDS_V1["cross_batch_monotonic_tolerance"]
            monotonic = len(points) >= 2 and all(current + tolerance >= previous for previous, current in zip(active, active[1:]))
            active_delta = max(active) - min(active) if len(active) >= 2 else 0.0
            working_delta = max(working) - min(working) if len(working) >= 2 else 0.0
            material = monotonic and (active_delta >= THRESHOLDS_V1["minimum_cross_batch_active_fraction_delta"]
                                      or working_delta >= THRESHOLDS_V1["minimum_cross_batch_working_set_fraction_delta"])
            series.append({"process_repeat": bundle["repeat"], "prompt_length": prompt, "points": points,
                           "active_expert_fraction_delta": active_delta, "working_set_fraction_delta": working_delta,
                           "monotonic": monotonic, "material": material})
    fraction = sum(item["material"] for item in series) / len(series) if series else 0.0
    return {"series_count": len(series), "material_series_fraction": fraction,
            "material": bool(series and fraction >= THRESHOLDS_V1["minimum_cross_batch_material_fraction"]), "series": series}


def _base(on: Sequence[Path], off: Sequence[Path]) -> dict[str, Any]:
    return {"schema": SCHEMA, "claim_ceiling": CLAIM_CEILING,
            "pivot_verdict": "STOP_ROUTE_CONTROL", "threshold_version": "v1", "thresholds": THRESHOLDS_V1,
            "route_ON_bundles": list(map(str, on)), "route_OFF_bundles": list(map(str, off)),
            "anti_claims": ["no action-conditioned counterfactual was measured",
                "route outcomes are not pre-action signals",
                "route-ON TPOT is instrumented diagnostic timing and never drives the action screen",
                "when token transparency fails, route-ON structure is telemetry-conditioned and cannot be transferred to route-OFF execution",
                "paired route-OFF timing remains observational, not a serving-policy gain",
                "prompt lengths and batch sizes are nested/prefix-related operating cells, not independent workload replications",
                "fixed batches do not establish online queue, P99, goodput, or SLO gains",
                "single-GPU evidence is not Expert Parallel evidence",
                "TEST_MARGINAL_PRESSURE_ACTION selects an experiment; it is not a method GO"]}


def analyze_bundles(route_on: Sequence[Path], route_off: Sequence[Path] = ()) -> dict[str, Any]:
    on_paths = [Path(path).resolve() for path in route_on]
    off_paths = [Path(path).resolve() for path in route_off]
    report = _base(on_paths, off_paths)
    try:
        on, errors = _load_sets(on_paths, True)
        off, off_errors = _load_sets(off_paths, False)
        errors.extend(off_errors)
        off_map, parity, pair_errors = _pair_off(on, off)
        errors.extend(pair_errors)
    except (
        EOFError,
        KeyError,
        OSError,
        TypeError,
        ValueError,
        ZeroDivisionError,
        zipfile.BadZipFile,
    ) as exc:
        report.update({"status": "INVALID_INPUT", "failure_category": "BUNDLE_PARSE_OR_VALIDATION",
                       "validation_errors": [str(exc)]})
        return report
    report["bundle_integrity"] = [{"path": str(bundle["path"]), "process_repeat": bundle["repeat"], **bundle["integrity"]} for bundle in on + off]
    report["telemetry_pairs"] = parity
    if not on_paths or errors:
        report.update({"status": "INVALID_INPUT", "failure_category": "BUNDLE_INTEGRITY_OR_COMPATIBILITY",
                       "validation_errors": errors or ["at least one route-ON bundle is required"]})
        return report
    token_drift_repeats = [pair["process_repeat"] for pair in parity if pair["status"] == "TELEMETRY_TOKEN_DRIFT"]
    timing_deviation_failure_repeats = [
        pair["process_repeat"]
        for pair in parity
        if pair["status"] == "ROUTE_EXPORT_TOO_EXPENSIVE_FOR_TIMING_CLAIM"
    ]
    timing_pair_qualified = bool(
        off
        and len(parity) == len(on)
        and len(off_map) == len(on)
        and all(
            pair["status"] == "TELEMETRY_OVERHEAD_QUALIFIED" for pair in parity
        )
    )
    all_bundles = on + off
    source_unverified_bundles = [
        str(bundle["path"])
        for bundle in all_bundles
        if not bundle["integrity"].get(
            "producer_source_semantics_approved", False
        )
    ]
    source_unverified = sorted({
        bundle["repeat"]
        for bundle in all_bundles
        if not bundle["integrity"].get(
            "producer_source_semantics_approved", False
        )
    })
    isolation_unverified_bundles = [
        str(bundle["path"])
        for bundle in all_bundles
        if not (
            bundle["config"].get("require_exclusive_gpu") is True
            and bundle["environment"].get("compute_processes_before_engine_init")
            == []
            and bundle["integrity"].get("exclusive_gpu_verified", False)
        )
    ]
    isolation_unverified = sorted({
        bundle["repeat"]
        for bundle in all_bundles
        if not (
            bundle["config"].get("require_exclusive_gpu") is True
            and bundle["environment"].get("compute_processes_before_engine_init")
            == []
            and bundle["integrity"].get("exclusive_gpu_verified", False)
        )
    })
    producer_source_qualified = not source_unverified
    gpu_isolation_qualified = not isolation_unverified
    # Every required repeat must qualify as one indivisible experiment. A
    # clean subset would be post-hoc selection. Source provenance and actual
    # empty-process evidence are action eligibility conditions as well.
    if not (
        timing_pair_qualified
        and producer_source_qualified
        and gpu_isolation_qualified
    ):
        off_map = {}
    if off and len(off_map) == len(on):
        timing_source = "PAIRED_ROUTE_OFF_ALL_REPEATS"
    else:
        timing_source = "ROUTE_ON_DIAGNOSTIC_ONLY"
    if token_drift_repeats:
        transparency_status = "FAILED_TOKEN_DRIFT"
    elif timing_deviation_failure_repeats:
        transparency_status = "FAILED_TIMING_DEVIATION"
    elif timing_pair_qualified:
        transparency_status = "QUALIFIED"
    else:
        transparency_status = "UNTESTED"
    report["telemetry_transparency"] = {
        "status": transparency_status,
        "token_drift_process_repeats": token_drift_repeats,
        "timing_deviation_failure_process_repeats": (
            timing_deviation_failure_repeats
        ),
        "timing_qualified_route_OFF_process_repeats": sorted(
            pair["process_repeat"]
            for pair in parity
            if pair["status"] == "TELEMETRY_OVERHEAD_QUALIFIED"
        ),
        "qualified_route_OFF_process_repeats": sorted(off_map),
        "all_required_repeats_must_match": True,
    }
    report["producer_source_provenance"] = {
        "status": (
            "APPROVED_ALL_BUNDLES"
            if producer_source_qualified
            else "PRODUCER_SOURCE_SEMANTICS_UNAPPROVED"
        ),
        "qualified_for_action": producer_source_qualified,
        "unverified_process_repeats": source_unverified,
        "unverified_bundles": source_unverified_bundles,
        "bundle_statuses": [
            {
                "path": str(bundle["path"]),
                "process_repeat": bundle["repeat"],
                "status": bundle["integrity"].get("producer_source_status"),
            }
            for bundle in all_bundles
        ],
    }
    report["gpu_isolation"] = {
        "status": (
            "VERIFIED_ALL_BUNDLES"
            if gpu_isolation_qualified
            else "GPU_ISOLATION_UNVERIFIED"
        ),
        "qualified_for_action": gpu_isolation_qualified,
        "unverified_process_repeats": isolation_unverified,
        "unverified_bundles": isolation_unverified_bundles,
        "requires_config_true_and_empty_environment_process_list": True,
    }
    try:
        cells = _fixed_cells(on, off_map)
        temporal, stability, cross_batch = _temporal(on), _stability(on, off_map), _cross_batch(on)
    except (
        EOFError,
        KeyError,
        OSError,
        TypeError,
        ValueError,
        zipfile.BadZipFile,
    ) as exc:
        report.update({"status": "INVALID_INPUT", "failure_category": "ANALYSIS_VALIDATION", "validation_errors": [str(exc)]})
        return report
    association_cells = sum(cell["supports_composition_association"] for cell in cells)
    structural_cells = sum(
        bool(cell["enough_groups"]
             and (cell["median_pressure_p90_p10_relative"] or 0)
             >= THRESHOLDS_V1["minimum_pressure_p90_p10_relative"]
             and (cell["median_pressure_p90_p10_absolute"] or 0)
             >= THRESHOLDS_V1["minimum_pressure_p90_p10_absolute"])
        for cell in cells
    )
    required = max(int(THRESHOLDS_V1["minimum_supportive_cells"]),
                   math.ceil(len(cells) * THRESHOLDS_V1["minimum_supportive_cell_fraction"]))
    coverage = bool(len(on) >= THRESHOLDS_V1["minimum_process_repeats"]
                    and len(cells) >= THRESHOLDS_V1["minimum_tested_cells"]
                    and all(cell["enough_groups"] for cell in cells))
    action = bool(coverage and timing_source != "ROUTE_ON_DIAGNOSTIC_ONLY" and association_cells >= required
                  and temporal["qualified"] and stability["stable_for_action_screen"])
    measurement = bool(coverage and stability["stable_for_structural_measurement"]
                       and (cross_batch["material"] or structural_cells > 0))
    if action:
        pivot, failure = "TEST_MARGINAL_PRESSURE_ACTION", None
        next_step = "Run one action-conditioned same-state fork: previous-step marginal route pressure versus the strongest token/KV baseline, with policy-specific batches and route-OFF TPOT/tail outcomes."
    elif measurement:
        pivot = "WORKING_SET_MEASUREMENT_ONLY"
        if token_drift_repeats:
            failure = "TELEMETRY_TRANSPARENCY_FAILED"
        elif timing_deviation_failure_repeats:
            failure = "TELEMETRY_TIMING_DEVIATION_FAILED"
        elif not producer_source_qualified:
            failure = "PRODUCER_SOURCE_SEMANTICS_UNAPPROVED"
        elif not gpu_isolation_qualified:
            failure = "GPU_ISOLATION_UNVERIFIED"
        else:
            failure = "NO_QUALIFIED_LOW_OVERHEAD_ACTION_SIGNAL"
        next_step = "Keep the result structural; implement a GPU-side aggregate pressure sketch and qualify its overhead before any controller or latency claim."
    else:
        pivot = "STOP_ROUTE_CONTROL"
        failure = "INSUFFICIENT_EVIDENCE" if not coverage else "ACTION_SCREEN_NOT_QUALIFIED"
        next_step = "Do not implement a controller in this regime; fill only missing repeats/groups, or reopen after a runtime/regime change alters the exposed denominator."
    report.update({"status": "COMPLETE" if coverage else "INSUFFICIENT_EVIDENCE", "failure_category": failure,
                   "pivot_verdict": pivot, "timing_source_for_decision": timing_source,
                   "structural_evidence_scope": (
                       "TELEMETRY_CONDITIONED_ROUTE_STRUCTURE_ONLY"
                       if token_drift_repeats
                       else (
                           "PRODUCER_SOURCE_UNAPPROVED_ROUTE_ON_STRUCTURAL_MEASUREMENT"
                           if not producer_source_qualified
                           else "ROUTE_ON_STRUCTURAL_MEASUREMENT"
                       )
                   ),
                   "coverage": {"process_repeats": len(on), "tested_cells": len(cells),
                                "qualified": coverage,
                                "evidence_unit": "CORRELATED_OPERATING_CELL_COVERAGE_ONLY",
                                "independent_workload_replications": 0},
                   "fixed_cell_composition": {"primary_pressure_metric": PRIMARY_PRESSURE,
                                              "structural_pressure_cells": structural_cells,
                                              "association_cells": association_cells,
                                              "required_association_cells": required, "cells": cells},
                   "temporal_predictability": temporal, "process_repeat_stability": stability,
                   "cross_batch_working_set": cross_batch,
                   "decision_gates": {"coverage": coverage,
                                      "paired_route_OFF_timing": timing_source != "ROUTE_ON_DIAGNOSTIC_ONLY",
                                      "producer_source_semantics_approved": producer_source_qualified,
                                      "exclusive_gpu_verified": gpu_isolation_qualified,
                                      "composition_association": association_cells >= required,
                                      "temporal_predictability": temporal["qualified"],
                                      "action_stability": stability["stable_for_action_screen"],
                                      "structural_measurement_stability": stability["stable_for_structural_measurement"]},
                   "one_next_experiment": next_step})
    assert report["pivot_verdict"] in PIVOTS
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route-on", type=Path, nargs="+", required=True)
    parser.add_argument("--route-off", type=Path, nargs="*", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; analysis artifacts are write-once")
    report = analyze_bundles(args.route_on, args.route_off)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    print(json.dumps({key: report.get(key) for key in ("status", "pivot_verdict", "failure_category")}))
    if report["status"] == "INVALID_INPUT":
        raise SystemExit(2)
    if report["pivot_verdict"] == "TEST_MARGINAL_PRESSURE_ACTION":
        raise SystemExit(0)
    raise SystemExit(1)


if __name__ == "__main__":
    main()
