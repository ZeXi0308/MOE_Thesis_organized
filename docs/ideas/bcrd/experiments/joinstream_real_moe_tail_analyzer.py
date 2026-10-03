from __future__ import annotations

"""Mechanical analyzer for the frozen REAL_MOE_TAIL_APPLICABILITY_GATE."""

import argparse
import csv
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Iterable, Mapping, Sequence


SCHEMA = "joinstream-real-moe-tail-analysis-v1"
RAW_SCHEMA = "joinstream-real-moe-tail-raw-v1"
CALIBRATION_SCHEMA = "joinstream-real-moe-tail-calibration-v1"
RUN_LOCK_SCHEMA = "joinstream-real-moe-tail-run-lock-v1"
ENVIRONMENT_SCHEMA = "joinstream-real-moe-tail-environment-v1"
ROUTES = ("BALANCED", "SKEWED")
SCALES = ("MEDIUM_RESIDENCY", "HIGH_RESIDENCY")
VARIANTS = (
    "A_ALL_DONE_SHAM",
    "B_EAGER_JOINSTREAM",
    "C_PROGRESS_GATED_JOINSTREAM",
)
MODES = ("correctness", "utility")
GATE_CANDIDATES = (0.0, 0.125, 0.25, 0.5)
PERMUTATIONS = (
    VARIANTS,
    (VARIANTS[0], VARIANTS[2], VARIANTS[1]),
    (VARIANTS[1], VARIANTS[0], VARIANTS[2]),
    (VARIANTS[1], VARIANTS[2], VARIANTS[0]),
    (VARIANTS[2], VARIANTS[0], VARIANTS[1]),
    (VARIANTS[2], VARIANTS[1], VARIANTS[0]),
)
REGRESSION_LIMIT = 0.05
OBSERVABLE_US = 1.0
MAD_SCALE = 3.0 * 1.4826
EPS = 1e-12
EVALUATION_TYPE = (
    "real_hardware_synthetic_grouped_expert_microbenchmark"
    "+self_supervised_variant_equivalence_proxy"
)
CORRECTNESS_SEMANTICS = (
    "A/B/C bitwise variant equivalence against ALL_DONE_SHAM; "
    "not independent numerical ground truth"
)

REQUIRED_COLUMNS = {
    "schema_version",
    "mode",
    "cell_id",
    "route_distribution",
    "problem_scale",
    "repeat_kind",
    "repeat_index",
    "permutation_slot",
    "variant",
    "route_seed",
    "gate_threshold_remaining_ratio",
    "total_tokens",
    "routed_tokens",
    "top_k",
    "expert_count",
    "theoretical_flops",
    "route_table_hash",
    "input_hash",
    "producer_work_hash",
    "consumer_work_hash",
    "progress_instrumentation_hash",
    "expert_token_counts_hash",
    "producer_launches",
    "consumer_launches",
    "producer_grid_size",
    "producer_block_size",
    "consumer_grid_size",
    "consumer_block_size",
    "expert_tiles_total",
    "producer_work_expected",
    "producer_work_done",
    "synthetic_delay_enabled",
    "artificial_sm_reservation_enabled",
    "critical_expert_a",
    "critical_expert_b",
    "critical_contributions_done",
    "critical_contributions_expected",
    "producer_start_ns",
    "join_close_ns",
    "gate_satisfied_ns",
    "consumer_entry_ns",
    "consumer_observe_ns",
    "consumer_start_ns",
    "consumer_end_ns",
    "producer_end_ns",
    "total_end_ns",
    "remaining_producer_work_at_consumer_start",
    "output_hash",
    "reference_output_hash",
    "correctness_pass",
    "stale_read",
    "timestamp_contract_pass",
    "cuda_error",
}


class ContractError(ValueError):
    pass


@dataclass(frozen=True)
class Distribution:
    median: float
    mad: float
    p10: float
    p90: float
    noise_guard: float
    count: int


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite(value: object, field: str) -> float:
    _require(type(value) in {int, float}, f"{field} must be numeric")
    number = float(value)
    _require(math.isfinite(number), f"{field} must be finite")
    return number


def _int(row: Mapping[str, str], field: str) -> int:
    value = row.get(field, "")
    try:
        return int(value, 10)
    except (TypeError, ValueError) as error:
        raise ContractError(f"{field} must be a decimal integer") from error


def _float(row: Mapping[str, str], field: str) -> float:
    try:
        value = float(row[field])
    except (KeyError, ValueError) as error:
        raise ContractError(f"{field} must be numeric") from error
    _require(math.isfinite(value), f"{field} must be finite")
    return value


def _bool(row: Mapping[str, str], field: str) -> bool:
    value = row.get(field, "").strip().lower()
    _require(value in {"0", "1", "false", "true"}, f"{field} must be boolean")
    return value in {"1", "true"}


def _percentile(values: Sequence[float], quantile: float) -> float:
    position = (len(values) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    weight = position - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def summarize(values: Iterable[float], timer_resolution_us: float) -> Distribution:
    ordered = sorted(float(value) for value in values)
    _require(bool(ordered) and all(math.isfinite(value) for value in ordered), "metric is empty/nonfinite")
    median = statistics.median(ordered)
    mad = statistics.median(abs(value - median) for value in ordered)
    return Distribution(
        median=median,
        mad=mad,
        p10=_percentile(ordered, 0.10),
        p90=_percentile(ordered, 0.90),
        noise_guard=max(timer_resolution_us, MAD_SCALE * mad),
        count=len(ordered),
    )


def _load_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), f"{path.name} must contain an object")
    return payload


def _validate_environment(environment: Mapping[str, object]) -> float:
    _require(environment.get("schema") == ENVIRONMENT_SCHEMA, "unexpected environment schema")
    _require(environment.get("status") == "CAPTURED", "environment capture is incomplete")
    _require(environment.get("gpu_available") is True, "CUDA GPU is unavailable")
    hardware = environment.get("hardware")
    software = environment.get("software")
    timer = environment.get("timer")
    _require(isinstance(hardware, dict) and isinstance(software, dict) and isinstance(timer, dict), "environment sections missing")
    _require(hardware.get("name") == "NVIDIA GeForce RTX 5090", "unexpected GPU model")
    _require(hardware.get("sm_count") == 170, "unexpected SM count")
    _require(str(hardware.get("compute_capability")) == "12.0", "unexpected compute capability")
    _require(str(software.get("driver_version")) == "595.71.05", "unexpected driver version")
    _require(str(software.get("cuda_toolkit_version")) == "12.8", "unexpected CUDA toolkit")
    resolution = _finite(timer.get("resolution_ns"), "timer.resolution_ns") / 1000.0
    _require(resolution > 0, "timer resolution must be positive")
    return resolution


def _cell_key(raw: Mapping[str, object]) -> tuple[str, str]:
    route = str(raw.get("route_distribution"))
    scale = str(raw.get("problem_scale"))
    _require(route in ROUTES and scale in SCALES, "unexpected 4-cell factors")
    return route, scale


def _validate_calibration(
    calibration: Mapping[str, object], run_lock: Mapping[str, object]
) -> dict[str, dict[str, object]]:
    _require(calibration.get("schema") == CALIBRATION_SCHEMA, "unexpected calibration schema")
    _require(calibration.get("status") == "CALIBRATION_COMPLETE_LOCKED", "calibration not locked")
    candidates = calibration.get("gate_candidates_remaining_ratio")
    _require(isinstance(candidates, list), "calibration gate candidates missing")
    _require(tuple(float(value) for value in candidates) == GATE_CANDIDATES, "gate candidates changed")
    repetitions = calibration.get("repetitions")
    _require(isinstance(repetitions, dict), "calibration repetitions missing")
    _require(
        repetitions.get("warmups_per_cell_candidate") == 10
        and repetitions.get("measured_per_cell_candidate") == 50,
        "calibration repeat counts changed",
    )
    calibration_seed = calibration.get("calibration_route_seed")
    formal_seed = run_lock.get("formal_route_seed")
    _require(type(calibration_seed) is int and type(formal_seed) is int, "route seeds must be integers")
    _require(calibration_seed != formal_seed, "calibration and formal route seeds overlap")
    rows = calibration.get("cells")
    _require(isinstance(rows, list) and len(rows) == 4, "calibration must have exactly 4 cells")
    expected = {(route, scale) for route in ROUTES for scale in SCALES}
    output: dict[str, dict[str, object]] = {}
    observed: set[tuple[str, str]] = set()
    for raw in rows:
        _require(isinstance(raw, dict), "calibration cell must be an object")
        factor = _cell_key(raw)
        _require(factor not in observed, "duplicate calibration cell")
        observed.add(factor)
        cell_id = str(raw.get("cell_id"))
        _require(bool(cell_id) and cell_id not in output, "invalid calibration cell_id")
        candidate_rows = raw.get("candidates")
        _require(isinstance(candidate_rows, list) and len(candidate_rows) == 4, "cell must contain four gate candidates")
        by_ratio: dict[float, float] = {}
        for candidate in candidate_rows:
            _require(isinstance(candidate, dict), "gate candidate must be an object")
            ratio = _finite(candidate.get("remaining_ratio"), "candidate remaining_ratio")
            regression = _finite(candidate.get("producer_regression_ratio"), "candidate regression")
            _require(ratio in GATE_CANDIDATES and ratio not in by_ratio, "invalid/duplicate gate candidate")
            by_ratio[ratio] = regression
        _require(set(by_ratio) == set(GATE_CANDIDATES), "gate candidate set incomplete")
        safe = [ratio for ratio, regression in by_ratio.items() if regression < REGRESSION_LIMIT]
        expected_status = "SELECTED" if safe else "NO_SAFE_GATE"
        expected_ratio = max(safe) if safe else None
        _require(raw.get("selection_status") == expected_status, "calibration selection status is not mechanical")
        selected = raw.get("selected_gate_remaining_ratio")
        if expected_ratio is None:
            _require(selected is None, "NO_SAFE_GATE must have null selected ratio")
        else:
            _require(_finite(selected, "selected gate") == expected_ratio, "selected gate is not earliest safe candidate")
        output[cell_id] = dict(raw)
    _require(observed == expected, "calibration 4-cell product incomplete")
    return output


def _validate_run_lock(
    run_lock: Mapping[str, object], calibration_path: Path, calibration_cells: Mapping[str, Mapping[str, object]]
) -> tuple[dict[str, dict[str, object]], dict[str, int]]:
    _require(run_lock.get("schema") == RUN_LOCK_SCHEMA, "unexpected run-lock schema")
    _require(run_lock.get("status") == "LOCKED_BEFORE_FORMAL_RUN", "formal run lock is not frozen")
    _require(run_lock.get("calibration_sha256") == _sha256(calibration_path), "run lock calibration digest mismatch")
    _require(run_lock.get("calibration_route_seed") != run_lock.get("formal_route_seed"), "route seed leakage")
    contracts = run_lock.get("contracts")
    _require(isinstance(contracts, dict), "run-lock contracts missing")
    for field in ("synthetic_delay", "artificial_sm_reservation"):
        _require(contracts.get(field) is False, f"formal {field} is prohibited")
    for field in ("single_producer_kernel", "matrix_multiply_like_work", "representative_grouped_expert"):
        _require(contracts.get(field) is True, f"formal contract {field} is not established")
    repetitions = run_lock.get("repetitions")
    _require(isinstance(repetitions, dict), "run-lock repetitions missing")
    counts: dict[str, int] = {}
    for key, field in (
        ("warmup", "warmups_per_cell_mode"),
        ("correctness", "correctness"),
        ("utility", "utility"),
    ):
        value = _finite(repetitions.get(field), f"{key} repeats")
        _require(value.is_integer(), f"{key} repeats must be integral")
        counts[key] = int(value)
    _require(all(value >= 3 for value in counts.values()), "paired MAD needs at least 3 rounds")
    rows = run_lock.get("cells")
    _require(isinstance(rows, list) and len(rows) == 4, "run lock must contain exactly 4 cells")
    expected = {(route, scale) for route in ROUTES for scale in SCALES}
    observed: set[tuple[str, str]] = set()
    output: dict[str, dict[str, object]] = {}
    for raw in rows:
        _require(isinstance(raw, dict), "run-lock cell must be an object")
        factor = _cell_key(raw)
        _require(factor not in observed, "duplicate run-lock cell")
        observed.add(factor)
        cell_id = str(raw.get("cell_id"))
        _require(cell_id in calibration_cells and cell_id not in output, "run-lock/calibration cell mismatch")
        calibration = calibration_cells[cell_id]
        _require(
            raw.get("gate_selection_status") == calibration.get("selection_status"),
            "run-lock gate status differs from calibration",
        )
        locked_ratio = raw.get("gate_remaining_ratio")
        selected = calibration.get("selected_gate_remaining_ratio")
        if selected is None:
            _require(locked_ratio is None, "NO_SAFE_GATE lock must keep null ratio")
        else:
            _require(_finite(locked_ratio, "locked gate ratio") == _finite(selected, "selected gate ratio"), "run-lock gate changed")
        for field in ("total_tokens", "routed_tokens", "top_k", "expert_count", "theoretical_flops"):
            _require(type(raw.get(field)) is int and int(raw[field]) > 0, f"invalid locked {field}")
        _require(int(raw["top_k"]) >= 2, "top_k must be at least 2")
        expert_counts = raw.get("expert_routed_token_counts")
        _require(
            isinstance(expert_counts, list)
            and len(expert_counts) == int(raw["expert_count"])
            and all(type(value) is int and value >= 0 for value in expert_counts)
            and sum(expert_counts) == int(raw["routed_tokens"]),
            "expert routed-token counts do not close",
        )
        if raw["route_distribution"] == "BALANCED":
            _require(max(expert_counts) - min(expert_counts) <= 1, "BALANCED route is not balanced")
        else:
            _require(max(expert_counts) > min(expert_counts), "SKEWED route has no long tail")
        output[cell_id] = dict(raw)
    _require(observed == expected, "run-lock 4-cell product incomplete")
    for scale in SCALES:
        pair = [row for row in output.values() if row["problem_scale"] == scale]
        for field in ("total_tokens", "routed_tokens", "top_k", "expert_count", "theoretical_flops"):
            _require(len({row[field] for row in pair}) == 1, f"BALANCED/SKEWED differ in locked {field}")
        balanced = next(row for row in pair if row["route_distribution"] == "BALANCED")
        skewed = next(row for row in pair if row["route_distribution"] == "SKEWED")
        _require(
            max(skewed["expert_routed_token_counts"])
            > max(balanced["expert_routed_token_counts"]),
            "SKEWED route is not more imbalanced than BALANCED",
        )
    return output, counts


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        _require(reader.fieldnames is not None, "formal CSV has no header")
        missing = REQUIRED_COLUMNS - set(reader.fieldnames)
        _require(not missing, f"formal CSV missing columns: {sorted(missing)}")
        rows = [dict(row) for row in reader]
    _require(bool(rows), "formal CSV is empty")
    return rows


def _validate_row(row: Mapping[str, str], formal_seed: int) -> tuple[str, str]:
    _require(row["schema_version"] == RAW_SCHEMA, "unexpected formal raw schema")
    _require(row["mode"] in MODES and row["repeat_kind"] in {"warmup", "measured"}, "invalid mode/repeat kind")
    _require(row["variant"] in VARIANTS, "invalid variant")
    route = row["route_distribution"]
    scale = row["problem_scale"]
    _require(route in ROUTES and scale in SCALES, "invalid formal cell factors")
    _require(_int(row, "route_seed") == formal_seed, "formal CSV does not use the locked formal seed")
    _require(not _bool(row, "synthetic_delay_enabled"), "formal workload contains synthetic delay")
    _require(not _bool(row, "artificial_sm_reservation_enabled"), "formal workload reserves SMs artificially")
    _require(_int(row, "producer_launches") == 1 and _int(row, "consumer_launches") == 1, "launch count contract failed")
    for field in (
        "total_tokens", "routed_tokens", "top_k", "expert_count", "theoretical_flops",
        "producer_grid_size", "producer_block_size", "consumer_grid_size", "consumer_block_size",
        "expert_tiles_total", "producer_work_expected",
    ):
        _require(_int(row, field) > 0, f"{field} must be positive")
    _require(_int(row, "top_k") >= 2, "formal top_k must be at least 2")
    _require(
        _int(row, "producer_work_done") == _int(row, "producer_work_expected"),
        "producer useful work is incomplete",
    )
    _require(
        _int(row, "critical_expert_a") != _int(row, "critical_expert_b")
        and _int(row, "critical_contributions_expected") == 2
        and _int(row, "critical_contributions_done") == 2,
        "critical token does not close a two-expert K-way join",
    )
    _require(_bool(row, "correctness_pass") and not _bool(row, "stale_read"), "correctness/stale-read contract failed")
    _require(_bool(row, "timestamp_contract_pass") and not row.get("cuda_error", "").strip(), "timestamp/CUDA contract failed")
    _require(bool(row["output_hash"]) and row["output_hash"] == row["reference_output_hash"], "consumer output mismatch")
    for field in (
        "route_table_hash",
        "input_hash",
        "producer_work_hash",
        "consumer_work_hash",
        "progress_instrumentation_hash",
        "expert_token_counts_hash",
    ):
        _require(bool(row[field]), f"{field} must be non-empty")
    start = _int(row, "producer_start_ns")
    join = _int(row, "join_close_ns")
    gate = _int(row, "gate_satisfied_ns")
    entry = _int(row, "consumer_entry_ns")
    observe = _int(row, "consumer_observe_ns")
    consumer_start = _int(row, "consumer_start_ns")
    consumer_end = _int(row, "consumer_end_ns")
    producer_end = _int(row, "producer_end_ns")
    total_end = _int(row, "total_end_ns")
    _require(entry <= start <= join <= producer_end <= total_end, "producer/join timestamp order invalid")
    _require(start <= gate <= producer_end, "progress gate timestamp outside producer lifetime")
    _require(observe <= consumer_start <= consumer_end <= total_end, "consumer timestamp order invalid")
    if row["variant"] == "A_ALL_DONE_SHAM":
        _require(observe >= producer_end, "ALL_DONE_SHAM observes before all producer work is done")
    elif row["variant"] == "B_EAGER_JOINSTREAM":
        _require(observe >= join, "EAGER consumer observes before join closure")
    else:
        _require(
            observe >= max(join, gate) and consumer_start >= gate,
            "GATED consumer observes/starts before join and progress gate",
        )
    remaining = _int(row, "remaining_producer_work_at_consumer_start")
    _require(0 <= remaining <= _int(row, "expert_tiles_total"), "remaining producer work is out of range")
    return route, scale


def _group_rows(
    rows: Sequence[dict[str, str]],
    run_cells: Mapping[str, Mapping[str, object]],
    counts: Mapping[str, int],
    formal_seed: int,
) -> dict[tuple[str, str], list[dict[str, dict[str, str]]]]:
    factors: dict[str, tuple[str, str]] = {}
    groups: dict[tuple[str, str, str, str, int], dict[str, dict[str, str]]] = {}
    for row in rows:
        factor = _validate_row(row, formal_seed)
        cell_id = row["cell_id"]
        _require(cell_id in run_cells, "formal cell_id absent from run lock")
        locked = run_cells[cell_id]
        _require(factor == _cell_key(locked), "formal factors differ from run lock")
        factors.setdefault(cell_id, factor)
        _require(factors[cell_id] == factor, "cell_id maps to multiple factors")
        for field in ("total_tokens", "routed_tokens", "top_k", "expert_count", "theoretical_flops"):
            _require(_int(row, field) == int(locked[field]), f"formal {field} differs from run lock")
        expected_ratio = locked["gate_remaining_ratio"]
        actual_ratio = _float(row, "gate_threshold_remaining_ratio")
        if expected_ratio is None:
            _require(actual_ratio == 0.0, "NO_SAFE_GATE formal fallback must be all-done ratio 0")
        else:
            _require(actual_ratio == float(expected_ratio), "formal gate ratio differs from lock")
        remaining = _int(row, "remaining_producer_work_at_consumer_start")
        total_work = _int(row, "expert_tiles_total")
        if row["variant"] == "A_ALL_DONE_SHAM":
            _require(remaining == 0, "ALL_DONE_SHAM starts with producer work remaining")
        elif row["variant"] == "C_PROGRESS_GATED_JOINSTREAM":
            _require(
                remaining / total_work <= actual_ratio + EPS,
                "GATED consumer starts before locked remaining-work threshold",
            )
        repeat = _int(row, "repeat_index")
        key = (row["mode"], row["repeat_kind"], cell_id, f"{factor[0]}::{factor[1]}", repeat)
        group = groups.setdefault(key, {})
        _require(row["variant"] not in group, "duplicate variant in paired repeat")
        group[row["variant"]] = row
    _require(set(factors.values()) == {(route, scale) for route in ROUTES for scale in SCALES}, "formal run is not exact 4-cell product")
    utility: dict[tuple[str, str], list[dict[str, dict[str, str]]]] = {
        factor: [] for factor in sorted(set(factors.values()))
    }
    parity = (
        "route_seed", "route_table_hash", "input_hash", "total_tokens", "routed_tokens", "top_k",
        "expert_count", "theoretical_flops", "producer_work_hash", "consumer_work_hash",
        "progress_instrumentation_hash", "producer_launches", "consumer_launches", "producer_grid_size",
        "expert_token_counts_hash",
        "producer_block_size", "consumer_grid_size", "consumer_block_size", "expert_tiles_total",
        "producer_work_expected", "producer_work_done", "synthetic_delay_enabled",
        "artificial_sm_reservation_enabled", "critical_expert_a", "critical_expert_b",
        "critical_contributions_expected", "critical_contributions_done", "reference_output_hash",
    )
    for mode in MODES:
        for repeat_kind in ("warmup", "measured"):
            expected_count = counts["warmup"] if repeat_kind == "warmup" else counts[mode]
            for cell_id, factor in factors.items():
                matching = [key for key in groups if key[:3] == (mode, repeat_kind, cell_id)]
                repeat_ids = sorted(key[4] for key in matching)
                _require(repeat_ids == list(range(expected_count)), f"{mode}/{repeat_kind}/{cell_id} repeat indices incomplete")
                for repeat in repeat_ids:
                    key = (mode, repeat_kind, cell_id, f"{factor[0]}::{factor[1]}", repeat)
                    group = groups[key]
                    _require(set(group) == set(VARIANTS), "paired repeat must have exact A/B/C")
                    expected_order = PERMUTATIONS[repeat % len(PERMUTATIONS)]
                    for slot, variant in enumerate(expected_order):
                        _require(_int(group[variant], "permutation_slot") == slot, "six-permutation rotation changed")
                    for field in parity:
                        _require(len({row[field] for row in group.values()}) == 1, f"paired A/B/C mismatch in {field}")
                    if mode == "utility" and repeat_kind == "measured":
                        utility[factor].append(group)
    return utility


def _elapsed_us(row: Mapping[str, str], field: str) -> float:
    return (_int(row, field) - _int(row, "producer_start_ns")) / 1000.0


def _paired_metrics(group: Mapping[str, Mapping[str, str]]) -> dict[str, float]:
    sham = group["A_ALL_DONE_SHAM"]
    eager = group["B_EAGER_JOINSTREAM"]
    gated = group["C_PROGRESS_GATED_JOINSTREAM"]
    baseline_producer = _elapsed_us(sham, "producer_end_ns")
    _require(baseline_producer > 0, "ALL_DONE_SHAM producer duration must be positive")
    baseline_total = _elapsed_us(sham, "total_end_ns")
    result: dict[str, float] = {}
    for label, row in (("eager", eager), ("gated", gated)):
        producer = _elapsed_us(row, "producer_end_ns")
        result[f"{label}_join_to_gate_latency_us"] = (
            _int(row, "gate_satisfied_ns") - _int(row, "join_close_ns")
        ) / 1000.0
        result[f"{label}_notification_latency_us"] = (
            _int(row, "consumer_observe_ns")
            - max(_int(row, "join_close_ns"), _int(row, "gate_satisfied_ns"))
        ) / 1000.0
        result[f"{label}_natural_overlap_window_us"] = (
            _int(row, "producer_end_ns") - _int(row, "consumer_start_ns")
        ) / 1000.0
        result[f"{label}_critical_completion_gain_us"] = (
            _elapsed_us(sham, "consumer_end_ns") - _elapsed_us(row, "consumer_end_ns")
        )
        result[f"{label}_producer_regression_ratio"] = (
            producer - baseline_producer
        ) / baseline_producer
        result[f"{label}_total_makespan_delta_us"] = (
            _elapsed_us(row, "total_end_ns") - baseline_total
        )
        result[f"{label}_remaining_work_at_start"] = float(
            _int(row, "remaining_producer_work_at_consumer_start")
        )
    result["gating_gain_us"] = (
        _elapsed_us(eager, "consumer_end_ns") - _elapsed_us(gated, "consumer_end_ns")
    )
    result["gating_regression_reduction_ratio"] = (
        result["eager_producer_regression_ratio"]
        - result["gated_producer_regression_ratio"]
    )
    return result


def _aggregate_cells(
    groups: Mapping[tuple[str, str], Sequence[Mapping[str, Mapping[str, str]]]],
    timer_resolution_us: float,
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for factor, paired_groups in sorted(groups.items()):
        paired = [_paired_metrics(group) for group in paired_groups]
        metrics = {
            name: asdict(
                summarize(
                    (row[name] for row in paired),
                    timer_resolution_us if name.endswith("_us") else 0.0,
                )
            )
            for name in paired[0]
        }
        gated_gain = metrics["gated_critical_completion_gain_us"]
        gated_overlap = metrics["gated_natural_overlap_window_us"]
        gated_regression = metrics["gated_producer_regression_ratio"]
        clear = gated_gain["median"] > gated_gain["noise_guard"]
        safe = (
            gated_overlap["median"] > 0.0
            and gated_regression["median"] < REGRESSION_LIMIT
            and clear
        )
        output.append(
            {
                "route_distribution": factor[0],
                "problem_scale": factor[1],
                "paired_utility_repeats": len(paired_groups),
                "metrics": metrics,
                "gated_clear_gain": clear,
                "gated_safe_benefit": safe,
                "gated_observable_natural_window": gated_overlap["median"] >= OBSERVABLE_US,
            }
        )
    return output


def _adjudicate(cells: Sequence[Mapping[str, object]]) -> tuple[str, str, str, dict[str, object]]:
    safe = [cell for cell in cells if cell["gated_safe_benefit"]]
    overlap = [cell for cell in cells if cell["metrics"]["gated_natural_overlap_window_us"]["median"] > 0.0]
    overlap_regsafe = [
        cell
        for cell in overlap
        if cell["metrics"]["gated_producer_regression_ratio"]["median"] < REGRESSION_LIMIT
    ]
    # The frozen rule asks whether any formal cell exposes a >=1 us natural
    # window; the separate `safe` count already enforces clear gain and the
    # producer-regression limit for SUPPORT.
    has_observable = any(
        cell["gated_observable_natural_window"] for cell in cells
    )
    has_skewed = any(cell["route_distribution"] == "SKEWED" for cell in safe)
    if len(safe) >= 2 and has_observable and has_skewed:
        primary = "SUPPORT_REAL_TAIL_ACTION_SPACE"
    elif not overlap:
        primary = "WEAKEN_NO_NATURAL_HEADROOM"
    elif len(overlap_regsafe) >= 1 and (
        len(safe) < 2 or not has_observable
    ):
        primary = "WEAKEN_UPPER_BOUND_TOO_SMALL"
    else:
        primary = "WEAKEN_REAL_MOE_APPLICABILITY"

    necessary_cells = [
        cell
        for cell in cells
        if cell["metrics"]["eager_producer_regression_ratio"]["median"] >= REGRESSION_LIMIT
        and cell["metrics"]["gated_producer_regression_ratio"]["median"] < REGRESSION_LIMIT
        and cell["gated_clear_gain"]
    ]
    eager_safe = [
        cell
        for cell in cells
        if cell["metrics"]["eager_natural_overlap_window_us"]["median"] > 0.0
        and cell["metrics"]["eager_producer_regression_ratio"]["median"] < REGRESSION_LIMIT
        and cell["metrics"]["eager_critical_completion_gain_us"]["median"]
        > cell["metrics"]["eager_critical_completion_gain_us"]["noise_guard"]
    ]
    substantive_gating_improvement = [
        cell
        for cell in eager_safe
        if cell["metrics"]["gating_gain_us"]["median"]
        > cell["metrics"]["gating_gain_us"]["noise_guard"]
        or cell["metrics"]["gating_regression_reduction_ratio"]["median"]
        > cell["metrics"]["gating_regression_reduction_ratio"]["noise_guard"]
    ]
    if necessary_cells:
        secondary = "GATING_NECESSARY"
    elif (
        len(eager_safe) >= len(safe)
        and len(eager_safe) >= 2
        and not substantive_gating_improvement
    ):
        secondary = "GATING_NOT_NECESSARY"
    else:
        secondary = "GATING_INSUFFICIENT"
    novelty = (
        "SUPPORTS"
        if primary == "SUPPORT_REAL_TAIL_ACTION_SPACE" and secondary == "GATING_NECESSARY"
        else "WEAKENS"
    )
    gates = {
        "safe_benefit_cells": len(safe),
        "safe_skewed_cells": sum(cell["route_distribution"] == "SKEWED" for cell in safe),
        "legal_overlap_cells": len(overlap),
        "overlap_and_regression_safe_cells": len(overlap_regsafe),
        "has_at_least_1us_natural_window": has_observable,
        "gating_rescue_cells": len(necessary_cells),
        "eager_safe_cells": len(eager_safe),
        "substantive_gating_improvement_cells": len(substantive_gating_improvement),
    }
    return primary, secondary, novelty, gates


def analyze(
    calibration_path: Path,
    run_lock_path: Path,
    formal_csv_path: Path,
    environment_path: Path,
) -> dict[str, object]:
    environment = _load_json(environment_path)
    if environment.get("status") == "BLOCKED_NO_GPU" or environment.get("gpu_available") is False:
        return {
            "schema": SCHEMA,
            "primary_verdict": "BLOCKED_NO_GPU",
            "secondary_gating_interpretation": "GATING_INSUFFICIENT",
            "novelty_positioning": "DOES_NOT_ADDRESS",
            "evidence_level": "SINGLE_GPU_REALISTIC_MOE_TAIL_MICROBENCHMARK",
            "evaluation_type": EVALUATION_TYPE,
            "correctness_semantics": CORRECTNESS_SEMANTICS,
        }
    timer_resolution = _validate_environment(environment)
    calibration = _load_json(calibration_path)
    run_lock = _load_json(run_lock_path)
    _require(run_lock.get("calibration_route_seed") == calibration.get("calibration_route_seed"), "calibration seed differs from run lock")
    calibration_cells = _validate_calibration(calibration, run_lock)
    run_cells, counts = _validate_run_lock(run_lock, calibration_path, calibration_cells)
    rows = _read_csv(formal_csv_path)
    groups = _group_rows(rows, run_cells, counts, int(run_lock["formal_route_seed"]))
    cells = _aggregate_cells(groups, timer_resolution)
    _require(len(cells) == 4, "analysis must aggregate exactly 4 cells")
    primary, secondary, novelty, gates = _adjudicate(cells)
    return {
        "schema": SCHEMA,
        "primary_verdict": primary,
        "secondary_gating_interpretation": secondary,
        "novelty_positioning": novelty,
        "evidence_level": "SINGLE_GPU_REALISTIC_MOE_TAIL_MICROBENCHMARK",
        "evaluation_type": EVALUATION_TYPE,
        "correctness_semantics": CORRECTNESS_SEMANTICS,
        "inputs": {
            "calibration_json": str(calibration_path),
            "calibration_sha256": _sha256(calibration_path),
            "run_lock_json": str(run_lock_path),
            "run_lock_sha256": _sha256(run_lock_path),
            "formal_run_csv": str(formal_csv_path),
            "formal_run_sha256": _sha256(formal_csv_path),
            "environment_json": str(environment_path),
            "environment_sha256": _sha256(environment_path),
        },
        "protocol": {
            "cells": "route_distribution(2) x problem_scale(2) = 4",
            "variants": list(VARIANTS),
            "gate_candidates_remaining_ratio": list(GATE_CANDIDATES),
            "producer_regression_limit": REGRESSION_LIMIT,
            "noise_guard": "max(timer_resolution, 3*1.4826*MAD(paired metric))",
            "paired_before_aggregation": True,
            "formal_synthetic_delay": False,
            "artificial_sm_reservation": False,
        },
        "cells": cells,
        "gates": gates,
        "claim_ceiling": (
            "Single-GPU real-hardware microbenchmark with synthetic routes, "
            "inputs, and weights; A/B/C correctness is variant-equivalence only. "
            "PROGRESS_GATED is an oracle upper bound, not an online policy, "
            "real-router prevalence result, production-kernel result, or serving result."
        ),
    }


def _invalid(error: Exception) -> dict[str, object]:
    return {
        "schema": SCHEMA,
        "primary_verdict": "INVALID_EXPERIMENT",
        "secondary_gating_interpretation": "GATING_INSUFFICIENT",
        "novelty_positioning": "DOES_NOT_ADDRESS",
        "evidence_level": "SINGLE_GPU_REALISTIC_MOE_TAIL_MICROBENCHMARK",
        "evaluation_type": EVALUATION_TYPE,
        "correctness_semantics": CORRECTNESS_SEMANTICS,
        "contract_failures": [str(error)],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--run-lock", type=Path, required=True)
    parser.add_argument("--formal-csv", type=Path, required=True)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        payload = analyze(args.calibration, args.run_lock, args.formal_csv, args.environment)
    except (ContractError, KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as error:
        payload = _invalid(error)
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(rendered)


if __name__ == "__main__":
    main()
