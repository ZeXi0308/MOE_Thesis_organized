from __future__ import annotations

"""Read-only GPU-to-CPU backfeed analyzer for the JoinStream pilot.

The analyzer never imports, mutates, or reruns the CPU Oracle.  It consumes the
frozen CPU result plus paired GPU CSV rows and emits one mechanical verdict.
"""

import argparse
import csv
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Iterable, Mapping, Sequence


SCHEMA = "joinstream-gpu-backfeed-v1"
RAW_SCHEMA = "joinstream-gpu-raw-v1"
CPU_SCHEMA = "joinstream-exploratory-pilot-v1"
TAIL_GAPS_US = (0.0, 5.0, 15.0, 30.0)
RESIDENCIES = ("tail-friendly", "near-saturating")
VARIANTS = ("A_WholeBarrier", "B_AllDoneSham", "C_JoinStream")
FROZEN_PERMUTATIONS = (
    VARIANTS,
    (VARIANTS[0], VARIANTS[2], VARIANTS[1]),
    (VARIANTS[1], VARIANTS[0], VARIANTS[2]),
    (VARIANTS[1], VARIANTS[2], VARIANTS[0]),
    (VARIANTS[2], VARIANTS[0], VARIANTS[1]),
    (VARIANTS[2], VARIANTS[1], VARIANTS[0]),
)
MODES = ("correctness", "utility")
EXPECTED_REPEATS = {"correctness": 30, "utility": 200}
EXPECTED_WARMUPS = 30
NONZERO_TAILS = (5.0, 15.0, 30.0)
REGRESSION_LIMIT = 0.05
OBSERVABLE_GAIN_US = 1.0
MAD_SCALE = 3.0 * 1.4826
EPS = 1e-9

DEFAULT_CPU_ORACLE = Path(
    "artifacts/joinstream_pilot/20260810_184136/joinstream_results.json"
)

REQUIRED_COLUMNS = {
    "schema_version",
    "mode",
    "cell_id",
    "tail_gap_us",
    "residency",
    "repeat_kind",
    "repeat_index",
    "permutation_slot",
    "variant",
    "producer_blocks",
    "producer_block_size",
    "consumer_grid_size",
    "consumer_block_size",
    "producer_launches",
    "consumer_launches",
    "input_hash",
    "work_contract_hash",
    "tail_fma_chunks_per_thread",
    "producer_start_ns",
    "join_close_ns",
    "row_materialized_ns",
    "flag_publish_ns",
    "consumer_entry_ns",
    "consumer_observe_ns",
    "consumer_start_ns",
    "consumer_end_ns",
    "producer_end_ns",
    "all_blocks_done_ns",
    "total_end_ns",
    "producer_elapsed_ns",
    "consumer_end_elapsed_ns",
    "total_elapsed_ns",
    "visibility_latency_ns",
    "overlap_window_ns",
    "tail_calibration_error_ns",
    "row_hash",
    "consumer_hash",
    "reference_row_hash",
    "reference_consumer_hash",
    "correctness_pass",
    "timestamp_contract_pass",
    "cuda_error",
    "contributors_claimed",
    "expected_contributors",
    "join_counter_final",
    "blocks_done_final",
    "expected_blocks",
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


@dataclass(frozen=True)
class CPUCell:
    episode_id: str
    batch_rows: int
    curve: str
    tax_us: float
    critical_request: str
    query_headroom_us: float


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def _finite(value: object, field: str) -> float:
    _require(type(value) in {int, float}, f"{field} must be numeric")
    result = float(value)
    _require(math.isfinite(result), f"{field} must be finite")
    return result


def _csv_float(row: Mapping[str, str], field: str) -> float:
    try:
        result = float(row[field])
    except (KeyError, ValueError) as error:
        raise ContractError(f"invalid numeric CSV field {field}") from error
    _require(math.isfinite(result), f"CSV field {field} must be finite")
    return result


def _csv_int(row: Mapping[str, str], field: str) -> int:
    # `%globaltimer` is an absolute 64-bit nanosecond counter.  Parsing it via
    # binary64 first loses low bits once the value exceeds 2**53 and can create
    # false elapsed-time mismatches.  The raw contract emits decimal integers,
    # so preserve them exactly.
    try:
        value = row[field].strip()
        _require(bool(value), f"CSV field {field} must be non-empty")
        return int(value, 10)
    except (KeyError, ValueError) as error:
        raise ContractError(f"invalid integral CSV field {field}") from error


def _csv_bool(row: Mapping[str, str], field: str) -> bool:
    value = row.get(field, "").strip().lower()
    _require(value in {"true", "false", "1", "0"}, f"invalid boolean {field}")
    return value in {"true", "1"}


def _percentile(sorted_values: Sequence[float], quantile: float) -> float:
    _require(bool(sorted_values), "percentile needs observations")
    position = (len(sorted_values) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def summarize(values: Iterable[float], timer_resolution_us: float) -> Distribution:
    rows = sorted(float(value) for value in values)
    _require(bool(rows) and all(math.isfinite(value) for value in rows), "empty/nonfinite metric")
    median = statistics.median(rows)
    deviations = [abs(value - median) for value in rows]
    mad = statistics.median(deviations)
    return Distribution(
        median=median,
        mad=mad,
        p10=_percentile(rows, 0.10),
        p90=_percentile(rows, 0.90),
        noise_guard=max(timer_resolution_us, MAD_SCALE * mad),
        count=len(rows),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_cpu_cells(path: Path) -> tuple[list[CPUCell], dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(payload.get("schema") == CPU_SCHEMA, "unexpected CPU Oracle schema")
    rows = payload.get("cells")
    _require(isinstance(rows, list) and len(rows) == 8, "CPU Oracle must have exactly 8 cells")
    expected = {
        (batch, curve, tax)
        for batch in (2, 4)
        for curve in ("uniform", "tail")
        for tax in (0.0, 2.0)
    }
    observed: set[tuple[int, str, float]] = set()
    cells: list[CPUCell] = []
    for raw in rows:
        _require(isinstance(raw, dict), "CPU cell must be an object")
        factors = raw.get("factors")
        _require(isinstance(factors, dict), "CPU cell factors missing")
        batch = int(_finite(factors.get("batch_rows"), "batch_rows"))
        curve = factors.get("curve")
        tax = _finite(factors.get("tax_us"), "tax_us")
        key = (batch, str(curve), tax)
        _require(key in expected and key not in observed, "invalid/duplicate CPU factor cell")
        observed.add(key)
        baseline = raw.get("baseline_atomic_exact")
        expanded = raw.get("expanded_joinstream_exact")
        _require(isinstance(baseline, dict) and isinstance(expanded, dict), "CPU results missing")
        base_completion = baseline.get("request_completion_us")
        expanded_completion = expanded.get("request_completion_us")
        _require(
            isinstance(base_completion, dict)
            and isinstance(expanded_completion, dict)
            and set(base_completion) == set(expanded_completion)
            and len(base_completion) == batch,
            "CPU request completion maps do not match M",
        )
        gains = {
            request_id: _finite(base_completion[request_id], "baseline completion")
            - _finite(expanded_completion[request_id], "expanded completion")
            for request_id in base_completion
        }
        critical = min(gains, key=lambda request_id: (-gains[request_id], request_id))
        headroom = gains[critical]
        _require(headroom > EPS, "CPU critical request has no positive headroom")
        cells.append(
            CPUCell(
                episode_id=str(raw.get("episode_id")),
                batch_rows=batch,
                curve=str(curve),
                tax_us=tax,
                critical_request=critical,
                query_headroom_us=headroom,
            )
        )
    _require(observed == expected, "CPU 8-cell factor product is incomplete")
    cells.sort(key=lambda cell: (cell.batch_rows, cell.curve, cell.tax_us))
    return cells, payload


def _timer_resolution_us(meta: Mapping[str, object]) -> float:
    timer = meta.get("timer")
    _require(isinstance(timer, dict), "meta.timer missing")
    for field in ("resolution_ns", "globaltimer_resolution_ns", "timer_resolution_ns"):
        if field in timer:
            value = _finite(timer[field], f"meta.timer.{field}") / 1000.0
            _require(value > 0, "timer resolution must be positive")
            return value
    raise ContractError("meta.timer resolution_ns missing")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        _require(reader.fieldnames is not None, "GPU raw CSV has no header")
        missing = REQUIRED_COLUMNS - set(reader.fieldnames)
        _require(not missing, f"GPU raw CSV missing columns: {sorted(missing)}")
        return [dict(row) for row in reader]


def _cell_key(row: Mapping[str, str]) -> tuple[float, str]:
    tail = _csv_float(row, "tail_gap_us")
    residency = row["residency"]
    _require(tail in TAIL_GAPS_US, f"unexpected tail_gap_us {tail}")
    _require(residency in RESIDENCIES, f"unexpected residency {residency}")
    return tail, residency


def _validate_row(row: Mapping[str, str]) -> None:
    _require(row["schema_version"] == RAW_SCHEMA, "unexpected raw schema_version")
    _require(row["mode"] in MODES, "unexpected mode")
    _require(row["repeat_kind"] in {"warmup", "measured"}, "unexpected repeat_kind")
    _require(row["variant"] in VARIANTS, "unexpected variant")
    _cell_key(row)
    _csv_int(row, "repeat_index")
    slot = _csv_int(row, "permutation_slot")
    _require(slot in {0, 1, 2}, "permutation_slot must be 0, 1, or 2")
    for field in (
        "producer_blocks",
        "producer_block_size",
        "consumer_grid_size",
        "consumer_block_size",
    ):
        _require(_csv_int(row, field) > 0, f"{field} must be positive")
    _require(
        _csv_int(row, "producer_launches") == 1
        and _csv_int(row, "consumer_launches") == 1,
        "each variant must use exactly one producer and one consumer launch",
    )
    _require(bool(row["input_hash"]), "input_hash must be non-empty")
    _require(bool(row["work_contract_hash"]), "work_contract_hash must be non-empty")
    producer_blocks = _csv_int(row, "producer_blocks")
    _require(
        _csv_int(row, "tail_fma_chunks_per_thread") >= 0,
        "tail_fma_chunks_per_thread must be non-negative",
    )
    _require(
        _csv_int(row, "contributors_claimed") == 4
        and _csv_int(row, "expected_contributors") == 4
        and _csv_int(row, "join_counter_final") == 4,
        "dynamic K-way join did not close exactly at frozen K=4",
    )
    _require(
        _csv_int(row, "blocks_done_final") == producer_blocks
        and _csv_int(row, "expected_blocks") == producer_blocks,
        "producer blocks-done counter does not equal launched grid",
    )
    _require(_csv_bool(row, "timestamp_contract_pass"), "timestamp contract failed")
    _require(not row.get("cuda_error", "").strip(), "CUDA error present")
    if row["variant"] in {"B_AllDoneSham", "C_JoinStream"}:
        _require(
            _csv_int(row, "consumer_entry_ns")
            <= _csv_int(row, "producer_start_ns"),
            "AllDoneSham/JoinStream consumer did not pre-enter before producer start",
        )
    if row["repeat_kind"] != "measured":
        return
    _require(_csv_bool(row, "correctness_pass"), "correctness_pass is false")
    _require(
        bool(row["row_hash"])
        and row["row_hash"] == row["reference_row_hash"],
        "row hash mismatch",
    )
    _require(
        bool(row["consumer_hash"])
        and row["consumer_hash"] == row["reference_consumer_hash"],
        "consumer hash mismatch",
    )
    start = _csv_int(row, "producer_start_ns")
    join = _csv_int(row, "join_close_ns")
    materialized = _csv_int(row, "row_materialized_ns")
    producer_end = _csv_int(row, "producer_end_ns")
    all_done = _csv_int(row, "all_blocks_done_ns")
    consumer_start = _csv_int(row, "consumer_start_ns")
    consumer_end = _csv_int(row, "consumer_end_ns")
    total_end = _csv_int(row, "total_end_ns")
    _require(
        start <= join <= materialized <= all_done <= producer_end,
        "illegal producer/all-blocks-done timestamp order",
    )
    _require(start <= consumer_start <= consumer_end <= total_end, "illegal consumer timestamp order")
    _require(producer_end <= total_end, "producer ends after total_end")
    variant = row["variant"]
    if variant == "A_WholeBarrier":
        _require(consumer_start >= producer_end, "WholeBarrier consumer starts before producer end")
    else:
        publish = _csv_int(row, "flag_publish_ns")
        entry = _csv_int(row, "consumer_entry_ns")
        observe = _csv_int(row, "consumer_observe_ns")
        _require(materialized <= publish <= observe <= consumer_start, "illegal publish/observe/start order")
        _require(publish <= producer_end, "producer publishes after producer end")
        if variant == "B_AllDoneSham":
            _require(publish >= all_done, "AllDoneSham publishes before all blocks are done")
        _require(entry <= observe, "consumer observes before entry")
        _require(
            _csv_int(row, "visibility_latency_ns") == observe - publish,
            "visibility_latency_ns mismatch",
        )
    _require(
        _csv_int(row, "producer_elapsed_ns") == producer_end - start,
        "producer_elapsed_ns mismatch",
    )
    _require(
        _csv_int(row, "consumer_end_elapsed_ns") == consumer_end - start,
        "consumer_end_elapsed_ns mismatch",
    )
    _require(
        _csv_int(row, "total_elapsed_ns") == total_end - start,
        "total_elapsed_ns mismatch",
    )
    _require(
        _csv_int(row, "overlap_window_ns") == producer_end - consumer_start,
        "overlap_window_ns mismatch",
    )
    target_ns = round(_csv_float(row, "tail_gap_us") * 1000.0)
    _require(
        _csv_int(row, "tail_calibration_error_ns")
        == (producer_end - materialized) - target_ns,
        "tail_calibration_error_ns mismatch",
    )


def _paired_groups(
    rows: Sequence[dict[str, str]],
    mode: str,
    *,
    repeat_kind: str = "measured",
) -> dict[tuple[float, str], list[dict[str, dict[str, str]]]]:
    measured = [
        row
        for row in rows
        if row["mode"] == mode and row["repeat_kind"] == repeat_kind
    ]
    grouped: dict[tuple[float, str, int], dict[str, dict[str, str]]] = {}
    cell_ids: dict[tuple[float, str], str] = {}
    for row in measured:
        cell = _cell_key(row)
        prior_id = cell_ids.setdefault(cell, row["cell_id"])
        _require(prior_id == row["cell_id"] and bool(prior_id), "cell_id is not one-to-one with factors")
        repeat = _csv_int(row, "repeat_index")
        group = grouped.setdefault((cell[0], cell[1], repeat), {})
        _require(row["variant"] not in group, "duplicate variant in paired repeat")
        group[row["variant"]] = row
    expected_cells = {(tail, residency) for tail in TAIL_GAPS_US for residency in RESIDENCIES}
    _require(set(cell_ids) == expected_cells, f"{mode} does not contain exactly the frozen 8 cells")
    result: dict[tuple[float, str], list[dict[str, dict[str, str]]]] = {
        cell: [] for cell in sorted(expected_cells)
    }
    expected_repeats = (
        EXPECTED_REPEATS[mode] if repeat_kind == "measured" else EXPECTED_WARMUPS
    )
    for cell in sorted(expected_cells):
        repeat_ids = sorted(key[2] for key in grouped if key[:2] == cell)
        _require(repeat_ids == list(range(expected_repeats)), f"{mode} repeat indices incomplete for {cell}")
        for repeat in repeat_ids:
            group = grouped[(cell[0], cell[1], repeat)]
            _require(set(group) == set(VARIANTS), "paired repeat must contain exactly A/B/C")
            _require(
                {_csv_int(row, "permutation_slot") for row in group.values()} == {0, 1, 2},
                "paired repeat permutation slots are not unique",
            )
            expected_order = FROZEN_PERMUTATIONS[repeat % len(FROZEN_PERMUTATIONS)]
            for slot, variant in enumerate(expected_order):
                _require(
                    _csv_int(group[variant], "permutation_slot") == slot,
                    "paired repeat does not follow frozen six-permutation rotation",
                )
            for field in (
                "producer_blocks",
                "producer_block_size",
                "consumer_grid_size",
                "consumer_block_size",
                "producer_launches",
                "consumer_launches",
                "input_hash",
                "work_contract_hash",
                "tail_fma_chunks_per_thread",
                "row_hash",
                "consumer_hash",
                "reference_row_hash",
                "reference_consumer_hash",
            ):
                _require(len({row[field] for row in group.values()}) == 1, f"paired A/B/C mismatch in {field}")
            result[cell].append(group)
    return result


def _us(ns: float) -> float:
    return ns / 1000.0


def _elapsed(row: Mapping[str, str], field: str) -> float:
    return _us(_csv_int(row, field) - _csv_int(row, "producer_start_ns"))


def _metrics_for_group(group: Mapping[str, Mapping[str, str]]) -> dict[str, float]:
    whole = group["A_WholeBarrier"]
    sham = group["B_AllDoneSham"]
    stream = group["C_JoinStream"]
    whole_producer = _elapsed(whole, "producer_end_ns")
    whole_total = _elapsed(whole, "total_end_ns")
    _require(whole_producer > 0 and whole_total > 0, "WholeBarrier durations must be positive")
    stream_producer = _elapsed(stream, "producer_end_ns")
    stream_total = _elapsed(stream, "total_end_ns")
    critical_vs_sham = _elapsed(sham, "consumer_end_ns") - _elapsed(stream, "consumer_end_ns")
    sham_structure = _elapsed(sham, "consumer_end_ns") - _elapsed(whole, "consumer_end_ns")
    critical_vs_whole = critical_vs_sham - sham_structure
    direct = _elapsed(whole, "consumer_end_ns") - _elapsed(stream, "consumer_end_ns")
    _require(abs(critical_vs_whole - direct) <= EPS, "critical gain algebra does not close")
    producer_tax = stream_producer - whole_producer
    total_delta = stream_total - whole_total
    return {
        "actual_tail_window_us": _us(
            _csv_int(whole, "producer_end_ns") - _csv_int(whole, "row_materialized_ns")
        ),
        "tail_calibration_error_us": _us(_csv_int(whole, "tail_calibration_error_ns")),
        "publish_tax_us": _us(_csv_int(stream, "flag_publish_ns") - _csv_int(stream, "row_materialized_ns")),
        "visibility_latency_us": _us(_csv_int(stream, "consumer_observe_ns") - _csv_int(stream, "flag_publish_ns")),
        "dispatch_tax_us": _us(_csv_int(stream, "consumer_start_ns") - _csv_int(stream, "consumer_observe_ns")),
        "useful_interference_tax_us": _us(
            (_csv_int(stream, "consumer_end_ns") - _csv_int(stream, "consumer_start_ns"))
            - (_csv_int(sham, "consumer_end_ns") - _csv_int(sham, "consumer_start_ns"))
        ),
        "overlap_window_us": _us(_csv_int(stream, "producer_end_ns") - _csv_int(stream, "consumer_start_ns")),
        "critical_completion_gain_us": critical_vs_sham,
        "sham_structure_tax_us": sham_structure,
        "critical_gain_vs_whole_us": critical_vs_whole,
        "producer_tax_us": producer_tax,
        "total_makespan_delta_us": total_delta,
        "producer_regression_ratio": producer_tax / whole_producer,
        "total_regression_ratio": total_delta / whole_total,
    }


def aggregate_gpu_cells(
    utility_groups: Mapping[tuple[float, str], Sequence[Mapping[str, Mapping[str, str]]]],
    timer_resolution_us: float,
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for (tail, residency), groups in sorted(utility_groups.items()):
        paired = [_metrics_for_group(group) for group in groups]
        names = tuple(paired[0])
        metrics = {
            name: asdict(summarize((row[name] for row in paired), timer_resolution_us))
            for name in names
        }
        output.append(
            {
                "tail_gap_us": tail,
                "residency": residency,
                "paired_repeats": len(groups),
                "metrics": metrics,
            }
        )
    return output


def _interpolate(points: Sequence[tuple[float, float]], query: float) -> float:
    _require(len(points) == 4, "response curve must have four tail points")
    ordered = sorted(points)
    _require(all(ordered[index][0] < ordered[index + 1][0] - EPS for index in range(3)), "actual tail windows must be strictly increasing")
    _require(ordered[0][0] - EPS <= query <= ordered[-1][0] + EPS, "CPU headroom is outside measured tail range; extrapolation forbidden")
    for left, right in zip(ordered, ordered[1:]):
        if query <= right[0] + EPS:
            weight = (query - left[0]) / (right[0] - left[0])
            return left[1] + weight * (right[1] - left[1])
    return ordered[-1][1]


def _query_available(points: Sequence[tuple[float, float]], query: float) -> bool:
    ordered = sorted(points)
    _require(len(ordered) == 4, "response curve must have four tail points")
    _require(
        all(ordered[index][0] < ordered[index + 1][0] - EPS for index in range(3)),
        "actual tail windows must be strictly increasing",
    )
    return ordered[0][0] - EPS <= query <= ordered[-1][0] + EPS


def backfeed_cpu_cells(cpu_cells: Sequence[CPUCell], gpu_cells: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    by_residency: dict[str, list[Mapping[str, object]]] = {
        residency: [row for row in gpu_cells if row["residency"] == residency]
        for residency in RESIDENCIES
    }
    output: list[dict[str, object]] = []
    for cell in cpu_cells:
        residency_rows: list[dict[str, object]] = []
        for residency in RESIDENCIES:
            rows = by_residency[residency]
            def points(metric: str, statistic: str) -> list[tuple[float, float]]:
                return [
                    (
                        float(row["metrics"]["actual_tail_window_us"]["median"]),
                        float(row["metrics"][metric][statistic]),
                    )
                    for row in rows
                ]
            gain_points = points("critical_gain_vs_whole_us", "median")
            if not _query_available(gain_points, cell.query_headroom_us):
                residency_rows.append(
                    {
                        "residency": residency,
                        "status": "unavailable_no_extrapolation",
                        "positive": False,
                    }
                )
                continue
            gain = _interpolate(gain_points, cell.query_headroom_us)
            tax_raw = _interpolate(points("producer_tax_us", "median"), cell.query_headroom_us)
            gain_noise = _interpolate(points("critical_gain_vs_whole_us", "noise_guard"), cell.query_headroom_us)
            tax_noise = _interpolate(points("producer_tax_us", "noise_guard"), cell.query_headroom_us)
            producer_tax = max(0.0, tax_raw)
            backfed_gain = gain - (cell.batch_rows - 1) * producer_tax
            backfed_noise = gain_noise + (cell.batch_rows - 1) * tax_noise
            positive = backfed_gain > 0.0 and backfed_gain > backfed_noise
            residency_rows.append(
                {
                    "residency": residency,
                    "status": "available_interpolated",
                    "interpolated_critical_gain_us": gain,
                    "interpolated_producer_tax_raw_us": tax_raw,
                    "charged_producer_tax_us": producer_tax,
                    "backfed_gain_us": backfed_gain,
                    "backfed_noise_us": backfed_noise,
                    "positive": positive,
                }
            )
        output.append(
            {
                **asdict(cell),
                "residencies": residency_rows,
                "positive": any(row["positive"] for row in residency_rows),
                "positive_in_both_residencies": all(row["positive"] for row in residency_rows),
            }
        )
    return output


def _mechanical_verdict(gpu_cells: Sequence[Mapping[str, object]], backfeed: Sequence[Mapping[str, object]]) -> tuple[str, dict[str, object]]:
    nonzero = [row for row in gpu_cells if float(row["tail_gap_us"]) in NONZERO_TAILS]
    overlap_count = sum(float(row["metrics"]["overlap_window_us"]["median"]) > 0.0 for row in nonzero)
    sham_gain_count = sum(
        float(row["metrics"]["critical_completion_gain_us"]["median"])
        > float(row["metrics"]["critical_completion_gain_us"]["noise_guard"])
        for row in nonzero
    )
    regression: dict[str, dict[str, float | bool]] = {}
    regression_failed = False
    for residency in RESIDENCIES:
        rows = [row for row in nonzero if row["residency"] == residency]
        producer = statistics.median(float(row["metrics"]["producer_regression_ratio"]["median"]) for row in rows)
        total = statistics.median(float(row["metrics"]["total_regression_ratio"]["median"]) for row in rows)
        failed = producer > REGRESSION_LIMIT or total > REGRESSION_LIMIT
        regression_failed = regression_failed or failed
        regression[residency] = {
            "median_producer_regression_ratio": producer,
            "median_total_regression_ratio": total,
            "exceeds_5_percent": failed,
        }
    positive = [row for row in backfeed if bool(row["positive"])]
    observable = any(
        float(residency["backfed_gain_us"]) >= OBSERVABLE_GAIN_US
        and float(residency["backfed_gain_us"]) > float(residency["backfed_noise_us"])
        for row in positive
        for residency in row["residencies"]
        if residency.get("positive")
    )
    gates = {
        "nonzero_tail_overlap_cells": overlap_count,
        "nonzero_tail_sham_gain_above_noise_cells": sham_gain_count,
        "regression_by_residency": regression,
        "positive_cpu_cells": len(positive),
        "observable_positive_cpu_cell": observable,
    }
    if overlap_count < 2:
        return "WEAKEN_GPU_SCHEDULABILITY", gates
    if sham_gain_count < 2 or regression_failed or len(positive) < 2 or not observable:
        return "WEAKEN_TAX_DOMINATES", gates
    return "SUPPORT_GPU_ACTION_SPACE", gates


def analyze(raw_csv: Path, meta_json: Path, cpu_oracle: Path) -> dict[str, object]:
    meta = json.loads(meta_json.read_text(encoding="utf-8"))
    _require(isinstance(meta, dict), "GPU meta must be an object")
    status = str(meta.get("status", ""))
    if status == "BLOCKED_NO_GPU":
        return {
            "schema": SCHEMA,
            "verdict": "BLOCKED_NO_GPU",
            "novelty_positioning": "DOES_NOT_ADDRESS",
            "gpu_meta_sha256": _sha256(meta_json),
            "contract_failures": [],
        }
    failures = meta.get("contract_failures", [])
    _require(isinstance(failures, list) and not failures, "GPU meta reports contract failures")
    timer_resolution = _timer_resolution_us(meta)
    cpu_cells, cpu_payload = load_cpu_cells(cpu_oracle)
    rows = _read_csv(raw_csv)
    _require(bool(rows), "GPU raw CSV is empty")
    for row in rows:
        _validate_row(row)
    # Both modes and their warmups are fully paired/count-checked before aggregation.
    _paired_groups(rows, "correctness", repeat_kind="warmup")
    _paired_groups(rows, "utility", repeat_kind="warmup")
    _paired_groups(rows, "correctness")
    utility = _paired_groups(rows, "utility")
    gpu_cells = aggregate_gpu_cells(utility, timer_resolution)
    _require(len(gpu_cells) == 8, "aggregated GPU result must have exactly 8 cells")
    backfeed = backfeed_cpu_cells(cpu_cells, gpu_cells)
    verdict, gates = _mechanical_verdict(gpu_cells, backfeed)
    novelty = "SUPPORTS" if verdict == "SUPPORT_GPU_ACTION_SPACE" else "WEAKENS"
    return {
        "schema": SCHEMA,
        "analysis_type": "READ_ONLY_POSTHOC_BACKFEED; CPU Oracle not imported or rerun",
        "verdict": verdict,
        "novelty_positioning": novelty,
        "inputs": {
            "gpu_raw_csv": str(raw_csv),
            "gpu_raw_sha256": _sha256(raw_csv),
            "gpu_meta_json": str(meta_json),
            "gpu_meta_sha256": _sha256(meta_json),
            "cpu_oracle": str(cpu_oracle),
            "cpu_oracle_sha256": _sha256(cpu_oracle),
            "cpu_schema": cpu_payload["schema"],
        },
        "protocol": {
            "gpu_cells": "tail_gap(4) x residency(2) = 8",
            "paired_before_aggregation": True,
            "noise_guard": "max(timer_resolution, 3*1.4826*MAD(paired metric))",
            "interpolation": "actual WholeBarrier tail window; piecewise linear; no extrapolation",
            "single_critical_request": True,
            "cpu_positive_collapse": "either residency positive; both residencies retained",
        },
        "gpu_cells": gpu_cells,
        "cpu_backfeed_cells": backfeed,
        "gates": gates,
        "claim_ceiling": (
            "Single-GPU microbenchmark plus posthoc CPU-Orcale backfeed only; "
            "not cross-layer, serving, EP/NCCL, multi-GPU, or paper evidence."
        ),
    }


def _invalid_payload(error: Exception, *, meta_json: Path | None = None) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema": SCHEMA,
        "verdict": "INVALID_MEMORY_CONTRACT",
        "novelty_positioning": "DOES_NOT_ADDRESS",
        "contract_failures": [str(error)],
    }
    if meta_json is not None and meta_json.exists():
        payload["gpu_meta_sha256"] = _sha256(meta_json)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-csv", type=Path, required=True)
    parser.add_argument("--meta-json", type=Path, required=True)
    parser.add_argument("--cpu-oracle", type=Path, default=DEFAULT_CPU_ORACLE)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        payload = analyze(args.raw_csv, args.meta_json, args.cpu_oracle)
    except (ContractError, KeyError, TypeError, json.JSONDecodeError, OSError) as error:
        payload = _invalid_payload(error, meta_json=args.meta_json)
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(rendered)


if __name__ == "__main__":
    main()
