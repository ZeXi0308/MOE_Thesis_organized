#!/usr/bin/env python3
"""Replay fixed-M=2 focal rows across four companion choices.

The completed partner-permutation run is the only selection authority.  This
script deterministically reduces its 512 focals to two safe and two unsafe
focals per layer, then replays every selected focal with four companions:
the original companion, safe rank 0, unsafe rank 0, and one layer-balanced
rank-1 safe/unsafe companion.

``plan`` is CPU-only.  ``run`` loads the already accepted model/captures and is
the only command that uses CUDA.  The result is calibration-only expert-stage
evidence; it is not a fresh-evaluation, serving, EP, or paper result.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import struct
import sys
from typing import Any, Iterable, Mapping, Sequence


EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT_DEFAULT = EXPERIMENT_DIR.parents[3]
PARTNER_RUNNER_PATH = EXPERIMENT_DIR / "run_partner_permutation_5090.py"
DEFAULT_PARTNER_DIR = (
    EXPERIMENT_DIR / "outputs" / "partner_permutation_20260810_run01"
)
DEFAULT_SOURCE_DIR = (
    EXPERIMENT_DIR / "outputs" / "semanticfence_pilot_20260810_run03"
)
SCHEDULE_SCHEMA = "semanticfence-cross-companion-schedule-v1"
RESULT_SCHEMA = "semanticfence-cross-companion-result-v1"
COMPLETE_SCHEMA = "semanticfence-cross-companion-complete-v1"
SELECTION_SEED = "semanticfence-cross-companion-metric-v1"
EXPECTED_LAYERS = 16
FOCALS_PER_LAYER_LABEL = 2
EXPECTED_FOCALS = 64
EXPECTED_CALLS = 256
WARMUPS = 3
REPEATS = 10


class ReplayError(RuntimeError):
    """The replay cannot produce an interpretable result."""


def _load_module(name: str, path: Path) -> Any:
    cached = sys.modules.get(name)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ReplayError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


PARTNER = _load_module("semanticfence_cross_companion_partner", PARTNER_RUNNER_PATH)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ReplayError(f"{path}:{line_number} is not an object")
            rows.append(value)
    return rows


def write_json_no_overwrite(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise ReplayError(f"refusing to overwrite {path}") from exc


def write_jsonl_no_overwrite(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            for row in rows:
                handle.write(
                    json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise ReplayError(f"refusing to overwrite {path}") from exc


def _selection_key(namespace: str, *values: Any) -> str:
    payload = "|".join([SELECTION_SEED, namespace, *(str(value) for value in values)])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validate_source_focal(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(rows) != 4:
        raise ReplayError("each partner focal must have exactly four alternate calls")
    first = rows[0]
    stable_fields = (
        "focal_row_id",
        "focal_baseline_label",
        "focal_original_slot",
        "focal_old_m1_reference_sha256",
        "original_partner_row_id",
        "original_pack_id",
        "layer",
        "expert_id",
    )
    if any(row.get(field) != first.get(field) for row in rows for field in stable_fields):
        raise ReplayError("partner schedule changes fixed focal metadata")
    if first.get("focal_baseline_label") not in {"safe", "unsafe"}:
        raise ReplayError("focal label is not stable safe/unsafe")
    if int(first.get("focal_original_slot", -1)) not in {0, 1}:
        raise ReplayError("focal original slot is invalid")

    alternatives: dict[tuple[str, int], Mapping[str, Any]] = {}
    for row in rows:
        key = (str(row.get("partner_baseline_label")), int(row.get("partner_rank_within_label", -1)))
        if key in alternatives:
            raise ReplayError("duplicate partner label/rank for one focal")
        alternatives[key] = row
    expected = {("safe", 0), ("safe", 1), ("unsafe", 0), ("unsafe", 1)}
    if set(alternatives) != expected:
        raise ReplayError("partner schedule lacks the frozen safe/unsafe ranks")
    return {"fixed": dict(first), "alternatives": alternatives}


def build_cross_companion_schedule(
    source_rows: Sequence[Mapping[str, Any]],
    *,
    source_schedule_sha256: str,
    expected_layers: int = EXPECTED_LAYERS,
) -> list[dict[str, Any]]:
    """Return the deterministic 64-focal/256-call reduced schedule."""

    by_focal: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in source_rows:
        focal = str(row.get("focal_row_id", ""))
        if len(focal) != 64:
            raise ReplayError("source schedule contains an invalid focal row id")
        by_focal[focal].append(row)
    focal_info = {
        focal: _validate_source_focal(rows) for focal, rows in by_focal.items()
    }

    by_layer_label: dict[tuple[int, str], list[str]] = defaultdict(list)
    for focal, info in focal_info.items():
        fixed = info["fixed"]
        by_layer_label[(int(fixed["layer"]), str(fixed["focal_baseline_label"]))].append(focal)

    selected_by_layer: dict[int, list[str]] = {}
    for layer in range(expected_layers):
        chosen: list[str] = []
        for label in ("safe", "unsafe"):
            candidates = sorted(
                by_layer_label[(layer, label)],
                key=lambda row_id: _selection_key(
                    "focal", source_schedule_sha256, layer, label, row_id
                ),
            )
            if len(candidates) < FOCALS_PER_LAYER_LABEL:
                raise ReplayError(f"layer {layer} lacks two {label} focals")
            chosen.extend(candidates[:FOCALS_PER_LAYER_LABEL])
        selected_by_layer[layer] = chosen

    selected_focals = [
        focal for layer in range(expected_layers) for focal in selected_by_layer[layer]
    ]
    expected_focals = expected_layers * 2 * FOCALS_PER_LAYER_LABEL
    if len(selected_focals) != expected_focals or len(set(selected_focals)) != expected_focals:
        raise ReplayError("selected focal count/uniqueness changed")

    rank1_label: dict[str, str] = {}
    for layer, focals in selected_by_layer.items():
        shuffled = sorted(
            focals,
            key=lambda row_id: _selection_key(
                "rank1-balance", source_schedule_sha256, layer, row_id
            ),
        )
        for index, focal in enumerate(shuffled):
            rank1_label[focal] = "safe" if index < len(shuffled) // 2 else "unsafe"

    calls: list[dict[str, Any]] = []
    for focal in selected_focals:
        info = focal_info[focal]
        fixed = info["fixed"]
        alternatives = info["alternatives"]
        rank1 = rank1_label[focal]
        companions = (
            (
                "original",
                str(fixed["original_partner_row_id"]),
                None,
                None,
            ),
            (
                "safe_rank0",
                str(alternatives[("safe", 0)]["partner_row_id"]),
                "safe",
                0,
            ),
            (
                "unsafe_rank0",
                str(alternatives[("unsafe", 0)]["partner_row_id"]),
                "unsafe",
                0,
            ),
            (
                f"{rank1}_rank1",
                str(alternatives[(rank1, 1)]["partner_row_id"]),
                rank1,
                1,
            ),
        )
        slot = int(fixed["focal_original_slot"])
        for kind, companion_id, companion_label, partner_rank in companions:
            ordered = [focal, companion_id] if slot == 0 else [companion_id, focal]
            calls.append(
                {
                    "schema_version": SCHEDULE_SCHEMA,
                    "source_schedule_sha256": source_schedule_sha256,
                    "layer": int(fixed["layer"]),
                    "expert_id": int(fixed["expert_id"]),
                    "focal_row_id": focal,
                    "focal_baseline_label": str(fixed["focal_baseline_label"]),
                    "focal_original_slot": slot,
                    "focal_old_m1_reference_sha256": str(
                        fixed["focal_old_m1_reference_sha256"]
                    ),
                    "original_pack_id": str(fixed["original_pack_id"]),
                    "companion_kind": kind,
                    "companion_row_id": companion_id,
                    "companion_baseline_label": companion_label,
                    "source_partner_rank_within_label": partner_rank,
                    "row_ids": ordered,
                }
            )

    calls.sort(
        key=lambda row: _selection_key(
            "call-order",
            source_schedule_sha256,
            row["focal_row_id"],
            row["companion_kind"],
            row["companion_row_id"],
        )
    )
    for index, row in enumerate(calls):
        row["call_index"] = index
        row["call_sha256"] = canonical_sha256(
            {key: value for key, value in row.items() if key != "call_sha256"}
        )

    expected_calls = expected_focals * 4
    if len(calls) != expected_calls:
        raise ReplayError("cross-companion call count changed")
    counts = Counter(row["companion_kind"] for row in calls)
    if counts["original"] != expected_focals or counts["safe_rank0"] != expected_focals or counts["unsafe_rank0"] != expected_focals:
        raise ReplayError("fixed companion strata are unbalanced")
    if counts["safe_rank1"] != expected_focals // 2 or counts["unsafe_rank1"] != expected_focals // 2:
        raise ReplayError("rank-1 companion stratum is not balanced")
    return calls


def bf16_bytes_to_floats(raw: bytes) -> tuple[float, ...]:
    if len(raw) % 2:
        raise ReplayError("BF16 byte string has odd length")
    values: list[float] = []
    for (bits,) in struct.iter_unpack("<H", raw):
        values.append(struct.unpack("<f", struct.pack("<I", bits << 16))[0])
    return tuple(values)


def difference_metrics(reference: bytes, observed: bytes) -> dict[str, Any]:
    """Compute strict-BF16 count and float-space L1/L2 diagnostics."""

    if len(reference) != len(observed) or len(reference) % 2:
        raise ReplayError("metric inputs have incompatible BF16 sizes")
    differing = sum(
        reference[index : index + 2] != observed[index : index + 2]
        for index in range(0, len(reference), 2)
    )
    ref_values = bf16_bytes_to_floats(reference)
    obs_values = bf16_bytes_to_floats(observed)
    deltas = [float(right) - float(left) for left, right in zip(ref_values, obs_values)]
    abs_values = [abs(value) for value in deltas]
    return {
        "differing_count": differing,
        "max_abs": max(abs_values, default=0.0),
        "l1": math.fsum(abs_values),
        "l2": math.sqrt(math.fsum(value * value for value in deltas)),
    }


def fixed_shape_descriptor(expert: Any, *, m: int, hidden_size: int) -> dict[str, Any]:
    parameters = [
        {
            "name": name,
            "shape": [int(value) for value in parameter.shape],
            "dtype": str(parameter.dtype),
        }
        for name, parameter in expert.named_parameters(recurse=True)
    ]
    return {
        "schema_version": "semanticfence-expert-shape-v1",
        "input_shape": [int(m), int(hidden_size)],
        "output_shape": [int(m), int(hidden_size)],
        "input_dtype": "torch.bfloat16",
        "parameters": parameters,
    }


def summarize_latency(m1_ms: Sequence[float], m2_ms: Sequence[float]) -> dict[str, Any]:
    if len(m1_ms) != REPEATS or len(m2_ms) != REPEATS:
        raise ReplayError("latency vectors must contain ten paired repeats")
    if any(value <= 0 or not math.isfinite(value) for value in (*m1_ms, *m2_ms)):
        raise ReplayError("latency vector contains a non-positive/non-finite value")
    ratios = [left / right for left, right in zip(m1_ms, m2_ms)]
    return {
        "unit": "milliseconds_for_all_256_pairs",
        "m1_two_single_row_calls_ms": list(m1_ms),
        "m2_one_two_row_call_ms": list(m2_ms),
        "paired_m1_over_m2": ratios,
        "m1_median_ms": statistics.median(m1_ms),
        "m2_median_ms": statistics.median(m2_ms),
        "median_paired_m1_over_m2": statistics.median(ratios),
        "ratio_of_medians_m1_over_m2": statistics.median(m1_ms)
        / statistics.median(m2_ms),
    }


def _source_schedule(partner_dir: Path) -> tuple[Path, list[dict[str, Any]], str]:
    schedule_path = Path(partner_dir) / "frozen_inputs" / "schedule.jsonl"
    if not schedule_path.is_file():
        raise ReplayError(f"partner schedule is absent: {schedule_path}")
    rows = load_jsonl(schedule_path)
    digest = sha256_file(schedule_path)
    if len(rows) != 2048:
        raise ReplayError("completed partner schedule does not contain 2,048 calls")
    complete = json.loads((Path(partner_dir) / "COMPLETE.json").read_text(encoding="utf-8"))
    if complete.get("status") != "SUCCESS_COMPLETE":
        raise ReplayError("partner source run is not SUCCESS_COMPLETE")
    if complete.get("artifact_sha256", {}).get("frozen_inputs/schedule.jsonl") != digest:
        raise ReplayError("partner COMPLETE does not bind the frozen schedule")
    config = PARTNER.validate_config(
        PARTNER.load_json(Path(partner_dir) / "frozen_inputs" / "config.json")
    )
    PARTNER.validate_schedule(config, rows)
    return schedule_path, rows, digest


def run_plan(args: argparse.Namespace) -> int:
    _path, source_rows, digest = _source_schedule(Path(args.partner_dir).resolve())
    calls = build_cross_companion_schedule(
        source_rows, source_schedule_sha256=digest
    )
    output = Path(args.output).resolve()
    write_jsonl_no_overwrite(output, calls)
    summary = {
        "schedule_sha256": sha256_file(output),
        "focal_count": len({row["focal_row_id"] for row in calls}),
        "call_count": len(calls),
        "companion_kind_counts": dict(sorted(Counter(row["companion_kind"] for row in calls).items())),
        "gpu_executed": False,
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


def _prepare_runtime(
    *, model: Any, rows_by_id: Mapping[str, Any], schedule: Sequence[Mapping[str, Any]]
) -> tuple[dict[str, tuple[Any, Any]], list[dict[str, Any]], dict[str, str]]:
    import torch

    target_batches: dict[str, tuple[Any, Any]] = {}
    prepared_calls: list[dict[str, Any]] = []
    signatures: dict[str, str] = {}
    for call in schedule:
        focal_id = str(call["focal_row_id"])
        row_ids = list(call["row_ids"])
        materialized = [rows_by_id[row_id] for row_id in row_ids]
        if any(
            row.record.layer != int(call["layer"])
            or row.record.expert_id != int(call["expert_id"])
            for row in materialized
        ):
            raise ReplayError("scheduled rows cross the frozen layer/expert")
        for row in materialized:
            if PARTNER.PILOT._gpu().tensor_storage_sha256(row.tensor) != row.record.hidden_sha256:
                raise ReplayError("materialized hidden hash mismatch")
        expert = model.model.layers[int(call["layer"])].mlp.experts[int(call["expert_id"])]
        batch = torch.stack(
            [row.tensor.to(device="cuda", dtype=torch.bfloat16) for row in materialized]
        )
        singles = [value.unsqueeze(0) for value in batch]
        hidden_size = int(batch.shape[1])
        m2_descriptor = fixed_shape_descriptor(expert, m=2, hidden_size=hidden_size)
        m1_descriptor = fixed_shape_descriptor(expert, m=1, hidden_size=hidden_size)
        m2_signature = canonical_sha256(m2_descriptor)
        m1_signature = canonical_sha256(m1_descriptor)
        signatures.setdefault("m2", m2_signature)
        signatures.setdefault("m1", m1_signature)
        if signatures["m2"] != m2_signature or signatures["m1"] != m1_signature:
            raise ReplayError("expert shape signature differs across scheduled calls")
        if focal_id not in target_batches:
            focal_slot = int(call["focal_original_slot"])
            target_batches[focal_id] = (expert, singles[focal_slot])
        prepared_calls.append(
            {
                "call": call,
                "expert": expert,
                "batch": batch,
                "singles": singles,
                "m2_shape_signature": m2_signature,
            }
        )
    return target_batches, prepared_calls, signatures


def _fresh_m1_references(
    target_batches: Mapping[str, tuple[Any, Any]], schedule: Sequence[Mapping[str, Any]]
) -> dict[str, bytes]:
    import torch

    gpu = PARTNER.PILOT._gpu()
    expected_old = {str(row["focal_row_id"]): str(row["focal_old_m1_reference_sha256"]) for row in schedule}
    references: dict[str, bytes] = {}
    with torch.inference_mode():
        for focal_id in sorted(target_batches):
            expert, batch = target_batches[focal_id]
            values: list[bytes] = []
            for _ in range(REPEATS):
                output = expert(batch).detach().cpu().clone()
                if output.dtype != torch.bfloat16 or not bool(torch.isfinite(output).all().item()):
                    raise ReplayError("fresh M1 output is invalid")
                values.append(gpu.bf16_storage_bytes(output[0]))
            hashes = {hashlib.sha256(value).hexdigest() for value in values}
            if len(hashes) != 1:
                raise ReplayError("fresh focal M1 output is not 10/10 stable")
            observed_hash = next(iter(hashes))
            if observed_hash != expected_old[focal_id]:
                raise ReplayError("fresh focal M1 output differs from run03 reference")
            references[focal_id] = values[0]
    return references


def _execute_metric_replay(
    prepared_calls: Sequence[Mapping[str, Any]], references: Mapping[str, bytes]
) -> list[dict[str, Any]]:
    import torch

    gpu = PARTNER.PILOT._gpu()
    numeric: list[dict[str, Any]] = []
    with torch.inference_mode():
        for prepared in prepared_calls:
            call = prepared["call"]
            repeat_metrics: list[dict[str, Any]] = []
            repeat_hashes: list[str] = []
            slot = int(call["focal_original_slot"])
            for _ in range(REPEATS):
                output = prepared["expert"](prepared["batch"]).detach().cpu().clone()
                if output.dtype != torch.bfloat16 or not bool(torch.isfinite(output).all().item()):
                    raise ReplayError("M2 replay output is invalid")
                raw = gpu.bf16_storage_bytes(output[slot])
                repeat_hashes.append(hashlib.sha256(raw).hexdigest())
                repeat_metrics.append(difference_metrics(references[call["focal_row_id"]], raw))
            labels = [metric["differing_count"] == 0 for metric in repeat_metrics]
            status = "safe" if all(labels) else "unsafe" if not any(labels) else "mixed"
            numeric.append(
                {
                    **dict(call),
                    "m2_shape_signature": prepared["m2_shape_signature"],
                    "target_repeat_exact_to_m1": labels,
                    "target_label_status": status,
                    "target_label_stable_10_of_10": len(set(labels)) == 1,
                    "target_output_bitwise_stable_10_of_10": len(set(repeat_hashes)) == 1,
                    "target_repeat_sha256": repeat_hashes,
                    "target_vs_m1_repeat_metrics": repeat_metrics,
                }
            )
    return numeric


def _measure_paired_latency(prepared_calls: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    import torch

    def execute(mode: str) -> None:
        for prepared in prepared_calls:
            expert = prepared["expert"]
            if mode == "m1":
                expert(prepared["singles"][0])
                expert(prepared["singles"][1])
            else:
                expert(prepared["batch"])

    with torch.inference_mode():
        for _ in range(WARMUPS):
            execute("m1")
            execute("m2")
        torch.cuda.synchronize()

    timings: dict[str, list[float]] = {"m1": [], "m2": []}
    with torch.inference_mode():
        for repeat in range(REPEATS):
            order = ("m1", "m2") if repeat % 2 == 0 else ("m2", "m1")
            for mode in order:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                execute(mode)
                end.record()
                end.synchronize()
                timings[mode].append(float(start.elapsed_time(end)))
    return summarize_latency(timings["m1"], timings["m2"])


def _aggregate_numeric(numeric: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    statuses = Counter(str(row["target_label_status"]) for row in numeric)
    stable = sum(bool(row["target_label_stable_10_of_10"]) for row in numeric)
    output_stable = sum(
        bool(row["target_output_bitwise_stable_10_of_10"]) for row in numeric
    )
    by_focal: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in numeric:
        by_focal[str(row["focal_row_id"])].append(row)
    if len(by_focal) != EXPECTED_FOCALS or any(len(rows) != 4 for rows in by_focal.values()):
        raise ReplayError("numeric result does not contain four calls for all 64 focals")

    focal_consistent = 0
    baseline_flip_calls = 0
    baseline_flip_focals: set[str] = set()
    binary_by_focal: dict[str, list[int]] = {}
    for focal_id, rows in by_focal.items():
        observed = [str(row["target_label_status"]) for row in rows]
        if len(set(observed)) == 1:
            focal_consistent += 1
        if any(value == "mixed" for value in observed):
            continue
        binary_by_focal[focal_id] = [int(value == "safe") for value in observed]
        for row, value in zip(rows, observed):
            if value != row["focal_baseline_label"]:
                baseline_flip_calls += 1
                baseline_flip_focals.add(focal_id)

    variance: dict[str, Any] | None = None
    if len(binary_by_focal) == len(by_focal):
        means = {
            focal_id: statistics.fmean(values)
            for focal_id, values in binary_by_focal.items()
        }
        grand_mean = statistics.fmean(
            value for values in binary_by_focal.values() for value in values
        )
        denominator = sum(len(values) for values in binary_by_focal.values())
        within = math.fsum(
            (value - means[focal_id]) ** 2
            for focal_id, values in binary_by_focal.items()
            for value in values
        ) / denominator
        between = math.fsum(
            len(binary_by_focal[focal_id]) * (mean - grand_mean) ** 2
            for focal_id, mean in means.items()
        ) / denominator
        variance = {
            "definition": "population Bernoulli variance decomposition over 4 stratified companions per focal",
            "grand_safe_rate": grand_mean,
            "within_focal": within,
            "between_focal": between,
            "total": within + between,
        }

    edge_strata: dict[str, dict[str, int]] = defaultdict(
        lambda: {"calls": 0, "whole_pair_safe_edges": 0}
    )
    whole_pair_safe = 0
    for row in numeric:
        key = (
            f"focal_{row['focal_baseline_label']}__"
            f"companion_{row['resolved_companion_baseline_label']}"
        )
        edge_strata[key]["calls"] += 1
        is_edge = (
            row["target_label_status"] == "safe"
            and row["resolved_companion_baseline_label"] == "safe"
        )
        edge_strata[key]["whole_pair_safe_edges"] += int(is_edge)
        whole_pair_safe += int(is_edge)
    edge_density = {
        "definition": (
            "stratified diagnostic: observed replay target is safe AND the companion's "
            "resolved reused-calibration baseline label is safe; companion output is not replay-labeled"
        ),
        "whole_pair_safe_edges": whole_pair_safe,
        "calls": len(numeric),
        "density": whole_pair_safe / len(numeric),
        "strata": {
            key: {
                **value,
                "density": value["whole_pair_safe_edges"] / value["calls"],
            }
            for key, value in sorted(edge_strata.items())
        },
    }
    by_kind: dict[str, dict[str, Any]] = {}
    for kind in sorted({str(row["companion_kind"]) for row in numeric}):
        rows = [row for row in numeric if row["companion_kind"] == kind]
        metrics = [metric for row in rows for metric in row["target_vs_m1_repeat_metrics"]]
        by_kind[kind] = {
            "calls": len(rows),
            "safe_calls": sum(row["target_label_status"] == "safe" for row in rows),
            "unsafe_calls": sum(row["target_label_status"] == "unsafe" for row in rows),
            "mixed_calls": sum(row["target_label_status"] == "mixed" for row in rows),
            "mean_differing_count": statistics.fmean(metric["differing_count"] for metric in metrics),
            "max_differing_count": max(metric["differing_count"] for metric in metrics),
            "mean_l1": statistics.fmean(metric["l1"] for metric in metrics),
            "mean_l2": statistics.fmean(metric["l2"] for metric in metrics),
            "max_abs": max(metric["max_abs"] for metric in metrics),
        }
    return {
        "call_count": len(numeric),
        "stable_label_calls": stable,
        "unstable_label_calls": len(numeric) - stable,
        "bitwise_stable_output_calls": output_stable,
        "bitwise_unstable_output_calls": len(numeric) - output_stable,
        "target_label_status_counts": dict(sorted(statuses.items())),
        "focal_consistency": {
            "consistent_4_of_4_focals": focal_consistent,
            "inconsistent_focals": len(by_focal) - focal_consistent,
            "total_focals": len(by_focal),
        },
        "baseline_flips": {
            "call_count": baseline_flip_calls,
            "call_rate": baseline_flip_calls / len(numeric),
            "focal_count": len(baseline_flip_focals),
            "focal_rate": len(baseline_flip_focals) / len(by_focal),
        },
        "bernoulli_variance": variance,
        "whole_pair_safe_edge_density": edge_density,
        "by_companion_kind": by_kind,
    }


def run_gpu(args: argparse.Namespace) -> int:
    import torch

    partner_dir = Path(args.partner_dir).resolve()
    source_dir = Path(args.source_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise ReplayError(f"refusing existing output directory: {output_dir}")
    output_dir.mkdir(parents=True)

    schedule_path, source_rows, source_digest = _source_schedule(partner_dir)
    schedule = build_cross_companion_schedule(
        source_rows, source_schedule_sha256=source_digest
    )
    write_jsonl_no_overwrite(output_dir / "cross_companion_schedule.jsonl", schedule)

    config_path = partner_dir / "frozen_inputs" / "config.json"
    config = PARTNER.validate_config(PARTNER.load_json(config_path))
    PARTNER.verify_source_artifacts(config, source_dir)
    evidence = PARTNER.load_m2_evidence(config, source_dir)
    PARTNER.load_and_verify_plan(
        config=config,
        config_path=config_path,
        plan_dir=partner_dir / "frozen_inputs",
        evidence=evidence,
    )
    original_config, acceptance, model = PARTNER._load_live_model(
        config=config,
        source_dir=source_dir,
        acceptance_path=partner_dir / "frozen_inputs" / "ACCEPTANCE.json",
        model_path_override=args.model_path,
    )
    gpu = PARTNER.PILOT._gpu()
    capture_path = source_dir / "calibration_captures.pt"
    if sha256_file(capture_path) != config["source"]["calibration_captures_sha256"]:
        raise ReplayError("calibration capture hash differs from frozen source")
    captures = torch.load(capture_path, map_location="cpu", weights_only=False)
    rows = gpu.materialize_routed_rows(captures)
    rows_by_id = {row.row_id: row for row in rows}

    for call in schedule:
        focal = evidence.get(call["focal_row_id"])
        companion = evidence.get(call["companion_row_id"])
        if focal is None or companion is None:
            raise ReplayError("scheduled row lacks frozen M2 evidence")
        if focal.baseline_label != call["focal_baseline_label"]:
            raise ReplayError("focal baseline label differs from frozen evidence")
        if (
            call["companion_baseline_label"] is not None
            and companion.baseline_label != call["companion_baseline_label"]
        ):
            raise ReplayError("companion baseline label differs from frozen evidence")
        call["resolved_companion_baseline_label"] = companion.baseline_label
        if focal.original_slot != int(call["focal_original_slot"]):
            raise ReplayError("focal slot differs from frozen evidence")
        if call["focal_row_id"] not in rows_by_id or call["companion_row_id"] not in rows_by_id:
            raise ReplayError("scheduled row is absent from captures")

    # Persist the runtime-resolved original-companion labels as a separate ledger.
    write_jsonl_no_overwrite(output_dir / "cross_companion_resolved_schedule.jsonl", schedule)
    target_batches, prepared_calls, signatures = _prepare_runtime(
        model=model, rows_by_id=rows_by_id, schedule=schedule
    )
    references = _fresh_m1_references(target_batches, schedule)
    numeric = _execute_metric_replay(prepared_calls, references)
    write_jsonl_no_overwrite(output_dir / "cross_companion_numeric.jsonl", numeric)
    latency = _measure_paired_latency(prepared_calls)
    aggregate = _aggregate_numeric(numeric)
    result = {
        "schema_version": RESULT_SCHEMA,
        "status": "COMPLETE_EXPLORATORY_REPLAY",
        "decision": (
            "UNINTERPRETABLE_REPEAT_INSTABILITY"
            if aggregate["unstable_label_calls"]
            or aggregate["bitwise_unstable_output_calls"]
            else "REPORT_FIXED_SHAPE_METRIC_REPLAY"
        ),
        "paper_result": False,
        "evidence_boundary": (
            "single_rtx5090_reused_run03_calibration_rows_expert_stage_only_"
            "cross_companion_fixed_m2_shape_metric_and_microbenchmark_"
            "not_fresh_evaluation_not_full_layer_not_serving_not_ep"
        ),
        "source": {
            "partner_schedule": str(schedule_path),
            "partner_schedule_sha256": source_digest,
            "calibration_captures_sha256": sha256_file(capture_path),
            "stack_digest": acceptance["stack"]["stack_digest"],
        },
        "selection": {
            "layers": EXPECTED_LAYERS,
            "focals": len(target_batches),
            "calls": len(schedule),
            "schedule_sha256": sha256_file(output_dir / "cross_companion_schedule.jsonl"),
            "resolved_schedule_sha256": sha256_file(output_dir / "cross_companion_resolved_schedule.jsonl"),
        },
        "fixed_shape_signatures": signatures,
        "metric_replay": aggregate,
        "paired_aggregate_latency": latency,
        "execution": {"warmups": WARMUPS, "repeats": REPEATS},
        "model_config_sha256": canonical_sha256(original_config),
    }
    write_json_no_overwrite(output_dir / "CROSS_COMPANION_RESULT.json", result)
    complete = {
        "schema_version": COMPLETE_SCHEMA,
        "status": "SUCCESS_COMPLETE",
        "artifact_sha256": {
            name: sha256_file(output_dir / name)
            for name in (
                "cross_companion_schedule.jsonl",
                "cross_companion_resolved_schedule.jsonl",
                "cross_companion_numeric.jsonl",
                "CROSS_COMPANION_RESULT.json",
            )
        },
    }
    write_json_no_overwrite(output_dir / "COMPLETE.json", complete)
    PARTNER.PILOT.assert_clean_gpu(
        acceptance["stack"]["gpu"]["uuid"], allowed_pids={os.getpid()}
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan = subparsers.add_parser("plan", help="write the deterministic CPU-only schedule")
    plan.add_argument("--partner-dir", default=str(DEFAULT_PARTNER_DIR))
    plan.add_argument("--output", required=True)
    plan.set_defaults(func=run_plan)

    run = subparsers.add_parser("run", help="execute the replay on the accepted RTX 5090 stack")
    run.add_argument("--partner-dir", default=str(DEFAULT_PARTNER_DIR))
    run.add_argument("--source-dir", default=str(DEFAULT_SOURCE_DIR))
    run.add_argument("--output-dir", required=True)
    run.add_argument("--model-path")
    run.set_defaults(func=run_gpu)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
