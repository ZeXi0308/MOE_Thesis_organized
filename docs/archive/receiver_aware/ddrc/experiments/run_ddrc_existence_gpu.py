#!/usr/bin/env python3
"""DDRC Phase-3 trace/accounting runner with a Phase-4 formal gate.

The executable evaluates frozen route matrices and action-matched quality rows.
It does not claim that its analytic link model or host-staging codec timings are
NCCL/RDMA measurements.  ``--mode dev`` always emits ``NOT_TESTED``.  Formal
mode opens inputs only after validating a Phase-4 ``SIGNED-OFF`` attestation
against protocol, config, source, and data-manifest hashes.

Expected trace JSONL record::

  {
    "trace_id": "...", "stream_id": "request-or-batch-stream", "model": "olmoe",
    "split": "calibration|sealed|dev",
    "step": 0, "layer": 0, "top_k": 8,
    "valid_origin_tokens": {"0": 1, ...},
    "lane_counts": [{"sender": 0, "receiver": 0, "count": 1}, ...]
  }

Optional quality CSV columns::

  trace_id,model,arm,incremental_accuracy_harm,cvar10_positive_harm,action_signature

Those quality rows must have been generated from the exact same action trace;
formal review is responsible for signing that producer into the source hash.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
from typing import Iterable, Mapping, Sequence

from ddrc_policy import (
    AccountingConfig,
    ActionPlan,
    CreditTiming,
    FormatTiming,
    Lane,
    LaneMatrix,
    PolicyState,
    Topology,
    account_step,
    combine_sender_local_plans,
    deterministic_origin_lpt,
    high_lane_bytes,
    make_receiver_resource_views,
    make_sender_local_views,
    plan_calib_static,
    plan_causal_prev_step,
    plan_ddrc,
    plan_global_oracle,
    plan_sender_local,
    plan_uniform,
)


HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE.parent / "configs" / "ddrc_v1.json"
DEFAULT_PROTOCOL = HERE.parent / "DDRC_Phase2_冻结实验协议_2026-07-22.md"
SOURCE_FILES = (
    HERE / "ddrc_policy.py",
    HERE / "run_ddrc_existence_gpu.py",
    HERE / "test_ddrc_policy.py",
)

STRONG_BASELINES = (
    "uniform_full",
    "uniform_low",
    "calib_static",
    "causal_prev_step",
    "sender_local_exact_handle",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--mode", choices=("dev", "formal"), default="dev")
    parser.add_argument("--trace-jsonl", type=Path)
    parser.add_argument("--quality-csv", type=Path)
    parser.add_argument("--data-manifest", type=Path)
    parser.add_argument("--review-attestation", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def canonical_json_hash(value: object) -> str:
    return sha256_bytes(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    )


def source_manifest() -> dict[str, object]:
    files: list[dict[str, str]] = []
    aggregate = hashlib.sha256()
    root = next(candidate for candidate in HERE.parents if (candidate / "experiments/shared").is_dir())
    for path in SOURCE_FILES:
        relative = str(path.relative_to(root))
        digest = sha256_file(path)
        files.append({"path": relative, "sha256": digest})
        aggregate.update(relative.encode())
        aggregate.update(b"\0")
        aggregate.update(path.read_bytes())
        aggregate.update(b"\0")
    return {"files": files, "sha256": aggregate.hexdigest()}


def git_state() -> dict[str, object]:
    def run(*args: str) -> str:
        completed = subprocess.run(
            args,
            cwd=HERE,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return completed.stdout.strip() if completed.returncode == 0 else "UNAVAILABLE"

    return {
        "commit": run("git", "rev-parse", "HEAD"),
        "status_porcelain": run("git", "status", "--short"),
    }


def load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def require_mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be an object")
    return value


def build_topology(raw: Mapping[str, object]) -> Topology:
    topology = require_mapping(raw["topology"], "topology")
    ep_size = int(topology["ep_size"])
    node_by_rank = tuple(int(value) for value in topology["node_by_rank"])
    resources = tuple(str(value) for value in topology["receive_resource_by_rank"])
    return Topology(ep_size, node_by_rank, resources, float(topology["link_gbps"]))


def _format_timing(raw: Mapping[str, object]) -> FormatTiming:
    return FormatTiming(
        pack_us=float(raw["pack_us"]),
        unpack_us=float(raw["unpack_us"]),
        h2d_us=float(raw["h2d_us"]),
        measured_rows=int(raw["measured_rows"]),
        source=str(raw["source"]),
    )


def build_accounting(raw: Mapping[str, object], model: str) -> AccountingConfig:
    model_raw = require_mapping(require_mapping(raw["models"], "models")[model], f"models.{model}")
    accounting = require_mapping(raw["accounting"], "accounting")
    credit = require_mapping(raw["credit_timing"], "credit_timing")
    return AccountingConfig(
        hidden_dim=int(model_raw["hidden_dim"]),
        high_bits=int(accounting["high_bits"]),
        low_bits=int(accounting["low_bits"]),
        high_scale_bytes_per_row=int(accounting["high_scale_bytes_per_row"]),
        low_scale_bytes_per_row=int(accounting["low_scale_bytes_per_row"]),
        lane_descriptor_bytes=int(accounting["lane_descriptor_bytes"]),
        lane_alignment_bytes=int(accounting["lane_alignment_bytes"]),
        codec_tile_rows=int(accounting["codec_tile_rows"]),
        codec_tax_mode=str(accounting["codec_tax_mode"]),
        high_timing=_format_timing(require_mapping(model_raw["high_timing"], "high_timing")),
        low_timing=_format_timing(require_mapping(model_raw["low_timing"], "low_timing")),
        credit_timing=CreditTiming(
            build_us=float(credit["build_us"]),
            aggregate_us=float(credit["aggregate_us"]),
            transfer_us=float(credit["transfer_us"]),
            parse_us=float(credit["parse_us"]),
            pack_deadline_slack_us=float(credit["pack_deadline_slack_us"]),
            overlap_proven=bool(credit["overlap_proven"]),
            source=str(credit["source"]),
        ),
        credit_header_bytes=int(accounting["credit_header_bytes"]),
        credit_record_bytes=int(accounting["credit_record_bytes"]),
        credit_alignment_bytes=int(accounting["credit_alignment_bytes"]),
        evidence_boundary=str(accounting["evidence_boundary"]),
    )


def _parse_lane_counts(raw: object) -> dict[Lane, int]:
    if not isinstance(raw, list):
        raise TypeError("lane_counts must be a list")
    counts: dict[Lane, int] = {}
    for index, item in enumerate(raw):
        row = require_mapping(item, f"lane_counts[{index}]")
        lane = int(row["sender"]), int(row["receiver"])
        if lane in counts:
            raise ValueError(f"duplicate lane in trace: {lane}")
        counts[lane] = int(row["count"])
    return counts


def parse_trace_record(
    raw: Mapping[str, object], topology: Topology, *, formal: bool
) -> tuple[str, LaneMatrix]:
    split = str(raw["split"])
    if split not in {"calibration", "sealed", "dev"}:
        raise ValueError(f"invalid split: {split}")
    stream_id = str(raw.get("stream_id", ""))
    if formal and not stream_id:
        raise ValueError("formal trace requires a non-empty stream_id")
    matrix = LaneMatrix(
        lane_counts=_parse_lane_counts(raw["lane_counts"]),
        valid_origin_tokens={int(key): int(value) for key, value in require_mapping(
            raw["valid_origin_tokens"], "valid_origin_tokens"
        ).items()},
        top_k=int(raw["top_k"]),
        step=int(raw["step"]),
        layer=int(raw["layer"]),
        trace_id=str(raw["trace_id"]),
        stream_id=stream_id,
        dropped_pairs_by_receiver={
            int(key): int(value) for key, value in require_mapping(
                raw.get("dropped_pairs_by_receiver", {}), "dropped_pairs_by_receiver"
            ).items()
        },
    )
    matrix.validate_closure(topology)
    if formal:
        if raw.get("origin_balancing") != "scheduler_visible_token_count_lpt":
            raise ValueError("formal trace lacks frozen request-origin balancing label")
        weights = {
            str(key): int(value) for key, value in require_mapping(
                raw.get("request_weights", {}), "request_weights"
            ).items()
        }
        assignment = {
            str(key): int(value) for key, value in require_mapping(
                raw.get("request_assignment", {}), "request_assignment"
            ).items()
        }
        if not weights or set(weights) != set(assignment):
            raise ValueError("formal trace requires matching request weights/assignment")
        expected_assignment = deterministic_origin_lpt(weights, topology.ep_size)
        if assignment != expected_assignment:
            raise ValueError("formal request assignment is not deterministic origin LPT")
        expected_tokens = {rank: 0 for rank in range(topology.ep_size)}
        for request_id, weight in weights.items():
            expected_tokens[assignment[request_id]] += weight
        observed_tokens = {
            rank: int(matrix.valid_origin_tokens.get(rank, 0))
            for rank in range(topology.ep_size)
        }
        if observed_tokens != expected_tokens:
            raise ValueError(
                "formal valid_origin_tokens do not close against request LPT weights"
            )
    return split, matrix


def built_in_dev_traces(config: Mapping[str, object], topology: Topology) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    models = require_mapping(config["models"], "models")
    for model, model_raw_obj in models.items():
        model_raw = require_mapping(model_raw_obj, f"models.{model}")
        top_k = int(model_raw["top_k"])
        for split_index, split in enumerate(("calibration", "dev")):
            for sample in range(2):
                lane_counts: list[dict[str, int]] = []
                valid = {str(rank): 1 for rank in range(topology.ep_size)}
                for receiver in range(topology.ep_size):
                    for index in range(top_k):
                        sender = (receiver + index + split_index * (sample + 1)) % topology.ep_size
                        lane_counts.append({"sender": sender, "receiver": receiver, "count": 1})
                merged: dict[Lane, int] = {}
                for row in lane_counts:
                    lane = row["sender"], row["receiver"]
                    merged[lane] = merged.get(lane, 0) + row["count"]
                records.append({
                    "trace_id": f"builtin-{model}-{split}-{sample}",
                    "stream_id": f"builtin-{model}-{split}",
                    "model": model,
                    "split": split,
                    "step": sample,
                    "layer": split_index,
                    "top_k": top_k,
                    "valid_origin_tokens": valid,
                    "lane_counts": [
                        {"sender": sender, "receiver": receiver, "count": count}
                        for (sender, receiver), count in sorted(merged.items())
                    ],
                })
    return records


def load_traces(
    path: Path | None,
    config: Mapping[str, object],
    topology: Topology,
    *,
    formal: bool,
) -> list[tuple[str, str, LaneMatrix]]:
    if path is None:
        raw_rows = built_in_dev_traces(config, topology)
    else:
        raw_rows = [
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
        ]
    parsed: list[tuple[str, str, LaneMatrix]] = []
    seen: set[str] = set()
    last_position_by_stream: dict[tuple[str, str, str], tuple[int, int]] = {}
    for index, raw_obj in enumerate(raw_rows):
        raw = require_mapping(raw_obj, f"trace[{index}]")
        model = str(raw["model"])
        models_cfg = require_mapping(config["models"], "models")
        if model not in models_cfg:
            raise ValueError(f"unknown trace model: {model}")
        split, matrix = parse_trace_record(raw, topology, formal=formal)
        if formal:
            model_cfg = require_mapping(models_cfg[model], f"models.{model}")
            expected_top_k = int(model_cfg["top_k"])
            if matrix.top_k != expected_top_k:
                raise ValueError(
                    f"formal trace top_k mismatch for {model}: "
                    f"observed={matrix.top_k} expected={expected_top_k}"
                )
            if split == "sealed" and matrix.dropped_pairs_by_receiver:
                raise ValueError(
                    "formal main sealed cell requires dropped_pairs_by_receiver to be empty"
                )
            stream_key = (model, split, matrix.stream_id)
            position = (matrix.step, matrix.layer)
            previous_position = last_position_by_stream.get(stream_key)
            if previous_position is not None and position <= previous_position:
                raise ValueError(
                    "formal trace is not strictly monotonic within stream: "
                    f"key={stream_key} previous={previous_position} current={position}"
                )
            last_position_by_stream[stream_key] = position
        if matrix.trace_id in seen:
            raise ValueError(f"duplicate trace_id: {matrix.trace_id}")
        seen.add(matrix.trace_id)
        parsed.append((model, split, matrix))
    if not parsed:
        raise ValueError("trace input is empty")
    return parsed


def quantile(values: Sequence[float], q: float) -> float:
    if not values:
        raise ValueError("quantile input is empty")
    if not 0.0 <= q <= 1.0:
        raise ValueError("quantile must be in [0, 1]")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def calibrate(
    traces: Sequence[tuple[str, str, LaneMatrix]],
    config: Mapping[str, object],
    topology: Topology,
) -> dict[str, dict[str, object]]:
    q = float(require_mapping(config["selection"], "selection")["threshold_quantile"])
    result: dict[str, dict[str, object]] = {}
    observed_models = sorted({model for model, _split, _matrix in traces})
    for model in observed_models:
        cfg = build_accounting(config, model)
        calibration = [matrix for trace_model, split, matrix in traces if trace_model == model and split == "calibration"]
        if not calibration:
            raise ValueError(f"model {model} has no calibration traces")
        resource_values: dict[str, list[int]] = {
            resource: [] for resource in set(topology.receive_resource_by_rank)
        }
        sender_values: dict[int, list[int]] = {rank: [] for rank in range(topology.ep_size)}
        lane_frequency: dict[Lane, int] = {}
        for matrix in calibration:
            receiver_views = make_receiver_resource_views(matrix, topology)
            sender_views = make_sender_local_views(matrix, topology)
            for resource, view in receiver_views.items():
                resource_values[resource].append(
                    sum(high_lane_bytes(rows, cfg).wire_bytes for rows in view.lane_counts.values())
                )
            for sender, view in sender_views.items():
                sender_values[sender].append(
                    sum(high_lane_bytes(rows, cfg).wire_bytes for rows in view.lane_counts.values())
                )
                for lane in view.lane_counts:
                    lane_frequency[lane] = lane_frequency.get(lane, 0) + 1
        frequency_threshold = quantile(list(lane_frequency.values()) or [math.inf], q)
        result[model] = {
            "receiver_threshold_bytes": {
                resource: int(math.ceil(quantile(values, q)))
                for resource, values in resource_values.items()
            },
            "sender_threshold_bytes": {
                str(sender): int(math.ceil(quantile(values, q)))
                for sender, values in sender_values.items()
            },
            "static_low_lanes": [
                [sender, receiver]
                for (sender, receiver), count in sorted(lane_frequency.items())
                if count >= frequency_threshold
            ],
            "threshold_quantile": q,
            "n_calibration_traces": len(calibration),
        }
    return result


def evaluate_traces(
    traces: Sequence[tuple[str, str, LaneMatrix]],
    calibration: Mapping[str, Mapping[str, object]],
    config: Mapping[str, object],
    topology: Topology,
    formal: bool,
    codec_sensitivity_rows: list[dict[str, object]] | None = None,
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    previous_by_stream_layer: dict[tuple[str, str, int], LaneMatrix] = {}
    states_by_stream_layer: dict[tuple[str, str, int], dict[int, PolicyState]] = {}
    allowed_split = "sealed" if formal else "dev"
    for model, split, matrix in traces:
        if split != allowed_split:
            continue
        cfg = build_accounting(config, model)
        cal = calibration[model]
        receiver_thresholds = {
            str(key): int(value) for key, value in require_mapping(
                cal["receiver_threshold_bytes"], "receiver_threshold_bytes"
            ).items()
        }
        sender_thresholds = {
            int(key): int(value) for key, value in require_mapping(
                cal["sender_threshold_bytes"], "sender_threshold_bytes"
            ).items()
        }
        static_lanes = {
            (int(item[0]), int(item[1])) for item in cal["static_low_lanes"]
        }
        sender_views = make_sender_local_views(matrix, topology)
        sender_plan = combine_sender_local_plans(
            plan_sender_local(
                view,
                threshold_bytes=sender_thresholds[sender],
                cfg=cfg,
                topology=topology,
            )
            for sender, view in sender_views.items()
        )
        stream_layer_key = (model, matrix.stream_id, matrix.layer)
        states = states_by_stream_layer.setdefault(
            stream_layer_key, {rank: PolicyState() for rank in range(topology.ep_size)}
        )
        ddrc_plan = plan_ddrc(
            matrix,
            receiver_threshold_bytes=receiver_thresholds,
            cfg=cfg,
            topology=topology,
            states=states,
        )
        plans = [
            plan_uniform("uniform_full", matrix, topology),
            plan_uniform("uniform_low", matrix, topology),
            plan_calib_static(
                matrix, static_low_lanes=static_lanes, cfg=cfg, topology=topology
            ),
            plan_causal_prev_step(
                matrix,
                previous_by_stream_layer.get(stream_layer_key),
                receiver_threshold_bytes=receiver_thresholds,
                cfg=cfg,
                topology=topology,
            ),
            sender_plan,
            ddrc_plan,
        ]
        plans.append(
            plan_global_oracle(
                matrix,
                cfg=cfg,
                topology=topology,
                deployable_seed_plans=(sender_plan, ddrc_plan),
            )
        )
        ledgers = [account_step(matrix, plan, cfg=cfg, topology=topology) for plan in plans]
        if len(plans) != len(ledgers):
            raise AssertionError("plan/accounting cardinality mismatch")
        oracle = next(ledger for ledger in ledgers if ledger.arm == "global_full_matrix_oracle")
        for plan, ledger in zip(plans, ledgers):
            if ledger.arm in {"DDRC", "sender_local_exact_handle"} and oracle.total_us > ledger.total_us + 1e-9:
                raise RuntimeError("global oracle is worse than a deployable arm")
            row = ledger.to_dict()
            row.update({
                "trace_id": matrix.trace_id,
                "stream_id": matrix.stream_id,
                "model": model,
                "split": split,
                "step": matrix.step,
                "layer": matrix.layer,
                "top_k": matrix.top_k,
                "codec_tax_mode": cfg.codec_tax_mode,
            })
            row["action_signature"] = canonical_json_hash({
                "trace_id": matrix.trace_id,
                "model": model,
                "arm": ledger.arm,
                "low_lanes": row["low_lanes"],
                "requested_lanes": row["requested_lanes"],
                "blocked_lanes": row["blocked_lanes"],
                "fallback_reason": row["fallback_reason"],
            })
            output.append(row)
            if codec_sensitivity_rows is not None:
                accounting_cfg = require_mapping(config["accounting"], "accounting")
                sensitivity_modes = accounting_cfg.get("sensitivity_codec_tax_modes", ())
                if not isinstance(sensitivity_modes, list):
                    raise TypeError("accounting.sensitivity_codec_tax_modes must be a list")
                for sensitivity_mode_raw in sensitivity_modes:
                    sensitivity_mode = str(sensitivity_mode_raw)
                    if sensitivity_mode == cfg.codec_tax_mode:
                        raise ValueError("codec sensitivity mode duplicates the primary mode")
                    sensitivity_cfg = replace(cfg, codec_tax_mode=sensitivity_mode)
                    sensitivity_ledger = account_step(
                        matrix, plan, cfg=sensitivity_cfg, topology=topology
                    )
                    sensitivity_row = sensitivity_ledger.to_dict()
                    sensitivity_row.update({
                        "trace_id": matrix.trace_id,
                        "stream_id": matrix.stream_id,
                        "model": model,
                        "split": split,
                        "step": matrix.step,
                        "layer": matrix.layer,
                        "top_k": matrix.top_k,
                        "codec_tax_mode": sensitivity_mode,
                        "action_signature": row["action_signature"],
                        "action_selection_codec_tax_mode": cfg.codec_tax_mode,
                    })
                    codec_sensitivity_rows.append(sensitivity_row)
        previous_by_stream_layer[stream_layer_key] = matrix
    if not output:
        raise ValueError(f"no {allowed_split} trace rows to evaluate")
    return output


def validate_same_action_codec_accounting(
    primary_rows: Sequence[Mapping[str, object]],
    sensitivity_rows: Sequence[Mapping[str, object]],
    config: Mapping[str, object],
) -> dict[str, object]:
    accounting_cfg = require_mapping(config["accounting"], "accounting")
    primary_mode = str(accounting_cfg["codec_tax_mode"])
    modes_raw = accounting_cfg.get("sensitivity_codec_tax_modes", ())
    if not isinstance(modes_raw, list):
        raise TypeError("accounting.sensitivity_codec_tax_modes must be a list")
    sensitivity_modes = [str(mode) for mode in modes_raw]
    primary = {
        (str(row["trace_id"]), str(row["model"]), str(row["arm"])): str(
            row["action_signature"]
        )
        for row in primary_rows
    }
    observed: dict[tuple[str, str, str, str], str] = {}
    for row in sensitivity_rows:
        key = (
            str(row["trace_id"]),
            str(row["model"]),
            str(row["arm"]),
            str(row["codec_tax_mode"]),
        )
        if key in observed:
            raise ValueError(f"duplicate codec sensitivity row: {key}")
        observed[key] = str(row["action_signature"])
    expected_keys = {
        (*key, mode) for key in primary for mode in sensitivity_modes
    }
    if set(observed) != expected_keys:
        raise ValueError("codec sensitivity rows do not close against the primary action trace")
    for key, signature in observed.items():
        if signature != primary[key[:3]]:
            raise ValueError(f"codec sensitivity action mismatch: {key}")
    return {
        "status": "ACCOUNTING_ONLY_NO_SCIENTIFIC_VERDICT",
        "primary_mode": primary_mode,
        "sensitivity_modes": sensitivity_modes,
        "same_action_signature": True,
        "primary_action_rows": len(primary_rows),
        "sensitivity_accounting_rows": len(sensitivity_rows),
        "unclosed_gate": "paired block-bootstrap and formal dual-codec decision gate",
    }


def load_quality(path: Path | None) -> list[dict[str, object]]:
    if path is None:
        return []
    required = {
        "trace_id",
        "model",
        "arm",
        "incremental_accuracy_harm",
        "cvar10_positive_harm",
        "action_signature",
    }
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"quality CSV missing columns: {sorted(required - set(reader.fieldnames or []))}")
        rows: list[dict[str, object]] = []
        for row in reader:
            rows.append({
                "trace_id": row["trace_id"],
                "model": row["model"],
                "arm": row["arm"],
                "incremental_accuracy_harm": float(row["incremental_accuracy_harm"]),
                "cvar10_positive_harm": float(row["cvar10_positive_harm"]),
                "action_signature": row["action_signature"],
            })
    return rows


def percentile(values: Sequence[float], q: float) -> float:
    return quantile(values, q)


def summarize_system(accounting_rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str], list[Mapping[str, object]]] = {}
    for row in accounting_rows:
        groups.setdefault((str(row["model"]), str(row["arm"])), []).append(row)
    summaries: list[dict[str, object]] = []
    p99_by_model_arm: dict[tuple[str, str], float] = {}
    for (model, arm), rows in sorted(groups.items()):
        times = [float(row["total_us"]) for row in rows]
        p99 = percentile(times, 0.99)
        p99_by_model_arm[(model, arm)] = p99
        summaries.append({
            "model": model,
            "arm": arm,
            "n_trace_rows": len(rows),
            "p50_critical_path_proxy_us": percentile(times, 0.50),
            "p95_critical_path_proxy_us": percentile(times, 0.95),
            "p99_critical_path_proxy_us": p99,
            "mean_net_saving_fraction": sum(
                float(row["net_saving_fraction"]) for row in rows
            ) / len(rows),
            "mean_credit_bytes": sum(float(row["credit_bytes"]) for row in rows) / len(rows),
            "credit_miss_or_fallback_rate": sum(
                bool(row["fallback_reason"]) for row in rows
            ) / len(rows),
            "evidence_boundary": "analytic critical-path proxy; NOT measured NCCL/RDMA P99",
        })
    for row in summaries:
        if row["arm"] != "DDRC":
            row["p99_gain_vs_best_strong_baseline"] = ""
            continue
        model = str(row["model"])
        best = min(p99_by_model_arm[(model, arm)] for arm in STRONG_BASELINES)
        row["p99_gain_vs_best_strong_baseline"] = (
            (best - float(row["p99_critical_path_proxy_us"])) / best if best > 0 else 0.0
        )
    return summaries


def paired_bootstrap(
    accounting_rows: Sequence[Mapping[str, object]],
    quality_rows: Sequence[Mapping[str, object]],
    *,
    lambdas: Sequence[float],
    repeats: int,
    seed: int,
) -> dict[str, object]:
    if not quality_rows:
        return {"status": "NOT_RUN", "reason": "quality CSV not supplied", "rows": []}
    accounting = {
        (str(row["trace_id"]), str(row["model"]), str(row["arm"])): {
            "saving": float(row["net_saving_fraction"]),
            "action_signature": str(row["action_signature"]),
        }
        for row in accounting_rows
    }
    quality = {
        (str(row["trace_id"]), str(row["model"]), str(row["arm"])): {
            "harm": float(row["incremental_accuracy_harm"]),
            "cvar": float(row["cvar10_positive_harm"]),
            "action_signature": str(row["action_signature"]),
        }
        for row in quality_rows
    }
    models = sorted({str(row["model"]) for row in accounting_rows})
    rng = random.Random(seed)
    output: list[dict[str, object]] = []
    for model in models:
        trace_ids = sorted({key[0] for key in accounting if key[1] == model})
        arms = sorted({key[2] for key in accounting if key[1] == model})
        required_arms = set(STRONG_BASELINES) | {"DDRC"}
        if not required_arms.issubset(arms):
            raise ValueError(f"model {model} missing required accounting arms")
        missing = [
            key for trace_id in trace_ids for arm in required_arms
            if (key := (trace_id, model, arm)) not in quality
        ]
        if missing:
            raise ValueError(f"quality rows do not match action traces; first missing={missing[0]}")
        mismatched = [
            key for key in quality
            if key in accounting
            and quality[key]["action_signature"] != accounting[key]["action_signature"]
        ]
        if mismatched:
            raise ValueError(
                f"quality/action signature mismatch; first mismatch={mismatched[0]}"
            )
        for lam in lambdas:
            def statistics(sampled: Sequence[str]) -> tuple[float, float, float, str]:
                utility: dict[str, float] = {}
                for arm in required_arms:
                    values = [
                        accounting[(trace_id, model, arm)]["saving"]
                        - lam * quality[(trace_id, model, arm)]["harm"]
                        for trace_id in sampled
                    ]
                    utility[arm] = sum(values) / len(values)
                best_baseline = min(
                    STRONG_BASELINES,
                    key=lambda arm: (-utility[arm], arm),
                )
                accuracy_delta = sum(
                    quality[(trace_id, model, "DDRC")]["harm"]
                    - quality[(trace_id, model, best_baseline)]["harm"]
                    for trace_id in sampled
                ) / len(sampled)
                cvar_delta = sum(
                    quality[(trace_id, model, "DDRC")]["cvar"]
                    - quality[(trace_id, model, best_baseline)]["cvar"]
                    for trace_id in sampled
                ) / len(sampled)
                return (
                    utility["DDRC"] - utility[best_baseline],
                    accuracy_delta,
                    cvar_delta,
                    best_baseline,
                )

            point, accuracy_delta, cvar_delta, best_baseline = statistics(trace_ids)
            samples = [
                statistics([trace_ids[rng.randrange(len(trace_ids))] for _ in trace_ids])
                for _ in range(repeats)
            ]
            output.append({
                "model": model,
                "lambda": lam,
                "margin": point,
                "ci_low": percentile([sample[0] for sample in samples], 0.025),
                "ci_high": percentile([sample[0] for sample in samples], 0.975),
                "matched_best_baseline": best_baseline,
                "accuracy_harm_delta_vs_matched_baseline": accuracy_delta,
                "accuracy_harm_delta_ci_high": percentile(
                    [sample[1] for sample in samples], 0.975
                ),
                "cvar_harm_delta_vs_matched_baseline": cvar_delta,
                "cvar_harm_delta_ci_high": percentile(
                    [sample[2] for sample in samples], 0.975
                ),
            })
    return {"status": "COMPLETE", "repeats": repeats, "rows": output}


def build_decision(
    mode: str,
    accounting_rows: Sequence[Mapping[str, object]],
    bootstrap: Mapping[str, object],
    config: Mapping[str, object],
) -> dict[str, object]:
    if mode != "formal":
        return {
            "status": "NOT_TESTED",
            "go": False,
            "reason": "dev mode cannot emit a scientific verdict",
        }
    decision_cfg = require_mapping(config["decision"], "decision")
    required_models_raw = decision_cfg.get("required_models", ())
    if not isinstance(required_models_raw, list) or not required_models_raw:
        raise ValueError("decision.required_models must be a non-empty list")
    required_models = [str(model) for model in required_models_raw]
    if len(required_models) != len(set(required_models)):
        raise ValueError("decision.required_models contains duplicates")
    observed_models = sorted({str(row["model"]) for row in accounting_rows})
    if set(observed_models) != set(required_models):
        return {
            "status": "PARTIAL",
            "go": False,
            "reason": "formal observed models do not exactly match required_models",
            "required_models": sorted(required_models),
            "observed_models": observed_models,
            "missing_models": sorted(set(required_models) - set(observed_models)),
            "unexpected_models": sorted(set(observed_models) - set(required_models)),
            "evidence_boundary": "no scientific GO/NO-GO may be emitted",
        }
    if bootstrap.get("status") != "COMPLETE":
        return {"status": "PARTIAL", "go": False, "reason": "quality/bootstrap missing"}

    formal_cfg = require_mapping(config["formal"], "formal")
    capabilities = require_mapping(formal_cfg.get("capabilities", {}), "formal.capabilities")
    missing_capabilities = sorted(
        key for key in (
            "native_route_capture",
            "native_action_matched_gpu_quality",
            "measured_credit_timing",
            "burst_block_bootstrap",
        )
        if capabilities.get(key) is not True
    )
    if missing_capabilities:
        return {
            "status": "PARTIAL",
            "go": False,
            "reason": "formal path is fail-closed until all frozen capabilities are implemented",
            "missing_capabilities": missing_capabilities,
            "evidence_boundary": "no scientific GO/NO-GO may be emitted",
        }

    rows = list(bootstrap["rows"])
    models = observed_models
    threshold = float(decision_cfg["utility_margin_pp"])
    lambdas = [float(value) for value in decision_cfg["lambdas"]]
    g0_by_model: dict[str, bool] = {}
    oracle_ceiling: dict[str, float] = {}
    for model in models:
        model_rows = [row for row in accounting_rows if row["model"] == model]
        by_trace: dict[str, dict[str, Mapping[str, object]]] = {}
        for row in model_rows:
            by_trace.setdefault(str(row["trace_id"]), {})[str(row["arm"])] = row
        g0_by_model[model] = any(
            trace["DDRC"]["requested_lanes"] != trace["sender_local_exact_handle"]["requested_lanes"]
            for trace in by_trace.values()
        )
        gaps = [
            float(trace["global_full_matrix_oracle"]["net_saving_fraction"])
            - float(trace["sender_local_exact_handle"]["net_saving_fraction"])
            for trace in by_trace.values()
        ]
        oracle_ceiling[model] = sum(gaps) / len(gaps)

    adjacent_by_model: dict[str, bool] = {}
    for model in models:
        passed = {
            float(row["lambda"])
            for row in rows
            if row["model"] == model
            and float(row["margin"]) >= threshold
            and float(row["ci_low"]) > 0.0
            and float(row["accuracy_harm_delta_ci_high"]) <= 0.0
            and float(row["cvar_harm_delta_ci_high"]) <= 0.0
        }
        adjacent_by_model[model] = any(
            lambdas[index] in passed and lambdas[index + 1] in passed
            for index in range(len(lambdas) - 1)
        )

    timing_sources = {
        str(require_mapping(config["credit_timing"], "credit_timing")["source"])
    }
    for model_raw in require_mapping(config["models"], "models").values():
        model_mapping = require_mapping(model_raw, "model")
        timing_sources.add(str(require_mapping(model_mapping["high_timing"], "high_timing")["source"]))
        timing_sources.add(str(require_mapping(model_mapping["low_timing"], "low_timing")["source"]))
    timing_verified = "assumed" not in timing_sources
    codec_mode_ok = (
        str(require_mapping(config["accounting"], "accounting")["codec_tax_mode"])
        == "serialized_tiles"
    )
    gates = {
        "g0_information_non_degenerate_both_models": all(g0_by_model.values()),
        "g1_oracle_ceiling_ge_3pp_both_models": all(
            value >= threshold for value in oracle_ceiling.values()
        ),
        "adjacent_lambda_margin_both_models": all(adjacent_by_model.values()),
        "serialized_tiles_primary": codec_mode_ok,
        "no_assumed_timing": timing_verified,
    }
    if not timing_verified:
        return {
            "status": "PARTIAL",
            "go": False,
            "reason": "formal timing contains assumed values",
            "gates": gates,
            "evidence_boundary": "no scientific GO/NO-GO may be emitted",
        }
    go = all(gates.values())
    return {
        "status": "GO_TO_REAL_TRANSPORT_NEXT_ITERATION" if go else "NO_GO",
        "go": go,
        "gates": gates,
        "g0_by_model": g0_by_model,
        "oracle_ceiling_by_model": oracle_ceiling,
        "adjacent_lambda_by_model": adjacent_by_model,
        "threshold": threshold,
        "evidence_boundary": "route-real/topology-proxy only; NOT NCCL/RDMA/P99",
    }


def build_status(
    mode: str,
    decision: Mapping[str, object],
    review: Mapping[str, object],
) -> dict[str, object]:
    decision_status = str(decision["status"])
    verdict_statuses = {"GO_TO_REAL_TRANSPORT_NEXT_ITERATION", "NO_GO"}
    scientific_verdict = decision_status if decision_status in verdict_statuses else None
    formal_run_valid = bool(
        mode == "formal"
        and review.get("status") == "SIGNED-OFF"
        and scientific_verdict is not None
    )
    go = bool(
        formal_run_valid
        and scientific_verdict == "GO_TO_REAL_TRANSPORT_NEXT_ITERATION"
        and decision.get("go") is True
    )
    return {
        "mode": mode,
        "status": decision_status,
        "review": dict(review),
        "formal_run_valid": formal_run_valid,
        "scientific_verdict": scientific_verdict,
        "go": go,
    }


def validate_formal_gate(
    args: argparse.Namespace,
    config: Mapping[str, object],
    manifest: Mapping[str, object],
) -> dict[str, object]:
    if args.review_attestation is None or args.data_manifest is None:
        raise RuntimeError("formal mode requires --review-attestation and --data-manifest")
    attestation = require_mapping(load_json(args.review_attestation), "review attestation")
    if attestation.get("status") != "SIGNED-OFF":
        raise RuntimeError("formal run blocked: Phase-4 status is not SIGNED-OFF")
    expected = {
        "protocol_sha256": sha256_file(args.protocol),
        "config_sha256": canonical_json_hash(config),
        "source_sha256": str(manifest["sha256"]),
        "data_manifest_sha256": sha256_file(args.data_manifest),
    }
    for key, value in expected.items():
        if attestation.get(key) != value:
            raise RuntimeError(
                f"formal run blocked: {key} mismatch; expected {value}, got {attestation.get(key)}"
            )
    return dict(attestation)


def validate_formal_inputs(args: argparse.Namespace, data_manifest: Mapping[str, object]) -> None:
    expected = {
        "trace_jsonl_sha256": sha256_file(args.trace_jsonl),
        "quality_csv_sha256": sha256_file(args.quality_csv),
    }
    for key, value in expected.items():
        if data_manifest.get(key) != value:
            raise RuntimeError(
                f"formal input blocked: {key} mismatch; expected {value}, "
                f"manifest has {data_manifest.get(key)}"
            )
    if data_manifest.get("sealed") is not True:
        raise RuntimeError("formal input blocked: data manifest is not sealed")


def write_csv(path: Path, rows: Sequence[Mapping[str, object]], fieldnames: Sequence[str] | None = None) -> None:
    if rows:
        names = list(fieldnames or rows[0].keys())
    elif fieldnames is not None:
        names = list(fieldnames)
    else:
        raise ValueError("empty CSV requires explicit fieldnames")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=names, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if not args.config.exists() or not args.protocol.exists():
        raise FileNotFoundError("config/protocol path does not exist")
    config = require_mapping(load_json(args.config), "config")
    if str(require_mapping(config["accounting"], "accounting")["codec_tax_mode"]) != "serialized_tiles":
        raise RuntimeError("frozen main configuration must use serialized_tiles")
    topology = build_topology(config)
    manifest = source_manifest()

    if args.mode == "formal":
        if args.trace_jsonl is None or args.quality_csv is None:
            raise RuntimeError("formal mode requires --trace-jsonl and --quality-csv")
        if args.output_dir.exists() and any(args.output_dir.iterdir()):
            raise RuntimeError("formal output directory must be new or empty")
        review = validate_formal_gate(args, config, manifest)
        signed_data_manifest = require_mapping(load_json(args.data_manifest), "data manifest")
        validate_formal_inputs(args, signed_data_manifest)
    else:
        review = {"status": "NOT_REQUIRED_DEV"}
        signed_data_manifest = None

    # Inputs are deliberately opened only after the formal review gate.
    traces = load_traces(
        args.trace_jsonl, config, topology, formal=args.mode == "formal"
    )
    quality = load_quality(args.quality_csv)
    calibration = calibrate(traces, config, topology)
    codec_sensitivity_rows: list[dict[str, object]] = []
    accounting_rows = evaluate_traces(
        traces,
        calibration,
        config,
        topology,
        formal=args.mode == "formal",
        codec_sensitivity_rows=codec_sensitivity_rows,
    )
    codec_sensitivity_status = validate_same_action_codec_accounting(
        accounting_rows, codec_sensitivity_rows, config
    )
    system_summary = summarize_system(accounting_rows)
    decision_cfg = require_mapping(config["decision"], "decision")
    bootstrap = paired_bootstrap(
        accounting_rows,
        quality,
        lambdas=[float(value) for value in decision_cfg["lambdas"]],
        repeats=int(decision_cfg["n_bootstrap"]),
        seed=int(decision_cfg["seed"]),
    )
    decision = build_decision(args.mode, accounting_rows, bootstrap, config)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    protocol_record = {
        "path": str(args.protocol.resolve()),
        "sha256": sha256_file(args.protocol),
        "status": "PHASE2_FROZEN",
    }
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "pid": os.getpid(),
        "git": git_state(),
        "evidence_boundary": "NOT_RDMA / host-staging and topology proxy",
    }
    if signed_data_manifest is not None:
        data_manifest = signed_data_manifest
    else:
        data_manifest = {
            "status": "DEV_GENERATED_NOT_SEALED",
            "trace_ids": sorted({str(row["trace_id"]) for row in accounting_rows}),
            "trace_input_sha256": sha256_file(args.trace_jsonl) if args.trace_jsonl else "BUILTIN",
        }
    topology_record = {
        **asdict(topology),
        "boundary": "virtual EP topology; no actual NIC/QP measurement",
    }

    (args.output_dir / "protocol.json").write_text(
        json.dumps(protocol_record, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (args.output_dir / "environment.json").write_text(
        json.dumps(environment, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (args.output_dir / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (args.output_dir / "data_manifest.json").write_text(
        json.dumps(data_manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (args.output_dir / "topology.json").write_text(
        json.dumps(topology_record, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (args.output_dir / "calibration.json").write_text(
        json.dumps(calibration, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_csv(args.output_dir / "action_trace.csv", accounting_rows)
    write_csv(args.output_dir / "system_summary.csv", system_summary)
    credit_rows = [
        {key: row[key] for key in (
            "trace_id", "model", "arm", "information_set", "credit_bytes",
            "credit_total_us", "credit_visible_us", "fallback_reason", "state_committed",
        )}
        for row in accounting_rows
    ]
    write_csv(args.output_dir / "credit_accounting.csv", credit_rows)
    codec_rows = [
        {key: row[key] for key in (
            "trace_id", "stream_id", "model", "arm", "codec_tax_mode",
            "action_signature", "payload_bytes", "scale_bytes",
            "descriptor_bytes", "padding_bytes", "wire_bytes", "critical_wire_bytes",
            "wire_us", "pack_us", "h2d_us", "unpack_us", "codec_us", "total_us",
            "evidence_boundary",
        )}
        for row in [*accounting_rows, *codec_sensitivity_rows]
    ]
    write_csv(args.output_dir / "codec_accounting.csv", codec_rows)
    (args.output_dir / "codec_sensitivity.json").write_text(
        json.dumps(codec_sensitivity_status, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_csv(
        args.output_dir / "quality.csv",
        quality,
        fieldnames=(
            "trace_id", "model", "arm", "incremental_accuracy_harm",
            "cvar10_positive_harm", "action_signature",
        ),
    )
    (args.output_dir / "paired_bootstrap.json").write_text(
        json.dumps(bootstrap, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (args.output_dir / "decision.json").write_text(
        json.dumps(decision, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    status = build_status(args.mode, decision, review)
    status["dual_codec_same_action_accounting"] = codec_sensitivity_status["status"]
    (args.output_dir / "status.json").write_text(
        json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    report = "\n".join((
        "# DDRC execution report",
        "",
        f"- Mode: `{args.mode}`",
        f"- Status: `{decision['status']}`",
        f"- Codec tax mode: `{require_mapping(config['accounting'], 'accounting')['codec_tax_mode']}`",
        f"- Trace rows: `{len(accounting_rows)}`",
        "- Evidence boundary: route-real/topology-proxy; `NOT_RDMA`, not real TPOT/P99.",
        "",
        "Phase 3 code generation and dev smoke do not constitute a scientific result.",
    ))
    (args.output_dir / "report.md").write_text(report + "\n", encoding="utf-8")
    print(json.dumps(status, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
