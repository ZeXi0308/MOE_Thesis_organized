#!/usr/bin/env python3
"""Run CPR-MoE single-RTX-5090 necessary-condition falsification gates.

This runner cannot validate EP/NCCL, receiver backlog, request-DAG criticality,
TPOT, or P99.  Passing means only that CPR-MoE was not falsified by the tested
single-GPU necessary condition.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import traceback
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent


def _find_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "experiments/shared").is_dir():
            return candidate
    raise RuntimeError("cannot locate repository root containing experiments/shared")


REPO_ROOT = _find_repo_root(HERE)
QUALITY_COLUMNS = ("sample_id", "rank1_int4__kl", "rankk_int4__kl")
EXPECTED_QUANTIZATION_CONTRACT = {
    "name": "per_row_symmetric_int_v1",
    "scale_dtype": "float32",
    "scale_formula": "clamp_min(absmax,1e-8)/qmax",
    "rounding": "nearest_ties_to_even",
    "zero_point": 0,
    "int4_qmin": -7,
    "int4_qmax": 7,
    "int4_storage": "two_signed_nibbles_per_uint8",
}
EXPECTED_QUALITY_RUNTIME_CONTRACT = {
    "gpu_name_substring": "RTX 5090",
    "compute_capability": [12, 0],
    "torch_version_prefix": "2.8.0",
    "torch_cuda_prefix": "12.8",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--experiment", choices=("quality", "codec", "all"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, help="override only the frozen seed")
    parser.add_argument("--smoke", action="store_true", help="logic smoke only; never scientific")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--allow-other-gpu", action="store_true")
    return parser.parse_args()


def read_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if config.get("schema_version") != "cpr-moe-5090-quick-validate-v1":
        raise ValueError("unsupported or missing config schema_version")
    if not isinstance(config.get("seed"), int):
        raise ValueError("seed must be an integer")
    formal_seeds = config.get("formal_seeds")
    if (
        not isinstance(formal_seeds, list)
        or not formal_seeds
        or not all(isinstance(value, int) for value in formal_seeds)
        or len(set(formal_seeds)) != len(formal_seeds)
    ):
        raise ValueError("formal_seeds must be a non-empty unique integer list")
    if config["seed"] not in formal_seeds:
        raise ValueError("default seed must be included in formal_seeds")
    if config.get("quantization_contract") != EXPECTED_QUANTIZATION_CONTRACT:
        raise ValueError("quantization_contract does not match the implemented codec")
    if not config.get("quality", {}).get("models"):
        raise ValueError("quality.models must be non-empty")
    if not config["quality"].get("producer"):
        raise ValueError("quality.producer must be set")
    if config["quality"].get("runtime_contract") != EXPECTED_QUALITY_RUNTIME_CONTRACT:
        raise ValueError("quality.runtime_contract does not match the frozen RTX 5090 stack")
    quality_models = config["quality"]["models"]
    identity_keys = {
        "model",
        "model_key",
        "model_revision",
        "dataset",
        "split",
        "samples",
        "offset",
        "seq_len",
        "dtype",
        "producer_seed",
    }
    for spec in quality_models:
        for key in ("name", "run_identity", "per_document_csv", "provenance_json"):
            if not spec.get(key):
                raise ValueError(f"quality model entry missing {key}")
        if set(spec["run_identity"]) != identity_keys:
            raise ValueError("quality run_identity fields do not match frozen schema")
        identity = spec["run_identity"]
        if identity["model_key"] != spec["name"]:
            raise ValueError("quality model name must equal run_identity.model_key")
        if (
            not isinstance(identity["model"], str)
            or not identity["model"]
            or not isinstance(identity["dataset"], str)
            or not identity["dataset"]
            or not isinstance(identity["split"], str)
            or not identity["split"]
            or identity["dtype"] != "bfloat16"
            or not isinstance(identity["samples"], int)
            or identity["samples"] < 8
            or not isinstance(identity["offset"], int)
            or identity["offset"] < 0
            or not isinstance(identity["seq_len"], int)
            or identity["seq_len"] <= 0
            or not isinstance(identity["producer_seed"], int)
        ):
            raise ValueError("quality run_identity contains invalid values or types")
        revision = identity["model_revision"]
        if not isinstance(revision, str) or len(revision) != 40 or any(
            char not in "0123456789abcdef" for char in revision.lower()
        ):
            raise ValueError("quality model_revision must be a 40-character commit SHA")
    for field, values in {
        "name": [spec["name"] for spec in quality_models],
        "model": [spec["run_identity"]["model"] for spec in quality_models],
        "per_document_csv": [spec["per_document_csv"] for spec in quality_models],
        "provenance_json": [spec["provenance_json"] for spec in quality_models],
    }.items():
        if len(values) != len(set(values)):
            raise ValueError(f"quality model entries contain duplicate {field}")
    codec = config.get("codec", {})
    for key in ("rows", "hidden_sizes", "modes", "link_gbps"):
        if not codec.get(key):
            raise ValueError(f"codec.{key} must be non-empty")
    if any(int(value) <= 0 for value in codec["rows"] + codec["hidden_sizes"]):
        raise ValueError("codec rows and hidden_sizes must be positive")
    if any(int(value) % 2 for value in codec["hidden_sizes"]):
        raise ValueError("codec hidden_sizes must be even")
    if any(float(value) <= 0 for value in codec["link_gbps"]):
        raise ValueError("codec link_gbps must be positive")
    if not set(codec["modes"]).issubset({"int8", "int4"}):
        raise ValueError("codec modes must be a subset of int8,int4")
    decision_mode = codec.get("decision_mode")
    if decision_mode != "int4" or decision_mode not in codec["modes"]:
        raise ValueError("codec.decision_mode must be int4 and included in codec.modes")
    if config["quality"].get("decision_mode") != decision_mode:
        raise ValueError("quality and codec decision_mode must match")
    return config


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def prepare_output(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"refusing to mix results in non-empty output directory: {path}")
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def summarize_samples(values: np.ndarray) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise ValueError("samples must be a non-empty finite 1D array")
    return {
        "count": int(array.size),
        "mean_us": float(array.mean()),
        "std_us": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "p50_us": float(np.percentile(array, 50)),
        "p95_us": float(np.percentile(array, 95)),
        "p99_us": float(np.percentile(array, 99)),
    }


def paired_bootstrap_ci(values: np.ndarray, repeats: int, seed: int) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size < 2 or not np.isfinite(array).all():
        raise ValueError("paired bootstrap needs at least two finite observations")
    if repeats < 100:
        raise ValueError("bootstrap repeats must be at least 100")
    rng = np.random.default_rng(seed)
    means = np.empty(repeats, dtype=np.float64)
    for index in range(repeats):
        means[index] = array[rng.integers(0, array.size, size=array.size)].mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def run_quality(config: dict[str, Any], output: Path, smoke: bool) -> dict[str, Any]:
    quality = config["quality"]
    repeats = 200 if smoke else int(quality["bootstrap_repeats"])
    threshold = float(quality["head_tail_ratio_threshold"])
    lcb_threshold = float(quality["paired_difference_lcb_threshold"])
    raw_frames: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    source_provenance: list[dict[str, Any]] = []
    producer_path = resolve_repo_path(quality["producer"])
    if not producer_path.is_file():
        raise FileNotFoundError(f"quality producer missing: {producer_path}")
    dependency_paths = {
        "fake_quant.py": REPO_ROOT / "experiments/shared/fake_quant.py",
        "metrics.py": REPO_ROOT / "experiments/shared/metrics.py",
        "modeling.py": REPO_ROOT / "experiments/shared/modeling.py",
        "policies.py": REPO_ROOT / "experiments/shared/policies.py",
        "prompts.py": REPO_ROOT / "experiments/shared/prompts.py",
        "capture_moe.py": REPO_ROOT / "experiments/shared/capture_moe.py",
    }
    expected_dependencies = {
        name: sha256_file(path) for name, path in sorted(dependency_paths.items())
    }

    for model_index, spec in enumerate(quality["models"]):
        source_path = resolve_repo_path(spec["per_document_csv"])
        if not source_path.is_file():
            raise FileNotFoundError(f"quality input missing: {source_path}")
        provenance_path = resolve_repo_path(spec["provenance_json"])
        if not provenance_path.is_file():
            raise FileNotFoundError(
                f"producer-emitted quality provenance missing: {provenance_path}; "
                "rerun the frozen quality producer instead of creating a retrospective sidecar"
            )
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        expected_source_hash = sha256_file(source_path)
        runtime = provenance.get("runtime_environment", {})
        runtime_contract = quality["runtime_contract"]
        runtime_matches = bool(
            isinstance(runtime.get("gpu"), str)
            and runtime_contract["gpu_name_substring"].lower() in runtime["gpu"].lower()
            and runtime.get("compute_capability") == runtime_contract["compute_capability"]
            and isinstance(runtime.get("torch"), str)
            and runtime["torch"].startswith(runtime_contract["torch_version_prefix"])
            and isinstance(runtime.get("torch_cuda"), str)
            and runtime["torch_cuda"].startswith(runtime_contract["torch_cuda_prefix"])
        )
        if (
            provenance.get("schema_version") != "rank-quality-int4-provenance-v1"
            or provenance.get("attestation") != "PRODUCER_EMITTED_DURING_FORWARD_RUN"
            or provenance.get("run_identity") != spec["run_identity"]
            or not runtime_matches
            or provenance.get("producer_sha256") != sha256_file(producer_path)
            or provenance.get("dependency_sha256") != expected_dependencies
            or provenance.get("per_document_sha256") != expected_source_hash
            or provenance.get("quantization_contract") != config["quantization_contract"]
        ):
            raise ValueError(f"quality provenance contract/hash mismatch: {provenance_path}")
        source_provenance.append(
            {
                "model": str(spec["name"]),
                "run_identity": spec["run_identity"],
                "per_document_sha256": expected_source_hash,
                "provenance_sha256": sha256_file(provenance_path),
                "producer_sha256": provenance["producer_sha256"],
                "dependency_sha256": provenance["dependency_sha256"],
                "runtime_environment": runtime,
            }
        )
        frame = pd.read_csv(source_path)
        missing = [column for column in QUALITY_COLUMNS if column not in frame.columns]
        if missing:
            raise ValueError(f"{source_path} missing required columns: {missing}")
        frame = frame.loc[:, QUALITY_COLUMNS].copy()
        if frame["sample_id"].duplicated().any():
            raise ValueError(f"duplicate sample_id in {source_path}")
        if smoke:
            frame = frame.head(min(16, len(frame))).copy()
        if len(frame) < 8:
            raise ValueError(f"quality gate needs at least 8 paired documents, got {len(frame)}")
        numeric = frame[["rank1_int4__kl", "rankk_int4__kl"]].to_numpy(dtype=np.float64)
        if not np.isfinite(numeric).all() or (numeric < 0).any():
            raise ValueError(f"invalid KL values in {source_path}")
        frame["model"] = str(spec["name"])
        frame["paired_head_minus_tail_kl"] = frame["rank1_int4__kl"] - frame["rankk_int4__kl"]
        differences = frame["paired_head_minus_tail_kl"].to_numpy(dtype=np.float64)
        ci_low, ci_high = paired_bootstrap_ci(
            differences, repeats, int(config["seed"]) + model_index * 1009
        )
        head_mean = float(frame["rank1_int4__kl"].mean())
        tail_mean = float(frame["rankk_int4__kl"].mean())
        ratio = head_mean / tail_mean if tail_mean > 0 else math.inf
        passed = bool(ci_low > lcb_threshold and ratio >= threshold)
        summaries.append(
            {
                "model": str(spec["name"]),
                "documents": int(len(frame)),
                "head_mean_kl": head_mean,
                "tail_mean_kl": tail_mean,
                "head_tail_ratio": ratio,
                "paired_difference_mean": float(differences.mean()),
                "paired_difference_ci_low": ci_low,
                "paired_difference_ci_high": ci_high,
                "passed": passed,
                "source": str(source_path),
                "source_sha256": sha256_file(source_path),
            }
        )
        raw_frames.append(frame)

    raw = pd.concat(raw_frames, ignore_index=True)
    summary = pd.DataFrame(summaries)
    raw.to_csv(output / "quality_paired_raw.csv", index=False)
    summary.to_csv(output / "quality_summary.csv", index=False)
    decision = {
        "status": "SMOKE_NOT_SCIENTIFIC" if smoke else "COMPLETE",
        "verdict": (
            "PASS_NECESSARY_QUALITY_SIGNAL"
            if bool(summary["passed"].all())
            else "NO_GO_CPR_QUALITY_SIGNAL"
        ),
        "all_models_passed": bool(summary["passed"].all()),
        "decision_mode": str(quality["decision_mode"]),
        "quantization_contract": config["quantization_contract"],
        "source_provenance": sorted(source_provenance, key=lambda item: item["model"]),
        "thresholds": {
            "head_tail_ratio": threshold,
            "paired_difference_ci_low_gt": lcb_threshold,
        },
        "claim_boundary": (
            "Paired single-forward combine-output KL only. This tests rank-tail quality "
            "ordering, not policy actionability, decode quality, communication, TPOT, or P99."
        ),
    }
    write_json(output / "quality_decision.json", decision)
    return decision


def time_cuda_components(
    functions: dict[str, Callable[[], None]], warmup: int, repeats: int, seed: int
) -> tuple[dict[str, np.ndarray], list[list[str]]]:
    """Interleave component timing to reduce clock/temperature order bias."""
    import torch

    for _ in range(warmup):
        for function in functions.values():
            function()
    torch.cuda.synchronize()
    samples = {
        name: np.empty(repeats, dtype=np.float64) for name in functions
    }
    orders: list[list[str]] = []
    rng = np.random.default_rng(seed)
    names = list(functions)
    for repeat_index in range(repeats):
        order = names.copy()
        rng.shuffle(order)
        orders.append(order)
        for name in order:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            functions[name]()
            end.record()
            end.synchronize()
            samples[name][repeat_index] = float(start.elapsed_time(end) * 1000.0)
    return samples, orders


def wire_time_us(bytes_count: int, gbps: float) -> float:
    if bytes_count < 0 or gbps <= 0:
        raise ValueError("bytes_count must be non-negative and gbps positive")
    return float(bytes_count) * 8.0 / (float(gbps) * 1.0e9) * 1.0e6


def run_codec(
    config: dict[str, Any], output: Path, smoke: bool, allow_other_gpu: bool
) -> dict[str, Any]:
    try:
        import torch
        import triton
        from codec_kernels import assert_codec_matches_reference, build_codec_case
    except ImportError as exc:
        raise RuntimeError("codec gate requires CUDA PyTorch and Triton") from exc

    if not torch.cuda.is_available():
        raise RuntimeError("codec gate requires a CUDA GPU; CPU results are forbidden")
    codec = config["codec"]
    gpu_name = torch.cuda.get_device_name(0)
    required = str(codec["required_gpu_substring"])
    if required.lower() not in gpu_name.lower() and not allow_other_gpu:
        raise RuntimeError(
            f"formal config requires GPU name containing {required!r}, got {gpu_name!r}; "
            "use --allow-other-gpu only for non-formal smoke"
        )
    warmup = 3 if smoke else int(codec["warmup"])
    repeats = 10 if smoke else int(codec["repeats"])
    if warmup < 1 or repeats < 5:
        raise ValueError("codec warmup/repeats are too small")
    torch.manual_seed(int(config["seed"]))
    torch.cuda.manual_seed_all(int(config["seed"]))

    raw_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    rows_values = [int(codec["rows"][0])] if smoke else [int(v) for v in codec["rows"]]
    hidden_values = [int(codec["hidden_sizes"][0])] if smoke else [int(v) for v in codec["hidden_sizes"]]
    modes = [str(codec["decision_mode"])] if smoke else [str(v) for v in codec["modes"]]
    reference_checks: dict[str, list[int]] = {mode: [] for mode in modes}

    for rows in rows_values:
        for hidden in hidden_values:
            source = torch.randn((rows, hidden), device="cuda", dtype=torch.bfloat16)
            baseline_wire_bytes = source.numel() * source.element_size()
            for mode in modes:
                if hidden not in reference_checks[mode]:
                    qmax = 127 if mode == "int8" else 7
                    edge = torch.zeros((2, hidden), device="cuda", dtype=torch.bfloat16)
                    edge[0, :8] = torch.tensor(
                        [-qmax, -6.5, -2.5, -0.5, 0.5, 2.5, 6.5, qmax],
                        device="cuda",
                        dtype=torch.bfloat16,
                    )
                    edge[1, :4] = torch.tensor(
                        [-qmax, -1.0, 0.0, qmax], device="cuda", dtype=torch.bfloat16
                    )
                    assert_codec_matches_reference(edge, mode)
                    reference_checks[mode].append(hidden)

                allocation_before_case = int(torch.cuda.memory_allocated())
                reserved_before_case = int(torch.cuda.memory_reserved())
                case = build_codec_case(source, mode)
                allocation_after_case = int(torch.cuda.memory_allocated())
                reserved_after_case = int(torch.cuda.memory_reserved())
                case.connected()
                torch.cuda.synchronize()
                reconstruction_mse = float(
                    (source.float() - case.output.float()).square().mean().item()
                )
                if not math.isfinite(reconstruction_mse):
                    raise RuntimeError(f"non-finite reconstruction for {mode}/{rows}/{hidden}")

                torch.cuda.reset_peak_memory_stats()
                components = {
                    "pack": case.pack,
                    "unpack": case.unpack,
                    "connected_pack_unpack": case.connected,
                }
                component_samples, component_orders = time_cuda_components(
                    components,
                    warmup,
                    repeats,
                    int(config["seed"]) + rows * 17 + hidden * 31 + len(mode),
                )
                torch.cuda.synchronize()
                peak_allocated = int(torch.cuda.max_memory_allocated())
                peak_reserved = int(torch.cuda.max_memory_reserved())

                component_stats = {
                    name: summarize_samples(samples)
                    for name, samples in component_samples.items()
                }
                order_positions = {
                    (repeat_index, component): order.index(component)
                    for repeat_index, order in enumerate(component_orders)
                    for component in order
                }
                for component, samples in component_samples.items():
                    for repeat_index, latency_us in enumerate(samples):
                        raw_rows.append(
                            {
                                "mode": mode,
                                "rows": rows,
                                "hidden": hidden,
                                "component": component,
                                "repeat_index": repeat_index,
                                "order_in_round": order_positions[(repeat_index, component)],
                                "latency_us": float(latency_us),
                            }
                        )

                saved_bytes = baseline_wire_bytes - case.policy_wire_bytes
                if saved_bytes <= 0:
                    raise RuntimeError(f"codec {mode} did not reduce wire bytes")
                connected_stats = component_stats["connected_pack_unpack"]
                for gbps in (float(v) for v in codec["link_gbps"]):
                    wire_saved = wire_time_us(saved_bytes, gbps)
                    codec_fraction = float(connected_stats["p50_us"]) / wire_saved
                    net_p50 = wire_saved - float(connected_stats["p50_us"])
                    net_p95 = wire_saved - float(connected_stats["p95_us"])
                    viable = bool(
                        net_p95 > 0
                        and codec_fraction
                        <= float(codec["max_codec_fraction_of_wire_saving"])
                    )
                    summaries.append(
                        {
                            "mode": mode,
                            "rows": rows,
                            "hidden": hidden,
                            "link_gbps": gbps,
                            "baseline_wire_bytes": baseline_wire_bytes,
                            "policy_wire_bytes": case.policy_wire_bytes,
                            "scale_bytes": case.scale_bytes,
                            "wire_saved_us_zero_start_analytic": wire_saved,
                            "connected_mean_us": connected_stats["mean_us"],
                            "connected_std_us": connected_stats["std_us"],
                            "connected_p50_us": connected_stats["p50_us"],
                            "connected_p95_us": connected_stats["p95_us"],
                            "connected_p99_us": connected_stats["p99_us"],
                            "codec_fraction_of_wire_saving_p50": codec_fraction,
                            "analytic_net_us_using_codec_p50": net_p50,
                            "analytic_net_us_using_codec_p95": net_p95,
                            "break_even_gbps_using_codec_p50": (
                                float(saved_bytes) * 8.0 / float(connected_stats["p50_us"]) * 1.0e-3
                            ),
                            "reconstruction_mse_random_tensor": reconstruction_mse,
                            "allocated_before_case_bytes": allocation_before_case,
                            "allocated_after_case_bytes": allocation_after_case,
                            "case_live_allocated_delta_bytes": (
                                allocation_after_case - allocation_before_case
                            ),
                            "peak_allocated_absolute_bytes": peak_allocated,
                            "peak_allocated_delta_from_pre_case_bytes": (
                                peak_allocated - allocation_before_case
                            ),
                            "reserved_before_case_bytes": reserved_before_case,
                            "reserved_after_case_bytes": reserved_after_case,
                            "peak_reserved_absolute_bytes": peak_reserved,
                            "viable": viable,
                        }
                    )

    raw = pd.DataFrame(raw_rows)
    summary = pd.DataFrame(summaries)
    raw.to_csv(output / "codec_raw_samples.csv", index=False)
    summary.to_csv(output / "codec_summary.csv", index=False)
    mode_gates: dict[str, Any] = {}
    for mode, group in summary.groupby("mode"):
        viable_fraction = float(group["viable"].mean())
        mode_gates[str(mode)] = {
            "cells": int(len(group)),
            "viable_cells": int(group["viable"].sum()),
            "viable_fraction": viable_fraction,
            "passed": viable_fraction >= float(codec["min_viable_fraction"]),
        }
    decision_mode = str(codec["decision_mode"])
    primary_mode_passed = bool(mode_gates[decision_mode]["passed"])
    decision = {
        "status": "SMOKE_NOT_SCIENTIFIC" if smoke else "COMPLETE",
        "verdict": (
            "PASS_UNFUSED_CODEC_NECESSARY_GATE"
            if primary_mode_passed
            else "NO_GO_CURRENT_UNFUSED_CODEC_PATH"
        ),
        "decision_mode": decision_mode,
        "primary_mode_passed": primary_mode_passed,
        "mode_gates_are_characterization_except_decision_mode": True,
        "mode_gates": mode_gates,
        "reference_self_checks": {
            mode: {"status": "PASSED", "hidden_sizes": values}
            for mode, values in reference_checks.items()
        },
        "quantization_contract": config["quantization_contract"],
        "thresholds": {
            "max_codec_fraction_of_wire_saving": float(
                codec["max_codec_fraction_of_wire_saving"]
            ),
            "min_viable_fraction": float(codec["min_viable_fraction"]),
            "net_using_connected_p95_gt_us": 0.0,
        },
        "environment": {
            "gpu": gpu_name,
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "triton": triton.__version__,
        },
        "claim_boundary": (
            "Connected same-stream GPU pack->unpack plus zero-start analytic byte-transfer "
            "saving. No H2D proxy, NCCL, EP ranks, overlap, fusion, receiver queue, TPOT, or P99."
        ),
    }
    write_json(output / "codec_decision.json", decision)
    return decision


def validate_inputs(config: dict[str, Any]) -> dict[str, Any]:
    quality_sources = []
    for spec in config["quality"]["models"]:
        path = resolve_repo_path(spec["per_document_csv"])
        provenance = resolve_repo_path(spec["provenance_json"])
        quality_sources.append(
            {
                "model": spec["name"],
                "path": str(path),
                "exists": path.is_file(),
                "provenance_path": str(provenance),
                "provenance_exists": provenance.is_file(),
            }
        )
    producer = resolve_repo_path(config["quality"]["producer"])
    inputs_ready = bool(
        producer.is_file()
        and all(item["exists"] and item["provenance_exists"] for item in quality_sources)
    )
    return {
        "schema_version": config["schema_version"],
        "resolved_seed": int(config["seed"]),
        "formal_seeds": [int(value) for value in config["formal_seeds"]],
        "decision_mode": str(config["codec"]["decision_mode"]),
        "quantization_contract": config["quantization_contract"],
        "quality_sources": quality_sources,
        "quality_producer": str(producer),
        "quality_producer_exists": producer.is_file(),
        "inputs_ready": inputs_ready,
        "codec_cells": len(config["codec"]["rows"])
        * len(config["codec"]["hidden_sizes"])
        * len(config["codec"]["modes"])
        * len(config["codec"]["link_gbps"]),
        "core_ep_gate_testable_on_single_gpu": False,
    }


def main() -> None:
    args = parse_args()
    prepare_output(args.output_dir)
    manifest = {
        "status": "RUNNING",
        "experiment": args.experiment,
        "config_path": str(args.config.resolve()),
        "python": sys.version,
        "platform": platform.platform(),
        "decisions": {},
        "global_claim_boundary": (
            "Single-GPU necessary conditions only. Even all PASS results remain BLOCKED "
            "until optimized 8xA100 EP return-path precedence-DAG Gate 0 passes."
        ),
    }
    write_json(args.output_dir / "run_manifest.json", manifest)
    try:
        config = read_config(args.config)
        if args.seed is not None:
            config["seed"] = int(args.seed)
        if not args.smoke and int(config["seed"]) not in set(config["formal_seeds"]):
            raise ValueError("formal run seed must be one of config.formal_seeds")
        if args.allow_other_gpu and not args.smoke:
            raise ValueError("--allow-other-gpu is permitted only with --smoke")
        manifest.update(
            {
                "seed": int(config["seed"]),
                "formal_seeds": [int(value) for value in config["formal_seeds"]],
                "config_sha256": sha256_file(args.config),
            }
        )
        write_json(args.output_dir / "run_manifest.json", manifest)
        resolved_plan = validate_inputs(config)
        write_json(args.output_dir / "resolved_plan.json", resolved_plan)
        if args.validate_only:
            manifest["status"] = (
                "VALIDATED_NOT_RUN" if resolved_plan["inputs_ready"] else "BLOCKED_MISSING_INPUTS"
            )
            write_json(args.output_dir / "run_manifest.json", manifest)
            if not resolved_plan["inputs_ready"]:
                raise SystemExit(2)
            return

        decisions: dict[str, Any] = {}
        if args.experiment in ("quality", "all"):
            decisions["quality"] = run_quality(config, args.output_dir, args.smoke)
        if args.experiment in ("codec", "all"):
            decisions["codec"] = run_codec(
                config, args.output_dir, args.smoke, args.allow_other_gpu
            )
        manifest["status"] = "SMOKE_NOT_SCIENTIFIC" if args.smoke else "COMPLETE"
        manifest["decisions"] = decisions
        if "quality" in decisions:
            manifest["quality_source_provenance"] = decisions["quality"][
                "source_provenance"
            ]
        write_json(args.output_dir / "run_manifest.json", manifest)
    except Exception as exc:
        manifest["status"] = "FAILED"
        manifest["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        write_json(args.output_dir / "run_manifest.json", manifest)
        raise


if __name__ == "__main__":
    main()
