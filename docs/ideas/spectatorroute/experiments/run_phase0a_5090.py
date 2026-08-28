#!/usr/bin/env python3
"""Run the frozen SpectatorRoute N05 Phase-0A capability gate.

The parent process deliberately imports only the Python standard library.  It
launches two fresh workers:

1. ``numeric`` captures frozen pretrained OLMoE victim rows and performs the
   ten-repeat M sweep without cuBLAS logging.
2. ``trace`` replays every captured cell once with cuBLASLt logging enabled.

Keeping the logger in a fresh process makes the order of trace records
mechanically alignable with an explicit call index.  A missing, extra, or
shape-mismatched GEMM makes the run INVALID rather than guessed.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping, Sequence


class ProtocolError(RuntimeError):
    """A fail-closed protocol or artifact-integrity violation."""


class ContaminationError(ProtocolError):
    """Foreign or drifted GPU activity invalidated the run."""


TRACE_PREFIX = "[Trace][cublasLtTSTMatmul]"
DESCRIPTOR_NAMES = ("Adesc", "Bdesc", "Cdesc", "Ddesc", "computeDesc")
ALLOWED_ALGO_FIELDS = {
    "algoId",
    "customOption",
    "tile",
    "stages",
    "reductionScheme",
    "numSplitsK",
}


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ProtocolError(f"expected JSON object: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ProtocolError(f"{path}:{line_no} is not a JSON object")
            rows.append(value)
    return rows


def write_json_no_overwrite(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_jsonl_row(handle: Any, value: Mapping[str, Any]) -> None:
    handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")


def _parse_key_values(text: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for token in text.split():
        if "=" not in token:
            continue
        key, raw = token.split("=", 1)
        if re.fullmatch(r"-?\d+", raw):
            result[key] = int(raw)
        elif re.fullmatch(r"-?(?:\d+\.\d*|\d*\.\d+)", raw):
            result[key] = float(raw)
        else:
            result[key] = raw
    return result


def parse_cublaslt_trace_line(line: str) -> dict[str, Any] | None:
    """Parse one cublasLt matmul Trace line, excluding pointer addresses."""

    if TRACE_PREFIX not in line:
        return None
    parsed: dict[str, Any] = {}
    for name in DESCRIPTOR_NAMES:
        match = re.search(rf"\b{name}=\[([^\]]*)\]", line)
        if match is None:
            raise ProtocolError(f"trace line lacks {name}: {line[:240]}")
        parsed[name] = _parse_key_values(match.group(1))
    algo = re.search(r"\balgo=\[([^\]]*)\]", line)
    workspace = re.search(r"\bworkSpaceSizeInBytes=(\d+)", line)
    beta = re.search(r"\bbeta=([^ ]+)", line)
    out_of_place = re.search(r"\boutOfPlace=([^ ]+)", line)
    if algo is None or workspace is None or beta is None or out_of_place is None:
        raise ProtocolError(f"trace line lacks algorithm fields: {line[:240]}")
    parsed["algo"] = _parse_key_values(algo.group(1))
    parsed["workspace_bytes"] = int(workspace.group(1))
    parsed["beta"] = beta.group(1)
    parsed["out_of_place"] = out_of_place.group(1)
    return parsed


def algorithm_signature(record: Mapping[str, Any], projection_role: str) -> dict[str, Any]:
    """Return regime fields without treating the intervention M as a regime.

    Projection role and compute semantics are retained.  Matrix ``cols=M`` is
    stored in the raw trace record but deliberately excluded here; otherwise
    every treatment would be a tautological regime change.
    """

    # Workspace bytes are retained in ``raw_trace_records`` but excluded from
    # the positive predicate.  Their size may scale mechanically with M even
    # when algo/tile/stages/split-K/reduction are unchanged; counting that
    # alone would manufacture a regime transition.
    return {
        "projection_role": projection_role,
        "compute_desc": record["computeDesc"],
        "algo": record["algo"],
    }


def validate_projection_triplet(
    records: Sequence[Mapping[str, Any]], *, m_value: int, hidden_size: int, intermediate_size: int
) -> list[str]:
    if len(records) != 3:
        raise ProtocolError(f"expert call produced {len(records)} traced GEMMs, expected 3")

    roles = ["gate_proj", "up_proj", "down_proj"]
    expected = [
        (hidden_size, intermediate_size, hidden_size, intermediate_size),
        (hidden_size, intermediate_size, hidden_size, intermediate_size),
        (intermediate_size, hidden_size, intermediate_size, hidden_size),
    ]
    for idx, (record, dims) in enumerate(zip(records, expected)):
        a_rows, a_cols, b_rows, d_rows = dims
        adesc = record["Adesc"]
        bdesc = record["Bdesc"]
        cdesc = record["Cdesc"]
        ddesc = record["Ddesc"]
        observed = (
            int(adesc.get("rows", -1)),
            int(adesc.get("cols", -1)),
            int(bdesc.get("rows", -1)),
            int(bdesc.get("cols", -1)),
            int(cdesc.get("rows", -1)),
            int(cdesc.get("cols", -1)),
            int(ddesc.get("rows", -1)),
            int(ddesc.get("cols", -1)),
        )
        wanted = (
            a_rows,
            a_cols,
            b_rows,
            m_value,
            d_rows,
            m_value,
            d_rows,
            m_value,
        )
        if observed != wanted:
            raise ProtocolError(
                f"{roles[idx]} trace shape {observed} != expected {wanted}"
            )
        for descriptor_name, descriptor, leading_dim in (
            ("Adesc", adesc, a_rows),
            ("Bdesc", bdesc, b_rows),
            ("Cdesc", cdesc, d_rows),
            ("Ddesc", ddesc, d_rows),
        ):
            if set(descriptor) != {"type", "rows", "cols", "ld"}:
                raise ProtocolError(
                    f"{roles[idx]} {descriptor_name} has unknown/missing fields: {descriptor}"
                )
            if descriptor.get("type") != "R_16BF":
                raise ProtocolError(
                    f"{roles[idx]} {descriptor_name} is not BF16: {descriptor}"
                )
            if int(descriptor.get("ld", -1)) != leading_dim:
                raise ProtocolError(
                    f"{roles[idx]} {descriptor_name} leading dimension mismatch"
                )
        compute = record["computeDesc"]
        expected_compute = {
            "computeType": "COMPUTE_32F",
            "scaleType": "R_32F",
            "transa": "OP_T",
            "smCountTarget": 170,
        }
        if compute != expected_compute:
            raise ProtocolError(
                f"{roles[idx]} compute descriptor {compute} != {expected_compute}"
            )
        algo = record["algo"]
        if "algoId" not in algo or not set(algo).issubset(ALLOWED_ALGO_FIELDS):
            raise ProtocolError(
                f"{roles[idx]} algorithm has missing/unknown fields: {algo}"
            )
        if ("numSplitsK" in algo) != ("reductionScheme" in algo):
            raise ProtocolError(
                f"{roles[idx]} split-K/reduction fields are incomplete: {algo}"
            )
        if record["beta"] != "0" or record["out_of_place"] != "0":
            raise ProtocolError(
                f"{roles[idx]} unexpected beta/out-of-place semantics: {record}"
            )
    return roles


def _tensor_sha256(tensor: Any) -> str:
    import torch

    cpu = tensor.detach().contiguous().cpu()
    byte_view = cpu.view(torch.uint8)
    return hashlib.sha256(byte_view.numpy().tobytes()).hexdigest()


def _bitwise_changed_bf16_elements(left: Any, right: Any) -> int:
    """Count BF16 elements whose raw 16-bit encodings differ."""

    import torch

    if left.dtype != torch.bfloat16 or right.dtype != torch.bfloat16:
        raise ProtocolError("bitwise BF16 comparison received non-BF16 tensor")
    if tuple(left.shape) != tuple(right.shape):
        raise ProtocolError("bitwise BF16 comparison shape mismatch")
    left_bits = left.detach().contiguous().cpu().view(torch.int16)
    right_bits = right.detach().contiguous().cpu().view(torch.int16)
    return int(torch.ne(left_bits, right_bits).sum().item())


def _nvidia_identity() -> dict[str, str]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=uuid,name,driver_version,compute_cap",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = [row.strip() for row in completed.stdout.splitlines() if row.strip()]
    if len(rows) != 1:
        raise ProtocolError(f"expected exactly one GPU, found {len(rows)}")
    fields = [field.strip() for field in rows[0].split(",")]
    if len(fields) != 4:
        raise ProtocolError(f"unexpected nvidia-smi identity row: {rows[0]}")
    return {
        "uuid": fields[0],
        "name": fields[1],
        "driver": fields[2],
        "compute_capability": fields[3],
    }


def _nvidia_compute_processes() -> list[dict[str, Any]]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    processes: list[dict[str, Any]] = []
    for row in completed.stdout.splitlines():
        if not row.strip():
            continue
        fields = [field.strip() for field in row.split(",", maxsplit=3)]
        if len(fields) != 4 or not fields[1].isdigit():
            raise ProtocolError(f"unexpected nvidia-smi process row: {row}")
        processes.append(
            {
                "gpu_uuid": fields[0],
                "pid": int(fields[1]),
                "process_name": fields[2],
                "used_gpu_memory_mib": fields[3],
            }
        )
    return processes


def _classify_gpu_process_scope(
    config: Mapping[str, Any],
    *,
    allowed_pids: set[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    expected_uuid = str(config["environment"]["gpu_uuid"])
    observed = _nvidia_compute_processes()
    foreign = [
        process
        for process in observed
        if process["gpu_uuid"] == expected_uuid
        and int(process["pid"]) not in allowed_pids
    ]
    unexpected_gpus = sorted(
        {
            str(process["gpu_uuid"])
            for process in observed
            if process["gpu_uuid"] != expected_uuid
        }
    )
    return observed, foreign, unexpected_gpus


def assert_no_foreign_gpu_processes(
    config: Mapping[str, Any],
    *,
    allowed_pids: set[int],
    output_dir: Path | None = None,
    stage: str | None = None,
) -> list[dict[str, Any]]:
    expected_uuid = str(config["environment"]["gpu_uuid"])
    observed, foreign, unexpected_gpus = _classify_gpu_process_scope(
        config, allowed_pids=allowed_pids
    )
    status = "CLEAN" if not foreign and not unexpected_gpus else "INVALID_CONTAMINATION"
    if output_dir is not None:
        if stage is None or not re.fullmatch(r"[a-z0-9_]+", stage):
            raise ProtocolError("GPU process snapshot requires a safe stage name")
        write_json_no_overwrite(
            output_dir / f"gpu_processes_{stage}.json",
            {
                "schema_version": "spectatorroute-phase0a-gpu-process-snapshot-v1",
                "stage": stage,
                "status": status,
                "expected_gpu_uuid": expected_uuid,
                "allowed_pids": sorted(allowed_pids),
                "observed_processes": observed,
                "foreign_processes": foreign,
                "unexpected_gpu_uuids": unexpected_gpus,
            },
        )
    if foreign or unexpected_gpus:
        raise ContaminationError(
            "GPU process scope is contaminated: "
            f"foreign={foreign!r}, unexpected_gpu_uuids={unexpected_gpus!r}"
        )
    return observed


def verify_static_bindings(
    config: Mapping[str, Any], repo_root: Path, *, include_weight_hashes: bool = True
) -> dict[str, Any]:
    data = config["data"]
    manifest = repo_root / str(data["manifest"])
    observed_manifest_hash = sha256_file(manifest)
    if observed_manifest_hash != data["manifest_sha256"]:
        raise ProtocolError(
            f"manifest hash {observed_manifest_hash} != frozen {data['manifest_sha256']}"
        )
    documents = load_jsonl(manifest)
    if len(documents) != int(data["document_count"]):
        raise ProtocolError(
            f"manifest count {len(documents)} != frozen {data['document_count']}"
        )
    bound_data_files: dict[str, str] = {}
    for path_field, hash_field in (
        ("calibration_manifest", "calibration_manifest_sha256"),
        ("provenance", "provenance_sha256"),
        ("historical_hash_registry", "historical_hash_registry_sha256"),
    ):
        path = repo_root / str(data[path_field])
        observed = sha256_file(path)
        if observed != data[hash_field]:
            raise ProtocolError(
                f"{path_field} hash {observed} != frozen {data[hash_field]}"
            )
        bound_data_files[path_field] = observed
    calibration_documents = load_jsonl(repo_root / str(data["calibration_manifest"]))
    if len(calibration_documents) != 8:
        raise ProtocolError(
            f"calibration manifest count {len(calibration_documents)} != frozen 8"
        )
    load_json(repo_root / str(data["provenance"]))
    load_json(repo_root / str(data["historical_hash_registry"]))

    model = config["model"]
    model_root = Path(str(model["local_path"]))
    hashes: dict[str, str] = {}
    if include_weight_hashes:
        for filename, expected in model["file_sha256"].items():
            path = model_root / filename
            if not path.is_file():
                raise ProtocolError(f"missing frozen model file: {path}")
            observed = sha256_file(path)
            hashes[filename] = observed
            if observed != expected:
                raise ProtocolError(f"{filename} hash {observed} != frozen {expected}")
    return {
        "manifest_path": str(manifest),
        "manifest_sha256": observed_manifest_hash,
        "document_count": len(documents),
        "calibration_document_count": len(calibration_documents),
        "bound_data_file_sha256": bound_data_files,
        "model_file_sha256": hashes,
    }


def verify_frozen_lock(
    *, lock_path: Path, expected_lock_sha256: str, repo_root: Path
) -> dict[str, Any]:
    """Verify the externally supplied, content-addressed pre-run seal."""

    if not re.fullmatch(r"[0-9a-f]{64}", expected_lock_sha256):
        raise ProtocolError("--frozen-lock-sha256 must be one lowercase SHA-256")
    observed_lock_sha256 = sha256_file(lock_path)
    if observed_lock_sha256 != expected_lock_sha256:
        raise ProtocolError(
            f"frozen lock hash {observed_lock_sha256} != supplied {expected_lock_sha256}"
        )
    lock = load_json(lock_path)
    if lock.get("schema_version") != "spectatorroute-phase0a-frozen-lock-v1":
        raise ProtocolError("unknown frozen-lock schema")
    if lock.get("status") != "FROZEN_PRE_RUN":
        raise ProtocolError("frozen lock status is not FROZEN_PRE_RUN")
    files = lock.get("files")
    if not isinstance(files, dict) or not files:
        raise ProtocolError("frozen lock has no file bindings")
    verified: dict[str, str] = {}
    root = repo_root.resolve()
    for raw_relative, expected_file_hash in files.items():
        if not isinstance(raw_relative, str) or not re.fullmatch(
            r"[0-9a-f]{64}", str(expected_file_hash)
        ):
            raise ProtocolError("invalid frozen file binding")
        relative = Path(raw_relative)
        if relative.is_absolute() or ".." in relative.parts:
            raise ProtocolError(f"unsafe frozen path: {raw_relative}")
        resolved = (root / relative).resolve()
        if root not in resolved.parents:
            raise ProtocolError(f"frozen path escapes repo root: {raw_relative}")
        if not resolved.is_file():
            raise ProtocolError(f"missing frozen input: {raw_relative}")
        observed = sha256_file(resolved)
        if observed != expected_file_hash:
            raise ProtocolError(
                f"frozen input {raw_relative} hash {observed} != {expected_file_hash}"
            )
        verified[raw_relative] = observed

    required = {
        "docs/ideas/spectatorroute/N05_PHASE0_FROZEN_PROTOCOL_20260729.md",
        "docs/ideas/spectatorroute/experiments/configs/phase0a_5090_v1.json",
        "docs/ideas/spectatorroute/experiments/run_phase0a_5090.py",
        "docs/ideas/spectatorroute/experiments/test_phase0a_5090.py",
    }
    if set(verified) != required:
        raise ProtocolError(
            f"frozen lock file set {sorted(verified)} != required {sorted(required)}"
        )
    constants = lock.get("frozen_constants")
    expected_constants = {
        "victim_count": 64,
        "document_count": 32,
        "token_offsets": [0, 256],
        "window_tokens": 16,
        "victim_position": 15,
        "m_values": [1, 2, 4, 8, 16, 32, 64],
        "reference_m": 64,
        "repeats": 10,
        "minimum_distinct_positive_victims": 8,
        "max_gpu_seconds": 1800,
    }
    if constants != expected_constants:
        raise ProtocolError(
            f"frozen constants {constants!r} != runner-required {expected_constants!r}"
        )
    return {
        "lock_path": str(lock_path),
        "lock_sha256": observed_lock_sha256,
        "file_sha256": verified,
        "frozen_constants": constants,
    }


def verify_frozen_semantics(config: Mapping[str, Any]) -> None:
    """Reject edits to frozen semantic fields even when code would ignore them."""

    expected_environment = {
        "gpu_uuid": "GPU-1a63767e-c187-5389-1617-565980faebf6",
        "gpu_exact_name": "NVIDIA GeForce RTX 5090",
        "compute_capability": [12, 0],
        "driver": "580.76.05",
        "cuda": "12.8",
        "torch": "2.8.0+cu128",
        "transformers": "4.57.6",
        "python": "3.12.3",
        "cublaslt": {
            "path": "/root/miniconda3/lib/python3.12/site-packages/nvidia/cublas/lib/libcublasLt.so.12",
            "version": 120804,
            "sha256": "10b5e6631cf8115c661eb895ed1533826308b58f7956466f53d236a40c9b622c",
        },
        "transformers_olmoe_source": {
            "path": "/root/miniconda3/lib/python3.12/site-packages/transformers/models/olmoe/modeling_olmoe.py",
            "sha256": "248717a8477fbcb4b16ce648a9ace829b6b2f4a002191a7f64ee720eb65f4c0d",
        },
        "matmul_state": {
            "allow_tf32": False,
            "allow_bf16_reduced_precision_reduction": True,
            "allow_fp16_reduced_precision_reduction": True,
            "preferred_blas_library": "Cublas",
        },
        "required_absent_environment_variables": [
            "CUBLAS_WORKSPACE_CONFIG",
            "CUBLASLT_HEURISTICS_CACHE_CAPACITY",
            "CUBLASLT_WORKSPACE_SIZE",
            "NVIDIA_TF32_OVERRIDE",
            "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE",
        ],
        "required_environment_variables_after_torch_import": {
            "CUDA_MODULE_LOADING": "LAZY"
        },
    }
    if config["environment"] != expected_environment:
        raise ProtocolError("frozen environment semantics mismatch")
    expected_model = {
        "repo_id": "allenai/OLMoE-1B-7B-0924",
        "revision": "6d84c48581ece794365f2b8e9cfb043c68ade9c5",
        "local_path": "/root/autodl-tmp/models/olmoe",
        "dtype": "bfloat16",
        "num_hidden_layers": 16,
        "hidden_size": 2048,
        "intermediate_size": 1024,
        "num_experts": 64,
        "num_experts_per_tok": 8,
    }
    model = config["model"]
    observed_model = {key: model.get(key) for key in expected_model}
    if observed_model != expected_model:
        raise ProtocolError(
            f"frozen model semantics {observed_model!r} != {expected_model!r}"
        )
    expected_data = {
        "manifest": "docs/ideas/routeguard_kv/experiments/data/r0a_5090_v1/sealed_manifest.jsonl",
        "manifest_sha256": "469e5da28dc794e50f9e3d8b1d6b2b13dfb7079d1bb6fdf9d00cd41b7c4d0d11",
        "calibration_manifest": "docs/ideas/routeguard_kv/experiments/data/r0a_5090_v1/calibration_manifest.jsonl",
        "calibration_manifest_sha256": "bfb8912539806d2948595eb7ba42cfb7d09aae0b31c7c00dfbc62136abc82630",
        "provenance": "docs/ideas/routeguard_kv/experiments/data/r0a_5090_v1/provenance.json",
        "provenance_sha256": "20ad389a694972b2b6895bb8479e1dec444e4a1d846668cf8ccba90a0cce4811",
        "historical_hash_registry": "docs/ideas/routeguard_kv/experiments/data/r0a_5090_v1/historical_hash_registry.json",
        "historical_hash_registry_sha256": "e3cb17d1a4649a6012b7394402fc083792607300418c42e28f76b2afc3c273c0",
        "document_count": 32,
        "token_offsets": [0, 256],
        "window_tokens": 16,
        "add_special_tokens": False,
        "victim_position": 15,
        "victim_count": 64,
    }
    observed_data = {key: config["data"].get(key) for key in expected_data}
    if observed_data != expected_data:
        raise ProtocolError("frozen data semantics mismatch")
    expected_intervention = {
        "m_values": [1, 2, 4, 8, 16, 32, 64],
        "reference_m": 64,
        "repeats": 10,
        "filler": "repeat_identical_victim_hidden_row",
        "selected_experts": "all_native_top8_for_each_victim_layer",
        "cublaslt_log_level": 5,
        "cublaslt_log_mask": 31,
    }
    if config["intervention"] != expected_intervention:
        raise ProtocolError("frozen intervention semantics mismatch")
    expected_gate = {
        "minimum_distinct_positive_victims": 8,
        "require_within_m_bitwise_stability": True,
        "require_actual_regime_signature_change": True,
        "require_cross_m_bitwise_output_difference": True,
        "max_gpu_seconds": 1800,
    }
    if config["gate"] != expected_gate:
        raise ProtocolError("frozen gate semantics mismatch")
    expected_boundary = (
        "single_rtx5090_pretrained_olmoe_expert_arithmetic_capability_only_"
        "not_prompt_attack_not_ep_not_serving"
    )
    if config.get("evidence_boundary") != expected_boundary:
        raise ProtocolError("frozen evidence boundary mismatch")


def verify_locked_invocation(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    repo_root = Path(args.repo_root).resolve()
    lock_info = verify_frozen_lock(
        lock_path=Path(args.frozen_lock).resolve(),
        expected_lock_sha256=str(args.frozen_lock_sha256),
        repo_root=repo_root,
    )
    config_path = Path(args.config).resolve()
    runner_path = Path(__file__).resolve()
    frozen_files = lock_info["file_sha256"]
    expected_config_hash = frozen_files[
        "docs/ideas/spectatorroute/experiments/configs/phase0a_5090_v1.json"
    ]
    expected_runner_hash = frozen_files[
        "docs/ideas/spectatorroute/experiments/run_phase0a_5090.py"
    ]
    if sha256_file(config_path) != expected_config_hash:
        raise ProtocolError("invoked config is not the frozen config bytes")
    if sha256_file(runner_path) != expected_runner_hash:
        raise ProtocolError("invoked runner is not the frozen runner bytes")
    config = load_json(config_path)
    observed_constants = {
        "victim_count": int(config["data"]["victim_count"]),
        "document_count": int(config["data"]["document_count"]),
        "token_offsets": [int(value) for value in config["data"]["token_offsets"]],
        "window_tokens": int(config["data"]["window_tokens"]),
        "victim_position": int(config["data"]["victim_position"]),
        "m_values": [int(value) for value in config["intervention"]["m_values"]],
        "reference_m": int(config["intervention"]["reference_m"]),
        "repeats": int(config["intervention"]["repeats"]),
        "minimum_distinct_positive_victims": int(
            config["gate"]["minimum_distinct_positive_victims"]
        ),
        "max_gpu_seconds": int(config["gate"]["max_gpu_seconds"]),
    }
    if observed_constants != lock_info["frozen_constants"]:
        raise ProtocolError("config constants do not match the frozen lock")
    if config.get("schema_version") != "spectatorroute-phase0a-v1":
        raise ProtocolError("unknown Phase-0A config schema")
    if config.get("status") != "FROZEN":
        raise ProtocolError("Phase-0A config status is not FROZEN")
    verify_frozen_semantics(config)
    return config, lock_info


def verify_runtime_environment(
    config: Mapping[str, Any],
    *,
    contamination_output_dir: Path | None = None,
    contamination_stage: str | None = None,
) -> dict[str, Any]:
    import ctypes
    import torch
    import transformers

    expected = config["environment"]
    gpu = _nvidia_identity()
    observed = {
        "gpu": gpu,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "transformers": transformers.__version__,
        "torch_compute_capability": list(torch.cuda.get_device_capability(0)),
        "python": sys.version.split()[0],
    }
    checks = {
        "gpu UUID": (gpu["uuid"], expected["gpu_uuid"]),
        "gpu name": (gpu["name"], expected["gpu_exact_name"]),
        "driver": (gpu["driver"], expected["driver"]),
        "nvidia compute capability": (
            gpu["compute_capability"],
            ".".join(str(part) for part in expected["compute_capability"]),
        ),
        "torch compute capability": (
            observed["torch_compute_capability"],
            expected["compute_capability"],
        ),
        "torch": (observed["torch"], expected["torch"]),
        "CUDA": (observed["cuda"], expected["cuda"]),
        "Transformers": (observed["transformers"], expected["transformers"]),
        "Python": (observed["python"], expected["python"]),
    }
    mismatches = [f"{name}: {actual!r} != {wanted!r}" for name, (actual, wanted) in checks.items() if actual != wanted]
    if mismatches:
        raise ProtocolError("environment mismatch: " + "; ".join(mismatches))

    observed["compute_processes"] = assert_no_foreign_gpu_processes(
        config,
        allowed_pids={os.getpid()},
        output_dir=contamination_output_dir,
        stage=contamination_stage,
    )

    for variable in expected["required_absent_environment_variables"]:
        if variable in os.environ:
            raise ProtocolError(
                f"environment variable {variable} must be absent, got {os.environ[variable]!r}"
            )
    for variable, frozen_value in expected[
        "required_environment_variables_after_torch_import"
    ].items():
        if os.environ.get(variable) != frozen_value:
            raise ProtocolError(
                f"environment variable {variable}={os.environ.get(variable)!r} != frozen {frozen_value!r}"
            )

    cublaslt = expected["cublaslt"]
    cublaslt_path = Path(str(cublaslt["path"]))
    observed_cublaslt_hash = sha256_file(cublaslt_path)
    if observed_cublaslt_hash != cublaslt["sha256"]:
        raise ProtocolError("loaded-stack libcublasLt content hash mismatch")
    library = ctypes.CDLL(str(cublaslt_path))
    get_version = library.cublasLtGetVersion
    get_version.restype = ctypes.c_size_t
    observed_cublaslt_version = int(get_version())
    if observed_cublaslt_version != int(cublaslt["version"]):
        raise ProtocolError(
            f"libcublasLt version {observed_cublaslt_version} != {cublaslt['version']}"
        )

    source = expected["transformers_olmoe_source"]
    observed_source_hash = sha256_file(Path(str(source["path"])))
    if observed_source_hash != source["sha256"]:
        raise ProtocolError("Transformers OLMoE source hash mismatch")

    expected_matmul = expected["matmul_state"]
    observed_matmul = {
        "allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "allow_bf16_reduced_precision_reduction": bool(
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        ),
        "allow_fp16_reduced_precision_reduction": bool(
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
        ),
        "preferred_blas_library": str(
            torch.backends.cuda.preferred_blas_library()
        ).split(".")[-1],
    }
    if observed_matmul != expected_matmul:
        raise ProtocolError(
            f"matmul state {observed_matmul!r} != frozen {expected_matmul!r}"
        )
    observed["cublaslt"] = {
        "path": str(cublaslt_path),
        "version": observed_cublaslt_version,
        "sha256": observed_cublaslt_hash,
    }
    observed["transformers_olmoe_source_sha256"] = observed_source_hash
    observed["matmul_state"] = observed_matmul
    return observed


def assert_loaded_cublaslt(config: Mapping[str, Any]) -> list[str]:
    expected = str(Path(config["environment"]["cublaslt"]["path"]).resolve())
    observed = sorted(
        {
            str(Path(line.split()[-1]).resolve())
            for line in Path("/proc/self/maps").read_text(encoding="utf-8").splitlines()
            if "libcublasLt" in line and line.split()[-1].startswith("/")
        }
    )
    if observed != [expected]:
        raise ProtocolError(
            f"mapped libcublasLt paths {observed!r} != frozen {[expected]!r}"
        )
    return observed


def load_frozen_model(config: Mapping[str, Any]) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_cfg = config["model"]
    local_path = str(model_cfg["local_path"])
    tokenizer = AutoTokenizer.from_pretrained(
        local_path, local_files_only=True, use_fast=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        local_path,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    model.eval()
    model.to("cuda")

    expected_fields = {
        "num_hidden_layers": model_cfg["num_hidden_layers"],
        "hidden_size": model_cfg["hidden_size"],
        "intermediate_size": model_cfg["intermediate_size"],
        "num_experts": model_cfg["num_experts"],
        "num_experts_per_tok": model_cfg["num_experts_per_tok"],
    }
    mismatches: list[str] = []
    for field, expected in expected_fields.items():
        actual = getattr(model.config, field, None)
        if actual != expected:
            mismatches.append(f"{field}: {actual!r} != {expected!r}")
    if getattr(model.config, "norm_topk_prob", None) is not False:
        mismatches.append(
            f"norm_topk_prob: {getattr(model.config, 'norm_topk_prob', None)!r} != False"
        )
    if mismatches:
        raise ProtocolError("model config mismatch: " + "; ".join(mismatches))
    blocks = [layer.mlp for layer in model.model.layers]
    if len(blocks) != int(model_cfg["num_hidden_layers"]):
        raise ProtocolError(f"found {len(blocks)} MoE blocks")
    for layer_idx, block in enumerate(blocks):
        if len(block.experts) != int(model_cfg["num_experts"]):
            raise ProtocolError(f"layer {layer_idx} expert count mismatch")
    return model, tokenizer


def run_real_gpu_acceptance(args: argparse.Namespace) -> int:
    """Exercise the pinned pretrained model and one real expert on the bound GPU."""

    import torch
    import torch.nn.functional as F

    config, lock_info = verify_locked_invocation(args)
    repo_root = Path(args.repo_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise ProtocolError(f"acceptance output already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    assert_no_foreign_gpu_processes(
        config,
        allowed_pids={os.getpid()},
        output_dir=output_dir,
        stage="acceptance_before_cuda",
    )
    bindings = verify_static_bindings(config, repo_root, include_weight_hashes=True)
    environment = verify_runtime_environment(
        config,
        contamination_output_dir=output_dir,
        contamination_stage="acceptance_after_torch_import",
    )
    model, tokenizer = load_frozen_model(config)

    data_cfg = config["data"]
    model_cfg = config["model"]
    documents = load_jsonl(repo_root / str(data_cfg["manifest"]))
    documents.sort(key=lambda row: int(row["document_index"]))
    if not documents or int(documents[0]["document_index"]) != 0:
        raise ProtocolError("real-GPU acceptance cannot locate frozen document 0")
    document = documents[0]
    text = str(document["text"])
    if hashlib.sha256(text.encode("utf-8")).hexdigest() != document["text_sha256"]:
        raise ProtocolError("real-GPU acceptance document hash mismatch")
    token_ids = tokenizer(
        text, add_special_tokens=bool(data_cfg["add_special_tokens"])
    )["input_ids"]
    window_tokens = int(data_cfg["window_tokens"])
    window = token_ids[:window_tokens]
    if len(window) != window_tokens:
        raise ProtocolError("real-GPU acceptance token window is incomplete")

    first_block = model.model.layers[0].mlp
    captured: dict[str, Any] = {}

    def capture_first_block(_module: Any, inputs: tuple[Any, ...]) -> None:
        if len(inputs) != 1:
            raise ProtocolError("real-GPU acceptance hook input mismatch")
        captured["hidden"] = inputs[0].detach().clone()

    handle = first_block.register_forward_pre_hook(capture_first_block)
    try:
        input_ids = torch.tensor([window], dtype=torch.long, device="cuda")
        with torch.inference_mode():
            output = model(
                input_ids=input_ids,
                use_cache=False,
                output_router_logits=True,
                return_dict=True,
            )
    finally:
        handle.remove()
    hidden = captured.get("hidden")
    expected_hidden_shape = (
        1,
        window_tokens,
        int(model_cfg["hidden_size"]),
    )
    if hidden is None or tuple(hidden.shape) != expected_hidden_shape:
        raise ProtocolError("real-GPU acceptance hidden capture mismatch")
    if output.router_logits is None or len(output.router_logits) != int(
        model_cfg["num_hidden_layers"]
    ):
        raise ProtocolError("real-GPU acceptance router capture mismatch")
    native_logits = output.router_logits[0].reshape(
        window_tokens, int(model_cfg["num_experts"])
    )
    with torch.inference_mode():
        replay_logits = first_block.gate(hidden.reshape(window_tokens, -1))
    if not torch.equal(native_logits, replay_logits):
        raise ProtocolError("real-GPU acceptance native/replay router mismatch")

    victim_position = int(data_cfg["victim_position"])
    victim_logits = native_logits[victim_position]
    probabilities = F.softmax(victim_logits, dim=-1, dtype=torch.float)
    _weights, experts = torch.topk(
        probabilities, k=int(model_cfg["num_experts_per_tok"]), dim=-1
    )
    expert_id = int(experts[0].item())
    expert = first_block.experts[expert_id]
    victim_hidden = hidden[0, victim_position]
    smoke_shapes: dict[str, list[int]] = {}
    for m_value in (1, 64):
        with torch.inference_mode():
            row = expert(victim_hidden.reshape(1, -1).repeat(m_value, 1))[0]
        if row.dtype != torch.bfloat16 or not bool(torch.isfinite(row).all().item()):
            raise ProtocolError("real-GPU acceptance expert output is invalid")
        smoke_shapes[str(m_value)] = list(row.shape)
    torch.cuda.synchronize()
    loaded_cublaslt = assert_loaded_cublaslt(config)
    processes = assert_no_foreign_gpu_processes(
        config,
        allowed_pids={os.getpid()},
        output_dir=output_dir,
        stage="acceptance_after_gpu",
    )
    write_json_no_overwrite(
        output_dir / "REAL_GPU_ACCEPTANCE.json",
        {
            "schema_version": "spectatorroute-phase0a-real-gpu-acceptance-v1",
            "status": "PASS",
            "evidence_boundary": "pre-run harness acceptance only; not scientific evidence",
            "frozen_lock_sha256": lock_info["lock_sha256"],
            "config_sha256": sha256_file(Path(args.config)),
            "runner_sha256": sha256_file(Path(__file__)),
            "bindings": bindings,
            "environment": environment,
            "loaded_cublaslt": loaded_cublaslt,
            "compute_processes": processes,
            "document_index": 0,
            "window_token_ids_sha256": canonical_sha256(list(map(int, window))),
            "router_logits_sha256": _tensor_sha256(native_logits),
            "selected_expert_id": expert_id,
            "expert_smoke_m_values": [1, 64],
            "expert_output_shapes": smoke_shapes,
            "paper_claim_authorized": False,
        },
    )
    return 0


def verify_real_gpu_acceptance_artifact(
    *,
    path: Path,
    config: Mapping[str, Any],
    lock_info: Mapping[str, Any],
    config_path: Path,
    runner_path: Path,
) -> dict[str, Any]:
    if not path.is_file():
        raise ProtocolError(f"mandatory real-GPU acceptance artifact is absent: {path}")
    artifact = load_json(path)
    expected_scalars = {
        "schema_version": "spectatorroute-phase0a-real-gpu-acceptance-v1",
        "status": "PASS",
        "evidence_boundary": "pre-run harness acceptance only; not scientific evidence",
        "frozen_lock_sha256": lock_info["lock_sha256"],
        "config_sha256": sha256_file(config_path),
        "runner_sha256": sha256_file(runner_path),
        "document_index": 0,
        "expert_smoke_m_values": [1, 64],
        "paper_claim_authorized": False,
    }
    for field, expected in expected_scalars.items():
        if artifact.get(field) != expected:
            raise ProtocolError(
                f"real-GPU acceptance field {field}={artifact.get(field)!r} != {expected!r}"
            )
    environment = artifact.get("environment")
    if not isinstance(environment, dict) or environment.get("gpu", {}).get(
        "uuid"
    ) != config["environment"]["gpu_uuid"]:
        raise ProtocolError("real-GPU acceptance GPU UUID mismatch")
    bindings = artifact.get("bindings")
    if not isinstance(bindings, dict):
        raise ProtocolError("real-GPU acceptance bindings are absent")
    if bindings.get("manifest_sha256") != config["data"]["manifest_sha256"]:
        raise ProtocolError("real-GPU acceptance sealed manifest mismatch")
    if bindings.get("bound_data_file_sha256") != {
        "calibration_manifest": config["data"]["calibration_manifest_sha256"],
        "provenance": config["data"]["provenance_sha256"],
        "historical_hash_registry": config["data"][
            "historical_hash_registry_sha256"
        ],
    }:
        raise ProtocolError("real-GPU acceptance data provenance bindings mismatch")
    if bindings.get("model_file_sha256") != config["model"]["file_sha256"]:
        raise ProtocolError("real-GPU acceptance model bindings mismatch")
    if not re.fullmatch(
        r"[0-9a-f]{64}", str(artifact.get("router_logits_sha256", ""))
    ):
        raise ProtocolError("real-GPU acceptance lacks real router output hash")
    expected_shape = [int(config["model"]["hidden_size"])]
    if artifact.get("expert_output_shapes") != {
        "1": expected_shape,
        "64": expected_shape,
    }:
        raise ProtocolError("real-GPU acceptance expert output shapes mismatch")
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "status": "PASS",
        "gpu_uuid": config["environment"]["gpu_uuid"],
    }


def _deadline_check(deadline_epoch: float, stage: str) -> None:
    if time.time() > deadline_epoch:
        raise TimeoutError(f"UNSOLVED_BUDGET during {stage}")


def _arm_parent_hard_deadline(deadline_epoch: float) -> Any:
    """Arm a process-wide wall-clock deadline covering parent post-processing."""

    if not hasattr(signal, "SIGALRM") or not hasattr(signal, "setitimer"):
        raise ProtocolError("hard parent watchdog requires POSIX SIGALRM/setitimer")
    remaining = deadline_epoch - time.time()
    if remaining <= 0:
        raise TimeoutError("UNSOLVED_BUDGET before parent watchdog")
    previous_handler = signal.getsignal(signal.SIGALRM)

    def deadline_handler(_signum: int, _frame: Any) -> None:
        raise TimeoutError("UNSOLVED_BUDGET: parent hard wall-clock watchdog fired")

    signal.signal(signal.SIGALRM, deadline_handler)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, remaining)
    if previous_timer != (0.0, 0.0):
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)
        raise ProtocolError(f"pre-existing real-time alarm is not allowed: {previous_timer}")
    return previous_handler


def _disarm_parent_hard_deadline(previous_handler: Any) -> None:
    signal.setitimer(signal.ITIMER_REAL, 0.0)
    signal.signal(signal.SIGALRM, previous_handler)


def capture_victims(
    *,
    model: Any,
    tokenizer: Any,
    config: Mapping[str, Any],
    repo_root: Path,
    output_dir: Path,
    deadline_epoch: float,
) -> list[dict[str, Any]]:
    import torch
    import torch.nn.functional as F

    data_cfg = config["data"]
    model_cfg = config["model"]
    documents = load_jsonl(repo_root / str(data_cfg["manifest"]))
    documents.sort(key=lambda row: int(row["document_index"]))
    expected_indices = list(range(int(data_cfg["document_count"])))
    actual_indices = [int(row["document_index"]) for row in documents]
    if actual_indices != expected_indices:
        raise ProtocolError(f"document indices are not exactly {expected_indices}")

    blocks = [layer.mlp for layer in model.model.layers]
    captured_by_layer: dict[int, Any] = {}
    handles = []

    def make_hook(layer_idx: int):
        def hook(_module: Any, inputs: tuple[Any, ...]) -> None:
            if len(inputs) != 1:
                raise ProtocolError(f"layer {layer_idx} hook received {len(inputs)} inputs")
            states = inputs[0]
            expected_shape = (
                1,
                int(data_cfg["window_tokens"]),
                int(model_cfg["hidden_size"]),
            )
            if tuple(states.shape) != expected_shape:
                raise ProtocolError(
                    f"layer {layer_idx} hidden shape {tuple(states.shape)} != {expected_shape}"
                )
            captured_by_layer[layer_idx] = states[0].detach().cpu().clone()

        return hook

    for layer_idx, block in enumerate(blocks):
        handles.append(block.register_forward_pre_hook(make_hook(layer_idx)))

    captures: list[dict[str, Any]] = []
    manifest_path = output_dir / "capture_manifest.jsonl"
    try:
        with manifest_path.open("x", encoding="utf-8") as manifest_out:
            for document in documents:
                text = str(document["text"])
                text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
                if text_hash != document["text_sha256"]:
                    raise ProtocolError(
                        f"document {document['document_index']} text hash mismatch"
                    )
                token_ids = tokenizer(
                    text, add_special_tokens=bool(data_cfg["add_special_tokens"])
                )["input_ids"]
                for offset in data_cfg["token_offsets"]:
                    _deadline_check(deadline_epoch, "victim capture")
                    offset = int(offset)
                    window_tokens = int(data_cfg["window_tokens"])
                    window = token_ids[offset : offset + window_tokens]
                    if len(window) != window_tokens:
                        raise ProtocolError(
                            f"document {document['document_index']} offset {offset} has only {len(window)} tokens"
                        )
                    captured_by_layer.clear()
                    input_ids = torch.tensor([window], dtype=torch.long, device="cuda")
                    with torch.inference_mode():
                        output = model(
                            input_ids=input_ids,
                            use_cache=False,
                            output_router_logits=True,
                            return_dict=True,
                        )
                    if len(captured_by_layer) != int(model_cfg["num_hidden_layers"]):
                        raise ProtocolError(
                            f"captured {len(captured_by_layer)} layers, expected {model_cfg['num_hidden_layers']}"
                        )
                    if output.router_logits is None or len(output.router_logits) != int(
                        model_cfg["num_hidden_layers"]
                    ):
                        raise ProtocolError("router logits capture is incomplete")

                    full_hidden_states = torch.stack(
                        [captured_by_layer[idx] for idx in range(len(blocks))], dim=0
                    )
                    selected_rows: list[Any] = []
                    routing_rows: list[Any] = []
                    router_hashes: list[str] = []
                    position = int(data_cfg["victim_position"])
                    for layer_idx, logits in enumerate(output.router_logits):
                        flat = logits.reshape(-1, logits.shape[-1])
                        victim_logits = flat[position]
                        probabilities = F.softmax(victim_logits, dim=-1, dtype=torch.float)
                        weights, experts = torch.topk(
                            probabilities,
                            k=int(model_cfg["num_experts_per_tok"]),
                            dim=-1,
                        )
                        direct_full = blocks[layer_idx].gate(
                            full_hidden_states[layer_idx].to("cuda").reshape(
                                int(data_cfg["window_tokens"]),
                                int(model_cfg["hidden_size"]),
                            )
                        )
                        if not torch.equal(direct_full, flat):
                            raise ProtocolError(
                                f"layer {layer_idx} same-M hook/gate logits do not exactly match native forward"
                            )
                        selected_rows.append(experts.detach().cpu())
                        routing_rows.append(weights.detach().cpu())
                        router_hashes.append(_tensor_sha256(victim_logits))

                    selected = torch.stack(selected_rows, dim=0)
                    routing = torch.stack(routing_rows, dim=0)
                    victim_id = f"doc{int(document['document_index']):03d}-offset{offset:04d}"
                    capture = {
                        "victim_id": victim_id,
                        "document_index": int(document["document_index"]),
                        "offset": offset,
                        "text_sha256": document["text_sha256"],
                        "window_token_ids": list(map(int, window)),
                        "window_token_ids_sha256": canonical_sha256(list(map(int, window))),
                        "full_hidden_states": full_hidden_states,
                        "full_hidden_states_sha256": _tensor_sha256(
                            full_hidden_states
                        ),
                        "selected_experts": selected,
                        "routing_weights": routing,
                        "router_logits_sha256_by_layer": router_hashes,
                    }
                    captures.append(capture)
                    write_jsonl_row(
                        manifest_out,
                        {
                            key: value
                            for key, value in capture.items()
                            if key
                            not in {
                                "full_hidden_states",
                                "selected_experts",
                                "routing_weights",
                            }
                        }
                        | {
                            "selected_experts": selected.tolist(),
                            "routing_weights": routing.float().tolist(),
                        },
                    )
                    manifest_out.flush()
            os.fsync(manifest_out.fileno())
    finally:
        for handle in handles:
            handle.remove()

    if len(captures) != int(data_cfg["victim_count"]):
        raise ProtocolError(
            f"captured {len(captures)} victims, expected {data_cfg['victim_count']}"
        )
    capture_path = output_dir / "captures.pt"
    if capture_path.exists():
        raise ProtocolError(f"refusing to overwrite {capture_path}")
    torch.save(captures, capture_path)
    return captures


def validate_capture_bundle(
    *,
    captures: Sequence[Mapping[str, Any]],
    manifest_path: Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    manifest_rows = load_jsonl(manifest_path)
    expected_count = int(config["data"]["victim_count"])
    if len(captures) != expected_count or len(manifest_rows) != expected_count:
        raise ProtocolError("capture bundle denominator is incomplete")
    manifest_by_id: dict[str, dict[str, Any]] = {}
    for row in manifest_rows:
        victim_id = str(row.get("victim_id"))
        if victim_id in manifest_by_id:
            raise ProtocolError(f"duplicate capture manifest victim {victim_id}")
        manifest_by_id[victim_id] = row

    summaries: list[dict[str, Any]] = []
    seen: set[str] = set()
    expected_shape = (
        int(config["model"]["num_hidden_layers"]),
        int(config["data"]["window_tokens"]),
        int(config["model"]["hidden_size"]),
    )
    expected_expert_shape = (
        int(config["model"]["num_hidden_layers"]),
        int(config["model"]["num_experts_per_tok"]),
    )
    for capture in captures:
        victim_id = str(capture["victim_id"])
        if victim_id in seen or victim_id not in manifest_by_id:
            raise ProtocolError(f"capture victim identity mismatch: {victim_id}")
        seen.add(victim_id)
        manifest = manifest_by_id[victim_id]
        hidden = capture["full_hidden_states"]
        selected = capture["selected_experts"]
        routing = capture["routing_weights"]
        if tuple(hidden.shape) != expected_shape:
            raise ProtocolError(f"{victim_id} full hidden shape mismatch")
        if tuple(selected.shape) != expected_expert_shape:
            raise ProtocolError(f"{victim_id} selected expert shape mismatch")
        hidden_hash = _tensor_sha256(hidden)
        if hidden_hash != capture["full_hidden_states_sha256"]:
            raise ProtocolError(f"{victim_id} capture hidden hash mismatch")
        summary = {
            "victim_id": victim_id,
            "document_index": int(capture["document_index"]),
            "offset": int(capture["offset"]),
            "text_sha256": capture["text_sha256"],
            "window_token_ids": list(map(int, capture["window_token_ids"])),
            "window_token_ids_sha256": capture["window_token_ids_sha256"],
            "full_hidden_states_sha256": hidden_hash,
            "selected_experts": selected.tolist(),
            "routing_weights_sha256": canonical_sha256(routing.float().tolist()),
            "router_logits_sha256_by_layer": list(
                capture["router_logits_sha256_by_layer"]
            ),
        }
        manifest_summary = {
            "victim_id": str(manifest["victim_id"]),
            "document_index": int(manifest["document_index"]),
            "offset": int(manifest["offset"]),
            "text_sha256": manifest["text_sha256"],
            "window_token_ids": list(map(int, manifest["window_token_ids"])),
            "window_token_ids_sha256": manifest["window_token_ids_sha256"],
            "full_hidden_states_sha256": manifest[
                "full_hidden_states_sha256"
            ],
            "selected_experts": manifest["selected_experts"],
            "routing_weights_sha256": canonical_sha256(
                manifest["routing_weights"]
            ),
            "router_logits_sha256_by_layer": manifest[
                "router_logits_sha256_by_layer"
            ],
        }
        if summary != manifest_summary:
            raise ProtocolError(f"{victim_id} capture/manifest semantic mismatch")
        summaries.append(summary)
    if seen != set(manifest_by_id):
        raise ProtocolError("capture/manifest victim sets differ")
    summaries.sort(key=lambda row: row["victim_id"])
    return {
        "capture_count": len(summaries),
        "capture_semantic_sha256": canonical_sha256(summaries),
    }


def run_numeric_sweep(
    *,
    model: Any,
    captures: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    output_dir: Path,
    deadline_epoch: float,
) -> dict[str, Any]:
    import torch

    intervention = config["intervention"]
    model_cfg = config["model"]
    m_values = [int(value) for value in intervention["m_values"]]
    reference_m = int(intervention["reference_m"])
    repeats = int(intervention["repeats"])
    if reference_m not in m_values:
        raise ProtocolError("reference M is absent from the frozen M grid")

    cell_count = 0
    unstable_cell_count = 0
    numerically_changed_cell_count = 0
    path = output_dir / "numeric_cells.jsonl"
    with path.open("x", encoding="utf-8") as out:
        for victim_idx, capture in enumerate(captures):
            full_hidden_states = capture["full_hidden_states"]
            selected = capture["selected_experts"]
            if tuple(full_hidden_states.shape) != (
                int(model_cfg["num_hidden_layers"]),
                int(config["data"]["window_tokens"]),
                int(model_cfg["hidden_size"]),
            ):
                raise ProtocolError(f"{capture['victim_id']} hidden capture shape mismatch")
            if tuple(selected.shape) != (
                int(model_cfg["num_hidden_layers"]),
                int(model_cfg["num_experts_per_tok"]),
            ):
                raise ProtocolError(f"{capture['victim_id']} expert capture shape mismatch")

            for layer_idx in range(int(model_cfg["num_hidden_layers"])):
                x = full_hidden_states[
                    layer_idx, int(config["data"]["victim_position"])
                ].to(device="cuda", dtype=torch.bfloat16)
                for expert_id_tensor in selected[layer_idx]:
                    expert_id = int(expert_id_tensor.item())
                    expert = model.model.layers[layer_idx].mlp.experts[expert_id]
                    per_m: dict[int, dict[str, Any]] = {}
                    representatives: dict[int, Any] = {}
                    for m_value in m_values:
                        _deadline_check(deadline_epoch, "numeric sweep")
                        batch = x.reshape(1, -1).repeat(m_value, 1)
                        hashes: list[str] = []
                        first_output = None
                        for _repeat in range(repeats):
                            with torch.inference_mode():
                                row = expert(batch)[0].detach()
                            if row.dtype != torch.bfloat16:
                                raise ProtocolError(
                                    f"expert output dtype {row.dtype} is not BF16"
                                )
                            if not bool(torch.isfinite(row).all().item()):
                                raise ProtocolError("non-finite expert output")
                            row_cpu = row.cpu().clone()
                            hashes.append(_tensor_sha256(row_cpu))
                            if first_output is None:
                                first_output = row_cpu
                        assert first_output is not None
                        representatives[m_value] = first_output
                        unique_hashes = sorted(set(hashes))
                        per_m[m_value] = {
                            "m": m_value,
                            "repeat_count": repeats,
                            "repeat_hashes": hashes,
                            "unique_repeat_hashes": unique_hashes,
                            "within_m_bitwise_stable": len(unique_hashes) == 1,
                            "representative_sha256": hashes[0],
                        }

                    reference = representatives[reference_m]
                    any_changed = False
                    all_stable = True
                    m_results: list[dict[str, Any]] = []
                    for m_value in m_values:
                        current = representatives[m_value]
                        delta = current.float() - reference.float()
                        changed_elements = _bitwise_changed_bf16_elements(
                            current, reference
                        )
                        changed = changed_elements > 0
                        any_changed = any_changed or changed
                        all_stable = all_stable and bool(
                            per_m[m_value]["within_m_bitwise_stable"]
                        )
                        m_results.append(
                            per_m[m_value]
                            | {
                                "cross_m_bitwise_equal_to_reference": not changed,
                                "changed_bf16_elements_to_reference": changed_elements,
                                "max_abs_delta_to_reference": float(
                                    delta.abs().max().item()
                                ),
                                "l2_delta_to_reference": float(
                                    torch.linalg.vector_norm(delta).item()
                                ),
                            }
                        )
                    cell_id = (
                        f"{capture['victim_id']}/L{layer_idx:02d}/E{expert_id:02d}"
                    )
                    row = {
                        "schema_version": "spectatorroute-phase0a-numeric-cell-v1",
                        "cell_id": cell_id,
                        "victim_id": capture["victim_id"],
                        "document_index": int(capture["document_index"]),
                        "offset": int(capture["offset"]),
                        "layer": layer_idx,
                        "expert_id": expert_id,
                        "hidden_row_sha256": _tensor_sha256(
                            full_hidden_states[
                                layer_idx, int(config["data"]["victim_position"])
                            ]
                        ),
                        "all_m_within_bitwise_stable": all_stable,
                        "any_cross_m_output_change": any_changed,
                        "m_results": m_results,
                    }
                    write_jsonl_row(out, row)
                    cell_count += 1
                    unstable_cell_count += int(not all_stable)
                    numerically_changed_cell_count += int(any_changed)
            out.flush()
            os.fsync(out.fileno())

    expected_cells = (
        int(config["data"]["victim_count"])
        * int(model_cfg["num_hidden_layers"])
        * int(model_cfg["num_experts_per_tok"])
    )
    if cell_count != expected_cells:
        raise ProtocolError(f"numeric cell count {cell_count} != {expected_cells}")
    return {
        "cell_count": cell_count,
        "unstable_cell_count": unstable_cell_count,
        "numerically_changed_cell_count": numerically_changed_cell_count,
        "numeric_cells_sha256": sha256_file(path),
    }


def worker_numeric(args: argparse.Namespace) -> int:
    import torch

    for variable in ("CUBLASLT_LOG_LEVEL", "CUBLASLT_LOG_MASK", "CUBLASLT_LOG_FILE"):
        if variable in os.environ:
            raise ProtocolError(f"numeric worker inherited unexpected {variable}")
    config, lock_info = verify_locked_invocation(args)
    repo_root = Path(args.repo_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    assert_no_foreign_gpu_processes(
        config,
        allowed_pids={os.getpid()},
        output_dir=output_dir,
        stage="numeric_before_cuda",
    )
    bindings = verify_static_bindings(config, repo_root)
    environment = verify_runtime_environment(
        config,
        contamination_output_dir=output_dir,
        contamination_stage="numeric_after_torch_import",
    )
    write_json_no_overwrite(
        output_dir / "environment.json",
        {
            "schema_version": "spectatorroute-phase0a-environment-v1",
            "observed": environment,
            "bindings": bindings,
            "config_sha256": sha256_file(Path(args.config)),
            "runner_sha256": sha256_file(Path(__file__)),
            "frozen_lock_sha256": lock_info["lock_sha256"],
        },
    )
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    model, tokenizer = load_frozen_model(config)
    captures = capture_victims(
        model=model,
        tokenizer=tokenizer,
        config=config,
        repo_root=repo_root,
        output_dir=output_dir,
        deadline_epoch=float(args.deadline_epoch),
    )
    capture_validation = validate_capture_bundle(
        captures=captures,
        manifest_path=output_dir / "capture_manifest.jsonl",
        config=config,
    )
    assert_no_foreign_gpu_processes(
        config,
        allowed_pids={os.getpid()},
        output_dir=output_dir,
        stage="numeric_before_sweep",
    )
    summary = run_numeric_sweep(
        model=model,
        captures=captures,
        config=config,
        output_dir=output_dir,
        deadline_epoch=float(args.deadline_epoch),
    )
    assert_no_foreign_gpu_processes(
        config,
        allowed_pids={os.getpid()},
        output_dir=output_dir,
        stage="numeric_after_sweep",
    )
    loaded_cublaslt = assert_loaded_cublaslt(config)
    write_json_no_overwrite(
        output_dir / "numeric_worker_status.json",
        {
            "status": "COMPLETE",
            "victim_count": len(captures),
            "captures_sha256": sha256_file(output_dir / "captures.pt"),
            "capture_manifest_sha256": sha256_file(
                output_dir / "capture_manifest.jsonl"
            ),
            "loaded_cublaslt_paths": loaded_cublaslt,
            **capture_validation,
            **summary,
        },
    )
    return 0


def worker_trace(args: argparse.Namespace) -> int:
    import torch

    config, _lock_info = verify_locked_invocation(args)
    repo_root = Path(args.repo_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    assert_no_foreign_gpu_processes(
        config,
        allowed_pids={os.getpid()},
        output_dir=output_dir,
        stage="trace_before_cuda",
    )
    verify_static_bindings(config, repo_root)
    verify_runtime_environment(
        config,
        contamination_output_dir=output_dir,
        contamination_stage="trace_after_torch_import",
    )
    expected_level = str(config["intervention"]["cublaslt_log_level"])
    expected_mask = str(config["intervention"]["cublaslt_log_mask"])
    if os.environ.get("CUBLASLT_LOG_LEVEL") != expected_level:
        raise ProtocolError("CUBLASLT_LOG_LEVEL is not frozen value")
    if os.environ.get("CUBLASLT_LOG_MASK") != expected_mask:
        raise ProtocolError("CUBLASLT_LOG_MASK is not frozen value")
    if not os.environ.get("CUBLASLT_LOG_FILE"):
        raise ProtocolError("CUBLASLT_LOG_FILE is not set")

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    numeric_status = load_json(output_dir / "numeric_worker_status.json")
    for artifact_name, status_field in (
        ("captures.pt", "captures_sha256"),
        ("capture_manifest.jsonl", "capture_manifest_sha256"),
        ("numeric_cells.jsonl", "numeric_cells_sha256"),
    ):
        observed = sha256_file(output_dir / artifact_name)
        if observed != numeric_status.get(status_field):
            raise ProtocolError(
                f"{artifact_name} hash {observed} != numeric status binding"
            )
    captures = torch.load(
        output_dir / "captures.pt", map_location="cpu", weights_only=True
    )
    if len(captures) != int(config["data"]["victim_count"]):
        raise ProtocolError("trace worker capture count mismatch")
    capture_validation = validate_capture_bundle(
        captures=captures,
        manifest_path=output_dir / "capture_manifest.jsonl",
        config=config,
    )
    if (
        capture_validation["capture_semantic_sha256"]
        != numeric_status.get("capture_semantic_sha256")
    ):
        raise ProtocolError("trace worker capture semantic hash mismatch")
    model, _tokenizer = load_frozen_model(config)
    model_cfg = config["model"]
    m_values = [int(value) for value in config["intervention"]["m_values"]]
    assert_no_foreign_gpu_processes(
        config,
        allowed_pids={os.getpid()},
        output_dir=output_dir,
        stage="trace_before_sweep",
    )

    call_count = 0
    index_path = output_dir / "trace_call_index.jsonl"
    with index_path.open("x", encoding="utf-8") as out:
        for capture in captures:
            full_hidden_states = capture["full_hidden_states"]
            selected = capture["selected_experts"]
            for layer_idx in range(int(model_cfg["num_hidden_layers"])):
                x = full_hidden_states[
                    layer_idx, int(config["data"]["victim_position"])
                ].to(device="cuda", dtype=torch.bfloat16)
                for expert_id_tensor in selected[layer_idx]:
                    expert_id = int(expert_id_tensor.item())
                    expert = model.model.layers[layer_idx].mlp.experts[expert_id]
                    cell_id = (
                        f"{capture['victim_id']}/L{layer_idx:02d}/E{expert_id:02d}"
                    )
                    for m_value in m_values:
                        _deadline_check(float(args.deadline_epoch), "trace sweep")
                        batch = x.reshape(1, -1).repeat(m_value, 1)
                        with torch.inference_mode():
                            result = expert(batch)
                        if result.dtype != torch.bfloat16 or not bool(
                            torch.isfinite(result).all().item()
                        ):
                            raise ProtocolError("trace expert output is invalid")
                        write_jsonl_row(
                            out,
                            {
                                "call_index": call_count,
                                "cell_id": cell_id,
                                "victim_id": capture["victim_id"],
                                "layer": layer_idx,
                                "expert_id": expert_id,
                                "m": m_value,
                                "trace_row0_output_sha256": _tensor_sha256(
                                    result[0]
                                ),
                            },
                        )
                        call_count += 1
            out.flush()
            os.fsync(out.fileno())
    torch.cuda.synchronize()
    assert_no_foreign_gpu_processes(
        config,
        allowed_pids={os.getpid()},
        output_dir=output_dir,
        stage="trace_after_sweep",
    )
    loaded_cublaslt = assert_loaded_cublaslt(config)

    expected_calls = (
        int(config["data"]["victim_count"])
        * int(model_cfg["num_hidden_layers"])
        * int(model_cfg["num_experts_per_tok"])
        * len(m_values)
    )
    if call_count != expected_calls:
        raise ProtocolError(f"trace call count {call_count} != {expected_calls}")
    write_json_no_overwrite(
        output_dir / "trace_worker_status.json",
        {
            "status": "COMPLETE",
            "trace_call_count": call_count,
            "expected_gemm_trace_count": call_count * 3,
            "trace_call_index_sha256": sha256_file(index_path),
            "loaded_cublaslt_paths": loaded_cublaslt,
            **capture_validation,
        },
    )
    return 0


def parse_trace_artifacts(
    *, config: Mapping[str, Any], output_dir: Path, trace_path: Path
) -> dict[str, Any]:
    trace_status = load_json(output_dir / "trace_worker_status.json")
    if trace_status.get("status") != "COMPLETE":
        raise ProtocolError("trace worker status is not COMPLETE")
    index_path = output_dir / "trace_call_index.jsonl"
    if sha256_file(index_path) != trace_status.get("trace_call_index_sha256"):
        raise ProtocolError("trace call index hash does not match worker status")
    indices = load_jsonl(index_path)
    if [int(row.get("call_index", -1)) for row in indices] != list(
        range(len(indices))
    ):
        raise ProtocolError("trace call indices are not contiguous and ordered")
    for row in indices:
        if not re.fullmatch(
            r"[0-9a-f]{64}", str(row.get("trace_row0_output_sha256", ""))
        ):
            raise ProtocolError("trace call lacks a valid row-0 output hash")
    if int(trace_status.get("trace_call_count", -1)) != len(indices):
        raise ProtocolError("trace worker call count mismatch")
    records: list[dict[str, Any]] = []
    with trace_path.open("r", encoding="utf-8", errors="strict") as handle:
        for line in handle:
            parsed = parse_cublaslt_trace_line(line)
            if parsed is not None:
                records.append(parsed)
    expected_records = len(indices) * 3
    if len(records) != expected_records:
        raise ProtocolError(
            f"found {len(records)} matmul traces for {len(indices)} calls; expected {expected_records}"
        )
    if int(trace_status.get("expected_gemm_trace_count", -1)) != len(records):
        raise ProtocolError("trace worker expected GEMM count mismatch")

    model_cfg = config["model"]
    regime_path = output_dir / "regime_cells.jsonl"
    with regime_path.open("x", encoding="utf-8") as out:
        for call_idx, index in enumerate(indices):
            triplet = records[call_idx * 3 : (call_idx + 1) * 3]
            roles = validate_projection_triplet(
                triplet,
                m_value=int(index["m"]),
                hidden_size=int(model_cfg["hidden_size"]),
                intermediate_size=int(model_cfg["intermediate_size"]),
            )
            signatures = [
                algorithm_signature(record, role)
                for record, role in zip(triplet, roles)
            ]
            write_jsonl_row(
                out,
                {
                    "schema_version": "spectatorroute-phase0a-regime-cell-v1",
                    **index,
                    "raw_trace_records": triplet,
                    "algorithm_signatures": signatures,
                    "algorithm_signature_sha256": canonical_sha256(signatures),
                },
            )
        out.flush()
        os.fsync(out.fileno())
    return {
        "trace_call_count": len(indices),
        "gemm_trace_count": len(records),
        "raw_trace_sha256": sha256_file(trace_path),
        "regime_cells_sha256": sha256_file(regime_path),
    }


def _require_sha256(value: Any, context: str) -> str:
    text = str(value)
    if not re.fullmatch(r"[0-9a-f]{64}", text):
        raise ProtocolError(f"{context} is not one lowercase SHA-256")
    return text


def validate_numeric_cell(
    row: Mapping[str, Any], config: Mapping[str, Any]
) -> tuple[dict[int, dict[str, Any]], bool, bool]:
    if row.get("schema_version") != "spectatorroute-phase0a-numeric-cell-v1":
        raise ProtocolError("unknown numeric-cell schema")
    cell_id = str(row["cell_id"])
    expected_cell_id = (
        f"{row['victim_id']}/L{int(row['layer']):02d}/E{int(row['expert_id']):02d}"
    )
    if cell_id != expected_cell_id:
        raise ProtocolError(f"numeric cell identity {cell_id} != {expected_cell_id}")
    _require_sha256(row.get("hidden_row_sha256"), f"{cell_id} hidden row hash")
    m_values = [int(value) for value in config["intervention"]["m_values"]]
    repeats = int(config["intervention"]["repeats"])
    raw_results = row.get("m_results")
    if not isinstance(raw_results, list) or len(raw_results) != len(m_values):
        raise ProtocolError(f"{cell_id} has wrong numeric M result count")
    by_m: dict[int, dict[str, Any]] = {}
    for result in raw_results:
        m_value = int(result.get("m", -1))
        if m_value in by_m:
            raise ProtocolError(f"{cell_id} duplicate numeric M={m_value}")
        hashes = result.get("repeat_hashes")
        if not isinstance(hashes, list) or len(hashes) != repeats:
            raise ProtocolError(
                f"{cell_id} M={m_value} does not have exactly {repeats} hashes"
            )
        hashes = [
            _require_sha256(value, f"{cell_id} M={m_value} repeat hash")
            for value in hashes
        ]
        unique = sorted(set(hashes))
        stable = len(unique) == 1
        if int(result.get("repeat_count", -1)) != repeats:
            raise ProtocolError(f"{cell_id} M={m_value} repeat_count mismatch")
        if result.get("unique_repeat_hashes") != unique:
            raise ProtocolError(f"{cell_id} M={m_value} unique hashes mismatch")
        if bool(result.get("within_m_bitwise_stable")) != stable:
            raise ProtocolError(f"{cell_id} M={m_value} stable flag mismatch")
        if result.get("representative_sha256") != hashes[0]:
            raise ProtocolError(f"{cell_id} M={m_value} representative mismatch")
        by_m[m_value] = dict(result) | {
            "repeat_hashes": hashes,
            "recomputed_stable": stable,
            "recomputed_representative_sha256": hashes[0],
        }
    if list(by_m) != m_values:
        raise ProtocolError(
            f"{cell_id} numeric M order/grid {list(by_m)} != frozen {m_values}"
        )

    reference_m = int(config["intervention"]["reference_m"])
    reference_hash = by_m[reference_m]["recomputed_representative_sha256"]
    all_stable = True
    any_changed = False
    for m_value in m_values:
        result = by_m[m_value]
        equal = result["recomputed_representative_sha256"] == reference_hash
        if bool(result.get("cross_m_bitwise_equal_to_reference")) != equal:
            raise ProtocolError(f"{cell_id} M={m_value} cross-M flag mismatch")
        changed_elements = int(result.get("changed_bf16_elements_to_reference", -1))
        max_abs = float(result.get("max_abs_delta_to_reference", math.nan))
        l2 = float(result.get("l2_delta_to_reference", math.nan))
        if not math.isfinite(max_abs) or not math.isfinite(l2):
            raise ProtocolError(f"{cell_id} M={m_value} non-finite delta metric")
        if equal and (changed_elements != 0 or max_abs != 0.0 or l2 != 0.0):
            raise ProtocolError(f"{cell_id} M={m_value} equal hash has nonzero delta")
        if not equal and (changed_elements <= 0 or max_abs <= 0.0 or l2 <= 0.0):
            raise ProtocolError(f"{cell_id} M={m_value} changed hash has zero delta")
        all_stable = all_stable and bool(result["recomputed_stable"])
        any_changed = any_changed or not equal
        result["recomputed_equal_to_reference"] = equal
    if bool(row.get("all_m_within_bitwise_stable")) != all_stable:
        raise ProtocolError(f"{cell_id} aggregate stable flag mismatch")
    if bool(row.get("any_cross_m_output_change")) != any_changed:
        raise ProtocolError(f"{cell_id} aggregate changed flag mismatch")
    return by_m, all_stable, any_changed


def validate_regime_cell(
    row: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, Any]:
    if row.get("schema_version") != "spectatorroute-phase0a-regime-cell-v1":
        raise ProtocolError("unknown regime-cell schema")
    cell_id = str(row["cell_id"])
    expected_cell_id = (
        f"{row['victim_id']}/L{int(row['layer']):02d}/E{int(row['expert_id']):02d}"
    )
    if cell_id != expected_cell_id:
        raise ProtocolError(f"regime cell identity {cell_id} != {expected_cell_id}")
    m_value = int(row["m"])
    if m_value not in [int(value) for value in config["intervention"]["m_values"]]:
        raise ProtocolError(f"{cell_id} has non-frozen regime M={m_value}")
    raw_records = row.get("raw_trace_records")
    if not isinstance(raw_records, list):
        raise ProtocolError(f"{cell_id} M={m_value} lacks raw trace records")
    roles = validate_projection_triplet(
        raw_records,
        m_value=m_value,
        hidden_size=int(config["model"]["hidden_size"]),
        intermediate_size=int(config["model"]["intermediate_size"]),
    )
    recomputed_signatures = [
        algorithm_signature(record, role)
        for record, role in zip(raw_records, roles)
    ]
    if row.get("algorithm_signatures") != recomputed_signatures:
        raise ProtocolError(f"{cell_id} M={m_value} algorithm signatures mismatch")
    recomputed_digest = canonical_sha256(recomputed_signatures)
    if row.get("algorithm_signature_sha256") != recomputed_digest:
        raise ProtocolError(f"{cell_id} M={m_value} signature digest mismatch")
    trace_output_hash = _require_sha256(
        row.get("trace_row0_output_sha256"),
        f"{cell_id} M={m_value} trace output hash",
    )
    return {
        "algorithm_signatures": recomputed_signatures,
        "algorithm_signature_sha256": recomputed_digest,
        "trace_row0_output_sha256": trace_output_hash,
    }


def analyze_validated_cell(
    *,
    cell_id: str,
    numeric_by_m: Mapping[int, Mapping[str, Any]],
    regime_by_m: Mapping[int, Mapping[str, Any]],
    m_values: Sequence[int],
    reference_m: int,
) -> dict[str, Any]:
    reference_regime = regime_by_m[reference_m][
        "algorithm_signature_sha256"
    ]
    joint_ms: list[int] = []
    output_changed_ms: list[int] = []
    regime_changed_ms: list[int] = []
    comparisons: list[dict[str, Any]] = []
    for m_value in m_values:
        output_hash = numeric_by_m[m_value][
            "recomputed_representative_sha256"
        ]
        trace_output_hash = regime_by_m[m_value]["trace_row0_output_sha256"]
        if trace_output_hash != output_hash:
            raise ProtocolError(
                f"{cell_id} M={m_value} traced output hash does not match numeric representative"
            )
        output_changed = not bool(
            numeric_by_m[m_value]["recomputed_equal_to_reference"]
        )
        regime_changed = (
            regime_by_m[m_value]["algorithm_signature_sha256"]
            != reference_regime
        )
        if output_changed:
            output_changed_ms.append(m_value)
        if regime_changed:
            regime_changed_ms.append(m_value)
        if output_changed and regime_changed:
            joint_ms.append(m_value)
        comparisons.append(
            {
                "m": m_value,
                "output_changed": output_changed,
                "regime_changed": regime_changed,
                "joint_changed": output_changed and regime_changed,
                "output_sha256": output_hash,
                "trace_output_sha256": trace_output_hash,
                "algorithm_signature_sha256": regime_by_m[m_value][
                    "algorithm_signature_sha256"
                ],
                "changed_bf16_elements_to_reference": numeric_by_m[m_value][
                    "changed_bf16_elements_to_reference"
                ],
                "max_abs_delta_to_reference": numeric_by_m[m_value][
                    "max_abs_delta_to_reference"
                ],
                "l2_delta_to_reference": numeric_by_m[m_value][
                    "l2_delta_to_reference"
                ],
            }
        )
    return {
        "output_changed_ms": output_changed_ms,
        "regime_changed_ms": regime_changed_ms,
        "joint_changed_ms": joint_ms,
        "comparisons": comparisons,
    }


def phase0a_decision(*, unstable_cells: int, positive_victims: int, minimum: int) -> str:
    if minimum != 8:
        raise ProtocolError(f"unfrozen positive-victim threshold: {minimum}")
    if unstable_cells > 0:
        return "INVALID_WITHIN_M_NONDETERMINISM"
    if positive_victims >= minimum:
        return "PASS_TO_PHASE0B"
    return "KILL_CURRENT_STACK"


def aggregate_gate(
    *, config: Mapping[str, Any], output_dir: Path
) -> dict[str, Any]:
    import torch

    numeric_path = output_dir / "numeric_cells.jsonl"
    regime_path = output_dir / "regime_cells.jsonl"
    numeric_status = load_json(output_dir / "numeric_worker_status.json")
    trace_status = load_json(output_dir / "trace_worker_status.json")
    if numeric_status.get("status") != "COMPLETE":
        raise ProtocolError("numeric worker status is not COMPLETE")
    if trace_status.get("status") != "COMPLETE":
        raise ProtocolError("trace worker status is not COMPLETE")
    artifact_bindings = (
        ("captures.pt", "captures_sha256"),
        ("capture_manifest.jsonl", "capture_manifest_sha256"),
        ("numeric_cells.jsonl", "numeric_cells_sha256"),
    )
    for artifact_name, status_field in artifact_bindings:
        observed_hash = sha256_file(output_dir / artifact_name)
        if observed_hash != numeric_status.get(status_field):
            raise ProtocolError(
                f"aggregate saw {artifact_name} hash mismatch against numeric status"
            )
    expected_cublas_path = str(Path(config["environment"]["cublaslt"]["path"]).resolve())
    for name, status in (("numeric", numeric_status), ("trace", trace_status)):
        if status.get("loaded_cublaslt_paths") != [expected_cublas_path]:
            raise ProtocolError(f"{name} worker libcublasLt mapping mismatch")

    captures = torch.load(
        output_dir / "captures.pt", map_location="cpu", weights_only=True
    )
    capture_validation = validate_capture_bundle(
        captures=captures,
        manifest_path=output_dir / "capture_manifest.jsonl",
        config=config,
    )
    for name, status in (("numeric", numeric_status), ("trace", trace_status)):
        if (
            status.get("capture_semantic_sha256")
            != capture_validation["capture_semantic_sha256"]
        ):
            raise ProtocolError(f"{name} worker capture semantic binding mismatch")

    expected_cells_from_capture: dict[str, dict[str, Any]] = {}
    victim_position = int(config["data"]["victim_position"])
    for capture in captures:
        victim_id = str(capture["victim_id"])
        for layer_idx in range(int(config["model"]["num_hidden_layers"])):
            hidden_hash = _tensor_sha256(
                capture["full_hidden_states"][layer_idx, victim_position]
            )
            for expert_tensor in capture["selected_experts"][layer_idx]:
                expert_id = int(expert_tensor.item())
                cell_id = f"{victim_id}/L{layer_idx:02d}/E{expert_id:02d}"
                if cell_id in expected_cells_from_capture:
                    raise ProtocolError(f"duplicate capture-derived cell {cell_id}")
                expected_cells_from_capture[cell_id] = {
                    "victim_id": victim_id,
                    "document_index": int(capture["document_index"]),
                    "offset": int(capture["offset"]),
                    "layer": layer_idx,
                    "expert_id": expert_id,
                    "hidden_row_sha256": hidden_hash,
                }

    numeric_rows = load_jsonl(numeric_path)
    regime_rows = load_jsonl(regime_path)
    m_values = [int(value) for value in config["intervention"]["m_values"]]
    reference_m = int(config["intervention"]["reference_m"])
    expected_cells = (
        int(config["data"]["victim_count"])
        * int(config["model"]["num_hidden_layers"])
        * int(config["model"]["num_experts_per_tok"])
    )
    if len(numeric_rows) != expected_cells:
        raise ProtocolError(f"numeric rows {len(numeric_rows)} != {expected_cells}")
    if len(regime_rows) != expected_cells * len(m_values):
        raise ProtocolError(
            f"regime rows {len(regime_rows)} != {expected_cells * len(m_values)}"
        )

    if int(numeric_status.get("cell_count", -1)) != expected_cells:
        raise ProtocolError("numeric worker status cell count mismatch")
    if int(numeric_status.get("victim_count", -1)) != int(
        config["data"]["victim_count"]
    ):
        raise ProtocolError("numeric worker status victim count mismatch")
    if int(trace_status.get("trace_call_count", -1)) != expected_cells * len(
        m_values
    ):
        raise ProtocolError("trace worker status call count mismatch")

    regime_map: dict[tuple[str, int], dict[str, Any]] = {}
    for row in regime_rows:
        key = (str(row["cell_id"]), int(row["m"]))
        if key in regime_map:
            raise ProtocolError(f"duplicate regime row {key}")
        recomputed = validate_regime_cell(row, config)
        regime_map[key] = dict(row) | recomputed

    unstable_cells = 0
    output_changed_cells = 0
    regime_changed_cells = 0
    joint_positive_cells = 0
    victim_positive_cells: dict[str, list[str]] = {}
    analyzed_path = output_dir / "analyzed_cells.jsonl"
    with analyzed_path.open("x", encoding="utf-8") as out:
        seen_cells: set[str] = set()
        for numeric in numeric_rows:
            cell_id = str(numeric["cell_id"])
            if cell_id in seen_cells:
                raise ProtocolError(f"duplicate numeric cell {cell_id}")
            expected_metadata = expected_cells_from_capture.get(cell_id)
            observed_metadata = {
                "victim_id": str(numeric["victim_id"]),
                "document_index": int(numeric["document_index"]),
                "offset": int(numeric["offset"]),
                "layer": int(numeric["layer"]),
                "expert_id": int(numeric["expert_id"]),
                "hidden_row_sha256": str(numeric["hidden_row_sha256"]),
            }
            if expected_metadata != observed_metadata:
                raise ProtocolError(
                    f"numeric cell {cell_id} does not match captured route/hidden identity"
                )
            seen_cells.add(cell_id)
            numeric_by_m, all_stable, any_output_changed = validate_numeric_cell(
                numeric, config
            )
            regime_by_m = {
                m_value: regime_map[(cell_id, m_value)] for m_value in m_values
            }
            analysis = analyze_validated_cell(
                cell_id=cell_id,
                numeric_by_m=numeric_by_m,
                regime_by_m=regime_by_m,
                m_values=m_values,
                reference_m=reference_m,
            )
            output_changed_ms = analysis["output_changed_ms"]
            regime_changed_ms = analysis["regime_changed_ms"]
            joint_ms = analysis["joint_changed_ms"]
            comparisons = analysis["comparisons"]
            if bool(output_changed_ms) != any_output_changed:
                raise ProtocolError(f"{cell_id} recomputed output-change mismatch")
            positive = all_stable and bool(joint_ms)
            unstable_cells += int(not all_stable)
            output_changed_cells += int(bool(output_changed_ms))
            regime_changed_cells += int(bool(regime_changed_ms))
            joint_positive_cells += int(positive)
            if positive:
                victim_positive_cells.setdefault(str(numeric["victim_id"]), []).append(
                    cell_id
                )
            write_jsonl_row(
                out,
                {
                    "schema_version": "spectatorroute-phase0a-analyzed-cell-v1",
                    "cell_id": cell_id,
                    "victim_id": numeric["victim_id"],
                    "layer": numeric["layer"],
                    "expert_id": numeric["expert_id"],
                    "all_m_within_bitwise_stable": all_stable,
                    "output_changed_ms": output_changed_ms,
                    "regime_changed_ms": regime_changed_ms,
                    "joint_changed_ms": joint_ms,
                    "positive": positive,
                    "comparisons": comparisons,
                },
            )
        out.flush()
        os.fsync(out.fileno())
    if seen_cells != set(expected_cells_from_capture):
        raise ProtocolError("numeric cells do not exactly cover capture-derived cells")

    all_victim_ids = sorted({str(row["victim_id"]) for row in numeric_rows})
    if len(all_victim_ids) != int(config["data"]["victim_count"]):
        raise ProtocolError("victim denominator is incomplete")
    victims_path = output_dir / "victims.jsonl"
    with victims_path.open("x", encoding="utf-8") as out:
        for victim_id in all_victim_ids:
            cells = sorted(victim_positive_cells.get(victim_id, []))
            write_jsonl_row(
                out,
                {
                    "victim_id": victim_id,
                    "positive": bool(cells),
                    "positive_cell_count": len(cells),
                    "positive_cells": cells,
                },
            )
        out.flush()
        os.fsync(out.fileno())

    positive_victims = len(victim_positive_cells)
    minimum = int(config["gate"]["minimum_distinct_positive_victims"])
    decision = phase0a_decision(
        unstable_cells=unstable_cells,
        positive_victims=positive_victims,
        minimum=minimum,
    )
    if int(numeric_status.get("unstable_cell_count", -1)) != unstable_cells:
        raise ProtocolError("numeric worker unstable-cell count mismatch")
    if int(numeric_status.get("numerically_changed_cell_count", -1)) != output_changed_cells:
        raise ProtocolError("numeric worker changed-cell count mismatch")
    return {
        "decision": decision,
        "evidence_boundary": config["evidence_boundary"],
        "victim_denominator": len(all_victim_ids),
        "minimum_distinct_positive_victims": minimum,
        "positive_victim_count": positive_victims,
        "positive_victim_ids": sorted(victim_positive_cells),
        "numeric_cell_count": len(numeric_rows),
        "unstable_cell_count": unstable_cells,
        "output_changed_cell_count": output_changed_cells,
        "regime_changed_cell_count": regime_changed_cells,
        "joint_positive_cell_count": joint_positive_cells,
        "analyzed_cells_sha256": sha256_file(analyzed_path),
        "victims_sha256": sha256_file(victims_path),
        "phase0b_authorized": decision == "PASS_TO_PHASE0B",
        "paper_claim_authorized": False,
    }


def write_verdict_markdown(output_dir: Path, summary: Mapping[str, Any]) -> None:
    decision = summary["decision"]
    if decision == "PASS_TO_PHASE0B":
        plain = (
            "Phase-0A 仅通过 arithmetic capability gate；只授权按原冻结协议执行 Phase-0B。"
            "它不是 prompt-only attack、downstream flip 或论文方法证据。"
        )
    elif decision == "KILL_CURRENT_STACK":
        plain = (
            "当前 RTX 5090 + Transformers eager OLMoE stack 未达到 frozen capability gate；"
            "不得换 M、victim、模型或阈值救活 N05。"
        )
    else:
        plain = "运行完整性失败；不得把任何部分信号解释为 positive evidence。"
    content = f"""# SpectatorRoute N05 Phase-0A verdict

> Decision: `{decision}`  
> Evidence tier: `SINGLE-GPU PRETRAINED EXPERT ARITHMETIC CAPABILITY ONLY`

{plain}

- Frozen victims: `{summary['victim_denominator']}`
- Positive victims: `{summary['positive_victim_count']}`
- Required positives: `{summary['minimum_distinct_positive_victims']}`
- Numeric cells: `{summary['numeric_cell_count']}`
- Cells with actual algorithm-regime change: `{summary['regime_changed_cell_count']}`
- Cells with cross-M BF16 output change: `{summary['output_changed_cell_count']}`
- Joint-positive cells: `{summary['joint_positive_cell_count']}`
- Within-M unstable cells: `{summary['unstable_cell_count']}`

禁止把本结果写成 EP/NCCL/RDMA、production serving、security exploit、CCF-B method GO 或正式科学结论。
"""
    path = output_dir / "VERDICT.md"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def _run_worker(
    *,
    worker: str,
    args: argparse.Namespace,
    config: Mapping[str, Any],
    deadline_epoch: float,
    extra_env: Mapping[str, str] | None = None,
) -> None:
    snapshot_dir = Path(args.output_dir).resolve() / "frozen_inputs"
    command = [
        sys.executable,
        str(snapshot_dir / "run_phase0a_5090.py"),
        "--worker",
        worker,
        "--config",
        str(snapshot_dir / "phase0a_5090_v1.json"),
        "--repo-root",
        str(Path(args.repo_root).resolve()),
        "--output-dir",
        str(Path(args.output_dir).resolve()),
        "--deadline-epoch",
        str(deadline_epoch),
        "--frozen-lock",
        str(snapshot_dir / "FROZEN_RUN_LOCK.json"),
        "--frozen-lock-sha256",
        str(args.frozen_lock_sha256),
    ]
    environment = os.environ.copy()
    if extra_env:
        environment.update(extra_env)
    remaining = deadline_epoch - time.time()
    if remaining <= 0:
        raise TimeoutError(f"UNSOLVED_BUDGET before {worker} worker")
    process = subprocess.Popen(command, env=environment)
    monitor_path = (
        Path(args.output_dir).resolve() / f"gpu_process_monitor_{worker}.jsonl"
    )

    def terminate_owned_worker() -> None:
        if process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=10.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10.0)

    try:
        with monitor_path.open("x", encoding="utf-8") as monitor:
            sample_index = 0
            while True:
                remaining = deadline_epoch - time.time()
                if remaining <= 0:
                    raise TimeoutError(
                        f"UNSOLVED_BUDGET: hard timeout in {worker} worker"
                    )
                try:
                    return_code = process.wait(timeout=min(1.0, remaining))
                    break
                except subprocess.TimeoutExpired:
                    identity = _nvidia_identity()
                    observed, foreign, unexpected = _classify_gpu_process_scope(
                        config, allowed_pids={process.pid}
                    )
                    uuid_drift = identity["uuid"] != config["environment"]["gpu_uuid"]
                    status = (
                        "CLEAN"
                        if not foreign and not unexpected and not uuid_drift
                        else "INVALID_CONTAMINATION"
                    )
                    write_jsonl_row(
                        monitor,
                        {
                            "schema_version": "spectatorroute-phase0a-gpu-process-monitor-v1",
                            "worker": worker,
                            "sample_index": sample_index,
                            "epoch": time.time(),
                            "status": status,
                            "worker_pid": process.pid,
                            "gpu_identity": identity,
                            "observed_processes": observed,
                            "foreign_processes": foreign,
                            "unexpected_gpu_uuids": unexpected,
                            "gpu_uuid_drift": uuid_drift,
                        },
                    )
                    monitor.flush()
                    sample_index += 1
                    if status != "CLEAN":
                        os.fsync(monitor.fileno())
                        raise ContaminationError(
                            f"continuous GPU monitor detected contamination in {worker}"
                        )
            monitor.flush()
            os.fsync(monitor.fileno())
    except BaseException:
        terminate_owned_worker()
        raise
    if return_code == 3:
        raise TimeoutError(f"UNSOLVED_BUDGET in {worker} worker")
    if return_code != 0:
        contamination_snapshots = sorted(
            Path(args.output_dir).resolve().glob("gpu_processes_*.json")
        )
        for snapshot in contamination_snapshots:
            if load_json(snapshot).get("status") == "INVALID_CONTAMINATION":
                raise ContaminationError(
                    f"{worker} worker detected GPU contamination; see {snapshot.name}"
                )
        raise ProtocolError(
            f"{worker} worker exited with status {return_code}"
        )


def _orchestrate_inner(args: argparse.Namespace) -> int:
    config, lock_info = verify_locked_invocation(args)
    config_path = Path(args.config).resolve()
    repo_root = Path(args.repo_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    if not args.acceptance_artifact:
        raise ProtocolError(
            "formal Phase-0A requires --acceptance-artifact from exact real-GPU preflight"
        )
    acceptance_info = verify_real_gpu_acceptance_artifact(
        path=Path(args.acceptance_artifact).resolve(),
        config=config,
        lock_info=lock_info,
        config_path=config_path,
        runner_path=Path(__file__).resolve(),
    )
    if output_dir.exists():
        raise ProtocolError(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    assert_no_foreign_gpu_processes(
        config,
        allowed_pids=set(),
        output_dir=output_dir,
        stage="parent_start",
    )
    snapshot_dir = output_dir / "frozen_inputs"
    snapshot_dir.mkdir()
    snapshot_sources = {
        "run_phase0a_5090.py": Path(__file__).resolve(),
        "phase0a_5090_v1.json": config_path,
        "FROZEN_RUN_LOCK.json": Path(args.frozen_lock).resolve(),
        "N05_PHASE0_FROZEN_PROTOCOL_20260729.md": repo_root
        / "docs/ideas/spectatorroute/N05_PHASE0_FROZEN_PROTOCOL_20260729.md",
        "test_phase0a_5090.py": repo_root
        / "docs/ideas/spectatorroute/experiments/test_phase0a_5090.py",
    }
    snapshot_hashes: dict[str, str] = {}
    for target_name, source in snapshot_sources.items():
        target = snapshot_dir / target_name
        shutil.copyfile(source, target)
        target.chmod(0o444)
        snapshot_hashes[target_name] = sha256_file(target)
    if snapshot_hashes["FROZEN_RUN_LOCK.json"] != lock_info["lock_sha256"]:
        raise ProtocolError("frozen lock snapshot mismatch")
    start_epoch = time.time()
    deadline_epoch = start_epoch + int(config["gate"]["max_gpu_seconds"])
    write_json_no_overwrite(
        output_dir / "run_request.json",
        {
            "schema_version": "spectatorroute-phase0a-run-request-v1",
            "started_at_epoch": start_epoch,
            "deadline_epoch": deadline_epoch,
            "config": str(config_path),
            "config_sha256": sha256_file(config_path),
            "repo_root": str(repo_root),
            "output_dir": str(output_dir),
            "runner_sha256": sha256_file(Path(__file__)),
            "frozen_lock_sha256": lock_info["lock_sha256"],
            "frozen_lock": str(Path(args.frozen_lock).resolve()),
            "frozen_snapshot_sha256": snapshot_hashes,
            "real_gpu_acceptance": acceptance_info,
        },
    )
    previous_alarm_handler = _arm_parent_hard_deadline(deadline_epoch)
    try:
        _run_worker(
            worker="numeric",
            args=args,
            config=config,
            deadline_epoch=deadline_epoch,
        )
        assert_no_foreign_gpu_processes(
            config,
            allowed_pids=set(),
            output_dir=output_dir,
            stage="parent_after_numeric",
        )
        trace_template = output_dir / "cublaslt_%i.log"
        _run_worker(
            worker="trace",
            args=args,
            config=config,
            deadline_epoch=deadline_epoch,
            extra_env={
                "CUBLASLT_LOG_LEVEL": str(
                    config["intervention"]["cublaslt_log_level"]
                ),
                "CUBLASLT_LOG_MASK": str(
                    config["intervention"]["cublaslt_log_mask"]
                ),
                "CUBLASLT_LOG_FILE": str(trace_template),
            },
        )
        assert_no_foreign_gpu_processes(
            config,
            allowed_pids=set(),
            output_dir=output_dir,
            stage="parent_after_trace",
        )
        # Re-verify the same external seal after both workers and immediately
        # before parsing/aggregation.  Mid-run edits therefore fail closed.
        verify_locked_invocation(args)
        for target_name, expected_hash in snapshot_hashes.items():
            observed_hash = sha256_file(snapshot_dir / target_name)
            if observed_hash != expected_hash:
                raise ProtocolError(
                    f"frozen snapshot {target_name} changed during the run"
                )
        trace_paths = sorted(output_dir.glob("cublaslt_*.log"))
        if len(trace_paths) != 1:
            raise ProtocolError(f"expected one cuBLASLt log, found {trace_paths}")
        trace_info = parse_trace_artifacts(
            config=config, output_dir=output_dir, trace_path=trace_paths[0]
        )
        summary = aggregate_gate(config=config, output_dir=output_dir)
        assert_no_foreign_gpu_processes(
            config,
            allowed_pids=set(),
            output_dir=output_dir,
            stage="parent_after_aggregate",
        )
        summary.update(trace_info)
        summary.update(
            {
                "schema_version": "spectatorroute-phase0a-summary-v1",
                "wall_seconds": time.time() - start_epoch,
                "within_frozen_budget": time.time() <= deadline_epoch,
                "config_sha256": sha256_file(config_path),
                "runner_sha256": sha256_file(Path(__file__)),
                "frozen_lock_sha256": lock_info["lock_sha256"],
                "completion_sentinel_required": "COMPLETE.json",
            }
        )
        if not summary["within_frozen_budget"]:
            summary["decision"] = "UNSOLVED_BUDGET"
            summary["phase0b_authorized"] = False
        write_json_no_overwrite(output_dir / "summary.json", summary)
        write_verdict_markdown(output_dir, summary)

        gzip_path = trace_paths[0].with_suffix(trace_paths[0].suffix + ".gz")
        with trace_paths[0].open("rb") as source, gzip.open(
            gzip_path, "xb", compresslevel=6
        ) as target:
            shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
        write_json_no_overwrite(
            output_dir / "artifact_hashes.json",
            {
                path.name: sha256_file(path)
                for path in sorted(output_dir.iterdir())
                if path.is_file()
            },
        )
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2))
        write_json_no_overwrite(
            output_dir / "COMPLETE.json",
            {
                "schema_version": "spectatorroute-phase0a-completion-v1",
                "status": "SUCCESS_COMPLETE",
                "decision": summary["decision"],
                "phase0b_authorized": bool(summary["phase0b_authorized"]),
                "paper_claim_authorized": False,
                "completed_at_epoch": time.time(),
                "frozen_lock_sha256": lock_info["lock_sha256"],
                "summary_sha256": sha256_file(output_dir / "summary.json"),
                "verdict_sha256": sha256_file(output_dir / "VERDICT.md"),
                "artifact_hashes_sha256": sha256_file(
                    output_dir / "artifact_hashes.json"
                ),
                "real_gpu_acceptance_sha256": acceptance_info["sha256"],
                "authority_rule": "absence_of_this_sentinel_means_invalid_or_incomplete",
            },
        )
        return 0
    except TimeoutError as error:
        write_json_no_overwrite(
            output_dir / "failure.json",
            {
                "decision": "UNSOLVED_BUDGET",
                "error_type": type(error).__name__,
                "error": str(error),
                "phase0b_authorized": False,
                "paper_claim_authorized": False,
            },
        )
        return 3
    except ContaminationError as error:
        failure_path = output_dir / "failure.json"
        if not failure_path.exists():
            write_json_no_overwrite(
                failure_path,
                {
                    "decision": "INVALID_CONTAMINATION",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "phase0b_authorized": False,
                    "paper_claim_authorized": False,
                },
            )
        return 2
    except Exception as error:
        failure_path = output_dir / "failure.json"
        if not failure_path.exists():
            write_json_no_overwrite(
                failure_path,
                {
                    "decision": "INVALID",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "phase0b_authorized": False,
                    "paper_claim_authorized": False,
                },
            )
        raise
    finally:
        _disarm_parent_hard_deadline(previous_alarm_handler)


def orchestrate(args: argparse.Namespace) -> int:
    """Protect initialization as well as the main run with failure authority."""

    output_dir = Path(args.output_dir).resolve()
    output_existed_before = output_dir.exists()
    try:
        return _orchestrate_inner(args)
    except TimeoutError as error:
        if not output_existed_before:
            output_dir.mkdir(parents=True, exist_ok=True)
            failure_path = output_dir / "failure.json"
            if not failure_path.exists():
                write_json_no_overwrite(
                    failure_path,
                    {
                        "decision": "UNSOLVED_BUDGET",
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "phase0b_authorized": False,
                        "paper_claim_authorized": False,
                    },
                )
        return 3
    except ContaminationError as error:
        if not output_existed_before:
            output_dir.mkdir(parents=True, exist_ok=True)
            failure_path = output_dir / "failure.json"
            if not failure_path.exists():
                write_json_no_overwrite(
                    failure_path,
                    {
                        "decision": "INVALID_CONTAMINATION",
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "phase0b_authorized": False,
                        "paper_claim_authorized": False,
                    },
                )
        return 2
    except Exception as error:
        if not output_existed_before:
            output_dir.mkdir(parents=True, exist_ok=True)
            failure_path = output_dir / "failure.json"
            if not failure_path.exists():
                write_json_no_overwrite(
                    failure_path,
                    {
                        "decision": "INVALID",
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "phase0b_authorized": False,
                        "paper_claim_authorized": False,
                    },
                )
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--frozen-lock", required=True)
    parser.add_argument("--frozen-lock-sha256", required=True)
    parser.add_argument("--acceptance-only", action="store_true")
    parser.add_argument("--acceptance-artifact")
    parser.add_argument("--deadline-epoch", type=float)
    parser.add_argument("--worker", choices=("numeric", "trace"))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.acceptance_only:
        if args.worker:
            raise ProtocolError("--acceptance-only cannot be combined with --worker")
        return run_real_gpu_acceptance(args)
    if args.worker:
        if args.deadline_epoch is None:
            raise ProtocolError("worker requires --deadline-epoch")
        if args.worker == "numeric":
            return worker_numeric(args)
        return worker_trace(args)
    return orchestrate(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except TimeoutError as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(3)
    except ProtocolError as error:
        print(f"INVALID: {error}", file=sys.stderr)
        raise SystemExit(2)
