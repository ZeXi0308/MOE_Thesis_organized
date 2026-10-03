#!/usr/bin/env python3
"""Run the first SemanticFence calibration-to-fresh-evaluation pilot.

The public commands are deliberately staged:

* ``acceptance`` observes a real, otherwise-idle RTX 5090 stack and performs a
  pretrained OLMoE smoke forward.  It creates no scientific result.
* ``seal`` binds that acceptance, the config, every producer/test, and both
  data manifests into a content-addressed lock.
* ``run`` executes calibration before fresh evaluation and writes
  ``COMPLETE.json`` last.  An incomplete directory has no authority.

Torch and Transformers are imported only inside GPU functions so all protocol,
packing, lock, and verdict tests remain runnable on a CPU-only host.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping, Sequence


SCHEMA = "semanticfence-p0-config-v1"
LOCK_SCHEMA = "semanticfence-p0-lock-v1"
ACCEPTANCE_SCHEMA = "semanticfence-p0-acceptance-v1"
ACCEPTANCE_COMPLETE_SCHEMA = "semanticfence-p0-acceptance-complete-v1"
CALL_SCHEMA = "semanticfence-p0-call-v1"
DESCRIPTOR_SCHEMA = "semanticfence-pre-call-descriptor-v2"
SUMMARY_SCHEMA = "semanticfence-p0-summary-v1"
HEX = set("0123456789abcdef")
EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT_DEFAULT = EXPERIMENT_DIR.parents[3]
CONTRACT_PATH = EXPERIMENT_DIR / "executor_contract.py"
GPU_EXECUTION_PATH = EXPERIMENT_DIR / "gpu_execution.py"
LEGACY_HELPER_PATH = (
    EXPERIMENT_DIR.parents[1]
    / "spectatorroute"
    / "experiments"
    / "run_phase0a_5090.py"
)


class ProtocolError(RuntimeError):
    """The pilot cannot produce an interpretable result."""


class ContaminationError(ProtocolError):
    """Foreign GPU activity invalidated the measurement."""


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ProtocolError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


CONTRACT = _load_module("semanticfence_executor_contract", CONTRACT_PATH)


def _legacy() -> Any:
    """Load only the frozen trace/hash helper implementation on demand."""

    cached = sys.modules.get("semanticfence_spectator_helpers")
    if cached is not None:
        return cached
    return _load_module("semanticfence_spectator_helpers", LEGACY_HELPER_PATH)


def _gpu() -> Any:
    cached = sys.modules.get("semanticfence_gpu_execution")
    if cached is not None:
        return cached
    return _load_module("semanticfence_gpu_execution", GPU_EXECUTION_PATH)


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


def is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= HEX


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ProtocolError(f"expected JSON object: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ProtocolError(f"{path}:{line_no} is not an object")
            result.append(value)
    return result


def write_json_no_overwrite(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_jsonl_no_overwrite(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _repo_path(repo_root: Path, value: str) -> Path:
    path = (repo_root / value).resolve()
    try:
        path.relative_to(repo_root.resolve())
    except ValueError as exc:
        raise ProtocolError(f"path escapes repository: {value}") from exc
    return path


def validate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "schema_version",
        "status",
        "evidence_boundary",
        "model",
        "data",
        "intervention",
        "decision",
        "budget",
    }
    if set(config) != required:
        raise ProtocolError("config has missing or unknown top-level fields")
    if config["schema_version"] != SCHEMA or config["status"] != "FROZEN":
        raise ProtocolError("SemanticFence config is not frozen schema v1")

    model = config["model"]
    required_model = {
        "repo_id",
        "revision",
        "local_path_candidates",
        "dtype",
        "num_hidden_layers",
        "hidden_size",
        "intermediate_size",
        "num_experts",
        "num_experts_per_tok",
        "file_sha256",
    }
    if not isinstance(model, Mapping) or set(model) != required_model:
        raise ProtocolError("model binding fields are incomplete")
    if model["dtype"] != "bfloat16":
        raise ProtocolError("pilot requires BF16")
    if not isinstance(model["local_path_candidates"], list) or not model[
        "local_path_candidates"
    ]:
        raise ProtocolError("model requires local path candidates")
    for digest in model["file_sha256"].values():
        if not is_sha256(digest):
            raise ProtocolError("model file binding is not SHA-256")

    data = config["data"]
    required_data = {
        "calibration_manifest",
        "calibration_manifest_sha256",
        "calibration_document_count",
        "calibration_provenance",
        "calibration_provenance_sha256",
        "historical_hash_registry",
        "historical_hash_registry_sha256",
        "evaluation_manifest",
        "evaluation_manifest_sha256",
        "evaluation_document_count",
        "evaluation_provenance",
        "evaluation_provenance_sha256",
        "evaluation_exclusion_report",
        "evaluation_exclusion_report_sha256",
        "evaluation_artifact_hashes",
        "evaluation_artifact_hashes_sha256",
        "token_offsets",
        "window_tokens",
        "add_special_tokens",
        "calibration_positions",
        "evaluation_position",
    }
    if not isinstance(data, Mapping) or set(data) != required_data:
        raise ProtocolError("data binding fields are incomplete")
    for key in (
        "calibration_manifest_sha256",
        "calibration_provenance_sha256",
        "historical_hash_registry_sha256",
        "evaluation_manifest_sha256",
        "evaluation_provenance_sha256",
        "evaluation_exclusion_report_sha256",
        "evaluation_artifact_hashes_sha256",
    ):
        if not is_sha256(data[key]):
            raise ProtocolError(f"data binding {key} is not SHA-256")
    if int(data["calibration_document_count"]) != 8:
        raise ProtocolError("calibration document count must remain 8")
    if int(data["evaluation_document_count"]) != 32:
        raise ProtocolError("fresh evaluation document count must remain 32")
    if [int(value) for value in data["token_offsets"]] != [0, 256]:
        raise ProtocolError("token offsets changed")
    if int(data["window_tokens"]) != 16 or int(data["evaluation_position"]) != 15:
        raise ProtocolError("window/evaluation position changed")
    if [int(value) for value in data["calibration_positions"]] != list(range(16)):
        raise ProtocolError("calibration must use all 16 real positions")

    intervention = config["intervention"]
    required_intervention = {
        "m_values",
        "reference_m",
        "fixed_padding_m",
        "repeats",
        "warmups",
        "min_calibration_packs",
        "min_calibration_documents",
        "cublaslt_log_level",
        "cublaslt_log_mask",
    }
    if not isinstance(intervention, Mapping) or set(intervention) != required_intervention:
        raise ProtocolError("intervention fields are incomplete")
    if [int(value) for value in intervention["m_values"]] != [1, 2, 4, 8, 16, 32, 64]:
        raise ProtocolError("M grid changed")
    constants = {
        "reference_m": 1,
        "fixed_padding_m": 64,
        "repeats": 10,
        "warmups": 3,
        "min_calibration_packs": 3,
        "min_calibration_documents": 3,
    }
    for key, expected in constants.items():
        if int(intervention[key]) != expected:
            raise ProtocolError(f"intervention {key} changed")

    decision = config["decision"]
    required_decision = {
        "minimum_unrestricted_mismatch_victims",
        "minimum_semanticfence_covered_victims",
        "minimum_distinct_admitted_m_gt_1",
        "minimum_latency_reduction_fraction",
    }
    if not isinstance(decision, Mapping) or set(decision) != required_decision:
        raise ProtocolError("decision fields are incomplete")
    if int(decision["minimum_unrestricted_mismatch_victims"]) != 8:
        raise ProtocolError("unrestricted mismatch threshold changed")
    if int(decision["minimum_semanticfence_covered_victims"]) != 8:
        raise ProtocolError("coverage threshold changed")
    if int(decision["minimum_distinct_admitted_m_gt_1"]) != 2:
        raise ProtocolError("admitted-M threshold changed")
    if float(decision["minimum_latency_reduction_fraction"]) != 0.10:
        raise ProtocolError("latency threshold changed")
    if int(config["budget"].get("max_gpu_seconds", -1)) != 5400:
        raise ProtocolError("GPU budget changed")
    return dict(config)


def verify_data_bindings(config: Mapping[str, Any], repo_root: Path) -> dict[str, Any]:
    data = config["data"]
    result: dict[str, Any] = {}
    seen_hashes: dict[str, set[str]] = {}
    for split, manifest_split, path_key, hash_key, count_key in (
        (
            "calibration",
            "calibration",
            "calibration_manifest",
            "calibration_manifest_sha256",
            "calibration_document_count",
        ),
        (
            "evaluation",
            "semanticfence_eval_fresh",
            "evaluation_manifest",
            "evaluation_manifest_sha256",
            "evaluation_document_count",
        ),
    ):
        path = _repo_path(repo_root, str(data[path_key]))
        observed_hash = sha256_file(path)
        if observed_hash != data[hash_key]:
            raise ProtocolError(f"{split} manifest hash mismatch")
        rows = load_jsonl(path)
        if len(rows) != int(data[count_key]):
            raise ProtocolError(f"{split} manifest count mismatch")
        document_indices = [int(row.get("document_index", -1)) for row in rows]
        if document_indices != list(range(len(rows))):
            raise ProtocolError(f"{split} document indices are not canonical")
        hashes: set[str] = set()
        for row in rows:
            if row.get("split") != manifest_split:
                raise ProtocolError(f"{split} manifest has wrong split label")
            text = str(row.get("text", ""))
            digest = hashlib.sha256(
                text.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")
            ).hexdigest()
            if digest != row.get("text_sha256") or digest in hashes:
                raise ProtocolError(f"{split} manifest text identity mismatch")
            hashes.add(digest)
        seen_hashes[split] = hashes
        result[split] = {
            "path": str(path),
            "sha256": observed_hash,
            "document_count": len(rows),
            "ordered_text_hashes_sha256": canonical_sha256(
                [row["text_sha256"] for row in rows]
            ),
        }
    overlap = seen_hashes["calibration"] & seen_hashes["evaluation"]
    if overlap:
        raise ProtocolError(f"calibration/evaluation overlap: {sorted(overlap)}")
    auxiliary: dict[str, str] = {}
    for path_key, hash_key in (
        ("calibration_provenance", "calibration_provenance_sha256"),
        ("historical_hash_registry", "historical_hash_registry_sha256"),
        ("evaluation_provenance", "evaluation_provenance_sha256"),
        ("evaluation_exclusion_report", "evaluation_exclusion_report_sha256"),
        ("evaluation_artifact_hashes", "evaluation_artifact_hashes_sha256"),
    ):
        path = _repo_path(repo_root, str(data[path_key]))
        observed = sha256_file(path)
        if observed != data[hash_key]:
            raise ProtocolError(f"{path_key} hash mismatch")
        auxiliary[path_key] = observed
    provenance = load_json(_repo_path(repo_root, str(data["evaluation_provenance"])))
    exclusion = load_json(
        _repo_path(repo_root, str(data["evaluation_exclusion_report"]))
    )
    artifact_hashes = load_json(
        _repo_path(repo_root, str(data["evaluation_artifact_hashes"]))
    )
    if provenance.get("status") != "FRESH_EVAL_PREPARED_NOT_EXECUTED":
        raise ProtocolError("fresh evaluation provenance status mismatch")
    if provenance.get("eval_manifest_sha256") != data["evaluation_manifest_sha256"]:
        raise ProtocolError("fresh provenance/manifest binding mismatch")
    if exclusion.get("selected_overlap_count") != 0:
        raise ProtocolError("fresh evaluation exclusion report has overlap")
    if set(exclusion.get("required_source_names", [])) != {
        "historical_registry",
        "calibration_manifest",
        "sealed_manifest",
        "smoke_manifest",
    }:
        raise ProtocolError("fresh exclusion source set mismatch")
    data_root = _repo_path(repo_root, str(data["evaluation_manifest"])).parent
    for name, digest in artifact_hashes.get("files", {}).items():
        path = (data_root / str(name)).resolve()
        try:
            path.relative_to(data_root)
        except ValueError as exc:
            raise ProtocolError("fresh artifact hash path escapes data root") from exc
        if not path.is_file() or sha256_file(path) != digest:
            raise ProtocolError(f"fresh artifact hash mismatch: {name}")
    result["auxiliary"] = auxiliary
    return result


def resolve_model_path(config: Mapping[str, Any], override: str | None = None) -> Path:
    candidates = [override] if override else list(config["model"]["local_path_candidates"])
    expected = config["model"]["file_sha256"]
    failures: list[str] = []
    for raw_path in candidates:
        if not raw_path:
            continue
        path = Path(str(raw_path)).expanduser().resolve()
        if not path.is_dir():
            failures.append(f"{path}: missing")
            continue
        mismatch = False
        for name, digest in expected.items():
            file_path = path / name
            if not file_path.is_file() or sha256_file(file_path) != digest:
                mismatch = True
                failures.append(f"{file_path}: missing or hash mismatch")
                break
        if not mismatch:
            return path
    raise ProtocolError("no exact model snapshot found: " + "; ".join(failures))


def source_bindings(repo_root: Path, config_path: Path) -> dict[str, str]:
    relative_paths = [
        "docs/ideas/semanticfence/experiments/prepare_eval_manifest.py",
        "docs/ideas/semanticfence/experiments/executor_contract.py",
        "docs/ideas/semanticfence/experiments/gpu_execution.py",
        "docs/ideas/semanticfence/experiments/run_pilot_5090.py",
        "docs/ideas/semanticfence/experiments/test_prepare_eval_manifest.py",
        "docs/ideas/semanticfence/experiments/test_executor_contract.py",
        "docs/ideas/semanticfence/experiments/test_gpu_execution.py",
        "docs/ideas/semanticfence/experiments/test_run_pilot.py",
        "docs/ideas/spectatorroute/experiments/run_phase0a_5090.py",
        "refine-logs/EXPERIMENT_PLAN_20260809_202112.md",
    ]
    result: dict[str, str] = {}
    for relative in relative_paths:
        path = _repo_path(repo_root, relative)
        if not path.is_file():
            raise ProtocolError(f"required producer/test is absent: {relative}")
        result[relative] = sha256_file(path)
    try:
        config_relative = str(config_path.resolve().relative_to(repo_root.resolve()))
    except ValueError as exc:
        raise ProtocolError("config must reside inside repository") from exc
    result[config_relative] = sha256_file(config_path)
    return dict(sorted(result.items()))


def _nvidia_identity() -> dict[str, str]:
    return dict(_legacy()._nvidia_identity())


def _nvidia_compute_processes() -> list[dict[str, Any]]:
    return list(_legacy()._nvidia_compute_processes())


def assert_clean_gpu(expected_uuid: str, *, allowed_pids: set[int]) -> list[dict[str, Any]]:
    processes = _nvidia_compute_processes()
    foreign = [
        row
        for row in processes
        if row["gpu_uuid"] != expected_uuid or int(row["pid"]) not in allowed_pids
    ]
    if foreign:
        raise ContaminationError(f"foreign GPU process detected: {foreign!r}")
    return processes


def _set_math_state() -> dict[str, Any]:
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    torch.backends.cuda.preferred_blas_library("cublas")
    return {
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


def load_model(config: Mapping[str, Any], model_path: Path) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), local_files_only=True, use_fast=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    model.eval().to("cuda")
    expected = config["model"]
    checks = {
        "num_hidden_layers": expected["num_hidden_layers"],
        "hidden_size": expected["hidden_size"],
        "intermediate_size": expected["intermediate_size"],
        "num_experts": expected["num_experts"],
        "num_experts_per_tok": expected["num_experts_per_tok"],
    }
    mismatches = [
        f"{key}={getattr(model.config, key, None)!r} expected {value!r}"
        for key, value in checks.items()
        if getattr(model.config, key, None) != value
    ]
    if getattr(model.config, "norm_topk_prob", None) is not False:
        mismatches.append("norm_topk_prob is not False")
    if mismatches:
        raise ProtocolError("model structure mismatch: " + "; ".join(mismatches))
    return model, tokenizer


def _loaded_cublaslt() -> dict[str, Any]:
    import ctypes

    paths = sorted(
        {
            str(Path(line.split()[-1]).resolve())
            for line in Path("/proc/self/maps").read_text(encoding="utf-8").splitlines()
            if "libcublasLt" in line and line.split()[-1].startswith("/")
        }
    )
    if len(paths) != 1:
        raise ProtocolError(f"expected exactly one loaded libcublasLt, found {paths}")
    library = ctypes.CDLL(paths[0])
    get_version = library.cublasLtGetVersion
    get_version.restype = ctypes.c_size_t
    return {
        "path": paths[0],
        "sha256": sha256_file(Path(paths[0])),
        "version": int(get_version()),
    }


def observe_stack(model: Any) -> dict[str, Any]:
    import torch
    import transformers
    from transformers.models.olmoe import modeling_olmoe

    identity = _nvidia_identity()
    source_path = Path(inspect.getsourcefile(modeling_olmoe) or "").resolve()
    observed = {
        "gpu": identity,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "transformers": transformers.__version__,
        "python": sys.version.split()[0],
        "torch_compute_capability": list(torch.cuda.get_device_capability(0)),
        "matmul_state": _set_math_state(),
        "cublaslt": _loaded_cublaslt(),
        "transformers_olmoe_source": {
            "path": str(source_path),
            "sha256": sha256_file(source_path),
        },
    }
    observed["stack_digest"] = canonical_sha256(observed)
    return observed


def _first_window(config: Mapping[str, Any], repo_root: Path, tokenizer: Any) -> Any:
    import torch

    path = _repo_path(repo_root, str(config["data"]["calibration_manifest"]))
    document = load_jsonl(path)[0]
    token_ids = tokenizer(
        str(document["text"]),
        add_special_tokens=bool(config["data"]["add_special_tokens"]),
    )["input_ids"]
    width = int(config["data"]["window_tokens"])
    window = token_ids[:width]
    if len(window) != width:
        raise ProtocolError("acceptance calibration window is too short")
    return torch.tensor([window], dtype=torch.long, device="cuda")


def run_acceptance(args: argparse.Namespace) -> int:
    import torch

    config_path = Path(args.config).resolve()
    repo_root = Path(args.repo_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise ProtocolError(f"acceptance output exists: {output_dir}")
    config = validate_config(load_json(config_path))
    data_bindings = verify_data_bindings(config, repo_root)
    sources = source_bindings(repo_root, config_path)
    model_path = resolve_model_path(config, args.model_path)
    identity = _nvidia_identity()
    if identity["name"] != "NVIDIA GeForce RTX 5090":
        raise ProtocolError(f"acceptance requires exact RTX 5090, got {identity}")
    assert_clean_gpu(identity["uuid"], allowed_pids=set())
    output_dir.mkdir(parents=True)
    try:
        math_state = _set_math_state()
        model, tokenizer = load_model(config, model_path)
        input_ids = _first_window(config, repo_root, tokenizer)
        with torch.inference_mode():
            outputs = model(
                input_ids=input_ids,
                use_cache=False,
                output_router_logits=True,
                return_dict=True,
            )
            hidden = model.model.layers[0].input_layernorm(
                model.model.embed_tokens(input_ids)
            )
            expert = model.model.layers[0].mlp.experts[0]
            smoke = expert(hidden[:, -1, :].reshape(1, -1))
        if smoke.dtype != torch.bfloat16 or not bool(torch.isfinite(smoke).all().item()):
            raise ProtocolError("acceptance expert output is not finite BF16")
        if outputs.router_logits is None or len(outputs.router_logits) != int(
            config["model"]["num_hidden_layers"]
        ):
            raise ProtocolError("acceptance router capture is incomplete")
        torch.cuda.synchronize()
        assert_clean_gpu(identity["uuid"], allowed_pids={os.getpid()})
        stack = observe_stack(model)
        if stack["matmul_state"] != math_state:
            raise ProtocolError("math state drifted during acceptance")
        model_bindings = {
            name: sha256_file(model_path / name)
            for name in config["model"]["file_sha256"]
        }
        artifact = {
            "schema_version": ACCEPTANCE_SCHEMA,
            "status": "ACCEPTED_REAL_GPU_CAPABILITY_ONLY",
            "paper_result": False,
            "config_sha256": sha256_file(config_path),
            "source_bindings": sources,
            "data_bindings": data_bindings,
            "model_path": str(model_path),
            "model_bindings": model_bindings,
            "stack": stack,
            "acceptance_smoke": {
                "input_shape": list(input_ids.shape),
                "expert_output_shape": list(smoke.shape),
                "expert_output_sha256": _legacy()._tensor_sha256(smoke),
            },
            "created_at_epoch": time.time(),
        }
        write_json_no_overwrite(output_dir / "ACCEPTANCE.json", artifact)
        write_json_no_overwrite(
            output_dir / "ACCEPTANCE_COMPLETE.json",
            {
                "schema_version": ACCEPTANCE_COMPLETE_SCHEMA,
                "status": "COMPLETE",
                "acceptance_sha256": sha256_file(output_dir / "ACCEPTANCE.json"),
                "paper_result": False,
            },
        )
    except BaseException:
        failure = output_dir / "failure.json"
        if not failure.exists():
            write_json_no_overwrite(
                failure,
                {"status": "INVALID", "error": "acceptance failed; see stderr"},
            )
        raise
    return 0


def load_acceptance(path: Path) -> dict[str, Any]:
    path = path.resolve()
    complete_path = path.parent / "ACCEPTANCE_COMPLETE.json"
    if (path.parent / "failure.json").exists():
        raise ProtocolError("acceptance directory contains a failure artifact")
    if not complete_path.is_file():
        raise ProtocolError("acceptance completion sentinel is absent")
    complete = load_json(complete_path)
    if (
        complete.get("schema_version") != ACCEPTANCE_COMPLETE_SCHEMA
        or complete.get("status") != "COMPLETE"
        or complete.get("paper_result") is not False
        or complete.get("acceptance_sha256") != sha256_file(path)
    ):
        raise ProtocolError("acceptance completion sentinel is invalid")
    acceptance = load_json(path)
    if acceptance.get("schema_version") != ACCEPTANCE_SCHEMA:
        raise ProtocolError("unknown acceptance schema")
    if acceptance.get("status") != "ACCEPTED_REAL_GPU_CAPABILITY_ONLY":
        raise ProtocolError("GPU acceptance status is not accepted")
    stack = acceptance.get("stack")
    if not isinstance(stack, Mapping) or not is_sha256(stack.get("stack_digest")):
        raise ProtocolError("acceptance stack binding is incomplete")
    recompute = dict(stack)
    digest = recompute.pop("stack_digest")
    if canonical_sha256(recompute) != digest:
        raise ProtocolError("acceptance stack digest mismatch")
    return acceptance


def create_lock(
    *,
    config_path: Path,
    repo_root: Path,
    acceptance_path: Path,
) -> dict[str, Any]:
    config = validate_config(load_json(config_path))
    acceptance = load_acceptance(acceptance_path)
    if acceptance["config_sha256"] != sha256_file(config_path):
        raise ProtocolError("acceptance/config mismatch")
    current_sources = source_bindings(repo_root, config_path)
    current_data = verify_data_bindings(config, repo_root)
    if acceptance.get("source_bindings") != current_sources:
        raise ProtocolError("acceptance/source mismatch")
    if acceptance.get("data_bindings") != current_data:
        raise ProtocolError("acceptance/data mismatch")
    payload = {
        "schema_version": LOCK_SCHEMA,
        "status": "SEALED_BEFORE_SCIENCE",
        "config_sha256": sha256_file(config_path),
        "acceptance_sha256": sha256_file(acceptance_path),
        "acceptance_complete_sha256": sha256_file(
            acceptance_path.parent / "ACCEPTANCE_COMPLETE.json"
        ),
        "stack_digest": acceptance["stack"]["stack_digest"],
        "source_bindings": current_sources,
        "data_bindings": current_data,
        "frozen_constants": {
            "m_values": list(config["intervention"]["m_values"]),
            "reference_m": int(config["intervention"]["reference_m"]),
            "fixed_padding_m": int(config["intervention"]["fixed_padding_m"]),
            "repeats": int(config["intervention"]["repeats"]),
            "warmups": int(config["intervention"]["warmups"]),
            "min_calibration_packs": int(
                config["intervention"]["min_calibration_packs"]
            ),
            "min_calibration_documents": int(
                config["intervention"]["min_calibration_documents"]
            ),
            "max_gpu_seconds": int(config["budget"]["max_gpu_seconds"]),
        },
    }
    return payload | {"lock_sha256": canonical_sha256(payload)}


def verify_lock(
    lock: Mapping[str, Any],
    *,
    config_path: Path,
    repo_root: Path,
    acceptance_path: Path,
    verify_data_files: bool = True,
) -> dict[str, Any]:
    if lock.get("schema_version") != LOCK_SCHEMA or lock.get("status") != "SEALED_BEFORE_SCIENCE":
        raise ProtocolError("lock schema/status mismatch")
    payload = dict(lock)
    observed_lock_sha = payload.pop("lock_sha256", None)
    if not is_sha256(observed_lock_sha) or canonical_sha256(payload) != observed_lock_sha:
        raise ProtocolError("lock content hash mismatch")
    if lock.get("config_sha256") != sha256_file(config_path):
        raise ProtocolError("config changed after seal")
    if lock.get("acceptance_sha256") != sha256_file(acceptance_path):
        raise ProtocolError("acceptance changed after seal")
    if lock.get("acceptance_complete_sha256") != sha256_file(
        acceptance_path.parent / "ACCEPTANCE_COMPLETE.json"
    ):
        raise ProtocolError("acceptance completion changed after seal")
    if lock.get("source_bindings") != source_bindings(repo_root, config_path):
        raise ProtocolError("producer/test changed after seal")
    config = validate_config(load_json(config_path))
    if verify_data_files and lock.get("data_bindings") != verify_data_bindings(
        config, repo_root
    ):
        raise ProtocolError("data changed after seal")
    acceptance = load_acceptance(acceptance_path)
    if lock.get("stack_digest") != acceptance["stack"]["stack_digest"]:
        raise ProtocolError("lock/acceptance stack mismatch")
    return dict(lock)


def run_seal(args: argparse.Namespace) -> int:
    output = Path(args.output).resolve()
    if output.exists():
        raise ProtocolError(f"lock output exists: {output}")
    lock = create_lock(
        config_path=Path(args.config).resolve(),
        repo_root=Path(args.repo_root).resolve(),
        acceptance_path=Path(args.acceptance_artifact).resolve(),
    )
    write_json_no_overwrite(output, lock)
    return 0


def row_record_from_mapping(value: Mapping[str, Any]) -> Any:
    materialized = dict(value)
    if "schema_version" in materialized:
        if materialized.pop("schema_version") != CONTRACT.ROW_ID_SCHEMA:
            raise ProtocolError("row record schema mismatch")
    required = {
        "split",
        "document_sha256",
        "document_index",
        "offset",
        "token_position",
        "layer",
        "expert_id",
        "route_rank",
        "hidden_sha256",
    }
    if set(materialized) != required:
        raise ProtocolError("row record fields are incomplete")
    return CONTRACT.RowRecord(**materialized)


def strict_bf16_mismatch_count(left: bytes, right: bytes) -> int:
    """Count raw BF16 element mismatches, including signed zero."""

    if len(left) != len(right) or len(left) % 2:
        raise ProtocolError("raw BF16 byte strings have incompatible shapes")
    return sum(
        left[index : index + 2] != right[index : index + 2]
        for index in range(0, len(left), 2)
    )


def descriptor_binding_context(
    config: Mapping[str, Any], acceptance: Mapping[str, Any]
) -> dict[str, Any]:
    """Return immutable experiment bindings required by every call."""

    required_acceptance = {
        "config_sha256",
        "model_bindings",
        "source_bindings",
        "stack",
    }
    if not required_acceptance <= set(acceptance):
        raise ProtocolError("acceptance lacks descriptor bindings")
    stack = acceptance["stack"]
    if not isinstance(stack, Mapping) or not is_sha256(stack.get("stack_digest")):
        raise ProtocolError("descriptor stack binding is invalid")
    context = {
        "config_sha256": acceptance["config_sha256"],
        "stack_digest": stack["stack_digest"],
        "model_bindings_sha256": canonical_sha256(acceptance["model_bindings"]),
        "source_bindings_sha256": canonical_sha256(acceptance["source_bindings"]),
        "math_state_sha256": canonical_sha256(stack.get("matmul_state")),
        "hidden_size": int(config["model"]["hidden_size"]),
        "intermediate_size": int(config["model"]["intermediate_size"]),
        "dtype": str(config["model"]["dtype"]),
    }
    for key in (
        "config_sha256",
        "stack_digest",
        "model_bindings_sha256",
        "source_bindings_sha256",
        "math_state_sha256",
    ):
        if not is_sha256(context[key]):
            raise ProtocolError(f"descriptor binding {key} is invalid")
    if context["dtype"] != "bfloat16":
        raise ProtocolError("descriptor dtype is not frozen BF16")
    return context


def _descriptor_from_call_fields(
    call: Mapping[str, Any], binding_context: Mapping[str, Any]
) -> dict[str, Any]:
    m_value = int(call["m"])
    hidden = int(binding_context["hidden_size"])
    intermediate = int(binding_context["intermediate_size"])
    return {
        "schema_version": DESCRIPTOR_SCHEMA,
        "config_sha256": binding_context["config_sha256"],
        "stack_digest": binding_context["stack_digest"],
        "model_bindings_sha256": binding_context["model_bindings_sha256"],
        "source_bindings_sha256": binding_context["source_bindings_sha256"],
        "math_state_sha256": binding_context["math_state_sha256"],
        "call_index": int(call["call_index"]),
        "arm": str(call["arm"]),
        "layer": int(call["layer"]),
        "expert_id": int(call["expert_id"]),
        "m": m_value,
        "dtype": binding_context["dtype"],
        "row_ids": list(call["row_ids"]),
        "padding_rows": int(call["padding_rows"]),
        "row_slot_policy": "layer_expert_then_canonical_row_id",
        "projection_shapes": {
            "gate_proj": [[m_value, hidden], [hidden, intermediate], [m_value, intermediate]],
            "up_proj": [[m_value, hidden], [hidden, intermediate], [m_value, intermediate]],
            "down_proj": [[m_value, intermediate], [intermediate, hidden], [m_value, hidden]],
        },
        "expected_signatures": list(call["expected_signatures"]),
    }


def validate_call_record(
    call: Mapping[str, Any],
    *,
    expected_index: int,
    binding_context: Mapping[str, Any],
) -> dict[str, Any]:
    required = {
        "schema_version",
        "call_index",
        "arm",
        "layer",
        "expert_id",
        "m",
        "row_ids",
        "padding_rows",
        "expected_signatures",
        "pre_call_descriptor",
        "pre_call_descriptor_sha256",
    }
    if set(call) != required or call["schema_version"] != CALL_SCHEMA:
        raise ProtocolError("call ledger schema mismatch")
    if int(call["call_index"]) != int(expected_index):
        raise ProtocolError("call indices are not contiguous")
    descriptor = call["pre_call_descriptor"]
    if not isinstance(descriptor, Mapping):
        raise ProtocolError("call descriptor is absent")
    expected_descriptor = _descriptor_from_call_fields(call, binding_context)
    if dict(descriptor) != expected_descriptor:
        raise ProtocolError("call descriptor content does not match its call/bindings")
    if call["pre_call_descriptor_sha256"] != canonical_sha256(expected_descriptor):
        raise ProtocolError("call descriptor hash mismatch")
    if int(call["m"]) != len(call["row_ids"]) + int(call["padding_rows"]):
        raise ProtocolError("call M does not match real plus padding rows")
    if len(set(call["row_ids"])) != len(call["row_ids"]):
        raise ProtocolError("one call duplicates a real row")
    return dict(call)


def validate_call_ledger(
    rows: Sequence[Any],
    calls: Sequence[Mapping[str, Any]],
    *,
    binding_context: Mapping[str, Any],
    starting_index: int = 0,
) -> dict[str, Any]:
    """Validate exact row coverage and pre-call evidence for one arm/repeat."""

    expected = {row.row_id for row in rows}
    if len(expected) != len(rows):
        raise ProtocolError("input row identities are not unique")
    observed: list[str] = []
    for local_index, call in enumerate(calls):
        validate_call_record(
            call,
            expected_index=int(starting_index) + local_index,
            binding_context=binding_context,
        )
        real_rows = list(call["row_ids"])
        observed.extend(real_rows)
    if len(set(observed)) != len(observed):
        raise ProtocolError("call ledger duplicates a row across calls")
    if set(observed) != expected:
        raise ProtocolError("call ledger does not exactly cover input rows")
    return {
        "row_count": len(expected),
        "call_count": len(calls),
        "ledger_sha256": canonical_sha256(list(calls)),
    }


def paired_latency_reduction(reference_ms: Sequence[float], candidate_ms: Sequence[float]) -> float:
    if len(reference_ms) != len(candidate_ms) or not reference_ms:
        raise ProtocolError("paired latency vectors are incomplete")
    ratios: list[float] = []
    for reference, candidate in zip(reference_ms, candidate_ms):
        if reference <= 0 or candidate <= 0:
            raise ProtocolError("latencies must be positive")
        ratios.append(float(candidate) / float(reference))
    return 1.0 - statistics.median(ratios)


def parse_trace_for_calls(
    *,
    trace_path: Path,
    calls: Sequence[Mapping[str, Any]],
    hidden_size: int,
    intermediate_size: int,
) -> list[dict[str, Any]]:
    """Bind every traced three-GEMM expert call to an explicit call ledger."""

    legacy = _legacy()
    records: list[Mapping[str, Any]] = []
    with trace_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            parsed = legacy.parse_cublaslt_trace_line(line)
            if parsed is not None:
                records.append(parsed)
    expected = len(calls) * 3
    if len(records) != expected:
        raise ProtocolError(
            f"trace has {len(records)} GEMMs for {len(calls)} calls; expected {expected}"
        )
    result: list[dict[str, Any]] = []
    for call_index, call in enumerate(calls):
        if int(call.get("call_index", -1)) != call_index:
            raise ProtocolError("trace call ledger is not contiguous")
        triplet = records[call_index * 3 : call_index * 3 + 3]
        roles = legacy.validate_projection_triplet(
            triplet,
            m_value=int(call["m"]),
            hidden_size=int(hidden_size),
            intermediate_size=int(intermediate_size),
        )
        signatures = [
            legacy.algorithm_signature(record, role)
            for record, role in zip(triplet, roles)
        ]
        result.append(
            {
                "call_index": call_index,
                "arm": call["arm"],
                "layer": int(call["layer"]),
                "expert_id": int(call["expert_id"]),
                "m": int(call["m"]),
                "row_ids": list(call["row_ids"]),
                "algorithm_signatures": signatures,
                "signature_sha256": canonical_sha256(signatures),
            }
        )
    return result


def bind_trace_to_numeric(
    *,
    trace_rows: Sequence[Mapping[str, Any]],
    trace_call_outputs: Sequence[Mapping[str, Any]],
    numeric_calls: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Require trace/numeric call identity and full-output hashes to close."""

    if not (
        len(trace_rows) == len(trace_call_outputs) == len(numeric_calls)
    ):
        raise ProtocolError("numeric/trace call denominators differ")
    merged: list[dict[str, Any]] = []
    for index, (trace, output, numeric) in enumerate(
        zip(trace_rows, trace_call_outputs, numeric_calls)
    ):
        identity = {
            "call_index": index,
            "arm": numeric["arm"],
            "layer": int(numeric["layer"]),
            "expert_id": int(numeric["expert_id"]),
            "m": int(numeric["m"]),
            "row_ids": list(numeric["row_ids"]),
        }
        for source in (trace, output):
            observed = {
                key: source[key]
                for key in (
                    "call_index",
                    "arm",
                    "layer",
                    "expert_id",
                    "m",
                    "row_ids",
                )
            }
            if observed != identity:
                raise ProtocolError("numeric/trace call identity mismatch")
        if output.get("full_output_sha256") != numeric.get(
            "representative_full_output_sha256"
        ):
            raise ProtocolError("numeric/trace output hash mismatch")
        merged.append(identity | {"signature_sha256": trace["signature_sha256"]})
    return merged


def unique_allowed_signature_lookup(contract: Any) -> dict[tuple[int, int, int], str]:
    """Expose only classes with one unambiguous calibration signature."""

    trusted = CONTRACT.validate_contract(contract)
    observed: dict[tuple[int, int, int], set[str]] = defaultdict(set)
    candidates: dict[tuple[int, int, int], set[str]] = defaultdict(set)
    for entry in trusted.entries:
        observed[(entry.layer, entry.expert_id, entry.m)].add(entry.signature)
        if entry.allowed:
            candidates[(entry.layer, entry.expert_id, entry.m)].add(entry.signature)
    return {
        key: next(iter(signatures))
        for key, signatures in candidates.items()
        if len(signatures) == 1 and len(observed[key]) == 1
    }


def decide_summary(summary: Mapping[str, Any], config: Mapping[str, Any]) -> str:
    """Apply the frozen SUPPORT/WEAKEN/UNABLE interpretation mechanically."""

    required = {
        "reference_all_stable",
        "unrestricted_mismatch_victims",
        "semanticfence_mismatch_rows",
        "semanticfence_covered_victims",
        "semanticfence_distinct_m_gt_1",
        "semanticfence_padding_rows",
        "semanticfence_latency_reduction_fraction",
        "fixed_control_dominates",
        "evidence_complete",
    }
    if set(summary) != required:
        raise ProtocolError("decision summary fields are incomplete")
    if not summary["evidence_complete"] or not summary["reference_all_stable"]:
        return "UNABLE"
    thresholds = config["decision"]
    if int(summary["semanticfence_mismatch_rows"]) > 0:
        return "WEAKEN"
    if int(summary["unrestricted_mismatch_victims"]) == 0:
        return "WEAKEN"
    supported = (
        int(summary["unrestricted_mismatch_victims"])
        >= int(thresholds["minimum_unrestricted_mismatch_victims"])
        and int(summary["semanticfence_covered_victims"])
        >= int(thresholds["minimum_semanticfence_covered_victims"])
        and int(summary["semanticfence_distinct_m_gt_1"])
        >= int(thresholds["minimum_distinct_admitted_m_gt_1"])
        and int(summary["semanticfence_padding_rows"]) == 0
        and float(summary["semanticfence_latency_reduction_fraction"])
        >= float(thresholds["minimum_latency_reduction_fraction"])
        and not bool(summary["fixed_control_dominates"])
    )
    return "SUPPORT" if supported else "WEAKEN"


def finalize_complete(output_dir: Path, summary: Mapping[str, Any]) -> None:
    """Write summary then the sole completion authority; never overwrite."""

    if (output_dir / "COMPLETE.json").exists():
        raise ProtocolError("completion sentinel already exists")
    write_json_no_overwrite(output_dir / "summary.json", summary)
    files = {
        path.name: sha256_file(path)
        for path in sorted(output_dir.iterdir())
        if path.is_file() and path.name != "COMPLETE.json"
    }
    write_json_no_overwrite(
        output_dir / "COMPLETE.json",
        {
            "status": "SUCCESS_COMPLETE",
            "summary_sha256": sha256_file(output_dir / "summary.json"),
            "artifact_sha256": files,
            "authority_rule": "absence_of_this_file_means_invalid_or_incomplete",
        },
    )


def capture_audit_rows(captures: Sequence[Any]) -> list[dict[str, Any]]:
    gpu = _gpu()
    result: list[dict[str, Any]] = []
    for capture in captures:
        result.append(
            {
                "schema_version": capture.schema_version,
                "window_id": capture.window_id,
                "split": capture.split,
                "document_sha256": capture.document_sha256,
                "document_index": int(capture.document_index),
                "offset": int(capture.offset),
                "selected_positions": list(capture.selected_positions),
                "window_token_ids": list(capture.window_token_ids),
                "full_hidden_states_sha256": gpu.tensor_storage_sha256(
                    capture.full_hidden_states
                ),
                "selected_experts_sha256": gpu.tensor_storage_sha256(
                    capture.selected_experts
                ),
                "routing_weights_sha256": gpu.tensor_storage_sha256(
                    capture.routing_weights
                ),
                "router_logits_sha256_by_layer": list(
                    capture.router_logits_sha256_by_layer
                ),
            }
        )
    return result


def planned_call_record(
    call: Any, *, call_index: int, binding_context: Mapping[str, Any]
) -> dict[str, Any]:
    record = {
        "schema_version": CALL_SCHEMA,
        "call_index": int(call_index),
        "arm": call.arm,
        "layer": int(call.layer),
        "expert_id": int(call.expert_id),
        "m": int(call.execution_m),
        "row_ids": list(call.row_ids),
        "padding_rows": int(call.padding_rows),
        "expected_signatures": list(call.expected_signatures),
    }
    descriptor = _descriptor_from_call_fields(record, binding_context)
    return record | {
        "pre_call_descriptor": descriptor,
        "pre_call_descriptor_sha256": canonical_sha256(descriptor),
    }


def verify_pre_call_seal(
    *,
    output_dir: Path,
    stage: str,
    calls: Sequence[Mapping[str, Any]],
    binding_context: Mapping[str, Any],
) -> dict[str, Any]:
    if stage not in {"calibration", "evaluation"}:
        raise ProtocolError("unknown pre-call seal stage")
    calls_path = output_dir / f"{stage}_calls.jsonl"
    seal = load_json(output_dir / f"{stage}_pre_call_seal.json")
    expected = {
        "schema_version": "semanticfence-pre-call-seal-v1",
        "status": "SEALED_BEFORE_EXECUTION",
        "call_count": len(calls),
        "calls_sha256": sha256_file(calls_path),
        "binding_context_sha256": canonical_sha256(binding_context),
    }
    if stage == "evaluation":
        expected["paired_schedule_sha256"] = sha256_file(
            output_dir / "evaluation_paired_schedule.json"
        )
    if seal != expected:
        raise ProtocolError(f"{stage} pre-call seal mismatch")
    for call_index, call in enumerate(calls):
        validate_call_record(
            call, expected_index=call_index, binding_context=binding_context
        )
    return seal


def row_execution_record(value: Any) -> dict[str, Any]:
    return {
        "schema_version": "semanticfence-row-execution-v1",
        "row_id": value.row_id,
        "reference_sha256": value.reference_sha256,
        "repeat_sha256": list(value.repeat_sha256),
        "repeat_mismatch_counts": list(value.repeat_mismatch_counts),
        "bitwise_stable": bool(value.bitwise_stable),
        "all_exact_to_reference": bool(value.all_exact_to_reference),
    }


def _manifest_rows(config: Mapping[str, Any], repo_root: Path, split: str) -> list[dict[str, Any]]:
    if split == "calibration":
        path = _repo_path(repo_root, str(config["data"]["calibration_manifest"]))
    elif split == "semanticfence_eval_fresh":
        path = _repo_path(repo_root, str(config["data"]["evaluation_manifest"]))
    else:
        raise ProtocolError(f"unsupported manifest split: {split}")
    return load_jsonl(path)


def _live_worker_context(
    args: argparse.Namespace, *, verify_data_files: bool = True
) -> tuple[dict[str, Any], dict[str, Any], Any, Any]:
    config_path = Path(args.config).resolve()
    repo_root = Path(args.repo_root).resolve()
    acceptance_path = Path(args.acceptance_artifact).resolve()
    config = validate_config(load_json(config_path))
    acceptance = load_acceptance(acceptance_path)
    verify_lock(
        load_json(Path(args.frozen_lock).resolve()),
        config_path=config_path,
        repo_root=repo_root,
        acceptance_path=acceptance_path,
        verify_data_files=verify_data_files,
    )
    if time.time() >= float(args.deadline_epoch):
        raise TimeoutError("GPU deadline reached before worker start")
    identity = _nvidia_identity()
    expected_uuid = acceptance["stack"]["gpu"]["uuid"]
    if identity != acceptance["stack"]["gpu"]:
        raise ProtocolError("live GPU identity differs from acceptance")
    assert_clean_gpu(expected_uuid, allowed_pids=set())
    model_path = resolve_model_path(config, args.model_path)
    _set_math_state()
    model, tokenizer = load_model(config, model_path)
    observed_stack = observe_stack(model)
    if observed_stack != acceptance["stack"]:
        raise ProtocolError("live software/hardware stack differs from acceptance")
    assert_clean_gpu(expected_uuid, allowed_pids={os.getpid()})
    return config, acceptance, model, tokenizer


def assert_numeric_logging_disabled() -> None:
    leaked = [
        name
        for name in ("CUBLASLT_LOG_LEVEL", "CUBLASLT_LOG_MASK", "CUBLASLT_LOG_FILE")
        if os.environ.get(name)
    ]
    if leaked:
        raise ProtocolError(f"numeric worker inherited cuBLASLt logging: {leaked}")


def worker_calibration(args: argparse.Namespace) -> int:
    assert_numeric_logging_disabled()
    import torch

    config, acceptance, model, tokenizer = _live_worker_context(
        args, verify_data_files=False
    )
    output_dir = Path(args.output_dir).resolve()
    gpu = _gpu()
    calibration_manifest = output_dir / "frozen_inputs" / "calibration_manifest.jsonl"
    if sha256_file(calibration_manifest) != config["data"][
        "calibration_manifest_sha256"
    ]:
        raise ProtocolError("frozen calibration manifest hash mismatch")
    calibration_captures = gpu.capture_olmoe_split(
        model=model,
        tokenizer=tokenizer,
        documents=load_jsonl(calibration_manifest),
        split="calibration",
        token_offsets=config["data"]["token_offsets"],
        window_tokens=int(config["data"]["window_tokens"]),
        add_special_tokens=bool(config["data"]["add_special_tokens"]),
        evaluation_position=int(config["data"]["evaluation_position"]),
    )
    if len(calibration_captures) != 16:
        raise ProtocolError("calibration capture window denominator mismatch")
    torch.save(calibration_captures, output_dir / "calibration_captures.pt")
    write_jsonl_no_overwrite(
        output_dir / "calibration_capture_manifest.jsonl",
        capture_audit_rows(calibration_captures),
    )

    rows = gpu.materialize_routed_rows(calibration_captures)
    expected_rows = (
        len(calibration_captures)
        * len(config["data"]["calibration_positions"])
        * int(config["model"]["num_hidden_layers"])
        * int(config["model"]["num_experts_per_tok"])
    )
    if len(rows) != expected_rows:
        raise ProtocolError("calibration routed-row denominator mismatch")
    packs = gpu.build_calibration_packs(
        rows, m_values=config["intervention"]["m_values"]
    )
    binding_context = descriptor_binding_context(config, acceptance)
    call_plan = gpu.calibration_call_plan(packs)
    calls = [
        planned_call_record(
            call, call_index=call_index, binding_context=binding_context
        )
        for call_index, call in enumerate(call_plan.calls)
    ]
    for call_index, call in enumerate(calls):
        validate_call_record(
            call, expected_index=call_index, binding_context=binding_context
        )
    write_jsonl_no_overwrite(output_dir / "calibration_calls.jsonl", calls)
    write_json_no_overwrite(
        output_dir / "calibration_pre_call_seal.json",
        {
            "schema_version": "semanticfence-pre-call-seal-v1",
            "status": "SEALED_BEFORE_EXECUTION",
            "call_count": len(calls),
            "calls_sha256": sha256_file(output_dir / "calibration_calls.jsonl"),
            "binding_context_sha256": canonical_sha256(binding_context),
        },
    )
    execution = gpu.execute_calibration(
        model=model,
        packs=packs,
        rows=rows,
        repeats=int(config["intervention"]["repeats"]),
    )
    torch.save(execution, output_dir / "calibration_raw_outputs.pt")
    write_jsonl_no_overwrite(
        output_dir / "calibration_reference_rows.jsonl",
        [row_execution_record(value) for value in execution.reference.rows],
    )
    calibration_records: list[dict[str, Any]] = []
    stack_digest = acceptance["stack"]["stack_digest"]
    for call_index, value in enumerate(execution.packs):
        pack = value.pack
        calibration_records.append(
            {
                "schema_version": "semanticfence-calibration-numeric-v1",
                "call_index": call_index,
                "arm": gpu.CALIBRATION_ARM,
                "pack_id": pack.pack_id,
                "layer": int(pack.layer),
                "expert_id": int(pack.expert_id),
                "m": int(pack.m),
                "row_ids": [row.row_id for row in pack.rows],
                "row_records": [row.identity_payload() for row in pack.rows],
                "repeat_row_exact": [list(repeat) for repeat in value.repeat_row_exact],
                "repeat_row_sha256": [list(repeat) for repeat in value.repeat_row_sha256],
                "representative_full_output_sha256": value.representative_full_output_sha256,
            }
        )
    write_jsonl_no_overwrite(
        output_dir / "calibration_numeric.jsonl", calibration_records
    )
    reference_all_stable = all(value.bitwise_stable for value in execution.reference.rows)
    write_json_no_overwrite(
        output_dir / "calibration_worker_status.json",
        {
            "status": "COMPLETE",
            "stack_digest": stack_digest,
            "calibration_window_count": len(calibration_captures),
            "calibration_row_count": len(rows),
            "calibration_pack_count": len(packs),
            "reference_all_stable": reference_all_stable,
            "calibration_calls_sha256": sha256_file(
                output_dir / "calibration_calls.jsonl"
            ),
            "calibration_numeric_sha256": sha256_file(
                output_dir / "calibration_numeric.jsonl"
            ),
            "calibration_reference_rows_sha256": sha256_file(
                output_dir / "calibration_reference_rows.jsonl"
            ),
            "calibration_pre_call_seal_sha256": sha256_file(
                output_dir / "calibration_pre_call_seal.json"
            ),
            "calibration_captures_sha256": sha256_file(
                output_dir / "calibration_captures.pt"
            ),
        },
    )
    assert_clean_gpu(
        acceptance["stack"]["gpu"]["uuid"], allowed_pids={os.getpid()}
    )
    return 0


def worker_trace(args: argparse.Namespace) -> int:
    import torch

    stage = str(args.stage)
    if stage not in {"calibration", "evaluation"}:
        raise ProtocolError("trace stage must be calibration or evaluation")
    config, acceptance, model, _tokenizer = _live_worker_context(
        args, verify_data_files=False
    )
    output_dir = Path(args.output_dir).resolve()
    if stage == "evaluation":
        load_sealed_contract(output_dir=output_dir, acceptance=acceptance)
    expected_level = str(config["intervention"]["cublaslt_log_level"])
    expected_mask = str(config["intervention"]["cublaslt_log_mask"])
    if os.environ.get("CUBLASLT_LOG_LEVEL") != expected_level:
        raise ProtocolError("trace worker log level mismatch")
    if os.environ.get("CUBLASLT_LOG_MASK") != expected_mask:
        raise ProtocolError("trace worker log mask mismatch")
    if not os.environ.get("CUBLASLT_LOG_FILE"):
        raise ProtocolError("trace worker log path is absent")
    capture_name = (
        "calibration_captures.pt"
        if stage == "calibration"
        else "evaluation_captures.pt"
    )
    calls_name = (
        "calibration_calls.jsonl" if stage == "calibration" else "evaluation_calls.jsonl"
    )
    # Register the dynamic CapturedWindow module before pickle resolves it.
    gpu = _gpu()
    captures = torch.load(
        output_dir / capture_name, map_location="cpu", weights_only=False
    )
    rows = gpu.materialize_routed_rows(captures)
    lookup = {row.row_id: row for row in rows}
    calls = load_jsonl(output_dir / calls_name)
    binding_context = descriptor_binding_context(config, acceptance)
    verify_pre_call_seal(
        output_dir=output_dir,
        stage=stage,
        calls=calls,
        binding_context=binding_context,
    )
    outputs: list[dict[str, Any]] = []
    hidden_size = int(config["model"]["hidden_size"])
    with torch.inference_mode():
        for call in calls:
            if time.time() >= float(args.deadline_epoch):
                raise TimeoutError(f"GPU deadline reached during {stage} trace")
            row_ids = list(call["row_ids"])
            try:
                materialized = [lookup[row_id] for row_id in row_ids]
            except KeyError as exc:
                raise ProtocolError("trace call references an unknown row") from exc
            if any(
                row.record.layer != int(call["layer"])
                or row.record.expert_id != int(call["expert_id"])
                for row in materialized
            ):
                raise ProtocolError("trace call crosses layer/expert identity")
            batch = torch.stack(
                [
                    row.tensor.to(device="cuda", dtype=torch.bfloat16)
                    for row in materialized
                ],
                dim=0,
            )
            padding_rows = int(call["padding_rows"])
            if padding_rows:
                batch = torch.cat(
                    (
                        batch,
                        torch.zeros(
                            (padding_rows, hidden_size),
                            device="cuda",
                            dtype=torch.bfloat16,
                        ),
                    ),
                    dim=0,
                )
            if int(batch.shape[0]) != int(call["m"]):
                raise ProtocolError("trace call batch M mismatch")
            expert = model.model.layers[int(call["layer"])].mlp.experts[
                int(call["expert_id"])
            ]
            result = expert(batch).detach()
            if result.dtype != torch.bfloat16 or tuple(result.shape) != (
                int(call["m"]),
                hidden_size,
            ):
                raise ProtocolError("trace expert output shape/dtype mismatch")
            identity = {
                key: call[key]
                for key in (
                    "call_index",
                    "arm",
                    "layer",
                    "expert_id",
                    "m",
                    "row_ids",
                )
            }
            outputs.append(
                identity
                | {"full_output_sha256": gpu.tensor_storage_sha256(result)}
            )
    torch.cuda.synchronize()
    write_jsonl_no_overwrite(
        output_dir / f"{stage}_trace_call_outputs.jsonl", outputs
    )
    write_json_no_overwrite(
        output_dir / f"{stage}_trace_worker_status.json",
        {
            "status": "COMPLETE",
            "stage": stage,
            "pid": os.getpid(),
            "call_count": len(calls),
            "call_output_sha256": sha256_file(
                output_dir / f"{stage}_trace_call_outputs.jsonl"
            ),
        },
    )
    assert_clean_gpu(
        acceptance["stack"]["gpu"]["uuid"], allowed_pids={os.getpid()}
    )
    return 0


def verify_calibration_reference_rows(
    *, config: Mapping[str, Any], output_dir: Path
) -> dict[str, str]:
    rows = load_jsonl(output_dir / "calibration_reference_rows.jsonl")
    expected_count = (
        16
        * len(config["data"]["calibration_positions"])
        * int(config["model"]["num_hidden_layers"])
        * int(config["model"]["num_experts_per_tok"])
    )
    if len(rows) != expected_count:
        raise ProtocolError("calibration reference row denominator mismatch")
    required = {
        "schema_version",
        "row_id",
        "reference_sha256",
        "repeat_sha256",
        "repeat_mismatch_counts",
        "bitwise_stable",
        "all_exact_to_reference",
    }
    references: dict[str, str] = {}
    for row in rows:
        if set(row) != required or row["schema_version"] != "semanticfence-row-execution-v1":
            raise ProtocolError("calibration reference row schema mismatch")
        row_id = row["row_id"]
        hashes = list(row["repeat_sha256"])
        mismatches = list(row["repeat_mismatch_counts"])
        if not is_sha256(row_id) or row_id in references:
            raise ProtocolError("calibration reference row identity mismatch")
        if len(hashes) != 10 or len(mismatches) != 10:
            raise ProtocolError("calibration reference does not have 10 repeats")
        if any(not is_sha256(value) for value in hashes):
            raise ProtocolError("calibration reference output hash is invalid")
        stable = len(set(hashes)) == 1
        exact = all(int(value) == 0 for value in mismatches)
        if row["reference_sha256"] != hashes[0] or not stable or not exact:
            raise ProtocolError("calibration M1 reference is not 10/10 stable")
        if bool(row["bitwise_stable"]) != stable or bool(
            row["all_exact_to_reference"]
        ) != exact:
            raise ProtocolError("calibration reference summary is inconsistent")
        references[row_id] = hashes[0]
    return references


def merge_calibration_contract(
    *, config: Mapping[str, Any], acceptance: Mapping[str, Any], output_dir: Path
) -> Any:
    calls = load_jsonl(output_dir / "calibration_calls.jsonl")
    numeric = load_jsonl(output_dir / "calibration_numeric.jsonl")
    trace_outputs = load_jsonl(output_dir / "calibration_trace_call_outputs.jsonl")
    binding_context = descriptor_binding_context(config, acceptance)
    verify_pre_call_seal(
        output_dir=output_dir,
        stage="calibration",
        calls=calls,
        binding_context=binding_context,
    )
    references = verify_calibration_reference_rows(
        config=config, output_dir=output_dir
    )
    worker_status = load_json(output_dir / "calibration_worker_status.json")
    if (
        worker_status.get("status") != "COMPLETE"
        or worker_status.get("reference_all_stable") is not True
        or worker_status.get("calibration_calls_sha256")
        != sha256_file(output_dir / "calibration_calls.jsonl")
        or worker_status.get("calibration_numeric_sha256")
        != sha256_file(output_dir / "calibration_numeric.jsonl")
        or worker_status.get("calibration_reference_rows_sha256")
        != sha256_file(output_dir / "calibration_reference_rows.jsonl")
    ):
        raise ProtocolError("calibration worker status/hash closure failed")
    if not (len(calls) == len(numeric) == len(trace_outputs)):
        raise ProtocolError("calibration call denominators differ")
    status = load_json(output_dir / "calibration_trace_worker_status.json")
    if (
        status.get("status") != "COMPLETE"
        or int(status.get("call_count", -1)) != len(calls)
        or status.get("call_output_sha256")
        != sha256_file(output_dir / "calibration_trace_call_outputs.jsonl")
    ):
        raise ProtocolError("calibration trace-worker status/hash closure failed")
    trace_path = output_dir / f"calibration_cublaslt_{int(status['pid'])}.log"
    if not trace_path.is_file():
        raise ProtocolError(f"calibration trace log is absent: {trace_path}")
    trace_rows = parse_trace_for_calls(
        trace_path=trace_path,
        calls=calls,
        hidden_size=int(config["model"]["hidden_size"]),
        intermediate_size=int(config["model"]["intermediate_size"]),
    )
    merged = bind_trace_to_numeric(
        trace_rows=trace_rows,
        trace_call_outputs=trace_outputs,
        numeric_calls=numeric,
    )
    write_jsonl_no_overwrite(output_dir / "calibration_trace.jsonl", trace_rows)
    observations: list[Any] = []
    numeric_required = {
        "schema_version",
        "call_index",
        "arm",
        "pack_id",
        "layer",
        "expert_id",
        "m",
        "row_ids",
        "row_records",
        "repeat_row_exact",
        "repeat_row_sha256",
        "representative_full_output_sha256",
    }
    for call_index, (call, numeric_row, merged_row) in enumerate(
        zip(calls, numeric, merged)
    ):
        if (
            set(numeric_row) != numeric_required
            or numeric_row["schema_version"] != "semanticfence-calibration-numeric-v1"
            or int(numeric_row["call_index"]) != call_index
            or numeric_row["arm"] != _gpu().CALIBRATION_ARM
            or numeric_row["arm"] != call["arm"]
        ):
            raise ProtocolError("calibration numeric row schema/identity mismatch")
        for key in ("layer", "expert_id", "m", "row_ids"):
            if numeric_row[key] != call[key]:
                raise ProtocolError("calibration numeric/call identity mismatch")
        if not is_sha256(numeric_row["representative_full_output_sha256"]):
            raise ProtocolError("calibration representative output hash is invalid")
        records = tuple(
            row_record_from_mapping(value) for value in numeric_row["row_records"]
        )
        pack = CONTRACT.Pack(
            layer=int(numeric_row["layer"]),
            expert_id=int(numeric_row["expert_id"]),
            rows=records,
        )
        if pack.pack_id != numeric_row["pack_id"]:
            raise ProtocolError("calibration pack identity changed during merge")
        if [row.row_id for row in records] != list(numeric_row["row_ids"]):
            raise ProtocolError("calibration row records do not match row ids")
        repeat_exact = list(numeric_row["repeat_row_exact"])
        repeat_hashes = list(numeric_row["repeat_row_sha256"])
        if len(repeat_exact) != 10 or len(repeat_hashes) != 10:
            raise ProtocolError("calibration numeric evidence is not 10 repeats")
        for flags, hashes in zip(repeat_exact, repeat_hashes):
            if len(flags) != pack.m or len(hashes) != pack.m:
                raise ProtocolError("calibration repeat width does not equal M")
            for row, flag, output_hash in zip(pack.rows, flags, hashes):
                if not isinstance(flag, bool) or not is_sha256(output_hash):
                    raise ProtocolError("calibration repeat evidence is malformed")
                if flag != (output_hash == references[row.row_id]):
                    raise ProtocolError("calibration exact flag/hash disagrees with M1")
        observations.append(
            CONTRACT.CalibrationObservation(
                pack=pack,
                signature=merged_row["signature_sha256"],
                repeat_row_exact=tuple(
                    tuple(value for value in repeat)
                    for repeat in repeat_exact
                ),
            )
        )
    contract = CONTRACT.build_contract(
        observations,
        stack_digest=acceptance["stack"]["stack_digest"],
        min_packs=int(config["intervention"]["min_calibration_packs"]),
        min_documents=int(config["intervention"]["min_calibration_documents"]),
    )
    for entry in contract.entries:
        if (
            entry.repeat_count != entry.pack_count * 10
            or entry.total_checks != entry.repeat_count * entry.m
        ):
            raise ProtocolError("contract entry repeat denominator is not frozen")
    contract_path = output_dir / "CONTRACT.json"
    write_json_no_overwrite(contract_path, contract.to_dict())
    write_json_no_overwrite(
        output_dir / "CONTRACT_SEAL.json",
        {
            "schema_version": "semanticfence-contract-seal-v1",
            "status": "SEALED_BEFORE_FRESH_EVALUATION",
            "contract_file_sha256": sha256_file(contract_path),
            "contract_sha256": contract.contract_sha256,
            "stack_digest": acceptance["stack"]["stack_digest"],
            "calibration_calls_sha256": sha256_file(
                output_dir / "calibration_calls.jsonl"
            ),
            "calibration_numeric_sha256": sha256_file(
                output_dir / "calibration_numeric.jsonl"
            ),
            "calibration_trace_sha256": sha256_file(
                output_dir / "calibration_trace.jsonl"
            ),
        },
    )
    return contract


def load_sealed_contract(
    *, output_dir: Path, acceptance: Mapping[str, Any]
) -> tuple[Any, dict[str, Any]]:
    contract_path = output_dir / "CONTRACT.json"
    seal = load_json(output_dir / "CONTRACT_SEAL.json")
    required = {
        "schema_version",
        "status",
        "contract_file_sha256",
        "contract_sha256",
        "stack_digest",
        "calibration_calls_sha256",
        "calibration_numeric_sha256",
        "calibration_trace_sha256",
    }
    if (
        set(seal) != required
        or seal["schema_version"] != "semanticfence-contract-seal-v1"
        or seal["status"] != "SEALED_BEFORE_FRESH_EVALUATION"
        or seal["contract_file_sha256"] != sha256_file(contract_path)
        or seal["stack_digest"] != acceptance["stack"]["stack_digest"]
        or seal["calibration_calls_sha256"]
        != sha256_file(output_dir / "calibration_calls.jsonl")
        or seal["calibration_numeric_sha256"]
        != sha256_file(output_dir / "calibration_numeric.jsonl")
        or seal["calibration_trace_sha256"]
        != sha256_file(output_dir / "calibration_trace.jsonl")
    ):
        raise ProtocolError("sealed contract artifact/hash closure failed")
    contract = CONTRACT.validate_contract(load_json(contract_path))
    if (
        contract.contract_sha256 != seal["contract_sha256"]
        or contract.stack_digest != seal["stack_digest"]
    ):
        raise ProtocolError("contract content differs from its pre-eval seal")
    return contract, seal


def worker_evaluation(args: argparse.Namespace) -> int:
    assert_numeric_logging_disabled()
    import torch

    config, acceptance, model, tokenizer = _live_worker_context(
        args, verify_data_files=False
    )
    output_dir = Path(args.output_dir).resolve()
    gpu = _gpu()
    contract, contract_seal = load_sealed_contract(
        output_dir=output_dir, acceptance=acceptance
    )
    evaluation_manifest = output_dir / "frozen_inputs" / "evaluation_manifest.jsonl"
    if sha256_file(evaluation_manifest) != config["data"][
        "evaluation_manifest_sha256"
    ]:
        raise ProtocolError("frozen evaluation manifest hash mismatch")
    # The fresh manifest is first materialized as a model workload only after
    # calibration, trace merge, and the content-addressed contract seal.
    captures = gpu.capture_olmoe_split(
        model=model,
        tokenizer=tokenizer,
        documents=load_jsonl(evaluation_manifest),
        split="semanticfence_eval_fresh",
        token_offsets=config["data"]["token_offsets"],
        window_tokens=int(config["data"]["window_tokens"]),
        add_special_tokens=bool(config["data"]["add_special_tokens"]),
        evaluation_position=int(config["data"]["evaluation_position"]),
    )
    if len(captures) != 64:
        raise ProtocolError("evaluation capture window denominator mismatch")
    torch.save(captures, output_dir / "evaluation_captures.pt")
    write_jsonl_no_overwrite(
        output_dir / "evaluation_capture_manifest.jsonl",
        capture_audit_rows(captures),
    )
    rows = gpu.materialize_routed_rows(captures)
    expected_rows = (
        len(captures)
        * 1
        * int(config["model"]["num_hidden_layers"])
        * int(config["model"]["num_experts_per_tok"])
    )
    if len(rows) != expected_rows:
        raise ProtocolError("evaluation routed-row denominator mismatch")
    write_jsonl_no_overwrite(
        output_dir / "evaluation_row_context.jsonl",
        [
            {
                "schema_version": "semanticfence-evaluation-row-context-v1",
                "row_id": row.row_id,
                "window_id": row.context.window_id,
                "absolute_token_position": int(row.context.absolute_token_position),
                "window_token_id": int(row.context.window_token_id),
                "routing_weight": float(row.context.routing_weight),
                "row_record": row.record.identity_payload(),
            }
            for row in sorted(rows, key=lambda value: value.row_id)
        ],
    )
    stack_digest = acceptance["stack"]["stack_digest"]
    planner_started = time.perf_counter()
    plans = gpu.plan_four_arms(rows, contract=contract, stack_digest=stack_digest)
    planner_cpu_ms = (time.perf_counter() - planner_started) * 1000.0
    order = tuple(gpu.FROZEN_ARM_ORDER)
    binding_context = descriptor_binding_context(config, acceptance)
    calls: list[dict[str, Any]] = []
    expected_signatures: list[dict[str, Any]] = []
    calls_by_arm: dict[str, list[dict[str, Any]]] = {}
    global_index = 0
    for arm in order:
        plan = plans[arm]
        arm_calls: list[dict[str, Any]] = []
        start_index = global_index
        for call in plan.calls:
            global_record = planned_call_record(
                call, call_index=global_index, binding_context=binding_context
            )
            calls.append(global_record)
            arm_calls.append(global_record)
            expected_signatures.append(
                {
                    "schema_version": "semanticfence-expected-signature-v1",
                    "call_index": global_index,
                    "expected_signatures": list(call.expected_signatures),
                }
            )
            global_index += 1
        validate_call_ledger(
            [row.record for row in rows],
            arm_calls,
            binding_context=binding_context,
            starting_index=start_index,
        )
        calls_by_arm[arm] = arm_calls
    write_jsonl_no_overwrite(output_dir / "evaluation_calls.jsonl", calls)
    write_jsonl_no_overwrite(
        output_dir / "evaluation_expected_signatures.jsonl", expected_signatures
    )
    paired_schedule = {
        "schema_version": "semanticfence-paired-schedule-v1",
        "status": "FROZEN_BEFORE_EXECUTION",
        "warmups": [
            {"warmup_id": index, "arm_order": list(gpu.frozen_arm_order(index))}
            for index in range(int(config["intervention"]["warmups"]))
        ],
        "paired_repeats": [
            {"pair_id": index, "arm_order": list(gpu.frozen_arm_order(index))}
            for index in range(int(config["intervention"]["repeats"]))
        ],
    }
    write_json_no_overwrite(
        output_dir / "evaluation_paired_schedule.json", paired_schedule
    )
    write_json_no_overwrite(
        output_dir / "evaluation_pre_call_seal.json",
        {
            "schema_version": "semanticfence-pre-call-seal-v1",
            "status": "SEALED_BEFORE_EXECUTION",
            "call_count": len(calls),
            "calls_sha256": sha256_file(output_dir / "evaluation_calls.jsonl"),
            "binding_context_sha256": canonical_sha256(binding_context),
            "paired_schedule_sha256": sha256_file(
                output_dir / "evaluation_paired_schedule.json"
            ),
        },
    )
    verify_pre_call_seal(
        output_dir=output_dir,
        stage="evaluation",
        calls=calls,
        binding_context=binding_context,
    )

    executions = gpu.execute_paired_arms(
        model=model,
        plans=plans,
        rows=rows,
        warmups=int(config["intervention"]["warmups"]),
        repeats=int(config["intervention"]["repeats"]),
        raw_output_dir=output_dir / "evaluation_raw_bf16",
    )
    raw_files: list[dict[str, Any]] = []
    expected_raw_bytes = len(rows) * int(config["model"]["hidden_size"]) * 2
    for pair_id in range(int(config["intervention"]["repeats"])):
        for arm in order:
            raw_path = (
                output_dir
                / "evaluation_raw_bf16"
                / f"pair_{pair_id:02d}_{arm}.bf16"
            )
            if raw_path.stat().st_size != expected_raw_bytes:
                raise ProtocolError("evaluation raw BF16 file size mismatch")
            raw_files.append(
                {
                    "pair_id": pair_id,
                    "arm": arm,
                    "relative_path": str(raw_path.relative_to(output_dir)),
                    "bytes": expected_raw_bytes,
                    "sha256": sha256_file(raw_path),
                }
            )
    write_json_no_overwrite(
        output_dir / "evaluation_raw_output_index.json",
        {
            "schema_version": "semanticfence-raw-bf16-index-v1",
            "dtype": "bfloat16-raw-little-endian-u16",
            "row_order": sorted(row.row_id for row in rows),
            "hidden_size": int(config["model"]["hidden_size"]),
            "files": raw_files,
        },
    )
    torch.save(executions, output_dir / "evaluation_raw_outputs.pt")
    numeric_calls: list[dict[str, Any]] = []
    for arm in order:
        execution = executions[arm]
        if len(execution.representative_call_output_sha256) != len(calls_by_arm[arm]):
            raise ProtocolError("evaluation representative call denominator mismatch")
        for call, output_hash in zip(
            calls_by_arm[arm], execution.representative_call_output_sha256
        ):
            numeric_calls.append(
                {
                    "schema_version": "semanticfence-evaluation-numeric-call-v1",
                    **{
                        key: call[key]
                        for key in (
                            "call_index",
                            "arm",
                            "layer",
                            "expert_id",
                            "m",
                            "row_ids",
                        )
                    },
                    "representative_full_output_sha256": output_hash,
                }
            )
    write_jsonl_no_overwrite(
        output_dir / "evaluation_numeric_calls.jsonl", numeric_calls
    )

    row_context = {row.row_id: row.context.window_id for row in rows}
    summaries: list[dict[str, Any]] = []
    for arm in order:
        execution = executions[arm]
        summaries.append(
            {
                "schema_version": "semanticfence-arm-result-v1",
                "arm": arm,
                "warmups": execution.warmups,
                "repeats": execution.repeats,
                "pair_ids": list(range(execution.repeats)),
                "latency_ms": list(execution.latency_ms),
                "median_latency_ms": statistics.median(execution.latency_ms),
                "call_count": execution.call_count,
                "padding_rows": execution.padding_rows,
                "mismatch_row_count": sum(
                    not value.all_exact_to_reference for value in execution.rows
                ),
                "unstable_row_count": sum(
                    not value.bitwise_stable for value in execution.rows
                ),
                "rows": [row_execution_record(value) for value in execution.rows],
            }
        )
    write_json_no_overwrite(
        output_dir / "evaluation_arm_results.json",
        {"schema_version": "semanticfence-arm-results-v1", "arms": summaries},
    )

    by_arm = {value["arm"]: value for value in summaries}
    b_mismatch_ids = {
        value.row_id
        for value in executions[gpu.ARM_B].rows
        if value.bitwise_stable
        and all(count > 0 for count in value.repeat_mismatch_counts)
    }
    d_mismatch_count = sum(
        not value.all_exact_to_reference for value in executions[gpu.ARM_D].rows
    )
    d_covered_ids = {
        row_id
        for call in plans[gpu.ARM_D].calls
        if call.execution_m > 1
        for row_id in call.row_ids
    }
    d_ms = {
        int(call.execution_m)
        for call in plans[gpu.ARM_D].calls
        if call.execution_m > 1
    }
    c_mismatch_count = int(by_arm[gpu.ARM_C]["mismatch_row_count"])
    fixed_dominates = (
        c_mismatch_count <= d_mismatch_count
        and float(by_arm[gpu.ARM_C]["median_latency_ms"])
        <= float(by_arm[gpu.ARM_D]["median_latency_ms"])
    )
    decision_input = {
        "reference_all_stable": all(
            value.bitwise_stable for value in executions[gpu.ARM_A].rows
        ),
        "unrestricted_mismatch_victims": len(
            {row_context[row_id] for row_id in b_mismatch_ids}
        ),
        "semanticfence_mismatch_rows": d_mismatch_count,
        "semanticfence_covered_victims": len(
            {row_context[row_id] for row_id in d_covered_ids}
        ),
        "semanticfence_distinct_m_gt_1": len(d_ms),
        "semanticfence_padding_rows": int(executions[gpu.ARM_D].padding_rows),
        "semanticfence_latency_reduction_fraction": paired_latency_reduction(
            executions[gpu.ARM_A].latency_ms,
            executions[gpu.ARM_D].latency_ms,
        ),
        "fixed_control_dominates": fixed_dominates,
        "evidence_complete": True,
    }
    write_json_no_overwrite(
        output_dir / "decision_input_numeric.json", decision_input
    )
    write_json_no_overwrite(
        output_dir / "evaluation_worker_status.json",
        {
            "status": "COMPLETE",
            "row_count": len(rows),
            "call_count": len(calls),
            "planner_cpu_ms": planner_cpu_ms,
            "contract_file_sha256": contract_seal["contract_file_sha256"],
            "contract_sha256": contract.contract_sha256,
            "evaluation_captures_sha256": sha256_file(
                output_dir / "evaluation_captures.pt"
            ),
            "evaluation_capture_manifest_sha256": sha256_file(
                output_dir / "evaluation_capture_manifest.jsonl"
            ),
            "evaluation_row_context_sha256": sha256_file(
                output_dir / "evaluation_row_context.jsonl"
            ),
            "evaluation_calls_sha256": sha256_file(
                output_dir / "evaluation_calls.jsonl"
            ),
            "evaluation_pre_call_seal_sha256": sha256_file(
                output_dir / "evaluation_pre_call_seal.json"
            ),
            "evaluation_paired_schedule_sha256": sha256_file(
                output_dir / "evaluation_paired_schedule.json"
            ),
            "evaluation_numeric_calls_sha256": sha256_file(
                output_dir / "evaluation_numeric_calls.jsonl"
            ),
            "evaluation_arm_results_sha256": sha256_file(
                output_dir / "evaluation_arm_results.json"
            ),
            "evaluation_raw_output_index_sha256": sha256_file(
                output_dir / "evaluation_raw_output_index.json"
            ),
            "decision_input_sha256": sha256_file(
                output_dir / "decision_input_numeric.json"
            ),
        },
    )
    assert_clean_gpu(
        acceptance["stack"]["gpu"]["uuid"], allowed_pids={os.getpid()}
    )
    return 0


def _worker_command(
    args: argparse.Namespace, worker: str, *, stage: str | None, deadline_epoch: float
) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        worker,
        "--config",
        str(Path(args.config).resolve()),
        "--repo-root",
        str(Path(args.repo_root).resolve()),
        "--acceptance-artifact",
        str(Path(args.acceptance_artifact).resolve()),
        "--frozen-lock",
        str(Path(args.frozen_lock).resolve()),
        "--output-dir",
        str(Path(args.output_dir).resolve()),
        "--deadline-epoch",
        str(deadline_epoch),
    ]
    if stage is not None:
        command.extend(("--stage", stage))
    if args.model_path:
        command.extend(("--model-path", str(args.model_path)))
    return command


def run_worker_monitored(
    *,
    command: Sequence[str],
    log_path: Path,
    expected_gpu_uuid: str,
    deadline_epoch: float,
    extra_env: Mapping[str, str] | None = None,
) -> None:
    if log_path.exists():
        raise ProtocolError(f"worker log exists: {log_path}")
    environment = os.environ.copy()
    for name in ("CUBLASLT_LOG_LEVEL", "CUBLASLT_LOG_MASK", "CUBLASLT_LOG_FILE"):
        environment.pop(name, None)
    environment.update(extra_env or {})
    with log_path.open("x", encoding="utf-8") as log:
        process = subprocess.Popen(
            list(command),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            env=environment,
        )
        try:
            while process.poll() is None:
                if time.time() >= deadline_epoch:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                    raise TimeoutError("SemanticFence GPU hard deadline exceeded")
                processes = _nvidia_compute_processes()
                foreign = [
                    row
                    for row in processes
                    if row["gpu_uuid"] != expected_gpu_uuid
                    or int(row["pid"]) != process.pid
                ]
                if foreign:
                    process.terminate()
                    raise ContaminationError(
                        f"continuous monitor found foreign GPU process: {foreign!r}"
                    )
                time.sleep(0.5)
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
            log.flush()
            os.fsync(log.fileno())
        if process.returncode != 0:
            raise ProtocolError(
                f"worker exited {process.returncode}; inspect {log_path.name}"
            )


def snapshot_inputs(
    *,
    output_dir: Path,
    repo_root: Path,
    config_path: Path,
    acceptance_path: Path,
    lock_path: Path,
    config: Mapping[str, Any],
) -> dict[str, str]:
    snapshot = output_dir / "frozen_inputs"
    snapshot.mkdir()
    sources: dict[str, Path] = {
        "config.json": config_path,
        "ACCEPTANCE.json": acceptance_path,
        "ACCEPTANCE_COMPLETE.json": acceptance_path.parent
        / "ACCEPTANCE_COMPLETE.json",
        "FROZEN_RUN_LOCK.json": lock_path,
        "calibration_manifest.jsonl": _repo_path(
            repo_root, str(config["data"]["calibration_manifest"])
        ),
        "evaluation_manifest.jsonl": _repo_path(
            repo_root, str(config["data"]["evaluation_manifest"])
        ),
        "calibration_provenance.json": _repo_path(
            repo_root, str(config["data"]["calibration_provenance"])
        ),
        "historical_hash_registry.json": _repo_path(
            repo_root, str(config["data"]["historical_hash_registry"])
        ),
        "evaluation_provenance.json": _repo_path(
            repo_root, str(config["data"]["evaluation_provenance"])
        ),
        "evaluation_exclusion_report.json": _repo_path(
            repo_root, str(config["data"]["evaluation_exclusion_report"])
        ),
        "evaluation_artifact_hashes.json": _repo_path(
            repo_root, str(config["data"]["evaluation_artifact_hashes"])
        ),
    }
    for relative in source_bindings(repo_root, config_path):
        sources[relative.replace("/", "__")] = _repo_path(repo_root, relative)
    result: dict[str, str] = {}
    for name, source in sorted(sources.items()):
        target = snapshot / name
        shutil.copyfile(source, target)
        target.chmod(0o444)
        result[name] = sha256_file(target)
    return result


def recompute_evaluation_from_raw(
    *,
    config: Mapping[str, Any],
    acceptance: Mapping[str, Any],
    output_dir: Path,
    calls: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Independently derive every numeric decision field from raw BF16 files."""

    import numpy as np

    gpu = _gpu()
    order = tuple(gpu.FROZEN_ARM_ORDER)
    binding_context = descriptor_binding_context(config, acceptance)
    verify_pre_call_seal(
        output_dir=output_dir,
        stage="evaluation",
        calls=calls,
        binding_context=binding_context,
    )

    contexts = load_jsonl(output_dir / "evaluation_row_context.jsonl")
    expected_row_count = (
        64
        * int(config["model"]["num_hidden_layers"])
        * int(config["model"]["num_experts_per_tok"])
    )
    if len(contexts) != expected_row_count:
        raise ProtocolError("evaluation row-context denominator mismatch")
    capture_rows = load_jsonl(output_dir / "evaluation_capture_manifest.jsonl")
    capture_required = {
        "schema_version",
        "window_id",
        "split",
        "document_sha256",
        "document_index",
        "offset",
        "selected_positions",
        "window_token_ids",
        "full_hidden_states_sha256",
        "selected_experts_sha256",
        "routing_weights_sha256",
        "router_logits_sha256_by_layer",
    }
    capture_by_window: dict[str, tuple[str, int, int]] = {}
    if len(capture_rows) != 64:
        raise ProtocolError("evaluation capture-audit denominator mismatch")
    for capture in capture_rows:
        expected_window_id = canonical_sha256(
            {
                "schema_version": capture.get("schema_version"),
                "split": capture.get("split"),
                "document_sha256": capture.get("document_sha256"),
                "document_index": int(capture.get("document_index", -1)),
                "offset": int(capture.get("offset", -1)),
                "window_token_ids": list(capture.get("window_token_ids", [])),
            }
        )
        if (
            set(capture) != capture_required
            or capture["schema_version"] != "semanticfence-olmoe-capture-v1"
            or capture["split"] != "evaluation"
            or capture["selected_positions"] != [15]
            or len(capture["window_token_ids"]) != 16
            or len(capture["router_logits_sha256_by_layer"]) != 16
            or capture["window_id"] != expected_window_id
            or capture["window_id"] in capture_by_window
            or any(
                not is_sha256(capture[key])
                for key in (
                    "document_sha256",
                    "full_hidden_states_sha256",
                    "selected_experts_sha256",
                    "routing_weights_sha256",
                )
            )
            or any(
                not is_sha256(value)
                for value in capture["router_logits_sha256_by_layer"]
            )
        ):
            raise ProtocolError("evaluation capture-audit identity mismatch")
        capture_by_window[capture["window_id"]] = (
            capture["document_sha256"],
            int(capture["document_index"]),
            int(capture["offset"]),
        )
    context_by_row: dict[str, str] = {}
    for value in contexts:
        required = {
            "schema_version",
            "row_id",
            "window_id",
            "absolute_token_position",
            "window_token_id",
            "routing_weight",
            "row_record",
        }
        if (
            set(value) != required
            or value["schema_version"] != "semanticfence-evaluation-row-context-v1"
        ):
            raise ProtocolError("evaluation row-context schema mismatch")
        record = row_record_from_mapping(value["row_record"])
        capture_identity = capture_by_window.get(value["window_id"])
        if (
            record.split != "evaluation"
            or record.row_id != value["row_id"]
            or record.row_id in context_by_row
            or not is_sha256(value["window_id"])
            or capture_identity
            != (
                record.document_sha256,
                int(record.document_index),
                int(record.offset),
            )
        ):
            raise ProtocolError("evaluation row-context identity mismatch")
        context_by_row[record.row_id] = value["window_id"]
    if len(set(context_by_row.values())) != 64:
        raise ProtocolError("evaluation victim/window denominator mismatch")

    arm_calls: dict[str, list[Mapping[str, Any]]] = {arm: [] for arm in order}
    for call_index, call in enumerate(calls):
        validate_call_record(
            call, expected_index=call_index, binding_context=binding_context
        )
        if call["arm"] not in arm_calls:
            raise ProtocolError("evaluation call has an unknown arm")
        arm_calls[call["arm"]].append(call)
    start = 0
    frozen_traversal: list[str] | None = None
    row_records = [
        row_record_from_mapping(value["row_record"]) for value in contexts
    ]
    for arm in order:
        validate_call_ledger(
            row_records,
            arm_calls[arm],
            binding_context=binding_context,
            starting_index=start,
        )
        traversal = [row_id for call in arm_calls[arm] for row_id in call["row_ids"]]
        if frozen_traversal is None:
            frozen_traversal = traversal
        elif traversal != frozen_traversal:
            raise ProtocolError("evaluation arms changed logical row traversal")
        start += len(arm_calls[arm])
    if start != len(calls):
        raise ProtocolError("evaluation call blocks are not contiguous by arm")

    schedule = load_json(output_dir / "evaluation_paired_schedule.json")
    expected_schedule = {
        "schema_version": "semanticfence-paired-schedule-v1",
        "status": "FROZEN_BEFORE_EXECUTION",
        "warmups": [
            {"warmup_id": index, "arm_order": list(gpu.frozen_arm_order(index))}
            for index in range(int(config["intervention"]["warmups"]))
        ],
        "paired_repeats": [
            {"pair_id": index, "arm_order": list(gpu.frozen_arm_order(index))}
            for index in range(int(config["intervention"]["repeats"]))
        ],
    }
    if schedule != expected_schedule:
        raise ProtocolError("evaluation paired schedule drifted")

    raw_index = load_json(output_dir / "evaluation_raw_output_index.json")
    row_order = sorted(context_by_row)
    if (
        raw_index.get("schema_version") != "semanticfence-raw-bf16-index-v1"
        or raw_index.get("dtype") != "bfloat16-raw-little-endian-u16"
        or raw_index.get("row_order") != row_order
        or int(raw_index.get("hidden_size", -1))
        != int(config["model"]["hidden_size"])
    ):
        raise ProtocolError("evaluation raw-output index header mismatch")
    files = raw_index.get("files")
    if not isinstance(files, list) or len(files) != len(order) * 10:
        raise ProtocolError("evaluation raw-output file denominator mismatch")
    file_by_key: dict[tuple[int, str], Mapping[str, Any]] = {}
    expected_bytes = len(row_order) * int(config["model"]["hidden_size"]) * 2
    for entry in files:
        key = (int(entry.get("pair_id", -1)), str(entry.get("arm", "")))
        if key in file_by_key or key[0] not in range(10) or key[1] not in order:
            raise ProtocolError("evaluation raw-output file identity mismatch")
        expected_relative = (
            f"evaluation_raw_bf16/pair_{key[0]:02d}_{key[1]}.bf16"
        )
        if entry.get("relative_path") != expected_relative:
            raise ProtocolError("evaluation raw-output path is not canonical")
        if int(entry.get("bytes", -1)) != expected_bytes:
            raise ProtocolError("evaluation raw-output declared size mismatch")
        file_by_key[key] = entry
    if set(file_by_key) != {(pair, arm) for pair in range(10) for arm in order}:
        raise ProtocolError("evaluation raw-output file matrix is incomplete")

    def raw_bytes(pair_id: int, arm: str) -> bytes:
        entry = file_by_key[(pair_id, arm)]
        path = (output_dir / str(entry["relative_path"])).resolve()
        try:
            path.relative_to(output_dir.resolve())
        except ValueError as exc:
            raise ProtocolError("raw-output path escapes output root") from exc
        payload = path.read_bytes()
        if len(payload) != expected_bytes or hashlib.sha256(payload).hexdigest() != entry.get(
            "sha256"
        ):
            raise ProtocolError("evaluation raw-output file hash mismatch")
        return payload

    reference_raw = raw_bytes(0, gpu.ARM_A)
    hidden_size = int(config["model"]["hidden_size"])
    reference_array = np.frombuffer(reference_raw, dtype="<u2").reshape(
        len(row_order), hidden_size
    )
    hashes_by_arm = {
        arm: [[] for _ in row_order] for arm in order
    }
    mismatches_by_arm = {
        arm: [[] for _ in row_order] for arm in order
    }
    row_bytes = hidden_size * 2
    for pair_id in range(10):
        for arm in order:
            payload = reference_raw if (pair_id, arm) == (0, gpu.ARM_A) else raw_bytes(
                pair_id, arm
            )
            values = np.frombuffer(payload, dtype="<u2").reshape(
                len(row_order), hidden_size
            )
            counts = np.count_nonzero(values != reference_array, axis=1).tolist()
            for row_index in range(len(row_order)):
                begin = row_index * row_bytes
                output_hash = hashlib.sha256(
                    payload[begin : begin + row_bytes]
                ).hexdigest()
                hashes_by_arm[arm][row_index].append(output_hash)
                mismatches_by_arm[arm][row_index].append(int(counts[row_index]))

    arm_artifact = load_json(output_dir / "evaluation_arm_results.json")
    if arm_artifact.get("schema_version") != "semanticfence-arm-results-v1":
        raise ProtocolError("evaluation arm-results header mismatch")
    arm_rows = arm_artifact.get("arms")
    if not isinstance(arm_rows, list) or [value.get("arm") for value in arm_rows] != list(order):
        raise ProtocolError("evaluation arm-results order mismatch")
    verified: dict[str, dict[str, Any]] = {}
    reference_hashes = hashes_by_arm[gpu.ARM_A]
    for arm, summary in zip(order, arm_rows):
        required = {
            "schema_version",
            "arm",
            "warmups",
            "repeats",
            "pair_ids",
            "latency_ms",
            "median_latency_ms",
            "call_count",
            "padding_rows",
            "mismatch_row_count",
            "unstable_row_count",
            "rows",
        }
        if set(summary) != required or summary["schema_version"] != "semanticfence-arm-result-v1":
            raise ProtocolError("evaluation arm-result schema mismatch")
        latencies = [float(value) for value in summary["latency_ms"]]
        if (
            int(summary["warmups"]) != 3
            or int(summary["repeats"]) != 10
            or summary["pair_ids"] != list(range(10))
            or len(latencies) != 10
            or any(value <= 0 or not math.isfinite(value) for value in latencies)
            or not math.isclose(
                float(summary["median_latency_ms"]),
                float(statistics.median(latencies)),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise ProtocolError("evaluation paired latency evidence mismatch")
        reported_rows = summary["rows"]
        if not isinstance(reported_rows, list) or len(reported_rows) != len(row_order):
            raise ProtocolError("evaluation arm row denominator mismatch")
        mismatch_rows = 0
        unstable_rows = 0
        stable_mismatch_ids: set[str] = set()
        for row_index, (row_id, reported) in enumerate(zip(row_order, reported_rows)):
            hashes = hashes_by_arm[arm][row_index]
            mismatches = mismatches_by_arm[arm][row_index]
            stable = len(set(hashes)) == 1
            exact = all(value == 0 for value in mismatches)
            expected_report = {
                "schema_version": "semanticfence-row-execution-v1",
                "row_id": row_id,
                "reference_sha256": reference_hashes[row_index][0],
                "repeat_sha256": hashes,
                "repeat_mismatch_counts": mismatches,
                "bitwise_stable": stable,
                "all_exact_to_reference": exact,
            }
            if reported != expected_report:
                raise ProtocolError("evaluation row result differs from raw BF16 bits")
            mismatch_rows += int(not exact)
            unstable_rows += int(not stable)
            if stable and all(value > 0 for value in mismatches):
                stable_mismatch_ids.add(row_id)
        if (
            int(summary["mismatch_row_count"]) != mismatch_rows
            or int(summary["unstable_row_count"]) != unstable_rows
            or int(summary["call_count"]) != len(arm_calls[arm])
            or int(summary["padding_rows"])
            != sum(int(call["padding_rows"]) for call in arm_calls[arm])
        ):
            raise ProtocolError("evaluation arm aggregate differs from raw/calls")
        verified[arm] = {
            "latency_ms": latencies,
            "median_latency_ms": statistics.median(latencies),
            "mismatch_rows": mismatch_rows,
            "unstable_rows": unstable_rows,
            "stable_mismatch_ids": stable_mismatch_ids,
        }

    d_covered_ids = {
        row_id
        for call in arm_calls[gpu.ARM_D]
        if int(call["m"]) > 1
        for row_id in call["row_ids"]
    }
    d_ms = {
        int(call["m"]) for call in arm_calls[gpu.ARM_D] if int(call["m"]) > 1
    }
    d_mismatch = int(verified[gpu.ARM_D]["mismatch_rows"])
    c_mismatch = int(verified[gpu.ARM_C]["mismatch_rows"])
    decision_input = {
        "reference_all_stable": verified[gpu.ARM_A]["unstable_rows"] == 0,
        "unrestricted_mismatch_victims": len(
            {
                context_by_row[row_id]
                for row_id in verified[gpu.ARM_B]["stable_mismatch_ids"]
            }
        ),
        "semanticfence_mismatch_rows": d_mismatch,
        "semanticfence_covered_victims": len(
            {context_by_row[row_id] for row_id in d_covered_ids}
        ),
        "semanticfence_distinct_m_gt_1": len(d_ms),
        "semanticfence_padding_rows": sum(
            int(call["padding_rows"]) for call in arm_calls[gpu.ARM_D]
        ),
        "semanticfence_latency_reduction_fraction": paired_latency_reduction(
            verified[gpu.ARM_A]["latency_ms"], verified[gpu.ARM_D]["latency_ms"]
        ),
        "fixed_control_dominates": (
            c_mismatch <= d_mismatch
            and verified[gpu.ARM_C]["median_latency_ms"]
            <= verified[gpu.ARM_D]["median_latency_ms"]
        ),
        "evidence_complete": True,
    }
    return decision_input, {
        "status": "PARENT_RECOMPUTED_FROM_RAW_BF16",
        "row_count": len(row_order),
        "victim_count": len(set(context_by_row.values())),
        "raw_file_count": len(files),
        "calls_sha256": sha256_file(output_dir / "evaluation_calls.jsonl"),
        "raw_index_sha256": sha256_file(
            output_dir / "evaluation_raw_output_index.json"
        ),
        "arm_results_sha256": sha256_file(
            output_dir / "evaluation_arm_results.json"
        ),
    }


def merge_evaluation(
    *,
    config: Mapping[str, Any],
    acceptance: Mapping[str, Any],
    output_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    load_sealed_contract(output_dir=output_dir, acceptance=acceptance)
    calls = load_jsonl(output_dir / "evaluation_calls.jsonl")
    numeric_calls = load_jsonl(output_dir / "evaluation_numeric_calls.jsonl")
    trace_outputs = load_jsonl(output_dir / "evaluation_trace_call_outputs.jsonl")
    expected = load_jsonl(output_dir / "evaluation_expected_signatures.jsonl")
    worker_status = load_json(output_dir / "evaluation_worker_status.json")
    status_bindings = {
        "evaluation_captures_sha256": "evaluation_captures.pt",
        "evaluation_capture_manifest_sha256": "evaluation_capture_manifest.jsonl",
        "evaluation_calls_sha256": "evaluation_calls.jsonl",
        "evaluation_numeric_calls_sha256": "evaluation_numeric_calls.jsonl",
        "evaluation_arm_results_sha256": "evaluation_arm_results.json",
        "evaluation_raw_output_index_sha256": "evaluation_raw_output_index.json",
        "evaluation_row_context_sha256": "evaluation_row_context.jsonl",
        "evaluation_pre_call_seal_sha256": "evaluation_pre_call_seal.json",
        "evaluation_paired_schedule_sha256": "evaluation_paired_schedule.json",
        "decision_input_sha256": "decision_input_numeric.json",
    }
    if worker_status.get("status") != "COMPLETE":
        raise ProtocolError("evaluation worker status is incomplete")
    for status_key, artifact_name in status_bindings.items():
        if worker_status.get(status_key) != sha256_file(output_dir / artifact_name):
            raise ProtocolError(f"evaluation worker hash closure failed: {status_key}")
    binding_context = descriptor_binding_context(config, acceptance)
    verify_pre_call_seal(
        output_dir=output_dir,
        stage="evaluation",
        calls=calls,
        binding_context=binding_context,
    )
    if not (len(calls) == len(numeric_calls) == len(trace_outputs) == len(expected)):
        raise ProtocolError("evaluation call evidence denominators differ")
    numeric_required = {
        "schema_version",
        "call_index",
        "arm",
        "layer",
        "expert_id",
        "m",
        "row_ids",
        "representative_full_output_sha256",
    }
    for index, (call, numeric) in enumerate(zip(calls, numeric_calls)):
        if (
            set(numeric) != numeric_required
            or numeric["schema_version"] != "semanticfence-evaluation-numeric-call-v1"
            or not is_sha256(numeric["representative_full_output_sha256"])
        ):
            raise ProtocolError("evaluation numeric-call schema mismatch")
        for key in ("call_index", "arm", "layer", "expert_id", "m", "row_ids"):
            if numeric[key] != call[key]:
                raise ProtocolError("evaluation numeric/call identity mismatch")
        if int(call["call_index"]) != index:
            raise ProtocolError("evaluation call index mismatch")
    status = load_json(output_dir / "evaluation_trace_worker_status.json")
    if (
        status.get("status") != "COMPLETE"
        or int(status.get("call_count", -1)) != len(calls)
        or status.get("call_output_sha256")
        != sha256_file(output_dir / "evaluation_trace_call_outputs.jsonl")
    ):
        raise ProtocolError("evaluation trace-worker status/hash closure failed")
    trace_path = output_dir / f"evaluation_cublaslt_{int(status['pid'])}.log"
    if not trace_path.is_file():
        raise ProtocolError("evaluation cuBLASLt trace log is absent")
    trace_rows = parse_trace_for_calls(
        trace_path=trace_path,
        calls=calls,
        hidden_size=int(config["model"]["hidden_size"]),
        intermediate_size=int(config["model"]["intermediate_size"]),
    )
    trace_validation: dict[str, Any] = {
        "trace_complete": True,
        "numeric_trace_output_match": True,
        "semanticfence_signature_match": True,
        "worker_parent_numeric_match": True,
        "errors": [],
    }
    try:
        bind_trace_to_numeric(
            trace_rows=trace_rows,
            trace_call_outputs=trace_outputs,
            numeric_calls=numeric_calls,
        )
    except ProtocolError as error:
        trace_validation["numeric_trace_output_match"] = False
        trace_validation["errors"].append(str(error))
    if len(expected) != len(trace_rows):
        raise ProtocolError("evaluation expected-signature denominator mismatch")
    for call, expected_row, traced in zip(calls, expected, trace_rows):
        if (
            set(expected_row)
            != {"schema_version", "call_index", "expected_signatures"}
            or expected_row["schema_version"] != "semanticfence-expected-signature-v1"
            or int(expected_row["call_index"]) != int(traced["call_index"])
            or list(expected_row["expected_signatures"])
            != list(call["expected_signatures"])
        ):
            raise ProtocolError("expected-signature call identity mismatch")
        signatures = list(expected_row["expected_signatures"])
        if traced["arm"] == _gpu().ARM_D and int(traced["m"]) > 1:
            if len(signatures) != 1 or signatures[0] != traced["signature_sha256"]:
                trace_validation["semanticfence_signature_match"] = False
                trace_validation["errors"].append(
                    f"D call {traced['call_index']} expected {signatures} observed {traced['signature_sha256']}"
                )
    parent_input, parent_validation = recompute_evaluation_from_raw(
        config=config,
        acceptance=acceptance,
        output_dir=output_dir,
        calls=calls,
    )
    worker_input = load_json(output_dir / "decision_input_numeric.json")
    for key, parent_value in parent_input.items():
        worker_value = worker_input.get(key)
        matches = (
            math.isclose(
                float(worker_value), float(parent_value), rel_tol=0.0, abs_tol=1e-12
            )
            if isinstance(parent_value, float) and isinstance(worker_value, (int, float))
            else worker_value == parent_value
        )
        if not matches:
            trace_validation["worker_parent_numeric_match"] = False
            trace_validation["errors"].append(
                f"worker/parent decision field mismatch: {key}"
            )
    decision_input = dict(parent_input)
    decision_input["evidence_complete"] = bool(
        parent_input["evidence_complete"]
        and trace_validation["trace_complete"]
        and trace_validation["numeric_trace_output_match"]
        and trace_validation["semanticfence_signature_match"]
        and trace_validation["worker_parent_numeric_match"]
    )
    write_jsonl_no_overwrite(output_dir / "evaluation_trace.jsonl", trace_rows)
    write_json_no_overwrite(
        output_dir / "parent_recompute.json",
        {
            "schema_version": "semanticfence-parent-recompute-v1",
            "decision_input": decision_input,
            "validation": parent_validation,
        },
    )
    write_json_no_overwrite(
        output_dir / "evaluation_trace_validation.json", trace_validation
    )
    return decision_input, trace_validation


def run_pilot(args: argparse.Namespace) -> int:
    config_path = Path(args.config).resolve()
    repo_root = Path(args.repo_root).resolve()
    acceptance_path = Path(args.acceptance_artifact).resolve()
    lock_path = Path(args.frozen_lock).resolve()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise ProtocolError(f"pilot output exists: {output_dir}")
    config = validate_config(load_json(config_path))
    acceptance = load_acceptance(acceptance_path)
    verify_lock(
        load_json(lock_path),
        config_path=config_path,
        repo_root=repo_root,
        acceptance_path=acceptance_path,
    )
    expected_uuid = acceptance["stack"]["gpu"]["uuid"]
    if _nvidia_identity() != acceptance["stack"]["gpu"]:
        raise ProtocolError("parent GPU identity differs from acceptance")
    assert_clean_gpu(expected_uuid, allowed_pids=set())
    output_dir.mkdir(parents=True)
    started = time.time()
    deadline = started + int(config["budget"]["max_gpu_seconds"])
    try:
        snapshots = snapshot_inputs(
            output_dir=output_dir,
            repo_root=repo_root,
            config_path=config_path,
            acceptance_path=acceptance_path,
            lock_path=lock_path,
            config=config,
        )
        write_json_no_overwrite(
            output_dir / "run_request.json",
            {
                "schema_version": "semanticfence-p0-run-request-v1",
                "started_at_epoch": started,
                "deadline_epoch": deadline,
                "config_sha256": sha256_file(config_path),
                "acceptance_sha256": sha256_file(acceptance_path),
                "lock_sha256": sha256_file(lock_path),
                "snapshot_sha256": snapshots,
                "evidence_boundary": config["evidence_boundary"],
            },
        )
        run_worker_monitored(
            command=_worker_command(
                args, "_worker-calibration", stage=None, deadline_epoch=deadline
            ),
            log_path=output_dir / "calibration_worker.log",
            expected_gpu_uuid=expected_uuid,
            deadline_epoch=deadline,
        )
        calibration_status = load_json(
            output_dir / "calibration_worker_status.json"
        )
        if (
            calibration_status.get("status") != "COMPLETE"
            or calibration_status.get("calibration_reference_rows_sha256")
            != sha256_file(output_dir / "calibration_reference_rows.jsonl")
        ):
            raise ProtocolError("calibration status does not bind M1 references")
        try:
            verify_calibration_reference_rows(config=config, output_dir=output_dir)
        except ProtocolError as error:
            if "not 10/10 stable" not in str(error):
                raise
            summary = {
                "schema_version": SUMMARY_SCHEMA,
                "decision": "UNABLE",
                "reason": "parent recompute found all-row isolated M1 instability",
                "paper_result": False,
                "wall_seconds": time.time() - started,
                "evidence_boundary": config["evidence_boundary"],
            }
            finalize_complete(output_dir, summary)
            return 0

        cal_trace_template = output_dir / "calibration_cublaslt_%i.log"
        run_worker_monitored(
            command=_worker_command(
                args,
                "_worker-trace",
                stage="calibration",
                deadline_epoch=deadline,
            ),
            log_path=output_dir / "calibration_trace_worker.log",
            expected_gpu_uuid=expected_uuid,
            deadline_epoch=deadline,
            extra_env={
                "CUBLASLT_LOG_LEVEL": str(
                    config["intervention"]["cublaslt_log_level"]
                ),
                "CUBLASLT_LOG_MASK": str(
                    config["intervention"]["cublaslt_log_mask"]
                ),
                "CUBLASLT_LOG_FILE": str(cal_trace_template),
            },
        )
        contract = merge_calibration_contract(
            config=config, acceptance=acceptance, output_dir=output_dir
        )
        sealed_contract, contract_seal = load_sealed_contract(
            output_dir=output_dir, acceptance=acceptance
        )
        if sealed_contract.contract_sha256 != contract.contract_sha256:
            raise ProtocolError("parent contract differs immediately after seal")
        run_worker_monitored(
            command=_worker_command(
                args, "_worker-evaluation", stage=None, deadline_epoch=deadline
            ),
            log_path=output_dir / "evaluation_worker.log",
            expected_gpu_uuid=expected_uuid,
            deadline_epoch=deadline,
        )
        evaluation_status = load_json(output_dir / "evaluation_worker_status.json")
        sealed_after_evaluation, _seal_after_evaluation = load_sealed_contract(
            output_dir=output_dir, acceptance=acceptance
        )
        if (
            sealed_after_evaluation.contract_sha256 != contract.contract_sha256
            or evaluation_status.get("contract_sha256") != contract.contract_sha256
            or evaluation_status.get("contract_file_sha256")
            != contract_seal["contract_file_sha256"]
        ):
            raise ProtocolError("evaluation did not use the parent-sealed contract")
        eval_trace_template = output_dir / "evaluation_cublaslt_%i.log"
        run_worker_monitored(
            command=_worker_command(
                args,
                "_worker-trace",
                stage="evaluation",
                deadline_epoch=deadline,
            ),
            log_path=output_dir / "evaluation_trace_worker.log",
            expected_gpu_uuid=expected_uuid,
            deadline_epoch=deadline,
            extra_env={
                "CUBLASLT_LOG_LEVEL": str(
                    config["intervention"]["cublaslt_log_level"]
                ),
                "CUBLASLT_LOG_MASK": str(
                    config["intervention"]["cublaslt_log_mask"]
                ),
                "CUBLASLT_LOG_FILE": str(eval_trace_template),
            },
        )
        decision_input, trace_validation = merge_evaluation(
            config=config, acceptance=acceptance, output_dir=output_dir
        )
        sealed_after_trace, _seal_after_trace = load_sealed_contract(
            output_dir=output_dir, acceptance=acceptance
        )
        if sealed_after_trace.contract_sha256 != contract.contract_sha256:
            raise ProtocolError("contract changed before final decision")
        decision = decide_summary(decision_input, config)
        summary = {
            "schema_version": SUMMARY_SCHEMA,
            "decision": decision,
            "paper_result": False,
            "decision_input": decision_input,
            "trace_validation": trace_validation,
            "contract_sha256": contract.contract_sha256,
            "allowed_contract_entry_count": sum(
                int(entry.allowed) for entry in contract.entries
            ),
            "wall_seconds": time.time() - started,
            "within_gpu_budget": time.time() <= deadline,
            "evidence_boundary": config["evidence_boundary"],
            "authorized_next_step": (
                "full_layer_propagation_pilot" if decision == "SUPPORT" else None
            ),
        }
        if not summary["within_gpu_budget"]:
            summary["decision"] = "UNABLE"
            summary["authorized_next_step"] = None
        finalize_complete(output_dir, summary)
        return 0
    except BaseException as error:
        failure = output_dir / "failure.json"
        if not failure.exists():
            write_json_no_overwrite(
                failure,
                {
                    "decision": "UNABLE",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "paper_result": False,
                    "complete_sentinel_written": False,
                },
            )
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    acceptance = subparsers.add_parser("acceptance")
    acceptance.add_argument("--config", required=True)
    acceptance.add_argument("--repo-root", default=str(REPO_ROOT_DEFAULT))
    acceptance.add_argument("--output-dir", required=True)
    acceptance.add_argument("--model-path")

    seal = subparsers.add_parser("seal")
    seal.add_argument("--config", required=True)
    seal.add_argument("--repo-root", default=str(REPO_ROOT_DEFAULT))
    seal.add_argument("--acceptance-artifact", required=True)
    seal.add_argument("--output", required=True)

    run = subparsers.add_parser("run")
    run.add_argument("--config", required=True)
    run.add_argument("--repo-root", default=str(REPO_ROOT_DEFAULT))
    run.add_argument("--acceptance-artifact", required=True)
    run.add_argument("--frozen-lock", required=True)
    run.add_argument("--output-dir", required=True)
    run.add_argument("--model-path")

    for worker_name in ("_worker-calibration", "_worker-evaluation", "_worker-trace"):
        worker = subparsers.add_parser(worker_name, help=argparse.SUPPRESS)
        worker.add_argument("--config", required=True)
        worker.add_argument("--repo-root", required=True)
        worker.add_argument("--acceptance-artifact", required=True)
        worker.add_argument("--frozen-lock", required=True)
        worker.add_argument("--output-dir", required=True)
        worker.add_argument("--deadline-epoch", required=True, type=float)
        worker.add_argument("--model-path")
        if worker_name == "_worker-trace":
            worker.add_argument(
                "--stage", required=True, choices=("calibration", "evaluation")
            )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "acceptance":
        return run_acceptance(args)
    if args.command == "seal":
        return run_seal(args)
    if args.command == "_worker-calibration":
        return worker_calibration(args)
    if args.command == "_worker-evaluation":
        return worker_evaluation(args)
    if args.command == "_worker-trace":
        return worker_trace(args)
    return run_pilot(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ProtocolError as error:
        print(f"INVALID: {error}", file=sys.stderr)
        raise SystemExit(2)
