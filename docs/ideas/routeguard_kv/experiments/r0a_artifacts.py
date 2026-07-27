#!/usr/bin/env python3
"""Fail-closed artifact and binding helpers for RouteGuard-KV R0-A."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import tempfile
from typing import Any, Iterable, Mapping, Sequence


class ArtifactError(RuntimeError):
    """An R0-A artifact is missing, mutable, incompatible, or incomplete."""


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def ordered_hash_of_hashes(values: Sequence[str]) -> str:
    return sha256_bytes(("\n".join(values) + "\n").encode("ascii"))


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"cannot load JSON artifact {path}: {exc}") from exc


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ArtifactError(f"{path}:{line_number} is not a JSON object")
            rows.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"cannot load JSONL artifact {path}: {exc}") from exc
    return rows


def write_json_no_overwrite(path: Path, value: Any, *, mode: int | None = None) -> None:
    """Create canonical JSON atomically and refuse to replace an existing artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise ArtifactError(f"refusing to overwrite artifact: {path}")
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(canonical_json_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
        if mode is not None:
            temporary.chmod(mode)
        # link() is atomic and, unlike replace(), cannot overwrite the target.
        os.link(temporary, path)
    except FileExistsError as exc:
        raise ArtifactError(f"refusing to overwrite artifact: {path}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def write_jsonl_no_overwrite(
    path: Path, rows: Iterable[Mapping[str, Any]], *, mode: int | None = None
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            for row in rows:
                handle.write(canonical_json_bytes(dict(row)).decode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise ArtifactError(f"refusing to overwrite artifact: {path}") from exc
    if mode is not None:
        path.chmod(mode)


def append_jsonl_fsync(path: Path, row: Mapping[str, Any]) -> None:
    """Append one journal record using one O_APPEND write followed by fsync."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_bytes(dict(row))
    fd = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        written = os.write(fd, payload)
        if written != len(payload):
            raise ArtifactError(f"short journal write to {path}: {written}/{len(payload)}")
        os.fsync(fd)
    finally:
        os.close(fd)


def validate_config(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != "routeguard-kv-r0a-5090-v1":
        raise ArtifactError("unexpected RouteGuard-KV config schema")
    if config.get("status") != "DESIGN_FROZEN_GPU_NOT_APPROVED_SEALED_PREPARED":
        raise ArtifactError("config status is not the frozen pre-run status")
    dataset = config["dataset"]
    if int(dataset["required_tokens"]) != max(dataset["prompt_lengths"]) + int(
        dataset["decode_steps"]
    ) + 1:
        raise ArtifactError("required_tokens must equal max prompt + decode steps + 1")
    if max(dataset["prompt_lengths"]) + int(dataset["decode_steps"]) >= int(
        config["model"]["expected"]["max_position_embeddings"]
    ):
        raise ArtifactError("decode window exceeds the model position domain")
    quant = config["quantization"]["primary"]
    if (quant["bits"], quant["qmin"], quant["qmax"], quant["group_axis"]) != (
        4,
        0,
        15,
        "head_dim",
    ):
        raise ArtifactError("primary quantizer is not the frozen INT4 contract")
    route = config["route_lock"]
    if route["set_locked_renormalization"] is not config["model"]["expected"]["norm_topk_prob"]:
        raise ArtifactError("route-lock renormalization differs from native OLMoE config")
    stats = config["statistics"]
    if stats["primary_cell"] != {
        "target": "k_only",
        "prompt_length": 2048,
        "mean_free_kl_min": 0.0001,
        "identity_kl_denominator_floor": 1e-12,
        "free_kl_over_identity_min": 100.0,
        "non_tie_set_flip_rate_min": 0.01,
        "router_contrast_lcb_min_exclusive": 0.0,
        "router_share_point_min": 0.4,
        "router_share_lcb_min_exclusive": 0.25,
        "leave_one_document_out_all_positive": True,
        "non_tie_fraction_of_flips_min": 0.9,
    }:
        raise ArtifactError("primary thresholds differ from the frozen contract")
    if stats["secondary_cells_can_rescue_primary"] is not False:
        raise ArtifactError("secondary cells must not rescue the primary")


def load_config(path: Path) -> dict[str, Any]:
    value = load_json(path)
    if not isinstance(value, dict):
        raise ArtifactError("config root must be an object")
    validate_config(value)
    return value


def assert_formal_approval(approval_path: Path, config: Mapping[str, Any], bindings_sha256: str) -> None:
    approval = load_json(approval_path)
    required = config["approval"]["required_literal"]
    if approval.get("approval") != required:
        raise ArtifactError(f"formal run requires approval literal {required!r}")
    if approval.get("frozen_bindings_sha256") != bindings_sha256:
        raise ArtifactError("approval is not bound to this frozen artifact set")


def build_frozen_bindings(
    *,
    repo_root: Path,
    config_path: Path,
    manifest_path: Path,
    source_paths: Sequence[Path],
) -> dict[str, Any]:
    config = load_config(config_path)
    protocol_path = repo_root / str(config["protocol"])
    if not protocol_path.is_file():
        raise ArtifactError(f"protocol does not exist: {protocol_path}")
    files = [config_path, protocol_path, manifest_path, *source_paths]
    normalized: dict[str, str] = {}
    for path in files:
        resolved = path.resolve()
        try:
            label = str(resolved.relative_to(repo_root.resolve()))
        except ValueError:
            label = str(resolved)
        if label in normalized:
            continue
        normalized[label] = sha256_file(resolved)
    return {
        "schema_version": "routeguard-kv-frozen-bindings-v1",
        "config_canonical_sha256": canonical_json_sha256(config),
        "files": dict(sorted(normalized.items())),
    }


def verify_frozen_bindings(bindings: Mapping[str, Any], repo_root: Path) -> None:
    if bindings.get("schema_version") != "routeguard-kv-frozen-bindings-v1":
        raise ArtifactError("unexpected frozen bindings schema")
    for label, expected in bindings.get("files", {}).items():
        path = Path(label)
        if not path.is_absolute():
            path = repo_root / path
        actual = sha256_file(path)
        if actual != expected:
            raise ArtifactError(f"frozen artifact hash mismatch for {label}: {actual} != {expected}")


def _cuda_driver_version(cuda_available: bool) -> str | None:
    """Return the NVIDIA driver version from the stable nvidia-smi CLI."""

    if not cuda_available:
        return None
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    versions = {line.strip() for line in result.stdout.splitlines() if line.strip()}
    if len(versions) != 1:
        return None
    return versions.pop()


def environment_snapshot() -> dict[str, Any]:
    import torch
    import transformers

    cuda_available = torch.cuda.is_available()
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "datasets": importlib.metadata.version("datasets"),
        "huggingface_hub": importlib.metadata.version("huggingface-hub"),
        "numpy": importlib.metadata.version("numpy"),
        "cuda_available": cuda_available,
        "torch_cuda": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if cuda_available else None,
        "compute_capability": list(torch.cuda.get_device_capability(0)) if cuda_available else None,
        "cuda_driver": _cuda_driver_version(cuda_available),
    }


def assert_5090_environment(config: Mapping[str, Any]) -> None:
    snapshot = environment_snapshot()
    if snapshot["torch"] != config["software"]["torch"]:
        raise ArtifactError(f"torch identity mismatch: {snapshot['torch']}")
    if snapshot["transformers"] != config["software"]["transformers"]:
        raise ArtifactError(f"transformers identity mismatch: {snapshot['transformers']}")
    if snapshot["datasets"] != config["software"]["datasets"]:
        raise ArtifactError(f"datasets identity mismatch: {snapshot['datasets']}")
    if snapshot["huggingface_hub"] != config["software"]["huggingface_hub"]:
        raise ArtifactError(f"huggingface_hub identity mismatch: {snapshot['huggingface_hub']}")
    if platform.python_version_tuple()[:2] not in {("3", "10"), ("3", "11")}:
        raise ArtifactError(f"Python identity mismatch: {snapshot['python']}")
    if snapshot["gpu_name"] != config["hardware"]["gpu_exact_name"]:
        raise ArtifactError(f"GPU identity mismatch: {snapshot['gpu_name']}")
    expected_cc = tuple(int(part) for part in config["hardware"]["compute_capability"].split("."))
    if tuple(snapshot["compute_capability"] or ()) != expected_cc:
        raise ArtifactError(f"compute capability mismatch: {snapshot['compute_capability']}")
    if snapshot["cuda_driver"] is None:
        raise ArtifactError("CUDA driver version could not be recorded")
