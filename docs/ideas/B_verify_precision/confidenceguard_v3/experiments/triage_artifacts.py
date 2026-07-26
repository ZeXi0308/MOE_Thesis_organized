"""Fail-closed artifact and resume helpers for TriageAudit runs."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Iterable, Mapping


class ArtifactError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_manifest(paths: Iterable[Path], *, root: Path | None = None) -> dict[str, object]:
    resolved = sorted({path.resolve() for path in paths})
    if not resolved or any(not path.is_file() for path in resolved):
        raise ArtifactError("source manifest contains a missing/non-file path")
    root_resolved = root.resolve() if root is not None else None
    files: dict[str, str] = {}
    for path in resolved:
        if root_resolved is None:
            key = str(path)
        else:
            try:
                key = str(path.relative_to(root_resolved))
            except ValueError as exc:
                raise ArtifactError(f"source path is outside manifest root: {path}") from exc
        files[key] = sha256_file(path)
    aggregate = hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {"schema_version": "triage-source-manifest-v2", "files": files, "aggregate_sha256": aggregate}


def environment_snapshot() -> dict[str, object]:
    result: dict[str, object] = {
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
        "pid": os.getpid(),
    }
    try:
        import torch

        result.update(
            torch_version=torch.__version__,
            torch_cuda_version=torch.version.cuda,
            cuda_available=torch.cuda.is_available(),
            gpu_name=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            gpu_total_bytes=int(torch.cuda.get_device_properties(0).total_memory) if torch.cuda.is_available() else None,
        )
    except ImportError:
        result["torch_version"] = None
    try:
        result["git_commit"] = subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
        result["git_dirty"] = bool(
            subprocess.run(["git", "status", "--porcelain"], check=True, capture_output=True, text=True).stdout
        )
    except (OSError, subprocess.CalledProcessError):
        result["git_commit"] = None
        result["git_dirty"] = None
    return result


def write_json_no_overwrite(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
    except FileExistsError as exc:
        raise ArtifactError(f"refusing to overwrite {path}") from exc


class JsonlJournal:
    def __init__(self, path: Path, *, resume: bool):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and not resume:
            raise ArtifactError(f"journal exists without --resume: {path}")
        self.completed_keys: set[str] = set()
        if path.exists():
            for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ArtifactError(f"corrupt journal line {line_number}") from exc
                key = row.get("resume_key")
                if not isinstance(key, str) or not key:
                    raise ArtifactError(f"journal line {line_number} lacks resume_key")
                if key in self.completed_keys:
                    raise ArtifactError(f"duplicate resume_key: {key}")
                self.completed_keys.add(key)

    def append(self, row: Mapping[str, object]) -> None:
        key = row.get("resume_key")
        if not isinstance(key, str) or not key:
            raise ArtifactError("journal row requires resume_key")
        if key in self.completed_keys:
            raise ArtifactError(f"duplicate resume_key: {key}")
        payload = json.dumps(dict(row), ensure_ascii=False, sort_keys=True, allow_nan=False)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(payload + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self.completed_keys.add(key)
