#!/usr/bin/env python3
"""Run one frozen G64 inflight guard qualification under the shared lock and existing A model.

The C source and separate A private model are rehashed before any GPU cell.
The controller never copies or removes either model repository.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import stat
import subprocess
import sys
import time
from typing import Any


PACKAGE_NAME = "candidate_g64_inflight_guard_qual_r01"
PACKAGE_SHA256 = "2013572450702fc7033cdfe8d35a5d781a57c9c0fbe7ec8aa986b3ec243160d6"
FROZEN_ARMS = ("ltr_t30_q1",)
GPU_UUID_RE = re.compile(r"GPU-[0-9a-fA-F-]{36}\Z")
SHA_RE = re.compile(r"[0-9a-f]{64}\Z")
LOCK_ID_RE = re.compile(r"[1-9][0-9]*:[1-9][0-9]*\Z")
MODEL_VERIFY_TIMEOUT_S = 360
MODEL_VIEW_TIMEOUT_S = 480
MODEL_RESOLVE_TIMEOUT_S = 60
EXPECTED_PRIVATE_PYTHON = "/root/autodl-tmp/moe-a-runtime-20260930/venv/bin/python"
EXPECTED_MODEL_CACHE = "/root/moe-a-model-cache-20260930-r02/hf"
EXPECTED_SHARED_MODEL_SOURCE_CACHE = "/root/autodl-tmp/c-research-20260930/hf-cache"
EXPECTED_MODEL_REVISION = "6d84c48581ece794365f2b8e9cfb043c68ade9c5"
MODEL_REPO_NAME = "models--allenai--OLMoE-1B-7B-0924"
MODEL_ID = "allenai/OLMoE-1B-7B-0924"
MODEL_VERIFIER_PATH = (
    "/root/autodl-tmp/moe-a-shared-assets-launch-20260930-v2/"
    "verify_shared_model_20260930.py"
)
MODEL_VERIFIER_SHA256 = "d78c6726a8d010f3a02d50c79220084520c186f9b05f4b4a10c1598ac5fe9a90"
MODEL_METADATA_SHA256 = {
    "config.json": "3643aa880d2f1c9b418156269ae791c73e5612d6b6b6fde0724d927cf89b6335",
    "generation_config.json": "d77272ffaa7e62a904e8e130bb25ab11585bd4a5026e388d6d682e4b82892ce2",
    "model.safetensors.index.json": "0e2e1e0d8d357ac7af817cff28410c3dbad398f060c517a433e4076b2aae5579",
    "special_tokens_map.json": "b77491e270c6fcc5b2ecf22370f7318a6a18d3cabea09ba7bab92e9bf12656c2",
    "tokenizer.json": "a094266ac6c4982efba277bc251349a5a6d6ad37efb39a2a90f53d8be2a40a40",
    "tokenizer_config.json": "78a839c7851f14f9fb30e664c2b46166dc0628f2900679e5ec160656f702edff",
}
MODEL_SHARDS = {
    "model-00001-of-00003.safetensors": (4997744872, "5e3cff7e367794685c241169072c940d200918617d5e2813f1c387dff52d845e"),
    "model-00002-of-00003.safetensors": (4997235176, "15ef5c730ee3cfed7199498788cd2faf337203fc74b529625e7502cdd759f4a7"),
    "model-00003-of-00003.safetensors": (3843741912, "a9abac4ac1b55c9adabac721a02fa39971f103eea9a65c310972b1246de76e04"),
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def absolute_path(value: Any, label: str) -> Path:
    require(isinstance(value, str) and value.startswith("/"), f"{label} must be absolute")
    return Path(value).resolve(strict=False)


def positive_int(value: Any, label: str) -> int:
    require(type(value) is int and value > 0, f"{label} must be a positive integer")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_lock_identity(fd: int, path: Path, expected: str) -> str:
    held = os.fstat(fd)
    named = path.stat()
    actual = f"{held.st_dev}:{held.st_ino}"
    require(held.st_dev == named.st_dev and held.st_ino == named.st_ino,
            "shared lock path no longer names the held inode")
    require(actual == expected, "shared lock differs from the frozen host inode")
    return actual


def is_within(path: Path, parent: Path) -> bool:
    return path == parent or parent in path.parents


def validate_plan(plan: Any) -> dict[str, Any]:
    require(isinstance(plan, dict) and plan.get("schema_version") == 1, "schema_version must be 1")
    require(isinstance(plan.get("authorization_reference"), str) and
            plan["authorization_reference"].strip(), "authorization reference required")
    gpu_uuid = plan.get("authorized_gpu_uuid")
    require(isinstance(gpu_uuid, str) and GPU_UUID_RE.fullmatch(gpu_uuid) is not None,
            "approved physical GPU UUID required")
    positive_int(plan.get("approved_host_bytes"), "approved_host_bytes")
    total_s = positive_int(plan.get("approved_total_wall_seconds"), "approved_total_wall_seconds")
    require(total_s > 52, "total wall budget must leave room for shutdown and archive")
    lock_path = absolute_path(plan.get("lock_path"), "lock_path")
    lock_id = plan.get("expected_lock_device_inode")
    require(isinstance(lock_id, str) and LOCK_ID_RE.fullmatch(lock_id) is not None,
            "expected_lock_device_inode must be a frozen device:inode pair")
    session_dir = absolute_path(plan.get("session_dir"), "session_dir")
    require(not session_dir.exists(), "session_dir must be new")
    for key in ("python", "hf_cache_dir", "shared_model_source_cache",
                "cgroup_memory_max_file"):
        absolute_path(plan.get(key), key)
    require(plan["python"] == EXPECTED_PRIVATE_PYTHON,
            "plan Python differs from the frozen A private runtime")
    require(plan["hf_cache_dir"] == EXPECTED_MODEL_CACHE,
            "plan HF cache differs from the frozen A private model cache")
    require(plan["shared_model_source_cache"] == EXPECTED_SHARED_MODEL_SOURCE_CACHE,
            "plan shared model source differs from the frozen C cache")
    require(Path(plan["python"]).is_file() and os.access(plan["python"], os.X_OK),
            "A private Python executable is missing")
    require(Path(plan["hf_cache_dir"]).is_dir() and
            not Path(plan["hf_cache_dir"]).is_symlink(),
            "existing A private model cache missing or symlinked")
    require(Path(plan["shared_model_source_cache"]).is_dir() and
            not Path(plan["shared_model_source_cache"]).is_symlink(),
            "shared model source cache is missing")
    require(not is_within(Path(plan["hf_cache_dir"]),
                          Path(plan["shared_model_source_cache"])) and
            not is_within(Path(plan["shared_model_source_cache"]),
                          Path(plan["hf_cache_dir"])),
            "A private cache overlaps C source cache")
    require(plan.get("model_revision") == EXPECTED_MODEL_REVISION,
            "model revision differs from the frozen identity")
    require(plan.get("model_verifier_path") == MODEL_VERIFIER_PATH,
            "model verifier path differs from the frozen identity")
    require(plan.get("model_verifier_sha256") == MODEL_VERIFIER_SHA256,
            "model verifier SHA-256 differs from the frozen identity")
    verifier = Path(MODEL_VERIFIER_PATH)
    require(verifier.is_file() and not verifier.is_symlink(),
            "frozen model verifier missing or symlinked")
    require(sha256_file(verifier) == MODEL_VERIFIER_SHA256,
            "frozen model verifier file bytes differ")
    cells = plan.get("cells")
    require(isinstance(cells, list) and 0 < len(cells) <= len(FROZEN_ARMS),
            "cells must contain one frozen G64 diagnostic arm")
    seen_outputs: set[Path] = set()
    seen_arms: set[str] = set()
    packages: list[Path] = []
    for index, cell in enumerate(cells):
        require(isinstance(cell, dict), f"cell {index} must be an object")
        arm = cell.get("arm")
        require(arm in FROZEN_ARMS, f"cell {index} has an unknown diagnostic arm")
        package_text = cell.get("package_dir")
        package = absolute_path(package_text, f"cell {index} package_dir")
        require(not Path(package_text).is_symlink() and package.is_dir(),
                f"cell {index} package missing or symlink")
        require(package.name == PACKAGE_NAME, f"cell {index} package name mismatch")
        require(sha256_file(package / "manifest.json") == PACKAGE_SHA256,
                f"cell {index} package manifest differs from frozen identity")
        launcher = package / "pkg" / "run.sh"
        require(launcher.is_file() and not launcher.is_symlink(), f"cell {index} launcher missing")
        output = absolute_path(cell.get("output_dir"), f"cell {index} output_dir")
        require(not output.exists() and output not in seen_outputs,
                f"cell {index} output_dir must be new and unique")
        require(not is_within(output, package) and not is_within(session_dir, package),
                f"cell {index} output/session path inside package")
        require(not is_within(output, session_dir) and not is_within(session_dir, output),
                f"cell {index} output and session paths overlap")
        max_s = positive_int(cell.get("max_wall_seconds"), f"cell {index} max_wall_seconds")
        require(max_s > 6, f"cell {index} wall budget too short")
        require(not any(k in cell for k in ("kind", "variant", "mode", "gate")),
                f"cell {index} includes obsolete controller selectors")
        require(arm not in seen_arms, f"cell {index} repeats a diagnostic arm")
        seen_arms.add(arm)
        seen_outputs.add(output)
        packages.append(package)
    for index, output in enumerate(seen_outputs):
        require(not any(is_within(output, p) for p in packages),
                f"output {index} overlaps another package")
    require(len(set(packages)) == 1, "all cells must use one frozen candidate package")
    private_cache_path = Path(plan["hf_cache_dir"]).resolve(strict=True)
    source_cache_path = Path(plan["shared_model_source_cache"]).resolve(strict=True)
    for path in (session_dir, *seen_outputs):
        require(not is_within(path, private_cache_path) and
                not is_within(path, source_cache_path),
                "session or output path must not enter a model cache")
        require(not is_within(lock_path, path),
                "session or output path must not contain the shared lock")
    for package in packages:
        require(not is_within(package, private_cache_path) and
                not is_within(package, source_cache_path),
                "candidate package must not enter a model cache")
    require(not is_within(lock_path, session_dir) and not any(is_within(lock_path, p) for p in packages),
            "lock path must live outside session and candidate packages")
    requested_cell_s = sum(cell["max_wall_seconds"] for cell in cells)
    require(total_s >= requested_cell_s + MODEL_VERIFY_TIMEOUT_S +
            MODEL_VIEW_TIMEOUT_S + MODEL_RESOLVE_TIMEOUT_S + 45 * len(cells) + 20,
            "total wall budget cannot fund model hashing, private rehash, full cells and archive")
    return plan


def write_json(path: Path, payload: Any) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, ensure_ascii=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def remaining_seconds(deadline: float) -> float:
    return deadline - time.monotonic()


def gpu_process_rows(deadline: float) -> list[str]:
    remain = remaining_seconds(deadline)
    require(remain > 5, "total wall budget exhausted before GPU process check")
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
        text=True, capture_output=True, timeout=min(10.0, remain - 2), check=False,
    )
    require(result.returncode == 0, "nvidia-smi process query failed")
    rows = [row.strip() for row in result.stdout.splitlines() if row.strip()]
    return [row for row in rows if row.lower() != "no running processes found"]


def require_authorized_gpu(plan: dict[str, Any], deadline: float) -> None:
    remain = remaining_seconds(deadline)
    require(remain > 5, "total wall budget exhausted before GPU identity check")
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
        text=True, capture_output=True, timeout=min(10.0, remain - 2), check=False,
    )
    require(result.returncode == 0, "nvidia-smi GPU identity query failed")
    uuids = [line.strip().replace(" ", "") for line in result.stdout.splitlines() if line.strip()]
    require(uuids.count(plan["authorized_gpu_uuid"]) == 1 and len(uuids) == 1,
            "authorized single GPU UUID absent or ambiguous")


def validate_model_receipt(payload: Any, plan: dict[str, Any]) -> None:
    require(isinstance(payload, dict), "model verifier did not return a JSON object")
    require(payload.get("status") == "VERIFIED_READ_ONLY", "model verifier did not succeed")
    require(payload.get("cache") == plan["shared_model_source_cache"],
            "shared model source cache identity differs")
    require(payload.get("revision") == plan["model_revision"], "model revision differs")
    require(payload.get("metadata_sha256") == MODEL_METADATA_SHA256,
            "model metadata hashes differ from the frozen revision")
    shards = payload.get("shards")
    require(isinstance(shards, dict) and set(shards) == set(MODEL_SHARDS),
            "model shard receipt has missing or extra files")
    for name, (size, digest) in MODEL_SHARDS.items():
        require(shards[name] == {"bytes": size, "sha256": digest},
                f"model shard receipt differs: {name}")


def verify_model_under_lock(plan: dict[str, Any], session_dir: Path,
                            deadline: float) -> dict[str, Any]:
    """Hash model bytes while fd 9 holds the same GPU lock as the cell."""
    status_path = session_dir / "model-identity-status.json"
    status: dict[str, Any] = {"status": "STARTED", "started_unix_s": time.time(),
                              "verifier_path": MODEL_VERIFIER_PATH,
                              "verifier_sha256": MODEL_VERIFIER_SHA256}
    write_json(status_path, status)
    try:
        verifier = Path(MODEL_VERIFIER_PATH)
        require(verifier.is_file() and not verifier.is_symlink() and
                sha256_file(verifier) == MODEL_VERIFIER_SHA256,
                "model verifier changed after plan validation")
        remaining = remaining_seconds(deadline)
        require(remaining > MODEL_VERIFY_TIMEOUT_S + 5,
                "total wall budget exhausted before model verification")
        env = os.environ.copy()
        for key in ("PYTHONPATH", "PYTHONHOME", "PYTHONINSPECT",
                    "PYTHONUSERBASE", "VIRTUAL_ENV"):
            env.pop(key, None)
        env.update(PYTHONOPTIMIZE="0", PYTHONDONTWRITEBYTECODE="1",
                   PYTHONNOUSERSITE="1", HF_HUB_OFFLINE="1",
                   CUDA_VISIBLE_DEVICES=plan["authorized_gpu_uuid"])
        result = subprocess.run(
            [plan["python"], str(verifier)], env=env, text=True,
            capture_output=True, timeout=MODEL_VERIFY_TIMEOUT_S, check=False,
        )
        (session_dir / "model-verifier.stdout").write_text(result.stdout)
        (session_dir / "model-verifier.stderr").write_text(result.stderr)
        require(result.returncode == 0,
                f"model verifier exited {result.returncode}; no cell may start")
        payload = json.loads(result.stdout)
        validate_model_receipt(payload, plan)
        write_json(session_dir / "model-identity-receipt.json", payload)
        status.update(status="VERIFIED_READ_ONLY", exit_code=0,
                      finished_unix_s=time.time())
        write_json(status_path, status)
        return payload
    except BaseException as error:
        if isinstance(error, subprocess.TimeoutExpired):
            for name, output in (("stdout", error.stdout), ("stderr", error.stderr)):
                if isinstance(output, bytes):
                    output = output.decode("utf-8", errors="replace")
                (session_dir / f"model-verifier.{name}").write_text(output or "")
        status.update(status="INCOMPLETE", error=f"{type(error).__name__}: {error}",
                      finished_unix_s=time.time())
        write_json(status_path, status)
        raise


def private_model_env(plan: dict[str, Any], session_dir: Path) -> dict[str, str]:
    cache = Path(plan["hf_cache_dir"])
    runtime_cache = session_dir / "runtime-cache"
    require(runtime_cache.is_dir() and not runtime_cache.is_symlink(),
            "session-local runtime cache is missing")
    env = os.environ.copy()
    for key in ("PYTHONPATH", "PYTHONHOME", "PYTHONINSPECT", "PYTHONUSERBASE",
                "VIRTUAL_ENV", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE",
                "TRANSFORMERS_CACHE", "HF_ASSETS_CACHE", "HF_MODULES_CACHE",
                "HF_XET_CACHE"):
        env.pop(key, None)
    env.update({
        "HF_HOME": str(cache),
        "HF_HUB_CACHE": str(cache / "hub"),
        "HUGGINGFACE_HUB_CACHE": str(cache / "hub"),
        "HF_ASSETS_CACHE": str(runtime_cache / "assets"),
        "HF_MODULES_CACHE": str(runtime_cache / "transformers_modules"),
        "HF_XET_CACHE": str(runtime_cache / "xet"),
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "XDG_CACHE_HOME": str(runtime_cache / "xdg"),
        "TMPDIR": str(runtime_cache / "tmp"),
        "TORCHINDUCTOR_CACHE_DIR": str(runtime_cache / "torchinductor"),
        "TRITON_CACHE_DIR": str(runtime_cache / "triton"),
        "VLLM_CACHE_ROOT": str(runtime_cache / "vllm"),
        "TORCH_HOME": str(runtime_cache / "torch"),
        "CUDA_CACHE_PATH": str(runtime_cache / "cuda" / "ComputeCache"),
        "PYTHONOPTIMIZE": "0", "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "CUDA_VISIBLE_DEVICES": plan["authorized_gpu_uuid"],
    })
    return env


def prepare_runtime_cache(session_dir: Path) -> None:
    runtime_cache = session_dir / "runtime-cache"
    runtime_cache.mkdir(mode=0o700)
    for name in ("assets", "transformers_modules", "xet", "xdg", "tmp",
                 "torchinductor", "triton", "vllm", "torch", "cuda/ComputeCache"):
        (runtime_cache / name).mkdir(parents=True, mode=0o700)


def bounded_sha256_file(path: Path, stop: float) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            require(time.monotonic() < stop, "private model verification timed out")
            chunk = stream.read(8 * 1024 * 1024)
            if not chunk:
                return digest.hexdigest()
            digest.update(chunk)


def regular_tree_bytes(root: Path) -> int:
    total = 0
    for directory, dirs, files in os.walk(root, followlinks=False):
        for name in dirs + files:
            entry = Path(directory) / name
            mode = entry.lstat().st_mode
            require(stat.S_ISDIR(mode) or stat.S_ISREG(mode) or stat.S_ISLNK(mode),
                    f"model repo contains a special file: {entry}")
            if stat.S_ISREG(mode):
                total += entry.stat().st_size
    return total


def filesystem_device(path: Path) -> int:
    return path.stat().st_dev


def verify_existing_private_model_under_lock(plan: dict[str, Any], session_dir: Path,
                                             deadline: float) -> dict[str, Any]:
    """Rehash A's existing exact-revision model without creating or deleting it."""
    status_path = session_dir / "model-private-view-status.json"
    status: dict[str, Any] = {"status": "STARTED", "started_unix_s": time.time(),
                              "source_cache": plan["shared_model_source_cache"],
                              "private_cache": plan["hf_cache_dir"]}
    write_json(status_path, status)
    source_cache = Path(plan["shared_model_source_cache"])
    cache = Path(plan["hf_cache_dir"])
    source_repo = source_cache / "hub" / MODEL_REPO_NAME
    repo = cache / "hub" / MODEL_REPO_NAME
    source_snapshot = source_repo / "snapshots" / plan["model_revision"]
    snapshot = repo / "snapshots" / plan["model_revision"]
    try:
        require(remaining_seconds(deadline) > MODEL_VIEW_TIMEOUT_S +
                MODEL_RESOLVE_TIMEOUT_S + 5,
                "total wall budget exhausted before private model rehash")
        stop = min(deadline - MODEL_RESOLVE_TIMEOUT_S - 5,
                   time.monotonic() + MODEL_VIEW_TIMEOUT_S)
        for name, directory in (("C cache", source_cache),
                                ("C hub", source_cache / "hub"),
                                ("C repo", source_repo),
                                ("C blobs", source_repo / "blobs"),
                                ("C snapshot", source_snapshot),
                                ("A cache", cache), ("A hub", cache / "hub"),
                                ("A repo", repo), ("A blobs", repo / "blobs"),
                                ("A snapshot", snapshot)):
            require(directory.is_dir() and not directory.is_symlink(),
                    f"{name} missing or symlinked")
        require(not is_within(cache.resolve(strict=True), source_cache.resolve(strict=True)) and
                not is_within(source_cache.resolve(strict=True), cache.resolve(strict=True)),
                "A model cache overlaps C source")
        require(filesystem_device(repo) != filesystem_device(source_repo),
                "A private model repo must be on a separate filesystem")
        source_bytes = regular_tree_bytes(source_repo)
        private_bytes = regular_tree_bytes(repo)
        require(source_bytes >= sum(size for size, _ in MODEL_SHARDS.values()) and
                private_bytes >= sum(size for size, _ in MODEL_SHARDS.values()),
                "C or A model repo has fewer bytes than its frozen shards")
        for directory, dirs, files in os.walk(repo, followlinks=False):
            for name in dirs + files:
                entry = Path(directory) / name
                mode = entry.lstat().st_mode
                require(stat.S_ISDIR(mode) or stat.S_ISREG(mode) or stat.S_ISLNK(mode),
                        f"A private model contains a special file: {entry}")
                if stat.S_ISLNK(mode):
                    require(is_within(entry.resolve(strict=True), repo.resolve(strict=True)),
                            f"A private model link escapes its own repo: {entry}")
        metadata: dict[str, str] = {}
        shards: dict[str, dict[str, Any]] = {}
        mapping: dict[str, str] = {}
        for name in (*MODEL_METADATA_SHA256, *MODEL_SHARDS):
            source_link = source_snapshot / name
            link = snapshot / name
            require(source_link.is_symlink() and link.is_symlink() and
                    os.readlink(link) == os.readlink(source_link),
                    f"A private snapshot link differs from C source: {name}")
            source_blob = source_link.resolve(strict=True)
            blob = link.resolve(strict=True)
            require(is_within(source_blob, (source_repo / "blobs").resolve(strict=True)) and
                    is_within(blob, (repo / "blobs").resolve(strict=True)),
                    f"snapshot link does not resolve into its own blobs: {name}")
            source_stat, private_stat = source_blob.stat(), blob.stat()
            require(stat.S_ISREG(source_stat.st_mode) and
                    stat.S_ISREG(private_stat.st_mode) and
                    (source_stat.st_dev, source_stat.st_ino) !=
                    (private_stat.st_dev, private_stat.st_ino),
                    f"A private blob is not an independent regular inode: {name}")
            require(private_stat.st_size == source_stat.st_size,
                    f"A private blob size differs from C source: {name}")
            digest = bounded_sha256_file(blob, stop)
            if name in MODEL_METADATA_SHA256:
                require(digest == MODEL_METADATA_SHA256[name],
                        f"A private model metadata SHA-256 differs: {name}")
                metadata[name] = digest
            else:
                size, expected = MODEL_SHARDS[name]
                require(private_stat.st_size == size and digest == expected,
                        f"A private model shard size/SHA-256 differs: {name}")
                shards[name] = {"bytes": size, "sha256": digest}
            mapping[name] = str(blob.relative_to(repo.resolve(strict=True)))
        require(time.monotonic() < stop, "private model rehash timed out")
        payload = {
            "status": "VERIFIED_PRIVATE_MODEL_VIEW",
            "access_mode": "existing_read_only",
            "source_cache": str(source_cache), "private_cache": str(cache),
            "revision": plan["model_revision"],
            "source_tree_bytes": source_bytes, "private_tree_bytes": private_bytes,
            "source_receipt": "model-identity-receipt.json",
            "metadata_sha256": metadata, "shards": shards,
            "private_blob_paths": mapping,
        }
        write_json(session_dir / "model-private-view-receipt.json", payload)
        status.update(status="VERIFIED_PRIVATE_MODEL_VIEW", finished_unix_s=time.time())
        write_json(status_path, status)
        return payload
    except BaseException as error:
        status.update(status="INCOMPLETE", error=f"{type(error).__name__}: {error}",
                      finished_unix_s=time.time())
        write_json(status_path, status)
        raise

def verify_private_offline_resolution(plan: dict[str, Any], session_dir: Path,
                                      deadline: float) -> dict[str, Any]:
    """Resolve config through the private HF_HOME with network disabled."""
    status_path = session_dir / "model-offline-resolution-status.json"
    status: dict[str, Any] = {"status": "STARTED", "started_unix_s": time.time()}
    write_json(status_path, status)
    try:
        require(remaining_seconds(deadline) > MODEL_RESOLVE_TIMEOUT_S + 5,
                "total wall budget exhausted before private offline resolution")
        script = r'''
import json, os, sys
from pathlib import Path
from huggingface_hub import hf_hub_download
from transformers import AutoConfig
model_id, revision = sys.argv[1:3]
path = Path(hf_hub_download(model_id, "config.json", revision=revision,
                            local_files_only=True))
config = AutoConfig.from_pretrained(model_id, revision=revision,
                                    local_files_only=True, trust_remote_code=False)
print(json.dumps({"status": "RESOLVED_OFFLINE", "path": str(path),
                  "resolved": str(path.resolve(strict=True)),
                  "model_type": config.model_type, "hf_home": os.environ["HF_HOME"]}))
'''
        result = subprocess.run(
            [plan["python"], "-I", "-c", script, MODEL_ID, plan["model_revision"]],
            env=private_model_env(plan, session_dir), text=True, capture_output=True,
            timeout=MODEL_RESOLVE_TIMEOUT_S, check=False,
        )
        (session_dir / "model-offline-resolution.stdout").write_text(result.stdout)
        (session_dir / "model-offline-resolution.stderr").write_text(result.stderr)
        require(result.returncode == 0,
                f"private offline model resolution exited {result.returncode}")
        payload = json.loads(result.stdout)
        cache = Path(plan["hf_cache_dir"])
        private_blobs = cache / "hub" / MODEL_REPO_NAME / "blobs"
        require(payload.get("status") == "RESOLVED_OFFLINE" and
                payload.get("hf_home") == str(cache) and
                payload.get("model_type") == "olmoe" and
                is_within(Path(payload["resolved"]), private_blobs.resolve()),
                "offline config did not resolve through A private model blobs")
        write_json(session_dir / "model-offline-resolution-receipt.json", payload)
        status.update(status="RESOLVED_OFFLINE", finished_unix_s=time.time())
        write_json(status_path, status)
        return payload
    except BaseException as error:
        if isinstance(error, subprocess.TimeoutExpired):
            for name, output in (("stdout", error.stdout), ("stderr", error.stderr)):
                if isinstance(output, bytes):
                    output = output.decode("utf-8", errors="replace")
                (session_dir / f"model-offline-resolution.{name}").write_text(output or "")
        status.update(status="INCOMPLETE", error=f"{type(error).__name__}: {error}",
                      finished_unix_s=time.time())
        write_json(status_path, status)
        raise


def run_group_child(argv: list[str], env: dict[str, str], log_path: Path,
                    timeout_s: float) -> tuple[int, bool]:
    with log_path.open("wb") as log:
        process = subprocess.Popen(
            argv, env=env, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True, pass_fds=(9,),
        )
        try:
            return process.wait(timeout=timeout_s), False
        except BaseException as error:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=5)
            if isinstance(error, subprocess.TimeoutExpired):
                return process.returncode if process.returncode is not None else 124, True
            raise


def tree_hashes(root: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for directory, dirs, files in os.walk(root, followlinks=False):
        for name in dirs + files:
            path = Path(directory) / name
            mode = path.lstat().st_mode
            require(stat.S_ISDIR(mode) or stat.S_ISREG(mode), "output contains nonregular file or symlink")
        for name in files:
            path = Path(directory) / name
            hashes[str(path.relative_to(root))] = sha256_file(path)
    return hashes


def archive_copy(source: Path, destination: Path) -> None:
    require(source.is_dir() and not source.is_symlink(), "output directory absent or symlink")
    require(not destination.exists(), "archive destination already exists")
    source_hashes = tree_hashes(source)
    shutil.copytree(source, destination, symlinks=True)
    require(tree_hashes(destination) == source_hashes, "archive readback hash mismatch")
    write_json(destination.parent / "output_sha256.json", source_hashes)


def stop_requested(signum: int, _frame: Any) -> None:
    raise InterruptedError(f"received signal {signum}")


def run_session(plan: dict[str, Any], plan_sha: str, plan_bytes: bytes) -> int:
    lock_path = Path(plan["lock_path"])
    require(lock_path.is_file() and not lock_path.is_symlink(), "shared lock file must already exist")
    lock_fd = os.open(lock_path, os.O_RDWR | os.O_NOFOLLOW)
    installed_fd9 = False
    try:
        require(stat.S_ISREG(os.fstat(lock_fd).st_mode), "shared lock is not regular")
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        os.dup2(lock_fd, 9, inheritable=True)
        installed_fd9 = True
        held_lock_identity = require_lock_identity(
            9, lock_path, plan["expected_lock_device_inode"])
        deadline = time.monotonic() + plan["approved_total_wall_seconds"]
        session_dir = absolute_path(plan["session_dir"], "session_dir")
        session_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
        (session_dir / "plan.json").write_bytes(plan_bytes)
        receipt: dict[str, Any] = {
            "plan_sha256": plan_sha,
            "authorization_reference": plan["authorization_reference"],
            "started_unix_s": time.time(),
            "approved_total_wall_seconds": plan["approved_total_wall_seconds"],
            "held_lock_device_inode": held_lock_identity,
            "cells": [],
            "status": "RUNNING",
        }
        write_json(session_dir / "receipt.json", receipt)
        try:
            require_authorized_gpu(plan, deadline)
            require(not gpu_process_rows(deadline), "GPU compute process already present at group start")
            receipt["model_identity"] = "VERIFYING_UNDER_GROUP_LOCK"
            write_json(session_dir / "receipt.json", receipt)
            verify_model_under_lock(plan, session_dir, deadline)
            receipt["model_identity"] = "REHASHING_EXISTING_PRIVATE_MODEL"
            write_json(session_dir / "receipt.json", receipt)
            verify_existing_private_model_under_lock(plan, session_dir, deadline)
            prepare_runtime_cache(session_dir)
            receipt["model_identity"] = "RESOLVING_PRIVATE_MODEL_OFFLINE"
            write_json(session_dir / "receipt.json", receipt)
            verify_private_offline_resolution(plan, session_dir, deadline)
            receipt["model_identity"] = {
                "status": "VERIFIED_PRIVATE_MODEL_VIEW",
                "shared_source_receipt": "model-identity-receipt.json",
                "private_view_receipt": "model-private-view-receipt.json",
                "offline_resolution_receipt": "model-offline-resolution-receipt.json",
                "verifier_sha256": MODEL_VERIFIER_SHA256,
            }
            write_json(session_dir / "receipt.json", receipt)
            for index, cell in enumerate(plan["cells"]):
                require_lock_identity(9, lock_path, plan["expected_lock_device_inode"])
                require_authorized_gpu(plan, deadline)
                require(not gpu_process_rows(deadline), f"GPU compute process present before cell {index}")
                remaining = remaining_seconds(deadline)
                cell_s = min(cell["max_wall_seconds"], int(remaining) - 45)
                require(cell_s == cell["max_wall_seconds"],
                        f"total wall budget insufficient for full cell {index}")
                cell_dir = session_dir / f"cell-{index:02d}-{cell['arm']}"
                cell_dir.mkdir()
                package = absolute_path(cell["package_dir"], "package_dir")
                output = absolute_path(cell["output_dir"], "output_dir")
                arm = cell["arm"]
                env = private_model_env(plan, session_dir)
                env.update({
                    "GPERF_AUTHORIZED_GPU_UUID": plan["authorized_gpu_uuid"],
                    "GPERF_PYTHON": plan["python"],
                    "GPERF_HF_CACHE_DIR": plan["hf_cache_dir"],
                    "GPERF_LOCK_PATH": str(lock_path),
                    "GPERF_MAX_WALL_SECONDS": str(cell_s),
                    "GPERF_APPROVED_HOST_BYTES": str(plan["approved_host_bytes"]),
                    "GPERF_CGROUP_MEMORY_MAX_FILE": plan["cgroup_memory_max_file"],
                    "GPERF_EXPECTED_MANIFEST_SHA256": PACKAGE_SHA256,
                })
                argv = [str(package / "pkg" / "run.sh"), arm, str(output)]
                cell_receipt: dict[str, Any] = {
                    "arm": arm, "argv": argv, "output_dir": str(output),
                    "started_unix_s": time.time(), "launch_status": "STARTING",
                    "archive_status": "NOT_STARTED", "gpu_process_state_after": "UNKNOWN",
                }
                receipt["cells"].append(cell_receipt)
                write_json(session_dir / "receipt.json", receipt)
                gpu_check_error: Exception | None = None
                rows: list[str] = []
                try:
                    exit_code, timed_out = run_group_child(
                        argv, env, cell_dir / "launch.log", cell_s + 2,
                    )
                    cell_receipt.update({
                        "finished_unix_s": time.time(), "exit_code": exit_code,
                        "timed_out": timed_out, "launch_status": "FINISHED",
                    })
                    write_json(session_dir / "receipt.json", receipt)
                    if output.is_dir():
                        archive_budget = remaining_seconds(deadline) - 20
                        require(archive_budget > 0, "total wall budget exhausted before archive")
                        archive_argv = [sys.executable, __file__, "--archive-copy",
                                        str(output), str(cell_dir / "archive")]
                        archive_exit, archive_timeout = run_group_child(
                            archive_argv, os.environ.copy(), cell_dir / "archive.log",
                            archive_budget,
                        )
                        cell_receipt["archive_status"] = (
                            "VERIFIED" if archive_exit == 0 and not archive_timeout else "FAILED"
                        )
                        write_json(session_dir / "receipt.json", receipt)
                        require(cell_receipt["archive_status"] == "VERIFIED",
                                f"cell {index} archive failed")
                    else:
                        cell_receipt["archive_status"] = "NO_OUTPUT"
                finally:
                    try:
                        rows = gpu_process_rows(deadline)
                        cell_receipt["gpu_process_state_after"] = "BUSY" if rows else "EMPTY"
                        cell_receipt["gpu_process_rows_after"] = rows
                    except Exception as error:
                        gpu_check_error = error
                        cell_receipt["gpu_process_state_after"] = f"UNKNOWN: {error}"
                    write_json(session_dir / "receipt.json", receipt)
                require(gpu_check_error is None, f"GPU process check failed after cell {index}")
                require(not rows, f"GPU compute process remains after cell {index}; stop before next cell")
                require(exit_code == 0 and not timed_out, f"cell {index} failed; no next controller")
                require(output.is_dir(), f"cell {index} returned success without output")
            receipt["status"] = "CELLS_COMPLETE"
            return 0
        except BaseException as error:
            receipt["status"] = "ABORTED"
            receipt["error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            receipt["finished_unix_s"] = time.time()
            receipt["elapsed_wall_s"] = receipt["finished_unix_s"] - receipt["started_unix_s"]
            write_json(session_dir / "receipt.json", receipt)
    finally:
        if installed_fd9 and lock_fd != 9:
            os.close(9)
        os.close(lock_fd)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", nargs="?", type=Path)
    parser.add_argument("--expected-plan-sha256")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--archive-copy", nargs=2, metavar=("SOURCE", "DESTINATION"))
    args = parser.parse_args()
    if args.archive_copy:
        archive_copy(Path(args.archive_copy[0]), Path(args.archive_copy[1]))
        return 0
    require(args.plan is not None, "plan path required")
    require(isinstance(args.expected_plan_sha256, str) and
            SHA_RE.fullmatch(args.expected_plan_sha256) is not None,
            "root-frozen --expected-plan-sha256 required")
    plan_bytes = args.plan.read_bytes()
    plan_sha = hashlib.sha256(plan_bytes).hexdigest()
    require(plan_sha == args.expected_plan_sha256, "plan SHA-256 differs from root-frozen value")
    plan = validate_plan(json.loads(plan_bytes))
    if args.validate_only:
        print(json.dumps({"status": "VALID_CPU_ONLY", "plan_sha256": plan_sha,
                          "cells": len(plan["cells"])}))
        return 0
    return run_session(plan, plan_sha, plan_bytes)


if __name__ == "__main__":
    try:
        signal.signal(signal.SIGTERM, stop_requested)
        raise SystemExit(main())
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f"ABORT_SERIAL_GROUP: {exc}", file=sys.stderr)
        raise SystemExit(75) from exc
