#!/usr/bin/env python3
"""Run frozen recovery cells serially under one lock and one wall budget.

This is an execution guard, not a policy controller. It accepts only the two
locally frozen candidate package identities below. The operator must first
verify the user's authorization, live machine, model revision and shared lock.
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


PACKAGE_SHA = {
    "ltr_r02_g64": (
        "candidate_ltr_r02_env",
        "f58340cd5228ea7339f0d4081176aab265583cb824a26026d19b8f7ae93cc8a7",
    ),
    "h1": (
        "candidate_h1",
        "04ad1217287491813be3c85839de00ff80ee2ec60011bdf762c58e52b3e3a6d4",
    ),
}
H1_CELLS = {
    ("native_full_native", "performance", "off"),
    ("eager", "performance", "off"),
    ("eager", "performance", "on"),
    ("eager", "diagnostic", "off"),
    ("eager", "diagnostic", "on"),
}
GPU_UUID_RE = re.compile(r"GPU-[0-9a-fA-F-]{36}\Z")
SHA_RE = re.compile(r"[0-9a-f]{64}\Z")
LOCK_ID_RE = re.compile(r"[1-9][0-9]*:[1-9][0-9]*\Z")


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
    for key in ("python", "hf_cache_dir", "cgroup_memory_max_file"):
        absolute_path(plan.get(key), key)
    cells = plan.get("cells")
    require(isinstance(cells, list) and 0 < len(cells) <= 12, "cells must contain 1..12 entries")
    if any(isinstance(cell, dict) and cell.get("kind") == "ltr_r02_g64" for cell in cells):
        require(len(cells) == 1, "G64 lifecycle qualification must be a separate one-cell session")
    seen_outputs: set[Path] = set()
    seen_cells: set[tuple[str, ...]] = set()
    packages: list[Path] = []
    for index, cell in enumerate(cells):
        require(isinstance(cell, dict), f"cell {index} must be an object")
        kind = cell.get("kind")
        require(kind in PACKAGE_SHA, f"cell {index} has an unknown package kind")
        package_text = cell.get("package_dir")
        package = absolute_path(package_text, f"cell {index} package_dir")
        require(not Path(package_text).is_symlink() and package.is_dir(),
                f"cell {index} package missing or symlink")
        expected_name, expected_sha = PACKAGE_SHA[kind]
        require(package.name == expected_name, f"cell {index} package name mismatch")
        require(sha256_file(package / "manifest.json") == expected_sha,
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
        if kind == "ltr_r02_g64":
            require(not any(k in cell for k in ("variant", "mode", "gate")),
                    "LTR qualification is one fixed G64 cell")
            identity = (kind,)
        else:
            triplet = (cell.get("variant"), cell.get("mode"), cell.get("gate"))
            require(triplet in H1_CELLS, f"cell {index} H1 mode not frozen")
            identity = (kind, *triplet)
        require(identity not in seen_cells, f"cell {index} repeats a controller cell")
        seen_cells.add(identity)
        seen_outputs.add(output)
        packages.append(package)
    for index, output in enumerate(seen_outputs):
        require(not any(is_within(output, p) for p in packages),
                f"output {index} overlaps another package")
    require(not is_within(lock_path, session_dir) and not any(is_within(lock_path, p) for p in packages),
            "lock path must live outside session and candidate packages")
    requested_cell_s = sum(cell["max_wall_seconds"] for cell in cells)
    require(total_s >= requested_cell_s + 45 * len(cells) + 20,
            "total wall budget cannot fund all full cells plus checks and archive")
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
            require(not gpu_process_rows(deadline), "GPU compute process already present at group start")
            for index, cell in enumerate(plan["cells"]):
                require_lock_identity(9, lock_path, plan["expected_lock_device_inode"])
                require(not gpu_process_rows(deadline), f"GPU compute process present before cell {index}")
                remaining = remaining_seconds(deadline)
                cell_s = min(cell["max_wall_seconds"], int(remaining) - 45)
                require(cell_s == cell["max_wall_seconds"],
                        f"total wall budget insufficient for full cell {index}")
                cell_dir = session_dir / f"cell-{index:02d}-{cell['kind']}"
                cell_dir.mkdir()
                package = absolute_path(cell["package_dir"], "package_dir")
                output = absolute_path(cell["output_dir"], "output_dir")
                kind = cell["kind"]
                prefix = "R02" if kind == "ltr_r02_g64" else "H1"
                env = os.environ.copy()
                for key in ("PYTHONPATH", "PYTHONHOME", "PYTHONINSPECT"):
                    env.pop(key, None)
                env.update({
                    f"{prefix}_AUTHORIZED_GPU_UUID": plan["authorized_gpu_uuid"],
                    f"{prefix}_PYTHON": plan["python"],
                    f"{prefix}_HF_CACHE_DIR": plan["hf_cache_dir"],
                    f"{prefix}_LOCK_PATH": str(lock_path),
                    f"{prefix}_MAX_WALL_SECONDS": str(cell_s),
                    f"{prefix}_APPROVED_HOST_BYTES": str(plan["approved_host_bytes"]),
                    f"{prefix}_CGROUP_MEMORY_MAX_FILE": plan["cgroup_memory_max_file"],
                    f"{prefix}_EXPECTED_MANIFEST_SHA256": PACKAGE_SHA[kind][1],
                })
                argv = [str(package / "pkg" / "run.sh")]
                if kind == "h1":
                    argv.extend([cell["variant"], cell["mode"], cell["gate"]])
                argv.append(str(output))
                cell_receipt: dict[str, Any] = {
                    "kind": kind, "argv": argv, "output_dir": str(output),
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
