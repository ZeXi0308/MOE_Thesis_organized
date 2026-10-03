#!/usr/bin/env python3
"""Run one Q1 / admission-aware Q10 / plain Q10 native triplet."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import signal
import stat
import sys
import time

import serial_group_h1_guard_qual_r02 as base


PACKAGE_SHA256 = "b77bdf94bcea2b98fbb7e191a379ba199ad8fd782f50b7b1b26af521ab757503"
ARMS = ("protection_q1", "protection_q10_yield", "protection_q10_plain")
LABELS = ("first", "second", "third")
GATES = ("q1", "q10_yield", "q10")
BASE = "/root/moe-a-protection-yield-stage-r01-20261001"
SESSION = "/root/moe-a-protection-yield-session-r01-20261001"
SOURCE_CACHE = Path("/root/moe-a-protection-scope-diag-session-r01-20261001/runtime-cache")
OUTPUTS = tuple("/root/moe-a-protection-yield-output-" + label + "-r01-20261001"
                for label in LABELS)
CACHE_ENV_KEYS = (
    "HF_ASSETS_CACHE", "HF_MODULES_CACHE", "HF_XET_CACHE", "XDG_CACHE_HOME",
    "TMPDIR", "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR", "VLLM_CACHE_ROOT",
    "TORCH_HOME", "CUDA_CACHE_PATH",
)

original_validate = base.validate_plan
original_run = base.run_session
original_prepare_cache = base.prepare_runtime_cache
original_private_env = base.private_model_env
original_run_child = base.run_group_child
base.PACKAGE_NAME = "candidate_protection_yield_r01"
base.PACKAGE_SHA256 = PACKAGE_SHA256
_seed_files: dict[str, tuple[int, int]] = {}


def validate(plan):
    base.require(plan.get("session_dir") == SESSION and
                 plan.get("approved_total_wall_seconds") == 4200 and
                 plan.get("warm_runtime_cache_source") == str(SOURCE_CACHE),
                 "protection yield triplet session, cache source, or wall bound changed")
    cells = plan.get("cells", [])
    base.require(tuple(c.get("arm") for c in cells) == ARMS, "protection yield triplet cell order changed")
    for index, cell in enumerate(cells):
        label = LABELS[index]
        base.require(cell.get("package_dir") == BASE + "/" + label + "/" + base.PACKAGE_NAME and
                     cell.get("output_dir") == OUTPUTS[index] and
                     cell.get("max_wall_seconds") == 900,
                     "protection yield triplet cell changed")
        base.FROZEN_ARMS = (ARMS[index],)
        original_validate(dict(plan, cells=[cell]))
    base.FROZEN_ARMS = ARMS
    base.require(plan["approved_total_wall_seconds"] >=
                 sum(c["max_wall_seconds"] for c in cells) + 360 + 480 + 60 + 135 + 20,
                 "insufficient total triplet time")
    return plan


def argv(package: Path, output: Path) -> list[str]:
    base.require(str(output) in OUTPUTS, "unknown protection yield triplet output")
    index = OUTPUTS.index(str(output))
    base.require(str(package) == BASE + "/" + LABELS[index] +
                 "/" + base.PACKAGE_NAME, "wrong per-cell package copy")
    return [str(package / "pkg/run.sh"), "eager", "performance", GATES[index], str(output)]


def cache_inventory(root: Path) -> dict[str, tuple[int, int]]:
    base.require(root.is_dir() and not root.is_symlink(), f"runtime cache missing: {root}")
    found = {}
    for directory, dirs, files in os.walk(root, followlinks=False):
        for name in dirs:
            item = Path(directory) / name
            base.require(stat.S_ISDIR(item.lstat().st_mode), f"runtime cache has non-directory: {item}")
        for name in files:
            item = Path(directory) / name
            info = item.lstat()
            base.require(stat.S_ISREG(info.st_mode), f"runtime cache has nonregular file: {item}")
            found[str(item.relative_to(root))] = (info.st_size, info.st_mtime_ns)
    return found


def prepare_runtime_cache(session_dir: Path) -> None:
    # The base default is used for offline model resolution. All measured
    # cells receive private copies of the same completed prior-run cache.
    original_prepare_cache(session_dir)
    base.require(session_dir == Path(SESSION), "unexpected protection yield triplet session")
    base.require(SOURCE_CACHE.is_dir() and not SOURCE_CACHE.is_symlink(),
                 "completed diagnostic runtime cache is missing")
    prior_receipt = json.loads((SOURCE_CACHE.parent / "receipt.json").read_text())
    base.require(prior_receipt.get("status") == "CELLS_COMPLETE" and
                 tuple(c.get("arm") for c in prior_receipt.get("cells", [])) == ("protection_scope_q10_diagnostic",),
                 "source runtime cache lacks complete diagnostic receipt")
    global _seed_files
    source_files = cache_inventory(SOURCE_CACHE)
    base.require(bool(source_files), "completed diagnostic runtime cache is empty")
    source_bytes = sum(size for size, _ in source_files.values())
    base.require(shutil.disk_usage("/root").free >= 2 * 1024**3 + 3 * source_bytes,
                 "need 2 GiB free after three private warm-cache copies")
    for label in LABELS:
        destination = session_dir / f"runtime-cache-{label}"
        base.require(not destination.exists(), "private runtime cache already exists")
        shutil.copytree(SOURCE_CACHE, destination, copy_function=shutil.copy2)
        base.require(cache_inventory(destination) == source_files,
                     f"private {label} runtime cache metadata differs from source")
    _seed_files = source_files
    base.write_json(session_dir / "runtime-cache-seed-receipt.json", {
        "status": "THREE_PRIVATE_COPIES_PREPARED_UNDER_LOCK",
        "source": str(SOURCE_CACHE), "source_files": len(source_files),
        "source_bytes": source_bytes,
        "destinations": [str(session_dir / f"runtime-cache-{label}")
                         for label in LABELS],
        "copied_unix_s": time.time(), "identity_check": "relative path, byte size and preserved mtime_ns",
    })


def private_model_env(plan, session_dir: Path):
    env = original_private_env(plan, session_dir)
    found = [(session_dir / f"cell-{i:02d}-{arm}", LABELS[i])
             for i, arm in enumerate(ARMS) if (session_dir / f"cell-{i:02d}-{arm}").exists()]
    if not found:
        return env  # Offline resolution precedes cell creation.
    cell_dir, label = found[-1]
    base.require(cell_dir.is_dir() and not cell_dir.is_symlink(),
                 "protection yield triplet cell directory is invalid")
    default = str(session_dir / "runtime-cache")
    selected = session_dir / f"runtime-cache-{label}"
    base.require(selected.is_dir() and not selected.is_symlink(),
                 "selected private runtime cache is missing")
    for key in CACHE_ENV_KEYS:
        value = env[key]
        base.require(value.startswith(default + os.sep), f"{key} no longer uses default cache")
        env[key] = str(selected) + value[len(default):]
    base.write_json(cell_dir / "runtime-cache-selection.json", {
        "status": "PRIVATE_CELL_CACHE_SELECTED", "cell": label,
        "cache": str(selected), "model_cache": env["HF_HOME"],
    })
    return env


def cache_write_summary(cache: Path, seed: dict[str, tuple[int, int]], timing_path: Path) -> dict:
    current = cache_inventory(cache)
    changed = {name: (size, mtime) for name, (size, mtime) in current.items()
               if seed.get(name) != (size, mtime)}
    summary = {"status": "RECORDED", "cache": str(cache),
               "seed_files": len(seed), "current_files": len(current),
               "new_or_modified_files": len(changed),
               "new_or_modified_bytes": sum(size for size, _ in changed.values()),
               "basis": "Final file size/mtime compared with the same pre-cell seed; no cache content hashing. Files rewritten after measurement may have a later final mtime and escape measurement classification."}
    if not timing_path.is_file():
        summary.update(status="TIMING_UNAVAILABLE", measurement_written_files=None)
        return summary
    timing = json.loads(timing_path.read_text())
    offset = timing["process_start_unix_s"] - timing["process_start_perf_s"]
    stages = {
        "initialization": (timing["engine_init_start_perf_s"], timing["engine_init_end_perf_s"]),
        "warmup": (timing["warmup_start_perf_s"], timing["warmup_end_perf_s"]),
        "measurement": (timing["measurement_start_perf_s"], timing["measurement_return_perf_s"]),
    }
    stage_counts = {}
    measurement_files = []
    for stage, (start, stop) in stages.items():
        matched = [(name, size, mtime) for name, (size, mtime) in changed.items()
                   if start <= mtime / 1e9 - offset <= stop]
        stage_counts[stage] = {"files": len(matched),
                               "bytes": sum(size for _, size, _ in matched)}
        if stage == "measurement":
            measurement_files = [{"relative_path": name, "bytes": size,
                                  "mtime_unix_s": mtime / 1e9,
                                  "existed_in_seed": name in seed}
                                 for name, size, mtime in sorted(matched)]
    summary.update(timing_path=str(timing_path),
                   clock_alignment="file mtime Unix clock minus process-start Unix/perf offset",
                   stage_new_or_modified=stage_counts,
                   measurement_written_files=measurement_files)
    return summary


def run_group_child(argv, env, log_path, timeout_s):
    result = original_run_child(argv, env, log_path, timeout_s)
    if log_path.name != "launch.log" or len(argv) != 5 or (argv[1:3] != ["eager", "performance"] or argv[3] not in GATES):
        return result
    if argv[-1] not in OUTPUTS:
        return result
    index = OUTPUTS.index(argv[-1])
    label = LABELS[index]
    expected_launcher = BASE + "/" + label + "/" + base.PACKAGE_NAME + "/pkg/run.sh"
    if argv[0] != expected_launcher:
        return result
    cell_dir = log_path.parent
    output = Path(argv[-1])
    try:
        summary = cache_write_summary(
            Path(SESSION) / f"runtime-cache-{label}", _seed_files,
            output / "timing.json")
    except Exception as error:
        summary = {"status": "COLLECTION_ERROR", "error": f"{type(error).__name__}: {error}"}
    try:
        base.write_json(cell_dir / "runtime-cache-write-summary.json", summary)
    except OSError:
        # Auxiliary cache timing must not suppress the original launch result
        # or prevent the base controller from archiving its raw output.
        pass
    return result


def run(plan, digest, plan_bytes):
    base.require(shutil.disk_usage("/root").free >= 2 * 1024**3,
                 "need 2 GiB system free before starting protection yield triplet")
    return original_run(plan, digest, plan_bytes)


base.validate_plan = validate
base.qualification_argv = argv
base.prepare_runtime_cache = prepare_runtime_cache
base.private_model_env = private_model_env
base.run_group_child = run_group_child
base.run_session = run
if __name__ == "__main__":
    signal.signal(signal.SIGTERM, base.stop_requested)
    try:
        raise SystemExit(base.main())
    except (OSError, ValueError, base.subprocess.SubprocessError) as exc:
        print("ABORT_PROTECTION_YIELD_TRIPLET:", exc, file=sys.stderr)
        raise SystemExit(75)
