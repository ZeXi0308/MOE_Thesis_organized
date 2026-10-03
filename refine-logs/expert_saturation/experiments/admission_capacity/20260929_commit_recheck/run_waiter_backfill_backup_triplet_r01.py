#!/usr/bin/env python3
"""Run the frozen ordinary-backfill control triplet on the authorized backup."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import signal
import sys
import time

import run_waiter_backfill_triplet_r01 as control
fresh = control.group


base = fresh.base
fresh.BASE = "/root/moe-a-waiter-backfill-backup-stage-r01-20261001"
fresh.SESSION = "/root/moe-a-waiter-backfill-backup-session-r01-20261001"
fresh.SOURCE_CACHE = Path("/root/moe-a-capacity-identical-session-r01-20261001/runtime-cache")
fresh.OUTPUTS = tuple(
    "/root/moe-a-waiter-backfill-backup-output-" + label + "-r01-20261001"
    for label in fresh.LABELS
)
SOURCE_ARMS = ("capacity_reference_first_on", "capacity_reference_second_on")
original_require_authorized_gpu = base.require_authorized_gpu
_start_gpu_checked = False


def prepare_runtime_cache(session_dir: Path) -> None:
    """Use the completed backup-host A/A cache; each cell gets a private copy."""
    fresh.original_prepare_cache(session_dir)
    base.require(session_dir == Path(fresh.SESSION), "unexpected backup triplet session")
    source = fresh.SOURCE_CACHE
    base.require(source.is_dir() and not source.is_symlink(),
                 "completed backup-host A/A runtime cache is missing")
    prior_receipt = json.loads((source.parent / "receipt.json").read_text())
    base.require(prior_receipt.get("status") == "CELLS_COMPLETE" and
                 tuple(c.get("arm") for c in prior_receipt.get("cells", [])) == SOURCE_ARMS,
                 "source runtime cache lacks completed A/A receipt")
    source_files = fresh.cache_inventory(source)
    base.require(bool(source_files), "completed A/A runtime cache is empty")
    source_bytes = sum(size for size, _ in source_files.values())
    base.require(shutil.disk_usage("/root").free >= 2 * 1024**3 + 3 * source_bytes,
                 "need 2 GiB free after three private warm-cache copies")
    for label in fresh.LABELS:
        destination = session_dir / f"runtime-cache-{label}"
        base.require(not destination.exists(), "private runtime cache already exists")
        shutil.copytree(source, destination, copy_function=shutil.copy2)
        base.require(fresh.cache_inventory(destination) == source_files,
                     f"private {label} runtime cache metadata differs from source")
    fresh._seed_files = source_files
    base.write_json(session_dir / "runtime-cache-seed-receipt.json", {
        "status": "THREE_PRIVATE_COPIES_PREPARED_UNDER_LOCK",
        "source": str(source), "source_receipt_arms": list(SOURCE_ARMS),
        "source_files": len(source_files), "source_bytes": source_bytes,
        "destinations": [str(session_dir / f"runtime-cache-{label}")
                         for label in fresh.LABELS],
        "copied_unix_s": time.time(),
        "identity_check": "relative path, byte size and preserved mtime_ns",
    })


def gpu_occupancy(plan: dict, deadline: float) -> dict:
    remaining = base.remaining_seconds(deadline)
    base.require(remaining > 5, "insufficient wall budget for start GPU occupancy query")
    result = base.subprocess.run(
        ["nvidia-smi", "--query-gpu=uuid,memory.used,utilization.gpu",
         "--format=csv,noheader,nounits"],
        text=True, capture_output=True, timeout=min(10.0, remaining - 2), check=False,
    )
    base.require(result.returncode == 0, "nvidia-smi occupancy query failed")
    rows = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    base.require(len(rows) == 1, "expected one physical GPU occupancy row")
    fields = [field.strip() for field in rows[0].split(",")]
    base.require(len(fields) == 3 and fields[0] == plan["authorized_gpu_uuid"],
                 "occupancy query GPU UUID differs from approved GPU")
    try:
        used_mib, utilization_percent = int(fields[1]), int(fields[2])
    except ValueError as error:
        raise ValueError("GPU occupancy counters unavailable") from error
    base.require(used_mib >= 0 and 0 <= utilization_percent <= 100,
                 "GPU occupancy counters out of range")
    return {"unix_s": time.time(), "uuid": fields[0],
            "memory_used_mib": used_mib,
            "utilization_gpu_percent": utilization_percent,
            "idle": used_mib <= 256 and utilization_percent == 0}


def require_authorized_gpu(plan: dict, deadline: float) -> None:
    """Do one start-only occupancy check under the existing shared lock."""
    global _start_gpu_checked
    original_require_authorized_gpu(plan, deadline)
    if _start_gpu_checked:
        return
    session_dir = Path(plan["session_dir"])
    readings = [gpu_occupancy(plan, deadline)]
    if not readings[-1]["idle"]:
        base.require(base.remaining_seconds(deadline) > 8,
                     "insufficient wall budget for occupancy recheck")
        time.sleep(2)
        readings.append(gpu_occupancy(plan, deadline))
    idle = readings[-1]["idle"]
    base.write_json(session_dir / "gpu-start-occupancy.json", {
        "status": "IDLE_BEFORE_INITIALIZATION" if idle else "BUSY_ABORTED_BEFORE_INITIALIZATION",
        "scope": "group start only, under the shared GPU lock",
        "threshold": "utilization.gpu == 0 and memory.used <= 256 MiB",
        "readings": readings,
    })
    base.require(idle, "backup GPU busy after 2-second start recheck")
    _start_gpu_checked = True


base.prepare_runtime_cache = prepare_runtime_cache
base.require_authorized_gpu = require_authorized_gpu

if __name__ == "__main__":
    signal.signal(signal.SIGTERM, base.stop_requested)
    try:
        raise SystemExit(base.main())
    except (OSError, ValueError, base.subprocess.SubprocessError) as error:
        print("ABORT_WAITER_BACKFILL_BACKUP_TRIPLET:", error, file=sys.stderr)
        raise SystemExit(75)
