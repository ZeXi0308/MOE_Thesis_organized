#!/usr/bin/env python3
"""CPU-only prerequisite gate for the three unrun G64 LTR calibration cells.

This gate does not lock, launch, or initialize a GPU. The frozen serial group
controller remains the only GPU executor, after this gate and human review.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
from typing import Any


PILOT_PLAN_SHA = "2f192a20309f5c306c7613f67dc73a034731fd72c7a9a8ceb0116642bf62eb37"
REMAINING_PLAN_SHA = "01d6164d1561968de356ef65fec1143967dd23b13f2e4940e2da8076bd9e1b93"
CONTROLLER_SHA = "b899cad3c36789ed7eaffdeac052e93bd20d0b37434a9001c0563e8587115073"
PACKAGE_MANIFEST_SHA = "2755945122e3c346ca7e7e5ceda0d4668a8e1d3c90ba5c8d9d7b6ab3e89e0eb9"
PILOT_ARMS = ("native_full_native", "eager", "ltr_t30_q10")
REMAINING_ARMS = ("ltr_t30_q1", "ltr_t200_q1", "ltr_t200_q10")
REMOTE_PILOT_SESSION = Path("/root/autodl-tmp/moe-a-g64-perf-session-r01-20260930")
REMOTE_REMAINING_SESSION = Path("/root/autodl-tmp/moe-a-g64-perf-remaining-session-r02-20260930")
SHA_RE = re.compile(r"[0-9a-f]{64}\Z")
SAME_MACHINE_FIELDS = (
    "authorized_gpu_uuid", "approved_host_bytes", "lock_path",
    "expected_lock_device_inode", "python", "hf_cache_dir",
    "shared_model_source_cache", "model_revision", "model_verifier_path",
    "model_verifier_sha256", "cgroup_memory_max_file",
)


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    require(path.is_file() and not path.is_symlink(), f"missing/symlinked file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    sha256_file(path)
    return json.loads(path.read_text(encoding="utf-8"))


def archive_hashes(root: Path) -> dict[str, str]:
    require(root.is_dir() and not root.is_symlink(), f"archive missing/symlinked: {root}")
    hashes: dict[str, str] = {}
    for directory, dirs, files in os.walk(root, followlinks=False):
        for name in dirs + files:
            path = Path(directory) / name
            mode = path.lstat().st_mode
            require(stat.S_ISDIR(mode) or stat.S_ISREG(mode),
                    f"archive contains symlink or special file: {path}")
        for name in files:
            path = Path(directory) / name
            hashes[str(path.relative_to(root))] = sha256_file(path)
    return hashes


def validate_pilot(session: Path, audit_path: Path, audit_sha: str) -> tuple[dict, dict]:
    require(session.is_dir() and not session.is_symlink(), "pilot session missing/symlinked")
    require(SHA_RE.fullmatch(audit_sha) is not None and sha256_file(audit_path) == audit_sha,
            "pilot audit differs from externally frozen SHA-256")
    require(sha256_file(session / "plan.json") == PILOT_PLAN_SHA,
            "pilot plan bytes differ from frozen identity")
    pilot_plan = read_json(session / "plan.json")
    receipt_path = session / "receipt.json"
    receipt = read_json(receipt_path)
    audit = read_json(audit_path)
    require(audit.get("status") == "PILOT_THREE_ARM_COMPLETE_PARTIAL_TQ_GRID" and
            audit.get("calibration_status") == "NOT_SELECTABLE_PARTIAL_GRID" and
            audit.get("plan_sha256") == PILOT_PLAN_SHA and
            audit.get("receipt_sha256") == sha256_file(receipt_path) and
            audit.get("package_manifest_sha256") == PACKAGE_MANIFEST_SHA,
            "pilot audit is incomplete or refers to different receipts/package")
    require(receipt.get("status") == "CELLS_COMPLETE" and
            receipt.get("plan_sha256") == PILOT_PLAN_SHA and
            receipt.get("held_lock_device_inode") == pilot_plan.get("expected_lock_device_inode"),
            "pilot group did not complete under its frozen lock")
    plan_cells = pilot_plan.get("cells")
    receipt_cells = receipt.get("cells")
    require(isinstance(plan_cells, list) and isinstance(receipt_cells, list) and
            len(plan_cells) == len(receipt_cells) == 3 and
            tuple(c.get("arm") for c in plan_cells) == PILOT_ARMS and
            tuple(c.get("arm") for c in receipt_cells) == PILOT_ARMS,
            "pilot arm identities/order differ")
    metrics = audit.get("metrics", {})
    cell_evidence = audit.get("cells", {})
    require(isinstance(metrics, dict) and set(metrics) == set(PILOT_ARMS) and
            isinstance(cell_evidence, dict) and set(cell_evidence) == set(PILOT_ARMS),
            "pilot audit omitted an arm")
    for index, arm in enumerate(PILOT_ARMS):
        planned, observed = plan_cells[index], receipt_cells[index]
        require(planned.get("package_dir", "").endswith("/candidate_g64_perf_r01") and
                observed.get("output_dir") == planned.get("output_dir") and
                observed.get("argv") == [planned["package_dir"] + "/pkg/run.sh",
                                         arm, planned["output_dir"]] and
                observed.get("launch_status") == "FINISHED" and
                observed.get("exit_code") == 0 and observed.get("timed_out") is False and
                observed.get("archive_status") == "VERIFIED" and
                observed.get("gpu_process_state_after") == "EMPTY" and
                observed.get("gpu_process_rows_after") == [],
                f"pilot {arm} did not finish cleanly")
        summary = metrics[arm]
        require(summary.get("status") == "COMPLETE" and
                summary.get("expected_requests") == summary.get("completed") == 64 and
                summary.get("failed") == summary.get("unfinished") == 0,
                f"pilot {arm} is not 64/64 complete")
        cell_dir = session / f"cell-{index:02d}-{arm}"
        map_path = cell_dir / "output_sha256.json"
        expected = read_json(map_path)
        require(isinstance(expected, dict) and {"raw.json", "config.json", "status.json"} <= set(expected)
                and all(isinstance(value, str) and SHA_RE.fullmatch(value) for value in expected.values()),
                f"pilot {arm} archive map incomplete")
        require(archive_hashes(cell_dir / "archive") == expected,
                f"pilot {arm} archive readback differs")
        require(cell_evidence[arm].get("archive_sha256_map_sha256") == sha256_file(map_path) and
                cell_evidence[arm].get("raw_sha256") == expected["raw.json"],
                f"pilot {arm} audit differs from archived bytes")
    return pilot_plan, audit


def validate_remaining_plan(plan: dict, pilot_plan: dict) -> None:
    require(plan.get("schema_version") == 1 and pilot_plan.get("schema_version") == 1,
            "plan schema differs")
    require(all(plan.get(key) == pilot_plan.get(key) for key in SAME_MACHINE_FIELDS),
            "remaining grid changes GPU, lock, model, runtime, or host budget")
    require(plan.get("approved_total_wall_seconds") == 4200 and
            plan.get("session_dir") == str(REMOTE_REMAINING_SESSION) and
            plan.get("session_dir") != pilot_plan.get("session_dir"),
            "remaining group time or session identity differs")
    pilot_cells, cells = pilot_plan["cells"], plan.get("cells")
    require(isinstance(cells, list) and len(cells) == 3 and
            tuple(c.get("arm") for c in cells) == REMAINING_ARMS,
            "remaining grid must contain exactly the three unrun LTR points")
    pilot_package = {c["package_dir"] for c in pilot_cells}
    require(len(pilot_package) == 1 and
            all(c.get("package_dir") in pilot_package for c in cells),
            "remaining grid does not reuse exact pilot package bytes")
    old_paths = {pilot_plan["session_dir"], *(c["output_dir"] for c in pilot_cells)}
    new_paths = [plan["session_dir"], *(c.get("output_dir") for c in cells)]
    require(len(new_paths) == len(set(new_paths)) and not old_paths.intersection(new_paths),
            "remaining grid reuses a consumed output/session path")
    for cell in cells:
        arm = cell["arm"]
        require(cell.get("max_wall_seconds") == 900 and
                cell.get("output_dir") ==
                f"/root/autodl-tmp/moe-a-g64-perf-output-{arm}-r02-20260930",
                f"{arm}: wall limit or new output identity differs")


def validate_remote_state(plan: dict, pilot_session: Path, package: Path) -> None:
    require(pilot_session == REMOTE_PILOT_SESSION,
            "remote pilot session path differs from frozen identity")
    for path in (Path(plan["session_dir"]), *(Path(c["output_dir"]) for c in plan["cells"])):
        require(not os.path.lexists(path), f"new group identity already consumed: {path}")
    require(package == Path(plan["cells"][0]["package_dir"]),
            "remote package path differs from plan")
    for arm in PILOT_ARMS:
        require((package / f"launch-once-{arm}").is_dir(),
                f"pilot launch marker missing: {arm}")
    for arm in REMAINING_ARMS:
        require(not os.path.lexists(package / f"launch-once-{arm}"),
                f"remaining arm identity already consumed: {arm}")
    lock = Path(plan["lock_path"])
    require(lock.is_file() and not lock.is_symlink() and
            f"{lock.stat().st_dev}:{lock.stat().st_ino}" == plan["expected_lock_device_inode"],
            "shared lock inode differs")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-session", type=Path, required=True)
    parser.add_argument("--pilot-audit", type=Path, required=True)
    parser.add_argument("--expected-pilot-audit-sha256", required=True)
    parser.add_argument("--remaining-plan", type=Path, required=True)
    parser.add_argument("--expected-remaining-plan-sha256", required=True)
    parser.add_argument("--controller", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--recompute-pilot-audit", action="store_true")
    parser.add_argument("--remote-state-check", action="store_true")
    args = parser.parse_args()
    require(args.expected_remaining_plan_sha256 == REMAINING_PLAN_SHA and
            sha256_file(args.remaining_plan) == REMAINING_PLAN_SHA,
            "remaining plan differs from frozen SHA-256")
    require(sha256_file(args.controller) == CONTROLLER_SHA,
            "serial group controller bytes differ")
    require(sha256_file(args.package / "manifest.json") == PACKAGE_MANIFEST_SHA,
            "candidate package manifest differs")
    result = subprocess.run([sys.executable, "-B", str(args.package / "verify_package.py")],
                            capture_output=True, text=True, check=False)
    require(result.returncode == 0, f"candidate package 28-file check failed: {result.stderr}")
    pilot_plan, audit = validate_pilot(args.pilot_session, args.pilot_audit,
                                       args.expected_pilot_audit_sha256)
    plan = read_json(args.remaining_plan)
    validate_remaining_plan(plan, pilot_plan)
    if args.recompute_pilot_audit:
        import audit_g64_perf_three_arm
        require(audit_g64_perf_three_arm.audit(args.pilot_session, PILOT_PLAN_SHA) == audit,
                "pilot full audit does not reproduce from archived session")
    if args.remote_state_check:
        validate_remote_state(plan, args.pilot_session, args.package)
    print(json.dumps({"status": "READY_CPU_ONLY", "remaining_plan_sha256": REMAINING_PLAN_SHA,
                      "pilot_audit_sha256": args.expected_pilot_audit_sha256,
                      "pilot_complete_cells": len(PILOT_ARMS),
                      "remaining_unrun_cells": list(REMAINING_ARMS),
                      "remote_state_checked": args.remote_state_check}, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"ABORT_G64_REMAINING_GRID: {error}", file=sys.stderr)
        raise SystemExit(75) from error
