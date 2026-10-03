#!/usr/bin/env python3
"""One locked CPU-qualified BidKV full-running baseline triplet; GPU unrun."""

import json
import shutil
from pathlib import Path
import signal
import sys

import run_native_current_guard_triplet_r02 as retry


group, base, current = retry.group, retry.base, retry.prior
base.PACKAGE_NAME = "candidate_native_bidkv_full_running_r01"
base.PACKAGE_SHA256 = "844caa2fa985d6101ecf74b2f5aae76a778711868f96dd8bd67df0b3d97855e2"
group.PACKAGE_SHA256 = base.PACKAGE_SHA256
current.prior.RULES = ("tail", "bidkv_score", "bidkv_score")
current.inherited.MODES = ("off", "off", "off")
current.GUARDS = ("off", "off", "off")
FULL_MODES = ("off", "off", "on")
group.ARMS = ("tail_break", "bidkv_suffix_break", "bidkv_full_break")
group.BASE = "/root/moe-a-native-bidkv-full-running-stage-r01-20261002"
group.SESSION = "/root/moe-a-native-bidkv-full-running-session-r01-20261002"
group.SOURCE_CACHE = Path(
    "/root/moe-a-native-backfill-only-primary-session-r01-20261001/runtime-cache-first"
)
group.OUTPUTS = tuple(
    "/root/moe-a-native-bidkv-full-running-output-" + label + "-r01-20261002"
    for label in group.LABELS
)
retry.DISK_RESERVE_BYTES = 1280 * 1024**2

previous_private_env = base.private_model_env
previous_prepare_cache = base.prepare_runtime_cache


def prepare_runtime_cache(session_dir):
    previous_prepare_cache(session_dir)
    receipt_path = session_dir / "runtime-cache-seed-receipt.json"
    receipt = json.loads(receipt_path.read_text())
    base.require(receipt.get("disk_reserve_after_copies_bytes") == retry.DISK_RESERVE_BYTES,
                 "runtime cache receipt reserve differs from 1.25 GiB plan")
    receipt["reserve_basis"] = (
        "This baseline group reserves 1.25 GiB after three private warm-cache copies; "
        "the numeric disk_reserve_after_copies_bytes field is authoritative."
    )
    base.write_json(receipt_path, receipt)


def private_model_env(plan, session_dir):
    env = previous_private_env(plan, session_dir)
    found = [index for index, arm in enumerate(group.ARMS)
             if (session_dir / f"cell-{index:02d}-{arm}").exists()]
    index = found[-1] if found else 0
    env["A_NATIVE_VICTIM_FULL_RUNNING"] = FULL_MODES[index]
    if found:
        base.write_json(
            session_dir / f"cell-{index:02d}-{group.ARMS[index]}" / "full-running-mode.json",
            dict(arm=group.ARMS[index], victim_rule=current.prior.RULES[index],
                 self_preempt_continue="off", current_guard="off",
                 full_running=FULL_MODES[index]),
        )
    return env


def validate_plan(plan):
    base.require(plan.get("session_dir") == group.SESSION
                 and plan.get("approved_total_wall_seconds") == 4800
                 and plan.get("warm_runtime_cache_source") == str(group.SOURCE_CACHE)
                 and plan.get("package_manifest_sha256") == base.PACKAGE_SHA256,
                 "full-running triplet identity or budget changed")
    cells = plan.get("cells", [])
    base.require(tuple(cell.get("arm") for cell in cells) == group.ARMS,
                 "full-running triplet order changed")
    try:
        for index, cell in enumerate(cells):
            base.require(cell.get("package_dir") ==
                         group.BASE + "/" + group.LABELS[index] + "/" + base.PACKAGE_NAME
                         and cell.get("output_dir") == group.OUTPUTS[index]
                         and cell.get("max_wall_seconds") == 900,
                         "full-running triplet cell changed")
            base.FROZEN_ARMS = (group.ARMS[index],)
            group.original_validate(dict(plan, cells=[cell]))
    finally:
        base.FROZEN_ARMS = group.ARMS
    base.require(plan["approved_total_wall_seconds"] >=
                 sum(cell["max_wall_seconds"] for cell in cells) + 360 + 480 + 60 + 135 + 20,
                 "insufficient total triplet time")
    return plan


def run_session(plan, digest, plan_bytes):
    base.require(shutil.disk_usage("/root").free >= retry.DISK_RESERVE_BYTES,
                 "less than 1.25 GiB system free before triplet")
    # retry.prepare_runtime_cache enforces the same reserve after three private copies.
    return group.original_run(plan, digest, plan_bytes)


base.private_model_env = private_model_env
base.prepare_runtime_cache = prepare_runtime_cache
base.validate_plan = validate_plan
base.run_session = run_session


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, base.stop_requested)
    try:
        raise SystemExit(base.main())
    except (OSError, ValueError, base.subprocess.SubprocessError) as error:
        print("ABORT_NATIVE_BIDKV_FULL_RUNNING:", error, file=sys.stderr)
        raise SystemExit(75)
