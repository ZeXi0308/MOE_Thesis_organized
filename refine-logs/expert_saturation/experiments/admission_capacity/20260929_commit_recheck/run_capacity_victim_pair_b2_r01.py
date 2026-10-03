#!/usr/bin/env python3
"""One reverse-order exploratory on/off pair; reuse the proven shared-lock and archive runner."""
import json
from pathlib import Path
import shutil
import signal
import sys
import serial_group_h1_guard_qual_r02 as base

PACKAGE_SHA256 = "f800a9171bb2e9f012bcbd16ecdd9695c7e6b49f6922d0dac100ae9860d2eaf9"
ARMS = ("eager_performance_on", "eager_performance_off")
BASE = "/root/moe-a-capacity-victim-stage-r01-20261001"
SESSION = "/root/moe-a-capacity-victim-session-b2-r01-20261001"
OUTPUTS = tuple("/root/moe-a-capacity-victim-b2-output-" + gate + "-r01-20261001"
                for gate in ("on", "off"))
original_validate = base.validate_plan
original_run = base.run_session
base.PACKAGE_NAME = "candidate_capacity_victim_r01"
base.PACKAGE_SHA256 = PACKAGE_SHA256

def validate(plan):
    base.require(plan.get("session_dir") == SESSION and
                 plan.get("approved_total_wall_seconds") == 3200,
                 "exploratory pair session or wall bound changed")
    cells = plan.get("cells", [])
    base.require(tuple(c.get("arm") for c in cells) == ARMS, "on/off order changed")
    for index, cell in enumerate(cells):
        base.require(cell.get("package_dir") == BASE + "/" + base.PACKAGE_NAME and
                     cell.get("output_dir") == OUTPUTS[index] and
                     cell.get("max_wall_seconds") == 900, "frozen cell changed")
        base.FROZEN_ARMS = (ARMS[index],)
        original_validate(dict(plan, cells=[cell]))
    base.FROZEN_ARMS = ARMS
    base.require(plan["approved_total_wall_seconds"] >=
                 sum(c["max_wall_seconds"] for c in cells) + 360 + 480 + 60 + 90 + 20,
                 "insufficient total pair time")
    return plan

def argv(package, output):
    base.require(str(output) in OUTPUTS, "unknown output arm")
    gate = ("on", "off")[OUTPUTS.index(str(output))]
    return [str(package / "pkg/run.sh"), "eager", "performance", gate, str(output)]

def run(plan, digest, plan_bytes):
    # Each pair's raw and archive total about 240 MB; reserve room for compile cache.
    base.require(shutil.disk_usage("/root").free >= 2 * 1024**3,
                 "need 2 GiB system free before starting this pair")
    return original_run(plan, digest, plan_bytes)

base.validate_plan = validate
base.qualification_argv = argv
base.run_session = run
if __name__ == "__main__":
    signal.signal(signal.SIGTERM, base.stop_requested)
    try:
        raise SystemExit(base.main())
    except (OSError, ValueError, base.subprocess.SubprocessError) as exc:
        print("ABORT_CAPACITY_VICTIM_PAIR:", exc, file=sys.stderr)
        raise SystemExit(75)
