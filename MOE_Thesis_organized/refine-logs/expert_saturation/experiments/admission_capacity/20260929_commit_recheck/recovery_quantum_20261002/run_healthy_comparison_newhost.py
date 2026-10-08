#!/usr/bin/env python3
"""Select native/ordinary comparators through the unchanged healthy controller.

The existing controller retains locking, model/package checks, private caches,
timeouts, launch-once cells and archive checks. Only the arm list, policy
environment and accurate controller/mode receipts are extended here.
"""
import argparse
import json
from pathlib import Path
import signal
import subprocess
import sys

import run_healthy_newhost as base

BASE_SHA256 = "a3c025af8ad5be2bc312887a3cebbe0d215408cd8f7f838df15509f8e7e5e5c7"
GROUPS = (("native", "ordinary"), ("native", "ordinary", "q1", "fixed4", "adaptive"))
ARM_CONFIGS = {
    "native": {"A_NATIVE_OLDEST_ADMISSION": "native", "A_NATIVE_OLDEST_REPEAT": "1", "A_RECOVERY_LEASE_MODE": "off"},
    "ordinary": {"A_NATIVE_OLDEST_ADMISSION": "ordinary", "A_NATIVE_OLDEST_REPEAT": "0", "A_RECOVERY_LEASE_MODE": "off"},
    "q1": {"A_NATIVE_OLDEST_ADMISSION": "queue_fund", "A_NATIVE_OLDEST_REPEAT": "1", "A_RECOVERY_LEASE_MODE": "q1"},
    "fixed4": {"A_NATIVE_OLDEST_ADMISSION": "queue_fund", "A_NATIVE_OLDEST_REPEAT": "1", "A_RECOVERY_LEASE_MODE": "fixed4"},
    "adaptive": {"A_NATIVE_OLDEST_ADMISSION": "queue_fund", "A_NATIVE_OLDEST_REPEAT": "1", "A_RECOVERY_LEASE_MODE": "adaptive"},
}
_private_env = base.private_env
_write_json = base.write_json


def private_env(plan, cache, arm, memory_file):
    env = _private_env(plan, cache, arm, memory_file)
    env.update(ARM_CONFIGS[arm])
    return env


def write_json(path, value, *, new=False):
    if Path(path).name == "oldest-admission-mode.json":
        value = dict(value)
        arm = value["comparator_arm"]
        policy = ARM_CONFIGS[arm]
        value.update(mode=policy["A_NATIVE_OLDEST_ADMISSION"],
                     recovery_lease_mode=policy["A_RECOVERY_LEASE_MODE"],
                     effective_ordinary_backfill=arm == "ordinary",
                     oldest_repeat=policy["A_NATIVE_OLDEST_REPEAT"] == "1",
                     effective_oldest_admission_mode=None if arm == "ordinary" else policy["A_NATIVE_OLDEST_ADMISSION"],
                     actual_policy_environment=dict(policy))
    elif Path(path).name == "receipt.json":
        value = dict(value)
        value.update(comparison_controller_sha256=base.digest(__file__),
                     comparison_controller_entry=Path(__file__).name,
                     base_controller_sha256=BASE_SHA256,
                     arm_policy_environment={arm: ARM_CONFIGS[arm] for arm in base.ARMS})
    return _write_json(path, value, new=new)


def configure(plan):
    base.require(base.digest(base.__file__) == BASE_SHA256, "Inherited healthy controller bytes changed")
    base.require(plan.get("comparison_kind") == "HEALTHY_NATIVE_ORDINARY_LEASE_COMPARISON", "Wrong comparison plan")
    arms = tuple(plan["arms"])
    base.require(arms in GROUPS, "Only the pinned two-arm or five-arm comparison is allowed")
    base.require(plan.get("comparison_controller_sha256") == base.digest(__file__), "Comparison wrapper bytes changed")
    base.require(plan.get("base_controller_sha256") == BASE_SHA256, "Wrong inherited controller")
    base.require(plan.get("actual_arm_configs") == {arm: ARM_CONFIGS[arm] for arm in arms}, "Plan arm environment differs")
    base.require(plan["approved_total_wall_seconds"] >= len(arms) * plan["per_cell_wall_seconds"] + 600,
                 "Group budget does not cover every full cell and setup")
    base.ARMS = arms
    base.private_env = private_env
    base.write_json = write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, lambda number, frame: (_ for _ in ()).throw(InterruptedError(f"signal {number}")))
    signal.signal(signal.SIGALRM, lambda number, frame: (_ for _ in ()).throw(TimeoutError("Group wall budget exhausted")))
    try:
        plan = json.loads(args.plan.read_text())
        configure(plan)
        return base.run(plan, args.plan)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"ABORT_HEALTHY_COMPARISON: {type(error).__name__}: {error}", file=sys.stderr)
        return 75


if __name__ == "__main__":
    raise SystemExit(main())
