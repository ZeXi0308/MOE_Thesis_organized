#!/usr/bin/env python3
"""Reuse the locked new-host runner for reverse-order adaptive/fixed4/q1 cells.

The cell's inherited package-directory label is only a path; exact candidate
identity is the new manifest in this plan, verified for every payload.
"""
import argparse
import json
from pathlib import Path
import signal
import sys

import run_strong_newhost as base


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    if plan.get("kind") != "RECOVERY_LEASE_SAME_PATH_TRIPLET":
        raise ValueError("This runner requires a lease comparison plan")
    base.MANIFEST = plan["package_manifest_sha256"]
    base.ARMS = ("adaptive", "fixed4", "q1")
    original_env = base.private_env

    def env(accepted_plan, cache, arm, memory_file):
        result = original_env(accepted_plan, cache, "queue_fund", memory_file)
        result["A_RECOVERY_LEASE_MODE"] = arm
        return result

    base.private_env = env
    signal.signal(signal.SIGTERM, lambda number, frame: (_ for _ in ()).throw(InterruptedError(f"signal {number}")))
    signal.signal(signal.SIGALRM, lambda number, frame: (_ for _ in ()).throw(TimeoutError("Group wall budget exhausted")))
    return base.run(plan, args.plan)


if __name__ == "__main__":
    raise SystemExit(main())
