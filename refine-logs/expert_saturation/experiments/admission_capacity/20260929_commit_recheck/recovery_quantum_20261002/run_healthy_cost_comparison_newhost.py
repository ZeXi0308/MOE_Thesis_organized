#!/usr/bin/env python3
"""Enable aggregate cost counters through the unchanged healthy comparison path.

The two inherited controllers retain policy selection, resource checks, locking,
private caches, timeouts and archives. This wrapper changes only the candidate
manifest, optional observer environment, and explicit provenance receipts.
"""
import argparse
import json
from pathlib import Path
import signal
import subprocess
import sys

import run_healthy_comparison_newhost as compare

base = compare.base
COST_MANIFEST = "1cccd7f0865fd7104a98a2a42e9a1c0b841a44733b9bb9ac84773c30f7f34f34"
COMPARISON_SHA256 = "d7798a0fb31c2a4f4197495cbecd8ef8c63f506904e314464074d47bce3d9b2e"
OBSERVER_ENVIRONMENT = {"A_AGGREGATE_OFFLOAD_COSTS": "1"}
_active_plan = None


def private_env(plan, cache, arm, memory_file):
    env = compare.private_env(plan, cache, arm, memory_file)
    env.update(OBSERVER_ENVIRONMENT)
    return env


def write_json(path, value, *, new=False):
    if Path(path).name in ("receipt.json", "oldest-admission-mode.json"):
        value = dict(value)
        value.update(aggregate_offload_costs_enabled=True,
                     actual_observer_environment=dict(OBSERVER_ENVIRONMENT))
    if Path(path).name == "receipt.json":
        value.update(cost_controller_entry=Path(__file__).name,
                     cost_controller_sha256=base.digest(__file__),
                     cost_package_manifest_sha256=COST_MANIFEST,
                     cost_source_package=_active_plan["source_package"],
                     cost_observer="NATIVE_COMPLETION_AGGREGATES_V1",
                     cost_scope="After warmup reset; capture and post-capture drain separated by completion-metadata observation. Summed native copy time may overlap and is not exposed latency.")
    return compare.write_json(path, value, new=new)


def configure(plan):
    global _active_plan
    base.require(base.digest(compare.__file__) == COMPARISON_SHA256,
                 "Inherited comparison controller bytes changed")
    base.require(plan.get("cost_kind") == "HEALTHY_AGGREGATE_COST_COMPARISON_R02",
                 "Wrong cost comparison plan")
    base.require(plan.get("controller_entry") == Path(__file__).name
                 and plan.get("cost_controller_sha256") == base.digest(__file__),
                 "Cost wrapper identity differs")
    base.require(plan.get("package_manifest_sha256") == COST_MANIFEST,
                 "Wrong aggregate cost candidate manifest")
    base.require(plan.get("actual_observer_environment") == OBSERVER_ENVIRONMENT,
                 "Cost observer environment differs")
    compare.configure(plan)
    base.MANIFEST = COST_MANIFEST
    base.private_env = private_env
    base.write_json = write_json
    _active_plan = plan


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
        print(f"ABORT_HEALTHY_COST_COMPARISON: {type(error).__name__}: {error}", file=sys.stderr)
        return 75


if __name__ == "__main__":
    raise SystemExit(main())
