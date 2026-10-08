#!/usr/bin/env python3
"""Describe one capacity-victim A/A pair on the same seen H128 input."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import audit_h128_guarded_transfer_r02 as common
from analyze_capacity_victim_pair_r01 import load_cell, require
from evaluate_goodput import pair, summarize


SESSION_NAME = "moe-a-capacity-identical-session-r01-20261001"
WARM_SESSION_NAME = "moe-a-capacity-warm-identical-session-r01-20261001"
ARMS = ("capacity_reference_first_on", "capacity_reference_second_on")
SOURCE_MAP_SHA256 = "9289044c9bf0af44e97148fe1d09d2df28f9f94dc97364c29f23d96fd9f07216"
RUNTIME_MAP_SHA256 = "2999a7c15358451974158e5704a30faff30b747b2988272e29061619252c5820"
WORKLOAD_SHA256 = "606f71fd7b55f08a1ce6c246a3cddb6f732d0b731201e5acfc4a74c5cf0c9c5a"


def map_digest(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def analyze(session: Path) -> dict:
    require(session.is_dir() and session.name in (SESSION_NAME, WARM_SESSION_NAME, "moe-a-capacity-warm-identical-session-r02-20261001"),
            "wrong capacity-identical session")
    plan = common.read(session / "plan.json")
    receipt = common.read(session / "receipt.json")
    require(receipt.get("status") == "CELLS_COMPLETE" and
            receipt.get("plan_sha256") == common.sha_file(session / "plan.json") and
            tuple(c.get("arm") for c in plan.get("cells", [])) == ARMS and
            tuple(c.get("arm") for c in receipt.get("cells", [])) == ARMS,
            "capacity-identical pair plan or receipt incomplete")
    raws, evidence = {}, {}
    configs = []
    for index, arm in enumerate(ARMS):
        cell = receipt["cells"][index]
        require(cell.get("exit_code") == 0 and cell.get("timed_out") is False and
                cell.get("archive_status") == "VERIFIED",
                f"{arm}: controller cell failed")
        raw, _store, cell_evidence = load_cell(session, index, arm)
        archive = session / f"cell-{index:02d}-{arm}" / "archive"
        config = common.read(archive / "config.json")
        environment = common.read(archive / "environment.json")
        require(config.get("workload_sha256") == WORKLOAD_SHA256 and
                config.get("max_seconds") == 180 and
                config.get("cap") == config.get("engine_max_num_seqs") == 32 and
                config.get("fixed_kv_cache_memory_bytes") == 8592031744 and
                config.get("global_cooldown_steps") == 0 and
                map_digest(environment.get("source_sha256")) == SOURCE_MAP_SHA256 and
                map_digest(environment.get("vllm_source_sha256")) == RUNTIME_MAP_SHA256,
                f"{arm}: frozen input, thresholds, or source differs from original capacity pair")
        configs.append(config)
        label = ("first", "second")[index]
        raws[label] = raw
        evidence[label] = {**cell_evidence,
                           "config_sha256": common.sha_file(archive / "config.json"),
                           "source_map_sha256": SOURCE_MAP_SHA256,
                           "runtime_map_sha256": RUNTIME_MAP_SHA256}
    require(configs[0] == configs[1], "A/A cell configurations differ")
    metrics = {label: summarize(raw, 128, 180) for label, raw in raws.items()}
    require(metrics["first"]["completed"] == metrics["second"]["completed"] == 128,
            "A/A pair lacks complete requests")
    return {
        "schema_version": 1,
        "session": session.name,
        "status": "CAPACITY_IDENTICAL_EXPLORATORY_PAIR_COMPLETE",
        "plan_sha256": common.sha_file(session / "plan.json"),
        "receipt_sha256": common.sha_file(session / "receipt.json"),
        "cells": evidence, "metrics": metrics,
        "request_comparison": pair(metrics["first"], metrics["second"]),
        "output_differences": common.output_differences(raws["first"], raws["second"]),
        "scope": "One same-policy first/second pair on previously viewed H128 input. It describes within-setting trajectory variation, not a statistical estimate or a causal correction to other pairs.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    require(not args.output.exists(), "output must be new")
    result = analyze(args.session)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": result["status"],
                      "completed": [result["metrics"][k]["completed"]
                                    for k in ("first", "second")]},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
