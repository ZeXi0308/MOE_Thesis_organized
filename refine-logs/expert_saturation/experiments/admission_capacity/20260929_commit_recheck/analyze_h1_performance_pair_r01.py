#!/usr/bin/env python3
"""Validate and summarize an H1 off/on pair against its frozen qualification.

The original 25-file package and complete diagnostic audit are prerequisites.
Performance traces are separate policy trajectories, not common-state replay.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import shlex

import audit_h128_guarded_transfer_r02 as common
from evaluate_goodput import pair, summarize

require, read, sha = common.require, common.read, common.sha_file
BASE = Path(__file__).resolve().parent
PACKAGE_SHA = "4231a687db066f113fcc14676f91e8e5825be05b76ef0834112e053726aa13aa"
PROTOCOL_SHA = "3a36ae6e42c33eacfd0623c0f3448641ebe5adf75e42e99890c070931662f8f5"
QUALIFICATION_SHA = "7a3b323673ac7b42d666f758de2109ad21ed625df9957bed8cfd149a75437f08"
QUAL_ARCHIVE = BASE / "moe-a-h1-guard-qual-session-r02-20260930/cell-00-eager_diagnostic_on/archive"
ORDERS = {1: ("eager_performance_off", "eager_performance_on"),
          2: ("eager_performance_on", "eager_performance_off")}
SESSION_NAMES = {1: "moe-a-h1-perf-b1-session-r01-20261001",
                 2: "moe-a-h1-perf-b2-session-r01-20261001"}
PACKAGE_NAMES = {1: "candidate_h1_perf_b1_r01", 2: "candidate_h1_perf_b2_r01"}
REMOTE_PACKAGE_ROOT = "/root/autodl-tmp/moe-a-pkg-20260930"
OUTPUTS = {
    1: ("/root/moe-a-h1-perf-b1-output-eager-off-r01-20261001",
        "/root/moe-a-h1-perf-b1-output-eager-on-r01-20261001"),
    2: ("/root/moe-a-h1-perf-b2-output-eager-on-r01-20261001",
        "/root/moe-a-h1-perf-b2-output-eager-off-r01-20261001"),
}
H1_REQUIRED_OUTPUTS = common.REQUIRED_OUTPUTS | {
    "metrics.json", "resolved-eos.json", "memory-before.json", "memory-after.json",
    "gpu-after.json",
}


def check_pair_identity(session: Path, plan: dict, receipt: dict) -> tuple[list, list, int]:
    block = plan.get("block_index")
    require(type(block) is int and block in ORDERS, "unknown block")
    require(session.is_dir() and not session.is_symlink() and
            session.name == SESSION_NAMES[block] and
            plan.get("session_dir") == "/root/" + SESSION_NAMES[block],
            "pair session identity changed")
    require(plan.get("approved_total_wall_seconds") == 3200,
            "pair total wall budget changed")
    planned, observed = plan.get("cells"), receipt.get("cells")
    require(isinstance(planned, list) and isinstance(observed, list) and
            len(planned) == len(observed) == 2 and
            tuple(c.get("arm") for c in planned) == ORDERS[block] and
            tuple(c.get("arm") for c in observed) == ORDERS[block],
            "pair plan or receipt arm order changed")
    package_dir = REMOTE_PACKAGE_ROOT + "/" + PACKAGE_NAMES[block]
    for i, (pc, rc) in enumerate(zip(planned, observed)):
        require(pc.get("package_dir") == package_dir and
                pc.get("output_dir") == rc.get("output_dir") == OUTPUTS[block][i] and
                pc.get("max_wall_seconds") == 900,
                f"pair cell {i} package/output/budget identity changed")
    return planned, observed, block


def summarize_pair(raws: dict) -> dict:
    metrics = {gate: summarize(raw, 128, 180) for gate, raw in raws.items()}
    off, on = metrics["off"], metrics["on"]
    require(all(metric["completed"] == 128 and metric["failed"] == metric["unfinished"] == 0
                for metric in (off, on)), "pair denominator requires two complete cohorts")
    require(all(isinstance(off[key], (int, float)) and math.isfinite(off[key]) and off[key] > 0
                for key in ("actual_output_tokens_s", "mean_completed_flow_s")) and
            all(isinstance(metric["max_gap_request_max_s"], (int, float)) and
                math.isfinite(metric["max_gap_request_max_s"]) for metric in (off, on)),
            "pair off denominator or maximum gap undefined")
    rate = on["actual_output_tokens_s"] / off["actual_output_tokens_s"]
    flow = on["mean_completed_flow_s"] / off["mean_completed_flow_s"]
    gap = on["max_gap_request_max_s"] < off["max_gap_request_max_s"]
    return {"metrics": metrics, "request_comparison": pair(off, on),
            "output_differences": common.output_differences(raws["off"], raws["on"]),
            "output_rate_ratio_on_off": rate, "mean_flow_ratio_on_off": flow,
            "within_efficiency_budget": rate >= .97 and flow <= 1.05,
            "max_gap_better": gap, "meets_block_criterion": rate >= .97 and flow <= 1.05 and gap,
            "ratio_denominator": "off_cell_in_same_block"}


def analyze(session: Path, plan_sha: str, qualification: Path, qualification_sha: str) -> dict:
    require(qualification_sha == QUALIFICATION_SHA, "qualification audit identity changed")
    require(sha(qualification) == qualification_sha, "qualification audit SHA mismatch")
    qual = read(qualification)
    # This audit is itself bound to the original source/resources/action receipts.
    require(qual.get("status") == "OBSERVED_DIRECT_ACTION_CHAIN", "H1 action not qualified")
    require(sha(session / "plan.json") == plan_sha, "frozen pair plan SHA mismatch")
    plan, receipt = read(session / "plan.json"), read(session / "receipt.json")
    planned, observed, block = check_pair_identity(session, plan, receipt)
    require(plan.get("protocol_sha256") == PROTOCOL_SHA, "protocol changed")
    require(plan.get("qualification_audit_sha256") == qualification_sha,
            "plan qualification differs")
    require(receipt.get("status") == "CELLS_COMPLETE" and receipt.get("plan_sha256") == plan_sha,
            "pair did not complete; retain failure separately without complete-case ranking")
    require(plan["authorized_gpu_uuid"] == "GPU-3fc910c2-bf65-5273-e6b5-6c0d8b6ce03e" and
            plan["approved_host_bytes"] == 96636764160 and
            receipt.get("held_lock_device_inode") == plan["expected_lock_device_inode"] == "2304:25841682495",
            "resource identity mismatch")
    common.model_check(session, plan, receipt)
    package = BASE / PACKAGE_NAMES[block]
    require(sha(package / "manifest.json") == PACKAGE_SHA, "qualified package manifest changed")
    manifest = read(package / "manifest.json")
    require(isinstance(manifest, dict) and len(manifest) == 25,
            "qualified package payload set changed")
    for name, digest in manifest.items():
        require(sha(package / name) == digest, f"qualified payload changed: {name}")
    workload = read(package / "pkg/inputs/workload.json")
    cohort = {r["request_id"]: (r["document_id"], r["prompt_token_ids_sha256"],
              workload["arrival_traces_s"]["steady"][i],
              len(workload["actual_prompt_token_ids"][i]), 1024)
              for i, r in enumerate(workload["source_requests"])}
    reference_config = read(QUAL_ARCHIVE / "config.json")
    reference_engine = read(QUAL_ARCHIVE / "engine_args.json")
    reference_environment = read(QUAL_ARCHIVE / "environment.json")
    raws, evidence = {}, {}
    for i, (pc, rc) in enumerate(zip(planned, observed)):
        arm, gate = pc["arm"], pc["arm"].rsplit("_", 1)[1]
        on = gate == "on"
        require(rc.get("arm") == arm and rc.get("launch_status") == "FINISHED" and
                rc.get("exit_code") == 0 and
                rc.get("timed_out") is False and rc.get("archive_status") == "VERIFIED" and
                rc.get("gpu_process_state_after") == "EMPTY" and rc.get("gpu_process_rows_after") == [] and
                rc.get("argv") == [pc["package_dir"] + "/pkg/run.sh", "eager", "performance", gate, pc["output_dir"]],
                f"{arm}: execution/exit/lock-boundary receipt failed")
        cell = session / f"cell-{i:02d}-{arm}"
        archive = cell / "archive"
        hashes = read(cell / "output_sha256.json")
        require(H1_REQUIRED_OUTPUTS <= set(hashes) and common.archive_hashes(archive) == hashes,
                f"{arm}: original archive hash mismatch")
        a = lambda name: read(archive / name)
        raw, config, env = a("raw.json"), a("config.json"), a("environment.json")
        common.cohort_identity(raw, cohort)
        expected_config = dict(reference_config)
        expected_config.update(commit_recheck=on, measurement_mode="performance_sparse_preemptions",
                               metric_role="Primary performance with sparse request preemption events")
        # metric_role is prose; all actual input/engine/policy fields must match.
        require({k: v for k, v in config.items() if k != "metric_role"} ==
                {k: v for k, v in expected_config.items() if k != "metric_role"},
                f"{arm}: config differs beyond on/off and measurement mode")
        require(a("engine_args.json") == reference_engine, f"{arm}: engine differs")
        expected_sources = {name: manifest[f"pkg/{name}"] for name in common.ARM_SOURCES["eager"]}
        require(env["source_sha256"] == expected_sources and env["vllm_source_sha256"] == common.PINNED_RUNTIME_SHA,
                f"{arm}: source/runtime bytes differ")
        require(all(env[k] == reference_environment[k] for k in
                    ("python", "torch", "cuda", "vllm", "transformers")), f"{arm}: runtime versions differ")
        require(plan["authorized_gpu_uuid"] in env["gpu_before"]["device"] and
                not env["gpu_before"]["compute_processes"], f"{arm}: GPU not isolated")
        require(a("safe-cap-qualification.json")["status"] == "QUALIFIED" and
                a("safe-cap-qualification.json")["usable_blocks"] == 4096 and
                a("memory-after-init.json")["kv_storage_bytes"] == common.KV_BYTES,
                f"{arm}: GPU KV budget differs")
        for name in ("host-after-init.json", "host-before.json"):
            host = a(name)
            require(host["cpu_kv"]["unique_storage_bytes"] == common.HOST_BYTES and
                    host["manager"]["capacity_blocks"] == 8192 and
                    host["parent_cgroup"]["values"]["memory.max"] == plan["approved_host_bytes"],
                    f"{arm}: host allocation or cgroup differs")
        for j, count in enumerate((32, 32, 2)):
            warmup = a(f"warmup-{j}.json")
            require(warmup["status"] == "COMPLETE" and len(warmup["requests"]) == count and
                    all(r["status"] == "completed" for r in warmup["requests"]), f"{arm}: warmup incomplete")
        require(a("warmup-cache-reset.json")["success"] is True, f"{arm}: warmup not reset")
        host = a("host-after.json")
        require(all(host["pending"][k] == 0 for k in
                    ("scheduler_store_jobs", "scheduler_load_jobs", "pending_worker_acknowledgements")) and
                host["manager"]["pending_store_entries"] == host["manager"]["active_load_references"] == 0,
                f"{arm}: final drain incomplete")
        status, installed = a("status.json"), a("selective-store.json")
        require(status["status"] == status["capture_status"] == "COMPLETE" and
                status["requests_completed"] == 128 and status["error"] is None and
                status["finish_reason_counts"] == dict(Counter(r["stop_reason"] for r in raw["requests"])),
                f"{arm}: completion receipt mismatch")
        require(installed["status"] == "DRAINED" and installed["commit_recheck"] is on and
                installed["store_scope"] == "selected" and installed["native_calc_overridden"] is True and
                installed["diagnostic"] is False, f"{arm}: installed policy mismatch")
        directs = [e for e in installed["events"] if e.get("event") == "direct_commit"]
        require(status["direct_commits"] == installed["direct_commits"] == len(directs) and
                (on or len(directs) == 0), f"{arm}: direct count mismatch")
        commands = shlex.split((archive / "commands.txt").read_text())
        require(("--commit-recheck" in commands) is on and
                commands[commands.index("--measurement-mode") + 1] == "performance",
                f"{arm}: executed CLI differs")
        evidence[gate] = {"archive_file_count": len(hashes), "archive_map_sha256": sha(cell / "output_sha256.json"),
                          "raw_sha256": hashes["raw.json"], "stop_reason_counts": status["finish_reason_counts"],
                          "forced_rotations": status["forced_rotations"], "direct_commits": len(directs),
                          "direct_events": directs, "native_reservation_gate": installed["native_reservation_gate"],
                          "timing": a("timing.json"), "post_request_drain": a("post-request-drain.json")}
        raws[gate] = raw
    return {"schema_version": 1, "status": "H1_PERFORMANCE_PAIR_COMPLETE", "block_index": block,
            "plan_sha256": plan_sha, "protocol_sha256": PROTOCOL_SHA,
            "qualification_audit_sha256": qualification_sha, "receipt_sha256": sha(session / "receipt.json"),
            "package_manifest_sha256": PACKAGE_SHA, "cells": evidence, **summarize_pair(raws),
            "scope": "Single ordered pair on previously viewed H128. Separate natural-output policy trajectories; no equal-work, per-action causal, quality or statistical-stability claim."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument("--qualification-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "output must be new")
    result = analyze(args.session, args.plan_sha256, args.qualification, args.qualification_sha256)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "criterion": result["meets_block_criterion"]}))


if __name__ == "__main__":
    main()
