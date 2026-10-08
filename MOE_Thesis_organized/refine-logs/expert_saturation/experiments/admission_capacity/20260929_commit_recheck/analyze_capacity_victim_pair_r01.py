#!/usr/bin/env python3
"""Describe one complete capacity-qualified-victim off/on pair on seen H128 input."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import audit_h128_guarded_transfer_r02 as common
from analyze_fit_first_pair_r01 import distribution, request_result
from evaluate_goodput import pair, summarize


ARMS_BY_SESSION = {
    "moe-a-capacity-victim-session-r01-20261001":
        ("eager_performance_off", "eager_performance_on"),
    "moe-a-capacity-victim-session-b2-r01-20261001":
        ("eager_performance_on", "eager_performance_off"),
}


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def load_cell(session: Path, index: int, arm: str) -> tuple[dict, dict, dict]:
    cell = session / f"cell-{index:02d}-{arm}"
    archive = cell / "archive"
    hashes = common.read(cell / "output_sha256.json")
    require(isinstance(hashes, dict) and
            {"raw.json", "status.json", "config.json", "selective-store.json"} <= set(hashes) and
            common.archive_hashes(archive) == hashes,
            f"{arm}: original archive hashes differ")
    raw = common.read(archive / "raw.json")
    status = common.read(archive / "status.json")
    config = common.read(archive / "config.json")
    installed = common.read(archive / "selective-store.json")
    metrics = common.read(archive / "metrics.json")
    enabled = arm.endswith("_on")
    require(raw.get("status") == "COMPLETE" and raw.get("error") is None and
            len(raw.get("requests", [])) == 128 and
            all(r.get("status") == "completed" for r in raw["requests"]) and
            status.get("status") == status.get("capture_status") == "COMPLETE" and
            status.get("requests_completed") == 128 and status.get("error") is None,
            f"{arm}: 128-request episode incomplete")
    require(config.get("variant") == "eager" and
            config.get("measurement_mode") == "performance_sparse_preemptions" and
            config.get("commit_recheck") is False and
            config.get("fit_first_resume") is False and
            config.get("capacity_victim") is enabled and
            installed.get("status") == "DRAINED" and
            installed.get("store_scope") == "selected" and
            installed.get("commit_recheck") is False and
            installed.get("fit_first_resume") is False and
            installed.get("capacity_victim") is enabled and
            status.get("direct_commits") == 0 and
            status.get("fit_first_resumes") == 0,
            f"{arm}: actual policy flag differs")
    count = installed.get("capacity_victim_commits")
    require(type(count) is int and count >= 0 and
            count == status.get("capacity_victim_commits") == metrics.get("capacity_victim_commits") and
            (enabled or count == 0),
            f"{arm}: capacity-victim commit count differs")
    commands = (archive / "commands.txt").read_text()
    require(("--capacity-victim" in commands) is enabled and
            "--commit-recheck" not in commands and
            "--fit-first-resume" not in commands,
            f"{arm}: executed flag differs")
    return raw, installed, {
        "archive_map_sha256": common.sha_file(cell / "output_sha256.json"),
        "raw_sha256": hashes["raw.json"],
        "archive_file_count": len(hashes),
        "actual_preemption_count": raw.get("actual_preemption_count"),
        "forced_rotations": status.get("forced_rotations"),
        "capacity_victim_commits": count,
        "stop_reason_counts": status.get("finish_reason_counts"),
        "timing": common.read(archive / "timing.json"),
    }


def capacity_actions(on_raw: dict, on_store: dict, off_raw: dict) -> dict:
    on_internal = {r["internal_request_id"]: r for r in on_raw["requests"]}
    off_source = {r["request_id"]: r for r in off_raw["requests"]}
    require(len(on_internal) == len(off_source) == 128, "request identity duplicated")
    events = on_store.get("events", [])
    choices = [e for e in events if e.get("event") == "capacity_victim_choice"]
    commits = [e for e in events if e.get("event") == "capacity_victim_commit"]
    checks = {(e["step"], e["target"], e["victim"]): e for e in events
              if e.get("event") == "commit_check"}
    preemptions = {(e["engine_call_index"], e["internal_request_id"]): e
                   for e in on_raw.get("preemption_events", [])}
    by_choice = {}
    for event in choices:
        key = (event["choice_step"], event["target"], event["new_victim"])
        require(key not in by_choice, "duplicate capacity-victim choice")
        require(event["step"] == event["choice_step"], "choice step differs")
        by_choice[key] = event
    require(len(commits) == on_store["capacity_victim_commits"],
            "capacity-victim event/counter mismatch")
    committed = {}
    for event in commits:
        key = (event["choice_step"], event["target"], event["new_victim"])
        choice = by_choice.get(key)
        require(choice is not None and key not in committed and
                event["step"] == event["choice_step"] + 1 and
                all(event[field] == choice[field] for field in (
                    "old_victim", "free_blocks", "required_blocks",
                    "old_held_blocks", "new_held_blocks")),
                "capacity-victim commit lacks matching next-step choice")
        check = checks.get((event["step"], event["target"], event["new_victim"]))
        require(check is not None and check["reason"] == "READY",
                "capacity-victim commit lacks READY commit check")
        native = preemptions.get((event["step"], event["new_victim"]))
        require(native is not None and native.get("original_preemption_called") is True and
                native.get("original_preemption_returned") is True,
                "capacity-victim commit lacks corresponding native preemption")
        committed[key] = event

    origin = on_raw["measurement_origin_perf_counter_s"]
    actions = []
    cancelled = []
    latencies = []
    for key, choice in by_choice.items():
        target, old_victim, new_victim = (choice[name] for name in
                                          ("target", "old_victim", "new_victim"))
        require(target in on_internal and old_victim in on_internal and
                new_victim in on_internal and old_victim != new_victim,
                "capacity-victim actor identity differs")
        base = {name: choice[name] for name in (
            "choice_step", "target", "old_victim", "new_victim", "free_blocks",
            "required_blocks", "old_held_blocks", "new_held_blocks")}
        base["choice_perf_counter_s"] = choice["host_perf_counter_s"]
        if key not in committed:
            check = checks.get((choice["step"] + 1, target, new_victim))
            base["cancel_reason"] = check["reason"] if check else "NO_MATCHING_COMMIT_CHECK"
            cancelled.append(base)
            continue
        commit = committed[key]
        commit_time = commit["host_perf_counter_s"] - origin
        target_row = on_internal[target]
        later_tokens = [t for t in target_row["token_times_s"] if t > commit_time]
        first_output = min(later_tokens) if later_tokens else None
        later_events = [e for e in events if e.get("event") in
                        ("target_new_output", "target_terminal") and
                        e.get("request") == target and e["step"] > commit["step"]]
        outcome = min(later_events, key=lambda e: e["step"]) if later_events else None
        base.update(commit_step=commit["step"],
                    commit_perf_counter_s=commit["host_perf_counter_s"],
                    native_preemption_engine_call_index=commit["step"],
                    target_outcome_event={name: outcome[name] for name in
                                          ("event", "step", "new_output_tokens", "status")
                                          if name in outcome} if outcome else None,
                    first_later_target_output_s=first_output,
                    commit_to_first_later_target_output_ms=(
                        1000 * (first_output - commit_time) if first_output is not None else None))
        if first_output is not None:
            latencies.append(base["commit_to_first_later_target_output_ms"])
        for role, internal in (("target", target), ("old_victim", old_victim),
                               ("new_victim", new_victim)):
            source = on_internal[internal]["request_id"]
            require(source in off_source, "actor source identity missing from off cell")
            base.setdefault("request_outcomes", {})[role] = {
                "off": request_result(off_source[source]),
                "on": request_result(on_internal[internal]),
            }
        actions.append(base)
    return {
        "event_counts": {"choice": len(choices), "native_committed": len(commits),
                         "cancelled_or_uncommitted": len(cancelled),
                         "target_with_later_output": sum(
                             a["first_later_target_output_s"] is not None for a in actions),
                         "target_new_output_event": sum(
                             a["target_outcome_event"] is not None and
                             a["target_outcome_event"]["event"] == "target_new_output"
                             for a in actions)},
        "commit_to_first_later_target_output_ms": distribution(latencies),
        "actions": actions, "cancelled_choices": cancelled,
        "scope": "A successful capacity action requires the same-step native preemption of its new victim and a READY check after its choice. Later target output is reported separately from successful commit. Off/on request outcomes are separate trajectories, not same-state counterfactuals.",
    }


def analyze(session: Path) -> dict:
    require(session.is_dir() and session.name in ARMS_BY_SESSION,
            "wrong capacity-victim session")
    arms = ARMS_BY_SESSION[session.name]
    plan = common.read(session / "plan.json")
    receipt = common.read(session / "receipt.json")
    require(receipt.get("status") == "CELLS_COMPLETE" and
            receipt.get("plan_sha256") == common.sha_file(session / "plan.json") and
            tuple(c.get("arm") for c in plan.get("cells", [])) == arms and
            tuple(c.get("arm") for c in receipt.get("cells", [])) == arms,
            "capacity-victim pair plan or receipt incomplete")
    raws, stores, evidence = {}, {}, {}
    for index, arm in enumerate(arms):
        cell = receipt["cells"][index]
        require(cell.get("exit_code") == 0 and cell.get("timed_out") is False and
                cell.get("archive_status") == "VERIFIED",
                f"{arm}: controller cell failed")
        gate = arm.rsplit("_", 1)[1]
        raws[gate], stores[gate], evidence[gate] = load_cell(session, index, arm)
    require(not any(e.get("event") in ("capacity_victim_choice", "capacity_victim_commit")
                    for e in stores["off"].get("events", [])),
            "off cell contains capacity-victim action")
    metrics = {gate: summarize(raw, 128, 180) for gate, raw in raws.items()}
    require(metrics["off"]["completed"] == metrics["on"]["completed"] == 128,
            "pair lacks complete requests")
    return {
        "schema_version": 1, "status": "CAPACITY_VICTIM_EXPLORATORY_PAIR_COMPLETE",
        "plan_sha256": common.sha_file(session / "plan.json"),
        "receipt_sha256": common.sha_file(session / "receipt.json"),
        "cells": evidence, "metrics": metrics,
        "request_comparison": pair(metrics["off"], metrics["on"]),
        "output_differences": common.output_differences(raws["off"], raws["on"]),
        "capacity_victim": capacity_actions(raws["on"], stores["on"], raws["off"]),
        "scope": "One exploratory off/on pair on previously viewed H128. Same-input policy trajectories describe service and request outcomes; they do not identify same-state counterfactual benefit or statistical stability.",
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
                      "capacity_victim_commits": result["capacity_victim"]["event_counts"]["native_committed"]},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
