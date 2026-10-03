#!/usr/bin/env python3
"""Describe one complete exploratory fit-first off/on pair on seen H128 input."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import median

import audit_h128_guarded_transfer_r02 as common
from evaluate_goodput import pair, summarize


ARMS = ("eager_performance_off", "eager_performance_on")
SESSION_NAME = "moe-a-fit-first-session-r01-20261001"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def distribution(values: list[float]) -> dict:
    return {"count": len(values), "min_ms": min(values) if values else None,
            "median_ms": median(values) if values else None,
            "max_ms": max(values) if values else None}


def request_result(row: dict) -> dict:
    times = row["token_times_s"]
    distinct_times = sorted(set(times))
    gaps = [b - a for a, b in zip(distinct_times, distinct_times[1:])]
    return {
        "request_id": row["request_id"], "status": row["status"],
        "stop_reason": row["stop_reason"], "output_tokens": len(row["output_token_ids"]),
        "output_token_ids_sha256": hashlib.sha256(json.dumps(
            row["output_token_ids"], separators=(",", ":")).encode()).hexdigest(),
        "arrival_s": row["arrival_s"], "completion_s": row["completion_s"],
        "flow_s": row["completion_s"] - row["arrival_s"],
        "ttft_s": times[0] - row["arrival_s"] if times else None,
        "max_observed_generation_gap_s": max(gaps) if gaps else None,
    }


def load_cell(session: Path, index: int, arm: str) -> tuple[dict, dict, dict]:
    cell = session / f"cell-{index:02d}-{arm}"
    archive = cell / "archive"
    hashes = common.read(cell / "output_sha256.json")
    require(isinstance(hashes, dict) and
            {"raw.json", "status.json", "config.json", "selective-store.json"} <= set(hashes) and
            common.archive_hashes(archive) == hashes,
            f"{arm}: original archive hashes differ")
    raw, status = common.read(archive / "raw.json"), common.read(archive / "status.json")
    config, installed = common.read(archive / "config.json"), common.read(archive / "selective-store.json")
    metrics = common.read(archive / "metrics.json")
    enabled = arm.endswith("_on")
    require(raw.get("status") == "COMPLETE" and raw.get("error") is None and
            len(raw.get("requests", [])) == 128 and
            all(row.get("status") == "completed" for row in raw["requests"]) and
            status.get("status") == status.get("capture_status") == "COMPLETE" and
            status.get("requests_completed") == 128 and status.get("error") is None,
            f"{arm}: 128-request episode incomplete")
    require(config.get("variant") == "eager" and
            config.get("measurement_mode") == "performance_sparse_preemptions" and
            config.get("commit_recheck") is False and
            config.get("fit_first_resume") is enabled and
            installed.get("status") == "DRAINED" and
            installed.get("commit_recheck") is False and
            installed.get("fit_first_resume") is enabled and
            installed.get("store_scope") == "selected" and
            status.get("direct_commits") == 0,
            f"{arm}: actual policy flag differs")
    count = installed.get("fit_first_resumes")
    require(type(count) is int and count >= 0 and
            count == status.get("fit_first_resumes") == metrics.get("fit_first_resumes") and
            (enabled or count == 0),
            f"{arm}: fit-first admission count differs")
    commands = (archive / "commands.txt").read_text()
    require(("--fit-first-resume" in commands) is enabled and
            "--commit-recheck" not in commands,
            f"{arm}: executed flag differs")
    return raw, installed, {
        "archive_map_sha256": common.sha_file(cell / "output_sha256.json"),
        "raw_sha256": hashes["raw.json"], "archive_file_count": len(hashes),
        "forced_rotations": status.get("forced_rotations"),
        "fit_first_resumes": count, "stop_reason_counts": status.get("finish_reason_counts"),
        "timing": common.read(archive / "timing.json"),
    }


def fit_first_actions(on_raw: dict, on_installed: dict, off_raw: dict) -> dict:
    on_internal = {r["internal_request_id"]: r for r in on_raw["requests"]}
    off_source = {r["request_id"]: r for r in off_raw["requests"]}
    require(len(on_internal) == len(off_source) == 128, "request identity duplicated")
    origin = on_raw["measurement_origin_perf_counter_s"]
    by_step_target: dict[tuple[int, str], dict] = {}
    pending: dict[str, dict] = {}
    actions = []
    counts = {"choice": 0, "admitted": 0, "target_new_output": 0,
              "target_terminal": 0}
    for event in on_installed.get("events", []):
        kind = event.get("event")
        if kind == "fit_first_choice":
            target = event["target"]
            require(target not in pending, "overlapping fit-first target choice")
            record = {"step": event["step"], "target_internal_id": target,
                      "original_target_internal_id": event["original_target"],
                      "planned_victim_internal_id": event["planned_victim"],
                      "free_blocks": event["free_blocks"],
                      "required_blocks": event["required_blocks"],
                      "original_target_required_blocks": event["original_target_required_blocks"],
                      "absence_steps": event["absence_steps"],
                      "original_absence_steps": event["original_absence_steps"],
                      "output_tokens_at_choice": event["output_tokens_at_choice"],
                      "choice_perf_counter_s": event["host_perf_counter_s"],
                      "admission": None, "outcome_event": None}
            by_step_target[(event["step"], target)] = record
            pending[target] = record
            actions.append(record)
            counts["choice"] += 1
        elif kind == "fit_first_admitted":
            key = (event["step"], event["target"])
            record = by_step_target.get(key)
            require(record is not None and record["admission"] is None,
                    "fit-first admission lacks unique same-step choice")
            record["admission"] = {
                "native_admission": event["native_admission"],
                "scheduled_tokens": event["scheduled_tokens"],
                "load_job_ids": event["load_job_ids"],
                "admission_perf_counter_s": event["host_perf_counter_s"],
                "choice_to_admission_ms": 1000 * (event["host_perf_counter_s"] -
                                                  record["choice_perf_counter_s"]),
            }
            counts["admitted"] += 1
        elif kind in ("target_new_output", "target_terminal"):
            record = pending.get(event.get("request"))
            if record is None:
                continue  # Ordinary rotation target, unrelated to fit-first.
            require(record["admission"] is not None and event["step"] > record["step"],
                    "fit-first outcome precedes admission")
            record["outcome_event"] = {key: event[key] for key in
                ("event", "step", "new_output_tokens", "status", "host_perf_counter_s")
                if key in event}
            counts[kind] += 1
            pending.pop(event["request"])
    require(counts["choice"] == counts["admitted"] == on_installed["fit_first_resumes"],
            "choice/admission/runner counts differ")
    for record in actions:
        chosen = on_internal[record["target_internal_id"]]
        start = record["output_tokens_at_choice"]
        require(type(start) is int and 0 <= start <= len(chosen["token_times_s"]),
                "choice output count exceeds final request")
        first_new = chosen["token_times_s"][start] if start < len(chosen["token_times_s"]) else None
        admit = record["admission"]["admission_perf_counter_s"] - origin
        record["first_new_output_after_choice_s"] = first_new
        record["admission_to_first_new_output_ms"] = (
            1000 * (first_new - admit) if first_new is not None and first_new >= admit else None)
        record["first_output_clock_order_valid"] = first_new is None or first_new >= admit
        for role, internal in (("chosen_target", record["target_internal_id"]),
                               ("original_target", record["original_target_internal_id"]),
                               ("planned_victim", record["planned_victim_internal_id"])):
            source = on_internal[internal]["request_id"]
            require(source in off_source, "actor source identity missing from off cell")
            record.setdefault("request_outcomes", {})[role] = {
                "off": request_result(off_source[source]),
                "on": request_result(on_internal[internal]),
            }
        record["target_request_id"] = chosen["request_id"]
        record.pop("choice_perf_counter_s")
        record["admission"].pop("admission_perf_counter_s")
    return {"event_counts": counts, "unresolved_outcome_event_count": len(pending),
            "latency_ms": {
                "choice_to_native_admission": distribution([
                    record["admission"]["choice_to_admission_ms"] for record in actions]),
                "native_admission_to_first_host_return_token": distribution([
                    record["admission_to_first_new_output_ms"] for record in actions
                    if record["admission_to_first_new_output_ms"] is not None]),
            },
            "actions": actions,
            "scope": "Choice and native admission are paired by step plus internal target ID. First later token uses the chosen request's output count at choice and host-return token clock. An outcome event can be absent if no later schedule call occurs; request completion still appears in raw. Off/on request outcomes are separate policy trajectories, not same-state action counterfactuals."}


def analyze(session: Path) -> dict:
    require(session.is_dir() and session.name == SESSION_NAME, "wrong fit-first session")
    plan, receipt = common.read(session / "plan.json"), common.read(session / "receipt.json")
    require(receipt.get("status") == "CELLS_COMPLETE" and
            receipt.get("plan_sha256") == common.sha_file(session / "plan.json") and
            tuple(cell.get("arm") for cell in plan.get("cells", [])) == ARMS and
            tuple(cell.get("arm") for cell in receipt.get("cells", [])) == ARMS,
            "fit-first pair plan or receipt incomplete")
    raws, installed, evidence = {}, {}, {}
    for index, arm in enumerate(ARMS):
        cell = receipt["cells"][index]
        require(cell.get("exit_code") == 0 and cell.get("timed_out") is False and
                cell.get("archive_status") == "VERIFIED", f"{arm}: controller cell failed")
        gate = arm.rsplit("_", 1)[1]
        raws[gate], installed[gate], evidence[gate] = load_cell(session, index, arm)
    metrics = {gate: summarize(raw, 128, 180) for gate, raw in raws.items()}
    require(metrics["off"]["completed"] == metrics["on"]["completed"] == 128,
            "pair lacks complete requests")
    action = fit_first_actions(raws["on"], installed["on"], raws["off"])
    require(not any(e.get("event") in ("fit_first_choice", "fit_first_admitted")
                    for e in installed["off"].get("events", [])),
            "off cell contains fit-first action")
    return {
        "schema_version": 1, "status": "FIT_FIRST_EXPLORATORY_PAIR_COMPLETE",
        "plan_sha256": common.sha_file(session / "plan.json"),
        "receipt_sha256": common.sha_file(session / "receipt.json"),
        "cells": evidence, "metrics": metrics,
        "request_comparison": pair(metrics["off"], metrics["on"]),
        "output_differences": common.output_differences(raws["off"], raws["on"]),
        "fit_first": action,
        "scope": "One exploratory pair on previously viewed H128. Request outcomes, service metrics and action latencies are descriptive. Separate trajectories do not identify same-state counterfactual benefit, statistical stability, or quality equivalence.",
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
                      "fit_first_actions": result["fit_first"]["event_counts"]["admitted"]},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
