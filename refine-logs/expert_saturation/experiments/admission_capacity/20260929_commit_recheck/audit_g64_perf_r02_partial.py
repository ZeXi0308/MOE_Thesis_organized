#!/usr/bin/env python3
"""Read-only audit of the archived T30/Q1 runtime-limit continuation; no selection."""
from __future__ import annotations

import argparse
from collections import Counter
import ast
import json
import math
from pathlib import Path
import shlex

import audit_g64_perf_three_arm as base
from evaluate_goodput import pair, summarize

PLANNED_ARMS = ("ltr_t30_q1", "ltr_t200_q1", "ltr_t200_q10")
COMMON_CONFIG = ("model", "requests", "workload_sha256", "seed", "output_tokens",
                 "output_mode", "ignore_eos", "min_tokens", "cap", "engine_max_num_seqs",
                 "max_num_batched_tokens", "max_seconds", "fixed_kv_cache_memory_bytes",
                 "offload_gib", "measurement_mode")


def liveness_signature(installed: dict, raw: dict) -> dict:
    """Summarize the archived scheduler trace without assigning a native cause."""
    events = installed.get("events")
    calls = installed.get("schedule_calls")
    base.require(isinstance(events, list) and events and
                 type(calls) is int and calls == raw.get("engine_call_count") and
                 calls == raw.get("engine_return_count") and
                 all(type(e.get("step")) is int and e["step"] >= 0 and
                     isinstance(e.get("event"), str) for e in events) and
                 all(a["step"] <= b["step"] for a, b in zip(events, events[1:])),
                 "r02 scheduler event clock/call accounting differs")
    output_events = raw.get("output_events")
    base.require(isinstance(output_events, list) and
                 all(type(e.get("received_s")) in (int, float) and
                     math.isfinite(e["received_s"]) and
                     0 <= e["received_s"] <= raw["observation_end_s"] for e in output_events) and
                 sum(len(e.get("new_token_ids", ())) for e in output_events) ==
                 sum(len(r["output_token_ids"]) for r in raw["requests"]),
                 "r02 output event accounting differs")
    last_output = max((e["received_s"] for e in output_events), default=None)
    target = events[-1].get("target") if events[-1]["event"] == "release" else None
    suffix_steps = 0
    first_step = None
    first_counter = None
    last_counter = None
    index = len(events) - 1
    expected_step = events[-1]["step"]
    while target is not None and index >= 0:
        step = events[index]["step"]
        if step != expected_step:
            break
        group = []
        while index >= 0 and events[index]["step"] == step:
            group.append(events[index])
            index -= 1
        accepted = [e for e in group if e["event"] == "accept" and
                    e.get("action") == "PRIORITIZE_WAITING" and
                    e.get("target_id") == target]
        censored = [e for e in group if e["event"] == "backend_censored" and
                    e.get("target") == target and e.get("reason") == "PREEMPTED"]
        released = [e for e in group if e["event"] == "release" and
                    e.get("target") == target and
                    e.get("reason") == "native admission censored; rescan without resetting counters"]
        if (len(accepted), len(censored), len(released)) != (1, 1, 1):
            break
        suffix_steps += 1
        first_step = step
        first_counter = accepted[0].get("host_perf_counter_s")
        if last_counter is None:
            last_counter = released[0].get("host_perf_counter_s")
        expected_step -= 1
    source_id = raw.get("internal_to_source", {}).get(target)
    return {
        "schedule_calls": calls,
        "engine_call_count": raw["engine_call_count"],
        "engine_return_count": raw["engine_return_count"],
        "selective_store_final_status": installed.get("status"),
        "applied_rotations": installed.get("applied_rotations"),
        "event_count": len(events),
        "event_type_counts": dict(Counter(e["event"] for e in events)),
        "output_event_count": len(output_events),
        "last_output_received_s": last_output,
        "seconds_from_last_output_to_capture_end":
            raw["observation_end_s"] - last_output if last_output is not None else None,
        "terminal_repeated_triad": {
            "target_internal_id": target,
            "target_source_request_id": source_id,
            "first_step": first_step,
            "last_step": events[-1]["step"] if suffix_steps else None,
            "consecutive_steps": suffix_steps,
            "accept_action": "PRIORITIZE_WAITING",
            "backend_censored_reason": "PREEMPTED",
            "release_reason": "native admission censored; rescan without resetting counters",
            "first_accept_perf_counter_s": first_counter,
            "last_release_perf_counter_s": last_counter,
            "elapsed_perf_counter_s": last_counter - first_counter
            if first_counter is not None and last_counter is not None else None,
        },
        "causal_limit": "The archived adapter events show repeated native non-admission and reselection. They do not log the native branch or resource check that caused each rejection.",
    }


def source_localization(package: Path, manifest: dict, env: dict) -> dict:
    rotation = package / "pkg/rotation_native.py"
    adapter = package / "pkg/ltr_style_native.py"
    tree = ast.parse(rotation.read_text())
    assignments = [n for n in tree.body if isinstance(n, ast.Assign) and
                   any(isinstance(t, ast.Name) and t.id == "SCHEDULER_SHA256"
                       for t in n.targets)]
    base.require(len(assignments) == 1 and isinstance(assignments[0].value, ast.Constant) and
                 assignments[0].value.value ==
                 base.PINNED_RUNTIME_SHA["v1/core/sched/scheduler.py"] and
                 base.sha_file(rotation) == manifest["pkg/rotation_native.py"] ==
                 env["source_sha256"]["rotation_native.py"] and
                 base.sha_file(adapter) == manifest["pkg/ltr_style_native.py"] ==
                 env["source_sha256"]["ltr_style_native.py"],
                 "r02 pinned scheduler/adapter source locality differs")
    return {
        "native_scheduler_sha256": assignments[0].value.value,
        "rotation_native_sha256": base.sha_file(rotation),
        "ltr_style_native_sha256": base.sha_file(adapter),
        "observed_code_path": [
            "rotation_native.py patches the native waiting loop to break when the peeked request differs from _rotation_target.",
            "ltr_style_native.py sets _rotation_target for an accepted recovery target and releases/reselects a PREEMPTED target after zero scheduled tokens.",
        ],
        "inference_limit": "This source path is consistent with the archived liveness trace; the archive does not identify the exact native rejection branch.",
    }


def audit(pilot_session: Path, pilot_audit_path: Path, pilot_plan_sha: str,
          pilot_audit_sha: str, r02_session: Path, r02_plan_sha: str) -> dict:
    base.require(pilot_session.is_dir() and r02_session.is_dir() and
                 not pilot_session.is_symlink() and not r02_session.is_symlink() and
                 pilot_session.resolve() != r02_session.resolve(),
                 "two separate ordinary session directories required")
    base.require(base.SHA_RE.fullmatch(pilot_audit_sha) is not None and
                 base.sha_file(pilot_audit_path) == pilot_audit_sha,
                 "pilot V2 audit differs from external SHA-256")
    old = base.read(pilot_audit_path)
    base.require(old == base.audit(pilot_session, pilot_plan_sha) and
                 old.get("status") == "PILOT_THREE_ARM_COMPLETE_PARTIAL_TQ_GRID",
                 "pilot audit does not recompute from immutable archive")
    pplan, prec = base.read(pilot_session / "plan.json"), base.read(pilot_session / "receipt.json")
    rplan_path = r02_session / "plan.json"
    base.require(base.SHA_RE.fullmatch(r02_plan_sha) is not None and
                 base.sha_file(rplan_path) == r02_plan_sha,
                 "r02 plan differs from external SHA-256")
    plan, receipt = base.read(rplan_path), base.read(r02_session / "receipt.json")
    base.require(tuple(c.get("arm") for c in plan.get("cells", [])) == PLANNED_ARMS and
                 receipt.get("plan_sha256") == r02_plan_sha and
                 receipt.get("status") == "ABORTED" and
                 receipt.get("held_lock_device_inode") == plan.get("expected_lock_device_inode"),
                 "r02 planned grid or aborted receipt differs")
    same = ("authorized_gpu_uuid", "approved_host_bytes", "lock_path",
            "expected_lock_device_inode", "python", "hf_cache_dir",
            "shared_model_source_cache", "model_revision", "model_verifier_sha256",
            "cgroup_memory_max_file")
    base.require(all(plan.get(k) == pplan.get(k) for k in same) and
                 receipt.get("started_unix_s", 0) > prec.get("finished_unix_s", float("inf")),
                 "r02 machine/model/lock identity or session order differs")
    base.model_check(r02_session, plan, receipt)
    observed = receipt.get("cells")
    base.require(isinstance(observed, list) and len(observed) == 1,
                 "runtime-limit continuation must stop after its first cell")
    planned, cell = plan["cells"][0], observed[0]
    arm = PLANNED_ARMS[0]
    base.require(planned.get("package_dir") == pplan["cells"][1]["package_dir"] and
                 planned.get("output_dir") not in {c["output_dir"] for c in pplan["cells"]} and
                 cell.get("arm") == arm and cell.get("output_dir") == planned.get("output_dir") and
                 cell.get("argv") == [planned["package_dir"] + "/pkg/run.sh",
                                      arm, planned["output_dir"]] and
                 cell.get("launch_status") == "FINISHED" and
                 cell.get("exit_code") != 0 and cell.get("timed_out") is False and
                 cell.get("archive_status") == "VERIFIED" and
                 cell.get("gpu_process_state_after") == "EMPTY" and
                 cell.get("gpu_process_rows_after") == [],
                 "r02 first-cell controller/archive receipt differs")
    cell_dir = r02_session / "cell-00-ltr_t30_q1"
    archive = cell_dir / "archive"
    hashes = base.read(cell_dir / "output_sha256.json")
    base.require(isinstance(hashes, dict) and base.REQUIRED_OUTPUTS <= set(hashes) and
                 all(isinstance(v, str) and base.SHA_RE.fullmatch(v) for v in hashes.values()) and
                 base.archive_hashes(archive) == hashes,
                 "r02 archive file set or SHA-256 readback differs")
    package = Path(__file__).parent / "candidate_g64_perf_r01"
    base.require(base.sha_file(package / "manifest.json") == base.PACKAGE_SHA,
                 "local frozen package manifest differs")
    manifest = base.read(package / "manifest.json")
    workload = base.read(package / "pkg/inputs/workload.json")
    source, prompts, arrivals = (workload["source_requests"],
                                 workload["actual_prompt_token_ids"],
                                 workload["arrival_traces_s"]["steady"])
    base.require(len(source) == len(prompts) == len(arrivals) == 64,
                 "G64 workload shape differs")
    cohort = {s["request_id"]: (s["document_id"], s["prompt_token_ids_sha256"],
              arrivals[i], len(prompts[i]), 1024) for i, s in enumerate(source)}
    base.require(len(cohort) == 64, "G64 source IDs duplicate")
    config, engine, env = (base.read(archive / name) for name in
                           ("config.json", "engine_args.json", "environment.json"))
    eager_archive = pilot_session / "cell-01-eager/archive"
    eager_config = base.read(eager_archive / "config.json")
    eager_engine = base.read(eager_archive / "engine_args.json")
    eager_env = base.read(eager_archive / "environment.json")
    base.require(all(config.get(k) == eager_config.get(k) for k in COMMON_CONFIG) and
                 config.get("variant") == "ltr_style_selected" and
                 config.get("store_scope") == "selected" and
                 config.get("ltr_config") == {"threshold": 30, "quantum": 1},
                 "r02 workload/policy config differs")
    base.require(engine == eager_engine and
                 all(env.get(k) == eager_env.get(k) for k in
                     ("python", "torch", "cuda", "vllm", "transformers", "vllm_source_sha256")) and
                 env.get("vllm_source_sha256") == base.PINNED_RUNTIME_SHA and
                 env.get("source_sha256") ==
                 {name: manifest[f"pkg/{name}"] for name in base.ARM_SOURCES["ltr_t30_q10"]},
                 "r02 engine or pinned source drift")
    before = env.get("gpu_before", {})
    base.require(plan["authorized_gpu_uuid"] in before.get("device", "") and
                 not before.get("compute_processes"), "r02 GPU isolation differs")
    cap, memory = base.read(archive / "safe-cap-qualification.json"), base.read(archive / "memory-after-init.json")
    base.require(cap.get("status") == "QUALIFIED" and cap.get("usable_blocks") == 4096 and
                 cap.get("total_blocks") == 4097 and cap.get("block_size") == 16 and
                 cap.get("engine_max_num_seqs") == 32 and
                 cap.get("observed_scheduler_reserve_full_isl") is True and
                 memory.get("kv_storage_bytes") == base.KV_BYTES,
                 "r02 physical GPU KV allocation differs")
    for name in ("host-after-init.json", "host-before.json"):
        host = base.read(archive / name)
        limit = host.get("parent_cgroup", {}).get("values", {}).get("memory.max")
        base.require(host.get("cpu_kv", {}).get("unique_storage_bytes") == base.HOST_BYTES and
                     host.get("manager", {}).get("capacity_blocks") == 8192 and
                     type(limit) is int and 0 < limit <= plan["approved_host_bytes"] and
                     limit == base.read(eager_archive / "host-after-init.json")
                                   ["parent_cgroup"]["values"]["memory.max"],
                     f"r02 host KV/cgroup drift in {name}")
    for i, count in enumerate((32, 32, 2)):
        warm = base.read(archive / f"warmup-{i}.json")
        base.require(warm.get("status") == "COMPLETE" and
                     len(warm.get("requests", [])) == count and
                     all(r.get("status") == "completed" for r in warm["requests"]),
                     f"r02 warmup {i} incomplete")
    base.require(base.read(archive / "warmup-cache-reset.json").get("success") is True,
                 "r02 warmup cache reset failed")
    for name in ("warmup-offload-drain.json", "post-request-drain.json"):
        drain = base.read(archive / name)
        base.require(type(drain.get("calls")) is int and drain["calls"] >= 0 and
                     type(drain.get("seconds")) in (float, int) and
                     math.isfinite(drain["seconds"]) and drain["seconds"] >= 0,
                     f"r02 {name} invalid")
    installed = base.read(archive / "selective-store.json")
    base.require(installed.get("store_scope") == "selected" and
                 installed.get("native_calc_overridden") is True and
                 installed.get("ltr_config") == {"threshold": 30, "quantum": 1},
                 "r02 installed LTR point differs")
    commands = shlex.split((archive / "commands.txt").read_text())
    base.require("--ltr-threshold" in commands and
                 commands[commands.index("--ltr-threshold") + 1] == "30" and
                 "--ltr-quantum" in commands and
                 commands[commands.index("--ltr-quantum") + 1] == "1" and
                 "--measurement-mode" in commands and
                 commands[commands.index("--measurement-mode") + 1] == "performance",
                 "r02 executed command differs")
    raw, status = base.read(archive / "raw.json"), base.read(archive / "status.json")
    base.require(raw.get("status") == "INCOMPLETE" and raw.get("error") == "runtime_limit" and
                 status.get("status") == status.get("capture_status") == "INCOMPLETE" and
                 "runtime_limit" in str(status.get("error")),
                 "r02 failure is not the archived runtime limit")
    rows = raw.get("requests")
    base.require(isinstance(rows, list) and len(rows) == 64 and
                 raw.get("regime") == "steady" and raw.get("arrival_scale") == 1.0 and
                 raw.get("observation_end_s", 0) >= 180,
                 "r02 capture clock or cohort size differs")
    seen = {r.get("request_id"): (r.get("document_id"), r.get("prompt_token_ids_sha256"),
            r.get("arrival_s"), r.get("prompt_tokens"), r.get("max_output_tokens")) for r in rows}
    base.require(len(seen) == 64 and seen == cohort and
                 all(r.get("status") in ("completed", "failed", "unfinished") and
                     r.get("arrived_at_observation_end") is True for r in rows),
                 "r02 capture cohort or status differs")
    completed = sum(r["status"] == "completed" for r in rows)
    base.require(completed < 64 and status.get("requests_completed") == completed and
                 status.get("finish_reason_counts") == dict(Counter(
                     str(r.get("finish_reason", r.get("stop_reason")) or "unfinished") for r in rows)),
                 "r02 completion/finish receipt differs")
    partial = summarize(raw, 64, 180)
    eager = old["metrics"]["eager"]
    liveness = liveness_signature(installed, raw)
    source = source_localization(package, manifest, env)
    return {
        "schema_version": 2, "status": "R02_FIRST_CELL_RUNTIME_LIMIT_ARCHIVED",
        "calibration_status": "INCOMPLETE_GRID_NO_SELECTION", "selected_ltr_arm": None,
        "pilot_plan_sha256": pilot_plan_sha, "pilot_audit_sha256": pilot_audit_sha,
        "r02_plan_sha256": r02_plan_sha,
        "r02_receipt_sha256": base.sha_file(r02_session / "receipt.json"),
        "archive_sha256_map_sha256": base.sha_file(cell_dir / "output_sha256.json"),
        "archive_file_count": len(hashes), "raw_sha256": base.sha_file(archive / "raw.json"),
        "model_receipts_r02": base.model_check(r02_session, plan, receipt),
        "cross_session": {"pilot_finished_unix_s": prec["finished_unix_s"],
                          "r02_started_unix_s": receipt["started_unix_s"],
                          "uncontrolled_drift_possible": True},
        "t30_q1_partial": partial, "paired_vs_pilot_eager": pair(eager, partial),
        "request_status_counts": dict(Counter(r["status"] for r in rows)),
        "stop_reason_counts": dict(Counter(r.get("stop_reason") or "unfinished" for r in rows)),
        "liveness_signature": liveness,
        "source_localization": source,
        "capture_error": raw["error"],
        "metric_semantics": "All 64 planned arrivals remain in the cohort. Failed/unfinished requests receive the evaluate_goodput 180 s flow penalty and cannot qualify a calibration point. Partial output sequences are truncated, not equal work.",
        "scope": "Cross-session G64 development diagnostic; T200/Q1 and T200/Q10 unrun, no LTR selection, no statistical or H128 claim.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-session", type=Path, required=True)
    parser.add_argument("--pilot-audit", type=Path, required=True)
    parser.add_argument("--expected-pilot-plan-sha256", required=True)
    parser.add_argument("--expected-pilot-audit-sha256", required=True)
    parser.add_argument("--r02-session", type=Path, required=True)
    parser.add_argument("--expected-r02-plan-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pilot_session = args.pilot_session.resolve(strict=True)
    r02_session = args.r02_session.resolve(strict=True)
    output = args.output.resolve(strict=False)
    base.require(not output.exists() and all(s != output and s not in output.parents
                 for s in (pilot_session, r02_session)),
                 "new analysis JSON must be outside both immutable sessions")
    result = audit(pilot_session, args.pilot_audit, args.expected_pilot_plan_sha256,
                   args.expected_pilot_audit_sha256, r02_session, args.expected_r02_plan_sha256)
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "output": str(output)}))


if __name__ == "__main__":
    main()
