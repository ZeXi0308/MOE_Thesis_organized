#!/usr/bin/env python3
"""Qualify actual recovery-lease actions; never estimate policy benefit.

Performance raw logs independently capture host outputs and actual preemptions.
The post-native lease receipts capture native scheduled-token/load-job plans;
they are not a second independent full scheduler trace. Step indices in these
receipts equal raw engine_call_index (both start at zero after policy install).
"""
import argparse
from bisect import bisect_left
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


def analyze(raw, store, config, status):
    checks = {}
    failures = []
    def check(name, condition, context=None):
        passed = bool(condition)
        checks[name] = checks.get(name, True) and passed
        if not passed:
            failures.append(dict(check=name, context=context))
        return passed

    requests = {r["request_id"]: r for r in raw["requests"]}
    mapping = raw["internal_to_source"]
    check("unique_source_requests", len(requests) == len(raw["requests"]))
    check("internal_source_identity", all(r.get("internal_request_id") == i
          for i, s in mapping.items() for r in [requests[s]]))
    mode = store.get("recovery_lease_mode")
    check("enabled_lease_mode_matches", mode in ("q1", "fixed4", "adaptive")
          and mode == config.get("recovery_lease_mode"))
    check("run_complete_and_adapter_drained", raw.get("status") == "COMPLETE"
          and raw.get("error") is None and status.get("status") == "COMPLETE"
          and status.get("error") is None and store.get("status") == "DRAINED")
    check("native_full_saving", store.get("store_scope") == "native_full"
          and store.get("native_calc_overridden") is False)
    by_source = defaultdict(list)
    by_call = defaultdict(list)
    for event in raw["output_events"]:
        by_source[event["request_id"]].append(event)
        by_call[(event["engine_call_index"], event["request_id"])].append(event)
    for events in by_source.values():
        events.sort(key=lambda e: e["engine_call_index"])
    calls = {s: [e["engine_call_index"] for e in es] for s, es in by_source.items()}
    actual_preempts = [p for p in raw["preemption_events"] if p.get("original_preemption_returned")]
    preempts_by_key = defaultdict(list)
    for p in actual_preempts:
        preempts_by_key[(p["engine_call_index"], p["internal_request_id"])].append(p)
    check("raw_call_count_matches_policy_steps", raw["engine_call_count"] == store["schedule_calls"])

    def prior_count(source, step):
        pos = bisect_left(calls.get(source, []), step)
        return by_source[source][pos - 1]["cumulative_tokens"] if pos else 0

    def outcome(internal, step):
        source = mapping[internal]
        previous = prior_count(source, step)
        later = next((e for e in by_source[source] if e["engine_call_index"] >= step
                      and e["cumulative_tokens"] > previous), None)
        request = requests[source]
        return dict(source_request_id=source, output_count_before_call=previous,
                    first_new_output_call=later["engine_call_index"] if later else None,
                    first_new_output_s=later["received_s"] if later else None,
                    later_new_output=later is not None, completed=request.get("status") == "completed",
                    completion_s=request.get("completion_s"))

    events = store["events"]
    starts = [e for e in events if e.get("event") == "protection_start"]
    releases = [e for e in events if e.get("event") == "protection_release"]
    commits = {e["episode_id"]: e for e in events if e.get("event") == "lease_commit_decision"}
    actual_commits = {e["episode_id"]: e for e in events if e.get("event") == "oldest_commit"}
    execution = store.get("lease_execution", [])
    check("one_commit_decision_per_lease", len(commits) == len(starts)
          == sum(e.get("event") == "lease_commit_decision" for e in events))
    check("one_release_per_lease", len(releases) == len(starts))
    check("unique_execution_target_steps", len({(e["target"], e["step"]) for e in execution}) == len(execution))
    episode_rows = []
    assigned_execution = set()
    for index, start in enumerate(starts):
        eid, target = start["oldest_episode_id"], start["request"]
        source = mapping[target]
        matches = [e for e in releases if e.get("oldest_episode_id") == eid and e.get("request") == target]
        if not check("release_identity_unique", len(matches) == 1, eid):
            continue
        release = matches[0]
        check("episode_step_order", start["step"] < release["step"], eid)
        if index + 1 < len(starts):
            check("no_overlapping_leases", release["step"] <= starts[index + 1]["step"], eid)
        decision = commits.get(eid)
        check("commit_matches_protection_start", decision is not None
              and decision["target"] == target and decision["step"] == start["step"]
              and decision["lease_decision"] == start["lease_decision"], eid)
        q = start["lease_decision"]["quantum"]
        begin_count = start["output_count"]
        check("quantum_respects_known_output_cap", 1 <= q <= config["recovery_lease_config"]["max_quantum"]
              and start["output_goal"] == min(begin_count + q, requests[source]["max_output_tokens"]), eid)
        check("start_output_matches_raw", begin_count == prior_count(source, start["step"]), eid)
        final_count = prior_count(source, release["step"])
        gained = final_count - begin_count
        check("release_output_matches_raw", release["output_count"] == final_count
              and release["new_output_tokens"] == gained, eid)
        check("released_within_quantum", 0 <= gained <= q, eid)
        reason = release["reason"]
        check("release_reason_consistent", (reason == "OUTPUT_GOAL_REACHED" and final_count >= start["output_goal"])
              or (reason.startswith("TERMINAL_") and requests[source]["status"] == "completed")
              or (reason.startswith("RELEASE_") and 1 <= gained < q), eid)
        active = [(i, e) for i, e in enumerate(execution) if e["target"] == target
                  and start["step"] <= e["step"] < release["step"]]
        check("every_protected_call_has_execution_receipt", {e["step"] for _, e in active}
              == set(range(start["step"], release["step"])), eid)
        assigned_execution.update(i for i, _ in active)
        same_call_outputs = 0
        for _, e in active:
            key = dict(episode_id=eid, step=e["step"])
            n = e["scheduled_tokens"]
            check("execution_capacity_reserved", e["free_blocks"] >= e["growth_reserved_blocks"]
                  + e["other_native_reserved_blocks"] >= 0, key)
            required = (requests[source]["prompt_tokens"] + start["output_goal"] - 1 + 15) // 16
            check("execution_growth_matches_quantum_endpoint", e["output_goal"] == start["output_goal"]
                  and e["growth_reserved_blocks"] == max(0, required - e["held_blocks"]), key)
            check("execution_current_output_matches_raw", e["output_count"] == prior_count(source, e["step"]), key)
            check("ready_target_scheduled_pending_target_not_executed",
                  (e["status"] == "RUNNING" and n > 0)
                  or (e["status"] == "WAITING_FOR_REMOTE_KVS" and n == 0), key)
            check("protected_target_not_actually_preempted", not preempts_by_key[(e["step"], target)], key)
            native_commit = actual_commits.get(eid)
            allowed_victim = (native_commit["victim"] if native_commit and native_commit["forced"]
                              and native_commit["step"] == e["step"] else None)
            check("no_unplanned_actual_preemption_during_lease", all(
                p["internal_request_id"] == allowed_victim for p in actual_preempts
                if p["engine_call_index"] == e["step"]), key)
            emitted = sum(x["chunk_size"] for x in by_call[(e["step"], source)])
            check("raw_new_output_requires_native_scheduled_tokens", emitted == 0 or n > 0, key)
            same_call_outputs += emitted
        check("protected_outputs_join_execution_calls", same_call_outputs == gained, eid)
        victim = None
        commit = actual_commits.get(eid)
        check("native_commit_present", commit is not None and commit["target"] == target, eid)
        if commit and commit["forced"]:
            actual = preempts_by_key[(commit["step"], commit["victim"])]
            check("forced_victim_preemption_matches_raw", len(actual) == 1, eid)
            victim = outcome(commit["victim"], commit["step"])
            check("victim_later_output_and_completion", victim["later_new_output"] and victim["completed"], eid)
        target_outcome = outcome(target, start["step"])
        check("target_later_output_or_terminal_and_completion", target_outcome["completed"]
              and (target_outcome["later_new_output"] or reason.startswith("TERMINAL_")), eid)
        episode_rows.append(dict(episode_id=eid, target_internal_id=target, source_request_id=source,
            start_step=start["step"], release_step=release["step"], chosen_quantum=q,
            observed_outputs_before_release=gained, release_reason=reason, execution_calls=len(active),
            native_pending_calls=sum(e["status"] == "WAITING_FOR_REMOTE_KVS" for _, e in active),
            target=target_outcome, victim=victim))
    check("all_execution_receipts_belong_to_one_lease", len(assigned_execution) == len(execution))
    check("forced_commit_count_matches_adapter", sum(bool(e["forced"]) for e in actual_commits.values())
          == store.get("applied_rotations") == store.get("oldest_forced_commits"))
    peer_rows = []
    for e in store.get("lease_peer_admissions", []):
        context = dict(step=e["step"], request=e["request"])
        check("peer_capacity_gate_consistent", e["free_blocks"] >= e["peer_need_blocks"]
              + e["lease_growth_blocks"] + e["other_native_reserved_blocks"], context)
        kind, n, jobs = e["native_admission"], e["scheduled_tokens"], e["load_job_ids"]
        check("peer_native_plan_consistent", (kind == "SCHEDULED_TOKENS" and n > 0)
              or (kind == "ASYNC_LOAD_ADMITTED" and n == 0 and len(jobs) > 0)
              or (kind == "NO_NEW_NATIVE_ADMISSION" and n == 0 and not jobs), context)
        peer = outcome(e["request"], e["step"])
        actual = kind in ("SCHEDULED_TOKENS", "ASYNC_LOAD_ADMITTED")
        if actual:
            check("admitted_peers_later_output_and_completion", peer["later_new_output"] and peer["completed"], context)
            check("admitted_peer_not_same_call_preempted", not preempts_by_key[(e["step"], e["request"])], context)
        peer_rows.append(dict(**context, native_admission=kind, scheduled_tokens=n, load_job_ids=jobs,
                              actual_admission=actual, **peer))
    samples = [e for e in events if e.get("event") == "lease_service_sample"]
    for e in samples:
        check("service_sample_accounting", abs(e["episode_overhead_s"]
              - max(0., e["first_output_host_interval_s"] - e["prior_decode_interval_s"])) < 1e-8,
              dict(step=e["step"], request=e["request"]))
    all_passed = all(checks.values())
    return dict(status="NO_LEASE_ACTION" if not starts and all_passed else "QUALIFIED_ACTIONS" if all_passed else "FAILED_QUALIFICATION",
        mode=mode, qualification=dict(all_checks_passed=all_passed, checks=checks, failures=failures),
        descriptive=dict(lease_episodes=len(starts), unique_targets=len({e["request"] for e in starts}),
            chosen_quantum_counts=dict(Counter(r["chosen_quantum"] for r in episode_rows)),
            actual_output_counts_before_release=dict(Counter(r["observed_outputs_before_release"] for r in episode_rows)),
            release_reason_counts=dict(Counter(r["release_reason"] for r in episode_rows)),
            native_execution_receipts=len(execution), peer_gate_pass_receipts=len(peer_rows),
            peer_actual_admissions=sum(r["actual_admission"] for r in peer_rows),
            peer_nonadmission_receipts=sum(not r["actual_admission"] for r in peer_rows),
            service_samples=len(samples), decision_cpu_s=store.get("lease_decision_cpu_s")),
        episodes=episode_rows, peer_receipts=peer_rows, service_samples=samples,
        evidence_provenance=dict(planned_tokens="Post-native SchedulerOutput fields recorded by lease_execution / lease_peer_admissions",
            independent_outcomes="raw.output_events and raw.preemption_events; joined using internal_to_source and zero-based engine_call_index",
            independent_full_native_plan_trace="NOT_CAPTURED_IN_PERFORMANCE_MODE"),
        limitations=["Action qualification does not establish performance benefit or independent repeats.",
            "Scheduled tokens are native plan receipts, not a separate device kernel-execution trace; raw outputs establish actual service.",
            "Partial prefill/recompute can schedule tokens without a new output in that call.",
            "Host episode overhead is not additive DMA cost and must not be summed with overlapping transfer times."])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; refusing overwrite")
    files = {name: args.archive / (name + ".json") for name in ("raw", "selective-store", "config", "status")}
    payloads = {name: json.loads(path.read_text()) for name, path in files.items()}
    result = analyze(payloads["raw"], payloads["selective-store"], payloads["config"], payloads["status"])
    result["archive"] = str(args.archive)
    result["source_sha256"] = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in files.items()}
    with args.output.open("x") as out:
        json.dump(result, out, indent=2, ensure_ascii=False, allow_nan=False)
        out.write("\n")
    print(json.dumps(dict(status=result["status"], qualification=result["qualification"], descriptive=result["descriptive"]), indent=2))
    return 0 if result["qualification"]["all_checks_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
