#!/usr/bin/env python3
"""CPU-only analysis of the frozen HL/LH pulse; retain failed and unmatched runs."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT.parents[1]), str(ROOT.parent / "mixed_arrivals")]
import analyze as native
from analyze_fixed import METRICS, percent, signature


def sha(value):
    return hashlib.sha256(value).hexdigest()


def pulse_report(raw, design):
    control = raw["pulse"]
    steps = raw["steps"]
    common, half = design["common_cap"], design["half_steps"]
    low, high = design["legal_caps"]
    errors, window, caps = [], [], Counter()
    first_hit = next((i for i, s in enumerate(steps) if s["pulse_decision"]["guard"]), None)
    formal = raw["policy"].startswith("pulse_")
    trigger = control["trigger_step"]
    for i, s in enumerate(steps):
        d = s["pulse_decision"]
        offset = d["pulse_offset"]
        guard = d["decode_count"] >= design["trigger"]["decode_count_at_least"] and d["prefill_backlog_tokens"] >= design["trigger"]["prefill_backlog_at_least"]
        if (d["step_index"] != i or d["requested_cap"] != s["budget"] or d["guard"] != guard
                or d["actual_prefill_tokens"] != s["prefill_tokens"] or d["actual_decode_tokens"] != s["decode_tokens"]
                or d["decode_count"] != s["decode_count_before"] or d["decode_context_sum"] != s["decode_context_sum"]
                or d["prefill_backlog_tokens"] != s["prefill_backlog_tokens_before"]):
            errors.append(f"Decision/native record mismatch at step {i}")
        completed = s["end_s"] is not None and d.get("forward_completed") is True and d.get("native_schedule_completed") is True
        if not completed:
            errors.append(f"Forward not completed at step {i}")
        if s["prefill_tokens"] > 0 and completed:
            caps[str(s["budget"])] += 1
        in_window = formal and trigger is not None and trigger <= i < trigger + design["action_steps"]
        expected_offset = i - trigger if in_window else None
        expected_cap = common if formal else int(raw["policy"].removeprefix("fixed"))
        if in_window:
            high_half = (expected_offset < half) == (raw["policy"] == "pulse_hl")
            expected_cap = high if high_half else low
        if offset != expected_offset or s["budget"] != expected_cap:
            errors.append(f"Frozen cap/offset sequence mismatch at step {i}")
        if offset is not None:
            window.append(dict(step_index=i, pulse_offset=offset, requested_cap=s["budget"],
                prefill_tokens=s["prefill_tokens"], decode_tokens=s["decode_tokens"],
                decode_count_before=d["decode_count"], decode_context_sum=d["decode_context_sum"],
                backlog_before=d["prefill_backlog_tokens"], decision_time_s=d["decision_time_s"],
                end_s=s["end_s"], forward_completed=completed,
                completed_prefill_ids=s["completed_prefill_ids"], unused_budget_reason=d["unused_budget_reason"]))
    if formal and trigger != first_hit:
        errors.append("Trigger is not the first current-state guard hit")
    if control["pending_decision"] is not None:
        errors.append("Controller has a pending/failed decision")
    if control.get("pending_forward_decision") is not None:
        errors.append("Controller has an unconfirmed forward")
    executed = [s for s in window if s["forward_completed"]]
    underfills = [s["step_index"] for s in window if s["prefill_tokens"] < s["requested_cap"]]
    actual_p = sum(s["prefill_tokens"] for s in executed)
    actual_d = sum(s["decode_tokens"] for s in executed)
    for key, value in (("executed_pulse_steps", len(executed)), ("executed_prefill_tokens", actual_p), ("executed_decode_tokens", actual_d)):
        if control[key] != value:
            errors.append("Controller completed-forward aggregate mismatch: "+key)
    counts = dict(Counter(str(s["requested_cap"]) for s in window))
    dose = (formal and control["status"] == "COMPLETE" and len(executed) == design["action_steps"]
            and counts == {str(low): half, str(high): half} and not underfills
            and actual_p == design["nominal_prefill_tokens_in_window"])
    point = None
    if trigger is not None and 0 <= trigger < len(steps):
        point = dict(steps[trigger]["pulse_decision"])
    start = point["decision_time_s"] if point else None
    end = executed[-1]["end_s"] if executed else None
    inside = lambda t: t is not None and start is not None and end is not None and start <= t <= end
    snapshot = control.get("trigger_snapshot")
    failed = sorted(q["request_id"] for q in (snapshot or {}).get("requests", [])
                    if any(q[k] for k in ("ttft_failed_observed", "gap_failed_observed", "completion_failed_observed")))
    return dict(controller_status=control["status"], trigger_step=trigger, trigger=point,
        trigger_snapshot=snapshot, observed_irreversible_failure_ids=failed,
        pulse_steps_recorded=len(window), pulse_steps_forward_completed=len(executed),
        requested_cap_counts=counts, actual_prefill_tokens=actual_p, actual_decode_tokens=actual_d,
        actual_up_steps=sum(s["prefill_tokens"] > common for s in executed),
        underfilled_step_indices=underfills, actual_prefill_deficit_tokens=design["nominal_prefill_tokens_in_window"]-actual_p,
        actual_dose_matches=dose, window_start_s=start, window_end_s=end,
        completed_prefill_ids=[rid for s in executed for rid in s["completed_prefill_ids"]],
        completed_request_ids=[q["request_id"] for q in raw["requests"] if inside(q["completion_s"])],
        added_request_ids=[q["request_id"] for q in raw["requests"] if inside(q["add_s"])],
        emitted_output_tokens=sum(inside(t) for q in raw["requests"] for t in q["token_times_s"]),
        positive_prefill_caps=dict(caps), steps=window, errors=errors,
        controller_summary={k: v for k, v in control.items() if k != "trigger_snapshot"})


def load_run(path, design, expected_signature, design_hash):
    out = dict(cell=path.parent.name, raw_path=str(path), valid=False, errors=[])
    try:
        data = path.read_bytes(); raw = json.loads(data)
        out.update(policy=raw["policy"], _raw=raw, raw_sha256=sha(data),
                   observed_request_count=len(raw["requests"]),
                   observed_finished_count=sum(q["finished"] for q in raw["requests"]))
        try:
            report = native.summarize(path, native.find_protocol(path, None), path.parent.name)
            out.update(summary=report["summary"], requests=report["requests"], request_statistics=report["request_statistics"],
                       failed_by_slo_field=report["failed_by_slo_field"], warnings=report["warnings"])
            out["errors"].extend(report["validation_errors"])
            if report["slo"] != design["primary_slo"]:
                out["errors"].append("Primary SLO differs from frozen design")
        except Exception as exc:
            out["errors"].append("Native summarize: " + repr(exc))
        if raw["policy"] not in design["policies"] + design["warm_policies"] or path.parent.name.split("_", 1)[1] != raw["policy"]:
            out["errors"].append("Unexpected policy/cell")
        out["input_contract_matches"] = signature(raw["requests"]) == expected_signature
        if not out["input_contract_matches"]:
            out["errors"].append("Frozen request/prompt/arrival/output-count contract mismatch")
        if raw["pulse"]["frozen_design_sha256"] != design_hash:
            out["errors"].append("Raw frozen design hash mismatch")
        out["pulse"] = pulse_report(raw, design)
        out["errors"].extend(out["pulse"]["errors"])
        last_arrival = max(q["arrival_s"] for q in raw["requests"])
        out["full_elapsed_after_last_arrival_s"] = raw["elapsed_s"]-last_arrival
        out["valid"] = not out["errors"] and out.get("summary", {}).get("all_finished", False)
    except Exception as exc:
        out["errors"].append(repr(exc))
    out["status"] = "VALID_COMPLETE" if out["valid"] else "INVALID_OR_INCOMPLETE"
    return out


def prefix_compare(hl, lh, design):
    a, b = hl["pulse"], lh["pulse"]
    if not a["trigger"] or not b["trigger"] or not a["trigger_snapshot"] or not b["trigger_snapshot"]:
        return dict(passes=False, status="NO_PAIRED_TRIGGER_SNAPSHOT")
    x, y = a["trigger"], b["trigger"]; sa, sb = a["trigger_snapshot"], b["trigger_snapshot"]
    aq, bq = ({q["request_id"]: q for q in s["requests"]} for s in (sa, sb))
    ia, ib = set(aq), set(bq); common = sorted(ia & ib)
    context_scale = max(x["decode_context_sum"], y["decode_context_sum"])
    distances = dict(trigger_time_difference_s=abs(x["decision_time_s"]-y["decision_time_s"]),
        D_difference=abs(x["decode_count"]-y["decode_count"]),
        backlog_difference_tokens=abs(x["prefill_backlog_tokens"]-y["prefill_backlog_tokens"]),
        context_relative_difference=abs(x["decode_context_sum"]-y["decode_context_sum"])/context_scale if context_scale else 0,
        present_ID_jaccard=len(ia & ib)/len(ia | ib) if ia | ib else 1)
    g = design["prefix_guard"]
    checks = {k: distances[k] <= g["max_paired_"+k] for k in
              ("trigger_time_difference_s", "D_difference", "backlog_difference_tokens", "context_relative_difference")}
    checks["present_ID_jaccard"] = distances["present_ID_jaccard"] >= g["min_paired_present_ID_jaccard"]
    checks["observed_irreversible_failures"] = all(len(p["observed_irreversible_failure_ids"]) == g["required_observed_irreversible_failures_each"] for p in (a, b))
    ar, br = ({q["request_id"]: q for q in r["_raw"]["requests"]} for r in (hl, lh))
    changed_outputs = [rid for rid in common if ar[rid]["output_token_ids"][:aq[rid]["observed_output_tokens"]]
                       != br[rid]["output_token_ids"][:bq[rid]["observed_output_tokens"]]]
    return dict(passes=all(checks.values()), status="APPROXIMATE_PREFIX_COMPARED", distances=distances, gate_checks=checks,
        hl_only_present_ids=sorted(ia-ib), lh_only_present_ids=sorted(ib-ia),
        running_order_matches=sa["running_order"] == sb["running_order"], waiting_order_matches=sa["waiting_order"] == sb["waiting_order"],
        computed_tokens_different_ids=[rid for rid in common if aq[rid]["computed_tokens"] != bq[rid]["computed_tokens"]],
        observed_output_count_different_ids=[rid for rid in common if aq[rid]["observed_output_tokens"] != bq[rid]["observed_output_tokens"]],
        observed_output_prefix_different_ids=changed_outputs,
        common_present_age_hl_minus_lh_s=native.describe([aq[r]["age_s"]-bq[r]["age_s"] for r in common]),
        note="abs(context_HL-context_LH)/max(context_HL,context_LH); zero/zero=0. Approximate policy prefix, not an exact state/KV counterfactual. Full elapsed is unchanged.")


def compare(hl, lh, design):
    result = dict(hl_cell=hl.get("cell") if hl else None, lh_cell=lh.get("cell") if lh else None, directions={})
    if not hl or not lh or any("summary" not in r or "pulse" not in r for r in (hl, lh)):
        return {**result, "status": "UNRUN_OR_UNREADABLE"}
    aq, bq = ({q["request_id"]: q for q in r["requests"]} for r in (hl, lh))
    if aq.keys() != bq.keys():
        return {**result, "status": "REQUEST_ID_MISMATCH"}
    prefix = prefix_compare(hl, lh, design)
    dose = all(r["pulse"]["actual_dose_matches"] for r in (hl, lh))
    for name, candidate, reference, cq, rq in (("hl_over_lh", hl, lh, aq, bq), ("lh_over_hl", lh, hl, bq, aq)):
        c, r = candidate["summary"], reference["summary"]
        cp, rp = ({rid for rid, q in rows.items() if q["joint_slo_pass"]} for rows in (cq, rq))
        cg, rg = ({rid for rid, q in rows.items() if q["max_observed_generation_gap_s"] is not None
                   and q["max_observed_generation_gap_s"] > design["primary_slo"]["gap_s"]} for rows in (cq, rq))
        effects = {k: percent(c[k], r[k]) for k in METRICS}
        gates = dict(both_complete_valid=candidate["valid"] and reference["valid"],
            goodput_gain_at_least_3_percent=effects["joint_slo_requests_per_s"] is not None and effects["joint_slo_requests_per_s"] >= 3,
            raw_output_throughput_not_decreased=c["output_tokens_per_s"] >= r["output_tokens_per_s"],
            no_added_100ms_gap_failure_ids=not (cg-rg))
        for k in ("ttft_p95_s", "completion_flow_p95_s", "request_max_gap_p95_s"):
            gates[k+"_within_1_05_reference"] = c[k] is not None and r[k] is not None and c[k] <= 1.05*r[k]
        result["directions"][name] = dict(service_gates_pass=all(gates.values()), passes_frozen_gate=all(gates.values()) and dose and prefix["passes"],
            service_gate_checks=gates, relative_changes_percent=effects, candidate_qualified_numerator=len(cp), reference_qualified_numerator=len(rp),
            candidate_full_elapsed_s=c["elapsed_s"], reference_full_elapsed_s=r["elapsed_s"], qualified_added_ids=sorted(cp-rp),
            qualified_lost_ids=sorted(rp-cp), additional_100ms_gap_failure_ids=sorted(cg-rg))
    ar, br = ({q["request_id"]: q for q in r["_raw"]["requests"]} for r in (hl, lh))
    different = [rid for rid in ar if ar[rid]["output_token_ids"] != br[rid]["output_token_ids"]]
    flow = {rid: aq[rid]["completion_flow_s"]-bq[rid]["completion_flow_s"]
            if aq[rid]["completion_flow_s"] is not None and bq[rid]["completion_flow_s"] is not None else None for rid in aq}
    return {**result, "status": "ALL_REQUESTS_COMPARED", "prefix": prefix, "both_actual_doses_match": dose,
        "output_lengths_match": all(len(ar[r]["output_token_ids"]) == len(br[r]["output_token_ids"]) for r in ar),
        "output_token_ids_different_count": len(different), "output_token_ids_different_ids": sorted(different),
        "flow_hl_minus_lh_s_by_request": flow, "flow_difference_statistics_s": native.describe(list(flow.values())),
        "hl_flow_harmed_ids": sorted(r for r, v in flow.items() if v is not None and v > 0),
        "hl_flow_improved_ids": sorted(r for r, v in flow.items() if v is not None and v < 0)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path, nargs="?", default=ROOT)
    parser.add_argument("--design", type=Path, default=ROOT / "design.json")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(); design_bytes = args.design.read_bytes(); design = json.loads(design_bytes)
    work = (args.design.parent / design["workload"]).read_bytes()
    if sha(work) != design["workload_sha256"]:
        parser.error("Frozen workload hash mismatch")
    expected = [f"{i:02d}_{p}" for i, p in enumerate(design["policies"])]
    warms = ["warm_"+p for p in design["warm_policies"]]
    rows = [load_run(p, design, signature(json.loads(work)), sha(design_bytes)) for p in sorted(args.run_directory.rglob("raw.json"))]
    def unique(name):
        matches = [r for r in rows if r["cell"] == name]
        return matches[0] if len(matches) == 1 else None
    missing = [n for n in expected+warms if not any(r["cell"] == n for r in rows)]
    duplicates = [n for n in expected+warms if sum(r["cell"] == n for r in rows) > 1]
    unexpected = [r["raw_path"] for r in rows if r["cell"] not in expected+warms]
    pairs = [compare(unique(expected[0]), unique(expected[1]), design), compare(unique(expected[3]), unique(expected[2]), design)]
    coverage = {str(cap): dict(warm_cell="warm_fixed"+str(cap),
        warm_valid_complete=bool((unique("warm_fixed"+str(cap)) or {}).get("valid")),
        warm_actual_prefill_steps=(unique("warm_fixed"+str(cap)) or {}).get("pulse", {}).get("positive_prefill_caps", {}).get(str(cap), 0))
        for cap in design["legal_caps"]}
    status_file = args.run_directory / "status.json"
    try:
        controller = json.loads(status_file.read_text()) if status_file.exists() else dict(status="MISSING")
    except Exception as exc:
        controller = dict(status="UNREADABLE", error=repr(exc))
    logs = []
    for path in sorted(set(args.run_directory.glob("*.log")) | {args.run_directory.with_suffix(".log")}):
        if path.is_file():
            lines = path.read_text(errors="replace").splitlines()
            logs.append(dict(path=str(path), stage_lines=[s for s in lines if s.startswith(("WARMUP", "MEASURE", "WAITING_SHARED_LOCK", "LOCK_BUSY"))], tail=lines[-12:]))
    observed = sum(r["cell"] in expected for r in rows)
    ready = (not missing and not duplicates and not unexpected and controller.get("status") == "COMPLETE"
             and all((unique(n) or {}).get("valid", False) for n in expected+warms)
             and all(v["warm_actual_prefill_steps"] > 0 for v in coverage.values()))
    winners = [name for name in ("hl_over_lh", "lh_over_hl") if ready and all(p["directions"].get(name, {}).get("passes_frozen_gate") for p in pairs)]
    status = "UNRUN" if not observed and not unexpected and controller.get("status") not in ("FAILED", "UNREADABLE") else "INCOMPLETE_OR_INVALID"
    if ready:
        if all((unique(n) or {})["pulse"]["actual_up_steps"] == 0 for n in expected): status = "NO_TRIGGER_OR_NO_ACTUAL_DIFFERENCE"
        elif not all(p.get("both_actual_doses_match", False) for p in pairs): status = "UNMATCHED_ACTUAL_PREFILL"
        elif not all(p["prefix"]["passes"] for p in pairs): status = "PREFIX_GUARD_FAILED"
        else: status = "PROVISIONAL_SERVICE_WIN" if winners else "NO_FROZEN_SERVICE_WIN"
    result = dict(schema="d-bounded-timing-v1", status=status, observed_formal_runs=observed,
        valid_complete_formal_runs=sum(r["cell"] in expected and r["valid"] for r in rows),
        missing_cells=missing, duplicate_cells=duplicates, unexpected_raw_paths=unexpected,
        controller_status=controller, controller_logs=logs, warm_actual_cap_coverage=coverage,
        winning_directions=winners, pairs=pairs, runs=[{k: v for k, v in r.items() if not k.startswith("_")} for r in rows],
        frozen_design=design, frozen_design_sha256=sha(design_bytes),
        scope="All arrivals and full elapsed retained, with no prefix subtraction. Passing approximate prefix guards is not exact logical/KV/compute equivalence. Equal prompt/output counts are not equal content/routing/quality. A two-pair order win only warrants comparison with concurrent fixed1024 and state-independent timing; no deployable-selector or statistical-equivalence claim.")
    output = args.output or args.run_directory / "timing_results.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    print(f"{observed} formal, {result['valid_complete_formal_runs']} valid complete; {status}; {output}")


if __name__ == "__main__":
    main()
