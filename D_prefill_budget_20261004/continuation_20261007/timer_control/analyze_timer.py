#!/usr/bin/env python3
"""CPU adapter: python3 -B analyze_timer.py RUN_DIRECTORY [--output FILE]."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT.parent), str(ROOT.parents[1]), str(ROOT.parents[1]/"confirmation_v2")]
import analyze as native
from prefill_policy import Policy
from paired_comparison import backlog_statistics

EXPECTED = ["00_timer19", "01_viability_escape", "02_viability_escape", "03_timer19"]
WARMS = ["warm_timer19", "warm_viability_escape"]
METRICS = ["joint_slo_requests_per_s", "output_tokens_per_s", "prompt_plus_output_tokens_per_s", "ttft_p95_s",
           "completion_flow_p95_s", "request_max_gap_p95_s", "elapsed_s"]
def digest(x): return hashlib.sha256(json.dumps(x, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
def signature(q): return digest(sorted((r["request_id"], r["arrival_s"], r["prompt_token_ids"], r["max_tokens"]) for r in q))
def percent(a, b): return 100*(a/b-1) if a is not None and b else None


def action_summary(raw):
    steps = raw["steps"]; replay = Policy("elapsed_replay", 24); bad = []; nominal = []; actual = []; coverage = {}
    for i, s in enumerate(steps):
        b, c, requested, executed = (s[k] for k in ("baseline_cap", "candidate_cap", "requested_cap", "executed_cap"))
        d, k, p = s["D"], s["K"], s["prefill_tokens"]; condition = d > 0 and k == 0 and s["prefill_backlog_tokens_before"] > b
        expected = c if raw["policy"] == "viability_escape" else (2048 if d > 0 and s["decision_time_s"] >= 19.0 and s["prefill_backlog_tokens_before"] > b else b)
        if not (b == replay.choose(d) and d == s["decode_count_before"] and 0 <= k <= d
                and s["viability_failure_reason_counts"]["any"] == d-k and s["escape_condition"] == condition
                and s["timer_condition"] == (d > 0 and s["decision_time_s"] >= 19.0 and s["prefill_backlog_tokens_before"] > b)
                and c == (2048 if condition else b) and requested == expected == executed == s["budget"]
                and p <= executed and s["start_s"] <= s["decision_time_s"] <= s["end_s"]): bad.append(i)
        if requested > b: nominal.append(i)
        if p > b: actual.append(i)
        if p > 0: coverage.setdefault(str(executed), Counter())[str(p)] += 1
        replay.observe(s["end_s"]-s["start_s"], p, s["decode_tokens"])
    return {"scope": "Completed-forward steps only; failed decisions are retained separately and never count as executed P. R cap is replayed on this run's state, not a baseline-run performance counterfactual.",
        "invalid_action_steps": bad, "condition_steps": sum(s["escape_condition"] for s in steps),
        "candidate_cap_above_R_steps": sum(s["candidate_cap"] > s["baseline_cap"] for s in steps),
        "nominal_requested_above_R_count": len(nominal), "actual_P_above_R_cap_count": len(actual),
        "nominal_without_actual_increase_count": len(set(nominal)-set(actual)),
        "actual_extra_P_above_R_cap_sum": sum(steps[i]["prefill_tokens"]-steps[i]["baseline_cap"] for i in actual),
        "first_nominal_step": nominal[0] if nominal else None, "first_actual_step": actual[0] if actual else None,
        "actual_P_distribution_all_steps": dict(Counter(str(s["prefill_tokens"]) for s in steps)),
        "positive_actual_P_by_executed_cap": coverage,
        "chosen_reason_counts": dict(Counter(str(s.get("chosen_reason")) for s in steps)),
        "unused_reason_counts": dict(Counter(str(s.get("unused_reason")) for s in steps))}


def load(path, design, workload_signature):
    out = {"cell": path.parent.name, "raw_path": str(path), "valid": False, "errors": []}
    try:
        raw = json.loads(path.read_text()); out.update(policy=raw["policy"], _raw=raw, failed_decisions=raw.get("failed_decisions", []))
        report = native.summarize(path, native.find_protocol(path, None), path.parent.name)
        out.update(summary=report["summary"], requests=report["requests"], request_statistics=report["request_statistics"],
                   errors=report["validation_errors"], warnings=report["warnings"], raw_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        if report["slo"] != design["primary_slo"]: out["errors"].append("Primary SLO differs from frozen design")
        if out["failed_decisions"]: out["errors"].append("Failed native decisions retained; actual execution is not assumed")
        expected_policy = path.parent.name.split("_", 1)[1]
        if raw["policy"] != expected_policy: out["errors"].append("Policy differs from formal/warm cell name")
        out["input_contract_matches_frozen_workload"] = signature(raw["requests"]) == workload_signature
        if not out["input_contract_matches_frozen_workload"]: out["errors"].append("Prompt/arrival/ID/fixed-length contract differs")
        out["actions"] = action_summary(raw); out["backlog"] = backlog_statistics(path)
        if out["actions"]["invalid_action_steps"]: out["errors"].append("Recorded action fields/R recurrence inconsistent")
        if not out["backlog"]["reconstruction"]["all_fields_match"]: out["errors"].append("Backlog reconstruction mismatch")
        last_arrival = max(r["arrival_s"] for r in raw["requests"])
        last_add = max((r["add_s"] for r in raw["requests"] if r.get("add_s") is not None), default=None)
        out["timing"] = {"last_external_arrival_s": last_arrival, "last_add_s": last_add,
            "full_elapsed_after_last_arrival_s": raw["elapsed_s"]-last_arrival if report["summary"]["all_finished"] else None,
            "full_elapsed_after_last_add_s": raw["elapsed_s"]-last_add if report["summary"]["all_finished"] and last_add is not None else None,
            "arrival_to_add_lag_s": report["request_statistics"]["arrival_to_add_s"],
            "drain_note": "Diagnostic tails only; primary denominator remains complete elapsed_s."}
        out["valid"] = not out["errors"] and report["summary"]["all_finished"]
    except Exception as exc: out["errors"].append(type(exc).__name__+": "+str(exc))
    out["status"] = "VALID_COMPLETE" if out["valid"] else "INVALID_OR_INCOMPLETE"
    return out


def pair(r, v):
    out = {"T_cell": r.get("cell") if r else None, "V_cell": v.get("cell") if v else None, "passes_frozen_gate": False}
    if not r or not v or "summary" not in r or "summary" not in v:
        return {**out, "status": "UNRUN_OR_UNREADABLE"}
    rq, vq = ({q["request_id"]: q for q in x["requests"]} for x in (r, v))
    rr, vr = ({q["request_id"]: q for q in x["_raw"]["requests"]} for x in (r, v))
    if rq.keys() != vq.keys(): return {**out, "status": "INVALID_REQUEST_ID_MISMATCH"}
    rp, vp = ({rid for rid, q in x.items() if q["joint_slo_pass"]} for x in (rq, vq))
    rg, vg = ({rid for rid, q in x.items() if q["max_observed_generation_gap_s"] is not None and q["max_observed_generation_gap_s"] > .1} for x in (rq, vq))
    differences = {}; flow = {}; lengths_equal = True
    for rid in rq:
        rt, vt = rr[rid]["output_token_ids"], vr[rid]["output_token_ids"]; lengths_equal &= len(rt) == len(vt)
        if rt != vt: differences[rid] = next((i for i, (a, b) in enumerate(zip(rt, vt)) if a != b), min(len(rt), len(vt)))
        a, b = vq[rid]["completion_flow_s"], rq[rid]["completion_flow_s"]
        flow[rid] = a-b if a is not None and b is not None else None
    rs, vs = r["summary"], v["summary"]; effects = {k: percent(vs[k], rs[k]) for k in METRICS}
    gates = {"both_complete_valid": r["valid"] and v["valid"], "same_fixed_output_lengths": lengths_equal,
        "goodput_gain_at_least_3_percent": effects["joint_slo_requests_per_s"] is not None and effects["joint_slo_requests_per_s"] >= 3,
        "no_lost_qualifying_request_ids": not (rp-vp), "no_additional_request_over_100ms_gap": not (vg-rg)}
    for k in ("ttft_p95_s", "completion_flow_p95_s"):
        gates[k+"_within_1_05_T"] = rs[k] is not None and vs[k] is not None and vs[k] <= 1.05*rs[k]
    cohorts = {"first_actual_step": v.get("actions", {}).get("first_actual_step"), "groups": {},
        "scope": "Groups use V's state at its first actual P above its R-state cap; not a matched causal prefix. Compare full arrival-relative V-minus-reference completion flow, without rescoring."}
    if cohorts["first_actual_step"] is not None:
        s = v["_raw"]["steps"][cohorts["first_actual_step"]]; t = s["decision_time_s"]; cohorts["decision_time_s"] = t
        dec = {q["request_id"] for q in s["requests"] if q["decode_tokens"] > 0}
        done = {rid for rid, q in vr.items() if q.get("completion_s") is not None and q["completion_s"] <= t}
        not_added = {rid for rid, q in vr.items() if q.get("add_s") is None or q["add_s"] > t}
        future = {rid for rid in not_added if vr[rid]["arrival_s"] > t}
        groups = {"incumbent_decoders": dec, "pending_submitted": set(vr)-done-not_added-dec,
                  "completed_before_action": done, "future_unarrived": future, "arrived_not_added": not_added-future}
        for label, ids in groups.items():
            values = [flow[rid] for rid in sorted(ids) if flow[rid] is not None]
            cohorts["groups"][label] = {"request_ids": sorted(ids), "count": len(ids), "observed_count": len(values),
                "mean_s": sum(values)/len(values) if values and len(values) == len(ids) else None, "min_s": min(values, default=None), "max_s": max(values, default=None),
                "harmed_count": sum(x > 0 for x in values), "improved_count": sum(x < 0 for x in values), "equal_count": sum(x == 0 for x in values)}
    return {**out, "status": "COMPARED_ALL_REQUESTS", "passes_frozen_gate": all(gates.values()), "gate_checks": gates,
        "action_time_cohorts": cohorts,
        "T_qualified_numerator": len(rp), "V_qualified_numerator": len(vp), "T_full_elapsed_s": rs["elapsed_s"], "V_full_elapsed_s": vs["elapsed_s"],
        "V_relative_to_T_percent": effects, "qualified_added_ids": sorted(vp-rp), "qualified_lost_ids": sorted(rp-vp),
        "additional_request_ids_over_100ms_gap": sorted(vg-rg), "output_token_ids_different_count": len(differences),
        "output_token_ids_different_ids": sorted(differences), "different_output_LCP_tokens": native.describe(list(differences.values())),
        "flow_V_minus_T_s_by_request": flow, "flow_difference_statistics_s": native.describe(list(flow.values())),
        "flow_improved_ids": sorted(k for k, x in flow.items() if x is not None and x < 0),
        "flow_worsened_ids": sorted(k for k, x in flow.items() if x is not None and x > 0),
        "flow_equal_ids": sorted(k for k, x in flow.items() if x == 0)}


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("run_directory", type=Path, nargs="?", default=ROOT)
    parser.add_argument("--output", type=Path); args = parser.parse_args(); design = json.loads((ROOT/"design.json").read_text())
    workload_bytes = (ROOT/design["workload"]).read_bytes()
    if hashlib.sha256(workload_bytes).hexdigest() != design["workload_sha256"]: parser.error("Frozen workload hash mismatch")
    workload = json.loads(workload_bytes); expected_sig = signature(workload)
    paths = sorted(args.run_directory.rglob("raw.json")); rows = [load(p, design, expected_sig) for p in paths]
    def unique(name):
        found = [r for r in rows if r["cell"] == name]
        return found[0] if len(found) == 1 else None
    missing = [n for n in EXPECTED if not any(r["cell"] == n for r in rows)]
    duplicate = [n for n in EXPECTED+WARMS if sum(r["cell"] == n for r in rows) > 1]
    unexpected = [r["raw_path"] for r in rows if r["cell"] not in EXPECTED+WARMS]
    pairs = [pair(unique("00_timer19"), unique("01_viability_escape")), pair(unique("03_timer19"), unique("02_viability_escape"))]
    coverage = {}
    for policy in ("timer19", "viability_escape"):
        warm = unique("warm_"+policy)
        actual = set().union(*(set(r.get("actions", {}).get("positive_actual_P_by_executed_cap", {})) for r in rows if r["cell"] in EXPECTED and r.get("policy") == policy))
        warmed = set((warm or {}).get("actions", {}).get("positive_actual_P_by_executed_cap", {}))
        coverage[policy] = {"formal_positive_P_caps": sorted(actual), "warm_positive_P_caps": sorted(warmed),
            "missing_caps": sorted(actual-warmed), "warm_valid_complete": bool(warm and warm["valid"]),
            "note": "Actual P distributions by cap retained per run; cap coverage alone does not prove identical kernel/graph state."}
    observed = sum(r["cell"] in EXPECTED for r in rows); ready = not missing and not duplicate and not unexpected
    warm_ready = all(c["warm_valid_complete"] and not c["missing_caps"] for c in coverage.values())
    passed = ready and warm_ready and all(p["passes_frozen_gate"] for p in pairs)
    result = {"schema": "d-timer-control-v1", "status": "UNRUN" if observed == 0 and not unexpected else "PROVISIONAL_SIGNAL_VALUE" if passed else "INCOMPLETE_OR_INVALID" if not ready or not warm_ready or not all((unique(n) or {}).get("valid") for n in EXPECTED) else "NO_INDEPENDENT_SIGNAL_VALUE_ESTABLISHED",
        "observed_formal_runs": observed, "valid_complete_formal_runs": sum(r["cell"] in EXPECTED and r["valid"] for r in rows),
        "missing_formal_cells": missing, "duplicate_cells": duplicate, "unexpected_raw_paths": unexpected,
        "runs": [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows], "pairs": pairs,
        "warm_actual_cap_coverage": coverage, "passes_both_pairs": passed, "frozen_design": design,
        "scope": "Two reversed adjacent T/V pairs on one inspected arrival, descriptive development probe only. All requests/full elapsed retained, including failed-SLO requests. Prompt and fixed output counts do not imply same generated content, routes, computation or quality; divergent output never excludes a sample."}
    output = args.output or args.run_directory/"timer_results.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    print(f"{observed} formal, {result['valid_complete_formal_runs']} valid complete; {result['status']}; {output}")


if __name__ == "__main__": main()
