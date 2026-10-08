#!/usr/bin/env python3
"""CPU-only fixed-cap comparison; defaults to the frozen 384/2048 design."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT.parents[1]), str(ROOT.parents[1]/"confirmation_v2")]
import analyze as native
from paired_comparison import backlog_statistics

SMALL_CAP, LARGE_CAP = 384, 2048
EXPECTED = ["00_fixed384", "01_fixed2048", "02_fixed2048", "03_fixed384"]
WARMS = ["warm_fixed384", "warm_fixed2048"]
METRICS = ["joint_slo_requests_per_s", "output_tokens_per_s", "prompt_plus_output_tokens_per_s", "ttft_p95_s",
           "completion_flow_p95_s", "request_max_gap_p95_s", "elapsed_s"]
def signature(q): return hashlib.sha256(json.dumps(sorted((r["request_id"], r["arrival_s"], r["prompt_token_ids"], r["max_tokens"]) for r in q), separators=(",", ":")).encode()).hexdigest()
def percent(a, b): return 100*(a/b-1) if a is not None and b else None


def load(path, design, expected_signature):
    out = {"cell": path.parent.name, "raw_path": str(path), "valid": False, "errors": []}
    try:
        raw_bytes = path.read_bytes(); raw = json.loads(raw_bytes); out.update(policy=raw["policy"], _raw=raw, raw_sha256=hashlib.sha256(raw_bytes).hexdigest())
        report = native.summarize(path, native.find_protocol(path, None), path.parent.name)
        out.update(summary=report["summary"], requests=report["requests"], request_statistics=report["request_statistics"],
                   failed_by_slo_field=report["failed_by_slo_field"], errors=report["validation_errors"], warnings=report["warnings"])
        if report["slo"] != design["primary_slo"]: out["errors"].append("Primary SLO differs from frozen design")
        if raw["policy"] not in (f"fixed{SMALL_CAP}", f"fixed{LARGE_CAP}") or path.parent.name.split("_", 1)[1] != raw["policy"]: raise ValueError("Unexpected policy/cell")
        out["input_contract_matches_frozen_workload"] = signature(raw["requests"]) == expected_signature
        if not out["input_contract_matches_frozen_workload"]: out["errors"].append("IDs/prompts/arrivals/fixed-output contract differs")
        cap = int(raw["policy"][5:]); steps = raw["steps"]; mixed = [s for s in steps if s["prefill_tokens"] > 0 and s["decode_tokens"] > 0]
        wrong = [i for i, s in enumerate(steps) if s["budget"] != cap]
        if wrong: out["errors"].append("Fixed cap differs at steps: "+str(wrong[:12]))
        coverage = {}
        for s in steps:
            if s["prefill_tokens"] > 0: coverage.setdefault(str(s["budget"]), Counter())[str(s["prefill_tokens"])] += 1
        out["execution"] = {"fixed_cap": cap, "mixed_steps": len(mixed), f"nominal_cap_above_{SMALL_CAP}_steps": sum(s["budget"] > SMALL_CAP for s in steps),
            f"nominal_above_{SMALL_CAP}_with_backlog_steps": sum(s["budget"] > SMALL_CAP and s["prefill_backlog_tokens_before"] > 0 for s in steps),
            f"actual_P_above_{SMALL_CAP}_steps": sum(s["prefill_tokens"] > SMALL_CAP for s in steps),
            f"mixed_actual_P_above_{SMALL_CAP}_steps": sum(s["prefill_tokens"] > SMALL_CAP for s in mixed),
            "actual_P_distribution_all_steps": dict(Counter(str(s["prefill_tokens"]) for s in steps)),
            "actual_P_distribution_mixed_steps": dict(Counter(str(s["prefill_tokens"]) for s in mixed)), "positive_actual_P_by_cap": coverage,
            "scope": f"Actual recorded execution and fixed numeric{SMALL_CAP} threshold; no state-matched or performance counterfactual."}
        out["backlog"] = backlog_statistics(path)
        if not out["backlog"]["reconstruction"]["all_fields_match"]: out["errors"].append("Backlog reconstruction mismatch")
        last_arrival = max(r["arrival_s"] for r in raw["requests"])
        last_add = max((r["add_s"] for r in raw["requests"] if r.get("add_s") is not None), default=None)
        out["timing"] = {"last_external_arrival_s": last_arrival, "last_add_s": last_add,
            "full_elapsed_after_last_arrival_s": raw["elapsed_s"]-last_arrival if report["summary"]["all_finished"] else None,
            "full_elapsed_after_last_add_s": raw["elapsed_s"]-last_add if report["summary"]["all_finished"] and last_add is not None else None,
            "client_arrival_to_add_lag_s": report["request_statistics"]["arrival_to_add_s"], "note": "Full elapsed, not drain alone, remains the denominator."}
        out["valid"] = not out["errors"] and report["summary"]["all_finished"]
    except Exception as exc: out["errors"].append(type(exc).__name__+": "+str(exc))
    out["status"] = "VALID_COMPLETE" if out["valid"] else "INVALID_OR_INCOMPLETE"
    return out


def compare(small, large):
    out = {"small_cell": small.get("cell") if small else None, "large_cell": large.get("cell") if large else None, "directions": {}}
    if not small or not large or "summary" not in small or "summary" not in large: return {**out, "status": "UNRUN_OR_UNREADABLE"}
    sq, lq = ({q["request_id"]: q for q in r["requests"]} for r in (small, large))
    sr, lr = ({q["request_id"]: q for q in r["_raw"]["requests"]} for r in (small, large))
    if sq.keys() != lq.keys(): return {**out, "status": "INVALID_REQUEST_ID_MISMATCH"}
    changed = {}; flow = {}
    for rid in sq:
        st, lt = sr[rid]["output_token_ids"], lr[rid]["output_token_ids"]
        if st != lt: changed[rid] = next((i for i, (a, b) in enumerate(zip(st, lt)) if a != b), min(len(st), len(lt)))
        a, b = lq[rid]["completion_flow_s"], sq[rid]["completion_flow_s"]; flow[rid] = a-b if a is not None and b is not None else None
    for name, candidate, reference, cq, rq in (("large_over_small", large, small, lq, sq), ("small_over_large", small, large, sq, lq)):
        c, r = candidate["summary"], reference["summary"]
        cp, rp = ({rid for rid, q in qs.items() if q["joint_slo_pass"]} for qs in (cq, rq))
        cg, rg = ({rid for rid, q in qs.items() if q["max_observed_generation_gap_s"] is not None and q["max_observed_generation_gap_s"] > .1} for qs in (cq, rq))
        effects = {k: percent(c[k], r[k]) for k in METRICS}
        gates = {"both_complete_valid": candidate["valid"] and reference["valid"],
            "actual_execution_difference_observed": large.get("execution", {}).get(f"actual_P_above_{SMALL_CAP}_steps", 0) > 0,
            "goodput_gain_at_least_3_percent": effects["joint_slo_requests_per_s"] is not None and effects["joint_slo_requests_per_s"] >= 3,
            "raw_output_throughput_not_decreased": c["output_tokens_per_s"] >= r["output_tokens_per_s"], "no_added_100ms_gap_failure_ids": not(cg-rg)}
        for k in ("ttft_p95_s", "completion_flow_p95_s", "request_max_gap_p95_s"):
            gates[k+"_within_1_05_reference"] = c[k] is not None and r[k] is not None and c[k] <= 1.05*r[k]
        out["directions"][name] = {"passes_frozen_gate": all(gates.values()), "gate_checks": gates, "relative_changes_percent": effects,
            "candidate_qualified_numerator": len(cp), "reference_qualified_numerator": len(rp), "candidate_full_elapsed_s": c["elapsed_s"], "reference_full_elapsed_s": r["elapsed_s"],
            "qualified_added_ids": sorted(cp-rp), "qualified_lost_ids": sorted(rp-cp), "additional_100ms_gap_failure_ids": sorted(cg-rg)}
    return {**out, "status": "COMPARED_ALL_REQUESTS", "output_lengths_match": all(len(sr[r]["output_token_ids"]) == len(lr[r]["output_token_ids"]) for r in sr),
        "output_token_ids_different_count": len(changed), "output_token_ids_different_ids": sorted(changed), "different_output_LCP_tokens": native.describe(list(changed.values())),
        "flow_large_minus_small_s_by_request": flow, "flow_difference_statistics_s": native.describe(list(flow.values())),
        "flow_harmed_ids": sorted(r for r, d in flow.items() if d is not None and d > 0), "flow_improved_ids": sorted(r for r, d in flow.items() if d is not None and d < 0)}


def main():
    global SMALL_CAP, LARGE_CAP, EXPECTED, WARMS
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("run_directory", type=Path, nargs="?", default=ROOT)
    parser.add_argument("--output", type=Path); parser.add_argument("--design", type=Path, default=ROOT/"design.json")
    parser.add_argument("--small-cap", type=int, default=384); parser.add_argument("--large-cap", type=int, default=2048)
    args = parser.parse_args(); SMALL_CAP, LARGE_CAP = args.small_cap, args.large_cap
    small, large = f"fixed{SMALL_CAP}", f"fixed{LARGE_CAP}"
    EXPECTED = [f"00_{small}", f"01_{large}", f"02_{large}", f"03_{small}"]; WARMS = [f"warm_{small}", f"warm_{large}"]
    design = json.loads(args.design.read_text()); wb = (args.design.parent/design["workload"]).read_bytes()
    if hashlib.sha256(wb).hexdigest() != design["workload_sha256"]: parser.error("Frozen workload hash mismatch")
    expected_signature = signature(json.loads(wb)); rows = [load(p, design, expected_signature) for p in sorted(args.run_directory.rglob("raw.json"))]
    def unique(name):
        matches = [r for r in rows if r["cell"] == name]
        return matches[0] if len(matches) == 1 else None
    missing = [n for n in EXPECTED if not any(r["cell"] == n for r in rows)]
    duplicates = [n for n in EXPECTED+WARMS if sum(r["cell"] == n for r in rows) > 1]
    unexpected = [r["raw_path"] for r in rows if r["cell"] not in EXPECTED+WARMS]
    pairs = [compare(unique(EXPECTED[0]), unique(EXPECTED[1])), compare(unique(EXPECTED[3]), unique(EXPECTED[2]))]
    coverage = {}
    for policy in (small, large):
        warm = unique("warm_"+policy); formal_caps = set().union(*(set(r.get("execution", {}).get("positive_actual_P_by_cap", {})) for r in rows if r["cell"] in EXPECTED and r.get("policy") == policy))
        warm_caps = set((warm or {}).get("execution", {}).get("positive_actual_P_by_cap", {}))
        coverage[policy] = {"formal_positive_P_caps": sorted(formal_caps), "warm_positive_P_caps": sorted(warm_caps), "missing_caps": sorted(formal_caps-warm_caps),
            "warm_valid_complete": bool(warm and warm["valid"]), "note": "Cap coverage plus per-run actual-P distributions; not proof of identical kernel/graph state."}
    status_file = args.run_directory/"status.json"
    try: driver = json.loads(status_file.read_text()) if status_file.exists() else {"status": "MISSING"}
    except Exception as exc: driver = {"status": "UNREADABLE", "error": type(exc).__name__+": "+str(exc)}
    observed = sum(r["cell"] in EXPECTED for r in rows)
    ready = not missing and not duplicates and not unexpected and all((unique(n) or {}).get("valid") for n in EXPECTED) and all(c["warm_valid_complete"] and not c["missing_caps"] for c in coverage.values()) and driver.get("status") == "COMPLETE"
    winners = [name for name in ("large_over_small", "small_over_large") if ready and all(p["directions"].get(name, {}).get("passes_frozen_gate") for p in pairs)]
    repeated_signs = {}
    for metric in METRICS:
        v = [p["directions"].get("large_over_small", {}).get("relative_changes_percent", {}).get(metric) for p in pairs]
        repeated_signs[metric] = "positive_both" if all(x is not None and x > 0 for x in v) else "negative_both" if all(x is not None and x < 0 for x in v) else "zero_both" if all(x == 0 for x in v) else "mixed_or_unobserved"
    result = {"schema": "d-mixed-arrival-fixed-v1", "status": "UNRUN" if observed == 0 and not unexpected and driver.get("status") not in ("FAILED", "UNREADABLE") else "INCOMPLETE_OR_INVALID" if not ready else "PROVISIONALLY_SUFFICIENT_FIXED_CAP" if winners else "FIXED_FRONTIER_NO_UNIFORM_WINNER",
        "observed_formal_runs": observed, "valid_complete_formal_runs": sum(r["cell"] in EXPECTED and r["valid"] for r in rows), "driver_status": driver,
        "missing_formal_cells": missing, "duplicate_cells": duplicates, "unexpected_raw_paths": unexpected, "sufficient_directions": winners,
        "pairs": pairs, "large_minus_small_repeated_effect_signs": repeated_signs, "warm_actual_cap_coverage": coverage,
        "runs": [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows], "frozen_design": design,
        "scope": "Two reversed pairs on one fresh development arrival, not statistical confirmation. All requests and full elapsed retained; fixed prompt/output counts are not content, routing, computation or quality equivalence. No per-window oracle or controller claim. Mixed signs remain inconclusive; a missing uniform winner alone is not evidence for dynamics."}
    output = args.output or args.run_directory/"fixed_results.json"; output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    print(f"{observed} formal, {result['valid_complete_formal_runs']} valid complete; {result['status']}; {output}")


if __name__ == "__main__": main()
