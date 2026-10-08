#!/usr/bin/env python3
"""CPU-only complete-service analysis; declared-length chunk component, not full Niyama."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT.parents[1]), str(ROOT.parent / "mixed_arrivals")]
import analyze as native
from analyze_fixed import METRICS, percent, signature


def sha(data): return hashlib.sha256(data).hexdigest()
def distribution(values): return dict(Counter(str(v) for v in values))


def longest_streak(rows):
    groups = []
    for row in rows:
        if not groups or row["index"] != groups[-1][-1]["index"]+1: groups.append([])
        groups[-1].append(row)
    longest = max(groups, key=len, default=[])
    return dict(streak_count=len(groups), steps=len(longest),
        first_step=longest[0]["index"] if longest else None, last_step=longest[-1]["index"] if longest else None,
        first_start_s=longest[0]["start"] if longest else None, last_end_s=longest[-1]["end"] if longest else None,
        observed_wall_duration_s=longest[-1]["end"]-longest[0]["start"] if longest else 0,
        max_streak_wall_duration_s=max((g[-1]["end"]-g[0]["start"] for g in groups), default=0),
        selection="Longest consecutive completed-step count; maximum wall-duration streak also reported.",
        backlog_tokens=native.describe([r["backlog"] for r in longest]))


def model_errors(rows):
    def describe(subset):
        return dict(steps=len(subset), actual_host_s=native.describe([r["host"] for r in subset]),
            reconstructed_prediction_s=native.describe([r["prediction"] for r in subset]),
            negative_prediction_steps=sum(r["prediction"] < 0 for r in subset),
            mean_absolute_error_s=sum(abs(r["host"]-r["prediction"]) for r in subset)/len(subset) if subset else None,
            actual_minus_prediction_s=native.describe([r["host"]-r["prediction"] for r in subset]),
            actual_minus_1_2_prediction_s=native.describe([r["host"]-1.2*r["prediction"] for r in subset]),
            actual_above_1_2_prediction_steps=sum(r["host"] > 1.2*r["prediction"] for r in subset))
    return dict(all_completed=describe(rows), by_actual_regime={k: describe([r for r in rows if r["regime"] == k]) for k in ("le512", "gt512")},
        observed_steps_over_1s=sum(r["host"] > 1 for r in rows),
        scope="No online fit, clipping or outlier removal. Reconstruct actual P,D,sum(computed_start+1 for decode rows),sum(computed_start for prefill rows); select regime by actual P+D. Requested-cap/head-estimate predictions are not scored against underfilled execution. Host time is not GPU-only time.")


def execution(raw, calibration):
    errors, records, requested = [], [], []
    control = raw["component"]
    for i, s in enumerate(raw["steps"]):
        d = s["component_decision"]; requested.append(d)
        if (d["step_index"] != i or d["requested_cap"] != s["budget"]
                or d["actual_prefill_tokens"] != s["prefill_tokens"] or d["actual_decode_tokens"] != s["decode_tokens"]
                or d["prefill_backlog_tokens"] != s["prefill_backlog_tokens_before"]):
            errors.append(f"Decision/native execution mismatch at step {i}")
        if raw["policy"] == "fixed1024" and d["requested_cap"] != 1024:
            errors.append(f"Fixed1024 cap changed at step {i}")
        completed = d.get("forward_completed") is True and s["end_s"] is not None
        if not completed:
            errors.append(f"Forward unconfirmed at step {i}")
            continue
        p, dec = s["prefill_tokens"], s["decode_tokens"]
        cd = sum(q["computed_start"]+1 for q in s["requests"] if q["decode_tokens"] > 0)
        cp = sum(q["computed_start"] for q in s["requests"] if q["prefill_tokens"] > 0)
        if d["actual_prefill_context"] != cp: errors.append(f"Actual prefill-context mismatch at step {i}")
        regime = "le512" if p+dec <= 512 else "gt512"; model = calibration["runtime_models"][regime]
        prediction = model["intercept_s"]+sum(a*b for a, b in zip(model["coefficients"], (p, dec, cd, cp)))
        if not math.isfinite(prediction): raise ValueError("Nonfinite actual-feature model prediction")
        records.append(dict(index=i, start=s["start_s"], end=s["end_s"], host=s["end_s"]-s["start_s"],
            p=p, d=dec, cap=d["requested_cap"], backlog=d["prefill_backlog_tokens"],
            tier=d["selected_total_tier"], reason=d["reason"], decision_us=s["decision_us"],
            unused_reason=d["unused_budget_reason"], regime=regime, prediction=prediction))
    if control.get("pending_decision") is not None:
        requested.append(control["pending_decision"]); errors.append("Pending/failed native decision retained")
    if control.get("pending_forward") is not None: errors.append("Pending forward retained")
    if control["completed_forward_count"] != len(records): errors.append("Completed-forward count mismatch")
    backlog = [r for r in records if r["backlog"] > 0]
    above = [r for r in records if r["p"] > 1024]
    binding = [r for r in backlog if r["cap"] < 1024 and r["p"] == r["cap"] and r["backlog"] > r["cap"]]
    zero = [r for r in backlog if r["p"] == 0]
    cap_zero = [r for r in backlog if r["cap"] == 0]
    return dict(errors=errors, controller_summary=control, requested_decisions=len(requested),
        completed_forward_steps=len(records), native_scheduled_steps=len(raw["steps"]),
        native_scheduled_prefill_tokens=sum(s["prefill_tokens"] for s in raw["steps"]),
        native_scheduled_decode_tokens=sum(s["decode_tokens"] for s in raw["steps"]),
        completed_forward_prefill_tokens=sum(r["p"] for r in records), completed_forward_decode_tokens=sum(r["d"] for r in records),
        requested_cap_distribution=distribution(d["requested_cap"] for d in requested),
        requested_selected_total_tier_distribution=distribution(d["selected_total_tier"] for d in requested if d["selected_total_tier"] is not None),
        requested_reason_distribution=distribution(d["reason"] for d in requested),
        requested_cap0_with_backlog=sum(d["requested_cap"] == 0 and d["prefill_backlog_tokens"] > 0 for d in requested),
        actual_P_distribution=distribution(r["p"] for r in records), actual_total_P_plus_D_distribution=distribution(r["p"]+r["d"] for r in records),
        completed_cap_distribution=distribution(r["cap"] for r in records),
        completed_selected_total_tier_distribution=distribution(r["tier"] for r in records if r["tier"] is not None),
        completed_backlog_steps=len(backlog), zero_actual_P_with_backlog_steps=len(zero),
        longest_zero_actual_P_with_backlog=longest_streak(zero), longest_cap0_with_backlog=longest_streak(cap_zero),
        actual_P_above1024_steps=len(above), actual_P_above1024_step_indices=[r["index"] for r in above],
        cap_binding_below1024_steps=len(binding), cap_binding_below1024_step_indices=[r["index"] for r in binding],
        actual_intervention_steps=len(above)+len(binding),
        intervention_definition="Completed-forward P>1024, or backlog>requested cap with cap<1024 and actual P==cap; not a state-matched performance counterfactual.",
        underfill_all_completed_steps=sum(r["p"] < r["cap"] for r in records),
        underfill_with_backlog_steps=sum(r["p"] < r["cap"] for r in backlog),
        backlog_underfill_reasons=distribution(r["unused_reason"] for r in backlog if r["p"] < r["cap"]),
        decision_overhead_us=native.describe([r["decision_us"] for r in records]),
        decision_overhead_total_s=sum(r["decision_us"] for r in records)/1e6,
        actual_feature_host_model_errors=model_errors(records))


def load_run(path, design, expected_signature, calibration, design_hash):
    out = dict(cell=path.parent.name, raw_path=str(path), valid=False, errors=[])
    try:
        data = path.read_bytes(); raw = json.loads(data); out.update(policy=raw["policy"], _raw=raw, raw_sha256=sha(data))
        out["observed_output_counts"] = dict(requests=len(raw["requests"]), finished=sum(q["finished"] for q in raw["requests"]),
            emitted_tokens=sum(len(q["output_token_ids"]) for q in raw["requests"]), declared_tokens=sum(q["max_tokens"] for q in raw["requests"]))
        protocol = native.find_protocol(path, None)
        try:
            report = native.summarize(path, protocol, path.parent.name)
            out.update(summary=report["summary"], requests=report["requests"], request_statistics=report["request_statistics"],
                       failed_by_slo_field=report["failed_by_slo_field"], warnings=report["warnings"])
            out["errors"].extend(report["validation_errors"])
            if report["slo"] != design["primary_slo"]: out["errors"].append("Primary SLO mismatch")
        except Exception as exc: out["errors"].append("Native summarize: "+repr(exc))
        if not protocol or json.loads(Path(protocol).read_text()).get("frozen_component_design_sha256") != design_hash:
            out["errors"].append("Frozen component protocol/design hash mismatch")
        if raw["policy"] not in design["policies"]+design["warm_policies"] or path.parent.name.split("_", 1)[1] != raw["policy"]:
            out["errors"].append("Unexpected policy/cell")
        out["input_contract_matches"] = signature(raw["requests"]) == expected_signature
        if not out["input_contract_matches"]: out["errors"].append("Frozen input contract mismatch")
        out["execution"] = execution(raw, calibration); out["errors"].extend(out["execution"]["errors"])
        out["valid"] = not out["errors"] and out.get("summary", {}).get("all_finished", False)
        out["timing"] = dict(full_elapsed_s=raw["elapsed_s"], last_external_arrival_s=max(q["arrival_s"] for q in raw["requests"]),
            full_elapsed_after_last_arrival_s=raw["elapsed_s"]-max(q["arrival_s"] for q in raw["requests"]) if out.get("summary", {}).get("all_finished") else None,
            client_arrival_to_add_lag_s=out.get("request_statistics", {}).get("arrival_to_add_s"), note="Full elapsed, never drain alone, is the service denominator.")
    except Exception as exc: out["errors"].append(repr(exc))
    out["status"] = "VALID_COMPLETE" if out["valid"] else "INVALID_OR_INCOMPLETE"
    return out


def compare(fixed, component, design):
    out = dict(fixed_cell=fixed.get("cell") if fixed else None, component_cell=component.get("cell") if component else None, directions={})
    if not fixed or not component or any("summary" not in r or "execution" not in r for r in (fixed, component)):
        return {**out, "status": "UNRUN_OR_UNREADABLE"}
    fq, cq = ({q["request_id"]: q for q in r["requests"]} for r in (fixed, component))
    if fq.keys() != cq.keys(): return {**out, "status": "REQUEST_ID_MISMATCH"}
    action = component["execution"]["actual_intervention_steps"] > 0
    for name, c, r, cqs, rqs in (("component_over_fixed", component, fixed, cq, fq), ("fixed_over_component", fixed, component, fq, cq)):
        cs, rs = c["summary"], r["summary"]; effects = {k: percent(cs[k], rs[k]) for k in METRICS}
        cp, rp = ({rid for rid, q in qs.items() if q["joint_slo_pass"]} for qs in (cqs, rqs))
        cg, rg = ({rid for rid, q in qs.items() if q["max_observed_generation_gap_s"] is not None and q["max_observed_generation_gap_s"] > design["primary_slo"]["gap_s"]} for qs in (cqs, rqs))
        gates = dict(both_complete_valid=c["valid"] and r["valid"], actual_component_intervention_observed=action,
            goodput_gain_at_least_3_percent=effects["joint_slo_requests_per_s"] is not None and effects["joint_slo_requests_per_s"] >= 3,
            raw_output_throughput_not_decreased=cs["output_tokens_per_s"] >= rs["output_tokens_per_s"], no_added_100ms_gap_failure_ids=not(cg-rg))
        for k in ("ttft_p95_s", "completion_flow_p95_s", "request_max_gap_p95_s"):
            gates[k+"_within_1_05_reference"] = cs[k] is not None and rs[k] is not None and cs[k] <= 1.05*rs[k]
        out["directions"][name] = dict(passes_frozen_gate=all(gates.values()), gate_checks=gates, relative_changes_percent=effects,
            candidate_qualified_numerator=len(cp), reference_qualified_numerator=len(rp), candidate_full_elapsed_s=cs["elapsed_s"], reference_full_elapsed_s=rs["elapsed_s"],
            qualified_added_ids=sorted(cp-rp), qualified_lost_ids=sorted(rp-cp), additional_100ms_gap_failure_ids=sorted(cg-rg))
    fr, cr = ({q["request_id"]: q for q in r["_raw"]["requests"]} for r in (fixed, component))
    changed = [rid for rid in fr if fr[rid]["output_token_ids"] != cr[rid]["output_token_ids"]]
    flow = {rid: cq[rid]["completion_flow_s"]-fq[rid]["completion_flow_s"] if cq[rid]["completion_flow_s"] is not None and fq[rid]["completion_flow_s"] is not None else None for rid in fq}
    return {**out, "status": "ALL_REQUESTS_COMPARED", "output_lengths_match": all(len(fr[r]["output_token_ids"]) == len(cr[r]["output_token_ids"]) for r in fr),
        "output_token_ids_different_count": len(changed), "output_token_ids_different_ids": sorted(changed),
        "flow_component_minus_fixed_s_by_request": flow, "flow_difference_statistics_s": native.describe(list(flow.values())),
        "component_flow_harmed_ids": sorted(r for r, v in flow.items() if v is not None and v > 0),
        "component_flow_improved_ids": sorted(r for r, v in flow.items() if v is not None and v < 0)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path, nargs="?", default=ROOT)
    parser.add_argument("--design", type=Path, default=ROOT/"design.json" if (ROOT/"design.json").exists() else None)
    parser.add_argument("--output", type=Path); args = parser.parse_args()
    if args.design is None: parser.error("--design is required when niyama_component/design.json is absent")
    db = args.design.read_bytes(); design = json.loads(db); work = (args.design.parent/design["workload"]).read_bytes()
    if sha(work) != design["workload_sha256"]: parser.error("Frozen workload hash mismatch")
    cp = args.run_directory/"calibration.json"
    if not cp.exists(): cp = args.design.parent/design.get("calibration_file", "calibration.json")
    cb = cp.read_bytes(); calibration = json.loads(cb)
    if sha(cb) != design["calibration_sha256"]: parser.error("Frozen calibration hash mismatch")
    expected = [f"{i:02d}_{p}" for i, p in enumerate(design["policies"])]; warms = ["warm_"+p for p in design["warm_policies"]]
    sig = signature(json.loads(work)); rows = [load_run(p, design, sig, calibration, sha(db)) for p in sorted(args.run_directory.rglob("raw.json"))]
    def unique(name):
        matches = [r for r in rows if r["cell"] == name]
        return matches[0] if len(matches) == 1 else None
    missing = [n for n in expected+warms if not any(r["cell"] == n for r in rows)]
    duplicates = [n for n in expected+warms if sum(r["cell"] == n for r in rows) > 1]
    unexpected = [r["raw_path"] for r in rows if r["cell"] not in expected+warms]
    pairs = [compare(unique(expected[0]), unique(expected[1]), design), compare(unique(expected[3]), unique(expected[2]), design)]
    sp = args.run_directory/"status.json"
    try: controller = json.loads(sp.read_text()) if sp.exists() else dict(status="MISSING")
    except Exception as exc: controller = dict(status="UNREADABLE", error=repr(exc))
    coverage = {}
    for policy in design["warm_policies"]:
        warm = unique("warm_"+policy); formal = [r for r in rows if r["cell"] in expected and r.get("policy") == policy]
        capset = set().union(*(set(r.get("execution", {}).get("completed_cap_distribution", {})) for r in formal))
        tiers = set().union(*(set(r.get("execution", {}).get("completed_selected_total_tier_distribution", {})) for r in formal))
        we = (warm or {}).get("execution", {})
        coverage[policy] = dict(warm_valid_complete=bool(warm and warm["valid"]), formal_caps=sorted(capset),
            warm_caps=sorted(we.get("completed_cap_distribution", {})), formal_selected_tiers=sorted(tiers),
            warm_selected_tiers=sorted(we.get("completed_selected_total_tier_distribution", {})),
            note="Descriptive coverage; no invented exact-integer-cap gate or identical-kernel-state claim.")
    observed = sum(r["cell"] in expected for r in rows)
    ready = not missing and not duplicates and not unexpected and controller.get("status") == "COMPLETE" and all((unique(n) or {}).get("valid", False) for n in expected+warms)
    no_action = [n for n in (expected[1], expected[2]) if unique(n) and unique(n).get("execution", {}).get("actual_intervention_steps", 0) == 0]
    winners = [name for name in ("component_over_fixed", "fixed_over_component") if ready and all(p["directions"].get(name, {}).get("passes_frozen_gate") for p in pairs)]
    status = "UNRUN" if not rows and controller.get("status") not in ("COMPLETE", "FAILED", "UNREADABLE") else "INCOMPLETE_OR_INVALID"
    if ready: status = "ZERO_ACTION_DIAGNOSIS" if no_action else "PROVISIONAL_SERVICE_WIN" if winners else "NO_FROZEN_SERVICE_WIN"
    result = dict(schema="d-niyama-chunk-component-v1", status=status, observed_formal_runs=observed,
        valid_complete_formal_runs=sum(r["cell"] in expected and r["valid"] for r in rows), controller_status=controller,
        missing_cells=missing, duplicate_cells=duplicates, unexpected_raw_paths=unexpected, component_runs_without_actual_intervention=no_action,
        winning_directions=winners, warm_execution_coverage=coverage, pairs=pairs,
        runs=[{k: v for k, v in r.items() if not k.startswith("_")} for r in rows], frozen_design=design,
        frozen_design_sha256=sha(db), calibration_sha256=sha(cb),
        scope="Declared-fixed-length chunk component, not full Niyama. All arrivals, failed observations, fixed output counts and full elapsed retained. Output counts do not establish equal content/routing/actual computation/quality. Both reversed pairs must pass unchanged 3%/cost gates; no automatic retries, refit, margin or threshold search.")
    output = args.output or args.run_directory/"component_results.json"; output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    print(f"{observed} formal, {result['valid_complete_formal_runs']} valid complete; {status}; {output}")


if __name__ == "__main__": main()
