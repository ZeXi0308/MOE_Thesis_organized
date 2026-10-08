#!/usr/bin/env python3
"""CPU-only analysis: python3 -B analyze_probe.py RUN_DIRECTORY [--output FILE]."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1]))
import analyze as native

NAMES = [f"s{a}_{o}_r{r}" for a in (291, 293) for r in (1, 2) for o in ("hl", "lh")]
THRESHOLDS = {"service_U": 1.0, "residual_mean_s": 5.0, "residual_p95_s": 5.0,
              "incumbent_maxgap_p95_s": 2.0, "head_ttft_s": 5.0}
def encoded(x): return json.dumps(x, separators=(",", ":"), sort_keys=True).encode()
def digest(x): return hashlib.sha256(encoded(x)).hexdigest()
def file_sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def percent(a, b): return 100 * (a / b - 1) if a is not None and b else None


def analyze_run(path, ledger, design_sha):
    result = {"raw_path": str(path), "eligible": False, "errors": []}
    if native.is_warmup_path(path): return {**result, "warmup": True}
    try:
        raw = json.loads(path.read_text()); name = raw["policy"]; result["policy"] = name
        if name.endswith("_warm") or native.is_warmup_path(path): return {**result, "warmup": True}
        if name not in NAMES: raise ValueError("Unexpected formal policy name")
        a, order, repeat = re.fullmatch(r"s(291|293)_(hl|lh)_r([12])", name).groups(); a = int(a)
        result.update(anchor=a, order=order, repeat=int(repeat), raw_sha256=file_sha(path))
        report = native.summarize(path, native.find_protocol(path, None), path.parent.name)
        result["full_service"] = report
        if report["slo"] != {"ttft_s": 4, "gap_s": .1, "completion_s": 20}: raise ValueError("Primary SLO changed")
        q = {r["request_id"]: r for r in raw["requests"]}; steps = raw["steps"]
        result["final_output_sha256_by_request"] = {rid: digest(r["output_token_ids"]) for rid, r in q.items()}
        cp = ledger["checkpoints"][str(a)]; state = raw["prefix_state"]; lengths = state["output_prefix_lengths"]
        visible = {k: state[k] for k in cp["inferred_online_state"]}
        if set(lengths) != set(q) or any(not isinstance(n, int) or n < 0 or n > len(q[r]["output_token_ids"]) for r, n in lengths.items()):
            raise ValueError("Invalid/missing output prefix lengths")
        fork = steps[a]["start_s"]; schedule = hashlib.sha256()
        for step in steps[:a]: schedule.update(encoded({"budget": step["budget"], "requests": step["requests"]}))
        hashes = {"schedule": schedule.hexdigest(), "state": digest(visible),
                  "output": digest({rid: r["output_token_ids"][:lengths[rid]] for rid, r in q.items()})}
        result["prefix_hashes"] = hashes
        refchecks = {"schedule": hashes["schedule"] == cp["schedule_prefix_sha256"],
                     "state": hashes["state"] == cp["inferred_online_state_sha256"],
                     "output": hashes["output"] == cp["output_prefix_sha256"]}
        refchecks["lengths_match_own_timestamps"] = all(lengths[rid] == sum(t < fork for t in r["token_times_s"]) for rid, r in q.items())
        refchecks["ledger_hash"] = raw.get("prefix_ledger_sha256") == file_sha(ROOT / "frozen_prefix.json")
        refchecks["design_hash"] = design_sha is not None and raw.get("frozen_design_sha256") == design_sha
        refchecks["input_identity"] = set(q) == {r["request_id"] for r in ledger["requests"]}
        refchecks["submission_ledger"] = refchecks["input_identity"] and all(
            q[r["request_id"]]["submit_step"] == r["submit_step"] and q[r["request_id"]]["arrival_s"] == r["arrival_s"]
            and q[r["request_id"]]["add_s"] >= r["arrival_s"] for r in ledger["requests"])
        refchecks["submission_order"] = sorted(q, key=lambda rid: q[rid]["add_s"]) == [r["request_id"] for r in ledger["requests"]]
        result["reference_prefix_checks"] = refchecks; result["reference_prefix_qualified"] = all(refchecks.values())
        positions = {r["request_id"]: r["computed_tokens"] for r in state["request_states"]}
        incumbent = [r["request_id"] for r in state["request_states"] if r["request_id"] in state["running_ids"] and r["computed_tokens"] >= r["prompt_tokens"]]
        two = steps[a:a+2]; actions = [512, 256] if order == "hl" else [256, 512]
        completed = [[r["request_id"] for r in s["requests"] if r["computed_start"] < q[r["request_id"]]["prompt_tokens"] <= r["computed_start"] + r["prefill_tokens"]] for s in two]
        for s in two:
            for r in s["requests"]: positions[r["request_id"]] += r["prefill_tokens"] + r["decode_tokens"]
        counts = {rid: sum(fork <= t <= two[-1]["end_s"] for t in q[rid]["token_times_s"]) for rid in ("high-long-10", "high-long-19")}
        expected_c = [0, 0] if a == 291 else ([1, 0] if order == "hl" else [0, 1])
        expected_d = [58, 59] if a == 293 and order == "hl" else [58, 58]
        action_checks = {"two_actual_P": [s["prefill_tokens"] for s in two] == actions,
            "two_caps": [s["budget"] for s in two] == actions == raw["actions"],
            "C": list(map(len, completed)) == expected_c,
            "recorded_C_ids": all(s["completed_prefill_ids"] == c for s, c in zip(two, completed)),
            "D_and_decode_work": [s["decode_count_before"] for s in two] == expected_d == [s["decode_tokens"] for s in two],
            "head_position": positions["high-long-10"] == (3438 if a == 291 else 3514 if order == "hl" else 3513),
            "next_position": positions["high-long-19"] == (0 if a == 291 else 437),
            "head_outputs": counts["high-long-10"] == (0 if a == 291 else 2 if order == "hl" else 1),
            "incumbent_count": len(incumbent) == 58, "diagnostic_tag": raw.get("diagnostic_only") is True and raw["anchor"] == a}
        mixed, bad_caps = 0, []
        for i, s in enumerate(steps):
            common = 2048 if s["decode_count_before"] == 0 else 1024 if mixed == 0 else 2048 if mixed < 9 else 1024 if mixed < 73 else 512 if mixed < 113 else 256
            if s["budget"] != (actions[i-a] if a <= i < a+2 else common) or s.get("common_budget") != common: bad_caps.append(i)
            mixed += bool(s["prefill_tokens"] and s["decode_tokens"])
        action_checks["only_two_cap_interventions"] = not bad_caps
        result["two_step"] = {"checks": action_checks, "completed_ids": completed, "P": [s["prefill_tokens"] for s in two],
            "D": expected_d if action_checks["D_and_decode_work"] else [s["decode_count_before"] for s in two],
            "actual_decode_tokens": sum(s["decode_tokens"] for s in two), "actual_total_tokens": sum(s["prefill_tokens"] + s["decode_tokens"] for s in two),
            "head_computed_after": positions["high-long-10"], "next_computed_after": positions["high-long-19"], "head_outputs": counts["high-long-10"], "bad_cap_steps": bad_caps}
        end = steps[a+31]["end_s"]; gaps, coverage = {}, {}
        for rid in incumbent:
            n = lengths[rid]; times = q[rid]["token_times_s"]
            emitted = [t for t in times[n:] if t <= end]; coverage[rid] = len(emitted)
            seq = times[n-1:n] + emitted; gaps[rid] = max((v-u for u, v in zip(seq, seq[1:])), default=None)
        residual = {rid: q[rid]["completion_s"]-fork if q[rid]["finished"] else None for rid in positions}
        head = q["high-long-10"]; first = head["token_times_s"][0] if head["token_times_s"] else None
        gapstats, flowstats = native.describe(list(gaps.values())), native.describe(list(residual.values()))
        result["short_window"] = {"engine_step_range_inclusive": [a, a+31], "start_s": fork, "end_s": end,
            "incumbent_request_maxgaps_s": gaps, "incumbent_output_counts": coverage, "all_58_have_32_outputs": len(coverage) == 58 and all(n == 32 for n in coverage.values()),
            "incumbent_request_maxgap_statistics_s": gapstats, "head_ttft_s": first-head["arrival_s"] if first is not None else None,
            "head_fork_to_first_output_s": first-fork if first is not None else None, "residual_completion_statistics_s": flowstats,
            "residual_completion_by_request_s": residual}
        action_checks["fixed_window_incumbent_coverage"] = result["short_window"]["all_58_have_32_outputs"]
        result["values"] = {"service_U": report["summary"]["joint_slo_requests_per_s"], "residual_mean_s": flowstats["mean"],
            "residual_p95_s": flowstats["p95"], "incumbent_maxgap_p95_s": gapstats["p95"], "head_ttft_s": result["short_window"]["head_ttft_s"]}
        result["timing_accounting"] = {"prefix_capture_s": raw.get("prefix_capture_s"), "explicit_arrival_waits": raw.get("explicit_arrival_waits"),
            "arrival_delay_s": {rid: r["add_s"]-r["arrival_s"] for rid, r in q.items()}, "reference_add_drift_s": {rid: r["add_s"]-r["reference_add_s"] for rid, r in q.items()}}
        result["eligible"] = report["summary"]["valid"] and report["summary"]["all_finished"] and all(refchecks.values()) and all(action_checks.values())
        if not result["eligible"]: result["errors"].append("Full-service validity, reference-prefix qualification or action assertion failed; retain run as ineligible.")
    except Exception as exc:
        result["errors"].append(type(exc).__name__ + ": " + str(exc))
    result["status"] = "eligible" if result["eligible"] else "fail_or_ineligible"
    return result


def compare_runs(early, late, anchor, repeat):
    hashes_match = bool(early and late and early.get("prefix_hashes")) and early["prefix_hashes"] == late.get("prefix_hashes")
    eligible = hashes_match and early["eligible"] and late["eligible"]
    a, b = (early or {}).get("values", {}), (late or {}).get("values", {})
    deltas = {k: percent(a.get(k), b.get(k)) if k == "service_U" else (1000*(a[k]-b[k]) if a.get(k) is not None and b.get(k) is not None else None) for k in THRESHOLDS}
    def passed(run): return {q["request_id"] for q in (run or {}).get("full_service", {}).get("requests", []) if q["joint_slo_pass"]}
    ah, bh = (early or {}).get("final_output_sha256_by_request"), (late or {}).get("final_output_sha256_by_request")
    changed = sorted(rid for rid in ah if ah[rid] != bh[rid]) if ah and bh and ah.keys() == bh.keys() else None
    return {"anchor": anchor, "repeat": repeat, "eligible": bool(eligible), "pair_prefix_qualified": bool(hashes_match),
        "both_reference_prefix_qualified": bool(early and late and early.get("reference_prefix_qualified") and late.get("reference_prefix_qualified")),
        "E_run": (early or {}).get("raw_path"), "L_run": (late or {}).get("raw_path"), "E_minus_L_deltas_percent_or_ms": deltas,
        "E_added_qualified_ids": sorted(passed(early)-passed(late)), "E_lost_qualified_ids": sorted(passed(late)-passed(early)),
        "final_output_different_ids": changed, "final_output_different_count": len(changed) if changed is not None else None,
        "output_difference_note": "Final token-ID hashes only, not quality. With qualified equal prefixes, differences occur after the fork; retain them without disqualifying post-treatment samples. No bitwise KV equality claim."}


def judge(anchor, index, pairs):
    out = {}
    for metric, floor in THRESHOLDS.items():
        rows = [index.get(f"s{anchor}_{order}_r{r}") for order in ("hl", "lh") for r in (1, 2)]
        values = [(row or {}).get("values", {}).get(metric) for row in rows]
        effects = [p["E_minus_L_deltas_percent_or_ms"][metric] for p in pairs]
        noise = None; direction = "inconclusive"
        if len(pairs) == 2 and all(p["eligible"] for p in pairs) and all(v is not None for v in values+effects) and (metric != "service_U" or all(v > 0 for v in values)):
            noise = max(abs(percent(values[i], values[i+1])) if metric == "service_U" else 1000*abs(values[i]-values[i+1]) for i in (0, 2))
            if all(v >= floor and abs(v) > 2*noise for v in effects): direction = "E" if metric == "service_U" else "L"
            if all(v <= -floor and abs(v) > 2*noise for v in effects): direction = "L" if metric == "service_U" else "E"
        out[metric] = {"direction": direction, "two_effects_percent_or_ms": effects, "engineering_floor_percent_or_ms": floor,
            "same_branch_repeat_drift_percent_or_ms": noise, "below_engineering_floor_both": all(v is not None and abs(v) < floor for v in effects),
            "interpretation": "n=2 local screening, not significance/equivalence. Head advancement alone is manipulation evidence, not action value."}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("run_directory", type=Path); parser.add_argument("--output", type=Path)
    args = parser.parse_args(); ledger = json.loads((ROOT/"frozen_prefix.json").read_text())
    design_path = ROOT/"design.json"; design_sha = file_sha(design_path) if design_path.exists() else None
    paths = [args.run_directory] if args.run_directory.is_file() else sorted(args.run_directory.rglob("raw.json"))
    rows = [analyze_run(p, ledger, design_sha) for p in paths]; warm = [r for r in rows if r.get("warmup")]; rows = [r for r in rows if not r.get("warmup")]
    index = {}; duplicates = []
    for row in rows:
        name = row.get("policy")
        if name in index:
            duplicates.append(name); index[name]["eligible"] = row["eligible"] = False
            index[name]["status"] = row["status"] = "fail_or_ineligible"
            index[name]["errors"].append("Duplicate formal name; no attempt selected."); row["errors"].append("Duplicate formal name; no attempt selected.")
        else: index[name] = row
    pairs = [compare_runs(index.get(f"s{a}_hl_r{r}"), index.get(f"s{a}_lh_r{r}"), a, r) for a in (291, 293) for r in (1, 2)]
    design = json.loads(design_path.read_text()) if design_path.exists() else None
    campaign_complete = len(rows) == 8 and set(index) == set(NAMES) and not duplicates and bool(design) and [r.get("policy") for r in rows] == design.get("policies")
    if not campaign_complete:
        for pair in pairs: pair["eligible"] = False; pair["campaign_issue"] = "Require exactly eight formal names in frozen order, no duplicate/extra attempts."
    judgments = {str(a): judge(a, index, [p for p in pairs if p["anchor"] == a]) for a in (291, 293)}
    directions = [judgments[str(a)]["service_U"]["direction"] for a in (291, 293)]
    result = {"schema": "d-phase-probe-analysis-v1", "diagnostic_only_not_normal_arrival_E2E": True, "runs": rows, "excluded_warmup_paths": [r["raw_path"] for r in warm],
        "missing_formal_names": sorted(set(NAMES)-set(index)), "duplicate_formal_names": duplicates, "campaign_names_and_order_valid": bool(campaign_complete), "pairs": pairs, "judgments": judgments,
        "complete_eligible_formal_runs": sum(r["eligible"] for r in rows), "service_preference_reversal_observed": sorted(directions) == ["E", "L"],
        "conclusion": "Local service-preference reversal observed; independent contribution not established." if sorted(directions) == ["E", "L"] else "Inconclusive / does not support independent action value; do not construct X from this probe.",
        "rule": "Two eligible same-anchor pairs; both effects beyond fixed engineering floor and >2x same-branch repeat drift. U is old qualified-count/full-elapsed; only U sets service preference. No extra repeats or significance/equivalence claim.",
        "thresholds_percent_for_U_ms_otherwise": THRESHOLDS, "frozen_design": design}
    output = args.output or (args.run_directory if args.run_directory.is_dir() else args.run_directory.parent)/"probe_results.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    print(f"{len(rows)} formal records, {result['complete_eligible_formal_runs']} eligible; " + result["conclusion"])


if __name__ == "__main__": main()
