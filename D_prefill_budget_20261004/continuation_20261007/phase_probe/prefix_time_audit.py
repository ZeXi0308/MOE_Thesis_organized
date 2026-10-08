#!/usr/bin/env python3
"""Read-only timing supplement: python3 -B prefix_time_audit.py RUN_DIRECTORY > FILE.json.
No prefix subtraction from the primary denominator; no eligibility or preference decisions.
"""
import argparse
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True
import analyze_probe as frozen
native = frozen.native


def audit(path):
    row = {"raw_path": str(path), "status": "error"}
    try:
        raw = json.loads(path.read_text()); name = raw["policy"]; row["policy"] = name
        if native.is_warmup_path(path) or name.endswith("_warm"): return {**row, "status": "warmup"}
        if name not in frozen.NAMES: raise ValueError("Unexpected formal name; retained without pairing")
        a = raw["anchor"]; steps = raw["steps"]; fork = steps[a]["start_s"]
        state = raw["prefix_state"]; lengths = state["output_prefix_lengths"]
        active = {r["request_id"]: r for r in state["request_states"]}
        report = native.summarize(path, native.find_protocol(path, None), path.parent.name)
        slo = report["slo"]
        if slo != {"ttft_s": 4, "gap_s": .1, "completion_s": 20}: raise ValueError("Unexpected primary SLO")
        timing = {}
        for r in raw["requests"]:
            rid = r["request_id"]; n = lengths[rid]; times = r["token_times_s"]
            if not isinstance(n, int) or not 0 <= n <= len(times) or n != sum(t < fork for t in times):
                raise ValueError("Invalid prefix length/timestamp: " + rid)
            prefix = times[:n]; age = fork-r["arrival_s"] if r["arrival_s"] <= fork else None; last = prefix[-1] if n else None
            gaps = [y-x for x, y in zip(prefix, prefix[1:])]; seen_gap = max(gaps, default=0)
            done = r.get("completion_s") is not None and r["completion_s"] <= fork
            observed, unavoidable = [], []
            if n and prefix[0]-r["arrival_s"] > slo["ttft_s"]: observed.append("ttft_s")
            if seen_gap > slo["gap_s"]: observed.append("gap_s")
            if done and r["completion_s"]-r["arrival_s"] > slo["completion_s"]: observed.append("completion_s")
            if not n and age is not None and age > slo["ttft_s"]: unavoidable.append("ttft_s")
            if not done and age is not None and age > slo["completion_s"]: unavoidable.append("completion_s")
            if not done and 0 < n < r["max_tokens"] and fork-last > slo["gap_s"]: unavoidable.append("gap_s")
            st = active.get(rid); prefill = bool(st and st["computed_tokens"] < st["prompt_tokens"])
            decoder = bool(st and rid in state["running_ids"] and not prefill)
            timing[rid] = {"prefix_output_count": n, "arrival_age_s": age,
                "admitted_age_s": fork-r["add_s"] if r.get("add_s") is not None and r["add_s"] <= fork else None,
                "prefill_pending": prefill, "incumbent_decoder": decoder, "last_prefix_output_s": last,
                "decoder_wait_s": fork-last if decoder and n else None,
                "next_token_slack_s": slo["gap_s"]-(fork-last) if decoder and n else None,
                "fork_to_next_output_s": times[n]-fork if decoder and n < len(times) else None,
                "cross_fork_observed_output_gap_s": times[n]-last if decoder and 0 < n < len(times) else None,
                "prefill_ttft_slack_s": slo["ttft_s"]-age if prefill and not n and age is not None else None,
                "fork_to_first_output_s": times[0]-fork if prefill and not n and times else None,
                "prefix_max_observed_gap_s": seen_gap, "observed_failure_reasons": observed,
                "already_unavoidable_failure_reasons": unavoidable, "irreparable": bool(observed or unavoidable)}
        passed = sorted(r["request_id"] for r in report["requests"] if r["joint_slo_pass"])
        engine = sum(s["end_s"]-s["start_s"] for s in steps[:a])
        late_ttft = sorted(rid for rid, t in timing.items() if "ttft_s" in t["already_unavoidable_failure_reasons"])
        head = timing.get("high-long-10", {})
        row.update(status="complete" if report["summary"]["valid"] and report["summary"]["all_finished"] else "invalid_or_incomplete",
            validation_errors=report["validation_errors"], fork_s=fork, elapsed_s=raw["elapsed_s"], residual_episode_s=raw["elapsed_s"]-fork,
            prefix_engine_step_s=engine, prefix_non_engine_s=fork-engine, prefix_capture_s=raw.get("prefix_capture_s"),
            prefix_explicit_wait_s=sum(w["elapsed_s"] for w in raw.get("explicit_arrival_waits", []) if w["before_step"] <= a),
            qualified_count=len(passed), qualified_ids=passed, service_U=report["summary"]["joint_slo_requests_per_s"],
            no_first_output_age_gt_ttft_count=len(late_ttft), no_first_output_age_gt_ttft_ids=late_ttft,
            head_arrival_age_s=head.get("arrival_age_s"), head_ttft_already_unavoidable="ttft_s" in head.get("already_unavoidable_failure_reasons", []),
            irreparable_ids=sorted(rid for rid, t in timing.items() if t["irreparable"]), request_prefix_timing=timing)
    except Exception as exc: row["error"] = type(exc).__name__ + ": " + str(exc)
    return row


def compare(e, l):
    if not e or not l or any(r["status"] != "complete" for r in (e, l)):
        return {"status": "missing_duplicate_invalid_or_incomplete; no supplemental comparison"}
    ne, nl, te, tl = e["qualified_count"], l["qualified_count"], e["elapsed_s"], l["elapsed_s"]
    df = e["fork_s"]-l["fork_s"]; dr = e["residual_episode_s"]-l["residual_episode_s"]
    fields = ("arrival_age_s", "admitted_age_s", "decoder_wait_s", "next_token_slack_s", "prefill_ttft_slack_s", "fork_to_next_output_s", "cross_fork_observed_output_gap_s", "fork_to_first_output_s")
    eq, lq = e["request_prefix_timing"], l["request_prefix_timing"]
    return {"status": "descriptive_only; consult frozen analyzer for prefix eligibility", "E": e["raw_path"], "L": l["raw_path"],
        "same_qualified_count": ne == nl, "same_qualified_ids": e["qualified_ids"] == l["qualified_ids"],
        "same_count_U_difference_is_denominator_only": ne == nl,
        "E_minus_L_elapsed_s": te-tl, "E_minus_L_fork_s": df, "E_minus_L_residual_episode_s": dr,
        "U_difference": e["service_U"]-l["service_U"], "U_algebraic_terms_not_causal": {
            "numerator_change_at_L_denominator": (ne-nl)/tl,
            "prefix_time_difference_term": -ne*df/(te*tl), "residual_time_difference_term": -ne*dr/(te*tl)},
        "E_only_irreparable_ids": sorted(set(e["irreparable_ids"])-set(l["irreparable_ids"])),
        "L_only_irreparable_ids": sorted(set(l["irreparable_ids"])-set(e["irreparable_ids"])),
        "E_minus_L_request_prefix_timing_s": {rid: {k: eq[rid][k]-lq[rid][k] if eq[rid][k] is not None and lq[rid][k] is not None else None
            for k in fields} for rid in sorted(eq.keys() & lq.keys())}, "request_id_sets_match": eq.keys() == lq.keys()}


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("run_directory", type=Path); args = parser.parse_args()
    paths = [args.run_directory] if args.run_directory.is_file() else sorted(args.run_directory.rglob("raw.json"))
    rows = [audit(p) for p in paths]; formal = [r for r in rows if r["status"] != "warmup"]
    def unique(name):
        matches = [r for r in formal if r.get("policy") == name]
        return matches[0] if len(matches) == 1 else None
    result = {"schema": "d-prefix-time-supplement-v1", "formal_records": len(formal),
        "complete_records": sum(r["status"] == "complete" for r in formal), "runs": formal,
        "excluded_warmup_paths": [r["raw_path"] for r in rows if r["status"] == "warmup"],
        "pairs": {f"s{a}_r{r}": compare(unique(f"s{a}_hl_r{r}"), unique(f"s{a}_lh_r{r}")) for a in (291, 293) for r in (1, 2)},
        "boundary": "No new eligibility, thresholds, rescoring or run exclusion. U remains qualified/full elapsed; residual time is diagnostic only. Algebra is not causal attribution. Logical prefix equality does not imply equal wall-clock age/slack or GPU/cache/thermal state. Advancing a head already beyond its TTFT deadline cannot restore its primary-SLO eligibility."}
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__": main()
