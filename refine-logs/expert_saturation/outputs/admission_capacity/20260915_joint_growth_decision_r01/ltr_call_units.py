#!/usr/bin/env python3
"""Describe LTR counter units on G's observed eager diagnostic, not a policy replay.

Only G inputs are accepted here by default. No H input, counter emulation,
candidate service ranking, future length feature, or GPU execution is involved.
"""
import argparse
import bisect
import collections
import json
import pathlib
import statistics

DEFAULT_G = pathlib.Path(__file__).parent.parent / (
    "20260915_natural_recovery_cadence_r01/execution_weste_26862"
)


def stats(values):
    ordered = sorted(values)
    if not ordered:
        return {"n": 0}

    def percentile(p):
        x = p * (len(ordered) - 1)
        low = int(x)
        return ordered[low] + (x - low) * (ordered[min(low + 1, len(ordered) - 1)] - ordered[low])

    return dict(n=len(ordered), min=ordered[0], p10=percentile(.1),
                median=statistics.median(ordered), p90=percentile(.9), max=ordered[-1])


def analyze(g):
    main = json.loads((g / "analysis.json").read_text())
    diag = main["diagnostic_combination"]["diagnostic"]
    raw = json.loads((g / "readback/results/diagnostic-eager/raw.json").read_text())
    schedules = raw["scheduler_steps"]
    engines = raw["engine_steps"]
    # These are unit-alignment requirements for the calculation, not a new audit.
    assert all(e["completed"] and e["scheduler_step_end"] - e["scheduler_step_start"] == 1
               for e in engines)
    engine = {e["scheduler_step_start"]: e for e in engines}
    assert len(engine) == len(schedules)
    schedule_start = [s["start_s"] for s in schedules]
    rows = []
    groups = collections.defaultdict(list)
    selected = collections.defaultdict(list)
    for s in schedules:
        e = engine[s["step"]]
        row = {
            "step": s["step"], "start_s": e["start_s"],
            "host_call_ms": 1000 * (e["returned_s"] - e["start_s"]),
            "active": s["actual_active"], "waiting": s["waiting_before"] > 0,
            "assigned_positions": s["total_scheduled_tokens"],
        }
        rows.append(row)
        groups["all"].append(row)
        groups["waiting_before_positive" if row["waiting"] else "waiting_before_zero"].append(row)
        active = row["active"]
        bucket = next((f"active_{lo}_{hi}" for lo, hi in ((1, 8), (9, 16), (17, 24), (25, 32))
                       if lo <= active <= hi), "active_other")
        groups[bucket].append(row)
        if s["recompute_tokens"] > 0:
            groups["contains_recompute"].append(row)
        if any(z["prefill_tokens"] > 0 for z in s["scheduled"]):
            groups["contains_prefill"].append(row)
        if s["total_scheduled_tokens"] > 0 and all(
                z["prefill_tokens"] == z["recompute_tokens"] == 0 for z in s["scheduled"]):
            groups["pure_decode"].append(row)
        for z in s["scheduled"]:
            if z["scheduled_tokens"] > 0:
                selected[z["request_id"]].append(s["step"])
    call_units = {k: {"host_call_ms": stats([r["host_call_ms"] for r in v]),
                      "assigned_positions": stats([r["assigned_positions"] for r in v])}
                  for k, v in groups.items()}
    windows = {}
    for threshold in (30, 200):
        observations = collections.defaultdict(list)
        for i in range(len(rows) - threshold):
            span = rows[i + threshold]["start_s"] - rows[i]["start_s"]
            observations["all_overlapping_windows_s"].append(span)
            # Homogeneous windows prevent calling a low-batch start a low-pressure interval.
            region = rows[i:i + threshold]
            if all(r["waiting"] for r in region):
                observations["all_calls_waiting_positive_s"].append(span)
            if all(not r["waiting"] for r in region):
                observations["all_calls_waiting_zero_s"].append(span)
        windows[str(threshold)] = {k: stats(v) for k, v in observations.items()}
    streaks = []
    for request in raw["requests"]:
        sid = request["request_id"]
        steps = selected[sid]
        first = bisect.bisect_left(schedule_start, request["engine_add_return_s"])
        candidates = [("initial", first, steps[0])]
        candidates.extend(("between_assignments", previous + 1, following)
                          for previous, following in zip(steps, steps[1:]))
        for kind, begin, end in candidates:
            if end > begin:
                streaks.append({"request_id": sid, "kind": kind,
                                "begin_step": begin, "next_assignment_step": end,
                                "unassigned_calls": end - begin,
                                "span_s": engine[end]["start_s"] - engine[begin]["start_s"]})
    streak_summary = {}
    for kind in ("initial", "between_assignments"):
        subset = [s for s in streaks if s["kind"] == kind]
        streak_summary[kind] = {
            "unassigned_calls": stats([s["unassigned_calls"] for s in subset]),
            "span_s": stats([s["span_s"] for s in subset]),
            "observed_streaks_reaching_threshold": {
                str(t): sum(s["unassigned_calls"] >= t for s in subset) for t in (30, 200)},
        }
    recoveries = []
    for segment in diag["segments"]:
        first_output = segment["F_s"]
        assignments = segment["executed_steps"]
        through_output = [s for s in assignments if first_output is not None
                          and engine[s["step"]]["returned_s"] <= first_output + 1e-8]
        recoveries.append({
            "request_id": segment["request_id"], "preempt_step": segment["preempt_step"],
            "has_actual_load_dispatch": bool(segment["actual_load_dispatches"]),
            "first_new_output_defined": first_output is not None,
            "assigned_calls_through_first_new_output": len(through_output) if first_output is not None else None,
            "assigned_positions_through_first_new_output": sum(s["scheduled_tokens"] for s in through_output)
                if first_output is not None else None,
            "first_assigned_call_positions": assignments[0]["scheduled_tokens"] if assignments else None,
            "total_assigned_calls_in_observed_segment": len(assignments),
            "observed_segment_new_outputs": segment["useful_outputs"],
        })
    recovery_summary = {}
    for load in (True, False):
        subset = [s for s in recoveries if s["has_actual_load_dispatch"] == load]
        defined = [s for s in subset if s["first_new_output_defined"]]
        recovery_summary["actual_load" if load else "no_actual_load"] = {
            "n": len(subset), "undefined_first_new_output": len(subset) - len(defined),
            "assigned_calls_to_first_output_histogram": dict(collections.Counter(
                s["assigned_calls_through_first_new_output"] for s in defined)),
            "positions_to_first_output": stats([s["assigned_positions_through_first_new_output"] for s in defined]),
            "first_assigned_call_positions": stats([s["first_assigned_call_positions"] for s in defined]),
            "total_assigned_calls_in_observed_segment": stats([s["total_assigned_calls_in_observed_segment"] for s in subset]),
        }
    return {
        "sources": [str(g / "analysis.json"), str(g / "readback/results/diagnostic-eager/raw.json")],
        "evidence": "G observed eager diagnostic unit characterization; not LTR replay or performance comparison",
        "semantics": {
            "threshold": "consecutive scheduler calls without positive assigned tokens; not elapsed time",
            "quantum": "positive-assignment calls, including prefill/recompute; pending load alone does not spend it",
            "window": "overlapping start-to-start windows including host gaps; correlated diagnostic observations",
            "streak": "existing eager trajectory, censored by its actions; not hypothetical LTR counters or action counts",
            "feature_boundary": "IDs are joins only; observed completion/output is evaluation, never a policy feature",
            "time_boundary": "host engine calls, not scheduler Python duration, pure GPU time, or load latency",
        },
        "call_units": call_units, "fixed_call_window_spans": windows,
        "unassigned_streak_summary": streak_summary, "recovery_summary": recovery_summary,
        "unassigned_streaks": streaks, "recovery_observations": recoveries,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--g", type=pathlib.Path, default=DEFAULT_G)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    result = analyze(args.g)
    with args.output.open("x") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(json.dumps({k: result[k] for k in (
        "fixed_call_window_spans", "unassigned_streak_summary", "recovery_summary")}, indent=2))
