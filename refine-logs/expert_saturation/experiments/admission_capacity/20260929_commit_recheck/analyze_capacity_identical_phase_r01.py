#!/usr/bin/env python3
"""Locate 5-second service differences in the completed capacity A/A pair."""
from __future__ import annotations

import argparse
from bisect import bisect_right
import json
import math
from pathlib import Path

import audit_h128_guarded_transfer_r02 as common


SESSION_NAME = "moe-a-capacity-identical-session-r01-20261001"
WARM_SESSION_NAME = "moe-a-capacity-warm-identical-session-r01-20261001"
ARMS = ("capacity_reference_first_on", "capacity_reference_second_on")


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def timeline(raw: dict):
    rows = raw["requests"]
    output_events = sorted(raw["output_events"], key=lambda e: e["received_s"])
    output_times = [e["received_s"] for e in output_events]
    output_prefix = [0]
    for event in output_events:
        output_prefix.append(output_prefix[-1] + event["chunk_size"])
    preemption_times = sorted(e["method_returned_s"] for e in raw["preemption_events"]
                              if e["original_preemption_returned"])
    arrivals = sorted(r["arrival_s"] for r in rows)
    completions = sorted(r["completion_s"] for r in rows)
    require(output_prefix[-1] == sum(len(r["output_token_ids"]) for r in rows) and
            len(preemption_times) == raw["actual_preemption_count"],
            "sparse output/preemption totals differ")
    def cumulative(end_s: float) -> dict:
        return {"arrived": bisect_right(arrivals, end_s),
                "output_tokens": output_prefix[bisect_right(output_times, end_s)],
                "completed": bisect_right(completions, end_s),
                "native_preemptions": bisect_right(preemption_times, end_s)}
    return cumulative, preemption_times


def first_sequence_divergence(first: dict, second: dict,
                              first_preempt: float, second_preempt: float) -> tuple[dict, set[str]]:
    first_rows = {r["request_id"]: r for r in first["requests"]}
    second_rows = {r["request_id"]: r for r in second["requests"]}
    require(first_rows.keys() == second_rows.keys() and len(first_rows) == 128,
            "A/A request identities differ")
    candidates = []
    different = set()
    for rid, a in first_rows.items():
        b = second_rows[rid]
        require(a["arrival_s"] == b["arrival_s"] and
                a["prompt_token_ids_sha256"] == b["prompt_token_ids_sha256"] and
                a["prompt_tokens"] == b["prompt_tokens"],
                f"{rid}: input differs between A/A cells")
        left, right = a["output_token_ids"], b["output_token_ids"]
        if left == right:
            continue
        different.add(rid)
        mismatch = next((i for i, (x, y) in enumerate(zip(left, right)) if x != y),
                        min(len(left), len(right)))
        left_time = a["token_times_s"][mismatch] if mismatch < len(left) else a["completion_s"]
        right_time = b["token_times_s"][mismatch] if mismatch < len(right) else b["completion_s"]
        candidates.append({
            "request_id": rid, "arrival_s": a["arrival_s"],
            "prompt_tokens": a["prompt_tokens"],
            "prompt_token_ids_sha256": a["prompt_token_ids_sha256"],
            "first_different_token_index_zero_based": mismatch,
            "first_token_id": left[mismatch] if mismatch < len(left) else None,
            "second_token_id": right[mismatch] if mismatch < len(right) else None,
            "first_token_or_completion_s": left_time,
            "second_token_or_completion_s": right_time,
            "observable_in_both_runs_by_s": max(left_time, right_time),
        })
    require(candidates, "no output sequence divergence to locate")
    earliest = min(candidates, key=lambda x: (x["observable_in_both_runs_by_s"], x["request_id"]))
    earliest["first_native_preemption_s"] = first_preempt
    earliest["second_native_preemption_s"] = second_preempt
    earliest["before_either_native_preemption"] = (
        earliest["observable_in_both_runs_by_s"] < min(first_preempt, second_preempt))
    return earliest, different


def analyze(session: Path) -> dict:
    require(session.is_dir() and session.name in (SESSION_NAME, WARM_SESSION_NAME, "moe-a-capacity-warm-identical-session-r02-20261001"), "wrong capacity A/A session")
    raws, source = {}, {}
    for index, arm in enumerate(ARMS):
        path = session / f"cell-{index:02d}-{arm}" / "archive" / "raw.json"
        raw = common.read(path)
        require(raw.get("status") == "COMPLETE" and raw.get("error") is None and
                len(raw.get("requests", [])) == 128 and
                all(r["status"] == "completed" for r in raw["requests"]),
                f"{arm}: incomplete raw")
        label = ("first", "second")[index]
        raws[label] = raw
        source[label] = {"raw_path": str(path), "raw_sha256": common.sha_file(path),
                         "observation_end_s": raw["observation_end_s"],
                         "engine_call_count": raw["engine_call_count"]}
    first_timeline, first_preempts = timeline(raws["first"])
    second_timeline, second_preempts = timeline(raws["second"])
    end = 5 * math.ceil(max(r["observation_end_s"] for r in raws.values()) / 5)
    bins = []
    previous = {label: {key: 0 for key in
               ("arrived", "output_tokens", "completed", "native_preemptions")}
               for label in ("first", "second")}
    for end_s in range(5, end + 1, 5):
        counts = {"first": first_timeline(end_s), "second": second_timeline(end_s)}
        bins.append({"window_s": [end_s - 5, end_s],
                     "cumulative": counts,
                     "window_increment": {label: {key: value - previous[label][key]
                                                 for key, value in count.items()}
                                          for label, count in counts.items()},
                     "second_minus_first_cumulative": {
                         key: counts["second"][key] - counts["first"][key]
                         for key in counts["first"]}})
        previous = counts
    earliest, different = first_sequence_divergence(
        raws["first"], raws["second"], first_preempts[0], second_preempts[0])
    first_rows = {r["request_id"]: r for r in raws["first"]["requests"]}
    second_rows = {r["request_id"]: r for r in raws["second"]["requests"]}
    cohorts = []
    for start in range(0, 30, 5):
        ids = [rid for rid, r in first_rows.items() if start <= r["arrival_s"] < start + 5]
        if not ids:
            continue
        deltas = [second_rows[rid]["completion_s"] - first_rows[rid]["completion_s"]
                  for rid in ids]
        cohorts.append({"arrival_window_s": [start, start + 5], "requests": len(ids),
                        "different_output_sequences": sum(rid in different for rid in ids),
                        "mean_second_minus_first_completion_s": sum(deltas) / len(deltas),
                        "second_completed_earlier": sum(delta < 0 for delta in deltas)})
    return {"schema_version": 1,
            "status": "CAPACITY_IDENTICAL_PHASES_DESCRIBED",
            "source": source,
            "five_second_bins": bins,
            "first_output_sequence_divergence": earliest,
            "different_output_sequence_requests": len(different),
            "completion_by_arrival_cohort": cohorts,
            "limits": [
                "Times are relative to each cell's own measurement origin; bins compare elapsed time, not simultaneous wall clocks.",
                "Performance raw records engine call indices and host return times for output events, but omits timing for calls without output; per-call latency by phase is unavailable.",
                "The first content difference and service curves locate divergence; they do not identify a cache, GPU, or policy cause.",
            ]}


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
                      "first_divergence": result["first_output_sequence_divergence"]["request_id"]},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
