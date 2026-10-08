"""Request-level goodput for one frozen capture, with all arrivals in the cohort.

This reads native_capture raw.json files. It does not infer action causality or
equivalent generated work from a paired run.
"""

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean, median


TTFT_DEADLINES_S = (10, 20, 30, 40)
GAP_DEADLINES_S = (1, 2, 4, 8, 12)


def quantile(values, q):
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * q
    lo = int(position)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (position - lo) * (values[hi] - values[lo])


def summarize(raw, expected_requests, timeout_s):
    duration = raw["observation_end_s"]
    requests = raw["requests"]
    if duration <= 0 or timeout_s <= 0 or len(requests) != expected_requests:
        raise ValueError("invalid duration, timeout or measurement cohort size")
    rows = []
    for request in requests:
        arrival = request["arrival_s"]
        times = request["token_times_s"]
        tokens = request["output_token_ids"]
        if (not 0 <= arrival <= duration or len(times) != len(tokens)
                or any(t < arrival or t > duration for t in times)
                or any(b < a for a, b in zip(times, times[1:]))):
            raise ValueError(f"invalid output clock for {request['request_id']}")
        completed = request["status"] == "completed"
        end = request.get("completion_s")
        if completed and (end is None or not arrival <= end <= duration):
            raise ValueError(f"missing/invalid completion for {request['request_id']}")
        unique_times = sorted(set(times))
        gap = max((b - a for a, b in zip(unique_times, unique_times[1:])), default=None)
        ttft = times[0] - arrival if times else None
        flow = end - arrival if completed else max(timeout_s, duration - arrival)
        rows.append(dict(
            request_id=request["request_id"], document_id=request.get("document_id"),
            prompt_sha256=request.get("prompt_token_ids_sha256"), arrival_s=arrival,
            max_output=request.get("max_output_tokens"), status=request["status"],
            stop_reason=request.get("stop_reason"), completed=completed,
            outputs=len(tokens), ttft_s=ttft, max_gap_s=gap,
            flow_s=flow, actual_completion_flow_s=end - arrival if completed else None,
        ))
    ids = [r["request_id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate request IDs")
    frontier = []
    for first_deadline in TTFT_DEADLINES_S:
        for gap_deadline in GAP_DEADLINES_S:
            count = sum(
                r["completed"] and r["ttft_s"] is not None
                and r["ttft_s"] <= first_deadline
                and (r["max_gap_s"] is None or r["max_gap_s"] <= gap_deadline)
                for r in rows
            )
            frontier.append(dict(ttft_deadline_s=first_deadline, gap_deadline_s=gap_deadline,
                                 qualifying_requests=count, goodput_requests_s=count / duration))
    gaps = [r["max_gap_s"] for r in rows if r["max_gap_s"] is not None]
    ttfts = [r["ttft_s"] for r in rows if r["ttft_s"] is not None]
    completed_flows = [r["actual_completion_flow_s"] for r in rows if r["completed"]]
    return dict(
        status=raw.get("status"), duration_s=duration, expected_requests=expected_requests,
        completed=sum(r["completed"] for r in rows),
        failed=sum(str(r["status"]).lower() in ("failed", "rejected", "aborted", "error", "cancelled")
                   for r in rows),
        unfinished=sum(not r["completed"] and str(r["status"]).lower()
                       not in ("failed", "rejected", "aborted", "error", "cancelled") for r in rows),
        total_output_tokens=sum(r["outputs"] for r in rows),
        actual_output_tokens_s=sum(r["outputs"] for r in rows) / duration,
        completed_requests_s=sum(r["completed"] for r in rows) / duration,
        mean_flow_with_incomplete_penalty_s=mean(r["flow_s"] for r in rows),
        mean_completed_flow_s=mean(completed_flows) if completed_flows else None,
        ttft_median_s=median(ttfts) if ttfts else None,
        ttft_p90_s=quantile(ttfts, .9),
        max_gap_request_median_s=median(gaps) if gaps else None,
        max_gap_request_p90_s=quantile(gaps, .9),
        max_gap_request_p95_s=quantile(gaps, .95),
        max_gap_request_max_s=max(gaps, default=None),
        undefined_gap_requests=len(rows) - len(gaps),
        frontier=frontier, requests=rows,
    )


def pair(reference, candidate):
    identity = lambda r: (r["request_id"], r["document_id"], r["prompt_sha256"],
                          r["arrival_s"], r["max_output"])
    ref = {r["request_id"]: r for r in reference["requests"]}
    cand = {r["request_id"]: r for r in candidate["requests"]}
    if set(ref) != set(cand) or any(identity(ref[rid]) != identity(cand[rid]) for rid in ref):
        raise ValueError("paired capture cohorts differ")
    differences = []
    for rid in sorted(ref):
        a, b = ref[rid], cand[rid]
        delta = lambda key: b[key] - a[key] if a[key] is not None and b[key] is not None else None
        differences.append(dict(request_id=rid, completed_reference=a["completed"],
                                completed_candidate=b["completed"],
                                output_difference=b["outputs"] - a["outputs"],
                                flow_difference_s=delta("flow_s"),
                                ttft_difference_s=delta("ttft_s"),
                                max_gap_difference_s=delta("max_gap_s")))
    return dict(
        actual_output_rate_ratio=candidate["actual_output_tokens_s"] / reference["actual_output_tokens_s"]
        if reference["actual_output_tokens_s"] else None,
        mean_flow_ratio=candidate["mean_flow_with_incomplete_penalty_s"]
        / reference["mean_flow_with_incomplete_penalty_s"],
        frontier=[dict(ttft_deadline_s=a["ttft_deadline_s"], gap_deadline_s=a["gap_deadline_s"],
                       goodput_difference_requests_s=b["goodput_requests_s"] - a["goodput_requests_s"],
                       qualifying_difference=b["qualifying_requests"] - a["qualifying_requests"])
                  for a, b in zip(reference["frontier"], candidate["frontier"])],
        per_request=differences,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True, help="candidate raw.json")
    parser.add_argument("--reference", type=Path, help="paired reference raw.json")
    parser.add_argument("--expected-requests", type=int, required=True)
    parser.add_argument("--timeout-s", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.expected_requests <= 0:
        parser.error("expected requests must be positive")
    read = lambda path: json.loads(path.read_text())
    result = dict(schema_version=1, semantics=dict(
        cohort="all measurement requests, including failed/rejected/unfinished",
        denominator="observation_end_s from measurement start through capture end",
        delivery="host engine-return token timestamps; not client acknowledgements",
        incomplete_flow="max(fixed timeout, elapsed observation after arrival)",
        no_observed_gap="gap undefined in distribution when all outputs share one host-return timestamp; gap condition vacuously true for a completed request with output",
        thresholds="fixed development frontier, not production SLO",
        work="output length and stop reason can differ; timing is not equal-work speedup"),
        candidate_sha256=hashlib.sha256(args.candidate.read_bytes()).hexdigest(),
        candidate=summarize(read(args.candidate), args.expected_requests, args.timeout_s))
    if args.reference:
        result["reference_sha256"] = hashlib.sha256(args.reference.read_bytes()).hexdigest()
        result["reference"] = summarize(read(args.reference), args.expected_requests, args.timeout_s)
        result["paired"] = pair(result["reference"], result["candidate"])
    with args.output.open("x") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
