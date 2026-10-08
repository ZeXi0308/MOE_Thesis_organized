"""Read native request ledgers; partition observed pauses without GPU attribution."""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import json
import math
from pathlib import Path


def read_raw(path):
    path = Path(path)
    with (gzip.open(path, "rt") if path.suffix == ".gz" else path.open()) as stream:
        return json.load(stream)


def analyze(raw):
    requests = raw["requests"]
    by_id = {r["request_id"]: r for r in requests}
    mapping = raw["internal_to_source"]
    if len(by_id) != len(requests) or len(set(mapping.values())) != len(mapping):
        raise ValueError("duplicate request identity")
    for internal, source in mapping.items():
        if source not in by_id or by_id[source].get("internal_request_id") != internal:
            raise ValueError("internal/source request identity mismatch")
    steps = raw["scheduler_steps"]
    if [s["step"] for s in steps] != list(range(len(steps))):
        raise ValueError("scheduler steps must be unique contiguous ledger indices")
    owners, recompute, first_service = {}, {rid: [] for rid in by_id}, {}
    for call in raw["engine_steps"]:
        begin, end = call["scheduler_step_start"], call["scheduler_step_end"]
        if not 0 <= begin <= end <= len(steps):
            raise ValueError("engine call references absent scheduler steps")
        for index in range(begin, end):
            if index in owners:
                raise ValueError("scheduler step belongs to multiple engine calls")
            owners[index] = call
        if call["completed"] and not 0 <= call["start_s"] <= call["returned_s"] <= raw["observation_end_s"]:
            raise ValueError("invalid completed engine-call interval")
    for step in steps:
        call = owners.get(step["step"])
        if call is None:
            raise ValueError("scheduler step has no engine-call owner")
        seen = set()
        for item in step["scheduled"]:
            rid = item["request_id"]
            if mapping.get(item["internal_request_id"]) != rid or rid in seen:
                raise ValueError("scheduled request identity mismatch or duplicate")
            seen.add(rid)
            if call["completed"]:
                first_service.setdefault(rid, step["start_s"])
                if item["recompute_tokens"] > 0:
                    recompute[rid].append((step["step"], call))
    victims = Counter()
    for event in raw.get("preemption_events", []):
        internal = event["victim_internal_request_id"]
        if internal not in mapping:
            raise ValueError("preemption victim is absent from request ledger")
        victims[mapping[internal]] += 1
    result = []
    for req in requests:
        rid, times = req["request_id"], req["token_times_s"]
        if (len(times) != len(req["output_token_ids"])
                or any(not math.isfinite(t) or not req["arrival_s"] <= t <= raw["observation_end_s"] for t in times)
                or any(b < a for a, b in zip(times, times[1:]))):
            raise ValueError("unaligned tokens or invalid receipt times")
        row = dict(request_id=rid, status=req["status"], arrival_s=req["arrival_s"],
                   arrived=req["arrival_s"] <= raw["observation_end_s"],
                   submission_s=req.get("admission_s"), first_service_s=first_service.get(rid),
                   completion_s=req.get("completion_s"),
                   output_tokens=len(times), preemption_events=victims[rid],
                   ttft_s=times[0] - req["arrival_s"] if times else None,
                   max_itl_s=None, partition=None)
        if len(times) >= 2:
            index = max(range(len(times) - 1), key=lambda i: times[i + 1] - times[i])
            start, end = times[index:index + 2]
            row.update(max_itl_s=end - start, gap_start_s=start, gap_end_s=end)
            in_gap = [(s, c) for s, c in recompute[rid]
                      if c["start_s"] < end and c["returned_s"] > start]
            if (victims[rid] and in_gap
                    and all(start <= c["start_s"] <= c["returned_s"] <= end for _, c in in_gap)):
                first = min(c["start_s"] for _, c in in_gap)
                last = max(c["returned_s"] for _, c in in_gap)
                row["partition"] = dict(before_first_recompute_call_s=first - start,
                    recompute_calls_span_s=last - first, after_last_recompute_call_s=end - last,
                    recompute_step_ids=[s for s, _ in in_gap])
        result.append(row)
    return dict(scope="Observed host receipt gaps; recompute-call span includes concurrent requests "
                      "and instrumentation, not pure GPU recompute time. Do not sum overlapping "
                      "victim gaps or add these spans again to request latency.",
                episode_status=raw["status"], episode_error=raw.get("error"),
                observation_end_s=raw["observation_end_s"], n_planned=len(result),
                n_arrived=sum(r["arrived"] for r in result),
                request_status_counts=dict(Counter(r["status"] for r in result)),
                preemption_events=sum(victims.values()),
                partition_unavailable_for_preempted_requests=[r["request_id"] for r in result
                    if r["preemption_events"] and r["partition"] is None], requests=result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = analyze(read_raw(args.raw))
    result["source_raw"] = str(args.raw.resolve())
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({k: result[k] for k in ("episode_status", "n_planned", "n_arrived",
                                            "request_status_counts", "preemption_events")}))


if __name__ == "__main__":
    main()
