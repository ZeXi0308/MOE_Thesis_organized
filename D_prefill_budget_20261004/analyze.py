#!/usr/bin/env python3
"""Summarize native serving traces without third-party dependencies.

Usage: python analyze.py RUN_DIRECTORY [--protocol PATH] [--output-dir PATH]
Each recursively discovered raw.json becomes one row in summary.csv. The JSON
also retains validation details. Timestamps measure observed token emission;
they are not kernel-level or client-network token interarrival measurements.
"""

import argparse
import csv
import json
import math
import statistics
from collections import Counter
from pathlib import Path


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def percentile(values, probability):
    """Linear interpolation at (n - 1) * p; no rounding before comparison."""
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def describe(values, total=None):
    """Do not silently omit unobserved requests from headline statistics."""
    observed = [float(value) for value in values if finite(value)]
    total = len(values) if total is None else total
    missing = total - len(observed)
    result = {
        "count": total,
        "observed_count": len(observed),
        "missing_count": missing,
        "mean": statistics.fmean(observed) if observed and not missing else None,
        "p95": percentile(observed, 0.95) if not missing else None,
        "max": max(observed) if observed and not missing else None,
    }
    if missing:
        result["observed_only_mean"] = statistics.fmean(observed) if observed else None
        result["observed_only_p95"] = percentile(observed, 0.95)
        result["observed_only_max"] = max(observed) if observed else None
    return result


def avg(values):
    values = [float(value) for value in values if finite(value)]
    return statistics.fmean(values) if values else None


def distribution(values):
    counts = Counter(str(value) for value in values if value is not None)
    return dict(sorted(counts.items(), key=lambda pair: pair[0]))


def is_warmup_path(raw_path):
    return any(part.lower().startswith("warm") for part in raw_path.parent.parts)


def find_protocol(raw_path, override):
    if override:
        return override
    for parent in raw_path.parents:
        candidate = parent / "protocol.json"
        if candidate.is_file():
            return candidate
    raise ValueError("No protocol.json found above %s; pass --protocol" % raw_path)


def read_json(path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_slo(protocol):
    source = protocol["slo"]
    result = {key: source[key] for key in ("ttft_s", "gap_s", "completion_s")}
    if any(not finite(value) or value <= 0 for value in result.values()):
        raise ValueError("SLO thresholds must be finite, positive seconds")
    return result


def summarize(raw_path, protocol_path, relative_path):
    raw = read_json(raw_path)
    protocol = read_json(protocol_path)
    slo = read_slo(protocol)
    requests = raw["requests"]
    steps = raw["steps"]
    elapsed = raw["elapsed_s"]
    if not finite(elapsed) or elapsed <= 0:
        raise ValueError("%s: elapsed_s must be finite and positive" % raw_path)
    errors, warnings = [], []
    seen_ids, preempted_ids = set(), set()
    ttfts, flows, max_gaps, waits, admission_waits, scheduler_waits = [], [], [], [], [], []
    output_tokens = prompt_tokens = finished_count = qualified_count = qualified_tokens = 0
    adjacent_pairs = repeated_pairs = repeated_requests = mismatch_count = 0
    arrival_times, completion_times = [], []
    failed_by_slo = Counter()
    expected_work = {}
    actual_prefill_by_request, actual_decode_by_request = Counter(), Counter()
    output_length_mismatches = 0
    request_metrics = []

    for request in requests:
        request_id = str(request["request_id"])
        if request_id in seen_ids:
            errors.append("Duplicate request_id: " + request_id)
        seen_ids.add(request_id)
        arrival = request.get("arrival_s")
        add = request.get("add_s")
        scheduled = request.get("first_scheduled_s")
        completion = request.get("completion_s")
        times = request.get("token_times_s", [])
        ids = request.get("output_token_ids", [])
        token_count = len(ids)
        expected_work[request_id] = (request.get("prompt_tokens", 0), token_count - 1)
        if token_count != request.get("max_tokens"):
            output_length_mismatches += 1
            errors.append("Fixed output length mismatch for %s: output_tokens=%d, max_tokens=%s" % (request_id, token_count, request.get("max_tokens")))
        output_tokens += token_count
        prompt_tokens += request.get("prompt_tokens", 0)
        finished = request.get("finished") is True
        finished_count += int(finished)
        if not finished:
            errors.append("Unfinished request: " + request_id)
        timestamp_valid = all(finite(value) for value in times)
        ordered = timestamp_valid and all(a <= b for a, b in zip(times, times[1:]))
        count_matches = len(times) == token_count
        if not count_matches:
            mismatch_count += 1
            errors.append("Token/timestamp count mismatch: " + request_id)
        if not ordered:
            errors.append("Nonfinite or decreasing token timestamps: " + request_id)
        for field in ("arrival_s", "add_s", "first_scheduled_s", "completion_s"):
            timestamp = request.get(field)
            if timestamp is not None and (not finite(timestamp) or timestamp < 0 or timestamp > elapsed):
                errors.append("Request %s timestamp outside [0, elapsed_s]: %s=%s" % (request_id, field, timestamp))
        if any(not finite(timestamp) or timestamp < 0 or timestamp > elapsed for timestamp in times):
            errors.append("Output timestamp outside [0, elapsed_s]: " + request_id)
        if not finite(arrival):
            errors.append("Missing/nonfinite arrival timestamp: " + request_id)
        else:
            arrival_times.append(arrival)
        valid_output = bool(times) and ordered and count_matches and finite(arrival)
        ttft = times[0] - arrival if valid_output else None
        if ttft is not None and ttft < 0:
            errors.append("Output before arrival: " + request_id)
            valid_output = False
            ttft = None
        gaps = [b - a for a, b in zip(times, times[1:])] if ordered else []
        repeated = sum(gap == 0 for gap in gaps)
        repeated_pairs += repeated
        adjacent_pairs += len(gaps)
        repeated_requests += int(repeated > 0)
        max_gap = max(gaps, default=0.0) if valid_output else None
        flow = completion - arrival if finished and finite(completion) and finite(arrival) else None
        if flow is not None and (flow < 0 or (times and ordered and completion < times[-1])):
            errors.append("Invalid completion timestamp: " + request_id)
            flow = None
        if finished and not finite(completion):
            errors.append("Finished request lacks completion timestamp: " + request_id)
        if finite(completion):
            completion_times.append(completion)
        wait = scheduled - arrival if finite(scheduled) and finite(arrival) else None
        admission_wait = add - arrival if finite(add) and finite(arrival) else None
        scheduler_wait = scheduled - add if finite(scheduled) and finite(add) else None
        ttfts.append(ttft)
        flows.append(flow)
        max_gaps.append(max_gap)
        waits.append(wait)
        admission_waits.append(admission_wait)
        scheduler_waits.append(scheduler_wait)
        failures = []
        for key, value in (("ttft_s", ttft), ("gap_s", max_gap), ("completion_s", flow)):
            if value is None or value > slo[key]:
                failures.append(key)
                failed_by_slo[key] += 1
        if not finished:
            failures.append("unfinished")
        if token_count != request.get("max_tokens"):
            failures.append("fixed_output_length")
        qualified = not failures
        qualified_count += int(qualified)
        qualified_tokens += token_count if qualified else 0
        request_metrics.append({
            "request_id": request_id, "output_tokens": token_count, "finished": finished,
            "ttft_s": ttft, "completion_flow_s": flow, "max_observed_generation_gap_s": max_gap,
            "arrival_to_first_schedule_s": wait, "joint_slo_pass": qualified,
            "failed_slo_fields": failures, "equal_adjacent_timestamp_pairs": repeated,
        })

    durations, mixed_durations, schedule_times, decision_times = [], [], [], []
    mixed_steps, prefill_steps, decode_steps = [], [], []
    preemptions = total_prefill = total_decode = 0
    budget_violations = 0
    for index, step in enumerate(steps):
        prefill = step.get("prefill_tokens", 0)
        decode = step.get("decode_tokens", 0)
        total_prefill += prefill
        total_decode += decode
        for field, value in (("prefill_tokens", prefill), ("decode_tokens", decode)):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                errors.append("Invalid step %s at index %d: %s" % (field, index, value))
        budget = step.get("budget")
        if not finite(budget) or budget < 0 or prefill > budget:
            budget_violations += 1
            errors.append("Prefill budget violation at step %d: actual=%s, budget=%s" % (index, prefill, budget))
        start, end = step.get("start_s"), step.get("end_s")
        duration = end - start if finite(start) and finite(end) and end >= start else None
        if duration is None:
            errors.append("Invalid step interval at index %d" % index)
        for field in ("start_s", "end_s"):
            timestamp = step.get(field)
            if not finite(timestamp) or timestamp < 0 or timestamp > elapsed:
                errors.append("Step %d timestamp outside [0, elapsed_s]: %s=%s" % (index, field, timestamp))
        durations.append(duration)
        if prefill > 0:
            prefill_steps.append(step)
        if decode > 0:
            decode_steps.append(step)
        if prefill > 0 and decode > 0:
            mixed_steps.append(step)
            mixed_durations.append(duration)
        schedule_times.append(step.get("schedule_s"))
        decision_times.append(step.get("decision_us"))
        preempted = step.get("preempted", [])
        preemptions += len(preempted)
        for item in preempted:
            preempted_ids.add(str(item.get("request_id")) if isinstance(item, dict) else str(item))
        entries = step.get("requests", [])
        for entry in entries:
            entry_id = str(entry["request_id"])
            actual_prefill_by_request[entry_id] += entry.get("prefill_tokens", 0)
            actual_decode_by_request[entry_id] += entry.get("decode_tokens", 0)
            if entry_id not in seen_ids:
                errors.append("Unknown scheduled request_id at step %d: %s" % (index, entry["request_id"]))
            for field in ("prefill_tokens", "decode_tokens"):
                value = entry.get(field, 0)
                if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                    errors.append("Invalid per-request %s for %s at step %d: %s" % (field, entry_id, index, value))
        for field in ("prefill_tokens", "decode_tokens"):
            if sum(entry.get(field, 0) for entry in entries) != step.get(field, 0):
                errors.append("Per-request %s sum mismatch at step %d" % (field, index))

    request_work_mismatches = 0
    for metrics in request_metrics:
        request_id = metrics["request_id"]
        expected_prefill, expected_decode = expected_work[request_id]
        actual_prefill = actual_prefill_by_request[request_id]
        actual_decode = actual_decode_by_request[request_id]
        balanced = actual_prefill == expected_prefill and actual_decode == expected_decode
        metrics.update({
            "actual_prefill_tokens": actual_prefill, "expected_prefill_tokens": expected_prefill,
            "actual_decode_tokens": actual_decode, "expected_decode_tokens": expected_decode,
            "work_conserved": balanced,
        })
        if not balanced:
            request_work_mismatches += 1
            errors.append("Request work mismatch for %s: prefill actual/expected=%s/%s, decode actual/expected=%s/%s" % (request_id, actual_prefill, expected_prefill, actual_decode, expected_decode))
    expected_total_decode = output_tokens - len(requests)
    total_work_conserved = total_prefill == prompt_tokens and total_decode == expected_total_decode
    if not total_work_conserved:
        errors.append("Total step work mismatch: prefill actual/expected=%s/%s, decode actual/expected=%s/%s" % (total_prefill, prompt_tokens, total_decode, expected_total_decode))

    expected_count = protocol.get("request_count")
    if expected_count is not None and expected_count != len(requests):
        errors.append("Request count %d differs from protocol.request_count %s" % (len(requests), expected_count))
    if not requests:
        errors.append("No requests recorded")
    if not mixed_steps:
        warnings.append("No step contains both prefill and decode work")
    if repeated_pairs:
        warnings.append("Adjacent output tokens share timestamps; gaps are emission observations, not true token ITL")
    if mismatch_count:
        warnings.append("Timestamp-count mismatches fail the joint SLO to avoid optimistic gap estimates")
    all_finished = bool(requests) and finished_count == len(requests)
    ttft_stats, flow_stats, gap_stats = describe(ttfts), describe(flows), describe(max_gaps)
    makespan = max(completion_times) - min(arrival_times) if all_finished and len(completion_times) == len(requests) and len(arrival_times) == len(requests) else None
    policy = raw.get("policy", raw_path.parent.name)
    row = {
        "run": relative_path,
        "is_warmup": is_warmup_path(raw_path),
        "policy": policy if isinstance(policy, str) else json.dumps(policy, sort_keys=True),
        "elapsed_s": elapsed,
        "request_count": len(requests), "finished_count": finished_count,
        "unfinished_count": len(requests) - finished_count,
        "all_finished": all_finished, "valid": not errors,
        "prompt_tokens": prompt_tokens, "output_tokens": output_tokens,
        "output_tokens_per_s": output_tokens / elapsed,
        "prompt_plus_output_tokens_per_s": (prompt_tokens + output_tokens) / elapsed,
        "ttft_mean_s": ttft_stats["mean"], "ttft_p95_s": ttft_stats["p95"], "ttft_max_s": ttft_stats["max"],
        "completion_flow_mean_s": flow_stats["mean"], "completion_flow_p95_s": flow_stats["p95"],
        "completion_flow_max_s": flow_stats["max"], "completion_makespan_s": makespan,
        "request_max_gap_mean_s": gap_stats["mean"], "request_max_gap_p95_s": gap_stats["p95"],
        "request_max_gap_max_s": gap_stats["max"],
        "joint_slo_qualified_requests": qualified_count,
        "joint_slo_request_fraction": qualified_count / len(requests) if requests else 0.0,
        "joint_slo_requests_per_s": qualified_count / elapsed,
        "joint_slo_qualified_output_tokens": qualified_tokens,
        "joint_slo_output_tokens_per_s": qualified_tokens / elapsed,
        "slo_ttft_s": slo["ttft_s"], "slo_gap_s": slo["gap_s"], "slo_completion_s": slo["completion_s"],
        "arrival_to_first_schedule_mean_s": describe(waits)["mean"],
        "arrival_to_first_schedule_p95_s": describe(waits)["p95"],
        "steps": len(steps), "mixed_steps": len(mixed_steps),
        "prefill_steps": len(prefill_steps), "decode_steps": len(decode_steps),
        "mixed_step_fraction": len(mixed_steps) / len(steps) if steps else 0.0,
        "actual_prefill_tokens": total_prefill, "actual_decode_tokens": total_decode,
        "expected_actual_prefill_tokens": prompt_tokens,
        "expected_actual_decode_tokens": expected_total_decode,
        "total_work_conserved": total_work_conserved,
        "request_work_mismatches": request_work_mismatches,
        "fixed_output_length_mismatches": output_length_mismatches,
        "prefill_budget_violations": budget_violations,
        "mean_actual_prefill_tokens_all_steps": avg([s.get("prefill_tokens", 0) for s in steps]),
        "mean_actual_prefill_tokens_prefill_steps": avg([s["prefill_tokens"] for s in prefill_steps]),
        "mean_actual_prefill_tokens_mixed_steps": avg([s["prefill_tokens"] for s in mixed_steps]),
        "mean_prefill_cap_mixed_steps": avg([s.get("budget") for s in mixed_steps]),
        "mean_prefill_cap_prefill_steps": avg([s.get("budget") for s in prefill_steps]),
        "mean_decode_count_before_mixed_steps": avg([s.get("decode_count_before") for s in mixed_steps]),
        "mean_decode_context_sum_mixed_steps": avg([s.get("decode_context_sum") for s in mixed_steps]),
        "max_resident_requests": max((s["running_count"] for s in steps if finite(s.get("running_count"))), default=None),
        "max_waiting_requests": max((s["waiting_count"] for s in steps if finite(s.get("waiting_count"))), default=None),
        "peak_kv_fraction": max((s["kv_used_blocks"] / s["kv_total_blocks"] for s in steps if finite(s.get("kv_used_blocks")) and finite(s.get("kv_total_blocks")) and s["kv_total_blocks"] > 0), default=None),
        "max_decode_count_before": max((s["decode_count_before"] for s in steps if finite(s.get("decode_count_before"))), default=None),
        "max_decode_context_sum": max((s["decode_context_sum"] for s in steps if finite(s.get("decode_context_sum"))), default=None),
        "step_duration_mean_s": describe(durations)["mean"],
        "step_duration_p95_s": describe(durations)["p95"],
        "mixed_step_duration_mean_s": describe(mixed_durations)["mean"],
        "mixed_step_duration_p95_s": describe(mixed_durations)["p95"],
        "schedule_mean_s": describe(schedule_times)["mean"],
        "decision_mean_us": describe(decision_times)["mean"],
        "decision_p95_us": describe(decision_times)["p95"],
        "preemption_events": preemptions, "unique_preempted_requests": len(preempted_ids),
        "adjacent_timestamp_pairs": adjacent_pairs, "equal_adjacent_timestamp_pairs": repeated_pairs,
        "equal_adjacent_timestamp_fraction": repeated_pairs / adjacent_pairs if adjacent_pairs else 0.0,
        "requests_with_equal_adjacent_timestamps": repeated_requests,
        "token_timestamp_count_mismatches": mismatch_count,
        "budget_distribution_all_steps": distribution([s.get("budget") for s in steps]),
        "budget_distribution_prefill_steps": distribution([s.get("budget") for s in prefill_steps]),
        "budget_distribution_mixed_steps": distribution([s.get("budget") for s in mixed_steps]),
    }
    return {
        "summary": row, "raw_path": str(raw_path), "protocol_path": str(protocol_path),
        "slo": slo, "validation_errors": errors, "warnings": warnings,
        "expected_request_count": expected_count,
        "request_count_validation_scope": "protocol.request_count" if expected_count is not None else "all requests present in raw.json; omitted arrivals cannot be detected",
        "request_statistics": {
            "ttft_s": ttft_stats, "completion_flow_s": flow_stats,
            "maximum_observed_generation_gap_s": gap_stats,
            "arrival_to_first_schedule_s": describe(waits),
            "arrival_to_add_s": describe(admission_waits), "add_to_first_schedule_s": describe(scheduler_waits),
        },
        "failed_by_slo_field": dict(failed_by_slo), "requests": request_metrics,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--protocol", type=Path, help="Use this protocol.json for every run")
    parser.add_argument("--output-dir", type=Path, help="Defaults to run_directory (or the raw file's parent)")
    parser.add_argument("--strict", action="store_true", help="Write reports, then exit 2 if any validation fails")
    parser.add_argument("--exclude-warmup", action="store_true", help="Exclude raw.json files with a parent path segment starting with warm (case insensitive)")
    args = parser.parse_args()
    root = args.run_directory.resolve()
    paths = [root] if root.is_file() else sorted(root.rglob("raw.json"))
    excluded_warmups = [path for path in paths if args.exclude_warmup and is_warmup_path(path)]
    paths = [path for path in paths if path not in excluded_warmups]
    if not paths:
        parser.error("No raw.json files remain under %s after the requested filters" % root)
    base = root.parent if root.is_file() else root
    output_dir = args.output_dir.resolve() if args.output_dir else base
    override = args.protocol.resolve() if args.protocol else None
    reports = []
    for raw_path in paths:
        protocol_path = find_protocol(raw_path, override)
        reports.append(summarize(raw_path, protocol_path, str(raw_path.parent.relative_to(base))))
    reference_slo = reports[0]["slo"]
    if any(report["slo"] != reference_slo for report in reports):
        parser.error("Discovered runs use different SLOs; pass one --protocol for a valid comparison")
    output = {
        "schema_version": 1, "slo": reference_slo,
        "definitions": {
            "throughput_denominator": "raw.elapsed_s, including arrival gaps, waiting, and final drain",
            "joint_slo": "finished AND TTFT <= ttft_s AND maximum observed generation gap <= gap_s AND completion-arrival <= completion_s; all recorded requests are in the denominator",
            "qualified_tokens": "All output tokens of requests meeting every joint SLO condition",
            "generation_gap": "Differences between adjacent observed token_times_s within a request; excludes TTFT; a single observed output token has gap zero; NOT a claim of true token ITL",
            "percentiles": "Linear interpolation at (n-1)*p; headline mean/p95/max are null if any request is unobserved; observed-only diagnostics remain in JSON",
            "makespan": "Latest completion minus earliest arrival; null unless all requests completed",
            "workload_completeness": "All raw.json requests must finish; omitted arrivals are detectable only when protocol.request_count is supplied",
            "prefill_cap": "Recorded budget field; averages distinguish mixed, prefill, and all steps",
            "fixed_lengths": "Each request must emit max_tokens outputs; cumulative per-request prefill must equal prompt_tokens and decode must equal output_tokens-1; step totals must conserve these amounts",
            "warmup": "is_warmup is true when any raw.json parent path segment starts with warm, case insensitive; --exclude-warmup removes these runs before comparison",
        },
        "excluded_warmup_paths": [str(path) for path in excluded_warmups],
        "run_count": len(reports), "all_valid": all(report["summary"]["valid"] for report in reports),
        "runs": reports,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path, csv_path = output_dir / "summary.json", output_dir / "summary.csv"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(reports[0]["summary"]))
        writer.writeheader()
        for report in reports:
            writer.writerow({key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value for key, value in report["summary"].items()})
    print("Wrote %s and %s (%d runs; all_valid=%s)" % (json_path, csv_path, len(reports), output["all_valid"]))
    for report in reports:
        row = report["summary"]
        print("%s: output=%.3f token/s, joint=%.3f request/s, completed=%d/%d, mixed_steps=%d, equal_timestamp_pairs=%d" % (row["run"], row["output_tokens_per_s"], row["joint_slo_requests_per_s"], row["finished_count"], row["request_count"], row["mixed_steps"], row["equal_adjacent_timestamp_pairs"]))
        for error in report["validation_errors"]:
            print("  INVALID: " + error)
    return 2 if args.strict and not output["all_valid"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
