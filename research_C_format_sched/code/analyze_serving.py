#!/usr/bin/env python3
"""Summarize fixed-trace serving runs; retain failed runs and raw repetitions.

  python code/analyze_serving.py runs/serve_*/serving --output runs/serving_summary.json
  python code/analyze_serving.py --self-test

Each configuration/rate needs exactly two distinct repeat IDs for the main
50/100-ms comparison. Both runs must complete every planned request, and both
request-level TPOT P95 values must meet the limit. Select maximum mean observed
successful output-token throughput among eligible measured points; no fitting.
One-token completions have no TPOT and are counted explicitly, never set to zero.
No confidence interval or quality claim is inferred from two repetitions.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import statistics


def percentile(values, q):
    """Linear quantile for finite values; any upper interpolation of inf is inf."""
    if not values:
        return None
    values = sorted(values)
    pos = (len(values) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return values[lo]
    if math.isinf(values[hi]):
        return math.inf
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def quantiles(values):
    result = {"n": len(values)}
    for name, q in (("p50", .5), ("p95", .95)):
        value = percentile(values, q)
        result[name + "_ms"] = value if value is None or math.isfinite(value) else None
        result[name + "_is_infinite"] = value is not None and math.isinf(value)
    return result


def summarize_run(rows, metadata, power):
    complete = [r for r in rows if r["status"] == "completed"]
    failures = [r for r in rows if r["status"] != "completed"]
    ttft, tpot = [], []
    single_token = zero_token = 0
    for r in complete:
        if r.get("ttft_ms") is not None:
            ttft.append(r["ttft_ms"])
        if r.get("tpot_ms") is not None:
            tpot.append(r["tpot_ms"])
        elif r.get("output_tokens") == 1:
            single_token += 1
        elif r.get("output_tokens") == 0:
            zero_token += 1
    output_tokens = sum(r.get("output_tokens") or 0 for r in complete)
    unknown_tokens = sum(r.get("output_tokens") is None for r in rows)
    duration = metadata.get("measurement_elapsed_s", 0)
    all_complete = bool(rows) and not failures and not metadata.get("aborted")
    power = power or {}
    idle_j = power.get("idle_energy_j")
    energy = {}
    for name, key in (("primary", "energy_j"), ("counter", "counter_energy_j"),
                      ("sampled", "sampled_energy_j")):
        gross = power.get(key)
        net = gross - idle_j if gross is not None and idle_j is not None else None
        energy[name] = {"gross_j": gross, "net_j": net}
        valid = name != "counter" or power.get("counter_valid", False)
        for metric, value in (("gross", gross), ("net", net)):
            energy[name][metric + "_j_per_generated_token"] = (
                value / output_tokens if all_complete and not unknown_tokens
                and output_tokens > 0 and value is not None and valid else None)
    c, s = power.get("counter_energy_j"), power.get("sampled_energy_j")
    failure_penalty = [math.inf] * len(failures)
    return {
        "schema_version": 1, "metadata": metadata,
        "population": {"planned": len(rows), "arrived": sum(bool(r.get("arrived")) for r in rows),
            "sent": sum(r.get("sent_ns") is not None for r in rows),
            "completed": len(complete), "noncompleted": len(failures),
            "status_counts": dict(Counter(r["status"] for r in rows)),
            "finish_reason_counts": dict(Counter(str(r.get("finish_reason")) for r in complete)),
            "single_token_tpot_not_applicable": single_token,
            "zero_token_tpot_not_applicable": zero_token,
            "unknown_output_token_count": unknown_tokens,
            "completion_fraction": len(complete) / len(rows) if rows else None},
        "latency": {"ttft_completed": quantiles(ttft), "tpot_completed_multitoken": quantiles(tpot),
            "client_send_delay": quantiles([r["client_send_delay_ms"] for r in rows if r.get("client_send_delay_ms") is not None]),
            "client_send_delay_max_ms": max((r["client_send_delay_ms"] for r in rows if r.get("client_send_delay_ms") is not None), default=None),
            "ttft_population_failure_as_infinite": quantiles(ttft + failure_penalty),
            "tpot_multitoken_population_failure_as_infinite": quantiles(tpot + failure_penalty),
            "definition": "TTFT: send to first streamed choice; TPOT: first to last choice / (usage tokens - 1). Single/zero-token successes are N/A; failures are retained and penalized."},
        "throughput": {"measurement_elapsed_s": duration,
            "scheduled_arrivals_per_s": len(rows) / metadata["arrival_window_s"] if metadata.get("arrival_window_s", 0) > 0 else None,
            "successful_requests_per_s": len(complete) / duration if duration > 0 else None,
            "successful_output_tokens_per_s": output_tokens / duration if duration > 0 else None,
            "successful_output_tokens": output_tokens,
            "observed_output_tokens_including_partial_lower_bound": sum(r.get("output_tokens") or 0 for r in rows)},
        "energy": {"method": power.get("method", "unavailable"), "elapsed_s": power.get("elapsed_s"),
            "counter_valid": power.get("counter_valid", False), "idle_power_w": power.get("idle_power_w"),
            "sample_vs_counter_pct": (s / c - 1) * 100 if c and s is not None else None,
            "estimates": energy, "complete_request_energy_boundary": all_complete,
            "scope": "whole device, arrival plus drain client window; cancellation does not prove GPU quiescence; per-token energy withheld for incomplete population"},
        "eligible_population": all_complete and unknown_tokens == 0 and bool(tpot) and duration > 0
            and len(ttft) == len(complete) and len(tpot) + single_token + zero_token == len(complete),
    }


def aggregate(summaries):
    groups = defaultdict(list)
    for summary in summaries:
        m = summary["metadata"]
        key = json.dumps({k: m[k] for k in ("config_id", "arm", "model", "server_metadata_sha256",
              "trace_content_sha256", "schedule_sha256", "request_rate", "request_timeout_s", "drain_s",
              "arrival_window_s", "warmup_trace_sha256", "idle_seconds")}, sort_keys=True)
        groups[key].append(summary)
    points = []
    for key, runs in groups.items():
        conf = json.loads(key)
        repeats = [r["metadata"]["repeat_id"] for r in runs]
        two_runs = len(runs) == 2 and len(set(repeats)) == 2
        def values(getter):
            raw = [getter(r) for r in runs]
            valid = [v for v in raw if isinstance(v, (int, float)) and math.isfinite(v)]
            return {"raw": raw, "mean": statistics.mean(valid) if len(valid) == len(raw) and raw else None,
                    "min": min(valid) if valid else None, "max": max(valid) if valid else None,
                    "range": max(valid) - min(valid) if valid else None}
        point = {"configuration": conf, "repeat_ids": repeats, "exactly_two_distinct_repeats": two_runs,
            "runs": runs, "throughput_output_tokens_per_s": values(lambda r: r["throughput"]["successful_output_tokens_per_s"]),
            "throughput_requests_per_s": values(lambda r: r["throughput"]["successful_requests_per_s"]),
            "ttft_p50_ms": values(lambda r: r["latency"]["ttft_completed"]["p50_ms"]),
            "ttft_p95_ms": values(lambda r: r["latency"]["ttft_completed"]["p95_ms"]),
            "tpot_p50_ms": values(lambda r: r["latency"]["tpot_completed_multitoken"]["p50_ms"]),
            "tpot_p95_ms": values(lambda r: r["latency"]["tpot_completed_multitoken"]["p95_ms"]),
            "noncompleted": values(lambda r: r["population"]["noncompleted"]), "energy": {}}
        for method in ("primary", "counter", "sampled"):
            point["energy"][method] = {metric: values(lambda r, method=method, metric=metric:
                r["energy"]["estimates"][method][metric]) for metric in (
                    "gross_j", "net_j", "gross_j_per_generated_token", "net_j_per_generated_token")}
        point["slo_feasible"] = {str(limit): bool(two_runs and all(r["eligible_population"] and
            r["latency"]["tpot_completed_multitoken"]["p95_ms"] <= limit for r in runs)) for limit in (50, 100)}
        points.append(point)
    frontiers = defaultdict(list)
    for i, point in enumerate(points):
        conf = point["configuration"]
        # Rates/schedules may vary; workload content and all other conditions must match.
        family = {k: v for k, v in conf.items() if k not in ("request_rate", "schedule_sha256", "arrival_window_s")}
        frontiers[json.dumps(family, sort_keys=True)].append(i)
    selections = []
    for family, indices in frontiers.items():
        for limit in (50, 100):
            eligible = [i for i in indices if points[i]["slo_feasible"][str(limit)]]
            best = max(eligible, key=lambda i: points[i]["throughput_output_tokens_per_s"]["mean"]) if eligible else None
            selections.append({"family": json.loads(family), "tpot_p95_limit_ms": limit,
                "selected_point_index": best, "eligible_point_indices": eligible,
                "criterion": "maximum mean measured successful output tokens/s; both runs complete all requests and meet TPOT P95"})
    return {"schema_version": 1, "points": points, "matched_slo": selections,
            "scope": "observed points and two-run raw/range only; failed populations retained; no interpolation or quality qualification"}


def self_test():
    meta = {"measurement_elapsed_s": 2, "aborted": False}
    rows = [{"status": "completed", "arrived": True, "sent_ns": 1, "output_tokens": 3,
             "ttft_ms": 10, "tpot_ms": 20, "finish_reason": "stop"},
            {"status": "timeout", "arrived": True, "sent_ns": 2, "output_tokens": 1}]
    power = {"energy_j": 10, "counter_energy_j": 10, "sampled_energy_j": 9,
             "counter_valid": True, "idle_energy_j": 2, "elapsed_s": 2}
    s = summarize_run(rows, meta, power)
    assert s["population"]["noncompleted"] == 1 and not s["eligible_population"]
    assert s["latency"]["tpot_multitoken_population_failure_as_infinite"]["p95_is_infinite"]
    assert s["throughput"]["observed_output_tokens_including_partial_lower_bound"] == 4
    assert s["energy"]["estimates"]["counter"]["net_j"] == 8
    assert s["energy"]["estimates"]["primary"]["gross_j_per_generated_token"] is None
    rows[1].update(status="completed", ttft_ms=12, finish_reason="stop")
    s = summarize_run(rows, meta, power)
    assert s["population"]["single_token_tpot_not_applicable"] == 1
    assert s["energy"]["estimates"]["primary"]["gross_j_per_generated_token"] == 2.5
    assert percentile([10, 20, 30, 40, 50], .95) == 48
    # Two runs must BOTH pass; one fast successful run cannot hide failed population.
    base = dict(config_id="a", arm="fp8", model="m", server_metadata_sha256="x",
        trace_content_sha256="t", schedule_sha256="s", request_rate=2, request_timeout_s=3,
        drain_s=5, arrival_window_s=1, warmup_trace_sha256=None, idle_seconds=3)
    good = summarize_run(rows, dict(meta, **base, repeat_id="1"), power)
    bad = summarize_run([dict(rows[0], status="error")], dict(meta, **base, repeat_id="2"), power)
    assert not aggregate([good, bad])["points"][0]["slo_feasible"]["100"]
    second = summarize_run(rows, dict(meta, **base, repeat_id="2"), power)
    a = aggregate([good, second])
    assert a["points"][0]["slo_feasible"]["50"] and a["matched_slo"][0]["selected_point_index"] == 0
    print("CPU serving analysis checks passed")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="*")
    p.add_argument("--output")
    p.add_argument("--self-test", action="store_true")
    a = p.parse_args()
    if a.self_test:
        self_test()
        return
    if not a.runs or not a.output:
        p.error("provide run directories and --output")
    summaries = []
    seen = set()
    for name in a.runs:
        root = Path(name).resolve()
        if root in seen:
            raise ValueError("duplicate run path")
        seen.add(root)
        meta = json.loads((root / "metadata.json").read_text())
        rows = [json.loads(x) for x in (root / "requests.jsonl").read_text().splitlines() if x.strip()]
        if meta.get("state") in ("PREPARING", "RUNNING"):
            # Abrupt termination may leave only the initial all-request table.
            # Recover completed events, but never invent a measurement end time.
            by_id = {r["request_id"]: r for r in rows}
            events = root / "events.jsonl"
            if events.exists():
                for line in events.read_text().splitlines():
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue  # Retain a partially written tail as crash evidence.
                    if event["event"] == "terminal":
                        by_id[event["request"]["request_id"]].update(event["request"])
                    elif event["event"] == "arrival":
                        by_id[event["request_id"]].update(arrived=True,
                            scheduled_arrival_ns=event["scheduled_arrival_ns"])
            meta.update(aborted=True, state="INCOMPLETE_RECORD", recovery="replayed terminal/arrival events; physical end time unknown")
        power = json.loads((root / "power.json").read_text()) if (root / "power.json").exists() else {}
        summary = summarize_run(rows, meta, power)
        summary["source"] = str(root)
        summaries.append(summary)
    output = Path(a.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(aggregate(summaries), indent=2, allow_nan=False) + "\n")
    print(output)


if __name__ == "__main__":
    main()
