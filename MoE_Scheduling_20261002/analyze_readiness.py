"""Summarize readiness observations; never turn local windows into savings."""
import argparse
import json
import math
from collections import Counter
from pathlib import Path


def fraction(numerator, denominator):
    return numerator / denominator if denominator else None


def quantiles(values):
    values = sorted(v for v in values if isinstance(v, (int, float)) and math.isfinite(v))
    def percentile(p):
        if not values:
            return None
        index = (len(values) - 1) * p
        lo, hi = math.floor(index), math.ceil(index)
        return values[lo] + (values[hi] - values[lo]) * (index - lo)
    return dict(n=len(values), p50_ms=percentile(.5), p90_ms=percentile(.9),
                max_ms=max(values) if values else None)


def describe(calls):
    rows = [row for call in calls for row in call["decode_rows"]]
    early = [r for r in rows if r.get("early_ready_before_last_group") is True]
    by_group = {}
    for index in range(max((len(c["groups"]) for c in calls), default=0)):
        eligible = [r for c in calls if len(c["groups"]) > index for r in c["decode_rows"]]
        newly = sum(r.get("ready_group_index") == index for r in eligible)
        cumulative = sum(type(r.get("ready_group_index")) is int
                         and r["ready_group_index"] <= index for r in eligible)
        by_group[str(index)] = dict(eligible_rows=len(eligible), newly_ready_rows=newly,
            newly_ready_fraction=fraction(newly, len(eligible)),
            cumulative_ready_rows=cumulative,
            cumulative_ready_fraction=fraction(cumulative, len(eligible)),
            newly_ready_fraction_of_all_rows=fraction(newly, len(rows)))
    resident = sum(r.get("all_topk_entry_resident") is True for r in rows)
    resident_cold = sum(r.get("entry_resident_but_first_group_cold") is True for r in rows)
    early_calls = sum(any(r.get("early_ready_before_last_group") is True
                          for r in c["decode_rows"]) for c in calls)
    return dict(calls_with_valid_decode_rows=len(calls), valid_decode_row_observations=len(rows),
        ready_group_index_zero_based=by_group,
        early_ready_rows=len(early), early_ready_fraction=fraction(len(early), len(rows)),
        calls_with_at_least_one_early_row=early_calls,
        early_call_fraction=fraction(early_calls, len(calls)),
        never_ready_rows=sum(r.get("ready_group_index") is None for r in rows),
        local_window_all_rows=quantiles([r.get("local_ready_to_layer_end_ms") for r in rows]),
        local_window_early_rows=quantiles([r.get("local_ready_to_layer_end_ms") for r in early]),
        all_topk_entry_resident_rows=resident,
        all_topk_entry_resident_fraction=fraction(resident, len(rows)),
        entry_resident_rows_waiting_for_mixed_cold_first_group=resident_cold,
        mixed_cold_fraction_of_all_rows=fraction(resident_cold, len(rows)),
        mixed_cold_fraction_of_entry_resident_rows=fraction(resident_cold, resident),
        executed_group_count_distribution=dict(sorted(Counter(len(c["groups"]) for c in calls).items())))


def service_status(directory):
    path = directory / "raw.json"
    if not path.exists():
        return dict(path=str(path), status="MISSING", complete=False,
                    bounded_capture_complete=False, natural_completion_or_quality_established=False)
    raw = json.loads(path.read_text())
    requests = raw.get("requests", [])
    complete = (raw.get("status") == "COMPLETE" and raw.get("error") is None
                and bool(requests) and all(r.get("status") == "completed" for r in requests))
    return dict(path=str(path), status=raw.get("status"), error=raw.get("error"),
        complete=complete, bounded_capture_complete=complete,
        natural_completion_or_quality_established=False,
        request_count=len(requests), completed_requests=sum(r.get("status") == "completed" for r in requests),
        output_tokens=sum(len(r.get("output_token_ids", [])) for r in requests),
        capture_wall_s=raw.get("observation_end_s"),
        note="本探针强制生成16个token、按length停止；完成仅指有界capture结束，不是自然完成或质量证据。cap24/cap64显存资源不同，不作同预算策略收益对比。"
             if complete else "原始运行不完整或缺失，不可用于完整服务对比；本强制16token探针不建立自然完成或质量结论。")


def analyze_arm(directory):
    path = directory / "request_readiness.json"
    result = dict(service=service_status(directory), readiness_path=str(path))
    if not path.exists():
        result["readiness_status"] = "MISSING"
        return result
    data = json.loads(path.read_text())
    all_calls = data.get("calls", [])
    calls = [c for c in all_calls if c.get("completed") is True
             and c.get("mapping_verified") is True and c.get("decode_rows")]
    result.update(readiness_status="AVAILABLE", recorded_calls=len(all_calls),
        incomplete_calls=sum(c.get("completed") is not True for c in all_calls),
        mapping_excluded_calls=sum(c.get("mapping_verified") is not True for c in all_calls),
        invalid_rows_excluded=sum(c.get("excluded_rows", 0) for c in all_calls),
        overall=describe(calls))
    result["by_exact_decode_rows_per_call"] = {
        str(width): describe([c for c in calls if len(c["decode_rows"]) == width])
        for width in sorted({len(c["decode_rows"]) for c in calls})}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path(__file__).parent / "results")
    parser.add_argument("--output", type=Path, help="New output path; default RESULTS/metrics.json")
    args = parser.parse_args()
    arms = {name: analyze_arm(args.results / name) for name in ("cap24", "cap64")}
    control = arms["cap64"].get("overall")
    negative = dict(status="UNAVAILABLE")
    if control and control["valid_decode_row_observations"]:
        max_groups = max(map(int, control["executed_group_count_distribution"]), default=0)
        passed = control["early_ready_rows"] == 0 and max_groups == 1
        negative = dict(status="PASS_NO_INTERGROUP_EARLY_READY" if passed else "UNEXPECTED_REQUIRES_INSPECTION",
            early_ready_rows=control["early_ready_rows"], maximum_groups_per_call=max_groups,
            residual_local_window_all_rows=control["local_window_all_rows"],
            note="Single-group return-tail intervals may be nonzero; do not subtract them as a noise correction.")
    result = dict(schema_version=1, evidence_type="LOCAL_LAYER_OPPORTUNITY_ONLY",
        units="Rows are request-token-layer observations, not independent requests or repetitions.",
        quantile_method="linear interpolation on sorted observations",
        group_fraction_denominator="Rows in valid decode calls containing that group index; also show all-row share.",
        limitations=["No summation of overlapping local windows into savings or speedup.",
            "Early ready excludes rows first completed in the final group.",
            "All-row windows include the last-group-to-return bookkeeping tail.",
            "CUDA event intervals include stream idle time and observer overhead.",
            "Entry-resident/mixed-cold co-membership does not measure avoidable first-group delay.",
            "cap64 has a different expert scratch budget; it is a structural control, not an equal-budget speed baseline."],
        both_raw_runs_complete=all(a["service"]["complete"] for a in arms.values()),
        arms=arms, cap64_negative_control=negative)
    destination = args.output or args.results / "metrics.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")
    print(destination)


if __name__ == "__main__":
    main()
