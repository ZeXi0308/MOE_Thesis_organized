"""Partition RR/static measured call cost, held requests and full request timing."""
import argparse
from collections import Counter
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def phase_of(index, action_index, old_decode, new_prefill, new_decode, rr_applied):
    if index < action_index:
        return "prefix"
    if rr_applied:
        return "three_running_schedule2"
    if new_prefill:
        return "new_prefill_with_old" if old_decode else "new_prefill_after_old"
    if new_decode and old_decode:
        return "three_request_decode" if old_decode == 2 else "two_request_decode"
    if old_decode:
        return "old_only_decode"
    if new_decode:
        return "new_only_decode"
    return "empty"


def summarize(calls):
    fields = ("engine_wall_s", "weight_copy_bytes", "groups", "host_apply_ms", "load_cuda_span_ms",
              "route_to_host_ms", "old_decode_tokens", "new_prefill_tokens", "new_decode_tokens", "held_new_steps", "held_old_request_steps")
    return dict(engine_calls=len(calls), call_indices=[c["index"] for c in calls],
        **{field: sum(c[field] for c in calls) for field in fields},
        old_critical_path_engine_wall_s=sum(c["engine_wall_s"] for c in calls if c["before_old_completion"]),
        old_decode_request_count_histogram=dict(Counter(c["old_decode_requests"] for c in calls)),
        gc_callback_overlap_ms=sum(e["overlap_ms"] for c in calls for e in c["gc_overlaps"]),
        gc_generation_counts=dict(Counter(e["generation"] for c in calls for e in c["gc_overlaps"])))


def analyze(root):
    entries = read(root / "episodes.json")
    trace = [json.loads(line) for line in (root / "pager/calls.jsonl").open() if line.strip()]
    events = read(root / "runtime_events.json")["events"]
    cells = {}
    for entry in entries:
        phase = entry["phase"]; name = phase.split("/")[0]; raw = read(root / name / "raw.json")
        action = raw["event_actions"][0]; old = set(action["before"]["old_output_tokens"]); new = action["request_id"]
        rows = {r["request_id"]: r for r in raw["requests"]}; old_done = max(rows[r]["completion_s"] for r in old)
        records = [r for r in trace if r["context"]["phase"] == phase]; calls = []; assigned = []
        for call in raw["engine_calls"]:
            steps = raw["scheduler_steps"][call["scheduler_step_start"]:call["scheduler_step_stop"]]
            scheduled = [r for step in steps for r in step["scheduled"]]
            old_decode = {r["request_id"] for r in scheduled if r["request_id"] in old and r["decode_tokens"]}
            new_prefill = sum(r["prefill_tokens"] for r in scheduled if r["request_id"] == new)
            new_decode = sum(r["decode_tokens"] for r in scheduled if r["request_id"] == new)
            width = call["decode_width_rr"]
            held = [raw["internal_to_source"][rid] for rid in width["held_request_ids"]]
            label = phase_of(call["index"], action["engine_call"], len(old_decode), new_prefill, new_decode, width["applied"])
            rr = [r for r in records if call["scheduler_step_start"] <= r["context"]["step_id"] < call["scheduler_step_stop"]]
            assigned.extend(r["call_id"] for r in rr)
            start_ns = round((raw["measurement_origin_perf_counter_s"] + call["start_s"]) * 1e9)
            stop_ns = round((raw["measurement_origin_perf_counter_s"] + call["return_s"]) * 1e9)
            overlaps = [dict(generation=e["generation"], start_perf_ns=e["start_perf_ns"], stop_perf_ns=e["stop_perf_ns"],
                callback_duration_ms=e["duration_ns"] / 1e6,
                overlap_ms=(min(stop_ns, e["stop_perf_ns"]) - max(start_ns, e["start_perf_ns"])) / 1e6)
                for e in events if e.get("stop_perf_ns") is not None and e["start_perf_ns"] < stop_ns and e["stop_perf_ns"] > start_ns]
            calls.append(dict(index=call["index"], phase=label, start_s=call["start_s"], return_s=call["return_s"],
                engine_wall_s=call["return_s"] - call["start_s"], before_old_completion=call["return_s"] <= old_done,
                old_decode_requests=len(old_decode), old_decode_tokens=sum(r["decode_tokens"] for r in scheduled if r["request_id"] in old),
                new_prefill_tokens=new_prefill, new_decode_tokens=new_decode,
                rr_applied=width["applied"], held_request_ids=held, held_new_steps=int(new in held),
                held_old_request_steps=sum(rid in old for rid in held),
                running_request_ids_before=[raw["internal_to_source"][rid] for rid in width["running_order_before"]],
                scheduled_request_ids=[raw["internal_to_source"][rid] for rid in width["selected_request_ids"]],
                weight_copy_bytes=sum(r["weight_copy_bytes"] for r in rr), groups=sum(len(r["groups"]) for r in rr),
                host_apply_ms=sum(r["host_apply_ms"] for r in rr), route_to_host_ms=sum(r["route_to_host_ms"] for r in rr),
                load_cuda_span_ms=sum(g["load_cuda_span_ms"] for r in rr for g in r["groups"]), gc_overlaps=overlaps))
        if Counter(assigned) != Counter(r["call_id"] for r in records):
            raise ValueError("Engine-call phase partition does not cover measured records exactly")
        stages = {label: summarize([c for c in calls if c["phase"] == label]) for label in sorted({c["phase"] for c in calls})}
        total = summarize(calls)
        request_timing = {}
        for rid, request in rows.items():
            times = request["token_times_s"]; gaps = [b - a for a, b in zip(times, times[1:])]
            request_timing[rid] = dict(role="new" if rid == new else "old", output_token_ids=request["output_token_ids"],
                ttft_s=times[0] - request["arrival_s"], max_itl_s=max(gaps), tpot_s=sum(gaps) / len(gaps),
                completion_latency_s=request["completion_s"] - request["arrival_s"], completion_s=request["completion_s"],
                itls_s=gaps)
        cells[name] = dict(arm=entry["arm"], chunk=entry["chunk"], block=entry["block"], whole_wall_s=raw["observation_end_s"],
            request_timing=request_timing, all_request_max_itl_s=max(r["max_itl_s"] for r in request_timing.values()),
            old_completion_s=old_done, new_first_token_s=rows[new]["token_times_s"][0],
            new_completion_s=rows[new]["completion_s"], total=total, stages=stages, calls=calls,
            online_gap_s=raw["observation_end_s"] - total["engine_wall_s"],
            old_critical_path_online_gap_s=old_done - total["old_critical_path_engine_wall_s"])
    lookup = {(v["block"], v["arm"]): name for name, v in cells.items()}
    pairs = [(lookup[b, "static32"], lookup[b, "rr32"], f"block{b}_rr32_minus_static32") for b in range(2)]
    pairs.extend((lookup[0, arm], lookup[1, arm], f"{arm}_repeat") for arm in ("static32", "rr32"))
    comparisons = []
    for a, b, kind in pairs:
        left, right = cells[a], cells[b]; zero = summarize([])
        deltas = {label: {k: right["stages"].get(label, zero)[k] - left["stages"].get(label, zero)[k]
            for k in ("engine_calls", "engine_wall_s", "weight_copy_bytes", "groups", "host_apply_ms", "load_cuda_span_ms", "old_critical_path_engine_wall_s", "gc_callback_overlap_ms")}
            for label in sorted(set(left["stages"]) | set(right["stages"]))}
        wall_delta = right["whole_wall_s"] - left["whole_wall_s"]
        gap_delta = right["online_gap_s"] - left["online_gap_s"]
        request_deltas = {rid: {k: right["request_timing"][rid][k] - row[k]
            for k in ("ttft_s", "max_itl_s", "tpot_s", "completion_latency_s", "completion_s")}
            for rid, row in left["request_timing"].items()}
        comparisons.append(dict(a=a, b=b, kind=kind, stages_delta_b_minus_a=deltas,
            request_timing_delta_b_minus_a=request_deltas,
            output_equal_by_request={rid: row["output_token_ids"] == right["request_timing"][rid]["output_token_ids"] for rid, row in left["request_timing"].items()},
            whole_wall_delta_s=wall_delta, online_gap_delta_s=gap_delta,
            reconstruction_residual_s=wall_delta - gap_delta - sum(d["engine_wall_s"] for d in deltas.values()),
            old_completion_delta_s=right["old_completion_s"] - left["old_completion_s"],
            old_critical_path_online_gap_delta_s=right["old_critical_path_online_gap_s"] - left["old_critical_path_online_gap_s"]))
    return dict(status="OBSERVED_PHASE_COST_DIAGNOSTIC", cells=cells, comparisons=comparisons,
        timing_scope="Phases partition actual engine-call wall exactly; online_gap is whole capture minus the call-wall sum. Host apply contains route-to-host and overlaps CUDA load spans; these columns are not additive. GC callback time overlap is descriptive and is not subtracted or asserted causal.",
        interpretation_scope="Call composition, actual held requests and critical-path membership are observed. The RR phase requires three running requests with two scheduled rows; request latency includes held intervals. Cross-arm phase subtraction is a decomposition of measured totals, not fixed-route or fixed-composition counterfactual cost. Each policy evolves its own routes/cache/request state.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", type=Path, required=True); p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(); result = analyze(args.input_dir)
    with args.out.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False); stream.write("\n")
