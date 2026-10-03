"""Read six independent-engine clock-arrival captures; no SLO or policy GO."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "admission_memory_observer_v026"))
from analyze_admission_memory_observer import account, allocator_trajectory, digest, gc_overlap, read, requests, require

SOURCES = list(range(12, 20))
PROMPTS = [64, 128, 32, 96, 128, 32, 64, 128]
OUTPUTS = [32, 16, 48, 24, 16, 64, 40, 24]
ARRIVALS = [0, 0, .4, .8, 2, 2.4, 4, 4.4]
ARMS = ["cap3_static32", "cap2_static32", "cap2_phase32"]
TORCH_MEMORY_SHA256 = "71a36cb635c0a076bf8cda0ae08c112d7b0e15d9d2fb21a0dee373e2967c4700"


def validate_schedule(raw, workload, cap, mode):
    rows = {r["request_id"]: r for r in raw["requests"]}; aliases = raw["internal_to_source"]
    progress = {rid: 0 for rid in rows}; first = {}; report = []; previous_threshold = 32
    require(raw["status"] == "COMPLETE" and not raw.get("error") and raw["event_arrival"] is None
        and not raw["event_actions"] and not raw["prefill_release_actions"], "not a complete clock-arrival capture")
    require(raw["target_cap"] == cap and raw["policy"] == "static" and not raw["allow_preemption"], "capture policy changed")
    calls, steps = raw["engine_calls"], raw["scheduler_steps"]
    require([i for c in calls for i in range(c["scheduler_step_start"], c["scheduler_step_stop"])] == list(range(len(steps))), "engine/step coverage")
    for i, call in enumerate(calls):
        require(call["index"] == i and call["returned"] and call["start_s"] <= call["return_s"], "engine call timing/status")
        require(call["scheduler_step_stop"] == call["scheduler_step_start"] + 1, "synchronous call did not contain exactly one scheduler step")
        step = steps[call["scheduler_step_start"]]; decision = step["phase_prefill"]
        require(step["step"] == call["scheduler_step_start"] and call["phase_prefill"] == decision, "phase decision/step alignment")
        ready = {aliases[rid] for rid in decision["ready_decode_ids"]}
        observable_ready = {rid for rid, row in rows.items() if row["token_times_s"][0] <= decision["decision_s"] < row["completion_s"]}
        require(ready == observable_ready, "ready decode differs from already-produced, unfinished requests")
        threshold = 32 if mode == "static32" or ready else 0
        require(decision["mode"] == mode and decision["status"] == "applied" and decision["effective_token_budget"] == 64
            and decision["chosen_threshold"] == threshold and decision["threshold_before"] == previous_threshold, "phase selection or threshold transition changed")
        previous_threshold = threshold
        require(call["start_s"] <= step["start_s"] <= decision["decision_s"] <= decision["applied_s"] <= step["end_s"] <= call["return_s"], "phase action timing")
        require(step["target_cap"] == step["actual_admission_cap"] == cap and step["actual_active"] <= cap
            and not step["n_preempted"] and not step["recomputed_tokens"] and not step["kv_adjusted_request_ids"], "admission or preemption changed")
        scheduled = {r["request_id"]: r for r in step["scheduled"]}
        require(len(scheduled) == len(step["scheduled"]) and set(scheduled) <= set(rows), "scheduled request identity")
        require(step["existing_decode_all_scheduled"] and set(step["existing_decode_request_ids"]) == ready
            and all(rid in scheduled and scheduled[rid]["scheduled_tokens"] == scheduled[rid]["decode_tokens"] == 1 for rid in ready), "ready decode skipped or over-advanced")
        for rid, row in scheduled.items():
            first.setdefault(rid, step["start_s"])
            require(rows[rid]["engine_add_return_s"] <= step["start_s"] and aliases[row["internal_request_id"]] == rid, "request scheduled before submission or alias mismatch")
            start, end, amount = row["scheduled_start_computed"], row["computed_after"], row["scheduled_tokens"]
            require(start == row["computed_before"] == progress[rid] and end - start == amount > 0 and not row["computed_adjustment"], "request position gap/recompute")
            expected_prefill = min(amount, max(0, rows[rid]["prompt_tokens"] - start))
            require(row["prefill_tokens"] == expected_prefill and row["decode_tokens"] == amount - expected_prefill
                and (not threshold or amount <= threshold), "prefill/decode or realized threshold accounting")
            progress[rid] = end
        prefill = sum(r["prefill_tokens"] for r in scheduled.values()); decode = sum(r["decode_tokens"] for r in scheduled.values())
        require(prefill + decode == step["total_scheduled_tokens"] <= 64 and decision["actual_prefill_rows"] == prefill
            and decision["actual_decode_rows"] == decode, "phase row/token budget mismatch")
        report.append(dict(step=step["step"], chosen_threshold=threshold, ready_decode_request_ids=sorted(ready),
            actual_prefill_rows=prefill, actual_decode_rows=decode, actual_active=step["actual_active"], waiting_requests=step["waiting_requests"],
            requests_with_prefill_above32=[rid for rid, r in scheduled.items() if r["prefill_tokens"] > 32],
            scheduled=step["scheduled"]))
    for rid, row in rows.items():
        require(progress[rid] == row["prompt_tokens"] + len(row["output_token_ids"]) - 1, "incomplete or duplicated request computation")
    require(sum(progress.values()) == 928, "total scheduled work changed")
    return first, dict(status="PASS", steps=report, total_scheduled_rows=sum(progress.values()),
        threshold_zero_steps=[s["step"] for s in report if s["chosen_threshold"] == 0],
        threshold_zero_prefill_total_above32_steps=[s["step"] for s in report if s["chosen_threshold"] == 0 and s["actual_prefill_rows"] > 32],
        actual_request_prefill_above32_steps=[s["step"] for s in report if s["requests_with_prefill_above32"]],
        final_observed_cap=cap, final_observed_threshold=previous_threshold,
        lifecycle_scope="One independent engine per cell. Capture restores its schedule callable; admission cap and threshold remain until engine shutdown, rather than being restored for reuse.")


def validate_warmup(path, workload, observer):
    signatures = {}; details = {}
    require(observer["mode"] == observer["warmup_mode"] == observer["measurement_mode"] == "nested" and observer["values_equal"], "common nested observer changed")
    require(observer["torch_memory_source_sha256"] == TORCH_MEMORY_SHA256, "installed Torch memory source differs from calibrated source")
    require(observer["check_start_s"] <= observer["check_end_s"], "nested observer check clock")
    samples = observer["samples_flat_nested_flat"]
    require(set(samples) == {"current", "peak"} and all(len(v) == 3 and len(set(v)) == 1 and all(type(x) is int and x >= 0 for x in v) for v in samples.values()), "nested observer field check unequal")
    evaluation_ids = {r["request_id"] for r in workload["source_requests"]}
    for name, mode in (("warmup", "static32"), ("warmup_phase32", "phase32")):
        raw = read(path / (name + ".json")); rows = raw["requests"]
        require(raw["status"] == "COMPLETE" and len(rows) == 3 and all(r["status"] == "completed" and r["prompt_tokens"] == 128
            and len(r["output_token_ids"]) == 2 for r in rows), "common warmup incomplete or workload changed")
        require(not ({r["request_id"] for r in rows} & evaluation_ids) and observer["check_end_s"] <= raw["measurement_origin_perf_counter_s"], "warmup/evaluation contamination or observer check ordering")
        require(all(s["phase_prefill"]["mode"] == mode and s["actual_admission_cap"] == 3 and s["existing_decode_all_scheduled"]
            and not s["n_preempted"] for s in raw["scheduler_steps"]), "warmup policy changed")
        signatures[name] = [{k: r[k] for k in ("request_id", "document_id", "prompt_token_ids_sha256", "prompt_tokens", "output_token_ids")} for r in rows]
        details[name] = dict(status="PASS", requests=3, output_tokens=6, scheduler_steps=len(raw["scheduler_steps"]), wall_s=raw["observation_end_s"])
    reset = read(path / "post_warmup_reset.json")
    require(reset["allocations_unchanged"] and not reset["kv_free_queue_modified"], "post-warmup allocation/reset changed")
    require(all(not c["expert_to_slot"] and c["lru_clock"] == 0 and all(x == -1 for x in c["slot_to_expert"] + c["expert_map_device"])
        and not any(c["lru_tick"]) for c in reset["cache_after"].values()), "post-warmup pager not empty")
    blocks = [b for b in reset["kv_blocks"] if not b["is_null"]]
    require({b["block_id"] for b in blocks} == set(reset["free_kv_block_ids_in_order"])
        and all(not b["ref_count"] and b["block_hash"] is None for b in blocks), "post-warmup KV not free")
    return details, signatures, reset


def cell(root):
    config = read(root / "config.json"); workload = read(root / "workload.json"); entries = read(root / "episodes.json")
    cap, mode = config["admission_limit"], config["phase_policy"]; arm = f"cap{cap}_{mode}"
    require(arm in ARMS and config["continuous_admission"] and config["nested_memory_observer"], "continuous arm or observer changed")
    require(workload["source_indices"] == SOURCES and list(map(len, workload["actual_prompt_token_ids"])) == PROMPTS
        and workload["arrival_traces_s"]["steady"] == ARRIVALS, "evaluation workload changed")
    ids = [r["request_id"] for r in workload["source_requests"]]
    require(len(set(ids)) == 8 and [workload["output_tokens_by_request"][rid] for rid in ids] == OUTPUTS, "output targets changed")
    require(config["warmup_source_indices"] == [0, 1, 2] and config["warmup_prompt_tokens"] == 128, "warmup sources/prefix changed")
    require((config["expert_cap"], config["kv_bytes"], config["token_budget"]) == (16, 536870912, 64)
        and config["execution"] == "expert" and not config["verify_kernel"] and config["trace_retention"] == "episode", "resource/trace protocol changed")
    require(len(entries) == 1 and entries[0]["execution"] == "expert" and entries[0]["variant"] == arm, "cell episode identity")
    entry = entries[0]; path = root / entry["phase"].split("/")[0]
    raw = read(path / "raw.json"); engine = read(root / "engine_args.json"); res = read(path / "measurement_resources.json")
    require((engine["max_num_seqs"], engine["max_num_batched_tokens"], engine["kv_cache_memory_bytes"]) == (3, 64, 536870912)
        and not engine["enable_prefix_caching"] and not engine["async_scheduling"], "compiled native budget changed")
    metrics, issues = requests(raw, workload); require(not issues, str(issues))
    require([r["request_id"] for r in raw["requests"]] == ids, "raw request/source ordering differs")
    require(all(r["status"] == "completed" and len(r["output_token_ids"]) == OUTPUTS[i] and r["arrival_s"] == ARRIVALS[i]
        and r["completion_s"] == r["token_times_s"][-1] for i, r in enumerate(raw["requests"])), "request targets/arrival/receipt mismatch")
    first, schedule = validate_schedule(raw, workload, cap, mode)
    for row in raw["requests"]:
        rid = row["request_id"]; metric = metrics[rid]
        require(row["submission_lag_s"] == row["admission_s"] - row["arrival_s"] >= 0, "submission lag denominator")
        metric.update(first_schedule_s=first[rid], completion_s=row["completion_s"],
            submission_lag_s=row["submission_lag_s"], native_queue_until_first_schedule_s=first[rid] - row["admission_s"],
            arrival_until_first_schedule_s=first[rid] - row["arrival_s"], first_schedule_to_first_token_s=row["token_times_s"][0] - first[rid])
        require(abs(metric["ttft_s"] - metric["submission_lag_s"] - metric["native_queue_until_first_schedule_s"] - metric["first_schedule_to_first_token_s"]) < 1e-9, "TTFT partition")
    observer = read(root / "memory_observer.json"); warmups, warm_signatures, reset = validate_warmup(path, workload, observer)
    require(res["expert_cap"] == 16 and res["actual_unique_kv_storage_bytes"] == 536870912 and res["measured_execution"] == "expert"
        and res["arm"] == arm and res["scheduler_requests"] == 0 and res["cache"] == reset["cache_after"]
        and res["allocations"] == reset["allocations"], "measurement resources or post-warmup state mismatch")
    pager = read(root / "pager_summary.json"); layers = {r["layer_name"]: r for r in pager["layers"]}
    require(len(layers) == 16 and pager["cap"] == 16 and all(r["cap"] == 16 for r in layers.values()), "expert layer capacity")
    trace = [json.loads(line) for line in (root / "pager/calls.jsonl").read_text().splitlines() if line.strip()]
    records = [r for r in trace if r["measurement"]]
    require(len(trace) == pager["all_calls"] and len({r["call_id"] for r in trace}) == len(trace)
        and all(r["context"]["phase"] == entry["phase"] for r in records), "trace coverage or phase")
    resident = {k: set(map(int, v["expert_to_slot"])) for k, v in res["cache"].items()}; accounts = []; signature = []; aliases = raw["internal_to_source"]
    for step in raw["scheduler_steps"]:
        rr = [r for r in records if r["context"]["step_id"] == step["step"]]
        require(Counter(r["layer_name"] for r in rr) == Counter(layers.keys()), "16-layer/step coverage")
        expected = Counter((r["request_id"], pos) for r in step["scheduled"] for pos in range(r["scheduled_start_computed"], r["computed_after"]))
        for record in rr:
            physical = [(aliases[r["internal_request_id"]], r["computed_position"]) for r in record["context"]["rows"]]
            require(record["execution"] == "expert" and Counter(physical) == expected and record["rows"] == step["total_scheduled_tokens"], "actual row/position alignment")
            accounts.append(account(record, resident[record["layer_name"]], layers[record["layer_name"]], 16))
            signature.append(dict(step=step["step"], layer=record["layer_name"], rows=physical, row_topk_experts=record["row_topk_experts"],
                groups=[{k: g[k] for k in ("required_experts", "loaded_experts", "evicted_experts")} for g in record["groups"]]))
    require(len(accounts) == len(records) == pager["measurement_calls"], "measurement trace lost/doubled")
    totals = {key: sum(a[key] for a in accounts) for key in accounts[0]}
    require(totals["weight_copy_bytes"] == pager["measurement"]["weight_copy_bytes"], "measurement byte total")
    mapping = read(path / "map_optimization.json")
    require(mapping["validation"]["status"] == "PASS" and mapping["validation"]["layers"] == 16 and mapping["statistics"]["mode"] == "batched"
        and mapping["statistics"]["totals"]["misses"] == totals["misses"] and mapping["statistics"]["totals"]["ensure_calls"] == totals["groups"]
        and mapping["statistics"]["totals"]["scalar_device_assignments"] == 0, "batched map/account mismatch")
    gpu = [json.loads(line) for line in (root / "gpu_checks.jsonl").read_text().splitlines() if line.strip()]
    stages = {g["stage"] for g in gpu}
    require(all(g["decision"] == "PASS" and not g["foreign_pids"] and not g["query_errors"] for g in gpu)
        and {"before_initialization", "before_model_load", "after_run", path.name + "/before_reset", entry["phase"] + "/after_measurement"} <= stages, "GPU boundary failure")
    events = read(root / "runtime_events.json")["events"]; origin = raw["measurement_origin_perf_counter_s"]; phases = {}
    for call in raw["engine_calls"]:
        step = raw["scheduler_steps"][call["scheduler_step_start"]]; decision = step["phase_prefill"]
        prefill, decode = decision["actual_prefill_rows"], decision["actual_decode_rows"]
        phase = "mixed_prefill_decode" if prefill and decode else "prefill_only" if prefill else f"decode_only_{decode}" if decode else "empty"
        rr = [r for r in records if r["context"]["step_id"] == step["step"]]
        values = dict(engine_calls=1, wall_s=call["return_s"] - call["start_s"], thread_cpu_s=call["cpu_delta"]["thread_cpu_ns"] / 1e9,
            process_cpu_s=call["cpu_delta"]["process_cpu_ns"] / 1e9, weight_copy_bytes=sum(r["weight_copy_bytes"] for r in rr),
            groups=sum(r["group_count"] for r in rr), host_apply_ms=sum(r["host_apply_ms"] for r in rr),
            load_cuda_span_ms=sum(g["load_cuda_span_ms"] for r in rr for g in r["groups"]),
            gc_callback_overlap_ms=gc_overlap(events, origin + call["start_s"], origin + call["return_s"]))
        phase_sum = phases.setdefault(phase, {k: 0 for k in values})
        for key, value in values.items(): phase_sum[key] += value
    phase_totals = {k: sum(p[k] for p in phases.values()) for k in next(iter(phases.values()))}
    require(phase_totals["weight_copy_bytes"] == totals["weight_copy_bytes"] and phase_totals["groups"] == totals["groups"], "phase byte/group partition")
    normalized = dict(workload=workload, warmups=warm_signatures, initial_cache=res["cache"], signature=signature,
        outputs={rid: m["output_token_ids"] for rid, m in metrics.items()})
    return dict(status="PASS", path=str(root), arm=arm, requests=metrics, schedule=schedule, warmups=warmups, memory_observer=observer,
        request_count=len(metrics), generated_tokens=sum(len(m["output_token_ids"]) for m in metrics.values()), engine_calls=len(raw["engine_calls"]), layer_calls=len(records),
        metrics=dict(whole_wall_s=raw["observation_end_s"], completion_makespan_s=max(r["completion_s"] for r in raw["requests"]),
            request_mean_completion_latency_s=sum(m["completion_latency_s"] for m in metrics.values()) / 8,
            request_mean_ttft_s=sum(m["ttft_s"] for m in metrics.values()) / 8, all_request_max_itl_s=max(m["max_itl_s"] for m in metrics.values()),
            max_submission_lag_s=max(m["submission_lag_s"] for m in metrics.values()), engine_wall_s=phase_totals["wall_s"],
            engine_observation_thread_cpu_s=phase_totals["thread_cpu_s"], engine_observation_process_cpu_s=phase_totals["process_cpu_s"],
            gc_callback_overlap_ms=gc_overlap(events, origin, origin + raw["observation_end_s"]), **totals),
        phases=phases, outside_engine_gap_s=raw["observation_end_s"] - phase_totals["wall_s"],
        allocator_trajectory=allocator_trajectory(records), cuda_memory=read(path / "cuda_memory.json"), map_optimization=mapping,
        resources=res, gpu_checks=gpu, environment=read(root / "environment.json"), engine_args=engine,
        cycle_complete=read(root / "cycle_complete.json"), config=config), normalized


def analyze(root):
    root = Path(root); stage = root.parent / "launch"; protocol = read(stage / "protocol.json")
    require(protocol["source_indices"] == SOURCES and protocol["prompt_tokens"] == PROMPTS and protocol["output_tokens"] == OUTPUTS
        and protocol["arrival_times_s"] == ARRIVALS, "frozen protocol differs from declared analysis inputs")
    launch = read(stage / "launch.json")
    require(launch["status"] == "EXITED" and launch["exit_code"] == 0 and not launch.get("error"), "parent launch incomplete/failed")
    paths = [root / c["name"] for c in protocol["cells"]]
    require(len(paths) == 6, "need all six independent-engine cells")
    require([c["name"] for c in launch["cells"]] == [p.name for p in paths], "parent execution order/coverage differs")
    source_hashes = {}
    for name, expected in protocol["source_sha256"].items():
        source_hashes[name] = hashlib.sha256((stage / name).read_bytes()).hexdigest()
        require(source_hashes[name] == expected, "returned source differs from frozen protocol: " + name)
    results, normalized = {}, {}
    for order, path in enumerate(paths):
        key = str(path.relative_to(root)); results[key], normalized[key] = cell(path)
        results[key].update(order=order, block=order // 3)
        environment = results[key]["environment"]
        for name, actual in environment["sources"].items():
            require(actual == source_hashes[name], "runtime/returned source mismatch: " + name)
        cell_launch = read(stage / key / "launch.json")
        require(cell_launch == launch["cells"][order] and cell_launch["status"] == "EXITED" and cell_launch["exit_code"] == 0, "cell launch incomplete/failed")
        hardware = [json.loads(line) for line in (stage / key / "hardware.jsonl").read_text().splitlines() if line.strip()]
        own = {g["caller_pid"] for g in results[key]["gpu_checks"]}
        foreign = sorted({p["pid"] for h in hardware for p in h["processes"]} - own)
        require(hardware and not foreign and not any(h.get("error") for h in hardware), "hardware sample failure/foreign process")
        preflight = [json.loads(line) for line in (stage / key / "preflight.jsonl").read_text().splitlines() if line.strip()]
        require(len(preflight) >= 3 and preflight[-1]["consecutive_idle"] == 3
            and all(not p["gpu"] and not p["launchers"] for p in preflight[-3:]), "three idle preflight boundaries missing")
        results[key].update(launch=cell_launch, hardware=dict(sample_count=len(hardware), observed_worker_pids=sorted(own), foreign_sampled_pids=foreign,
            temperature_range_c=[min(h["temperature_c"] for h in hardware), max(h["temperature_c"] for h in hardware)],
            power_range_mw=[min(h["power_mw"] for h in hardware), max(h["power_mw"] for h in hardware)]))
    names = list(results)
    require([results[n]["arm"] for n in names] == ARMS + list(reversed(ARMS)), "six-cell execution order differs from declared ABCCBA")
    require(all(n["workload"] == normalized[names[0]]["workload"] for n in normalized.values()), "cross-cell declared evaluation inputs differ")
    lookup = {(r["block"], r["arm"]): name for name, r in results.items()}
    pairs = [(lookup[block, a], lookup[block, b], f"block{block}_{b}_minus_{a}") for block in range(2)
        for a, b in (("cap3_static32", "cap2_static32"), ("cap2_static32", "cap2_phase32"), ("cap3_static32", "cap2_phase32"))]
    pairs.extend((lookup[0, arm], lookup[1, arm], arm + "_repeat") for arm in ARMS)
    comparisons = []
    request_fields = ("ttft_s", "tpot_s", "max_itl_s", "completion_latency_s", "submission_lag_s", "native_queue_until_first_schedule_s", "first_schedule_to_first_token_s")
    for a, b, kind in pairs:
        x, y = results[a], results[b]
        comparisons.append(dict(a=a, b=b, kind=kind, delta_b_minus_a={k: y["metrics"][k] - value for k, value in x["metrics"].items()},
            change_b_over_a_pct={k: 100 * (y["metrics"][k] / value - 1) if value else None for k, value in x["metrics"].items()},
            per_request_delta_b_minus_a={rid: {k: y["requests"][rid][k] - row[k] for k in request_fields} for rid, row in x["requests"].items()},
            observed_equality={k: normalized[a][k] == normalized[b][k] for k in normalized[a]},
            output_equal_by_request={rid: normalized[a]["outputs"][rid] == normalized[b]["outputs"][rid] for rid in x["requests"]}))
    return dict(status="MEASUREMENT_ONLY", integrity="PASS", cells=results, comparisons=comparisons, launch=launch,
        source_sha256=source_hashes, runtime_source_hashes_matched=sorted(next(iter(results.values()))["environment"]["sources"]),
        source_scope="Returned sources match frozen protocol; each runtime-recorded source hash matches returned source. Nested observer separately checks the installed Torch memory source. Finite GPU snapshots support their recorded boundaries, not continuous exclusivity proof.",
        total_request_executions=sum(c["request_count"] for c in results.values()), total_generated_tokens=sum(c["generated_tokens"] for c in results.values()),
        strongest_phase_baseline="cap2_static32", scope="Eight clock arrivals per independent engine, six ABCCBA cells, common disjoint calibration sources. Full request metrics and all same-arm drift retained; no SLO, confidence/noise bound, Oracle, statistical or method GO. Routes and outputs evolve under each actual policy; they are compared descriptively without requiring equality.",
        timing_scope="Whole capture equals exclusive engine-call phase wall plus outside gap, including clock idle, synchronous submission lag and observation/bookkeeping. Request TTFT equals submission lag plus native time to first schedule plus first-schedule-to-first-token. Process/thread CPU, host apply, CUDA load spans and GC overlap and must not be summed or subtracted.",
        lifecycle_scope="Common nested observer covers warmups and measurement. Each cell loads, runs, drains and shuts down its own engine; process wall and engine cycles retained separately, without allocation to requests. No claim that admission cap/threshold are restored for shared-engine reuse. Matching pager metadata is not KV tensor bitwise equivalence.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path); parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(); result = analyze(args.input_dir)
    with args.out.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False); stream.write("\n")
