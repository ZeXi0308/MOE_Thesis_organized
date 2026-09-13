"""Read the frozen Qwen static32/16/16/32 captures without altering evidence."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "admission_memory_observer_v026"))
from analyze_admission_memory_observer import account, allocator_trajectory, digest, gc_overlap, read, requests, require
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "qwen3_native_qualification_v026"))
from analyze_qualification_v2 import check_block_rounded_kv

CHUNKS = [32, 16, 16, 32]
PROMPTS, OUTPUTS, ARRIVALS = [64, 64, 128, 32], [32, 32, 16, 24], [0, 0, 2, 4]
PREPARED_SHA256 = "714931e38ed05732c04a476e18e43124ab230ef0955ffdbd6d8a4c6cdf920780"
TORCH_MEMORY_SHA256 = "71a36cb635c0a076bf8cda0ae08c112d7b0e15d9d2fb21a0dee373e2967c4700"
TOKENIZER = dict(repository="Qwen/Qwen3-30B-A3B", revision="ad44e777bcd18fa416d9da3bd8f70d33ebb85d39")
MODEL_SHAPE = dict(layers=48, experts=128, top_k=8, hidden=2048, intermediate=768)
EXPERT_BYTES = 9437184


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def validate_schedule(raw, chunk, expected_work, initial_threshold=None):
    rows = {r["request_id"]: r for r in raw["requests"]}; aliases = raw["internal_to_source"]
    progress = {rid: 0 for rid in rows}; first = {}; report = []; previous = initial_threshold
    require(raw["status"] == "COMPLETE" and not raw.get("error") and raw["event_arrival"] is None
        and not raw["event_actions"] and not raw["prefill_release_actions"], "capture failed or event action appeared")
    require(raw["target_cap"] == 4 and raw["policy"] == "static" and not raw["allow_preemption"], "capture policy changed")
    calls, steps = raw["engine_calls"], raw["scheduler_steps"]
    require([i for c in calls for i in range(c["scheduler_step_start"], c["scheduler_step_stop"])] == list(range(len(steps))), "engine/step coverage")
    last_return = 0
    for i, call in enumerate(calls):
        require(call["index"] == i and call["returned"] and last_return <= call["start_s"] <= call["return_s"] <= raw["observation_end_s"], "engine call timing/status")
        last_return = call["return_s"]
        require(call["scheduler_step_stop"] == call["scheduler_step_start"] + 1, "synchronous call needs one scheduler step")
        step = steps[call["scheduler_step_start"]]; decision = step["phase_prefill"]
        require(step["step"] == call["scheduler_step_start"] and call["phase_prefill"] == decision, "decision/step alignment")
        ready = {aliases[rid] for rid in decision["ready_decode_ids"]}
        observed = {rid for rid, row in rows.items() if row["token_times_s"][0] <= decision["decision_s"] < row["completion_s"]}
        require(ready == observed, "ready set differs from output-bearing unfinished requests")
        require(decision["mode"] == f"static{chunk}" and decision["status"] == "applied"
            and decision["effective_token_budget"] == 64 and decision["chosen_threshold"] == chunk
            and (previous is None or decision["threshold_before"] == previous), "actual static threshold/budget mismatch")
        previous = chunk
        require(call["start_s"] <= step["start_s"] <= decision["decision_s"] <= decision["applied_s"] <= step["end_s"] <= call["return_s"], "action timing")
        require(step["target_cap"] == step["actual_admission_cap"] == 4 and step["actual_active"] <= 4
            and not step["n_preempted"] and not step["recomputed_tokens"] and not step["kv_adjusted_request_ids"], "admission/preemption changed")
        scheduled = {r["request_id"]: r for r in step["scheduled"]}
        require(len(scheduled) == len(step["scheduled"]) and set(scheduled) <= set(rows), "scheduled identity")
        require(step["existing_decode_all_scheduled"] and set(step["existing_decode_request_ids"]) == ready
            and all(rid in scheduled and scheduled[rid]["scheduled_tokens"] == scheduled[rid]["decode_tokens"] == 1 for rid in ready), "ready decode skipped or advanced twice")
        for rid, row in scheduled.items():
            first.setdefault(rid, step["start_s"])
            require(rows[rid]["engine_add_return_s"] <= step["start_s"] and aliases[row["internal_request_id"]] == rid, "schedule before submission or alias mismatch")
            start, end, amount = row["scheduled_start_computed"], row["computed_after"], row["scheduled_tokens"]
            require(start == row["computed_before"] == progress[rid] and 0 < end - start == amount <= chunk and not row["computed_adjustment"], "request position/threshold mismatch")
            prefill = min(amount, max(0, rows[rid]["prompt_tokens"] - start))
            require(row["prefill_tokens"] == prefill and row["decode_tokens"] == amount - prefill, "prefill/decode row accounting")
            progress[rid] = end
        prefill = sum(r["prefill_tokens"] for r in scheduled.values()); decode = sum(r["decode_tokens"] for r in scheduled.values())
        require(prefill + decode == step["total_scheduled_tokens"] <= 64 and decision["actual_prefill_rows"] == prefill
            and decision["actual_decode_rows"] == decode, "realized rows/token budget mismatch")
        report.append(dict(step=step["step"], chosen_threshold=chunk, ready_decode_request_ids=sorted(ready),
            prefill_rows=prefill, decode_rows=decode, active=step["actual_active"], waiting=step["waiting_requests"], scheduled=step["scheduled"]))
    for rid, row in rows.items():
        require(progress[rid] == row["prompt_tokens"] + len(row["output_token_ids"]) - 1, "incomplete or duplicate computation")
    require(sum(progress.values()) == expected_work, "scheduled work changed")
    prefill_chunks = {rid: [r["prefill_tokens"] for s in report for r in s["scheduled"] if r["request_id"] == rid and r["prefill_tokens"]] for rid in rows}
    require(max(n for v in prefill_chunks.values() for n in v) == chunk, "static prefill threshold never realized")
    return first, dict(status="PASS", total_scheduled_rows=expected_work, actual_prefill_chunks=prefill_chunks,
        request_reached_chunk_bound={rid: max(v) == chunk for rid, v in prefill_chunks.items()}, steps=report)


def validate_reset(reset):
    require(reset["allocations_unchanged"] and not reset["kv_free_queue_modified"], "reset changed allocations or KV free queue")
    require(len(reset["cache_after"]) == 48 and all(not c["expert_to_slot"] and c["lru_clock"] == 0
        and all(x == -1 for x in c["slot_to_expert"] + c["expert_map_device"]) and not any(c["lru_tick"])
        for c in reset["cache_after"].values()), "post-reset pager not empty")
    blocks = [b for b in reset["kv_blocks"] if not b["is_null"]]
    require({b["block_id"] for b in blocks} == set(reset["free_kv_block_ids_in_order"])
        and all(not b["ref_count"] and b["block_hash"] is None for b in blocks), "post-reset KV not free")


def validate_warmups(path, workload, measurement):
    details, signatures = {}, {}; previous_end = None
    require({p.stem for p in path.glob("warmup*.json")} == {"warmup_static16", "warmup_static32"}, "common warmup set changed")
    for chunk in (16, 32):
        name = f"warmup_static{chunk}"; raw = read(path / (name + ".json")); rows = raw["requests"]
        _, issues = requests(raw, workload); require(not issues, str(issues))
        require(len(rows) == 4 and all(r["status"] == "completed" and len(r["output_token_ids"]) == 2 and r["arrival_s"] == 0 for r in rows), "warmup workload/status changed")
        _, schedule = validate_schedule(raw, chunk, sum(PROMPTS) + 4)
        start = raw["measurement_origin_perf_counter_s"]; end = start + raw["observation_end_s"]
        require((previous_end is None or previous_end <= start) and end <= measurement["measurement_origin_perf_counter_s"], "warmup order or measurement boundary")
        previous_end = end
        details[name] = dict(status="PASS", requests=4, outputs=8, scheduled_rows=schedule["total_scheduled_rows"], engine_calls=len(raw["engine_calls"]), wall_s=raw["observation_end_s"])
        signatures[name] = [{k: r[k] for k in ("request_id", "document_id", "prompt_token_ids_sha256", "prompt_tokens", "output_token_ids")} for r in rows]
    reset = read(path / "post_warmup_reset.json"); validate_reset(reset); validate_reset(read(path / "reset.json"))
    require(reset["common_warmup_sequence"] == [16, 32] and reset["warmup_outputs_per_request"] == 2
        and reset["warmup_arrivals_s"] == [0] * 4 and not reset["reference_validation_enabled"], "post-warmup policy mismatch")
    return details, signatures, reset


def cell(root, entry, workload, layers, trace, events, kv_inputs):
    chunk = entry["chunk"]; path = root / entry["phase"].split("/")[0]; raw = read(path / "raw.json")
    ids = [r["request_id"] for r in workload["source_requests"]]
    require(entry["status"] == "COMPLETE" and entry["execution"] == "expert" and entry["observation_end_s"] == raw["observation_end_s"], "episode failed or timing changed")
    require([r["request_id"] for r in raw["requests"]] == ids and all(r["status"] == "completed" and len(r["output_token_ids"]) == OUTPUTS[i]
        and r["arrival_s"] == ARRIVALS[i] and r["completion_s"] == r["token_times_s"][-1] for i, r in enumerate(raw["requests"])), "request targets/arrivals/status changed")
    metrics, issues = requests(raw, workload); require(not issues, str(issues))
    first, schedule = validate_schedule(raw, chunk, 388, initial_threshold=chunk)
    for row in raw["requests"]:
        rid = row["request_id"]; metric = metrics[rid]
        require(row["arrival_s"] <= row["admission_s"] <= row["engine_add_return_s"] <= first[rid]
            and row["submission_lag_s"] == row["admission_s"] - row["arrival_s"], "arrival/submission/queue order")
        metric.update(admission_s=row["admission_s"], engine_add_return_s=row["engine_add_return_s"], first_schedule_s=first[rid], completion_s=row["completion_s"],
            submission_lag_s=row["submission_lag_s"], native_queue_until_first_schedule_s=first[rid] - row["admission_s"],
            arrival_until_first_schedule_s=first[rid] - row["arrival_s"], first_schedule_to_first_token_s=row["token_times_s"][0] - first[rid])
        require(abs(metric["ttft_s"] - metric["submission_lag_s"] - metric["native_queue_until_first_schedule_s"] - metric["first_schedule_to_first_token_s"]) < 1e-9, "TTFT partition")
    warmups, warm_signatures, reset = validate_warmups(path, workload, raw)
    policy = read(path / "policy_application.json"); res = read(path / "measurement_resources.json")
    require(policy["measured_prefill_limit"] == chunk and policy["applied_on_drained_engine"] and policy["common_warmup_sequence"] == [16, 32]
        and not policy["reference_validation_enabled"], "drained policy application changed")
    kv_accounting = check_block_rounded_kv(measured=res, **kv_inputs)
    require(res["expert_cap"] == 48 and res["measured_execution"] == "expert"
        and res["measured_chunk"] == chunk and res["arm"] == entry["arm"] and res["scheduler_requests"] == 0
        and res["cache"] == reset["cache_after"] == read(path / "measurement_initial_cache.json") and res["allocations"] == reset["allocations"], "resource/reset mismatch")
    records = [r for r in trace if r["context"]["phase"] == entry["phase"]]
    resident = {k: set(map(int, v["expert_to_slot"])) for k, v in res["cache"].items()}; accounts = []; signature = []; phases = {}; pure = []
    aliases = raw["internal_to_source"]; origin = raw["measurement_origin_perf_counter_s"]
    for call, step in zip(raw["engine_calls"], raw["scheduler_steps"]):
        rr = [r for r in records if r["context"]["step_id"] == step["step"]]
        require(Counter(r["layer_name"] for r in rr) == Counter(layers.keys()), "48-layer/step coverage")
        expected = Counter((r["request_id"], pos) for r in step["scheduled"] for pos in range(r["scheduled_start_computed"], r["computed_after"]))
        prefill = sum(r["prefill_tokens"] for r in step["scheduled"]); decode = sum(r["decode_tokens"] for r in step["scheduled"])
        for record in rr:
            physical = [(aliases[r["internal_request_id"]], r["computed_position"]) for r in record["context"]["rows"]]
            require(record["execution"] == record["grouping_axis"] == "expert" and Counter(physical) == expected
                and record["rows"] == step["total_scheduled_tokens"] and not record.get("validation_run"), "actual row alignment or reference validation")
            require(all(len(row) == len(set(row)) == 8 for row in record["row_topk_experts"]), "real top-k differs from 8")
            accounts.append(account(record, resident[record["layer_name"]], layers[record["layer_name"]], 48))
            signature.append(dict(step=step["step"], layer=record["layer_name"], rows=physical, topk=record["row_topk_experts"],
                groups=[{k: g[k] for k in ("required_experts", "loaded_experts", "evicted_experts")} for g in record["groups"]]))
            if decode and not prefill:
                require(decode <= 4 and len(record["active_experts"]) <= decode * 8 <= 32 and record["group_count"] == 1, "pure-decode single-group bound violated")
                pure.append(dict(step=step["step"], layer=record["layer_name"], rows=decode, active_experts=len(record["active_experts"]),
                    groups=record["group_count"], misses=record["miss"], weight_copy_bytes=record["weight_copy_bytes"]))
        label = "mixed_prefill_decode" if prefill and decode else "prefill_only" if prefill else f"decode_only_{decode}" if decode else "empty"
        values = dict(engine_calls=1, wall_s=call["return_s"] - call["start_s"], thread_cpu_s=call["cpu_delta"]["thread_cpu_ns"] / 1e9,
            process_cpu_s=call["cpu_delta"]["process_cpu_ns"] / 1e9, weight_copy_bytes=sum(r["weight_copy_bytes"] for r in rr),
            groups=sum(r["group_count"] for r in rr), host_apply_ms=sum(r["host_apply_ms"] for r in rr),
            load_cuda_span_ms=sum(g["load_cuda_span_ms"] for r in rr for g in r["groups"]),
            gc_callback_overlap_ms=gc_overlap(events, origin + call["start_s"], origin + call["return_s"]))
        subtotal = phases.setdefault(label, {k: 0 for k in values})
        for k, v in values.items(): subtotal[k] += v
    require(len(accounts) == len(records), "lost or repeated layer call")
    totals = {k: sum(a[k] for a in accounts) for k in accounts[0]}
    phase_totals = {k: sum(p[k] for p in phases.values()) for k in next(iter(phases.values()))}
    require(phase_totals["weight_copy_bytes"] == totals["weight_copy_bytes"] and phase_totals["groups"] == totals["groups"], "phase byte/group partition")
    mapping = read(path / "map_optimization.json")
    require(mapping["validation"]["status"] == "PASS" and mapping["validation"]["layers"] == 48 and mapping["statistics"]["mode"] == "batched"
        and mapping["statistics"]["totals"]["misses"] == totals["misses"] and mapping["statistics"]["totals"]["ensure_calls"] == totals["groups"]
        and mapping["statistics"]["totals"]["scalar_device_assignments"] == 0, "batched map/account mismatch")
    normalized = dict(initial_cache=res["cache"], allocations=res["allocations"], warmups=warm_signatures, signature=signature,
        outputs={rid: m["output_token_ids"] for rid, m in metrics.items()})
    result = dict(status="PASS", path=str(path), arm=entry["arm"], chunk=chunk, block=entry["block"], order=entry["order"], requests=metrics,
        schedule=schedule, warmups=warmups, request_count=4, generated_tokens=104, engine_calls=len(raw["engine_calls"]), layer_calls=len(records),
        metrics=dict(whole_wall_s=raw["observation_end_s"], engine_call_count=len(raw["engine_calls"]), layer_call_count=len(records),
            completion_makespan_s=max(r["completion_s"] for r in raw["requests"]),
            request_mean_completion_latency_s=sum(m["completion_latency_s"] for m in metrics.values()) / 4,
            request_mean_ttft_s=sum(m["ttft_s"] for m in metrics.values()) / 4, request_mean_tpot_s=sum(m["tpot_s"] for m in metrics.values()) / 4,
            all_request_max_itl_s=max(m["max_itl_s"] for m in metrics.values()), max_submission_lag_s=max(m["submission_lag_s"] for m in metrics.values()),
            engine_wall_s=phase_totals["wall_s"], engine_observation_thread_cpu_s=phase_totals["thread_cpu_s"], engine_observation_process_cpu_s=phase_totals["process_cpu_s"],
            gc_callback_overlap_ms=gc_overlap(events, origin, origin + raw["observation_end_s"]), **totals),
        phases=phases, outside_engine_gap_s=raw["observation_end_s"] - phase_totals["wall_s"],
        pure_decode=dict(status="PASS" if pure else "NOT_COVERED", layer_calls=len(pure), layers_with_h2d=sum(p["misses"] > 0 for p in pure),
            weight_copy_bytes=sum(p["weight_copy_bytes"] for p in pure), observations=pure,
            scope="At most four decode rows times top8 gives at most32 active experts, within cap48. One expert group does not imply cache hits or no H2D."),
        allocator_trajectory=allocator_trajectory(records), cuda_memory=read(path / "cuda_memory.json"), map_optimization=mapping, resources=res, kv_accounting=kv_accounting,
        signature_sha256=digest(signature), measurement_origin_perf_counter_s=origin)
    return result, normalized


def analyze(root, stage=None):
    root = Path(root)
    if stage is None:
        stage = root.parent / "launch"
        if not stage.is_dir(): stage = root.parent.parent / "launch"
    stage = Path(stage)
    prepared_path = Path(__file__).resolve().parent / "prepared/workload.json"
    require(hashlib.sha256(prepared_path.read_bytes()).hexdigest() == PREPARED_SHA256, "frozen prepared input changed")
    frozen = read(prepared_path); workload = read(root / "workload.json"); config = read(root / "config.json")
    require(all(workload[k] == v for k, v in frozen.items()) and workload["source_indices"] == [4, 5, 6, 7]
        and workload["tokenizer_identity"] == TOKENIZER and list(map(len, workload["actual_prompt_token_ids"])) == PROMPTS
        and workload["arrival_traces_s"] == {"steady": ARRIVALS}, "frozen workload mismatch")
    ids = [r["request_id"] for r in workload["source_requests"]]
    require(len(set(ids)) == 4 and [workload["output_tokens_by_request"][rid] for rid in ids] == OUTPUTS, "output targets changed")
    require(config["qwen_static_prefill"] and config["execution"] == "expert" and not config["verify_kernel"]
        and (config["requests"], config["expert_cap"], config["kv_bytes"], config["token_budget"]) == (4, 48, 536870912, 64)
        and config["trace_retention"] == "episode" and config["group_retention"] == "none", "configuration changed")
    engine = read(root / "engine_args.json"); shape = read(root / "model_shape.json")
    require(shape == MODEL_SHAPE and (engine["max_num_seqs"], engine["max_num_batched_tokens"], engine["kv_cache_memory_bytes"]) == (4, 64, 536870912)
        and not engine["enable_prefix_caching"] and not engine["async_scheduling"] and engine["dtype"] in ("bfloat16", "torch.bfloat16"), "native budget/model/dtype changed")
    model_path = Path(__file__).resolve().parent / "model_metadata/config.json"
    require(hashlib.sha256(model_path.read_bytes()).hexdigest() == read(model_path.parent.parent / "qwen3.manifest.json")["config_sha256"], "frozen model KV configuration")
    kv_inputs = dict(engine=engine, resources=read(root / "resources.json"), model_config=read(model_path))
    observer = read(root / "memory_observer.json"); samples = observer["samples_flat_nested_flat"]
    require(observer["mode"] == observer["warmup_mode"] == observer["measurement_mode"] == "nested" and observer["values_equal"]
        and observer["torch_memory_source_sha256"] == TORCH_MEMORY_SHA256 and set(samples) == {"current", "peak"}
        and all(len(v) == 3 and len(set(v)) == 1 and all(type(x) is int and x >= 0 for x in v) for v in samples.values()), "common nested observer mismatch")
    require(read(root / "status.json")["status"] == "COMPLETE", "runner did not complete")
    pager = read(root / "pager_summary.json"); layers = {r["layer_name"]: r for r in pager["layers"]}
    require(len(layers) == 48 and pager["cap"] == 48 and not pager["validation_run"] and not pager["kernel_validation"]
        and all(r["cap"] == 48 and r["num_experts"] == 128 and r["pinned_bytes"] == 128 * EXPERT_BYTES
            and r["scratch_bytes"] == 48 * EXPERT_BYTES for r in layers.values()), "expert allocations or disabled reference validation mismatch")
    trace = lines(root / "pager/calls.jsonl"); entries = read(root / "episodes.json"); events = read(root / "runtime_events.json")["events"]
    require(len(trace) == pager["all_calls"] and len({r["call_id"] for r in trace}) == len(trace) and not any(r.get("validation_run") for r in trace), "trace coverage/reference work")
    require(len(entries) == 4 and all(e["chunk"] == CHUNKS[i] and e["arm"] == e["variant"] == f"static{CHUNKS[i]}"
        and (e["block"], e["order"], e["order_in_block"]) == (i // 2, i, i % 2) and e["ordering"] == "32/16/16/32 ABBA"
        and e["phase"] == f"repeat_{i}_static{CHUNKS[i]}/measurement" for i, e in enumerate(entries)), "ABBA episode identity/order changed")
    cells, normalized = {}, {}
    for entry in entries:
        key = entry["phase"].split("/")[0]; cells[key], normalized[key] = cell(root, entry, workload, layers, trace, events, kv_inputs)
    require(sum(c["layer_calls"] for c in cells.values()) == sum(r["measurement"] for r in trace) == pager["measurement_calls"]
        and sum(c["metrics"]["weight_copy_bytes"] for c in cells.values()) == pager["measurement"]["weight_copy_bytes"], "campaign measurement accounting")
    require(all(v["initial_cache"] == next(iter(normalized.values()))["initial_cache"] and v["allocations"] == next(iter(normalized.values()))["allocations"]
        for v in normalized.values()), "same-engine empty initial cache/allocation metadata changed")
    gpu = lines(root / "gpu_checks.jsonl"); required_stages = {"before_initialization", "before_model_load", "after_run"}
    required_stages.update(e["phase"].split("/")[0] + "/before_reset" for e in entries); required_stages.update(e["phase"] + "/after_measurement" for e in entries)
    require(all(g["decision"] == "PASS" and not g["foreign_pids"] and not g["query_errors"] for g in gpu)
        and required_stages <= {g["stage"] for g in gpu}, "GPU boundary failure")
    protocol = read(stage / "protocol.json"); environment = read(root / "environment.json"); sources = {}
    for name, expected in protocol["source_sha256"].items():
        source = stage / name
        if not source.is_file(): source = stage / "source" / name
        actual = hashlib.sha256(source.read_bytes()).hexdigest(); require(actual == expected, "returned/frozen source mismatch: " + name)
        sources[name] = actual
    for name, actual in environment["sources"].items():
        require(name in sources and actual == sources[name], "runtime source mismatch: " + name)
    hardware = lines(stage / "hardware.jsonl"); own = {g["caller_pid"] for g in gpu}
    foreign = sorted({p["pid"] for h in hardware for p in h["processes"]} - own)
    require(hardware and len(own) == 1 and not foreign and not any(h.get("error") for h in hardware), "hardware sampling failure/foreign process")
    names = list(cells); comparisons = []
    fields = ("ttft_s", "tpot_s", "max_itl_s", "completion_latency_s", "submission_lag_s", "native_queue_until_first_schedule_s", "first_schedule_to_first_token_s")
    for i, j, kind in ((0, 1, "block0_static16_minus_static32"), (3, 2, "block1_static16_minus_static32"), (0, 3, "static32_repeat"), (1, 2, "static16_repeat")):
        a, b = names[i], names[j]; x, y = cells[a], cells[b]
        comparisons.append(dict(a=a, b=b, kind=kind, delta_b_minus_a={k: y["metrics"][k] - v for k, v in x["metrics"].items()},
            change_b_over_a_pct={k: 100 * (y["metrics"][k] / v - 1) if v else None for k, v in x["metrics"].items()},
            per_request_delta_b_minus_a={rid: {k: y["requests"][rid][k] - row[k] for k in fields} for rid, row in x["requests"].items()},
            observed_equality={k: normalized[a][k] == normalized[b][k] for k in normalized[a]},
            output_equal_by_request={rid: normalized[a]["outputs"][rid] == normalized[b]["outputs"][rid] for rid in ids}))
    return dict(status="MEASUREMENT_ONLY", integrity="PASS", cells=cells, comparisons=comparisons, total_request_executions=16,
        total_generated_tokens=416, total_scheduled_rows=1552, prepared_sha256=PREPARED_SHA256, memory_observer=observer,
        model_shape=shape, expert_cpu_master_bytes=sum(r["pinned_bytes"] for r in layers.values()), expert_gpu_scratch_bytes=sum(r["scratch_bytes"] for r in layers.values()),
        source_sha256=sources, environment=environment, engine_args=engine, gpu_checks=gpu,
        hardware=dict(sample_count=len(hardware), observed_worker_pids=sorted(own), foreign_sampled_pids=foreign,
            temperature_range_c=[min(h["temperature_c"] for h in hardware), max(h["temperature_c"] for h in hardware)], power_range_mw=[min(h["power_mw"] for h in hardware), max(h["power_mw"] for h in hardware)]),
        cycle_complete=read(root / "cycle_complete.json"), launch=read(stage / "launch.json"), protocol=protocol,
        scope="Four frozen requests per cell, one engine ABBA, both ordinary static thresholds at identical budgets. Common warmups use the same four prompts with two outputs, followed by drain/reset; this is not a quality holdout. Every repeat and policy-specific route/output difference is retained. No noise bound, significance, quality, SLO, Oracle or method GO.",
        timing_scope="Full capture wall equals exclusive engine-call phase wall plus outside gap, including clock idle and synchronous submission delay. TTFT equals submission lag plus native queue to first schedule plus first-schedule-to-first-token. CPU time, host apply, CUDA spans and GC overlap are not added together or subtracted. Shared engine/process costs are reported separately, not allocated to arms.",
        boundary_scope="Matching empty pager/allocation metadata is not KV tensor bitwise equivalence. Weight-copy bytes are actual tensor payload, not measured PCIe wire traffic. Reference qualification is disabled in these captures. GPU checks and finite hardware samples only cover recorded boundaries. Static thresholds take effect before each native schedule; drained resets reestablish shared-engine state between cells.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path); parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--launch-dir", type=Path, help="Launch evidence directory; defaults to adjacent or one-level-up launch/")
    args = parser.parse_args(); result = analyze(args.input_dir, stage=args.launch_dir)
    with args.out.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False); stream.write("\n")
