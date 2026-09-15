"""Read immediate/admit2/release64 native admission baselines and full request costs."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[5] / "experiments/admission_capacity"))
from analyze_native_pager_transfer import requests


def read(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def account(record, resident, layer, cap):
    """Validate actual residency transitions, plus the appropriate partition invariant."""
    groups = record["groups"]; size = layer["pinned_bytes"] // layer["num_experts"]
    expert = record.get("grouping_axis") == "expert"
    needed = [e for g in groups for e in g["required_experts"]]
    if expert:
        require(set(record["entry_resident_experts"]) == resident, "expert entry cache mismatch")
        require(Counter(needed) == Counter(set(record["active_experts"])), "expert partition overlap/coverage")
        require(set(e for row in record["row_topk_experts"] for e in row) == set(needed), "route/active mismatch")
        require(len(record["row_topk_experts"]) == record["rows"], "expert route row count")
        require(record["retention"]["mode"] == "none", "unexpected retention intervention")
    else:
        require([p for g in groups for p in range(g["start"], g["stop"])] == list(range(record["rows"])), "token partition coverage")
    loads = []; entry_missing = set(needed) - resident
    for group in groups:
        needed_group = set(group["required_experts"]); ensure = set(group.get("ensure_experts", needed_group))
        loaded = group["loaded_experts"]; evicted = set(group["evicted_experts"])
        require(ensure == needed_group and len(ensure) <= cap, "unexpected ensure demand or cap")
        require(len(needed_group) == group["unique_experts"] and all(0 <= e < layer["num_experts"] for e in ensure), "expert identity")
        require(Counter(loaded) == Counter(ensure - resident), "load identity/count mismatch")
        require(evicted <= resident - ensure and len(evicted) == group["evict"], "eviction identity/count mismatch")
        require(len(loaded) == group["miss"] and len(loaded) * size == group["weight_copy_bytes"], "group bytes mismatch")
        resident.difference_update(evicted); resident.update(loaded); loads.extend(loaded)
        require(ensure <= resident and len(resident) <= cap, "residency/cap violation")
        require(group["load_cuda_span_ms"] is not None and group["load_cuda_span_ms"] >= 0, "unresolved CUDA event")
        if expert:
            require((group["start"], group["stop"]) == (0, record["rows"]), "expert partial row coverage")
    if expert:
        require(Counter(loads) == Counter(entry_missing), "expert entry misses not loaded exactly once")
    require(record["measurement"] and record["status"] == "complete", "unmeasured/failed call")
    require(record["group_count"] == len(groups), "group count mismatch")
    require(all(record[k] == sum(g[k] for g in groups) for k in ("miss", "evict", "weight_copy_bytes")), "call totals mismatch")
    return dict(weight_copy_bytes=len(loads) * size, groups=len(groups), misses=len(loads),
        evictions=sum(g["evict"] for g in groups), repeated_load_bytes=(len(loads) - len(set(loads))) * size,
        load_cuda_span_ms=sum(g["load_cuda_span_ms"] for g in groups), host_apply_ms=record["host_apply_ms"],
        route_to_host_ms=record["route_to_host_ms"])


def validate_admission(raw, entry, boundary):
    action = raw["event_actions"][0]; arm = entry["arm"]; new = action["request_id"]
    old = set(action["before"]["old_output_tokens"]); rows = {r["request_id"]: r for r in raw["requests"]}
    new_row = rows[new]; old_done = max(rows[rid]["completion_s"] for rid in old)
    cap = 3 if arm == "immediate32" else 2
    require(action["admission_mode"] == arm and action["admission_cap_before"] == 3 and action["admission_cap_after"] == cap, "admission action mismatch")
    require(action["snapshot_end_s"] <= action["admission_applied_s"] <= action["before_add_end_s"] <= new_row["engine_add_return_s"], "admission action timing")
    require(new_row["arrival_s"] == action["release_s"] == max(rows[rid]["token_times_s"][3] for rid in old), "arrival denominator changed")
    require(new_row["arrival_s"] <= new_row["admission_s"] <= new_row["engine_add_return_s"] <= raw["engine_calls"][action["engine_call"]]["start_s"], "third request was held outside native queue")
    waiting_steps = []; new_steps = []
    for step in raw["scheduler_steps"]:
        require(step["actual_admission_cap"] == (3 if step["step"] < boundary else cap), "native admission cap mismatch")
        scheduled = {r["request_id"]: r for r in step["scheduled"]}; existing = step["existing_decode_request_ids"]
        require(step["existing_decode_all_scheduled"] and all(rid in scheduled and scheduled[rid]["decode_tokens"] == 1 for rid in existing), "old request skipped decode")
        if new in scheduled:
            new_steps.append(step)
        if step["step"] >= boundary and arm != "immediate32" and step["start_s"] < old_done:
            wait = step["admission_wait"]
            require(wait["internal_request_id"] == new_row["internal_request_id"] and wait["status"] == "WAITING"
                and wait["computed"] == wait["preemptions"] == wait["scheduled_tokens"] == 0 and not any(wait["kv_block_ids"])
                and wait["never_started_wait_verified"] and not wait["old_completed"] and new not in scheduled, "new request advanced before old completion")
            waiting_steps.append(step["step"])
    first = new_steps[0]
    prefill = [r["prefill_tokens"] for step in new_steps for r in step["scheduled"] if r["request_id"] == new and r["prefill_tokens"]]
    require(prefill == ([64, 64] if arm == "admit2_release64" else [32] * 4), "actual prefill workload or budget changed")
    if arm != "immediate32":
        require(waiting_steps and first["start_s"] >= old_done, "delayed admission did not wait for old completion")
    releases = raw["prefill_release_actions"]
    require(len(releases) == int(arm == "admit2_release64"), "unexpected/missing release action")
    if releases:
        release = releases[0]; call = raw["engine_calls"][release["engine_call"]]
        require((release["threshold_before"], release["threshold_after"], release["effective_token_budget"]) == (32, 0, 64), "release budget mismatch")
        require((release["computed_tokens"], release["prompt_tokens"], release["remaining_prefill_tokens"]) == (0, 128, 128), "release used partially prefetched request")
        require(release["eligible_s"] == old_done <= release["start_s"] <= release["applied_s"] <= release["end_s"] <= call["start_s"], "release timing mismatch")
        require(call["scheduler_step_start"] == first["step"] and set(release["old_requests"]) == old
            and all(r["status"] == "completed" and r["output_tokens"] == 16 and not r["present_in_scheduler"] for r in release["old_requests"].values()), "release old-completion witness")
    restored = raw["admission_restore"]
    require(restored == dict(drained=True, cap_before=cap, cap_after=3, threshold_before=0 if releases else 32, threshold_after=32), "admission cap/threshold not restored on drained engine")
    return dict(status="PASS", never_started_wait_steps=waiting_steps, first_new_schedule_step=first["step"],
        arrival_to_first_schedule_s=first["start_s"] - new_row["arrival_s"],
        enqueue_to_first_schedule_s=first["start_s"] - new_row["admission_s"],
        first_schedule_to_first_token_s=new_row["token_times_s"][0] - first["start_s"],
        actual_new_prefill_tokens_per_call=prefill, restoration=restored,
        timing_scope="Arrival-to-first-schedule plus first-schedule-to-first-token equals TTFT. Queue delay overlaps old-request work and must not be added to phase wall.")


def summarize_phases(raw, records):
    action = raw["event_actions"][0]; new = action["request_id"]; old = set(action["before"]["old_output_tokens"])
    phases = {}; totals = []; assigned = []
    for call in raw["engine_calls"]:
        steps = raw["scheduler_steps"][call["scheduler_step_start"]:call["scheduler_step_stop"]]
        scheduled = [r for step in steps for r in step["scheduled"]]
        prefill = sum(r["prefill_tokens"] for r in scheduled if r["request_id"] == new)
        decode = sum(r["decode_tokens"] for r in scheduled if r["request_id"] == new)
        old_decode = sum(r["decode_tokens"] for r in scheduled if r["request_id"] in old)
        waiting = any(step.get("admission_wait", {}).get("never_started_wait_verified") for step in steps)
        label = ("prefix" if call["index"] < action["engine_call"] else "mixed_new_prefill" if prefill and old_decode
                 else "new_prefill" if prefill else "mixed_decode" if decode and old_decode else "new_decode" if decode
                 else "old_decode_while_new_waits" if waiting else "old_decode_only" if old_decode else "empty")
        rr = [r for r in records if call["scheduler_step_start"] <= r["context"]["step_id"] < call["scheduler_step_stop"]]
        assigned.extend(r["call_id"] for r in rr)
        row = dict(engine_calls=1, engine_wall_s=call["return_s"] - call["start_s"], weight_copy_bytes=sum(r["weight_copy_bytes"] for r in rr),
            groups=sum(len(r["groups"]) for r in rr), host_apply_ms=sum(r["host_apply_ms"] for r in rr),
            load_cuda_span_ms=sum(g["load_cuda_span_ms"] for r in rr for g in r["groups"]))
        phases.setdefault(label, []).append(row); totals.append(row)
    require(Counter(assigned) == Counter(r["call_id"] for r in records), "phase partition lost/doubled trace calls")
    aggregate = lambda rows: {k: sum(r[k] for r in rows) for k in totals[0]}
    return dict(stages={k: aggregate(v) for k, v in phases.items()}, total=aggregate(totals),
        outside_gap_s=raw["observation_end_s"] - sum(r["engine_wall_s"] for r in totals),
        scope="Exclusive engine-call phases; full wall equals their wall sum plus outside gap. Host apply and CUDA load span overlap and are not added together.")


def analyze(root):
    root = Path(root); entries = read(root / "episodes.json"); config = read(root / "config.json")
    pager = read(root / "pager_summary.json"); workload = read(root / "workload.json")
    layers = {r["layer_name"]: r for r in pager["layers"]}; engine = read(root / "engine_args.json")
    require(len(layers) == 16 and pager["cap"] == 16 and all(r["cap"] == 16 for r in layers.values()), "realized expert-layer capacity")
    arms = ["immediate32", "admit2_32", "admit2_release64"]
    require(len(entries) == 6 and all(e["execution"] == "expert" and e["chunk"] == 32 for e in entries), "incomplete/changed admission matrix")
    require([e["arm"] for e in entries] == arms + list(reversed(arms)), "counterbalanced arm order changed")
    require(all(e["block"] == i // 3 and e["order"] == i and e["order_in_block"] == i % 3
        and e["ordering"] == "declared_first_block_then_reverse" for i, e in enumerate(entries)), "order metadata changed")
    require((config["expert_cap"], config["kv_bytes"], config["token_budget"], config["injection_chunk"]) == (16, 536870912, 64, 16), "resource protocol changed")
    require(workload["source_indices"] == [4, 5, 10] and list(map(len, workload["actual_prompt_token_ids"])) == [32, 32, 128], "input protocol changed")
    require((engine["max_num_seqs"], engine["max_num_batched_tokens"], engine["kv_cache_memory_bytes"]) == (3, 64, 536870912), "realized engine budget")
    require(not config["verify_kernel"] and config["trace_retention"] == "episode", "validation/trace protocol")
    trace = [json.loads(line) for line in (root / "pager/calls.jsonl").open() if line.strip()]
    require(len(trace) == pager["all_calls"] and len({r["call_id"] for r in trace}) == len(trace), "trace coverage/duplicate call IDs")
    gpu = [json.loads(line) for line in (root / "gpu_checks.jsonl").open() if line.strip()]
    require(gpu and all(g["decision"] == "PASS" and not g["foreign_pids"] and not g["query_errors"] for g in gpu), "GPU boundary check failed")
    stages = {g["stage"] for g in gpu}
    require({"before_initialization", "after_run"} <= stages and all(e["phase"].split("/")[0] + "/before_reset" in stages and e["phase"] + "/after_measurement" in stages for e in entries), "missing initialization/repeat GPU checks")
    results = {}; normalized = {}; accounted_all = []
    for entry in entries:
        phase = entry["phase"]; name = phase.split("/")[0]; path = root / name; chunk = entry["chunk"]
        raw = read(path / "raw.json"); res = read(path / "measurement_resources.json"); memory = read(path / "cuda_memory.json")
        metrics, issues = requests(raw, workload); require(not issues, str(issues))
        require(raw["status"] == "COMPLETE" and len(raw["event_actions"]) == 1, "incomplete capture/event")
        rows = {r["request_id"]: r for r in raw["requests"]}; aliases = raw["internal_to_source"]
        action = raw["event_actions"][0]; before = action["before"]; old = list(before["old_output_tokens"]); new = action["request_id"]
        require(len(old) == 2 and new not in old and set(rows) == set(old + [new]), "request roles")
        require(all(r["status"] == "completed" and len(r["output_token_ids"]) == (8 if rid == new else 16) for rid, r in rows.items()), "request completion/token count")
        require(all(r["completion_s"] == r["token_times_s"][-1] for r in rows.values()), "completion/output receipt mismatch")
        require(all(len(ts) == 4 and ts == rows[rid]["output_token_ids"][:4] for rid, ts in before["old_output_tokens"].items()), "prefix output identity")
        calls = raw["engine_calls"]; steps = raw["scheduler_steps"]; first = calls[action["engine_call"]]; boundary = first["scheduler_step_start"]
        require(action["engine_call"] == 4 and action["execution_before"] == "token" and action["execution_after"] == entry["execution"], "execution switch identity")
        require(action["snapshot_end_s"] <= action["execution_applied_s"] <= action["before_add_end_s"] <= rows[new]["engine_add_return_s"] <= first["start_s"], "execution switch timing")
        require((action["threshold_before"], action["threshold_after"]) == (32, chunk) and action["inject"], "injection action")
        require(steps[boundary]["total_scheduled_tokens"] == (34 if entry["arm"] == "immediate32" else 2) and all(s["total_scheduled_tokens"] <= 64 and not s["n_preempted"] and not s["recomputed_tokens"] for s in steps), "schedule budget/preemption")
        require(all(r["prefill_tokens"] in (0, 64 if entry["arm"] == "admit2_release64" else 32) for s in steps for r in s["scheduled"] if r["request_id"] == new), "incoming chunk changed")
        require([i for c in calls for i in range(c["scheduler_step_start"], c["scheduler_step_stop"])] == list(range(len(steps))), "engine/step coverage")
        require(all(c["index"] == i and c["returned"] and c["start_s"] <= c["return_s"] for i, c in enumerate(calls)), "engine timing/status")
        logical = dict(before)
        # Runner hashed native integer expert IDs; JSON round trips stringify keys.
        logical["cache"] = {k: dict(v, expert_to_slot={int(e): slot for e, slot in v["expert_to_slot"].items()}) for k, v in before["cache"].items()}
        logical["requests"] = {aliases[rid]: dict({k: v for k, v in r.items() if k != "kv_block_ids"}, kv_block_counts=[len(g) for g in r["kv_block_ids"]]) for rid, r in before["requests"].items()}
        for key in ("running", "waiting"):
            logical[key] = [aliases[rid] for rid in before[key]]
        require(digest(logical) == action["prestate_sha256"], "prestate digest mismatch")
        require(res["expert_cap"] == 16 and res["actual_unique_kv_storage_bytes"] == 536870912 and res["measured_execution"] == entry["execution"], "realized resources")
        admission = validate_admission(raw, entry, boundary)
        records = [r for r in trace if r["context"]["phase"] == phase]
        phase_cost = summarize_phases(raw, records)
        resident = {k: set(map(int, v["expert_to_slot"])) for k, v in res["cache"].items()}; accounts = []; signature = []
        for step in steps:
            rr = [r for r in records if r["context"]["step_id"] == step["step"]]
            require(Counter(r["layer_name"] for r in rr) == Counter(layers.keys()), "layer/step coverage")
            expected = Counter((r["request_id"], p) for r in step["scheduled"] for p in range(r["scheduled_start_computed"], r["computed_after"]))
            for r in rr:
                physical = [(aliases[x["internal_request_id"]], x["computed_position"]) for x in r["context"]["rows"]]
                require(Counter(physical) == expected and r["rows"] == step["total_scheduled_tokens"], "physical row/step identity")
                require(r.get("execution", "token") == ("token" if step["step"] < boundary else entry["execution"]), "observed execution differs from action")
                accounts.append(account(r, resident[r["layer_name"]], layers[r["layer_name"]], 16))
                signature.append(dict(step=step["step"], layer=r["layer_name"], rows=physical,
                    groups=[{k: g[k] for k in ("start", "stop", "required_experts", "loaded_experts", "evicted_experts")} for g in r["groups"]]))
        require(len(accounts) == len(records), "unjoined trace calls")
        totals = {k: sum(a[k] for a in accounts) for k in accounts[0]}; accounted_all.extend(accounts)
        first_records = [r for r in records if first["scheduler_step_start"] <= r["context"]["step_id"] < first["scheduler_step_stop"]]
        first_totals = dict(first_post_arrival_weight_copy_bytes=sum(r["weight_copy_bytes"] for r in first_records),
            first_post_arrival_groups=sum(len(r["groups"]) for r in first_records),
            first_post_arrival_loads=sum(g["miss"] for r in first_records for g in r["groups"]))
        outputs = {rid: r["output_token_ids"] for rid, r in rows.items()}
        normalized[name] = dict(prestate=logical, initial_cache=res["cache"], allocations=res["allocations"], outputs=outputs,
            physical_kv_ids={aliases[k]: r["kv_block_ids"] for k, r in before["requests"].items()},
            prefix_signature=[r for r in signature if r["step"] < boundary], full_signature=signature)
        map_path = path / "map_optimization.json"; map_report = read(map_path) if map_path.exists() else None
        if map_report is not None:
            require(map_report["validation"]["status"] == "PASS" and map_report["validation"]["layers"] == len(layers), "map validation failed")
            stats = map_report["statistics"]
            require(stats["mode"] == "batched" and stats["totals"]["scalar_device_assignments"] == 0, "map mode mismatch")
            require(stats["totals"]["misses"] == totals["misses"] and stats["totals"]["ensure_calls"] == totals["groups"], "map/trace counter mismatch")
        old_done = max(rows[rid]["completion_s"] for rid in old)
        results[name] = dict(execution=entry["execution"], arm=entry["arm"], block=entry["block"], chunk=chunk, requests=metrics, prestate_sha256=digest(logical), admission=admission, phase_cost=phase_cost,
            metrics=dict(whole_wall_s=raw["observation_end_s"], new_ttft_s=metrics[new]["ttft_s"], old_completion_s=old_done,
                new_completion_latency_s=metrics[new]["completion_latency_s"], new_completion_s=rows[new]["completion_s"],
                all_request_max_itl_s=max(r["max_itl_s"] for r in metrics.values()), new_max_itl_s=metrics[new]["max_itl_s"],
                new_tpot_s=metrics[new]["tpot_s"], new_arrival_to_first_schedule_s=admission["arrival_to_first_schedule_s"], **first_totals,
                old_max_itl_s=max(metrics[rid]["max_itl_s"] for rid in old), first_post_arrival_call_s=first["return_s"] - first["start_s"],
                engine_wall_s=sum(c["return_s"] - c["start_s"] for c in calls), **totals),
            request_count=len(rows), generated_tokens=sum(map(len, outputs.values())), engine_calls=len(calls), layer_calls=len(records),
            cuda_memory=memory, map_optimization=map_report, scratch_bytes=sum(r["scratch_bytes"] for r in layers.values()),
            actual_unique_kv_storage_bytes=res["actual_unique_kv_storage_bytes"],
            expert_temporary_memory=[dict(call_id=r["call_id"], **r["temporary_memory"]) for r in records if "temporary_memory" in r],
            token_temporary_memory_scope="No per-call token workspace instrumentation; whole-episode allocator peaks are measured for both arms.")
    names = list(results); reference = normalized[names[0]]
    require(all(n["prestate"] == reference["prestate"] and n["prefix_signature"] == reference["prefix_signature"] for n in normalized.values()), "common prefix/prestate differs")
    require(all(n["allocations"] == reference["allocations"] and n["initial_cache"] == reference["initial_cache"] for n in normalized.values()), "initial allocation/cache differs")
    require(len(accounted_all) == pager["measurement_calls"] and sum(a["weight_copy_bytes"] for a in accounted_all) == pager["measurement"]["weight_copy_bytes"], "measurement summary coverage/bytes")
    comparisons = []
    lookup = {(r["block"], r["arm"]): name for name, r in results.items()}
    pairs = [(lookup[block, a], lookup[block, b], f"block{block}_{b}_minus_{a}") for block in range(2)
             for a, b in (("immediate32", "admit2_32"), ("immediate32", "admit2_release64"), ("admit2_32", "admit2_release64"))]
    pairs.extend((lookup[0, arm], lookup[1, arm], f"{arm}_repeat") for arm in arms)
    for a, b, kind in pairs:
        x, y = results[a]["metrics"], results[b]["metrics"]
        request_deltas = {rid: {k: results[b]["requests"][rid][k] - row[k]
            for k in ("ttft_s", "tpot_s", "max_itl_s", "completion_latency_s")} for rid, row in results[a]["requests"].items()}
        comparisons.append(dict(a=a, b=b, kind=kind, request_timing_delta_b_minus_a=request_deltas, delta_b_minus_a={k: y[k] - x[k] for k in x},
            change_b_over_a_pct={k: 100 * (y[k] / x[k] - 1) if x[k] else None for k in x},
            equality={k: normalized[a][k] == normalized[b][k] for k in normalized[a]},
            output_equal_by_request={rid: normalized[a]["outputs"][rid] == normalized[b]["outputs"][rid] for rid in normalized[a]["outputs"]}))
    launch_path = next((p for p in (root / "launch.json", root.parent / "launch.json", root.parent / "launch/launch.json") if p.exists()), None)
    return dict(status="MEASUREMENT_ONLY", integrity="PASS", cells=results, comparisons=comparisons,
        total_request_executions=sum(r["request_count"] for r in results.values()), total_generated_tokens=sum(r["generated_tokens"] for r in results.values()),
        gpu_checks=gpu, cycle_complete=read(root / "cycle_complete.json"), launch=read(launch_path) if launch_path else None,
        environment=read(root / "environment.json"), engine_args=engine,
        scope="One source triplet, six same-engine episodes in I,D,R,R,D,I order. All three requests truly arrive and enter the native queue; delayed admission is charged in new-request TTFT. Counterbalanced observations are descriptive, not independent repetitions, confidence bounds, statistical significance, or a method GO. No cross-campaign absolute-time comparison.",
        state_scope="Source-normalized request/token/KV-count/pager metadata and prefix trace checked; physical KV IDs compared separately; KV tensor bytes untested.",
        cost_scope="Whole capture includes online observation/action overhead; cycle includes reset/warmup/flush/shutdown. Cycle and process wall are not allocated across arms. Host apply includes route transfer and overlaps CUDA load spans; do not add these timings. Weight payload excludes map DMA and allocator/sorting internals.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True); parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(); result = analyze(args.input_dir)
    with args.out.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False); stream.write("\n")
