"""Read-only Qwen loading, request and numerical qualification; no performance summary."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def lines(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def require(ok, message):
    if not ok:
        raise ValueError(message)



def check_block_rounded_kv(engine, resources, measured, model_config):
    """Qwen BF16 KV tensor payload: floor requested bytes to whole 48-layer blocks."""
    shape = tuple(model_config[k] for k in ("num_hidden_layers", "num_key_value_heads", "head_dim"))
    require(shape == (48, 4, 128) and model_config["torch_dtype"] == engine["dtype"] == "bfloat16", "Qwen BF16 KV geometry")
    layers, heads, head_dim = shape
    block_size, requested = resources["block_size"], engine["kv_cache_memory_bytes"]
    require(block_size == 16 and resources["kv_cache_memory_bytes"] == requested == 536870912, "frozen KV budget/block size")
    layer_page_bytes = 2 * block_size * heads * head_dim * 2
    model_page_bytes = layers * layer_page_bytes
    expected_blocks, remainder = divmod(requested, model_page_bytes)
    expected_bytes = expected_blocks * model_page_bytes
    require(resources["num_gpu_blocks"] == resources["pool_num_gpu_blocks"] == measured["pool_num_gpu_blocks"] == expected_blocks, "KV block count differs from exact budget floor")
    require(measured["actual_unique_kv_storage_bytes"] == expected_bytes and measured["kv_storage_count"] == layers, "KV storage differs from exact block-rounded capacity")
    # allocation_metadata appends runner.kv_caches after expert tensors; no KV alias dedup ambiguity.
    allocations = measured["allocations"][-layers:]
    require(len(allocations) == layers and len({(r["device"], r["storage_pointer"]) for r in allocations}) == layers, "48 distinct KV allocations")
    require(all(r["shape"] == [expected_blocks, heads, block_size, 2 * head_dim]
                and r["storage_bytes"] == expected_blocks * layer_page_bytes
                and r["device"].startswith("cuda:") for r in allocations), "KV allocation shape/storage bytes")
    require(sum(r["storage_bytes"] for r in allocations) == expected_bytes, "KV unique allocation sum")
    # The native pool reserves one null block; it remains included in physical allocation.
    require(resources["free_blocks_after_init"] == measured["free_blocks"] == expected_blocks - 1, "drained KV pool/null-block accounting")
    return dict(requested_bytes=requested, block_size_tokens=block_size, layers=layers, kv_heads=heads,
                head_dim=head_dim, dtype_bytes=2, layer_block_bytes=layer_page_bytes,
                model_block_bytes=model_page_bytes, allocated_blocks=expected_blocks,
                allocated_bytes=expected_bytes, unallocated_remainder_bytes=remainder,
                reserved_null_blocks=1, free_blocks_at_measurement=expected_blocks - 1)

def digest(value):
    return hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()


def account(record, resident, layer):
    """Reuse actual expert residency accounting from analyze_admission_width.account."""
    groups = record["groups"]
    needed = [e for g in groups for e in g["required_experts"]]
    require(record["grouping_axis"] == "expert" and record["execution"] == "expert", "expert execution")
    require(set(record["entry_resident_experts"]) == resident, "expert entry cache mismatch")
    require(Counter(needed) == Counter(set(record["active_experts"])), "expert partition coverage")
    require(set(e for row in record["row_topk_experts"] for e in row) == set(needed), "route/active mismatch")
    require(len(record["row_topk_experts"]) == record["rows"] and all(len(r) == 8 and len(set(r)) == 8 for r in record["row_topk_experts"]), "top8 row identity")
    require(record["retention"]["mode"] == "none", "unexpected retention")
    initial_misses = set(needed) - resident
    loaded_all = []
    for group in groups:
        needed_group = set(group["required_experts"])
        ensure = set(group.get("ensure_experts", needed_group))
        loaded, evicted = group["loaded_experts"], set(group["evicted_experts"])
        require(ensure == needed_group and len(ensure) <= 48, "group cap/demand")
        require(group["unique_experts"] == len(needed_group) and all(0 <= e < 128 for e in ensure), "expert range")
        require(Counter(loaded) == Counter(ensure - resident), "load identity/count")
        require(evicted <= resident - ensure and len(evicted) == group["evict"], "eviction identity/count")
        require(len(loaded) == group["miss"] and len(loaded) * 9437184 == group["weight_copy_bytes"], "expert copy bytes")
        resident.difference_update(evicted); resident.update(loaded); loaded_all.extend(loaded)
        require(ensure <= resident and len(resident) <= 48, "resident cap")
        require((group["start"], group["stop"]) == (0, record["rows"]), "expert full-row partial coverage")
        require(group["load_cuda_span_ms"] is not None and group["load_cuda_span_ms"] >= 0, "unresolved CUDA event")
    require(Counter(loaded_all) == Counter(initial_misses), "expert misses loaded exactly once")
    require(record["status"] == "complete" and not record["measurement"] and record["validation_run"], "qualification flags")
    require(record["group_count"] == len(groups), "group count")
    require(all(record[k] == sum(g[k] for g in groups) for k in ("miss", "evict", "weight_copy_bytes")), "pager call totals")
    require(layer["pinned_bytes"] == 128 * 9437184, "expert tensor shape bytes")


def check_loader(receipt, manifest_path, index_path):
    manifest, index = read(manifest_path), read(index_path)
    require(hashlib.sha256(Path(index_path).read_bytes()).hexdigest() == manifest["index_sha256"], "frozen index SHA")
    events = lines(receipt)
    require(events[0]["event"] == "START" and events[-1]["event"] == "COMPLETE", "loader did not complete")
    require(events[0]["revision"] == manifest["revision"] and events[0]["index_sha256"] == manifest["index_sha256"], "loader revision/index")
    require(events[0]["manifest_sha256"] == hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest(), "loader manifest SHA")
    cursor = 1
    for name, spec in sorted(manifest["shards"].items()):
        group = events[cursor:cursor + 3]; cursor += 3
        require([e["event"] for e in group] == ["DOWNLOAD", "DOWNLOADED", "CONSUMED_REMOVED"], "nonserial/incomplete shard")
        require(all(e["shard"] == name for e in group), "shard order/identity")
        require(group[0]["expected_bytes"] == spec["bytes"], "download byte contract")
        require(all(e["bytes"] == spec["bytes"] and e["sha256"] == spec["sha256"] for e in group[1:]), "shard size/hash")
        require(group[2]["tensors"] == sum(v == name for v in index["weight_map"].values()), "source tensor coverage")
    done = events[-1]
    require(cursor == len(events) - 1 and done["shards"] == len(manifest["shards"]) and done["source_tensors"] == len(index["weight_map"]), "source EOF/key coverage")
    target = read(str(receipt) + ".targets.json")
    raw, expected, aliases = target["raw_returned_names"], target["expected_named_parameters"], target["aliases"]
    require(target["status"] == "PASS" and not target["missing"], "target coverage failed")
    require(len(raw) == len(set(raw)) and digest(sorted(raw)) == target["raw_names_sha256"], "raw target names/hash")
    require(all(name.rsplit(".", 1)[-1] in ("w13_weight", "w2_weight") and mapped == name.rpartition(".")[0] + ("." if "." in name else "") + "routed_experts." + name.rsplit(".", 1)[-1] for name, mapped in aliases.items()), "unexpected target alias")
    normalized = {aliases.get(name, name) for name in raw}
    require(normalized == set(target["normalized_returned_names"]) and set(expected) <= normalized, "normalized target coverage")
    require(done["target_names"] == len(raw), "target receipt count")
    return dict(status="PASS", source_tensors=done["source_tensors"], target_names=len(raw), expected_parameters=len(expected), shards=done["shards"])


def analyze(root, receipt, manifest, index, source_dir=None):
    root = Path(root); source_dir = Path(source_dir or Path(__file__).parent / "source")
    status, config, engine = (read(root / n) for n in ("status.json", "config.json", "engine_args.json"))
    require(status["status"] == "COMPLETE", "runner not complete")
    require(config["qwen_qualification"] and config["verify_kernel"] and config["trace_retention"] == "episode", "qualification config")
    require((config["requests"], config["output_tokens"], config["expert_cap"], config["prefill_limit"], config["arrival_interval"]) == (4, 8, 48, 32, .05), "request/action protocol")
    require((engine["max_num_seqs"], engine["max_num_batched_tokens"], engine["kv_cache_memory_bytes"]) == (4, 64, 536870912), "engine resource budget")
    require(engine["load_format"] == "qwen_bf16_serial_v026" and engine["dtype"] == "bfloat16" and not engine["async_scheduling"] and not engine["enable_prefix_caching"], "runtime protocol")
    environment = read(root / "environment.json")
    for name, sha in environment["sources"].items():
        require(hashlib.sha256((source_dir / name).read_bytes()).hexdigest() == sha, "source mismatch: " + name)
    entries = read(root / "episodes.json")
    require(len(entries) == 1 and entries[0]["phase"] == "repeat_0_qwen_qualification/measurement", "one measurement episode")
    phase = entries[0]["phase"]; path = root / phase.split("/")[0]
    raw, workload, pager = read(path / "raw.json"), read(root / "workload.json"), read(root / "pager_summary.json")
    require(workload.get("tokenizer_identity") == {"repository": "Qwen/Qwen3-30B-A3B", "revision": "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"}, "frozen Qwen tokenizer identity")
    require(list(map(len, workload["actual_prompt_token_ids"])) == [64] * 4, "four frozen 64-token prefixes")
    require(raw["status"] == "COMPLETE" and raw["cpu_diagnostics"] and raw["event_arrival"] is None and not raw["event_actions"], "static native capture")
    rows = {r["request_id"]: r for r in raw["requests"]}
    require(len(rows) == 4 and set(rows) == {r["request_id"] for r in workload["source_requests"]}, "four request IDs")
    outputs = {}
    for i, (source, tokens) in enumerate(zip(workload["source_requests"], workload["actual_prompt_token_ids"])):
        row = rows[source["request_id"]]; times = row["token_times_s"]
        require(row["document_id"] == source["document_id"] and row["prompt_token_ids_sha256"] == digest(tokens), "source/prompt join")
        require(row["status"] == "completed" and len(row["output_token_ids"]) == len(times) == row["max_output_tokens"] == 8, "complete output receipts")
        require(row["prompt_tokens"] == len(tokens) and row["arrival_s"] == i * .05, "prompt/clock arrival")
        require(all(math.isfinite(t) for t in times) and times == sorted(times) and times[0] >= row["admission_s"] >= row["arrival_s"], "receipt time alignment")
        require(row["completion_s"] == times[-1] and abs(row["submission_lag_s"] - (row["admission_s"] - row["arrival_s"])) < 1e-9, "completion/submission lag")
        outputs[row["request_id"]] = digest(row["output_token_ids"])
    warm = read(path / "warmup.json")
    require(warm["status"] == "COMPLETE" and len(warm["requests"]) == 1 and len(warm["requests"][0]["output_token_ids"]) == 2, "separate two-output warmup")
    layers = {r["layer_name"]: r for r in pager["layers"]}
    require(len(layers) == 48 and pager["cap"] == 48 and all(r["cap"] == 48 and r["num_experts"] == 128 and r["scratch_bytes"] == 48 * 9437184 for r in layers.values()), "48-layer/128-expert realization")
    res = read(path / "measurement_resources.json")
    model_path = Path(__file__).parent / "model_metadata/config.json"
    require(hashlib.sha256(model_path.read_bytes()).hexdigest() == read(manifest)["config_sha256"], "frozen model KV configuration")
    kv_accounting = check_block_rounded_kv(engine, read(root / "resources.json"), res, read(model_path))
    require(res["expert_cap"] == 48 and res["scheduler_requests"] == 0, "realized cap/empty start")
    trace = lines(root / "pager/calls.jsonl")
    require(len(trace) == pager["all_calls"] and len({r["call_id"] for r in trace}) == len(trace), "trace coverage")
    records = [r for r in trace if r["context"]["phase"] == phase]
    by_call = {r["call_id"]: r for r in records}
    resident = {name: set(map(int, v["expert_to_slot"])) for name, v in res["cache"].items()}
    positions = {rid: [] for rid in rows}; joined = 0
    for step in raw["scheduler_steps"]:
        require(step["actual_admission_cap"] == 4 and step["actual_active"] <= 4 and step["total_scheduled_tokens"] <= 64, "native schedule budget")
        require(not step["n_preempted"] and not step["recomputed_tokens"] and not step["kv_adjusted_request_ids"], "preemption/recompute")
        decision = step["phase_prefill"]
        require(decision["mode"] == "static32" and decision["chosen_threshold"] == 32 and decision["status"] == "applied", "static32 action")
        ready = {rid for rid, r in rows.items() if r["token_times_s"][0] <= step["start_s"] < r["completion_s"]}
        require(set(step["existing_decode_request_ids"]) == ready and step["existing_decode_all_scheduled"], "all-ready decode identity")
        scheduled = {r["request_id"]: r for r in step["scheduled"]}
        require(all(rid in scheduled and scheduled[rid]["decode_tokens"] == scheduled[rid]["scheduled_tokens"] == 1 for rid in ready), "ready decode did not advance exactly once")
        expected = Counter()
        for rid, r in scheduled.items():
            require(r["computed_adjustment"] == 0 and r["computed_after"] - r["computed_before"] == r["scheduled_tokens"], "computed progression")
            ps = list(range(r["scheduled_start_computed"], r["computed_after"]))
            positions[rid].extend(ps); expected.update((rid, p) for p in ps)
        rr = [r for r in records if r["context"]["step_id"] == step["step"]]
        require(Counter(r["layer_name"] for r in rr) == (Counter(layers.keys()) if expected else Counter()), "48-layer/step coverage")
        for r in rr:
            physical = [(raw["internal_to_source"][x["internal_request_id"]], x["computed_position"]) for x in r["context"]["rows"]]
            require(r["context"]["row_request_order_verified"] and Counter(physical) == expected and r["rows"] == sum(expected.values()), "physical position/request/layer join")
            account(r, resident[r["layer_name"]], layers[r["layer_name"]]); joined += 1
    require(joined == len(records), "unjoined expert calls")
    require(all(ps == list(range(rows[rid]["prompt_tokens"] + 7)) for rid, ps in positions.items()), "complete contiguous computed positions")
    validation = read(root / "qualification_validation.json"); references = validation["layer_validation"]
    require(len(references) == validation["validation_calls"] == validation["unique_validated_layers"] == 48 and {r["layer_name"] for r in references} == set(layers), "reference layer coverage")
    for ref in references:
        call = by_call[ref["call_id"]]
        require(ref["layer_name"] == call["layer_name"] and ref["context"] == call["context"] and ref["rows"] == call["rows"], "same-precall reference join")
        require(ref["actual_expert_groups"] == [g["required_experts"] for g in call["groups"]] and "all pre-call rows" in ref["scope"], "full-row reference scope")
        require(ref["allfinite"] and math.isfinite(ref["maxabs"]) and math.isfinite(ref["relative_l2"]), "nonfinite reference")
        require(ref["rtol"] == ref["atol"] == .01 and isinstance(ref["allclose"], bool), "original diagnostic tolerance")
    require(validation["reference_full_weight_copy_bytes"] == sum(layers[r["layer_name"]]["pinned_bytes"] for r in references) == 54 * 2**30, "separate reference copy accounting")
    require(all(pager["all"][k] == sum(r[k] for r in trace) for k in ("miss", "evict", "weight_copy_bytes", "group_count")), "all-trace pager accounting")
    maps = read(path / "map_optimization.json")
    require(maps["validation"]["status"] == "PASS" and maps["validation"]["layers"] == 48, "map validation")
    require(maps["statistics"]["totals"]["misses"] == sum(r["miss"] for r in records), "map/pager copy count")
    memory = read(root / "memory_observer.json")
    require(memory["values_equal"] and memory["warmup_mode"] == memory["measurement_mode"] == "nested", "common observer validation")
    gpu = lines(root / "gpu_checks.jsonl")
    require(gpu and all(g["decision"] == "PASS" and not g["foreign_pids"] and not g["query_errors"] for g in gpu), "GPU isolation boundary")
    loader = check_loader(receipt, manifest, index)
    return dict(status="PASS", evidence="READBACK_QUALIFICATION_ONLY_NO_PERFORMANCE_CLAIM", requests_completed=4,
        measurement_output_tokens=32, warmup_output_tokens=2, computed_positions=sum(map(len, positions.values())),
        kv_accounting=kv_accounting,
        expert_layer_calls=len(records), layers=48, all_ready_decode_advanced=True, output_hashes=outputs,
        numerical=dict(reference_calls=48, allfinite=True, allclose_diagnostic=all(r["allclose"] for r in references),
            maxabs=max(r["maxabs"] for r in references), max_relative_l2=max(r["relative_l2"] for r in references)),
        copy_accounting=dict(pager_measurement_phase_payload_bytes=sum(r["weight_copy_bytes"] for r in records),
            reference_payload_bytes_derived=validation["reference_full_weight_copy_bytes"],
            scope="Separate tensor payload counters; loader/postload H2D and wire overhead not included; no bandwidth or latency conclusion"), loader=loader)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("--source-dir", type=Path)
    for name in ("receipt", "manifest", "index", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    args = p.parse_args()
    result = analyze(args.root, args.receipt, args.manifest, args.index, source_dir=args.source_dir)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
