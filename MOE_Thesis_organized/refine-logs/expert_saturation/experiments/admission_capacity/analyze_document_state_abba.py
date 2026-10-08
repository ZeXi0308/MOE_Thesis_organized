"""Safe analysis for fixed-static16 old/new document-state ABBA episodes."""
import argparse, hashlib, json
from pathlib import Path
from analyze_prefill_midpoint import prestate
from analyze_static_calibration import metric_row, static_decisions
from analyze_wisp_injection import hardware_summary, resources, summarize
sha = lambda data: hashlib.sha256(data).hexdigest()
jsha = lambda value: sha(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())
EXPECTED = [("r0-old16", 0, "old"), ("r0-new16", 0, "new"),
            ("r1-new16", 1, "new"), ("r1-old16", 1, "old")]
def preaction_calls(raw):
    by_index, rows = {row["index"]: row for row in raw["steps"]}, []
    for index in (1, 2, 3):
        step = by_index.get(index, {})
        rows.append(dict(engine_call=index, returned=step.get("returned"),
            wall_s=step.get("return_s", 0) - step["start_s"] if "return_s" in step else None,
            scheduled_tokens=step.get("total_scheduled_tokens")))
    old = set(raw["action"]["old_output_tokens"])
    valid = raw["action"]["engine_call"] == 4 and all(row["returned"] is True and row["wall_s"] is not None
        and not by_index[row["engine_call"]].get("error") and not by_index[row["engine_call"]].get("preempted_request_ids")
        and sum(s["request_id"] in old and s["tokens"] == s["decode_tokens"] == 1 and s["prefill_tokens"] == 0
            for s in by_index[row["engine_call"]]["scheduled"]) == 2 for row in rows)
    return dict(valid=valid, calls=rows)
def map_counters(raw):
    row, keys = raw.get("first_action_map_counters", {}), (
        "ensure_calls", "misses", "evictions", "full_map_copy_calls")
    before, after, stored = row.get("before", {}), row.get("after", {}), row.get("delta", {})
    delta = {key: after.get(key, 0) - before.get(key, 0) for key in keys}
    read = row.get("counter_read_s", {})
    valid = (row.get("engine_call") == raw["action"]["engine_call"] and row.get("layer_count") == 16
        and all(set(part) == set(keys) for part in (before, after, stored)) and stored == delta
        and all(value >= 0 for value in delta.values()) and delta["ensure_calls"] > 0
        and delta["full_map_copy_calls"] <= delta["ensure_calls"]
        and row.get("expert_tensor_payload_bytes") == delta["misses"] * 12 * 1024**2
        and set(read) == {"before", "after", "total"}
        and all(isinstance(read[key], (int, float)) and read[key] >= 0 for key in read)
        and abs(read.get("total", 0) - read.get("before", 0) - read.get("after", 0)) <= 1e-12)
    return dict(valid=valid, engine_call=row.get("engine_call"), layer_count=row.get("layer_count"),
        before=before, after=after, delta=stored, counter_read_s=read,
        expert_tensor_payload_bytes=row.get("expert_tensor_payload_bytes"), accounting=row.get("accounting"))
def analyze(base):
    config_data = (base / "config.json").read_bytes(); config = json.loads(config_data)
    execution = json.loads((base / "execution.json").read_text()) if (base / "execution.json").exists() else {}
    declared = [(c.get("cell"), c.get("repeat"), c.get("document_set")) for c in config.get("cells", [])]
    protocol_ok = declared == EXPECTED and all(c.get("policy") == "static16" and c.get("chunk") == 16
        and c.get("new_prompt_length") == 128 and c.get("map_mode") == "batched" for c in config.get("cells", []))
    execution_rows = {row["cell"]: row for row in execution.get("cells", [])}
    out = dict(schema="document-state-abba-safe-v1", status=config.get("status", "UNRUN_OR_MISSING"),
        config=config, config_sha256=sha(config_data), execution={key: execution.get(key) for key in
            ("status", "pid", "start_unix", "end_unix", "document_prestate_sha256")},
        protocol=dict(valid=protocol_ok, expected_order=EXPECTED), cells={}, groups={}, consistency={})
    raws, summaries, invalid = {}, {}, not protocol_ok
    for cell in config.get("cells", []):
        name, group, path = cell["cell"], cell["document_set"], base / cell["cell"] / "result.json"
        if not path.exists():
            out["cells"][name] = dict(run_status="UNRUN_OR_MISSING", cell_config=cell); continue
        data, raw = path.read_bytes(), json.loads(path.read_text())
        entry = out["cells"][name] = dict(run_status=raw.get("status"), error=raw.get("error"),
            cell_config=cell, raw_sha256=sha(data))
        if raw.get("status") != "COMPLETED": invalid = True; continue
        raws[name] = raw; summaries[name] = summary = summarize(raw, execution_rows.get(name, {}))
        decision, counters, before = static_decisions(raw, 16), map_counters(raw), preaction_calls(raw)
        validation, measured, resource = raw.get("map_validation", {}), raw.get("map_measurement", {}), resources(raw)
        hashes = dict(prestate_sha256=prestate(raw), workload_sha256=raw.get("workload_sha256"),
            resources_sha256=jsha(resource), output_sha256={rid:jsha(row.get("output_token_ids", [])) for rid,row in raw["requests"].items()})
        map_pass = validation.get("status") == "PASS" and validation.get("layers") == 16 \
            and measured.get("mode") == "batched" and measured.get("totals", {}).get("scalar_device_assignments") == 0
        checks = dict(requests_complete=summary["complete_expected_requests"], identity=summary["identity_valid"]
            and summary["preaction_old_prefix_valid"], first_shape_18=summary["first_shape_valid"]
            and summary["first_action_tokens"] == 18, preaction_calls=before["valid"], decode_once=decision["ready_decode_advanced_once"],
            timing=all(row["timing_valid"] for row in summary["requests"].values()),
            no_preemption_or_recompute=not summary["scheduler_violations"], decisions=decision["valid"],
            map_pass=map_pass, first_action_map_counters=counters["valid"],
            workload_source=hashes["workload_sha256"] == config["workload_sources"][group]["sha256"],
            driver_prestate=hashes["prestate_sha256"] == execution.get("document_prestate_sha256", {}).get(group))
        entry.update(analysis_status="VALID" if all(checks.values()) else "INVALID_COMPARISON", metrics=metric_row(raw, summary),
            preaction_engine_calls_1_3=before, first_action_map_counters=counters,
            hashes=hashes, checks=checks, hardware=hardware_summary(base / "hardware.jsonl", raw, config["resources"]["cpu_affinity"]))
        invalid |= not all(checks.values())
    complete = len(raws) == 4 and execution.get("status") == "COMPLETED"
    for group in ("old", "new"):
        names = [c["cell"] for c in config.get("cells", []) if c.get("document_set") == group]
        if len(names) == 2 and all(name in raws for name in names):
            checks = dict(prestate_repeat=len({out["cells"][name]["hashes"]["prestate_sha256"] for name in names}) == 1,
                input_repeat=len({out["cells"][name]["hashes"]["workload_sha256"] for name in names}) == 1,
                output_repeat=out["cells"][names[0]]["hashes"]["output_sha256"] == out["cells"][names[1]]["hashes"]["output_sha256"])
            out["groups"][group] = dict(cells=names, checks=checks); invalid |= not all(checks.values())
    reuse = [row.get("reuse_record", {}) for row in execution.get("cells", [])]
    engine_keys = {(r.get("pid"), r.get("llm_id"), r.get("scheduler_id"), r.get("pool_id"), r.get("queue_id"),
        r.get("worker", {}).get("allocation_sha256")) for r in reuse}
    same_engine = len(reuse) == 4 and [r.get("invocation") for r in reuse] == list(range(4)) \
        and len(engine_keys) == 1 and None not in next(iter(engine_keys), (None,)) and all(r.get("status") == "READY"
        and r.get("native_schedule_restored") and r.get("threshold") == 32 and r.get("worker", {}).get("allocation_unchanged")
        and r.get("worker", {}).get("pager_empty") for r in reuse)
    out["consistency"] = dict(all_four_completed=complete, same_engine=same_engine,
        resources_across_documents=len({entry["hashes"]["resources_sha256"] for entry in out["cells"].values() if "hashes" in entry}) == 1 if complete else None,
        map_selftest=execution.get("map_selftest", {}).get("status") == "PASS" and execution.get("map_selftest", {}).get("cases") == 5 if complete else None)
    invalid |= complete and not all(out["consistency"].values())
    run_status = execution.get("status", config.get("status", "UNRUN_OR_MISSING"))
    out["status"] = "INVALID_PROTOCOL" if not protocol_ok else (run_status if run_status in ("STOPPED", "ABORT", "INVALID")
        else "INVALID_COMPARISON" if complete and invalid else "MEASUREMENT_ONLY" if complete else run_status)
    return out
def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--results",required=True,type=Path); parser.add_argument("--out",required=True,type=Path); args=parser.parse_args()
    encoded=json.dumps(analyze(args.results),indent=2,allow_nan=False)+"\n"
    if any(f'"{key}"' in encoded for key in ("prompt_token_ids","output_token_ids","token_received_s","requests","steps")): raise RuntimeError("unsafe raw field")
    with args.out.open("x") as handle: handle.write(encoded)
if __name__ == "__main__": main()
