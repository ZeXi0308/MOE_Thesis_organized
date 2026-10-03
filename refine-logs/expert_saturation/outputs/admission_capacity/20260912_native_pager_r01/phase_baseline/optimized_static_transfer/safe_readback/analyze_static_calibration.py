"""Export a safe summary for the frozen static8/32 -> unseen static16 run."""
import argparse, hashlib, json, statistics
from pathlib import Path
from analyze_prefill_midpoint import prestate
from analyze_wisp_injection import comparison, hardware_summary, resources, summarize
def sha(data):
    return hashlib.sha256(data).hexdigest()
def error(actual, predicted):
    delta = actual - predicted
    return dict(signed_s=delta, absolute_s=abs(delta),
                relative=delta / predicted if predicted else None)
def static_decisions(raw, chunk):
    action, issues, enabled, ready_seen = raw["action"]["engine_call"], [], 0, 0
    for step in raw["steps"]:
        if step["index"] < action:
            continue
        enabled += 1
        decision, scheduled = step.get("phase_prefill", {}), step.get("scheduled", [])
        ready = sorted(rid for rid in step["before"]["running"]
            if any(t < step["start_s"] for t in raw["requests"][rid]["token_received_s"])
            and raw["requests"][rid]["completion_s"] >= step["start_s"])
        ready_seen += len(ready)
        valid = (decision.get("enabled") is True
                 and decision.get("mode") == f"static{chunk}"
                 and decision.get("chosen_threshold") == chunk
                 and sorted(decision.get("ready_decode_ids", [])) == ready
                 and decision.get("status") == "applied"
                 and decision.get("actual_scheduled") == scheduled)
        for rid in ready:
            rows = [row for row in scheduled if row.get("request_id") == rid]
            valid &= (len(rows) == 1 and rows[0].get("tokens") == 1
                      and rows[0].get("prefill_tokens") == 0
                      and rows[0].get("decode_tokens") == 1)
        if not valid:
            issues.append(step["index"])
    return dict(valid=not issues and enabled > 0 and ready_seen > 0,
                issue_engine_calls=issues, enabled_engine_calls=enabled,
                expected_cap=chunk, ready_decode_observations=ready_seen,
                ready_decode_advanced_once=not issues)
def metric_row(raw, summary):
    old, new, request = summary["old_ids"], summary["new_ids"], summary["requests"]
    present_max = lambda values: max((v for v in values if v is not None), default=None)
    return dict(wall_s=summary["recorded_wall_s"],
        old_done_s=present_max(request[rid]["completion_s"] for rid in old),
        old_max_itl_s=present_max(request[rid]["max_itl_s"] for rid in old),
        new_ttft_s=request[new[0]]["ttft_s"] if len(new) == 1 else None,
        copy_payload_bytes=sum(layer["weight_copy_payload_bytes"]
            for worker in raw["final_worker"] for layer in worker["pager_layers"]),
        engine_calls=len(raw["steps"]), first_action_step_s=summary["first_action_step_s"],
        max_old_cross_action_itl_s=present_max(summary["old_cross_action_itl_s"].values()))
def midpoint_row(base, config, execution, raws, cells):
    path = base / "midpoint_prediction.json"
    complete = len(raws) == 6
    if not path.exists():
        return dict(status="INVALID_PROTOCOL" if complete else "UNRUN_OR_MISSING"), complete
    data, frozen = path.read_bytes(), json.loads(path.read_text())
    endpoints, unseen = config["cells"][:4], config["cells"][4:]
    stored = {row["cell"]: row for row in frozen.get("calibration", [])}
    hashes, values, stored_match = [], [], True
    for cell in endpoints:
        name, row = cell["cell"], stored.get(cell["cell"], {})
        observed = cells.get(name, {}).get("raw_sha256")
        hashes.append(dict(cell=name, expected=row.get("raw_sha256"),
                           observed=observed, matches=observed == row.get("raw_sha256")))
        if name in raws:
            metric = cells[name]["metrics"]
            values.append((cell["chunk"], metric))
            stored_match &= (row.get("first_step_s") == metric["first_action_step_s"]
                and row.get("first_old_itl_s") == metric["max_old_cross_action_itl_s"])
    prediction_match = False
    if len(values) == 4:
        means = {chunk: {key: statistics.mean(row[key] for cap, row in values if cap == chunk)
            for key in ("first_action_step_s", "max_old_cross_action_itl_s")}
            for chunk in (8, 32)}
        predicted = dict(first_step_s=means[8]["first_action_step_s"]
            + (means[32]["first_action_step_s"] - means[8]["first_action_step_s"]) / 3,
            first_old_itl_s=means[8]["max_old_cross_action_itl_s"]
            + (means[32]["max_old_cross_action_itl_s"]
               - means[8]["max_old_cross_action_itl_s"]) / 3)
        prediction_match = all(frozen.get("prediction", {}).get(key) == value
                               for key, value in predicted.items())
    starts = {row["cell"]: row.get("start_unix") for row in execution.get("cells", [])}
    actual, frozen_before, frozen_before_cells = [], True, True
    for cell in unseen:
        name, raw = cell["cell"], raws.get(cell["cell"])
        if raw is None:
            actual.append(dict(cell=name, status="UNRUN_OR_MISSING"))
            frozen_before = False
            continue
        action_start = raw["measurement_origin_unix_s"] + raw["action"]["start_s"]
        before = frozen["created_unix"] < action_start
        before_cell = frozen["created_unix"] < starts.get(name, float("-inf"))
        frozen_before &= before
        frozen_before_cells &= before_cell
        metric, predicted = cells[name]["metrics"], frozen["prediction"]
        actual.append(dict(cell=name, action_start_unix_s=action_start,
            frozen_before_cell_start=before_cell, frozen_before_action_start=before,
            first_step_s=metric["first_action_step_s"],
            first_step_prediction_s=predicted["first_step_s"],
            first_step_error=error(metric["first_action_step_s"], predicted["first_step_s"]),
            max_old_cross_action_itl_s=metric["max_old_cross_action_itl_s"],
            max_old_cross_action_itl_prediction_s=predicted["first_old_itl_s"],
            max_old_cross_action_itl_error=error(
                metric["max_old_cross_action_itl_s"], predicted["first_old_itl_s"])))
    checks = dict(frozen_status=frozen.get("status") == "FROZEN_BEFORE_CAP16",
        four_endpoint_raw_hashes=len(hashes) == 4 and all(row["matches"] for row in hashes),
        stored_endpoint_metrics=stored_match and len(values) == 4,
        prediction_recomputed=prediction_match,
        frozen_before_both_static16_cells=frozen_before_cells and len(actual) == 2,
        frozen_before_both_static16_actions=frozen_before and len(actual) == 2,
        embedded_copy_matches=execution.get("midpoint_prediction") == frozen)
    valid = complete and all(checks.values())
    return dict(status="VALIDATED" if valid else ("INVALID_PROTOCOL" if complete else "INCOMPLETE"),
        file_sha256=sha(data), created_unix=frozen.get("created_unix"),
        model=frozen.get("model"), scope=frozen.get("scope"),
        endpoint_means=frozen.get("endpoint_means"), prediction=frozen.get("prediction"),
        endpoint_raw_hashes=hashes, checks=checks, unseen_static16=actual), complete and not valid
def analyze(base, prediction_row=midpoint_row):
    config_path, execution_path = base / "config.json", base / "execution.json"
    config_data = config_path.read_bytes()
    config = json.loads(config_data)
    execution = json.loads(execution_path.read_text()) if execution_path.exists() else {}
    expected = [("r0-static8", 0, 8), ("r0-static32", 0, 32),
                ("r1-static32", 1, 32), ("r1-static8", 1, 8),
                ("r0-static16", 0, 16), ("r1-static16", 1, 16)]
    declared = [(row.get("cell"), row.get("repeat"), row.get("chunk"))
                for row in config.get("cells", [])]
    rows = {row["cell"]: row for row in execution.get("cells", [])}
    artifacts = {name: sha((base / name).read_bytes()) if (base / name).exists() else None
        for name in ("config.json", "run_phase_repeated.py", "freeze_prefill_midpoint.py",
                     "execution.json", "midpoint_prediction.json", "source_manifest.json")}
    out = dict(schema="optimized-static-calibration-safe-v1", status="INCOMPLETE",
        config=config, config_sha256=sha(config_data), artifact_sha256=artifacts,
        execution=dict(status=execution.get("status", "UNRUN_OR_MISSING"), pid=execution.get("pid"),
            start_unix=execution.get("start_unix"), end_unix=execution.get("end_unix"),
            reference_prestate_sha256=execution.get("reference_prestate_sha256")),
        protocol=dict(declared_order_valid=declared == expected,
            engine_call_definition="len(result.steps), distinct from MoE ensure calls and 16 per-layer calls"),
        cells={}, output_hash_boundary="Hashes are record-only; different fixed caps may produce different outputs.",
        scope=config.get("holdout_scope"))
    invalid, raws, summaries = declared != expected, {}, {}
    for cell in config.get("cells", []):
        name, path, execution_row = cell["cell"], base / cell["cell"] / "result.json", rows.get(cell["cell"], {})
        if not path.exists():
            out["cells"][name] = dict(status="UNRUN_OR_MISSING", cell_config=cell)
            continue
        data, raw = path.read_bytes(), json.loads(path.read_text())
        entry = out["cells"][name] = dict(status=raw.get("status"), cell_config=cell,
                                               raw_sha256=sha(data))
        if raw.get("status") != "COMPLETED":
            continue
        raws[name] = raw
        summaries[name] = summary = summarize(raw, execution_row)
        decision = static_decisions(raw, cell["chunk"])
        validation, measured = raw.get("map_validation", {}), raw.get("map_measurement", {})
        map_pass = (validation.get("status") == "PASS" and validation.get("layers") == 16
                    and measured.get("mode") == "batched"
                    and measured.get("totals", {}).get("scalar_device_assignments") == 0)
        checks = dict(requests_complete=summary["complete_expected_requests"],
            identity=summary["identity_valid"] and summary["preaction_old_prefix_valid"],
            first_action_shape=summary["first_shape_valid"],
            timing=all(row["timing_valid"] for row in summary["requests"].values()),
            no_preemption_or_recompute=not summary["scheduler_violations"],
            prestate_matches_reference=prestate(raw) == execution.get("reference_prestate_sha256"),
            decisions=decision, map_pass=map_pass)
        entry.update(metrics=metric_row(raw, summary), checks=checks,
            hashes=dict(prestate_sha256=prestate(raw), workload_sha256=raw.get("workload_sha256"),
                resources_sha256=sha(json.dumps(resources(raw), sort_keys=True,
                    separators=(",", ":")).encode()), output_sha256={rid: sha(json.dumps(
                    row.get("output_token_ids", [])).encode()) for rid, row in raw["requests"].items()}),
            hardware=hardware_summary(base / "hardware.jsonl", raw,
                                      config["resources"]["cpu_affinity"]))
        if not all(value for key, value in checks.items() if key != "decisions") or not decision["valid"]:
            entry["status"], invalid = "INVALID_COMPARISON", True
    complete = len(raws) == 6 and execution.get("status") == "COMPLETED"
    consistency = dict(all_six_cells_completed=complete, same_prestate=None,
                       same_tasks=None, same_resources=None)
    if complete:
        reference = config["cells"][0]["cell"]
        pairs = [comparison(raws[reference], raws[name], summaries[reference], summaries[name],
                            "same static-calibration task") for name in raws]
        consistency.update(same_prestate=len({prestate(raw) for raw in raws.values()}) == 1,
            same_tasks=all(row["same_input_tasks"] for row in pairs),
            same_resources=all(row["resources_same"] for row in pairs))
        invalid |= not all(consistency.values())
    out["consistency"] = consistency
    out["midpoint"], midpoint_invalid = prediction_row(base, config, execution, raws, out["cells"])
    invalid |= midpoint_invalid
    out["status"] = "INVALID_COMPARISON" if invalid else (
        "MEASUREMENT_ONLY" if complete and out["midpoint"]["status"] == "VALIDATED" else "INCOMPLETE")
    return out
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    encoded = json.dumps(analyze(args.results), indent=2, allow_nan=False) + "\n"
    if any(f'"{key}"' in encoded for key in
           ("prompt_token_ids", "output_token_ids", "token_received_s", "requests", "steps")):
        raise RuntimeError("unsafe raw field in safe export")
    with args.out.open("x") as handle:
        handle.write(encoded)
if __name__ == "__main__":
    main()
