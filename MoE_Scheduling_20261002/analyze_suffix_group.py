"""Per-cell complete-service and suffix-action accounting; retain failed cells.

Example: /private/tmp/moe-c-input-env/bin/python analyze_suffix_group.py \
    --results results_suffix_r01
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re

from analyze_kv_swap import ROOT, analyze_kv, comparison, load, ratio, sha
from analyze_optimization import distribution


def optional_json(path, errors):
    if not path.is_file():
        return None
    try:
        return load(path)
    except (OSError, ValueError) as exc:
        errors.append(f"{path.name}: {type(exc).__name__}: {exc}")
        return None


def decision_key(row):
    return row.get("phase"), row.get("step_id"), row.get("layer")


def measured_phase(phase, phases):
    return phase in phases if phases else isinstance(phase, str) and phase.endswith("measurement")


def compaction_accounting(directory, config, phases, suffix_trace, pager_rows, step_mapping):
    """Minimal execution/causal-prefix ledger for the optional compact adapter."""
    errors = []
    path = directory / "compaction_trace.json"
    if not path.is_file():
        path = directory / "compaction_trace_failure.json"
    trace = optional_json(path, errors)
    if trace is None:
        return dict(trace_status="MISSING" if config.get("suffix_compact") else "NOT_REQUESTED",
                    available=False, errors=errors)
    records = [r for r in trace.get("steps", []) if measured_phase(r.get("phase"), phases)]
    states = {s["step_id"]: s for s in (suffix_trace or {}).get("state_steps", [])}
    totals, layers, reasons = Counter(), defaultdict(Counter), Counter()
    checks = []
    for record in records:
        applied = bool(record.get("compact_applied"))
        totals.update(measured_forward_steps=1, compacted_steps=int(applied),
            complete_compacted_steps=int(applied and record.get("status") == "complete"),
            failed_steps=int(record.get("status") == "failed"),
            no_cut_steps=int(record.get("fallback_reason") == "no_cut"),
            fallback_steps=int(not applied and record.get("fallback_reason") not in (None, "no_cut")),
            compacted_logits_steps=int(record.get("logits_compacted", False)))
        if record.get("fallback_reason"):
            reasons[record["fallback_reason"]] += 1
        for row in record.get("layer_rows", []):
            original, actual = row["original_rows"], row["input_rows"]
            values = dict(recorded_layer_calls=1, original_query_rows=original,
                actual_input_query_rows=actual, removed_query_row_layer_positions=original - actual,
                compacted_layer_calls=int(row.get("compact", False)))
            layers[str(row["layer"])].update(values)
            if row.get("compact"):
                totals.update(tail_layer_calls=1, tail_original_query_rows=original,
                    tail_actual_query_rows=actual, tail_removed_query_row_layer_positions=original - actual)
        if record.get("logits_compacted"):
            totals.update(lmhead_original_rows=record["logits_original_rows"],
                lmhead_executed_rows=record["logits_executed_rows"],
                lmhead_removed_rows=record["logits_original_rows"] - record["logits_executed_rows"])
        if not applied:
            continue
        plan, state = record.get("plan", {}), states.get(record["step_id"])
        q, seq, context = plan.get("query_lens", []), plan.get("seq_lens", []), plan.get("context_lens", [])
        starts, indices = plan.get("query_start_loc", []), plan.get("indices", [])
        count = len(q)
        current = dict(phase=record.get("phase"), suffix_step_id=record["step_id"],
            status=record.get("status"), after_layer=record.get("after_layer"),
            causal_query_seq_consistent=(bool(q) and len(seq) == len(context) == count
                and all(length >= 1 and s - length == c for length, s, c in zip(q, seq, context))),
            query_offsets_consistent=(len(starts) == count + 1 and starts[0] == 0
                and [b - a for a, b in zip(starts, starts[1:])] == q
                and starts[-1] == plan.get("num_actual_tokens") == len(indices)),
            original_state_prefix_match=None, lmhead_prefix_match=None)
        if state is not None:
            expected_q = [k + 1 if eligible else original for k, eligible, original in zip(
                state["keep_drafts"], state["eligible"], state["num_scheduled_tokens"])]
            expected_indices = [i for start, length in zip(state["row_starts"], expected_q)
                                for i in range(start, start + length)]
            current["original_state_prefix_match"] = (q == expected_q and indices == expected_indices
                and context == state["computed_starts"] and len(q) == len(state["req_ids"]))
            take, offset = [], 0
            for original, keep in zip(state["original_draft_counts"], state["keep_drafts"]):
                take.extend(range(offset, offset + keep + 1)); offset += original + 1
            if record.get("logits_compacted"):
                current["lmhead_prefix_match"] = (record["logits_original_rows"] == offset
                    and record["logits_executed_rows"] == len(take)
                    and record.get("logits_kept_indices") == take)
        tail = [r for r in record.get("layer_rows", []) if r["layer"] > record["after_layer"]]
        current["tail_layer_rows_match_plan"] = all(r.get("compact") and r["input_rows"] == len(indices) for r in tail) if tail else None
        current["hidden_return_shape_restored"] = (record["returned_rows"] == state["total_rows"]
            if state is not None and record.get("returned_rows") is not None else None)
        scheduler_step = step_mapping.get((record.get("phase"), record["step_id"]))
        actual_calls = pager_rows.get((record.get("phase"), scheduler_step), []) if scheduler_step is not None else []
        if actual_calls and tail:
            current["pager_tail_rows_match"] = all(
                len([call for call in actual_calls if call["layer"] == row["layer"]
                    and call["input_rows"] == row["input_rows"] == call["moe_rows"]
                    and call["original_index_count"] == len(indices)]) == 1 for row in tail)
        else:
            current["pager_tail_rows_match"] = None
        checks.append(current)
    timing = {}
    for key in ("forward_host_s", "compact_setup_host_s", "hidden_scatter_host_s", "logits_adapter_host_s"):
        values = [r[key] for r in records if r.get(key) is not None]
        timing[key] = dict(distribution(values), sum_s=sum(values))
    check_names = ("causal_query_seq_consistent", "query_offsets_consistent", "original_state_prefix_match",
                   "tail_layer_rows_match_plan", "hidden_return_shape_restored", "lmhead_prefix_match", "pager_tail_rows_match")
    return dict(available=True, trace_status="PRESENT" if path.name == "compaction_trace.json" else "PARTIAL_FAILURE_TRACE",
        source_file=path.name, activity=dict(totals), fallback_reason_counts=dict(reasons),
        by_layer={key: dict(value) for key, value in sorted(layers.items(), key=lambda item: int(item[0]))},
        checks={name: dict(checked=sum(c[name] is not None for c in checks),
                          passed=sum(c[name] is True for c in checks), failed=sum(c[name] is False for c in checks)) for name in check_names},
        step_checks=checks, host_timing=timing, errors=errors,
        scope="Recorded layer input shapes establish the tail execution shape; query/seq checks validate recorded plan arithmetic and original surviving prefixes, not an independent GPU metadata readback or numerical-equivalence proof. The decision layer's attention stays full length. Missing checks remain unknown.",
        timing_scope="forward_host includes all decoder layers plus compact_setup and hidden_scatter. logits_adapter includes the reduced LM-head call and its gather/scatter, after inner forward but inside engine.step. These host windows include submissions/possible waits; none is added to engine wall or CUDA spans.")


def suffix_accounting(directory, raw, config):
    errors = []
    trace_path = directory / "suffix_trace.json"
    if not trace_path.is_file():
        trace_path = directory / "suffix_trace_failure.json"
    trace = optional_json(trace_path, errors)
    totals, by_layer = Counter(), defaultdict(Counter)
    phases, embedded, comparisons = set(), {}, []
    pager_rows, step_mapping = defaultdict(list), {}
    calls_path = directory / "pager/calls.jsonl"
    if calls_path.is_file():
        with calls_path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                try:
                    call = json.loads(line)
                except ValueError as exc:
                    errors.append(f"pager/calls.jsonl line {line_number}: {exc}")
                    continue
                if not call.get("measurement") or call.get("validation_run"):
                    continue
                phase = call.get("context", {}).get("phase")
                if phase is not None:
                    phases.add(phase)
                layer = call.get("layer_name", "unknown")
                groups = call.get("groups", [])
                actual_bytes = call.get("weight_copy_bytes")
                group_bytes = sum(g.get("weight_copy_bytes", 0) for g in groups)
                loaded = [e for g in groups for e in g.get("loaded_experts", [])]
                after_cold = set(call.get("active_experts", [])).difference(call.get("entry_resident_experts", []))
                scheduled_rows = call.get("scheduled_rows", call.get("rows", 0))
                input_rows = call.get("input_rows", scheduled_rows)
                removed_rows = scheduled_rows - call.get("rows", 0)
                values = dict(calls=1, failed_calls=int(call.get("status") != "complete"),
                    groups=len(groups), actual_weight_copy_bytes=actual_bytes or 0,
                    actual_load_events=len(loaded), executed_moe_rows=call.get("rows", 0),
                    scheduled_row_layer_positions=scheduled_rows, arriving_moe_rows=input_rows,
                    cross_layer_removed_row_positions=scheduled_rows - input_rows,
                    within_moe_removed_row_positions=input_rows - call.get("rows", 0),
                    removed_row_layer_positions=removed_rows,
                    calls_with_removed_rows=int(removed_rows > 0),
                    route_to_host_recorded_s=call.get("route_to_host_ms", 0) / 1000,
                    host_apply_inclusive_s=call.get("host_apply_ms", 0) / 1000,
                    group_byte_total_mismatches=int(actual_bytes != group_bytes),
                    after_cold_load_mismatches=int(sorted(loaded) != sorted(after_cold)))
                spans = [g.get("load_cuda_span_ms") for g in groups]
                values.update(load_cuda_span_recorded_s=sum(x for x in spans if x is not None) / 1000,
                    load_cuda_span_missing_groups=sum(x is None for x in spans))
                totals.update(values)
                by_layer[layer].update(values)
                layer_match = re.search(r"layers\.(\d+)", layer)
                if layer_match and "input_rows" in call:
                    pager_rows[(phase, call.get("context", {}).get("step_id"))].append(dict(
                        layer=int(layer_match[1]), input_rows=input_rows, moe_rows=call.get("rows"),
                        original_index_count=len(call.get("execution_original_row_indices", []))))
                decision = call.get("suffix_decision")
                if not decision:
                    continue
                key = decision_key(decision)
                step_mapping[(phase, decision.get("step_id"))] = call.get("context", {}).get("step_id")
                if key in embedded:
                    errors.append(f"duplicate measured decision key {key}")
                embedded[key] = decision
                expected_after = decision.get("entry_cold_bytes_after")
                before = decision.get("entry_cold_bytes_before")
                saved = decision.get("current_layer_saved_bytes")
                checks = dict(
                    call_complete=call.get("status") == "complete",
                    after_bytes_match=expected_after == actual_bytes if expected_after is not None else None,
                    after_cold_count_match=decision.get("entry_cold_experts_after") == len(after_cold),
                    each_entry_miss_loaded_once=sorted(loaded) == sorted(after_cold),
                    group_bytes_match=actual_bytes == group_bytes,
                    eliminated_experts_not_loaded=not set(decision.get("eliminated_cold_experts", [])).intersection(loaded),
                    local_byte_accounting=(before - saved == expected_after
                                           if all(x is not None for x in (before, saved, expected_after)) else None))
                comparisons.append(dict(phase=phase, suffix_step_id=decision.get("step_id"),
                    scheduler_step_id=call.get("context", {}).get("step_id"), layer=layer,
                    pager_call_id=call.get("call_id"), mode=decision.get("mode"),
                    removed_input_rows=decision.get("removed_input_rows"),
                    current_layer_predicted_saved_bytes=saved,
                    current_layer_expected_after_bytes=expected_after,
                    actual_current_call_copy_bytes=actual_bytes,
                    predicted_remaining_saved_s=decision.get("predicted_remaining_saved_s"),
                    checks=checks, matches=all(value is True for value in checks.values())))
    else:
        errors.append("pager/calls.jsonl missing; actual call accounting unavailable")

    decisions = dict(embedded)
    observations, state_steps, cancellations = [], [], []
    if trace is not None:
        observations = [o for o in trace.get("observations", []) if measured_phase(o.get("phase"), phases)]
        for record in trace.get("decisions", []):
            if measured_phase(record.get("phase"), phases):
                key = decision_key(record)
                if key in embedded and record != embedded[key]:
                    errors.append(f"trace/embedded decision differs at {key}")
                decisions[key] = record
        measured_ids = {o.get("step_id") for o in observations}
        measured_ids.update(d.get("step_id") for d in decisions.values())
        state_steps = [s for s in trace.get("state_steps", []) if s.get("step_id") in measured_ids]
        cancellations = [c for c in trace.get("cancellations", []) if c.get("step_id") in measured_ids]
        if len({s["step_id"] for s in state_steps}) != len(state_steps):
            errors.append("duplicate selected suffix state step_id")
    measured_decisions = list(decisions.values())
    source_map = (raw or {}).get("internal_to_source", {})
    requests = defaultdict(Counter)
    steps = Counter()
    for state in state_steps:
        ids = state.get("req_ids", [])
        original, keep = state.get("original_draft_counts", []), state.get("keep_drafts", [])
        accepted = state.get("accepted_before_stop_filter")
        sampled = state.get("sampler_token_counts_before_stop_filter")
        if len(ids) != len(original) or len(ids) != len(keep):
            errors.append(f"unaligned state lists at step {state.get('step_id')}")
            continue
        if accepted is not None and len(accepted) != len(ids):
            errors.append(f"unaligned accepted list at step {state.get('step_id')}")
            accepted = None
        if sampled is not None and len(sampled) != len(ids):
            errors.append(f"unaligned sampler list at step {state.get('step_id')}")
            sampled = None
        cut = any(k < m for k, m in zip(keep, original))
        steps.update(measured_state_steps=1, cut_steps=int(cut),
            eligible_steps=int(any(state.get("eligible", []))),
            sampler_count_steps=int(sampled is not None),
            sample_repack_host_s=state.get("sample_repack_host_ms", 0) / 1000,
            cut_sample_repack_host_s=state.get("sample_repack_host_ms", 0) / 1000 if cut else 0)
        for i, rid in enumerate(ids):
            source = source_map.get(rid, rid)
            values = dict(request_steps=1, cut_request_steps=int(keep[i] < original[i]),
                original_draft_rows=original[i], kept_draft_rows=keep[i],
                removed_draft_rows=original[i] - keep[i])
            if sampled is not None:
                values["sampler_tokens_before_stop_filter"] = sampled[i]
            if accepted is not None and accepted[i] is not None:
                values["accepted_drafts_before_stop_filter"] = accepted[i]
            requests[source].update(values)
            steps.update(values)
    final_output = sum(len(r.get("output_token_ids", [])) for r in (raw or {}).get("requests", [])) if raw else None
    observed_output = sum(o.get("committed_tokens", 0) for o in observations) if observations else None
    before_stop = steps.get("sampler_tokens_before_stop_filter") if steps.get("sampler_count_steps") else None
    matched = [r for r in comparisons if r["matches"]]
    measured_pager_bytes = totals.get("actual_weight_copy_bytes") if calls_path.is_file() else None
    known_ar = config.get("ngram_speculative_tokens") == 0
    if trace is None and config.get("ngram_speculative_tokens", 0) > 0:
        errors.append("suffix_trace.json missing; sampler and per-request cut accounting unavailable")
    activity = dict(steps, available=trace is not None or known_ar,
        unique_cut_requests=sum(v["cut_request_steps"] > 0 for v in requests.values()) if trace is not None or known_ar else None,
        cancellation_events=len(cancellations) if trace is not None else None,
        measured_decisions=len(measured_decisions))
    if trace is None and not known_ar:
        activity.update(cut_steps=None, cut_request_steps=None, removed_draft_rows=None)
    return dict(
        trace_status=("PRESENT" if trace_path.name == "suffix_trace.json" else "PARTIAL_FAILURE_TRACE")
            if trace is not None else "NOT_EXPECTED_FOR_AR" if known_ar else "MISSING",
        trace_source_file=trace_path.name if trace is not None else None,
        mode=(trace or {}).get("mode", config.get("suffix_policy")),
        decision_layer=(trace or {}).get("decision_layer", config.get("suffix_layer")),
        measurement_phases=sorted(p for p in phases if p is not None),
        actual_pager=dict(totals), by_layer={name: dict(value) for name, value in sorted(by_layer.items())},
        pager_row_scope="scheduled_rows is the original scheduled total; input_rows is the actual tensor arriving at MoE; rows is the MoE-executed tensor after any local gather. Older records without input_rows use scheduled_rows for that pre-gather size. Cross-layer and within-MoE row removals partition the total, not extra tokens saved.",
        activity=activity,
        per_request={rid: dict(value) for rid, value in sorted(requests.items())},
        submission=dict(sampler_tokens_before_stop_filter=before_stop,
            accepted_drafts_before_stop_filter=steps.get("accepted_drafts_before_stop_filter") if before_stop is not None else None,
            controller_observed_returned_tokens=observed_output, final_returned_tokens=final_output,
            observed_returned_equals_raw=observed_output == final_output if observed_output is not None and raw else None,
            before_stop_minus_final=before_stop - final_output if before_stop is not None and final_output is not None else None,
            scope="Native sampler counts precede EOS/stop filtering; final returned IDs define delivered output and throughput."),
        overhead=dict(decision_host_s=sum(d.get("decision_host_s", 0) for d in measured_decisions),
            decision_host_max_s=max((d.get("decision_host_s", 0) for d in measured_decisions), default=None),
            sample_repack_host_s=steps.get("sample_repack_host_s") if state_steps else None,
            cut_sample_repack_host_s=steps.get("cut_sample_repack_host_s") if state_steps else None,
            separate_gather_scatter_s=None,
            scope="Diagnostic times already occur inside engine/capture wall. host_apply includes route transfer, controller and MoE-local gather/scatter. route_to_host follows the archived runtime timer boundary (r01 records it before controller.select, so it covers D2H/tolist only). No sums are added to service time. MoE-local gather/scatter have no separate timers; optional cross-layer compaction timers are reported separately under compaction."),
        local_prediction=dict(decision_calls_checked=len(comparisons), matched_calls=len(matched),
            all_current_calls_match=all(r["matches"] for r in comparisons) if comparisons else None,
            decisions_without_measured_pager_call=[dict(phase=k[0], step_id=k[1], layer=k[2]) for k in decisions if k not in embedded],
            current_layer_set_savings_bytes_sum=sum(d.get("current_layer_saved_bytes", 0) for d in measured_decisions),
            matched_current_layer_set_savings_bytes_sum=sum(r["current_layer_predicted_saved_bytes"] or 0 for r in matched),
            horizon_or_local_predicted_remaining_saved_s=sum(d.get("predicted_remaining_saved_s") or 0 for d in measured_decisions),
            actual_whole_episode_copy_bytes=measured_pager_bytes,
            calls=comparisons,
            scope="Only the decision layer's same-state before/after union is checked against actual after loads. The sum is local modeled avoidance, not measured episode byte/time reduction. Later layers report actual execution only; their dead-row routes are never a no-cut counterfactual."),
        acceptance_calibrations=(trace or {}).get("acceptance_calibrations", []),
        compaction=compaction_accounting(directory, config, phases, trace, pager_rows, step_mapping), errors=errors)


def discover_cells(results, group, requested):
    names = set(requested or [])
    names.update(p.name if p.is_dir() else p.stem for p in results.iterdir()
                 if re.match(r"^\d+_", p.name) and (p.is_dir() or p.suffix == ".log"))
    names.update(group.get("completed", []))
    if group.get("arm"):
        names.add(group["arm"])
    return sorted(names)


def analyze_group(results, workload, tokenizer, *, budget=2048, requested=None, baseline=None):
    group_errors = []
    group = optional_json(results / "group_status.json", group_errors) or {}
    cells = {}
    for name in discover_cells(results, group, requested):
        directory = results / name
        errors = []
        status = optional_json(directory / "status.json", errors) or {}
        raw = optional_json(directory / "raw.json", errors)
        config = optional_json(directory / "config.json", errors) or {}
        result = dict(path=str(directory), status=status.get("status", "MISSING_STATUS"),
            raw_status=(raw or {}).get("status"), analysis_status="UNAVAILABLE", errors=errors,
            run_error=status.get("error") or (raw or {}).get("error"),
            log_path=str(results / (name + ".log")) if (results / (name + ".log")).is_file() else None,
            config=config, metrics=None)
        if raw is None:
            errors.append("raw.json unavailable; no complete-service performance claim")
        else:
            try:
                metrics = analyze_kv(directory, workload["source_requests"],
                    workload["arrival_traces_s"]["steady"], tokenizer, budget)
                result.update(metrics=metrics, analysis_status="COMPLETE" if metrics["all_16_complete"] else "PARTIAL")
            except Exception as exc:
                errors.append(f"analyze_kv: {type(exc).__name__}: {exc}")
                result["analysis_status"] = "ANALYSIS_FAILED"
        try:
            result["suffix"] = suffix_accounting(directory, raw, config)
        except Exception as exc:
            result["suffix"] = dict(errors=[f"suffix accounting: {type(exc).__name__}: {exc}"])
        if result["metrics"] and "actual_pager" in result["suffix"]:
            result["suffix"]["pager_total_matches_service_metrics"] = (
                result["suffix"]["actual_pager"].get("actual_weight_copy_bytes") == result["metrics"]["weight_copy_bytes"])
        cells[name] = result
    complete = {n: c["metrics"] for n, c in cells.items() if c["analysis_status"] == "COMPLETE"}
    baseline = baseline or next(iter(complete), None)
    comparisons = []
    if baseline in complete:
        for name in complete:
            if name == baseline:
                continue
            record = comparison(name, baseline, complete)
            record["drained_ratios"] = dict(
                time=ratio(complete[name]["kv_service_cost"].get("drained_s"), complete[baseline]["kv_service_cost"].get("drained_s")),
                actual_output_rate=ratio(complete[name]["drained_output_tokens_per_s"], complete[baseline]["drained_output_tokens_per_s"]))
            comparisons.append(record)
    elif baseline is not None:
        group_errors.append(f"baseline {baseline} is not a complete analyzed cell; comparisons omitted")
    return dict(evidence_type="ACTUAL_SUFFIX_PROTOTYPE_PER_CELL", results_path=str(results),
        group_status=group, order=list(cells), cells=cells, baseline=baseline,
        comparisons=comparisons, errors=group_errors,
        notes=["Every discovered directory/log cell is retained, including missing and failed captures; no fixed number of repeats is assumed.",
            "analyze_kv supplies unchanged natural-EOS scoring, request outputs, flow/TTFT, host chunk gaps, and served/drained cost.",
            "Engine wall, host sub-times and CUDA load spans overlap in scope and are never added together.",
            "Different output lengths/content are preserved; episode ratios are not equal-work speedups. Multi-token receipt internal ITL remains unresolved.",
            "Only observed decision-layer routes support local before/after union accounting; no later-layer dead-row counterfactual is used."])


def markdown_report(report):
    def fmt(value, digits=4):
        return "—" if value is None else f"{value:.{digits}f}" if isinstance(value, float) else str(value)
    lines = ["# Suffix prototype: per-cell results", "",
        f"Group status: **{report['group_status'].get('status', 'UNKNOWN')}**. Baseline: `{report['baseline']}`.", "",
        "All discovered cells are retained. Time includes complete served/drained cost; initialization and warmup follow the existing capture boundary.", "",
        "| Cell | Analysis / run | Drained s | Returned tokens | Token/s | Mean flow s | Mean TTFT s | Max chunk gap s | Correct | Expert GB |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, cell in report["cells"].items():
        m = cell.get("metrics") or {}
        row = [name, f"{cell['analysis_status']} / {cell['status']}", (m.get("kv_service_cost") or {}).get("drained_s"),
            m.get("output_tokens"), m.get("drained_output_tokens_per_s"), (m.get("flow") or {}).get("mean_s"),
            (m.get("ttft") or {}).get("mean_s"), (m.get("inter_chunk_gap") or {}).get("max_s"),
            f"{m['correct_requests']}/{m['requests']}" if "correct_requests" in m else None,
            m["weight_copy_bytes"] / 1e9 if m.get("weight_copy_bytes") is not None else None]
        lines.append("| " + " | ".join(fmt(x) for x in row) + " |")
    lines += ["", "| Cell | Cut steps / request-steps / unique requests | Removed draft rows | Sampler tokens before stop | Returned tokens | Decision / repack s | Local union eliminated GB | Decision-call after-load matches |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, cell in report["cells"].items():
        s = cell.get("suffix") or {}; a = s.get("activity", {}); u = s.get("submission", {}); h = s.get("overhead", {}); p = s.get("local_prediction", {})
        counts = "/".join(fmt(a.get(k, 0)) for k in ("cut_steps", "cut_request_steps", "unique_cut_requests")) if a.get("available") else None
        row = [name, counts, a.get("removed_draft_rows"), u.get("sampler_tokens_before_stop_filter"), u.get("final_returned_tokens"),
            f"{fmt(h.get('decision_host_s'))} / {fmt(h.get('sample_repack_host_s'))}",
            p["current_layer_set_savings_bytes_sum"] / 1e9 if "current_layer_set_savings_bytes_sum" in p else None,
            f"{p['matched_calls']}/{p['decision_calls_checked']}" if "matched_calls" in p else None]
        lines.append("| " + " | ".join(fmt(x) for x in row) + " |")
    lines += ["", "Local union eliminated bytes describe the same decision-layer state, not a measured whole-episode byte reduction. Later layers contribute only their actual loads. Horizon time savings remain predictions.",
        "Decision/repack and CUDA spans already lie inside measured service time. MoE-local gather/scatter has no isolated timer. The archived r01 source records route-to-host before controller.select (D2H/tolist only); host_apply includes controller and MoE-local gather/scatter. Optional cross-layer compaction timers appear separately below. These diagnostics are not added to engine wall.", ""]
    compact_cells = {name: cell.get("suffix", {}).get("compaction", {}) for name, cell in report["cells"].items()
                     if cell.get("suffix", {}).get("compaction", {}).get("trace_status") not in (None, "NOT_REQUESTED")}
    if compact_cells:
        lines += ["| Cell | Compact / measured forwards | Fallback / failed | Tail query row-layer positions: original → actual | LM-head rows: original → executed | Causal q/seq matches | Pager tail matches |",
            "|---|---:|---:|---:|---:|---:|---:|"]
        for name, compact in compact_cells.items():
            a, checks = compact.get("activity", {}), compact.get("checks", {})
            def count_check(key):
                value = checks.get(key)
                return f"{value['passed']}/{value['checked']}" if value else "—"
            row = [name, f"{a.get('compacted_steps', 0)}/{a.get('measured_forward_steps', 0)}" if a else None,
                f"{a.get('fallback_steps', 0)}/{a.get('failed_steps', 0)}" if a else None,
                f"{a.get('tail_original_query_rows', 0)} → {a.get('tail_actual_query_rows', 0)}" if a else None,
                f"{a.get('lmhead_original_rows', 0)} → {a.get('lmhead_executed_rows', 0)}" if a else None,
                count_check("causal_query_seq_consistent"), count_check("pager_tail_rows_match")]
            lines.append("| " + " | ".join(fmt(x) for x in row) + " |")
        lines += ["", "Tail rows come from recorded per-layer execution inputs; causal checks verify the recorded prefix plan, not independent GPU metadata readback. The decision-layer attention keeps its original shape. LM-head rows refer to compacted calls only.", "",
            "| Cell | Compact setup host s | Hidden scatter host s | Logits adapter host s | Forward host s (inclusive) | Fallback reasons |",
            "|---|---:|---:|---:|---:|---|"]
        for name, compact in compact_cells.items():
            timing = compact.get("host_timing", {})
            row = [name] + [timing.get(key, {}).get("sum_s") for key in
                ("compact_setup_host_s", "hidden_scatter_host_s", "logits_adapter_host_s", "forward_host_s")]
            row.append(json.dumps(compact.get("fallback_reason_counts", {}), ensure_ascii=False))
            lines.append("| " + " | ".join(fmt(x) for x in row) + " |")
        lines += ["", "Forward host includes compact setup and hidden scatter. The logits adapter includes the reduced LM-head call plus its gather/scatter and also lies inside engine.step. These host timers include submissions/possible waits; their sums are never added to engine or CUDA time.", ""]
    for record in report["comparisons"]:
        lines.append(f"- `{record['left']}` / `{record['right']}`: drained ratio {fmt(record['drained_ratios']['time'])}, actual output-rate ratio {fmt(record['drained_ratios']['actual_output_rate'])}; identical complete outputs {record['identical_full_sequences']}/16, same extracted answers {record['same_answers']}/16.")
    issues = list(report["errors"])
    if report["group_status"].get("error"):
        issues.append("Group run error: " + str(report["group_status"]["error"]))
    for name, cell in report["cells"].items():
        issues.extend(f"{name}: {message}" for message in cell["errors"] + cell.get("suffix", {}).get("errors", []))
        if cell.get("run_error"):
            issues.append(f"{name}: run error: {cell['run_error']}")
        if cell.get("suffix", {}).get("local_prediction", {}).get("all_current_calls_match") is False:
            issues.append(f"{name}: one or more decision-layer after-load checks mismatch; inspect JSON call records.")
        compact = cell.get("suffix", {}).get("compaction", {})
        issues.extend(f"{name}: {message}" for message in compact.get("errors", []))
        if compact.get("trace_status") == "MISSING":
            issues.append(f"{name}: requested compaction trace is missing; execution-shape results unavailable.")
        for check, counts in compact.get("checks", {}).items():
            if counts["failed"]:
                issues.append(f"{name}: compaction {check} failed in {counts['failed']} recorded steps.")
    if issues:
        lines += ["", "Retained incomplete observations / issues:", ""]
        lines.extend("- " + message.replace("\n", " ") for message in issues)
    lines += ["", "Output lengths and contents may differ. Multi-token chunks do not resolve token ITL; scores on this small task do not establish quality equivalence.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, default=ROOT / "inputs/olmoe_gsm8k_natural16")
    parser.add_argument("--metadata", type=Path, default=ROOT.parent / "C_research_artifacts/20261001/20261001_c_instruct_model_metadata_v1")
    parser.add_argument("--budget", type=int, default=2048)
    parser.add_argument("--cells", help="Optional comma-separated additional cell names, including missing attempts")
    parser.add_argument("--baseline", help="Complete cell for comparisons; defaults to the first complete cell")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args()
    from tokenizers import Tokenizer
    tokenizer_path = args.metadata / "tokenizer.json"
    expected = next(row["sha256"] for row in load(args.metadata / "metadata-receipt.json")["files"] if row["filename"] == "tokenizer.json")
    if sha(tokenizer_path) != expected:
        raise ValueError("tokenizer differs from pinned receipt")
    workload_path = args.inputs / "workload.json"
    report = analyze_group(args.results, load(workload_path), Tokenizer.from_file(str(tokenizer_path)),
        budget=args.budget, requested=args.cells.split(",") if args.cells else None, baseline=args.baseline)
    report.update(workload_sha256=sha(workload_path), tokenizer_sha256=expected)
    output = args.output or args.results / "suffix_metrics.json"
    markdown = args.markdown or args.results / "suffix_summary.md"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    markdown.write_text(markdown_report(report), encoding="utf-8")
    print(json.dumps(dict(json=str(output), markdown=str(markdown),
        cells={name: cell["analysis_status"] for name, cell in report["cells"].items()})))


if __name__ == "__main__":
    main()
