"""Observe local decode-row readiness without changing expert execution.

CUDA events use the caller's current stream. Their intervals include any idle
time on that stream: they are not pure kernel or copy times. A ready row is not
a request completion, and overlapping windows must not be summed as savings.
"""
import json
from pathlib import Path


def ready_group(topk, executed_groups):
    """Zero-based first group covering ALL required experts, otherwise None."""
    required, done = set(topk), set()
    if not required:
        return None
    for index, experts in enumerate(executed_groups):
        done.update(experts)
        if required <= done:
            return index
    return None


def _state(runtime):
    state = getattr(runtime, "_request_readiness_observer", None)
    if state is None:
        state = dict(calls=[], active={}, finalized=False)
        runtime._request_readiness_observer = state
    return state


def _event(runtime):
    event = runtime.torch.cuda.Event(enable_timing=True)
    event.record()
    return event


def start_call(runtime, record):
    """Call after route/plan creation, immediately before the group loop."""
    if not record.get("measurement") or record.get("validation_run"):
        return
    state = _state(runtime)
    if state["finalized"]:
        raise RuntimeError("readiness observer already finalized")
    if id(record) in state["active"]:
        raise RuntimeError("duplicate readiness start")
    context = record.get("context") or {}
    metadata, routes = context.get("rows"), record.get("row_topk_experts")
    mapped = (context.get("row_request_order_verified") is True
              and context.get("valid_row_start") == 0
              and isinstance(metadata, list) and isinstance(routes, list)
              and context.get("valid_row_stop") == len(metadata)
              and len(metadata) <= len(routes))
    call = dict(call_id=record.get("call_id"), step_id=context.get("step_id"),
                phase=context.get("phase"), layer_name=record.get("layer_name"),
                mapping_verified=mapped, decode_rows=[], excluded_rows=0,
                excluded_reason=None if mapped else "unknown_or_invalid_row_mapping",
                entry_resident_experts=list(record.get("entry_resident_experts", [])),
                groups=[], group_events=[], start_event=None, finish_event=None)
    if mapped:
        for index, row in enumerate(metadata):
            valid = (isinstance(row, dict)
                     and isinstance(row.get("internal_request_id"), str)
                     and bool(row["internal_request_id"])
                     and type(row.get("computed_position")) is int
                     and type(row.get("prompt_tokens")) is int
                     and row["computed_position"] >= 0 and row["prompt_tokens"] > 0
                     and isinstance(routes[index], list) and bool(routes[index])
                     and all(type(e) is int and e >= 0 for e in routes[index]))
            if not valid:
                call["excluded_rows"] += 1
            elif row["computed_position"] >= row["prompt_tokens"]:
                call["decode_rows"].append(dict(row_index=index,
                    internal_request_id=row["internal_request_id"],
                    computed_position=row["computed_position"],
                    prompt_tokens=row["prompt_tokens"], topk=list(routes[index])))
    else:
        call["excluded_rows"] = len(routes) if isinstance(routes, list) else 0
    # Prefill-only or unmapped calls need no device events.
    if call["decode_rows"]:
        call["start_event"] = _event(runtime)
    state["calls"].append(call)
    state["active"][id(record)] = call


def group_done(runtime, record, group):
    """Call after the group's kernel and partial-result accumulation."""
    state = getattr(runtime, "_request_readiness_observer", None)
    call = state and state["active"].get(id(record))
    if call is None or not call["decode_rows"]:
        return
    # Record first; CPU metadata preparation must not delay this boundary.
    call["group_events"].append(_event(runtime))
    call["groups"].append(dict(
        required_experts=list(group["required_experts"]),
        loaded_experts=list(group.get("loaded_experts", [])),
        weight_copy_bytes=group.get("weight_copy_bytes")))


def finish_call(runtime, record):
    """Call once at the layer's normal return boundary, after all groups."""
    state = getattr(runtime, "_request_readiness_observer", None)
    call = state and state["active"].pop(id(record), None)
    if call is not None:
        call["finished"] = True
        if call["decode_rows"]:
            call["finish_event"] = _event(runtime)


def finalize(runtime, outputpath):
    """After episode drain, synchronize once, resolve events, write new JSON."""
    state = _state(runtime)
    if state["finalized"]:
        raise RuntimeError("readiness observer already finalized")
    runtime.torch.cuda.synchronize()
    records = []
    totals = dict(calls=len(state["calls"]), completed_calls=0,
                  mapped_decode_rows=0, early_ready_rows=0, never_ready_rows=0,
                  entry_resident_rows_in_mixed_cold_first_group=0,
                  maximum_local_window_ms=0.0)
    for call in state["calls"]:
        out = {k: v for k, v in call.items() if not k.endswith("_event")
               and k != "group_events"}
        complete = call.get("finished", False)
        out["completed"] = complete
        totals["completed_calls"] += int(complete)
        groups = call["groups"]
        if not complete or not call["decode_rows"]:
            if not complete:
                out["excluded_reason"] = "call_did_not_reach_normal_return"
            records.append(out)
            continue
        start, finish = call["start_event"], call["finish_event"]
        ends = [float(start.elapsed_time(e)) for e in call["group_events"]]
        total_ms = float(start.elapsed_time(finish))
        out["layer_observed_span_ms"] = total_ms
        for index, group in enumerate(groups):
            group["completion_from_start_ms"] = ends[index]
            group["stream_interval_ms"] = ends[index] - (ends[index - 1] if index else 0.0)
        resident = set(call["entry_resident_experts"])
        mixed_cold = bool(groups and groups[0]["loaded_experts"]
                          and resident.intersection(groups[0]["required_experts"]))
        out["first_group_mixed_resident_and_cold"] = mixed_cold
        for row in out["decode_rows"]:
            index = ready_group(row["topk"], [g["required_experts"] for g in groups])
            row["ready_group_index"] = index
            row["all_topk_entry_resident"] = set(row["topk"]) <= resident
            row["entry_resident_but_first_group_cold"] = (
                row["all_topk_entry_resident"] and mixed_cold)
            row["local_ready_to_layer_end_ms"] = (
                max(0.0, total_ms - ends[index]) if index is not None else None)
            row["early_ready_before_last_group"] = index is not None and index < len(groups) - 1
            totals["mapped_decode_rows"] += 1
            totals["early_ready_rows"] += int(row["early_ready_before_last_group"])
            totals["never_ready_rows"] += int(index is None)
            totals["entry_resident_rows_in_mixed_cold_first_group"] += int(
                row["entry_resident_but_first_group_cold"])
            if index is not None:
                totals["maximum_local_window_ms"] = max(totals["maximum_local_window_ms"],
                    row["local_ready_to_layer_end_ms"])
        records.append(out)
    result = dict(schema_version=1, evidence_type="LOCAL_LAYER_OPPORTUNITY_ONLY",
        scope="Measurement non-validation decode rows; original execution unchanged.",
        limitations=["Ready means all top-k expert contributions accumulated, not request completion.",
            "Windows overlap and cannot be added into request savings or speedup.",
            "CUDA stream intervals include stream idle time, not pure kernel/copy time.",
            "First-group cold co-membership is structural; its avoided delay is unmeasured.",
            "Instrumentation costs remain; unknown row mapping and incomplete calls excluded.",
            "internal_request_id joins the episode raw internal_to_source mapping."],
        totals=totals, calls=records)
    path = Path(outputpath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")
    state["finalized"] = True
    state["calls"].clear()
    state["active"].clear()
    return result


if __name__ == "__main__":
    # One partial top-k contribution must never count as a ready row.
    assert ready_group([1, 3], [[1], [2]]) is None
    assert ready_group([1, 3], [[1], [2], [3]]) == 2
    assert ready_group([1, 3], [[1, 3], [2]]) == 0
    print("PASS: all-top-k completion reducer counterexample")
