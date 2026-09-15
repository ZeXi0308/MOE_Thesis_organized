"""Synchronous native request measurement without scheduler or memory observers."""
import hashlib
import json
import math
import time


def measure_episode(engine, workload, config, regime, arrival_scale, run_id, max_seconds=120):
    """Use a diagnostic-free engine; keep its policy hooks and host-return timestamps."""
    sources = workload["source_requests"]
    prompts = workload["actual_prompt_token_ids"]
    arrivals = workload["arrival_traces_s"][regime]
    if not sources or len(sources) != len(prompts) or len(sources) != len(arrivals):
        raise ValueError("source, prompt and arrival identities must align")
    finite = lambda v: type(v) in (int, float) and math.isfinite(v)
    if not finite(arrival_scale) or arrival_scale < 0 or not finite(max_seconds) or max_seconds <= 0:
        raise ValueError("arrival scale and runtime limit must be finite and nonnegative/positive")
    if any(not finite(v) or v < 0 or not math.isfinite(v * arrival_scale) for v in arrivals):
        raise ValueError("scaled arrivals must be finite and nonnegative")
    ids = [source["request_id"] for source in sources]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate source request IDs")
    overrides = config.get("output_tokens_by_request", {})
    if set(overrides) - set(ids):
        raise ValueError("output override refers to an unknown request")
    limits = {rid: overrides.get(rid, config["output_tokens"]) for rid in ids}
    if any(type(n) is not int or n < 1 for n in limits.values()):
        raise ValueError("output limits must be positive integers")
    ignore_eos = config.get("ignore_eos", True)
    if type(ignore_eos) is not bool:
        raise ValueError("ignore_eos must be boolean")
    minima = {rid: config.get("min_tokens", count if ignore_eos else 0) for rid, count in limits.items()}
    if any(type(n) is not int or not 0 <= n <= limits[rid] for rid, n in minima.items()):
        raise ValueError("min_tokens must be an integer between zero and the output limit")
    rows = {rid: dict(request_id=rid, document_id=source["document_id"],
        arrival_s=arrival * arrival_scale, admission_s=None, engine_add_return_s=None,
        completion_s=None, prompt_tokens=len(tokens), max_output_tokens=limits[rid],
        prompt_token_ids_sha256=hashlib.sha256(json.dumps(tokens, separators=(",", ":")).encode()).hexdigest(),
        output_token_ids=[], token_times_s=[], status="unfinished")
        for rid, source, tokens, arrival in zip(ids, sources, prompts, arrivals)}
    pending = sorted(range(len(ids)), key=lambda i: (rows[ids[i]]["arrival_s"], i))
    internal_to_source, external_to_source, events = {}, {}, []
    call_count = returned_count = next_request = 0
    error = None
    origin, epoch_origin = time.perf_counter(), time.time()
    now = lambda: time.perf_counter() - origin
    try:
        from vllm import SamplingParams
        from vllm.sampling_params import RequestOutputKind
        scheduling = engine.vllm_config.scheduler_config
        if scheduling.async_scheduling is not False or scheduling.stream_interval != 1:
            raise ValueError("measurement requires async_scheduling=False and stream_interval=1")
        while next_request < len(pending) or engine.has_unfinished_requests():
            if now() >= max_seconds:
                error = "runtime_limit"
                break
            while next_request < len(pending):
                index = pending[next_request]
                rid, row = ids[index], rows[ids[index]]
                if row["arrival_s"] > now():
                    break
                params = SamplingParams(n=1, temperature=0.0, max_tokens=limits[rid],
                    min_tokens=minima[rid], ignore_eos=ignore_eos, detokenize=False,
                    output_kind=RequestOutputKind.CUMULATIVE)
                external_id = f"{run_id}/{rid}"
                row.update(external_request_id=external_id, admission_s=now())
                internal_id = engine.add_request(external_id, {"prompt_token_ids": prompts[index]},
                    params, arrival_time=epoch_origin + row["arrival_s"])
                row["engine_add_return_s"] = now()
                if internal_id in internal_to_source or external_id in external_to_source:
                    raise ValueError("native request ID collision")
                internal_to_source[internal_id] = rid
                external_to_source[external_id] = rid
                row["internal_request_id"] = internal_id
                next_request += 1
            if not engine.has_unfinished_requests():
                if next_request < len(pending):
                    delay = rows[ids[pending[next_request]]]["arrival_s"] - now()
                    time.sleep(max(0.0, min(0.001, delay)))
                continue
            call_count += 1
            outputs = engine.step()
            received = now()
            returned_count += 1
            for output in outputs:
                rid = external_to_source[output.request_id]
                row = rows[rid]
                if len(output.outputs) != 1 or row["status"] == "completed":
                    raise ValueError("unexpected completion count or output after completion")
                completion = output.outputs[0]
                tokens, previous = list(completion.token_ids), row["output_token_ids"]
                event = dict(request_id=rid, external_request_id=output.request_id,
                    engine_call_index=call_count - 1, received_s=received, cumulative_tokens=len(tokens),
                    new_token_ids=[], chunk_size=0, finished=bool(output.finished),
                    finish_reason=completion.finish_reason, prefix_valid=False)
                events.append(event)
                if tokens[:len(previous)] != previous or not len(previous) <= len(tokens) <= limits[rid]:
                    raise ValueError("cumulative token prefix changed or output limit exceeded")
                added = tokens[len(previous):]
                event.update(new_token_ids=added, chunk_size=len(added), prefix_valid=True)
                row["output_token_ids"] = tokens
                row["token_times_s"].extend([received] * len(added))
                if output.finished:
                    valid = len(tokens) == limits[rid] and completion.finish_reason == "length"
                    if not ignore_eos:
                        valid |= minima[rid] <= len(tokens) <= limits[rid] and completion.finish_reason == "stop"
                    if not valid:
                        raise ValueError("native completion violated configured output limits")
                    row.update(status="completed", completion_s=received, stop_reason=completion.finish_reason)
        if error is None and any(row["status"] != "completed" for row in rows.values()):
            error = "engine drained before every planned request completed"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        for row in rows.values():
            if row["status"] != "completed" and row["admission_s"] is not None:
                row.update(status="failed", error=error)
    end = now()
    for row in rows.values():
        row["arrived_at_observation_end"] = row["arrival_s"] <= end
    sizes = [event["chunk_size"] for event in events if event["prefix_valid"]]
    return dict(status="COMPLETE" if error is None else "INCOMPLETE", error=error,
        requests=list(rows.values()), observation_end_s=end, output_events=events,
        engine_call_count=call_count, engine_return_count=returned_count,
        internal_to_source=internal_to_source, measurement_origin_perf_counter_s=origin,
        measurement_origin_unix_s=epoch_origin, regime=regime, arrival_scale=arrival_scale,
        diagnostics="DIAGNOSTICS_DISABLED", scheduler_diagnostics="DIAGNOSTICS_DISABLED",
        memory_diagnostics="DIAGNOSTICS_DISABLED", actual_preemption_count=None,
        diagnostics_scope="This function installs no observers; caller must supply an engine without diagnostic wrappers.",
        recomputed_tokens=None, recovery_count=None, policy_hooks_modified=False,
        evidence_type="NATIVE_SERVING_INPROCESS_HOST_MEASUREMENT",
        queue_semantics="admission_s minus arrival_s is client submission lag; native waiting is not measured separately.",
        timing_semantics="Host receipt immediately after synchronous engine.step; all policy execution costs remain included.",
        host_chunk_diagnostics=dict(multi_token_chunks=sum(n > 1 for n in sizes),
            max_chunk_size=max(sizes, default=0), token_level_itl_resolved=all(n <= 1 for n in sizes),
            interpolated=False, semantics="Tokens returned together share one timestamp; intra-chunk ITL is unresolved."))
