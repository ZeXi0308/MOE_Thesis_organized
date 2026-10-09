"""Host-return request timing; native async observation is explicitly opt-in."""
import hashlib
import json
import math
import time


def measure_episode(engine, workload, config, regime, arrival_scale, run_id, max_seconds=120,
                    *, record_preemptions=False, record_gc=False, allow_async=False):
    """Keep policy hooks and host-return times; optionally observe preemption and GC."""
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
    stop_strings = config.get("stop_strings", [])
    if not isinstance(stop_strings, list) or any(not isinstance(s, str) or not s for s in stop_strings):
        raise ValueError("stop_strings must be a list of nonempty strings")
    if stop_strings and ignore_eos:
        raise ValueError("This task-specific stop profile also requires natural EOS")
    if type(record_preemptions) is not bool:
        raise ValueError("record_preemptions must be boolean")
    if type(record_gc) is not bool:
        raise ValueError("record_gc must be boolean")
    if type(allow_async) is not bool:
        raise ValueError("allow_async must be boolean")
    minima = {rid: config.get("min_tokens", count if ignore_eos else 0) for rid, count in limits.items()}
    if any(type(n) is not int or not 0 <= n <= limits[rid] for rid, n in minima.items()):
        raise ValueError("min_tokens must be an integer between zero and the output limit")
    rows = {rid: dict(request_id=rid, document_id=source["document_id"],
        arrival_s=arrival * arrival_scale, admission_s=None, engine_add_return_s=None,
        completion_s=None, prompt_tokens=len(tokens), max_output_tokens=limits[rid],
        prompt_token_ids_sha256=hashlib.sha256(json.dumps(tokens, separators=(",", ":")).encode()).hexdigest(),
        output_token_ids=[], token_times_s=[], status="unfinished")
        for rid, source, tokens, arrival in zip(ids, sources, prompts, arrivals)}
    if stop_strings:
        for row in rows.values():
            row.update(first_visible_text_s=None, output_text=None, native_stop_reason=None)
    pending = sorted(range(len(ids)), key=lambda i: (rows[ids[i]]["arrival_s"], i))
    internal_to_source, external_to_source, events = {}, {}, []
    call_count = returned_count = next_request = 0
    preemption_events = []
    scheduler = original_preempt = None
    had_instance_preempt = False
    error = None
    gc_observer = None
    backend_core = final_backend_state = async_scheduling = None
    backend_observation_error = frontend_complete_s = None
    completed_count = tail_step_calls = empty_output_step_calls = 0
    if record_gc:
        from gc_observer import GCObserver
        gc_observer = GCObserver()
        gc_observer.install()
    origin, epoch_origin = time.perf_counter(), time.time()
    now = lambda: time.perf_counter() - origin

    def observe_backend():
        # InprocClient owns EngineCore, not EngineCoreProc. In particular, do not
        # call has_work(), and do not drain/free/flush anything from this observer.
        core_scheduler = backend_core.scheduler
        queue = backend_core.batch_queue
        connector = core_scheduler.connector
        state = dict(batch_queue_depth=0 if queue is None else len(queue),
            scheduler_has_requests=bool(core_scheduler.has_requests()),
            deferred_free_entries=len(core_scheduler.deferred_frees),
            connector_pending_push_work=bool(connector.has_pending_push_work())
                if connector is not None else False,
            observed_s=now())
        state["drained"] = not (state["batch_queue_depth"] or state["scheduler_has_requests"]
            or state["deferred_free_entries"] or state["connector_pending_push_work"])
        return state

    def work_remaining():
        nonlocal final_backend_state
        if allow_async:
            final_backend_state = observe_backend()
        return (next_request < len(pending) or engine.has_unfinished_requests()
            or (allow_async and not final_backend_state["drained"]))

    try:
        from vllm import SamplingParams
        from vllm.sampling_params import RequestOutputKind
        scheduling = engine.vllm_config.scheduler_config
        if (not allow_async and scheduling.async_scheduling is not False) or scheduling.stream_interval != 1:
            raise ValueError("measurement requires async_scheduling=False and stream_interval=1")
        if allow_async:
            if type(scheduling.async_scheduling) is not bool:
                raise ValueError("explicit async observation requires resolved boolean async_scheduling")
            async_scheduling = scheduling.async_scheduling
            backend_core = engine.engine_core.engine_core
        if record_preemptions:
            scheduler = engine.engine_core.engine_core.scheduler
            original_preempt = scheduler._preempt_request
            had_instance_preempt = "_preempt_request" in vars(scheduler)

            def preempt(request, *args, **kwargs):
                rid = internal_to_source[request.request_id]
                row = rows[rid]
                event = dict(request_id=rid, internal_request_id=request.request_id,
                    engine_call_index=call_count - 1,
                    last_returned_output_count=len(row["output_token_ids"]),
                    last_new_output_s=row["token_times_s"][-1] if row["token_times_s"] else None,
                    native_output_count_before=request.num_output_tokens,
                    method_entered_s=now(), method_returned_s=None,
                    original_preemption_called=True, original_preemption_returned=False)
                preemption_events.append(event)
                try:
                    value = original_preempt(request, *args, **kwargs)
                except BaseException as exc:
                    event.update(method_failed_s=now(), error=f"{type(exc).__name__}: {exc}")
                    raise
                event.update(method_returned_s=now(), original_preemption_returned=True)
                return value

            scheduler._preempt_request = preempt
        while work_remaining():
            if now() >= max_seconds:
                error = "runtime_limit"
                break
            while next_request < len(pending):
                index = pending[next_request]
                rid, row = ids[index], rows[ids[index]]
                if row["arrival_s"] > now():
                    break
                params = SamplingParams(n=1, temperature=0.0, max_tokens=limits[rid],
                    min_tokens=minima[rid], ignore_eos=ignore_eos, detokenize=bool(stop_strings),
                    **(dict(stop=stop_strings, include_stop_str_in_output=False) if stop_strings else {}),
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
                if allow_async:
                    final_backend_state = observe_backend()
                if not allow_async or final_backend_state["drained"]:
                    if next_request < len(pending):
                        delay = rows[ids[pending[next_request]]]["arrival_s"] - now()
                        time.sleep(max(0.0, min(0.001, delay)))
                    continue
            if allow_async and frontend_complete_s is not None:
                tail_step_calls += 1
            call_count += 1
            if gc_observer is not None:
                gc_observer.phase = "engine_step"
            step_started = now() if allow_async else None
            outputs = engine.step()
            received = now()
            if gc_observer is not None:
                gc_observer.phase = "output_recording"
            returned_count += 1
            if allow_async and not outputs:
                empty_output_step_calls += 1
            for output in outputs:
                rid = external_to_source[output.request_id]
                row = rows[rid]
                if len(output.outputs) != 1 or row["status"] == "completed":
                    raise ValueError("unexpected completion count or output after completion")
                completion = output.outputs[0]
                if stop_strings and completion.text and row["first_visible_text_s"] is None:
                    row["first_visible_text_s"] = received
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
                    if allow_async:
                        completed_count += 1
                        if completed_count == len(ids):
                            frontend_complete_s = received
                    if stop_strings:
                        row.update(output_text=completion.text, native_stop_reason=completion.stop_reason)
            if gc_observer is not None:
                gc_observer.phase = "ingress"
            if (allow_async and frontend_complete_s is not None and not outputs
                    and not final_backend_state["batch_queue_depth"]
                    and not final_backend_state["scheduler_has_requests"]):
                # A deferred-free-only tail may have no executable native work.
                # Keep observing to the common deadline without a busy spin.
                time.sleep(max(0.0, min(0.001 - (received - step_started), max_seconds - now())))
        if error is None and any(row["status"] != "completed" for row in rows.values()):
            error = "engine drained before every planned request completed"
    except Exception as exc:
        if gc_observer is not None:
            gc_observer.phase = "ingress"
        error = f"{type(exc).__name__}: {exc}"
        for row in rows.values():
            if row["status"] != "completed" and row["admission_s"] is not None:
                row.update(status="failed", error=error)
    finally:
        try:
            if allow_async and backend_core is not None:
                try:
                    final_backend_state = observe_backend()
                except Exception as exc:
                    final_backend_state = None
                    backend_observation_error = f"{type(exc).__name__}: {exc}"
                    if error is None:
                        error = backend_observation_error
            if scheduler is not None and original_preempt is not None:
                if had_instance_preempt:
                    scheduler._preempt_request = original_preempt
                elif "_preempt_request" in vars(scheduler):
                    del scheduler._preempt_request
        finally:
            if gc_observer is not None:
                gc_observer.close()
    end = now()
    if allow_async and error is None and end >= max_seconds:
        # A blocking native step is not cancellable here. Preserve its outputs
        # and actual elapsed time, but never label an over-deadline run COMPLETE.
        error = "runtime_limit"
    for row in rows.values():
        row["arrived_at_observation_end"] = row["arrival_s"] <= end
    sizes = [event["chunk_size"] for event in events if event["prefix_valid"]]
    result = dict(status="COMPLETE" if error is None else "INCOMPLETE", error=error,
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
    if allow_async:
        result["timing_semantics"] = ("Host receipt immediately after native engine.step; all calls, "
            "including empty async returns and backend drain, share the external-arrival observation window. "
            "Host call indices are not GPU execution or scheduler-round indices.")
        result["async_observation"] = dict(async_scheduling=async_scheduling,
            frontend_complete_s=frontend_complete_s,
            backend_drained=final_backend_state["drained"] if final_backend_state is not None else None,
            final_backend_state=final_backend_state, backend_observation_error=backend_observation_error,
            tail_step_calls=tail_step_calls,
            tail_elapsed_s=end - frontend_complete_s if frontend_complete_s is not None else None,
            empty_output_step_calls=empty_output_step_calls,
            runtime_limit_s=max_seconds, runtime_overrun_s=max(0.0, end - max_seconds),
            backend_semantics="Read-only EngineCore batch_queue, scheduler.has_requests(), deferred_frees "
                "and connector.has_pending_push_work(); natural engine.step performs all drain work. "
                "No EngineCoreProc.has_work or manual free/flush is used. Missing state is an error, not idle.",
            progress_semantics="Empty engine.step returns are not output/token progress. Cumulative tokens "
                "share their host receipt timestamp; intra-chunk ITL and GPU token-completion times are unresolved. "
                "Frontend completion and backend drain are separate; neither tail nor observer cost is subtracted.")
    if stop_strings:
        result['task_termination'] = dict(stop_strings=stop_strings, include_stop_str_in_output=False,
            detokenize=True, natural_eos=True, final_text_recorded=True,
            visible_text_timing='First nonempty cumulative detokenized text returned by engine.step; '
                'separate from token TTFT because native stop buffering may delay visible characters.')
    if record_preemptions:
        result.update(preemption_events=preemption_events,
            actual_preemption_count=sum(e["original_preemption_returned"] for e in preemption_events),
            preemption_attempt_count=len(preemption_events),
            policy_hooks_modified=True,
            measurement_hooks_modified=["_preempt_request"],
            diagnostics="SPARSE_PREEMPTION_EVENTS",
            scheduler_diagnostics="SPARSE_PREEMPTION_EVENTS",
            diagnostics_scope="Only the existing _preempt_request method is chained; no schedule/allocation/KV snapshots. Caller policy hooks are preserved.",
            preemption_event_semantics="Method entry/return are host boundaries, not DMA/GPU completion. Last returned output uses measurement history, not native internal progress. A successful preempt does not imply its engine call completed. Both forced and native calls are recorded; cause is not inferred.")
    if gc_observer is not None:
        result["gc_observation"] = gc_observer.report(origin)
        if record_preemptions:
            result["diagnostics_scope"] += " Passive GC callbacks are also recorded without changing GC policy."
        else:
            result.update(diagnostics="PASSIVE_GC_EVENTS",
                diagnostics_scope="Only passive GC callbacks are recorded, without changing GC policy. "
                    "No scheduler/allocation/KV snapshots; caller policy hooks are preserved.")
    return result
