"""Synchronous request timing with optional sparse preemption events."""
import hashlib
import json
import math
import sys
import time


def _pending_store_job_ids(scheduler, request_id):
    """Scheduler confirmations outstanding, not a query of DMA completion."""
    try:
        cs = scheduler.connector.connector_scheduler
        ids = sorted(cs._req_status[request_id].transfer_jobs)
        for job_id in ids:
            job = cs._jobs[job_id]
            if (type(job_id) is not int or job.req_id != request_id
                    or job.is_store is not True or type(job.pending_count) is not int
                    or job.pending_count <= 0):
                raise ValueError('Unqualified pending STORE job')
        return dict(status='KNOWN', job_ids=ids)
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        return dict(status='UNKNOWN', job_ids=None, error=f'{type(exc).__name__}: {exc}')


def _install_flush_wait_observer(now, current_step):
    """Time existing calls only; never query events or add synchronization."""
    data = dict(status='UNKNOWN', installed=False, methods_restored=True, events=[], errors=[],
        observer_overhead_s=None,
        semantics='Host wall time of actual native flush calls. STORE jobs have predecessor dependencies; '
            'a wait interval is not per-job DMA time or removable victim cost. Empty flushes are not logged. '
            'Observer overhead is not isolated and can perturb exposed wait; this is characterization, not a policy comparison.')
    saved, active = [], [None]

    def unknown(where, error):
        data['status'] = 'UNKNOWN'
        data['errors'].append(dict(where=where, error=str(error)))

    def restore():
        for obj, name, had_instance, value in reversed(saved):
            if had_instance:
                setattr(obj, name, value)
            elif name in vars(obj):
                delattr(obj, name)
        saved.clear()
        data['methods_restored'] = True
        if data['status'] == 'INSTALLED':
            data['status'] = 'INSTALLED_AND_RESTORED'

    try:
        module = sys.modules.get('vllm.distributed.kv_transfer.kv_transfer_state')
        agent = getattr(module, '_KV_CONNECTOR_AGENT', None)
        wrapper = getattr(agent, 'connector_worker', None)
        worker = getattr(wrapper, 'worker', None)
        if (type(agent).__name__ != 'OffloadingConnector'
                or type(wrapper).__name__ != 'OffloadingConnectorWorker'
                or type(worker).__name__ != 'CPUOffloadingWorker'):
            raise ValueError('Existing in-process OffloadingConnector/CPUOffloadingWorker unavailable')
        original_handle, original_wait = wrapper.handle_preemptions, worker.wait
        if not callable(original_handle) or not callable(original_wait):
            raise ValueError('Existing handle_preemptions/wait is not callable')

        def handle(metadata, *args, **kwargs):
            try:
                ids = sorted(metadata.jobs_to_flush)
                if any(type(jid) is not int for jid in ids):
                    raise ValueError('Unknown flush job ID type')
            except Exception as exc:
                unknown('flush_context', exc)
                return original_handle(metadata, *args, **kwargs)
            if not ids:
                return original_handle(metadata, *args, **kwargs)
            row = dict(engine_call_index=current_step(), job_ids=ids, wait_calls=[],
                handle_entered_s=None, handle_returned_s=None, handle_completed=False)
            data['events'].append(row)
            previous, active[0] = active[0], row
            row['handle_entered_s'] = now()
            try:
                value = original_handle(metadata, *args, **kwargs)
                row['handle_completed'] = True
                return value
            except BaseException as exc:
                row['handle_error'] = f'{type(exc).__name__}: {exc}'
                raise
            finally:
                row['handle_returned_s'] = now()
                row['handle_wall_s'] = row['handle_returned_s'] - row['handle_entered_s']
                active[0] = previous
                if row['handle_completed'] and len(row['wait_calls']) != 1:
                    unknown('wait_call_count', len(row['wait_calls']))

        def wait(job_ids, *args, **kwargs):
            row = active[0]
            if row is None:
                unknown('wait_context', 'Native wait called outside a nonempty flush context')
                return original_wait(job_ids, *args, **kwargs)
            try:
                ids = sorted(job_ids)
            except Exception as exc:
                unknown('wait_job_ids', exc)
                return original_wait(job_ids, *args, **kwargs)
            if ids != row['job_ids']:
                unknown('wait_job_ids', 'Wait IDs differ from the actual flush context')
            call = dict(job_ids=ids, entered_s=None, returned_s=None, completed=False)
            row['wait_calls'].append(call)
            call['entered_s'] = now()
            try:
                value = original_wait(job_ids, *args, **kwargs)
            except BaseException as exc:
                call['error'] = f'{type(exc).__name__}: {exc}'
                raise
            else:
                call['completed'] = True
                return value
            finally:
                call['returned_s'] = now()
                call['wall_s'] = call['returned_s'] - call['entered_s']

        for obj, name, replacement in ((wrapper, 'handle_preemptions', handle), (worker, 'wait', wait)):
            had_instance = name in vars(obj)
            old_value = vars(obj).get(name)
            saved.append((obj, name, had_instance, old_value))
            data['methods_restored'] = False
            setattr(obj, name, replacement)
        data.update(status='INSTALLED', installed=True)
    except Exception as exc:
        unknown('installation', f'{type(exc).__name__}: {exc}')
        restore()
    return data, restore


def measure_episode(engine, workload, config, regime, arrival_scale, run_id, max_seconds=120,
                    *, record_preemptions=False, record_flush_wait=False):
    """Keep policy hooks and host-return times; optionally observe only preempt calls."""
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
    if type(record_preemptions) is not bool:
        raise ValueError("record_preemptions must be boolean")
    if type(record_flush_wait) is not bool:
        raise ValueError("record_flush_wait must be boolean")
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
    preemption_events = []
    scheduler = original_preempt = None
    had_instance_preempt = False
    flush_wait = dict(status='DISABLED', installed=False, methods_restored=True, events=[], errors=[])
    uninstall_flush_wait = lambda: None
    error = None
    origin, epoch_origin = time.perf_counter(), time.time()
    now = lambda: time.perf_counter() - origin
    try:
        from vllm import SamplingParams
        from vllm.sampling_params import RequestOutputKind
        scheduling = engine.vllm_config.scheduler_config
        if scheduling.async_scheduling is not False or scheduling.stream_interval != 1:
            raise ValueError("measurement requires async_scheduling=False and stream_interval=1")
        if record_flush_wait:
            flush_wait, uninstall_flush_wait = _install_flush_wait_observer(now, lambda: call_count - 1)
        if record_preemptions:
            scheduler = engine.engine_core.engine_core.scheduler
            pool = scheduler.kv_cache_manager.block_pool
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
                    free_blocks_before_preempt=pool.get_num_free_blocks(),
                    method_entered_s=now(), method_returned_s=None,
                    original_preemption_called=True, original_preemption_returned=False)
                if record_flush_wait:
                    event['pending_native_store_jobs'] = _pending_store_job_ids(scheduler, request.request_id)
                preemption_events.append(event)
                try:
                    value = original_preempt(request, *args, **kwargs)
                except BaseException as exc:
                    event.update(method_failed_s=now(), error=f"{type(exc).__name__}: {exc}")
                    raise
                finally:
                    free_after=pool.get_num_free_blocks()
                    event.update(free_blocks_after_preempt=free_after,
                        actual_released_blocks=free_after-event['free_blocks_before_preempt'],
                        release_measurement_scope='Global free-block delta across native _preempt_request only; excludes later allocation retries.')
                event.update(method_returned_s=now(), original_preemption_returned=True)
                return value

            scheduler._preempt_request = preempt
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
    finally:
        try:
            uninstall_flush_wait()
        finally:
            if scheduler is not None and original_preempt is not None:
                if had_instance_preempt:
                    scheduler._preempt_request = original_preempt
                elif "_preempt_request" in vars(scheduler):
                    del scheduler._preempt_request
    end = now()
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
    if record_preemptions:
        result.update(preemption_events=preemption_events,
            actual_preemption_count=sum(e["original_preemption_returned"] for e in preemption_events),
            preemption_attempt_count=len(preemption_events),
            policy_hooks_modified=True,
            measurement_hooks_modified=["_preempt_request"],
            diagnostics="SPARSE_PREEMPTION_EVENTS",
            scheduler_diagnostics="SPARSE_PREEMPTION_EVENTS",
            diagnostics_scope="Only the existing _preempt_request method is chained, with two scalar free-pool readings; no schedule/allocation/per-request KV snapshots. Caller policy hooks are preserved.",
            preemption_event_semantics="Method entry/return are host boundaries, not DMA/GPU completion. Last returned output uses measurement history, not native internal progress. A successful preempt does not imply its engine call completed. Both forced and native calls are recorded; cause is not inferred.")
    if record_flush_wait:
        result['flush_wait_observation'] = flush_wait
        result['diagnostics'] = 'SPARSE_PREEMPTION_AND_NATIVE_FLUSH_WAIT_EVENTS'
        result.setdefault('measurement_hooks_modified', []).extend(
            ['OffloadingConnectorWorker.handle_preemptions', 'CPUOffloadingWorker.wait'] if flush_wait['installed'] else [])
        result['diagnostics_scope'] = ('Existing preemption observer plus read-only timing of nonempty native '
            'flush contexts and their original wait calls. No event queries, added synchronization or transfer changes; '
            'worker methods restored before measurement returns. Missing observer state is UNKNOWN.')
    return result
