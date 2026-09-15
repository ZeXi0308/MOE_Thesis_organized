"""Arrival-driven in-process vLLM capture; timestamps are host receipt times."""
from __future__ import annotations

import hashlib
import json
import math
import time


def set_empty_admission_cap(engine, cap) -> dict:
    """Change only admission on a drained engine; keep its compiled capacity fixed."""
    maximum = engine.vllm_config.scheduler_config.max_num_seqs
    if type(cap) is not int or not 1 <= cap <= maximum:
        raise ValueError("admission cap must be an integer within engine max_num_seqs")
    scheduler = engine.engine_core.engine_core.scheduler
    running, waiting = scheduler.get_request_counts()
    if engine.has_unfinished_requests() or running != 0 or waiting != 0 or scheduler.requests:
        raise ValueError("admission cap can only change on a completely drained engine")
    previous = scheduler.max_num_running_reqs
    scheduler.max_num_running_reqs = cap
    return dict(engine_max_num_seqs=maximum, previous_admission_cap=previous, admission_cap=cap)


def capture_episode(engine, workload, config, *, regime, arrival_scale, run_id, max_seconds=120):
    sources = workload["source_requests"]
    prompts = workload["actual_prompt_token_ids"]
    arrivals = workload["arrival_traces_s"][regime]
    if not sources or len(sources) != len(prompts) or len(sources) != len(arrivals):
        raise ValueError("source, prompt and arrival identities must align")
    if not math.isfinite(arrival_scale) or arrival_scale < 0 or not math.isfinite(max_seconds) or max_seconds <= 0:
        raise ValueError("arrival scale must be nonnegative and runtime limit positive, both finite")
    if any(not math.isfinite(v) or v < 0 for v in arrivals):
        raise ValueError("arrivals must be finite and nonnegative")
    count = config["output_tokens"]
    if type(count) is not int or count < 2 or type(config["cap"]) is not int or config["cap"] < 1:
        raise ValueError("need positive cap and at least two fixed output tokens")
    rows = {source["request_id"]: dict(request_id=source["request_id"], document_id=source["document_id"],
            arrival_s=arrival * arrival_scale, admission_s=None, engine_add_return_s=None, completion_s=None,
            prompt_token_ids_sha256=hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest(),
            prompt_tokens=len(ids), output_token_ids=[], token_times_s=[], status="unfinished")
            for source, ids, arrival in zip(sources, prompts, arrivals)}
    if len(rows) != len(sources):
        raise ValueError("duplicate source request IDs")
    pending = sorted(range(len(sources)), key=lambda i: (arrivals[i] * arrival_scale, i))
    internal_to_source, external_to_source = {}, {}
    scheduler_steps, output_events, engine_steps = [], [], []
    executed_high_water = {}
    scheduler = original = None
    had_instance_schedule = False
    error = None
    origin = time.perf_counter()
    epoch_origin = time.time()
    now = lambda: time.perf_counter() - origin

    try:
        from vllm import SamplingParams
        from vllm.sampling_params import RequestOutputKind
        scheduler = engine.engine_core.engine_core.scheduler
        original = scheduler.schedule
        had_instance_schedule = "schedule" in vars(scheduler)
        if engine.vllm_config.scheduler_config.async_scheduling is not False or engine.vllm_config.scheduler_config.stream_interval != 1:
            raise ValueError("capture requires async_scheduling=False and stream_interval=1")

        def schedule(*args, **kwargs):
            started = now()
            before = {rid: (int(req.num_computed_tokens), int(req.num_prompt_tokens), int(req.num_output_tokens))
                      for rid, req in scheduler.requests.items()}
            running_before, waiting_before = scheduler.get_request_counts()
            result = original(*args, **kwargs)
            scheduled = []
            for rid, amount in result.num_scheduled_tokens.items():
                prior, prompt_length, output_count = before[rid]
                after = int(scheduler.requests[rid].num_computed_tokens)
                # schedule() already increments computed; a reset/cache adjustment
                # inside it is retained explicitly instead of silently misclassifying.
                actual_start = after - amount
                if actual_start < 0 or amount <= 0:
                    raise ValueError("invalid native scheduled/computed token accounting")
                high_water = executed_high_water.get(rid, 0)
                recompute = max(0, min(after, high_water) - actual_start)
                new_start = max(actual_start, high_water)
                prefill = max(0, min(after, prompt_length) - new_start)
                decode = amount - recompute - prefill
                if min(recompute, prefill, decode) < 0:
                    raise ValueError("invalid exclusive new/recomputed token accounting")
                scheduled.append(dict(request_id=internal_to_source[rid], internal_request_id=rid,
                    computed_before=prior, computed_after=after, scheduled_start_computed=actual_start,
                    computed_adjustment=actual_start - prior, prompt_tokens=prompt_length,
                    output_tokens_before=output_count, executed_high_water_before=high_water,
                    scheduled_tokens=amount, prefill_tokens=prefill, decode_tokens=decode,
                    recompute_tokens=recompute))
            running_after, waiting_after = scheduler.get_request_counts()
            scheduler_steps.append(dict(step=len(scheduler_steps), start_s=started, end_s=now(),
                target_cap=config["cap"], running_before=running_before, waiting_before=waiting_before,
                actual_active=running_after, waiting_requests=waiting_after,
                decode_requests=sum(row["decode_tokens"] > 0 for row in scheduled), scheduled=scheduled,
                recompute_tokens=sum(row["recompute_tokens"] for row in scheduled),
                total_scheduled_tokens=sum(row["scheduled_tokens"] for row in scheduled),
                preempted_request_ids=[internal_to_source[rid] for rid in (result.preempted_req_ids or [])]))
            return result

        scheduler.schedule = schedule
        next_request = 0
        while next_request < len(pending) or engine.has_unfinished_requests():
            if now() >= max_seconds:
                error = "runtime_limit"
                break
            # Every due request reaches the native queue; no client inflight cap.
            while next_request < len(pending):
                index = pending[next_request]
                source_id = sources[index]["request_id"]
                row = rows[source_id]
                if row["arrival_s"] > now():
                    break
                external_id = f"{run_id}/{source_id}"
                params = SamplingParams(n=1, temperature=0.0, max_tokens=count, min_tokens=count,
                    ignore_eos=True, detokenize=False, output_kind=RequestOutputKind.CUMULATIVE)
                row["admission_s"] = now()
                internal_id = engine.add_request(external_id, {"prompt_token_ids": prompts[index]}, params,
                                                arrival_time=epoch_origin + row["arrival_s"])
                row["engine_add_return_s"] = now()
                if internal_id in internal_to_source or external_id in external_to_source:
                    raise ValueError("native request ID collision")
                internal_to_source[internal_id] = source_id
                external_to_source[external_id] = source_id
                row.update(external_request_id=external_id, internal_request_id=internal_id)
                next_request += 1
            if not engine.has_unfinished_requests():
                if next_request < len(pending):
                    delay = rows[sources[pending[next_request]]["request_id"]]["arrival_s"] - now()
                    time.sleep(max(0.0, min(0.001, delay)))
                continue
            step_start = len(scheduler_steps)
            call = dict(call_index=len(engine_steps), start_s=now(), scheduler_step_start=step_start,
                        returned_s=None, completed=False, output_request_ids=[], new_output_tokens=0)
            engine_steps.append(call)
            try:
                outputs = engine.step()
            except Exception as error:
                call.update(returned_s=now(), scheduler_step_end=len(scheduler_steps), error=repr(error))
                raise
            received = now()  # Retain calls that return no output as well.
            call.update(returned_s=received, scheduler_step_end=len(scheduler_steps), completed=True,
                        output_request_ids=[output.request_id for output in outputs])
            for step in scheduler_steps[step_start:]:
                for item in step['scheduled']:
                    rid = item['internal_request_id']
                    executed_high_water[rid] = max(executed_high_water.get(rid, 0), item['computed_after'])
            for output in outputs:
                source_id = external_to_source[output.request_id]
                row = rows[source_id]
                if len(output.outputs) != 1 or row["status"] == "completed":
                    raise ValueError("unexpected completion count or output after completion")
                completion = output.outputs[0]
                tokens = list(completion.token_ids)
                previous = row["output_token_ids"]
                metrics = output.metrics
                metrics_copy = {name: getattr(metrics, name) for name in (
                    "arrival_time", "queued_ts", "scheduled_ts", "first_token_ts", "last_token_ts",
                    "first_token_latency", "num_generation_tokens", "is_corrupted") if metrics is not None and hasattr(metrics, name)}
                event = dict(request_id=source_id, external_request_id=output.request_id,
                    received_s=received, cumulative_token_ids=tokens, cumulative_tokens=len(tokens),
                    chunk_size=0, finished=bool(output.finished), native_metrics=metrics_copy,
                    finish_reason=completion.finish_reason, prefix_valid=False)
                output_events.append(event)  # Retain even a malformed output before failing closed.
                if tokens[:len(previous)] != previous or len(tokens) < len(previous) or len(tokens) > count:
                    raise ValueError("cumulative token prefix changed or output length exceeded")
                added = tokens[len(previous):]
                call['new_output_tokens'] += len(added)
                event.update(new_token_ids=added, chunk_size=len(added), prefix_valid=True)
                row["output_token_ids"] = tokens
                row["token_times_s"].extend([received] * len(added))
                row["native_metrics"] = metrics_copy
                if output.finished:
                    if len(tokens) != count or completion.finish_reason != "length":
                        raise ValueError("native completion violated fixed output length")
                    row.update(status="completed", completion_s=received, stop_reason="length")
        if error is None and any(row["status"] != "completed" for row in rows.values()):
            error = "engine drained before every planned request completed"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        for row in rows.values():
            if row["status"] != "completed" and row["admission_s"] is not None:
                row["status"] = "failed"
    finally:
        if scheduler is not None and original is not None:
            if had_instance_schedule:
                scheduler.schedule = original
            elif "schedule" in vars(scheduler):
                del scheduler.schedule
    sizes = [event["chunk_size"] for event in output_events]
    return dict(status="COMPLETE" if error is None else "INCOMPLETE", error=error,
        requests=list(rows.values()), scheduler_steps=scheduler_steps, output_events=output_events,
        engine_steps=engine_steps,
        scheduled_token_semantics="scheduled = first-time prefill + first-time decode + recompute; "
            "recompute uses the high-water mark from prior completed engine.step calls, not output-token count. "
            "Scheduled intervals in a failed engine.step are attempts, not confirmed executed work.",
        host_clock_origin_perf_counter=origin, observation_end_s=now(), regime=regime, arrival_scale=arrival_scale, target_cap=config["cap"],
        internal_to_source=internal_to_source, evidence_type="NATIVE_SERVING_INPROCESS_HOST_CAPTURE",
        queue_semantics="admission_s minus arrival_s is client submission lag; native waiting is scheduler telemetry",
        native_clock_semantics="arrival_time is epoch; native *_ts are core monotonic; host *_s share perf_counter origin",
        host_chunk_diagnostics=dict(multi_token_chunks=sum(size > 1 for size in sizes),
            max_chunk_size=max(sizes, default=0), token_level_itl_resolved=all(size <= 1 for size in sizes),
            interpolated=False, semantics="tokens in one chunk share receipt time; intra-chunk ITL is unresolved"))
