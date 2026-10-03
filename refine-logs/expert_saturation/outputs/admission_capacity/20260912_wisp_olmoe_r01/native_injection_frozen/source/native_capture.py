"""Arrival-driven in-process vLLM capture; timestamps are host receipt times."""
from __future__ import annotations

import hashlib
import json
import math
import time

from admission_feedback import AdmissionFeedback, apply_nonpreemptive_limit


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


def output_limits(sources, config):
    overrides = config.get("output_tokens_by_request", {})
    limits = {r["request_id"]: overrides.get(r["request_id"], config["output_tokens"]) for r in sources}
    if set(overrides) - set(limits) or any(type(n) is not int or n < 2 for n in limits.values()):
        raise ValueError("output limits require known request IDs and integers >=2")
    return limits


def event_release_time(rows, event):
    targets = event["after_output_tokens"]
    if any(len(rows[rid]["output_token_ids"]) > n for rid, n in targets.items()):
        raise ValueError("missed exact event output boundary")
    if all(len(rows[rid]["output_token_ids"]) == n for rid, n in targets.items()):
        return max(rows[rid]["token_times_s"][n - 1] for rid, n in targets.items())
    return None


def capture_episode(engine, workload, config, *, regime, arrival_scale, run_id, max_seconds=120,
                    event_arrival=None, before_event_add=None):
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
    limits = output_limits(sources, config)
    if type(count) is not int or count < 2 or type(config["cap"]) is not int or config["cap"] < 1:
        raise ValueError("need positive cap and at least two fixed output tokens")
    policy = config.get("policy", "static")
    if policy not in ("static", "shadow", "feedback", "single_shadow", "single_down"):
        raise ValueError("unknown admission policy")
    rows = {source["request_id"]: dict(request_id=source["request_id"], document_id=source["document_id"],
            arrival_s=arrival * arrival_scale, admission_s=None, engine_add_return_s=None, completion_s=None,
            prompt_token_ids_sha256=hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest(),
            prompt_tokens=len(ids), max_output_tokens=limits[source["request_id"]],
            output_token_ids=[], token_times_s=[], status="unfinished")
            for source, ids, arrival in zip(sources, prompts, arrivals)}
    if len(rows) != len(sources):
        raise ValueError("duplicate source request IDs")
    pending = sorted(range(len(sources)), key=lambda i: (arrivals[i] * arrival_scale, i))
    if event_arrival is not None:
        event_id, targets = event_arrival["request_id"], event_arrival["after_output_tokens"]
        if (event_id not in rows or not targets or event_id in targets or policy != "static"
                or any(rid not in rows or type(n) is not int or not 1 <= n < limits[rid]
                       for rid, n in targets.items())):
            raise ValueError("invalid event request, trigger, or non-static event policy")
        event_index = next(i for i in pending if sources[i]["request_id"] == event_id)
        pending.remove(event_index)
        rows[event_id]["arrival_s"] = None
    internal_to_source, external_to_source = {}, {}
    scheduler_steps, output_events = [], []
    event_actions, engine_calls = [], []
    scheduler = original = None
    had_instance_schedule = False
    error = None
    origin = time.perf_counter()
    epoch_origin = time.time()
    now = lambda: time.perf_counter() - origin
    feedback = None

    try:
        from vllm import SamplingParams
        from vllm.sampling_params import RequestOutputKind
        scheduler = engine.engine_core.engine_core.scheduler
        original = scheduler.schedule
        had_instance_schedule = "schedule" in vars(scheduler)
        if engine.vllm_config.scheduler_config.async_scheduling is not False or engine.vllm_config.scheduler_config.stream_interval != 1:
            raise ValueError("capture requires async_scheduling=False and stream_interval=1")
        if event_arrival and (engine.vllm_config.speculative_config is not None
                             or engine.vllm_config.cache_config.enable_prefix_caching):
            raise ValueError("event capture requires speculative and prefix caching disabled")
        maximum = engine.vllm_config.scheduler_config.max_num_seqs
        if policy != "static":
            feedback = AdmissionFeedback(config, maximum)

        def schedule(*args, **kwargs):
            started = now()
            before = {rid: (int(req.num_computed_tokens), int(req.num_prompt_tokens))
                      for rid, req in scheduler.requests.items()}
            running_before, waiting_before = scheduler.get_request_counts()
            decision = None
            existing_decode_ids = []
            target = config["cap"]
            if event_arrival:
                existing_decode_ids = [r.request_id for r in scheduler.running
                    if internal_to_source[r.request_id] in targets and r.num_computed_tokens >= r.num_prompt_tokens]
            if feedback is not None:
                existing_decode_ids = [request.request_id for request in scheduler.running
                    if request.num_computed_tokens >= request.num_prompt_tokens]
                decision = feedback.decide(decision_start_s=started, active=running_before,
                                           waiting=waiting_before)
                target = decision["target_cap"]
                decision["decision_end_s"] = now()
                decision["effective_scheduler_limit"] = apply_nonpreemptive_limit(scheduler, target, maximum)
                decision["applied_s"] = now()
            result = original(*args, **kwargs)
            scheduled = []
            for rid, amount in result.num_scheduled_tokens.items():
                prior, prompt_length = before[rid]
                after = int(scheduler.requests[rid].num_computed_tokens)
                # schedule() already increments computed; a reset/cache adjustment
                # inside it is retained explicitly instead of silently misclassifying.
                actual_start = after - amount
                if actual_start < 0 or amount <= 0:
                    raise ValueError("invalid native scheduled/computed token accounting")
                prefill = min(amount, max(0, prompt_length - actual_start))
                scheduled.append(dict(request_id=internal_to_source[rid], internal_request_id=rid,
                    computed_before=prior, computed_after=after, scheduled_start_computed=actual_start,
                    computed_adjustment=actual_start - prior, prompt_tokens=prompt_length,
                    scheduled_tokens=amount, prefill_tokens=prefill, decode_tokens=amount - prefill))
            running_after, waiting_after = scheduler.get_request_counts()
            preempted = [internal_to_source[rid] for rid in (result.preempted_req_ids or [])]
            # Under KV pressure the engine may legitimately preempt and later
            # recompute. Those are the quantities under study, so they are
            # measured per step rather than only rejected.
            recomputed = sum(max(0, -row["computed_adjustment"]) for row in scheduled)
            scheduler_steps.append(dict(step=len(scheduler_steps), start_s=started, end_s=now(),
                target_cap=target, running_before=running_before, waiting_before=waiting_before,
                actual_active=running_after, waiting_requests=waiting_after,
                decode_requests=sum(row["decode_tokens"] > 0 for row in scheduled), scheduled=scheduled,
                total_scheduled_tokens=sum(row["scheduled_tokens"] for row in scheduled),
                preempted_request_ids=preempted,
                n_preempted=len(preempted), recomputed_tokens=recomputed,
                kv_adjusted_request_ids=[row["request_id"] for row in scheduled
                                         if row["computed_adjustment"]]))
            if decision is not None:
                step = scheduler_steps[-1]
                step.update(effective_scheduler_limit=decision["effective_scheduler_limit"],
                    decision_index=decision["decision_index"])
            if decision is not None or event_arrival:
                step = scheduler_steps[-1]
                step["existing_decode_request_ids"] = [internal_to_source[rid] for rid in existing_decode_ids]
                decoded = {row["internal_request_id"] for row in scheduled if row["decode_tokens"] > 0}
                missing = sorted(set(existing_decode_ids) - decoded)
                step["existing_decode_all_scheduled"] = not missing
                # The non-preemptive invariant is asserted only when the caller
                # claims it. Under --allow-preemption these same facts are still
                # recorded per step, so the guarantee is never silently weakened.
                if (event_arrival or not config.get("allow_preemption", False)) and (
                        missing or step["preempted_request_ids"]
                        or any(row["computed_adjustment"] for row in scheduled)):
                    raise ValueError("nonpreemptive feedback requires every existing decode to advance without preemption or KV adjustment")
                if event_arrival and any(result.num_scheduled_tokens.get(rid) != 1 for rid in existing_decode_ids):
                    raise ValueError("event probe requires each existing old decode to advance exactly once")
                if event_actions and event_actions[-1]["engine_call"] == len(engine_calls) - 1:
                    if step["total_scheduled_tokens"] != event_arrival["expected_first_tokens"]:
                        raise ValueError("first injection step shape differs from planned old decode + new prefill")
            return result

        scheduler.schedule = schedule
        next_request = 0
        while next_request < len(pending) or engine.has_unfinished_requests() or (event_arrival and not event_actions):
            if now() >= max_seconds:
                error = "runtime_limit"
                break
            if event_arrival and not event_actions:
                released = event_release_time(rows, event_arrival)
                if released is not None:
                    action = dict(request_id=event_id, release_s=released, detected_s=now(),
                                  engine_call=len(engine_calls), inject=event_arrival.get("inject", True))
                    event_actions.append(action)  # Retain partial snapshot/action on failure.
                    rows[event_id]["arrival_s"] = released if action["inject"] else None
                    if before_event_add is not None:
                        before_event_add(scheduler, action, rows, now)
                    action["before_add_end_s"] = now()
                    if action["inject"]:
                        pending.append(event_index)
                        pending[next_request:] = sorted(pending[next_request:],
                            key=lambda i: (rows[sources[i]["request_id"]]["arrival_s"], i))
                    else:
                        rows[event_id]["status"] = "not_injected"
            # Every due request reaches the native queue; no client inflight cap.
            while next_request < len(pending):
                index = pending[next_request]
                source_id = sources[index]["request_id"]
                row = rows[source_id]
                if row["arrival_s"] > now():
                    break
                external_id = f"{run_id}/{source_id}"
                count = limits[source_id]
                params = SamplingParams(n=1, temperature=0.0, max_tokens=count, min_tokens=count,
                    ignore_eos=True, detokenize=False, output_kind=RequestOutputKind.CUMULATIVE)
                row["admission_s"] = now()
                internal_id = engine.add_request(external_id, {"prompt_token_ids": prompts[index]}, params,
                                                arrival_time=epoch_origin + row["arrival_s"])
                row["engine_add_return_s"] = now()
                if internal_id in internal_to_source or external_id in external_to_source:
                    raise ValueError("native request ID collision")
                if event_arrival and internal_id not in scheduler.requests:
                    raise ValueError("native add_request did not register the returned internal ID")
                internal_to_source[internal_id] = source_id
                external_to_source[external_id] = source_id
                row.update(external_request_id=external_id, internal_request_id=internal_id)
                next_request += 1
            if not engine.has_unfinished_requests():
                if next_request < len(pending):
                    delay = rows[sources[pending[next_request]]["request_id"]]["arrival_s"] - now()
                    time.sleep(max(0.0, min(0.001, delay)))
                elif event_arrival and not event_actions:
                    error = "event arrival never triggered before engine drained"
                    break
                continue
            call = dict(index=len(engine_calls), start_s=now(), scheduler_step_start=len(scheduler_steps))
            engine_calls.append(call)
            try:
                outputs = engine.step()
                received = now()  # Immediate host receipt for this output batch.
                call.update(return_s=received, returned=True, output_request_ids=[o.request_id for o in outputs])
            except Exception as exc:
                call.update(return_s=now(), returned=False, error=f"{type(exc).__name__}: {exc}")
                raise
            finally:
                call["scheduler_step_stop"] = len(scheduler_steps)
            request_itls = {} if feedback is not None else None
            for output in outputs:
                source_id = external_to_source[output.request_id]
                row = rows[source_id]
                count = limits[source_id]
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
                if request_itls is not None and previous and len(added) == 1:
                    request_itls[source_id] = received - row["token_times_s"][-1]
                event.update(new_token_ids=added, chunk_size=len(added), prefix_valid=True)
                row["output_token_ids"] = tokens
                row["token_times_s"].extend([received] * len(added))
                row["native_metrics"] = metrics_copy
                if output.finished:
                    if len(tokens) != count or completion.finish_reason != "length":
                        raise ValueError("native completion violated fixed output length")
                    row.update(status="completed", completion_s=received, stop_reason="length")
            if feedback is not None:
                feedback.observe(request_itls, received_s=received, available_s=now())
                if request_itls:
                    feedback.observations[-1]["available_s"] = now()
        if error is None and any(row["status"] not in ("completed", "not_injected") for row in rows.values()):
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
    total_preempted = sum(step["n_preempted"] for step in scheduler_steps)
    total_recomputed = sum(step["recomputed_tokens"] for step in scheduler_steps)
    return dict(status="COMPLETE" if error is None else "INCOMPLETE", error=error,
        requests=list(rows.values()), scheduler_steps=scheduler_steps, output_events=output_events,
        event_arrival=event_arrival, event_actions=event_actions, engine_calls=engine_calls,
        measurement_origin_unix_s=epoch_origin,
        observation_end_s=now(), regime=regime, arrival_scale=arrival_scale, target_cap=config["cap"],
        policy=policy, actions=[] if feedback is None else feedback.actions,
        feedback_observations=[] if feedback is None else feedback.observations,
        feedback_decisions=[] if feedback is None else feedback.decisions,
        internal_to_source=internal_to_source, evidence_type="NATIVE_SERVING_INPROCESS_HOST_CAPTURE",
        allow_preemption=bool(config.get("allow_preemption", False)),
        preemption_summary=dict(
            total_preemption_events=total_preempted,
            steps_with_preemption=sum(step["n_preempted"] > 0 for step in scheduler_steps),
            total_recomputed_tokens=total_recomputed,
            distinct_preempted_requests=len({rid for step in scheduler_steps
                                             for rid in step["preempted_request_ids"]}),
            semantics=("preemption counts come from the native scheduler; recomputed_tokens is the "
                       "negative computed-token adjustment observed when a preempted request resumes")),
        queue_semantics="admission_s minus arrival_s is client submission lag; native waiting is scheduler telemetry",
        native_clock_semantics="arrival_time is epoch; native *_ts are core monotonic; host *_s share perf_counter origin",
        host_chunk_diagnostics=dict(multi_token_chunks=sum(size > 1 for size in sizes),
            max_chunk_size=max(sizes, default=0), token_level_itl_resolved=all(size <= 1 for size in sizes),
            interpolated=False, semantics="tokens in one chunk share receipt time; intra-chunk ITL is unresolved"))
