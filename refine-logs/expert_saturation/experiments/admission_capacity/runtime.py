"""Non-preemptive admission probe using the existing Transformers KV helpers.

All timestamps are host wall time, including cache copies and telemetry. This
is a custom runtime, with one FCFS prefill before each full active-set decode.
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "docs/ideas/bcrd/experiments"))
import capture_continuous_decode as kv


def validate_schedule(schedule):
    if not schedule or schedule[0][0] != 0:
        raise ValueError("cap schedule must start at time zero")
    previous = -1.0
    for at, cap in schedule:
        if not math.isfinite(at) or at <= previous or type(cap) is not int or cap < 1:
            raise ValueError("schedule requires increasing finite times and positive integer caps")
        previous = at


def pressure_stats(router_logits, config, batch_size):
    """GPU reductions, one compact D2H transfer; no per-token route export.

    Reconstructs OLMoE softmax/top-k from returned logits, using its unchanged
    selection algorithm. This is not a fused-backend dispatch trace.
    """
    import torch
    if router_logits is None or len(router_logits) != config.num_hidden_layers:
        raise ValueError("missing per-layer router logits")
    compact = []
    experts = config.num_experts
    for logits in router_logits:
        if tuple(logits.shape) != (batch_size, experts):
            raise ValueError("router batch/layer identity mismatch")
        if not batch_size:
            compact.append(torch.zeros(4, device=logits.device))
            continue
        selected = torch.topk(torch.softmax(logits.float(), dim=-1),
                              config.num_experts_per_tok, dim=-1).indices
        counts = torch.zeros(experts, dtype=torch.float32, device=logits.device)
        counts.scatter_add_(0, selected.flatten(), torch.ones_like(selected.flatten(), dtype=torch.float32))
        total = batch_size * config.num_experts_per_tok
        compact.append(torch.stack(((counts > 0).float().mean(),
                                    counts.max() / (total / experts), counts.max(), counts.sum())))
    values = torch.stack(compact).cpu().tolist()
    return [{"layer": i, "U": x[0] if x[3] else None,
             "C": x[1] if x[3] else None, "max_expert_tokens": int(x[2]),
             "routed_tokens": int(x[3]), "zero_tokens": not bool(x[3])}
            for i, x in enumerate(values)]


def recent_pressure(steps):
    """Completed four-step window; averages per-step statistics, never a union."""
    window = steps[-4:]
    stats = [p for step in window for p in step["pressure"] if p["U"] is not None]
    return dict(completed_steps=len(window), observed_layer_steps=len(stats),
                U_mean=sum(p["U"] for p in stats) / len(stats) if stats else None,
                C_mean=sum(p["C"] for p in stats) / len(stats) if stats else None,
                max_expert_tokens=max((p["max_expert_tokens"] for p in stats), default=None))


def run_episode(model, requests, *, cap_schedule, output_tokens, telemetry=True,
                eos_token_id=None, max_seconds=600, clock=time.perf_counter,
                sleep=time.sleep, step_action=None):
    import torch
    validate_schedule(cap_schedule)
    if step_action is not None:
        if (not isinstance(step_action, (tuple, list)) or len(step_action) != 2
                or any(type(x) is not int or x < 1 for x in step_action)):
            raise ValueError("step action requires positive integer completed steps and cap")
        if len(cap_schedule) != 1:
            raise ValueError("step action permits only one initial time-zero cap")
    ordered = sorted(requests, key=lambda r: (r.arrival_us, r.request_id))
    if not ordered or len({r.request_id for r in ordered}) != len(ordered):
        raise ValueError("requests must be nonempty with unique IDs")
    if output_tokens < 2 or not math.isfinite(max_seconds) or max_seconds <= 0:
        raise ValueError("need at least two output tokens and a finite positive runtime limit")
    if any(not math.isfinite(r.arrival_us) or r.arrival_us < 0 for r in ordered):
        raise ValueError("arrivals must be finite and nonnegative")
    if ordered[-1].arrival_us / 1e6 >= max_seconds:
        raise ValueError("runtime limit must extend beyond the last arrival")
    rows = {r.request_id: dict(request_id=r.request_id, document_id=r.document_id,
            arrival_s=r.arrival_us / 1e6, admission_s=None, completion_s=None,
            prompt_tokens=int(r.input_ids.shape[1]),
            prompt_token_ids_sha256=kv._prompt_token_ids_sha256(r.input_ids),
            token_times_s=[], output_token_ids=[],
            status="unfinished") for r in ordered}
    pending, active, steps, actions = list(ordered), [], [], []
    target, next_action, previous_target = cap_schedule[0][1], 0, cap_schedule[0][1]
    step_action_pending = step_action is not None
    error = None
    if model.device.type == "cuda":
        torch.cuda.synchronize(model.device)
    origin = clock()
    now = lambda: clock() - origin

    def record_token(state, token, emitted):
        row = rows[state.spec.request_id]
        row["output_token_ids"].append(token)
        row["token_times_s"].append(emitted)
        if token == eos_token_id or len(row["output_token_ids"]) >= output_tokens:
            row.update(status="completed", completion_s=emitted,
                       stop_reason="eos" if token == eos_token_id else "length")
            return True
        return False

    def settle_actions():
        for action in actions:
            if action["effective_s"] is None and action["superseded_s"] is None:
                reached = (len(active) <= action["target"] if action["direction"] == "down"
                           else len(active) >= action["target"])
                if reached:
                    action["effective_s"] = now()
                    action["effect_delay_s"] = action["effective_s"] - action["applied_s"]

    try:
        with torch.inference_mode():
            while pending or active:
                t = now()
                if t >= max_seconds:
                    error = "runtime_limit"
                    break
                while next_action < len(cap_schedule) and cap_schedule[next_action][0] <= t:
                    at, new_target = cap_schedule[next_action]
                    for a in actions:
                        if a["effective_s"] is None and a["superseded_s"] is None:
                            a["superseded_s"] = t
                    previous_target, target = target, new_target
                    actions.append(dict(requested_s=at, applied_s=t, previous_target=previous_target,
                        target=target, actual_active=len(active), direction="down" if target < previous_target else "up",
                        effective_s=None, effect_delay_s=None, superseded_s=None,
                        recent_pressure=recent_pressure(steps)))
                    next_action += 1
                if step_action_pending and len(steps) == step_action[0]:
                    # Observe completed history before this iteration's admission/prefill.
                    waiting = [r for r in pending if r.arrival_us / 1e6 <= t]
                    future = [r for r in pending if r.arrival_us / 1e6 > t]
                    latest = steps[-1]
                    snapshot = dict(observed_s=t, target_cap=target,
                        actual_active=len(active),
                        active_request_ids=[s.spec.request_id for s in active],
                        active_decode_steps=[s.decode_step for s in active],
                        kv_lengths=[int(s.attention_mask.shape[1]) for s in active],
                        waiting_requests=len(waiting), future_requests=len(future),
                        waiting_request_ids=[r.request_id for r in waiting],
                        future_request_ids=[r.request_id for r in future],
                        oldest_wait_s=max((t - r.arrival_us / 1e6 for r in waiting), default=0),
                        recent_model_call_s=latest["model_call_s"],
                        recent_iteration_s=latest["iteration_s"],
                        recent_itl_s=sum(latest["itl_s"]) / len(latest["itl_s"]),
                        latest_completed_step={key: latest[key] for key in
                            ("step", "request_ids", "decode_steps", "completed_s", "pressure")},
                        history_step_indices=[s["step"] for s in steps[-4:]])
                    snapshot["latest_completed_step"]["decode_requests"] = len(latest["request_ids"])
                    history_pressure = recent_pressure(steps)
                    applied = now()  # Snapshot cost remains inside request wall time.
                    for a in actions:
                        if a["effective_s"] is None and a["superseded_s"] is None:
                            a["superseded_s"] = applied
                    previous_target, target = target, step_action[1]
                    direction = "down" if target < previous_target else "up" if target > previous_target else "hold"
                    actions.append(dict(trigger="completed_decode_steps",
                        requested_completed_steps=step_action[0], completed_steps=len(steps),
                        requested_s=latest["completed_s"], applied_s=applied,
                        previous_target=previous_target, target=target, actual_active=len(active),
                        direction=direction, effective_s=applied if direction == "hold" else None,
                        effect_delay_s=0.0 if direction == "hold" else None, superseded_s=None,
                        pre_action=snapshot, recent_pressure=history_pressure))
                    step_action_pending = False
                settle_actions()
                if not active and pending[0].arrival_us / 1e6 > now():
                    sleep(max(0.0, min(0.01, pending[0].arrival_us / 1e6 - now())))
                    continue
                prefill_s = 0.0
                # Fixed prefill quota = one request per iteration, at every cap.
                if pending and len(active) < target and pending[0].arrival_us / 1e6 <= now():
                    spec = pending.pop(0)
                    row = rows[spec.request_id]
                    row["admission_s"] = now()
                    out = model(input_ids=spec.input_ids, attention_mask=spec.attention_mask,
                                use_cache=True, output_router_logits=False, return_dict=True)
                    token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
                    state = kv._ActiveRequest(spec, out.past_key_values, spec.attention_mask,
                                              token, int(spec.input_ids.shape[1]))
                    active.append(state)
                    emitted_token = int(token.item())
                    emitted = now()
                    row["prefill_end_s"] = emitted
                    prefill_s = emitted - row["admission_s"]
                    if record_token(state, emitted_token, emitted):
                        active.remove(state)
                    del out
                    settle_actions()
                if not active:
                    continue
                # Never slice active[:target]: a down action only blocks admission.
                batch = list(active)
                start = now()
                queue = [r for r in pending if r.arrival_us / 1e6 <= start]
                ordinary = dict(target_cap=target, previous_target=previous_target,
                    actual_active=len(active), decode_requests=len(batch), waiting_requests=len(queue),
                    future_requests=len(pending) - len(queue),
                    oldest_wait_s=max((start - r.arrival_us / 1e6 for r in queue), default=0),
                    prefill_requests_this_iteration=int(prefill_s > 0), prefill_s=prefill_s,
                    recent_model_call_s=steps[-1]["model_call_s"] if steps else None,
                    recent_iteration_s=steps[-1]["iteration_s"] if steps else None,
                    recent_itl_s=(sum(steps[-1]["itl_s"]) / len(steps[-1]["itl_s"])) if steps else None)
                ordinary["completed_pressure_window"] = recent_pressure(steps)
                ids, mask, positions, cache, lengths, maximum = kv._pad_decode_inputs(batch)
                ordinary.update(kv_lengths=lengths, logical_kv_tokens=sum(lengths),
                                padded_kv_tokens=maximum * len(batch))
                out, elapsed_us = kv._timed_call(model, "decode", len(batch), None,
                    input_ids=ids, attention_mask=mask, position_ids=positions,
                    cache_position=torch.tensor([maximum], device=ids.device),
                    past_key_values=cache, use_cache=True, output_router_logits=telemetry, return_dict=True)
                split = kv.split_left_padded_cache(out.past_key_values,
                    prior_lengths=lengths, prior_max_length=maximum)
                tokens = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
                token_ids = tokens.flatten().cpu().tolist()
                emitted = now()
                itls = []
                for i, state in enumerate(batch):
                    itls.append(emitted - rows[state.spec.request_id]["token_times_s"][-1])
                    state.cache, state.next_token = split[i], tokens[i:i+1]
                    state.attention_mask = torch.cat((state.attention_mask,
                                                     state.attention_mask.new_ones((1, 1))), dim=1)
                    state.decode_step += 1
                    if record_token(state, token_ids[i], emitted):
                        active.remove(state)
                stats_start = now()
                pressure = pressure_stats(out.router_logits, model.config, len(batch)) if telemetry else []
                stats_s = now() - stats_start
                completed = now()
                steps.append(dict(step=len(steps), start_s=start, token_emission_s=emitted,
                    completed_s=completed,
                    request_ids=[s.spec.request_id for s in batch],
                    decode_steps=[s.decode_step for s in batch], ordinary=ordinary,
                    model_call_s=elapsed_us / 1e6, iteration_s=completed - start + prefill_s,
                    itl_s=itls, telemetry_s=stats_s, pressure=pressure))
                settle_actions()
                del out, cache, split, batch
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        for row in rows.values():
            if row["status"] != "completed" and row["admission_s"] is not None:
                row["status"] = "failed"
    end = now()
    return dict(status="COMPLETE" if error is None else "INCOMPLETE", error=error,
                requests=list(rows.values()), steps=steps, actions=actions,
                observation_end_s=end, evidence_type="CUSTOM_CONTINUOUS_RUNTIME",
                strict_same_prestate_pair=False, queue_policy="FCFS",
                prefill_policy="one_full_prefill_per_iteration_before_all_active_decode",
                pressure_source="OLMoE_returned_logits_softmax_topk_reconstruction")
