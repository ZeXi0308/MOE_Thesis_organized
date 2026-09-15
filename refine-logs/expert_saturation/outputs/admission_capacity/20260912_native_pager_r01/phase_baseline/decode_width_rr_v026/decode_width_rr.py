"""Native v0.26 pure-decode width2 RR; no request eviction or future routing."""


def _state(scheduler, request):
    return dict(status=request.status.name, computed=request.num_computed_tokens,
                tokens=request.num_tokens, preemptions=request.num_preemptions,
                kv_block_ids=[list(g) for g in scheduler.kv_cache_manager.get_block_ids(request.request_id)])


def schedule_with_width(scheduler, original, *args, mode, decision, now, **kwargs):
    """Observe all arms; temporarily budget two ready rows only in rr2 mode."""
    if mode not in ("none", "rr2"):
        raise ValueError("decode width mode must be none or rr2")
    running = list(scheduler.running)
    budget = scheduler.max_num_scheduled_tokens
    if budget != 64:
        raise RuntimeError("decode width probe requires native budget64")
    eligible = (len(running) == 3 and not scheduler.waiting and not scheduler.skipped_waiting
                and all(r.status.name == "RUNNING" and r.num_preemptions == 0
                        and not r.num_output_placeholders and not r.spec_token_ids
                        and not r.has_encoder_inputs and r.num_computed_tokens >= r.num_prompt_tokens
                        and r.num_tokens_with_spec - r.num_computed_tokens == 1
                        and scheduler.current_step + 1 >= r.next_decode_eligible_step for r in running))
    applied = mode == "rr2" and eligible
    ids = [r.request_id for r in running]
    decision.update(mode=mode, eligible=eligible, applied=applied, start_s=now(),
        budget_before=budget, budget_applied=2 if applied else budget,
        running_order_before=ids, held_request_ids=[], selected_request_ids=[], status="started")
    if scheduler.scheduler_config.async_scheduling or scheduler.num_spec_tokens or scheduler.num_lookahead_tokens:
        raise RuntimeError("decode width probe requires synchronous non-speculative scheduling")
    before = {r.request_id: _state(scheduler, r) for r in running} if eligible else {}
    decision["request_state_before"] = before
    try:
        if applied:
            scheduler.max_num_scheduled_tokens = 2
        decision["budget_at_native_call"] = scheduler.max_num_scheduled_tokens
        output = original(*args, **kwargs)
        decision["budget_after_native_call"] = scheduler.max_num_scheduled_tokens
        selected = dict(output.num_scheduled_tokens)
        decision["selected_request_ids"] = list(selected)
        if eligible:
            expected = ids[:2] if applied else ids
            held = ids[2:] if applied else []
            after = {r.request_id: _state(scheduler, r) for r in running}
            decision.update(request_state_after=after, declared_held_request_ids=held)
            if selected != {rid: 1 for rid in expected} or output.preempted_req_ids:
                raise RuntimeError("eligible decode must schedule exactly the selected rows once without preemption")
            if [r.request_id for r in scheduler.running] != ids:
                raise RuntimeError("native running set/order changed during width step")
            for rid in expected:
                if (after[rid]["computed"] != before[rid]["computed"] + 1
                        or after[rid]["status"] != "RUNNING" or after[rid]["preemptions"]):
                    raise RuntimeError("selected decode progress/status changed unexpectedly")
            for rid in held:
                if after[rid] != before[rid]:
                    raise RuntimeError("held decode token/status/KV metadata changed")
            decision.update(held_request_ids=held, held_state_verified=True)
            if applied:
                scheduler.running[:] = running[2:] + running[:2]
        decision.update(running_order_after=[r.request_id for r in scheduler.running], status="applied")
        return output
    except BaseException as exc:
        decision.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        scheduler.max_num_scheduled_tokens = budget
        decision.update(budget_restored=scheduler.max_num_scheduled_tokens, end_s=now())
