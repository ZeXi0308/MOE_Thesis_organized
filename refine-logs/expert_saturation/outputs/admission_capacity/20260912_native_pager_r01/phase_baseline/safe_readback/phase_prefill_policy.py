"""Simple pre-KV phase baseline for the synchronous vLLM 0.11.2 probe.

Install outside its existing schedule/logging wrapper. The probe owns arrivals,
request limits, resources, common preaction state and completion accounting.
This module changes only the per-request prefill threshold before scheduling.
"""
from functools import wraps


def _check_request(request):
    if (request.num_preemptions or request.num_output_placeholders
            or request.spec_token_ids):
        raise RuntimeError("phase baseline excludes preemption, async and speculative state")
    prompt, total, computed = (request.num_prompt_tokens, request.num_tokens,
                               request.num_computed_tokens)
    if not 0 <= computed <= total or total < prompt:
        raise RuntimeError("invalid request progress")
    if total > prompt and computed != total - 1:
        raise RuntimeError("output-bearing request is not ready for one synchronous decode")


def choose_threshold(running, *, enabled, mode):
    """Use only current running progress; 0 removes the per-request cap."""
    if mode not in ("phase8", "static8"):
        raise ValueError("mode must be phase8 or static8")
    ready = []
    for request in running:
        _check_request(request)
        if request.status.name != "RUNNING":
            raise RuntimeError("running list contains an unsupported request state")
        if request.num_tokens > request.num_prompt_tokens:
            ready.append(request.request_id)
    return dict(enabled=bool(enabled), mode=mode, ready_decode_ids=ready,
                chosen_threshold=32 if not enabled else
                (8 if mode == "static8" or ready else 0))


def install(scheduler, *, vllm_config, mode, is_enabled, current_call):
    """Wrap the probe's schedule(); current_call() returns its current step dict."""
    if (vllm_config.scheduler_config.async_scheduling
            or vllm_config.speculative_config is not None
            or vllm_config.cache_config.enable_prefix_caching
            or scheduler.num_lookahead_tokens or scheduler.num_spec_tokens):
        raise RuntimeError("phase baseline requires sync/no-spec/no-prefix execution")
    if scheduler.scheduler_config.long_prefill_token_threshold != 32:
        raise RuntimeError("install before the common threshold32 action boundary")
    if mode not in ("phase8", "static8"):
        raise ValueError("mode must be phase8 or static8")
    original = scheduler.schedule
    started = False

    @wraps(original)
    def schedule():
        nonlocal started
        enabled = bool(is_enabled())
        if started and not enabled:
            raise RuntimeError("the action boundary cannot be reversed")
        if not enabled and scheduler.scheduler_config.long_prefill_token_threshold != 32:
            raise RuntimeError("common preaction threshold changed")
        for request in scheduler.requests.values():
            _check_request(request)
        decision = choose_threshold(scheduler.running, enabled=enabled, mode=mode)
        decision["threshold_before"] = scheduler.scheduler_config.long_prefill_token_threshold
        call = current_call()
        call["phase_prefill"] = decision
        started = started or enabled
        scheduler.scheduler_config.long_prefill_token_threshold = decision["chosen_threshold"]
        try:
            output = original()  # Native threshold selection precedes allocate_slots.
            rows = call["scheduled"]  # Existing probe records actual scheduled work.
            decision["actual_prefill_rows"] = sum(r["prefill_tokens"] for r in rows)
            decision["actual_decode_rows"] = sum(r["decode_tokens"] for r in rows)
            decision["actual_scheduled"] = [dict(r) for r in rows]
            by_id = {r["request_id"]: r for r in rows}
            for rid in decision["ready_decode_ids"]:
                row = by_id.get(rid)
                if (row is None or row["tokens"] != 1
                        or row["prefill_tokens"] != 0 or row["decode_tokens"] != 1):
                    raise RuntimeError("ready incumbent decode did not advance exactly once")
            if call.get("preempted_request_ids"):
                raise RuntimeError("preemption is outside the phase baseline")
            decision["status"] = "applied"
            return output
        except Exception as exc:
            decision.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            raise

    scheduler.schedule = schedule
    return original
