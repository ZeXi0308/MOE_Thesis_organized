"""Native admission-only baselines; every event request enters the real queue."""

MODES = ("immediate32", "admit2_32", "admit2_release64")


def apply_admission(scheduler, action, mode, now):
    if mode not in MODES:
        raise ValueError("unknown admission baseline")
    if (scheduler.max_num_running_reqs != 3 or len(scheduler.running) != 2
            or scheduler.waiting or scheduler.skipped_waiting
            or scheduler.max_num_scheduled_tokens != 64):
        raise RuntimeError("admission boundary requires two old running requests, cap3 and budget64")
    action.update(admission_mode=mode, admission_cap_before=3)
    scheduler.max_num_running_reqs = 3 if mode == "immediate32" else 2
    action.update(admission_cap_after=scheduler.max_num_running_reqs, admission_applied_s=now())


def observe_waiting(scheduler, mode, event_id, targets, rows, scheduled, step):
    """Check the actual queued request while both old requests remain unfinished."""
    old_done = [rid for rid in targets if rows[rid]["status"] == "completed"]
    rid = rows[event_id]["internal_request_id"]
    request = scheduler.requests.get(rid)
    if request is None:
        if rows[event_id]["status"] != "completed":
            raise RuntimeError("queued event request disappeared before completion")
        return
    state = dict(internal_request_id=rid, status=request.status.name,
        computed=request.num_computed_tokens, preemptions=request.num_preemptions,
        kv_block_ids=[list(g) for g in scheduler.kv_cache_manager.get_block_ids(rid)],
        scheduled_tokens=scheduled.get(rid, 0), old_completed=old_done)
    step["admission_wait"] = state
    if mode != "immediate32" and not old_done:
        if (state["status"] != "WAITING" or state["computed"] or state["preemptions"]
                or state["scheduled_tokens"] or any(state["kv_block_ids"])):
            raise RuntimeError("delayed new request must remain never-started WAITING with no KV")
        state["never_started_wait_verified"] = True
