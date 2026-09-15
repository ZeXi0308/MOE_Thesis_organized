"""Pinned native rotation: actual preemption plus temporary recovery reservation.

This changes execution as well as queue order. Held requests retain their KV;
the resumed request recomputes its own history through the native worker path.
"""
from __future__ import annotations

import ast
from dataclasses import asdict
import hashlib
import inspect
import time
from types import MethodType

from absence_rotation import AbsenceRotation, RequestView, RotationConfig

SCHEDULER_SHA256 = "2ed2a550b6558b2495eda845a97ae38bcf0225027b9e25fbf00fc3880c1d3941"


def patched_schedule_tree(source):
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Scheduler")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "schedule")
    matches = [0, 0, 0, 0]
    for i, node in enumerate(fn.body):
        if isinstance(node, ast.Assign) and ast.unparse(node) == "scheduled_timestamp = time.monotonic()":
            fn.body[i + 1:i + 1] = ast.parse("self._rotation_begin(preempted_reqs, scheduled_timestamp)").body
            matches[0] += 1
            break
    for node in ast.walk(fn):
        if isinstance(node, ast.While) and ast.unparse(node.test) == "req_index < len(self.running) and token_budget > 0":
            if ast.unparse(node.body[0]) != "request = self.running[req_index]":
                raise ValueError("native running loop changed")
            node.body[1:1] = ast.parse("if self._rotation_hold(request):\n    req_index += 1\n    continue").body
            matches[1] += 1
        elif isinstance(node, ast.If) and ast.unparse(node.test) == "not preempted_reqs and self._pause_state == PauseState.UNPAUSED":
            node.test = ast.parse("len(preempted_reqs) == self._rotation_forced_count and self._pause_state == PauseState.UNPAUSED", mode="eval").body
            matches[2] += 1
        elif isinstance(node, ast.While) and ast.unparse(node.test) == "(self.waiting or self.skipped_waiting) and token_budget > 0":
            for i, statement in enumerate(node.body):
                if isinstance(statement, ast.Assign) and ast.unparse(statement) == "request = request_queue.peek_request()":
                    node.body[i + 1:i + 1] = ast.parse("if self._rotation_target is not None and request.request_id != self._rotation_target:\n    break").body
                    matches[3] += 1
                    break
    if matches != [1, 1, 1, 1] or fn.decorator_list:
        raise ValueError(f"unsupported native scheduler structure: {matches}")
    return ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))


def install(scheduler, *, vllm_config, block_size, expected_requests=32,
            mode="rotate", rotation_config=None, victim_order="least_progress"):
    """Install on the drained scheduler BEFORE memory/native capture wrappers."""
    manager, pool = scheduler.kv_cache_manager, scheduler.kv_cache_manager.block_pool
    if mode not in ("native", "rotate") or block_size <= 0:
        raise ValueError("unsupported arm or block size")
    if (vllm_config.scheduler_config.async_scheduling
            or vllm_config.speculative_config is not None
            or manager.enable_caching or manager.num_kv_cache_groups != 1
            or manager.use_eagle or manager.watermark_blocks
            or scheduler.num_lookahead_tokens or scheduler.num_spec_tokens
            or scheduler.dcp_world_size != 1 or scheduler.pcp_world_size != 1
            or scheduler.use_v2_model_runner or scheduler.defer_block_free
            or scheduler.connector is not None or scheduler.ec_connector is not None
            or scheduler.policy.name != "FCFS" or scheduler.is_encoder_decoder
            or not scheduler.scheduler_reserve_full_isl
            or scheduler.max_num_scheduled_tokens <= expected_requests):
        raise ValueError("requires pinned synchronous full-history FCFS execution")
    if scheduler.requests or "schedule" in vars(scheduler):
        raise ValueError("install requires a drained, unwrapped scheduler")
    singles = manager.coordinator.single_type_managers
    if (len(singles) != 1 or type(singles[0]).__name__ != "FullAttentionManager"
            or type(manager.coordinator).__name__ != "KVCacheCoordinatorNoPrefixCache"
            or singles[0].block_pool is not pool):
        raise ValueError("requires unshared full-attention blocks")
    owned = singles[0].req_to_blocks
    original = scheduler.schedule
    source = inspect.getsourcefile(type(scheduler))
    with open(source, "rb") as handle:
        source_bytes = handle.read()
    if hashlib.sha256(source_bytes).hexdigest() != SCHEDULER_SHA256:
        raise ValueError("installed scheduler differs from pinned source")
    namespace = dict(original.__func__.__globals__)
    exec(compile(patched_schedule_tree(source_bytes.decode()), source, "exec"), namespace)
    patched = MethodType(namespace["schedule"], scheduler)
    tracker = AbsenceRotation(rotation_config or RotationConfig(), victim_order=victim_order)
    decisions, cohort, protected, output_at_start = [], None, None, None
    current, proposed_victim, held_before = None, None, {}
    scheduler._rotation_target, scheduler._rotation_forced_count = None, 0

    def need(request):
        return max(0, (request.num_tokens + block_size - 1) // block_size
                   - len(owned.get(request.request_id, ())))

    def begin(preempted, timestamp):
        nonlocal proposed_victim
        scheduler._rotation_forced_count = 0
        if proposed_victim is not None:
            victim = proposed_victim
            scheduler.running.remove(victim)
            # Resolve this method dynamically so memory_telemetry sees the event.
            scheduler._preempt_request(victim, timestamp)
            preempted.append(victim)
            scheduler._rotation_forced_count = 1
            current["forced_preempted"].append(victim.request_id)
            proposed_victim = None
        if protected is not None and protected.status.name == "PREEMPTED":
            scheduler.waiting.remove_request(protected)
            scheduler.waiting.prepend_request(protected)

    def hold(request):
        if protected is None or request is protected:
            return False
        if request.num_computed_tokens != request.num_tokens - 1:
            raise RuntimeError("unexpected concurrent recovery during protection")
        blocks = owned.get(request.request_id)
        if blocks is None:
            raise RuntimeError("running request has no owned blocks")
        cost = max(0, (request.num_computed_tokens + block_size) // block_size - len(blocks))
        if cost <= pool.get_num_free_blocks() - need(protected):
            return False
        current["held"].append(request.request_id)
        held_before[request.request_id] = (request.num_computed_tokens, tuple(b.block_id for b in blocks))
        return True

    def schedule(*args, **kwargs):
        nonlocal cohort, protected, output_at_start, current, proposed_victim, held_before
        started = time.perf_counter()
        step = len(decisions)
        pure = lambda r: r.num_output_tokens > 0 and r.num_computed_tokens == r.num_tokens - 1
        if cohort is None and (len(scheduler.running) == expected_requests
                and all(pure(r) and not r.num_preemptions for r in scheduler.running)
                and not scheduler.waiting and not scheduler.skipped_waiting):
            for r in scheduler.running:
                blocks = owned.get(r.request_id)
                if (blocks is None or any(b.is_null or pool.blocks[b.block_id] is not b for b in blocks)
                        or [b.block_id for b in blocks] != manager.get_blocks(r.request_id).get_block_ids()[0]):
                    raise RuntimeError("block ownership is not qualified")
            cohort = set(scheduler.requests)
        current = dict(step=step, mode=mode, active=cohort is not None, held=[],
                       forced_preempted=[], natural_preempted=[], status="ATTEMPTED",
                       free_before=pool.get_num_free_blocks(), proposal=None,
                       effective_victim_order=tracker.effective_victim_order,
                       applied_rotations_before=tracker.applied_rotations)
        decisions.append(current)
        held_before, proposed_victim = {}, None
        if cohort is not None:
            if (not set(scheduler.requests) <= cohort or scheduler.skipped_waiting
                    or any(r.num_output_placeholders or r.num_in_flight_tokens or r.spec_token_ids
                           or r.has_encoder_inputs or r.num_prompt_tokens + r.max_tokens > scheduler.max_model_len
                           for r in scheduler.requests.values())):
                raise RuntimeError("closed synchronous text cohort invariant changed")
        if protected is not None and protected.num_output_tokens > output_at_start:
            current["recovery_completed"] = protected.request_id
            protected, output_at_start = None, None
        if protected is not None and protected.request_id not in scheduler.requests:
            raise RuntimeError("protected request disappeared before new output")
        if mode == "rotate" and cohort is not None and protected is None:
            if all(pure(r) for r in scheduler.running):
                waiting = [r for r in scheduler.waiting if r.status.name == "PREEMPTED"]
                rows = [RequestView(r.request_id, r.num_computed_tokens, r.num_prompt_tokens,
                                    r.num_prompt_tokens + r.max_tokens, r.num_output_tokens)
                        for r in scheduler.running]
                previous_swap = tracker.last_swap_step
                proposal = tracker.decide(step, rows, [r.request_id for r in waiting],
                    pool.get_num_free_blocks(), {r.request_id: need(r) for r in waiting})
                current["proposal"] = asdict(proposal)
                if proposal.action == "rotate":
                    target, victim = scheduler.requests[proposal.resume_id], scheduler.requests[proposal.victim_id]
                    released = len(owned.get(victim.request_id, ()))
                    current.update(candidate_released_blocks=released, candidate_required_blocks=need(target))
                    if pool.get_num_free_blocks() + released < need(target):
                        tracker.last_swap_step = previous_swap
                        current["not_applied_reason"] = "victim cannot fund complete recovery history"
                    else:
                        protected, proposed_victim = target, victim
                        output_at_start = target.num_output_tokens
            else:
                current["not_applied_reason"] = "native recovery already underway"
        scheduler._rotation_target = protected.request_id if protected is not None else None
        current.update(recovery_target=scheduler._rotation_target,
                       recovery_output_at_start=output_at_start,
                       recovery_remaining_blocks_before=need(protected) if protected is not None else 0,
                       decision_seconds=time.perf_counter() - started)
        try:
            result = patched(*args, **kwargs)
            preempted = set(result.preempted_req_ids or ())
            resumed = set(result.scheduled_cached_reqs.resumed_req_ids)
            current.update(actual_scheduled=dict(result.num_scheduled_tokens),
                preempted=sorted(preempted), natural_preempted=sorted(preempted - set(current["forced_preempted"])),
                resumed=sorted(resumed), free_after=pool.get_num_free_blocks(),
                recovery_remaining_blocks_after=need(protected) if protected is not None else 0)
            tracker.note_preempted(step, sorted(preempted))
            tracker.note_resumed(step, sorted(resumed))
            if protected is not None:
                if (current["natural_preempted"] or result.num_scheduled_tokens.get(protected.request_id, 0) <= 0
                        or pool.get_num_free_blocks() < need(protected)):
                    raise RuntimeError("native execution violated recovery reservation")
                for rid, before in held_before.items():
                    r = scheduler.requests[rid]
                    if (r.status.name != "RUNNING" or result.num_scheduled_tokens.get(rid, 0)
                            or (r.num_computed_tokens, tuple(b.block_id for b in owned[rid])) != before):
                        raise RuntimeError("held request lost KV or progress")
            if current["forced_preempted"]:
                tracker.note_rotation_applied()
            current["status"] = "APPLIED"
            return result
        except Exception as error:
            current.update(status="ERROR", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            current["wrapped_schedule_seconds"] = time.perf_counter() - started

    scheduler._rotation_begin, scheduler._rotation_hold = begin, hold
    scheduler.schedule = schedule

    def uninstall():
        for name in ("schedule", "_rotation_begin", "_rotation_hold", "_rotation_target", "_rotation_forced_count"):
            delattr(scheduler, name)

    return decisions, uninstall
