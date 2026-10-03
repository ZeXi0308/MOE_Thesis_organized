"""LTR-style priority component on the pinned native selected-save backend.

This is a recovery-only component, not the complete LTR system.  The policy
owns waiting counters and a call-count quantum; this adapter owns the native
store, preemption, queue and load lifecycle.  No predicted EOS or future KV
state is used.  ``install`` is deliberately limited to the qualified vLLM
OffloadingConnector configuration and refuses scheduler source drift.
"""
from __future__ import annotations

import hashlib
import inspect
from types import MethodType

from ltr_fair_policy import LTRFairPolicy, Request
from native_store_delta import inspect_store_delta
from rotation_native import SCHEDULER_SHA256, patched_schedule_tree
from staged_save_contract import RequestState, commit_reason, prepare


def _attach(scheduler, native, connector_scheduler, *, block_size, policy):
    """Attach to a qualified scheduler.  Separated for CPU lifecycle fixtures."""
    if block_size != 16:
        raise ValueError("selected-store contract requires 16-token blocks")
    manager = scheduler.kv_cache_manager
    pool = manager.block_pool
    owned = manager.coordinator.single_type_managers[0].req_to_blocks
    cs = connector_scheduler
    hooks = ("_rotation_begin", "_rotation_hold", "_rotation_target",
             "_rotation_forced_count")
    if "schedule" in vars(scheduler) or any(x in vars(scheduler) for x in hooks):
        raise ValueError("scheduler already has an adapter")
    oldcalc = cs._calc_num_offloadable_tokens
    hadcalc = "_calc_num_offloadable_tokens" in vars(cs)
    step = 0
    plan = None
    phase = None
    pending_flush = set()
    committed_victim_id = None
    birth_order = {}
    last_output = {}
    data = dict(status="INSTALLED", events=[], applied_rotations=0,
                free_priorities=0, threshold=policy.counters.threshold,
                quantum=policy.counters.quantum, no_op_counts={})

    def view(req):
        if req is None:
            return None
        blocks = owned.get(req.request_id, ())
        if any(b.is_null or pool.blocks[b.block_id] is not b for b in blocks):
            raise RuntimeError("invalid physical block ownership")
        return RequestState(req.request_id, req.num_computed_tokens,
                            req.num_prompt_tokens, req.num_output_tokens,
                            req.max_tokens, req.status.name,
                            tuple(b.block_id for b in blocks))

    def selected_store(rs, n):
        if (phase != "prepare" or plan is None
                or rs.req.request_id != plan.victim.request_id):
            return 0
        return min(oldcalc(rs, n), plan.saved_tokens)

    def live_rows():
        for req in scheduler.requests.values():
            birth_order.setdefault(req.request_id, len(birth_order))
        rows = []
        for req in scheduler.requests.values():
            state = view(req)
            jobs = cs._req_status.get(req.request_id)
            pending_load = bool(jobs and any(
                not cs._jobs[j].is_store for j in jobs.transfer_jobs))
            eligible = (state.status == "RUNNING" and state.pure_decode
                        and state.output + 1 < state.max_output
                        and bool(state.blocks) and jobs is not None
                        and not pending_load)
            rows.append(Request(req.request_id, state.status,
                                float(req.arrival_time), birth_order[req.request_id],
                                state.output, len(state.blocks),
                                state.remaining_blocks, eligible))
        return rows

    def order_running():
        ranked = policy.ordered_running_ids()
        running = {r.request_id: r for r in scheduler.running}
        if len(running) != len(scheduler.running) or set(ranked) != set(running):
            raise RuntimeError("policy running order lost a request")
        scheduler.running[:] = [running[rid] for rid in ranked]

    def put_waiting_first(request_id):
        target = scheduler.requests.get(request_id)
        if target is None or target.status.name != "PREEMPTED":
            return False
        if target not in scheduler.waiting:
            return False
        scheduler.waiting.remove_request(target)
        scheduler.waiting.prepend_request(target)
        return True

    def begin(preempted, timestamp):
        nonlocal plan, phase, pending_flush, committed_victim_id
        scheduler._rotation_forced_count = 0
        scheduler._rotation_target = None
        phase = None
        pending_flush = set()
        committed_victim_id = None
        excluded_targets = set()
        excluded_victims = set()

        previous_active = policy.active_target
        if previous_active is not None:
            previous_req = scheduler.requests.get(previous_active)
            if previous_req is None:
                data["events"].append(dict(step=step, event="target_terminal",
                                           target=previous_active))
            elif (last_output.get(previous_active) is not None
                  and previous_req.num_output_tokens > last_output[previous_active]):
                data["events"].append(dict(
                    step=step, event="new_output", target=previous_active,
                    count=previous_req.num_output_tokens-last_output[previous_active]))
                last_output[previous_active] = previous_req.num_output_tokens

        rows = live_rows()
        policy.observe(rows)
        order_running()
        active = policy.active_target
        if active is not None:
            target = scheduler.requests.get(active)
            if target is None or target.status.name not in (
                    "PREEMPTED", "RUNNING", "WAITING_FOR_REMOTE_KVS"):
                policy.release_active()
                data["events"].append(dict(step=step, event="cancel", reason="target_terminal"))

        if plan is not None:
            # A selected store is prepared on one call and committed on the
            # next.  Recheck request identity, physical blocks and funding.
            phase = "commit"
            victim = scheduler.requests.get(plan.victim.request_id)
            target = scheduler.requests.get(plan.target.request_id)
            reason = commit_reason(plan, step, view(victim), view(target),
                                   pool.get_num_free_blocks(), None,
                                   save_enabled=False)
            if reason == "READY":
                rs = cs._req_status.get(victim.request_id)
                pending_flush = set(rs.transfer_jobs) if rs else set()
                if any(j not in cs._jobs or not cs._jobs[j].is_store
                       for j in pending_flush):
                    reason = "CANCEL_VICTIM_PENDING_LOAD"
            data["events"].append(dict(step=step, event="commit_check",
                                       target=plan.target.request_id,
                                       victim=plan.victim.request_id, reason=reason))
            if reason == "READY":
                scheduler.running.remove(victim)
                scheduler._preempt_request(victim, timestamp)
                preempted.append(victim)
                scheduler._rotation_forced_count = 1
                committed_victim_id = victim.request_id
                put_waiting_first(target.request_id)
                data["applied_rotations"] += 1
                plan = None
                return
            else:
                policy.release_active()
                phase = None
                if reason in ("CANCEL_TARGET_CHANGED", "CANCEL_REQUEST_FINISHED"):
                    excluded_targets.add(plan.target.request_id)
                if reason in ("CANCEL_BLOCK_OWNERSHIP_CHANGED",
                              "CANCEL_VICTIM_PENDING_LOAD", "CANCEL_VICTIM_CHANGED"):
                    excluded_victims.add(plan.victim.request_id)
            plan = None

        if policy.active_target is not None:
            put_waiting_first(policy.active_target)
            return

        # A failed pair is not a head-of-line stop.  There are finitely many
        # targets/victims, so these exclusions bound the local scan.
        while len(excluded_targets) <= len(rows) and len(excluded_victims) <= len(rows):
            intent = policy.propose(pool.get_num_free_blocks(),
                                    scheduler.max_num_running_reqs-len(scheduler.running),
                                    excluded_target_ids=excluded_targets,
                                    excluded_victim_ids=excluded_victims)
            if intent is None:
                return
            target = scheduler.requests.get(intent.target_id)
            if target is None or target.status.name != "PREEMPTED":
                policy.reject(intent, "target_not_preempted")
                excluded_targets.add(intent.target_id)
                continue
            if intent.victim_id is None:
                if not put_waiting_first(target.request_id):
                    policy.reject(intent, "target_not_waiting")
                    excluded_targets.add(intent.target_id)
                    continue
                policy.accept(intent)
                last_output[target.request_id] = target.num_output_tokens
                data["free_priorities"] += 1
                data["events"].append(dict(step=step, event="free_priority",
                                           target=target.request_id))
                return
            victim = scheduler.requests.get(intent.victim_id)
            try:
                candidate = prepare(step, view(victim), view(target),
                                    pool.get_num_free_blocks())
            except ValueError as exc:
                policy.reject(intent, str(exc))
                excluded_victims.add(intent.victim_id)
                data["events"].append(dict(step=step, event="prepare_rejected",
                                           target=intent.target_id,
                                           victim=intent.victim_id, reason=str(exc)))
                continue
            plan = candidate
            phase = "prepare"
            policy.accept(intent)
            last_output[target.request_id] = target.num_output_tokens
            data["events"].append(dict(step=step, event="prepare",
                                       target=target.request_id,
                                       victim=victim.request_id,
                                       saved_tokens=plan.saved_tokens))
            return

    def hold(_request):
        # Let native allocation handle peer growth and natural preemption.
        # There is no blanket reservation of every running request's next block.
        return False

    def schedule(*args, **kwargs):
        nonlocal step, phase, plan
        try:
            result = native(*args, **kwargs)
            active_for_call = policy.active_target
            meta = result.kv_connector_metadata
            preempted_ids = set(result.preempted_req_ids or ())
            if phase == "prepare":
                # Native may preempt peers while preparing.  It must really
                # execute the victim once for the staged store to exist.
                if plan.victim.request_id in preempted_ids:
                    data["events"].append(dict(step=step, event="cancel",
                                               reason="prepare_victim_preempted",
                                               victim=plan.victim.request_id))
                    plan = None
                    policy.release_active()
                elif result.num_scheduled_tokens.get(plan.victim.request_id) != 1:
                    data["events"].append(dict(step=step, event="cancel",
                                               reason="prepare_not_executed",
                                               victim=plan.victim.request_id))
                    plan = None
                    policy.release_active()
                else:
                    rs = cs._req_status.get(plan.victim.request_id)
                    if rs is None:
                        raise RuntimeError("selected victim has no connector state")
                    delta = inspect_store_delta(plan, meta, rs, cs._jobs)
                    data["events"].append(dict(step=step, event="store_delta", **delta))
            elif phase == "commit":
                if plan is not None:
                    raise RuntimeError("commit plan was not cleared")
                if scheduler._rotation_forced_count:
                    forced = set(result.preempted_req_ids or ())
                    if committed_victim_id not in forced:
                        raise RuntimeError("forced native preemption notification absent")
                    if not pending_flush <= set(meta.jobs_to_flush):
                        raise RuntimeError("native flush omitted selected stores")
            scheduled = {rid: count for rid, count in result.num_scheduled_tokens.items()
                         if count > 0}
            if active_for_call is not None:
                target = scheduler.requests.get(active_for_call)
                if (target is not None and target.status.name == "WAITING_FOR_REMOTE_KVS"
                        and active_for_call in scheduled):
                    raise RuntimeError("pending load was counted as computation")
            forced_ids = ({committed_victim_id} if committed_victim_id is not None
                          and scheduler._rotation_forced_count else set())
            policy.feedback(scheduled, preempted=preempted_ids, terminal=(),
                            forced_preempted=forced_ids)
            if (active_for_call is not None and active_for_call == policy.active_target
                    and active_for_call not in scheduled and phase != "prepare"):
                target = scheduler.requests.get(active_for_call)
                if target is None or target.status.name not in (
                        "RUNNING", "WAITING_FOR_REMOTE_KVS"):
                    policy.release_active()
                    data["events"].append(dict(step=step, event="cancel",
                                               reason="no_progress_or_load",
                                               target=active_for_call))
            if (meta.store_jobs or meta.load_jobs or meta.jobs_to_flush or phase):
                data["events"].append(dict(step=step, event="metadata",
                                           stores=sorted(meta.store_jobs),
                                           loads=sorted(meta.load_jobs),
                                           flush=sorted(meta.jobs_to_flush)))
            if policy.last_noop_reason is not None:
                counts = data["no_op_counts"]
                reason = policy.last_noop_reason
                counts[reason] = counts.get(reason, 0) + 1
            step += 1
            phase = None
            data["status"] = "EXECUTING"
            return result
        except Exception as exc:
            data.update(status="ERROR", error=repr(exc))
            raise

    scheduler._rotation_begin = begin
    scheduler._rotation_hold = hold
    scheduler._rotation_target = None
    scheduler._rotation_forced_count = 0
    scheduler.schedule = schedule
    cs._calc_num_offloadable_tokens = selected_store

    def uninstall():
        for key in ("schedule",) + hooks:
            delattr(scheduler, key)
        if hadcalc:
            cs._calc_num_offloadable_tokens = oldcalc
        else:
            delattr(cs, "_calc_num_offloadable_tokens")
        if data["status"] != "ERROR":
            data["status"] = ("DRAINED" if not scheduler.requests
                              else "UNINSTALLED_WITH_PENDING_REQUESTS")
        data["schedule_calls"] = step
        data["censor_counts"] = dict(sorted(policy.censor_counts.items()))
        data["last_no_op_reason"] = policy.last_noop_reason
        data["forced_preempted_last_call"] = policy.last_forced_preempted
        data["natural_preempted_last_call"] = policy.last_natural_preempted
        return data

    return data, uninstall


def install(scheduler, *, vllm_config, block_size, threshold=200, quantum=10):
    """Install on a drained synchronous vLLM 0.26 scheduler before observers."""
    manager = scheduler.kv_cache_manager
    connector = scheduler.connector
    if type(connector).__name__ != "OffloadingConnector":
        raise ValueError("native OffloadingConnector required")
    cs = connector.connector_scheduler
    if (vllm_config.scheduler_config.async_scheduling
            or vllm_config.speculative_config is not None
            or manager.enable_caching or manager.num_kv_cache_groups != 1
            or manager.use_eagle or manager.watermark_blocks
            or scheduler.num_lookahead_tokens or scheduler.num_spec_tokens
            or scheduler.dcp_world_size != 1 or scheduler.pcp_world_size != 1
            or scheduler.use_v2_model_runner or scheduler.defer_block_free
            or scheduler.ec_connector is not None
            or scheduler.policy.name != "FCFS" or scheduler.is_encoder_decoder
            or not scheduler.scheduler_reserve_full_isl
            or cs.config.offload_prompt_only or cs.config.blocks_per_chunk != 1
            or cs.config.num_workers != 1):
        raise ValueError("unsupported execution or offload configuration")
    singles = manager.coordinator.single_type_managers
    if (len(singles) != 1 or type(singles[0]).__name__ != "FullAttentionManager"
            or type(manager.coordinator).__name__ != "KVCacheCoordinatorNoPrefixCache"
            or singles[0].block_pool is not manager.block_pool):
        raise ValueError("requires unshared full-attention blocks")
    if scheduler.requests or "schedule" in vars(scheduler):
        raise ValueError("install before capture on a drained scheduler")
    original = scheduler.schedule
    path = inspect.getsourcefile(type(scheduler))
    if path is None:
        raise ValueError("scheduler source unavailable")
    with open(path, "rb") as f:
        source = f.read()
    if hashlib.sha256(source).hexdigest() != SCHEDULER_SHA256:
        raise ValueError("native scheduler source drift")
    namespace = dict(original.__func__.__globals__)
    exec(compile(patched_schedule_tree(source.decode()), path, "exec"), namespace)
    native = MethodType(namespace["schedule"], scheduler)
    return _attach(scheduler, native, cs, block_size=block_size,
                   policy=LTRFairPolicy(threshold=threshold, quantum=quantum))


def install_on_engine(engine, *, threshold=200, quantum=10):
    """Use the existing in-process LLMEngine scheduler and cache configuration."""
    return install(engine.engine_core.engine_core.scheduler,
                   vllm_config=engine.vllm_config,
                   block_size=engine.vllm_config.cache_config.block_size,
                   threshold=threshold, quantum=quantum)
