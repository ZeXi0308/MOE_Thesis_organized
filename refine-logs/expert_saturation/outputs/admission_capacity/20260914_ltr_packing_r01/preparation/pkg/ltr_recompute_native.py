"""LTR 200/10 counters + FCFS component on a shared native recompute backend.

Not full LTR: no predictor or CPU SWAP. The greedy feasible-set planner below
is our integration rule, shared by boost=True/False. It reserves each selected
request's current history (not future EOS growth), may skip infeasible requests,
and retains unselected residents. Reservations expire every scheduling step;
there is no extra first-output, cooldown, near-EOS, or residency protection.
Default fit_scan skips an infeasible candidate. Optional rank_prefix stops at
that candidate: an LTR prefix-stop component on this shared recompute backend,
not the paper's complete ranking predictor or original SWAP implementation.
Pinned native code performs allocation, chunked recompute, and worker payloads.
The first four fit-scan boost-off/on episodes ran natively; rank-prefix GPU
execution is UNRUN. CPU allocator fixtures only qualify interfaces.
"""
import ast
from dataclasses import dataclass
import hashlib
import inspect
import time
from types import MethodType

from recovery_service_components import LTRCounters
from rotation_native import SCHEDULER_SHA256


@dataclass(frozen=True)
class Candidate:
    request_id: str
    priority: int
    arrival: float
    resident: bool
    owned_blocks: int
    history_blocks: int
    pending_tokens: int


@dataclass
class Plan:
    tokens: dict[str, int]
    victims: list[str]
    free_after_reservation: int


def plan(rows, free_blocks, token_budget, resident_limit, chunk_limit=0, *, packing="fit_scan"):
    """Greedy priority packing; only lower-priority residents can fund a choice."""
    if packing not in ("fit_scan", "rank_prefix"):
        raise ValueError("unsupported packing policy")
    if min(free_blocks, token_budget, chunk_limit) < 0 or resident_limit <= 0:
        raise ValueError("invalid resource budget")
    ordered = sorted(rows, key=lambda r: (r.priority, r.arrival, r.request_id))
    if len({r.request_id for r in ordered}) != len(ordered):
        raise ValueError("duplicate request identity")
    if any(min(r.owned_blocks, r.history_blocks, r.pending_tokens) < 0 for r in ordered):
        raise ValueError("invalid request resource state")
    selected, victims, residents = {}, [], sum(r.resident for r in ordered)
    for i, row in enumerate(ordered):
        if not token_budget:
            break
        if row.request_id in victims or not row.pending_tokens:
            continue
        amount = min(row.pending_tokens, token_budget, chunk_limit or token_budget)
        need = max(0, row.history_blocks - row.owned_blocks)
        trial, released = [], 0
        for victim in reversed(ordered[i + 1:]):
            if (free_blocks + released >= need
                    and residents + int(not row.resident) - len(trial) <= resident_limit):
                break
            if victim.resident and victim.request_id not in victims:
                trial.append(victim.request_id)
                released += victim.owned_blocks
        if (free_blocks + released < need
                or residents + int(not row.resident) - len(trial) > resident_limit):
            if packing == "rank_prefix":
                break  # Trial victims and reservations have not been committed.
            continue  # Failed trial has no eviction or budget side effects.
        victims.extend(trial)
        residents += int(not row.resident) - len(trial)
        free_blocks += released - need
        selected[row.request_id] = amount
        token_budget -= amount
    return Plan(selected, victims, free_blocks)


def patched_schedule_tree(source):
    cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == "Scheduler")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "schedule")
    hits = [0] * 5

    class Patch(ast.NodeTransformer):
        def visit_Assign(self, node):
            text = ast.unparse(node)
            if text == "scheduled_timestamp = time.monotonic()":
                hits[0] += 1
                return [node] + ast.parse("self._ltr_begin(preempted_reqs, scheduled_timestamp)").body
            if text == "request = request_queue.peek_request()":
                hits[1] += 1
                return [node] + ast.parse("if request.request_id not in self._ltr_tokens:\n    break").body
            if isinstance(node.value, ast.Call) and ast.unparse(node.value.func) == "self.kv_cache_manager.allocate_slots":
                hits[2] += 1
                return (ast.parse("num_new_tokens = self._ltr_tokens[request.request_id]").body
                        + [node] + ast.parse("if new_blocks is None:\n    raise RuntimeError('LTR plan allocation failed')").body)
            return node

        def visit_While(self, node):
            self.generic_visit(node)
            if ast.unparse(node.test) == "req_index < len(self.running) and token_budget > 0":
                hits[3] += 1
                node.body[1:1] = ast.parse("if request.request_id not in self._ltr_tokens:\n    req_index += 1\n    continue").body
            return node

        def visit_If(self, node):
            self.generic_visit(node)
            if ast.unparse(node.test) == "not preempted_reqs and self._pause_state == PauseState.UNPAUSED":
                hits[4] += 1
                node.test = ast.parse("len(preempted_reqs) == self._ltr_forced_count and self._pause_state == PauseState.UNPAUSED", mode="eval").body
            return node

    Patch().visit(fn)
    if hits != [1, 1, 2, 1, 1]:
        raise ValueError(f"unsupported scheduler structure: {hits}")
    return ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))


def install(scheduler, *, vllm_config, block_size, boost=True, threshold=200, quantum=10, packing="fit_scan"):
    """Install before observers on a drained pinned synchronous text scheduler."""
    if packing not in ("fit_scan", "rank_prefix"):
        raise ValueError("unsupported packing policy")
    manager, config = scheduler.kv_cache_manager, scheduler.scheduler_config
    pool = manager.block_pool
    singles = manager.coordinator.single_type_managers
    if (scheduler.requests or "schedule" in vars(scheduler) or block_size <= 0
            or config.async_scheduling or vllm_config.speculative_config is not None
            or manager.enable_caching or manager.num_kv_cache_groups != 1
            or manager.use_eagle or manager.watermark_blocks or scheduler.lora_config
            or scheduler.num_lookahead_tokens or scheduler.num_spec_tokens
            or scheduler.dcp_world_size != 1 or scheduler.pcp_world_size != 1
            or scheduler.use_v2_model_runner or scheduler.defer_block_free or scheduler.use_pp
            or scheduler.connector is not None or scheduler.ec_connector is not None
            or scheduler.is_encoder_decoder or scheduler.need_mamba_block_aligned_split
            or scheduler.policy.name != "FCFS" or not scheduler.scheduler_reserve_full_isl
            or not config.enable_chunked_prefill or scheduler.num_sampled_tokens_per_step != 1
            or len(singles) != 1 or type(singles[0]).__name__ != "FullAttentionManager"
            or type(manager.coordinator).__name__ != "KVCacheCoordinatorNoPrefixCache"):
        raise ValueError("requires drained synchronous unshared full-attention FCFS recompute")
    original = scheduler.schedule
    source = inspect.getsourcefile(type(scheduler))
    with open(source, "rb") as handle:
        data = handle.read()
    if hashlib.sha256(data).hexdigest() != SCHEDULER_SHA256:
        raise ValueError("native scheduler source changed")
    namespace = dict(original.__func__.__globals__)
    exec(compile(patched_schedule_tree(data.decode()), source, "exec"), namespace)
    native = MethodType(namespace["schedule"], scheduler)
    owned, counters, decisions = singles[0].req_to_blocks, LTRCounters(threshold, quantum), []
    current = None

    def begin(preempted, timestamp):
        for rid in current.victims:
            request = scheduler.requests[rid]
            scheduler.running.remove(request)
            scheduler._preempt_request(request, timestamp)
            preempted.append(request)
        scheduler._ltr_forced_count = len(current.victims)
        positions = {rid: i for i, rid in enumerate(current.tokens)}
        scheduler.running.sort(key=lambda r: positions.get(r.request_id, len(positions)))
        waiting = {r.request_id: r for r in scheduler.waiting}
        for rid in reversed(current.tokens):
            if rid in waiting:
                scheduler.waiting.remove_request(waiting[rid])
                scheduler.waiting.prepend_request(waiting[rid])

    def schedule(throttle_prefills=False):
        nonlocal current
        started = time.perf_counter()
        if (throttle_prefills or scheduler._pause_state.name != "UNPAUSED"
                or scheduler.skipped_waiting or scheduler.num_waiting_for_streaming_input):
            raise ValueError("unsupported paused/throttled/blocked request state")
        live = scheduler.requests
        if (set(live) != {r.request_id for r in list(scheduler.running) + list(scheduler.waiting)}
                or any(r.has_encoder_inputs or r.num_output_placeholders or r.num_in_flight_tokens
                       or r.spec_token_ids or r.status.name not in ("WAITING", "RUNNING", "PREEMPTED")
                       or r.num_tokens >= scheduler.max_model_len for r in live.values())):
            raise ValueError("unsupported live request state")
        priorities = counters.begin_schedule(live)
        rows = [Candidate(rid, priorities[rid] if boost else 0, r.arrival_time,
                          r.status.name == "RUNNING", len(owned.get(rid, ())),
                          (r.num_tokens + block_size - 1) // block_size,
                          r.num_tokens - r.num_computed_tokens) for rid, r in live.items()]
        current = plan(rows, pool.get_num_free_blocks(), scheduler.max_num_scheduled_tokens,
                       scheduler.max_num_running_reqs, config.long_prefill_token_threshold, packing=packing)
        scheduler._ltr_tokens = current.tokens
        record = dict(step=len(decisions), status="ATTEMPTED", boost=boost, packing=packing,
                      free_before=pool.get_num_free_blocks(),
                      priorities={r.request_id: r.priority for r in rows},
                      tokens=dict(current.tokens), victims=list(current.victims),
                      free_after_reservation=current.free_after_reservation,
                      counter_state_before={rid: dict(idle=s.idle, priority=s.priority,
                          quantum_remaining=s.quantum_remaining) for rid, s in counters.states.items()},
                      decision_seconds=time.perf_counter()-started)
        decisions.append(record)
        try:
            result = native(throttle_prefills=False)
            if (result.num_scheduled_tokens != current.tokens
                    or set(result.preempted_req_ids or ()) != set(current.victims)):
                raise RuntimeError("native action differs from LTR component plan")
            remaining = sum(max(0, (live[rid].num_tokens + block_size - 1) // block_size
                                - len(owned.get(rid, ()))) for rid in current.tokens)
            if pool.get_num_free_blocks() < remaining:
                raise RuntimeError("selected histories lost their resource reservation")
            counters.after_schedule(result.num_scheduled_tokens)
            record.update(status="APPLIED", actual_scheduled=dict(result.num_scheduled_tokens),
                          free_after=pool.get_num_free_blocks(), remaining_reserved_blocks=remaining)
            return result
        except Exception as error:
            record.update(status="ERROR", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            record['wrapped_schedule_seconds'] = time.perf_counter()-started

    scheduler._ltr_begin, scheduler.schedule = begin, schedule

    def uninstall():
        for name in ("schedule", "_ltr_begin", "_ltr_tokens", "_ltr_forced_count"):
            if name in vars(scheduler):
                delattr(scheduler, name)

    return decisions, counters, uninstall
