"""Retained-KV completion headroom probe for the pinned synchronous vLLM 0.26.

Only the already-admitted, pure-decode closed cohort is controlled. No future
routes or measured completion times enter decisions. CPU checks certify block
arithmetic and adapter structure, not GPU correctness or request performance.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
import inspect
import time
from types import MethodType

SCHEDULER_SHA256 = "2ed2a550b6558b2495eda845a97ae38bcf0225027b9e25fbf00fc3880c1d3941"


@dataclass(frozen=True)
class DecodeState:
    request_id: str
    computed: int
    prompt: int
    output: int
    max_output: int
    allocated: int

    def next_blocks(self, block_size):
        return max(0, (self.computed + block_size) // block_size - self.allocated)

    def remaining_blocks(self, block_size):
        # The final emitted token is not fed through the model again.
        terminal = self.prompt + self.max_output - 1
        return max(0, (terminal + block_size - 1) // block_size - self.allocated)


class CompletionHeadroom:
    def __init__(self, block_size):
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        self.block_size = block_size
        self.leader = None

    def decide(self, rows, free):
        if free < 0 or len({r.request_id for r in rows}) != len(rows):
            raise ValueError("invalid pool or duplicate request")
        b = self.block_size
        for r in rows:
            if (r.prompt <= 0 or not 0 < r.output < r.max_output
                    or r.computed != r.prompt + r.output - 1
                    or r.allocated != (r.computed + b - 1) // b):
                raise ValueError("requires unshared, synchronous pure decode KV")
        by_id = {r.request_id: r for r in rows}
        if self.leader not in by_id:
            self.leader = None
        if not rows:
            return dict(leader=None, held=[], scheduled=[], reserve=0, free=free)
        candidate = by_id.get(self.leader) or min(
            rows, key=lambda r: r.max_output - r.output)  # Stable FCFS tie.
        reserve = candidate.remaining_blocks(b)
        costs = {r.request_id: r.next_blocks(b) for r in rows}
        # Before pressure, preserve the native order and all decode work.
        if self.leader is None and free - sum(costs.values()) >= reserve - costs[candidate.request_id]:
            return dict(leader=None, held=[], scheduled=list(by_id), reserve=reserve, free=free)
        if reserve > free:
            raise RuntimeError("completion headroom was unavailable at activation")
        self.leader = candidate.request_id
        # Reserve all remaining leader growth, including its next allocation.
        expendable = free - reserve
        held, scheduled = [], []
        for r in rows:
            cost = costs[r.request_id]
            if r.request_id == self.leader:
                scheduled.append(r.request_id)
            elif cost <= expendable:
                scheduled.append(r.request_id)
                expendable -= cost
            else:
                held.append(r.request_id)
        return dict(leader=self.leader, held=held, scheduled=scheduled,
                    reserve=reserve, free=free, unreserved_after=expendable)


def patched_schedule_tree(source):
    """Insert two skips; retain native queue, allocation, and output semantics."""
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Scheduler")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "schedule")
    matches = 0
    for node in ast.walk(fn):
        if isinstance(node, ast.While) and ast.unparse(node.test) == "req_index < len(self.running) and token_budget > 0":
            if ast.unparse(node.body[0]) != "request = self.running[req_index]":
                raise ValueError("native running loop changed")
            node.body[1:1] = ast.parse(
                "if request.request_id in self._headroom_held:\n"
                "    req_index += 1\n    continue\n").body
            matches += 1
        elif isinstance(node, ast.If) and ast.unparse(node.test) == "not preempted_reqs and self._pause_state == PauseState.UNPAUSED":
            node.test = ast.BoolOp(op=ast.And(), values=[node.test,
                ast.parse("not self._headroom_closed", mode="eval").body])
            matches += 1
    if matches != 2 or fn.decorator_list:
        raise ValueError("unsupported scheduler structure")
    return ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))


def install(scheduler, *, vllm_config, block_size, expected_requests=32, mode="headroom", observer="checked"):
    """Install before capture wrappers; return decisions and an uninstall callable."""
    if mode not in ("native", "headroom"):
        raise ValueError("unknown arm")
    if observer not in ("checked", "fast"):
        raise ValueError("unknown observer")
    manager = scheduler.kv_cache_manager
    if (vllm_config.scheduler_config.async_scheduling
            or vllm_config.speculative_config is not None
            or manager.enable_caching or manager.num_kv_cache_groups != 1
            or manager.use_eagle or manager.watermark_blocks
            or scheduler.num_lookahead_tokens or scheduler.num_spec_tokens
            or scheduler.dcp_world_size != 1 or scheduler.pcp_world_size != 1
            or scheduler.use_v2_model_runner or scheduler.defer_block_free
            or scheduler.connector is not None
            or scheduler.max_num_scheduled_tokens < expected_requests):
        raise ValueError("unsupported execution or cache mode")
    if scheduler.requests or "schedule" in vars(scheduler):
        raise ValueError("install on a drained, unwrapped scheduler")
    single = manager.coordinator.single_type_managers
    if (len(single) != 1 or type(single[0]).__name__ != "FullAttentionManager"
            or type(manager.coordinator).__name__ != "KVCacheCoordinatorNoPrefixCache"
            or single[0].block_pool is not manager.block_pool):
        raise ValueError("direct block access requires unshared full attention")
    owned = single[0].req_to_blocks
    original = scheduler.schedule
    source = inspect.getsourcefile(type(scheduler))
    with open(source, "rb") as handle:
        source_bytes = handle.read()
    if hashlib.sha256(source_bytes).hexdigest() != SCHEDULER_SHA256:
        raise ValueError("installed scheduler differs from the measured source")
    namespace = dict(original.__func__.__globals__)
    exec(compile(patched_schedule_tree(source_bytes.decode()), source, "exec"), namespace)
    patched = MethodType(namespace["schedule"], scheduler)
    governor, decisions, cohort = CompletionHeadroom(block_size), [], None
    scheduler._headroom_held, scheduler._headroom_closed = set(), False

    def schedule(*args, **kwargs):
        nonlocal cohort
        started = time.perf_counter()
        running = scheduler.running
        pure = all(r.num_output_tokens > 0 and r.num_computed_tokens == r.num_tokens - 1
                   and not r.num_preemptions and not r.num_output_placeholders
                   and not r.num_in_flight_tokens and not r.spec_token_ids
                   and r.num_prompt_tokens + r.max_tokens <= scheduler.max_model_len
                   for r in running)
        if cohort is None and (len(running) == expected_requests and pure
                               and not scheduler.waiting and not scheduler.skipped_waiting):
            if observer == "fast":
                for r in running:
                    blocks = owned.get(r.request_id)
                    if blocks is None or any(b.is_null or manager.block_pool.blocks[b.block_id] is not b
                                             for b in blocks):
                        raise RuntimeError("direct block ownership is not qualified")
                    if [b.block_id for b in blocks] != manager.get_blocks(r.request_id).get_block_ids()[0]:
                        raise RuntimeError("direct block access differs from native API")
            cohort = set(scheduler.requests)
        active = cohort is not None
        scheduler._headroom_closed = active and mode == "headroom"
        decision = dict(step=len(decisions), mode=mode, observer=observer,
                        active=active, held=[], leader=None)
        before = {}
        if active and mode == "headroom":
            if not pure or not set(scheduler.requests) <= cohort:
                raise RuntimeError("closed pure-decode cohort invariant changed")
            rows = []
            for r in running:
                if observer == "fast":
                    blocks = owned.get(r.request_id)
                    if blocks is None:
                        raise RuntimeError("running request has no owned block list")
                    count = len(blocks)
                else:
                    blocks = manager.get_blocks(r.request_id).get_block_ids()
                    if len(blocks) != 1 or 0 in blocks[0]:
                        raise RuntimeError("unsupported block layout")
                    count = len(blocks[0])
                    before[r.request_id] = (r.num_computed_tokens, tuple(blocks[0]))
                rows.append(DecodeState(r.request_id, r.num_computed_tokens,
                    r.num_prompt_tokens, r.num_output_tokens, r.max_tokens, count))
            decision.update(governor.decide(rows, manager.block_pool.get_num_free_blocks()))
            if observer == "fast":
                for rid in decision["held"]:
                    ids = tuple(block.block_id for block in owned[rid])
                    if 0 in ids:
                        raise RuntimeError("held request contains null blocks")
                    before[rid] = (scheduler.requests[rid].num_computed_tokens, ids)
        scheduler._headroom_held = set(decision["held"])
        decision["decision_seconds"] = time.perf_counter() - started
        decisions.append(decision)  # Retain attempted action even on native failure.
        result = patched(*args, **kwargs)
        decision["actual_scheduled"] = dict(result.num_scheduled_tokens)
        decision["preempted"] = list(result.preempted_req_ids or [])
        if active and mode == "headroom":
            if decision["preempted"] or set(result.num_scheduled_tokens) != set(decision["scheduled"]):
                raise RuntimeError("native action differs from headroom decision")
            for rid, amount in result.num_scheduled_tokens.items():
                if amount != 1:
                    raise RuntimeError("expected one synchronous decode token")
            for rid in decision["held"]:
                r = scheduler.requests[rid]
                blocks = (tuple(block.block_id for block in owned[rid]) if observer == "fast" else
                          manager.get_blocks(rid).get_block_ids()[0])
                after = (r.num_computed_tokens, tuple(blocks))
                if after != before[rid] or r.status.name != "RUNNING":
                    raise RuntimeError("held request lost KV or progress")
            if decision["leader"] is not None:
                leader = scheduler.requests[decision["leader"]]
                allocated = (len(owned[leader.request_id]) if observer == "fast" else
                             len(manager.get_blocks(leader.request_id).get_block_ids()[0]))
                terminal = leader.num_prompt_tokens + leader.max_tokens - 1
                remaining = max(0, (terminal + block_size - 1) // block_size - allocated)
                if manager.block_pool.get_num_free_blocks() < remaining:
                    raise RuntimeError("leader completion reservation was consumed")
        decision["status"] = "APPLIED"
        return result

    scheduler.schedule = schedule

    def uninstall():
        del scheduler.schedule
        del scheduler._headroom_held
        del scheduler._headroom_closed

    return decisions, uninstall
