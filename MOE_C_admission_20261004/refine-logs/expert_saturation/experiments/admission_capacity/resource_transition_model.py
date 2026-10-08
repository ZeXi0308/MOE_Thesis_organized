"""CPU structural accounting only; no serving, latency, or performance simulator.

Observed events carry cache occupancy separately from returned output tokens.
The KV model assumes exclusive, non-speculative per-request blocks: no shared
prefix blocks, lookahead allocation, or hidden reservations. Supply the actual
usable block pool, never gpu_memory_utilization. Enumeration reserves one more
cached token per active decode; native preemption may be required if none fit.
It neither predicts completion nor executes the resulting action.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import combinations, product
from math import expm1, isfinite, log1p
from typing import Iterable, Mapping

Expert = tuple[int, int]
STATUSES = {"waiting", "prefill", "decode", "recovering", "completed"}


def blocks_for(tokens: int, block_size: int) -> int:
    if tokens < 0 or block_size <= 0:
        raise ValueError("invalid token count or block size")
    return (tokens + block_size - 1) // block_size


@dataclass(frozen=True)
class Request:
    request_id: str
    prompt_tokens: int
    arrival_ms: float = 0.0
    status: str = "waiting"
    prompt_processed: int = 0
    output_observed: int = 0
    cached_tokens: int = 0
    cached_blocks: int = 0
    last_token_ms: float | None = None
    ever_admitted: bool = False
    resume_status: str | None = None


@dataclass(frozen=True)
class State:
    requests: tuple[Request, ...]
    usable_kv_blocks: int
    block_size: int
    cap_target: int
    now_ms: float = 0.0
    token_budget: int = 1024

    def __post_init__(self):
        if not isinstance(self.requests, tuple):
            raise ValueError("requests must be an immutable tuple")
        if self.usable_kv_blocks < 0 or self.block_size <= 0 or self.cap_target < 0 or self.token_budget < 0:
            raise ValueError("invalid capacity")
        if not isfinite(self.now_ms) or len({r.request_id for r in self.requests}) != len(self.requests):
            raise ValueError("invalid time or duplicate request identity")
        for r in self.requests:
            if r.status not in STATUSES or r.prompt_tokens <= 0:
                raise ValueError("invalid request")
            if not 0 <= r.prompt_processed <= r.prompt_tokens or r.output_observed < 0:
                raise ValueError("invalid logical progress")
            if not 0 <= r.cached_tokens <= r.prompt_processed + r.output_observed:
                raise ValueError("cache exceeds known logical history")
            if r.cached_blocks != blocks_for(r.cached_tokens, self.block_size):
                raise ValueError("unsupported block sharing/reservation or wrong rounding")
            if not isfinite(r.arrival_ms) or (r.last_token_ms is not None and
                    (not isfinite(r.last_token_ms) or not r.arrival_ms <= r.last_token_ms <= self.now_ms)):
                raise ValueError("invalid request time")
            if r.status in {"waiting", "completed"} and r.cached_tokens:
                raise ValueError("waiting/completed request owns KV")
            if r.status in {"prefill", "decode", "recovering"} and not r.ever_admitted:
                raise ValueError("active request must have admission history")
            if r.status == "decode" and r.prompt_processed != r.prompt_tokens:
                raise ValueError("decode before prefill completion")
            if r.status == "recovering" and r.resume_status not in {"prefill", "decode"}:
                raise ValueError("missing native recovery destination")
        if self.used_kv_blocks > self.usable_kv_blocks:
            raise ValueError("actual KV pool exceeded")

    @property
    def used_kv_blocks(self) -> int:
        return sum(r.cached_blocks for r in self.requests)

    @property
    def active_count(self) -> int:
        # Conservative logical admission reservations, NOT native running length:
        # recovery retains its slot, and lowering cap never revokes a reservation.
        return sum(r.status in {"prefill", "decode", "recovering"} for r in self.requests)


@dataclass(frozen=True)
class Event:
    kind: str
    request_id: str
    at_ms: float
    tokens: int = 0
    cached_tokens_after: int | None = None
    resumed: bool = False


def reduce_events(state: State, events: Iterable[Event]) -> State:
    """Replay observed engine events, including explicit native recovery only.

    A decode event means one observed output, not one inferred KV allocation.
    The final prefill's first output is a separate decode observation with the
    same cached_tokens_after as the completed prompt. Recompute emits no output.
    """
    result = state
    for event in events:
        if not isfinite(event.at_ms) or event.at_ms < result.now_ms:
            raise ValueError("events must be causally ordered")
        indexed = {r.request_id: r for r in result.requests}
        if event.request_id not in indexed:
            raise ValueError("unknown request identity")
        r = indexed[event.request_id]
        if event.at_ms < r.arrival_ms:
            raise ValueError("event precedes arrival")
        if event.kind == "admit":
            if r.status != "waiting" or r.ever_admitted:
                raise ValueError("only never-admitted waiting requests are new admissions")
            r = replace(r, status="prefill", ever_admitted=True)
        elif event.kind == "prefill":
            if r.status != "prefill" or not 0 < event.tokens <= r.prompt_tokens - r.prompt_processed:
                raise ValueError("illegal prefill progress")
            processed = r.prompt_processed + event.tokens
            r = replace(r, prompt_processed=processed, cached_tokens=r.cached_tokens + event.tokens,
                        status="decode" if processed == r.prompt_tokens else "prefill")
        elif event.kind == "decode":
            if r.status != "decode" or event.cached_tokens_after is None:
                raise ValueError("decode requires active decode and observed cache occupancy")
            if not r.cached_tokens <= event.cached_tokens_after <= r.cached_tokens + 1:
                raise ValueError("decode KV growth must be zero or one token")
            r = replace(r, output_observed=r.output_observed + 1,
                        cached_tokens=event.cached_tokens_after, last_token_ms=event.at_ms)
        elif event.kind == "preempt":
            if r.status not in {"prefill", "decode"}:
                raise ValueError("native preemption requires active execution")
            r = replace(r, status="recovering", resume_status=r.status, cached_tokens=0)
        elif event.kind == "recompute":
            if r.status != "recovering" or event.tokens <= 0:
                raise ValueError("recompute requires explicit native recovery")
            cached = r.cached_tokens + event.tokens
            required = r.prompt_processed + max(0, r.output_observed - 1)
            if event.resumed and cached < required:
                raise ValueError("cannot resume before rebuilding required history")
            r = replace(r, cached_tokens=cached,
                        status=r.resume_status if event.resumed else "recovering",
                        resume_status=None if event.resumed else r.resume_status)
        elif event.kind == "complete":
            if r.status in {"waiting", "completed"}:
                raise ValueError("completion requires admitted unfinished request")
            r = replace(r, status="completed", cached_tokens=0, resume_status=None)
        else:
            raise ValueError("unknown event kind")
        r = replace(r, cached_blocks=blocks_for(r.cached_tokens, result.block_size))
        result = replace(result, now_ms=event.at_ms,
                         requests=tuple(r if old.request_id == r.request_id else old for old in result.requests))
    return result


@dataclass(frozen=True)
class Action:
    admissions: tuple[str, ...]
    prefill_chunks: tuple[tuple[str, int], ...]
    decode_requests: tuple[str, ...]


def enumerate_actions(state: State, chunk_options=(0, 512, 1024), max_new: int = 1) -> tuple[Action, ...]:
    """Small-instance plans; no future lengths, routes, completion or recovery decisions.

    Existing decodes all receive a slot, including after cap reduction. If their
    next KV growth cannot fit, no plan is returned: resolve via native events.
    Existing prefills can receive zero work; admitted new requests must progress.
    """
    if max_new < 0 or not chunk_options or any(c < 0 for c in chunk_options):
        raise ValueError("invalid finite action grid")
    decodes = tuple(r for r in state.requests if r.status == "decode")
    base = state.used_kv_blocks + sum(
        blocks_for(r.cached_tokens + 1, state.block_size) - r.cached_blocks for r in decodes)
    candidates = tuple(r for r in state.requests if r.status == "waiting"
                       and not r.ever_admitted and r.arrival_ms <= state.now_ms)
    limit = min(max_new, len(candidates), max(0, state.cap_target - state.active_count))
    actions = []
    for count in range(limit + 1):
        for newcomers in combinations(candidates, count):
            prefills = tuple(r for r in state.requests if r.status == "prefill") + newcomers
            choices = [tuple(sorted({min(c, r.prompt_tokens - r.prompt_processed)
                                     for c in chunk_options if c > 0 or r not in newcomers})) for r in prefills]
            for chunks in product(*choices):
                if len(decodes) + sum(chunks) > state.token_budget:
                    continue
                needed = base + sum(blocks_for(r.cached_tokens + c, state.block_size) - r.cached_blocks
                                    for r, c in zip(prefills, chunks))
                if needed <= state.usable_kv_blocks:
                    actions.append(Action(tuple(r.request_id for r in newcomers),
                                          tuple((r.request_id, c) for r, c in zip(prefills, chunks)),
                                          tuple(r.request_id for r in decodes)))
    return tuple(actions)


def unique_load_bytes(routes: Iterable[Iterable[Expert]], resident: Iterable[Expert],
                      weights: Mapping[Expert, int]) -> int:
    """POST_ROUTER/OFFLINE lower bound: one load per unique missing expert.

    These known-layer routes must not be treated as future online action inputs.
    Reloads caused by capacity or ordering are excluded here.
    """
    required = set().union(*(set(route) for route in routes))
    if any(weights[e] <= 0 for e in required):
        raise ValueError("expert weights must be positive bytes")
    return sum(weights[e] for e in required - set(resident))


@dataclass(frozen=True)
class EstimatedDemand:
    experts: frozenset[Expert]
    information_available_ms: float


def estimated_unique_load_bytes(demand: EstimatedDemand, decision_ms: float,
                                resident: Iterable[Expert], weights: Mapping[Expert, int]) -> int:
    """Estimate only; caller must supply causal predictions, never realized future routes."""
    if not isfinite(demand.information_available_ms) or not isfinite(decision_ms):
        raise ValueError("invalid information timestamp")
    if demand.information_available_ms > decision_ms:
        raise ValueError("future information is unavailable at decision time")
    return unique_load_bytes((demand.experts,), resident, weights)


@dataclass(frozen=True)
class CacheTrace:
    unique_lower_bound_bytes: int
    total_load_bytes: int
    reload_bytes: int
    final_lru: tuple[Expert, ...]


def lru_dry_run(accesses: Iterable[Expert], initial_lru: tuple[Expert, ...],
                capacity_bytes: int, weights: Mapping[Expert, int]) -> CacheTrace:
    """POST_ROUTER/OFFLINE sequential cache accounting, without overlap or timing."""
    accesses = tuple(accesses)
    if len(set(initial_lru)) != len(initial_lru) or capacity_bytes < 0:
        raise ValueError("invalid initial cache")
    if any(weights[e] <= 0 or weights[e] > capacity_bytes for e in set(accesses) | set(initial_lru)):
        raise ValueError("invalid weight or expert cannot fit")
    lru = list(initial_lru)
    used = sum(weights[e] for e in lru)
    if used > capacity_bytes:
        raise ValueError("initial cache exceeds capacity")
    total = 0
    for expert in accesses:
        if expert in lru:
            lru.remove(expert)
        else:
            while used + weights[expert] > capacity_bytes:
                used -= weights[lru.pop(0)]
            total += weights[expert]
            used += weights[expert]
        lru.append(expert)
    lower = unique_load_bytes((accesses,), initial_lru, weights)
    return CacheTrace(lower, total, total - lower, tuple(lru))


@dataclass(frozen=True)
class GroupedSlotState:
    slot_to_expert: tuple[int, ...]
    lru_tick: tuple[int, ...]
    lru_clock: int

    def __post_init__(self):
        if (not isinstance(self.slot_to_expert, tuple) or not isinstance(self.lru_tick, tuple)
                or not self.slot_to_expert or len(self.slot_to_expert) != len(self.lru_tick)
                or type(self.lru_clock) is not int or self.lru_clock < 0
                or any(type(e) is not int or e < -1 for e in self.slot_to_expert)
                or any(type(t) is not int or not 0 <= t <= self.lru_clock for t in self.lru_tick)):
            raise ValueError("invalid immutable grouped slot state")
        residents = [e for e in self.slot_to_expert if e != -1]
        if len(set(residents)) != len(residents):
            raise ValueError("duplicate resident expert")


@dataclass(frozen=True)
class GroupedLoad:
    required_experts: tuple[int, ...]
    loaded_experts: tuple[int, ...]
    evicted_experts: tuple[int, ...]
    loaded_slots: tuple[int, ...]
    load_bytes: int
    state_after: GroupedSlotState


@dataclass(frozen=True)
class GroupedCacheTrace:
    groups: tuple[GroupedLoad, ...]
    final_state: GroupedSlotState

    @property
    def total_load_bytes(self) -> int:
        return sum(group.load_bytes for group in self.groups)


def grouped_slot_dry_run(initial: GroupedSlotState, required_groups: Iterable[Iterable[int]],
                         weight_bytes: Mapping[int, int]) -> GroupedCacheTrace:
    """POST_ROUTER/OFFLINE WiSP demand-load accounting for one fixed-cap layer.

    Groups and initial slot/tick/clock state must be supplied, not predicted by
    this function. Mirrors upstream integer-set missing order, whole-group
    protection, stable slot ties and free_slots.pop(); prefetch/resize are off.
    No copies, timing, live paging or action-conditioned future routes execute.
    """
    state, groups = initial, []
    for required in required_groups:
        required = tuple(required)
        if any(type(e) is not int or e < 0 for e in required):
            raise ValueError("required experts must be nonnegative integer IDs")
        # torch.unique produces sorted IDs; upstream then builds this integer set.
        needed = set(int(e) for e in sorted(set(required)))
        if any(type(weight_bytes[e]) is not int or weight_bytes[e] <= 0 for e in needed):
            raise ValueError("expert weights must be positive integer bytes")
        slots, ticks = list(state.slot_to_expert), list(state.lru_tick)
        resident = {e: s for s, e in enumerate(slots) if e != -1}
        missing = [e for e in needed if e not in resident]
        free = [s for s, e in enumerate(slots) if e == -1]
        # Equal ticks preserve ascending physical slot order; expert recency alone
        # loses this tie and can change later loads when free.pop() assigns slots.
        victims = sorted(((ticks[s], s) for s, e in enumerate(slots)
                          if e != -1 and e not in needed), key=lambda item: item[0])
        deficit = max(0, len(missing) - len(free))
        if deficit > len(victims):
            raise RuntimeError("group working set exceeds slot capacity")
        evicted, loaded_slots = [], []
        for _, slot in victims[:deficit]:
            evicted.append(slots[slot])
            resident.pop(slots[slot])
            slots[slot] = -1
            free.append(slot)
        clock = state.lru_clock + 1
        for expert in missing:
            slot = free.pop()
            slots[slot], resident[expert] = expert, slot
            loaded_slots.append(slot)
        for expert in needed:
            ticks[resident[expert]] = clock
        state = GroupedSlotState(tuple(slots), tuple(ticks), clock)
        groups.append(GroupedLoad(tuple(sorted(needed)), tuple(missing), tuple(evicted),
                                  tuple(loaded_slots), sum(weight_bytes[e] for e in missing), state))
    return GroupedCacheTrace(tuple(groups), state)


def uniform_expected_union(experts: int, top_k: int, tokens: int) -> float:
    """Analytical independent uniform top-k baseline; not measured route behavior."""
    if experts <= 0 or not 0 <= top_k <= experts or tokens < 0:
        raise ValueError("invalid uniform routing parameters")
    if tokens == 0 or top_k == 0:
        return 0.0
    if top_k == experts:
        return float(experts)
    return -experts * expm1(tokens * log1p(-top_k / experts))


def exposed_transfer_seconds(transfer_seconds: float, overlap_seconds: float) -> float:
    """Use supplied measured transfer duration and its proven hidden intersection.

    This does not add profiler spans or infer overlap from unrelated GPU work.
    """
    if any(not isfinite(x) or x < 0 for x in (transfer_seconds, overlap_seconds)):
        raise ValueError("invalid measured duration")
    if overlap_seconds > transfer_seconds:
        raise ValueError("hidden intersection exceeds transfer duration")
    return max(0.0, transfer_seconds - overlap_seconds)
