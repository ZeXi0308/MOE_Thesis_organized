"""CPU-only LTR-style recovery policy for the shared native selected-save backend.

The LTR priority/idle/quantum rule applies to *all* live requests.  Forced
recovery is deliberately narrower: only a boosted PREEMPTED request can be a
target.  The native adapter remains responsible for block ownership, staged
store/commit, queue, and load-completion checks.  This module sees only state
available at the beginning of a scheduler call and its actual outcome.

The original LTR implementation sorts equal-priority requests by a learned
aux_model_score.  This recovery-only component has no predictor; arrival time
and the current queue order give a deterministic, stable fallback.  It is not
a full implementation of LTR, CPU SWAP, or its predictor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True)
class Request:
    request_id: str
    status: str
    arrival_time: float
    queue_order: int
    output_tokens: int = 0
    owned_blocks: int = 0
    need_blocks: int = 0
    eligible_victim: bool = False


@dataclass
class LTRRequestState:
    idle: int = 0
    priority: int = 0
    quantum_remaining: int = 0

    @property
    def boosted(self) -> bool:
        return self.priority == -1


@dataclass(frozen=True)
class Intent:
    action: str
    target_id: str | None
    victim_id: str | None = None
    quantum_remaining: int = 0
    reason: str = ""


class LTRCounters:
    """Official LTR-style call counters, with output tokens excluded.

    observe() runs before selection: idle >= threshold re-arms even an
    already boosted request; otherwise an exhausted quantum demotes it.
    feedback() runs after native scheduling: each request with positive
    num_scheduled_tokens gets one quantum debit, independent of token count.
    A pending load with zero scheduled tokens remains idle and spends no Q.
    """

    def __init__(self, threshold: int = 200, quantum: int = 10):
        if threshold < 1 or quantum < 1:
            raise ValueError("threshold and quantum must be positive")
        self.threshold = threshold
        self.quantum = quantum
        self.states: dict[str, LTRRequestState] = {}

    def observe(self, live_ids: Iterable[str]) -> None:
        live = set(live_ids)
        for request_id in tuple(self.states):
            if request_id not in live:
                del self.states[request_id]
        for request_id in live:
            state = self.states.setdefault(request_id, LTRRequestState())
            if state.idle >= self.threshold:
                state.priority = -1
                state.idle = 0
                state.quantum_remaining = self.quantum
            elif state.boosted and state.quantum_remaining <= 0:
                state.priority = 0
                state.quantum_remaining = 0

    def feedback(self, positive_ids: Iterable[str]) -> None:
        positive = set(positive_ids)
        unknown = positive - self.states.keys()
        if unknown:
            raise ValueError(f"positive allocation for unobserved requests: {sorted(unknown)}")
        for request_id, state in self.states.items():
            if request_id in positive:
                state.idle = 0
                if state.boosted:
                    state.quantum_remaining = max(0, state.quantum_remaining - 1)
            else:
                state.idle += 1


class LTRFairPolicy:
    """One-call observe/propose/accept/feedback protocol.

    A proposal is speculative.  The adapter calls accept() only after its
    backend-side checks or prepare have succeeded.  If the first proposal
    cannot be executed, call propose() again with that target or victim in an
    exclusion set; no target is latched until accept().  A None proposal is a
    valid no-action step, followed by the usual feedback().
    """

    def __init__(self, threshold: int = 200, quantum: int = 10):
        self.counters = LTRCounters(threshold, quantum)
        self._live: dict[str, Request] = {}
        self._proposal: Intent | None = None
        self._active_intent: Intent | None = None
        self._observed = False
        self.skipped: list[dict[str, object]] = []
        self.last_noop_reason: str | None = None
        # Counts unique target/reason observations within each call, not
        # distinct requests or a counterfactual count of feasible actions.
        self.censor_counts: dict[str, int] = {}
        self.last_forced_preempted: tuple[str, ...] = ()
        self.last_natural_preempted: tuple[str, ...] = ()

    @property
    def active_intent(self) -> Intent | None:
        return self._active_intent

    @property
    def active_target(self) -> str | None:
        return self._active_intent.target_id if self._active_intent else None

    def observe(self, live: Sequence[Request]) -> None:
        if self._observed:
            raise RuntimeError("feedback is required before the next observe")
        rows: dict[str, Request] = {}
        for row in live:
            if row.request_id in rows:
                raise ValueError(f"duplicate live request: {row.request_id}")
            if min(row.queue_order, row.output_tokens, row.owned_blocks, row.need_blocks) < 0:
                raise ValueError(f"negative request resource or order: {row.request_id}")
            if not _terminal_status(row.status):
                rows[row.request_id] = row
        self._live = rows
        self.counters.observe(rows)
        self._proposal = None
        self.skipped = []
        self.last_noop_reason = None
        if self.active_target not in rows or (self.active_target is not None
                and rows[self.active_target].status not in
                ("PREEMPTED", "WAITING_FOR_REMOTE_KVS", "RUNNING")):
            self.release_active()
        elif self.active_target is not None and not self.counters.states[self.active_target].boosted:
            self.release_active()
        self._observed = True

    def rank(self, request_id: str) -> tuple[int, float, int, str]:
        if request_id not in self._live:
            raise KeyError(request_id)
        row = self._live[request_id]
        state = self.counters.states[request_id]
        return (state.priority, row.arrival_time, row.queue_order, request_id)

    def ordered_running_ids(self) -> tuple[str, ...]:
        """Priority order to apply to RUNNING as well as recovery requests."""
        return tuple(sorted((rid for rid, row in self._live.items()
                             if row.status == "RUNNING"), key=self.rank))

    def propose(self, free_blocks: int, free_slots: int,
                excluded_target_ids: Iterable[str] = (),
                excluded_victim_ids: Iterable[str] = ()) -> Intent | None:
        if not self._observed:
            raise RuntimeError("observe must precede propose")
        if free_blocks < 0 or free_slots < 0:
            raise ValueError("negative free resource count")
        self._proposal = None
        if self.active_target is not None:
            self.last_noop_reason = "ACTIVE_TARGET"
            return None
        excluded_targets = set(excluded_target_ids)
        excluded_victims = set(excluded_victim_ids)
        targets = sorted((row for row in self._live.values()
                          if row.status == "PREEMPTED"
                          and self.counters.states[row.request_id].boosted
                          and row.request_id not in excluded_targets),
                         key=lambda row: self.rank(row.request_id))
        victims = sorted((row for row in self._live.values()
                          if row.status == "RUNNING" and row.eligible_victim
                          and row.request_id not in excluded_victims),
                         key=lambda row: (-row.output_tokens, row.queue_order,
                                          row.arrival_time, row.request_id))
        if not targets:
            self.last_noop_reason = "NO_BOOSTED_PREEMPTED"
            return None
        for target in targets:
            remaining = self.counters.states[target.request_id].quantum_remaining
            if free_blocks >= target.need_blocks and free_slots > 0:
                self._proposal = Intent("PRIORITIZE_WAITING", target.request_id,
                                        quantum_remaining=remaining)
                self.last_noop_reason = None
                return self._proposal
            # The fallback tie order is part of the effective priority order.
            # A later boosted RUNNING request can be below an earlier boosted
            # PREEMPTED target even though both have priority == -1.
            lower = [victim for victim in victims
                     if self.rank(victim.request_id) > self.rank(target.request_id)]
            for victim in victims:
                if self.rank(victim.request_id) <= self.rank(target.request_id):
                    continue
                if free_blocks + victim.owned_blocks < target.need_blocks:
                    continue
                # One victim frees one running slot.  No extra cap/cooldown or
                # output-progress restriction is imposed by the selector.
                self._proposal = Intent("PREPARE_SELECTED", target.request_id,
                                        victim.request_id, remaining)
                self.last_noop_reason = None
                return self._proposal
            aggregate = free_blocks + sum(v.owned_blocks for v in lower)
            single = any(free_blocks + v.owned_blocks >= target.need_blocks for v in lower)
            if free_slots == 0 and free_blocks >= target.need_blocks and not lower:
                reason = "SLOT_BLOCKED"
            elif not lower:
                reason = "NO_ELIGIBLE_VICTIM"
            elif aggregate >= target.need_blocks and not single:
                reason = "UNFUNDED_SINGLE_VICTIM"
            elif aggregate < target.need_blocks:
                reason = "UNFUNDED_TOTAL"
            else:
                reason = "NO_ELIGIBLE_VICTIM"
            diagnostic = dict(target=target.request_id, reason=reason,
                              need_blocks=target.need_blocks, free_blocks=free_blocks,
                              free_slots=free_slots, eligible_victims=len(lower),
                              eligible_victim_blocks=sum(v.owned_blocks for v in lower),
                              aggregate_fundable=aggregate >= target.need_blocks,
                              single_victim_fundable=single)
            if diagnostic not in self.skipped:
                self.skipped.append(diagnostic)
                self.censor_counts[reason] = self.censor_counts.get(reason, 0) + 1
            self.last_noop_reason = reason
        return None

    def accept(self, intent: Intent) -> None:
        if not self._observed or intent != self._proposal or intent.target_id is None:
            raise ValueError("only the current executable proposal may be accepted")
        if self.active_target is not None:
            raise RuntimeError("an active recovery target is already latched")
        self._active_intent = intent
        self._proposal = None

    def no_op(self, reason: str = "NO_EXECUTABLE_TARGET") -> Intent:
        """Explicitly discard an unaccepted proposal for this call."""
        if not self._observed:
            raise RuntimeError("observe must precede no_op")
        self._proposal = None
        self.last_noop_reason = reason
        return Intent("NO_OP", None, reason=reason)

    def reject(self, intent: Intent, reason: str) -> None:
        """Record a backend-censored proposal, then allow a fresh scan."""
        if not self._observed or intent != self._proposal:
            raise ValueError("only the current proposal may be rejected")
        self.skipped.append(dict(target=intent.target_id, victim=intent.victim_id,
                                 reason=reason, backend_censored=True))
        self.censor_counts[reason] = self.censor_counts.get(reason, 0) + 1
        self.last_noop_reason = reason
        self._proposal = None

    def cancel(self, intent: Intent | None = None) -> None:
        """Cancel a proposal or active backend plan without resetting LTR Q."""
        if intent is None or intent == self._active_intent:
            self.release_active()
        if intent is None or intent == self._proposal:
            self._proposal = None

    def release_active(self) -> None:
        self._active_intent = None

    def feedback(self, positive_scheduled: Mapping[str, int] | Iterable[str],
                 preempted: Iterable[str] = (), terminal: Iterable[str] = (),
                 forced_preempted: Iterable[str] = ()) -> None:
        if not self._observed:
            raise RuntimeError("observe is required before feedback")
        if isinstance(positive_scheduled, Mapping):
            if any(count < 0 for count in positive_scheduled.values()):
                raise ValueError("negative scheduled token count")
            positive_ids = {rid for rid, count in positive_scheduled.items() if count > 0}
        else:
            positive_ids = set(positive_scheduled)
        self.counters.feedback(positive_ids)
        ended = set(terminal)
        observed_preempted = set(preempted)
        forced_ids = set(forced_preempted)
        preempted_ids = observed_preempted | forced_ids
        self.last_forced_preempted = tuple(sorted(forced_ids))
        self.last_natural_preempted = tuple(sorted(observed_preempted - forced_ids))
        if self.active_target in ended or self.active_target in preempted_ids:
            self.release_active()
        for request_id in ended:
            self.counters.states.pop(request_id, None)
            self._live.pop(request_id, None)
        self._proposal = None
        self._observed = False


def _terminal_status(status: str) -> bool:
    return status.startswith("FINISHED") or status in {"CANCELLED", "CANCELED", "ABORTED", "ERROR"}
