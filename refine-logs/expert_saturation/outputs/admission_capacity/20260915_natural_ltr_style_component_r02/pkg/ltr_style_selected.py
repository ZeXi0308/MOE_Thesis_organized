"""CPU intent selector for an LTR-style component on G's selected-offload backend.

This is not a vLLM adapter or a full LTR reproduction.  It reuses the verified
LTR counters, restricts intervention to preempted recovery requests, and emits
one target/victim intent for G's existing two-stage native save path to check.
No future state enters a decision; current returned output counts may rank victims.
"""
from dataclasses import dataclass
from recovery_service_components import LTRCounters

@dataclass(frozen=True)
class Request:
    request_id: str
    arrival: float
    status: str
    remaining_blocks: int = 0
    output_tokens: int = 0
    held_blocks: int = 0
    backend_eligible: bool = False

@dataclass(frozen=True)
class Intent:
    action: str
    reason: str
    target_id: str | None = None
    victim_id: str | None = None
    quantum_remaining: int = 0

class LTRStyleSelected:
    """Counter/selection seam; the native adapter owns execution and validation."""

    def __init__(self, threshold=200, quantum=10):
        self.counters = LTRCounters(threshold, quantum)
        self.active_target = None
    def begin_step(self, requests, free_blocks, *, backend_idle=True, executable=None, free_slots=1):
        rows = list(requests)
        by_id = {r.request_id: r for r in rows}
        if len(by_id) != len(rows) or free_blocks < 0 or free_slots < 0:
            raise ValueError("invalid observed request state")
        if any(min(r.remaining_blocks, r.output_tokens, r.held_blocks) < 0 for r in rows):
            raise ValueError("negative request resource state")
        priorities = self.counters.begin_schedule(by_id)
        if (self.active_target not in by_id
                or priorities.get(self.active_target) != -1):
            self.active_target = None
        self.skipped = []
        if self.active_target is None:
            waiting = sorted((r for r in rows if r.status == "PREEMPTED"
                              and priorities[r.request_id] == -1), key=lambda r: r.arrival)
            for target in waiting:
                intent = self._candidate(target, rows, priorities, free_blocks, backend_idle, free_slots)
                reason = executable(intent) if executable and intent.action != "DEFER" else None
                if intent.action != "DEFER" and reason is None:
                    return intent  # The backend calls accept only after accepting this intent.
                self.skipped.append(dict(target=target.request_id, reason=reason or intent.reason))
            return Intent("DEFER" if waiting else "NOOP", "no executable boosted recovery request")
        target = by_id[self.active_target]
        state = self.counters.states[target.request_id]
        base = dict(target_id=target.request_id,
                    quantum_remaining=state.quantum_remaining)
        if target.status == "WAITING_FOR_REMOTE_KVS":
            return Intent("WAIT_LOAD", "pending native load; quantum is not spent", **base)
        if target.status == "RUNNING":
            return Intent("PRIORITIZE_RUNNING", "serve while LTR quantum remains", **base)
        if target.status != "PREEMPTED":
            return Intent("DEFER", "target is outside recovery action space", **base)
        return self._candidate(target, rows, priorities, free_blocks, backend_idle, free_slots)
    def _candidate(self, target, rows, priorities, free_blocks, backend_idle, free_slots):
        state = self.counters.states[target.request_id]
        base = dict(target_id=target.request_id, quantum_remaining=state.quantum_remaining)
        if not backend_idle:
            return Intent("DEFER", "another staged action owns the backend", **base)
        if free_slots > 0 and free_blocks >= target.remaining_blocks:
            return Intent("PRIORITIZE_WAITING", "no eviction needed", **base)
        victims = [r for r in rows if r.status == "RUNNING" and r.backend_eligible
                   and priorities[r.request_id] > priorities[target.request_id]
                   and free_blocks + r.held_blocks >= target.remaining_blocks]
        if not victims:
            return Intent("DEFER", "no legal single victim can fund full history", **base)
        victim = min(victims, key=lambda r: -r.output_tokens)
        return Intent("PREPARE_SELECTED", "fund KV and/or sequence slot through shared two-stage save", target.request_id,
                      victim.request_id, state.quantum_remaining)
    def accept(self, intent):
        if intent.action not in ("PRIORITIZE_WAITING", "PREPARE_SELECTED"):
            raise ValueError("only an executable recovery intent may acquire the active latch")
        self.active_target = intent.target_id
    def release_active(self):
        self.active_target = None  # Censored work preserves idle/priority/remaining quantum.
    def after_step(self, actual_scheduled_tokens):
        """Charge quantum only to requests in native num_scheduled_tokens."""
        if any(type(n) is not int or n < 0 for n in actual_scheduled_tokens.values()):
            raise ValueError("invalid actual scheduled-token map")
        self.counters.after_schedule(r for r, n in actual_scheduled_tokens.items() if n)
    def finish(self, request_id):
        self.counters.finish(request_id)
        if self.active_target == request_id:
            self.active_target = None
