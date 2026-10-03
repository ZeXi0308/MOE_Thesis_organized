"""Past-visible bookkeeping for recovery-to-first-output obligations.

Call ``begin_schedule`` before planning and ``after_schedule`` only after the
native tokens/victims match the plan.
"""
from dataclasses import asdict, dataclass

@dataclass(frozen=True)
class RequestView:
    status: str
    output_tokens: int
    pending_tokens: int
@dataclass(frozen=True)
class Obligation:
    start_step: int
    start_output_tokens: int
    start_pending_tokens: int
    start_priority: int
class RestoreObligations:
    """Track every applied partial restore until visible output or completion."""
    def __init__(self, obligation_priority=-2):
        self.obligation_priority = obligation_priority
        self.active = {}
        self.events = []
        self._open_step = None
        self._last_step = -1
    def _release(self, request_id, step, reason, observed):
        event = dict(type="release", request_id=request_id, release_step=step,
                     reason=reason, observed_output_tokens=observed,
                     **asdict(self.active.pop(request_id)))
        self.events.append(event)
        return event
    def _observe(self, step, visible, completed_ids):
        completed = set(completed_ids)
        released = []
        for request_id, obligation in sorted(tuple(self.active.items())):
            view = visible.get(request_id)
            if request_id in completed:
                reason, observed = "completed", None if view is None else view.output_tokens
            elif view is None:
                raise RuntimeError(f"active obligation disappeared: {request_id}")
            elif view.output_tokens < obligation.start_output_tokens:
                raise RuntimeError(f"output count regressed: {request_id}")
            elif view.output_tokens == obligation.start_output_tokens:
                continue
            else:
                reason, observed = "new_output", view.output_tokens
            released.append(self._release(request_id, step, reason, observed))
        return released
    def begin_schedule(self, step, visible, completed_ids=()):
        """Consume only state visible before this schedule and release progress."""
        if self._open_step is not None or type(step) is not int or step <= self._last_step:
            raise RuntimeError("scheduling steps must be paired and increasing")
        released = self._observe(step, visible, completed_ids)
        self._open_step, self._last_step = step, step
        return dict(outstanding=sorted(self.active), released=released)
    def effective_priorities(self, original, *, enabled):
        """Overlay -2 only for obligations still active after begin_schedule."""
        missing = set(self.active).difference(original)
        if missing:
            raise RuntimeError(f"outstanding obligations are not live: {sorted(missing)}")
        return {rid: self.obligation_priority if enabled and rid in self.active else priority
                for rid, priority in original.items()}
    def after_schedule(self, step, before, resumed_ids, actual_tokens, preempted_ids,
                       remaining_history_blocks, free_blocks, *, enabled,
                       effective_priorities):
        """Apply actual interruptions/starts, then check retained history."""
        if step != self._open_step:
            raise RuntimeError("after_schedule must close the open step")
        resumed, preempted = sorted(set(resumed_ids)), set(preempted_ids)
        if not set(resumed).issubset(actual_tokens) or any(actual_tokens[r] <= 0 for r in resumed):
            raise RuntimeError("resumed request was not actually scheduled")
        interrupted = sorted(set(self.active) & preempted)
        if enabled and interrupted:
            raise RuntimeError(f"enabled obligation was preempted: {interrupted}")
        for request_id in interrupted:
            self._release(request_id, step, "interrupted", before[request_id].output_tokens)
        started = []
        for request_id in resumed:
            view = before[request_id]
            if view.status != "PREEMPTED":
                raise RuntimeError(f"resumed request was not PREEMPTED: {request_id}")
            if view.output_tokens == 0 or view.pending_tokens <= 1:
                continue
            if request_id in self.active:
                raise RuntimeError(f"recovery restarted without interruption: {request_id}")
            self.active[request_id] = Obligation(
                step, view.output_tokens, view.pending_tokens,
                effective_priorities[request_id])
            started.append(request_id)
            self.events.append(dict(type="start", request_id=request_id, step=step,
                                    output_tokens=view.output_tokens,
                                    pending_tokens=view.pending_tokens,
                                    actual_tokens=actual_tokens[request_id],
                                    effective_priority=effective_priorities[request_id]))
        missing = set(self.active).difference(remaining_history_blocks)
        if missing or any(type(remaining_history_blocks[r]) is not int
                          or remaining_history_blocks[r] < 0 for r in self.active):
            raise RuntimeError(f"invalid remaining history for obligations: {sorted(missing)}")
        required = sum(remaining_history_blocks[r] for r in self.active)
        if enabled and (type(free_blocks) is not int or free_blocks < required):
            raise RuntimeError(f"restore history reservation lost: required={required}, free={free_blocks}")
        self._open_step = None
        return dict(interrupted=interrupted, started=started,
                    outstanding=sorted(self.active), remaining_history_blocks=required)
    def finalize(self, step, visible, completed_ids=()):
        """Release a final completion/output when no later schedule is invoked."""
        if self._open_step is not None:
            raise RuntimeError("cannot finalize an open scheduling step")
        released = self._observe(step, visible, completed_ids)
        if self.active:
            raise RuntimeError(f"unresolved restore obligations: {sorted(self.active)}")
        return released
    def snapshot(self):
        return dict(active={rid: asdict(value) for rid, value in sorted(self.active.items())},
                    events=list(self.events))
