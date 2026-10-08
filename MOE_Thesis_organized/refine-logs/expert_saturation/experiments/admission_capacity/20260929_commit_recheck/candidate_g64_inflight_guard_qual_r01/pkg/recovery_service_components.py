"""CPU scheduling components; neither class implements a paper's full system.

LTR counters follow hao-ai-lab/vllm-ltr commit
13bbf6ff3dab661791d41362551b089e5f77c91c, vllm/core/scheduler.py
lines 984-998 and 1359-1365; 200/10 comes from fair-lmsys-70B.sh.
Its released resource path uses CPU SWAP. Reusing these counters with recompute
is a component transplant, not a reproduction of that path or its predictor.

The geometric guard follows https://arxiv.org/html/2606.18431v1, Section 3.3,
Eq. (8), and Appendix A.4 (k=256). Only the paper rule is verified. A residency
start is an explicit integration event, not every microbatch dispatch. Its
fixed fence is absolute attained NEW decode output, not recomputation work.
This omits UniBoost priorities, adaptive gamma, victim costs, and KV allocation.
The caller must check physical feasibility; neither component guarantees a
finite wall-clock wait or authorizes eviction of a protected request.
"""

from dataclasses import dataclass, field
from typing import Iterable


def _integer(value: int, name: str, minimum: int = 0) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


@dataclass
class LTRRequestState:
    idle: int = 0
    priority: int = 0
    quantum_remaining: int = 0


@dataclass
class LTRCounters:
    """One begin/after pair per scheduling iteration, before model execution."""

    threshold: int = 200
    quantum: int = 10
    states: dict[str, LTRRequestState] = field(default_factory=dict, init=False)
    _open: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        _integer(self.threshold, "threshold", 1)
        _integer(self.quantum, "quantum", 1)

    def begin_schedule(self, live_ids: Iterable[str]) -> dict[str, int]:
        if self._open:
            raise RuntimeError("previous scheduling iteration has no after_schedule")
        self.states = {rid: self.states.get(rid, LTRRequestState()) for rid in live_ids}
        for state in self.states.values():
            if state.idle >= self.threshold:
                state.priority, state.idle = -1, 0
                state.quantum_remaining = self.quantum
            elif state.priority == -1 and state.quantum_remaining <= 0:
                state.priority = 0
        self._open = True
        return {rid: state.priority for rid, state in self.states.items()}

    def after_schedule(self, selected_ids: Iterable[str]) -> None:
        """Use the actual feasible batch: prefill/recompute also spend one turn."""
        if not self._open:
            raise RuntimeError("begin_schedule must precede after_schedule")
        selected = set(selected_ids)
        if not selected <= self.states.keys():
            raise ValueError("selected request is not live in this iteration")
        for rid, state in self.states.items():
            if rid in selected:
                state.idle = 0
                if state.priority == -1:
                    state.quantum_remaining -= 1
            else:
                state.idle += 1
        self._open = False

    def finish(self, request_id: str) -> None:
        self.states.pop(request_id, None)


@dataclass
class ServiceResidency:
    attained_decode_tokens: int
    fence: int


@dataclass
class GeometricServiceGuard:
    """Protect a residency through the next geometric attained-output fence."""

    k: int = 256
    residencies: dict[str, ServiceResidency] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        _integer(self.k, "k", 1)

    def quantized_work(self, attained_decode_tokens: int) -> int:
        _integer(attained_decode_tokens, "attained_decode_tokens")
        quotient = max(attained_decode_tokens, self.k) // self.k
        return self.k * (1 << (quotient.bit_length() - 1))

    def start_residency(self, request_id: str, attained_decode_tokens: int) -> int:
        """Call once on acquisition/reacquisition, before any recovery work."""
        fence = 2 * self.quantized_work(attained_decode_tokens)
        if request_id in self.residencies:
            raise ValueError("residency already active; microbatches must not reset its fence")
        self.residencies[request_id] = ServiceResidency(attained_decode_tokens, fence)
        return fence

    def note_output(self, request_id: str, new_decode_tokens: int) -> None:
        """Report newly returned decode tokens only; pure recompute contributes zero."""
        _integer(new_decode_tokens, "new_decode_tokens")
        self.residencies[request_id].attained_decode_tokens += new_decode_tokens

    def can_evict(self, request_id: str) -> bool:
        state = self.residencies[request_id]
        return state.attained_decode_tokens >= state.fence

    def end_residency(self, request_id: str) -> None:
        """Notify actual eviction; eligibility/physical feasibility belong to caller."""
        self.residencies.pop(request_id, None)

    def finish(self, request_id: str) -> None:
        """EOS/completion releases state even when the service fence was not reached."""
        self.end_residency(request_id)
