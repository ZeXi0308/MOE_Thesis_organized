"""CPU intent selector for an LTR-style component on G's selected-offload backend.

This is not a vLLM adapter or a full LTR reproduction.  It reuses the verified
LTR counters, restricts intervention to preempted recovery requests, and emits
one target/victim intent for G's existing two-stage native save path to check.
No future state enters a decision; current returned output counts may rank victims.
"""
from dataclasses import dataclass
from pathlib import Path
import sys

_PKG = Path(__file__).resolve().parents[1] / "preparation" / "pkg"
sys.path.insert(0, str(_PKG))
from recovery_service_components import LTRCounters  # noqa: E402

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
    def begin_step(self, requests, free_blocks, *, backend_idle=True):
        rows = list(requests)
        by_id = {r.request_id: r for r in rows}
        if len(by_id) != len(rows) or free_blocks < 0:
            raise ValueError("invalid observed request state")
        if any(min(r.remaining_blocks, r.output_tokens, r.held_blocks) < 0 for r in rows):
            raise ValueError("negative request resource state")
        priorities = self.counters.begin_schedule(by_id)
        if (self.active_target not in by_id
                or priorities.get(self.active_target) != -1):
            self.active_target = None
        if self.active_target is None:
            waiting = [r for r in rows
                       if r.status == "PREEMPTED" and priorities[r.request_id] == -1]
            if waiting:
                self.active_target = min(waiting, key=lambda r: r.arrival).request_id
        if self.active_target is None:
            return Intent("NOOP", "no boosted recovery request")
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
        if not backend_idle:
            return Intent("DEFER", "another staged action owns the backend", **base)
        if free_blocks >= target.remaining_blocks:
            return Intent("PRIORITIZE_WAITING", "no eviction needed", **base)
        victims = [r for r in rows if r.status == "RUNNING" and r.backend_eligible
                   and priorities[r.request_id] > priorities[target.request_id]
                   and free_blocks + r.held_blocks >= target.remaining_blocks]
        if not victims:
            return Intent("DEFER", "no legal single victim can fund full history", **base)
        victim = min(victims, key=lambda r: -r.output_tokens)
        return Intent("PREPARE_SELECTED", "run shared two-stage save", target.request_id,
                      victim.request_id, state.quantum_remaining)
    def after_step(self, actual_scheduled_tokens):
        """Charge quantum only to requests in native num_scheduled_tokens."""
        if any(type(n) is not int or n < 0 for n in actual_scheduled_tokens.values()):
            raise ValueError("invalid actual scheduled-token map")
        self.counters.after_schedule(r for r, n in actual_scheduled_tokens.items() if n)
    def finish(self, request_id):
        self.counters.finish(request_id)
        if self.active_target == request_id:
            self.active_target = None

def _self_check():
    target = Request("t", 0.0, "PREEMPTED", remaining_blocks=4)
    victim = Request("v", 1.0, "RUNNING", output_tokens=8,
                     held_blocks=4, backend_eligible=True)
    selector = LTRStyleSelected(threshold=5, quantum=2)
    for _ in range(5):
        assert selector.begin_step([target, victim], 0).action == "NOOP"
        selector.after_step({"v": 1})
    assert selector.begin_step([target, victim], 0).action == "PREPARE_SELECTED"
    selector.after_step({"v": 1})
    loading = Request("t", 0.0, "WAITING_FOR_REMOTE_KVS", remaining_blocks=4)
    assert selector.begin_step([loading, victim], 0, backend_idle=False).quantum_remaining == 2
    selector.after_step({"v": 1})
    for remaining, observed_output in ((2, 0), (1, 1)):
        running = Request("t", 0.0, "RUNNING", output_tokens=observed_output)
        intent = selector.begin_step([running, victim], 0)
        assert (intent.action, intent.quantum_remaining) == ("PRIORITIZE_RUNNING", remaining)
        selector.after_step({"t": 128, "v": 1})
    assert selector.begin_step([running, victim], 0).action == "NOOP"
    selector.after_step({"v": 1})
    enough = LTRStyleSelected(threshold=1, quantum=1)
    assert enough.begin_step([target, victim], 4).action == "NOOP"
    enough.after_step({"v": 1})
    intent = enough.begin_step([target, victim], 4)
    assert intent.action == "PRIORITIZE_WAITING" and intent.victim_id is None
    enough.after_step({})
    def renamed_choice(names):
        rows = [Request(names[0], 0.0, "PREEMPTED", remaining_blocks=4),
                Request(names[1], 0.0, "PREEMPTED", remaining_blocks=4),
                Request(names[2], 1.0, "RUNNING", output_tokens=8, held_blocks=4, backend_eligible=True),
                Request(names[3], 1.0, "RUNNING", output_tokens=8, held_blocks=4, backend_eligible=True)]
        probe = LTRStyleSelected(threshold=1, quantum=1)
        assert probe.begin_step(rows, 0).action == "NOOP"
        probe.after_step({names[2]: 1, names[3]: 1})
        choice = probe.begin_step(rows, 0); probe.after_step({})
        ids = [r.request_id for r in rows]
        return ids.index(choice.target_id), ids.index(choice.victim_id)
    assert renamed_choice(("t0", "t1", "v0", "v1")) == renamed_choice(("z", "a", "y", "b")) == (0, 2)
    return {"status": "PASS_CPU_ONLY", "gpu_actions": 0,
            "checked": ["load_does_not_spend_quantum",
                        "first_output_does_not_change_quantum",
                        "free_capacity_keeps_priority_action", "id_rename_keeps_queue_positions"]}

if __name__ == "__main__":
    print(_self_check())
