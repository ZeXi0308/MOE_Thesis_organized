"""Deterministic full-output-cap ablation of the frozen native AE core port.

Only score_waiting changes: use each request's known max_output_tokens in
AE NormalReq tuples and the same future-peak formula, 5% reserve, strict
inequality, FIFO head, joint-batch guard, native preemption, and completion
history bookkeeping. History is recorded but never sampled for decisions.
This isolates the incremental action of the author's history/sampling core;
it is a baseline ablation, not a new C method or full LightLLM router.
"""
from __future__ import annotations

from contextlib import contextmanager

from C_NATIVE_PAST_FUTURE_AE_ADMISSION_V1 import (
    NativePastFutureAEAdmission, OUTPUT_CAP, _self_check as native_lifecycle_check)
from C_PAST_FUTURE_AE_PREDICTOR_V1 import (
    AERequest, AEPrediction, PastFutureAEPredictor, REVERSED,
    future_peak_tokens, normal_req_tuple)

POLICY = "author_AE_full_output_cap_ablation_native_batch4096"


class FullCapAEPredictor(PastFutureAEPredictor):
    def score_waiting(self, running: list[AERequest], head: AERequest,
                      *, capacity_tokens: int, other_reserved_tokens: int = 0) -> AEPrediction:
        if (any(req.state != "RUNNING" for req in running)
                or head.state not in ("WAIT_IN_QUEUE", "PAUSED_AND_OFFLOAD")
                or head.request_id in {req.request_id for req in running}
                or type(capacity_tokens) is not int or capacity_tokens <= 0
                or type(other_reserved_tokens) is not int or other_reserved_tokens < 0):
            raise ValueError("invalid full-cap AE boundary")
        tuples = [normal_req_tuple(req, req.max_output_tokens)
                  for req in [*running, head]]
        peak = future_peak_tokens(tuples)
        limit = capacity_tokens * (1 - REVERSED) - other_reserved_tokens
        return AEPrediction(1, (peak,), peak, limit, peak < limit)


class NativePastFutureCapAblation(NativePastFutureAEAdmission):
    def __init__(self, engine, seed: int = 20261001):
        super().__init__(engine, seed=seed)
        self.predictor = FullCapAEPredictor(OUTPUT_CAP, seed)

    def receipt(self) -> dict:
        result = super().receipt()
        result.update(policy=POLICY,
            scope="Full known output caps in AE NormalReq peak formula; same native FCFS/recompute and one-head full-prefill guard; no history sampling or full LightLLM router",
            score_rule="max_output_tokens for every resident and waiting head",
            peak_lists_are_deterministic=True,
            history_recorded_but_not_scored=True)
        return result


@contextmanager
def install(engine, seed: int = 20261001):
    gate = NativePastFutureCapAblation(engine, seed=seed)
    s = gate.scheduler
    try:
        s.schedule = gate.schedule
        s._free_request = gate.free
        yield gate
    finally:
        if gate.had_schedule:
            s.schedule = gate.original_schedule
        else:
            del s.schedule
        if gate.had_free:
            s._free_request = gate.original_free
        else:
            del s._free_request


def _self_check() -> None:
    native_lifecycle_check()  # Inherited pure-decode, preemption, and free/history lifecycle.
    assert NativePastFutureCapAblation.schedule is NativePastFutureAEAdmission.schedule
    assert NativePastFutureCapAblation.free is NativePastFutureAEAdmission.free
    assert NativePastFutureCapAblation._state is NativePastFutureAEAdmission._state
    resident = AERequest("r", 100, 10, 20, "RUNNING")
    head = AERequest("h", 50, 0, 20, "WAIT_IN_QUEUE")
    predictor = FullCapAEPredictor(20, 7)
    expected = future_peak_tokens([normal_req_tuple(r, r.max_output_tokens)
                                   for r in (resident, head)])
    before = predictor.score_waiting([resident], head, capacity_tokens=300)
    predictor.record_completed_measured("past", 3, phase="measured", finish_reason="stop")
    after = predictor.score_waiting([resident], head, capacity_tokens=300)
    assert before == after and before.worst_peak_tokens == expected
    assert after.sampled_lists == 1 and after.token_limit == 285.0

    # Actual viewed PF pilot decision 290: 21 residents, one FIFO head.
    # Only current P/O and known M=1024 are used; no future EOS is read.
    pairs = [(1679, 289), (2292, 281), (839, 270), (2020, 249),
             (1047, 214), (1756, 206), (2798, 168), (2814, 167),
             (1503, 141), (2484, 140), (1995, 139), (1958, 100),
             (2792, 90), (3042, 89), (1878, 88), (1910, 53),
             (1714, 50), (2609, 43), (999, 7), (2266, 4), (1820, 3)]
    running = [AERequest(str(i), p, o, 1024, "RUNNING")
               for i, (p, o) in enumerate(pairs)]
    waiting = AERequest("head", 2697, 0, 1024, "WAIT_IN_QUEUE")
    scored = FullCapAEPredictor(1024, 20261001).score_waiting(
        running, waiting, capacity_tokens=4096 * 16)
    assert scored.worst_peak_tokens == 63852
    assert scored.token_limit == 62259.2 and not scored.fits
    assert 61295 < scored.token_limit  # Frozen PF pilot's sampled peak on call 290.
    print("full-cap AE score, history independence, call-290 split: PASS")


if __name__ == "__main__":
    _self_check()
