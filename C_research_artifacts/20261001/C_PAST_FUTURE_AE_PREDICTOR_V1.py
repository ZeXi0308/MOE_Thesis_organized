"""Author-AE Past-Future statistical core, isolated from any native scheduler.

Source: https://raw.githubusercontent.com/WuSiYu/lightllm-ae/d93ff69c07d4097b0a6a4ee67ea355e8780e3095/lightllm-server-pastfuture/lightllm/server/router/req_queue.py
SHA-256: 79b3526ff983b759f41dd2fdc0a2eae1df8134486c11a4ee98c5e7b6d86bfbdc
Tuple semantics: https://raw.githubusercontent.com/WuSiYu/lightllm-ae/d93ff69c07d4097b0a6a4ee67ea355e8780e3095/lightllm-server-pastfuture/lightllm/server/io_struct.py
SHA-256: 5107b13ef94c861b86ff5e066c1c15749687f2a8d15816fc1d9d1e64979f2ccb

This takes LightLLM NormalReq-equivalent state, not raw vLLM requests. Pass
the AE startup length args.max_req_total_len - args.max_req_input_len as
configured_max_output_tokens (1024 for the intended C setup). A future native
port must establish the output-count, prefill, prompt-cache, paused-KV, and
block-capacity conversions before its score can control admission. Strictly
greater historical knots do not imply a strictly greater *interpolated* value:
rounding can return the already-generated count, as the AE source allows.
"""
from __future__ import annotations

import bisect
from collections import deque
from dataclasses import dataclass
import random

import numpy as np

WINDOW_SIZE = 40
MINIMUM_SAMPLES = 200
MAXIMUM_LISTS = 5
REVERSED = 0.05  # Author AE spelling: reserved fraction of token capacity.


@dataclass(frozen=True)
class AERequest:
    """NormalReq boundary fields; output_tokens means len(output_ids)."""
    request_id: str
    prompt_tokens: int
    output_tokens: int
    max_output_tokens: int
    state: str  # RUNNING, WAIT_IN_QUEUE, or AE PAUSED_AND_OFFLOAD.
    prompt_cache_tokens: int = 0
    ignore_eos: bool = False

    def __post_init__(self):
        if (not self.request_id or self.state not in (
                    "RUNNING", "WAIT_IN_QUEUE", "PAUSED_AND_OFFLOAD")
                or any(type(value) is not int for value in (
                    self.prompt_tokens, self.output_tokens, self.max_output_tokens,
                    self.prompt_cache_tokens))
                or not 0 <= self.prompt_cache_tokens < self.prompt_tokens
                or not 0 <= self.output_tokens < self.max_output_tokens
                or (self.state == "WAIT_IN_QUEUE" and self.output_tokens != 0)):
            raise ValueError("requires unfinished LightLLM NormalReq boundary state")


def normal_req_tuple(req: AERequest, sampled_final_output: int) -> tuple[int, int]:
    """AE NormalReq.get_tuple_tokens(False, sample, factor=1), no splitfuse.

    AE's WAIT tuple starts with prompt+1 and subtracts two from predicted
    output; RUNNING starts with prompt+output and subtracts output+1. These
    are source bookkeeping conventions, not vLLM num_computed_tokens.
    """
    if type(sampled_final_output) is not int or sampled_final_output < 0:
        raise ValueError("sampled final output must be a nonnegative integer")
    predicted = (req.max_output_tokens if req.ignore_eos else
                 min(req.max_output_tokens, max(req.output_tokens, sampled_final_output)))
    if req.state == "RUNNING":
        return (req.prompt_tokens + req.output_tokens - req.prompt_cache_tokens,
                max(0, predicted - req.output_tokens - 1))
    if req.state == "WAIT_IN_QUEUE":
        return (req.prompt_tokens + 1 - req.prompt_cache_tokens,
                max(0, predicted - 2))
    # AE PAUSED_AND_OFFLOAD is the closest tuple for vLLM native recompute:
    # both lost physical KV and retain generated output IDs. The native port
    # still must verify that the entire recompute prefill fits this schedule.
    return (req.prompt_tokens + req.output_tokens + 1 - req.prompt_cache_tokens,
            max(0, predicted - req.output_tokens - 2))


def future_peak_tokens(state_tuples: list[tuple[int, int]]) -> int:
    """AE sorted-remaining prefix maximum, in tokens, before block rounding."""
    if not state_tuples:
        return 0
    if any(type(used) is not int or type(left) is not int or used < 0 or left < 0
           for used, left in state_tuples):
        raise ValueError("state tuples must be nonnegative integer pairs")
    total = peak = 0
    for size, (used, left) in enumerate(
            sorted(state_tuples, key=lambda item: -item[1]), start=1):
        total += used
        peak = max(peak, total + left * size)
    return peak


@dataclass(frozen=True)
class AEPrediction:
    sampled_lists: int
    peaks_tokens: tuple[int, ...]
    worst_peak_tokens: int
    token_limit: float
    fits: bool  # Source uses strict <, not <=.


class PastFutureAEPredictor:
    """One independent AE history/RNG stream per measurement cell."""
    def __init__(self, configured_max_output_tokens: int, seed: int):
        if (type(configured_max_output_tokens) is not int
                or configured_max_output_tokens < 1
                or type(seed) is not int or not 0 <= seed < 2**32):
            raise ValueError("need a positive configured cap and uint32 seed")
        self.history_output_len = deque(
            [configured_max_output_tokens] * (WINDOW_SIZE // 2), maxlen=WINDOW_SIZE)
        self._python_rng = random.Random(seed)
        self._numpy_rng = np.random.RandomState(seed)  # AE global NumPy RNG family.
        self._recorded_ids: set[str] = set()

    def history(self) -> tuple[int, ...]:
        return tuple(self.history_output_len)

    def record_completed_measured(self, request_id: str, output_tokens: int,
                                  *, phase: str, finish_reason: str) -> None:
        """Append only a newly completed measured stop/length request."""
        if (not request_id or request_id in self._recorded_ids
                or phase != "measured" or finish_reason not in ("stop", "length")
                or type(output_tokens) is not int or output_tokens < 0):
            raise ValueError("history accepts each completed measured request once")
        self._recorded_ids.add(request_id)
        self.history_output_len.append(output_tokens)

    def _conditional_knots(self, req: AERequest) -> list[int]:
        history = sorted(self.history_output_len)
        # bisect_right is strict conditioning: historical lengths <= O excluded.
        knots = [req.output_tokens] + history[bisect.bisect(history, req.output_tokens):]
        if knots[-1] < req.max_output_tokens:
            knots.append(req.max_output_tokens)
        return knots

    def sample_running_tuples(self, running: list[AERequest]) -> list[list[tuple[int, int]]]:
        """AE _sample_cache_list order: request outer, sampled list inner.

        Preserve the source's round-down-to-current-output case; it yields
        zero remaining tokens through NormalReq's max(0, sampled-O-1).
        """
        if any(req.state != "RUNNING" for req in running):
            raise ValueError("running sample requires RUNNING states")
        if not running:
            return [[]]
        count = min(MAXIMUM_LISTS, int(MINIMUM_SAMPLES / len(running)) + 1)
        lists: list[list[tuple[int, int]]] = [[] for _ in range(count)]
        for req in running:
            knots = self._conditional_knots(req)
            for tuples in lists:
                point = self._numpy_rng.random_sample() * (len(knots) - 1)
                position = int(point)
                left, right = knots[position:position + 2]
                sampled = round(left + (right - left) * (point - position))
                tuples.append(normal_req_tuple(req, sampled))
        return lists

    def score_waiting(self, running: list[AERequest], head: AERequest,
                      *, capacity_tokens: int, other_reserved_tokens: int = 0) -> AEPrediction:
        """Score one FIFO head; never mutate a scheduler or admit a request.

        `other_reserved_tokens` is the caller's paused/prompt-cache deduction.
        Sequence-slot, batch-token, prefill and paused-resume checks belong to
        the future native port. The AE samples the waiting head once per list.
        """
        if (head.state not in ("WAIT_IN_QUEUE", "PAUSED_AND_OFFLOAD")
                or head.request_id in {r.request_id for r in running}
                or type(capacity_tokens) is not int or capacity_tokens <= 0
                or type(other_reserved_tokens) is not int or other_reserved_tokens < 0):
            raise ValueError("invalid waiting head or token capacity boundary")
        lists = self.sample_running_tuples(running)
        peaks = []
        for tuples in lists:
            sampled = self._python_rng.choice(self.history_output_len)
            peaks.append(future_peak_tokens(tuples + [normal_req_tuple(head, sampled)]))
        worst = max(peaks)
        limit = capacity_tokens * (1 - REVERSED) - other_reserved_tokens
        return AEPrediction(len(lists), tuple(peaks), worst, limit, worst < limit)


def _self_check() -> None:
    running = [AERequest("r", 100, 10, 11, "RUNNING")]
    head = AERequest("h", 50, 0, 11, "WAIT_IN_QUEUE")
    a, b = (PastFutureAEPredictor(11, 1) for _ in range(2))
    for _ in range(3):
        assert a.score_waiting(running, head, capacity_tokens=300) == \
               b.score_waiting(running, head, capacity_tokens=300)
    c = PastFutureAEPredictor(11, 1)
    assert c._conditional_knots(running[0]) == [10] + [11] * 20
    # First AE interpolation rounds down to the already-generated count.
    assert c.sample_running_tuples(running)[0][0] == (110, 0)
    assert normal_req_tuple(AERequest("p", 100, 10, 20, "PAUSED_AND_OFFLOAD"), 15) == (111, 3)
    for i in range(45):
        a.record_completed_measured(str(i), i + 1, phase="measured", finish_reason="length")
    assert a.history() == tuple(range(6, 46))
    try:
        a.record_completed_measured("warmup", 4, phase="warmup", finish_reason="stop")
    except ValueError:
        pass
    else:
        raise AssertionError("warmup incorrectly entered measured history")
    assert future_peak_tokens([(10, 3), (20, 1)]) == 32
    print("AE predictor replay, strict conditioning, interpolation and history rollover: PASS")


if __name__ == "__main__":
    _self_check()
