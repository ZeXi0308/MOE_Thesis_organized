"""Author-AE Past-Future peak core on the pinned native vLLM scheduler.

This is an AE statistical/peak admission baseline, not a reproduction of the
LightLLM router or its preemption policy. It uses vLLM's native FCFS and
recompute preemption. A separate batch-4096 regime lets each admitted prompt
or resumed recompute fit in one schedule call. No partial-prefill state is
passed to the AE NormalReq predictor.

Pinned source: vLLM 0.26 scheduler.py SHA-256
2ed2a550b6558b2495eda845a97ae38bcf0225027b9e25fbf00fc3880c1d3941.
The running loop precedes the waiting loop (lines 456, 664); a preemption
skips waiting that call. Waiting admission is FCFS and respects the temporary
max_num_running_reqs (lines 667-680). PREEMPTED resets computed tokens to zero
but keeps output IDs (lines 1210-1235). _free_request releases native KV and
removes the request (lines 2207-2245). The adapter never owns physical KV.
"""
from __future__ import annotations

from contextlib import contextmanager
from time import perf_counter

from C_NATIVE_MAX_BOUND_ADMISSION import CAPACITY, MaxBoundAdmission
from C_NATIVE_RETIREMENT_ADMISSION_V1 import decode_ready
from C_PAST_FUTURE_AE_PREDICTOR_V1 import AERequest, PastFutureAEPredictor

TOKEN_CAPACITY = CAPACITY * 16
BATCH_TOKENS = 4096
OUTPUT_CAP = 1024
PROMPT_CEILING = 3072


def joint_batch_need(running: list, head) -> int:
    """Exact next-call input count in the qualified pure-decode regime."""
    needs = [r.num_tokens_with_spec + r.num_output_placeholders
             - r.num_computed_tokens for r in running]
    outstanding = head.num_tokens - head.num_computed_tokens
    if any(value != 1 for value in needs) or not 0 < outstanding <= BATCH_TOKENS:
        raise ValueError("joint-batch calculation requires pure decode and full head")
    return sum(needs) + outstanding


class NativePastFutureAEAdmission(MaxBoundAdmission):
    def __init__(self, engine, seed: int = 20261001):
        super().__init__(engine)  # Empty FCFS, 32 slots, no offload/APC/spec, 4096 usable blocks.
        s = self.scheduler
        config = s.scheduler_config
        if not (s.max_num_scheduled_tokens == config.max_num_batched_tokens == BATCH_TOKENS
                and s.max_model_len == 4096 and s.num_sampled_tokens_per_step == 1
                and s.scheduler_reserve_full_isl is True
                and (config.long_prefill_token_threshold <= 0
                     or config.long_prefill_token_threshold >= BATCH_TOKENS)
                and config.async_scheduling is False
                and engine.vllm_config.max_concurrent_batches == 1
                and s._pause_state.name == "UNPAUSED"
                and s.block_size == 16
                and s.dcp_world_size == s.pcp_world_size == 1
                and not s.is_encoder_decoder and not s.has_mamba_layers
                and s.ec_connector is None):
            raise ValueError("AE native batch4096/synchronous single-GPU qualification differs")
        self.predictor = PastFutureAEPredictor(OUTPUT_CAP, seed)
        self.seed = seed
        self.admitted_ids: set[str] = set()
        self.completed_ids: set[str] = set()
        self.resumptions = 0
        self.decision_seconds = 0.0
        self.physical_blocks_peak = 0
        self.decisions: list[dict] = []

    def _state(self, request) -> AERequest:
        status = request.status
        states = {self.RequestStatus.RUNNING: "RUNNING",
                  self.RequestStatus.WAITING: "WAIT_IN_QUEUE",
                  self.RequestStatus.PREEMPTED: "PAUSED_AND_OFFLOAD"}
        if status not in states:
            self.fail("unsupported native request state at AE conversion")
        prompt, output, cap = (request.num_prompt_tokens,
                               request.num_output_tokens, request.max_tokens)
        params = request.sampling_params
        if not (type(prompt) is int and 0 < prompt <= PROMPT_CEILING
                and type(output) is int and 0 <= output < cap == OUTPUT_CAP
                and prompt + cap <= 4096 and params is not None
                and not params.ignore_eos and params.min_tokens == 0
                and not request.has_encoder_inputs and not request.spec_token_ids
                and request.num_output_placeholders == request.num_in_flight_tokens == 0):
            self.fail("request differs from natural-EOS text-only AE boundary")
        if status == self.RequestStatus.RUNNING:
            if not decode_ready(request):
                self.fail("surviving resident is not pure decode at next schedule")
        elif request.num_computed_tokens != 0:
            self.fail("waiting/recompute head has retained computed KV")
        if status == self.RequestStatus.WAITING and output != 0:
            self.fail("fresh waiting head already has output")
        return AERequest(request.request_id, prompt, output, cap, states[status])

    def _physical_used(self) -> int:
        free = self.scheduler.kv_cache_manager.block_pool.get_num_free_blocks()
        used = CAPACITY - free
        if not 0 <= used <= CAPACITY:
            self.fail("physical GPU KV use outside qualified usable pool")
        self.physical_blocks_peak = max(self.physical_blocks_peak, used)
        return used

    def schedule(self, *args, **kwargs):
        start = perf_counter()
        s = self.scheduler
        self.schedule_calls += 1
        if s.skipped_waiting or s.num_waiting_for_streaming_input or s._pause_state.name != "UNPAUSED":
            self.fail("unmodeled skipped, streaming, or paused native queue")
        running, waiting = list(s.running), list(s.waiting)
        residents = [self._state(req) for req in running]
        # Native schedule increments current_step before its eligibility test.
        if any(req.next_decode_eligible_step > s.current_step + 1 for req in running):
            self.fail("resident cannot receive its next decode this schedule")
        if any(req.status not in (self.RequestStatus.WAITING,
                                  self.RequestStatus.PREEMPTED) for req in waiting):
            self.fail("unsupported waiting status")
        head = waiting[0] if waiting else None
        allowed = False
        decision = None
        if head is not None:
            head_state = self._state(head)
            head_tokens = head.num_tokens - head.num_computed_tokens
            head_need = joint_batch_need(running, head)
            reason = "slots" if len(running) == 32 else "joint_batch" if head_need > BATCH_TOKENS else "ae_peak"
            prediction = None
            if reason == "ae_peak":
                prediction = self.predictor.score_waiting(
                    residents, head_state, capacity_tokens=TOKEN_CAPACITY)
                allowed = prediction.fits
                if not allowed:
                    reason = "ae_reject"
            decision = dict(call=self.schedule_calls, head=head.request_id,
                head_status=head_state.state, running_inputs=[vars(row) for row in residents],
                head_input=vars(head_state), history_before=self.predictor.history(),
                head_tokens=head_tokens, joint_batch_need=head_need,
                sampled_peaks=(prediction.peaks_tokens
                    if prediction else ()), sampled_peak=(prediction.worst_peak_tokens
                    if prediction else None), token_limit=(prediction.token_limit
                    if prediction else TOKEN_CAPACITY * .95), allowed=allowed, reason=reason)
            self.decisions.append(decision)
            if not allowed:
                self.hold_calls += 1
        self.decision_seconds += perf_counter() - start
        before_ids = {req.request_id for req in running}
        old_cap = s.max_num_running_reqs
        s.max_num_running_reqs = len(running) + int(allowed)
        try:
            result = self.original_schedule(*args, **kwargs)
        finally:
            s.max_num_running_reqs = old_cap
        after = {req.request_id: req for req in s.running}
        new_ids = set(after) - before_ids
        preempted = set(result.preempted_req_ids or ())
        self.preemptions += len(preempted)
        if (before_ids - set(after) != preempted
                or not new_ids.issubset({head.request_id} if allowed else set())
                or (new_ids and preempted)):
            self.fail("native preemption/admission escaped selected FCFS head")
        if new_ids:
            rid = head.request_id
            if result.num_scheduled_tokens.get(rid) != head_tokens:
                self.fail("selected head received partial prefill/recompute")
            if head_state.state == "WAIT_IN_QUEUE":
                if rid in self.admitted_ids:
                    self.fail("duplicate first admission")
                self.admitted_ids.add(rid)
            else:
                if rid not in self.admitted_ids:
                    self.fail("recompute resumed before first admission")
                self.resumptions += 1
            self.events.append(dict(event="admit" if head_state.state == "WAIT_IN_QUEUE"
                else "resume", request_id=rid, call=self.schedule_calls,
                scheduled_tokens=head_tokens, joint_batch_need=head_need))
        physical = self._physical_used()
        self.steps.append(dict(call=self.schedule_calls, running_before=len(running),
            waiting_before=len(waiting), selected=head.request_id if allowed else None,
            newly_running=sorted(new_ids), preempted=sorted(preempted),
            physical_blocks=physical))
        return result

    def free(self, request, *args, **kwargs):
        result = self.original_free(request, *args, **kwargs)
        rid = request.request_id
        if (rid not in self.admitted_ids or rid in self.completed_ids
                or rid in self.scheduler.requests or
                request.status not in (self.RequestStatus.FINISHED_STOPPED,
                                       self.RequestStatus.FINISHED_LENGTH_CAPPED)):
            self.fail("AE history update lacked unique completed measured native free")
        finish = ("stop" if request.status == self.RequestStatus.FINISHED_STOPPED
                  else "length")
        self.predictor.record_completed_measured(
            rid, request.num_output_tokens, phase="measured", finish_reason=finish)
        self.completed_ids.add(rid)
        self.events.append(dict(event="complete", request_id=rid,
            output_tokens=request.num_output_tokens, finish_reason=finish))
        self._physical_used()
        return result

    def receipt(self) -> dict:
        s = self.scheduler
        drained = (not s.requests and not s.running and not s.waiting
                   and not s.skipped_waiting and
                   s.kv_cache_manager.block_pool.get_num_free_blocks() == CAPACITY)
        return dict(status="DRAINED" if drained and not self.violations
                    and len(self.admitted_ids) == len(self.completed_ids) == 128
                    else "ERROR", policy="author_AE_statistical_peak_core_native_batch4096",
                    scope="Native FCFS/recompute with one-head AE sampling and full-prefill batch guard; not full LightLLM",
                    seed=self.seed, usable_blocks=CAPACITY,
                    token_capacity=TOKEN_CAPACITY, batch_tokens=BATCH_TOKENS,
                    admitted=len(self.admitted_ids), completed=len(self.completed_ids),
                    resumptions=self.resumptions, preemptions=self.preemptions,
                    schedule_calls=self.schedule_calls, hold_calls=self.hold_calls,
                    decision_seconds=self.decision_seconds,
                    physical_blocks_peak=self.physical_blocks_peak,
                    history=list(self.predictor.history()), drained=drained,
                    violations=list(self.violations), decisions=self.decisions,
                    events=self.events, steps=self.steps)


@contextmanager
def install(engine, seed: int = 20261001):
    gate = NativePastFutureAEAdmission(engine, seed=seed)
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
    from types import SimpleNamespace

    class Req:
        def __init__(self, tokens, computed=0):
            self.num_tokens = self.num_tokens_with_spec = tokens
            self.num_computed_tokens = computed
            self.num_output_placeholders = 0
    assert joint_batch_need([Req(100, 99)] * 31, Req(4065)) == 4096
    assert joint_batch_need([Req(100, 99)] * 32, Req(4065)) == 4097
    try:
        joint_batch_need([Req(100, 98)], Req(100))
    except ValueError:
        pass
    else:
        raise AssertionError("partial prefill passed pure-decode guard")

    # A native recompute preemption is an outcome, not an admission violation.
    statuses = SimpleNamespace(RUNNING="running", WAITING="waiting",
        PREEMPTED="preempted", FINISHED_STOPPED="stop", FINISHED_LENGTH_CAPPED="length")
    resident = SimpleNamespace(request_id="r", status=statuses.RUNNING,
        num_prompt_tokens=100, num_output_tokens=10, max_tokens=OUTPUT_CAP,
        num_tokens=110, num_tokens_with_spec=110, num_computed_tokens=109,
        num_output_placeholders=0, num_in_flight_tokens=0, spec_token_ids=[],
        has_encoder_inputs=False, next_decode_eligible_step=0,
        sampling_params=SimpleNamespace(ignore_eos=False, min_tokens=0))
    pool = SimpleNamespace(get_num_free_blocks=lambda: CAPACITY - 7)
    scheduler = SimpleNamespace(running=[resident], waiting=[], skipped_waiting=[],
        num_waiting_for_streaming_input=0, _pause_state=SimpleNamespace(name="UNPAUSED"),
        current_step=0, max_num_running_reqs=32, requests={"r": resident},
        kv_cache_manager=SimpleNamespace(block_pool=pool))
    gate = NativePastFutureAEAdmission.__new__(NativePastFutureAEAdmission)
    gate.scheduler, gate.RequestStatus = scheduler, statuses
    gate.schedule_calls = gate.preemptions = gate.hold_calls = 0
    gate.decision_seconds = gate.physical_blocks_peak = 0
    gate.decisions, gate.steps, gate.events, gate.violations = [], [], [], []
    gate.admitted_ids, gate.completed_ids = {"r"}, set()
    gate.predictor = PastFutureAEPredictor(OUTPUT_CAP, 7)

    def native_preempt():
        scheduler.running.clear()
        resident.status = statuses.PREEMPTED
        resident.num_computed_tokens = 0
        scheduler.waiting.insert(0, resident)
        return SimpleNamespace(preempted_req_ids={"r"}, num_scheduled_tokens={})

    gate.original_schedule = native_preempt
    gate.schedule()
    assert gate.preemptions == 1 and scheduler.max_num_running_reqs == 32
    assert gate._state(resident).state == "PAUSED_AND_OFFLOAD"
    resident.status = statuses.FINISHED_STOPPED
    gate.original_free = lambda req: scheduler.requests.pop(req.request_id)
    gate.free(resident)
    assert gate.completed_ids == {"r"} and gate.predictor.history()[-1] == 10
    assert len(gate.predictor.history()) == 21
    print("AE native joint-batch, recompute preemption and free/history lifecycle: PASS")


if __name__ == "__main__":
    _self_check()
