from __future__ import annotations

"""Fail-closed continuous-decode harness for Energy-SLO Phase 3.

This file is scheduling/accounting infrastructure, not a model simulator.  It
never falls back to a full-sequence forward.  A formal run requires a backend
that explicitly advertises real continuous-engine, incremental-decode and KV
capabilities.  CPU tests may use a non-formal backend only to exercise the
scheduler and ledger invariants; such results are never scientifically
eligible.
"""

from dataclasses import dataclass, field
import math
import time
from typing import Any, Protocol, Sequence


REQUIRED_EVENT_FIELDS = (
    "request_id",
    "token_id",
    "phase",
    "arrival_ns",
    "enqueue_ns",
    "batch_id",
    "batch_seal_ns",
    "gpu_start_ns",
    "gpu_end_ns",
    "first_output_ns",
    "completion_ns",
    "precision_action",
    "completed",
    "slo_deadline_ns",
)


@dataclass(frozen=True)
class RequestSpec:
    request_id: str
    arrival_ns: int
    prompt_token_ids: tuple[int, ...]
    output_token_ids: tuple[int, ...]
    slo_deadline_ns: int

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id must be non-empty")
        if self.arrival_ns < 0:
            raise ValueError("arrival_ns must be non-negative")
        if self.slo_deadline_ns < self.arrival_ns:
            raise ValueError("slo deadline cannot precede arrival")
        if not self.prompt_token_ids:
            raise ValueError("prompt_token_ids must be non-empty")
        if not self.output_token_ids:
            raise ValueError("output_token_ids must be non-empty")


@dataclass(frozen=True)
class ServingEvent:
    request_id: str
    token_id: int
    phase: str
    arrival_ns: int
    enqueue_ns: int
    batch_id: int
    batch_seal_ns: int
    gpu_start_ns: int
    gpu_end_ns: int
    first_output_ns: int
    completion_ns: int | None
    precision_action: str
    completed: bool
    slo_deadline_ns: int

    def __post_init__(self) -> None:
        if self.token_id < 0:
            raise ValueError("token_id is the zero-based output-token index")
        if self.phase not in ("prefill", "decode"):
            raise ValueError("phase must be prefill or decode")
        if self.batch_id < 0:
            raise ValueError("batch_id must be non-negative")
        if not self.precision_action:
            raise ValueError("precision_action must be explicit")
        if not self.completed:
            raise ValueError(
                "ledger commits completed output-token events only; unfinished work stays in backlog"
            )
        if not (
            0 <= self.arrival_ns <= self.enqueue_ns <= self.batch_seal_ns
            <= self.gpu_start_ns <= self.gpu_end_ns
        ):
            raise ValueError("event timestamps violate arrival/enqueue/GPU order")
        if self.first_output_ns < self.gpu_end_ns and self.token_id == 0:
            raise ValueError("first output cannot precede the prefill result")
        if self.completion_ns is not None and self.completion_ns < self.gpu_end_ns:
            raise ValueError("request completion cannot precede this token")
        if self.slo_deadline_ns < self.arrival_ns:
            raise ValueError("slo deadline cannot precede arrival")

    def as_dict(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in REQUIRED_EVENT_FIELDS}


@dataclass(frozen=True)
class ServingSummary:
    registered_requests: int
    completed_requests: int
    unfinished_requests: int
    unfinished_request_ids: tuple[str, ...]
    completed_output_tokens: int
    backlog_output_tokens: int
    p99_ttft_ns: float | None
    p99_tpot_ns: float | None
    p99_tbt_ns: float | None
    slo_violation_rate: float | None


def _percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be in [0, 1]")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    index = quantile * (len(ordered) - 1)
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


class EventLedger:
    """Request registry plus append-only completed-output-token events."""

    def __init__(self) -> None:
        self._requests: dict[str, RequestSpec] = {}
        self._events: list[ServingEvent] = []
        self._events_by_request: dict[str, list[ServingEvent]] = {}

    @property
    def events(self) -> tuple[ServingEvent, ...]:
        return tuple(self._events)

    def register_request(self, request: RequestSpec) -> None:
        if request.request_id in self._requests:
            raise ValueError(f"duplicate request_id: {request.request_id}")
        self._requests[request.request_id] = request
        self._events_by_request[request.request_id] = []

    def append(self, event: ServingEvent) -> None:
        request = self._requests.get(event.request_id)
        if request is None:
            raise ValueError(f"event for unregistered request: {event.request_id}")
        if event.arrival_ns != request.arrival_ns:
            raise ValueError("event arrival_ns differs from registered request")
        if event.slo_deadline_ns != request.slo_deadline_ns:
            raise ValueError("event slo_deadline_ns differs from registered request")

        prior = self._events_by_request[event.request_id]
        if event.token_id != len(prior):
            raise ValueError("output-token events must be appended exactly once in order")
        if event.token_id >= len(request.output_token_ids):
            raise ValueError("event token_id exceeds requested output length")
        if event.token_id == 0 and event.phase != "prefill":
            raise ValueError("the first output must be produced by the one-time prefill")
        if event.token_id > 0 and event.phase != "decode":
            raise ValueError("subsequent outputs must come from length-1 decode")

        expected_first_output = event.gpu_end_ns if not prior else prior[0].first_output_ns
        if event.first_output_ns != expected_first_output:
            raise ValueError("first_output_ns must be stable for the request")
        is_final = event.token_id == len(request.output_token_ids) - 1
        if is_final and event.completion_ns != event.gpu_end_ns:
            raise ValueError("the final output must close the request at gpu_end_ns")
        if not is_final and event.completion_ns is not None:
            raise ValueError("non-final output cannot set request completion_ns")

        prior.append(event)
        self._events.append(event)

    @property
    def completed_output_tokens(self) -> int:
        return sum(1 for event in self._events if event.completed)

    def summary(self) -> ServingSummary:
        completed_ids: list[str] = []
        unfinished_ids: list[str] = []
        ttft_values: list[float] = []
        tpot_values: list[float] = []
        tbt_values: list[float] = []
        violations = 0
        backlog_tokens = 0

        for request_id, request in self._requests.items():
            events = self._events_by_request[request_id]
            backlog_tokens += len(request.output_token_ids) - len(events)
            complete = (
                len(events) == len(request.output_token_ids)
                and bool(events)
                and events[-1].completion_ns is not None
            )
            for left, right in zip(events, events[1:]):
                tbt_values.append(float(right.gpu_end_ns - left.gpu_end_ns))
            if not complete:
                unfinished_ids.append(request_id)
                continue

            completed_ids.append(request_id)
            first_output_ns = events[0].first_output_ns
            completion_ns = events[-1].completion_ns
            assert completion_ns is not None
            ttft_values.append(float(first_output_ns - request.arrival_ns))
            if len(events) > 1:
                tpot_values.append(
                    float(completion_ns - first_output_ns) / (len(events) - 1)
                )
            if completion_ns > request.slo_deadline_ns:
                violations += 1

        violation_rate = None
        if completed_ids:
            violation_rate = violations / len(completed_ids)
        return ServingSummary(
            registered_requests=len(self._requests),
            completed_requests=len(completed_ids),
            unfinished_requests=len(unfinished_ids),
            unfinished_request_ids=tuple(sorted(unfinished_ids)),
            completed_output_tokens=self.completed_output_tokens,
            backlog_output_tokens=backlog_tokens,
            p99_ttft_ns=_percentile(ttft_values, 0.99),
            p99_tpot_ns=_percentile(tpot_values, 0.99),
            p99_tbt_ns=_percentile(tbt_values, 0.99),
            slo_violation_rate=violation_rate,
        )


@dataclass(frozen=True)
class BackendCapabilities:
    incremental_decode: bool
    decode_input_length_one: bool
    per_request_kv: bool
    independent_policy_kv: bool
    mutable_active_set: bool
    prefill_once: bool
    no_kv_repack: bool
    real_continuous_engine: bool

    def missing(self, *, formal: bool) -> tuple[str, ...]:
        required = {
            "incremental_decode": self.incremental_decode,
            "decode_input_length_one": self.decode_input_length_one,
            "per_request_kv": self.per_request_kv,
            "independent_policy_kv": self.independent_policy_kv,
            "mutable_active_set": self.mutable_active_set,
            "prefill_once": self.prefill_once,
            "no_kv_repack": self.no_kv_repack,
        }
        if formal:
            required["real_continuous_engine"] = self.real_continuous_engine
        return tuple(name for name, available in required.items() if not available)


@dataclass(frozen=True)
class BackendStepResult:
    request_id: str
    kv_state: Any
    kv_handle_id: str
    kv_length: int
    gpu_start_ns: int
    gpu_end_ns: int
    precision_action: str

    def __post_init__(self) -> None:
        if not self.kv_handle_id:
            raise ValueError("backend must expose a stable per-request KV handle id")
        if self.kv_length < 1:
            raise ValueError("kv_length must be positive")
        if self.gpu_end_ns < self.gpu_start_ns:
            raise ValueError("gpu_end_ns cannot precede gpu_start_ns")
        if not self.precision_action:
            raise ValueError("precision_action must be explicit")


@dataclass(frozen=True)
class DecodeInput:
    request_id: str
    input_token_ids: tuple[int, ...]
    kv_state: Any
    kv_handle_id: str
    kv_length: int

    def __post_init__(self) -> None:
        if len(self.input_token_ids) != 1:
            raise ValueError("decode input length must be exactly 1")


class ContinuousDecodeBackend(Protocol):
    @property
    def name(self) -> str:
        ...

    @property
    def capabilities(self) -> BackendCapabilities:
        ...

    def prefill(
        self, request_id: str, prompt_token_ids: tuple[int, ...]
    ) -> BackendStepResult:
        ...

    def decode_batch(
        self, inputs: Sequence[DecodeInput]
    ) -> Sequence[BackendStepResult]:
        ...


class HarnessClock(Protocol):
    def now_ns(self) -> int:
        ...

    def wait_until_ns(self, target_ns: int) -> None:
        ...


class MonotonicHarnessClock:
    def now_ns(self) -> int:
        return time.monotonic_ns()

    def wait_until_ns(self, target_ns: int) -> None:
        remaining_ns = target_ns - self.now_ns()
        if remaining_ns > 0:
            time.sleep(remaining_ns / 1_000_000_000)


class BackendUnavailableError(RuntimeError):
    """A real backend may raise this to produce BLOCKED, never proxy results."""


@dataclass
class _ActiveRequest:
    spec: RequestSpec
    kv_state: Any
    kv_handle_id: str
    kv_length: int
    next_output_index: int
    enqueue_ns: int
    first_output_ns: int


@dataclass(frozen=True)
class HarnessResult:
    status: str
    block_reason: str | None
    backend_name: str | None
    policy_id: str
    ledger: EventLedger
    summary: ServingSummary
    active_set_sizes: tuple[int, ...]
    prefill_counts: dict[str, int]
    decode_input_lengths: tuple[int, ...]
    scientific_result_eligible: bool


class ContinuousDecodeHarness:
    def __init__(
        self,
        backend: ContinuousDecodeBackend | None,
        *,
        policy_id: str,
        formal: bool = True,
        clock: HarnessClock | None = None,
    ) -> None:
        if not policy_id:
            raise ValueError("policy_id must be non-empty")
        self.backend = backend
        self.policy_id = policy_id
        self.formal = formal
        self.clock = clock or MonotonicHarnessClock()

    def _blocked(
        self,
        ledger: EventLedger,
        reason: str,
        *,
        active_set_sizes: Sequence[int] = (),
        prefill_counts: dict[str, int] | None = None,
        decode_input_lengths: Sequence[int] = (),
    ) -> HarnessResult:
        return HarnessResult(
            status="BLOCKED",
            block_reason=reason,
            backend_name=self.backend.name if self.backend is not None else None,
            policy_id=self.policy_id,
            ledger=ledger,
            summary=ledger.summary(),
            active_set_sizes=tuple(active_set_sizes),
            prefill_counts=dict(prefill_counts or {}),
            decode_input_lengths=tuple(decode_input_lengths),
            scientific_result_eligible=False,
        )

    def run(
        self,
        requests: Sequence[RequestSpec],
        *,
        max_iterations: int | None = None,
    ) -> HarnessResult:
        if not requests:
            raise ValueError("at least one request is required")
        if max_iterations is not None and max_iterations < 1:
            raise ValueError("max_iterations must be positive")

        ledger = EventLedger()
        for request in requests:
            ledger.register_request(request)
        if self.backend is None:
            return self._blocked(ledger, "no continuous-decode backend configured")
        missing = self.backend.capabilities.missing(formal=self.formal)
        if missing:
            return self._blocked(
                ledger,
                "backend lacks frozen capabilities: " + ",".join(missing),
            )
        if self.formal:
            for request in requests:
                if len(request.prompt_token_ids) != 64 or len(request.output_token_ids) != 64:
                    return self._blocked(
                        ledger,
                        "formal protocol requires prompt=64 and teacher-forced decode=64",
                    )

        pending = sorted(requests, key=lambda item: (item.arrival_ns, item.request_id))
        pending_index = 0
        active: dict[str, _ActiveRequest] = {}
        prefill_counts = {request.request_id: 0 for request in requests}
        decode_input_lengths: list[int] = []
        active_set_sizes: list[int] = [0]
        batch_id = 0
        iterations = 0

        try:
            while pending_index < len(pending) or active:
                if max_iterations is not None and iterations >= max_iterations:
                    return self._blocked(
                        ledger,
                        "iteration limit reached before complete drain",
                        active_set_sizes=active_set_sizes,
                        prefill_counts=prefill_counts,
                        decode_input_lengths=decode_input_lengths,
                    )
                if not active and pending_index < len(pending):
                    self.clock.wait_until_ns(pending[pending_index].arrival_ns)

                boundary_ns = self.clock.now_ns()
                admitted: list[RequestSpec] = []
                while (
                    pending_index < len(pending)
                    and pending[pending_index].arrival_ns <= boundary_ns
                ):
                    admitted.append(pending[pending_index])
                    pending_index += 1

                for request in admitted:
                    prefill_counts[request.request_id] += 1
                    if prefill_counts[request.request_id] != 1:
                        raise RuntimeError("request was prefetched/prefilled more than once")
                    batch_seal_ns = self.clock.now_ns()
                    result = self.backend.prefill(
                        request.request_id, request.prompt_token_ids
                    )
                    if result.request_id != request.request_id:
                        raise RuntimeError("prefill returned a different request_id")
                    if result.precision_action.upper() != "BF16":
                        raise RuntimeError("frozen protocol requires BF16 prefill")
                    if result.kv_length != len(request.prompt_token_ids):
                        raise RuntimeError("prefill KV length does not equal prompt length")
                    if result.gpu_start_ns < batch_seal_ns:
                        raise RuntimeError("backend GPU start precedes batch seal")
                    completion_ns = (
                        result.gpu_end_ns if len(request.output_token_ids) == 1 else None
                    )
                    ledger.append(
                        ServingEvent(
                            request_id=request.request_id,
                            token_id=0,
                            phase="prefill",
                            arrival_ns=request.arrival_ns,
                            enqueue_ns=request.arrival_ns,
                            batch_id=batch_id,
                            batch_seal_ns=batch_seal_ns,
                            gpu_start_ns=result.gpu_start_ns,
                            gpu_end_ns=result.gpu_end_ns,
                            first_output_ns=result.gpu_end_ns,
                            completion_ns=completion_ns,
                            precision_action="BF16",
                            completed=True,
                            slo_deadline_ns=request.slo_deadline_ns,
                        )
                    )
                    if completion_ns is None:
                        active[request.request_id] = _ActiveRequest(
                            spec=request,
                            kv_state=result.kv_state,
                            kv_handle_id=result.kv_handle_id,
                            kv_length=result.kv_length,
                            next_output_index=1,
                            enqueue_ns=request.arrival_ns,
                            first_output_ns=result.gpu_end_ns,
                        )
                active_set_sizes.append(len(active))

                handle_ids = [state.kv_handle_id for state in active.values()]
                if len(handle_ids) != len(set(handle_ids)):
                    raise RuntimeError("two active requests share a KV handle")

                if active:
                    batch_seal_ns = self.clock.now_ns()
                    decode_inputs: list[DecodeInput] = []
                    for request_id, state in active.items():
                        previous_teacher_token = state.spec.output_token_ids[
                            state.next_output_index - 1
                        ]
                        decode_input = DecodeInput(
                            request_id=request_id,
                            input_token_ids=(previous_teacher_token,),
                            kv_state=state.kv_state,
                            kv_handle_id=state.kv_handle_id,
                            kv_length=state.kv_length,
                        )
                        decode_inputs.append(decode_input)
                        decode_input_lengths.append(len(decode_input.input_token_ids))

                    outputs = tuple(self.backend.decode_batch(decode_inputs))
                    output_by_id = {output.request_id: output for output in outputs}
                    if len(output_by_id) != len(outputs):
                        raise RuntimeError("decode backend returned duplicate request ids")
                    if set(output_by_id) != set(active):
                        raise RuntimeError("decode backend did not return exactly the active set")

                    completed_ids: list[str] = []
                    for request_id, state in active.items():
                        output = output_by_id[request_id]
                        if output.kv_handle_id != state.kv_handle_id:
                            raise RuntimeError("backend changed a request's KV identity")
                        if output.kv_length != state.kv_length + 1:
                            raise RuntimeError("decode KV length must grow monotonically by one")
                        if output.gpu_start_ns < batch_seal_ns:
                            raise RuntimeError("backend GPU start precedes batch seal")
                        token_id = state.next_output_index
                        is_final = token_id == len(state.spec.output_token_ids) - 1
                        completion_ns = output.gpu_end_ns if is_final else None
                        ledger.append(
                            ServingEvent(
                                request_id=request_id,
                                token_id=token_id,
                                phase="decode",
                                arrival_ns=state.spec.arrival_ns,
                                enqueue_ns=state.enqueue_ns,
                                batch_id=batch_id,
                                batch_seal_ns=batch_seal_ns,
                                gpu_start_ns=output.gpu_start_ns,
                                gpu_end_ns=output.gpu_end_ns,
                                first_output_ns=state.first_output_ns,
                                completion_ns=completion_ns,
                                precision_action=output.precision_action,
                                completed=True,
                                slo_deadline_ns=state.spec.slo_deadline_ns,
                            )
                        )
                        state.kv_state = output.kv_state
                        state.kv_length = output.kv_length
                        state.next_output_index += 1
                        if is_final:
                            completed_ids.append(request_id)
                    for request_id in completed_ids:
                        del active[request_id]
                    active_set_sizes.append(len(active))

                batch_id += 1
                iterations += 1
        except BackendUnavailableError as exc:
            return self._blocked(
                ledger,
                f"backend unavailable: {exc}",
                active_set_sizes=active_set_sizes,
                prefill_counts=prefill_counts,
                decode_input_lengths=decode_input_lengths,
            )

        summary = ledger.summary()
        if summary.unfinished_requests or summary.backlog_output_tokens:
            return self._blocked(
                ledger,
                "run ended without complete drain",
                active_set_sizes=active_set_sizes,
                prefill_counts=prefill_counts,
                decode_input_lengths=decode_input_lengths,
            )
        if any(count != 1 for count in prefill_counts.values()):
            raise RuntimeError("every request must prefill exactly once")
        if any(length != 1 for length in decode_input_lengths):
            raise RuntimeError("every decode input must have length 1")
        if len(set(active_set_sizes)) < 2:
            return self._blocked(
                ledger,
                "active set never changed",
                active_set_sizes=active_set_sizes,
                prefill_counts=prefill_counts,
                decode_input_lengths=decode_input_lengths,
            )
        if self.formal and max(active_set_sizes) < 2:
            return self._blocked(
                ledger,
                "formal continuous-batching trace never had two active requests",
                active_set_sizes=active_set_sizes,
                prefill_counts=prefill_counts,
                decode_input_lengths=decode_input_lengths,
            )

        # This module is an implementation-audit harness, not the frozen
        # top-level experiment runner.  Backend capability flags are useful for
        # early refusal, but they are self-reported and therefore cannot grant
        # scientific eligibility.  A future concrete runner must bind an
        # independently reviewed backend type/commit, Phase-4 attestation and
        # the complete code/config/data manifest before it can authorize a
        # formal result.
        if self.formal:
            return self._blocked(
                ledger,
                "abstract harness cannot authorize a formal scientific result; "
                "reviewed top-level runner and backend attestation are missing",
                active_set_sizes=active_set_sizes,
                prefill_counts=prefill_counts,
                decode_input_lengths=decode_input_lengths,
            )

        return HarnessResult(
            status="COMPLETED",
            block_reason=None,
            backend_name=self.backend.name,
            policy_id=self.policy_id,
            ledger=ledger,
            summary=summary,
            active_set_sizes=tuple(active_set_sizes),
            prefill_counts=dict(prefill_counts),
            decode_input_lengths=tuple(decode_input_lengths),
            scientific_result_eligible=False,
        )
