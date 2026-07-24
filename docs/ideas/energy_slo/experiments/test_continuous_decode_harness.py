from __future__ import annotations

from dataclasses import replace
import sys
from pathlib import Path
import unittest


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from continuous_decode_harness import (  # noqa: E402
    BackendCapabilities,
    BackendStepResult,
    ContinuousDecodeHarness,
    DecodeInput,
    EventLedger,
    REQUIRED_EVENT_FIELDS,
    RequestSpec,
    ServingEvent,
)


class FakeClock:
    def __init__(self, now_ns: int = 0) -> None:
        self.value = now_ns

    def now_ns(self) -> int:
        return self.value

    def wait_until_ns(self, target_ns: int) -> None:
        self.value = max(self.value, target_ns)

    def advance(self, delta_ns: int) -> None:
        self.value += delta_ns


class AuditOnlyBackend:
    """Scheduler test double; never eligible for a formal scientific run."""

    def __init__(self, clock: FakeClock) -> None:
        self.clock = clock
        self.prefill_calls: list[tuple[str, tuple[int, ...]]] = []
        self.decode_calls: list[tuple[DecodeInput, ...]] = []
        self.kv_length_history: dict[str, list[int]] = {}
        self._capabilities = BackendCapabilities(
            incremental_decode=True,
            decode_input_length_one=True,
            per_request_kv=True,
            independent_policy_kv=True,
            mutable_active_set=True,
            prefill_once=True,
            no_kv_repack=True,
            real_continuous_engine=False,
        )

    @property
    def name(self) -> str:
        return "audit-only-test-backend"

    @property
    def capabilities(self) -> BackendCapabilities:
        return self._capabilities

    def prefill(
        self, request_id: str, prompt_token_ids: tuple[int, ...]
    ) -> BackendStepResult:
        self.prefill_calls.append((request_id, prompt_token_ids))
        start = self.clock.now_ns()
        self.clock.advance(10)
        kv_length = len(prompt_token_ids)
        self.kv_length_history[request_id] = [kv_length]
        return BackendStepResult(
            request_id=request_id,
            kv_state={"owner": request_id, "length": kv_length},
            kv_handle_id=f"kv:{request_id}",
            kv_length=kv_length,
            gpu_start_ns=start,
            gpu_end_ns=self.clock.now_ns(),
            precision_action="BF16",
        )

    def decode_batch(
        self, inputs: tuple[DecodeInput, ...] | list[DecodeInput]
    ) -> tuple[BackendStepResult, ...]:
        inputs = tuple(inputs)
        self.decode_calls.append(inputs)
        start = self.clock.now_ns()
        self.clock.advance(10)
        end = self.clock.now_ns()
        outputs = []
        for item in inputs:
            self.assert_owned(item)
            next_length = item.kv_length + 1
            self.kv_length_history[item.request_id].append(next_length)
            outputs.append(
                BackendStepResult(
                    request_id=item.request_id,
                    kv_state={"owner": item.request_id, "length": next_length},
                    kv_handle_id=item.kv_handle_id,
                    kv_length=next_length,
                    gpu_start_ns=start,
                    gpu_end_ns=end,
                    precision_action="BF16",
                )
            )
        return tuple(outputs)

    @staticmethod
    def assert_owned(item: DecodeInput) -> None:
        if item.kv_state["owner"] != item.request_id:
            raise AssertionError("request consumed another request's KV")
        if len(item.input_token_ids) != 1:
            raise AssertionError("decode was not length 1")


def request(
    request_id: str,
    *,
    arrival_ns: int,
    prompt_len: int,
    output_len: int,
) -> RequestSpec:
    return RequestSpec(
        request_id=request_id,
        arrival_ns=arrival_ns,
        prompt_token_ids=tuple(range(prompt_len)),
        output_token_ids=tuple(range(100, 100 + output_len)),
        slo_deadline_ns=10_000,
    )


class ContinuousDecodeHarnessTest(unittest.TestCase):
    def test_prefill_once_length_one_decode_independent_kv_and_active_mutation(self) -> None:
        clock = FakeClock()
        backend = AuditOnlyBackend(clock)
        harness = ContinuousDecodeHarness(
            backend, policy_id="cpu-audit", formal=False, clock=clock
        )
        result = harness.run(
            (
                request("r1", arrival_ns=0, prompt_len=3, output_len=4),
                request("r2", arrival_ns=15, prompt_len=2, output_len=3),
            )
        )

        self.assertEqual(result.status, "COMPLETED")
        self.assertFalse(result.scientific_result_eligible)
        self.assertEqual(result.prefill_counts, {"r1": 1, "r2": 1})
        self.assertTrue(result.decode_input_lengths)
        self.assertEqual(set(result.decode_input_lengths), {1})
        self.assertIn(2, result.active_set_sizes)
        self.assertIn(0, result.active_set_sizes)
        self.assertEqual(backend.kv_length_history["r1"], [3, 4, 5, 6])
        self.assertEqual(backend.kv_length_history["r2"], [2, 3, 4])
        self.assertEqual(result.summary.completed_requests, 2)
        self.assertEqual(result.summary.completed_output_tokens, 7)
        self.assertEqual(result.summary.backlog_output_tokens, 0)
        self.assertIsNotNone(result.summary.p99_ttft_ns)
        self.assertIsNotNone(result.summary.p99_tpot_ns)
        self.assertIsNotNone(result.summary.p99_tbt_ns)
        self.assertEqual(
            set(result.ledger.events[0].as_dict()), set(REQUIRED_EVENT_FIELDS)
        )

    def test_missing_backend_capability_returns_blocked_without_calls(self) -> None:
        clock = FakeClock()
        backend = AuditOnlyBackend(clock)
        backend._capabilities = replace(
            backend.capabilities, incremental_decode=False
        )
        result = ContinuousDecodeHarness(
            backend, policy_id="blocked", formal=False, clock=clock
        ).run((request("r1", arrival_ns=0, prompt_len=2, output_len=2),))
        self.assertEqual(result.status, "BLOCKED")
        self.assertIn("incremental_decode", result.block_reason or "")
        self.assertEqual(backend.prefill_calls, [])
        self.assertEqual(backend.decode_calls, [])

    def test_audit_backend_cannot_be_used_as_formal_engine(self) -> None:
        clock = FakeClock()
        backend = AuditOnlyBackend(clock)
        result = ContinuousDecodeHarness(
            backend, policy_id="formal", formal=True, clock=clock
        ).run((request("r1", arrival_ns=0, prompt_len=64, output_len=64),))
        self.assertEqual(result.status, "BLOCKED")
        self.assertIn("real_continuous_engine", result.block_reason or "")
        self.assertFalse(result.scientific_result_eligible)

    def test_self_reported_real_engine_cannot_grant_formal_eligibility(self) -> None:
        clock = FakeClock()
        backend = AuditOnlyBackend(clock)
        backend._capabilities = replace(
            backend.capabilities, real_continuous_engine=True
        )
        result = ContinuousDecodeHarness(
            backend, policy_id="spoofed-formal", formal=True, clock=clock
        ).run(
            (
                request("r1", arrival_ns=0, prompt_len=64, output_len=64),
                request("r2", arrival_ns=0, prompt_len=64, output_len=64),
            )
        )
        self.assertEqual(result.status, "BLOCKED")
        self.assertIn("cannot authorize", result.block_reason or "")
        self.assertFalse(result.scientific_result_eligible)

    def test_no_backend_is_explicitly_blocked(self) -> None:
        result = ContinuousDecodeHarness(
            None, policy_id="none", formal=True, clock=FakeClock()
        ).run((request("r1", arrival_ns=0, prompt_len=64, output_len=64),))
        self.assertEqual(result.status, "BLOCKED")
        self.assertIn("no continuous-decode backend", result.block_reason or "")

    def test_iteration_limit_keeps_completed_tokens_but_excludes_backlog(self) -> None:
        clock = FakeClock()
        backend = AuditOnlyBackend(clock)
        result = ContinuousDecodeHarness(
            backend, policy_id="partial", formal=False, clock=clock
        ).run(
            (request("r1", arrival_ns=0, prompt_len=2, output_len=5),),
            max_iterations=1,
        )
        self.assertEqual(result.status, "BLOCKED")
        self.assertEqual(result.summary.completed_output_tokens, 2)
        self.assertEqual(result.summary.backlog_output_tokens, 3)
        self.assertEqual(result.summary.completed_requests, 0)
        self.assertEqual(result.summary.unfinished_request_ids, ("r1",))
        self.assertIsNone(result.summary.p99_ttft_ns)
        self.assertIsNone(result.summary.p99_tpot_ns)


class EventLedgerTest(unittest.TestCase):
    def _event(
        self,
        request_id: str,
        token_id: int,
        *,
        gpu_start_ns: int,
        gpu_end_ns: int,
        first_output_ns: int,
        completion_ns: int | None,
    ) -> ServingEvent:
        return ServingEvent(
            request_id=request_id,
            token_id=token_id,
            phase="prefill" if token_id == 0 else "decode",
            arrival_ns=0,
            enqueue_ns=0,
            batch_id=token_id,
            batch_seal_ns=gpu_start_ns,
            gpu_start_ns=gpu_start_ns,
            gpu_end_ns=gpu_end_ns,
            first_output_ns=first_output_ns,
            completion_ns=completion_ns,
            precision_action="BF16",
            completed=True,
            slo_deadline_ns=10_000,
        )

    def test_p99_uses_completed_requests_and_unfinished_tokens_stay_backlog(self) -> None:
        ledger = EventLedger()
        ledger.register_request(request("done", arrival_ns=0, prompt_len=2, output_len=2))
        ledger.register_request(request("open", arrival_ns=0, prompt_len=2, output_len=3))

        ledger.append(
            self._event(
                "done", 0, gpu_start_ns=0, gpu_end_ns=10,
                first_output_ns=10, completion_ns=None,
            )
        )
        ledger.append(
            self._event(
                "done", 1, gpu_start_ns=10, gpu_end_ns=20,
                first_output_ns=10, completion_ns=20,
            )
        )
        ledger.append(
            self._event(
                "open", 0, gpu_start_ns=0, gpu_end_ns=10,
                first_output_ns=10, completion_ns=None,
            )
        )

        summary = ledger.summary()
        self.assertEqual(summary.completed_output_tokens, 3)
        self.assertEqual(summary.completed_requests, 1)
        self.assertEqual(summary.unfinished_requests, 1)
        self.assertEqual(summary.unfinished_request_ids, ("open",))
        self.assertEqual(summary.backlog_output_tokens, 2)
        self.assertEqual(summary.p99_ttft_ns, 10.0)
        self.assertEqual(summary.p99_tpot_ns, 10.0)
        self.assertEqual(summary.p99_tbt_ns, 10.0)


if __name__ == "__main__":
    unittest.main()
