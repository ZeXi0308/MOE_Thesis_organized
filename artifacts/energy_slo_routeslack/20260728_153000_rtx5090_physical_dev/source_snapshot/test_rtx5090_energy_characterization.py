from __future__ import annotations

import unittest

from run_rtx5090_energy_characterization import (
    ActivationRecord,
    CharacterizationError,
    TelemetrySampler,
    characterization_status,
    counter_delta_j,
    exact_expert_row_denominator,
    plan_disjoint_activation_groups,
    validate_telemetry_window,
)


class CounterAndDenominatorTest(unittest.TestCase):
    def test_counter_delta_requires_monotonic_cumulative_counter(self) -> None:
        self.assertEqual(counter_delta_j(1_000, 4_250), 3.25)
        with self.assertRaisesRegex(CharacterizationError, "moved backwards"):
            counter_delta_j(4_250, 1_000)

    def test_processed_expert_rows_are_exactly_repeats_times_rows(self) -> None:
        self.assertEqual(exact_expert_row_denominator(repeats=17, rows=32), 544)
        with self.assertRaises(CharacterizationError):
            exact_expert_row_denominator(repeats=0, rows=32)

    def test_filtered_windows_cannot_claim_unqualified_completion(self) -> None:
        self.assertEqual(
            characterization_status(
                {1: 4, 8: 4}, requested_rows=(1, 8), trials=4
            ),
            "CHARACTERIZATION_COMPLETE",
        )
        self.assertEqual(
            characterization_status(
                {1: 2, 8: 3}, requested_rows=(1, 8), trials=4
            ),
            "CHARACTERIZATION_COMPLETE_WITH_FILTERED_WINDOWS",
        )
        self.assertEqual(
            characterization_status(
                {1: 1, 8: 4}, requested_rows=(1, 8), trials=4
            ),
            "CHARACTERIZATION_INCOMPLETE_INVALID_WINDOWS",
        )


class ActivationGroupingTest(unittest.TestCase):
    def test_groups_never_reuse_a_source_request(self) -> None:
        records = tuple(
            ActivationRecord(
                request_id=f"request-{index}",
                forward_id=f"request-{index}:prefill:0",
                row_count=8,
                record_index=index,
            )
            for index in range(20)
        )
        plans = plan_disjoint_activation_groups(records, row_grid=(1, 8, 16), trials=2)
        requests = [request for plan in plans.values() for request in plan.request_ids]
        self.assertEqual(len(requests), len(set(requests)))
        self.assertEqual(set(plans), {(1, 0), (1, 1), (8, 0), (8, 1), (16, 0), (16, 1)})

    def test_insufficient_disjoint_records_fail_closed(self) -> None:
        records = (
            ActivationRecord("one", "one:prefill:0", 4, 0),
            ActivationRecord("two", "two:prefill:0", 4, 1),
        )
        with self.assertRaisesRegex(CharacterizationError, "not enough"):
            plan_disjoint_activation_groups(records, row_grid=(8,), trials=2)


class TelemetryContractTest(unittest.TestCase):
    def _sample(self, timestamp: int, temperature: float = 55.0) -> dict[str, object]:
        return {
            "monotonic_ns": timestamp,
            "temperature_c": temperature,
            "gpu_uuid": "GPU-one",
            "power_limit_w": 575.0,
        }

    def test_gap_and_thermal_drift_invalidate_the_whole_window(self) -> None:
        good = (self._sample(0), self._sample(5_000_000, 55.5))
        self.assertEqual(
            validate_telemetry_window(
                good, max_gap_s=0.02, max_temperature_range_c=2.0
            ),
            (),
        )
        bad = (self._sample(0), self._sample(30_000_000, 58.0))
        reasons = validate_telemetry_window(
            bad, max_gap_s=0.02, max_temperature_range_c=2.0
        )
        self.assertIn("telemetry_gap_exceeds_limit", reasons)
        self.assertIn("temperature_range_exceeds_limit", reasons)

    def test_sampler_surfaces_background_failures(self) -> None:
        calls = 0

        def sample(*, kind: str) -> dict[str, object]:
            nonlocal calls
            calls += 1
            if calls > 1:
                raise RuntimeError("synthetic NVML failure")
            return self._sample(calls)

        sampler = TelemetrySampler(object(), interval_s=0.001, sample_function=sample)
        sampler.start()
        sampler._thread.join(timeout=1.0)
        with self.assertRaisesRegex(CharacterizationError, "background read failed"):
            sampler.stop()


if __name__ == "__main__":
    unittest.main()
