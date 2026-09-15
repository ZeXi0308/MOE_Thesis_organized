from __future__ import annotations

import sys
from pathlib import Path
import unittest


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from power_accounting import (  # noqa: E402
    MonotonicNVMLSampler,
    PowerSample,
    PowerTrace,
    account_power_trace,
    assert_matching_gpu_uuid,
    cumulative_counter_delta_j,
    integrate_power_samples,
    validate_formal_sample_intervals,
)


class PowerAccountingTest(unittest.TestCase):
    def _constant_trace(
        self, power_w: float = 100.0, *, counter_delta_j: float | None = None
    ) -> PowerTrace:
        samples = (
            PowerSample(0, power_w),
            PowerSample(5_000_000_000, power_w),
            PowerSample(10_000_000_000, power_w),
        )
        return PowerTrace(
            samples=samples,
            start_ns=0,
            end_ns=10_000_000_000,
            gpu_uuid="GPU-1234-5678",
            total_energy_counter_delta_j=counter_delta_j,
        )

    def test_frozen_synthetic_total_dynamic_and_completed_token_denominator(self) -> None:
        result = account_power_trace(
            self._constant_trace(),
            completed_output_tokens=100,
            idle_power_w=30.0,
        )
        self.assertAlmostEqual(result.total_energy_j, 1000.0)
        self.assertAlmostEqual(result.dynamic_energy_j or -1.0, 700.0)
        self.assertAlmostEqual(result.total_j_per_completed_token, 10.0)
        self.assertAlmostEqual(result.dynamic_j_per_completed_token or -1.0, 7.0)
        self.assertEqual(result.total_source, "monotonic_power_integral")

    def test_total_energy_counter_is_primary_but_dynamic_stays_sampled(self) -> None:
        result = account_power_trace(
            self._constant_trace(counter_delta_j=900.0),
            completed_output_tokens=100,
            idle_power_w=30.0,
        )
        self.assertEqual(result.total_source, "nvml_total_energy_counter")
        self.assertAlmostEqual(result.total_energy_j, 900.0)
        self.assertAlmostEqual(result.total_j_per_completed_token, 9.0)
        self.assertAlmostEqual(result.dynamic_energy_j or -1.0, 700.0)

    def test_zero_or_boolean_completed_token_count_is_rejected(self) -> None:
        for invalid in (0, -1, False, 1.5):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                account_power_trace(
                    self._constant_trace(),
                    completed_output_tokens=invalid,
                    idle_power_w=30.0,
                )

    def test_timestamps_must_be_strictly_monotonic(self) -> None:
        samples = (
            PowerSample(0, 100.0),
            PowerSample(1_000, 110.0),
            PowerSample(1_000, 120.0),
        )
        with self.assertRaises(ValueError):
            integrate_power_samples(samples)

    def test_trace_requires_explicit_boundary_samples(self) -> None:
        samples = (
            PowerSample(1, 100.0),
            PowerSample(10, 100.0),
        )
        with self.assertRaisesRegex(ValueError, "start boundary"):
            PowerTrace(samples=samples, start_ns=0, end_ns=10)

    def test_nvml_cuda_uuid_match_is_normalized_and_mismatch_fails(self) -> None:
        assert_matching_gpu_uuid("GPU-abcd-1234", "ABCD1234")
        with self.assertRaises(RuntimeError):
            assert_matching_gpu_uuid("GPU-abcd-1234", "GPU-dead-beef")

    def test_formal_trace_rejects_observed_gap_over_twenty_ms(self) -> None:
        samples = (
            PowerSample(0, 100.0),
            PowerSample(10_000_000, 100.0),
            PowerSample(110_000_000, 100.0),
        )
        trace = PowerTrace(
            samples=samples,
            start_ns=0,
            end_ns=110_000_000,
            gpu_uuid="GPU-test",
        )
        with self.assertRaisesRegex(RuntimeError, "sampling-gap"):
            validate_formal_sample_intervals(samples)
        with self.assertRaisesRegex(RuntimeError, "sampling-gap"):
            account_power_trace(
                trace,
                completed_output_tokens=1,
                idle_power_w=None,
                formal=True,
            )

    def test_counter_wraparound_needs_explicit_modulus(self) -> None:
        self.assertAlmostEqual(cumulative_counter_delta_j(10.0, 12.5), 2.5)
        self.assertAlmostEqual(
            cumulative_counter_delta_j(98.0, 3.0, counter_modulus_j=100.0),
            5.0,
        )
        with self.assertRaisesRegex(RuntimeError, "wrap metadata"):
            cumulative_counter_delta_j(98.0, 3.0)

    def test_idle_subtraction_changes_auxiliary_metric_not_raw_board_energy(self) -> None:
        low_idle = account_power_trace(
            self._constant_trace(), completed_output_tokens=100, idle_power_w=20.0
        )
        high_idle = account_power_trace(
            self._constant_trace(), completed_output_tokens=100, idle_power_w=40.0
        )
        self.assertEqual(low_idle.total_energy_j, high_idle.total_energy_j)
        self.assertNotEqual(low_idle.dynamic_energy_j, high_idle.dynamic_energy_j)

    def test_background_sampler_failure_is_not_silent(self) -> None:
        class FailingBackend:
            gpu_uuid = "GPU-test"

            def __init__(self) -> None:
                self.reads = 0

            def read_power_w(self) -> float:
                self.reads += 1
                if self.reads >= 2:
                    raise RuntimeError("synthetic NVML failure")
                return 100.0

            def read_total_energy_j(self) -> float | None:
                return None

        sampler = MonotonicNVMLSampler(
            FailingBackend(), interval_s=0.001, formal=False
        )
        sampler.start()
        sampler._thread.join(timeout=1.0)  # deterministic test of background path
        with self.assertRaisesRegex(RuntimeError, "background read failed"):
            sampler.stop()


if __name__ == "__main__":
    unittest.main()
