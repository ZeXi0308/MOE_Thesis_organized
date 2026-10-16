"""CPU-only accounting checks; these do not establish GPU sensor accuracy."""

import json
import time
import unittest

from power import PowerSampler, integrate_power


class FakeNVML:
    def __init__(self, counter="ok", power=True):
        self.counter = counter
        self.power = power
        self.start_ns = time.monotonic_ns()
        self.shutdown_count = 0

    def nvmlInit(self):
        pass

    def nvmlShutdown(self):
        self.shutdown_count += 1

    def nvmlDeviceGetHandleByIndex(self, index):
        return index

    def nvmlDeviceGetPowerUsage(self, handle):
        if not self.power:
            raise RuntimeError("power unavailable")
        return 100_000  # 100 W

    def nvmlDeviceGetTotalEnergyConsumption(self, handle):
        if self.counter == "unsupported":
            raise RuntimeError("counter unavailable")
        if self.counter == "stalled":
            return 42
        # Deliberately report 200 W in the counter vs 100 W sample: precedence.
        return int((time.monotonic_ns() - self.start_ns) * 200 / 1e6)


class PowerTests(unittest.TestCase):
    def test_clipped_trapezoids_and_missing_brackets(self):
        samples = [{"monotonic_ns": 0, "power_w": 0.0},
                   {"monotonic_ns": 2_000_000_000, "power_w": 20.0}]
        self.assertAlmostEqual(integrate_power(samples, 500_000_000, 1_500_000_000), 10.0)
        self.assertIsNone(integrate_power(samples, -1, 1_000_000_000))
        self.assertIsNone(integrate_power(samples[:1], 0, 1))

    def test_counter_precedence_context_sync_and_raw_data(self):
        backend = FakeNVML()
        meter = PowerSampler(interval_s=0.005, idle_power_w=50, nvml_module=backend)
        sync_calls = []
        with meter.measure(lambda: sync_calls.append(1)) as measurement:
            time.sleep(0.025)
        result = measurement.result
        self.assertEqual(sync_calls, [1, 1])
        self.assertEqual(result["method"], "nvml_total_energy_counter")
        self.assertAlmostEqual(result["energy_j"], result["elapsed_s"] * 200, delta=0.005)
        self.assertAlmostEqual(result["sampled_energy_j"], result["elapsed_s"] * 100, places=7)
        self.assertAlmostEqual(result["dynamic_energy_j"], result["energy_j"] - 50 * result["elapsed_s"])
        self.assertGreaterEqual(len(result["samples"]), 3)
        self.assertFalse(result["workload_failed"])
        json.dumps(result, allow_nan=False)
        meter.close()
        meter.close()
        self.assertEqual(backend.shutdown_count, 1)

    def test_unsupported_and_stalled_counter_fallback_preserves_negative(self):
        for mode in ("unsupported", "stalled"):
            meter = PowerSampler(interval_s=0.005, idle_power_w=120,
                                 nvml_module=FakeNVML(counter=mode))
            meter.start()
            time.sleep(0.01)
            result = meter.stop()
            self.assertEqual(result["method"], "nvml_power_trapezoid")
            self.assertAlmostEqual(result["energy_j"], result["elapsed_s"] * 100, places=7)
            self.assertLess(result["dynamic_energy_j"], 0)
            meter.close()

    def test_unavailable_energy_is_not_fabricated_zero(self):
        meter = PowerSampler(nvml_module=FakeNVML(counter="unsupported", power=False))
        with meter.measure() as measurement:
            pass
        self.assertIsNone(measurement.result["energy_j"])
        self.assertEqual(measurement.result["method"], "unavailable")
        meter.close()

    def test_exception_stops_measurement_and_preserves_failure(self):
        meter = PowerSampler(nvml_module=FakeNVML())
        with self.assertRaisesRegex(ValueError, "workload failure"):
            with meter.measure() as measurement:
                raise ValueError("workload failure")
        self.assertTrue(measurement.result["workload_failed"])
        self.assertFalse(meter._running)
        meter.close()


if __name__ == "__main__":
    unittest.main()
