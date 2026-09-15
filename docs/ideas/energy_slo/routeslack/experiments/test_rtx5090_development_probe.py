from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from run_rtx5090_development_probe import (  # noqa: E402
    ABBA,
    ARM_A,
    ARM_B,
    EnergyEnvelope,
    ProbeError,
    SCHEMA,
    TimedResult,
    abba_schedule,
    choose_equal_repeats,
    measure_same_window,
    validate_development_records,
    validate_telemetry,
    write_manifest,
)


def telemetry(timestamp_ns: int, temperature_c: float = 50.0) -> dict[str, object]:
    return {
        "timestamp_ns": timestamp_ns,
        "temperature_c": temperature_c,
        "power_w": 300.0,
        "sm_clock_mhz": 2200,
        "graphics_clock_mhz": 2200,
        "memory_clock_mhz": 14000,
        "power_limit_w": 575.0,
        "performance_state": 0,
        "clock_throttle_reasons": 0,
        "gpu_utilization_percent": 95,
        "memory_utilization_percent": 10,
    }


def envelope() -> EnergyEnvelope:
    return EnergyEnvelope(
        raw_board_energy_j=25.0,
        energy_source="monotonic_power_integral",
        power_integral_j=25.0,
        counter_energy_j=None,
        energy_window_start_ns=1_000_000_000,
        energy_window_end_ns=1_010_000_000,
        host_cuda_record_start_ns=1_001_000_000,
        host_cuda_sync_end_ns=1_009_000_000,
        power_samples=(
            {"timestamp_ns": 1_000_000_000, "power_w": 250.0},
            {"timestamp_ns": 1_005_000_000, "power_w": 300.0},
            {"timestamp_ns": 1_010_000_000, "power_w": 275.0},
        ),
        maximum_power_sample_gap_s=0.005,
        telemetry_start=telemetry(999_000_000, 50.0),
        telemetry_end=telemetry(1_011_000_000, 51.0),
        counter_reads=(
            {"read_start_ns": 999_500_000, "read_end_ns": 999_600_000, "energy_j": None},
            {"read_start_ns": 1_010_100_000, "read_end_ns": 1_010_200_000, "energy_j": None},
        ),
    )


def raw_record(block: int, position: int, arm: str, repeats: int = 8) -> dict[str, object]:
    value = envelope()
    return {
        "schema": SCHEMA,
        "formal_result": False,
        "scientific_result_eligible": False,
        "block": block,
        "position": position,
        "arm": arm,
        "same_execution_cuda_latency_and_energy": True,
        "repeats": repeats,
        "cuda_latency_ms": 8.0,
        "raw_board_energy_j": value.raw_board_energy_j,
        "energy_source": value.energy_source,
        "counter_energy_j": value.counter_energy_j,
        "power_integral_j": value.power_integral_j,
        "maximum_power_sample_gap_s": value.maximum_power_sample_gap_s,
        "energy_window_start_ns": value.energy_window_start_ns,
        "energy_window_end_ns": value.energy_window_end_ns,
        "host_cuda_record_start_ns": value.host_cuda_record_start_ns,
        "host_cuda_sync_end_ns": value.host_cuda_sync_end_ns,
        "power_samples": list(value.power_samples),
        "telemetry_start": dict(value.telemetry_start),
        "telemetry_end": dict(value.telemetry_end),
        "counter_reads": list(value.counter_reads),
        "logical_work_ids": ["expert-0/rows-1", "expert-0/rows-4"],
        "output_sha256": "a" * 64,
        "gpu_name": "NVIDIA GeForce RTX 5090",
        "gpu_uuid": "GPU-01234567-89ab-cdef-0123-456789abcdef",
    }


class ScheduleAndCalibrationTest(unittest.TestCase):
    def test_schedule_is_strict_abba_in_every_block(self) -> None:
        schedule = abba_schedule(2)
        self.assertEqual(tuple(row[2] for row in schedule[:4]), ABBA)
        self.assertEqual(tuple(row[2] for row in schedule[4:]), ABBA)
        self.assertEqual({row[0] for row in schedule}, {0, 1})
        with self.assertRaisesRegex(ProbeError, "positive"):
            abba_schedule(0)

    def test_calibration_freezes_one_repeat_denominator_for_both_arms(self) -> None:
        calls: list[tuple[str, int]] = []

        def measure(arm: str, repeats: int) -> float:
            calls.append((arm, repeats))
            return repeats * (0.10 if arm == ARM_A else 0.08)

        repeats, durations = choose_equal_repeats(
            measure, minimum_window_s=0.5, maximum_repeats=64
        )
        self.assertEqual(repeats, 8)
        self.assertGreaterEqual(min(durations.values()), 0.5)
        self.assertEqual(
            {repeat for _arm, repeat in calls},
            {1, 2, 4, 8},
        )
        for repeat in (1, 2, 4, 8):
            self.assertEqual([arm for arm, value in calls if value == repeat], [ARM_A, ARM_B])


class SameWindowTest(unittest.TestCase):
    def test_timer_and_meter_wrap_exactly_one_workload_execution(self) -> None:
        executions: list[str] = []
        timer_calls: list[str] = []
        meter_calls: list[str] = []

        def workload() -> object:
            executions.append("work")
            return {"output": 1}

        def timer(callback: object) -> TimedResult:
            timer_calls.append("timer")
            return TimedResult(10.0, callback())

        def meter(callback: object) -> EnergyEnvelope:
            meter_calls.append("meter")
            callback()
            return envelope()

        result = measure_same_window(workload, timer=timer, meter=meter)
        self.assertEqual(executions, ["work"])
        self.assertEqual(timer_calls, ["timer"])
        self.assertEqual(meter_calls, ["meter"])
        self.assertEqual(result.timed.payload, {"output": 1})
        self.assertEqual(result.energy.raw_board_energy_j, 25.0)

    def test_meter_that_skips_or_duplicates_work_fails_closed(self) -> None:
        timer = lambda callback: TimedResult(1.0, callback())

        with self.assertRaisesRegex(ProbeError, "did not invoke"):
            measure_same_window(
                lambda: 1,
                timer=timer,
                meter=lambda _callback: envelope(),
            )

        def duplicate(callback: object) -> EnergyEnvelope:
            callback()
            callback()
            return envelope()

        with self.assertRaisesRegex(ProbeError, "more than once"):
            measure_same_window(lambda: 1, timer=timer, meter=duplicate)


class RecordValidationTest(unittest.TestCase):
    def records(self, blocks: int = 2) -> list[dict[str, object]]:
        return [raw_record(block, position, arm) for block, position, arm in abba_schedule(blocks)]

    def test_complete_records_preserve_abba_equal_work_and_nonformal_status(self) -> None:
        validate_development_records(
            self.records(),
            blocks=2,
            expected_repeats=8,
            maximum_observed_gap_s=0.020,
            maximum_temperature_drift_c=3.0,
        )

    def test_formal_promotion_repeat_drift_and_output_drift_are_rejected(self) -> None:
        mutations = (
            (0, "formal_result", True, "formal"),
            (1, "repeats", 7, "repeat"),
            (2, "output_sha256", "b" * 64, "equal work/output"),
        )
        for index, key, value, message in mutations:
            with self.subTest(key=key):
                records = self.records()
                records[index][key] = value
                with self.assertRaisesRegex(ProbeError, message):
                    validate_development_records(
                        records,
                        blocks=2,
                        expected_repeats=8,
                        maximum_observed_gap_s=0.020,
                        maximum_temperature_drift_c=3.0,
                    )

    def test_missing_telemetry_and_unbracketed_cuda_window_are_rejected(self) -> None:
        records = self.records()
        records[0]["telemetry_start"] = {}
        with self.assertRaisesRegex(ProbeError, "telemetry"):
            validate_development_records(
                records,
                blocks=2,
                expected_repeats=8,
                maximum_observed_gap_s=0.020,
                maximum_temperature_drift_c=3.0,
            )

        records = self.records()
        records[0]["host_cuda_record_start_ns"] = 999_000_000
        with self.assertRaisesRegex(ProbeError, "bracket"):
            validate_development_records(
                records,
                blocks=2,
                expected_repeats=8,
                maximum_observed_gap_s=0.020,
                maximum_temperature_drift_c=3.0,
            )

    def test_telemetry_requires_thermal_clock_and_power_fields(self) -> None:
        incomplete = telemetry(1)
        del incomplete["sm_clock_mhz"]
        with self.assertRaisesRegex(ProbeError, "sm_clock_mhz"):
            validate_telemetry(incomplete)


class ManifestTest(unittest.TestCase):
    def test_manifest_is_hard_coded_nonformal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            (output / "raw").mkdir()
            (output / "raw" / "windows.jsonl").write_text("{}\n", encoding="utf-8")
            write_manifest(output, status="DEVELOPMENT_PROBE_COMPLETE", source_hashes={"probe.py": "a" * 64})
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertFalse(manifest["formal_result"])
            self.assertFalse(manifest["scientific_result_eligible"])
            self.assertEqual(manifest["verdict"], "DEVELOPMENT_MEASUREMENT_PATH_ONLY")
            self.assertIn("raw/windows.jsonl", manifest["artifact_sha256"])


if __name__ == "__main__":
    unittest.main()
