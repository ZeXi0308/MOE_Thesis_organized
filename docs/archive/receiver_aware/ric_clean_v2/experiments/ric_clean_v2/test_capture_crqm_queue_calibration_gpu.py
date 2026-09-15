#!/usr/bin/env python3

import json
from pathlib import Path
import tempfile
import unittest

try:
    from . import capture_crqm_queue_calibration_gpu as calibration
except ImportError:
    import capture_crqm_queue_calibration_gpu as calibration


class CRQMQueueCalibrationTests(unittest.TestCase):
    def raw_fixture(self):
        rows = []
        ordinal = 0
        producer = "a" * 64
        for model_key, shape in calibration.MODEL_SHAPES.items():
            for phase, count in (("warmup", calibration.WARMUPS), ("measured", calibration.MEASURED)):
                for trial_index in range(count):
                    for depth in calibration.DEPTHS:
                        cuda_us = 0.0 if depth == 0 else float(depth + trial_index + 1)
                        rows.append(
                            {
                                "model_key": model_key,
                                "model_revision": shape["model_revision"],
                                "hidden": shape["hidden"],
                                "top_k": shape["top_k"],
                                "dtype": "torch.bfloat16",
                                "queue_depth": depth,
                                "primitive_invocations": depth,
                                "candidate_included": False,
                                "measurement_semantics": calibration.MEASUREMENT_SEMANTICS,
                                "phase": phase,
                                "trial_index": trial_index,
                                "execution_ordinal": ordinal,
                                "cuda_event_us": cuda_us,
                                "wall_time_us": cuda_us + 10.0,
                                "stream_id": 7,
                                "source": calibration.SOURCE,
                                "evidence_boundary": calibration.EVIDENCE_BOUNDARY,
                                "producer_source_sha256": producer,
                                "packed_tensor_sha256": "b" * 64,
                            }
                        )
                        ordinal += 1
        return rows

    def test_complete_surface_summarizes_measured_only(self):
        summary = calibration.validate_and_summarize(self.raw_fixture())
        self.assertEqual(len(summary), len(calibration.MODEL_SHAPES) * len(calibration.DEPTHS))
        row = next(item for item in summary if item["model_key"] == "olmoe" and item["queue_depth"] == 2)
        self.assertEqual(row["warmup_count"], 20)
        self.assertEqual(row["measured_count"], 100)
        self.assertEqual(row["median_cuda_event_us"], 52.5)
        self.assertEqual(row["p95_cuda_event_us"], 97.0)
        self.assertEqual(row["max_cuda_event_us"], 102.0)
        self.assertEqual(row["backlog_only_queue_work_us"], 52.5)
        depth_zero = next(item for item in summary if item["model_key"] == "olmoe" and item["queue_depth"] == 0)
        self.assertEqual(depth_zero["backlog_only_queue_work_us"], 0.0)
        self.assertFalse(depth_zero["candidate_included"])

    def test_duplicate_trial_is_rejected(self):
        rows = self.raw_fixture()
        rows[-1] = dict(rows[-2], execution_ordinal=rows[-1]["execution_ordinal"])
        with self.assertRaisesRegex(calibration.CRQMCalibrationError, "duplicate"):
            calibration.validate_and_summarize(rows)

    def test_nonzero_depth_must_have_positive_cuda_time(self):
        rows = self.raw_fixture()
        target = next(row for row in rows if row["queue_depth"] == 2)
        target["cuda_event_us"] = 0.0
        with self.assertRaisesRegex(calibration.CRQMCalibrationError, "frozen schema"):
            calibration.validate_and_summarize(rows)

    def test_atomic_writer_is_self_hashed_and_no_overwrite(self):
        payload = calibration.add_self_hash({"schema_version": "fixture", "scientific_result": False})
        with tempfile.TemporaryDirectory() as raw:
            output = Path(raw) / "calibration.json"
            calibration.write_json_atomic_no_overwrite(output, payload)
            materialized = json.loads(output.read_text(encoding="utf-8"))
            calibration.validate_self_hash(materialized)
            self.assertEqual(materialized, payload)
            with self.assertRaisesRegex(calibration.CRQMCalibrationError, "overwrite"):
                calibration.write_json_atomic_no_overwrite(output, payload)

    def test_tampered_self_hash_is_rejected_before_write(self):
        payload = calibration.add_self_hash({"schema_version": "fixture"})
        payload["schema_version"] = "tampered"
        with tempfile.TemporaryDirectory() as raw:
            with self.assertRaisesRegex(calibration.CRQMCalibrationError, "self hash"):
                calibration.write_json_atomic_no_overwrite(Path(raw) / "x.json", payload)


if __name__ == "__main__":
    unittest.main()
