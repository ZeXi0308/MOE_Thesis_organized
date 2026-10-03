#!/usr/bin/env python3

import json
from pathlib import Path
import tempfile
import unittest

try:
    from . import capture_fjrc_lut_gpu as lut
except ImportError:
    import capture_fjrc_lut_gpu as lut


class FJRCLUTTests(unittest.TestCase):
    def test_frozen_protocol_is_present(self):
        self.assertTrue(lut.PROTOCOL.is_file())
        self.assertEqual(
            lut.file_sha256(lut.PROTOCOL),
            "51dbf1d01e8220d3945b0d88009f37f962624837793342541e2500327983ca56",
        )

    def fixture(self):
        rows = []
        ordinal = 0
        source_sha = "a" * 64
        for model, shape in lut.MODEL_SHAPES.items():
            points = [("sender_pack", None), ("canonical_combine", None)] + [("receiver_unpack", depth) for depth in lut.UNPACK_DEPTHS]
            for phase, count in (("warmup", lut.WARMUPS), ("measured", lut.MEASURED)):
                for trial in range(count):
                    for component, depth in points:
                        if component == "receiver_unpack":
                            completions = [float(index + 1 + trial) for index in range(int(depth))]
                            cuda_us = completions[-1] if completions else 0.5
                        else:
                            completions = []
                            cuda_us = float(trial + 1)
                        rows.append(
                            {
                                "model_key": model, "model_revision": shape["model_revision"],
                                "hidden": shape["hidden"], "top_k": shape["top_k"], "dtype": "torch.bfloat16",
                                "component": component, "queue_depth": depth,
                                "primitive_invocations": int(depth) if component == "receiver_unpack" else 1,
                                "candidate_included": False if component == "receiver_unpack" else None,
                                "phase": phase, "trial_index": trial, "execution_ordinal": ordinal,
                                "cuda_event_us": cuda_us, "wall_time_us": cuda_us + 10.0,
                                "invocation_completion_us": completions, "stream_id": 11,
                                "source": lut.SOURCES[component], "evidence_boundary": lut.CUDA_BOUNDARY,
                                "producer_source_sha256": source_sha,
                                "input_tensor_sha256": "b" * 64,
                                "input_descriptor_sha256": "c" * 64,
                                "output_descriptor_sha256": "d" * 64,
                            }
                        )
                        ordinal += 1
        for phase, count in (("warmup", lut.WARMUPS), ("measured", lut.MEASURED)):
            for trial in range(count):
                rows.append(
                    {
                        "model_key": None, "component": "host_lookup_tax", "phase": phase,
                        "trial_index": trial, "execution_ordinal": ordinal,
                        "lookup_repeats": lut.HOST_LOOKUP_REPEATS, "lookups_per_joint_decision": 6,
                        "lookup_tax_us_per_joint_decision": float(trial + 1),
                        "enters_zero_tax_r0": False,
                        "source": lut.SOURCES["host_lookup_tax"], "evidence_boundary": lut.HOST_BOUNDARY,
                        "producer_source_sha256": source_sha,
                    }
                )
                ordinal += 1
        return rows

    def test_full_surface_and_completion_summary(self):
        summary = lut.validate_and_summarize(self.fixture())
        self.assertEqual(len(summary), len(lut.MODEL_SHAPES) * (2 + len(lut.UNPACK_DEPTHS)) + 1)
        row = next(item for item in summary if item.get("model_key") == "olmoe" and item["component"] == "receiver_unpack" and item["queue_depth"] == 2)
        self.assertEqual(row["measured_count"], 100)
        self.assertEqual(row["median_invocation_completion_us"], [50.5, 51.5])
        self.assertEqual(row["backlog_only_queue_work_us"], 51.5)
        zero = next(item for item in summary if item.get("model_key") == "olmoe" and item["component"] == "receiver_unpack" and item["queue_depth"] == 0)
        self.assertEqual(zero["backlog_only_queue_work_us"], 0.0)
        host = next(item for item in summary if item["component"] == "host_lookup_tax")
        self.assertFalse(host["enters_zero_tax_r0"])

    def test_missing_completion_timestamp_is_rejected(self):
        rows = self.fixture()
        row = next(item for item in rows if item["component"] == "receiver_unpack" and item["queue_depth"] == 2)
        row["invocation_completion_us"].pop()
        with self.assertRaisesRegex(lut.FJRCLUTError, "completion timestamp"):
            lut.validate_and_summarize(rows)

    def test_depth_above_16_is_rejected(self):
        rows = self.fixture()
        row = next(item for item in rows if item["component"] == "receiver_unpack" and item["queue_depth"] == 16)
        row["queue_depth"] = 17
        row["primitive_invocations"] = 17
        with self.assertRaisesRegex(lut.FJRCLUTError, "unpack completion"):
            lut.validate_and_summarize(rows)

    def test_duplicate_identity_is_rejected(self):
        rows = self.fixture()
        rows[-1] = dict(rows[-2], execution_ordinal=rows[-1]["execution_ordinal"])
        with self.assertRaisesRegex(lut.FJRCLUTError, "duplicate"):
            lut.validate_and_summarize(rows)

    def test_atomic_self_hash_and_no_overwrite(self):
        payload = lut.add_self_hash({"schema_version": "fixture", "scientific_result": False})
        with tempfile.TemporaryDirectory() as raw:
            output = Path(raw) / "fjrc.json"
            lut.write_json_atomic_no_overwrite(output, payload)
            materialized = json.loads(output.read_text(encoding="utf-8"))
            lut.validate_self_hash(materialized)
            self.assertEqual(materialized, payload)
            with self.assertRaisesRegex(lut.FJRCLUTError, "overwrite"):
                lut.write_json_atomic_no_overwrite(output, payload)


if __name__ == "__main__":
    unittest.main()
