#!/usr/bin/env python3

import json
from pathlib import Path
import tempfile
import unittest

try:
    from . import capture_phasemap_lut_gpu as lut
except ImportError:
    import capture_phasemap_lut_gpu as lut


class PhaseMapLUTTests(unittest.TestCase):
    def fixture_rows(self):
        rows = []
        ordinal = 0
        source_sha = lut.file_sha256(Path(lut.__file__).resolve())
        protocol_sha = lut.file_sha256(lut.PROTOCOL)
        row_descriptor = {
            "shape": [1, 2048],
            "stride": [2048, 1],
            "dtype": "torch.bfloat16",
            "numel": 2048,
            "element_size_bytes": 2,
            "payload_bytes": 4096,
        }
        sibling_descriptor = {
            "shape": [1, 8, 2048],
            "stride": [8 * 2048, 2048, 1],
            "dtype": "torch.bfloat16",
            "numel": 16384,
            "element_size_bytes": 2,
            "payload_bytes": 32768,
        }
        hashes = {
            "row_tensor": "b" * 64,
            "siblings_tensor": "c" * 64,
            "row_descriptor": lut.object_sha256(row_descriptor),
            "siblings_descriptor": lut.object_sha256(sibling_descriptor),
        }
        for model, shape in lut.MODEL_SHAPES.items():
            for phase, count in (("warmup", lut.WARMUPS), ("measured", lut.MEASURED)):
                for trial in range(count):
                    for component in lut.COMPONENTS:
                        combine = component == "canonical_combine"
                        cuda_us = float(trial + 1)
                        rows.append(
                            {
                                "model_key": model,
                                "model_revision": shape["model_revision"],
                                "hidden": shape["hidden"],
                                "top_k": shape["top_k"],
                                "dtype": "torch.bfloat16",
                                "rows": 1,
                                "component": component,
                                "primitive_invocations": 1,
                                "phase": phase,
                                "trial_index": trial,
                                "execution_ordinal": ordinal,
                                "cuda_event_us": cuda_us,
                                "wall_time_us": cuda_us + 10.0,
                                "stream_id": 11 if model == "olmoe" else 12,
                                "source": lut.SOURCES[component],
                                "evidence_boundary": lut.CUDA_BOUNDARY,
                                "producer_source_sha256": source_sha,
                                "protocol_sha256": protocol_sha,
                                "input_tensor_sha256": hashes["siblings_tensor"] if combine else hashes["row_tensor"],
                                "input_descriptor_sha256": hashes["siblings_descriptor"] if combine else hashes["row_descriptor"],
                                "output_descriptor_sha256": hashes["row_descriptor"],
                            }
                        )
                        ordinal += 1
        return rows, hashes

    def fixture_artifact(self):
        rows, hashes = self.fixture_rows()
        producer_path = Path(lut.__file__).resolve()
        producer_sha = lut.file_sha256(producer_path)
        protocol_sha = lut.file_sha256(lut.PROTOCOL)
        model_inputs = {}
        for model, shape in lut.MODEL_SHAPES.items():
            row_descriptor = {
                "shape": [1, shape["hidden"]],
                "stride": [shape["hidden"], 1],
                "dtype": "torch.bfloat16",
                "numel": shape["hidden"],
                "element_size_bytes": 2,
                "payload_bytes": shape["hidden"] * 2,
            }
            sibling_descriptor = {
                "shape": [1, shape["top_k"], shape["hidden"]],
                "stride": [shape["top_k"] * shape["hidden"], shape["hidden"], 1],
                "dtype": "torch.bfloat16",
                "numel": shape["top_k"] * shape["hidden"],
                "element_size_bytes": 2,
                "payload_bytes": shape["top_k"] * shape["hidden"] * 2,
            }
            row_bits = [
                lut._float_to_bf16_bits(float(index % 31))
                for index in range(shape["hidden"])
            ]
            sibling_bits = [
                lut._float_to_bf16_bits(float(index % 37))
                for index in range(shape["top_k"] * shape["hidden"])
            ]
            model_inputs[model] = {
                **shape,
                "dtype": "torch.bfloat16",
                "rows": 1,
                "row_tensor_sha256": lut._tensor_sha256_from_bf16_bits(
                    row_bits, row_descriptor
                ),
                "siblings_tensor_sha256": lut._tensor_sha256_from_bf16_bits(
                    sibling_bits, sibling_descriptor
                ),
                "row_descriptor": row_descriptor,
                "row_descriptor_sha256": lut.object_sha256(row_descriptor),
                "siblings_descriptor": sibling_descriptor,
                "siblings_descriptor_sha256": lut.object_sha256(sibling_descriptor),
                "fixture_row_bits": row_bits,
                "fixture_sibling_bits": sibling_bits,
            }
        # Bind raw descriptors to each model's actual descriptor hashes.
        for row in rows:
            inputs = model_inputs[row["model_key"]]
            combine = row["component"] == "canonical_combine"
            row["input_tensor_sha256"] = inputs[
                "siblings_tensor_sha256" if combine else "row_tensor_sha256"
            ]
            row["input_descriptor_sha256"] = inputs["siblings_descriptor_sha256"] if combine else inputs["row_descriptor_sha256"]
            row["output_descriptor_sha256"] = inputs["row_descriptor_sha256"]

        correctness = []
        for model, inputs in model_inputs.items():
            row_bits = inputs.pop("fixture_row_bits")
            sibling_bits = inputs.pop("fixture_sibling_bits")
            for component in lut.COMPONENTS:
                combine = component == "canonical_combine"
                input_bits = sibling_bits if combine else row_bits
                reference = lut._reference_bits(
                    component,
                    input_bits,
                    lut.MODEL_SHAPES[model]["hidden"],
                    lut.MODEL_SHAPES[model]["top_k"],
                )
                correctness.append(
                    lut.build_correctness_record(
                        model=model,
                        component=component,
                        input_bits=input_bits,
                        observed_bits=reference,
                        input_descriptor=(
                            inputs["siblings_descriptor"]
                            if combine
                            else inputs["row_descriptor"]
                        ),
                        output_descriptor=inputs["row_descriptor"],
                        input_tensor_sha256=(
                            inputs["siblings_tensor_sha256"]
                            if combine
                            else inputs["row_tensor_sha256"]
                        ),
                    )
                )

        payload = {
            "schema_version": lut.SCHEMA_VERSION,
            "status": lut.STATUS,
            "scientific_result": False,
            "protocol_file": str(lut.PROTOCOL),
            "protocol_sha256": protocol_sha,
            "producer_source_sha256": producer_sha,
            "source_manifest": {
                "protocol": {"path": str(lut.PROTOCOL), "sha256": protocol_sha},
                "producer": {"path": str(producer_path), "sha256": producer_sha},
            },
            "warmups_per_point": lut.WARMUPS,
            "measured_trials_per_point": lut.MEASURED,
            "row_count": 1,
            "components": list(lut.COMPONENTS),
            "shared_cut": {
                "bandwidth_gbps": 200,
                "source": "ANALYTIC_NETWORK_L2_PROXY_NOT_RDMA",
                "payload_formula": "hidden*2",
                "descriptor_bytes": 16,
                "alignment_bytes": 16,
            },
            "model_inputs": model_inputs,
            "correctness_certificates": correctness,
            "environment": {
                "producer_pid": 4242,
                "python_executable": "/fixture/python",
                "python_version": "3.fixture",
                "pytorch_version": "fixture",
                "cuda_version": "fixture",
                "gpu_uuid": "GPU-fixture",
                "gpu_name": "NVIDIA GeForce RTX 5090",
                "driver_version": "fixture",
                "clock_sm_mhz": 1.0,
                "power_limit_w": 1.0,
                "temperature_c": 1.0,
                "compute_apps_before": [],
                "compute_apps_after": [
                    {
                        "pid": 4242,
                        "gpu_uuid": "GPU-fixture",
                        "process_name": "fixture-python",
                        "used_gpu_memory_mib": 128.0,
                    }
                ],
            },
            "raw_trials": rows,
            "summary": lut.validate_and_summarize(rows),
        }
        return lut.add_self_hash(payload)

    def test_full_row1_surface_and_20_plus_100_summary(self):
        rows, _hashes = self.fixture_rows()
        # fixture_artifact rewrites model-specific descriptor hashes; raw schema
        # itself is valid before the stronger artifact-level binding check.
        summary = lut.validate_and_summarize(rows)
        self.assertEqual(len(summary), len(lut.MODEL_SHAPES) * len(lut.COMPONENTS))
        target = next(
            item
            for item in summary
            if item["model_key"] == "olmoe" and item["component"] == "receiver_unpack"
        )
        self.assertEqual(target["rows"], 1)
        self.assertEqual(target["warmup_count"], 20)
        self.assertEqual(target["measured_count"], 100)
        self.assertEqual(target["median_cuda_event_us"], 50.5)
        self.assertEqual(target["p95_cuda_event_us"], 95.0)

    def test_artifact_recomputes_raw_summary_and_provenance(self):
        artifact = self.fixture_artifact()
        lut.validate_artifact(artifact)

    def test_rehashed_non_5090_environment_is_rejected(self):
        artifact = self.fixture_artifact()
        payload = {key: value for key, value in artifact.items() if key != "artifact_sha256"}
        payload["environment"]["gpu_name"] = "NOT_A_5090"
        with self.assertRaisesRegex(lut.PhaseMapLUTError, "5090"):
            lut.validate_artifact(lut.add_self_hash(payload))

    def test_rehashed_missing_or_foreign_compute_app_census_is_rejected(self):
        artifact = self.fixture_artifact()
        for mutation in ("missing", "foreign"):
            with self.subTest(mutation=mutation):
                payload = {
                    key: value for key, value in self.fixture_artifact().items()
                    if key != "artifact_sha256"
                }
                if mutation == "missing":
                    payload["environment"].pop("compute_apps_before")
                else:
                    payload["environment"]["compute_apps_after"][0]["pid"] = 9999
                with self.assertRaisesRegex(lut.PhaseMapLUTError, "environment|compute-app"):
                    lut.validate_artifact(lut.add_self_hash(payload))

    def test_missing_trial_is_rejected(self):
        rows, _hashes = self.fixture_rows()
        rows.pop()
        with self.assertRaisesRegex(lut.PhaseMapLUTError, "census"):
            lut.validate_and_summarize(rows)

    def test_protocol_drift_is_rejected_even_with_new_self_hash(self):
        artifact = self.fixture_artifact()
        payload = {key: value for key, value in artifact.items() if key != "artifact_sha256"}
        payload["protocol_sha256"] = "0" * 64
        with self.assertRaisesRegex(lut.PhaseMapLUTError, "provenance"):
            lut.validate_artifact(lut.add_self_hash(payload))

    def test_raw_source_drift_is_rejected_even_when_summary_is_rebuilt(self):
        artifact = self.fixture_artifact()
        payload = {key: value for key, value in artifact.items() if key != "artifact_sha256"}
        for row in payload["raw_trials"]:
            row["producer_source_sha256"] = "0" * 64
        payload["summary"] = lut.validate_and_summarize(payload["raw_trials"])
        with self.assertRaisesRegex(lut.PhaseMapLUTError, "provenance"):
            lut.validate_artifact(lut.add_self_hash(payload))

    def test_summary_tamper_is_rejected(self):
        artifact = self.fixture_artifact()
        payload = {key: value for key, value in artifact.items() if key != "artifact_sha256"}
        payload["summary"][0]["median_cuda_event_us"] += 1.0
        with self.assertRaisesRegex(lut.PhaseMapLUTError, "summary"):
            lut.validate_artifact(lut.add_self_hash(payload))

    def test_correctness_tamper_fails_even_with_rehashed_artifact(self):
        artifact = self.fixture_artifact()
        payload = {key: value for key, value in artifact.items() if key != "artifact_sha256"}
        payload["correctness_certificates"][0]["observed_bf16_bits"][0] ^= 1
        with self.assertRaisesRegex(lut.PhaseMapLUTError, "correctness"):
            lut.validate_artifact(lut.add_self_hash(payload))

    def test_wrong_combine_implementation_is_not_certified(self):
        artifact = self.fixture_artifact()
        payload = {key: value for key, value in artifact.items() if key != "artifact_sha256"}
        target = next(
            row
            for row in payload["correctness_certificates"]
            if row["component"] == "canonical_combine"
        )
        # A wrong implementation that returns only slot zero cannot obtain a
        # passing certificate against the frozen sequential top-k reference.
        hidden = lut.MODEL_SHAPES[target["model_key"]]["hidden"]
        target["observed_bf16_bits"] = target["input_bf16_bits"][:hidden]
        with self.assertRaisesRegex(lut.PhaseMapLUTError, "correctness"):
            lut.validate_artifact(lut.add_self_hash(payload))

    def test_atomic_self_hash_and_no_overwrite(self):
        artifact = self.fixture_artifact()
        with tempfile.TemporaryDirectory() as raw:
            output = Path(raw) / "phasemap_lut.json"
            lut.write_json_atomic_no_overwrite(output, artifact)
            materialized = json.loads(output.read_text(encoding="utf-8"))
            lut.validate_artifact(materialized)
            self.assertEqual(materialized, artifact)
            with self.assertRaisesRegex(lut.PhaseMapLUTError, "overwrite"):
                lut.write_json_atomic_no_overwrite(output, artifact)


if __name__ == "__main__":
    unittest.main()
