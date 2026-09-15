from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).with_name("run_n0c_capture_stage_arm.py")
SPEC = importlib.util.spec_from_file_location("n0c_capture_arm", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FakeTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [ord(character) % 31 + 1 for character in text]


class FakeSamplingParams:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class N0cCaptureArmTest(unittest.TestCase):
    def test_sampling_contract_separates_device_capture_from_export(self) -> None:
        device = MODULE._sampling_params(FakeSamplingParams, 512, 16, False)
        exported = MODULE._sampling_params(FakeSamplingParams, 512, 16, True)
        self.assertNotIn("routed_experts_prompt_start", device.kwargs)
        self.assertEqual(exported.kwargs["routed_experts_prompt_start"], 511)
        for item in (device, exported):
            self.assertEqual(item.kwargs["temperature"], 0.0)
            self.assertEqual(item.kwargs["seed"], MODULE.SEED)
            self.assertEqual(item.kwargs["max_tokens"], 16)

    def test_warmup_shapes_and_prompt_builder_match_frozen_denominator(self) -> None:
        self.assertEqual(
            MODULE.WARMUP_SHAPES,
            ((128, 4), (128, 8), (128, 16), (512, 4), (512, 8), (512, 16)),
        )
        first = MODULE._build_prompts(FakeTokenizer(), ["alpha", "beta"], 32, 4)
        second = MODULE._build_prompts(FakeTokenizer(), ["alpha", "beta"], 32, 4)
        self.assertEqual(first, second)
        self.assertEqual([len(row) for row in first], [32] * 4)

    def test_prefix_loader_verifies_order_artifact_and_prompt_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "inputs").mkdir()
            rows = []
            for order, width in enumerate((2, 3)):
                prompts = [[order + request + token for token in range(8)] for request in range(width)]
                artifact = root / "inputs" / f"{order}.npz"
                np.savez_compressed(artifact, prompt_token_ids=np.asarray(prompts, dtype=np.int32))
                rows.append(
                    {
                        "execution_order": order,
                        "batch_size": width,
                        "prompt_length": 8,
                        "input_artifact": f"inputs/{order}.npz",
                        "input_artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                        "prompt_token_ids_sha256": MODULE._json_sha256(prompts),
                    }
                )
            spec = {
                "prefix_records": rows,
                "target_record": rows[-1],
                "prefix_plan_sha256": MODULE._json_sha256(rows),
            }
            loaded = MODULE._load_prefix(spec, root)
            self.assertEqual(len(loaded), 2)
            self.assertEqual(loaded[-1][1], np.load(root / "inputs/1.npz")["prompt_token_ids"].tolist())
            rows[1]["prompt_token_ids_sha256"] = "0" * 64
            spec["prefix_plan_sha256"] = MODULE._json_sha256(rows)
            with self.assertRaisesRegex(RuntimeError, "prompt-token SHA-256 mismatch"):
                MODULE._load_prefix(spec, root)

    def test_parser_has_no_model_or_workload_shape_overrides(self) -> None:
        parser = MODULE.build_parser()
        options = {option for action in parser._actions for option in action.option_strings}
        for forbidden in ("--model", "--revision", "--seed", "--output-tokens", "--batch-size"):
            self.assertNotIn(forbidden, options)

    def test_runtime_import_root_validation_is_exact_and_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            expected = Path(directory) / "runtime/stock"
            module_file = expected / "vllm/__init__.py"
            self.assertEqual(
                MODULE._verify_runtime_import_root(str(module_file), expected),
                expected.resolve(),
            )
            escaped = Path(directory) / "site-packages/vllm/__init__.py"
            with self.assertRaisesRegex(RuntimeError, "escaped expected runtime root"):
                MODULE._verify_runtime_import_root(str(escaped), expected)

    def test_logical_runtime_variant_is_bound_to_capture_mode(self) -> None:
        self.assertEqual(MODULE._expected_runtime_variant("stock", "off"), "stock")
        self.assertEqual(
            MODULE._expected_runtime_variant("valid-window", "full_export"),
            "valid-window",
        )
        self.assertEqual(
            MODULE._expected_runtime_variant("stock", "device"), "stock-device"
        )


if __name__ == "__main__":
    unittest.main()
