from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from run_triage_audit_gpu import (
    RunnerError,
    _load_model_and_tokenizer,
    _protocol_path,
    build_period_plan,
    load_manifest,
)
from triage_manifest import text_sha256
from triage_policy import FEATURE_NAMES, fit_frozen_ridge


def feature_row(index: int) -> dict[str, float]:
    return {name: float(index + offset * 0.01) for offset, name in enumerate(FEATURE_NAMES)}


class GpuRunnerLogicTests(unittest.TestCase):
    def test_protocol_path_tracks_v2_and_v3_configs(self) -> None:
        self.assertEqual(
            _protocol_path(Path("config_v3.json")).name,
            "ConfidenceGuard_v3_冻结实验设计_2026-07-23.md",
        )
        self.assertEqual(
            _protocol_path(Path("config_v2.json")).name,
            "TriageAudit_Phase2_v2_冻结实验设计_2026-07-23.md",
        )

    def test_loader_uses_transformers_453_compatible_dtype_keyword(self) -> None:
        calls: dict[str, object] = {}

        class FakeTokenizerFactory:
            @staticmethod
            def from_pretrained(repo: str, **kwargs: object) -> object:
                calls["tokenizer"] = (repo, kwargs)
                return object()

        class FakeModel:
            training = True

            def eval(self) -> "FakeModel":
                self.training = False
                return self

            def to(self, device: str) -> "FakeModel":
                calls["device"] = device
                return self

            def parameters(self):
                yield SimpleNamespace(device=SimpleNamespace(type="cuda"))

        class FakeModelFactory:
            @staticmethod
            def from_pretrained(repo: str, **kwargs: object) -> FakeModel:
                calls["model"] = (repo, kwargs)
                return FakeModel()

        fake_transformers = ModuleType("transformers")
        fake_transformers.AutoModelForCausalLM = FakeModelFactory
        fake_transformers.AutoTokenizer = FakeTokenizerFactory
        model_config = {"repo_id": "org/model", "revision": "pinned-revision"}

        with patch.dict(sys.modules, {"transformers": fake_transformers}):
            model, _ = _load_model_and_tokenizer(model_config, offline=True)

        _, model_kwargs = calls["model"]
        self.assertEqual(model_kwargs["torch_dtype"], torch.bfloat16)
        self.assertNotIn("dtype", model_kwargs)
        self.assertEqual(model_kwargs["revision"], "pinned-revision")
        self.assertTrue(model_kwargs["local_files_only"])
        self.assertEqual(calls["device"], "cuda")
        self.assertFalse(model.training)

    def test_period_plan_exactly_matches_budget(self) -> None:
        documents = [{"text_sha256": f"{index + 1:064x}"} for index in range(12)]
        features = {str(row["text_sha256"]): feature_row(index) for index, row in enumerate(documents)}
        predictor = fit_frozen_ridge(list(features.values()), [0.01 + index * 0.001 for index in range(12)])
        plan = build_period_plan(documents, features, predictor, model_key="olmoe", split="sealed")
        self.assertEqual(sorted(plan["triage_2_4_8"].values()), sorted(plan["hash_budget_matched_2_4_8"].values()))

    def test_manifest_hash_and_split_are_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "manifest.jsonl"
            text = "alpha"
            row = {
                "schema_version": "triage-document-v2",
                "split": "calibration",
                "text_sha256": text_sha256(text),
                "text": text,
            }
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            self.assertEqual(len(load_manifest(path, expected_split="calibration", expected_count=1)), 1)
            with self.assertRaises(RunnerError):
                load_manifest(path, expected_split="sealed", expected_count=1)
            row["text_sha256"] = "0" * 64
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaises(RunnerError):
                load_manifest(path, expected_split="calibration", expected_count=1)


if __name__ == "__main__":
    unittest.main()
