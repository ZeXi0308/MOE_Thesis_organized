from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from run_model_patch_parity_probe import (  # noqa: E402
    NativeRouterObserver,
    NativeRouterRecorder,
    PERMANENT_FORMAL_BLOCKERS,
    ProbeError,
    SCHEMA,
    Thresholds,
    PrefillObservation,
    _load_reused_components,
    _moe_layer_ids,
    _run_prefill,
    compare_observations,
    hash_snapshot_files,
    logit_metrics,
    require_exact_revision,
    route_metrics,
    validate_cache_growth,
    validate_token_ids,
    write_result_bundle,
)


class RevisionAndInputContractTest(unittest.TestCase):
    def test_only_exact_hugging_face_commit_is_accepted(self) -> None:
        revision = "a" * 40
        self.assertEqual(require_exact_revision(revision.upper()), revision)
        for value in ("main", "v1.0", "a" * 39, "g" * 40, "a" * 64):
            with self.subTest(value=value), self.assertRaisesRegex(ProbeError, "40-hex"):
                require_exact_revision(value)

    def test_token_ids_must_fit_model_vocab_and_decode_is_exactly_two_steps(self) -> None:
        validate_token_ids((1, 2, 3), (4, 5), vocab_size=8)
        with self.assertRaisesRegex(ProbeError, "exactly 2"):
            validate_token_ids((1,), (2,), vocab_size=8)
        with self.assertRaisesRegex(ProbeError, "outside"):
            validate_token_ids((1, 8), (2, 3), vocab_size=8)


class NumericalAndRouteContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - explicit environment gate
            raise unittest.SkipTest(f"PyTorch unavailable: {exc}")
        cls.torch = torch

    def route_batches(self, rows: int, *, changed: bool = False):
        torch = self.torch
        output = []
        for layer in (0, 2):
            selected = torch.tensor(
                [[(row + layer) % 4, (row + layer + 1) % 4] for row in range(rows)],
                dtype=torch.long,
            )
            if changed and layer == 2:
                selected[0, 0] = (selected[0, 0] + 2) % 4
            output.append(
                {
                    "sample_id": 0,
                    "layer": layer,
                    "selected_experts": selected,
                    "routing_weights": torch.full((rows, 2), 0.5, dtype=torch.float32),
                }
            )
        return tuple(output)

    def step(self, index: int, *, cache_length: int | None = None, changed: bool = False):
        torch = self.torch
        return SimpleNamespace(
            decode_step=index,
            token_id=5 + index,
            absolute_position=4 + index,
            cache_length=(5 + index if cache_length is None else cache_length),
            route_batches=self.route_batches(1, changed=changed),
            logits=torch.zeros((1, 1, 11), dtype=torch.float32),
        )

    def prefill(self, *, changed: bool = False) -> PrefillObservation:
        torch = self.torch
        return PrefillObservation(
            logits=torch.zeros((1, 4, 11), dtype=torch.float32),
            cache_length=4,
            route_batches=self.route_batches(4, changed=changed),
        )

    def test_logit_metrics_record_max_error_and_kl_and_reject_shape_drift(self) -> None:
        torch = self.torch
        native = torch.tensor([[[1.0, 0.0, -1.0]]])
        exact = logit_metrics(native, native.clone())
        self.assertEqual(exact["max_abs_logit_error"], 0.0)
        self.assertEqual(exact["max_kl_divergence_native_to_patched"], 0.0)
        changed = logit_metrics(native, torch.tensor([[[0.5, 0.5, -1.0]]]))
        self.assertEqual(changed["max_abs_logit_error"], 0.5)
        self.assertGreater(changed["max_kl_divergence_native_to_patched"], 0.0)
        with self.assertRaisesRegex(ProbeError, "shape mismatch"):
            logit_metrics(native, torch.zeros((1, 2, 3)))

    def test_route_metrics_require_layer_topk_rows_and_expert_identity_closure(self) -> None:
        exact = route_metrics(self.route_batches(4), self.route_batches(4), expected_rows=4)
        self.assertTrue(exact["layer_topk_signature_equal"])
        self.assertTrue(exact["expected_row_count_closed"])
        self.assertTrue(exact["selected_experts_equal"])
        changed = route_metrics(
            self.route_batches(4), self.route_batches(4, changed=True), expected_rows=4
        )
        self.assertFalse(changed["selected_experts_equal"])
        wrong_rows = route_metrics(self.route_batches(4), self.route_batches(4), expected_rows=1)
        self.assertFalse(wrong_rows["expected_row_count_closed"])

    def test_exact_prefill_and_two_step_cached_decode_pass(self) -> None:
        thresholds = Thresholds(0.0, 0.0, 0.0)
        result = compare_observations(
            self.prefill(),
            self.prefill(),
            (self.step(0), self.step(1)),
            (self.step(0), self.step(1)),
            prompt_length=4,
            forced_decode_ids=(5, 6),
            thresholds=thresholds,
        )
        self.assertTrue(result["parity_pass"])
        self.assertTrue(result["native_cache_growth"]["cache_advanced_by_one"])
        self.assertTrue(result["patched_cache_growth"]["cache_advanced_by_one"])
        self.assertTrue(result["layer_topk_closed_across_prefill_and_decode"])

    def test_cache_or_route_drift_fails_closed(self) -> None:
        thresholds = Thresholds(0.0, 0.0, 0.0)
        bad_cache = compare_observations(
            self.prefill(),
            self.prefill(),
            (self.step(0), self.step(1)),
            (self.step(0), self.step(1, cache_length=99)),
            prompt_length=4,
            forced_decode_ids=(5, 6),
            thresholds=thresholds,
        )
        self.assertFalse(bad_cache["parity_pass"])
        self.assertFalse(bad_cache["patched_cache_growth"]["cache_advanced_by_one"])

        bad_route = compare_observations(
            self.prefill(),
            self.prefill(changed=True),
            (self.step(0), self.step(1)),
            (self.step(0), self.step(1)),
            prompt_length=4,
            forced_decode_ids=(5, 6),
            thresholds=thresholds,
        )
        self.assertFalse(bad_route["parity_pass"])

    def test_cache_growth_requires_observable_plus_one(self) -> None:
        self.assertTrue(validate_cache_growth((5, 6), prompt_length=4)["cache_advanced_by_one"])
        self.assertFalse(validate_cache_growth((None, 6), prompt_length=4)["cache_advanced_by_one"])

    def test_tiny_olmoe_native_and_shared_full_patch_execute_same_cpu_protocol(self) -> None:
        torch = self.torch
        try:
            from transformers import OlmoeConfig, OlmoeForCausalLM
        except ImportError as exc:  # pragma: no cover - environment-specific
            self.skipTest(f"Olmoe test model unavailable: {exc}")

        torch.manual_seed(19)
        config = OlmoeConfig(
            vocab_size=31,
            hidden_size=16,
            intermediate_size=24,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            num_experts=4,
            num_experts_per_tok=2,
            max_position_embeddings=32,
            norm_topk_prob=False,
        )
        model = OlmoeForCausalLM(config).eval()
        prompt = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
        forced = torch.tensor([[5, 6]], dtype=torch.long)
        inputs = {"input_ids": prompt, "attention_mask": torch.ones_like(prompt)}
        (
            patch_mixtral_moe,
            cache_length_fn,
            clear_recorder_fn,
            (snapshot_fn, run_cached_decode_steps),
        ) = _load_reused_components()
        layer_ids = _moe_layer_ids(model)

        native_recorder = NativeRouterRecorder()
        native_recorder.set_sample_id(0)
        native_model = NativeRouterObserver(model, native_recorder, layer_ids)
        native_prefill = _run_prefill(
            native_model,
            native_recorder,
            inputs,
            cache_length_fn=cache_length_fn,
            clear_recorder_fn=clear_recorder_fn,
            snapshot_fn=snapshot_fn,
        )
        native_decode = run_cached_decode_steps(
            native_model,
            native_recorder,
            inputs,
            max_steps=2,
            eos_token_id=None,
            forced_decode_ids=forced,
            capture_logits=True,
        )

        patched_recorder = patch_mixtral_moe(
            model,
            "full",
            record_routes=True,
            record_diagnostics=False,
        )
        patched_recorder.set_sample_id(0)
        patched_prefill = _run_prefill(
            model,
            patched_recorder,
            inputs,
            cache_length_fn=cache_length_fn,
            clear_recorder_fn=clear_recorder_fn,
            snapshot_fn=snapshot_fn,
        )
        patched_decode = run_cached_decode_steps(
            model,
            patched_recorder,
            inputs,
            max_steps=2,
            eos_token_id=None,
            forced_decode_ids=forced,
            capture_logits=True,
        )
        result = compare_observations(
            native_prefill,
            patched_prefill,
            native_decode,
            patched_decode,
            prompt_length=4,
            forced_decode_ids=(5, 6),
            thresholds=Thresholds(1e-6, 1e-10, 1e-6),
        )
        self.assertTrue(result["parity_pass"], result)


class ArtifactContractTest(unittest.TestCase):
    def test_snapshot_hashes_every_file_and_requires_weights(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            snapshot = Path(tmp)
            (snapshot / "config.json").write_text('{"model_type":"unit"}\n', encoding="utf-8")
            (snapshot / "model.safetensors").write_bytes(b"unit-weights")
            result = hash_snapshot_files(snapshot)
            self.assertTrue(result["all_files_hashed"])
            self.assertEqual(result["weight_file_count"], 1)
            self.assertEqual(len(result["files"]), 2)
            self.assertEqual(len(result["aggregate_sha256"]), 64)

        with tempfile.TemporaryDirectory() as tmp:
            snapshot = Path(tmp)
            (snapshot / "config.json").write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ProbeError, "checkpoint weight"):
                hash_snapshot_files(snapshot)

    def test_result_bundle_is_hard_coded_nonformal_even_when_parity_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            result = write_result_bundle(
                output,
                {
                    "status": "DEVELOPMENT_PARITY_PASS",
                    "parity": {"parity_pass": True},
                    "formal_eligible": True,
                    "scientific_result_eligible": True,
                    "source_sha256": {"probe.py": "a" * 64},
                },
            )
            self.assertFalse(result["formal_eligible"])
            self.assertFalse(result["formal_result"])
            self.assertFalse(result["scientific_result_eligible"])
            self.assertEqual(tuple(result["formal_blockers"]), PERMANENT_FORMAL_BLOCKERS)
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["schema"], f"{SCHEMA}-manifest")
            self.assertFalse(manifest["formal_eligible"])
            self.assertEqual(len(manifest["result_sha256"]), 64)

    def test_execution_error_is_persisted_as_blocked_not_passed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            result = write_result_bundle(
                output,
                {"status": "DEVELOPMENT_PARITY_PASS", "parity": {"parity_pass": True}},
                error=ProbeError("unit failure"),
            )
            self.assertEqual(result["status"], "DEVELOPMENT_PARITY_BLOCKED")
            self.assertEqual(result["error"]["type"], "ProbeError")
            self.assertIn("probe execution failed closed", result["formal_blockers"][-1])


if __name__ == "__main__":
    unittest.main()
