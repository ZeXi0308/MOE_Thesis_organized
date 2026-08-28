from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from capture_continuous_decode import (  # noqa: E402
    ContinuousRequest,
    _arrival_trace_sha256,
    _decode_stop_reason,
    _prepare_requests,
    load_preregistration,
    load_workload_manifest,
    run_continuous_decode,
    validate_formal_contract,
    validate_formal_workload_source,
    validate_output_isolation,
)
from core import ProtocolError  # noqa: E402


class ContinuousDecodeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            import torch
            from transformers import OlmoeConfig, OlmoeForCausalLM
        except ImportError as exc:  # pragma: no cover - explicit environment gate
            raise unittest.SkipTest(f"PyTorch/Transformers unavailable: {exc}")
        cls.torch = torch
        torch.manual_seed(11)
        config = OlmoeConfig(
            vocab_size=101,
            hidden_size=32,
            intermediate_size=48,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            num_experts=4,
            num_experts_per_tok=2,
            max_position_embeddings=64,
            norm_topk_prob=False,
        )
        cls.model = OlmoeForCausalLM(config).eval()

    def _request(self, request_id: str, sample_id: int, tokens: list[int], arrival: float):
        torch = self.torch
        input_ids = torch.tensor([tokens], dtype=torch.long)
        return ContinuousRequest(
            request_id=request_id,
            sample_id=sample_id,
            document_id=hashlib.sha256(request_id.encode("utf-8")).hexdigest(),
            arrival_us=arrival,
            deadline_us=arrival + 10_000.0,
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
        )

    def test_mutable_active_set_cache_and_serial_route_identity_close(self) -> None:
        requests = [
            self._request("r0", 0, [3, 5, 7], 0.0),
            self._request("r1", 1, [11, 13, 17, 19, 23], 0.0),
            self._request("r2", 2, [29, 31, 37, 41], 150.0),
        ]

        def duration(phase: str, _batch_size: int) -> float:
            return 10.0 if "prefill" in phase else 100.0

        capture = run_continuous_decode(
            self.model,
            requests,
            model_key="tiny-olmoe",
            max_decode_steps=3,
            max_batch_size=3,
            eos_token_id=None,
            serial_audit_request_ids=["r0", "r1", "r2"],
            duration_provider=duration,
        )
        batch_sizes = [row["batch_size"] for row in capture.batch_rows]
        self.assertIn(2, batch_sizes)
        self.assertIn(3, batch_sizes)
        self.assertIn(1, batch_sizes)
        self.assertEqual(capture.serial_audit["status"], "PASS")
        self.assertEqual(capture.serial_audit["requests"], 3)
        self.assertEqual(capture.serial_audit["steps"], 9)
        self.assertEqual(capture.serial_audit["token_match_fraction"], 1.0)
        self.assertEqual(capture.serial_audit["route_identity_match_fraction"], 1.0)
        self.assertEqual(
            capture.serial_audit["identity_summary"],
            {"contributions": 36, "tokens": 9, "requests": 3},
        )
        for row in capture.request_rows.values():
            self.assertEqual(len(row["steps"]), 3)
            self.assertEqual(row["stop_reason"], "max_decode_steps")
        for request in requests:
            token_ids = [int(value) for value in request.input_ids[0].tolist()]
            expected_hash = hashlib.sha256(
                json.dumps(token_ids, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            self.assertEqual(
                capture.request_rows[request.request_id]["prompt_token_ids_sha256"],
                expected_hash,
            )

    def test_manifest_rejects_prompt_hash_drift_and_unfrozen_audit_set(self) -> None:
        prompt = "frozen prompt"
        manifest = {
            "schema": "bcrd-continuous-workload-v1",
            "run_class": "development",
            "seed": 20260725,
            "expected_requests": 1,
            "max_prompt_tokens": 16,
            "model": {
                "id": "tiny",
                "key": "tiny",
                "revision": "rev",
                "tokenizer_revision": "tok-rev",
                "dtype": "float32",
            },
            "dataset": {"id": "fixture", "revision": "rev", "split": "test"},
            "generation": {"mode": "greedy", "do_sample": False, "max_decode_steps": 3},
            "scheduler": {"max_batch_size": 2},
            "software": {"execution_policy": "clean_committed_head"},
            "serial_audit_request_ids": ["r0"],
            "requests": [
                {
                    "request_id": "r0",
                    "sample_id": 0,
                    "document_id": "doc0",
                    "prompt": prompt,
                    "prompt_sha256": "0" * 64,
                    "arrival_us": 0.0,
                    "deadline_us": 1000.0,
                }
            ],
        }
        manifest["scheduler"]["arrival_trace_sha256"] = _arrival_trace_sha256(
            manifest["requests"]
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "manifest.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ProtocolError, "prompt SHA-256 mismatch"):
                load_workload_manifest(path)
            manifest["requests"][0]["prompt_sha256"] = hashlib.sha256(
                prompt.encode("utf-8")
            ).hexdigest()
            manifest["serial_audit_request_ids"] = []
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ProtocolError, "serial_audit_request_ids"):
                load_workload_manifest(path)

    def test_formal_contract_fails_while_preregistration_is_unauthorized(self) -> None:
        manifest = {"run_class": "formal"}
        preregistration = {
            "schema": "bcrd-gate0-continuous-decode-prereg-v1",
            "formal_execution_authorized": False,
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            canonical = Path(temp_dir) / "canonical.json"
            with self.assertRaisesRegex(ProtocolError, "not authorized"):
                validate_formal_contract(
                    manifest,
                    preregistration,
                    preregistration_sha256="a" * 64,
                    preregistration_path=canonical,
                    canonical_preregistration_path=canonical,
                    committed_preregistration_sha256="a" * 64,
                )

    def test_formal_contract_rejects_caller_selected_preregistration(self) -> None:
        manifest = {"run_class": "formal"}
        preregistration = {
            "schema": "bcrd-gate0-continuous-decode-prereg-v1",
            "formal_execution_authorized": True,
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with self.assertRaisesRegex(ProtocolError, "non-canonical"):
                validate_formal_contract(
                    manifest,
                    preregistration,
                    preregistration_sha256="a" * 64,
                    preregistration_path=root / "caller-selected.json",
                    canonical_preregistration_path=root / "canonical.json",
                    committed_preregistration_sha256="a" * 64,
                )

    def test_authorized_formal_contract_has_no_self_referential_hash(self) -> None:
        arrival_source = {
            "kind": "real_world_llm_serving_trace",
            "revision": "arrival-rev",
        }
        manifest = {
            "run_class": "formal",
            "seed": 11,
            "max_prompt_tokens": 16,
            "model": {
                "key": "model-a",
                "id": "org/model-a",
                "revision": "model-rev",
                "tokenizer_revision": "model-rev",
                "dtype": "bfloat16",
            },
            "dataset": {
                "id": "dataset",
                "config": "raw",
                "split": "test",
                "revision": "dataset-rev",
                "arrow_sha256": "a" * 64,
                "fingerprint": "fingerprint",
                "selection_rule": "first_row",
                "prompt_policy": "verbatim",
            },
            "generation": {"mode": "greedy", "max_decode_steps": 3},
            "scheduler": {
                "max_batch_size": 2,
                "arrival_trace_sha256": "b" * 64,
                "arrival_source": arrival_source,
            },
            "serial_audit_request_ids": ["r0"],
            "requests": [{"request_id": "r0"}],
        }
        preregistration = {
            "schema": "bcrd-gate0-continuous-decode-prereg-v1",
            "formal_execution_authorized": True,
            "formal_blockers": [],
            "seed": 11,
            "generation": {"mode": "greedy", "max_decode_steps": 3},
            "scheduler": {
                "max_batch_size": 2,
                "arrival_trace_sha256": "b" * 64,
                "arrival_source": arrival_source,
            },
            "dataset": {
                **manifest["dataset"],
                "documents_per_model": 1,
                "max_prompt_tokens": 16,
            },
            "models": [
                {
                    "key": "model-a",
                    "id": "org/model-a",
                    "revision": "model-rev",
                    "dtype": "bfloat16",
                }
            ],
            "acceptance": {"serial_audit_requests_per_cell": 1},
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            canonical = Path(temp_dir) / "canonical.json"
            validate_formal_contract(
                manifest,
                preregistration,
                preregistration_sha256="c" * 64,
                preregistration_path=canonical,
                canonical_preregistration_path=canonical,
                committed_preregistration_sha256="c" * 64,
            )

    def test_canonical_formal_manifests_and_token_ids_are_frozen(self) -> None:
        from transformers import AutoTokenizer

        workload_dir = HERE / "configs" / "workloads"
        expected_trace = None
        expected_documents = None
        for model_key in ("olmoe", "llmjp"):
            manifest = load_workload_manifest(
                workload_dir / f"{model_key}.formal.json"
            )
            self.assertEqual(len(manifest["requests"]), 128)
            trace = manifest["scheduler"]["arrival_trace_sha256"]
            documents = [row["document_id"] for row in manifest["requests"]]
            if expected_trace is None:
                expected_trace = trace
                expected_documents = documents
            else:
                self.assertEqual(trace, expected_trace)
                self.assertEqual(documents, expected_documents)
            model = manifest["model"]
            tokenizer = AutoTokenizer.from_pretrained(
                model["id"],
                revision=model["tokenizer_revision"],
                local_files_only=True,
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            requests = _prepare_requests(manifest, tokenizer, self.torch.device("cpu"))
            self.assertEqual(len(requests), 128)

    def test_canonical_formal_contracts_match_preregistration(self) -> None:
        preregistration_path = HERE / "configs" / "gate0_continuous_decode_v1.json"
        preregistration = load_preregistration(preregistration_path)
        preregistration_hash = hashlib.sha256(
            preregistration_path.read_bytes()
        ).hexdigest()
        for model_key in ("olmoe", "llmjp"):
            manifest = load_workload_manifest(
                HERE / "configs" / "workloads" / f"{model_key}.formal.json"
            )
            validate_formal_contract(
                manifest,
                preregistration,
                preregistration_sha256=preregistration_hash,
                preregistration_path=preregistration_path,
                canonical_preregistration_path=preregistration_path,
                committed_preregistration_sha256=preregistration_hash,
            )

    def test_output_isolation_rejects_cross_class_directories(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            formal_run = (
                root / "artifacts" / "bcrd_gate0" / "formal" / "run-1" / "olmoe"
            )
            validate_output_isolation(formal_run, root, "formal", "olmoe")
            with self.assertRaisesRegex(ProtocolError, "must be below"):
                validate_output_isolation(
                    root / "other" / "formal" / "run-1" / "olmoe",
                    root,
                    "formal",
                    "olmoe",
                )
            with self.assertRaisesRegex(ProtocolError, "model-key"):
                validate_output_isolation(
                    root / "artifacts" / "bcrd_gate0" / "formal" / "run-1" / "wrong",
                    root,
                    "formal",
                    "olmoe",
                )
            with self.assertRaisesRegex(ProtocolError, "model-key"):
                validate_output_isolation(
                    formal_run / "extra",
                    root,
                    "formal",
                    "extra",
                )
        validate_output_isolation(
            Path(tempfile.gettempdir()) / "bcrd-gate0-smoke-test",
            Path("/unused"),
            "development",
            "tiny",
        )
        with self.assertRaisesRegex(ProtocolError, "bcrd-gate0-smoke"):
            validate_output_isolation(
                Path(tempfile.gettempdir()) / "arbitrary-development-output",
                Path("/unused"),
                "development",
                "tiny",
            )

    def test_formal_workload_requires_preregistered_canonical_committed_bytes(self) -> None:
        manifest = {"run_class": "formal", "model": {"key": "olmoe"}}
        preregistration = {
            "formal_workloads": {
                "olmoe": {
                    "path": (
                        "docs/ideas/bcrd/experiments/configs/workloads/"
                        "olmoe.formal.json"
                    ),
                    "sha256": "a" * 64,
                }
            }
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            canonical = (
                root
                / "docs"
                / "ideas"
                / "bcrd"
                / "experiments"
                / "configs"
                / "workloads"
                / "olmoe.formal.json"
            )
            validate_formal_workload_source(
                manifest,
                preregistration,
                repo_root=root,
                workload_manifest_path=canonical,
                workload_manifest_sha256="a" * 64,
                committed_workload_manifest_sha256="a" * 64,
            )
            with self.assertRaisesRegex(ProtocolError, "non-canonical workload"):
                validate_formal_workload_source(
                    manifest,
                    preregistration,
                    repo_root=root,
                    workload_manifest_path=root / "caller-selected.json",
                    workload_manifest_sha256="a" * 64,
                    committed_workload_manifest_sha256="a" * 64,
                )
            with self.assertRaisesRegex(ProtocolError, "preregistered SHA-256"):
                validate_formal_workload_source(
                    manifest,
                    preregistration,
                    repo_root=root,
                    workload_manifest_path=canonical,
                    workload_manifest_sha256="b" * 64,
                    committed_workload_manifest_sha256="a" * 64,
                )
            with self.assertRaisesRegex(ProtocolError, "executing git commit"):
                validate_formal_workload_source(
                    manifest,
                    preregistration,
                    repo_root=root,
                    workload_manifest_path=canonical,
                    workload_manifest_sha256="a" * 64,
                    committed_workload_manifest_sha256="b" * 64,
                )

    def test_eos_wins_when_it_occurs_on_the_final_allowed_step(self) -> None:
        self.assertEqual(
            _decode_stop_reason(
                predicted_token_id=7,
                eos_token_id=7,
                completed_decode_steps=3,
                max_decode_steps=3,
            ),
            "eos",
        )
        self.assertEqual(
            _decode_stop_reason(
                predicted_token_id=8,
                eos_token_id=7,
                completed_decode_steps=3,
                max_decode_steps=3,
            ),
            "max_decode_steps",
        )


if __name__ == "__main__":
    unittest.main()
