from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

try:
    from .prepare_clean_v2_data import (
        CLEAN_BUNDLE_ROOT,
        CleanDataError,
        add_self_hash,
        file_sha256,
        historical_exclusion_hashes,
        model_tree_sha256,
        prepare,
        reserve_sealed,
        select_requests,
        source_sha256,
        validate_clean_bundle_precondition,
        validate_fixed_output_ancestry,
        validate_historical_inventory,
        validate_phase4_signoff,
        validate_runtime_identity,
        DEFAULT_CONFIG,
        DEFAULT_PROTOCOL,
        REPO_ROOT,
    )
except ImportError:
    from prepare_clean_v2_data import (  # type: ignore
        CLEAN_BUNDLE_ROOT,
        CleanDataError,
        add_self_hash,
        file_sha256,
        historical_exclusion_hashes,
        model_tree_sha256,
        prepare,
        reserve_sealed,
        select_requests,
        source_sha256,
        validate_clean_bundle_precondition,
        validate_fixed_output_ancestry,
        validate_historical_inventory,
        validate_phase4_signoff,
        validate_runtime_identity,
        DEFAULT_CONFIG,
        DEFAULT_PROTOCOL,
        REPO_ROOT,
    )


class Tokenizer:
    def __call__(self, text: str, *, add_special_tokens: bool) -> dict[str, list[int]]:
        del add_special_tokens
        return {"input_ids": list(range(len(text.split())))}


class CleanV2DataTests(unittest.TestCase):
    def test_collision_is_hard_blocked_without_replacement(self) -> None:
        rows = (
            "a b c d e",
            "f g h i j",
            "k l m n o",
        )
        first = select_requests(
            rows,
            role="calibration",
            source_start=100,
            required=2,
            seed=7,
            min_tokens=5,
            tokenizers={"a": Tokenizer(), "b": Tokenizer()},
            excluded=set(),
        )
        with self.assertRaisesRegex(CleanDataError, "BLOCKED_DATA_SPLIT"):
            select_requests(
                rows,
                role="sealed",
                source_start=200,
                required=2,
                seed=7,
                min_tokens=5,
                tokenizers={"a": Tokenizer(), "b": Tokenizer()},
                excluded={str(first[0]["text_sha256"])},
            )

    def test_phase4_signoff_is_exact_and_self_hashed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve(strict=True)
            review = root / "review"
            review.mkdir()
            report = review / "RIC_Clean_v2_CodeReview.md"
            test_report = review / "RIC_Clean_v2_TestReport.json"
            source_manifest_path = review / "reviewed_source_manifest.json"
            producer = Path(validate_phase4_signoff.__code__.co_filename).resolve(strict=True)
            tests = Path(__file__).resolve(strict=True)
            source_manifest = add_self_hash(
                {
                    "schema_version": "ric-clean-v2-reviewed-source-manifest-v1",
                    "status": "REVIEWED",
                    "sources": {
                        "config": {
                            "path": DEFAULT_CONFIG.resolve(strict=True).relative_to(REPO_ROOT).as_posix(),
                            "file_sha256": file_sha256(DEFAULT_CONFIG),
                        },
                        "protocol": {
                            "path": DEFAULT_PROTOCOL.resolve(strict=True).relative_to(REPO_ROOT).as_posix(),
                            "file_sha256": file_sha256(DEFAULT_PROTOCOL),
                        },
                        "producer": {
                            "path": producer.relative_to(REPO_ROOT).as_posix(),
                            "file_sha256": file_sha256(producer),
                            "source_sha256": source_sha256(),
                        },
                        "tests": {
                            "path": tests.relative_to(REPO_ROOT).as_posix(),
                            "file_sha256": file_sha256(tests),
                        },
                    },
                }
            )
            source_manifest_path.write_text(json.dumps(source_manifest), encoding="utf-8")
            report.write_text(
                "STATUS: SIGNED-OFF\nOPEN_P0: 0\n"
                f"REVIEWED_SOURCE_MANIFEST_SHA256: {source_manifest['manifest_sha256']}\n",
                encoding="utf-8",
            )
            test_payload = add_self_hash(
                {
                    "schema_version": "ric-clean-v2-test-report-v1",
                    "status": "PASS",
                    "errors": 0,
                    "failures": 0,
                    "tests_run": 7,
                    "reviewed_source_manifest_sha256": source_manifest["manifest_sha256"],
                    "reviewed_source_manifest_file_sha256": file_sha256(source_manifest_path),
                }
            )
            test_report.write_text(json.dumps(test_payload), encoding="utf-8")
            path = review / "signoff_data_calibration.json"
            payload = add_self_hash(
                {
                    "schema_version": "ric-clean-v2-phase4-signoff-v1",
                    "status": "SIGNED-OFF",
                    "open_p0": 0,
                    "role": "calibration",
                    "config_sha256": file_sha256(DEFAULT_CONFIG),
                    "protocol_sha256": file_sha256(DEFAULT_PROTOCOL),
                    "prepare_clean_v2_data_source_sha256": source_sha256(),
                    "review_report_sha256": file_sha256(report),
                    "test_report_sha256": file_sha256(test_report),
                    "reviewed_source_manifest_sha256": source_manifest["manifest_sha256"],
                    "reviewed_source_manifest_file_sha256": file_sha256(source_manifest_path),
                },
                "signoff_sha256",
            )
            path.write_text(json.dumps(payload), encoding="utf-8")
            with patch.dict(
                validate_phase4_signoff.__globals__,
                {
                    "CLEAN_REVIEW_DIR": review,
                    "REVIEW_REPORT": report,
                    "TEST_REPORT": test_report,
                    "REVIEWED_SOURCE_MANIFEST": source_manifest_path,
                },
            ):
                verified = validate_phase4_signoff(path, role="calibration")
                self.assertEqual(
                    verified["signoff_sha256"], payload["signoff_sha256"]
                )
                tampered = dict(payload)
                tampered["open_p0"] = 1
                path.write_text(json.dumps(tampered), encoding="utf-8")
                with self.assertRaisesRegex(CleanDataError, "invalid signoff_sha256"):
                    validate_phase4_signoff(path, role="calibration")

    def test_phase4_rejects_unbound_pass_strings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve(strict=True)
            review = root / "review"
            review.mkdir()
            report = review / "RIC_Clean_v2_CodeReview.md"
            test_report = review / "RIC_Clean_v2_TestReport.json"
            source_manifest = review / "reviewed_source_manifest.json"
            signoff = review / "signoff_data_calibration.json"
            report.write_text("STATUS: SIGNED-OFF\nOPEN_P0: 0\n", encoding="utf-8")
            test_report.write_text('{"status":"PASS"}\n', encoding="utf-8")
            source_manifest.write_text('{"status":"REVIEWED"}\n', encoding="utf-8")
            signoff.write_text('{"status":"SIGNED-OFF"}\n', encoding="utf-8")
            with patch.dict(
                validate_phase4_signoff.__globals__,
                {
                    "CLEAN_REVIEW_DIR": review,
                    "REVIEW_REPORT": report,
                    "TEST_REPORT": test_report,
                    "REVIEWED_SOURCE_MANIFEST": source_manifest,
                },
            ), self.assertRaises(CleanDataError):
                validate_phase4_signoff(signoff, role="calibration")

    def test_historical_registry_reads_only_explicit_text_hash_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve(strict=True)
            wanted = "a" * 64
            (root / "evidence.json").write_text(
                json.dumps(
                    {
                        "selected_text_sha256": [wanted],
                        "unrelated_source_sha256": "b" * 64,
                    }
                ),
                encoding="utf-8",
            )
            hashes, sources = historical_exclusion_hashes((root,))
            self.assertEqual(hashes, {wanted})
            self.assertEqual(len(sources), 1)

    def test_historical_registry_rejects_symlink_root_or_child(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory).resolve(strict=True)
            real = parent / "real"
            real.mkdir()
            (real / "evidence.json").write_text("{}", encoding="utf-8")
            alias = parent / "alias"
            alias.symlink_to(real, target_is_directory=True)
            with self.assertRaisesRegex(CleanDataError, "root identity"):
                historical_exclusion_hashes((alias,))
            child = real / "linked.json"
            child.symlink_to(real / "evidence.json")
            with self.assertRaisesRegex(CleanDataError, "contains symlink"):
                historical_exclusion_hashes((real,))

    def test_historical_inventory_rejects_deletion_or_replacement(self) -> None:
        sources = [{"path": "/fixed/a.json", "file_sha256": "a" * 64}]
        expected = {
            "historical_exclusion_expected_json_file_count": 1,
            "historical_exclusion_expected_inventory_sha256": (
                "9fee14711a85835a466b6a57ce7c0e65a28dc83a268f3cfe89a4983abafff23d"
            ),
        }
        validate_historical_inventory(sources, expected)
        with self.assertRaisesRegex(CleanDataError, "inventory differs"):
            validate_historical_inventory([], expected)
        changed = [{"path": "/fixed/a.json", "file_sha256": "b" * 64}]
        with self.assertRaisesRegex(CleanDataError, "inventory differs"):
            validate_historical_inventory(changed, expected)

    def test_model_tree_hash_detects_replacement_and_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve(strict=True)
            weight = root / "weights.bin"
            weight.write_bytes(b"first")
            before = model_tree_sha256(root)
            weight.write_bytes(b"second")
            self.assertNotEqual(before, model_tree_sha256(root))
            link = root / "linked.bin"
            link.symlink_to(weight)
            with self.assertRaisesRegex(CleanDataError, "symlink"):
                model_tree_sha256(root)

    def test_prepare_requires_fixed_output_and_external_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve(strict=True)
            bad_output = root / "clean_v2/data/not-calibration"
            with patch.dict(prepare.__globals__, {"CLEAN_BUNDLE_ROOT": root}), self.assertRaisesRegex(
                CleanDataError, "fixed role path"
            ):
                prepare(
                    role="calibration",
                    output_dir=bad_output,
                    phase4_signoff_path=root / "unused.json",
                    cache_dir=root.parent / "external-cache",
                )
            fixed_output = root / "clean_v2/data/calibration"
            with patch.dict(prepare.__globals__, {"CLEAN_BUNDLE_ROOT": root}), self.assertRaisesRegex(
                CleanDataError, "cache must be outside"
            ):
                prepare(
                    role="calibration",
                    output_dir=fixed_output,
                    phase4_signoff_path=root / "unused.json",
                    cache_dir=root / "cache",
                )
            outside = root.parent / f"{root.name}-alias"
            outside.symlink_to(root, target_is_directory=True)
            try:
                with patch.dict(prepare.__globals__, {"CLEAN_BUNDLE_ROOT": root}), self.assertRaisesRegex(
                    CleanDataError, "cache must be outside"
                ):
                    prepare(
                        role="calibration",
                        output_dir=fixed_output,
                        phase4_signoff_path=root / "unused.json",
                        cache_dir=outside / "cache",
                    )
            finally:
                outside.unlink()

    def test_output_ancestor_symlink_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory).resolve(strict=True)
            root = parent / "bundle"
            target = parent / "redirected"
            root.mkdir()
            target.mkdir()
            (root / "clean_v2").symlink_to(target, target_is_directory=True)
            with patch.dict(
                validate_fixed_output_ancestry.__globals__,
                {"CLEAN_BUNDLE_ROOT": root},
            ), self.assertRaisesRegex(CleanDataError, "ancestor identity"):
                validate_fixed_output_ancestry(root / "clean_v2/data/calibration")

    def test_runtime_identity_rejects_executable_and_version_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve(strict=True)
            python = root / "python"
            python.write_bytes(b"executable identity fixture")
            identity = {
                "python_environment": str(python),
                "datasets_library_version": "4.5.0",
                "transformers_library_version": "4.53.3",
                "tokenizers_library_version": "0.21.4",
            }
            validate_runtime_identity(
                identity,
                datasets_version="4.5.0",
                transformers_version="4.53.3",
                tokenizers_version="0.21.4",
                executable=python,
            )
            with self.assertRaisesRegex(CleanDataError, "Python environment"):
                validate_runtime_identity(
                    identity,
                    datasets_version="4.5.0",
                    transformers_version="4.53.3",
                    tokenizers_version="0.21.4",
                    executable=root / "other-python",
                )
            with self.assertRaisesRegex(CleanDataError, "tokenizers"):
                validate_runtime_identity(
                    identity,
                    datasets_version="4.5.0",
                    transformers_version="4.53.3",
                    tokenizers_version="0.21.5",
                    executable=python,
                )

    def test_prepare_calibration_happy_path_writes_versioned_manifest(self) -> None:
        class FakeDataset(list):
            _fingerprint = "fixture-fingerprint"

        class FakeAutoTokenizer:
            @staticmethod
            def from_pretrained(path: Path, *, local_files_only: bool) -> Tokenizer:
                del path, local_files_only
                return Tokenizer()

        datasets_module = types.ModuleType("datasets")
        datasets_module.__version__ = "4.5.0"
        datasets_module.DownloadConfig = lambda **kwargs: kwargs
        datasets_module.load_dataset = lambda *args, **kwargs: FakeDataset(
            {"text": ("word " * 130) + str(index)} for index in range(4000)
        )
        transformers_module = types.ModuleType("transformers")
        transformers_module.__version__ = "4.53.3"
        transformers_module.AutoTokenizer = FakeAutoTokenizer
        tokenizers_module = types.ModuleType("tokenizers")
        tokenizers_module.__version__ = "0.21.4"
        config = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve(strict=True)
            signoff_path = root / "review/signoff_data_calibration.json"
            signoff_path.parent.mkdir()
            signoff_path.write_text("{}", encoding="utf-8")
            output = root / "clean_v2/data/calibration"
            signoff = {
                "signoff_sha256": "s" * 64,
                "config_sha256": file_sha256(DEFAULT_CONFIG),
                "protocol_sha256": file_sha256(DEFAULT_PROTOCOL),
                "prepare_clean_v2_data_source_sha256": source_sha256(),
            }

            def fake_model_tree(path: Path) -> str:
                return config["models"][path.name][
                    "expected_local_model_tree_manifest_sha256"
                ]

            globals_patch = {
                "CLEAN_BUNDLE_ROOT": root,
                "validate_phase4_signoff": lambda path, role: signoff,
                "validate_runtime_identity": lambda *args, **kwargs: None,
                "historical_exclusion_hashes": lambda roots: (set(), []),
                "validate_historical_inventory": lambda sources, data: None,
                "model_tree_sha256": fake_model_tree,
            }
            with patch.dict(
                sys.modules,
                {
                    "datasets": datasets_module,
                    "transformers": transformers_module,
                    "tokenizers": tokenizers_module,
                },
            ), patch.dict(prepare.__globals__, globals_patch):
                manifest = prepare(
                    role="calibration",
                    output_dir=output,
                    phase4_signoff_path=signoff_path,
                    cache_dir=root.parent / f"{root.name}-cache",
                )
            self.assertEqual(manifest["tokenizers_library_version"], "0.21.4")
            self.assertEqual(len(manifest["requests"]), 64)
            self.assertTrue((output / "manifest.json").is_file())

    def test_sealed_ledger_is_fixed_and_one_shot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve(strict=True)
            state = root / "state"
            ledger = state / "sealed_data_consumption.json"
            output = root / "data/sealed"
            signoff = {"signoff_sha256": "s" * 64}
            lock = {"manifest_sha256": "l" * 64}
            globals_patch = {
                "CLEAN_BUNDLE_ROOT": root,
                "CLEAN_STATE_DIR": state,
                "SEALED_LEDGER": ledger,
            }
            with patch.dict(reserve_sealed.__globals__, globals_patch):
                first = reserve_sealed(output_dir=output, signoff=signoff, n1_lock=lock)
                self.assertEqual(first["state"], "RESERVED_FAIL_CLOSED")
                with self.assertRaisesRegex(CleanDataError, "already consumed"):
                    reserve_sealed(output_dir=output, signoff=signoff, n1_lock=lock)

    def test_clean_bundle_rejects_preexisting_route_before_calibration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve(strict=True)
            route = root / "clean_v2/routes/olmoe"
            route.mkdir(parents=True)
            output = root / "clean_v2/data/calibration"
            with patch.dict(
                validate_clean_bundle_precondition.__globals__,
                {"CLEAN_BUNDLE_ROOT": root},
            ), self.assertRaisesRegex(CleanDataError, "precondition failed"):
                validate_clean_bundle_precondition(
                    role="calibration", output_dir=output
                )


if __name__ == "__main__":
    unittest.main()
