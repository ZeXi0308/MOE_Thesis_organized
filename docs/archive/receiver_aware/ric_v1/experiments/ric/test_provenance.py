from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import formal_provenance as provenance
import prepare_data
import run_experiment


REPO_ROOT = next(candidate for candidate in HERE.parents if (candidate / "experiments/shared").is_dir())


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _tiny_config() -> dict[str, object]:
    return {
        "data": {
            "config": "wikitext-103-raw-v1",
            "split": "train",
            "selection_seed": 7,
            "selection_method": "sha256(selection_seed||canonical_text_sha256)",
            "formal_dataset_identity": {
                "producer": "prepare_data.py",
                "python_environment": ".venv/bin/python",
                "datasets_library_version": "9.9.9",
                "dataset_repo_id": "wikitext",
                "dataset_revision": "c" * 40,
                "calibration": {
                    "dataset_slice_fingerprint": "f" * 16,
                    "dataset_slice_canonical_content_sha256": "5" * 64,
                },
                "sealed": {
                    "dataset_slice_fingerprint": "e" * 16,
                    "dataset_slice_canonical_content_sha256": "6" * 64,
                },
            },
            "sequence_length": 128,
            "batch_size": 1,
            "padding_allowed": False,
            "min_tokens_both_frozen_tokenizers": 2,
            "calibration": {
                "candidate_row_start_inclusive": 10,
                "candidate_row_end_exclusive": 14,
                "document_count": 2,
            },
            "sealed": {
                "candidate_row_start_inclusive": 20,
                "candidate_row_end_exclusive": 24,
                "document_count": 2,
            },
        },
        "models": {
            "olmoe": {"repo_id": "repo/olmoe", "revision": "a" * 40},
            "llmjp": {"repo_id": "repo/llmjp", "revision": "b" * 40},
        },
        "go_no_go": {
            "required_models": ["olmoe", "llmjp"],
            "required_main_cells": ["poisson_rho60", "ctmc_mmpp_rho85"],
        },
    }


def _valid_calibration_lock(
    *, config: dict[str, object], protocol_sha: str, config_sha: str
) -> dict[str, object]:
    models = tuple(config["go_no_go"]["required_models"])
    cells = tuple(config["go_no_go"]["required_main_cells"])
    payload = {
        "schema_version": provenance.CALIBRATION_LOCK_SCHEMA,
        "status": "CALIBRATION_LOCKED",
        "scientific_result": False,
        "mode": "formal",
        "role": "calibration",
        "config_sha256": config_sha,
        "protocol_sha256": protocol_sha,
        "run_experiment_source_sha256": (
            prepare_data._run_experiment_source_sha256()
        ),
        "scenario_tree_sha256": {model: "1" * 64 for model in models},
        "service_lut_metadata_sha256": {model: "2" * 64 for model in models},
        "capability_probe_sha256": {model: "3" * 64 for model in models},
        "scenario_producer_signoff_sha256": {
            model: "5" * 64 for model in models
        },
        "capability_producer_signoff_sha256": {
            model: "6" * 64 for model in models
        },
        "signoff_sha256": "7" * 64,
        "g1_by_model": {model: True for model in models},
        "g1_pass": True,
        "models": {
            model: {
                "cells": {
                    cell: {"closure_budget_us": 100.0} for cell in cells
                }
            }
            for model in models
        },
        "policy_semantics_sha256": "4" * 64,
    }
    return provenance.add_self_hash(payload)


def _valid_manifest(
    *, mode: str = "formal", role: str = "calibration"
) -> tuple[dict[str, object], dict[str, object], str, str, str, str]:
    config = _tiny_config()
    protocol_sha = "1" * 64
    config_sha = "2" * 64
    source_sha = "3" * 64
    registry_sha = "4" * 64
    role_cfg = config["data"][role]
    identity = config["data"]["formal_dataset_identity"]
    role_identity = identity[role]
    rows = []
    for index, text in enumerate(("alpha", "beta")):
        text_hash = provenance.sha256_bytes(text.encode("utf-8"))
        rows.append(
            {
                "request_id": f"ric:{role}:{index:04d}:{text_hash[:12]}",
                "text": text,
                "text_sha256": text_hash,
                "source_row": role_cfg["candidate_row_start_inclusive"] + index,
                "rank_sha256": provenance.frozen_concat_sha256(7, text_hash),
                "token_lengths": {"olmoe": 3, "llmjp": 3},
            }
        )
    rows.sort(key=lambda row: (row["rank_sha256"], row["source_row"]))
    for index, row in enumerate(rows):
        row["request_id"] = f"ric:{role}:{index:04d}:{row['text_sha256'][:12]}"
    payload = {
        "schema_version": provenance.DATA_MANIFEST_SCHEMA,
        "status": "INPUT_ONLY" if mode == "formal" else "NOT_TESTED",
        "scientific_result": False,
        "mode": mode,
        "role": role,
        "dataset_loader": "wikitext",
        "dataset_config": config["data"]["config"],
        "dataset_split": config["data"]["split"],
        "data_preparation_producer": identity["producer"],
        "data_preparation_python_environment": identity["python_environment"],
        "dataset_repo_id": identity["dataset_repo_id"],
        "dataset_revision": identity["dataset_revision"],
        "dataset_source_urls_sha256": "9" * 64,
        "datasets_library_version": identity["datasets_library_version"],
        "dataset_slice_fingerprint": role_identity[
            "dataset_slice_fingerprint"
        ],
        "dataset_slice_row_count": 4,
        "dataset_slice_canonical_content_sha256": role_identity[
            "dataset_slice_canonical_content_sha256"
        ],
        "candidate_window": [
            role_cfg["candidate_row_start_inclusive"],
            role_cfg["candidate_row_end_exclusive"],
        ],
        "selection_seed": 7,
        "selection_method": config["data"]["selection_method"],
        "sequence_tokens": 128,
        "batch_size": 1,
        "padding_allowed": False,
        "model_revisions": {
            key: f"{spec['repo_id']}@{spec['revision']}"
            for key, spec in config["models"].items()
        },
        "tokenizer_revisions": {
            key: f"{spec['repo_id']}@{spec['revision']}"
            for key, spec in config["models"].items()
        },
        "historical_exclusion_registry_sha256": registry_sha,
        "protocol_sha256": protocol_sha,
        "config_sha256": config_sha,
        "prepare_data_source_sha256": source_sha,
        "signoff_sha256": "d" * 64 if mode == "formal" else None,
        "selected_text_sha256": [row["text_sha256"] for row in rows],
        "requests": rows,
    }
    if mode == "formal" and role == "sealed":
        payload["sealed_reservation_sha256"] = "6" * 64
        payload["sealed_nonce_sha256"] = "7" * 64
        payload["calibration_manifest_self_hash"] = "8" * 64
        payload["calibration_manifest_file_sha256"] = "9" * 64
        payload["calibration_selected_list_sha256"] = "a" * 64
        payload["calibration_lock_self_hash"] = "b" * 64
        payload["calibration_lock_file_sha256"] = "c" * 64
    return (
        provenance.add_self_hash(payload),
        config,
        protocol_sha,
        config_sha,
        source_sha,
        registry_sha,
    )


class FrozenDataManifestTest(unittest.TestCase):
    def test_formal_self_hash_roundtrip_and_collision_rejection(self) -> None:
        value = provenance.add_self_hash(
            {"schema_version": "x", "keys": {7: "seven", 14: "fourteen"}}
        )
        self.assertEqual(value["keys"], {"7": "seven", "14": "fourteen"})
        reloaded = json.loads(json.dumps(value))
        self.assertEqual(
            provenance.validate_self_hash(reloaded), value["manifest_sha256"]
        )
        with self.assertRaisesRegex(
            provenance.FormalProvenanceError, "duplicate JSON object key"
        ):
            provenance.add_self_hash(
                {"schema_version": "x", "bad": {1: "int", "1": "str"}}
            )
        with self.assertRaisesRegex(provenance.FormalProvenanceError, "strict JSON"):
            provenance.add_self_hash({"schema_version": "x", "bad": float("nan")})

    def test_calibration_source_digest_matches_actual_producer(self) -> None:
        self.assertEqual(
            prepare_data._run_experiment_source_sha256(),
            run_experiment._source_sha256(),
        )
        self.assertEqual(
            tuple(prepare_data.RUN_EXPERIMENT_SOURCE_PATHS),
            tuple(run_experiment.RUN_EXPERIMENT_SOURCE_PATHS),
        )

    def test_strict_json_loader_rejects_nested_duplicate_keys(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate.json"
            path.write_text('{"outer":{"key":1,"key":2}}\n', encoding="utf-8")
            with self.assertRaisesRegex(
                provenance.FormalProvenanceError, "duplicate JSON key"
            ):
                provenance.load_json_mapping_strict(path, label="fixture")
            with self.assertRaisesRegex(
                prepare_data.DataPreparationError, "duplicate JSON key"
            ):
                prepare_data._load_config(path)

    def test_strict_json_loader_rejects_nonfinite_extensions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nonfinite.json"
            for token in ("NaN", "Infinity", "-Infinity"):
                path.write_text(f'{{"value":{token}}}\n', encoding="utf-8")
                with self.assertRaisesRegex(
                    provenance.FormalProvenanceError, "non-finite JSON constant"
                ):
                    provenance.load_json_mapping_strict(path, label="fixture")

    def test_exact_concat_golden_vector(self) -> None:
        text_hash = provenance.sha256_bytes(b"text-0")
        self.assertEqual(
            provenance.frozen_concat_sha256(7, text_hash),
            "5b991e2c17218193a4913fbfdb42adafb62f17ae540cc544fe0b6ff916bd050b",
        )
        self.assertNotEqual(
            provenance.frozen_concat_sha256(7, text_hash),
            provenance.sha256_bytes(f"7:{text_hash}".encode("utf-8")),
        )

    def test_strict_manifest_accepts_exact_formal_and_binds_current_source(self) -> None:
        manifest, config, protocol_sha, config_sha, source_sha, registry_sha = (
            _valid_manifest()
        )
        provenance.validate_data_manifest_fields(
            manifest,
            mode="formal",
            role="calibration",
            config=config,
            protocol_sha256=protocol_sha,
            config_sha256=config_sha,
            expected_prepare_data_source_sha256=source_sha,
            expected_historical_registry_sha256=registry_sha,
        )
        with self.assertRaisesRegex(
            provenance.FormalProvenanceError, "prepare_data_source_sha256"
        ):
            provenance.validate_data_manifest_fields(
                manifest,
                mode="formal",
                role="calibration",
                config=config,
                protocol_sha256=protocol_sha,
                config_sha256=config_sha,
                expected_prepare_data_source_sha256="8" * 64,
                expected_historical_registry_sha256=registry_sha,
            )

    def test_formal_rejects_dev_manifest(self) -> None:
        manifest, config, protocol_sha, config_sha, source_sha, registry_sha = (
            _valid_manifest(mode="dev")
        )
        with self.assertRaisesRegex(
            provenance.FormalProvenanceError, "status|mode"
        ):
            provenance.validate_data_manifest_fields(
                manifest,
                mode="formal",
                role="calibration",
                config=config,
                protocol_sha256=protocol_sha,
                config_sha256=config_sha,
                expected_prepare_data_source_sha256=source_sha,
                expected_historical_registry_sha256=registry_sha,
            )

    def test_formal_dataset_identity_missing_or_mismatch_is_hard_failure(self) -> None:
        manifest, config, protocol_sha, config_sha, source_sha, registry_sha = (
            _valid_manifest()
        )
        missing_config = json.loads(json.dumps(config))
        del missing_config["data"]["formal_dataset_identity"]["dataset_revision"]
        with self.assertRaisesRegex(
            provenance.FormalProvenanceError, "formal_dataset_identity"
        ):
            provenance.validate_data_manifest_fields(
                manifest,
                mode="formal",
                role="calibration",
                config=missing_config,
                protocol_sha256=protocol_sha,
                config_sha256=config_sha,
                expected_prepare_data_source_sha256=source_sha,
                expected_historical_registry_sha256=registry_sha,
            )
        for field, replacement in (
            ("datasets_library_version", "9.9.8"),
            ("dataset_slice_fingerprint", "0" * 16),
            ("dataset_slice_canonical_content_sha256", "0" * 64),
            ("dataset_revision", "d" * 40),
            ("data_preparation_python_environment", "other/bin/python"),
        ):
            with self.subTest(field=field):
                tampered = dict(manifest)
                tampered.pop("manifest_sha256")
                tampered[field] = replacement
                tampered = provenance.add_self_hash(tampered)
                with self.assertRaisesRegex(
                    provenance.FormalProvenanceError,
                    "formal dataset identity mismatch",
                ):
                    provenance.validate_data_manifest_fields(
                        tampered,
                        mode="formal",
                        role="calibration",
                        config=config,
                        protocol_sha256=protocol_sha,
                        config_sha256=config_sha,
                        expected_prepare_data_source_sha256=source_sha,
                        expected_historical_registry_sha256=registry_sha,
                    )


class HistoricalAndOneShotTest(unittest.TestCase):
    def test_sealed_binds_exact_current_calibration_manifest(self) -> None:
        manifest, config, protocol_sha, config_sha, source_sha, _ = _valid_manifest()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data_manifest_calibration.json"
            _write_json(path, manifest)
            with mock.patch.object(
                prepare_data,
                "verify_embedded_formal_signoff",
                return_value={"status": "SIGNED-OFF"},
            ):
                binding = prepare_data.load_formal_calibration_manifest(
                    path,
                    config=config,
                    protocol_sha256=protocol_sha,
                    config_sha256=config_sha,
                    producer_source_sha256=source_sha,
                )
            self.assertEqual(binding.manifest_sha256, manifest["manifest_sha256"])
            self.assertEqual(binding.file_sha256, prepare_data.sha256_file(path))
            self.assertEqual(
                binding.selected_text_sha256,
                tuple(manifest["selected_text_sha256"]),
            )
            self.assertEqual(
                binding.request_ids,
                tuple(row["request_id"] for row in manifest["requests"]),
            )
            with self.assertRaisesRegex(
                prepare_data.DataPreparationError,
                "collide with historical evidence",
            ):
                prepare_data.select_requests(
                    [manifest["requests"][0]["text"]],
                    source_row_start=20,
                    required_count=1,
                    selection_seed=7,
                    min_tokens=2,
                    token_lengths=lambda _text: {"olmoe": 3, "llmjp": 3},
                    historical_hashes=set(binding.selected_text_sha256),
                    role="sealed",
                )
            with self.assertRaisesRegex(
                prepare_data.DataPreparationError,
                "requires --calibration-manifest",
            ):
                prepare_data.load_formal_calibration_manifest(
                    None,
                    config=config,
                    protocol_sha256=protocol_sha,
                    config_sha256=config_sha,
                    producer_source_sha256=source_sha,
                )
            with self.assertRaisesRegex(
                prepare_data.DataPreparationError,
                "config_sha256",
            ):
                prepare_data.load_formal_calibration_manifest(
                    path,
                    config=config,
                    protocol_sha256=protocol_sha,
                    config_sha256="0" * 64,
                    producer_source_sha256=source_sha,
                )

    def test_sealed_manifest_requires_calibration_binding(self) -> None:
        manifest, config, protocol_sha, config_sha, source_sha, registry_sha = (
            _valid_manifest(role="sealed")
        )
        for field in (
            "calibration_manifest_self_hash",
            "calibration_manifest_file_sha256",
            "calibration_selected_list_sha256",
            "calibration_lock_self_hash",
            "calibration_lock_file_sha256",
        ):
            with self.subTest(field=field):
                missing = dict(manifest)
                missing.pop("manifest_sha256")
                missing.pop(field)
                missing = provenance.add_self_hash(missing)
                with self.assertRaisesRegex(
                    provenance.FormalProvenanceError,
                    "calibration",
                ):
                    provenance.validate_data_manifest_fields(
                        missing,
                        mode="formal",
                        role="sealed",
                        config=config,
                        protocol_sha256=protocol_sha,
                        config_sha256=config_sha,
                        expected_prepare_data_source_sha256=source_sha,
                        expected_historical_registry_sha256=registry_sha,
                    )

    def test_calibration_lock_is_g1_and_source_bound_before_sealed(self) -> None:
        config = _tiny_config()
        protocol_sha = "1" * 64
        config_sha = "2" * 64
        lock = _valid_calibration_lock(
            config=config, protocol_sha=protocol_sha, config_sha=config_sha
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibration_lock.json"
            _write_json(path, lock)
            with mock.patch.object(
                prepare_data,
                "verify_calibration_lock_producer_signoff",
                return_value={"status": "SIGNED-OFF"},
            ):
                binding = prepare_data.load_formal_calibration_lock(
                    path,
                    config=config,
                    protocol_sha256=protocol_sha,
                    config_sha256=config_sha,
                )
            self.assertEqual(binding.manifest_sha256, lock["manifest_sha256"])
            self.assertEqual(binding.file_sha256, prepare_data.sha256_file(path))
            for field, replacement, message in (
                ("g1_pass", False, "G1 did not pass"),
                ("g1_pass", "false", "G1 did not pass"),
                ("g1_pass", 1, "G1 did not pass"),
                (
                    "run_experiment_source_sha256",
                    "0" * 64,
                    "run_experiment_source_sha256",
                ),
                ("status", "NO_GO_CURRENT_ACTUATOR", "status"),
            ):
                with self.subTest(field=field):
                    tampered = dict(lock)
                    tampered.pop("manifest_sha256")
                    tampered[field] = replacement
                    _write_json(path, provenance.add_self_hash(tampered))
                    with mock.patch.object(
                        prepare_data,
                        "verify_calibration_lock_producer_signoff",
                        return_value={"status": "SIGNED-OFF"},
                    ):
                        with self.assertRaisesRegex(
                            prepare_data.DataPreparationError, message
                        ):
                            prepare_data.load_formal_calibration_lock(
                                path,
                                config=config,
                                protocol_sha256=protocol_sha,
                                config_sha256=config_sha,
                            )

    def test_self_hashed_lock_without_embedded_runner_signoff_is_rejected(self) -> None:
        config = _tiny_config()
        protocol_sha = "1" * 64
        config_sha = "2" * 64
        lock = _valid_calibration_lock(
            config=config, protocol_sha=protocol_sha, config_sha=config_sha
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibration_lock.json"
            _write_json(path, lock)
            with self.assertRaisesRegex(
                prepare_data.DataPreparationError,
                "embedded producer signoff mismatch",
            ):
                prepare_data.load_formal_calibration_lock(
                    path,
                    config=config,
                    protocol_sha256=protocol_sha,
                    config_sha256=config_sha,
                )

    def test_duplicate_key_calibration_manifest_is_rejected(self) -> None:
        manifest, config, protocol_sha, config_sha, source_sha, _ = _valid_manifest()
        encoded = json.dumps(manifest, sort_keys=True)
        duplicate = encoded[:-1] + ',"role":"calibration"}'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data_manifest_calibration.json"
            path.write_text(duplicate, encoding="utf-8")
            with self.assertRaisesRegex(
                prepare_data.DataPreparationError, "duplicate JSON key"
            ):
                prepare_data.load_formal_calibration_manifest(
                    path,
                    config=config,
                    protocol_sha256=protocol_sha,
                    config_sha256=config_sha,
                    producer_source_sha256=source_sha,
                )

    def test_sealed_prepare_signoff_binds_lock_and_calibration_source_closure(self) -> None:
        binding = prepare_data.CalibrationLockBinding(
            manifest_sha256="1" * 64,
            file_sha256="2" * 64,
            run_experiment_source_sha256="3" * 64,
            scenario_tree_sha256={"olmoe": "4" * 64, "llmjp": "5" * 64},
        )
        with mock.patch.object(
            prepare_data,
            "verify_phase4_signoff",
            return_value={"status": "SIGNED-OFF"},
        ) as verifier:
            prepare_data.require_formal_signoff(
                Path("unused-signoff.json"),
                protocol_sha256="6" * 64,
                config_sha256="7" * 64,
                producer_source_sha256="8" * 64,
                data_role="sealed",
                calibration_lock_binding=binding,
            )
        kwargs = verifier.call_args.kwargs
        expected = kwargs["expected_fields"]
        self.assertEqual(expected["calibration_lock_sha256"], "1" * 64)
        self.assertEqual(expected["calibration_lock_file_sha256"], "2" * 64)
        self.assertEqual(expected["run_experiment_source_sha256"], "3" * 64)
        self.assertEqual(
            expected["scenario_tree_sha256"], dict(binding.scenario_tree_sha256)
        )
        required = set(kwargs["required_source_paths"])
        self.assertTrue(
            set(prepare_data.RUN_EXPERIMENT_SIGNOFF_SOURCE_PATHS).issubset(required)
        )

    def test_formal_prepare_environment_and_loaded_identity_are_exact(self) -> None:
        config = _tiny_config()
        expected_python = REPO_ROOT / ".venv" / "bin" / "python"
        with mock.patch.object(
            prepare_data.sys, "executable", str(expected_python.absolute())
        ):
            prepare_data.validate_formal_data_preparation_environment(
                config, role="calibration"
            )
        with mock.patch.object(
            prepare_data.sys, "executable", "/usr/bin/python3"
        ):
            with self.assertRaisesRegex(
                prepare_data.DataPreparationError, "wrong interpreter"
            ):
                prepare_data.validate_formal_data_preparation_environment(
                    config, role="calibration"
                )
        loaded = prepare_data.LoadedDatasetSlice(
            rows=("a", "b", "c", "d"),
            dataset_repo_id="wikitext",
            dataset_revision="c" * 40,
            dataset_source_urls_sha256="9" * 64,
            datasets_library_version="9.9.9",
            dataset_slice_fingerprint="f" * 16,
            canonical_content_sha256="5" * 64,
        )
        prepare_data.validate_loaded_dataset_identity(
            loaded, config=config, role="calibration"
        )
        drifted = prepare_data.LoadedDatasetSlice(
            **{
                **loaded.__dict__,
                "datasets_library_version": "9.9.8",
            }
        )
        with self.assertRaisesRegex(
            prepare_data.DataPreparationError,
            "datasets_library_version",
        ):
            prepare_data.validate_loaded_dataset_identity(
                drifted, config=config, role="calibration"
            )

    def test_formal_rejects_fake_historical_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fake = Path(directory)
            with self.assertRaisesRegex(
                prepare_data.DataPreparationError, "canonical repository docs"
            ):
                prepare_data.build_historical_registry(fake, formal=True)

    def test_explicit_manifest_without_hash_is_incomplete(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_json(
                root / "data_manifest_old.json",
                {"schema_version": "old-data-manifest-v1", "requests": []},
            )
            registry = prepare_data.build_historical_registry(root)
            self.assertFalse(registry["complete"])
            self.assertEqual(len(registry["parse_failures"]), 1)

    def test_global_o_excl_ledger_blocks_changed_output_and_crash_retry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            signoff_path = root / "signoff.json"
            signoff = {"signoff_sha256": "a" * 64}
            _write_json(signoff_path, signoff)
            lock_binding = prepare_data.CalibrationLockBinding(
                manifest_sha256="1" * 64,
                file_sha256="2" * 64,
                run_experiment_source_sha256="3" * 64,
                scenario_tree_sha256={"olmoe": "4" * 64, "llmjp": "5" * 64},
            )
            ledger = root / "ledger"
            first = prepare_data.reserve_sealed_consumption(
                ledger,
                signoff_path=signoff_path,
                signoff=signoff,
                protocol_sha256="b" * 64,
                config_sha256="c" * 64,
                producer_source_sha256="d" * 64,
                calibration_lock_binding=lock_binding,
                output_dir=root / "output-a",
            )
            self.assertEqual(first["state"], "RESERVED_FAIL_CLOSED")
            tampered = dict(first)
            tampered["output_identity_sha256"] = "0" * 64
            with self.assertRaisesRegex(
                prepare_data.DataPreparationError, "record_sha256 mismatch"
            ):
                prepare_data.finalize_sealed_consumption(
                    ledger,
                    reservation=tampered,
                    manifest_sha256="e" * 64,
                    dataset_slice_canonical_content_sha256="f" * 64,
                )
            consumed = prepare_data.finalize_sealed_consumption(
                ledger,
                reservation=first,
                manifest_sha256="e" * 64,
                dataset_slice_canonical_content_sha256="f" * 64,
            )
            self.assertEqual(consumed["state"], "CONSUMED")
            with self.assertRaisesRegex(
                prepare_data.DataPreparationError, "already reserved"
            ):
                prepare_data.finalize_sealed_consumption(
                    ledger,
                    reservation=first,
                    manifest_sha256="e" * 64,
                    dataset_slice_canonical_content_sha256="f" * 64,
                )
            with self.assertRaisesRegex(
                prepare_data.DataPreparationError, "already reserved"
            ):
                prepare_data.reserve_sealed_consumption(
                    ledger,
                    signoff_path=signoff_path,
                    signoff=signoff,
                    protocol_sha256="b" * 64,
                    config_sha256="c" * 64,
                    producer_source_sha256="d" * 64,
                    calibration_lock_binding=lock_binding,
                    output_dir=root / "output-b",
                )
            crashed = root / "crashed-ledger"
            crashed.mkdir()
            (crashed / "reservation.json").touch()
            with self.assertRaisesRegex(
                prepare_data.DataPreparationError, "already reserved"
            ):
                prepare_data.reserve_sealed_consumption(
                    crashed,
                    signoff_path=signoff_path,
                    signoff=signoff,
                    protocol_sha256="b" * 64,
                    config_sha256="c" * 64,
                    producer_source_sha256="d" * 64,
                    calibration_lock_binding=lock_binding,
                    output_dir=root / "output-c",
                )


class Phase4SignoffTest(unittest.TestCase):
    def _bundle(
        self,
        root: Path,
        *,
        review_text: str = "STATUS: SIGNED-OFF\nOPEN_P0: 0\n",
        test_text: str = (
            "STATUS: PASS\nTOTAL: 11\nPASSED: 10\nFAILED: 0\n"
            "ERRORS: 0\nSKIPPED: 1\n"
        ),
    ) -> tuple[Path, dict[str, object], list[Path]]:
        source = root / "producer.py"
        source.write_text("VALUE = 1\n", encoding="utf-8")
        test_guard = root / "test_guard.py"
        test_guard.write_text("def test_guard(): return True\n", encoding="utf-8")
        review = root / "review.md"
        review.write_text(review_text, encoding="utf-8")
        test_report = root / "tests.txt"
        test_report.write_text(test_text, encoding="utf-8")
        git_head = provenance.current_git_head(REPO_ROOT)
        expected = {
            "protocol_sha256": "1" * 64,
            "config_sha256": "3" * 64,
            "producer_sha256": "2" * 64,
        }
        patch = root / "reviewed.patch"
        reviewed_scope = provenance.add_self_hash(
            {
                "schema_version": "ric-reviewed-worktree-v1",
                "status": "REVIEWED",
                "git_head": git_head,
                "protocol_sha256": expected["protocol_sha256"],
                "config_sha256": expected["config_sha256"],
                "sources": [
                    {
                        "path": source.relative_to(REPO_ROOT).as_posix(),
                        "sha256": provenance.sha256_file(source),
                    },
                    {
                        "path": test_guard.relative_to(REPO_ROOT).as_posix(),
                        "sha256": provenance.sha256_file(test_guard),
                    },
                ],
            },
            field="scope_sha256",
        )
        _write_json(patch, reviewed_scope)
        test_report.write_text(
            test_text
            + f"REVIEWED_SCOPE_SHA256: {provenance.sha256_file(patch)}\n",
            encoding="utf-8",
        )
        head_artifact = root / "git_head.txt"
        head_artifact.write_text(f"{git_head}\n", encoding="ascii")
        source_manifest_path = root / "source_manifest.json"
        source_manifest = provenance.build_source_manifest_payload(
            repo_root=REPO_ROOT,
            source_paths=[source],
            git_head=git_head,
            worktree_patch_sha256=provenance.sha256_file(patch),
        )
        _write_json(source_manifest_path, source_manifest)
        signoff = provenance.build_phase4_signoff_payload(
            repo_root=REPO_ROOT,
            stage="fixture",
            expected_fields=expected,
            artifact_paths={
                "review_report": review,
                "source_manifest": source_manifest_path,
                "test_report": test_report,
                "reviewed_patch": patch,
                "git_head_artifact": head_artifact,
            },
            git_head=git_head,
            test_summary={
                "status": "PASS",
                "total": 11,
                "passed": 10,
                "failed": 0,
                "errors": 0,
                "skipped": 1,
            },
        )
        signoff_path = root / "signoff.json"
        _write_json(signoff_path, signoff)
        return signoff_path, expected, [source]

    def _verify(self, signoff_path: Path, expected: dict[str, object], sources: list[Path]) -> None:
        provenance.verify_phase4_signoff(
            signoff_path,
            repo_root=REPO_ROOT,
            expected_fields={"stage": "fixture", **expected},
            required_source_paths=sources,
            required_reviewed_scope_paths=[
                *sources,
                signoff_path.parent / "test_guard.py",
            ],
        )

    def test_builder_outputs_verify(self) -> None:
        with tempfile.TemporaryDirectory(prefix=".ric-prov-", dir=REPO_ROOT) as directory:
            signoff_path, expected, sources = self._bundle(Path(directory))
            self._verify(signoff_path, expected, sources)

    def test_builder_rejects_fake_review_and_inconsistent_test_report(self) -> None:
        cases = (
            (
                "STATUS: BLOCKED\nOPEN_P0: 1\n",
                "STATUS: PASS\nTOTAL: 11\nPASSED: 10\nFAILED: 0\n"
                "ERRORS: 0\nSKIPPED: 1\n",
                "review report",
            ),
            (
                "STATUS: SIGNED-OFF\nOPEN_P0: 0\n",
                "STATUS: PASS\nTOTAL: 11\nPASSED: 9\nFAILED: 0\n"
                "ERRORS: 0\nSKIPPED: 2\n",
                "count mismatch",
            ),
        )
        for review_text, test_text, message in cases:
            with self.subTest(message=message):
                with tempfile.TemporaryDirectory(
                    prefix=".ric-prov-", dir=REPO_ROOT
                ) as directory:
                    with self.assertRaisesRegex(
                        provenance.FormalProvenanceError, message
                    ):
                        self._bundle(
                            Path(directory),
                            review_text=review_text,
                            test_text=test_text,
                        )

    def test_path_escape_and_symlink_substitution_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix=".ric-prov-", dir=REPO_ROOT) as directory:
            root = Path(directory)
            signoff_path, expected, sources = self._bundle(root)
            value = json.loads(signoff_path.read_text(encoding="utf-8"))
            value.pop("signoff_sha256")
            value["review_report"]["path"] = "../escape.md"
            _write_json(
                signoff_path,
                provenance.add_self_hash(value, field="signoff_sha256"),
            )
            with self.assertRaisesRegex(
                provenance.FormalProvenanceError,
                "repo-relative|outside|forbidden component",
            ):
                self._verify(signoff_path, expected, sources)
        with tempfile.TemporaryDirectory(prefix=".ric-prov-", dir=REPO_ROOT) as directory:
            root = Path(directory)
            signoff_path, expected, sources = self._bundle(root)
            real = root / "real-review.md"
            real.write_text("SIGNED-OFF\n", encoding="utf-8")
            link = root / "review-link.md"
            link.symlink_to(real)
            value = json.loads(signoff_path.read_text(encoding="utf-8"))
            value.pop("signoff_sha256")
            value["review_report"] = {
                "path": link.relative_to(REPO_ROOT).as_posix(),
                "sha256": provenance.sha256_file(real),
            }
            _write_json(
                signoff_path,
                provenance.add_self_hash(value, field="signoff_sha256"),
            )
            with self.assertRaisesRegex(
                provenance.FormalProvenanceError, "symlink"
            ):
                self._verify(signoff_path, expected, sources)

    def test_tampered_report_source_and_test_are_rejected(self) -> None:
        for target_name in ("review.md", "producer.py", "tests.txt"):
            with self.subTest(target=target_name):
                with tempfile.TemporaryDirectory(
                    prefix=".ric-prov-", dir=REPO_ROOT
                ) as directory:
                    root = Path(directory)
                    signoff_path, expected, sources = self._bundle(root)
                    with (root / target_name).open("a", encoding="utf-8") as handle:
                        handle.write("tampered\n")
                    with self.assertRaises(provenance.FormalProvenanceError):
                        self._verify(signoff_path, expected, sources)

    def test_stale_reviewed_scope_cannot_be_resigned_around(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix=".ric-prov-", dir=REPO_ROOT
        ) as directory:
            root = Path(directory)
            signoff_path, expected, sources = self._bundle(root)
            source = sources[0]
            source.write_text("VALUE = 2\n", encoding="utf-8")
            signoff = json.loads(signoff_path.read_text(encoding="utf-8"))
            source_manifest_path = REPO_ROOT / signoff["source_manifest"]["path"]
            source_manifest = json.loads(
                source_manifest_path.read_text(encoding="utf-8")
            )
            source_manifest.pop("manifest_sha256")
            source_manifest["sources"][0]["sha256"] = provenance.sha256_file(source)
            source_manifest = provenance.add_self_hash(source_manifest)
            _write_json(source_manifest_path, source_manifest)
            signoff.pop("signoff_sha256")
            signoff["source_manifest"]["sha256"] = provenance.sha256_file(
                source_manifest_path
            )
            _write_json(
                signoff_path,
                provenance.add_self_hash(signoff, field="signoff_sha256"),
            )
            with self.assertRaisesRegex(
                provenance.FormalProvenanceError,
                "reviewed scope source is stale",
            ):
                self._verify(signoff_path, expected, sources)

    def test_scope_only_test_file_mutation_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix=".ric-prov-", dir=REPO_ROOT
        ) as directory:
            root = Path(directory)
            signoff_path, expected, sources = self._bundle(root)
            (root / "test_guard.py").write_text(
                "def test_guard(): return False\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(
                provenance.FormalProvenanceError,
                "reviewed scope source is stale",
            ):
                self._verify(signoff_path, expected, sources)

    def test_scope_test_omission_cannot_be_resigned_around(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix=".ric-prov-", dir=REPO_ROOT
        ) as directory:
            root = Path(directory)
            signoff_path, expected, sources = self._bundle(root)
            signoff = json.loads(signoff_path.read_text(encoding="utf-8"))
            scope_path = REPO_ROOT / signoff["reviewed_patch"]["path"]
            scope = json.loads(scope_path.read_text(encoding="utf-8"))
            scope.pop("scope_sha256")
            scope["sources"] = [
                row for row in scope["sources"] if not row["path"].endswith("test_guard.py")
            ]
            _write_json(
                scope_path,
                provenance.add_self_hash(scope, field="scope_sha256"),
            )
            scope_file_sha = provenance.sha256_file(scope_path)

            test_report_path = REPO_ROOT / signoff["test_report"]["path"]
            lines = test_report_path.read_text(encoding="utf-8").splitlines()
            test_report_path.write_text(
                "\n".join(
                    f"REVIEWED_SCOPE_SHA256: {scope_file_sha}"
                    if line.startswith("REVIEWED_SCOPE_SHA256: ")
                    else line
                    for line in lines
                )
                + "\n",
                encoding="utf-8",
            )

            source_manifest_path = REPO_ROOT / signoff["source_manifest"]["path"]
            source_manifest = json.loads(
                source_manifest_path.read_text(encoding="utf-8")
            )
            source_manifest.pop("manifest_sha256")
            source_manifest["worktree_patch_sha256"] = scope_file_sha
            _write_json(
                source_manifest_path,
                provenance.add_self_hash(source_manifest),
            )

            signoff.pop("signoff_sha256")
            signoff["worktree_patch_sha256"] = scope_file_sha
            signoff["reviewed_patch"]["sha256"] = scope_file_sha
            signoff["test_report"]["sha256"] = provenance.sha256_file(
                test_report_path
            )
            signoff["source_manifest"]["sha256"] = provenance.sha256_file(
                source_manifest_path
            )
            _write_json(
                signoff_path,
                provenance.add_self_hash(signoff, field="signoff_sha256"),
            )
            with self.assertRaisesRegex(
                provenance.FormalProvenanceError,
                "exact code/test/protocol/config universe",
            ):
                self._verify(signoff_path, expected, sources)


if __name__ == "__main__":
    unittest.main()
