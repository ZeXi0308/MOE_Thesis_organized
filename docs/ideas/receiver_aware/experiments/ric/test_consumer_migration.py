from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock


try:
    from .build_scenarios import (
        DEFAULT_CONFIG,
        DEFAULT_CONSUMER_AMENDMENT,
        DEFAULT_PROTOCOL,
        HISTORICAL_REVIEWED_SOURCE_SNAPSHOT_SHA256,
        FORMAL_AUTHORITATIVE_BUNDLE_ROOT,
        FORMAL_CENSUS_RELATIVE_ROOTS,
        GLOBAL_SEALED_STATE_DIR,
        GLOBAL_SEALED_EVALUATION_CONSUMPTION,
        REPO_ROOT,
        ScenarioBuildError,
        _snapshot_sources,
        object_sha256,
        verify_immutable_upstream_signoff,
        validate_consumer_amendment_path,
        validate_frozen_formal_paths,
        validate_formal_output_path,
        verify_pre_outcome_attestation,
        verify_sealed_calibration_manifest_binding,
    )
    from .capture_preoutcome_attestation import (
        PreOutcomeAttestationError,
        _sealed_outcome_reason,
        _is_atomic_partial_path,
        capture,
    )
    from .formal_provenance import add_self_hash, load_json_mapping_strict, sha256_file
except ImportError:
    from build_scenarios import (  # type: ignore
        DEFAULT_CONFIG,
        DEFAULT_CONSUMER_AMENDMENT,
        DEFAULT_PROTOCOL,
        HISTORICAL_REVIEWED_SOURCE_SNAPSHOT_SHA256,
        FORMAL_AUTHORITATIVE_BUNDLE_ROOT,
        FORMAL_CENSUS_RELATIVE_ROOTS,
        GLOBAL_SEALED_STATE_DIR,
        GLOBAL_SEALED_EVALUATION_CONSUMPTION,
        REPO_ROOT,
        ScenarioBuildError,
        _snapshot_sources,
        object_sha256,
        verify_immutable_upstream_signoff,
        validate_consumer_amendment_path,
        validate_frozen_formal_paths,
        validate_formal_output_path,
        verify_pre_outcome_attestation,
        verify_sealed_calibration_manifest_binding,
    )
    from capture_preoutcome_attestation import (  # type: ignore
        PreOutcomeAttestationError,
        _sealed_outcome_reason,
        _is_atomic_partial_path,
        capture,
    )
    from formal_provenance import (  # type: ignore
        add_self_hash,
        load_json_mapping_strict,
        sha256_file,
    )


V5 = REPO_ROOT / "docs/ideas/receiver_aware/formal_signoff/v5"
SNAPSHOT = V5 / "historical_snapshot/RIC_Phase4_ReviewedSources_v5.tar"


class ConsumerMigrationTests(unittest.TestCase):
    def test_historical_data_route_lut_signoffs_replay_complete_chain(self) -> None:
        cases = (
            V5 / "signoff_prepare_data_calibration.json",
            V5 / "signoff_prepare_data_sealed.json",
            V5 / "signoff_capture_routes_olmoe.json",
            V5 / "signoff_capture_routes_llmjp.json",
            V5 / "signoff_capture_routes_sealed_olmoe.json",
            V5 / "signoff_capture_routes_sealed_llmjp.json",
            V5 / "signoff_measure_service_lut_olmoe.json",
            V5 / "signoff_measure_service_lut_llmjp.json",
            V5 / "signoff_measure_capability_olmoe.json",
            V5 / "signoff_measure_capability_llmjp.json",
        )
        for path in cases:
            signoff = load_json_mapping_strict(path, label=path.name)
            expected = {
                field: signoff[field]
                for field in ("stage", "protocol_sha256", "config_sha256")
            }
            verified = verify_immutable_upstream_signoff(
                path,
                snapshot_path=SNAPSHOT,
                expected_fields=expected,
                expected_signoff_file_sha256=sha256_file(path),
            )
            self.assertEqual(verified["signoff_sha256"], signoff["signoff_sha256"])

    def test_reself_hashed_post_review_signoff_is_not_whitelisted(self) -> None:
        original = V5 / "signoff_prepare_data_calibration.json"
        value = load_json_mapping_strict(original, label="original signoff")
        unhashed = dict(value)
        unhashed.pop("signoff_sha256")
        unhashed["forged_after_review"] = True
        forged = add_self_hash(unhashed, field="signoff_sha256")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "producer_signoff.json"
            path.write_text(
                json.dumps(forged, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ScenarioBuildError, "not the pre-outcome registered file"
            ):
                verify_immutable_upstream_signoff(
                    path,
                    snapshot_path=SNAPSHOT,
                    expected_fields={
                        "stage": "prepare_data",
                        "protocol_sha256": value["protocol_sha256"],
                        "config_sha256": value["config_sha256"],
                    },
                    expected_signoff_file_sha256=sha256_file(original),
                )

    def test_corrupted_historical_snapshot_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            corrupt = Path(directory) / "snapshot.tar"
            corrupt.write_bytes(SNAPSHOT.read_bytes() + b"corrupt")
            with self.assertRaisesRegex(ScenarioBuildError, "snapshot hash mismatch"):
                _snapshot_sources(corrupt)

    def test_calibration_manifest_substitution_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibration_manifest.json"
            calibration = {
                "manifest_sha256": "a" * 64,
                "selected_text_sha256": ["b" * 64, "c" * 64],
            }
            path.write_text(json.dumps(calibration) + "\n", encoding="utf-8")
            sealed = {
                "calibration_manifest_self_hash": calibration["manifest_sha256"],
                "calibration_manifest_file_sha256": sha256_file(path),
                "calibration_selected_list_sha256": object_sha256(
                    calibration["selected_text_sha256"]
                ),
            }
            verified = verify_sealed_calibration_manifest_binding(
                sealed, calibration, calibration_manifest_path=path
            )
            self.assertEqual(
                verified["calibration_manifest_file_sha256"], sha256_file(path)
            )
            substituted = dict(calibration)
            substituted["selected_text_sha256"] = ["d" * 64]
            with self.assertRaisesRegex(ScenarioBuildError, "substitution detected"):
                verify_sealed_calibration_manifest_binding(
                    sealed, substituted, calibration_manifest_path=path
                )

    def test_pre_outcome_census_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "bundle"
            (root / "formal_outputs").mkdir(parents=True)
            (root / "state").mkdir()
            root = root.resolve(strict=True)
            fixture = root / "formal_outputs/input.txt"
            fixture.write_text("input-only\n", encoding="utf-8")
            output = root / "docs/pre_outcome.json"
            signoff = Path(directory) / "producer_signoff.json"
            signoff.write_text("{}\n", encoding="utf-8")
            verified_signoff = {"signoff_sha256": "a" * 64}
            with (
                mock.patch.dict(
                    capture.__globals__,
                    {
                        "FORMAL_AUTHORITATIVE_BUNDLE_ROOT": root,
                        "FORMAL_CENSUS_RELATIVE_ROOTS": ("formal_outputs", "state"),
                        "FORMAL_PREOUTCOME_ATTESTATION_PATH": output,
                        "verify_phase4_signoff": mock.Mock(
                            return_value=verified_signoff
                        ),
                    },
                ),
                mock.patch.dict(
                    verify_pre_outcome_attestation.__globals__,
                    {
                        "FORMAL_AUTHORITATIVE_BUNDLE_ROOT": root,
                        "FORMAL_CENSUS_RELATIVE_ROOTS": ("formal_outputs", "state"),
                        "FORMAL_PREOUTCOME_ATTESTATION_PATH": output,
                        "verify_phase4_signoff": mock.Mock(
                            return_value=verified_signoff
                        ),
                    },
                ),
            ):
                payload = capture(
                    scanned_root=root,
                    output_path=output,
                    config_path=DEFAULT_CONFIG,
                    protocol_path=DEFAULT_PROTOCOL,
                    amendment_path=DEFAULT_CONSUMER_AMENDMENT,
                    snapshot_path=SNAPSHOT,
                    required_inputs={"fixture_input": fixture},
                    signoff_path=signoff,
                )
                verified = verify_pre_outcome_attestation(
                    output,
                    protocol_sha256=sha256_file(DEFAULT_PROTOCOL),
                    config_sha256=sha256_file(DEFAULT_CONFIG),
                    consumer_amendment_sha256=sha256_file(
                        DEFAULT_CONSUMER_AMENDMENT
                    ),
                    authoritative_bundle_root=root,
                    required_input_paths=(fixture,),
                    producer_signoff_path=signoff,
                )
            self.assertEqual(
                payload["historical_reviewed_source_snapshot_sha256"],
                HISTORICAL_REVIEWED_SOURCE_SNAPSHOT_SHA256,
            )
            self.assertEqual(
                verified["attestation_sha256"], payload["attestation_sha256"]
            )
            signoff.write_text('{"tampered":true}\n', encoding="utf-8")
            with (
                mock.patch.dict(
                    verify_pre_outcome_attestation.__globals__,
                    {
                        "FORMAL_AUTHORITATIVE_BUNDLE_ROOT": root,
                        "FORMAL_CENSUS_RELATIVE_ROOTS": ("formal_outputs", "state"),
                        "FORMAL_PREOUTCOME_ATTESTATION_PATH": output,
                    },
                ),
                self.assertRaisesRegex(
                    ScenarioBuildError, "producer_signoff_file_sha256"
                ),
            ):
                verify_pre_outcome_attestation(
                    output,
                    protocol_sha256=sha256_file(DEFAULT_PROTOCOL),
                    config_sha256=sha256_file(DEFAULT_CONFIG),
                    consumer_amendment_sha256=sha256_file(
                        DEFAULT_CONSUMER_AMENDMENT
                    ),
                    authoritative_bundle_root=root,
                    required_input_paths=(fixture,),
                    producer_signoff_path=signoff,
                )

    def test_pre_outcome_alternate_output_path_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "bundle"
            (root / "formal_outputs").mkdir(parents=True)
            (root / "state").mkdir()
            root = root.resolve(strict=True)
            fixed = root / "docs/pre_outcome.json"
            signoff = Path(directory) / "signoff.json"
            signoff.write_text("{}\n", encoding="utf-8")
            fixture = root / "formal_outputs/input.bin"
            fixture.write_bytes(b"input")
            with (
                mock.patch.dict(
                    capture.__globals__,
                    {
                        "FORMAL_AUTHORITATIVE_BUNDLE_ROOT": root,
                        "FORMAL_PREOUTCOME_ATTESTATION_PATH": fixed,
                    },
                ),
                self.assertRaisesRegex(
                    PreOutcomeAttestationError, "write-once attestation path"
                ),
            ):
                capture(
                    scanned_root=root,
                    output_path=root / "docs/second_attestation.json",
                    config_path=DEFAULT_CONFIG,
                    protocol_path=DEFAULT_PROTOCOL,
                    amendment_path=DEFAULT_CONSUMER_AMENDMENT,
                    snapshot_path=SNAPSHOT,
                    required_inputs={"fixture": fixture},
                    signoff_path=signoff,
                )

    def test_legacy_evaluation_ledger_is_forbidden_but_data_ledger_is_allowed(self) -> None:
        legacy = {
            "schema_version": "ric-sealed-consumption-v1",
            "role": "sealed",
            "scenario_tree_sha256": {},
            "oracle_status_sha256": "o" * 64,
            "run_experiment_source_sha256": "r" * 64,
        }
        self.assertEqual(
            _sealed_outcome_reason(Path("legacy.json"), legacy),
            "legacy sealed evaluation was already consumed",
        )
        data_consumption = {
            "schema_version": "ric-sealed-consumption-v1",
            "state": "CONSUMED",
            "role": "sealed",
            "mode": "formal",
            "reservation_sha256": "a" * 64,
        }
        self.assertIsNone(
            _sealed_outcome_reason(Path("consumption.json"), data_consumption)
        )

    def test_atomic_crash_partial_is_recognized_anywhere_under_root(self) -> None:
        root = Path("/formal")
        self.assertTrue(
            _is_atomic_partial_path(
                root, root / "legacy/.sealed-run.partial-deadbeef/action_trace.csv"
            )
        )
        self.assertFalse(
            _is_atomic_partial_path(root, root / "formal_outputs/calibration")
        )

    def test_pre_outcome_census_rejects_existing_sealed_scenario(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "bundle"
            (root / "formal_outputs").mkdir(parents=True)
            (root / "state").mkdir()
            root = root.resolve(strict=True)
            (root / "legacy_outputs").mkdir()
            scenario = root / "legacy_outputs/scenario_tree.json"
            scenario.write_text(
                json.dumps(
                    {
                        "schema_version": "ric-scenario-tree-v1",
                        "role": "sealed",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            signoff = Path(directory) / "signoff.json"
            signoff.write_text("{}\n", encoding="utf-8")
            output = root / "docs/pre_outcome.json"
            with (
                mock.patch.dict(
                    capture.__globals__,
                    {
                        "FORMAL_AUTHORITATIVE_BUNDLE_ROOT": root,
                        "FORMAL_CENSUS_RELATIVE_ROOTS": (".",),
                        "FORMAL_PREOUTCOME_ATTESTATION_PATH": output,
                        "verify_phase4_signoff": mock.Mock(
                            return_value={"signoff_sha256": "a" * 64}
                        ),
                    },
                ),
                self.assertRaisesRegex(
                    PreOutcomeAttestationError, "sealed outcome already exists"
                ),
            ):
                capture(
                    scanned_root=root,
                    output_path=output,
                    config_path=DEFAULT_CONFIG,
                    protocol_path=DEFAULT_PROTOCOL,
                    amendment_path=DEFAULT_CONSUMER_AMENDMENT,
                    snapshot_path=SNAPSHOT,
                    required_inputs={"existing_sealed_scenario": scenario},
                    signoff_path=signoff,
                )

    def test_pre_outcome_decoy_root_is_rejected_by_consumer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            authoritative = base / "formal_outputs"
            decoy = base / "decoy"
            authoritative.mkdir()
            decoy.mkdir()
            (authoritative / "input.bin").write_bytes(b"immutable")
            (decoy / "input.bin").write_bytes(b"immutable")
            signoff = base / "signoff.json"
            signoff.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(
                PreOutcomeAttestationError, "reviewed formal bundle root"
            ):
                capture(
                    scanned_root=decoy,
                    output_path=base / "pre_outcome.json",
                    config_path=DEFAULT_CONFIG,
                    protocol_path=DEFAULT_PROTOCOL,
                    amendment_path=DEFAULT_CONSUMER_AMENDMENT,
                    snapshot_path=SNAPSHOT,
                    required_inputs={"decoy": decoy / "input.bin"},
                    signoff_path=signoff,
                )

    def test_formal_custom_amendment_substitution_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            substituted = Path(directory) / "amendment.md"
            substituted.write_text(
                DEFAULT_CONSUMER_AMENDMENT.read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ScenarioBuildError, "differs from reviewed Amendment Q"
            ):
                validate_consumer_amendment_path(substituted, mode="formal")

    def test_formal_custom_config_and_protocol_paths_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            custom_config = root / "ric_v1.json"
            custom_protocol = root / "protocol.md"
            custom_config.write_bytes(DEFAULT_CONFIG.read_bytes())
            custom_protocol.write_bytes(DEFAULT_PROTOCOL.read_bytes())
            with self.assertRaisesRegex(ScenarioBuildError, "formal config path"):
                validate_frozen_formal_paths(
                    config_path=custom_config,
                    protocol_path=DEFAULT_PROTOCOL,
                    mode="formal",
                )
            with self.assertRaisesRegex(ScenarioBuildError, "formal protocol path"):
                validate_frozen_formal_paths(
                    config_path=DEFAULT_CONFIG,
                    protocol_path=custom_protocol,
                    mode="formal",
                )

    def test_formal_output_must_stay_inside_authoritative_namespace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve(strict=True)
            (root / "formal_outputs").mkdir()
            with mock.patch.dict(
                validate_formal_output_path.__globals__,
                {"FORMAL_AUTHORITATIVE_BUNDLE_ROOT": root},
            ):
                accepted = validate_formal_output_path(
                    root / "formal_outputs/scenario_v6", mode="formal"
                )
                self.assertEqual(accepted, root / "formal_outputs/scenario_v6")
                with self.assertRaisesRegex(
                    ScenarioBuildError, "outside the reviewed formal_outputs root"
                ):
                    validate_formal_output_path(root / "elsewhere", mode="formal")

    def test_global_sealed_ledgers_are_anchored_to_authoritative_root(self) -> None:
        relative = GLOBAL_SEALED_STATE_DIR.relative_to(
            FORMAL_AUTHORITATIVE_BUNDLE_ROOT
        ).as_posix()
        self.assertTrue(relative)
        self.assertEqual(FORMAL_CENSUS_RELATIVE_ROOTS, (".",))
        self.assertEqual(
            GLOBAL_SEALED_EVALUATION_CONSUMPTION.parent,
            GLOBAL_SEALED_STATE_DIR,
        )


if __name__ == "__main__":
    unittest.main()
