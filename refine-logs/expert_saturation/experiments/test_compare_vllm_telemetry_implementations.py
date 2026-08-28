from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).with_name("compare_vllm_telemetry_implementations.py")
SPEC = importlib.util.spec_from_file_location("telemetry_implementation_gate", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

FIXTURE_PATH = Path(__file__).with_name("test_compare_vllm_route_probe_runs.py")
FIXTURE_SPEC = importlib.util.spec_from_file_location("route_pair_test_fixtures", FIXTURE_PATH)
assert FIXTURE_SPEC is not None and FIXTURE_SPEC.loader is not None
FIXTURES = importlib.util.module_from_spec(FIXTURE_SPEC)
FIXTURE_SPEC.loader.exec_module(FIXTURES)


def pair(status: str, token_parity: bool = True, **updates: object) -> dict[str, object]:
    result: dict[str, object] = {
        "status": status,
        "pair_count": 1,
        "token_parity": token_parity,
        "config_drift": {},
        "duplicate_keys": False,
        "missing_on": [],
        "missing_off": [],
        "incomplete_on": [],
        "incomplete_off": [],
        "unexpected_on": [],
        "unexpected_off": [],
        "prompt_digest_mismatches": [],
        "timing_errors": [],
        "row_schema_errors": [],
        "threshold_valid": True,
    }
    result.update(updates)
    return result


def routes(**updates: object) -> dict[str, object]:
    result: dict[str, object] = {
        "expected_cells": 1,
        "comparable_cells": 1,
        "exact_route_cells": 1,
        "route_mismatch_keys": [],
        "token_drift_keys": [],
        "prompt_digest_mismatch_keys": [],
        "missing_keys": [],
        "validation_errors": [],
        "qualified": True,
    }
    result.update(updates)
    return result


def implementation_bundle(
    root: Path,
    *,
    arm: str,
    repeat: int,
    capture_routes: bool,
) -> Path:
    path = FIXTURES._make_bundle(
        root,
        f"{arm}-r{repeat}",
        capture_routes,
        wall_ms=10.0,
        tpot_ms=2.0,
    )
    implementation = "stock" if arm.startswith("stock_") else "optimized"
    state = "original" if implementation == "stock" else "patched"
    producer = Path(__file__).with_name("run_vllm_route_shape_probe.py").read_bytes()
    producer_hash = hashlib.sha256(producer).hexdigest()
    (path / "producer_source.py").write_bytes(producer)

    environment = FIXTURES._read_json(path / "environment.json")
    environment["vllm_runtime_sources"] = {
        relative.removeprefix("vllm/"): {"sha256": values[state]}
        for relative, values in MODULE.VALIDATOR.FILES.items()
    }
    FIXTURES._write_json(path / "environment.json", environment)

    config = FIXTURES._read_json(path / "config.json")
    config["process_repeat"] = repeat
    config["runtime_patch_id"] = MODULE.EXPECTED_PATCH_IDS[implementation]
    config["runtime_identity"] = {
        key: environment[key]
        for key in environment
        if key != "compute_processes_before_engine_init"
    }
    config.update(
        {
            "probe_script_sha256": producer_hash,
            "producer_source_artifact": "producer_source.py",
            "producer_source_artifact_sha256": producer_hash,
        }
    )
    FIXTURES._write_json(path / "config.json", config)

    rows = FIXTURES._read_rows(path)
    for row in rows:
        row["process_repeat"] = repeat
    FIXTURES._write_rows(path, rows)

    manifest = FIXTURES._read_json(path / "ARTIFACT_HASHES.json")
    manifest["environment.json"] = hashlib.sha256(
        (path / "environment.json").read_bytes()
    ).hexdigest()
    manifest["producer_source.py"] = producer_hash
    FIXTURES._write_json(path / "ARTIFACT_HASHES.json", manifest)
    FIXTURES._seal_top_level(path)
    return path


class TelemetryImplementationDecisionTest(unittest.TestCase):
    def choose(
        self,
        optimized: dict[str, object] | None = None,
        *,
        stock: dict[str, object] | None = None,
        source_valid: bool = True,
        off_parity: bool = True,
        route_semantics: dict[str, object] | None = None,
    ) -> tuple[str, str | None]:
        return MODULE.choose_status(
            stock or pair("INVALID_TELEMETRY_PAIR", token_parity=False),
            optimized or pair("TELEMETRY_TIMING_DEVIATION_QUALIFIED"),
            {"valid": source_valid},
            off_parity,
            {},
            route_semantics or routes(),
        )

    def test_qualified_allows_perturbative_but_structurally_valid_stock_control(self) -> None:
        status, failure = self.choose()
        self.assertEqual(status, "VALID_WINDOW_TELEMETRY_QUALIFIED")
        self.assertIsNone(failure)

    def test_token_drift_fails_transparency(self) -> None:
        status, failure = self.choose(
            pair("INVALID_TELEMETRY_PAIR", token_parity=False)
        )
        self.assertEqual(status, "VALID_WINDOW_NOT_TRANSPARENT")
        self.assertEqual(failure, "TELEMETRY_TOKEN_DRIFT")

    def test_timing_deviation_above_gate_is_not_qualified(self) -> None:
        status, failure = self.choose(
            pair("ROUTE_EXPORT_TIMING_DEVIATION_ABOVE_THRESHOLD")
        )
        self.assertEqual(status, "VALID_WINDOW_TOO_PERTURBATIVE")
        self.assertEqual(failure, "TELEMETRY_TIMING_DEVIATION_ABOVE_THRESHOLD")

    def test_route_off_control_must_be_inert(self) -> None:
        status, failure = self.choose(off_parity=False)
        self.assertEqual(status, "INVALID_PATCH_CONTROL")
        self.assertEqual(failure, "PATCH_AFFECTS_ROUTE_OFF_EXECUTION")

    def test_zero_comparable_or_cross_token_drift_cannot_qualify(self) -> None:
        status, failure = self.choose(
            route_semantics=routes(
                expected_cells=1,
                comparable_cells=0,
                exact_route_cells=0,
                token_drift_keys=[[128, 4, 0, 0]],
                qualified=False,
            )
        )
        self.assertEqual(status, "VALID_WINDOW_ROUTE_SEMANTICS_INCONCLUSIVE")
        self.assertEqual(failure, "CROSS_IMPLEMENTATION_TOKEN_DRIFT_OR_ZERO_SUPPORT")

    def test_stock_pair_structure_is_not_dead_control(self) -> None:
        status, failure = self.choose(stock=pair("INVALID_TELEMETRY_PAIR", missing_on=[[1]]))
        self.assertEqual(status, "INVALID_IMPLEMENTATION_PAIR")
        self.assertEqual(failure, "CONFIG_COVERAGE_OR_ARTIFACT_DRIFT")

    def test_label_only_source_identity_cannot_qualify(self) -> None:
        status, failure = self.choose(source_valid=False)
        self.assertEqual(status, "INVALID_IMPLEMENTATION_IDENTITY")
        self.assertEqual(failure, "SOURCE_IDENTITY_NOT_VALIDATOR_APPROVED")

    def test_single_repeat_is_invalid_coverage(self) -> None:
        bundle_sets = {
            name: {0: {}} for name in ("stock_off", "stock_on", "optimized_off", "optimized_on")
        }
        errors = MODULE._repeat_coverage_errors(bundle_sets)
        self.assertTrue(any("minimum_process_repeats" in error for error in errors))

    def test_campaign_token_drift_takes_precedence_over_timing_failure(self) -> None:
        timing = {
            "process_repeat": 0,
            "status": "VALID_WINDOW_TOO_PERTURBATIVE",
            "failure_category": "TELEMETRY_TIMING_DEVIATION_ABOVE_THRESHOLD",
        }
        drift = {
            "process_repeat": 1,
            "status": "VALID_WINDOW_NOT_TRANSPARENT",
            "failure_category": "TELEMETRY_TOKEN_DRIFT",
        }
        for reports in ([timing, drift], [drift, timing]):
            with self.subTest(order=[row["process_repeat"] for row in reports]):
                selected = MODULE._select_campaign_verdict(reports)
                self.assertIsNotNone(selected)
                assert selected is not None
                self.assertEqual(selected["process_repeat"], 1)
                self.assertEqual(selected["failure_category"], "TELEMETRY_TOKEN_DRIFT")

    def test_campaign_precedence_is_explicit_for_route_and_invalid_failures(self) -> None:
        route_inconclusive = {
            "status": "VALID_WINDOW_ROUTE_SEMANTICS_INCONCLUSIVE",
        }
        route_mismatch = {"status": "VALID_WINDOW_ROUTE_SEMANTICS_MISMATCH"}
        self.assertIs(
            MODULE._select_campaign_verdict([route_inconclusive, route_mismatch]),
            route_mismatch,
        )
        patch_control = {"status": "INVALID_PATCH_CONTROL"}
        source_identity = {"status": "INVALID_IMPLEMENTATION_IDENTITY"}
        self.assertIs(
            MODULE._select_campaign_verdict([patch_control, source_identity]),
            source_identity,
        )

    def test_cli_exit_contract_is_fail_closed(self) -> None:
        self.assertEqual(MODULE.exit_code("VALID_WINDOW_TELEMETRY_QUALIFIED"), 0)
        self.assertEqual(MODULE.exit_code("VALID_WINDOW_NOT_TRANSPARENT"), 1)
        self.assertEqual(MODULE.exit_code("VALID_WINDOW_TOO_PERTURBATIVE"), 1)
        self.assertEqual(MODULE.exit_code("INVALID_INPUT"), 2)

    def test_function_rejects_nonfinite_or_nonfrozen_threshold_before_bundle_io(self) -> None:
        missing = Path("does-not-need-to-exist")
        cases = (
            (float("nan"), "threshold_is_not_finite"),
            (float("inf"), "threshold_is_not_finite"),
            (-1.0, "threshold_is_negative"),
            (100.0, "threshold_does_not_match_frozen_gate"),
        )
        for threshold, expected_error in cases:
            with self.subTest(threshold=threshold):
                report = MODULE.compare_implementations(
                    missing,
                    missing,
                    missing,
                    missing,
                    max_p95_overhead_pct=threshold,
                )
                self.assertEqual(report["status"], "INVALID_INPUT")
                self.assertEqual(report["failure_category"], "INVALID_THRESHOLD")
                self.assertEqual(
                    report["max_p95_absolute_timing_deviation_pct"], 5.0
                )
                self.assertIn(expected_error, report["errors"]["threshold"])
                json.dumps(report, allow_nan=False)

    def test_cli_writes_structured_invalid_report_for_nan_and_relaxed_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            missing = root / "missing-bundle"
            for threshold in ("nan", "100"):
                with self.subTest(threshold=threshold):
                    output = root / f"invalid-{threshold}.json"
                    completed = subprocess.run(
                        [
                            sys.executable,
                            str(MODULE_PATH),
                            "--stock-off",
                            str(missing),
                            "--stock-on",
                            str(missing),
                            "--optimized-off",
                            str(missing),
                            "--optimized-on",
                            str(missing),
                            "--output",
                            str(output),
                            "--max-p95-overhead-pct",
                            threshold,
                        ],
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    self.assertEqual(completed.returncode, 2, completed.stderr)
                    self.assertTrue(output.is_file())
                    report = json.loads(output.read_text())
                    self.assertEqual(report["status"], "INVALID_INPUT")
                    self.assertEqual(report["failure_category"], "INVALID_THRESHOLD")
                    self.assertEqual(
                        report["max_p95_absolute_timing_deviation_pct"], 5.0
                    )
                    self.assertNotIn("NaN", output.read_text())

    def test_source_identity_is_bound_to_validator_hashes_and_embedded_producer(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            producer = Path(__file__).with_name("run_vllm_route_shape_probe.py").read_bytes()
            producer_hash = hashlib.sha256(producer).hexdigest()
            bundle_sets: dict[str, dict[int, dict[str, object]]] = {}
            for arm in ("stock_off", "stock_on", "optimized_off", "optimized_on"):
                path = root / arm
                path.mkdir()
                (path / "producer_source.py").write_bytes(producer)
                (path / "ARTIFACT_HASHES.json").write_text(
                    json.dumps({"producer_source.py": producer_hash})
                )
                implementation = "stock" if arm.startswith("stock_") else "optimized"
                state = "original" if implementation == "stock" else "patched"
                sources = {
                    relative.removeprefix("vllm/"): {"sha256": values[state]}
                    for relative, values in MODULE.VALIDATOR.FILES.items()
                }
                config = {
                    "runtime_patch_id": MODULE.EXPECTED_PATCH_IDS[implementation],
                    "probe_script_sha256": producer_hash,
                    "producer_source_artifact": "producer_source.py",
                    "producer_source_artifact_sha256": producer_hash,
                    "runtime_identity": {
                        "vllm": "0.26.0",
                        "vllm_runtime_sources": sources,
                    },
                }
                bundle_sets[arm] = {0: {"path": path, "config": config}}

            report = MODULE._validate_source_identity(bundle_sets)
            self.assertTrue(report["valid"], report["errors"])

            optimized = bundle_sets["optimized_on"][0]["config"]
            optimized["runtime_identity"]["vllm_runtime_sources"] = {
                relative.removeprefix("vllm/"): {"sha256": values["original"]}
                for relative, values in MODULE.VALIDATOR.FILES.items()
            }
            report = MODULE._validate_source_identity(bundle_sets)
            self.assertFalse(report["valid"])
            self.assertTrue(
                any("runtime_source_hashes_not_validator_approved" in error for error in report["errors"])
            )

    def test_route_semantics_rejects_malformed_topk_even_when_arrays_match(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundles = []
            for name in ("stock", "optimized"):
                path = root / name
                path.mkdir()
                routes_array = np.zeros((1, 3, 2, 2), dtype=np.int16)
                route_path = path / "route.npz"
                np.savez_compressed(route_path, routes=routes_array)
                digest = hashlib.sha256(route_path.read_bytes()).hexdigest()
                (path / "ARTIFACT_HASHES.json").write_text(
                    json.dumps({"route.npz": digest})
                )
                row = {
                    "prompt_length": 8,
                    "batch_size": 1,
                    "group": 0,
                    "within_process_repeat": 0,
                    "prompt_token_ids_sha256": "same",
                    "request_metrics": [{"token_ids": [1, 2, 3, 4]}],
                    "route_artifact": "route.npz",
                    "route_artifact_sha256": digest,
                }
                bundles.append(
                    {
                        "path": path.resolve(),
                        "config": {
                            "output_tokens": 4,
                            "model_shape": {
                                "num_experts": 8,
                                "num_layers": 2,
                                "top_k": 2,
                            },
                        },
                        "map": {(8, 1, 0, 0): row},
                    }
                )

            report = MODULE._route_semantics(*bundles)

        self.assertFalse(report["qualified"])
        self.assertEqual(report["comparable_cells"], 1)
        self.assertTrue(report["validation_errors"])

    def test_two_complete_repeats_execute_the_full_implementation_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = {
                arm: [
                    implementation_bundle(
                        root,
                        arm=arm,
                        repeat=repeat,
                        capture_routes=arm.endswith("_on"),
                    )
                    for repeat in range(2)
                ]
                for arm in (
                    "stock_off",
                    "stock_on",
                    "optimized_off",
                    "optimized_on",
                )
            }
            report = MODULE.compare_implementations(
                paths["stock_off"],
                paths["stock_on"],
                paths["optimized_off"],
                paths["optimized_on"],
            )

        self.assertEqual(report["status"], "VALID_WINDOW_TELEMETRY_QUALIFIED")
        self.assertEqual(report["process_repeats"], [0, 1])
        self.assertTrue(report["all_required_repeats_retained"])
        self.assertTrue(report["source_identity"]["valid"])
        self.assertEqual(len(report["repeat_reports"]), 2)

    def test_full_gate_requires_exclusive_gpu_in_every_arm_and_repeat(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = {
                arm: [
                    implementation_bundle(
                        root,
                        arm=arm,
                        repeat=repeat,
                        capture_routes=arm.endswith("_on"),
                    )
                    for repeat in range(2)
                ]
                for arm in (
                    "stock_off",
                    "stock_on",
                    "optimized_off",
                    "optimized_on",
                )
            }
            for bundles in paths.values():
                for path in bundles:
                    config = FIXTURES._read_json(path / "config.json")
                    config["require_exclusive_gpu"] = False
                    FIXTURES._write_json(path / "config.json", config)
                    FIXTURES._seal_top_level(path)
            report = MODULE.compare_implementations(
                paths["stock_off"],
                paths["stock_on"],
                paths["optimized_off"],
                paths["optimized_on"],
            )

        self.assertEqual(report["status"], "INVALID_INPUT")
        self.assertEqual(report["failure_category"], "ENVIRONMENT_NOT_QUALIFIED")
        self.assertEqual(len(report["errors"]["environment"]), 8)

    def test_full_gate_rejects_a_single_repeat(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = {
                arm: [
                    implementation_bundle(
                        root,
                        arm=arm,
                        repeat=0,
                        capture_routes=arm.endswith("_on"),
                    )
                ]
                for arm in (
                    "stock_off",
                    "stock_on",
                    "optimized_off",
                    "optimized_on",
                )
            }
            report = MODULE.compare_implementations(
                paths["stock_off"],
                paths["stock_on"],
                paths["optimized_off"],
                paths["optimized_on"],
            )

        self.assertEqual(report["status"], "INVALID_INPUT")
        self.assertTrue(
            any(
                "minimum_process_repeats" in error
                for error in report["errors"]["repeat_coverage"]
            )
        )


if __name__ == "__main__":
    unittest.main()
