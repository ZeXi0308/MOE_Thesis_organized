from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Callable

import numpy as np


MODULE_PATH = Path(__file__).with_name("evaluate_n0c_capture_stage.py")
SPEC = importlib.util.spec_from_file_location("n0c_capture_evaluator", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


class Fixture:
    TARGETS = tuple(MODULE.TARGET_RUNTIMES.items())
    BASE_PATCHES = {
        "stock": "stock-vllm-0.26.0",
        "valid-window": "valid-window-clear-v1",
    }

    def __init__(self, root: Path) -> None:
        self.root = root
        self.target_plans: dict[str, dict[str, Any]] = {}
        self.target_identities: dict[str, dict[str, str]] = {}
        self._make_frozen_targets()
        self.runtime_manifest_sha = self._make_runtimes()
        self.bundles = root / "bundles"
        self.bundles.mkdir(parents=True)
        for target, runtime in self.TARGETS:
            for round_id in MODULE.ROUNDS:
                for arm in MODULE.ARMS:
                    self._make(target, runtime, round_id, arm)
        schedule = []
        target_ids = tuple(MODULE.TARGET_RUNTIMES)
        for round_id, arms in enumerate(MODULE.LATIN_ARM_ORDERS):
            targets = target_ids if round_id % 2 == 0 else tuple(reversed(target_ids))
            for target in targets:
                runtime = MODULE.TARGET_RUNTIMES[target]
                for arm in arms:
                    schedule.append(
                        {
                            "target_id": target,
                            "target_runtime": runtime,
                            "round": round_id,
                            "arm": arm,
                            "capture_mode": MODULE.CAPTURE_MODES[arm],
                            "runtime_patch_id": self._patch_id(runtime, arm),
                            "bundle": f"{target}-r{round_id}-{arm}",
                        }
                    )
        frozen_manifest = {
            f"frozen/{relative}": digest
            for relative, digest in MODULE._package_manifest(root / "frozen").items()
        }
        _write(
            root / "run_plan.json",
            {
                "schema": "n0c-capture-stage-orchestration-v1",
                "claim_ceiling": MODULE.CLAIM_CEILING,
                "rounds": 4,
                "targets": self.target_plans,
                "schedule": schedule,
                "runtime_package_manifest_sha256": self.runtime_manifest_sha,
                "campaign_runtime_imports": {
                    variant: self._runtime_import_probe(variant)
                    for variant in self.runtime_manifest_sha
                },
                "frozen_input_sha256": frozen_manifest,
            },
        )

    def path(self, target: str, round_id: int, arm: str) -> Path:
        return self.bundles / f"{target}-r{round_id}-{arm}"

    def _patch_id(self, runtime: str, arm: str) -> str:
        base = self.BASE_PATCHES[runtime]
        return f"{base}+device-capture-no-export-v1" if arm == "capture_only" else base

    def _runtime_import_probe(self, variant: str) -> dict[str, Any]:
        expected = (self.root / "runtime" / variant).resolve()
        return {
            "source_root": str(expected),
            "module_file": str(expected / "vllm/__init__.py"),
            "version": "0.26.0",
            "logical_runtime_variant": variant,
            "expected_runtime_root": str(expected),
            "runtime_import_root_verified": True,
        }

    def _make_frozen_targets(self) -> None:
        frozen = self.root / "frozen"
        frozen.mkdir(parents=True)
        (frozen / "run_n0c_capture_stage_arm.py").write_text("fixture runner\n")
        _write(frozen / "workload.json", {"fixture": True})
        for target, runtime in self.TARGETS:
            expected = MODULE.RUNTIME_SHAPES[runtime]
            target_root = frozen / "targets" / target
            inputs = target_root / "inputs"
            inputs.mkdir(parents=True)
            prompts = np.arange(expected["batch"] * 512, dtype=np.int32).reshape(
                expected["batch"], 512
            )
            artifact = inputs / "target.npz"
            np.savez(artifact, prompt_token_ids=prompts)
            input_sha = _sha(artifact)
            prompt_sha = MODULE._json_sha256(prompts.astype(np.int64).tolist())
            records = [
                {
                    "batch_id": f"fixture-{target}-{index}",
                    "execution_order": index,
                    "prompt_length": 512,
                    "batch_size": expected["batch"],
                    "group": 2 if runtime == "stock" else 1,
                    "within_process_repeat": 0,
                    "input_artifact": "inputs/target.npz",
                    "input_artifact_sha256": input_sha,
                    "prompt_token_ids_sha256": prompt_sha,
                }
                for index in range(expected["prefix_cells"])
            ]
            source_bundle = "stock_on-r1" if runtime == "stock" else "optimized_on-r1"
            source_batches_sha = _digest(f"source-batches:{target}")
            spec = {
                "schema": "n0c-capture-target-spec-v1",
                "target_id": target,
                "target_runtime": runtime,
                "source_bundle": source_bundle,
                "source_batches_sha256": source_batches_sha,
                "prefix_plan_sha256": MODULE._json_sha256(records),
                "prefix_records": records,
                "target_record": records[-1],
            }
            target_root.mkdir(parents=True, exist_ok=True)
            _write(target_root / "target-spec.json", spec)
            self.target_identities[target] = {
                "target_spec_sha256": _sha(target_root / "target-spec.json"),
                "prefix_plan_sha256": spec["prefix_plan_sha256"],
                "target_input_artifact_sha256": input_sha,
                "target_prompt_token_ids_sha256": prompt_sha,
            }
            self.target_plans[target] = {
                "target_runtime": runtime,
                "source_bundle": source_bundle,
                "source_batches_sha256": source_batches_sha,
                "batch_id": records[-1]["batch_id"],
                "batch_size": expected["batch"],
                "execution_order": expected["prefix_cells"] - 1,
                "target_input_artifact_sha256": input_sha,
                "target_prompt_token_ids_sha256": prompt_sha,
                "base_runtime_patch_id": self.BASE_PATCHES[runtime],
            }

    def _make_runtimes(self) -> dict[str, str]:
        manifests: dict[str, dict[str, str]] = {}
        for runtime in MODULE.RUNTIME_SHAPES:
            for suffix in ("", "-device"):
                variant = f"{runtime}{suffix}"
                package = self.root / "runtime" / variant / "vllm"
                capturer = package / "model_executor/layers/fused_moe/routed_experts_capturer.py"
                runner = package / "v1/worker/gpu_model_runner.py"
                capturer.parent.mkdir(parents=True)
                runner.parent.mkdir(parents=True)
                capturer.write_text(f"capturer:{runtime}\n")
                runner.write_text(f"runner:{runtime}{suffix}\n")
                manifests[variant] = MODULE._package_manifest(package.parent)
        _write(self.root / "runtime-package-manifest.json", manifests)
        _write(
            self.root / "frozen" / "n0b-runtime-package-manifest.json",
            {"stock": manifests["stock"], "optimized": manifests["valid-window"]},
        )
        return {variant: MODULE._json_sha256(manifest) for variant, manifest in manifests.items()}

    def _make(self, target: str, runtime: str, round_id: int, arm: str) -> None:
        path = self.path(target, round_id, arm)
        path.mkdir()
        expected = MODULE.RUNTIME_SHAPES[runtime]
        identity = self.target_identities[target]
        variant = f"{runtime}{'-device' if arm == 'capture_only' else ''}"
        manifest = json.loads((self.root / "runtime-package-manifest.json").read_text())[variant]
        config = {
            "schema": MODULE.CONFIG_SCHEMA,
            "target_id": target,
            "target_runtime": runtime,
            "round": round_id,
            "arm": arm,
            "capture_mode": MODULE.CAPTURE_MODES[arm],
            "logical_runtime_variant": variant,
            "runtime_import_root_verified": True,
            "runtime_patch_id": self._patch_id(runtime, arm),
            "claim_ceiling": MODULE.CLAIM_CEILING,
            **identity,
            "runtime_package_manifest_sha256": self.runtime_manifest_sha[variant],
            "workload_manifest_sha256": _sha(self.root / "frozen" / "workload.json"),
            "producer_source_sha256": _sha(
                self.root / "frozen" / "run_n0c_capture_stage_arm.py"
            ),
            "runtime_identity": {
                "vllm": "0.26.0",
                "vllm_module_file": str(
                    (self.root / "runtime" / variant / "vllm/__init__.py").resolve()
                ),
                "vllm_package": str(
                    (self.root / "runtime" / variant / "vllm").resolve()
                ),
                "vllm_source_root": str(
                    (self.root / "runtime" / variant).resolve()
                ),
                "expected_runtime_root": str(
                    (self.root / "runtime" / variant).resolve()
                ),
                "logical_runtime_variant": variant,
                "runtime_import_root_verified": True,
                "vllm_batch_invariant": "0",
                "n0c_device_capture_only": "1" if arm == "capture_only" else "0",
                "source_sha256": {
                    "model_executor/layers/fused_moe/routed_experts_capturer.py": manifest[
                        "vllm/model_executor/layers/fused_moe/routed_experts_capturer.py"
                    ],
                    "v1/worker/gpu_model_runner.py": manifest[
                        "vllm/v1/worker/gpu_model_runner.py"
                    ],
                },
            },
        }
        result = {
            **{key: config[key] for key in MODULE.IDENTITY_FIELDS},
            "schema": MODULE.RESULT_SCHEMA,
            "status": "COMPLETE",
            "output_token_ids": [[request * 100 + step for step in range(16)] for request in range(expected["batch"])],
            "warmup_count": 6,
            "prefix_cells_executed": expected["prefix_cells"],
        }
        route_sha: str | None = None
        if arm == "full_export":
            shape = [expected["batch"], 16, 16, 8]
            routes = np.arange(np.prod(shape), dtype=np.int16).reshape(shape) % 64
            route_path = path / "routes.npz"
            np.savez(route_path, routes=routes)
            route_sha = _sha(route_path)
            result.update(
                {
                    "full_export_includes_prompt_tail": True,
                    "route_mapping": [
                        {"route_row": index, "input_position": 511 + index, "produces_output_token_index": index}
                        for index in range(16)
                    ],
                    "route_artifact": "routes.npz",
                    "route_artifact_sha256": route_sha,
                    "route_shape": shape,
                }
            )
        _write(path / "config.json", config)
        _write(path / "result.json", result)
        _write(
            path / "RUN_COMPLETE.json",
            {
                "status": "RUN_COMPLETE",
                "config_sha256": _sha(path / "config.json"),
                "result_sha256": _sha(path / "result.json"),
                "route_sha256": route_sha,
            },
        )

    def mutate_result(
        self, target: str, round_id: int, arm: str, change: Callable[[dict[str, Any]], None]
    ) -> None:
        path = self.path(target, round_id, arm)
        result = json.loads((path / "result.json").read_text())
        change(result)
        _write(path / "result.json", result)
        seal = json.loads((path / "RUN_COMPLETE.json").read_text())
        seal["result_sha256"] = _sha(path / "result.json")
        _write(path / "RUN_COMPLETE.json", seal)

    def mutate_identity(
        self, target: str, round_id: int, arm: str, change: Callable[[dict[str, Any]], None]
    ) -> None:
        path = self.path(target, round_id, arm)
        for filename in ("config.json", "result.json"):
            payload = json.loads((path / filename).read_text())
            change(payload)
            _write(path / filename, payload)
        seal = json.loads((path / "RUN_COMPLETE.json").read_text())
        seal["config_sha256"] = _sha(path / "config.json")
        seal["result_sha256"] = _sha(path / "result.json")
        _write(path / "RUN_COMPLETE.json", seal)

    def drift_token(
        self,
        target: str,
        rounds: range | tuple[int, ...],
        arm: str,
        *,
        value: int = 999999,
        request_row: int = 0,
        output_token_index: int = 0,
    ) -> None:
        for round_id in rounds:
            self.mutate_result(
                target, round_id, arm,
                lambda result: result["output_token_ids"][request_row].__setitem__(
                    output_token_index, value
                ),
            )

    def mutate_routes(self, target: str, round_id: int, value: int) -> None:
        path = self.path(target, round_id, "full_export")
        with np.load(path / "routes.npz", allow_pickle=False) as archive:
            routes = np.array(archive["routes"], copy=True)
        routes[0, 0, 0, 0] = value
        np.savez(path / "routes.npz", routes=routes)
        route_sha = _sha(path / "routes.npz")
        result = json.loads((path / "result.json").read_text())
        result["route_artifact_sha256"] = route_sha
        _write(path / "result.json", result)
        seal = json.loads((path / "RUN_COMPLETE.json").read_text())
        seal["route_sha256"] = route_sha
        seal["result_sha256"] = _sha(path / "result.json")
        _write(path / "RUN_COMPLETE.json", seal)


class N0cCaptureEvaluationTest(unittest.TestCase):
    def evaluate(self, fixture: Fixture) -> dict[str, Any]:
        return MODULE.evaluate_campaign(fixture.root)

    def test_clean_campaign_is_not_reproduced_and_never_unlocks_controller(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            report = self.evaluate(Fixture(Path(temporary)))
        self.assertEqual(report["status"], "NOT_REPRODUCED")
        self.assertTrue(report["structurally_valid"])
        self.assertFalse(report["controller_unlocked"])
        self.assertEqual(report["claim_ceiling"], MODULE.CLAIM_CEILING)

    def test_distinct_device_patch_and_manifest_are_valid_schedule_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary))
            base = json.loads(
                (fixture.path("stock_p512_b8_g2_w0", 0, "n_a") / "config.json").read_text()
            )
            device = json.loads(
                (fixture.path("stock_p512_b8_g2_w0", 0, "capture_only") / "config.json").read_text()
            )
            self.assertNotEqual(base["runtime_patch_id"], device["runtime_patch_id"])
            self.assertNotEqual(
                base["runtime_package_manifest_sha256"],
                device["runtime_package_manifest_sha256"],
            )
            self.assertEqual(self.evaluate(fixture)["status"], "NOT_REPRODUCED")

    def test_common_capture_threshold_requires_same_signature_in_three_of_four(self) -> None:
        for count, expected in (
            (2, "INTERMITTENT_OR_UNRESOLVED"),
            (3, "CAPTURE_NO_EXPORT_ASSOCIATION"),
        ):
            with self.subTest(count=count), tempfile.TemporaryDirectory() as temporary:
                fixture = Fixture(Path(temporary))
                fixture.drift_token("stock_p512_b8_g2_w0", tuple(range(count)), "capture_only")
                fixture.drift_token("stock_p512_b8_g2_w0", tuple(range(count)), "full_export")
                self.assertEqual(self.evaluate(fixture)["status"], expected)

    def test_export_requires_stable_capture_and_same_signature_in_three_of_four(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary))
            fixture.drift_token("stock_p512_b8_g2_w0", (0, 1, 2), "full_export")
            self.assertEqual(self.evaluate(fixture)["status"], "EXPORT_PATH_ASSOCIATION")
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary))
            fixture.drift_token("stock_p512_b8_g2_w0", (3,), "capture_only")
            fixture.drift_token("stock_p512_b8_g2_w0", (0, 1, 2), "full_export")
            self.assertEqual(self.evaluate(fixture)["status"], "INTERMITTENT_OR_UNRESOLVED")

    def test_capture_only_drift_without_full_export_is_a_contradiction(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary))
            fixture.drift_token("stock_p512_b8_g2_w0", (0, 1, 2), "capture_only")
            report = self.evaluate(fixture)
        self.assertEqual(report["status"], "INTERMITTENT_OR_UNRESOLVED")
        stock = next(target for target in report["targets"] if target["target_runtime"] == "stock")
        self.assertEqual(
            [row["contrast"] for row in stock["per_round_token_contrasts"]],
            ["CAPTURE_ONLY_CONTRADICTION"] * 3 + ["NO_DIVERGENCE"],
        )

    def test_capture_and_export_with_different_signatures_are_discordant(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary))
            fixture.drift_token("stock_p512_b8_g2_w0", (0, 1, 2), "capture_only")
            fixture.drift_token(
                "stock_p512_b8_g2_w0", (0, 1, 2), "full_export", value=999998
            )
            report = self.evaluate(fixture)
        self.assertEqual(report["status"], "INTERMITTENT_OR_UNRESOLVED")
        stock = next(target for target in report["targets"] if target["target_runtime"] == "stock")
        self.assertEqual(
            [row["contrast"] for row in stock["per_round_token_contrasts"]],
            ["DISCORDANT"] * 3 + ["NO_DIVERGENCE"],
        )

    def test_three_nonmatching_common_signatures_do_not_pass_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary))
            for round_id, value in enumerate((999991, 999992, 999993)):
                fixture.drift_token(
                    "stock_p512_b8_g2_w0", (round_id,), "capture_only", value=value
                )
                fixture.drift_token(
                    "stock_p512_b8_g2_w0", (round_id,), "full_export", value=value
                )
            report = self.evaluate(fixture)
        self.assertEqual(report["status"], "INTERMITTENT_OR_UNRESOLVED")

    def test_baseline_nondeterminism_precedes_capture_perturbation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary))
            fixture.drift_token("stock_p512_b8_g2_w0", (0,), "n_b")
            fixture.drift_token("stock_p512_b8_g2_w0", (0, 1, 2, 3), "capture_only")
            self.assertEqual(self.evaluate(fixture)["status"], "BASELINE_DISCRETE_NONDETERMINISM")

    def test_invalid_seal_precedes_scientific_classification(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary))
            fixture.drift_token("stock_p512_b8_g2_w0", (0,), "n_b")
            seal_path = fixture.path("stock_p512_b8_g2_w0", 0, "n_a") / "RUN_COMPLETE.json"
            seal = json.loads(seal_path.read_text())
            seal["config_sha256"] = "0" * 64
            _write(seal_path, seal)
            report = self.evaluate(fixture)
        self.assertEqual(report["status"], "INVALID_CAMPAIGN")
        self.assertEqual(report["failure_category"], "IDENTITY_SEAL_INPUT_OR_SOURCE_INVALID")

    def test_invalid_source_identity_hash_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary))
            path = fixture.path("stock_p512_b8_g2_w0", 0, "n_a")
            for filename in ("config.json", "result.json"):
                payload = json.loads((path / filename).read_text())
                payload["runtime_package_manifest_sha256"] = "not-a-sha"
                _write(path / filename, payload)
            seal = json.loads((path / "RUN_COMPLETE.json").read_text())
            seal["config_sha256"] = _sha(path / "config.json")
            seal["result_sha256"] = _sha(path / "result.json")
            _write(path / "RUN_COMPLETE.json", seal)
            report = self.evaluate(fixture)
        self.assertEqual(report["status"], "INVALID_CAMPAIGN")
        self.assertTrue(any("invalid_identity_hash" in error for error in report["errors"]))

    def test_missing_identity_field_is_structured_invalid_and_cli_exit_two(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = Fixture(root / "campaign")
            fixture.mutate_identity(
                "stock_p512_b8_g2_w0", 0, "n_a",
                lambda payload: payload.pop("target_spec_sha256", None),
            )
            output = root / "invalid.json"
            completed = subprocess.run(
                [sys.executable, str(MODULE_PATH), "--campaign-root", str(fixture.root), "--output", str(output)],
                text=True, capture_output=True, check=False,
            )
            self.assertEqual(completed.returncode, 2, completed.stderr)
            self.assertEqual(json.loads(output.read_text())["status"], "INVALID_CAMPAIGN")

    def test_coherent_fake_bundle_target_hashes_do_not_override_frozen_spec(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary))
            for round_id in MODULE.ROUNDS:
                for arm in MODULE.ARMS:
                    fixture.mutate_identity(
                        "stock_p512_b8_g2_w0", round_id, arm,
                        lambda payload: payload.update({"target_spec_sha256": "a" * 64}),
                    )
            report = self.evaluate(fixture)
        self.assertEqual(report["status"], "INVALID_CAMPAIGN")
        self.assertTrue(any("target_spec_hash_mismatch" in error for error in report["errors"]))

    def test_runtime_manifest_must_be_stable_within_base_variant(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary))
            path = fixture.path("stock_p512_b8_g2_w0", 1, "n_b")
            for filename in ("config.json", "result.json"):
                payload = json.loads((path / filename).read_text())
                payload["runtime_package_manifest_sha256"] = _digest("unexpected-base-package")
                _write(path / filename, payload)
            seal = json.loads((path / "RUN_COMPLETE.json").read_text())
            seal["config_sha256"] = _sha(path / "config.json")
            seal["result_sha256"] = _sha(path / "result.json")
            _write(path / "RUN_COMPLETE.json", seal)
            report = self.evaluate(fixture)
        self.assertEqual(report["status"], "INVALID_CAMPAIGN")
        self.assertTrue(any("runtime_variant_identity_drift" in error for error in report["errors"]))

    def test_escaped_arm_import_root_is_invalid_even_when_bundle_is_resealed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary))
            path = fixture.path("stock_p512_b8_g2_w0", 0, "n_a")
            config = json.loads((path / "config.json").read_text())
            escaped = Path("/installed/site-packages")
            config["runtime_identity"].update(
                {
                    "expected_runtime_root": str(escaped),
                    "vllm_source_root": str(escaped),
                    "vllm_package": str(escaped / "vllm"),
                    "vllm_module_file": str(escaped / "vllm/__init__.py"),
                }
            )
            _write(path / "config.json", config)
            seal = json.loads((path / "RUN_COMPLETE.json").read_text())
            seal["config_sha256"] = _sha(path / "config.json")
            _write(path / "RUN_COMPLETE.json", seal)
            report = self.evaluate(fixture)
        self.assertEqual(report["status"], "INVALID_CAMPAIGN")
        self.assertTrue(any("runtime_identity_mismatch" in error for error in report["errors"]))

    def test_malformed_campaign_import_probe_fails_closed_without_crashing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary))
            plan_path = fixture.root / "run_plan.json"
            plan = json.loads(plan_path.read_text())
            plan["campaign_runtime_imports"]["stock"] = "not-an-import-probe"
            _write(plan_path, plan)
            report = self.evaluate(fixture)
        self.assertEqual(report["status"], "INVALID_CAMPAIGN")
        self.assertTrue(any("import_probe_invalid:stock" in error for error in report["errors"]))

    def test_prompt_tail_and_true_off_contracts_fail_closed(self) -> None:
        changes = (
            ("full_export", lambda result: result["route_mapping"][0].update({"produces_output_token_index": 1})),
            ("n_a", lambda result: result.update({"route_shape": [8, 16, 16, 8]})),
        )
        for arm, change in changes:
            with self.subTest(arm=arm), tempfile.TemporaryDirectory() as temporary:
                fixture = Fixture(Path(temporary))
                fixture.mutate_result("stock_p512_b8_g2_w0", 0, arm, change)
                self.assertEqual(self.evaluate(fixture)["status"], "INVALID_CAMPAIGN")

    def test_route_set_drift_is_diagnostic_and_never_replaces_token_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary))
            fixture.mutate_routes("stock_p512_b8_g2_w0", 1, 63)
            fixture.mutate_result(
                "stock_p512_b8_g2_w0", 0, "n_a", lambda result: result.update({"float_logits_sha256": "f" * 64})
            )
            report = self.evaluate(fixture)
        self.assertEqual(report["status"], "NOT_REPRODUCED")
        stock = next(target for target in report["targets"] if target["target_runtime"] == "stock")
        self.assertEqual(stock["route_diagnostic_only"]["exact_route_drift_rounds"], [1])
        self.assertFalse(stock["route_diagnostic_only"]["used_for_token_attribution"])
        self.assertFalse(report["float_hashes_used_for_token_explanation"])

    def test_out_of_range_expert_id_is_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary))
            fixture.mutate_routes("stock_p512_b8_g2_w0", 1, 64)
            self.assertEqual(self.evaluate(fixture)["status"], "INVALID_CAMPAIGN")

    def test_corrupt_but_correctly_sealed_route_npz_is_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary))
            path = fixture.path("stock_p512_b8_g2_w0", 0, "full_export")
            (path / "routes.npz").write_bytes(b"PK-corrupt-npz")
            route_sha = _sha(path / "routes.npz")
            result = json.loads((path / "result.json").read_text())
            result["route_artifact_sha256"] = route_sha
            _write(path / "result.json", result)
            seal = json.loads((path / "RUN_COMPLETE.json").read_text())
            seal["route_sha256"] = route_sha
            seal["result_sha256"] = _sha(path / "result.json")
            _write(path / "RUN_COMPLETE.json", seal)
            report = self.evaluate(fixture)
        self.assertEqual(report["status"], "INVALID_CAMPAIGN")
        self.assertTrue(any("invalid_route_artifact" in error for error in report["errors"]))

    def test_run_plan_latin_order_is_part_of_input_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary))
            plan_path = fixture.root / "run_plan.json"
            plan = json.loads(plan_path.read_text())
            plan["schedule"][0], plan["schedule"][1] = plan["schedule"][1], plan["schedule"][0]
            _write(plan_path, plan)
            report = self.evaluate(fixture)
        self.assertEqual(report["status"], "INVALID_CAMPAIGN")
        self.assertTrue(any("frozen_latin_order" in error for error in report["errors"]))

    def test_cli_output_is_write_once(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            Fixture(root / "campaign")
            output = root / "report.json"
            output.write_text("sentinel")
            completed = subprocess.run(
                [sys.executable, str(MODULE_PATH), "--campaign-root", str(root / "campaign"), "--output", str(output)],
                text=True, capture_output=True, check=False,
            )
            self.assertEqual(completed.returncode, 2)
            self.assertEqual(output.read_text(), "sentinel")

    def test_cli_exit_zero_for_valid_scientific_verdict(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            Fixture(root / "campaign")
            output = root / "report.json"
            completed = subprocess.run(
                [sys.executable, str(MODULE_PATH), "--campaign-root", str(root / "campaign"), "--output", str(output)],
                text=True, capture_output=True, check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(json.loads(output.read_text())["status"], "NOT_REPRODUCED")


if __name__ == "__main__":
    unittest.main()
