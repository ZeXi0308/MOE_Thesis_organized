from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


MODULE_PATH = Path(__file__).with_name("run_n0c_capture_stage.py")
SPEC = importlib.util.spec_from_file_location("n0c_capture_orchestrator", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

LOCAL_N0B = (
    Path(__file__).parents[1]
    / "outputs/native_route_shape/n0b_valid_window_20260823_westd_r01"
)


class N0cCaptureOrchestratorTest(unittest.TestCase):
    def test_minimal_uploaded_source_bundle_is_independently_importable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            patches = root / "vllm_patches"
            patches.mkdir()
            for source, destination in (
                (MODULE_PATH, root / MODULE_PATH.name),
                (MODULE.COMMON_PATH, root / MODULE.COMMON_PATH.name),
                (
                    MODULE.COMMON_VALIDATOR_PATH,
                    patches / MODULE.COMMON_VALIDATOR_PATH.name,
                ),
                (MODULE.DEVICE_PATCH, patches / MODULE.DEVICE_PATCH.name),
            ):
                shutil.copyfile(source, destination)
            completed = subprocess.run(
                [sys.executable, str(root / MODULE_PATH.name), "--help"],
                text=True,
                capture_output=True,
                check=False,
                env={"PATH": "/usr/bin:/bin:/usr/sbin:/sbin"},
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_schedule_is_exact_32_arm_counterbalanced_contract(self) -> None:
        schedule = MODULE.frozen_schedule()
        self.assertEqual(len(schedule), 32)
        self.assertEqual(len({row["bundle"] for row in schedule}), 32)
        for target_id in MODULE.TARGETS:
            rows = [row for row in schedule if row["target_id"] == target_id]
            self.assertEqual(len(rows), 16)
            for round_id, order in enumerate(MODULE.LATIN_ORDERS):
                actual = [row["arm"] for row in rows if row["round"] == round_id]
                self.assertEqual(actual, list(order))
        for row in schedule:
            self.assertEqual(row["capture_mode"], MODULE.CAPTURE_MODES[row["arm"]])
            self.assertEqual("+device-capture-no-export-v1" in row["runtime_patch_id"], row["arm"] == "capture_only")

    def test_freeze_target_uses_sealed_prefix_and_prompt_tail_cell(self) -> None:
        self.assertTrue(LOCAL_N0B.is_dir())
        with tempfile.TemporaryDirectory() as directory:
            frozen = Path(directory) / "frozen"
            frozen.mkdir()
            paths = {
                target_id: MODULE._freeze_target(LOCAL_N0B, frozen, target_id, target)
                for target_id, target in MODULE.TARGETS.items()
            }
            stock = json.loads(paths["stock_p512_b8_g2_w0"].read_text())
            optimized = json.loads(paths["valid_window_p512_b16_g1_w0"].read_text())
            self.assertEqual(len(stock["prefix_records"]), 12)
            self.assertEqual(len(optimized["prefix_records"]), 36)
            self.assertEqual(stock["target_record"]["execution_order"], 11)
            self.assertEqual(optimized["target_record"]["execution_order"], 35)

    def test_device_patch_applies_to_both_sealed_sources_and_only_runner(self) -> None:
        patch = MODULE.DEVICE_PATCH.resolve()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for runtime in ("stock", "valid-window"):
                destination = root / runtime / "vllm/v1/worker"
                destination.mkdir(parents=True)
                source = LOCAL_N0B / "runtime" / runtime / "vllm/v1/worker/gpu_model_runner.py"
                shutil.copyfile(source, destination / "gpu_model_runner.py")
                completed = subprocess.run(
                    ["git", "-C", str(root / runtime), "apply", "--check", "--include=vllm/**", str(patch)],
                    text=True,
                    capture_output=True,
                    check=False,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_arm_command_contains_only_frozen_identity(self) -> None:
        row = MODULE.frozen_schedule()[0]
        paths = {
            "runner": Path("/frozen/runner.py"),
            "workload": Path("/frozen/workload.json"),
            f"target:{row['target_id']}": Path(f"/frozen/targets/{row['target_id']}/target-spec.json"),
        }
        command = MODULE._arm_command(
            Path("/venv/python"), paths["runner"], Path("/campaign"), paths, row, "a" * 64
        )
        self.assertIn("--require-exclusive-gpu", command)
        self.assertEqual(command[command.index("--capture-mode") + 1], row["capture_mode"])
        self.assertEqual(
            command[command.index("--logical-runtime-variant") + 1],
            MODULE._runtime_variant(row),
        )
        self.assertEqual(
            command[command.index("--expected-runtime-root") + 1],
            str(Path("/campaign/runtime") / MODULE._runtime_variant(row)),
        )
        for forbidden in ("--model", "--revision", "--seed", "--batch-size", "--output-tokens"):
            self.assertNotIn(forbidden, command)

    def test_campaign_import_probe_covers_all_exact_runtime_variants(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def probe(_python: Path, environment: dict[str, str]) -> dict[str, str]:
                source_root = environment["PYTHONPATH"].split(":", 1)[0]
                return {
                    "source_root": source_root,
                    "module_file": str(Path(source_root) / "vllm/__init__.py"),
                    "version": "0.26.0",
                }

            with mock.patch.object(MODULE.COMMON, "_import_probe", side_effect=probe) as patched:
                reports = MODULE._verify_campaign_runtime_imports(
                    Path("/venv/python"), root, {}
                )
            self.assertEqual(set(reports), set(MODULE.RUNTIME_VARIANTS))
            self.assertEqual(patched.call_count, 4)
            self.assertTrue(
                all(report["runtime_import_root_verified"] is True for report in reports.values())
            )

    def test_campaign_import_probe_rejects_escaped_import_root(self) -> None:
        escaped = {
            "source_root": "/installed/site-packages",
            "module_file": "/installed/site-packages/vllm/__init__.py",
            "version": "0.26.0",
        }
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            MODULE.COMMON, "_import_probe", return_value=escaped
        ):
            with self.assertRaisesRegex(RuntimeError, "escaped expected root"):
                MODULE._verify_campaign_runtime_imports(
                    Path("/venv/python"), Path(directory), {}
                )

    def test_cli_exposes_no_scientific_override(self) -> None:
        parser = MODULE.build_parser()
        run_parser = parser._subparsers._group_actions[0].choices["run"]
        options = {option for action in run_parser._actions for option in action.option_strings}
        for forbidden in ("--targets", "--rounds", "--arm-order", "--seed", "--model", "--threshold"):
            self.assertNotIn(forbidden, options)


if __name__ == "__main__":
    unittest.main()
