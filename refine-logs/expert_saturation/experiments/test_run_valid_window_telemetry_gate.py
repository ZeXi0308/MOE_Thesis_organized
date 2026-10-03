from __future__ import annotations

import argparse
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock


MODULE_PATH = Path(__file__).with_name("run_valid_window_telemetry_gate.py")
SPEC = importlib.util.spec_from_file_location("valid_window_gate_runner", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

CANONICAL_WORKLOAD = (
    Path(__file__).parents[1]
    / "outputs/native_route_shape/remote_snapshot_20260823/runs/full-off-r0-20260823"
    / "workload_manifest.json"
)


def _args(root: Path, workload: Path = CANONICAL_WORKLOAD) -> argparse.Namespace:
    return argparse.Namespace(
        python="/usr/bin/python3",
        vllm_source_root=None,
        output_root=str(root / "result"),
        workload_manifest=str(workload),
        arm_timeout_seconds=30,
        comparison_timeout_seconds=30,
    )


class ValidWindowGateOrchestratorTest(unittest.TestCase):
    def test_canonical_workload_fixture_matches_frozen_hash(self) -> None:
        self.assertTrue(CANONICAL_WORKLOAD.is_file())
        self.assertEqual(MODULE._sha256(CANONICAL_WORKLOAD), MODULE.FROZEN_WORKLOAD_SHA256)

    def test_schedule_is_exact_and_counterbalanced(self) -> None:
        rows = MODULE.FROZEN_SCHEDULE
        self.assertEqual(len(rows), 8)
        for repeat in MODULE.PROCESS_REPEATS:
            arms = [row[0] for row in rows if row[1] == repeat]
            self.assertCountEqual(
                arms, ["stock_off", "stock_on", "optimized_off", "optimized_on"]
            )
        self.assertEqual([row[0] for row in rows[:4]], [
            "stock_off", "stock_on", "optimized_off", "optimized_on"
        ])
        self.assertEqual([row[0] for row in rows[4:]], [
            "optimized_on", "optimized_off", "stock_on", "stock_off"
        ])

    def test_arm_command_has_only_frozen_scientific_configuration(self) -> None:
        command = MODULE._arm_command(
            Path("/venv/python"), Path("/frozen/runner.py"),
            Path("/frozen/workload.json"), Path("/bundle"),
            1, True, "valid-window-clear-v1",
        )
        self.assertIn("--capture-routes", command)
        self.assertIn("--require-exclusive-gpu", command)
        self.assertEqual(command[command.index("--process-repeat") + 1], "1")
        self.assertEqual(command[command.index("--within-process-repeats") + 1], "1")
        self.assertEqual(command[command.index("--order-seed") + 1], str(MODULE.ORDER_SEED))
        self.assertEqual(command[command.index("--model") + 1], MODULE.MODEL)

    def test_run_cli_does_not_expose_scientific_gate_overrides(self) -> None:
        parser = MODULE.build_parser()
        run_parser = parser._subparsers._group_actions[0].choices["run"]
        options = {option for action in run_parser._actions for option in action.option_strings}
        for forbidden in (
            "--model", "--batch-sizes", "--prompt-lengths", "--seed",
            "--order-seed", "--process-repeats", "--max-p95-overhead-pct",
            "--no-require-exclusive-gpu", "--runtime-patch-id",
        ):
            self.assertNotIn(forbidden, options)

    def test_comparison_command_names_all_original_bundles_once(self) -> None:
        command = MODULE._comparison_command(
            Path("/venv/python"), Path("/frozen/comparator.py"), Path("/campaign")
        )
        joined = " ".join(command)
        for arm in ("stock_off", "stock_on", "optimized_off", "optimized_on"):
            for repeat in MODULE.PROCESS_REPEATS:
                self.assertEqual(joined.count(f"{arm}-r{repeat}"), 1)
        self.assertNotIn("--max-p95-overhead-pct", command)

    def test_overlay_environment_precedes_existing_pythonpath(self) -> None:
        environment = MODULE._overlay_environment({"PYTHONPATH": "/existing"}, Path("/overlay"))
        self.assertEqual(environment["PYTHONPATH"], "/overlay:/existing")

    def test_base_environment_freezes_offline_model_cache(self) -> None:
        inherited_aliases = {key: "/external-cache" for key in MODULE.CACHE_ALIAS_KEYS}
        with mock.patch.dict(MODULE.os.environ, inherited_aliases, clear=False):
            environment = MODULE._base_environment()
        self.assertEqual(environment["HF_HOME"], MODULE.HF_HOME)
        self.assertEqual(environment["HF_HUB_CACHE"], MODULE.HF_HUB_CACHE)
        self.assertEqual(environment["HF_HUB_OFFLINE"], "1")
        self.assertEqual(environment["TRANSFORMERS_OFFLINE"], "1")
        for key in MODULE.CACHE_ALIAS_KEYS:
            self.assertNotIn(key, environment)

    def test_interpreter_path_preserves_venv_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "base-python"
            target.write_text("placeholder")
            link = root / "venv-python"
            link.symlink_to(target)
            selected = MODULE._interpreter_path(str(link))
            self.assertEqual(selected, link)
            self.assertNotEqual(selected, target.resolve())

    def test_overlay_dereferences_symlinked_ancestor_before_patch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stock = root / "stock"
            external = root / "external/model_executor"
            external.mkdir(parents=True)
            (external / "target.py").write_text("stock")
            (stock / "vllm").mkdir(parents=True)
            (stock / "vllm/model_executor").symlink_to(external, target_is_directory=True)
            overlay = root / "overlay"
            with mock.patch.object(
                MODULE.VALIDATOR, "FILES", {"vllm/model_executor/target.py": {}}
            ):
                MODULE._copy_package(stock, overlay)
            self.assertFalse((overlay / "vllm/model_executor").is_symlink())
            (overlay / "vllm/model_executor/target.py").write_text("patched")
            self.assertEqual((external / "target.py").read_text(), "stock")

    def test_existing_output_is_write_once(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = _args(Path(directory))
            Path(args.output_root).mkdir()
            with self.assertRaisesRegex(RuntimeError, "already exists"):
                MODULE.execute(args)

    def test_noncanonical_workload_stops_before_preflight(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workload = root / "workload.json"
            workload.write_text('{"requests": [{"prompt": "easy"}]}')
            with mock.patch.object(MODULE, "_preflight") as preflight:
                with self.assertRaisesRegex(RuntimeError, "frozen SHA-256"):
                    MODULE.execute(_args(root, workload))
            preflight.assert_not_called()

    def test_control_source_must_match_e0_reviewed_allowlist(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with mock.patch.dict(
                MODULE.E0_REVIEWED_SHA256, {"runner": "0" * 64}, clear=False
            ):
                with self.assertRaisesRegex(RuntimeError, "E0-reviewed snapshot"):
                    MODULE._freeze_inputs(root, CANONICAL_WORKLOAD.read_bytes())

    def test_workload_mutation_during_preflight_cannot_change_staged_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workload = root / "workload.json"
            canonical_bytes = CANONICAL_WORKLOAD.read_bytes()
            workload.write_bytes(canonical_bytes)
            args = _args(root, workload)
            preflight = {
                "python": args.python,
                "vllm_source_root": str(root / "site-packages"),
                "vllm_import": {"source_root": str(root / "site-packages"),
                                "module_file": "vllm/__init__.py", "version": "0.26.0"},
                "source": {"valid": True, "source_state": "original"},
                "exclusive_gpu_verified": True,
                "compute_processes": [],
            }

            def mutate_then_return(_args):
                workload.write_bytes(b"mutated-after-initial-read")
                return preflight

            runtime = {
                "stock_root": str(root / "result/runtime/stock"),
                "optimized_root": str(root / "result/runtime/valid-window"),
                "stock_validation": {"valid": True, "source_state": "original"},
                "optimized_before_patch_validation": {"valid": True, "source_state": "original"},
                "optimized_validation": {"valid": True, "source_state": "patched"},
                "stock_manifest": {"vllm/a.py": "a"},
                "optimized_manifest": {"vllm/a.py": "a"},
                "differing_files": [],
            }
            probes = [
                {"source_root": runtime["stock_root"], "module_file": "vllm/__init__.py", "version": "0.26.0"},
                {"source_root": runtime["optimized_root"], "module_file": "vllm/__init__.py", "version": "0.26.0"},
            ]
            with (
                mock.patch.object(MODULE, "_preflight", side_effect=mutate_then_return),
                mock.patch.object(MODULE, "_create_runtime_snapshots", return_value=runtime),
                mock.patch.object(MODULE, "_import_probe", side_effect=probes),
                mock.patch.object(MODULE, "_verify_runtime_snapshots"),
                mock.patch.object(MODULE, "_gpu_processes", return_value=[]),
                mock.patch.object(MODULE, "_run_arm", return_value=(7, 0.1, {
                    "process_group_absent": True, "gpu_idle_verified": True,
                })),
            ):
                self.assertEqual(MODULE.execute(args), 2)
            staged = Path(args.output_root) / "frozen/workload_manifest.json"
            self.assertEqual(staged.read_bytes(), canonical_bytes)
            self.assertEqual(MODULE._sha256(staged), MODULE.FROZEN_WORKLOAD_SHA256)

    def test_failed_arm_aborts_without_retry_or_comparator(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = _args(root)
            preflight = {
                "python": args.python,
                "vllm_source_root": str(root / "site-packages"),
                "vllm_import": {"source_root": str(root / "site-packages"),
                                "module_file": "vllm/__init__.py", "version": "0.26.0"},
                "source": {"valid": True, "source_state": "original"},
                "exclusive_gpu_verified": True,
                "compute_processes": [],
            }
            runtime = {
                "stock_root": str(root / "result/runtime/stock"),
                "optimized_root": str(root / "result/runtime/valid-window"),
                "stock_validation": {"valid": True, "source_state": "original"},
                "optimized_before_patch_validation": {"valid": True, "source_state": "original"},
                "optimized_validation": {"valid": True, "source_state": "patched"},
                "stock_manifest": {"vllm/a.py": "a"},
                "optimized_manifest": {"vllm/a.py": "a"},
                "differing_files": [],
            }
            probes = [
                {"source_root": runtime["stock_root"], "module_file": "vllm/__init__.py", "version": "0.26.0"},
                {"source_root": runtime["optimized_root"], "module_file": "vllm/__init__.py", "version": "0.26.0"},
            ]

            def fake_arm(command, log_path, timeout, environment):
                log_path.write_bytes(b"failed")
                return 9, 0.1, {
                    "process_group_absent": True, "gpu_idle_verified": True,
                }

            with (
                mock.patch.object(MODULE, "_preflight", return_value=preflight),
                mock.patch.object(MODULE, "_create_runtime_snapshots", return_value=runtime),
                mock.patch.object(MODULE, "_import_probe", side_effect=probes),
                mock.patch.object(MODULE, "_verify_runtime_snapshots"),
                mock.patch.object(MODULE, "_gpu_processes", return_value=[]),
                mock.patch.object(MODULE, "_run_arm", side_effect=fake_arm),
                mock.patch.object(MODULE, "_run_comparator") as comparator,
            ):
                code = MODULE.execute(args)

            self.assertEqual(code, 2)
            campaign = Path(args.output_root)
            self.assertTrue((campaign / "CAMPAIGN_ABORTED.json").is_file())
            self.assertFalse((campaign / "CAMPAIGN_COMPLETE.json").exists())
            plan_path = campaign / "run_plan.json"
            self.assertTrue(plan_path.is_file(), (campaign / "CAMPAIGN_ABORTED.json").read_text())
            plan = json.loads(plan_path.read_text())
            manifest = campaign / "runtime-package-manifest.json"
            self.assertEqual(
                plan["frozen_input_sha256"]["runtime-package-manifest.json"],
                MODULE._sha256(manifest),
            )
            comparator.assert_not_called()

    def test_preflight_rejects_interpreter_source_root_mismatch(self) -> None:
        args = argparse.Namespace(
            python="/usr/bin/python3", vllm_source_root="/tmp/not-detected"
        )
        probe = {
            "source_root": "/tmp/detected", "module_file": "/tmp/detected/vllm/__init__.py",
            "version": "0.26.0",
        }
        with mock.patch.object(MODULE, "_import_probe", return_value=probe):
            with self.assertRaisesRegex(RuntimeError, "does not back interpreter"):
                MODULE._preflight(args)

    def test_model_cache_probe_fails_closed_on_wrong_revision(self) -> None:
        result = mock.Mock(returncode=0, stdout='{"snapshot_path":"/cache/wrong"}', stderr="")
        with (
            mock.patch.object(MODULE.subprocess, "run", return_value=result),
            mock.patch.object(MODULE.Path, "is_dir", return_value=True),
            self.assertRaisesRegex(RuntimeError, "wrong revision"),
        ):
            MODULE._model_cache_probe(
                Path("/python"), {"HF_HOME": MODULE.HF_HOME,
                                  "HF_HUB_CACHE": MODULE.HF_HUB_CACHE}
            )

    def test_model_cache_probe_rejects_external_path_with_correct_basename(self) -> None:
        external = f'/external/snapshots/{MODULE.REVISION}'
        result = mock.Mock(
            returncode=0,
            stdout=json.dumps({"snapshot_path": external}),
            stderr="",
        )
        with (
            mock.patch.object(MODULE.subprocess, "run", return_value=result),
            mock.patch.object(MODULE.Path, "is_dir", return_value=True),
            self.assertRaisesRegex(RuntimeError, "wrong revision"),
        ):
            MODULE._model_cache_probe(
                Path("/python"), {"HF_HOME": MODULE.HF_HOME,
                                  "HF_HUB_CACHE": MODULE.HF_HUB_CACHE}
            )

    def test_cleanup_escalates_when_leader_exits_before_descendants(self) -> None:
        process = mock.Mock()
        process.pid = 4321
        process.wait.return_value = 0
        with (
            mock.patch.object(MODULE, "_process_group_exists", side_effect=[True]),
            mock.patch.object(
                MODULE, "_wait_for_process_group_exit", side_effect=[False, True]
            ) as wait_group,
            mock.patch.object(MODULE.os, "killpg") as killpg,
            mock.patch.object(MODULE, "_wait_for_gpu_idle", return_value=[]) as gpu_idle,
        ):
            cleanup = MODULE._terminate_process_group(
                process, term_timeout=0.01, kill_timeout=0.01, gpu_idle_timeout=0.01
            )
        self.assertEqual(
            killpg.call_args_list,
            [mock.call(4321, MODULE.signal.SIGTERM), mock.call(4321, MODULE.signal.SIGKILL)],
        )
        self.assertEqual(
            wait_group.call_args_list,
            [mock.call(4321, 0.01, process), mock.call(4321, 0.01, process)],
        )
        gpu_idle.assert_called_once_with(0.01)
        self.assertTrue(cleanup["sent_sigkill"])
        self.assertTrue(cleanup["gpu_idle_verified"])

    def test_main_installs_hangup_and_terminate_handlers(self) -> None:
        args = argparse.Namespace(
            command="run", arm_timeout_seconds=1, comparison_timeout_seconds=1
        )
        with (
            mock.patch.object(MODULE, "build_parser") as parser,
            mock.patch.object(MODULE, "execute", return_value=0),
            mock.patch.object(MODULE.signal, "signal") as install,
            self.assertRaises(SystemExit) as exited,
        ):
            parser.return_value.parse_args.return_value = args
            MODULE.main()
        self.assertEqual(exited.exception.code, 0)
        self.assertIn(
            mock.call(MODULE.signal.SIGTERM, MODULE._raise_keyboard_interrupt),
            install.call_args_list,
        )
        self.assertIn(
            mock.call(MODULE.signal.SIGHUP, MODULE._raise_keyboard_interrupt),
            install.call_args_list,
        )


if __name__ == "__main__":
    unittest.main()
