"""CPU fixtures only; all generated records/artifacts live in a temporary tree."""
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import prepare_compiled_controller_evaluation as prep


class CompiledPreparationTests(unittest.TestCase):
    def test_eager_and_resource_changes_are_not_compiled_inputs(self):
        plan = prep.read(prep.ROOT / "configs/olmoe_ngram_compiled_controller_evaluation.json")
        prep.check_config(plan, calibration=False)
        for update in ({"enforce_eager": True}, {"native_route_return": True},
                       {"max_num_seqs": 32}, {"input_offset": 0}):
            with self.subTest(update=update), self.assertRaises(ValueError):
                prep.check_config({**plan, **update}, calibration=False)

    def test_real_builders_and_zero_policies_remain_separate(self):
        original_root = prep.ROOT
        with tempfile.TemporaryDirectory(prefix="compiled_prep_cpu_fixture_") as folder:
            root = Path(folder)
            (root / "code").mkdir()
            (root / "configs").mkdir()
            for name in (*prep.CRITICAL_SOURCE, "build_calibration.py", "plot_results.py"):
                shutil.copyfile(original_root / "code" / name, root / "code" / name)
            shutil.copyfile(original_root / "configs/controller_protocol.json", root / "configs/controller_protocol.json")
            workload = {"records": [{"source_index": i, "prompt_token_ids": [1]} for i in range(200)]}
            prep.write(root / "configs/workload.json", workload)
            evaluation = prep.read(original_root / "configs/olmoe_ngram_compiled_controller_evaluation.json")
            evaluation["workload"] = "configs/workload.json"
            config = root / "configs/evaluation.json"
            prep.write(config, evaluation)
            for calibration in (False, True):
                name = "calibration" if calibration else "native"
                plan = {**evaluation, "input_offset": 0, "gammas": prep.GAMMAS,
                    "batches": prep.BATCHES if calibration else prep.NATIVE_BATCHES,
                    "repeats": 1 if calibration else 2, "output_tokens": 64 if calibration else 128}
                if not calibration:
                    plan.pop("warmup_offset")  # Preserve the actual historical default80.
                run = root / "runs" / name
                (run / "source").mkdir(parents=True)
                source_hashes = {}
                for filename in prep.CRITICAL_SOURCE:
                    shutil.copyfile(root / "code" / filename, run / "source" / filename)
                    source_hashes[filename] = prep.sha(run / "source" / filename)
                prep.write(run / "plan.json", plan)
                prep.write(run / "status.json", {"status": "complete",
                    "cells": [{"gamma": gamma, "returncode": 0} for gamma in prep.GAMMAS]})
                for gamma in prep.GAMMAS:
                    cell = run / f"gamma{gamma}"
                    cell.mkdir()
                    budget, mode = (7, "static") if calibration else (gamma, "native")
                    prep.write(cell / "status.json", {"status": "complete"})
                    prep.write(cell / "configuration.json", {"plan": plan,
                        "args": {"mode": mode, "gamma": gamma, "proposal_budget": budget, "skip_trace": True},
                        "workload_sha256": prep.sha(root / plan["workload"])})
                    sig = prep.domain_signature(plan)
                    engine = {key: sig[key] for key in prep.ENGINE_FIELDS}
                    engine.update(async_scheduling=False, enable_prefix_caching=False,
                                  speculative_config=prep.expected_spec(plan, budget))
                    prep.write(cell / "engine_args.json", engine)
                    prep.write(cell / "environment.json", {"vllm": "0.26.0", "torch": "synthetic_cpu_fixture",
                        "cuda": "synthetic_cpu_fixture", "gpu": "synthetic_cpu_fixture", "model_runner_version": "v1_explicit",
                        "VLLM_USE_V2_MODEL_RUNNER": "0", "source_sha256": source_hashes})
                    prep.write(cell / "evaluation_initial_state.json", {"input_offset": 0,
                        "warmup_offset": plan.get("warmup_offset", 80), "shape_warmup_offset": 0,
                        "acceptance_snapshot_before": prep.AcceptanceEWMA().snapshot()})
                    for repeat in range(plan["repeats"]):
                        for batch in plan["batches"]:
                            label = f"perf-r{repeat}-b{batch}"
                            episodes = []
                            for offset in range(0, 64, batch):
                                wave = f"{label}-w{offset // batch}"
                                ids = [f"req{i}" for i in range(offset, offset + batch)]
                                requests = [{"source_index": i, "finished": True,
                                    "output_token_ids": [1] * plan["output_tokens"],
                                    "effective_tpot_ms": 1 + gamma + repeat / 10,
                                    "chunks": [{"elapsed_s": 2}]} for i in range(offset, offset + batch)]
                                steps = [{"step": index, "episode": wave, "phase": "decode", "gamma": gamma,
                                    "decode_batch": batch, "scheduled_tokens": batch * (gamma + 1),
                                    "scheduled_decode_request_ids": ids, "proposal_budget_next": budget,
                                    "proposed_before_by_request": {rid: budget for rid in ids},
                                    "scheduled_drafts_by_request": {rid: gamma for rid in ids},
                                    "accepted_by_request": {rid: 0 for rid in ids},
                                    "ewma_censored_requests": ids if index == 1 else []} for index in (0, 1)]
                                episodes.append({"label": wave, "requests": requests, "plugin": {"steps": steps},
                                    "host_steps": [{"index": index, "elapsed_s": index + 1,
                                                    "host_step_ms": 2 + gamma} for index in (0, 1)]})
                            prep.write(cell / f"{label}.json", {"label": label, "model": plan["model"],
                                "draft_source": plan["draft_source"], "batch": batch, "gamma": gamma,
                                "proposal_budget": budget, "mode": mode, "all_finished": True,
                                "measurement_kind": "fixed_length_generation", "episodes": episodes,
                                "wall_s": 2, "output_tokens": 64 * plan["output_tokens"]})
            with patch.object(prep, "ROOT", root):
                output = root / "runs/preparation"
                manifest = prep.prepare(root / "runs/calibration", root / "runs/native", output, config)
                selection = prep.read(manifest["selection"])
                self.assertEqual(selection["static"]["selected_gamma"], 0)
                self.assertEqual(selection["native_threshold"]["disable_above_batch"], 0)
                for role in ("native_unrestricted", "native_threshold"):
                    self.assertEqual(selection[role]["table"], [[1, 64, 0]])
                commands = {item["role"]: item for item in manifest["commands"]}
                self.assertEqual(len({item["run"] for item in commands.values()}), 5)
                for role in ("ar", "static", "native_unrestricted", "native_threshold", "adaptive"):
                    argv = commands[role]["argv"]
                    self.assertEqual(argv[argv.index("--proposal-budget") + 1], "0" if role in ("ar", "static") else "7")
                    self.assertIn("--skip-trace", argv)
                self.assertEqual(len(prep.read(manifest["calibration_table"])["records"]), 32)
                self.assertEqual(manifest["selection_evidence"]["validated_performance_runs"], 112)
                self.assertEqual(manifest["selection_evidence"]["selection_performance_runs"], 64)
                self.assertEqual(manifest["selection_evidence"]["general_warmup_record_range"], [80, 143])
                self.assertEqual(manifest["selection_evidence"]["warmup_overlap_heldout_source_indices"], list(range(80, 128)))
                self.assertEqual(manifest["calibration_evidence"]["warmup_overlap_heldout_source_indices"], [])
                self.assertEqual(manifest["calibration_evidence"]["regime"]["execution_mode"], "compiled")
                with self.assertRaises(FileExistsError):
                    prep.prepare(root / "runs/calibration", root / "runs/native", output, config)
                # An incomplete campaign or wrong production budget must not emit commands.
                status_path = root / "runs/native/status.json"
                complete = prep.read(status_path)
                prep.write(status_path, {**complete, "status": "running"})
                refused = root / "runs/refused"
                with self.assertRaisesRegex(ValueError, "complete"):
                    prep.prepare(root / "runs/calibration", root / "runs/native", refused, config)
                self.assertFalse(refused.exists())
                prep.write(status_path, complete)
                path = root / "runs/calibration/gamma0/perf-r0-b1.json"
                prep.write(path, {**prep.read(path), "proposal_budget": 0})
                with self.assertRaisesRegex(ValueError, "budget/mode"):
                    prep.prepare(root / "runs/calibration", root / "runs/native", refused, config)
                self.assertFalse(refused.exists())


if __name__ == "__main__":
    unittest.main()
