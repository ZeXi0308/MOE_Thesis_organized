"""CPU fixtures are temporary and never written into measured runs/."""
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import compare_reduced_evaluations as compare
from plot_results import resolve_regime, _context


class ReducedComparisonTests(unittest.TestCase):
    def write(self, path, obj):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(obj))
        return path

    def fixture(self, root):
        real = compare.PROJECT
        plan = json.loads((real / "configs/granite3b_ngram_reduced_evaluation.json").read_text())
        plan["workload"] = "configs/workload.json"
        workload = self.write(root / plan["workload"], {"records": [{"source_index": 1000 + i} for i in range(128)]})
        plan.pop("workload_sha256", None)
        config_path = self.write(root / "configs/evaluation.json", plan)
        protocol = json.loads((real / "configs/reduced_protocol.json").read_text())
        protocol_path = self.write(root / "configs/protocol.json", protocol)
        prep = root / "runs/preparation"
        env = {"vllm": "0.26.0", "torch": "CPU_fixture", "cuda": "CPU_fixture", "gpu": "CPU_fixture",
               "model_runner_version": "v1_explicit", "VLLM_USE_V2_MODEL_RUNNER": "0"}
        source_hashes = {name: compare.sha(real / "code" / name) for name in compare.RUNTIME_SOURCE}
        behavior = {name: compare.behavior_sha(real / "code" / name) for name in compare.RUNTIME_SOURCE}
        sig = compare.domain_signature(plan)
        engine = {k: sig[k] for k in ("model", "dtype", "quantization", "kv_cache_memory_bytes", "max_model_len",
                                    "max_num_seqs", "max_num_batched_tokens", "enforce_eager", "kernel_config")}
        engine.update(async_scheduling=False, enable_return_routed_experts=False)
        config = {"plan": plan, "workload_sha256": compare.sha(workload)}
        temp = root / "context"
        self.write(temp / "configuration.json", config)
        self.write(temp / "engine_args.json", engine)
        self.write(temp / "environment.json", env)
        record_context = {"model": plan["model"], "draft_source": plan["draft_source"], "measurement_kind": "fixed_length_generation"}
        regime = resolve_regime(record_context, _context(temp / "dummy.json"))
        dev_requests = [{"source_index": 1000 + i} for i in range(64)]
        dev_path = self.write(root / "runs/development/perf.json", {"requests": dev_requests})
        source = {"source": str(dev_path), "sha256": compare.sha(dev_path)}
        evidence = {"domain": sig, "regime": regime, "environment": env, "runtime_behavior": behavior,
            "workload_sha256": compare.sha(workload), "source_files": [source],
            "development_source_indices": list(range(1000, 1064)), "heldout_source_indices": list(range(1064, 1128)),
            "warmup_offset": 0, "shape_warmup_offset": 0}
        table = {"metadata": regime, "records": [{"batch": b, "gamma": g, "proposal_budget": 7,
            "total_step_ms": 3, "eligible_steps": 2, "evidence_kind": "measured_complete_step_cost", "source": str(dev_path)}
            for b in plan["batches"] for g in plan["candidate_gammas"]]}
        table_path = self.write(prep / "calibration/calibration.json", table)
        selection = {"status": "selected_pending_evaluation", "regime": regime,
            "static": {"selected_gamma": 3}, "normalized_batch_weights": {str(b): .25 for b in plan["batches"]},
            "source_provenance": [{**source, "sample_source_indices": list(range(1000, 1064))}]}
        for role in ("native_unrestricted", "native_threshold"):
            selection[role] = {"runner": {"gamma": 7, "proposal_budget": 7}, "table": [[1, 64, 3]]}
        selection_path = self.write(prep / "selection/selection.json", selection)
        commands = []
        for role in compare.ROLES:
            run = root / "runs" / ("evaluation_" + role)
            gamma = 0 if role == "ar" else 3 if role == "static" else 7
            mode = "adaptive" if role == "adaptive" else "native"
            folder = run / f"gamma{gamma}"
            self.write(run / "status.json", {"status": "complete", "cells": [{"gamma": gamma, "returncode": 0}]})
            self.write(run / "plan.json", plan)
            self.write(folder / "status.json", {"status": "complete"})
            args = {"mode": mode, "gamma": gamma, "proposal_budget": gamma}
            cell_engine = {**engine, "speculative_config": compare.expected_spec(plan, gamma)}
            if role.startswith("native_"):
                args["native_table"] = str(run / "native_table.json")
                self.write(run / "native_table.json", selection[role]["table"])
                cell_engine["speculative_config"]["num_speculative_tokens_per_batch_size"] = selection[role]["table"]
            if role == "adaptive":
                args["calibration"] = str(run / "calibration.json")
                self.write(run / "calibration.json", table)
            self.write(folder / "configuration.json", {**config, "args": args})
            self.write(folder / "engine_args.json", cell_engine)
            self.write(folder / "environment.json", {**env, "source_sha256": source_hashes})
            self.write(folder / "evaluation_initial_state.json", {"input_offset": 64, "warmup_offset": 0, "shape_warmup_offset": 0})
            (run / "source").mkdir()
            for name in compare.RUNTIME_SOURCE:
                shutil.copyfile(real / "code" / name, run / "source" / name)
            for repeat in range(2):
                for batch in plan["batches"]:
                    label = f"perf-r{repeat}-b{batch}"
                    step = {"step": 0, "episode": label, "phase": "decode", "scheduled_tokens": batch,
                        "gamma": 3 if role == "adaptive" else gamma, "control_ns": 5000,
                        "acceptance_update_ns": 2000, "schedule_instrumentation_ns": 8000,
                        "drafts_available_before": 7, "truncated_by_controller": 4 if role == "adaptive" else 0,
                        "scheduled_drafts": 3, "decision": {"decision_cpu_us": 3} if role == "adaptive" else {}}
                    requests = [{"source_index": 1064 + i, "finished": True, "output_token_ids": [1] * 128,
                                 "effective_tpot_ms": 4 + repeat * 2} for i in range(64)]
                    obj = {**record_context, "label": label, "batch": batch, "gamma": gamma, "proposal_budget": gamma,
                        "mode": mode, "wall_s": 10 + repeat * 2, "population_wall_s": 11 + repeat * 2,
                        "population_inter_episode_gap_s": 1, "output_tokens": 64 * 128,
                        "energy_j": 20 + repeat * 2, "all_finished": True,
                        "episodes": [{"label": label, "requests": requests,
                            "host_steps": [{"index": 0, "host_step_ms": 100}], "plugin": {"steps": [step]}}]}
                    self.write(folder / (label + ".json"), obj)
            commands.append({"role": role, "run": run.name, "config": str(config_path), "expected_performance_run_records": 8})
        self.write(prep / "evaluation_commands.json", {"status": "ready_for_serialized_evaluation", "domain": plan["preparation_domain"],
            "candidate_gammas": [0, 3, 7], "commands": commands, "evaluation_config": str(config_path),
            "evaluation_config_sha256": compare.sha(config_path), "protocol": str(protocol_path), "protocol_sha256": compare.sha(protocol_path),
            "selection": str(selection_path), "selection_sha256": compare.sha(selection_path),
            "calibration_table": str(table_path), "calibration_sha256": compare.sha(table_path), "calibration_cells": 12,
            "calibration_evidence": evidence, "selection_evidence": evidence})
        return prep

    def test_missing_preparation_is_pending_not_synthetic(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = compare.analyze(root / "missing", root / "out")
            self.assertEqual(result["status"], "pending")
            self.assertEqual(result["runs"], [])
            self.assertEqual(result["F3"], [])

    def test_full_manifest_actual_roles_and_missing_repeat(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prep = self.fixture(root)
            with patch.object(compare, "PROJECT", root):
                result = compare.analyze(prep, root / "complete")
                self.assertEqual(result["status"], "measured_comparison_complete", result["issues"])
                self.assertEqual(len(result["runs"]), 40)
                self.assertEqual(result["F3"][0]["effective_tpot_ms_raw"], [4, 6])
                self.assertEqual(result["F3"][0]["effective_tpot_ms_median"], 5)
                self.assertIsNone(result["F4"][0]["population_counter_energy_j_median"])
                self.assertIsNone(result["F6"][0]["policy_choose_cpu_us_mean_median"])
                (root / "runs/evaluation_adaptive/gamma7/perf-r1-b4.json").unlink()
                partial = compare.analyze(prep, root / "partial")
                self.assertEqual(partial["status"], "pending")
                cell = next(r for r in partial["F3"] if r["role"] == "adaptive" and r["batch"] == 4)
                self.assertEqual(cell["effective_tpot_ms_raw"], [4])
                self.assertIsNone(cell["effective_tpot_ms_median"])
                self.assertEqual(len(partial["runs"]), 39)

    def test_wrong_source_and_population_are_explicitly_invalid(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prep = self.fixture(root)
            path = root / "runs/evaluation_static/gamma3/perf-r0-b1.json"
            obj = compare.read(path)
            obj["episodes"][0]["requests"][0]["source_index"] = 1000
            self.write(path, obj)
            engine_path = root / "runs/evaluation_adaptive/gamma7/engine_args.json"
            obj = compare.read(engine_path)
            obj["speculative_config"]["model"] = "unexpected_draft"
            self.write(engine_path, obj)
            with patch.object(compare, "PROJECT", root):
                result = compare.analyze(prep, root / "invalid")
            self.assertEqual(result["status"], "invalid_evidence")
            self.assertEqual(len(result["runs"]), 31)
            self.assertTrue(any("held-out workload slice" in r["reason"] for r in result["issues"]))
            self.assertTrue(any("draft method/model/budget" in r["reason"] for r in result["issues"]))


if __name__ == "__main__":
    unittest.main()
