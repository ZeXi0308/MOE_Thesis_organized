from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np


MODULE_PATH = Path(__file__).with_name("analyze_route_regrouping_oracle.py")
SPEC = importlib.util.spec_from_file_location("route_regrouping_oracle", MODULE_PATH)
assert SPEC and SPEC.loader
oracle = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(oracle)


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _reseal(run: Path) -> None:
    seal = json.loads((run / "RUN_COMPLETE.json").read_text())
    seal.update({
        "config_sha256": _hash(run / "config.json"),
        "raw_sha256": _hash(run / "batches.jsonl"),
        "summary_sha256": _hash(run / "summary.json"),
        "artifact_hashes_sha256": _hash(run / "ARTIFACT_HASHES.json"),
    })
    (run / "RUN_COMPLETE.json").write_text(json.dumps(seal, sort_keys=True))


def _write_bundle(root: Path, repeat: int) -> Path:
    run = root / f"on-r{repeat}"
    (run / "routes").mkdir(parents=True)
    (run / "inputs").mkdir()
    (run / "workload_manifest.json").write_text(json.dumps({"requests": []}))
    claim_ceiling = "NATIVE_OFFLINE_FIXED_BATCH_MEASUREMENT_ONLY"
    config = {
        "model": "test/olmoe",
        "revision": "fixed",
        "dtype": "bfloat16",
        "batch_sizes": [16],
        "prompt_lengths": [32],
        "output_tokens": 4,
        "groups": 6,
        "within_process_repeats": 1,
        "process_repeat": repeat,
        "seed": 7,
        "order_seed": 11 + repeat,
        "max_model_len": 64,
        "max_num_seqs": 16,
        "max_num_batched_tokens": 512,
        "gpu_memory_utilization": 0.8,
        "runtime_patch_id": "test-patch",
        "enforce_eager": True,
        "require_exclusive_gpu": False,
        "capture_routes": True,
        "claim_ceiling": claim_ceiling,
        "workload_manifest_sha256": _hash(run / "workload_manifest.json"),
        "probe_script_sha256": "probe-sha",
        "runtime_identity": {"vllm": "test", "gpu": "test"},
        "model_shape": {"num_experts": 8, "num_layers": 2, "top_k": 2},
    }
    (run / "config.json").write_text(json.dumps(config, sort_keys=True))
    (run / "environment.json").write_text(
        json.dumps({"vllm": "test", "gpu": "test"})
    )
    (run / "model_shape.json").write_text(json.dumps(config["model_shape"]))
    artifact_hashes = {
        name: _hash(run / name)
        for name in ("environment.json", "model_shape.json", "workload_manifest.json")
    }
    rows = []
    for group in range(6):
        batch_id = f"r{repeat:02d}-p32-b16-g{group:02d}-w00"
        prompts = np.stack([
            np.full(32, group * 16 + slot, dtype=np.int32) for slot in range(16)
        ])
        # Original groups mix six request types.  This gives both balance and
        # coalescing policies non-trivial but deterministic structural work.
        routes = np.empty((16, 3, 2, 2), dtype=np.uint8)
        for slot in range(16):
            request_type = (group * 16 + slot) % 6
            for step in range(3):
                for layer in range(2):
                    first = (request_type + step + layer) % 8
                    routes[slot, step, layer] = [first, (first + 1) % 8]
        input_path = run / "inputs" / f"{batch_id}.npz"
        route_path = run / "routes" / f"{batch_id}.npz"
        np.savez_compressed(input_path, prompt_token_ids=prompts)
        np.savez_compressed(route_path, routes=routes)
        input_relative = str(input_path.relative_to(run))
        route_relative = str(route_path.relative_to(run))
        artifact_hashes[input_relative] = _hash(input_path)
        artifact_hashes[route_relative] = _hash(route_path)
        rows.append({
            "batch_id": batch_id,
            "execution_order": len(rows),
            "batch_size": 16,
            "prompt_length": 32,
            "group": group,
            "within_process_repeat": 0,
            "process_repeat": repeat,
            "prompt_token_ids_sha256": _json_hash(prompts.tolist()),
            "input_artifact": input_relative,
            "input_artifact_sha256": _hash(input_path),
            "route_artifact": route_relative,
            "route_artifact_sha256": _hash(route_path),
            "request_metrics": [
                {
                    "generated_tokens": 4,
                    "token_ids": [group, slot, repeat, 0],
                }
                for slot in range(16)
            ],
            "timing": {
                "wall_ms": 10.0 + group,
                "request_tpot_p95_ms": 2.0 + group * 0.1,
            },
            "route": {
                "batch_size": 16,
                "decode_route_steps": 3,
                "num_layers": 2,
                "top_k": 2,
                "num_experts": 8,
                "total_assignments": int(routes.size),
            },
        })
    raw = run / "batches.jsonl"
    raw.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    (run / "summary.json").write_text(json.dumps({
        "schema": "vllm-native-route-shape-probe-v1",
        "status": "COMPLETE",
        "claim_ceiling": claim_ceiling,
        "capture_routes": True,
        "record_count": len(rows),
        "cell_summaries": [
            {"prompt_length": 32, "batch_size": 16, "samples": len(rows)}
        ],
    }, sort_keys=True))
    (run / "ARTIFACT_HASHES.json").write_text(json.dumps(artifact_hashes, sort_keys=True))
    (run / "RUN_COMPLETE.json").write_text(json.dumps({
        "status": "RUN_COMPLETE",
        "config_sha256": _hash(run / "config.json"),
        "raw_sha256": _hash(raw),
        "summary_sha256": _hash(run / "summary.json"),
        "artifact_hashes_sha256": _hash(run / "ARTIFACT_HASHES.json"),
    }, sort_keys=True))
    return run


class RouteRegroupingOracleTest(unittest.TestCase):
    def test_partition_contract_rejects_duplicate_request(self) -> None:
        partition = [list(range(group * 16, (group + 1) * 16)) for group in range(6)]
        partition[-1][-1] = 0
        with self.assertRaisesRegex(ValueError, "exactly once"):
            oracle._validate_partition(partition, 96)

    def test_future_local_search_is_not_worse_than_strongest_simple(self) -> None:
        counts = np.zeros((96, 2, 1, 8), dtype=np.int16)
        # Six original batches are each hot on one expert; the pool is balanced.
        for request in range(96):
            counts[request, :, 0, request // 16] = 1
        policies, metadata = oracle._policy_partitions(counts, 0, 0, 32)
        current = counts[:, 0]
        search_objective = oracle._balance_objective(
            oracle._loads(current, policies["future_route_local_search"])
        )
        baseline_objective = oracle._balance_objective(
            oracle._loads(current, policies[metadata["strongest_simple_balance_baseline"]])
        )
        original_objective = oracle._balance_objective(
            oracle._loads(current, policies["original"])
        )
        self.assertLessEqual(search_objective, baseline_objective)
        self.assertLess(search_objective, original_objective)
        self.assertTrue(metadata["balance_local_search"]["global_optimality_claimed"] is False)

    def test_history_uses_tminus1_and_step_zero_uses_original(self) -> None:
        counts = np.zeros((96, 2, 1, 8), dtype=np.int16)
        for request in range(96):
            expert = request // 16
            counts[request, 0, 0, expert] = 1
            counts[request, 1, 0, expert] = 1
        step_zero, zero_meta = oracle._policy_partitions(counts, 0, 0, 32)
        step_one, one_meta = oracle._policy_partitions(counts, 1, 0, 32)
        self.assertEqual(
            oracle._partition_signature(step_zero["history_greedy_tminus1"]),
            oracle._partition_signature(step_zero["original"]),
        )
        self.assertEqual(zero_meta["history_step_zero_fallback"], "original")
        self.assertEqual(one_meta["history_signal_step"], 0)
        history = oracle._balance_objective(
            oracle._loads(counts[:, 1], step_one["history_greedy_tminus1"])
        )
        original = oracle._balance_objective(
            oracle._loads(counts[:, 1], step_one["original"])
        )
        self.assertLess(history, original)

    def test_history_partition_cannot_read_current_step_routes(self) -> None:
        counts = np.zeros((96, 2, 1, 8), dtype=np.int16)
        for request in range(96):
            counts[request, 0, 0, request // 16] = 1
            counts[request, 1, 0, request % 8] = 1
        changed_current = counts.copy()
        changed_current[:, 1] = np.roll(changed_current[:, 1], 3, axis=-1)
        original_policy, _ = oracle._policy_partitions(counts, 1, 0, 32)
        changed_policy, _ = oracle._policy_partitions(changed_current, 1, 0, 32)
        self.assertEqual(
            oracle._partition_signature(original_policy["history_greedy_tminus1"]),
            oracle._partition_signature(changed_policy["history_greedy_tminus1"]),
        )

    def test_working_set_coalesce_is_a_separate_opposite_objective(self) -> None:
        counts = np.zeros((96, 1, 1, 8), dtype=np.int16)
        for request in range(96):
            counts[request, 0, 0, request % 6] = 1
        policies, _ = oracle._policy_partitions(counts, 0, 0, 32)
        original = oracle._partition_metrics(
            oracle._loads(counts[:, 0], policies["original"]), 1, 8
        )
        coalesced = oracle._partition_metrics(
            oracle._loads(counts[:, 0], policies["working_set_coalesce"]), 1, 8
        )
        self.assertLess(
            coalesced["mean_layer_active_experts"], original["mean_layer_active_experts"]
        )
        self.assertGreater(
            coalesced["mean_pairwise_route_overlap_fraction"],
            original["mean_pairwise_route_overlap_fraction"],
        )

    def test_two_sealed_repeats_produce_structural_only_report(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundles = [_write_bundle(root, repeat) for repeat in range(2)]
            report = oracle.analyze_bundles(bundles)

        self.assertEqual(report["status"], "COMPLETE")
        self.assertEqual(
            report["claim_ceiling"], "STRUCTURAL_FIXED_TRACE_DIAGNOSTIC_ONLY"
        )
        self.assertEqual(
            len(report["fixed_design"]["predeclared_for_v3_reanalysis_static_shuffle_seeds"]),
            8,
        )
        self.assertEqual(report["process_repeats"], [0, 1])
        self.assertEqual(report["trajectory_count"], 2)
        self.assertEqual(report["diagnostic_step_cell_count"], 6)
        self.assertEqual(len(report["diagnostic_step_cells"]), 6)
        transfer = report["step_level_diagnostics"][
            "canonical_partition_transfer_vs_original_diagnostic"
        ]
        self.assertEqual(transfer["canonical_process_repeat"], 0)
        self.assertEqual(transfer["holdout_process_repeats"], [1])
        self.assertIn(
            report["trajectory_analysis"]["trajectories"][0][
                "strongest_route_blind_balance_baseline"
            ],
            oracle.ROUTE_BLIND_POLICIES,
        )
        trajectory = report["trajectory_analysis"]["trajectories"][0]
        strongest = trajectory["strongest_route_blind_balance_baseline"]
        expected = oracle._effect(
            trajectory["policies"][strongest]["metrics"],
            trajectory["policies"]["history_greedy_tminus1"]["metrics"],
        )
        self.assertEqual(
            trajectory["effects_vs_strongest_route_blind"]["history_greedy_tminus1"],
            expected,
        )
        self.assertNotIn(
            "history_vs_original",
            report["trajectory_analysis"]["history_vs_strongest_route_blind"],
        )
        self.assertIn("not six independent", report["trajectory_analysis"]["independence_warning"])
        self.assertTrue(all(
            policy["validation"]["preserves_every_request_exactly_once"]
            for cell in report["diagnostic_step_cells"]
            for policy in cell["policies"].values()
        ))
        self.assertIn("no TPOT", " ".join(report["anti_claims"]))

    def test_route_blind_ladder_is_static_across_steps_and_repeats(self) -> None:
        counts = np.zeros((96, 2, 1, 8), dtype=np.int16)
        for request in range(96):
            counts[request, :, 0, request % 8] = 1
        keys = [f"request-{index}" for index in range(96)]
        first, _ = oracle._policy_partitions(counts, 0, 0, 32, keys)
        later, _ = oracle._policy_partitions(counts, 1, 9, 32, keys)
        self.assertGreaterEqual(len(oracle.ROUTE_BLIND_POLICIES), 11)
        for policy in oracle.ROUTE_BLIND_POLICIES:
            self.assertEqual(
                oracle._partition_signature(first[policy]),
                oracle._partition_signature(later[policy]),
            )

    def test_one_repeat_is_insufficient_and_tamper_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = _write_bundle(root, 0)
            insufficient = oracle.analyze_bundles([bundle])
            self.assertEqual(insufficient["status"], "INSUFFICIENT_EVIDENCE")
            with (bundle / "batches.jsonl").open("a") as raw:
                raw.write("{}\n")
            invalid = oracle.analyze_bundles([bundle, bundle])

        self.assertEqual(invalid["status"], "INVALID_INPUT")
        self.assertTrue(any("seal_hash:raw_sha256" in error for error in invalid["validation_errors"]))

    def test_resealed_row_provenance_and_runtime_drift_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            left, right = _write_bundle(root, 0), _write_bundle(root, 1)
            rows = [json.loads(line) for line in (left / "batches.jsonl").read_text().splitlines()]
            rows[0]["process_repeat"] = 9
            (left / "batches.jsonl").write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
            )
            _reseal(left)
            provenance = oracle.analyze_bundles([left, right])
            self.assertEqual(provenance["status"], "INVALID_INPUT")
            self.assertTrue(
                any("process_repeat drift" in error for error in provenance["validation_errors"])
            )

            left = _write_bundle(root / "fresh", 0)
            right = _write_bundle(root / "fresh", 1)
            config = json.loads((right / "config.json").read_text())
            config["max_model_len"] = 999
            (right / "config.json").write_text(json.dumps(config, sort_keys=True))
            _reseal(right)
            drift = oracle.analyze_bundles([left, right])
            self.assertEqual(drift["status"], "INVALID_INPUT")
            self.assertTrue(any("config_drift" in error for error in drift["validation_errors"]))


if __name__ == "__main__":
    unittest.main()
