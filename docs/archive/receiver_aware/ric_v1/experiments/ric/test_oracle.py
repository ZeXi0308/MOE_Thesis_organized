from __future__ import annotations

from dataclasses import replace
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
import json


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

from oracle import (  # noqa: E402
    MatchedWorld,
    OracleError,
    assert_information_monotonicity,
    brute_force_matched_pair,
    build_observation_history_nodes,
    empirical_cvar99_two_world,
    fixed_route_flowshop_position_completions,
    literal_fcfs_capacity_position_completions,
    solve_matched_pair,
    synthetic_matched_pair,
)
import run_oracle  # noqa: E402
from run_oracle import route_pair_from_world  # noqa: E402
import formal_provenance as provenance  # noqa: E402
import prepare_data  # noqa: E402
from ric.scenario import build_complete_fixture_world  # noqa: E402


HAS_SCIPY = importlib.util.find_spec("scipy") is not None


class MatchedWorldOracleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.pair = synthetic_matched_pair(
            "olmoe", 0, service_us=3.0, payload_bytes=4096
        )

    def test_sender_nonanticipativity_matched_worlds(self) -> None:
        blind = brute_force_matched_pair(self.pair, "B")
        self.assertEqual(len(set(blind.orders_by_world.values())), 1)
        self.assertEqual(blind.nonanticipativity_nodes, 1)

    def test_observation_history_builder_derives_information_nodes(self) -> None:
        self.assertEqual(
            len(set(build_observation_history_nodes(self.pair, "S").values())), 1
        )
        self.assertEqual(
            len(set(build_observation_history_nodes(self.pair, "B").values())), 1
        )
        self.assertEqual(
            len(set(build_observation_history_nodes(self.pair, "R0").values())), 2
        )
        self.assertEqual(
            len(set(build_observation_history_nodes(self.pair, "C").values())), 2
        )

    def test_starvation_bound_is_explicit_and_fail_closed(self) -> None:
        impossible = replace(
            self.pair,
            starvation_us=self.pair.tasks[0].sender_egress_us - 1e-6,
        )
        with self.assertRaisesRegex(OracleError, "BLOCKED_STARVATION_INFEASIBLE"):
            brute_force_matched_pair(impossible, "B")
        if HAS_SCIPY:
            with self.assertRaisesRegex(OracleError, "BLOCKED_STARVATION_INFEASIBLE"):
                solve_matched_pair(impossible, "B")

    def test_receiver_information_can_flip_same_sender_action(self) -> None:
        aware = brute_force_matched_pair(self.pair, "R0")
        first = [order[0] for order in aware.orders_by_world.values()]
        self.assertEqual(len(set(first)), 2)
        self.assertEqual(set(first), {task.task_id for task in self.pair.tasks})
        self.assertEqual(aware.violation_count, 0)

    @unittest.skipUnless(HAS_SCIPY, "scipy.optimize.milp unavailable")
    def test_milp_matches_independent_bruteforce(self) -> None:
        for level in ("B", "R0"):
            milp_result = solve_matched_pair(self.pair, level)
            brute_result = brute_force_matched_pair(self.pair, level)
            self.assertEqual(milp_result.violation_count, brute_result.violation_count)
            self.assertAlmostEqual(
                milp_result.empirical_cvar99_us,
                brute_result.empirical_cvar99_us,
            )
            self.assertAlmostEqual(
                milp_result.mean_closure_us,
                brute_result.mean_closure_us,
            )
            self.assertEqual(
                milp_result.position_completion_us,
                brute_result.position_completion_us,
            )

    def test_information_monotonicity(self) -> None:
        results = {
            level: brute_force_matched_pair(self.pair, level)
            for level in ("S", "B", "R0", "C")
        }
        assert_information_monotonicity(results)

    def test_scenario_rename_does_not_change_public_action(self) -> None:
        renamed = replace(
            self.pair,
            worlds=tuple(
                MatchedWorld(
                    world_name=f"renamed-{index}",
                    closing_task_id=world.closing_task_id,
                    hidden_join_fingerprint=world.hidden_join_fingerprint,
                )
                for index, world in enumerate(self.pair.worlds)
            ),
        )
        original = brute_force_matched_pair(self.pair, "B")
        changed = brute_force_matched_pair(renamed, "B")
        self.assertEqual(
            tuple(original.orders_by_world.values())[0],
            tuple(changed.orders_by_world.values())[0],
        )

    def test_path_mismatch_is_rejected(self) -> None:
        mismatched = replace(
            self.pair.tasks[1],
            shared_cut_resource="cut:adversarial-other-path",
        )
        with self.assertRaisesRegex(OracleError, "exact four-resource path"):
            replace(self.pair, tasks=(self.pair.tasks[0], mismatched))

    def test_receiver_mismatch_is_rejected(self) -> None:
        mismatched = replace(self.pair.tasks[1], receiver_rank=1)
        with self.assertRaisesRegex(OracleError, "exact receiver"):
            replace(self.pair, tasks=(self.pair.tasks[0], mismatched))

    def test_release_drift_is_rejected(self) -> None:
        drifted = replace(self.pair.tasks[1], release_us=1e-6)
        with self.assertRaisesRegex(OracleError, "common release_us=0"):
            replace(self.pair, tasks=(self.pair.tasks[0], drifted))

    def test_closing_state_bound_to_wrong_join_is_rejected(self) -> None:
        left, right = self.pair.worlds
        swapped_bindings = (
            replace(left, hidden_join_fingerprint=right.hidden_join_fingerprint),
            replace(right, hidden_join_fingerprint=left.hidden_join_fingerprint),
        )
        with self.assertRaisesRegex(OracleError, "wrong application join"):
            replace(self.pair, worlds=swapped_bindings)

    def test_three_stage_completion_matches_independent_resource_timeline(self) -> None:
        stages = (7.0, 2.0, 5.0)
        observed = fixed_route_flowshop_position_completions(stages, 4)

        # Independent explicit timeline: record every start/end and validate
        # both job precedence and no overlap on each of the three resources.
        resource_free = [0.0, 0.0, 0.0]
        timelines: list[list[tuple[float, float]]] = [[], [], []]
        expected = []
        for _job in range(4):
            predecessor_end = 0.0
            for stage, service in enumerate(stages):
                start = max(predecessor_end, resource_free[stage])
                end = start + service
                timelines[stage].append((start, end))
                resource_free[stage] = end
                predecessor_end = end
            expected.append(predecessor_end)
        for timeline in timelines:
            for previous, current in zip(timeline, timeline[1:]):
                self.assertGreaterEqual(current[0], previous[1])
        self.assertEqual(observed, tuple(expected))
        self.assertEqual(observed, (14.0, 21.0, 28.0, 35.0))

    @unittest.skipUnless(HAS_SCIPY, "scipy.optimize.linprog unavailable")
    def test_recurrence_matches_literal_fcfs_capacity_model(self) -> None:
        for stages, task_count in (
            ((7.0, 2.0, 5.0), 4),
            ((1.0, 10.0, 1.0), 2),
            ((3.25, 4.5, 2.75), 5),
        ):
            self.assertEqual(
                fixed_route_flowshop_position_completions(stages, task_count),
                literal_fcfs_capacity_position_completions(stages, task_count),
            )

    def test_oracle_closure_adds_once_only_join_combine(self) -> None:
        contribution = fixed_route_flowshop_position_completions(
            self.pair.tasks[0].stage_service_us, 2
        )
        result = brute_force_matched_pair(self.pair, "R0")
        combine = self.pair.tasks[0].join_combine_us
        self.assertEqual(
            result.position_completion_us,
            tuple(value + combine for value in contribution),
        )
        self.assertTrue(result.unique_optimal_first_action)
        self.assertTrue(
            all(
                len(actions) == 1
                for actions in result.optimal_first_actions_by_world.values()
            )
        )

    def test_two_world_empirical_cvar99_is_exact_maximum(self) -> None:
        self.assertEqual(empirical_cvar99_two_world((14.0, 21.0)), 21.0)
        blind = brute_force_matched_pair(self.pair, "B")
        self.assertEqual(
            blind.empirical_cvar99_us,
            max(blind.closure_by_world_us.values()),
        )

    def test_route_pair_is_exact_path_matched_and_release_zero(self) -> None:
        world = build_complete_fixture_world(
            model_key="fixture",
            model_revision="fixture/moe@ric-v1",
            top_k=2,
            num_experts=8,
            trace_index=0,
            seed=202607223001,
            payload_bytes=1024,
            closure_budget_us=200.0,
        )
        record = route_pair_from_world(
            world,
            pair_index=0,
            closure_budget_us=200.0,
            starvation_us=1000.0,
        )
        left, right = record.pair.tasks
        self.assertNotEqual(
            record.source_join_fingerprints[0],
            record.source_join_fingerprints[1],
        )
        self.assertEqual(left.sender_rank, right.sender_rank)
        self.assertEqual(left.receiver_rank, right.receiver_rank)
        self.assertEqual(left.stage_resources, right.stage_resources)
        self.assertEqual(left.stage_service_us, right.stage_service_us)
        self.assertEqual(
            left.receiver_combine_resource, right.receiver_combine_resource
        )
        self.assertEqual(left.join_combine_us, right.join_combine_us)
        self.assertEqual(left.payload_bytes, right.payload_bytes)
        self.assertEqual((left.release_us, right.release_us), (0.0, 0.0))


class OracleFormalBoundaryTests(unittest.TestCase):
    def _config(self) -> dict[str, object]:
        return {
            "schema_version": "ric-config-v1",
            "go_no_go": {
                "required_models": ["olmoe", "llmjp"],
                "required_main_cells": ["poisson_rho60", "ctmc_mmpp_rho85"],
            },
        }

    def _lock(
        self, *, config_sha: str, protocol_sha: str, g1_pass: bool = True
    ) -> dict[str, object]:
        models = ("olmoe", "llmjp")
        cells = ("poisson_rho60", "ctmc_mmpp_rho85")
        return provenance.add_self_hash(
            {
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
                "service_lut_metadata_sha256": {
                    model: "2" * 64 for model in models
                },
                "capability_probe_sha256": {model: "3" * 64 for model in models},
                "scenario_producer_signoff_sha256": {
                    model: "5" * 64 for model in models
                },
                "capability_producer_signoff_sha256": {
                    model: "6" * 64 for model in models
                },
                "signoff_sha256": "7" * 64,
                "g1_by_model": {model: g1_pass for model in models},
                "g1_pass": g1_pass,
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
        )

    def test_g1_false_hard_stops_before_scenario_or_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "config.json"
            protocol_path = root / "protocol.md"
            config_path.write_text(json.dumps(self._config()), encoding="utf-8")
            protocol_path.write_text("frozen\n", encoding="utf-8")
            lock_path = root / "lock.json"
            lock_path.write_text(
                json.dumps(
                    self._lock(
                        config_sha=provenance.sha256_file(config_path),
                        protocol_sha=provenance.sha256_file(protocol_path),
                        g1_pass=False,
                    )
                ),
                encoding="utf-8",
            )
            with mock.patch.object(
                run_oracle, "validate_frozen_formal_paths"
            ), mock.patch.object(
                run_oracle, "validate_formal_output_path"
            ), self.assertRaisesRegex(run_oracle.OracleRunnerError, "G1 did not pass"):
                run_oracle.run_oracle_pipeline(
                    scenario_dirs=(),
                    calibration_lock_path=lock_path,
                    output_dir=root / "oracle-output",
                    mode="formal",
                    config_path=config_path,
                    protocol_path=protocol_path,
                )
            self.assertFalse((root / "oracle-output").exists())

    def test_formal_signoff_is_full_chain_and_source_closure_bound(self) -> None:
        with tempfile.TemporaryDirectory(prefix=".ric-oracle-", dir=run_oracle.REPO_ROOT) as directory:
            root = Path(directory)
            config_path = root / "config.json"
            protocol_path = root / "protocol.md"
            lock_path = root / "lock.json"
            config_path.write_text("{}\n", encoding="utf-8")
            protocol_path.write_text("frozen\n", encoding="utf-8")
            lock = {"manifest_sha256": "1" * 64}
            lock_path.write_text(json.dumps(lock), encoding="utf-8")
            tree_hashes = {"olmoe": "2" * 64, "llmjp": "3" * 64}
            tree_file_hashes = {"olmoe": "4" * 64, "llmjp": "5" * 64}
            with mock.patch.object(
                run_oracle, "verify_phase4_signoff", return_value={"status": "SIGNED-OFF"}
            ) as verifier:
                run_oracle._require_formal_signoff(
                    root / "signoff.json",
                    config_path=config_path,
                    protocol_path=protocol_path,
                    calibration_lock_path=lock_path,
                    calibration_lock=lock,
                    scenario_tree_hashes=tree_hashes,
                    scenario_tree_file_hashes=tree_file_hashes,
                    scenario_producer_signoff_sha256={
                        "olmoe": "6" * 64,
                        "llmjp": "7" * 64,
                    },
                )
            kwargs = verifier.call_args.kwargs
            self.assertEqual(
                tuple(kwargs["required_source_paths"]), run_oracle.ORACLE_SOURCE_PATHS
            )
            self.assertEqual(kwargs["expected_fields"]["stage"], "oracle")
            self.assertEqual(
                kwargs["expected_fields"]["scenario_tree_sha256"], tree_hashes
            )
            self.assertEqual(
                kwargs["expected_fields"]["calibration_lock_file_sha256"],
                provenance.sha256_file(lock_path),
            )

    def test_fake_oracle_signoff_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix=".ric-oracle-", dir=run_oracle.REPO_ROOT) as directory:
            root = Path(directory)
            for name, content in (
                ("config.json", "{}\n"),
                ("protocol.md", "frozen\n"),
                ("lock.json", "{}\n"),
                ("signoff.json", '{"status":"SIGNED-OFF","open_p0":0}\n'),
            ):
                (root / name).write_text(content, encoding="utf-8")
            with self.assertRaisesRegex(
                run_oracle.OracleRunnerError, "schema mismatch"
            ):
                run_oracle._require_formal_signoff(
                    root / "signoff.json",
                    config_path=root / "config.json",
                    protocol_path=root / "protocol.md",
                    calibration_lock_path=root / "lock.json",
                    calibration_lock={"manifest_sha256": "1" * 64},
                    scenario_tree_hashes={"olmoe": "2" * 64, "llmjp": "3" * 64},
                    scenario_tree_file_hashes={
                        "olmoe": "4" * 64,
                        "llmjp": "5" * 64,
                    },
                    scenario_producer_signoff_sha256={
                        "olmoe": "6" * 64,
                        "llmjp": "7" * 64,
                    },
                )

    def test_existing_output_directory_hard_stops(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(
                run_oracle.OracleRunnerError, "refusing to overwrite"
            ):
                run_oracle.run_oracle_pipeline(
                    scenario_dirs=(),
                    calibration_lock_path=root / "missing-lock.json",
                    output_dir=root,
                    mode="formal",
                    config_path=root / "missing-config.json",
                    protocol_path=root / "missing-protocol.md",
                )


if __name__ == "__main__":
    unittest.main()
