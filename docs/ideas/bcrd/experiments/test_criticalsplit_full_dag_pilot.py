from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

try:
    from . import criticalsplit_full_dag_pilot as critical
    from . import frontiercredit_full_dag_pilot as frontier
except ImportError:
    import criticalsplit_full_dag_pilot as critical  # type: ignore
    import frontiercredit_full_dag_pilot as frontier  # type: ignore


def _find_state(
    *,
    require_split: bool = False,
    require_sham_difference: bool = False,
) -> tuple[frontier.Episode, frontier.SimulationState]:
    catalog = frontier.make_service_catalog()
    for episode in frontier.generate_eight_cells():
        initial = frontier._settle(episode, frontier._initial_state(episode))
        stack = [initial]
        seen: set[frontier.SimulationState] = set()
        while stack and len(seen) < 20_000:
            state = frontier._settle(episode, stack.pop())
            if state in seen or frontier._terminal(episode, state):
                continue
            seen.add(state)
            actual = critical.canonical_actions(episode, state, sham=False)
            sham = critical.canonical_actions(episode, state, sham=True)
            has_split = any(action.kind == "critical" for action in actual)
            sham_differs = set(actual) != set(sham)
            if (not require_split or has_split) and (
                not require_sham_difference or sham_differs
            ):
                return episode, state
            for action in actual:
                next_state, _service, _selected = critical._apply_action(
                    episode, state, catalog, action, sham=False
                )
                stack.append(next_state)
    raise AssertionError("no qualifying frozen state found")


def _write_valid_lock(directory: Path) -> tuple[Path, str]:
    lock_path = directory / "criticalsplit_lock.json"
    source_files = {
        relative: critical._sha256(critical._source_path(relative))
        for relative in critical.REQUIRED_LOCK_SOURCES
    }
    preflight_path = directory / "criticalsplit_preflight.json"
    critical._write_json(
        preflight_path,
        {
            "schema": "criticalsplit-preflight-v1",
            "status": "PASS",
            "created_at_utc": "2026-08-10T00:00:00+00:00",
            "source_files_tested": source_files,
            "source_set_sha256": critical._mapping_digest(source_files),
            "checks": {
                "py_compile": {"command": "fixture py_compile", "exit_code": 0},
                "frontier_tests": {
                    "command": "fixture frontier tests",
                    "exit_code": 0,
                    "tests_run": 13,
                },
                "criticalsplit_contract_tests": {
                    "command": "fixture criticalsplit tests",
                    "exit_code": 0,
                    "tests_run": 11,
                },
                "old_artifact_replay": {
                    "command": "fixture artifact replay",
                    "exit_code": 0,
                    "artifact_sha256": {
                        relative: digest
                        for relative, digest in critical.FROZEN_EXPECTED_SHA256.items()
                        if relative.startswith("artifacts/frontiercredit_pilot/")
                    },
                },
            },
        },
    )
    payload = {
        "schema": critical.LOCK_SCHEMA,
        "status": "LOCKED_BEFORE_COMPUTE",
        "created_at_utc": "2026-08-10T00:01:00+00:00",
        "max_oracle_states": critical.FROZEN_MAX_ORACLE_STATES,
        "decision_thresholds": dict(critical.DECISION_THRESHOLDS),
        "episode_ids": [
            episode.episode_id for episode in frontier.generate_eight_cells()
        ],
        "experiment_plan": "refine-logs/EXPERIMENT_PLAN_20260810_164938.md",
        "action_space": critical.FROZEN_ACTION_SPACE,
        "claim_ceiling": critical.CLAIM_CEILING,
        "evaluation_type": "simulation_only",
        "scientific_result_eligible": False,
        "paper_result": False,
        "source_files": source_files,
        "source_set_sha256": critical._mapping_digest(source_files),
        "preflight": preflight_path.name,
        "preflight_sha256": critical._sha256(preflight_path),
    }
    critical._write_json(lock_path, payload)
    return lock_path, critical._sha256(lock_path)


def _fake_payload() -> dict[str, object]:
    supporting = [
        {
            "episode_id": "fake-0",
            "actual_split_improves_whole": True,
            "immediate_flow_us": 10.0,
            "whole_oracle_flow_us": 8.0,
            "actual_split_flow_us": 6.0,
            "whole_capture": 0.50,
            "identity_gap": 0.20,
            "critical_launches": 1,
            "bulk_launches": 1,
            "sham_applicable_decisions": 1,
            "sham_partition_changed_decisions": 1,
            "deadline_miss_delta_vs_whole": 0,
        },
        {
            "episode_id": "fake-1",
            "actual_split_improves_whole": True,
            "immediate_flow_us": 11.0,
            "whole_oracle_flow_us": 9.0,
            "actual_split_flow_us": 7.0,
            "whole_capture": 0.50,
            "identity_gap": 0.20,
            "critical_launches": 1,
            "bulk_launches": 1,
            "sham_applicable_decisions": 1,
            "sham_partition_changed_decisions": 1,
            "deadline_miss_delta_vs_whole": 0,
        },
    ]
    return {
        "schema": "criticalsplit-full-dag-pilot-v1",
        "status": "COMPUTED_SIMULATION_ONLY",
        "evaluation_type": "simulation_only",
        "scientific_result_eligible": False,
        "paper_result": False,
        "protocol": {
            "cells": "frozen FrontierCredit overlap(2) x arrival(2) x deadline(2)",
            "action_space": critical.FROZEN_ACTION_SPACE,
            "partition_visibility": (
                "revealed physical identity or revealed-prefix identity sham"
            ),
            "oracle_visibility": (
                "future-known exact search over identical physical transitions"
            ),
            "max_oracle_states": critical.FROZEN_MAX_ORACLE_STATES,
            "decision_thresholds": dict(critical.DECISION_THRESHOLDS),
        },
        "cells": [
            {"episode_id": row["episode_id"], "summary": row, "results": {}}
            for row in supporting
        ],
        "decision": critical.decide(supporting),
        "claim_ceiling": critical.CLAIM_CEILING,
    }


class CriticalSplitFullDagPilotTest(unittest.TestCase):
    def setUp(self) -> None:
        self.catalog = frontier.make_service_catalog()

    def test_subset_launch_preserves_bulk_ready_nodes_and_age(self) -> None:
        episode, state = _find_state(require_split=True)
        action = next(
            value
            for value in critical.canonical_actions(episode, state, sham=False)
            if value.kind == "critical"
        )
        before = {item.node_id: item for item in state.ready}
        next_state, service, selected = critical._launch_ready_subset(
            episode,
            state,
            self.catalog,
            action.queue_key,
            action.node_ids,
        )
        after = {item.node_id: item for item in next_state.ready}
        self.assertEqual(selected, action.node_ids)
        self.assertEqual(service, self.catalog.estimate_us(critical.MODEL, action.queue_key[1], len(selected)))
        self.assertFalse(set(selected) & set(after))
        for node_id in set(before) - set(selected):
            self.assertEqual(after[node_id], before[node_id])

    def test_subset_launch_rejects_empty_duplicate_unsorted_and_cross_queue(self) -> None:
        episode = frontier.generate_eight_cells()[0]
        state = frontier._settle(episode, frontier._initial_state(episode))
        eligible = frontier._eligible_queues(episode, state)
        keys = sorted(eligible)
        queue_key = keys[0]
        node_id = eligible[queue_key][0].node_id
        with self.assertRaises(frontier.ProtocolError):
            critical._launch_ready_subset(episode, state, self.catalog, queue_key, ())
        with self.assertRaises(frontier.ProtocolError):
            critical._launch_ready_subset(
                episode, state, self.catalog, queue_key, (node_id, node_id)
            )
        if len(eligible[queue_key]) >= 2:
            ids = tuple(item.node_id for item in eligible[queue_key])
            with self.assertRaises(frontier.ProtocolError):
                critical._launch_ready_subset(
                    episode, state, self.catalog, queue_key, tuple(reversed(ids))
                )
        other = keys[1]
        foreign = eligible[other][0].node_id
        with self.assertRaises(frontier.ProtocolError):
            critical._launch_ready_subset(
                episode, state, self.catalog, queue_key, (foreign,)
            )

    def test_subset_launch_rejects_busy_executor(self) -> None:
        episode = frontier.generate_eight_cells()[0]
        state = frontier._settle(episode, frontier._initial_state(episode))
        queue_key, items = next(iter(sorted(frontier._eligible_queues(episode, state).items())))
        busy = list(state.running)
        busy[queue_key[0]] = frontier.RunningBatch(
            queue_key=queue_key,
            node_ids=(items[0].node_id,),
            start_us=state.now_us,
            finish_us=state.now_us + 1.0,
            service_us=1.0,
        )
        state = replace(state, running=tuple(busy))
        with self.assertRaises(frontier.ProtocolError):
            critical._launch_ready_subset(
                episode, state, self.catalog, queue_key, (items[0].node_id,)
            )

    def test_partition_is_proper_and_identity_sham_changes_only_actions(self) -> None:
        episode, state = _find_state(require_split=True, require_sham_difference=True)
        before = state
        actual = critical.canonical_actions(episode, state, sham=False)
        sham = critical.canonical_actions(episode, state, sham=True)
        self.assertNotEqual(set(actual), set(sham))
        self.assertEqual(state, before)
        for action in actual:
            if action.kind not in {"critical", "bulk"}:
                continue
            whole = next(
                value
                for value in actual
                if value.kind == "whole" and value.queue_key == action.queue_key
            )
            self.assertLess(len(action.node_ids), len(whole.node_ids))
            self.assertGreater(len(action.node_ids), 0)

    def test_partition_does_not_read_unrevealed_routes(self) -> None:
        episode, state = _find_state(require_split=True)
        revealed = set(dict(state.revealed_at))
        hidden = {node.node_id for node in episode.nodes} - revealed
        self.assertTrue(hidden)
        changed = replace(
            episode,
            nodes=tuple(
                replace(node, expert_id=node.expert_id + 1000)
                if node.node_id in hidden
                else node
                for node in episode.nodes
            ),
        )
        self.assertEqual(
            critical.canonical_actions(episode, state, sham=False),
            critical.canonical_actions(changed, state, sham=False),
        )

    def test_sham_applicability_ignores_pure_critical_bulk_label_swap(self) -> None:
        queue_key = (0, 0, 0)
        actual = (
            critical.Action("critical", queue_key, ("a",)),
            critical.Action("bulk", queue_key, ("b",)),
        )
        swapped = (
            critical.Action("bulk", queue_key, ("a",)),
            critical.Action("critical", queue_key, ("b",)),
        )
        self.assertNotEqual(set(actual), set(swapped))
        self.assertEqual(
            critical._physical_action_signature(actual),
            critical._physical_action_signature(swapped),
        )

    def test_canonical_actions_are_unique_and_round_trip_tokens(self) -> None:
        episode, state = _find_state(require_split=True)
        actions = critical.canonical_actions(episode, state, sham=False)
        self.assertEqual(len(actions), len(set(actions)))
        for action in actions:
            self.assertEqual(critical.Action.from_token(action.token()), action)
        with self.assertRaisesRegex(frontier.ProtocolError, "queue key"):
            critical.Action.from_token(
                '{"kind":"whole","node_ids":["n"],"queue_key":[true,0,0]}'
            )

    def test_replay_rejects_stale_or_tampered_subset(self) -> None:
        episode, state = _find_state(require_split=True)
        action = next(
            value
            for value in critical.canonical_actions(episode, state, sham=False)
            if value.kind == "critical"
        )
        whole = next(
            value
            for value in critical.canonical_actions(episode, state, sham=False)
            if value.kind == "whole" and value.queue_key == action.queue_key
        )
        tampered_ids = tuple(sorted(set(whole.node_ids) - set(action.node_ids)))
        tampered = critical.Action("critical", action.queue_key, tampered_ids)
        with self.assertRaisesRegex(frontier.ProtocolError, "stale|noncanonical"):
            critical._apply_action(
                episode, state, self.catalog, tampered, sham=False
            )

    def test_expanded_oracle_is_not_worse_than_whole_ready_oracle(self) -> None:
        for episode in frontier.generate_eight_cells():
            whole = frontier.solve_exact_oracle(episode, self.catalog)
            expanded = critical.solve_split_oracle(episode, self.catalog)
            self.assertLessEqual(
                critical._objective_prefix(expanded),
                critical._objective_prefix(whole),
                episode.episode_id,
            )

    def test_split_oracle_state_cap_fails_closed(self) -> None:
        episode = frontier.generate_eight_cells()[0]
        with self.assertRaisesRegex(frontier.ProtocolError, "UNSOLVED_EXACT_STATE_LIMIT"):
            critical.solve_split_oracle(episode, self.catalog, max_states=1)

    def test_decision_is_mechanical(self) -> None:
        supporting = [
            {
                "actual_split_improves_whole": True,
                "whole_capture": 0.50,
                "identity_gap": 0.20,
                "critical_launches": 1,
                "sham_applicable_decisions": 1,
                "deadline_miss_delta_vs_whole": 0,
            },
            {
                "actual_split_improves_whole": True,
                "whole_capture": 0.70,
                "identity_gap": 0.10,
                "critical_launches": 2,
                "sham_applicable_decisions": 1,
                "deadline_miss_delta_vs_whole": -1,
            },
        ]
        self.assertEqual(
            critical.decide(supporting)["verdict"], "SUPPORT_ACTION_SPACE"
        )
        weakened = [dict(row) for row in supporting]
        weakened[0]["critical_launches"] = 0
        self.assertEqual(
            critical.decide(weakened)["verdict"], "WEAKEN_ACTION_SPACE"
        )
        noneligible_miss_regression = supporting + [
            {
                "actual_split_improves_whole": False,
                "whole_capture": None,
                "identity_gap": None,
                "critical_launches": 0,
                "sham_applicable_decisions": 0,
                "deadline_miss_delta_vs_whole": 1,
            }
        ]
        self.assertEqual(
            critical.decide(noneligible_miss_regression)["verdict"],
            "WEAKEN_ACTION_SPACE",
        )

    def test_completion_last_bundle_verifies_and_tamper_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lock_path, lock_sha = _write_valid_lock(root)
            report = critical._validate_lock(
                lock_path,
                lock_sha,
                max_oracle_states=critical.FROZEN_MAX_ORACLE_STATES,
            )
            output_dir = root / "success"
            output_dir.mkdir()
            (output_dir / "RUN_LOCK.json").write_bytes(lock_path.read_bytes())
            (output_dir / "PREFLIGHT.json").write_bytes(
                Path(str(report["preflight_path"])).read_bytes()
            )
            original_seal = critical._write_json_atomic_exclusive
            with mock.patch.object(
                critical,
                "_write_json_atomic_exclusive",
                wraps=original_seal,
            ) as seal:
                critical.write_outputs(
                    output_dir,
                    _fake_payload(),
                    report,
                    lock_path=lock_path,
                    expected_lock_sha256=lock_sha,
                    max_oracle_states=critical.FROZEN_MAX_ORACLE_STATES,
                )
            seal.assert_called_once()
            self.assertEqual(seal.call_args.args[0].name, "COMPLETE.json")
            complete = critical.verify_complete(output_dir)
            self.assertEqual(complete["status"], "SUCCESS_COMPLETE")
            self.assertFalse(complete["scientific_result_eligible"])
            self.assertFalse(complete["paper_result"])
            self.assertEqual(
                complete["manifest_sha256"],
                critical._sha256(output_dir / "MANIFEST.json"),
            )
            (output_dir / "decision.json").write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(frontier.ProtocolError, "hash mismatch"):
                critical.verify_complete(output_dir)

    def test_state_cap_failure_writes_failure_without_success_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lock_path, lock_sha = _write_valid_lock(root)
            output_dir = root / "failed"
            with mock.patch.object(
                critical,
                "run_pilot",
                side_effect=frontier.ProtocolError("UNSOLVED_EXACT_STATE_LIMIT"),
            ):
                with self.assertRaisesRegex(
                    frontier.ProtocolError, "UNSOLVED_EXACT_STATE_LIMIT"
                ):
                    critical.execute_once(output_dir, lock_path, lock_sha)
            self.assertEqual(
                {path.name for path in output_dir.iterdir()},
                {"RUN_LOCK.json", "PREFLIGHT.json", "failure.json"},
            )
            failure = json.loads(
                (output_dir / "failure.json").read_text(encoding="utf-8")
            )
            self.assertEqual(failure["stage"], "EXACT_SEARCH")
            self.assertEqual(failure["error_code"], "UNSOLVED_EXACT_STATE_LIMIT")
            self.assertFalse(failure["complete_written"])
            self.assertFalse(failure["verdict_authorized"])

    def test_source_post_drift_fails_before_manifest_and_complete(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lock_path, lock_sha = _write_valid_lock(root)
            report = critical._validate_lock(
                lock_path,
                lock_sha,
                max_oracle_states=critical.FROZEN_MAX_ORACLE_STATES,
            )
            drift = dict(report)
            drift_sources = dict(report["source_files"])
            first = critical.REQUIRED_LOCK_SOURCES[0]
            drift_sources[first] = "0" * 64
            drift["source_files"] = drift_sources
            drift["source_set_sha256"] = critical._mapping_digest(drift_sources)
            output_dir = root / "drift"
            with (
                mock.patch.object(critical, "run_pilot", return_value=_fake_payload()),
                mock.patch.object(critical, "_validate_payload"),
                mock.patch.object(
                    critical,
                    "_validate_lock",
                    side_effect=[report, report, drift],
                ),
            ):
                with self.assertRaisesRegex(frontier.ProtocolError, "changed"):
                    critical.execute_once(output_dir, lock_path, lock_sha)
            self.assertTrue((output_dir / "failure.json").is_file())
            self.assertTrue((output_dir / "SOURCE_POST.json").is_file())
            self.assertFalse((output_dir / "MANIFEST.json").exists())
            self.assertFalse((output_dir / "COMPLETE.json").exists())

    def test_existing_output_directory_is_rejected_without_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lock_path, lock_sha = _write_valid_lock(root)
            output_dir = root / "existing"
            output_dir.mkdir()
            marker = output_dir / "keep.txt"
            marker.write_text("keep\n", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                critical.execute_once(output_dir, lock_path, lock_sha)
            self.assertEqual(marker.read_text(encoding="utf-8"), "keep\n")


if __name__ == "__main__":
    unittest.main()
