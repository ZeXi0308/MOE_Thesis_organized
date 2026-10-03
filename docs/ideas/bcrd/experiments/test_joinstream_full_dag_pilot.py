from __future__ import annotations

import math
from pathlib import Path
import tempfile
import unittest

try:
    from . import joinstream_full_dag_pilot as joinstream
except ImportError:
    import joinstream_full_dag_pilot as joinstream  # type: ignore


class JoinStreamFullDagPilotTest(unittest.TestCase):
    def setUp(self) -> None:
        self.catalog = joinstream.make_service_catalog()

    @staticmethod
    def _initial(cell: joinstream.JoinStreamCell) -> joinstream.SimulationState:
        return joinstream._settle(
            cell.episode, joinstream._initial_state(cell.episode)
        )

    @staticmethod
    def _action(
        cell: joinstream.JoinStreamCell,
        state: joinstream.SimulationState,
        kind: str,
        replica: int = 0,
    ) -> joinstream.Action:
        return next(
            action
            for action in joinstream.canonical_actions(cell, state, expanded=True)
            if action.kind == kind and action.queue_key is not None
            and action.queue_key[0] == replica
        )

    def test_atomic_batch_rows_share_one_completion_milestone(self) -> None:
        cell = joinstream.build_cell(batch_rows=4, curve="uniform", tax_us=0)
        state = self._initial(cell)
        action = self._action(cell, state, "atomic")
        next_state, service, selected, milestones = joinstream._launch(
            cell, state, self.catalog, action, expanded=True
        )
        self.assertEqual(service, 20.0)
        self.assertEqual(set(selected), set(action.node_ids))
        self.assertEqual({item.at_us for item in milestones}, {20.0})
        batch = next_state.running[0]
        self.assertIsNotNone(batch)
        assert batch is not None
        self.assertEqual(batch.finish_us, 20.0)
        self.assertEqual(batch.kind, "atomic")

    def test_stream_emits_prefix_but_executor_stays_busy_to_final(self) -> None:
        cell = joinstream.build_cell(batch_rows=2, curve="uniform", tax_us=0)
        state = self._initial(cell)
        action = self._action(cell, state, "stream")
        state, service, _selected, milestones = joinstream._launch(
            cell, state, self.catalog, action, expanded=True
        )
        self.assertEqual(service, 14.0)
        self.assertEqual([item.at_us for item in milestones], [7.0, 14.0])
        first = joinstream._settle(
            cell.episode, joinstream.replace(state, now_us=7.0)
        )
        self.assertIn(action.row_order[0], dict(first.completed_at))
        self.assertNotIn(action.row_order[1], dict(first.completed_at))
        self.assertIsNotNone(first.running[0])
        final = joinstream._settle(
            cell.episode, joinstream.replace(first, now_us=14.0)
        )
        self.assertIn(action.row_order[1], dict(final.completed_at))
        self.assertIsNone(final.running[0])

    def test_frozen_uniform_tail_curves_and_tax(self) -> None:
        uniform = joinstream._milestone_offsets(
            rows=4, service_us=20.0, curve="uniform", tax_us=2.0
        )
        self.assertEqual(uniform, (7.0, 12.0, 17.0, 22.0))
        tail = joinstream._milestone_offsets(
            rows=4, service_us=20.0, curve="tail", tax_us=2.0
        )
        expected = tuple(2.0 + 20.0 * math.sqrt(index / 4) for index in range(1, 5))
        for actual, wanted in zip(tail, expected):
            self.assertAlmostEqual(actual, wanted)
        self.assertEqual(tail[-1], 22.0)

        cell = joinstream.build_cell(batch_rows=4, curve="tail", tax_us=2.0)
        state = self._initial(cell)
        action = self._action(cell, state, "stream")
        _state, service, _selected, milestones = joinstream._launch(
            cell, state, self.catalog, action, expanded=True
        )
        self.assertEqual(service, 22.0)
        self.assertEqual(max(item.at_us for item in milestones), 22.0)

    def test_expanded_action_set_contains_every_atomic_action(self) -> None:
        cell = joinstream.build_cell(batch_rows=4, curve="uniform", tax_us=0)
        state = self._initial(cell)
        baseline = set(joinstream.canonical_actions(cell, state, expanded=False))
        expanded = set(joinstream.canonical_actions(cell, state, expanded=True))
        self.assertLessEqual(baseline, expanded)
        self.assertTrue(any(action.kind == "stream" for action in expanded))
        for stream in (action for action in expanded if action.kind == "stream"):
            atomic = joinstream.Action("atomic", stream.queue_key, stream.node_ids)
            self.assertIn(atomic, baseline)
            self.assertEqual(set(stream.row_order), set(stream.node_ids))

    def test_m1_has_no_fake_stream_and_exact_oracles_are_equivalent(self) -> None:
        cell = joinstream.build_cell(batch_rows=1, curve="uniform", tax_us=2)
        state = self._initial(cell)
        baseline_actions = joinstream.canonical_actions(cell, state, expanded=False)
        expanded_actions = joinstream.canonical_actions(cell, state, expanded=True)
        self.assertEqual(baseline_actions, expanded_actions)
        self.assertFalse(any(action.kind == "stream" for action in expanded_actions))
        baseline = joinstream.solve_exact_oracle(
            cell, self.catalog, expanded=False
        )
        expanded = joinstream.solve_exact_oracle(
            cell, self.catalog, expanded=True
        )
        self.assertEqual(
            joinstream._objective_prefix(baseline),
            joinstream._objective_prefix(expanded),
        )
        self.assertEqual(expanded["stream_launches"], 0)

    def test_early_stream_milestone_preserves_join_combine_release(self) -> None:
        cell = joinstream.build_cell(batch_rows=2, curve="uniform", tax_us=0)
        state = self._initial(cell)
        first_action = self._action(cell, state, "stream", replica=0)
        state, _service, _selected, _milestones = joinstream._launch(
            cell, state, self.catalog, first_action, expanded=True
        )
        second_action = self._action(cell, state, "stream", replica=1)
        state, _service, _selected, _milestones = joinstream._launch(
            cell, state, self.catalog, second_action, expanded=True
        )

        first_milestone = joinstream._advance_without_action(cell.episode, state)
        self.assertEqual(first_milestone.now_us, 7.0)
        self.assertTrue(all(batch is not None for batch in first_milestone.running))
        self.assertIn(("r00", 0, 0), first_milestone.joined_groups)
        release = joinstream._advance_without_action(cell.episode, first_milestone)
        self.assertEqual(release.now_us, 8.0)
        revealed = dict(release.revealed_at)
        for node_id in cell.episode.group_map[("r00", 0, 1)]:
            self.assertEqual(revealed[node_id], 8.0)
        self.assertTrue(all(batch is not None for batch in release.running))

    def test_exact_action_trace_replay_reproduces_all_objective_fields(self) -> None:
        cell = joinstream.build_cell(batch_rows=4, curve="uniform", tax_us=0)
        solved = joinstream.solve_exact_oracle(
            cell, self.catalog, expanded=True
        )
        ledger, replay = joinstream._replay_actions(
            cell,
            self.catalog,
            solved["action_tokens"],
            expanded=True,
        )
        self.assertEqual(ledger, solved["actions"])
        for field in (
            "flow_us",
            "total_tardiness_us",
            "deadline_misses",
            "launches",
            "total_service_us",
            "request_completion_us",
        ):
            self.assertEqual(replay[field], solved[field])
        self.assertTrue(replay["node_conservation"]["exactly_once"])

    def test_frozen_eight_cells_solve_under_cap_and_close_every_node(self) -> None:
        cells = joinstream.generate_eight_cells()
        self.assertEqual(len(cells), 8)
        self.assertEqual(len({cell.episode.episode_id for cell in cells}), 8)
        for cell in cells:
            self.assertEqual(len(cell.episode.requests), cell.batch_rows)
            self.assertEqual(cell.episode.replicas, 2)
            self.assertEqual(cell.episode.top_k, 2)
            self.assertEqual(cell.episode.layers, 2)
            self.assertEqual(cell.episode.decode_steps, 1)
            baseline = joinstream.solve_exact_oracle(
                cell, self.catalog, expanded=False
            )
            expanded = joinstream.solve_exact_oracle(
                cell, self.catalog, expanded=True
            )
            self.assertLessEqual(
                joinstream._objective_prefix(expanded),
                joinstream._objective_prefix(baseline),
                cell.episode.episode_id,
            )
            for result in (baseline, expanded):
                self.assertLessEqual(result["states_evaluated"], 500_000)
                self.assertTrue(result["node_conservation"]["exactly_once"])
                self.assertEqual(
                    result["node_conservation"]["expected_nodes"],
                    cell.batch_rows * 2 * 2,
                )

    def test_state_cap_fails_closed(self) -> None:
        cell = joinstream.build_cell(batch_rows=2, curve="uniform", tax_us=0)
        with self.assertRaisesRegex(
            joinstream.frontier.ProtocolError, "UNSOLVED_EXACT_STATE_LIMIT"
        ):
            joinstream.solve_exact_oracle(
                cell, self.catalog, expanded=True, max_states=1
            )

    def test_cli_writer_is_explicitly_unsealed(self) -> None:
        payload = {
            "status": "EXPLORATORY_COMPUTED_NOT_SEALED",
            "cells": [],
            "claim_ceiling": joinstream.CLAIM_CEILING,
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "joinstream"
            joinstream.write_outputs(output, payload)
            self.assertEqual(
                {path.name for path in output.iterdir()},
                {"joinstream_results.json", "joinstream_summary.md"},
            )
            self.assertFalse((output / "COMPLETE.json").exists())


if __name__ == "__main__":
    unittest.main()
