from __future__ import annotations

from dataclasses import replace
import json
import unittest

try:
    from .core import ProtocolError, validate_causal_route_v3
    from .frontiercredit_full_dag_pilot import (
        EDFPolicy,
        Episode,
        FrontierCreditPolicy,
        ImmediatePolicy,
        MaxRowsPolicy,
        QueueLocalCreditPolicy,
        _build_view,
        _decision_queues,
        _eligible_queues,
        _initial_state,
        _observation_map,
        _settle,
        build_episode,
        generate_eight_cells,
        make_service_catalog,
        oracle_capture,
        run_pilot,
        simulate_policy,
        solve_exact_oracle,
    )
except ImportError:
    from core import ProtocolError, validate_causal_route_v3
    from frontiercredit_full_dag_pilot import (
        EDFPolicy,
        Episode,
        FrontierCreditPolicy,
        ImmediatePolicy,
        MaxRowsPolicy,
        QueueLocalCreditPolicy,
        _build_view,
        _decision_queues,
        _eligible_queues,
        _initial_state,
        _observation_map,
        _settle,
        build_episode,
        generate_eight_cells,
        make_service_catalog,
        oracle_capture,
        run_pilot,
        simulate_policy,
        solve_exact_oracle,
    )


def tiny_episode() -> Episode:
    source = build_episode(overlap="crossed", arrival="aligned", deadline="loose")
    request = source.requests[0]
    contributions = tuple(
        row
        for row in source.contributions
        if row.request_id == request.request_id and row.decode_step == 0 and row.layer_id == 0
    )
    node_ids = {row.contribution_id for row in contributions}
    nodes = tuple(node for node in source.nodes if node.node_id in node_ids)
    return Episode(
        episode_id="tiny",
        requests=(request,),
        contributions=contributions,
        nodes=nodes,
        decode_steps=1,
        layers=1,
        top_k=2,
    )


class FrontierCreditFullDagPilotTest(unittest.TestCase):
    def setUp(self) -> None:
        self.catalog = make_service_catalog()

    def test_eight_cells_are_schema_closed(self) -> None:
        cells = generate_eight_cells()
        self.assertEqual(len(cells), 8)
        self.assertEqual(len({cell.episode_id for cell in cells}), 8)
        for cell in cells:
            summary = validate_causal_route_v3(
                cell.contributions,
                require_observed_stages=False,
            )
            self.assertEqual(
                summary,
                {"contributions": 16, "events": 8, "requests": 2, "documents": 2},
            )
            for members in cell.group_map.values():
                self.assertEqual(
                    sorted(cell.node_map[node_id].topk_slot for node_id in members),
                    [0, 1],
                )

    def test_counterfactual_completion_releases_next_layer_and_step(self) -> None:
        episode = build_episode(overlap="crossed", arrival="aligned", deadline="loose")
        result = simulate_policy(episode, self.catalog, ImmediatePolicy())
        revealed = result["node_revealed_us"]
        completed = result["node_completion_us"]
        for request in episode.requests:
            for step in range(episode.decode_steps):
                for layer in range(episode.layers):
                    group = (request.request_id, step, layer)
                    finish = max(completed[node_id] for node_id in episode.group_map[group])
                    if layer + 1 < episode.layers:
                        successor = (request.request_id, step, layer + 1)
                    elif step + 1 < episode.decode_steps:
                        successor = (request.request_id, step + 1, 0)
                    else:
                        self.assertEqual(
                            result["request_completion_us"][request.request_id],
                            finish + episode.combine_us,
                        )
                        continue
                    successor_ready = {
                        revealed[node_id] for node_id in episode.group_map[successor]
                    }
                    self.assertEqual(successor_ready, {finish + episode.combine_us})

    def test_online_view_hides_future_route_suffix(self) -> None:
        base = build_episode(overlap="crossed", arrival="aligned", deadline="loose")
        changed_nodes = tuple(
            replace(node, expert_id=node.expert_id + 10)
            if node.decode_step > 0 or node.layer_id > 0
            else node
            for node in base.nodes
        )
        changed = replace(base, episode_id="changed-hidden-suffix", nodes=changed_nodes)
        base_state = _settle(base, _initial_state(base))
        changed_state = _settle(changed, _initial_state(changed))
        base_view = _build_view(base, base_state, self.catalog, sham=False)
        changed_view = _build_view(changed, changed_state, self.catalog, sham=False)
        self.assertEqual(base_view, changed_view)
        policy = FrontierCreditPolicy()
        self.assertEqual(policy.choose(base_view), policy.choose(changed_view))

    def test_identity_sham_changes_only_observed_sibling_mapping(self) -> None:
        episode = build_episode(overlap="crossed", arrival="aligned", deadline="loose")
        before = json.dumps(
            [node.__dict__ for node in episode.nodes],
            sort_keys=True,
        )
        state = _settle(episode, _initial_state(episode))
        actual = _build_view(episode, state, self.catalog, sham=False)
        sham = _build_view(episode, state, self.catalog, sham=True)
        self.assertEqual(actual.queues, sham.queues)
        self.assertNotEqual(actual.frontiers, sham.frontiers)
        self.assertEqual(
            sorted(len(frontier.members) for frontier in actual.frontiers),
            sorted(len(frontier.members) for frontier in sham.frontiers),
        )
        self.assertTrue(all(frontier.complete_observation for frontier in sham.frontiers))
        self.assertNotEqual(
            _observation_map(episode, state, sham=False),
            _observation_map(episode, state, sham=True),
        )
        self.assertLessEqual(
            {
                group_key[0]
                for group_key in _observation_map(episode, state, sham=True).values()
            },
            set(state.arrived_requests),
        )
        simulate_policy(
            episode,
            self.catalog,
            FrontierCreditPolicy(name="frontier_identity_sham"),
            sham=True,
        )
        after = json.dumps([node.__dict__ for node in episode.nodes], sort_keys=True)
        self.assertEqual(before, after)

    def test_identity_sham_never_exposes_a_future_arrival(self) -> None:
        episode = build_episode(overlap="crossed", arrival="staggered", deadline="loose")
        state = _settle(episode, _initial_state(episode))
        self.assertEqual(state.arrived_requests, ("r0",))
        actual = _observation_map(episode, state, sham=False)
        sham = _observation_map(episode, state, sham=True)
        self.assertEqual(actual, sham)
        self.assertEqual(set(sham), set(dict(state.revealed_at)))
        self.assertEqual({group[0] for group in sham.values()}, {"r0"})

    def test_sham_is_invariant_to_unrevealed_request_identity_and_route(self) -> None:
        episode = build_episode(overlap="crossed", arrival="staggered", deadline="loose")
        changed_requests = tuple(
            replace(request, request_id="future-renamed")
            if request.request_id == "r1"
            else request
            for request in episode.requests
        )
        changed_nodes = tuple(
            replace(node, request_id="future-renamed", expert_id=node.expert_id + 100)
            if node.request_id == "r1"
            else node
            for node in episode.nodes
        )
        changed = replace(
            episode,
            episode_id="future-renamed-and-rerouted",
            requests=changed_requests,
            nodes=changed_nodes,
        )
        state = _settle(episode, _initial_state(episode))
        changed_state = _settle(changed, _initial_state(changed))
        view = _build_view(episode, state, self.catalog, sham=True)
        changed_view = _build_view(changed, changed_state, self.catalog, sham=True)
        self.assertEqual(view, changed_view)
        policy = FrontierCreditPolicy(name="frontier_identity_sham")
        self.assertEqual(policy.choose(view), policy.choose(changed_view))

    def test_action_space_includes_every_queue_on_every_idle_executor(self) -> None:
        for episode in generate_eight_cells():
            state = _settle(episode, _initial_state(episode))
            self.assertEqual(
                _decision_queues(episode, state),
                _eligible_queues(episode, state),
                episode.episode_id,
            )

    def test_whole_queue_actions_conserve_every_node(self) -> None:
        episode = build_episode(overlap="aligned", arrival="aligned", deadline="loose")
        result = simulate_policy(episode, self.catalog, QueueLocalCreditPolicy())
        flushed = [
            node_id
            for action in result["actions"]
            if action["kind"] == "flush"
            for node_id in action["node_ids"]
        ]
        self.assertEqual(set(flushed), set(episode.node_map))
        self.assertEqual(len(flushed), len(set(flushed)))
        for action in result["actions"]:
            if action["kind"] == "flush":
                self.assertEqual(action["rows"], len(action["node_ids"]))

    def test_exact_oracle_matches_independent_tiny_expectation(self) -> None:
        episode = tiny_episode()
        oracle = solve_exact_oracle(episode, self.catalog)
        # Two fixed idle executors launch the two top-k siblings concurrently:
        # 10 us singleton service + 1 us combine tail.
        self.assertTrue(oracle["exact"])
        self.assertEqual(oracle["flow_us"], 11.0)
        self.assertEqual(oracle["launches"], 2)
        self.assertTrue(all(action["kind"] == "flush" for action in oracle["actions"][:2]))

    def test_oracle_is_not_worse_than_any_online_policy(self) -> None:
        for episode in generate_eight_cells():
            oracle = solve_exact_oracle(episode, self.catalog)
            self.assertTrue(oracle["exact"], episode.episode_id)
            self.assertTrue(oracle["actions"], episode.episode_id)
            for action in oracle["actions"]:
                if action["kind"] == "flush":
                    self.assertEqual(action["rows"], len(action["node_ids"]))
            for policy in (
                ImmediatePolicy(),
                EDFPolicy(),
                MaxRowsPolicy(),
                QueueLocalCreditPolicy(),
                FrontierCreditPolicy(),
            ):
                result = simulate_policy(episode, self.catalog, policy)
                self.assertGreaterEqual(
                    result["flow_us"], oracle["flow_us"], episode.episode_id
                )

    def test_decision_uses_one_fixed_reference_not_a_cellwise_winner(self) -> None:
        payload = run_pilot()
        decision = payload["decision"]
        self.assertEqual(decision["fixed_simple_reference"], "queue_local_credit")
        self.assertLess(
            decision["maximum_fixed_simple_median_capture"],
            decision["thresholds"]["maximum_fixed_simple_median_capture"],
        )
        self.assertIn(
            "cellwise_best_simple_oracle_envelope_median_capture_diagnostic_only",
            decision,
        )
        for cell in payload["cells"]:
            summary = cell["summary"]
            self.assertEqual(summary["fixed_simple_reference"], "queue_local_credit")
            self.assertNotIn("strongest_simple", summary)

    def test_zero_headroom_is_not_normalized_to_one(self) -> None:
        self.assertIsNone(oracle_capture(10.0, 10.0, 10.0))

    def test_exact_state_cap_fails_closed(self) -> None:
        with self.assertRaisesRegex(ProtocolError, "UNSOLVED_EXACT_STATE_LIMIT"):
            solve_exact_oracle(tiny_episode(), self.catalog, max_states=1)


if __name__ == "__main__":
    unittest.main()
