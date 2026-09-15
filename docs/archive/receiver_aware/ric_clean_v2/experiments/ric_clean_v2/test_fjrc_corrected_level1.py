from __future__ import annotations

import unittest

try:
    from .fjrc_corrected_level1 import (
        Level1Error,
        ServiceLUT,
        request_split,
        select_holdout_scenarios,
        select_split_scenarios,
    )
    from .fjrc_corrected_level0 import observation
except ImportError:  # pragma: no cover
    from fjrc_corrected_level1 import (  # type: ignore
        Level1Error,
        ServiceLUT,
        request_split,
        select_holdout_scenarios,
        select_split_scenarios,
    )
    from fjrc_corrected_level0 import observation  # type: ignore


def joins_fixture():
    joins = []
    for receiver in range(8):
        for request_index in range(8):
            request = f"r{receiver}-{request_index}"
            for position in range(2):
                siblings = [
                    {"topk_slot": slot, "sender_rank": slot, "expert_id": slot}
                    for slot in range(4)
                ]
                joins.append(
                    {
                        "join_id": f"j-{request}-{position}",
                        "request_id": request,
                        "receiver_rank": receiver,
                        "layer_id": 0,
                        "token_position": position,
                        "siblings": siblings,
                    }
                )
    return joins


def service():
    return ServiceLUT("olmoe", 1.0, 0.1, 2.0, 0.5, 3.1, "a" * 64)


class CorrectedFJRCLevel1Tests(unittest.TestCase):
    def test_request_split_is_32_plus_32_and_disjoint(self):
        selection, holdout = request_split(joins_fixture())
        self.assertEqual(len(selection), 32)
        self.assertEqual(len(holdout), 32)
        self.assertFalse(selection & holdout)

    def test_selects_16_request_disjoint_scenarios(self):
        scenarios = select_holdout_scenarios("olmoe", joins_fixture(), service())
        self.assertEqual(len(scenarios), 16)
        requests = [join.request_id for scenario in scenarios for join in scenario.joins]
        self.assertEqual(len(requests), 32)
        self.assertEqual(len(set(requests)), 32)
        for scenario in scenarios:
            self.assertEqual(len(scenario.candidate_task_ids), 2)
            self.assertEqual(observation(scenario, 0, "Q"), observation(scenario, 1, "Q"))
            self.assertNotEqual(observation(scenario, 0, "J"), observation(scenario, 1, "J"))

    def test_selection_and_holdout_scenarios_are_request_disjoint(self):
        selection = select_split_scenarios("olmoe", joins_fixture(), service(), split="selection")
        holdout = select_split_scenarios("olmoe", joins_fixture(), service(), split="holdout")
        selection_requests = {join.request_id for scenario in selection for join in scenario.joins}
        holdout_requests = {join.request_id for scenario in holdout for join in scenario.joins}
        self.assertEqual(len(selection_requests), 32)
        self.assertEqual(len(holdout_requests), 32)
        self.assertFalse(selection_requests & holdout_requests)

    def test_pair_selection_does_not_read_outcomes(self):
        scenarios_a = select_holdout_scenarios("olmoe", joins_fixture(), service())
        scenarios_b = select_holdout_scenarios("olmoe", list(reversed(joins_fixture())), service())
        self.assertEqual([row.scenario_id for row in scenarios_a], [row.scenario_id for row in scenarios_b])

    def test_wrong_service_model_is_rejected(self):
        bad = ServiceLUT("llmjp", 1.0, 0.1, 2.0, 0.5, 3.1, "a" * 64)
        with self.assertRaisesRegex(Level1Error, "model/service"):
            select_holdout_scenarios("olmoe", joins_fixture(), bad)

    def test_insufficient_common_sender_support_blocks(self):
        broken = joins_fixture()
        for join in broken:
            if join["request_id"].endswith("-7"):
                for sibling in join["siblings"]:
                    sibling["sender_rank"] += 10
        with self.assertRaisesRegex(Level1Error, "BLOCKED_INSUFFICIENT_MATCHED_SUPPORT"):
            select_holdout_scenarios("olmoe", broken, service())


if __name__ == "__main__":
    unittest.main()
