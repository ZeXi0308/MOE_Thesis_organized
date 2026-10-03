from __future__ import annotations

from dataclasses import replace
import unittest

try:
    from .census import build_request_census, summarize_census
    from .test_schema import route_row
except ImportError:
    from census import build_request_census, summarize_census
    from test_schema import route_row


class RouteCensusTest(unittest.TestCase):
    def test_concentration_and_prefix_persistence(self) -> None:
        rows = [
            route_row(slot=0, expert=0, target_rank=0, chunk=0),
            route_row(slot=1, expert=1, target_rank=0, chunk=0),
            route_row(slot=0, expert=0, target_rank=0, chunk=1),
            route_row(slot=1, expert=2, target_rank=1, chunk=1),
        ]
        result = build_request_census(
            rows,
            num_experts={"toy": 4},
            num_ranks=2,
            prefix_fraction=0.25,
        )
        self.assertEqual(len(result), 1)
        census = result[0]
        self.assertAlmostEqual(census.rank_max_share, 0.75)
        self.assertAlmostEqual(census.rank_imbalance_factor, 1.5)
        self.assertAlmostEqual(census.prefix_future_rank_persistence, 0.5)
        self.assertTrue(census.prefix_future_dominant_rank_match)
        self.assertAlmostEqual(census.observed_prefix_contribution_fraction, 0.5)
        self.assertGreater(census.expert_normalized_entropy, 0.0)

    def test_summary_keeps_request_as_independent_unit(self) -> None:
        rows = [
            route_row(slot=0, expert=0, target_rank=0, chunk=0),
            route_row(slot=1, expert=1, target_rank=1, chunk=0),
        ]
        census = build_request_census(
            rows,
            num_experts={"toy": 4},
            num_ranks=2,
            prefix_fraction=0.25,
        )
        summary = summarize_census(census)
        self.assertEqual(summary[0]["request_count"], 1)
        self.assertEqual(summary[0]["independent_document_prompt_cluster_count"], 1)
        self.assertIsNone(summary[0]["median_prefix_future_rank_persistence"])

    def test_development_expert_only_suppresses_rank_claims(self) -> None:
        rows = [
            route_row(slot=0, expert=0, target_rank=0),
            route_row(slot=1, expert=1, target_rank=1),
        ]
        census = build_request_census(
            rows,
            num_experts={"toy": 4},
            num_ranks=2,
            prefix_fraction=0.25,
            require_rank_binding=False,
        )[0]
        self.assertIsNone(census.rank_max_share)
        self.assertIsNone(census.prefix_future_rank_persistence)

    def test_prefix_uses_causal_observation_time_not_chunk_id(self) -> None:
        rows = [
            *[
                replace(
                    route_row(
                        slot=index,
                        expert=index % 4,
                        target_rank=index % 2,
                        chunk=0,
                    ),
                    route_observed_us=3.0,
                )
                for index in range(4)
            ],
            *[
                replace(
                    route_row(
                        slot=index,
                        expert=index % 4,
                        target_rank=index % 2,
                        chunk=1,
                    ),
                    route_observed_us=1.0,
                )
                for index in range(3)
            ],
            replace(
                route_row(slot=0, expert=0, target_rank=0, chunk=2),
                route_observed_us=2.0,
            ),
        ]
        census = build_request_census(
            rows,
            num_experts={"toy": 4},
            num_ranks=2,
            prefix_fraction=0.25,
        )[0]
        self.assertEqual(census.causal_observation_us, 1.0)
        self.assertAlmostEqual(census.observed_prefix_contribution_fraction, 3 / 8)
        self.assertIsNone(census.remaining_service_work_fraction)
        self.assertIsNone(census.causal_action_eligible)

    def test_prefix_is_unresolved_when_causal_history_never_reaches_target(self) -> None:
        rows = [
            replace(
                route_row(slot=0, expert=0, target_rank=0, chunk=0),
                route_observed_us=1.0,
            ),
            *[
                replace(
                    route_row(
                        slot=index,
                        expert=index % 4,
                        target_rank=index % 2,
                        chunk=1,
                    ),
                    route_observed_us=2.0,
                )
                for index in range(9)
            ],
        ]
        census = build_request_census(
            rows,
            num_experts={"toy": 4},
            num_ranks=2,
            prefix_fraction=0.25,
        )[0]
        self.assertIsNone(census.observed_prefix_contribution_fraction)
        self.assertIsNone(census.prefix_future_rank_persistence)


if __name__ == "__main__":
    unittest.main()
