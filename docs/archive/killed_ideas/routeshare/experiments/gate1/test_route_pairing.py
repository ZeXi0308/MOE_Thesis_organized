from __future__ import annotations

import unittest

from route_pairing import Request, pair_greedy, pair_random, union_invocations


class PairingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.requests = [
            Request(0, frozenset({0, 1})), Request(1, frozenset({0, 1})),
            Request(2, frozenset({2, 3})), Request(3, frozenset({2, 3})),
        ]

    def test_max_overlap_minimizes_union_on_clustered_fixture(self) -> None:
        best = pair_greedy(self.requests, maximize=True)
        worst = pair_greedy(self.requests, maximize=False)
        self.assertEqual(union_invocations(best), 4)
        self.assertEqual(union_invocations(worst), 8)

    def test_pairings_are_partitions(self) -> None:
        for pairs in (pair_random(self.requests, 3), pair_greedy(self.requests, maximize=True)):
            ids = [item.request_id for pair in pairs for item in pair]
            self.assertEqual(sorted(ids), [0, 1, 2, 3])


if __name__ == "__main__":
    unittest.main()
