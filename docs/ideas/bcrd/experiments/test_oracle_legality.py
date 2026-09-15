from __future__ import annotations

import unittest

try:
    from .core import Contribution, CurvePoint, ProtocolError, ReplayConfig, ServiceCatalog, simulate_assignment
    from .policies import BCRDPolicy, assign_online
except ImportError:
    from core import Contribution, CurvePoint, ProtocolError, ReplayConfig, ServiceCatalog, simulate_assignment
    from policies import BCRDPolicy, assign_online


def catalog() -> ServiceCatalog:
    return ServiceCatalog({("m", 0): [CurvePoint(1, 10, 11), CurvePoint(2, 13, 14), CurvePoint(4, 18, 20)]})


def rows() -> list[Contribution]:
    return [
        Contribution("m", "decode", f"r{i}", i, float(i), 100.0, 0, 0, 1, 0, 1.0, i % 2)
        for i in range(2)
    ]


class OracleLegalityTest(unittest.TestCase):
    def test_assignment_must_cover_every_contribution(self) -> None:
        with self.assertRaisesRegex(ProtocolError, "cover every"):
            simulate_assignment(rows(), [0], catalog(), ReplayConfig(2))

    def test_illegal_replica_fails(self) -> None:
        with self.assertRaisesRegex(ProtocolError, "illegal replica"):
            simulate_assignment(rows(), [0, 2], catalog(), ReplayConfig(2))

    def test_online_policy_receives_prefix_state_only(self) -> None:
        assignments = assign_online(rows(), BCRDPolicy(), catalog(), 2)
        self.assertEqual(len(assignments), 2)
        self.assertTrue(all(replica in (0, 1) for replica in assignments))

    def test_fork_join_completion_is_request_max(self) -> None:
        items = [
            Contribution("m", "decode", "joined", 0, 0.0, 100.0, 0, 0, 1, 0, 0.5, 0),
            Contribution("m", "decode", "joined", 0, 0.0, 100.0, 0, 0, 2, 1, 0.5, 0),
        ]
        result = simulate_assignment(items, [0, 0], catalog(), ReplayConfig(2))
        self.assertEqual(result["requests"], 1)
        self.assertEqual(result["launches"], 2)
        self.assertEqual(result["mean_completion_us"], 20.0)


if __name__ == "__main__":
    unittest.main()
