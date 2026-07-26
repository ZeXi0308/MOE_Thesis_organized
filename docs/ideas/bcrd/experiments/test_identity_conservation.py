from __future__ import annotations

import unittest

try:
    from .core import Contribution, ProtocolError, validate_identity_conservation
except ImportError:
    from core import Contribution, ProtocolError, validate_identity_conservation


def row(*, rank: int, expert: int) -> Contribution:
    return Contribution("m", "decode", "r0", 0, 0.0, 100.0, 0, 0, rank, expert, 0.5, 0)


class IdentityConservationTest(unittest.TestCase):
    def test_valid_topk_is_conserved(self) -> None:
        result = validate_identity_conservation([row(rank=1, expert=1), row(rank=2, expert=2)])
        self.assertEqual(result, {"contributions": 2, "tokens": 1, "requests": 1})

    def test_duplicate_identity_fails(self) -> None:
        item = row(rank=1, expert=1)
        with self.assertRaisesRegex(ProtocolError, "duplicate routed contribution"):
            validate_identity_conservation([item, item])

    def test_duplicate_expert_fails(self) -> None:
        with self.assertRaisesRegex(ProtocolError, "duplicate expert"):
            validate_identity_conservation([row(rank=1, expert=1), row(rank=2, expert=1)])

    def test_rank_gap_fails(self) -> None:
        with self.assertRaisesRegex(ProtocolError, "non-contiguous"):
            validate_identity_conservation([row(rank=1, expert=1), row(rank=3, expert=2)])


if __name__ == "__main__":
    unittest.main()
