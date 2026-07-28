from __future__ import annotations

import unittest
from dataclasses import replace

try:
    from .core import (
        Contribution,
        ProtocolError,
        validate_identity_conservation,
        validate_stage_identity_conservation,
    )
except ImportError:
    from core import (
        Contribution,
        ProtocolError,
        validate_identity_conservation,
        validate_stage_identity_conservation,
    )


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

    def test_frozen_manifest_detects_dropped_last_slot(self) -> None:
        with self.assertRaisesRegex(ProtocolError, "expected top_k"):
            validate_identity_conservation(
                [row(rank=1, expert=1)],
                expected_top_k=2,
                expected_layer_ids=(0,),
                expected_input_events={"r0": ("r0:decode:0",)},
            )

    def test_frozen_manifest_detects_dropped_layer_and_event(self) -> None:
        event0_l0 = row(rank=1, expert=1)
        event0_l1 = replace(
            event0_l0,
            layer=1,
            layer_id=1,
            expert_id=2,
        )
        event1_l0 = replace(
            event0_l0,
            input_event_id="r0:decode:1",
            token_id=9,
            decode_step=1,
            token_position=1,
        )
        with self.assertRaisesRegex(ProtocolError, "sibling closure|layer closure"):
            validate_identity_conservation(
                [event0_l0, event0_l1, event1_l0],
                expected_top_k=1,
                expected_layer_ids=(0, 1),
                expected_input_events={"r0": ("r0:decode:0", "r0:decode:1")},
            )
        with self.assertRaisesRegex(ProtocolError, "input-event closure"):
            validate_identity_conservation(
                [event0_l0, event0_l1],
                expected_top_k=1,
                expected_layer_ids=(0, 1),
                expected_input_events={"r0": ("r0:decode:0", "r0:decode:1")},
            )

    def test_routed_dispatch_executed_combined_identity_is_exact(self) -> None:
        routed = [row(rank=1, expert=1), row(rank=2, expert=2)]
        self.assertEqual(
            validate_stage_identity_conservation(routed, routed, routed, routed),
            {"routed": 2, "dispatched": 2, "executed": 2, "combined": 2},
        )
        swapped = [routed[0], replace(routed[1], expert_id=3)]
        with self.assertRaisesRegex(ProtocolError, "identity conservation"):
            validate_stage_identity_conservation(routed, routed, swapped, routed)
        with self.assertRaisesRegex(ProtocolError, "identity conservation"):
            validate_stage_identity_conservation(routed, routed[:1], routed[:1], routed[:1])

    def test_target_replica_cannot_change_after_dispatch(self) -> None:
        routed = [row(rank=1, expert=1), row(rank=2, expert=2)]
        dispatched = [replace(item, target_replica=0) for item in routed]
        changed = [dispatched[0], replace(dispatched[1], target_replica=1)]
        with self.assertRaisesRegex(ProtocolError, "target replica changed"):
            validate_stage_identity_conservation(
                routed,
                dispatched,
                changed,
                dispatched,
                require_assigned_target=True,
            )
        with self.assertRaisesRegex(ProtocolError, "unassigned target"):
            validate_stage_identity_conservation(
                routed,
                routed,
                routed,
                routed,
                require_assigned_target=True,
            )


if __name__ == "__main__":
    unittest.main()
