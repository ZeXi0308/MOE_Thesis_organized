from __future__ import annotations

from dataclasses import replace
import unittest

try:
    from .core import Contribution, ProtocolError, validate_causal_route_v3
except ImportError:
    from core import Contribution, ProtocolError, validate_causal_route_v3


def event(step: int, layer: int, *, ready: float, combine: float) -> Contribution:
    return Contribution(
        "m",
        "decode",
        "request",
        0,
        0.0,
        100.0,
        layer,
        step,
        1,
        layer,
        1.0,
        0,
        input_event_id=f"request:decode:{step}",
        token_id=step,
        decode_step=step,
        layer_id=layer,
        topk_slot=0,
        source_rank=0,
        target_replica=0,
        document_id="document",
        request_arrival_us=0.0,
        layer_ready_us=ready,
        route_end_us=ready,
        dispatch_end_us=ready,
        expert_start_us=ready,
        expert_end_us=combine,
        combine_end_us=combine,
        legal_replica_set=(0, 1),
    )


class RouteV3CausalityTest(unittest.TestCase):
    def test_formal_route_without_stage_timestamps_is_rejected(self) -> None:
        missing = replace(
            event(0, 0, ready=0, combine=1),
            dispatch_end_us=-1,
            expert_start_us=-1,
            expert_end_us=-1,
            combine_end_us=-1,
        )
        with self.assertRaisesRegex(ProtocolError, "requires dispatch"):
            validate_causal_route_v3([missing], require_observed_stages=True)

    def test_decode_step_waits_for_previous_combine(self) -> None:
        rows = [event(0, 0, ready=0, combine=10), event(1, 0, ready=9, combine=12)]
        with self.assertRaisesRegex(ProtocolError, "before prior combine"):
            validate_causal_route_v3(rows)

    def test_layer_waits_for_previous_layer_combine(self) -> None:
        rows = [event(0, 0, ready=0, combine=10), event(0, 1, ready=10, combine=20)]
        summary = validate_causal_route_v3(rows)
        self.assertEqual(summary["events"], 2)

    def test_one_decode_step_cannot_name_multiple_input_events(self) -> None:
        original = [
            event(0, 0, ready=0, combine=10),
            event(0, 1, ready=10, combine=20),
        ]
        duplicate = [
            replace(row, input_event_id="request:decode:duplicate")
            for row in original
        ]
        rows = original + duplicate
        with self.assertRaisesRegex(ProtocolError, "multiple input events"):
            validate_causal_route_v3(rows)

    def test_request_arrival_and_deadline_are_immutable(self) -> None:
        rows = [
            event(0, 0, ready=0, combine=10),
            replace(event(0, 1, ready=10, combine=20), deadline_us=101.0),
        ]
        with self.assertRaisesRegex(ProtocolError, "changes document, arrival, or deadline"):
            validate_causal_route_v3(rows)

        alias_drift = replace(
            event(0, 0, ready=2, combine=10),
            arrival_us=1.0,
        )
        with self.assertRaisesRegex(ProtocolError, "legacy arrival_us"):
            validate_causal_route_v3([alias_drift])


if __name__ == "__main__":
    unittest.main()
