from __future__ import annotations

from argparse import Namespace
import unittest

try:
    from .core import Contribution, CurvePoint, ProtocolError, ReplayConfig, ServiceCatalog, simulate_assignment
    from .solve_assignment_oracle import _structured_assignment_count, solve_instance
except ImportError:
    from core import Contribution, CurvePoint, ProtocolError, ReplayConfig, ServiceCatalog, simulate_assignment
    from solve_assignment_oracle import _structured_assignment_count, solve_instance


class StructuredOracleTest(unittest.TestCase):
    def test_different_input_events_are_never_symmetry_merged(self) -> None:
        base = Contribution(
            "m", "decode", "r", 0, 0.0, 100.0, 0, 0, 1, 10, 1.0,
            input_event_id="r:event:0", decode_step=0, legal_replica_set=(0, 1),
        )
        later = Contribution(
            "m", "decode", "r", 1, 0.0, 100.0, 0, 1, 1, 11, 1.0,
            input_event_id="r:event:1", token_id=1, decode_step=1,
            legal_replica_set=(0, 1),
        )
        count, _explicit, groups = _structured_assignment_count([base, later], 2)
        self.assertEqual(count, 4)
        self.assertEqual(len(groups), 2)

    def test_non_object_contribution_fails_closed(self) -> None:
        instance = {
            "contribution_count": 2,
            "contributions": [
                Contribution(
                    "m", "decode", "r", 0, 0.0, 10.0, 0, 0, 1, 0, 1.0
                ).to_json(),
                "not-an-object",
            ],
        }
        with self.assertRaisesRegex(ProtocolError, "contribution count mismatch"):
            solve_instance(
                instance,
                ServiceCatalog({("m", 0): [CurvePoint(1, 1, 1)]}),
                Namespace(holds_us=[0.0], max_exact_states=10, remote_bytes_per_row=0),
                0.0,
            )

    def test_top16_nontrivial_oracle_is_structured_not_raw_bruteforce(self) -> None:
        contributions = [
            Contribution(
                "m",
                "decode",
                "request",
                0,
                0.0,
                1000.0,
                0,
                0,
                slot + 1,
                slot,
                1.0 / 16,
                0,
                input_event_id="request:event",
                token_id=0,
                decode_step=0,
                legal_replica_set=(0, 1),
            )
            for slot in range(16)
        ]
        instance = {
            "instance_id": "top16-single-token",
            "model": "m",
            "phase": "decode",
            "layer": 0,
            "split": "evaluation",
            "cluster_id": "request",
            "replica_count": 2,
            "contribution_count": len(contributions),
            "contributions": [item.to_json() for item in contributions],
        }
        result = solve_instance(
            instance,
            ServiceCatalog({("m", 0): [CurvePoint(1, 10, 11)]}),
            Namespace(holds_us=[0.0, 5.0], max_exact_states=1000, remote_bytes_per_row=0),
            0.0,
        )
        self.assertTrue(result["exact"])
        self.assertEqual(result["solver"], "SYMMETRY_REDUCED_EXACT_ENUMERATION")
        self.assertEqual(result["raw_assignment_states"], 2**16)
        self.assertEqual(result["structured_assignment_states"], 17)
        # Joint (replica, hold) symmetry: C(16 + 2*2 - 1, 2*2 - 1).
        self.assertEqual(result["structured_action_states"], 969)
        self.assertEqual(result["states_evaluated"], 969)

    def test_positive_singleton_hold_is_not_pruned_as_dominated(self) -> None:
        def row(request: str, expert: int, ready: float, deadline: float) -> Contribution:
            return Contribution(
                "m",
                "decode",
                request,
                expert,
                ready,
                deadline,
                0,
                0,
                1,
                expert,
                1.0,
                0,
                input_event_id=f"{request}:event",
                token_id=0,
                decode_step=0,
                route_end_us=ready,
                legal_replica_set=(0,),
            )

        contributions = [
            row("long-deadline", 0, 0.0, 100.0),
            row("short-deadline", 1, 1.0, 12.0),
        ]
        instance = {
            "instance_id": "singleton-hold-reorders-edf",
            "model": "m",
            "phase": "decode",
            "layer": 0,
            "split": "evaluation",
            "cluster_id": "two-requests",
            "replica_count": 2,
            "contribution_count": len(contributions),
            "contributions": [item.to_json() for item in contributions],
        }
        result = solve_instance(
            instance,
            ServiceCatalog({("m", 0): [CurvePoint(1, 10, 11)]}),
            Namespace(holds_us=[0.0, 5.0], max_exact_states=100, remote_bytes_per_row=0),
            0.0,
        )
        self.assertTrue(result["exact"])
        self.assertEqual(result["oracle_metrics"]["on_time"], 2)
        self.assertEqual(result["oracle_holds_us"]["replica=0,expert=0"], 5.0)
        self.assertEqual(result["oracle_holds_us"]["replica=0,expert=1"], 0.0)

    def test_reported_oracle_metrics_match_reported_representative(self) -> None:
        contributions = [
            Contribution(
                "m", "decode", request, index, 0.0, 100.0, 0, 0, 1,
                index, 1.0, 0, input_event_id=f"{request}:event",
                legal_replica_set=(0, 1),
            )
            for index, request in enumerate(("a", "b"))
        ]
        instance = {
            "instance_id": "equal-objective-representatives",
            "model": "m",
            "phase": "decode",
            "layer": 0,
            "split": "evaluation",
            "cluster_id": "a|b",
            "replica_count": 2,
            "contribution_count": 2,
            "contributions": [item.to_json() for item in contributions],
        }
        service = ServiceCatalog({("m", 0): [CurvePoint(1, 10, 11)]})
        result = solve_instance(
            instance,
            service,
            Namespace(
                holds_us=[0.0], max_exact_states=100,
                remote_bytes_per_row=0, seed=7,
            ),
            5.0,
        )
        hold_map = {
            tuple(
                int(part.split("=")[1])
                for part in key.split(",")
            ): value
            for key, value in result["oracle_holds_us"].items()
        }
        replayed = simulate_assignment(
            contributions,
            result["oracle_assignment"],
            service,
            ReplayConfig(2, remote_latency_us=5.0, hold_by_queue=hold_map),
        )
        self.assertEqual(
            replayed["request_completion_us"],
            result["oracle_metrics"]["request_completion_us"],
        )

    def test_serial_controller_slots_are_not_symmetry_merged(self) -> None:
        contributions = [
            Contribution(
                "m", "decode", "request", 0, 0.0, 21.0, 0, 0,
                slot + 1, 100 + slot, 0.5, 0,
                input_event_id="request:event", token_id=0, decode_step=0,
                topk_slot=slot, legal_replica_set=(0, 1),
            )
            for slot in range(2)
        ]
        instance = {
            "instance_id": "serial-controller-order",
            "model": "m",
            "phase": "decode",
            "layer": 0,
            "split": "evaluation",
            "cluster_id": "request",
            "replica_count": 2,
            "contribution_count": 2,
            "contributions": [item.to_json() for item in contributions],
        }
        result = solve_instance(
            instance,
            ServiceCatalog({("m", 0): [CurvePoint(1, 10, 11)]}),
            Namespace(
                holds_us=[0.0], max_exact_states=100,
                remote_bytes_per_row=0, controller_latency_us=5.0,
                seal_cost_us=0.0, launch_cost_us=0.0,
                max_batch_rows=None, seed=1,
            ),
            2.0,
        )
        self.assertTrue(result["exact"])
        self.assertEqual(result["oracle_metrics"]["on_time"], 1)
        self.assertEqual(result["oracle_assignment"], [1, 0])


if __name__ == "__main__":
    unittest.main()
