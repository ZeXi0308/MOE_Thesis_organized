from __future__ import annotations

import sys
from pathlib import Path
import unittest


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from gate0_contracts import (  # noqa: E402
    ContributionIdentity,
    DecodeStepAudit,
    EnergyMeasurement,
    OnlineObservation,
    OracleTrace,
    SurfaceCatalog,
    SurfacePoint,
    ThermalState,
    assert_identity_conservation,
    completed_token_count,
    counter_delta_j,
    energy_summary,
    formal_gate_status,
    normalize_measurement,
    paired_energy_difference,
    require_online_observation,
    slack_ns,
    validate_abba_pairing,
    validate_cached_decode_audit,
    validate_thermal_pair,
)


def contribution(slot: int, *, replica: int = 0) -> ContributionIdentity:
    return ContributionIdentity(
        request_id="request-0",
        input_event_id="request-0:decode:0",
        token_id=0,
        decode_step=0,
        layer_id=3,
        expert_id=7 + slot,
        topk_slot=slot,
        source_rank=0,
        target_replica=replica,
    )


class DecodeCorrectnessTest(unittest.TestCase):
    def test_cached_decode_matches_full_and_advances_one_token(self) -> None:
        steps = (
            DecodeStepAudit(0, (11,), 5, 4, 5, (1.0, 2.0), (1.0, 2.0), ("a", "b")),
            DecodeStepAudit(1, (12,), 6, 5, 6, (3.0, 4.0), (3.0, 4.0), ("c", "d")),
        )
        result = validate_cached_decode_audit(
            steps, prompt_length=4, contributions_per_step=2, atol=1e-6, rtol=1e-6
        )
        self.assertEqual(result, {"decode_steps": 2, "route_contributions": 4})

    def test_cached_decode_rejects_recomputation_mismatch_or_duplicate_route(self) -> None:
        mismatch = (
            DecodeStepAudit(0, (11,), 5, 4, 5, (1.0,), (2.0,), ("a", "b")),
        )
        with self.assertRaisesRegex(RuntimeError, "logits mismatch"):
            validate_cached_decode_audit(mismatch, prompt_length=4, contributions_per_step=2)
        duplicate = (
            DecodeStepAudit(0, (11,), 5, 4, 5, (1.0,), (1.0,), ("a", "a")),
        )
        with self.assertRaisesRegex(RuntimeError, "duplicate route"):
            validate_cached_decode_audit(duplicate, prompt_length=4, contributions_per_step=2)


class IdentityAndDeadlineTest(unittest.TestCase):
    def test_routed_dispatched_executed_combined_identity_is_exact(self) -> None:
        routed = (contribution(0), contribution(1))
        result = assert_identity_conservation(
            routed=routed, dispatched=routed, executed=routed, combined=routed
        )
        self.assertEqual(result["contributions"], 2)
        broken = (contribution(0),)
        with self.assertRaisesRegex(RuntimeError, "executed"):
            assert_identity_conservation(
                routed=routed, dispatched=routed, executed=broken, combined=routed
            )

    def test_completed_token_denominator_requires_sibling_closure(self) -> None:
        routed = (contribution(0), contribution(1))
        self.assertEqual(completed_token_count(routed, routed), 1)
        with self.assertRaisesRegex(RuntimeError, "partial top-k sibling"):
            completed_token_count(routed, (contribution(0),))

    def test_completed_token_denominator_does_not_count_layers(self) -> None:
        first_layer = (contribution(0), contribution(1))
        second_layer = tuple(
            ContributionIdentity(
                request_id=row.request_id,
                input_event_id=row.input_event_id,
                token_id=row.token_id,
                decode_step=row.decode_step,
                layer_id=4,
                expert_id=row.expert_id + 10,
                topk_slot=row.topk_slot,
                source_rank=row.source_rank,
                target_replica=row.target_replica,
            )
            for row in first_layer
        )
        routed = first_layer + second_layer
        self.assertEqual(completed_token_count(routed, routed), 1)
        with self.assertRaisesRegex(RuntimeError, "partial top-k sibling"):
            completed_token_count(routed, first_layer)

    def test_deadline_slack_is_deadline_minus_now_and_remaining_work(self) -> None:
        self.assertEqual(slack_ns(deadline_ns=1_000, now_ns=600, predicted_remaining_ns=250), 150)
        self.assertEqual(slack_ns(deadline_ns=1_000, now_ns=900, predicted_remaining_ns=250), -150)


class SurfaceAndIsolationTest(unittest.TestCase):
    def test_surface_lookup_is_exact_and_out_of_range_falls_back(self) -> None:
        catalog = SurfaceCatalog(
            (
                SurfacePoint("m", 3, 7, 4, "default", 20.0, 0.8),
                SurfacePoint("m", 3, 7, 4, "low", 25.0, 0.6),
            )
        )
        self.assertEqual(catalog.lookup("m", 3, 7, 4, "low").energy_j, 0.6)
        fallback = catalog.lookup_or_fallback("m", 3, 7, 8, "low")
        self.assertEqual(fallback.action, "immediate_default")
        self.assertIn("unmeasured surface cell", fallback.reason)

    def test_oracle_trace_cannot_enter_online_policy_interface(self) -> None:
        online = OnlineObservation(100, ("ready-0",), "surface-v1")
        self.assertIs(require_online_observation(online), online)
        oracle = OracleTrace(online, ("future-arrival",), ("future-route",), (10.0,))
        with self.assertRaisesRegex(TypeError, "OracleTrace"):
            require_online_observation(oracle)


class EnergyAccountingTest(unittest.TestCase):
    def measurement(
        self,
        arm: str,
        *,
        raw_energy_j: float,
        repeats: int = 4,
        fixed_overhead_j: float = 2.0,
        completed=("t0", "t1"),
    ) -> EnergyMeasurement:
        return EnergyMeasurement(
            arm=arm,
            pair_id="pair-0",
            order_index=0,
            raw_board_energy_j=raw_energy_j,
            duration_s=1.0,
            idle_power_w=2.0,
            fixed_meter_overhead_j=fixed_overhead_j,
            inner_repeats=repeats,
            logical_work_ids=("w0", "w1"),
            completed_token_ids=tuple(completed),
        )

    def test_counter_wraparound_requires_explicit_modulus(self) -> None:
        self.assertEqual(counter_delta_j(95.0, 3.0, modulus_j=100.0), 8.0)
        with self.assertRaisesRegex(RuntimeError, "moved backwards"):
            counter_delta_j(95.0, 3.0)

    def test_repeat_denominator_removes_fixed_meter_overhead(self) -> None:
        a = self.measurement("A", raw_energy_j=42.0, repeats=4)
        b = self.measurement("B", raw_energy_j=82.0, repeats=8)
        self.assertEqual(normalize_measurement(a).work_j_per_repeat, 10.0)
        self.assertEqual(normalize_measurement(b).work_j_per_repeat, 10.0)

    def test_equal_work_and_fixed_overhead_produce_zero_paired_difference(self) -> None:
        a = self.measurement("A", raw_energy_j=42.0)
        b = self.measurement("B", raw_energy_j=42.0)
        result = paired_energy_difference(a, b, formal=True)
        self.assertEqual(result.raw_board_delta_j_per_completed_token, 0.0)
        self.assertEqual(result.work_delta_j_per_repeat, 0.0)

    def test_formal_pair_requires_equal_repeats_and_completed_identity(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "inner repeat"):
            paired_energy_difference(
                self.measurement("A", raw_energy_j=42.0, repeats=4),
                self.measurement("B", raw_energy_j=82.0, repeats=8),
                formal=True,
            )
        with self.assertRaisesRegex(RuntimeError, "completed-token identity"):
            paired_energy_difference(
                self.measurement("A", raw_energy_j=42.0),
                self.measurement("B", raw_energy_j=42.0, completed=("t0",)),
                formal=True,
            )

    def test_abba_pairing_and_idle_subtraction_sensitivity(self) -> None:
        validate_abba_pairing(("A", "B", "B", "A"))
        validate_abba_pairing(("B", "A", "A", "B"))
        with self.assertRaisesRegex(RuntimeError, "AB/BA"):
            validate_abba_pairing(("A", "A", "B", "B"))
        summary = energy_summary(
            raw_board_energy_j=100.0,
            duration_s=2.0,
            idle_power_w=20.0,
            completed_tokens=10,
        )
        self.assertEqual(summary.raw_j_per_completed_token, 10.0)
        self.assertEqual(summary.dynamic_j_per_completed_token, 6.0)

    def test_thermal_pair_gate_is_fail_closed(self) -> None:
        a = ThermalState(1, 60.0, 2_000, 1_000, 450.0, 250.0, 90.0, "0x0")
        b = ThermalState(2, 61.0, 2_000, 1_000, 450.0, 250.0, 90.0, "0x0")
        validate_thermal_pair(a, b, max_temperature_delta_c=2.0)
        with self.assertRaisesRegex(RuntimeError, "temperature"):
            validate_thermal_pair(a, ThermalState(2, 65.0, 2_000, 1_000, 450.0, 250.0, 90.0, "0x0"))

    def test_partial_success_cannot_be_marked_formal_pass(self) -> None:
        self.assertEqual(formal_gate_status({"identity": True, "energy": False}), "FAIL")
        self.assertEqual(formal_gate_status({"identity": True, "energy": True}), "PASS")
        with self.assertRaises(ValueError):
            formal_gate_status({})


if __name__ == "__main__":
    unittest.main()
