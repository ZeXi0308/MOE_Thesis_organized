from __future__ import annotations

from dataclasses import replace
import json
import sys
from pathlib import Path
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from routeslack_protocol import (  # noqa: E402
    CompletionSet,
    ContributionIdentity,
    DeadlineState,
    EnergyTrial,
    Gate0Evidence,
    IdentityLedger,
    OnlineObservation,
    OracleInput,
    ProtocolError,
    ServiceEnergySurface,
    SurfacePoint,
    ThermalPair,
    assert_ab_ba_pairs,
    assert_cached_decode_equivalence,
    assert_matched_completion,
    counter_delta,
    evaluate_gate0,
    no_op_tax_ratio,
    normalize_trial,
    paired_bootstrap_mean_ci,
    run_online_policy,
)
from run_routeslack_dry_run import _environment, run_dry_run  # noqa: E402


def identity(*, slot: int = 0, expert: int = 3) -> ContributionIdentity:
    return ContributionIdentity(
        request_id="request-0",
        input_event_id="document-0",
        token_id=11,
        decode_step=2,
        layer_id=1,
        expert_id=expert,
        topk_slot=slot,
        source_rank=0,
        target_replica=1,
    )


class IdentityConservationTest(unittest.TestCase):
    def test_all_four_stages_conserve_exact_contributions(self) -> None:
        contributions = (identity(slot=0, expert=3), identity(slot=1, expert=7))
        ledger = IdentityLedger(expected_top_k=2)
        for stage in ("routed", "dispatched", "executed", "combined"):
            ledger.record(stage, contributions)
        ledger.assert_conserved()

    def test_duplicate_or_missing_contribution_fails_closed(self) -> None:
        contributions = (identity(slot=0, expert=3), identity(slot=1, expert=7))
        ledger = IdentityLedger(expected_top_k=2)
        ledger.record("routed", contributions)
        ledger.record("dispatched", contributions)
        with self.assertRaisesRegex(ProtocolError, "top-k slots"):
            ledger.record("executed", contributions[:1])

        changed = IdentityLedger(expected_top_k=2)
        changed.record("routed", contributions)
        changed.record("dispatched", contributions)
        changed.record("executed", contributions)
        changed.record(
            "combined",
            (contributions[0], replace(contributions[1], target_replica=0)),
        )
        with self.assertRaisesRegex(ProtocolError, "identity conservation"):
            changed.assert_conserved()

        duplicate = IdentityLedger(expected_top_k=2)
        with self.assertRaisesRegex(ProtocolError, "duplicate"):
            duplicate.record("routed", (contributions[0], contributions[0]))

    def test_topk_slot_must_be_complete_and_unique(self) -> None:
        duplicate_slot = (identity(slot=0, expert=3), identity(slot=0, expert=7))
        ledger = IdentityLedger(expected_top_k=2)
        with self.assertRaisesRegex(ProtocolError, "top-k slots"):
            ledger.record("routed", duplicate_slot)


class EnvironmentProvenanceTest(unittest.TestCase):
    def test_capability_probe_never_claims_scientific_measurement(self) -> None:
        environment = _environment()
        self.assertIn("environment capability probe only", environment["evidence_label"])
        self.assertIn("cuda_available", environment["torch"])
        if environment["torch"]["cuda_available"]:
            self.assertIsInstance(environment["gpu"], dict)
            self.assertIn("name", environment["gpu"])
        else:
            self.assertIsNone(environment["gpu"])
        self.assertIsInstance(environment["nvml"], dict)
        self.assertIn("available", environment["nvml"])


class DecodeAndDeadlineTest(unittest.TestCase):
    def test_cached_decode_logits_equal_full_recomputation(self) -> None:
        # Audit fixture only: the production Gate still requires a native MoE backend.
        cached = ((0.1, 0.2), (0.3, 0.4), (0.5, 0.6))
        full = ((0.1, 0.2), (0.3, 0.4), (0.5, 0.6 + 1e-9))
        assert_cached_decode_equivalence(cached, full, atol=1e-8, rtol=1e-6)
        with self.assertRaisesRegex(ProtocolError, "cache equivalence"):
            assert_cached_decode_equivalence(cached, ((0.1, 0.2),), atol=1e-8, rtol=1e-6)

    def test_slack_updates_monotonically_with_time(self) -> None:
        state = DeadlineState(deadline_ns=1_000, now_ns=100, predicted_remaining_ns=300)
        self.assertEqual(state.slack_ns, 600)
        later = state.advance(now_ns=250, predicted_remaining_ns=350)
        self.assertEqual(later.slack_ns, 400)
        with self.assertRaisesRegex(ProtocolError, "backwards"):
            later.advance(now_ns=249, predicted_remaining_ns=1)


class SurfaceAndIsolationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.surface = ServiceEnergySurface(
            (
                SurfacePoint(rows=1, tier="default", latency_us=10.0, raw_energy_j=1.0),
                SurfacePoint(rows=4, tier="default", latency_us=20.0, raw_energy_j=2.0),
                SurfacePoint(rows=1, tier="low", latency_us=15.0, raw_energy_j=0.8),
                SurfacePoint(rows=4, tier="low", latency_us=30.0, raw_energy_j=1.6),
            ),
            default_tier="default",
        )

    def test_exact_and_conservative_ceiling_lookup(self) -> None:
        exact = self.surface.lookup(rows=1, tier="low")
        self.assertEqual(exact.status, "EXACT")
        ceiling = self.surface.lookup(rows=2, tier="low")
        self.assertEqual(ceiling.status, "CONSERVATIVE_CEILING")
        self.assertEqual(ceiling.point.rows, 4)

    def test_out_of_range_or_unknown_tier_falls_back_to_default(self) -> None:
        result = self.surface.lookup(rows=99, tier="unknown")
        self.assertEqual(result.status, "FALLBACK_DEFAULT")
        self.assertFalse(result.action_eligible)
        self.assertEqual(result.point.tier, "default")

    def test_oracle_input_cannot_enter_online_policy(self) -> None:
        observation = OnlineObservation(now_ns=10, queue_depth=2, visible_rows=(1, 4))
        self.assertEqual(run_online_policy(lambda row: row.queue_depth, observation), 2)
        leaked = OracleInput(online=observation, future_arrival_ns=(20, 30))
        with self.assertRaisesRegex(ProtocolError, "future-known"):
            run_online_policy(lambda row: row.queue_depth, leaked)


class EnergyAccountingTest(unittest.TestCase):
    def test_counter_wraparound_requires_known_modulus(self) -> None:
        self.assertEqual(counter_delta(90.0, 10.0, modulus_j=100.0), 20.0)
        with self.assertRaisesRegex(ProtocolError, "wraparound"):
            counter_delta(90.0, 10.0)

    def test_repeat_denominator_removes_fixed_meter_overhead(self) -> None:
        a = normalize_trial(raw_board_energy_j=102.0, repeats=10, meter_overhead_j=2.0)
        b = normalize_trial(raw_board_energy_j=202.0, repeats=20, meter_overhead_j=2.0)
        self.assertAlmostEqual(a.work_energy_j_per_repeat, 10.0)
        self.assertAlmostEqual(b.work_energy_j_per_repeat, 10.0)
        self.assertNotEqual(a.raw_energy_j_per_repeat, b.raw_energy_j_per_repeat)

    def test_equal_work_ab_ba_pair_and_completed_identity_match(self) -> None:
        completed = CompletionSet(
            token_keys=frozenset({("r0", 0), ("r0", 1)}),
            output_sha256="a" * 64,
        )
        a = EnergyTrial("A", "pair-0", "AB", 100.0, 10, completed, 50.0, 52.0)
        b = EnergyTrial("B", "pair-0", "AB", 100.0, 10, completed, 50.5, 51.5)
        c = replace(a, pair_id="pair-1", order="BA", temperature_start_c=51.0)
        d = replace(b, pair_id="pair-1", order="BA", temperature_start_c=51.2)
        assert_ab_ba_pairs((a, b, c, d), max_temperature_delta_c=2.0)
        assert_matched_completion(a.completed, b.completed)
        self.assertAlmostEqual(
            normalize_trial(a.raw_board_energy_j, a.repeats).raw_energy_j_per_repeat,
            normalize_trial(b.raw_board_energy_j, b.repeats).raw_energy_j_per_repeat,
        )

    def test_thermal_drift_and_completion_mismatch_fail(self) -> None:
        first = CompletionSet(frozenset({("r0", 0)}), "a" * 64)
        second = CompletionSet(frozenset({("r0", 1)}), "a" * 64)
        with self.assertRaisesRegex(ProtocolError, "completed-token"):
            assert_matched_completion(first, second)
        with self.assertRaisesRegex(ProtocolError, "thermal"):
            assert_ab_ba_pairs(
                (
                    ThermalPair("p0", "AB", 40.0, 45.0),
                    ThermalPair("p1", "BA", 40.0, 45.0),
                ),
                max_temperature_delta_c=2.0,
            )

    def test_paired_bootstrap_uses_pair_as_independent_unit(self) -> None:
        point, low, high = paired_bootstrap_mean_ci(
            (1.0, 2.0, 3.0, 4.0), replicates=500, seed=7
        )
        self.assertAlmostEqual(point, 2.5)
        self.assertLessEqual(low, point)
        self.assertGreaterEqual(high, point)


class GateAndDryRunTest(unittest.TestCase):
    def test_gate0_fails_when_native_decode_or_energy_state_is_missing(self) -> None:
        evidence = Gate0Evidence.all_true()
        self.assertEqual(evaluate_gate0(evidence).status, "PASS")
        failed = replace(evidence, native_continuous_decode=False, thermal_state_logged=False)
        result = evaluate_gate0(failed)
        self.assertEqual(result.status, "FAIL")
        self.assertIn("native_continuous_decode", result.open_items)
        self.assertIn("thermal_state_logged", result.open_items)

    def test_noop_tax_is_relative_to_gross_saving(self) -> None:
        self.assertAlmostEqual(no_op_tax_ratio(default_cost=100.0, no_op_cost=101.0, proposed_cost=90.0), 0.1)
        with self.assertRaises(ProtocolError):
            no_op_tax_ratio(default_cost=100.0, no_op_cost=101.0, proposed_cost=100.0)

    def test_dry_run_is_complete_but_never_formal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = run_dry_run(Path(tmp), seed=20260728)
            self.assertEqual(result["status"], "DRY_RUN_COMPLETE")
            self.assertFalse(result["formal_result"])
            self.assertEqual(result["gate0"], "FAIL")
            required = {
                "manifest.json",
                "environment.json",
                "config.yaml",
                "commands.sh",
                "raw/contributions.jsonl",
                "raw/policy_results.jsonl",
                "processed/dry_run_summary.json",
                "figures/README.md",
                "logs/dry_run.log",
                "verdict.md",
            }
            actual = {
                str(path.relative_to(tmp))
                for path in Path(tmp).rglob("*")
                if path.is_file()
            }
            self.assertTrue(required.issubset(actual))
            manifest = json.loads((Path(tmp) / "manifest.json").read_text(encoding="utf-8"))
            self.assertNotIn("FAILED(", manifest["git_commit"])
            self.assertNotIn("FAILED(", manifest["git_status"])


if __name__ == "__main__":
    unittest.main()
