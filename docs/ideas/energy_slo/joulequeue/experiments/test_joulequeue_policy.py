from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from joulequeue_policy import (  # noqa: E402
    AmoEStylePolicy,
    FestinaLikeProfiledPolicy,
    CausalJouleQueuePolicy,
    EDFPolicy,
    FixedTimeoutPolicy,
    ImmediatePolicy,
    Job,
    JobIdentity,
    ProtocolError,
    StaticRowsPolicy,
    SurfaceCatalog,
    SurfaceCurve,
    SurfaceOutOfRange,
    SurfacePoint,
    ThroughputMuQueuePolicy,
    exact_clairvoyant_oracle,
    board_j_per_completed_token,
    millijoules_to_joules,
    milliwatts_to_watts,
    numerical_equivalence_gate,
    oracle_gate_passes,
    paired_hierarchical_bootstrap,
    schedule_metrics,
    select_calibration_only,
    simulate_causal,
    validate_full_drain,
    validate_jobs,
    validate_review_attestation,
    validate_route_closure,
)
from run_joulequeue_oracle import _load_surface, validate_formal_capabilities  # noqa: E402


def identity(index: int, *, expert: int = 0, token: int | None = None) -> JobIdentity:
    return JobIdentity("r", 0, 0, index if token is None else token, index, expert)


def job(
    index: int,
    *,
    arrival: float,
    deadline: float = 100.0,
    expert: int = 0,
    token: int | None = None,
) -> Job:
    return Job(identity(index, expert=expert, token=token), arrival, 1, deadline)


def catalog(*, second_expert: bool = False) -> SurfaceCatalog:
    points = (
        SurfacePoint(1, 1.0, 10.0),
        SurfacePoint(2, 1.2, 12.0),
        SurfacePoint(4, 1.8, 16.0),
    )
    curves = {(0, 0): SurfaceCurve(points)}
    if second_expert:
        curves[(0, 1)] = SurfaceCurve(points)
    return SurfaceCatalog(curves)


class IdentityAndSurfaceTest(unittest.TestCase):
    def test_duplicate_identity_is_rejected(self) -> None:
        one = job(0, arrival=0)
        with self.assertRaisesRegex(ProtocolError, "duplicate"):
            validate_jobs((one, one))

    def test_only_same_expert_can_coalesce(self) -> None:
        surface = catalog(second_expert=True)
        with self.assertRaisesRegex(ProtocolError, "same layer/expert"):
            surface.estimate((job(0, arrival=0), job(1, arrival=0, expert=1)))

    def test_surface_interpolates_only_inside_grid(self) -> None:
        curve = SurfaceCurve((SurfacePoint(1, 1.0, 10.0), SurfacePoint(4, 2.5, 25.0)))
        estimate = curve.estimate(2)
        self.assertTrue(estimate.interpolated)
        self.assertAlmostEqual(estimate.energy_j, 1.5)
        with self.assertRaises(SurfaceOutOfRange):
            curve.estimate(5)

    def test_route_closure_and_identity_schema(self) -> None:
        base = {
            "model_revision": "model@sha",
            "data_manifest_sha256": "a" * 64,
            "request_id": "r",
            "forward_id": 0,
            "batch_id": 0,
            "phase": "decode",
            "decode_step": 0,
            "layer_id": 0,
            "token_id": 0,
            "token_position": 0,
            "expert_id": 0,
            "sender_rank": 0,
            "receiver_rank": 0,
            "valid": True,
            "route_weight": 0.5,
            "placement_sha256": "b" * 64,
        }
        records = [dict(base, topk_slot=0), dict(base, topk_slot=1, expert_id=1)]
        validate_route_closure(records, top_k=2)
        bad = [records[0], dict(records[1], topk_slot=2)]
        with self.assertRaisesRegex(ProtocolError, "closure"):
            validate_route_closure(bad, top_k=2)
        incomplete = [dict(records[0])]
        del incomplete[0]["receiver_rank"]
        with self.assertRaisesRegex(ProtocolError, "identity-incomplete"):
            validate_route_closure(incomplete, top_k=1)

    def test_runner_accepts_pooled_surface_only_for_development(self) -> None:
        point = {
            "coalesced_energy_mean_j": 1.0,
            "coalesced_energy_ucb95_j": 1.1,
            "coalesced_latency_mean_us": 10.0,
            "coalesced_latency_ucb95_us": 11.0,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "surface.json"
            path.write_text(json.dumps({"surface": {"1": point, "2": point}}))
            surface = _load_surface(path, formal=False)
            estimate = surface.estimate((job(0, arrival=0, expert=7),))
            self.assertEqual(estimate.energy_j, 1.1)


class SchedulerAccountingTest(unittest.TestCase):
    def test_full_drain_completed_token_denominator_and_wait_idle_energy(self) -> None:
        jobs = (job(0, arrival=0, token=0), job(1, arrival=5, token=0))
        result = simulate_causal(
            jobs,
            catalog(),
            FixedTimeoutPolicy(5.0),
            idle_power_w=100.0,
            arm="timeout",
        )
        validate_full_drain(jobs, result)
        self.assertEqual(len(result.completion_us), 2)
        self.assertEqual(len(result.actions), 1)
        self.assertAlmostEqual(result.idle_energy_j, 100.0 * 17.0 / 1_000_000.0)
        metrics = schedule_metrics(jobs, result, slo_us=100.0)
        self.assertEqual(metrics.completed_tokens, 1)
        self.assertAlmostEqual(metrics.board_j_per_completed_token, result.total_board_energy_j)

    def test_total_launch_energy_does_not_double_count_busy_idle_power(self) -> None:
        total_surface = SurfaceCatalog(
            {},
            default_curve=SurfaceCurve((
                SurfacePoint(1, 1.0, 10.0),
                SurfacePoint(2, 1.2, 12.0),
            )),
            energy_basis="total_during_launch",
        )
        jobs = (job(0, arrival=0), job(1, arrival=5))
        result = simulate_causal(
            jobs,
            total_surface,
            FixedTimeoutPolicy(5),
            idle_power_w=100,
            arm="total-energy-surface",
        )
        # Surface already includes the 12 us busy window; only the 5 us
        # pre-launch wait is charged at idle power.
        self.assertAlmostEqual(result.total_board_energy_j, 1.2 + 100 * 5e-6)

    def test_causal_decision_is_invariant_to_unseen_future(self) -> None:
        policy = CausalJouleQueuePolicy(
            target_rows=4,
            min_saving_fraction=0.5,
            max_age_us=20,
            urgent_margin_us=0,
        )
        current = (job(0, arrival=0),)
        first = policy.decide(0, current, catalog())
        # Future traces are deliberately not accepted by the API.
        future_a = (job(1, arrival=1),)
        future_b = (job(2, arrival=50),)
        del future_a, future_b
        second = policy.decide(0, current, catalog())
        self.assertEqual(first, second)
        self.assertEqual(first.kind, "defer")

    def test_all_strong_causal_baselines_complete_same_jobs(self) -> None:
        jobs = (job(0, arrival=0), job(1, arrival=1), job(2, arrival=2))
        policies = (
            ImmediatePolicy(),
            FixedTimeoutPolicy(2),
            StaticRowsPolicy(2, 5),
            EDFPolicy(),
            ThroughputMuQueuePolicy(2, 5),
            AmoEStylePolicy(2, 5),
            FestinaLikeProfiledPolicy(2, 5),
        )
        expected = {item.identity for item in jobs}
        for index, policy in enumerate(policies):
            with self.subTest(policy=type(policy).__name__):
                result = simulate_causal(
                    jobs, catalog(), policy, idle_power_w=30, arm=str(index)
                )
                self.assertEqual(set(result.completion_us), expected)

    def test_full_drain_validator_rejects_missing_action(self) -> None:
        jobs = (job(0, arrival=0),)
        result = simulate_causal(
            jobs, catalog(), ImmediatePolicy(), idle_power_w=0, arm="immediate"
        )
        broken = replace(result, actions=())
        with self.assertRaisesRegex(ProtocolError, "exactly-once"):
            validate_full_drain(jobs, broken)

    def test_frozen_energy_identity_and_units(self) -> None:
        self.assertEqual(board_j_per_completed_token(1000.0, 100), 10.0)
        self.assertEqual(milliwatts_to_watts(100000.0), 100.0)
        self.assertEqual(millijoules_to_joules(1000.0), 1.0)
        with self.assertRaises(ValueError):
            board_j_per_completed_token(1.0, False)


class OracleAndStatisticsTest(unittest.TestCase):
    def test_exact_oracle_finds_known_coalesced_optimum(self) -> None:
        jobs = (job(0, arrival=0), job(1, arrival=0))
        oracle = exact_clairvoyant_oracle(jobs, catalog(), idle_power_w=0)
        self.assertEqual(len(oracle.actions), 1)
        self.assertAlmostEqual(oracle.total_board_energy_j, 1.2)
        immediate = simulate_causal(
            jobs, catalog(), ImmediatePolicy(), idle_power_w=0, arm="immediate"
        )
        self.assertAlmostEqual(immediate.total_board_energy_j, 2.0)

    def test_oracle_accounts_for_idle_wait_and_deadlines(self) -> None:
        jobs = (job(0, arrival=0, deadline=11), job(1, arrival=5, deadline=100))
        oracle = exact_clairvoyant_oracle(jobs, catalog(), idle_power_w=100)
        self.assertEqual(len(oracle.actions), 2)
        self.assertGreater(oracle.idle_energy_j, 0)

    def test_paired_bootstrap_gate(self) -> None:
        jobs = (job(0, arrival=0), job(1, arrival=0))
        baseline_result = simulate_causal(
            jobs, catalog(), ImmediatePolicy(), idle_power_w=0, arm="baseline"
        )
        candidate_result = exact_clairvoyant_oracle(jobs, catalog(), idle_power_w=0)
        baseline = schedule_metrics(jobs, baseline_result, slo_us=100)
        candidate = schedule_metrics(jobs, candidate_result, slo_us=100)
        gate = paired_hierarchical_bootstrap(
            [baseline] * 5, [candidate] * 5, replicates=200
        )
        self.assertGreaterEqual(gate.energy_improvement_lcb95, 0.10)
        self.assertTrue(oracle_gate_passes(gate))

    def test_baseline_selection_is_calibration_only(self) -> None:
        jobs = (job(0, arrival=0), job(1, arrival=0))
        immediate = schedule_metrics(
            jobs,
            simulate_causal(jobs, catalog(), ImmediatePolicy(), idle_power_w=0, arm="i"),
            slo_us=100,
        )
        oracle = schedule_metrics(
            jobs,
            exact_clairvoyant_oracle(jobs, catalog(), idle_power_w=0),
            slo_us=100,
        )
        choice = select_calibration_only(
            {"immediate": immediate, "coalesced": oracle},
            split="calibration",
            calibration_manifest_sha256="a" * 64,
            p99_limit_us=100,
            violation_limit=0,
        )
        self.assertEqual(choice.parameter, "coalesced")
        with self.assertRaisesRegex(ProtocolError, "calibration-only"):
            select_calibration_only(
                {"immediate": immediate},
                split="sealed",
                calibration_manifest_sha256="a" * 64,
                p99_limit_us=100,
                violation_limit=0,
            )


class QualityAndFormalGateTest(unittest.TestCase):
    def test_numerical_equivalence_gate(self) -> None:
        self.assertTrue(numerical_equivalence_gate([1.0, 2.0], [1.0, 2.0]).passed)
        self.assertFalse(numerical_equivalence_gate([1.0, 2.0], [1.0, 2.1]).passed)
        with self.assertRaisesRegex(ProtocolError, "row-aligned"):
            numerical_equivalence_gate([1.0], [1.0, 2.0])

    def test_signed_attestation_binds_exact_file_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.py"
            path.write_text("x = 1\n", encoding="utf-8")
            import hashlib
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            attestation = {
                "status": "SIGNED-OFF",
                "protocol_version": "joulequeue-v1",
                "file_sha256": {"source.py": digest},
            }
            validate_review_attestation(
                attestation,
                protocol_version="joulequeue-v1",
                files={"source.py": path},
            )
            path.write_text("x = 2\n", encoding="utf-8")
            with self.assertRaisesRegex(ProtocolError, "hash drift"):
                validate_review_attestation(
                    attestation,
                    protocol_version="joulequeue-v1",
                    files={"source.py": path},
                )

    def test_formal_capability_booleans_are_strict(self) -> None:
        names = (
            "identity_complete_native_route_producer",
            "native_5090_surface_producer",
            "real_board_energy_queue_executor",
            "full_dependency_replay",
        )
        validate_formal_capabilities({
            "formal_capabilities": {name: True for name in names}
        })
        with self.assertRaisesRegex(ProtocolError, "not implemented"):
            validate_formal_capabilities({
                "formal_capabilities": {name: "false" for name in names}
            })


if __name__ == "__main__":
    unittest.main()
