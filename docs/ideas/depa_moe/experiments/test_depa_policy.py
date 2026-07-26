from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from depa_policy import (  # noqa: E402
    DEPARollingPolicy,
    FCFSPolicy,
    ProtocolError,
    RequestSpec,
    ServiceCatalog,
    ServiceCurve,
    SurfaceOutOfRange,
    SurfacePoint,
    exact_slo_goodput_oracle,
    schedule_metrics,
    simulate_causal,
)
from run_depa_gates import (  # noqa: E402
    development_fixture,
    gate1_bottleneck_share,
    load_breakdown,
    load_episodes,
    load_surfaces,
    run_serial_gates,
)


def request(
    index: int,
    *,
    arrival: float,
    deadline: float,
    expert: int = 0,
    model: str = "m",
    cell: str = "c",
) -> RequestSpec:
    return RequestSpec(
        request_id=f"r{index}",
        model=model,
        cell=cell,
        arrival_us=arrival,
        deadline_us=deadline,
        expert_rows=((expert, 1),),
        request_class="tight" if index % 2 else "normal",
    )


def catalog() -> ServiceCatalog:
    return ServiceCatalog(
        {},
        default_curve=ServiceCurve(
            (
                SurfacePoint(1, 10.0),
                SurfacePoint(2, 12.0),
                SurfacePoint(4, 16.0),
            )
        ),
    )


class SurfaceAndLedgerTest(unittest.TestCase):
    def test_surface_is_conservative_and_never_extrapolates(self) -> None:
        curve = ServiceCurve((SurfacePoint(1, 10.0, 11.0), SurfacePoint(4, 20.0, 22.0)))
        self.assertAlmostEqual(curve.estimate_us(1), 11.0)
        self.assertAlmostEqual(curve.estimate_us(2), 11.0 + (22.0 - 11.0) / 3.0)
        with self.assertRaises(SurfaceOutOfRange):
            curve.estimate_us(5)

    def test_surface_rejects_non_monotonic_conservative_latency(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-decreasing"):
            ServiceCurve((SurfacePoint(1, 10.0), SurfacePoint(2, 9.0)))

    def test_every_offered_request_has_exactly_one_disposition(self) -> None:
        requests = (request(0, arrival=0, deadline=5), request(1, arrival=0, deadline=100))
        result = simulate_causal(
            requests,
            catalog(),
            DEPARollingPolicy(2, min_batch=1, max_wait_us=0, reject_infeasible=True),
        )
        self.assertEqual({entry.request_id for entry in result.ledger}, {"r0", "r1"})
        self.assertEqual(len(result.ledger), 2)
        self.assertIn("rejected", {entry.disposition for entry in result.ledger})

    def test_causal_policy_api_cannot_observe_future_arrivals(self) -> None:
        policy = DEPARollingPolicy(2, min_batch=2, max_wait_us=5)
        visible = (request(0, arrival=0, deadline=100),)
        first = policy.decide(0, visible, catalog())
        future_a = request(1, arrival=1, deadline=100)
        future_b = request(2, arrival=90, deadline=100)
        del future_a, future_b
        second = policy.decide(0, visible, catalog())
        self.assertEqual(first, second)
        self.assertEqual(first.kind, "defer")


class OracleTest(unittest.TestCase):
    def test_oracle_improves_known_deadline_case(self) -> None:
        requests = (
            request(0, arrival=0, deadline=13, expert=0),
            request(1, arrival=0, deadline=13, expert=0),
            request(2, arrival=0, deadline=11, expert=1),
        )
        current = simulate_causal(requests, catalog(), FCFSPolicy(1))
        oracle = exact_slo_goodput_oracle(requests, catalog(), max_batch=2)
        current_metrics = schedule_metrics(requests, current, window_start_us=0, window_end_us=100)
        oracle_metrics = schedule_metrics(requests, oracle, window_start_us=0, window_end_us=100)
        self.assertGreater(oracle_metrics.on_time, current_metrics.on_time)
        self.assertEqual(oracle_metrics.on_time, 2)

    def test_oracle_size_is_bounded(self) -> None:
        requests = tuple(request(i, arrival=0, deadline=100) for i in range(4))
        with self.assertRaisesRegex(ProtocolError, "at most 3"):
            exact_slo_goodput_oracle(requests, catalog(), max_batch=2, max_exact_requests=3)


class GateTest(unittest.TestCase):
    def test_gate1_requires_common_replicated_cell(self) -> None:
        records = (
            {"model": "a", "cell": "x", "seed": 1, "total_critical_path_us": 100, "target_exposed_us": 30},
            {"model": "b", "cell": "y", "seed": 1, "total_critical_path_us": 100, "target_exposed_us": 30},
        )
        config = {
            "required_models": ["a", "b"],
            "minimum_seeds_per_model_cell": 1,
            "bootstrap_replicates": 20,
            "bootstrap_seed": 1,
            "pass_mean_share_min": 0.2,
            "pass_lcb95_min": 0.1,
            "kill_mean_share_below": 0.1,
        }
        result = gate1_bottleneck_share(records, config)
        self.assertEqual(result["decision"], "BLOCKED_INSUFFICIENT_COMMON_CELLS")

    def test_gate1_rejects_duplicate_seed_pseudoreplication(self) -> None:
        records = (
            {"model": "a", "cell": "x", "seed": 1, "total_critical_path_us": 100, "target_exposed_us": 30},
            {"model": "a", "cell": "x", "seed": 1, "total_critical_path_us": 100, "target_exposed_us": 31},
        )
        config = {
            "required_models": ["a"],
            "minimum_seeds_per_model_cell": 1,
            "bootstrap_replicates": 20,
            "bootstrap_seed": 1,
            "pass_mean_share_min": 0.2,
            "pass_lcb95_min": 0.1,
            "kill_mean_share_below": 0.1,
        }
        with self.assertRaisesRegex(ProtocolError, "duplicate Gate 1 seed"):
            gate1_bottleneck_share(records, config)

    def test_serial_runner_stops_before_episode_evaluation_when_gate1_fails(self) -> None:
        breakdown, episodes_raw, surface_raw = development_fixture()
        for record in breakdown["records"]:
            record["target_exposed_us"] = 1.0
        config = json.loads((HERE / "configs" / "depa_v1.json").read_text())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, payload in (
                ("breakdown.json", breakdown),
                ("episodes.json", episodes_raw),
                ("surface.json", surface_raw),
            ):
                (root / name).write_text(json.dumps(payload))
            result = run_serial_gates(
                config,
                load_breakdown(root / "breakdown.json"),
                load_episodes(root / "episodes.json"),
                load_surfaces(root / "surface.json"),
                development=True,
            )
        self.assertEqual(result["overall_decision"], "STOP_AFTER_GATE_1")
        self.assertEqual(len(result["gates"]), 1)
        self.assertFalse(result["scientific_result_eligible"])

    def test_formal_mode_fails_closed_on_missing_capabilities(self) -> None:
        breakdown, episodes_raw, surface_raw = development_fixture()
        config = json.loads((HERE / "configs" / "depa_v1.json").read_text())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, payload in (
                ("breakdown.json", breakdown),
                ("episodes.json", episodes_raw),
                ("surface.json", surface_raw),
            ):
                (root / name).write_text(json.dumps(payload))
            with self.assertRaisesRegex(ProtocolError, "formal run blocked"):
                run_serial_gates(
                    config,
                    load_breakdown(root / "breakdown.json"),
                    load_episodes(root / "episodes.json"),
                    load_surfaces(root / "surface.json"),
                    development=False,
                )


if __name__ == "__main__":
    unittest.main()
