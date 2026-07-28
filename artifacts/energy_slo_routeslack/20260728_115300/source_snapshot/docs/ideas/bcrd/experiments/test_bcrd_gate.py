from __future__ import annotations

from argparse import Namespace
from pathlib import Path
import tempfile
import unittest

try:
    from .core import Contribution, CurvePoint, ProtocolError, ReplayConfig, ServiceCatalog, bootstrap_mean_ci, clustered_bootstrap_mean_ci, relative_latency_gain, simulate_assignment
    from .policies import HashPolicy, LeastLoadPolicy, assign_online
    from .census_fragmentation import _require_routeslack_formal_provenance
except ImportError:
    from core import Contribution, CurvePoint, ProtocolError, ReplayConfig, ServiceCatalog, bootstrap_mean_ci, clustered_bootstrap_mean_ci, relative_latency_gain, simulate_assignment
    from policies import HashPolicy, LeastLoadPolicy, assign_online
    from census_fragmentation import _require_routeslack_formal_provenance


class CoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self.catalog = ServiceCatalog(
            {("m", 0): [CurvePoint(1, 10, 11), CurvePoint(2, 14, 15), CurvePoint(4, 20, 22)]}
        )

    def test_curve_interpolates_but_never_extrapolates(self) -> None:
        self.assertEqual(self.catalog.estimate_us("m", 0, 3), 17.0)
        with self.assertRaisesRegex(ProtocolError, "outside measured curve"):
            self.catalog.estimate_us("m", 0, 5)

    def test_consolidation_reduces_total_work(self) -> None:
        rows = [
            Contribution("m", "decode", f"r{i}", i, 0.0, 100.0, 0, i, 1, 0, 1.0, 0)
            for i in range(4)
        ]
        split = simulate_assignment(rows, [0, 0, 1, 1], self.catalog, ReplayConfig(2))
        together = simulate_assignment(rows, [0, 0, 0, 0], self.catalog, ReplayConfig(2))
        self.assertGreater(split["total_service_us"], together["total_service_us"])

    def test_hash_is_deterministic(self) -> None:
        rows = [Contribution("m", "decode", "r", 0, 0.0, 100.0, 0, i, 1, i, 1.0, 0) for i in range(4)]
        self.assertEqual(
            assign_online(rows, HashPolicy(7), self.catalog, 2),
            assign_online(rows, HashPolicy(7), self.catalog, 2),
        )

    def test_bootstrap_and_gain(self) -> None:
        point, low, high = bootstrap_mean_ci([0.1, 0.2, 0.3], replicates=100, seed=7)
        self.assertAlmostEqual(point, 0.2)
        self.assertLessEqual(low, point)
        self.assertGreaterEqual(high, point)
        self.assertAlmostEqual(relative_latency_gain(100.0, 80.0), 0.2)

    def test_clustered_bootstrap_averages_within_request_first(self) -> None:
        point, _, _ = clustered_bootstrap_mean_ci(
            [0.0, 1.0, 1.0], ["same", "same", "other"], replicates=20, seed=7
        )
        self.assertAlmostEqual(point, 0.75)


class FormalProvenanceTest(unittest.TestCase):
    def test_smoke_bypasses_formal_provenance_but_non_smoke_fails_closed(self) -> None:
        _require_routeslack_formal_provenance(
            Namespace(smoke=True, trace=["missing.csv"], service_curve="missing.csv")
        )
        with tempfile.TemporaryDirectory() as tmp:
            trace = Path(tmp) / "trace.csv"
            curve = Path(tmp) / "curve.csv"
            trace.write_text("", encoding="utf-8")
            curve.write_text("", encoding="utf-8")
            with self.assertRaisesRegex(ProtocolError, "metadata is missing"):
                _require_routeslack_formal_provenance(
                    Namespace(
                        smoke=False,
                        trace=[str(trace)],
                        service_curve=str(curve),
                    )
                )


if __name__ == "__main__":
    unittest.main()
