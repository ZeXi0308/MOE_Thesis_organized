from __future__ import annotations

import unittest

from route_share_core import (
    build_plans,
    counts_for_shape,
    paired_shape_differences,
    simple_features,
)
from run_route_share_gate0 import shape_range_ratio


class RouteShareCoreTest(unittest.TestCase):
    def test_partitions_are_positive_and_exact(self) -> None:
        for shape in ("uniform", "linear_skew", "zipf", "hot50"):
            counts = counts_for_shape(37, 8, shape)
            self.assertEqual(sum(counts), 37)
            self.assertTrue(all(value > 0 for value in counts))

    def test_grid_is_deterministic_and_valid(self) -> None:
        kwargs = dict(
            num_experts=16,
            row_grid=(16, 32),
            active_grid=(2, 4),
            shapes=("uniform", "zipf"),
            replicas=2,
            seed=7,
        )
        left = build_plans(**kwargs)
        right = build_plans(**kwargs)
        self.assertEqual(left, right)
        self.assertEqual(len(left), 16)
        for plan in left:
            plan.validate(16)
        matched: dict[tuple[int, int, int], set[tuple[int, ...]]] = {}
        for plan in left:
            matched.setdefault(
                (plan.total_rows, plan.active_count, plan.replica), set()
            ).add(plan.active_experts)
        self.assertTrue(all(len(values) == 1 for values in matched.values()))

    def test_features_include_identity_without_tenant_labels(self) -> None:
        plan = build_plans(
            num_experts=8,
            row_grid=(16,),
            active_grid=(4,),
            shapes=("uniform",),
            replicas=1,
            seed=3,
        )[0]
        features = simple_features(plan, 8)
        self.assertEqual(len(features), 5 + 8)
        self.assertEqual(sum(features[5:]), 4.0)

    def test_paired_contrast_does_not_mix_blocks(self) -> None:
        rows = [
            {"block": 0, "total_rows": 32, "active_count": 4, "replica": 0, "latency_us": 10.0},
            {"block": 0, "total_rows": 32, "active_count": 4, "replica": 0, "latency_us": 15.0},
            {"block": 1, "total_rows": 32, "active_count": 4, "replica": 0, "latency_us": 20.0},
            {"block": 1, "total_rows": 32, "active_count": 4, "replica": 0, "latency_us": 23.0},
        ]
        self.assertEqual(sorted(paired_shape_differences(rows)), [3.0, 5.0])

    def test_shape_range_excludes_identical_histograms(self) -> None:
        plans = build_plans(
            num_experts=8,
            row_grid=(8,),
            active_grid=(1, 3),
            shapes=("uniform", "hot50"),
            replicas=1,
            seed=11,
        )
        means = {}
        for plan in plans:
            if plan.active_count == 1:
                means[plan.plan_id] = 1000.0  # identical histogram, huge fake range
            else:
                means[plan.plan_id] = 100.0 if plan.shape == "uniform" else 120.0
        self.assertAlmostEqual(shape_range_ratio(plans, means), 20.0 / 110.0)


if __name__ == "__main__":
    unittest.main()
