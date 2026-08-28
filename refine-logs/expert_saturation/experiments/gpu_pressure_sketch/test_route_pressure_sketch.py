from __future__ import annotations

import unittest
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from route_pressure_sketch import (
    InitialScope,
    aggregate_reference,
    compact_snapshot_nbytes,
    lossless_snapshot_nbytes,
)


class RoutePressureContractTest(unittest.TestCase):
    def test_counts_summary_and_last_token_signature(self) -> None:
        # Three requests own 2, 1, and 3 flattened token rows.
        routes = np.array(
            [
                [[0, 1], [2, 3]],
                [[1, 2], [3, 4]],  # request 0 last row
                [[2, 2], [4, 4]],  # request 1 last row
                [[0, 0], [1, 1]],
                [[1, 3], [2, 4]],
                [[3, 4], [0, 2]],  # request 2 last row
            ],
            dtype=np.int32,
        )
        req_indices = np.array([0, 0, 1, 2, 2, 2], dtype=np.int64)
        query_start = np.array([0, 2, 3, 6], dtype=np.int32)
        result = aggregate_reference(routes, req_indices, query_start, num_experts=5)

        np.testing.assert_array_equal(result.load_counts.sum(axis=1), [12, 12])
        np.testing.assert_array_equal(result.layer_max_load, [3, 4])
        np.testing.assert_array_equal(result.layer_active_experts, [5, 5])
        np.testing.assert_array_equal(result.request_signature, routes[[1, 2, 5]])

    def test_invalid_expert_ids_are_not_counted_or_leaked(self) -> None:
        routes = np.array([[[0, -1]], [[4, 9]]], dtype=np.int32)
        result = aggregate_reference(
            routes,
            np.array([0, 0]),
            np.array([0, 2]),
            num_experts=5,
        )
        np.testing.assert_array_equal(result.load_counts, [[1, 0, 0, 0, 1]])
        np.testing.assert_array_equal(result.request_signature, [[[4, -1]]])

    def test_rejects_misaligned_request_mapping(self) -> None:
        routes = np.zeros((3, 1, 1), dtype=np.int32)
        with self.assertRaisesRegex(ValueError, "different ownership"):
            aggregate_reference(
                routes,
                np.array([0, 1, 0]),
                np.array([0, 1, 3]),
                num_experts=2,
            )

    def test_copy_budget_for_current_olmoe_shape(self) -> None:
        # OLMoE probe shape: B=16, L=16, top-k=8.  Normal decode has one
        # forwarded token per request.  This is a byte-accounting check, not a
        # latency claim.
        compact = compact_snapshot_nbytes(16, 16, 8)
        lossless = lossless_snapshot_nbytes(16, 16, 8)
        self.assertEqual(compact, 4_224)
        self.assertEqual(lossless, 8_320)
        self.assertLess(compact, lossless)

    def test_initial_scope_is_fail_closed(self) -> None:
        InitialScope().validate()
        with self.assertRaisesRegex(ValueError, "async scheduling"):
            InitialScope(async_scheduling=True).validate()
        with self.assertRaisesRegex(ValueError, "TP/SP"):
            InitialScope(tensor_parallel_size=2).validate()


if __name__ == "__main__":
    unittest.main()
