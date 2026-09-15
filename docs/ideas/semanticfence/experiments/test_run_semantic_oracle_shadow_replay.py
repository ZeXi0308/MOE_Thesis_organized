from __future__ import annotations

import unittest

from run_semantic_oracle_shadow_replay_5090 import (
    ShadowReplayError,
    build_shadow_schedule,
    compare_route_traces,
    endpoint_route_topk_safe,
    maximum_safe_matching,
    project_matching_microcost,
    require_stable_hashes,
    semantic_surface_contract,
    validate_native_noop_observations,
)


def _hex(value: int) -> str:
    return f"{value:064x}"


def _fixture(layers: int = 2):
    calls = []
    records = {}
    windows = {}
    cursor = 1
    for layer in range(layers):
        for label in ("safe", "unsafe"):
            for focal_index in range(2):
                focal = _hex(cursor)
                companion_original = _hex(cursor + 1)
                companion_safe = _hex(cursor + 2)
                companion_unsafe = _hex(cursor + 3)
                cursor += 4
                record = {
                    "schema_version": "semanticfence-row-id-v1",
                    "split": "calibration",
                    "document_sha256": _hex(1000 + layer),
                    "document_index": layer,
                    "offset": 0,
                    "token_position": focal_index,
                    "layer": layer,
                    "expert_id": focal_index,
                    "route_rank": focal_index + 1,
                    "hidden_sha256": _hex(2000 + cursor),
                }
                records[focal] = record
                window_id = _hex(3000 + layer)
                windows[(layer, record["document_sha256"], 0)] = {
                    "window_id": window_id,
                    "window_token_ids": list(range(16)),
                    "full_hidden_states_sha256": _hex(4000 + layer),
                }
                common = {
                    "layer": layer,
                    "expert_id": focal_index,
                    "focal_row_id": focal,
                    "focal_baseline_label": label,
                    "focal_original_slot": 0,
                    "focal_old_m1_reference_sha256": _hex(5000 + cursor),
                }
                calls.extend(
                    [
                        {
                            **common,
                            "companion_kind": "original",
                            "companion_row_id": companion_original,
                            "companion_baseline_label": None,
                            "row_ids": [focal, companion_original],
                        },
                        {
                            **common,
                            "companion_kind": "safe_rank0",
                            "companion_row_id": companion_safe,
                            "companion_baseline_label": "safe",
                            "row_ids": [focal, companion_safe],
                        },
                        {
                            **common,
                            "companion_kind": "unsafe_rank0",
                            "companion_row_id": companion_unsafe,
                            "companion_baseline_label": "unsafe",
                            "row_ids": [focal, companion_unsafe],
                        },
                        {
                            **common,
                            "companion_kind": "safe_rank1",
                            "companion_row_id": _hex(cursor),
                            "companion_baseline_label": "safe",
                            "row_ids": [focal, _hex(cursor)],
                        },
                    ]
                )
                cursor += 1
    row_info = {
        row_id: {
            "record": record,
            "baseline_label": "safe" if index % 2 == 0 else "unsafe",
        }
        for index, (row_id, record) in enumerate(records.items())
    }
    for call in calls:
        for row_id in call["row_ids"]:
            if row_id in row_info:
                continue
            document_index = 100 + len(row_info)
            document_sha = _hex(6000 + document_index)
            record = {
                "schema_version": "semanticfence-row-id-v1",
                "split": "calibration",
                "document_sha256": document_sha,
                "document_index": document_index,
                "offset": 0,
                "token_position": 0,
                "layer": int(call["layer"]),
                "expert_id": int(call["expert_id"]),
                "route_rank": 1,
                "hidden_sha256": _hex(7000 + document_index),
            }
            label = call.get("companion_baseline_label") or "unsafe"
            row_info[row_id] = {"record": record, "baseline_label": label}
            windows[(document_index, document_sha, 0)] = {
                "window_id": _hex(8000 + document_index),
                "window_token_ids": list(range(16)),
                "full_hidden_states_sha256": _hex(9000 + document_index),
            }
    return calls, row_info, windows


class SemanticOracleShadowReplayTest(unittest.TestCase):
    def test_oracle_b_uses_route_topk_and_reports_greedy_separately(self):
        endpoint = {
            "route_delta": {"any_ordered_topk_change": False},
            "greedy_changed": True,
        }
        self.assertTrue(endpoint_route_topk_safe(endpoint))
        endpoint["route_delta"]["any_ordered_topk_change"] = True
        self.assertFalse(endpoint_route_topk_safe(endpoint))

    def test_schedule_selects_one_target_per_layer_label_and_two_companions(self):
        calls, row_info, windows = _fixture()
        first = build_shadow_schedule(
            calls,
            row_info=row_info,
            windows=windows,
            source_schedule_sha256="a" * 64,
            expected_layers=2,
        )
        second = build_shadow_schedule(
            calls,
            row_info=row_info,
            windows=windows,
            source_schedule_sha256="a" * 64,
            expected_layers=2,
        )
        self.assertEqual(first, second)
        self.assertEqual(len(first), 8)
        self.assertEqual(len({row["focal_row_id"] for row in first}), 4)
        for focal in {row["focal_row_id"] for row in first}:
            rows = [row for row in first if row["focal_row_id"] == focal]
            self.assertEqual(
                {row["intervention_kind"] for row in rows},
                {"original_companion", "opposite_label_rank0_companion"},
            )
        for row in first:
            self.assertEqual(
                [endpoint["row_id"] for endpoint in row["endpoints"]],
                row["row_ids"],
            )
            self.assertEqual(len(row["endpoints"]), 2)

    def test_route_comparison_only_counts_downstream_layers(self):
        native = [[0, 1], [2, 3], [4, 5], [6, 7]]
        observed = [[9, 9], [8, 8], [5, 4], [6, 9]]
        result = compare_route_traces(native, observed, start_layer=2)
        self.assertEqual(result["ordered_topk_changed_layers"], [2, 3])
        self.assertEqual(result["membership_changed_layers"], [3])
        self.assertTrue(result["any_ordered_topk_change"])

    def test_side_call_repeat_stability_fails_closed(self):
        self.assertEqual(
            require_stable_hashes("side", ["a"] * 10, expected_repeats=10),
            "a",
        )
        with self.assertRaises(ShadowReplayError):
            require_stable_hashes(
                "side", ["a"] * 9 + ["b"], expected_repeats=10
            )

    def test_semantic_surface_is_m1_baseline_vs_paired_m2_treatment(self):
        contract = semantic_surface_contract()
        self.assertEqual(contract["baseline"], "fresh_M1_target_output_injected")
        self.assertEqual(contract["treatment"], "paired_M2_target_output_injected")
        self.assertEqual(contract["other_contributions"], "native")
        self.assertEqual(contract["side_call_repeats"], 10)

    def test_native_noop_requires_exact_repeated_observations(self):
        native = [{"routes": [[1, 2]], "greedy": 7}] * 2
        noop = [{"routes": [[1, 2]], "greedy": 7}] * 2
        validate_native_noop_observations(native, noop, expected_repeats=2)
        with self.assertRaises(ShadowReplayError):
            validate_native_noop_observations(
                native,
                [noop[0], {"routes": [[2, 1]], "greedy": 7}],
                expected_repeats=2,
            )

    def test_maximum_matching_and_microcost_projection(self):
        pairs = [
            {"call_index": 0, "row_ids": ["a", "b"], "semantic_safe": True},
            {"call_index": 1, "row_ids": ["c", "b"], "semantic_safe": True},
            {"call_index": 2, "row_ids": ["c", "d"], "semantic_safe": True},
        ]
        matching = maximum_safe_matching(pairs)
        self.assertEqual(matching["matching_edges"], 2)
        self.assertEqual(matching["covered_vertices"], 4)
        projection = project_matching_microcost(
            matching,
            {"m1_median_ms": 12.0, "m2_median_ms": 6.0},
        )
        self.assertEqual(projection["all_vertices_isolated_projection_ms"], 8.0)
        self.assertEqual(projection["maximum_safe_matching_projection_ms"], 4.0)


if __name__ == "__main__":
    unittest.main()
