from __future__ import annotations

from collections import Counter
import struct
import unittest

from run_cross_companion_metric_replay_5090 import (
    _aggregate_numeric,
    build_cross_companion_schedule,
    difference_metrics,
    summarize_latency,
)


def _hex(index: int) -> str:
    return f"{index:064x}"


def _source_schedule(layers: int = 2):
    rows = []
    cursor = 1
    for layer in range(layers):
        for label in ("safe", "unsafe"):
            for focal_index in range(4):
                focal = _hex(cursor)
                cursor += 1
                common = {
                    "focal_row_id": focal,
                    "focal_baseline_label": label,
                    "focal_original_slot": focal_index % 2,
                    "focal_old_m1_reference_sha256": _hex(cursor),
                    "original_partner_row_id": _hex(cursor + 1),
                    "original_pack_id": _hex(cursor + 2),
                    "layer": layer,
                    "expert_id": focal_index,
                }
                cursor += 3
                for partner_label in ("safe", "unsafe"):
                    for rank in (0, 1):
                        rows.append(
                            {
                                **common,
                                "partner_baseline_label": partner_label,
                                "partner_rank_within_label": rank,
                                "partner_row_id": _hex(cursor),
                            }
                        )
                        cursor += 1
    return rows


class CrossCompanionReplayTest(unittest.TestCase):
    def test_schedule_is_deterministic_and_balanced(self):
        source = _source_schedule()
        first = build_cross_companion_schedule(
            source, source_schedule_sha256="a" * 64, expected_layers=2
        )
        second = build_cross_companion_schedule(
            source, source_schedule_sha256="a" * 64, expected_layers=2
        )
        self.assertEqual(first, second)
        self.assertEqual(len(first), 32)
        self.assertEqual(len({row["focal_row_id"] for row in first}), 8)
        counts = Counter(row["companion_kind"] for row in first)
        self.assertEqual(counts["original"], 8)
        self.assertEqual(counts["safe_rank0"], 8)
        self.assertEqual(counts["unsafe_rank0"], 8)
        self.assertEqual(counts["safe_rank1"], 4)
        self.assertEqual(counts["unsafe_rank1"], 4)
        for row in first:
            self.assertEqual(row["row_ids"][row["focal_original_slot"]], row["focal_row_id"])

    def test_difference_metrics_use_strict_bf16_storage(self):
        # 1.0, -0.0, 2.0 versus 1.5, +0.0, 1.0.
        reference = struct.pack("<HHH", 0x3F80, 0x8000, 0x4000)
        observed = struct.pack("<HHH", 0x3FC0, 0x0000, 0x3F80)
        metrics = difference_metrics(reference, observed)
        self.assertEqual(metrics["differing_count"], 3)
        self.assertEqual(metrics["max_abs"], 1.0)
        self.assertEqual(metrics["l1"], 1.5)
        self.assertAlmostEqual(metrics["l2"], (1.25) ** 0.5)

    def test_latency_summary_is_paired(self):
        m1 = [4.0 + index * 0.1 for index in range(10)]
        m2 = [2.0 + index * 0.05 for index in range(10)]
        summary = summarize_latency(m1, m2)
        self.assertEqual(summary["median_paired_m1_over_m2"], 2.0)
        self.assertEqual(summary["ratio_of_medians_m1_over_m2"], 2.0)

    def test_aggregate_reports_focal_consistency_and_bitwise_stability(self):
        numeric = []
        for focal_index in range(64):
            label = "safe" if focal_index % 2 == 0 else "unsafe"
            for companion_index, companion_label in enumerate(
                ("safe", "unsafe", "safe", "unsafe")
            ):
                numeric.append(
                    {
                        "focal_row_id": _hex(focal_index + 1),
                        "focal_baseline_label": label,
                        "resolved_companion_baseline_label": companion_label,
                        "companion_kind": f"kind_{companion_index}",
                        "target_label_status": label,
                        "target_label_stable_10_of_10": True,
                        "target_output_bitwise_stable_10_of_10": True,
                        "target_vs_m1_repeat_metrics": [
                            {
                                "differing_count": int(label == "unsafe"),
                                "max_abs": float(label == "unsafe"),
                                "l1": float(label == "unsafe"),
                                "l2": float(label == "unsafe"),
                            }
                            for _ in range(10)
                        ],
                    }
                )
        result = _aggregate_numeric(numeric)
        self.assertEqual(result["focal_consistency"]["consistent_4_of_4_focals"], 64)
        self.assertEqual(result["baseline_flips"]["call_count"], 0)
        self.assertEqual(result["bitwise_unstable_output_calls"], 0)
        self.assertIsNotNone(result["bernoulli_variance"])


if __name__ == "__main__":
    unittest.main()
