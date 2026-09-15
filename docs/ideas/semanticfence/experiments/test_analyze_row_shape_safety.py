from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from analyze_row_shape_safety import _best_nested_pooling_plan, analyze


def _record(m, row_ids, safe, *, call_index=0):
    return {
        "m": m,
        "layer": 0,
        "expert_id": 0,
        "row_ids": row_ids,
        "repeat_row_exact": [safe, safe, safe],
        "call_index": call_index,
    }


class RowShapeSafetyAnalysisTest(unittest.TestCase):
    def test_nested_pooling_prefers_call_reduction(self):
        plan = _best_nested_pooling_plan({2: 10, 4: 8, 8: 8}, (2, 4, 8))
        self.assertEqual(plan, {2: 1, 4: 0, 8: 1})

    def test_analysis_counts_and_boundary(self):
        records = [
            _record(2, ["a", "b"], [True, True], call_index=0),
            _record(2, ["c", "d"], [True, False], call_index=1),
            _record(4, ["a", "b", "c", "d"], [True, True, False, False]),
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "calibration.jsonl"
            path.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            result = analyze(path, (2, 4))

        self.assertEqual(result["per_m"]["2"]["safe_rows"], 3)
        self.assertEqual(result["m2_independence_diagnostic"]["both_safe"], 1)
        self.assertEqual(result["m2_independence_diagnostic"]["exactly_one_safe"], 1)
        self.assertEqual(result["cross_m"][0]["nested_violation_count"], 0)
        oracle = result["safe_pooling_oracle"]
        self.assertEqual(oracle["batch_counts"], {"2": 1, "4": 0})
        self.assertEqual(oracle["saved_calls"], 1)
        self.assertIn("not a latency", " ".join(result["claim_boundary"]))

    def test_repeat_instability_is_reported(self):
        record = _record(2, ["a", "b"], [True, False])
        record["repeat_row_exact"] = [[True, False], [False, False]]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "calibration.jsonl"
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            result = analyze(path, (2,))
        self.assertEqual(result["invariants"]["repeat_stability_violations"], 1)


if __name__ == "__main__":
    unittest.main()
