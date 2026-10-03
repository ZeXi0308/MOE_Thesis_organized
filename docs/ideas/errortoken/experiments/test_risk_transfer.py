import unittest

from risk_transfer import aggregate_risks, calibration_risk, policy_point, roc_auc


class RiskTransferTest(unittest.TestCase):
    def test_calibration_risk(self) -> None:
        self.assertEqual(calibration_risk({"exact_checks": 3, "total_checks": 4}), 0.25)
        with self.assertRaises(ValueError):
            calibration_risk({"exact_checks": 5, "total_checks": 4})

    def test_aggregate_risks_weights_checks(self) -> None:
        entries = [
            {"m": 2, "exact_checks": 1, "total_checks": 2},
            {"m": 2, "exact_checks": 8, "total_checks": 8},
            {"m": 4, "exact_checks": 0, "total_checks": 3},
        ]
        risks = aggregate_risks(entries, ("m",))
        self.assertAlmostEqual(risks[(2,)], 0.1)
        self.assertEqual(risks[(4,)], 1.0)

    def test_auc_with_ties(self) -> None:
        self.assertEqual(roc_auc([0.0, 0.0, 1.0, 1.0], [0, 0, 1, 1]), 1.0)
        self.assertEqual(roc_auc([0.5, 0.5], [0, 1]), 0.5)
        self.assertIsNone(roc_auc([0.1, 0.2], [1, 1]))

    def test_policy_point_counts_fallback_launches(self) -> None:
        calls = [
            {"row_ids": ["a", "b"], "mismatch_rows": 1, "key_risk": 0.2},
            {"row_ids": ["c", "d", "e"], "mismatch_rows": 3, "key_risk": 0.8},
            {"row_ids": ["f", "g"], "mismatch_rows": 2, "key_risk": None},
        ]
        point = policy_point(calls, 0.25)
        self.assertEqual(point["admitted_calls"], 1)
        self.assertEqual(point["admitted_rows"], 2)
        self.assertEqual(point["mismatch_rows_after_policy"], 1)
        self.assertEqual(point["launch_count_proxy"], 6)
        self.assertAlmostEqual(point["launch_reduction_fraction"], 1 / 7)


if __name__ == "__main__":
    unittest.main()
