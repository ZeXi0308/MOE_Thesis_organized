from __future__ import annotations

import unittest

from metrics import MetricAccumulator, SampleMetric
from run_creditreduce_p0 import (
    NLL_MARGIN,
    _walk_sha256,
    decide_p0_1,
    gate_threshold,
    hash_lines,
    paired_nll_bootstrap,
)


def _metrics(means: list[float]) -> MetricAccumulator:
    return MetricAccumulator(
        [
            SampleMetric(sample_id=index, nll_sum=value * 10, token_count=10)
            for index, value in enumerate(means)
        ]
    )


class CreditReduceDriverTest(unittest.TestCase):
    def test_recursive_manifest_hash_extraction(self) -> None:
        first = "a" * 64
        second = "b" * 64
        value = {
            "calibration": [{"sha256": first}],
            "test": {"nested": [{"sha256": second}, {"sha256": "short"}]},
        }
        self.assertEqual(_walk_sha256(value), [first, second])

    def test_hash_lines_has_terminal_newline_contract(self) -> None:
        self.assertEqual(hash_lines(["a", "b"]), hash_lines(["a", "b"]))
        self.assertNotEqual(hash_lines(["a", "b"]), hash_lines(["ab"]))

    def test_quality_three_state_logic(self) -> None:
        reference = _metrics([1.0] * 8)
        noninferior = paired_nll_bootstrap(
            _metrics([1.0] * 8), reference, 500, seed=1
        )
        failed = paired_nll_bootstrap(
            _metrics([1.0 + 2 * NLL_MARGIN] * 8), reference, 500, seed=1
        )
        mixed = paired_nll_bootstrap(
            _metrics([0.99, 1.02] * 4), reference, 500, seed=1
        )
        self.assertEqual(noninferior["quality_status"], "NONINFERIOR")
        self.assertEqual(failed["quality_status"], "QUALITY_FAIL")
        self.assertEqual(mixed["quality_status"], "INCONCLUSIVE")

    def test_opportunity_gate_three_state(self) -> None:
        self.assertEqual(
            gate_threshold({"lcb95": 0.21, "ucb95": 0.25}, 0.20), "PASS"
        )
        self.assertEqual(
            gate_threshold({"lcb95": 0.10, "ucb95": 0.19}, 0.20), "FAIL"
        )
        self.assertEqual(
            gate_threshold({"lcb95": 0.19, "ucb95": 0.21}, 0.20),
            "INCONCLUSIVE",
        )

    def test_dev_can_never_decide_p0(self) -> None:
        decision = decide_p0_1(
            "dev",
            "COMPLETE",
            {},
            {},
            {},
            True,
            {},
        )
        self.assertEqual(decision["overall"], "NOT_TESTED")


if __name__ == "__main__":
    unittest.main()

