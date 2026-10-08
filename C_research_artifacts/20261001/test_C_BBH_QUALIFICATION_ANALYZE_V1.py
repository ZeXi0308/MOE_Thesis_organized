"""CPU checks for the frozen BBH qualification scoring and diagnostics."""

import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path(__file__).with_name("C_BBH_QUALIFICATION_ANALYZE_V1.py")
SPEC = importlib.util.spec_from_file_location("bbh_qualification_analyze", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def sample_rows():
    rows = []
    for task in sorted(MODULE.EXPECTED_TASKS):
        rows.append(
            {
                "task": task,
                "request_id": task,
                "gold": "A",
                "prompt_token_ids": [5, 6],
                "output_text": "The answer is (A).",
                "output_token_ids": [1, 2, 3],
                "finish_reason": "stop",
                "stop_reason": "\n\n",
                "host_elapsed_s": 0.1,
            }
        )
    return rows


class QualificationAnalyzerTests(unittest.TestCase):
    def test_historical_first_match_and_no_rescue(self):
        self.assertEqual(MODULE.extract_answer("The answer is (A). The answer is B."), ("(A)", "first_the_answer_is_period"))
        self.assertEqual(MODULE.extract_answer("The answer is A"), ("The answer is A", "whole_output_fallback"))
        self.assertEqual(MODULE.extract_answer("the answer is B."), ("B", "first_the_answer_is_period"))

    def test_exact_match_deletes_ascii_punctuation_only(self):
        self.assertEqual(MODULE.normalize_exact_match("(A)"), "a")
        self.assertEqual(MODULE.normalize_exact_match("A-B"), "ab")
        self.assertNotEqual(MODULE.normalize_exact_match("The A"), MODULE.normalize_exact_match("A"))
        self.assertNotEqual(MODULE.normalize_exact_match("A  B"), MODULE.normalize_exact_match("A B"))

    def test_periodic_suffix_is_diagnostic(self):
        tokens = [9, 8, 7] + [1, 2] * 150
        self.assertEqual(MODULE.periodic_suffix(tokens), {"period": 2, "token_count": 300})
        self.assertEqual(MODULE.trailing_identical_run([1, 2, 2, 2]), 3)
        self.assertEqual(MODULE.periodic_suffix([1, 2, 3, 4]), {"period": None, "token_count": 0})

    def test_all_27_scored_and_raw_retained(self):
        gold = {task: "A" for task in MODULE.EXPECTED_TASKS}
        rows = sample_rows()
        rows[0]["output_text"] = "The answer is B. The answer is A."
        rows[1]["output_token_ids"] = [1, 2] * 150
        report = MODULE.analyze(rows, gold)
        self.assertEqual(report["denominator"], {"expected_tasks": 27, "present_tasks": 27, "scored_tasks": 27, "correct": 26, "accuracy": 26 / 27})
        self.assertEqual(report["repetition_diagnostic"]["flagged_count"], 1)
        self.assertEqual(report["finish_reason_counts"], {"stop": 27})
        self.assertEqual(sum(report["output_token_lengths"]["bins"].values()), 27)
        self.assertEqual(report["tasks"][0]["raw_record"]["output_text"], rows[0]["output_text"])

    def test_duplicate_missing_and_wrong_gold_fail(self):
        gold = {task: "A" for task in MODULE.EXPECTED_TASKS}
        rows = sample_rows()
        with self.assertRaisesRegex(ValueError, "27 output rows"):
            MODULE.analyze(rows[:-1], gold)
        rows[1]["task"] = rows[0]["task"]
        with self.assertRaisesRegex(ValueError, "duplicate task"):
            MODULE.analyze(rows, gold)
        rows = sample_rows()
        rows[0]["gold"] = "B"
        with self.assertRaisesRegex(ValueError, "official first example"):
            MODULE.analyze(rows, gold)

    def test_official_source_is_the_frozen_first_example_set(self):
        gold = MODULE.load_official_first_gold(MODULE.BBH_SOURCE_DEFAULT)
        self.assertEqual(set(gold), MODULE.EXPECTED_TASKS)
        self.assertTrue(all(isinstance(value, str) and value for value in gold.values()))


if __name__ == "__main__":
    unittest.main()
