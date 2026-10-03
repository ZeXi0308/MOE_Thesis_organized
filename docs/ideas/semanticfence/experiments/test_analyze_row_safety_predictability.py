import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np


PATH = Path(__file__).resolve().parent / "analyze_row_safety_predictability.py"
SPEC = importlib.util.spec_from_file_location("semanticfence_row_predictability", PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class RowSafetyPredictabilityTest(unittest.TestCase):
    def test_rotating_document_split_is_disjoint_and_exhaustive(self):
        documents = np.repeat(np.arange(8), 3)
        seen_test = np.zeros(documents.size, dtype=np.int64)
        for fold in range(8):
            train, validation, test = MODULE.rotating_document_split(documents, fold)
            self.assertTrue(np.all((train.astype(int) + validation + test) == 1))
            self.assertEqual(set(documents[test]), {fold})
            self.assertEqual(set(documents[validation]), {(fold + 1) % 8})
            seen_test += test
        np.testing.assert_array_equal(seen_test, np.ones(documents.size, dtype=np.int64))

    def test_threshold_is_strictly_above_all_validation_unsafe(self):
        scores = np.asarray([-2.0, 0.25, 0.25, 3.0])
        outcomes = np.asarray([False, False, True, True])
        threshold = MODULE.fail_closed_threshold(scores, outcomes)
        self.assertGreater(threshold, 0.25)
        self.assertEqual(int(np.sum((scores > threshold) & ~outcomes)), 0)

    def test_any_input_false_admission_kills_v1(self):
        gate = {
            "minimum_true_admissions": 32,
            "minimum_safe_coverage": 0.01,
            "minimum_admitted_documents": 4,
            "minimum_admitted_layer_expert_cells": 4,
            "minimum_input_over_shape_coverage_gain": 0.005,
            "any_test_false_admission": "KILL_ZERO",
            "zero_false_but_below_coverage_or_span": "KILL_COVERAGE",
            "shape_control_meets_gate_without_input_gain": "SHAPE_ONLY",
            "input_model_meets_all_gates": "SUPPORT",
        }
        shape = {
            "held_out_false_admissions": 0,
            "held_out_true_admissions": 0,
            "held_out_safe_coverage": 0.0,
            "admitted_document_count": 0,
            "admitted_layer_expert_cell_count": 0,
        }
        input_value = {
            "held_out_false_admissions": 1,
            "held_out_true_admissions": 100,
            "held_out_safe_coverage": 0.1,
            "admitted_document_count": 8,
            "admitted_layer_expert_cell_count": 20,
        }
        decision, _ = MODULE.decide(shape, input_value, gate)
        self.assertEqual(decision, "KILL_ZERO")

    def test_input_model_must_add_value_over_shape_control(self):
        gate = {
            "minimum_true_admissions": 32,
            "minimum_safe_coverage": 0.01,
            "minimum_admitted_documents": 4,
            "minimum_admitted_layer_expert_cells": 4,
            "minimum_input_over_shape_coverage_gain": 0.005,
            "any_test_false_admission": "KILL_ZERO",
            "zero_false_but_below_coverage_or_span": "KILL_COVERAGE",
            "shape_control_meets_gate_without_input_gain": "SHAPE_ONLY",
            "input_model_meets_all_gates": "SUPPORT",
        }
        base = {
            "held_out_false_admissions": 0,
            "held_out_true_admissions": 40,
            "held_out_safe_coverage": 0.02,
            "admitted_document_count": 5,
            "admitted_layer_expert_cell_count": 8,
        }
        decision, _ = MODULE.decide(base, dict(base), gate)
        self.assertEqual(decision, "SHAPE_ONLY")
        input_value = dict(base)
        input_value["held_out_true_admissions"] = 80
        input_value["held_out_safe_coverage"] = 0.03
        decision, _ = MODULE.decide(base, input_value, gate)
        self.assertEqual(decision, "SUPPORT")


if __name__ == "__main__":
    unittest.main()
