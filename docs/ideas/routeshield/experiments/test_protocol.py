from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

try:
    from .protocol import (
        MetricCell,
        evaluate_metric_cells,
        load_config,
        oracle_gain,
        oracle_recovery,
        proposed_residual,
        readiness_report,
        route_specific_harm,
        simple_capture,
    )
    from .schema import ProtocolError
except ImportError:
    from protocol import (
        MetricCell,
        evaluate_metric_cells,
        load_config,
        oracle_gain,
        oracle_recovery,
        proposed_residual,
        readiness_report,
        route_specific_harm,
        simple_capture,
    )
    from schema import ProtocolError


CONFIG = Path(__file__).parent / "configs" / "gate0_v1.json"


def passing_cell(model: str) -> MetricCell:
    return MetricCell(
        model=model,
        load_cell="70pct",
        traffic_class="ADV_TEXT",
        metric_name="REPLAYED_TTFT_P99",
        harm_point=0.25,
        harm_lcb=0.15,
        oracle_gain_point=0.15,
        oracle_gain_lcb=0.08,
        oracle_recovery_lcb=0.55,
        simple_capture_ucb=0.80,
        benign_goodput_loss_ucb=0.03,
        exactness_pass=True,
        queue_stable=True,
        no_drop_or_starvation=True,
        full_request_dag_exact=True,
        legal_action_space=True,
        oracle_exact=True,
    )


def negative_control_cell(model: str) -> MetricCell:
    return MetricCell(
        model=model,
        load_cell="30pct",
        traffic_class="NAT_BENIGN",
        metric_name="REPLAYED_TTFT_P99",
        harm_point=0.0,
        harm_lcb=0.0,
        oracle_gain_point=0.0,
        oracle_gain_lcb=0.0,
        oracle_recovery_lcb=0.0,
        simple_capture_ucb=0.0,
        benign_goodput_loss_ucb=0.03,
        exactness_pass=True,
        queue_stable=True,
        no_drop_or_starvation=True,
        full_request_dag_exact=True,
        legal_action_space=True,
        oracle_exact=True,
    )


def structured_control_cell(model: str) -> MetricCell:
    return replace(
        negative_control_cell(model),
        load_cell="70pct",
        traffic_class="NAT_PATHOLOGICAL",
    )


def complete_cells(models: list[str]) -> list[MetricCell]:
    return [
        *[passing_cell(model) for model in models],
        *[negative_control_cell(model) for model in models],
        *[structured_control_cell(model) for model in models],
    ]


class ProtocolTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_config(CONFIG)
        self.models = [row["key"] for row in self.config["models"]]

    def test_frozen_config_is_deliberately_blocked(self) -> None:
        report = readiness_report(self.config)
        self.assertEqual(report["status"], "BLOCKED_PROTOCOL_NOT_AUTHORIZED")
        self.assertGreater(report["unresolved_count"], 10)
        self.assertIn("formal_execution_authorized=false", report["blockers"])

    def test_empty_or_malformed_evidence_hash_is_invalid_config(self) -> None:
        for bad_value in ("", "not-a-sha"):
            bad = deepcopy(self.config)
            bad["required_evidence"]["raw_request_ledger_sha256"] = bad_value
            report = readiness_report(bad)
            self.assertEqual(report["status"], "INVALID_CONFIG")
            self.assertFalse(report["formal_result"])

    def test_authorization_status_must_agree(self) -> None:
        bad = deepcopy(self.config)
        bad["formal_execution_authorized"] = True
        report = readiness_report(bad)
        self.assertEqual(report["status"], "INVALID_CONFIG")

    def test_frozen_threshold_cannot_be_silently_weakened(self) -> None:
        bad = deepcopy(self.config)
        bad["statistics"]["thresholds"]["harm_point_min"] = 0.0
        report = readiness_report(bad)
        self.assertEqual(report["status"], "INVALID_CONFIG")

    def test_duplicate_config_json_key_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text('{"schema":"one","schema":"two"}', encoding="utf-8")
            with self.assertRaisesRegex(ProtocolError, "duplicate config JSON key"):
                load_config(path)

    def test_counterfactual_formulas_use_one_denominator(self) -> None:
        self.assertAlmostEqual(
            route_specific_harm(attack_p99=120.0, matched_benign_p99=100.0), 0.2
        )
        self.assertAlmostEqual(oracle_gain(attack_p99=120.0, oracle_p99=110.0), 1 / 12)
        self.assertAlmostEqual(
            oracle_recovery(
                attack_p99=120.0, matched_benign_p99=100.0, oracle_p99=110.0
            ),
            0.5,
        )
        self.assertAlmostEqual(
            simple_capture(attack_p99=120.0, oracle_p99=110.0, simple_p99=111.0),
            0.9,
        )
        self.assertAlmostEqual(
            proposed_residual(simple_p99=111.0, proposed_p99=110.5, oracle_p99=110.0),
            0.5 / 111.0,
        )

    def test_policy_cannot_outperform_exact_legal_oracle(self) -> None:
        with self.assertRaisesRegex(ProtocolError, "cannot outperform"):
            proposed_residual(simple_p99=111.0, proposed_p99=109.0, oracle_p99=110.0)

    def test_self_reported_summary_cannot_qualify_a100_gate(self) -> None:
        result = evaluate_metric_cells(self.config, complete_cells(self.models))
        self.assertEqual(result["status"], "UNTRUSTED_AGGREGATE_SHAPE_ONLY")
        self.assertEqual(
            result["threshold_branch"], "ALL_THRESHOLDS_PASS"
        )
        self.assertNotIn("QUALIFIED_FOR_8XA100", str(result))
        self.assertFalse(result["formal_result"])

    def test_simple_capture_at_ninety_percent_kills_complexity(self) -> None:
        cells = [passing_cell(model) for model in self.models]
        cells[0] = replace(cells[0], simple_capture_ucb=0.90)
        cells.extend(negative_control_cell(model) for model in self.models)
        cells.extend(structured_control_cell(model) for model in self.models)
        result = evaluate_metric_cells(self.config, cells)
        self.assertEqual(result["status"], "UNTRUSTED_AGGREGATE_SHAPE_ONLY")
        self.assertEqual(result["threshold_branch"], "SIMPLE_CAPTURE_THRESHOLD_FAIL")

    def test_missing_full_request_dag_is_invalid_not_negative(self) -> None:
        cells = [passing_cell(model) for model in self.models]
        cells[0] = replace(cells[0], full_request_dag_exact=False)
        cells.extend(negative_control_cell(model) for model in self.models)
        cells.extend(structured_control_cell(model) for model in self.models)
        result = evaluate_metric_cells(self.config, cells)
        self.assertEqual(result["status"], "INVALID_REQUEST_DAG")

    def test_models_cannot_be_pooled(self) -> None:
        result = evaluate_metric_cells(
            self.config,
            [passing_cell(self.models[0]), negative_control_cell(self.models[0])],
        )
        self.assertEqual(result["status"], "INVALID_ARTIFACT")

    def test_extra_unregistered_cell_is_not_silently_ignored(self) -> None:
        cells = complete_cells(self.models)
        cells.append(replace(passing_cell(self.models[0]), load_cell="50pct"))
        result = evaluate_metric_cells(self.config, cells)
        self.assertEqual(result["status"], "INVALID_ARTIFACT")

    def test_structured_false_positive_controls_are_mandatory(self) -> None:
        cells = [
            *[passing_cell(model) for model in self.models],
            *[negative_control_cell(model) for model in self.models],
        ]
        result = evaluate_metric_cells(self.config, cells)
        self.assertEqual(result["status"], "INVALID_ARTIFACT")


if __name__ == "__main__":
    unittest.main()
