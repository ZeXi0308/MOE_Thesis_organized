"""Request-denominator and calibration-gate checks for LTR comparison."""
import json
from pathlib import Path
import tempfile
import unittest

from ltr_baseline_compare import compare


RESOURCES = dict(usable_gpu_blocks=4096, gpu_kv_bytes=8592031744,
    host_kv_bytes=17179869184, block_tokens=16, offload_backend="OffloadingConnector")


def row(rid, arrival, outputs, completion, *, status="completed", prompt="p"):
    return dict(request_id=rid, document_id=rid, prompt_token_ids_sha256=prompt,
        arrival_s=arrival, max_output_tokens=1024, output_token_ids=list(range(len(outputs))),
        token_times_s=outputs, completion_s=completion, status=status,
        stop_reason="length" if status == "completed" else None)


class CompareChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def manifest(self, candidate_rows, *, candidate_status="COMPLETE", receipt=None):
        base = [row("a", 0, [1, 3], 5), row("b", 1, [2, 4], 6)]
        candidate = candidate_rows
        cells = []
        for name, rows, status, role in (("eager", base, "COMPLETE", "reference"),
                                         ("ltr_t30_q10", candidate, candidate_status, "ltr_calibration")):
            folder = self.root/name
            folder.mkdir()
            (folder/"raw.json").write_text(json.dumps(dict(status=status, requests=rows,
                observation_end_s=10)))
            r = dict(RESOURCES, save_scope="selected", runtime_verified=True,
                     policy_id=name)
            if name == "ltr_t30_q10" and receipt is not None:
                r.update(receipt)
            (folder/"resource.json").write_text(json.dumps(r))
            cells.append(dict(name=name, role=role, threshold=30 if role == "ltr_calibration" else None,
                quantum=10 if role == "ltr_calibration" else None,
                save_scope="selected", raw=str(folder/"raw.json"),
                resource_receipt=str(folder/"resource.json")))
        return dict(common_horizon_s=20, ttft_limit_s=2.5, max_gap_limit_s=1.5,
            slo_front=[dict(ttft_limit_s=4, max_gap_limit_s=3)],
            physical_resources=RESOURCES, calibration_reference="eager",
            min_output_rate_ratio=.97, max_mean_flow_ratio=1.05, cells=cells)

    def test_complete_trace_selects_and_reports_per_request_harms(self):
        candidate = [row("a", 0, [1, 2], 5), row("b", 1, [2, 3], 6)]
        result = compare(self.manifest(candidate))
        self.assertEqual(result["selected_ltr"], "ltr_t30_q10")
        self.assertEqual(result["arms"]["ltr_t30_q10"]["goodput_rps"], 2/20)
        self.assertEqual(result["paired_vs_reference"]["ltr_t30_q10"]["max_gap_s"]["improved"], 2)
        self.assertEqual(result["paired_vs_reference"]["ltr_t30_q10"]["outputs"]["equal_sequence"], 2)

    def test_failed_request_is_retained_and_disqualifies(self):
        candidate = [row("a", 0, [1, 2], 5),
                     row("b", 1, [2], None, status="failed")]
        result = compare(self.manifest(candidate, candidate_status="INCOMPLETE"))
        self.assertIsNone(result["selected_ltr"])
        self.assertEqual(result["arms"]["ltr_t30_q10"]["failed"], 1)
        self.assertEqual(result["status"], "INCOMPLETE_COMPARISON")

    def test_lower_gap_cannot_hide_output_or_flow_cost(self):
        candidate = [row("a", 0, [1], 5), row("b", 1, [2, 2.5], 6)]
        result = compare(self.manifest(candidate))
        self.assertIsNone(result["selected_ltr"])
        self.assertFalse(result["calibration_candidates"][0]["eligible"])
        self.assertEqual(result["paired_vs_reference"]["ltr_t30_q10"]["outputs"]["total_delta"], -1)

    def test_cohort_and_physical_drift_rejected(self):
        candidate = [row("a", 0, [1, 2], 5, prompt="changed"),
                     row("b", 1, [2, 3], 6)]
        with self.assertRaisesRegex(ValueError, "cohort"):
            compare(self.manifest(candidate))
        self.temp.cleanup()
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        candidate[0]["prompt_token_ids_sha256"] = "p"
        with self.assertRaisesRegex(ValueError, "physical resource"):
            compare(self.manifest(candidate, receipt=dict(usable_gpu_blocks=2048)))

    def test_partial_calibration_grid_cannot_be_selected(self):
        candidate = [row("a", 0, [1, 2], 5), row("b", 1, [2, 3], 6)]
        manifest = self.manifest(candidate)
        manifest["calibration_grid"] = dict(threshold_calls=[30, 200],
            positive_allocation_call_quantum=[1, 10])
        with self.assertRaisesRegex(ValueError, "calibration grid"):
            compare(manifest)

    def test_runtime_and_policy_label_drift_rejected(self):
        candidate = [row("a", 0, [1, 2], 5), row("b", 1, [2, 3], 6)]
        with self.assertRaisesRegex(ValueError, "policy identity"):
            compare(self.manifest(candidate, receipt=dict(policy_id="eager")))
        self.temp.cleanup()
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        manifest = self.manifest(candidate)
        manifest["runtime_attestation"] = dict(scheduler_source_sha256="pinned")
        with self.assertRaisesRegex(ValueError, "runtime/workload attestation"):
            compare(manifest)

    def test_on_off_ablation_has_no_ltr_selection(self):
        candidate = [row("a", 0, [1, 2], 5), row("b", 1, [2, 3], 6)]
        manifest = self.manifest(candidate)
        manifest["cells"][1]["role"] = "ablation_treatment"
        result = compare(manifest)
        self.assertEqual(result["selected_ltr_status"], "NOT_APPLICABLE")
        self.assertIsNone(result["selected_ltr"])
        self.assertIn("ltr_t30_q10", result["paired_vs_reference"])


if __name__ == "__main__":
    unittest.main()
