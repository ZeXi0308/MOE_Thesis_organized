import ast
import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
VERIFIER_PATH = HERE / "verify_semantic_online_observability_raw.py"
FORMAL_DIR = HERE / "outputs" / "semantic_online_observability_20260810_run01"


def _load_verifier():
    spec = importlib.util.spec_from_file_location("sfv2_raw_verifier", VERIFIER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class IndependentRawVerifierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.verifier = _load_verifier()
        cls.report = cls.verifier.verify(FORMAL_DIR)

    def test_import_boundary_excludes_primary_experiment_modules(self):
        tree = ast.parse(VERIFIER_PATH.read_text(encoding="utf-8"))
        imported = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.append(node.module or "")
        forbidden_fragments = (
            "run_semantic_online_observability_5090",
            "audit_semantic_online_observability",
            "gpu_execution",
            "executor_contract",
            "semanticfence.experiments",
        )
        self.assertFalse(any(fragment in name for name in imported for fragment in forbidden_fragments))
        self.assertFalse(self.report["primary_module_imported"])
        self.assertTrue(self.report["summary_not_trusted_as_input"])

    def test_formal_recompute_expected_decision_metrics(self):
        self.assertEqual(self.report["status"], "PASS_INDEPENDENT_RAW_RECOMPUTE")
        oracle = self.report["natural_oracle_recomputed"]
        certificate = self.report["certificate_recomputed"]
        self.assertEqual(oracle["matching"]["safe_edges"], 77)
        self.assertEqual(oracle["matching"]["matching_edges"], 77)
        self.assertEqual(oracle["matching"]["covered_vertices"], 154)
        self.assertEqual(certificate["admitted_endpoints"], 19)
        self.assertEqual(certificate["greedy_executed_pairs"], 5)
        self.assertEqual(certificate["unsafe_greedy_executed_pairs"], 4)
        self.assertEqual(self.report["mechanical_verdict"], "PIVOT_TO_SHADOW_VERIFY")

    def test_semantic_and_freeze_integrity_are_recomputed(self):
        freeze = self.report["pre_outcome_freeze"]
        self.assertEqual(freeze["pre_outcome_lock_test_outcome_count"], 0)
        self.assertEqual(freeze["test_admission_plan_test_outcome_count"], 0)
        self.assertTrue(freeze["threshold_hash_binding_exact"])
        for split in ("train", "validation", "test"):
            integrity = self.report["semantic_integrity"][split]
            self.assertTrue(integrity["native_self_noop_exact"])
            self.assertTrue(integrity["native_m1_m2_stability_exact"])
            self.assertTrue(integrity["endpoint_label_recomputed_as_not_route_topk_changed"])
            self.assertTrue(integrity["pair_safe_recomputed_as_endpoint_and"])

    def test_cli_writes_only_to_sibling_audit_location(self):
        with tempfile.TemporaryDirectory(dir=FORMAL_DIR.parent) as temp_dir:
            output = Path(temp_dir) / "RAW_LEDGER_RECOMPUTE.json"
            completed = subprocess.run(
                [
                    str(Path(__import__("sys").executable)),
                    str(VERIFIER_PATH),
                    "--formal-output-dir",
                    str(FORMAL_DIR),
                    "--output",
                    str(output),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "PASS_INDEPENDENT_RAW_RECOMPUTE")
            self.assertTrue(FORMAL_DIR.is_dir())

    def test_cli_rejects_output_inside_formal_directory(self):
        forbidden = FORMAL_DIR / "SHOULD_NOT_BE_WRITTEN.json"
        self.assertFalse(forbidden.exists())
        completed = subprocess.run(
            [
                str(Path(__import__("sys").executable)),
                str(VERIFIER_PATH),
                "--formal-output-dir",
                str(FORMAL_DIR),
                "--output",
                str(forbidden),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 2)
        self.assertIn("must not be inside", completed.stderr)
        self.assertFalse(forbidden.exists())


if __name__ == "__main__":
    unittest.main()
