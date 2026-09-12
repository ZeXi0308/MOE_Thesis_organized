import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch

from budget_adapter import POLICIES, Selector, run_policy
from run_qualification import prepare_documents
from analyze_qualification import analyze


class Cache:
    def __init__(self):
        self.values = torch.zeros((1, 1, 1, 1))

    def get_seq_length(self):
        return self.values.shape[-2]


def forward(delta):
    def invoke(token, cache):
        value = cache.values[:, :, -1:, :] * 1.01 + delta
        cache.values = torch.cat((cache.values, value), dim=-2)
        score = value.flatten()[0]
        return SimpleNamespace(past_key_values=cache,
                               logits=torch.stack((score, -score)).reshape(1, 1, 2))
    return invoke


def run(policy, references=None):
    original = Cache()
    result = run_policy(original, torch.ones((1, 32), dtype=torch.long),
        references or [torch.zeros((1, 2)) for _ in range(32)],
        high_forward=forward(1), low_forward=forward(3), policy=policy, threshold=100.0)
    return original, result


class BudgetTests(unittest.TestCase):
    def test_exact_served_and_physical_budgets_including_forced_fill(self):
        results = {}
        for policy in POLICIES:
            _, result = run(policy)
            results[policy] = dict(result, document_sha256="test-document", repeat=0)
            ledger = result["summary"]
            self.assertEqual((ledger["served_high_steps"], ledger["physical_high_calls"],
                              ledger["physical_low_calls"]), (4, 32, 32))
            if policy == "budget_capped_reactive":
                self.assertEqual(ledger["forced_high_steps"], 4)
                self.assertEqual(ledger["threshold_selected_high_steps"], 0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "config.json").write_text(json.dumps({"repeats": 1}))
            (path / "documents.json").write_text(json.dumps({"documents": [{"document_sha256": "test-document"}]}))
            (path / "COMPLETE.json").write_text(json.dumps({"status": "COMPLETE"}))
            for policy, result in results.items():
                (path / f"r00-d000-{policy}.json").write_text(json.dumps(result))
            self.assertEqual(analyze(path)["status"], "COMPLETE")
            for policy, result in results.items():
                result["summary"]["accumulated_kl"] = 0
                for row in result["steps"]:
                    row["reference_kl"] = 0
                (path / f"r00-d000-{policy}.json").write_text(json.dumps(result))
            self.assertIsNone(analyze(path)["pooled"]["fixed_minus_reactive"]["ratio_of_sums"])
            bad = results["fixed_period"]
            bad["summary"]["physical_high_calls"] = 33
            (path / "r00-d000-fixed_period.json").write_text(json.dumps(bad))
            self.assertEqual(analyze(path)["status"], "INVALID")
            (path / "r00-d000-fixed_period.json").unlink()
            self.assertEqual(analyze(path)["status"], "PARTIAL")

    def test_each_policy_advances_its_selected_cache_without_mutating_prefix(self):
        initial_a, a = run("fixed_period")
        initial_b, b = run("budget_capped_reactive")
        self.assertTrue(torch.equal(initial_a.values, initial_b.values))
        self.assertEqual(initial_a.get_seq_length(), 1)
        self.assertEqual([row["committed_cache_length"] for row in a["steps"]], list(range(2, 34)))
        self.assertNotEqual(a["steps"][1]["same_state_kl"], b["steps"][1]["same_state_kl"])

    def test_reference_outcomes_do_not_select_actions_and_threshold_is_required(self):
        _, a = run("budget_capped_reactive")
        _, b = run("budget_capped_reactive", [torch.tensor([[100.0, -100.0]]) for _ in range(32)])
        self.assertEqual([row["action"] for row in a["steps"]], [row["action"] for row in b["steps"]])
        with self.assertRaises(ValueError):
            Selector("budget_capped_reactive", float("nan"))

    def test_fresh_document_selection_rejects_overlap_duplicates_and_short_text(self):
        tokenizer = lambda text, **kwargs: {"input_ids": list(range(len(text)))}
        text = "fresh article " * 10
        digest = hashlib.sha256(text.encode()).hexdigest()
        for documents, excluded in (([text], {digest}), ([text, text], set()), (["short"], set())):
            with self.assertRaises(ValueError):
                prepare_documents(tokenizer, documents, excluded, 64)


if __name__ == "__main__":
    unittest.main()
