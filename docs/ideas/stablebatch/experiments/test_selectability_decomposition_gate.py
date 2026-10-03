#!/usr/bin/env python3
"""CPU contract tests for the StableBatch Selectability Decomposition Gate."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import unittest


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
sys.path.insert(0, str(HERE))

import recompute_selectability_decomposition_gate as independent
import run_selectability_decomposition_gate as runner
import selectability_policy as policy


CONFIG_PATH = HERE / "configs" / "selectability_decomposition_gate_v1.json"
CALIBRATION_PATH = (
    REPO_ROOT
    / "docs/ideas/stablebatch/experiments/outputs/"
    "oracle_action_sweep_20260810_run01/cell_results.jsonl"
)


def load_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def synthetic_cells(rows):
    cells = []
    for index, row in enumerate(rows):
        cells.append(
            {
                "cell_identity": f"unit-manifest|{row['victim_id']}|layer={row['layer']}",
                "victim_id": str(row["victim_id"]),
                "document_index": int(row["document_index"]),
                "layer": int(row["layer"]),
                "expert_ids": list(row["expert_ids"]),
                "gate_weights": list(row["gate_weights"]),
                "current_layer_topk_cutoff_margin": float(
                    row["current_layer_topk_cutoff_margin"]
                ),
                "document_text_sha256": f"{index:064x}"[-64:],
                "token_offset": 512,
                "window_tokens": 16,
                "window_token_ids_sha256": f"{index + 1:064x}"[-64:],
            }
        )
    return cells


class SelectabilityPolicyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        cls.rows = load_jsonl(CALIBRATION_PATH)
        cls.cells = synthetic_cells(cls.rows)
        cls.lock = policy.build_preoutcome_policy_lock(
            cls.rows, cls.cells, cls.config
        )

    def test_route_decomposition_splits_recovery_and_harm(self):
        value = policy.route_decomposition([2, 3, 5], [3, 4])
        self.assertEqual(value["recovered_layers"], [2, 5])
        self.assertEqual(value["harmed_layers"], [4])
        self.assertEqual(value["persistent_layers"], [3])
        self.assertEqual(value["reward"], 1)

    def test_calibration_contract_and_fixed_models(self):
        calibration = policy.calibration_candidates(self.rows)
        self.assertEqual(len(calibration), 1920)
        self.assertEqual(self.lock["calibration_action_count"], 1920)
        self.assertEqual(self.lock["static_model"]["shrinkage_lambda"], 4.0)
        self.assertEqual(self.lock["online_ridge_model"]["alpha"], 1.0)
        self.assertEqual(len(self.lock["online_ridge_model"]["coefficients"]), 95)

    def test_every_preoutcome_policy_uses_exact_unique_budget(self):
        for name in ("static_plan", "online_plan", "shuffle_plan"):
            plan = self.lock[name]
            self.assertEqual(len(plan["ranking"]), 240)
            self.assertEqual(len(plan["selected"]), 33)
            self.assertEqual(
                len({row["cell_identity"] for row in plan["selected"]}), 33
            )
            self.assertTrue(all(row["selected"] for row in plan["selected"]))
        self.assertLessEqual(
            max(self.lock["shuffle_plan"]["selected_rank_counts"])
            - min(self.lock["shuffle_plan"]["selected_rank_counts"]),
            1,
        )

    def test_cell_identity_binds_manifest_text_window_and_layer(self):
        row = dict(self.cells[0])
        identity = policy.selectability_cell_identity(row, "a" * 64)
        for expected in (
            "manifest=" + "a" * 64,
            "text=" + row["document_text_sha256"],
            "offset=0512",
            "width=16",
            "window=" + row["window_token_ids_sha256"],
            f"layer={row['layer']:02d}",
        ):
            self.assertIn(expected, identity)

    def test_sidecall_assignment_is_preoutcome_and_balanced(self):
        cell = self.cells[0]
        assignment = runner.sidecall_assignment(cell, self.lock, self.config)
        schedule = assignment["sidecall_m_order_per_rank"]
        self.assertEqual(schedule.count(1), 3)
        self.assertEqual(schedule.count(64), 3)
        self.assertEqual(
            set(assignment["preoutcome_policy_rank"]), {"static", "online", "shuffle"}
        )

    def test_primary_and_independent_aggregation_match(self):
        rows = copy.deepcopy(self.rows)
        identities = {
            (str(cell["victim_id"]), int(cell["layer"])): str(cell["cell_identity"])
            for cell in self.cells
        }
        for row in rows:
            row["cell_identity"] = identities[(str(row["victim_id"]), int(row["layer"]))]
        primary = policy.classify_selectability(rows, self.lock, self.config)
        recomputed = independent.classify(rows, self.lock, self.config)
        self.assertEqual(primary, recomputed)
        self.assertEqual(primary["action_budget_cells"], 33)
        self.assertEqual(primary["cell_count"], 240)


if __name__ == "__main__":
    unittest.main()
