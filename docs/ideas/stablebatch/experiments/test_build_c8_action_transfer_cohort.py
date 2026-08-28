#!/usr/bin/env python3

from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import build_c8_action_transfer_cohort as cohort


def action_fixture(
    rank: int,
    expert_id: int,
    unprotected_layers: set[int],
    changed_layers: set[int],
    reference_hash: str,
    action_matches_reference: bool = False,
) -> dict:
    recovered = len(unprotected_layers - changed_layers)
    harmed = len(changed_layers - unprotected_layers)
    return {
        "rank": rank,
        "expert_id": expert_id,
        "changed_layers_vs_R": sorted(changed_layers),
        "distance_vs_R": len(changed_layers),
        "reward": recovered - harmed,
        "full_restoration": bool(unprotected_layers and not changed_layers),
        "arm": {
            "final_logits_sha256": reference_hash
            if action_matches_reference
            else f"action-{rank}",
        },
    }


def row_fixture(
    cell_index: int,
    victim: str,
    document_index: int,
    layer: int,
    unprotected_layers: set[int],
    changed_by_rank: list[set[int]],
    stored_primary_rank: int,
) -> dict:
    reference_hash = f"reference-{cell_index}"
    actions = {
        str(rank): action_fixture(
            rank,
            100 + rank,
            unprotected_layers,
            changed_by_rank[rank],
            reference_hash,
            action_matches_reference=not changed_by_rank[rank],
        )
        for rank in cohort.EXPECTED_RANKS
    }
    primary = actions[str(stored_primary_rank)]
    return {
        "schema_version": "fixture",
        "integrity_status": "PASS",
        "cell_id": f"cell-{cell_index:03d}",
        "cell_index": cell_index,
        "document_index": document_index,
        "victim_id": victim,
        "layer": layer,
        "flat_token_idx": 15,
        "window_token_ids": [1, 2, 3],
        "window_token_ids_sha256": f"window-{cell_index}",
        "target_hidden_sha256": f"hidden-{cell_index}",
        "target_router_logits_sha256": f"router-{cell_index}",
        "expert_ids": [100 + rank for rank in cohort.EXPECTED_RANKS],
        "gate_weights": [1.0 / (rank + 1) for rank in cohort.EXPECTED_RANKS],
        "current_layer_topk_cutoff_margin": 0.125,
        "sidecall_m_order_per_rank": [1, 64, 1, 64],
        "source_maxgate_rank": 0,
        "source_shuffled_rank": 7,
        "reference_arm": {"final_logits_sha256": reference_hash},
        "unprotected_arm": {"final_logits_sha256": f"unprotected-{cell_index}"},
        "unprotected_changed_layers_vs_R": sorted(unprotected_layers),
        "unprotected_distance_vs_R": len(unprotected_layers),
        "actions": actions,
        "forced_oracle_rank": stored_primary_rank,
        "forced_oracle_reward": primary["reward"],
        "abstaining_oracle_action": {
            "action": "protect_rank",
            "rank": stored_primary_rank,
            "reward": primary["reward"],
        },
        "selected_positive_action_confirmation": {
            "rank": stored_primary_rank,
            "status": "PASS",
            "signature_sha256": f"confirmation-{cell_index}",
            "changed_layers_vs_R": primary["changed_layers_vs_R"],
        },
    }


def fixture_rows() -> list[dict]:
    # Rank 2 wins: ranks 1/2/3 recover three old mismatches, but rank 1 harms
    # two new layers while ranks 2/3 harm one; rank 2 then wins by index.
    first = row_fixture(
        0,
        "doc-a",
        1,
        0,
        {1, 2, 3},
        [
            {2},
            {4, 5},
            {4},
            {5},
            {1, 2, 3},
            {1, 2, 3},
            {1, 2, 3},
            {1, 2, 3},
        ],
        2,
    )
    second = row_fixture(
        1,
        "doc-b",
        2,
        0,
        {7},
        [set(), {7}, {7}, {7}, {7}, {7}, {7}, {7}],
        0,
    )
    return [first, second]


def fixture_summary(rows: list[dict]) -> dict:
    positive = [row for row in rows if row["forced_oracle_reward"] > 0]
    net_sum = sum(
        action["reward"] for row in positive for action in row["actions"].values()
    )
    rank_counts = {
        str(rank): sum(row["forced_oracle_rank"] == rank for row in positive)
        for rank in cohort.EXPECTED_RANKS
    }
    return {
        "schema_version": "fixture-summary",
        "status": "COMPLETE",
        "verdict": "fixture",
        "cell_count": len(rows),
        "candidate_action_count": len(rows) * len(cohort.EXPECTED_RANKS),
        "positive_oracle_cell_count": len(positive),
        "abstaining_oracle_action_budget": len(positive),
        "abstaining_oracle_total_reward": sum(
            row["forced_oracle_reward"] for row in positive
        ),
        "budget_matched_conditional_random_expected_reward_exact": cohort.exact_fraction(
            net_sum, len(cohort.EXPECTED_RANKS)
        ),
        "positive_oracle_rank_counts": rank_counts,
    }


def write_fixture(directory: Path) -> tuple[Path, Path]:
    rows = fixture_rows()
    ledger = directory / "cell_results.jsonl"
    with ledger.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    summary = directory / "summary.json"
    summary.write_text(json.dumps(fixture_summary(rows), sort_keys=True), encoding="utf-8")
    return ledger, summary


class C8ActionTransferCohortTests(unittest.TestCase):
    def test_primary_tie_break_and_exact_random_are_recomputed(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            ledger, summary = write_fixture(Path(raw_directory))
            manifest = cohort.build_manifest(
                ledger, summary, enforce_frozen_counts=False
            )

        self.assertEqual(manifest["status"], "SEALED_PRE_C8_OUTCOMES")
        self.assertFalse(manifest["c8_outcomes_read"])
        self.assertEqual(manifest["counts"]["primary_unique_cells"], 2)
        self.assertEqual(manifest["counts"]["raw_positive_rank_actions"], 5)
        self.assertEqual(manifest["counts"]["multi_positive_cells"], 1)
        first = manifest["cells"][0]
        self.assertEqual(first["m1_primary"]["rank"], 2)
        self.assertEqual(first["m1_primary"]["route_recovered_count"], 3)
        self.assertEqual(first["m1_primary"]["route_harmed_count"], 1)
        self.assertEqual(first["m1_primary"]["selection_tuple"], [-3, 1, 2])
        self.assertEqual(len(first["m1_actions"]), 8)
        exact = manifest["m1_exact_uniform_random_rank"]
        self.assertEqual(exact["route_net_expected"], {"numerator": 1, "denominator": 1})
        self.assertEqual(len(manifest["per_document"]), 2)
        self.assertEqual(len(manifest["leave_one_document_out"]), 2)

    def test_manifest_hash_covers_exact_deterministic_payload(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            ledger, summary = write_fixture(Path(raw_directory))
            first = cohort.build_manifest(ledger, summary, enforce_frozen_counts=False)
            second = cohort.build_manifest(ledger, summary, enforce_frozen_counts=False)

        self.assertEqual(
            first["deterministic_content_sha256"],
            second["deterministic_content_sha256"],
        )
        payload = {
            key: value
            for key, value in first.items()
            if key != "deterministic_content_sha256"
        }
        self.assertEqual(
            first["deterministic_content_sha256"],
            cohort.hashlib.sha256(cohort.canonical_json_bytes(payload)).hexdigest(),
        )

    def test_builder_rejects_stored_primary_that_differs_from_frozen_rule(self) -> None:
        row = fixture_rows()[0]
        row["forced_oracle_rank"] = 0
        row["forced_oracle_reward"] = row["actions"]["0"]["reward"]
        row["abstaining_oracle_action"]["rank"] = 0
        row["abstaining_oracle_action"]["reward"] = row["actions"]["0"]["reward"]
        row["selected_positive_action_confirmation"]["rank"] = 0
        row["selected_positive_action_confirmation"]["changed_layers_vs_R"] = row[
            "actions"
        ]["0"]["changed_layers_vs_R"]
        with self.assertRaisesRegex(cohort.CohortError, "tie-break differs"):
            cohort.build_positive_cell(row)

    def test_builder_rejects_reward_not_equal_to_recovered_minus_harmed(self) -> None:
        row = fixture_rows()[0]
        row["actions"]["2"]["reward"] += 1
        with self.assertRaisesRegex(cohort.CohortError, "stored reward"):
            cohort.build_positive_cell(row)

    def test_output_is_create_only(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            output = Path(raw_directory) / "cohort.json"
            cohort.write_json_new(output, {"first": True})
            with self.assertRaises(FileExistsError):
                cohort.write_json_new(output, {"second": True})

    def test_actual_frozen_oracle_ledger_closes_if_present(self) -> None:
        root = HERE.parents[3]
        output = root / "docs/ideas/stablebatch/experiments/outputs/oracle_action_sweep_20260810_run01"
        ledger = output / "cell_results.jsonl"
        summary = output / "summary.json"
        if not ledger.is_file() or not summary.is_file():
            self.skipTest("frozen oracle evidence is not present")
        manifest = cohort.build_manifest(ledger, summary)
        self.assertEqual(manifest["counts"]["primary_unique_cells"], 33)
        self.assertEqual(manifest["counts"]["raw_positive_rank_actions"], 139)
        self.assertEqual(manifest["counts"]["multi_positive_cells"], 27)
        self.assertEqual(manifest["m1_primary_aggregate"]["route_recovered_count"], 37)
        self.assertEqual(manifest["m1_primary_aggregate"]["route_harmed_count"], 0)
        self.assertEqual(
            manifest["m1_exact_uniform_random_rank"]["route_net_expected"],
            {"numerator": 39, "denominator": 2},
        )


if __name__ == "__main__":
    unittest.main()
