#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import sys
import unittest


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import recompute_c8_action_transfer as independent
import run_c8_action_transfer as runner


def packed(indices: list[int], count: int = 8) -> dict:
    payload = bytearray((count + 7) // 8)
    for index in indices:
        payload[index // 8] |= 1 << (index % 8)
    return {
        "encoding": "packed-bitset-lsb0-v1",
        "num_elements": count,
        "dtype": "torch.bfloat16",
        "packed_hex": bytes(payload).hex(),
        "set_bit_count": len(indices),
        "vector_bitwise_mismatch": bool(indices),
    }


def action(
    rank: int,
    recovered: int,
    harmed: int,
    persistent: int,
    final_indices: list[int],
    final_recovered: int,
    final_harmed: int,
    final_persistent: int,
) -> dict:
    return {
        "rank": rank,
        "expert_id": rank + 10,
        "is_frozen_m1_rank": False,
        "changed_layers_vs_R": list(range(persistent + harmed)),
        "distance_vs_R": persistent + harmed,
        "route_recovered_layers": list(range(recovered)),
        "route_recovered_count": recovered,
        "route_harmed_layers": list(range(harmed)),
        "route_harmed_count": harmed,
        "route_persistent_layers": list(range(persistent)),
        "route_persistent_count": persistent,
        "route_net_reward": recovered - harmed,
        "final_logits_mismatch_vs_R": packed(final_indices),
        "final_logit_recovered_count": final_recovered,
        "final_logit_harmed_count": final_harmed,
        "final_logit_persistent_count": final_persistent,
        "final_logit_net_reward": final_recovered - final_harmed,
        "arm": {},
    }


def bind_route(item: dict, unprotected: list[int], changed: list[int]) -> dict:
    item["changed_layers_vs_R"] = changed
    item["distance_vs_R"] = len(changed)
    item.update(runner.route_decomposition(unprotected, changed))
    return item


def row_a() -> dict:
    unprotected_layers = [0, 1]
    m1 = bind_route(
        action(0, 2, 0, 0, [0, 1], 0, 0, 2), unprotected_layers, []
    )
    a0 = bind_route(
        action(0, 2, 0, 0, [0], 1, 0, 1), unprotected_layers, []
    )
    a1 = bind_route(
        action(1, 0, 1, 2, [0, 1, 2], 0, 1, 2),
        unprotected_layers,
        [0, 1, 2],
    )
    a0["is_frozen_m1_rank"] = True
    return {
        "cell_key": "doc-a|layer=00",
        "document_index": 1,
        "frozen_m1_rank": 0,
        "integrity_status": "PASS",
        "reference_arm": {"final_logits_mismatch_vs_R": packed([])},
        "unprotected_arm": {
            "distance_vs_R": 2,
            "changed_layers_vs_R": unprotected_layers,
            "final_logits_mismatch_vs_R": packed([0, 1]),
        },
        "m1_same_rank_arm": m1,
        "c8_actions": {"0": a0, "1": a1},
    }


def row_b() -> dict:
    unprotected_layers = [0]
    m1 = bind_route(
        action(1, 1, 0, 0, [3], 0, 0, 1), unprotected_layers, []
    )
    a0 = bind_route(
        action(0, 0, 0, 1, [3], 0, 0, 1), unprotected_layers, [0]
    )
    a1 = bind_route(
        action(1, 1, 0, 0, [], 1, 0, 0), unprotected_layers, []
    )
    a1["is_frozen_m1_rank"] = True
    return {
        "cell_key": "doc-b|layer=00",
        "document_index": 2,
        "frozen_m1_rank": 1,
        "integrity_status": "PASS",
        "reference_arm": {"final_logits_mismatch_vs_R": packed([])},
        "unprotected_arm": {
            "distance_vs_R": 1,
            "changed_layers_vs_R": unprotected_layers,
            "final_logits_mismatch_vs_R": packed([3]),
        },
        "m1_same_rank_arm": m1,
        "c8_actions": {"0": a0, "1": a1},
    }


def config_fixture() -> dict:
    return {
        "action_space": {"candidate_ranks": [0, 1]},
        "cohort": {"expected_unique_cells": 2, "expected_documents": 2},
        "thresholds": {"low_transfer_ratio": 0.3, "go_transfer_ratio": 0.7},
    }


class C8ActionTransferTests(unittest.TestCase):
    def test_route_decomposition_separates_recovery_and_harm(self) -> None:
        result = runner.route_decomposition([3, 5], [5, 7])
        self.assertEqual(result["route_recovered_layers"], [3])
        self.assertEqual(result["route_harmed_layers"], [7])
        self.assertEqual(result["route_persistent_layers"], [5])
        self.assertEqual(result["route_net_reward"], 0)

    def test_packed_final_logit_decomposition(self) -> None:
        result = runner.final_decomposition(packed([0, 1, 5]), packed([1, 2, 5]))
        self.assertEqual(result["final_logit_recovered_count"], 1)
        self.assertEqual(result["final_logit_harmed_count"], 1)
        self.assertEqual(result["final_logit_persistent_count"], 2)
        self.assertEqual(result["final_logit_net_reward"], 0)

    def test_primary_rank_uses_recovery_then_harm_then_rank(self) -> None:
        source = {
            "victim_id": "v",
            "layer": 0,
            "unprotected_changed_layers_vs_R": [1, 2],
            "actions": {
                "0": {"reward": 1, "changed_layers_vs_R": [2]},
                "1": {"reward": 1, "changed_layers_vs_R": [1]},
                "2": {"reward": 0, "changed_layers_vs_R": [2, 3]},
            },
        }
        self.assertEqual(runner.select_primary_m1_rank(source), 0)

    def test_summary_has_exact_random_lodo_and_go_candidate(self) -> None:
        summary = runner.summarize_rows([row_a(), row_b()], config_fixture())
        core = summary["core"]
        self.assertEqual(
            core["m1_same_rank"]["route_recovered_count"]["numerator"], 3
        )
        self.assertEqual(
            core["c8_same_rank"]["route_recovered_count"]["numerator"], 3
        )
        self.assertEqual(
            core["c8_exact_uniform_random_rank"]["route_net_reward"],
            {"numerator": 1, "denominator": 1, "value": 1.0},
        )
        self.assertEqual(summary["document_count"], 2)
        self.assertEqual(len(summary["leave_one_document_out"]), 2)
        self.assertEqual(
            summary["decision"]["gate_candidate"],
            "GO_SHAPEABI_PLUS_STABILITYBUDGET",
        )

    def test_independent_recompute_matches_runner(self) -> None:
        rows = [row_a(), row_b()]
        config = config_fixture()
        expected = runner.summarize_rows(rows, config)
        observed = independent.recompute_metrics(rows, config)
        self.assertEqual(independent.differences(expected, observed), [])

    def test_actual_oracle_ledger_closes_to_33_unique_cells(self) -> None:
        path = (
            HERE
            / "outputs/oracle_action_sweep_20260810_run01/cell_results.jsonl"
        )
        if not path.is_file():
            self.skipTest("frozen oracle ledger is unavailable")
        rows = runner.base.load_jsonl(path)
        cohort = runner.derive_source_cohort(rows, 33, 8)
        self.assertEqual(len(cohort), 33)
        self.assertEqual(
            sum(int(row["m1_route_recovered_count"]) for row in cohort), 37
        )
        self.assertEqual(
            sum(int(row["m1_route_harmed_count"]) for row in cohort), 0
        )
        manifest_path = HERE / "configs/C8_ACTION_TRANSFER_COHORT_V1.json"
        if manifest_path.is_file():
            runner.validate_cohort_manifest(
                runner.base.load_json(manifest_path), cohort, 33, 8
            )


if __name__ == "__main__":
    unittest.main()
