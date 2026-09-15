#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import types
import unittest

import torch
import torch.nn.functional as F


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_observable_selector_pilot as pilot  # noqa: E402


class ToyExpert(torch.nn.Module):
    def __init__(self, scale: float) -> None:
        super().__init__()
        self.scale = scale

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values * self.scale


class ToyMoe(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.num_experts = 3
        self.top_k = 2
        self.norm_topk_prob = False
        self.gate = torch.nn.Linear(4, 3, bias=False)
        with torch.no_grad():
            self.gate.weight.copy_(
                torch.tensor(
                    [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                    ]
                )
            )
        self.experts = torch.nn.ModuleList(
            [ToyExpert(1.0), ToyExpert(2.0), ToyExpert(3.0)]
        )

    def forward(self, hidden_states: torch.Tensor):
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        flat = hidden_states.view(-1, hidden_dim)
        router_logits = self.gate(flat)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(
            routing_weights, self.top_k, dim=-1
        )
        routing_weights = routing_weights.to(flat.dtype)
        final = torch.zeros_like(flat)
        mask = F.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)
        for expert_idx in range(self.num_experts):
            idx, top_x = torch.where(mask[expert_idx])
            current = flat[None, top_x].reshape(-1, hidden_dim)
            raw = self.experts[expert_idx](current)
            final.index_add_(0, top_x, raw * routing_weights[top_x, idx, None])
        return final.reshape(batch_size, sequence_length, hidden_dim), router_logits


def load_config() -> dict:
    return json.loads(
        (HERE / "configs" / "observable_selector_pilot_v1.json").read_text(
            encoding="utf-8"
        )
    )


def synthetic_cells() -> list[dict]:
    rows = []
    for victim in range(16):
        for layer in range(15):
            rows.append(
                {
                    "victim_id": f"v{victim:02d}",
                    "document_index": 16 + victim,
                    "layer": layer,
                    "gate_weights": [0.20 - rank * 0.01 for rank in range(8)],
                    "expert_ids": list(range(8)),
                    "current_layer_topk_cutoff_margin": 0.01,
                }
            )
    return rows


class ObservableSelectorTests(unittest.TestCase):
    def test_maxgate_sees_only_allowlisted_view(self) -> None:
        view = pilot.ObservableCellView(
            gate_weights=(0.1, 0.4, 0.3), expert_ids=(9, 8, 7)
        )
        self.assertEqual(pilot.select_maxgate_rank(view), 1)
        tied = pilot.ObservableCellView(
            gate_weights=(0.4, 0.4, 0.3), expert_ids=(9, 1, 7)
        )
        self.assertEqual(pilot.select_maxgate_rank(tied), 0)

    def test_assignment_ledger_is_balanced_and_order_independent(self) -> None:
        config = load_config()
        cells = synthetic_cells()
        ledger = pilot.build_assignment_ledger(cells, config)
        reversed_ledger = pilot.build_assignment_ledger(list(reversed(cells)), config)
        self.assertEqual(ledger, {**reversed_ledger, "created_at": ledger["created_at"]})
        self.assertEqual(ledger["shuffle_rank_counts"], [30] * 8)
        self.assertEqual(len(ledger["cells"]), 240)
        self.assertEqual(
            ledger["assignment_content_sha256"],
            pilot.hashlib.sha256(
                pilot.base.canonical_json_bytes(
                    {
                        key: value
                        for key, value in ledger.items()
                        if key not in {"created_at", "assignment_content_sha256"}
                    }
                )
            ).hexdigest(),
        )
        self.assertTrue(all(row["observable_rank"] == 0 for row in ledger["cells"]))
        for repeat in range(3):
            counts = {
                arm: [
                    sum(row["arm_orders_by_repeat"][repeat][position] == arm for row in ledger["cells"])
                    for position in range(4)
                ]
                for arm in pilot.ARM_LABELS
            }
            self.assertTrue(all(values == [60, 60, 60, 60] for values in counts.values()))
        self.assertEqual(
            ledger["work_signature"]["observable_surface_m_multiset"],
            ledger["work_signature"]["shuffled_surface_m_multiset"],
        )

    def test_forbidden_selector_field_fails_closed(self) -> None:
        config = load_config()
        cells = synthetic_cells()
        cells[0]["next_layer_topk_margin"] = 0.0
        with self.assertRaises(pilot.ProtocolError):
            pilot.build_assignment_ledger(cells, config)

    def test_surfaces_have_exact_action_budget(self) -> None:
        observable = pilot.surface_for_arm("O", 0, 3, 8)
        shuffled = pilot.surface_for_arm("S", 0, 3, 8)
        self.assertEqual(sorted(observable.values()), [1] + [64] * 7)
        self.assertEqual(sorted(shuffled.values()), [1] + [64] * 7)
        self.assertEqual(sum(value == 1 for value in observable.values()), 1)
        self.assertEqual(sum(value == 1 for value in shuffled.values()), 1)

    def test_multi_patch_noop_and_replaces_all_target_ranks(self) -> None:
        block = ToyMoe()
        model = types.SimpleNamespace(
            model=types.SimpleNamespace(layers=[types.SimpleNamespace(mlp=block)])
        )
        hidden = torch.tensor(
            [
                [
                    [2.0, 0.1, 0.0, 1.0],
                    [0.2, 3.0, 0.1, 1.0],
                    [0.1, 0.2, 4.0, 1.0],
                ]
            ]
        )
        native, logits = block(hidden)
        weights, experts = pilot.base.topk_from_logits(logits[1], block.top_k)
        cell = {
            "layer": 0,
            "flat_token_idx": 1,
            "expert_ids": list(map(int, experts.tolist())),
        }
        with pilot.patched_topk_contributions(model, cell, None, "self") as trace:
            noop, noop_logits = block(hidden)
        self.assertTrue(torch.equal(native, noop))
        self.assertTrue(torch.equal(logits, noop_logits))
        self.assertEqual(trace["pair_match_count_by_rank"], {"0": 1, "1": 1})
        replacements = {0: torch.zeros(4), 1: torch.ones(4)}
        with pilot.patched_topk_contributions(
            model, cell, replacements, "replacement"
        ) as changed_trace:
            changed, changed_logits = block(hidden)
        self.assertTrue(torch.equal(logits, changed_logits))
        self.assertTrue(torch.equal(native[:, 0], changed[:, 0]))
        self.assertFalse(torch.equal(native[:, 1], changed[:, 1]))
        self.assertTrue(torch.equal(native[:, 2], changed[:, 2]))
        self.assertEqual(
            changed_trace["routing_weight_apply_count_by_rank"], {"0": 1, "1": 1}
        )

    def test_frozen_verdicts(self) -> None:
        config = load_config()

        def rows(mode: str) -> list[dict]:
            result = []
            for victim in range(16):
                for layer in range(15):
                    opportunity_victim_count = 8 if mode != "unable" else 3
                    opportunity = layer == 0 and victim < opportunity_victim_count
                    observable_reward = int(layer == 0 and victim < 8)
                    shuffled_reward = int(layer == 0 and victim < 4)
                    if mode == "weaken":
                        shuffled_reward = observable_reward
                    result.append(
                        {
                            "victim_id": f"v{victim:02d}",
                            "d_unprotected_vs_R": int(opportunity),
                            "observable_reward": observable_reward,
                            "shuffled_reward": shuffled_reward,
                            "observable_full_restoration": bool(
                                opportunity and observable_reward > 0
                            ),
                            "shuffled_full_restoration": bool(
                                opportunity and shuffled_reward > 0
                            ),
                            "observable_harm": observable_reward < 0,
                            "shuffled_harm": shuffled_reward < 0,
                            "observable_shuffled_same_rank": False,
                        }
                    )
            return result

        self.assertEqual(
            pilot.classify_results(rows("support"), config)["verdict"],
            "SUPPORT_MAXGATE_V1_ACTION_VALUE",
        )
        self.assertEqual(
            pilot.classify_results(rows("weaken"), config)["verdict"],
            "WEAKENS_MAXGATE_V1_NOT_BETTER_THAN_SHUFFLE",
        )
        self.assertEqual(
            pilot.classify_results(rows("unable"), config)["verdict"],
            "UNABLE_TO_DECIDE_INSUFFICIENT_OPPORTUNITY",
        )
        negative_shuffle = rows("support")
        for row in negative_shuffle:
            row["shuffled_reward"] = -1
            row["shuffled_harm"] = True
        negative_summary = pilot.classify_results(negative_shuffle, config)
        self.assertEqual(
            negative_summary["verdict"], "SUPPORT_MAXGATE_V1_ACTION_VALUE"
        )
        self.assertEqual(negative_summary["frozen_ratio_threshold"], 8)

        below_magnitude = rows("support")
        for row in below_magnitude:
            row["shuffled_reward"] = 0
            if row["victim_id"] == "v07" and row["observable_reward"] == 1:
                row["observable_reward"] = 0
        self.assertEqual(
            pilot.classify_results(below_magnitude, config)["verdict"],
            "UNABLE_TO_DECIDE_BELOW_FROZEN_MAGNITUDE_OR_COVERAGE",
        )

        below_coverage = rows("support")
        for row in below_coverage:
            row["shuffled_reward"] = 0
            row["observable_reward"] = int(
                row["victim_id"] in {"v00", "v01", "v02"} and row["d_unprotected_vs_R"]
            ) * 3
        self.assertEqual(
            pilot.classify_results(below_coverage, config)["verdict"],
            "UNABLE_TO_DECIDE_BELOW_FROZEN_MAGNITUDE_OR_COVERAGE",
        )

        signed = rows("support")
        signed[-1]["observable_reward"] = -2
        signed[-1]["observable_harm"] = True
        signed_summary = pilot.classify_results(signed, config)
        self.assertEqual(signed_summary["observable_total_reward"], 6)
        self.assertEqual(signed_summary["observable_positive_tie_negative"]["negative"], 1)

    def test_config_freezes_heldout_window_and_scope(self) -> None:
        config = load_config()
        self.assertEqual(config["status"], "FROZEN_PRE_RUN")
        self.assertEqual(config["data"]["document_indices"], list(range(16, 32)))
        self.assertEqual(config["data"]["token_offset"], 512)
        self.assertEqual(config["selection"]["cell_count"], 240)
        self.assertEqual(config["selection"]["shuffle_rank_count_each"], 30)
        self.assertEqual(config["selection"]["allowed_signal_fields"], ["gate_weights"])
        self.assertEqual(config["selection"]["selector_identity_fields"], ["expert_ids"])
        self.assertIn("not_dynamic_controller", config["research_boundary"])

    def test_selector_lock_rejects_old_or_incomplete_bindings(self) -> None:
        config = load_config()
        valid = {
            "schema_version": pilot.LOCK_SCHEMA,
            "status": "FROZEN_PRE_RUN",
            "files": {key: "sha256" for key in pilot.expected_lock_files(config)},
            "frozen_semantics": pilot.expected_frozen_semantics(config),
            "claim_boundary": config["research_boundary"],
        }
        pilot.validate_selector_lock_document(valid, config)
        old = {**valid, "schema_version": "stablebatch-single-contribution-frozen-lock-v2"}
        with self.assertRaises(pilot.ProtocolError):
            pilot.validate_selector_lock_document(old, config)
        incomplete = {**valid, "files": {pilot.RUNNER_RELATIVE: "sha256"}}
        with self.assertRaises(pilot.ProtocolError):
            pilot.validate_selector_lock_document(incomplete, config)

    def test_success_manifest_prebinds_atomic_run_status(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as directory:
            output_dir = Path(directory)
            pilot.base.write_json_new(output_dir / "evidence.json", {"ok": True})
            status = {
                "status": "COMPLETE",
                "scientific_result_eligible": True,
            }
            pilot.base.write_json_new(
                output_dir / "MANIFEST.json",
                pilot.build_manifest(output_dir, pending_run_status=status),
            )
            self.assertFalse((output_dir / "RUN_STATUS.json").exists())
            pilot.write_bound_run_status(output_dir, status)
            pilot.verify_output_manifest(output_dir)

    def test_formal_rows_fail_closed_on_incomplete_or_duplicate_cells(self) -> None:
        config = load_config()
        rows = [
            {"victim_id": f"v{victim:02d}", "layer": layer, "integrity_status": "PASS"}
            for victim in range(16)
            for layer in range(15)
        ]
        pilot.validate_formal_rows(rows, config)
        with self.assertRaises(pilot.ProtocolError):
            pilot.validate_formal_rows(rows[:-1], config)
        duplicate = rows[:-1] + [dict(rows[0])]
        with self.assertRaises(pilot.ProtocolError):
            pilot.validate_formal_rows(duplicate, config)


if __name__ == "__main__":
    unittest.main()
