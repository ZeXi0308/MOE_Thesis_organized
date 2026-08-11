"""Targeted CPU tests for the calibration-only slot-permutation probe."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import unittest


PATH = Path(__file__).with_name("run_slot_permutation_5090.py")
SPEC = importlib.util.spec_from_file_location("semanticfence_slot_tested", PATH)
assert SPEC is not None and SPEC.loader is not None
M = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = M
SPEC.loader.exec_module(M)
CONFIG_PATH = Path(__file__).with_name("configs") / "slot_permutation_v1.json"


def fixture():
    evidence, references, partner_schedule = {}, {}, []
    focal_ids = []
    cursor = 0
    for pair_index in range(508):
        rows = []
        for slot in range(2):
            row = M.CONTRACT.RowRecord(
                split="calibration",
                document_sha256=f"{cursor + 1:064x}",
                document_index=cursor,
                offset=0,
                token_position=cursor % 16,
                layer=pair_index % 16,
                expert_id=pair_index % 64,
                route_rank=1,
                hidden_sha256=f"{cursor + 10000:064x}",
            )
            cursor += 1
            rows.append(row)
            references[row.row_id] = f"{cursor + 20000:064x}"
        pack = M.CONTRACT.Pack(rows[0].layer, rows[0].expert_id, tuple(rows))
        selected_slots = (0, 1) if pair_index < 4 else (pair_index % 2,)
        for slot, row in enumerate(rows):
            selected = slot in selected_slots
            if selected:
                focal_ids.append(row.row_id)
                label = "safe" if len(focal_ids) <= 256 else "unsafe"
            else:
                label = "unsafe"
            evidence[row.row_id] = M.P.RowEvidence(
                record=row,
                baseline_label=label,
                original_partner_row_id=rows[1 - slot].row_id,
                original_slot=slot,
                original_pack_id=pack.pack_id,
            )
    for focal_id in focal_ids:
        label = evidence[focal_id].baseline_label
        partner_schedule.extend(
            {"focal_row_id": focal_id, "focal_baseline_label": label}
            for _ in range(4)
        )
    return evidence, references, partner_schedule


def numeric(schedule, override=None):
    override = override or {}
    result = []
    for call in schedule:
        focals = []
        for descriptor in call["focals"]:
            flags = [descriptor["baseline_label"] == "safe"] * 10
            if descriptor["focal_row_id"] in override:
                flags = override[descriptor["focal_row_id"]]
            focals.append(
                descriptor
                | {"repeat_exact": flags, "repeat_sha256": ["a" * 64] * 10}
            )
        result.append(
            {
                "schema_version": M.NUMERIC_SCHEMA,
                "call_index": call["call_index"],
                "swapped_pack_id": call["swapped_pack_id"],
                "swapped_row_ids": call["swapped_row_ids"],
                "focals": focals,
            }
        )
    return result


class SlotPermutationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = M.validate_config(json.loads(CONFIG_PATH.read_text()))
        cls.evidence, cls.references, cls.partner_schedule = fixture()
        cls.schedule = M.build_slot_schedule(
            cls.config, cls.partner_schedule, cls.evidence, cls.references
        )

    def test_exact_cohort_pair_and_actual_swap(self):
        counts = M.validate_slot_schedule(self.config, self.schedule)
        self.assertEqual(
            counts,
            {
                "physical_call_count": 508,
                "focal_count": 512,
                "single_focal_call_count": 504,
                "dual_focal_call_count": 4,
                "unique_row_count": 1016,
            },
        )
        for call in self.schedule:
            self.assertEqual(call["swapped_row_ids"], call["original_row_ids"][::-1])
            for focal in call["focals"]:
                self.assertEqual(focal["swapped_slot"], 1 - focal["original_slot"])
                self.assertEqual(
                    call["swapped_row_ids"][focal["swapped_slot"]],
                    focal["focal_row_id"],
                )

    def test_wrong_stage1_cohort_fails(self):
        with self.assertRaisesRegex(M.ProtocolError, "cohort"):
            M.build_slot_schedule(
                self.config,
                self.partner_schedule[:-4],
                self.evidence,
                self.references,
            )

    def test_decisions_use_run03_label_and_new_slot_output(self):
        reference = {"status": "ALL_MATCH", "old_reference_mismatch_count": 0}
        supported = M.summarize(
            self.config, self.schedule, numeric(self.schedule), reference
        )
        self.assertEqual(supported["decision"], "SUPPORT_CALIBRATION_ONLY")
        first = self.schedule[0]["focals"][0]
        opposite = [first["baseline_label"] != "safe"] * 10
        falsified = M.summarize(
            self.config,
            self.schedule,
            numeric(self.schedule, {first["focal_row_id"]: opposite}),
            reference,
        )
        self.assertEqual(falsified["decision"], "FALSIFY_SLOT_INVARIANCE")
        mixed = M.summarize(
            self.config,
            self.schedule,
            numeric(self.schedule, {first["focal_row_id"]: [True, False] * 5}),
            reference,
        )
        self.assertEqual(mixed["decision"], "WEAKEN_SLOT_STABILITY")


if __name__ == "__main__":
    unittest.main()
