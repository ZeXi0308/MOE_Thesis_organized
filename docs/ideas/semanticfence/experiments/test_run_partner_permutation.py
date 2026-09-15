"""CPU-only tests for the SemanticFence partner-permutation runner."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


MODULE_PATH = Path(__file__).with_name("run_partner_permutation_5090.py")
SPEC = importlib.util.spec_from_file_location(
    "semanticfence_partner_permutation", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

CONFIG_PATH = Path(__file__).with_name("configs") / "partner_permutation_v1.json"


def frozen_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def make_row(index: int, *, layer: int, expert: int):
    return MODULE.CONTRACT.RowRecord(
        split="calibration",
        document_sha256=f"{index + 1:064x}",
        document_index=index,
        offset=0,
        token_position=index % 16,
        layer=layer,
        expert_id=expert,
        route_rank=1,
        hidden_sha256=f"{index + 100000:064x}",
    )


def synthetic_evidence() -> tuple[dict, dict]:
    evidence = {}
    references = {}
    cursor = 0
    for layer in range(16):
        for expert in range(16):
            by_label = {"safe": [], "unsafe": []}
            for label in ("safe", "unsafe"):
                for _ in range(6):
                    row = make_row(cursor, layer=layer, expert=expert)
                    cursor += 1
                    by_label[label].append(row)
                    references[row.row_id] = f"{cursor + 200000:064x}"
            for label, rows in by_label.items():
                for pair_start in range(0, len(rows), 2):
                    left, right = rows[pair_start : pair_start + 2]
                    pack = MODULE.CONTRACT.Pack(
                        layer=layer, expert_id=expert, rows=(left, right)
                    )
                    evidence[left.row_id] = MODULE.RowEvidence(
                        record=left,
                        baseline_label=label,
                        original_partner_row_id=right.row_id,
                        original_slot=0,
                        original_pack_id=pack.pack_id,
                    )
                    evidence[right.row_id] = MODULE.RowEvidence(
                        record=right,
                        baseline_label=label,
                        original_partner_row_id=left.row_id,
                        original_slot=1,
                        original_pack_id=pack.pack_id,
                    )
    return evidence, references


def numeric_for_schedule(schedule, *, override=None):
    override = override or {}
    result = []
    for call in schedule:
        baseline = call["focal_baseline_label"]
        flags = [baseline == "safe"] * 10
        if call["call_index"] in override:
            flags = list(override[call["call_index"]])
        result.append(
            {
                "schema_version": MODULE.NUMERIC_SCHEMA,
                "call_index": call["call_index"],
                "pack_id": "a" * 64,
                "layer": call["layer"],
                "expert_id": call["expert_id"],
                "m": 2,
                "row_ids": call["row_ids"],
                "focal_row_id": call["focal_row_id"],
                "focal_original_slot": call["focal_original_slot"],
                "focal_baseline_label": baseline,
                "partner_row_id": call["partner_row_id"],
                "partner_baseline_label": call["partner_baseline_label"],
                "focal_repeat_exact": flags,
                "partner_repeat_exact": [True] * 10,
                "focal_repeat_sha256": ["b" * 64] * 10,
                "partner_repeat_sha256": ["c" * 64] * 10,
                "representative_full_output_sha256": "d" * 64,
            }
        )
    return result


REFERENCE_OK = {
    "status": "ALL_MATCH",
    "new_reference_all_stable": True,
    "old_reference_mismatch_count": 0,
    "scheduled_unique_row_count": 1,
}


class ConfigTest(unittest.TestCase):
    def test_frozen_config_accepts_and_protocol_change_fails(self) -> None:
        config = frozen_config()
        self.assertEqual(MODULE.validate_config(config)["status"], "FROZEN_PRE_RUN")
        changed = copy.deepcopy(config)
        changed["selection"]["preserve_focal_original_slot"] = False
        with self.assertRaisesRegex(MODULE.ProtocolError, "preserve_focal"):
            MODULE.validate_config(changed)


class ScheduleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = MODULE.validate_config(frozen_config())
        cls.evidence, cls.references = synthetic_evidence()
        cls.schedule, cls.summary = MODULE.select_partner_schedule(
            cls.config, cls.evidence, cls.references
        )

    def test_schedule_has_frozen_denominators_and_controls(self) -> None:
        self.assertEqual(len(self.schedule), 2048)
        self.assertEqual(self.summary["selected_cell_count"], 256)
        self.assertEqual(self.summary["focal_count"], 512)
        self.assertEqual(
            self.summary["scheduled_document_count"],
            self.summary["unique_scheduled_row_count"],
        )
        focal_counts = {}
        pairs = set()
        for call in self.schedule:
            focal = self.evidence[call["focal_row_id"]]
            partner = self.evidence[call["partner_row_id"]]
            self.assertEqual(call["row_ids"][focal.original_slot], focal.row_id)
            self.assertNotEqual(partner.row_id, focal.original_partner_row_id)
            self.assertNotEqual(
                partner.record.document_sha256, focal.record.document_sha256
            )
            self.assertNotEqual(
                partner.record.hidden_sha256, focal.record.hidden_sha256
            )
            focal_counts[focal.row_id] = focal_counts.get(focal.row_id, 0) + 1
            pair = tuple(sorted(call["row_ids"]))
            self.assertNotIn(pair, pairs)
            pairs.add(pair)
        self.assertEqual(set(focal_counts.values()), {4})

    def test_selection_is_independent_of_input_mapping_order(self) -> None:
        reversed_evidence = dict(reversed(list(self.evidence.items())))
        second, _ = MODULE.select_partner_schedule(
            self.config, reversed_evidence, self.references
        )
        self.assertEqual(self.schedule, second)

    def test_missing_one_required_cell_fails_closed(self) -> None:
        reduced = {
            row_id: value
            for row_id, value in self.evidence.items()
            if not (value.record.layer == 0 and value.record.expert_id == 0)
        }
        with self.assertRaisesRegex(MODULE.ProtocolError, "layer 0"):
            MODULE.select_partner_schedule(self.config, reduced, self.references)

    def test_slot_tamper_is_rejected(self) -> None:
        changed = copy.deepcopy(self.schedule)
        changed[0]["focal_original_slot"] = 1 - changed[0]["focal_original_slot"]
        with self.assertRaisesRegex(MODULE.ProtocolError, "slot"):
            MODULE.validate_schedule(self.config, changed)


class ResultTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = MODULE.validate_config(frozen_config())
        evidence, references = synthetic_evidence()
        cls.schedule, _ = MODULE.select_partner_schedule(
            cls.config, evidence, references
        )

    def test_all_invariant_is_calibration_only_support(self) -> None:
        result = MODULE.summarize_partner_outcomes(
            self.config,
            self.schedule,
            numeric_for_schedule(self.schedule),
            REFERENCE_OK,
        )
        self.assertEqual(result["decision"], "SUPPORT_CALIBRATION_ONLY")
        self.assertFalse(result["paper_result"])
        self.assertEqual(result["stable_flip_call_count"], 0)
        self.assertEqual(result["mixed_call_count"], 0)

    def test_stable_opposite_outcome_falsifies(self) -> None:
        baseline = self.schedule[0]["focal_baseline_label"]
        opposite_flags = [baseline != "safe"] * 10
        result = MODULE.summarize_partner_outcomes(
            self.config,
            self.schedule,
            numeric_for_schedule(self.schedule, override={0: opposite_flags}),
            REFERENCE_OK,
        )
        self.assertEqual(result["decision"], "FALSIFY_PARTNER_INVARIANCE")
        self.assertEqual(result["stable_flip_call_count"], 1)

    def test_mixed_repeat_outcome_weakens_stability(self) -> None:
        mixed = [True, False] * 5
        result = MODULE.summarize_partner_outcomes(
            self.config,
            self.schedule,
            numeric_for_schedule(self.schedule, override={0: mixed}),
            REFERENCE_OK,
        )
        self.assertEqual(result["decision"], "WEAKEN_ROW_STABILITY")
        self.assertEqual(result["mixed_call_count"], 1)

    def test_reference_mismatch_fails_before_scientific_decision(self) -> None:
        reference = dict(REFERENCE_OK) | {"old_reference_mismatch_count": 1}
        with self.assertRaisesRegex(MODULE.ProtocolError, "reference"):
            MODULE.summarize_partner_outcomes(
                self.config,
                self.schedule,
                numeric_for_schedule(self.schedule),
                reference,
            )


class ArtifactAuthorityTest(unittest.TestCase):
    def test_complete_is_written_last_and_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            (output / "raw.json").write_text("{}\n", encoding="utf-8")
            MODULE.finalize_complete(
                output,
                {
                    "schema_version": MODULE.RESULT_SCHEMA,
                    "decision": "SUPPORT_CALIBRATION_ONLY",
                },
            )
            complete = json.loads((output / "COMPLETE.json").read_text())
            self.assertEqual(complete["status"], "SUCCESS_COMPLETE")
            self.assertIn("PARTNER_RESULT.json", complete["artifact_sha256"])
            with self.assertRaises(MODULE.ProtocolError):
                MODULE.finalize_complete(output, {"decision": "WEAKEN_ROW_STABILITY"})


if __name__ == "__main__":
    unittest.main()
