"""Unit tests for the pure SemanticFence executor-contract core."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import copy
import importlib.util
from pathlib import Path
import sys
import unittest


MODULE_PATH = Path(__file__).with_name("executor_contract.py")
SPEC = importlib.util.spec_from_file_location("semanticfence_executor_contract", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


CalibrationObservation = MODULE.CalibrationObservation
ContractError = MODULE.ContractError
Pack = MODULE.Pack
RowRecord = MODULE.RowRecord


STACK = "a" * 64
SIG_M2 = "b" * 64
SIG_M4 = "c" * 64


def row(
    index: int,
    *,
    split: str = "calibration",
    document: int | None = None,
    layer: int = 3,
    expert: int = 7,
) -> RowRecord:
    document_index = index if document is None else document
    return RowRecord(
        split=split,
        document_sha256=f"{document_index + 1:064x}",
        document_index=document_index,
        offset=index * 16,
        token_position=index,
        layer=layer,
        expert_id=expert,
        route_rank=(index % 8) + 1,
        hidden_sha256=f"{index + 1000:064x}",
    )


def observation(
    rows: tuple[RowRecord, ...],
    *,
    signature: str = SIG_M2,
    repeats: tuple[tuple[bool, ...], ...] | None = None,
) -> CalibrationObservation:
    pack = Pack(layer=rows[0].layer, expert_id=rows[0].expert_id, rows=rows)
    if repeats is None:
        repeats = (tuple(True for _ in rows),) * 3
    return CalibrationObservation(
        pack=pack,
        signature=signature,
        repeat_row_exact=repeats,
    )


class IdentityAndPackingTest(unittest.TestCase):
    def test_row_and_pack_are_immutable_and_row_id_is_canonical(self) -> None:
        first = row(0)
        rebuilt = row(0)
        self.assertEqual(first.row_id, rebuilt.row_id)
        self.assertEqual(len(first.row_id), 64)
        self.assertNotEqual(
            first.row_id,
            replace(first, token_position=first.token_position + 1).row_id,
        )
        with self.assertRaises(FrozenInstanceError):
            first.layer = 9  # type: ignore[misc]

        pack = Pack(layer=3, expert_id=7, rows=(first, row(1)))
        with self.assertRaises(FrozenInstanceError):
            pack.expert_id = 9  # type: ignore[misc]

    def test_packing_is_same_layer_expert_distinct_and_covers_remainder(self) -> None:
        rows = tuple(row(index) for index in range(5)) + (
            row(20, layer=4, expert=8),
            row(21, layer=4, expert=8),
        )
        packs = MODULE.pack_distinct_rows(rows, allowed_ms=(4, 2))
        self.assertEqual([pack.m for pack in packs], [4, 1, 2])
        self.assertTrue(
            all(
                all(
                    item.layer == pack.layer and item.expert_id == pack.expert_id
                    for item in pack.rows
                )
                for pack in packs
            )
        )
        MODULE.validate_row_coverage(rows, packs)

    def test_cross_expert_pack_and_duplicate_input_fail(self) -> None:
        with self.assertRaises(ContractError):
            Pack(layer=3, expert_id=7, rows=(row(0), row(1, expert=8)))
        duplicate = row(0)
        with self.assertRaises(ContractError):
            MODULE.pack_distinct_rows((duplicate, duplicate), allowed_ms=(2,))

    def test_coverage_detects_missing_and_duplicate_rows(self) -> None:
        rows = (row(0), row(1), row(2))
        missing = (Pack(layer=3, expert_id=7, rows=rows[:2]),)
        with self.assertRaisesRegex(ContractError, "coverage mismatch"):
            MODULE.validate_row_coverage(rows, missing)

        duplicated = (
            Pack(layer=3, expert_id=7, rows=rows[:2]),
            Pack(layer=3, expert_id=7, rows=(rows[1], rows[2])),
        )
        with self.assertRaisesRegex(ContractError, "duplicate row"):
            MODULE.validate_row_coverage(rows, duplicated)


class ContractBuildTest(unittest.TestCase):
    def test_contract_rejects_heldout_leakage(self) -> None:
        contaminated = observation(
            (row(0), row(1, split="heldout")), signature=SIG_M2
        )
        with self.assertRaisesRegex(ContractError, "calibration rows only"):
            MODULE.build_contract(
                (contaminated,),
                stack_digest=STACK,
                min_packs=1,
                min_documents=1,
            )

    def test_minimum_pack_and_document_support_is_required(self) -> None:
        same_doc_a = observation((row(0, document=0), row(1, document=0)))
        same_doc_b = observation((row(2, document=0), row(3, document=0)))
        under_supported = MODULE.build_contract(
            (same_doc_a, same_doc_b),
            stack_digest=STACK,
            min_packs=2,
            min_documents=2,
        )
        self.assertFalse(under_supported.entries[0].allowed)

        distinct_doc_b = observation((row(4, document=1), row(5, document=1)))
        supported = MODULE.build_contract(
            (same_doc_a, distinct_doc_b),
            stack_digest=STACK,
            min_packs=2,
            min_documents=2,
        )
        self.assertTrue(supported.entries[0].allowed)

    def test_one_non_exact_repeat_disallows_the_entire_entry(self) -> None:
        exact = observation((row(0, document=0), row(1, document=0)))
        changed = observation(
            (row(4, document=1), row(5, document=1)),
            repeats=((True, True), (True, False), (True, True)),
        )
        contract = MODULE.build_contract(
            (exact, changed),
            stack_digest=STACK,
            min_packs=2,
            min_documents=2,
        )
        entry = contract.entries[0]
        self.assertFalse(entry.all_repeats_exact)
        self.assertFalse(entry.allowed)
        self.assertLess(entry.exact_checks, entry.total_checks)

    def test_unknown_stack_m_or_signature_falls_back_to_m1(self) -> None:
        contract = MODULE.build_contract(
            (
                observation((row(0, document=0), row(1, document=0))),
                observation((row(4, document=1), row(5, document=1))),
            ),
            stack_digest=STACK,
            min_packs=2,
            min_documents=2,
        )
        self.assertEqual(
            MODULE.choose_pack_size(
                contract,
                stack_digest=STACK,
                layer=3,
                expert_id=7,
                requested_m=2,
                signature=SIG_M2,
            ),
            2,
        )
        cases = (
            ("d" * 64, 2, SIG_M2),
            (STACK, 4, SIG_M2),
            (STACK, 2, SIG_M4),
        )
        for stack_digest, requested_m, signature in cases:
            with self.subTest(
                stack_digest=stack_digest,
                requested_m=requested_m,
                signature=signature,
            ):
                self.assertEqual(
                    MODULE.choose_pack_size(
                        contract,
                        stack_digest=stack_digest,
                        layer=3,
                        expert_id=7,
                        requested_m=requested_m,
                        signature=signature,
                    ),
                    1,
                )

    def test_other_expert_evidence_cannot_authorize_current_expert(self) -> None:
        contract = MODULE.build_contract(
            (
                observation(
                    (row(0, document=0, expert=8), row(1, document=0, expert=8))
                ),
                observation(
                    (row(4, document=1, expert=8), row(5, document=1, expert=8))
                ),
            ),
            stack_digest=STACK,
            min_packs=2,
            min_documents=2,
        )
        self.assertEqual(
            MODULE.choose_pack_size(
                contract,
                stack_digest=STACK,
                layer=3,
                expert_id=8,
                requested_m=2,
                signature=SIG_M2,
            ),
            2,
        )
        self.assertEqual(
            MODULE.choose_pack_size(
                contract,
                stack_digest=STACK,
                layer=3,
                expert_id=7,
                requested_m=2,
                signature=SIG_M2,
            ),
            1,
        )
        self.assertEqual(
            set(MODULE.allowed_entries(contract)),
            {(3, 8, 2, SIG_M2)},
        )

    def test_contract_round_trip_and_content_tamper_fail_closed(self) -> None:
        contract = MODULE.build_contract(
            (observation((row(0), row(1))),),
            stack_digest=STACK,
            min_packs=1,
            min_documents=1,
        )
        encoded = contract.to_dict()
        decoded = MODULE.contract_from_dict(encoded)
        self.assertEqual(decoded, contract)

        tampered = copy.deepcopy(encoded)
        tampered["entries"][0]["pack_count"] = 99
        with self.assertRaises(ContractError):
            MODULE.validate_contract(tampered)

        tampered_hash = copy.deepcopy(encoded)
        tampered_hash["contract_sha256"] = "f" * 64
        with self.assertRaisesRegex(ContractError, "content hash mismatch"):
            MODULE.validate_contract(tampered_hash)

    def test_duplicate_calibration_pack_cannot_inflate_minimum_support(self) -> None:
        repeated = observation((row(0), row(1)))
        with self.assertRaisesRegex(ContractError, "duplicate calibration pack"):
            MODULE.build_contract(
                (repeated, repeated),
                stack_digest=STACK,
                min_packs=2,
                min_documents=1,
            )


if __name__ == "__main__":
    unittest.main()
