"""CPU-only tests for SemanticFence GPU capture/planning boundaries."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import random
import sys
import unittest


MODULE_PATH = Path(__file__).with_name("gpu_execution.py")
TORCH_WAS_IMPORTED = "torch" in sys.modules
SPEC = importlib.util.spec_from_file_location("semanticfence_gpu_execution", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

CONTRACT = MODULE.CONTRACT
STACK = "a" * 64
SIG_M2 = "b" * 64
SIG_M4 = "c" * 64


def row(
    index: int,
    *,
    split: str = "evaluation",
    document: int | None = None,
    layer: int = 2,
    expert: int = 3,
) -> object:
    document_index = index if document is None else document
    return CONTRACT.RowRecord(
        split=split,
        document_sha256=f"{document_index + 1:064x}",
        document_index=document_index,
        offset=(index % 2) * 256,
        token_position=index % 16,
        layer=layer,
        expert_id=expert,
        route_rank=(index % 8) + 1,
        hidden_sha256=f"{index + 1000:064x}",
    )


def observation(rows: tuple[object, ...], signature: str):
    pack = CONTRACT.Pack(
        layer=rows[0].layer,
        expert_id=rows[0].expert_id,
        rows=rows,
    )
    return CONTRACT.CalibrationObservation(
        pack=pack,
        signature=signature,
        repeat_row_exact=(tuple(True for _ in rows),) * 10,
    )


def allowed_contract():
    m2 = tuple(
        row(index, split="calibration", document=index, expert=3)
        for index in range(2)
    )
    m4 = tuple(
        row(index + 10, split="calibration", document=index + 10, expert=3)
        for index in range(4)
    )
    return CONTRACT.build_contract(
        (observation(m2, SIG_M2), observation(m4, SIG_M4)),
        stack_digest=STACK,
        min_packs=1,
        min_documents=1,
    )


class LazyImportAndSplitTest(unittest.TestCase):
    def test_module_import_does_not_import_torch(self) -> None:
        self.assertEqual("torch" in sys.modules, TORCH_WAS_IMPORTED)

    def test_split_positions_are_frozen(self) -> None:
        self.assertEqual(MODULE.positions_for_split("calibration"), tuple(range(16)))
        self.assertEqual(MODULE.positions_for_split("evaluation"), (15,))
        self.assertEqual(MODULE.positions_for_split("semanticfence_eval_fresh"), (15,))
        with self.assertRaises(MODULE.GPUExecutionError):
            MODULE.positions_for_split("heldout")

    def test_raw_bf16_mismatch_counts_signed_zero(self) -> None:
        self.assertEqual(
            MODULE.strict_bf16_mismatch_count(
                bytes.fromhex("0000"), bytes.fromhex("0080")
            ),
            1,
        )


class CalibrationPackingTest(unittest.TestCase):
    def test_calibration_packs_are_deterministic_and_within_m_disjoint(self) -> None:
        rows = [
            row(
                index,
                split="calibration",
                document=index % 4,
                layer=1,
                expert=5,
            )
            for index in range(12)
        ]
        first = MODULE.build_calibration_packs(rows, m_values=(2, 4))
        shuffled = list(rows)
        random.Random(7).shuffle(shuffled)
        second = MODULE.build_calibration_packs(shuffled, m_values=(4, 2))
        self.assertEqual(
            [(pack.m, pack.pack_id) for pack in first],
            [(pack.m, pack.pack_id) for pack in second],
        )
        self.assertEqual([pack.m for pack in first].count(2), 6)
        self.assertEqual([pack.m for pack in first].count(4), 3)

        seen_by_m: dict[int, set[str]] = {}
        for pack in first:
            seen = seen_by_m.setdefault(pack.m, set())
            self.assertFalse(seen & {item.row_id for item in pack.rows})
            seen.update(item.row_id for item in pack.rows)

    def test_calibration_packs_reject_eval_rows_and_unknown_m(self) -> None:
        with self.assertRaisesRegex(MODULE.GPUExecutionError, "calibration rows"):
            MODULE.build_calibration_packs((row(0), row(1)), m_values=(2,))
        calibration = (
            row(0, split="calibration"),
            row(1, split="calibration"),
        )
        with self.assertRaisesRegex(MODULE.GPUExecutionError, "frozen M grid"):
            MODULE.build_calibration_packs(calibration, m_values=(3,))


class FourArmPlanningTest(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = tuple(row(index, expert=3) for index in range(6)) + tuple(
            row(index + 20, expert=4) for index in range(2)
        )
        self.contract = allowed_contract()

    def test_four_arms_cover_same_rows_and_have_expected_shapes(self) -> None:
        plans = MODULE.plan_four_arms(
            self.rows, contract=self.contract, stack_digest=STACK
        )
        expected = {item.row_id for item in self.rows}
        for plan in plans.values():
            self.assertEqual(set(plan.row_ids), expected)
            self.assertEqual(len(plan.row_ids), len(expected))
            self.assertEqual(plan.row_ids, plans[MODULE.ARM_A].row_ids)

        arm_a = plans[MODULE.ARM_A]
        self.assertEqual(len(arm_a.calls), len(self.rows))
        self.assertTrue(all(call.execution_m == 1 for call in arm_a.calls))

        arm_b = plans[MODULE.ARM_B]
        self.assertEqual(
            sorted(call.execution_m for call in arm_b.calls), [2, 6]
        )
        self.assertEqual(arm_b.padding_rows, 0)

        arm_c = plans[MODULE.ARM_C]
        self.assertTrue(all(call.execution_m == 64 for call in arm_c.calls))
        self.assertEqual(arm_c.padding_rows, 120)

        arm_d = plans[MODULE.ARM_D]
        expert3 = [
            call.execution_m for call in arm_d.calls if call.expert_id == 3
        ]
        expert4 = [
            call.execution_m for call in arm_d.calls if call.expert_id == 4
        ]
        self.assertEqual(expert3, [4, 2])
        self.assertEqual(expert4, [1, 1])
        self.assertEqual(arm_d.padding_rows, 0)
        self.assertEqual(
            [call.expected_signatures for call in arm_d.calls if call.expert_id == 3],
            [(SIG_M4,), (SIG_M2,)],
        )

    def test_unknown_stack_fails_closed_to_m1(self) -> None:
        plan = MODULE.plan_arm_d(
            self.rows, contract=self.contract, stack_digest="f" * 64
        )
        self.assertTrue(all(call.execution_m == 1 for call in plan.calls))
        self.assertEqual(plan.padding_rows, 0)

    def test_ambiguous_signature_fails_closed_to_m1(self) -> None:
        rows_a = tuple(
            row(index, split="calibration", document=index, expert=3)
            for index in range(2)
        )
        rows_b = tuple(
            row(index + 30, split="calibration", document=index + 30, expert=3)
            for index in range(2)
        )
        ambiguous = CONTRACT.build_contract(
            (observation(rows_a, SIG_M2), observation(rows_b, "d" * 64)),
            stack_digest=STACK,
            min_packs=1,
            min_documents=1,
        )
        plan = MODULE.plan_arm_d(
            self.rows, contract=ambiguous, stack_digest=STACK
        )
        self.assertTrue(all(call.execution_m == 1 for call in plan.calls))

    def test_rare_disallowed_signature_still_makes_admission_ambiguous(self) -> None:
        first = tuple(
            row(index, split="calibration", document=index, expert=3)
            for index in range(2)
        )
        second = tuple(
            row(index + 30, split="calibration", document=index + 30, expert=3)
            for index in range(2)
        )
        rare = tuple(
            row(index + 60, split="calibration", document=index + 60, expert=3)
            for index in range(2)
        )
        contract = CONTRACT.build_contract(
            (
                observation(first, SIG_M2),
                observation(second, SIG_M2),
                observation(rare, "e" * 64),
            ),
            stack_digest=STACK,
            min_packs=2,
            min_documents=1,
        )
        plan = MODULE.plan_arm_d(self.rows, contract=contract, stack_digest=STACK)
        self.assertTrue(all(call.execution_m == 1 for call in plan.calls))

    def test_plans_are_independent_of_input_order(self) -> None:
        shuffled = list(self.rows)
        random.Random(11).shuffle(shuffled)
        first = MODULE.plan_four_arms(
            self.rows, contract=self.contract, stack_digest=STACK
        )
        second = MODULE.plan_four_arms(
            shuffled, contract=self.contract, stack_digest=STACK
        )
        for arm in (MODULE.ARM_A, MODULE.ARM_B, MODULE.ARM_C, MODULE.ARM_D):
            self.assertEqual(
                [call.call_id for call in first[arm].calls],
                [call.call_id for call in second[arm].calls],
            )

    def test_natural_and_fixed_plans_reject_out_of_range_group(self) -> None:
        too_many = tuple(row(index + 100, expert=9) for index in range(65))
        with self.assertRaisesRegex(MODULE.GPUExecutionError, "maximum M"):
            MODULE.plan_arm_b(too_many)
        with self.assertRaisesRegex(MODULE.GPUExecutionError, "larger"):
            MODULE.plan_arm_c(too_many)

    def test_repeat_orders_are_frozen_rotations(self) -> None:
        base = (MODULE.ARM_A, MODULE.ARM_B, MODULE.ARM_C, MODULE.ARM_D)
        self.assertEqual(MODULE.frozen_arm_order(0), base)
        self.assertEqual(MODULE.frozen_arm_order(1), base[1:] + base[:1])
        self.assertEqual(MODULE.frozen_arm_order(3), base[3:] + base[:3])
        self.assertEqual(MODULE.frozen_arm_order(4), base)
        with self.assertRaises(MODULE.GPUExecutionError):
            MODULE.frozen_arm_order(-1)


if __name__ == "__main__":
    unittest.main()
