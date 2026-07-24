from __future__ import annotations

from dataclasses import fields, replace
from pathlib import Path
import struct
import sys
import unittest


HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

from ric.policy_views import (  # noqa: E402
    AggregateResourceView,
    BView,
    RView,
    ReadyTaskView,
    ReceiverJoinView,
    ResourceBacklogView,
    SView,
    observation_fingerprint,
    validate_b_view,
    validate_r_view,
    validate_s_view,
)
from ric.schema import (  # noqa: E402
    ContributionIdentity,
    ContributionRecord,
    JoinIdentity,
    RICValidationError,
    validate_full_background,
)
from ric.wire import (  # noqa: E402
    HEADER_BYTES,
    HEADER_STRUCT,
    MAGIC,
    RECORD_BYTES,
    ContractMessage,
    ContractRecord,
    ContractTax,
    IdentityBinding,
    IdentityTable,
    SenderContractCache,
    WireProtocolError,
    apply_wire_contract,
    decode_contract,
    encode_contract,
    join_identity_hash_parts,
)


MODEL = "model@revision"


def make_identity(
    request: str,
    block: int,
    slot: int,
    *,
    receiver: int,
    epoch: int = 1,
    layer: int = 3,
) -> ContributionIdentity:
    return ContributionIdentity(
        request_id=request,
        forward_id=f"{request}:forward:0",
        batch_id=f"{request}:batch:0",
        phase="prefill",
        decode_step=0,
        layer_id=layer,
        token_id=f"{request}:token:{block}",
        token_block_id=f"{request}:block:{block}",
        topk_slot=slot,
        expert_id=slot,
        sender_rank=slot,
        receiver_rank=receiver,
        epoch=epoch,
    )


def make_record(
    request: str,
    block: int,
    slot: int,
    *,
    receiver: int,
    epoch: int = 1,
) -> ContributionRecord:
    arrival = float(block * 10)
    return ContributionRecord(
        identity=make_identity(request, block, slot, receiver=receiver, epoch=epoch),
        model_revision=MODEL,
        valid=True,
        arrival_us=arrival,
        ready_us=arrival + slot + 1.0,
        service_us=2.0 + slot,
        deadline_us=arrival + 100.0,
        payload_bytes=32,
        descriptor_bytes=16,
        alignment_bytes=0,
        source_tag="derived_from_measured_lut",
    )


def full_fixture() -> list[ContributionRecord]:
    receivers = {"r0": 2, "r1": 3}
    return [
        make_record(request, block, slot, receiver=receivers[request])
        for request in ("r0", "r1")
        for block in range(2)
        for slot in range(2)
    ]


def join(
    token: str = "t0", *, receiver: int = 2, epoch: int = 1, layer: int = 3
) -> JoinIdentity:
    return JoinIdentity(
        request_id="r0",
        forward_id="r0:forward:0",
        batch_id="r0:batch:0",
        phase="prefill",
        decode_step=0,
        layer_id=layer,
        token_id=token,
        token_block_id=f"block:{token}",
        receiver_rank=receiver,
        epoch=epoch,
    )


def contract_record(identity: JoinIdentity, mask: int = 1) -> ContractRecord:
    hash64, tag16 = join_identity_hash_parts(identity)
    return ContractRecord(hash64, identity.layer_id, mask, tag16, 2, 0)


def contract_payload(
    identity: JoinIdentity,
    *,
    sequence: int = 1,
    sender: int = 0,
    mask: int = 1,
    records: tuple[ContractRecord, ...] | None = None,
) -> bytes:
    return encode_contract(
        ContractMessage(
            sender_rank=sender,
            receiver_rank=identity.receiver_rank,
            epoch=identity.epoch,
            sequence=sequence,
            records=records or (contract_record(identity, mask),),
        )
    )


class FullBackgroundSchemaTests(unittest.TestCase):
    def audit(self, records: list[ContributionRecord], score=None):
        return validate_full_background(
            records,
            top_k=2,
            num_experts=4,
            ep_size=4,
            expected_request_ids=("r0", "r1"),
            expected_token_blocks_per_request=2,
            expert_to_sender={0: 0, 1: 1, 2: 2, 3: 3},
            request_to_receiver={"r0": 2, "r1": 3},
            expected_layer_by_request={"r0": 3, "r1": 3},
            expected_model_revision=MODEL,
            score_join_identities=score,
        )

    def test_complete_fixture_and_score_mask_cannot_change_load(self) -> None:
        rows = full_fixture()
        all_audit = self.audit(rows)
        one_join = rows[0].identity.join_identity
        masked = self.audit(rows, (one_join,))
        self.assertEqual(all_audit.record_count, 8)
        self.assertEqual(all_audit.join_count, 4)
        self.assertEqual(masked.scored_join_count, 1)
        self.assertEqual(masked.full_task_fingerprint, all_audit.full_task_fingerprint)
        self.assertEqual(
            masked.resource_demand_fingerprint,
            all_audit.resource_demand_fingerprint,
        )
        self.assertEqual(masked.wire_bytes, all_audit.wire_bytes)
        self.assertEqual(masked.service_us, all_audit.service_us)
        self.assertNotEqual(masked.score_mask_fingerprint, all_audit.score_mask_fingerprint)

    def test_duplicate_drop_padding_and_topk_mismatch_fail_closed(self) -> None:
        rows = full_fixture()
        with self.assertRaisesRegex(RICValidationError, "duplicate"):
            self.audit(rows + [rows[0]])
        with self.assertRaisesRegex(RICValidationError, "top_k"):
            self.audit(rows[:-1])
        invalid = list(rows)
        invalid[0] = replace(invalid[0], valid=False)
        with self.assertRaisesRegex(RICValidationError, "padding/drop"):
            self.audit(invalid)
        bad_slot = list(rows)
        bad_slot[0] = replace(
            bad_slot[0], identity=replace(bad_slot[0].identity, topk_slot=3)
        )
        with self.assertRaisesRegex(RICValidationError, "topk_slot"):
            self.audit(bad_slot)

    def test_forward_batch_bijection_and_shared_sibling_arrival(self) -> None:
        rows = full_fixture()
        bad_batch = list(rows)
        bad_batch[-1] = replace(
            bad_batch[-1],
            identity=replace(bad_batch[-1].identity, batch_id="r0:batch:0"),
        )
        with self.assertRaisesRegex(RICValidationError, "batch_id"):
            self.audit(bad_batch)
        bad_arrival = list(rows)
        bad_arrival[1] = replace(bad_arrival[1], arrival_us=0.5, ready_us=2.0)
        with self.assertRaisesRegex(RICValidationError, "share one micro-coflow"):
            self.audit(bad_arrival)

    def test_origin_placement_and_metric_mask_outside_world_fail(self) -> None:
        rows = full_fixture()
        bad_sender = list(rows)
        bad_sender[0] = replace(
            bad_sender[0], identity=replace(bad_sender[0].identity, sender_rank=3)
        )
        with self.assertRaisesRegex(RICValidationError, "expert owner"):
            self.audit(bad_sender)
        with self.assertRaisesRegex(RICValidationError, "score mask"):
            self.audit(rows, (join("not-present"),))


class PolicyViewIsolationTests(unittest.TestCase):
    def ready(self, name: str = "task") -> ReadyTaskView:
        identity = make_identity("r0", 0, 0, receiver=2)
        return ReadyTaskView(
            name,
            identity,
            1.0,
            2.0,
            48,
            100.0,
            3.0,
            4.0,
            ("sender:0:egress", "cut:node0->node0", "receiver:2:ingress"),
            (2.0, 1.0, 1.0),
            "receiver:2:combine",
            0.5,
        )

    def test_s_and_b_have_no_dynamic_join_or_contract_field(self) -> None:
        s = SView(0, 5.0, (self.ready(),))
        b = BView(
            s,
            (AggregateResourceView(2, 4),),
            (ResourceBacklogView("receiver:2:ingress", "receiver_ingress", 1, 2.0),),
        )
        validate_s_view(s)
        validate_b_view(b)
        s_names = {field.name for field in fields(SView)} | {
            field.name for field in fields(ReadyTaskView)
        }
        b_names = {field.name for field in fields(BView)} | {
            field.name for field in fields(AggregateResourceView)
        }
        forbidden = {"missing_slot_mask", "receiver_join_state", "contract", "bitmap"}
        self.assertFalse(s_names & forbidden)
        self.assertFalse(b_names & forbidden)

    def test_matched_world_s_b_equal_while_r_can_differ(self) -> None:
        s_a = SView(0, 5.0, (self.ready(),))
        s_b = SView(0, 5.0, (self.ready(),))
        resources = (
            ResourceBacklogView("receiver:2:ingress", "receiver_ingress", 1, 2.0),
        )
        b_a = BView(s_a, (AggregateResourceView(2, 4),), resources)
        b_b = BView(s_b, (AggregateResourceView(2, 4),), resources)
        self.assertEqual(observation_fingerprint(s_a), observation_fingerprint(s_b))
        self.assertEqual(observation_fingerprint(b_a), observation_fingerprint(b_b))
        j = self.ready().identity.join_identity
        r_a = RView(b_a, (ReceiverJoinView(j, 0b01, 1, 1),))
        r_b = RView(b_b, (ReceiverJoinView(j, 0b11, 1, 1),))
        validate_r_view(r_a)
        validate_r_view(r_b)
        self.assertNotEqual(observation_fingerprint(r_a), observation_fingerprint(r_b))

    def test_sender_view_cannot_contain_another_senders_ready_task(self) -> None:
        other = replace(
            self.ready(), identity=replace(self.ready().identity, sender_rank=1)
        )
        with self.assertRaisesRegex(RICValidationError, "another sender"):
            validate_s_view(SView(0, 5.0, (other,)))


class FrozenCodecTests(unittest.TestCase):
    def test_exact_struct_and_round_trip(self) -> None:
        identity = join()
        record = contract_record(identity, 0b101)
        message = ContractMessage(0, 2, 1, 7, (record,))
        payload = encode_contract(message)
        self.assertEqual(HEADER_BYTES, struct.calcsize("<4sBBBBII"))
        self.assertEqual(RECORD_BYTES, struct.calcsize("<QHHHBB"))
        self.assertEqual(len(payload), 32)
        self.assertEqual(HEADER_STRUCT.unpack_from(payload)[0], MAGIC)
        self.assertEqual(decode_contract(payload), message)
        self.assertEqual(record.missing_count, 2)

    def test_malformed_magic_length_count_and_zero_mask_fail(self) -> None:
        identity = join()
        payload = bytearray(contract_payload(identity))
        bad_magic = bytearray(payload)
        bad_magic[:4] = b"BAD!"
        with self.assertRaisesRegex(WireProtocolError, "magic"):
            decode_contract(bytes(bad_magic))
        with self.assertRaisesRegex(WireProtocolError, "length/alignment"):
            decode_contract(bytes(payload[:-1]))
        bad_count = bytearray(payload)
        bad_count[7] = 2
        with self.assertRaisesRegex(WireProtocolError, "record_count"):
            decode_contract(bytes(bad_count))
        with self.assertRaisesRegex(WireProtocolError, "already-closed"):
            contract_record(identity, 0)


class CollisionEpochSequenceApplyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.e1 = join("t1", epoch=1)
        self.e2 = join("t2", epoch=1)
        self.e_next = join("t1", epoch=2)
        self.table = IdentityTable.from_joins((self.e1, self.e2, self.e_next))
        self.tax = ContractTax(1, 2, 3, 4, 5, 6, 7, 8)

    def apply(
        self,
        cache: SenderContractCache,
        identity: JoinIdentity,
        *,
        sequence: int = 1,
        sender: int = 0,
        table: IdentityTable | None = None,
        records: tuple[ContractRecord, ...] | None = None,
    ):
        return apply_wire_contract(
            contract_payload(
                identity,
                sequence=sequence,
                sender=sender,
                records=records,
            ),
            cache=cache,
            identity_table=table or self.table,
            expected_sender_rank=0,
            tax=self.tax,
        )

    def assert_fallback_tax(self, result, payload_len: int = 32) -> None:
        self.assertFalse(result.applied)
        self.assertTrue(result.fallback)
        self.assertEqual(result.charged_bytes, payload_len)
        self.assertEqual(result.charged_us, 36.0)

    def test_successful_apply_uses_full_identity_table(self) -> None:
        cache = SenderContractCache(0)
        result = self.apply(cache, self.e1)
        self.assertTrue(result.applied)
        self.assertFalse(result.fallback)
        self.assertEqual(result.charged_us, 36.0)
        self.assertEqual(result.entries[0].join_identity, self.e1)
        self.assertTrue(result.entries[0].is_last_sibling)
        self.assertIn(self.e1, cache.snapshot())
        self.assertEqual(cache.last_sequence(2, 1), 1)

    def test_duplicate_missing_out_of_order_stale_and_epoch_start_fallback(self) -> None:
        cache = SenderContractCache(0)
        self.assertTrue(self.apply(cache, self.e1, sequence=1).applied)
        duplicate = self.apply(cache, self.e1, sequence=1)
        self.assert_fallback_tax(duplicate)
        self.assertEqual(duplicate.fault, "duplicate_sequence")
        missing = self.apply(cache, self.e1, sequence=3)
        self.assert_fallback_tax(missing)
        self.assertEqual(missing.fault, "missing_sequence")
        self.assertTrue(self.apply(cache, self.e1, sequence=2).applied)
        out_of_order = self.apply(cache, self.e1, sequence=1)
        self.assert_fallback_tax(out_of_order)
        self.assertEqual(out_of_order.fault, "out_of_order_sequence")
        bad_new_epoch = self.apply(cache, self.e_next, sequence=2)
        self.assert_fallback_tax(bad_new_epoch)
        self.assertEqual(bad_new_epoch.fault, "unknown_or_missing_epoch_start")
        self.assertTrue(self.apply(cache, self.e_next, sequence=1).applied)
        stale = self.apply(cache, self.e1, sequence=3)
        self.assert_fallback_tax(stale)
        self.assertEqual(stale.fault, "stale_epoch")

    def test_unknown_epoch_wrong_sender_and_malformed_keep_tax(self) -> None:
        cache = SenderContractCache(0)
        unknown = join("unknown", epoch=3)
        result = self.apply(cache, unknown)
        self.assert_fallback_tax(result)
        self.assertEqual(result.fault, "unknown_epoch")
        wrong_sender = self.apply(cache, self.e1, sender=1)
        self.assert_fallback_tax(wrong_sender)
        self.assertEqual(wrong_sender.fault, "wrong_sender")
        malformed_payload = contract_payload(self.e1)[:-1]
        malformed = apply_wire_contract(
            malformed_payload,
            cache=cache,
            identity_table=self.table,
            expected_sender_rank=0,
            tax=self.tax,
            produced_bytes=32,
        )
        self.assert_fallback_tax(malformed, 32)
        self.assertEqual(malformed.received_bytes, len(malformed_payload))
        self.assertTrue(malformed.fault.startswith("malformed:"))

    def test_mask_bits_beyond_model_topk_fail_closed(self) -> None:
        table = IdentityTable.from_joins((self.e1,), top_k=2)
        cache = SenderContractCache(0)
        result = self.apply(
            cache,
            self.e1,
            table=table,
            records=(contract_record(self.e1, mask=0b100),),
        )
        self.assert_fallback_tax(result)
        self.assertIn("top_k", result.fault)
        self.assertEqual(cache.snapshot(), {})

    def test_collision_and_multi_record_failure_are_atomic(self) -> None:
        hash64, tag16 = join_identity_hash_parts(self.e1)
        colliding_identity = join("collision", epoch=1)
        collision_table = IdentityTable(
            (
                IdentityBinding.from_join(self.e1),
                IdentityBinding(colliding_identity, hash64, tag16),
            )
        )
        cache = SenderContractCache(0)
        collision = self.apply(cache, self.e1, table=collision_table)
        self.assert_fallback_tax(collision)
        self.assertIn("collision", collision.fault)
        self.assertEqual(cache.snapshot(), {})

        unknown_record = replace(contract_record(self.e2), join_key_hash64=123)
        records = (contract_record(self.e1), unknown_record)
        payload_len = HEADER_BYTES + 2 * RECORD_BYTES
        atomic = self.apply(cache, self.e1, records=records)
        self.assert_fallback_tax(atomic, payload_len)
        self.assertEqual(cache.snapshot(), {})


if __name__ == "__main__":
    unittest.main()
