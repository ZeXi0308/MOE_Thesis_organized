from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

try:
    from .schema import (
        ExpectedRouteEvent,
        ProtocolError,
        ROUTE_COLUMNS,
        RouteContribution,
        load_expected_events_jsonl,
        load_route_csv,
        validate_route_rows,
    )
except ImportError:
    from schema import (
        ExpectedRouteEvent,
        ProtocolError,
        ROUTE_COLUMNS,
        RouteContribution,
        load_expected_events_jsonl,
        load_route_csv,
        validate_route_rows,
    )


def route_row(
    *,
    slot: int,
    expert: int,
    target_rank: int,
    chunk: int = 0,
    split: str = "evaluation",
    prompt_hash: str = "a" * 64,
) -> RouteContribution:
    return RouteContribution.from_mapping(
        {
            "model": "toy",
            "model_revision": "b" * 40,
            "tenant_id": "tenant-victim",
            "request_id": "request-0",
            "document_id": "document-0",
            "isolation_domain": "engine-0",
            "split": split,
            "role": "victim",
            "traffic_class": "NAT_BENIGN",
            "prompt_hash": prompt_hash,
            "prompt_tokens": 512,
            "phase": "prefill",
            "chunk_id": chunk,
            "decode_step": -1,
            "token_position": chunk,
            "token_id": 100 + chunk,
            "layer_id": 0,
            "topk_slot": slot,
            "expert_id": expert,
            "gate_weight": 0.5,
            "placement_id": "toy-placement",
            "target_rank": target_rank,
            "rank_binding_stage": "EXECUTED_DISPATCH",
            "replica_instance_id": f"replica-{target_rank}",
            "device_uuid": f"GPU-toy-{target_rank}",
            "dispatch_event_id": f"{chunk * 100 + slot + 1:064x}",
            "request_arrival_us": 0.0,
            "route_observed_us": float(chunk + 1),
        }
    )


RANK_BINDINGS = {
    "toy": {
        0: frozenset(
            {(0, "replica-0", "GPU-toy-0"), (1, "replica-1", "GPU-toy-1")}
        ),
        1: frozenset({(1, "replica-1", "GPU-toy-1")}),
        2: frozenset(
            {(0, "replica-0", "GPU-toy-0"), (1, "replica-1", "GPU-toy-1")}
        ),
        3: frozenset(
            {(0, "replica-0", "GPU-toy-0"), (1, "replica-1", "GPU-toy-1")}
        ),
    }
}


def expected_events(
    rows: list[RouteContribution], top_k: int
) -> list[ExpectedRouteEvent]:
    unique: dict[tuple[object, ...], RouteContribution] = {}
    for row in rows:
        unique.setdefault(row.token_event_key, row)
    return [
        ExpectedRouteEvent(
            model=row.model,
            model_revision=row.model_revision,
            tokenizer_sha256="d" * 64,
            tenant_id=row.tenant_id,
            request_id=row.request_id,
            prompt_hash=row.prompt_hash,
            prompt_tokens=row.prompt_tokens,
            phase=row.phase,
            chunk_id=row.chunk_id,
            decode_step=row.decode_step,
            token_position=row.token_position,
            token_id=row.token_id,
            layer_id=row.layer_id,
            expected_top_k=top_k,
        )
        for row in unique.values()
    ]


class RouteSchemaTest(unittest.TestCase):
    def test_valid_tenant_route_closes_topk(self) -> None:
        rows = [
            route_row(slot=0, expert=0, target_rank=0),
            route_row(slot=1, expert=1, target_rank=1),
        ]
        validate_route_rows(
            rows,
            expected_topk={"toy": 2},
            expected_revisions={"toy": "b" * 40},
            expected_tokenizers={"toy": "d" * 64},
            num_experts={"toy": 4},
            expected_dispatch_bindings=RANK_BINDINGS,
            expected_events=expected_events(rows, 2),
        )

    def test_expected_manifest_tokenizer_is_hash_bound(self) -> None:
        rows = [
            route_row(slot=0, expert=0, target_rank=0),
            route_row(slot=1, expert=1, target_rank=1),
        ]
        with self.assertRaisesRegex(ProtocolError, "tokenizer hash mismatch"):
            validate_route_rows(
                rows,
                expected_topk={"toy": 2},
                expected_tokenizers={"toy": "e" * 64},
                num_experts={"toy": 4},
                expected_dispatch_bindings=RANK_BINDINGS,
                expected_events=expected_events(rows, 2),
            )

    def test_missing_topk_sibling_fails_closed(self) -> None:
        with self.assertRaisesRegex(ProtocolError, "top-k closure"):
            rows = [route_row(slot=0, expert=0, target_rank=0)]
            validate_route_rows(
                rows,
                expected_topk={"toy": 2},
                expected_tokenizers={"toy": "d" * 64},
                num_experts={"toy": 4},
                expected_dispatch_bindings=RANK_BINDINGS,
                expected_events=expected_events(rows, 2),
            )

    def test_duplicate_expert_in_topk_fails_closed(self) -> None:
        with self.assertRaisesRegex(ProtocolError, "duplicate expert"):
            rows = [
                route_row(slot=0, expert=0, target_rank=0),
                route_row(slot=1, expert=0, target_rank=0),
            ]
            validate_route_rows(
                rows,
                expected_topk={"toy": 2},
                expected_tokenizers={"toy": "d" * 64},
                num_experts={"toy": 4},
                expected_dispatch_bindings=RANK_BINDINGS,
                expected_events=expected_events(rows, 2),
            )

    def test_prompt_hash_cannot_cross_splits(self) -> None:
        evaluation = route_row(slot=0, expert=0, target_rank=0)
        calibration = replace(
            evaluation,
            tenant_id="tenant-attacker",
            request_id="request-1",
            document_id="document-1",
            split="calibration",
            role="attacker",
            traffic_class="ADV_TEXT",
            dispatch_event_id="c" * 64,
        )
        with self.assertRaisesRegex(ProtocolError, "prompt hashes cross"):
            validate_route_rows(
                [evaluation, calibration],
                expected_topk={"toy": 1},
                expected_tokenizers={"toy": "d" * 64},
                num_experts={"toy": 4},
                expected_dispatch_bindings=RANK_BINDINGS,
                expected_events=expected_events([evaluation, calibration], 1),
            )

    def test_rank_binding_is_explicit(self) -> None:
        raw = {
            **route_row(slot=0, expert=0, target_rank=0).__dict__,
            "placement_id": "UNRESOLVED_PLACEMENT",
            "target_rank": -1,
        }
        with self.assertRaisesRegex(ProtocolError, "formal rank metrics"):
            RouteContribution.from_mapping(raw, require_rank_binding=True)
        development = RouteContribution.from_mapping(raw, require_rank_binding=False)
        self.assertEqual(development.target_rank, -1)

    def test_replica_membership_allows_only_frozen_expert_rank_pairs(self) -> None:
        rows = [
            route_row(slot=0, expert=0, target_rank=0, chunk=0),
            route_row(slot=1, expert=1, target_rank=1, chunk=0),
            route_row(slot=0, expert=0, target_rank=1, chunk=1),
            route_row(slot=1, expert=2, target_rank=0, chunk=1),
        ]
        validate_route_rows(
            rows,
            expected_topk={"toy": 2},
            expected_tokenizers={"toy": "d" * 64},
            num_experts={"toy": 4},
            expected_dispatch_bindings=RANK_BINDINGS,
            expected_events=expected_events(rows, 2),
        )
        invalid = [replace(rows[0], target_rank=1), *rows[1:]]
        with self.assertRaisesRegex(ProtocolError, "absent from the frozen"):
            validate_route_rows(
                invalid,
                expected_topk={"toy": 2},
                expected_tokenizers={"toy": "d" * 64},
                num_experts={"toy": 4},
                expected_dispatch_bindings=RANK_BINDINGS,
                expected_events=expected_events(invalid, 2),
            )

    def test_token_id_must_be_stable_across_layers(self) -> None:
        layer_zero = [
            route_row(slot=0, expert=0, target_rank=0),
            route_row(slot=1, expert=1, target_rank=1),
        ]
        layer_one = [
            replace(layer_zero[0], layer_id=1, token_id=999, dispatch_event_id="e" * 64),
            replace(layer_zero[1], layer_id=1, token_id=999, dispatch_event_id="f" * 64),
        ]
        with self.assertRaisesRegex(ProtocolError, "token_id changed"):
            validate_route_rows(
                [*layer_zero, *layer_one],
                expected_topk={"toy": 2},
                expected_tokenizers={"toy": "d" * 64},
                num_experts={"toy": 4},
                expected_dispatch_bindings=RANK_BINDINGS,
                expected_events=expected_events([*layer_zero, *layer_one], 2),
            )

    def test_expected_manifest_rejects_truncated_route_ledger(self) -> None:
        rows = [
            route_row(slot=0, expert=0, target_rank=0),
            route_row(slot=1, expert=1, target_rank=1),
        ]
        manifest = expected_events(rows, 2)
        second_event = replace(
            manifest[0],
            token_position=1,
            token_id=101,
        )
        with self.assertRaisesRegex(ProtocolError, "does not close the expected"):
            validate_route_rows(
                rows,
                expected_topk={"toy": 2},
                expected_tokenizers={"toy": "d" * 64},
                num_experts={"toy": 4},
                expected_dispatch_bindings=RANK_BINDINGS,
                expected_events=[*manifest, second_event],
            )

    def test_csv_duplicate_or_unknown_header_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "routes.csv"
            path.write_text(
                ",".join([*ROUTE_COLUMNS, ROUTE_COLUMNS[0]]) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ProtocolError, "duplicate headers"):
                load_route_csv(path, expected_topk={"toy": 2}, require_rank_binding=False)
            path.write_text(
                ",".join([*ROUTE_COLUMNS, "unexpected_field"]) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ProtocolError, "fields mismatch"):
                load_route_csv(path, expected_topk={"toy": 2}, require_rank_binding=False)

    def test_expected_event_duplicate_json_key_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            path.write_text('{"model":"toy","model":"duplicate"}\n', encoding="utf-8")
            with self.assertRaisesRegex(ProtocolError, "duplicate expected-event JSON key"):
                load_expected_events_jsonl(path)


if __name__ == "__main__":
    unittest.main()
