#!/usr/bin/env python3
"""Freeze the M1-positive cohort for the C8 action-transfer experiment.

This builder is deliberately CPU-only.  It reads the sealed M1 oracle ledger and
its summary, recomputes every route metric from layer-id sets, and writes a
deterministic manifest.  It has no input for, and never reads, a C8 outcome.
"""

from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "stablebatch-c8-action-transfer-cohort-v1"
STATUS = "SEALED_PRE_C8_OUTCOMES"
EXPECTED_RANKS = tuple(range(8))
FROZEN_COUNTS = {
    "source_cells": 240,
    "primary_unique_cells": 33,
    "raw_positive_rank_actions": 139,
    "multi_positive_cells": 27,
}


class CohortError(RuntimeError):
    """The frozen source evidence cannot produce the requested cohort."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise CohortError(f"expected JSON object in {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise CohortError(f"blank JSONL row at {path}:{line_number}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise CohortError(f"expected object at {path}:{line_number}")
            rows.append(value)
    if not rows:
        raise CohortError(f"empty oracle ledger: {path}")
    return rows


def write_json_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CohortError(message)


def exact_fraction(numerator: int, denominator: int) -> dict[str, int]:
    value = Fraction(numerator, denominator)
    return {"numerator": value.numerator, "denominator": value.denominator}


def integer_layer_set(value: Any, label: str) -> set[int]:
    require(isinstance(value, list), f"{label} must be a list")
    layers = [int(item) for item in value]
    require(len(layers) == len(set(layers)), f"{label} contains duplicate layers")
    require(layers == sorted(layers), f"{label} is not sorted")
    return set(layers)


def stable_cell_key(row: Mapping[str, Any]) -> str:
    return f"{row['victim_id']}|layer={int(row['layer']):02d}"


def final_logit_metrics(
    reference_hash: str, unprotected_hash: str, action_hash: str
) -> dict[str, Any]:
    unprotected_mismatch = unprotected_hash != reference_hash
    action_mismatch = action_hash != reference_hash
    return {
        "final_logits_sha256": action_hash,
        "final_logit_mismatch_vs_reference": action_mismatch,
        "final_logit_recovered": bool(unprotected_mismatch and not action_mismatch),
        "final_logit_harmed": bool(not unprotected_mismatch and action_mismatch),
    }


def build_action_ledger(
    row: Mapping[str, Any], rank: int, unprotected_layers: set[int]
) -> dict[str, Any]:
    cell = str(row.get("cell_id", stable_cell_key(row)))
    try:
        action = row["actions"][str(rank)]
        reference_hash = str(row["reference_arm"]["final_logits_sha256"])
        unprotected_hash = str(row["unprotected_arm"]["final_logits_sha256"])
        action_hash = str(action["arm"]["final_logits_sha256"])
    except (KeyError, TypeError) as error:
        raise CohortError(f"{cell}: malformed rank-{rank} action") from error

    changed_layers = integer_layer_set(
        action.get("changed_layers_vs_R"),
        f"{cell} rank {rank} changed_layers_vs_R",
    )
    recovered_layers = sorted(unprotected_layers - changed_layers)
    harmed_layers = sorted(changed_layers - unprotected_layers)
    remaining_layers = sorted(unprotected_layers & changed_layers)
    recovered = len(recovered_layers)
    harmed = len(harmed_layers)
    net = recovered - harmed

    require(int(action.get("rank", rank)) == rank, f"{cell}: action rank mismatch")
    require(
        int(action["distance_vs_R"]) == len(changed_layers),
        f"{cell} rank {rank}: stored route distance mismatch",
    )
    require(
        int(action["reward"]) == net,
        f"{cell} rank {rank}: stored reward is not recovered minus harmed",
    )
    require(
        int(action["expert_id"]) == int(row["expert_ids"][rank]),
        f"{cell} rank {rank}: expert identity mismatch",
    )
    require(
        bool(action["full_restoration"])
        == bool(unprotected_layers and not changed_layers),
        f"{cell} rank {rank}: full-restoration flag mismatch",
    )

    return {
        "rank": rank,
        "expert_id": int(action["expert_id"]),
        "changed_layers_vs_R": sorted(changed_layers),
        "distance_vs_R": len(changed_layers),
        "route_recovered_layers": recovered_layers,
        "route_recovered_count": recovered,
        "route_harmed_layers": harmed_layers,
        "route_harmed_count": harmed,
        "remaining_original_mismatch_layers": remaining_layers,
        "route_net_reward": net,
        "full_restoration": bool(action["full_restoration"]),
        **final_logit_metrics(reference_hash, unprotected_hash, action_hash),
    }


def exact_random_summary(actions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rank_count = len(actions)
    require(rank_count == len(EXPECTED_RANKS), "exact-random requires all eight ranks")
    recovered_sum = sum(int(action["route_recovered_count"]) for action in actions)
    harmed_sum = sum(int(action["route_harmed_count"]) for action in actions)
    net_sum = sum(int(action["route_net_reward"]) for action in actions)
    final_recovered_sum = sum(
        int(bool(action["final_logit_recovered"])) for action in actions
    )
    final_harmed_sum = sum(
        int(bool(action["final_logit_harmed"])) for action in actions
    )
    return {
        "rank_count": rank_count,
        "route_recovered_sum_over_ranks": recovered_sum,
        "route_recovered_expected": exact_fraction(recovered_sum, rank_count),
        "route_harmed_sum_over_ranks": harmed_sum,
        "route_harmed_expected": exact_fraction(harmed_sum, rank_count),
        "route_net_sum_over_ranks": net_sum,
        "route_net_expected": exact_fraction(net_sum, rank_count),
        "final_logit_recovered_sum_over_ranks": final_recovered_sum,
        "final_logit_recovered_expected": exact_fraction(
            final_recovered_sum, rank_count
        ),
        "final_logit_harmed_sum_over_ranks": final_harmed_sum,
        "final_logit_harmed_expected": exact_fraction(final_harmed_sum, rank_count),
    }


def build_positive_cell(row: Mapping[str, Any]) -> dict[str, Any] | None:
    cell_id = str(row.get("cell_id", stable_cell_key(row)))
    require(row.get("integrity_status") == "PASS", f"{cell_id}: integrity is not PASS")
    require(
        set(row.get("actions", {})) == {str(rank) for rank in EXPECTED_RANKS},
        f"{cell_id}: action ledger is not exactly ranks 0..7",
    )
    require(len(row["expert_ids"]) == len(EXPECTED_RANKS), f"{cell_id}: expert count")
    require(len(row["gate_weights"]) == len(EXPECTED_RANKS), f"{cell_id}: gate count")

    unprotected_layers = integer_layer_set(
        row.get("unprotected_changed_layers_vs_R"),
        f"{cell_id} unprotected_changed_layers_vs_R",
    )
    require(
        int(row["unprotected_distance_vs_R"]) == len(unprotected_layers),
        f"{cell_id}: unprotected route distance mismatch",
    )
    actions = [build_action_ledger(row, rank, unprotected_layers) for rank in EXPECTED_RANKS]
    positive_actions = [action for action in actions if int(action["route_net_reward"]) > 0]

    stored_positive = int(row["forced_oracle_reward"]) > 0
    require(
        stored_positive == bool(positive_actions),
        f"{cell_id}: recomputed and stored positive-cell status differ",
    )
    if not positive_actions:
        return None

    # This selection intentionally does not consult forced_oracle_rank/reward.
    primary = min(
        positive_actions,
        key=lambda action: (
            -int(action["route_recovered_count"]),
            int(action["route_harmed_count"]),
            int(action["rank"]),
        ),
    )
    primary_rank = int(primary["rank"])
    require(
        primary_rank == int(row["forced_oracle_rank"]),
        f"{cell_id}: frozen transfer tie-break differs from forced_oracle_rank",
    )
    require(
        int(primary["route_net_reward"]) == int(row["forced_oracle_reward"]),
        f"{cell_id}: primary net reward differs from forced_oracle_reward",
    )
    abstaining = row.get("abstaining_oracle_action", {})
    require(
        abstaining.get("action") == "protect_rank"
        and int(abstaining.get("rank")) == primary_rank
        and int(abstaining.get("reward")) == int(primary["route_net_reward"]),
        f"{cell_id}: abstaining-oracle record differs from primary action",
    )
    confirmation = row.get("selected_positive_action_confirmation")
    require(isinstance(confirmation, dict), f"{cell_id}: missing positive confirmation")
    require(confirmation.get("status") == "PASS", f"{cell_id}: confirmation failed")
    require(int(confirmation.get("rank")) == primary_rank, f"{cell_id}: confirmation rank")
    require(
        sorted(map(int, confirmation.get("changed_layers_vs_R", [])))
        == primary["changed_layers_vs_R"],
        f"{cell_id}: confirmation route outcome mismatch",
    )

    exact_random = exact_random_summary(actions)
    reference_hash = str(row["reference_arm"]["final_logits_sha256"])
    unprotected_hash = str(row["unprotected_arm"]["final_logits_sha256"])
    result = {
        "cell_id": cell_id,
        "cell_index": int(row["cell_index"]),
        "cell_key": stable_cell_key(row),
        "document_index": int(row["document_index"]),
        "victim_id": str(row["victim_id"]),
        "layer": int(row["layer"]),
        "flat_token_idx": int(row["flat_token_idx"]),
        "window_token_ids": list(map(int, row["window_token_ids"])),
        "window_token_ids_sha256": str(row["window_token_ids_sha256"]),
        "target_hidden_sha256": str(row["target_hidden_sha256"]),
        "target_router_logits_sha256": str(row["target_router_logits_sha256"]),
        "expert_ids": list(map(int, row["expert_ids"])),
        "gate_weights": list(map(float, row["gate_weights"])),
        "current_layer_topk_cutoff_margin": float(
            row["current_layer_topk_cutoff_margin"]
        ),
        "sidecall_m_order_per_rank": list(map(int, row["sidecall_m_order_per_rank"])),
        "source_maxgate_rank": int(row["source_maxgate_rank"]),
        "source_shuffled_rank": int(row["source_shuffled_rank"]),
        "reference_final_logits_sha256": reference_hash,
        "unprotected_final_logits_sha256": unprotected_hash,
        "unprotected_final_logit_mismatch_vs_reference": unprotected_hash
        != reference_hash,
        "unprotected_changed_layers_vs_R": sorted(unprotected_layers),
        "unprotected_distance_vs_R": len(unprotected_layers),
        "m1_raw_positive_ranks": [int(action["rank"]) for action in positive_actions],
        "m1_raw_positive_rank_count": len(positive_actions),
        "m1_has_multiple_positive_ranks": len(positive_actions) > 1,
        "m1_primary": {
            **primary,
            "selection_tuple": [
                -int(primary["route_recovered_count"]),
                int(primary["route_harmed_count"]),
                primary_rank,
            ],
            "confirmation_signature_sha256": str(confirmation["signature_sha256"]),
            "confirmation_status": str(confirmation["status"]),
        },
        "m1_actions": actions,
        "m1_exact_uniform_random_rank": exact_random,
    }
    return result


def sum_exact_random(cells: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rank_count = len(EXPECTED_RANKS)
    field_names = (
        "route_recovered",
        "route_harmed",
        "route_net",
        "final_logit_recovered",
        "final_logit_harmed",
    )
    result: dict[str, Any] = {"rank_count": rank_count}
    for name in field_names:
        source = f"{name}_sum_over_ranks"
        total = sum(
            int(cell["m1_exact_uniform_random_rank"][source]) for cell in cells
        )
        result[source] = total
        result[f"{name}_expected"] = exact_fraction(total, rank_count)
    return result


def summarize_cells(cells: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    primary = [cell["m1_primary"] for cell in cells]
    return {
        "cell_count": len(cells),
        "unprotected_route_mismatch_count": sum(
            int(cell["unprotected_distance_vs_R"]) for cell in cells
        ),
        "route_recovered_count": sum(
            int(action["route_recovered_count"]) for action in primary
        ),
        "route_harmed_count": sum(
            int(action["route_harmed_count"]) for action in primary
        ),
        "route_net_reward": sum(int(action["route_net_reward"]) for action in primary),
        "remaining_route_mismatch_count": sum(
            int(action["distance_vs_R"]) for action in primary
        ),
        "full_restoration_cell_count": sum(
            int(bool(action["full_restoration"])) for action in primary
        ),
        "unprotected_final_logit_mismatch_count": sum(
            int(bool(cell["unprotected_final_logit_mismatch_vs_reference"]))
            for cell in cells
        ),
        "final_logit_recovered_count": sum(
            int(bool(action["final_logit_recovered"])) for action in primary
        ),
        "final_logit_harmed_count": sum(
            int(bool(action["final_logit_harmed"])) for action in primary
        ),
    }


def build_per_document(cells: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    identities = sorted(
        {(int(cell["document_index"]), str(cell["victim_id"])) for cell in cells}
    )
    per_document: list[dict[str, Any]] = []
    for document_index, victim_id in identities:
        document_cells = [
            cell
            for cell in cells
            if int(cell["document_index"]) == document_index
            and str(cell["victim_id"]) == victim_id
        ]
        per_document.append(
            {
                "document_index": document_index,
                "victim_id": victim_id,
                "cell_keys": [str(cell["cell_key"]) for cell in document_cells],
                "m1_primary": summarize_cells(document_cells),
                "m1_exact_uniform_random_rank": sum_exact_random(document_cells),
            }
        )
    return per_document


def build_leave_one_document_out(
    cells: Sequence[Mapping[str, Any]], per_document: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for document in per_document:
        held_out_victim = str(document["victim_id"])
        remaining = [cell for cell in cells if str(cell["victim_id"]) != held_out_victim]
        result.append(
            {
                "held_out_document_index": int(document["document_index"]),
                "held_out_victim_id": held_out_victim,
                "m1_primary": summarize_cells(remaining),
                "m1_exact_uniform_random_rank": sum_exact_random(remaining),
            }
        )
    return result


def validate_summary(
    summary: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], cells: Sequence[Mapping[str, Any]]
) -> None:
    require(summary.get("status") == "COMPLETE", "oracle summary is not COMPLETE")
    require(int(summary["cell_count"]) == len(rows), "summary source-cell count mismatch")
    require(
        int(summary["candidate_action_count"]) == len(rows) * len(EXPECTED_RANKS),
        "summary candidate-action count mismatch",
    )
    require(
        int(summary["positive_oracle_cell_count"]) == len(cells),
        "summary positive-cell count mismatch",
    )
    require(
        int(summary["abstaining_oracle_action_budget"]) == len(cells),
        "summary action budget mismatch",
    )
    primary = summarize_cells(cells)
    exact_random = sum_exact_random(cells)
    require(
        int(summary["abstaining_oracle_total_reward"])
        == int(primary["route_net_reward"]),
        "summary oracle reward mismatch",
    )
    require(
        summary["budget_matched_conditional_random_expected_reward_exact"]
        == exact_random["route_net_expected"],
        "summary conditional exact-random reward mismatch",
    )
    rank_counts = {
        str(rank): sum(int(cell["m1_primary"]["rank"]) == rank for cell in cells)
        for rank in EXPECTED_RANKS
    }
    require(
        summary["positive_oracle_rank_counts"] == rank_counts,
        "summary positive-rank counts mismatch",
    )


def build_manifest(
    oracle_ledger_path: Path,
    oracle_summary_path: Path,
    *,
    enforce_frozen_counts: bool = True,
) -> dict[str, Any]:
    rows = load_jsonl(oracle_ledger_path)
    summary = load_json(oracle_summary_path)
    source_keys = [stable_cell_key(row) for row in rows]
    require(len(source_keys) == len(set(source_keys)), "oracle ledger has duplicate cell keys")

    cells = [cell for row in rows if (cell := build_positive_cell(row)) is not None]
    cells.sort(key=lambda cell: (str(cell["victim_id"]), int(cell["layer"])))
    require(
        len(cells) == len({str(cell["cell_key"]) for cell in cells}),
        "primary cohort is not unique-cell",
    )
    raw_positive_count = sum(int(cell["m1_raw_positive_rank_count"]) for cell in cells)
    multi_positive_count = sum(
        int(bool(cell["m1_has_multiple_positive_ranks"])) for cell in cells
    )
    counts = {
        "source_cells": len(rows),
        "candidate_ranks_per_cell": len(EXPECTED_RANKS),
        "source_candidate_actions": len(rows) * len(EXPECTED_RANKS),
        "primary_unique_cells": len(cells),
        "raw_positive_rank_actions": raw_positive_count,
        "multi_positive_cells": multi_positive_count,
        "distinct_documents": len({str(cell["victim_id"]) for cell in cells}),
    }
    if enforce_frozen_counts:
        for field, expected in FROZEN_COUNTS.items():
            require(int(counts[field]) == expected, f"frozen count {field} != {expected}")

    validate_summary(summary, rows, cells)
    per_document = build_per_document(cells)
    deterministic_content = {
        "schema_version": SCHEMA_VERSION,
        "status": STATUS,
        "experiment": "M1-positive cohort C8 action-transfer test",
        "c8_outcomes_read": False,
        "claim_boundary": (
            "Selected retrospectively on frozen M1-positive cells; this is an "
            "action-transfer cohort, not an unbiased population sample."
        ),
        "selection": {
            "eligible_action": "recomputed M1 route_net_reward > 0",
            "primary_unit": "unique victim_id x layer cell",
            "primary_tie_break": [
                "route_recovered_count descending",
                "route_harmed_count ascending",
                "rank ascending",
            ],
            "selection_uses_c8_outcome": False,
            "candidate_ranks": list(EXPECTED_RANKS),
        },
        "source": {
            "oracle_ledger": {
                "path": oracle_ledger_path.as_posix(),
                "sha256": sha256_file(oracle_ledger_path),
            },
            "oracle_summary": {
                "path": oracle_summary_path.as_posix(),
                "sha256": sha256_file(oracle_summary_path),
            },
            "oracle_summary_schema_version": str(summary.get("schema_version")),
            "oracle_summary_verdict": str(summary.get("verdict")),
        },
        "counts": counts,
        "m1_primary_aggregate": summarize_cells(cells),
        "m1_exact_uniform_random_rank": sum_exact_random(cells),
        "per_document": per_document,
        "leave_one_document_out": build_leave_one_document_out(cells, per_document),
        "cells": cells,
    }
    return {
        **deterministic_content,
        "deterministic_content_sha256": hashlib.sha256(
            canonical_json_bytes(deterministic_content)
        ).hexdigest(),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-ledger", required=True, type=Path)
    parser.add_argument("--oracle-summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = build_manifest(args.oracle_ledger, args.oracle_summary)
    write_json_new(args.output, manifest)
    print(
        json.dumps(
            {
                "output": args.output.as_posix(),
                "status": manifest["status"],
                "primary_unique_cells": manifest["counts"]["primary_unique_cells"],
                "raw_positive_rank_actions": manifest["counts"][
                    "raw_positive_rank_actions"
                ],
                "deterministic_content_sha256": manifest[
                    "deterministic_content_sha256"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
