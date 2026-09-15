#!/usr/bin/env python3
"""Independently recompute C8 action-transfer metrics from sealed cell rows."""

from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise ValueError(f"blank JSONL row at line {line_number}")
            rows.append(json.loads(line))
    return rows


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def fraction_payload(value: Fraction | int) -> dict[str, Any]:
    fraction = value if isinstance(value, Fraction) else Fraction(value, 1)
    return {
        "numerator": fraction.numerator,
        "denominator": fraction.denominator,
        "value": float(fraction),
    }


def fraction_from_payload(value: Mapping[str, Any]) -> Fraction:
    return Fraction(int(value["numerator"]), int(value["denominator"]))


def bitset_int(payload: Mapping[str, Any]) -> int:
    if payload.get("encoding") != "packed-bitset-lsb0-v1":
        raise ValueError("wrong packed-bitset encoding")
    count = int(payload["num_elements"])
    raw = bytes.fromhex(str(payload["packed_hex"]))
    if len(raw) != (count + 7) // 8:
        raise ValueError("packed-bitset byte length mismatch")
    value = int.from_bytes(raw, "little")
    valid_mask = (1 << count) - 1
    if value & ~valid_mask:
        raise ValueError("packed-bitset has nonzero padding bits")
    observed_count = bin(value).count("1")
    if observed_count != int(payload["set_bit_count"]):
        raise ValueError("packed-bitset set-bit count mismatch")
    if bool(observed_count) != bool(payload["vector_bitwise_mismatch"]):
        raise ValueError("packed-bitset vector mismatch flag disagrees")
    return value


def final_decomposition(
    unprotected: Mapping[str, Any], action: Mapping[str, Any]
) -> dict[str, int]:
    if int(unprotected["num_elements"]) != int(action["num_elements"]):
        raise ValueError("final-logit bitset lengths differ")
    count = int(unprotected["num_elements"])
    valid_mask = (1 << count) - 1
    old = bitset_int(unprotected)
    new = bitset_int(action)
    recovered = old & ~new & valid_mask
    harmed = new & ~old & valid_mask
    persistent = old & new
    recovered_count = bin(recovered).count("1")
    harmed_count = bin(harmed).count("1")
    persistent_count = bin(persistent).count("1")
    return {
        "final_logit_recovered_count": recovered_count,
        "final_logit_harmed_count": harmed_count,
        "final_logit_persistent_count": persistent_count,
        "final_logit_net_reward": recovered_count - harmed_count,
    }


def route_decomposition(
    unprotected_changed: Sequence[int], action_changed: Sequence[int]
) -> dict[str, Any]:
    old = set(map(int, unprotected_changed))
    new = set(map(int, action_changed))
    recovered = sorted(old - new)
    harmed = sorted(new - old)
    persistent = sorted(old & new)
    return {
        "route_recovered_layers": recovered,
        "route_recovered_count": len(recovered),
        "route_harmed_layers": harmed,
        "route_harmed_count": len(harmed),
        "route_persistent_layers": persistent,
        "route_persistent_count": len(persistent),
        "route_net_reward": len(recovered) - len(harmed),
    }


def validate_action_evidence(
    row: Mapping[str, Any], action: Mapping[str, Any]
) -> None:
    route = route_decomposition(
        row["unprotected_arm"]["changed_layers_vs_R"],
        action["changed_layers_vs_R"],
    )
    if int(action["distance_vs_R"]) != len(action["changed_layers_vs_R"]):
        raise ValueError("action route distance disagrees with changed layers")
    for field, expected in route.items():
        if action[field] != expected:
            raise ValueError(f"action route evidence mismatch for {field}")
    final = final_decomposition(
        row["unprotected_arm"]["final_logits_mismatch_vs_R"],
        action["final_logits_mismatch_vs_R"],
    )
    for field, expected in final.items():
        if int(action[field]) != expected:
            raise ValueError(f"action final-logit evidence mismatch for {field}")


def validate_cell_evidence(row: Mapping[str, Any]) -> None:
    reference_bits = row["reference_arm"]["final_logits_mismatch_vs_R"]
    if bitset_int(reference_bits) != 0:
        raise ValueError("reference arm has a nonzero final-logit mismatch bitset")
    unprotected_bits = row["unprotected_arm"]["final_logits_mismatch_vs_R"]
    bitset_int(unprotected_bits)
    if int(row["unprotected_arm"]["distance_vs_R"]) != len(
        row["unprotected_arm"]["changed_layers_vs_R"]
    ):
        raise ValueError("unprotected route distance disagrees with changed layers")
    validate_action_evidence(row, row["m1_same_rank_arm"])
    for action in row["c8_actions"].values():
        validate_action_evidence(row, action)


def action_metric_sum(
    actions: Sequence[Mapping[str, Any]], divisor: int = 1
) -> dict[str, Any]:
    fields = (
        "route_recovered_count",
        "route_harmed_count",
        "route_persistent_count",
        "route_net_reward",
        "distance_vs_R",
        "final_logit_recovered_count",
        "final_logit_harmed_count",
        "final_logit_persistent_count",
        "final_logit_net_reward",
    )
    result = {
        field: fraction_payload(Fraction(sum(int(row[field]) for row in actions), divisor))
        for field in fields
    }
    final_mismatch_bits = sum(
        int(row["final_logits_mismatch_vs_R"]["set_bit_count"]) for row in actions
    )
    final_mismatch_vectors = sum(
        int(bool(row["final_logits_mismatch_vs_R"]["vector_bitwise_mismatch"]))
        for row in actions
    )
    result.update(
        {
            "final_logit_mismatch_element_count": fraction_payload(
                Fraction(final_mismatch_bits, divisor)
            ),
            "final_logit_mismatch_vector_count": fraction_payload(
                Fraction(final_mismatch_vectors, divisor)
            ),
            "action_count": len(actions),
            "expectation_divisor": divisor,
        }
    )
    return result


def choose_c8_oracle_action(row: Mapping[str, Any]) -> Mapping[str, Any]:
    return min(
        row["c8_actions"].values(),
        key=lambda item: (
            -int(item["route_net_reward"]),
            -int(item["route_recovered_count"]),
            int(item["route_harmed_count"]),
            int(item["rank"]),
        ),
    )


def core_aggregates(
    rows: Sequence[Mapping[str, Any]], ranks: Sequence[int]
) -> dict[str, Any]:
    m1 = [row["m1_same_rank_arm"] for row in rows]
    same = [row["c8_actions"][str(int(row["frozen_m1_rank"]))] for row in rows]
    all_c8 = [row["c8_actions"][str(rank)] for row in rows for rank in ranks]
    oracle_actions = [choose_c8_oracle_action(row) for row in rows]
    abstaining = [
        action for action in oracle_actions if int(action["route_net_reward"]) > 0
    ]
    baseline_route_distance = sum(
        int(row["unprotected_arm"]["distance_vs_R"]) for row in rows
    )
    baseline_final_elements = sum(
        int(row["unprotected_arm"]["final_logits_mismatch_vs_R"]["set_bit_count"])
        for row in rows
    )
    baseline_final_vectors = sum(
        int(
            bool(
                row["unprotected_arm"]["final_logits_mismatch_vs_R"][
                    "vector_bitwise_mismatch"
                ]
            )
        )
        for row in rows
    )
    return {
        "baseline": {
            "route_mismatch_count": fraction_payload(baseline_route_distance),
            "route_recovered_count": fraction_payload(0),
            "route_harmed_count": fraction_payload(0),
            "route_net_reward": fraction_payload(0),
            "final_logit_mismatch_element_count": fraction_payload(
                baseline_final_elements
            ),
            "final_logit_mismatch_vector_count": fraction_payload(
                baseline_final_vectors
            ),
            "action_count": 0,
        },
        "m1_same_rank": action_metric_sum(m1),
        "c8_same_rank": action_metric_sum(same),
        "c8_exact_uniform_random_rank": action_metric_sum(all_c8, len(ranks)),
        "c8_forced_best_rank_oracle": action_metric_sum(oracle_actions),
        "c8_abstaining_best_rank_oracle": action_metric_sum(abstaining),
    }


def single_cell_summary(
    row: Mapping[str, Any], ranks: Sequence[int]
) -> dict[str, Any]:
    best = choose_c8_oracle_action(row)
    return {
        "cell_key": str(row["cell_key"]),
        "document_index": int(row["document_index"]),
        "frozen_m1_rank": int(row["frozen_m1_rank"]),
        "c8_best_rank": int(best["rank"]),
        "core": core_aggregates([row], ranks),
    }


def decision_metrics(
    core: Mapping[str, Any],
    lodo: Sequence[Mapping[str, Any]],
    thresholds: Mapping[str, Any],
) -> dict[str, Any]:
    m1_recovered = fraction_from_payload(
        core["m1_same_rank"]["route_recovered_count"]
    )
    same_recovered = fraction_from_payload(
        core["c8_same_rank"]["route_recovered_count"]
    )
    same_net = fraction_from_payload(core["c8_same_rank"]["route_net_reward"])
    random_net = fraction_from_payload(
        core["c8_exact_uniform_random_rank"]["route_net_reward"]
    )
    oracle_net = fraction_from_payload(
        core["c8_forced_best_rank_oracle"]["route_net_reward"]
    )
    oracle_recovered = fraction_from_payload(
        core["c8_forced_best_rank_oracle"]["route_recovered_count"]
    )
    transfer = Fraction(same_recovered, m1_recovered) if m1_recovered else Fraction(0)
    oracle_transfer = (
        Fraction(oracle_recovered, m1_recovered) if m1_recovered else Fraction(0)
    )
    rank_gap = same_net - random_net
    oracle_gap = oracle_net - same_net
    lodo_positive = all(
        fraction_from_payload(item["core"]["c8_same_rank"]["route_net_reward"])
        > 0
        for item in lodo
    )
    lodo_rank_specific = all(
        fraction_from_payload(item["core"]["c8_same_rank"]["route_net_reward"])
        > fraction_from_payload(
            item["core"]["c8_exact_uniform_random_rank"]["route_net_reward"]
        )
        for item in lodo
    )
    low = Fraction(str(thresholds["low_transfer_ratio"]))
    go = Fraction(str(thresholds["go_transfer_ratio"]))
    same_harmed = fraction_from_payload(
        core["c8_same_rank"]["route_harmed_count"]
    )
    if (
        transfer >= go
        and rank_gap > 0
        and same_net > 0
        and same_harmed <= same_recovered
        and lodo_positive
        and lodo_rank_specific
    ):
        candidate = "GO_SHAPEABI_PLUS_STABILITYBUDGET"
    elif transfer <= low and oracle_transfer >= go and oracle_gap > 0:
        candidate = "CONDITIONAL_C8_SPECIFIC_RANK_SPACE"
    elif transfer <= low:
        candidate = "STOP_FIXED_C8_AS_QUALITY_ACTION"
    elif rank_gap <= 0 and same_net > 0:
        candidate = "CONDITIONAL_GLOBAL_SHAPEABI_NOT_SPARSE_SELECTOR"
    else:
        candidate = "CONDITIONAL_ONE_MECHANISM_DIAGNOSTIC_ONLY"
    return {
        "c8_transfer_ratio": fraction_payload(transfer),
        "rank_specificity_gap": fraction_payload(rank_gap),
        "c8_oracle_gap": fraction_payload(oracle_gap),
        "c8_oracle_transfer_ratio": fraction_payload(oracle_transfer),
        "lodo_same_rank_net_positive_all": lodo_positive,
        "lodo_rank_specificity_positive_all": lodo_rank_specific,
        "low_transfer_ratio": fraction_payload(low),
        "go_transfer_ratio": fraction_payload(go),
        "gate_candidate": candidate,
    }


def recompute_metrics(
    rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> dict[str, Any]:
    ranks = list(map(int, config["action_space"]["candidate_ranks"]))
    expected_cells = int(config["cohort"]["expected_unique_cells"])
    expected_docs = int(config["cohort"]["expected_documents"])
    if len(rows) != expected_cells:
        raise ValueError("wrong cell count")
    if len({str(row["cell_key"]) for row in rows}) != len(rows):
        raise ValueError("duplicate cells")
    if len({int(row["document_index"]) for row in rows}) != expected_docs:
        raise ValueError("wrong document count")
    if any(row.get("integrity_status") != "PASS" for row in rows):
        raise ValueError("failed cell row")
    if any(set(map(int, row["c8_actions"])) != set(ranks) for row in rows):
        raise ValueError("incomplete C8 action surface")
    for row in rows:
        validate_cell_evidence(row)
    core = core_aggregates(rows, ranks)
    all_actions = [row["c8_actions"][str(rank)] for row in rows for rank in ranks]
    action_level = {
        **action_metric_sum(all_actions),
        "positive_net_action_count": sum(
            int(int(action["route_net_reward"]) > 0) for action in all_actions
        ),
        "zero_net_action_count": sum(
            int(int(action["route_net_reward"]) == 0) for action in all_actions
        ),
        "negative_net_action_count": sum(
            int(int(action["route_net_reward"]) < 0) for action in all_actions
        ),
    }
    documents = sorted({int(row["document_index"]) for row in rows})
    per_document = [
        {
            "document_index": document,
            "cell_count": sum(
                int(int(row["document_index"]) == document) for row in rows
            ),
            "core": core_aggregates(
                [row for row in rows if int(row["document_index"]) == document],
                ranks,
            ),
        }
        for document in documents
    ]
    lodo = [
        {
            "left_out_document_index": document,
            "remaining_cell_count": sum(
                int(int(row["document_index"]) != document) for row in rows
            ),
            "core": core_aggregates(
                [row for row in rows if int(row["document_index"]) != document],
                ranks,
            ),
        }
        for document in documents
    ]
    return {
        "cell_count": len(rows),
        "document_count": len(documents),
        "candidate_action_count": len(all_actions),
        "core": core,
        "action_level": action_level,
        "unique_cell_level": [single_cell_summary(row, ranks) for row in rows],
        "per_document": per_document,
        "leave_one_document_out": lodo,
        "decision": decision_metrics(core, lodo, config["thresholds"]),
    }


def differences(expected: Any, observed: Any, path: str = "metrics") -> list[str]:
    if type(expected) is not type(observed):
        return [f"{path}: type {type(expected).__name__} != {type(observed).__name__}"]
    if isinstance(expected, Mapping):
        problems: list[str] = []
        if set(expected) != set(observed):
            problems.append(
                f"{path}: keys {sorted(expected)} != {sorted(observed)}"
            )
        for key in sorted(set(expected).intersection(observed)):
            problems.extend(differences(expected[key], observed[key], f"{path}.{key}"))
        return problems
    if isinstance(expected, list):
        if len(expected) != len(observed):
            return [f"{path}: length {len(expected)} != {len(observed)}"]
        problems = []
        for index, (left, right) in enumerate(zip(expected, observed)):
            problems.extend(differences(left, right, f"{path}[{index}]"))
        return problems
    return [] if expected == observed else [f"{path}: {expected!r} != {observed!r}"]


def write_json_new(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell-results", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"refusing to reuse output path {args.output}")
    rows = load_jsonl(args.cell_results)
    summary = load_json(args.summary)
    config = load_json(args.config)
    recomputed = recompute_metrics(rows, config)
    mismatches = differences(summary["metrics"], recomputed)
    result = {
        "schema_version": "stablebatch-c8-action-transfer-independent-recompute-v1",
        "status": "PASS" if not mismatches else "FAIL",
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "cell_results_sha256": sha256_file(args.cell_results),
        "summary_sha256": sha256_file(args.summary),
        "config_sha256": sha256_file(args.config),
        "recomputed_metrics_sha256": hashlib.sha256(
            canonical_bytes(recomputed)
        ).hexdigest(),
        "recomputed_metrics": recomputed,
    }
    write_json_new(args.output, result)
    return 0 if not mismatches else 1


if __name__ == "__main__":
    raise SystemExit(main())
