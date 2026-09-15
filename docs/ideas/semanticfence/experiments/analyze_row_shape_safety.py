#!/usr/bin/env python3
"""Analyze row-level exactness and safe-pooling headroom in calibration data.

This is a read-only, standard-library analysis.  It deliberately reports call
count bounds rather than latency or serving speedup.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple


RowKey = Tuple[int, int, str]
Cell = Tuple[int, int]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _parse_ms(value: str) -> Tuple[int, ...]:
    result = tuple(sorted({int(item) for item in value.split(",") if item.strip()}))
    if not result or result[0] < 2:
        raise argparse.ArgumentTypeError("oracle batch sizes must be integers >= 2")
    return result


def _best_nested_pooling_plan(safe_counts: Mapping[int, int], ms: Sequence[int]) -> Dict[int, int]:
    """Solve one cell exactly for nested safe sets.

    A row safe at a larger M must also be safe at every smaller requested M.
    The objective is isolated-call reduction: sum((M - 1) * batches[M]).
    """

    descending = tuple(sorted(ms, reverse=True))
    best_key: Optional[Tuple[Any, ...]] = None
    best_plan: Optional[Dict[int, int]] = None

    def search(index: int, used_rows: int, plan: MutableMapping[int, int]) -> None:
        nonlocal best_key, best_plan
        if index == len(descending):
            saved = sum((m - 1) * plan.get(m, 0) for m in descending)
            covered = sum(m * plan.get(m, 0) for m in descending)
            batches = sum(plan.values())
            # The first component is the actual objective.  Remaining fields
            # make ties deterministic without altering the call-count bound.
            key = (saved, covered, -batches) + tuple(plan.get(m, 0) for m in descending)
            if best_key is None or key > best_key:
                best_key = key
                best_plan = dict(plan)
            return

        m = descending[index]
        capacity = safe_counts[m] - used_rows
        if capacity < 0:
            return
        for batch_count in range(capacity // m + 1):
            plan[m] = batch_count
            search(index + 1, used_rows + m * batch_count, plan)
        plan.pop(m, None)

    search(0, 0, {})
    if best_plan is None:
        raise ValueError("no feasible nested pooling plan")
    return {m: best_plan.get(m, 0) for m in sorted(ms)}


def analyze(path: Path, oracle_ms: Sequence[int] = (2, 4, 8, 16)) -> Dict[str, Any]:
    pack_counts: Dict[int, int] = defaultdict(int)
    row_counts: Dict[int, int] = defaultdict(int)
    exact_pack_counts: Dict[int, int] = defaultdict(int)
    safe_rows_by_m: Dict[int, Set[RowKey]] = defaultdict(set)
    present_rows_by_m: Dict[int, Set[RowKey]] = defaultdict(set)
    safe_rows_by_cell_m: Dict[Cell, Dict[int, Set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    repeat_counts: Set[int] = set()
    repeat_stability_violations = 0
    duplicate_row_keys = 0
    m2_contingency = {
        "both_safe": 0,
        "first_only_safe": 0,
        "second_only_safe": 0,
        "neither_safe": 0,
    }

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            m = int(record["m"])
            layer = int(record["layer"])
            expert = int(record["expert_id"])
            row_ids = list(record["row_ids"])
            repeats = list(record["repeat_row_exact"])
            if len(row_ids) != m:
                raise ValueError(f"line {line_number}: len(row_ids) != M")
            if not repeats or any(len(repeat) != m for repeat in repeats):
                raise ValueError(f"line {line_number}: malformed repeat_row_exact")
            repeat_counts.add(len(repeats))

            row_safe: List[bool] = []
            for row_index, row_id in enumerate(row_ids):
                values = [bool(repeat[row_index]) for repeat in repeats]
                if any(value != values[0] for value in values[1:]):
                    repeat_stability_violations += 1
                is_safe = all(values)
                row_safe.append(is_safe)
                key = (layer, expert, str(row_id))
                if key in present_rows_by_m[m]:
                    duplicate_row_keys += 1
                present_rows_by_m[m].add(key)
                if is_safe:
                    safe_rows_by_m[m].add(key)
                    safe_rows_by_cell_m[(layer, expert)][m].add(str(row_id))

            pack_counts[m] += 1
            row_counts[m] += m
            if all(row_safe):
                exact_pack_counts[m] += 1
            if m == 2:
                if row_safe == [True, True]:
                    m2_contingency["both_safe"] += 1
                elif row_safe == [True, False]:
                    m2_contingency["first_only_safe"] += 1
                elif row_safe == [False, True]:
                    m2_contingency["second_only_safe"] += 1
                else:
                    m2_contingency["neither_safe"] += 1

    if duplicate_row_keys:
        raise ValueError(f"found {duplicate_row_keys} duplicate (layer, expert, row_id) keys")
    if not set(oracle_ms).issubset(present_rows_by_m):
        missing = sorted(set(oracle_ms) - set(present_rows_by_m))
        raise ValueError(f"oracle batch sizes absent from input: {missing}")

    per_m: Dict[str, Any] = {}
    for m in sorted(pack_counts):
        safe_count = len(safe_rows_by_m[m])
        per_m[str(m)] = {
            "packs": pack_counts[m],
            "rows": row_counts[m],
            "safe_rows": safe_count,
            "safe_row_rate": _rate(safe_count, row_counts[m]),
            "arrival_order_all_safe_packs": exact_pack_counts[m],
            "arrival_order_all_safe_pack_rate": _rate(exact_pack_counts[m], pack_counts[m]),
        }

    cross_m: List[Dict[str, Any]] = []
    nested_violations: List[Dict[str, Any]] = []
    sorted_ms = sorted(present_rows_by_m)
    for smaller, larger in combinations(sorted_ms, 2):
        shared_safe = safe_rows_by_m[smaller] & safe_rows_by_m[larger]
        larger_present_at_smaller = safe_rows_by_m[larger] & present_rows_by_m[smaller]
        violations = larger_present_at_smaller - safe_rows_by_m[smaller]
        cross_m.append(
            {
                "smaller_m": smaller,
                "larger_m": larger,
                "safe_intersection": len(shared_safe),
                "p_smaller_safe_given_larger_safe": _rate(
                    len(shared_safe), len(larger_present_at_smaller)
                ),
                "p_larger_safe_given_smaller_safe": _rate(
                    len(shared_safe), len(safe_rows_by_m[smaller])
                ),
                "nested_violation_count": len(violations),
            }
        )
        if violations:
            nested_violations.append(
                {"smaller_m": smaller, "larger_m": larger, "count": len(violations)}
            )

    if nested_violations:
        raise ValueError(f"safe-row sets are not nested: {nested_violations}")

    m2_packs = pack_counts[2]
    m2_rows = row_counts[2]
    m2_safe = len(safe_rows_by_m[2])
    p_safe = _rate(m2_safe, m2_rows)
    expected_both_uniform_pairing = (
        m2_packs * m2_safe * (m2_safe - 1) / (m2_rows * (m2_rows - 1))
        if m2_rows > 1
        else 0.0
    )
    independence = {
        **m2_contingency,
        "exactly_one_safe": (
            m2_contingency["first_only_safe"] + m2_contingency["second_only_safe"]
        ),
        "observed_both_safe_rate": _rate(m2_contingency["both_safe"], m2_packs),
        "global_row_safe_rate": p_safe,
        "expected_both_safe_packs_iid": p_safe * p_safe * m2_packs,
        "expected_both_safe_packs_uniform_random_pairing": expected_both_uniform_pairing,
        "observed_to_iid_expected_ratio": _rate(
            m2_contingency["both_safe"], round(p_safe * p_safe * m2_packs, 12)
        ),
        "observed_to_uniform_pairing_expected_ratio": _rate(
            m2_contingency["both_safe"], expected_both_uniform_pairing
        ),
    }

    cells = sorted(safe_rows_by_cell_m)
    batch_counts = {m: 0 for m in oracle_ms}
    cells_with_pooling = 0
    for cell in cells:
        counts = {m: len(safe_rows_by_cell_m[cell][m]) for m in oracle_ms}
        # Validate the per-cell version of the global nesting invariant.
        for smaller, larger in zip(sorted(oracle_ms), sorted(oracle_ms)[1:]):
            if not safe_rows_by_cell_m[cell][larger].issubset(
                safe_rows_by_cell_m[cell][smaller]
            ):
                raise ValueError(f"cell {cell}: safe rows are not nested for M={smaller},{larger}")
        plan = _best_nested_pooling_plan(counts, oracle_ms)
        if any(plan.values()):
            cells_with_pooling += 1
        for m, count in plan.items():
            batch_counts[m] += count

    baseline_calls = m2_rows
    oracle_batches = sum(batch_counts.values())
    covered_rows = sum(m * count for m, count in batch_counts.items())
    remaining_m1_calls = baseline_calls - covered_rows
    saved_calls = sum((m - 1) * count for m, count in batch_counts.items())
    if baseline_calls - (remaining_m1_calls + oracle_batches) != saved_calls:
        raise AssertionError("pooling call accounting is inconsistent")

    m2_only_batches = sum(
        len(safe_rows_by_cell_m[cell][2]) // 2 for cell in cells
    )
    oracle = {
        "interpretation": "offline regrouping upper bound on expert-call count; not measured latency or serving speedup",
        "oracle_batch_sizes": list(sorted(oracle_ms)),
        "layer_expert_cells_with_any_safe_row": len(cells),
        "layer_expert_cells_with_nonzero_pooling": cells_with_pooling,
        "baseline_isolated_m1_calls": baseline_calls,
        "batch_counts": {str(m): batch_counts[m] for m in sorted(batch_counts)},
        "covered_rows": covered_rows,
        "row_coverage": _rate(covered_rows, baseline_calls),
        "remaining_m1_calls": remaining_m1_calls,
        "new_batched_calls": oracle_batches,
        "total_calls": remaining_m1_calls + oracle_batches,
        "saved_calls": saved_calls,
        "call_reduction_upper_bound": _rate(saved_calls, baseline_calls),
        "m2_only_regrouping": {
            "batches": m2_only_batches,
            "covered_rows": 2 * m2_only_batches,
            "row_coverage": _rate(2 * m2_only_batches, baseline_calls),
            "saved_calls": m2_only_batches,
            "call_reduction_upper_bound": _rate(m2_only_batches, baseline_calls),
        },
        "arrival_order_exact_m2": {
            "batches": exact_pack_counts[2],
            "covered_rows": 2 * exact_pack_counts[2],
            "row_coverage": _rate(2 * exact_pack_counts[2], baseline_calls),
            "saved_calls": exact_pack_counts[2],
            "call_reduction_upper_bound": _rate(exact_pack_counts[2], baseline_calls),
        },
    }

    return {
        "schema_version": "semanticfence-row-shape-safety-analysis-v1",
        "input": {"path": str(path), "sha256": sha256_file(path)},
        "invariants": {
            "repeat_counts": sorted(repeat_counts),
            "repeat_stability_violations": repeat_stability_violations,
            "duplicate_row_keys": duplicate_row_keys,
            "safe_sets_nested_across_adjacent_m": True,
        },
        "per_m": per_m,
        "m2_independence_diagnostic": independence,
        "cross_m": cross_m,
        "safe_pooling_oracle": oracle,
        "claim_boundary": [
            "The row-level and nesting statistics are observed on this frozen calibration artifact.",
            "Near-IID pair counts are evidence against, not proof excluding, pair-composition effects.",
            "The pooling result is a perfect-oracle call-count bound and is not a latency or serving result.",
            "Fresh partner-permutation replay is required before claiming row-local generalization.",
        ],
    }


def _markdown(result: Mapping[str, Any]) -> str:
    oracle = result["safe_pooling_oracle"]
    iid = result["m2_independence_diagnostic"]
    lines = [
        "# SemanticFence row-shape safety analysis",
        "",
        f"Input SHA256: `{result['input']['sha256']}`",
        "",
        "## Verdict",
        "",
        "The frozen calibration data are strongly consistent with row/shape-local stability rather than a special-pair effect, but this is not yet a causal proof. Even a perfect row-safety oracle has a limited expert-call-count ceiling on this distribution.",
        "",
        "## M=2 diagnostic",
        "",
        f"- Safe rows: {result['per_m']['2']['safe_rows']}/{result['per_m']['2']['rows']} ({result['per_m']['2']['safe_row_rate']:.4%}).",
        f"- Both-safe arrival-order pairs: {iid['both_safe']}; uniform-random-pairing expectation from the global row rate: {iid['expected_both_safe_packs_uniform_random_pairing']:.2f}.",
        f"- Exactly-one-safe pairs: {iid['exactly_one_safe']} ({iid['first_only_safe']} first-only, {iid['second_only_safe']} second-only).",
        "",
        "## Perfect-oracle call-count bound",
        "",
        f"- Batch counts: {oracle['batch_counts']}.",
        f"- Covered rows: {oracle['covered_rows']}/{oracle['baseline_isolated_m1_calls']} ({oracle['row_coverage']:.4%}).",
        f"- Saved expert calls: {oracle['saved_calls']}/{oracle['baseline_isolated_m1_calls']} ({oracle['call_reduction_upper_bound']:.4%}).",
        "- This is not measured latency or serving speedup.",
        "",
        "## Required falsification",
        "",
        "Replay frozen safe rows with multiple new partners inside the same layer/expert/M cell. A partner-dependent flip falsifies the row-local interpretation; invariance across held-out partners supports a row-safety model.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--oracle-ms", type=_parse_ms, default=(2, 4, 8, 16))
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    args = parser.parse_args()

    result = analyze(args.input, args.oracle_ms)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(_markdown(result), encoding="utf-8")


if __name__ == "__main__":
    main()
