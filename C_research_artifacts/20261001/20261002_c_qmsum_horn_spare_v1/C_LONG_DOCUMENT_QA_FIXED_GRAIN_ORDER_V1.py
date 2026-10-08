#!/usr/bin/env python3
"""Fixed first-32-token prefix-group release quantum over a frozen density order."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from C_LONG_DOCUMENT_QA_DENSITY_ORDER_V1 import density_order

WORKLOAD_SHA = "3e34bc8a46abc1329744e22b6c97b18582a540f0abd98de55f012f560aac92fc"
PREFIX_TOKENS = 32


def fixed_grain_order(sequences: list[list[int]], q: int | str):
    """Return a new-request enqueue order; running decodes are never gated."""
    if q != "whole" and (type(q) is not int or q not in (1, 2)):
        raise ValueError("q must be 'whole', 1, or 2")
    if not sequences or any(len(ids) < PREFIX_TOKENS for ids in sequences):
        raise ValueError("every prompt must contain at least 32 token IDs")
    base = density_order(sequences)
    groups: dict[tuple[int, ...], list[int]] = {}
    previous = None
    for index in base:
        key = tuple(sequences[index][:PREFIX_TOKENS])
        if key != previous and key in groups:
            raise ValueError("first-32-token group is not contiguous in density order")
        groups.setdefault(key, []).append(index)
        previous = key
    ordered_groups = list(groups.values())
    if q == "whole":
        order = [index for group in ordered_groups for index in group]
        if order != base:
            raise AssertionError("whole-group order differs from density order")
    else:
        order = [index for offset in range(0, max(map(len, ordered_groups)), q)
                 for group in ordered_groups for index in group[offset:offset + q]]
    if sorted(order) != list(range(len(sequences))):
        raise AssertionError("release order omitted or duplicated a request")
    receipt = dict(quantum=q, requests=len(sequences),
        group_count=len(ordered_groups),
        group_sizes=[len(group) for group in ordered_groups],
        groups=[dict(first32_sha256=hashlib.sha256(json.dumps(
            list(key), separators=(",", ":")).encode()).hexdigest(),
            source_indices_in_density_order=group)
            for key, group in groups.items()],
        source_indices_in_submission_order=order,
        changed_positions_vs_density=sum(base[i] != index
                                         for i, index in enumerate(order)),
        scope="New-request enqueue order only; all external arrivals remain zero; no active-count cap, decode pause, preemption, output or gold inputs")
    return order, receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    path = args.input_dir / "workload.json"
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != WORKLOAD_SHA:
        raise ValueError("frozen full150 workload SHA differs")
    rows = json.loads(raw)["requests"]
    if len(rows) != 150 or [r["source_index"] for r in rows] != list(range(150)):
        raise ValueError("frozen source-order inventory differs")
    sequences = [r["prompt_token_ids"] for r in rows]
    arms = {str(q): fixed_grain_order(sequences, q)[1] for q in ("whole", 1, 2)}
    if arms["whole"]["source_indices_in_submission_order"] != density_order(sequences):
        raise AssertionError("whole-group baseline identity differs")
    report = dict(schema="c-longqa-fixed-grain-order-v1",
        workload_sha256=WORKLOAD_SHA, prefix_group_rule="identical first 32 prompt token IDs",
        request_count=150, arms=arms,
        interpretation="Fixed operational prefix groups; this is not BatchLLM, k-LPM, DLPM, or a novelty claim")
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, ensure_ascii=False)
        stream.write("\n")
    print(json.dumps({q: dict(groups=arms[q]["group_count"],
        sizes=arms[q]["group_sizes"],
        changed_positions=arms[q]["changed_positions_vs_density"])
        for q in arms}))


if __name__ == "__main__":
    main()
