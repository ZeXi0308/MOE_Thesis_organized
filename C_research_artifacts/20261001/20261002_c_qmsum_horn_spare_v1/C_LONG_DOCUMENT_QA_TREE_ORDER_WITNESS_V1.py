#!/usr/bin/env python3
"""Check a concrete non-contiguous tree-order witness in the infinite-cache proxy."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

from C_LONG_DOCUMENT_QA_DENSITY_ORDER_V1 import density_order
from C_LONG_DOCUMENT_QA_FIXED_GRAIN_ORDER_V1 import fixed_grain_order

WORKLOAD_SHA = "3e34bc8a46abc1329744e22b6c97b18582a540f0abd98de55f012f560aac92fc"
MOVE_FROM, MOVE_TO = 29, 57  # Zero-based request positions in whole density order.


def make_tree(sequences):
    nodes = {}  # Exact token prefixes at reusable 16-token boundaries.
    paths, tails, parents = [], [], []
    for tokens in sequences:
        n = (len(tokens) - 1) // 16
        path, parent = [], None
        for level in range(1, n + 1):
            key = tuple(tokens[:16 * level])
            if key not in nodes:
                nodes[key] = len(nodes)
                parents.append(parent)
            current = nodes[key]
            if parents[current] != parent:
                raise ValueError("shared prefix has inconsistent parent")
            path.append(current)
            parent = current
        paths.append(path)
        tails.append(len(tokens) - 16 * n)
    if any(not 1 <= tail <= 16 for tail in tails):
        raise ValueError("invalid exact non-reusable tail")
    return paths, tails, parents


def score(order, paths, tails, parents):
    if sorted(order) != list(range(len(paths))):
        raise ValueError("not a complete request permutation")
    done, elapsed, completion_sum = set(), 0, 0
    for request in order:
        for node in paths[request]:
            if node not in done:
                parent = parents[node]
                if parent is not None and parent not in done:
                    raise ValueError("root-to-leaf precedence violated")
                done.add(node)
                elapsed += 16
        elapsed += tails[request]
        completion_sum += elapsed
    if len(done) != len(parents):
        raise ValueError("some reusable prefix jobs were skipped")
    return dict(total_unique_prompt_work=elapsed,
                sum_prompt_completion_work=completion_sum,
                mean_prompt_completion_work=completion_sum / len(order),
                unique_reusable_block_jobs=len(parents),
                tail_jobs=len(order), root_to_leaf_valid=True)


def contiguity(order, paths, sequences):
    positions = defaultdict(list)
    for position, request in enumerate(order):
        for node in paths[request]:
            positions[node].append(position)
    violations = sum(max(ps) - min(ps) + 1 != len(ps) for ps in positions.values())
    keys = [tuple(sequences[request][:32]) for request in order]
    segments = 1 + sum(left != right for left, right in zip(keys, keys[1:]))
    return dict(noncontiguous_prefix_subtrees=violations, first32_group_segments=segments)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.input_dir / "workload.json"
    raw = source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != WORKLOAD_SHA:
        raise ValueError("frozen LongBench150 workload SHA differs")
    rows = json.loads(raw)["requests"]
    if len(rows) != 150 or [r["source_index"] for r in rows] != list(range(150)):
        raise ValueError("frozen source-order inventory differs")
    sequences = [row["prompt_token_ids"] for row in rows]
    paths, tails, parents = make_tree(sequences)
    whole = density_order(sequences)
    moved_request = whole[MOVE_FROM]
    witness = whole.copy()
    witness.insert(MOVE_TO, witness.pop(MOVE_FROM))
    orders = dict(whole=whole,
                  q1=fixed_grain_order(sequences, 1)[0],
                  q2=fixed_grain_order(sequences, 2)[0],
                  witness=witness)
    results = {name: dict(score(order, paths, tails, parents),
                          **contiguity(order, paths, sequences))
               for name, order in orders.items()}
    if (results["whole"]["sum_prompt_completion_work"] != 23101594
            or results["q1"]["sum_prompt_completion_work"] != 34424265
            or results["q2"]["sum_prompt_completion_work"] != 27990815
            or len({r["total_unique_prompt_work"] for r in results.values()}) != 1
            or results["whole"]["noncontiguous_prefix_subtrees"] != 0
            or results["witness"]["sum_prompt_completion_work"] >=
               results["whole"]["sum_prompt_completion_work"]):
        raise ValueError("frozen proxy or witness benefit differs")
    report = dict(schema="c-longqa-tree-order-witness-v1",
                  workload_sha256=WORKLOAD_SHA, requests=150,
                  model="Serial one-pass 16-token prefix jobs plus exact private tails; infinite cache; all arrivals zero; each terminal tail weight one; no EOS or GPU service model",
                  discovery=dict(
                      method="Exploratory single-request remove/insert moves from the frozen whole-subtree order",
                      adjacent_swaps_checked=149,
                      adjacent_strict_improvements=0,
                      sampled_moves=3000,
                      random_seed=20261002,
                      witness_selection="One sampled move with a strictly smaller sum of prompt-completion work; neither exhaustive nor claimed best in sample",
                  ),
                  source_sha256=dict(density_order=hashlib.sha256(
                      Path(__file__).with_name("C_LONG_DOCUMENT_QA_DENSITY_ORDER_V1.py").read_bytes()).hexdigest(),
                      fixed_grain_order=hashlib.sha256(
                      Path(__file__).with_name("C_LONG_DOCUMENT_QA_FIXED_GRAIN_ORDER_V1.py").read_bytes()).hexdigest()),
                  witness_move=dict(from_position=MOVE_FROM, to_position=MOVE_TO,
                                    source_index=moved_request,
                                    request_id=rows[moved_request]["request_id"]),
                  results=results, witness_order=witness,
                  gain_vs_whole_sum_work=(results["whole"]["sum_prompt_completion_work"]
                                          - results["witness"]["sum_prompt_completion_work"]),
                  gain_vs_whole_mean_work=(results["whole"]["mean_prompt_completion_work"]
                                           - results["witness"]["mean_prompt_completion_work"]),
                  interpretation="Constructive headroom only; not an optimal Horn/Sidney solver, not a GPU policy or service benefit")
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, ensure_ascii=False)
        stream.write("\n")
    print(json.dumps({name: report["results"][name] for name in orders}, indent=2))


if __name__ == "__main__":
    main()
