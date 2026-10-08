#!/usr/bin/env python3
"""Classical maximum-density rooted-subtree reference, not a new GPU policy.

Internal prefix jobs have processing cost 16 and weight zero. Each request
ends in a private positive-cost tail of weight one. Cache is unbounded and
execution serial. A cardinality DP computes maximum rooted-subtree density
exactly; ready jobs are then processed in decreasing Horn density.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import heapq
import itertools
import json
from pathlib import Path
import time

from C_LONG_DOCUMENT_QA_DENSITY_ORDER_V1 import density_order
from C_LONG_DOCUMENT_QA_TREE_ORDER_WITNESS_V1 import (
    WORKLOAD_SHA, make_tree, score, contiguity)


def prefix_jobs(sequences):
    nodes = [dict(parent=None, children={}, p=0, terminal=None)]
    for source, tokens in enumerate(sequences):
        if not tokens:
            raise ValueError("empty prompt")
        parent = 0
        reusable = (len(tokens) - 1) // 16
        for offset in range(0, reusable * 16, 16):
            key = tuple(tokens[offset:offset + 16])
            if key not in nodes[parent]["children"]:
                nodes[parent]["children"][key] = len(nodes)
                nodes.append(dict(parent=parent, children={}, p=16, terminal=None))
            parent = nodes[parent]["children"][key]
        nodes[parent]["children"][("tail", source)] = len(nodes)
        nodes.append(dict(parent=parent, children={},
                          p=len(tokens) - 16 * reusable, terminal=source))
    return nodes


def horn_jobs(nodes):
    """Return job order and exact density metadata for a weighted leaf outtree."""
    minimum, densities, first = {}, {}, {}
    for index in reversed(range(len(nodes))):
        node = nodes[index]
        if node["terminal"] is not None:
            costs = {1: node["p"]}
            first[index] = node["terminal"]
        else:
            costs = {0: node["p"]}
            first[index] = min(first[child] for child in node["children"].values())
            for child in node["children"].values():
                combined = dict(costs)  # Exclude this entire child subtree.
                for left_count, left_cost in costs.items():
                    for right_count, right_cost in minimum[child].items():
                        if right_count == 0:
                            continue  # A weight-zero branch only adds cost.
                        count, cost = left_count + right_count, left_cost + right_cost
                        if count not in combined or cost < combined[count]:
                            combined[count] = cost
                costs = combined
        minimum[index] = costs
        densities[index] = max(Fraction(k, cost) for k, cost in costs.items() if k)

    ready, order, done = [(-densities[0], first[0], 0)], [], set()
    while ready:
        _, _, index = heapq.heappop(ready)
        parent = nodes[index]["parent"]
        if parent is not None and parent not in done:
            raise AssertionError("precedence violated")
        done.add(index)
        order.append(index)
        for child in nodes[index]["children"].values():
            heapq.heappush(ready, (-densities[child], first[child], child))
    if len(order) != len(nodes):
        raise AssertionError("incomplete job order")
    elapsed, weighted = 0, 0
    for index in order:
        elapsed += nodes[index]["p"]
        if nodes[index]["terminal"] is not None:
            weighted += elapsed
    return order, dict(total_job_work=elapsed, sum_terminal_completion_work=weighted,
                       job_count=len(nodes), density_rule="max k/minimum rooted-subtree work for k terminal jobs")


def horn_order(sequences):
    nodes = prefix_jobs(sequences)
    jobs, receipt = horn_jobs(nodes)
    order = [nodes[index]["terminal"] for index in jobs
             if nodes[index]["terminal"] is not None]
    if sorted(order) != list(range(len(sequences))):
        raise AssertionError("invalid terminal permutation")
    return order, receipt


def small_exact_checks():
    """Exhaustive request permutations on fixed small trees, no sampled outcomes."""
    # Same runtime model: shared blocks and positive private tails. Distinct
    # branches, nested branching, duplicates, short and full-block tails.
    a, b, c, d = ([i] * 16 for i in range(1, 5))
    cases = [
        [[8], [9, 9], a + [7]],
        [a + [7], a + b + c + [8], b + [9]],
        [a + b + [7], a + b + [7], a + c + [9], d + [8]],
        [a + b + [7], a + b + c + [8], a + c + [9],
         d + [8], d + a + [9], [6]],
        [a + b, a + b + c, a + b + d, b + a, b + c, b + d],
    ]
    checks = []
    for case, sequences in enumerate(cases):
        paths, tails, parents = make_tree(sequences)
        order, native = horn_order(sequences)
        value = score(order, paths, tails, parents)["sum_prompt_completion_work"]
        exact = min(score(list(p), paths, tails, parents)["sum_prompt_completion_work"]
                    for p in itertools.permutations(range(len(sequences))))
        if value != exact or value != native["sum_terminal_completion_work"]:
            raise AssertionError((case, value, exact, native))
        checks.append(dict(case=case, requests=len(sequences), objective=value,
                           exhaustive_optimum=exact, terminal_projection_matches=True))
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = (args.input_dir / "workload.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != WORKLOAD_SHA:
        raise ValueError("frozen full150 input differs")
    rows = json.loads(raw)["requests"]
    sequences = [row["prompt_token_ids"] for row in rows]
    checks = small_exact_checks()
    started = time.perf_counter()
    order, jobs = horn_order(sequences)
    elapsed = time.perf_counter() - started
    paths, tails, parents = make_tree(sequences)
    whole = density_order(sequences)
    base, result = (score(candidate, paths, tails, parents) for candidate in (whole, order))
    if (base["total_unique_prompt_work"] != result["total_unique_prompt_work"]
            or result["sum_prompt_completion_work"] != jobs["sum_terminal_completion_work"]
            or result["sum_prompt_completion_work"] > base["sum_prompt_completion_work"]):
        raise AssertionError("job-to-request projection or objective differs")
    report = dict(schema="c-longqa-horn-order-v1", workload_sha256=WORKLOAD_SHA,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        reference="https://arxiv.org/html/2404.17544#S4.SS1",
        source_indices_in_submission_order=order, job_schedule=jobs,
        local_reorder_s=elapsed, whole=base, horn=result,
        relative_mean_work_reduction=(1-result["mean_prompt_completion_work"]/base["mean_prompt_completion_work"]),
        changed_positions_vs_whole=sum(a != b for a, b in zip(order, whole)),
        contiguity=contiguity(order, paths, sequences), small_exhaustive_checks=checks,
        scope="Classical serial infinite-cache prompt-only comparison. No GPU execution, EOS, quality, finite-cache or new-method claim.")
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({k: report[k] for k in ("whole", "horn", "relative_mean_work_reduction",
                                           "changed_positions_vs_whole", "local_reorder_s", "contiguity")}))


if __name__ == "__main__":
    main()
