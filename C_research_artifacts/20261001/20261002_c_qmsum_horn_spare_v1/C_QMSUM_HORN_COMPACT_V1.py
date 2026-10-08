#!/usr/bin/env python3
"""Equivalent classical Horn implementation with positive-cost unary chains contracted."""
import argparse
import hashlib
import json
from pathlib import Path
import time

from C_LONG_DOCUMENT_QA_HORN_ORDER_V1 import prefix_jobs, horn_jobs, horn_order


def compact_jobs(nodes):
    """A selected zero-weight unary chain remains highest-density until its branch.

    Removing each positive-cost ancestor increases the next node's density,
    while all other ready-node densities remain fixed. It cannot profitably be
    interrupted within that chain under the classical rule. Preserve terminal
    weights and total path cost; branch and terminal decisions remain explicit.
    """
    result = []

    def visit(start, parent):
        current, cost = start, nodes[start]['p']
        while nodes[current]['terminal'] is None and len(nodes[current]['children']) == 1:
            current = next(iter(nodes[current]['children'].values()))
            cost += nodes[current]['p']
        index = len(result)
        result.append(dict(parent=parent, children={}, p=cost,
                           terminal=nodes[current]['terminal']))
        for key, child in nodes[current]['children'].items():
            result[index]['children'][key] = visit(child, index)
        return index

    visit(0, None)
    return result


def compact_horn_order(sequences):
    nodes = compact_jobs(prefix_jobs(sequences))
    jobs, _ = horn_jobs(nodes)
    return [nodes[i]['terminal'] for i in jobs if nodes[i]['terminal'] is not None]


def measure(sequences, expected):
    started = time.perf_counter()
    native = prefix_jobs(sequences)
    compact = compact_jobs(native)
    jobs, receipt = horn_jobs(compact)
    order = [compact[i]['terminal'] for i in jobs if compact[i]['terminal'] is not None]
    elapsed = time.perf_counter() - started
    if order != expected or sorted(order) != list(range(len(sequences))):
        raise AssertionError('compressed classical terminal permutation differs')
    if sum(n['p'] for n in native) != sum(n['p'] for n in compact):
        raise AssertionError('contraction changed work')
    return dict(request_count=len(sequences), original_jobs=len(native),
        compact_jobs=len(compact), measured_local_reorder_s=elapsed,
        terminal_order_identical=True, source_indices_in_submission_order=order,
        compact_job_schedule=receipt)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-dir', type=Path, required=True)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    raw = (args.input_dir / 'workload.json').read_bytes()
    reference = json.loads(args.reference.read_text())
    if hashlib.sha256(raw).hexdigest() != reference['workload_sha256']:
        raise ValueError('input and existing classical reference differ')
    rows = json.loads(raw)['requests']
    sequences = [r['prompt_token_ids'] for r in rows]
    # Distinct branches, nested branches, full-block tails and duplicate prompts.
    a, b, c, d = ([i] * 16 for i in range(1, 5))
    cases = [[a+[7], a+b+c+[8], b+[9]],
             [a+b+[7], a+b+[7], a+c+[9], d+[8]],
             [a+b, a+b+c, a+b+d, b+a, b+c, b+d],
             [[8], [9,9], a+[7]],
             [a+b+[7], a+b+c+[8], a+c+[9], d+[8], d+a+[9], [6]]]
    for seq in cases:
        if compact_horn_order(seq) != horn_order(seq)[0]:
            raise AssertionError('fixed small-tree contraction differs')
    result = measure(sequences, reference['source_indices_in_submission_order'])
    if result['compact_job_schedule']['sum_terminal_completion_work'] != reference['horn']['sum_prompt_completion_work']:
        raise AssertionError('compressed objective differs from existing reference')
    result.update(schema='c-classical-horn-unary-contraction-v1',
        workload_sha256=reference['workload_sha256'],
        reference_sha256=hashlib.sha256(args.reference.read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        small_tree_equivalence_checks=len(cases),
        scope='Classical reference implementation only; same exact terminal order and serial infinite-cache prompt objective. Local CPU timing is not remote GPU or online service evidence.')
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='source_indices_in_submission_order'}))


if __name__ == '__main__':
    main()
