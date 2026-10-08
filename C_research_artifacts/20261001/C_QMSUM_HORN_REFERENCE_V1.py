#!/usr/bin/env python3
"""Apply the existing classical prompt-only tree reference to frozen QMSum200."""
import argparse
import hashlib
import json
from pathlib import Path
import time

from C_LONG_DOCUMENT_QA_HORN_ORDER_V1 import horn_order, density_order
from C_LONG_DOCUMENT_QA_TREE_ORDER_WITNESS_V1 import make_tree, score, contiguity

SHA = '4408737dbc897eedbd8481de1995c69ceb2d4693256d7030034e502ea0c08004'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    raw = (args.input_dir / 'workload.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != SHA:
        raise ValueError('frozen QMSum200 input differs')
    rows = json.loads(raw)['requests']
    if len(rows) != 200 or [r['source_index'] for r in rows] != list(range(200)):
        raise ValueError('full200 inventory differs')
    sequences = [r['prompt_token_ids'] for r in rows]
    started = time.perf_counter()
    order, jobs = horn_order(sequences)
    elapsed = time.perf_counter() - started
    paths, tails, parents = make_tree(sequences)
    whole = density_order(sequences)
    base, result = [score(candidate, paths, tails, parents) for candidate in (whole, order)]
    if (base['total_unique_prompt_work'] != result['total_unique_prompt_work']
            or result['sum_prompt_completion_work'] != jobs['sum_terminal_completion_work']
            or result['sum_prompt_completion_work'] > base['sum_prompt_completion_work']):
        raise AssertionError('projection, total work or classical comparison differs')
    report = dict(schema='c-qmsum-horn-reference-v1', workload_sha256=SHA,
        source_sha256={n: hashlib.sha256(Path(n).read_bytes()).hexdigest() for n in
            ('C_QMSUM_HORN_REFERENCE_V1.py', 'C_LONG_DOCUMENT_QA_HORN_ORDER_V1.py',
             'C_LONG_DOCUMENT_QA_TREE_ORDER_WITNESS_V1.py', 'C_LONG_DOCUMENT_QA_DENSITY_ORDER_V1.py')},
        request_count=200, source_indices_in_submission_order=order, job_schedule=jobs,
        local_reorder_s=elapsed, whole=base, horn=result,
        relative_mean_work_reduction=1-result['mean_prompt_completion_work']/base['mean_prompt_completion_work'],
        changed_positions_vs_whole=sum(a != b for a,b in zip(order, whole)),
        contiguity=contiguity(order, paths, sequences),
        scope='Existing classical Horn reference on viewed QMSum. Serial, infinite-cache, prompt-only model; no GPU-time, decode, finite-cache optimum bound or new-method claim.')
    with args.output.open('x') as f:
        json.dump(report, f, indent=2, sort_keys=True)
        f.write('\n')
    print(json.dumps({k:report[k] for k in ('whole','horn','relative_mean_work_reduction','changed_positions_vs_whole','local_reorder_s')}))


if __name__ == '__main__':
    main()
