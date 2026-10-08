"""Current-layer, identity-free upper bound; never a future-trajectory speedup."""
import argparse
from collections import defaultdict
from fractions import Fraction
from itertools import combinations
import json
from pathlib import Path


def bounds(rows, missing, budget):
    refs = {e: {i for i, row in enumerate(rows) if e in row} for e in missing}
    eligible = {e: indices for e, indices in refs.items() if len(indices) <= budget}
    charge = [Fraction(0) for _ in rows]
    for indices in eligible.values():
        assert indices
        for i in indices:
            charge[i] += Fraction(1, len(indices))
    upper = min(len(eligible), int(sum(sorted(charge, reverse=True)[:budget])))
    # This arbitrary-row witness is feasible only in the relaxation, not in serving.
    deleted = set(sorted(range(len(rows)), key=lambda i: (-charge[i], i))[:budget])
    witness = sum(indices <= deleted for indices in refs.values())
    assert 0 <= witness <= upper <= len(missing)
    return upper, witness, len(eligible)


def sanity():
    # Exhaustive deletions check the mathematical bound and shared-reference cases.
    fixtures = [([{0}, {0}], {0}),
                ([{0, 1}, {0}, {1, 2}, {2}], {0, 1, 2}),
                ([{0, 1}, {0, 2}, {1, 2}, {3}, {3}, {4}], {0, 1, 2, 3, 4})]
    for rows, missing in fixtures:
        for d in range(len(rows)+1):
            upper, witness, _ = bounds(rows, missing, d)
            exact = max(sum(all(e not in rows[i] for i in range(len(rows)) if i not in deleted)
                            for e in missing)
                        for deleted in map(set, combinations(range(len(rows)), d)))
            assert witness <= exact <= upper


def analyze(directory):
    raw = json.loads((directory/'raw.json').read_text())
    assert raw['status'] == 'COMPLETE'
    steps = {s['step']: s for s in raw['scheduler_steps']}
    by_kind, records, seen = defaultdict(lambda: defaultdict(int)), [], set()
    with (directory/'pager/calls.jsonl').open() as f:
        for line in f:
            c = json.loads(line)
            if not c['measurement']:
                continue
            s = steps[c['context']['step_id']]
            counts = {r['internal_request_id']: r['scheduled_tokens'] for r in s['scheduled']}
            assert c['status'] == 'complete' and c['context']['scheduled_tokens'] == counts
            rows = [set(row) for row in c['row_topk_experts']]
            assert len(rows) == c['rows'] == c['context']['expected_rows'] == sum(counts.values())
            assert c['context']['row_request_order_verified'] is False
            d = sum(s['scheduled_draft_tokens'].values())
            assert 0 <= d < len(rows)
            missing = set(c['missing_experts'])
            assert missing == set.union(*rows) - set(c['entry_resident_experts'])
            assert sorted(c['loaded_experts']) == sorted(missing)
            size = c['weight_copy_bytes']//len(missing) if missing else 0
            assert size*len(missing) == c['weight_copy_bytes']
            upper, witness, eligible = bounds(rows, missing, d)
            prefill = sum(r['prefill_tokens'] for r in s['scheduled'])
            kind = 'pure_decode' if not prefill else 'pure_prefill' if prefill == len(rows) else 'mixed'
            item = dict(step=s['step'], layer=c['layer_name'], rows=len(rows), draft_budget=d,
                missing_experts=len(missing), eligible_experts=eligible, upper_experts=upper,
                relaxed_witness_experts=witness, original_bytes=c['weight_copy_bytes'],
                upper_bytes=upper*size, relaxed_witness_bytes=witness*size)
            records.append(item); seen.add(s['step'])
            totals = by_kind[kind]
            for k in ('original_bytes', 'upper_bytes', 'relaxed_witness_bytes'):
                totals[k] += item[k]
            totals['calls'] += 1
            totals['draft_calls'] += d > 0
            totals['positive_bound_calls'] += upper > 0
            totals['draft_zero_bound_calls'] += d > 0 and upper == 0
    assert seen == set(steps)
    return dict(by_kind=dict(by_kind), calls=records)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--results', type=Path, required=True)
    args = p.parse_args(); sanity()
    cells = {name: analyze(args.results/name) for name in ('01_ngram4', '02_ngram4')}
    report = dict(status='FROZEN_STATE_ROW_RELAXATION_ONLY', exact_small_case_checks=True,
        cells=cells, repeated_structure_identical=cells['01_ngram4'] == cells['02_ngram4'],
        scope=['Each current layer independently allows deletion of any d rows, including mandatory rows.',
               'd equals actual scheduled draft count; no request-row identity is assumed.',
               'The charge bound is an upper bound on saved entry-miss experts under that relaxation.',
               'The arbitrary-row witness is not a legal request-prefix policy or its lower bound.',
               'Sum of local bounds is descriptive on the frozen trace; not a policy-trajectory bound.',
               'No bound on accepted-token loss, future cache/KV/routes, GPU time, or full-request gain.'])
    with (args.results/'draft_row_relaxation.json').open('x') as f:
        json.dump(report, f, indent=2); f.write('\n')
    print(json.dumps(dict(status=report['status'], identical=report['repeated_structure_identical'],
        by_kind=cells['01_ngram4']['by_kind']), indent=2))


if __name__ == '__main__':
    main()
