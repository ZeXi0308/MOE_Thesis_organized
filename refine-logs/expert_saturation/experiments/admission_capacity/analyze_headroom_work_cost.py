#!/usr/bin/env python3
"""Partition measured engine remainder by disjoint work classes and decode width."""
import argparse
import json
import math
from pathlib import Path
from analyze_completion_headroom import FOUR_ARMS, FOUR_LABELS, LABELS, read, require

CLASSES = ('new_prefill', 'recompute', 'pure_new_decode', 'empty')
FIELDS = ('calls', 'engine_non_schedule_s', 'scheduled_tokens', 'new_prefill_tokens', 'recomputed_tokens', 'new_decode_tokens')


def blank():
    return dict.fromkeys(FIELDS, 0)


def inspect(directory, label):
    raw = read(directory/'raw.json')
    calls, steps = raw['engine_steps'], raw['scheduler_steps']
    require(raw['status'] == read(directory/'status.json')['status'] == 'COMPLETE', 'incomplete cell')
    require(len(calls) == len(steps), 'requires one scheduler step per engine call')
    classes, widths, previous = {k: blank() for k in CLASSES}, {}, min(r['arrival_s'] for r in raw['requests'])
    for index, (call, step) in enumerate(zip(calls, steps)):
        require(call['call_index'] == call['scheduler_step_start'] == step['step'] == index and call['scheduler_step_end'] == index+1, 'call/step index mismatch')
        times = [previous, call['start_s'], step['start_s'], step['end_s'], call['returned_s'], raw['observation_end_s']]
        require(call['completed'] and all(math.isfinite(t) for t in times) and all(a <= b for a, b in zip(times, times[1:])), 'noncompleted or overlapping/nonnested interval')
        prefill, recompute, decode = [sum(r[k] for r in step['scheduled']) for k in ('prefill_tokens', 'recompute_tokens', 'decode_tokens')]
        require(min(prefill, recompute, decode) >= 0 and prefill+recompute+decode == step['total_scheduled_tokens'], 'work conservation failed')
        require(not (prefill and recompute), 'prefill/recompute mixed call needs an explicit additional class')
        kind = 'new_prefill' if prefill else 'recompute' if recompute else 'pure_new_decode' if decode else 'empty'
        remainder = call['returned_s']-call['start_s']-(step['end_s']-step['start_s'])
        values = dict(calls=1, engine_non_schedule_s=remainder, scheduled_tokens=step['total_scheduled_tokens'],
                      new_prefill_tokens=prefill, recomputed_tokens=recompute, new_decode_tokens=decode)
        buckets = [classes[kind]]
        if kind == 'pure_new_decode':
            require(all(r['scheduled_tokens'] == 1 for r in step['scheduled']), 'decode width is not one token per request')
            buckets.append(widths.setdefault(str(len(step['scheduled'])), blank()))
        for bucket in buckets:
            for key, value in values.items():
                bucket[key] += value
        previous = call['returned_s']
    totals = {k: sum(b[k] for b in classes.values()) for k in FIELDS}
    require(totals['calls'] == len(calls) and all(math.isclose(sum(b[k] for b in widths.values()), classes['pure_new_decode'][k], rel_tol=0, abs_tol=1e-12) for k in FIELDS), 'class/width partition failed')
    return dict(label=label, totals=totals, classes=classes, pure_decode_by_scheduled_width=dict(sorted(widths.items(), key=lambda x: int(x[0]))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--four-arm', action='store_true', help='native / safe29 / headroom / rotate, two repeats')
    args = parser.parse_args()
    require(not args.output.exists(), 'output must not exist; refusing overwrite')
    labels = FOUR_LABELS if args.four_arm else LABELS
    arms = FOUR_ARMS if args.four_arm else ('native', 'headroom')
    rows = [inspect(args.run_dir/'gpu_results'/label, label) for label in labels]
    indexed, pairs = {r['label']: r for r in rows}, []
    pair_labels = [(i, a, b) for i in (0, 1) for n, a in enumerate(arms) for b in arms[n+1:]]
    for repeat, baseline, action in pair_labels:
        a, b = [indexed[f'repeat{repeat}-{p}'] for p in (baseline, action)]
        difference = lambda x, y: {k: y[k]-x[k] for k in FIELDS}
        classes = {k: difference(a['classes'][k], b['classes'][k]) for k in CLASSES}
        width_keys = sorted(set(a['pure_decode_by_scheduled_width']) | set(b['pure_decode_by_scheduled_width']), key=int)
        widths = {k: difference(a['pure_decode_by_scheduled_width'].get(k, blank()), b['pure_decode_by_scheduled_width'].get(k, blank())) for k in width_keys}
        totals = difference(a['totals'], b['totals'])
        require(all(math.isclose(totals[k], sum(c[k] for c in classes.values()), rel_tol=0, abs_tol=1e-12) for k in FIELDS), 'paired class differences do not close')
        pair = dict(repeat=repeat, baseline=a['label'], action=b['label'], classes=classes, pure_decode_by_scheduled_width=widths)
        pair['action_minus_baseline' if args.four_arm else 'action_minus_native'] = totals
        pairs.append(pair)
    result = dict(status='MEASUREMENT_ONLY', rows=rows, comparisons=pairs,
        classification='Disjoint successful engine calls: contains new prefill; contains recomputation; exclusively new decode; or empty. Prefill and recompute classes may also contain new decode; their entire durations are not isolated prefill/recompute costs. Simultaneous new prefill and recompute is rejected.',
        scope='Host engine-call wall time minus its nested scheduler interval. Includes model execution, sampling, synchronization and host overhead; not pure GPU time. Width buckets describe realized policy-specific trajectories and are not matched-state causal comparisons or executable counterfactual savings.')
    if args.four_arm:
        result['pairing_rule'] = 'All eight COMPLETE cells; six descriptive within-repeat pairs. Same-policy repeats are not statistical noise bounds.'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
