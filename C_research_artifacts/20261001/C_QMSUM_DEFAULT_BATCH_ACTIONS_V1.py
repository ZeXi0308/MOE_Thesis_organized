#!/usr/bin/env python3
"""Native budget work and prefix-availability witnesses for the fixed8192/512 pair."""
import argparse
import json
from pathlib import Path
from C_QMSUM_CHUNK_ACTIONS_V1 import cell
from C_QMSUM_ANALYZE_V1 import load, sha


def lcp(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b))


def describe(directory, budget):
    measured = cell(directory, budget)
    mix = load(directory / 'measured-service-mix.json')
    shape = load(directory / 'shape-warmup-source.json')['batch_shape']
    if (shape['prompt_tokens_total'] != 8190
            or sum(shape['scheduled_tokens_per_call']) + shape['first_allocation_cached_tokens'] != 8190
            or shape['reset_status'] != 'QUALIFIED'):
        raise ValueError('common batch warmup differs')
    first = mix['first_successful_allocation']
    actual_order = [measured['outputs'][r['external_request_id']]['source_index'] for r in first]
    if actual_order != measured['frozen_order']:
        raise ValueError('first allocation differs from frozen density order')
    # Completed-by-earlier-call witnesses; no current cache-residency or eviction oracle.
    previous_progress = {}
    first_by_call = {}
    for i, row in enumerate(first):
        first_by_call.setdefault(row['schedule_call'], []).append((i, row))
    losses = []
    expected_hits = observed_hits = 0
    for call in mix['scheduler_calls']:
        for i, row in first_by_call.get(call['call'], []):
            rid = row['external_request_id']
            prompt = measured['outputs'][rid]['prompt_token_ids']
            predecessor = first[i-1] if i else None
            previous_id = predecessor['external_request_id'] if predecessor else None
            expected = (min(lcp(measured['outputs'][previous_id]['prompt_token_ids'], prompt),
                            len(prompt)-1) // 16 * 16 if predecessor else 0)
            observed = row['new_prefix_cached_tokens']
            expected_hits += expected
            observed_hits += observed
            if expected > observed:
                progress = previous_progress.get(previous_id, 0)
                classification = ('predecessor_first_allocated_in_same_call'
                    if predecessor['schedule_call'] == call['call'] else
                    'shared_prefix_completed_in_an_earlier_call' if progress >= expected else
                    'shared_prefix_not_completed_by_preceding_calls')
                losses.append(dict(source_index=measured['outputs'][rid]['source_index'],
                    predecessor_source_index=measured['outputs'][previous_id]['source_index'],
                    call=call['call'], predecessor_first_call=predecessor['schedule_call'],
                    expected_adjacent_reusable_tokens=expected,
                    observed_cached_tokens=observed, shortfall_tokens=expected-observed,
                    predecessor_max_computed_by_prior_calls=progress, classification=classification))
        for served in call['requests']:
            rid = served['external_request_id']
            previous_progress[rid] = max(previous_progress.get(rid, 0),
                served['computed_before_execution'] + served['scheduled_tokens'])
    totals = {}
    for witness in losses:
        kind = witness['classification']
        target = totals.setdefault(kind, dict(requests=0, shortfall_tokens=0))
        target['requests'] += 1
        target['shortfall_tokens'] += witness['shortfall_tokens']
    return dict(metrics=measured['metrics'], raw_sha256=measured['raw_sha256'],
        gpu=measured['environment']['gpu_before']['gpu'].split(',')[0],
        first_source_order=actual_order, batch_warmup=shape,
        adjacent_prefix_reference=dict(expected_hits=expected_hits, observed_hits=observed_hits,
            net_hit_shortfall_tokens=expected_hits-observed_hits,
            gross_shortfall_tokens=sum(w['shortfall_tokens'] for w in losses),
            loss_classification=totals, witnesses=losses))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    arms = {str(b): describe(args.root / str(b) / 'native', b) for b in (8192, 512)}
    if (arms['8192']['gpu'] != arms['512']['gpu']
            or arms['8192']['first_source_order'] != arms['512']['first_source_order']):
        raise ValueError('same-host/order pair differs')
    result = dict(schema='c-qmsum-offline-native-batch-actions-v1', arms=arms,
        scope='One viewed full200-per-cell same-host8192 then512 pair; fixed128seq/4096KV; official offline budget value only, not full defaults',
        limits=['Cache shortfalls are relative to the immediately preceding prompt in the frozen density order, not a general GPU performance bound.',
                'Prior execution proves a prefix was computed, not that it remained resident; no exact eviction event is inferred.',
                'Same-call first allocations cannot benefit from a preceding request whose first prefill only executes in that call; this is an availability witness, not a tested alternative action.',
                'Outputs and batch shapes may change; all full-request outcomes are evaluated separately, without equal-work or new-method claims.'])
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({b: v['adjacent_prefix_reference']['loss_classification'] for b, v in arms.items()}))


if __name__ == '__main__':
    main()
