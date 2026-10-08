#!/usr/bin/env python3
"""Conditional page-demand envelopes on observed native suffixes; stdout only.

No future EOS/output trace is read. k is a fixed structural scale, not a tuned
horizon. Request admissions, scheduling, victim restoration and execution time
are not predicted. Pair counts are descriptive, not independent repetitions.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
KS = (1, 4, 8, 16, 32, 64)


def row_state(row, requests, block_size):
    request = requests[row['request']]
    p = row.get('prompt_tokens', request['prompt_tokens'])
    values = dict(prompt=p, computed=row.get('computed_tokens'), output=row.get('output_tokens'),
                  held=row.get('held_blocks'), release=row.get('immediate_releasable_blocks'),
                  shared=row.get('shared_blocks'), maximum=row.get('max_tokens'))
    if any(type(v) is not int or v < 0 for v in values.values()):
        return None
    if row.get('release_state_error') is not None or values['release'] + values['shared'] != values['held']:
        return None
    values.update(request=request['request_id'], index=row['index'], qualified=row['qualified'])
    values['history_end'] = p + values['output'] - 1
    values['pure'] = values['output'] > 0 and values['computed'] == values['history_end']
    values['partial_history'] = values['output'] > 0 and values['computed'] < values['history_end']
    values['surplus'] = values['held'] - (values['computed'] + block_size - 1) // block_size
    return values


def demand(row, k, block_size, max_model_len):
    # For a partial row this covers its outstanding history before k new outputs;
    # it does NOT mean k native scheduling iterations or k additional input tokens.
    if not (row['pure'] or row['partial_history']):
        return None
    if row['output'] + k > row['maximum'] or row['history_end'] + k > max_model_len:
        return None
    return max(0, (row['history_end'] + k + block_size - 1) // block_size - row['held'])


def analyze(archive):
    store = json.loads((archive / 'selective-store.json').read_text())
    raw = json.loads((archive / 'raw.json').read_text())
    profile = json.loads((archive / 'normal_capacity_profile.json').read_text())
    block_size = profile['block_size_tokens']; max_model_len = profile['max_model_len']
    if type(block_size) is not int or block_size <= 0:
        raise ValueError('Unknown block size')
    requests = {r['internal_request_id']: r for r in raw['requests']}
    stats = Counter(); surplus = Counter(); groups = {}; partials = []
    for ordinal, decision in enumerate(store['victim_decisions']):
        stats['decisions'] += 1
        if decision.get('active_protection_or_phase') is not False:
            stats['active_or_unknown_phase_decisions'] += 1
            continue
        rows = [row_state(row, requests, block_size) for row in decision['candidates']]
        for row in rows:
            if row is None: stats['unknown_rows'] += 1; continue
            stats['pure_rows' if row['pure'] else 'other_rows'] += 1
            if row['pure']: surplus[row['surplus']] += 1
        if any(row is None for row in rows):
            stats['unknown_physical_or_progress_decisions'] += 1
            continue
        tail = rows[-1]; others = rows[:-1]
        if tail['partial_history']:
            partials.append(dict(step=decision['step'], tail=tail, suffix=len(rows),
                next_output_history_pages=demand(tail, 1, block_size, max_model_len)))
        if not others: continue
        min_distance = min(abs(row['release'] - tail['release']) for row in others)
        adequate = [row for row in others if row['release'] >= tail['release']]
        min_adequate = min((row['release'] for row in adequate), default=None)
        sets = dict(equal_release=[row for row in others if row['release'] == tail['release']],
            nearest_release=[row for row in others if abs(row['release'] - tail['release']) == min_distance],
            nearest_sufficient_release=[row for row in adequate if row['release'] == min_adequate],
            equal_held=[row for row in others if row['held'] == tail['held']])
        for category, alternatives in sets.items():
            for alt in alternatives:
                layer = ('pure_shared_involved' if tail['shared'] or alt['shared'] else 'pure_private') if tail['pure'] and alt['pure'] else 'partial_or_other'
                key = category + '/' + layer
                group = groups.setdefault(key, dict(pairs=0, decisions=set(), release_delta=Counter(),
                    held_delta=Counter(), demand_delta={k: Counter() for k in KS}, unknown_k=Counter(),
                    declared_cap_excluded_k=Counter(), context_limit_excluded_k=Counter(),
                    largest_abs_difference=-1, example=None))
                group['pairs'] += 1; group['decisions'].add(ordinal)
                group['release_delta'][alt['release']-tail['release']] += 1
                group['held_delta'][alt['held']-tail['held']] += 1
                deltas = {}
                for k in KS:
                    if any(row['output'] + k > row['maximum'] for row in (tail, alt)):
                        group['declared_cap_excluded_k'][k] += 1
                        continue
                    if any(row['history_end'] + k > max_model_len for row in (tail, alt)):
                        group['context_limit_excluded_k'][k] += 1
                        continue
                    gt = demand(tail, k, block_size, max_model_len)
                    ga = demand(alt, k, block_size, max_model_len)
                    if gt is None or ga is None:
                        group['unknown_k'][k] += 1; continue
                    # Choosing alt retains tail instead: common other requests cancel.
                    deltas[k] = gt - ga
                    group['demand_delta'][k][gt-ga] += 1
                magnitude = max((abs(v) for v in deltas.values()), default=-1)
                if magnitude > group['largest_abs_difference']:
                    group['largest_abs_difference'] = magnitude
                    group['example'] = dict(step=decision['step'], tail=tail, alternative=alt,
                        remaining_set_demand_delta=deltas,
                        immediate_free_delta=alt['release']-tail['release'])
    for group in groups.values(): group['decisions'] = len(group['decisions'])
    return dict(archive=str(archive), block_size=block_size, stats=dict(stats),
        pure_allocated_surplus=dict(surplus), groups=groups, partial_tail_observations=partials,
        semantics='Positive demand delta means more conditional pages remain after choosing the alternative. '
                  'Equal-held/equal-release are separate strata. Shared observations are slot demand; '
                  'The running fastpath adds slots without new prefix lookup; admission/prefix-hit COW is excluded. '
                  'Partial rows cover backlog before k new outputs. '
                  'No full-running total, wall-time forward progress, future admission or restored victim demand is inferred.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, action='append')
    args = parser.parse_args()
    archives = args.archive
    if not archives:
        latest = HERE / 'session-native-partial-restore-once-westd53005-20261008-r01'
        archives = [cell / 'archive' for cell in sorted(latest.glob('cell-*'))
                    if (cell / 'archive/selective-store.json').exists()]
        archives.append(HERE / 'session-native-shared-prefix-probe-westd53005-20261008-r01/cell-01-shared-prefix-tail/archive')
    for archive in archives:
        print(json.dumps(analyze(archive), sort_keys=True))


if __name__ == '__main__':
    main()
