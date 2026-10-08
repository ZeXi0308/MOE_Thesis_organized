#!/usr/bin/env python3
"""Partition retained headroom campaign host wall intervals without overlap."""
import argparse
import json
import math
from pathlib import Path

from analyze_completion_headroom import FOUR_ARMS, FOUR_LABELS, LABELS, read, require


def inspect(directory, label, cutoff=None):
    raw, decisions = read(directory/'raw.json'), read(directory/'headroom-decisions.json')
    require(raw['status'] == read(directory/'status.json')['status'] == 'COMPLETE', 'cell incomplete')
    calls, steps = raw['engine_steps'], raw['scheduler_steps']
    require(len(calls) == len(steps) == len(decisions) and (cutoff is None or len(steps) >= cutoff), 'requires one scheduler step per engine call')
    policy = label.split('-')[1]
    mode = 'native' if policy == 'safe29' else policy
    origin, end = min(r['arrival_s'] for r in raw['requests']), raw['observation_end_s']
    previous, schedules, remainders, costs = origin, [], [], []
    for index, (call, step, decision) in enumerate(zip(calls, steps, decisions)):
        require(call['call_index'] == call['scheduler_step_start'] == step['step'] == decision['step'] == index
                and call['scheduler_step_end'] == index+1, 'call/step/decision indices differ')
        times = [previous, call['start_s'], step['start_s'], step['end_s'], call['returned_s'], end]
        require(all(math.isfinite(t) for t in times) and all(a <= b for a, b in zip(times, times[1:])), 'intervals overlap or are not nested')
        require(call['completed'] and decision['status'] == 'APPLIED' and decision['mode'] == mode, 'call or action not completed')
        schedule = step['end_s']-step['start_s']
        cost = decision['decision_seconds']
        require(math.isfinite(cost) and 0 <= cost <= schedule, 'decision duration outside scheduler duration')
        schedules.append(schedule)
        remainders.append(call['returned_s']-call['start_s']-schedule)
        costs.append(cost)
        previous = call['returned_s']
    wall, scheduler, remaining = end-origin, sum(schedules), sum(remainders)
    outside = wall-sum(c['returned_s']-c['start_s'] for c in calls)
    require(outside >= 0 and math.isclose(wall, scheduler+remaining+outside, rel_tol=0, abs_tol=1e-12), 'host wall conservation failed')
    row = dict(label=label, wall_s=wall, scheduler_inclusive_s=scheduler, engine_non_schedule_s=remaining,
               outside_engine_calls_s=outside, decision_subset_s=sum(costs), engine_calls=len(calls),
               scheduled_tokens=sum(s['total_scheduled_tokens'] for s in steps))
    if cutoff is not None:
        row.update({f'prefix_through{cutoff-1}_scheduler_s': sum(schedules[:cutoff]),
                    f'prefix_through{cutoff-1}_engine_non_schedule_s': sum(remainders[:cutoff])})
    return row

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--four-arm', action='store_true', help='native / safe29 / headroom / rotate, two repeats')
    args = parser.parse_args()
    require(not args.output.exists(), 'output must not exist; refusing overwrite')
    root = args.run_dir/'gpu_results'
    labels = FOUR_LABELS if args.four_arm else LABELS
    cutoff = None
    if not args.four_arm:
        cuts = [next((d['step'] for d in read(root/label/'headroom-decisions.json') if d['held']), None)
                for label in LABELS if label.endswith('-headroom')]
        require(None not in cuts and len(set(cuts)) == 1, 'intervention prefixes differ or no held action')
        cutoff = cuts[0]
    rows = [inspect(root/label, label, cutoff) for label in labels]
    result = dict(status='MEASUREMENT_ONLY', rows=rows,
        identity='wall = scheduler_inclusive + engine_non_schedule + outside_engine_calls; decision_subset is included within scheduler_inclusive.',
        scope='Host wall intervals only. engine_non_schedule includes model execution, host execution overhead, sampling and synchronization; not pure GPU time. Removing a measured interval is not an executed counterfactual. '+('Action onset differs across four policies, so no common-prefix comparison is made.' if args.four_arm else 'Prefix means steps before first held action, not identical timing or equal instrumentation.'))
    if args.four_arm:
        indexed, pairs = {r['label']: r for r in rows}, []
        for repeat in (0, 1):
            for i, baseline in enumerate(FOUR_ARMS):
                for action in FOUR_ARMS[i+1:]:
                    a, b = [indexed[f'repeat{repeat}-{p}'] for p in (baseline, action)]
                    delta = {k: b[k]-a[k] for k in a if k != 'label'}
                    require(math.isclose(delta['wall_s'], sum(delta[k] for k in ('scheduler_inclusive_s', 'engine_non_schedule_s', 'outside_engine_calls_s')), rel_tol=0, abs_tol=1e-12), 'paired host wall conservation failed')
                    pairs.append(dict(repeat=repeat, baseline=a['label'], action=b['label'], action_minus_baseline=delta))
        result.update(comparisons=pairs, pairing_rule='All eight COMPLETE cells; six descriptive within-repeat pairs. Same-policy repeats are not statistical noise bounds.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
