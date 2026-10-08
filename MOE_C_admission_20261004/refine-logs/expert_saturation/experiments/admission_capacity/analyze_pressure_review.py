#!/usr/bin/env python3
"""Summarize frozen pressure cells, retaining failures and all call classes."""
import argparse
import json
from pathlib import Path
import statistics

from analyze_call_progress import analyze as call_progress


def read(path):
    return json.loads(path.read_text())


def cell(root, label):
    p = root / label
    row = dict(label=label, status='UNRUN', eligible=False)
    if not (p / 'status.json').exists():
        return row
    status = read(p / 'status.json')
    row.update(status=status['status'], error=status.get('error'))
    if not (p / 'raw.json').exists():
        return row
    raw, met = read(p / 'raw.json'), read(p / 'metrics.json')
    cfg, qual = read(p / 'config.json'), read(p / 'safe-cap-qualification.json')
    completed = [r for r in met['per_request'] if r['status'] == 'completed']
    row.update(raw_status=raw['status'], completed=len(completed),
               usable_blocks=qual['usable_blocks'],
               kv_bytes=cfg['fixed_kv_cache_memory_bytes'],
               policy=cfg['completion_policy'])
    zero_action_control = ('-d0-' in label and cfg['completion_policy'] == 'rotate'
        and raw['status'] == 'INVALID_NO_ACTION'
        and raw.get('error') == 'rotation never applied a forced preemption'
        and raw['actual_preemption_count'] == 0)
    ordinary_complete = raw['status'] == status['status'] == 'COMPLETE'
    if len(completed) != 32 or not (ordinary_complete or zero_action_control):
        return row
    progress = call_progress(p / 'raw.json')
    if progress['returned_tokens'] != 32768:
        raise ValueError('frozen output workload differs')
    n_pure, onset = 0, None
    for step in raw['scheduler_steps']:
        if onset is None and step.get('preempted_request_ids'):
            onset = n_pure + 1  # legacy report uses one-based pure-step coordinate
        pure = (not step.get('recompute_tokens') and
                not any(r.get('prefill_tokens') for r in step['scheduled']) and
                step.get('decode_requests', 0) > 0)
        n_pure += bool(pure)
    # Preserve the original prediction's exclusion of the final scheduler call.
    last = raw['scheduler_steps'][-1]
    last_pure = (not last.get('recompute_tokens') and
                 not any(r.get('prefill_tokens') for r in last['scheduled']) and
                 last.get('decode_requests', 0) > 0)
    row.update(eligible=ordinary_complete, negative_control_only=zero_action_control,
               wall_s=met['observation_duration_s'],
               throughput_rps=met['throughput_rps'],
               mean_completion_s=statistics.mean(r['request_latency_s'] for r in completed),
               max_itl_s=max(max(r['itl_s']) for r in completed),
               preemptions=raw['actual_preemption_count'],
               onset_legacy_pure_coordinate=onset,
               legacy_pure_calls=n_pure-int(last_pure), progress=progress)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    rows = [cell(args.results, f'block{b}-{d}-{a}')
            for b in (0, 1) for d in ('d0', 'd2', 'd4', 'd6')
            for a in ('native', 'rotate')]
    index = {r['label']: r for r in rows}
    pairs = []
    for b in (0, 1):
        for d in ('d0', 'd2', 'd4', 'd6'):
            n, r = [index[f'block{b}-{d}-{a}'] for a in ('native', 'rotate')]
            if not (n['eligible'] and r['eligible']):
                continue
            if n['kv_bytes'] != r['kv_bytes'] or n['usable_blocks'] != r['usable_blocks']:
                raise ValueError('paired resource mismatch')
            pairs.append(dict(block=b, point=d,
                delta_wall_s=r['wall_s']-n['wall_s'],
                throughput_delta_pct=100*(r['throughput_rps']/n['throughput_rps']-1),
                mean_completion_delta_pct=100*(r['mean_completion_s']/n['mean_completion_s']-1),
                native_max_itl_s=n['max_itl_s'], rotate_max_itl_s=r['max_itl_s'],
                native_calls=n['progress']['scheduler_calls'],
                rotate_calls=r['progress']['scheduler_calls']))
    out = dict(cells=rows, pairs=pairs,
               scope='Descriptive matched resource pairs; no significance, noninferiority, '
                     'noise bound, recoverable-waste estimate or online predictor claim. '
                     'Original non-COMPLETE states retained; no-action controls require separate interpretation.')
    with args.output.open('x') as f:
        json.dump(out, f, indent=2, allow_nan=False)
        f.write('\n')


if __name__ == '__main__':
    main()
