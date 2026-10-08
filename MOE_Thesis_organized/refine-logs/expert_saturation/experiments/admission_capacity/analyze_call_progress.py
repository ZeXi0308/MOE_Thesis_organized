#!/usr/bin/env python3
"""Count all calls and actual output progress without discarding mixed calls.

Read-only diagnostic for synchronous one-scheduler-step-per-engine-call traces.
No optimality, removable waste, counterfactual saving, or significance inference.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


def analyze(path):
    content = path.read_bytes()
    raw = json.loads(content)
    steps, calls = raw['scheduler_steps'], raw['engine_steps']
    by_step = {s['step']: s for s in steps}
    if len(by_step) != len(steps):
        raise ValueError('duplicate scheduler step')
    seen = set()
    phases = defaultdict(lambda: dict(calls=0, returned_tokens=0,
                                     decode_positions=0, engine_seconds=0.0))
    widths = Counter()
    previous_end = None
    for call in calls:
        start, end = call['scheduler_step_start'], call['scheduler_step_end']
        if end != start + 1 or start in seen or not call['completed']:
            raise ValueError('requires completed one-to-one synchronous calls')
        if previous_end is not None and call['start_s'] < previous_end:
            raise ValueError('overlapping engine call times')
        previous_end = call['returned_s']
        seen.add(start)
        s = by_step[start]
        rows = s['scheduled']
        phase = ('recompute' if s.get('recompute_tokens', 0) else
                 'prefill' if any(r.get('prefill_tokens', 0) for r in rows) else
                 'pure_decode' if s.get('decode_requests', 0) else 'other')
        item = phases[phase]
        item['calls'] += 1
        item['returned_tokens'] += call['new_output_tokens']
        item['decode_positions'] += sum(r.get('decode_tokens', 0) for r in rows)
        duration = call['returned_s'] - call['start_s']
        if duration < 0:
            raise ValueError('negative duration')
        item['engine_seconds'] += duration
        widths[len(rows)] += 1
    if seen != set(by_step):
        raise ValueError('unmatched scheduler calls')
    returned = sum(p['returned_tokens'] for p in phases.values())
    retained = sum(len(r['output_token_ids']) for r in raw['requests'])
    if returned != retained:
        raise ValueError(f'output mismatch: returned={returned}, retained={retained}')
    return dict(raw_path=str(path), raw_sha256=hashlib.sha256(content).hexdigest(),
                raw_status=raw['status'], scheduler_calls=len(steps),
                engine_calls=len(calls), phases=dict(phases),
                scheduled_width_histogram=dict(sorted(widths.items())),
                returned_tokens=returned, retained_tokens=retained,
                includes_final_call=True,
                engine_seconds=sum(p['engine_seconds'] for p in phases.values()),
                interpretation='Observed phase accounting; mixed calls retain useful output. '
                'No physical lower bound, recoverable waste or action counterfactual inferred.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('raw', nargs='+', type=Path)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    result = [analyze(path) for path in args.raw]
    with args.output.open('x') as out:
        json.dump(dict(cells=result), out, indent=2, allow_nan=False)
        out.write('\n')


if __name__ == '__main__':
    main()
