"""Rebuild descriptive H128 preemption/output locations; no counterfactual effects."""
import argparse
import hashlib
import json
from pathlib import Path


def analyze(session):
    out = {'schema_version': 1,
           'scope': 'Descriptive locations within the observed complete H128 trajectories; no counterfactuals, action values, or blind hypotheses',
           'cells': {}}
    for cell in sorted(session.glob('cell-*')):
        data = (cell / 'archive/raw.json').read_bytes()
        sha = hashlib.sha256(data).hexdigest()
        if sha != json.loads((cell / 'output_sha256.json').read_text())['raw.json']:
            raise ValueError('Archived raw hash differs')
        raw = json.loads(data)
        if raw['status'] != 'COMPLETE' or len(raw['requests']) != 128:
            raise ValueError('Incomplete H128 cohort')
        requests = {r['request_id']: r for r in raw['requests']}
        events, paired, largest = raw['preemption_events'], [], []
        if raw['actual_preemption_count'] != len(events) or not all(
                e['original_preemption_called'] and e['original_preemption_returned'] for e in events):
            raise ValueError('Preemption events include unsuccessful attempts')
        for rid, request in requests.items():
            es = sorted([e for e in events if e['request_id'] == rid], key=lambda e: e['method_entered_s'])
            for first, second in zip(es, es[1:]):
                n = second['last_returned_output_count'] - first['last_returned_output_count']
                if n < 0:
                    raise ValueError('Returned prefix shrank')
                paired.append({'request_id': rid, 'first_preemption_s': first['method_entered_s'],
                               'next_preemption_s': second['method_entered_s'],
                               'new_returned_tokens_between_preemptions': n})
        for rid, request in requests.items():
            if len(request['token_times_s']) < 2:
                continue
            start, end = max(zip(request['token_times_s'], request['token_times_s'][1:]), key=lambda x: x[1]-x[0])
            es = [e for e in events if e['request_id'] == rid and start <= e['method_entered_s'] <= end]
            largest.append({'request_id': rid, 'gap_start_s': start, 'gap_end_s': end, 'gap_s': end-start,
                            'preemptions_inside_gap': [{'at_s': e['method_entered_s'],
                                'ms_since_last_returned_output': 1000*(e['method_entered_s']-start),
                                'last_returned_output_count': e['last_returned_output_count']} for e in es]})
        out['cells'][cell.name] = {'raw_sha256': sha, 'preemptions': len(events),
            'consecutive_preemption_pairs': len(paired),
            'zero_outputs_between': sum(e['new_returned_tokens_between_preemptions'] == 0 for e in paired),
            'one_two_outputs_between': sum(1 <= e['new_returned_tokens_between_preemptions'] <= 2 for e in paired),
            'top_three_observed_gaps': sorted(largest, key=lambda x: x['gap_s'], reverse=True)[:3],
            'preemption_pairs': paired}
    if len(out['cells']) != 3:
        raise ValueError('Need all three complete H128 cells')
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = analyze(Path(__file__).parent / 'moe-a-h128-perf-guard-session-r02-20260930')
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
