"""Independent cumulative-wait decomposition after the owner's pair qualification."""
import argparse
import json
from pathlib import Path

from partition import partition


def compare(results, main):
    cells = {}
    pairs = []
    owners = {p['block']: p for p in main['performance_comparisons']}
    for block in (0, 1):
        if owners[block]['status'] != 'COMPLETE':
            raise ValueError('Owner has not qualified this full-service pair')
        for scope in ('selected', 'native_full'):
            name = f'block{block}-{scope}'
            cells[name] = partition(json.loads((results/name/'raw.json').read_text()))
            if not cells[name]['comparable_complete_service']:
                raise ValueError('Incomplete source cannot enter a comparison')
        selected, full = [cells[f'block{block}-{scope}'] for scope in ('selected','native_full')]
        old = {r['request_id']:r for r in selected['requests']}
        if set(old) != {r['request_id'] for r in full['requests']}:
            raise ValueError('Request identity mismatch')
        rows = []
        for r in full['requests']:
            s = old[r['request_id']]
            rows.append(dict(request_id=r['request_id'],
                full_minus_selected={k:r[k]-s[k] for k in selected['totals']},
                marked_gap_counts=dict(selected=s['marked_gap_count'], native_full=r['marked_gap_count']),
                outputs=dict(selected=s['outputs'], native_full=r['outputs'])))
        pairs.append(dict(block=block,
            full_minus_selected_request_seconds={k:full['totals'][k]-v for k,v in selected['totals'].items()},
            requests=rows))
    consistently_higher = set.intersection(*[
        {r['request_id'] for r in p['requests'] if r['full_minus_selected']['gap_with_preemption_s']>0}
        for p in pairs])
    return dict(status='MEASUREMENT_ONLY', pairs=pairs,
        consistently_higher_marked_wait=sorted(consistently_higher),
        cells={name:dict(counts=c['counts'], totals=c['totals']) for name,c in cells.items()},
        semantics=[
            'Pair validity and primary service metrics are reused from the unique owner analysis.',
            'Each arm evolves its own outputs and preemptions; no diagnostic event projection or trace matching.',
            'Marked and other intervals may change category across policies, so bucket deltas are not isolated causal taxes.',
            'Different EOS/output quantities remain observed outcomes, not equal-work speedup.',
            'Two interleaved pairs and shared requests do not establish independent episode generalization.',
            'This adds cumulative-wait attribution, not another sample, main performance table or SLO threshold.'])


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--results', type=Path, required=True)
    p.add_argument('--main-analysis', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = compare(args.results, json.loads(args.main_analysis.read_text()))
    result['sources'] = dict(results=str(args.results), main_analysis=str(args.main_analysis))
    with args.output.open('x') as f:
        json.dump(result, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')
    print(json.dumps([p['full_minus_selected_request_seconds'] for p in result['pairs']], indent=2))
