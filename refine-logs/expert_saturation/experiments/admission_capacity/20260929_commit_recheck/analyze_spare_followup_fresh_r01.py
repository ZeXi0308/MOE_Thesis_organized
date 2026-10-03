"""Frozen new-input off/on test with an additional native full-save reference."""
import argparse
import hashlib
import json
from pathlib import Path
from collections import Counter
from analyze_spare_followup_pair_r01 import analyze
from evaluate_goodput import summarize, pair


def run(session):
    result = analyze(session, ('off', 'on'))
    folder = session / 'cell-02-native_full_native' / 'archive'
    raw = json.loads((folder / 'raw.json').read_text())
    config = json.loads((folder / 'config.json').read_text())
    status = json.loads((folder / 'status.json').read_text())
    store = json.loads((folder / 'selective-store.json').read_text())
    assert config['variant'] == 'native_full_native'
    assert config.get('spare_followup') is False
    assert config.get('capacity_victim') is False
    assert store.get('store_scope') == 'native_full'
    assert store.get('scheduler_schedule_overridden') is False
    assert store.get('native_calc_overridden') is False
    metrics = summarize(raw, 128, 180)
    complete = (raw.get('status') == 'COMPLETE' and raw.get('error') is None
        and status.get('status') == 'COMPLETE' and status.get('error') is None
        and status.get('capture_status') == 'COMPLETE'
        and len(raw['requests']) == metrics['completed'] == status['requests_completed'] == 128
        and metrics['failed'] == metrics['unfinished'] == 0)
    native_ids = {r['request_id'] for r in raw['requests']}
    assert len(native_ids) == 128
    comparisons = {}
    for label in ('off', 'on'):
        arm_metrics = result['arms'][label]['metrics']
        comparison = pair(metrics, arm_metrics)
        other = json.loads((session / f'cell-{0 if label == "off" else 1:02d}-spare_followup_{label}'
                           / 'archive' / 'raw.json').read_text())
        requests = {r['request_id']: r for r in other['requests']}
        assert set(requests) == native_ids
        comparisons[label] = dict(full_cohort_pair=comparison,
            output_sequence_difference_requests=[r['request_id'] for r in raw['requests']
                if r['output_token_ids'] != requests[r['request_id']]['output_token_ids']],
            stop_reason_difference_requests=[r['request_id'] for r in raw['requests']
                if r.get('stop_reason') != requests[r['request_id']].get('stop_reason')])
    result['native_reference'] = dict(complete=complete, metrics=metrics,
        actual_preemption_count=raw.get('actual_preemption_count'),
        stop_reason_counts=dict(Counter(r.get('stop_reason') for r in raw['requests'])),
        source_sha256={name: hashlib.sha256((folder / name).read_bytes()).hexdigest()
                       for name in ('raw.json', 'config.json', 'selective-store.json')},
        comparisons=comparisons)
    result['predeclared_exploration_criteria']['native_reference_complete'] = complete
    result['all_exploration_criteria_met'] = all(result['predeclared_exploration_criteria'].values())
    result['status'] = 'COMPLETE_FRESH_TRIPLET' if complete and result['status'] == 'COMPLETE_PAIR' else 'INCOMPLETE_FRESH_TRIPLET'
    result['limitations'] = [
        'One frozen new-input episode; input exclusion coverage is specified in INPUT_STATS, not global unseen text.',
        'Primary acceptance remains on versus capacity-qualified Q1 off; native is a system reference, not a replacement denominator.',
        'Policy trajectories may differ in output content; no equal-work, output-quality, statistical-significance or novelty claim.',
        'All primary targets, victims and bypassed requests remain in full-cohort metrics; overlapping actor costs are not additive.']
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = run(args.session)
    with args.output.open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps(dict(status=result['status'], criteria=result['predeclared_exploration_criteria'])))
