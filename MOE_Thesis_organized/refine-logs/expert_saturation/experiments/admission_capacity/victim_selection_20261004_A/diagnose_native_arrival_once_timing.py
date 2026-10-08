#!/usr/bin/env python3
"""Read-only timing context for a completed arrival-once ABBA; stdout only."""
import argparse
import gzip
import json
from pathlib import Path
from statistics import mean


def read(path):
    with (gzip.open(path, 'rt') if path.suffix == '.gz' else path.open()) as stream:
        return json.load(stream)


def cell_data(session, name, cell):
    archive = session / name / 'archive'
    path = next(p for p in (archive / 'raw.json', archive / 'raw.json.gz') if p.is_file())
    raw = read(path)
    events = [event for event in cell['native_victim_observation']['raw_decisions']
              if event.get('arrival_once', {}).get('consumed_before') is False
              and event.get('arrival_once', {}).get('consumed_after') is True]
    if len(events) != 1:
        raise ValueError(f'{name}: expected one proposal/shadow proposal, got {len(events)}')
    event = events[0]
    cutoff = event['host_perf_counter_s'] - raw['measurement_origin_perf_counter_s']
    requests = {row['request_id']: row for row in raw['requests']}
    metrics = {row['request_id']: row for row in cell['metrics']['requests']}
    if requests.keys() != metrics.keys():
        raise ValueError(f'{name}: raw and canonical request IDs differ')
    before = {rid for rid, row in requests.items()
              if row['status'] == 'completed' and row['completion_s'] < cutoff}
    return dict(metrics=metrics, before=before, proposal_s=cutoff, step=event['step'],
                mean_flow_s=cell['primary_mean_flow_s'], duration_s=cell['metrics']['duration_s'],
                engine_calls=raw['engine_call_count'], engine_returns=raw['engine_return_count'],
                output_tokens=cell['metrics']['total_output_tokens'],
                successful_preemptions=sum(e.get('original_preemption_returned') is True
                                          for e in raw.get('preemption_events', [])),
                recovery_count=raw.get('recovery_count'),
                recomputed_tokens=raw.get('recomputed_tokens'))


def comparison(label, old, new):
    ids = sorted(old['before'] & new['before'])
    deltas = {key: mean(new['metrics'][rid][key] - old['metrics'][rid][key] for rid in ids)
              if ids else None for key in ('flow_s', 'ttft_s')}
    def fmt(value):
        return 'UNKNOWN' if value is None else f'{value:+.9f}'
    print(f'{label}: before_ref={len(old["before"])} before_new={len(new["before"])} '
          f'before_both={len(ids)} cohort_mean_flow_delta_s={fmt(deltas["flow_s"])} '
          f'cohort_mean_ttft_delta_s={fmt(deltas["ttft_s"])}')
    print(f'  proposal_steps={old["step"]}/{new["step"]} '
          f'proposal_clock_delta_s={new["proposal_s"]-old["proposal_s"]:+.9f} '
          f'all_request_mean_flow_delta_s={new["mean_flow_s"]-old["mean_flow_s"]:+.9f} '
          f'capture_duration_delta_s={new["duration_s"]-old["duration_s"]:+.9f}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', type=Path, required=True)
    parser.add_argument('--analysis', type=Path, required=True)
    args = parser.parse_args()
    analysis = read(args.analysis)
    if analysis['status'] != 'FOUR_CELLS_COMPLETE_COMPARABLE':
        raise ValueError('Expected the completed comparable four-cell analysis')
    names = sorted(analysis['cells'])
    if [analysis['cells'][name]['arm'] for name in names] != [
            'tail', 'arrival_once', 'arrival_once', 'tail']:
        raise ValueError('Expected tail/arrival_once/arrival_once/tail order')
    cells = {name: cell_data(args.session, name, analysis['cells'][name]) for name in names}
    print('Differences are new minus reference; negative means shorter. All time units are seconds.')
    print('Cohort: completed strictly before each own proposal clock in BOTH runs; tail uses its '
          'shadow proposal. This is descriptive timing context, not causal prefix adjustment.')
    print('Capture duration is canonical metrics.duration_s (raw observation_end_s), '
          'not process/init/warmup time. No per-engine-call latency is inferred from output receipts.')
    for label, pair in analysis['comparisons'].items():
        comparison(label, cells[pair['reference']], cells[pair['candidate']])
    for label, old, new in [('same_tail_later_minus_earlier', names[0], names[3]),
                            ('same_once_later_minus_earlier', names[1], names[2])]:
        comparison(label, cells[old], cells[new])
    for name, cell in cells.items():
        print(f'{name}: mean_flow_s={cell["mean_flow_s"]:.9f} duration_s={cell["duration_s"]:.9f} '
              f'engine_calls/returns={cell["engine_calls"]}/{cell["engine_returns"]} '
              f'output_tokens={cell["output_tokens"]} successful_preemptions={cell["successful_preemptions"]} '
              f'recovery_count={cell["recovery_count"]} recomputed_tokens={cell["recomputed_tokens"]}')
    print('None means UNKNOWN. Equal observed counts do not establish equal internal compute work. '
          'This is one ABBA; requests/events are not independent run repetitions.')


if __name__ == '__main__':
    main()
