#!/usr/bin/env python3
"""Join existing recovery allocation records to host LOAD acknowledgements."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path


def _time(record):
    return record.get('begin_host_perf_s', record.get('host_perf_s'))


def _blocks(record):
    before, after = record['before'], record['after']
    args, desc = record['arguments'], record['descriptor']
    free, reserved = before['free_gpu_blocks'], args['reserved_blocks']
    result = dict(held_before=before['held_gpu_blocks'], held_after=after['held_gpu_blocks'],
        free_before=free, free_after=after['free_gpu_blocks'], reserved_blocks=reserved,
        exact=desc['exact'], slot_extra_blocks=desc.get('slot_extra_blocks'),
        full_fit_extra_blocks=desc.get('full_fit_extra_blocks'),
        required_free_blocks=desc.get('required_free_blocks'), shortfall_blocks=desc.get('shortfall_blocks'))
    if desc['exact']:
        result['slot_with_reserve_deficit'] = max(0, desc['slot_extra_blocks']+reserved-free)
        result['full_fit_deficit'] = max(0, desc['full_fit_extra_blocks']-free) if args['full_sequence_must_fit'] else None
    return result


def analyze(raw, observer, recovery):
    origin = raw['measurement_origin_perf_counter_s']
    end = origin+raw['observation_end_s']
    mapping = raw['internal_to_source']
    episodes, pending, orphans = [], {}, []
    for index, record in sorted(enumerate(observer['events']), key=lambda pair: (_time(pair[1]), pair[0])):
        rid, kind, stamp = record['request'], record['kind'], _time(record)
        if kind == 'preempt' and not record.get('exception'):
            if rid in pending:
                pending.pop(rid).update(end_host_perf_s=stamp, end_reason='next_preempt_before_resumed_schedule')
            episode = dict(episode=len(episodes), request=rid, source_request=mapping.get(rid),
                begin_host_perf_s=stamp, begin_s=stamp-origin, preempt_record=record,
                preempt_record_index=index, attempts=[], resumed_schedule_record=None)
            episodes.append(episode); pending[rid] = episode
        elif rid in pending and kind == 'allocate':
            pending[rid]['attempts'].append(dict(record_index=index, record=record))
        elif rid in pending and kind == 'resumed_schedule':
            pending.pop(rid).update(end_host_perf_s=stamp, end_reason='first_resumed_schedule',
                                    resumed_schedule_record=record)
        else:
            orphans.append(dict(record_index=index, record=record))
    for episode in pending.values():
        episode.update(end_host_perf_s=end, end_reason='observation_end_without_resumed_schedule')

    jobs = {}
    for event in recovery['events']:
        if event.get('is_store') is False and 'job_id' in event:
            job = jobs.setdefault(event['job_id'], dict(job_id=event['job_id'], request=event['request'], times={}))
            job['times'][event['kind']] = event['host_perf_s']
    by_request = defaultdict(list)
    for job in jobs.values():
        by_request[job['request']].append(job)
    all_counts = Counter()
    for episode in episodes:
        begin, finish = episode['begin_host_perf_s'], episode['end_host_perf_s']
        loads = [job for job in by_request[episode['request']] if begin <= min(job['times'].values()) < finish]
        episode['load_jobs'] = loads
        acks = sorted((job['times']['ack_retired'], job['job_id']) for job in loads
                      if 'ack_retired' in job['times'] and job['times']['ack_retired'] <= finish)
        counts, failures = Counter(), []
        for position, attempt in enumerate(episode['attempts']):
            record = attempt['record']; t0, t1 = record['begin_host_perf_s'], record['end_host_perf_s']
            prior = [(t, jid) for t, jid in acks if t <= t0]
            if any(t0 < t <= t1 for t, _ in acks): phase = 'ack_during_attempt'
            elif prior: phase = 'after_load_ack'
            elif acks: phase = 'before_load_ack'
            else: phase = 'load_without_ack_in_episode' if loads else 'no_load_in_episode'
            outcome = 'exception' if record.get('exception') else 'success' if record['success'] else 'failure'
            attempt.update(phase=phase, outcome=outcome, begin_s=t0-origin, end_s=t1-origin,
                acknowledged_load_jobs=[jid for _, jid in prior],
                since_latest_ack_s=t0-prior[-1][0] if prior else None, blocks=_blocks(record))
            counts[phase+'/'+outcome] += 1
            counts['formula_exact' if record['descriptor']['exact'] else 'formula_unsupported'] += 1
            if phase == 'after_load_ack' and outcome == 'failure':
                later = next((x for x in episode['attempts'][position+1:]
                              if x['record'].get('success') and not x['record'].get('exception')), None)
                failures.append(dict(record_index=attempt['record_index'], begin_s=t0-origin,
                    blocks=attempt['blocks'], first_later_success=(dict(record_index=later['record_index'],
                        begin_s=later['record']['begin_host_perf_s']-origin, blocks=_blocks(later['record'])) if later else None)))
        episode.update(end_s=finish-origin, counts=dict(counts), post_ack_failures=failures)
        all_counts.update(counts)
    return dict(status='ANALYZED', observer_status=observer.get('status'),
        qualification=observer.get('qualification'), episodes=episodes, counts=dict(all_counts),
        episode_end_counts=dict(Counter(e['end_reason'] for e in episodes)), orphan_records=orphans,
        unmapped_requests=sorted({e['request'] for e in episodes if e['source_request'] is None}),
        semantics='All allocation records retained. Each successful preempt starts a separate episode, ending at '
            'its first resumed schedule, another preempt, or observation end. after_load_ack means a recorded '
            'LOAD ack preceded allocation begin; it does not establish absolute GPU completion. Deficits only '
            'use the observer exact descriptor. Equal held counts do not establish unchanged block identity. '
            'No request-benefit estimate; use the existing all-request analyzer separately.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cell-output', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    destination = args.output or args.cell_output.parent/'capacity-allocation-analysis.json'
    if destination.exists():
        raise FileExistsError(destination)
    names = ('raw.json', 'capacity-handoff.json', 'recovery-order.json')
    inputs = [json.loads((args.cell_output/name).read_text()) for name in names]
    result = analyze(*inputs)
    result['inputs'] = [str(args.cell_output/name) for name in names]
    with destination.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(output=str(destination), episodes=len(result['episodes']), counts=result['counts'])))


if __name__ == '__main__':
    main()
