#!/usr/bin/env python3
"""All-request outcomes and observed funding episodes; never same-state replay."""
import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
from statistics import mean
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / '20260929_commit_recheck'))
from evaluate_goodput import summarize, pair, quantile

ARMS = ('tail', 'host_missing', 'host_missing', 'tail')
SEMANTICS = dict(
    evidence='NATIVE_SERVING_INPROCESS_HOST_MEASUREMENT',
    clock='External arrival to host engine-return output/completion; no client acknowledgement.',
    cohort='All external requests, including failed and unfinished; missing cells retained.',
    incomplete_flow='Legacy max(timeout_s, observation_end_s-arrival_s) penalty, separately from completed flow.',
    gap='Per-request maximum interval between distinct host output timestamps; TTFT is separate.',
    frontier='Inherited fixed development diagnostic frontier, NOT application SLO or a selected main threshold.',
    causality='Independent policy trajectories with identical external arrivals; matched request/episode IDs are NOT same-state causal interventions.',
    work='Natural output lengths/sequences and recovery counts can differ; rate is NOT equal-work speedup.',
    repeats='R01 cell00 vs cell01; reverse R02 cell03 vs cell02. Runs, not events, are statistical units.',
    cost='Recorded successful funding-selector wall time, including candidate observation in both arms; not complete controller CPU cost.',
    episodes='Observed target/victim intervals and subsequent events are absolute outcomes, not incremental victim harm.',
)


def read(path):
    if path.suffix == '.gz':
        with gzip.open(path, 'rt') as handle:
            return json.load(handle)
    return json.loads(path.read_text())


def optional(path, default=None):
    return read(path) if path.exists() else default


def distribution(values):
    values = [value for value in values if value is not None]
    return dict(n=len(values), mean=mean(values) if values else None,
                p50=quantile(values, .5), p90=quantile(values, .9),
                p95=quantile(values, .95), p99=quantile(values, .99),
                maximum=max(values, default=None))


def selector_summary(store, duration):
    decisions = store.get('funding_victim_decisions', [])
    varying = all_zero = exact_opportunity = better_opportunity = 0
    for event in decisions:
        candidates = event.get('candidates', [])
        signals = [row.get('host_missing_suffix_blocks') for row in candidates]
        known = [value for value in signals if value is not None]
        varying += len(set(known)) > 1
        all_zero += bool(signals) and len(known) == len(signals) and max(known) == 0
        baseline = next((row for row in candidates
                         if row['request'] == event['baseline_victim']), None)
        if baseline and baseline.get('host_missing_suffix_blocks') is not None:
            better_opportunity += any(row.get('host_missing_suffix_blocks') is not None
                and row['host_missing_suffix_blocks'] < baseline['host_missing_suffix_blocks']
                for row in candidates)
            exact_opportunity += any(row['held_blocks'] == baseline['held_blocks']
                and row.get('host_missing_suffix_blocks') is not None
                and row['host_missing_suffix_blocks'] < baseline['host_missing_suffix_blocks']
                for row in candidates)
    wall = [row.get('selector_wall_s') for row in decisions]
    total = sum(value for value in wall if value is not None)
    changed = [row for row in decisions if row.get('changed')]
    return dict(decisions=len(decisions), changed=len(changed),
        signal_varies_decisions=varying, all_candidates_zero_missing=all_zero,
        better_signal_than_tail_opportunities=better_opportunity,
        exact_capacity_better_signal_opportunities=exact_opportunity,
        changed_exact_capacity=sum(row['baseline_held_blocks'] == row['selected_held_blocks'] for row in changed),
        changed_capacity_deltas_blocks=[row['selected_held_blocks']-row['baseline_held_blocks'] for row in changed],
        candidate_count=distribution([len(row.get('candidates', [])) for row in decisions]),
        fallback_counts=dict(Counter(row.get('fallback') or 'NONE' for row in decisions)),
        selector_wall_s=distribution(wall), selector_wall_total_s=total,
        selector_wall_fraction_capture=total/duration if duration else None,
        raw_decisions=decisions)


def episodes(raw, store):
    origin = raw['measurement_origin_perf_counter_s']
    mapping = raw['internal_to_source']
    requests = {row['request_id']: row for row in raw['requests']}
    outputs = defaultdict(list)
    preemptions = defaultdict(list)
    for event in raw.get('output_events', []):
        outputs[event['request_id']].append(event)
    actual = [event for event in raw.get('preemption_events', [])
              if event.get('original_preemption_called') is True
              and event.get('original_preemption_returned') is True]
    for event in actual:
        preemptions[event['request_id']].append(event)
    times = {rid: [event['received_s'] for event in rows] for rid, rows in outputs.items()}
    by_episode = defaultdict(dict)
    for event in store.get('events', []):
        eid = event.get('episode_id') or event.get('oldest_episode_id')
        if eid:
            by_episode[eid][event['event']] = event
    role_sets = dict(target=set(), victim=set())
    rows = []

    def outcome(internal, cutoff, after_step, minimum_output=None):
        rid = mapping.get(internal)
        if rid not in requests:
            return dict(internal_request_id=internal, status='MISSING_REQUEST_JOIN')
        request = requests[rid]
        events = outputs[rid]
        index = bisect_right(times.get(rid, []), cutoff)
        following = next((event for event in events[index:]
                          if minimum_output is None or event['cumulative_tokens'] > minimum_output), None)
        previous = events[index-1] if index else None
        later_preempt = [event for event in preemptions[rid]
                         if event['engine_call_index'] > after_step]
        completion = request.get('completion_s')
        return dict(request_id=rid, internal_request_id=internal,
            status=request['status'], outputs=len(request['output_token_ids']),
            completion_s=completion,
            completion_flow_s=completion-request['arrival_s'] if completion is not None else None,
            completion_after_cutoff_s=completion-cutoff if completion is not None else None,
            previous_output_s=previous['received_s'] if previous else None,
            next_output_s=following['received_s'] if following else None,
            next_output_engine_call=following['engine_call_index'] if following else None,
            next_output_after_cutoff_s=following['received_s']-cutoff if following else None,
            output_gap_containing_cutoff_s=(following['received_s']-previous['received_s']
                                          if following and previous else None),
            subsequent_actual_preemptions=len(later_preempt),
            first_subsequent_preempt=({key: later_preempt[0][key] for key in
                ('engine_call_index', 'method_entered_s', 'native_output_count_before')}
                if later_preempt else None))

    for event in store.get('events', []):
        if event.get('event') != 'oldest_anchor':
            continue
        linked = by_episode[event['episode_id']]
        commit = linked.get('oldest_commit')
        anchor_s = event['host_perf_counter_s'] - origin
        target = outcome(event['target'], anchor_s, event['step'], event['target_output_count'])
        if target.get('request_id'):
            role_sets['target'].add(target['request_id'])
        victim_source = mapping.get(event['victim'])
        forced = bool(commit and commit.get('forced'))
        actual_match = [p for p in preemptions.get(victim_source, [])
                        if commit and p['engine_call_index'] == commit['step']]
        preempt = actual_match[0] if len(actual_match) == 1 else None
        victim = None
        if forced and preempt:
            victim = outcome(event['victim'], preempt['method_returned_s'],
                             commit['step'], preempt['native_output_count_before'])
            role_sets['victim'].add(victim_source)
        rows.append(dict(episode_id=event['episode_id'], anchor_step=event['step'],
            anchor_s=anchor_s, anchor=event, commit=commit,
            cancellation=linked.get('oldest_cancel'), target=target, victim=victim,
            forced_preempt_join_count=len(actual_match),
            actual_preempt=preempt,
            join_status=('JOINED' if not forced or len(actual_match) == 1 else 'INVALID_PREEMPT_JOIN')))
    return dict(actual_preemption_events=len(actual),
        repeated_preemption_requests={rid: len(events) for rid, events in preemptions.items() if len(events) > 1},
        role_request_ids={key: sorted(value) for key, value in role_sets.items()},
        episode_count=len(rows), forced_victim_episodes=sum(row['victim'] is not None for row in rows),
        invalid_forced_joins=sum(row['join_status'] != 'JOINED' for row in rows),
        actual_released_blocks=distribution([row['commit'].get('actual_released_blocks')
            for row in rows if row['commit']]),
        target_anchor_to_next_output_s=distribution([row['target'].get('next_output_after_cutoff_s') for row in rows]),
        victim_actual_output_gap_s=distribution([row['victim']['output_gap_containing_cutoff_s']
            for row in rows if row['victim']]), episodes=rows)


def read_cell(directory, arm, expected_requests, timeout_s):
    data = dict(directory=str(directory), arm=arm)
    if not directory.exists():
        return dict(**data, status='NOT_RUN'), None
    archive = directory / 'archive'
    if not archive.exists():
        archive = directory / 'output'
    data['engine_status'] = optional(archive / 'status.json')
    data['config'] = optional(archive / 'config.json')
    path = next((archive / name for name in ('raw.json', 'raw.json.gz')
                 if (archive / name).exists()), None)
    if path is None:
        log = directory / 'launch.log'
        data['launch_log_tail'] = log.read_text(errors='replace')[-2000:] if log.exists() else None
        return dict(**data, status='NO_MEASUREMENT'), None
    raw = read(path)
    store = optional(archive / 'selective-store.json', {})
    data.update(raw_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                capture_status=raw.get('status'), capture_error=raw.get('error'),
                observed_requests=len(raw.get('requests', [])), expected_requests=expected_requests,
                missing_expected_request_records=max(0, expected_requests-len(raw.get('requests', []))))
    try:
        metrics = summarize(raw, expected_requests, timeout_s)
        metrics['distributions'] = {key: distribution([row[key] for row in metrics['requests']])
            for key in ('ttft_s', 'flow_s', 'actual_completion_flow_s', 'max_gap_s')}
        metrics['failed_or_unfinished_requests'] = [row for row in metrics['requests'] if not row['completed']]
        data['metrics'] = metrics
        data['status'] = ('COMPLETE' if raw.get('status') == 'COMPLETE'
            and raw.get('error') is None and metrics['completed'] == expected_requests
            and (data['engine_status'] or {}).get('status') == 'COMPLETE' else 'INCOMPLETE')
        data['funding_selector'] = selector_summary(store, raw['observation_end_s'])
        data['action_outcomes'] = episodes(raw, store)
        data['event_counts'] = dict(Counter(row.get('event') for row in store.get('events', [])))
    except (ValueError, KeyError, TypeError, IndexError) as error:
        data.update(status='ANALYSIS_ERROR', analysis_error=f'{type(error).__name__}: {error}',
            retained_request_statuses=[{key: row.get(key) for key in
                ('request_id', 'status', 'arrival_s', 'completion_s')} for row in raw.get('requests', [])])
    return data, raw


def compare(reference, candidate, old_raw, new_raw):
    result = pair(reference['metrics'], candidate['metrics'])
    old = {row['request_id']: row for row in old_raw['requests']}
    new = {row['request_id']: row for row in new_raw['requests']}
    result['sequence_differences'] = [rid for rid in sorted(old)
        if old[rid]['output_token_ids'] != new[rid]['output_token_ids']]
    result['stop_differences'] = [rid for rid in sorted(old)
        if old[rid].get('stop_reason') != new[rid].get('stop_reason')]
    role_sets = []
    for cell in (reference, candidate):
        roles = cell['action_outcomes']['role_request_ids']
        role_sets.append({key: set(values) for key, values in roles.items()})

    def role(rid, roles):
        target, victim = rid in roles['target'], rid in roles['victim']
        return 'target_and_victim' if target and victim else 'target' if target else 'victim' if victim else 'other'

    buckets = defaultdict(list)
    for row in result['per_request']:
        rid = row['request_id']
        row.update(reference_role=role(rid, role_sets[0]), candidate_role=role(rid, role_sets[1]))
        buckets[row['reference_role'] + '->' + row['candidate_role']].append(row)

    def effects(rows):
        return {key: dict(distribution=distribution([row[key] for row in rows]),
            worse=sum(row[key] is not None and row[key] > 0 for row in rows),
            better=sum(row[key] is not None and row[key] < 0 for row in rows),
            sum_s=sum(row[key] for row in rows if row[key] is not None))
            for key in ('flow_difference_s', 'ttft_difference_s', 'max_gap_difference_s')}

    result['all_request_effects'] = effects(result['per_request'])
    result['role_transition_effects'] = {key: dict(requests=len(rows), effects=effects(rows)) for key, rows in buckets.items()}
    result['both_complete'] = reference['status'] == candidate['status'] == 'COMPLETE'
    result['raw_distribution_differences'] = {metric: {q:
        (candidate['metrics']['distributions'][metric][q]-reference['metrics']['distributions'][metric][q]
         if candidate['metrics']['distributions'][metric][q] is not None
         and reference['metrics']['distributions'][metric][q] is not None else None)
        for q in ('mean', 'p50', 'p95', 'p99', 'maximum')}
        for metric in ('ttft_s', 'actual_completion_flow_s', 'max_gap_s')}
    return result


def analyze(session, expected_requests, timeout_s):
    plan = optional(session / 'plan.json', {})
    arms = tuple(plan.get('arms', ARMS))
    if len(arms) != 4 or arms[0] != arms[3] or arms[1] != arms[2]:
        raise ValueError('Requires a four-cell ABBA plan')
    cells, raws = {}, {}
    for index, arm in enumerate(arms):
        name = f'cell-{index:02d}-{arm}'
        try:
            cells[name], raws[name] = read_cell(session / name, arm, expected_requests, timeout_s)
        except (OSError, ValueError, KeyError, TypeError) as error:
            cells[name] = dict(directory=str(session / name), arm=arm,
                status='ARTIFACT_READ_ERROR', error=f'{type(error).__name__}: {error}')
            raws[name] = None
    comparisons = {}
    for label, ref_index, cand_index in [('R01', 0, 1), ('R02_reverse', 3, 2)]:
        ref = f'cell-{ref_index:02d}-{arms[ref_index]}'
        cand = f'cell-{cand_index:02d}-{arms[cand_index]}'
        row = dict(reference=ref, candidate=cand)
        if all('metrics' in cells[key] and 'action_outcomes' in cells[key] for key in (ref, cand)):
            try:
                row.update(status='PAIRED', **compare(cells[ref], cells[cand], raws[ref], raws[cand]))
            except (ValueError, KeyError, TypeError) as error:
                row.update(status='INVALID_PAIR', error=f'{type(error).__name__}: {error}')
        else:
            row['status'] = 'UNAVAILABLE'
        comparisons[label] = row
    return dict(schema_version=1, session=str(session), semantics=SEMANTICS,
        analysis_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        receipt=optional(session / 'receipt.json'), plan=optional(session / 'plan.json'),
        status='FOUR_CELLS_COMPLETE' if all(row['status'] == 'COMPLETE' for row in cells.values()) else 'PARTIAL_OR_FAILED',
        cells=cells, comparisons=comparisons)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', type=Path, required=True)
    parser.add_argument('--expected-requests', type=int, default=128)
    parser.add_argument('--timeout-s', type=float, default=180)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.session, args.expected_requests, args.timeout_s)
    with args.output.open('x') as output:
        json.dump(result, output, indent=2, ensure_ascii=False, allow_nan=False)
        output.write('\n')
