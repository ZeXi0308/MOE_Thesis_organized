#!/usr/bin/env python3
"""Frozen repeated oldest-admission pilot: full cohort and every action chain.

The three arms are independent trajectories of the same 128 seen requests.
Episode receipts establish decisions; raw output and preemption records establish
effects. Episodes are not independent statistical replicates.
"""

import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
import json
from pathlib import Path

from analyze_native_oldest_admission_triplet_r01 import (
    MODES, PAIRS, discover, _request_metrics,
)
from analyze_native_residency_victim_triplet_r01 import score, victim_receipts
from analyze_protection_yield_triplet_r01 import compare, require


OLDEST_EVENTS = (
    'oldest_anchor', 'oldest_prepare', 'oldest_commit_job_gate',
    'oldest_commit', 'oldest_cancel', 'oldest_native_admission',
    'oldest_target_new_output', 'oldest_target_terminal', 'oldest_retire',
)


def _one(rows, name):
    matches = [row for row in rows if row.get('event') == name]
    require(len(matches) <= 1, f'Multiple {name} receipts in one episode')
    return matches[0] if matches else None


def _source_request(raw, internal_id):
    matches = [r for r in raw['requests'] if r['internal_request_id'] == internal_id]
    require(len(matches) == 1, f'Missing or duplicate internal request {internal_id}')
    return matches[0]


def _preemptions(raw, internal_id):
    return sorted((event for event in raw.get('preemption_events', [])
                   if event.get('internal_request_id') == internal_id
                   and event.get('original_preemption_called') is True
                   and event.get('original_preemption_returned') is True),
                  key=lambda event: event['method_entered_s'])


def _admissions(store, origin, internal_id):
    return [dict(step=row.get('step'), time_s=row['host_perf_counter_s'] - origin,
                 output_count=row.get('output_count'),
                 num_preemptions=row.get('num_preemptions'),
                 held_blocks=row.get('held_blocks'))
            for row in store['residency_admissions']
            if row.get('request') == internal_id]


def _request_chain(cell, internal_id, after_s):
    """Preserve the complete raw preemption/admission trail for an episode peer."""
    raw, store = cell['raw'], cell['store']
    request = _source_request(raw, internal_id)
    times = request['token_times_s']
    origin = raw['measurement_origin_perf_counter_s']
    preemptions = _preemptions(raw, internal_id)
    admissions = _admissions(store, origin, internal_id)
    trail = []
    for event in preemptions:
        when = event['method_entered_s']
        position = bisect_right(times, when)
        later = [row for row in admissions if row['time_s'] > when]
        trail.append(dict(
            step=event['engine_call_index'], time_s=when,
            returned_s=event.get('method_returned_s'),
            output_count_at_preemption=event.get('last_returned_output_count'),
            prior_output_s=times[position - 1] if position else None,
            next_output_s=times[position] if position < len(times) else None,
            next_recorded_running_admission=later[0] if later else None,
            output_gap_s=(times[position] - times[position - 1]
                          if 0 < position < len(times) else None),
        ))
    position = bisect_right(times, after_s)
    return dict(
        internal_request_id=internal_id,
        source_request_id=request['request_id'],
        status=request.get('status'), stop_reason=request.get('stop_reason'),
        output_count=len(request['output_token_ids']),
        completion_s=request.get('completion_s'),
        output_count_at_reference=position,
        last_output_at_or_before_reference_s=times[position - 1] if position else None,
        first_output_after_reference_s=times[position] if position < len(times) else None,
        all_actual_preemptions=trail,
        recorded_running_admissions=admissions,
        request_metrics=_request_metrics(cell, request['request_id']),
    )


def _episode(cell, mode, episode_id, rows, indexed_events):
    raw, store = cell['raw'], cell['store']
    anchor = _one(rows, 'oldest_anchor')
    require(anchor is not None, f'Episode {episode_id} has no anchor')
    prepare = _one(rows, 'oldest_prepare')
    job_gate = _one(rows, 'oldest_commit_job_gate')
    commit = _one(rows, 'oldest_commit')
    cancel = _one(rows, 'oldest_cancel')
    receipt = _one(rows, 'oldest_native_admission')
    observed_output = _one(rows, 'oldest_target_new_output')
    terminal = _one(rows, 'oldest_target_terminal')
    retire = _one(rows, 'oldest_retire')
    require(retire is not None, f'Episode {episode_id} did not retire')
    require(not (commit and cancel), f'Episode {episode_id} both committed and cancelled')
    target_id, victim_id = anchor.get('target'), anchor.get('victim')
    require(target_id and victim_id and target_id != victim_id,
            f'Episode {episode_id} missing distinct target/victim')
    origin = raw['measurement_origin_perf_counter_s']
    anchor_s = anchor['host_perf_counter_s'] - origin
    target = _request_chain(cell, target_id, anchor_s)
    victim = _request_chain(cell, victim_id, anchor_s)
    raw_preemptions = _preemptions(raw, victim_id)
    actual = [event for event in raw_preemptions
              if commit and event.get('engine_call_index') == commit.get('step')]
    forced = bool(commit and commit.get('forced') is True)
    actual_forced = forced and len(actual) == 1
    victim_preempt_s = actual[0]['method_entered_s'] if actual_forced else None
    if victim_preempt_s is not None:
        victim_at_preempt = _request_chain(cell, victim_id, victim_preempt_s)
        later_admissions = [row for row in victim_at_preempt['recorded_running_admissions']
                            if row['time_s'] > victim_preempt_s]
        victim_readmit = later_admissions[0] if later_admissions else None
        later_preemptions = [row for row in victim_at_preempt['all_actual_preemptions']
                             if row['time_s'] > victim_preempt_s]
    else:
        victim_at_preempt, victim_readmit, later_preemptions = None, None, []
    target_admissions = [row for row in target['recorded_running_admissions']
                         if row['time_s'] >= anchor_s and row.get('num_preemptions', 0) > 0]
    target_readmit = target_admissions[0] if target_admissions else None
    target_first = target['first_output_after_reference_s']
    target_last = target['last_output_at_or_before_reference_s']
    prior_target_preemptions = [event for event in _preemptions(raw, target_id)
                                if event['method_entered_s'] <= anchor_s]
    anchor_valid = (
        anchor.get('mode') == mode and anchor.get('trigger_threshold_s') == 1.0
        and type(anchor.get('target_output_count')) is int
        and anchor['target_output_count'] > 0
        and target['output_count_at_reference'] == anchor['target_output_count']
        and anchor.get('target_observed_output_age_s', -1) >= 1.0
        and abs(anchor_s - (anchor['target_last_output_observed_perf_s'] - origin)
                - anchor['target_observed_output_age_s']) < .001
        and type(anchor.get('target_waiting_index_before')) is int
        and anchor['target_waiting_index_before'] >= 0
        and type(anchor.get('free_blocks')) is int
        and type(anchor.get('target_full_history_need_blocks')) is int
        and type(anchor.get('victim_held_blocks')) is int
        and anchor['free_blocks'] < anchor['target_full_history_need_blocks']
        and anchor['free_blocks'] + anchor['victim_held_blocks']
            >= anchor['target_full_history_need_blocks']
        and anchor.get('deficit_blocks') ==
            anchor['target_full_history_need_blocks'] - anchor['free_blocks']
        and anchor.get('native_inflight_reserved_blocks') == 0
        and bool(prior_target_preemptions)
        and anchor.get('target_num_preemptions') == len(prior_target_preemptions)
    )
    queue_verified = (
        anchor.get('head_after') == anchor.get('head_before') if mode == 'native'
        else anchor.get('head_after') == target_id
    )
    linked = all(event is None or event.get('target') == target_id
                 for event in (prepare, commit, cancel, receipt, observed_output,
                               terminal, retire))
    linked = linked and all(event is None or event.get('victim') == victim_id
                             for event in (prepare, commit))
    native_receipt_verified = bool(receipt and receipt.get('step', -1) >= anchor['step']
        and (commit is None or receipt.get('step', -1) >= commit.get('step', 0))
        and receipt.get('native_admission') in
            ('SCHEDULED_TOKENS', 'ASYNC_LOAD_ADMITTED'))
    first_output_or_terminal = target_first is not None or (
        terminal is not None and target['status'] == 'completed')
    protection_starts = [event for event in store['events']
        if event.get('event') == 'protection_start'
        and event.get('oldest_episode_id') == episode_id]
    protection_releases = [event for event in store['events']
        if event.get('event') == 'protection_release'
        and event.get('oldest_episode_id') == episode_id]
    commit_admissions = [event for event in store['events']
        if event.get('event') in ('recovery_commit_admitted', 'direct_commit')
        and event.get('oldest_episode_id') == episode_id]
    protection_valid = (len(protection_starts) == len(protection_releases) == 1
        and protection_starts[0].get('protection_origin') == 'OLDEST_FUNDED'
        and protection_starts[0].get('request') == target_id
        and protection_starts[0].get('primary_victim') ==
            (victim_id if forced else None)
        and protection_starts[0].get('recovery_min_outputs') == 1
        and protection_releases[0].get('request') == target_id
        and protection_releases[0].get('protection_start_step') ==
            protection_starts[0].get('step')
        and ((protection_releases[0].get('reason') == 'OUTPUT_GOAL_REACHED'
              and protection_releases[0].get('new_output_tokens', 0) >= 1)
             or str(protection_releases[0].get('reason', '')).startswith('TERMINAL_')))
    event_positions = {name: next((index for index, event in indexed_events
                                   if event.get('episode_id') == episode_id
                                   and event.get('event') == name), None)
                       for name in ('oldest_anchor', 'oldest_prepare', 'oldest_commit',
                                    'oldest_cancel', 'oldest_retire')}
    ordered = (event_positions['oldest_anchor'] is not None
        and event_positions['oldest_retire'] is not None
        and event_positions['oldest_anchor'] < event_positions['oldest_retire']
        and all(position is None or event_positions['oldest_anchor'] < position
                < event_positions['oldest_retire']
                for name, position in event_positions.items()
                if name not in ('oldest_anchor', 'oldest_retire')))
    retire_reason = retire.get('reason')
    retire_consistent = (
        retire.get('target_num_preemptions_at_anchor') ==
            anchor.get('target_num_preemptions')
        and ((cancel is not None and retire_reason == cancel.get('reason')
              and isinstance(retire_reason, str)
              and retire_reason.startswith('CANCEL_'))
             or (cancel is None and retire_reason == 'FIRST_NEW_OUTPUT'
                 and observed_output is not None and target_first is not None)
             or (cancel is None and isinstance(retire_reason, str)
                 and (retire_reason.startswith('TERMINAL_')
                      or retire_reason == 'TARGET_DISAPPEARED_UNKNOWN')
                 and terminal is not None and terminal.get('reason') == retire_reason
                 and target['status'] == 'completed')))
    prepared_and_rechecked = (prepare is not None
        and prepare.get('step') == anchor['step']
        and prepare.get('target') == target_id
        and prepare.get('victim') == victim_id
        and prepare.get('saved_tokens', 0) > 0
        and job_gate is not None
        and job_gate.get('target') == target_id
        and job_gate.get('victim') == victim_id
        and job_gate.get('reason') == 'READY'
        and commit is not None and commit.get('step', -1) >= prepare['step']
        and (commit.get('disposition') != 'DIRECT_READY') == forced
        and len(commit_admissions) == 1
        and commit_admissions[0].get('target') == target_id
        and commit_admissions[0].get('native_admission') ==
            (receipt or {}).get('native_admission')
        and (not forced or (
            commit.get('free_blocks', -1) >= commit.get('required_blocks', 0)
            and victim_id in (receipt or {}).get('actual_preempted_req_ids', []))))
    if forced:
        require(mode == 'queue_fund', 'Only fund arm may force a preemption')
    return dict(
        episode_id=episode_id, mode=mode, anchor=anchor,
        anchor_time_s=anchor_s, anchor_valid=anchor_valid,
        pause_key=anchor.get('pause_key'),
        target_num_preemptions_at_anchor=anchor.get('target_num_preemptions'),
        target_source_request=target['source_request_id'],
        victim_source_request=victim['source_request_id'],
        queue_move_verified=queue_verified, event_ids_verified=linked,
        event_order_verified=ordered,
        prepare=prepare, commit_job_gate_reason=job_gate.get('reason') if job_gate else None,
        generic_commit_admissions=commit_admissions,
        commit=commit, cancel=cancel,
        retire_consistent=retire_consistent,
        prepared_and_rechecked=prepared_and_rechecked if commit else None,
        native_admission=receipt, native_admission_verified=native_receipt_verified,
        observed_new_output=observed_output, terminal=terminal, retire=retire,
        raw_first_new_output_or_terminal=first_output_or_terminal,
        target_last_to_next_output_gap_s=(target_first - target_last
            if target_first is not None and target_last is not None else None),
        anchor_to_first_new_output_s=(target_first - anchor_s
            if target_first is not None else None),
        target_recorded_running_readmission=target_readmit,
        target_first_output_after_readmission=(target_first is not None
            and target_readmit is not None and target_first > target_readmit['time_s']),
        target=target,
        actual_forced_preemption=actual_forced,
        actual_forced_preemption_record=actual[0] if actual_forced else None,
        victim_preemption_time_s=victim_preempt_s,
        victim_recorded_running_readmission=victim_readmit,
        victim_first_output_after_preemption_s=(
            victim_at_preempt['first_output_after_reference_s']
            if victim_at_preempt is not None else None),
        victim_preempt_to_readmit_s=(victim_readmit['time_s'] - victim_preempt_s
            if victim_readmit is not None else None),
        victim_later_preemptions=later_preemptions,
        victim=victim,
        target_completed=target['status'] == 'completed',
        victim_completed=victim['status'] == 'completed',
        protection_starts=protection_starts,
        protection_releases=protection_releases,
        funded_q1_protection_verified=protection_valid,
    )


def episode_chains(cell, mode):
    store = cell['store']
    require(store.get('oldest_repeat') is True, 'Repeated mode was not applied')
    indexed_events = list(enumerate(store['events']))
    grouped = defaultdict(list)
    for _, event in indexed_events:
        if event.get('event') in OLDEST_EVENTS:
            require(isinstance(event.get('episode_id'), str),
                    'Oldest event lacks episode_id')
            grouped[event['episode_id']].append(event)
    anchors = [(index, event) for index, event in indexed_events
               if event.get('event') == 'oldest_anchor']
    require(len(grouped) == len(anchors) == store.get('oldest_episode_count')
            == store.get('oldest_anchor_count'), 'Episode/anchor counters differ')
    episodes = [_episode(cell, mode, event['episode_id'],
                         grouped[event['episode_id']], indexed_events)
                for _, event in anchors]
    require(len({episode['episode_id'] for episode in episodes}) == len(episodes),
            'Repeated episode IDs')
    retired = sum(episode['retire'] is not None for episode in episodes)
    cancelled = sum(episode['cancel'] is not None for episode in episodes)
    forced = sum(episode['actual_forced_preemption'] for episode in episodes)
    require(store.get('oldest_retired_count') == retired
            and store.get('oldest_cancelled_count') == cancelled
            and store.get('oldest_forced_commits') == sum(
                bool(episode['commit'] and episode['commit'].get('forced'))
                for episode in episodes), 'Repeat outcome counters differ')
    require(store.get('oldest_native_admissions') == sum(
                episode['native_admission'] is not None for episode in episodes)
            and store.get('oldest_target_new_outputs') == sum(
                episode['observed_new_output'] is not None for episode in episodes),
            'Admission/output receipt counters differ')
    require(store.get('oldest_queue_moves') ==
            (0 if mode == 'native' else len(episodes)),
            'Queue move count differs')
    positions = {event['episode_id']: (index, next(
        i for i, row in indexed_events
        if row.get('event') == 'oldest_retire'
        and row.get('episode_id') == event['episode_id']))
        for index, event in anchors}
    no_overlap = all(positions[older['episode_id']][1]
                     < positions[newer['episode_id']][0]
                     for older, newer in zip(episodes, episodes[1:]))
    pause_keys = [(episode['anchor']['target'],
                   episode['target_num_preemptions_at_anchor'])
                  for episode in episodes]
    pause_unique = all(key[1] is not None for key in pause_keys)
    pause_unique = pause_unique and len(set(pause_keys)) == len(pause_keys)
    no_funded_events_in_controls = (mode == 'queue_fund' or all(
        episode['prepare'] is None and episode['commit'] is None
        and episode['cancel'] is None and not episode['protection_starts']
        and not episode['protection_releases'] for episode in episodes))
    no_hold_in_controls = (mode == 'queue_fund' or not any(
        event.get('protection_origin') == 'OLDEST_FUNDED'
        for event in store['events']))
    return dict(
        mode=mode, episode_count=len(episodes), retired_count=retired,
        cancelled_count=cancelled, actual_forced_count=forced,
        actual_forced_target_sources=sorted({episode['target_source_request']
            for episode in episodes if episode['actual_forced_preemption']}),
        no_overlapping_active_episodes=no_overlap,
        no_same_pause_retry=pause_unique,
        all_retire_reasons_consistent=all(episode['retire_consistent']
                                          for episode in episodes),
        no_funded_events_in_controls=no_funded_events_in_controls,
        no_hold_in_controls=no_hold_in_controls,
        gate_counts=store.get('oldest_gate_counts'),
        exhausted_pause_count=store.get('oldest_exhausted_pause_count'),
        episodes=episodes,
    )


def analyze(session):
    cells = discover(session)
    chains = {mode: episode_chains(cell, mode) for mode, cell in cells.items()}
    victims = {mode: victim_receipts(cell) for mode, cell in cells.items()}
    comparisons = {name: compare(cells[old], cells[new])
                   for name, (old, new) in PAIRS.items()}
    for name, (old, new) in PAIRS.items():
        old_requests = {r['request_id']: r for r in cells[old]['raw']['requests']}
        new_requests = {r['request_id']: r for r in cells[new]['raw']['requests']}
        comparisons[name]['output_count_difference_requests'] = sorted(
            rid for rid in old_requests if len(old_requests[rid]['output_token_ids'])
            != len(new_requests[rid]['output_token_ids']))
    require(all(len(item['full_cohort_pair']['frontier']) == 20
                and len(item['full_cohort_pair']['per_request']) == 128
                for item in comparisons.values()),
            'Full 128-request/20-point comparison incomplete')
    scores = {name: score(cells[old], cells[new])
              for name, (old, new) in PAIRS.items()}
    fund_episodes = chains['queue_fund']['episodes']
    forced_episodes = [episode for episode in fund_episodes
                       if episode['actual_forced_preemption']]
    all_episodes = [episode for mode in MODES for episode in chains[mode]['episodes']]
    all_commits = [episode for episode in fund_episodes if episode['commit'] is not None]
    all_cancels = [episode for episode in fund_episodes if episode['cancel'] is not None]
    validity = dict(
        all_128_complete=all(cell['complete'] for cell in cells.values()),
        native_full_saving_all=all(cell['store']['store_scope'] == 'native_full'
            and cell['store']['native_calc_overridden'] is False
            for cell in cells.values()),
        all_anchor_gates_and_ids_legal=all(episode['anchor_valid']
            and episode['event_ids_verified'] and episode['event_order_verified']
            and episode['queue_move_verified'] for episode in all_episodes),
        all_episodes_retired_once_without_overlap=all(
            chain['retired_count'] == chain['episode_count']
            and chain['no_overlapping_active_episodes'] for chain in chains.values()),
        all_retire_reasons_consistent=all(chain['all_retire_reasons_consistent']
            for chain in chains.values()),
        no_same_pause_retry=all(chain['no_same_pause_retry']
                                for chain in chains.values()),
        all_native_victim_choices_join_raw=all(
            item['unmatched_decisions'] == 0
            and item['raw_actual_preemptions_not_joined'] ==
                cells[mode]['store']['applied_rotations']
            for mode, item in victims.items()),
        forced_counts_join_store_and_raw=all(
            cells[mode]['status'].get('forced_rotations') ==
                cells[mode]['store'].get('applied_rotations') ==
                chains[mode]['actual_forced_count']
            for mode in MODES),
        controls_forced_zero_and_no_hold=all(
            chains[mode]['actual_forced_count'] == 0
            and chains[mode]['no_funded_events_in_controls']
            and chains[mode]['no_hold_in_controls']
            for mode in ('native', 'queue_only')),
    )
    mechanism = dict(
        fund_at_least_two_actual_preemptions=len(forced_episodes) >= 2,
        fund_at_least_two_target_sources=len({episode['target_source_request']
            for episode in forced_episodes}) >= 2,
        every_forced_commit_has_actual_raw_preemption=all(
            not episode['commit'] or not episode['commit'].get('forced')
            or episode['actual_forced_preemption'] for episode in fund_episodes),
        every_commit_prepared_and_rechecked=all(
            episode['prepared_and_rechecked'] for episode in all_commits),
        every_commit_native_admission_and_first_output_or_terminal=all(
            episode['native_admission_verified']
            and episode['raw_first_new_output_or_terminal']
            and episode['target_completed'] for episode in all_commits),
        every_commit_q1_lifecycle_legal=all(
            episode['funded_q1_protection_verified'] for episode in all_commits),
        every_cancel_has_no_commit=all(episode['commit'] is None
            and episode['retire'] is not None for episode in all_cancels),
        all_displaced_victims_complete=all(episode['victim_completed']
            for episode in forced_episodes),
    )
    fund_p95 = cells['queue_fund']['arm']['metrics']['max_gap_request_p95_s']
    service = dict(
        funded_p95_gap_strictly_below_both=(fund_p95 is not None and all(
            cells[other]['arm']['metrics']['max_gap_request_p95_s'] is not None
            and fund_p95 < cells[other]['arm']['metrics']['max_gap_request_p95_s']
            for other in ('native', 'queue_only'))),
        funded_rate_at_least_97pct_both=all(scores[name]['rate_at_least_97pct']
            for name in ('fund_vs_native', 'fund_vs_queue_only')),
        funded_mean_flow_at_most_105pct_both=all(
            scores[name]['mean_flow_at_most_105pct']
            for name in ('fund_vs_native', 'fund_vs_queue_only')),
    )
    criteria = dict(**validity, **mechanism, **service)
    return dict(
        status='COMPLETE_REPEATED_TRIPLET' if validity['all_128_complete']
            else 'INCOMPLETE_TRIPLET',
        session=str(session),
        arms={mode: dict(
            archive=cell['archive'], source_sha256=cell['source_sha256'],
            metrics=cell['arm']['metrics'],
            actual_preemption_count=cell['raw'].get('actual_preemption_count'),
            forced_rotations=cell['status'].get('forced_rotations'),
            applied_rotations=cell['store'].get('applied_rotations'),
            oldest_admission_mode=cell['store'].get('oldest_admission_mode'),
            oldest_repeat=cell['store'].get('oldest_repeat'),
            ordinary_backfill=cell['store'].get('ordinary_backfill'),
            stop_reason_counts=dict(Counter(r.get('stop_reason')
                for r in cell['raw']['requests'])),
        ) for mode, cell in cells.items()},
        pairwise_comparisons=comparisons,
        pairwise_criterion_values=scores,
        victim_actions=victims,
        repeated_episode_chains=chains,
        validity_criteria=validity,
        mechanism_criteria=mechanism,
        service_criteria=service,
        predeclared_criteria=criteria,
        all_criteria_met=all(criteria.values()),
        limitations=[
            'Three trajectories share one already-seen input; differing schedules are not identical-state counterfactuals.',
            'Episode events are decisions in one trajectory, not independent statistical replicates.',
            'All target and displaced-victim costs are included in full-cohort service metrics.',
            'Raw token times and terminal status determine output effects; wrapper output observation can miss a final token.',
            'This exploratory pilot alone does not establish stable benefit or a hard stall bound.',
        ],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'Output must be a new file')
    result = analyze(args.session)
    with args.output.open('x') as destination:
        json.dump(result, destination, indent=2, ensure_ascii=False, allow_nan=False)
        destination.write('\n')
    print(json.dumps(dict(status=result['status'],
        all_criteria_met=result['all_criteria_met'],
        validity=result['validity_criteria'],
        mechanism=result['mechanism_criteria'],
        service=result['service_criteria'])))


if __name__ == '__main__':
    main()
