#!/usr/bin/env python3
"""One-shot oldest paused admission: native, queue move, and funded queue move.

All service comparisons use the complete 128-request cohort. An anchor is an
online selection receipt; raw output and preemption records establish effects.
"""
import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
import json
from pathlib import Path

from analyze_native_backfill_only_pair_r01 import sha256
from analyze_native_residency_victim_triplet_r01 import score, victim_receipts
from analyze_protection_yield_triplet_r01 import compare, require
from evaluate_goodput import summarize


MODES = ('native', 'queue_only', 'queue_fund')
PAIRS = {'queue_only_vs_native': ('native', 'queue_only'),
         'fund_vs_native': ('native', 'queue_fund'),
         'fund_vs_queue_only': ('queue_only', 'queue_fund')}


def read_oldest_cell(directory):
    archive = directory / 'archive'
    paths = {key: archive / (key + '.json') for key in
             ('raw', 'config', 'status', 'selective-store')}
    raw, config, status, store = [json.loads(paths[key].read_text()) for key in paths]
    mode = config.get('oldest_admission_mode')
    require(mode in MODES and store.get('oldest_admission_mode') == mode,
            'Oldest admission requested/applied mode differs')
    metrics = summarize(raw, 128, 180)
    requests = raw.get('requests', [])
    require(len(requests) == 128 and len({r['request_id'] for r in requests}) == 128,
            'Expected 128 unique source requests')
    complete = (raw.get('status') == status.get('status') == 'COMPLETE'
                and raw.get('error') is None and status.get('error') is None
                and status.get('capture_status') == 'COMPLETE'
                and status.get('requests_completed') == metrics['completed'] == 128
                and metrics['failed'] == metrics['unfinished'] == 0)
    require(store.get('status') == 'DRAINED', 'Native-full adapter did not drain')
    require(config.get('store_scope') == store.get('store_scope') == 'native_full'
            and store.get('native_calc_overridden') is False,
            'Native-full saving differs')
    require(config.get('ordinary_backfill') is False
            and store.get('ordinary_backfill') is False
            and store.get('ordinary_backfill_choices') == 0
            and store.get('ordinary_backfill_admissions') == 0,
            'Legacy ordinary backfill was active')
    require(config.get('allow_forced_rotations') is (mode == 'queue_fund')
            and store.get('allow_forced_rotations') is (mode == 'queue_fund'),
            'Fund-only forced-rotation permission differs')
    require(config.get('native_victim_rule') == store.get('native_victim_rule') == 'tail'
            and config.get('native_victim_full_running') is False
            and store.get('native_victim_full_running_enabled') is False
            and store.get('current_victim_guard_enabled') is False
            and store.get('self_preempt_continue_enabled') is False
            and config.get('capacity_deferral_mode') == store.get('capacity_deferral_mode') == 'off',
            'Unrelated native victim/deferral policy differs')
    require(config.get('spare_followup') is False and store.get('spare_followup') is False,
            'Spare follow-up active')
    require(store.get('oldest_output_age_threshold_s') == 1.0,
            'One-second online trigger differs')
    require(isinstance(store.get('events'), list)
            and isinstance(store.get('residency_admissions'), list),
            'One-shot or running-admission receipts missing')
    return dict(archive=str(archive), raw=raw, config=config, status=status,
                store=store, arm=dict(metrics=metrics), complete=complete,
                source_sha256={key: sha256(path) for key, path in paths.items()})


def discover(session):
    directories = sorted(session.glob('cell-[0-9][0-9]-*'))
    require(len(directories) == 3, 'Expected exactly three serial cells')
    cells = {}
    for directory in directories:
        cell = read_oldest_cell(directory)
        mode = cell['store']['oldest_admission_mode']
        require(mode not in cells, 'Duplicate oldest-admission arm')
        cells[mode] = cell
    require(set(cells) == set(MODES), 'Three oldest-admission modes incomplete')
    require(len({cell['config'].get('workload_sha256') for cell in cells.values()}) == 1,
            'Arm workload identities differ')
    return cells


def _one(events, name):
    rows = [row for row in events if row.get('event') == name]
    require(len(rows) <= 1, 'More than one ' + name + ' receipt')
    return rows[0] if rows else None


def _request_metrics(cell, source):
    rows = {r['request_id']: r for r in cell['arm']['metrics']['requests']}
    return rows.get(source)


def action_chain(cell, mode):
    raw, store = cell['raw'], cell['store']
    events = store['events']
    anchor = _one(events, 'oldest_anchor')
    prepare = _one(events, 'oldest_prepare')
    commit = _one(events, 'oldest_commit')
    cancel = _one(events, 'oldest_cancel')
    native_admission = _one(events, 'oldest_native_admission')
    observed_output = _one(events, 'oldest_target_new_output')
    protections = [event for event in events if event.get('event') == 'protection_start']
    oldest_protections = [event for event in protections
                          if event.get('protection_origin') == 'OLDEST_FUNDED']
    protection_releases = [event for event in events
        if event.get('event') == 'protection_release']
    require(store.get('oldest_anchor_count') == int(anchor is not None)
            and store.get('oldest_native_admissions') == int(native_admission is not None)
            and store.get('oldest_target_new_outputs') == int(observed_output is not None),
            'One-shot receipt counters differ')
    require(not (prepare or commit or cancel or oldest_protections) or mode == 'queue_fund',
            'Nonfunded arm has funded action')
    require(len(oldest_protections) <= 1, 'More than one funded protection')
    require(store.get('oldest_forced_commits') == sum(
        bool(event.get('forced')) for event in [commit] if event),
        'Funded commit counter differs')
    require(store.get('oldest_queue_moves') == int(anchor is not None
                                                  and mode != 'native'),
            'One-shot queue move count differs')
    if anchor is None:
        require(not (prepare or commit or cancel or native_admission or observed_output),
                'One-shot events exist without an anchor')
        return dict(mode=mode, anchor=None, selected=False,
                    gate_counts=store.get('oldest_gate_counts'),
                    actual_forced_preemption=False, target_readmitted=False,
                    target_next_output=False, target_completed=False,
                    victim_completed=False)

    by_internal = {r['internal_request_id']: r for r in raw['requests']}
    target_id, victim_id = anchor['target'], anchor['victim']
    target, victim = by_internal.get(target_id), by_internal.get(victim_id)
    require(target is not None and victim is not None and target_id != victim_id,
            'Anchor target/victim absent or identical')
    origin = raw['measurement_origin_perf_counter_s']
    anchor_s = anchor['host_perf_counter_s'] - origin
    last_observed_s = anchor['target_last_output_observed_perf_s'] - origin
    target_times = target['token_times_s']
    prior_index = bisect_right(target_times, anchor_s)
    last_actual_s = target_times[prior_index - 1] if prior_index else None
    next_actual_s = target_times[prior_index] if prior_index < len(target_times) else None
    previous_target_preemptions = [event for event in raw.get('preemption_events', [])
        if event.get('internal_request_id') == target_id
        and event.get('original_preemption_returned') is True
        and event.get('method_entered_s') is not None
        and event['method_entered_s'] <= anchor_s]
    valid_anchor = (anchor.get('mode') == mode and anchor.get('trigger_threshold_s') == 1.0
        and anchor.get('target_output_count', 0) > 0
        and anchor.get('target_observed_output_age_s', 0) >= 1.0
        and abs(anchor_s - last_observed_s
                - anchor['target_observed_output_age_s']) < 0.001
        and type(anchor.get('target_waiting_index_before')) is int
        and anchor['target_waiting_index_before'] >= 0
        and type(anchor.get('free_blocks')) is int
        and type(anchor.get('target_full_history_need_blocks')) is int
        and type(anchor.get('victim_held_blocks')) is int
        and anchor['free_blocks'] < anchor['target_full_history_need_blocks']
        and anchor['free_blocks'] + anchor['victim_held_blocks']
            >= anchor['target_full_history_need_blocks']
        and anchor.get('deficit_blocks') == anchor['target_full_history_need_blocks']
            - anchor['free_blocks']
        and anchor.get('native_inflight_reserved_blocks') == 0
        and len(previous_target_preemptions) > 0
        and prior_index == anchor['target_output_count'])
    queue_move = (anchor['head_after'] == target_id
                  and store['oldest_queue_moves'] == 1) if mode != 'native' else (
                      anchor['head_after'] == anchor['head_before']
                      and store['oldest_queue_moves'] == 0)
    actual = [event for event in raw.get('preemption_events', [])
        if commit is not None and event.get('engine_call_index') == commit.get('step')
        and event.get('internal_request_id') == victim_id
        and event.get('original_preemption_called') is True
        and event.get('original_preemption_returned') is True]
    forced = (mode == 'queue_fund' and commit is not None
              and commit.get('target') == target_id and commit.get('victim') == victim_id
              and commit.get('forced') is True and len(actual) == 1
              and store.get('applied_rotations') == 1
              and store.get('oldest_forced_commits') == 1)
    native_receipt = (native_admission is not None
        and native_admission.get('target') == target_id
        and native_admission.get('native_admission') in
            ('SCHEDULED_TOKENS', 'ASYNC_LOAD_ADMITTED')
        and native_admission.get('step', -1) >= anchor['step'])
    admissions = [row for row in store['residency_admissions']
                  if row.get('request') == target_id
                  and row.get('step', -1) >= anchor['step']
                  and row.get('num_preemptions', 0) > 0]
    target_readmit = admissions[0] if admissions else None
    target_readmit_s = (target_readmit['host_perf_counter_s'] - origin
                       if target_readmit else None)
    target_next_after_readmit = (next_actual_s is not None
        and target_readmit_s is not None and next_actual_s > target_readmit_s)
    victim_preempt_s = actual[0]['method_entered_s'] if actual else None
    victim_admissions = [row for row in store['residency_admissions']
                         if row.get('request') == victim_id
                         and victim_preempt_s is not None
                         and row.get('host_perf_counter_s', 0) - origin > victim_preempt_s]
    victim_readmit = victim_admissions[0] if victim_admissions else None
    victim_readmit_s = (victim_readmit['host_perf_counter_s'] - origin
                       if victim_readmit else None)
    victim_next_s = (next((time for time in victim['token_times_s']
                          if victim_preempt_s is not None and time > victim_preempt_s), None))
    victim_later_preemptions = [event for event in raw.get('preemption_events', [])
        if victim_preempt_s is not None and event.get('internal_request_id') == victim_id
        and event.get('original_preemption_returned') is True
        and event.get('method_entered_s', -1) > victim_preempt_s]
    return dict(
        mode=mode, selected=True, anchor=anchor,
        anchor_time_s=anchor_s, anchor_valid=valid_anchor,
        target_source_request=target['request_id'],
        victim_source_request=victim['request_id'],
        queue_move_verified=queue_move,
        prepare=prepare, commit=commit, cancel=cancel,
        protection_starts=len(protections),
        funded_protection_starts=len(oldest_protections),
        funded_q1_protection_verified=(len(oldest_protections) == 1
            and oldest_protections[0].get('request') == target_id
            and oldest_protections[0].get('recovery_min_outputs') == 1),
        protection_releases=protection_releases,
        oldest_native_admission=native_admission,
        native_admission_verified=native_receipt,
        oldest_target_new_output_observation=observed_output,
        raw_target_prior_output_count=prior_index,
        target_last_actual_output_before_anchor_s=last_actual_s,
        target_first_actual_output_after_anchor_s=next_actual_s,
        target_last_to_next_output_gap_s=(next_actual_s - last_actual_s
                                           if last_actual_s is not None and next_actual_s is not None else None),
        target_anchor_to_next_output_s=(next_actual_s - anchor_s
                                        if next_actual_s is not None else None),
        target_prior_actual_preemption=(previous_target_preemptions[-1]
                                        if previous_target_preemptions else None),
        target_recorded_running_readmission=target_readmit,
        target_readmission_time_s=target_readmit_s,
        target_next_output_after_readmission=target_next_after_readmit,
        target_completed=target.get('status') == 'completed',
        target_stop_reason=target.get('stop_reason'),
        target_final_output_count=len(target['output_token_ids']),
        target_request_metrics=_request_metrics(cell, target['request_id']),
        actual_forced_preemption=forced,
        victim_actual_preemption=(actual[0] if actual else None),
        victim_recorded_running_readmission=victim_readmit,
        victim_readmission_time_s=victim_readmit_s,
        victim_next_actual_output_after_preemption_s=victim_next_s,
        victim_preempt_to_readmit_s=(victim_readmit_s - victim_preempt_s
                                     if victim_readmit_s is not None else None),
        victim_preempt_to_next_output_s=(victim_next_s - victim_preempt_s
                                         if victim_next_s is not None else None),
        victim_later_preemption_steps=[e['engine_call_index']
                                      for e in victim_later_preemptions],
        victim_completed=victim.get('status') == 'completed',
        victim_stop_reason=victim.get('stop_reason'),
        victim_request_metrics=_request_metrics(cell, victim['request_id']),
        gate_counts=store.get('oldest_gate_counts'),
    )


def analyze(session):
    cells = discover(session)
    victims = {mode: victim_receipts(cell) for mode, cell in cells.items()}
    pairs = {name: compare(cells[old], cells[new])
             for name, (old, new) in PAIRS.items()}
    for name, (old, new) in PAIRS.items():
        old_requests = {r['request_id']: r for r in cells[old]['raw']['requests']}
        new_requests = {r['request_id']: r for r in cells[new]['raw']['requests']}
        pairs[name]['output_count_difference_requests'] = sorted(
            rid for rid in old_requests if len(old_requests[rid]['output_token_ids'])
            != len(new_requests[rid]['output_token_ids']))
    scores = {name: score(cells[old], cells[new])
              for name, (old, new) in PAIRS.items()}
    require(all(len(pair['full_cohort_pair']['frontier']) == 20
                and len(pair['full_cohort_pair']['per_request']) == 128
                for pair in pairs.values()),
            'Full 128-request/20-point comparison incomplete')
    chains = {mode: action_chain(cell, mode) for mode, cell in cells.items()}
    anchor_sources = [chains[mode].get('target_source_request') for mode in MODES]
    paired = all(anchor_sources) and len(set(anchor_sources)) == 1
    target_comparison = dict(status='PAIRED_SAME_SOURCE' if paired else 'UNPAIRED',
        source_request=anchor_sources[0] if paired else None,
        anchor_source_requests=dict(zip(MODES, anchor_sources)),
        last_output_to_first_postanchor_output_gap_s={mode: chains[mode].get(
            'target_last_to_next_output_gap_s') for mode in MODES},
        anchor_to_first_postanchor_output_delay_s={mode: chains[mode].get(
            'target_anchor_to_next_output_s') for mode in MODES})
    def strictly_better(field):
        values=[chains[mode].get(field) for mode in MODES]
        return bool(paired and all(value is not None for value in values)
                    and values[2] < values[0] and values[2] < values[1])
    target_comparison['fund_gap_strictly_below_both'] = strictly_better(
        'target_last_to_next_output_gap_s')
    target_comparison['fund_postanchor_delay_strictly_below_both'] = strictly_better(
        'target_anchor_to_next_output_s')
    fund = chains['queue_fund']
    validity = dict(
        all_128_complete=all(cell['complete'] for cell in cells.values()),
        all_native_full_saving=all(cell['store']['store_scope'] == 'native_full'
            and cell['store']['native_calc_overridden'] is False for cell in cells.values()),
        all_one_legal_anchor=all(chains[mode]['selected']
            and chains[mode]['anchor_valid'] for mode in MODES),
        all_native_victim_decisions_match_raw=all(
            item['unmatched_decisions'] == 0
            and item['raw_actual_preemptions_not_joined'] ==
                cells[mode]['store']['applied_rotations']
            for mode, item in victims.items()),
        forced_counts_match=all(cells[mode]['status'].get('forced_rotations')
            == cells[mode]['store'].get('applied_rotations')
            == (1 if mode == 'queue_fund' and fund.get('actual_forced_preemption') else 0)
            for mode in MODES),
    )
    mechanism = dict(
        native_shadow_only=(chains['native']['selected']
            and chains['native']['queue_move_verified']
            and chains['native']['protection_starts'] == 0),
        queue_only_moved_no_hold=(chains['queue_only']['selected']
            and chains['queue_only']['queue_move_verified']
            and chains['queue_only']['protection_starts'] == 0),
        fund_selected_same_target=(fund['selected'] and fund['queue_move_verified']
            and fund.get('prepare') is not None and fund.get('commit') is not None
            and fund['prepare'].get('target') == fund['anchor']['target']
            and fund['commit'].get('target') == fund['anchor']['target']),
        fund_actual_native_preemption=fund['actual_forced_preemption'],
        fund_q1_protection=fund.get('funded_q1_protection_verified') is True,
        fund_recorded_readmission_and_new_output=(
            fund.get('native_admission_verified') is True
            and fund.get('target_recorded_running_readmission') is not None
            and fund.get('target_next_output_after_readmission') is True),
        fund_target_and_victim_completed=(fund.get('target_completed') is True
            and fund.get('victim_completed') is True),
    )
    service = dict(
        same_source_anchor_all=bool(paired),
        funded_target_gap_strictly_lower_both=target_comparison[
            'fund_gap_strictly_below_both'],
        funded_target_postanchor_delay_strictly_lower_both=target_comparison[
            'fund_postanchor_delay_strictly_below_both'],
        funded_rate_at_least_97pct_both=all(scores[name]['rate_at_least_97pct']
            for name in ('fund_vs_native', 'fund_vs_queue_only')),
        funded_mean_flow_at_most_105pct_both=all(
            scores[name]['mean_flow_at_most_105pct']
            for name in ('fund_vs_native', 'fund_vs_queue_only')),
    )
    criteria = dict(**validity, **mechanism, **service)
    peer_sources = {source for source in (fund.get('target_source_request'),
                                          fund.get('victim_source_request')) if source}
    peer_costs = {source: {mode: _request_metrics(cells[mode], source)
                           for mode in MODES} for source in peer_sources}
    return dict(
        status=('INCOMPLETE_TRIPLET' if not validity['all_128_complete'] else
                'NO_ANCHOR' if not all(chains[mode]['selected'] for mode in MODES) else
                'UNPAIRED' if not paired else 'COMPLETE_ONE_SHOT_TRIPLET'),
        session=str(session),
        arms={mode: dict(archive=cell['archive'],source_sha256=cell['source_sha256'],
            metrics=cell['arm']['metrics'],
            actual_preemption_count=cell['raw'].get('actual_preemption_count'),
            forced_rotations=cell['status'].get('forced_rotations'),
            applied_rotations=cell['store'].get('applied_rotations'),
            oldest_admission_mode=cell['store'].get('oldest_admission_mode'),
            ordinary_backfill=cell['store'].get('ordinary_backfill'),
            stop_reason_counts=dict(Counter(r.get('stop_reason')
                for r in cell['raw']['requests']))) for mode, cell in cells.items()},
        pairwise_comparisons=pairs,pairwise_criterion_values=scores,
        victim_actions=victims,one_shot_chains=chains,
        target_comparison=target_comparison,peer_request_costs=peer_costs,
        validity_criteria=validity,mechanism_criteria=mechanism,
        service_criteria=service,predeclared_criteria=criteria,
        all_criteria_met=all(criteria.values()),
        limitations=[
            'Three independent trajectories use the same already-seen input; matching source anchors do not imply identical physical pre-state or a same-state causal effect.',
            'One action per arm establishes only mechanism and descriptive service headroom, not repeated-policy or statistically stable performance.',
            'The funded action includes queue order, a real victim preemption, native restoration, and Q1 protection. All target and displaced-victim costs remain in the full-cohort metrics.',
            'The wrapper may miss a first post-anchor output at terminal removal; raw token times, not its auxiliary observation count, determine the output chain.',
            'Global maximum gap and 20-point goodput are reported, but are not one-action success criteria.',
        ],
    )


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    require(not args.output.exists(),'Output must be a new file')
    result=analyze(args.session)
    with args.output.open('x') as destination:
        json.dump(result,destination,indent=2,ensure_ascii=False,allow_nan=False)
        destination.write('\n')
    print(json.dumps(dict(status=result['status'],
        all_criteria_met=result['all_criteria_met'],
        validity=result['validity_criteria'],
        mechanism=result['mechanism_criteria'],service=result['service_criteria'])))


if __name__=='__main__':
    main()
