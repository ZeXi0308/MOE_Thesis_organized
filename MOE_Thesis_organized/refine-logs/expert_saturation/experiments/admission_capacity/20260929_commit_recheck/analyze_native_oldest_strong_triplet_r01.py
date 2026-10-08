#!/usr/bin/env python3
"""Seen-input challenge: native, ordinary free-fit backfill, repeated oldest fund.

All service quantities use the full 128-request cohort. Funded episodes and
ordinary admissions are descriptive actions within three independent paths.
"""

import argparse
from collections import Counter
import json
from pathlib import Path

from analyze_native_backfill_only_pair_r01 import read_cell
from analyze_native_oldest_admission_triplet_r01 import read_oldest_cell
from analyze_native_oldest_repeat_triplet_r01 import episode_chains
from analyze_native_residency_victim_triplet_r01 import score, victim_receipts
from analyze_protection_yield_triplet_r01 import compare, require
from analyze_waiter_backfill_triplet_r01 import ordinary_actions


MODES = ('native', 'ordinary', 'queue_fund')
PAIRS = {
    'ordinary_vs_native': ('native', 'ordinary'),
    'fund_vs_native': ('native', 'queue_fund'),
    'fund_vs_ordinary': ('ordinary', 'queue_fund'),
}


def discover(session):
    directories = sorted(session.glob('cell-[0-9][0-9]-*'))
    require(len(directories) == 3, 'Expected exactly three serial cells')
    cells = {}
    for directory in directories:
        index = int(directory.name[5:7])
        name = directory.name[8:]
        config = json.loads((directory / 'archive' / 'config.json').read_text())
        mode = config.get('oldest_admission_mode')
        if config.get('ordinary_backfill') is True:
            require(mode is None, 'Ordinary baseline also enabled oldest admission')
            cell = read_cell(session, index, name, ordinary=True)
            mode = 'ordinary'
        else:
            require(mode in ('native', 'queue_fund'), 'Unknown oldest strong arm')
            cell = read_oldest_cell(directory)
            require(cell['config'].get('oldest_repeat') is True
                    and cell['store'].get('oldest_repeat') is True,
                    'Repeated policy not applied to native/fund arm')
        require(mode not in cells, 'Duplicate strong-baseline arm')
        cells[mode] = cell
    require(set(cells) == set(MODES), 'Strong-baseline triplet incomplete')
    require(len({cell['config'].get('workload_sha256')
                 for cell in cells.values()}) == 1,
            'Strong-baseline workloads differ')
    ordinary = cells['ordinary']
    require(ordinary['config'].get('oldest_admission_mode') is None
            and ordinary['store'].get('oldest_admission_mode') is None,
            'Ordinary baseline has oldest policy')
    require(ordinary['config'].get('oldest_admission_cli_requested') == 'ordinary'
            and ordinary['config'].get('oldest_repeat') is False
            and ordinary['store'].get('oldest_repeat') is False
            and ordinary['store'].get('oldest_episode_count') == 0,
            'Ordinary baseline requested/effective mode differs')
    require(ordinary['config'].get('allow_forced_rotations') is False
            and ordinary['store'].get('allow_forced_rotations') is False,
            'Ordinary baseline allowed forced rotation')
    require(ordinary['config'].get('native_victim_rule') ==
            ordinary['store'].get('native_victim_rule') == 'tail'
            and ordinary['config'].get('native_victim_full_running') is False
            and ordinary['store'].get('native_victim_full_running_enabled') is False
            and ordinary['store'].get('current_victim_guard_enabled') is False
            and ordinary['store'].get('self_preempt_continue_enabled') is False
            and ordinary['config'].get('capacity_deferral_mode') ==
                ordinary['store'].get('capacity_deferral_mode') == 'off',
            'Ordinary baseline changed unrelated native policy')
    require(not any(event.get('event', '').startswith('oldest_')
                    for event in ordinary['store'].get('events', [])),
            'Ordinary baseline contains oldest episode events')
    return cells


def _fund_mechanism(cells, episodes, victims):
    fund = episodes['queue_fund']['episodes']
    all_episodes = [item for mode in ('native', 'queue_fund')
                    for item in episodes[mode]['episodes']]
    forced = [item for item in fund if item['actual_forced_preemption']]
    commits = [item for item in fund if item['commit'] is not None]
    cancels = [item for item in fund if item['cancel'] is not None]
    return dict(
        native_and_fund_episode_gates_legal=all(item['anchor_valid']
            and item['event_ids_verified'] and item['event_order_verified']
            and item['queue_move_verified'] for item in all_episodes),
        native_and_fund_retirement_consistent=all(
            episodes[mode]['retired_count'] == episodes[mode]['episode_count']
            and episodes[mode]['no_overlapping_active_episodes']
            and episodes[mode]['all_retire_reasons_consistent']
            and episodes[mode]['no_same_pause_retry']
            for mode in ('native', 'queue_fund')),
        all_native_victim_choices_join_raw=all(
            result['unmatched_decisions'] == 0
            and result['raw_actual_preemptions_not_joined'] ==
                cells[mode]['store']['applied_rotations']
            for mode, result in victims.items()),
        all_forced_counts_join_store_and_raw=all(
            cells[mode]['status'].get('forced_rotations') ==
                cells[mode]['store'].get('applied_rotations') ==
                (episodes[mode]['actual_forced_count']
                 if mode != 'ordinary' else 0)
            for mode in MODES),
        native_and_ordinary_forced_zero=(
            episodes['native']['actual_forced_count'] == 0
            and cells['ordinary']['store'].get('applied_rotations') == 0
            and cells['ordinary']['status'].get('forced_rotations') == 0),
        native_no_funded_hold=(episodes['native']['no_hold_in_controls']
                               and episodes['native']['no_funded_events_in_controls']),
        fund_at_least_two_actual_preemptions=len(forced) >= 2,
        fund_at_least_two_target_sources=len({item['target_source_request']
            for item in forced}) >= 2,
        every_forced_commit_has_actual_raw_preemption=all(
            not item['commit'] or not item['commit'].get('forced')
            or item['actual_forced_preemption'] for item in fund),
        every_commit_prepared_and_rechecked=all(
            item['prepared_and_rechecked'] for item in commits),
        every_commit_native_admission_and_first_output_or_terminal=all(
            item['native_admission_verified']
            and item['raw_first_new_output_or_terminal']
            and item['target_completed'] for item in commits),
        every_commit_q1_lifecycle_legal=all(
            item['funded_q1_protection_verified'] for item in commits),
        every_cancel_retired_without_commit=all(
            item['commit'] is None and item['retire'] is not None
            for item in cancels),
        every_displaced_victim_completed=all(
            item['victim_completed'] for item in forced),
    )


def analyze(session):
    cells = discover(session)
    episodes = {mode: episode_chains(cells[mode], mode)
                for mode in ('native', 'queue_fund')}
    ordinary = ordinary_actions(cells['ordinary'])
    ordinary.update(
        gate_counts=cells['ordinary']['store'].get('ordinary_backfill_gate_counts'),
        store_choice_count=cells['ordinary']['store'].get('ordinary_backfill_choices'),
        store_admission_count=cells['ordinary']['store'].get('ordinary_backfill_admissions'))
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
    actual_free_fit = [row for row in ordinary['actions']
        if row['choice'].get('free_blocks', -1) >=
                row['choice'].get('required_blocks', float('inf'))
        and row['actual_admission_output_completion_chain']]
    validity = dict(
        all_128_complete=all(cell['complete'] for cell in cells.values()),
        all_native_full_saving=all(
            cell['store'].get('store_scope') == 'native_full'
            and cell['store'].get('native_calc_overridden') is False
            for cell in cells.values()),
        ordinary_choice_and_admission_counters_match=(
            ordinary['store_choice_count'] == ordinary['choices']
            and ordinary['store_admission_count'] == ordinary['native_admissions']),
        ordinary_has_actual_free_fit_admission=len(actual_free_fit) >= 1,
    )
    mechanism = _fund_mechanism(cells, episodes, victims)
    fund_p95 = cells['queue_fund']['arm']['metrics']['max_gap_request_p95_s']
    service = dict(
        fund_p95_gap_strictly_below_native_and_ordinary=(
            fund_p95 is not None and all(
                cells[other]['arm']['metrics']['max_gap_request_p95_s'] is not None
                and fund_p95 < cells[other]['arm']['metrics']['max_gap_request_p95_s']
                for other in ('native', 'ordinary'))),
        fund_rate_at_least_97pct_both=all(
            scores[name]['rate_at_least_97pct']
            for name in ('fund_vs_native', 'fund_vs_ordinary')),
        fund_mean_flow_at_most_105pct_both=all(
            scores[name]['mean_flow_at_most_105pct']
            for name in ('fund_vs_native', 'fund_vs_ordinary')),
    )
    criteria = dict(**validity, **mechanism, **service)
    return dict(
        status='COMPLETE_STRONG_TRIPLET' if validity['all_128_complete']
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
        ordinary_actions=ordinary,
        ordinary_actual_free_fit_admission_count=len(actual_free_fit),
        repeated_episode_chains=episodes,
        validity_criteria=validity,
        mechanism_criteria=mechanism,
        service_criteria=service,
        predeclared_criteria=criteria,
        all_criteria_met=all(criteria.values()),
        limitations=[
            'All three paths use the same already-seen input but have independent scheduling trajectories.',
            'Ordinary free-fit backfill is a strong simple baseline action, not a complete reproduction of CacheOPT or UniBoost.',
            'Action events within one run are not independent replicates; all displaced-request costs remain in the full-cohort metrics.',
            'This challenge is neither fresh confirmation nor a novelty claim.',
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
