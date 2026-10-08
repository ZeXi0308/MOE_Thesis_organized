"""Inspect existing MC phase-guard refusals without replaying a new policy.

Inputs use declared bounds, recorded charges and outputs returned before each
decision call. H0 rows are inferred from the pinned synchronous scheduler,
not directly recorded per-request scheduled-token snapshots.
"""
import argparse
from bisect import bisect_left
from collections import Counter
import json
from pathlib import Path

from peak_cost_probe import load_function


def inspect_arm(path, peak):
    gate = json.loads((path/'admission.json').read_text())
    raw = json.loads((path/'raw.json').read_text())
    assert raw['status'] == 'COMPLETE' and raw['actual_preemption_count'] == 0
    events, charges = raw['output_events'], gate['budget_events']
    bounds = {r['request_id']: (r['prompt_tokens'], r['max_output_tokens']) for r in raw['requests']}
    times = sorted({e['engine_call_index']: e['received_s'] for e in events}.items())
    clocks = [t for _, t in times]
    offset = gate['origin_perf_s']-raw['measurement_origin_perf_counter_s']
    counts, rows, active, output = Counter(), [], {}, {}
    event_cursor = charge_cursor = 0
    previous_borrow = {}
    for index, decision in enumerate(gate['decisions']):
        t = decision['t']+offset
        pos = bisect_left(clocks, t)
        call = (times[pos][0] if 0 < pos < len(times)
                and times[pos][0] == times[pos-1][0]+1 else None)
        if decision.get('changed_by_mc_budget') and decision.get('native_allocation_result'):
            previous_borrow[call] = index
        if decision.get('mc_guard_reason') != 'not_complete_prompt_decode':
            continue
        counts['recorded_phase_refusals'] += 1
        if call is None:
            counts['unresolved_call'] += 1
            continue
        while event_cursor < len(events) and events[event_cursor]['engine_call_index'] < call:
            event = events[event_cursor]
            output[event['request_id']] = event['cumulative_tokens']
            event_cursor += 1
        while charge_cursor < len(charges) and charges[charge_cursor]['t'] <= decision['t']:
            charge = charges[charge_cursor]
            if charge['action'] == 'charge':
                active[charge['request_id']] = charge['budget_required_blocks']
            else:
                active.pop(charge['request_id'])
            charge_cursor += 1
        assert len(active) == decision['active']
        assert sum(active.values()) == decision['budget_before_blocks']
        old_rows, without_prior_output = [], 0
        for rid in active:
            source_id = raw['internal_to_source'][rid]
            prompt, maximum = bounds[source_id]
            prior = output.get(source_id, 0)
            assert prior < maximum
            without_prior_output += prior == 0
            old_rows.append(dict(n=prompt+prior, r=maximum-prior))
        physical = sum((row['n']+15)//16 for row in old_rows)
        physical_match = physical+decision['free_blocks'] == gate['budget_blocks']
        native = bool(decision['native_fit'] and decision['baseline_allowed']
                      and decision['token_budget'] > 0)
        ceiling = decision['budget_after_if_admitted_blocks'] <= 39138
        envelope = peak(old_rows, decision['declared_prompt_tokens'], decision['declared_max_tokens'])
        feasible = envelope['peak_blocks'] <= gate['budget_blocks']
        counts['native_fit_with_token_budget'] += native
        counts['native_fit_and_common_ceiling'] += native and ceiling
        counts['h0_physical_match'] += physical_match
        counts['same_call_earlier_borrow'] += call in previous_borrow
        counts['inferred_feasible_with_common_ceiling'] += native and ceiling and physical_match and feasible
        rows.append(dict(decision_index=index, request_id=raw['internal_to_source'][decision['request_id']],
            call=call, gate_external_s=t, previous_return_s=times[pos-1][1], next_return_s=times[pos][1],
            earlier_borrow_in_call=previous_borrow.get(call), active=len(active),
            old_without_prior_output=without_prior_output, token_budget=decision['token_budget'],
            native_fit_with_budget=native, within_common_ceiling=ceiling,
            inferred_h0_physical_pages=physical, recorded_free_blocks=decision['free_blocks'],
            h0_physical_match=physical_match, inferred_peak_blocks=envelope['peak_blocks'],
            inferred_peak_h=envelope['peak_h'], inferred_peak_feasible=feasible))
    counts['unique_recorded_refused_requests'] = len({r['request_id'] for r in rows})
    counts['unique_inferred_feasible_requests'] = len({r['request_id'] for r in rows
        if r['native_fit_with_budget'] and r['within_common_ceiling']
        and r['h0_physical_match'] and r['inferred_peak_feasible']})
    return dict(cell=str(path), counts=dict(counts), rows=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Preserved output must not exist.')
    root = Path(__file__).resolve().parent
    peak = load_function((root/'mc_budget.py').read_text())
    arms = [inspect_arm(root/'runs/westd-20261008'/group/cell, peak)
        for group, cells in (
            ('mc-budget-r01', ('probe-01-mcbudget', 'probe-02-mcbudget')),
            ('mc-budget-ablation-r02', ('probe-01-mccapped', 'probe-02-mccapped')))
        for cell in cells]
    result = dict(new_gpu_runs=0, new_online_actions=0,
        role='Existing-state diagnosis, not counterfactual service performance.',
        inference='With zero preemptions, synchronous FCFS, no long-prefill chunk threshold and '
            'positive token budget at the waiting gate, scheduled preceding prefills have fit '
            'their remaining prompt. Candidate runtime must explicitly check computed+scheduled '
            'equals known tokens; matching physical pages alone does not prove that condition.',
        inputs='Known prompt/max declarations, charges and outputs strictly before the decision call; '
            'no realized future output length or post-decision outputs are used to calculate the envelope.',
        common_ceiling=39138, arms=arms)
    with args.output.open('x') as handle:
        json.dump(result, handle, indent=2)
        handle.write('\n')
    print(json.dumps([dict(cell=a['cell'], **a['counts']) for a in arms]))


if __name__ == '__main__':
    main()
