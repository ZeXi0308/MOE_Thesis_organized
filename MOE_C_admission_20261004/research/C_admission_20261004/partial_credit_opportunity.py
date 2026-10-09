"""Bounded existing-state diagnosis of credit during v19 asynchronous recovery.

Not a replay of a different policy. Resident old requests are optimistically
assumed to progress every execution round; the actual runtime would still have
to check their scheduled-token and eligibility conditions. Nonrunning requests
receive their full original declaration, with no release credit.
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
    assert raw['status'] == 'COMPLETE'
    offset = gate['origin_perf_s']-raw['measurement_origin_perf_counter_s']
    outputs, charges = raw['output_events'], gate['budget_events']
    bounds = {r['request_id']: (r['prompt_tokens'], r['max_output_tokens'])
              for r in raw['requests']}
    starts = {e['request_id']: e['first_prefill_perf_s'] for e in gate['starts']}
    returns = sorted({e['engine_call_index']: e['received_s'] for e in outputs}.items())
    clocks = [t for _, t in returns]
    transitions = [(a['t'], 'allocate', a) for a in gate['native_allocations'] if a['actual']]
    for event in raw['preemption_events']:
        assert event['original_preemption_called'] and event['original_preemption_returned']
        transitions.append((event['method_returned_s']-offset, 'preempt', event))
    transitions.sort(key=lambda item: item[0])
    active, nonrunning, output = {}, {}, {}
    charge_cursor = output_cursor = transition_cursor = 0
    counts, rows = Counter(), []
    for index, decision in enumerate(gate['decisions']):
        if decision.get('mc_guard_reason') != 'admitted_not_all_running':
            continue
        counts['recorded_nonrunning_refusals'] += 1
        t = decision['t']
        pos = bisect_left(clocks, t+offset)
        if not (0 < pos < len(returns) and returns[pos][0] == returns[pos-1][0]+1):
            counts['unresolved_call'] += 1
            continue
        call = returns[pos][0]
        while output_cursor < len(outputs) and outputs[output_cursor]['engine_call_index'] < call:
            event = outputs[output_cursor]
            output[event['request_id']] = event['cumulative_tokens']
            output_cursor += 1
        while charge_cursor < len(charges) and charges[charge_cursor]['t'] <= t:
            event = charges[charge_cursor]
            if event['action'] == 'charge':
                active[event['request_id']] = event['budget_required_blocks']
            else:
                active.pop(event['request_id'])
            charge_cursor += 1
        while transition_cursor < len(transitions) and transitions[transition_cursor][0] <= t:
            when, kind, event = transitions[transition_cursor]
            rid = event['internal_request_id'] if kind == 'preempt' else event['request_id']
            if kind == 'preempt':
                nonrunning[rid] = dict(allocated=0, preempt_t=when, allocations=[])
            elif event['load_kv_async']:
                # The preceding pinned preemption frees old GPU pages. Only in
                # this bounded lifecycle do newly allocated pages equal residency.
                assert rid in nonrunning, 'Async allocation without recorded preemption'
                nonrunning[rid]['allocated'] += event['chunk_required_blocks']
                nonrunning[rid]['allocations'].append(dict(t=when,
                    blocks=event['chunk_required_blocks'], free_before=event['free_blocks']))
            else:
                nonrunning.pop(rid, None)
            transition_cursor += 1
        assert len(active) == decision['active']
        assert sum(active.values()) == decision['budget_before_blocks']
        inactive = {rid: value for rid, value in nonrunning.items() if rid in active}
        membership_match = len(active)-len(inactive) == decision['running']
        resident = []
        for rid in active.keys()-inactive.keys():
            source = raw['internal_to_source'][rid]
            prompt, maximum = bounds[source]
            prior = output.get(source, 0)
            assert 0 <= prior < maximum
            resident.append(dict(n=prompt+prior, r=maximum-prior))
        physical_resident = sum((old['n']+15)//16 for old in resident)
        physical_inactive = sum(old['allocated'] for old in inactive.values())
        physical_match = (physical_resident+physical_inactive+decision['free_blocks']
                          == gate['budget_blocks'])
        platform = sum(active[rid] for rid in inactive)
        envelope = peak(resident, decision['declared_prompt_tokens'], decision['declared_max_tokens'])
        combined_peak = envelope['peak_blocks']+platform
        new_prompt_pages = (decision['declared_prompt_tokens']+15)//16
        h0_room = decision['free_blocks']-new_prompt_pages-(platform-physical_inactive)
        native = bool(decision['native_fit'] and decision['baseline_allowed']
                      and decision['token_budget'] > 0)
        ceiling = decision['budget_after_if_admitted_blocks'] <= 39138
        peak_fit = combined_peak <= gate['budget_blocks']
        qualified = native and ceiling and membership_match and physical_match
        counts['membership_count_match'] += membership_match
        counts['h0_physical_match'] += physical_match
        counts['native_fit_with_token_budget'] += native
        counts['within_common_ceiling'] += ceiling
        counts['h0_with_platform_fits'] += qualified and h0_room >= 0
        counts['optimistic_partial_peak_fits'] += qualified and peak_fit
        rows.append(dict(decision_index=index, call=call, gate_external_s=t+offset,
            request_id=raw['internal_to_source'][decision['request_id']],
            observed_later_prefill_wait_s=starts[decision['request_id']]-(gate['origin_perf_s']+t),
            active=len(active), running=decision['running'], nonrunning=inactive,
            membership_count_match=membership_match, physical_match=physical_match,
            resident_pages=physical_resident, nonrunning_allocated_pages=physical_inactive,
            recorded_free_blocks=decision['free_blocks'], nonrunning_platform_pages=platform,
            new_prompt_pages=new_prompt_pages, h0_headroom_with_platform=h0_room,
            native_fit_with_budget=native, within_common_ceiling=ceiling,
            optimistic_peak_blocks=combined_peak, peak_h=envelope['peak_h'],
            optimistic_peak_excess_blocks=combined_peak-gate['budget_blocks'],
            optimistic_peak_fits=peak_fit, qualified=qualified))
    counts['unique_refused_requests'] = len({r['request_id'] for r in rows})
    counts['unique_optimistic_peak_fit_requests'] = len({r['request_id'] for r in rows
        if r['qualified'] and r['optimistic_peak_fits']})
    return dict(cell=str(path), counts=dict(counts), rows=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Preserved output must not exist.')
    root = Path(__file__).resolve().parent
    peak = load_function((root/'mc_budget.py').read_text())
    arms = [inspect_arm(root/'runs/westd-20261008/mc-budget-ablation-r02'/cell, peak)
            for cell in ('probe-00-mcstatic', 'probe-03-mcstatic')]
    report = dict(new_gpu_runs=0, new_online_actions=0, common_ceiling=39138,
        evidence='Existing static-policy states, not another policy trajectory or service benefit.',
        assumptions='No unrecorded invalid-load clearing or rollback; preemption frees old pages. '
            'Charge/release and successful allocations reconstruct membership; only prior returned '
            'outputs and original declared bounds determine n/r. No future natural EOS is used.',
        limitation='Physical totals and counts do not prove per-request scheduled-token eligibility. '
            'All remaining resident requests optimistically progress every execution round. This '
            'is a particular conservative-platform envelope, not a global admission upper bound. '
            'Observed later first-prefill is descriptive only and never enters the envelope; '
            'it is not a counterfactual TTFT/flow saving or a bound on downstream effects.',
        arms=arms)
    with args.output.open('x') as handle:
        json.dump(report, handle, indent=2)
        handle.write('\n')
    print(json.dumps([dict(cell=a['cell'], **a['counts']) for a in arms]))


if __name__ == '__main__':
    main()
