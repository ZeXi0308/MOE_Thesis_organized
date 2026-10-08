#!/usr/bin/env python3
"""Host timelines for v13 KV maxgaps>1s and the largest v14 cadence maxgap.

Usage: python3 gap_timeline.py V13_ANALYSIS.json V14_ANALYSIS.json --output NEW.json
Only recorded events; allocation/overlap does not establish execution or causality.
"""
import argparse
import collections
import json
from pathlib import Path

from slo_failure_breakdown import read_hashed
from preoutput_coverage import FIT_KEYS


def allocation(row, offset):
    keys = ('status_before', 'preemptions', 'predicted', 'actual', 'fits', 'free_blocks',
        'full_required_blocks', 'chunk_required_blocks', 'native_watermark_blocks',
        'native_reserved_blocks', 'num_new_tokens', 'external_computed_tokens', 'load_kv_async')
    return dict(external_time_s=offset+row['t'], gate_relative_time_s=row['t'],
        **{k: row[k] for k in keys})


def start_counts(starts, lo, hi):
    rows = [s for s in starts if lo <= s['external_time_s'] < hi]
    return dict(evaluations=len(rows), unique_requests=len({s['request_id'] for s in rows}),
        request_ids=sorted({s['request_id'] for s in rows}),
        first=rows[0] if rows else None, last=rows[-1] if rows else None)


def first_admission(rows, allocations, starts, offset):
    succeeded = [a for a in allocations if a['actual'] and a['preemptions'] == 0 and a['status_before'] == 'WAITING']
    decisions = [r for r in rows if r.get('native_allocation_result') is True]
    if len(succeeded) != 1 or len(decisions) != 1 or len(starts) != 1:
        return dict(status='UNKNOWN', reason='No unique first successful new allocation/decision/prefill alignment')
    a, r = succeeded[0], decisions[0]
    if r.get('native_fit') is not True or r.get('base_allowed') is not True:
        return dict(status='UNKNOWN', reason='Missing proven native-fit and base allowance at first admission')
    if (not r['t'] <= a['gate_relative_time_s'] or a['external_time_s'] > starts[0]['external_time_s']
            or any(k not in r or r[k] != a[k] for k in FIT_KEYS)):
        return dict(status='UNKNOWN', reason='First allocation does not exactly match decision before first prefill')
    margin = r['free_blocks']-r['full_required_blocks']-r['native_watermark_blocks']
    local = r['num_new_tokens'] > 0 and not r['load_kv_async']
    known = r['native_reserved_blocks'] == 0 and local
    keys = ('free_blocks', 'full_required_blocks', 'chunk_required_blocks', 'active',
        'recovery_count', 'age_s', 'signal_wait_limit_bypass', 'native_reserved_blocks',
        'native_watermark_blocks', 'native_fit', 'base_allowed', 'native_allocation_result',
        'num_new_tokens', 'load_kv_async')
    return dict(status='ALIGNED', decision_external_time_s=offset+r['t'], gate_relative_time_s=r['t'],
        allocation_external_time_s=a['external_time_s'], first_prefill_external_time_s=starts[0]['external_time_s'],
        state={k:r[k] for k in keys}, full_fit_margin_blocks=margin,
        existing_headroom32_shadow=margin < 32 if known else None,
        shadow_status='CHECKED_FIRST_ADMISSION_STATE' if known else 'UNKNOWN_RESERVED_OR_NONLOCAL',
        semantics='The existing ordinary32-page shadow is margin<32 only for native-fit/base-allowed local '
            'new allocation with native_reserved_blocks=0. This is coverage at the actual first admission, '
            'not an online intervention, causal benefit, or proof a one-shot probe would select this request.')


def summarize_cell(cell, selection):
    path = Path(cell['cell'])
    raw, raw_sha = read_hashed(path/'raw.json')
    if raw_sha != cell['raw_sha256']:
        raise ValueError('Raw differs from service analysis')
    admission, admission_sha = read_hashed(path/'admission.json')
    metrics, metrics_sha = read_hashed(Path(cell['per_request_json']))
    by_id = {r['request_id']: r for r in raw['requests']}
    identity = {r[k]: r['request_id'] for r in raw['requests']
        for k in ('request_id', 'internal_request_id', 'external_request_id') if r.get(k)}
    origin = raw['measurement_origin_perf_counter_s']
    offset = admission['origin_perf_s']-origin
    if selection == 'maxgap_gt_1s':
        selected = [m for m in metrics if m['max_generation_gap_s'] > 1]
    else:
        largest = max(m['max_generation_gap_s'] for m in metrics)
        selected = [m for m in metrics if m['max_generation_gap_s'] == largest]
    selected_ids = {m['request_id'] for m in selected}
    starts = sorted((dict(request_id=identity[s['request_id']],
        external_time_s=s['first_prefill_perf_s']-origin) for s in admission['starts']),
        key=lambda s: s['external_time_s'])
    if len({s['request_id'] for s in starts}) != len(starts):
        raise ValueError('Duplicate first-prefill record')
    events, allocations, outputs, decisions = (collections.defaultdict(list) for _ in range(4))
    for decision in admission['decisions']:
        rid = identity[decision['request_id']]
        if rid in selected_ids:
            decisions[rid].append(decision)
    for e in raw['preemption_events']:
        rid = identity[e['request_id']]
        if e['original_preemption_returned'] and rid in selected_ids:
            if not e['original_preemption_called']:
                raise ValueError('Successful preemption without native call')
            events[rid].append(dict(e, request_id=rid))
    for a in admission['native_allocations']:
        rid = identity[a['request_id']]
        if rid in selected_ids:
            allocations[rid].append(allocation(a, offset))
    for o in raw['output_events']:
        rid = identity[o['request_id']]
        if rid in selected_ids and o['chunk_size'] > 0:
            outputs[rid].append({k:o[k] for k in ('engine_call_index', 'received_s',
                'cumulative_tokens', 'chunk_size', 'finished')})
    targets = []
    for metric in selected:
        rid = metric['request_id']
        token_times = by_id[rid]['token_times_s']
        index = max(range(len(token_times)-1), key=lambda i:token_times[i+1]-token_times[i])
        lo, hi = token_times[index:index+2]
        if hi-lo != metric['max_generation_gap_s']:
            raise ValueError('Raw token gap disagrees with derived per-request metric')
        attempts = [a for a in allocations[rid] if lo <= a['external_time_s'] < hi and a['preemptions'] > 0]
        success = [a for a in attempts if a['actual']]
        first_success = success[0]['external_time_s'] if success else None
        local_success = [a for a in success if a['num_new_tokens'] > 0 and not a['load_kv_async']]
        epochs = []
        for ordinal, event in enumerate(events[rid], 1):
            entered = event['method_entered_s']
            if not lo <= entered < hi:
                continue
            same = [a for a in attempts if a['preemptions'] == ordinal and a['external_time_s'] >= entered]
            okay = [a for a in same if a['actual']]
            failed = [a for a in same if not a['actual']]
            endpoint = okay[0]['external_time_s'] if okay else hi
            epochs.append(dict(successful_preemption_ordinal=ordinal, preemption=event,
                allocation_attempts=len(same), failed_allocation_attempts=len(failed),
                first_attempt=same[0] if same else None, last_failed_attempt=failed[-1] if failed else None,
                successful_allocations=okay,
                preempt_to_first_attempt_s=same[0]['external_time_s']-entered if same else None,
                preempt_to_first_success_s=endpoint-entered if okay else None,
                first_success_to_next_host_output_s=hi-endpoint if okay else None,
                first_prefill_starts_until_first_success_or_gap_end=start_counts(starts, entered, endpoint)))
        before = [o for o in outputs[rid] if o['received_s'] == lo]
        after = [o for o in outputs[rid] if o['received_s'] == hi]
        if not before or not after:
            raise ValueError('Missing host output event at raw token-gap boundary')
        progress = next((p for internal,p in admission['recovery_progress'].items()
            if identity[internal] == rid), None)
        schedule = progress.get('first_scheduled') if progress else None
        schedule_in_gap = (schedule and lo <= schedule['perf_s']-origin < hi)
        prior_starts = [s for s in starts if s['external_time_s'] < lo]
        next_starts = [s for s in starts if s['external_time_s'] >= hi]
        first_preempt = events[rid][0] if events[rid] else None
        first_token = token_times[0]
        later_preempts = [e for e in events[rid] if e['method_entered_s'] >= first_token]
        targets.append(dict(request_id=rid, gap_start_s=lo, gap_end_s=hi, gap_s=hi-lo,
            gap_start_output_count=index+1, gap_end_output_count=index+2,
            maxgap_ties=sum(b-a == hi-lo for a,b in zip(token_times,token_times[1:])),
            tie_handling='earliest maximum consecutive-token interval',
            host_output_before=before[-1], host_output_after=after[0],
            first_admission=first_admission(decisions[rid], allocations[rid],
                [s for s in starts if s['request_id'] == rid], offset),
            first_host_token_s=first_token,
            first_successful_preemption_s=first_preempt['method_entered_s'] if first_preempt else None,
            first_host_token_to_first_preemption_s=first_preempt['method_entered_s']-first_token if first_preempt else None,
            first_host_token_to_first_post_output_preemption_s=later_preempts[0]['method_entered_s']-first_token
                if later_preempts else None,
            successful_preemptions_in_gap=len(epochs), preemption_epochs=epochs,
            recovery_allocation_attempts=len(attempts), failed_recovery_allocations=sum(not a['actual'] for a in attempts),
            first_successful_recovery_allocation=success[0] if success else None,
            first_successful_local_recovery_allocation=local_success[0] if local_success else None,
            gap_start_to_first_successful_recovery_allocation_s=first_success-lo if success else None,
            first_successful_recovery_allocation_to_next_output_s=hi-first_success if success else None,
            recorded_first_schedule_return_in_gap=dict(schedule, external_time_s=schedule['perf_s']-origin)
                if schedule_in_gap else None,
            execution_observation='A per-request schedule return is only observed for the frozen first-opportunity '
                'cohort. Null is unobserved, not proof of no execution. A successful local allocation records '
                'requested local tokens, not GPU completion. The next host output is directly observed.',
            new_first_prefill_starts_during_gap=start_counts(starts, lo, hi),
            new_first_prefill_starts_before_first_success=start_counts(starts, lo, first_success) if success else None,
            new_first_prefill_starts_after_first_success=start_counts(starts, first_success, hi) if success else None,
            nearest_new_first_prefill_before_gap=prior_starts[-1] if prior_starts else None,
            nearest_new_first_prefill_at_or_after_gap=next_starts[0] if next_starts else None,
            output_tokens=metric['output_tokens'], stop_reason=metric['stop_reason'],
            ttft_s=metric['ttft_s'], flow_s=metric['flow_s']))
    union_starts = [s for s in starts if any(t['gap_start_s'] <= s['external_time_s'] < t['gap_end_s'] for t in targets)]
    return dict(cell=str(path), selection=selection, raw_sha256=raw_sha, admission_sha256=admission_sha,
        source_per_request_json=cell['per_request_json'], source_per_request_sha256=metrics_sha,
        clock_conversion=dict(gate_origin_perf_s=admission['origin_perf_s'], measurement_origin_perf_s=origin,
            gate_to_external_offset_s=offset, formula='gate_origin + allocation.t - measurement_origin; '
                'first_prefill_perf_s - measurement_origin; raw preemption/output times already external'),
        selected_requests=len(targets), actual_first_prefill_records=len(starts),
        selected_gaps_with_new_first_prefill=sum(t['new_first_prefill_starts_during_gap']['evaluations'] > 0 for t in targets),
        unique_new_first_prefill_starts_in_selected_gap_union=len(union_starts),
        new_first_prefill_starts_in_selected_gap_union=union_starts, targets=targets)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('native_cap_analysis', type=Path)
    parser.add_argument('cadence_analysis', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Refusing to overwrite output')
    native, native_sha = read_hashed(args.native_cap_analysis)
    cadence, cadence_sha = read_hashed(args.cadence_analysis)
    kv = {Path(c['cell']).name:c for c in native['cells'] if Path(c['cell']).name in ('probe-01-kv256', 'probe-02-kv256')}
    if len(kv) != 2 or len(cadence['cells']) != 1 or Path(cadence['cells'][0]['cell']).name != 'probe-00-baseline':
        raise ValueError('Expected v13 KV01/KV02 plus the one v14 baseline')
    result = dict(source_analyses=[dict(path=str(args.native_cap_analysis.resolve()), sha256=native_sha),
            dict(path=str(args.cadence_analysis.resolve()), sha256=cadence_sha)],
        cells=[summarize_cell(kv[name], 'maxgap_gt_1s') for name in sorted(kv)] +
            [summarize_cell(cadence['cells'][0], 'largest_maxgap')],
        semantics='Select every v13 KV request with max generation gap>1s and v14 largest-gap request(s). '
            'Use consecutive host token receipts, not TTFT. Report actual native preempt method success, '
            'actual recovery allocation results, and genuinely first-prefill schedule-return records. '
            'Recovery allocations cover the instrumented waiting-loop calls, not every running-request '
            'execution. Failed allocations do not reveal the intervening queue or transfer state; no '
            'snapshots fill the gaps. Preemption ordinals count successful calls from the run start and '
            'match native allocation preemptions. A later local allocation with fewer external tokens '
            'does not by itself identify why the available prefix changed. All boundaries are host '
            'observations, not GPU/DMA completion. Concurrent intervals across targets are not additive '
            'stall costs or avoidable benefit. No overlap or absence is an intervention outcome; the '
            'cadence case is descriptive and not a cross-trace policy comparison.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    for c in result['cells']:
        print(Path(c['cell']).name, 'selected', c['selected_requests'], 'with_new_starts',
            c['selected_gaps_with_new_first_prefill'], 'unique_new_starts', c['unique_new_first_prefill_starts_in_selected_gap_union'])
    print(args.output.resolve())


if __name__ == '__main__':
    main()
