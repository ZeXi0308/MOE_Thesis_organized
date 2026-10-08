#!/usr/bin/env python3
"""Describe first-allocation signal coverage in the two reservation-ABBA baselines.

Usage: python3 preoutput_coverage.py ANALYSIS.json --output NEW.json
This is a host-timeline diagnostic, not an intervention or causal savings bound.
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path

from analyze import finite, request_rows


FIT_KEYS = ('fits', 'free_blocks', 'full_required_blocks', 'chunk_required_blocks',
    'native_watermark_blocks', 'native_reserved_blocks', 'num_new_tokens',
    'external_computed_tokens', 'load_kv_async')
SIGNALS = ('full_commit_opportunity', 'positive_known_prefill_unallocated', 'positive_pending_recovery')


def analyze_baseline(cell, admission, admission_sha):
    path = Path(cell['cell'])
    payload = (path / 'raw.json').read_bytes()
    if hashlib.sha256(payload).hexdigest() != cell['raw_sha256']:
        raise ValueError('raw.json differs from input analysis: ' + str(path))
    raw = json.loads(payload)
    if 'preemption_events' not in raw:
        raise ValueError('Sparse native preemption records are required: ' + str(path))
    requests = {r['request_id']: r for r in request_rows(raw)}
    identity = {r[k]: r['request_id'] for r in raw['requests']
        for k in ('request_id', 'internal_request_id', 'external_request_id') if r.get(k)}
    origin, gate_origin = raw['measurement_origin_perf_counter_s'], admission['origin_perf_s']
    starts, decisions, allocations = (collections.defaultdict(list) for _ in range(3))
    for row in admission['starts']:
        starts[identity[row['request_id']]].append(row['first_prefill_perf_s']-origin)
    for row in admission['decisions']:
        if row.get('native_allocation_result') is True:
            decisions[identity[row['request_id']]].append(row)
    for row in admission['native_allocations']:
        if row['actual'] is True:
            allocations[identity[row['request_id']]].append(row)
    aligned, unknown = {}, {}
    for rid in requests:
        if len(decisions[rid]) != 1 or len(starts[rid]) != 1:
            unknown[rid] = 'Expected one successful new-request hook and one first-prefill record'
            continue
        row = decisions[rid][0]
        matches = [a for a in allocations[rid] if row['t'] <= a['t'] and
            gate_origin+a['t']-origin <= starts[rid][0] and a['status_before'] == 'WAITING' and
            a['preemptions'] == 0 and all(a[k] == row[k] for k in FIT_KEYS)]
        if len(matches) != 1 or min(allocations[rid], key=lambda a: a['t']) is not matches[0] or (
                row['denied'] or row['base_allowed'] is not True or row['native_fit'] is not True):
            unknown[rid] = 'No unique first successful native allocation matching the allowed hook before prefill'
            continue
        aligned[rid] = (row, matches[0])
    actual, events, unclassified = [], collections.defaultdict(list), []
    for source in raw['preemption_events']:
        if type(source.get('original_preemption_returned')) is not bool:
            raise ValueError('Missing native preemption success flag')
        if not source['original_preemption_returned']:
            continue
        rid = identity[source['request_id']]
        if source['original_preemption_called'] is not True or not all(finite(source.get(k)) for k in
                ('method_entered_s', 'method_returned_s')):
            raise ValueError('Invalid successful native preemption record')
        event = dict(source, request_id=rid)
        actual.append(event)
        first = requests[rid]['first_token_s']
        if len(starts[rid]) != 1 or not finite(first):
            unclassified.append(dict(event, unknown_reason='Missing unique prefill or first host token'))
        elif starts[rid][0] <= event['method_entered_s'] < first:
            events[rid].append(dict(event, after_first_prefill_s=event['method_entered_s']-starts[rid][0],
                before_first_host_token_s=first-event['method_entered_s']))
    targets = []
    for rid in sorted(events):
        row, allocation = aligned.get(rid, (None, None))
        signal_values = dict.fromkeys(SIGNALS)
        if row is not None:
            signal_values.update(full_commit_opportunity=row.get('full_commit_opportunity'),
                positive_known_prefill_unallocated=row['known_prefill_unallocated_blocks'] > 0
                    if 'known_prefill_unallocated_blocks' in row else None,
                positive_pending_recovery=row['recovery_count'] > 0 if 'recovery_count' in row else None)
        kept = dict(row, request_id=rid) if row is not None else None
        if kept is not None:
            keep_keys = ('request_id', 't', *FIT_KEYS, 'active', 'running', 'recovery_count',
                'known_prefill_unallocated_blocks', 'known_recovery_unallocated_blocks',
                'known_nonrecovery_unallocated_blocks', 'full_commit_opportunity',
                'chunk_commit_opportunity', 'base_allowed', 'native_fit', 'native_allocation_result',
                'denied', 'reason')
            kept = {k: kept[k] for k in keep_keys if k in kept}
            kept['external_time_s'] = gate_origin+row['t']-origin
        targets.append(dict(request_id=rid, preoutput_preemption_events=len(events[rid]),
            allocation_alignment='MATCHED' if row is not None else 'UNKNOWN',
            allocation_unknown_reason=unknown.get(rid), first_allocation_decision=kept,
            first_native_allocation=dict(external_time_s=gate_origin+allocation['t']-origin,
                **{k: allocation[k] for k in ('t', 'status_before', 'preemptions', 'predicted', 'actual')})
                if allocation is not None else None,
            first_prefill_schedule_return_s=starts[rid][0], first_token_host_s=requests[rid]['first_token_s'],
            first_prefill_to_first_host_token_s=requests[rid]['first_token_s']-starts[rid][0],
            arrival_s=requests[rid]['arrival_s'], ttft_s=requests[rid]['ttft_s'], flow_s=requests[rid]['flow_s'],
            signals_at_first_allocation=signal_values, preemptions=events[rid]))
    coverage = {name: {unit: {label: sum(weight(r) for r in targets
        if r['signals_at_first_allocation'][name] is value) for label, value in
        (('true', True), ('false', False), ('unknown', None))} for unit, weight in
        (('requests', lambda r: 1), ('events', lambda r: r['preoutput_preemption_events']))} for name in SIGNALS}

    def metrics(ids):
        result = dict(requests=len(ids))
        for key in ('flow_s', 'ttft_s'):
            values = [requests[rid][key] for rid in ids if finite(requests[rid][key])]
            result[key] = dict(observed_requests=len(values), missing_requests=len(ids)-len(values),
                sum=sum(values), mean=sum(values)/len(values) if values else None)
        return result

    cohort, population = metrics(events), metrics(requests)
    return dict(cell=cell['cell'], raw_sha256=cell['raw_sha256'], admission_sha256=admission_sha,
        planned_requests=cell['planned_requests'], outcomes=cell['outcomes'],
        all_successful_preemptions=len(actual), unclassified_successful_preemptions=unclassified,
        preoutput_after_firstprefill_requests=len(events),
        preoutput_after_firstprefill_events=sum(map(len, events.values())),
        native_output_zero_events=sum(e['native_output_count_before'] == 0 for es in events.values() for e in es),
        population_allocation_alignment=dict(matched=len(aligned), unknown=len(unknown), unknown_requests=unknown),
        coverage=coverage, events_per_affected_request=dict(collections.Counter(map(len, events.values()))),
        cohort=cohort, all_population=population,
        cohort_observed_time_sum_fraction={key: cohort[key]['sum']/population[key]['sum']
            if population[key]['sum'] else None for key in ('flow_s', 'ttft_s')}, requests=targets)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('analysis', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Refusing to overwrite an existing output.')
    source = json.loads(args.analysis.read_text())
    loaded = []
    for cell in source['cells']:
        payload = (Path(cell['cell']) / 'admission.json').read_bytes()
        report = json.loads(payload)
        if report['probe_enabled'] is not False or type(report['reservation_probe']['probe_enabled']) is not bool:
            raise ValueError('Expected reservation-probe schema with the older probe disabled')
        loaded.append((cell, report, hashlib.sha256(payload).hexdigest()))
    if [r['reservation_probe']['probe_enabled'] for _, r, _ in loaded] != [False, True, True, False]:
        parser.error('Supply the four reservation ABBA cells in execution order; only the two baselines are analyzed.')
    cells = [analyze_baseline(*item) for item in (loaded[0], loaded[3])]
    result = dict(schema_version=1, source_analysis=str(args.analysis.resolve()), independent_unit='run',
        selection='Both disabled reservation-probe baselines from the fixed ABBA order; candidates excluded.',
        alignment_semantics='A successful-new decision is the same row mutated by after_allocate. Require one '
            'such row, one first-prefill record, and a unique first native success with identical fit fields '
            'between that decision and first prefill. Failures to align remain explicit unknowns.',
        timing_semantics='First prefill is host schedule return. Count successful original preempt-method calls '
            'whose entry is >= first prefill and < first host-received token. Native output counts are retained '
            'separately. These host boundaries are not GPU or DMA completion times.',
        inference='Coverage and whole-lifetime flow/TTFT sums describe the observed cohort, including its prior '
            'queueing and generation work. They are not preemption-caused costs, removable savings, a causal '
            'upper bound, or evidence that another admission action improves service.', cells=cells)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write('\n')
    for c in cells:
        print(Path(c['cell']).name, 'requests/events=', c['preoutput_after_firstprefill_requests'],
            c['preoutput_after_firstprefill_events'], 'R coverage=',
            c['coverage']['positive_known_prefill_unallocated'], 'unknown allocation=',
            c['population_allocation_alignment']['unknown'])
    print(args.output.resolve())


if __name__ == '__main__':
    main()
