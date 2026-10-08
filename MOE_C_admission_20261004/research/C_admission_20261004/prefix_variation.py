#!/usr/bin/env python3
"""Describe the strictly pre-binding prefix of the fixed177/budget ABBA.

python3 prefix_variation.py ANALYSIS.json --output NEW.json
This is observed run variation, never a causal correction or an online policy.
"""
import argparse
import collections
import json
from pathlib import Path

from analyze import quantile
from compare_quartet import ORDERS, validate_matched_calibration
from gc_summary import gc_intervals, partition
from preoutput_coverage import FIT_KEYS
from slo_failure_breakdown import read_hashed


def load_arm(cell):
    path = Path(cell['cell'])
    raw, raw_sha = read_hashed(path/'raw.json')
    if raw_sha != cell['raw_sha256']:
        raise ValueError('Raw differs from source service analysis')
    gate, gate_sha = read_hashed(path/'admission.json')
    config, config_sha = read_hashed(path/'config.json')
    validate_matched_calibration(config)
    fixed = path.name.endswith('fixed177')
    expected = dict(admission_cap=177 if fixed else 256,
        native_running_limit=177 if fixed else 256, engine_max_num_seqs=256,
        kv_floor=0, probe_enabled=False, reservation_probe_enabled=False)
    if any(config.get(k) != v for k, v in expected.items()):
        raise ValueError('Unexpected configuration for the pre-binding prefix')
    source = dict(cell=path.name, raw_sha256=raw_sha, admission_sha256=gate_sha,
                  config_sha256=config_sha, native_guard_hit_time=None,
                  native_guard_hit_time_status='NOT_RECORDED_NOT_INFERRED')
    required_raw = ('measurement_origin_perf_counter_s', 'requests', 'output_events')
    required_gate = ('origin_perf_s', 'native_allocations', 'decisions', 'starts')
    missing = [k for k in required_raw if k not in raw] + [k for k in required_gate if k not in gate]
    if missing:
        return dict(source=source, errors=['Missing fields: '+', '.join(missing)])
    origin, gate_origin = raw['measurement_origin_perf_counter_s'], gate['origin_perf_s']
    offset = gate_origin-origin
    requests = {r['request_id']:r for r in raw['requests']}
    identity = {r[k]:r['request_id'] for r in raw['requests']
        for k in ('request_id', 'internal_request_id', 'external_request_id') if r.get(k)}
    allocations, decisions = collections.defaultdict(list), collections.defaultdict(list)
    for row in gate['native_allocations']:
        if row['actual'] is True and row['status_before'] == 'WAITING' and row['preemptions'] == 0:
            allocations[identity[row['request_id']]].append(row)
    for row in gate['decisions']:
        if row.get('native_allocation_result') is True:
            decisions[identity[row['request_id']]].append(row)
    starts = {identity[r['request_id']]:r['first_prefill_perf_s']-origin for r in gate['starts']}
    errors, first_capacity = [], None
    # Use the recorded successful active==176 decision directly, not a replay
    # of all intervening scheduler states. Check population only at this edge.
    edges = [d for rows in decisions.values() for d in rows if d.get('active') == 176]
    if edges:
        d = min(edges,key=lambda row:row['t']);rid = identity[d['request_id']]
        matches = [a for a in allocations[rid] if d['t'] <= a['t'] and
            offset+a['t'] <= starts.get(rid,float('-inf')) and all(d.get(k) == a.get(k) for k in FIT_KEYS)]
        if (len(matches) != 1 or d.get('native_fit') is not True or
                d.get('base_allowed') is not True or d['denied']):
            errors.append('Cannot align the successful active176 boundary: '+rid)
        else:
            now = offset+d['t'];stamp = offset+matches[0]['t']
            earlier = {old for old, rows in allocations.items() if any(offset+a['t'] < now for a in rows)}
            live = {old for old in earlier if requests[old]['completion_s'] is None or
                    requests[old]['completion_s'] >= now}
            if len(live) != 176 or d.get('admitted_inflight') != 176:
                errors.append('Boundary complete-active population differs: '+rid)
            first_capacity = dict(request_id=rid, allocation_external_s=stamp,
                decision_external_s=now, first_prefill_schedule_return_s=starts[rid],
                active_before=d['active'], reconstructed_active_before=len(live),
                conservative_live_after=177, free_blocks=d['free_blocks'])
    charged, first_charge_capacity = set(), None
    if not fixed:
        if 'budget_events' not in gate:
            errors.append('Budget charge/release records unavailable')
        for event in gate.get('budget_events', []):
            rid = identity[event['request_id']]
            if event['action'] == 'charge':
                charged.add(rid)
            elif event['action'] == 'release':
                charged.discard(rid)
            if len(charged) >= 177:
                first_charge_capacity = dict(request_id=rid, external_s=offset+event['t'],
                                             live_charged_requests=len(charged))
                break
    denials = [d for d in gate['decisions'] if d.get('denied') is True]
    budget_denials = [d for d in denials if d['reason'] == 'declared_budget'
        and d.get('native_fit') is True and d.get('base_allowed') is True]
    first_denial = min(budget_denials, key=lambda d:d['t']) if budget_denials else None
    first_denial = (dict(request_id=identity[first_denial['request_id']],
        external_s=offset+first_denial['t'], active=first_denial['active'],
        reason=first_denial['reason']) if first_denial else None)
    candidates = ([first_capacity['allocation_external_s']] if first_capacity else [])
    if first_denial:
        candidates.append(first_denial['external_s'])
    if first_charge_capacity:
        candidates.append(first_charge_capacity['external_s'])
    boundary = min(candidates) if candidates else None
    if boundary is None:
        errors.append('No recorded capacity or budget boundary to define the requested prefix')
    if boundary is not None and any(offset+d['t'] < boundary for d in denials):
        errors.append('Another recorded denial precedes the proposed safe boundary')
    source.update(clock_conversion=dict(gate_origin_perf_s=gate_origin,
        measurement_origin_perf_s=origin, gate_to_external_offset_s=offset,
        formula='external=(gate_origin_perf_s-measurement_origin_perf_s)+gate_record.t; '
                'prefill=first_prefill_perf_s-measurement_origin_perf_s; output received_s is already external'),
        recorded_first_successful_allocations=sum(map(len,allocations.values())),
        complete_active_check_only_at_live177_boundary=first_capacity is not None,
        first_live177_allocation=first_capacity, first_live177_charge=first_charge_capacity,
        first_native_fit_budget_denial=first_denial, safe_boundary_s=boundary,
        status='CHECKED' if not errors else 'INDETERMINATE', errors=errors)
    return dict(source=source, errors=errors, raw=raw, requests=requests, starts=starts)


def prefix(arm, cutoff):
    raw, requests = arm['raw'], arm['requests']
    origin = raw['measurement_origin_perf_counter_s']
    tokens, times, calls = collections.defaultdict(list), collections.defaultdict(list), {}
    invalid = set()
    for e in raw['output_events']:
        if not 0 <= e['received_s'] < cutoff or e['chunk_size'] == 0:
            continue
        rid = e['request_id']
        if e['chunk_size'] != len(e['new_token_ids']):
            raise ValueError('Output chunk geometry differs')
        if not e['prefix_valid']:
            invalid.add(rid)
        tokens[rid].extend(e['new_token_ids'])
        times[rid].extend([e['received_s']]*e['chunk_size'])
        idx = e['engine_call_index']
        if idx in calls and calls[idx] != e['received_s']:
            raise ValueError('Inconsistent host return time for one engine call')
        calls[idx] = e['received_s']
    intervals, gc = gc_intervals(raw)
    window_gc = dict(available=gc['available'])
    if intervals is not None:
        union = partition(intervals, origin, origin+cutoff)
        window_gc.update(union_s=union['union_s'], intersecting_events=len(union['event_indices']),
            disjoint_by_generation_start_phase=union['disjoint_by_generation_start_phase'],
            callback_cost_in_prefix=None, callback_cost_status='ONLY_WHOLE_RUN_AGGREGATES_RECORDED',
            whole_run_unpaired_start_count=gc['unpaired_start_count'],
            whole_run_unpaired_stop_count=gc['unpaired_stop_count'], whole_run_error_count=gc['error_count'])
    indices = sorted(calls)
    gaps = [calls[b]-calls[a] for a,b in zip(indices, indices[1:]) if b == a+1]
    summary = dict(cell=arm['source']['cell'], window_start_s=0, window_end_exclusive_s=cutoff,
        arrived_requests=sum(r['arrival_s'] < cutoff for r in requests.values()),
        output_tokens=sum(map(len, tokens.values())), requests_with_first_token=len(tokens),
        completed_requests=sum(r['completion_s'] is not None and r['completion_s'] < cutoff
                               for r in requests.values()),
        first_prefill_schedule_returns=sum(0 <= t < cutoff for t in arm['starts'].values()),
        output_receiving_engine_calls=len(calls), last_output_engine_call_index=max(calls, default=None),
        all_engine_call_count_in_prefix=None,
        engine_call_count_status='NO_TIMESTAMPS_FOR_CALLS_WITHOUT_OUTPUT; DISTINCT_OUTPUT_RETURNS_ONLY',
        max_consecutive_output_return_interval_s=max(gaps, default=None), gc=window_gc,
        invalid_output_prefix_request_ids=sorted(invalid))
    return summary, tokens, times, invalid


def delta_summary(rows, key):
    values = sorted(r[key] for r in rows if r[key] is not None)
    return dict(requests=len(values), mean=sum(values)/len(values) if values else None,
        p50=quantile(values,.5), p95=quantile(values,.95),
        min=min(values,default=None), max=max(values,default=None),
        candidate_lower=sum(x < 0 for x in values), candidate_higher=sum(x > 0 for x in values),
        equal=sum(x == 0 for x in values))


def compare_prefix(candidate, baseline):
    result = dict(candidate=candidate['source']['cell'], baseline=baseline['source']['cell'])
    if candidate['errors'] or baseline['errors']:
        return dict(result, status='INDETERMINATE', common_cutoff_s=None)
    cutoff = min(candidate['source']['safe_boundary_s'], baseline['source']['safe_boundary_s'])
    summaries, tokens, times, invalid = zip(*(prefix(a,cutoff) for a in (candidate,baseline)))
    rows = []
    for rid in sorted(tokens[0].keys() & tokens[1].keys()):
        a,b = tokens[0][rid],tokens[1][rid]
        mismatch = (next((i for i,(x,y) in enumerate(zip(a,b)) if x != y),None)
                    if rid not in invalid[0] | invalid[1] else None)
        starts = [arm['starts'].get(rid) for arm in (candidate,baseline)]
        rows.append(dict(request_id=rid, candidate_prefix_tokens=len(a), baseline_prefix_tokens=len(b),
            candidate_ttft_s=times[0][rid][0]-candidate['requests'][rid]['arrival_s'],
            baseline_ttft_s=times[1][rid][0]-baseline['requests'][rid]['arrival_s'],
            ttft_delta_s=times[0][rid][0]-times[1][rid][0],
            candidate_first_prefill_s=starts[0], baseline_first_prefill_s=starts[1],
            first_prefill_delta_s=starts[0]-starts[1] if None not in starts else None,
            common_token_positions=min(len(a),len(b)), first_content_mismatch_token_index=mismatch,
            content_comparison_available=rid not in invalid[0] | invalid[1]))
    mismatch_ids = [r['request_id'] for r in rows if r['first_content_mismatch_token_index'] is not None]
    result.update(status='DESCRIPTIVE_PRE_BINDING_PREFIX', common_cutoff_s=cutoff,
        candidate_prefix=summaries[0], baseline_prefix=summaries[1],
        output_token_delta=summaries[0]['output_tokens']-summaries[1]['output_tokens'],
        same_id_both_first_output_requests=len(rows), ttft_delta_s=delta_summary(rows,'ttft_delta_s'),
        first_prefill_delta_s=delta_summary(rows,'first_prefill_delta_s'),
        content_mismatch_requests=len(mismatch_ids), content_mismatch_request_ids=mismatch_ids,
        unequal_observed_prefix_length_requests=sum(r['candidate_prefix_tokens'] != r['baseline_prefix_tokens'] for r in rows),
        candidate_only_first_output_ids=sorted(tokens[0].keys()-tokens[1].keys()),
        baseline_only_first_output_ids=sorted(tokens[1].keys()-tokens[0].keys()), per_request=rows)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('analysis', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Refusing to overwrite output')
    analysis, sha = read_hashed(args.analysis)
    cells = {Path(c['cell']).name:c for c in analysis['cells']}
    order = ORDERS['declared-budget-matched']
    if set(cells) != set(order) or len({c['workload_identity_sha256'] for c in cells.values()}) != 1:
        raise ValueError('Expected the matched fixed177/budget quartet with identical workload')
    arms = {name:load_arm(cells[name]) for name in order}
    result = dict(source_analysis=str(args.analysis.resolve()), source_analysis_sha256=sha,
        arm_boundaries=[arms[n]['source'] for n in order],
        pairs=[compare_prefix(arms[order[a]],arms[order[b]]) for a,b in ((1,0),(2,3))],
        semantics='For each fixed pair use [0, min(recorded first live177 successful-allocation marker '
            'in either arm, first native-fit budget denial, first live177 charge marker)). Reconstruct '
            'previous successful first admissions minus only already observed host completions only at '
            'the first successful decision.active==176 boundary. Same-step earlier allocations count '
            'before schedule-return. This is a boundary population check, not full-trajectory replay. '
            'Allocation markers precede admitted bookkeeping, so this cutoff is conservative and is not '
            'an invented native-guard hit time or the actual first policy-divergent action. Before it '
            'running cannot reach guard177/256 and no '
            'budget denial is recorded. This does not make GPU state, batch shape, native scheduling, '
            'controller overhead, or numerical trajectories identical. Only tokens actually returned '
            'strictly inside the window enter content comparisons. Different prefix lengths alone '
            'are timing variation, not proven token-content divergence. GC unions are callback-boundary '
            'intervals, not pure GC CPU or GPU savings; no latency is subtracted. This post-hoc '
            'pre-binding diagnostic does not estimate a net policy causal effect, explain all full-run '
            'variation, provide an avoidable-benefit bound, or select future-output-aware decisions.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    for p in result['pairs']:
        print(p['candidate'],p['baseline'],p['status'],p.get('common_cutoff_s'),
              'tokens delta',p.get('output_token_delta'),'content mismatch',p.get('content_mismatch_requests'))
    print(args.output.resolve())


if __name__ == '__main__':
    main()
