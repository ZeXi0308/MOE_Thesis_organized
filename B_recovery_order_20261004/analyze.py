#!/usr/bin/env python3
"""B recovery ordering: host request metrics and narrow job joins; no DMA summation claim."""
import argparse, bisect, collections, gzip, json
from pathlib import Path


def read(path):
    with (gzip.open(path, 'rt') if path.suffix == '.gz' else path.open()) as f:
        return json.load(f)


def summary(values):
    known = sorted(x for x in values if x is not None)
    def quantile(q):
        if not known:
            return None
        pos = (len(known)-1)*q; lo = int(pos); hi = min(lo+1, len(known)-1)
        return known[lo] + (known[hi]-known[lo])*(pos-lo)
    return dict(n=len(known), missing=len(values)-len(known), mean=sum(known)/len(known) if known else None,
                p50=quantile(.5), p95=quantile(.95), maximum=max(known) if known else None)


def analyze(directory):
    rawpath = next((directory/n for n in ('raw.json', 'raw.json.gz') if (directory/n).exists()), None)
    if rawpath is None:
        return dict(status='RAW_UNAVAILABLE', directory=str(directory), error='No request metrics; cell retained')
    raw = read(rawpath); origin = raw['measurement_origin_perf_counter_s']; end = raw['observation_end_s']
    mapping = raw.get('internal_to_source', {}); source = lambda rid: mapping.get(rid, rid)
    observer = read(directory/'recovery-order.json') if (directory/'recovery-order.json').exists() else {}
    events = observer.get('events', []); rows = []; times = {}; outputs = 0
    for request in raw['requests']:
        rid = request['request_id']; ts = request.get('token_times_s', [])
        times[rid] = ts; complete = request['status'] == 'completed'; arrival = request['arrival_s']
        ttft = ts[0]-arrival if ts else None; gaps = [b-a for a,b in zip(ts, ts[1:])]
        gap = max(gaps, default=0.) if ts else None
        flow = request.get('completion_s')-arrival if complete and request.get('completion_s') is not None else None
        lower_gap = max(gaps + ([max(0., end-ts[-1])] if ts and not complete else [0.])) if ts else None
        passed = complete and ttft is not None and ttft <= 5. and gap <= .2
        rows.append(dict(request=rid, status=request['status'], ttft_s=ttft, flow_s=flow, maxgap_s=gap,
                         ttft_lower_bound_s=ttft if ts else max(0., end-arrival),
                         flow_lower_bound_s=flow if flow is not None else max(0., end-arrival),
                         maxgap_lower_bound_s=lower_gap, outputs=len(request.get('output_token_ids', ts)), slo_pass=passed))
        outputs += rows[-1]['outputs']
    jobs = collections.defaultdict(list); by_request = collections.defaultdict(list)
    for event in events:
        if event.get('job_id') is not None:
            jobs[event['job_id']].append(event)
        if event.get('request') is not None:
            by_request[source(event['request'])].append(event)
    job_rows = []
    for jid, history in jobs.items():
        stages = {e['kind']: e for e in history}; first = history[0]
        row = dict(job_id=jid, request=source(next((e['request'] for e in history if e.get('request')), None)),
                   is_store=next((e['is_store'] for e in history if 'is_store' in e), None))
        for kind in ('ready', 'submit_begin', 'submit_end', 'job_completed', 'ack_retired'):
            row[kind+'_s'] = stages[kind]['host_perf_s']-origin if kind in stages else None
        for label, a, b in [('ready_to_submit_s','ready','submit_begin'), ('submit_to_host_done_s','submit_begin','job_completed'), ('host_done_to_ack_s','job_completed','ack_retired')]:
            row[label] = stages[b]['host_perf_s']-stages[a]['host_perf_s'] if a in stages and b in stages else None
        row.update(bytes=stages.get('job_completed', {}).get('bytes'), gpu_elapsed_s=stages.get('job_completed', {}).get('gpu_elapsed_s'))
        job_rows.append(row)
    recoveries = []
    for event in events:
        if event['kind'] != 'preempt':
            continue
        rid = source(event.get('request')); start = event['host_perf_s']-origin; ts = times.get(rid, [])
        at = bisect.bisect_right(ts, start); nxt = ts[at] if at < len(ts) else None; stop = nxt if nxt is not None else end
        related = [e for e in by_request[rid] if start <= e['host_perf_s']-origin <= stop]
        loads = [j for j in job_rows if j['request'] == rid and j['is_store'] is False and j['ready_s'] is not None and start <= j['ready_s'] <= stop]
        scheduled = [e['host_perf_s']-origin for e in related if e['kind']=='scheduled' and e.get('scheduled_tokens', 1)>0]
        recoveries.append(dict(request=rid, demand_host_s=start, next_output_s=nxt,
                               demand_to_next_output_s=nxt-start if nxt is not None else None,
                               censored_wait_lower_bound_s=stop-start, first_scheduled_plan_s=min(scheduled) if scheduled else None,
                               load_job_ids=[j['job_id'] for j in loads], repeated_preempts_before_output=sum(e['kind']=='preempt' for e in related)-1))
        recoveries[-1]['request_host_boundaries'] = {kind: [e['host_perf_s']-origin for e in related if e['kind']==kind]
            for kind in ('logical_free_after_preempt', 'capacity_wait', 'allocation_ok', 'lookup', 'scheduled')}
        recoveries[-1]['scheduled_to_next_output_s'] = nxt-min(scheduled) if nxt is not None and scheduled else None
    waits = []; pending_waits = []
    for event in events:
        if event['kind'] == 'wait_begin':
            pending_waits.append(event)
        elif event['kind'] == 'wait_end' and pending_waits:
            begin = pending_waits.pop(); waits.append(dict(begin_s=begin['host_perf_s']-origin,
                elapsed_host_s=event['host_perf_s']-begin['host_perf_s'], required=begin.get('required', begin.get('job_ids'))))
    copy_work = {}
    for is_store, label in ((False, 'load'), (True, 'store')):
        group = [j for j in job_rows if j['is_store'] is is_store]; completed = [j for j in group if j['job_completed_s'] is not None]
        copy_work[label] = dict(jobs=len(group), completed=len(completed), bytes=sum(j['bytes'] or 0 for j in completed),
                               missing_bytes=sum(j['bytes'] is None for j in completed), gpu_elapsed_sum_s=sum(j['gpu_elapsed_s'] or 0 for j in completed),
                               missing_gpu_elapsed=sum(j['gpu_elapsed_s'] is None for j in completed))
    counts = collections.Counter(e['kind'] for e in events); reorders = [e for e in events if e['kind']=='reorder']
    states = dict(collections.Counter(r['status'] for r in rows)); duration = end-min((r['arrival_s'] for r in raw['requests']), default=0.)
    result = dict(directory=str(directory), status=raw['status'], error=raw.get('error'), planned=len(rows), statuses=states,
                  completed=states.get('completed', 0), failed=states.get('failed', 0), unfinished=sum(r['status'] not in ('completed','failed') for r in rows),
                  duration_s=duration, outputs=outputs, output_tokens_per_s=outputs/duration if duration else None,
                  throughput_rps=states.get('completed',0)/duration if duration else None,
                  joint_slo=dict(ttft_limit_s=5., maxgap_limit_s=.2, provenance='Inherited runner research thresholds, not a production SLO; completed only; 1-output gap vacuous',
                                 passes=sum(r['slo_pass'] for r in rows), denominator=len(rows), goodput_rps=sum(r['slo_pass'] for r in rows)/duration if duration else None),
                  event_counts=dict(counts), reorder_attempts=len(reorders), order_changes=sum(bool(e.get('changed')) for e in reorders),
                  observed={k:summary([r[k] for r in rows]) for k in ('ttft_s','flow_s','maxgap_s')},
                  all_request_lower_bounds={k:summary([r[k] for r in rows]) for k in ('ttft_lower_bound_s','flow_lower_bound_s','maxgap_lower_bound_s')},
                  per_request=rows, jobs=job_rows, recovery_events=recoveries, reorders=reorders,
                  waits=waits, unclosed_waits=len(pending_waits), copy_work=copy_work)
    result['timing_semantics'] = 'All stage times are host perf_counter observations relative to raw origin. job_completed is host polling, ack is scheduler observation, scheduled is a plan. gpu_elapsed_s and its sum are separate GPU copy-work durations, never absolute timestamps or exposed latency. Recovery start uses preemption, not exact capacity wait onset.'
    return result


def comparisons(cells):
    """Descriptive run contrasts; matching source IDs does not match runtime state."""
    native = [i for i, c in enumerate(cells) if Path(c['directory']).parent.name.endswith('-native')]
    result = []
    def sequences(cell):
        directory = Path(cell['directory'])
        path = next((directory/n for n in ('raw.json', 'raw.json.gz') if (directory/n).exists()), None)
        return {r['request_id']: r.get('output_token_ids') for r in read(path)['requests']} if path else {}
    for i, candidate in enumerate(cells):
        if i in native:
            continue
        j = min(native, key=lambda j: (abs(i-j), j)) if native else None
        baseline = cells[j] if j is not None else None
        contrast = dict(candidate=candidate['directory'], native=baseline['directory'] if baseline else None,
                        reference_rule='Nearest native cell in execution order, earlier on ties; preserves forward/reverse repeats',
                        status='AVAILABLE', per_request=[], metric_counts={})
        result.append(contrast)
        if baseline is None or any(c.get('status') == 'RAW_UNAVAILABLE' for c in (candidate, baseline)):
            contrast.update(status='UNAVAILABLE', reason='Candidate or native raw unavailable; cell retained')
            continue
        crows = {r['request']: r for r in candidate['per_request']}; brows = {r['request']: r for r in baseline['per_request']}
        ids = sorted(set(crows) | set(brows)); ctokens = sequences(candidate); btokens = sequences(baseline)
        for metric in ('ttft_s', 'flow_s', 'maxgap_s'):
            contrast['metric_counts'][metric] = dict(denominator=len(ids), better=0, worse=0, equal=0, unavailable=0)
        token_counts = dict(denominator=len(ids), equal=0, different=0, unavailable=0)
        for rid in ids:
            c = crows.get(rid, {}); b = brows.get(rid, {})
            row = dict(request=rid, candidate_status=c.get('status', 'missing'), native_status=b.get('status', 'missing'))
            for metric, counts in contrast['metric_counts'].items():
                cv = c.get(metric); bv = b.get(metric); delta = cv-bv if cv is not None and bv is not None else None
                row[metric+'_delta'] = delta
                # A partial maxgap is only a lower bound, not an observed final maximum.
                comparable = delta is not None and (metric != 'maxgap_s' or c.get('status') == b.get('status') == 'completed')
                counts['unavailable' if not comparable else 'better' if delta < 0 else 'worse' if delta > 0 else 'equal'] += 1
            a = ctokens.get(rid); b = btokens.get(rid)
            row['output_sequence'] = 'unavailable' if a is None or b is None else 'equal' if a == b else 'different'
            token_counts[row['output_sequence']] += 1; contrast['per_request'].append(row)
        contrast['output_sequences'] = token_counts
        contrast['aggregate_delta_candidate_minus_native'] = {
            key: candidate[key]-baseline[key] if candidate.get(key) is not None and baseline.get(key) is not None else None
            for key in ('completed', 'failed', 'unfinished', 'outputs', 'throughput_rps', 'output_tokens_per_s')}
        contrast['aggregate_delta_candidate_minus_native']['joint_goodput_rps'] = candidate['joint_slo']['goodput_rps']-baseline['joint_slo']['goodput_rps'] if candidate['joint_slo']['goodput_rps'] is not None and baseline['joint_slo']['goodput_rps'] is not None else None
        contrast['observed_summary_delta'] = {metric: {key: candidate['observed'][metric][key]-baseline['observed'][metric][key]
            if candidate['observed'][metric][key] is not None and baseline['observed'][metric][key] is not None else None
            for key in ('mean', 'p95')} for metric in ('ttft_s', 'flow_s', 'maxgap_s')}
        contrast['semantics'] = 'All request IDs retained. Negative latency delta is faster; unavailable is not a win. Partial maxgap cannot establish improvement. Observed summaries have missing counts in cells; failures/unfinished remain in statuses and SLO denominators. Output differences include partial sequences and do not establish quality. Run-level descriptive differences are not same-state causal effects.'
    return result


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--session', type=Path, required=True); parser.add_argument('--output', type=Path)
    args = parser.parse_args(); destination = args.output or args.session/'metrics.json'
    if destination.exists():
        raise FileExistsError(destination)
    directories = sorted(p/'output' for p in args.session.glob('cell-*') if p.is_dir())
    if not directories:
        directories = [args.session/'output' if (args.session/'output').is_dir() else args.session]
    result = dict(cells=[analyze(p) for p in directories], comparison_semantics='Run-level comparisons only; events across runs are not matched-state counterfactuals. All cells retained, missing raw explicit.')
    result['comparisons'] = comparisons(result['cells'])
    with destination.open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(destination)


if __name__ == '__main__':
    main()
