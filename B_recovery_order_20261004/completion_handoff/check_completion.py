#!/usr/bin/env python3
"""Join host completion polls to the one instrumented native finalizer; reuse request analysis."""
import argparse
import bisect
import collections
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analyze import analyze, read


def check(directory):
    required = [directory/name for name in ('completion-finalize.json', 'recovery-order.json', 'timing.json')]
    rawpath = next((directory/name for name in ('raw.json', 'raw.json.gz') if (directory/name).exists()), None)
    missing = [str(path) for path in required if not path.exists()]
    if rawpath is None:
        missing.append(str(directory/'raw.json[.gz]'))
    if missing:
        return dict(status='UNAVAILABLE', missing=missing, scientific_status='NO_ACTION_CONCLUSION')
    finalizer, order, timing = (read(path) for path in required)
    raw = read(rawpath); parent = analyze(directory); origin = raw['measurement_origin_perf_counter_s']
    mode = finalizer.get('mode'); rows = finalizer.get('events', []); errors = []; contexts = []
    def require(condition, reason, iteration=None):
        if not condition:
            errors.append(dict(reason=reason, iteration=iteration))
    require(mode in ('native', 'after_sample'), 'unknown_mode')
    require(finalizer.get('instrumented_context_enters') == len(rows), 'enter_count_mismatch')
    require(finalizer.get('instrumented_native_finalizations') == len(rows), 'finalization_count_mismatch')
    iterations = [row.get('iteration') for row in rows]
    require(None not in iterations and len(set(iterations)) == len(rows), 'missing_or_duplicate_iteration')
    require(finalizer.get('extra_cuda_queries') == 0 and finalizer.get('extra_cuda_synchronizations') == 0,
            'extra_cuda_observation_declared')
    for row in rows:
        iteration = row.get('iteration'); entry = row.get('enter_perf_s'); forward = row.get('forward_exit_perf_s')
        begin = row.get('finalize_begin_perf_s'); end = row.get('finalize_end_perf_s'); book = row.get('bookkeeping_return_perf_s')
        complete = all(isinstance(value, (float, int)) for value in (entry, forward, begin, end, book))
        require(complete, 'missing_normal_forward_boundary', iteration)
        if not complete:
            continue
        require(entry <= forward <= begin <= end, 'invalid_forward_finalize_order', iteration)
        deferred = bool(row.get('deferred'))
        if deferred:
            require(mode == 'after_sample' and row.get('finalize_reason') == 'after_bookkeeping', 'invalid_deferred_mode_or_reason', iteration)
            require(forward <= book <= begin, 'deferred_finalize_before_bookkeeping', iteration)
        else:
            require(row.get('finalize_reason') == 'native_exit', 'unexpected_non_deferred_reason', iteration)
            require(end <= book, 'native_finalize_after_bookkeeping', iteration)
        if timing.get('process_start_perf_s') is not None and timing.get('process_end_perf_s') is not None:
            require(timing['process_start_perf_s'] <= entry <= end <= timing['process_end_perf_s'], 'outside_recorded_process_clock', iteration)
        contexts.append(dict(iteration=iteration, enter_s=entry-origin, forward_exit_s=forward-origin,
                             bookkeeping_return_s=book-origin, finalize_begin_s=begin-origin, finalize_end_s=end-origin,
                             deferred=deferred, reason=row.get('finalize_reason'), completed_load_ids=[], completed_store_ids=[]))
    contexts.sort(key=lambda row: row['finalize_begin_s']); starts = [row['finalize_begin_s'] for row in contexts]
    require(all(a['finalize_end_s'] <= b['finalize_begin_s'] for a,b in zip(contexts,contexts[1:])), 'overlapping_finalization_intervals')
    polls = [event for event in order.get('events', []) if event['kind'] == 'job_completed']; match = {}; counts = collections.Counter(); unmatched = []
    for event in polls:
        host = event['host_perf_s']; t = host-origin; pos = bisect.bisect_right(starts, t)-1
        context = contexts[pos] if pos >= 0 and t <= contexts[pos]['finalize_end_s'] else None
        direction = 'store' if event.get('is_store') is True else 'load' if event.get('is_store') is False else 'unknown'
        if timing.get('measurement_start_perf_s', float('inf')) <= host <= timing.get('measurement_return_perf_s', float('-inf')):
            phase = 'capture'
        elif timing.get('measurement_return_perf_s', float('inf')) < host <= timing.get('post_request_drain_end_perf_s', float('-inf')):
            phase = 'drain'
        else:
            phase = 'outside_capture_or_unknown'
        assignment = dict(iteration=context['iteration'] if context else None, deferred=context['deferred'] if context else None,
                          host_observed_s=t, phase=phase)
        if event['job_id'] in match:
            require(False, 'duplicate_job_completed_observation')
        match[event['job_id']] = assignment
        counts[(direction, 'deferred' if context and context['deferred'] else 'native' if context else 'unmapped', phase)] += 1
        if context and direction in ('load','store'):
            context['completed_'+direction+'_ids'].append(event['job_id'])
        elif not context:
            unmatched.append(dict(job_id=event['job_id'], direction=direction, **assignment))
    associations = collections.defaultdict(list)
    for recovery in parent.get('recovery_events', []):
        for jid in recovery['load_job_ids']:
            associations[jid].append(recovery)
    loads = []
    for job in parent.get('jobs', []):
        if job['is_store'] is not False:
            continue
        load = dict(job, completion_finalize=match.get(job['job_id']), recovery_associations=associations[job['job_id']])
        load['ack_to_first_scheduled_s'] = [r['first_scheduled_plan_s']-job['ack_retired_s']
            if r['first_scheduled_plan_s'] is not None and job['ack_retired_s'] is not None else None for r in load['recovery_associations']]
        loads.append(load)
    recovery_loads = [load for load in loads if load['recovery_associations']]
    result = dict(status='PASS_ACTION_CHECK' if not errors else 'ACTION_CHECK_FAILED', scientific_status='ACTION_AND_HOST_TIMING_ONLY',
        directory=str(directory), mode=mode, observer_status=finalizer.get('status'), errors=errors,
        instrumentation=dict(context_enters=finalizer.get('instrumented_context_enters'), native_finalizations=finalizer.get('instrumented_native_finalizations'),
            rows=len(rows), normal_intervals=len(contexts), deferred_contexts=sum(row['deferred'] for row in contexts),
            finalize_reasons=dict(collections.Counter(row.get('finalize_reason','missing') for row in rows))),
        completion_poll_counts=[dict(direction=k[0], finalizer=k[1], phase=k[2], jobs=n) for k,n in sorted(counts.items())],
        load_jobs=len(loads), recovery_load_jobs=len(recovery_loads),
        recovery_loads_observed_in_deferred_finalize=sum(bool((load['completion_finalize'] or {}).get('deferred')) for load in recovery_loads),
        unmapped_polls=unmatched, load_chains=loads, contexts=contexts,
        request_outcomes=dict(status=parent['status'], completed=parent['completed'], failed=parent['failed'], unfinished=parent['unfinished']),
        clocks=dict(raw_origin_perf_s=origin, finalizer=finalizer.get('clock'), transfer=order.get('clock')),
        provenance=dict(parent_analyzer_sha256=hashlib.sha256((ROOT/'analyze.py').read_bytes()).hexdigest(), timing_file=str(directory/'timing.json')),
        semantics='Finalizer intervals and job_completed are host observations on one perf_counter clock. GPU elapsed is a separate duration. A poll inside the moved finalizer proves observation placement, not earlier GPU completion or request benefit. Unmapped polls may use untouched no-forward paths and remain explicit. Request joins are reused from the parent analyzer; cross-run events are not same-state counterfactuals.')
    return result


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--cell-output', type=Path, required=True); parser.add_argument('--output', type=Path)
    args = parser.parse_args(); destination = args.output or args.cell_output.parent/'completion-action-check.json'
    if destination.exists():
        raise FileExistsError(destination)
    result = check(args.cell_output)
    with destination.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(status=result['status'], output=str(destination), errors=len(result.get('errors', [])))))
    if result['status'] != 'PASS_ACTION_CHECK':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
