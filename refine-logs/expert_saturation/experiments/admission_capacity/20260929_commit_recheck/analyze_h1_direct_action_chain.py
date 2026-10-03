"""Validate observed H1 direct admissions and subsequent output, without timing ranks.

The caller must first qualify source, model, resources, complete cohort and archive.
No direct action is a stopping result, not a reason to manufacture another input.
"""
from __future__ import annotations


def check(condition, message):
    if not condition:
        raise ValueError(message)


def analyze(raw, installed, offload):
    check(raw.get('status') == 'COMPLETE' and raw.get('error') is None,
          'Incomplete diagnostic cannot qualify the action')
    check(installed.get('commit_recheck') is True and installed.get('status') == 'DRAINED',
          'H1 on adapter was not applied or drained')
    events = installed['events']
    direct = [e for e in events if e.get('event') == 'direct_commit']
    gates = [e for e in events if e.get('event') == 'commit_recheck']
    check(installed.get('direct_commits') == len(direct), 'Direct count differs from events')
    counters = installed['native_reservation_gate']
    check(all(type(counters.get(k)) is int and counters[k] >= 0
              for k in ('checked', 'zero', 'positive_keep', 'unknown_keep')), 'Invalid reservation counters')
    check(counters['checked'] == counters['zero'] + counters['positive_keep'] + counters['unknown_keep'],
          'Reservation dispositions do not sum')
    check(counters['zero'] == len(direct), 'Successful zero-reservation decisions lack direct receipts')
    check(counters['checked'] == sum(e.get('base_reason') == 'DIRECT_READY' for e in gates),
          'Reservation probes disagree with commit events')
    if not direct:
        return {'status': 'NO_ACTION', 'direct_commits': 0, 'chains': [],
                'reservation_gate': counters,
                'next': 'Stop H1 performance expansion on this frozen H128 domain; do not retune inputs to create actions.'}
    origin = raw['measurement_origin_perf_counter_s']
    requests = {r['internal_request_id']: r for r in raw['requests']}
    steps = {s['step']: s for s in raw['scheduler_steps']}
    completed_loads = {}
    for receipt in offload['completed_jobs']:
        for job in receipt['jobs']:
            if job.get('is_store') is False:
                completed_loads[(job['job_id'], job['request'])] = receipt['time_s'] - origin
    chains = []
    for event in direct:
        step, target = event['step'], event['target']
        matches = [e for e in gates if e['step'] == step and e['target'] == target]
        check(len(matches) == 1, 'Direct action lacks unique commit-time decision')
        gate = matches[0]
        check(gate['reason'] == gate['base_reason'] == 'DIRECT_READY' and
              gate['native_inflight_reserved_blocks'] == 0, 'Direct action bypassed reservation gate')
        check(step in steps and target in requests, 'Missing native step or target identity')
        native, request = steps[step], requests[target]
        check(request['status'] == 'completed', 'Direct target did not complete')
        scheduled = [s for s in native['scheduled'] if s['internal_request_id'] == target]
        admission = event['native_admission']
        loads = event.get('load_job_ids', [])
        if admission == 'SCHEDULED_TOKENS':
            check(len(scheduled) == 1 and event['scheduled_tokens'] > 0 and
                  scheduled[0]['scheduled_tokens'] == event['scheduled_tokens'],
                  'Claimed direct compute absent from native schedule')
            ready_s = native['start_s']
        elif admission == 'ASYNC_LOAD_ADMITTED':
            check(event['scheduled_tokens'] == 0 and loads, 'Async direct lacks load admission')
            check(all((job, target) in completed_loads for job in loads), 'Direct target load did not complete')
            ready_s = max(completed_loads[(job, target)] for job in loads)
            check(ready_s >= native['start_s'], 'Load completion precedes direct admission')
        else:
            raise ValueError('Unknown native admission')
        outputs = [t for t in request['token_times_s'] if t >= ready_s]
        check(outputs, 'Admitted direct target never returned a subsequent new token')
        victim = gate['planned_victim']
        check(victim in requests, 'Planned victim identity absent from cohort')
        # The capture records preempted IDs as source request IDs. Verify both forms.
        victim_source = requests[victim]['request_id']
        check(not {victim, victim_source}.intersection(native['preempted_request_ids']),
              'Direct step also preempted the planned victim')
        chains.append({'step': step, 'target_internal_id': target,
                       'target_request_id': request['request_id'],
                       'planned_victim_request_id': victim_source,
                       'native_admission': admission, 'load_job_ids': loads,
                       'native_admission_start_s': native['start_s'],
                       'native_ready_s': ready_s, 'first_observed_output_after_ready_s': min(outputs),
                       'target_stop_reason': request.get('stop_reason'),
                       'target_completion_s': request['completion_s']})
    return {'status': 'OBSERVED_DIRECT_ACTION_CHAIN', 'direct_commits': len(direct),
            'chains': chains, 'reservation_gate': counters,
            'scope': 'Observed on-policy native admission and later output. No off/on causal speedup, peer benefit, tensor equality, or performance ranking established.'}
