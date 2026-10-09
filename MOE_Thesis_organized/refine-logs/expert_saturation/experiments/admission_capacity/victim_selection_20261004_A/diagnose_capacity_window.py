#!/usr/bin/env python3
"""Bounded, stdout-only observed capacity windows; no counterfactual rollout."""
import argparse
import gc
import json
from pathlib import Path
from pprint import pprint


FIELDS = ('index', 'request', 'request_status', 'qualified', 'prompt_tokens',
          'computed_tokens', 'output_tokens', 'max_tokens', 'held_blocks',
          'immediate_releasable_blocks', 'shared_blocks', 'release_state_error',
          'host_ready_prefix_blocks', 'host_missing_suffix_blocks',
          'pending_native_store_dependencies')
PROPOSALS = ('remaining_budget', 'max_release_shadow', 'cacheopt_cap_bucket')


def read(path):
    return json.loads(path.read_text())


def matches(request, query):
    if request == query:
        return True
    return ('article-' in request
            and request.split('article-', 1)[1].split('-', 1)[0] == query)


def time_of(row, origin):
    if row is None or not isinstance(row.get('host_perf_counter_s'), (int, float)):
        return None
    return row['host_perf_counter_s'] - origin


def diagnose(archive, steps, request):
    raw = read(archive / 'raw.json')
    store = read(archive / 'selective-store.json')
    offload_path = archive / 'offload-events.json'
    offload = read(offload_path) if offload_path.is_file() else {}
    origin = raw['measurement_origin_perf_counter_s']
    identity = raw.get('internal_to_source', {})
    outputs = raw['output_events']
    actual = [p for p in raw.get('preemption_events', [])
              if p.get('original_preemption_called') is True
              and p.get('original_preemption_returned') is True]
    decisions = [e for e in store.get('victim_decisions', []) if
                 (matches(identity.get(e.get('selected'), e.get('selected', '')), request)
                  if request else e.get('step') in steps)]
    print('\nCELL', archive.parent.name, 'selected_decisions', len(decisions))
    for event in decisions:
        rows = {r['request']: r for r in event.get('candidates', [])}
        selected = event.get('selected')
        print('\nACTUAL DECISION', event['step'], 'rule', event.get('rule'))
        proposals, visible = {}, {selected, event.get('native_tail')}
        for name in PROPOSALS:
            recorded = event.get(name)
            proposed = recorded.get('proposed_request') if isinstance(recorded, dict) else None
            visible.add(proposed)
            proposals[name] = dict(proposed_request=proposed,
                recorded_mode=recorded.get('mode') if isinstance(recorded, dict) else None,
                fallback=recorded.get('fallback') if isinstance(recorded, dict) else 'NOT_RECORDED',
                outcome=('SEE_ACTUAL_VICTIM_ONLY_NOT_POLICY_COUNTERFACTUAL' if proposed == selected
                         else 'UNKNOWN_NOT_EXECUTED'))
        pprint(dict(actual_selected=selected, native_tail=event.get('native_tail'),
            recorded_proposals=proposals,
            candidate_states=[{k: r.get(k) for k in FIELDS}
                              for r in event.get('candidates', []) if r['request'] in visible]),
            width=120, sort_dicts=False)
        joined = [p for p in actual if p.get('engine_call_index') == event['step']
                  and p.get('internal_request_id') == selected]
        if len(joined) != 1:
            print('ACTUAL_PREEMPT_JOIN UNKNOWN', 'matches', len(joined))
            continue
        pre = joined[0]
        start = pre['method_returned_s']
        rid = pre['request_id']
        alloc = next((a for a in offload.get('allocated', [])
                      if a.get('request') == selected and time_of(a, origin) is not None
                      and time_of(a, origin) > start), None)
        admission = next((a for a in store.get('residency_admissions', [])
                          if a.get('request') == selected and time_of(a, origin) is not None
                          and time_of(a, origin) > start), None)
        next_output = next((e for e in outputs if e.get('request_id') == rid
                            and e['received_s'] > start
                            and e['cumulative_tokens'] > pre['native_output_count_before']), None)
        next_preempt = next((p for p in actual if p['method_entered_s'] > start), None)
        next_same = next((p for p in actual if p['method_entered_s'] > start
                          and p.get('internal_request_id') == selected), None)
        completion = next((e for e in outputs if e.get('finished') is True
                           and e['received_s'] > start), None)
        kept = {identity.get(r['request']): r for r in rows.values()
                if r['request'] != selected and identity.get(r['request']) is not None}
        unknown_identity = sum(identity.get(r['request']) is None for r in rows.values()
                               if r['request'] != selected)
        alloc_view = None
        if alloc is not None:
            alloc_view = dict(time_s=time_of(alloc, origin),
                engine_call_index=alloc.get('engine_call_index'),
                local_computed_tokens=alloc.get('local_computed_tokens'),
                requested_external_tokens=alloc.get('requested_external_tokens'),
                held_snapshot=alloc.get('allocated_held_blocks'),
                exact_reoccupation_delta='UNKNOWN',
                free_blocks_after_allocation=alloc.get('free_blocks_after_allocation'),
                native_load_jobs=alloc.get('native_load_jobs'), acknowledgements=[])
            for job in alloc.get('native_load_jobs', []):
                acks = [a for a in offload.get('load_acknowledgements', [])
                        if a.get('job_id') == job.get('job_id') and a.get('request') == selected
                        and time_of(a, origin) is not None
                        and time_of(a, origin) >= time_of(alloc, origin)]
                alloc_view['acknowledgements'].append(dict(job_id=job.get('job_id'),
                    status='RECORDED' if acks else 'UNKNOWN_NOT_RECORDED',
                    events=[dict(time_s=time_of(a, origin),
                        acknowledged_workers=a.get('acknowledged_workers'),
                        pending_workers_before=a.get('pending_workers_before'),
                        native_job_removed=a.get('native_job_removed')) for a in acks]))
        def pre_view(p):
            return None if p is None else dict(request=p['request_id'],
                call=p['engine_call_index'], time_s=p['method_entered_s'],
                output_count=p.get('native_output_count_before'))
        pprint(dict(actual_preempt=dict(call=event['step'], entered_s=pre['method_entered_s'],
                returned_s=start, released_blocks=pre.get('actual_released_blocks'),
                free_before=pre.get('free_blocks_before_preempt'), free_after=pre.get('free_blocks_after_preempt')),
            first_allocated_callback=alloc_view,
            first_readmission=None if admission is None else dict(call=admission['step'],
                time_s=time_of(admission, origin), held_snapshot=admission.get('held_blocks'),
                output_count=admission.get('output_count'), exact_reoccupation_delta='UNKNOWN'),
            first_next_output=None if next_output is None else dict(call=next_output['engine_call_index'],
                time_s=next_output['received_s'], output_count=next_output['cumulative_tokens']),
            next_native_preempt=pre_view(next_preempt), next_same_victim_preempt=pre_view(next_same),
            next_preempt_before_readmission=(None if next_preempt is None or admission is None else
                next_preempt['method_entered_s'] < time_of(admission, origin)),
            next_completion=None if completion is None else dict(request=completion['request_id'],
                call=completion['engine_call_index'], time_s=completion['received_s'],
                in_recorded_retained_suffix=completion['request_id'] in kept)),
            width=120, sort_dicts=False)
        # An absent allocation observation is not an infinite no-reoccupation window.
        if alloc is None or completion is None:
            print('RETAINED_WINDOW UNKNOWN: first allocation or next completion not recorded')
            continue
        end = min(time_of(alloc, origin), completion['received_s'])
        latest, finished = {}, []
        for output in outputs:
            if start < output['received_s'] <= end:
                source = output['request_id']
                if source in kept:
                    latest[source] = output['cumulative_tokens']
                if output.get('finished') is True:
                    finished.append(dict(request=source, call=output['engine_call_index'],
                        in_recorded_retained_suffix=source in kept))
        deltas = [latest.get(source, row['output_tokens']) - row['output_tokens']
                  for source, row in kept.items()]
        pprint(dict(retained_window=dict(end_s=end,
            end_reason='FIRST_ALLOCATION' if time_of(alloc, origin) <= completion['received_s'] else 'NEXT_COMPLETION',
            elapsed_s=end-start, recorded_retained_requests=len(kept), unknown_identity_rows=unknown_identity,
            requests_with_output=len(latest), total_new_output_tokens=sum(deltas),
            per_request_delta_min=min(deltas) if deltas else None,
            per_request_delta_max=max(deltas) if deltas else None,
            completions=finished,
            intervening_actual_preemptions=sum(start < p['method_entered_s'] <= end for p in actual))),
            width=120, sort_dicts=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', type=Path, required=True)
    parser.add_argument('--cell', help='Optional exact cell directory name; otherwise visit cells serially.')
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--steps', help='Comma-separated actual decision steps; default 662,673,679.')
    selection.add_argument('--request', help='Actual victim source/internal ID or seven-digit article ID.')
    args = parser.parse_args()
    steps = set(map(int, (args.steps or '662,673,679').split(',')))
    cells = ([args.session / args.cell] if args.cell else sorted(args.session.glob('cell-*')))
    print('OBSERVED WINDOWS ONLY. Shadow outcomes UNKNOWN; held is total snapshot, not new allocation.')
    print('External tokens are requested, with native job ACK reported separately. Prefix state is unrecorded;')
    print('no whole-running causal claim, no addition of overlapping windows, no service-benefit prediction.')
    print('Window membership is the initial recorded suffix; subsequent preemptions are counted, not assumed absent.')
    for cell in cells:
        diagnose(cell / 'archive', steps, args.request)
        gc.collect()


if __name__ == '__main__':
    main()
