#!/usr/bin/env python3
"""Past-client-output age versus arrival order in the recorded legal fit set.

Offline decision diagnostic, not a replay or benefit prediction. Both suggestions
select before applying the original repeat8 episode/budget guard. In unique8
cells the guard history still comes only from that cell's actual past actions.
"""
import argparse
import bisect
from collections import Counter
import hashlib
import json
import math
from pathlib import Path


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def read(path):
    content = path.read_bytes()
    return json.loads(content), hashlib.sha256(content).hexdigest()


def past_output(events, token_times, decision_s):
    """Strictly earlier host-return boundary; equality is deliberately excluded."""
    index = bisect.bisect_left(events, (decision_s, -1))-1
    token_index = bisect.bisect_left(token_times, decision_s)-1
    event = events[index] if index >= 0 else None
    token = token_times[token_index] if token_index >= 0 else None
    if (event[0] if event else None) != token:
        return dict(status='UNKNOWN', reason='PAST_OUTPUT_EVENT_TOKEN_TIME_MISMATCH')
    if event is None:
        return dict(status='UNKNOWN', reason='NO_OUTPUT_STRICTLY_BEFORE_DECISION')
    return dict(status='KNOWN', last_client_output_s=event[0],
                last_output_engine_call_index=event[1], stall_s=decision_s-event[0])


def choose(candidates, key, used_episodes, action_count, action_limit):
    if not candidates:
        return dict(request=None, would_execute=False, skip_reasons=['NO_LEGAL_CANDIDATE'])
    chosen = min(candidates, key=key)
    reasons = []
    if (chosen['request'], chosen['num_preemptions']) in used_episodes:
        reasons.append('EPISODE_ALREADY_BYPASSED')
    if action_count >= action_limit:
        reasons.append('BUDGET_EXHAUSTED')
    return dict(request=chosen['request'], source_request=chosen['source_request'],
                num_preemptions=chosen['num_preemptions'],
                stall_s=chosen['past_output'].get('stall_s'),
                would_execute=not reasons, skip_reasons=reasons)


def analyze_cell(cell):
    raw, raw_sha = read(cell/'output/raw.json')
    policy, policy_sha = read(cell/'output/recovery-repeat-unique.json')
    origin = raw['measurement_origin_perf_counter_s']
    if not finite(origin): raise ValueError('Unknown host measurement origin')
    mapping = raw['internal_to_source']
    tokens = {r['request_id']:r['token_times_s'] for r in raw['requests']}
    history = {rid:[] for rid in tokens}
    for event in raw['output_events']:
        if event.get('prefix_valid') and event.get('chunk_size', 0) > 0:
            stamp, call = event['received_s'], event['engine_call_index']
            if not finite(stamp) or type(call) is not int:
                raise ValueError('Invalid client output boundary')
            history[event['request_id']].append((stamp, call))
    for rid, times in tokens.items():
        if any(not finite(t) for t in times) or times != sorted(times):
            raise ValueError('Invalid token-time order')
        if history[rid] != sorted(history[rid]): raise ValueError('Invalid host output order')
    used, actions, decisions, counts = set(), 0, [], Counter()
    previous = -math.inf
    for index, event in enumerate(policy['events']):
        stamp = event['host_perf_s']-origin
        if not finite(stamp) or stamp < previous: raise ValueError('Decision clock order differs')
        previous = stamp
        if event['action_count_before'] != actions: raise ValueError('Actual action-prefix count differs')
        candidates = []
        for native in event['candidates']:
            if not native['eligible']: continue
            rid = native['request']; source = mapping.get(rid)
            past = (past_output(history[source], tokens[source], stamp) if source in history
                    else dict(status='UNKNOWN', reason='REQUEST_MAPPING_MISSING'))
            candidates.append(dict(request=rid, source_request=source,
                arrival_time=native['arrival_time'], num_preemptions=native['num_preemptions'],
                native_eligible=True, past_output=past,
                episode_guard_pass=(rid,native['num_preemptions']) not in used,
                budget_guard_pass=actions < event['action_limit']))
        arrival = choose(candidates, lambda c:(c['arrival_time'],c['request']),
                         used, actions, event['action_limit'])
        if arrival['request'] != event['repeat8_suggestion']:
            raise ValueError('Recorded repeat8 suggestion is not frozen arrival choice')
        if arrival['would_execute'] != event['repeat8_would_execute']:
            raise ValueError('Recorded repeat8 guard differs from actual prior episodes/budget')
        known = bool(candidates) and all(c['past_output']['status']=='KNOWN' for c in candidates)
        stall = (choose(candidates, lambda c:(c['past_output']['last_client_output_s'],
                        c['arrival_time'],c['request']), used, actions, event['action_limit']) if known
                 else dict(request=None, would_execute=None, skip_reasons=['UNKNOWN_CANDIDATE_OUTPUT_AGE']))
        completed = event['action']=='REORDERED' and event['queue_changed']
        different = arrival['request'] != stall['request'] if known else None
        row = dict(event_index=index, decision_host_perf_s=event['host_perf_s'], decision_s=stamp,
            callback_index=event['callback_index'], actual_mode=policy['mode'],
            actual_action=event['action'], actual_candidate=event['candidate_head'],
            actual_completed_reorder=completed, actual_action_skip_reasons=event['action_skip_reasons'],
            action_count_before=actions, action_limit=event['action_limit'],
            past_executed_episodes=[dict(request=rid,num_preemptions=n) for rid,n in sorted(used)],
            legal_candidates=candidates, comparison_status='KNOWN' if known else 'UNKNOWN',
            arrival_age=arrival, longest_stall=stall, suggestions_differ=different,
            both_suggestions_executable=bool(known and arrival['would_execute'] and stall['would_execute']))
        decisions.append(row)
        counts['legal_decisions'] += 1
        counts['actual_completed_reorders'] += completed
        counts['multiple_legal_candidates'] += len(candidates)>1
        counts['multiple_candidates_at_actual_reorders'] += completed and len(candidates)>1
        counts['known_age_comparisons'] += known
        counts['unknown_age_comparisons'] += not known
        counts['different_suggestions'] += different is True
        counts['different_at_actual_reorders'] += completed and different is True
        counts['different_and_both_executable'] += different is True and row['both_suggestions_executable']
        counts['different_both_executable_at_actual_reorders'] += completed and different is True and row['both_suggestions_executable']
        counts['budget_exhausted_decisions'] += actions >= event['action_limit']
        if completed:
            used.add((event['candidate_head'],event['candidate_num_preemptions'])); actions += 1
        if event['action_count_after'] != actions: raise ValueError('Actual action outcome count differs')
    if actions != policy['action_count']: raise ValueError('Final action count differs')
    return dict(cell=cell.name, mode=policy['mode'], raw_status=raw['status'],
        input_sha256={'raw.json':raw_sha,'recovery-repeat-unique.json':policy_sha},
        summary=dict(counts), decisions=decisions)


def self_check():
    # The same-call/future receipt cannot affect an earlier decision, nor does
    # an exactly equal timestamp enter the strictly-past observation set.
    assert past_output([(1.,3),(3.,4)], [1.,3.], 2.)['last_client_output_s']==1.
    assert past_output([(1.,3),(3.,4)], [1.,3.], 3.)['last_client_output_s']==1.
    assert past_output([(3.,4)], [3.], 2.)['status']=='UNKNOWN'
    candidates=[dict(request='a',source_request='a',arrival_time=0,num_preemptions=1,
        past_output=dict(last_client_output_s=1.,stall_s=2.)),
        dict(request='b',source_request='b',arrival_time=1,num_preemptions=1,
        past_output=dict(last_client_output_s=2.,stall_s=1.))]
    # Select first, then guard: do not silently fall through to executable b.
    suggestion=choose(candidates,lambda c:c['past_output']['last_client_output_s'],{('a',1)},1,8)
    assert suggestion['request']=='a' and suggestion['skip_reasons']==['EPISODE_ALREADY_BYPASSED']
    assert choose(candidates,lambda c:c['arrival_time'],set(),8,8)['skip_reasons']==['BUDGET_EXHAUSTED']
    print('PASS: strict past/equal/future boundaries and select-before-episode/budget guard; no replay or predicted benefit.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session',type=Path); parser.add_argument('--output',type=Path)
    parser.add_argument('--self-check',action='store_true'); args=parser.parse_args()
    if args.self_check:self_check()
    if args.session is None:
        if args.self_check:return 0
        parser.error('--session required')
    output=args.output or args.session/'service-age-diagnostic.json'
    if output.exists():raise FileExistsError(output)
    cells=[analyze_cell(p) for p in sorted(args.session.glob('cell-*')) if p.is_dir()]
    result=dict(status='CPU_DIAGNOSTIC_NOT_INTERVENTION', cells=cells,
        timing='Policy host perf_counter minus raw origin. Client outputs use valid positive chunks stamped after synchronous engine.step returned. Only received_s strictly less than decision_s is visible; equality/future excluded. No GPU timestamps.',
        semantics='Same recorded native eligible set. Arrival chooses earliest arrival/id; longest-stall chooses earliest last actual client output, ties arrival/id. A candidate without known prior output makes the age comparison UNKNOWN. Both select before the unchanged repeat8 episode/budget guard using only this cell actual earlier actions. Unique8 raw is described separately, not spliced into another trajectory. No counterfactual outputs, latency savings, avoided transfers, oracle or performance claim.',
        analyzer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    with output.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps(dict(output=str(output),cells=[dict(cell=c['cell'],**c['summary']) for c in cells])))
    return 0


if __name__=='__main__':raise SystemExit(main())
