#!/usr/bin/env python3
"""CPU victim/pressure joins; reuse full analysis and preserve every decision/request."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
from health_analyze_v2 import native_id_mapping
from math_restore_order_evidence_v1 import read_bytes

SCHEDULER_SHA = '2ed2a550b6558b2495eda845a97ae38bcf0225027b9e25fbf00fc3880c1d3941'


def analyze(run, analysis_path):
    run, analysis_path = Path(run), Path(analysis_path)
    hashes, issues = {}, []
    def check(ok, message):
        if not ok: issues.append(message)
    def read(name):
        data = read_bytes(run / name)
        hashes[name] = hashlib.sha256(data).hexdigest()
        return json.loads(data)
    analysis_bytes = read_bytes(analysis_path)
    quality = json.loads(analysis_bytes)
    source, outputs = read('source-input.json'), read('measured-outputs.json')
    status, config = read('status.json'), read('config.json')
    sampling, pressure = read('measured-native-sampling.json'), read('measured-pressure.json')
    policy = read('victim-choice-policy.json')
    check(quality.get('integrity') == 'VALID', 'full_analysis_not_valid')
    check(all(x.get('status') == 'COMPLETE' for x in (status, pressure, policy)), 'incomplete_run_or_receipt')
    check(policy.get('schema') == 'c-victim-choice-policy-v1' and policy.get('scheduler_source_sha256') == SCHEDULER_SHA,
          'policy_schema_or_native_source_mismatch')
    mode = policy.get('mode')
    check(mode in ('bidkv-full', 'max-free-full') and config.get('policy') == 'victim-' + str(mode)
          and config.get('fixed_margin_blocks') == 48, 'mode_or_fixed_margin_mismatch')
    for name in tuple(hashes)[:-1]:
        check(quality.get('input_sha256', {}).get(name) == hashes[name], 'analysis_hash_mismatch:' + name)
    source_ids = [r['request_id'] for r in (source['requests'] if isinstance(source, dict) else source)]
    check(len(set(source_ids)) == len(source_ids), 'duplicate_source_id')
    mapping, errors = native_id_mapping(sampling, {'measured/' + rid for rid in source_ids})
    issues.extend(errors)
    def sid(rid):
        check(rid in mapping, 'unknown_native_id:' + str(rid))
        return mapping[rid][len('measured/'):] if rid in mapping else None
    out_by, q_by = ({r['request_id']: r for r in rows} for rows in (outputs, quality['per_request']))
    check(len(out_by) == len(outputs) and set(out_by) == set(source_ids), 'output_inventory_mismatch')
    check(len(q_by) == len(quality['per_request']) and set(q_by) == set(source_ids), 'quality_inventory_mismatch')
    calls, attempts = pressure['scheduler_calls'], pressure['allocation_attempts']
    check([c['call_id'] for c in calls] == list(range(len(calls))), 'noncontiguous_or_duplicate_call_ids')
    by_call = {c['call_id']: c for c in calls}; by_attempt = {a['attempt_id']: a for a in attempts}
    check(len(by_attempt) == len(attempts), 'duplicate_attempt_id')
    records = {rid: dict(trigger_decisions=[], victim_decisions=[], native_running_none_attempt_ids=[],
        restore_attempt_ids=[], final_scheduled_calls=0, final_scheduled_tokens=0) for rid in source_ids}
    failed, linked = defaultdict(list), []
    for c in calls:
        for aid in c['attempt_ids']:
            a = by_attempt[aid]; linked.append(aid)
            check(a['call_id'] == c['call_id'], 'attempt_call_mismatch:' + str(aid))
            if a['status_before'] == 'RUNNING' and a.get('returned_none') is True: failed[c['call_id']].append(a)
        for rid, n in c.get('scheduled_tokens', {}).items():
            key = sid(rid)
            if key in records:
                records[key]['final_scheduled_calls'] += 1; records[key]['final_scheduled_tokens'] += n
    check(Counter(linked) == Counter(a['attempt_id'] for a in attempts), 'attempt_inventory_not_exactly_linked')
    for a in attempts:
        key = sid(a['request_id'])
        if key not in records: continue
        if a['status_before'] == 'RUNNING' and a.get('returned_none') is True:
            records[key]['native_running_none_attempt_ids'].append(a['attempt_id'])
        if a['status_before'] == 'PREEMPTED': records[key]['restore_attempt_ids'].append(a['attempt_id'])
    initial = policy.get('initial_scheduler_step')
    if type(initial) is not int or initial < 0: raise ValueError('missing valid initial_scheduler_step')
    offset, seen, used, joined = initial + 1, Counter(), set(), []
    victims = defaultdict(list)
    counts = Counter({k: 0 for k in ('selector_decisions', 'joined_native_running_none', 'actual_preemptions',
        'different_native_tail', 'selected_scheduled_prefix', 'selected_current', 'selected_suffix',
        'candidate_evaluations', 'predicted_released_blocks', 'actual_released_blocks', 'physical_delta_matches',
        'scheduled_prefix_refunded_tokens', 'immediate_current_retries', 'current_self_preempt_breaks')})
    candidate_hist = Counter()
    for position, e in enumerate(policy['decisions']):
        counts['selector_decisions'] += 1
        did, cid = e['decision'], e['scheduler_step'] - offset
        label = str(did); issue_start = len(issues); ordinal = seen[cid]; seen[cid] += 1
        check(did == position and e['trigger_index_in_call'] == ordinal, 'decision_order:' + label)
        check(position == 0 or e['scheduler_step'] >= policy['decisions'][position - 1]['scheduler_step'], 'step_order:' + label)
        c = by_call.get(cid); matches = failed.get(cid, [])
        check(c is not None and ordinal < len(matches), 'missing_running_none_trigger:' + label)
        row = dict(policy_decision=e, observer_call_id=cid, observer_attempt_id=None,
            trigger_source_request_id=sid(e['trigger_current_id']), selected_source_request_id=sid(e['selected_id']))
        if c is None or ordinal >= len(matches):
            row['join_valid'] = False; joined.append(row); continue
        a = matches[ordinal]; aid = a['attempt_id']; at = c['attempt_ids'].index(aid)
        check(aid not in used, 'trigger_reused:' + label); used.add(aid)
        check(a['request_id'] == e['trigger_current_id'] and a['arguments']['num_new_tokens'] == e['trigger_num_new_tokens']
              and a['free_blocks_after'] == e['free_blocks_at_trigger'] and a.get('succeeded') is False,
              'trigger_fields:' + label)
        pending = {by_attempt[x]['request_id']: by_attempt[x]['arguments']['num_new_tokens'] for x in c['attempt_ids'][:at]
                   if by_attempt[x]['status_before'] == 'RUNNING' and by_attempt[x].get('succeeded') is True}
        for rid in victims[cid]: pending.pop(rid, None)
        cur, n, chosen, tail = e['current_running_index'], e['running_count'], e['selected_candidate'], e['native_tail_candidate']
        kind = 'scheduled-prefix' if chosen['running_index'] < cur else 'current' if chosen['running_index'] == cur else 'suffix'
        check(0 <= cur < n and n == c['before']['running'] - len(victims[cid])
              and e['candidate_count'] == len(pending) + n - cur
              and e['excluded_unscheduled_prefix'] == cur - len(pending) >= 0, 'candidate_counts:' + label)
        check(chosen['request_id'] == e['selected_id'] and chosen['kind'] == e['selected_kind'] == kind
              and 0 <= chosen['running_index'] < n and tail['request_id'] == e['native_tail_id']
              and tail['running_index'] == n - 1 and tail['kind'] == ('current' if cur == n - 1 else 'suffix')
              and e['changed_from_native_tail'] == (e['selected_id'] != e['native_tail_id']),
              'candidate_identity_or_kind:' + label)
        prefix = kind == 'scheduled-prefix'; current = kind == 'current'
        if 'apc_domain' in policy or any(k in chosen for k in ('num_tokens', 'num_computed_tokens', 'single_token_decode')):
            check(all(type(chosen.get(k)) is int for k in ('num_tokens', 'num_computed_tokens', 'num_prompt_tokens'))
                  and chosen.get('single_token_decode') is True and chosen.get('num_computed_tokens', -1) >= chosen.get('num_prompt_tokens', 0)
                  and chosen.get('num_tokens') == chosen.get('num_computed_tokens', -2) + 1
                  and e['selected_scheduled_tokens'] in (0, 1), 'selected_apc_decode_domain:' + label)
            check(0 <= e.get('candidate_non_single_decode_count', -1) < e['candidate_count'], 'candidate_decode_counts:' + label)
        if 'preemption' not in e or 'rollback' not in e:
            check(False, 'missing_actual_preemption_or_rollback:' + label)
            row.update(observer_attempt_id=aid, observer_allocation=a, join_valid=False); joined.append(row); continue
        r, b = e['preemption'], e['rollback']
        check((e['selected_id'] in pending) == prefix and (e['selected_id'] == a['request_id']) == current
              and e['selected_scheduled_tokens'] == pending.get(e['selected_id'], 0), 'prefix_or_current_identity:' + label)
        refund = e['selected_scheduled_tokens'] if prefix else 0
        check(b['scheduled_prefix'] == prefix and b['removed_from_final_pending_maps'] == prefix
              and b['refunded_tokens'] == refund and b['budget_after'] == e['budget_before'] + refund
              and b['req_index_after'] == cur - int(prefix), 'rollback_fields:' + label)
        delta = r['free_blocks_after'] - r['free_blocks_before']
        check(r['request_id'] == e['selected_id'] and r['native_returned'] is True and 'error' not in r
              and r['free_blocks_before'] == a['free_blocks_after'] and delta == r['actual_freed_blocks']
              == e['selected_releasable_blocks'] == chosen['releasable_blocks'] >= 0
              and r['predicted_free_delta_matches'] is True, 'preempt_or_physical_delta:' + label)
        check(e['selected_id'] in c.get('preempted_request_ids', []) and e['selected_id'] not in c.get('scheduled_tokens', {}),
              'final_preempt_or_schedule:' + label)
        next_a = by_attempt[c['attempt_ids'][at + 1]] if at + 1 < len(c['attempt_ids']) else None
        check((next_a is None and c['after']['free_blocks'] == r['free_blocks_after']) if current else
              (next_a is not None and next_a['request_id'] == a['request_id'] and next_a['status_before'] == 'RUNNING'
               and next_a['free_blocks_before'] == r['free_blocks_after']), 'native_retry_or_self_break:' + label)
        victims[cid].append(e['selected_id']); candidate_hist[e['candidate_count']] += 1
        counts.update(joined_native_running_none=1, actual_preemptions=int(r['native_returned']),
            different_native_tail=int(e['changed_from_native_tail']), candidate_evaluations=e['candidate_count'],
            predicted_released_blocks=e['selected_releasable_blocks'], actual_released_blocks=delta,
            physical_delta_matches=int(delta == e['selected_releasable_blocks']), scheduled_prefix_refunded_tokens=refund,
            immediate_current_retries=int(not current), current_self_preempt_breaks=int(current))
        counts['selected_' + kind.replace('-', '_')] += 1
        for rid, name in ((a['request_id'], 'trigger_decisions'), (e['selected_id'], 'victim_decisions')):
            key = sid(rid)
            if key in records: records[key][name].append(did)
        row.update(observer_attempt_id=aid, observer_allocation=a, next_observer_allocation=next_a,
            reconstructed_scheduled_prefix_ids=list(pending), join_valid=len(issues) == issue_start)
        joined.append(row)
    for c in calls:
        cid = c['call_id']
        check(seen[cid] == len(failed[cid]), 'unmatched_running_none:' + str(cid))
        check(Counter(victims[cid]) == Counter(c.get('preempted_request_ids', [])), 'preempt_inventory:' + str(cid))
    check(policy.get('selection_decisions') == len(policy['decisions']) == counts['selector_decisions']
          and policy.get('actual_preemptions') == counts['actual_preemptions'], 'declared_policy_counts')
    for key in source_ids:
        o, q = out_by[key], q_by[key]
        check(o['external_request_id'] == 'measured/' + key and o['finished'] == q['completed']
              and o['finish_reason'] == q['finish_reason'] and len(o['output_token_ids']) == q['output_tokens'], 'final_output:' + key)
        records[key].update(final_output=o, full_analysis_result=q)
    counts['observer_native_running_none'] = sum(len(rows) for rows in failed.values())
    counts['requests_selected_more_than_once'] = sum(len(r['victim_decisions']) > 1 for r in records.values())
    return dict(schema='c-math-victim-choice-evidence-v1', integrity='VALID' if not issues else 'INVALID', issues=issues,
        run=str(run.resolve()), analysis=str(analysis_path.resolve()), mode=mode, counts=dict(counts),
        policy_hook_wall_s=policy.get('policy_hook_wall_s'), policy_apc_domain=policy.get('apc_domain'),
        input_sha256=hashes, analysis_sha256=hashlib.sha256(analysis_bytes).hexdigest(),
        analyzer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), policy_step_minus_observer_call_id=offset,
        candidate_count_histogram=dict(sorted(candidate_hist.items())), joined_decisions=joined, all_requests=records,
        limitations=['Joins rely on the pinned native None-to-selector source branch and per-call failure order; no GPU timing inference.',
            'Pressure preempt IDs are an unordered set; receipts establish event order, checked against the complete per-call inventory.',
            'Physical release is checked against native preempt pool deltas; independent owner graph and unlogged candidate scores are unavailable.',
            'All input requests, complete outcomes and decisions are retained; existing full analysis is reused without regrading.',
            'Observed victim changes, release and rollback are mechanism evidence, not causal speedup or novelty claims.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('run', 'analysis', 'output'): parser.add_argument('--' + key, type=Path, required=True)
    args = parser.parse_args(); result = analyze(args.run, args.analysis)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps(dict(integrity=result['integrity'], counts=result['counts'], issues=result['issues'])))
    sys.exit(0 if result['integrity'] == 'VALID' else 2)
