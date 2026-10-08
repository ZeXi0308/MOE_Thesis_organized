#!/usr/bin/env python3
"""CPU-only joins for saved new-request bypass receipts; exclusive output, no causal claim."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import lzma
from pathlib import Path
import sys

sys.dont_write_bytecode = True
from health_analyze_v2 import native_id_mapping


def analyze(run, analysis_path):
    hashes, issues = {}, []
    def read(name):
        path = run / name
        data = path.read_bytes() if path.exists() else lzma.open(str(path) + '.xz', 'rb').read()
        hashes[name] = hashlib.sha256(data).hexdigest()
        return json.loads(data)
    def check(ok, message):
        if not ok:
            issues.append(message)
    analysis_bytes = analysis_path.read_bytes()
    quality = json.loads(analysis_bytes)
    source, outputs = read('source-input.json'), read('measured-outputs.json')
    status, config = read('status.json'), read('config.json')
    sampling, pressure = read('measured-native-sampling.json'), read('measured-pressure.json')
    policy = read('restore-new-bypass-policy.json')
    if policy.get('fixed_gate_receipt') != 'restore-new-bypass-fixed-gate.json':
        raise ValueError('unexpected fixed-gate receipt path')
    gate = read('restore-new-bypass-fixed-gate.json')
    check(quality.get('integrity') == 'VALID', 'full_analysis_not_valid')
    check(status.get('status') == pressure.get('status') == policy.get('status') == gate.get('status') == 'COMPLETE',
          'incomplete_run_or_receipt')
    check(config.get('policy') == 'restore-new-bypass' and config.get('fixed_margin_blocks') == 48 and
          policy.get('fixed_margin_blocks') == gate.get('fixed_margin_blocks') == 48 and
          policy.get('maximum_fresh_candidates') == 16, 'wrong_policy_margin_or_scan_limit')
    for name in ('source-input.json', 'measured-outputs.json', 'status.json', 'config.json',
                 'measured-native-sampling.json', 'measured-pressure.json'):
        check(quality.get('input_sha256', {}).get(name) == hashes[name], 'analysis_hash_mismatch:' + name)
    source_ids = [r['request_id'] for r in source['requests']]
    check(len(set(source_ids)) == len(source_ids), 'duplicate_source_id')
    mapping, errors = native_id_mapping(sampling, {'measured/' + rid for rid in source_ids})
    issues.extend(errors)
    def source_id(native):
        if native not in mapping:
            raise ValueError('unknown native request ID: ' + str(native))
        return mapping[native][len('measured/'):]
    out_by = {r['request_id']: r for r in outputs}
    q_by = {r['request_id']: r for r in quality['per_request']}
    check(len(out_by) == len(outputs) and set(out_by) == set(source_ids), 'output_inventory_mismatch')
    check(len(q_by) == len(quality['per_request']) and set(q_by) == set(source_ids), 'quality_inventory_mismatch')
    calls = {c['call_id']: c for c in pressure['scheduler_calls']}
    waiting, successes = defaultdict(list), defaultdict(list)
    for a in pressure['allocation_attempts']:
        if a['status_before'] == 'WAITING':
            waiting[a['request_id']].append(a)
            if a.get('succeeded') is True and a.get('returned_none') is False:
                successes[a['request_id']].append(a)
    decisions = {d['decision']: d for d in policy['decisions']}
    check(len(decisions) == len(policy['decisions']), 'duplicate_policy_decision')
    joined, offsets, involved = [], set(), set()
    counts = Counter({k: 0 for k in (
        'competition_decisions', 'head_gate_deferred', 'head_full_plus_one_fits_but_48_margin_defers',
        'head_full_plus_one_itself_exceeds_free_including_watermark', 'head_passes_margin_gate',
        'candidate_queries', 'candidate_queries_fit_with_48_margin', 'decisions_querying_all_16_candidates',
        'queue_entries_visited', 'queue_rollbacks', 'policy_allocation_receipts',
        'native_successful_fresh_allocations', 'native_failed_fresh_allocations',
        'fresh_allocation_errors', 'confirmed_executed_bypasses')})
    scan_hist = Counter()
    decision_rows = []
    for d in policy['decisions']:
        head, candidates = d['restore_head'], d['candidates']
        head_id = source_id(d['restore_head_id']); involved.add(head_id)
        without_margin = head['candidate_full_plus_one_blocks'] + head['watermark'] <= head['free_blocks']
        check(without_margin == d['head_plus_one_fits_without_margin'], 'head_fit_flag_mismatch:' + str(d['decision']))
        counts['competition_decisions'] += 1
        counts['head_gate_deferred'] += head['policy_deferred']
        counts['head_full_plus_one_fits_but_48_margin_defers'] += head['policy_deferred'] and without_margin
        counts['head_full_plus_one_itself_exceeds_free_including_watermark'] += not without_margin
        counts['head_passes_margin_gate'] += not head['policy_deferred']
        counts['candidate_queries'] += len(candidates)
        counts['candidate_queries_fit_with_48_margin'] += sum(not c['policy_deferred'] for c in candidates)
        counts['decisions_querying_all_16_candidates'] += len(candidates) == 16
        counts['queue_entries_visited'] += d['queue_entries_visited']
        counts['action_' + d['action'].lower()] += 1
        counts['queue_rollbacks'] += d.get('queue_rollback') is True
        scan_hist[len(candidates)] += 1
        check(len(candidates) <= 16, 'candidate_scan_limit_exceeded:' + str(d['decision']))
        selected = source_id(d['selected_id']) if d.get('selected_id') else None
        if selected:
            involved.add(selected)
        decision_rows.append(dict(d, restore_head_source_id=head_id, selected_source_id=selected,
            candidates=[dict(c, source_request_id=source_id(c['request_id'])) for c in candidates]))
    for index, r in enumerate(policy['allocation_receipts']):
        rid = r['request_id']; sid = source_id(rid); involved.add(sid)
        d = decisions.get(r['decision'])
        check(d is not None and d.get('selected_id') == rid and d['scheduler_step'] == r['scheduler_step'],
              'receipt_decision_identity_mismatch:' + str(index))
        ok = r.get('native_succeeded') is True
        candidates = successes[rid] if ok else waiting[rid]
        def same(a):
            return (a['status_before'] == r['status_before'] == 'WAITING' and
                a['arguments']['num_new_tokens'] == r['num_new_tokens'] and
                a['free_blocks_before'] == r['free_blocks_before'] and
                a['free_blocks_after'] == r['free_blocks_after'] and
                (a.get('succeeded') is True if ok else
                 a.get('returned_none') is True if r.get('native_succeeded') is False else bool(a.get('error'))))
        matches = [a for a in candidates if same(a)]
        unique = len(matches) == 1 and (not ok or len(candidates) == 1)
        row = dict(policy_receipt=r, source_request_id=sid,
            restore_head_source_id=source_id(d['restore_head_id']) if d else None,
            queue_rollback=d.get('queue_rollback', False) if d else None,
            observer_candidate_attempt_ids=[a['attempt_id'] for a in candidates],
            exact_matching_attempt_ids=[a['attempt_id'] for a in matches],
            unique_observer_join=unique, final_scheduled_tokens=None, confirmed_executed_bypass=False)
        if ok:
            check(unique, 'successful_fresh_join_not_unique_or_fields_differ:' + str(index))
        if unique:
            a = matches[0]; c = calls.get(a['call_id'])
            linked = c is not None and a['attempt_id'] in c['attempt_ids']
            check(linked, 'observer_attempt_missing_from_call:' + str(index))
            n = c.get('scheduled_tokens', {}).get(rid, 0) if c else 0
            row.update(observer_allocation=a, observer_call_id=a['call_id'], final_scheduled_tokens=n,
                confirmed_executed_bypass=bool(ok and linked and n > 0 and not row['queue_rollback']))
            offsets.add(r['scheduler_step'] - a['call_id'])
            if ok:
                check(n > 0 and not row['queue_rollback'], 'successful_allocation_not_final_scheduled:' + str(index))
        counts['policy_allocation_receipts'] += 1
        counts['native_successful_fresh_allocations'] += ok
        counts['native_failed_fresh_allocations'] += r.get('native_succeeded') is False
        counts['fresh_allocation_errors'] += 'error' in r
        counts['confirmed_executed_bypasses'] += row['confirmed_executed_bypass']
        joined.append(row)
    check(len(offsets) <= 1, 'policy_scheduler_step_observer_call_offset_inconsistent')
    selected_decisions = {d['decision'] for d in policy['decisions'] if d['action'] == 'BYPASS_WITH_FRESH'}
    receipt_decisions = Counter(r['decision'] for r in policy['allocation_receipts'])
    for did in selected_decisions:
        check(receipt_decisions[did] == 1 or (receipt_decisions[did] == 0 and decisions[did].get('queue_rollback')),
              'selected_decision_receipt_missing_or_duplicate:' + str(did))
    counts['selected_decisions_without_allocation_receipt'] = len(selected_decisions - set(receipt_decisions))
    counts['fixed_gate_policy_deferred_without_native_call'] = sum(d.get('policy_deferred') is True and d.get('native_called') is False for d in gate['decisions'])
    counts['fixed_gate_native_none_after_forwarding'] = sum(d.get('native_called') is True and d.get('native_returned_none') is True for d in gate['decisions'])
    counts['observer_native_none_all_statuses'] = sum(a.get('returned_none') is True for a in pressure['allocation_attempts'])
    for name, expected in (
            ('competition_decisions', len(policy['decisions'])),
            ('applied_reorders', sum(d['action'] == 'BYPASS_WITH_FRESH' for d in policy['decisions'])),
            ('native_successful_bypass_allocations', sum(r.get('native_succeeded') is True for r in policy['allocation_receipts']))):
        check(policy.get(name) == expected, 'policy_declared_count_mismatch:' + name)
    check(gate.get('policy_deferred_count') == sum(d.get('policy_deferred') is True for d in gate['decisions']),
          'fixed_gate_declared_defer_count_mismatch')
    results = {}
    for sid in sorted(involved):
        o, q = out_by[sid], q_by[sid]
        token_ids = o['output_token_ids']; times = o['token_times_s']
        arrival = o.get('arrival_s', 0.0)
        ttft = times[0] - arrival if times else None
        flow = o['host_elapsed_s'] - arrival if o.get('host_elapsed_s') is not None else None
        check(o['external_request_id'] == 'measured/' + sid and len(token_ids) == q['output_tokens'] and
              o['finished'] == q['completed'] and o['finish_reason'] == q['finish_reason'] and
              ttft == q['ttft_s'] and flow == q['completion_s'], 'final_output_quality_timing_mismatch:' + sid)
        results[sid] = dict(request_id=sid, full_output_text=o['output_text'], output_token_ids=token_ids,
            finished=o['finished'], finish_reason=o['finish_reason'], stop_reason=o['stop_reason'],
            arrival_s=arrival, ttft_s=ttft, flow_s=flow, quality=q)
    return dict(schema='c-math-new-bypass-evidence-v1', integrity='VALID' if not issues else 'INVALID',
        run=str(run.resolve()), analysis=str(analysis_path.resolve()), issues=issues,
        input_sha256=hashes, analysis_sha256=hashlib.sha256(analysis_bytes).hexdigest(),
        analyzer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        fixed_margin_blocks=48, maximum_fresh_candidates=16, counts=dict(counts),
        queried_candidates_per_decision=dict(sorted(scan_hist.items())),
        policy_step_minus_observer_call_id=sorted(offsets), decisions=decision_rows,
        joined_allocation_receipts=joined, final_results_by_source_request=results,
        limitations=['Observed actions and outcomes only; no causal performance or quality benefit attribution.',
            'Head fit means full-known-sequence+1 with native watermark; it is not an assertion that native current-sequence admission would fail.',
            'Candidate query counts cover tested fresh candidates up to16; untested suffix feasibility is unknown.',
            'Policy-gate deferrals are separate from observer-native None; failed allocations and rollbacks remain in the evidence.',
            'TTFT/flow are emitted-token host-return times from arrival, not device latency.'])


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--analysis', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = analyze(args.run, args.analysis)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(integrity=result['integrity'], counts=result['counts'], issues=result['issues'])))
    sys.exit(0 if result['integrity'] == 'VALID' else 2)
