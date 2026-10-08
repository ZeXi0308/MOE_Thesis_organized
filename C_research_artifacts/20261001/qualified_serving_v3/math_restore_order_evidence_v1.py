#!/usr/bin/env python3
"""CPU joins for bounded restoration ordering; repeated restores are distinct events."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import lzma
from pathlib import Path
import sys

sys.dont_write_bytecode = True
from health_analyze_v2 import native_id_mapping


def read_bytes(path):
    path = Path(path)
    if not path.exists():
        path = Path(str(path) + '.xz')
    data = path.read_bytes()
    return lzma.decompress(data) if path.suffix == '.xz' else data


def analyze(run, analysis_path):
    run, analysis_path = Path(run), Path(analysis_path)
    hashes, issues = {}, []
    def read(name):
        data = read_bytes(run / name)
        hashes[name] = hashlib.sha256(data).hexdigest()
        return json.loads(data)
    def check(ok, message):
        if not ok:
            issues.append(message)
    analysis_bytes = read_bytes(analysis_path)
    quality = json.loads(analysis_bytes)
    source, outputs = read('source-input.json'), read('measured-outputs.json')
    status, config = read('status.json'), read('config.json')
    sampling, pressure = read('measured-native-sampling.json'), read('measured-pressure.json')
    policy = read('restore-cost-order-policy.json')
    if policy.get('fixed_gate_receipt') != 'restore-cost-order-fixed-gate.json':
        raise ValueError('unexpected fixed-gate receipt path')
    gate = read('restore-cost-order-fixed-gate.json')
    check(quality.get('integrity') == 'VALID', 'full_analysis_not_valid')
    check(all(x.get('status') == 'COMPLETE' for x in (status, pressure, policy, gate)), 'incomplete_run_or_receipt')
    mode = policy.get('mode')
    check(mode in ('first-fit', 'min-recompute') and config.get('policy') == 'restore-' + str(mode)
          and config.get('fixed_margin_blocks') == policy.get('fixed_margin_blocks') == gate.get('fixed_margin_blocks') == 48
          and policy.get('maximum_restore_candidates') == 16, 'wrong_mode_margin_or_limit')
    for name in ('source-input.json', 'measured-outputs.json', 'status.json', 'config.json',
                 'measured-native-sampling.json', 'measured-pressure.json'):
        check(quality.get('input_sha256', {}).get(name) == hashes[name], 'analysis_hash_mismatch:' + name)
    source_ids = [r['request_id'] for r in (source['requests'] if isinstance(source, dict) else source)]
    check(len(set(source_ids)) == len(source_ids), 'duplicate_source_id')
    mapping, errors = native_id_mapping(sampling, {'measured/' + rid for rid in source_ids})
    issues.extend(errors)
    def source_id(rid):
        return mapping[rid][len('measured/'):]
    out_by, q_by = ({r['request_id']: r for r in rows} for rows in (outputs, quality['per_request']))
    check(len(out_by) == len(outputs) and set(out_by) == set(source_ids), 'output_inventory_mismatch')
    check(len(q_by) == len(quality['per_request']) and set(q_by) == set(source_ids), 'quality_inventory_mismatch')
    calls = {c['call_id']: c for c in pressure['scheduler_calls']}
    check(list(calls) == list(range(len(pressure['scheduler_calls']))), 'noncontiguous_or_duplicate_call_ids')
    initial = policy.get('initial_scheduler_step')
    if type(initial) is not int or initial < 0:
        raise ValueError('missing valid initial_scheduler_step')
    offset = initial + 1
    by_key, attempts = defaultdict(list), pressure['allocation_attempts']
    check(len({a['attempt_id'] for a in attempts}) == len(attempts), 'duplicate_attempt_id')
    for a in attempts:
        source_id(a['request_id'])
        by_key[(a['call_id'], a['request_id'], a['status_before'])].append(a)
    decisions = {d['decision']: d for d in policy['decisions']}
    gates = {g['decision']: g for g in gate['decisions']}
    check(len(decisions) == len(policy['decisions']) and len(gates) == len(gate['decisions']), 'duplicate_decision_id')
    counts = Counter({k: 0 for k in ('selection_decisions', 'candidate_queries', 'candidate_queries_fit',
        'requested_reorders', 'kept_head_decisions', 'no_fit_decisions', 'queue_rollbacks',
        'confirmed_executed_reorders', 'confirmed_executed_kept_heads', 'executed_blocked_head_bypasses',
        'executed_lower_cost_over_fit_head', 'selected_policy_deferred_without_native_call',
        'selected_native_none', 'selected_native_successes', 'selected_allocation_errors',
        'executed_head_bypasses_margin_only', 'executed_head_bypasses_full_plus_one_exceeds_free')})
    rows, query_hist, selected = [], Counter(), {}
    for d in policy['decisions']:
        did, cs, prefix = d['decision'], d['candidates'], d['prefix_request_ids']
        for rid in prefix:
            source_id(rid)
        check(2 <= len(prefix) <= 16 and len(set(prefix)) == len(prefix) and prefix[0] == d['restore_head_id'], 'bad_prefix:' + str(did))
        check(0 < len(cs) <= len(prefix) and d['queried_candidates'] == len(cs), 'bad_query_count:' + str(did))
        for pos, c in enumerate(cs):
            check(c['request_id'] == prefix[pos] and c['original_queue_position'] == pos, 'query_order:' + str(did))
            check(c['known_recompute_tokens'] == c['candidate_known_tokens'] - c['candidate_cached_tokens'] > 0,
                  'known_recompute_mismatch:' + str(did))
            need = c['candidate_full_plus_one_blocks'] + 48 + c['watermark'] + c['original_reserved_blocks']
            check(c['required_blocks'] == need and c['policy_deferred'] == (need > c['free_blocks']), 'gate_math:' + str(did))
        fits = [c for c in cs if not c['policy_deferred']]
        chosen = (fits[0] if mode == 'first-fit' else min(fits, key=lambda c: (c['known_recompute_tokens'], c['original_queue_position']))) if fits else None
        if mode == 'first-fit':
            check((chosen is not None and chosen is cs[-1]) or (chosen is None and len(cs) == len(prefix)), 'first_fit_query_stop:' + str(did))
        else:
            check(len(cs) == len(prefix), 'min_recompute_incomplete_queries:' + str(did))
        position = chosen['original_queue_position'] if chosen else None
        action = 'NO_RESTORE_CANDIDATE_FITS' if chosen is None else 'KEEP_RESTORE_HEAD' if position == 0 else 'REORDER_RESTORE_PREFIX'
        check(d['action'] == action and d['selected_id'] == (chosen['request_id'] if chosen else None)
              and d['selected_original_position'] == position, 'selection_rule:' + str(did))
        reorder = position is not None and position > 0
        blocked = bool(reorder and cs[0]['policy_deferred'])
        lower = bool(reorder and not cs[0]['policy_deferred'])
        check(d['bypasses_unfit_head'] == blocked and d['reorders_fit_head_for_lower_cost'] == lower, 'reason_flags:' + str(did))
        if lower:
            check(chosen['known_recompute_tokens'] < cs[0]['known_recompute_tokens'], 'nonlower_cost_reorder:' + str(did))
        selected[did] = chosen
        counts.update(selection_decisions=1, candidate_queries=len(cs), candidate_queries_fit=len(fits),
            requested_reorders=int(reorder), kept_head_decisions=int(position == 0), no_fit_decisions=int(chosen is None),
            queue_rollbacks=int(d.get('queue_rollback') is True))
        query_hist[len(cs)] += 1
        rows.append(dict(d, restore_head_source_id=source_id(d['restore_head_id']),
            selected_source_id=source_id(d['selected_id']) if chosen else None,
            candidates=[dict(c, source_request_id=source_id(c['request_id'])) for c in cs]))
    for g in gate['decisions']:
        matches = by_key[(g['scheduler_step'] - offset, g['request_id'], 'PREEMPTED')]
        check(len(matches) == (1 if g.get('native_called') else 0), 'gate_observer_offset_or_call_mismatch:' + str(g['decision']))
        if g.get('native_called') and len(matches) == 1:
            a = matches[0]
            check(a['num_tokens'] == g['candidate_known_tokens'] and a['free_blocks_before'] == g['free_blocks']
                  and a['arguments']['num_new_computed_tokens'] == g['candidate_cached_tokens']
                  and a.get('returned_none') == g.get('native_returned_none'), 'gate_observer_fields:' + str(g['decision']))
    joined, used, receipt_counts, success_by_id = [], set(), Counter(), Counter()
    for r in policy['allocation_receipts']:
        did, rid, cid = r['decision'], r['request_id'], r['scheduler_step'] - offset
        d, g = decisions[did], gates[r['fixed_gate_decision']]
        receipt_counts[did] += 1
        check(rid == d['selected_id'] == g['request_id'] and r['scheduler_step'] == d['scheduler_step'] == g['scheduler_step']
              and r['status_before'] == 'PREEMPTED', 'receipt_identity:' + str(did))
        check(r.get('native_called') == g.get('native_called') and r['policy_deferred'] == g['policy_deferred'], 'receipt_gate_flags:' + str(did))
        matches = by_key[(cid, rid, 'PREEMPTED')]
        row = dict(policy_receipt=r, source_request_id=source_id(rid), observer_call_id=cid,
            observer_attempt_id=None, unique_observer_join=False, final_scheduled_tokens=0, confirmed_executed=False)
        if not r.get('native_called'):
            check(r['policy_deferred'] and not matches and r.get('native_succeeded') is not True, 'policy_defer_has_native_attempt:' + str(did))
            counts['selected_policy_deferred_without_native_call'] += 1
        else:
            check(len(matches) == 1, 'selected_restore_join_not_unique:' + str(did))
            if len(matches) == 1:
                a, c = matches[0], calls[cid]
                check(a['attempt_id'] not in used and a['attempt_id'] in c['attempt_ids'], 'attempt_reused_or_unlinked:' + str(did))
                used.add(a['attempt_id'])
                check(a['arguments']['num_new_tokens'] == r['num_new_tokens'] and a['free_blocks_before'] == r['free_blocks_before']
                      and a['free_blocks_after'] == r['free_blocks_after'] and a.get('succeeded') == r.get('native_succeeded')
                      and a.get('returned_none') == r.get('native_returned_none')
                      and a.get('succeeded') == r.get('allocation_succeeded'), 'allocation_fields:' + str(did))
                n = c.get('scheduled_tokens', {}).get(rid, 0)
                executed = a.get('succeeded') is True and n > 0 and not d.get('queue_rollback')
                row.update(observer_attempt_id=a['attempt_id'], observer_allocation=a, unique_observer_join=True,
                    final_scheduled_tokens=n, confirmed_executed=executed)
                if a.get('succeeded'):
                    check(executed, 'allocation_success_not_final_scheduled:' + str(did))
                    success_by_id[source_id(rid)] += 1
                counts['selected_native_none'] += a.get('returned_none') is True
                counts['selected_native_successes'] += a.get('succeeded') is True
                if executed:
                    reorders = d['action'] == 'REORDER_RESTORE_PREFIX'
                    counts['confirmed_executed_reorders' if reorders else 'confirmed_executed_kept_heads'] += 1
                    counts['executed_blocked_head_bypasses'] += d['bypasses_unfit_head']
                    counts['executed_lower_cost_over_fit_head'] += d['reorders_fit_head_for_lower_cost']
                    if d['bypasses_unfit_head']:
                        h = d['candidates'][0]
                        fits_without_margin = h['candidate_full_plus_one_blocks'] + h['watermark'] + h['original_reserved_blocks'] <= h['free_blocks']
                        counts['executed_head_bypasses_margin_only' if fits_without_margin else 'executed_head_bypasses_full_plus_one_exceeds_free'] += 1
        counts['selected_allocation_errors'] += 'error' in r
        joined.append(row)
    for did, candidate in selected.items():
        check(receipt_counts[did] == int(candidate is not None), 'missing_or_duplicate_selected_receipt:' + str(did))
    for key, value in (('selection_decisions', len(rows)), ('applied_reorders', counts['requested_reorders']),
                       ('native_successful_selected_allocations', counts['selected_native_successes']),
                       ('native_successful_reordered_allocations', sum(r.get('native_succeeded') is True and
                        decisions[r['decision']]['action'] == 'REORDER_RESTORE_PREFIX' for r in policy['allocation_receipts']))):
        check(policy.get(key) == value, 'declared_count:' + key)
    counts['fixed_gate_policy_deferred_without_native_call'] = sum(g.get('policy_deferred') and g.get('native_called') is False for g in gate['decisions'])
    check(gate.get('policy_deferred_count') == sum(g.get('policy_deferred') is True for g in gate['decisions']), 'gate_declared_defer_count')
    counts['observer_native_none_all_statuses'] = sum(a.get('returned_none') is True for a in attempts)
    counts['requests_with_multiple_successful_selected_restores'] = sum(n > 1 for n in success_by_id.values())
    results = {}
    for sid in source_ids:
        o, q = out_by[sid], q_by[sid]
        arrival, times = o.get('arrival_s', 0.0), o['token_times_s']
        ttft = times[0] - arrival if times else None
        flow = o['host_elapsed_s'] - arrival if o.get('host_elapsed_s') is not None else None
        check(o['finished'] is True and o['external_request_id'] == 'measured/' + sid and len(o['output_token_ids']) == q['output_tokens']
              and o['finished'] == q['completed'] and o['finish_reason'] == q['finish_reason']
              and ttft == q['ttft_s'] and flow == q['completion_s'], 'final_output_quality_timing:' + sid)
        results[sid] = dict(request_id=sid, full_output_text=o['output_text'], output_token_ids=o['output_token_ids'],
            finished=o['finished'], finish_reason=o['finish_reason'], stop_reason=o['stop_reason'],
            arrival_s=arrival, ttft_s=ttft, flow_s=flow, quality=q)
    return dict(schema='c-math-restore-order-evidence-v1', integrity='VALID' if not issues else 'INVALID', issues=issues,
        run=str(run.resolve()), analysis=str(analysis_path.resolve()), mode=mode, counts=dict(counts),
        input_sha256=hashes, analysis_sha256=hashlib.sha256(analysis_bytes).hexdigest(),
        analyzer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        policy_step_minus_observer_call_id=offset, queried_candidates_per_decision=dict(sorted(query_hist.items())),
        successful_selected_restores_by_source_request=dict(success_by_id), decisions=rows,
        joined_allocation_receipts=joined, final_results_by_source_request=results,
        limitations=['Observed selection, allocation, final schedule and request outcomes; no causal speedup or quality claim.',
            'Repeated restores join by call and native request, not by globally unique successful request.',
            'Candidate queries cover only the bounded PREEMPTED prefix; unqueried feasibility/cost is unknown.',
            'Policy deferrals are not native allocation None; full-known-plus-one fit is not a future progress guarantee.',
            'Final outcomes are per request and must not be counted once per restoration; host times are not device latency.'])


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('run', 'analysis', 'output'):
        p.add_argument('--' + key, type=Path, required=True)
    args = p.parse_args()
    result = analyze(args.run, args.analysis)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps(dict(integrity=result['integrity'], counts=result['counts'], issues=result['issues'])))
    sys.exit(0 if result['integrity'] == 'VALID' else 2)
