#!/usr/bin/env python3
"""Frozen complete-service metrics plus arrival/client-receipt selection evidence."""
import argparse
from collections import Counter
import hashlib
import importlib.util
import inspect
import json
import math
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
PARENT = BASE/'recovery_repeat_unique/analyze.py'
PARENT_SHA = '6fa51d0680e79fb67b20bdb3af2f9c32cf0ebcd10b76421fa3704601f14688cf'
EXPECTED = ['age8', 'stall8', 'stall8', 'age8']
ARTIFACT = 'recovery-service-age'
BASE_CAPTURE_SHA = '7c76da06f688192e68a13f21a0e1d003d816008bff52144e096d2cb5923b67be'
RECEIPT_CAPTURE_SHA = '0d71af580baa93c26d16516202e3321a2168654550e3fc7a6b668e4fcf5f3f21'
SEMANTICS = (
    'Both modes share the frozen legal recovery set, eight actual-bypass budget and per-request/episode guard. '
    'Age8 selects by original arrival then ID; stall8 selects the earliest last real client receipt, then arrival/ID. '
    'If any legal candidate lacks a finite strictly past receipt, stall8 falls back to the age suggestion. '
    'Selection precedes the unchanged episode/budget guard. Suggestions, fallbacks, completed mutations and '
    'age-executable/stall-episode-blocked suppressions are separate observations. Suppression eligibility uses '
    'the current arm\'s actual prior actions, not a maintained counterfactual trajectory. Only actual queue '
    'changes are joined to containing preemption episodes, native allocation, LOAD submit/host poll/ACK, '
    'schedule, next client output and later preemptions. Complete-service status/failure accounting, '
    'mean/P99/maxgap, TTFT, flow, throughput, drain, fixed-output contracts/content, copy work and recovery '
    'unions retain frozen calculations. Shared GC intervals and GPU copy durations are not additive request '
    'savings; post-observation compact materialization remains in phase/process costs. Historical joint SLO '
    'is diagnostic. Nearest-age8 run contrasts are descriptive, not matched-state effects or request-level '
    'independent repetitions. Missing receipt evidence is unknown, not zero fallback or zero intervention.')


def load_parent():
    if hashlib.sha256(PARENT.read_bytes()).hexdigest() != PARENT_SHA:
        raise RuntimeError('Frozen complete-service analyzer changed')
    spec = importlib.util.spec_from_file_location('service_age_frozen_analysis', PARENT)
    parent = importlib.util.module_from_spec(spec); spec.loader.exec_module(parent)
    return parent


def group_helpers(repeat):
    parent = load_parent()
    source = inspect.getsource(parent.group_helpers).replace('repeat8', 'age8').replace('unique8', 'stall8')
    source = source.replace("'(once|age8)'", "'(once|repeat8)'")  # Frozen source match, not the new mode regex.
    namespace = dict(vars(parent))
    exec(compile(source, '<service-age-modes-and-reference>', 'exec'), namespace)
    return namespace['group_helpers'](repeat)


def inherited_actions(repeat, evidence=None):
    # Both modes retain the original episode guard; do not inherit the unique-ID guard.
    source = repeat.replace_once(inspect.getsource(repeat.actions),
        "('recovery-repeat', data)", "('recovery-service-age', data)")
    namespace = dict(vars(repeat), LIMITS={'age8': 8, 'stall8': 8})
    if evidence is not None: namespace['request_evidence'] = evidence
    exec(compile(source, '<service-age-original-episode-parser>', 'exec'), namespace)
    return namespace['actions']


def capture_diagnostic(raw, config):
    if raw is None: return dict(status='UNAVAILABLE', missing=['raw'])
    record, storage = raw.get('service_age_capture'), raw.get('output_event_storage')
    if not isinstance(record, dict) or not isinstance(storage, dict):
        return dict(status='UNVERIFIED', missing=['raw.service_age_capture/output_event_storage'])
    observed = dict(config=config.get('service_age_capture_sha256'),
        receipt=record.get('compiled_source_sha256'), storage=storage.get('compiled_source_sha256'),
        base_compact=storage.get('base_compact_compiled_source_sha256'))
    missing = [key for key, value in observed.items() if value is None]
    failures = [key+'_source_mismatch' for key, value in observed.items()
                if value is not None and value != (BASE_CAPTURE_SHA if key=='base_compact' else RECEIPT_CAPTURE_SHA)]
    if type(record.get('receipt_id_count')) is not int or record['receipt_id_count'] < 0:
        missing.append('valid_receipt_id_count')
    return dict(status='FAIL' if failures else 'UNVERIFIED' if missing else 'ANALYZED',
        missing=missing, failed_checks=failures, observed_source_sha256=observed, record=record,
        semantics='Only the pinned post-valid-output receipt dictionary update extends frozen compact capture; no additional clock/CUDA observation. Receipt count is recorded metadata, not an independent execution proof. Original storage/materialization checks remain intact.')


def actions(data, raw, cell, allocations, base_actions):
    result = base_actions(data, raw, cell, allocations)
    if result['status'] == 'UNAVAILABLE': return result
    failures, missing = result['failed_checks'][:], result['missing'][:]
    episodes, used = set(), set()
    counts, fallback_reasons = Counter(), Counter()
    rows = {row['event_index']: row for row in result['rows']}
    selection_missing = False
    required = ('candidates', 'age8_suggestion', 'stall8_suggestion',
        'age8_suggestion_num_preemptions', 'stall8_suggestion_num_preemptions',
        'age8_would_execute', 'stall8_would_execute', 'receipt_fallback_reasons', 'stall8_fallback_reason',
        'receipt_candidates', 'receipt_observation_host_perf_s')
    number = lambda value: type(value) in (int, float) and math.isfinite(value)
    for opportunity in result['opportunities']:
        index = opportunity['event_index']; event = data['events'][index]
        selected = event['candidate_head']; completed = opportunity['completed_bypass']
        prior_episodes, prior_ids = episodes.copy(), used.copy()
        if completed:
            episodes.add((selected, event['candidate_num_preemptions'])); used.add(selected)
            counts['actual_repeated_request_bypasses'] += selected in prior_ids
        absent = [key for key in required if key not in event]
        if absent:
            missing.extend('event.'+key for key in absent); selection_missing = True; continue
        candidates, receipts = event['candidates'], event['receipt_candidates']
        candidate_keys = ('request', 'eligible', 'arrival_time', 'num_preemptions')
        receipt_keys = ('request', 'arrival_time', 'num_preemptions', 'last_receipt_host_perf_s',
                        'stall_s', 'known', 'unknown_reason')
        if (not isinstance(candidates, list) or any(not isinstance(row, dict) or any(k not in row for k in candidate_keys) for row in candidates)
                or not isinstance(receipts, list) or any(not isinstance(row, dict) or any(k not in row for k in receipt_keys) for row in receipts)):
            missing.append('event.complete_candidate_receipt_records'); selection_missing = True; continue
        legal = {row['request']: row for row in candidates if row['eligible'] is True}
        receipt_by_id = {row['request']: row for row in receipts}
        if (len(legal) != sum(row['eligible'] is True for row in candidates)
                or len(receipt_by_id) != len(receipts) or set(receipt_by_id) != set(legal) or not legal):
            failures.append('receipt_candidates_differ_from_legal_set'); continue
        now = event['receipt_observation_host_perf_s']
        if not number(now) or now != event['host_perf_s']:
            failures.append('receipt_observation_clock_mismatch'); continue
        unknown = []
        for rid, receipt in receipt_by_id.items():
            candidate = legal[rid]
            if (not number(candidate['arrival_time']) or type(candidate['num_preemptions']) is not int
                    or receipt['arrival_time'] != candidate['arrival_time']
                    or receipt['num_preemptions'] != candidate['num_preemptions']):
                failures.append('receipt_candidate_identity_or_arrival_mismatch')
            stamp, stall = receipt['last_receipt_host_perf_s'], receipt['stall_s']
            if receipt['known'] is True:
                if (not number(stamp) or not stamp < now or not number(stall)
                        or not math.isclose(stall, now-stamp, rel_tol=1e-12, abs_tol=1e-9)
                        or receipt['unknown_reason'] is not None):
                    failures.append('known_receipt_age_mismatch')
            elif receipt['known'] is False:
                if receipt['unknown_reason'] not in ('MISSING_RECEIPT', 'NONFINITE_RECEIPT', 'NOT_STRICTLY_PAST_RECEIPT') or stall is not None:
                    failures.append('unknown_receipt_reason_mismatch')
                unknown.append(receipt['unknown_reason'])
            else: failures.append('receipt_known_not_boolean')
        fallback = event['receipt_fallback_reasons']
        if fallback != sorted(set(unknown)):
            failures.append('receipt_fallback_reasons_mismatch')
        if event['stall8_fallback_reason'] != ('UNKNOWN_CANDIDATE_RECEIPT' if unknown else None):
            failures.append('stall_fallback_reason_mismatch')
        age, stall = event['age8_suggestion'], event['stall8_suggestion']
        age_expected = min(legal, key=lambda rid: (legal[rid]['arrival_time'], rid))
        known_receipts = all(row['known'] is True and number(row['last_receipt_host_perf_s']) for row in receipts)
        stall_expected = (min(legal, key=lambda rid: (receipt_by_id[rid]['last_receipt_host_perf_s'], legal[rid]['arrival_time'], rid))
                          if not unknown and known_receipts else age_expected)
        if age != age_expected or stall != stall_expected:
            failures.append('age_or_stall_selection_order_mismatch')
        executable = {}
        for label, suggested in (('age8', age), ('stall8', stall)):
            if suggested not in legal:
                failures.append(label+'_suggestion_not_legal'); executable[label] = False; continue
            episode = (suggested, legal[suggested]['num_preemptions'])
            if event[label+'_suggestion_num_preemptions'] != episode[1]: failures.append(label+'_episode_mismatch')
            executable[label] = event['action_count_before'] < 8 and episode not in prior_episodes
            if type(event[label+'_would_execute']) is not bool or event[label+'_would_execute'] != executable[label]:
                failures.append(label+'_would_execute_mismatch')
        if selected != (age if cell['mode'] == 'age8' else stall): failures.append('mode_selected_suggestion_mismatch')
        different = cell['mode'] == 'stall8' and completed and selected != age
        suppression = (cell['mode'] == 'stall8' and executable['age8'] and not executable['stall8']
            and event['action'] == 'SHADOW_ONLY' and not opportunity['queue_order_changed_observed']
            and event['action_skip_reasons'] == ['EPISODE_ALREADY_BYPASSED'])
        counts['suggestion_disagreement_opportunities'] += age != stall
        counts['suggestion_disagreements_age_would_execute'] += age != stall and executable['age8']
        counts['fallback_opportunities'] += bool(fallback)
        counts['fallback_completed_bypasses'] += bool(fallback) and completed
        counts['applied_stall_fallback_bypasses'] += cell['mode'] == 'stall8' and bool(fallback) and completed
        counts['fallback_shadow_opportunities'] += bool(fallback) and event['action'] == 'SHADOW_ONLY'
        counts['actual_different_reorder_count'] += different
        counts['stall_legal_suppression_count'] += suppression
        fallback_reasons.update(fallback)
        detail = dict(legal_candidate_requests=list(legal), age8_suggestion=age, stall8_suggestion=stall,
            age8_source_request=raw.get('internal_to_source', {}).get(age),
            stall8_source_request=raw.get('internal_to_source', {}).get(stall),
            age8_would_execute=event['age8_would_execute'], stall8_would_execute=event['stall8_would_execute'],
            receipt_observation_host_perf_s=now, receipt_candidates=receipts, receipt_fallback_reasons=fallback,
            stall8_fallback_reason=event['stall8_fallback_reason'],
            suggestions_differ=age != stall, actual_different_reorder=different, actual_legal_suppression=suppression,
            selected_request_previously_used=selected in prior_ids,
            same_state_age8_suggested_final_head=age if executable['age8'] else event['baseline_head'],
            observed_final_head=event['final_head'])
        opportunity.update(detail)
        if index in rows: rows[index].update(detail)
    keys = ('suggestion_disagreement_opportunities', 'suggestion_disagreements_age_would_execute',
            'fallback_opportunities', 'fallback_completed_bypasses', 'applied_stall_fallback_bypasses', 'fallback_shadow_opportunities',
            'actual_repeated_request_bypasses', 'actual_different_reorder_count', 'stall_legal_suppression_count')
    result.update(status='FAIL' if failures else 'UNVERIFIED' if missing else 'ANALYZED',
        failed_checks=sorted(set(failures)), missing=sorted(set(missing)), used_request_ids=sorted(used),
        selection_observations={key: None if selection_missing else counts[key] for key in keys},
        receipt_fallback_reason_counts=None if selection_missing else dict(fallback_reasons),
        fallback_count_semantics='Fallback-condition counts are shared diagnostics in both modes; applied_stall_fallback_bypasses counts only real stall8 mutations after falling back. Age8 already selects by arrival.',
        effective_stall_execution_difference_count=None if selection_missing else
            counts['actual_different_reorder_count']+counts['stall_legal_suppression_count'],
        completed_bypasses_below_budget=result['actual_completed_bypass_count'] < 8)
    return result


def analyze_session(session):
    parent = load_parent()
    source = inspect.getsource(parent.analyze_session)
    for old, new in (('unique8', 'stall8'), ('repeat8', 'age8'),
                     ('recovery_repeat_unique', 'recovery_service_age'),
                     ('repeat_unique_policy_evidence', 'service_age_policy_evidence'),
                     ('effective_unique_execution_difference_count', 'effective_stall_execution_difference_count'),
                     ('REPEAT_EPISODE_VS_UNIQUE_REQUEST_BUDGET8', 'ARRIVAL_VS_CLIENT_RECEIPT_AGE_BUDGET8')):
        source = source.replace(old, new)
    start = source.index("    result['recovery_service_age_semantics'] = (")
    end = source.index('    for name in (*PINS,', start)
    source = source[:start]+"    result['recovery_service_age_semantics'] = SERVICE_AGE_SEMANTICS\n"+source[end:]
    anchor = "        rows = cell.get('per_request', []); planned = (config or {}).get('requests', cell.get('planned'))"
    if source.count(anchor) != 1: raise RuntimeError('Frozen per-cell diagnostic boundary changed')
    source = source.replace(anchor, "        cell['service_age_capture'] = capture_diagnostic(raw, config or {})\n"+anchor)
    namespace = dict(vars(parent), group_helpers=group_helpers, inherited_actions=inherited_actions,
                     actions=actions, EXPECTED=EXPECTED, ARTIFACT=ARTIFACT, SERVICE_AGE_SEMANTICS=SEMANTICS,
                     capture_diagnostic=capture_diagnostic)
    exec(compile(source, '<service-age-frozen-complete-service>', 'exec'), namespace)
    result = namespace['analyze_session'](session)
    result['analyzer_sources_sha256']['recovery_repeat_unique/analyze.py'] = PARENT_SHA
    return result


def self_check():
    import tempfile
    parent = load_parent(); repeat = parent.load('recovery_repeat/analyze.py')
    group, _, _ = group_helpers(repeat)
    cells = [dict(directory=f'/UNRUN/cell-{i:02d}-cap256-{mode}/output', status='RAW_UNAVAILABLE') for i, mode in enumerate(EXPECTED)]
    pairs = group.__globals__['comparisons'](cells)
    assert [(p['candidate'],p['native']) for p in pairs] == [(cells[1]['directory'],cells[0]['directory']), (cells[2]['directory'],cells[3]['directory'])]
    parser = inherited_actions(repeat, lambda rid,*args:dict(source_request=rid))
    raw = dict(measurement_origin_perf_counter_s=0., internal_to_source={rid:rid for rid in ('h','a','b')})
    events = []; count = 0; episodes = set()
    for index in range(3):
        fallback = index == 2; age, stall = 'a', 'a' if fallback else 'b'
        selected = stall; changed = index != 1; now = 10.+index
        reasons = [] if changed else ['EPISODE_ALREADY_BYPASSED']
        receipts = [dict(request=rid,arrival_time=arrival,num_preemptions=1,
            last_receipt_host_perf_s=None if fallback and rid=='b' else 7. if rid=='a' else 5.,
            stall_s=None if fallback and rid=='b' else now-(7. if rid=='a' else 5.),
            known=not(fallback and rid=='b'),unknown_reason='MISSING_RECEIPT' if fallback and rid=='b' else None)
            for rid,arrival in [('a',1.),('b',2.)]]
        before = ['h','a','b']; after = [selected]+[rid for rid in before if rid!=selected] if changed else before[:]
        event = dict(kind='fit_opportunity',host_perf_s=now,baseline_head='h',candidate_head=selected,
            final_head=after[0],queue_changed=changed,waiting_before=before,waiting_after=after,
            candidate_num_preemptions=1,action='REORDERED' if changed else 'SHADOW_ONLY',
            action_count_before=count,action_count_after=count+int(changed),action_limit=8,action_skip_reasons=reasons,
            candidates=[dict(request=rid,eligible=True,arrival_time=arrival,num_preemptions=1) for rid,arrival in [('a',1.),('b',2.)]],
            age8_suggestion=age,stall8_suggestion=stall,age8_suggestion_num_preemptions=1,stall8_suggestion_num_preemptions=1,
            age8_would_execute=(age,1) not in episodes,stall8_would_execute=(stall,1) not in episodes,
            receipt_fallback_reasons=['MISSING_RECEIPT'] if fallback else [],receipt_candidates=receipts,
            stall8_fallback_reason='UNKNOWN_CANDIDATE_RECEIPT' if fallback else None,
            receipt_observation_host_perf_s=now)
        events.append(event)
        if changed:count+=1;episodes.add((selected,1))
    data = dict(mode='stall8',status='UNINSTALLED',action_limit=8,action_count=count,shadow_count=3,events=events)
    result = actions(data,raw,dict(mode='stall8'),None,parser)
    assert result['status']=='ANALYZED',result
    assert result['actual_completed_bypass_count']==2 and len(result['rows'])==2
    assert result['selection_observations']['actual_different_reorder_count']==1
    assert result['selection_observations']['stall_legal_suppression_count']==1
    assert result['selection_observations']['fallback_completed_bypasses']==1
    assert actions(None,raw,dict(mode='stall8'),None,parser)['status']=='UNAVAILABLE'
    missing = json.loads(json.dumps(data)); del missing['events'][0]['receipt_candidates']
    absent = actions(missing,raw,dict(mode='stall8'),None,parser)
    assert absent['status']=='UNVERIFIED' and absent['selection_observations']['fallback_opportunities'] is None
    bad = json.loads(json.dumps(data));bad['events'][0]['stall8_suggestion']='a'
    assert actions(bad,raw,dict(mode='stall8'),None,parser)['status']=='FAIL'
    capture = dict(service_age_capture=dict(compiled_source_sha256=RECEIPT_CAPTURE_SHA,receipt_id_count=2),
        output_event_storage=dict(compiled_source_sha256=RECEIPT_CAPTURE_SHA,base_compact_compiled_source_sha256=BASE_CAPTURE_SHA))
    assert capture_diagnostic(capture,dict(service_age_capture_sha256=RECEIPT_CAPTURE_SHA))['status']=='ANALYZED'
    assert capture_diagnostic(capture,dict(service_age_capture_sha256=BASE_CAPTURE_SHA))['status']=='FAIL'
    # Exercise the inherited full-session adapter only on an empty synthetic
    # session: no measured cells/results and no historical data reprocessing.
    with tempfile.TemporaryDirectory(prefix='b-service-age-analyzer-') as directory:
        empty = analyze_session(Path(directory))
        assert empty['cells']==[] and not empty['execution_layout']['complete_abba']
        assert empty['execution_layout']['expected_abba']==EXPECTED
        assert len(empty['execution_layout']['planned_but_not_started'])==4
    print('PASS: original pair mapping/full-service adapter; actual stall reorder, episode suppression and fallback separated; missing/invalid receipt evidence; no shadow outcome joins. CPU fixtures only, no scientific results.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--self-check',action='store_true')
    args = parser.parse_args()
    if args.self_check:self_check()
    if args.session is None:
        if args.self_check:return
        parser.error('--session required')
    destination = args.output or args.session/'service-age-metrics.json'
    if destination.exists():raise FileExistsError(destination)
    result = analyze_session(args.session)
    with destination.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps(dict(output=str(destination),layout=result['execution_layout'],cells=[dict(
        directory=cell['directory'],**{key:cell['recovery_service_age_actions'].get(key) for key in
        ('status','actual_completed_bypass_count','unique_bypassed_requests','selection_observations')}) for cell in result['cells']])))


if __name__ == '__main__':main()
