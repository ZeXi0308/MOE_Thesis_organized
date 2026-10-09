#!/usr/bin/env python3
"""Native/stall8 held-out-arrival comparison using frozen complete-service metrics."""
import argparse
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
PARENT = BASE/'recovery_service_age/analyze.py'
PARENT_SHA = '0e983a65db8ec6deadc458afe75d879613bf3a47c8bb073a357b9c27d0c22af1'
EXPECTED = ['native', 'stall8', 'stall8', 'native']
ARTIFACT = 'recovery-service-age-native'
SEMANTICS = (
    'Native observes the same frozen candidate/receipt suggestions but preserves the native head: '
    'NATIVE_OBSERVATION_ONLY, zero actual actions and no committed bypass episodes. Stall8 uses the '
    'frozen eight-action/episode guard and client-receipt ordering, including whole-set arrival fallback '
    'for an unknown receipt. Age8 remains a same-state suggestion diagnostic, not the performance '
    'reference in this group. Actual stall8 mutations versus native are reported separately from '
    'selection differences versus the age8 suggestion; shadow suggestions have no alternative output '
    'trajectory. Full request/status/failure/unfinished, maxgap mean/P99/max, TTFT/flow/throughput/drain, '
    'fixed-output content, copy/recovery, passive GC and phase/materialization calculations are inherited '
    'unchanged. Historical joint SLO remains diagnostic. The receipt capture is the same pinned extension '
    'in both modes. Input identity uses this session plan/config canonical workload hash, not an old '
    'arrival permutation. Native/stall8 nearest-run contrasts are descriptive, with runs as repetitions; '
    'request rows are not independent repeats, and copy/GC durations are not additive request savings.')


def load_service():
    if hashlib.sha256(PARENT.read_bytes()).hexdigest() != PARENT_SHA:
        raise RuntimeError('Frozen service-age analyzer changed')
    spec = importlib.util.spec_from_file_location('native_service_age_analysis', PARENT)
    service = importlib.util.module_from_spec(spec); spec.loader.exec_module(service)
    return service


def group_helpers(repeat):
    unique = load_service().load_parent()
    source = inspect.getsource(unique.group_helpers).replace('repeat8', 'native').replace('unique8', 'stall8')
    source = source.replace("'(once|native)'", "'(once|repeat8)'")
    namespace = dict(vars(unique))
    exec(compile(source, '<native-stall8-reference>', 'exec'), namespace)
    return namespace['group_helpers'](repeat)


def inherited_actions(repeat, evidence=None):
    source = repeat.replace_once(inspect.getsource(repeat.actions),
        "('recovery-repeat', data)", "('recovery-service-age-native', data)")
    source = repeat.replace_once(source,
        "        if limit is not None and completed_count >= limit: expected_reasons.append('BUDGET_EXHAUSTED')",
        "        if mode == 'native': expected_reasons.insert(0, 'NATIVE_OBSERVATION_ONLY')\n"
        "        if limit is not None and completed_count >= limit: expected_reasons.append('BUDGET_EXHAUSTED')")
    namespace = dict(vars(repeat), LIMITS={'native':8, 'stall8':8})
    if evidence is not None: namespace['request_evidence'] = evidence
    exec(compile(source, '<native-explicit-observation-parser>', 'exec'), namespace)
    return namespace['actions']


def actions(data, raw, cell, allocations, base_actions):
    result = load_service().actions(data, raw, cell, allocations, base_actions)
    if result['status'] == 'UNAVAILABLE': return result
    native = cell['mode'] == 'native'
    zero = all(result.get(key) == 0 for key in ('actual_completed_bypass_count',
        'queue_change_reported_count', 'queue_order_change_observed_count'))
    failures = result['failed_checks'][:]
    if native and (not zero or data.get('action_count') != 0 or data.get('bypassed_episodes') != []
            or any(row['action'] != 'SHADOW_ONLY' or row['action_skip_reasons'] != ['NATIVE_OBSERVATION_ONLY']
                   for row in result['opportunities'])):
        failures.append('native_observation_only_violated')
    result.update(status='FAIL' if failures else result['status'], failed_checks=sorted(set(failures)),
        native_observation_only_verified=(not failures and result['status']=='ANALYZED') if native else None,
        actual_mutations_versus_native=result['actual_completed_bypass_count'],
        selection_difference_reference='Existing age8-suggestion differences are same-observed-state diagnostics; the run-level reference is passive native, not age8.')
    return result


def input_identity(plan, config):
    expected = (plan.get('fixed') or {}).get('workload_sha256')
    observed = config.get('workload_sha256')
    known = lambda value: isinstance(value, str) and len(value)==64 and all(ch in '0123456789abcdef' for ch in value)
    return dict(status='UNAVAILABLE' if not known(expected) or not known(observed) else 'MATCH' if expected==observed else 'MISMATCH',
        plan_canonical_workload_sha256=expected, config_canonical_workload_sha256=observed,
        plan_inputs_dir=plan.get('inputs_dir'),
        semantics='Recorded plan/config canonical identity; no hard-coded old workload. Serialized file hashes are distinct. The frozen runtime input loader validates canonical workload content.')


def analyze_session(session):
    service = load_service()
    source = inspect.getsource(service.analyze_session)
    replacements = {
        "('repeat8', 'age8')": "('repeat8', 'native')",
        "('recovery_repeat_unique', 'recovery_service_age')": "('recovery_repeat_unique', 'recovery_service_age_native')",
        "('repeat_unique_policy_evidence', 'service_age_policy_evidence')": "('repeat_unique_policy_evidence', 'service_age_native_policy_evidence')",
        "'recovery_service_age_semantics'": "'recovery_service_age_native_semantics'",
        "'ARRIVAL_VS_CLIENT_RECEIPT_AGE_BUDGET8'": "'NATIVE_VS_CLIENT_RECEIPT_AGE_BUDGET8'",
    }
    for old, new in replacements.items():
        if old not in source: raise RuntimeError('Frozen service analysis boundary changed: '+old)
        source = source.replace(old, new)
    namespace = dict(vars(service), group_helpers=group_helpers, inherited_actions=inherited_actions,
                     actions=actions, EXPECTED=EXPECTED, ARTIFACT=ARTIFACT, SEMANTICS=SEMANTICS)
    exec(compile(source, '<native-stall8-frozen-full-service>', 'exec'), namespace)
    result = namespace['analyze_session'](session)
    receipt = service.load_parent().load('recovery_repeat/analyze.py').ORIGINAL_OPTIONAL(session, 'receipt') or {}
    plan = receipt.get('plan') or {}
    for cell in result['cells']:
        config = cell.get('resource_observations', {}).get('config') or {}
        cell['input_identity'] = input_identity(plan, config)
    by_path = {cell['directory']:cell for cell in result['cells']}
    for pair in result['comparisons']:
        identities = {role:by_path.get(pair.get(role), {}).get('input_identity') for role in ('native','stall8')}
        pair['input_identity'] = dict(status='MATCH' if all(value and value['status']=='MATCH' for value in identities.values())
            else 'MISMATCH' if any(value and value['status']=='MISMATCH' for value in identities.values()) else 'UNAVAILABLE',
            cells=identities, semantics='Only MATCH verifies the recorded shared input; raw outcomes/deltas are retained even when identity is unknown or mismatched.')
    result['analyzer_sources_sha256']['recovery_service_age/analyze.py'] = PARENT_SHA
    result['analyzer_sources_sha256']['recovery_service_age_native/analyze.py'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return result


def self_check():
    import tempfile
    service = load_service(); repeat = service.load_parent().load('recovery_repeat/analyze.py')
    group, _, _ = group_helpers(repeat)
    cells = [dict(directory=f'/UNRUN/cell-{i:02d}-cap256-{mode}/output',status='RAW_UNAVAILABLE') for i,mode in enumerate(EXPECTED)]
    pairs = group.__globals__['comparisons'](cells)
    assert [(p['candidate'],p['native']) for p in pairs]==[(cells[1]['directory'],cells[0]['directory']), (cells[2]['directory'],cells[3]['directory'])]
    parser = inherited_actions(repeat, lambda rid,*args:dict(source_request=rid))
    raw = dict(measurement_origin_perf_counter_s=0.,internal_to_source={rid:rid for rid in ('h','a','b')})
    event = dict(kind='fit_opportunity',host_perf_s=10.,baseline_head='h',candidate_head='b',
        final_head='h',queue_changed=False,waiting_before=['h','a','b'],waiting_after=['h','a','b'],
        candidate_num_preemptions=1,action='SHADOW_ONLY',action_count_before=0,action_count_after=0,
        action_limit=8,action_skip_reasons=['NATIVE_OBSERVATION_ONLY'],age8_suggestion='a',stall8_suggestion='b',
        age8_suggestion_num_preemptions=1,stall8_suggestion_num_preemptions=1,age8_would_execute=True,stall8_would_execute=True,
        receipt_fallback_reasons=[],stall8_fallback_reason=None,receipt_observation_host_perf_s=10.,
        candidates=[dict(request=rid,eligible=True,arrival_time=arrival,num_preemptions=1) for rid,arrival in [('a',1.),('b',2.)]],
        receipt_candidates=[dict(request=rid,arrival_time=arrival,num_preemptions=1,last_receipt_host_perf_s=stamp,
            stall_s=10.-stamp,known=True,unknown_reason=None) for rid,arrival,stamp in [('a',1.,7.),('b',2.,5.)]])
    data = dict(mode='native',status='UNINSTALLED',action_limit=8,action_count=0,shadow_count=1,bypassed_episodes=[],events=[event])
    result = actions(data,raw,dict(mode='native'),None,parser)
    assert result['status']=='ANALYZED' and result['native_observation_only_verified'] is True and result['rows']==[]
    candidate = json.loads(json.dumps(data));candidate.update(mode='stall8',action_count=1,bypassed_episodes=[dict(request='b',num_preemptions=1)])
    candidate['events'][0].update(action='REORDERED',action_skip_reasons=[],queue_changed=True,final_head='b',waiting_after=['b','h','a'],action_count_after=1)
    result = actions(candidate,raw,dict(mode='stall8'),None,parser)
    assert result['status']=='ANALYZED' and result['actual_mutations_versus_native']==1
    candidate['mode']='native';candidate['events'][0]['action_skip_reasons']=['NATIVE_OBSERVATION_ONLY']
    assert actions(candidate,raw,dict(mode='native'),None,parser)['status']=='FAIL'
    assert actions(None,raw,dict(mode='native'),None,parser)['status']=='UNAVAILABLE'
    new_hash = 'a'*64;plan = dict(fixed=dict(workload_sha256=new_hash))
    assert input_identity(plan,dict(workload_sha256=new_hash))['status']=='MATCH'
    assert input_identity(plan,dict(workload_sha256='b'*64))['status']=='MISMATCH'
    with tempfile.TemporaryDirectory(prefix='b-native-stall-analysis-') as directory:
        result = analyze_session(Path(directory))
        assert result['cells']==[] and result['execution_layout']['expected_abba']==EXPECTED
        assert len(result['execution_layout']['planned_but_not_started'])==4
    print('PASS: native explicit shadow/zero mutation, frozen stall8 action, mutation rejection, original ABBA pairing, missing raw and new plan/config input identity; empty adapter only. CPU fixtures, no scientific results.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session',type=Path);parser.add_argument('--output',type=Path)
    parser.add_argument('--self-check',action='store_true');args = parser.parse_args()
    if args.self_check:self_check()
    if args.session is None:
        if args.self_check:return
        parser.error('--session required')
    destination = args.output or args.session/'service-age-native-metrics.json'
    if destination.exists():raise FileExistsError(destination)
    result = analyze_session(args.session)
    with destination.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps(dict(output=str(destination),layout=result['execution_layout'])))


if __name__=='__main__':main()
