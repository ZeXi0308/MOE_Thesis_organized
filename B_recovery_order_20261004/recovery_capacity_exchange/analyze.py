#!/usr/bin/env python3
"""Frozen full-service metrics plus one observed native donor preemption."""
import argparse
import bisect
from collections import Counter
import hashlib
import importlib.util
import inspect
import json
import math
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
PARENT = BASE/'recovery_service_age/analyze.py'
PARENT_SHA = '0e983a65db8ec6deadc458afe75d879613bf3a47c8bb073a357b9c27d0c22af1'
EXPECTED = ['stall8', 'exchange_once', 'exchange_once', 'stall8']
ARTIFACT = 'recovery-capacity-exchange'
SEMANTICS = (
    'Both arms install the frozen stall8 baseline. Its complete nested baseline_stall8 record is parsed '
    'with its real stall8 mode and original episode/eight-bypass guards, separately from the outer policy. '
    'An exchange is one native donor preemption, not a trigger, protection entry or peer hold. Actual '
    'preemption is cross-checked against raw native-preemption and capacity-handoff observations. '
    'Target, donor and the recorded pre-exchange capacity peers retain whole-request outcomes and later '
    'preemptions. Peer membership is a recorded selection cohort, not the full causal influence set. '
    'Receipt/release/repreemption comparisons use host observations on the existing raw origin; '
    'schedule records are not GPU execution. Sixteen scheduler entries are not elapsed time or an '
    'extra-delay estimate. Full-service failures, unfinished requests, maxgap mean/P99/max, TTFT, flow, '
    'throughput, drain, fixed-output content, recovery/copy work, passive GC and materialization retain '
    'frozen calculations. Joint SLO is diagnostic. Copy/GC durations cannot be added as request savings. '
    'Recorded plan/config hashes establish this group input identity, without a hard-coded historical '
    'input hash. Pair comparisons are whole-run descriptive contrasts, not matched runtime states or '
    'independent request repetitions. Missing execution evidence remains unknown; shadow decisions '
    'have no alternative executed trajectory.')


def load_service():
    if hashlib.sha256(PARENT.read_bytes()).hexdigest() != PARENT_SHA:
        raise RuntimeError('Frozen service-age analyzer changed')
    spec = importlib.util.spec_from_file_location('exchange_service_analysis', PARENT)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def group_helpers(repeat):
    parent = load_service().load_parent()
    source = inspect.getsource(parent.group_helpers).replace('repeat8', 'stall8').replace('unique8', 'exchange_once')
    source = source.replace("'(once|stall8)'", "'(once|repeat8)'")
    namespace = dict(vars(parent))
    exec(compile(source, '<exchange-modes-and-stall8-reference>', 'exec'), namespace)
    return namespace['group_helpers'](repeat)


def nested_actions(data, raw, cell, allocations, parser):
    # Only the genuinely installed nested baseline is parsed as stall8; the outer policy is never aliased.
    nested = data.get('baseline_stall8') if isinstance(data, dict) else None
    result = load_service().actions(nested, raw, dict(cell, mode='stall8'), allocations, parser)
    result['artifact_schema_path'] = ARTIFACT+'.json.baseline_stall8'
    return result


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


def input_identity(plan, config):
    expected = (plan.get('fixed') or {}).get('workload_sha256'); observed = config.get('workload_sha256')
    known = lambda value: isinstance(value, str) and len(value)==64 and all(ch in '0123456789abcdef' for ch in value)
    return dict(status='UNAVAILABLE' if not known(expected) or not known(observed) else 'MATCH' if expected==observed else 'MISMATCH',
        plan_canonical_workload_sha256=expected, config_canonical_workload_sha256=observed,
        semantics='This receipt.plan.fixed and cell config canonical input identity; missing evidence does not become a match.')


def request_outcome(rid, role, stamp, release, raw, cell, repeat, full=False):
    origin = raw['measurement_origin_perf_counter_s']; relative = stamp-origin
    source = raw.get('internal_to_source', {}).get(rid)
    request = next((row for row in raw.get('requests', []) if source is not None and row['request_id']==source), None)
    outputs = request.get('token_times_s', []) if request else []
    index = bisect.bisect_right(outputs, relative); first = outputs[index] if index < len(outputs) else None
    preempts = raw.get('preemption_events')
    later = [event for event in preempts if event.get('internal_request_id')==rid
             and event.get('original_preemption_returned') is True and event['method_entered_s'] > relative] if isinstance(preempts, list) else None
    next_preempt = later[0]['method_entered_s'] if later else None
    result = dict(role=role, internal_request=rid, source_request=source,
        status='AVAILABLE' if request else 'UNAVAILABLE',
        whole_request_metrics=next((row for row in cell.get('per_request', []) if source is not None and row['request']==source), None),
        whole_request_recovery_summary=next((row for row in cell.get('run_summary', {}).get('per_request', []) if source is not None and row['request']==source), None),
        first_client_receipt_after_action_s=first, later_successful_preemptions=later,
        first_later_preempt_s=next_preempt,
        receipt_before_release=first < release if first is not None and number(release) else None,
        receipt_before_later_preempt=first < next_preempt if first is not None and next_preempt is not None else None,
        no_later_preempt_observed=False if later else True if later is not None else None)
    if full:
        result['native_recovery_evidence'] = repeat.request_evidence(rid, stamp, raw, cell, cell.get('recovery_allocations'))
    return result


def exchange_actions(data, raw, observer, recovery, cell, repeat):
    if not isinstance(data, dict) or raw is None:
        return dict(status='UNAVAILABLE', missing=[key for key, value in ((ARTIFACT, data), ('raw', raw)) if value is None],
                    actual_donor_preemptions_verified=None, request_outcomes=[])
    failures, missing = [], []
    events = data.get('events')
    if not isinstance(events, list): events=[]; missing.append('events')
    kinds = Counter(row.get('kind') for row in events)
    decisions = [row for row in events if row.get('kind')=='decision']
    actions = [row for row in events if row.get('kind')=='action']
    releases = [row for row in events if row.get('kind')=='release']
    if data.get('mode') != cell['mode']: failures.append('mode_mismatch')
    if data.get('status') != 'UNINSTALLED': missing.append('uninstalled_observation')
    if data.get('status')=='ERROR' or any(row.get('error') for row in events): failures.append('policy_error')
    if type(data.get('action_count')) is not int or data['action_count'] not in (0,1) or data['action_count'] != len(actions):
        failures.append('donor_action_count_mismatch')
    if len(decisions)>1 or len(actions)>1: failures.append('one_exchange_limit_exceeded')
    if cell['mode']=='stall8' and (actions or releases or kinds['peer_hold'] or kinds['protection_entry']):
        failures.append('reference_performed_exchange_or_protection')
    if data.get('max_protected_rounds') != 16: failures.append('protection_bound_mismatch')
    if decisions and decisions[0].get('requested_action') != (cell['mode']=='exchange_once'):
        failures.append('requested_action_mode_mismatch')
    origin = raw['measurement_origin_perf_counter_s']; evidence = []; outcomes = []
    verified = 0 if isinstance(raw.get('preemption_events'), list) and isinstance(observer, dict) else None
    for action in actions:
        decision = decisions[0] if len(decisions)==1 else {}
        required = ('host_perf_s','target','donor','step','free_before','free_after','released_blocks','native_preempt_called')
        absent = [key for key in required if key not in action]
        missing.extend('action.'+key for key in absent)
        if absent or not number(decision.get('host_perf_s')):
            missing.append('decision_action_boundary'); continue
        stamp, end = decision['host_perf_s'], action['host_perf_s']
        if (not number(end) or end < stamp or action['native_preempt_called'] is not True
                or any(action[key] != decision.get(key) for key in ('target','donor','step'))):
            failures.append('action_identity_or_boundary_mismatch'); continue
        donors = [row for row in decision.get('donors', []) if row.get('request')==action['donor']]
        if (len(donors)!=1 or not all(number(action[key]) for key in ('free_before','free_after','released_blocks'))
                or action['free_after']-action['free_before'] != action['released_blocks']
                or action['released_blocks'] != donors[0].get('immediate_releasable_blocks')):
            failures.append('recorded_native_release_mismatch')
        preempts = raw.get('preemption_events')
        joined = [row for row in preempts if row.get('internal_request_id')==action['donor']
            and row.get('original_preemption_called') is True and row.get('original_preemption_returned') is True
            and stamp-origin <= row['method_entered_s'] <= row['method_returned_s'] <= end-origin] if isinstance(preempts,list) else None
        capacity = [row for row in observer.get('events', []) if row.get('kind')=='preempt'
            and row.get('request')==action['donor'] and stamp <= row['begin_host_perf_s'] <= row['end_host_perf_s'] <= end] if isinstance(observer,dict) else None
        if joined is None or capacity is None: missing.append('native_preempt_observers')
        elif len(joined)!=1 or len(capacity)!=1: failures.append('native_preempt_trace_join_not_unique')
        else:
            before, after = capacity[0].get('before',{}), capacity[0].get('after',{})
            if before.get('status')!='RUNNING' or after.get('status')!='PREEMPTED' or before.get('free_gpu_blocks')!=action['free_before'] or after.get('free_gpu_blocks')!=action['free_after']:
                failures.append('native_preempt_capacity_transition_mismatch')
            else: verified += 1
        release = releases[0] if len(releases)==1 else None
        release_s = release['host_perf_s']-origin if release and number(release.get('host_perf_s')) else None
        if not release: missing.append('single_release_record')
        else:
            entries = release.get('protected_entries'); selected = release.get('selected_step')
            if (release.get('target') != action['target'] or selected!=action['step'] or type(entries) is not int
                    or entries!=release.get('step',-1)-selected or not 0<=entries<=16
                    or (release.get('reason')=='SIXTEEN_ROUND_LIMIT' and entries!=16)):
                failures.append('release_identity_or_entry_bound_mismatch')
        cohort = decision.get('joint_capacity_before_exchange',{}).get('per_request')
        peers = [row['request'] for row in cohort] if isinstance(cohort,list) else []
        if not isinstance(cohort,list): missing.append('joint_capacity_before_exchange.per_request')
        roles = [(action['target'],'target'),(action['donor'],'donor')]+[(rid,'selected_peer') for rid in peers if rid not in (action['target'],action['donor'])]
        for rid, role in roles:
            outcomes.append(request_outcome(rid,role,end,release_s,raw,cell,repeat,full=role!='selected_peer'))
        target = next(row for row in outcomes if row['role']=='target')
        if release and release.get('reason')=='FIRST_CLIENT_RECEIPT' and target['receipt_before_release'] is not True:
            failures.append('receipt_release_not_supported_by_raw_output')
        evidence.append(dict(decision=decision, action=action, native_preempt_records=joined,
            capacity_preempt_records=capacity, release=release, release_s=release_s,
            decision_cpu_s=decision.get('decision_cpu_s'),
            sixteen_round_release_observed=release.get('reason')=='SIXTEEN_ROUND_LIMIT' if release else None,
            protection_lifetime_host_s=release['host_perf_s']-end if release_s is not None else None,
            target_schedule_records=[row for row in events if row.get('kind')=='target_scheduled'],
            existing_native_schedule_records=[row for row in recovery.get('events',[]) if row.get('kind')=='scheduled'
                and row.get('request')==action['target'] and row['host_perf_s']>=stamp] if isinstance(recovery,dict) else None))
    if verified is None: missing.append('raw_preemption_and_capacity_observers')
    if any(row['source_request'] is None for row in outcomes): missing.append('request_source_mapping')
    return dict(status='FAIL' if failures else 'UNVERIFIED' if missing else 'ANALYZED',
        failed_checks=sorted(set(failures)), missing=sorted(set(missing)), policy_status=data.get('status'),
        recorded_action_count=data.get('action_count'), actual_donor_preemptions_verified=verified,
        trigger_count=len(decisions), protection_entry_count=kinds['protection_entry'], peer_hold_count=kinds['peer_hold'],
        release_reasons=dict(Counter(row.get('reason') for row in releases)), actual_action_evidence=evidence,
        request_outcomes=outcomes, raw_policy_record={key:value for key,value in data.items() if key!='baseline_stall8'},
        semantics='Only action plus independent native observer joins verifies donor preemption. Triggers/holds are not forced preemptions. Post-action client receipt and later preemptions are actual observations; absent receipt remains unknown. Selection peers do not bound indirect effects.')


def analyze_session(session):
    service = load_service(); parent = service.load_parent()
    source = inspect.getsource(parent.analyze_session)
    for old,new in (('unique8','exchange_once'),('repeat8','stall8'),
        ('recovery_repeat_unique','recovery_capacity_exchange'),('repeat_unique_policy_evidence','capacity_exchange_policy_evidence'),
        ('effective_unique_execution_difference_count','effective_stall_execution_difference_count')):
        source = source.replace(old,new)
    source = source.replace('recovery_capacity_exchange_actions','baseline_stall8_actions')
    start = source.index("    result['recovery_capacity_exchange_semantics'] = (")
    end = source.index('    for name in (*PINS,',start)
    source = source[:start]+"    result['recovery_capacity_exchange_semantics'] = EXCHANGE_SEMANTICS\n"+source[end:]
    namespace = dict(vars(parent),group_helpers=group_helpers,inherited_actions=service.inherited_actions,
        actions=nested_actions,EXPECTED=EXPECTED,ARTIFACT=ARTIFACT,EXCHANGE_SEMANTICS=SEMANTICS)
    exec(compile(source,'<exchange-frozen-full-service>','exec'),namespace)
    result = namespace['analyze_session'](session)
    repeat = parent.load('recovery_repeat/analyze.py'); optional = repeat.ORIGINAL_OPTIONAL
    plan = (optional(session,'receipt') or {}).get('plan') or {}
    for cell in result['cells']:
        directory = Path(cell['directory'])
        raw,data,observer,recovery = [optional(directory,name) for name in ('raw',ARTIFACT,'capacity-handoff','recovery-order')]
        config = cell.get('resource_observations',{}).get('config') or {}
        cell['service_age_capture'] = service.capture_diagnostic(raw,config)
        cell['input_identity'] = input_identity(plan,config)
        cell['capacity_exchange_actions'] = exchange_actions(data,raw,observer,recovery,cell,repeat)
    by_path = {cell['directory']:cell for cell in result['cells']}
    for pair in result['comparisons']:
        identities = {role:by_path.get(pair.get(role),{}).get('input_identity') for role in ('stall8','exchange_once')}
        pair['input_identity'] = dict(status='MATCH' if all(row and row['status']=='MATCH' for row in identities.values())
            else 'MISMATCH' if any(row and row['status']=='MISMATCH' for row in identities.values()) else 'UNAVAILABLE',cells=identities)
        pair['baseline_stall8_execution_observations'] = pair.pop('execution_observations')
        pair['exchange_execution_observations'] = {role:{key:by_path[path]['capacity_exchange_actions'].get(key) for key in
            ('status','recorded_action_count','actual_donor_preemptions_verified','trigger_count','peer_hold_count','release_reasons')}
            if path in by_path else None for role,path in ((role,pair.get(role)) for role in ('stall8','exchange_once'))}
        changes = {row['request']:row for row in pair.get('per_request',[])}
        pair['selected_request_descriptive_contrasts'] = [dict(role=row['role'],source_request=row['source_request'],
            paired_whole_request_change=changes.get(row['source_request'])) for row in
            by_path.get(pair.get('exchange_once'),{}).get('capacity_exchange_actions',{}).get('request_outcomes',[])]
    result['execution_layout']['design'] = 'STALL8_VS_ONE_NATIVE_CAPACITY_EXCHANGE'
    result['analyzer_sources_sha256']['recovery_service_age/analyze.py'] = PARENT_SHA
    result['analyzer_sources_sha256']['recovery_repeat_unique/analyze.py'] = service.PARENT_SHA
    result['analyzer_sources_sha256']['recovery_capacity_exchange/analyze.py'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return result


def self_check():
    import tempfile
    repeat = load_service().load_parent().load('recovery_repeat/analyze.py')
    group,_,_ = group_helpers(repeat)
    cells = [dict(directory=f'/UNRUN/cell-{i:02d}-cap256-{mode}/output',status='RAW_UNAVAILABLE') for i,mode in enumerate(EXPECTED)]
    pairs = group.__globals__['comparisons'](cells)
    assert [(row['candidate'],row['native']) for row in pairs]==[(cells[1]['directory'],cells[0]['directory']),(cells[2]['directory'],cells[3]['directory'])]
    assert exchange_actions(None,None,None,None,dict(mode='exchange_once'),repeat)['actual_donor_preemptions_verified'] is None
    assert input_identity(dict(fixed=dict(workload_sha256='a'*64)),dict(workload_sha256='a'*64))['status']=='MATCH'
    with tempfile.TemporaryDirectory(prefix='exchange-analysis-empty-') as directory:
        result = analyze_session(Path(directory))
        assert result['cells']==[] and result['execution_layout']['expected_abba']==EXPECTED
    print('PASS: ABBA pairing, missing execution stays unknown, session input identity and empty inherited analyzer; no scientific data generated.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session',type=Path); parser.add_argument('--output',type=Path)
    parser.add_argument('--self-check',action='store_true'); args=parser.parse_args()
    if args.self_check: self_check()
    if args.session is None:
        if args.self_check:return
        parser.error('--session required')
    destination = args.output or args.session/'capacity-exchange-metrics.json'
    if destination.exists():raise FileExistsError(destination)
    result = analyze_session(args.session)
    with destination.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps(dict(output=str(destination),layout=result['execution_layout'])))


if __name__=='__main__':main()
