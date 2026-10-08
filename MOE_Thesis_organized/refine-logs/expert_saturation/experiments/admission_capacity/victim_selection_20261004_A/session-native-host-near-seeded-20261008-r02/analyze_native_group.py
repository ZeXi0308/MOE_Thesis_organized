#!/usr/bin/env python3
"""All-request native-victim ABBA comparisons; events are not repetitions."""
import argparse
import ast
from bisect import bisect_right
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from analyze import distribution, optional, pair, read, read_cell
from analyze_native_probe import count_known, native_summary
from analyze_pro import archive_dir, finite, policy_observations


RULES = ('tail', 'host_near', 'host_near', 'tail')
SEMANTICS = dict(
    primary='All-request mean external-arrival-to-completion flow, defined only for a completely finished cohort. The inherited incomplete-flow penalty is reported separately.',
    repeats='Two run-level blocks: cell01 versus cell00, then cell02 versus cell03. No event/request bootstrap, p-value, or significance claim.',
    roles='failed_request labels the allocation-failed prospective beneficiary, not a funding recovery target or proof of benefit. It can also be the selected victim.',
    causality='Independent end-to-end trajectories with identical external request IDs/prompts/arrivals. Role membership is policy-dependent. Cross-trajectory differences are descriptive, not same-state causal effects.',
    work='Natural output lengths, sequences, EOS and recovery counts may differ. Throughput is actual output rate, not equal-work speedup.',
    slo='No application SLO. Every inherited goodput frontier and its differences are diagnostic only; no threshold is selected.',
    cost='Candidate observation time is contained in selector time; never add them. Preempt method time excludes later native flush and is not DMA wait.',
    missing='Failed, unfinished and unavailable cells are retained. Missing values and missing policy guards are not filled with zero.',
)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def delta(a, b):
    return b-a if finite(a) and finite(b) else None


def measurement_jit_warnings(log_path):
    """Warnings within existing phase markers; no inferred compilation duration."""
    if not log_path.exists():
        return dict(status='LOG_UNAVAILABLE', warning_count=None)
    active, begins, ends, entries = False, 0, 0, []
    for number, line in enumerate(log_path.read_text(errors='replace').splitlines(), 1):
        if line == 'PHASE MEASUREMENT_BEGIN':
            begins += 1
            active = True
        elif line == 'PHASE MEASUREMENT_END':
            ends += 1
            active = False
        elif active and 'JIT compilation during inference:' in line:
            entries.append(dict(line_number=number, text=line))
    known = begins == ends == 1
    return dict(status='PHASE_MARKERS_COMPLETE' if known else 'PHASE_MARKERS_INCOMPLETE',
                warning_count=len(entries) if known else None, entries=entries,
                semantics='Observed JIT warnings only; logger coverage/deduplication may omit compilations. '
                          'Warning timestamps do not measure compilation duration or causal latency cost.')


def arrival_submission_and_drain(raw, expected_requests):
    if raw is None:
        return dict(status='RAW_UNAVAILABLE', expected_requests=expected_requests,
                    semantics='No capture: arrival, submission, rejection, timeout and drain counts are unknown, not zero.')
    requests = raw.get('requests', [])
    end = raw.get('observation_end_s')
    statuses = Counter(str(row.get('status', 'UNKNOWN')) for row in requests)
    rows = []
    for request in requests:
        arrival, admission = request.get('arrival_s'), request.get('admission_s')
        returned = request.get('engine_add_return_s')
        rows.append(dict(request_id=request.get('request_id'), status=request.get('status'),
            arrival_s=arrival, admission_s=admission, engine_add_return_s=returned,
            arrived_by_observation_end=(arrival <= end if finite(arrival) and finite(end) else None),
            client_submission_lag_s=delta(arrival, admission),
            engine_add_call_wall_s=delta(admission, returned)))
    all_arrivals_known = (len(rows) == expected_requests and
        len({row['request_id'] for row in rows}) == expected_requests and
        all(finite(row['arrival_s']) for row in rows))
    last_arrival = max((row['arrival_s'] for row in rows), default=None) if all_arrivals_known else None
    completed_times = [row['completion_s'] for row in requests
                      if row.get('status') == 'completed' and finite(row.get('completion_s'))]
    last_completion = max(completed_times, default=None)
    fully_completed = (all_arrivals_known and len(completed_times) == expected_requests
        and statuses.get('completed', 0) == expected_requests and raw.get('status') == 'COMPLETE'
        and raw.get('error') is None and finite(end)
        and all(row['arrived_by_observation_end'] is True for row in rows)
        and all(t <= end for t in completed_times))
    return dict(status='ANALYZED', expected_requests=expected_requests, recorded_requests=len(rows),
        missing_expected_records=max(0, expected_requests-len(rows)),
        arrived_by_observation_end=sum(row['arrived_by_observation_end'] is True for row in rows),
        not_yet_arrived_at_observation_end=sum(row['arrived_by_observation_end'] is False for row in rows),
        unknown_arrival_at_observation_end=sum(row['arrived_by_observation_end'] is None for row in rows),
        submission_started=sum(finite(row['admission_s']) for row in rows),
        submitted_engine_add_returned=sum(finite(row['engine_add_return_s']) for row in rows),
        completed=statuses.get('completed', 0), failed=statuses.get('failed', 0),
        unfinished=statuses.get('unfinished', 0), status_counts=dict(statuses),
        client_submission_lag_s=distribution([row['client_submission_lag_s'] for row in rows]),
        engine_add_call_wall_s=distribution([row['engine_add_call_wall_s'] for row in rows]),
        request_rejections=dict(status='NOT_SEPARATELY_RECORDED', count=None,
            explicit_rejected_status_records=statuses.get('rejected', 0)),
        request_timeouts=dict(status='NOT_SEPARATELY_RECORDED', count=None,
            explicit_timeout_status_records=sum(statuses.get(key, 0) for key in ('timeout', 'timed_out'))),
        capture_runtime_limit_hit=raw.get('error') == 'runtime_limit', capture_error=raw.get('error'),
        observation_end_s=end, last_external_arrival_s=last_arrival,
        last_observed_completion_s=last_completion,
        observation_end_minus_last_external_arrival_s=delta(last_arrival, end),
        last_observed_completion_minus_last_external_arrival_s=delta(last_arrival, last_completion),
        request_cohort_drain_status='FULLY_COMPLETED' if fully_completed else 'NOT_FULLY_COMPLETED_OR_UNKNOWN',
        full_request_drain_after_last_arrival_s=delta(last_arrival, last_completion) if fully_completed else None,
        per_request=rows,
        semantics='admission_s marks the start of synchronous engine.add_request, so admission minus external arrival is client submission lag, not native queue waiting or scheduler admission. engine_add_return minus admission is host call duration. Failed and unfinished are exact raw status counts; other statuses remain in status_counts. No separate exhaustive rejection or per-request timeout counter exists in this raw schema; they remain UNKNOWN and are not inferred from failed. runtime_limit is a capture-level stop, not an individual timeout classification. Last observed completion is not full drain unless the entire planned cohort completed; request drain does not assert connector/KV cleanup completion.')


def role(rid, roles):
    beneficiary, victim = rid in roles['allocation_failed_beneficiary'], rid in roles['victim']
    return ('beneficiary_and_victim' if beneficiary and victim else
            'allocation_failed_beneficiary' if beneficiary else 'victim' if victim else 'other')


def native_roles(decisions, raw):
    mapping = raw.get('internal_to_source', {})
    known = {row['request_id'] for row in raw.get('requests', [])}
    roles = dict(allocation_failed_beneficiary=set(), victim=set())
    rows = []
    for index, event in enumerate(decisions):
        row = dict(decision_index=index, step=event.get('step'), changed=event.get('changed'))
        for field, label in (('failed_request', 'allocation_failed_beneficiary'), ('selected', 'victim')):
            internal = event.get(field)
            rid = mapping.get(internal)
            row[label] = dict(internal_request_id=internal, request_id=rid,
                             join_status='JOINED' if rid in known else 'MISSING_REQUEST_JOIN')
            if rid in known:
                roles[label].add(rid)
        rows.append(row)
    return dict(role_request_ids={key: sorted(values) for key, values in roles.items()},
                per_decision=rows, missing_role_joins=sum(
                    row[label]['join_status'] != 'JOINED' for row in rows for label in roles))


def native_event_outcomes(decisions, raw, residency_admissions=None):
    """Observed absolute waits on each executed trajectory, with no event pairing."""
    requests = {r['request_id']: r for r in raw.get('requests', [])}
    mapping = raw.get('internal_to_source', {})
    outputs = defaultdict(list)
    for output in raw.get('output_events', []):
        outputs[output['request_id']].append(output)
    times = {rid: [e['received_s'] for e in events] for rid, events in outputs.items()}
    if any(values != sorted(values) for values in times.values()):
        return dict(status='INVALID_OUTPUT_TIME_ORDER')
    admissions = defaultdict(list)
    for admission in residency_admissions or []:
        if finite(admission.get('host_perf_counter_s')):
            admissions[admission.get('request')].append(admission)
    for values in admissions.values():
        values.sort(key=lambda row: row['host_perf_counter_s'])
    result = []
    for index, event in enumerate(decisions):
        cutoff = delta(raw.get('measurement_origin_perf_counter_s'), event.get('host_perf_counter_s'))
        record = dict(decision_index=index, step=event.get('step'), decision_s=cutoff,
                      self_preemption=event.get('failed_request') == event.get('selected'))
        for field, label in (('failed_request', 'allocation_failed_beneficiary'), ('selected', 'victim')):
            internal = event.get(field)
            rid = mapping.get(internal)
            row = dict(internal_request_id=internal, request_id=rid)
            if rid not in requests or cutoff is None:
                row['status'] = 'MISSING_REQUEST_OR_DECISION_CLOCK'
            else:
                request = requests[rid]
                candidates = [c for c in event.get('candidates', []) if c.get('request') == internal]
                ordinal = candidates[0].get('output_tokens') if len(candidates) == 1 else None
                position = bisect_right(times.get(rid, []), cutoff)
                following = outputs[rid][position] if position < len(outputs[rid]) else None
                previous = outputs[rid][position-1] if position else None
                row.update(status='OBSERVED' if following else 'NO_LATER_OUTPUT_IN_CAPTURE',
                    decision_output_ordinal=ordinal,
                    last_host_output_cumulative_tokens=previous.get('cumulative_tokens') if previous else None,
                    last_host_output_s=previous.get('received_s') if previous else None,
                    next_host_output_cumulative_tokens=following.get('cumulative_tokens') if following else None,
                    next_host_output_s=following.get('received_s') if following else None,
                    next_host_output_after_decision_s=delta(cutoff, following.get('received_s')) if following else None,
                    next_host_output_from_external_arrival_s=delta(request.get('arrival_s'), following.get('received_s')) if following else None,
                    next_host_output_engine_call=following.get('engine_call_index') if following else None,
                    completion_after_decision_s=delta(cutoff, request.get('completion_s')),
                    completion_flow_s=delta(request.get('arrival_s'), request.get('completion_s')),
                    request_status=request.get('status'), stop_reason=request.get('stop_reason'),
                    observed_outputs=len(request.get('output_token_ids', [])))
                if label == 'victim':
                    admission = next((a for a in admissions[internal]
                        if a['host_perf_counter_s'] > event['host_perf_counter_s']), None)
                    joined = dict(status=('JOINED' if admission else 'NOT_RECORDED'
                        if residency_admissions is None else 'NO_LATER_ADMISSION_IN_CAPTURE'))
                    if admission:
                        admission_s = delta(raw.get('measurement_origin_perf_counter_s'), admission['host_perf_counter_s'])
                        ready = candidates[0].get('host_ready_prefix_blocks') if len(candidates) == 1 else None
                        held, output = admission.get('held_blocks'), admission.get('output_count')
                        after_admission = delta(admission_s, following.get('received_s')) if following else None
                        joined.update(step=admission.get('step'), host_perf_counter_s=admission['host_perf_counter_s'],
                            admission_s=admission_s, held_blocks=held, output_count=output,
                            preempt_host_ready_prefix_blocks=ready,
                            output_ordinal_unchanged=(output == ordinal if count_known(output) and count_known(ordinal) else None),
                            held_below_preempt_ready=(held < ready if count_known(held) and count_known(ready) else None),
                            decision_to_admission_s=delta(cutoff, admission_s),
                            admission_to_next_output_s=after_admission if after_admission is not None and after_admission >= 0 else None,
                            next_output_precedes_admission=after_admission < 0 if after_admission is not None else None)
                    row['next_residency_admission'] = joined
            record[label] = row
        result.append(record)
    summaries = {label: dict(status_counts=dict(Counter(row[label]['status'] for row in result)),
        next_host_output_after_decision_s=distribution([
            row[label].get('next_host_output_after_decision_s') for row in result]))
        for label in ('allocation_failed_beneficiary', 'victim')}
    resumed = [row['victim'].get('next_residency_admission', {}) for row in result]
    joined = [row for row in resumed if row.get('status') == 'JOINED']
    positive_ready = [row for row in joined if count_known(row.get('preempt_host_ready_prefix_blocks'))
                      and row['preempt_host_ready_prefix_blocks'] > 0]
    persistence = dict(status_counts=dict(Counter(row.get('status', 'MISSING_REQUEST_OR_DECISION_CLOCK') for row in resumed)),
        output_ordinal_unchanged=sum(row['output_ordinal_unchanged'] is True for row in joined),
        unknown_output_ordinal=sum(row['output_ordinal_unchanged'] is None for row in joined),
        positive_preempt_host_ready=len(positive_ready),
        unknown_preempt_host_ready=sum(not count_known(row.get('preempt_host_ready_prefix_blocks')) for row in joined),
        positive_ready_with_smaller_next_held=sum(row['held_below_preempt_ready'] is True for row in positive_ready),
        positive_ready_with_unknown_next_held=sum(row['held_below_preempt_ready'] is None for row in positive_ready),
        decision_to_admission_s=distribution([row.get('decision_to_admission_s') for row in joined]),
        admission_to_next_output_s=distribution([row.get('admission_to_next_output_s') for row in joined]),
        semantics='First later residency admission with the same internal request ID, not a reconstructed load event. Next held pages below the earlier ready prefix rules out retaining that full prefix in GPU capacity at this admission; actual cache hits, recomputation, and copy cost remain UNKNOWN. Decision-to-admission includes queueing and native load waiting; admission-to-next-output also includes ordinary scheduling. Neither interval isolates transfer cost.')
    return dict(status='ANALYZED', per_decision=result, roles=summaries,
        victim_next_admission=persistence,
        self_preemption_decisions=sum(row['self_preemption'] for row in result),
        semantics='Decision clock is host_perf_counter_s minus measurement origin. The next output is the first strictly later host-return output event, with both engine ordinal at selection and next cumulative host output recorded. These are absolute observed waits, not incremental effects; failed_request is only a prospective beneficiary, may self-preempt, and is not a funding target. No cross-run event match or imputation after EOS.')


def least_generated_shadow(decisions):
    records = []
    for index, event in enumerate(decisions):
        rows = event.get('candidates', [])
        tail = next((row for row in rows if row.get('request') == event.get('native_tail')), None)
        item = dict(decision_index=index, step=event.get('step'))
        if (tail is None or type(event.get('fallback_unknown')) is not bool
                or any(type(row.get('qualified')) is not bool for row in rows)
                or event['fallback_unknown'] != any(not row['qualified'] for row in rows)):
            item['status'] = 'UNKNOWN_OR_INCONSISTENT_QUALIFICATION'
        elif event['fallback_unknown']:
            item.update(status='UNKNOWN_SUFFIX_FALLBACK', selected=tail['request'])
        elif any(type(row.get('output_tokens')) is not int or row['output_tokens'] < 0
                 or type(row.get('index')) is not int for row in rows):
            item['status'] = 'UNKNOWN_OUTPUT_OR_INDEX'
        else:
            selected = min(rows, key=lambda row: (row['output_tokens'], -row['index']))
            item.update(status='RANKED', selected=selected['request'])
        if 'selected' in item:
            item.update(differs_from_tail=item['selected'] != event.get('native_tail'),
                        differs_from_executed=item['selected'] != event.get('selected'))
        records.append(item)
    return dict(status_counts=dict(Counter(row['status'] for row in records)),
        known_decisions=sum('selected' in row for row in records),
        differs_from_tail=sum(row.get('differs_from_tail') is True for row in records),
        differs_from_executed=sum(row.get('differs_from_executed') is True for row in records),
        raw_shadows=records,
        semantics='Read-only suffix shadow: minimum produced output_tokens, then latest suffix index; any original unqualified suffix state falls back to native tail. This only checks whether the simple ranking remains equivalent on each executed trajectory, not its unrun end-to-end outcome.')


def native_selection_differences(decisions, actual_release):
    """Selected versus contemporaneous tail state; only selected release is observed."""
    joins = defaultdict(list)
    for joined in actual_release.get('joins', []):
        joins[joined.get('decision_index')].append(joined)
    records = []
    for index, event in enumerate(decisions):
        candidates = event.get('candidates', [])
        selected = [r for r in candidates if r.get('request') == event.get('selected')]
        tail = [r for r in candidates if r.get('request') == event.get('native_tail')]
        known_identity = event.get('selected') is not None and event.get('native_tail') is not None
        row = dict(decision_index=index, step=event.get('step'), selected=event.get('selected'),
            native_tail=event.get('native_tail'), changed_recorded=event.get('changed'),
            changed=event['selected'] != event['native_tail'] if known_identity else None,
            candidate_identity_status='UNIQUE' if len(selected) == len(tail) == 1 else 'MISSING_OR_AMBIGUOUS')
        selected, tail = (selected[0] if len(selected) == 1 else {}), (tail[0] if len(tail) == 1 else {})
        for field in ('held_blocks', 'host_missing_suffix_blocks', 'output_tokens'):
            a, b = selected.get(field), tail.get(field)
            row['selected_'+field], row['tail_'+field] = a, b
            row['selected_minus_tail_'+field] = a-b if count_known(a) and count_known(b) else None
        joined = joins[index][0] if len(joins[index]) == 1 else {}
        unique = (joined.get('join_status') == 'UNIQUE' and joined.get('step') == event.get('step')
                  and joined.get('selected') == event.get('selected'))
        row['release_join_status'] = joined.get('join_status') if unique else 'MISSING_OR_AMBIGUOUS'
        row['release_capacity_status'] = joined.get('capacity_status') if unique else None
        released = joined.get('actual_released_blocks') if unique else None
        row['actual_released_blocks'] = released
        row['actual_release_minus_tail_held_blocks'] = (released-tail['held_blocks']
            if count_known(released) and count_known(tail.get('held_blocks'))
            and joined.get('free_counter_delta_consistent') is not False else None)
        records.append(row)
    changed = [row for row in records if row['changed'] is True]
    fields = ('selected_minus_tail_held_blocks', 'selected_minus_tail_host_missing_suffix_blocks',
              'selected_minus_tail_output_tokens', 'actual_released_blocks', 'actual_release_minus_tail_held_blocks')
    aggregates = {}
    for field in fields:
        values = [row[field] for row in changed if row[field] is not None]
        aggregates[field] = dict(known=len(values), unknown=len(changed)-len(values),
            distribution=distribution(values), total=sum(values) if values else None)
    return dict(decisions=len(records), changed_decisions=len(changed),
        unchanged_decisions=sum(row['changed'] is False for row in records),
        unknown_change_decisions=sum(row['changed'] is None for row in records),
        recorded_change_disagreements=sum(row['changed'] is not None and row['changed'] != row['changed_recorded'] for row in records),
        changed_event_differences=aggregates, per_decision=records,
        semantics='Selected-minus-tail fields compare pre-action candidates in the same executed decision. actual_released_blocks reuses the exact decision_index release join. Actual release minus counterfactual tail held pages is NOT a measured causal difference between two releases: tail was not executed when selection changed, and held pages are only a release proxy. Extra released capacity remains a confound of any host-signal benefit. Aggregates are dependent changed-event descriptions, not independent repeats; unknown fields remain unknown.')


def load_choice(source):
    if source is None or not source.exists():
        return None, None
    tree = ast.parse(source.read_text())
    function = next((node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name == '_host_near_choice'), None)
    if function is None:
        return None, None
    namespace = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), 'exec'), namespace)
    return namespace['_host_near_choice'], hashlib.sha256(source.read_bytes()).hexdigest()


def verify_choices(decisions, source, executed_sha):
    choose, source_sha = load_choice(source)
    records = []
    for index, event in enumerate(decisions):
        item = dict(decision_index=index, step=event.get('step'), rule=event.get('rule'))
        rows = event.get('candidates', [])
        indices = [row.get('index') for row in rows]
        if (not rows or any(type(i) is not int for i in indices)
                or indices != list(range(indices[0], indices[-1]+1))
                or type(event.get('unprocessed_suffix_start')) is not int
                or indices[0] != event['unprocessed_suffix_start']
                or rows[-1].get('request') != event.get('native_tail')):
            item['status'] = 'INVALID_SUFFIX_IDENTITY'
        elif event.get('rule') == 'tail':
            item.update(status='MATCH' if (event.get('selected') == event.get('native_tail')
                        and event.get('changed') is False)
                        else 'MISMATCH', expected_selected=event.get('native_tail'))
        elif event.get('rule') != 'host_near':
            item['status'] = 'UNSUPPORTED_RULE'
        elif choose is None or source_sha != executed_sha:
            item['status'] = 'EXECUTED_POLICY_SOURCE_UNAVAILABLE'
        elif (type(event.get('fallback_unknown')) is not bool
                or type(event.get('active_protection_or_phase')) is not bool
                or any(type(row.get('qualified')) is not bool for row in rows)):
            item['status'] = 'REQUIRED_GUARDS_NOT_RECORDED'
        elif event['fallback_unknown'] != any(not row['qualified'] for row in rows):
            item['status'] = 'INCONSISTENT_RECORDED_QUALIFICATION'
        else:
            selected, reason = choose(rows, event['fallback_unknown'], event['active_protection_or_phase'])
            expected = next(row['request'] for row in rows if row['index'] == selected)
            matches = (expected == event.get('selected') and reason == event.get('host_near_fallback')
                       and event.get('changed') == (expected != event['native_tail']))
            item.update(status='MATCH' if matches else 'MISMATCH', expected_selected=expected,
                        expected_fallback=reason)
        records.append(item)
    return dict(source=str(source) if source else None, source_sha256=source_sha,
                executed_source_sha256=executed_sha,
                status_counts=dict(Counter(row['status'] for row in records)), checks=records,
                semantics='Executed rule recomputation uses the recorded original qualification and protection/phase guards. Raw host-missing shadows are not host_near decisions.')


def configuration_check(spec, cell, store, engine, profile, plan):
    config = cell.get('config') or {}
    checks = []
    def check(name, actual, expected):
        checks.append(dict(field=name, actual=actual, expected=expected, matches=actual == expected))
    rule = spec['native_victim_rule']
    check('config.native_victim_rule', config.get('native_victim_rule'), rule)
    check('store.native_victim_rule', store.get('native_victim_rule'), rule)
    check('plan.funding_victim_rule', spec.get('funding_victim_rule', 'tail'), 'tail')
    check('plan.legacy_victim_rule', spec.get('victim_rule', 'tail'), 'tail')
    for key, expected in dict(funding_victim_rule='tail', recovery_lease_mode='q1',
            recovery_min_outputs=1, oldest_admission_mode='queue_fund', oldest_repeat=True,
            ordinary_backfill=False, store_scope='native_full', commit_recheck=False,
            fit_first_resume=False, capacity_victim=False, yield_to_ready_head=False,
            spare_followup=False, capacity_deferral_mode='off').items():
        check(f'config.{key}', config.get(key), expected)
        check(f'store.{key}', store.get(key), expected)
    for key in ('native_victim_full_running_enabled', 'self_preempt_continue_enabled', 'current_victim_guard_enabled'):
        check(f'store.{key}', store.get(key), False)
    check('store.oldest_output_age_threshold_s', store.get('oldest_output_age_threshold_s'), 1.0)
    check('store.native_calc_overridden', store.get('native_calc_overridden'), False)
    check('config.requests', config.get('requests'), spec['requests'])
    check('config.input_case', config.get('input_case'), spec['input_case'])
    for key in ('max_num_seqs', 'max_num_batched_tokens', 'max_model_len'):
        if key in plan.get('configuration', {}):
            check(f'engine.{key}', engine.get(key), plan['configuration'][key])
    if plan.get('pinned_gpu_kv_bytes') is not None:
        pin = plan['pinned_gpu_kv_bytes']
        check('engine.kv_cache_memory_bytes', engine.get('kv_cache_memory_bytes'), pin)
        check('profile.actual_kv_storage_bytes', profile.get('actual_kv_storage_bytes'), pin)
    check('engine.async_scheduling', engine.get('async_scheduling'), False)
    check('engine.kv_offloading_backend', engine.get('kv_offloading_backend'), 'native')
    if 'model_revision' in plan:
        check('engine.revision', engine.get('revision'), plan['model_revision'])
        check('engine.tokenizer_revision', engine.get('tokenizer_revision'), plan['model_revision'])
    if 'max_seconds' in plan.get('configuration', {}):
        check('config.max_seconds', config.get('max_seconds'), plan['configuration']['max_seconds'])
    if 'host_kv_gib' in plan.get('configuration', {}):
        check('profile.host_kv_bytes', profile.get('host_kv_bytes'),
              plan['configuration']['host_kv_gib'] * 1024**3)
    check('all_recorded_decision_rules', sorted({e.get('rule') for e in store.get('victim_decisions', [])}),
          [rule] if store.get('victim_decisions') else [])
    return dict(status='VERIFIED' if all(c['matches'] for c in checks) else 'MISMATCH_OR_MISSING', checks=checks)


def read_native_cell(directory, spec, plan, timeout_s, policy_source=None):
    cell, raw = read_cell(directory, spec['native_victim_rule'], spec['requests'], timeout_s)
    archive = archive_dir(directory)
    store = optional(archive / 'selective-store.json', {})
    environment = optional(archive / 'environment.json', {})
    engine = optional(archive / 'engine_args.json', {})
    profile = optional(archive / 'normal_capacity_profile.json', {})
    cell.update(plan_cell=spec, engine_args=engine, capacity_profile=profile,
                measurement_jit_warnings=measurement_jit_warnings(directory / 'launch.log'),
                policy_observations=policy_observations(store),
                native_victim_observation=native_summary(store, raw),
                least_generated_shadow=least_generated_shadow(store.get('victim_decisions', [])),
                configuration_check=configuration_check(spec, cell, store, engine, profile, plan))
    cell['native_selection_differences'] = native_selection_differences(
        store.get('victim_decisions', []), cell['native_victim_observation'].get('actual_release', {}))
    source = directory / 'candidate_native_oldest_strong_r01/pkg/staged_store_rotation.py'
    source = source if source.exists() else policy_source
    cell['executed_choice_check'] = verify_choices(store.get('victim_decisions', []), source,
        environment.get('source_sha256', {}).get('staged_store_rotation.py'))
    cell['timing'] = optional(archive / 'timing.json', {})
    seed_record = optional(directory / 'triton-cache-seed.json')
    cell['triton_cache_seed'] = seed_record
    t = cell['timing']
    cell['duration_breakdown_s'] = {label: delta(t.get(start), t.get(end)) for label, start, end in (
        ('engine_init', 'engine_init_start_perf_s', 'engine_init_end_perf_s'),
        ('warmup', 'warmup_start_perf_s', 'warmup_end_perf_s'),
        ('measurement', 'measurement_start_perf_s', 'measurement_return_perf_s'),
        ('drain', 'measurement_return_perf_s', 'post_request_drain_end_perf_s'),
        ('process', 'process_start_perf_s', 'process_end_perf_s'))}
    warmup = []
    for path in sorted(archive.glob('warmup-[0-9]*.json')):
        value = read(path)
        identity = [{key: row.get(key) for key in ('request_id', 'prompt_token_ids_sha256',
                    'prompt_tokens', 'max_output_tokens', 'arrival_s')} for row in value.get('requests', [])]
        warmup.append(dict(file=path.name, status=value.get('status'), requests=len(identity),
                           input_contract_sha256=digest(identity)))
    cell['comparability_signature'] = dict(
        config={k: v for k, v in (cell.get('config') or {}).items() if k != 'native_victim_rule'},
        engine_args=engine, runtime_versions={k: environment.get(k) for k in ('python', 'torch', 'cuda', 'vllm', 'transformers')},
        runtime_source=environment.get('vllm_source_sha256'), source=environment.get('source_sha256'),
        warmup=warmup, warmup_cache_reset=optional(archive / 'warmup-cache-reset.json'),
        measured_capacity=profile)
    if plan.get('triton_cache_seed'):
        seed_signature = {key: (seed_record or {}).get(key) for key in
                          ('normalized_sha256', 'files', 'key_dirs', 'cache_initial')}
        cell['comparability_signature']['triton_cache_seed'] = seed_signature
        expected = plan['triton_cache_seed']
        checks = cell['configuration_check']['checks']
        for key in ('normalized_sha256', 'files', 'key_dirs'):
            checks.append(dict(field='triton_cache_seed.'+key, actual=seed_signature[key],
                               expected=expected[key], matches=seed_signature[key] == expected[key]))
        checks.append(dict(field='triton_cache_seed.cache_initial', actual=seed_signature['cache_initial'],
                           expected=plan['runtime_cache_initial'],
                           matches=seed_signature['cache_initial'] == plan['runtime_cache_initial']))
        cell['configuration_check']['status'] = ('VERIFIED' if all(c['matches'] for c in checks)
                                                  else 'MISMATCH_OR_MISSING')
    cell['arrival_submission_and_drain'] = arrival_submission_and_drain(raw, spec['requests'])
    if raw is None:
        return cell, raw
    cell['native_roles'] = native_roles(store.get('victim_decisions', []), raw)
    cell['native_event_outcomes'] = native_event_outcomes(store.get('victim_decisions', []), raw,
                                                        store.get('residency_admissions'))
    actual = [e for e in raw.get('preemption_events', [])
              if e.get('original_preemption_called') is True and e.get('original_preemption_returned') is True]
    counts = Counter(e['request_id'] for e in actual)
    cell['native_preemptions'] = dict(successful_calls=len(actual), per_request_counts=dict(counts),
        repeated_requests={rid: n for rid, n in counts.items() if n > 1},
        unconfirmed_or_failed_calls=len(raw.get('preemption_events', []))-len(actual))
    eos = optional(archive / 'resolved-eos.json', {})
    eos_ids = eos.get('hf_eos_token_id')
    eos_ids = set(eos_ids if isinstance(eos_ids, list) else [eos_ids]) - {None}
    cell['outputs'] = dict(stop_reason_counts=dict(Counter(str(r.get('stop_reason')) for r in raw['requests'])),
        reported_stop_requests=[r['request_id'] for r in raw['requests'] if r.get('stop_reason') == 'stop'],
        observed_final_eos_token_requests=[r['request_id'] for r in raw['requests']
            if r.get('output_token_ids') and r['output_token_ids'][-1] in eos_ids],
        eos_token_ids=sorted(eos_ids),
        semantics='Reported stop is not assumed to mean EOS. EOS tokens may be omitted by the engine; emitted final EOS IDs are only direct observations.')
    if 'metrics' in cell:
        cell['metrics']['frontier_role'] = 'DIAGNOSTIC_ONLY_NO_APPLICATION_SLO'
        cell['primary_mean_flow_s'] = (cell['metrics']['mean_completed_flow_s']
                                      if cell['status'] == 'COMPLETE' else None)
    else:
        cell['raw_requests_when_metrics_unavailable'] = raw['requests']
    return cell, raw


def effects(rows):
    return {key: dict(distribution=distribution([row[key] for row in rows]),
        worse=sum(row[key] is not None and row[key] > 0 for row in rows),
        better=sum(row[key] is not None and row[key] < 0 for row in rows),
        signed_sum_s=sum(row[key] for row in rows if row[key] is not None))
        for key in ('flow_difference_s', 'actual_completion_flow_difference_s',
                    'ttft_difference_s', 'max_gap_difference_s')}


def compare_cells(reference, candidate, old_raw, new_raw):
    result = pair(reference['metrics'], candidate['metrics'])
    result['diagnostic_goodput_frontier_differences'] = result.pop('frontier')
    result['legacy_penalized_flow_ratio'] = result.pop('mean_flow_ratio')
    old = {r['request_id']: r for r in reference['metrics']['requests']}
    new = {r['request_id']: r for r in candidate['metrics']['requests']}
    raw_old = {r['request_id']: r for r in old_raw['requests']}
    raw_new = {r['request_id']: r for r in new_raw['requests']}
    roles = [cell['native_roles']['role_request_ids'] for cell in (reference, candidate)]
    buckets, transitions = defaultdict(list), defaultdict(list)
    for row in result['per_request']:
        rid = row['request_id']
        row.update(reference_role=role(rid, roles[0]), candidate_role=role(rid, roles[1]),
            actual_completion_flow_difference_s=delta(old[rid]['actual_completion_flow_s'], new[rid]['actual_completion_flow_s']),
            output_sequence_changed=raw_old[rid]['output_token_ids'] != raw_new[rid]['output_token_ids'],
            stop_reason_reference=raw_old[rid].get('stop_reason'), stop_reason_candidate=raw_new[rid].get('stop_reason'))
        buckets[row['candidate_role']].append(row)
        transitions[row['reference_role']+'->'+row['candidate_role']].append(row)
    result['all_request_effects'] = effects(result['per_request'])
    result['candidate_role_effects'] = {key: dict(requests=len(rows), effects=effects(rows)) for key, rows in buckets.items()}
    result['role_transition_effects'] = {key: dict(requests=len(rows), effects=effects(rows)) for key, rows in transitions.items()}
    result['sequence_changed_requests'] = [r['request_id'] for r in result['per_request'] if r['output_sequence_changed']]
    result['primary_mean_flow_difference_s'] = delta(reference.get('primary_mean_flow_s'), candidate.get('primary_mean_flow_s'))
    result['summary_differences'] = {key: dict(reference=reference['metrics'].get(key),
        candidate=candidate['metrics'].get(key), difference=delta(reference['metrics'].get(key), candidate['metrics'].get(key)))
        for key in ('duration_s', 'actual_output_tokens_s', 'completed_requests_s', 'total_output_tokens',
                    'completed', 'failed', 'unfinished', 'ttft_median_s', 'ttft_p90_s', 'max_gap_request_p95_s')}
    result['distribution_differences'] = {metric: {q: delta(reference['metrics']['distributions'][metric][q],
        candidate['metrics']['distributions'][metric][q]) for q in ('mean', 'p50', 'p95', 'p99', 'maximum')}
        for metric in ('ttft_s', 'actual_completion_flow_s', 'max_gap_s')}
    result['config_differences'] = [key for key in reference['comparability_signature']
        if reference['comparability_signature'][key] != candidate['comparability_signature'].get(key)]
    result['both_complete'] = reference['status'] == candidate['status'] == 'COMPLETE'
    result['comparison_qualified'] = (not result['config_differences'] and all(
        cell['configuration_check']['status'] == 'VERIFIED'
        and not (set(cell['executed_choice_check']['status_counts']) - {'MATCH'})
        for cell in (reference, candidate)))
    return result


def analyze(session, timeout_s=600, policy_source=None):
    if not finite(timeout_s) or timeout_s <= 0:
        raise ValueError('Positive finite timeout required')
    plan = read(session / 'plan.json')
    specs = plan['cells']
    if (plan.get('kind') != 'PRO6000_NATIVE_HOST' or len(specs) != 4
            or tuple(spec.get('native_victim_rule') for spec in specs) != RULES
            or any(spec.get('profile_only') for spec in specs)
            or plan.get('arms') != [spec['label'] for spec in specs]
            or len(set(plan['arms'])) != 4
            or len({(spec.get('input_case'), spec.get('requests')) for spec in specs}) != 1):
        raise ValueError('Requires four measured native rules tail/host_near/host_near/tail and unique matching plan labels')
    cells, raws = {}, {}
    names = [f"cell-{i:02d}-{spec['label']}" for i, spec in enumerate(specs)]
    for name, spec in zip(names, specs):
        if Path(name).name != name:
            raise ValueError('Unsafe cell label')
        try:
            cells[name], raws[name] = read_native_cell(session/name, spec, plan, timeout_s, policy_source)
        except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
            cells[name], raws[name] = dict(status='ARTIFACT_ANALYSIS_ERROR', error=f'{type(exc).__name__}: {exc}'), None
    signatures = [cells[name].get('comparability_signature') for name in names]
    group_differences = ({key: [name for name, signature in zip(names, signatures)
        if signature.get(key) != signatures[0].get(key)] for key in signatures[0]}
        if all(signatures) else None)
    if group_differences is not None:
        group_differences = {key: different for key, different in group_differences.items() if different}
    group_comparable = group_differences == {}
    comparisons = {}
    for label, old, new in (('R01', 0, 1), ('R02_reverse', 3, 2)):
        a, b = names[old], names[new]
        row = dict(reference=a, candidate=b, status='UNAVAILABLE')
        if all(raws[k] is not None and 'metrics' in cells[k] for k in (a, b)):
            try:
                row.update(compare_cells(cells[a], cells[b], raws[a], raws[b]), status='PAIRED')
            except (ValueError, KeyError, TypeError) as exc:
                row.update(status='INVALID_PAIR', error=f'{type(exc).__name__}: {exc}')
        comparisons[label] = row
    eligible = [row for row in comparisons.values() if group_comparable
                and row.get('both_complete') and row.get('comparison_qualified')]
    differences = [row['primary_mean_flow_difference_s'] for row in eligible]
    return dict(schema_version=1, session=str(session), plan=plan, receipt=optional(session/'receipt.json'),
        semantics=SEMANTICS, analysis_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        status='FOUR_CELLS_COMPLETE_COMPARABLE' if len(eligible) == 2 else 'PARTIAL_FAILED_OR_NONCOMPARABLE',
        cells=cells, comparisons=comparisons,
        group_configuration=dict(status='CONSISTENT' if group_comparable else 'DIFFERENT_OR_MISSING',
            differences_from_cell00=group_differences),
        run_level_summary=dict(planned_blocks=2, qualified_complete_blocks=len(eligible),
            mean_flow_difference_s=distribution(differences),
            improving_blocks=sum(v < 0 for v in differences), worsening_blocks=sum(v > 0 for v in differences),
            statistical_unit='Complete paired runs; no confidence interval with only two exploratory blocks'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', type=Path, required=True)
    parser.add_argument('--timeout-s', type=float, default=600)
    parser.add_argument('--policy-source', type=Path,
                        default=Path(__file__).parent/'candidate_native_host_near_r01/pkg/staged_store_rotation.py')
    parser.add_argument('--output', type=Path, help='Omit to emit JSON to stdout; existing files are never overwritten')
    args = parser.parse_args()
    result = analyze(args.session, args.timeout_s, args.policy_source)
    if args.output:
        with args.output.open('x') as stream:
            json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write('\n')
    else:
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
