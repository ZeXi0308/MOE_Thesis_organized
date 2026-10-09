#!/usr/bin/env python3
"""All-request ABBA analysis of one equal-release Host-prefix victim replacement."""
import argparse
import ast
from collections import Counter, defaultdict
import json
from pathlib import Path

import analyze_native_arrival_once_group as once
import analyze_native_group as base
import analyze_native_mixed_budget_probe as mixed
import analyze_native_shared_prefix_probe as shared
import analyze_native_shared_prefix_profiled as profiled
import analyze_native_partial_restore_once_group as partial


HERE = Path(__file__).resolve().parent
PACKAGE = HERE / 'candidate_native_equal_release_host_once_r01'
KIND = 'PRO6000_NATIVE_EQUAL_RELEASE_HOST_ONCE'
ROLE = 'EQUAL_RELEASE_HOST_ONCE_INTERVENTION'
RULES = ('tail', 'equal_release_host_once', 'equal_release_host_once', 'tail')
FIELD = 'equal_release_host_once'


def verify_choices(decisions, source, executed_sha, block_size=None):
    choose, source_sha = None, None
    if source is not None and source.is_file():
        source_sha = mixed.sha(source)
        if source_sha == executed_sha:
            node = next((n for n in ast.parse(source.read_bytes()).body
                if isinstance(n, ast.FunctionDef) and n.name == '_equal_release_host_once_choice'), None)
            if node is not None:
                namespace = {}
                exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), namespace)
                choose = namespace[node.name]
    records, consumed, known = [], False, True
    triggers = proposals = requests = 0
    for index, event in enumerate(decisions):
        item = dict(decision_index=index, step=event.get('step'), rule=event.get('rule'))
        rows = event.get('candidates', [])
        indices = [r.get('index') for r in rows]
        if choose is None:
            item['status'] = 'EXECUTED_POLICY_SOURCE_UNAVAILABLE'
        elif not known:
            item['status'] = 'PRIOR_PROPOSAL_STATE_UNVERIFIABLE'
        elif (not rows or any(type(i) is not int for i in indices)
                or indices != list(range(indices[0], indices[-1] + 1))
                or type(event.get('unprocessed_suffix_start')) is not int
                or indices[0] != event['unprocessed_suffix_start']
                or rows[0].get('request') != event.get('failed_request')
                or rows[-1].get('request') != event.get('native_tail')
                or any(not isinstance(r.get('request'), str) for r in rows)
                or len({r['request'] for r in rows}) != len(rows)):
            item['status'], known = 'INVALID_SUFFIX_IDENTITY', False
        elif event.get('rule') not in RULES:
            item['status'], known = 'UNSUPPORTED_RULE', False
        elif type(event.get('active_protection_or_phase')) is not bool:
            item['status'], known = 'REQUIRED_GUARD_NOT_RECORDED', False
        else:
            proposed, reason, triggered, matching, improving = choose(rows,
                event['active_protection_or_phase'], consumed, block_size)
            picked = next(r for r in rows if r['index'] == proposed)
            tail = rows[-1]
            active = event['rule'] == 'equal_release_host_once'
            different = picked['request'] != event['native_tail']
            requested = active and different
            selected = picked['request'] if active else event['native_tail']
            after = consumed or different
            a, b = tail.get('immediate_releasable_blocks'), picked.get('immediate_releasable_blocks')
            expected = dict(mode='ACTIVE' if active else 'SHADOW', triggered=triggered,
                proposed_request=picked['request'], returned_request=selected,
                action_requested=requested, fallback=reason, consumed_before=consumed,
                consumed_after=after, matching_candidate_count=matching, improving_candidate_count=improving,
                tail_releasable_blocks=a, proposed_releasable_blocks=b,
                tail_host_missing_blocks=tail.get('host_missing_suffix_blocks'),
                proposed_host_missing_blocks=picked.get('host_missing_suffix_blocks'))
            recorded = event.get(FIELD) or {}
            mismatches = [FIELD + '.' + key for key, value in expected.items()
                if key not in recorded or recorded[key] != value or type(recorded[key]) is not type(value)]
            for key, value in dict(selected=selected, unrestricted_selected=selected, changed=requested,
                    current_guard_applied=False, current_guard_eligible=False).items():
                if event.get(key) != value or type(event.get(key)) is not type(value):
                    mismatches.append(key)
            triggers += int(triggered)
            proposals += int(not consumed and after)
            requests += int(requested)
            consumed = after
            item.update(status='MISMATCH' if mismatches else 'MATCH', expected_selected=selected,
                        expected_probe=expected, mismatches=mismatches)
        records.append(item)
    known = known and choose is not None
    return dict(source=str(source) if source else None, source_sha256=source_sha,
        executed_source_sha256=executed_sha, source_verified=choose is not None,
        status_counts=dict(Counter(r['status'] for r in records)), checks=records,
        recomputed_consumed=consumed if known else None,
        recomputed_trigger_count=triggers if known else None,
        recomputed_proposal_count=proposals if known else None,
        recomputed_action_request_count=requests if known else None,
        semantics='SHA-matched helper replay in recorded order from fresh measurement state. '
            'Block size comes from the measured capacity profile; the helper validates each row and skips known ineligible alternatives. '
            'Tail consumes the same proposal in shadow; active consumption is a returned replacement, '
            'not proof of successful preemption. Replay uses no future output labels.')


def adapt_events(events):
    return [dict(e, arrival_once=e.get(FIELD)) for e in events]


def partial_cell(cell):
    observation = cell.get('native_victim_observation', {})
    return dict(cell, native_victim_observation=dict(observation, raw_decisions=[
        dict(e, partial_restore_once=e.get(FIELD)) for e in observation.get('raw_decisions', [])]))


def intervention_prefix(reference, candidate, old_raw, new_raw):
    result = partial.intervention_prefix(partial_cell(reference), partial_cell(candidate), old_raw, new_raw)
    result['proposal_field_mapping'] = 'equal_release_host_once -> partial_restore_once -> arrival_once -> equal_held_once, reader adaptation only'
    return result


def once_outcome(cell, raw, store):
    mapped = dict(store, victim_decisions=[dict(e, partial_restore_once=e.get(FIELD))
                                         for e in store.get('victim_decisions', [])])
    result = partial.once_outcome(cell, raw, mapped)
    result['semantics'] = ('The inherited unique step+internal-ID join confirms actual preemption/free. '
        'Original-tail next output and later preemptions are observed trajectory outcomes, not a '
        'counterfactual recovery cost. Equal immediate release is verified physically; observed Host '
        'prefix availability is not presumed to survive. All-request and output-sequence effects remain separate.')
    return result


RECOVERY_SEMANTICS = ('Native scalar observations for previously preempted requests only. Lookup local '
    'prefix and offered external tokens, successful allocation requested external tokens, and worker '
    'acknowledgements are distinct stages; repeated offers must not be summed as restored tokens. '
    'An acknowledgement is not DMA latency; native_job_removed is a scheduler completion observation. '
    'Held pages are not newly allocated pages. No exposed load time, recompute, or counterfactual '
    'cache survival is inferred. Empty measured lists mean no such callback was recorded, not zero recovery work.')


def recovery_observations(data, raw, store, cell, source, executed_sha):
    source_sha = mixed.sha(source) if source is not None and source.is_file() else None
    source_verified = source_sha is not None and source_sha == executed_sha
    result = dict(recorded_status=data.get('status') if isinstance(data, dict) else None,
        source=str(source) if source else None, source_sha256=source_sha,
        executed_source_sha256=executed_sha, source_verified=source_verified,
        semantics=RECOVERY_SEMANTICS)
    if not isinstance(data, dict) or data.get('status') == 'NOT_MEASURED':
        return dict(result, status='NOT_MEASURED', callback_counts=None, episodes=None)
    fields = ('lookup', 'allocated', 'load_acknowledgements')
    required = dict(lookup=('output_tokens', 'total_tokens', 'request_status', 'local_computed_tokens',
                           'offered_external_tokens', 'asynchronous', 'free_blocks'),
        allocated=('output_tokens', 'total_tokens', 'request_status', 'local_computed_tokens',
                   'requested_external_tokens', 'allocated_held_blocks', 'free_blocks_after_allocation', 'native_load_jobs'),
        load_acknowledgements=('job_id', 'acknowledged_workers', 'pending_workers_before', 'native_job_removed'))
    if data.get('status') != 'NATIVE_RECOVERY_SCALARS' or any(not isinstance(data.get(k), list) for k in fields):
        return dict(result, status='UNKNOWN_OBSERVER_SCHEMA', callback_counts=None, episodes=None)
    if raw is None:
        return dict(result, status='NO_MEASUREMENT', callback_counts={k: len(data[k]) for k in fields}, episodes=None)
    origin = raw.get('measurement_origin_perf_counter_s')
    identity = raw.get('internal_to_source', {})
    admissions, calls, groups = defaultdict(list), defaultdict(list), defaultdict(lambda: defaultdict(list))
    for event in store.get('residency_admissions', []):
        admissions[event.get('request')].append(event)
    for event in raw.get('preemption_events', []):
        calls[event.get('internal_request_id')].append(event)
    preempts, unknown_calls = {}, []
    # Fresh measured requests begin at num_preemptions=0. Verify that anchor
    # rather than treating every successful call's list position as an ordinal.
    for rid, events in calls.items():
        initial = admissions.get(rid, [])[:1]
        anchor = (bool(initial) and initial[0].get('num_preemptions') == 0
                  and base.finite(initial[0].get('host_perf_counter_s')))
        previous = base.delta(origin, initial[0]['host_perf_counter_s']) if anchor else None
        ordinal, known = 0, anchor and base.finite(previous)
        for event in events:
            entered, returned = event.get('method_entered_s'), event.get('method_returned_s')
            known = (known and event.get('original_preemption_called') is True
                and event.get('original_preemption_returned') is True and base.finite(entered)
                and base.finite(returned) and previous <= entered <= returned)
            if known:
                ordinal += 1
                preempts[(rid, ordinal)] = event
                groups[(rid, ordinal)]  # Include episodes with no recorded recovery callback.
                previous = returned
            else:
                unknown_calls.append(event)
    unkeyed = []
    for field in fields:
        for index, event in enumerate(data[field]):
            rid, count = event.get('request'), event.get('preemptions')
            if not isinstance(rid, str) or type(count) is not int or count <= 0:
                unkeyed.append(dict(kind=field, index=index, event=event))
            else:
                groups[(rid, count)][field].append(event)
    joins = cell.get('native_victim_observation', {}).get('actual_release', {}).get('joins', [])
    outcomes = {e['decision_index']: e.get('victim')
                for e in cell.get('native_event_outcomes', {}).get('per_decision', [])}
    episodes = []
    for (rid, count), events in sorted(groups.items()):
        preempt = preempts.get((rid, count))
        checks = dict(source_request_identity=rid in identity, preemption_ordinal=preempt is not None)
        later = preempts.get((rid, count + 1))
        start = preempt.get('method_returned_s') if preempt else None
        stop = later.get('method_entered_s') if later else None
        for field in fields:
            for index, event in enumerate(events[field]):
                checks[f'{field}[{index}].recorded_fields'] = all(key in event for key in required[field])
                clock = base.delta(origin, event.get('host_perf_counter_s'))
                valid = base.finite(start) and base.finite(clock) and clock >= start
                # A load acknowledgement may arrive after a later preemption;
                # job identity remains authoritative, so retain it without a window rejection.
                if field != 'load_acknowledgements' and base.finite(stop):
                    valid = valid and clock < stop
                checks[f'{field}[{index}].clock'] = valid
                if field != 'load_acknowledgements':
                    checks[f'{field}[{index}].output_ordinal'] = (preempt is not None
                        and event.get('output_tokens') == preempt.get('native_output_count_before'))
        job_ids = {j.get('job_id') for e in events['allocated'] for j in e.get('native_load_jobs', [])
                   if type(j.get('job_id')) is int}
        for i, event in enumerate(events['load_acknowledgements']):
            checks[f'ack[{i}].allocated_job'] = event.get('job_id') in job_ids
        selected_join = [j for j in joins if preempt is not None
                         and j.get('join_status') == 'UNIQUE' and j.get('actual_preemption') == preempt]
        decision_index = selected_join[0]['decision_index'] if len(selected_join) == 1 else None
        checks['unique_native_preemption_free_join'] = decision_index is not None
        residence = [e for e in admissions.get(rid, []) if e.get('num_preemptions') == count]
        for i, event in enumerate(residence):
            clock = base.delta(origin, event.get('host_perf_counter_s'))
            checks[f'residency[{i}].clock'] = base.finite(start) and base.finite(clock) and clock >= start
            checks[f'residency[{i}].output_ordinal'] = (preempt is not None
                and event.get('output_count') == preempt.get('native_output_count_before'))
        episodes.append(dict(internal_request_id=rid, request_id=identity.get(rid), preemptions=count,
            status='ALIGNED' if all(checks.values()) else 'UNKNOWN_OR_MISMATCH', checks=checks,
            native_decision_index=decision_index, preemption=preempt, lookup=events['lookup'],
            allocated=events['allocated'], load_acknowledgements=events['load_acknowledgements'],
            native_load_job_ids=sorted(job_ids), native_jobs_observed_removed=sorted({
                e['job_id'] for e in events['load_acknowledgements'] if e.get('native_job_removed') is True
                and type(e.get('job_id')) is int}),
            residency_admissions=residence, next_output_observation=outcomes.get(decision_index)))
    valid = source_verified and not unkeyed and not unknown_calls and all(e['status'] == 'ALIGNED' for e in episodes)
    return dict(result, status='VERIFIED' if valid else 'UNKNOWN_OR_MISMATCH',
        callback_counts={k: len(data[k]) for k in fields}, observer_body_wall_s=data.get('observer_body_wall_s'),
        observer_timing_semantics='Body timing excludes native calls; it does not isolate Python wrapper overhead.',
        episodes=episodes, unkeyed_observations=unkeyed, preemptions_with_unknown_ordinal=unknown_calls,
        ordinal_semantics='Per internal request, start from a recorded zero-preemption residency; '
            'count only the chronological successful native calls until any failed/unknown call. '
            'Match observer num_preemptions and output ordinal, then join actual free/output by the inherited native decision.')


def analyze(session, timeout_s=600, policy_source=None, input_config=None):
    session = Path(session)
    policy_source = Path(policy_source) if policy_source else PACKAGE / 'pkg/staged_store_rotation.py'
    input_config = Path(input_config) if input_config else PACKAGE / 'pkg/inputs/pro_high/config.json'
    plan = base.read(session / 'plan.json')
    if (plan.get('experiment_role') != ROLE or plan.get('pinned_gpu_kv_bytes') != 77076627456
            or any(c.get('requests') != 320 or c.get('input_case') != 'high' for c in plan.get('cells', []))):
        raise ValueError('Expected the shared320 equal-release Host-prefix intervention with fresh-device pinned normal capacity')
    profile, provenance = partial.source_profile(session, plan)
    previous = base.KIND, base.RULES, base.verify_choices, base.read_native_cell, base.compare_cells
    original_read, original_compare = base.read_native_cell, base.compare_cells

    def read_cell(directory, spec, cell_plan, timeout, source):
        cell, raw = original_read(directory, spec, cell_plan, timeout, source)
        archive = base.archive_dir(directory)
        store = base.optional(archive / 'selective-store.json', {})
        environment = base.optional(archive / 'environment.json', {})
        executed = directory / 'candidate_native_oldest_strong_r01/pkg'
        chosen_source = executed / 'staged_store_rotation.py'
        chosen_source = chosen_source if chosen_source.is_file() else source
        config_path = executed / 'inputs/pro_high/config.json'
        config_path = config_path if config_path.is_file() else input_config
        config, workload = base.read(config_path), base.read(config_path.with_name('workload.json'))
        cell['budget_consistency'] = mixed.budget_check(config, workload, cell.get('config') or {}, raw)
        cell['input_consistency'] = shared.input_check(config, workload, raw)
        cell['input_consistency'].update(input_config_path=str(config_path), input_config_sha256=mixed.sha(config_path),
            input_workload_sha256=mixed.sha(config_path.with_name('workload.json')))
        cell['fixed_warmup_check'] = mixed.warmup_check(archive, cell.get('engine_args') or {})
        cell['shared_prefix_domain'] = shared.domain_check(cell, store)
        cell['shared_prefix_observations'] = partial.physical_observations(cell, store, raw, chosen_source,
            environment.get('source_sha256', {}).get('staged_store_rotation.py'))
        cell['source_profile_capacity'] = partial.capacity_check(profile, provenance, cell, environment, plan)
        cell['equal_release_host_once_outcome'] = once_outcome(cell, raw, store)
        observer_source = chosen_source.parent / 'native_offload_observer.py'
        cell['native_recovery_observations'] = recovery_observations(
            base.optional(archive / 'offload-events.json'), raw, store, cell, observer_source,
            environment.get('source_sha256', {}).get('native_offload_observer.py'))
        choice = cell['executed_choice_check']
        checks = cell['configuration_check']['checks']
        expected = dict(mode='ACTIVE' if spec['native_victim_rule'] == 'equal_release_host_once' else 'SHADOW',
            consumed=choice['recomputed_consumed'], trigger_count=choice['recomputed_trigger_count'],
            proposal_count=choice['recomputed_proposal_count'], action_request_count=choice['recomputed_action_request_count'])
        recorded = store.get('equal_release_host_once_probe') or {}
        for key, value in expected.items():
            actual = recorded.get(key)
            checks.append(dict(field='equal_release_host_once_probe.'+key, actual=actual, expected=value,
                matches=value is not None and actual == value and type(actual) is type(value)))
        checks.append(dict(field='equal_release_host_once_helper_source', actual=choice['source_verified'],
            expected=True, matches=choice['source_verified']))
        for key in ('budget_consistency', 'input_consistency', 'fixed_warmup_check',
                    'shared_prefix_domain', 'shared_prefix_observations', 'source_profile_capacity',
                    'native_recovery_observations'):
            checks.append(dict(field=key, actual=cell[key]['status'], expected='VERIFIED',
                               matches=cell[key]['status'] == 'VERIFIED'))
        timings = [e[FIELD].get('selector_wall_s') for e in store.get('victim_decisions', [])
                   if isinstance(e.get(FIELD), dict)]
        valid = [v for v in timings if base.finite(v) and v >= 0]
        cell['equal_release_host_selector_cost'] = dict(recorded_decisions=len(valid),
            unknown_decisions=len(store.get('victim_decisions', []))-len(valid),
            total_recorded_s=sum(valid) if valid else None, distribution=base.distribution(valid),
            semantics='Nested helper wall time is contained in outer selector_wall_s; do not sum them.')
        cell['configuration_check']['status'] = 'VERIFIED' if all(c['matches'] for c in checks) else 'MISMATCH_OR_MISSING'
        return cell, raw

    def compare(reference, candidate, old_raw, new_raw):
        result = original_compare(reference, candidate, old_raw, new_raw)
        result['pre_intervention_prefix'] = intervention_prefix(reference, candidate, old_raw, new_raw)
        _, event = once.proposal(adapt_events(candidate['native_victim_observation']['raw_decisions']))
        if event is not None:
            identity = new_raw.get('internal_to_source', {})
            participants = {name: identity.get(event.get(field)) for name, field in
                (('original_tail', 'native_tail'), ('selected_victim', 'selected'), ('allocation_failed_request', 'failed_request'))}
            result['intervention_participants'] = dict(source_request_ids=participants,
                all_request_comparison_indices=[i for i, r in enumerate(result['per_request'])
                                                if r['request_id'] in participants.values()],
                semantics='Indices into full-request effects, not additional observations or causal matched events.')
        return result

    try:
        base.KIND, base.RULES, base.verify_choices = KIND, RULES, verify_choices
        base.read_native_cell, base.compare_cells = read_cell, compare
        result = base.analyze(session, timeout_s, policy_source)
    finally:
        base.KIND, base.RULES, base.verify_choices, base.read_native_cell, base.compare_cells = previous
    if all(c.get('status') in ('NOT_RUN', 'NO_MEASUREMENT') for c in result['cells'].values()):
        result['status'] = 'NO_MEASUREMENT'
    result['experiment_role'] = ROLE
    result['source_profile_provenance'] = provenance
    result['analysis_code_sha256'] = {Path(m.__file__).name: mixed.sha(Path(m.__file__))
        for m in (base, once, once.equal, mixed, shared, profiled, partial)} | {Path(__file__).name: mixed.sha(Path(__file__))}
    result['semantics'] = dict(result['semantics'], intervention=
        'At most one state-triggered private victim replacement, then tail. Tail uses the same once shadow. '
        'The private pure-decode pair must have equal immediate physical release and computed whole pages. '
        'The chosen alternative has strictly fewer currently missing Host pages; cache survival and exposed '
        'recovery cost are measured separately, not assumed. Full-request natural-EOS output differences '
        'and observed pre-action discrepancies remain explicit. No equal-work, strict same-state or '
        'recovery-cost benefit is assumed. No application goodput threshold is selected.')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', type=Path, required=True)
    parser.add_argument('--timeout-s', type=float, default=600)
    parser.add_argument('--policy-source', type=Path)
    parser.add_argument('--input-config', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = analyze(args.session, args.timeout_s, args.policy_source, args.input_config)
    if args.output:
        with args.output.open('x') as stream:
            json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write('\n')
        print(json.dumps(dict(status=result['status'], output=str(args.output)), ensure_ascii=False))
    else:
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
