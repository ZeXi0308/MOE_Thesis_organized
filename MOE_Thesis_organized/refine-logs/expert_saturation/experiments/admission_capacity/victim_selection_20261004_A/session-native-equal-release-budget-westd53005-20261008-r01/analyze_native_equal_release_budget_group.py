#!/usr/bin/env python3
"""All-request ABBA analysis of online equal-release declared-budget selection."""
import argparse
import ast
from collections import Counter
import json
from pathlib import Path

import analyze_native_group as base
import analyze_native_mixed_budget_probe as mixed
import analyze_native_partial_restore_once_group as partial
import analyze_native_equal_release_host_once_group as exacthost


HERE = Path(__file__).resolve().parent
PACKAGE = HERE / 'candidate_native_equal_release_budget_r01'
KIND = 'PRO6000_NATIVE_EQUAL_RELEASE_BUDGET'
ROLE = 'EQUAL_RELEASE_BUDGET_INTERVENTION'
FIELD = 'equal_release_budget'
RULES = ('tail', FIELD, FIELD, 'tail')
MIXED_WORKLOAD_SHA256 = '748e857953dbc6072cabf2827e3d5b1719f0096430871fac099b836edd7817a1'
PROFILE_ENGINE_DIFFERENCES = {'enable_prefix_caching': {'profile': True, 'measurement': False}}


def remaining(row):
    cap, output = row.get('max_tokens'), row.get('output_tokens')
    return cap-output if type(cap) is int and type(output) is int and 0 <= output <= cap else None


def verify_choices(decisions, source, executed_sha, block_size=None):
    choose, source_sha = None, None
    if source is not None and source.is_file():
        source_sha = mixed.sha(source)
        if source_sha == executed_sha:
            node = next((n for n in ast.parse(source.read_bytes()).body
                if isinstance(n, ast.FunctionDef) and n.name == '_equal_release_budget_choice'), None)
            if node is not None:
                namespace = {}
                exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), namespace)
                choose = namespace[node.name]
    records, fallbacks = [], Counter()
    proposals = requests = 0
    required = ('held_blocks', 'immediate_releasable_blocks', 'shared_blocks', 'release_state_error',
        'prompt_tokens', 'computed_tokens', 'output_tokens', 'qualified', 'request_status',
        'max_tokens', 'pending_native_store_dependencies')
    for index, event in enumerate(decisions):
        item = dict(decision_index=index, step=event.get('step'), rule=event.get('rule'))
        rows = event.get('candidates', [])
        indices = [r.get('index') for r in rows]
        if choose is None:
            item['status'] = 'EXECUTED_POLICY_SOURCE_UNAVAILABLE'
        elif (not rows or any(type(i) is not int for i in indices)
                or indices != list(range(indices[0], indices[-1] + 1))
                or type(event.get('unprocessed_suffix_start')) is not int
                or indices[0] != event['unprocessed_suffix_start']
                or rows[0].get('request') != event.get('failed_request')
                or rows[-1].get('request') != event.get('native_tail')
                or any(not isinstance(r.get('request'), str) for r in rows)
                or len({r['request'] for r in rows}) != len(rows)):
            item['status'] = 'INVALID_SUFFIX_IDENTITY'
        elif event.get('rule') not in RULES:
            item['status'] = 'UNSUPPORTED_RULE'
        elif type(event.get('active_protection_or_phase')) is not bool:
            item['status'] = 'REQUIRED_GUARD_NOT_RECORDED'
        elif any(key not in r for r in rows for key in required):
            item['status'] = 'UNKNOWN_REQUIRED_FIELDS_NOT_RECORDED'
        else:
            proposed, reason, triggered, matching, improving = choose(
                rows, event['active_protection_or_phase'], block_size)
            picked = next(r for r in rows if r['index'] == proposed)
            tail = rows[-1]
            active = event['rule'] == FIELD
            different = picked['request'] != event['native_tail']
            requested = active and different
            selected = picked['request'] if active else event['native_tail']
            expected = dict(mode='ACTIVE' if active else 'SHADOW', triggered=triggered,
                proposed_request=picked['request'], returned_request=selected,
                action_requested=requested, fallback=reason, matching_candidate_count=matching,
                improving_candidate_count=improving,
                tail_releasable_blocks=tail.get('immediate_releasable_blocks'),
                proposed_releasable_blocks=picked.get('immediate_releasable_blocks'),
                tail_remaining_output_budget=remaining(tail), proposed_remaining_output_budget=remaining(picked))
            recorded = event.get(FIELD) or {}
            mismatches = [FIELD + '.' + key for key, value in expected.items()
                if key not in recorded or recorded[key] != value or type(recorded[key]) is not type(value)]
            for key, value in dict(selected=selected, unrestricted_selected=selected, changed=requested,
                    current_guard_applied=False, current_guard_eligible=False).items():
                if event.get(key) != value or type(event.get(key)) is not type(value):
                    mismatches.append(key)
            proposals += int(different)
            requests += int(requested)
            if reason is not None:
                fallbacks[reason] += 1
            item.update(status='MISMATCH' if mismatches else 'MATCH', expected_selected=selected,
                        expected_probe=expected, mismatches=mismatches)
        records.append(item)
    known = choose is not None and all('expected_probe' in r for r in records)
    return dict(source=str(source) if source else None, source_sha256=source_sha,
        executed_source_sha256=executed_sha, source_verified=choose is not None,
        status_counts=dict(Counter(r['status'] for r in records)), checks=records,
        recomputed_decision_count=len(records) if known else None,
        recomputed_proposal_count=proposals if known else None,
        recomputed_action_request_count=requests if known else None,
        recomputed_fallback_counts=dict(fallbacks) if known else None,
        semantics='SHA-matched stateless helper replay at every native suffix decision. No once '
            'consumption or future outcome is used. Missing fields remain UNKNOWN; recorded unknown '
            'values can verify a fail-closed fallback. Host fields do not enter this rule. Declared '
            'max_tokens minus produced tokens is an output-budget bound, not a natural-EOS prediction.')


def proposals(events):
    return [(i, e) for i, e in enumerate(events) if isinstance(e.get(FIELD), dict)
        and isinstance(e[FIELD].get('proposed_request'), str)
        and e[FIELD]['proposed_request'] != e.get('native_tail')]


def mark_one(events, index):
    """Reader-only alias; only this event gets the inherited one-proposal marker."""
    return [dict(e, partial_restore_once=dict(e.get(FIELD) or {},
        consumed_before=False, consumed_after=i == index)) for i, e in enumerate(events)]


def intervention_prefix(reference, candidate, old_raw, new_raw):
    adapted, first = [], []
    for cell in (reference, candidate):
        observation = cell.get('native_victim_observation', {})
        events = observation.get('raw_decisions', [])
        found = proposals(events)
        index = found[0][0] if found else None
        first.append(index)
        adapted.append(dict(cell, native_victim_observation=dict(observation,
            raw_decisions=mark_one(events[:index+1] if index is not None else events, index))))
    result = partial.intervention_prefix(*adapted, old_raw, new_raw)
    result['first_proposal_decision_indices'] = first
    result['proposal_field_mapping'] = 'Only first equal_release_budget proposal -> partial_restore_once reader marker; raw unchanged'
    result['semantics'] = ('Prefix before each trajectory\'s FIRST online proposal only, including its '
        'candidate snapshot. The policy keeps re-evaluating subsequent decisions; no once restriction '
        'is asserted. Tail proposals are shadows. Pre-action clocks, outputs, host/progress/physical '
        'states and arrivals remain observed comparisons, not proof of identical hidden state or '
        'a causal adjustment. Actual execution is separately joined to preempt/free.')
    return result


def intervention_outcomes(cell, raw, store):
    events = store.get('victim_decisions', [])
    records = []
    for index, event in proposals(events):
        outcome = partial.once_outcome(cell, raw, dict(store, victim_decisions=mark_one(events, index)))
        outcome['semantics'] = ('One of potentially many online proposals. Actual preemption/free '
            'uses the unique step+internal-ID join. Original-tail waits and later preemptions are '
            'observed subsequent outcomes, potentially affected by further choices; no per-event '
            'causal benefit or counterfactual recovery cost is identified.')
        records.append(outcome)
    unknown = sum(not isinstance(e.get(FIELD), dict)
        or not isinstance((e.get(FIELD) or {}).get('proposed_request'), str)
        or type((e.get(FIELD) or {}).get('action_requested')) is not bool
        or not isinstance(e.get('native_tail'), str) for e in events)
    return dict(status='NO_MEASUREMENT' if raw is None else 'UNKNOWN_PROPOSAL_SCHEMA' if unknown else 'ANALYZED',
        recorded_proposals=len(records), unknown_probe_decisions=unknown,
        actual_execution_confirmed=sum(r.get('action_execution_confirmed') is True for r in records)
            if raw is not None and unknown == 0 else None,
        per_proposal=records, semantics='All proposals retained, including tail shadows. Per-request '
            'metrics, selected-victim next outputs and recovery episodes are inherited once at cell/group level. '
            'These overlapping event outcomes must not be summed as independent gains or repetitions.')


def input_check(config, workload, raw):
    rows, prompts = workload.get('source_requests', []), workload.get('actual_prompt_token_ids', [])
    ids = [r.get('request_id') for r in rows]
    hashes = [base.hashlib.sha256(json.dumps(p, separators=(',', ':')).encode()).hexdigest() for p in prompts]
    shape = len(rows) == len(prompts) == len(set(ids)) == 320
    checks = dict(original_mixed_workload=base.digest(workload) == MIXED_WORKLOAD_SHA256,
        input_shape=shape, independent_documents=len({r.get('document_id') for r in rows}) == 320,
        unique_tokenized_prompts=len(set(hashes)) == 320,
        input_prefix_disabled=config.get('enable_prefix_caching', False) is False,
        workload_digest=config.get('workload_sha256') == base.digest(workload),
        external_arrival_trace=workload.get('arrival_traces_s') == {'steady': [i/100 for i in range(320)]})
    actual = {r.get('request_id'): r for r in raw.get('requests', [])} if raw is not None else {}
    measured = raw is not None and len(raw.get('requests', [])) == len(actual) == 320 and set(actual) == set(ids)
    checks['measured_request_ids'] = measured if raw is not None else None
    checks['measured_arrivals_and_prompts'] = (shape and measured and all(
        actual[rid].get('arrival_s') == i/100 and actual[rid].get('prompt_token_ids_sha256') == hashes[i]
        and actual[rid].get('prompt_tokens') == len(prompts[i]) for i, rid in enumerate(ids))) if raw is not None else None
    return dict(status='VERIFIED' if all(v is True for v in checks.values()) else 'MISMATCH_OR_MISSING',
        checks=checks, expected_workload_sha256=MIXED_WORKLOAD_SHA256,
        semantics='Original mixed320 independent articles and i/100 external arrivals; no shared-input adaptation. '
            'The original input omits enable_prefix_caching, whose runner default is False; measured engine flags are explicit.')


def capacity_check(profile, provenance, cell, environment, plan):
    result = partial.capacity_check(profile, provenance, cell, environment, plan)
    if profile is None:
        return result
    before, after = profile['engine_args'], cell.get('engine_args', {})
    differences = {k: dict(profile=before.get(k), measurement=after.get(k))
        for k in before.keys() | after.keys() if k != 'kv_cache_memory_bytes' and before.get(k) != after.get(k)}
    declared = plan.get('source_profile_engine_differences')
    permitted = differences == declared == PROFILE_ENGINE_DIFFERENCES
    result['checks']['exact_declared_prefix_delta'] = permitted
    result['accepted_check_exceptions'] = ['same_engine_configuration'] if permitted else []
    result['source_profile_engine_differences'] = dict(declared=declared, observed=differences)
    result['status'] = ('VERIFIED' if all(v or k in result['accepted_check_exceptions']
        for k, v in result['checks'].items()) else 'MISMATCH_OR_MISSING')
    result['semantics'] = ('Reuses the referenced same-device normal profile physical KV-byte budget. '
        'That profile enabled prefix caching; this group explicitly disables it. Strict inherited '
        'engine equality is retained and only the declared True-to-False prefix flag is excepted. '
        'The separately verified None-to-pin KV allocation, all other engine/native/page geometry '
        'and GPU UUID checks remain required. This is not a new prefix-off normal-capacity profile.')
    return result


def analyze(session, timeout_s=600, policy_source=None, input_config=None):
    session = Path(session)
    policy_source = Path(policy_source) if policy_source else PACKAGE / 'pkg/staged_store_rotation.py'
    input_config = Path(input_config) if input_config else mixed.PACKAGE / 'pkg/inputs/pro_high/config.json'
    plan = base.read(session / 'plan.json')
    if (plan.get('experiment_role') != ROLE or plan.get('pinned_gpu_kv_bytes') != 77076627456
            or plan.get('source_profile_engine_differences') != PROFILE_ENGINE_DIFFERENCES
            or any(c.get('requests') != 320 or c.get('input_case') != 'high' for c in plan.get('cells', []))):
        raise ValueError('Expected original mixed320 online equal-release budget ABBA, pinned normal bytes and declared prefix delta')
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
        cell['input_consistency'] = input_check(config, workload, raw)
        cell['input_consistency'].update(input_config_path=str(config_path), input_config_sha256=mixed.sha(config_path),
            input_workload_sha256=mixed.sha(config_path.with_name('workload.json')))
        cell['fixed_warmup_check'] = mixed.warmup_check(archive, cell.get('engine_args') or {})
        cell['physical_release_observations'] = partial.physical_observations(cell, store, raw, chosen_source,
            environment.get('source_sha256', {}).get('staged_store_rotation.py'))
        cell['physical_release_observations']['semantics'] = ('Inherited physical and max-release SHADOW '
            'checks, with this online rule\'s SHA-matched verdict for actual selected victims. Actual '
            'free-pool increments are checked against immediate releasable pages, not presumed from held pages.')
        cell['source_profile_capacity'] = capacity_check(profile, provenance, cell, environment, plan)
        cell['equal_release_budget_outcomes'] = intervention_outcomes(cell, raw, store)
        cell['native_recovery_observations'] = exacthost.recovery_observations(
            base.optional(archive / 'offload-events.json'), raw, store, cell,
            chosen_source.parent / 'native_offload_observer.py',
            environment.get('source_sha256', {}).get('native_offload_observer.py'))
        choice, checks = cell['executed_choice_check'], cell['configuration_check']['checks']
        def check(field, actual, expected):
            checks.append(dict(field=field, actual=actual, expected=expected,
                matches=expected is not None and actual == expected and type(actual) is type(expected)))
        recorded = store.get(FIELD + '_probe') or {}
        expected = dict(mode='ACTIVE' if spec['native_victim_rule'] == FIELD else 'SHADOW',
            **{k: choice['recomputed_' + k] for k in
                ('decision_count', 'proposal_count', 'action_request_count', 'fallback_counts')})
        for key, value in expected.items():
            check(FIELD + '_probe.' + key, recorded.get(key), value)
        check('equal_release_budget_helper_source', choice['source_verified'], True)
        # Original mixed inputs and the dumped config omit this key. The frozen
        # runner explicitly defaults that absence to False; engine/manager must record it.
        check('config.enable_prefix_caching_default_false',
            (cell.get('config') or {}).get('enable_prefix_caching', False), False)
        check('engine.enable_prefix_caching', (cell.get('engine_args') or {}).get('enable_prefix_caching'), False)
        check('store.prefix_caching', store.get('prefix_caching'), False)
        check('store.coordinator_type', store.get('coordinator_type'), 'UnitaryKVCacheCoordinator')
        for key in ('budget_consistency', 'input_consistency', 'fixed_warmup_check',
                    'physical_release_observations', 'source_profile_capacity', 'native_recovery_observations'):
            check(key, cell[key]['status'], 'VERIFIED')
        timings = [(e.get(FIELD) or {}).get('selector_wall_s') for e in store.get('victim_decisions', [])]
        valid = [v for v in timings if base.finite(v) and v >= 0]
        cell['equal_release_budget_selector_cost'] = dict(recorded_decisions=len(valid),
            unknown_decisions=len(timings)-len(valid),
            total_recorded_s=sum(valid) if raw is not None and (valid or not timings) else None,
            distribution=base.distribution(valid), recorded_probe_total_s=recorded.get('selector_wall_s'),
            semantics='Nested helper timing is contained in outer selector_wall_s, not additional overhead. '
                'Missing event timing is counted as unknown; measured empty decisions total zero.')
        total = sum(valid) if len(valid) == len(timings) and raw is not None else None
        check(FIELD + '_probe.selector_wall_s', recorded.get('selector_wall_s'), total)
        cell['configuration_check']['status'] = 'VERIFIED' if all(c['matches'] for c in checks) else 'MISMATCH_OR_MISSING'
        return cell, raw

    def compare(reference, candidate, old_raw, new_raw):
        result = original_compare(reference, candidate, old_raw, new_raw)
        result['pre_first_intervention_prefix'] = intervention_prefix(reference, candidate, old_raw, new_raw)
        identity, participants = new_raw.get('internal_to_source', {}), []
        for index, event in proposals(candidate['native_victim_observation']['raw_decisions']):
            ids = {name: identity.get(event.get(field)) for name, field in
                (('original_tail', 'native_tail'), ('selected_victim', 'selected'), ('allocation_failed_request', 'failed_request'))}
            participants.append(dict(decision_index=index, step=event.get('step'), source_request_ids=ids,
                all_request_comparison_indices=[i for i, r in enumerate(result['per_request']) if r['request_id'] in ids.values()]))
        result['intervention_participants'] = dict(per_proposal=participants,
            semantics='References into the single full-request comparison; requests and event windows can recur. '
                'No repeated-request delta is summed and later proposals are not matched causal interventions.')
        return result

    try:
        base.KIND, base.RULES, base.verify_choices = KIND, RULES, verify_choices
        base.read_native_cell, base.compare_cells = read_cell, compare
        result = base.analyze(session, timeout_s, policy_source)
    finally:
        base.KIND, base.RULES, base.verify_choices, base.read_native_cell, base.compare_cells = previous
    if all(c.get('status') in ('NOT_RUN', 'NO_MEASUREMENT') for c in result['cells'].values()):
        result['status'] = 'NO_MEASUREMENT'
    result['experiment_role'], result['source_profile_provenance'] = ROLE, provenance
    result['analysis_code_sha256'] = {Path(m.__file__).name: mixed.sha(Path(m.__file__))
        for m in (base, mixed, partial, partial.once, partial.once.equal, partial.shared, partial.profiled, exacthost)}
    result['analysis_code_sha256'][Path(__file__).name] = mixed.sha(Path(__file__))
    result['semantics'] = dict(result['semantics'], intervention='Online repeated victim selection with '
        'tail shadows in both arms. Alternatives require equal immediate physical release and computed '
        'whole pages; choose greater declared remaining output budget, not predicted natural EOS. '
        'All actions, failed/uncompleted requests, recovery observations and output differences remain '
        'visible. First-prefix comparison is only a diagnostic; no once-only, equal-work, net-benefit '
        'or novelty claim follows from a selector proxy. No goodput threshold is selected.')
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
