#!/usr/bin/env python3
"""All-request ABBA comparison of two active full-suffix victim baselines."""
import argparse
import ast
from collections import Counter
import json
from pathlib import Path

import analyze_native_equal_release_budget_group_r02 as inherited


base, mixed, partial, exacthost = inherited.base, inherited.mixed, inherited.partial, inherited.exacthost
HERE = Path(__file__).resolve().parent
PACKAGE = HERE / 'candidate_native_max_release_r01'
KIND, ROLE = 'PRO6000_NATIVE_MAX_RELEASE', 'MAX_RELEASE_INTERVENTION'
RULES = ('remaining_budget', 'max_release', 'max_release', 'remaining_budget')
FIELDS = {'remaining_budget': 'remaining_budget', 'max_release': 'max_release_shadow'}


def verify_choices(decisions, source, executed_sha, block_size=None):
    helpers, source_sha = {}, None
    if source is not None and source.is_file():
        source_sha = mixed.sha(source)
        if source_sha == executed_sha:
            names = {'_remaining_budget_choice', '_max_release_shadow_choice'}
            nodes = [n for n in ast.parse(source.read_bytes()).body
                if isinstance(n, ast.FunctionDef) and n.name in names]
            if {n.name for n in nodes} == names:
                exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), helpers)
    records = []
    for index, event in enumerate(decisions):
        item = dict(decision_index=index, step=event.get('step'), rule=event.get('rule'))
        rows = event.get('candidates', [])
        indices = [row.get('index') for row in rows]
        if not helpers:
            item['status'] = 'EXECUTED_POLICY_SOURCE_UNAVAILABLE'
        elif (not rows or any(type(i) is not int for i in indices)
                or indices != list(range(indices[0], indices[-1] + 1))
                or type(event.get('unprocessed_suffix_start')) is not int
                or indices[0] != event['unprocessed_suffix_start']
                or rows[0].get('request') != event.get('failed_request')
                or rows[-1].get('request') != event.get('native_tail')
                or any(not isinstance(row.get('request'), str) for row in rows)
                or len({row['request'] for row in rows}) != len(rows)):
            item['status'] = 'INVALID_SUFFIX_IDENTITY'
        elif event.get('rule') not in FIELDS:
            item['status'] = 'UNSUPPORTED_RULE'
        elif type(event.get('active_protection_or_phase')) is not bool:
            item['status'] = 'REQUIRED_GUARD_NOT_RECORDED'
        elif any(key not in row for row in rows for key in
                ('max_tokens', 'output_tokens', 'immediate_releasable_blocks')):
            item['status'] = 'UNKNOWN_REQUIRED_FIELDS_NOT_RECORDED'
        else:
            phase, tail = event['active_protection_or_phase'], rows[-1]
            by_index = {row['index']: row for row in rows}
            release, release_reason = helpers['_max_release_shadow_choice'](rows, phase)
            budget, budget_reason, triggered, matching, improving = helpers['_remaining_budget_choice'](
                rows, phase, block_size)
            proposals = {'max_release': release, 'remaining_budget': budget}
            selected = by_index[proposals[event['rule']]]['request']
            changed = selected != event['native_tail']
            expected, mismatches = {}, []
            for rule, field in FIELDS.items():
                picked, active = by_index[proposals[rule]], event['rule'] == rule
                different = picked['request'] != event['native_tail']
                probe = dict(mode='ACTIVE' if active else 'SHADOW', proposed_request=picked['request'],
                    returned_request=selected, action_requested=active and different,
                    fallback=release_reason if rule == 'max_release' else budget_reason,
                    tail_releasable_blocks=tail.get('immediate_releasable_blocks'),
                    proposed_releasable_blocks=picked.get('immediate_releasable_blocks'))
                if rule == 'max_release':
                    probe['changed_from_tail'] = different
                else:
                    probe.update(triggered=triggered, matching_candidate_count=matching,
                        improving_candidate_count=improving,
                        tail_remaining_output_budget=inherited.remaining(tail),
                        proposed_remaining_output_budget=inherited.remaining(picked))
                expected[rule] = probe
                recorded = event.get(field) or {}
                mismatches.extend(field + '.' + key for key, value in probe.items()
                    if key not in recorded or recorded[key] != value or type(recorded[key]) is not type(value))
            for key, value in dict(selected=selected, unrestricted_selected=selected, changed=changed,
                    current_guard_applied=False, current_guard_eligible=False).items():
                if event.get(key) != value or type(event.get(key)) is not type(value):
                    mismatches.append(key)
            item.update(status='MISMATCH' if mismatches else 'MATCH', expected_selected=selected,
                expected_probes=expected, mismatches=mismatches)
        records.append(item)
    known = bool(helpers) and all('expected_probes' in row for row in records)
    summaries = {}
    for rule in FIELDS:
        probes = [row['expected_probes'][rule] for row in records if 'expected_probes' in row]
        summaries[rule] = dict(decision_count=len(records) if known else None,
            proposal_count=sum(p['proposed_request'] != event.get('native_tail')
                for p, event in zip(probes, decisions)) if known else None,
            action_request_count=sum(p['action_requested'] for p in probes) if known else None,
            fallback_counts=dict(Counter(p['fallback'] for p in probes if p['fallback'] is not None)) if known else None)
    return dict(source=str(source) if source else None, source_sha256=source_sha,
        executed_source_sha256=executed_sha, source_verified=bool(helpers),
        status_counts=dict(Counter(row['status'] for row in records)), checks=records,
        recomputed_probe_summaries=summaries,
        semantics='Both SHA-matched helpers replay every native suffix. Each arm executes its own '
            'rule; the other rule is shadow-only but returned_request still records the actual choice. '
            'Non-tail remaining-budget actions remain interventions in the reference arm. '
            'Missing fields are UNKNOWN; explicit nulls can verify fail-closed fallback.')


def proposals(events, field):
    return [(i, event) for i, event in enumerate(events) if isinstance(event.get(field), dict)
        and isinstance(event[field].get('proposed_request'), str)
        and event[field]['proposed_request'] != event.get('native_tail')]


def mark_one(events, index, field):
    return [dict(event, partial_restore_once=dict(event.get(field) or {},
        consumed_before=False, consumed_after=i == index)) for i, event in enumerate(events)]


def intervention_prefix(reference, candidate, old_raw, new_raw):
    adapted, indices, disagreements, unknown_probes = [], [], [], []
    for cell, raw in zip((reference, candidate), (old_raw, new_raw)):
        observation = cell.get('native_victim_observation', {})
        events = observation.get('raw_decisions', [])
        found = proposals(events, FIELDS['max_release'])
        index = found[0][0] if found else None
        indices.append(index)
        prefix_events = events[:index+1] if index is not None else events
        unknown_probes.append(sum(not isinstance((event.get(field) or {}).get('proposed_request'), str)
            for event in prefix_events for field in FIELDS.values()))
        adapted.append(dict(cell, native_victim_observation=dict(observation,
            raw_decisions=mark_one(prefix_events, index, FIELDS['max_release']))))
        disagreement = None
        for i, event in enumerate(events):
            a, b = (event.get(field) or {} for field in FIELDS.values())
            if (isinstance(a.get('proposed_request'), str) and isinstance(b.get('proposed_request'), str)
                    and a['proposed_request'] != b['proposed_request']):
                identity = raw.get('internal_to_source', {})
                disagreement = dict(decision_index=i, step=event.get('step'),
                    remaining_budget=identity.get(a['proposed_request']), max_release=identity.get(b['proposed_request']),
                    selected=identity.get(event.get('selected')),
                    before_first_max_release_proposal=i < index if index is not None else None)
                break
        disagreements.append(disagreement)
    result = partial.intervention_prefix(*adapted, old_raw, new_raw)
    result.update(first_max_release_proposal_decision_indices=indices,
        first_observed_policy_disagreements=disagreements,
        unknown_policy_proposal_fields=unknown_probes,
        proposal_field_mapping='First max_release_shadow non-tail proposal -> inherited prefix marker only',
        semantics='Each prefix stops before its first max-release non-tail proposal, with its candidate '
            'snapshot included. Earlier ACTUAL choices, including remaining-budget replacements in '
            'the reference, are retained. The first observed helper disagreement is reported separately '
            'and may precede this boundary. Only an observed equal prefix is common; no hidden-state '
            'equivalence, once restriction, timing correction, or later trajectory equality is required.')
    return result


def intervention_outcomes(cell, raw, store):
    events = store.get('victim_decisions', [])
    rule = cell['plan_cell']['native_victim_rule']
    field, records = FIELDS[rule], []
    for index, _ in proposals(events, field):
        outcome = partial.once_outcome(cell, raw, dict(store, victim_decisions=mark_one(events, index, field)))
        outcome['semantics'] = ('One active-rule proposal, joined to unique successful step+internal-ID '
            'preemption/free. Subsequent waits and repeated preemptions can be affected by later choices; '
            'overlapping event outcomes are not independent or additive causal gains.')
        records.append(outcome)
    unknown = sum(not isinstance(event.get(field), dict)
        or not isinstance((event.get(field) or {}).get('proposed_request'), str)
        or type((event.get(field) or {}).get('action_requested')) is not bool
        or not isinstance(event.get('native_tail'), str) for event in events)
    return dict(status='NO_MEASUREMENT' if raw is None else 'UNKNOWN_PROPOSAL_SCHEMA' if unknown else 'ANALYZED',
        active_rule=rule, recorded_proposals=len(records), unknown_probe_decisions=unknown,
        actual_execution_confirmed=sum(row.get('action_execution_confirmed') is True for row in records)
            if raw is not None and unknown == 0 else None,
        per_proposal=records, semantics='Active non-tail proposals in BOTH arms; reference is not an '
            'unmodified tail run. Shadow proposals are validated separately and are not executions.')


def physical_observations(cell, store, raw, source, executed_sha):
    result = partial.shared.shared_observations(cell, store, raw, source, executed_sha)
    checks = {row['decision_index']: row for row in cell['executed_choice_check']['checks']}
    for row in result.get('per_decision', []):
        check = checks.get(row['decision_index'], {})
        expected = check.get('expected_probes', {}).get('max_release')
        row.pop('actual_tail_verified', None)
        row.pop('recomputed_shadow', None)
        row.pop('shadow_check', None)
        row['actual_choice_verified'] = check.get('status') == 'MATCH'
        row['recomputed_max_release_probe'] = expected
        row['max_release_probe_check'] = ('UNKNOWN' if expected is None else 'MISMATCH'
            if any(key.startswith('max_release_shadow.') for key in check['mismatches']) else 'MATCH')
    rows = result.get('per_decision', [])
    release = cell['native_victim_observation'].get('actual_release', {})
    result['status'] = ('VERIFIED' if raw is not None and result.get('source_verified') is True
        and release.get('status') == 'ANALYZED' and not release.get('actual_without_unique_native_decision')
        and all(row['actual_choice_verified'] and row['max_release_probe_check'] == 'MATCH'
            and row['actual_vs_immediate_release'] == 'MATCH' for row in rows) else 'MISMATCH_OR_MISSING')
    result.pop('shadow_check_counts', None)
    result['max_release_probe_check_counts'] = dict(Counter(row['max_release_probe_check'] for row in rows))
    counts = result.get('counts', {})
    counts.pop('shadow_changed_decisions', None)
    counts['unknown_max_release_probe_decisions'] = sum(row.get('recomputed_max_release_probe') is None for row in rows)
    counts['max_release_non_tail_proposals'] = (sum(
        row['recomputed_max_release_probe']['changed_from_tail'] for row in rows)
        if not counts['unknown_max_release_probe_decisions'] else None)
    cost = result.pop('max_release_shadow_cost', None)
    if cost is not None:
        cost['semantics'] = 'Max-release helper time in ACTIVE or SHADOW mode, already contained in outer selector time.'
        result['max_release_helper_cost'] = cost
    result['semantics'] = ('Inherited physical candidate observations and exact preempt/free joins. '
        'Replace only the old tail-only/SHADOW-only verdicts with both actual rules and their SHA-matched '
        'probe records. Actual free-pool increments must match selected immediate releasable pages. '
        'Tail capacity is an unexecuted pre-action observation when either active rule replaces it.')
    return result


def analyze(session, timeout_s=600, policy_source=None, input_config=None):
    session = Path(session)
    source = Path(policy_source) if policy_source else PACKAGE / 'pkg/staged_store_rotation.py'
    input_config = Path(input_config) if input_config else PACKAGE / 'pkg/inputs/pro_high/config.json'
    plan = base.read(session / 'plan.json')
    if (plan.get('experiment_role') != ROLE or plan.get('pinned_gpu_kv_bytes') != 77076627456
            or plan.get('source_profile_engine_differences') != inherited.PROFILE_ENGINE_DIFFERENCES
            or any(c.get('requests') != 320 or c.get('input_case') != 'high' for c in plan.get('cells', []))):
        raise ValueError('Expected original mixed320 remaining-budget/max-release ABBA with declared normal capacity')
    profile, provenance = partial.source_profile(session, plan)
    previous = base.KIND, base.RULES, base.verify_choices, base.read_native_cell, base.compare_cells
    original_read, original_compare = base.read_native_cell, base.compare_cells

    def read_cell(directory, spec, cell_plan, timeout, policy):
        cell, raw = original_read(directory, spec, cell_plan, timeout, policy)
        archive = base.archive_dir(directory)
        store = base.optional(archive / 'selective-store.json', {})
        environment = base.optional(archive / 'environment.json', {})
        executed = directory / 'candidate_native_oldest_strong_r01/pkg'
        chosen_source = executed / 'staged_store_rotation.py'
        chosen_source = chosen_source if chosen_source.is_file() else policy
        config_path = executed / 'inputs/pro_high/config.json'
        config_path = config_path if config_path.is_file() else input_config
        config, workload = base.read(config_path), base.read(config_path.with_name('workload.json'))
        cell['budget_consistency'] = mixed.budget_check(config, workload, cell.get('config') or {}, raw)
        cell['input_consistency'] = inherited.input_check(config, workload, raw)
        cell['input_consistency'].update(input_config_path=str(config_path), input_config_sha256=mixed.sha(config_path),
            input_workload_sha256=mixed.sha(config_path.with_name('workload.json')))
        cell['fixed_warmup_check'] = mixed.warmup_check(archive, cell.get('engine_args') or {})
        cell['physical_release_observations'] = physical_observations(cell, store, raw, chosen_source,
            environment.get('source_sha256', {}).get('staged_store_rotation.py'))
        cell['source_profile_capacity'] = inherited.capacity_check(profile, provenance, cell, environment, plan)
        cell['active_policy_outcomes'] = intervention_outcomes(cell, raw, store)
        cell['native_recovery_observations'] = exacthost.recovery_observations(
            base.optional(archive / 'offload-events.json'), raw, store, cell,
            chosen_source.parent / 'native_offload_observer.py',
            environment.get('source_sha256', {}).get('native_offload_observer.py'))
        choice, checks = cell['executed_choice_check'], cell['configuration_check']['checks']
        def check(field, actual, expected):
            checks.append(dict(field=field, actual=actual, expected=expected,
                matches=expected is not None and actual == expected and type(actual) is type(expected)))
        cell['policy_selector_costs'] = {}
        for rule, field in FIELDS.items():
            recorded = store.get(rule + '_probe') or {}
            expected = dict(mode='ACTIVE' if spec['native_victim_rule'] == rule else 'SHADOW',
                **choice['recomputed_probe_summaries'][rule])
            for key, value in expected.items():
                check(rule + '_probe.' + key, recorded.get(key), value)
            timings = [(event.get(field) or {}).get('selector_wall_s') for event in store.get('victim_decisions', [])]
            valid = [value for value in timings if base.finite(value) and value >= 0]
            total = sum(valid) if len(valid) == len(timings) and raw is not None else None
            cell['policy_selector_costs'][rule] = dict(recorded_decisions=len(valid),
                unknown_decisions=len(timings)-len(valid), total_recorded_s=total,
                distribution=base.distribution(valid), recorded_probe_total_s=recorded.get('selector_wall_s'),
                semantics='Helper timing in this arm is contained in outer selector time; do not add it again.')
            check(rule + '_probe.selector_wall_s', recorded.get('selector_wall_s'), total)
        check('both_policy_helpers_source', choice['source_verified'], True)
        check('config.enable_prefix_caching_default_false',
            (cell.get('config') or {}).get('enable_prefix_caching', False), False)
        check('engine.enable_prefix_caching', (cell.get('engine_args') or {}).get('enable_prefix_caching'), False)
        check('store.prefix_caching', store.get('prefix_caching'), False)
        check('store.coordinator_type', store.get('coordinator_type'), 'KVCacheCoordinatorNoPrefixCache')
        for key in ('budget_consistency', 'input_consistency', 'fixed_warmup_check',
                'physical_release_observations', 'source_profile_capacity', 'native_recovery_observations'):
            check(key, cell[key]['status'], 'VERIFIED')
        cell['configuration_check']['status'] = 'VERIFIED' if all(c['matches'] for c in checks) else 'MISMATCH_OR_MISSING'
        return cell, raw

    def compare(reference, candidate, old_raw, new_raw):
        result = original_compare(reference, candidate, old_raw, new_raw)
        result['pre_first_max_release_proposal_prefix'] = intervention_prefix(reference, candidate, old_raw, new_raw)
        result['active_intervention_participants'] = {}
        for label, cell, raw in (('reference', reference, old_raw), ('candidate', candidate, new_raw)):
            identity, participants = raw.get('internal_to_source', {}), []
            rule = cell['plan_cell']['native_victim_rule']
            events = cell['native_victim_observation']['raw_decisions']
            for index, event in proposals(events, FIELDS[rule]):
                ids = {name: identity.get(event.get(key)) for name, key in
                    (('original_tail', 'native_tail'), ('selected_victim', 'selected'), ('allocation_failed_request', 'failed_request'))}
                participants.append(dict(decision_index=index, step=event.get('step'), source_request_ids=ids,
                    all_request_comparison_indices=[i for i, row in enumerate(result['per_request']) if row['request_id'] in ids.values()]))
            result['active_intervention_participants'][label] = dict(rule=rule, per_proposal=participants)
        return result

    try:
        base.KIND, base.RULES, base.verify_choices = KIND, RULES, verify_choices
        base.read_native_cell, base.compare_cells = read_cell, compare
        result = base.analyze(session, timeout_s, source)
    finally:
        base.KIND, base.RULES, base.verify_choices, base.read_native_cell, base.compare_cells = previous
    if all(cell.get('status') in ('NOT_RUN', 'NO_MEASUREMENT') for cell in result['cells'].values()):
        result['status'] = 'NO_MEASUREMENT'
    result['experiment_role'], result['source_profile_provenance'] = ROLE, provenance
    result['analysis_code_sha256'] = {Path(module.__file__).name: mixed.sha(Path(module.__file__))
        for module in (base, inherited, mixed, partial, partial.once, partial.once.equal,
            partial.shared, partial.profiled, exacthost)}
    result['analysis_code_sha256'][Path(__file__).name] = mixed.sha(Path(__file__))
    result['semantics'] = dict(result['semantics'], intervention=('Two active complete online baselines: maximum declared '
        'remaining output budget versus maximum current physical releasable pages. Both use the native '
        'unprocessed suffix and can replace native tail repeatedly. Reference changes are counted and '
        'joined exactly as candidate changes. All-request metrics include failures, output changes and '
        'recovery costs. First-prefix diagnostics do not require subsequent trajectories to match. '
        'No EOS prediction, equal-capacity/work, causal net-benefit or novelty claim is implied.'))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', type=Path, required=True)
    parser.add_argument('--timeout-s', type=float, default=600)
    parser.add_argument('--policy-source', type=Path)
    parser.add_argument('--input-config', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.session, args.timeout_s, args.policy_source, args.input_config)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(status=result['status'], output=str(args.output))))
