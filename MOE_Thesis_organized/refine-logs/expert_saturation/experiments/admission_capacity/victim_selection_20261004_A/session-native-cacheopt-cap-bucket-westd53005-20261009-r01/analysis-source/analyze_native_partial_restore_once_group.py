#!/usr/bin/env python3
"""All-request ABBA analysis of one partial-restore native victim replacement."""
import argparse
import ast
from collections import Counter
import json
from pathlib import Path

import analyze_native_arrival_once_group as once
import analyze_native_group as base
import analyze_native_mixed_budget_probe as mixed
import analyze_native_shared_prefix_probe as shared
import analyze_native_shared_prefix_profiled as profiled


HERE = Path(__file__).resolve().parent
PACKAGE = HERE / 'candidate_native_partial_restore_once_r01'
KIND = 'PRO6000_NATIVE_PARTIAL_RESTORE_ONCE'
ROLE = 'PARTIAL_RESTORE_ONCE_INTERVENTION'
RULES = ('tail', 'partial_restore_once', 'partial_restore_once', 'tail')
FIELD = 'partial_restore_once'


def verify_choices(decisions, source, executed_sha, block_size=None):
    choose, source_sha = None, None
    if source is not None and source.is_file():
        source_sha = mixed.sha(source)
        if source_sha == executed_sha:
            node = next((n for n in ast.parse(source.read_bytes()).body
                if isinstance(n, ast.FunctionDef) and n.name == '_partial_restore_once_choice'), None)
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
            proposed, reason, triggered, eligible = choose(rows, indices[0],
                event['active_protection_or_phase'], consumed)
            picked = next(r for r in rows if r['index'] == proposed)
            tail = rows[-1]
            active = event['rule'] == 'partial_restore_once'
            different = picked['request'] != event['native_tail']
            requested = active and different
            selected = picked['request'] if active else event['native_tail']
            after = consumed or different
            a, b = tail.get('immediate_releasable_blocks'), picked.get('immediate_releasable_blocks')
            expected = dict(mode='ACTIVE' if active else 'SHADOW', triggered=triggered,
                proposed_request=picked['request'], returned_request=selected,
                action_requested=requested, fallback=reason, consumed_before=consumed,
                consumed_after=after, eligible_candidate_count=eligible,
                tail_releasable_blocks=a, proposed_releasable_blocks=b,
                extra_releasable_blocks=b-a if base.count_known(a) and base.count_known(b) else None)
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
            'Native suffix start is current_index. The old whole-suffix pure-decode flag is not a new guard. '
            'Tail consumes the same proposal in shadow; active consumption is a returned replacement, '
            'not proof of successful preemption. Replay uses no future output labels.')


def adapt_events(events):
    """Explicit schema mapping for the existing once-only observation readers."""
    return [dict(e, arrival_once=e.get(FIELD)) for e in events]


def adapt_cell(cell):
    observation = cell.get('native_victim_observation', {})
    return dict(cell, native_victim_observation=dict(observation,
        raw_decisions=adapt_events(observation.get('raw_decisions', []))))


def physical_observations(cell, store, raw, source, executed_sha):
    result = shared.shared_observations(cell, store, raw, source, executed_sha)
    checks = {c['decision_index']: c for c in cell['executed_choice_check']['checks']}
    for row in result.get('per_decision', []):
        row.pop('actual_tail_verified', None)
        row['actual_choice_verified'] = checks.get(row['decision_index'], {}).get('status') == 'MATCH'
    release = cell['native_victim_observation'].get('actual_release', {})
    verified = (raw is not None and result.get('source_verified') is True
        and release.get('status') == 'ANALYZED' and not release.get('actual_without_unique_native_decision')
        and all(r['actual_choice_verified'] and r['shadow_check'] == 'MATCH'
                and r['actual_vs_immediate_release'] == 'MATCH' for r in result.get('per_decision', [])))
    result['status'] = 'VERIFIED' if verified else 'MISMATCH_OR_MISSING'
    result['semantics'] = ('Shared physical observer reused on actual selected victims. Only its old '
        'tail-only verdict is replaced by this package\'s SHA-matched once-rule verdict. Max-release '
        'remains SHADOW; immediate release is checked against actual free-pool increments, not held '
        'pages. UNKNOWN physical state is retained; candidate observations are not independent runs.')
    return result


def once_outcome(cell, raw, store):
    adapted = dict(store, victim_decisions=adapt_events(store.get('victim_decisions', [])))
    result = once.once_outcome(cell, raw, adapted)
    index, event = once.proposal(adapted['victim_decisions'])
    if event is None:
        return result
    rows = {r['request']: r for r in event['candidates']}
    tail, proposed = rows[event['native_tail']], rows[event[FIELD]['proposed_request']]
    result['proposal_candidate_states'] = dict(original_tail=tail, proposed_victim=proposed)
    release = result.get('actual_released_blocks')
    tail_release = tail.get('immediate_releasable_blocks')
    result['actual_release_minus_original_tail_immediate_blocks'] = (
        release-tail_release if base.count_known(release) and base.count_known(tail_release) else None)
    result['semantics'] += (' Original-tail immediate release is an action-time physical observation, '
        'not a measured counterfactual action. Extra pages are reported; no equal-capacity or recovery '
        'cost benefit is inferred. Partial progress is a state trigger, not past-cost protection.')
    return result


def intervention_prefix(reference, candidate, old_raw, new_raw):
    adapted = [adapt_cell(c) for c in (reference, candidate)]
    result = once.intervention_prefix(*adapted, old_raw, new_raw)
    result['proposal_field_mapping'] = 'partial_restore_once -> arrival_once -> equal_held_once, reader adaptation only'
    fields = ('prompt_tokens', 'request_status', 'immediate_releasable_blocks', 'shared_blocks',
              'release_state_error', 'original_arrival_relative_s')
    snapshots, unknown = [], []
    for cell, raw in zip(adapted, (old_raw, new_raw)):
        events = cell['native_victim_observation']['raw_decisions']
        index, proposal = once.proposal(events)
        if proposal is None:
            return result
        states, missing = {}, 0
        identity = raw.get('internal_to_source', {})
        for event in events[:index+1]:
            for row in event.get('candidates', []):
                key = (event.get('step'), identity.get(event.get('failed_request')),
                       row.get('index'), identity.get(row.get('request')))
                missing += int(any(v is None for v in key)) + int(key in states)
                missing += sum(f not in row for f in fields[:-1])
                states[key] = {f: row.get(f) for f in fields[:-1]}
                relative = base.delta(raw.get('measurement_origin_unix_s'), row.get('arrival_time'))
                missing += int(relative is None)
                states[key]['original_arrival_relative_s'] = relative
        snapshots.append(states); unknown.append(missing)
    common = snapshots[0].keys() & snapshots[1].keys()
    changes, examples = Counter(), []
    for key in sorted(common, key=repr):
        differences = {f: dict(reference=snapshots[0][key][f], candidate=snapshots[1][key][f])
                       for f in fields if snapshots[0][key][f] != snapshots[1][key][f]}
        if differences:
            changes.update(differences.keys())
            if len(examples) < 12:
                examples.append(dict(step=key[0], failed_request=key[1], index=key[2], request=key[3], differences=differences))
    result['physical_progress_prefix'] = dict(fields=fields, row_counts=[len(x) for x in snapshots],
        common_rows=len(common), exclusive_rows=[len(x)-len(common) for x in snapshots],
        changed_fields=dict(changes), difference_examples=examples, unknown_counts=unknown,
        exact_observed_equal=not any(unknown) and snapshots[0] == snapshots[1],
        semantics='Additional recorded prompt/progress and live physical counts through each proposal. '
            'Original epoch arrivals are expressed relative to each run measurement_origin_unix_s; '
            'tiny clock differences are preserved without interpreting them as workload changes. '
            'Host/progress/output and original-arrival order comparisons are inherited. Equal fields '
            'cannot prove identical hidden state; differing prefixes are reported, not adjusted away.')
    if any(unknown):
        result['status'] = 'INCOMPLETE_OBSERVATION'
    return result


def source_profile(session, plan):
    declared = plan.get('source_profile') or {}
    source_session, relative = declared.get('session'), declared.get('relative_path')
    path = (session.parent / source_session / relative
            if isinstance(source_session, str) and isinstance(relative, str) else None)
    exists = path is not None and path.is_file()
    actual_sha = mixed.sha(path) if exists else None
    profile = profiled.read_profile(path.parent.parent) if exists else None
    checks = dict(source_exists=exists, source_sha256=exists and actual_sha == declared.get('sha256'),
        authorized_uuid=bool(plan.get('authorized_gpu_uuid'))
            and declared.get('authorized_gpu_uuid') == plan.get('authorized_gpu_uuid'),
        normal_utilization=declared.get('normal_gpu_memory_utilization') == 0.9)
    return profile, dict(status='UNKNOWN_SOURCE_PROFILE' if not exists else
        'VERIFIED' if all(checks.values()) else 'MISMATCH_OR_MISSING', checks=checks,
        path=str(path) if path else None, declared=declared, observed_sha256=actual_sha,
        semantics='Previously completed normal profile on this authorized device, not a new profile '
            'or an outcome from this ABBA group. Missing referenced evidence is UNKNOWN.')


def capacity_check(profile, provenance, cell, environment, plan):
    if profile is None:
        return dict(status='UNKNOWN_SOURCE_PROFILE', source_profile=provenance['path'],
                    semantics='No source profile evidence; measured request outcomes remain available.')
    result = profiled.capacity_check(profile, cell, environment, plan)
    # This group intentionally adds one selector to the prior profile package.
    # Runtime source, engine layout, physical capacities and UUID must still match.
    result['checks'].pop('same_candidate_source')
    result['checks']['source_profile_provenance'] = provenance['status'] == 'VERIFIED'
    result['checks']['plan_pin_from_source_profile'] = (
        profile['capacity'].get('pin_kv_cache_memory_bytes') == plan.get('pinned_gpu_kv_bytes'))
    result['status'] = 'VERIFIED' if all(result['checks'].values()) else 'MISMATCH_OR_MISSING'
    result['semantics'] = ('Each arm uses the referenced completed normal profile on the same GPU UUID, '
        'with matching Host/GPU page layout, engine settings and native source. Candidate selector '
        'source differs intentionally from that prior profile; current arms must match each other.')
    return result


def analyze(session, timeout_s=600, policy_source=None, input_config=None):
    session = Path(session)
    policy_source = Path(policy_source) if policy_source else PACKAGE / 'pkg/staged_store_rotation.py'
    input_config = Path(input_config) if input_config else PACKAGE / 'pkg/inputs/pro_high/config.json'
    plan = base.read(session / 'plan.json')
    if (plan.get('experiment_role') != ROLE or plan.get('pinned_gpu_kv_bytes') != 77076627456
            or any(c.get('requests') != 320 or c.get('input_case') != 'high' for c in plan.get('cells', []))):
        raise ValueError('Expected the shared320 partial-restore intervention with fresh-device pinned normal capacity')
    profile, provenance = source_profile(session, plan)
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
        cell['shared_prefix_observations'] = physical_observations(cell, store, raw, chosen_source,
            environment.get('source_sha256', {}).get('staged_store_rotation.py'))
        cell['source_profile_capacity'] = capacity_check(profile, provenance, cell, environment, plan)
        cell['partial_restore_once_outcome'] = once_outcome(cell, raw, store)
        choice = cell['executed_choice_check']
        checks = cell['configuration_check']['checks']
        expected = dict(mode='ACTIVE' if spec['native_victim_rule'] == 'partial_restore_once' else 'SHADOW',
            consumed=choice['recomputed_consumed'], trigger_count=choice['recomputed_trigger_count'],
            proposal_count=choice['recomputed_proposal_count'], action_request_count=choice['recomputed_action_request_count'])
        recorded = store.get('partial_restore_once_probe') or {}
        for key, value in expected.items():
            actual = recorded.get(key)
            checks.append(dict(field='partial_restore_once_probe.'+key, actual=actual, expected=value,
                matches=value is not None and actual == value and type(actual) is type(value)))
        checks.append(dict(field='partial_restore_once_helper_source', actual=choice['source_verified'],
            expected=True, matches=choice['source_verified']))
        for key in ('budget_consistency', 'input_consistency', 'fixed_warmup_check',
                    'shared_prefix_domain', 'shared_prefix_observations', 'source_profile_capacity'):
            checks.append(dict(field=key, actual=cell[key]['status'], expected='VERIFIED',
                               matches=cell[key]['status'] == 'VERIFIED'))
        timings = [e[FIELD].get('selector_wall_s') for e in store.get('victim_decisions', [])
                   if isinstance(e.get(FIELD), dict)]
        valid = [v for v in timings if base.finite(v) and v >= 0]
        cell['partial_restore_selector_cost'] = dict(recorded_decisions=len(valid),
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
        for m in (base, once, once.equal, mixed, shared, profiled)} | {Path(__file__).name: mixed.sha(Path(__file__))}
    result['semantics'] = dict(result['semantics'], intervention=
        'At most one state-triggered private victim replacement, then tail. Tail uses the same once shadow. '
        'A partial restore has existing outputs but insufficient computed history; an alternative pure decode '
        'must release at least as many immediate physical pages. Full-request natural-EOS output differences, '
        'extra released pages and observed pre-action discrepancies remain explicit. No equal-work, '
        'equal-capacity, strict same-state or recovery-cost benefit is assumed. No application goodput threshold is selected.')
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
