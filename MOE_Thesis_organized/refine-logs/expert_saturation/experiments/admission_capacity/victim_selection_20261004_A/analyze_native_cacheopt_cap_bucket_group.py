#!/usr/bin/env python3
"""Thin dual-active analysis for a fixed CacheOPT-inspired cap-bucket component."""
import argparse
import ast
from collections import Counter
import json
from pathlib import Path

import analyze_native_max_release_group as inherited


base, mixed, partial = inherited.base, inherited.mixed, inherited.partial
HERE = Path(__file__).resolve().parent
PACKAGE = HERE / 'candidate_native_cacheopt_cap_bucket_r01'
KIND, ROLE = 'PRO6000_NATIVE_CACHEOPT_CAP_BUCKET', 'CACHEOPT_CAP_BUCKET_INTERVENTION'
RULE = 'cacheopt_cap_bucket'
RULES = ('remaining_budget', RULE, RULE, 'remaining_budget')
FIELDS = {'remaining_budget': 'remaining_budget', RULE: RULE, 'max_release': 'max_release_shadow'}


def verify_choices(decisions, source, executed_sha, block_size=None):
    helpers, source_sha = {}, None
    if source is not None and source.is_file():
        source_sha = mixed.sha(source)
        if source_sha == executed_sha:
            names = {'_remaining_budget_choice', '_max_release_shadow_choice', '_cacheopt_cap_bucket_choice'}
            nodes = [node for node in ast.parse(source.read_bytes()).body
                if isinstance(node, ast.FunctionDef) and node.name in names]
            if {node.name for node in nodes} == names:
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
        elif event.get('rule') not in RULES:
            item['status'] = 'UNSUPPORTED_RULE'
        elif type(event.get('active_protection_or_phase')) is not bool:
            item['status'] = 'REQUIRED_GUARD_NOT_RECORDED'
        elif any(key not in row for row in rows for key in
                ('max_tokens', 'output_tokens', 'held_blocks', 'request_status', 'immediate_releasable_blocks')):
            item['status'] = 'UNKNOWN_REQUIRED_FIELDS_NOT_RECORDED'
        else:
            phase, tail = event['active_protection_or_phase'], rows[-1]
            by_index = {row['index']: row for row in rows}
            release, release_reason = helpers['_max_release_shadow_choice'](rows, phase)
            budget, budget_reason, triggered, matching, improving = helpers['_remaining_budget_choice'](
                rows, phase, block_size)
            bucket, bucket_reason = helpers['_cacheopt_cap_bucket_choice'](rows, phase, block_size)
            proposed = {'max_release': release, 'remaining_budget': budget, RULE: bucket}
            reasons = {'max_release': release_reason, 'remaining_budget': budget_reason, RULE: bucket_reason}
            selected = by_index[proposed[event['rule']]]['request']
            changed = selected != event['native_tail']
            expected, mismatches = {}, []
            for rule, field in FIELDS.items():
                picked, active = by_index[proposed[rule]], event['rule'] == rule
                different = picked['request'] != event['native_tail']
                probe = dict(mode='ACTIVE' if active else 'SHADOW', proposed_request=picked['request'],
                    returned_request=selected, action_requested=active and different, fallback=reasons[rule])
                if rule == RULE:
                    probe['changed_from_tail'] = different
                else:
                    probe.update(tail_releasable_blocks=tail.get('immediate_releasable_blocks'),
                        proposed_releasable_blocks=picked.get('immediate_releasable_blocks'))
                    if rule == 'max_release':
                        probe['changed_from_tail'] = different
                    else:
                        probe.update(triggered=triggered, matching_candidate_count=matching,
                            improving_candidate_count=improving,
                            tail_remaining_output_budget=inherited.inherited.remaining(tail),
                            proposed_remaining_output_budget=inherited.inherited.remaining(picked))
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
            proposal_count=sum(probe['proposed_request'] != event.get('native_tail')
                for probe, event in zip(probes, decisions)) if known else None,
            action_request_count=sum(probe['action_requested'] for probe in probes) if known else None,
            fallback_counts=dict(Counter(probe['fallback'] for probe in probes if probe['fallback'] is not None))
                if known else None)
    return dict(source=str(source) if source else None, source_sha256=source_sha,
        executed_source_sha256=executed_sha, source_verified=bool(helpers),
        status_counts=dict(Counter(row['status'] for row in records)), checks=records,
        recomputed_probe_summaries=summaries,
        semantics='SHA-matched replay of both active alternatives and the max-release shadow. '
            'Fixed 128-token buckets rank ceil(declared remaining cap/128) descending, '
            'ceil(held pages * actual block size/128) ascending, then latest suffix index. '
            'No SLO dimension, EOS predictor, host/pending ranking, or marginal-release substitution. '
            'SHADOW returned_request is still the actual arm choice; reference replacements count. '
            'Missing fields remain UNKNOWN; explicit nulls may verify a fail-closed fallback.')


def intervention_prefix(reference, candidate, old_raw, new_raw):
    adapted, indices, boundaries, unknown_counts = [], [], [], []
    for cell, raw in zip((reference, candidate), (old_raw, new_raw)):
        observation = cell.get('native_victim_observation', {})
        events = observation.get('raw_decisions', [])
        index, boundary, unknown = None, None, 0
        identity = raw.get('internal_to_source', {})
        for i, event in enumerate(events):
            budget = (event.get('remaining_budget') or {}).get('proposed_request')
            bucket = (event.get(RULE) or {}).get('proposed_request')
            if not isinstance(budget, str) or not isinstance(bucket, str):
                unknown += 1
                continue
            if budget != bucket:
                index = i
                boundary = dict(decision_index=i, step=event.get('step'),
                    remaining_budget=identity.get(budget), cacheopt_cap_bucket=identity.get(bucket),
                    selected=identity.get(event.get('selected')), native_tail=identity.get(event.get('native_tail')),
                    cap_component_selects_tail=bucket == event.get('native_tail'))
                break
        indices.append(index); boundaries.append(boundary); unknown_counts.append(unknown)
        prefix_events = events[:index+1] if index is not None else events
        adapted.append(dict(cell, native_victim_observation=dict(observation,
            raw_decisions=inherited.mark_one(prefix_events, index, RULE))))
    result = partial.intervention_prefix(*adapted, old_raw, new_raw)
    if any(unknown_counts):
        result['status'] = 'INCOMPLETE_OBSERVATION'
    elif any(index is None for index in indices):
        result['status'] = 'NO_OBSERVED_ACTIVE_RULE_DISAGREEMENT_IN_ONE_OR_BOTH_RUNS'
    result.update(first_active_rule_disagreement_indices=indices,
        first_active_rule_disagreements=boundaries,
        unknown_active_rule_proposals_before_boundary=unknown_counts,
        proposal_field_mapping='First remaining_budget != cacheopt_cap_bucket suggestion -> inherited prefix marker; even if cap component selects tail',
        semantics='Only the observed prefix before the FIRST disagreement of the two active-rule '
            'suggestions is compared, including the boundary candidate snapshot. Earlier agreed '
            'non-tail actual actions are retained. The cutoff is not postponed to the cap component\'s '
            'first non-tail proposal. Missing earlier suggestions make the first boundary UNKNOWN. '
            'Two trajectories may reach different boundaries; no same-state causal claim, clock '
            'subtraction, once restriction, or later-trajectory equality is required.')
    return result


def analyze(session, timeout_s=600, policy_source=None, input_config=None):
    names = ('PACKAGE', 'KIND', 'ROLE', 'RULES', 'FIELDS', 'verify_choices', 'intervention_prefix')
    previous = {name: getattr(inherited, name) for name in names}
    try:
        inherited.PACKAGE, inherited.KIND, inherited.ROLE = PACKAGE, KIND, ROLE
        inherited.RULES, inherited.FIELDS = RULES, FIELDS
        inherited.verify_choices, inherited.intervention_prefix = verify_choices, intervention_prefix
        result = inherited.analyze(session, timeout_s, policy_source,
            input_config or PACKAGE / 'pkg/inputs/pro_high/config.json')
    finally:
        for name, value in previous.items():
            setattr(inherited, name, value)
    for cell in result['cells'].values():
        for check in cell.get('configuration_check', {}).get('checks', []):
            if check['field'] == 'both_policy_helpers_source':
                check['field'] = 'both_active_helpers_and_max_release_shadow_source'
    for comparison in result['comparisons'].values():
        old = 'pre_first_max_release_proposal_prefix'
        if old in comparison:
            comparison['pre_first_active_rule_disagreement_prefix'] = comparison.pop(old)
    result['analysis_code_sha256'][Path(__file__).name] = mixed.sha(Path(__file__))
    result['semantics']['intervention'] = ('Two complete online active rules on the native unprocessed '
        'suffix: exact maximum declared remaining budget and fixed 128-token cap/held-capacity buckets. '
        'The reference also replaces native tail. Max-release remains an observed shadow in both arms. '
        'This is a no-application-SLO, cap-proxy victim component inspired by CacheOPT section 3.4, '
        'not its output predictor, complete system, faithful reproduction or an innovation claim. '
        'All-request mean flow remains primary; preserve output/termination differences, actual release, '
        'recovery, costs and failures. First disagreement is diagnostic, not a later-prefix validity gate.')
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
