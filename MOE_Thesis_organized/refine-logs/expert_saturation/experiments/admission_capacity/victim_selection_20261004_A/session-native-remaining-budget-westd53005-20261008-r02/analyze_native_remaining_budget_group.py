#!/usr/bin/env python3
"""Reuse all-request analysis; replace only the full-suffix budget policy contract."""
import argparse
import ast
from collections import Counter
import json
from pathlib import Path

import analyze_native_equal_release_budget_group_r02 as inherited

HERE = Path(__file__).resolve().parent
PACKAGE = HERE / 'candidate_native_remaining_budget_r01'
FIELD = 'remaining_budget'


def verify_choices(decisions, source, executed_sha, block_size=None):
    choose, source_sha = None, None
    if source is not None and source.is_file():
        source_sha = inherited.mixed.sha(source)
        if source_sha == executed_sha:
            node = next((n for n in ast.parse(source.read_bytes()).body
                if isinstance(n, ast.FunctionDef) and n.name == '_remaining_budget_choice'), None)
            if node is not None:
                namespace = {}
                exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), namespace)
                choose = namespace[node.name]
    records, fallbacks = [], Counter()
    proposals = requests = 0
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
        elif event.get('rule') not in ('tail', FIELD):
            item['status'] = 'UNSUPPORTED_RULE'
        elif type(event.get('active_protection_or_phase')) is not bool:
            item['status'] = 'REQUIRED_GUARD_NOT_RECORDED'
        elif any(key not in row for row in rows for key in ('max_tokens', 'output_tokens')):
            item['status'] = 'UNKNOWN_REQUIRED_FIELDS_NOT_RECORDED'
        else:
            proposed, reason, triggered, matching, improving = choose(
                rows, event['active_protection_or_phase'], block_size)
            picked = next(row for row in rows if row['index'] == proposed)
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
                tail_remaining_output_budget=inherited.remaining(tail),
                proposed_remaining_output_budget=inherited.remaining(picked))
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
    known = choose is not None and all('expected_probe' in row for row in records)
    return dict(source=str(source) if source else None, source_sha256=source_sha,
        executed_source_sha256=executed_sha, source_verified=choose is not None,
        status_counts=dict(Counter(row['status'] for row in records)), checks=records,
        recomputed_decision_count=len(records) if known else None,
        recomputed_proposal_count=proposals if known else None,
        recomputed_action_request_count=requests if known else None,
        recomputed_fallback_counts=dict(fallbacks) if known else None,
        semantics='SHA-matched online helper over the complete native unprocessed suffix. '
            'Only declared cap/output and active phase/protection qualify the ranking. '
            'Physical release, progress stage, host state and pending STORE are observations, '
            'not eligibility gates. Remaining cap is not predicted EOS or wallclock finish time.')


def analyze(session, timeout_s=600, policy_source=None, input_config=None):
    names = ('PACKAGE', 'KIND', 'ROLE', 'FIELD', 'RULES', 'verify_choices')
    previous = {name: getattr(inherited, name) for name in names}
    try:
        inherited.PACKAGE = PACKAGE
        inherited.KIND = 'PRO6000_NATIVE_REMAINING_BUDGET'
        inherited.ROLE = 'REMAINING_BUDGET_INTERVENTION'
        inherited.FIELD = FIELD
        inherited.RULES = ('tail', FIELD, FIELD, 'tail')
        inherited.verify_choices = verify_choices
        result = inherited.analyze(session, timeout_s, policy_source,
            input_config or PACKAGE / 'pkg/inputs/pro_high/config.json')
    finally:
        for name, value in previous.items():
            setattr(inherited, name, value)
    for cell in result['cells'].values():
        for suffix in ('outcomes', 'selector_cost'):
            old = 'equal_release_budget_' + suffix
            if old in cell:
                cell[FIELD + '_' + suffix] = cell.pop(old)
        for check in cell.get('configuration_check', {}).get('checks', []):
            if check['field'] == 'equal_release_budget_helper_source':
                check['field'] = FIELD + '_helper_source'
    for comparison in result['comparisons'].values():
        prefix = comparison.get('pre_first_intervention_prefix', {})
        if 'proposal_field_mapping' in prefix:
            prefix['proposal_field_mapping'] = prefix['proposal_field_mapping'].replace('equal_release_budget', FIELD)
    result.pop('analysis_correction', None)
    result['analysis_code_sha256'][Path(__file__).name] = inherited.mixed.sha(Path(__file__))
    result['semantics']['intervention'] = ('Complete online maximum declared remaining-budget '
        'victim baseline on the native unprocessed suffix, including native-legal current, '
        'pending, partial and zero-output requests. No equal-release/computed gate. '
        'Physical-release differences and later repeated preemptions are required side effects. '
        'First-prefix comparison is diagnostic, not a full-strategy validity requirement. '
        'No predicted EOS, equal-work, new-method or full CacheOPT reproduction claim.')
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
