#!/usr/bin/env python3
"""One tail cell: assigned-budget consistency and descriptive victim shadows."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import analyze_native_group as base


HERE = Path(__file__).resolve().parent
PACKAGE = HERE / 'candidate_native_mixed_budget_probe_r01'
ROLE = 'MIXED_BUDGET_NATIVE_TAIL_CHARACTERIZATION'
PREFIX = 'A-budget-mixture-v1:'
SEMANTICS = ('Development opportunity diagnosis on one synthetic budget-mixture tail run. '
    'Remaining means assigned max_tokens minus emitted output_tokens, not tokens until natural EOS. '
    'Shadows describe alternatives at observed tail states; they were not executed and are not '
    'same-state causal comparisons, predictions of completion benefit, or gains versus uniform-budget runs. '
    'One KV page is a descriptive token count, not an application SLO. Decisions are not independent repeats.')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def budget_check(input_config, workload, measured_config, raw):
    """Independently reconstruct the preregistered, outcome-blind assignment."""
    ids = [row['request_id'] for row in workload['source_requests']]
    short = sorted(ids, key=lambda rid: (hashlib.sha256((PREFIX + rid).encode()).hexdigest(), rid))[:40]
    overrides = {rid: 128 for rid in short}
    expected = {rid: overrides.get(rid, 1024) for rid in ids}
    checks = dict(input_count=len(ids) == len(set(ids)) == 320,
        input_global_cap=type(input_config.get('output_tokens')) is int and input_config['output_tokens'] == 1024,
        input_overrides=input_config.get('output_tokens_by_request') == overrides
            and all(type(v) is int for v in input_config.get('output_tokens_by_request', {}).values()),
        input_natural_eos=input_config.get('ignore_eos') is False and input_config.get('min_tokens') == 0,
        measured_global_cap=type(measured_config.get('output_tokens')) is int and measured_config['output_tokens'] == 1024,
        measured_overrides=measured_config.get('output_tokens_by_request') == overrides
            and all(type(v) is int for v in measured_config.get('output_tokens_by_request', {}).values()),
        measured_natural_eos=measured_config.get('ignore_eos') is False and measured_config.get('min_tokens') == 0)
    rows = raw.get('requests', []) if raw is not None else []
    actual_ids = [row.get('request_id') for row in rows]
    checks['measured_request_ids'] = (len(actual_ids) == len(set(actual_ids)) == 320
                                    and set(actual_ids) == set(ids)) if raw is not None else None
    mismatches = [dict(request=row.get('request_id'), actual=row.get('max_output_tokens'),
                       expected=expected.get(row.get('request_id'))) for row in rows
                  if type(row.get('max_output_tokens')) is not int
                  or row.get('max_output_tokens') != expected.get(row.get('request_id'))]
    checks['measured_request_caps'] = not mismatches if raw is not None else None
    checks['actual_prompt_context_bound'] = (all(type(row.get('prompt_tokens')) is int
        and type(row.get('max_output_tokens')) is int
        and 0 < row['prompt_tokens'] + row['max_output_tokens'] <= 4096 for row in rows)
        if raw is not None else None)
    return dict(status='VERIFIED' if all(v is True for v in checks.values()) else 'MISMATCH_OR_MISSING',
        checks=checks, expected_cap_counts={'128': 40, '1024': 280},
        observed_cap_counts=dict(Counter(str(r.get('max_output_tokens')) for r in rows)),
        mismatches=mismatches, assigned_short_request_ids=short,
        assignment_prefix=PREFIX, assignment_metadata=input_config.get('budget_mixture'))


def warmup_check(archive, engine):
    """Check the inherited fixed-length warmups, including the wide warmup formula."""
    width = engine.get('max_num_seqs')
    batched = engine.get('max_num_batched_tokens')
    expected_widths = [min(32, width), width, 2] if type(width) is int and width > 0 else [None] * 3
    records = []
    for index, expected_width in enumerate(expected_widths):
        path = archive / f'warmup-{index}.json'
        warm = base.optional(path, {})
        rows, coverage = warm.get('requests', []), warm.get('warmup_coverage', {})
        valid_prompts = bool(rows) and all(type(r.get('prompt_tokens')) is int and r['prompt_tokens'] > 0 for r in rows)
        expected_cap = None
        if expected_width is not None and type(batched) is int and valid_prompts:
            expected_cap = (16 if expected_width <= 32 else 16 + math.ceil(expected_width
                * max(r['prompt_tokens'] for r in rows) / max(1, batched - expected_width)))
        checks = dict(complete=warm.get('status') == 'COMPLETE',
            width=expected_width is not None and len(rows) == expected_width == coverage.get('requested_width'),
            coverage_budget=expected_cap is not None and coverage.get('requested_output_tokens') == expected_cap,
            fixed_caps=bool(rows) and expected_cap is not None
                and all(r.get('max_output_tokens') == expected_cap for r in rows),
            fixed_outputs=bool(rows) and expected_cap is not None and all(r.get('status') == 'completed'
                and len(r.get('output_token_ids', [])) == expected_cap for r in rows))
        records.append(dict(file=path.name, checks=checks, expected_width=expected_width,
            expected_output_cap=expected_cap, observed_requests=len(rows),
            observed_cap_counts=dict(Counter(str(r.get('max_output_tokens')) for r in rows)), coverage=coverage))
    return dict(status='VERIFIED' if all(all(r['checks'].values()) for r in records)
                else 'MISMATCH_OR_MISSING', cells=records,
                semantics='Warmups retain the inherited fixed-output rule; measurement budget overrides must not leak into them.')


def remaining(row):
    cap, emitted = row.get('max_tokens'), row.get('output_tokens')
    return cap - emitted if type(cap) is int and type(emitted) is int and 2 <= cap and 0 <= emitted <= cap else None


def remaining_distribution(values):
    known = [v for v in values if v is not None]
    return dict(base.distribution(known), minimum=min(known) if known else None)


def remaining_shadows(decisions, raw, block_size):
    """Reuse recorded legal suffix and guards; never infer missing qualification."""
    if type(block_size) is not int or block_size <= 0:
        return dict(status='UNKNOWN_BLOCK_SIZE', block_size_tokens=block_size, semantics=SEMANTICS)
    identity = (raw or {}).get('internal_to_source', {})
    caps = {r['request_id']: r.get('max_output_tokens') for r in (raw or {}).get('requests', [])}
    events, budget_mismatches = [], []
    for ordinal, event in enumerate(decisions):
        rows = event.get('candidates', [])
        indices = [r.get('index') for r in rows]
        tail = rows[-1] if rows else {}
        reason = None
        if (not rows or any(type(i) is not int for i in indices)
                or indices != list(range(indices[0], indices[-1] + 1))
                or event.get('unprocessed_suffix_start') != indices[0]
                or len({r.get('request') for r in rows}) != len(rows)
                or tail.get('request') != event.get('native_tail')):
            reason = 'INVALID_SUFFIX'
        elif (type(event.get('fallback_unknown')) is not bool
                or type(event.get('active_protection_or_phase')) is not bool
                or any(type(r.get('qualified')) is not bool for r in rows)):
            reason = 'UNKNOWN_GUARDS'
        elif event['fallback_unknown'] != any(not r['qualified'] for r in rows):
            reason = 'INCONSISTENT_QUALIFICATION'
        elif event['fallback_unknown']:
            reason = 'UNKNOWN_SUFFIX_STATE'
        elif event['active_protection_or_phase']:
            reason = 'ACTIVE_PROTECTION_OR_PHASE'
        elif any(remaining(r) is None for r in rows):
            reason = 'UNKNOWN_OUTPUT_BUDGET'
        bad_caps = [dict(decision_index=ordinal, request=r.get('request'),
                        source_request=identity.get(r.get('request')), candidate_cap=r.get('max_tokens'),
                        measured_cap=caps.get(identity.get(r.get('request')))) for r in rows
                    if identity.get(r.get('request')) not in caps
                    or r.get('max_tokens') != caps.get(identity.get(r.get('request')))]
        budget_mismatches.extend(bad_caps)
        if reason is None and bad_caps:
            reason = 'UNKNOWN_OR_MISMATCHED_MEASURED_CAP'
        item = dict(decision_index=ordinal, step=event.get('step'), native_tail=event.get('native_tail'),
            actual_selected=event.get('selected'), actual_tail_verified=(event.get('rule') == 'tail'
                and event.get('changed') is False and event.get('selected') == event.get('native_tail')),
            tail_remaining=remaining(tail), candidates=[dict(request=r.get('request'), index=r.get('index'),
                qualified=r.get('qualified'), max_tokens=r.get('max_tokens'), output_tokens=r.get('output_tokens'),
                remaining=remaining(r)) for r in rows])
        for version in ('all_legal_suffix', 'same_held_and_computed_pages'):
            fallback, choices = reason, rows
            if version == 'same_held_and_computed_pages' and fallback is None:
                if any(type(r.get('held_blocks')) is not int or r['held_blocks'] < 0
                        or type(r.get('computed_tokens')) is not int or r['computed_tokens'] < 0 for r in rows):
                    fallback = 'UNKNOWN_CAPACITY_STATE'
                else:
                    choices = [r for r in rows if r['held_blocks'] == tail['held_blocks']
                        and r['computed_tokens'] // block_size == tail['computed_tokens'] // block_size]
            selected = tail if fallback else max(choices, key=lambda r: (remaining(r), r['index']))
            item[version] = dict(fallback=fallback, proposed_request=selected.get('request'),
                changed_from_tail=selected.get('request') != event.get('native_tail') if rows else None,
                eligible_candidates=len(choices) if fallback is None else None,
                candidates_remaining_le_one_page=(sum(remaining(r) <= block_size for r in choices)
                    if fallback is None else None),
                proposed_remaining=remaining(selected),
                held_difference=(selected['held_blocks'] - tail['held_blocks']
                    if type(selected.get('held_blocks')) is int and type(tail.get('held_blocks')) is int else None))
        events.append(item)
    invalid_fallbacks = {'INVALID_SUFFIX', 'UNKNOWN_GUARDS', 'INCONSISTENT_QUALIFICATION'}
    result = dict(status='VERIFIED' if not budget_mismatches and all(e['actual_tail_verified']
                  and e['all_legal_suffix']['fallback'] not in invalid_fallbacks for e in events)
                  else 'MISMATCH_OR_MISSING', block_size_tokens=block_size, decision_count=len(events),
        actual_non_tail_decisions=sum(not e['actual_tail_verified'] for e in events),
        candidate_budget_mismatches=budget_mismatches,
        tail_remaining_distribution=remaining_distribution([e['tail_remaining'] for e in events]),
        tail_remaining_unknown=sum(e['tail_remaining'] is None for e in events),
        tail_remaining_le_one_page=sum(e['tail_remaining'] is not None and e['tail_remaining'] <= block_size for e in events),
        decisions=events, semantics=SEMANTICS)
    for version in ('all_legal_suffix', 'same_held_and_computed_pages'):
        ranked = [e for e in events if e[version]['fallback'] is None]
        changed = [e for e in ranked if e[version]['changed_from_tail']]
        result[version] = dict(rankable_decisions=len(ranked),
            fallback_counts=dict(Counter(e[version]['fallback'] for e in events if e[version]['fallback'])),
            alternative_opportunities=sum(e[version]['eligible_candidates'] > 1 for e in ranked),
            changed_from_tail=len(changed),
            tail_remaining_distribution=remaining_distribution([e['tail_remaining'] for e in ranked]),
            tail_remaining_le_one_page=sum(e['tail_remaining'] <= block_size for e in ranked),
            near_budget_with_alternative=sum(e['tail_remaining'] <= block_size and e[version]['eligible_candidates'] > 1 for e in ranked),
            near_budget_with_changed_shadow=sum(e['tail_remaining'] <= block_size for e in changed),
            candidate_observations_remaining_le_one_page=sum(e[version]['candidates_remaining_le_one_page'] for e in ranked),
            decisions_with_any_candidate_remaining_le_one_page=sum(e[version]['candidates_remaining_le_one_page'] > 0 for e in ranked),
            changed_held_difference=base.distribution([e[version]['held_difference'] for e in changed]),
            changed_remaining_increase=base.distribution([e[version]['proposed_remaining'] - e['tail_remaining'] for e in changed]))
    return result


def analyze(session, policy_source, input_config):
    plan = base.read(session / 'plan.json')
    cells = plan.get('cells', [])
    if (plan.get('kind') != 'PRO6000_NATIVE_EQUAL_HELD_ONCE' or plan.get('experiment_role') != ROLE
            or len(cells) != 1 or cells[0].get('label') != 'mixed-budget-tail'
            or cells[0].get('native_victim_rule') != 'tail' or cells[0].get('requests') != 320
            or cells[0].get('input_case') != 'high'):
        raise ValueError('Expected the declared single mixed-budget-tail characterization plan')
    spec = cells[0]
    directory = session / ('cell-00-' + spec['label'])
    timeout = plan.get('configuration', {}).get('max_seconds', 600)
    cell, raw = base.read_native_cell(directory, spec, plan, timeout, policy_source)
    archive = base.archive_dir(directory)
    executed_input = directory / 'candidate_native_oldest_strong_r01/pkg/inputs/pro_high/config.json'
    config_path = executed_input if executed_input.exists() else input_config
    config, workload = base.read(config_path), base.read(config_path.with_name('workload.json'))
    budget = budget_check(config, workload, cell.get('config') or {}, raw)
    budget.update(input_config_path=str(config_path), input_config_sha256=sha(config_path),
        input_workload_sha256=sha(config_path.with_name('workload.json')),
        input_source='EXECUTED_CELL_PACKAGE' if executed_input.exists() else 'SUPPLIED_PACKAGE_FALLBACK')
    warmup = warmup_check(archive, cell.get('engine_args', {}))
    decisions = cell.get('native_victim_observation', {}).get('raw_decisions', [])
    shadows = remaining_shadows(decisions, raw, cell.get('capacity_profile', {}).get('block_size_tokens'))
    status = ('COMPLETE_CHARACTERIZATION' if cell.get('status') == 'COMPLETE'
        and budget['status'] == warmup['status'] == shadows['status'] == 'VERIFIED'
        and cell['configuration_check']['status'] == 'VERIFIED' else 'INCOMPLETE_OR_INVALID')
    return dict(status=status, experiment_role=ROLE, session=str(session), plan=plan, cell=cell,
        budget_consistency=budget, fixed_warmup_check=warmup, remaining_budget_shadows=shadows,
        analysis_code_sha256={Path(base.__file__).name: sha(Path(base.__file__)), Path(__file__).name: sha(Path(__file__))},
        semantics=SEMANTICS)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', required=True, type=Path)
    parser.add_argument('--policy-source', type=Path, default=PACKAGE / 'pkg/staged_store_rotation.py')
    parser.add_argument('--input-config', type=Path, default=PACKAGE / 'pkg/inputs/pro_high/config.json')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = analyze(args.session.resolve(), args.policy_source.resolve(), args.input_config.resolve())
    if args.output:
        with args.output.open('x') as handle:
            json.dump(result, handle, indent=2, ensure_ascii=False)
            handle.write('\n')
        print(json.dumps(dict(status=result['status'], output=str(args.output)), ensure_ascii=False))
    else:
        print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
