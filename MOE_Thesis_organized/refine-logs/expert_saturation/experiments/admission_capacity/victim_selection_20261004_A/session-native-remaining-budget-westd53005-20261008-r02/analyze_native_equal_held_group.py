#!/usr/bin/env python3
"""Thin all-request ABBA analysis for the capacity-neutral one-action probe."""
import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path

import analyze_native_group as base


KIND = 'PRO6000_NATIVE_EQUAL_HELD_ONCE'
RULES = ('tail', 'equal_held_once', 'equal_held_once', 'tail')
POLICY_SOURCE = Path(__file__).parent / 'candidate_native_equal_held_once_r01/pkg/staged_store_rotation.py'


def load_choice(source, executed_sha):
    if source is None or not source.is_file():
        return None, None
    payload = source.read_bytes()
    source_sha = hashlib.sha256(payload).hexdigest()
    if source_sha != executed_sha:
        return None, source_sha
    function = next((node for node in ast.parse(payload).body
                     if isinstance(node, ast.FunctionDef)
                     and node.name == '_equal_held_once_choice'), None)
    if function is None:
        return None, source_sha
    namespace = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), 'exec'), namespace)
    return namespace['_equal_held_once_choice'], source_sha


def verify_choices(decisions, source, executed_sha, block_size=None):
    choose, source_sha = load_choice(source, executed_sha)
    records = []
    consumed, state_known = False, True
    proposals = requests = 0
    for index, event in enumerate(decisions):
        item = dict(decision_index=index, step=event.get('step'), rule=event.get('rule'))
        rows = event.get('candidates', [])
        indices = [row.get('index') for row in rows]
        recorded = event.get('equal_held_once')
        if choose is None:
            item['status'] = 'EXECUTED_POLICY_SOURCE_UNAVAILABLE'
        elif type(block_size) is not int or block_size <= 0:
            item['status'] = 'PROFILE_BLOCK_SIZE_UNAVAILABLE_OR_INVALID'
        elif not state_known:
            item['status'] = 'PRIOR_PROPOSAL_STATE_UNVERIFIABLE'
        elif (not rows or any(type(i) is not int for i in indices)
                or indices != list(range(indices[0], indices[-1]+1))
                or type(event.get('unprocessed_suffix_start')) is not int
                or indices[0] != event['unprocessed_suffix_start']
                or rows[-1].get('request') != event.get('native_tail')
                or any(not isinstance(row.get('request'), str) for row in rows)
                or len({row['request'] for row in rows}) != len(rows)):
            item['status'] = 'INVALID_SUFFIX_IDENTITY'
            state_known = False
        elif event.get('rule') not in ('tail', 'equal_held_once'):
            item['status'] = 'UNSUPPORTED_RULE'
            state_known = False
        elif (type(event.get('fallback_unknown')) is not bool
                or type(event.get('active_protection_or_phase')) is not bool
                or any(type(row.get('qualified')) is not bool for row in rows)):
            item['status'] = 'REQUIRED_GUARDS_NOT_RECORDED'
            state_known = False
        elif event['fallback_unknown'] != any(not row['qualified'] for row in rows):
            item['status'] = 'INCONSISTENT_RECORDED_QUALIFICATION'
            state_known = False
        else:
            proposed, reason, after, matching = choose(rows, event['fallback_unknown'],
                event['active_protection_or_phase'], consumed, block_size)
            proposed_request = next(row['request'] for row in rows if row['index'] == proposed)
            active = event['rule'] == 'equal_held_once'
            requested = active and proposed_request != event['native_tail']
            selected = proposed_request if active else event['native_tail']
            expected = dict(mode='ACTIVE' if active else 'SHADOW',
                proposed_request=proposed_request, action_requested=requested,
                fallback=reason, consumed_before=consumed, consumed_after=after,
                matching_candidate_count=matching)
            mismatches = []
            if not isinstance(recorded, dict):
                mismatches.append('equal_held_once')
            else:
                for key, value in expected.items():
                    actual = recorded.get(key)
                    if key not in recorded or actual != value or (key in ('action_requested', 'consumed_before',
                            'consumed_after', 'matching_candidate_count') and type(actual) is not type(value)):
                        mismatches.append('equal_held_once.'+key)
            for key, value in dict(selected=selected, unrestricted_selected=selected,
                    changed=requested, current_guard_applied=False, current_guard_eligible=False).items():
                actual = event.get(key)
                if actual != value or (type(value) is bool and type(actual) is not bool):
                    mismatches.append(key)
            proposals += int(not consumed and after)
            requests += int(requested)
            consumed = after  # Advance the recomputed state, never the recorded state.
            item.update(status='MISMATCH' if mismatches else 'MATCH',
                expected_selected=selected, expected_probe=expected, mismatches=mismatches)
        records.append(item)
    return dict(source=str(source) if source else None, source_sha256=source_sha,
        executed_source_sha256=executed_sha, profile_block_size_tokens=block_size,
        status_counts=dict(Counter(row['status'] for row in records)), checks=records,
        recomputed_proposal_count=proposals if state_known and choose is not None
            and type(block_size) is int and block_size > 0 else None,
        recomputed_action_request_count=requests if state_known and choose is not None
            and type(block_size) is int and block_size > 0 else None,
        recorded_action_request_count=sum(isinstance(e.get('equal_held_once'), dict)
            and e['equal_held_once'].get('action_requested') is True for e in decisions),
        semantics='Sequential recomputation from a fresh measurement-only state using the SHA-matched helper and actual profile block_size_tokens. A tail shadow consumes one proposal but requests no replacement. The active arm can request at most one replacement, including if execution later fails. Actual execution and released capacity remain the separate exact preempt/free joins; shadow proposals are never counted as executed actions.')


def intervention_prefix(reference, candidate, old_raw, new_raw):
    """Compare observed prefixes, without inferring equality of hidden state."""
    semantics = ('Each prefix ends immediately before its own first once-only proposal. '
        'Output comparison includes engine call, source request, cumulative count and token IDs; '
        'wall clocks are reported separately, not required equal. Candidate snapshots include the '
        'proposal decision. Matching observed fields cannot establish identical hidden state or '
        'strict same-state causality; actual execution is checked separately by preempt/free joins.')
    decision_lists = [c.get('native_victim_observation', {}).get('raw_decisions', [])
                      for c in (reference, candidate)]
    proposals = [[d for d in ds if isinstance(d.get('equal_held_once'), dict)
                  and d['equal_held_once'].get('consumed_before') is False
                  and d['equal_held_once'].get('consumed_after') is True]
                 for ds in decision_lists]
    if any(len(rows) != 1 for rows in proposals):
        return dict(status='NO_UNIQUE_PROPOSAL', proposal_counts=[len(x) for x in proposals],
                    semantics=semantics)
    cuts = [rows[0].get('step') for rows in proposals]
    if any(type(step) is not int for step in cuts):
        return dict(status='UNKNOWN_PROPOSAL_STEP', semantics=semantics)
    raw_pairs = (old_raw, new_raw)
    unknown = [0, 0]

    def source(i, internal):
        value = raw_pairs[i].get('internal_to_source', {}).get(internal)
        if value is None:
            unknown[i] += 1
        return value

    fields = ('held_blocks', 'computed_tokens', 'output_tokens', 'max_tokens',
              'qualified', 'residency_outputs', 'density', 'host_ready_prefix_blocks',
              'host_missing_suffix_blocks', 'host_state_error', 'pending_native_store_dependencies', 'bidkv')
    snapshots, actions, outputs, clocks = [], [], [], []
    absent_output_fields, absent_state_fields, duplicate_state_keys = [], [], []
    for i, (raw, ds, cut, proposal) in enumerate(zip(raw_pairs, decision_lists, cuts,
                                                  (p[0] for p in proposals))):
        states, acts, output_rows = {}, [], []
        missing_output = missing_state = duplicates = 0
        for event in raw.get('output_events', []):
            step = event.get('engine_call_index')
            if type(step) is not int:
                missing_output += 1
                continue
            if step < cut:
                keys = ('engine_call_index', 'request_id', 'cumulative_tokens', 'new_token_ids')
                missing_output += sum(key not in event for key in keys)
                output_rows.append([event.get(key) for key in keys])
        for d in ds:
            step = d.get('step')
            if type(step) is not int:
                missing_state += 1
                continue
            if step > cut:
                continue
            failed = source(i, d.get('failed_request'))
            tail = source(i, d.get('native_tail'))
            if step < cut:
                acts.append([step, failed, tail, source(i, d.get('selected'))])
            for row in d.get('candidates', []):
                key = (step, failed, row.get('index'), source(i, row.get('request')))
                missing_state += sum(field not in row for field in fields)
                duplicates += int(key in states)
                states[key] = {field: row.get(field) for field in fields}
        origin, at = raw.get('measurement_origin_perf_counter_s'), proposal.get('host_perf_counter_s')
        clocks.append(at-origin if base.finite(at) and base.finite(origin) else None)
        snapshots.append(states); actions.append(acts); outputs.append(output_rows)
        absent_output_fields.append(missing_output); absent_state_fields.append(missing_state)
        duplicate_state_keys.append(duplicates)
    common = snapshots[0].keys() & snapshots[1].keys()
    changed_fields, examples, changed_rows = Counter(), [], 0
    for key in sorted(common, key=repr):
        differences = {field: dict(reference=snapshots[0][key][field],
                                  candidate=snapshots[1][key][field])
                       for field in fields if snapshots[0][key][field] != snapshots[1][key][field]}
        if differences:
            changed_rows += 1
            changed_fields.update(differences.keys())
            if len(examples) < 12:
                examples.append(dict(step=key[0], failed_request=key[1], index=key[2],
                                     request=key[3], differences=differences))
    envelopes = []
    for i, proposal in enumerate(p[0] for p in proposals):
        envelope = {key: proposal.get(key) for key in ('step', 'free_blocks',
            'unprocessed_suffix_start', 'fallback_unknown', 'active_protection_or_phase')}
        envelope.update(failed_request=source(i, proposal.get('failed_request')),
                        native_tail=source(i, proposal.get('native_tail')),
                        proposed_request=source(i, proposal['equal_held_once'].get('proposed_request')))
        envelopes.append(envelope)
    incomplete = any(unknown + absent_output_fields + absent_state_fields + duplicate_state_keys)
    return dict(status='INCOMPLETE_OBSERVATION' if incomplete else 'OBSERVED_PREFIX_COMPARISON',
        proposal_steps=cuts, same_proposal_engine_call=cuts[0] == cuts[1],
        proposal_elapsed_from_measurement_start_s=clocks,
        proposal_envelopes=envelopes, proposal_envelopes_equal=envelopes[0] == envelopes[1],
        prior_actions=dict(counts=[len(x) for x in actions], exact_equal=actions[0] == actions[1],
                           sha256=[base.digest(x) for x in actions]),
        prior_outputs=dict(counts=[len(x) for x in outputs], exact_equal=outputs[0] == outputs[1],
                           sha256=[base.digest(x) for x in outputs],
                           comparison_fields=['engine_call_index', 'request_id', 'cumulative_tokens', 'new_token_ids']),
        observed_candidate_states=dict(counts=[len(x) for x in snapshots], matched_keys=len(common),
            reference_only_keys=len(snapshots[0].keys()-snapshots[1].keys()),
            candidate_only_keys=len(snapshots[1].keys()-snapshots[0].keys()),
            different_rows=changed_rows, different_fields=dict(changed_fields), first_differences=examples,
            compared_fields=list(fields)), unknown_identity_counts=unknown,
        missing_output_field_counts=absent_output_fields, missing_state_field_counts=absent_state_fields,
        duplicate_state_key_counts=duplicate_state_keys, semantics=semantics)


def analyze(session, timeout_s=600, policy_source=POLICY_SOURCE):
    previous = base.KIND, base.RULES, base.verify_choices, base.compare_cells
    original_compare = base.compare_cells

    def compare(reference, candidate, old_raw, new_raw):
        result = original_compare(reference, candidate, old_raw, new_raw)
        result['pre_intervention_prefix'] = intervention_prefix(reference, candidate, old_raw, new_raw)
        return result

    try:
        base.KIND, base.RULES, base.verify_choices = KIND, RULES, verify_choices
        base.compare_cells = compare
        result = base.analyze(session, timeout_s, policy_source)
    finally:
        base.KIND, base.RULES, base.verify_choices, base.compare_cells = previous
    result['analysis_code_sha256'] = {
        Path(base.__file__).name: hashlib.sha256(Path(base.__file__).read_bytes()).hexdigest(),
        Path(__file__).name: hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    result['semantics'] = dict(result['semantics'], intervention=
        'Capacity-neutral one-request probe: equal held blocks and equal complete computed pages, nearest suffix index, then permanently tail. Both arms record a once-only proposal. Identical arrivals do not establish identical pre-action runtime state or cache contents.')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', type=Path, required=True)
    parser.add_argument('--timeout-s', type=float, default=600)
    parser.add_argument('--policy-source', type=Path, default=POLICY_SOURCE)
    parser.add_argument('--output', type=Path, help='Omit for stdout; existing files are never overwritten')
    args = parser.parse_args()
    result = analyze(args.session, args.timeout_s, args.policy_source)
    if args.output:
        with args.output.open('x') as stream:
            json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write('\n')
    else:
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
