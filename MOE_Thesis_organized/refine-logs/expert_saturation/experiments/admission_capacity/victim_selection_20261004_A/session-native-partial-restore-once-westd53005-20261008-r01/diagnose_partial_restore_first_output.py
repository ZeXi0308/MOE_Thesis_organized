#!/usr/bin/env python3
"""Observed-baseline action equivalence only; no alternate-policy service estimate.

Actions use decision-time state and the latest preceding residency admission.
Past token timestamps validate those counts; future output never selects an action.
Missing epoch/clock/progress evidence is UNKNOWN, not zero new outputs.
"""
import argparse
import ast
import bisect
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def choices(source):
    helper = next(n for n in ast.parse(source).body
                  if isinstance(n, ast.FunctionDef) and n.name == '_partial_restore_once_choice')
    namespace = {}
    exec(compile(ast.Module(body=[helper], type_ignores=[]), '<frozen-stage-helper>', 'exec'), namespace)

    class ReplaceStage(ast.NodeTransformer):
        count = 0

        def visit_Compare(self, node):
            if ast.unparse(node) == "tail['computed_tokens'] >= tail['prompt_tokens'] + tail['output_tokens'] - 1":
                self.count += 1
                return ast.copy_location(ast.parse(
                    "tail['output_tokens'] != tail['epoch_output_count']", mode='eval').body, node)
            return self.generic_visit(node)

    replace = ReplaceStage()
    simple = replace.visit(deepcopy(helper))
    if replace.count != 1:
        raise ValueError('Frozen stage predicate changed; do not silently compare another rule')
    simple.name = 'first_output_choice'
    ast.fix_missing_locations(simple)
    exec(compile(ast.Module(body=[simple], type_ignores=[]), '<first-output-only-substitution>', 'exec'), namespace)
    return namespace[helper.name], namespace[simple.name]


def diagnose(session, cell, policy_source):
    archive = session / cell / 'archive'
    store = json.loads((archive / 'selective-store.json').read_text())
    raw = json.loads((archive / 'raw.json').read_text())
    stage, first_output = choices(policy_source.read_text())
    requests = {r['internal_request_id']: r for r in raw['requests']}
    origin = raw['measurement_origin_perf_counter_s']
    admissions = {}
    for admission in store['residency_admissions']:
        admissions.setdefault(admission['request'], []).append(admission)
    for records in admissions.values():
        records.sort(key=lambda a: a['host_perf_counter_s'])

    checks = Counter()
    phases = Counter()
    unknown = []
    differences = []
    opportunities = [[], []]
    first_choices = [None, None]
    consumed = [False, False]
    once_state_known = True
    examples = []
    for decision in store['victim_decisions']:
        rows = deepcopy(decision['candidates'])
        decision_time = decision['host_perf_counter_s']
        epochs = {}
        errors = []
        phases[str(decision.get('active_protection_or_phase'))] += 1
        for row in rows:
            try:
                rid = row['request']
                request = requests[rid]
                records = [a for a in admissions.get(rid, [])
                           if a['host_perf_counter_s'] <= decision_time]
                if not records:
                    raise ValueError('NO_PRECEDING_RESIDENCY_ADMISSION')
                epoch = records[-1]
                if epoch['num_preemptions'] != row['bidkv']['num_preemptions']:
                    raise ValueError('CURRENT_RESIDENCY_EPOCH_MISMATCH')
                if epoch['output_count'] > row['output_tokens']:
                    raise ValueError('OUTPUT_COUNT_BELOW_ADMISSION')
                if bisect.bisect_right(request['token_times_s'], decision_time-origin) != row['output_tokens']:
                    raise ValueError('DECISION_OUTPUT_CLOCK_MISMATCH')
                if bisect.bisect_right(request['token_times_s'], epoch['host_perf_counter_s']-origin) != epoch['output_count']:
                    raise ValueError('ADMISSION_OUTPUT_CLOCK_MISMATCH')
                new_outputs = row['output_tokens'] - epoch['output_count']
                if row['residency_outputs'] is not None and row['residency_outputs'] != new_outputs:
                    raise ValueError('RECORDED_RESIDENCY_OUTPUT_MISMATCH')
                row['prompt_tokens'] = request['prompt_tokens']
                # These recorded candidates are members of scheduler.running.
                row['request_status'] = 'RUNNING'
                row['epoch_output_count'] = epoch['output_count']
                epochs[rid] = epoch
                checks['aligned_candidate_rows'] += 1
                checks['reconstructed_null_residency_outputs' if row['residency_outputs'] is None
                       else 'checked_recorded_residency_outputs'] += 1
            except (KeyError, TypeError, ValueError) as error:
                errors.append(dict(request=row.get('request'), reason=str(error)))
        if errors or type(decision.get('active_protection_or_phase')) is not bool:
            unknown.append(dict(step=decision.get('step'), errors=errors or ['UNKNOWN_PHASE']))
            once_state_known = False
            continue

        tail = rows[-1]
        current = decision['unprocessed_suffix_start']
        active = decision['active_protection_or_phase']
        result = [fn(rows, current, active, False) for fn in (stage, first_output)]
        checks['compared_decisions'] += 1
        checks['stateless_same_selected'] += result[0][0] == result[1][0]
        checks['stateless_same_full_result'] += result[0] == result[1]
        if result[0] != result[1]:
            differences.append(dict(step=decision['step'], stage=result[0], first_output=result[1]))
        for i, fn in enumerate((stage, first_output)):
            if result[i][0] != tail['index']:
                opportunities[i].append(dict(step=decision['step'], index=result[i][0]))
        if once_state_known:
            once_results = [fn(rows, current, active, consumed[i])
                            for i, fn in enumerate((stage, first_output))]
            checks['once_same_selected'] += once_results[0][0] == once_results[1][0]
            for i, result_once in enumerate(once_results):
                if result_once[0] != tail['index']:
                    chosen = next(row for row in rows if row['index'] == result_once[0])
                    first_choices[i] = dict(step=decision['step'], tail=raw['internal_to_source'][tail['request']],
                        selected=raw['internal_to_source'][chosen['request']], selected_index=chosen['index'],
                        tail_release=tail['immediate_releasable_blocks'],
                        selected_release=chosen['immediate_releasable_blocks'],
                        eligible_candidates=result_once[3])
                    consumed[i] = True
        computed_gap = tail['prompt_tokens'] + tail['output_tokens'] - 1 - tail['computed_tokens']
        if computed_gap > 0:
            epoch = epochs[tail['request']]
            examples.append(dict(step=decision['step'], tail=raw['internal_to_source'][tail['request']],
                suffix_count=len(rows), admission_step=epoch['step'], epoch_output_count=epoch['output_count'],
                decision_output_count=tail['output_tokens'], computed_tokens=tail['computed_tokens'],
                computed_gap=computed_gap, stage_result=result[0], first_output_result=result[1]))
    return dict(status='UNKNOWN' if unknown else 'COMPLETE_OBSERVED_ACTION_COMPARISON',
        session=str(session), cell=cell, policy_source_sha256=hashlib.sha256(policy_source.read_bytes()).hexdigest(),
        decision_count=len(store['victim_decisions']), alignment_and_match_counts=dict(checks),
        active_protection_or_phase_counts=dict(phases), unknown_decisions=unknown,
        first_stage_choice=first_choices[0], first_output_choice=first_choices[1],
        stateless_opportunities=dict(stage=opportunities[0], first_output=opportunities[1]),
        differing_results=differences, partial_tail_examples=examples,
        scope='Observed baseline action equivalence only. No alternate-policy service results. '
              'Equivalence cannot establish incremental stage-signal contribution, even if the action later benefits service.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', type=Path, default=HERE / 'session-native-shared-prefix-probe-westd53005-20261008-r01')
    parser.add_argument('--cell', default='cell-01-shared-prefix-tail')
    parser.add_argument('--policy-source', type=Path,
                        default=HERE / 'candidate_native_partial_restore_once_r01/pkg/staged_store_rotation.py')
    args = parser.parse_args()
    print(json.dumps(diagnose(args.session, args.cell, args.policy_source), indent=2))


if __name__ == '__main__':
    main()
