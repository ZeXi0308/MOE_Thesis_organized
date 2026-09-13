#!/usr/bin/env python3
"""Analyze the frozen six-cell first-successful-swap diagnostic campaign."""
import argparse
import hashlib
import json
from pathlib import Path

import analyze_rotation_victim_order as victim

outcome, shared, recovery = victim.outcome, victim.shared, victim.recovery
ROLES = ('least_progress', 'first_most_then_least', 'most_output')
COHORTS = victim.COHORTS[:1]
ORDER = ((0, ROLES), (1, tuple(reversed(ROLES))))
PAIRS = ((ROLES[0], ROLES[1]), (ROLES[2], ROLES[1]), (ROLES[0], ROLES[2]))
SCOPE = ('Same-document exploratory diagnostic on the previously observed cohort0; '
         'six fresh engines are one reused workload, two order blocks and three roles, '
         'not six independent workloads or a new holdout. Prior victim-order runs '
         'do not replace the newly executed A/B controls. Same-role block differences '
         'are observations, not a noise bound. No significance, fairness guarantee, '
         'quality equivalence, noninferiority or method GO is inferred. '
         'Reference 5 s TTFT / 0.2 s mean-TPOT SLO does not constrain maximum ITL.')


def validate_manifest(manifest, frozen):
    require = outcome.require
    require(manifest['schema_version'] == 3 and manifest['experiment_id'] == '20260913_rotation_first_swap_r01', 'unsupported first-swap campaign')
    require(manifest['cohorts'] == COHORTS, 'requires the frozen reused cohort0')
    require(manifest['only_treatment_config_key'] == 'rotation_victim_order', 'unexpected treatment config key')
    cohort = COHORTS[0]
    root = (frozen/cohort['input_root']).resolve()
    require(root.is_relative_to(frozen.resolve()), 'cohort path escapes frozen source')
    prepared = root/'inputs_preparation/prepared/long'
    config, workload = [outcome.read(prepared/n) for n in ('config.json', 'workload.json')]
    digest = hashlib.sha256(json.dumps(workload, sort_keys=True).encode()).hexdigest()
    require(digest == config['workload_sha256'] == cohort['workload_sha256'], 'manifest/input workload hash mismatch')
    require(config['requests'] == 32 and config['prompt_tokens'] == 3072 and config['output_tokens'] == 1024, 'cohort dimensions differ')
    rows = workload['source_requests']
    documents = {r['document_id'] for r in rows}
    require(len(rows) == len(documents) == len({r['request_id'] for r in rows}) == 32, 'duplicate cohort identity')
    expected = [dict(label=f'cohort0-block{b}-{r}', cohort_id='cohort0', block=b, role=r,
                     completion_policy='rotate', cap=32, victim_order=r) for b, roles in ORDER for r in roles]
    require(len(manifest['cells']) == 6, 'expected all six cells')
    for cell, spec in zip(manifest['cells'], expected):
        require(type(cell['block']) is int and all(cell[k] == v for k, v in spec.items()), 'cell identity/config/order differs')
    return {'cohort0': cohort}, {'cohort0': dict(workload_sha256=digest,
        input_root=cohort['input_root'], document_ids=sorted(documents))}


def transition_check(decisions, role):
    """Only a completed forced action consumes C's first-swap opportunity."""
    outcome.require(role in ROLES and decisions, 'missing/unsupported transition trace')
    applied, swaps = 0, []
    for step, d in enumerate(decisions):
        expected = ('most_output' if applied == 0 else 'least_progress') if role == ROLES[1] else role
        outcome.require(d['step'] == step and d['status'] == 'APPLIED', 'transition step/status differs')
        outcome.require(type(d['applied_rotations_before']) is int and d['applied_rotations_before'] == applied,
                        'applied rotation count differs from past successful forced actions')
        outcome.require(d['effective_victim_order'] == expected, 'effective victim order violates first-successful-swap rule')
        forced = d['forced_preempted']
        outcome.require(len(forced) <= 1, 'multiple forced victims in one swap')
        if forced:
            outcome.require(forced[0] in d['preempted'] and d['proposal']['action'] == 'rotate'
                            and d['proposal']['victim_id'] == forced[0] and not d.get('not_applied_reason'),
                            'forced action not successfully applied')
            swaps.append(dict(step=step, effective_victim_order=expected,
                              applied_rotations_before=applied, victim_internal_id=forced[0]))
            applied += 1
    outcome.require(applied > 0, 'rotation performed no forced action')
    return dict(status='PASS', steps_checked=len(decisions), successful_forced_count=applied,
                first_successful_forced_step=swaps[0]['step'], successful_swaps=swaps,
                scope='Successful forced labels are separately reconciled with raw native events; natural preemptions and failed proposals do not advance this counter.')


def inspect_cell(run_dir, spec, native, frozen, cohort):
    row = victim.inspect_cell(run_dir, spec, native, frozen, cohort)
    if row['full_episode_comparison_eligible']:
        try:
            decisions = outcome.read(run_dir/'gpu_results'/spec['label']/'headroom-decisions.json')
            row['transition_check'] = transition_check(decisions, spec['role'])
            outcome.require(row['transition_check']['successful_forced_count'] == row['action_counts']['forced_count'], 'transition/native action count differs')
        except (OSError, ValueError, TypeError, KeyError, IndexError) as exc:
            row.update(status='INVALID_EVIDENCE', error=str(exc), full_episode_comparison_eligible=False)
    return row


def analyze(run_dir):
    frozen = run_dir/'frozen'
    if not frozen.exists():
        frozen = run_dir.parent/'preparation/source'
    manifest_path = frozen/'campaign.json'
    manifest = outcome.read(manifest_path)
    cohorts, workloads = validate_manifest(manifest, frozen)
    outcome.module('metrics', frozen/'metrics.py')
    outputs = Path(__file__).resolve().parents[2]/'outputs/admission_capacity'
    native = outcome.module('first_swap_native_helpers', outputs/'20260908_native_preemption_r01/analyze_native_preemption.py')
    rows = [inspect_cell(run_dir, s, native, frozen, cohorts[s['cohort_id']]) for s in manifest['cells']]
    index = {(r['block'], r['role']): r for r in rows}
    eligible = all(r['full_episode_comparison_eligible'] for r in rows)
    pairs = [victim.compare(index[(b, a)], index[(b, c)], eligible, native, 'within_block_first_swap')
             for b in (0, 1) for a, c in PAIRS]
    repeats = [victim.compare(index[(0, r)], index[(1, r)], eligible, native, 'same_role_across_blocks') for r in ROLES]
    matched = eligible and all(p['status'] == 'DESCRIPTIVE_MATCHED_PAIR' for p in pairs+repeats)
    status = ('MEASUREMENT_ONLY' if matched else 'UNRUN' if all(r['status'] == 'UNRUN' for r in rows)
              else 'INVALID_NO_ACTION' if any(r['status'] == 'INVALID_NO_ACTION' for r in rows)
              else 'INVALID_EVIDENCE' if eligible or any(r['status'] == 'INVALID_EVIDENCE' for r in rows) else 'INCOMPLETE_CAMPAIGN')
    recoveries = []
    if matched:
        for row in rows:
            item = recovery.cell(run_dir, row['label'], native, row, policy='rotate', campaign=True)
            item.update({k: row[k] for k in ('cohort_id', 'block', 'role', 'victim_order')})
            recoveries.append(item)
    return dict(status=status, all_six_cells_qualified=eligible, comparisons_eligible=matched,
        campaign_manifest=manifest, campaign_manifest_sha256=shared.digest(manifest_path), cohort_inputs=workloads,
        cells=rows, comparisons=pairs, same_role_repeats=repeats, recovery_accounting=recoveries,
        metrics_sha256=shared.digest(frozen/'metrics.py'),
        sample_structure=dict(distinct_reused_document_cohorts=1, blocks_per_cohort=2, engine_executions=6, policy_pairs=6, same_role_repeat_pairs=3),
        pairing_rule='All six newly executed cells must be COMPLETE and qualified before numeric comparisons. Within each block compare A→C, B→C, A→B, then identical roles across blocks. A=least_progress; C=first_most_then_least; B=most_output. Only verified rotation_victim_order differs; subsequent action times and policy-specific states need not match.',
        cost_scope='wall = scheduler_inclusive + engine_non_schedule + outside_engine_calls; decision time is included in scheduler time. Work classes are disjoint calls. Recompute positions are not newly produced outputs; recovery spans may include useful concurrent decode. Observed width/path changes are not independent savings or counterfactual bounds.', scope=SCOPE)


def readable(result):
    return victim.readable(result).replace('# Victim-order ablation', '# First-successful-swap diagnostic', 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    outcome.require(not args.output_dir.exists(), 'output directory must be new; refusing overwrite')
    result = analyze(args.run_dir)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir/'analysis.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    (args.output_dir/'report.md').write_text(readable(result))
    print(json.dumps(dict(status=result['status'], cells=[dict(label=r['label'], status=r['status'], error=r.get('error')) for r in result['cells']]), indent=2))


if __name__ == '__main__':
    main()
