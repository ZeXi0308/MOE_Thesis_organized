#!/usr/bin/env python3
"""Analyze the fixed, same-document victim-order ablation without inference gates."""
import argparse
import hashlib
import json
from pathlib import Path

import analyze_completion_headroom as outcome
import analyze_rotation_holdout as shared
import analyze_rotation_recovery as recovery

ROLES = ('least_progress', 'most_output')
COHORTS = [dict(id=f'cohort{i}', input_root=f'cohorts/cohort{i}', workload_sha256=h) for i, h in enumerate((
    '3175f644cd86a3a82ce3300c2ec9fe5ffec6818640e2e9f0ed1ef307f68d4662',
    'f4e2663bb3d3a5e29cb20d967992c9c5dda0da3ef8147435f12b4c99a99a1ce3'))]
ORDER = ((0, 0, ROLES), (1, 0, tuple(reversed(ROLES))),
         (0, 1, tuple(reversed(ROLES))), (1, 1, ROLES))
SCOPE = ('Same-document exploratory ablation on the two previously observed cohorts; '
         'this is not a new text holdout. Eight fresh engines represent two cohorts, '
         'two order blocks and two victim orders, not eight independent workloads. '
         'Same-role block differences describe observed drift, not a noise bound. '
         'No significance, noninferiority, quality or method GO is inferred. '
         'Fixed 5 s TTFT / 0.2 s mean-TPOT reference SLO does not constrain maximum ITL; '
         'all-pass goodput equals throughput.')


def validate_manifest(manifest, frozen):
    require = outcome.require
    require(manifest['schema_version'] == 2 and manifest['experiment_id'] == '20260913_rotation_victim_order_r01', 'unsupported victim-order campaign')
    require(manifest['cohorts'] == COHORTS, 'cohort metadata differs from the frozen prior campaign')
    cohorts = {c['id']: c for c in manifest['cohorts']}
    workloads, documents = {}, {}
    for name, cohort in cohorts.items():
        root = (frozen/cohort['input_root']).resolve()
        require(root.is_relative_to(frozen.resolve()), 'cohort path escapes frozen source')
        prepared = root/'inputs_preparation/prepared/long'
        config, workload = [outcome.read(prepared/n) for n in ('config.json', 'workload.json')]
        canonical = hashlib.sha256(json.dumps(workload, sort_keys=True).encode()).hexdigest()
        require(canonical == config['workload_sha256'] == cohort['workload_sha256'], 'manifest/input workload hash mismatch')
        require(config['requests'] == 32 and config['prompt_tokens'] == 3072 and config['output_tokens'] == 1024, 'cohort workload dimensions differ')
        rows = workload['source_requests']
        documents[name] = {r['document_id'] for r in rows}
        require(len(rows) == len(documents[name]) == len({r['request_id'] for r in rows}) == 32, 'cohort documents or request IDs duplicate')
        workloads[name] = dict(workload_sha256=canonical, input_root=cohort['input_root'], document_ids=sorted(documents[name]))
    require(not documents['cohort0'].intersection(documents['cohort1']), 'cohorts share document identities')
    expected = [dict(label=f'cohort{c}-block{b}-{r}', cohort_id=f'cohort{c}', block=b, role=r,
                     completion_policy='rotate', cap=32, victim_order=r) for c, b, roles in ORDER for r in roles]
    require(len(manifest['cells']) == 8, 'expected eight cells')
    for cell, spec in zip(manifest['cells'], expected):
        require(type(cell['block']) is int and all(cell[k] == v for k, v in spec.items()), 'cell identity/config/order differs')
    return cohorts, workloads


def inspect_cell(run_dir, spec, native, frozen, cohort):
    directory = run_dir/'gpu_results'/spec['label']
    row = shared.inspect_cell(directory, spec, native, frozen, cohort)
    row['victim_order'] = spec['victim_order']
    if row.get('error') == 'rotation performed no forced action':
        row.update(status='INVALID_NO_ACTION', full_episode_comparison_eligible=False)
    if row.get('config') is not None:
        if row['config'].get('rotation_victim_order') != spec['victim_order']:
            row.update(status='INVALID_EVIDENCE', error='runtime victim order differs from manifest', full_episode_comparison_eligible=False)
    counts = row.get('preemption_accounting')
    if counts:
        row['action_counts'] = {k: counts[k] for k in ('forced_count', 'natural_count')}
    elif row['status'] == 'INVALID_NO_ACTION':
        row['action_counts'] = dict(forced_count=0, natural_count=row['preemptions'])
    row['action_exposure'] = 'ACTION_OBSERVED' if row.get('action_counts', {}).get('forced_count', 0) else 'INVALID_NO_ACTION' if row['status'] == 'INVALID_NO_ACTION' else 'UNQUALIFIED'
    return row


def compare(a, b, eligible, native, kind):
    pair = shared.compare_cells(a, b, eligible, native, kind, additional_config_exclusions=('rotation_victim_order',))
    if pair['status'] == 'DESCRIPTIVE_MATCHED_PAIR':
        pair['action_count_changes'] = shared.difference(a['action_counts'], b['action_counts'])
        pair['treatment_config_key'] = 'rotation_victim_order'
    return pair


def analyze(run_dir):
    frozen = run_dir/'frozen'
    if not frozen.exists():
        frozen = run_dir.parent/'preparation/source'
    manifest_path = frozen/'campaign.json'
    manifest = outcome.read(manifest_path)
    cohorts, workloads = validate_manifest(manifest, frozen)
    outcome.module('metrics', frozen/'metrics.py')
    outputs = Path(__file__).resolve().parents[2]/'outputs/admission_capacity'
    native = outcome.module('victim_order_native_helpers', outputs/'20260908_native_preemption_r01/analyze_native_preemption.py')
    rows = [inspect_cell(run_dir, s, native, frozen, cohorts[s['cohort_id']]) for s in manifest['cells']]
    index = {(r['cohort_id'], r['block'], r['role']): r for r in rows}
    eligible = all(r['full_episode_comparison_eligible'] for r in rows)
    pairs = [compare(index[(c, b, ROLES[0])], index[(c, b, ROLES[1])], eligible, native, 'within_block_victim_order') for c in cohorts for b in (0, 1)]
    repeats = [compare(index[(c, 0, r)], index[(c, 1, r)], eligible, native, 'same_role_across_blocks') for c in cohorts for r in ROLES]
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
    return dict(status=status, all_eight_cells_qualified=eligible, comparisons_eligible=matched,
        campaign_manifest=manifest, campaign_manifest_sha256=shared.digest(manifest_path), cohort_inputs=workloads,
        cells=rows, comparisons=pairs, same_role_repeats=repeats, recovery_accounting=recoveries,
        metrics_sha256=shared.digest(frozen/'metrics.py'),
        sample_structure=dict(distinct_reused_document_cohorts=2, blocks_per_cohort=2, engine_executions=8, policy_pairs=4, same_role_repeat_pairs=4),
        pairing_rule='All eight cells must be COMPLETE and qualified before numeric comparisons. Compare least_progress to most_output within each cohort/block and identical roles across blocks within each cohort. No cross-cohort request pairs. Only the verified rotation_victim_order treatment key is additionally excluded from config equality; recovery paths need not match.',
        cost_scope='wall = scheduler_inclusive + engine_non_schedule + outside_engine_calls; decision time is a subset of scheduler time. Work classes are disjoint calls, not disjoint GPU kernels. Recovery spans can include concurrent useful decode and host overhead. Actual policy-specific work is not a counterfactual bound.', scope=SCOPE)


def readable(result):
    lines = ['# Victim-order ablation', '', result['status'], '',
        '| Cell | Status | Forced / natural | Requests/s | Mean completion s | Max ITL s |',
        '|---|---|---:|---:|---:|---:|']
    for row in result['cells']:
        counts = row.get('action_counts')
        action = f"{counts['forced_count']} / {counts['natural_count']}" if counts else '—'
        suffix = f"{row['metrics']['throughput_rps']:.6f} | {row['mean_completion_s']:.6f} | {row['max_itl_s']:.6f}" if row['full_episode_comparison_eligible'] else '— | — | —'
        lines.append(f"| {row['label']} | {row['status']} | {action} | {suffix} |")
    for title, pairs in (('Within-block treatment comparisons', result['comparisons']), ('Same-role block observations', result['same_role_repeats'])):
        lines += ['', '## '+title, '', '| Baseline → action | Status | Throughput Δ | Mean completion Δ | Max ITL Δ s | Forced count Δ |', '|---|---|---:|---:|---:|---:|']
        for pair in pairs:
            suffix = f"{pair['throughput_relative_change']:+.3%} | {pair['mean_completion_relative_change']:+.3%} | {pair['max_itl_change_s']:+.6f} | {pair['action_count_changes']['forced_count']:+}" if pair['status'] == 'DESCRIPTIVE_MATCHED_PAIR' else '— | — | — | —'
            lines.append(f"| {pair['baseline']} → {pair['action']} | {pair['status']} | {suffix} |")
    return '\n'.join(lines+['', result['pairing_rule'], '', result['scope'], '', result['cost_scope'], '', 'Full per-request changes, metric vectors, work costs and actual recovery accounting are in analysis.json.', ''])


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
