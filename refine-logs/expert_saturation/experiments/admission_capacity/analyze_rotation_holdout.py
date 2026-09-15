#!/usr/bin/env python3
"""Analyze frozen two-cohort rotation controls using existing raw accounting."""
import argparse
from collections import Counter
from itertools import combinations
import hashlib
import json
import math
from pathlib import Path

import analyze_completion_headroom as outcome
import analyze_headroom_cost as host_cost
import analyze_headroom_work_cost as work_cost

ROLES = ('native', 'native_aa', 'safe29', 'headroom', 'rotate')
PRIMARY = ('native', 'safe29', 'headroom', 'rotate')
SCOPE = ('Two distinct document cohorts, each run in two order blocks; twenty engine '
         'executions are not twenty independent workloads. Request/token observations '
         'are not independent experimental repeats. Four native A/A pairs describe '
         'observed drift only, not a statistical noise bound. No significance, '
         'noninferiority or method GO is inferred. Fixed 5 s TTFT / 0.2 s mean-TPOT '
         'reference SLO does not constrain maximum ITL; all-pass goodput equals throughput.')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_manifest(manifest, frozen):
    require = outcome.require
    require(manifest['schema_version'] == 1, 'unsupported campaign schema')
    require(isinstance(manifest['experiment_id'], str) and manifest['experiment_id'], 'experiment ID missing')
    require(manifest['randomization_seed'] == 2026091301, 'randomization seed differs')
    cohorts = {c['id']: c for c in manifest['cohorts']}
    require(len(manifest['cohorts']) == len(cohorts) == 2 and set(cohorts) == {'cohort0', 'cohort1'}, 'expected exactly two named cohorts')
    documents, workloads = {}, {}
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
    require(workloads['cohort0']['workload_sha256'] != workloads['cohort1']['workload_sha256'], 'cohorts share the same workload')
    cells = manifest['cells']
    require(len(cells) == len({c['label'] for c in cells}) == 20, 'expected twenty uniquely labelled cells')
    observed = Counter()
    for cell in cells:
        cohort, block, role = (cell[k] for k in ('cohort_id', 'block', 'role'))
        require(cohort in cohorts and type(block) is int and block in (0, 1) and role in ROLES, 'invalid cohort/block/role')
        require(cell['label'] == f'{cohort}-block{block}-{role}', 'cell label differs from identity')
        require(cell['completion_policy'] == ('native' if role == 'native_aa' else role), 'role/policy mismatch')
        require(cell['cap'] == (29 if role == 'safe29' else 32), 'role/cap mismatch')
        observed[(cohort, block, role)] += 1
    expected = Counter((c, b, r) for c in cohorts for b in (0, 1) for r in ROLES)
    require(observed == expected, 'each cohort/block must contain all five roles exactly once')
    return cohorts, workloads


def inspect_cell(directory, cell, native, frozen, cohort):
    # Existing helpers parse the second hyphen component as the actual policy.
    helper_label = 'cell-'+cell['completion_policy']
    row = outcome.inspect(directory, helper_label, native, frozen/cohort['input_root'], four_arm=True)
    row.update(label=cell['label'], cohort_id=cell['cohort_id'], block=cell['block'], role=cell['role'])
    if row['full_episode_comparison_eligible']:
        try:
            outcome.require(row['identity_check']['workload_sha256'] == cohort['workload_sha256'], 'cell differs from manifest cohort')
            row['host_cost'] = host_cost.inspect(directory, helper_label)
            row['work_cost'] = work_cost.inspect(directory, helper_label)
            row['host_cost']['label'] = row['work_cost']['label'] = cell['label']
            outcome.require(math.isclose(row['host_cost']['engine_non_schedule_s'], row['work_cost']['totals']['engine_non_schedule_s'], rel_tol=0, abs_tol=1e-12), 'host/work cost partitions differ')
        except (OSError, KeyError, ValueError, TypeError, IndexError) as exc:
            row.update(status='INVALID_EVIDENCE', error=str(exc), full_episode_comparison_eligible=False)
    return row


def metric_vector(row):
    metrics = row['metrics']
    result = {k: metrics[k] for k in ('throughput_rps', 'goodput_rps', 'observation_duration_s', 'n_slo_pass', 'slo_attainment')}
    result.update(mean_completion_s=row['mean_completion_s'], max_itl_s=row['max_itl_s'])
    for kind, values in metrics['latency_s'].items():
        result.update({f'{kind}_{p}_s': values[p] for p in ('p50', 'p95', 'p99')})
    for kind in ('completion_latency_s', 'request_max_itl_s'):
        result.update({f'{kind}_{p}': row['effects'][kind][p] for p in ('p50', 'p95', 'p99')})
    for kind in ('ttft', 'tpot', 'queue'):
        values = [r[kind+'_s'] for r in row['requests']]
        result[kind+'_mean_s'] = sum(values)/len(values)
    result['completion_max_s'] = max(r['completion_latency_s'] for r in row['requests'])
    return result


def difference(a, b):
    return {k: b[k]-a[k] for k in a}


def compare_cells(a, b, eligible, native, kind, additional_config_exclusions=()):
    pair = outcome.compare(a, b, eligible, four_arm=True, distribution=native.distribution,
                           additional_config_exclusions=additional_config_exclusions)
    pair.update(kind=kind, cohort_id=a['cohort_id'], baseline_block=a['block'], action_block=b['block'],
                baseline_role=a['role'], action_role=b['role'])
    if pair['status'] != 'DESCRIPTIVE_MATCHED_PAIR':
        return pair
    outcome.require(a['cohort_id'] == b['cohort_id'], 'cross-cohort request pairing forbidden')
    x, y = metric_vector(a), metric_vector(b)
    pair['metric_changes'] = {k: dict(baseline=x[k], action=y[k], absolute_change=y[k]-x[k],
                                    relative_change=y[k]/x[k]-1 if x[k] != 0 else None) for k in x}
    host = difference({k: v for k, v in a['host_cost'].items() if k != 'label'},
                      {k: v for k, v in b['host_cost'].items() if k != 'label'})
    outcome.require(math.isclose(host['wall_s'], sum(host[k] for k in ('scheduler_inclusive_s', 'engine_non_schedule_s', 'outside_engine_calls_s')), rel_tol=0, abs_tol=1e-12), 'paired host cost does not close')
    aw, bw = a['work_cost'], b['work_cost']
    classes = {k: difference(aw['classes'][k], bw['classes'][k]) for k in work_cost.CLASSES}
    totals = difference(aw['totals'], bw['totals'])
    outcome.require(all(math.isclose(totals[k], sum(c[k] for c in classes.values()), rel_tol=0, abs_tol=1e-12) for k in work_cost.FIELDS), 'paired work cost does not close')
    widths = sorted(set(aw['pure_decode_by_scheduled_width']) | set(bw['pure_decode_by_scheduled_width']), key=int)
    pair['cost_changes'] = dict(host=host, totals=totals, classes=classes,
        pure_decode_by_scheduled_width={k: difference(aw['pure_decode_by_scheduled_width'].get(k, work_cost.blank()), bw['pure_decode_by_scheduled_width'].get(k, work_cost.blank())) for k in widths})
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
    native = outcome.module('holdout_native_helpers', outputs/'20260908_native_preemption_r01/analyze_native_preemption.py')
    rows = [inspect_cell(run_dir/'gpu_results'/c['label'], c, native, frozen, cohorts[c['cohort_id']]) for c in manifest['cells']]
    index = {(r['cohort_id'], r['block'], r['role']): r for r in rows}
    eligible = all(r['full_episode_comparison_eligible'] for r in rows)
    pairs, aa, repeats = [], [], []
    for cohort in cohorts:
        for block in (0, 1):
            for a, b in combinations(PRIMARY, 2):
                pairs.append(compare_cells(index[(cohort, block, a)], index[(cohort, block, b)], eligible, native, 'within_block_policy'))
            aa.append(compare_cells(index[(cohort, block, 'native')], index[(cohort, block, 'native_aa')], eligible, native, 'within_block_native_aa'))
        for role in ROLES:
            repeats.append(compare_cells(index[(cohort, 0, role)], index[(cohort, 1, role)], eligible, native, 'same_role_across_blocks'))
    matched = eligible and all(p['status'] == 'DESCRIPTIVE_MATCHED_PAIR' for p in pairs+aa+repeats)
    status = 'MEASUREMENT_ONLY' if matched else 'UNRUN' if all(r['status'] == 'UNRUN' for r in rows) else 'INVALID_EVIDENCE' if eligible or any(r['status'] == 'INVALID_EVIDENCE' for r in rows) else 'INCOMPLETE_CAMPAIGN'
    return dict(status=status, all_twenty_cells_qualified=eligible, comparisons_eligible=matched,
        campaign_manifest=manifest, campaign_manifest_sha256=digest(manifest_path), cohort_inputs=workloads,
        cells=rows, comparisons=pairs, native_aa=aa, same_role_repeats=repeats,
        metrics_sha256=digest(frozen/'metrics.py'),
        sample_structure=dict(distinct_document_cohorts=2, blocks_per_cohort=2, engine_executions=20, native_aa_pairs=4),
        pairing_rule='All twenty cells must be COMPLETE and qualified before numeric comparisons. Six primary pairs per cohort/block; native_aa is compared to primary native, never selected as a replacement baseline. No cross-cohort request pairing.',
        cost_scope='wall = scheduler_inclusive + engine_non_schedule + outside_engine_calls; decision time is a subset of scheduler time. Work classes are disjoint calls; recompute/prefill calls may also produce new decode. Engine remainder includes host overhead, sampling, synchronization and model execution, not pure GPU time. Width differences are observed policy-specific paths, not counterfactual savings.',
        scope=SCOPE)


def readable(result):
    lines = ['# Independent-cohort rotation controls', '', result['status'], '',
        '| Cell | Status | Complete | Wall s | Requests/s | Mean completion s | Max ITL s |',
        '|---|---|---:|---:|---:|---:|---:|']
    for r in result['cells']:
        if not r['full_episode_comparison_eligible']:
            lines.append(f"| {r['label']} | {r['status']} | — | — | — | — | — |")
        else:
            m = r['metrics']
            lines.append(f"| {r['label']} | COMPLETE | {m['n_completed']}/32 | {m['observation_duration_s']:.6f} | {m['throughput_rps']:.6f} | {r['mean_completion_s']:.6f} | {r['max_itl_s']:.6f} |")
    for name, pairs in (('Primary paired observations', result['comparisons']), ('Native A/A observed drift', result['native_aa'])):
        lines += ['', '## '+name, '', '| Cohort/block | Baseline → action | Status | Throughput Δ | Mean completion Δ | Max ITL Δ s |', '|---|---|---|---:|---:|---:|']
        for p in pairs:
            prefix = f"| {p['cohort_id']}/{p['baseline_block']} | {p['baseline_role']} → {p['action_role']} | {p['status']} |"
            lines.append(prefix+(f" {p['throughput_relative_change']:+.3%} | {p['mean_completion_relative_change']:+.3%} | {p['max_itl_change_s']:+.6f} |" if p['status'] == 'DESCRIPTIVE_MATCHED_PAIR' else ' — | — | — |'))
    return '\n'.join(lines+['', result['pairing_rule'], '', result['scope'], '', result['cost_scope'], '', 'Full paired metric vectors, per-request changes, costs and same-role block drift are retained in analysis.json.', ''])


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
