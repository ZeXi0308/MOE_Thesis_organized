#!/usr/bin/env python3
"""Analyze the eight-cell cohort2 native/headroom/most-output comparison."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import analyze_rotation_holdout as shared
import analyze_rotation_first_swap as first_swap

outcome, recovery = shared.outcome, first_swap.recovery
ROLES = ('native', 'native_aa', 'headroom', 'most_output')
ORDER = ('native', 'headroom', 'most_output', 'native_aa')
PAIRS = (('native', 'headroom'), ('native', 'most_output'), ('headroom', 'most_output'))
SCOPE = ('One document cohort, two reverse-order blocks and four roles; eight engines are not '
         'eight independent workloads. Primary native is the sole native baseline; native A/A '
         'is retained observed drift, never a replacement baseline, correction or noise bound. '
         'No significance, quality equivalence, noninferiority or method GO is inferred. '
         'Reference 5 s TTFT / 0.2 s mean-TPOT SLO does not constrain maximum ITL.')


def validate_novelty(manifest, frozen, rows, workload_sha256):
    require = outcome.require
    reference = manifest['novelty_inputs']
    require(reference['receipt_path'] == 'fresh_inputs_receipt.json' and reference['excluded_prior_requests'] == 96, 'novelty receipt reference differs')
    path = (frozen/reference['receipt_path']).resolve()
    require(path.is_relative_to(frozen.resolve()) and shared.digest(path) == reference['receipt_sha256'], 'novelty receipt hash/path mismatch')
    receipt = outcome.read(path)
    require(receipt['schema_version'] == 1 and receipt['status'] == 'INPUTS_PREPARED_GPU_UNRUN' and receipt['cohort_id'] == 'cohort2'
            and receipt['workload_sha256'] == workload_sha256, 'novelty receipt identity/status differs')
    priors, excluded = receipt['prior_inputs'], receipt['excluded_prior_requests']
    require(len(priors) == len({p['workload_sha256'] for p in priors}) == 3 and len(excluded) == 96, 'expected three prior inputs and 96 exclusions')
    checks = ('prior96_documents_and_tokens_reproduced', 'all128_document_hashes_unique', 'all128_prompt_hashes_unique', 'source_intervals_disjoint', 'closed_article_boundaries')
    require(all(receipt['checks'][k] is True for k in checks), 'input preparation did not pass every novelty check')
    combined = excluded+rows
    for key in ('document_id', 'document_sha256', 'prompt_token_ids_sha256'):
        require(len({r[key] for r in combined}) == 128, 'new/prior identity or hash overlaps: '+key)
    require(not {r['request_id'] for r in rows}.intersection(r['document_id'] for r in excluded), 'request identity overlaps prior inputs')
    intervals = sorted((r['dataset_row_index'], r['dataset_row_end_exclusive']) for r in combined)
    require(all(type(a) is int and type(b) is int and 0 <= a < b for a, b in intervals), 'invalid source row interval')
    require(all(a[1] <= b[0] for a, b in zip(intervals, intervals[1:])), 'new/prior source row intervals overlap')
    return dict(status='PASS_RECEIPT_AND_128_IDENTITIES', receipt_path=str(path), receipt_sha256=reference['receipt_sha256'],
                prior_inputs=priors, excluded_prior_requests=96, selected_requests=32, checks=receipt['checks'])


def validate_manifest(manifest, frozen):
    require = outcome.require
    require(manifest['schema_version'] == 4 and manifest['experiment_id'] == '20260913_rotation_strong_baseline_r01', 'unsupported strong-baseline campaign')
    require(len(manifest['cohorts']) == 1, 'expected only cohort2')
    cohort = manifest['cohorts'][0]
    require(cohort['id'] == 'cohort2' and cohort['input_root'] == 'cohorts/cohort2', 'cohort identity/path differs')
    root = (frozen/cohort['input_root']).resolve()
    require(root.is_relative_to(frozen.resolve()), 'cohort path escapes frozen source')
    prepared = root/'inputs_preparation/prepared/long'
    config, workload = [outcome.read(prepared/n) for n in ('config.json', 'workload.json')]
    canonical = hashlib.sha256(json.dumps(workload, sort_keys=True).encode()).hexdigest()
    require(canonical == config['workload_sha256'] == cohort['workload_sha256'], 'manifest/input workload hash mismatch')
    require(config['requests'] == 32 and config['prompt_tokens'] == 3072 and config['output_tokens'] == 1024 and config['arrival_gap_s'] == .05, 'cohort dimensions/arrival gap differ')
    rows, tokens, arrivals = workload['source_requests'], workload['actual_prompt_token_ids'], workload['arrival_traces_s']['steady']
    require(len(rows) == len(tokens) == len(arrivals) == 32, 'expected 32 aligned inputs')
    require(all(math.isclose(t, i*.05, rel_tol=0, abs_tol=1e-12) for i, t in enumerate(arrivals)), 'steady arrivals differ from 50 ms spacing')
    for key in ('document_id', 'request_id', 'document_sha256', 'prompt_token_ids_sha256'):
        require(len({r[key] for r in rows}) == 32, 'duplicate cohort identity/hash: '+key)
    for row, ids in zip(rows, tokens):
        require(row['prompt_token_count'] == len(ids) == 3072, 'prompt token length differs')
        require(hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest() == row['prompt_token_ids_sha256'], 'prompt token hash mismatch')
        require(hashlib.sha256(row['prompt'].encode()).hexdigest() == row['prompt_sha256'] == row['document_sha256'], 'document text hash mismatch')
    expected = [dict(label=f'cohort2-block{b}-{r}', cohort_id='cohort2', block=b, role=r,
        completion_policy='rotate' if r == 'most_output' else 'native' if r == 'native_aa' else r,
        cap=32, victim_order='most_output' if r == 'most_output' else 'least_progress')
        for b, roles in ((0, ORDER), (1, tuple(reversed(ORDER)))) for r in roles]
    require(len(manifest['cells']) == 8, 'expected all eight cells')
    for cell, spec in zip(manifest['cells'], expected):
        require(type(cell['block']) is int and all(cell[k] == v for k, v in spec.items()), 'cell identity/config/order differs')
    novelty = validate_novelty(manifest, frozen, rows, canonical)
    return {'cohort2': cohort}, {'cohort2': dict(workload_sha256=canonical, input_root=cohort['input_root'],
        document_ids=sorted(r['document_id'] for r in rows), novelty_check=novelty,
        inputs={n: shared.digest(prepared/n) for n in ('config.json', 'workload.json')})}


def inspect_cell(run_dir, spec, native, frozen, cohort):
    directory = run_dir/'gpu_results'/spec['label']
    row = shared.inspect_cell(directory, spec, native, frozen, cohort)
    row['victim_order'] = spec['victim_order']
    if row.get('error') in ('rotation performed no forced action', 'headroom did not perform an action'):
        row.update(status='INVALID_NO_ACTION', full_episode_comparison_eligible=False)
    if row['full_episode_comparison_eligible']:
        try:
            outcome.require(row['config']['rotation_victim_order'] == spec['victim_order'], 'runtime victim order differs from manifest')
            counts = row['preemption_accounting']
            row['action_counts'] = {k: counts[k] for k in ('forced_count', 'natural_count')}
            if spec['role'] == 'most_output':
                row['transition_check'] = first_swap.transition_check(outcome.read(directory/'headroom-decisions.json'), 'most_output')
                outcome.require(row['transition_check']['successful_forced_count'] == counts['forced_count'], 'transition/native action count differs')
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
    native = outcome.module('strong_baseline_native_helpers', outputs/'20260908_native_preemption_r01/analyze_native_preemption.py')
    rows = [inspect_cell(run_dir, s, native, frozen, cohorts[s['cohort_id']]) for s in manifest['cells']]
    index = {(r['block'], r['role']): r for r in rows}
    qualified = all(r['full_episode_comparison_eligible'] for r in rows)
    common = None
    if qualified:
        signatures = [(r['engine_args'], {k: v for k, v in r['config'].items() if k not in ('completion_policy', 'rotation_victim_order')}) for r in rows]
        common = all(s == signatures[0] for s in signatures)
    eligible = qualified and common
    def compare(a, b, kind):
        pair = shared.compare_cells(a, b, eligible, native, kind, additional_config_exclusions=('rotation_victim_order',))
        if pair['status'] == 'DESCRIPTIVE_MATCHED_PAIR':
            pair['action_count_changes'] = shared.difference(a['action_counts'], b['action_counts'])
        return pair
    pairs = [compare(index[(b, a)], index[(b, c)], 'within_block_policy') for b in (0, 1) for a, c in PAIRS]
    aa = [compare(index[(b, 'native')], index[(b, 'native_aa')], 'within_block_native_aa') for b in (0, 1)]
    repeats = [compare(index[(0, r)], index[(1, r)], 'same_role_across_blocks') for r in ROLES]
    matched = bool(eligible and all(p['status'] == 'DESCRIPTIVE_MATCHED_PAIR' for p in pairs+aa+repeats))
    status = ('MEASUREMENT_ONLY' if matched else 'UNRUN' if all(r['status'] == 'UNRUN' for r in rows)
        else 'INVALID_NO_ACTION' if any(r['status'] == 'INVALID_NO_ACTION' for r in rows)
        else 'INVALID_EVIDENCE' if qualified or any(r['status'] == 'INVALID_EVIDENCE' for r in rows) else 'INCOMPLETE_CAMPAIGN')
    recoveries = []
    if matched:
        for row in rows:
            item = recovery.cell(run_dir, row['label'], native, row, policy=row['policy'], campaign=True)
            item.update({k: row[k] for k in ('cohort_id', 'block', 'role', 'victim_order')})
            recoveries.append(item)
    return dict(status=status, all_eight_cells_qualified=qualified, common_execution_config=common, comparisons_eligible=matched,
        campaign_manifest=manifest, campaign_manifest_sha256=shared.digest(manifest_path), cohort_inputs=workloads,
        cells=rows, comparisons=pairs, native_aa=aa, same_role_repeats=repeats, recovery_accounting=recoveries,
        metrics_sha256=shared.digest(frozen/'metrics.py'),
        sample_structure=dict(distinct_document_cohorts=1, blocks_per_cohort=2, engine_executions=8, policy_pairs=6, native_aa_pairs=2, same_role_repeat_pairs=4),
        pairing_rule='All eight cells must qualify with common engine/config before numeric pairing. Per block: native→headroom, native→most_output, headroom→most_output. Native A/A and same-role repeats are separate; primary native is never replaced or drift-corrected. Only completion_policy and verified rotation_victim_order differ; cap stays 32.',
        cost_scope='wall = scheduler_inclusive + engine_non_schedule + outside_engine_calls; decision time is included in scheduler time. Disjoint work classes retain prefill/recompute, new decode, held work and every completion/failure. Recovery spans can contain useful concurrent decode. Width/path changes are observed costs, not independent causal savings.', scope=SCOPE)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    outcome.require(not args.output_dir.exists(), 'output directory must be new; refusing overwrite')
    result = analyze(args.run_dir)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir/'analysis.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    (args.output_dir/'report.md').write_text(shared.readable(result).replace('# Independent-cohort rotation controls', '# Strong-baseline rotation controls', 1))
    print(json.dumps(dict(status=result['status'], cells=[dict(label=r['label'], status=r['status'], error=r.get('error')) for r in result['cells']]), indent=2))


if __name__ == '__main__':
    main()
