"""One report for six actual R16/H10/S16 native CPU-KV executions."""
import argparse
from itertools import combinations
import json
from pathlib import Path
import statistics
from analyze_kv_swap import ROOT, analyze_kv, comparison, load, ratio, sha

NAMES = ('00_recompute16', '01_headroom', '02_swap16', '03_swap16', '04_headroom', '05_recompute16')
GROUPS = {'R16': (NAMES[0], NAMES[5]), 'H10': (NAMES[1], NAMES[4]), 'S16': (NAMES[2], NAMES[3])}
SPEC = {n: (10 if g == 'H10' else 16, 1 if g == 'S16' else 0) for g, ns in GROUPS.items() for n in ns}
mean = lambda values: statistics.mean(values) if all(v is not None for v in values) else None

def main():
    global NAMES, GROUPS, SPEC
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--design', choices=('recovery', 'reserve', 'ngram', 'ngram_short'), default='recovery')
    p.add_argument('--results', type=Path, default=ROOT / 'results_kv_swap_r01')
    p.add_argument('--inputs', type=Path, default=ROOT / 'inputs/olmoe_gsm8k_natural16')
    p.add_argument('--metadata', type=Path, default=ROOT.parent / 'C_research_artifacts/20261001/20261001_c_instruct_model_metadata_v1')
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    if args.design == 'reserve':
        NAMES = ('00_swap16', '01_headroom', '02_reserve16', '03_reserve16', '04_headroom', '05_swap16')
        GROUPS = {'S16': (NAMES[0], NAMES[5]), 'H10': (NAMES[1], NAMES[4]), 'P16': (NAMES[2], NAMES[3])}
        SPEC = {n: (10 if g == 'H10' else 16, 0 if g == 'H10' else 1) for g, ns in GROUPS.items() for n in ns}
    elif args.design == 'ngram':
        NAMES = ('00_ar16', '01_ngram4', '02_ngram4', '03_ar16')
        GROUPS = {'AR16': (NAMES[0], NAMES[3]), 'N4': (NAMES[1], NAMES[2])}
        SPEC = {n: (16, 1) for n in NAMES}
    elif args.design == 'ngram_short':
        NAMES = ('00_ar16', '01_ngram1', '02_ngram2', '03_ngram2', '04_ngram1', '05_ar16')
        GROUPS = {'AR16': (NAMES[0], NAMES[5]), 'N1': (NAMES[1], NAMES[4]), 'N2': (NAMES[2], NAMES[3])}
        SPEC = {n: (16, 1) for n in NAMES}
    for n in NAMES:
        for f in ('raw.json', 'status.json'):
            path = args.results / n / f
            if not path.is_file() or load(path).get('status') != 'COMPLETE':
                p.error(f'No metrics written: {path} missing or not COMPLETE')
    from tokenizers import Tokenizer
    workload = load(args.inputs / 'workload.json')
    expected = next(f['sha256'] for f in load(args.metadata / 'metadata-receipt.json')['files'] if f['filename'] == 'tokenizer.json')
    if sha(args.metadata / 'tokenizer.json') != expected:
        raise ValueError('tokenizer differs from pinned receipt')
    tokenizer = Tokenizer.from_file(str(args.metadata / 'tokenizer.json'))
    cells = {n: analyze_kv(args.results / n, workload['source_requests'], workload['arrival_traces_s']['steady'], tokenizer, 2048) for n in NAMES}
    reference = cells[NAMES[0]]['actual_resources']['before_measurement'] or {}
    for n, c in cells.items():
        cap, cpu = SPEC[n]; config, raw = load(args.results / n / 'config.json'), load(args.results / n / 'raw.json')
        snapshots = [c['actual_resources'][k] or {} for k in ('after_init', 'before_measurement', 'after_measurement')]
        c['resource_checks'] = dict(cap=config.get('admission_cap') == raw.get('target_cap') == cap,
            observed_active_within_cap=all(s['actual_active'] <= cap for s in raw['scheduler_steps']),
            cpu_requested=config.get('cpu_kv_gib') == cpu, actual_cpu=all(s.get('cpu_kv_unique_bytes') == cpu * 2**30 for s in snapshots),
            connector=all(s.get('connector') == ('OffloadingConnector' if cpu else 'NoneType') for s in snapshots),
            actual_gpu_equal=all(s.get('gpu_kv_unique_bytes') == reference.get('gpu_kv_unique_bytes') and s.get('gpu_blocks') == reference.get('gpu_blocks') for s in snapshots) and bool(reference),
            budget2048=all(c['budget_checks'].values()), drained_observed=c['kv_service_cost'].get('drained_s') is not None,
            transfers_observed=c['completed_transfer_bytes'] is not None, captured_overlap_matches=c['captured_overlap_matches'] is True)
        if args.design == 'reserve':
            c['resource_checks']['reservation_flag'] = config.get('restore_reserve_block') == ('reserve16' in n)
        if args.design.startswith('ngram'):
            c['resource_checks']['ngram_flag'] = config.get('ngram_speculative_tokens') == (int(n[-1]) if 'ngram' in n else 0)
    def values(n):
        c = cells[n]
        out = {k: c[k] for k in ('episode_wall_s', 'output_tokens', 'output_tokens_per_s', 'accuracy', 'cap_truncated_requests', 'served_output_tokens_per_s', 'drained_output_tokens_per_s', 'weight_copy_bytes', 'group_count')}
        out.update({k: c['kv_service_cost'].get(k) for k in ('served_s', 'drained_s', 'capture_end_s')})
        out.update({f'{kind}_{stat}': c[kind][stat] for kind in ('flow', 'ttft', 'itl', 'request_maxgap') for stat in ('mean_s', 'p95_s', 'max_s')})
        out.update({f'inter_chunk_gap_{stat}': c['inter_chunk_gap'][stat] for stat in ('mean_s', 'p95_s', 'max_s')})
        out.update({f'kv_{k}_bytes': (c['completed_transfer_bytes'] or {}).get(k) for k in ('load', 'store')})
        out.update(engine_wall_s=c['stages']['total_engine_wall_s'])
        out.update(preemptions=c['preemption']['total_preemption_events'], repeated_positions=c['scheduled_position_overlap']['repeated_scheduled_positions'], prompt_excess=c['prompt_reprocessing']['observed_prompt_reprocessing_excess'])
        return out
    flat = {n: values(n) for n in NAMES}
    means = {g: {k: mean([flat[n][k] for n in ns]) for k in flat[ns[0]]} for g, ns in GROUPS.items()}
    def paired(a, b):
        result = comparison(a, b, cells)
        result['served_drained_rate_ratios'] = {k: ratio(flat[a][k], flat[b][k]) for k in ('served_s', 'drained_s', 'served_output_tokens_per_s', 'drained_output_tokens_per_s')}
        return result
    r01 = args.results.resolve().name == 'results_kv_swap_r01'
    report = dict(order=NAMES, design=args.design, fixed_token_budget=2048, group_status=load(args.results / 'group_status.json')['status'],
        system_ab_scope='CPU_PREFIX_REUSE_PLUS_RECOVERY' if r01 else 'ACTUAL_CPU_OFFLOAD_SYSTEM_AB',
        pure_recovery_causal_test='INVALID_CONFOUNDED_CPU_CROSS_REQUEST_PREFIX_REUSE' if r01 else 'NOT_ESTABLISHED_BY_THIS_WRAPPER',
        all_six_complete=all(c['all_16_complete'] for c in cells.values()), all_resource_checks_pass=all(all(c['resource_checks'].values()) for c in cells.values()),
        workload_sha256=sha(args.inputs / 'workload.json'), tokenizer_sha256=expected, cells=cells, two_repeat_means=means,
        paired_comparisons=[paired(GROUPS[b][i], GROUPS[a][i]) for a, b in combinations(GROUPS, 2) for i in (0, 1)],
        within_group_repeats={g: paired(ns[1], ns[0]) for g, ns in GROUPS.items()},
        mean_comparisons=[dict(candidate=b, baseline=a, ratios={k: ratio(means[b][k], means[a][k]) for k in means[a] if k not in ('accuracy', 'cap_truncated_requests')}, accuracy_difference=means[b]['accuracy'] - means[a]['accuracy']) for a, b in combinations(GROUPS, 2)],
        notes=['All six complete captures and both repeats retained; candidate repeat1/2 pairs with baseline repeat1/2.', 'Natural quality/output differences remain visible; wall ratios are actual episode comparisons, not equal-work speedups.', 'Served and drained costs are separate; all captured engine calls, including connector-only/no-MoE calls, contribute to engine wall.', 'S16 adds 1 GiB CPU KV; actual GPU allocation and blocks are checked against R16. Transfer bytes are completed native load/store observations.', 'Scheduled-position overlap and prompt excess are actual work observations, not predicted savings or a complete recomputation-time estimate.', 'Each cells.*.transfer_observations.lookup retains original potential prefix-match offers; these are not committed restored-token counts. r01 remains a system A/B with CPU cross-request prefix reuse plus recovery, while its pure-recovery causal interpretation is confounded.'])
    if args.design == 'reserve':
        report['notes'][3] = 'S16 and P16 each use 1 GiB CPU KV; H10 uses none. Actual GPU KV allocation and block count are checked equal across all arms. P16 reserves within that same block pool.'
    if args.design.startswith('ngram'):
        report['all_cells_complete'] = report['all_six_complete']
        if args.design == 'ngram':
            report.pop('all_six_complete')
        report['system_ab_scope'] = 'ACTUAL_FIXED_NGRAM_WITH_NATIVE_CPU_OFFLOAD'
        report['pure_recovery_causal_test'] = 'NOT_APPLICABLE_TO_SPECULATIVE_INTERVENTION'
        report['notes'][0] = f'All {len(NAMES)} complete captures and both repeats retained; candidate repeat1/2 pairs with baseline repeat1/2.'
        report['notes'][3] = 'All arms use 1 GiB CPU KV. Actual GPU KV allocation and block count are checked equal; ngram arms use fixed min/max 2/5 and their recorded fixed draft length, with no suffix cancellation.'
        report['notes'].append('ITL is null where chunk-internal timestamps are unresolved. request_maxgap is a host emission gap; scheduled decode positions include verification work, and position overlap is not exclusively preemption recomputation. Warmup inputs/budget/length are shared but each engine warms its own algorithm; row identities are unqualified in both arms.')
    destination = args.output or args.results / 'metrics.json'
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(destination)

if __name__ == '__main__':
    main()
