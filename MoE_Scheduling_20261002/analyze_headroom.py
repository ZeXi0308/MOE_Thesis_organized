"""Actual cap16/10/8 natural-EOS executions at fixed token budget 2048."""
import argparse
from itertools import combinations
import json
from pathlib import Path
import statistics
from analyze_natural import ROOT, analyze, comparison, load, ratio, sha
from natural_pause import diagnose

NAMES = ('00_cap16', '01_cap10', '02_cap8', '03_cap8', '04_cap10', '05_cap16')
REPEATS = {16: (NAMES[0], NAMES[5]), 10: (NAMES[1], NAMES[4]), 8: (NAMES[2], NAMES[3])}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=ROOT / 'results_headroom_r01')
    parser.add_argument('--inputs', type=Path, default=ROOT / 'inputs/olmoe_gsm8k_natural16')
    parser.add_argument('--metadata', type=Path, default=ROOT.parent / 'C_research_artifacts/20261001/20261001_c_instruct_model_metadata_v1')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    missing = [str(args.results / n / 'raw.json') for n in NAMES if not (args.results / n / 'raw.json').is_file()]
    if missing:
        parser.error('No metrics written; all six raw captures required. Missing: ' + ', '.join(missing))
    from tokenizers import Tokenizer
    workload = load(args.inputs / 'workload.json')
    sources, arrivals = workload['source_requests'], workload['arrival_traces_s']['steady']
    if ([s['example_index'] for s in sources] != list(range(32, 48)) or arrivals != [0.0] * 16
            or workload['actual_prompt_token_ids'] != [s['prompt_token_ids'] for s in sources]
            or workload['sampling'] != dict(ignore_eos=False, min_tokens=0, max_tokens=512, temperature=0.0, stop=[])):
        raise ValueError('source32..47/full-prompt/natural-sampling contract differs')
    tokenizer_path = args.metadata / 'tokenizer.json'
    expected_sha = next(r['sha256'] for r in load(args.metadata / 'metadata-receipt.json')['files'] if r['filename'] == 'tokenizer.json')
    if sha(tokenizer_path) != expected_sha:
        raise ValueError('local tokenizer differs from pinned metadata receipt')
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    cells, pauses = {}, {}
    for name in NAMES:
        directory, cap = args.results / name, int(name.split('_cap')[1])
        cell = analyze(directory, sources, arrivals, tokenizer, expected_budget=2048)
        raw, config = load(directory / 'raw.json'), load(directory / 'config.json')
        peak = load(directory / 'cuda_memory.json')['after_measurement']
        actual_peak = max(s['actual_active'] for s in raw['scheduler_steps'])
        cell.update(admission_cap=cap, observed_peak_active=actual_peak,
            cap_checks=dict(configured=config.get('admission_cap') == cap, raw_target=raw.get('target_cap') == cap,
                all_step_targets=all(s['target_cap'] == cap for s in raw['scheduler_steps']), observed_active_within_cap=actual_peak <= cap),
            peak_allocated_bytes=peak['peak_allocated_bytes'], peak_reserved_bytes=peak['peak_reserved_bytes'])
        cells[name], pauses[name] = cell, diagnose(raw)
    means = {}
    for cap, repeats in REPEATS.items():
        means[cap] = {k: statistics.mean(cells[n][k] for n in repeats) for k in
            ('episode_wall_s', 'output_tokens', 'output_tokens_per_s', 'completed_requests_per_s', 'accuracy',
             'correct_requests', 'correct_and_eos_requests', 'cap_truncated_requests', 'weight_copy_bytes',
             'group_count', 'peak_allocated_bytes', 'peak_reserved_bytes', 'observed_peak_active')}
        means[cap].update({f'{kind}_{stat}': statistics.mean(cells[n][kind][stat] for n in repeats)
            if all(cells[n][kind][stat] is not None for n in repeats) else None
            for kind in ('flow', 'ttft', 'itl', 'request_maxgap') for stat in ('mean_s', 'p50_s', 'p95_s', 'max_s')})
        means[cap].update(preemption_events=statistics.mean(cells[n]['preemption']['total_preemption_events'] for n in repeats),
            prompt_reprocessing_excess=statistics.mean(cells[n]['prompt_reprocessing']['observed_prompt_reprocessing_excess'] for n in repeats))
    report = dict(evidence_type='ACTUAL_NATURAL_EOS_STATIC_HEADROOM_EXECUTIONS', order=NAMES, fixed_token_budget=2048,
        group_status=load(args.results / 'group_status.json')['status'], all_six_complete=all(c['all_16_complete'] for c in cells.values()),
        all_budget_checks_pass=all(all(c['budget_checks'].values()) for c in cells.values()),
        all_cap_checks_pass=all(all(c['cap_checks'].values()) for c in cells.values()),
        workload_sha256=sha(args.inputs / 'workload.json'), tokenizer_sha256=expected_sha,
        extraction_source=str(ROOT.parent / 'C_research_artifacts/20261001/C_INSTRUCT_NATIVE128_ANALYZE_V1.py'),
        cells=cells, two_repeat_means=means,
        matched_cap_pairs=[comparison(REPEATS[candidate][i], REPEATS[baseline][i], cells)
            for baseline, candidate in combinations(REPEATS, 2) for i in (0, 1)],
        all_15_cell_pairs=[comparison(a, b, cells) for a, b in combinations(NAMES, 2)],
        two_repeat_mean_cap_pairs=[dict(left=candidate, right=baseline,
            ratios={k: ratio(means[candidate][k], means[baseline][k]) for k in means[baseline]
                if k not in ('accuracy', 'correct_requests', 'correct_and_eos_requests', 'cap_truncated_requests')},
            accuracy_difference=means[candidate]['accuracy'] - means[baseline]['accuracy'])
            for baseline, candidate in combinations(REPEATS, 2)], pause_diagnostic=pauses,
        notes=['All 16 source32..47 inputs retained; these are exploratory inputs already observed in the natural-budget run.',
            'Fixed token budget2048; cap16 is the current strong static baseline. cap10 tests one-request headroom below the previously observed peak11; it is not a KV-reservation guarantee.',
            'cap8 is an existing simple baseline; no new controller or selection among future outcomes is evaluated.',
            'Natural analyze() retains the historical last-number scorer, decoded text, token IDs, per-request quality, EOS/cap status and full latency distributions.',
            'All 16 inputs remain in the accuracy denominator. Incomplete requests score zero; 512-token truncations are separately reported.',
            'Actual token/s uses returned token count and full episode wall. Output lengths/content may differ; wall ratios are not equal-work speedups or quality equivalence.',
            'Both repeats retained and paired first/second; all15 pairwise output/answer differences are retained. Means average cell statistics, including quantiles.',
            'Prefill excess is observed original-prompt reprocessing, not all recomputation. Preemption counts alone do not measure pause severity.',
            'Pause diagnosis matches observed events only; waiting/recovery overlap is not causal attribution or recoverable savings.'])
    destination = args.output or args.results / 'metrics.json'
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(destination)
    for name, c in cells.items():
        print(name, 'tokens=', c['output_tokens'], 'wall=', c['episode_wall_s'], 'correct=', c['correct_requests'],
              'preemptions=', c['preemption']['total_preemption_events'], 'maxgap=', c['request_maxgap']['max_s'])

if __name__ == '__main__':
    main()
