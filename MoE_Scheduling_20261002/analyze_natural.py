"""Decode/score every natural-EOS request; retain all six budget executions.

Local invocation: /private/tmp/moe-c-input-env/bin/python analyze_natural.py
No downloads or GPU calls. No output file is written until all six raw captures exist.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path
import re
import statistics

from analyze_optimization import distribution
from analyze_budget import step_summary

ROOT = Path(__file__).resolve().parent
NAMES = ('00_b512', '01_b2048', '02_b4096', '03_b4096', '04_b2048', '05_b512')
REPEATS = {512: (NAMES[0], NAMES[5]), 2048: (NAMES[1], NAMES[4]), 4096: (NAMES[2], NAMES[3])}
NUMBERS = re.compile(r"[-+]?\d*\.\d+|\d+")
GROUP_COMMA = re.compile(r"(\d),(\d)")


def normalized_last_number(text: str) -> str | None:
    # Exact historical eval/gsm/run_eval.py extraction: remove each digit
    # grouping comma, then take the last regex number from model output.
    normalized = GROUP_COMMA.sub(r"\1\2", text)
    numbers = NUMBERS.findall(normalized)
    return numbers[-1] if numbers else None


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ratio(a, b):
    return a / b if a is not None and b not in (None, 0) else None


def analyze(directory, sources, arrivals, tokenizer, expected_budget=None, stage_summarizer=step_summary):
    raw, status = load(directory / 'raw.json'), load(directory / 'status.json')
    policy, pager = load(directory / 'policy_application.json'), load(directory / 'pager_summary.json')
    rows = raw['requests']
    by_id = {r['request_id']: r for r in rows}
    expected = {s['request_id'] for s in sources}
    if len(by_id) != len(rows) or set(by_id) != expected:
        raise ValueError(f'{directory.name}: request inventory differs from all 16 frozen inputs')
    contract = raw.get('generation_contract', {})
    if not (contract.get('natural_eos') is True and contract.get('ignore_eos') is False
            and contract.get('min_tokens') == 0
            and contract.get('max_output_tokens_by_request') == {rid: 512 for rid in expected}):
        raise ValueError(f'{directory.name}: natural-EOS/512-token contract differs')
    budget = int(directory.name.split('_b')[1]) if expected_budget is None else expected_budget
    checks = dict(measured_budget=policy.get('measured_scheduler_token_budget') == budget,
        common_engine4096=policy.get('engine_token_budget') == 4096,
        common_warmup512=policy.get('warmup_token_budget') == 512,
        common_fixed32_warmup=policy.get('full_warmup_fixed_output_tokens') == 32,
        natural_eos=policy.get('natural_eos') is True,
        output_cap512=policy.get('measurement_output_cap') == 512)
    per_request, all_itls, all_chunk_gaps = [], [], []
    positive_events = [e for e in raw['output_events'] if e.get('chunk_size', 0) > 0]
    token_level_resolved = all(e['chunk_size'] == 1 for e in positive_events)
    for source, arrival in zip(sources, arrivals):
        row = by_id[source['request_id']]
        ids, times = row['output_token_ids'], row['token_times_s']
        if (row['prompt_token_ids_sha256'] != source['prompt_token_ids_sha256']
                or row['prompt_tokens'] != len(source['prompt_token_ids']) or row['arrival_s'] != arrival
                or row['max_output_tokens'] != 512):
            raise ValueError(f'{directory.name}/{source["request_id"]}: input identity differs')
        if (len(ids) != len(times) or len(ids) > 512
                or any(type(t) is not int or t < 0 for t in ids)
                or any(not math.isfinite(t) or t < arrival for t in times)
                or any(a > b for a, b in zip(times, times[1:]))):
            raise ValueError(f'{directory.name}/{source["request_id"]}: invalid tokens/times')
        finished, reason = row['status'] == 'completed', row.get('finish_reason')
        if finished and (reason not in ('stop', 'length') or (reason == 'length' and len(ids) != 512)):
            raise ValueError(f'{directory.name}/{source["request_id"]}: inconsistent finish/cap')
        text = tokenizer.decode(ids, skip_special_tokens=True)
        answer = normalized_last_number(text)
        events = sorted((e for e in positive_events if e['request_id'] == source['request_id']), key=lambda e: e['received_s'])
        if sum(e['chunk_size'] for e in events) != len(ids):
            raise ValueError(f'{directory.name}: output event/token counts differ')
        chunk_times = [e['received_s'] for e in events]
        chunk_gaps = [b - a for a, b in zip(chunk_times, chunk_times[1:])]
        resolved = all(e['chunk_size'] == 1 for e in events)
        itls = [b - a for a, b in zip(times, times[1:])] if resolved else []
        all_itls.extend(itls)
        all_chunk_gaps.extend(chunk_gaps)
        completion = row.get('completion_s')
        if finished and (completion is None or completion < arrival or (times and completion < times[-1])):
            raise ValueError(f'{directory.name}/{source["request_id"]}: invalid completion time')
        flow = completion - arrival if finished else None
        per_request.append(dict(request_id=source['request_id'], example_index=source['example_index'],
            question=source['question'], gold=source['gold'], output_text=text, answer=answer,
            score=int(finished and answer == source['gold']),
            correct_and_eos=int(finished and reason == 'stop' and answer == source['gold']),
            finished=finished, finish_reason=reason, stop_reason=row.get('stop_reason'),
            eos_only_stop_inferred=finished and reason == 'stop', cap_truncated=finished and reason == 'length',
            reached_output_cap=len(ids) == 512, output_tokens=len(ids), output_token_ids=ids,
            arrival_s=arrival, completion_s=completion, flow_s=flow,
            ttft_s=times[0] - arrival if times else None, itl=distribution(itls),
            inter_chunk_gap=distribution(chunk_gaps), token_level_itl_resolved=resolved,
            maxgap_s=max(chunk_gaps) if chunk_gaps else None,
            maxgap_scope='Maximum positive-output host receipt gap; excludes TTFT and unresolved intra-chunk token gaps',
            output_tokens_per_flow_s=ratio(len(ids), flow),
            decode_tokens_per_s=ratio(len(ids) - 1, times[-1] - times[0]) if len(times) > 1 and resolved else None))
    measurement = pager['measurement']
    tokens, wall = sum(r['output_tokens'] for r in per_request), raw['observation_end_s']
    original_prompt_tokens = sum(len(s['prompt_token_ids']) for s in sources)
    scheduled_prefill_tokens = sum(r['prefill_tokens'] for step in raw['scheduler_steps'] for r in step['scheduled'])
    preemption = raw.get('preemption_summary', {})
    complete = (raw.get('status') == status.get('status') == 'COMPLETE' and raw.get('error') is None
                and all(r['finished'] for r in per_request))
    cell = dict(status=status.get('status'), raw_status=raw.get('status'), error=raw.get('error'),
        all_16_complete=complete, budget=budget, budget_checks=checks, policy_application=policy,
        requests=16, completed_requests=sum(r['finished'] for r in per_request),
        output_tokens=tokens, episode_wall_s=wall, output_tokens_per_s=ratio(tokens, wall),
        completed_requests_per_s=ratio(sum(r['finished'] for r in per_request), wall),
        accuracy=sum(r['score'] for r in per_request) / 16,
        correct_requests=sum(r['score'] for r in per_request),
        correct_and_eos_requests=sum(r['correct_and_eos'] for r in per_request),
        finish_counts=dict(Counter(r['finish_reason'] or 'unfinished' for r in per_request)),
        cap_truncated_requests=sum(r['cap_truncated'] for r in per_request),
        flow=distribution([r['flow_s'] for r in per_request if r['flow_s'] is not None]),
        ttft=distribution([r['ttft_s'] for r in per_request if r['ttft_s'] is not None]),
        itl=distribution(all_itls) if token_level_resolved else distribution([]),
        inter_chunk_gap=distribution(all_chunk_gaps),
        request_maxgap=distribution([r['maxgap_s'] for r in per_request if r['maxgap_s'] is not None]),
        request_maxgap_scope='Per-request maximum positive-output host receipt gap; comparable emission cadence, not reconstructed intra-chunk ITL',
        token_level_itl_resolved=token_level_resolved,
        preemption={k: preemption.get(k) for k in
            ('total_preemption_events', 'steps_with_preemption', 'distinct_preempted_requests')},
        prompt_reprocessing=dict(original_prompt_tokens=original_prompt_tokens,
            actual_scheduled_prefill_tokens=scheduled_prefill_tokens,
            observed_prompt_reprocessing_excess=scheduled_prefill_tokens - original_prompt_tokens,
            scope='Observed scheduled prefill minus original prompt tokens; not all recomputation. Incomplete runs can have negative excess.'),
        weight_copy_bytes=measurement['weight_copy_bytes'],
        group_count=measurement['group_count'], failed_pager_calls=pager.get('failed_calls'),
        expert_cap=pager.get('cap'), expert_scratch_bytes=pager.get('scratch_bytes'),
        per_request=per_request)
    if complete:
        cell['stages'] = stage_summarizer(directory, raw, cell)
        cell['budget_checks']['observed_tokens_within_budget'] = cell['stages']['scheduled_token_max'] <= budget
    return cell


def comparison(left, right, cells):
    a, b = cells[left], cells[right]
    differences = []
    for x, y in zip(a['per_request'], b['per_request']):
        assert x['request_id'] == y['request_id']
        ix, iy = x['output_token_ids'], y['output_token_ids']
        common_prefix = next((i for i, (u, v) in enumerate(zip(ix, iy)) if u != v), min(len(ix), len(iy)))
        differences.append(dict(request_id=x['request_id'], identical_token_ids=ix == iy,
            identical_text=x['output_text'] == y['output_text'], common_prefix_tokens=common_prefix,
            left_tokens=len(ix), right_tokens=len(iy), left_answer=x['answer'], right_answer=y['answer'],
            left_score=x['score'], right_score=y['score'],
            left_finish=x['finish_reason'], right_finish=y['finish_reason']))
    return dict(left=left, right=right,
        ratios_left_over_right={k: ratio(a[k], b[k]) for k in
            ('episode_wall_s', 'output_tokens', 'output_tokens_per_s', 'weight_copy_bytes', 'group_count')},
        latency_ratios={f'{kind}_{stat}': ratio(a[kind][stat], b[kind][stat])
            for kind in ('flow', 'ttft', 'itl', 'request_maxgap') for stat in ('mean_s', 'p95_s')},
        accuracy_difference=a['accuracy'] - b['accuracy'],
        left_only_correct=sum(d['left_score'] > d['right_score'] for d in differences),
        right_only_correct=sum(d['right_score'] > d['left_score'] for d in differences),
        identical_full_sequences=sum(d['identical_token_ids'] for d in differences),
        same_answers=sum(d['left_answer'] == d['right_answer'] for d in differences),
        per_request=differences)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=ROOT / 'results_natural_r01')
    parser.add_argument('--inputs', type=Path, default=ROOT / 'inputs/olmoe_gsm8k_natural16')
    parser.add_argument('--metadata', type=Path, default=ROOT.parent / 'C_research_artifacts/20261001/20261001_c_instruct_model_metadata_v1')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    missing = [str(args.results / name / 'raw.json') for name in NAMES if not (args.results / name / 'raw.json').is_file()]
    if missing:
        parser.error('No metrics written; all six raw captures are required. Missing: ' + ', '.join(missing))
    from tokenizers import Tokenizer
    workload = load(args.inputs / 'workload.json')
    sources, arrivals = workload['source_requests'], workload['arrival_traces_s']['steady']
    if ([s['example_index'] for s in sources] != list(range(32, 48)) or arrivals != [0.0] * 16
            or workload['actual_prompt_token_ids'] != [s['prompt_token_ids'] for s in sources]
            or workload['sampling'] != dict(ignore_eos=False, min_tokens=0, max_tokens=512, temperature=0.0, stop=[])):
        raise ValueError('frozen source32..47/full-prompt/natural-sampling contract differs')
    tokenizer_path = args.metadata / 'tokenizer.json'
    expected_sha = next(r['sha256'] for r in load(args.metadata / 'metadata-receipt.json')['files'] if r['filename'] == 'tokenizer.json')
    if sha(tokenizer_path) != expected_sha:
        raise ValueError('local tokenizer differs from its pinned metadata receipt')
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    cells = {name: analyze(args.results / name, sources, arrivals, tokenizer) for name in NAMES}
    means = {}
    for budget, repeats in REPEATS.items():
        means[budget] = {k: statistics.mean(cells[n][k] for n in repeats) for k in
            ('episode_wall_s', 'output_tokens', 'output_tokens_per_s', 'accuracy', 'cap_truncated_requests', 'weight_copy_bytes', 'group_count')}
        means[budget].update({f'{kind}_{stat}': statistics.mean(cells[n][kind][stat] for n in repeats)
            if all(cells[n][kind][stat] is not None for n in repeats) else None
            for kind in ('flow', 'ttft', 'itl', 'request_maxgap') for stat in ('mean_s', 'p50_s', 'p95_s', 'max_s')})
    report = dict(evidence_type='NATIVE_NATURAL_EOS_BUDGET_EXECUTIONS', order=NAMES,
        group_status=load(args.results / 'group_status.json')['status'],
        all_six_complete=all(c['all_16_complete'] for c in cells.values()),
        all_budget_checks_pass=all(all(c['budget_checks'].values()) for c in cells.values()),
        workload_sha256=sha(args.inputs / 'workload.json'), tokenizer_sha256=expected_sha,
        extraction_source=str(ROOT.parent / 'C_research_artifacts/20261001/C_INSTRUCT_NATIVE128_ANALYZE_V1.py'),
        cells=cells, two_repeat_means=means,
        matched_budget_pairs=[comparison(REPEATS[high][i], REPEATS[low][i], cells)
            for low, high in combinations(REPEATS, 2) for i in (0, 1)],
        all_15_cell_pairs=[comparison(a, b, cells) for a, b in combinations(NAMES, 2)],
        two_repeat_mean_budget_pairs=[dict(left=high, right=low,
            ratios={k: ratio(means[high][k], means[low][k]) for k in means[low] if k not in ('accuracy', 'cap_truncated_requests')},
            accuracy_difference=means[high]['accuracy'] - means[low]['accuracy']) for low, high in combinations(REPEATS, 2)],
        notes=['All 16 source-order requests remain in the accuracy denominator; incomplete requests score zero.',
            'Score is the unchanged historical last-number string match, not a new answer parser. Length-capped outputs are scored and separately marked; correct_and_eos excludes caps.',
            'stop is inferred EOS under the no-extra-stop sampling contract; native stop identifier is retained, EOS need not appear in returned token IDs.',
            'Actual returned token count includes only captured IDs. Token/s divides it by full episode wall, excluding initialization/warmup.',
            'Natural output length/content can differ: wall ratios are observed episode comparisons, not equal-work speedups or quality-equivalence evidence.',
            'Flow/TTFT start at arrival; ITL is adjacent host token time. request_maxgap.p95_s is p95 across per-request maximum ITLs, excluding TTFT.',
            'Native preemption counts are reported. The raw total_recomputed_tokens observes only negative computed-token adjustments and is not a complete recomputation counter; zero does not mean no recomputation.',
            'prompt_reprocessing compares all actually scheduled prefill tokens with original prompt tokens. Its excess is observed prompt reprocessing, not all recomputation.',
            'Both repeats retained; matched comparisons pair first/second repeat. All 15 cell pairs expose within-budget and cross-budget output divergence.',
            'Two-repeat means average cell statistics, including quantiles; pooled ITL quantiles within each cell are token-weighted.',
            '16 GSM8K items detect obvious quality loss only; forced 512-token caps are not natural answer completion.'])
    destination = args.output or args.results / 'metrics.json'
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(destination)
    for name, cell in cells.items():
        print(name, 'tokens=', cell['output_tokens'], 'wall=', cell['episode_wall_s'],
              'correct=', cell['correct_requests'], '/16', 'finish=', cell['finish_counts'])


if __name__ == '__main__':
    main()
