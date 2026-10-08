"""Actual budget runs, both repetitions and their corresponding b512 controls."""
import argparse
import collections
import json
import statistics
from pathlib import Path
from analyze_optimization import analyze_cell, equivalent, load

def step_summary(directory, raw, cell):
    steps, engines = raw['scheduler_steps'], {}
    for call in raw['engine_calls']:
        assert call['returned'] and call['scheduler_step_stop'] == call['scheduler_step_start'] + 1
        key = call['scheduler_step_start']
        assert key not in engines
        engines[key] = call['return_s'] - call['start_s']
    groups, copies = collections.Counter(), collections.Counter()
    for line in (directory / 'pager/calls.jsonl').open():
        call = json.loads(line)
        if call.get('measurement') and not call.get('validation_run'):
            assert call['status'] == 'complete'
            key = call['context']['step_id']
            groups[key] += call['group_count']
            copies[key] += call['weight_copy_bytes']
    assert set(engines) == set(groups) == {s['step'] for s in steps}
    kinds = {kind: dict(steps=0, prefill_tokens=0, decode_tokens=0, scheduled_tokens=0,
        total_engine_wall_s=0, groups=0, weight_copy_bytes=0) for kind in ('pure_prefill', 'mixed', 'pure_decode')}
    per_request = collections.Counter()
    for step in steps:
        prefill = sum(r['prefill_tokens'] for r in step['scheduled'])
        decode = sum(r['decode_tokens'] for r in step['scheduled'])
        assert prefill + decode == step['total_scheduled_tokens'] and prefill + decode > 0
        for request in step['scheduled']:
            per_request[request['request_id']] += request['prefill_tokens']
        kind = 'mixed' if prefill and decode else 'pure_prefill' if prefill else 'pure_decode'
        entry, key = kinds[kind], step['step']
        for field, value in dict(steps=1, prefill_tokens=prefill, decode_tokens=decode,
                scheduled_tokens=prefill + decode, total_engine_wall_s=engines[key],
                groups=groups[key], weight_copy_bytes=copies[key]).items():
            entry[field] += value
    for entry in kinds.values():
        entry.update(groups_per_step=entry['groups'] / entry['steps'] if entry['steps'] else None,
            copy_bytes_per_group=entry['weight_copy_bytes'] / entry['groups'] if entry['groups'] else None,
            engine_wall_ms_per_group=1000 * entry['total_engine_wall_s'] / entry['groups'] if entry['groups'] else None)
    assert sum(groups.values()) == cell['group_count'] and sum(copies.values()) == cell['weight_copy_bytes']
    expected = {r['request_id']: r['prompt_tokens'] for r in raw['requests']}
    return dict(by_kind=kinds, total_engine_wall_s=sum(engines.values()),
        scheduled_token_max=max(s['total_scheduled_tokens'] for s in steps),
        prefill_check=dict(expected_tokens=sum(expected.values()), observed_tokens=sum(per_request.values()),
            each_request_matches=dict(per_request) == expected),
        total_decode_tokens=sum(v['decode_tokens'] for v in kinds.values()),
        engine_wall_ms_per_group=1000 * sum(engines.values()) / cell['group_count'],
        copy_bytes_per_group=cell['weight_copy_bytes'] / cell['group_count'])

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=Path(__file__).parent / 'results_budget_r01')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    names = ('00_b512', '01_b2048', '02_b4096', '03_b4096', '04_b2048', '05_b512')
    repeats = {512: (names[0], names[5]), 2048: (names[1], names[4]), 4096: (names[2], names[3])}
    cells, outputs = {}, {}
    for name in names:
        directory = args.results / name
        cells[name], outputs[name] = analyze_cell(directory)
        if not cells[name]['complete_16_requests_512_tokens']:
            raise ValueError(f'{name}: complete 16-request/512-token capture required; no metrics written')
        raw, policy = load(directory / 'raw.json'), load(directory / 'policy_application.json')
        steps = step_summary(directory, raw, cells[name])
        budget = int(name.split('_b')[1])
        actual = policy.get('measured_scheduler_token_budget')
        cells[name].update(preemption=raw.get('preemption_summary'), stages=steps,
            policy_application=policy, budget_check=dict(expected=budget, actual=actual,
                matches=actual == budget, observed_tokens_within_budget=steps['scheduled_token_max'] <= budget,
                common_engine4096_warmup512=policy.get('engine_token_budget') == 4096 and policy.get('warmup_token_budget') == 512))
    pairs = [(repeats[b][i], repeats[512][i]) for b in (2048, 4096) for i in (0, 1)]
    comparisons = []
    for candidate, baseline in pairs:
        a, b = cells[candidate], cells[baseline]
        comparisons.append(dict(candidate=candidate, baseline=baseline,
            wall_ratio=a['episode_wall_s'] / b['episode_wall_s'], mean_flow_ratio=a['flow']['mean_s'] / b['flow']['mean_s'],
            mean_ttft_ratio=a['ttft']['mean_s'] / b['ttft']['mean_s'], mean_itl_ratio=a['itl']['mean_s'] / b['itl']['mean_s'],
            copy_bytes_ratio=a['weight_copy_bytes'] / b['weight_copy_bytes'], groups_ratio=a['group_count'] / b['group_count'],
            engine_wall_per_group_ratio=a['stages']['engine_wall_ms_per_group'] / b['stages']['engine_wall_ms_per_group'],
            output_equivalence=equivalent(candidate, baseline, outputs)))
    means = {}
    for budget, pair in repeats.items():
        means[budget] = {k: statistics.mean(cells[n][k] for n in pair) for k in
            ('episode_wall_s', 'weight_copy_bytes', 'group_count', 'peak_allocated_bytes', 'peak_reserved_bytes')}
        means[budget].update({f'{kind}_{stat}': statistics.mean(cells[n][kind][stat] for n in pair)
            for kind in ('flow', 'ttft', 'itl') for stat in ('mean_s', 'p50_s', 'p95_s')})
    report = dict(evidence_type='INDEPENDENT_ACTUAL_BUDGET_EXECUTIONS', order=list(names),
        group_status=load(args.results / 'group_status.json').get('status', 'MISSING'), all_six_complete=True,
        cells=cells, two_repeat_means=means, paired_vs_512=comparisons,
        two_repeat_mean_ratios_vs512={b: {k: means[b][k] / means[512][k] for k in
            ('episode_wall_s', 'flow_mean_s', 'ttft_mean_s', 'itl_mean_s', 'weight_copy_bytes', 'group_count')} for b in (2048, 4096)},
        within_budget_output_equivalence=[equivalent(*pair, outputs) for pair in repeats.values()],
        notes=['Both repeats retained; candidate first/second pairs with first/second b512. Means average cell metrics.',
               'Actual measured scheduler budget and common engine4096/warmup512 are checked from policy_application.',
               'Stages use scheduled prefill/decode tokens, joined to one returned engine call per scheduler step.',
               'Group/byte costs are actual measurement pager sums. Engine wall per group includes all engine work and host overhead; not kernel latency or pure H2D cost.',
               'Episode wall excludes initialization/warmup. Flow/TTFT start at arrival; ITL is host-observed.',
               '16 requests forced to 32 output tokens; exploratory input, not natural EOS, answer quality or a complete paper evaluation.'])
    destination = args.output or args.results / 'metrics.json'
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(destination)

if __name__ == '__main__':
    main()
