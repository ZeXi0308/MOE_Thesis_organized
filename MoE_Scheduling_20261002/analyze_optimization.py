"""Summarize all six A/B/C/C/B/A cells, without selecting the best repetition."""
import argparse
import json
import math
import statistics
from pathlib import Path

load = lambda path: json.loads(path.read_text()) if path.exists() else {}
names = ('00_default24', '01_serial12', '02_overlap12', '03_overlap12', '04_serial12', '05_default24')
arms = dict(A=(names[0], names[5]), B=(names[1], names[4]), C=(names[2], names[3]))

def distribution(values):
    values = sorted(values)
    def q(p):
        i = (len(values) - 1) * p
        return values[math.floor(i)] * (1 - i % 1) + values[math.ceil(i)] * (i % 1)
    return dict(n=len(values), mean_s=statistics.mean(values) if values else None,
                p50_s=q(.5) if values else None, p95_s=q(.95) if values else None,
                max_s=max(values) if values else None)

def analyze_cell(directory):
    raw, status = load(directory / 'raw.json'), load(directory / 'status.json')
    pager, memory = load(directory / 'pager_summary.json'), load(directory / 'cuda_memory.json')
    resources = load(directory / 'resources.json')
    requests = raw.get('requests', [])
    outputs = {r['request_id']: r for r in requests}
    tokens = sum(len(r.get('output_token_ids', [])) for r in requests)
    complete = (raw.get('status') == status.get('status') == 'COMPLETE' and raw.get('error') is None
                and len(requests) == 16 and tokens == 512
                and all(r.get('status') == 'completed' and len(r.get('output_token_ids', [])) == 32 for r in requests))
    flow = [r['completion_s'] - r['arrival_s'] for r in requests if r.get('completion_s') is not None]
    ttft = [r['token_times_s'][0] - r['arrival_s'] for r in requests if r.get('token_times_s')]
    itl = [b - a for r in requests for a, b in zip(r.get('token_times_s', []), r.get('token_times_s', [])[1:])]
    measurement, peak = pager.get('measurement', {}), memory.get('after_measurement', {})
    metrics = dict(status=status.get('status', 'MISSING'), raw_status=raw.get('status', 'MISSING'),
        complete_16_requests_512_tokens=complete, requests=len(requests), output_tokens=tokens,
        completed_requests=sum(r.get('status') == 'completed' for r in requests),
        episode_wall_s=raw.get('observation_end_s'), flow=distribution(flow), ttft=distribution(ttft),
        itl=distribution(itl), token_level_itl_resolved=raw.get('host_chunk_diagnostics', {}).get('token_level_itl_resolved'),
        weight_copy_bytes=measurement.get('weight_copy_bytes'), group_count=measurement.get('group_count'),
        measurement_calls=pager.get('measurement_calls'), failed_pager_calls=pager.get('failed_calls'),
        expert_cap=pager.get('cap'), expert_scratch_bytes=pager.get('scratch_bytes'),
        kv_cache_bytes=resources.get('kv_cache_memory_bytes'),
        peak_allocated_bytes=peak.get('peak_allocated_bytes'), peak_reserved_bytes=peak.get('peak_reserved_bytes'))
    return metrics, outputs

def equivalent(left, right, outputs):
    a, b = outputs[left], outputs[right]
    keys, common = set(a) | set(b), set(a) & set(b)
    same = [k for k in common if a[k]['output_token_ids'] == b[k]['output_token_ids']]
    return dict(left=left, right=right, requests=len(keys), identical_full_sequences=len(same),
        exact_request_fraction=len(same) / len(keys) if keys else None,
        same_request_set=set(a) == set(b),
        prompts_identical=bool(common) and all(a[k]['prompt_token_ids_sha256'] == b[k]['prompt_token_ids_sha256'] for k in common),
        differing_request_ids=sorted(keys - set(same)))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=Path(__file__).parent / 'results_opt_r01')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    cells, outputs = {}, {}
    for name in names:
        cells[name], outputs[name] = analyze_cell(args.results / name)
    pairs = [(arms[a][0], arms[a][1]) for a in arms]
    pairs += [(arms[a][i], arms['C'][i]) for a in ('A', 'B') for i in (0, 1)]
    scalar_keys = ('episode_wall_s', 'weight_copy_bytes', 'group_count', 'measurement_calls',
                   'peak_allocated_bytes', 'peak_reserved_bytes')
    means = {}
    for arm, repeats in arms.items():
        if not all(cells[n]['complete_16_requests_512_tokens'] for n in repeats):
            means[arm] = None
            continue
        means[arm] = {k: statistics.mean(cells[n][k] for n in repeats) for k in scalar_keys}
        means[arm].update({f'{kind}_{stat}': statistics.mean(cells[n][kind][stat] for n in repeats)
            for kind in ('flow', 'ttft', 'itl') for stat in ('mean_s', 'p50_s', 'p95_s', 'max_s')})
    ratios = {}
    for baseline in ('A', 'B'):
        if means['C'] is None or means[baseline] is None:
            ratios[f'C/{baseline}'] = None
            continue
        ratios[f'C/{baseline}'] = dict(two_repeat_mean_wall_ratio=means['C']['episode_wall_s'] / means[baseline]['episode_wall_s'],
            two_repeat_mean_flow_ratio=means['C']['flow_mean_s'] / means[baseline]['flow_mean_s'],
            paired_repeats=[dict(C=arms['C'][i], baseline=arms[baseline][i],
                wall_ratio=cells[arms['C'][i]]['episode_wall_s'] / cells[arms[baseline][i]]['episode_wall_s'],
                mean_flow_ratio=cells[arms['C'][i]]['flow']['mean_s'] / cells[arms[baseline][i]]['flow']['mean_s']) for i in (0, 1)])
    report = dict(order=list(names), arms=dict(A='24-expert serial', B='12-expert serial', C='12-expert copy/compute overlap'),
        group_status=load(args.results / 'group_status.json').get('status', 'MISSING'),
        all_six_complete=all(c['complete_16_requests_512_tokens'] for c in cells.values()),
        cells=cells, two_repeat_means=means, C_ratios=ratios, output_equivalence=[equivalent(*p, outputs) for p in pairs],
        notes=['Episode/net wall is raw.observation_end_s, excluding engine initialization and full-workload warmup; no time subtractions.',
               'Flow/TTFT use workload arrival; ITL uses host-observed adjacent token times. Quantiles use linear interpolation.',
               'Arm summaries average both cell metrics, including cell quantiles; all six cells are retained.',
               'Copy bytes/group counts are measurement-only; CUDA peak is after drained warmup with peak reset.',
               'All requests are forced to 32 tokens; this measures bounded workload latency, not natural completion or answer quality.',
               'Ratios below one favor C; two fresh-engine repeats do not establish statistical confidence.'])
    destination = args.output or args.results / 'metrics.json'
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(destination)
    for name, c in cells.items():
        print(name, c['status'], 'tokens=', c['output_tokens'], 'wall=', c['episode_wall_s'], 'mean_flow=', c['flow']['mean_s'])
    print(json.dumps(ratios, indent=2))

if __name__ == '__main__':
    main()
