"""Partition four fixed-cap runs' generation intervals; no causal cost inference."""
import argparse
from collections import defaultdict
import json
from pathlib import Path


def analyze(path):
    raw = json.loads(path.read_text())
    steps, requests = raw['steps'], raw['requests']
    if not requests or not all(q['finished'] for q in requests):
        raise ValueError(f'{path}: requires complete requests')
    ends = defaultdict(list)
    for index, step in enumerate(steps):
        ends[step['end_s']].append(index)
    categories = ('positive_prefill', 'zero_prefill')
    totals = {k: dict(interval_count=0, gap_s=0.0, engine_s=0.0,
                      external_s=0.0) for k in categories}
    mapping = dict(interval_count=0, uniquely_mapped_interval_count=0,
                   unmapped=0, ambiguous=0, zero_interval=0,
                   same_step=0, nonadjacent_steps=0, producing_step_without_decode=0)
    rows = {}
    errors = []
    for request in requests:
        times = request['token_times_s']
        if len(times) < 2:
            raise ValueError(f'{path}: requires at least two outputs per request')
        row = {k: 0.0 for k in categories}
        row['generation_s'] = times[-1] - times[0]
        for previous, current in zip(times, times[1:]):
            mapping['interval_count'] += 1
            if not ends[previous] or not ends[current]:
                mapping['unmapped'] += 1
                continue
            if len(ends[previous]) != 1 or len(ends[current]) != 1:
                mapping['ambiguous'] += 1
                continue
            mapping['uniquely_mapped_interval_count'] += 1
            index, prior_index = ends[current][0], ends[previous][0]
            step = steps[index]
            mapping['zero_interval'] += current == previous
            mapping['same_step'] += index == prior_index
            mapping['nonadjacent_steps'] += index != prior_index + 1
            mapping['producing_step_without_decode'] += step['decode_tokens'] == 0
            key = 'positive_prefill' if step['prefill_tokens'] > 0 else 'zero_prefill'
            gap = current - previous
            row[key] += gap
            totals[key]['interval_count'] += 1
            totals[key]['gap_s'] += gap
            totals[key]['engine_s'] += step['end_s'] - step['start_s']
            totals[key]['external_s'] += step['start_s'] - previous
        errors.append(abs(sum(row[k] for k in categories) - row['generation_s']))
        rows[request['request_id']] = row
    unsupported = ('unmapped', 'ambiguous', 'zero_interval', 'same_step',
                   'nonadjacent_steps', 'producing_step_without_decode')
    if any(mapping[k] for k in unsupported) or max(errors) > 1e-10:
        raise ValueError(f'{path}: unsupported interval mapping/sumcheck: {mapping}, {max(errors)}')
    if len(rows) != len(requests):
        raise ValueError(f'{path}: duplicate request IDs')
    n = len(requests)
    means = {k: {field: totals[k][field] / n
                 for field in ('gap_s', 'engine_s', 'external_s')} for k in categories}
    host = {k: dict(step_count=0, engine_sum_s=0.0)
            for k in ('mixed', 'pure_decode', 'prefill_only', 'empty')}
    for step in steps:
        p, d = step['prefill_tokens'] > 0, step['decode_tokens'] > 0
        kind = ('mixed' if d else 'prefill_only') if p else ('pure_decode' if d else 'empty')
        host[kind]['step_count'] += 1
        host[kind]['engine_sum_s'] += step['end_s'] - step['start_s']
    result = dict(raw_path=str(path.resolve()), policy=raw['policy'], request_count=n,
                  interval_mapping=mapping, sumcheck=dict(passed=True,
                  tolerance_s=1e-10, max_per_request_error_s=max(errors)),
                  mean_generation_s=sum(row['generation_s'] for row in rows.values()) / n,
                  mean_per_request_components_s=means,
                  interval_count={k: totals[k]['interval_count'] for k in categories},
                  mean_interval_ms={k: totals[k]['gap_s'] / totals[k]['interval_count'] * 1000
                                    if totals[k]['interval_count'] else None for k in categories},
                  host_step_disjoint_aggregates=host, elapsed_s=raw['elapsed_s'])
    return result, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_directory', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    directory = args.run_directory.resolve()
    paths = sorted(directory.glob('[0-9][0-9]_*/raw.json'))
    if len(paths) != 4:
        raise ValueError('Requires exactly four formal raw files in small/large/large/small order')
    analyzed = [analyze(path) for path in paths]
    policies = [item[0]['policy'] for item in analyzed]
    if policies[0] != policies[3] or policies[1] != policies[2]:
        raise ValueError(f'Expected ABBA policy order, got {policies}')
    caps = [int(policy.removeprefix('fixed')) for policy in policies]
    if caps[0] >= caps[1]:
        raise ValueError('Expected smaller fixed cap first')
    pairs = []
    for small, large in ((0, 1), (3, 2)):
        sr, sq = analyzed[small]
        lr, lq = analyzed[large]
        if sq.keys() != lq.keys():
            raise ValueError('Paired request IDs differ')
        pairs.append(dict(small_run=paths[small].parent.name, large_run=paths[large].parent.name,
            mean_request_delta_large_minus_small_s={k: sum(lq[q][k] - sq[q][k] for q in sq) / len(sq)
                for k in ('positive_prefill', 'zero_prefill', 'generation_s')},
            mean_component_delta_large_minus_small_s={k: {
                field: lr['mean_per_request_components_s'][k][field] - sr['mean_per_request_components_s'][k][field]
                for field in ('gap_s', 'engine_s', 'external_s')}
                for k in ('positive_prefill', 'zero_prefill')},
            host_engine_sum_delta_large_minus_small_s={k:
                lr['host_step_disjoint_aggregates'][k]['engine_sum_s'] - sr['host_step_disjoint_aggregates'][k]['engine_sum_s']
                for k in sr['host_step_disjoint_aggregates']}))
    output = args.output or directory / 'generation_partition.json'
    report = dict(schema='generation_interval_partition_v1', status='VALID',
        scope='Descriptive, noncausal partition of complete observed trajectories. Every inter-output interval is classified by the native step producing its later token. Different runs have different states and category exposure; differences are not marginal prefill costs. Request means overlap across concurrent requests. Host engine.step sums are separate timeline totals, not extra terms to add to generation or E2E latency, and not GPU kernel time.',
        timing='gap = producing_step.end_s - prior_output_s = engine_s + external_s; engine_s = end_s - start_s; external_s = start_s - prior_output_s. Only unique, positive, adjacent-step intervals are supported; invalid inputs raise and produce no new result.',
        run_directory=str(directory), runs=[item[0] for item in analyzed], pairs=pairs)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(status=report['status'], output=str(output.resolve()), runs=len(analyzed))))


if __name__ == '__main__':
    main()
