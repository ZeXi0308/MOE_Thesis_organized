"""Full-episode break-even accounting; descriptive, not a counterfactual oracle."""
import argparse
import json
import math
from pathlib import Path
from statistics import mean


def cell_cost(c):
    stages = c['stages']['by_kind']
    engine = c['stages']['total_engine_wall_s']
    assert math.isclose(sum(s['total_engine_wall_s'] for s in stages.values()),
                        engine, rel_tol=1e-9, abs_tol=1e-8)
    total, output = c['kv_service_cost']['drained_s'], c['output_tokens']
    outside = total - engine
    assert outside >= -1e-7 and output > 0
    decode = stages['pure_decode']
    return dict(output_tokens=output, drained_s=total, engine_s=engine,
        outside_engine_and_drain_s=outside, seconds_per_output_token=total/output,
        expert_bytes_per_output_token=c['weight_copy_bytes']/output,
        pure_decode_steps=decode['scheduler_steps'],
        pure_decode_s=decode['total_engine_wall_s'],
        pure_decode_s_per_step=decode['total_engine_wall_s']/decode['scheduler_steps'],
        pure_decode_expert_bytes_per_step=decode['weight_copy_bytes']/decode['scheduler_steps'])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--results', type=Path, required=True)
    args = p.parse_args()
    m = json.loads((args.results/'metrics.json').read_text())
    assert m['design'] in ('ngram', 'ngram_short') and m['all_cells_complete']
    costs = {name: cell_cost(c) for name, c in m['cells'].items()}
    ar = [n for n in m['order'] if n.endswith('_ar16')]
    arms = sorted({n.split('_', 1)[1] for n in m['order'] if n not in ar})
    pairs = []
    for arm in arms:
        names = [n for n in m['order'] if n.endswith('_'+arm)]
        assert len(names) == len(ar) == 2
        for a, b in zip(ar, names):
            ca, cb = costs[a], costs[b]
            threshold = ca['drained_s'] * cb['output_tokens']/ca['output_tokens']
            pairs.append(dict(baseline=a, candidate=b,
                output_ratio=cb['output_tokens']/ca['output_tokens'],
                time_ratio=cb['drained_s']/ca['drained_s'],
                throughput_ratio=(cb['output_tokens']/cb['drained_s']) /
                                 (ca['output_tokens']/ca['drained_s']),
                time_to_match_baseline_rate_at_observed_candidate_output_s=threshold,
                required_time_reduction_at_observed_candidate_output_s=cb['drained_s']-threshold,
                required_time_reduction_fraction=1-threshold/cb['drained_s']))
    result = dict(status='DESCRIPTIVE_ACCOUNTING_COMPLETE', cells=costs, pairs=pairs,
        means={arm: {k: mean(v[k] for n, v in costs.items() if n.endswith('_'+arm))
                     for k in next(iter(costs.values()))} for arm in ['ar16']+arms},
        scope=['T equals mutually exclusive engine-stage times plus outside-engine/drain time.',
               'Throughput break-even is T_candidate/T_AR < output_candidate/output_AR.',
               'Thresholds hold observed outputs fixed algebraically; they are not attainable savings estimates.',
               'Policy changes can alter output, routes, cache, KV, and later work.',
               'Expert bytes per step/token are descriptive; no DMA critical-path attribution.',
               'No exact draft acceptance count/rate or layer-suffix opportunity is inferred.'])
    with (args.results/'ngram_cost_model.json').open('x') as f:
        json.dump(result, f, indent=2); f.write('\n')
    print(json.dumps({'status': result['status'], 'pairs': pairs}, indent=2))


if __name__ == '__main__':
    main()
