#!/usr/bin/env python3
"""Compare recorded executions; no counterfactual, trimming, or raw writes."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

WORKSPACE = next(p for p in Path(__file__).resolve().parents if (p / 'refine-logs').is_dir())
sys.dont_write_bytecode = True
sys.path.insert(0, str(WORKSPACE / 'refine-logs/expert_saturation/experiments/admission_capacity'))
from analyze_completion_headroom import held_accounting
from analyze_pause_ledger import read_raw


def fingerprint(path):
    return dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def normalize(value, aliases, skip):
    if isinstance(value, dict):
        return {aliases.get(k, k): normalize(v, aliases, skip) for k, v in value.items() if k not in skip}
    if isinstance(value, list):
        return [normalize(v, aliases, skip) for v in value]
    return aliases.get(value, value) if isinstance(value, str) else value


def first_recovery(raw, decisions):
    event = next((d for d in decisions if d.get('forced_preempted')), None)
    if event is None:
        return None
    internal = event['recovery_target']
    request = next(r for r in raw['requests'] if r['internal_request_id'] == internal)
    output_index = event['recovery_output_at_start']
    returned = request['token_times_s'][output_index]
    call = next(c for c in raw['engine_steps'] if c['returned_s'] == returned)
    first, last = event['step'], call['scheduler_step_start']
    return dict(target=request['request_id'], victim=raw['internal_to_source'][event['forced_preempted'][0]],
                outputs_before=output_index, first_step=first, last_step=last,
                recomputed_positions=sum(r['recompute_tokens'] for s in raw['scheduler_steps'][first:last + 1]
                                         for r in s['scheduled'] if r['request_id'] == request['request_id']),
                final_call=call_cost(raw, decisions, last))


def call_cost(raw, decisions, index):
    c, s = raw['engine_steps'][index], raw['scheduler_steps'][index]
    return dict(step=index, start_s=c['start_s'], returned_s=c['returned_s'],
                engine_s=c['returned_s'] - c['start_s'], scheduler_s=s['end_s'] - s['start_s'],
                decision_s=decisions[index]['decision_seconds'], width=len(s['scheduled']),
                recompute_tokens=s['recompute_tokens'], scheduled_tokens=s['total_scheduled_tokens'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--labels', nargs=2, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    raw, decisions, inputs, cells = [], [], [], []
    for label in args.labels:
        base = args.run_dir / 'gpu_results' / label
        raw_path = next(p for p in [base / 'raw.json', base / 'raw.json.gz'] if p.exists())
        paths = [raw_path, base / 'headroom-decisions.json', base / 'config.json', base / 'engine_args.json']
        r, d, config, engine_args = map(read_raw, paths)
        raw.append(r)
        decisions.append(d)
        inputs.extend(map(fingerprint, paths))
        held = held_accounting(r, d, config['completion_policy'])
        engine = sum(c['returned_s'] - c['start_s'] for c in r['engine_steps'])
        scheduler = sum(s['end_s'] - s['start_s'] for s in r['scheduler_steps'])
        origin = r['requests'][0]['native_metrics']['arrival_time'] - r['requests'][0]['arrival_s']
        cells.append(dict(label=label, epoch_origin=origin, wall_s=r['observation_end_s'],
                          scheduler_s=scheduler, engine_non_schedule_s=engine - scheduler,
                          outside_engine_s=r['observation_end_s'] - engine,
                          held_request_steps=held['held_request_steps'],
                          first_successful_forced_recovery=first_recovery(r, d)))
    checks = {}
    for key, seq, skip in [
        ('scheduler_with_queue', [r['scheduler_steps'] for r in raw], ['start_s', 'end_s']),
        ('executed_schedule', [r['scheduler_steps'] for r in raw], ['start_s', 'end_s', 'waiting_before', 'waiting_requests']),
        ('decisions', decisions, ['decision_seconds', 'wrapped_schedule_seconds']),
        ('engine_logical', [r['engine_steps'] for r in raw], ['start_s', 'returned_s']),
    ]:
        values = [normalize(s, r['internal_to_source'], skip) for s, r in zip(seq, raw)]
        different = [i for i, (u, v) in enumerate(zip(*values)) if u != v]
        checks[key] = dict(lengths=list(map(len, values)), different_indices=different)
    outputs = [{r['request_id']: r['output_token_ids'] for r in x['requests']} for x in raw]
    checks['output_sequences_equal_count'] = sum(v == outputs[1].get(k) for k, v in outputs[0].items())
    costs = []
    schedule = checks['executed_schedule']
    if len(set(schedule['lengths'])) == 1 and not schedule['different_indices']:
        for i in range(len(raw[0]['engine_steps'])):
            pair = [call_cost(r, d, i) for r, d in zip(raw, decisions)]
            costs.append(dict(step=i, cells=pair, engine_delta_s=pair[0]['engine_s'] - pair[1]['engine_s']))
    result = dict(scope='Observed host execution only; timing differences do not identify GPU or compilation cost.',
                  inputs=inputs, script=fingerprint(Path(__file__)), cells=cells, identity=checks,
                  largest_positive_same_step_deltas=sorted(costs, key=lambda c: c['engine_delta_s'], reverse=True)[:5])
    data = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
    if args.output:
        with args.output.open('x') as stream:
            stream.write(data)
        print(args.output)
    else:
        print(data, end='')


if __name__ == '__main__':
    main()
