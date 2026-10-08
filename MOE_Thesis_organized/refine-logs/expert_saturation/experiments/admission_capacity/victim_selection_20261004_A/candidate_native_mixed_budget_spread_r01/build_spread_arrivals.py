#!/usr/bin/env python3
"""Reproduce the isolated 0.2-second arrival input from its frozen 0.01-second parent."""
import argparse
import copy
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / 'candidate_native_mixed_budget_probe_r01'
INPUTS = Path('pkg/inputs/pro_high')


def read(path):
    return json.loads(path.read_text())


def build(source, check):
    config, workload, stats = [read(source / INPUTS / name)
                              for name in ('config.json', 'workload.json', 'stats.json')]
    count = len(workload['source_requests'])
    if (count != 320 or workload['arrival_traces_s'] != {'steady': [i / 100 for i in range(count)]}
            or config['arrival_gap_s'] != .01 or config['output_tokens'] != 1024
            or len(config['output_tokens_by_request']) != 40
            or set(config['output_tokens_by_request'].values()) != {128}):
        raise ValueError('Source must be the frozen 320-request, 40/280 budget-mixture 0.01s input')
    workload = copy.deepcopy(workload)
    workload['arrival_traces_s']['steady'] = [i * .2 for i in range(count)]
    workload['arrival_rule'] = 'External arrival at i*0.2 seconds in deterministic length-stratified order.'
    identity = hashlib.sha256(json.dumps(workload, sort_keys=True).encode()).hexdigest()
    config.update(arrival_gap_s=.2, arrival_span_s=workload['arrival_traces_s']['steady'][-1],
                  workload_sha256=identity)
    stats.update(arrival_span_s=config['arrival_span_s'], workload_sha256=identity)
    for name, value in zip(('config.json', 'workload.json', 'stats.json'), (config, workload, stats)):
        path = ROOT / INPUTS / name
        serialized = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
        if check:
            if path.read_text() != serialized:
                raise ValueError(f'Spread-arrival input differs from deterministic builder: {name}')
        else:
            path.write_text(serialized)
    print(json.dumps(dict(status='CHECKED' if check else 'BUILT', requests=count,
        arrival_gap_s=.2, arrival_span_s=config['arrival_span_s'], workload_sha256=identity,
        semantics='Only the external arrival trace changes; development opportunity diagnosis, not a policy gain.'),
        ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    build(args.source.resolve(), args.check)
