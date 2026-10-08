#!/usr/bin/env python3
"""Assign synthetic budget variation from stable input IDs, without outcome data."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics


PREFIX = 'A-budget-mixture-v1:'


def build(root, check=False):
    folder = root / 'pkg/inputs/pro_high'
    path = folder / 'config.json'
    config = json.loads(path.read_text())
    workload_bytes = (folder / 'workload.json').read_bytes()
    workload = json.loads(workload_bytes)
    ids = [row['request_id'] for row in workload['source_requests']]
    if len(ids) != 320 or len(set(ids)) != 320 or config['output_tokens'] != 1024:
        raise ValueError('This diagnostic requires the frozen 320-input, 1024-default configuration')
    selected = sorted(ids, key=lambda rid: (
        hashlib.sha256((PREFIX+rid).encode('utf-8')).hexdigest(), rid))[:40]
    overrides = {rid: 128 for rid in selected}
    metadata = dict(kind='SYNTHETIC_BUDGET_MIXTURE_DEVELOPMENT_OPPORTUNITY_DIAGNOSTIC',
        selection='First 40 request IDs sorted by (sha256(UTF-8(prefix + request_id)), request_id).',
        prefix=PREFIX, short_request_count=40, short_max_tokens=128,
        remaining_request_count=280, default_max_tokens=1024, natural_eos_allowed=True,
        semantics='Assigned maximum output budgets, not future EOS labels. No execution outcomes used. Not a performance-gain comparison with the uniform-budget workload.')
    if check:
        if config.get('output_tokens_by_request') != overrides or config.get('budget_mixture') != metadata:
            raise ValueError('Budget allocation differs from the fixed hash rule')
    else:
        config.update(output_tokens_by_request=overrides, budget_mixture=metadata)
        path.write_text(json.dumps(config, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    positions = [i for i, rid in enumerate(ids) if rid in overrides]
    lengths = [len(workload['actual_prompt_token_ids'][i]) for i in positions]
    arrivals = [workload['arrival_traces_s']['steady'][i] for i in positions]
    describe = lambda values: dict(min=min(values), median=statistics.median(values), max=max(values))
    return dict(short_requests=40, remaining_requests=280,
        prompt_tokens=describe(lengths), mean_prompt_tokens=statistics.mean(lengths),
        zero_based_arrival_positions=describe(positions), arrival_s=describe(arrivals),
        counts_by_64_request_arrival_phase=[sum(i//64 == phase for i in positions) for phase in range(5)],
        source_pools=dict(Counter(workload['source_requests'][i]['source_pool'] for i in positions)),
        workload_file_sha256=hashlib.sha256(workload_bytes).hexdigest(),
        config_file_sha256=hashlib.sha256(path.read_bytes()).hexdigest())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    print(json.dumps(build(Path(__file__).resolve().parent, args.check), indent=2))
