#!/usr/bin/env python3
"""Reproduce one fixed burst-plus-late-stream input from the frozen mixed parent."""
import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / 'candidate_native_mixed_budget_probe_r01'
INPUTS = Path('pkg/inputs/pro_high')


def read(path):
    return json.loads(path.read_text())


def serialized(value):
    return json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n'


def build(source, check):
    config, workload, stats = [read(source / INPUTS / name)
                              for name in ('config.json', 'workload.json', 'stats.json')]
    ids = [row['request_id'] for row in workload['source_requests']]
    if (len(ids) != 320 or len(set(ids)) != 320
            or workload['arrival_traces_s'] != {'steady': [i / 100 for i in range(320)]}
            or config['arrival_gap_s'] != .01 or config['output_tokens'] != 1024
            or len(config['output_tokens_by_request']) != 40
            or set(config['output_tokens_by_request'].values()) != {128}):
        raise ValueError('Source must be the frozen 320-request, 40/280 budget-mixture 0.01s input')
    metadata = dict(kind='SYNTHETIC_TEMPORAL_OVERLAP_DIAGNOSTIC',
        authoritative_trace='workload.arrival_traces_s.steady',
        initial_burst_requests=256, initial_burst_gap_s=.01, initial_burst_last_arrival_s=2.55,
        late_stream_requests=64, late_stream_start_s=30., late_stream_gap_s=.2,
        late_stream_last_arrival_s=42.6,
        rule='Preserve source arrivals i/100 for indices 0..255; use 30+(i-256)*0.2 for indices 256..319.',
        split_basis='Existing 256-request knee prefix; final 64-request length-stratum phase.',
        design_basis='Development design informed by prior aggregate pressure timing; no per-request future outcomes used.',
        semantics='Only external arrival times change. Same IDs/order/prompts and assigned output budgets; '
                  'not an independent holdout or policy-gain comparison. Ongoing-arrival pressure is not guaranteed.')
    workload['arrival_traces_s']['steady'] = workload['arrival_traces_s']['steady'][:256] + [
        30 + (i - 256) * .2 for i in range(256, 320)]
    workload['arrival_rule'] = metadata['rule']
    workload['temporal_overlap_diagnostic'] = metadata
    identity = hashlib.sha256(json.dumps(workload, sort_keys=True).encode()).hexdigest()
    config.update(arrival_gap_s=None, arrival_span_s=42.6, workload_sha256=identity,
        arrival_process='deterministic_length_stratified_initial_burst_then_late_stream',
        temporal_overlap_diagnostic=metadata)
    stats.update(arrival_span_s=42.6, workload_sha256=identity,
        no_outcome_information_used=False, no_per_request_future_outcomes_used=True,
        temporal_overlap_diagnostic=metadata)
    payloads = {str(INPUTS / name): serialized(value).encode() for name, value in zip(
        ('config.json', 'workload.json', 'stats.json'), (config, workload, stats))}
    manifest = read(source / 'manifest.json')
    manifest.update({name: hashlib.sha256(value).hexdigest() for name, value in payloads.items()})
    payloads['manifest.json'] = serialized(manifest).encode()
    for name, payload in payloads.items():
        path = ROOT / name
        if check:
            if path.read_bytes() != payload:
                raise ValueError(f'Overlap input differs from deterministic builder: {name}')
        else:
            path.write_bytes(payload)
    short = config['output_tokens_by_request']
    counts = [sum(rid in short for rid in ids[start:end]) for start, end in ((0, 256), (256, 320))]
    print(json.dumps(dict(status='CHECKED' if check else 'BUILT', requests=320,
        initial_short_long=[counts[0], 256-counts[0]], late_short_long=[counts[1], 64-counts[1]],
        arrival_span_s=42.6, workload_sha256=identity,
        manifest_sha256=hashlib.sha256(payloads['manifest.json']).hexdigest(),
        semantics=metadata['semantics']), ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    build(args.source.resolve(), args.check)
