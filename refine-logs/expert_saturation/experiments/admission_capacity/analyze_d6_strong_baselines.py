#!/usr/bin/env python3
"""Describe the frozen d6 four-arm campaign, preserving every terminal status."""
import argparse
import hashlib
import json
from pathlib import Path
from analyze_pressure_review import cell, read


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--bundle', type=Path, required=True)
    ap.add_argument('--results', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    protocol = read(args.bundle / 'STATUS.json')
    rows, per_request, envs = [], {}, []
    for spec in protocol['cells']:
        label = spec['label']
        row = cell(args.results, label)
        row.update(role=spec['role'], block=spec['block'])
        rows.append(row)
        if not row['eligible']:
            continue
        path = args.results / label
        cfg, env = read(path/'config.json'), read(path/'environment.json')
        qual = read(path/'safe-cap-qualification.json')
        assert row['kv_bytes'] == 13960740864 and row['usable_blocks'] == 6656
        assert cfg['completion_policy'] == spec['policy']
        assert cfg['rotation_victim_order'] == spec['victim_order']
        assert not qual['prefix_caching'] and not env['gpu_before']['compute_processes'].strip()
        for name, digest in env['source_sha256'].items():
            assert digest == protocol['files'][name], (label, name)
        envs.append(env)
        raw, met = read(path/'raw.json'), read(path/'metrics.json')
        metrics_by_id = {r['request_id']: r for r in met['per_request']}
        requests = {}
        for r in raw['requests']:
            rid = r['request_id']
            assert rid not in requests
            assert len(r['output_token_ids']) == 1024
            requests[rid] = dict(completion_s=metrics_by_id[rid]['request_latency_s'],
                output_sha256=hashlib.sha256(json.dumps(r['output_token_ids']).encode()).hexdigest())
        assert len(requests) == 32
        per_request[label] = requests
        row['configuration_check'] = 'PASS'
        row['workload_sha256'] = cfg['workload_sha256']
    eligible = [r for r in rows if r['eligible']]
    if eligible:
        assert len({r['workload_sha256'] for r in eligible}) == 1
        assert all(e['vllm_source_sha256'] == envs[0]['vllm_source_sha256'] for e in envs)
    index = {r['label']: r for r in rows}
    pairs = []
    for b in (0, 1):
        for base, target in [('native','least_progress'),('least_progress','headroom'),
                             ('least_progress','most_output')]:
            n, r = [index[f'block{b}-d6-{a}'] for a in (base,target)]
            if not (n['eligible'] and r['eligible']):
                continue
            nr, rr = per_request[n['label']], per_request[r['label']]
            assert nr.keys() == rr.keys()
            differences = {rid:rr[rid]['completion_s']-nr[rid]['completion_s'] for rid in nr}
            pairs.append(dict(block=b, baseline=base, target=target,
                mean_completion_delta_pct=100*(r['mean_completion_s']/n['mean_completion_s']-1),
                throughput_delta_pct=100*(r['throughput_rps']/n['throughput_rps']-1),
                max_itl_delta_s=r['max_itl_s']-n['max_itl_s'],
                slower_requests=sum(v>0 for v in differences.values()),
                per_request_completion_delta_s=differences,
                equal_output_sequences=sum(nr[k]['output_sha256']==rr[k]['output_sha256'] for k in nr)))
    result = dict(status='MEASUREMENT_ONLY', cells=rows, pairs=pairs,
                  scope='Same old workload, two reversed-order blocks, APC off. No significance, '
                        'noise bound, quality, Oracle or method GO. Mixed calls retain useful outputs.')
    with args.output.open('x') as f:
        json.dump(result,f,indent=2,allow_nan=False)
        f.write('\n')


if __name__ == '__main__':
    main()
