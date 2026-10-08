#!/usr/bin/env python3
"""Verify two fixed reverse-order replications and describe all four executions."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys

sys.dont_write_bytecode = True
import docqa128_integrity_analyze_v1 as native_check
import docqa_concurrency64_integrity_v1 as capped_check

BASE = Path(__file__).resolve().parent
def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p, value):
    with p.open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2); f.write('\n')


def analyze():
    confirmation = BASE / 'docqa_concurrency_confirmation_raw_v1_20261002'
    receipt = read(confirmation / 'docqa-concurrency-confirmation-v1.json')
    protocol = read(BASE / 'docqa_concurrency_confirmation_protocol_v1.json')
    assert receipt['status'] == 'COMPLETE' and receipt['order'] == protocol['order'] == ['concurrency64', 'native128']
    assert receipt['driver_sha256'] == protocol['driver_sha256'] == sha(BASE / 'docqa_concurrency_confirmation_v1.py')
    assert receipt['protocol_sha256'] == sha(BASE / 'docqa_concurrency_confirmation_protocol_v1.json')
    assert len(receipt['units']) == 2
    integrity = {}
    for unit, module in zip(receipt['units'], (capped_check, native_check)):
        assert unit['status'] == 'COMPLETE' and unit['exit_code'] == 0
        assert unit['output_name'] in ('qwen7b-docqa-concurrency64-confirm-v1', 'qwen7b-native-docqa128-confirm-v1')
        # The frozen wrapper explicitly changes only the output directory.
        original = module.RUN_NAME
        try:
            assert original == unit['original_output_name']
            module.RUN_NAME = unit['output_name']
            checked = module.analyze(confirmation / unit['output_name'])
            assert checked['validity'] == 'COMPLETE' and checked['structural_gate'], checked['issues']
            integrity[unit['label']] = checked
        finally:
            module.RUN_NAME = original
    roots = {
        'native128_first': BASE / 'docqa128_raw_v1_20261002/qwen7b-native-docqa128-v1',
        'concurrency64_first': BASE / 'docqa_concurrency64_raw_v1_20261002/qwen7b-docqa-concurrency64-v1',
        'concurrency64_repeat': confirmation / 'qwen7b-docqa-concurrency64-confirm-v1',
        'native128_repeat': confirmation / 'qwen7b-native-docqa128-confirm-v1',
    }
    qualities = {'native128': read(BASE / 'docqa128_quality_v2.json'),
                 'concurrency64': read(BASE / 'docqa_concurrency64_quality_v1.json')}
    originals, runs, configs = {}, {}, {}
    for name, root in roots.items():
        arm = name.rsplit('_', 1)[0]
        source = root / 'native/measured-outputs.json'; rows = read(source)
        quality = qualities[arm]; labels = {r['request_id']: r for r in quality['per_request']}
        assert quality['qualification'] == 'PASS_DEVELOPMENT_SCREEN'
        assert len(rows) == len(labels) == len({r['request_id'] for r in rows}) == 128
        configs[name] = read(root / 'native/config.json')
        if name.endswith('_first'):
            assert quality['raw_measured_outputs_sha256'] == sha(source)
            originals[arm] = {r['request_id']: r for r in rows}
        else:
            for r in rows:
                old = originals[arm][r['request_id']]
                assert all(old[k] == r[k] for k in ('article_text', 'questions', 'prompt_token_ids',
                    'output_text', 'output_token_ids')), 'Repeat output changed; fresh quality review required'
        status = read(root / 'native/status.json'); per = []
        assert status['status'] == 'COMPLETE'
        for r in rows:
            j = labels[r['request_id']]
            assert hashlib.sha256(r['output_text'].encode()).hexdigest() == j['response_sha256']
            assert hashlib.sha256(r['article_text'].encode()).hexdigest() == j['article_sha256']
            returns = [e['return_s'] for e in r['host_returns'] if e['delta_token_ids']]
            gap = max((b-a for a,b in zip(returns, returns[1:])), default=0)
            per.append(dict(request_id=r['request_id'], ttft_s=r['token_times_s'][0]-r['arrival_s'],
                flow_s=r['host_elapsed_s']-r['arrival_s'], maximum_host_return_gap_s=gap,
                output_tokens=len(r['output_token_ids']), quality_success=j['request_success'],
                conservative_quality_success=j['request_success'] and not bool(j.get('uncertainty'))))
        duration = status['observation_end_s']
        count = lambda quality, ttft=20, gap=4: sum(r[quality] and r['ttft_s'] <= ttft and
            r['maximum_host_return_gap_s'] <= gap for r in per)
        joint = count('quality_success'); conservative = count('conservative_quality_success')
        curves = [dict(ttft_s=t, gap_s=g, qualified=count('quality_success', t, g),
            goodput=count('quality_success', t, g)/duration) for t in (10, 15, 20, 25, 30)
            for g in (.2, .5, 1., 4.)]
        runs[name] = dict(episode_duration_s=duration, completed=128,
            natural_eos=status['finish_reason_counts'].get('stop'), output_tokens=status['output_tokens'],
            output_tokens_per_s=status['output_tokens']/duration, preemptions=status['preemptions'],
            mean_flow_s=statistics.mean(r['flow_s'] for r in per),
            mean_ttft_s=statistics.mean(r['ttft_s'] for r in per),
            maximum_host_return_gap_s=max(r['maximum_host_return_gap_s'] for r in per),
            quality_successful_requests=sum(r['quality_success'] for r in per),
            joint_qualified=joint, joint_goodput=joint/duration,
            conservative_joint_qualified=conservative, conservative_joint_goodput=conservative/duration,
            raw_outputs_sha256=sha(source), output_identity_to_same_arm_first='EXACT_ALL128',
            threshold_curves_descriptive=curves, per_request=per)
    reference = configs['native128_first']
    for name, config in configs.items():
        assert config['sampling'] == reference['sampling']
        actual, expected = dict(config['engine_args']), dict(reference['engine_args'])
        assert actual.pop('max_num_seqs') == (64 if name.startswith('concurrency64') else 128)
        expected.pop('max_num_seqs'); assert actual == expected
    comparisons = {}
    for pair in ('first', 'repeat'):
        n, c = runs['native128_' + pair], runs['concurrency64_' + pair]
        comparisons[pair] = {k:100*(c[k]/n[k]-1) for k in
            ('episode_duration_s', 'mean_flow_s', 'mean_ttft_s', 'output_tokens_per_s',
             'joint_goodput', 'conservative_joint_goodput')}
    pooled = {}
    for arm in qualities:
        records = [runs[arm + '_' + pair] for pair in ('first', 'repeat')]
        duration = sum(r['episode_duration_s'] for r in records)
        pooled[arm] = dict(joint_goodput=sum(r['joint_qualified'] for r in records)/duration,
            conservative_joint_goodput=sum(r['conservative_joint_qualified'] for r in records)/duration,
            mean_flow_s=statistics.mean(r['mean_flow_s'] for r in records),
            output_tokens_per_s=sum(r['output_tokens'] for r in records)/duration)
    pooled_change = {k:100*(pooled['concurrency64'][k]/pooled['native128'][k]-1) for k in pooled['native128']}
    return dict(schema='c-docqa-concurrency-four-execution-comparison-v1',
        scope='Two executions per configuration of the same development input; descriptive replication only.',
        primary_thresholds_s=dict(ttft=20, maximum_host_return_gap=4),
        runs=runs, paired_change_percent=comparisons, pooled=pooled, pooled_change_percent=pooled_change,
        confirmation_integrity=integrity,
        quality_sources={name:sha(BASE / name) for name in ('docqa128_quality_v2.json', 'docqa_concurrency64_quality_v1.json')},
        limitations=['128 coupled requests are not independent episodes; n=2 executions per configuration is not a publication confirmation.',
            'All measurements include identical diagnostic observer/journal paths; deployment performance is not established.',
            '20s/4s is a historical diagnostic,not a business SLO;4s gap is nonbinding in all four runs.',
            'Natural outputs differ across configurations;output token rate is not equal-work speedup.',
            'Same-family LLM quality judgments and conservative boundary sensitivity are both retained.',
            'Fixed64 is a simple baseline; no algorithmic novelty or CCF-C paper GO follows.'])


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(); result = analyze(); save(args.output, result)
    print(json.dumps(dict(paired_change_percent=result['paired_change_percent'],
                         pooled_change_percent=result['pooled_change_percent'])))
