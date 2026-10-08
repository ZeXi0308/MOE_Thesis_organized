"""Summarize only the two preselected historical D fixed2048 arms, CPU-only."""
from collections import Counter
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
D = ROOT.parent / 'D_prefill_budget_20261004'
SLO = {'ttft_s': 4, 'gap_s': 0.1, 'completion_s': 20}
sources = {}
sys.dont_write_bytecode = True


def read(path):
    data = path.read_bytes()
    sources[str(path.relative_to(ROOT.parent))] = hashlib.sha256(data).hexdigest()
    return json.loads(data)


def quantile(values, p):
    values = sorted(values)
    i = (len(values) - 1) * p
    lo, hi = math.floor(i), math.ceil(i)
    return values[lo] + (values[hi] - values[lo]) * (i - lo)


def distribution(values):
    assert values and all(math.isfinite(v) for v in values)
    return {'count': len(values), 'min': min(values),
            'p50': quantile(values, 0.5), 'p95': quantile(values, 0.95),
            'p99': quantile(values, 0.99), 'max': max(values),
            'mean': sum(values) / len(values)}


env = read(D / 'fixed-normal-r01/environment.json')
prior = read(ROOT / 'evidence/historical_feasibility.json')
spec = importlib.util.spec_from_file_location('d_historical_analyzer', D / 'analyze.py')
analyzer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analyzer)
sources['D_prefill_budget_20261004/analyze.py'] = hashlib.sha256((D / 'analyze.py').read_bytes()).hexdigest()
result = {
    'status': 'HISTORICAL_FEASIBILITY_BACKGROUND_ONLY_NOT_G_PRIMARY_COMPARISON',
    'original_gpu_uuid': env['gpu'].strip().splitlines()[1].split(', ')[1],
    'current_gpu_distinction': prior['current_gpu_distinction'],
    'scope': 'Exactly one preselected fixed2048 arm per low/high D workload; no alternative arms or traces selected.',
    'frozen_slo': SLO,
    'metric_definitions': {
        'goodput_requests_per_s': 'Fully completed fixed-length requests meeting all three SLOs, divided by raw.elapsed_s.',
        'complete_request_throughput_per_s': 'All fully completed fixed-length requests divided by raw.elapsed_s.',
        'denominator': 'Episode origin before first admission through drained-loop end; includes arrival waiting and final drain. Both first arrivals are zero.',
        'ttft_s': 'First observed output timestamp minus planned arrival_s.',
        'completion_latency_s': 'completion_s minus planned arrival_s.',
        'request_max_output_gap_s': 'Maximum adjacent output timestamp difference within each request; first-token waiting is measured separately by TTFT.',
        'pooled_adjacent_output_gap_s': 'All adjacent token gaps pooled; token-weighted, distinct from request-level max-gap distribution.',
        'percentiles': 'Linear interpolation at (n-1)*p, as in existing D analyzer; no rounding before SLO comparisons.',
        'batch_distributions': 'One unweighted observation per recorded engine.step; running_count is after scheduling. Scheduled tokens = prefill_tokens + decode_tokens.',
    },
    'limitations': [
        'D fixed2048 modifies aggregate prefill budget; this is not an unmodified G scheduler baseline.',
        'Measurements were recorded on the original D GPU UUID, not the current endpoint GPU.',
        'One historical arm per load has no repeated-run uncertainty estimate and cannot establish a strongest simple baseline or any G method gain.',
        'Host engine.step output timestamps include scheduler/output handling, exclude network/client delivery, and use fixed output lengths with ignore_eos.',
    ],
    'workloads': {},
}
for label in ('low', 'high'):
    raw_path = D / f'{label}-normal-r01/00_fixed2048/raw.json'
    protocol_path = D / f'{label}-normal-r01/protocol.json'
    raw, protocol = read(raw_path), read(protocol_path)
    work_path = ROOT / f'inputs/workload_{label}.json'
    work = read(work_path)
    assert sources[str(work_path.relative_to(ROOT.parent))] == protocol['workload_sha256']
    assert protocol['slo'] == SLO and raw['policy'] == 'fixed2048'
    assert len(work) == len(raw['requests']) == protocol['request_count']
    assert min(x['arrival_s'] for x in work) == 0
    source_by_id = {x['request_id']: x for x in work}
    ttfts, completions, max_gaps, all_gaps = [], [], [], []
    passed, completed, output_tokens = 0, 0, 0
    failures = Counter({'ttft_s': 0, 'gap_s': 0, 'completion_s': 0})
    for request in raw['requests']:
        source = source_by_id[request['request_id']]
        assert all(request[k] == source[k] for k in ('arrival_s', 'max_tokens', 'prompt_token_ids'))
        times = request['token_times_s']
        assert len(times) == len(request['output_token_ids']) == request['max_tokens']
        assert times and all(a <= b for a, b in zip(times, times[1:]))
        assert request['finished'] and request['completion_s'] == times[-1]
        assert 0 <= request['arrival_s'] <= times[0] <= times[-1] <= raw['elapsed_s']
        gaps = [b-a for a, b in zip(times, times[1:])]
        ttft = times[0] - request['arrival_s']
        completion = request['completion_s'] - request['arrival_s']
        gap = max(gaps, default=0)
        values = {'ttft_s': ttft, 'completion_s': completion, 'gap_s': gap}
        for key, value in values.items():
            failures[key] += value > SLO[key]
        passed += all(value <= SLO[key] for key, value in values.items())
        completed += 1
        output_tokens += len(times)
        ttfts.append(ttft); completions.append(completion); max_gaps.append(gap)
        all_gaps.extend(gaps)
    # Independent agreement with the existing source analyzer's request decisions.
    check = analyzer.summarize(raw_path, protocol_path, '00_fixed2048')
    assert not check['validation_errors'], check['validation_errors']
    assert passed == sum(x['joint_slo_pass'] for x in check['requests'])
    steps = raw['steps']
    result['workloads'][label] = {
        'source_raw': str(raw_path.relative_to(ROOT.parent)),
        'historical_policy': raw['policy'], 'requests': len(work),
        'completed_requests': completed, 'completion_rate': completed / len(work),
        'joint_slo_pass_requests': passed, 'joint_slo_pass_fraction': passed / len(work),
        'slo_failure_request_counts_nonexclusive': dict(failures),
        'denominator_elapsed_s': raw['elapsed_s'],
        'last_request_completion_s': max(x['completion_s'] for x in raw['requests']),
        'goodput_requests_per_s': passed / raw['elapsed_s'],
        'complete_request_throughput_per_s': completed / raw['elapsed_s'],
        'complete_output_tokens': output_tokens,
        'complete_output_tokens_per_s': output_tokens / raw['elapsed_s'],
        'ttft_s': distribution(ttfts),
        'completion_latency_s': distribution(completions),
        'request_max_output_gap_s': distribution(max_gaps),
        'pooled_adjacent_output_gap_s': distribution(all_gaps),
        'batch': {
            'running_requests_after_schedule': distribution([x['running_count'] for x in steps]),
            'scheduled_requests': distribution([len(x['requests']) for x in steps]),
            'scheduled_tokens': distribution([x['prefill_tokens'] + x['decode_tokens'] for x in steps]),
            'prefill_tokens': distribution([x['prefill_tokens'] for x in steps]),
            'decode_tokens': distribution([x['decode_tokens'] for x in steps]),
            'decode_requests_before_schedule': distribution([x['decode_count_before'] for x in steps]),
        },
        'existing_D_analyzer_validation_errors': 0,
        'existing_D_analyzer_joint_slo_decisions_match': True,
    }
result['source_sha256'] = sources
target = ROOT / 'evidence/historical_service_metrics.json'
target.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
print(json.dumps({label: {key: row[key] for key in ('requests', 'joint_slo_pass_requests', 'goodput_requests_per_s', 'complete_request_throughput_per_s')} for label, row in result['workloads'].items()}, indent=2))
