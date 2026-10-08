"""CPU-only extraction of existing D evidence; never loads a model or uses GPU."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import shutil

ROOT = Path(__file__).resolve().parent
D = ROOT.parent / 'D_prefill_budget_20261004'
sources = {}


def read(path):
    data = path.read_bytes()
    sources[str(path.relative_to(ROOT.parent))] = hashlib.sha256(data).hexdigest()
    return data.decode()


log = read(D / 'normal-r01-controller.log')
env = json.loads(read(D / 'fixed-normal-r01/environment.json'))
engine = json.loads(read(D / 'fixed-normal-r01/engine_args.json'))
gpu = env['gpu'].strip().splitlines()[1].split(', ')
count = re.search(r'Profiling CUDA graph memory: PIECEWISE=(\d+) \(largest=(\d+)\), FULL=(\d+) \(largest=(\d+)\)', log)
memory = re.search(r'CUDA graph pool memory: ([\d.]+) GiB \(actual\), ([\d.]+) GiB \(estimated\)', log)
capture = re.search(r'Graph capturing finished in (\d+) secs, took ([\d.]+) GiB', log)
assert count and memory and capture
evidence = {
    'evidence_status': 'HISTORICAL_D_SINGLE_ENGINE_NOT_CURRENT_G_EXPERIMENT',
    'historical_environment': dict(env, parsed_gpu_name=gpu[0], original_gpu_uuid=gpu[1]),
    'historical_engine_args': engine,
    'current_gpu_distinction': {
        'current_endpoint_gpu_uuid': 'GPU-51b8e4bb-27b8-4b82-5254-7317aae7298c',
        'source': 'E_restore_choice_20261004/checkpoint.json, local 2026-10-08 18:49 record',
        'live_verified_by_this_extractor': False,
        'warning': 'D measurements belong to original_gpu_uuid, not current_endpoint_gpu_uuid.'
    },
    'graph_pool': {
        'piecewise_count': int(count[1]), 'piecewise_largest_tokens': int(count[2]),
        'full_count': int(count[3]), 'full_largest_tokens': int(count[4]),
        'actual_gib_log_rounded': float(memory[1]),
        'estimated_gib_log_rounded': float(memory[2]),
        'capture_seconds_log_rounded': int(capture[1]),
        'scope': 'Combined graph pool as reported by native vLLM log; not sum of individual peaks.'
    },
    'workloads': {},
}
inputs = ROOT / 'inputs'
inputs.mkdir(exist_ok=True)
for label in ('low', 'high'):
    path = D / f'workload_{label}.json'
    work = json.loads(read(path))
    protocol = json.loads(read(D / f'{label}-normal-r01/protocol.json'))
    raw = json.loads(read(D / f'{label}-normal-r01/00_fixed2048/raw.json'))
    assert sources[str(path.relative_to(ROOT.parent))] == protocol['workload_sha256']
    assert len(work) == len(raw['requests']) == protocol['request_count']
    assert all(x['finished'] and len(x['output_token_ids']) == x['max_tokens'] for x in raw['requests'])
    totals = {s['kv_total_blocks'] for s in raw['steps']}
    assert len(totals) == 1
    total = totals.pop()
    peak = max(s['kv_used_blocks'] for s in raw['steps'])
    evidence['workloads'][label] = {
        'input_source': str(path.relative_to(ROOT.parent)),
        'raw_source': str((D / f'{label}-normal-r01/00_fixed2048/raw.json').relative_to(ROOT.parent)),
        'historical_policy': raw['policy'], 'request_count': len(work),
        'arrival_first_s': min(r['arrival_s'] for r in work),
        'arrival_last_s': max(r['arrival_s'] for r in work),
        'prompt_tokens': sum(len(r['prompt_token_ids']) for r in work),
        'prompt_length_min': min(len(r['prompt_token_ids']) for r in work),
        'prompt_length_max': max(len(r['prompt_token_ids']) for r in work),
        'fixed_output_tokens': sum(r['max_tokens'] for r in work),
        'fixed_output_length_counts': dict(sorted(Counter(r['max_tokens'] for r in work).items())),
        'historical_elapsed_s': raw['elapsed_s'],
        'native_kv_pool_blocks_including_reserved': env['kv_blocks'],
        'usable_kv_blocks': total, 'peak_used_kv_blocks': peak,
        'peak_fraction_of_usable_blocks': peak / total,
        'headroom_blocks_at_peak': total - peak,
        'headroom_gib_at_peak': (total - peak) * 2 / 1024,
        'kv_bytes_per_block': 2 * 1024**2,
        'preemption_events': sum(len(s['preempted']) for s in raw['steps']),
        'slo': protocol['slo'],
        'output_contract': protocol['output_contract'],
        'timing': protocol['timing'],
    }
    target = inputs / path.name
    if target.exists():
        assert target.read_bytes() == path.read_bytes(), f'Refuse overwrite: {target}'
    else:
        shutil.copyfile(path, target)
evidence['frozen_slo_for_G'] = dict(evidence['workloads']['low']['slo'])
assert evidence['frozen_slo_for_G'] == evidence['workloads']['high']['slo']
evidence['high_pressure_interpretation'] = (
    'Existing normal-memory high arrival-pressure workload, not fabricated for G. '
    'It approaches capacity but never preempts in the recorded fixed2048 arm. '
    'Its 9.244 GiB KV headroom greatly exceeds the entire 0.17 GiB rounded graph pool; '
    'therefore it is reasonable serving pressure but not demonstrated graph-induced KV pressure. '
    'Do not rescale arrivals, increase requests, pad memory or lower KV quota to create a gain.'
)
evidence['limitations'] = [
    'Logs round GiB; cannot infer exact graph bytes or use these numbers as a new measured portfolio cost.',
    'The fixed2048 D scheduler intervention differs from an unmodified scheduler; historical traces only provide feasibility evidence.',
    'Host-step SLO is an experimental contract, not a production or HTTP-network SLA.',
]
evidence['source_sha256'] = sources
(ROOT / 'evidence').mkdir(exist_ok=True)
(ROOT / 'evidence/historical_feasibility.json').write_text(json.dumps(evidence, indent=2, ensure_ascii=False) + '\n')
print(json.dumps({'output': 'evidence/historical_feasibility.json', 'workloads': {k: {'requests': v['request_count'], 'peak_kv_blocks': v['peak_used_kv_blocks']} for k, v in evidence['workloads'].items()}}, indent=2))
