"""Five save-on cells: eager minus current, with separate realized diagnostics."""
import argparse
import copy
import json
from pathlib import Path

from saved_kv_analysis_base import cell


NAMES = ('diagnostic-eager', 'block0-current', 'block0-eager',
         'block1-eager', 'block1-current')
METRICS = ('full_wall_s', 'request_rate_s', 'total_output_tokens', 'output_token_rate_s',
           'mean_completion_s', 'mean_ttft_s', 'max_engine_return_gap_s')


def normalize_config(cfg):
    value = copy.deepcopy(cfg)
    value['rotation_config'].pop('min_steps_between_swaps')
    return value


def analyze(root):
    cells, comparisons = {}, []
    for name in NAMES:
        try:
            cells[name] = cell(root/name, name.startswith('diagnostic'))
        except Exception as exc:
            cells[name] = dict(cell=name, status='ANALYSIS_FAILED', comparable=False, errors=[repr(exc)])
    for block in (0, 1):
        current, eager = (cells[f'block{block}-{arm}'] for arm in ('current', 'eager'))
        row = dict(block=block, status='NOT_COMPARABLE')
        if current['comparable'] and eager['comparable']:
            ca, cb = current['config'], eager['config']
            if ca['selective_save'] != 'on' or cb['selective_save'] != 'on' or \
                    ca['rotation_config']['min_steps_between_swaps'] != 20 or \
                    cb['rotation_config']['min_steps_between_swaps'] != 0 or \
                    normalize_config(ca) != normalize_config(cb):
                row['reason'] = 'Frozen save-on/current/eager configuration mismatch'
                comparisons.append(row)
                continue
            a, b = current['requests'], eager['requests']
            identity = lambda x: sorted((r['request_id'], r['document_id'], r['prompt_sha256'],
                r['arrival_s'], r['max_output']) for r in x['requests'])
            if identity(a) != identity(b):
                row['reason'] = 'Request/input/arrival/max-output identity mismatch'
            else:
                row.update(status='COMPLETE', eager_minus_current={k:
                    b[k]-a[k] if a[k] is not None and b[k] is not None else None for k in METRICS},
                    eager_relative_to_current_percent={k:
                    100*(b[k]/a[k]-1) if a[k] and b[k] is not None else None for k in METRICS})
                by_id = {r['request_id']: r for r in a['requests']}
                row['request_deltas'] = [dict(request_id=r['request_id'],
                    output_identical=r['output_sha256'] == by_id[r['request_id']]['output_sha256'],
                    completion_latency_eager_minus_current_s=r['completion_latency_s']-by_id[r['request_id']]['completion_latency_s'],
                    max_gap_eager_minus_current_s=r['max_engine_return_gap_s']-by_id[r['request_id']]['max_engine_return_gap_s'],
                    ttft_eager_minus_current_s=r['ttft_s']-by_id[r['request_id']]['ttft_s']) for r in b['requests']]
        comparisons.append(row)
    qualification = root/'diagnostic-eager/action-qualification.json'
    return dict(cells=cells, performance_comparisons=comparisons,
        action_qualification=json.loads(qualification.read_text()) if qualification.exists() else dict(status='UNRUN'),
        primary_objective='Reduce long generation gaps (max engine-return output gap, per-request distribution).',
        necessary_costs=['Full-cohort makespan/output rate', 'Mean request completion', 'TTFT as secondary cost'],
        semantics=[
            'Both arms save KV; eager/current denote global cooldown 0/20 only.',
            'The two contemporaneous balanced pairs are retained separately; no best-repeat selection.',
            'Diagnostic timings do not enter primary performance comparisons.',
            'L/F are returned-new-output times. E is observed resource funding, not policy dispatch readiness.',
            'S is host load submission or recovery-bearing engine-call entry, not physical DMA/kernel start.',
            'A host engine call can enclose preemption/E; do not read negative sub-call waiting causally.',
            'Completion/throughput include policy, saving/loading, recomputation and holds in the engine path.',
            'CPU KV capacity, valid cached entries, RSS/HWM and parent cgroup are overlapping memory views.',
            'This fixed-length closed cohort tests one residual policy gate; no general SLO or quality claim.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    with args.output.open('x') as f:
        json.dump(analyze(args.results), f, indent=2, ensure_ascii=False)
        f.write('\n')
