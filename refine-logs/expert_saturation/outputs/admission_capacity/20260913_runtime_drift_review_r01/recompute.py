"""Read-only reanalysis of existing negative controls; no noise-bound estimator."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'AGENTS.md').exists())
SOURCES = {
    'waiting': ROOT / 'refine-logs/independent_ideas_20260910/waiting_bypass_limit_r01',
    'prefill': ROOT / 'refine-logs/independent_ideas_20260911/per_request_prefill_share_r01',
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def analyze():
    result = {'evidence': 'CPU_REANALYSIS_OF_NATIVE_REQUEST_MEASUREMENTS',
              'scope': 'Descriptive observed control drift, not independent null draws, '
                       'a population noise bound, significance, or cross-workload calibration.',
              'campaigns': {}}
    for name, root in SOURCES.items():
        source = root / 'analysis/summary.json'
        summary = json.loads(source.read_text())
        controls = [p for p in summary['comparisons'] if p['cohort'] == 'all_short']
        assert len(controls) == 6 and all(p['valid'] for p in controls)
        metrics = controls[0]['metrics']['all']
        vectors = {m: [p['metrics']['all'][m]['delta_pct'] for p in controls] for m in metrics}
        entry = {
            'source': str(source.relative_to(ROOT)), 'source_sha256': sha(source),
            'control_pairs': controls,
            'delta_pct_vectors': vectors,
            'observed_max_abs_delta_pct': {
                m: max(abs(v) for v in values if v is not None)
                if any(v is not None for v in values) else None
                for m, values in vectors.items()},
            'formal_all_pass_goodput_equals_throughput': all(
                c['groups']['all']['n_slo_pass'] == c['groups']['all']['n_completed'] == 16
                and c['groups']['all']['goodput_rps'] == c['groups']['all']['throughput_rps']
                for c in summary['cells']),
            'sample_structure': 'Two blocks; three measured arms and three shared-arm pairwise '
                                'differences per block. Six differences are not six independent pairs.'}
        if name == 'waiting':
            traces = {}
            entry['raw_sha256'] = {}
            for cell in summary['cells']:
                if cell['plan']['cohort'] != 'all_short':
                    continue
                path = Path(cell['source'])
                raw = json.loads(path.read_text())
                ids = raw['internal_to_source']
                traces[(cell['block'], cell['plan']['waiting_order'])] = [
                    [(ids.get(r['request_id'], r['request_id']), r['scheduled_tokens'],
                      r['prefill_tokens'], r['decode_tokens']) for r in step['scheduled']]
                    for step in raw['scheduler_steps']]
                entry['raw_sha256'][str(path.relative_to(ROOT))] = sha(path)
            entry['full_schedule_comparisons'] = []
            for pair in controls:
                a, b = [traces[(pair['block'], pair[k])] for k in ('baseline', 'intervention')]
                entry['full_schedule_comparisons'].append({
                    'block': pair['block'], 'baseline': pair['baseline'],
                    'intervention': pair['intervention'], 'step_counts': [len(a), len(b)],
                    'equal_per_step_request_token_allocations': a == b,
                    'first_different_step_index': next((i for i, (x, y) in enumerate(zip(a, b))
                                                       if x != y), min(len(a), len(b)) if len(a) != len(b) else None)})
            entry['mixed_bounded_vs_spt'] = [p for p in summary['comparisons']
                if p['cohort'] == 'mixed' and p['baseline'] == 'short_prompt_first'
                and p['intervention'] == 'bounded_bypass_once']
            entry['mixed_reverse_bounded_vs_fcfs_max_itl'] = next(
                p['metrics']['all']['max_itl_max_s'] for p in summary['comparisons']
                if p['cohort'] == 'mixed' and p['block'] == 'reverse'
                and p['baseline'] == 'fcfs' and p['intervention'] == 'bounded_bypass_once')
        result['campaigns'][name] = entry
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name('observations.json'))
    output = parser.parse_args().output
    with output.open('x') as handle:
        json.dump(analyze(), handle, indent=2, allow_nan=False)
        handle.write('\n')
    print(output)
