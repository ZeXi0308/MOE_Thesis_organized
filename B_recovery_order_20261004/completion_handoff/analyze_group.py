#!/usr/bin/env python3
"""Reuse all-request analysis and attach literal completion-finalizer counts."""
import argparse, collections, hashlib, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PARENT_SHA = 'bab5488cdbc0d586645c5ba905c9d4661ea3e36fd5083613acc80825895f089c'


def analyze_group(session):
    path = ROOT.parent/'normal_capacity/analyze_group.py'; payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != PARENT_SHA:
        raise RuntimeError('Normal-capacity analysis source changed')
    text = payload.decode(); old = '(native|age|flush_first)'
    if text.count(old) != 1:
        raise RuntimeError('Analysis mode boundary changed')
    namespace = dict(__name__='completion_handoff_analysis', __file__=str(__file__))
    exec(compile(text.replace(old, '(native|after_sample)'), str(path)+'[completion_handoff]', 'exec'), namespace)
    result = namespace['analyze_group'](session); by_path = {c['directory']: c for c in result['cells']}
    for cell in result['cells']:
        artifact = Path(cell['directory'])/'completion-finalize.json'; data = namespace['optional'](artifact)
        cell['completion_finalize'] = dict(source=str(artifact), mode=data.get('mode'), status=data.get('status'),
            event_count=len(data['events']) if 'events' in data else None,
            deferred_count=sum(bool(e.get('deferred')) for e in data['events']) if 'events' in data else None,
            finalize_reasons=dict(collections.Counter(e.get('finalize_reason', 'missing') for e in data.get('events', []))),
            extra_cuda_queries=data.get('extra_cuda_queries'), extra_cuda_synchronizations=data.get('extra_cuda_synchronizations'))
    for index, contrast in enumerate(result['comparisons']):
        pair = [by_path.get(contrast[key]) for key in ('candidate', 'native')]
        def fixed_budget(cell):
            resources = cell['resource_observations'] if cell else {}
            return (cell and cell['actual_cap'] == 256
                and resources.get('memory-after-init', {}).get('kv_storage_bytes') == 77242302464
                and (resources.get('host-before', {}).get('cpu_kv') or {}).get('unique_storage_bytes') == 17179869184)
        if not all(fixed_budget(cell) for cell in pair):
            result['comparisons'][index] = dict(candidate=contrast['candidate'], native=contrast['native'],
                status='UNAVAILABLE', reason='Actual cap256 / GPU KV 77242302464B / Host KV 16GiB unverified or mismatched')
    result['semantics'] += ' Completion counts are literal observations; deferral or copy elapsed is not output gain. Same 5s/.2s exploratory joint SLO; all incomplete requests retained.'
    return result


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--session', type=Path, required=True)
    parser.add_argument('--output', type=Path); args = parser.parse_args()
    destination = args.output or args.session/'completion-metrics.json'
    if destination.exists():
        raise FileExistsError(destination)
    result = analyze_group(args.session)
    with destination.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(destination)


if __name__ == '__main__':
    main()
