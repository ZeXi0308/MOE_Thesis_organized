"""Compare independently recorded headroom paths; no cross-run timing claims."""
import argparse
import hashlib
import json
from pathlib import Path


def normalize(value, identities):
    if isinstance(value, dict):
        return {identities.get(k, k): normalize(v, identities) for k, v in value.items()}
    if isinstance(value, list):
        return [normalize(v, identities) for v in value]
    return identities.get(value, value) if isinstance(value, str) else value


def read(cell):
    names = ('raw.json', 'headroom-decisions.json')
    raw, decisions = [json.loads((cell / n).read_text()) for n in names]
    identities = raw['internal_to_source']
    assert len(set(identities.values())) == len(identities), 'nonunique source mapping'
    assert raw['status'] == 'COMPLETE', 'raw run did not complete'
    steps, traces = raw['scheduler_steps'], raw['memory_trace']
    assert len(steps) == len(traces) == len(decisions), 'unaligned ledgers'
    for i, (s, t, d) in enumerate(zip(steps, traces, decisions)):
        assert s['step'] == t['attempted_step'] == d['step'] == i, 'step identity mismatch'
    result = dict(
        decisions=[{k: v for k, v in d.items() if k not in ('observer', 'decision_seconds')}
                   for d in decisions],
        scheduled_intervals=[dict(step=s['step'], scheduled=s['scheduled'],
            preempted=s['preempted_request_ids'], total=s['total_scheduled_tokens'],
            recompute=s['recompute_tokens']) for s in steps],
        memory=[dict(step=t['attempted_step'], before=t['before'], after=t['after']) for t in traces],
        outputs={r['request_id']: r['output_token_ids'] for r in raw['requests']},
        input_identity={r['request_id']: [r['document_id'], r['prompt_token_ids_sha256'],
                                        r['prompt_tokens']] for r in raw['requests']})
    return normalize(result, identities), {n: hashlib.sha256((cell / n).read_bytes()).hexdigest() for n in names}


def sequence_comparison(left, right):
    differences = [i for i in range(max(len(left), len(right)))
                   if i >= min(len(left), len(right)) or left[i] != right[i]]
    return dict(exact=not differences, reference_steps=len(left), candidate_steps=len(right),
                different_steps=len(differences), first_different_step=next(iter(differences), None))


def compare(reference, candidate):
    missing = [str(c / n) for c in (reference, candidate)
               for n in ('raw.json', 'headroom-decisions.json') if not (c / n).exists()]
    if missing:
        return dict(status='UNRUN', missing=missing)
    left, lhs = read(reference)
    right, rhs = read(candidate)
    paths = {key: sequence_comparison(left[key], right[key])
             for key in ('decisions', 'scheduled_intervals', 'memory')}
    starts = [next((i for i, d in enumerate(r['decisions']) if d['active']), None) for r in (left, right)]
    active_memory = (sequence_comparison(left['memory'][starts[0]:], right['memory'][starts[1]:])
                     if all(s is not None for s in starts) else None)
    shared = sorted(left['outputs'].keys() & right['outputs'].keys())
    mismatched = [rid for rid in shared if left['outputs'][rid] != right['outputs'][rid]]
    request_sets_equal = left['outputs'].keys() == right['outputs'].keys()
    identity_equal = left['input_identity'] == right['input_identity']
    exact = all(p['exact'] for p in paths.values()) and request_sets_equal and identity_equal and not mismatched
    return dict(status='MATCH' if exact else 'DIVERGED', paths=paths, input_identity_equal=identity_equal,
        active_start_steps=starts, memory_from_activation=active_memory,
        outputs=dict(reference_requests=len(left['outputs']), candidate_requests=len(right['outputs']),
            request_sets_equal=request_sets_equal, shared_requests=len(shared),
            exact_sequences=len(shared) - len(mismatched), different_request_ids=mismatched),
        input_sha256=dict(reference=lhs, candidate=rhs))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('refuse to overwrite an artifact')
    rows = {f'repeat{i}-headroom': compare(args.reference / 'gpu_results' / f'repeat{i}-headroom',
            args.candidate / 'gpu_results' / f'repeat{i}-headroom') for i in range(2)}
    result = dict(evidence_type='HISTORICAL_RECORDED_PATH_COMPARISON', reference=str(args.reference),
        candidate=str(args.candidate), scope='Independent actual raw runs, normalized source request IDs. '
        'Decision clocks/observer omitted. Scheduled intervals and memory include request progress, '
        'block counts, pool state and running order; physical block IDs and route tensors were not recorded. '
        'No cross-run timing comparison or performance counterfactual.', rows=rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(result, handle, indent=2)
        handle.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
