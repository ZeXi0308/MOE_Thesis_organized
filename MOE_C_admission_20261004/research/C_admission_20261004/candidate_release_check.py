"""CPU diagnosis of candidate-release credit in saved, same-state MC examples.

This does not run a policy or predict end-to-end performance. Immediate completed
prefill is optimistic; the candidate is removed at h=B in the alternative.
"""
import argparse
import hashlib
import json
from pathlib import Path

from mc_budget import peak_envelope


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('group', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    results = []
    for path in sorted(args.group.glob('probe-*/admission.json')):
        payload = path.read_bytes()
        report = json.loads(payload)
        seen = set()
        for label in ('first_eligible', 'first_scheduled_prefill_relaxation'):
            event = report['mc_budget'].get(label)
            if event is None or event['decision_index'] in seen:
                continue
            seen.add(event['decision_index'])
            rows = event['old_rows']
            p, B = event['new_prompt_tokens'], event['new_max_tokens']
            b, M = report['block_size'], report['budget_blocks']
            max_r = max((r['r'] for r in rows), default=0)
            retained = peak_envelope(rows, p, B, b)['peak_blocks']
            released = max(sum((r['n']+h+b-1)//b for r in rows if h < r['r'])
                           + ((p+h+b-1)//b if h < B else 0)
                           for h in range(max(max_r, B)+1))
            condition = max_r <= B and (p+B+b-1)//b <= M
            if condition and (retained <= M) != (released <= M):
                raise AssertionError('Conditional feasibility equivalence violated')
            results.append(dict(cell=path.parent.name, saved_event=label,
                decision_index=event['decision_index'], request_id=event['request_id'],
                source_sha256=hashlib.sha256(payload).hexdigest(), old_count=len(rows),
                new_prompt=p, new_bound=B, maximum_old_remaining=max_r,
                candidate_full_pages=(p+B+b-1)//b, physical_pages=M,
                condition=condition, retained_peak=retained, optimistic_released_peak=released,
                retained_fits=retained <= M, optimistic_released_fits=released <= M))
    if not results:
        raise ValueError('No saved MC examples')
    output = dict(scope='Saved-state CPU diagnosis only; not all decisions or a policy counterfactual.',
        assumption='Old envelopes valid; one-token rounds; candidate release at h=B assumes '
            'immediate completed prefill. No future realized output is used.',
        conditional_argument='For h<B envelopes coincide. For h>=B, max(old r)<=B removes '
            'all old rows; the retained candidate alone fits M. Thus feasibility is identical '
            'when the condition holds; peak values need not be identical.',
        examples=len(results), condition_holds=sum(r['condition'] for r in results),
        feasibility_changes=sum(r['retained_fits'] != r['optimistic_released_fits'] for r in results),
        results=results)
    with args.output.open('x') as stream:
        json.dump(output, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({k: output[k] for k in ('examples', 'condition_holds', 'feasibility_changes')}))


if __name__ == '__main__':
    main()
