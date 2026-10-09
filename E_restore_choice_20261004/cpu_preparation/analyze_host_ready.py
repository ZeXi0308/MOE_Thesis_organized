"""Read completed Host arms; compare LOAD issue/report steps, not wall savings."""
import argparse
from bisect import bisect_right
import json
from pathlib import Path


def inspect(cell):
    assert json.loads((cell/'status.json').read_text())['status'] == 'COMPLETE'
    raw = json.loads((cell/'raw.json').read_text())
    resources = json.loads((cell/'resources.json').read_text())
    group, = resources['group_config']
    chunk = group['tokens_per_chunk']
    token_bytes = resources['gpu_bytes']//resources['gpu_blocks']//group['tokens_per_block']
    starts = [step['start_s'] for step in raw['steps']]

    def step_at(t):
        i = bisect_right(starts, t)-1
        assert i >= 0 and t <= raw['steps'][i]['end_s']
        return i

    issued, reported, measures = {}, {}, []
    commits = [c for c in raw['commits'] if c['external_tokens'] > 0]
    for c in commits:
        issued.setdefault(step_at(c['allocation_s']), []).append(c['external_tokens']*token_bytes)
    for transfer in raw['transfers']:
        load = transfer.get('load', {})
        if not load.get('bytes', 0):
            continue
        i = step_at(transfer['time_s'])
        assert sum(load['sizes']) == load['bytes']
        reported.setdefault(i, []).extend(load['sizes'])
        measures.append(dict(step=i, load_report_cuda_s=load['time'],
                             load_count=len(load['sizes']),
                             engine_step_s=raw['steps'][i]['end_s']-starts[i]))
    return dict(cell=str(cell), load_commits=len(commits),
        reported_loads=sum(map(len, reported.values())),
        unequal_issue_report_step_multisets=[i for i in sorted(set(issued)|set(reported))
            if sorted(issued.get(i, [])) != sorted(reported.get(i, []))],
        inline_structural_candidates=sum(c['eligible'] and
            0 < c['known_tokens']-c['external_tokens'] < chunk and
            c['external_tokens'] == c['known_tokens']//chunk*chunk for c in commits),
        reports=measures)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('cells', nargs='+', type=Path)
    p.add_argument('--out', required=True, type=Path)
    a = p.parse_args()
    result = dict(evidence='Existing completed native Host traces; no new GPU run',
        limits=['CUDA durations exclude queueing before start_event and are not exposed wall time.',
                'Equality compares complete per-step LOAD byte multisets, not arbitrary per-request attribution.',
                'Readiness in the issue step does not prove zero effect on step wall time or a strict gain bound.',
                'Structural candidates are shadow action opportunities, not actual inline execution.'],
        arms=[inspect(cell) for cell in a.cells])
    a.out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps([{k:v for k,v in arm.items() if k != 'reports'} for arm in result['arms']]))


if __name__ == '__main__':
    main()
