"""Exact-content reuse opportunities on recorded X traces, never GPU savings.

Each measurement cell starts a cold shadow map cache, with no future input.
Model/cache/request trajectories remain the original policy's recorded states.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


def read_campaign(root):
    cells = json.loads((root/'run_cells.json').read_text())
    results, events = [], []
    for spec in cells:
        if spec['arm'] != 'X':
            continue
        trace = root/'attempt01/results'/spec['label']/'pager/calls.jsonl'
        previous = {}
        totals, by_stage, by_rows, transitions = Counter(), defaultdict(Counter), defaultdict(Counter), Counter()
        last_call = {}
        for line in trace.open():
            r = json.loads(line)
            if not r['measurement']:
                continue
            assert not r['validation_run'] and r['status'] == 'complete'
            assert r['execution'] == 'shared_pool_oneshot' and r['group_count'] == 1
            p = r['shared_plan']; layer = r['layer_name']
            local, execute = tuple(p['final_state']['expert_map_device']), tuple(p['expert_map_device'])
            assert len(local) == 64 and len(execute) == 384
            assert all(x == -1 for x in execute[64:])
            active = {e for row in r['row_topk_experts'] for e in row}
            assert active == set(r['active_experts']) == {e for e,slot in enumerate(execute) if slot >= 0}
            assert len({execute[e] for e in active}) == len(active)
            assert r['miss'] == len(p['h2d']) and r['d2d_copy_bytes'] == sum(g['d2d_copy_bytes'] for g in r['groups'])
            coords = r['context']['rows']
            decode = sum(x['computed_position'] >= x['prompt_tokens'] for x in coords)
            stage = 'decode' if decode == len(coords) else ('mixed' if decode else 'prefill')
            old = previous.get(layer)
            same_local = old is not None and local == old[0]
            same_execute = old is not None and execute == old[1]
            count = dict(calls=1, baseline_map_uploads=2,
                local_same=int(same_local), execute_same=int(same_execute), both_same=int(same_local and same_execute),
                skippable_uploads=int(same_local)+int(same_execute),
                skippable_payload_bytes=256*int(same_local)+1536*int(same_execute),
                execute_same_with_weight_work=int(same_execute and bool(p['h2d'] or p['d2d'])),
                execute_same_with_h2d=int(same_execute and bool(p['h2d'])),
                both_same_with_h2d=int(same_local and same_execute and bool(p['h2d'])),
                h2d_bytes_under_same_execute=r['weight_copy_bytes'] if same_execute else 0)
            totals.update(count); by_stage[stage].update(count); by_rows[r['rows']].update(count)
            if old is not None:
                transitions[(sum(a != b for a,b in zip(local, old[0])),
                             sum(a != b for a,b in zip(execute, old[1])))] += 1
            events.append(dict(campaign=root.name, cell=spec['label'], layer=layer,
                call_id=r['call_id'], previous_call_id=last_call.get(layer),
                step=r['context']['step_id'], rows=r['rows'], stage=stage,
                active_experts=len(active), same_local=same_local, same_execute=same_execute,
                miss=r['miss'], weight_copy_bytes=r['weight_copy_bytes'], d2d_copy_bytes=r['d2d_copy_bytes']))
            previous[layer]=(local,execute); last_call[layer]=r['call_id']
        assert len(previous) == 16
        results.append(dict(campaign=root.name, cell=spec['label'], block=spec['block'],
            input_sha256=hashlib.sha256(trace.read_bytes()).hexdigest(), totals=totals,
            by_stage=by_stage, by_rows=dict(sorted(by_rows.items())),
            changed_entry_histogram=[dict(local_changed=k[0],execute_changed=k[1],count=v)
                                     for k,v in sorted(transitions.items())]))
    assert len(results) == 2
    return results, events


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign', type=Path, action='append', required=True)
    p.add_argument('--out-dir', type=Path, required=True)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    cells, events = [], []
    for campaign in args.campaign:
        rows, trace = read_campaign(campaign)
        cells.extend(rows); events.extend(trace)
    result = dict(status='RECORDED_TRAJECTORY_OPPORTUNITY', cells=cells,
        scope='Only existing X arm executions; map equality available after current planning. Cold per-layer shadow starts at measurement. No state, route, request or timing rerun.',
        invalidation='Persistent device map needs invalidation on reset/ordinary ensure/external writes. A private execution-map buffer is never shared with another layer.',
        time_saving_ms=None, additional_GPU_runs=0)
    for name, obj in [('analysis.json',result),('events.json',events)]:
        with (args.out_dir/name).open('x') as f:
            json.dump(obj,f,ensure_ascii=False,indent=2); f.write('\n')
    print(json.dumps([dict(campaign=r['campaign'],cell=r['cell'],totals=r['totals'],by_stage=r['by_stage']) for r in cells],ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
