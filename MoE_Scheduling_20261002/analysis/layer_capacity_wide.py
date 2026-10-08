"""Wider cap8..64 fixed-trace static diagnostic; no controller or time prediction."""
from collections import Counter
from pathlib import Path
import hashlib
import json
import runpy

W = Path(__file__).resolve().parents[1]
R = W.parent/'MOE_Thesis_organized/refine-logs/expert_saturation/experiments/admission_capacity'
oldpath = Path(__file__).with_name('layer_capacity_profile.json')
old = json.loads(oldpath.read_text())
tracepath = W/'results_opt_r01/00_default24/pager/calls.jsonl'
LRU = runpy.run_path(str(R/'analyze_layer_budget.py'))['LRU']
partition = runpy.run_path(str(W/'optimization_src/wisp_expert_groups.py'))['partition_experts']
traces = {}
for line in tracepath.open():
    r = json.loads(line)
    assert r['status'] == 'complete' and r['retention']['mode'] == 'none'
    active = {e for row in r['row_topk_experts'] for e in row}
    assert active == set(r['active_experts'])
    traces.setdefault(r['layer_name'],[]).append((active,r['measurement']))
names = list(old['curves']); size = old['domain']['expert_row_bytes']

def replay(n,cap):
    state = LRU(cap); costs = {k:Counter() for k in ('warmup','measurement')}
    for active,measured in traces[n]:
        groups = partition(active,set(state.mapping),cap); miss = evict = 0
        for g in groups:
            m,e = state.ensure(g); miss += len(m); evict += len(e)
        costs['measurement' if measured else 'warmup'].update(calls=1,miss=miss,evict=evict,groups=len(groups),bytes=miss*size)
    return dict(cap=cap,**costs)

curves = {n:[replay(n,c) for c in range(8,65)] for n in names}
for n in names:
    for c in old['curves'][n]:
        assert curves[n][c['cap']-8] == c  # All 272 old full-route curve entries agree.

def solve(objective):
    states = {0:(0,0,())}
    for i,n in enumerate(names):
        new = {}; remaining = len(names)-i-1
        for slots,(primary,secondary,caps) in states.items():
            for c in curves[n]:
                used = slots+c['cap']
                if not 8*remaining <= 384-used <= 64*remaining:
                    continue
                m = c['measurement']; first,second = ('miss','groups') if objective == 'miss' else ('groups','miss')
                v = (primary+m[first],secondary+m[second],caps+(c['cap'],))
                if used not in new or v < new[used]:
                    new[used] = v
        states = new
    allocation = dict(zip(names,states[384][2])); totals = {k:Counter() for k in ('warmup','measurement')}
    for n,cap in allocation.items():
        c = curves[n][cap-8]
        for k in totals:
            totals[k].update(c[k])
    base = old['allocations']['uniform']['measurement']
    return dict(caps=allocation,total_slots=384,scratch_bytes=384*size,**totals,
        versus_uniform={k:dict(delta=totals['measurement'][k]-base[k],percent=100*(totals['measurement'][k]/base[k]-1)) for k in ('miss','bytes','groups')})

result = dict(status='STRUCTURAL_FIXED_TRAINING_TRACE_WIDE_ONLY',
    source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (tracepath,oldpath)},
    verification='Union-only ordinary planner replay exactly matches all 272 prior full-route cap16..32 curve entries; prior cap24 exact GPU-log match inherited.',
    domain=dict(caps=list(range(8,65)),total_slots=384,expert_row_bytes=size),
    caveats=old['caveats']+['The widened cap8..64 runtime allocation itself is UNRUN; optimizer consumes the whole old training trace.',
        'No monotonic/convex marginal assumption: dynamic programming enumerates every allowed static cap, including full64.'],
    uniform=old['allocations']['uniform'], minimum_miss=solve('miss'), minimum_groups=solve('groups'),curves=curves)
result['caveats'] = [s for s in result['caveats'] if 'cap16..32 static allocation domain' not in s]
result['caveats'].append('Exact optimum only within fixed old trace and cap8..64 static allocation domain; not a serving Oracle.')
out = Path(__file__).with_suffix('.json')
with out.open('x') as f:
    json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
print(json.dumps({k:v for k,v in result.items() if k in ('status','minimum_miss','minimum_groups')}))
