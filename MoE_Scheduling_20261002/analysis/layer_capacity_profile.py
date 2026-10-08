"""Exact fixed-training-trace cache allocation diagnostic; no serving simulation."""
from collections import Counter
from pathlib import Path
import hashlib
import json
import runpy
import statistics

W = Path(__file__).resolve().parents[1]
R = W.parent / 'MOE_Thesis_organized'
SRC = W / 'results_opt_r01/00_default24'
HELPER = R / 'refine-logs/expert_saturation/experiments/admission_capacity/analyze_layer_budget.py'
PLANNER = W / 'optimization_src/wisp_expert_groups.py'
h = runpy.run_path(str(HELPER))
planner = runpy.run_path(str(PLANNER))['retention_plan']
summary = json.loads((SRC/'pager/summary.json').read_text())
records = [json.loads(s) for s in (SRC/'pager/calls.jsonl').open()]
names = [x['layer_name'] for x in summary['layers']]
sizes = {x['layer_name']:x['pinned_bytes']//x['num_experts'] for x in summary['layers']}
assert summary['upstream_sha256'] == h['UPSTREAM_SHA']
assert len(names) == 16 and len(set(sizes.values())) == 1
assert all(r['status'] == 'complete' and r['retention']['mode'] == 'none' for r in records)
verified = h['replay'](records, dict.fromkeys(names,24), planner, sizes,
    {('measurement',0):summary['measurement_initial_cache']}, True)
curves = {}
for n in names:
    layer_records = [r for r in records if r['layer_name'] == n]
    curves[n] = [dict(cap=cap, **h['replay'](layer_records,{n:cap},planner,sizes)[n]) for cap in range(16,33)]

def cost(caps):
    total = {k:Counter() for k in ('warmup','measurement')}
    for n,cap in caps.items():
        c = next(c for c in curves[n] if c['cap'] == cap)
        for k in total:
            total[k].update(c[k])
    return dict(caps=caps, total_slots=sum(caps.values()),
        scratch_bytes=sum(sizes[n]*cap for n,cap in caps.items()), **total)

baseline = cost(dict.fromkeys(names,24))
assert baseline['measurement']['miss'] == summary['measurement']['miss']
assert baseline['measurement']['bytes'] == summary['measurement']['weight_copy_bytes']
assert baseline['measurement']['groups'] == summary['measurement']['group_count']
states = {0:(0,0,())}
for index,n in enumerate(names):
    new = {}; remaining = len(names)-index-1
    for slots,(miss,groups,caps) in states.items():
        for c in curves[n]:
            use = slots+c['cap']
            if not 16*remaining <= 384-use <= 32*remaining:
                continue
            value = (miss+c['measurement']['miss'], groups+c['measurement']['groups'], caps+(c['cap'],))
            if use not in new or value < new[use]:
                new[use] = value
    states = new
unconstrained = cost(dict(zip(names,states[384][2])))
selected, min_groups, dp = h['allocate'](curves,names,384,baseline['measurement']['groups'])
allocations = dict(uniform=baseline, minimum_miss=unconstrained,
    minimum_miss_no_more_groups=cost(selected), minimum_groups=cost(min_groups))
for label,c in allocations.items():
    c['versus_uniform'] = {k:dict(delta=c['measurement'][k]-baseline['measurement'][k],
        percent=100*(c['measurement'][k]/baseline['measurement'][k]-1)) for k in ('miss','bytes','groups')}
layer_summary = []
for n in names:
    at = {c['cap']:c['measurement'] for c in curves[n]}
    calls = [r for r in records if r['measurement'] and r['layer_name'] == n]
    demand = sum(len(r['active_experts']) for r in calls)
    layer_summary.append(dict(layer=n, actual_miss=at[24]['miss'], actual_bytes=at[24]['bytes'],
        active_expert_call_demands=demand, hit_fraction=1-at[24]['miss']/demand,
        miss_reduction_24_to_25=at[24]['miss']-at[25]['miss'],
        miss_reduction_24_to_32=at[24]['miss']-at[32]['miss'],
        miss_penalty_24_to_16=at[16]['miss']-at[24]['miss']))
misses = [r['actual_miss'] for r in layer_summary]
paths = [SRC/'pager/calls.jsonl',SRC/'pager/summary.json',HELPER,PLANNER]
result = dict(status='STRUCTURAL_FIXED_TRAINING_TRACE_ONLY',
    source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
    verified=dict(calls=len(records), measurement_calls=summary['measurement_calls'], mismatches=0,
        checks='Actual cap24 group plans, entry/final residents, each group load/evict/count/bytes, and measurement-start full slot/LRU anchor.'),
    domain=dict(layer_caps=list(range(16,33)), slots=384, expert_row_bytes=next(iter(sizes.values())),
        tie_break='Minimum measurement miss, then groups, then lexicographic caps in recorded layer order.'),
    caveats=['Only 00_default24 old training trace read; no holdout.',
        'All initialization/warmup routes replayed to form each allocation-specific LRU state; warmup field includes initialization.',
        'Future routes/batch shapes/KV/tokens are held to the measured trajectory, not regenerated.',
        'Exact optimum only within this fixed-trace cap16..32 static allocation domain; not a serving Oracle or deployable guarantee.',
        'Bytes derive from measured expert-row size; no wall-time, queue, quality or counterfactual benefit claimed.'],
    layer_heterogeneity=dict(min_miss=min(misses),max_miss=max(misses),
        max_min_ratio=max(misses)/min(misses),population_cv=statistics.pstdev(misses)/statistics.mean(misses)),
    layer_summary=layer_summary, allocations=allocations, curves=curves, group_constrained_dp_states=dp)
out = Path(__file__).with_suffix('.json')
with out.open('x') as f:
    json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
print(json.dumps(dict(output=str(out),heterogeneity=result['layer_heterogeneity'],allocations=allocations,layer_summary=layer_summary)))
