"""WiSP LRU/grouping calibration on one frozen trace; no serving counterfactual."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import runpy
import sys

UPSTREAM_SHA = '5572d4f05593a5a9fc4adaa14421cc596886a435b6f5f51f476a1dae6e521857'


def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


class ReplayMismatch(Exception):
    def __init__(self, field, got, expected, record, group=None):
        self.detail = dict(field=field, got=got, expected=expected, call_id=record['call_id'],
            phase=record['context']['phase'], layer=record['layer_name'], group=group)
        super().__init__(str(self.detail))


def equal(got, expected, field, record, group=None):
    if got != expected: raise ReplayMismatch(field, got, expected, record, group)


class LRU:
    def __init__(self, cap):
        self.cap, self.slots, self.mapping = cap, [-1] * cap, {}
        self.ticks, self.clock = [0] * cap, 0

    def ensure(self, experts):
        # torch.unique(...).tolist() is sorted; upstream then constructs an int set.
        needed = set(int(e) for e in sorted(set(experts)))
        missing = [e for e in needed if e not in self.mapping]
        victims = []
        if missing:
            free = [s for s in range(self.cap) if self.slots[s] == -1]
            candidates = sorted([(self.ticks[s], s) for s in range(self.cap)
                if self.slots[s] != -1 and self.slots[s] not in needed], key=lambda x:x[0])
            count = max(0, len(missing)-len(free))
            if count > len(candidates): raise ValueError('ensure exceeds capacity')
            for _, s in candidates[:count]:
                expert = self.slots[s]; victims.append(expert)
                del self.mapping[expert]; self.slots[s] = -1; free.append(s)
            self.clock += 1
            for expert in missing:
                s = free.pop(); self.slots[s] = expert; self.mapping[expert] = s; self.ticks[s] = self.clock
        else:
            self.clock += 1
        for expert in needed: self.ticks[self.mapping[expert]] = self.clock
        return sorted(missing), sorted(victims)

    def snapshot(self):
        return dict(slot_to_expert=list(self.slots), expert_to_slot={str(e):s for e,s in self.mapping.items()},
            lru_tick=list(self.ticks), lru_clock=self.clock,
            expert_map_device=[self.mapping.get(e,-1) for e in range(64)])


def replay(records, caps, plan_fn, sizes, anchors=None, verify=False):
    states = {name:LRU(cap) for name,cap in caps.items()}
    cost = {name:{k:Counter() for k in ('warmup','measurement')} for name in caps}
    for r in records:
        name = r['layer_name']; state = states[name]; context = r['context']
        entry = set(state.mapping); active = set(e for row in r['row_topk_experts'] for e in row)
        if verify:
            equal(sorted(entry), r['entry_resident_experts'], 'entry_resident', r)
            equal(sorted(active), r['active_experts'], 'active_union', r)
            anchor = (anchors or {}).get((context['phase'],context['step_id']),{}).get(name)
            if anchor:
                snap = state.snapshot()
                for field in snap.keys() & anchor.keys(): equal(snap[field],anchor[field],'anchor/'+field,r)
        t = r['retention']
        plan, info = plan_fn(r['row_topk_experts'], context, entry, state.cap, t['mode'],
            salt=t.get('hash_salt') or 'unused', order=t['order'], guard=t.get('guard','none'))
        if verify:
            expected = [dict(execute_experts=g['required_experts'],ensure_experts=g['ensure_experts'],
                             protected_after=g['protected_after']) for g in r['groups']]
            equal(plan, expected, 'group_plan', r)
            equal(info['chosen_protected_experts'],t['chosen_protected_experts'],'protected_selection',r)
        execute = [e for g in plan for e in g['execute_experts']]
        equal(dict(Counter(execute)),dict(Counter(active)),'disjoint_active_coverage',r)
        loaded, evicted = [], []
        for i,g in enumerate(plan):
            missing,victims = state.ensure(g['ensure_experts']); loaded.extend(missing); evicted.extend(victims)
            if not set(g['protected_after']) <= set(state.mapping): raise ValueError('protection evicted')
            if verify:
                for field,value in (('loaded_experts',missing),('evicted_experts',victims),
                        ('miss',len(missing)),('evict',len(victims)),('weight_copy_bytes',len(missing)*sizes[name])):
                    equal(value,r['groups'][i][field],field,r,i)
        equal(dict(Counter(loaded)),dict(Counter(active-entry)),'entry_misses_once',r)
        if verify: equal(sorted(state.mapping),t['final_resident_experts'],'final_resident',r)
        bucket = 'measurement' if r['measurement'] else 'warmup'
        cost[name][bucket].update(calls=1,miss=len(loaded),evict=len(evicted),groups=len(plan),bytes=len(loaded)*sizes[name])
    return cost


def allocate(curves, names, budget, group_limit):
    # For each used-slot/group pair retain minimum miss, lexicographic-cap tie break.
    states = {(0,0):(0,())}; sizes=[]
    for i,name in enumerate(names):
        next_states={}; remaining=len(names)-i-1
        minimum_future_groups=sum(min(c['measurement']['groups'] for c in curves[n]) for n in names[i+1:])
        for (slots,groups),(miss,caps) in states.items():
            for c in curves[name]:
                s,g=slots+c['cap'],groups+c['measurement']['groups']
                if not remaining*16 <= budget-s <= remaining*32 or g+minimum_future_groups>group_limit: continue
                value=(miss+c['measurement']['miss'],caps+(c['cap'],)); key=(s,g)
                if key not in next_states or value<next_states[key]: next_states[key]=value
        # At equal slot use, fewer groups and no greater miss dominates safely.
        states={}; best={}
        for key,value in sorted(next_states.items()):
            if value[0] < best.get(key[0],float('inf')):
                states[key]=value; best[key[0]]=value[0]
        sizes.append(dict(layers=i+1,states=len(states)))
    feasible=[(miss,g,caps) for (s,g),(miss,caps) in states.items() if s==budget]
    if not feasible: raise ValueError('DP lost the feasible uniform allocation')
    chosen=min(feasible); min_groups=min(feasible,key=lambda x:(x[1],x[0],x[2]))
    return dict(zip(names,chosen[2])),dict(zip(names,min_groups[2])),sizes


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input-dir',type=Path,required=True);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    result=dict(status='STRUCTURAL_FIXED_TRACE_CALIBRATION_ONLY',issues=[],scope=[
        'Only 0_forward/repeat0 none calibration routes, including all five common warmups, are used.',
        'Each cap replays its own LRU/grouping state on these fixed calibration rows; future model outputs/routes are not regenerated.',
        'This is neither counterfactual GPU benefit nor a serving Oracle; no latency or holdout result is estimated.',
        'Objective is calibration measurement miss, constrained by 384 slots and measurement groups <= uniform; warmup costs are also retained.',
        'Reset does not clear ever_loaded; reloaded_experts depends on pre-reset history and is not inferred. It does not affect LRU/load decisions.'])
    try:
        root=args.input_dir; source=root/'source/wisp_expert_groups.py'; protocol=read(root/'protocol.json')
        if sha(source)!=protocol['source_sha256']['wisp_expert_groups.py']: raise ValueError('frozen planner source mismatch')
        plan_fn=runpy.run_path(str(source))['retention_plan']; cell=root/'results/0_forward'; pager=read(cell/'pager_summary.json')
        if pager['upstream_sha256']!=UPSTREAM_SHA: raise ValueError('unsupported upstream ensure implementation')
        episode=read(cell/'episodes.json')[0]; prefix=episode['phase'].split('/')[0]; d=cell/prefix
        if prefix!='repeat_0_retention_none_early': raise ValueError('calibration identity differs')
        raw=read(d/'raw.json'); reset=read(d/'reset.json'); names=[x['layer_name'] for x in pager['layers']]
        sizes={x['layer_name']:x['pinned_bytes']//x['num_experts'] for x in pager['layers']}
        if len(names)!=16 or len(set(sizes.values()))!=1: raise ValueError('384 slots require 16 identical expert-row sizes')
        for name in names:
            if reset['cache_after'][name]!=LRU(24).snapshot(): raise ValueError('reset is not an empty cap24 LRU: '+name)
        records=[]
        with (cell/'pager/calls.jsonl').open() as stream:
            for line in stream:
                r=json.loads(line)
                if r['context']['phase'].startswith(prefix+'/'): records.append(r)
        if any(r['status']!='complete' for r in records): raise ValueError('incomplete calibration call')
        ids=[r['call_id'] for r in records]
        if ids!=list(range(ids[0],ids[-1]+1)): raise ValueError('nonconsecutive calibration call IDs')
        expected={prefix+'/'+s:n for s,n in [('warmup',32),('warmup_injection_none_early_release8',384),
            ('warmup_injection_frequency_early_release8',384),('warmup_injection_frequency_late_release8',384),
            ('warmup_injection_decode_late_release8',384),('measurement',384)]}
        if Counter(r['context']['phase'] for r in records)!=expected: raise ValueError('common warmup/measurement coverage mismatch')
        action=raw['event_actions'][0]; action_step=raw['engine_calls'][action['engine_call']]['scheduler_step_start']
        anchors={(episode['phase'],0):read(d/'measurement_initial_cache.json'),(episode['phase'],action_step):action['before']['cache']}
        uniform={n:24 for n in names}; verified=replay(records,uniform,plan_fn,sizes,anchors,True)
        result.update(calibration=dict(engine='0_forward',repeat=prefix,phase_counts=expected,call_id_first=ids[0],call_id_last=ids[-1],
            planner_sha256=sha(source),upstream_sha256=UPSTREAM_SHA,python=sys.version),
            verification=dict(status='PASS',calls=len(records),layers=len(names),
                checks=['Exact frozen group plan/selection and every call entry/final resident set.',
                    'Every group loaded/evicted expert set and copy/miss/evict counts match.',
                    'All active contributions exactly once, capacity and entry-miss-once invariants.',
                    'Full slot/map/LRU state matches measurement-start and pre-injection anchors.']))
        curves={}
        for name in names:
            layer_records=[r for r in records if r['layer_name']==name]
            curves[name]=[dict(cap=cap,**replay(layer_records,{name:cap},plan_fn,sizes)[name]) for cap in range(16,33)]
        result['curves']=curves
        def cost(allocation):
            totals={k:Counter() for k in ('warmup','measurement')}
            for name,cap in allocation.items():
                row=next(c for c in curves[name] if c['cap']==cap)
                for k in totals: totals[k].update(row[k])
            return dict(allocation=allocation,total_slots=sum(allocation.values()),
                expert_scratch_bytes=sum(sizes[n]*cap for n,cap in allocation.items()),**totals)
        baseline=cost(uniform); selected,min_groups,dp=allocate(curves,names,384,baseline['measurement']['groups'])
        result.update(uniform=baseline,min_groups=cost(min_groups),selected=cost(selected),dp_states=dp,
            tie_break='Minimum measurement miss, then groups, then lexicographic caps in recorded layer order; min-groups reverses first two objectives.')
        for key in ('warmup','measurement'):
            if dict(baseline[key])!=dict(sum((verified[n][key] for n in names),Counter())): raise ValueError('uniform curve/verification aggregate mismatch')
    except ReplayMismatch as exc:
        result.update(status='REPLAY_MISMATCH_STOPPED',first_mismatch=exc.detail);result['issues'].append('cap24 replay failed; curves/allocation not evaluated')
    except (OSError,KeyError,ValueError,TypeError,IndexError) as exc:
        result['issues'].append(repr(exc));result['status']='CALIBRATION_FAILED'
    with args.out.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    if result['issues']:raise SystemExit('calibration stopped; diagnostic retained')


if __name__=='__main__':main()
