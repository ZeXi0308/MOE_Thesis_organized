"""Replay the unchanged baseline; propose only the first different guarded plan."""
import ast
from dataclasses import dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

B=Path(__file__).resolve().parents[1]
P=B/'preparation/pkg'
R=B.parent/'20260914_ltr_packing_r01/execution/readback/results/block0-d6-packing-rank_prefix'
raw=json.loads((R/'raw.json').read_text()); decisions=json.loads((R/'component-decisions.json').read_text())
spec=importlib.util.spec_from_file_location('restore_frozen',P/'restore_obligation.py')
h=importlib.util.module_from_spec(spec);sys.modules[spec.name]=h;spec.loader.exec_module(h)
tree=ast.parse((P/'ltr_recompute_native.py').read_text());ns={'dataclass':dataclass,'__name__':__name__}
exec(compile(ast.Module(body=[n for n in tree.body if getattr(n,'name',None) in ('Candidate','Plan','plan')],type_ignores=[]),str(P/'ltr_recompute_native.py'),'exec'),ns)
aliases=raw['internal_to_source'];req={q['request_id']:q for q in raw['requests']}
off,on=h.RestoreObligations(),h.RestoreObligations(); previous=set();first=None; begins=0
for k,(d,m,step) in enumerate(zip(decisions,raw['memory_trace'],raw['scheduler_steps'])):
    before,after=m['before'],m['after'];state=before['requests']
    views={rid:h.RequestView('RUNNING' if rid in before['running_ids'] else 'PREEMPTED' if v['num_preemptions'] else 'WAITING',v['output_tokens'],v['prompt_tokens']+v['output_tokens']-v['computed_tokens']) for rid,v in state.items()}
    completed=previous-set(state)
    assert all(req[aliases[rid]]['status']=='completed' and req[aliases[rid]]['completion_s']<=step['start_s'] for rid in completed)
    off.begin_schedule(k,views,completed)
    if first is None: on.begin_schedule(k,views,completed)
    def plan(priorities):
        rows=[ns['Candidate'](rid,priorities[rid],req[aliases[rid]]['arrival_s'],rid in before['running_ids'],v['block_counts'][0],(v['prompt_tokens']+v['output_tokens']+15)//16,views[rid].pending_tokens) for rid,v in state.items()]
        return ns['plan'](rows,before['pool']['free_blocks'],1024,32,0,packing='rank_prefix')
    native=plan(d['priorities'])
    assert (native.tokens,native.victims,native.free_after_reservation)==(d['tokens'],d['victims'],d['free_after_reservation'])
    if first is None:
        effective=on.effective_priorities(d['priorities'],enabled=True);protected=plan(effective)
        if (protected.tokens,protected.victims)!=(native.tokens,native.victims):
            first=dict(step=k,free_before=before['pool']['free_blocks'],active=on.snapshot()['active'],
                       baseline=vars(native),guard=vars(protected),effective_priorities=effective,
                       boundary='First proposed different action only. No old future state is used to predict guard performance.')
    resumed=[rid for rid in d['actual_scheduled'] if views[rid].status=='PREEMPTED']
    remaining={rid:max(0,(v['prompt_tokens']+v['output_tokens']+15)//16-v['block_counts'][0]) for rid,v in after['requests'].items()}
    off.after_schedule(k,views,resumed,d['actual_scheduled'],d['victims'],remaining,after['pool']['free_blocks'],enabled=False,effective_priorities=d['priorities'])
    if first is None:
        on.after_schedule(k,views,resumed,d['actual_scheduled'],d['victims'],remaining,after['pool']['free_blocks'],enabled=True,effective_priorities=effective)
    previous=set(state)
off.finalize(len(decisions),{},[q['internal_request_id'] for q in raw['requests'] if q['status']=='completed'])
assert first is not None and not off.active
result=dict(status='CPU_FIRST_ACTION_FEASIBLE',baseline_replayed_steps=len(decisions),first_different_action=first,
            baseline_ledger=off.snapshot(),source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [P/'ltr_recompute_native.py',P/'restore_obligation.py',Path(__file__)]},
            chunk_assumption='CPU chunk_limit=0 matches every observed baseline action. Live source config/scheduler.py defaults threshold0 and max_num_partial_prefills1; old resolved value not recorded. New runner records actual resolved threshold; not an old frozen config claim.')
out=Path(__file__).with_name('first_action.json');out.open('x').write(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(status=result['status'],first_step=first['step'],baseline_calls=len(decisions),guard=first['guard'])))
