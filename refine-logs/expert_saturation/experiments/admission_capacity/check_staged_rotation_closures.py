"""Exercise actual adapter closures with fake scheduler/transfer objects.

No native scheduler allocation, GPU or physical transfer is executed.
"""
import ast
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace as NS
import staged_store_rotation as adapter

O=Path(__file__).resolve().parents[2]/'outputs/admission_capacity'
class Queue(list):
    def remove_request(self,r):self.remove(r)
    def prepend_request(self,r):self.insert(0,r)


def factory():
    source=Path(adapter.__file__).read_text();tree=ast.parse(source)
    install=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='install')
    start=next(i for i,n in enumerate(install.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='step' for t in n.targets))
    fn=ast.parse('def fixture(scheduler,native,owned,pool,cs,save):\n expected_requests=32\n oldcalc=cs._calc_num_offloadable_tokens\n hadcalc=False\n').body[0]
    fn.body+=deepcopy(install.body[start:])
    env=dict(vars(adapter));exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),'<actual-staged-closures>','exec'),env)
    return env['fixture']


def run_case(save=False,changed_target=False,omit_flush=False,partial=False):
    inp=json.loads((O/'20260914_recovery_progress_model_r01/input.json').read_text())
    states=inp['state'];owned={};requests={};blocks={}
    for rid,r in states['requests'].items():
        owned[rid]=[blocks.setdefault(n,NS(block_id=n,is_null=False)) for n in r['blocks']]
        requests[rid]=NS(request_id=rid,num_computed_tokens=r['computed'],num_prompt_tokens=r['prompt'],num_output_tokens=r['output'],max_tokens=r['max_tokens'],status=NS(name=r['status']))
    free=[states['free']];pool=NS(blocks=blocks,get_num_free_blocks=lambda:free[0])
    scheduler=NS(requests=requests,running=[requests[r] for r in states['running']],waiting=Queue(requests[r] for r in states['waiting']),skipped_waiting=Queue())
    cs=NS(_calc_num_offloadable_tokens=lambda rs,n:n,_jobs={},_req_status={})
    for rid,req in requests.items():
        cs._req_status[rid]=NS(req=req,transfer_jobs=set(),group_states=[NS(block_ids=list(states['requests'][rid]['blocks']),offload_keys=[f'{rid}:{n}' for n in range(len(owned[rid]))])])
    def preempt(req,timestamp):
        free[0]+=len(owned[req.request_id]);owned[req.request_id]=[]
        req.status=NS(name='PREEMPTED');req.num_computed_tokens=0;scheduler.waiting.prepend_request(req)
    scheduler._preempt_request=preempt
    visits=[]
    def native():
        pre=[];scheduler._rotation_begin(pre,0.0)
        meta=NS(store_jobs={},load_jobs={},jobs_to_flush=set())
        scheduled={r.request_id:1 for r in scheduler.running};resumed=[]
        if pre:
            target=requests[scheduler._rotation_target]
            # Fake full-history reservation and one admitted chunk; CPU only.
            count=(target.num_prompt_tokens+target.num_output_tokens+15)//16
            fake=[NS(block_id=10000+n,is_null=False) for n in range(count)]
            for b in fake:blocks[b.block_id]=b
            owned[target.request_id]=fake;free[0]-=count
            scheduler.waiting.remove_request(target);scheduler.running.append(target);target.status=NS(name='RUNNING')
            scheduled[target.request_id]=1;resumed=[target.request_id]
            if not omit_flush:meta.jobs_to_flush=set(cs._req_status[pre[0].request_id].transfer_jobs)
        elif save:
            for rid,rs in cs._req_status.items():
                cap=cs._calc_num_offloadable_tokens(rs,rs.req.num_computed_tokens)
                if not cap:continue
                n=cap//16;indices=list(range(max(0,n-2),n)) if partial else list(range(n))
                payload=[rs.group_states[0].block_ids[i] for i in indices]
                meta.store_jobs[7]=NS(req_id=rid,src_spec=NS(block_ids=payload))
                cs._jobs[7]=NS(req_id=rid,is_store=True,keys={rs.group_states[0].offload_keys[i] for i in indices});rs.transfer_jobs.add(7)
        visits.append(dict(preempted=[r.request_id for r in pre],target=scheduler._rotation_target,flush=sorted(meta.jobs_to_flush)))
        return NS(num_scheduled_tokens=scheduled,preempted_req_ids=[r.request_id for r in pre],scheduled_cached_reqs=NS(resumed_req_ids=resumed),kv_connector_metadata=meta)
    data,uninstall=factory()(scheduler,native,owned,pool,cs,save)
    cells=dict(zip(scheduler.schedule.__code__.co_freevars,scheduler.schedule.__closure__))
    cells['step'].cell_contents=329
    # The fixture starts at a captured post-activation state; activation itself
    # and install-time backend validation are outside this check.
    begin_cells=dict(zip(scheduler._rotation_begin.__code__.co_freevars,scheduler._rotation_begin.__closure__))
    begin_cells['cohort'].cell_contents=set(requests)
    tracker=cells['tracker'].cell_contents
    for h in inp['history']:
        tracker.note_preempted(h['step'],h['preempted']);tracker.note_resumed(h['step'],h['resumed'])
    scheduler.schedule()
    plan=next(e for e in data['events'] if e['event']=='prepare')
    assert plan['victim'].endswith('0000001') and plan['target'].endswith('0003640')
    victim=requests[plan['victim']];victim.num_computed_tokens+=1;victim.num_output_tokens+=1
    if changed_target:requests[plan['target']].num_output_tokens+=1
    try:scheduler.schedule()
    except RuntimeError as exc:
        assert omit_flush and 'flush omitted' in str(exc)
        outcome='REJECTED_MISSING_FLUSH'
    else:
        if changed_target:
            assert not visits[-1]['preempted'] and data['applied_rotations']==0;outcome='CANCELLED_CHANGED_TARGET'
        else:
            assert visits[-1]['preempted']==[plan['victim']] and visits[-1]['target']==plan['target'];outcome='COMMITTED'
    result=dict(outcome=outcome,visits=visits,store_deltas=[e for e in data['events'] if e['event']=='store_delta'])
    uninstall();assert not hasattr(scheduler,'schedule')
    return result

if __name__=='__main__':
    results=dict(save_off=run_case(),save_full=run_case(save=True),save_increment=run_case(save=True,partial=True),changed_target=run_case(changed_target=True),missing_flush=run_case(save=True,omit_flush=True))
    print(json.dumps(dict(status='ACTUAL_CLOSURE_FIXTURES_PASS',cases=results,scope='Actual repeated adapter closures. Captured start-state counts/IDs, synthetic keys/jobs/allocator/engine progression. No full native schedule or GPU execution.'),indent=2))
