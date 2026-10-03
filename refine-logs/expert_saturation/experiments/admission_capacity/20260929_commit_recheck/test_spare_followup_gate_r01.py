"""Targeted actual-closure checks; fixtures are not native GPU evidence."""
import ast
from copy import deepcopy
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS

ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('qfixture',ROOT/'protection_quantity_probe_r02/test_protection_quantity_cpu.py')
q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q)
source=ROOT/'candidate_spare_followup_r01/pkg/staged_store_rotation.py'
spec=importlib.util.spec_from_file_location('spare_adapter',source)
adapter=importlib.util.module_from_spec(spec);spec.loader.exec_module(adapter)
q.base.Queue.peek_request=lambda self:self[0]
enabled=True

def fixture(quantity):
    install=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='install')
    start=next(i for i,n in enumerate(install.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='step' for t in n.targets))
    seed=ast.parse(f'''def fixture(scheduler,native,owned,pool,cs):
 manager=scheduler.kv_cache_manager
 expected_requests=32
 open_population=True
 diagnostic=False
 store_scope='selected'
 global_cooldown_steps=0
 population_mode='open'
 save=True
 commit_recheck=False
 fit_first_resume=False
 capacity_victim=True
 recovery_min_outputs=1
 yield_to_ready_head=False
 spare_followup={enabled}
 block_size=16
 oldcalc=cs._calc_num_offloadable_tokens
 hadcalc=True
''').body[0]
    seed.body+=deepcopy(install.body[start:])
    ns=vars(adapter).copy();exec(compile(ast.fix_missing_locations(ast.Module(body=[seed],type_ignores=[])),'<actual-spare-closures>','exec'),ns)
    return ns['fixture']
q.fixture=fixture
for mode in ('admitted','unknown_reserved','unfunded','pending','head_fits','off','not_admitted'):
    enabled=mode!='off'
    c=q.recovered_case(1);q.output(c,1)
    cs=c.cells['cs'].cell_contents;cs.has_pending_push_work=lambda:False
    step=c.cells['step'].cell_contents
    for rid,prompt,age in [('blocked_oldest',1600,60),('follower',16,40)]:
        req=NS(request_id=rid,num_prompt_tokens=prompt,num_output_tokens=1,num_computed_tokens=0,
            max_tokens=1024,status=NS(name='PREEMPTED'),is_finished=lambda:False)
        c.reqs[rid]=req;c.owned[rid]=[];c.s.waiting.append(req)
        cs._req_status[rid]=NS(transfer_jobs=set())
        c.cells['tracker'].cell_contents.absent_since[rid]=step-age
    if mode=='unknown_reserved':c.s._inflight_prefill_reserved_blocks=lambda:None
    if mode=='unfunded':c.reqs['follower'].num_prompt_tokens=6400
    if mode=='pending':cs._jobs['pending']=NS()
    if mode=='head_fits':
        c.s.waiting[0].num_prompt_tokens=0;c.s.waiting[0].num_output_tokens=1
    if mode=='not_admitted':
        def native():
            c.s._rotation_begin([],0.)
            return NS(num_scheduled_tokens={'t':1,'old':1},preempted_req_ids=set(),
                scheduled_cached_reqs=NS(resumed_req_ids=[]),
                kv_connector_metadata=NS(store_jobs={},load_jobs={},jobs_to_flush=set()))
        c.schedule_cells['native'].cell_contents=native
        try:c.s.schedule()
        except RuntimeError as exc:assert 'Ready target made no progress' in str(exc)
        else:raise AssertionError('Missing native admission did not stop')
    else:c.s.schedule()
    choices=[e for e in c.data['events'] if e['event']=='spare_followup_choice']
    receipts=[e for e in c.data['events'] if e['event']=='spare_followup_admission']
    if mode in ('admitted','not_admitted'):
        assert len(choices)==len(receipts)==1
        choice=choices[0];assert choice['primary_target']=='t' and choice['target']=='follower'
        assert choice['displaced_oldest']=='blocked_oldest' and choice['primary_start_step']==546
        assert receipts[0]['native_admission']==('SCHEDULED_TOKENS' if mode=='admitted' else 'NO_NATIVE_ADMISSION')
        release=next(e for e in c.data['events'] if e['event']=='protection_release' and e['request']=='t')
        assert release['new_output_tokens']==1 and release['protection_origin']=='REGULAR_COMMIT'
        assert c.data['applied_rotations']==1
        if mode=='admitted':
            follower=c.reqs['follower'];follower.num_output_tokens+=1
            follower.num_computed_tokens=follower.num_prompt_tokens+follower.num_output_tokens-1
            c.s.schedule()
            assert len([e for e in c.data['events'] if e['event']=='spare_followup_choice'])==1
            assert any(e['event']=='protection_release' and e['protection_origin']=='SPARE_FOLLOWUP' for e in c.data['events'])
    else:assert not choices and not receipts
    c.undo()
print('7 actual-closure cases passed, including no chain and explicit failed admission; no GPU claim')
