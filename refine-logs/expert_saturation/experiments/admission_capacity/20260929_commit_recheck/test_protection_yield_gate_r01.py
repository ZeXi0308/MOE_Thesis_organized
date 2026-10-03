"""Targeted CPU closure checks; not native admission or performance evidence."""
import ast
from copy import deepcopy
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS

ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('qfixture',ROOT/'protection_quantity_probe_r02/test_protection_quantity_cpu.py')
q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q)
source=ROOT/'candidate_protection_yield_r01/pkg/staged_store_rotation.py'
spec=importlib.util.spec_from_file_location('yield_adapter',source)
adapter=importlib.util.module_from_spec(spec);spec.loader.exec_module(adapter)

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
 recovery_min_outputs={quantity}
 yield_to_ready_head=True
 block_size=16
 oldcalc=cs._calc_num_offloadable_tokens
 hadcalc=True
''').body[0]
    seed.body+=deepcopy(install.body[start:])
    ns=vars(adapter).copy();exec(compile(ast.fix_missing_locations(ast.Module(body=[seed],type_ignores=[])),'<actual-yield-closures>','exec'),ns)
    return ns['fixture']
q.fixture=fixture
for mode in ('admitted','not_admitted','unfunded','pending'):
    c=q.recovered_case(10);q.output(c,1)
    head=NS(request_id='head',num_prompt_tokens=16,num_output_tokens=2,num_tokens=18,
        num_computed_tokens=0,max_tokens=1024,status=NS(name='PREEMPTED'),is_finished=lambda:False)
    c.reqs['head']=head;c.owned['head']=[];c.s.waiting.prepend_request(head)
    cs=c.cells['cs'].cell_contents;cs._req_status['head']=NS(transfer_jobs=set());cs.has_pending_push_work=lambda:False
    if mode=='unfunded':head.num_tokens=10000;head.num_prompt_tokens=9998
    if mode=='pending':cs._jobs['pending']=NS()
    def native():
        c.s._rotation_begin([],0.0)
        scheduled={'t':1}
        if c.s._rotation_yield_to_ready_head(head,1000) and mode=='admitted':
            scheduled['head']=16;head.status=NS(name='RUNNING')
        return NS(num_scheduled_tokens=scheduled,preempted_req_ids=set(),
            scheduled_cached_reqs=NS(resumed_req_ids=[]),
            kv_connector_metadata=NS(store_jobs={},load_jobs={},jobs_to_flush=set()))
    c.schedule_cells['native'].cell_contents=native
    c.s.schedule()
    receipts=[e for e in c.data['events'] if e['event']=='yield_head_admission']
    if mode in ('admitted','not_admitted'):
        assert c.data['yield_releases']==1 and c.cells['protected'].cell_contents is None
        assert receipts[0]['native_admission']==('SCHEDULED_TOKENS' if mode=='admitted' else 'NO_NATIVE_ADMISSION')
        assert next(e for e in c.data['events'] if e['event']=='protection_release')['reason']=='YIELD_TO_READY_HEAD'
    else:
        assert c.data['yield_releases']==0 and not receipts and c.cells['protected'].cell_contents is c.target
    c.undo()
print('4 targeted CPU closure cases passed; no GPU result')
