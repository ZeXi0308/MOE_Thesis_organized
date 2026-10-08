"""Bounded CPU execution of pinned native waiting fragments; no vLLM/CUDA."""
import ast
import copy
import hashlib
from pathlib import Path
import runpy
import sys
from types import MethodType, SimpleNamespace as NS

import yield_once as Y

B = Path(__file__).resolve().parents[1]
fixture_path = B/'recovery_queue/check_cpu.py'
assert hashlib.sha256(fixture_path.read_bytes()).hexdigest() == '702e80971cbff2c193980dd29d82aa48d38584634b73bd96db681ddcb6000506'
sys.path.insert(0, str(fixture_path.parent))
F = runpy.run_path(str(fixture_path))
full_tree = F['full_tree']


class RemoveCallbacks(ast.NodeTransformer):
    def visit_Expr(self, node):
        if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == Y.H.CALLBACK:
            return None
        return self.generic_visit(node)

    def visit_If(self, node):
        if isinstance(node.test, ast.Call) and isinstance(node.test.func, ast.Name) and node.test.func.id == Y.H.CALLBACK:
            assert isinstance(node.body[-1], ast.Break) and len(node.body) == 2
            return None
        return self.generic_visit(node)


instrumented = Y._instrument(full_tree)
assert ast.dump(RemoveCallbacks().visit(copy.deepcopy(instrumented)), include_attributes=True) == ast.dump(full_tree, include_attributes=True)
compile(instrumented, str(F['SCHEDULER']), 'exec', dont_inherit=True)

# Keep the real native queue selection, blocked-request handling, allocation,
# connector registration, async branch and final skipped-queue merge. Replace
# only model lookup/token computation with explicit CPU inputs and the forward
# result with a returned snapshot of the fake connector's native metadata.
outer = copy.deepcopy(next(n for n in full_tree.body[0].body if isinstance(n, ast.If)
    and ast.unparse(n.test).startswith('len(preempted_reqs) == self._rotation_forced_count')))
loop = next(n for n in outer.body if isinstance(n, ast.While))
blocked = next(i for i,n in enumerate(loop.body) if isinstance(n, ast.If)
    and ast.unparse(n.test).startswith('self._is_blocked_waiting_status'))
allocation = next(i for i,n in enumerate(loop.body) if isinstance(n, ast.Assign)
    and ast.unparse(n.targets[0]) == 'new_blocks')
async_branch = next(i for i,n in enumerate(loop.body) if isinstance(n, ast.If)
    and ast.unparse(n.test) == 'load_kv_async' and any(isinstance(v, ast.Continue) for v in n.body))
preparation = ast.parse('''
load_kv_async = request.request_id == 'early'
num_external_computed_tokens = 32 if load_kv_async else 0
num_new_tokens = 0 if load_kv_async else min(16, token_budget)
num_new_local_computed_tokens = 0
num_computed_tokens = num_external_computed_tokens
new_computed_blocks = ()
effective_lookahead_tokens = num_encoder_tokens = reserved_blocks = 0
''').body
loop.body = loop.body[:blocked+1] + preparation + loop.body[allocation:async_branch+1] + ast.parse('''
self.running.append(request)
token_budget -= num_new_tokens
request.status = RequestStatus.RUNNING
''').body
merge = next(n for n in outer.body if isinstance(n, ast.If) and ast.unparse(n.test) == 'step_skipped_waiting')
outer.body = outer.body[:outer.body.index(loop)+1] + [merge]
fragment = ast.parse('def schedule(self, token_budget, preempted_reqs):\n    pass\n')
fragment.body[0].body = [outer] + ast.parse('return self.connector.build_connector_meta()').body
ast.fix_missing_locations(fragment)


class Queue(F['Queue']):
    def pop_request(self): return self.pop(0)
    def prepend_request(self, request): self.insert(0, request)
    def prepend_requests(self, requests): self[:0] = list(requests)


def fixture():
    s, _, manager, single, cs = F['fixture']()
    head = s.waiting[0]
    early = F['request']('early', 0)
    s.waiting = Queue([early, head]); s.skipped_waiting = Queue()
    statuses = NS(**{name:NS(name=name) for name in ('PREEMPTED','RUNNING','WAITING_FOR_REMOTE_KVS','WAITING')})
    s._is_blocked_waiting_status = lambda status: status.name == 'WAITING_FOR_REMOTE_KVS'
    s._try_promote_blocked_waiting_request = lambda r: False
    s.scheduler_reserve_full_isl = True
    s.needs_kv_cache_zeroing = False
    inflight = []; s._inflight_prefills = NS(add=inflight.append)
    cs._req_status = {r.request_id:NS(req=r, transfer_jobs=set(), group_states=[NS(offload_keys=list(range(4)))]) for r in (early,head)}
    cs._jobs = {}; cs._current_batch_load_jobs = {}
    cs.manager._policy.blocks.clear()  # Current head's original lookup is a MISS.
    events = []; refs = [0]
    def allocate(request, num_new_tokens, **kwargs):
        events.append(('allocate',request.request_id,num_new_tokens,kwargs))
        blocks = [NS(is_null=False,ref_cnt=1)] * (2 if kwargs['delay_cache_blocks'] else 4)
        single.req_to_blocks[request.request_id] = blocks
        return blocks
    manager.allocate_slots = allocate
    manager.get_blocks = lambda rid: single.req_to_blocks[rid]
    def update(request, blocks, external):
        events.append(('update',request.request_id,external))
        if external:
            refs[0] += 1
            cs._req_status[request.request_id].transfer_jobs.add(7)
            cs._jobs[7] = NS(req_id=request.request_id,is_store=False,pending_count=1)
            cs._current_batch_load_jobs[7] = NS(req_id=request.request_id)
    def metadata():
        result = sorted(cs._current_batch_load_jobs)
        cs._current_batch_load_jobs.clear()
        events.append(('metadata',result))
        return result
    s.kv_cache_manager = manager
    s.connector = NS(update_state_after_alloc=update, build_connector_meta=metadata)
    s.connector_prefix_cache_stats = None
    env = dict(PauseState=NS(UNPAUSED=s._pause_state), RequestStatus=statuses,
        create_request_queue=lambda _:Queue(), logger=NS(debug=lambda *args:None))
    exec(compile(fragment,str(F['SCHEDULER']),'exec',dont_inherit=True),env)
    native = MethodType(env['schedule'],s)
    def outer_wrapper():
        def schedule(*args,**kwargs): return native(*args,**kwargs)
        return schedule
    s.schedule = outer_wrapper()
    return s, manager, single, cs, head, early, events, refs


for mode in ('native','yield_once'):
    s,manager,single,cs,head,early,events,refs = fixture()
    wrapper = s.schedule; slot = Y.H._native_slot(wrapper,s); original = slot.cell_contents
    ticks = iter(range(100,200))
    data,undo = Y._attach(s,manager,single,cs,mode,{},fragment,now=lambda:next(ticks))
    assert s.schedule is wrapper
    assert s.schedule(16,[]) == [7]  # Earlier native LOAD still becomes metadata.
    row = data['events'][0]
    assert row['deferred_request']=='head' and row['local_skipped_order']==['early']
    assert row['public_skipped_order']==[] and row['beneficiaries'][0]['job_ids']==[7]
    assert row['beneficiaries'][0]['current_batch_job_ids']==[7]
    assert row['beneficiaries'][0]['locations']==['local_skipped']
    assert row['native_locals']['num_external_computed_tokens']==0
    assert refs==[1] and cs._req_status['early'].transfer_jobs=={7}
    assert not cs._req_status['head'].transfer_jobs
    assert [r.request_id for r in s.skipped_waiting]==['early']
    calls = [e[1] for e in events if e[0]=='allocate']
    assert calls == (['early','head'] if mode=='native' else ['early'])
    assert data['executed_breaks']==data['action_count']==int(mode=='yield_once')
    assert s.schedule(16,[]) == []
    assert [e[1] for e in events if e[0]=='allocate']==['early','head']
    assert [r.request_id for r in s.running]==['head']
    assert data['events'][1]['kind']=='next_schedule_entry_pass' and data['events'][1]['step']==2
    assert refs==[1] and cs._req_status['early'].transfer_jobs=={7}
    s.schedule(16,[])
    assert len(data['events'])==2 and data['executed_breaks']==int(mode=='yield_once')
    # A native completion still owns exactly the one LOAD reference.
    refs[0]-=1; cs._req_status['early'].transfer_jobs.remove(7); del cs._jobs[7]
    assert refs==[0]
    undo(); undo()
    assert slot.cell_contents is original and s.schedule is wrapper and data['status']=='UNINSTALLED'

# Direct decision tests isolate conservative rejection without invoking lookup,
# allocation or any connector reference operation.
def decision_fixture():
    s,m,one,cs,head,early,events,refs = fixture()
    s.waiting=Queue([head]); early.status=NS(name='WAITING_FOR_REMOTE_KVS')
    local=Queue([early]); cs._req_status['early'].transfer_jobs={7}
    cs._jobs[7]=NS(req_id='early',is_store=False,pending_count=1)
    data,callback=Y._decision(s,m,one,cs,'yield_once',{},now=lambda:1.0)
    callback(s,'entry',16,[])
    def decision(**overrides):
        args=dict(queue=s.waiting,request=head,local=local,new_tokens=16,external=0,
                  asynchronous=False,local_tokens=0,lookahead=0,encoder_tokens=0)
        args.update(overrides)
        return callback(s,'decision',16,[],**args)
    return s,m,one,cs,head,early,local,data,decision

for invalid in (dict(external=16),dict(asynchronous=True),dict(new_tokens=0)):
    *_,data,decide=decision_fixture()
    assert not decide(**invalid) and not data['events']
    assert decide() and len(data['events'])==1
s,m,one,cs,head,early,local,data,decide=decision_fixture()
cs._req_status['head'].transfer_jobs={9}
assert not decide(); cs._req_status['head'].transfer_jobs.clear()
cs._jobs[7].is_store=True
assert not decide(); cs._jobs[7].is_store=False
cs._jobs[7].pending_count=0
assert not decide(); cs._jobs[7].pending_count=1
cs._req_status['early'].req=copy.copy(early)
assert not decide(); cs._req_status['early'].req=early
m.block_pool.free_block_queue.num_free_blocks=3
assert not decide(); m.block_pool.free_block_queue.num_free_blocks=8
s.waiting.append(F['request']('new',3)); s.waiting[-1].status=NS(name='WAITING')
assert not decide(); s.waiting.pop()
s.skipped_waiting=Queue(local); local.clear()
assert decide() and data['events'][0]['beneficiaries'][0]['locations']==['public_skipped']

# Original allocator exceptions propagate and wrapper removal restores identity.
s,m,one,cs,head,early,events,refs=fixture()
slot=Y.H._native_slot(s.schedule,s); original=slot.cell_contents
data,undo=Y._attach(s,m,one,cs,'yield_once',{},fragment)
def fail(*args,**kwargs): raise ValueError('native allocation error')
m.allocate_slots=fail
try: s.schedule(16,[])
except ValueError as error: assert str(error)=='native allocation error'
else: raise AssertionError('Native allocator exception swallowed')
assert not data['events'] and refs==[0]
undo(); assert slot.cell_contents is original
print('PASS: full native AST restoration; native fragments preserve earlier LOAD metadata/ref ownership; local/public skipped; one real break then native; conservative exclusions; native exceptions and uninstall. GPU_UNRUN')
