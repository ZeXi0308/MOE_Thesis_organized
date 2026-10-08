"""Actual adapter closures + pinned native AST; CPU checks, GPU UNRUN."""
import ast
from copy import deepcopy
from pathlib import Path
import sys
import time
from types import SimpleNamespace as NS
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'pkg'))
import staged_store_rotation as adapter
from rotation_native import patched_schedule_tree


class Queue(list):
    def remove_request(self, request): self.remove(request)
    def prepend_request(self, request): self.insert(0, request)
    def pop_request(self): return self.pop(0)
    def peek_request(self): return self[0]


class Request:
    def __init__(self, rid, prompt, output, status):
        self.request_id=rid;self.num_prompt_tokens=prompt;self.num_output_tokens=output
        self.num_computed_tokens=prompt+output-1 if status=='RUNNING' else 0
        self.max_tokens=100;self.status=NS(name=status);self.num_preemptions=int(status=='PREEMPTED')
        self.arrival_time=0
    @property
    def num_tokens(self): return self.num_prompt_tokens+self.num_output_tokens
    def is_finished(self): return self.status.name.startswith('FINISHED')


def closures():
    install=next(n for n in ast.parse((HERE/'pkg/staged_store_rotation.py').read_text()).body
                 if isinstance(n,ast.FunctionDef) and n.name=='install')
    start=next(i for i,n in enumerate(install.body) if isinstance(n,ast.Assign)
               and any(isinstance(t,ast.Name) and t.id=='step' for t in n.targets))
    seed=ast.parse('''def fixture(scheduler,native,owned,pool,cs,mode):
 manager=scheduler.kv_cache_manager
 expected_requests=32
 open_population=True
 diagnostic=False
 store_scope='native_full'
 global_cooldown_steps=0
 population_mode='open'
 save=True
 commit_recheck=False
 fit_first_resume=False
 capacity_victim=False
 recovery_min_outputs=1
 yield_to_ready_head=False
 spare_followup=False
 ordinary_backfill=False
 allow_forced_rotations=True
 oldest_admission_mode='queue_fund'
 oldest_repeat=True
 block_size=16
 lease_mode=mode
 lease_enabled=True
 oldcalc=lambda rs,n:n
 hadcalc=False
''').body[0]
    seed.body+=deepcopy(install.body[start:])
    namespace=vars(adapter).copy()
    namespace['time']=NS(perf_counter=lambda:100.)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[seed],type_ignores=[])),
                 '<actual-native-lease-closures>', 'exec'),namespace)
    return namespace['fixture']


def case(mode='adaptive'):
    victim=Request('victim',96,10,'RUNNING')
    target=Request('target',205,3,'PREEMPTED')
    requests={r.request_id:r for r in (victim,target)}
    blocks={i:NS(block_id=i,is_null=False,ref_cnt=int(i<=7)) for i in range(1,16)}
    owned={'victim':[blocks[i] for i in range(1,8)],'target':[]}
    pool=NS(blocks=blocks,num_gpu_blocks=len(blocks)+1,
            get_num_free_blocks=lambda:sum(b.ref_cnt==0 for b in blocks.values()))
    manager=NS(block_pool=pool,get_blocks=lambda rid:NS(get_block_ids=lambda:
                [[b.block_id for b in owned.get(rid,())]]))
    scheduler=NS(requests=requests,running=[victim],waiting=Queue([target]),skipped_waiting=Queue(),
        max_num_running_reqs=32,num_waiting_for_streaming_input=0,kv_cache_manager=manager,
        _inflight_prefills=[])
    scheduler._request_remaining_blocks=lambda r:max(0,(r.num_tokens+15)//16-len(owned.get(r.request_id,())))
    scheduler._inflight_prefill_reserved_blocks=lambda:sum(scheduler._request_remaining_blocks(r)
                                                         for r in scheduler._inflight_prefills)
    cs=NS(_req_status={rid:NS(transfer_jobs=set()) for rid in requests},_jobs={},
          has_pending_push_work=lambda:False,manager=NS(has_pending_work=lambda:False),
          _current_batch_load_jobs=[],_current_batch_jobs_to_flush=[])
    def preempt(req,timestamp):
        for block in owned[req.request_id]:block.ref_cnt=0
        owned[req.request_id]=[];req.status=NS(name='PREEMPTED');req.num_computed_tokens=0
        req.num_preemptions+=1;scheduler.waiting.prepend_request(req)
    scheduler._preempt_request=preempt
    trace=[]
    def allocate(req):
        need=scheduler._request_remaining_blocks(req)
        free=[b for b in blocks.values() if b.ref_cnt==0]
        assert len(free)>=need,'fixture would preempt under insufficient reservation'
        for block in free[:need]:block.ref_cnt=1
        owned.setdefault(req.request_id,[]).extend(free[:need])
    def native():
        preempted=[];scheduler._rotation_begin(preempted,0.)
        scheduled={};budget=1024;resumed=[]
        for req in list(scheduler.running):
            if scheduler._rotation_hold(req):continue
            n=scheduler._rotation_token_limit(req,1,budget)
            if not n:continue
            allocate(req);scheduled[req.request_id]=n;budget-=n
        deferred=[]
        while scheduler.waiting or scheduler.skipped_waiting:
            default=scheduler.skipped_waiting or scheduler.waiting
            queue=scheduler._rotation_waiting_queue(default);req=queue.peek_request()
            if not scheduler._rotation_waiting_allowed(req,budget):
                deferred.append(queue.pop_request());continue
            if pool.get_num_free_blocks()<scheduler._request_remaining_blocks(req):break
            allocate(req);queue.pop_request();scheduler._rotation_admit_running(req)
            n=scheduler._rotation_token_limit(req,1,budget)
            assert n>0
            scheduled[req.request_id]=n;budget-=n;resumed.append(req.request_id)
            req.status=NS(name='RUNNING');req.num_computed_tokens=req.num_tokens-1
        scheduler.skipped_waiting.extend(deferred)
        trace.append(dict(scheduled=scheduled.copy(),free=pool.get_num_free_blocks()))
        return NS(num_scheduled_tokens=scheduled,preempted_req_ids={r.request_id for r in preempted},
            scheduled_cached_reqs=NS(resumed_req_ids=resumed),
            kv_connector_metadata=NS(store_jobs={},load_jobs={},jobs_to_flush=set()))
    data,undo=closures()(scheduler,native,owned,pool,cs,mode)
    cells={}
    pending=[scheduler.schedule,scheduler._rotation_begin,scheduler._rotation_hold,
             scheduler._rotation_waiting_allowed,scheduler._rotation_token_limit];visited=set()
    while pending:
        function=pending.pop()
        if id(function) in visited:continue
        visited.add(id(function))
        if not getattr(function,'__closure__',None):continue
        nested=dict(zip(function.__code__.co_freevars,function.__closure__))
        cells.update(nested)
        pending.extend(c.cell_contents for c in nested.values() if callable(c.cell_contents)
                       and hasattr(c.cell_contents,'__closure__'))
    now=100.
    cells['last_output_observed_perf'].cell_contents.update(target=now-2.,victim=now)
    cells['lease_decode_samples'].cell_contents.append(.01)
    cells['lease_overhead_samples'].cell_contents.append(.06)
    return NS(s=scheduler,target=target,victim=victim,data=data,undo=undo,cells=cells,
              owned=owned,pool=pool,trace=trace)


def emit(req):
    req.num_output_tokens+=1;req.num_computed_tokens+=1


class NativeLeaseTests(unittest.TestCase):
    def test_actual_anchor_commit_whole_horizon_and_release(self):
        c=case();c.s.schedule()
        anchor=next(e for e in c.data['events'] if e['event']=='oldest_anchor')
        self.assertEqual(anchor['lease_decision']['quantum'],6)
        self.assertEqual(anchor['lease_decision']['required_total_blocks'],14)
        emit(c.victim);c.s.schedule()
        self.assertEqual(c.cells['protection_output_goal'].cell_contents,9)
        self.assertEqual(c.pool.get_num_free_blocks(),2)
        # The first protected decode crosses a page boundary. The last page is
        # reserved before Q1, and the old victim cannot take its seven blocks.
        emit(c.target);c.s.schedule()
        self.assertEqual(c.pool.get_num_free_blocks(),1)
        self.assertNotIn('victim',c.trace[-1]['scheduled'])
        self.assertEqual(c.trace[-1]['scheduled']['target'],1)
        for _ in range(4):emit(c.target);c.s.schedule()
        self.assertIs(c.cells['protected'].cell_contents,c.target)
        emit(c.target);c.s.schedule()
        self.assertIsNone(c.cells['protected'].cell_contents)
        release=next(e for e in c.data['events'] if e['event']=='protection_release')
        self.assertEqual((release['reason'],release['new_output_tokens']),('OUTPUT_GOAL_REACHED',6))

    def test_waiting_peer_can_use_spare_blocks_but_cannot_take_lease_growth(self):
        c=case();c.s.schedule();emit(c.victim);c.s.schedule()
        peer=Request('peer',15,1,'PREEMPTED');c.s.requests['peer']=peer;c.owned['peer']=[]
        self.assertTrue(c.s._rotation_waiting_allowed(peer,100))
        peer.num_prompt_tokens=31
        self.assertFalse(c.s._rotation_waiting_allowed(peer,100))
        self.assertEqual(c.data['lease_waiting_gate_counts']['KEEP_LEASE_GROWTH_RESERVED'],2)

    def test_other_inflight_reservation_is_preserved_without_double_counting(self):
        c=case();c.s.schedule();emit(c.victim);c.s.schedule()
        peer=Request('peer',15,1,'PREEMPTED');other=Request('load',15,1,'WAITING_FOR_REMOTE_KVS')
        c.s.requests.update(peer=peer,load=other);c.owned.update(peer=[],load=[])
        c.s._inflight_prefills=[c.target,peer]
        self.assertTrue(c.s._rotation_waiting_allowed(peer,100))
        c.s._inflight_prefills.append(other)
        self.assertFalse(c.s._rotation_waiting_allowed(peer,100))

    def test_target_budget_and_skipped_queue_priority(self):
        c=case();c.s.schedule();emit(c.victim);c.s.schedule()
        c.s.running.remove(c.target);c.s.waiting.prepend_request(c.target)
        c.s.skipped_waiting.prepend_request(c.victim)
        c.cells['lease_target_scheduled'].cell_contents=False
        self.assertIs(c.s._rotation_waiting_queue(c.s.skipped_waiting),c.s.waiting)
        self.assertEqual(c.s._rotation_token_limit(c.victim,1024,1024),1023)
        self.assertFalse(c.s._rotation_waiting_allowed(c.victim,1))
        self.assertEqual(c.s._rotation_token_limit(c.target,1,1),1)

    def test_commit_rechecks_quantum_and_cost_modes_share_frontier(self):
        for mode,want in (('adaptive',6),('fixed4',4),('q1',1)):
            c=case(mode);c.s.schedule()
            self.assertEqual(c.cells['lease_plan_decision'].cell_contents.quantum,want)
        c=case();c.s.schedule();emit(c.victim)
        # A newly stalled peer has no extra age headroom at the commit boundary.
        c.cells['last_output_observed_perf'].cell_contents['victim']=100.-1.995
        c.cells['last_output_counts'].cell_contents['victim']=c.victim.num_output_tokens
        c.s.schedule()
        commit=next(e for e in c.data['events'] if e['event']=='lease_commit_decision')
        self.assertEqual(commit['lease_decision']['quantum'],1)

    def test_peer_frontier_releases_extra_lease(self):
        c=case();c.s.schedule();emit(c.victim);c.s.schedule();emit(c.target)
        c.cells['last_output_observed_perf'].cell_contents['victim']=100.-3.
        c.s.schedule()
        release=next(e for e in c.data['events'] if e['event']=='protection_release')
        self.assertEqual(release['reason'],'RELEASE_PEER_AGE_FRONTIER')

    def test_async_target_retains_last_running_slot(self):
        c=case();c.s.schedule();emit(c.victim);c.s.schedule()
        c.target.status=NS(name='WAITING_FOR_REMOTE_KVS')
        c.s.running=[Request('filler'+str(i),15,1,'RUNNING') for i in range(31)]
        peer=Request('peer',15,1,'PREEMPTED');c.owned['peer']=[]
        self.assertFalse(c.s._rotation_waiting_allowed(peer,100))
        self.assertEqual(c.data['lease_waiting_gate_counts']['KEEP_TARGET_RUNNING_SLOT'],1)
        c.s.running.pop()
        self.assertTrue(c.s._rotation_waiting_allowed(peer,100))

    def test_lease_deferred_peer_returns_to_ordinary_waiting_for_next_recovery(self):
        c=case('q1');c.s.schedule();emit(c.victim);c.s.schedule()
        self.assertIn(c.victim,c.s.skipped_waiting)
        c.cells['last_output_observed_perf'].cell_contents['victim']=97.
        emit(c.target);c.s.schedule()
        anchors=[e for e in c.data['events'] if e['event']=='oldest_anchor']
        self.assertEqual(len(anchors),2)
        self.assertEqual(anchors[-1]['target'],'victim')
        self.assertNotIn(c.victim,c.s.skipped_waiting)

    def test_pinned_native_ast_has_both_compute_limits_and_selective_defer(self):
        source=HERE.parent.parent / '20260929_commit_recheck'/'liveness_pinned_sources_20260930/scheduler.py'
        tree=patched_schedule_tree(source.read_text())
        compile(tree,'<pinned-native-scheduler-with-lease>','exec')
        calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
        self.assertEqual(sum(n.func.attr=='_rotation_token_limit' for n in calls),2)
        self.assertEqual(sum(n.func.attr=='_rotation_waiting_allowed' for n in calls),1)


if __name__=='__main__':unittest.main()
