"""Bounded exchange tests with exact pinned native preempt/metadata/flush methods."""
import ast
import hashlib
from pathlib import Path
from types import SimpleNamespace as NS, MethodType

import exchange_once as p

B = Path(__file__).resolve().parents[1]
SCHED = B.parent/'MOE_Thesis_organized/refine-logs/expert_saturation/experiments/admission_capacity/20260929_commit_recheck/liveness_pinned_sources_20260930/scheduler.py'


def native_method(path, sha, cls, name, namespace):
    source = path.read_bytes(); assert hashlib.sha256(source).hexdigest() == sha
    tree = ast.parse(source)
    method = next(n for c in tree.body if isinstance(c,ast.ClassDef) and c.name==cls
                  for n in c.body if isinstance(n,ast.FunctionDef) and n.name==name)
    method.decorator_list = []
    ns = dict(namespace)
    text = ast.unparse(ast.Module(body=[method],type_ignores=[]))
    exec('from __future__ import annotations\n'+text,ns)
    return ns[name]


STATUS = NS(**{x:NS(name=x) for x in ('RUNNING','PREEMPTED','WAITING','WAITING_FOR_REMOTE_KVS')})
preempt = native_method(SCHED,p.CAPACITY.SOURCE_SHA256['scheduler.py'],'Scheduler','_preempt_request',{'RequestStatus':STATUS})


class Queue(list):
    def peek_request(self): return self[0]
    def remove_request(self,r): self.remove(r)
    def prepend_request(self,r): self.insert(0,r)


class Request:
    def __init__(self,rid,n,h,status,receipt):
        self.request_id=rid; self.num_tokens=n; self.num_computed_tokens=n-1 if h else 0
        self.num_prompt_tokens=32; self.num_output_tokens=max(n-32,1);self.max_tokens=1024
        self.num_in_flight_tokens=0;self.num_output_placeholders=0;self.spec_token_ids=[]
        self.has_encoder_inputs=False;self.is_prefill_chunk=False;self.num_tokens_with_spec=n
        self.status=status;self.num_preemptions=1 if status.name=='PREEMPTED' else 0
        self.priority=0;self.arrival_time=receipt;self.skip_reading_prefix_cache=False
    def is_finished(self): return self.status.name.startswith('FINISHED')


def fixture(mode='exchange_once', free=4):
    t=Request('target',160,0,STATUS.PREEMPTED,1.)
    d=Request('donor',96,6,STATUS.RUNNING,4.)
    bigger=Request('larger',128,8,STATUS.RUNNING,5.)
    peer=Request('peer',64,4,STATUS.RUNNING,6.)
    owned={};blocks=[NS(block_id=0,is_null=True,ref_cnt=1)]
    for r,h in ((t,0),(d,6),(bigger,8),(peer,4)):
        owned[r.request_id]=[]
        for _ in range(h):
            block=NS(block_id=len(blocks),is_null=False,ref_cnt=1)
            blocks.append(block);owned[r.request_id].append(block)
    pool=NS(blocks=blocks,free=free)
    pool.get_num_free_blocks=lambda:pool.free
    pool.free_block_queue=NS(num_free_blocks=free)
    single=NS(block_size=16,req_to_blocks=owned)
    manager=NS(block_pool=pool,max_model_len=4096)
    reads={}
    def get_blocks(rid):
        reads[rid]=reads.get(rid,0)+1
        return NS(get_block_ids=lambda:[[b.block_id for b in owned.get(rid,())]])
    manager.get_blocks=get_blocks
    cache={};states={}
    for r in (t,d,bigger,peer):
        keys=[r.request_id+str(i) for i in range(r.num_tokens//16)]
        for k in keys:cache[k]=NS(ref_cnt=0,is_ready=True)
        states[r.request_id]=NS(req=r,transfer_jobs=set(),group_states=[NS(offload_keys=keys,
            block_ids=[b.block_id for b in owned[r.request_id]])])
    cs=NS(_req_status=states,_jobs={},_chunks_being_loaded=set(),manager=NS(_policy=NS(blocks=cache)))
    s=NS(requests={r.request_id:r for r in (t,d,bigger,peer)},running=[d,bigger,peer],waiting=Queue([t]),
        skipped_waiting=Queue(),_inflight_prefills=set(),_rotation_target=None,_rotation_forced_count=0,
        _rotation_lease_enabled=False,_pause_state=NS(name='UNPAUSED'),num_waiting_for_streaming_input=0,
        max_num_running_reqs=256,max_num_scheduled_tokens=1024,reset_preempted_req_ids=set(),log_stats=False)
    s.encoder_cache_manager=NS(free=lambda req:None)
    def release(req):
        for b in owned.pop(req.request_id,[]):
            b.ref_cnt-=1
            if b.ref_cnt==0:pool.free+=1
        pool.free_block_queue.num_free_blocks=pool.free
    s._free_request_blocks=release
    s._preempt_request=MethodType(preempt,s)
    s._request_remaining_blocks=lambda req:max((req.num_tokens+15)//16-len(owned.get(req.request_id,())),0)
    s._inflight_prefill_reserved_blocks=lambda:sum(s._request_remaining_blocks(r) for r in s._inflight_prefills)
    calls=[]
    def begin(preempted,stamp):
        calls.append('native_begin');s._rotation_forced_count=0;s._rotation_target=None
    s._rotation_begin=begin;s._rotation_hold=lambda req:False
    baseline={'action_count':2,'bypassed_episodes':[{'request':'prior','num_preemptions':1}]}
    def schedule():
        preempted=[];s._rotation_begin(preempted,10.)
        return NS(num_scheduled_tokens={},preempted_req_ids=[r.request_id for r in preempted])
    s.schedule=schedule;old=(s._rotation_begin,s._rotation_hold,s.schedule)
    receipts={r.request_id:r.arrival_time for r in (t,d,bigger,peer)}
    data,undo=p._attach(s,manager,single,cs,mode,receipts,baseline,lambda:baseline)
    return NS(s=s,t=t,d=d,bigger=bigger,peer=peer,pool=pool,owned=owned,cs=cs,receipts=receipts,
        data=data,undo=undo,old=old,baseline=baseline,calls=calls,reads=reads)


original_clock=p.time.perf_counter
p.time.perf_counter=lambda:100.
try:
    # Both arms observe identical legal decision; native stall8 is untouched.
    a=fixture('stall8');b=fixture()
    host_refs={k:v.ref_cnt for k,v in b.cs.manager._policy.blocks.items()}
    a.s.schedule();out=b.s.schedule()
    da=next(e for e in a.data['events'] if e['kind']=='decision')
    db=next(e for e in b.data['events'] if e['kind']=='decision')
    assert da['targets']==db['targets'] and da['donors']==db['donors']
    assert a.data['action_count']==0 and b.data['action_count']==1
    assert b.s.running==[b.bigger,b.peer] and out.preempted_req_ids==['donor']
    assert b.s.waiting==[b.t,b.d] and b.s._rotation_forced_count==1
    assert b.d.num_computed_tokens==0 and b.d.status is STATUS.PREEMPTED
    assert b.d.request_id in b.s.reset_preempted_req_ids
    assert b.pool.free==10 and not b.owned.get('donor')
    assert b.reads=={'target':2,'donor':1,'larger':1,'peer':1}
    assert all(x.ref_cnt==0 for x in b.pool.blocks[1:7])
    assert host_refs=={k:v.ref_cnt for k,v in b.cs.manager._policy.blocks.items()}
    assert db['selected_capacity']['target_total_blocks']==10
    assert db['selected_capacity']['inflight_remaining_blocks']==0
    assert db['selected_capacity']['running_increment_blocks']==0
    # At the following page boundary peers are held only if native need would
    # consume the still-unmaterialized target's reservation, not all peers.
    assert not b.s._rotation_hold(b.peer)
    b.peer.num_tokens=65;b.peer.num_tokens_with_spec=65;b.peer.num_computed_tokens=64
    assert b.s._rotation_hold(b.peer)
    b.pool.free=11;assert not b.s._rotation_hold(b.peer)
    b.receipts['target']=99.
    b.s.schedule();assert b.s._rotation_target is None
    assert next(e for e in b.data['events'] if e['kind']=='release')['reason']=='FIRST_CLIENT_RECEIPT'
    for _ in range(3):b.s.schedule()
    assert b.data['action_count']==1 and b.baseline['action_count']==2
    a.undo();b.undo();b.undo();assert b.data['status']=='UNINSTALLED'
    assert (b.s._rotation_begin,b.s._rotation_hold,b.s.schedule)==b.old

    c=fixture();c.s.schedule()
    for _ in range(15):c.s.schedule()
    assert not any(e['kind']=='release' for e in c.data['events'])
    c.s.schedule()
    assert c.data['events'][-1]['reason']=='SIXTEEN_ROUND_LIMIT' and c.s._rotation_target is None
    c.undo()

    # Reservation counts are deducted once for a donor in both memberships.
    c=fixture();c.d.num_tokens=97;c.d.num_tokens_with_spec=97;c.d.num_computed_tokens=96
    c.s._inflight_prefills.add(c.d)
    c.s.schedule();assert c.data['action_count']==1 and c.d not in c.s._inflight_prefills
    decision=next(e for e in c.data['events'] if e['kind']=='decision')
    assert decision['joint_capacity_before_exchange']['inflight_remaining_blocks']==1
    assert decision['selected_capacity']['inflight_remaining_blocks']==0
    assert decision['selected_capacity']['donor_excluded_inflight_blocks']==1;c.undo()
    c=fixture();c.peer.num_tokens=65;c.peer.num_tokens_with_spec=65;c.peer.num_computed_tokens=64
    c.s.schedule();decision=next(e for e in c.data['events'] if e['kind']=='decision')
    assert decision['donor']=='larger' and decision['selected_capacity']['running_increment_blocks']==1
    assert [d['request'] for d in decision['donors'] if d['capacity']['capacity_fit']]==['larger'];c.undo()
    # Head priority boundary, pending target, unknown Host state, non-private
    # blocks and no sufficient donor must remain native without touching refs.
    for kind in ('priority','pending','global_load','shared','insufficient','mixed','new'):
        c=fixture(free=0 if kind=='insufficient' else 4)
        if kind=='priority':
            barrier=Request('barrier',16,0,STATUS.PREEMPTED,2.);barrier.priority=1
            c.s.waiting.insert(0,barrier);c.s.requests['barrier']=barrier
            c.cs._req_status['barrier']=NS(req=barrier,transfer_jobs=set(),group_states=[NS(offload_keys=['b'])])
            c.receipts['barrier']=2.
        if kind=='pending':
            c.cs._req_status['target'].transfer_jobs.add(1);c.cs._jobs[1]=NS(req_id='target',is_store=True)
        if kind=='global_load':c.cs._chunks_being_loaded.add('target0')
        if kind=='shared':c.pool.blocks[1].ref_cnt=2
        if kind=='mixed':c.peer.num_computed_tokens-=2
        if kind=='new':c.t.status=STATUS.WAITING
        c.s.schedule();assert c.data['action_count']==0,kind;c.undo()

    c=fixture();c.s.schedule();c.t.status=STATUS.WAITING
    c.s.schedule();assert c.data['events'][-1]['reason']=='NATIVE_OR_NEW_REQUEST_CONFLICT';c.undo()
    # Native failure propagates without attempting rollback of a partial native
    # mutation or silently retrying another donor.
    c=fixture()
    def fail(*a):raise RuntimeError('native preempt failure')
    c.s._preempt_request=fail
    try:c.s.schedule()
    except RuntimeError as e:assert str(e)=='native preempt failure'
    else:raise AssertionError('Native exception swallowed')
    assert c.data['status']=='ERROR';c.undo()

    # Exact native metadata and worker fence: existing donor STORE jobs are
    # flushed before any reused memory is touched; no fresh STORE is fabricated.
    build=native_method(B/'native_sources/offloading_scheduler_20261007.py',
        p.CAPACITY.SOURCE_SHA256['offloading_scheduler.py'],'OffloadingConnectorScheduler','build_connector_meta',
        {'ScheduleEndContext':lambda **kw:NS(**kw),'OffloadingConnectorMetadata':lambda **kw:NS(**kw)})
    pending={17};meta_cs=NS(_req_status={'donor':NS(transfer_jobs=pending)},_jobs={17:NS(is_store=True)},
        _current_batch_jobs_to_flush=set(),_current_batch_allocated_block_ids={1},_current_batch_load_jobs={},
        _block_id_to_pending_jobs={},manager=NS(on_schedule_end=lambda context:None),
        _update_req_states=lambda out:None,_build_store_jobs=lambda out:{})
    meta=build(meta_cs,NS(scheduled_new_reqs=[],preempted_req_ids=['donor']))
    assert meta.jobs_to_flush=={17} and meta.store_jobs=={}
    class GPUSpec:pass
    flush=native_method(B/'native_sources/offloading_worker.py',
        'd8f1a45cd01ea97307552c12bb8420dabbb63388fce10bd8b4b77d27bc5f1d35',
        'OffloadingConnectorWorker','handle_preemptions',{'GPULoadStoreSpec':GPUSpec})
    order=[]
    worker=NS(submit_store=lambda *args:order.append(('submit',args[0])) or True,
        wait=lambda ids:order.append(('wait',set(ids))))
    cw=NS(worker=worker,_unsubmitted_store_jobs=[(17,GPUSpec(),object())])
    flush(cw,meta)
    assert order==[('submit',17),('wait',{17})] and cw._unsubmitted_store_jobs==[]
finally:p.time.perf_counter=original_clock

print('PASS: one exact native preempt/free + forced-count; same baseline decision; minimum sufficient donor; physical/Host refs; Q1 and 16-round release; peer current growth guard; no extra exchange; priority/pending/unknown fallbacks; exact native metadata STORE fence; exception/uninstall. CPU only.')
