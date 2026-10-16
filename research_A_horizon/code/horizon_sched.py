"""Per-instance HORIZON adapter for the pinned synchronous vLLM v1 scheduler.

No vLLM imports until install. Native token allocation, ACKs and transfer
concurrency are unchanged. STORE cursor replay is common to all configurations.
Current prototype rebuilds its ledger lazily after invalidation. Each rebuild
visits all running/inflight owners and their held blocks; recovery reentry can
invalidate it repeatedly within one waiting phase. Cached gate queries are
O(1) for the two configured horizons. Victim search and prefix selection are
not constant-time. Do not claim a global thrash-free guarantee.
"""
from __future__ import annotations
import ast
from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import inspect
import math
from pathlib import Path
import textwrap
import time
from types import MethodType

PINS = {
    'Scheduler': '2ed2a550b6558b2495eda845a97ae38bcf0225027b9e25fbf00fc3880c1d3941',
    'KVCacheManager': '3f4af8d247f3fe9570b0132818b832b66ae6a2ac12942588828f899f6ff77ccf',
    'OffloadingConnectorScheduler': '89ac26a80fbc29b9bcaa5a0daba88fb6309247f9bb053e48d2d61efa7d9d66f1',
}

@dataclass(frozen=True)
class Config:
    H_a: int = 64
    H_r: int = 32
    lease_victim: bool = True
    stall_order: bool = True
    estimate_Hr: bool = False

    def __post_init__(self):
        if any(type(x) is not int or x < 0 for x in (self.H_a, self.H_r)):
            raise ValueError('Nonnegative integer horizons required')
        if any(type(x) is not bool for x in (self.lease_victim,self.stall_order,self.estimate_Hr)):
            raise ValueError('Boolean switches required')


def pages(n, b=16):
    return (n+b-1)//b


def instrument(source):
    """Transform only exact pinned native boundaries; reusable in CPU checks."""
    tree=ast.parse(textwrap.dedent(source))
    # Accept either the method alone or the complete pinned source file.
    classes=[x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='Scheduler']
    fn=next(x for x in (classes[0].body if classes else tree.body)
            if isinstance(x,ast.FunctionDef) and x.name=='schedule')
    counts=Counter()
    class Hooks(ast.NodeTransformer):
        def visit_Assign(self,node):
            text=ast.unparse(node)
            if text=='request_queue = self._select_waiting_queue_for_scheduling()':
                counts['order']+=1
                return ast.parse('request_queue = _hz.order(self._select_waiting_queue_for_scheduling())').body[0]
            if text.startswith('new_blocks = self.kv_cache_manager.allocate_slots(') and any(
                    k.arg=='num_external_computed_tokens' for k in node.value.keywords):
                counts['gate']+=1
                before=ast.parse('''if _hz.reject(request, num_scheduled_tokens, num_new_local_computed_tokens,
    effective_lookahead_tokens, num_encoder_tokens, new_computed_blocks):
    break''').body
                after=ast.parse('''if new_blocks is not None:
    _hz.allocated(request, load_kv_async)''').body
                return before+[node]+after
            return self.generic_visit(node)
        def visit_If(self,node):
            if ast.unparse(node.test)=='self.policy == SchedulingPolicy.PRIORITY':
                if len(node.orelse)!=1 or ast.unparse(node.orelse[0])!='preempted_req = self.running.pop()':
                    raise RuntimeError('Victim boundary changed')
                counts['victim']+=1
                node.orelse=ast.parse('''_hz_victim_index = _hz.victim()
preempted_req = self.running.pop(_hz_victim_index)
if preempted_req in scheduled_running_reqs:
    preempted_req_id = preempted_req.request_id
    scheduled_running_reqs.remove(preempted_req)
    token_budget += num_scheduled_tokens.pop(preempted_req_id)
    req_to_new_blocks.pop(preempted_req_id)
    scheduled_spec_decode_tokens.pop(preempted_req_id, None)
    preempted_encoder_inputs = scheduled_encoder_inputs.pop(preempted_req_id, None)
    if preempted_encoder_inputs:
        encoder_compute_budget += sum(preempted_req.get_num_encoder_embeds(i)
                                      for i in preempted_encoder_inputs)
if _hz_victim_index < req_index:
    req_index -= 1''').body
                return node
            return self.generic_visit(node)
    fn=Hooks().visit(fn)
    if counts!=dict(order=1,gate=1,victim=1):
        raise RuntimeError('Unexpected native hook counts: '+str(counts))
    fn.decorator_list=[]
    return ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[]))


class Adapter:
    def __init__(self,scheduler,cfg):
        self.s=scheduler
        self.cfg=cfg
        self.m=scheduler.kv_cache_manager
        self.single=self.m.coordinator.single_type_managers[0]
        self.cs=scheduler.connector.connector_scheduler if scheduler.connector else None
        self.step=0
        self.leases={}
        self.lease_expirations={}
        self.pending={}
        self.receipts={}
        self.ewma=float(cfg.H_r)
        self.undo=[]
        self.active=True
        self.ledger=None
        self.data=dict(status='INSTALLED',config=asdict(cfg),source_sha256=dict(PINS),
                       counts=Counter(),steps=[],events=[],scheduler_seconds=0.,
                       controller_seconds=0.,ledger_seconds=0.,
                       complexity='O(owners + held blocks) per ledger rebuild; reentry can cause multiple rebuilds per waiting phase; O(1) cached gate query; linear victim/prefix operations',
                       guarantee='conditional capacity only; UNKNOWN uses native; no global thrash-free guarantee')

    def _replace(self,obj,name,new):
        own=name in vars(obj)
        previous=vars(obj).get(name)
        setattr(obj,name,new)
        self.undo.append((obj,name,own,previous,new))

    def _count(self,name):
        self.data['counts'][name]+=1

    def _event(self,kind,r=None,**fields):
        self.data['events'].append(dict(kind=kind,step=self.step,
            request=None if r is None else r.request_id,host_perf_s=time.perf_counter(),**fields))

    def receipt(self,request_id,host_perf_s):
        """Called after a validated positive output delta by the capture loop."""
        if not math.isfinite(host_perf_s):
            raise ValueError('Finite host receipt required')
        request=self.s.requests.get(request_id)
        # The capture loop may receive the final output after _free_request.
        # Do not recreate lifecycle state that the finish hook just removed.
        if request is None or request.is_finished():
            return
        self.receipts[request_id]=host_perf_s
        pending=self.pending.pop(request_id,None)
        if pending is not None:
            delay=max(1,self.step-pending)
            self.ewma=.8*self.ewma+.2*delay
            self._event('recovery_first_output',request_id_value=request_id,
                        recovery_steps=delay,receipt_host_perf_s=host_perf_s)

    def hr(self):
        # Whole block epochs prevent the fluid estimator undercounting phases.
        return max(16,min(256,16*math.ceil(self.ewma/16))) if self.cfg.estimate_Hr else self.cfg.H_r

    def begin(self):
        self.step+=1
        self.ledger=None
        # Each lease is visited at its expiry, rather than on every step.
        # Completion/preemption are handled by the native lifecycle hooks.
        # An old expiry must not cancel a newer lease for the same request.
        for rid in self.lease_expirations.pop(self.step, ()):
            if self.leases.get(rid)==self.step:
                self.leases.pop(rid)

    def _lease(self,rid,horizon):
        self._unlease(rid)
        expiry=self.step+horizon
        self.leases[rid]=expiry
        self.lease_expirations.setdefault(expiry,set()).add(rid)

    def _unlease(self,rid):
        expiry=self.leases.pop(rid,None)
        bucket=self.lease_expirations.get(expiry)
        if bucket is not None:
            bucket.discard(rid)
            if not bucket:
                del self.lease_expirations[expiry]

    def _row(self,r,scheduled):
        if self.s.requests.get(r.request_id) is not r:
            raise ValueError('REQUEST_IDENTITY')
        n,c,i,q=(r.num_tokens,r.num_computed_tokens,r.num_in_flight_tokens,
                 scheduled.get(r.request_id,0))
        blocks=self.single.req_to_blocks.get(r.request_id,())
        if any(type(x) is not int or x<0 for x in (n,c,i,q)):
            raise ValueError('TOKEN_COUNTS')
        if (n>self.m.max_model_len or i>c or c+q>n or c+q>len(blocks)*16
                or r.has_encoder_inputs or r.num_output_placeholders or r.spec_token_ids
                or r.request_id in self.single._partial_hit_reqs):
            raise ValueError('UNSUPPORTED_TOKEN_BOUNDARY')
        if any(b.is_null or b.ref_cnt!=1 for b in blocks):
            raise ValueError('NONPRIVATE_BLOCKS')
        return n,len(blocks)

    def _build(self,scheduled):
        start=time.perf_counter()
        try:
            running={r.request_id:r for r in self.s.running}
            inflight={r.request_id:r for r in self.s._inflight_prefills}
            if len(running)!=len(self.s.running) or any(
                rid in running and running[rid] is not r for rid,r in inflight.items()):
                raise ValueError('MEMBERSHIP')
            rows={rid:self._row(r,scheduled) for rid,r in (running|inflight).items()}
            def deficit(rid,h=0):
                n,held=rows[rid]
                return max(0,pages(min(self.m.max_model_len,n+h))-held)
            reserved=sum(deficit(rid) for rid in inflight)
            native=self.s._inflight_prefill_reserved_blocks()
            if native!=reserved:
                raise ValueError('NATIVE_RESERVATION_MISMATCH')
            horizons={self.cfg.H_a,self.hr()}
            # An inflight LOAD can enter decoding within this horizon. Keep
            # its growth charge across rebuilds, matching allocated().
            demand={h:sum(deficit(rid,h) for rid in rows) for h in horizons}
            # R is the physical remainder of running prefills. I excludes R.
            R=sum(deficit(rid) for rid in inflight if rid in running)
            self.ledger=dict(status='KNOWN',demand=demand,R=R,I=reserved-R,
                             native_reserved=reserved,n=len(running),
                             owners=set(rows),
                             reentry_rows={rid:(r,*rows[rid]) for rid,r in inflight.items()
                                           if rid not in running and deficit(rid)==0})
        except (AttributeError,KeyError,ValueError,TypeError) as error:
            self.ledger=dict(status='UNKNOWN',reason=str(error))
            self._count('unknown_'+str(error))
        self.data['ledger_seconds']+=time.perf_counter()-start

    def reject(self,r,scheduled,local,lookahead,encoder,blocks):
        start=time.perf_counter()
        try:
            recovering=r.num_preemptions>0
            h=self.hr() if recovering else self.cfg.H_a
            if not h or r.request_id in self.pending:
                return False
            if (local or lookahead or encoder or any(blocks.blocks)
                    or r.num_computed_tokens or r.num_in_flight_tokens
                    or self.single.req_to_blocks.get(r.request_id)):
                self._count('unknown_target_boundary')
                return False
            self._row(r,scheduled)
            if self.ledger is None:
                self._build(scheduled)
            if self.ledger['status']!='KNOWN':
                return False
            free=self.m.block_pool.get_num_free_blocks()
            target=pages(min(self.m.max_model_len,r.num_tokens+h))
            need=self.ledger['demand'][h]+target
            denied=free<need
            self.data['steps'].append(dict(step=self.step,request=r.request_id,
                recovery=recovering,n=self.ledger['n'],F=free,R=self.ledger['R'],
                I=self.ledger['I'],D=(self.ledger['n']*h+15)//16,
                exact_peer_demand=self.ledger['demand'][h],target=target,
                H=h,required=need,denied=denied,active_leases=len(self.leases)))
            if denied:
                self._count('recovery_denied' if recovering else 'admission_denied')
            return denied
        except (AttributeError,KeyError,ValueError,TypeError) as error:
            self._count('unknown_target_'+type(error).__name__)
            return False
        finally:
            self.data['controller_seconds']+=time.perf_counter()-start

    def allocated(self,r,asynchronous):
        recovering=r.request_id in self.pending
        represented=(self.ledger is not None and self.ledger['status']=='KNOWN'
                     and r.request_id in self.ledger['owners'])
        if recovering:
            self._event('recovery_reentry_allocated',r,asynchronous=asynchronous)
        if recovering or represented:
            # A fully allocated LOAD already contributes its entire horizon
            # demand. Successful ACK/reentry with unchanged history/footprint
            # only moves this owner into running; its R/I contributions are 0.
            # This includes new requests loading a shared prefix, which have
            # no recovery episode in pending.
            # Consume the row so a repeated callback cannot increment n twice.
            if not asynchronous and self.ledger is not None and self.ledger['status']=='KNOWN':
                previous=self.ledger['reentry_rows'].pop(r.request_id,None)
                if (previous is not None and previous[0] is r
                        and previous[1:]==(r.num_tokens,len(self.single.req_to_blocks.get(r.request_id,())))):
                    self.ledger['n']+=1
                    self._count('ledger_reentry_cached_recovery' if recovering else 'ledger_reentry_cached_new')
                    return
            # Partial/changed allocations and unrepresented owners still need
            # the full native-boundary validation before the next gate.
            self.ledger=None
            self._count('ledger_reentry_invalidated_recovery' if recovering else 'ledger_reentry_invalidated_new')
            return
        if self.ledger is not None and self.ledger['status']=='KNOWN':
            # Update once using the actual post-allocation footprint. Native
            # appends/inflight membership happen just after this hook.
            held=len(self.single.req_to_blocks.get(r.request_id,()))
            self.ledger['owners'].add(r.request_id)
            deficit=max(0,pages(r.num_tokens)-held)
            for h in self.ledger['demand']:
                # Pending LOAD may begin decoding during another lease window.
                # Include its H-step growth conservatively as well as history.
                self.ledger['demand'][h]+=max(0,pages(min(self.m.max_model_len,r.num_tokens+h))-held)
            if asynchronous:
                self.ledger['I']+=deficit
            else:
                self.ledger['R']+=deficit
                self.ledger['n']+=1
        if r.num_preemptions and r.request_id not in self.pending:
            self.pending[r.request_id]=self.step
            if self.hr():
                self._lease(r.request_id,self.hr())
            self._event('recovery_allocated',r,asynchronous=asynchronous,
                        lease_expiry=self.leases.get(r.request_id))

    def victim(self):
        if not self.cfg.lease_victim:
            return len(self.s.running)-1
        for index in range(len(self.s.running)-1,-1,-1):
            r=self.s.running[index]
            if self.leases.get(r.request_id,0)<=self.step:
                return index
            self._count('victim_skipped')
        self._count('victim_all_leased_fallback')
        return len(self.s.running)-1

    def order(self,queue):
        if queue is None or not self.cfg.stall_order:
            return queue
        prefix=[]
        for r in queue:
            if r.status.name!='PREEMPTED':
                break
            prefix.append(r)
        if len(prefix)<2:
            return queue
        now=time.perf_counter()
        known=all(r.request_id in self.receipts and self.receipts[r.request_id]<now for r in prefix)
        if not known:
            self._count('stall_unknown_receipt')
        chosen=min(prefix,key=(lambda r:(self.receipts[r.request_id],r.arrival_time,r.request_id))
                   if known else (lambda r:(r.arrival_time,r.request_id)))
        if chosen is not prefix[0]:
            queue.remove_request(chosen)
            queue.prepend_request(chosen)
            self._count('stall_reordered')
        return queue

    def _clear(self):
        self.leases.clear(); self.pending.clear(); self.receipts.clear()
        self.lease_expirations.clear()
        self.ledger=None

    def attach(self,native,*,schedule_hook=True):
        if schedule_hook:
            namespace=dict(native.__func__.__globals__,_hz=self)
            exec(compile(instrument(inspect.getsource(native.__func__)),
                         __file__+'[pinned-schedule]','exec'),namespace)
            rebuilt=MethodType(namespace['schedule'],self.s)
            def schedule(*args,**kwargs):
                start=time.perf_counter(); self.begin()
                try:
                    return rebuilt(*args,**kwargs)
                finally:
                    self.data['scheduler_seconds']+=time.perf_counter()-start
                    self._count('schedule_calls')
            self._replace(self.s,'schedule',schedule)
        apply_ack=self.s._update_waiting_for_remote_kv
        def on_ack(r,*args,**kwargs):
            # Failed LOAD acknowledgement may free or alter its physical
            # footprint. Do not reuse pre-ACK ownership validation afterward.
            failed=r.request_id in self.s.failed_recving_kv_req_ids
            result=apply_ack(r,*args,**kwargs)
            if failed:
                self.ledger=None
            if r.num_preemptions:
                self._event('recovery_ack_applied',r,computed_tokens=r.num_computed_tokens)
            return result
        self._replace(self.s,'_update_waiting_for_remote_kv',on_ack)
        preempt=self.s._preempt_request
        def on_preempt(r,*args,**kwargs):
            result=preempt(r,*args,**kwargs)
            self._unlease(r.request_id)
            self.pending.pop(r.request_id,None)
            self._event('preempted',r)
            return result
        self._replace(self.s,'_preempt_request',on_preempt)
        finish=self.s._free_request
        def on_finish(r,*args,**kwargs):
            result=finish(r,*args,**kwargs)
            self._unlease(r.request_id)
            for m in (self.pending,self.receipts):m.pop(r.request_id,None)
            return result
        self._replace(self.s,'_free_request',on_finish)
        for name in ('reset_prefix_cache','reset_connector_cache'):
            old=getattr(self.s,name)
            def reset(*args,_old=old,**kwargs):
                result=_old(*args,**kwargs)
                if result:self._clear()
                return result
            self._replace(self.s,name,reset)
        if self.cs is not None:
            original=self.cs.update_state_after_alloc
            def store_replay(r,blocks,external):
                state=self.cs._req_status.get(r.request_id)
                # Capture native lookup result before its allocation callback.
                rewind=(state is not None and state.req is r and r.num_preemptions>0
                        and r.num_computed_tokens==0 and external==0
                        and not state.transfer_jobs)
                hit=state.group_states[0].num_hit_chunks if rewind else None
                result=original(r,blocks,external)
                if rewind and state.offloading_context.policy.name=='BLOCK_LEVEL':
                    group=state.group_states[0]
                    if type(hit) is int and 0<=hit<group.next_stored_chunk_idx:
                        before=group.next_stored_chunk_idx
                        group.next_stored_chunk_idx=hit
                        self._count('store_cursor_rewinds')
                        self._event('store_cursor_rewind',r,before=before,after=hit)
                return result
            self._replace(self.cs,'update_state_after_alloc',store_replay)
        return self

    def uninstall(self):
        if not self.active:return self.data
        # Check all hooks first; do not partially uninstall over another owner.
        if any(getattr(obj,name) is not new for obj,name,_,_,new in self.undo):
            raise RuntimeError('A later adapter owns a HORIZON hook; remove it first')
        for obj,name,own,previous,new in reversed(self.undo):
            if own:setattr(obj,name,previous)
            else:delattr(obj,name)
        self._clear(); self.active=False
        self.data['status']='UNINSTALLED'
        return self.data


def install(scheduler,cfg=Config(),*,_adapter_factory=Adapter):
    """Return Adapter with .uninstall(), .data and .receipt(id, host_time)."""
    cfg=Config(**cfg) if isinstance(cfg,dict) else cfg
    if not isinstance(cfg,Config):raise TypeError('Config or dict required')
    if 'schedule' in vars(scheduler):
        raise RuntimeError('Install on an unmodified native scheduler instance')
    manager=scheduler.kv_cache_manager
    for obj,key in ((scheduler,'Scheduler'),(manager,'KVCacheManager')):
        path=inspect.getsourcefile(type(obj))
        if type(obj).__name__!=key or path is None or hashlib.sha256(Path(path).read_bytes()).hexdigest()!=PINS[key]:
            raise RuntimeError('Unpinned '+key)
    groups=manager.kv_cache_config.kv_cache_groups
    singles=manager.coordinator.single_type_managers
    if len(groups)!=1 or len(singles)!=1:
        raise ValueError('One KV group required')
    spec=groups[0].kv_cache_spec
    if (type(spec).__name__!='FullAttentionSpec' or spec.block_size!=16
            or spec.sliding_window is not None or spec.attention_chunk_size is not None
            or groups[0].is_eagle_group or manager.enable_caching or manager.use_eagle
            or scheduler.num_spec_tokens or scheduler.num_lookahead_tokens
            or scheduler.is_encoder_decoder or manager.watermark_blocks
            or scheduler.dcp_world_size!=1 or scheduler.pcp_world_size!=1
            or singles[0]._max_admission_blocks_per_request is not None
            or scheduler.policy.name!='FCFS'):
        raise ValueError('Unsupported runtime layout')
    if scheduler.connector is not None:
        cs=scheduler.connector.connector_scheduler
        path=inspect.getsourcefile(type(cs))
        if path is None or hashlib.sha256(Path(path).read_bytes()).hexdigest()!=PINS['OffloadingConnectorScheduler']:
            raise RuntimeError('Unpinned offload scheduler')
        if (len(cs.config.kv_group_configs)!=1 or cs.config.kv_group_configs[0].is_eagle_group
                or cs.config.kv_group_configs[0].sliding_window_size_in_chunks is not None):
            raise ValueError('Unsupported STORE replay layout')
    adapter=_adapter_factory(scheduler,cfg)
    try:return adapter.attach(scheduler.schedule)
    except BaseException:
        adapter.uninstall()
        raise
