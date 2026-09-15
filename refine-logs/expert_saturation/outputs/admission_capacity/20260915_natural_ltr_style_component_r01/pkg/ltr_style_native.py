"""Recovery-only LTR-style counters/quantum on the qualified native selected-save path.
Mutually exclusive scheduler adapter. CPU qualification does not establish GPU execution.
"""
import hashlib
import inspect
import time
from dataclasses import asdict
from types import MethodType
from staged_save_contract import RequestState, prepare, commit_reason, recovery_guard
from rotation_native import patched_schedule_tree, SCHEDULER_SHA256
from native_store_delta import inspect_store_delta
from ltr_style_selected import LTRStyleSelected, Request


def install(scheduler, *, vllm_config, save, block_size, diagnostic=True, threshold=30, quantum=10):
    if block_size!=16:raise ValueError('Qualification requires 16-token blocks')
    if scheduler.max_num_running_reqs!=32:raise ValueError('Requires unchanged running cap32')
    if not save:raise ValueError('This contrast requires native saving enabled')
    manager=scheduler.kv_cache_manager
    connector=scheduler.connector
    if type(connector).__name__!='OffloadingConnector':
        raise ValueError('Native OffloadingConnector required')
    cs=connector.connector_scheduler
    if (vllm_config.scheduler_config.async_scheduling or vllm_config.speculative_config is not None
        or manager.enable_caching or manager.num_kv_cache_groups!=1 or manager.use_eagle
        or manager.watermark_blocks or scheduler.num_lookahead_tokens or scheduler.num_spec_tokens
        or scheduler.dcp_world_size!=1 or scheduler.pcp_world_size!=1 or scheduler.use_v2_model_runner
        or scheduler.defer_block_free or scheduler.ec_connector is not None
        or scheduler.policy.name!='FCFS' or scheduler.is_encoder_decoder
        or not scheduler.scheduler_reserve_full_isl or cs.config.offload_prompt_only
        or cs.config.blocks_per_chunk!=1 or cs.config.num_workers!=1):
        raise ValueError('Unsupported execution or offload configuration')
    singles=manager.coordinator.single_type_managers
    if (len(singles)!=1 or type(singles[0]).__name__!='FullAttentionManager'
        or type(manager.coordinator).__name__!='KVCacheCoordinatorNoPrefixCache'):
        raise ValueError('Unshared full attention required')
    if scheduler.requests or 'schedule' in vars(scheduler):
        raise ValueError('Install before capture on a drained scheduler')
    original=scheduler.schedule
    path=inspect.getsourcefile(type(scheduler))
    source=open(path,'rb').read()
    if hashlib.sha256(source).hexdigest()!=SCHEDULER_SHA256:
        raise ValueError('Native scheduler source drift')
    namespace=dict(original.__func__.__globals__)
    exec(compile(patched_schedule_tree(source.decode()),path,'exec'),namespace)
    native=MethodType(namespace['schedule'],scheduler)
    pool=manager.block_pool; owned=singles[0].req_to_blocks
    oldcalc=cs._calc_num_offloadable_tokens
    hadcalc='_calc_num_offloadable_tokens' in vars(cs)
    step=0;plan=None;phase=None;pending_flush=set();cancelled=False;target=None
    output_start=0;output_seen=False;reserve=False;intent=None
    selector=LTRStyleSelected(threshold,quantum)
    data=dict(save=True,store_scope='selected',native_calc_overridden=True,
        ltr_config=dict(threshold=threshold,quantum=quantum),applied_rotations=0,
        events=[],schedule_calls=0,status='INSTALLED',diagnostic=diagnostic)
    hooks=('_rotation_begin','_rotation_hold','_rotation_target','_rotation_forced_count')
    if any(k in vars(scheduler) for k in hooks):raise ValueError('Existing rotation hook')

    def view(req):
        if req is None:return None
        blocks=owned.get(req.request_id,())
        if any(b.is_null or pool.blocks[b.block_id] is not b for b in blocks):
            raise RuntimeError('Invalid physical ownership')
        return RequestState(req.request_id,req.num_computed_tokens,req.num_prompt_tokens,
            req.num_output_tokens,req.max_tokens,req.status.name,tuple(b.block_id for b in blocks))

    def event(kind,**values):
        data['events'].append(dict(event=kind,step=step,host_perf_counter_s=time.perf_counter(),**values))

    def release(reason):
        nonlocal target,reserve
        if selector.active_target is not None:event('release',target=selector.active_target,reason=reason)
        selector.release_active();target=None;reserve=False

    def limited(rs,n):
        if phase!='prepare' or plan is None or rs.req.request_id!=plan.victim.request_id:return 0
        return min(oldcalc(rs,n),plan.saved_tokens)

    def executable(proposal):
        if scheduler.skipped_waiting:return 'unrelated blocked native queue'
        if proposal.action=='PRIORITIZE_WAITING':
            if len(scheduler.running)>=scheduler.max_num_running_reqs:return 'native sequence slots full; no free-capacity eviction'
            return None
        rows=[view(r) for r in scheduler.running]
        if any(not r.pure_decode for r in rows):return 'mixed prefill or recovery in prepare'
        if sum(r.remaining_blocks for r in rows)>pool.get_num_free_blocks():return 'prepare growth not funded'
        try:prepare(step,view(scheduler.requests[proposal.victim_id]),
                    view(scheduler.requests[proposal.target_id]),pool.get_num_free_blocks())
        except ValueError as exc:return str(exc)
        return None

    def begin(preempted,timestamp):
        nonlocal plan,phase,pending_flush,cancelled,target,reserve,intent,output_start,output_seen
        scheduler._rotation_forced_count=0;phase=None;pending_flush=set();cancelled=False
        if selector.active_target is not None and any(r.request_id!=selector.active_target for r in scheduler.skipped_waiting):
            release('unrelated blocked native queue; release gate without resetting counters')
        prior=selector.active_target
        rows=[]
        ordered=list(scheduler.waiting)+list(scheduler.skipped_waiting)+list(scheduler.running)
        ordered+= [r for r in scheduler.requests.values() if r not in ordered]
        for req in ordered:
            r=view(req);rs=cs._req_status.get(r.request_id)
            no_load=not rs or all(cs._jobs[j].is_store for j in rs.transfer_jobs)
            rows.append(Request(r.request_id,req.arrival_time,r.status,r.remaining_blocks,r.output,
                len(r.blocks),r.pure_decode and r.output+1<r.max_output and r.computed>=16 and no_load))
        intent=selector.begin_step(rows,pool.get_num_free_blocks(),backend_idle=plan is None,executable=executable,
            free_slots=scheduler.max_num_running_reqs-len(scheduler.running))
        if prior is not None and selector.active_target is None:
            event('release',target=prior,reason='terminal' if prior not in scheduler.requests else 'quantum_expired')
        target=scheduler.requests.get(selector.active_target)
        if plan is not None:
            phase='commit'
            victim=scheduler.requests.get(plan.victim.request_id);candidate=scheduler.requests.get(plan.target.request_id)
            reason=commit_reason(plan,step,view(victim),view(candidate),pool.get_num_free_blocks(),None,save_enabled=False)
            if reason=='READY':
                if selector.active_target!=plan.target.request_id:reason='CANCEL_NO_ACTIVE_QUANTUM'
                elif scheduler.skipped_waiting or any(not view(r).pure_decode for r in scheduler.running):reason='CANCEL_NATIVE_QUEUE_OR_MIXED_WORK'
                elif selector.counters.states[victim.request_id].priority<=selector.counters.states[candidate.request_id].priority:reason='CANCEL_VICTIM_PRIORITY'
                else:
                    rs=cs._req_status.get(victim.request_id)
                    pending_flush=set(rs.transfer_jobs) if rs else set()
                    if any(not cs._jobs[j].is_store for j in pending_flush):reason='CANCEL_VICTIM_LOAD_PENDING'
            event('commit_check',reason=reason,victim=plan.victim.request_id,target=plan.target.request_id)
            if reason!='READY':cancelled=True;plan=None;release(reason)
            else:
                scheduler.running.remove(victim);scheduler._preempt_request(victim,timestamp)
                preempted.append(victim);scheduler._rotation_forced_count=1
                target=candidate;data['applied_rotations']+=1
        elif intent.action in ('PREPARE_SELECTED','PRIORITIZE_WAITING'):
            if intent.action=='PREPARE_SELECTED':
                plan=prepare(step,view(scheduler.requests[intent.victim_id]),view(scheduler.requests[intent.target_id]),pool.get_num_free_blocks())
                phase='prepare';event('prepare',victim=intent.victim_id,target=intent.target_id,saved_tokens=plan.saved_tokens,
                    victim_output=plan.victim.output,victim_declared_cap=plan.victim.max_output,
                    target_output=plan.target.output,target_quantum_remaining=intent.quantum_remaining,
                    slot_required=len(scheduler.running)>=scheduler.max_num_running_reqs,
                    kv_funding_required=pool.get_num_free_blocks()<plan.target.remaining_blocks)
            selector.accept(intent);target=scheduler.requests[intent.target_id]
            output_start=target.num_output_tokens;output_seen=False
            event('accept',**asdict(intent))
        if target is not None and target.num_output_tokens>output_start and not output_seen:
            event('target_new_output',target=target.request_id,output_tokens=target.num_output_tokens,
                  quantum_remaining=selector.counters.states[target.request_id].quantum_remaining)
            output_seen=True
        # Native FCFS preempts the tail. Preserve all LTR boosted running priorities,
        # then native arrival/tie order. A one-block growth miss may preempt a peer.
        scheduler.running.sort(key=lambda r:(selector.counters.states[r.request_id].priority,r.arrival_time))
        reserve=target is not None and phase!='prepare' and pool.get_num_free_blocks()>=view(target).remaining_blocks
        if target is not None and phase!='prepare' and target.status.name=='PREEMPTED':
            scheduler.waiting.remove_request(target);scheduler.waiting.prepend_request(target)
        scheduler._rotation_target=target.request_id if target is not None and phase!='prepare' else None
        if diagnostic:
            event('intent',**asdict(intent),active_target=selector.active_target,reserve_enabled=reserve,
                skipped=selector.skipped,boosted={rid:asdict(s) for rid,s in selector.counters.states.items() if s.priority==-1})

    def hold(req):
        if not reserve or target is None or req is target:return False
        return not recovery_guard(view(target),pool.get_num_free_blocks(),view(req).remaining_blocks)['other_growth_allowed']

    def schedule(*args,**kwargs):
        nonlocal step,plan,phase
        try:
            result=native(*args,**kwargs);meta=result.kv_connector_metadata
            if phase=='prepare':
                if result.num_scheduled_tokens.get(plan.victim.request_id)!=1:
                    event('backend_censored',target=plan.target.request_id,reason='preparation victim did not decode once')
                    plan=None;release('preparation censored')
                elif diagnostic:
                    delta=inspect_store_delta(plan,meta,cs._req_status[plan.victim.request_id],cs._jobs)
                    event('store_delta',**delta)
            if phase=='commit' and not cancelled:
                if plan.victim.request_id not in (result.preempted_req_ids or ()):
                    raise RuntimeError('Native preemption notification absent')
                if not pending_flush<=set(meta.jobs_to_flush):raise RuntimeError('Native flush omitted pending stores')
            forced={plan.victim.request_id} if phase=='commit' and plan and not cancelled else set()
            natural=set(result.preempted_req_ids or ())-forced
            if natural:event('native_preemption',requests=sorted(natural))
            if target is not None:
                tokens=result.num_scheduled_tokens.get(target.request_id,0)
                if target.status.name=='WAITING_FOR_REMOTE_KVS' and tokens:raise RuntimeError('Pending load executed')
                if not tokens and phase!='prepare':event('backend_censored',target=target.request_id,reason=target.status.name)
                if target.request_id in natural:release('target naturally preempted; counters retained')
                elif not tokens and phase!='prepare' and target.status.name=='PREEMPTED':
                    release('native admission censored; rescan without resetting counters')
            selector.after_step(result.num_scheduled_tokens)
            if diagnostic:
                event('allocation',scheduled=dict(result.num_scheduled_tokens),active_target=selector.active_target,
                    boosted={rid:asdict(s) for rid,s in selector.counters.states.items() if s.priority==-1})
            if diagnostic and (meta.store_jobs or meta.load_jobs or meta.jobs_to_flush or phase):
                event('metadata',stores=sorted(meta.store_jobs),loads=sorted(meta.load_jobs),flush=sorted(meta.jobs_to_flush),
                    jobs=[dict(job_id=j,request=cs._jobs[j].req_id,is_store=cs._jobs[j].is_store)
                          for j in sorted(set(meta.store_jobs)|set(meta.load_jobs)|set(meta.jobs_to_flush)) if j in cs._jobs])
            if phase=='commit':plan=None
            step+=1;data['status']='EXECUTING'
            return result
        except Exception as exc:
            data.update(status='ERROR',error=repr(exc));raise

    scheduler._rotation_begin=begin;scheduler._rotation_hold=hold
    scheduler._rotation_target=None;scheduler._rotation_forced_count=0
    scheduler.schedule=schedule;cs._calc_num_offloadable_tokens=limited
    def uninstall():
        for rid in list(selector.counters.states):
            if rid not in scheduler.requests:selector.finish(rid)
        for key in ('schedule',)+hooks:delattr(scheduler,key)
        if hadcalc:cs._calc_num_offloadable_tokens=oldcalc
        else:delattr(cs,'_calc_num_offloadable_tokens')
        if data['status']!='ERROR':data['status']='DRAINED' if not scheduler.requests else 'UNINSTALLED_WITH_PENDING_REQUESTS'
        data.update(schedule_calls=step,final_counters={rid:asdict(s) for rid,s in selector.counters.states.items()},
                    active_target=selector.active_target,pending_plan=asdict(plan) if plan else None)
        return data
    return data,uninstall
