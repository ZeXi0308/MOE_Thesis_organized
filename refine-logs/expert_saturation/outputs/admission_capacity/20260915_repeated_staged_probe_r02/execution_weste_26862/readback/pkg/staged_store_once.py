"""Single staged most-output save/preempt qualification; GPU validation UNRUN.

Uses native offload metadata and flushes. Not a complete rotation controller.
Install before capture wrappers, on the pinned drained synchronous scheduler.
"""
import hashlib
import inspect
from types import MethodType
from staged_save_contract import RequestState, StoreEvidence, prepare, commit_reason, recovery_guard
from rotation_native import patched_schedule_tree, SCHEDULER_SHA256


def install(scheduler, *, vllm_config, save, block_size, select_step=329):
    if block_size!=16:raise ValueError('Qualification requires 16-token blocks')
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
    step=0; plan=None; evidence=None; protected=None; output_start=None
    data=dict(save=save,events=[],preempted=False,status='INSTALLED_GPU_UNVALIDATED')
    hooks=('_rotation_begin','_rotation_hold','_rotation_target','_rotation_forced_count')
    if any(k in vars(scheduler) for k in hooks):raise ValueError('Existing rotation hook')

    def view(req):
        if req is None:return None
        blocks=owned.get(req.request_id,())
        if any(b.is_null or pool.blocks[b.block_id] is not b for b in blocks):
            raise RuntimeError('Invalid physical ownership')
        return RequestState(req.request_id,req.num_computed_tokens,req.num_prompt_tokens,
                            req.num_output_tokens,req.max_tokens,req.status.name,
                            tuple(b.block_id for b in blocks))

    def limited(rs,n):
        if not save or plan is None or rs.req.request_id!=plan.victim.request_id:return 0
        return min(oldcalc(rs,n),plan.saved_tokens)

    def begin(preempted,timestamp):
        nonlocal plan,protected,output_start
        scheduler._rotation_forced_count=0
        if step==select_step:
            running=[view(r) for r in scheduler.running]
            waiting=[r for r in scheduler.waiting if r.status.name=='PREEMPTED']
            if len(running)!=31 or len(waiting)!=1 or scheduler.skipped_waiting or not all(r.pure_decode for r in running):
                raise RuntimeError('Unexpected qualification state')
            victim=max(running,key=lambda r:r.output)
            plan=prepare(step,victim,view(waiting[0]),pool.get_num_free_blocks())
            data['events'].append(dict(event='prepare',step=step,victim=victim.request_id,
                target=plan.target.request_id,saved_tokens=plan.saved_tokens,source_blocks=list(plan.source_blocks)))
        if step==select_step+1:
            victim=scheduler.requests.get(plan.victim.request_id)
            target=scheduler.requests.get(plan.target.request_id)
            reason=commit_reason(plan,step,view(victim),view(target),pool.get_num_free_blocks(),evidence,save_enabled=save)
            data['events'].append(dict(event='commit_check',step=step,reason=reason))
            if reason!='READY':
                data['status']=reason
                raise RuntimeError(reason)
            # Dynamic dispatch retains telemetry; native metadata must flush stores.
            scheduler.running.remove(victim)
            scheduler._preempt_request(victim,timestamp)
            preempted.append(victim)
            scheduler._rotation_forced_count=1
            protected=target;output_start=target.num_output_tokens
            data['preempted']=True
        if protected is not None and any(r is not protected for r in scheduler.skipped_waiting):
            raise RuntimeError('Unqualified unrelated blocked queue during protection')
        if protected is not None and protected.status.name=='PREEMPTED':
            scheduler.waiting.remove_request(protected)
            scheduler.waiting.prepend_request(protected)
        scheduler._rotation_target=protected.request_id if protected is not None else None

    def hold(req):
        if protected is None or req is protected:return False
        r=view(req)
        if not r.pure_decode:raise RuntimeError('Concurrent unqualified recovery')
        return not recovery_guard(view(protected),pool.get_num_free_blocks(),r.remaining_blocks)['other_growth_allowed']

    def schedule(*args,**kwargs):
        nonlocal step,evidence,protected
        if protected is not None and protected.num_output_tokens>output_start:
            data['events'].append(dict(event='target_new_output',step=step,request=protected.request_id))
            protected=None
        result=native(*args,**kwargs)
        meta=result.kv_connector_metadata
        if step==select_step:
            if result.num_scheduled_tokens.get(plan.victim.request_id)!=1:
                raise RuntimeError('Preparation victim did not decode exactly once')
            stores=[(jid,job) for jid,job in meta.store_jobs.items() if job.req_id==plan.victim.request_id]
            if save:
                if len(stores)!=1:raise RuntimeError('Expected one complete-prefix native store')
                jid,job=stores[0]
                blocks=tuple(int(b) for b in job.src_spec.block_ids)
                registered=cs._jobs.get(jid)
                if (blocks!=plan.source_blocks or registered is None or not registered.is_store
                    or registered.req_id!=plan.victim.request_id):
                    raise RuntimeError('Native store coverage/registration mismatch')
                evidence=StoreEvidence(plan.victim.request_id,plan.saved_tokens,blocks,(jid,))
            elif meta.store_jobs:
                raise RuntimeError('Save-off unexpectedly emitted stores')
        if step==select_step+1:
            if plan.victim.request_id not in (result.preempted_req_ids or ()):
                raise RuntimeError('Missing native preemption notification')
            if save and not set(evidence.registered_job_ids)<=set(meta.jobs_to_flush):
                raise RuntimeError('Missing native flush before worker block reuse')
        if protected is not None:
            unexpected=set(result.preempted_req_ids or ())-{plan.victim.request_id}
            if unexpected:raise RuntimeError('Unexpected natural preemption during protection')
            state=view(protected)
            if pool.get_num_free_blocks()<state.remaining_blocks:
                raise RuntimeError('Target reserve consumed')
            if (state.status!='WAITING_FOR_REMOTE_KVS'
                and result.num_scheduled_tokens.get(protected.request_id,0)<=0):
                raise RuntimeError('Ready target made no progress')
            if state.status=='WAITING_FOR_REMOTE_KVS' and result.num_scheduled_tokens.get(protected.request_id,0):
                raise RuntimeError('Pending load was executed')
        if select_step<=step<=select_step+10:
            data['events'].append(dict(event='metadata',step=step,stores=sorted(meta.store_jobs),
                loads=sorted(meta.load_jobs),flush=sorted(meta.jobs_to_flush),
                protected=protected.request_id if protected else None))
        step+=1
        return result

    scheduler._rotation_begin=begin;scheduler._rotation_hold=hold
    scheduler._rotation_target=None;scheduler._rotation_forced_count=0
    scheduler.schedule=schedule;cs._calc_num_offloadable_tokens=limited
    def uninstall():
        for key in ('schedule',)+hooks:delattr(scheduler,key)
        if hadcalc:cs._calc_num_offloadable_tokens=oldcalc
        else:delattr(cs,'_calc_num_offloadable_tokens')
        data['schedule_calls']=step
        return data
    return data,uninstall
