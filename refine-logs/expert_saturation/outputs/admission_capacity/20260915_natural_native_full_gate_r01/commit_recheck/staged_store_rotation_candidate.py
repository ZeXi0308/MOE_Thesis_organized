"""Repeated staged most-output native adapter; GPU validation UNRUN.

Uses native offload metadata and flushes. Incremental stores use native residency and fallback.
Install before capture wrappers, on the pinned drained synchronous scheduler.
"""
import hashlib
import inspect
import time
from dataclasses import asdict
from types import MethodType
from staged_save_contract import RequestState, prepare, commit_reason, recovery_guard
from rotation_native import patched_schedule_tree, SCHEDULER_SHA256
from absence_rotation import AbsenceRotation, RequestView, RotationConfig
from native_store_delta import inspect_store_delta
from native_full_store_evidence import inspect_native_full_jobs
from commit_disposition import choose_commit_action


def _eligibility_snapshot(scheduler, tracker, view, free_blocks, step,
                          cohort, plan, protected):
    requests = {}
    for rid, req in scheduler.requests.items():
        row = view(req)
        requests[rid] = dict(computed=row.computed, prompt=row.prompt,
            output=row.output, max_output=row.max_output, status=row.status,
            held_blocks=len(row.blocks))
    return dict(step=step, host_perf_counter_s=time.perf_counter(),
        free_blocks=free_blocks, running_ids=[r.request_id for r in scheduler.running],
        waiting_ids=[r.request_id for r in scheduler.waiting],
        skipped_ids=[r.request_id for r in scheduler.skipped_waiting],
        requests=requests, cohort_active=cohort is not None,
        plan_victim=plan.victim.request_id if plan is not None else None,
        plan_target=plan.target.request_id if plan is not None else None,
        protected_id=protected.request_id if protected is not None else None,
        protected_reserve=view(protected).remaining_blocks if protected is not None else 0,
        tracker=dict(absent_since=dict(tracker.absent_since),
            absence_count=dict(tracker.absence_count),
            resident_since=dict(tracker.resident_since),
            last_swap_step=tracker.last_swap_step, config=asdict(tracker.config)))


def install(scheduler, *, vllm_config, save, block_size, expected_requests=32,
            diagnostic=False, global_cooldown_steps=20, population_mode="closed", store_scope="selected", recheck_commit_funding=False):
    if block_size!=16:raise ValueError('Qualification requires 16-token blocks')
    if store_scope not in ('selected','native_full'):raise ValueError('Unsupported native store scope')
    if population_mode not in ('closed','open'):raise ValueError('Unsupported population mode')
    open_population=population_mode=='open'
    if open_population and scheduler.max_num_running_reqs!=32:
        raise ValueError('Open population requires unchanged running cap32')
    if global_cooldown_steps not in (0,20):raise ValueError('Only cooldown 0/20 is qualified')
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
    if store_scope=='native_full' and hadcalc:
        raise ValueError('Full baseline requires the unmodified native class method')
    step=0;plan=None;protected=None;output_start=None;cohort=None;phase=None
    pending_flush=set();cancelled=False
    if open_population:cohort=set()
    tracker=AbsenceRotation(config=RotationConfig(min_steps_between_swaps=global_cooldown_steps),
        victim_order='most_output')
    data=dict(save=save,store_scope=store_scope,native_calc_overridden=store_scope=='selected',
        events=[],status='INSTALLED',applied_rotations=0,direct_resumes=0,
        recheck_commit_funding=recheck_commit_funding,
        diagnostic=diagnostic, population_mode=population_mode, gate_observations=[],
        rotation_config=asdict(tracker.config),
        eligibility_snapshots=[], selector_decisions=[],
        eligibility_logging='ENABLED' if diagnostic else 'DISABLED')
    hooks=('_rotation_begin','_rotation_hold','_rotation_target','_rotation_forced_count')
    if any(k in vars(scheduler) for k in hooks):raise ValueError('Existing rotation hook')

    def view(req):
        if req is None:return None
        blocks=owned.get(req.request_id,())
        if any(b.is_null or pool.blocks[b.block_id] is not b for b in blocks):
            raise RuntimeError('Invalid physical ownership')
        return RequestState(req.request_id,req.num_computed_tokens,req.num_prompt_tokens,
            req.num_output_tokens,req.max_tokens,req.status.name,tuple(b.block_id for b in blocks))

    def limited(rs,n):
        if not save or phase!='prepare' or plan is None or rs.req.request_id!=plan.victim.request_id:return 0
        return min(oldcalc(rs,n),plan.saved_tokens)

    def begin(preempted,timestamp):
        nonlocal plan,protected,output_start,phase,pending_flush,cancelled,cohort
        scheduler._rotation_forced_count=0;phase=None;pending_flush=set();cancelled=False
        rows=[view(r) for r in scheduler.running]
        if open_population:
            for state in (tracker.absent_since,tracker.absence_count,tracker.resident_since):
                for rid in list(state):
                    if rid not in scheduler.requests:state.pop(rid)
            if diagnostic:
                gate='selector_eligible'
                if plan is not None:gate='plan_commit'
                elif protected is not None:gate='protected_recovery_active'
                elif scheduler.skipped_waiting:gate='skipped_waiting_or_pending_load'
                elif any(not r.pure_decode for r in rows):
                    gate='mixed_prefill' if any(r.output==0 for r in rows) else 'native_recovery_underway'
                elif not any(r.status.name=='PREEMPTED' for r in scheduler.waiting):gate='no_preempted_waiter'
                data['gate_observations'].append(dict(step=step,gate=gate,live_requests=len(scheduler.requests),
                    running_requests=len(rows),waiting_requests=len(scheduler.waiting),
                    legacy_closed_activation_condition_met=len(rows)==expected_requests and not scheduler.waiting
                        and not scheduler.skipped_waiting and all(r.pure_decode for r in rows)))
        if cohort is None and len(rows)==expected_requests and not scheduler.waiting and not scheduler.skipped_waiting and all(r.pure_decode for r in rows):
            cohort=set(scheduler.requests)
        if not open_population and cohort is not None and not set(scheduler.requests)<=cohort:
            raise RuntimeError('Closed cohort changed')
        if diagnostic and (cohort is not None or any(
                r.status.name=='PREEMPTED' for r in scheduler.requests.values())):
            data['eligibility_snapshots'].append(_eligibility_snapshot(
                scheduler,tracker,view,pool.get_num_free_blocks(),step,cohort,plan,protected))
        if plan is not None:
            phase='commit'
            victim=scheduler.requests.get(plan.victim.request_id);target=scheduler.requests.get(plan.target.request_id)
            # New-store coverage was checked at prepare. Older host residency
            # is intentionally unknown; native lookup/recompute handles misses.
            reason=commit_reason(plan,step,view(victim),view(target),pool.get_num_free_blocks(),None,save_enabled=False)
            if open_population and reason=='READY':
                if scheduler.skipped_waiting:reason='CANCEL_OPEN_SKIPPED_WAITING'
                elif any(not r.pure_decode for r in rows):reason='CANCEL_OPEN_MIXED_PREFILL_OR_RECOVERY'
            data['events'].append(dict(event='commit_check',step=step,reason=reason,victim=plan.victim.request_id,target=plan.target.request_id))
            if reason!='READY':
                cancelled=True;plan=None
            else:
                rs=cs._req_status.get(victim.request_id)
                victim_jobs=set(rs.transfer_jobs) if rs else set()
                if any(not cs._jobs[j].is_store for j in victim_jobs):
                    raise RuntimeError('Victim still has an in-flight load')
                direct=(recheck_commit_funding and choose_commit_action(
                    reason,pool.get_num_free_blocks(),view(target).remaining_blocks)=='RESUME_WITHOUT_VICTIM')
                if direct:
                    phase='direct_resume'
                    data['direct_resumes']+=1
                    data['events'].append(dict(event='direct_resume',step=step,
                        victim_retained=victim.request_id,target=target.request_id,
                        free_blocks=pool.get_num_free_blocks(),
                        target_remaining_blocks=view(target).remaining_blocks))
                else:
                    pending_flush=victim_jobs
                    scheduler.running.remove(victim);scheduler._preempt_request(victim,timestamp)
                    preempted.append(victim);scheduler._rotation_forced_count=1
                    data['applied_rotations']+=1
                protected=target;output_start=target.num_output_tokens
        elif cohort is not None and protected is None and not scheduler.skipped_waiting and all(r.pure_decode for r in rows):
            waiting=[r for r in scheduler.waiting if r.status.name=='PREEMPTED']
            old_swap=tracker.last_swap_step
            proposal=tracker.decide(step,[RequestView(r.request_id,r.computed,r.prompt,r.prompt+r.max_output,r.output) for r in rows],
                [r.request_id for r in waiting],pool.get_num_free_blocks(),{r.request_id:view(r).remaining_blocks for r in waiting})
            if diagnostic:
                data['selector_decisions'].append(asdict(proposal))
            if proposal.action=='rotate':
                victim=view(scheduler.requests[proposal.victim_id]);target=view(scheduler.requests[proposal.resume_id])
                try:plan=prepare(step,victim,target,pool.get_num_free_blocks())
                except ValueError as exc:
                    tracker.last_swap_step=old_swap
                    data['events'].append(dict(event='prepare_rejected',step=step,reason=str(exc)))
                else:
                    phase='prepare'
                    data['events'].append(dict(event='prepare',step=step,victim=victim.request_id,target=target.request_id,saved_tokens=plan.saved_tokens))
        if protected is not None:
            if any(r is not protected for r in scheduler.skipped_waiting):
                raise RuntimeError('Unqualified unrelated blocked queue during protection')
            if protected.status.name=='PREEMPTED':
                scheduler.waiting.remove_request(protected);scheduler.waiting.prepend_request(protected)
        scheduler._rotation_target=protected.request_id if protected is not None else None

    def hold(req):
        if protected is None or req is protected:return False
        r=view(req)
        if not r.pure_decode:raise RuntimeError('Concurrent unqualified recovery')
        return not recovery_guard(view(protected),pool.get_num_free_blocks(),r.remaining_blocks)['other_growth_allowed']

    def schedule(*args,**kwargs):
        nonlocal step,protected,plan
        if open_population and protected is not None and protected.is_finished():
            data['events'].append(dict(event='target_terminal',step=step,request=protected.request_id,
                new_output_tokens=protected.num_output_tokens-output_start,status=protected.status.name))
            protected=None
        if open_population and protected is not None and protected.request_id not in scheduler.requests:
            raise RuntimeError('Protected request disappeared without terminal status')
        if protected is not None and protected.num_output_tokens>output_start:
            data['events'].append(dict(event='target_new_output',step=step,request=protected.request_id))
            protected=None
        try:
            result=native(*args,**kwargs);meta=result.kv_connector_metadata
            if phase=='prepare':
                if result.num_scheduled_tokens.get(plan.victim.request_id)!=1:
                    raise RuntimeError('Preparation victim did not decode exactly once')
                if store_scope=='selected':
                    delta=inspect_store_delta(plan,meta,cs._req_status[plan.victim.request_id],cs._jobs)
                    data['events'].append(dict(event='store_delta',step=step,**delta))
            if store_scope=='native_full' and diagnostic and (meta.store_jobs or meta.jobs_to_flush):
                evidence=dict(event='native_full_metadata',step=step,phase=phase,
                    host_perf_counter_s=time.perf_counter(),store_jobs=[],flush_jobs=[],
                    attempted_store_jobs=[dict(job_id=j,request=x.req_id,source_gpu_blocks=list(x.src_spec.block_ids))
                        for j,x in meta.store_jobs.items()])
                data['events'].append(evidence)
                try:
                    evidence.update(inspect_native_full_jobs(result,cs,plan if phase=='prepare' else None))
                    evidence.pop('attempted_store_jobs')
                except Exception as exc:
                    evidence['error']=repr(exc)
                    raise
            if phase=='commit' and not cancelled:
                if plan.victim.request_id not in (result.preempted_req_ids or ()):
                    raise RuntimeError('Native preemption notification absent')
                if not pending_flush<=set(meta.jobs_to_flush):
                    raise RuntimeError('Native flush omitted pending stores')
                tracker.note_rotation_applied()
            if protected is not None:
                allowed={plan.victim.request_id} if phase=='commit' and plan else set()
                if set(result.preempted_req_ids or ())-allowed:
                    raise RuntimeError('Unexpected natural preemption during protection')
                state=view(protected);tokens=result.num_scheduled_tokens.get(protected.request_id,0)
                if pool.get_num_free_blocks()<state.remaining_blocks:raise RuntimeError('Target reserve consumed')
                if state.status=='WAITING_FOR_REMOTE_KVS':
                    if tokens:raise RuntimeError('Pending load executed')
                elif tokens<=0:raise RuntimeError('Ready target made no progress')
            tracker.note_preempted(step,sorted(result.preempted_req_ids or ()))
            tracker.note_resumed(step,sorted(result.scheduled_cached_reqs.resumed_req_ids))
            if meta.store_jobs or meta.load_jobs or meta.jobs_to_flush or phase:
                event=dict(event='metadata',step=step,stores=sorted(meta.store_jobs),loads=sorted(meta.load_jobs),flush=sorted(meta.jobs_to_flush))
                if diagnostic:
                    event['jobs']=[dict(job_id=jid,request=cs._jobs[jid].req_id,
                        is_store=cs._jobs[jid].is_store)
                        for jid in sorted(set(meta.store_jobs)|set(meta.load_jobs)|set(meta.jobs_to_flush))
                        if jid in cs._jobs]
                data['events'].append(event)
            if phase in ('commit','direct_resume'):plan=None
            step+=1;data['status']='EXECUTING'
            return result
        except Exception as exc:
            data.update(status='ERROR',error=repr(exc));raise

    scheduler._rotation_begin=begin;scheduler._rotation_hold=hold
    scheduler._rotation_target=None;scheduler._rotation_forced_count=0
    scheduler.schedule=schedule
    if store_scope=='selected':cs._calc_num_offloadable_tokens=limited
    def uninstall():
        for key in ('schedule',)+hooks:delattr(scheduler,key)
        if store_scope=='selected':
            if hadcalc:cs._calc_num_offloadable_tokens=oldcalc
            else:delattr(cs,'_calc_num_offloadable_tokens')
        if data['status']!='ERROR':data['status']='DRAINED' if not scheduler.requests else 'UNINSTALLED_WITH_PENDING_REQUESTS'
        data['schedule_calls']=step
        return data
    return data,uninstall
