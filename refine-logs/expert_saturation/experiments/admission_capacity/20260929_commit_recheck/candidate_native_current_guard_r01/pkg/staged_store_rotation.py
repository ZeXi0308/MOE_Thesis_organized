"""Repeated staged most-output native adapter; GPU validation UNRUN.

Uses native offload metadata and flushes. Incremental stores use native residency and fallback.
Install before capture wrappers, on the pinned drained synchronous scheduler.
"""
import hashlib
import inspect
import os
import time
from dataclasses import asdict
from types import MethodType
from staged_save_contract import RequestState, prepare, commit_reason, recovery_guard
from rotation_native import patched_schedule_tree, SCHEDULER_SHA256
from absence_rotation import AbsenceRotation, RequestView, RotationConfig
from native_store_delta import inspect_store_delta
from native_full_store_evidence import inspect_native_full_jobs
from bidkv_score import bidkv_score, bidkv_order


def _rank_native_victim(rows, rule):
    if rule=='arrival':return max(rows,key=lambda r:(r['arrival_time'],r['index']))['index']
    if rule=='service_density':return max(rows,key=lambda r:(r['density'],r['index']))['index']
    if rule=='bidkv_score':return min(rows,key=bidkv_order)['index']
    return rows[-1]['index']


def _native_victim_current_guard(rows, rule, unknown, enabled):
    """Retain the failed current request while another qualified victim exists."""
    default=rows[-1]['index']
    unrestricted=default if unknown else _rank_native_victim(rows,rule)
    eligible=not unknown and len(rows)>1 and unrestricted==rows[0]['index']
    selected=_rank_native_victim(rows[1:],rule) if enabled and eligible else unrestricted
    return selected,unrestricted,eligible


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


def _direct_resume_reason(scheduler, manager, pool, owned, cs, target):
    """Conservative, read-only commit-time gate; native still admits the target."""
    if target.status.name!='PREEMPTED' or target not in scheduler.waiting:
        return 'KEEP_TARGET_NOT_WAITING'
    if target.is_finished() or scheduler.skipped_waiting:
        return 'KEEP_TERMINAL_OR_PENDING_QUEUE'
    slots=getattr(scheduler,'max_num_running_reqs',None)
    streaming=getattr(scheduler,'num_waiting_for_streaming_input',None)
    if (type(slots) is not int or type(streaming) is not int
            or streaming<0 or len(scheduler.running)+streaming>=slots):
        return 'KEEP_NO_SEQUENCE_SLOT'
    if owned.get(target.request_id):
        return 'KEEP_PARTIAL_TARGET_KV'
    need=(target.num_prompt_tokens+target.num_output_tokens+15)//16
    if pool.get_num_free_blocks()<need:
        return 'KEEP_INSUFFICIENT_FREE_BLOCKS'
    status=cs._req_status.get(target.request_id)
    if status is None or getattr(status,'transfer_jobs',None) is None:
        return 'KEEP_UNKNOWN_TARGET_TRANSFER'
    if status.transfer_jobs:
        return 'KEEP_PENDING_TARGET_TRANSFER'
    seen=set();running_ids={request.request_id for request in scheduler.running}
    if any(blocks for rid,blocks in owned.items() if rid not in running_ids):
        return 'KEEP_NONRUNNING_BLOCKS'
    try:
        for request in scheduler.running:
            blocks=owned[request.request_id]
            ids=[block.block_id for block in blocks]
            if ids!=manager.get_blocks(request.request_id).get_block_ids()[0]:
                return 'KEEP_MANAGER_OWNERSHIP_MISMATCH'
            for block in blocks:
                if (block.is_null or block.ref_cnt!=1 or block.block_id in seen
                        or pool.blocks[block.block_id] is not block):
                    return 'KEEP_SHARED_OR_INVALID_BLOCK'
                seen.add(block.block_id)
    except (AttributeError,IndexError,KeyError,TypeError):
        return 'KEEP_UNKNOWN_OWNERSHIP'
    return 'DIRECT_READY'


def _native_reservation_disposition(scheduler, disposition):
    """Defensive commit gate; an unreadable native reservation preserves the victim path."""
    if disposition != 'DIRECT_READY':
        return disposition, None
    try:
        probe = getattr(scheduler, '_inflight_prefill_reserved_blocks', None)
        reserved = probe() if callable(probe) else None
    except Exception:
        reserved = None
    if type(reserved) is not int or reserved < 0:
        return 'KEEP_UNKNOWN_NATIVE_RESERVATION', None
    if reserved > 0:
        return 'KEEP_NATIVE_INFLIGHT_RESERVATION', reserved
    return disposition, 0


def install(scheduler, *, vllm_config, save, block_size, expected_requests=32,
            diagnostic=False, global_cooldown_steps=20, population_mode="closed", store_scope="selected",
            commit_recheck=False, fit_first_resume=False, capacity_victim=False,
            recovery_min_outputs=1, yield_to_ready_head=False, spare_followup=False,
            ordinary_backfill=False, allow_forced_rotations=True):
    if spare_followup and (recovery_min_outputs!=1 or not capacity_victim or yield_to_ready_head or fit_first_resume or commit_recheck):
        raise ValueError('Spare followup requires capacity-only Q1')
    if ordinary_backfill and (spare_followup or recovery_min_outputs!=1
            or (not capacity_victim and allow_forced_rotations)
            or yield_to_ready_head or fit_first_resume or commit_recheck):
        raise ValueError('Ordinary backfill requires capacity-only Q1 without spare followup')
    if not allow_forced_rotations and (store_scope!='native_full' or not ordinary_backfill
            or capacity_victim or spare_followup or fit_first_resume or commit_recheck
            or yield_to_ready_head or recovery_min_outputs!=1):
        raise ValueError('No forced rotations is reserved for native-full ordinary backfill')
    if type(recovery_min_outputs) is not int or recovery_min_outputs not in (1,10):
        raise ValueError('Only recovery_min_outputs=1/10 is supported by this probe')
    if recovery_min_outputs!=1 and (not capacity_victim or fit_first_resume or commit_recheck):
        raise ValueError('Extended recovery requires capacity victim only')
    if yield_to_ready_head and (recovery_min_outputs!=10 or not capacity_victim):
        raise ValueError('Yield requires capacity-qualified Q10')
    if capacity_victim and (fit_first_resume or commit_recheck):
        raise ValueError('Capacity-victim probe requires fit_first_resume=False and commit_recheck=False')
    if fit_first_resume and commit_recheck:
        raise ValueError('Fit-first probe requires commit_recheck=False')
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
    step=0;plan=None;plan_request_refs=None;protected=None;output_start=None;cohort=None;phase=None
    protection_output_goal=None;protection_first_output_seen=False;protection_extended=False
    pending_yield=None
    protection_origin=None;protection_start_step=None;protection_primary_victim=None
    followup_pending=None;followup_choice=None;ordinary_choice=None
    capacity_plan_choice=None
    pending_flush=set();cancelled=False
    if open_population:cohort=set()
    tracker=AbsenceRotation(config=RotationConfig(min_steps_between_swaps=global_cooldown_steps),
        victim_order='most_output')
    data=dict(save=save,store_scope=store_scope,native_calc_overridden=store_scope=='selected',
        allow_forced_rotations=allow_forced_rotations,
        events=[],status='INSTALLED',applied_rotations=0,direct_commits=0,
        commit_recheck=commit_recheck,fit_first_resume=fit_first_resume,fit_first_resumes=0,
        capacity_victim=capacity_victim,capacity_victim_commits=0,
        recovery_min_outputs=recovery_min_outputs, yield_to_ready_head=yield_to_ready_head,
        yield_gate_counts={}, yield_releases=0,
        spare_followup=spare_followup,spare_followup_admissions=0,spare_followup_gate_counts={},
        ordinary_backfill=ordinary_backfill,ordinary_backfill_choices=0,
        ordinary_backfill_admissions=0,ordinary_backfill_gate_counts={},
        native_reservation_gate=dict(checked=0,zero=0,positive_keep=0,unknown_keep=0),
        diagnostic=diagnostic, population_mode=population_mode, gate_observations=[],
        rotation_config=asdict(tracker.config),
        eligibility_snapshots=[], selector_decisions=[],
        eligibility_logging='ENABLED' if diagnostic else 'DISABLED')
    victim_rule=os.environ.get('A_NATIVE_VICTIM_RULE','tail')
    if victim_rule not in ('tail','arrival','service_density','bidkv_score'):
        raise ValueError('Unknown native victim rule')
    if allow_forced_rotations or store_scope!='native_full' or not ordinary_backfill:
        raise ValueError('Residency victim pilot requires native full ordinary-only')
    residence={}
    data['native_victim_rule']=victim_rule
    data['victim_decisions']=[]
    data['residency_admissions']=[]
    continue_mode=os.environ.get('A_SELF_PREEMPT_CONTINUE','off')
    if continue_mode not in ('off','on'):raise ValueError('Unknown self-preempt continuation mode')
    data['self_preempt_continue_enabled']=continue_mode=='on'
    current_guard=os.environ.get('A_NATIVE_VICTIM_CURRENT_GUARD','off')
    if current_guard not in ('off','on'):raise ValueError('Unsupported current victim guard')
    data['current_victim_guard_enabled']=current_guard=='on'
    data['self_preempt_continuations']=[]
    pending_continuation_visit=None
    hooks=('_rotation_begin','_rotation_hold','_rotation_target','_rotation_forced_count',
        '_rotation_yield_to_ready_head','_rotation_pick_victim','_rotation_admit_running',
        '_rotation_continue_after_self_preempt')
    if any(k in vars(scheduler) for k in hooks):raise ValueError('Existing rotation hook')

    def view(req):
        if req is None:return None
        blocks=owned.get(req.request_id,())
        if any(b.is_null or pool.blocks[b.block_id] is not b for b in blocks):
            raise RuntimeError('Invalid physical ownership')
        return RequestState(req.request_id,req.num_computed_tokens,req.num_prompt_tokens,
            req.num_output_tokens,req.max_tokens,req.status.name,tuple(b.block_id for b in blocks))

    def admit_running(request):
        # Native calls this after any asynchronous load has completed, before
        # the first scheduled computation of this residence. No extra scan.
        scheduler.running.append(request)
        residence[request.request_id]=(request.num_preemptions,request.num_output_tokens)
        data['residency_admissions'].append(dict(step=step,request=request.request_id,
            host_perf_counter_s=time.perf_counter(),num_preemptions=request.num_preemptions,
            output_count=request.num_output_tokens,held_blocks=len(owned.get(request.request_id,()))))

    def pick_victim(req_index):
        # Only the unprocessed suffix is eligible: native need not roll back
        # tokens or allocations already assigned earlier in this scheduler step.
        running=scheduler.running
        if not 0<=req_index<len(running):raise RuntimeError('Invalid victim suffix')
        default=len(running)-1
        rows=[];unknown=False
        for index in range(req_index,len(running)):
            req=running[index];state=view(req);epoch=residence.get(req.request_id)
            valid=(state.pure_decode and state.status=='RUNNING' and bool(state.blocks)
                and epoch is not None and epoch[0]==req.num_preemptions
                and req.num_output_tokens>=epoch[1])
            if not valid:unknown=True
            served=req.num_output_tokens-epoch[1] if valid else None
            rows.append(dict(index=index,request=req.request_id,arrival_time=req.arrival_time,
                held_blocks=len(state.blocks),residency_outputs=served,
                density=served/len(state.blocks) if valid else None,qualified=valid,
                bidkv=bidkv_score(req)))
        selected,unrestricted,guard_eligible=_native_victim_current_guard(
            rows,victim_rule,unknown,current_guard=='on')
        data['victim_decisions'].append(dict(step=step,host_perf_counter_s=time.perf_counter(),
            rule=victim_rule,failed_request=running[req_index].request_id,
            unrestricted_selected=running[unrestricted].request_id,
            current_guard_eligible=guard_eligible,current_guard_applied=selected!=unrestricted,
            native_tail=running[default].request_id,selected=running[selected].request_id,
            changed=selected!=default,unprocessed_suffix_start=req_index,
            free_blocks=pool.get_num_free_blocks(),fallback_unknown=unknown,candidates=rows))
        residence.pop(running[selected].request_id,None)
        return selected

    def continue_after_self_preempt(request, req_index):
        nonlocal pending_continuation_visit
        running=scheduler.running
        decision=data['victim_decisions'][-1] if data['victim_decisions'] else None
        if (decision is None or decision['step']!=step
                or decision['selected']!=request.request_id
                or decision['failed_request']!=request.request_id
                or request.status.name!='PREEMPTED' or request in running
                or not 0<=req_index<len(running)):
            return False
        disposition,reserved=_native_reservation_disposition(scheduler,'DIRECT_READY')
        eligible=(not decision['fallback_unknown'] and protected is None
                  and phase is None and disposition=='DIRECT_READY')
        applied=continue_mode=='on' and eligible
        row=dict(step=step,request=request.request_id,req_index=req_index,
                 enabled=continue_mode=='on',eligible=eligible,applied=applied,
                 protected=protected.request_id if protected is not None else None,
                 phase=phase,native_reservation_disposition=disposition,
                 native_inflight_reserved_blocks=reserved,
                 suffix_ids=[r.request_id for r in running[req_index:]],
                 expected_next_request=running[req_index].request_id,
                 next_running_visited=None,host_perf_counter_s=time.perf_counter())
        data['self_preempt_continuations'].append(row)
        if applied:pending_continuation_visit=row
        return applied

    def limited(rs,n):
        if not save or phase!='prepare' or plan is None or rs.req.request_id!=plan.victim.request_id:return 0
        return min(oldcalc(rs,n),plan.saved_tokens)

    def protection_resources():
        state=view(protected)
        # Producing output number G requires computing through prompt+G-1.
        # This is the full future computation endpoint, with no extra block.
        compute_tokens=state.prompt+protection_output_goal-1
        total_blocks=(compute_tokens+15)//16
        disposition,reserved=_native_reservation_disposition(scheduler,'DIRECT_READY')
        return dict(free_blocks=pool.get_num_free_blocks(),held_blocks=len(state.blocks),
            future_compute_tokens=compute_tokens,future_required_blocks=total_blocks,
            future_growth_blocks=max(0,total_blocks-len(state.blocks)),
            native_inflight_reserved_blocks=reserved,native_reservation_disposition=disposition)

    def start_protection(target,origin='REGULAR_COMMIT',primary_victim=None):
        nonlocal protected,output_start,protection_output_goal,protection_first_output_seen,protection_extended
        nonlocal protection_origin,protection_start_step,protection_primary_victim
        protection_origin=origin;protection_start_step=step;protection_primary_victim=primary_victim
        protected=target;output_start=target.num_output_tokens
        protection_output_goal=min(output_start+recovery_min_outputs,target.max_tokens)
        protection_first_output_seen=False;protection_extended=False
        data['events'].append(dict(event='protection_start',step=step,request=target.request_id,
            protection_origin=protection_origin,primary_victim=protection_primary_victim,
            host_perf_counter_s=time.perf_counter(),recovery_min_outputs=recovery_min_outputs,
            output_count=target.num_output_tokens,new_output_tokens=0,output_goal=protection_output_goal,
            native_admission='PENDING',**protection_resources()))

    def release_protection(reason,resources=None):
        nonlocal protected,protection_output_goal,protection_first_output_seen,protection_extended
        nonlocal protection_origin,protection_start_step,protection_primary_victim
        data['events'].append(dict(event='protection_release',step=step,request=protected.request_id,
            protection_origin=protection_origin,protection_start_step=protection_start_step,
            host_perf_counter_s=time.perf_counter(),reason=reason,recovery_min_outputs=recovery_min_outputs,
            output_count=protected.num_output_tokens,new_output_tokens=protected.num_output_tokens-output_start,
            output_goal=protection_output_goal,extended=protection_extended,
            **(protection_resources() if resources is None else resources)))
        protected=None;protection_output_goal=None;protection_first_output_seen=False;protection_extended=False
        protection_origin=None;protection_start_step=None;protection_primary_victim=None

    def extension_reason(resources):
        if not view(protected).pure_decode or protected.status.name!='RUNNING':
            return 'KEEP_TARGET_NOT_PURE_DECODE'
        if scheduler.skipped_waiting:
            return 'KEEP_PENDING_QUEUE'
        if resources['native_reservation_disposition']!='DIRECT_READY':
            return resources['native_reservation_disposition']
        if resources['free_blocks']<resources['future_growth_blocks']:
            return 'KEEP_FUTURE_GROWTH_UNFUNDED'
        return 'EXTEND_READY'

    def future_growth_blocks():
        return max(0,(protected.num_prompt_tokens+protection_output_goal-1+15)//16
            -len(owned.get(protected.request_id,())))

    def try_yield_to_head(head,token_budget):
        nonlocal pending_yield
        if not yield_to_ready_head or not protection_extended:return False
        resources=protection_resources()
        need=(head.num_tokens+block_size-1)//block_size
        reason='YIELD_READY'
        if token_budget<=0:reason='KEEP_NO_TOKEN_BUDGET'
        elif resources['native_reservation_disposition']!='DIRECT_READY':
            reason=resources['native_reservation_disposition']
        elif resources['free_blocks']<need+resources['future_growth_blocks']:
            reason='KEEP_HEAD_AND_TARGET_UNFUNDED'
        elif cs._jobs:reason='KEEP_REGISTERED_TRANSFER_JOBS'
        else:
            probe=getattr(cs,'has_pending_push_work',None)
            try:pending=probe() if callable(probe) else None
            except Exception:pending=None
            if pending is not False:reason='KEEP_PENDING_OR_UNKNOWN_PUSH'
            else:
                reason=_direct_resume_reason(scheduler,manager,pool,owned,cs,head)
                if reason=='DIRECT_READY':reason='YIELD_READY'
        counts=data['yield_gate_counts'];counts[reason]=counts.get(reason,0)+1
        if reason!='YIELD_READY':return False
        pending_yield=dict(step=step,head=head.request_id,protected=protected.request_id,
            head_output_start=head.num_output_tokens,protected_output_count=protected.num_output_tokens,
            output_goal=protection_output_goal,remaining_token_budget=token_budget,
            head_full_history_tokens=head.num_tokens,head_need_blocks=need,
            **resources)
        data['events'].append(dict(event='protection_yield_to_ready_head',
            host_perf_counter_s=time.perf_counter(),**pending_yield))
        release_protection('YIELD_TO_READY_HEAD',resources)
        scheduler._rotation_target=None
        data['yield_releases']+=1
        return True

    def attempt_followup(context):
        nonlocal followup_choice
        counts=data['spare_followup_gate_counts']
        def reject(reason):
            counts[reason]=counts.get(reason,0)+1
            return False
        if scheduler.skipped_waiting or not scheduler.waiting:
            return reject('KEEP_PENDING_OR_EMPTY_QUEUE')
        head=scheduler.waiting.peek_request()
        if head.status.name!='PREEMPTED':return reject('KEEP_HEAD_NOT_PREEMPTED')
        free=pool.get_num_free_blocks()
        head_need=(head.num_prompt_tokens+head.num_output_tokens+15)//16
        if head_need<=free:return reject('KEEP_HEAD_ALREADY_FUNDED')
        disposition,reserved=_native_reservation_disposition(scheduler,'DIRECT_READY')
        if disposition!='DIRECT_READY':return reject(disposition)
        if cs._jobs:return reject('KEEP_REGISTERED_TRANSFER_JOBS')
        probe=getattr(cs,'has_pending_push_work',None)
        try:pending=probe() if callable(probe) else None
        except Exception:pending=None
        if pending is not False:return reject('KEEP_PENDING_OR_UNKNOWN_PUSH')
        if any(not view(r).pure_decode for r in scheduler.running):
            return reject('KEEP_MIXED_RUNNING_WORK')
        waiting=sorted(((step-tracker.absent_since[r.request_id],r.request_id,r)
            for r in scheduler.waiting if r.status.name=='PREEMPTED'
            and r.request_id in tracker.absent_since),key=lambda x:(x[0],x[1]),reverse=True)
        oldest=waiting[0][1] if waiting else None
        for age,rid,candidate in waiting:
            if age<tracker.config.min_absence_steps:continue
            reason=_direct_resume_reason(scheduler,manager,pool,owned,cs,candidate)
            if reason!='DIRECT_READY':continue
            followup_choice=dict(step=step,target=rid,primary_target=context['primary_target'],
                primary_victim=context['primary_victim'],primary_start_step=context['primary_start_step'],
                primary_release_step=context['primary_release_step'],
                displaced_oldest=oldest if oldest!=rid else None,queue_head=head.request_id,
                head_required_blocks=head_need,head_output_count=head.num_output_tokens,
                output_tokens_at_choice=candidate.num_output_tokens,absence_steps=age,
                free_blocks=free,required_blocks=view(candidate).remaining_blocks,
                native_inflight_reserved_blocks=reserved)
            data['events'].append(dict(event='spare_followup_choice',
                host_perf_counter_s=time.perf_counter(),**followup_choice))
            start_protection(candidate,origin='SPARE_FOLLOWUP')
            counts['FOLLOWUP_READY']=counts.get('FOLLOWUP_READY',0)+1
            return True
        return reject('KEEP_NO_ELIGIBLE_FUNDED_WAITER')

    def attempt_ordinary_backfill():
        """Try one native-fit waiter when the ordinary queue head cannot fit."""
        nonlocal ordinary_choice
        counts=data['ordinary_backfill_gate_counts']
        def reject(reason):
            counts[reason]=counts.get(reason,0)+1
            return False
        if scheduler.skipped_waiting or not scheduler.waiting:
            return reject('KEEP_PENDING_OR_EMPTY_QUEUE')
        head=scheduler.waiting.peek_request()
        if head.status.name!='PREEMPTED':return reject('KEEP_HEAD_NOT_PREEMPTED')
        free=pool.get_num_free_blocks()
        head_need=(head.num_prompt_tokens+head.num_output_tokens+15)//16
        if head_need<=free:return reject('KEEP_HEAD_ALREADY_FUNDED')
        disposition,reserved=_native_reservation_disposition(scheduler,'DIRECT_READY')
        if disposition!='DIRECT_READY':return reject(disposition)
        if cs._jobs:return reject('KEEP_REGISTERED_TRANSFER_JOBS')
        probe=getattr(cs,'has_pending_push_work',None)
        try:pending=probe() if callable(probe) else None
        except Exception:pending=None
        if pending is not False:return reject('KEEP_PENDING_OR_UNKNOWN_PUSH')
        if any(not view(r).pure_decode for r in scheduler.running):
            return reject('KEEP_MIXED_RUNNING_WORK')
        waiting=sorted(((step-tracker.absent_since[r.request_id],r.request_id,r)
            for r in scheduler.waiting if r.status.name=='PREEMPTED'
            and r.request_id in tracker.absent_since),key=lambda x:(x[0],x[1]),reverse=True)
        oldest=waiting[0][1] if waiting else None
        for age,rid,candidate in waiting:
            if age<tracker.config.min_absence_steps:continue
            reason=_direct_resume_reason(scheduler,manager,pool,owned,cs,candidate)
            if reason!='DIRECT_READY':continue
            ordinary_choice=dict(step=step,target=rid,queue_head=head.request_id,
                head_required_blocks=head_need,head_output_count=head.num_output_tokens,
                head_absence_steps=(step-tracker.absent_since[head.request_id]
                    if head.request_id in tracker.absent_since else None),
                oldest_waiter=oldest,displaced_oldest=oldest if oldest!=rid else None,
                output_tokens_at_choice=candidate.num_output_tokens,absence_steps=age,
                free_blocks=free,required_blocks=view(candidate).remaining_blocks,
                native_inflight_reserved_blocks=reserved,primary_context_present=False)
            data['events'].append(dict(event='ordinary_backfill_choice',
                host_perf_counter_s=time.perf_counter(),**ordinary_choice))
            start_protection(candidate,origin='ORDINARY_BACKFILL')
            data['ordinary_backfill_choices']+=1
            counts['BACKFILL_READY']=counts.get('BACKFILL_READY',0)+1
            return True
        return reject('KEEP_NO_ELIGIBLE_FUNDED_WAITER')

    def begin(preempted,timestamp):
        nonlocal plan,plan_request_refs,protected,output_start,phase,pending_flush,cancelled,cohort
        nonlocal capacity_plan_choice,followup_pending,followup_choice,ordinary_choice
        context=followup_pending;followup_pending=None;followup_choice=None;ordinary_choice=None
        scheduler._rotation_forced_count=0;phase=None;pending_flush=set();cancelled=False
        # Native-only ordinary backfill checks running ownership at its actual
        # admission boundary; it does not consume selector RequestState rows.
        # Keep the original rows for forced selection, diagnostic snapshots,
        # closed-cohort activation, or a pending commit.
        needs_rows=allow_forced_rotations or diagnostic or cohort is None or plan is not None
        rows=[view(r) for r in scheduler.running] if needs_rows else None
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
        if context is not None and attempt_followup(context):
            phase='spare_followup'
        elif ordinary_backfill and plan is None and protected is None and cohort is not None \
                and attempt_ordinary_backfill():
            phase='ordinary_backfill'
        elif plan is not None:
            phase='commit'
            victim=scheduler.requests.get(plan.victim.request_id);target=scheduler.requests.get(plan.target.request_id)
            # New-store coverage was checked at prepare. Older host residency
            # is intentionally unknown; native lookup/recompute handles misses.
            reason=commit_reason(plan,step,view(victim),view(target),pool.get_num_free_blocks(),None,save_enabled=False)
            if reason=='READY' and (plan_request_refs is None
                    or plan_request_refs[0] is not victim or plan_request_refs[1] is not target):
                reason='CANCEL_REQUEST_IDENTITY_CHANGED'
            if open_population and reason=='READY':
                if scheduler.skipped_waiting:reason='CANCEL_OPEN_SKIPPED_WAITING'
                elif any(not r.pure_decode for r in rows):reason='CANCEL_OPEN_MIXED_PREFILL_OR_RECOVERY'
            if reason=='READY':
                rs=cs._req_status.get(victim.request_id)
                pending_flush=set(rs.transfer_jobs) if rs else set()
                if any(j not in cs._jobs or not cs._jobs[j].is_store for j in pending_flush):
                    reason='CANCEL_VICTIM_PENDING_OR_UNKNOWN_TRANSFER'
            data['events'].append(dict(event='commit_check',step=step,reason=reason,victim=plan.victim.request_id,target=plan.target.request_id))
            if reason!='READY':
                cancelled=True;plan=None;plan_request_refs=None;capacity_plan_choice=None
            else:
                base_disposition=(_direct_resume_reason(scheduler,manager,pool,owned,cs,target)
                    if commit_recheck else 'KEEP_RECHECK_OFF')
                disposition,reserved=_native_reservation_disposition(scheduler,base_disposition)
                if base_disposition=='DIRECT_READY':
                    counters=data['native_reservation_gate'];counters['checked']+=1
                    if disposition=='DIRECT_READY':counters['zero']+=1
                    elif disposition=='KEEP_NATIVE_INFLIGHT_RESERVATION':counters['positive_keep']+=1
                    else:counters['unknown_keep']+=1
                data['events'].append(dict(event='commit_recheck',step=step,reason=disposition,
                    base_reason=base_disposition,native_inflight_reserved_blocks=reserved,
                    target=target.request_id,planned_victim=victim.request_id))
                if disposition=='DIRECT_READY':
                    phase='direct_resume'
                else:
                    scheduler.running.remove(victim);scheduler._preempt_request(victim,timestamp)
                    preempted.append(victim);scheduler._rotation_forced_count=1
                    data['applied_rotations']+=1
                start_protection(target,origin='REGULAR_COMMIT' if phase=='commit' else 'DIRECT_RESUME',
                    primary_victim=plan.victim.request_id if phase=='commit' else None)
        elif allow_forced_rotations and cohort is not None and protected is None and not scheduler.skipped_waiting and all(r.pure_decode for r in rows):
            waiting=[r for r in scheduler.waiting if r.status.name=='PREEMPTED']
            old_swap=tracker.last_swap_step
            proposal=tracker.decide(step,[RequestView(r.request_id,r.computed,r.prompt,r.prompt+r.max_output,r.output) for r in rows],
                [r.request_id for r in waiting],pool.get_num_free_blocks(),{r.request_id:view(r).remaining_blocks for r in waiting},
                reclaimable_blocks={r.request_id:len(r.blocks) for r in rows} if capacity_victim else None)
            if diagnostic:
                data['selector_decisions'].append(asdict(proposal))
            if proposal.action=='rotate':
                victim=view(scheduler.requests[proposal.victim_id]);target=view(scheduler.requests[proposal.resume_id])
                if fit_first_resume:
                    # Preserve the selector's longest-absence priority and id tie break.
                    # Only an already-qualified rotation proposal opens this action.
                    candidates=sorted((step-tracker.absent_since[r.request_id],r.request_id)
                        for r in waiting if r.request_id!=target.request_id
                        and r.request_id in tracker.absent_since
                        and step-tracker.absent_since[r.request_id]>=tracker.config.min_absence_steps)
                    for absence,rid in reversed(candidates):
                        candidate=scheduler.requests[rid]
                        base_disposition=_direct_resume_reason(scheduler,manager,pool,owned,cs,candidate)
                        disposition,reserved=_native_reservation_disposition(scheduler,base_disposition)
                        if base_disposition=='DIRECT_READY':
                            counters=data['native_reservation_gate'];counters['checked']+=1
                            if disposition=='DIRECT_READY':counters['zero']+=1
                            elif disposition=='KEEP_NATIVE_INFLIGHT_RESERVATION':counters['positive_keep']+=1
                            else:counters['unknown_keep']+=1
                        if disposition!='DIRECT_READY':continue
                        start_protection(candidate,origin='FIT_FIRST_RESUME')
                        phase='fit_first_resume'
                        data['events'].append(dict(event='fit_first_choice',step=step,
                            host_perf_counter_s=time.perf_counter(),output_tokens_at_choice=output_start,
                            original_target=target.request_id,planned_victim=victim.request_id,
                            target=rid,free_blocks=pool.get_num_free_blocks(),
                            required_blocks=view(candidate).remaining_blocks,
                            original_target_required_blocks=target.remaining_blocks,
                            absence_steps=absence,original_absence_steps=proposal.absence_steps,
                            native_inflight_reserved_blocks=reserved))
                        break
                if phase!='fit_first_resume':
                    try:plan=prepare(step,victim,target,pool.get_num_free_blocks())
                    except ValueError as exc:
                        tracker.last_swap_step=old_swap
                        data['events'].append(dict(event='prepare_rejected',step=step,reason=str(exc)))
                    else:
                        phase='prepare'
                        plan_request_refs=(scheduler.requests[victim.request_id],scheduler.requests[target.request_id])
                        data['events'].append(dict(event='prepare',step=step,victim=victim.request_id,target=target.request_id,saved_tokens=plan.saved_tokens))
                        if capacity_victim and proposal.original_victim_id!=victim.request_id:
                            original=view(scheduler.requests[proposal.original_victim_id])
                            capacity_plan_choice=dict(choice_step=step,target=target.request_id,
                                old_victim=original.request_id,new_victim=victim.request_id,
                                free_blocks=pool.get_num_free_blocks(),required_blocks=target.remaining_blocks,
                                old_held_blocks=len(original.blocks),new_held_blocks=len(victim.blocks))
                            data['events'].append(dict(event='capacity_victim_choice',step=step,
                                host_perf_counter_s=time.perf_counter(),**capacity_plan_choice))
        if protected is not None:
            if any(r is not protected for r in scheduler.skipped_waiting):
                raise RuntimeError('Unqualified unrelated blocked queue during protection')
            if protected.status.name=='PREEMPTED':
                scheduler.waiting.remove_request(protected);scheduler.waiting.prepend_request(protected)
        scheduler._rotation_target=protected.request_id if protected is not None else None

    def hold(req):
        nonlocal pending_continuation_visit
        if pending_continuation_visit is not None:
            if req.request_id!=pending_continuation_visit['expected_next_request']:
                raise RuntimeError('Continuation skipped the original next running request')
            pending_continuation_visit['next_running_visited']=req.request_id
            pending_continuation_visit=None
        if protected is None or req is protected:return False
        r=view(req)
        if not r.pure_decode:raise RuntimeError('Concurrent unqualified recovery')
        free=pool.get_num_free_blocks()
        allowed=recovery_guard(view(protected),free,r.remaining_blocks)['other_growth_allowed']
        if protection_extended:
            allowed=allowed and r.remaining_blocks<=free-future_growth_blocks()
        return not allowed

    def schedule(*args,**kwargs):
        nonlocal step,protected,plan,plan_request_refs,capacity_plan_choice
        nonlocal protection_first_output_seen,protection_extended,pending_yield,followup_pending
        pending_yield=None
        if protected is not None and protected.num_output_tokens>output_start and not protection_first_output_seen:
            data['events'].append(dict(event='target_new_output',step=step,request=protected.request_id,
                host_perf_counter_s=time.perf_counter(),recovery_min_outputs=recovery_min_outputs,
                output_count=protected.num_output_tokens,new_output_tokens=protected.num_output_tokens-output_start))
            protection_first_output_seen=True
        if (open_population or recovery_min_outputs!=1) and protected is not None and protected.is_finished():
            data['events'].append(dict(event='target_terminal',step=step,request=protected.request_id,
                host_perf_counter_s=time.perf_counter(),recovery_min_outputs=recovery_min_outputs,
                new_output_tokens=protected.num_output_tokens-output_start,status=protected.status.name))
            release_protection('TERMINAL_'+protected.status.name)
        if open_population and protected is not None and protected.request_id not in scheduler.requests:
            raise RuntimeError('Protected request disappeared without terminal status')
        if protected is not None and protection_first_output_seen:
            if protected.num_output_tokens>=protection_output_goal:
                if spare_followup and protection_origin=='REGULAR_COMMIT':
                    followup_pending=dict(primary_target=protected.request_id,
                        primary_victim=protection_primary_victim,primary_start_step=protection_start_step,
                        primary_release_step=step)
                release_protection('OUTPUT_GOAL_REACHED')
            else:
                resources=protection_resources();reason=extension_reason(resources)
                if reason!='EXTEND_READY':
                    # A changed/unknown reservation or lost budget returns to the
                    # original unprotected policy instead of keeping a blocked latch.
                    release_protection(reason,resources)
                elif not protection_extended:
                    protection_extended=True
                    data['events'].append(dict(event='protection_extend',step=step,request=protected.request_id,
                        host_perf_counter_s=time.perf_counter(),reason=reason,recovery_min_outputs=recovery_min_outputs,
                        output_count=protected.num_output_tokens,new_output_tokens=protected.num_output_tokens-output_start,
                        output_goal=protection_output_goal,**resources))
        try:
            continuation_start=len(data['self_preempt_continuations'])
            victim_start=len(data['victim_decisions'])
            result=native(*args,**kwargs);meta=result.kv_connector_metadata
            for event in data['victim_decisions'][victim_start:]:
                if event['current_guard_applied']:
                    event['failed_request_scheduled_tokens']=result.num_scheduled_tokens.get(event['failed_request'],0)
                    event['selected_scheduled_tokens']=result.num_scheduled_tokens.get(event['selected'],0)
                    event['failed_request_preempted_after_guard']=event['failed_request'] in (result.preempted_req_ids or ())
            for event in data['self_preempt_continuations'][continuation_start:]:
                event['suffix_scheduled_tokens']={
                    rid:result.num_scheduled_tokens.get(rid,0) for rid in event['suffix_ids']}
                event['source_scheduled_tokens']=result.num_scheduled_tokens.get(event['request'],0)
                event['suffix_preempted']=[rid for rid in event['suffix_ids']
                                          if rid in (result.preempted_req_ids or ())]
            if followup_choice is not None:
                target_id=followup_choice['target'];target=scheduler.requests.get(target_id)
                tokens=result.num_scheduled_tokens.get(target_id,0)
                jobs=sorted(jid for jid,job in meta.load_jobs.items()
                    if job.req_id==target_id and jid in cs._jobs
                    and cs._jobs[jid].req_id==target_id and not cs._jobs[jid].is_store)
                kind=('SCHEDULED_TOKENS' if tokens>0 else
                    'ASYNC_LOAD_ADMITTED' if target is not None and target.status.name=='WAITING_FOR_REMOTE_KVS'
                    and jobs and owned.get(target_id) else 'NO_NATIVE_ADMISSION')
                data['events'].append(dict(event='spare_followup_admission',step=step,target=target_id,
                    host_perf_counter_s=time.perf_counter(),native_admission=kind,
                    scheduled_tokens=tokens,load_job_ids=jobs,
                    primary_target=followup_choice['primary_target'],
                    primary_start_step=followup_choice['primary_start_step'],
                    held_blocks=len(owned.get(target_id,()))))
                if kind!='NO_NATIVE_ADMISSION':data['spare_followup_admissions']+=1
            if ordinary_choice is not None:
                target_id=ordinary_choice['target'];target=scheduler.requests.get(target_id)
                tokens=result.num_scheduled_tokens.get(target_id,0)
                jobs=sorted(jid for jid,job in meta.load_jobs.items()
                    if job.req_id==target_id and jid in cs._jobs
                    and cs._jobs[jid].req_id==target_id and not cs._jobs[jid].is_store)
                kind=('SCHEDULED_TOKENS' if tokens>0 else
                    'ASYNC_LOAD_ADMITTED' if target is not None and target.status.name=='WAITING_FOR_REMOTE_KVS'
                    and jobs and owned.get(target_id) else 'NO_NATIVE_ADMISSION')
                data['events'].append(dict(event='ordinary_backfill_admission',step=step,
                    target=target_id,host_perf_counter_s=time.perf_counter(),
                    native_admission=kind,scheduled_tokens=tokens,load_job_ids=jobs,
                    held_blocks=len(owned.get(target_id,())),
                    forced_preemptions_same_step=scheduler._rotation_forced_count,
                    actual_preempted_req_ids=sorted(result.preempted_req_ids or ())))
                if kind!='NO_NATIVE_ADMISSION':data['ordinary_backfill_admissions']+=1
            if pending_yield is not None:
                head_id=pending_yield['head']
                tokens=result.num_scheduled_tokens.get(head_id,0)
                head=scheduler.requests.get(head_id)
                jobs=sorted(jid for jid,job in meta.load_jobs.items()
                    if job.req_id==head_id and jid in cs._jobs
                    and cs._jobs[jid].req_id==head_id and not cs._jobs[jid].is_store)
                kind=('SCHEDULED_TOKENS' if tokens>0 else
                    'ASYNC_LOAD_ADMITTED' if head is not None
                    and head.status.name=='WAITING_FOR_REMOTE_KVS' and jobs and owned.get(head_id)
                    else 'NO_NATIVE_ADMISSION')
                data['events'].append(dict(event='yield_head_admission',
                    host_perf_counter_s=time.perf_counter(),step=step,head=head_id,
                    protected=pending_yield['protected'],head_output_start=pending_yield['head_output_start'],
                    native_admission=kind,scheduled_tokens=tokens,load_job_ids=jobs,
                    head_status=head.status.name if head is not None else 'ABSENT',
                    held_blocks=len(owned.get(head_id,())),
                    original_target_scheduled_tokens=result.num_scheduled_tokens.get(pending_yield['protected'],0)))
            if phase=='prepare':
                if result.num_scheduled_tokens.get(plan.victim.request_id)!=1:
                    raise RuntimeError('Preparation victim did not decode exactly once')
                if store_scope=='selected' and diagnostic:
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
                if protection_extended and pool.get_num_free_blocks()<future_growth_blocks():
                    raise RuntimeError('Future recovery growth reserve consumed')
                if state.status=='WAITING_FOR_REMOTE_KVS':
                    if tokens:raise RuntimeError('Pending load executed')
                elif tokens<=0:raise RuntimeError('Ready target made no progress')
            tracker.note_preempted(step,sorted(result.preempted_req_ids or ()))
            tracker.note_resumed(step,sorted(result.scheduled_cached_reqs.resumed_req_ids))
            if diagnostic and (meta.store_jobs or meta.load_jobs or meta.jobs_to_flush or phase):
                event=dict(event='metadata',step=step,stores=sorted(meta.store_jobs),loads=sorted(meta.load_jobs),flush=sorted(meta.jobs_to_flush))
                if diagnostic:
                    event['jobs']=[dict(job_id=jid,request=cs._jobs[jid].req_id,
                        is_store=cs._jobs[jid].is_store)
                        for jid in sorted(set(meta.store_jobs)|set(meta.load_jobs)|set(meta.jobs_to_flush))
                        if jid in cs._jobs]
                data['events'].append(event)
            if phase in ('commit','direct_resume','fit_first_resume') and not cancelled:
                target_id=protected.request_id
                direct_tokens=result.num_scheduled_tokens.get(target_id,0)
                load_job_ids=[]
                if direct_tokens>0:
                    admission='SCHEDULED_TOKENS'
                elif protected.status.name=='WAITING_FOR_REMOTE_KVS':
                    load_job_ids=sorted(jid for jid,job in meta.load_jobs.items()
                        if (job.req_id==target_id and jid in cs._jobs
                            and cs._jobs[jid].req_id==target_id and not cs._jobs[jid].is_store))
                    if not load_job_ids or not owned.get(target_id):
                        raise RuntimeError('Direct async target lacks native load admission')
                    admission='ASYNC_LOAD_ADMITTED'
                else:
                    raise RuntimeError('Direct target not admitted by native scheduler')
                if phase=='commit':
                    event_name='recovery_commit_admitted'
                elif phase=='fit_first_resume':
                    data['fit_first_resumes']+=1
                    event_name='fit_first_admitted'
                else:
                    data['direct_commits']+=1
                    event_name='direct_commit'
                data['events'].append(dict(event=event_name,step=step,target=target_id,
                    host_perf_counter_s=time.perf_counter(),
                    recovery_min_outputs=recovery_min_outputs,
                    planned_victim=plan.victim.request_id if phase=='commit' else None,
                    native_admission=admission,scheduled_tokens=direct_tokens,
                    load_job_ids=load_job_ids))
            if phase=='commit' and not cancelled and capacity_plan_choice is not None:
                data['capacity_victim_commits']+=1
                data['events'].append(dict(event='capacity_victim_commit',step=step,
                    host_perf_counter_s=time.perf_counter(),**capacity_plan_choice))
            if phase in ('commit','direct_resume'):plan=None;plan_request_refs=None;capacity_plan_choice=None
            if not allow_forced_rotations and (data['applied_rotations'] or scheduler._rotation_forced_count
                    or plan is not None):
                raise RuntimeError('Native-full ordinary arm attempted a forced rotation')
            step+=1;data['status']='EXECUTING'
            return result
        except Exception as exc:
            data.update(status='ERROR',error=repr(exc));raise

    scheduler._rotation_pick_victim=pick_victim;scheduler._rotation_admit_running=admit_running
    scheduler._rotation_continue_after_self_preempt=continue_after_self_preempt
    scheduler._rotation_begin=begin;scheduler._rotation_hold=hold
    scheduler._rotation_target=None;scheduler._rotation_forced_count=0
    scheduler._rotation_yield_to_ready_head=try_yield_to_head
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
