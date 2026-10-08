"""One native recovery LOAD start, waiting for cohort finish or one block epoch."""
import ast
import hashlib
import importlib.util
import inspect
from pathlib import Path
import time

BASE=Path(__file__).resolve().parents[1]
ONCE_SHA='250b25a4260a7f59cbe7e183dab8d5e6fe96d8dfe1d6f6f9f441b609087d5fef'
path=BASE/'recovery_start_yield/yield_once.py'
if hashlib.sha256(path.read_bytes()).hexdigest()!=ONCE_SHA: raise RuntimeError('Frozen waiting boundary changed')
spec=importlib.util.spec_from_file_location('start_gate_frozen_boundary',path)
Y=importlib.util.module_from_spec(spec);spec.loader.exec_module(Y)
H=Y.H


def _instrument(tree):
    result=Y._instrument(tree);count=0
    for node in ast.walk(result):
        if (isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id==H.CALLBACK
                and isinstance(node.args[1],ast.Constant) and node.args[1].value in ('decision','break_executed')):
            node.args.extend([ast.Name(id='new_computed_blocks',ctx=ast.Load()),ast.Name(id='reserved_blocks',ctx=ast.Load())])
            count+=1
    if count!=2:raise RuntimeError('Expected decision and physical-break callbacks')
    return ast.fix_missing_locations(result)


def _fit(manager,single,cs,request,new_tokens,external,asynchronous,local_tokens,lookahead,encoder_tokens,new_blocks,reserved):
    rid=request.request_id
    if (request.status.name!='PREEMPTED' or request.num_preemptions<1 or request.num_computed_tokens
            or single.req_to_blocks.get(rid) or request.num_in_flight_tokens or request.has_encoder_inputs
            or request.skip_reading_prefix_cache is not False or rid in single._partial_hit_reqs):
        return None,'NOT_CLEAN_PREEMPTED_RECOVERY'
    if (asynchronous is not True or new_tokens!=0 or type(external) is not int or external<=0
            or local_tokens!=0 or lookahead!=0 or encoder_tokens!=0
            or type(reserved) is not int or reserved<0 or not 0<external<=request.num_tokens<=manager.max_model_len
            or new_blocks is None or len(new_blocks.blocks)!=1 or any(new_blocks.blocks)):
        return None,'NOT_SUPPORTED_NATIVE_ASYNC_LOAD'
    state=cs._req_status.get(rid)
    if (state is None or state.req is not request or state.transfer_jobs
            or any(j.req_id==rid for j in cs._jobs.values())
            or any(j.req_id==rid for j in cs._current_batch_load_jobs.values())):
        return None,'TRANSFER_PREPARED_OR_IDENTITY_UNKNOWN'
    # These are the original allocator's two pure capacity queries. In this
    # qualified FullAttention/held0 case remove_skipped_blocks cannot free KV.
    common=dict(request_id=rid,new_computed_blocks=new_blocks.blocks,num_encoder_tokens=0,
                total_computed_tokens=external,num_local_computed_tokens=0)
    try:
        full=manager.coordinator.get_num_blocks_to_allocate(**common,num_tokens=request.num_tokens,
            num_tokens_main_model=request.num_tokens,apply_admission_cap=True)
        slots=manager.coordinator.get_num_blocks_to_allocate(**common,num_tokens=external,
            num_tokens_main_model=external)
        free=manager.block_pool.get_num_free_blocks()
    except Exception as error:
        return None,'UNKNOWN_READ_ONLY_CAPACITY_'+type(error).__name__
    if any(type(n) is not int or n<0 for n in (full,slots,free)):
        return None,'UNKNOWN_READ_ONLY_CAPACITY'
    row=dict(request=rid,priority=request.priority,num_preemptions=request.num_preemptions,
        history_tokens=request.num_tokens,computed_tokens=0,held_gpu_blocks=0,
        external_tokens=external,full_fit_blocks=full,slot_blocks=slots,
        native_reserved_blocks=reserved,required_free_blocks=max(full,slots+reserved),free_gpu_blocks=free,
        native_full_fit=free>=full,native_slot_reservation_fit=free-reserved>=slots,
        fit_source='Pinned native coordinator: full-history query and external-slot query; no allocate/lookup/touch')
    return row,None if row['native_full_fit'] and row['native_slot_reservation_fit'] else 'NATIVE_CAPACITY_SHORTFALL'


def _decision(scheduler,manager,single,cs,mode,selective,now=time.perf_counter):
    if mode not in ('native','wait_release'):raise ValueError(mode)
    bound=single.block_size
    if bound!=16:raise RuntimeError('This probe is fixed to the native block_size16 epoch')
    data=dict(mode=mode,status='INSTALLED',outcome='NO_OPPORTUNITY',action_count=0,requested_breaks=0,
        executed_breaks=0,decision_calls=0,skip_counts={},events=[],block_epoch_entries=bound,
        frozen_boundary_sha256=ONCE_SHA,source_sha256=dict(H.PINS),
        clock='Host perf_counter; no CUDA query or synchronization',
        scope='First legal preempted async LOAD only. Break its own native head until an initially RUNNING object is_finished() or entry>=initial+16; total breaks<=16. Other heads may progress. No queue/ref/victim/quantum/admission change.',
        timing_semantics='Observed selection lifetime includes native work/gates; it is not counterfactual added latency. Cohort finished status is a completion-event proxy, not proof of GPU copy completion or increased free capacity; release does not guarantee allocation.',
        lookup_semantics='Original scheduler lookup runs before this boundary on every attempt; this adapter adds no lookup or Host pin.')
    step,selected,target,target_state,reset=0,None,None,None,None
    active=False;cohort=[];pending_break=None

    def skip(reason):
        data['skip_counts'][reason]=data['skip_counts'].get(reason,0)+1
        return False

    def release(reason,finished=()):
        nonlocal active,pending_break
        data['events'].append(dict(kind='release',request=target.request_id,step=step,host_perf_s=now(),
            reason=reason,cohort_finished=list(finished),requested_breaks=data['requested_breaks'],executed_breaks=data['executed_breaks']))
        data['events'][-1]['free_gpu_blocks']=manager.block_pool.get_num_free_blocks()
        selected['release_reason']=reason;active=False;pending_break=None

    def gate(current,budget,preempted,local=()):
        if (budget<=0 or preempted or current._pause_state.name!='UNPAUSED'
                or len(current.running)+current.num_waiting_for_streaming_input>=current.max_num_running_reqs):
            return 'NATIVE_WAITING_GATE_CLOSED'
        if current._rotation_target is not None or current._rotation_lease_enabled or current._rotation_forced_count:
            return 'BASELINE_PROTECTION'
        if any(r.status.name=='WAITING' for q in (current.waiting,current.skipped_waiting,local) for r in q):
            return 'NEW_WAITER_PRESENT'
        return None

    def safety(current):
        if cs._stale_job_threshold!=reset:return 'CACHE_RESET'
        state=cs._req_status.get(target.request_id)
        if (not any(r is target for r in current.waiting) or state is not target_state or state.req is not target
                or target.priority!=selected['candidate']['priority']
                or target.num_preemptions!=selected['candidate']['num_preemptions']):
            return 'TARGET_IDENTITY_OR_EPISODE_CHANGED'
        if (target.status.name!='PREEMPTED' or target.num_computed_tokens
                or single.req_to_blocks.get(target.request_id) or state.transfer_jobs):
            return 'TARGET_NO_LONGER_UNPREPARED'
        return None

    def request_break():
        nonlocal pending_break
        if mode=='native':return False
        pending_break=step;data['requested_breaks']+=1
        selected['requested_breaks']=data['requested_breaks']
        data['events'].append(dict(kind='actual_break_requested',request=target.request_id,
            step=step,ordinal=data['requested_breaks'],host_perf_s=now()))
        return True

    def callback(current,phase,budget,preempted,queue=None,request=None,local=None,
                 new_tokens=None,external=None,asynchronous=None,local_tokens=None,
                 lookahead=None,encoder_tokens=None,new_blocks=None,reserved=None):
        nonlocal step,selected,target,target_state,reset,active,cohort,pending_break
        if phase=='uninstall':
            if active:release('UNINSTALL')
            return False
        if phase=='entry':
            step+=1
            if active:
                finished=[dict(request=r.request_id,status=r.status.name,
                    held_gpu_blocks=len(single.req_to_blocks.get(r.request_id,()))) for r in cohort if r.is_finished()]
                data['events'].append(dict(kind='entry_observation',request=target.request_id,
                    step=step,initial_step=selected['step'],host_perf_s=now(),cohort_finished=finished,
                    free_gpu_blocks=manager.block_pool.get_num_free_blocks(),
                    current_target_state=dict(status=target.status.name,computed_tokens=target.num_computed_tokens,
                        held_gpu_blocks=len(single.req_to_blocks.get(target.request_id,())))))
                reason=(safety(current) or ('COHORT_REQUEST_FINISHED' if finished else None)
                        or ('BLOCK_EPOCH_LIMIT' if step>=selected['step']+bound else None)
                        or gate(current,budget,preempted))
                if reason:release(reason,finished)
            return False
        if phase=='break_executed':
            if (not active or mode!='wait_release' or pending_break!=step or request is not target
                    or data['executed_breaks']>=bound or step>=selected['step']+bound):
                raise RuntimeError('Unqualified or excess recovery-start break')
            pending_break=None;data['executed_breaks']+=1;data['action_count']=1
            data['outcome']='WAITED_FOR_COHORT_FINISH_OR_BLOCK_EPOCH'
            stamp=now();selected.update(executed_breaks=data['executed_breaks'],final_action='WAITING_LOOP_BREAK')
            selected.setdefault('first_break_host_perf_s',stamp)
            data['events'].append(dict(kind='actual_break_executed',request=target.request_id,
                step=step,ordinal=data['executed_breaks'],host_perf_s=stamp))
            return False
        if phase!='decision':raise ValueError(phase)
        data['decision_calls']+=1
        if selected is not None and not active:return False
        reason=gate(current,budget,preempted,local)
        if active:reason=reason or safety(current)
        if reason:
            if active:release(reason)
            else:skip(reason)
            return False
        if active and request is not target:
            data['events'].append(dict(kind='native_other_head_pass_through',request=target.request_id,
                native_head=request.request_id,step=step,host_perf_s=now()))
            return False
        if queue is not current.waiting or current.waiting.peek_request() is not request:
            if active:release('NOT_NATIVE_WAITING_HEAD')
            return skip('NOT_NATIVE_WAITING_HEAD')
        fit,reason=_fit(manager,single,cs,request,new_tokens,external,asynchronous,local_tokens,
                        lookahead,encoder_tokens,new_blocks,reserved)
        if reason:
            if active:release('RECHECK_'+reason)
            else:skip(reason)
            return False
        if selected is None:
            if not current.running:return skip('NO_INITIAL_RUNNING_COHORT')
            if any(r.status.name!='RUNNING' or not callable(getattr(r,'is_finished',None)) for r in current.running):
                return skip('UNSUPPORTED_RUNNING_COHORT')
            target=request;target_state=cs._req_status[request.request_id];reset=cs._stale_job_threshold
            cohort=list(current.running);active=True
            selected=dict(kind='legal_start_gate_opportunity',request=request.request_id,step=step,
                host_perf_s=now(),candidate=fit,token_budget=budget,
                native_action='CONTINUE_NATIVE_ASYNC_LOAD_ALLOCATION',
                candidate_action='WAIT_FOR_COHORT_FINISH_OR_BLOCK_EPOCH',
                final_action='SHADOW_ONLY' if mode=='native' else 'BREAK_REQUESTED',
                running_cohort=[dict(request=r.request_id,status=r.status.name,
                    held_gpu_blocks=len(single.req_to_blocks.get(r.request_id,()))) for r in cohort],
                waiting_order=[r.request_id for r in current.waiting],
                local_skipped_order=[r.request_id for r in local],initial_stale_job_threshold=reset,
                deadline_step=step+bound,requested_breaks=0,executed_breaks=0)
            data['events'].append(selected);data['outcome']='SHADOW_ONLY' if mode=='native' else 'BREAK_REQUESTED'
        else:
            data['events'].append(dict(kind='continued_legal_boundary',request=request.request_id,
                step=step,host_perf_s=now(),candidate=fit,token_budget=budget))
        return request_break()
    return data,callback


def _attach(scheduler,manager,single,cs,mode,selective,tree,now=time.perf_counter):
    slot=H._native_slot(scheduler.schedule,scheduler);old=slot.cell_contents
    rebuilt=H._compile(tree,old)
    if not H._same_code(rebuilt.__func__.__code__,old.__func__.__code__):raise RuntimeError('Native schedule drift')
    data,callback=_decision(scheduler,manager,single,cs,mode,selective,now)
    new=H._compile(_instrument(tree),old,callback);slot.cell_contents=new
    data['native_code_verified']=True;attached=True
    def uninstall():
        nonlocal attached
        if attached:
            if slot.cell_contents is not new:raise RuntimeError('Native schedule slot changed')
            callback(scheduler,'uninstall',0,[])
            slot.cell_contents=old;attached=False;data['status']='UNINSTALLED'
        return data
    return data,uninstall


def install(scheduler,mode='native',*,selective):
    if mode not in ('native','wait_release'):raise ValueError(mode)
    _,undo=Y.install(scheduler,'native',selective=selective);undo()
    qualifier=Y.FIT._load('tail_reservation/reserve_tail.py','start_gate_qualifier',H.PINS['tail_reservation/reserve_tail.py'])
    manager,single,reason=qualifier.qualify(scheduler);cs=scheduler.connector.connector_scheduler
    if reason or type(cs._stale_job_threshold) is not int:raise RuntimeError('Unsupported native allocator/reset generation')
    import rotation_native
    tree=rotation_native.patched_schedule_tree(Path(inspect.getsourcefile(type(scheduler))).read_text())
    return _attach(scheduler,manager,single,cs,mode,selective,tree)
