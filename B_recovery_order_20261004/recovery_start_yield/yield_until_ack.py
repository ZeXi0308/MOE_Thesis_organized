"""One selected recovery start: at most two native rounds, bounded by scheduler ACK."""
import hashlib
import importlib.util
import inspect
from pathlib import Path
import time

ONCE_SHA = '250b25a4260a7f59cbe7e183dab8d5e6fe96d8dfe1d6f6f9f441b609087d5fef'
path = Path(__file__).with_name('yield_once.py')
if hashlib.sha256(path.read_bytes()).hexdigest() != ONCE_SHA:
    raise RuntimeError('Frozen one-round yield changed')
spec = importlib.util.spec_from_file_location('yield_ack_frozen_once',path)
Y = importlib.util.module_from_spec(spec); spec.loader.exec_module(Y)
H = Y.H


def _decision(scheduler, manager, single, cs, mode, selective, now=time.perf_counter):
    if mode not in ('native','yield_ack'): raise ValueError(mode)
    initial, choose = Y._decision(scheduler,manager,single,cs,'native',selective,now)
    data = dict(mode=mode,status='INSTALLED',outcome='NO_OPPORTUNITY',action_count=0,
        requested_breaks=0,executed_breaks=0,decision_calls=0,skip_counts=initial['skip_counts'],events=[],
        clock=initial['clock'],frozen_once_sha256=ONCE_SHA,source_sha256=dict(H.PINS),
        scope='One selected recovery start; initial break plus at most one at the next schedule entry. Entry >= initial+2 unconditionally releases. Other native heads may proceed. No wall-clock gate.',
        ack_semantics='Only native scheduler removal from both original jobs and request transfer_jobs, with unchanged cache-reset threshold and matching request identity, is ACK retirement; GPU duration/host polling are not ACK.',
        timing_semantics='The bounded observation lifetime includes native work and gates, not counterfactual added or saved request latency.')
    step, selected, target, target_state = 0,None,None,None
    active, pending_break, reset_threshold = False,None,None
    tracked = {}

    def observe_ack():
        rows=[]
        reset = cs._stale_job_threshold != reset_threshold
        for jid,(job,state,request) in tracked.items():
            current_state=cs._req_status.get(request.request_id)
            current_job=cs._jobs.get(jid)
            same_state=current_state is state and state.req is request
            membership=jid in state.transfer_jobs if same_state else None
            same_job=current_job is job
            status=('UNKNOWN_RESET' if reset else 'UNKNOWN_REQUEST_IDENTITY' if not same_state
                else 'ACK_RETIRED' if current_job is None and membership is False
                else 'PENDING' if same_job and membership is True and current_job.req_id==request.request_id
                    and current_job.is_store is False and current_job.pending_count>0
                else 'UNKNOWN_JOB_STATE')
            rows.append(dict(job_id=jid,request=request.request_id,state=status,
                request_identity_match=same_state,job_identity_match=same_job,
                request_transfer_contains=membership,
                pending_count=current_job.pending_count if current_job is not None else None))
        unknown=next((r['state'] for r in rows if r['state'].startswith('UNKNOWN')),None)
        return rows,unknown,bool(rows) and all(r['state']=='ACK_RETIRED' for r in rows)

    def release(reason,ack_jobs=None):
        nonlocal active,pending_break
        data['events'].append(dict(kind='release',step=step,host_perf_s=now(),
            deferred_request=selected['deferred_request'],reason=reason,
            ack_jobs=ack_jobs,ack_observation_phase='decision' if ack_jobs is not None else None,
            requested_breaks=data['requested_breaks'],executed_breaks=data['executed_breaks']))
        selected['release_reason']=reason; active=False; pending_break=None

    def safety(current,budget,preempted,local=()):
        if current._rotation_target is not None or current._rotation_lease_enabled or current._rotation_forced_count:
            return 'BASELINE_PROTECTION'
        if any(r.status.name=='WAITING' for q in (current.waiting,current.skipped_waiting,local) for r in q):
            return 'NEW_WAITER_PRESENT'
        if (budget<=0 or preempted or current._pause_state.name!='UNPAUSED'
                or len(current.running)+current.num_waiting_for_streaming_input>=current.max_num_running_reqs):
            return 'NATIVE_WAITING_GATE_CLOSED'
        state=cs._req_status.get(target.request_id)
        if not any(r is target for r in current.waiting) or state is not target_state or state.req is not target:
            return 'TARGET_LEFT_WAITING_OR_IDENTITY_CHANGED'
        if (target.status.name!='PREEMPTED' or target.num_computed_tokens
                or single.req_to_blocks.get(target.request_id) or state.transfer_jobs
                or target.priority!=selected['candidate']['priority']
                or target.num_preemptions!=selected['candidate']['num_preemptions']):
            return 'TARGET_NOT_CLEAN_RECOVERY'
        return None

    def request_break():
        nonlocal pending_break
        if mode!='yield_ack': return False
        pending_break=dict(kind='actual_break_requested',step=step,host_perf_s=now(),
            ordinal=data['requested_breaks']+1,deferred_request=target.request_id)
        data['events'].append(pending_break); data['requested_breaks']+=1
        selected['requested_breaks']=data['requested_breaks']
        return True

    def callback(current,phase,budget,preempted,queue=None,request=None,local=None,
                 new_tokens=None,external=None,asynchronous=None,local_tokens=None,
                 lookahead=None,encoder_tokens=None):
        nonlocal step,selected,target,target_state,active,pending_break,reset_threshold
        args=(current,phase,budget,preempted,queue,request,local,new_tokens,external,
              asynchronous,local_tokens,lookahead,encoder_tokens)
        if phase=='entry':
            step+=1
            if selected is None:
                choose(*args)
            elif active:
                rows,error,all_acked=observe_ack()
                data['events'].append(dict(kind='ack_entry_observation',step=step,host_perf_s=now(),
                    deferred_request=target.request_id,initial_step=selected['step'],jobs=rows,
                    all_original_loads_ack_retired=all_acked,stale_job_threshold=cs._stale_job_threshold))
                reason=('TWO_ROUND_LIMIT' if step>=selected['step']+2 else error
                        or ('ALL_ORIGINAL_LOADS_ACK_RETIRED' if all_acked else safety(current,budget,preempted)))
                if reason: release(reason)
            return False
        if phase=='break_executed':
            if (mode!='yield_ack' or not active or pending_break is None
                    or pending_break['step']!=step or request is not target
                    or data['executed_breaks']>=2 or step>selected['step']+1):
                raise RuntimeError('Unqualified or repeated ACK-bounded break')
            stamp=now()
            data['executed_breaks']+=1; data['action_count']=1
            data['outcome']='YIELDED_UNTIL_ACK_OR_ROUND_LIMIT'
            selected.update(executed_breaks=data['executed_breaks'],final_action='WAITING_LOOP_BREAK')
            selected.setdefault('first_break_host_perf_s',stamp)
            data['events'].append(dict(kind='actual_break_executed',step=step,host_perf_s=stamp,
                ordinal=data['executed_breaks'],deferred_request=target.request_id))
            pending_break=None
            return False
        if phase!='decision': raise ValueError(phase)
        data['decision_calls']+=1
        if selected is None:
            choose(*args)
            if not initial['events']: return False
            selected=initial['events'][0]; target=request; target_state=cs._req_status[target.request_id]
            reset_threshold=cs._stale_job_threshold
            for beneficiary in selected['beneficiaries']:
                state=cs._req_status[beneficiary['request']]
                for jid in beneficiary['job_ids']: tracked[jid]=(cs._jobs[jid],state,state.req)
            rows,error,all_acked=observe_ack()
            if error or all_acked: raise RuntimeError('Selected LOAD changed during synchronous decision')
            selected.update(initial_jobs=rows,initial_stale_job_threshold=reset_threshold,
                requested_breaks=0,executed_breaks=0,
                final_action='SHADOW_ONLY' if mode=='native' else 'BREAK_REQUESTED')
            data['events'].append(selected); active=True
            data['outcome']='SHADOW_ONLY' if mode=='native' else 'BREAK_REQUESTED'
            return request_break()
        if not active: return False
        reason=safety(current,budget,preempted,local)
        if reason: release(reason); return False
        if request is not target:
            data['events'].append(dict(kind='native_other_head_pass_through',step=step,host_perf_s=now(),
                deferred_request=target.request_id,native_head=request.request_id,
                selected_queue='waiting' if queue is current.waiting else 'skipped',
                local_skipped_order=[r.request_id for r in local]))
            return False
        if step!=selected['step']+1 or data['executed_breaks']>=2:
            release('TWO_ROUND_LIMIT'); return False
        rows,error,all_acked=observe_ack()
        if error or all_acked: release(error or 'ALL_ORIGINAL_LOADS_ACK_RETIRED',rows); return False
        # Reuse the frozen selection predicate at the actual allocation boundary;
        # no extra lookup, reference change, or choice of a different request.
        check,verify=Y._decision(scheduler,manager,single,cs,'native',selective,now)
        verify(current,'entry',budget,preempted); verify(*args)
        if not check['events']:
            release('RECHECK_'+next(iter(check['skip_counts']),'NO_OPPORTUNITY')); return False
        recheck=check['events'][0]
        visible={jid for b in recheck['beneficiaries'] for jid in b['job_ids']}
        if any(r['job_id'] not in visible for r in rows if r['state']=='PENDING'):
            release('ORIGINAL_PENDING_LOAD_NOT_IN_EARLIER_SKIPPED'); return False
        data['events'].append(dict(kind='continued_legal_boundary',step=step,host_perf_s=now(),
            deferred_request=target.request_id,jobs=rows,candidate=recheck['candidate'],
            native_locals=recheck['native_locals'],token_budget=budget,
            waiting_order=recheck['waiting_order'],local_skipped_order=recheck['local_skipped_order']))
        return request_break()
    return data,callback


def _attach(scheduler,manager,single,cs,mode,selective,tree,now=time.perf_counter):
    slot=H._native_slot(scheduler.schedule,scheduler); old=slot.cell_contents
    rebuilt=H._compile(tree,old)
    if not H._same_code(rebuilt.__func__.__code__,old.__func__.__code__):
        raise RuntimeError('Native schedule differs from frozen reconstruction')
    data,callback=_decision(scheduler,manager,single,cs,mode,selective,now)
    new=H._compile(Y._instrument(tree),old,callback); slot.cell_contents=new
    data['native_code_verified']=True; attached=True
    def uninstall():
        nonlocal attached
        if attached:
            if slot.cell_contents is not new: raise RuntimeError('Native schedule slot changed')
            slot.cell_contents=old; attached=False; data['status']='UNINSTALLED'
        return data
    return data,uninstall


def install(scheduler,mode='native',*,selective):
    if mode not in ('native','yield_ack'): raise ValueError(mode)
    _,undo=Y.install(scheduler,'native',selective=selective); undo()
    qualifier=Y.FIT._load('tail_reservation/reserve_tail.py','yield_ack_qualifier',H.PINS['tail_reservation/reserve_tail.py'])
    manager,single,reason=qualifier.qualify(scheduler)
    cs=scheduler.connector.connector_scheduler
    if reason or type(cs._stale_job_threshold) is not int:
        raise RuntimeError('Qualified allocator and native reset generation required')
    import rotation_native
    tree=rotation_native.patched_schedule_tree(Path(inspect.getsourcefile(type(scheduler))).read_text())
    return _attach(scheduler,manager,single,cs,mode,selective,tree)
