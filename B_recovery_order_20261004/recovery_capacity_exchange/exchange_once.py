"""One ordinary recovery-capacity exchange on top of frozen stall8.

Single private FullAttention pool only. A's immediate-release definition counts
ref_cnt==1 pages; its Host cost is materialized whole pages beyond the ready
contiguous CPU prefix. These are current facts, not predicted donor latency.
Native preempt frees logical ownership; native metadata/worker flush fences
physical reuse. No new STORE, LOAD, lookup, reference or allocator operation.
"""
import hashlib
import importlib.util
import math
from pathlib import Path
import time

BASE = Path(__file__).resolve().parents[1]
PINS = {
    'recovery_service_age/service_age.py': 'fd9c358d7aae9dfb475e5e06930d66c556ab2fed2cff74e82be1a481e5349b2f',
    'recovery_start_gate/progress_capacity.py': '8ed4aab468763f880f5f8fe3adbf2517ed741f4602d12d882ca4eaf8d9e7b73d',
    'pkg/staged_store_rotation.py': '11b986fe0a28fc8931e6785229935cd98027cbe8475a3d61c2ac792f29c428cb',
}


def _load(relative, name):
    path = BASE / relative
    if hashlib.sha256(path.read_bytes()).hexdigest() != PINS[relative]:
        raise RuntimeError('Frozen exchange dependency changed: ' + relative)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


FROZEN = _load('recovery_service_age/service_age.py', 'exchange_frozen_stall8')
CAPACITY = _load('recovery_start_gate/progress_capacity.py', 'exchange_joint_capacity')
make_capture = FROZEN.make_capture
HELPERS = FROZEN.REPEAT.FROZEN.HELPERS
MODES = ('stall8', 'exchange_once')


def _exclude_donor(joint, contribution, released):
    """Exact O(1) subtraction from the one qualified per-RID snapshot."""
    r = joint['inflight_remaining_blocks']-contribution['inflight_remaining_blocks']
    g = joint['running_increment_blocks']-contribution['running_increment_blocks']
    t = joint['target_total_blocks']; free = joint['free_gpu_blocks']+released
    return dict(status='KNOWN', target_total_blocks=t, target_current_history_blocks=t,
        target_growth_increment_blocks=0, inflight_remaining_blocks=r,
        running_increment_blocks=g, required_free_blocks=t+r+g,
        free_gpu_blocks=free, margin_blocks=free-t-r-g, capacity_fit=free>=t+r+g,
        donor_excluded_inflight_blocks=contribution['inflight_remaining_blocks'],
        donor_excluded_running_blocks=contribution['running_increment_blocks'],
        growth_tokens=0,target_growth_tokens=0)


def _attach(scheduler, manager, single, cs, mode, receipts, baseline, undo_baseline):
    """Hook-only core; production qualification is performed by install below."""
    pool, owned, block_size = manager.block_pool, single.req_to_blocks, single.block_size
    old_begin, old_hold, old_schedule = scheduler._rotation_begin, scheduler._rotation_hold, scheduler.schedule
    data = dict(mode=mode, status='INSTALLED', action_count=0, events=[], skip_counts={},
        baseline_stall8=baseline, source_sha256=PINS, max_protected_rounds=16,
        scope='One single-donor exchange; frozen stall8 remains installed. Its existing protection guard pauses bypasses during this window and retains its original budget/episode history. Q1 ends at a past real client receipt or 16 schedule entries; no wall-time/service guarantee.',
        donor_cost_definition='A _native_release_state ref_cnt==1; funding_victim.host_prefix contiguous ready prefix. Rank release excess, missing materialized whole pages, native-tail tie. No runtime dependency on A files.',
        a_definition_source_sha256={'staged_store_rotation.py':'18f0e3f17e56b8a6b1e8516a9b47f33c2f0ca025e9ab90caa67522b5a56f6784',
            'funding_victim.py':'fa8cb428983f0c331b2975d6325f5d2b0a9067689a55d860102fe48a98117d5d'})
    step, selected, active, consumed, installed = 0, None, False, False, True

    def emit(kind, **fields):
        row = dict(kind=kind, step=step, host_perf_s=time.perf_counter(), **fields)
        data['events'].append(row)
        return row

    def skip(reason):
        data['skip_counts'][reason] = data['skip_counts'].get(reason, 0) + 1

    def free():
        value = pool.get_num_free_blocks()
        if type(value) is not int or value < 0: raise ValueError('UNKNOWN_FREE')
        return value

    def row(request):
        rid = request.request_id
        if scheduler.requests.get(rid) is not request: raise ValueError('REQUEST_IDENTITY')
        blocks = owned.get(rid, ())
        ids = [b.block_id for b in blocks]
        if ids != manager.get_blocks(rid).get_block_ids()[0] or len(ids) != len(set(ids)):
            raise ValueError('BLOCK_MAPPING')
        if any(b.is_null or b.ref_cnt != 1 or pool.blocks[b.block_id] is not b for b in blocks):
            raise ValueError('NONPRIVATE_BLOCKS')
        values = (request.num_tokens, len(blocks), request.num_computed_tokens,
                  request.num_in_flight_tokens)
        if any(type(n) is not int or n < 0 for n in values): raise ValueError('UNKNOWN_COUNTS')
        if request.num_output_placeholders or request.spec_token_ids or request.has_encoder_inputs:
            raise ValueError('UNSUPPORTED_REQUEST')
        return dict(request=rid, history_tokens=values[0], held_gpu_blocks=values[1],
            computed_tokens=values[2], num_in_flight_tokens=values[3], scheduled_tokens_this_step=0)

    def status(request):
        state = cs._req_status.get(request.request_id)
        if state is None or state.req is not request or type(state.transfer_jobs) is not set:
            raise ValueError('UNKNOWN_TRANSFER_IDENTITY')
        for jid in state.transfer_jobs:
            job = cs._jobs.get(jid)
            if job is None or job.req_id != request.request_id: raise ValueError('UNKNOWN_JOB')
        return state

    def pure_decode(request):
        return (request.status.name == 'RUNNING' and request.num_output_tokens > 0
            and request.num_in_flight_tokens == 0 and not request.is_prefill_chunk
            and request.num_tokens_with_spec - request.num_computed_tokens == 1)

    def snapshot(target):
        running = list(scheduler.running)
        inflight = list(scheduler._inflight_prefills)
        by_id = {}
        for request in running + inflight:
            previous = by_id.setdefault(request.request_id, request)
            if previous is not request: raise ValueError('MEMBERSHIP_IDENTITY')
        # Exactly one ownership/alias scan per decision boundary, not per donor.
        distinct_ids = [b.block_id for r in by_id.values()
                        for b in owned.get(r.request_id, ())]
        if len(distinct_ids) != len(set(distinct_ids)): raise ValueError('CROSS_REQUEST_ALIAS')
        rows = [row(request) for request in by_id.values()]
        reserved = scheduler._inflight_prefill_reserved_blocks()
        result = CAPACITY.estimate(target=row(target), requests=rows,
            running_ids=[r.request_id for r in running], inflight_ids=[r.request_id for r in inflight],
            native_reserved_blocks=reserved, free_gpu_blocks=free(),
            block_size=block_size, max_model_len=manager.max_model_len,
            target_growth_tokens=0, growth_tokens=0, qualified=True)
        if result['status'] != 'KNOWN': raise ValueError(result['reason'])
        return result

    def donor_cost(request):
        state = status(request); blocks = owned.get(request.request_id, ())
        for jid in state.transfer_jobs:
            job = cs._jobs[jid]
            if not job.is_store: raise ValueError('DONOR_LOAD_PENDING')
            src = job.non_sliding_window_block_ids
            if job.sliding_window_block_ids or type(src) is not list or not set(src) <= {b.block_id for b in blocks}:
                raise ValueError('DONOR_STORE_SOURCE_UNKNOWN')
        group = state.group_states[0]; count = request.num_computed_tokens // block_size
        if len(state.group_states) != 1 or len(group.offload_keys) < count or list(group.block_ids[:count]) != [b.block_id for b in blocks[:count]]:
            raise ValueError('DONOR_HOST_MAPPING')
        prefix = 0; pending = 0
        for key in group.offload_keys[:count]:
            block = cs.manager._policy.blocks.get(key)
            if block is not None and not block.is_ready: pending += 1
        for key in group.offload_keys[:count]:
            block = cs.manager._policy.blocks.get(key)
            if block is None or block.ref_cnt < 0 or not block.is_ready: break
            prefix += 1
        return dict(immediate_releasable_blocks=len(blocks), host_ready_prefix_blocks=prefix,
            host_missing_materialized_blocks=count-prefix, pending_store_jobs=sorted(state.transfer_jobs),
            host_pending_materialized_blocks=pending,
            computed_tokens=request.num_computed_tokens, output_tokens=request.num_output_tokens,
            held_blocks=len(blocks), last_receipt_host_perf_s=receipts.get(request.request_id))

    def release(reason):
        nonlocal active
        if not active: return
        target = selected['target_ref']
        emit('release', reason=reason, target=target.request_id,
            selected_step=selected['step'], protected_entries=step-selected['step'],
            target_receipt_host_perf_s=receipts.get(target.request_id),
            free_gpu_blocks=free(), target_status=target.status.name,
            timing_scope='gate lifetime includes native waiting/LOAD; not counterfactual extra latency')
        active = False
        scheduler._rotation_target = None

    def begin(preempted, timestamp):
        nonlocal step, selected, active, consumed
        old_begin(preempted, timestamp); step += 1
        now = time.perf_counter()
        if active:
            target = selected['target_ref']
            receipt = receipts.get(target.request_id)
            if scheduler.requests.get(target.request_id) is not target or target.is_finished():
                release('TARGET_FINISHED_OR_IDENTITY_CHANGED'); return
            if type(receipt) in (int,float) and selected['receipt'] < receipt < now:
                release('FIRST_CLIENT_RECEIPT'); return
            if step >= selected['step'] + 16:
                release('SIXTEEN_ROUND_LIMIT'); return
            if (scheduler._rotation_target is not None or scheduler._rotation_lease_enabled
                    or preempted or scheduler._pause_state.name != 'UNPAUSED'
                    or any(r.status.name == 'WAITING' for r in list(scheduler.waiting)+list(scheduler.skipped_waiting))):
                release('NATIVE_OR_NEW_REQUEST_CONFLICT'); return
            if any(r is not target and not pure_decode(r) for r in scheduler.running):
                release('PEER_NO_LONGER_PLAIN_DECODE'); return
            try:
                row(target); status(target)
                for peer in scheduler.running: row(peer)
            except (AttributeError, IndexError, KeyError, TypeError, ValueError) as error:
                release('UNKNOWN_ACTIVE_STATE:'+str(error)); return
            if target.status.name == 'PREEMPTED' and target.num_preemptions != selected['preemptions']:
                release('TARGET_REPREEMPTED'); return
            if target.status.name not in ('PREEMPTED','WAITING_FOR_REMOTE_KVS','RUNNING'):
                release('UNSUPPORTED_TARGET_STATE'); return
            scheduler._rotation_target = target.request_id
            for queue in (scheduler.skipped_waiting, scheduler.waiting):
                if target in queue:
                    queue.remove_request(target); queue.prepend_request(target); break
            emit('protection_entry', target=target.request_id, free_gpu_blocks=free())
            return
        if consumed: return
        if (preempted or scheduler._rotation_target is not None or scheduler._rotation_lease_enabled
                or scheduler._pause_state.name != 'UNPAUSED' or scheduler.skipped_waiting
                or not scheduler.waiting): return skip('NATIVE_OR_PROTECTION_GATE')
        if any(r.status.name == 'WAITING' for r in scheduler.waiting): return skip('NEW_WAITER')
        if not scheduler.running or any(not pure_decode(r) for r in scheduler.running):
            return skip('RUNNING_NOT_PLAIN_DECODE')
        if (len(scheduler.running)-1+ scheduler.num_waiting_for_streaming_input >= scheduler.max_num_running_reqs
                or len(scheduler.running)-1 >= scheduler.max_num_scheduled_tokens):
            return skip('NO_FUNDED_RUNNING_SLOT_OR_TOKEN')
        head = scheduler.waiting.peek_request(); targets = []; references = {}
        try:
            for request in scheduler.waiting:
                if request.status.name != 'PREEMPTED' or request.priority != head.priority: break
                if (request.num_computed_tokens or owned.get(request.request_id)
                        or request.num_in_flight_tokens): continue
                need = (min(request.num_tokens,manager.max_model_len)+block_size-1)//block_size
                if need <= free(): continue
                state = row(request); transfer = status(request)
                if transfer.transfer_jobs:
                    continue
                stamp = receipts.get(request.request_id)
                if type(stamp) not in (int,float) or not math.isfinite(stamp) or not stamp < now:
                    return skip('UNKNOWN_TARGET_RECEIPT')
                native = HELPERS._candidate(request, manager, single, cs,
                    scheduler._inflight_prefill_reserved_blocks(), free())
                if not native['context']['ready']: continue
                targets.append(dict(request=request.request_id, priority=request.priority,
                    arrival_time=request.arrival_time, num_preemptions=request.num_preemptions,
                    last_receipt_host_perf_s=stamp, stall_s=now-stamp,
                    target_current_history_blocks=need, free_gpu_blocks=free(),
                    native_context=native['context']))
                references[request.request_id] = request
        except (AttributeError, IndexError, KeyError, TypeError, ValueError) as error:
            return skip('UNKNOWN_INITIAL_STATE:'+str(error))
        if not targets: return skip('NO_NONFIT_TARGET')
        chosen = min(targets, key=lambda r:(r['last_receipt_host_perf_s'],r['arrival_time'],r['request']))
        target = references[chosen['request']]; donors = []
        try: joint = snapshot(target)
        except (AttributeError, IndexError, KeyError, TypeError, ValueError) as error:
            return skip('UNKNOWN_JOINT_SNAPSHOT:'+str(error))
        contributions = {r['request']:r for r in joint['per_request']}
        for index, request in enumerate(scheduler.running):
            if request.priority != target.priority: continue
            try:
                cost = donor_cost(request)
            except (AttributeError, IndexError, KeyError, TypeError, ValueError): continue
            contribution = contributions[request.request_id]
            capacity = _exclude_donor(joint, contribution, cost['immediate_releasable_blocks'])
            donors.append(dict(request=request.request_id, running_index=index,
                release_excess_blocks=capacity['margin_blocks'], capacity=capacity, **cost))
        sufficient = [r for r in donors if r['capacity']['capacity_fit']]
        if not sufficient: return skip('NO_SUFFICIENT_SINGLE_DONOR')
        donor_row = min(sufficient, key=lambda r:(r['release_excess_blocks'],r['host_missing_materialized_blocks'],-r['running_index']))
        donor = scheduler.running[donor_row['running_index']]
        consumed = True
        selected = dict(target_ref=target, step=step, receipt=chosen['last_receipt_host_perf_s'], preemptions=target.num_preemptions)
        decision = emit('decision', targets=targets, donors=donors, target=target.request_id,
            donor=donor.request_id, selected_capacity=donor_row['capacity'],
            joint_capacity_before_exchange=joint,
            baseline_action='NO_EXCHANGE_FROZEN_STALL8', requested_action=mode=='exchange_once',
            free_gpu_blocks=free(), target_receipt_at_selection=selected['receipt'],
            decision_cpu_s=time.perf_counter()-now)
        if mode == 'stall8': return
        before = free(); pending = donor_row['pending_store_jobs']
        try:
            scheduler.running.remove(donor)
            scheduler._preempt_request(donor, timestamp)
            data['action_count'] = 1
            preempted.append(donor)
            scheduler._rotation_forced_count += 1
            scheduler.waiting.remove_request(target); scheduler.waiting.prepend_request(target)
            scheduler._rotation_target = target.request_id
            active = True
            if free()-before != donor_row['immediate_releasable_blocks']:
                raise RuntimeError('Native donor release differed from private physical model')
            emit('action', target=target.request_id, donor=donor.request_id,
                free_before=before, free_after=free(), released_blocks=free()-before,
                forced_count=scheduler._rotation_forced_count, pending_store_jobs=pending,
                donor_num_preemptions=donor.num_preemptions, native_preempt_called=True)
        except BaseException as error:
            decision['error'] = repr(error); data['status'] = 'ERROR'; raise

    def hold(request):
        if old_hold(request): return True
        if not active or request is selected['target_ref']: return False
        target = selected['target_ref']
        try:
            t = row(target); current = row(request)
            needed = lambda r:max((min(r['history_tokens'],manager.max_model_len)+block_size-1)//block_size-r['held_gpu_blocks'],0)
            reserved = scheduler._inflight_prefill_reserved_blocks()
            excluded = sum(scheduler._request_remaining_blocks(r) for r in scheduler._inflight_prefills if r is target or r is request)
            if type(reserved) is not int or not 0 <= excluded <= reserved: raise ValueError('UNKNOWN_RESERVATION')
            required = needed(t)+needed(current)+reserved-excluded
            blocked = free() < required
        except (AttributeError, IndexError, KeyError, TypeError, ValueError) as error:
            release('UNKNOWN_HOLD_STATE:'+str(error)); return False
        if blocked:
            emit('peer_hold', target=target.request_id, peer=request.request_id,
                free_gpu_blocks=free(), target_remaining_blocks=needed(t),
                peer_current_need_blocks=needed(current), other_inflight_reserved_blocks=reserved-excluded)
        return blocked

    def schedule(*args, **kwargs):
        result = old_schedule(*args, **kwargs)
        if active:
            target = selected['target_ref']
            emit('target_scheduled', target=target.request_id,
                scheduled_tokens=result.num_scheduled_tokens.get(target.request_id,0),
                target_status=target.status.name, target_held_blocks=len(owned.get(target.request_id,())),
                preempted_req_ids=list(result.preempted_req_ids or ()), free_gpu_blocks=free())
        return result

    scheduler._rotation_begin, scheduler._rotation_hold, scheduler.schedule = begin, hold, schedule

    def uninstall():
        nonlocal installed
        if installed:
            if scheduler._rotation_begin is not begin or scheduler._rotation_hold is not hold or scheduler.schedule is not schedule:
                raise RuntimeError('Exchange hook replaced before uninstall')
            release('UNINSTALL')
            scheduler._rotation_begin, scheduler._rotation_hold, scheduler.schedule = old_begin, old_hold, old_schedule
            data['baseline_stall8'] = undo_baseline(); installed = False
            if data['status'] != 'ERROR': data['status'] = 'UNINSTALLED'
        return data
    return data, uninstall


def install(scheduler, mode='stall8', *, last_receipts, selective):
    if mode not in MODES or type(last_receipts) is not dict: raise ValueError('Unsupported exchange mode/receipts')
    if (selective.get('allow_forced_rotations') is not False or selective.get('native_victim_rule') != 'tail'
            or selective.get('recovery_lease_mode') != 'off' or selective.get('ordinary_backfill') is not False
            or scheduler.defer_block_free):
        raise RuntimeError('Requires native-only tail base without another capacity policy')
    if hashlib.sha256((BASE/'pkg/staged_store_rotation.py').read_bytes()).hexdigest() != PINS['pkg/staged_store_rotation.py']:
        raise RuntimeError('Frozen native hooks changed')
    baseline, undo = FROZEN.install(scheduler, 'stall8', last_receipts=last_receipts)
    try:
        manager = scheduler.kv_cache_manager
        single = manager.coordinator.single_type_managers[0]
        cs = scheduler.connector.connector_scheduler
        return _attach(scheduler, manager, single, cs, mode, last_receipts, baseline, undo)
    except BaseException:
        undo(); raise
