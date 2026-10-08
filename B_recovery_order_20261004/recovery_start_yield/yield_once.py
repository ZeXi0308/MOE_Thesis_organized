"""One native schedule-round yield before starting a later recovery recompute."""
import ast
import copy
import hashlib
import importlib.util
import inspect
from pathlib import Path
import time

BASE = Path(__file__).resolve().parents[1]
FIT_SHA = '7d2965b21f60516879b3a14a9a92533e882c9193890f4cf3cf3f77004bf8cbd6'
path = BASE/'recovery_fit/fit_once.py'
if hashlib.sha256(path.read_bytes()).hexdigest() != FIT_SHA:
    raise RuntimeError('Frozen native interface helpers changed')
spec = importlib.util.spec_from_file_location('yield_once_fit_helpers', path)
FIT = importlib.util.module_from_spec(spec); spec.loader.exec_module(FIT)
H = FIT.HELPERS


def _instrument(tree):
    result = copy.deepcopy(tree)
    fn = next(n for n in result.body if isinstance(n, ast.FunctionDef) and n.name == 'schedule')
    matches = [0, 0]
    for i, node in enumerate(fn.body):
        if isinstance(node, ast.If) and ast.unparse(node.test) == 'len(preempted_reqs) == self._rotation_forced_count and self._pause_state == PauseState.UNPAUSED':
            call = ast.parse(f'{H.CALLBACK}(self, "entry", token_budget, preempted_reqs)').body[0]
            fn.body.insert(i, ast.copy_location(call, node)); matches[0] += 1
            break
    arguments = ('self, PHASE, token_budget, preempted_reqs, request_queue, request, '
        'step_skipped_waiting, num_new_tokens, num_external_computed_tokens, load_kv_async, '
        'num_new_local_computed_tokens, effective_lookahead_tokens, num_encoder_tokens')
    for loop in ast.walk(fn):
        if isinstance(loop, ast.While) and ast.unparse(loop.test) == '(self.waiting or self.skipped_waiting) and token_budget > 0':
            for i, node in enumerate(loop.body):
                if (isinstance(node, ast.Assign) and len(node.targets) == 1
                        and ast.unparse(node.targets[0]) == 'new_blocks'
                        and isinstance(node.value, ast.Call)
                        and ast.unparse(node.value.func) == 'self.kv_cache_manager.allocate_slots'):
                    guard = ast.parse(f'if {H.CALLBACK}({arguments.replace("PHASE", repr("decision"))}):\n'
                        f'    {H.CALLBACK}({arguments.replace("PHASE", repr("break_executed"))})\n'
                        '    break').body[0]
                    loop.body.insert(i, ast.copy_location(guard, node)); matches[1] += 1
                    break
    if matches != [1, 1]:
        raise RuntimeError('Expected one native waiting allocation boundary')
    return ast.fix_missing_locations(result)


def _beneficiaries(current, request, local, single, cs):
    rows, seen = [], {}
    for location, queue in [('local_skipped', local), ('public_skipped', current.skipped_waiting)]:
        for earlier in queue:
            if (earlier is request or earlier.status.name != 'WAITING_FOR_REMOTE_KVS'
                    or earlier.num_preemptions < 1 or earlier.priority != request.priority):
                continue
            rid = earlier.request_id
            if rid in seen:
                if seen[rid][0] is not earlier:
                    return [], 'DUPLICATE_REQUEST_IDENTITY'
                seen[rid][1]['locations'].append(location)
                continue
            state = cs._req_status.get(rid)
            if state is None or state.req is not earlier or not state.transfer_jobs:
                continue
            jobs = sorted(state.transfer_jobs)
            if any(jid not in cs._jobs or cs._jobs[jid].req_id != rid
                   or cs._jobs[jid].is_store is not False or cs._jobs[jid].pending_count <= 0
                   for jid in jobs):
                continue
            batch = [jid for jid in jobs if jid in cs._current_batch_load_jobs]
            if any(cs._current_batch_load_jobs[jid].req_id != rid for jid in batch):
                return [], 'CURRENT_BATCH_JOB_IDENTITY_MISMATCH'
            row = dict(request=rid, priority=earlier.priority, arrival_time=earlier.arrival_time,
                num_preemptions=earlier.num_preemptions,
                computed_tokens=earlier.num_computed_tokens,
                held_gpu_blocks=len(single.req_to_blocks.get(rid, ())),
                locations=[location], job_ids=jobs, current_batch_job_ids=batch,
                status='WAITING_FOR_REMOTE_KVS', dependency='Previously processed native async restore with its own registered LOAD; not guaranteed earlier arrival or a hard dependency of the deferred request')
            rows.append(row); seen[rid] = (earlier, row)
    return rows, None


def _decision(scheduler, manager, single, cs, mode, selective, now=time.perf_counter):
    if mode not in ('native', 'yield_once'):
        raise ValueError(mode)
    data = dict(mode=mode, status='INSTALLED', outcome='NO_OPPORTUNITY', action_count=0,
        executed_breaks=0, decision_calls=0, skip_counts={}, events=[],
        clock='time.perf_counter host observations; no CUDA query or synchronization',
        scope='Once-only one native schedule round. No cross-round gate or timeout; later schedule decisions are native. A break also defers the remaining waiting suffix.',
        dependency_semantics='Beneficiaries await their own earlier registered LOADs. Inter-request waiting is scheduling precedence, not a native hard dependency; a fixed one-round yield may be equivalent.',
        timing_semantics='Observed decision/next-entry intervals are not counterfactual added or saved latency.',
        frozen_helper_sha256=FIT_SHA, source_sha256=dict(H.PINS))
    step, chosen, next_entry_pending = 0, None, False

    def skip(reason):
        data['skip_counts'][reason] = data['skip_counts'].get(reason, 0)+1
        return False

    def callback(current, phase, budget, preempted, queue=None, request=None, local=None,
                 new_tokens=None, external=None, asynchronous=None, local_tokens=None,
                 lookahead=None, encoder_tokens=None):
        nonlocal step, chosen, next_entry_pending
        if phase == 'entry':
            step += 1
            if next_entry_pending:
                data['events'].append(dict(kind='next_schedule_entry_pass', step=step,
                    host_perf_s=now(), decision_step=chosen['step'],
                    deferred_request=chosen['deferred_request'], previous_executed_breaks=chosen['executed_breaks']))
                next_entry_pending = False
            return False
        if phase == 'break_executed':
            if (mode != 'yield_once' or chosen is None or chosen['executed_breaks']
                    or request.request_id != chosen['deferred_request'] or step != chosen['step']):
                raise RuntimeError('Unqualified or repeated waiting-loop break')
            chosen.update(executed_breaks=1, first_break_host_perf_s=now(), final_action='WAITING_LOOP_BREAK')
            data.update(action_count=1, executed_breaks=1, outcome='YIELDED_ONE_ROUND')
            return False
        if phase != 'decision':
            raise ValueError(phase)
        data['decision_calls'] += 1
        if chosen is not None:
            return False
        if queue is not current.waiting or not current.waiting or current.waiting.peek_request() is not request:
            return skip('NOT_NATIVE_WAITING_HEAD')
        if (budget <= 0 or preempted or current._pause_state.name != 'UNPAUSED'
                or len(current.running)+current.num_waiting_for_streaming_input >= current.max_num_running_reqs):
            return skip('NATIVE_WAITING_GATE_CLOSED')
        if current._rotation_target is not None or current._rotation_lease_enabled or current._rotation_forced_count:
            return skip('BASELINE_PROTECTION')
        if any(r.status.name == 'WAITING' for q in (current.waiting, current.skipped_waiting, local) for r in q):
            return skip('NEW_WAITER_PRESENT')
        if (request.status.name != 'PREEMPTED' or request.num_preemptions < 1
                or request.num_computed_tokens or single.req_to_blocks.get(request.request_id)):
            return skip('HEAD_NOT_CLEAN_RECOVERY')
        if (asynchronous is not False or external != 0 or local_tokens != 0
                or lookahead != 0 or encoder_tokens != 0 or type(new_tokens) is not int
                or new_tokens <= 0 or new_tokens > budget):
            return skip('NOT_NATIVE_LOCAL_RECOMPUTE')
        state = cs._req_status.get(request.request_id)
        if state is None or state.req is not request or state.transfer_jobs:
            return skip('CURRENT_REQUEST_TRANSFER_OR_IDENTITY')
        free = manager.block_pool.free_block_queue.num_free_blocks
        reserved = current._inflight_prefill_reserved_blocks()
        if type(reserved) is not int or reserved < 0:
            return skip('UNKNOWN_NATIVE_RESERVATION')
        candidate = H._candidate(request, manager, single, cs, reserved, free)
        if not candidate['conservative_fit']:
            return skip('HEAD_NOT_CONSERVATIVE_FIT')
        beneficiaries, error = _beneficiaries(current, request, local, single, cs)
        if error or not beneficiaries:
            return skip(error or 'NO_EARLIER_REGISTERED_RECOVERY_LOAD')
        chosen = dict(kind='legal_start_yield_opportunity', step=step, host_perf_s=now(),
            deferred_request=request.request_id, beneficiaries=beneficiaries, candidate=candidate,
            native_locals=dict(num_new_tokens=new_tokens, num_external_computed_tokens=external,
                load_kv_async=asynchronous, num_new_local_computed_tokens=local_tokens,
                effective_lookahead_tokens=lookahead, num_encoder_tokens=encoder_tokens),
            token_budget=budget, free_gpu_blocks=free, native_reserved_blocks=reserved,
            running_count=len(current.running), running_cap=current.max_num_running_reqs,
            running_with_streaming=len(current.running)+current.num_waiting_for_streaming_input,
            pause=current._pause_state.name, preempted_ids=[r.request_id for r in preempted],
            waiting_order=[r.request_id for r in current.waiting],
            public_skipped_order=[r.request_id for r in current.skipped_waiting],
            local_skipped_order=[r.request_id for r in local],
            baseline_action='CONTINUE_NATIVE_RECOMPUTE_ALLOCATION',
            final_action='SHADOW_ONLY' if mode == 'native' else 'BREAK_REQUESTED',
            requested_breaks=int(mode == 'yield_once'), executed_breaks=0)
        data['events'].append(chosen); next_entry_pending = True
        data['outcome'] = 'SHADOW_ONLY' if mode == 'native' else 'BREAK_REQUESTED'
        return mode == 'yield_once'
    return data, callback


def _attach(scheduler, manager, single, cs, mode, selective, tree, now=time.perf_counter):
    slot = H._native_slot(scheduler.schedule, scheduler); old = slot.cell_contents
    rebuilt = H._compile(tree, old)
    if not H._same_code(rebuilt.__func__.__code__, old.__func__.__code__):
        raise RuntimeError('Native schedule differs from frozen reconstruction or contains another policy')
    data, callback = _decision(scheduler, manager, single, cs, mode, selective, now)
    new = H._compile(_instrument(tree), old, callback)
    slot.cell_contents = new; data['native_code_verified'] = True
    attached = True
    def uninstall():
        nonlocal attached
        if attached:
            if slot.cell_contents is not new:
                raise RuntimeError('Native schedule slot changed during one-round yield')
            slot.cell_contents = old; attached = False; data['status'] = 'UNINSTALLED'
        return data
    return data, uninstall


def install(scheduler, mode='native', *, selective):
    # Reuse source/runtime qualification, immediately restoring its temporary
    # callback. No fit/defer/repeat policy remains in the measured call chain.
    unused, undo = FIT.install(scheduler, 'native'); undo()
    qualifier = FIT._load('tail_reservation/reserve_tail.py', 'start_yield_qualifier', H.PINS['tail_reservation/reserve_tail.py'])
    manager, single, reason = qualifier.qualify(scheduler)
    if (reason or selective.get('allow_forced_rotations') is not False
            or selective.get('native_victim_rule') != 'tail'
            or selective.get('capacity_deferral_mode') != 'off'):
        raise RuntimeError('Requires unchanged native victim/quantum/ownership baseline')
    import rotation_native
    source = Path(inspect.getsourcefile(type(scheduler))).read_text()
    return _attach(scheduler, manager, single, scheduler.connector.connector_scheduler,
        mode, selective, rotation_native.patched_schedule_tree(source))
