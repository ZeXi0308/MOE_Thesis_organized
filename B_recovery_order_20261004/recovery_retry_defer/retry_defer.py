"""One bounded waiting-head retry deferral; native ownership and transfers unchanged."""
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
spec = importlib.util.spec_from_file_location('retry_fit_helpers', path)
FIT = importlib.util.module_from_spec(spec); spec.loader.exec_module(FIT)
H = FIT.HELPERS
DEFER_SECONDS = 0.5


def _instrument(tree):
    result = copy.deepcopy(tree)
    fn = next(n for n in result.body if isinstance(n, ast.FunctionDef) and n.name == 'schedule')
    matches = [0, 0]
    for i, node in enumerate(fn.body):
        if isinstance(node, ast.If) and ast.unparse(node.test) == 'len(preempted_reqs) == self._rotation_forced_count and self._pause_state == PauseState.UNPAUSED':
            expr = ast.parse(f'{H.CALLBACK}(self, "entry", token_budget, preempted_reqs, None, None)').body[0]
            fn.body.insert(i, ast.copy_location(expr, node)); matches[0] += 1
            break
    for loop in ast.walk(fn):
        if isinstance(loop, ast.While) and ast.unparse(loop.test) == '(self.waiting or self.skipped_waiting) and token_budget > 0':
            for i, node in enumerate(loop.body):
                if isinstance(node, ast.Assign) and ast.unparse(node) == 'request = request_queue.peek_request()':
                    guard = ast.parse(f'if {H.CALLBACK}(self, "decision", token_budget, preempted_reqs, request_queue, request):\n'
                        f'    {H.CALLBACK}(self, "break_executed", token_budget, preempted_reqs, request_queue, request)\n'
                        '    break').body[0]
                    loop.body.insert(i + 1, ast.copy_location(guard, node)); matches[1] += 1
                    break
    if matches != [1, 1]:
        raise RuntimeError('Native waiting decision boundary changed')
    return ast.fix_missing_locations(result)


def _decision(scheduler, manager, single, cs, mode, selective, now=time.perf_counter):
    if mode not in ('native', 'defer_once'):
        raise ValueError(mode)
    data = dict(mode=mode, status='INSTALLED', outcome='NO_OPPORTUNITY', events=[],
        action_count=0, executed_breaks=0, decision_calls=0, skip_counts={},
        clock='time.perf_counter host observations; no GPU clock/query/sync',
        max_extra_gate_s=DEFER_SECONDS,
        scope='One legal native PREEMPTED head retry after prior resumed output; queue-wide break keeps order. May delay later waiting requests. Deadline removes only this extra gate.',
        frozen_helper_sha256=FIT_SHA)
    victim_index, step, active, consumed = 0, 0, None, False
    latest = {}

    def skip(reason):
        data['skip_counts'][reason] = data['skip_counts'].get(reason, 0)+1
        return False

    def refresh_victims():
        nonlocal victim_index
        decisions = selective['victim_decisions']
        for event in decisions[victim_index:]:
            rows = [r for r in event['candidates'] if r['request'] == event['selected']]
            if len(rows) != 1:
                continue
            row = rows[0]
            before = row.get('bidkv', {}).get('num_preemptions')
            if type(before) is int:
                latest[event['selected']] = dict(victim_step=event['step'],
                    victim_host_perf_s=event['host_perf_counter_s'],
                    failed_request=event['failed_request'], selected=event['selected'],
                    native_tail=event['native_tail'], changed=event['changed'],
                    preemptions_before=before, qualified=row.get('qualified', False),
                    residency_outputs=row.get('residency_outputs'))
        victim_index = len(decisions)

    def release(reason, observed, **extra):
        nonlocal active
        gate = active
        gate['event'].update(release_reason=reason, release_observed_perf_s=observed,
            extra_gate_observed_s=observed-gate['start'],
            deadline_observation_lateness_s=max(0., observed-gate['deadline']), **extra)
        data['events'].append(dict(kind='release', step=step, host_perf_s=observed,
            target=gate['target'].request_id, reason=reason, **extra))
        data['outcome'] = 'DEFER_RELEASED' if reason in ('COHORT_FINISHED_AND_FREED', 'DEADLINE', 'RUNNING_EMPTY') else 'DEFER_CANCELLED'
        active = None

    def check_release(current, observed):
        if active is None:
            return
        target = active['target']
        queues = list(current.waiting)+list(current.skipped_waiting)
        if current._rotation_target is not None or current._rotation_lease_enabled or current._rotation_forced_count:
            return release('BASELINE_OVERRIDE', observed)
        if any(r.status.name == 'WAITING' for r in queues):
            return release('NEW_WAITER', observed)
        if target.status.name != 'PREEMPTED' or not any(r is target for r in queues):
            return release('TARGET_LEFT_PREEMPTED_QUEUE', observed)
        status = cs._req_status.get(target.request_id)
        if status is None or status.transfer_jobs or single.req_to_blocks.get(target.request_id):
            return release('TARGET_NATIVE_STATE_CHANGED', observed)
        finished = [dict(request=r.request_id, status=r.status.name)
                    for r in active['cohort'] if r.is_finished() and not single.req_to_blocks.get(r.request_id)]
        if finished:
            return release('COHORT_FINISHED_AND_FREED', observed, completed_released=finished)
        if not current.running:
            return release('RUNNING_EMPTY', observed)
        if observed >= active['deadline']:
            return release('DEADLINE', observed)

    def callback(current, phase, budget, preempted, queue, request):
        nonlocal step, active, consumed
        if phase == 'entry':
            step += 1
            if not consumed:
                refresh_victims()
            if active is not None:
                check_release(current, now())
            return False
        if phase == 'break_executed':
            if active is None or request is not active['target']:
                raise RuntimeError('Retry break without selected active target')
            observed = now()
            event = active['event']
            if not event['executed_breaks']:
                data['action_count'] += 1
                event['first_break_host_perf_s'] = observed
            event['executed_breaks'] += 1
            event['last_break_host_perf_s'] = observed
            event['final_action'] = 'WAITING_LOOP_BREAK'
            data['executed_breaks'] += 1
            data['outcome'] = 'DEFER_ACTIVE'
            return False
        if phase != 'decision':
            raise ValueError(phase)
        data['decision_calls'] += 1
        if active is not None:
            check_release(current, now())
            if active is None:
                return False
            if queue is not current.waiting or request is not active['target'] or current.skipped_waiting:
                release('NATIVE_HEAD_OR_QUEUE_CHANGED', now())
                return False
            active['event']['requested_breaks'] += 1
            return True
        if consumed:
            return False
        if queue is not current.waiting or current.skipped_waiting:
            return skip('NOT_UNBLOCKED_WAITING_QUEUE')
        running = len(current.running)+current.num_waiting_for_streaming_input
        if budget <= 0 or preempted or running >= current.max_num_running_reqs or current._pause_state.name != 'UNPAUSED':
            return skip('NATIVE_WAITING_GATE_CLOSED')
        if current._rotation_target is not None or current._rotation_lease_enabled or current._rotation_forced_count:
            return skip('BASELINE_PROTECTION')
        if any(r.status.name == 'WAITING' for r in current.waiting):
            return skip('NEW_WAITER_PRESENT')
        if request.status.name != 'PREEMPTED' or request.num_computed_tokens or single.req_to_blocks.get(request.request_id):
            return skip('HEAD_NOT_CLEAN_PREEMPTED')
        prior = latest.get(request.request_id)
        if (not prior or not prior['qualified'] or prior['preemptions_before'] < 1
                or prior['preemptions_before']+1 != request.num_preemptions
                or not isinstance(prior['residency_outputs'], int) or prior['residency_outputs'] <= 0):
            return skip('NO_PREVIOUS_RESUMED_OUTPUT_PREEMPTION')
        if not current.running or any(not callable(getattr(r, 'is_finished', None)) or r.is_finished() for r in current.running):
            return skip('NO_LIVE_RUNNING_COHORT')
        free = manager.block_pool.free_block_queue.num_free_blocks
        reserved = current._inflight_prefill_reserved_blocks()
        if type(reserved) is not int or reserved < 0:
            return skip('UNKNOWN_NATIVE_RESERVATION')
        candidate = H._candidate(request, manager, single, cs, reserved, free)
        if not candidate['conservative_fit']:
            return skip('HEAD_NOT_CONSERVATIVE_FIT')
        observed = now()
        event = dict(kind='legal_retry_opportunity', step=step, host_perf_s=observed,
            target=request.request_id, num_preemptions=request.num_preemptions,
            previous_preemption=prior, candidate=candidate,
            baseline_action='CONTINUE_NATIVE_LOOKUP_AND_ALLOCATION', candidate_action='BOUNDED_WAITING_LOOP_BREAK',
            final_action='SHADOW_ONLY' if mode == 'native' else 'BREAK_REQUESTED',
            waiting_order=[r.request_id for r in current.waiting], skipped_order=[],
            running_cohort=[r.request_id for r in current.running],
            token_budget=budget, free_gpu_blocks=free, native_reserved_blocks=reserved,
            requested_breaks=0 if mode == 'native' else 1, executed_breaks=0,
            deadline_perf_s=observed+DEFER_SECONDS,
            release_reason=None, release_observed_perf_s=None)
        data['events'].append(event); consumed = True
        if mode == 'native':
            data['outcome'] = 'SHADOW_ONLY'
            return False
        active = dict(target=request, cohort=list(current.running), start=observed,
            deadline=observed+DEFER_SECONDS, event=event)
        return True

    def finish():
        if active is not None:
            release('UNINSTALL_WITH_ACTIVE_GATE', now())
        if data['status'] == 'INSTALLED':
            data['status'] = 'UNINSTALLED'
        return data
    return data, callback, finish


def _attach(scheduler, manager, single, cs, mode, selective, tree, now=time.perf_counter):
    slot = H._native_slot(scheduler.schedule, scheduler)
    old = slot.cell_contents
    rebuilt = H._compile(tree, old)
    if not H._same_code(rebuilt.__func__.__code__, old.__func__.__code__):
        raise RuntimeError('Native schedule differs from frozen reconstruction')
    data, callback, finish = _decision(scheduler, manager, single, cs, mode, selective, now)
    new = H._compile(_instrument(tree), old, callback)
    slot.cell_contents = new
    data['native_code_verified'] = True
    attached = True
    def uninstall():
        nonlocal attached
        if attached:
            if slot.cell_contents is not new:
                raise RuntimeError('Native schedule slot changed during retry probe')
            slot.cell_contents = old; attached = False
            finish()
        return data
    return data, uninstall


def install(scheduler, mode='native', *, selective):
    # Reuse the complete existing qualification without leaving its callback installed.
    unused, undo = FIT.install(scheduler, 'native')
    undo()
    qualifier = FIT._load('tail_reservation/reserve_tail.py', 'retry_qualifier', H.PINS['tail_reservation/reserve_tail.py'])
    manager, single, reason = qualifier.qualify(scheduler)
    if reason or not isinstance(selective.get('victim_decisions'), list):
        raise RuntimeError('Missing native qualification/victim observations')
    import rotation_native
    source = Path(inspect.getsourcefile(type(scheduler))).read_text()
    return _attach(scheduler, manager, single, scheduler.connector.connector_scheduler,
        mode, selective, rotation_native.patched_schedule_tree(source))
