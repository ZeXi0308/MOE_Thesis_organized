"""One same-priority recovery backfill at the original native waiting decision."""
import ast
import copy
import hashlib
import importlib.util
import inspect
from pathlib import Path
import time

OBSERVER_SHA = '8f130a900b4b0ac6443ba6991a56d2ae80f7132dea2ad4bb72cb4ec3ed5fc28b'
ROOT = Path(__file__).resolve().parents[1]


def _load(relative, name, expected):
    path = ROOT/relative
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise RuntimeError('Recovery-fit dependency changed: '+relative)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


HELPERS = _load('recovery_queue/observe_waiting.py', 'fit_once_frozen_observer', OBSERVER_SHA)


def _instrument(tree):
    result, matches = copy.deepcopy(tree), 0
    for loop in ast.walk(result):
        if isinstance(loop, ast.While) and ast.unparse(loop.test) == '(self.waiting or self.skipped_waiting) and token_budget > 0':
            for i, node in enumerate(loop.body):
                if isinstance(node, ast.Assign) and ast.unparse(node) == 'request = request_queue.peek_request()':
                    call = ast.parse(f'{HELPERS.CALLBACK}(self, token_budget, preempted_reqs, request_queue)').body[0]
                    loop.body.insert(i, ast.copy_location(call, node)); matches += 1
                    break
    if matches != 1:
        raise RuntimeError('Expected one native waiting peek')
    return ast.fix_missing_locations(result)


def _decision(scheduler, manager, single, cs, mode):
    data = dict(mode=mode, status='INSTALLED', outcome='NO_OPPORTUNITY', events=[],
        callback_calls=0, skip_counts={}, action_count=0, shadow_count=0,
        clock='time.perf_counter host; no CUDA timing/query/sync',
        scope='First capacity-blocked PREEMPTED head only; one same-priority fit candidate in the continuous PREEMPTED prefix; no pin/protection/new-request crossing',
        observer_sha256=OBSERVER_SHA, source_sha256=dict(HELPERS.PINS))

    def skip(reason):
        data['skip_counts'][reason] = data['skip_counts'].get(reason, 0)+1

    def callback(current, budget, preempted, queue):
        if data['events']:
            return
        data['callback_calls'] += 1
        if queue is not current.waiting or current.skipped_waiting:
            return skip('NOT_UNBLOCKED_WAITING_QUEUE')
        running = len(current.running)+current.num_waiting_for_streaming_input
        if (budget <= 0 or preempted or running >= current.max_num_running_reqs
                or current._pause_state.name != 'UNPAUSED'):
            return skip('NATIVE_WAITING_GATE_CLOSED')
        if current._rotation_target is not None or current._rotation_lease_enabled:
            return skip('OTHER_PROTECTION_ACTIVE')
        head = next(iter(queue), None)
        if head is None or head.status.name != 'PREEMPTED':
            return skip('HEAD_NOT_PREEMPTED')
        if head.num_computed_tokens or single.req_to_blocks.get(head.request_id):
            return skip('HEAD_HAS_COMPUTED_OR_HELD_KV')
        free = manager.block_pool.free_block_queue.num_free_blocks
        need = (min(head.num_tokens, manager.max_model_len)+single.block_size-1)//single.block_size
        if need <= free:
            return skip('HEAD_FULL_HISTORY_FITS')
        reserved = current._inflight_prefill_reserved_blocks()
        if type(reserved) is not int or reserved < 0:
            return skip('UNKNOWN_NATIVE_RESERVATION')
        started = time.perf_counter()
        prefix = []
        for request in queue:
            if request.status.name != 'PREEMPTED' or request.priority != head.priority:
                break
            prefix.append(request)
        candidates, eligible = [], []
        for request in prefix:
            row = HELPERS._candidate(request, manager, single, cs, reserved, free)
            row['priority_match'] = request.priority == head.priority
            row['computed_and_held_zero'] = not request.num_computed_tokens and not row['held_gpu_blocks']
            row['eligible'] = bool(request is not head and row['priority_match']
                                   and row['computed_and_held_zero'] and row['conservative_fit'])
            if not row['priority_match']: row['reasons'].append('DIFFERENT_PRIORITY')
            if not row['computed_and_held_zero']: row['reasons'].append('COMPUTED_OR_HELD_KV')
            candidates.append(row)
            if row['eligible']:
                eligible.append(request)
        if not eligible:
            return skip('NO_FIT_RECOVERY_IN_CONTINUOUS_PREFIX')
        chosen = min(eligible, key=lambda r: (r.arrival_time, r.request_id))
        before = [r.request_id for r in queue]
        event = dict(kind='first_fit_opportunity', host_perf_s=started,
            callback_index=data['callback_calls'], baseline_head=head.request_id,
            candidate_head=chosen.request_id, final_head=head.request_id,
            waiting_before=before, waiting_after=before[:], skipped_order=[],
            prefix_order=[r.request_id for r in prefix], candidates=candidates,
            token_budget=budget, running_count=len(current.running), running_with_streaming=running,
            running_cap=current.max_num_running_reqs, pause=current._pause_state.name,
            preempted_ids=[r.request_id for r in preempted], free_gpu_blocks=free,
            native_inflight_reserved_blocks=reserved, head_full_fit_extra_blocks=need,
            reason='HEAD_FULL_HISTORY_SHORTFALL_AND_SAME_PRIORITY_RECOVERY_FITS',
            queue_changed=False, action='SHADOW_ONLY' if mode == 'native' else 'REORDER_PENDING')
        data['events'].append(event)  # At most one decision, including exceptional mutation.
        data['shadow_count'] = 1
        if mode == 'native':
            data['outcome'] = 'SHADOW_ONLY'
            event['return_host_perf_s'] = time.perf_counter()
            return
        event['action_begin_host_perf_s'] = time.perf_counter()
        try:
            queue.remove_request(chosen)
            queue.prepend_request(chosen)
            event.update(waiting_after=[r.request_id for r in queue], final_head=next(iter(queue)).request_id,
                         queue_changed=True, action='REORDERED')
            data.update(outcome='REORDERED', action_count=1)
        except BaseException as error:
            event.update(waiting_after=[r.request_id for r in queue], action='ERROR', error=repr(error))
            data.update(status='ERROR', outcome='ERROR')
            raise
        finally:
            event['return_host_perf_s'] = time.perf_counter()
    return data, callback


def _attach(scheduler, manager, single, cs, mode, tree):
    if mode not in ('native', 'fit_once'):
        raise ValueError(mode)
    slot = HELPERS._native_slot(scheduler.schedule, scheduler)
    old = slot.cell_contents
    rebuilt = HELPERS._compile(tree, old)
    if not HELPERS._same_code(rebuilt.__func__.__code__, old.__func__.__code__):
        raise RuntimeError('Native schedule differs from pinned reconstruction')
    data, callback = _decision(scheduler, manager, single, cs, mode)
    new = HELPERS._compile(_instrument(tree), old, callback)
    slot.cell_contents = new
    data['native_code_verified'] = True
    active = True
    def uninstall():
        nonlocal active
        if active:
            if slot.cell_contents is not new:
                raise RuntimeError('Native schedule slot changed during fit_once')
            slot.cell_contents = old; active = False
            if data['status'] == 'INSTALLED': data['status'] = 'UNINSTALLED'
        return data
    return data, uninstall


def install(scheduler, mode='native'):
    qualifier = _load('tail_reservation/reserve_tail.py', 'fit_once_qualifier', HELPERS.PINS['tail_reservation/reserve_tail.py'])
    source_guard = _load('source_handoff/source_handoff.py', 'fit_once_host_pins', HELPERS.PINS['source_handoff/source_handoff.py'])
    manager, single, reason = qualifier.qualify(scheduler)
    cs = scheduler.connector.connector_scheduler
    for obj in (cs, cs.manager, cs.manager._policy):
        source = inspect.getsourcefile(type(obj))
        if (source is None or source_guard.PINS.get(type(obj).__name__)
                != hashlib.sha256(Path(source).read_bytes()).hexdigest()):
            raise RuntimeError('Native Host source changed')
    groups = cs.config.kv_group_configs
    if (reason or scheduler.policy.name != 'FCFS' or scheduler.scheduler_config.async_scheduling
            or not scheduler.scheduler_reserve_full_isl or scheduler._rotation_lease_enabled
            or scheduler._rotation_target is not None or scheduler._rotation_forced_count
            or scheduler.max_num_scheduled_tokens != 1024 or len(groups) != 1
            or groups[0].tokens_per_chunk != single.block_size
            or groups[0].sliding_window_size_in_chunks is not None or groups[0].is_eagle_group
            or cs.config.blocks_per_chunk != 1):
        raise RuntimeError('Requires qualified native single-group FCFS without protection')
    import rotation_native
    if hashlib.sha256(Path(rotation_native.__file__).read_bytes()).hexdigest() != HELPERS.PINS['pkg/rotation_native.py']:
        raise RuntimeError('Native patch builder changed')
    source = Path(inspect.getsourcefile(type(scheduler))).read_text()
    return _attach(scheduler, manager, single, cs, mode, rotation_native.patched_schedule_tree(source))
