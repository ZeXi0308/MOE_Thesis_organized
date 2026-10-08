"""Read-only waiting decisions for one existing source_handoff(native) anchor."""
import ast
import copy
import hashlib
import importlib.util
import inspect
from pathlib import Path
import time
from types import CodeType, FunctionType, MethodType

PINS = {
    'pkg/rotation_native.py': '822b22d283ecd2bc0d1ab4782fe3afa3e8d7266e61a6021b73f475fc9c468535',
    'tail_reservation/reserve_tail.py': 'fd663e48fbe4291878c16319ffb6294ef63e4444e79f7f84789e13d2aad0ef76',
    'source_handoff/source_handoff.py': '0c952554e87b83c2cd31b9875bdfcb6469a73ce0eefcd089a2f5f28af7d43da7',
}
CALLBACK = '_b_waiting_observation'


def _same_code(a, b):
    fields = ('co_code', 'co_names', 'co_varnames', 'co_freevars', 'co_cellvars',
              'co_argcount', 'co_posonlyargcount', 'co_kwonlyargcount', 'co_flags',
              'co_exceptiontable', 'co_linetable', 'co_firstlineno', 'co_filename')
    return (all(getattr(a, k) == getattr(b, k) for k in fields)
            and len(a.co_consts) == len(b.co_consts)
            and all(_same_code(x, y) if isinstance(x, CodeType) and isinstance(y, CodeType)
                    else type(x) is type(y) and x == y
                    for x, y in zip(a.co_consts, b.co_consts)))


def _native_slot(wrapped, scheduler):
    """Walk only function closures; do not traverse scheduler/model objects."""
    seen, found = set(), []
    def visit(fn, depth=0):
        if isinstance(fn, MethodType):
            fn = fn.__func__
        if not isinstance(fn, FunctionType) or id(fn) in seen or depth > 12:
            return
        seen.add(id(fn))
        for name, cell in zip(fn.__code__.co_freevars, fn.__closure__ or ()):
            value = cell.cell_contents
            if (name == 'native' and isinstance(value, MethodType)
                    and value.__self__ is scheduler and value.__func__.__name__ == 'schedule'):
                found.append(cell)
            elif isinstance(value, (FunctionType, MethodType)):
                visit(value, depth + 1)
    visit(wrapped)
    if len(found) != 1:
        raise RuntimeError('Expected exactly one existing native schedule closure slot')
    return found[0]


def _instrument(tree):
    """Insert two expressions; every original AST node remains untouched."""
    result = copy.deepcopy(tree)
    fn = next(n for n in result.body if isinstance(n, ast.FunctionDef) and n.name == 'schedule')
    matches = [0, 0]
    def call(phase, node):
        suffix = ', request_queue, request' if phase == 'queue_decision' else ', None, None'
        expr = ast.parse(f'{CALLBACK}(self, {phase!r}, token_budget, preempted_reqs{suffix})').body[0]
        return ast.copy_location(expr, node)
    for i, node in enumerate(fn.body):
        if (isinstance(node, ast.If) and ast.unparse(node.test)
                == 'len(preempted_reqs) == self._rotation_forced_count and self._pause_state == PauseState.UNPAUSED'):
            fn.body.insert(i, call('waiting_entry', node)); matches[0] += 1
            break
    for node in ast.walk(fn):
        if isinstance(node, ast.While) and ast.unparse(node.test) == '(self.waiting or self.skipped_waiting) and token_budget > 0':
            for i, item in enumerate(node.body):
                if isinstance(item, ast.Assign) and ast.unparse(item) == 'request = request_queue.peek_request()':
                    node.body.insert(i + 1, call('queue_decision', item)); matches[1] += 1
                    break
    if matches != [1, 1]:
        raise RuntimeError('Waiting decision AST changed')
    return ast.fix_missing_locations(result)


def _compile(tree, native, callback=None):
    namespace = dict(native.__func__.__globals__)
    if CALLBACK in namespace:
        raise RuntimeError('Observation global already exists')
    if callback is not None:
        namespace[CALLBACK] = callback
    exec(compile(tree, native.__func__.__code__.co_filename, 'exec', dont_inherit=True), namespace)
    return MethodType(namespace['schedule'], native.__self__)


def _candidate(request, manager, single, cs, reserved, free):
    rid, size = request.request_id, single.block_size
    held = single.req_to_blocks.get(rid, ())
    plain = all(not b.is_null and b.ref_cnt == 1 for b in held)
    history = request.num_tokens
    need = max(0, (min(history, manager.max_model_len) + size - 1)//size - len(held))
    status = cs._req_status.get(rid)
    context = dict(source='existing offload_keys and LRU.blocks only; no lookup/touch/key generation',
                   known=False, ready=False, state='UNKNOWN', registered_jobs=None,
                   global_load_state_known=False, global_load_conflict_key_indices=[],
                   global_load_conflict_keys=[], skip_reading_prefix_cache=getattr(request, 'skip_reading_prefix_cache', None))
    if status is not None and status.req is request and len(status.group_states) == 1:
        keys = status.group_states[0].offload_keys
        count = history // size
        blocks = [cs.manager._policy.blocks.get(k) for k in keys[:count]]
        pending = sum(b is not None and not b.is_ready for b in blocks)
        leading = 0
        for b in blocks:
            if b is None or not b.is_ready:
                break
            leading += 1
        known = len(keys) >= count
        loading = getattr(cs, '_chunks_being_loaded', ...)
        load_state_known = loading is None or type(loading) is set
        conflict = [i for i, key in enumerate(keys[:leading]) if key in loading] if type(loading) is set else []
        prefix_read_enabled = context['skip_reading_prefix_cache'] is False
        context.update(known=known, ready=known and not status.transfer_jobs and not pending
            and load_state_known and not conflict and prefix_read_enabled,
            state='UNSUPPORTED_PREFIX_READ' if not prefix_read_enabled else 'UNKNOWN' if not known
            else 'PENDING' if pending or conflict else 'HIT_PREFIX' if leading else 'MISS',
            global_load_state_known=load_state_known, global_load_tracking_disabled=loading is None,
            global_load_conflict_key_indices=conflict,
            global_load_conflict_keys=[repr(keys[i]) for i in conflict],
            registered_jobs=sorted(status.transfer_jobs), inspected_keys=len(blocks),
            missing_keys=sum(b is None for b in blocks), pending_keys=pending,
            leading_ready_tokens=leading*size,
            note='Leading prefix is a read-only observation, not an extra native lookup result')
    # This upper bound fits either recompute or any possible async matched prefix.
    # It intentionally rejects some legal states rather than claiming unknown fits.
    upper = need + reserved
    valid = (plain and not request.num_in_flight_tokens and not request.has_encoder_inputs
             and 0 <= request.num_computed_tokens <= history <= manager.max_model_len)
    reasons = []
    if not valid: reasons.append('UNSUPPORTED_REQUEST_OR_OWNERSHIP')
    if free < need: reasons.append('FULL_HISTORY_MEMORY_SHORTFALL')
    if free < upper: reasons.append('CONSERVATIVE_NATIVE_RESERVATION_SHORTFALL')
    if not context['known']: reasons.append('UNKNOWN_HOST_KEYS')
    if context.get('pending_keys'): reasons.append('HOST_PREFIX_PENDING')
    if context['registered_jobs']: reasons.append('REQUEST_TRANSFER_PENDING')
    if not context['global_load_state_known']: reasons.append('UNKNOWN_GLOBAL_LOAD_STATE')
    if context['global_load_conflict_key_indices']: reasons.append('MATCHED_PREFIX_IN_GLOBAL_LOAD')
    if context['skip_reading_prefix_cache'] is not False: reasons.append('UNSUPPORTED_SKIP_READING_PREFIX_CACHE')
    return dict(request=rid, priority=request.priority, arrival_time=request.arrival_time,
        num_preemptions=request.num_preemptions, history_tokens=history,
        computed_tokens=request.num_computed_tokens, held_gpu_blocks=len(held),
        full_fit_extra_blocks=need, conservative_free_required=upper,
        plain_unshared=plain, context=context, memory_fit=free >= need,
        native_reservation_pass=free >= upper,
        reservation_test='conservative full-history-extra + native reservation; may reject legal async partial-prefix fits',
        reasons=reasons,
        conservative_fit=bool(valid and context['ready'] and free >= upper))


def _observer(scheduler, source_data, manager, single, cs):
    data = dict(status='INSTALLED', mode='observe_only', events=[], source_sha256=dict(PINS),
        clock='time.perf_counter host decision snapshots; no CUDA clock/query/sync',
        shadow='oldest arrival among conservative-fit PREEMPTED candidates, ordered by (priority, arrival_time, request_id); no action',
        interval='source selected_request until its first release; missing conditions remain unverified',
        observation_calls=0, observation_errors=[])
    source_index, closed, selected, step = 0, False, None, 0
    def observe(current, phase, budget, preempted, queue, request):
        nonlocal source_index, closed, selected, step
        if phase == 'waiting_entry':
            step += 1
        if closed:
            return
        selected = source_data.get('selected_request')
        for event in source_data['events'][source_index:]:
            if selected and event['kind'] == 'release':
                closed = True
        source_index = len(source_data['events'])
        if not selected or closed:
            return
        started = time.perf_counter()
        try:
            waiting, skipped = list(current.waiting), list(current.skipped_waiting)
            free = manager.block_pool.free_block_queue.num_free_blocks
            reserved = current._inflight_prefill_reserved_blocks()
            if type(reserved) is not int or reserved < 0:
                raise RuntimeError('Unknown native reservation')
            candidates = [_candidate(r, manager, single, cs, reserved, free)
                          for r in waiting + skipped if r.status.name == 'PREEMPTED']
            fits = [r for r in candidates if r['conservative_fit']]
            shadow = min(fits, key=lambda r: (r['priority'], r['arrival_time'], r['request'])) if fits else None
            running = len(current.running) + current.num_waiting_for_streaming_input
            gate = ('PREEMPTED_THIS_STEP' if preempted else 'PAUSED' if current._pause_state.name != 'UNPAUSED'
                    else 'TOKEN_BUDGET_EMPTY' if budget <= 0 else 'RUNNING_CAP' if running >= current.max_num_running_reqs
                    else 'NO_WAITERS' if not waiting and not skipped else 'MAY_ENTER_WAITING')
            for candidate in candidates:
                candidate['eligible'] = candidate['conservative_fit'] and gate == 'MAY_ENTER_WAITING'
                if gate != 'MAY_ENTER_WAITING': candidate['reasons'].append(gate)
            data['events'].append(dict(kind=phase, step=step, host_perf_s=started, selected_request=selected,
                token_budget=budget, running_count=len(current.running), running_with_streaming=running,
                running_cap=current.max_num_running_reqs, pause=current._pause_state.name,
                preempted_ids=[r.request_id for r in preempted], free_gpu_blocks=free,
                native_inflight_reserved_blocks=reserved, entry_gate=gate,
                waiting_order=[r.request_id for r in waiting], skipped_order=[r.request_id for r in skipped],
                waiting_head=waiting[0].request_id if waiting else None,
                skipped_head=skipped[0].request_id if skipped else None,
                waiting_head_status=waiting[0].status.name if waiting else None,
                skipped_head_status=skipped[0].status.name if skipped else None,
                native_head=request.request_id if request is not None else (skipped or waiting)[0].request_id if skipped or waiting else None,
                selected_queue='waiting' if queue is current.waiting else 'skipped' if queue is current.skipped_waiting else None,
                candidates=candidates, oldest_arrival_fit=None if shadow is None else shadow['request'],
                shadow_gate_open=gate == 'MAY_ENTER_WAITING', observer_return_perf_s=time.perf_counter()))
            data['observation_calls'] += 1
        except Exception as error:
            # Observation failure cannot mutate or stop native scheduling.
            data['status'] = 'UNVERIFIED'
            data['observation_errors'].append(dict(host_perf_s=started, error=repr(error)))
            closed = True
    return data, observe


def install(scheduler, source_data):
    root = Path(__file__).resolve().parents[1]
    for relative, expected in PINS.items():
        if hashlib.sha256((root/relative).read_bytes()).hexdigest() != expected:
            raise RuntimeError('Waiting observer dependency changed: '+relative)
    if source_data.get('mode') != 'native' or source_data.get('selected_request') is not None:
        raise RuntimeError('Install after native source observer, before measurement')
    spec = importlib.util.spec_from_file_location('waiting_qualifier', root/'tail_reservation/reserve_tail.py')
    qualifier = importlib.util.module_from_spec(spec); spec.loader.exec_module(qualifier)
    manager, single, reason = qualifier.qualify(scheduler)
    if (reason or scheduler.policy.name != 'FCFS' or scheduler.scheduler_config.async_scheduling
            or scheduler._rotation_lease_enabled or scheduler._rotation_target is not None):
        raise RuntimeError('Requires qualified native FCFS without protection')
    source = Path(inspect.getsourcefile(type(scheduler))).read_text()
    # Import only the already qualified patch builder used by the installed wrapper.
    import rotation_native
    if hashlib.sha256(Path(rotation_native.__file__).read_bytes()).hexdigest() != PINS['pkg/rotation_native.py']:
        raise RuntimeError('Loaded rotation source differs')
    return _attach(scheduler, source_data, manager, single,
                   scheduler.connector.connector_scheduler, rotation_native.patched_schedule_tree(source))


def _attach(scheduler, source_data, manager, single, cs, tree):
    """Qualified production entry above; CPU fixture uses this exact lifecycle."""
    slot = _native_slot(scheduler.schedule, scheduler)
    old = slot.cell_contents
    rebuilt = _compile(tree, old)
    if not _same_code(rebuilt.__func__.__code__, old.__func__.__code__):
        raise RuntimeError('Existing native schedule does not match pinned reconstruction')
    data, callback = _observer(scheduler, source_data, manager, single, cs)
    new = _compile(_instrument(tree), old, callback)
    slot.cell_contents = new
    data['native_code_verified'] = True
    active = True
    def uninstall():
        nonlocal active
        if active:
            if slot.cell_contents is not new:
                raise RuntimeError('Native schedule slot changed while observing')
            slot.cell_contents = old; active = False
            if data['status'] == 'INSTALLED':
                data['status'] = 'UNINSTALLED'
        return data
    return data, uninstall
