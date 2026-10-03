"""Default-off compute sharing after an existing rotation adapter is installed.

The base adapter owns target/victim, saving, KV protection and its release.
This hook only withholds peer decode when a resident recomputing target would
otherwise lose its compute-only minimum-call bound. All hook/observation cost
belongs to treatment time. No first-output or elapsed-time guarantee is implied.
"""
from recovery_execution_share import Request, State, allocate


def install(scheduler, *, enabled=False, block_size=16, diagnostic=False):
    if type(enabled) is not bool:
        raise ValueError('enabled must be bool')
    data = dict(enabled=enabled, planned_changes=0, actual_compute_holds=0,
                resident_recompute_checks=0, events=[])
    if not enabled:
        return data, lambda: data  # Baseline methods stay exactly intact.
    if block_size != 16 or not all(hasattr(scheduler, k) for k in
            ('_rotation_begin', '_rotation_hold', '_rotation_target')):
        raise ValueError('install after the qualified 16-token rotation adapter')
    manager = scheduler.kv_cache_manager
    groups = manager.coordinator.single_type_managers
    if (manager.enable_caching or len(groups) != 1
            or type(groups[0]).__name__ != 'FullAttentionManager'
            or scheduler.scheduler_config.async_scheduling
            or scheduler.scheduler_config.long_prefill_token_threshold):
        raise ValueError('requires synchronous unshared full attention without a prefill clamp')
    owned, pool = groups[0].req_to_blocks, manager.block_pool
    original_begin, original_hold = scheduler._rotation_begin, scheduler._rotation_hold
    held, calls = frozenset(), 0

    def view(r):
        return Request(r.request_id, r.num_tokens, r.num_computed_tokens,
            len(owned.get(r.request_id, ())), r.num_output_tokens,
            r.max_tokens-r.num_output_tokens)

    def begin(preempted, timestamp):
        nonlocal held, calls
        held = frozenset(); calls += 1
        original_begin(preempted, timestamp)
        target = scheduler.requests.get(scheduler._rotation_target)
        # Load transitions, new target selection and post-output service belong
        # to the base adapter. Never move a target to make this rule applicable.
        if (target is None or target.status.name != 'RUNNING'
                or not scheduler.running or scheduler.running[-1] is not target
                or target.num_tokens-target.num_computed_tokens <= 1):
            return
        peers = tuple(r for r in scheduler.running if r is not target)
        if any(r.status.name != 'RUNNING' or not r.num_output_tokens
               or r.num_tokens-r.num_computed_tokens != 1 for r in peers):
            return
        state = State(view(target), tuple(view(r) for r in peers),
            pool.get_num_free_blocks(), block_size, scheduler.max_num_scheduled_tokens)
        data['resident_recompute_checks'] += 1
        baseline, candidate = allocate(state), allocate(state, 'preserve_calls')
        if (baseline['status'] != 'READY' or candidate['status'] != 'READY'
                or candidate['scheduled'] == baseline['scheduled']):
            return
        held = frozenset(candidate['held'])
        data['planned_changes'] += 1
        if diagnostic:
            data['events'].append(dict(hook_call=calls, target=target.request_id,
                pending=candidate['target_pending'],
                minimum_target_positions=candidate['target_min_positions'],
                planned_scheduled=candidate['scheduled'], planned_held=candidate['held']))

    def hold(request):
        if original_hold(request):
            return True
        if request.request_id in held:
            data['actual_compute_holds'] += 1
            return True
        return False

    scheduler._rotation_begin, scheduler._rotation_hold = begin, hold

    def uninstall():
        if scheduler._rotation_begin is not begin or scheduler._rotation_hold is not hold:
            raise RuntimeError('uninstall compute sharing before the base adapter')
        scheduler._rotation_begin, scheduler._rotation_hold = original_begin, original_hold
        return data
    return data, uninstall
