"""Allocate a recovering LOAD's remaining current history in its native call."""
import hashlib
import inspect
from pathlib import Path
import time

PINS = {
    'Scheduler': '2ed2a550b6558b2495eda845a97ae38bcf0225027b9e25fbf00fc3880c1d3941',
    'KVCacheManager': '3f4af8d247f3fe9570b0132818b832b66ae6a2ac12942588828f899f6ff77ccf',
    'KVCacheCoordinatorNoPrefixCache': '4c8fbb341f0bd3714eff54ce633f534c1475220decb17c0caed40cbd02b352a2',
    'FullAttentionManager': 'bcb27e38895332bf6a4c55608f2917eb9fd941ec5a629c14b88358f00aadeba8',
    'BlockPool': '202a13cb129174849d798019aaedc04c59775ec2a4b9dfcc7c1e3c563a43a661',
}


def qualify(scheduler):
    from vllm.v1.kv_cache_interface import FullAttentionSpec

    manager = scheduler.kv_cache_manager
    coordinator, pool = manager.coordinator, manager.block_pool
    singles = coordinator.single_type_managers
    groups = manager.kv_cache_config.kv_cache_groups
    if len(singles) != 1 or len(groups) != 1:
        return manager, None, 'requires_one_group'
    single, group = singles[0], groups[0]
    for obj in (scheduler, manager, coordinator, single, pool):
        expected = PINS.get(type(obj).__name__)
        source = inspect.getsourcefile(type(obj))
        if expected is None or source is None or hashlib.sha256(Path(source).read_bytes()).hexdigest() != expected:
            raise RuntimeError('Native allocation source changed: '+type(obj).__name__)
    spec = group.kv_cache_spec
    if (type(spec) is not FullAttentionSpec or group.is_eagle_group
            or spec.sliding_window is not None or spec.attention_chunk_size is not None
            or manager.num_kv_cache_groups != 1 or manager.enable_caching or coordinator.enable_caching
            or manager.use_eagle or scheduler.num_spec_tokens or scheduler.num_lookahead_tokens
            or scheduler.is_encoder_decoder or manager.watermark_blocks
            or scheduler.dcp_world_size != 1 or scheduler.pcp_world_size != 1
            or single._max_admission_blocks_per_request is not None
            or single.block_pool is not pool or single.block_size != spec.block_size
            or coordinator.scheduler_block_size != single.block_size):
        return manager, single, 'unsupported_runtime'
    return manager, single, None


def install(scheduler, mode='native'):
    if mode not in ('native', 'full_tail'):
        raise ValueError(mode)
    manager, single, fallback = qualify(scheduler)
    return _install(manager, single, mode, fallback)


def _install(manager, single, mode, fallback=None):
    old, own = manager.allocate_slots, 'allocate_slots' in vars(manager)
    if getattr(old, '_tail_reservation_wrapper', False):
        raise RuntimeError('Tail reservation already installed')
    data = dict(mode=mode, qualification_fallback=fallback, events=[], source_sha256=dict(PINS),
        clock='time.perf_counter host; original allocator included in call interval',
        scope='Recovering async LOAD only; physical current-history allocation through one native call')
    active = True

    def snapshot(request):
        if single is None:
            return dict(held_blocks=None, free_blocks=None)
        return dict(held_blocks=len(single.req_to_blocks.get(request.request_id, ())),
                    free_blocks=int(manager.block_pool.free_block_queue.num_free_blocks))

    def call(request, num_new_tokens, num_new_computed_tokens=0, new_computed_blocks=None,
             num_lookahead_tokens=0, num_external_computed_tokens=0, delay_cache_blocks=False,
             num_encoder_tokens=0, full_sequence_must_fit=False, reserved_blocks=0,
             has_scheduled_reqs=True):
        kwargs = dict(num_new_computed_tokens=num_new_computed_tokens,
            new_computed_blocks=new_computed_blocks, num_lookahead_tokens=num_lookahead_tokens,
            num_external_computed_tokens=num_external_computed_tokens, delay_cache_blocks=delay_cache_blocks,
            num_encoder_tokens=num_encoder_tokens, full_sequence_must_fit=full_sequence_must_fit,
            reserved_blocks=reserved_blocks, has_scheduled_reqs=has_scheduled_reqs)
        if not (request.num_preemptions > 0 and num_new_tokens == 0 and num_external_computed_tokens > 0):
            return old(request, num_new_tokens, **kwargs)
        started = time.perf_counter()
        reason, tail = fallback, 0
        if reason is None:
            if delay_cache_blocks is not True or full_sequence_must_fit is not True:
                reason = 'unsupported_call_flags'
            elif (num_lookahead_tokens or num_encoder_tokens or num_new_computed_tokens
                    or (new_computed_blocks is not None and any(new_computed_blocks.blocks))
                    or request.has_encoder_inputs or request.num_in_flight_tokens
                    or request.request_id in single._partial_hit_reqs):
                reason = 'unsupported_call_state'
            else:
                history = request.num_tokens
                computed = request.num_computed_tokens + num_new_computed_tokens + num_external_computed_tokens
                if not 0 <= computed <= history <= manager.max_model_len:
                    reason = 'invalid_history_bounds'
                else:
                    tail = history - computed
                    if not tail:
                        reason = 'no_tail'
        effective = tail if mode == 'full_tail' and reason is None else num_new_tokens
        requested = {key: value for key, value in kwargs.items() if key != 'new_computed_blocks'}
        requested['num_new_tokens'] = num_new_tokens
        row = dict(request=request.request_id, host_perf_s=started,
            num_preemptions=request.num_preemptions, request_num_tokens=request.num_tokens,
            request_num_computed_tokens=request.num_computed_tokens,
            requested=requested, effective=dict(requested, num_new_tokens=effective),
            tail_tokens=tail, changed=effective != num_new_tokens, fallback=reason,
            before=snapshot(request), original_calls=0, outcome='pending')
        data['events'].append(row)
        try:
            row['original_calls'] = 1
            result = old(request, effective, **kwargs)
            row.update(outcome='allocation_failed' if result is None else 'allocated',
                returned_new_blocks=None if result is None else sum(len(blocks) for blocks in result.blocks))
            return result
        except BaseException as error:
            row.update(outcome='exception', error=type(error).__name__+': '+str(error))
            raise
        finally:
            row.update(after=snapshot(request), return_host_perf_s=time.perf_counter())

    call._tail_reservation_wrapper = True
    manager.allocate_slots = call

    def uninstall():
        nonlocal active
        if active:
            if manager.allocate_slots is not call:
                raise RuntimeError('Uninstall tail reservation before outer wrappers')
            if own:
                manager.allocate_slots = old
            else:
                del manager.allocate_slots
            active = False
        return data

    return data, uninstall
