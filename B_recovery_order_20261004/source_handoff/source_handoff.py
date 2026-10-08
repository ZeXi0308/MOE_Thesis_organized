"""One recovery's ready Host prefix: native read-reference acquisition and handoff."""
import hashlib
import importlib.util
import inspect
from pathlib import Path
import time

PINS = {
    'OffloadingConnectorScheduler': '89ac26a80fbc29b9bcaa5a0daba88fb6309247f9bb053e48d2d61efa7d9d66f1',
    'CPUOffloadingManager': 'a7a166f7756d76e18a0e571828555d5ec66113425bd09911ecab7cf27f146416',
    'LRUCachePolicy': 'fd7ddee1f288f19197f5519fe7cd73d4e92a2460e9354892cfedcba4664b7f32',
    'OffloadingManager': '6b89cf3e146bdbb995ac531e588f14326941bc804a5be103b8bfc27b57d30c31',
    'CPULoadStoreSpec': '0b7952a73376f04eb89a974e443d2dd5ce27b1a1001d8ebb628bce755c913f67',
    'BlockStatus': 'ca9d2fa873813e2887b1aae1d4667835a64ef85371deb4f1c88702f08d32ef9f',
}
QUALIFIER_SHA = 'fd663e48fbe4291878c16319ffb6294ef63e4444e79f7f84789e13d2aad0ef76'


def install(scheduler, mode='native'):
    if mode not in ('native', 'early_pin'):
        raise ValueError(mode)
    path = Path(__file__).resolve().parents[1]/'tail_reservation/reserve_tail.py'
    if hashlib.sha256(path.read_bytes()).hexdigest() != QUALIFIER_SHA:
        raise RuntimeError('Frozen allocation qualifier changed')
    spec = importlib.util.spec_from_file_location('source_handoff_qualifier', path)
    qualifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(qualifier)
    gpu, single, reason = qualifier.qualify(scheduler)  # Qualification only; no tail policy installed.
    cs = scheduler.connector.connector_scheduler
    host = cs.manager
    classes = [type(cs), type(host), type(host._policy), type(host).__mro__[1],
        host._get_load_store_spec.__func__.__globals__['CPULoadStoreSpec'],
        host._allocate_blocks.__func__.__globals__['BlockStatus']]
    for cls in classes:
        expected = PINS.get(cls.__name__)
        source = inspect.getsourcefile(cls)
        if expected is None or source is None or hashlib.sha256(Path(source).read_bytes()).hexdigest() != expected:
            raise RuntimeError('Unqualified native Host source: '+cls.__name__)
    groups = cs.config.kv_group_configs
    if (reason or scheduler.max_num_scheduled_tokens != 1024 or len(groups) != 1
            or groups[0].sliding_window_size_in_chunks is not None or groups[0].is_eagle_group
            or groups[0].tokens_per_block != single.block_size
            or groups[0].tokens_per_chunk != single.block_size or cs.config.blocks_per_chunk != 1
            or cs._chunks_being_loaded is not None):
        raise RuntimeError('Requires the pinned single plain FullAttention/no-APC 1024-token runtime')
    return _install(gpu, single, cs, mode)


def _install(gpu, single, cs, mode):
    """CPU fixtures use this entry; production must qualify through install()."""
    host, undo, tracked, owned = cs.manager, [], None, []
    data = dict(mode=mode, status='INSTALLED', selected_request=None, events=[],
        source_sha256=dict(PINS), qualifier_sha256=QUALIFIER_SHA,
        scope='First preempted async LOAD with >=1024 external tokens and full-history capacity failure; at most one request per run',
        clock='time.perf_counter host; no GPU query or transfer added')

    def emit(kind, **fields):
        data['events'].append(dict(kind=kind, host_perf_s=time.perf_counter(), **fields))

    def snapshot():
        rows = []
        for key in tracked['keys']:
            block = host._policy.get(key)
            rows.append(dict(cached=block is not None, ready=bool(block is not None and block.is_ready),
                ref_cnt=None if block is None else block.ref_cnt,
                block_id=None if block is None else block.block_id))
        return rows

    def release(reason):
        nonlocal owned
        before = snapshot()
        if owned:
            keys = list(owned)
            refs = {key: host._policy.get(key).ref_cnt for key in keys}
            try:
                host.complete_load(keys, tracked['context'])
            except BaseException as error:
                # Native complete_load is not transactional: never release a completed decrement twice.
                owned = [key for key in keys if host._policy.get(key) is not None
                         and host._policy.get(key).ref_cnt == refs[key]]
                emit('release_error', reason=reason, error=repr(error), before=before,
                     after=snapshot(), remaining_early_refs=len(owned))
                raise
            owned = []
        tracked['closed'] = True
        emit('release', reason=reason, before=before, after=snapshot(), remaining_early_refs=0)

    def prepare_early():
        nonlocal owned
        keys = tracked['keys']
        refs = {key: host._policy.get(key).ref_cnt for key in keys}
        try:
            host.prepare_load(keys, tracked['context'])  # Descriptor intentionally unused; no transfer.
        except BaseException as error:
            owned = [key for key in keys if host._policy.get(key) is not None
                     and host._policy.get(key).ref_cnt == refs[key] + 1]
            emit('prepare_error', error=repr(error), acquired_refs=len(owned), after=snapshot())
            release('prepare_exception')
            raise
        owned = list(keys)
        emit('early_pin', before_refs=[refs[key] for key in keys], after=snapshot(), acquired_refs=len(owned))

    def wrap(obj, name, factory):
        old, own = getattr(obj, name), name in vars(obj)
        new = factory(old)
        undo.append((obj, name, old, own, new))
        setattr(obj, name, new)

    def allocate(old):
        def call(request, num_new_tokens, num_new_computed_tokens=0, new_computed_blocks=None,
                 num_lookahead_tokens=0, num_external_computed_tokens=0, delay_cache_blocks=False,
                 num_encoder_tokens=0, full_sequence_must_fit=False, reserved_blocks=0,
                 has_scheduled_reqs=True):
            nonlocal tracked
            arguments = dict(num_new_computed_tokens=num_new_computed_tokens, new_computed_blocks=new_computed_blocks,
                num_lookahead_tokens=num_lookahead_tokens, num_external_computed_tokens=num_external_computed_tokens,
                delay_cache_blocks=delay_cache_blocks, num_encoder_tokens=num_encoder_tokens,
                full_sequence_must_fit=full_sequence_must_fit, reserved_blocks=reserved_blocks,
                has_scheduled_reqs=has_scheduled_reqs)
            observing = tracked is not None and not tracked['closed'] and request.request_id == tracked['request']
            before = snapshot() if observing else None
            try:
                result = old(request, num_new_tokens, **arguments)
            except BaseException:
                if observing:
                    release('allocation_exception')
                raise
            if observing:
                emit('allocation', request=request.request_id, num_new_tokens=num_new_tokens,
                     external_tokens=num_external_computed_tokens, success=result is not None,
                     gpu_held=len(single.req_to_blocks.get(request.request_id, ())),
                     gpu_free=gpu.block_pool.get_num_free_blocks(), before=before, after=snapshot())
            if tracked is not None or result is not None:
                return result
            status = cs._req_status.get(request.request_id)
            eligible = (request.num_preemptions > 0 and request.status.name == 'PREEMPTED'
                and request.num_computed_tokens == 0 and request.num_in_flight_tokens == 0
                and not request.has_encoder_inputs and not request.is_finished()
                and num_new_tokens == num_new_computed_tokens == num_lookahead_tokens == num_encoder_tokens == 0
                and 1024 <= num_external_computed_tokens <= min(request.num_tokens, gpu.max_model_len)
                and delay_cache_blocks is True and full_sequence_must_fit is True
                and not (new_computed_blocks is not None and any(new_computed_blocks.blocks))
                and not single.req_to_blocks.get(request.request_id)
                and (min(request.num_tokens, gpu.max_model_len)+single.block_size-1)//single.block_size > gpu.block_pool.get_num_free_blocks()
                and status is not None and status.req is request and not status.transfer_jobs
                and status.num_locally_computed_tokens == 0 and len(status.group_states) == 1)
            if not eligible:
                return result
            count = num_external_computed_tokens // cs.config.kv_group_configs[0].tokens_per_chunk
            keys = tuple(status.group_states[0].offload_keys[:count])
            tracked = dict(request=request.request_id, keys=keys, context=status.req_context, closed=False)
            data['selected_request'] = request.request_id
            blocks = [host._policy.get(key) for key in keys]
            valid = (num_external_computed_tokens % single.block_size == 0 and len(keys) == count
                and len(set(keys)) == count and all(b is not None and b.is_ready and b.ref_cnt >= 0 for b in blocks)
                and len({b.block_id for b in blocks if b is not None}) == count
                and host._num_evictable_cache_blocks == len(host._policy.evictable_blocks)
                and all((b.ref_cnt == 0) == (key in host._policy.evictable_blocks) for key, b in zip(keys, blocks) if b is not None))
            emit('selected', request=request.request_id, external_tokens=num_external_computed_tokens,
                 history_tokens=request.num_tokens, gpu_free=gpu.block_pool.get_num_free_blocks(),
                 key_repr=[repr(key) for key in keys], before=snapshot(), valid=valid,
                 action='early_pin' if mode == 'early_pin' and valid else 'no_early_pin')
            if not valid:
                release('unsupported_source_state')
            elif mode == 'early_pin':
                prepare_early()
            return result
        return call

    def update(old):
        def call(request, blocks, num_external_tokens):
            active = tracked is not None and not tracked['closed'] and request.request_id == tracked['request']
            try:
                result = old(request, blocks, num_external_tokens)
            except BaseException:
                if active:
                    release('native_update_exception')
                raise
            if active:
                status = cs._req_status.get(request.request_id)
                jobs = [] if status is None else [jid for jid in status.transfer_jobs if not cs._jobs[jid].is_store]
                emit('native_handoff', external_tokens=num_external_tokens, load_jobs=jobs,
                     before_early_release=snapshot())
                release('native_load_handoff' if jobs else 'native_recompute_handoff')
            return result
        return call

    def finished(old):
        def call(request, *args, **kwargs):
            if tracked is not None and not tracked['closed'] and request.request_id == tracked['request']:
                release('request_finished')
            return old(request, *args, **kwargs)
        return call

    def reset(old):
        def call(*args, **kwargs):
            if tracked is not None and not tracked['closed']:
                release('reset_cache')
            return old(*args, **kwargs)
        return call

    wrap(gpu, 'allocate_slots', allocate)
    wrap(cs, 'update_state_after_alloc', update)
    wrap(cs, 'request_finished', finished)
    wrap(cs, 'reset_cache', reset)

    def uninstall():
        try:
            if tracked is not None and not tracked['closed']:
                release('uninstall')
        finally:
            while undo:
                obj, name, old, own, new = undo.pop()
                if getattr(obj, name) is not new:
                    raise RuntimeError('Uninstall source handoff before outer wrappers')
                if own: setattr(obj, name, old)
                else: delattr(obj, name)
        data.update(status='UNINSTALLED', remaining_early_refs=len(owned))
        return data
    return data, uninstall
