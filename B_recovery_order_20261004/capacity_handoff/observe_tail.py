"""Recovering-request allocation observations; never calls allocation/lookup twice."""
import hashlib
import inspect
import time

PINS = {
    'scheduler': '2ed2a550b6558b2495eda845a97ae38bcf0225027b9e25fbf00fc3880c1d3941',
    'manager': '3f4af8d247f3fe9570b0132818b832b66ae6a2ac12942588828f899f6ff77ccf',
    'coordinator': '4c8fbb341f0bd3714eff54ce633f534c1475220decb17c0caed40cbd02b352a2',
    'single_type': 'bcb27e38895332bf6a4c55608f2917eb9fd941ec5a629c14b88358f00aadeba8',
    'pool': '202a13cb129174849d798019aaedc04c59775ec2a4b9dfcc7c1e3c563a43a661',
}
ARGUMENTS = ('request', 'num_new_tokens', 'num_new_computed_tokens', 'new_computed_blocks',
             'num_lookahead_tokens', 'num_external_computed_tokens', 'delay_cache_blocks',
             'num_encoder_tokens', 'full_sequence_must_fit', 'reserved_blocks', 'has_scheduled_reqs')
DEFAULTS = dict(num_new_computed_tokens=0, new_computed_blocks=None, num_lookahead_tokens=0,
                num_external_computed_tokens=0, delay_cache_blocks=False, num_encoder_tokens=0,
                full_sequence_must_fit=False, reserved_blocks=0, has_scheduled_reqs=True)


def _qualify(scheduler):
    manager = scheduler.kv_cache_manager
    coordinator = manager.coordinator
    single = coordinator.single_type_managers
    objects = dict(scheduler=scheduler, manager=manager, coordinator=coordinator, pool=manager.block_pool)
    if len(single) == 1:
        objects['single_type'] = single[0]
    hashes = {}
    for name, obj in objects.items():
        with open(inspect.getsourcefile(type(obj)), 'rb') as stream:
            hashes[name] = hashlib.sha256(stream.read()).hexdigest()
        if hashes[name] != PINS[name]:
            raise RuntimeError('Unqualified native allocation source: '+name)
    groups = manager.kv_cache_config.kv_cache_groups
    supported = (len(groups) == len(single) == 1
        and type(coordinator).__name__ == 'KVCacheCoordinatorNoPrefixCache'
        and type(single[0]).__name__ == 'FullAttentionManager'
        and type(groups[0].kv_cache_spec).__name__ == 'FullAttentionSpec'
        and not manager.enable_caching and not manager.use_eagle
        and manager.watermark_blocks == 0
        and single[0]._max_admission_blocks_per_request is None
        and type(single[0].block_size) is int and single[0].block_size > 0)
    return dict(source_sha256=hashes, exact_layout=supported,
                block_size=single[0].block_size if len(single) == 1 else None,
                admission_caps=[getattr(m, '_max_admission_blocks_per_request', 'unknown') for m in single],
                watermark_blocks=manager.watermark_blocks)


def _snapshot(manager, request):
    groups = [m.req_to_blocks.get(request.request_id, ()) for m in manager.coordinator.single_type_managers]
    return dict(status=getattr(request.status, 'name', str(request.status)),
        num_tokens=request.num_tokens, num_computed_tokens=request.num_computed_tokens,
        num_in_flight_tokens=getattr(request, 'num_in_flight_tokens', None),
        held_gpu_blocks=[len(blocks) for blocks in groups],
        plain_unshared_blocks=all(getattr(b, 'is_null', None) is False and getattr(b, 'ref_cnt', None) == 1
                                  for blocks in groups for b in blocks),
        free_gpu_blocks=manager.block_pool.get_num_free_blocks())


def _describe(manager, qualified, before, arguments):
    blocks = arguments['new_computed_blocks']
    counts = [len(group) for group in blocks.blocks] if blocks is not None else []
    raw = {key: arguments[key] for key in ARGUMENTS[1:] if key != 'new_computed_blocks'}
    raw['new_computed_block_counts'] = counts
    exact = (qualified['exact_layout'] and before['plain_unshared_blocks']
        and len(before['held_gpu_blocks']) == 1 and not any(counts)
        and arguments['num_new_computed_tokens'] == 0
        and arguments['num_lookahead_tokens'] == 0 and arguments['num_encoder_tokens'] == 0
        and all(type(arguments[k]) is int and arguments[k] >= 0 for k in
                ('num_new_tokens', 'num_external_computed_tokens', 'reserved_blocks')))
    descriptor = dict(exact=exact, scope='Pinned plain FullAttention, no local hits/sharing/lookahead/admission cap/watermark')
    if exact:
        size = qualified['block_size']; held = before['held_gpu_blocks'][0]
        computed = min(before['num_computed_tokens']+arguments['num_external_computed_tokens'], manager.max_model_len)
        tokens = min(computed+arguments['num_new_tokens'], manager.max_model_len)
        extra = max(0, (tokens+size-1)//size-held)
        full_tokens = min(before['num_tokens'], manager.max_model_len)
        full_extra = max(0, (full_tokens+size-1)//size-held) if arguments['full_sequence_must_fit'] else 0
        required_free = max(full_extra, extra+arguments['reserved_blocks'])
        descriptor.update(slot_tokens=tokens, slot_extra_blocks=extra, full_fit_extra_blocks=full_extra,
            required_free_blocks=required_free, shortfall_blocks=max(0, required_free-before['free_gpu_blocks']))
    return raw, descriptor


def install(scheduler):
    return _install(scheduler, _qualify(scheduler))


def _install(scheduler, qualified):
    """Internal entry for CPU doubles; production callers must use install()."""
    manager = scheduler.kv_cache_manager
    data = dict(status='INSTALLED', qualification=qualified, events=[], allocation_attempts=0,
        clock='host time.perf_counter; schedule is a native plan, not GPU execution or output',
        scope='Preempt through first resumed schedule only; no new lookup/allocation/CUDA calls; no repeat suppression')
    recovering, undo = {}, []

    def wrap(obj, name, function):
        old, own = getattr(obj, name), name in vars(obj)
        undo.append((obj, name, old, own)); setattr(obj, name, function(old))

    def preempt(old):
        def call(*args, **kwargs):
            request = args[0] if args else kwargs.get('request')
            if request is None:
                return old(*args, **kwargs)
            row = dict(kind='preempt', request=request.request_id, before=_snapshot(manager, request),
                       begin_host_perf_s=time.perf_counter())
            try:
                result = old(*args, **kwargs)
            except BaseException as error:
                row.update(exception=type(error).__name__, end_host_perf_s=time.perf_counter())
                data['events'].append(row); raise
            row.update(end_host_perf_s=time.perf_counter(), after=_snapshot(manager, request))
            data['events'].append(row)
            recovering[request.request_id] = request
            return result
        return call

    def allocate(old):
        def call(*args, **kwargs):
            request = args[0] if args else kwargs.get('request')
            if request is None or request.request_id not in recovering:
                return old(*args, **kwargs)
            if (len(args) > len(ARGUMENTS) or any(k not in ARGUMENTS for k in kwargs)
                    or any(k in kwargs for k in ARGUMENTS[:len(args)])
                    or len(args) < 2 and 'num_new_tokens' not in kwargs):
                return old(*args, **kwargs)  # Preserve native argument-validation exceptions.
            arguments = dict(DEFAULTS); arguments.update(zip(ARGUMENTS, args)); arguments.update(kwargs)
            before = _snapshot(manager, request)
            raw, descriptor = _describe(manager, qualified, before, arguments)
            data['allocation_attempts'] += 1
            row = dict(kind='allocate', request=request.request_id, attempt=data['allocation_attempts'],
                before=before, arguments=raw, descriptor=descriptor, begin_host_perf_s=time.perf_counter())
            try:
                result = old(*args, **kwargs)
            except BaseException as error:
                row.update(success=False, exception=type(error).__name__, end_host_perf_s=time.perf_counter(),
                           after=_snapshot(manager, request))
                data['events'].append(row); raise
            row.update(success=result is not None, end_host_perf_s=time.perf_counter(), after=_snapshot(manager, request))
            data['events'].append(row)
            return result
        return call

    def schedule(old):
        def call(*args, **kwargs):
            result = old(*args, **kwargs)
            for rid, tokens in result.num_scheduled_tokens.items():
                if tokens > 0 and rid in recovering:
                    request = recovering.pop(rid)
                    data['events'].append(dict(kind='resumed_schedule', request=rid, scheduled_tokens=tokens,
                        host_perf_s=time.perf_counter(), after=_snapshot(manager, request)))
            return result
        return call

    wrap(scheduler, '_preempt_request', preempt)
    wrap(manager, 'allocate_slots', allocate)
    wrap(scheduler, 'schedule', schedule)

    def uninstall():
        while undo:
            obj, name, old, own = undo.pop()
            if own: setattr(obj, name, old)
            else: delattr(obj, name)
        data.update(status='UNINSTALLED', still_recovering=list(recovering))
        return data
    return data, uninstall
