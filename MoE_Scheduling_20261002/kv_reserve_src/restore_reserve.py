"""Simple native CPU recovery baseline: reserve one aligned tail block."""
import inspect


def install(scheduler, *, enabled=False):
    data = dict(enabled=enabled, installed=False, calls=0, eligible=0, allocated=0,
                denied=0, actions=[], scope="Measurement-only native allocation; no loaded-token or computed-token inflation")
    if not enabled:
        return data, lambda: None
    connector = scheduler.connector
    if type(connector).__name__ != "OffloadingConnector":
        raise ValueError("restore reservation requires native OffloadingConnector")
    cs = connector.connector_scheduler
    groups = cs.config.kv_group_configs
    if (type(cs.manager).__name__ != "CPUOffloadingManager" or cs.config.num_workers != 1
            or len(groups) != 1 or groups[0].sliding_window_size_in_chunks is not None
            or groups[0].tokens_per_block != scheduler.block_size
            or scheduler.is_encoder_decoder or scheduler.num_spec_tokens or scheduler.num_lookahead_tokens
            or getattr(scheduler.scheduler_config, "async_scheduling", False)):
        raise ValueError("restore reservation requires single-worker full-attention synchronous non-speculative CPU offload")
    manager, block_size = scheduler.kv_cache_manager, scheduler.block_size
    original, had_instance = manager.allocate_slots, "allocate_slots" in vars(manager)

    def allocate(request, num_new_tokens, num_new_computed_tokens=0, new_computed_blocks=None,
                 num_lookahead_tokens=0, num_external_computed_tokens=0, delay_cache_blocks=False,
                 num_encoder_tokens=0, full_sequence_must_fit=False, reserved_blocks=0,
                 has_scheduled_reqs=True):
        data["calls"] += 1
        cached = request.num_computed_tokens + num_new_computed_tokens + num_external_computed_tokens
        tail = request.num_tokens - cached
        eligible = (request.status.name == "PREEMPTED" and request.num_preemptions > 0
            and request.num_computed_tokens == 0 and num_new_tokens == 0
            and num_external_computed_tokens > 0 and delay_cache_blocks
            and num_lookahead_tokens == 0 and num_encoder_tokens == 0
            and not request.spec_token_ids and request.num_in_flight_tokens == 0
            and cached > 0 and cached % block_size == 0 and 0 < tail <= block_size
            and cached + 1 <= manager.max_model_len)
        receipt = None
        if eligible:
            data["eligible"] += 1
            num_lookahead_tokens = 1
            receipt = dict(request_id=request.request_id, scheduler_step_seq=getattr(scheduler, "sched_step_seq", None),
                num_preemptions=request.num_preemptions, cached_prefix_tokens=cached, known_tail_tokens=tail,
                block_size=block_size, external_tokens=num_external_computed_tokens, num_new_tokens=num_new_tokens,
                lookahead_before=0, lookahead_after=1, free_blocks_before=manager.block_pool.get_num_free_blocks(),
                outcome="pending")
            data["actions"].append(receipt)
        result = original(request, num_new_tokens, num_new_computed_tokens=num_new_computed_tokens,
            new_computed_blocks=new_computed_blocks, num_lookahead_tokens=num_lookahead_tokens,
            num_external_computed_tokens=num_external_computed_tokens, delay_cache_blocks=delay_cache_blocks,
            num_encoder_tokens=num_encoder_tokens, full_sequence_must_fit=full_sequence_must_fit,
            reserved_blocks=reserved_blocks, has_scheduled_reqs=has_scheduled_reqs)
        if receipt is not None:
            outcome = "allocated" if result is not None else "denied"
            data[outcome] += 1
            receipt.update(outcome=outcome, free_blocks_after=manager.block_pool.get_num_free_blocks(),
                total_request_blocks_after=[len(ids) for ids in manager.get_block_ids(request.request_id)]
                    if result is not None else None)
        return result

    # The pinned bound v0.26 API must match; never silently drop a native allocation parameter.
    shape = lambda fn: [(p.name, p.kind, p.default) for p in inspect.signature(fn).parameters.values()]
    if shape(original) != shape(allocate):
        raise ValueError("pinned KVCacheManager.allocate_slots signature changed")
    manager.allocate_slots = allocate
    data.update(installed=True, block_size=block_size, native_signature=str(inspect.signature(original)))

    def uninstall():
        if had_instance:
            manager.allocate_slots = original
        else:
            del manager.allocate_slots
    return data, uninstall
