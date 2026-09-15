"""Derive one static cap from the live, qualified full-attention KV block pool."""


def qualify_safe_cap(engine, config):
    result = dict(status='QUALIFICATION_FAILED', measurement_status='UNRUN',
        formula='min(engine_max_num_seqs, usable_blocks // ceil((prompt_tokens + output_tokens) / block_size))')

    def require(condition, message):
        if not condition:
            raise ValueError(message)

    try:
        from vllm.v1.kv_cache_interface import FullAttentionSpec
        scheduler = engine.engine_core.engine_core.scheduler
        manager = scheduler.kv_cache_manager
        layout, pool, coordinator = scheduler.kv_cache_config, manager.block_pool, manager.coordinator
        groups = layout.kv_cache_groups
        result.update(group_count=len(groups), manager_group_count=manager.num_kv_cache_groups,
            coordinator_type=type(coordinator).__name__, watermark_blocks=manager.watermark_blocks,
            prefix_caching=manager.enable_caching, use_eagle=manager.use_eagle,
            num_lookahead_tokens=scheduler.num_lookahead_tokens, num_spec_tokens=scheduler.num_spec_tokens,
            dcp_world_size=scheduler.dcp_world_size, pcp_world_size=scheduler.pcp_world_size)
        require(len(groups) == manager.num_kv_cache_groups == 1, 'requires one KV cache group')
        require(type(coordinator).__name__ == 'KVCacheCoordinatorNoPrefixCache', 'unverified KV coordinator')
        group, spec = groups[0], groups[0].kv_cache_spec
        result.update(spec_type=type(spec).__name__, layer_names=list(group.layer_names),
            is_eagle_group=group.is_eagle_group)
        require(type(spec) is FullAttentionSpec and not group.is_eagle_group, 'requires plain FullAttentionSpec')
        require(spec.sliding_window is None and spec.attention_chunk_size is None, 'window/chunk attention unsupported')
        require(manager.watermark_blocks == 0, 'nonzero watermark unsupported')
        require(not manager.enable_caching and not engine.vllm_config.cache_config.enable_prefix_caching,
                'prefix sharing unsupported')
        require(not manager.use_eagle and engine.vllm_config.speculative_config is None
                and scheduler.num_spec_tokens == scheduler.num_lookahead_tokens == 0, 'speculative/lookahead unsupported')
        require(scheduler.dcp_world_size == scheduler.pcp_world_size == 1, 'context parallel unsupported')
        single = coordinator.single_type_managers
        require(len(single) == 1 and single[0].block_pool is pool, 'group does not share the verified block pool')
        block_size = spec.block_size
        result.update(single_type_block_size=single[0].block_size,
                      scheduler_block_size=coordinator.scheduler_block_size)
        require(type(block_size) is int and block_size > 0
                and single[0].block_size == coordinator.scheduler_block_size == block_size,
                'unverified logical block size')
        total, free = int(pool.num_gpu_blocks), int(pool.get_num_free_blocks())
        usable = total - 1
        result.update(block_size=block_size, total_blocks=total, usable_blocks=usable, free_blocks=free,
            null_block_id=pool.null_block.block_id, null_block_is_null=pool.null_block.is_null)
        require(pool.null_block.block_id == 0 and pool.null_block.is_null, 'unverified null block reservation')
        require(layout.num_blocks == manager.kv_cache_config.num_blocks == total and usable > 0 and free == usable,
                'pool layout mismatch or pool not empty')
        require(not scheduler.requests and scheduler.get_request_counts() == (0, 0)
                and not engine.has_unfinished_requests(), 'engine not drained')
        require(config['prompt_tokens'] == 3072 and config['output_tokens'] == 1024
                and config['requests'] == 32, 'frozen request dimensions differ')
        maximum = engine.vllm_config.scheduler_config.max_num_seqs
        require(maximum == 32 and engine.vllm_config.model_config.max_model_len == 4096, 'engine bounds differ')
        per_request = (config['prompt_tokens'] + config['output_tokens'] + block_size - 1) // block_size
        safe = min(maximum, usable // per_request)
        result.update(per_request_reserved_blocks=per_request, maximum_request_tokens=4096,
            engine_max_num_seqs=maximum, safe_cap=safe, reserved_blocks_at_safe_cap=safe * per_request,
            remaining_blocks_at_full_reservation=usable - safe * per_request)
        result.update(status='QUALIFIED', measurement_status='NOT_YET_RUN')
    except (AttributeError, ImportError, IndexError, KeyError, TypeError, ValueError) as error:
        result['error'] = f'{type(error).__name__}: {error}'
    return result
