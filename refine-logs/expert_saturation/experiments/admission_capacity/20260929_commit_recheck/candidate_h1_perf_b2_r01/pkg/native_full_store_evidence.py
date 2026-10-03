"""Read-only evidence for native complete incremental stores; never caps a job."""


def inspect_native_full_jobs(output, connector_scheduler, preparation=None):
    cs, metadata = connector_scheduler, output.kv_connector_metadata
    if cs.config.blocks_per_chunk != 1 or len(cs.config.kv_group_configs) != 1:
        raise ValueError('Evidence requires the pinned one-group/one-block chunks')
    group_config = cs.config.kv_group_configs[0]
    rows = []
    for jid, job in metadata.store_jobs.items():
        state, registered = cs._req_status.get(job.req_id), cs._jobs.get(jid)
        if (state is None or registered is None or not registered.is_store
                or registered.req_id != job.req_id or jid not in state.transfer_jobs):
            raise ValueError('Native full store registration/identity mismatch')
        if len(state.group_states) != 1:
            raise ValueError('Unexpected native store group count')
        req, group = state.req, state.group_states[0]
        finished = req.is_finished()
        if not finished and job.req_id not in output.num_scheduled_tokens:
            raise ValueError('Native full store is neither scheduled nor finished')
        # Called after native _update_after_schedule: computed already includes
        # scheduled tokens. Adding output.num_scheduled_tokens again is wrong.
        token_boundary = req.num_tokens if finished else req.num_computed_tokens
        native_limit = cs._calc_num_offloadable_tokens(state, token_boundary)
        chunk_limit = state.storable_chunks(group_config, native_limit)
        positions = {int(b): i for i, b in enumerate(group.block_ids) if b != 0}
        blocks = [int(b) for b in job.src_spec.block_ids]
        if not blocks or len(set(blocks)) != len(blocks) or any(b not in positions for b in blocks):
            raise ValueError('Native full store source registration mismatch')
        if list(registered.non_sliding_window_block_ids or ()) != blocks:
            raise ValueError('Native full registered source payload differs')
        indices = [positions[b] for b in blocks]
        if indices != sorted(indices) or any(i >= chunk_limit for i in indices):
            raise ValueError('Native full job exceeds its native scheduled/finished range')
        keys = [group.offload_keys[i] for i in indices]
        if set(keys) != set(registered.keys):
            raise ValueError('Native full source-to-key mapping mismatch')
        selected = preparation is not None and job.req_id == preparation.victim.request_id
        rows.append(dict(job_id=jid, request=job.req_id, source_gpu_blocks=blocks,
            logical_chunk_indices=indices, registered_key_reprs=[repr(k) for k in keys],
            finished_at_metadata=finished, scheduled_tokens=output.num_scheduled_tokens.get(job.req_id, 0),
            request_prompt_tokens=req.num_prompt_tokens, known_request_tokens=req.num_tokens,
            computed_after_schedule=req.num_computed_tokens, native_token_limit=native_limit,
            native_complete_chunk_limit=chunk_limit, outside_selected_preparation=not selected,
            preparation_victim=None if preparation is None else preparation.victim.request_id,
            beyond_selected_prefix_blocks=None if not selected else
                sum(i >= preparation.saved_tokens // 16 for i in indices),
            chunks_containing_decode_positions=sum((i+1)*group_config.tokens_per_chunk > req.num_prompt_tokens for i in indices),
            registered_non_sliding_source_blocks=list(registered.non_sliding_window_block_ids or ()),
            transfer_completion='UNOBSERVED_AT_METADATA'))
    flushes = []
    for jid in sorted(metadata.jobs_to_flush):
        registered = cs._jobs.get(jid)
        flushes.append(dict(job_id=jid, registration_present=registered is not None,
            request=registered.req_id if registered is not None else None,
            is_store=registered.is_store if registered is not None else None,
            registered_source_blocks=list(registered.non_sliding_window_block_ids or ())
                if registered is not None else None))
    return dict(store_jobs=rows, flush_jobs=flushes,
        semantics='Native registered source/range evidence only. Does not assert full residency, transfer completion, or GPU-computed correctness.')
