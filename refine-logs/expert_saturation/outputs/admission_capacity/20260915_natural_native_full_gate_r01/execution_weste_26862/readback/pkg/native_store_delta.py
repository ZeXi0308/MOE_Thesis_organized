"""Read-only validation of new native store jobs for one full-attention group.

No lookup API is called. Validating new stores does not establish residency of
older keys, full-prefix coverage, transfer completion or future load success.
"""

def inspect_store_delta(plan, metadata, request_state, registered_jobs):
    if request_state.req.request_id!=plan.victim.request_id:
        raise ValueError('Request identity mismatch')
    if len(request_state.group_states)!=1:
        raise ValueError('Only one full-attention group supported')
    group=request_state.group_states[0]
    expected=plan.source_blocks
    if tuple(group.block_ids[:len(expected)])!=expected:
        raise ValueError('Preparation source ownership changed')
    if len(group.offload_keys)<len(expected):
        raise ValueError('Missing logical offload keys')
    positions={bid:n for n,bid in enumerate(expected)}
    if len(positions)!=len(expected):raise ValueError('Duplicate source blocks')
    seen=set();rows=[]
    for jid,job in metadata.store_jobs.items():
        if job.req_id!=plan.victim.request_id:continue
        registered=registered_jobs.get(jid)
        if (registered is None or not registered.is_store
            or registered.req_id!=plan.victim.request_id
            or jid not in request_state.transfer_jobs):
            raise ValueError('Store registration mismatch')
        blocks=tuple(int(b) for b in job.src_spec.block_ids)
        if not blocks or len(set(blocks))!=len(blocks) or seen.intersection(blocks):
            raise ValueError('Empty or duplicate store payload')
        if any(b not in positions for b in blocks):
            raise ValueError('Store outside prepared complete prefix')
        indices=[positions[b] for b in blocks]
        if indices!=sorted(indices):raise ValueError('Logical source order mismatch')
        keys={group.offload_keys[n] for n in indices}
        if keys!=set(registered.keys):raise ValueError('Store key and source mapping mismatch')
        seen.update(blocks)
        rows.append(dict(job=jid,logical_indices=indices,blocks=list(blocks)))
    return dict(status='NEW_STORES_VALID' if rows else 'NO_NEW_STORE',jobs=rows,
                new_store_blocks=len(seen),prepared_prefix_blocks=len(expected),
                full_prefix_residency='UNKNOWN',transfer_completion='UNKNOWN')
