"""Observation-only block counts for the inspected vLLM 0.26 coordinator."""


def make_reader(manager, mode):
    if mode == 'original':
        return lambda request_id: [len(g) for g in manager.get_blocks(request_id).blocks]
    if mode != 'direct':
        raise ValueError('Unknown block-count observation mode: '+mode)
    groups = manager.coordinator.single_type_managers
    empty_counts = [len(g) for g in manager.empty_kv_cache_blocks.blocks]

    def read(request_id):
        # No list of blocks, tuple, generator or KVCacheBlocks wrapper is built.
        # A fresh counts list is retained by each historical observation.
        counts = [len(g.req_to_blocks.get(request_id) or ()) for g in groups]
        return counts if any(counts) else empty_counts.copy()

    return read
