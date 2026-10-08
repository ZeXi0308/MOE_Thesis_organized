"""Stable ordering of one native ready-LOAD batch; no transfer lifecycle changes."""
import hashlib
import inspect
from operator import index
from pathlib import Path
import sys
import time

WORKER_SHA = 'd8f1a45cd01ea97307552c12bb8420dabbb63388fce10bd8b4b77d27bc5f1d35'
HANDLER_SHA = 'a3957e8f1868478a58bc2e4d54c707baef4e6b3f7383dae0ec868cf7d3a3ec98'


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def layout(scheduler, worker, full_attention_type):
    """Conservative one-time metadata check; no tensor reads or CUDA queries."""
    try:
        groups = scheduler.kv_cache_config.kv_cache_groups
        require(len(groups) == 1 and type(groups[0].kv_cache_spec) is full_attention_type,
                'unknown_attention_layout')
        spec = groups[0].kv_cache_spec
        require(spec.sliding_window is None and spec.attention_chunk_size is None
                and not groups[0].is_eagle_group, 'unsupported_attention_layout')
        h = worker._load_handler
        require(type(h).__name__ == 'SingleDirectionOffloadingHandler' and h.gpu_to_cpu is False
                and h.dst_blocks_per_chunk == 1, 'unknown_load_handler')
        chunk = index(h.src_blocks_per_chunk)
        require(chunk > 0 and len(h.kv_cache_groups_data_refs) == 1, 'unknown_group_mapping')
        refs = h.kv_cache_groups_data_refs[0]
        require(bool(refs), 'empty_group_refs')
        intervals, capacities, used = [], set(), set()
        page_bytes = 0
        for ref in refs:
            ti, size = index(ref.tensor_idx), index(ref.page_size_bytes)
            require(0 <= ti < len(h.dst_tensors) and ti < len(h.src_tensors)
                    and ti not in used, 'unknown_or_aliased_tensor_ref')
            used.add(ti)
            dst, src = h.dst_tensors[ti], h.src_tensors[ti]
            for tensor, device in ((dst, 'cuda'), (src, 'cpu')):
                require(str(tensor.dtype) == 'torch.int8' and tensor.ndim == 2
                        and tensor.device.type == device and tensor.element_size() == 1,
                        'unknown_tensor_layout')
                require(tensor.shape[0] > 0 and tensor.shape[1] > 0
                        and tensor.stride(1) == 1 and tensor.stride(0) >= tensor.shape[1],
                        'overlapping_tensor_rows')
            require(0 < size <= dst.shape[1] and src.shape[1] == dst.shape[1] * chunk,
                    'unknown_page_geometry')
            begin = int(dst.data_ptr())
            require(begin > 0, 'unknown_destination_pointer')
            end = begin + (int(dst.shape[0])-1)*int(dst.stride(0)) + int(dst.shape[1])
            intervals.append((begin, end))
            capacities.add((int(dst.shape[0]), int(src.shape[0])))
            page_bytes += size
        intervals.sort()
        require(all(a[1] <= b[0] for a, b in zip(intervals, intervals[1:])),
                'aliased_destination_tensors')
        require(len(capacities) == 1, 'inconsistent_pool_capacities')
        gpu_blocks, cpu_blocks = capacities.pop()
        return dict(gpu_blocks=gpu_blocks, cpu_blocks=cpu_blocks,
                    bytes_per_gpu_block=page_bytes, blocks_per_chunk=chunk), None
    except (AttributeError, TypeError, ValueError, IndexError) as error:
        return None, str(error) or type(error).__name__


def batch_bytes(items, geometry, cpu_type, gpu_type, array_type):
    sizes = {jid: None for jid, _ in items}
    try:
        seen = set()
        for jid, entry in items:
            require(type(entry.src_spec) is cpu_type and type(entry.dst_spec) is gpu_type,
                    'unknown_load_spec')
            src, dst = entry.src_spec, entry.dst_spec
            for ids in (src.block_ids, dst.block_ids):
                require(isinstance(ids, array_type) and ids.ndim == 1
                        and ids.dtype.kind in ('i', 'u'), 'unknown_block_ids')
            groups, offsets = list(dst.group_sizes), list(dst.block_indices)
            require(len(groups) == len(offsets) == 1, 'unknown_spec_groups')
            count, offset = index(groups[0]), index(offsets[0])
            require(count >= 0 and offset >= 0 and count == len(dst.block_ids), 'invalid_group_size')
            gpu_ids, cpu_ids = list(map(index, dst.block_ids)), list(map(index, src.block_ids))
            require(all(0 < bid < geometry['gpu_blocks'] for bid in gpu_ids)
                    and all(0 <= bid < geometry['cpu_blocks'] for bid in cpu_ids), 'block_out_of_range')
            chunk = geometry['blocks_per_chunk']
            expected_src = (count + offset % chunk + chunk - 1)//chunk if count else 0
            require(len(cpu_ids) == expected_src, 'invalid_source_size')
            require(len(set(gpu_ids)) == len(gpu_ids) and not seen.intersection(gpu_ids),
                    'destination_alias')
            seen.update(gpu_ids)
            sizes[jid] = count * geometry['bytes_per_gpu_block']
        return sizes, None
    except (AttributeError, TypeError, ValueError, IndexError) as error:
        return sizes, str(error) or type(error).__name__


def install(scheduler, mode='native'):
    if mode not in ('native', 'short_load'):
        raise ValueError(mode)
    from numpy import ndarray
    from vllm.v1.kv_cache_interface import FullAttentionSpec
    from vllm.v1.kv_offload.base import GPULoadStoreSpec
    from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec

    agent = sys.modules['vllm.distributed.kv_transfer.kv_transfer_state']._KV_CONNECTOR_AGENT
    cw = agent.connector_worker
    worker = cw.worker
    if (type(agent).__name__ != 'OffloadingConnector'
            or type(scheduler.connector).__name__ != 'OffloadingConnector'
            or type(cw).__name__ != 'OffloadingConnectorWorker'
            or type(cw.spec).__name__ != 'CPUOffloadingSpec'
            or type(worker).__name__ != 'CPUOffloadingWorker'):
        raise RuntimeError('Requires the native single-process CPU offload connector')
    for cls, expected in ((type(cw), WORKER_SHA), (type(worker), HANDLER_SHA)):
        source = inspect.getsourcefile(cls)
        if source is None or hashlib.sha256(Path(source).read_bytes()).hexdigest() != expected:
            raise RuntimeError('Native offload source changed: '+cls.__name__)
    geometry, layout_fallback = layout(scheduler, worker, FullAttentionSpec)
    data = dict(mode=mode, events=[], layout=geometry, layout_fallback=layout_fallback,
                clock='time.perf_counter host; decision cost excludes native submission',
                source_sha256=dict(worker=WORKER_SHA, handler=HANDLER_SHA))
    old, own = cw.start_kv_transfers, 'start_kv_transfers' in vars(cw)
    if getattr(old, '_load_order_wrapper', False):
        raise RuntimeError('LOAD ordering already installed')
    active = True

    def call(metadata):
        original = metadata.load_jobs
        if len(original) < 2:
            return old(metadata)
        started = time.perf_counter()
        items = list(original.items())
        before = [jid for jid, _ in items]
        sizes, fallback = ({jid: None for jid in before}, layout_fallback)
        if fallback is None:
            sizes, fallback = batch_bytes(items, geometry, CPULoadStoreSpec, GPULoadStoreSpec, ndarray)
        ordered = sorted(items, key=lambda item: sizes[item[0]]) if mode == 'short_load' and fallback is None else items
        after = [jid for jid, _ in ordered]
        replacement = dict(ordered) if after != before else original
        data['events'].append(dict(before=before, after=after, bytes=sizes, fallback=fallback,
            host_perf_s=started, decision_s=time.perf_counter()-started))
        try:
            metadata.load_jobs = replacement
            return old(metadata)
        finally:
            metadata.load_jobs = original

    call._load_order_wrapper = True
    cw.start_kv_transfers = call

    def uninstall():
        nonlocal active
        if active:
            if cw.start_kv_transfers is not call:
                raise RuntimeError('Uninstall LOAD ordering before outer wrappers')
            if own:
                cw.start_kv_transfers = old
            else:
                del cw.start_kv_transfers
            active = False
        return data

    return data, uninstall
