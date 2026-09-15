"""Prepared partial-map transport, CPU-qualified only; GPU performance unrun.

One bank per layer, one slot per possible expert group, allocated before warmup.
Caller enqueues the consuming kernel on this same stream before reusing a slot.
Both control and treatment must hold identical bank allocations. This changes
only group-map upload; expert loads, LRU, group order and kernel inputs stay put.
"""
import inspect
import math
from types import MethodType


class PartialMapBank:
    def __init__(self, torch, num_experts, max_groups, slot_capacity, device):
        if min(num_experts, max_groups, slot_capacity) < 1:
            raise ValueError('positive dimensions required')
        self.torch, self.device = torch, device
        self.num_experts, self.slot_capacity = num_experts, slot_capacity
        self.stream_id = None
        self.entries = []
        for _ in range(max_groups):
            host = torch.empty(num_experts, dtype=torch.int32, pin_memory=True)
            gpu = torch.empty(num_experts, dtype=torch.int32, device=device)
            self.entries.append(dict(host=host, view=host.numpy(), gpu=gpu,
                                     done=torch.cuda.Event(), pending=False))

    def materialize(self, mapping, group_index):
        if (len(mapping) != self.num_experts or
                any(type(x) is not int or not -1 <= x < self.slot_capacity for x in mapping)):
            raise ValueError('invalid global-to-slot map')
        if not 0 <= group_index < len(self.entries):
            raise ValueError('group exceeds preallocated bank')
        stream = self.torch.cuda.current_stream(self.device)
        if self.stream_id not in (None, stream.cuda_stream):
            raise RuntimeError('bank requires a fixed consumer stream')
        self.stream_id = stream.cuda_stream
        e = self.entries[group_index]
        if e['pending']:
            # Host buffer cannot change until its last DMA has consumed it.
            e['done'].synchronize()
        e['view'][:] = mapping
        e['gpu'].copy_(e['host'], non_blocking=True)
        e['done'].record(stream)
        e['pending'] = True
        return e['gpu']

    @property
    def extra_bytes(self):
        n = self.num_experts * 4 * len(self.entries)
        return dict(host_pinned=n, gpu=n)


def install(runtime, banks, *, enabled=False):
    """Install into the inspected expert-major implementation, no source file edits.

The runtime must be drained when mode changes. Qualified GPU use remains pending.
"""
    original = runtime.apply.__func__
    source = inspect.getsource(original)
    anchor = 'group_map = torch.tensor(mapping, dtype=torch.int32, device=x.device)'
    if source.count(anchor) != 1 or not source.startswith('def _apply('):
        raise RuntimeError('unsupported expert-major apply source')
    if set(banks) != set(runtime.layers):
        raise ValueError('allocate one bank for every registered layer')
    replacement = ('group_map = (self.partial_map_banks[id(layer)].materialize(mapping, '
                   'len(record["groups"])-1) if self.partial_map_async else '
                   'torch.tensor(mapping, dtype=torch.int32, device=x.device))')
    ns = dict(original.__globals__)
    exec(compile(source.replace(anchor, replacement), '<partial-map-transport>', 'exec'), ns)
    runtime.partial_map_banks = banks
    runtime.partial_map_async = enabled
    runtime.apply = MethodType(ns['_apply'], runtime)
    return original


def submission_schedule(*, pending_ms, map_dma_ms, host_launch_ms, extra_stage_ms):
    """Two-lane precedence model, not measurements or a full request simulator.

Origin follows common map construction; pending work is already in this stream.
Async staging delays map enqueue by extra_stage_ms; copy then kernel are ordered.
Both variants execute the same kernel. Its common duration cancels below.
"""
    w, m, h, c = pending_ms, map_dma_ms, host_launch_ms, extra_stage_ms
    if any(not math.isfinite(v) or v < 0 for v in (w, m, h, c)):
        raise ValueError('finite nonnegative costs required')
    blocking = w+m+h
    asynchronous = max(max(w, c)+m, c+h)
    gain = blocking-asynchronous
    return dict(blocking_kernel_start_ms=blocking,
                asynchronous_kernel_start_ms=asynchronous, saving_ms=gain,
                choice='async' if gain > 1e-12 else ('blocking' if gain < -1e-12 else 'tie'),
                scope='Analytical supplied costs; no calibrated recoverable time claim')
