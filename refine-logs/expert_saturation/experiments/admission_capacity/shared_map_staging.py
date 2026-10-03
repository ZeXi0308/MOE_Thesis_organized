"""Prepare X's two map publications without changing pool or kernel geometry.

blocking: exact old publications; execution_only: retains the first barrier;
all_async: both map staging copies nonblocking on the consuming stream.
Allocate the same banks in every arm before warmup. CUDA qualification unrun.
"""
import inspect
import math
from partial_map_staging import PartialMapBank


class SharedMapPair:
    def __init__(self, torch, *, num_experts, private_cap, pool_cap, device):
        self.torch, self.device = torch, device
        self.local = PartialMapBank(torch, num_experts, 1, private_cap, device)
        self.execution = PartialMapBank(torch, pool_cap, 1, pool_cap, device)

    def publish(self, state, local_map, execution_map, *, mode):
        if mode not in ('blocking', 'execution_only', 'all_async'):
            raise ValueError('unknown publication mode')
        if execution_map is None and mode != 'blocking':
            raise ValueError('async candidate is for oneshot X, not fullstage')
        if mode == 'all_async':
            temporary = self.local.materialize(local_map, 0)
            # Preserve the old local-map H2D followed by D2D publication.
            state.expert_map_device.copy_(temporary, non_blocking=True)
        else:
            state.expert_map_device.copy_(self.torch.tensor(
                local_map, dtype=self.torch.int32, device=self.device))
        if execution_map is None:
            return None
        if mode == 'blocking':
            return self.torch.tensor(execution_map, dtype=self.torch.int32, device=self.device)
        return self.execution.materialize(execution_map, 0)


def allocate(runtime):
    """Call after shared pool initialization, outside all timed measurements."""
    pool_cap = runtime.shared_weights[0].shape[0]
    return {key: SharedMapPair(runtime.torch, num_experts=e['state'].num_experts,
                private_cap=len(e['state'].slot_to_expert), pool_cap=pool_cap,
                device=e['state'].scratch_w13.device) for key,e in runtime.layers.items()}


def _publish(runtime, layer, state, local_map, execution_map):
    return runtime.shared_map_banks[id(layer)].publish(
        state, local_map, execution_map, mode=runtime.shared_map_mode)


def install(module, runtime, banks, *, mode='blocking'):
    """Replace only the two inspected publications; ordered_dispatch stays intact."""
    if mode not in ('blocking', 'execution_only', 'all_async') or set(banks) != set(runtime.layers):
        raise ValueError('invalid mode/layer banks')
    original = module.oneshot
    source = inspect.getsource(original)
    first = "state.expert_map_device.copy_(torch.tensor(final['expert_map_device'],dtype=torch.int32,device=x.device))"
    second = "mapping=None if fullstage else torch.tensor(plan['expert_map_device'],dtype=torch.int32,device=x.device)"
    if source.count(first) != 1 or source.count(second) != 1 or not source.startswith('def oneshot('):
        raise RuntimeError('unsupported shared-pool source')
    source = source.replace(first, "mapping=_publish(runtime,layer,state,final['expert_map_device'],None if fullstage else plan['expert_map_device'])")
    source = source.replace(second, '# mapping published by the two-map transport above')
    ns = dict(original.__globals__, _publish=_publish)
    exec(compile(source, '<shared-map-transport>', 'exec'), ns)
    runtime.shared_map_mode, runtime.shared_map_banks = mode, banks
    module.oneshot = ns['oneshot']
    return original


def publication_timeline(pending_ms, stages, launch_ms):
    """Two queues with explicit preparation and barrier locations, supplied costs.

Each stage has host_prepare_ms, device_ms, blocking. Model the local map's
H2D factory and following D2D as separate stages, because only its factory
blocks. No hidden-state prediction or measured wall-clock bound is asserted.
"""
    costs = [pending_ms, launch_ms] + [s[k] for s in stages for k in ('host_prepare_ms','device_ms')]
    if any(not math.isfinite(x) or x < 0 for x in costs):
        raise ValueError('finite nonnegative individual costs required')
    host, device, trace = 0., pending_ms, []
    for stage in stages:
        host += stage['host_prepare_ms']
        copy_start = max(host, device)
        device = copy_start+stage['device_ms']
        if stage['blocking']:
            host = max(host, device)
        trace.append(dict(host_after_ms=host, device_after_ms=device,
                          copy_start_ms=copy_start, blocking=bool(stage['blocking'])))
    submitted = host+launch_ms
    return dict(kernel_submitted_ms=submitted, kernel_start_ms=max(submitted,device),
                stages=trace, scope='Analytical supplied costs; same downstream kernel assumed')
