"""Expert-major baseline for the existing BF16 vLLM 0.26 WiSP runtime.

Each expert contributes in one group. Partial native outputs are added in BF16,
so this preserves top-k contributions but does not promise bitwise equivalence.
No new expert scratch or KV allocation is made by this wrapper.
"""
import time
from types import MethodType


def partition_experts(active, entry_resident, cap):
    """Consume active entry residents first; load each other expert once."""
    if type(cap) is not int or cap < 1:
        raise ValueError("cap must be a positive integer")
    active, resident = set(active), set(entry_resident)
    if len(resident) > cap:
        raise ValueError("entry resident set exceeds cap")
    if not active:
        return []
    first, missing = sorted(active & resident), sorted(active - resident)
    fill = min(cap - len(first), len(missing))
    first += missing[:fill]
    return [first] + [missing[i:i + cap] for i in range(fill, len(missing), cap)]


def install_expert_groups(runtime):
    """Replace only apply; return the bound method so callers can switch modes."""
    if runtime.summary is not None or runtime.measurement:
        raise RuntimeError("install before measurement/finalization")
    runtime.apply = MethodType(_apply, runtime)
    return runtime.apply


def _apply(self, method, layer, x, weights, ids):
    torch = self.torch
    if self.summary is not None:
        raise RuntimeError("cannot execute after finalize")
    entry, started = self.layers[id(layer)], time.perf_counter()
    state = entry["state"]
    if (x.dtype != torch.bfloat16 or x.ndim != 2 or weights.shape != ids.shape
            or ids.ndim != 2 or len(x) != len(ids)):
        raise ValueError("invalid native MoE tensors or hidden/router row identity")
    record = dict(call_id=len(self.records), layer_name=entry["layer_name"],
        context=dict(self.context), execution="expert", grouping_axis="expert",
        measurement=self.measurement and not self.validation_enabled,
        validation_run=self.validation_enabled, rows=len(x), groups=[], status="started")
    self.records.append(record)
    try:
        validate = (self.validation_enabled and self.measurement and len(x) > 0
                    and not entry.get("validated", False))
        fixed = tuple(t.clone() for t in (x, weights, ids)) if validate else None
        host_start = time.perf_counter()
        ids_cpu = ids.to(device="cpu", dtype=torch.long)
        rows = ids_cpu.tolist()
        record["route_to_host_ms"] = (time.perf_counter() - host_start) * 1000
        active, resident = set(e for row in rows for e in row), set(state.expert_to_slot)
        if any(e < 0 or e >= state.num_experts for e in active):
            raise ValueError("invalid global expert id")
        groups = partition_experts(active, resident, self.cap)
        record.update(active_experts=sorted(active), entry_resident_experts=sorted(resident),
                      missing_experts=sorted(active - resident))
        full_x = x.contiguous()
        tensor_bytes = x.numel() * x.element_size()
        # Native 0.26 BF16 cache13 and cache2 sizes; sorting buffers are excluded.
        workspace = (len(x) * ids.shape[1] * x.element_size()
                     * (max(state.scratch_w13.shape[1], state.scratch_w2.shape[1])
                        + state.scratch_w2.shape[2]))
        record["temporary_memory"] = dict(
            input_contiguous_copy_bytes=0 if x.is_contiguous() else tensor_bytes,
            full_row_kernel_activation_bytes_derived=workspace if groups else 0,
            activation_scope="native BF16 cache13/cache2; excludes sorting/kernel internals",
            extra_retained_output_bytes=tensor_bytes if len(groups) > 1 else 0,
            partial_output_storage_bytes=0, group_map_storage_bytes=0,
            scratch_extra_bytes=0, kv_extra_bytes=0,
            allocator_allocated_before=torch.cuda.memory_allocated(x.device))
        state.stats_forward += 1
        result = torch.empty_like(x) if not groups else None
        for experts in groups:
            current = set(state.expert_to_slot)
            missing = set(experts) - current
            before = (state.stats_miss, state.stats_evict)
            group = dict(start=0, stop=len(x), unique_experts=len(experts),
                required_experts=experts, loaded_experts=sorted(missing),
                reloaded_experts=sorted(missing & entry["ever_loaded"]))
            record["groups"].append(group)
            event_pair = None
            if missing:
                event_pair = (torch.cuda.Event(enable_timing=True),
                              torch.cuda.Event(enable_timing=True))
                event_pair[0].record()
            host_start = time.perf_counter()
            try:
                state.ensure_resident(torch.tensor(experts, dtype=torch.long, device="cpu"))
            finally:
                if event_pair:
                    event_pair[1].record()
                    self.events.append((group, event_pair))
                group["host_ensure_ms"] = (time.perf_counter() - host_start) * 1000
                group["miss"] = state.stats_miss - before[0]
                group["evict"] = state.stats_evict - before[1]
                group["evicted_experts"] = sorted(current - set(state.expert_to_slot))
                group["weight_copy_bytes"] = group["miss"] * self.expert_bytes(state)
                group["load_cuda_span_ms"] = None if missing else 0.0
            entry["ever_loaded"].update(missing)
            host_start = time.perf_counter()
            mapping = [-1] * state.num_experts
            for expert in experts:
                mapping[expert] = state.expert_to_slot[expert]
            # A fresh map excludes other live scratch residents from this partial.
            group_map = torch.tensor(mapping, dtype=torch.int32, device=x.device)
            group["host_map_ms"] = (time.perf_counter() - host_start) * 1000
            y = self.kernel(hidden_states=full_x, w1=state.scratch_w13,
                w2=state.scratch_w2, topk_weights=weights, topk_ids=ids,
                activation=layer.activation, quant_config=method.moe_quant_config,
                apply_router_weight_on_input=layer.apply_router_weight_on_input,
                global_num_experts=state.num_experts, expert_map=group_map)
            host_start = time.perf_counter()
            if result is None:
                result = y
            else:
                result.add_(y)
            group["host_sum_ms"] = (time.perf_counter() - host_start) * 1000
            record["temporary_memory"].update(
                partial_output_storage_bytes=y.untyped_storage().nbytes(),
                group_map_storage_bytes=group_map.untyped_storage().nbytes())
            del y, group_map
        loaded = [e for group in record["groups"] for e in group["loaded_experts"]]
        if sorted(loaded) != record["missing_experts"]:
            raise RuntimeError("expert-major execution did not load entry misses exactly once")
        record.update(status="complete", loaded_experts=loaded)
        if validate:
            self.verify_current_call(method, layer, fixed, result, record)
            self.validation_results[-1].update(grouping_axis="expert",
                actual_expert_groups=groups,
                scope="all pre-call rows; disjoint expert partials summed in BF16 vs full weights; native top-k weights unchanged; diagnostic tolerances, no acceptance threshold")
            entry["validated"] = True
        return result
    except Exception as exc:
        record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        record["host_apply_ms"] = (time.perf_counter() - started) * 1000
        record["group_count"] = len(record["groups"])
        for key in ("miss", "evict", "weight_copy_bytes"):
            record[key] = sum(g.get(key, 0) for g in record["groups"])
        if "temporary_memory" in record:
            record["temporary_memory"].update(
                allocator_allocated_after=torch.cuda.memory_allocated(x.device),
                allocator_peak_since_runner_reset=torch.cuda.max_memory_allocated(x.device))
