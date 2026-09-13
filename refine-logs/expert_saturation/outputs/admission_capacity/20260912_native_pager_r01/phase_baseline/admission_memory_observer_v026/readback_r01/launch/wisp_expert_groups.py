"""Expert-major baseline for the existing BF16 vLLM 0.26 WiSP runtime.

Each expert contributes in one group. Partial native outputs are added in BF16,
so this preserves top-k contributions but does not promise bitwise equivalence.
No new expert scratch or KV allocation is made by this wrapper.
"""
import hashlib
from collections import Counter
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


def partition_protected_experts(active, entry_resident, protected, cap):
    """Pure plan only: execute each expert once while retaining loaded protection.

    Each step separates the kernel's execute_experts from ensure_experts, which
    also contains previously executed protected experts. The runtime maps only
    execute_experts. An empty protection set reproduces the ordinary partition.
    """
    active, resident, protected = set(active), set(entry_resident), set(protected)
    if type(cap) is not int or cap < 1 or len(resident) > cap:
        raise ValueError("positive integer cap must cover the entry resident set")
    if not protected <= active or (protected and len(protected) >= cap):
        raise ValueError("protection must be an active subset smaller than cap")
    missing = sorted(active - resident, key=lambda e: (e not in protected, e))
    execute = sorted(active & resident)
    fill = cap - len(execute)
    execute += missing[:fill]
    missing = missing[fill:]
    plan, held = [], set()
    while execute:
        held.update(protected.intersection(execute))
        plan.append(dict(execute_experts=execute, ensure_experts=sorted(held.union(execute)),
                         protected_after=sorted(held)))
        available = cap - len(held)
        execute, missing = missing[:available], missing[available:]
    return plan


def partition_protected_experts_late(active, entry_resident, protected, cap):
    """Minimum groups at entry-miss-only load cost; protected executes last.

    Let a=|protected & entry| and n=|active|. Those a slots cannot be
    evicted without extra loads, so g groups cover at most a+g*(cap-a)
    experts. For n>0 this attains max(1, ceil((n-a)/(cap-a))).
    """
    active, resident, protected = set(active), set(entry_resident), set(protected)
    if type(cap) is not int or cap < 1 or len(resident) > cap:
        raise ValueError("positive integer cap must cover the entry resident set")
    if not protected <= active or (protected and len(protected) >= cap):
        raise ValueError("protection must be an active subset smaller than cap")
    held = protected & resident
    pending = sorted((active & resident) - protected) + sorted(active - resident - protected)
    plan, width = [], cap - len(held)
    while len(pending) > cap - len(protected):
        execute, pending = pending[:width], pending[width:]
        plan.append(dict(execute_experts=execute, ensure_experts=sorted(held | set(execute)),
                         protected_after=sorted(held)))
    execute = pending + sorted(protected)
    if execute:
        plan.append(dict(execute_experts=execute, ensure_experts=sorted(execute),
                         protected_after=sorted(protected)))
    return plan


def matched_hash_protection(active, entry_resident, decode_experts, *, salt):
    """Select a deterministic control with matched protected/resident counts.

    Salt must omit arm/repeat IDs. Callers record intersection with decode_experts;
    an overlap is allowed and must not be removed by choosing a different salt.
    """
    active, resident, decode = set(active), set(entry_resident), set(decode_experts)
    if not decode <= active:
        raise ValueError("decode experts must be active")
    rank = lambda e: (hashlib.sha256(f"{salt}|{e}".encode()).digest(), e)
    chosen = []
    for pool, count in ((active & resident, len(decode & resident)),
                        (active - resident, len(decode - resident))):
        chosen.extend(sorted(pool, key=rank)[:count])
    return sorted(chosen)


def retention_plan(rows, context, resident, cap, mode, *, salt, order="early"):
    """Plan from verified current rows only; keep padding/unmapped calls ordinary."""
    if mode not in ("none", "frequency", "decode", "matched_hash"):
        raise ValueError("unknown group retention mode")
    if order not in ("early", "late"):
        raise ValueError("unknown group retention order")
    active, metadata = set(e for row in rows for e in row), context.get("rows") or []
    mapped = (context.get("row_request_order_verified") is True and context.get("valid_row_start") == 0
              and context.get("valid_row_stop") == len(rows) == len(metadata))
    mapped = mapped and all(type(r.get(k)) is int and r[k] >= (1 if k == "prompt_tokens" else 0)
                            for r in metadata for k in ("computed_position", "prompt_tokens"))
    decode_rows = ([i for i, r in enumerate(metadata) if r["computed_position"] >= r["prompt_tokens"]]
                   if mapped else [])
    decode = set(e for i in decode_rows for e in rows[i])
    mixed = mapped and 0 < len(decode_rows) < len(rows)
    eligible = mixed and 0 < len(decode) < cap
    chosen = set()
    if eligible and mode != "none":
        if mode == "decode":
            chosen = decode
        elif mode == "frequency":
            frequency = Counter(e for row in rows for e in row)
            chosen = set(sorted(active, key=lambda e: (-frequency[e], e))[:len(decode)])
        else:
            chosen = set(matched_hash_protection(active, resident, decode, salt=salt))
    planner = partition_protected_experts_late if order == "late" else partition_protected_experts
    plan = planner(active, resident, chosen, cap)
    applied = bool(chosen) and any(len(g["ensure_experts"]) > len(g["execute_experts"]) for g in plan)
    if order == "late" and chosen and not applied:
        applied = ([set(g["execute_experts"]) for g in plan]
                   != [set(g) for g in partition_experts(active, resident, cap)])
    reason = ("unmapped_or_padded_rows" if not mapped else "not_mixed" if not mixed
              else "decode_set_outside_capacity" if not eligible else "disabled" if mode == "none"
              else "no_cross_group_protection" if not applied else None)
    return plan, dict(mode=mode, order=order, eligible=eligible, applied=applied, fallback_reason=reason,
        real_decode_experts=sorted(decode) if mapped else None, chosen_protected_experts=sorted(chosen),
        protected_decode_intersection=sorted(chosen & decode), protected_entry_resident=sorted(chosen & resident),
        decode_row_count=len(decode_rows) if mapped else None,
        prefill_row_count=len(rows) - len(decode_rows) if mapped else None, hash_salt=salt if mode == "matched_hash" else None)


def install_expert_groups(runtime):
    """Replace only apply; return the bound method so callers can switch modes."""
    if runtime.summary is not None or runtime.measurement:
        raise RuntimeError("install before measurement/finalization")
    runtime.group_retention, runtime.retention_validated = "none", False
    runtime.group_retention_order = "early"
    runtime.apply = MethodType(_apply, runtime)
    return runtime.apply


from memory_observer import observer_read

def _apply(self, method, layer, x, weights, ids):
    torch = self.torch
    if self.summary is not None or self._flush_error is not None:
        raise RuntimeError("cannot execute after finalize or a failed trace flush")
    entry, started = self.layers[id(layer)], time.perf_counter()
    state = entry["state"]
    if (x.dtype != torch.bfloat16 or x.ndim != 2 or weights.shape != ids.shape
            or ids.ndim != 2 or len(x) != len(ids)):
        raise ValueError("invalid native MoE tensors or hidden/router row identity")
    record = dict(call_id=self.next_call_id, layer_name=entry["layer_name"],
        context=dict(self.context), execution="expert", grouping_axis="expert",
        order=getattr(self, "group_retention_order", "early"),
        measurement=self.measurement and not self.validation_enabled,
        validation_run=self.validation_enabled, rows=len(x), groups=[], status="started")
    self.records.append(record)
    try:
        host_start = time.perf_counter()
        ids_cpu = ids.to(device="cpu", dtype=torch.long)
        rows = ids_cpu.tolist()
        record["row_topk_experts"] = rows
        record["route_to_host_ms"] = (time.perf_counter() - host_start) * 1000
        active, resident = set(e for row in rows for e in row), set(state.expert_to_slot)
        if any(e < 0 or e >= state.num_experts for e in active):
            raise ValueError("invalid global expert id")
        mode = self.group_retention
        plan, retention = retention_plan(rows, self.context, resident, self.cap, mode,
            salt=f"seed20260912|{entry['layer_name']}|{self.context.get('step_id')}", order=record["order"])
        groups = [g["execute_experts"] for g in plan]
        protected = set(retention["chosen_protected_experts"])
        record["retention"] = retention
        validate = (self.validation_enabled and self.measurement and len(x) > 0 and
                    ((retention["applied"] and not self.retention_validated) if mode != "none"
                     else not entry.get("validated", False)))
        fixed = tuple(t.clone() for t in (x, weights, ids)) if validate else None
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
            allocator_allocated_before=observer_read(torch.cuda, x.device, False, getattr(self, 'memory_observer_mode', 'flat')))
        state.stats_forward += 1
        result = torch.empty_like(x) if not groups else None
        for planned in plan:
            experts, ensure = planned["execute_experts"], planned["ensure_experts"]
            current = set(state.expert_to_slot)
            missing = set(ensure) - current
            before = (state.stats_miss, state.stats_evict)
            group = dict(start=0, stop=len(x), unique_experts=len(experts),
                required_experts=experts, ensure_experts=ensure, protected_after=planned["protected_after"],
                loaded_experts=sorted(missing),
                reloaded_experts=sorted(missing & entry["ever_loaded"]))
            record["groups"].append(group)
            event_pair = None
            if missing:
                event_pair = (torch.cuda.Event(enable_timing=True),
                              torch.cuda.Event(enable_timing=True))
                event_pair[0].record()
            host_start = time.perf_counter()
            try:
                state.ensure_resident(torch.tensor(ensure, dtype=torch.long, device="cpu"))
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
            after_resident = set(state.expert_to_slot)
            if (not set(ensure) <= after_resident or after_resident - current != missing
                    or group["miss"] != len(missing)):
                raise RuntimeError("actual ensure loads differ from the planned entry misses")
            if not set(planned["protected_after"]) <= set(state.expert_to_slot):
                raise RuntimeError("loaded protection was evicted inside the layer")
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
        if not protected <= set(state.expert_to_slot):
            raise RuntimeError("protected experts missing at layer completion")
        retention["final_resident_experts"] = sorted(state.expert_to_slot)
        record.update(status="complete", loaded_experts=loaded)
        if validate:
            self.verify_current_call(method, layer, fixed, result, record)
            self.validation_results[-1].update(grouping_axis="expert",
                actual_expert_groups=groups, retention=retention,
                scope="all pre-call rows; disjoint expert partials summed in BF16 vs full weights; native top-k weights unchanged; diagnostic tolerances, no acceptance threshold")
            if mode == "none":
                entry["validated"] = True
            else:
                self.retention_validated = True
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
                allocator_allocated_after=observer_read(torch.cuda, x.device, False, getattr(self, 'memory_observer_mode', 'flat')),
                allocator_peak_since_runner_reset=observer_read(torch.cuda, x.device, True, getattr(self, 'memory_observer_mode', 'flat')))
