"""Same bounded expert groups/LRU, optional one-group copy lookahead.

Only the existing 24-slot scratch is used. Events/order protect scratch readers;
all native top-k contributions remain, with the same partial accumulation order.
"""
import time
from types import MethodType


def install(runtime, group_size=12, overlap=False):
    if runtime.summary is not None or runtime.measurement:
        raise RuntimeError("install before measurement/finalization")
    if type(group_size) is not int or not 1 <= group_size <= 12 or runtime.cap != 24:
        raise ValueError("candidate requires cap24 and group_size in 1..12")
    runtime._bounded_group_size, runtime._bounded_group_overlap = group_size, bool(overlap)
    runtime.group_retention, runtime.retention_validated = "none", False
    runtime.group_retention_order, runtime.group_retention_guard = "early", "none"
    runtime.apply = MethodType(_apply, runtime)
    return runtime.apply


def _apply(self, method, layer, x, weights, ids):
    torch, started = self.torch, time.perf_counter()
    if self.summary is not None or self._flush_error is not None:
        raise RuntimeError("cannot execute after finalize/flush failure")
    entry = self.layers[id(layer)]
    state = entry["state"]
    width, overlap = self._bounded_group_size, self._bounded_group_overlap
    if state.cap_experts != 24 or width * 2 > state.cap_experts:
        raise ValueError("two disjoint groups must fit the unchanged cap24 scratch")
    if (x.dtype != torch.bfloat16 or x.ndim != 2 or ids.ndim != 2
            or weights.shape != ids.shape or len(x) != len(ids)):
        raise ValueError("invalid native MoE row tensors")
    record = dict(call_id=self.next_call_id, layer_name=entry["layer_name"],
        context=dict(self.context), execution="expert_bounded", grouping_axis="expert",
        group_width=width, overlap=overlap, measurement=self.measurement and not self.validation_enabled,
        validation_run=self.validation_enabled, rows=len(x), groups=[], status="started")
    self.records.append(record)
    try:
        begin = time.perf_counter()
        rows = ids.to(device="cpu", dtype=torch.long).tolist()
        record.update(row_topk_experts=rows, route_to_host_ms=(time.perf_counter() - begin) * 1000)
        active, resident = set(e for row in rows for e in row), set(state.expert_to_slot)
        if any(e < 0 or e >= state.num_experts for e in active):
            raise ValueError("invalid expert identity")
        order = sorted(active & resident) + sorted(active - resident)
        plan = [order[i:i + width] for i in range(0, len(order), width)]
        record.update(active_experts=sorted(active), entry_resident_experts=sorted(resident),
                      missing_experts=sorted(active - resident))
        validate = self.validation_enabled and self.measurement and len(x) > 0 and not entry.get("validated", False)
        fixed = tuple(t.clone() for t in (x, weights, ids)) if validate else None
        full_x, compute = x.contiguous(), torch.cuda.current_stream(x.device)
        state.stats_forward += 1
        if overlap:
            # Bootstrap may overwrite old-call slots only after their readers.
            state.copy_stream.wait_stream(compute)

        def load(index, prior_done=None):
            experts = plan[index]
            current = set(state.expert_to_slot)
            missing = set(experts) - current
            before = state.stats_miss, state.stats_evict
            group = dict(start=0, stop=len(x), unique_experts=len(experts),
                required_experts=list(experts), ensure_experts=list(experts),
                loaded_experts=sorted(missing), reloaded_experts=sorted(missing & entry["ever_loaded"]),
                load_stream="copy" if overlap else "compute",
                wait_for_previous_compute_group=prior_done is not None,
                load_cuda_span_ms=None if missing else 0.0)
            record["groups"].append(group)
            stream = state.copy_stream if overlap else compute
            pair = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) if missing else None
            ready = torch.cuda.Event()
            begin = time.perf_counter()
            with torch.cuda.stream(stream):
                if prior_done is not None:
                    stream.wait_event(prior_done)
                if pair:
                    pair[0].record(stream)
                try:
                    state.ensure_resident(torch.tensor(experts, dtype=torch.long, device="cpu"))
                finally:
                    if pair:
                        pair[1].record(stream)
                        self.events.append((group, pair))
                    ready.record(stream)
                    group.update(host_ensure_ms=(time.perf_counter() - begin) * 1000,
                        miss=state.stats_miss - before[0], evict=state.stats_evict - before[1],
                        evicted_experts=sorted(current - set(state.expert_to_slot)))
                    group["weight_copy_bytes"] = group["miss"] * self.expert_bytes(state)
            entry["ever_loaded"].update(missing)
            if not set(experts) <= set(state.expert_to_slot) or group["miss"] != len(missing):
                raise RuntimeError("ensure bookkeeping differs from the actual group's misses")
            return group, ready

        result, done_events = None, []
        prepared = load(0) if plan else None
        for index, experts in enumerate(plan):
            group, ready = prepared if overlap or index == 0 else load(index)
            compute.wait_event(ready)
            slots = {e: state.expert_to_slot[e] for e in experts}
            mapping = [-1] * state.num_experts
            for expert, slot in slots.items():
                mapping[expert] = slot
            begin = time.perf_counter()
            group_map = torch.tensor(mapping, dtype=torch.int32, device=x.device)
            group["host_map_ms"] = (time.perf_counter() - begin) * 1000
            if overlap and index + 1 < len(plan):
                # May evict i-1, never i. Wait only for i-1's actual completion.
                prepared = load(index + 1, done_events[-1] if done_events else None)
                if any(state.expert_to_slot.get(e) != slot or state.slot_to_expert[slot] != e
                       for e, slot in slots.items()):
                    raise RuntimeError("lookahead would overwrite current group's scratch slots")
                group["current_slots_preserved_by_lookahead"] = True
            y = self.kernel(hidden_states=full_x, w1=state.scratch_w13, w2=state.scratch_w2,
                topk_weights=weights, topk_ids=ids, activation=layer.activation,
                quant_config=method.moe_quant_config,
                apply_router_weight_on_input=layer.apply_router_weight_on_input,
                global_num_experts=state.num_experts, expert_map=group_map)
            begin = time.perf_counter()
            if result is None:
                result = y
            else:
                result.add_(y)
            group["host_sum_ms"] = (time.perf_counter() - begin) * 1000
            done = torch.cuda.Event()
            done.record(compute)
            done_events.append(done)
            del y, group_map
        if result is None:
            result = torch.empty_like(x)
        record.update(status="complete", loaded_experts=[e for g in record["groups"] for e in g["loaded_experts"]],
                      final_resident_experts=sorted(state.expert_to_slot))
        if validate:
            self.verify_current_call(method, layer, fixed, result, record)
            self.validation_results[-1].update(grouping_axis="expert", actual_expert_groups=plan,
                group_width=width, overlap=overlap,
                scope="Same-input native reference diagnostic; BF16 grouped sum need not be bitwise equal.")
            entry["validated"] = True
        return result
    except Exception as exc:
        record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        record.update(host_apply_ms=(time.perf_counter() - started) * 1000,
                      group_count=len(record["groups"]))
        for key in ("miss", "evict", "weight_copy_bytes"):
            record[key] = sum(g.get(key, 0) for g in record["groups"])
