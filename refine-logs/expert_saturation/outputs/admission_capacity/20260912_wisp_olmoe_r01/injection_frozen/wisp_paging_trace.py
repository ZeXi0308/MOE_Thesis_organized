"""WiSP fixed-pager diagnostics; tensor bytes, not PCIe wire transactions.

Runner: TRACE.begin_step(index); engine.step(); TRACE.end_step(error=None).
Call TRACE.export() only after the episode. Row/request mapping is unavailable.
Host read waits overlap GPU work; never add them to CUDA or request wall time.
The CUDA load section includes map updates and possible host launch gaps.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time


class PagerTrace:
    def __init__(self, cuda=None):
        self.cuda, self.step, self.current, self.group = cuda, None, None, None
        self.rows, self.events, self.caps = [], [], {}

    def begin_step(self, scheduler_step, engine_call=None):
        if self.step is not None:
            raise RuntimeError("previous step not closed")
        if os.environ.get("WISP_PREFETCH", "0").lower() not in {"0", "false", "no", ""}:
            raise ValueError("fixed-pager trace requires prefetch disabled")
        self.step = dict(scheduler_step=scheduler_step, engine_call_index=engine_call)

    def end_step(self, error=None):
        if self.current is not None:
            self.current.update(successful=False, error=str(error or "layer return hook missing"),
                                host_end_ns=time.perf_counter_ns())
        self.step = self.current = self.group = None

    def enter(self, state, physical_rows):
        if self.step is None:
            return
        if self.current is not None:
            raise RuntimeError("nested or unfinished layer")
        if self.caps.setdefault(state.layer_idx, state.cap_experts) != state.cap_experts:
            raise ValueError("expert cap changed during fixed-cap trace")
        size = sum(t[0].numel() * t.element_size() for t in (state.cpu_w13, state.cpu_w2))
        self.current = dict(self.step, layer_idx=state.layer_idx, layer_call_index=len(self.rows),
            physical_token_rows=physical_rows, request_row_mapping="UNAVAILABLE",
            cap_experts=state.cap_experts, expert_weight_bytes=size,
            resident_at_entry=sorted(state.expert_to_slot), host_start_ns=time.perf_counter_ns(),
            subgroups=[], host_read_spans=[], successful=False)
        self.rows.append(self.current)

    def host_begin(self):
        return time.perf_counter_ns() if self.current is not None else None

    def host_end(self, kind, started):
        if started is not None:
            self.current["host_read_spans"].append(dict(kind=kind, start_ns=started,
                                                     end_ns=time.perf_counter_ns()))

    def need(self, needed, missing):
        if self.current is None:
            return
        self.group = dict(subgroup_index=len(self.current["subgroups"]),
            required_experts=sorted(needed), hits=sorted(set(needed) - set(missing)),
            missing_experts=sorted(missing), loaded_experts=[], evicted_experts=[],
            load_bytes=0, load_section_cuda_ms=None if missing else 0.0)
        self.current["subgroups"].append(self.group)

    def evict(self, expert):
        if self.current is not None:
            self.group["evicted_experts"].append(expert)

    def loaded(self, expert):
        if self.current is not None:
            self.group["loaded_experts"].append(expert)
            self.group["load_bytes"] += self.current["expert_weight_bytes"]

    def load_start(self):
        if self.current is None:
            return
        if self.cuda is None:
            import torch
            self.cuda = torch.cuda
        start, end = (self.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        self.events.append([self.group, start, end, False])

    def load_end(self):
        if self.current is not None:
            self.events[-1][2].record()
            self.events[-1][3] = True

    def leave(self, state):
        if self.current is None:
            return
        row = self.current
        required = set().union(*(set(g["required_experts"]) for g in row["subgroups"]))
        actual = sum(g["load_bytes"] for g in row["subgroups"])
        lower = len(required - set(row["resident_at_entry"])) * row["expert_weight_bytes"]
        if actual < lower or any(sorted(g["loaded_experts"]) != g["missing_experts"]
                                 for g in row["subgroups"]):
            raise RuntimeError("incomplete or inconsistent demand-load accounting")
        row.update(required_union=sorted(required), resident_at_exit=sorted(state.expert_to_slot),
                   unique_lower_bound_bytes=lower, actual_load_bytes=actual,
                   extra_load_bytes=actual - lower, successful=True, host_end_ns=time.perf_counter_ns())
        self.current = self.group = None

    def export(self, resolve=True):
        if self.step is not None:
            raise RuntimeError("resolve/export after the episode, outside engine steps")
        if resolve and self.events:
            self.cuda.synchronize()
            for group, start, end, ended in self.events:
                if ended:
                    group["load_section_cuda_ms"] = start.elapsed_time(end)
        return dict(schema_version=1, request_row_mapping="UNAVAILABLE", layers=self.rows,
                    timing_resolved=resolve, incomplete_load_sections=sum(not e[3] for e in self.events),
                    accounting="Layer-batch shared tensor bytes; extra includes reloads and eviction of "
                    "entry residents. Host waits and CUDA spans overlap; do not sum.")


TRACE = PagerTrace()


def memory_snapshot(states):
    """Real WiSP storage, independently deduplicated; do not sum with allocator totals."""
    buckets = {name: {} for name in ("gpu_scratch", "gpu_map", "cpu_master")}
    pinned = True
    for state in states:
        for name, tensors in (("gpu_scratch", (state.scratch_w13, state.scratch_w2)),
                              ("gpu_map", (state.expert_map_device,)),
                              ("cpu_master", (state.cpu_w13, state.cpu_w2))):
            for tensor in tensors:
                storage = tensor.untyped_storage()
                buckets[name][(str(tensor.device), storage.data_ptr())] = int(storage.nbytes())
                if name == "cpu_master":
                    pinned = pinned and tensor.is_pinned()
    all_device = {**buckets["gpu_scratch"], **buckets["gpu_map"]}
    return dict(**{name + "_storage_bytes": sum(values.values()) for name, values in buckets.items()},
                gpu_total_storage_bytes=sum(all_device.values()), cpu_master_all_pinned=pinned,
                accounting="Scratch/map/CPU masters deduplicated by storage. GPU total is their union, "
                "not an additional allocation. CPU master includes its pinned storage, not a second copy.")


def patched_source(source):
    """Each anchor must match exactly once; this never writes the source file."""
    replacements = [
        ("from __future__ import annotations", "from __future__ import annotations\nfrom wisp_paging_trace import TRACE"),
        ("    state.stats_forward += 1\n\n    # If the caller", "    state.stats_forward += 1\n    TRACE.enter(state, int(x.shape[0]))\n\n    # If the caller"),
        ("        missing = [e for e in needed_set if e not in self.expert_to_slot]",
         "        missing = [e for e in needed_set if e not in self.expert_to_slot]\n        TRACE.need(needed_set, missing)"),
        ("                self.stats_evict += 1\n\n        # First absorb", "                self.stats_evict += 1\n                TRACE.evict(evicted)\n\n        # First absorb"),
        ("        for expert in missing:\n", "        TRACE.load_start()\n        for expert in missing:\n"),
        ("            self.stats_miss += 1\n\n        # Touch LRU", "            self.stats_miss += 1\n            TRACE.loaded(expert)\n        TRACE.load_end()\n\n        # Touch LRU"),
        ("    if zero_expert_num != 0 and zero_expert_type is not None:\n        return result, zero_expert_result",
         "    TRACE.leave(state)\n    if zero_expert_num != 0 and zero_expert_type is not None:\n        return result, zero_expert_result"),
    ]
    for kind, line in (
        ("subgroup_route_read", '        needed = torch.unique(needed_experts).to(device="cpu", dtype=torch.long).tolist()'),
        ("global_unique", "    unique_global = int(torch.unique(topk_ids).numel())"),
        ("grouping_route_read", '        topk_ids_cpu = topk_ids.to(device="cpu", dtype=torch.long)'),
        ("postforward_history_read", "        last_set = {int(e) for e in torch.unique(topk_ids).cpu().tolist()}"),
    ):
        indent = line[:len(line) - len(line.lstrip())]
        replacements.append((line, indent + "_trace_started = TRACE.host_begin()\n" + line
                             + "\n" + indent + f"TRACE.host_end({kind!r}, _trace_started)"))
    for old, new in replacements:
        if source.count(old) != 1:
            raise ValueError(f"patch anchor count {source.count(old)}; expected 1: {old[:100]!r}")
        source = source.replace(old, new, 1)
    compile(source, "patched_wisp_fused_moe.py", "exec")
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sha256", required=True, help="Expected unmodified input SHA256")
    args = parser.parse_args()
    original = args.apply.read_bytes()
    digest = hashlib.sha256(original).hexdigest()
    if digest != args.sha256.lower():
        raise ValueError(f"input SHA256 mismatch: {digest}")
    result = patched_source(original.decode()).encode()
    with args.output.open("xb") as output:  # Also refuses overwriting the input itself.
        output.write(result)
    print(json.dumps(dict(original_sha256=digest, patched_sha256=hashlib.sha256(result).hexdigest(),
                          output=str(args.output), adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())))


if __name__ == "__main__":
    main()
