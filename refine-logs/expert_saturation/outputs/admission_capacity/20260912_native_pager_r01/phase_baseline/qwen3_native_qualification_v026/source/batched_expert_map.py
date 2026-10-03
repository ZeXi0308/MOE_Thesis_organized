"""Exact-source WiSP map-copy diagnostic; expert loads and LRU stay upstream.

set_mode() before a probe; prepare() after LLM creation; stats(reset=True)
after warmup; stats()/validate() after the measured wall is captured.
"""
import hashlib
import inspect
import os
from pathlib import Path
import textwrap

_SOURCE_SHA256 = "5572d4f05593a5a9fc4adaa14421cc596886a435b6f5f51f476a1dae6e521857"
_MODE, _MODULE, _ORIGINAL, _BATCHED = "original", None, None, None
_ENTRIES = {}
_COUNTERS = ("ensure_calls", "misses", "evictions", "scalar_device_assignments",
             "full_map_copy_calls", "full_map_copy_payload_bytes")


def _require(ok, message):
    if not ok:
        raise RuntimeError(message)


def _entry(state):
    import torch
    if state not in _ENTRIES:
        host = torch.empty(state.num_experts, dtype=torch.int32, pin_memory=True)
        view = host.numpy()
        view.fill(-1)
        _ENTRIES[state] = dict(host=host, view=view, event=torch.cuda.Event(), pending=False,
                               counts=dict.fromkeys(_COUNTERS, 0))
    return _ENTRIES[state]


def _flush_map(state):
    import torch
    entry = _entry(state)
    if entry["pending"]:
        entry["event"].synchronize()  # Previous DMA must finish before CPU overwrites.
    entry["view"].fill(-1)
    for expert, slot in state.expert_to_slot.items():
        entry["view"][expert] = slot
    state.expert_map_device.copy_(entry["host"], non_blocking=True)
    entry["event"].record(torch.cuda.current_stream(state.expert_map_device.device))
    entry["pending"] = True
    entry["counts"]["full_map_copy_calls"] += 1
    entry["counts"]["full_map_copy_payload_bytes"] += state.num_experts * 4


def _dispatch(state, needed_experts):
    entry = _entry(state)  # Both modes allocate the same buffers/events.
    before = state.stats_miss, state.stats_evict
    result = (_ORIGINAL if _MODE == "original" else _BATCHED)(state, needed_experts)
    miss, evict = state.stats_miss - before[0], state.stats_evict - before[1]
    counts = entry["counts"]
    counts["ensure_calls"] += 1
    counts["misses"] += miss
    counts["evictions"] += evict
    if _MODE == "original":
        counts["scalar_device_assignments"] += miss + evict
    return result


def set_mode(mode):
    global _MODE, _MODULE, _ORIGINAL, _BATCHED
    _require(mode in ("original", "batched"), "unsupported map mode")
    _require(all(os.environ.get(k, "0") == "0" for k in ("WISP_PREFETCH", "WISP_DYNAMIC")),
             "map batching requires prefetch/dynamic disabled")
    if _ORIGINAL is None:
        import wisp.integrations.vllm.fused_moe as module
        _require(hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() == _SOURCE_SHA256,
                 "WiSP source differs from frozen 86f697")
        original = module.WispMoEState.ensure_resident
        source = textwrap.dedent(inspect.getsource(original))
        for old, new in (("def ensure_resident(", "def _ensure_resident_batched("),
                         ("            self.expert_map_device[evicted] = -1\n", ""),
                         ("        self.expert_map_device[expert] = slot\n", "")):
            _require(source.count(old) == 1, "ensure_resident source anchor changed: " + old)
            source = source.replace(old, new, 1)
        source = source.rstrip() + "\n    _flush_map(self)\n"
        namespace = dict(original.__globals__, _flush_map=_flush_map)
        exec(compile(source, "<batched_expert_map.ensure_resident>", "exec"), namespace)
        _MODULE, _ORIGINAL, _BATCHED = module, original, namespace["_ensure_resident_batched"]
        module.WispMoEState.ensure_resident = _dispatch
    _MODE = mode


def prepare():
    """Allocate every registered layer's identical small staging buffer outside measurement."""
    set_mode(_MODE)
    for state in _MODULE._LAYER_STATES:
        _require(state.mode == "paged", "requires paged WiSP state")
        _entry(state)
    return stats()


def stats(reset=False):
    layers = []
    for state, entry in _ENTRIES.items():
        layers.append(dict(layer_idx=state.layer_idx, counts=dict(entry["counts"]),
            pinned_map_bytes=entry["host"].numel() * entry["host"].element_size(),
            host_ptr=entry["host"].data_ptr(), device_map_ptr=state.expert_map_device.data_ptr(),
            event_id=id(entry["event"])))
        if reset:
            entry["counts"].update(dict.fromkeys(_COUNTERS, 0))
    return dict(mode=_MODE, layers=layers,
                totals={k: sum(x["counts"][k] for x in layers) for k in _COUNTERS},
                pinned_map_bytes=sum(x["pinned_map_bytes"] for x in layers),
                accounting="Source-operation counters, not profiler-measured DMA events")


def validate():
    """Post-wall map invariant check; never copies expert weights for validation."""
    import torch
    torch.cuda.synchronize()
    for state in _ENTRIES:
        expected = [-1] * state.num_experts
        for slot, expert in enumerate(state.slot_to_expert):
            if expert != -1:
                _require(state.expert_to_slot.get(expert) == slot, "host map inverse mismatch")
                expected[expert] = slot
        _require(sum(x >= 0 for x in expected) == len(state.expert_to_slot), "extra host map entry")
        _require(state.expert_map_device.cpu().tolist() == expected, "device map differs from host map")
    return dict(status="PASS", layers=len(_ENTRIES), checked="host inverse and full device map")


def gpu_selftest():
    """Small real-CUDA BF16 state/copy check; call before full-request episodes."""
    import torch
    previous = _MODE
    set_mode("original")
    w13 = torch.arange(6 * 4 * 4, dtype=torch.float32).reshape(6, 4, 4).to(torch.bfloat16).pin_memory()
    w2 = torch.arange(6 * 4 * 2, dtype=torch.float32).reshape(6, 4, 2).to(torch.bfloat16).pin_memory()
    states = [_MODULE.WispMoEState(w13, w2, 3, "paged", torch.device("cuda"), -i-1) for i in range(2)]
    cases = ([0, 1, 1, 2], [0, 2], [2, 3, 4], [1, 4, 5], [1, 4, 5])
    try:
        for needed in cases:
            ids = torch.tensor(needed, dtype=torch.int32, device="cuda")
            for mode, state in zip(("original", "batched"), states):
                set_mode(mode)
                state.ensure_resident(ids)
            torch.cuda.synchronize()
            left, right = states
            for key in ("slot_to_expert", "expert_to_slot", "lru_tick", "lru_clock"):
                _require(getattr(left, key) == getattr(right, key), "toy state differs: " + key)
            for key in left.__slots__:
                if key.startswith("stats_"):
                    _require(getattr(left, key) == getattr(right, key), "toy counter differs: " + key)
            for key in ("expert_map_device", "scratch_w13", "scratch_w2"):
                _require(torch.equal(getattr(left, key), getattr(right, key)), "toy tensor differs: " + key)
        return dict(status="PASS", cases=len(cases), dtype="bfloat16",
                    checked="all-hit, fill, eviction; maps, full scratch, LRU and upstream counters")
    finally:
        torch.cuda.synchronize()
        for state in states:
            _ENTRIES.pop(state, None)
        set_mode(previous)
