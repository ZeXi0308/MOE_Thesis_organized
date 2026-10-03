"""Known cache/staging separation: private20 x16 plus a full 64-expert stage.

Copy hit_stage, h2d, then writeback; compute with pool[320:384],
global_num_experts=64 and expert_map=None. Private writeback leaves staging
intact. Every current active expert is copied on every call; staging is no cache.
"""
from analyze_layer_budget import LRU
from wisp_expert_groups import partition_experts

PRIVATE_CAP, LAYERS, STAGE_BASE, POOL_CAP = 20, 16, 320, 384


def plan_full_stage(active, metadata, layer_index):
    """Return a pure current-call plan and the exact ordinary cap20 LRU end state."""
    def require(condition, message):
        if not condition:
            raise ValueError(message)

    require(type(layer_index) is int and 0 <= layer_index < LAYERS, "invalid layer")
    active = list(active)
    require(all(type(e) is int and 0 <= e < 64 for e in active), "invalid active expert")
    active = set(active)
    state = LRU(PRIVATE_CAP)
    state.slots = list(metadata["slot_to_expert"])
    state.ticks = list(metadata["lru_tick"])
    state.clock = metadata["lru_clock"]
    require(len(state.slots) == len(state.ticks) == PRIVATE_CAP, "cap20 metadata required")
    require(all(type(e) is int and -1 <= e < 64 for e in state.slots), "invalid slot expert")
    require(type(state.clock) is int and state.clock >= 0, "invalid LRU clock")
    require(all(type(t) is int and 0 <= t <= state.clock for t in state.ticks), "invalid LRU tick")
    supplied = metadata["expert_to_slot"]
    string_ids = {str(i) for i in range(64)}
    require(all(type(e) is int and 0 <= e < 64 or type(e) is str and e in string_ids
                for e in supplied), "invalid mapping key")
    state.mapping = {int(e): s for e, s in supplied.items()}
    require(len(state.mapping) == len(supplied), "duplicate normalized expert key")
    require(all(type(s) is int and 0 <= s < PRIVATE_CAP for s in state.mapping.values()), "invalid mapping slot")
    expected = {e: s for s, e in enumerate(state.slots) if e != -1}
    require(len(expected) == sum(e != -1 for e in state.slots) and state.mapping == expected,
            "slot/map mismatch")
    if "expert_map_device" in metadata:
        require(list(metadata["expert_map_device"]) == state.snapshot()["expert_map_device"], "stale local map")
    entry = dict(state.mapping)
    groups = partition_experts(active, entry, PRIVATE_CAP)
    loaded, evicted = [], []
    for group in groups:
        missing, victims = state.ensure(group)
        loaded.extend(missing)
        evicted.extend(victims)
    require(sorted(e for group in groups for e in group) == sorted(active), "canonical coverage mismatch")
    require(sorted(loaded) == sorted(active - entry.keys()), "canonical miss/reload mismatch")
    require(all(state.mapping[e] == entry[e] for e in entry.keys() & state.mapping.keys()), "surviving entry moved")
    base = layer_index * PRIVATE_CAP
    hit_stage = [dict(expert=e, src=base + entry[e], dst=STAGE_BASE + e)
                  for e in sorted(active & entry.keys())]
    h2d = [dict(expert=e, dst=STAGE_BASE + e) for e in sorted(active - entry.keys())]
    writeback = [dict(expert=e, src=STAGE_BASE + e, dst=base + state.mapping[e])
                 for e in sorted(state.mapping.keys() - entry.keys())]
    require(sorted(c["expert"] for c in hit_stage + h2d) == sorted(active), "stage coverage mismatch")
    require(len({c["dst"] for c in hit_stage + h2d}) == len(active), "stage destinations alias")
    require(all(base <= c["src"] < base + PRIVATE_CAP for c in hit_stage), "stage source outside layer")
    require(all(STAGE_BASE <= c["dst"] < POOL_CAP for c in hit_stage + h2d), "stage destination outside pool")
    require(all(c["expert"] in active - entry.keys() and STAGE_BASE <= c["src"] < POOL_CAP
                and base <= c["dst"] < base + PRIVATE_CAP for c in writeback), "unsafe writeback")
    # No previous staging contents are available to this symbolic execution.
    contents = {base + s: e for e, s in entry.items()}
    for c in hit_stage:
        require(contents.get(c["src"]) == c["expert"], "stage source mismatch")
        contents[c["dst"]] = contents[c["src"]]
    for c in h2d:
        contents[c["dst"]] = c["expert"]
    require(all(contents.get(STAGE_BASE + e) == e for e in active), "active stage content mismatch")
    for c in writeback:
        require(contents.get(c["src"]) == c["expert"], "writeback source mismatch")
        contents[c["dst"]] = contents[c["src"]]
    require(all(contents.get(STAGE_BASE + e) == e for e in active), "writeback changed active stage")
    require(all(contents.get(base + s) == e for e, s in state.mapping.items()), "final cache content mismatch")
    entry_evicted = sorted(entry.keys() - state.mapping.keys())
    return dict(layer_index=layer_index, private_cap=PRIVATE_CAP, pool_cap=POOL_CAP,
                stage_slice=[STAGE_BASE, POOL_CAP], global_num_experts=64, expert_map=None,
                active_experts=sorted(active), hit_stage=hit_stage, h2d=h2d, writeback=writeback,
                copy_order=["hit_stage", "h2d", "writeback"], kernel_after_copies=True,
                final_state=state.snapshot(), canonical_groups=groups, canonical_group_count=len(groups),
                canonical_miss=len(loaded), canonical_evict=len(evicted), canonical_loaded_experts=loaded,
                canonical_evicted_experts=evicted, entry_evicted_experts=entry_evicted, entry_evict=len(entry_evicted))
