"""Pure cap21 LRU-preserving plan for 16 private regions and 48 shared slots.

Execute ALL d2d copies before ANY h2d copy, then publish final_state. The
384-entry expert_map_device addresses the pool for this call only (IDs 64+
are inactive sentinels);
final_state retains local private-slot indices for the next ordinary LRU call.
"""
from analyze_layer_budget import LRU
from wisp_expert_groups import partition_experts

PRIVATE_CAP, LAYERS, SHARED_CAP = 21, 16, 48
SHARED_BASE = PRIVATE_CAP * LAYERS
POOL_CAP = SHARED_BASE + SHARED_CAP


def plan_shared_pool(active_experts, metadata, layer_index):
    """Plan from current active IDs and exact entry metadata; never mutate either."""
    def require(condition, message):
        if not condition:
            raise ValueError(message)

    require(type(layer_index) is int and 0 <= layer_index < LAYERS, "invalid layer")
    active_values = list(active_experts)
    require(all(type(e) is int and 0 <= e < 64 for e in active_values), "invalid active expert")
    active = set(active_values)
    state = LRU(PRIVATE_CAP)
    state.slots = list(metadata["slot_to_expert"])
    state.ticks = list(metadata["lru_tick"])
    state.clock = metadata["lru_clock"]
    require(len(state.slots) == len(state.ticks) == PRIVATE_CAP, "cap21 metadata required")
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
    require(sorted(e for g in groups for e in g) == sorted(active), "canonical coverage mismatch")
    require(sorted(loaded) == sorted(active - entry.keys()), "canonical miss/reload mismatch")
    final = state.snapshot()
    require(all(state.mapping[e] == entry[e] for e in entry.keys() & state.mapping.keys()),
            "surviving entry expert moved")
    shared_experts = sorted(active - state.mapping.keys())
    require(len(shared_experts) <= min(SHARED_CAP, 64 - PRIVATE_CAP), "shared pool overflow")
    shared = {e: SHARED_BASE + i for i, e in enumerate(shared_experts)}
    base = layer_index * PRIVATE_CAP
    destinations = {e: base + state.mapping[e] if e in state.mapping else shared[e] for e in active}
    d2d = [dict(expert=e, src=base + entry[e], dst=shared[e])
           for e in shared_experts if e in entry]
    h2d = [dict(expert=e, dst=destinations[e]) for e in sorted(active - entry.keys())]
    require(len(set(destinations.values())) == len(active), "active destinations alias")
    require(all(0 <= s < POOL_CAP and (base <= s < base + PRIVATE_CAP or s >= SHARED_BASE)
                for s in destinations.values()), "destination outside owned region")
    require(all(base <= c["src"] < base + PRIVATE_CAP and SHARED_BASE <= c["dst"] < POOL_CAP
                and state.slots[c["src"] - base] != c["expert"] for c in d2d), "unsafe d2d")
    require(len({c["dst"] for c in d2d + h2d}) == len(d2d) + len(h2d), "copy destinations alias")
    # Symbolic physical execution also checks surviving inactive cache contents.
    contents = {base + slot: e for e, slot in entry.items()}
    for copy in d2d:
        require(contents.get(copy["src"]) == copy["expert"], "d2d source mismatch")
        contents[copy["dst"]] = contents[copy["src"]]
    for copy in h2d:
        contents[copy["dst"]] = copy["expert"]
    require(all(contents.get(destinations[e]) == e for e in active), "active content mismatch")
    require(all(contents.get(base + s) == e for e, s in state.mapping.items()), "final cache mismatch")
    entry_evicted = sorted(entry.keys() - state.mapping.keys())
    return dict(layer_index=layer_index, private_cap=PRIVATE_CAP, pool_cap=POOL_CAP,
                expert_map_device=[destinations.get(e, -1) for e in range(POOL_CAP)],
                d2d=d2d, h2d=h2d, copy_order=["d2d", "h2d"], final_state=final,
                shared_experts=shared_experts, canonical_groups=groups,
                canonical_group_count=len(groups), canonical_miss=len(loaded),
                canonical_evict=len(evicted), canonical_loaded_experts=loaded,
                canonical_evicted_experts=evicted, entry_evicted_experts=entry_evicted,
                entry_evict=len(entry_evicted))
