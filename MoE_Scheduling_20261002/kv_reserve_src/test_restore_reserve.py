"""Small CPU allocation-model checks; these do not qualify CUDA transfers."""
import ast
from pathlib import Path
from types import SimpleNamespace as N
from restore_reserve import install


class ModelManager:
    def __init__(self, free):
        self.free, self.max_model_len, self.forwarded, self.blocks = free, 4096, [], {}
        self.block_pool = N(get_num_free_blocks=lambda: self.free)

    def get_block_ids(self, rid):
        return (self.blocks.get(rid, []),)

    def allocate_slots(self, request, num_new_tokens, num_new_computed_tokens=0, new_computed_blocks=None,
                       num_lookahead_tokens=0, num_external_computed_tokens=0, delay_cache_blocks=False,
                       num_encoder_tokens=0, full_sequence_must_fit=False, reserved_blocks=0,
                       has_scheduled_reqs=True):
        self.forwarded.append(dict(external=num_external_computed_tokens, new=num_new_tokens,
            lookahead=num_lookahead_tokens, computed=request.num_computed_tokens))
        total = request.num_computed_tokens + num_new_computed_tokens + num_external_computed_tokens
        needed = (total + num_new_tokens + num_lookahead_tokens + 15) // 16
        if needed + reserved_blocks > self.free:
            return None
        self.free -= needed
        self.blocks[request.request_id] = list(range(needed))
        return self.get_block_ids(request.request_id)


def setup(free=2, **request_changes):
    manager = ModelManager(free)
    config = N(num_workers=1, kv_group_configs=[N(tokens_per_block=16, sliding_window_size_in_chunks=None)])
    connector = type("OffloadingConnector", (), {})()
    connector.connector_scheduler = N(config=config, manager=type("CPUOffloadingManager", (), {})())
    scheduler = N(connector=connector, kv_cache_manager=manager, block_size=16, is_encoder_decoder=False,
        num_spec_tokens=0, num_lookahead_tokens=0, scheduler_config=N(async_scheduling=False), sched_step_seq=51)
    request = N(request_id="r", status=N(name="PREEMPTED"), num_preemptions=1,
        num_computed_tokens=0, num_tokens=19, spec_token_ids=[], num_in_flight_tokens=0)
    vars(request).update(request_changes)
    return scheduler, manager, request


def run():
    scheduler, manager, request = setup()
    data, remove = install(scheduler, enabled=True)
    result = manager.allocate_slots(request, 0, num_external_computed_tokens=16, delay_cache_blocks=True)
    assert result == ([0, 1],) and manager.free == 0 and data["allocated"] == 1
    assert manager.forwarded == [dict(external=16, new=0, lookahead=1, computed=0)]
    assert request.num_computed_tokens == 0 and data["actions"][0]["known_tail_tokens"] == 3
    remove()
    assert "allocate_slots" not in vars(manager)

    scheduler, manager, request = setup(free=1)
    data, remove = install(scheduler, enabled=True)
    assert manager.allocate_slots(request, 0, num_external_computed_tokens=16, delay_cache_blocks=True) is None
    assert manager.free == 1 and len(manager.forwarded) == 1 and data["denied"] == 1
    assert manager.forwarded[0]["external"] == 16 and manager.forwarded[0]["new"] == 0
    remove()  # No fallback to an unreserved prefix-only allocation.

    for changes, external in [({"status": N(name="WAITING")}, 16), ({"num_tokens": 33}, 16),
                              ({"num_tokens": 16}, 16), ({"num_tokens": 19}, 15)]:
        scheduler, manager, request = setup(free=4, **changes)
        data, remove = install(scheduler, enabled=True)
        manager.allocate_slots(request, 0, num_external_computed_tokens=external, delay_cache_blocks=True)
        assert data["eligible"] == 0 and manager.forwarded[0]["lookahead"] == 0
        assert manager.forwarded[0]["external"] == external
        remove()

    scheduler, manager, request = setup(free=1)
    data, remove = install(scheduler, enabled=False)
    assert "allocate_slots" not in vars(manager) and not data["installed"]
    assert manager.allocate_slots(request, 0, num_external_computed_tokens=16, delay_cache_blocks=True) == ([0],)
    remove()

    pinned = Path(__file__).resolve().parents[2] / "MOE_Thesis_organized/refine-logs/expert_saturation/experiments/admission_capacity/20260929_commit_recheck/liveness_pinned_sources_20260930/kv_cache_manager.py"
    source = ast.parse(pinned.read_text())
    native = next(n for n in ast.walk(source) if isinstance(n, ast.FunctionDef) and n.name == "allocate_slots")
    model = next(n for n in ast.walk(ast.parse(Path(__file__).read_text())) if isinstance(n, ast.FunctionDef) and n.name == "allocate_slots")
    assert [a.arg for a in native.args.args] == [a.arg for a in model.args.args]
    assert [ast.literal_eval(x) for x in native.args.defaults] == [ast.literal_eval(x) for x in model.args.defaults]
    print("PASS: success, capacity denial, inapplicable requests, unchanged external/new/computed counts, disabled identity and pinned API. CPU model only; no CUDA qualification.")


if __name__ == "__main__":
    run()
