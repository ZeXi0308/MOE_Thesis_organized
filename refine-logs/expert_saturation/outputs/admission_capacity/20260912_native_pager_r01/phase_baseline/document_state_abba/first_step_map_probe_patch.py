"""Patch the frozen probe for per-document workloads and first-action map counts."""
import hashlib
from pathlib import Path


def _require(ok, message):
    if not ok:
        raise RuntimeError(message)


def cell_workload(base, cell):
    """Resolve one declared workload and bind it to the cell's document group."""
    root = Path(base).resolve()
    path = (root / cell["workload"]).resolve()
    _require(root in path.parents and path.is_file(), "cell workload missing or outside bundle")
    data = path.read_bytes()
    return dict(doc_group=cell["doc_group"], path=path,
                sha256=hashlib.sha256(data).hexdigest())


def check_group_prestate(state, row, raw, actual, workload):
    """Use the first declared cell in each document group as its canonical state."""
    _require(raw["workload_sha256"] == workload["sha256"], "executed workload differs from cell")
    refs = state.setdefault("reference_prestate_by_docgroup", {})
    reference = refs.setdefault(workload["doc_group"], dict(cell=row["cell"],
        workload_sha256=workload["sha256"], prestate_sha256=actual))
    row.update(doc_group=workload["doc_group"], workload_sha256=workload["sha256"],
               prestate_sha256=actual, canonical_prestate_cell=reference["cell"])
    valid = (reference["workload_sha256"] == workload["sha256"]
             and reference["prestate_sha256"] == actual)
    if not valid:
        row["status"] = "INVALID_PRESTATE"
    _require(valid, "preaction state differs within document group")


def patch_first_step_map_probe(source):
    """Read host counters around the action engine call; never synchronize CUDA."""
    reset_anchor = '        result["map_initial"] = map_patch.stats(reset=True)\n'
    _require(source.count(reset_anchor) == 1, "apply patch_map_probe before this patch")
    old = '''            call["start_s"] = now()
            try:
                outputs = engine.step()
                received = now()
'''
    new = '''            try:
                first_map_before = None
                if result.get("action", {}).get("engine_call") == call["index"]:
                    counter_started = time.perf_counter()
                    first_map_before = map_patch.stats()
                    counter_before_s = time.perf_counter() - counter_started
                call["start_s"] = now()
                outputs = engine.step()
                received = now()
                if first_map_before is not None:
                    counter_started = time.perf_counter()
                    first_map_after = map_patch.stats()
                    counter_after_s = time.perf_counter() - counter_started
                    before_totals, after_totals = first_map_before["totals"], first_map_after["totals"]
                    changes = {key: after_totals[key] - before_totals[key]
                        for key in ("ensure_calls", "misses", "evictions", "full_map_copy_calls")}
                    require(all(value >= 0 for value in changes.values()), "Map counters decreased")
                    require(changes["ensure_calls"] == 16, "First action did not cover all 16 MoE layers")
                    require("first_action_map_counters" not in result, "First-action map counters repeated")
                    result["first_action_map_counters"] = dict(engine_call=call["index"],
                        before={key: before_totals[key] for key in changes},
                        after={key: after_totals[key] for key in changes}, delta=changes,
                        counter_read_s=dict(before=counter_before_s, after=counter_after_s,
                            total=counter_before_s + counter_after_s),
                        expert_tensor_payload_bytes=changes["misses"] * (12 * 1024**2),
                        accounting="misses * 12 MiB expert tensor payload; not PCIe wire bytes")
'''
    _require(source.count(old) == 1, "first action engine-step anchor changed")
    source = source.replace(old, new, 1)
    compile(source, "first_step_map_probe.py", "exec")
    return source
