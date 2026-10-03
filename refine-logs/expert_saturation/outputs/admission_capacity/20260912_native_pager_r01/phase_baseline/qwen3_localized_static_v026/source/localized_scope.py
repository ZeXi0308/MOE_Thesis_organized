"""Drained qualification-to-performance handoff on one installed pager object."""
import hashlib
import json
from pathlib import Path
import time

QUALIFICATION_SHA = "03f81c893459d270e78c0be8e4717f656fa0f165c32c35184c9d7c96e91ede20"
PATTERN_SHA = "42f94ff25c2d3ed76b7bd76c9e0d977f79c0339fe32900e8f0331ce9bdddc6ce"


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def read_qualification(path):
    path = Path(path)
    require(hashlib.sha256((path / "workload.json").read_bytes()).hexdigest() == QUALIFICATION_SHA, "original qualification workload changed")
    require(hashlib.sha256((path / "original_layer47_pattern.json").read_bytes()).hexdigest() == PATTERN_SHA, "original diagnostic pattern changed")
    workload = json.loads((path / "workload.json").read_text())
    # Original qualifier constructed its clock with multiplication after reading prepared input.
    workload = dict(workload, arrival_traces_s={"steady": [i * .05 for i in range(4)]})
    return workload, json.loads((path / "original_layer47_pattern.json").read_text())


def pattern_comparison(raw, record, diagnostic, original):
    aliases = raw["internal_to_source"]
    rows = [dict(request_id=aliases[r["internal_request_id"]], computed_position=r["computed_position"], prompt_tokens=r["prompt_tokens"])
            for r in record["context"]["rows"]]
    current = dict(rows=rows, row_topk_experts=record["row_topk_experts"],
                   required_experts=[g["required_experts"] for g in record["groups"]],
                   diagnostic={k: diagnostic[k] for k in original["diagnostic"]})
    return dict(equal_fields={k: current[k] == original[k] for k in current}, current=current,
                original_hidden_state_available=False, old_hidden_bit_identity="UNAVAILABLE",
                scope="Recorded row/topk/group and original tolerance diagnostic comparison; old hidden tensors were not retained.")


def qualification_gate(runtime, probe, raw, original):
    values = runtime.validation_results; names = {e["layer_name"] for e in runtime.layers.values()}
    local = probe.gate()
    finite = len(values) == len(names) == 48 and {v["layer_name"] for v in values} == names and all(v["allfinite"] for v in values)
    rows_ok = (raw["status"] == "COMPLETE" and len(raw["requests"]) == 4
               and all(r["status"] == "completed" and len(r["output_token_ids"]) == 8 for r in raw["requests"]))
    target = [v for v in values if v["layer_name"] == "model.layers.47.mlp.experts"]
    records = [r for r in runtime.records if target and r["call_id"] == target[0]["call_id"]]
    target_ok = len(target) == len(records) == 1 and target[0]["rows"] == records[0]["rows"] == 32
    comparison = pattern_comparison(raw, records[0], target[0], original) if target_ok else None
    passed = finite and rows_ok and target_ok and local["status"] == "PASS"
    return dict(status="PASS" if passed else "STOP_QUALIFICATION_RESIDUAL", performance_eligible=passed,
        all_48_layers_finite=finite, four_requests_complete=rows_ok, target_32_rows_reached=target_ok,
        local_attribution=local, original_pattern_comparison=comparison,
        original_allclose_passed=sum(v["allclose"] for v in values), original_diagnostics=values,
        performance_episodes_completed=0,
        scope="Only local execution attribution: every partial and their BF16 sum must match same-partition full-weight reference bitwise. Original .01/.01 diagnostics remain unchanged. No FP32/reverse improvement, task-quality or universal-shape guarantee is required or claimed.")


def allocation_identity(runtime, runner):
    def describe(name, tensor):
        return dict(name=name, device=str(tensor.device), pointer=tensor.data_ptr(),
                    storage_pointer=tensor.untyped_storage().data_ptr(), storage_bytes=tensor.untyped_storage().nbytes(),
                    shape=list(tensor.shape), dtype=str(tensor.dtype))
    rows = [describe("parameter:" + name, t) for name, t in runner.model.named_parameters()]
    for entry in runtime.layers.values():
        state = entry["state"]
        rows.extend(describe(entry["layer_name"] + ":" + name, getattr(state, name))
                    for name in ("cpu_w13", "cpu_w2", "scratch_w13", "scratch_w2", "expert_map_device"))
    rows.extend(describe(f"kv:{i}", t) for i, t in enumerate(runner.kv_caches))
    return rows


def reset_artifact_scope(runtime, engine, runner, outdir):
    """No hook installation, model reload, tensor allocation or cache-state mutation."""
    started = time.perf_counter_ns(); scheduler = engine.engine_core.engine_core.scheduler
    require(not engine.has_unfinished_requests() and not scheduler.requests and not scheduler.running and not scheduler.waiting, "scope handoff requires a drained engine")
    require(runtime.summary is not None and not runtime.records and not runtime.events and runtime._flush_error is None, "qualification must finalize successfully before scope reset")
    base = {"cap", "outdir", "torch", "upstream", "kernel", "layers", "records", "events", "context", "measurement", "summary",
            "validation_enabled", "validation_results", "flushed_calls", "flushes", "_flush_error", "_measurement_calls", "_failed_calls", "_totals"}
    extra = {"apply", "group_retention", "group_retention_order", "retention_validated", "memory_observer_mode", "measurement_initial_cache", "numerical_localization"}
    require(set(vars(runtime)) <= base | extra, "unknown runtime state needs an explicit handoff decision")
    runtime.torch.cuda.synchronize(); before = allocation_identity(runtime, runner)
    identity = id(runtime); layers = runtime.layers; layer_ids = {k: id(v["state"]) for k, v in layers.items()}
    keep = {k: getattr(runtime, k) for k in ("cap", "torch", "upstream", "kernel", "apply", "group_retention", "group_retention_order", "memory_observer_mode")}
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=False)
    prior = dict(outdir=str(runtime.outdir), call_count=runtime.flushed_calls,
                 validation_count=len(runtime.validation_results), fields=sorted(vars(runtime)))
    for entry in layers.values():
        require(set(entry) <= {"layer", "state", "ever_loaded", "layer_name", "validated"}, "unknown layer metadata at scope handoff")
        entry.pop("validated", None); entry["ever_loaded"] = set()
    runtime.__dict__.clear()
    type(runtime).__init__(runtime, keep["cap"], outdir, keep["torch"], keep["upstream"], keep["kernel"])
    runtime.layers = layers
    for key in ("apply", "group_retention", "group_retention_order", "memory_observer_mode"):
        setattr(runtime, key, keep[key])
    runtime.retention_validated = False
    after = allocation_identity(runtime, runner)
    require(id(runtime) == identity and layer_ids == {k: id(v["state"]) for k, v in layers.items()} and before == after, "native allocations or runtime/state identities changed")
    require(not runtime.validation_enabled and not runtime.validation_results and runtime.next_call_id == 0 and runtime.summary is None, "performance accounting not empty")
    return dict(status="PASS", start_perf_ns=started, end_perf_ns=time.perf_counter_ns(), runtime_object_id=identity,
                previous=prior, new_outdir=str(outdir), allocations_before=before, allocations_after=after,
                allocations_identical=True, engine_drained=True, model_hooks_reinstalled=False,
                scope="Accounting/validation/context reset only. Same runtime, model, layers, master/scratch/map/KV allocations. Cache is reset by the original formal-cell pre-warmup reset; no KV tensor bitwise comparison.")
