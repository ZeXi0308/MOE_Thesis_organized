"""Fixed-cap WiSP pager bridge for native vLLM 0.26 OLMoE BF16.

Call install() in the model worker before constructing the model; the upstream
WiSP plugin must remain disabled. Native routing is preserved. Both full and
partial caps use the same scratch-backed Triton wrapper. CUDA load spans enclose
ensure_resident (copies, map updates and possible host submission gaps), not
isolated memcpy duration or request waiting. This is an instrumented prototype.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import os
from pathlib import Path
import time

_runtime = None


def partition_rows(rows, cap):
    """Contiguous greedy groups; each token's complete expert set must fit."""
    if type(cap) is not int or cap < 1:
        raise ValueError("cap must be a positive integer")
    groups, current, start = [], set(), 0
    for i, row in enumerate(rows):
        experts = set(row)
        if len(experts) > cap:
            raise ValueError("one token's expert set exceeds cap")
        if len(current | experts) > cap:
            groups.append((start, i))
            current, start = set(), i
        current.update(experts)
    if rows:
        groups.append((start, len(rows)))
    return groups


def install(cap, outdir):
    """Patch one fresh worker. Returns runtime; repeated identical calls are safe."""
    global _runtime
    if type(cap) is not int or cap < 1:
        raise ValueError("cap must be a positive integer")
    outdir = Path(outdir).resolve()
    if _runtime is not None:
        if (_runtime.cap, _runtime.outdir) != (cap, outdir):
            raise RuntimeError("restart the worker to change pager configuration")
        return _runtime
    os.environ["WISP_PLUGIN_DISABLE"] = "1"
    os.environ["WISP_PREFETCH"] = "0"
    os.environ["WISP_DYNAMIC"] = "0"
    import torch
    import vllm
    from wisp.integrations.vllm import fused_moe as upstream
    from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
        UnquantizedFusedMoEMethod as Method,
    )
    from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts

    if vllm.__version__ != "0.26.0" or not torch.cuda.is_available():
        raise RuntimeError("adapter requires CUDA vLLM 0.26.0")
    if getattr(Method, "_wisp_patched", False):
        raise RuntimeError("disable upstream WiSP plugin before importing vLLM")
    expected = ["self", "layer", "x", "topk_weights", "topk_ids",
                "shared_experts", "shared_experts_input"]
    if list(inspect.signature(Method.apply).parameters) != expected:
        raise RuntimeError("native apply signature differs from inspected 0.26 source")
    if "inplace" in inspect.signature(fused_experts).parameters:
        raise RuntimeError("unexpected legacy fused_experts interface")
    outdir.mkdir(parents=True, exist_ok=True)
    if any((outdir / name).exists() for name in ("calls.jsonl", "summary.json")):
        raise FileExistsError("pager outputs already exist")
    runtime = _Runtime(cap, outdir, torch, upstream, fused_experts)
    upstream._WISP_RUNTIME_CONFIG.update(mode="paged", cap_experts_override=cap)
    Method._wisp_original_process_weights_after_loading = Method.process_weights_after_loading

    def create(self, layer, num_experts, hidden_size,
               intermediate_size_per_partition, params_dtype, **extra):
        runtime.check_config(self, layer, num_experts, params_dtype)
        return upstream._patched_create_weights(
            self, layer, num_experts, hidden_size,
            intermediate_size_per_partition, params_dtype, **extra)

    def postload(self, layer):
        upstream._patched_process_weights_after_loading(self, layer)
        state = layer._wisp_state
        if state is None or state.cap_experts != cap:
            raise RuntimeError("WiSP state did not initialize at requested cap")
        runtime.layers[id(layer)] = {
            "layer": layer, "state": state, "ever_loaded": set(),
            "layer_name": layer.layer_name,
        }

    def apply(self, layer, x, topk_weights, topk_ids,
              shared_experts, shared_experts_input):
        if shared_experts is not None or shared_experts_input is not None:
            raise RuntimeError("shared experts are outside this adapter")
        return runtime.apply(self, layer, x, topk_weights, topk_ids)

    Method.create_weights, Method.process_weights_after_loading = create, postload
    Method.apply = apply
    _runtime = runtime
    return runtime


class _Runtime:
    def __init__(self, cap, outdir, torch, upstream, kernel):
        self.cap, self.outdir = cap, outdir
        self.torch, self.upstream, self.kernel = torch, upstream, kernel
        self.layers, self.records, self.events = {}, [], []
        self.context = {"phase": "initialization", "step_id": None, "rows": None}
        self.measurement = False
        self.summary = None
        self.validation_enabled, self.validation_results = False, []
        self.flushed_calls, self.flushes, self._flush_error = 0, [], None
        self._measurement_calls, self._failed_calls = 0, 0
        self._totals = {label: dict.fromkeys(("miss", "evict", "weight_copy_bytes",
            "group_count", "load_cuda_span_ms"), 0) for label in ("all", "measurement")}

    @property
    def next_call_id(self):
        return self.flushed_calls + len(self.records)

    def check_config(self, method, layer, experts, dtype):
        from vllm.config import get_current_vllm_config
        config = get_current_vllm_config()
        parallel = config.parallel_config
        if dtype != self.torch.bfloat16 or method.moe.has_bias:
            raise RuntimeError("only bias-free BF16 experts are supported")
        if method.unquantized_backend.name != "TRITON" or method.is_monolithic:
            raise RuntimeError("set kernel_config.moe_backend='triton'")
        if not layer.top_k <= self.cap <= experts or layer.expert_map is not None:
            raise RuntimeError("invalid cap or unsupported EP expert map")
        if not config.model_config.enforce_eager:
            raise RuntimeError("the host-driven pager requires enforce_eager=True")
        if (config.scheduler_config.async_scheduling
                or config.cache_config.enable_prefix_caching
                or config.lora_config is not None):
            raise RuntimeError("disable async scheduling, prefix caching and LoRA")
        if (any(getattr(parallel, key, 1) != 1 for key in (
                "tensor_parallel_size", "pipeline_parallel_size", "data_parallel_size"))
                or parallel.enable_dbo or parallel.enable_expert_parallel
                or parallel.enable_eplb):
            raise RuntimeError("adapter supports one GPU without DBO/EP/EPLB")

    def apply(self, method, layer, x, weights, ids):
        torch = self.torch
        if self.summary is not None or self._flush_error is not None:
            raise RuntimeError("cannot execute after finalize or a failed trace flush")
        entry = self.layers[id(layer)]
        state = entry["state"]
        if x.dtype != torch.bfloat16 or x.ndim != 2 or weights.shape != ids.shape:
            raise ValueError("invalid native MoE tensor shape/dtype")
        if len(x) != len(ids):
            raise ValueError("hidden/router row identity mismatch")
        validate = (self.validation_enabled and self.measurement and len(x) > 0
                    and not entry.get("validated", False))
        fixed = tuple(t[:16].clone() for t in (x, weights, ids)) if validate else None
        record = {"call_id": self.next_call_id, "layer_name": entry["layer_name"],
                  "context": dict(self.context),
                  "measurement": self.measurement and not self.validation_enabled,
                  "validation_run": self.validation_enabled,
                  "rows": len(x), "groups": [], "status": "started"}
        self.records.append(record)
        started = time.perf_counter()
        try:
            host_start = time.perf_counter()
            ids_cpu = ids.to(device="cpu", dtype=torch.long)
            rows = ids_cpu.tolist()
            record["route_to_host_ms"] = (time.perf_counter() - host_start) * 1000
            if any(e < 0 or e >= state.num_experts for row in rows for e in row):
                raise ValueError("invalid global expert id")
            groups = partition_rows(rows, self.cap)
            result = torch.empty_like(x) if len(groups) != 1 else None
            state.stats_forward += 1
            for lo, hi in groups:
                needed = set(e for row in rows[lo:hi] for e in row)
                resident = set(state.expert_to_slot)
                missing = needed - resident
                before = (state.stats_miss, state.stats_evict)
                group = {"start": lo, "stop": hi, "unique_experts": len(needed),
                         "required_experts": sorted(needed),
                         "loaded_experts": sorted(missing),
                         "reloaded_experts": sorted(missing & entry["ever_loaded"])}
                record["groups"].append(group)
                event_pair = None
                if missing:
                    event_pair = (torch.cuda.Event(enable_timing=True),
                                  torch.cuda.Event(enable_timing=True))
                    event_pair[0].record()
                host_start = time.perf_counter()
                # Reuse WiSP's complete state transition; avoid a second D2H copy.
                try:
                    state.ensure_resident(ids_cpu[lo:hi])
                finally:
                    if event_pair:
                        event_pair[1].record()
                        self.events.append((group, event_pair))
                    group["host_ensure_ms"] = (time.perf_counter() - host_start) * 1000
                    group["miss"] = state.stats_miss - before[0]
                    group["evict"] = state.stats_evict - before[1]
                    group["evicted_experts"] = sorted(resident - set(state.expert_to_slot))
                    group["weight_copy_bytes"] = group["miss"] * self.expert_bytes(state)
                    group["load_cuda_span_ms"] = None if missing else 0.0
                entry["ever_loaded"].update(missing)
                y = self.kernel(
                    hidden_states=x[lo:hi].contiguous(), w1=state.scratch_w13,
                    w2=state.scratch_w2, topk_weights=weights[lo:hi], topk_ids=ids[lo:hi],
                    activation=layer.activation, quant_config=method.moe_quant_config,
                    apply_router_weight_on_input=layer.apply_router_weight_on_input,
                    global_num_experts=state.num_experts, expert_map=state.expert_map_device)
                if len(groups) == 1:
                    result = y
                else:
                    result[lo:hi].copy_(y)
            record["status"] = "complete"
            if validate:
                self.verify_current_call(method, layer, fixed, result[:len(fixed[0])], record)
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

    @staticmethod
    def expert_bytes(state):
        return sum(t[0].numel() * t.element_size() for t in (state.cpu_w13, state.cpu_w2))

    def set_context(self, step_id=None, rows=None, **metadata):
        context = {**self.context, **metadata, "step_id": step_id, "rows": rows}
        self.context = json.loads(json.dumps(context, allow_nan=False))

    def begin_measurement(self, phase="measurement"):
        self.torch.cuda.synchronize()
        self.measurement_initial_cache = {entry["layer_name"]: {
            "slot_to_expert": list(entry["state"].slot_to_expert),
            "lru_tick": list(entry["state"].lru_tick),
            "lru_clock": entry["state"].lru_clock} for entry in self.layers.values()}
        self.measurement = True
        self.context = {"phase": phase, "step_id": None, "rows": None}
        # Retain warmup records and cache state; the flag separates accounting.

    def enable_validation(self):
        """Only for a separate correctness run; its requests are not cost evidence."""
        if self.measurement or self.summary is not None:
            raise RuntimeError("enable validation before begin_measurement")
        self.validation_enabled = True

    def verify_current_call(self, method, layer, fixed, actual, record):
        """Same pre-call rows/top-k, full weights, no paging or routing replay."""
        torch, state = self.torch, self.layers[id(layer)]["state"]
        x, weights, ids = fixed
        w13, w2 = (t.to(device=x.device) for t in (state.cpu_w13, state.cpu_w2))
        reference = self.kernel(
            hidden_states=x, w1=w13, w2=w2, topk_weights=weights, topk_ids=ids,
            activation=layer.activation, quant_config=method.moe_quant_config,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            global_num_experts=state.num_experts, expert_map=None)
        finite = bool(torch.isfinite(actual).all() & torch.isfinite(reference).all())
        delta, ref = actual.float() - reference.float(), reference.float()
        denominator = float(torch.linalg.vector_norm(ref)) if finite else None
        numerator = float(torch.linalg.vector_norm(delta)) if finite else None
        self.validation_results.append({
            "layer_name": layer.layer_name, "call_id": record["call_id"],
            "context": record["context"], "rows": len(x),
            "actual_group_ranges": [[g["start"], g["stop"]] for g in record["groups"]],
            "allfinite": finite,
            "maxabs": float(delta.abs().max()) if finite else None,
            "relative_l2": numerator / denominator if denominator else
                (0.0 if numerator == 0 else None),
            "reference_l2": denominator,
            "bit_equal": bool(torch.equal(actual.contiguous().view(torch.uint8),
                                          reference.contiguous().view(torch.uint8))),
            "allclose": bool(torch.allclose(actual, reference, rtol=0.01, atol=0.01)),
            "rtol": 0.01, "atol": 0.01, "renormalize": bool(layer.renormalize),
            "activation": layer.activation.value,
            "scope": "first <=16 pre-call rows; actual paged group shape vs full-weight subset; native top-k weights unchanged; diagnostic tolerances, no acceptance threshold"})
        del w13, w2, reference

    def flush_records(self, label):
        """Caller must drain the engine. Failed writes retain refs and forbid retry."""
        if self._flush_error is not None or self.summary is not None:
            raise RuntimeError("trace flush is closed after failure/finalize; no retry")
        report = dict(label=label, status="started", start_perf_ns=time.perf_counter_ns(),
            held_records_before=len(self.records), held_events_before=len(self.events),
            flushed_calls_before=self.flushed_calls, records_written=0, bytes_written=0)
        self.flushes.append(report)
        try:
            if any(r["status"] == "started" for r in self.records):
                raise RuntimeError("cannot flush an in-progress layer call")
            report["sync_start_perf_ns"] = time.perf_counter_ns()
            try:
                self.torch.cuda.synchronize()
            finally:
                report["sync_end_perf_ns"] = time.perf_counter_ns()
            for group, (start, end) in self.events:
                group["load_cuda_span_ms"] = start.elapsed_time(end)
            path = self.outdir / "calls.jsonl"
            with path.open("xb" if len(self.flushes) == 1 else "ab") as stream:
                for record in self.records:
                    record["load_cuda_span_ms"] = sum(g.get("load_cuda_span_ms", 0) or 0 for g in record["groups"])
                    line = (json.dumps(record, allow_nan=False) + "\n").encode("utf-8")
                    written = stream.write(line)
                    report["bytes_written"] += written
                    if written != len(line):
                        raise OSError("short trace write; partial file retained")
                    report["records_written"] += 1
            # Commit counts only after the entire batch and file close succeeded.
            for record in self.records:
                self._measurement_calls += bool(record["measurement"])
                self._failed_calls += record["status"] != "complete"
                for label in ("all", "measurement") if record["measurement"] else ("all",):
                    for key in self._totals[label]:
                        self._totals[label][key] += record.get(key, 0)
            self.flushed_calls += len(self.records)
            self.records.clear()
            self.events.clear()
            report["status"] = "complete"
        except BaseException as exc:
            self._flush_error = f"{type(exc).__name__}: {exc}"
            report.update(status="failed", error=self._flush_error,
                          failure_scope="partial file retained; batch references retained; no retry")
            raise
        finally:
            report.update(held_records_after=len(self.records), held_events_after=len(self.events),
                flushed_calls_after=self.flushed_calls, end_perf_ns=time.perf_counter_ns())
        return report

    def finalize(self, status="complete"):
        if self.summary is not None:
            return self.summary
        self.flush_records("finalize")
        layers = []
        for entry in self.layers.values():
            state, layer = entry["state"], entry["layer"]
            layers.append({"layer_name": entry["layer_name"], "cap": state.cap_experts,
                           "num_experts": state.num_experts,
                           "pinned_bytes": sum(t.numel() * t.element_size()
                                               for t in (state.cpu_w13, state.cpu_w2)),
                           "scratch_bytes": sum(t.numel() * t.element_size()
                                                for t in (state.scratch_w13, state.scratch_w2)),
                           "parameter_bytes": sum(t.numel() * t.element_size()
                                                  for t in (layer.w13_weight, layer.w2_weight))})
        failures = self._failed_calls
        summary = {"status": "failed" if failures else status,
                   "failed_calls": failures, "cap": self.cap, "layers": layers,
                   "all_calls": self.flushed_calls, "measurement_calls": self._measurement_calls,
                   "trace_flushes": self.flushes,
                   "measurement_started": self.measurement,
                   "measurement_initial_cache": getattr(self, "measurement_initial_cache", None),
                   "validation_run": self.validation_enabled,
                   "kernel_validation": self.validation_results,
                   "upstream_sha256": hashlib.sha256(
                       Path(self.upstream.__file__).read_bytes()).hexdigest(),
                   "load_span_scope": "ensure_resident CUDA span: copies, mapping and host submission gaps; not pure H2D or request wait",
                   "bytes_scope": "physical expert rows copied by WiSP ensure_resident; excludes load-time transient, map and token transfers",
                   "failed_copy_scope": "failed calls retain completed-row byte counts; a partial row copy on CUDA failure is not measured",
                   "initialization_scope": "pre-measurement call records retained separately; model loader H2D is not instrumented",
                   "pinned_bytes": sum(r["pinned_bytes"] for r in layers),
                   "scratch_bytes": sum(r["scratch_bytes"] for r in layers)}
        summary.update({label: dict(totals) for label, totals in self._totals.items()})
        with (self.outdir / "summary.json").open("x") as stream:
            json.dump(summary, stream, indent=2, allow_nan=False)
        self.summary = summary
        return summary


def set_context(step_id=None, rows=None, **metadata):
    return _runtime.set_context(step_id, rows, **metadata)


def begin_measurement(phase="measurement"):
    return _runtime.begin_measurement(phase)


def enable_validation():
    return _runtime.enable_validation()


def finalize(status="complete"):
    return _runtime.finalize(status)
