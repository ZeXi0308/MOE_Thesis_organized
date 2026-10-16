"""Process-local vLLM 0.26 speculative-length control and MoE route tracing.

Install before constructing a single-process, synchronous V1 engine.
The patch never changes site-packages. Target drafts are truncated *before*
Scheduler.schedule allocates tokens; native rejection/termination code remains
responsible for token accounting. ``proposal_budget='max'`` deliberately pays
for drafts that are subsequently discarded. ``mode='native'`` leaves vLLM's
num_speculative_tokens_per_batch_size behavior untouched.

Route diagnostics record the actual top-k IDs passed to fused modular MoE.
Fixed-size scatter histograms stay on the GPU until end_episode(); no .item(),
unique(), bincount(), or CPU route copy occurs in a forward pass. Route tracing
is a separate diagnostic run, not a performance measurement. Expert byte counts
are selected parameter storage, NOT measured HBM traffic.

CUDA Graph diagnostics use ``native_route_return=True`` together with engine
``enable_return_routed_experts=True`` set BEFORE graph capture. The native
capturer's GPU buffer contains all scheduled verification rows, including
rejected positions. We snapshot it after execute_model and before sampling or
drafting. Native vLLM still performs its own per-step D2H/slot-buffer writes;
use a separate diagnostic engine, never these timings as uninstrumented data.
``native_route_backend='cpu'`` instead reads the raw ModelRunnerOutput payload
before Scheduler.update_from_output truncates per-request routes to accepts.
``native_capture_padding=True`` additionally counts model-forward padding rows
from the GPU buffer. These go into graph_padding_routes; existing routes remain
scheduled verification rows only. Physical-row counts exclude GEMM tile padding
and are still not HBM traffic measurements.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from functools import wraps
import importlib
import os
import time
from typing import Any


@dataclass
class PluginConfig:
    mode: str = "static"  # disabled/static/dense_heuristic/adaptive/native
    max_gamma: int = 7
    static_gamma: int = 3
    disable_above_batch: int = 16
    proposal_budget: str = "max"  # max/selected; native ignores this
    policy: Any = None
    acceptance: Any = None
    expert_weight_bytes: Any = None  # scalar or {layer_name: bytes_per_expert}
    max_route_records: int = 50000
    keep_token_routes: bool = False  # diagnostic-only exact top-k row replay
    native_route_return: bool = False
    native_route_backend: str = "gpu"  # gpu buffer before sample / raw cpu runner payload
    native_capture_padding: bool = False  # trace only; GPU buffer includes graph padding
    strict: bool = True

    def __post_init__(self):
        if self.mode not in {"disabled", "static", "dense_heuristic", "adaptive", "native"}:
            raise ValueError(f"Unknown mode {self.mode!r}")
        if self.proposal_budget not in {"max", "selected"}:
            raise ValueError("proposal_budget must be max or selected")
        if self.native_route_backend not in {"gpu", "cpu"}:
            raise ValueError("native_route_backend must be gpu or cpu")
        if self.native_capture_padding and (not self.native_route_return or self.native_route_backend != "gpu"):
            raise ValueError("native_capture_padding requires native_route_return=True and native_route_backend='gpu'")
        if self.max_gamma < 0 or not 0 <= self.static_gamma <= self.max_gamma:
            raise ValueError("Require 0 <= static_gamma <= max_gamma")
        if self.mode == "adaptive" and self.policy is None:
            raise ValueError("adaptive mode requires a calibrated policy")


_ACTIVE: Runtime | None = None
_PATCHES: list[tuple[type, str, Any]] = []


def _decode_request(request: Any) -> bool:
    return (getattr(request, "num_computed_tokens", 0)
            >= getattr(request, "num_prompt_tokens", 1)
            and not getattr(request, "is_prefill_chunk", False))


def _jsonable_decision(value: Any) -> dict:
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    if isinstance(value, dict):
        return dict(value)
    return {"gamma": int(getattr(value, "gamma", value))}


class Runtime:
    def __init__(self, config: PluginConfig):
        self.config = config
        self.acceptance = config.acceptance or getattr(config.policy, "acceptance", None)
        self.label: str | None = None
        self.trace = False
        self.steps: list[dict] = []
        self.routes: list[dict] = []
        self.graph_padding_routes: list[dict] = []
        self._pending: dict[int, dict] = {}
        self._current: dict | None = None
        self._phase = "target"
        self._route_tensors: list[tuple[dict, Any]] = []
        self._token_route_tensors: list[tuple[dict, Any]] = []
        self._graph_padding_tensors: list[tuple[dict, dict, Any]] = []
        self._layer_bytes: dict[int, tuple[float | None, str]] = {}
        self._schedulers: set[int] = set()
        self._route_errors: set[str] = set()
        self._update_events: list[tuple[int, int, str]] | None = None
        self._episode_started = 0.0
        self._fallback_accepts = 0
        self._fallback_trials = 0
        self._native_layers: dict[int, Any] = {}
        self._native_binding_errors: set[str] = set()
        self._native_buffer_shape: tuple[int, ...] | None = None
        self._native_route_enabled = False
        self._engine_eager: bool | None = None

    def public_config(self) -> dict:
        result = {k: v for k, v in vars(self.config).items()
                  if k not in {"policy", "acceptance"}}
        result["policy_class"] = (type(self.config.policy).__name__
                                  if self.config.policy is not None else None)
        return result

    def attach_engine(self, engine: Any) -> dict:
        """Verify that the scheduler lives in this patched process."""
        node = getattr(engine, "llm_engine", engine)
        for _ in range(4):
            if hasattr(node, "scheduler"):
                scheduler = node.scheduler
                self._validate_scheduler(scheduler)
                self._schedulers.add(id(scheduler))
                return {"scheduler_class": type(scheduler).__name__,
                        "same_process": True,
                        "native_num_spec_tokens": scheduler.num_spec_tokens,
                        "native_dynamic_schedule": scheduler.dynamic_sd_lookup is not None,
                        "native_route_return_enabled": self._native_route_enabled,
                        "native_layer_mapping": self.native_layer_mapping()}
            node = getattr(node, "engine_core", None)
            if node is None:
                break
        raise RuntimeError("Cannot find in-process scheduler; set VLLM_ENABLE_V1_MULTIPROCESSING=0 before importing vLLM")

    def _validate_scheduler(self, scheduler: Any):
        if id(scheduler) in self._schedulers:
            return
        if getattr(scheduler.scheduler_config, "async_scheduling", False):
            raise RuntimeError("This patch requires async_scheduling=False")
        if getattr(scheduler, "use_v2_model_runner", False):
            raise RuntimeError("This patch targets vLLM V1 GPUModelRunner, not V2")
        model_config = getattr(getattr(scheduler, "vllm_config", None), "model_config", None)
        if model_config is not None:
            self._engine_eager = bool(model_config.enforce_eager)
            self._native_route_enabled = bool(getattr(model_config, "enable_return_routed_experts", False))
        if self.config.native_route_return and not self._native_route_enabled:
            raise RuntimeError("native_route_return requires EngineArgs(enable_return_routed_experts=True) before engine/graph creation")
        if (self.config.strict and model_config is not None and not model_config.enforce_eager
                and not self.config.native_route_return):
            raise RuntimeError("Use eager hooks or native_route_return for graph-compatible trace; graph performance-only runs can set strict=False")
        parallel = getattr(scheduler, "parallel_config", None)
        if self.config.strict and parallel is not None and (
                parallel.tensor_parallel_size != 1 or parallel.pipeline_parallel_size != 1):
            raise RuntimeError("Route storage byte accounting currently requires a single GPU")
        if self.config.native_route_return and parallel is not None and (
                getattr(parallel, "tensor_parallel_size", 1) != 1
                or getattr(parallel, "pipeline_parallel_size", 1) != 1
                or getattr(parallel, "data_parallel_size", 1) != 1
                or getattr(parallel, "enable_eplb", False)):
            raise RuntimeError("Native logical-ID route accounting currently requires single-GPU, single-DP and EPLB disabled")
        if self.config.native_capture_padding and parallel is not None and getattr(parallel, "enable_dbo", False):
            raise RuntimeError("Native graph padding accounting currently requires DBO disabled")
        if self.config.mode != "native" and scheduler.dynamic_sd_lookup is not None:
            raise RuntimeError("Use mode='native' with num_speculative_tokens_per_batch_size; avoid two simultaneous controllers")
        if self.config.mode != "native" and scheduler.num_spec_tokens < self.config.max_gamma:
            raise RuntimeError("Engine speculative maximum is below PluginConfig.max_gamma")

    def start_episode(self, label: str, trace: bool = False, reset_acceptance: bool = False):
        if self.label is not None:
            raise RuntimeError("end_episode must precede the next start_episode")
        if trace and self._engine_eager is False and not self.config.native_route_return:
            raise RuntimeError("Graph replay bypasses Python MoE hooks; enable native_route_return for trace diagnostics")
        self.label = label
        self.trace = bool(trace)
        self.steps = []
        self.routes = []
        self.graph_padding_routes = []
        self._pending = {}
        self._current = None
        self._route_tensors = []
        self._token_route_tensors = []
        self._graph_padding_tensors = []
        self._route_errors = set(self._native_binding_errors) if self.config.native_route_return else set()
        self._episode_started = time.perf_counter()
        if reset_acceptance:
            if self.acceptance is None or not hasattr(self.acceptance, "reset"):
                raise ValueError("Supply a new AcceptanceEWMA to reset the estimate")
            self.acceptance.reset()

    def choose_gamma(self, batch: int) -> tuple[int, dict]:
        cfg = self.config
        if cfg.mode == "disabled":
            gamma = 0
        elif cfg.mode == "dense_heuristic":
            gamma = cfg.static_gamma if batch <= cfg.disable_above_batch else 0
        elif cfg.mode == "adaptive" and batch > 0:
            try:
                decision = _jsonable_decision(cfg.policy.choose(
                    batch, draft_gamma=cfg.max_gamma if cfg.proposal_budget == "max" else None))
            except (KeyError, ValueError) as exc:
                # Only an unavailable calibrated action is a deployment fallback.
                # Shape/configuration and malformed-policy errors stay visible.
                reason = str(exc).lower()
                if "no calibrated candidate" not in reason:
                    raise
                return 0, {"gamma": 0, "mode": "adaptive",
                           "fallback": "uncovered_calibration", "reason": str(exc)}
            gamma = int(decision["gamma"])
            if not 0 <= gamma <= cfg.max_gamma:
                raise ValueError(f"Policy chose gamma={gamma} outside [0,{cfg.max_gamma}]")
            return gamma, decision
        else:
            gamma = cfg.static_gamma if cfg.mode == "static" else cfg.max_gamma
        return gamma, {"gamma": gamma, "mode": cfg.mode}

    def before_schedule(self, scheduler: Any) -> dict:
        self._validate_scheduler(scheduler)
        self._schedulers.add(id(scheduler))
        begin = time.perf_counter_ns()
        requests = list(scheduler.running)
        batch = sum(_decode_request(req) for req in requests)
        gamma, decision = self.choose_gamma(batch)
        proposed = {req.request_id: len(req.spec_token_ids) for req in requests}
        truncated: dict[str, int] = {}
        if self.config.mode != "native":
            for req in requests:
                old_len = len(req.spec_token_ids)
                if old_len > gamma:
                    req.spec_token_ids = req.spec_token_ids[:gamma]
                    truncated[req.request_id] = old_len - gamma
        choose_ns = time.perf_counter_ns() - begin
        return {"episode": self.label, "step": len(self.steps),
                "control_batch": batch, "gamma": gamma, "decision": decision,
                "control_ns": choose_ns,
                "decode_request_ids_before": [req.request_id for req in requests if _decode_request(req)],
                "decode_before": {rid: _decode_request(req)
                                  for rid, req in scheduler.requests.items()},
                "proposed_before_by_request": proposed,
                "drafts_available_before": sum(proposed.values()),
                "truncated_by_controller": sum(truncated.values()),
                "truncated_by_request": truncated,
                "accepted_by_request": {}, "accepted_drafts": 0,
                "observed_drafts": 0, "acceptance_update_ns": 0}

    def after_schedule(self, scheduler: Any, output: Any, record: dict):
        if self.config.mode != "native":
            output.num_spec_tokens_to_schedule = (self.config.max_gamma
                if self.config.proposal_budget == "max" else record["gamma"])
        token_counts = dict(output.num_scheduled_tokens)
        decode_before = record.pop("decode_before")
        decode = [rid for rid in token_counts if decode_before.get(rid, False)]
        scheduled_drafts = {rid: len(ids) for rid, ids in
                            output.scheduled_spec_decode_tokens.items()}
        if self.config.mode == "native":
            # A native schedule selects next-step production, so report the
            # current actual maximum instead of pretending max_gamma was chosen.
            record["gamma"] = max(scheduled_drafts.values(), default=0)
            record["gamma_semantics"] = "maximum_current_scheduled_draft_length"
        else:
            record["gamma_semantics"] = "current_controller_cap"
        record.update({"scheduled_requests": len(token_counts),
                       "decode_batch": len(decode),
                       "scheduled_decode_request_ids": decode,
                       "prefill_batch": len(token_counts) - len(decode),
                       "phase": ("mixed" if decode and len(decode) != len(token_counts)
                                 else "decode" if decode else "prefill"),
                       "scheduled_tokens": sum(token_counts.values()),
                       "scheduled_tokens_by_request": token_counts,
                       "scheduled_drafts_by_request": scheduled_drafts,
                       "scheduled_drafts": sum(scheduled_drafts.values()),
                       "control_vs_actual_batch_delta": record["control_batch"] - len(decode),
                       "proposal_budget_next": int(output.num_spec_tokens_to_schedule)})
        if self.label is not None:
            self.steps.append(record)
            self._pending[id(output)] = record
        self._current = record if self.label is not None else None

    def observe_acceptance(self, draft: int, accepted: int, request_id: str):
        if not 0 <= accepted <= draft:
            raise RuntimeError(f"Invalid native acceptance {accepted}/{draft}")
        if self._update_events is not None and draft > 0:
            self._update_events.append((accepted, draft, request_id))

    def finish_update(self, output: Any, elapsed_ns: int, censored: set[str] | None = None):
        events = self._update_events or []
        self._update_events = None
        update_begin = time.perf_counter_ns()
        record = self._pending.pop(id(output), None)
        eligible = [x for x in events if x[2] not in (censored or set())]
        if eligible and self.acceptance is not None:
            self.acceptance.update([x[0] for x in eligible], [x[1] for x in eligible])
        availability_updated = False
        if (record is not None and self.acceptance is not None
                and hasattr(self.acceptance, "update_availability")
                and self.config.proposal_budget == "max" and self.config.mode != "native"):
            available = [record["proposed_before_by_request"].get(rid, 0)
                         for rid in record["scheduled_decode_request_ids"]
                         if rid not in (censored or set())]
            if available:
                self.acceptance.update_availability(available, self.config.max_gamma)
                availability_updated = True
        self._fallback_accepts += sum(x[0] for x in eligible)
        self._fallback_trials += sum(min(x[0] + 1, x[1]) for x in eligible)
        if record is not None:
            record.update({"accepted_by_request": {x[2]: x[0] for x in events},
                           "observed_drafts_by_request": {x[2]: x[1] for x in events},
                           "accepted_drafts": sum(x[0] for x in events),
                           "observed_drafts": sum(x[1] for x in events),
                           "ewma_eligible_requests": len(eligible),
                           "ewma_censored_requests": sorted(censored or set()),
                           "availability_updated": availability_updated,
                           "scheduler_update_ns": elapsed_ns,
                           "acceptance_update_ns": time.perf_counter_ns() - update_begin,
                           "alpha_after": self._alpha()})

    def _alpha(self):
        if self.acceptance is not None:
            for name in ("alpha", "value", "mean"):
                value = getattr(self.acceptance, name, None)
                if value is not None and not callable(value):
                    return float(value)
        return (self._fallback_accepts / self._fallback_trials
                if self._fallback_trials else None)

    def _weight_bytes(self, layer: Any) -> tuple[float | None, str]:
        if id(layer) in self._layer_bytes:
            return self._layer_bytes[id(layer)]
        explicit = self.config.expert_weight_bytes
        if isinstance(explicit, dict):
            explicit = explicit.get(layer.layer_name)
        if explicit is not None:
            value = float(explicit), "configured_bytes_per_expert"
        else:
            num_experts = int(layer.local_num_experts)
            parameters = [(name, p) for name, p in layer.named_parameters(recurse=False)
                          if p.ndim > 0 and p.shape[0] == num_experts]
            size = sum(p.numel() * p.element_size() for _, p in parameters)
            value = ((size / num_experts if size else None),
                     "expert_shaped_local_parameters_including_scales")
        self._layer_bytes[id(layer)] = value
        return value

    def register_native_layers(self, runner: Any, capturer: Any):
        """Remember the exact layer objects bound by vLLM's native binder.

        No name guessing or indexing every hidden layer: dense layers may have
        unfilled columns in the native buffer. Native layer_id maps to the
        actual RoutedExperts object used for both names and parameter bytes.
        """
        if not self.config.native_route_return:
            return
        self._native_layers = {}
        self._native_binding_errors = set()
        self._native_buffer_shape = tuple(capturer.get_device_buffer().shape)
        for module in runner.model.modules():
            layer = getattr(module, "routed_experts", None)
            router = getattr(module, "router", None)
            if layer is None or router is None or not hasattr(module, "layer_id"):
                continue
            layer_id = int(module.layer_id)
            if (not hasattr(router, "capture_fn") or router.capture_fn is None
                    or getattr(getattr(layer, "quant_method", None), "is_monolithic", False)):
                self._native_binding_errors.add(f"Native target route binding unsupported: {layer.layer_name}")
                continue
            if not 0 <= layer_id < self._native_buffer_shape[1]:
                self._native_binding_errors.add(f"Native layer_id outside buffer: {layer.layer_name}={layer_id}")
                continue
            if layer_id in self._native_layers:
                self._native_binding_errors.add(f"Duplicate native layer_id {layer_id}")
                continue
            router_top_k = getattr(router, "top_k", self._native_buffer_shape[2])
            if router_top_k != self._native_buffer_shape[2]:
                self._native_binding_errors.add(f"Native top-k dimension differs at {layer.layer_name}")
                continue
            self._native_layers[layer_id] = layer
            self._weight_bytes(layer)
        if not self._native_layers:
            self._native_binding_errors.add("No bound native target MoE layers discovered")
        self._route_errors.update(self._native_binding_errors)

    def native_layer_mapping(self) -> list[dict]:
        result = []
        for layer_id, layer in sorted(self._native_layers.items()):
            weight_bytes, bytes_basis = self._weight_bytes(layer)
            result.append({"native_layer_id": layer_id, "layer": layer.layer_name,
                           "num_experts": int(layer.global_num_experts),
                           "bytes_per_expert": weight_bytes, "bytes_basis": bytes_basis})
        return result

    def _native_capture_ready(self, total: int, shape: Any) -> bool:
        if not self._native_layers:
            self._route_errors.add("Native layer map unavailable; install plugin before engine creation")
            return False
        if total != self._current["scheduled_tokens"]:
            self._route_errors.add(f"Runner/scheduler scheduled-row mismatch: {total} != {self._current['scheduled_tokens']}")
            return False
        if len(shape) != 3 or shape[0] < total or any(index >= shape[1] for index in self._native_layers):
            self._route_errors.add(f"Unexpected native routing buffer shape {tuple(shape)} for {total} rows")
            return False
        if self._current.get("native_target_routes_captured"):
            self._route_errors.add(f"Duplicate native route snapshot at step {self._current['step']}")
            return False
        return True

    def capture_native_gpu(self, runner: Any, scheduler_output: Any):
        if (not self.trace or self._current is None or not self.config.native_route_return
                or self.config.native_route_backend != "gpu"):
            return
        if not getattr(runner, "routed_experts_initialized", False):
            self._route_errors.add("Native capturer was not initialized before graph capture")
            return
        total = int(scheduler_output.total_num_scheduled_tokens)
        if total == 0:
            return
        begin = time.perf_counter_ns()
        buffer = runner.routed_experts_capturer.get_device_buffer()
        if not self._native_capture_ready(total, buffer.shape):
            return
        padded = None
        if self.config.native_capture_padding:
            padded = self._current.get("target_cudagraph_padded_tokens")
            if type(padded) is not int or not total <= padded <= buffer.shape[0]:
                self._route_errors.add(
                    f"Missing/invalid physical model row count at step {self._current['step']}: "
                    f"scheduled={total}, physical={padded!r}, buffer_capacity={buffer.shape[0]}; cannot infer zero padding")
                self._current["native_graph_padding_capture_status"] = "error_missing_or_invalid_physical_rows"
                padded = None
        # execute_model has queued the target graph; these reductions follow on
        # the same stream and complete before any next-step buffer.clear_().
        previous_phase = self._phase
        self._phase = "target"
        try:
            for layer_id, layer in sorted(self._native_layers.items()):
                valid_meta = self.capture_routes(layer, buffer[:total, layer_id, :],
                    route_path="native_capturer_gpu_after_target_before_sampling", native_layer_id=layer_id)
                if self.config.native_capture_padding and padded is not None:
                    if valid_meta is None:
                        self._route_errors.add(f"No valid-route histogram to pair with physical rows at {layer.layer_name}")
                    else:
                        self._capture_graph_padding(layer, buffer[total:padded, layer_id, :], valid_meta, padded)
        finally:
            self._phase = previous_phase
        self._current.update(native_target_routes_captured=True,
                             native_route_rows=total,
                             native_route_snapshot_host_ns=time.perf_counter_ns() - begin,
                             native_route_request_order=list(runner.input_batch.req_ids))
        if self.config.native_capture_padding and padded is not None:
            self._current.update(native_physical_route_rows=padded,
                                 native_padding_route_rows=padded - total,
                                 native_graph_padding_capture_status="queued_for_deferred_validation")

    def _capture_graph_padding(self, layer, ids, valid_meta, physical_tokens):
        """Queue only extra rows; combine with the valid histogram at drain."""
        scheduled = valid_meta["tokens"]
        padding = physical_tokens - scheduled
        if tuple(ids.shape) != (padding, valid_meta["top_k"]):
            self._route_errors.add(f"Physical-row slice shape mismatch at {valid_meta['layer']} step {valid_meta['step']}")
            return
        meta = {key: valid_meta[key] for key in ("episode", "step", "phase", "layer", "batch", "gamma",
                "native_layer_id", "top_k", "num_experts", "bytes_per_expert", "bytes_basis")}
        meta.update(scheduled_tokens=scheduled, physical_tokens=physical_tokens, padding_tokens=padding,
                    model_phase="target", route_path="native_gpu_model_forward_padding_rows",
                    physical_row_scope="all model-forward rows; excludes expert GEMM tile padding",
                    target_cudagraph_runtime_mode=self._current.get("target_cudagraph_runtime_mode", "unrecorded"))
        histogram = self._device_route_histogram(ids, valid_meta["num_experts"]) if padding else None
        self._graph_padding_tensors.append((valid_meta, meta, histogram))

    def _finalize_graph_padding(self, valid_meta, meta, padding_histogram):
        experts, top_k = meta["num_experts"], meta["top_k"]
        scheduled, padding, physical = meta["scheduled_tokens"], meta["padding_tokens"], meta["physical_tokens"]
        valid = valid_meta.get("expert_token_counts")
        valid_invalid = valid_meta.get("invalid_expert_assignments", 0)
        complete = (valid is not None and len(valid) == experts and len(padding_histogram) == experts + 1
                    and physical == scheduled + padding and padding >= 0)
        if not complete:
            meta["complete"] = False
            self._route_errors.add(f"Missing/malformed physical histogram pair at {meta['layer']} step {meta['step']}")
            return
        pad, pad_invalid = padding_histogram[:-1], padding_histogram[-1]
        combined = [a + b for a, b in zip(valid, pad)]
        complete = (not valid_invalid and not pad_invalid
                    and sum(valid) == scheduled * top_k and sum(pad) == padding * top_k
                    and all(0 <= n <= scheduled for n in valid)
                    and all(0 <= n <= padding for n in pad)
                    and sum(combined) == physical * top_k)
        new_ids = [expert for expert in range(experts) if valid[expert] == 0 and pad[expert] > 0]
        distinct = sum(count > 0 for count in combined)
        meta.update(padding_expert_token_counts=pad, physical_expert_token_counts=combined,
                    valid_distinct_experts=sum(count > 0 for count in valid),
                    padding_distinct_experts=sum(count > 0 for count in pad),
                    physical_distinct_experts=distinct, padding_only_new_experts=len(new_ids),
                    padding_only_new_expert_ids=new_ids, invalid_padding_expert_assignments=pad_invalid,
                    physical_selected_expert_storage_bytes=(distinct * meta["bytes_per_expert"]
                        if meta["bytes_per_expert"] is not None else None),
                    physical_bytes_semantics="selected parameter storage, not measured HBM traffic", complete=bool(complete))
        if not complete:
            self._route_errors.add(f"Invalid/unfilled graph padding histogram at {meta['layer']} step {meta['step']}")

    def capture_native_cpu(self, scheduler_output: Any, model_runner_output: Any):
        if (not self.trace or self._current is None or not self.config.native_route_return
                or self.config.native_route_backend != "cpu"):
            return
        total = int(scheduler_output.total_num_scheduled_tokens)
        if total == 0:
            return
        begin = time.perf_counter_ns()
        payload = getattr(model_runner_output, "routed_experts", None)
        if payload is None:
            self._route_errors.add("ModelRunnerOutput has no native routed_experts payload")
            return
        data = payload.routing_data
        if not self._native_capture_ready(total, data.shape):
            return
        if data.shape[0] != total:
            self._route_errors.add("Native CPU payload includes unexpected padded rows")
            return
        import numpy as np
        for layer_id, layer in sorted(self._native_layers.items()):
            if len(self.routes) >= self.config.max_route_records:
                self._route_errors.add("max_route_records reached; increase diagnostic budget")
                break
            ids = data[:total, layer_id, :]
            experts = int(layer.global_num_experts)
            flat = ids.reshape(-1).astype(np.int64, copy=False)
            valid = (flat >= 0) & (flat < experts)
            histogram = np.bincount(np.where(valid, flat, experts), minlength=experts + 1).tolist()
            meta = self._route_metadata(layer, ids.shape,
                "ModelRunnerOutput.routed_experts.routing_data_before_scheduler_acceptance_slice",
                native_layer_id=layer_id)
            self._finalize_route_histogram(meta, histogram)
            if self.config.keep_token_routes:
                meta["topk_ids"] = ids.tolist()
                meta["token_routes_kind"] = "exact_native_target_topk_ids_no_eplb"
            self.routes.append(meta)
        self._current.update(native_target_routes_captured=True,
                             native_route_rows=total,
                             native_route_snapshot_host_ns=time.perf_counter_ns() - begin,
                             native_route_request_order=list(model_runner_output.req_ids))

    def _route_metadata(self, layer, shape, route_path, native_layer_id=None):
        weight_bytes, bytes_basis = self._weight_bytes(layer)
        meta = {"episode": self.label, "step": self._current["step"],
                "model_phase": "target" if native_layer_id is not None else self._phase,
                "phase": self._current["phase"], "batch": self._current["decode_batch"],
                "gamma": self._current["gamma"], "layer": layer.layer_name,
                "actual_gamma_counts": dict(Counter(
                    self._current["scheduled_drafts_by_request"].get(rid, 0)
                    for rid in self._current["scheduled_tokens_by_request"])),
                "tokens": int(shape[0]), "top_k": int(shape[-1]),
                "num_experts": int(layer.global_num_experts), "bytes_per_expert": weight_bytes,
                "bytes_basis": bytes_basis, "route_path": route_path,
                "verification_population": ("all_scheduled_positions_including_rejected_drafts"
                    if self._phase == "target" or native_layer_id is not None else "executed_draft_positions")}
        if native_layer_id is not None:
            meta.update(native_layer_id=native_layer_id,
                        native_ids_semantics="logical_topk_ids_equal_kernel_ids_when_EPLB_disabled")
        return meta

    def _finalize_route_histogram(self, meta, histogram):
        distinct = sum(count > 0 for count in histogram[:-1])
        meta.update(expert_token_counts=histogram[:-1],
                    invalid_expert_assignments=histogram[-1], distinct_experts=distinct,
                    selected_expert_storage_bytes=(distinct * meta["bytes_per_expert"]
                                                   if meta["bytes_per_expert"] is not None else None))
        if sum(histogram) != meta["tokens"] * meta["top_k"] or histogram[-1]:
            self._route_errors.add(f"Invalid route population at {meta['layer']} step {meta['step']}")
        if max(histogram[:-1], default=0) > meta["tokens"]:
            # In particular this detects zero-filled native columns when a
            # capture callback was never recorded into a replayed graph.
            self._route_errors.add(f"Repeated expert within token or unfilled native buffer at {meta['layer']} step {meta['step']}")

    def capture_routes(self, layer: Any, topk_ids: Any,
                       route_path="RoutedExperts.forward_modular/topk_ids", native_layer_id=None):
        if not self.trace or self._current is None:
            return
        if len(self._route_tensors) >= self.config.max_route_records:
            self._route_errors.add("max_route_records reached; increase diagnostic budget")
            return
        experts = int(layer.global_num_experts)
        ids = topk_ids
        expected = self._current["scheduled_tokens"] if self._phase == "target" else None
        if expected is not None:
            ids = ids[:expected]
        histogram = self._device_route_histogram(ids, experts)
        meta = self._route_metadata(layer, ids.shape, route_path, native_layer_id)
        self._route_tensors.append((meta, histogram))
        if self.config.keep_token_routes:
            self._token_route_tensors.append((meta, ids.detach().clone()))
        return meta

    @staticmethod
    def _device_route_histogram(ids, experts):
        import torch
        # Fixed output size avoids CUDA bincount/unique's host synchronization.
        flat = ids.reshape(-1).to(dtype=torch.int64)
        valid = (flat >= 0) & (flat < experts)
        safe = torch.where(valid, flat, experts)
        histogram = torch.zeros(experts + 1, device=ids.device, dtype=torch.int64)
        histogram.scatter_add_(0, safe, torch.ones_like(safe))
        return histogram

    def end_episode(self) -> dict:
        if self.label is None:
            raise RuntimeError("No active episode")
        # Record wall duration before diagnostic D2H; performance episodes do not
        # allocate or synchronize any route tensor at all.
        elapsed = time.perf_counter() - self._episode_started
        grouped = defaultdict(list)
        for meta, tensor in self._route_tensors:
            grouped[(str(tensor.device), tensor.shape[0])].append((meta, tensor))
        if grouped:
            import torch
            for group in grouped.values():
                rows = torch.stack([tensor for _, tensor in group]).cpu().tolist()
                for (meta, _), row in zip(group, rows):
                    self._finalize_route_histogram(meta, row)
                    self.routes.append(meta)
        padding_groups = defaultdict(list)
        for valid_meta, meta, tensor in self._graph_padding_tensors:
            if tensor is None:
                self._finalize_graph_padding(valid_meta, meta, [0] * (meta["num_experts"] + 1))
                self.graph_padding_routes.append(meta)
            else:
                padding_groups[(str(tensor.device), tensor.shape[0])].append((valid_meta, meta, tensor))
        if padding_groups:
            import torch
            for group in padding_groups.values():
                rows = torch.stack([tensor for _, _, tensor in group]).cpu().tolist()
                for (valid_meta, meta, _), row in zip(group, rows):
                    self._finalize_graph_padding(valid_meta, meta, row)
                    self.graph_padding_routes.append(meta)
        token_groups = defaultdict(list)
        for meta, tensor in self._token_route_tensors:
            token_groups[(str(tensor.device), tuple(tensor.shape), str(tensor.dtype))].append((meta, tensor))
        if token_groups:
            import torch
            for group in token_groups.values():
                arrays = torch.stack([tensor for _, tensor in group]).cpu().tolist()
                for (meta, _), rows in zip(group, arrays):
                    meta["topk_ids"] = rows
                    meta["token_routes_kind"] = ("exact_native_target_topk_ids_no_eplb"
                                                  if "native_layer_id" in meta else "exact_kernel_input_topk_ids")
        self.routes.sort(key=lambda row: (row["step"], row["model_phase"], row["layer"]))
        self.graph_padding_routes.sort(key=lambda row: (row["step"], row["layer"]))
        if self.trace and any(s["scheduled_tokens"] for s in self.steps) and not self.routes:
            self._route_errors.add("No modular MoE route calls were captured")
        target_layers = ({layer.layer_name for layer in self._native_layers.values()}
                         if self.config.native_route_return else
                         {r["layer"] for r in self.routes if r["model_phase"] == "target"})
        target_by_step = defaultdict(set)
        for route in self.routes:
            if route["model_phase"] == "target":
                target_by_step[route["step"]].add(route["layer"])
        if self.trace and target_layers:
            for step in self.steps:
                if step["scheduled_tokens"] and target_by_step[step["step"]] != target_layers:
                    self._route_errors.add(f"Incomplete target-layer coverage at step {step['step']}")
        if self.trace and self.config.native_capture_padding:
            physical_by_step = defaultdict(list)
            for row in self.graph_padding_routes:
                physical_by_step[row["step"]].append(row["layer"])
            for step in self.steps:
                layers = physical_by_step[step["step"]]
                if step["scheduled_tokens"] and (not target_layers or set(layers) != target_layers
                                                  or len(layers) != len(target_layers)):
                    self._route_errors.add(f"Incomplete physical target-layer coverage at step {step['step']}")
        summary = {"steps": len(self.steps), "elapsed_before_trace_drain_s": elapsed,
                   "scheduled_drafts": sum(s["scheduled_drafts"] for s in self.steps),
                   "accepted_drafts": sum(s["accepted_drafts"] for s in self.steps),
                   "truncated_by_controller": sum(s["truncated_by_controller"] for s in self.steps),
                   "control_ns": sum(s["control_ns"] for s in self.steps),
                   "schedule_instrumentation_ns": sum(s.get("schedule_instrumentation_ns", 0) for s in self.steps),
                   "acceptance_update_ns": sum(s["acceptance_update_ns"] for s in self.steps),
                   "gamma_histogram": dict(Counter(s["gamma"] for s in self.steps)),
                   "route_records": len(self.routes), "route_errors": sorted(self._route_errors),
                   "graph_padding_route_records": len(self.graph_padding_routes),
                   "graph_padding_capture_enabled": self.config.native_capture_padding,
                   "trace_complete": self.trace and not self._route_errors,
                   "native_route_return_enabled": self._native_route_enabled,
                   "native_route_extra_costs_active": self._native_route_enabled,
                   "route_model_scope": ("target_verification_only" if self.config.native_route_return
                                          else "target_and_draft_modular_calls"),
                   "native_route_cost_scope": ("native capture buffer writes, per-step native D2H and CPU slot storage remain active"
                                               if self._native_route_enabled else "native route return disabled"),
                   "uninstrumented_timing_eligible": not self.trace and not self._native_route_enabled,
                   "native_route_snapshot_host_ns": sum(s.get("native_route_snapshot_host_ns", 0) for s in self.steps),
                   "target_cudagraph_runtime_mode_counts": dict(Counter(
                       s["target_cudagraph_runtime_mode"] for s in self.steps
                       if "target_cudagraph_runtime_mode" in s)),
                   "alpha_final": self._alpha()}
        result = {"label": self.label, "config": self.public_config(),
                  "trace_enabled": self.trace, "summary": summary,
                  "steps": self.steps, "routes": self.routes,
                  "graph_padding_routes": self.graph_padding_routes,
                  "native_layer_mapping": self.native_layer_mapping(),
                  "native_buffer_shape": self._native_buffer_shape}
        self.label = None
        self.trace = False
        self._current = None
        self._route_tensors = []
        self._token_route_tensors = []
        self._graph_padding_tensors = []
        return result


def _patch(cls: type, name: str, wrapper):
    original = getattr(cls, name)
    _PATCHES.append((cls, name, original))
    setattr(cls, name, wrapper(original))


def _schedule_wrapper(original):
    @wraps(original)
    def wrapped(scheduler, *args, **kwargs):
        runtime = _ACTIVE
        if runtime is None:
            return original(scheduler, *args, **kwargs)
        instrument_begin = time.perf_counter_ns()
        record = runtime.before_schedule(scheduler)
        begin = time.perf_counter_ns()
        output = original(scheduler, *args, **kwargs)
        record["native_schedule_ns"] = time.perf_counter_ns() - begin
        runtime.after_schedule(scheduler, output, record)
        record["schedule_instrumentation_ns"] = (
            time.perf_counter_ns() - instrument_begin - record["native_schedule_ns"])
        return output
    return wrapped


def _update_wrapper(original):
    @wraps(original)
    def wrapped(scheduler, scheduler_output, *args, **kwargs):
        runtime = _ACTIVE
        if runtime is None:
            return original(scheduler, scheduler_output, *args, **kwargs)
        runtime._update_events = []
        runner_output = args[0] if args else kwargs.get("model_runner_output")
        runtime.capture_native_cpu(scheduler_output, runner_output)
        requests = {rid: scheduler.requests.get(rid)
                    for rid in scheduler_output.num_scheduled_tokens}
        begin = time.perf_counter_ns()
        result = original(scheduler, scheduler_output, *args, **kwargs)
        censored = {rid for rid, req in requests.items()
                    if req is None or req.is_finished()}
        runtime.finish_update(scheduler_output, time.perf_counter_ns() - begin, censored)
        return result
    return wrapped


def _stats_wrapper(original):
    @wraps(original)
    def wrapped(scheduler, spec_decoding_stats, num_draft_tokens,
                num_accepted_tokens, num_invalid_spec_tokens, request_id):
        if _ACTIVE is not None:
            invalid = (num_invalid_spec_tokens or {}).get(request_id, 0)
            _ACTIVE.observe_acceptance(num_draft_tokens - invalid,
                                       num_accepted_tokens, request_id)
        return original(scheduler, spec_decoding_stats, num_draft_tokens,
                        num_accepted_tokens, num_invalid_spec_tokens, request_id)
    return wrapped


def _modular_wrapper(original):
    @wraps(original)
    def wrapped(layer, x, topk_weights, topk_ids, *args, **kwargs):
        if _ACTIVE is not None and not _ACTIVE.config.native_route_return:
            _ACTIVE.capture_routes(layer, topk_ids)
        return original(layer, x, topk_weights, topk_ids, *args, **kwargs)
    return wrapped


def _monolithic_wrapper(original):
    @wraps(original)
    def wrapped(layer, *args, **kwargs):
        if _ACTIVE is not None and _ACTIVE.trace:
            _ACTIVE._route_errors.add(f"Monolithic route not exposed: {layer.layer_name}")
        return original(layer, *args, **kwargs)
    return wrapped


def _native_bind_wrapper(original):
    @wraps(original)
    def wrapped(runner, capturer, *args, **kwargs):
        result = original(runner, capturer, *args, **kwargs)
        if _ACTIVE is not None:
            _ACTIVE.register_native_layers(runner, capturer)
        return result
    return wrapped


def _forward_mode_wrapper(original, get_context=None):
    @wraps(original)
    def wrapped(runner, *args, **kwargs):
        runtime = _ACTIVE
        if runtime is not None and runtime.trace and runtime._current is not None:
            if get_context is None:
                from vllm.forward_context import get_forward_context
                context = get_forward_context()
            else:
                context = get_context()
            runtime._current["target_cudagraph_runtime_mode"] = str(
                getattr(context, "cudagraph_runtime_mode", "unavailable"))
            descriptor = getattr(context, "batch_descriptor", None)
            if descriptor is not None and hasattr(descriptor, "num_tokens"):
                runtime._current["target_cudagraph_padded_tokens"] = int(descriptor.num_tokens)
            runtime._current["target_cudagraph_mode_evidence"] = (
                "actual forward_context dispatch mode; capture versus replay not individually instrumented")
        return original(runner, *args, **kwargs)
    return wrapped


def _phase_wrapper(phase):
    def decorate(original):
        @wraps(original)
        def wrapped(runner, *args, **kwargs):
            runtime = _ACTIVE
            if runtime is None:
                return original(runner, *args, **kwargs)
            previous = runtime._phase
            runtime._phase = phase
            begin = time.perf_counter_ns()
            try:
                result = original(runner, *args, **kwargs)
                if phase == "target":
                    state = getattr(runner, "execute_model_state", None)
                    scheduler_output = getattr(state, "scheduler_output", None)
                    if scheduler_output is None:
                        scheduler_output = args[0] if args else kwargs.get("scheduler_output")
                    if scheduler_output is not None:
                        runtime.capture_native_gpu(runner, scheduler_output)
                return result
            finally:
                if runtime._current is not None:
                    key = f"{phase}_host_call_ns"
                    runtime._current[key] = runtime._current.get(key, 0) + time.perf_counter_ns() - begin
                runtime._phase = previous
        return wrapped
    return decorate


def install(config: PluginConfig | dict | None = None) -> Runtime:
    """Activate one process-local runtime; calling again switches configuration.

    Install before engine construction. The single-process requirement is
    intentional: globals cannot control an independently spawned worker.
    """
    global _ACTIVE
    cfg = config if isinstance(config, PluginConfig) else PluginConfig(**(config or {}))
    if cfg.strict and os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING") != "0":
        raise RuntimeError("Set VLLM_ENABLE_V1_MULTIPROCESSING=0 before install/import vLLM")
    runtime = Runtime(cfg)
    if not _PATCHES:
        scheduler = importlib.import_module("vllm.v1.core.sched.scheduler").Scheduler
        routed = importlib.import_module("vllm.model_executor.layers.fused_moe.routed_experts").RoutedExperts
        runner = importlib.import_module("vllm.v1.worker.gpu_model_runner").GPUModelRunner
        _patch(scheduler, "schedule", _schedule_wrapper)
        _patch(scheduler, "update_from_output", _update_wrapper)
        _patch(scheduler, "make_spec_decoding_stats", _stats_wrapper)
        _patch(routed, "forward_modular", _modular_wrapper)
        _patch(routed, "forward_monolithic", _monolithic_wrapper)
        _patch(runner, "execute_model", _phase_wrapper("target"))
        _patch(runner, "propose_draft_token_ids", _phase_wrapper("draft"))
        _patch(runner, "_bind_routed_experts_capturer", _native_bind_wrapper)
        _patch(runner, "_model_forward", _forward_mode_wrapper)
    _ACTIVE = runtime
    return runtime


def uninstall():
    """Restore original Python methods after an engine has been shut down."""
    global _ACTIVE
    for cls, name, original in reversed(_PATCHES):
        setattr(cls, name, original)
    _PATCHES.clear()
    _ACTIVE = None
