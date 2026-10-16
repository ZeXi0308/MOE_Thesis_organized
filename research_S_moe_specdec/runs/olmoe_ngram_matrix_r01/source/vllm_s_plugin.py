"""Process-local vLLM 0.26 speculative-length control and MoE route tracing.

Install before constructing an eager, single-process, synchronous V1 engine.
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
    strict: bool = True

    def __post_init__(self):
        if self.mode not in {"disabled", "static", "dense_heuristic", "adaptive", "native"}:
            raise ValueError(f"Unknown mode {self.mode!r}")
        if self.proposal_budget not in {"max", "selected"}:
            raise ValueError("proposal_budget must be max or selected")
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
        self._pending: dict[int, dict] = {}
        self._current: dict | None = None
        self._phase = "target"
        self._route_tensors: list[tuple[dict, Any]] = []
        self._layer_bytes: dict[int, tuple[float | None, str]] = {}
        self._schedulers: set[int] = set()
        self._route_errors: set[str] = set()
        self._update_events: list[tuple[int, int, str]] | None = None
        self._episode_started = 0.0
        self._fallback_accepts = 0
        self._fallback_trials = 0

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
                        "native_dynamic_schedule": scheduler.dynamic_sd_lookup is not None}
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
        if self.config.strict and model_config is not None and not model_config.enforce_eager:
            raise RuntimeError("Use enforce_eager=True: Python route hooks must execute each step")
        if self.config.mode != "native" and scheduler.dynamic_sd_lookup is not None:
            raise RuntimeError("Use mode='native' with num_speculative_tokens_per_batch_size; avoid two simultaneous controllers")
        if self.config.mode != "native" and scheduler.num_spec_tokens < self.config.max_gamma:
            raise RuntimeError("Engine speculative maximum is below PluginConfig.max_gamma")

    def start_episode(self, label: str, trace: bool = False, reset_acceptance: bool = False):
        if self.label is not None:
            raise RuntimeError("end_episode must precede the next start_episode")
        self.label = label
        self.trace = bool(trace)
        self.steps = []
        self.routes = []
        self._pending = {}
        self._current = None
        self._route_tensors = []
        self._route_errors = set()
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
            decision = _jsonable_decision(cfg.policy.choose(
                batch, draft_gamma=cfg.max_gamma if cfg.proposal_budget == "max" else None))
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
        record.update({"scheduled_requests": len(token_counts),
                       "decode_batch": len(decode),
                       "prefill_batch": len(token_counts) - len(decode),
                       "phase": ("mixed" if decode and len(decode) != len(token_counts)
                                 else "decode" if decode else "prefill"),
                       "scheduled_tokens": sum(token_counts.values()),
                       "scheduled_tokens_by_request": token_counts,
                       "scheduled_drafts_by_request": scheduled_drafts,
                       "scheduled_drafts": sum(scheduled_drafts.values()),
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
        eligible = [x for x in events if x[2] not in (censored or set())]
        if eligible and self.acceptance is not None:
            self.acceptance.update([x[0] for x in eligible], [x[1] for x in eligible])
        self._fallback_accepts += sum(x[0] for x in eligible)
        self._fallback_trials += sum(min(x[0] + 1, x[1]) for x in eligible)
        record = self._pending.pop(id(output), None)
        if record is not None:
            record.update({"accepted_by_request": {x[2]: x[0] for x in events},
                           "accepted_drafts": sum(x[0] for x in events),
                           "observed_drafts": sum(x[1] for x in events),
                           "ewma_eligible_requests": len(eligible),
                           "ewma_censored_requests": sorted(censored or set()),
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

    def capture_routes(self, layer: Any, topk_ids: Any):
        if not self.trace or self._current is None:
            return
        if len(self._route_tensors) >= self.config.max_route_records:
            self._route_errors.add("max_route_records reached; increase diagnostic budget")
            return
        import torch
        # Fixed output size avoids CUDA bincount/unique's host synchronization.
        experts = int(layer.global_num_experts)
        ids = topk_ids
        expected = self._current["scheduled_tokens"] if self._phase == "target" else None
        if expected is not None:
            ids = ids[:expected]
        flat = ids.reshape(-1).to(dtype=torch.int64)
        valid = (flat >= 0) & (flat < experts)
        safe = torch.where(valid, flat, experts)
        histogram = torch.zeros(experts + 1, device=ids.device, dtype=torch.int64)
        histogram.scatter_add_(0, safe, torch.ones_like(safe))
        weight_bytes, bytes_basis = self._weight_bytes(layer)
        meta = {"episode": self.label, "step": self._current["step"],
                "model_phase": self._phase, "phase": self._current["phase"],
                "batch": self._current["decode_batch"],
                "gamma": self._current["gamma"], "layer": layer.layer_name,
                "actual_gamma_counts": dict(Counter(
                    self._current["scheduled_drafts_by_request"].get(rid, 0)
                    for rid in self._current["scheduled_tokens_by_request"])),
                "tokens": int(ids.shape[0]), "top_k": int(ids.shape[-1]),
                "num_experts": experts, "bytes_per_expert": weight_bytes,
                "bytes_basis": bytes_basis,
                "route_path": "RoutedExperts.forward_modular/topk_ids"}
        self._route_tensors.append((meta, histogram))

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
                    distinct = sum(count > 0 for count in row[:-1])
                    meta.update({"expert_token_counts": row[:-1],
                                 "invalid_expert_assignments": row[-1],
                                 "distinct_experts": distinct,
                                 "selected_expert_storage_bytes":
                                 (distinct * meta["bytes_per_expert"]
                                  if meta["bytes_per_expert"] is not None else None)})
                    self.routes.append(meta)
        self.routes.sort(key=lambda row: (row["step"], row["model_phase"], row["layer"]))
        if self.trace and any(s["scheduled_tokens"] for s in self.steps) and not self.routes:
            self._route_errors.add("No modular MoE route calls were captured")
        summary = {"steps": len(self.steps), "elapsed_before_trace_drain_s": elapsed,
                   "scheduled_drafts": sum(s["scheduled_drafts"] for s in self.steps),
                   "accepted_drafts": sum(s["accepted_drafts"] for s in self.steps),
                   "truncated_by_controller": sum(s["truncated_by_controller"] for s in self.steps),
                   "control_ns": sum(s["control_ns"] for s in self.steps),
                   "acceptance_update_ns": sum(s["acceptance_update_ns"] for s in self.steps),
                   "gamma_histogram": dict(Counter(s["gamma"] for s in self.steps)),
                   "route_records": len(self.routes), "route_errors": sorted(self._route_errors),
                   "trace_complete": self.trace and not self._route_errors,
                   "alpha_final": self._alpha()}
        result = {"label": self.label, "config": self.public_config(),
                  "trace_enabled": self.trace, "summary": summary,
                  "steps": self.steps, "routes": self.routes}
        self.label = None
        self.trace = False
        self._current = None
        self._route_tensors = []
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
        record = runtime.before_schedule(scheduler)
        begin = time.perf_counter_ns()
        output = original(scheduler, *args, **kwargs)
        record["native_schedule_ns"] = time.perf_counter_ns() - begin
        runtime.after_schedule(scheduler, output, record)
        return output
    return wrapped


def _update_wrapper(original):
    @wraps(original)
    def wrapped(scheduler, scheduler_output, *args, **kwargs):
        runtime = _ACTIVE
        if runtime is None:
            return original(scheduler, scheduler_output, *args, **kwargs)
        runtime._update_events = []
        requests = {rid: scheduler.requests.get(rid)
                    for rid in scheduler_output.scheduled_spec_decode_tokens}
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
        if _ACTIVE is not None:
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
                return original(runner, *args, **kwargs)
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
    _ACTIVE = runtime
    return runtime


def uninstall():
    """Restore original Python methods after an engine has been shut down."""
    global _ACTIVE
    for cls, name, original in reversed(_PATCHES):
        setattr(cls, name, original)
    _PATCHES.clear()
    _ACTIVE = None
