"""vLLM 0.26 Worker integration for the eager dual-format MoE prototype.

Example launch (inside the existing GPU-lock launcher, after all artifacts exist)::

  PYTHONPATH=/root/autodl-tmp/research_C_format_sched/code \
  C_FORMAT_DUAL_CONFIG=/absolute/path/dual.json \
    vllm serve /local/base-model --dtype bfloat16 --enforce-eager \
    --compilation-config '{"mode":0,"cudagraph_mode":"NONE"}' \
    --tensor-parallel-size 1 --worker-cls dual_format_worker.DualFormatWorker

The JSON must enumerate EVERY actual MoE layer; paths are relative to its file::

  {"schema_version":1, "policy":"threshold", "threshold":32,
   "lowbit_format":"awq", "mechanism_only":false,
   "report_path":"dual_install.json", "layers":[
     {"layer":0, "module":"model.layers.0.mlp.experts",
      "fp8_weights":"layer00_fp8.pt", "lowbit_weights":"layer00_awq.pt",
      "worker_activation_basis_id":"rmsnorm_input_basis_v1:sha256:<hash>"}
   ]}

Repeat the layer record with that layer's files and verified expected ID. The
worker derives an independent ID from the ACTUAL loaded RMSNorm parameter and
epsilon, using export_bf16_fp8_artifact.activation_basis_descriptor. It never
passes a caller's string straight to the installer. Production-scope sidecars
must carry matching descriptors AND auxiliary norm tensors whose bytes verify
the descriptor. Public AWQ in different smoothing coordinates is rejected; no
input adapter is implemented. mechanism_only=true permits artifact-coordinate
mismatch for dispatch diagnostics, explicitly without output/quality semantics.
Even that mode must correctly declare the worker's actual expected basis.

Threshold, expert_split and phase are supported. The phase rule compares each
scheduled request's computed-token position with its actual prompt boundary;
one-token prefills remain prefills and prompt-crossing/recompute batches can be
mixed. This is CPU scheduler metadata, not a batch-size heuristic. Dummy memory
profiling/warmup uses an explicitly labeled synthetic mixed context; its counters
are saved separately and reset before the first real scheduler execution.
The base model, its router and nonexpert parameters stay under the stock loader.
The local norm contract does not certify upstream activation equality or quality.

The override runs after normal weight loading/postprocessing and before memory
profiling, KV allocation and warmup. Net CUDA allocation is added to vLLM's own
model_memory_usage so both resident formats count against the KV budget. Peak
installation also holds the old layer plus both sidecars. TP/PP/DP/EP must be 1;
LoRA, sleep, weight transfer, speculation, DBO, compilation and graphs are rejected.
No server/GPU validation is claimed by this source-only integration.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import time
import traceback

import torch
from vllm.config import CUDAGraphMode, CompilationMode, set_current_vllm_config
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.utils.mem_utils import DeviceMemoryProfiler
from vllm.v1.worker.gpu_worker import Worker

try:
    from .dual_format_moe import install_dual_format, format_phase
    from .export_bf16_fp8_artifact import activation_basis_descriptor
except ImportError:
    from dual_format_moe import install_dual_format, format_phase
    from export_bf16_fp8_artifact import activation_basis_descriptor


CONFIG_ENV = "C_FORMAT_DUAL_CONFIG"
_DTYPES = {torch.bfloat16: "BF16", torch.float16: "F16", torch.float32: "F32"}


def _write_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def _resolve_file(root, value):
    if not isinstance(value, str) or not value:
        raise ValueError("artifact path must be a nonempty string")
    path = (root / value).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _read_config(path):
    config = json.loads(path.read_text())
    allowed = {"schema_version", "policy", "threshold", "lowbit_format", "layers",
               "mechanism_only", "allow_unverified_source", "release_original_weights",
               "nvfp4_backend", "record_limit", "record_cuda_events", "validate_route_values",
               "report_path"}
    if set(config) - allowed:
        raise ValueError(f"unknown dual worker config keys: {sorted(set(config) - allowed)}")
    if config.get("schema_version") != 1:
        raise ValueError("dual worker config requires schema_version=1")
    if config.get("policy") not in ("threshold", "expert_split", "phase"):
        raise ValueError("policy must be threshold, expert_split or phase")
    threshold = config.get("threshold")
    if type(threshold) is not int or threshold < 1:
        raise ValueError("threshold must be a positive integer")
    fmt = config.get("lowbit_format")
    if fmt not in ("awq", "nvfp4") or (config["policy"] == "expert_split" and fmt != "awq"):
        raise ValueError("threshold supports AWQ/NVFP4; expert_split supports AWQ only")
    for key, default in (("mechanism_only", False), ("allow_unverified_source", False),
                         ("release_original_weights", True), ("record_cuda_events", False),
                         ("validate_route_values", False)):
        config.setdefault(key, default)
        if type(config[key]) is not bool:
            raise TypeError(f"{key} must be boolean")
    if config["allow_unverified_source"] and not config["mechanism_only"]:
        raise ValueError("unverified checkpoint lineage is permitted only for mechanism_only diagnostics")
    config.setdefault("record_limit", 1024)
    if type(config["record_limit"]) is not int or config["record_limit"] < 0:
        raise ValueError("record_limit must be a nonnegative integer")
    config.setdefault("nvfp4_backend", "cutlass")
    if config["nvfp4_backend"] not in ("cutlass", "marlin"):
        raise ValueError("nvfp4_backend must be cutlass or marlin")
    layers = config.get("layers")
    if not isinstance(layers, list) or not layers:
        raise ValueError("layers must enumerate all MoE layers")
    seen = set()
    required = {"layer", "module", "fp8_weights", "lowbit_weights", "worker_activation_basis_id"}
    for entry in layers:
        if not isinstance(entry, dict) or set(entry) != required:
            raise ValueError(f"each layer requires exactly {sorted(required)}")
        layer = entry["layer"]
        if type(layer) is not int or layer < 0 or layer in seen:
            raise ValueError("layer indices must be distinct nonnegative integers")
        seen.add(layer)
        if entry["module"] != f"model.layers.{layer}.mlp.experts":
            raise ValueError("only the stock OLMoE/Qwen3 MoE decoder path is supported")
        if not isinstance(entry["worker_activation_basis_id"], str) or not re.fullmatch(
                r"rmsnorm_input_basis_v1:sha256:[0-9a-f]{64}", entry["worker_activation_basis_id"]):
            raise ValueError("each layer must declare its expected canonical worker activation basis ID")
        for key in ("fp8_weights", "lowbit_weights"):
            entry[key] = str(_resolve_file(path.parent, entry[key]))
    if not isinstance(config.get("report_path"), str) or not config["report_path"]:
        raise ValueError("report_path is required to retain installation/failure evidence")
    config["report_path"] = str((path.parent / config["report_path"]).resolve())
    return config


def _tensor_descriptor(weight, model_type, layer, hidden_size, epsilon):
    if weight.dtype not in _DTYPES or tuple(weight.shape) != (hidden_size,):
        raise ValueError("expected a one-dimensional BF16/FP16/FP32 RMSNorm parameter")
    cpu = weight.detach().to("cpu").contiguous()
    if not torch.isfinite(cpu).all():
        raise ValueError("non-finite RMSNorm parameter")
    raw = cpu.reshape(-1).view(torch.uint8).numpy().tobytes()
    return activation_basis_descriptor(model_type, layer, hidden_size,
        {"dtype": _DTYPES[cpu.dtype], "shape": list(cpu.shape)},
        hashlib.sha256(raw).hexdigest(), epsilon)


def _artifact_basis(path, *, expected_format, model_type, layer, hidden_size):
    # mmap avoids materializing every expert tensor merely to check metadata.
    data = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    meta = data["metadata"]
    if meta.get("format") != expected_format or str(meta.get("layer")) != str(layer):
        raise ValueError(f"artifact format/layer mismatch: {path}")
    descriptor = meta.get("activation_basis_descriptor")
    norm = data.get("auxiliary_tensors", {}).get("post_attention_layernorm_weight")
    evidence = {"declared_id": meta.get("activation_basis_id"), "verified_from_auxiliary_tensor": False,
                "descriptor": descriptor}
    if descriptor is not None and norm is not None:
        epsilon = float.fromhex(descriptor["epsilon_hex"])
        if not math.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("artifact RMSNorm epsilon is invalid")
        rebuilt, identity = _tensor_descriptor(norm, model_type, layer, hidden_size, epsilon)
        if rebuilt != descriptor or identity != meta.get("activation_basis_id"):
            raise ValueError(f"artifact basis metadata disagrees with its saved norm tensor: {path}")
        evidence["verified_from_auxiliary_tensor"] = meta.get("activation_basis_verified") is True
        evidence["computed_id"] = identity
    return evidence


def _validate_runtime(config):
    if importlib.metadata.version("vllm").split("+", 1)[0] != "0.26.0":
        raise RuntimeError("this Worker extension targets installed vLLM 0.26.0")
    model, parallel, compilation = config.model_config, config.parallel_config, config.compilation_config
    if not model.enforce_eager or compilation.mode != CompilationMode.NONE or compilation.cudagraph_mode != CUDAGraphMode.NONE:
        raise ValueError("dual-format worker requires --enforce-eager and compilation mode=0/cudagraph_mode=NONE")
    for key in ("tensor_parallel_size", "pipeline_parallel_size", "data_parallel_size",
                "decode_context_parallel_size", "prefill_context_parallel_size"):
        if getattr(parallel, key, 1) != 1:
            raise ValueError(f"single-GPU prototype requires {key}=1")
    for key in ("enable_expert_parallel", "enable_eplb", "enable_dbo", "use_ubatching",
                "use_sequence_parallel_moe"):
        if getattr(parallel, key, False):
            raise ValueError(f"unsupported parallel feature: {key}")
    for key in ("lora_config", "speculative_config", "weight_transfer_config"):
        if getattr(config, key, None) is not None:
            raise ValueError(f"{key} is unsupported by resident dual-format weights")
    if getattr(model, "enable_sleep_mode", False) or getattr(config, "use_v2_model_runner", False):
        raise ValueError("sleep mode and v2 model runner are unsupported")
    if model.dtype != torch.bfloat16:
        raise ValueError("dual-format delegates require model dtype bfloat16")


def scheduler_phase(scheduler_output, requests):
    """Classify actual input positions, including chunking and preemption.

    Cached num_computed_tokens is read from this scheduler output, not an async
    runner counter. Prompt length comes from NewRequestData or the cached real
    RequestState. No GPU reads or inferred phase based on query length occur.
    """
    scheduled = scheduler_output.num_scheduled_tokens
    if sum(scheduled.values()) != scheduler_output.total_num_scheduled_tokens:
        raise ValueError("scheduler token accounting mismatch")
    cached = scheduler_output.scheduled_cached_reqs
    if len(cached.req_ids) != len(cached.num_computed_tokens):
        raise ValueError("cached scheduler arrays differ in length")
    current = dict(zip(cached.req_ids, cached.num_computed_tokens))
    new = {item.req_id: item for item in scheduler_output.scheduled_new_reqs}
    if set(new) & set(current):
        raise ValueError("request appears in both new and cached schedules")
    totals = {"prefill_tokens": 0, "decode_tokens": 0, "requests": 0}
    for request_id, count in scheduled.items():
        if type(count) is not int or count <= 0:
            raise ValueError("scheduled token count must be a positive integer")
        if request_id in new:
            state = new[request_id]
            computed = state.num_computed_tokens
            prompt = state.prompt_token_ids
        else:
            if request_id not in current or request_id not in requests:
                raise ValueError("missing real cached request phase metadata")
            computed = current[request_id]
            prompt = requests[request_id].prompt_token_ids
        if prompt is None or type(computed) is not int or computed < 0:
            raise ValueError("phase extension requires text token IDs and valid computed positions")
        prefill = min(count, max(0, len(prompt) - computed))
        totals["prefill_tokens"] += prefill
        totals["decode_tokens"] += count - prefill
        totals["requests"] += 1
    phase = "mixed" if totals["prefill_tokens"] and totals["decode_tokens"] else (
        "prefill" if totals["prefill_tokens"] else "decode")
    return phase, totals


class DualFormatWorker(Worker):
    """Stock worker plus a guarded post-load/pre-profile MoE installation."""

    def load_model(self, *, load_dummy_weights: bool = False) -> None:
        _validate_runtime(self.vllm_config)
        if load_dummy_weights:
            raise ValueError("dual-format installation requires actual checkpoint weights")
        if getattr(self, "_dual_format_installed", False):
            raise RuntimeError("live model reload/reinstallation is unsupported")
        value = os.environ.get(CONFIG_ENV)
        if not value:
            raise ValueError(f"set {CONFIG_ENV} to the local dual-format JSON config")
        path = Path(value).resolve()
        config = _read_config(path)
        report_path = Path(config["report_path"])
        if report_path.exists():
            raise FileExistsError(f"preserve the previous installation report; choose a new report_path: {report_path}")
        report = {"status": "loading_base_model", "started_unix_s": time.time(),
                  "worker_class": "dual_format_worker.DualFormatWorker", "config_path": str(path),
                  "config_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "config": config,
                  "scope": "mechanism_only_no_output_or_quality_semantics" if config["mechanism_only"] else "basis_checked_unvalidated_prototype",
                  "quality_validated": False, "gpu_runtime_validated": False, "layers": {}}
        self._dual_format_report = report
        self._dual_format_methods = {}
        _write_report(report_path, report)
        try:
            super().load_model(load_dummy_weights=False)
            model = self.model_runner.get_model()
            hf = self.model_config.hf_text_config
            model_type = hf.model_type
            if model_type not in ("olmoe", "qwen3_moe"):
                raise ValueError("only stock OLMoE/Qwen3-MoE architectures are supported")
            discovered = {name: module for name, module in model.named_modules() if isinstance(module, MoERunner)}
            requested = {entry["module"] for entry in config["layers"]}
            if set(discovered) != requested:
                raise ValueError(f"all actual MoE layers must be specified; missing={sorted(set(discovered)-requested)}, extra={sorted(requested-set(discovered))}")
            # Check ALL layer contracts before replacing any original weights.
            for entry in config["layers"]:
                layer, name = entry["layer"], entry["module"]
                runner = discovered[name]
                if not isinstance(runner.routed_experts, RoutedExperts):
                    raise TypeError("actual MoERunner must own v0.26 RoutedExperts")
                if runner.routed_input_transform is not None or runner.routed_output_transform is not None:
                    raise ValueError("additional routed input/output transforms require a different basis contract")
                norm = model.get_submodule(f"model.layers.{layer}.post_attention_layernorm")
                if not isinstance(norm, RMSNorm) or not norm.has_weight or norm.variance_size_override is not None:
                    raise ValueError("expected standard weighted RMSNorm over the complete hidden axis")
                epsilon = float(norm.variance_epsilon)
                if not math.isfinite(epsilon) or epsilon <= 0:
                    raise ValueError("actual RMSNorm epsilon is invalid")
                descriptor, identity = _tensor_descriptor(norm.weight, model_type, layer, hf.hidden_size, epsilon)
                if identity != entry["worker_activation_basis_id"]:
                    raise ValueError(f"layer {layer}: declared worker basis does not match actual loaded RMSNorm bytes/epsilon")
                evidence = {"module": name, "actual_worker_basis_id": identity,
                            "actual_worker_basis_descriptor": descriptor,
                            "verification_scope": "local post_attention_RMSNorm transform, not upstream activations/router quality"}
                for arm, fmt in (("fp8", "fp8"), ("lowbit", config["lowbit_format"])):
                    evidence[arm] = _artifact_basis(entry[f"{arm}_weights"], expected_format=fmt,
                        model_type=model_type, layer=layer, hidden_size=hf.hidden_size)
                matched = all(evidence[arm]["verified_from_auxiliary_tensor"] and
                    evidence[arm].get("computed_id") == identity for arm in ("fp8", "lowbit"))
                evidence["both_artifacts_match_actual_worker_basis"] = matched
                report["layers"][str(layer)] = evidence
                _write_report(report_path, report)
                if not matched and not config["mechanism_only"]:
                    raise ValueError(f"layer {layer}: both sidecars must have independently verified norm tensors matching the worker; AWQ smoothing mismatch has no implemented input adapter")
            report["status"] = "installing_dual_weights"
            original_usage = int(self.model_runner.model_memory_usage)
            with self._maybe_get_memory_pool_context(tag="weights"), set_current_vllm_config(self.vllm_config), DeviceMemoryProfiler(self.device) as memory:
                for entry in config["layers"]:
                    layer = entry["layer"]
                    method = install_dual_format(discovered[entry["module"]].routed_experts,
                        fp8_weights=entry["fp8_weights"], lowbit_weights=entry["lowbit_weights"],
                        activation_basis_id=report["layers"][str(layer)]["actual_worker_basis_id"],
                        device=self.device, **{key: config[key] for key in (
                            "lowbit_format", "threshold", "policy", "nvfp4_backend", "mechanism_only",
                            "allow_unverified_source", "release_original_weights", "record_limit",
                            "record_cuda_events", "validate_route_values")})
                    self._dual_format_methods[layer] = method
                    report["layers"][str(layer)]["installation"] = method.summary()
                    _write_report(report_path, report)
                torch.cuda.synchronize(self.device)
            net_bytes = int(memory.consumed_memory)
            self.model_runner.model_memory_usage = original_usage + net_bytes
            if self.model_runner.model_memory_usage < 0:
                raise RuntimeError("invalid post-install model memory accounting")
            report["memory"] = {"original_model_memory_usage_bytes": original_usage,
                "net_install_cuda_allocated_bytes": net_bytes,
                "updated_model_memory_usage_bytes": int(self.model_runner.model_memory_usage),
                "measurement": "vLLM DeviceMemoryProfiler; net live CUDA allocations; before KV profiling"}
            self._dual_format_installed = True
            report.update(status="installed_before_kv_profile", completed_unix_s=time.time())
        except BaseException as error:
            report.update(status="failed", error=f"{type(error).__name__}: {error}",
                          traceback=traceback.format_exc(), completed_unix_s=time.time())
            raise
        finally:
            _write_report(report_path, report)

    def _require_dual_installation(self):
        if not getattr(self, "_dual_format_installed", False):
            raise RuntimeError("dual-format installation must finish before KV sizing/allocation/warmup")

    def determine_available_memory(self):
        self._require_dual_installation()
        with format_phase("mixed", source="synthetic_memory_profile"):
            return super().determine_available_memory()

    def initialize_from_config(self, kv_cache_config):
        self._require_dual_installation()
        return super().initialize_from_config(kv_cache_config)

    def compile_or_warm_up_model(self):
        self._require_dual_installation()
        with format_phase("mixed", source="synthetic_warmup"):
            return super().compile_or_warm_up_model()

    def execute_model(self, scheduler_output):
        self._require_dual_installation()
        if not getattr(self, "_dual_started_requests", False):
            self._dual_format_report["startup_method_statistics"] = {
                str(layer): method.summary() for layer, method in self._dual_format_methods.items()}
            for method in self._dual_format_methods.values():
                method.reset_statistics()
            self._dual_started_requests = True
            self._dual_scheduler_steps = {"prefill": 0, "decode": 0, "mixed": 0}
            self._dual_scheduler_tokens = {"prefill_tokens": 0, "decode_tokens": 0}
            _write_report(Path(self._dual_format_report["config"]["report_path"]), self._dual_format_report)
        if scheduler_output.total_num_scheduled_tokens == 0:
            return super().execute_model(scheduler_output)
        phase, counts = scheduler_phase(scheduler_output, self.model_runner.requests)
        self._dual_scheduler_steps[phase] += 1
        for key in self._dual_scheduler_tokens:
            self._dual_scheduler_tokens[key] += counts[key]
        with format_phase(phase, source="scheduler_computed_position_and_prompt_boundary"):
            return super().execute_model(scheduler_output)

    def dual_format_summary(self):
        """Read via engine collective_rpc, outside a timed request interval."""
        self._require_dual_installation()
        return {"installation": self._dual_format_report,
                "scheduler_steps": getattr(self, "_dual_scheduler_steps", {}),
                "scheduler_tokens": getattr(self, "_dual_scheduler_tokens", {}),
                "layers": {str(layer): method.summary() for layer, method in self._dual_format_methods.items()}}

    def dual_format_take_records(self, synchronize=False):
        """Drain bounded per-layer records through worker collective_rpc."""
        self._require_dual_installation()
        return {str(layer): method.take_records(synchronize=synchronize)
                for layer, method in self._dual_format_methods.items()}
