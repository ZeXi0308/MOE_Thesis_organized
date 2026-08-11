#!/usr/bin/env python3
"""StableBatch-MoE single-contribution execution-shape propagation pilot.

The full model input is identical in every arm.  M=1 and M=64 are used only
to precompute one raw expert output.  The full forward then executes natively
and replaces exactly one (token, top-k rank, expert) contribution immediately
before its routing weight is applied.  This is an execution-shape-M causal
probe; it does not identify or pin a cuBLASLt algorithm and is not a serving or
expert-parallel benchmark.
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import dataclasses
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import types
from typing import Any, Iterable, Mapping, Sequence


class ProtocolError(RuntimeError):
    """Fail-closed integrity or protocol error."""


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(tensor: Any) -> str:
    import torch

    cpu = tensor.detach().contiguous().cpu().reshape(-1)
    return hashlib.sha256(cpu.view(torch.uint8).numpy().tobytes()).hexdigest()


def tensor_bytes(tensor: Any) -> bytes:
    import torch

    cpu = tensor.detach().contiguous().cpu().reshape(-1)
    return cpu.view(torch.uint8).numpy().tobytes()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ProtocolError(f"{path}:{line_number} is not an object")
                rows.append(value)
    return rows


def write_json_new(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def write_jsonl_new(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def command_output(argv: Sequence[str]) -> str:
    result = subprocess.run(
        list(argv), check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    return result.stdout.strip()


def gpu_snapshot() -> dict[str, Any]:
    rows = command_output(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,memory.total,memory.used,driver_version",
            "--format=csv,noheader,nounits",
        ]
    ).splitlines()
    if len(rows) != 1:
        raise ProtocolError(f"expected exactly one GPU, observed {rows!r}")
    parts = [item.strip() for item in rows[0].split(",")]
    if len(parts) != 5:
        raise ProtocolError(f"unrecognized nvidia-smi row: {rows[0]!r}")
    process_lines = command_output(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ]
    ).splitlines()
    processes: list[dict[str, Any]] = []
    for row in process_lines:
        if not row.strip():
            continue
        fields = [item.strip() for item in row.split(",")]
        if len(fields) >= 3:
            processes.append(
                {"pid": int(fields[0]), "process_name": fields[1], "used_memory_mib": fields[2]}
            )
    return {
        "name": parts[0],
        "uuid": parts[1],
        "memory_total_mib": parts[2],
        "memory_used_mib": parts[3],
        "driver_version": parts[4],
        "compute_processes": processes,
    }


def verify_environment(
    config: Mapping[str, Any], pre_import_gpu: Mapping[str, Any]
) -> dict[str, Any]:
    import torch
    import transformers

    expected = config["environment"]
    if pre_import_gpu["name"] != expected["gpu_exact_name"]:
        raise ProtocolError(
            f"GPU name {pre_import_gpu['name']!r} != {expected['gpu_exact_name']!r}"
        )
    if pre_import_gpu["uuid"] != expected["gpu_uuid"]:
        raise ProtocolError(
            f"GPU UUID {pre_import_gpu['uuid']!r} != {expected['gpu_uuid']!r}"
        )
    if pre_import_gpu["driver_version"] != expected["driver_version"]:
        raise ProtocolError(
            f"driver {pre_import_gpu['driver_version']!r} != {expected['driver_version']!r}"
        )
    if pre_import_gpu["compute_processes"]:
        raise ProtocolError(
            f"GPU was not idle before torch import: {pre_import_gpu['compute_processes']!r}"
        )
    if torch.__version__ != expected["torch"]:
        raise ProtocolError(f"torch {torch.__version__!r} != {expected['torch']!r}")
    if transformers.__version__ != expected["transformers"]:
        raise ProtocolError(
            f"transformers {transformers.__version__!r} != {expected['transformers']!r}"
        )
    source_cfg = expected["transformers_olmoe_source"]
    source_path = Path(str(source_cfg["path"])).resolve()
    if not source_path.is_file() or sha256_file(source_path) != source_cfg["sha256"]:
        raise ProtocolError("Transformers OLMoE source path/hash mismatch")
    cublas_cfg = expected["cublaslt"]
    cublas_path = Path(str(cublas_cfg["path"])).resolve()
    if not cublas_path.is_file() or sha256_file(cublas_path) != cublas_cfg["sha256"]:
        raise ProtocolError("libcublasLt path/hash mismatch")
    cublas = ctypes.CDLL(str(cublas_path))
    cublas.cublasLtGetVersion.restype = ctypes.c_size_t
    observed_cublas_version = int(cublas.cublasLtGetVersion())
    if observed_cublas_version != int(cublas_cfg["version"]):
        raise ProtocolError(
            f"libcublasLt version {observed_cublas_version} != {cublas_cfg['version']}"
        )
    torch.backends.cuda.matmul.allow_tf32 = bool(expected["allow_tf32"])
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = bool(
        expected["allow_bf16_reduced_precision_reduction"]
    )
    if not torch.cuda.is_available():
        raise ProtocolError("CUDA is unavailable")
    gpu = gpu_snapshot()
    # AutoDL exposes host PIDs through nvidia-smi but container PIDs through
    # os.getpid(), so equality is not a valid ownership check.  A clean
    # pre-import snapshot plus at most one Python context closes this boundary.
    post_processes = gpu["compute_processes"]
    if len(post_processes) > 1 or any(
        "python" not in row["process_name"].lower() for row in post_processes
    ):
        raise ProtocolError(
            f"unexpected GPU processes after torch import: {post_processes!r}"
        )
    return {
        "captured_at": utc_now(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_runtime": torch.version.cuda,
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "gpu_before_torch_import": dict(pre_import_gpu),
        "gpu_after_torch_import": gpu,
        "matmul": {
            "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "allow_bf16_reduced_precision_reduction": (
                torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
            ),
        },
        "transformers_olmoe_source": {
            "path": str(source_path),
            "sha256": sha256_file(source_path),
        },
        "cublaslt": {
            "path": str(cublas_path),
            "version": observed_cublas_version,
            "sha256": sha256_file(cublas_path),
        },
    }


def verify_final_runtime(config: Mapping[str, Any]) -> dict[str, Any]:
    expected_path = str(Path(config["environment"]["cublaslt"]["path"]).resolve())
    mapped = sorted(
        {
            str(Path(line.split()[-1]).resolve())
            for line in Path("/proc/self/maps").read_text(encoding="utf-8").splitlines()
            if "libcublasLt" in line and line.split()[-1].startswith("/")
        }
    )
    if mapped != [expected_path]:
        raise ProtocolError(f"mapped libcublasLt paths {mapped!r} != {[expected_path]!r}")
    gpu = gpu_snapshot()
    processes = gpu["compute_processes"]
    if len(processes) > 1 or any(
        "python" not in row["process_name"].lower() for row in processes
    ):
        raise ProtocolError(f"GPU process isolation lost during run: {processes!r}")
    return {"captured_at": utc_now(), "mapped_cublaslt_paths": mapped, "gpu": gpu}


def verify_static_inputs(
    config: Mapping[str, Any],
    repo_root: Path,
    runner_path: Path,
    config_path: Path,
    lock_path: Path,
) -> dict[str, Any]:
    if config.get("status") != "FROZEN_PRE_RUN":
        raise ProtocolError("config is not FROZEN_PRE_RUN")
    model_cfg = config["model"]
    model_root = Path(str(model_cfg["local_path"])).resolve()
    observed_model: dict[str, str] = {}
    for relative, expected_hash in model_cfg["file_sha256"].items():
        path = model_root / relative
        if not path.is_file():
            raise ProtocolError(f"missing model file {path}")
        observed = sha256_file(path)
        if observed != expected_hash:
            raise ProtocolError(f"model hash mismatch for {relative}: {observed}")
        observed_model[relative] = observed
    manifest = repo_root / str(config["data"]["manifest"])
    if not manifest.is_file():
        raise ProtocolError(f"missing workload manifest {manifest}")
    manifest_hash = sha256_file(manifest)
    if manifest_hash != config["data"]["manifest_sha256"]:
        raise ProtocolError(f"workload manifest hash mismatch: {manifest_hash}")
    lock = load_json(lock_path)
    if lock.get("status") != "FROZEN_PRE_RUN":
        raise ProtocolError("frozen lock is not FROZEN_PRE_RUN")
    locked_files = lock.get("files")
    if not isinstance(locked_files, dict) or not locked_files:
        raise ProtocolError("frozen lock has no file bindings")
    observed_locked: dict[str, str] = {}
    for relative, expected_hash in locked_files.items():
        path = repo_root / str(relative)
        if not path.is_file():
            raise ProtocolError(f"frozen lock file is absent: {path}")
        observed = sha256_file(path)
        if observed != expected_hash:
            raise ProtocolError(f"frozen lock mismatch for {relative}: {observed}")
        observed_locked[str(relative)] = observed
    return {
        "runner_path": str(runner_path),
        "runner_sha256": sha256_file(runner_path),
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "frozen_lock_path": str(lock_path),
        "frozen_lock_sha256": sha256_file(lock_path),
        "locked_file_sha256": observed_locked,
        "manifest_path": str(manifest),
        "manifest_sha256": manifest_hash,
        "model_path": str(model_root),
        "model_file_sha256": observed_model,
    }


def load_workloads(config: Mapping[str, Any], repo_root: Path, tokenizer: Any) -> list[dict[str, Any]]:
    data = config["data"]
    documents = {int(row["document_index"]): row for row in load_jsonl(repo_root / data["manifest"])}
    workloads: list[dict[str, Any]] = []
    for document_index in data["document_indices"]:
        document_index = int(document_index)
        if document_index not in documents:
            raise ProtocolError(f"document {document_index} is absent")
        document = documents[document_index]
        text = str(document["text"])
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if text_hash != document["text_sha256"]:
            raise ProtocolError(f"document {document_index} text hash mismatch")
        token_ids = tokenizer(text, add_special_tokens=bool(data["add_special_tokens"]))[
            "input_ids"
        ]
        offset = int(data["token_offset"])
        width = int(data["window_tokens"])
        window = list(map(int, token_ids[offset : offset + width]))
        if len(window) != width:
            raise ProtocolError(f"document {document_index} has an incomplete window")
        workloads.append(
            {
                "victim_id": f"doc{document_index:03d}-offset{offset:04d}",
                "document_index": document_index,
                "text_sha256": text_hash,
                "window_token_ids": window,
                "window_token_ids_sha256": hashlib.sha256(canonical_json_bytes(window)).hexdigest(),
            }
        )
    if len(workloads) != 16:
        raise ProtocolError(f"expected 16 workloads, observed {len(workloads)}")
    return workloads


def load_model(config: Mapping[str, Any]) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_cfg = config["model"]
    local_path = str(model_cfg["local_path"])
    tokenizer = AutoTokenizer.from_pretrained(local_path, local_files_only=True, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        local_path,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation=str(config["environment"]["attn_implementation"]),
    )
    model.eval()
    model.to("cuda")
    expected_fields = {
        "num_hidden_layers": model_cfg["num_hidden_layers"],
        "hidden_size": model_cfg["hidden_size"],
        "intermediate_size": model_cfg["intermediate_size"],
        "num_experts": model_cfg["num_experts"],
        "num_experts_per_tok": model_cfg["num_experts_per_tok"],
    }
    for field, expected in expected_fields.items():
        if getattr(model.config, field, None) != expected:
            raise ProtocolError(
                f"model field {field}={getattr(model.config, field, None)!r} != {expected!r}"
            )
    if getattr(model.config, "norm_topk_prob", None) is not False:
        raise ProtocolError("expected norm_topk_prob=False")
    return model, tokenizer


@dataclasses.dataclass(frozen=True)
class PairIdentity:
    layer: int
    flat_token_idx: int
    topk_rank: int
    expert_id: int


def topk_from_logits(logits: Any, top_k: int) -> tuple[Any, Any]:
    import torch
    import torch.nn.functional as F

    probabilities = F.softmax(logits, dim=-1, dtype=torch.float)
    weights, experts = torch.topk(probabilities, k=top_k, dim=-1)
    return weights, experts


def topk_margin(logits: Any, top_k: int) -> float:
    import torch

    values = torch.topk(logits.float(), k=top_k + 1, dim=-1).values
    return float((values[top_k - 1] - values[top_k]).item())


def select_targets(
    candidates: Sequence[Mapping[str, Any]], selection: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Deterministic band-balanced selection; pure and unit-testable."""

    selected: list[dict[str, Any]] = []
    per_victim: dict[str, int] = {}
    max_per_victim = int(selection["max_targets_per_victim"])
    per_band = int(selection["targets_per_band"])
    for band_index, bounds in enumerate(selection["layer_bands"]):
        low, high = map(int, bounds)
        eligible = [row for row in candidates if low <= int(row["layer"]) <= high]
        eligible.sort(
            key=lambda row: (
                -float(row["selection_score"]),
                str(row["victim_id"]),
                int(row["layer"]),
                int(row["topk_rank"]),
                int(row["expert_id"]),
            )
        )
        band_rows: list[dict[str, Any]] = []
        for row in eligible:
            victim = str(row["victim_id"])
            if per_victim.get(victim, 0) >= max_per_victim:
                continue
            materialized = dict(row)
            materialized["layer_band_index"] = band_index
            band_rows.append(materialized)
            per_victim[victim] = per_victim.get(victim, 0) + 1
            if len(band_rows) == per_band:
                break
        if len(band_rows) != per_band:
            raise ProtocolError(
                f"band {band_index} selected {len(band_rows)}, expected {per_band}"
            )
        selected.extend(band_rows)
    if len(selected) != int(selection["target_count"]):
        raise ProtocolError(
            f"selected {len(selected)} targets, expected {selection['target_count']}"
        )
    return selected


def classify_summary(
    target_rows: Sequence[Mapping[str, Any]], gate: Mapping[str, Any]
) -> dict[str, Any]:
    reproducible = [row for row in target_rows if bool(row["reproducible_route_propagation"])]
    distinct_victims = sorted({str(row["victim_id"]) for row in reproducible})
    local_changed = [row for row in target_rows if bool(row["local_replacement_changed"])]
    token_flips = [row for row in target_rows if bool(row["reproducible_token_flip"])]
    route_count = len(reproducible)
    if route_count >= int(gate["support_min_reproducible_route_targets"]) and len(
        distinct_victims
    ) >= int(gate["support_min_distinct_victims"]):
        verdict = "SUPPORT"
    elif int(gate["suggestive_min_reproducible_route_targets"]) <= route_count <= int(
        gate["suggestive_max_reproducible_route_targets"]
    ):
        verdict = "SUGGESTIVE_TARGETED_RERUN"
    elif not local_changed:
        verdict = "LOCAL_EXECUTION_SHAPE_SIGNAL_ABSENT"
    else:
        verdict = "NO_REPRODUCIBLE_DOWNSTREAM_ROUTE_PROPAGATION"
    return {
        "verdict": verdict,
        "target_count": len(target_rows),
        "local_changed_target_count": len(local_changed),
        "reproducible_route_target_count": route_count,
        "reproducible_route_distinct_victim_count": len(distinct_victims),
        "reproducible_route_victims": distinct_victims,
        "reproducible_token_flip_target_count": len(token_flips),
    }


def bitwise_changed_elements(left: Any, right: Any) -> int:
    import torch

    if left.dtype != right.dtype or tuple(left.shape) != tuple(right.shape):
        raise ProtocolError("cannot compare tensors with different dtype or shape")
    if left.dtype == torch.bfloat16:
        return int((left.view(torch.uint16) != right.view(torch.uint16)).sum().item())
    return int((left != right).sum().item())


def run_native_capture(model: Any, input_ids: Any, config: Mapping[str, Any]) -> dict[str, Any]:
    """Run one unmodified prompt forward and capture every MoE input."""

    import torch

    blocks = [layer.mlp for layer in model.model.layers]
    captured: dict[int, Any] = {}
    handles = []

    def make_hook(layer_idx: int):
        def hook(_module: Any, inputs: tuple[Any, ...]) -> None:
            if len(inputs) != 1:
                raise ProtocolError(f"layer {layer_idx} received {len(inputs)} inputs")
            captured[layer_idx] = inputs[0].detach().clone()

        return hook

    for layer_idx, block in enumerate(blocks):
        handles.append(block.register_forward_pre_hook(make_hook(layer_idx)))
    attention_mask = torch.ones_like(input_ids)
    try:
        with torch.inference_mode():
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                output_router_logits=True,
                return_dict=True,
            )
    finally:
        for handle in handles:
            handle.remove()
    if getattr(output, "past_key_values", None) is not None:
        raise ProtocolError("use_cache=False unexpectedly returned a mutable cache")
    expected_layers = int(config["model"]["num_hidden_layers"])
    if len(captured) != expected_layers:
        raise ProtocolError(f"captured {len(captured)} MoE inputs, expected {expected_layers}")
    if output.router_logits is None or len(output.router_logits) != expected_layers:
        raise ProtocolError("router logits are incomplete")
    return {
        "output": output,
        "moe_inputs": captured,
        "attention_mask_sha256": tensor_sha256(attention_mask),
    }


def warmup_model(model: Any, input_ids: Any, config: Mapping[str, Any]) -> None:
    import torch

    captured = run_native_capture(model, input_ids, config)
    block = model.model.layers[0].mlp
    hidden = captured["moe_inputs"][0].reshape(-1, int(config["model"]["hidden_size"]))
    victim = int(config["data"]["victim_position"])
    logits = captured["output"].router_logits[0].reshape(
        -1, int(config["model"]["num_experts"])
    )[victim]
    _, experts = topk_from_logits(logits, int(config["model"]["num_experts_per_tok"]))
    expert = block.experts[int(experts[0].item())]
    with torch.inference_mode():
        expert(hidden[victim].reshape(1, -1))
        expert(hidden[victim].reshape(1, -1).repeat(64, 1))
    torch.cuda.synchronize()


def scan_candidates(
    model: Any,
    workloads: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    import torch

    model_cfg = config["model"]
    data_cfg = config["data"]
    target_layers = {int(value) for value in config["selection"]["target_layers"]}
    top_k = int(model_cfg["num_experts_per_tok"])
    hidden_size = int(model_cfg["hidden_size"])
    victim_position = int(data_cfg["victim_position"])
    candidates: list[dict[str, Any]] = []
    for workload in workloads:
        input_ids = torch.tensor(
            [workload["window_token_ids"]], dtype=torch.long, device="cuda"
        )
        capture = run_native_capture(model, input_ids, config)
        output = capture["output"]
        for layer_idx in sorted(target_layers):
            block = model.model.layers[layer_idx].mlp
            flat_hidden = capture["moe_inputs"][layer_idx].reshape(-1, hidden_size)
            victim_hidden = flat_hidden[victim_position]
            logits = output.router_logits[layer_idx].reshape(
                -1, int(model_cfg["num_experts"])
            )[victim_position]
            with torch.inference_mode():
                replay = block.gate(flat_hidden)
            native_full = output.router_logits[layer_idx].reshape(
                -1, int(model_cfg["num_experts"])
            )
            if not torch.equal(replay, native_full):
                raise ProtocolError(
                    f"{workload['victim_id']} layer {layer_idx} gate replay mismatch"
                )
            weights, experts = topk_from_logits(logits, top_k)
            next_logits = output.router_logits[layer_idx + 1].reshape(
                -1, int(model_cfg["num_experts"])
            )[victim_position]
            margin = topk_margin(next_logits, top_k)
            for rank in range(top_k):
                expert_id = int(experts[rank].item())
                expert = block.experts[expert_id]
                with torch.inference_mode():
                    out_m1 = expert(victim_hidden.reshape(1, -1))[0]
                    out_m64 = expert(victim_hidden.reshape(1, -1).repeat(64, 1))[0]
                delta = out_m64.float() - out_m1.float()
                local_l2 = float(torch.linalg.vector_norm(delta).item())
                gate_weight = float(weights[rank].item())
                weighted_l2 = gate_weight * local_l2
                score = weighted_l2 / (margin + 1.0e-6)
                candidates.append(
                    {
                        "victim_id": workload["victim_id"],
                        "document_index": int(workload["document_index"]),
                        "window_token_ids": list(workload["window_token_ids"]),
                        "window_token_ids_sha256": workload["window_token_ids_sha256"],
                        "layer": layer_idx,
                        "flat_token_idx": victim_position,
                        "topk_rank": rank,
                        "expert_id": expert_id,
                        "target_hidden_sha256": tensor_sha256(victim_hidden),
                        "target_router_logits_sha256": tensor_sha256(logits),
                        "gate_weight": gate_weight,
                        "next_layer_topk_margin": margin,
                        "local_m1_sha256": tensor_sha256(out_m1),
                        "local_m64_sha256": tensor_sha256(out_m64),
                        "local_changed_bf16_elements": bitwise_changed_elements(out_m1, out_m64),
                        "local_l2": local_l2,
                        "gate_weighted_local_l2": weighted_l2,
                        "selection_score": score,
                        "_hidden_cpu": victim_hidden.detach().cpu().clone(),
                    }
                )
    torch.cuda.synchronize()
    return candidates


def public_candidate(row: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if not key.startswith("_")}


@contextlib.contextmanager
def patched_single_contribution(
    model: Any,
    identity: PairIdentity,
    replacement: Any | None,
    mode: str,
):
    """Temporarily copy native OLMoE combine and replace one raw contribution."""

    import torch
    import torch.nn.functional as F

    if mode not in {"self", "replacement"}:
        raise ValueError(mode)
    if mode == "replacement" and replacement is None:
        raise ProtocolError("replacement mode requires a tensor")
    block = model.model.layers[identity.layer].mlp
    original_forward = block.forward
    trace: dict[str, Any] = {}

    def injected_forward(this: Any, hidden_states: Any):
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        flat_hidden = hidden_states.view(-1, hidden_dim)
        router_logits = this.gate(flat_hidden)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(
            routing_weights, this.top_k, dim=-1
        )
        if this.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(flat_hidden.dtype)
        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim),
            dtype=flat_hidden.dtype,
            device=flat_hidden.device,
        )
        expert_mask = torch.nn.functional.one_hot(
            selected_experts, num_classes=this.num_experts
        ).permute(2, 1, 0)
        pair_matches = 0
        non_target_hasher = hashlib.sha256()
        target_native_raw = None
        target_applied_raw = None
        target_weight = None
        for expert_idx in range(this.num_experts):
            expert_layer = this.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])
            current_state = flat_hidden[None, top_x].reshape(-1, hidden_dim)
            raw_outputs = expert_layer(current_state)
            local_index: int | None = None
            if expert_idx == identity.expert_id:
                matches = torch.where(
                    (top_x == identity.flat_token_idx) & (idx == identity.topk_rank)
                )[0]
                pair_matches += int(matches.numel())
                if matches.numel() == 1:
                    local_index = int(matches[0].item())
                    target_native_raw = raw_outputs[local_index].detach().clone()
                    target_weight = routing_weights[
                        identity.flat_token_idx, identity.topk_rank
                    ].detach().clone()
                    applied = (
                        target_native_raw
                        if mode == "self"
                        else replacement.to(
                            device=raw_outputs.device, dtype=raw_outputs.dtype
                        )
                    )
                    if tuple(applied.shape) != (hidden_dim,):
                        raise ProtocolError(
                            f"replacement shape {tuple(applied.shape)} != {(hidden_dim,)}"
                        )
                    raw_outputs = raw_outputs.clone()
                    raw_outputs[local_index] = applied
                    target_applied_raw = applied.detach().clone()
            hashable = raw_outputs.detach().clone()
            if local_index is not None:
                hashable[local_index].zero_()
            non_target_hasher.update(int(expert_idx).to_bytes(4, "little"))
            non_target_hasher.update(tensor_bytes(idx))
            non_target_hasher.update(tensor_bytes(top_x))
            non_target_hasher.update(tensor_bytes(hashable))
            current_hidden_states = raw_outputs * routing_weights[top_x, idx, None]
            final_hidden_states.index_add_(
                0, top_x, current_hidden_states.to(flat_hidden.dtype)
            )
        if pair_matches != 1 or target_native_raw is None or target_applied_raw is None:
            raise ProtocolError(
                f"target pair matched {pair_matches} times: {dataclasses.asdict(identity)}"
            )
        final_hidden_states = final_hidden_states.reshape(
            batch_size, sequence_length, hidden_dim
        )
        victim_logits = router_logits.reshape(-1, router_logits.shape[-1])[
            identity.flat_token_idx
        ]
        victim_weights, victim_experts = topk_from_logits(victim_logits, this.top_k)
        trace.update(
            {
                "pair_match_count": pair_matches,
                "identity": dataclasses.asdict(identity),
                "target_input_sha256": tensor_sha256(
                    flat_hidden[identity.flat_token_idx]
                ),
                "target_router_logits_sha256": tensor_sha256(victim_logits),
                "target_selected_experts": victim_experts.detach().cpu().tolist(),
                "target_routing_weights_sha256": tensor_sha256(victim_weights),
                "target_native_raw_sha256": tensor_sha256(target_native_raw),
                "target_applied_raw_sha256": tensor_sha256(target_applied_raw),
                "target_gate_weight_sha256": tensor_sha256(target_weight),
                "routing_weight_apply_count": 1,
                "non_target_contributions_sha256": non_target_hasher.hexdigest(),
                "target_moe_output_sha256": tensor_sha256(final_hidden_states),
            }
        )
        return final_hidden_states, router_logits

    block.forward = types.MethodType(injected_forward, block)
    try:
        yield trace
    finally:
        block.forward = original_forward


def run_observation(
    model: Any,
    input_ids: Any,
    config: Mapping[str, Any],
    identity: PairIdentity,
) -> dict[str, Any]:
    import torch

    block = model.model.layers[identity.layer].mlp
    captured: dict[str, Any] = {}

    def pre_hook(_module: Any, inputs: tuple[Any, ...]) -> None:
        captured["input"] = inputs[0].detach().clone()

    def post_hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
        captured["output"] = output[0].detach().clone()

    pre_handle = block.register_forward_pre_hook(pre_hook)
    post_handle = block.register_forward_hook(post_hook)
    attention_mask = torch.ones_like(input_ids)
    try:
        with torch.inference_mode():
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                output_router_logits=True,
                return_dict=True,
            )
    finally:
        post_handle.remove()
        pre_handle.remove()
    if getattr(output, "past_key_values", None) is not None:
        raise ProtocolError("observation unexpectedly returned a cache")
    victim = int(config["data"]["victim_position"])
    top_k = int(config["model"]["num_experts_per_tok"])
    routes: list[list[int]] = []
    router_hashes: list[str] = []
    for logits in output.router_logits:
        victim_logits = logits.reshape(-1, logits.shape[-1])[victim]
        _, experts = topk_from_logits(victim_logits, top_k)
        routes.append(list(map(int, experts.detach().cpu().tolist())))
        router_hashes.append(tensor_sha256(victim_logits))
    final_logits = output.logits[0, victim].detach().cpu().clone()
    target_input = captured["input"].reshape(-1, captured["input"].shape[-1])[
        identity.flat_token_idx
    ]
    return {
        "input_ids_sha256": tensor_sha256(input_ids),
        "attention_mask_sha256": tensor_sha256(attention_mask),
        "target_input_sha256": tensor_sha256(target_input),
        "target_router_logits_sha256": router_hashes[identity.layer],
        "target_moe_output_sha256": tensor_sha256(captured["output"]),
        "router_logits_sha256_by_layer": router_hashes,
        "topk_experts_by_layer": routes,
        "final_logits_sha256": tensor_sha256(final_logits),
        "greedy_token_id": int(torch.argmax(final_logits).item()),
        "_final_logits_cpu": final_logits,
    }


def public_observation(observation: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in observation.items() if not key.startswith("_")}


def arm_signature(row: Mapping[str, Any]) -> bytes:
    trace = row["intervention_trace"]
    observation = row["observation"]
    return canonical_json_bytes(
        {
            "input_ids_sha256": observation["input_ids_sha256"],
            "attention_mask_sha256": observation["attention_mask_sha256"],
            "target_input_sha256": observation["target_input_sha256"],
            "target_router_logits_sha256": observation[
                "target_router_logits_sha256"
            ],
            "target_moe_output_sha256": observation["target_moe_output_sha256"],
            "router_logits_sha256_by_layer": observation[
                "router_logits_sha256_by_layer"
            ],
            "topk_experts_by_layer": observation["topk_experts_by_layer"],
            "final_logits_sha256": observation["final_logits_sha256"],
            "greedy_token_id": observation["greedy_token_id"],
            "pair_match_count": trace["pair_match_count"],
            "target_native_raw_sha256": trace["target_native_raw_sha256"],
            "target_applied_raw_sha256": trace["target_applied_raw_sha256"],
            "non_target_contributions_sha256": trace[
                "non_target_contributions_sha256"
            ],
        }
    )


def precompute_replacements(
    model: Any, target: Mapping[str, Any], config: Mapping[str, Any]
) -> tuple[dict[int, Any], dict[str, Any]]:
    import torch

    layer = int(target["layer"])
    expert_id = int(target["expert_id"])
    expert = model.model.layers[layer].mlp.experts[expert_id]
    hidden = target["_hidden_cpu"].to(device="cuda", dtype=torch.bfloat16)
    if tensor_sha256(hidden) != target["target_hidden_sha256"]:
        raise ProtocolError("selected target hidden hash changed before side-call")
    repeats = int(config["intervention"]["repeats_per_arm"])
    replacements: dict[int, Any] = {}
    metadata: dict[str, Any] = {}
    for m_value in (
        int(config["intervention"]["baseline_m"]),
        int(config["intervention"]["treatment_m"]),
    ):
        outputs: list[Any] = []
        hashes: list[str] = []
        for _ in range(repeats):
            with torch.inference_mode():
                output = expert(hidden.reshape(1, -1).repeat(m_value, 1))[0]
            outputs.append(output.detach().clone())
            hashes.append(tensor_sha256(output))
        if len(set(hashes)) != 1:
            raise ProtocolError(
                f"same-M local expert output unstable for M={m_value}: {hashes}"
            )
        replacements[m_value] = outputs[0]
        metadata[str(m_value)] = {
            "m": m_value,
            "repeats": repeats,
            "raw_output_sha256_by_repeat": hashes,
            "raw_output_sha256": hashes[0],
        }
    m1 = int(config["intervention"]["baseline_m"])
    m64 = int(config["intervention"]["treatment_m"])
    metadata["changed_bf16_elements"] = bitwise_changed_elements(
        replacements[m1], replacements[m64]
    )
    metadata["l2"] = float(
        torch.linalg.vector_norm(
            replacements[m64].float() - replacements[m1].float()
        ).item()
    )
    torch.cuda.synchronize()
    return replacements, metadata


def changed_membership_layers(
    left_routes: Sequence[Sequence[int]], right_routes: Sequence[Sequence[int]], start: int
) -> list[int]:
    if len(left_routes) != len(right_routes):
        raise ProtocolError("route layer counts differ")
    return [
        layer
        for layer in range(start, len(left_routes))
        if set(map(int, left_routes[layer])) != set(map(int, right_routes[layer]))
    ]


def run_target(
    model: Any,
    target_index: int,
    target: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    import torch

    identity = PairIdentity(
        layer=int(target["layer"]),
        flat_token_idx=int(target["flat_token_idx"]),
        topk_rank=int(target["topk_rank"]),
        expert_id=int(target["expert_id"]),
    )
    input_ids = torch.tensor(
        [target["window_token_ids"]], dtype=torch.long, device="cuda"
    )
    replacements, local = precompute_replacements(model, target, config)

    # Side-calls above are complete before the integrity baseline and every arm.
    native = run_observation(model, input_ids, config, identity)
    if native["target_input_sha256"] != target["target_hidden_sha256"]:
        raise ProtocolError("formal native target input differs from selection hidden")
    if native["target_router_logits_sha256"] != target["target_router_logits_sha256"]:
        raise ProtocolError("formal native target router differs from selection router")
    with patched_single_contribution(model, identity, None, "self") as noop_trace:
        noop = run_observation(model, input_ids, config, identity)
    if noop_trace.get("pair_match_count") != 1:
        raise ProtocolError("native self-replacement did not uniquely match the pair")
    if noop_trace["target_input_sha256"] != target["target_hidden_sha256"]:
        raise ProtocolError("no-op target input differs from selection hidden")
    if noop_trace["target_router_logits_sha256"] != target[
        "target_router_logits_sha256"
    ]:
        raise ProtocolError("no-op target router differs from selection router")
    noop_checks = {
        "input_equal": native["input_ids_sha256"] == noop["input_ids_sha256"],
        "attention_mask_equal": (
            native["attention_mask_sha256"] == noop["attention_mask_sha256"]
        ),
        "target_input_equal": (
            native["target_input_sha256"] == noop["target_input_sha256"]
        ),
        "target_router_equal": (
            native["target_router_logits_sha256"]
            == noop["target_router_logits_sha256"]
        ),
        "target_moe_output_equal": (
            native["target_moe_output_sha256"] == noop["target_moe_output_sha256"]
        ),
        "all_routes_equal": (
            native["topk_experts_by_layer"] == noop["topk_experts_by_layer"]
        ),
        "final_logits_equal": (
            native["final_logits_sha256"] == noop["final_logits_sha256"]
        ),
    }
    if not all(noop_checks.values()):
        raise ProtocolError(f"native no-op bitwise closure failed: {noop_checks}")
    if noop_trace["target_native_raw_sha256"] != noop_trace["target_applied_raw_sha256"]:
        raise ProtocolError("self replacement changed the target raw output")
    if noop_trace["target_moe_output_sha256"] != noop["target_moe_output_sha256"]:
        raise ProtocolError("wrapper trace and forward hook disagree on target MoE output")

    repeats = int(config["intervention"]["repeats_per_arm"])
    arm_rows: dict[int, list[dict[str, Any]]] = {}
    for m_value in (
        int(config["intervention"]["baseline_m"]),
        int(config["intervention"]["treatment_m"]),
    ):
        arm_rows[m_value] = []
        for repeat in range(repeats):
            with patched_single_contribution(
                model, identity, replacements[m_value], "replacement"
            ) as trace:
                observation = run_observation(model, input_ids, config, identity)
            if trace.get("pair_match_count") != 1:
                raise ProtocolError("formal intervention did not uniquely match the pair")
            if trace["routing_weight_apply_count"] != 1:
                raise ProtocolError("routing weight was not applied exactly once")
            if trace["target_applied_raw_sha256"] != local[str(m_value)][
                "raw_output_sha256"
            ]:
                raise ProtocolError("injected raw output does not match precomputed side-call")
            if observation["input_ids_sha256"] != native["input_ids_sha256"]:
                raise ProtocolError("formal arm input IDs differ from native")
            if observation["attention_mask_sha256"] != native[
                "attention_mask_sha256"
            ]:
                raise ProtocolError("formal arm attention mask differs from native")
            if observation["target_input_sha256"] != target[
                "target_hidden_sha256"
            ]:
                raise ProtocolError("formal arm target input differs from selection hidden")
            if observation["target_router_logits_sha256"] != target[
                "target_router_logits_sha256"
            ]:
                raise ProtocolError("formal arm target router differs from selection router")
            if trace["target_input_sha256"] != observation["target_input_sha256"]:
                raise ProtocolError("wrapper and forward hook disagree on target input")
            if trace["target_router_logits_sha256"] != observation[
                "target_router_logits_sha256"
            ]:
                raise ProtocolError("wrapper and model output disagree on target router")
            if trace["target_native_raw_sha256"] != noop_trace[
                "target_native_raw_sha256"
            ]:
                raise ProtocolError("formal arm native raw output differs from no-op")
            if trace["target_selected_experts"] != noop_trace[
                "target_selected_experts"
            ] or trace["target_routing_weights_sha256"] != noop_trace[
                "target_routing_weights_sha256"
            ]:
                raise ProtocolError("formal arm target route/weights differ from no-op")
            for layer_idx in range(identity.layer + 1):
                if observation["router_logits_sha256_by_layer"][layer_idx] != native[
                    "router_logits_sha256_by_layer"
                ][layer_idx]:
                    raise ProtocolError(
                        f"formal arm differs from native before intervention at layer {layer_idx}"
                    )
            arm_rows[m_value].append(
                {
                    "repeat": repeat,
                    "m": m_value,
                    "intervention_trace": dict(trace),
                    "observation": observation,
                }
            )
        signatures = [arm_signature(row) for row in arm_rows[m_value]]
        if len(set(signatures)) != 1:
            raise ProtocolError(f"same-M full-forward arm unstable for M={m_value}")

    m1 = int(config["intervention"]["baseline_m"])
    m64 = int(config["intervention"]["treatment_m"])
    first_m1 = arm_rows[m1][0]
    first_m64 = arm_rows[m64][0]
    for repeat in range(repeats):
        left = arm_rows[m1][repeat]
        right = arm_rows[m64][repeat]
        left_trace = left["intervention_trace"]
        right_trace = right["intervention_trace"]
        for field in (
            "target_input_sha256",
            "target_router_logits_sha256",
            "target_selected_experts",
            "target_routing_weights_sha256",
            "target_native_raw_sha256",
            "target_gate_weight_sha256",
            "non_target_contributions_sha256",
        ):
            if left_trace[field] != right_trace[field]:
                raise ProtocolError(f"M1/M64 isolation mismatch in {field}")
        for layer_idx in range(identity.layer + 1):
            if (
                left["observation"]["router_logits_sha256_by_layer"][layer_idx]
                != right["observation"]["router_logits_sha256_by_layer"][layer_idx]
            ):
                raise ProtocolError(
                    f"upstream/target router changed before intervention at layer {layer_idx}"
                )

    changed_by_repeat = [
        changed_membership_layers(
            arm_rows[m1][repeat]["observation"]["topk_experts_by_layer"],
            arm_rows[m64][repeat]["observation"]["topk_experts_by_layer"],
            identity.layer + 1,
        )
        for repeat in range(repeats)
    ]
    if len({tuple(value) for value in changed_by_repeat}) != 1:
        raise ProtocolError(f"cross-arm downstream route difference is unstable: {changed_by_repeat}")
    token_pairs = [
        [
            arm_rows[m1][repeat]["observation"]["greedy_token_id"],
            arm_rows[m64][repeat]["observation"]["greedy_token_id"],
        ]
        for repeat in range(repeats)
    ]
    if len({tuple(value) for value in token_pairs}) != 1:
        raise ProtocolError(f"cross-arm greedy-token comparison is unstable: {token_pairs}")

    local_changed = bool(local["changed_bf16_elements"] > 0)
    changed_layers = changed_by_repeat[0]
    combine_changed = bool(
        first_m1["observation"]["target_moe_output_sha256"]
        != first_m64["observation"]["target_moe_output_sha256"]
    )
    if changed_layers and not combine_changed:
        raise ProtocolError("downstream route changed without a target MoE output change")
    if (
        not combine_changed
        and first_m1["observation"]["final_logits_sha256"]
        != first_m64["observation"]["final_logits_sha256"]
    ):
        raise ProtocolError("final logits changed without a target MoE output change")
    final_l2 = float(
        torch.linalg.vector_norm(
            first_m64["observation"]["_final_logits_cpu"].float()
            - first_m1["observation"]["_final_logits_cpu"].float()
        ).item()
    )
    serial_arms: dict[str, Any] = {}
    for m_value, rows in arm_rows.items():
        serial_arms[str(m_value)] = [
            {
                "repeat": row["repeat"],
                "m": row["m"],
                "intervention_trace": row["intervention_trace"],
                "observation": public_observation(row["observation"]),
            }
            for row in rows
        ]
    return {
        "target_index": target_index,
        "target_id": f"target-{target_index:02d}",
        **public_candidate(target),
        "identity": dataclasses.asdict(identity),
        "local_side_call": local,
        "local_replacement_changed": local_changed,
        "target_moe_output_changed_after_combine": combine_changed,
        "native_noop_checks": noop_checks,
        "native_observation": public_observation(native),
        "noop_observation": public_observation(noop),
        "noop_intervention_trace": dict(noop_trace),
        "arms": serial_arms,
        "changed_downstream_membership_layers_by_repeat": changed_by_repeat,
        "reproducible_route_propagation": bool(
            local_changed and combine_changed and changed_layers
        ),
        "earliest_changed_downstream_layer": (
            min(changed_layers) if changed_layers else None
        ),
        "greedy_token_pairs_by_repeat": token_pairs,
        "reproducible_token_flip": bool(
            local_changed and combine_changed and token_pairs[0][0] != token_pairs[0][1]
        ),
        "final_logits_l2_m1_vs_m64": final_l2,
        "integrity_status": "PASS",
    }


def build_manifest(output_dir: Path) -> dict[str, Any]:
    files: dict[str, dict[str, Any]] = {}
    for path in sorted(output_dir.iterdir()):
        if not path.is_file() or path.name in {"MANIFEST.json", "RUN_STATUS.json"}:
            continue
        files[path.name] = {"size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
    return {
        "schema_version": "stablebatch-single-contribution-manifest-v1",
        "created_at": utc_now(),
        "files": files,
    }


def run_acceptance(
    model: Any,
    workloads: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    output_dir: Path,
) -> None:
    import torch

    input_ids = torch.tensor(
        [workloads[0]["window_token_ids"]], dtype=torch.long, device="cuda"
    )
    warmup_model(model, input_ids, config)
    rows = scan_candidates(model, workloads[:1], config)
    if len(rows) != 15 * int(config["model"]["num_experts_per_tok"]):
        raise ProtocolError(f"acceptance candidate count {len(rows)} is unexpected")
    write_json_new(
        output_dir / "REAL_GPU_ACCEPTANCE.json",
        {
            "schema_version": "stablebatch-single-contribution-acceptance-v1",
            "status": "PASS",
            "evidence_boundary": "harness_and_real_model_smoke_only_not_scientific_result",
            "victim_id": rows[0]["victim_id"],
            "candidate_count": len(rows),
            "local_changed_candidate_count": sum(
                int(row["local_changed_bf16_elements"] > 0) for row in rows
            ),
            "first_candidate": public_candidate(rows[0]),
        },
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frozen-lock", type=Path, required=True)
    parser.add_argument("--acceptance-only", action="store_true")
    parser.add_argument("--max-wall-seconds", type=int, default=7200)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    runner_path = Path(__file__).resolve()
    config_path = args.config.resolve()
    lock_path = args.frozen_lock.resolve()
    repo_root = args.repo_root.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise ProtocolError(f"refusing to reuse output directory {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir()
    started = time.time()
    config = load_json(config_path)
    write_json_new(
        output_dir / "run_request.json",
        {
            "schema_version": "stablebatch-single-contribution-run-request-v1",
            "started_at": utc_now(),
            "argv": sys.argv,
            "pid": os.getpid(),
            "acceptance_only": bool(args.acceptance_only),
            "max_wall_seconds": args.max_wall_seconds,
            "runner_path": str(runner_path),
            "runner_sha256": sha256_file(runner_path),
            "config_path": str(config_path),
            "config_sha256": sha256_file(config_path),
            "frozen_lock_path": str(lock_path),
            "frozen_lock_sha256": sha256_file(lock_path),
            "repo_root": str(repo_root),
            "git_head": command_output(["git", "-C", str(repo_root), "rev-parse", "HEAD"]),
            "git_status_short": command_output(
                ["git", "-C", str(repo_root), "status", "--short"]
            ),
        },
    )
    try:
        pre_import_gpu = gpu_snapshot()
        environment = verify_environment(config, pre_import_gpu)
        static = verify_static_inputs(
            config, repo_root, runner_path, config_path, lock_path
        )
        write_json_new(output_dir / "environment.json", environment)
        write_json_new(output_dir / "static_bindings.json", static)
        write_json_new(output_dir / "config_snapshot.json", config)
        model, tokenizer = load_model(config)
        workloads = load_workloads(config, repo_root, tokenizer)
        write_jsonl_new(output_dir / "workloads.jsonl", workloads)
        if args.acceptance_only:
            run_acceptance(model, workloads, config, output_dir)
            write_json_new(
                output_dir / "runtime_final.json", verify_final_runtime(config)
            )
            write_json_new(output_dir / "MANIFEST.json", build_manifest(output_dir))
            write_json_new(
                output_dir / "RUN_STATUS.json",
                {
                    "status": "COMPLETE_ACCEPTANCE_ONLY",
                    "scientific_result_eligible": False,
                    "completed_at": utc_now(),
                    "wall_seconds": time.time() - started,
                },
            )
            return 0

        first_ids = __import__("torch").tensor(
            [workloads[0]["window_token_ids"]], dtype=__import__("torch").long, device="cuda"
        )
        warmup_model(model, first_ids, config)
        candidates = scan_candidates(model, workloads, config)
        write_jsonl_new(
            output_dir / "candidate_sweep.jsonl",
            (public_candidate(row) for row in candidates),
        )
        targets = select_targets(candidates, config["selection"])
        write_jsonl_new(
            output_dir / "selected_targets.jsonl",
            (public_candidate(row) for row in targets),
        )
        target_results: list[dict[str, Any]] = []
        result_path = output_dir / "target_results.jsonl"
        with result_path.open("x", encoding="utf-8") as stream:
            for target_index, target in enumerate(targets):
                if time.time() - started > args.max_wall_seconds:
                    raise TimeoutError("pilot exceeded --max-wall-seconds")
                row = run_target(model, target_index, target, config)
                target_results.append(row)
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
        write_json_new(output_dir / "runtime_final.json", verify_final_runtime(config))
        summary = {
            "schema_version": "stablebatch-single-contribution-summary-v1",
            "status": "COMPLETE",
            "evidence_boundary": config["research_boundary"],
            **classify_summary(target_results, config["gate"]),
            "support_rule": config["interpretation"]["support"],
            "suggestive_rule": config["interpretation"]["suggestive"],
            "weakening_rule": config["interpretation"]["weakens"],
            "all_target_integrity_pass": all(
                row["integrity_status"] == "PASS" for row in target_results
            ),
            "wall_seconds": time.time() - started,
            "completed_at": utc_now(),
        }
        write_json_new(output_dir / "summary.json", summary)
        write_json_new(output_dir / "MANIFEST.json", build_manifest(output_dir))
        write_json_new(
            output_dir / "RUN_STATUS.json",
            {
                "status": "COMPLETE",
                "scientific_result_eligible": True,
                "verdict": summary["verdict"],
                "completed_at": utc_now(),
                "wall_seconds": time.time() - started,
            },
        )
        return 0
    except BaseException as error:
        failure = {
            "status": "FAILED",
            "scientific_result_eligible": False,
            "error_type": type(error).__name__,
            "error": str(error),
            "failed_at": utc_now(),
            "wall_seconds": time.time() - started,
        }
        if not (output_dir / "FAILURE.json").exists():
            write_json_new(output_dir / "FAILURE.json", failure)
        if not (output_dir / "RUN_STATUS.json").exists():
            write_json_new(output_dir / "RUN_STATUS.json", failure)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
