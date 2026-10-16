"""Save the AWQ-smoothed, unquantized model before CT mutates it on compression.

The original checkpoint index defines the complete set of model tensors. Added
quantization buffers are deliberately excluded. This is a full-model master,
including attention, norms and router; it is not an expert-only reconstruction.
Imports are CPU-safe; tensor access occurs only when explicitly called.
"""
from __future__ import annotations

import hashlib
from contextlib import ExitStack, contextmanager
import json
import math
from pathlib import Path
import shutil

from export_bf16_fp8_artifact import activation_basis_descriptor, canonical_json, read_header


def raw_tensor_hash(value):
    import torch
    return hashlib.sha256(value.detach().cpu().contiguous().reshape(-1)
                          .view(torch.uint8).numpy().tobytes()).hexdigest()


def content_revision(identity):
    payload = {key: identity[key] for key in ("config_sha256", "weight_files")}
    return "sha256:" + hashlib.sha256(canonical_json(payload).encode()).hexdigest()


def require_disk_space(outputs, reserve_gib):
    """Sum outstanding writes per filesystem, keeping one reserve on each."""
    devices = {}
    for path, estimated_bytes in outputs:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        entry = devices.setdefault(path.stat().st_dev, {"path": path, "bytes": 0})
        entry["bytes"] += int(estimated_bytes)
    result = []
    for entry in devices.values():
        free = shutil.disk_usage(entry["path"]).free
        required = entry["bytes"] + int(reserve_gib * 2**30)
        row = {"path": str(entry["path"]), "free_bytes": free,
               "estimated_remaining_output_bytes": entry["bytes"],
               "reserve_bytes": int(reserve_gib * 2**30)}
        result.append(row)
        if free < required:
            raise RuntimeError(f"Insufficient disk: {row}; need {required / 2**30:.2f} GiB")
    return result


def source_layout(source, manifest):
    source = Path(source)
    index_path = source / "model.safetensors.index.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else None
    files = {row["path"] for row in manifest["identity"]["weight_files"]}
    headers, mapping = {}, {}
    for file in sorted(files):
        path = (source / file).resolve()
        if not path.is_relative_to(source.resolve()):
            raise ValueError(f"Invalid shard path: {file}")
        header, _, _ = read_header(path)
        headers[file] = {key: value for key, value in header.items() if key != "__metadata__"}
        for key in headers[file]:
            if key in mapping:
                raise ValueError(f"Duplicate tensor in source shards: {key}")
            mapping[key] = file
    if index and index["weight_map"] != mapping:
        raise ValueError("Source index and actual shard tensor keys disagree")
    return headers, mapping


def model_basis(model, config):
    import torch
    model_type = config["model_type"]
    eps = 1e-5 if model_type == "olmoe" else float(config["rms_norm_eps"])
    if model_type not in ("olmoe", "qwen3_moe") or not math.isfinite(eps) or eps <= 0:
        raise ValueError("Unsupported model/epsilon for common activation basis")
    result = {}
    for layer in range(config["num_hidden_layers"]):
        prefix = f"model.layers.{layer}"
        module = model.get_submodule(prefix + ".post_attention_layernorm")
        actual_eps = getattr(module, "variance_epsilon", getattr(module, "eps", None))
        if actual_eps is None or float(actual_eps) != eps:
            raise ValueError(f"Runtime norm epsilon differs from installed vLLM: {prefix}: {actual_eps} vs {eps}")
        norm = module.weight.detach().cpu().contiguous()
        router = model.get_submodule(prefix + ".mlp.gate").weight.detach().cpu().contiguous()
        if norm.dtype not in (torch.bfloat16, torch.float32) or not torch.isfinite(norm).all():
            raise ValueError(f"Invalid unquantized norm: {prefix}")
        if not torch.isfinite(router).all():
            raise ValueError(f"Invalid router: {prefix}")
        descriptor, basis_id = activation_basis_descriptor(
            model_type, layer, config["hidden_size"],
            {"dtype": "BF16" if norm.dtype == torch.bfloat16 else "F32", "shape": list(norm.shape)},
            raw_tensor_hash(norm), eps)
        result[str(layer)] = {"activation_basis_id": basis_id,
                              "activation_basis_descriptor": descriptor,
                              "norm_tensor": prefix + ".post_attention_layernorm.weight",
                              "router_tensor": prefix + ".mlp.gate.weight",
                              "router_raw_bytes_sha256": raw_tensor_hash(router),
                              "scope": "post-attention norm and hidden-axis convention; not upstream hidden-state equality"}
    return result


def save_bf16_master(model, tokenizer, source, output, manifest, basis):
    """Write raw, smoothed tensors while CT status is FROZEN, never dequantize."""
    import torch
    from safetensors.torch import save_file

    source, output = Path(source), Path(output)
    headers, mapping = source_layout(source, manifest)
    for name, module in model.named_modules():
        if "compressed" in str(getattr(module, "quantization_status", "")).lower():
            raise RuntimeError(f"Cannot reconstruct the BF16 master after compression: {name}")
    # Resolve every original key before writing shards. HF MoE linearization
    # must expose the original per-expert projection names; no remap is guessed.
    for key in mapping:
        parent, attr = key.rsplit(".", 1)
        module = model.get_submodule(parent)
        if attr not in module._parameters and attr not in module._buffers:
            raise ValueError(f"Linearized model lacks original checkpoint tensor: {key}")
    torch_dtypes = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32,
                    "I64": torch.int64, "I32": torch.int32, "BOOL": torch.bool}
    total_bytes = 0
    for filename, header in headers.items():
        shard = {}
        for key, spec in header.items():
            parent, attr = key.rsplit(".", 1)
            # Access goes through CT's OffloadCache, which onloads an actual
            # tensor, not an Accelerate meta placeholder. The raw master is
            # unmodified outside the temporary quantized-forward context.
            value = getattr(model.get_submodule(parent), attr).detach().cpu().contiguous().clone()
            if list(value.shape) != spec["shape"] or value.dtype != torch_dtypes.get(spec["dtype"]):
                raise ValueError(f"Master shape/dtype changed: {key}: {value.shape}/{value.dtype}, expected {spec}")
            if value.is_floating_point() and not bool(torch.isfinite(value).all()):
                raise ValueError(f"Nonfinite master weight: {key}")
            shard[key] = value
            total_bytes += value.numel() * value.element_size()
        save_file(shard, str(output / filename), metadata={"format": "pt"})
        del shard
    # Original structure/config deliberately excludes CT quantization_config.
    shutil.copyfile(source / "config.json", output / "config.json")
    if (source / "generation_config.json").exists():
        shutil.copyfile(source / "generation_config.json", output / "generation_config.json")
    tokenizer.save_pretrained(output)
    (output / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": total_bytes}, "weight_map": mapping}, indent=2) + "\n")
    verify_saved_basis(output, basis)
    return {"tensor_count": len(mapping), "tensor_bytes": total_bytes,
            "saved_original_parameter_keys_only": True,
            "quantization_buffers_saved": False,
            "weights": "post-AWQ-smoothing BF16 before compression; no dequantization"}


def saved_headers(directory):
    directory = Path(directory)
    index_path = directory / "model.safetensors.index.json"
    if index_path.exists():
        mapping = json.loads(index_path.read_text())["weight_map"]
        files = set(mapping.values())
    else:
        mapping, files = {}, {p.name for p in directory.glob("*.safetensors")}
    result = {}
    for filename in files:
        header, _, _ = read_header(directory / filename)
        for key, spec in header.items():
            if key != "__metadata__":
                if key in result or (mapping and mapping.get(key) != filename):
                    raise ValueError(f"Saved tensor/index mismatch: {key}")
                result[key] = (filename, spec)
    if mapping and result.keys() != mapping.keys():
        raise ValueError("Saved index refers to missing tensors")
    return result


def verify_saved_basis(directory, basis):
    headers = saved_headers(directory)
    with saved_tensor_reader(directory, headers) as read:
        for row in basis.values():
            checks = ((row["norm_tensor"], row["activation_basis_descriptor"]["weight_raw_bytes_sha256"]),
                      (row["router_tensor"], row["router_raw_bytes_sha256"]))
            for key, expected_hash in checks:
                if raw_tensor_hash(read(key)) != expected_hash:
                    raise ValueError(f"Saved norm/router differs from common basis: {key}")
    return {"layers_checked": len(basis), "norm_and_router_bytes_match": True}


@contextmanager
def saved_tensor_reader(directory, headers):
    from safetensors import safe_open
    with ExitStack() as stack:
        handles = {}
        def read(key):
            filename = headers[key][0]
            if filename not in handles:
                handles[filename] = stack.enter_context(safe_open(
                    str(Path(directory) / filename), framework="pt", device="cpu"))
            return handles[filename].get_tensor(key)
        yield read


def check_compressed_expert_layout(directory, shape, fmt):
    """Require the concrete CT layouts consumed by vLLM 0.26 loaders.

    This catches missing NVFP4 input_global_scale (a CT 0.19 vs 0.17
    compression_param_names difference) but does not claim runtime validation.
    """
    headers = saved_headers(directory)
    checked = 0
    with saved_tensor_reader(directory, headers) as read:
        for layer in range(shape["layers"]):
            for expert in range(shape["experts"]):
                globals_by_projection = {}
                for projection in ("gate_proj", "up_proj", "down_proj"):
                    n, k = ((shape["d"], shape["intermediate"]) if projection == "down_proj"
                            else (shape["intermediate"], shape["d"]))
                    prefix = f"model.layers.{layer}.mlp.experts.{expert}.{projection}."
                    expected = ({"weight_packed": ([n, k // 8], ("I32",)),
                                 "weight_scale": ([n, k // 128], ("BF16", "F16", "F32")),
                                 "weight_shape": ([2], ("I64",))} if fmt == "awq" else
                                {"weight_packed": ([n, k // 2], ("U8",)),
                                 "weight_scale": ([n, k // 16], ("F8_E4M3",)),
                                 "weight_global_scale": ([1], ("F32",)),
                                 "input_global_scale": ([1], ("F32",))})
                    if prefix + "weight" in headers or prefix + "weight_zero_point" in headers:
                        raise ValueError(f"Expected symmetric compressed expert: {prefix}")
                    for suffix, (dims, dtypes) in expected.items():
                        if prefix + suffix not in headers:
                            raise ValueError(f"Missing saved CT tensor: {prefix + suffix}")
                        _, spec = headers[prefix + suffix]
                        if spec["shape"] != dims or spec["dtype"] not in dtypes:
                            raise ValueError(f"Unexpected CT layout: {prefix + suffix}: {spec}")
                    if fmt == "awq":
                        if read(prefix + "weight_shape").tolist() != [n, k]:
                            raise ValueError(f"Packed weight_shape value mismatch: {prefix}")
                    else:
                        values = [read(prefix + key).item() for key in ("weight_global_scale", "input_global_scale")]
                        if any(not math.isfinite(value) or value <= 0 for value in values):
                            raise ValueError(f"Invalid calibrated NVFP4 global scales: {prefix}")
                        globals_by_projection[projection] = values
                    checked += 1
                if fmt == "nvfp4" and any(not math.isclose(a, b, rel_tol=1e-5, abs_tol=1e-8)
                        for a, b in zip(globals_by_projection["gate_proj"], globals_by_projection["up_proj"])):
                    raise ValueError(f"vLLM fused gate/up require shared NVFP4 globals: layer={layer}, expert={expert}")
    return {"expert_projections_checked": checked, "format": fmt,
            "header_layout_check": "passed", "serving_ct_version": "0.17.0",
            "calibration_ct_version": "0.19.0", "runtime_validation": "not_run",
            "nvfp4_sidecar_export": "requires CT-specific adapter; ModelOpt exporter incompatible" if fmt == "nvfp4" else None}
