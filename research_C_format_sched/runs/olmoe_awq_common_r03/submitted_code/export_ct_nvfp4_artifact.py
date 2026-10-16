#!/usr/bin/env python3
"""Export one calibrated OLMoE compressed-tensors NVFP4 layer for vLLM CUTLASS.

This is the CT loader path, NOT the ModelOpt scale convention. Both CT global
scales are reciprocated by the installed vLLM 0.26 loader before conversion.
The real loader, oracle and CUTLASS post-load scale fusion run unchanged; only
live kernel assembly is replaced with a capture hook. No model is loaded.

  python code/export_ct_nvfp4_artifact.py --manifest /data/ct/checkpoint_manifest.json \
    --layer 0 --inspect-only
  # The actual export needs a scheduled GPU: vLLM's scale swizzle calls .cuda().
  python code/export_ct_nvfp4_artifact.py --manifest /data/ct/checkpoint_manifest.json \
    --layer 0 --device cuda:0 --out runs/olmoe_layer0_ct_nvfp4.pt
  # CPU only; tiny artificial tensors are never saved as weight artifacts:
  python code/export_ct_nvfp4_artifact.py --self-test --source-root runs/source_snapshot/vllm

Only explicit local-derived calibrated manifests are accepted. Inspection is
stdlib-only and can use --allow-header-copies; it does not verify weight bytes.
Export verifies every selected shard's SHA-256 and the actual saved norm/router
bytes against the common-basis manifest. That basis describes the local RMSNorm
transform, not equality of upstream activations or a model-quality guarantee.
"""
from __future__ import annotations

import argparse
import ast
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import inspect
import json
import math
from pathlib import Path
import re
from types import FunctionType, SimpleNamespace

from export_bf16_fp8_artifact import (
    activation_basis_descriptor, canonical_json, read_header, resolve_child, sha256,
)
from calibration_common_basis import content_revision, raw_tensor_hash


PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
FIELDS = ("weight_packed", "weight_scale", "weight_global_scale", "input_global_scale")
SOURCE_FILES = {
    "ct_loader": "model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_w4a4_nvfp4.py",
    "oracle": "model_executor/layers/fused_moe/oracle/nvfp4.py",
    "prepare": "model_executor/layers/quantization/utils/flashinfer_fp4_moe.py",
    "swizzle": "model_executor/layers/quantization/utils/nvfp4_utils.py",
    "cutlass": "model_executor/layers/fused_moe/experts/cutlass_moe.py",
}


def _check_quantization(config):
    qc = config.get("quantization_config") or {}
    groups = qc.get("config_groups") or {}
    if (qc.get("quant_method") != "compressed-tensors" or
            qc.get("format") != "nvfp4-pack-quantized" or not groups):
        raise ValueError("Expected a compressed-tensors nvfp4-pack-quantized checkpoint")
    for group in groups.values():
        for field in ("weights", "input_activations"):
            spec = group.get(field) or {}
            if (spec.get("num_bits") != 4 or spec.get("type") != "float" or
                    spec.get("group_size") != 16 or spec.get("strategy") != "tensor_group" or
                    spec.get("symmetric") is not True):
                raise ValueError(f"Expected symmetric NVFP4 W4A4 tensor_group16: {field}")
        if group["weights"].get("dynamic") is not False:
            raise ValueError("Expected static calibrated NVFP4 weights")
        if group["input_activations"].get("dynamic") not in ("local", False):
            raise ValueError("Expected calibrated activation global scale (dynamic=local or false)")
    return qc


def inspect_checkpoint(manifest_path, directory_override, layer, *, allow_header_copies=False):
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    identity = manifest["identity"]
    calibration = manifest.get("calibration") or {}
    derivation = manifest.get("derivation") or {}
    if (manifest.get("model_key") != "olmoe" or manifest.get("format") != "nvfp4" or
            manifest.get("checkpoint_kind") != "local_derived"):
        raise ValueError("Expected an explicit local-derived OLMoE NVFP4 manifest")
    if (identity.get("revision") != content_revision(identity) or
            not str(identity.get("repo_id", "")).startswith("local:")):
        raise ValueError("Local checkpoint content revision is absent or inconsistent")
    base, revision = manifest.get("base_checkpoint"), manifest.get("base_checkpoint_revision", "")
    if (not manifest.get("exact_base_revision_known") or not base or
            not re.fullmatch(r"[0-9a-fA-F]{40}", revision) or
            identity.get("base_model") != base or
            derivation.get("source_checkpoint") != base or derivation.get("source_revision") != revision or
            derivation.get("method") != "awq_nvfp4_calibrated_compression"):
        raise ValueError("Missing/inconsistent fixed base identity or calibrated local derivation")
    method = calibration.get("method", "")
    if (not method or "rtn" in method.lower() or "nvfp4" not in method.lower() or
            not calibration.get("evidence")):
        raise ValueError("NVFP4 calibration method and evidence are required; RTN is unsupported")
    directory = Path(directory_override or manifest["checkpoint_path"]).resolve()
    config_path = directory / "config.json"
    if sha256(config_path) != identity["config_sha256"]:
        raise ValueError("Config SHA-256 differs from the calibrated manifest")
    config = json.loads(config_path.read_text())
    if config.get("model_type") != "olmoe" or config.get("hidden_act", "silu") != "silu":
        raise ValueError("Only vLLM OLMoE gated-SiLU expert layout is supported")
    qc = _check_quantization(config)
    if manifest.get("quantization_config") != qc:
        raise ValueError("Config and manifest quantization specifications disagree")
    shape = {"d": int(config["hidden_size"]), "intermediate": int(config["intermediate_size"]),
             "experts": int(config["num_experts"]), "top_k": int(config["num_experts_per_tok"])}
    if (any(shape[k] != manifest["shape"].get(k) for k in shape) or
            not 0 <= layer < int(config["num_hidden_layers"]) or
            not 0 < shape["top_k"] <= shape["experts"] or
            any(shape[k] <= 0 or shape[k] % 128 for k in ("d", "intermediate"))):
        raise ValueError("Invalid model/layer shape or unsupported unpadded CUTLASS dimensions")
    prefix = f"model.layers.{layer}"
    basis = (manifest.get("activation_bases") or {}).get(str(layer))
    norm_name, router_name = prefix + ".post_attention_layernorm.weight", prefix + ".mlp.gate.weight"
    if not basis or basis.get("norm_tensor") != norm_name or basis.get("router_tensor") != router_name:
        raise ValueError("Manifest lacks the saved norm/router common-basis evidence")
    files = {row["path"]: row for row in identity["weight_files"]}
    if len(files) != len(identity["weight_files"]):
        raise ValueError("Duplicate checkpoint shard in manifest")
    index = directory / "model.safetensors.index.json"
    headers, offsets, header_sources = {}, {}, {}

    def get_header(name):
        if name not in files:
            raise ValueError(f"Shard is absent from the calibrated manifest: {name}")
        if name not in headers:
            record = files[name]
            if not re.fullmatch(r"[0-9a-fA-F]{64}", record.get("sha256", "")):
                raise ValueError(f"Missing source shard SHA-256: {name}")
            path = resolve_child(directory, name)
            if path.is_file() and path.stat().st_size != record["bytes"]:
                raise ValueError(f"Source shard size differs from manifest: {name}")
            headers[name], offsets[name], header_sources[name] = read_header(path, allow_copy=allow_header_copies)
        return headers[name]

    if index.is_file():
        mapping = json.loads(index.read_text())["weight_map"]
    else:
        mapping = {}
        for name in files:
            for key in get_header(name):
                if key == "__metadata__":
                    continue
                if key in mapping:
                    raise ValueError(f"Duplicate source tensor: {key}")
                mapping[key] = name
    expected = {}
    for expert in range(shape["experts"]):
        for projection in PROJECTIONS:
            n, k = ((shape["d"], shape["intermediate"]) if projection == "down_proj"
                    else (shape["intermediate"], shape["d"]))
            key_prefix = f"{prefix}.mlp.experts.{expert}.{projection}."
            if any(key_prefix + suffix in mapping for suffix in ("weight", "weight_zero_point")):
                raise ValueError(f"Expected symmetric packed CT weights: {key_prefix}")
            for field, dims, dtype in (("weight_packed", [n, k // 2], "U8"),
                                      ("weight_scale", [n, k // 16], "F8_E4M3"),
                                      ("weight_global_scale", [1], "F32"),
                                      ("input_global_scale", [1], "F32")):
                expected[key_prefix + field] = (dims, (dtype,))
    expected[norm_name] = ([shape["d"]], ("BF16", "F32"))
    expected[router_name] = ([shape["experts"], shape["d"]], ("BF16", "F32"))
    missing = set(expected) - mapping.keys()
    if missing:
        raise ValueError(f"Missing {len(missing)} CT/basis tensors: {sorted(missing)[:4]}")
    item_sizes = {"U8": 1, "F8_E4M3": 1, "BF16": 2, "F32": 4}
    for key, (dims, dtypes) in expected.items():
        shard = mapping[key]
        spec = get_header(shard)[key]
        if spec["shape"] != dims or spec["dtype"] not in dtypes:
            raise ValueError(f"Unexpected CT header for {key}: {spec}")
        start, end = spec["data_offsets"]
        if (not isinstance(start, int) or not isinstance(end, int) or start < 0 or
                end - start != math.prod(dims) * item_sizes[spec["dtype"]] or
                (offsets[shard] is not None and offsets[shard] + end > files[shard]["bytes"])):
            raise ValueError(f"Invalid tensor byte extent: {key}")
    norm_header = get_header(mapping[norm_name])[norm_name]
    declared = basis["activation_basis_descriptor"]
    descriptor, basis_id = activation_basis_descriptor("olmoe", layer, shape["d"], norm_header,
        declared["weight_raw_bytes_sha256"], 1e-5)
    if descriptor != declared or basis_id != basis["activation_basis_id"]:
        raise ValueError("Declared basis is inconsistent with saved norm header/vLLM OLMoE epsilon")
    report = {"schema": 1, "model": "olmoe", "layer": layer, "shape": shape,
        "format": "nvfp4", "layout": "vllm_cutlass_nvfp4_processed",
        "checkpoint": identity["repo_id"], "checkpoint_revision": identity["revision"],
        "checkpoint_kind": "local_derived", "base_checkpoint": base,
        "base_checkpoint_revision": revision, "exact_base_revision_known": True,
        "checkpoint_dir": str(directory), "checkpoint_derivation": derivation,
        "manifest_path": str(manifest_path.resolve()), "manifest_sha256": sha256(manifest_path),
        "source_checkpoint_fingerprint": manifest.get("checkpoint_fingerprint"),
        "config_sha256": identity["config_sha256"], "index_sha256": sha256(index) if index.is_file() else None,
        "calibration_method": method, "calibration": calibration,
        "quantization_config": qc, "norm_tensor": norm_name, "router_tensor": router_name,
        "rms_norm_eps": 1e-5, "rms_norm_eps_source": "vLLM 0.26 OlmoeDecoderLayer literal 1e-5",
        "source_files": [{**files[name], "header_source": header_sources[name]}
                         for name in sorted({mapping[key] for key in expected})],
        "selected_tensor_count": len(expected), "header_shapes_and_dtypes_checked": True,
        "payload_sha256_verified": False, "activation_basis_verified": False,
        "expert_layout": "gate_proj then up_proj along output rows; packed low/high nibbles unchanged",
        "source_format": "compressed-tensors NVFP4 group16; calibrated global scales are inverse dequant scales",
        "runtime_validation": "not_run"}
    return manifest, mapping, expected, report


def load_source(manifest, mapping, expected, report):
    import torch
    from safetensors import safe_open

    directory = Path(report["checkpoint_dir"])
    for record in report["source_files"]:
        if sha256(resolve_child(directory, record["path"])) != record["sha256"]:
            raise ValueError(f"Source shard SHA-256 mismatch: {record['path']}")
    report["payload_sha256_verified"] = True
    e, n, d = (report["shape"][key] for key in ("experts", "intermediate", "d"))
    arrays = {}
    for proj, dims in (("w13", (e, 2 * n, d)), ("w2", (e, d, n))):
        arrays[proj + "_weight_packed"] = torch.empty((*dims[:-1], dims[-1] // 2), dtype=torch.uint8)
        arrays[proj + "_weight_scale"] = torch.empty((*dims[:-1], dims[-1] // 16), dtype=torch.float8_e4m3fn)
        for suffix in ("weight_global_scale", "input_global_scale"):
            arrays[proj + "_" + suffix] = torch.empty((e, 2) if proj == "w13" else (e,), dtype=torch.float32)
    digest, hashes = hashlib.sha256(), {}
    with ExitStack() as stack:
        handles = {name: stack.enter_context(safe_open(str(resolve_child(directory, name)), framework="pt", device="cpu"))
                   for name in {mapping[key] for key in expected}}

        def read(key):
            value = handles[mapping[key]].get_tensor(key).contiguous()
            raw = value.reshape(-1).view(torch.uint8).numpy().tobytes()
            hashes[key] = hashlib.sha256(raw).hexdigest()
            digest.update(canonical_json({"key": key, "dtype": str(value.dtype), "shape": list(value.shape)}).encode() + b"\n" + raw)
            if value.dtype != torch.uint8 and not bool(torch.isfinite(value.float()).all()):
                raise ValueError(f"Nonfinite source tensor: {key}")
            return value

        for expert in range(e):
            for idx, projection in enumerate(PROJECTIONS):
                for field in FIELDS:
                    key = f"model.layers.{report['layer']}.mlp.experts.{expert}.{projection}.{field}"
                    value = read(key)
                    target = arrays[("w2_" if idx == 2 else "w13_") + field][expert]
                    if idx != 2:
                        target = target[idx * n:(idx + 1) * n] if field in FIELDS[:2] else target[idx]
                    if field != "weight_packed":
                        values = value.float()
                        if bool((values < 0).any()) or (field.endswith("global_scale") and bool((values <= 0).any())):
                            raise ValueError(f"Invalid calibrated scale: {key}")
                    target.copy_(value.reshape(target.shape))
        norm, router = read(report["norm_tensor"]), read(report["router_tensor"])
    # Native loader warns then uses the gate weight scale. Reject that lossy
    # fallback; do not silently ignore a different calibrated up-proj scale.
    pair = arrays["w13_weight_global_scale"]
    if not torch.allclose(pair[:, 0], pair[:, 1]):
        raise ValueError("Gate/up weight_global_scale differ: native single-scale fusion would be lossy")
    basis = manifest["activation_bases"][str(report["layer"])]
    norm_header = {"shape": list(norm.shape), "dtype": "BF16" if norm.dtype == torch.bfloat16 else "F32"}
    descriptor, basis_id = activation_basis_descriptor("olmoe", report["layer"], d, norm_header,
        raw_tensor_hash(norm), report["rms_norm_eps"])
    if (descriptor != basis["activation_basis_descriptor"] or basis_id != basis["activation_basis_id"] or
            raw_tensor_hash(router) != basis["router_raw_bytes_sha256"]):
        raise ValueError("Actual source norm/router bytes differ from the calibrated common basis")
    report.update(selected_tensor_payload_sha256=digest.hexdigest(), source_tensor_raw_bytes_sha256=hashes,
        selected_tensor_hash_order="expert, gate/up/down, packed/blockscale/weight_global/input_global; norm; router",
        gate_up_weight_global_scale_allclose=True,
        gate_up_weight_global_scale_max_relative_difference=float(((pair[:, 0] - pair[:, 1]).abs() / pair[:, 0]).max()),
        gate_up_input_global_scale_allclose=bool(torch.allclose(arrays["w13_input_global_scale"][:, 0], arrays["w13_input_global_scale"][:, 1])),
        activation_basis_id=basis_id, activation_basis_descriptor=descriptor, activation_basis_verified=True,
        activation_basis_verification_scope="actual saved norm bytes, epsilon and hidden-axis convention; not upstream activation equality",
        activation_basis_norm_tensor_location="auxiliary_tensors.post_attention_layernorm_weight",
        router_raw_bytes_sha256=raw_tensor_hash(router), router_basis_compensation_bytes_verified=True)
    return arrays, {"post_attention_layernorm_weight": norm, "router_weight": router}


def _capture_loader(native_process, native_get_config, native_fuse, backend, experts_cls, layer):
    """Execute the native loader with live kernel construction omitted, locally.

    A new function globals dict replaces exactly one assembly function; there
    is no mutation/monkeypatch of the installed vLLM module or system files.
    """
    called = []

    def make_kernel(**kwargs):
        if kwargs["backend"] != backend or kwargs["experts_cls"] != experts_cls:
            raise RuntimeError("Unexpected backend in native CT kernel setup")
        called.append(True)
        return SimpleNamespace(fused_experts=SimpleNamespace(
            process_weights_after_loading=lambda layer: native_fuse(None, layer)))

    owner = SimpleNamespace(moe=SimpleNamespace(is_act_and_mul=True), nvfp4_backend=backend, experts_cls=experts_cls)
    owner.get_fused_moe_quant_config = lambda layer: native_get_config(owner, layer)
    capture = FunctionType(native_process.__code__,
        {**native_process.__globals__, "make_nvfp4_moe_kernel": make_kernel}, native_process.__name__)
    capture(owner, layer)
    if called != [True]:
        raise RuntimeError("Native CT loader did not execute the expected kernel setup hook once")
    qc = owner.moe_quant_config
    return {"w1": layer.w13_weight, "w2": layer.w2_weight,
            "w1_scale": layer.w13_weight_scale, "w2_scale": layer.w2_weight_scale,
            "g1_alphas": qc.g1_alphas, "g2_alphas": qc.g2_alphas,
            "a1_gscale": qc.a1_gscale, "a2_gscale": qc.a2_gscale}


def convert_native(arrays, shape, device):
    import torch
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.experts.cutlass_moe import CutlassExpertsFp4
    from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import NvFp4MoeBackend, convert_to_nvfp4_moe_kernel_format, make_nvfp4_moe_quant_config
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w4a4_nvfp4 import CompressedTensorsW4A4Nvfp4MoEMethod as CT
    from vllm.model_executor.layers.quantization.utils.flashinfer_fp4_moe import prepare_nvfp4_moe_layer_for_fi_or_cutlass
    from vllm.model_executor.layers.quantization.utils.nvfp4_utils import swizzle_blockscale

    versions = {name: importlib.metadata.version(name) for name in ("torch", "vllm", "compressed-tensors")}
    if versions["vllm"].split("+")[0] != "0.26.0" or versions["compressed-tensors"].split("+")[0] != "0.17.0":
        raise RuntimeError(f"Expected inference vLLM 0.26.0 / CT 0.17.0, got {versions}")
    if torch.device(device).type != "cuda":
        raise ValueError("Native vLLM scale swizzle requires CUDA; use --self-test for CPU-only source checks")
    methods = (CT.process_weights_after_loading, CT.get_fused_moe_quant_config,
               convert_to_nvfp4_moe_kernel_format, make_nvfp4_moe_quant_config,
               prepare_nvfp4_moe_layer_for_fi_or_cutlass, swizzle_blockscale,
               CutlassExpertsFp4.process_weights_after_loading)
    provenance = {method.__module__ + "." + method.__qualname__: {
        "source_file": inspect.getsourcefile(method), "source_file_sha256": sha256(inspect.getsourcefile(method)),
        "function_source_sha256": hashlib.sha256(inspect.getsource(method).encode()).hexdigest()} for method in methods}
    with torch.inference_mode(), torch.cuda.device(device):
        layer = torch.nn.Module()
        for name, value in arrays.items():
            layer.register_parameter(name, torch.nn.Parameter(value.to(device), requires_grad=False))
        layer.activation = MoEActivation.SILU
        layer.num_experts = shape["experts"]
        layer._expert_routing_tables = lambda: None
        result = _capture_loader(CT.process_weights_after_loading, CT.get_fused_moe_quant_config,
            CutlassExpertsFp4.process_weights_after_loading, NvFp4MoeBackend.VLLM_CUTLASS, CutlassExpertsFp4, layer)
        for name in ("g1_alphas", "g2_alphas", "a1_gscale", "a2_gscale"):
            value = result[name]
            if (list(value.shape) != [shape["experts"]] or value.dtype != torch.float32 or
                    not bool((torch.isfinite(value) & (value > 0)).all())):
                raise ValueError(f"Native CUTLASS output has invalid per-expert scale: {name}")
        result = {name: value.detach().cpu().contiguous() for name, value in result.items()}
        torch.cuda.synchronize(device)
    return result, {"versions": versions, "native_conversion_sources": provenance,
        "conversion_device": device, "native_ct_loader_executed": True,
        "live_kernel_assembly": "omitted; capture hook invokes native CUTLASS post-load fusion exactly once",
        "scale_convention": "CT globals reciprocated by native loader; a13=max(1/input_global, gate/up); g1=(1/weight_global_gate)*a13; a1=1/a13; g2=1/(weight_global_down*input_global_down); a2=input_global_down",
        "runtime_validation": "conversion_only; no MoE forward, numeric-quality or serving validation"}


def self_test(source_root):
    """Run source bodies on tiny CPU tensors; never import vLLM or touch CUDA.

    Remove only type annotations/imports and swizzle's final .cuda() device
    transfer. Kernel-assembly and quant-config containers are inert captures.
    The CT/oracle/prepare/fusion arithmetic and permutation remain source code.
    """
    import __future__
    from enum import Enum
    import torch

    source_root = Path(source_root)
    source_paths = {name: source_root / file for name, file in SOURCE_FILES.items()}
    ns = {"torch": torch, "round_up": lambda x, y: (x + y - 1) // y * y,
          "replace_parameter": lambda layer, name, value: setattr(layer, name, torch.nn.Parameter(value, requires_grad=False)),
          "nvfp4_moe_quant_config": lambda **kwargs: SimpleNamespace(**kwargs)}
    oracle_tree = ast.parse(source_paths["oracle"].read_text())
    enum_node = next(node for node in oracle_tree.body if isinstance(node, ast.ClassDef) and node.name == "NvFp4MoeBackend")
    ns["Enum"] = Enum
    exec(compile(ast.Module(body=[enum_node], type_ignores=[]), str(source_paths["oracle"]), "exec"), ns)
    Backend = ns["NvFp4MoeBackend"]
    ns["FLASHINFER_NVFP4_MOE_BACKENDS"] = [x for x in Backend if x.name.startswith("FLASHINFER_")]
    adaptations = []

    def extract(file_key, function, class_name=None, cpu_swizzle=False):
        tree = ast.parse(source_paths[file_key].read_text())
        body = tree.body if class_name is None else next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name).body
        node = next(n for n in body if isinstance(n, ast.FunctionDef) and n.name == function)
        node.decorator_list = []
        node.body = [stmt for stmt in node.body if not isinstance(stmt, (ast.Import, ast.ImportFrom))]
        if cpu_swizzle:
            class CpuTransfer(ast.NodeTransformer):
                def visit_Call(self, node):
                    self.generic_visit(node)
                    if isinstance(node.func, ast.Attribute) and node.func.attr == "cuda" and not node.args and not node.keywords:
                        adaptations.append("swizzle final .cuda() removed for CPU-only permutation check")
                        return node.func.value
                    return node
            node = CpuTransfer().visit(node)
        module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
        exec(compile(module, str(source_paths[file_key]), "exec", flags=__future__.annotations.compiler_flag), ns)
        return ns[function]

    extract("oracle", "is_global_sf_supported_for_nvfp4_backend")
    swizzle = extract("swizzle", "swizzle_blockscale", cpu_swizzle=True)
    extract("prepare", "prepare_nvfp4_moe_layer_for_fi_or_cutlass")
    extract("oracle", "convert_to_nvfp4_moe_kernel_format")
    extract("oracle", "make_nvfp4_moe_quant_config")
    cls = "CompressedTensorsW4A4Nvfp4MoEMethod"
    get_config = extract("ct_loader", "get_fused_moe_quant_config", cls)
    process = extract("ct_loader", "process_weights_after_loading", cls)
    fuse = extract("cutlass", "process_weights_after_loading", "CutlassExpertsFp4")
    if len(adaptations) != 1:
        raise RuntimeError("Expected exactly one native swizzle CUDA transfer")
    tests = []
    for equal_inputs in (True, False):
        e, n, d = 2, 128, 128
        layer = torch.nn.Module()
        # Artificial bytes/scales for conversion testing only; not calibration.
        originals = {}
        for prefix, dims in (("w13", (e, 2 * n, d)), ("w2", (e, d, n))):
            weight = (torch.arange(math.prod(dims) // 2).reshape(*dims[:-1], dims[-1] // 2) % 256).to(torch.uint8)
            scale = ((torch.arange(math.prod(dims) // 16).reshape(*dims[:-1], dims[-1] // 16) % 15) + 1).to(torch.float8_e4m3fn)
            originals[prefix] = (weight.clone(), scale.clone())
            setattr(layer, prefix + "_weight_packed", torch.nn.Parameter(weight, requires_grad=False))
            setattr(layer, prefix + "_weight_scale", torch.nn.Parameter(scale, requires_grad=False))
        wg1, wg2 = torch.tensor([[2., 2.], [4., 4.]]), torch.tensor([8., 16.])
        ag1 = torch.tensor([[2., 2.], [8., 8.]]) if equal_inputs else torch.tensor([[2., 4.], [16., 8.]])
        ag2 = torch.tensor([32., 64.])
        for name, value in (("w13_weight_global_scale", wg1), ("w2_weight_global_scale", wg2),
                            ("w13_input_global_scale", ag1), ("w2_input_global_scale", ag2)):
            setattr(layer, name, torch.nn.Parameter(value.clone(), requires_grad=False))
        layer.activation = SimpleNamespace(is_gated=True)
        layer._expert_routing_tables = lambda: None
        result = _capture_loader(process, get_config, fuse, Backend.VLLM_CUTLASS, "CPU test capture", layer)
        for key, expected in {"g1_alphas": 1 / (wg1[:, 0] * ag1.min(dim=1).values),
                              "g2_alphas": 1 / (wg2 * ag2), "a1_gscale": ag1.min(dim=1).values,
                              "a2_gscale": ag2}.items():
            torch.testing.assert_close(result[key], expected, rtol=0, atol=0)
        for source_prefix, out_prefix in (("w13", "w1"), ("w2", "w2")):
            weight, scale = originals[source_prefix]
            assert torch.equal(result[out_prefix], weight), "CUTLASS must preserve gate/up packed rows"
            actual = result[out_prefix + "_scale"].view(torch.uint8)
            # Independent coordinate mapping for native reshape/permute layout.
            expected = torch.empty_like(actual).flatten()
            bsize, msize, ksize = scale.shape
            for b in range(bsize):
                for m in range(msize):
                    for k in range(ksize):
                        flat = (((((b * (msize // 128) + m // 128) * (ksize // 4) + k // 4) * 32 + m % 32) * 4 + (m % 128) // 32) * 4 + k % 4)
                        expected[flat] = scale.view(torch.uint8)[b, m, k]
            assert torch.equal(actual.flatten(), expected), "CUTLASS blockscale permutation mismatch"
        tests.append("equal_gate_up_activation_globals" if equal_inputs else "unequal_gate_up_activation_globals_native_max_of_reciprocal")
    # The source swizzle pads both axes; verify padding/shape on a small odd size.
    padded = swizzle(torch.ones((2, 3, 5), dtype=torch.float8_e4m3fn))
    assert tuple(padded.shape) == (2, 128, 8) and int((padded.float() == 1).sum()) == 30
    return {"status": "passed", "device": "cpu", "gpu_execution": False, "weights": "tiny artificial test tensors; no artifact emitted",
            "tests": tests + ["packed_weight_identity", "blockscale_coordinate_mapping", "swizzle_padding"],
            "adaptations": adaptations + ["imports supplied from extracted source bodies; config/kernel construction captured"],
            "source_files": {key: {"path": str(path), "sha256": sha256(path)} for key, path in source_paths.items()},
            "limitation": "CPU source-body arithmetic/layout checks; installed imports, CUDA transfer and MoE numerical execution not validated"}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--inspect-only", action="store_true")
    parser.add_argument("--allow-header-copies", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[1] / "runs/source_snapshot/vllm")
    args = parser.parse_args()
    if args.self_test:
        if args.manifest or args.out:
            parser.error("--self-test neither reads a checkpoint nor writes a weight artifact")
        print(json.dumps(self_test(args.source_root), indent=2))
        return
    if not args.manifest or (not args.inspect_only and not args.out):
        parser.error("--manifest is required; export also requires --out")
    if args.allow_header_copies and not args.inspect_only:
        parser.error("Header copies may be used only for inspection")
    if args.out and args.out.exists():
        raise FileExistsError(f"Refusing to replace an existing artifact/report: {args.out}")
    manifest, mapping, expected, report = inspect_checkpoint(args.manifest, args.checkpoint_dir,
        args.layer, allow_header_copies=args.allow_header_copies)
    if args.inspect_only:
        report["mode"] = "header_inspection_only_no_payload_or_gpu"
        text = json.dumps(report, indent=2) + "\n"
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(text)
        print(text, end="")
        return
    arrays, auxiliary = load_source(manifest, mapping, expected, report)
    tensors, conversion = convert_native(arrays, report["shape"], args.device)
    import torch
    report.update(conversion, created_utc=datetime.now(timezone.utc).isoformat(),
                  exporter_sha256=sha256(Path(__file__)), synthetic_weights=False,
                  resident_weight_and_scale_bytes=sum(value.numel() * value.element_size() for value in tensors.values()))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Open exclusively; never overwrite an earlier run's artifact.
    with args.out.open("xb") as stream:
        torch.save({"metadata": report, "tensors": tensors, "auxiliary_tensors": auxiliary}, stream)
    print(json.dumps({"artifact": str(args.out.resolve()), "sha256": sha256(args.out), "metadata": report}, indent=2))


if __name__ == "__main__":
    main()
