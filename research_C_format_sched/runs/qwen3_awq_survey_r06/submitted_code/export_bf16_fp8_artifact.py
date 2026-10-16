#!/usr/bin/env python3
"""Export real BF16 experts and native vLLM 0.26 online-FP8 expert artifacts.

Only the requested layer is read. The input is a pinned BF16 checkpoint
manifest produced by prepare_quantization.py; no synthetic weight fallback
exists. Explicit local-derived AWQ-smoothed BF16 masters are also accepted;
their content revision and original base commit remain separate identities.
FP8 uses Fp8PerTensorOnlineMoEMethod.process_weights_after_loading:
gate and up are concatenated BEFORE one per-expert tensor quantization.
This differs from separately quantized serialized FP8 gate/up checkpoints.

Examples (GPU execution is scheduled separately by the experiment owner)::

  python code/export_bf16_fp8_artifact.py \
    --manifest runs/olmoe_checkpoint/checkpoint_manifest.json --layer 0 \
    --formats bf16 fp8 --out-dir runs/olmoe_real_layer0

  python code/export_bf16_fp8_artifact.py --manifest checkpoint_manifest.json \
    --checkpoint-dir /data/copied-checkpoint --layer 0 --inspect-only

--inspect-only uses only the Python standard library, config/index and
safetensors headers. It can use <shard>.header.json copies if payload files
are absent; these are explicitly not payload/hash verification or exports.
Actual exports verify SHA-256 of every source shard used by this layer.

Each artifact also saves auxiliary_tensors.post_attention_layernorm_weight.
The activation_basis_id hashes a canonical RMSNorm descriptor containing the
raw norm-vector hash, dtype/shape and exact epsilon. It describes this layer's
local normalization transform, not equality of entire upstream activations
or model quality. No low-bit input adapter is implemented here.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import inspect
import json
import math
from pathlib import Path
import re
import struct


PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
MODEL_TYPES = {"olmoe": "olmoe", "qwen3": "qwen3_moe"}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def resolve_child(directory, name):
    path = Path(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Expected a relative checkpoint shard name: {name}")
    return directory / path


def read_header(path, *, allow_copy=False):
    """Read actual safetensors metadata without torch or loading tensor values."""
    if path.is_file():
        with path.open("rb") as stream:
            raw = stream.read(8)
            if len(raw) != 8:
                raise ValueError(f"Truncated safetensors file: {path}")
            length = struct.unpack("<Q", raw)[0]
            if not 2 <= length <= min(100 << 20, path.stat().st_size - 8):
                raise ValueError(f"Invalid safetensors header length: {path}")
            header = json.loads(stream.read(length))
        return header, 8 + length, "safetensors_header"
    copy = path.with_name(path.name + ".header.json")
    if allow_copy and copy.is_file():
        return json.loads(copy.read_text()), None, "header_copy_no_payload"
    raise FileNotFoundError(path)


def activation_basis_descriptor(model_type, layer, hidden_size, norm_header, norm_hash, eps):
    """The JSON encoding and raw-byte hash are shared by BF16/FP8 exports."""
    descriptor = {
        "schema": "rmsnorm_input_basis_v1", "model_type": model_type,
        "layer": layer, "hidden_size": hidden_size,
        "normalization": "RMSNorm: x * rsqrt(mean(x**2) + epsilon) * weight",
        "epsilon_hex": float(eps).hex(),
        "weight_dtype": norm_header["dtype"], "weight_shape": norm_header["shape"],
        "weight_raw_bytes_sha256": norm_hash,
        "hidden_axis": "checkpoint_order_no_additional_permutation_or_scaling",
    }
    identity = "rmsnorm_input_basis_v1:sha256:" + hashlib.sha256(canonical_json(descriptor).encode()).hexdigest()
    return descriptor, identity


def inspect_checkpoint(manifest_path, directory_override, layer, *, allow_header_copies=False):
    manifest = json.loads(manifest_path.read_text())
    identity = manifest["identity"]
    model = manifest["model_key"]
    if model not in MODEL_TYPES or manifest["format"] != "bf16":
        raise ValueError("Expected an olmoe/qwen3 BF16 checkpoint manifest")
    revision = identity.get("revision", "")
    derived = manifest.get("checkpoint_kind") == "local_derived"
    if derived:
        content = {key: identity[key] for key in ("config_sha256", "weight_files")}
        expected_revision = "sha256:" + hashlib.sha256(canonical_json(content).encode()).hexdigest()
        if (manifest.get("derivation", {}).get("method") != "awq_smoothed_bf16_master" or
                not identity.get("repo_id", "").startswith("local:") or revision != expected_revision):
            raise ValueError("Derived BF16 must be an explicit AWQ-smoothed master with a verified content identity")
        base_checkpoint = manifest.get("base_checkpoint")
        base_revision = manifest.get("base_checkpoint_revision", "")
        if (not manifest.get("exact_base_revision_known") or
                not re.fullmatch(r"[0-9a-fA-F]{40}", base_revision) or
                base_checkpoint != identity.get("base_model") or
                manifest["derivation"].get("source_checkpoint") != base_checkpoint or
                manifest["derivation"].get("source_revision") != base_revision):
            raise ValueError("Derived BF16 must retain the exact original source/base identity")
        if str(layer) not in (manifest.get("activation_bases") or {}):
            raise ValueError("Derived master lacks the saved layer's activation-basis evidence")
    else:
        if not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
            raise ValueError("Manifest must identify a fixed 40-hex source commit, not a moving/unknown revision")
        if not identity.get("repo_id") or identity.get("base_model") != identity["repo_id"]:
            raise ValueError("BF16 source repository and base_model must explicitly identify the same checkpoint")
        base_checkpoint, base_revision = identity["repo_id"], revision
    directory = (directory_override or Path(manifest["checkpoint_path"])).resolve()
    config_path = directory / "config.json"
    if sha256(config_path) != identity["config_sha256"]:
        raise ValueError("Source config SHA-256 differs from the pinned manifest")
    raw_config = json.loads(config_path.read_text())
    config = raw_config.get("text_config", raw_config)
    if config["model_type"] != MODEL_TYPES[model]:
        raise ValueError("Config model_type differs from manifest model_key")
    if raw_config.get("quantization_config") or config.get("quantization_config") or manifest.get("quantization_config"):
        raise ValueError("This exporter requires unquantized BF16 checkpoint weights")
    shape = {"d": int(config["hidden_size"]),
             "intermediate": int(config["intermediate_size"] if model == "olmoe" else config["moe_intermediate_size"]),
             "experts": int(config["num_experts"]), "top_k": int(config["num_experts_per_tok"])}
    if any(manifest["shape"].get(k) != v for k, v in shape.items()):
        raise ValueError("Config expert shape differs from the source manifest")
    if not 0 <= layer < int(config["num_hidden_layers"]):
        raise ValueError("Requested layer is outside the source model")
    if config.get("hidden_act", "silu") != "silu":
        raise ValueError("The artifact backend only supports gated SiLU")
    # Installed vLLM models/olmoe.py uses a literal 1e-5; the original OLMoE
    # config has no rms_norm_eps field. Qwen3 reads the config's epsilon.
    eps = 1e-5 if model == "olmoe" else float(config["rms_norm_eps"])
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("Expected a finite positive RMSNorm epsilon")
    files = {record["path"]: record for record in identity["weight_files"]}
    index_path = directory / "model.safetensors.index.json"
    if index_path.is_file():
        mapping = json.loads(index_path.read_text())["weight_map"]
        index_hash = sha256(index_path)
    else:
        mapping, index_hash = {}, None
        for name in files:
            header, _, _ = read_header(resolve_child(directory, name), allow_copy=allow_header_copies)
            mapping.update({key: name for key in header if key != "__metadata__"})
    prefix = f"model.layers.{layer}"
    norm_name = prefix + ".post_attention_layernorm.weight"
    expected = {}
    for expert in range(shape["experts"]):
        for projection in PROJECTIONS:
            dims = [shape["d"], shape["intermediate"]] if projection == "down_proj" else [shape["intermediate"], shape["d"]]
            expected[f"{prefix}.mlp.experts.{expert}.{projection}.weight"] = dims
    expected[norm_name] = [shape["d"]]
    missing = sorted(set(expected) - set(mapping))
    if missing:
        raise ValueError(f"Source index lacks {len(missing)} required tensors: {missing[:4]}")
    shards = sorted({mapping[name] for name in expected})
    headers, offsets, header_sources, source_files = {}, {}, {}, []
    for name in shards:
        if name not in files:
            raise ValueError(f"Required shard is absent from the pinned manifest: {name}")
        record = files[name]
        if not re.fullmatch(r"[0-9a-fA-F]{64}", record.get("sha256", "")):
            raise ValueError(f"Manifest lacks a SHA-256 for {name}")
        path = resolve_child(directory, name)
        headers[name], offsets[name], header_sources[name] = read_header(path, allow_copy=allow_header_copies)
        if path.is_file() and path.stat().st_size != record["bytes"]:
            raise ValueError(f"Shard size differs from source manifest: {name}")
        source_files.append({**record, "header_source": header_sources[name]})
    for name, dims in expected.items():
        header = headers[mapping[name]][name]
        dtype = header["dtype"]
        if header["shape"] != dims or dtype not in (("BF16", "F32") if name == norm_name else ("BF16",)):
            raise ValueError(f"Unexpected checkpoint tensor {name}: {header}")
        start, end = header["data_offsets"]
        if not isinstance(start, int) or not isinstance(end, int) or start < 0 or end <= start:
            raise ValueError(f"Invalid safetensors offsets for {name}")
        if end - start != math.prod(dims) * (4 if dtype == "F32" else 2):
            raise ValueError(f"Tensor byte extent disagrees with dtype/shape: {name}")
        shard = mapping[name]
        if offsets[shard] is not None and offsets[shard] + end > files[shard]["bytes"]:
            raise ValueError(f"Tensor bytes exceed source shard: {name}")
    report = {"schema": 1, "model": model, "layer": layer, "shape": shape,
              "checkpoint": identity["repo_id"], "checkpoint_revision": revision,
              "checkpoint_kind": "local_derived" if derived else "pinned_hub_checkpoint",
              "base_checkpoint": base_checkpoint, "base_checkpoint_revision": base_revision,
              "checkpoint_derivation": manifest.get("derivation") if derived else None,
              "checkpoint_dir": str(directory), "manifest_sha256": sha256(manifest_path),
              "config_sha256": identity["config_sha256"], "index_sha256": index_hash,
              "selected_expert_tensors": 3 * shape["experts"], "norm_tensor": norm_name,
              "rms_norm_eps": eps,
              "rms_norm_eps_source": "vLLM 0.26 OlmoeDecoderLayer literal 1e-5" if model == "olmoe" else "config.rms_norm_eps",
              "source_files": source_files,
              "payload_sha256_verified": False, "header_shapes_and_dtypes_checked": True,
              "expert_layout": "gate_proj then up_proj along output rows; down_proj unchanged",
              "source_checkpoint_fingerprint": manifest.get("checkpoint_fingerprint")}
    return manifest, config, mapping, expected, report


def load_source(manifest, mapping, expected, report):
    import torch
    from safetensors import safe_open

    directory = Path(report["checkpoint_dir"])
    for record in report["source_files"]:
        if sha256(resolve_child(directory, record["path"])) != record["sha256"]:
            raise ValueError(f"Source payload SHA-256 mismatch: {record['path']}")
    report["payload_sha256_verified"] = True
    shape = report["shape"]
    e, d, n = shape["experts"], shape["d"], shape["intermediate"]
    tensors = {"w1": torch.empty((e, 2 * n, d), dtype=torch.bfloat16),
               "w2": torch.empty((e, d, n), dtype=torch.bfloat16)}
    digest, individual_hashes = hashlib.sha256(), {}
    with ExitStack() as stack:
        handles = {name: stack.enter_context(safe_open(resolve_child(directory, name), framework="pt", device="cpu"))
                   for name in {mapping[key] for key in expected}}
        def read(name):
            value = handles[mapping[name]].get_tensor(name).contiguous()
            if list(value.shape) != expected[name] or not bool(torch.isfinite(value).all()):
                raise ValueError(f"Invalid tensor payload: {name}")
            raw = value.reshape(-1).view(torch.uint8).numpy().tobytes()
            descriptor = {"name": name, "shape": list(value.shape), "dtype": str(value.dtype)}
            digest.update(canonical_json(descriptor).encode() + b"\n")
            digest.update(raw)
            individual_hashes[name] = hashlib.sha256(raw).hexdigest()
            return value
        for expert in range(e):
            prefix = f"model.layers.{report['layer']}.mlp.experts.{expert}"
            for projection, dest in (("gate_proj", tensors["w1"][expert, :n]),
                                     ("up_proj", tensors["w1"][expert, n:]),
                                     ("down_proj", tensors["w2"][expert])):
                value = read(prefix + "." + projection + ".weight")
                if value.dtype != torch.bfloat16:
                    raise ValueError("Expert source payload is not BF16")
                dest.copy_(value)
                if not torch.equal(value, dest):
                    raise RuntimeError("Lossless expert layout copy check failed")
        norm = read(report["norm_tensor"])
    norm_header = {"dtype": "BF16" if norm.dtype == torch.bfloat16 else "F32", "shape": list(norm.shape)}
    descriptor, basis_id = activation_basis_descriptor(MODEL_TYPES[report["model"]], report["layer"],
        d, norm_header, individual_hashes[report["norm_tensor"]], report["rms_norm_eps"])
    if manifest.get("checkpoint_kind") == "local_derived":
        declared = manifest["activation_bases"][str(report["layer"])]
        if (declared["activation_basis_id"] != basis_id or
                declared["activation_basis_descriptor"] != descriptor):
            raise ValueError("Saved derived-master basis does not match actual source norm bytes/epsilon")
    report.update(selected_tensor_payload_sha256=digest.hexdigest(),
                  selected_tensor_hash_order="expert, gate/up/down; post_attention_layernorm last",
                  source_tensor_raw_bytes_sha256=individual_hashes,
                  activation_basis_id=basis_id, activation_basis_descriptor=descriptor,
                  activation_basis_verified=True,
                  activation_basis_verification_scope="source norm bytes, epsilon and canonical descriptor; not upstream activation equality",
                  activation_basis_norm_tensor_location="auxiliary_tensors.post_attention_layernorm_weight")
    return tensors, norm


def quantize_online_fp8(tensors, device):
    """Execute the installed online method, capturing its native Triton layout."""
    import torch
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.quantization.online.fp8 import Fp8PerTensorOnlineMoEMethod
    from vllm.model_executor.layers.fused_moe.oracle.fp8 import Fp8MoeBackend, convert_to_fp8_moe_kernel_format

    version = importlib.metadata.version("vllm")
    if version.split("+")[0] != "0.26.0":
        raise RuntimeError(f"Expected validated vLLM 0.26.0 interfaces, got {version}")
    if torch.device(device).type != "cuda":
        raise ValueError("Native vLLM online FP8 export requires the scheduled CUDA device")
    layer = torch.nn.Module()
    layer.w13_weight = torch.nn.Parameter(tensors["w1"].to(device), requires_grad=False)
    layer.w2_weight = torch.nn.Parameter(tensors["w2"].to(device), requires_grad=False)
    layer.num_experts = layer.local_num_experts = tensors["w1"].shape[0]
    layer.intermediate_size_per_partition = tensors["w2"].shape[-1]
    layer.weight_block_size = None

    class CaptureTritonWeights:
        # The native online method calls only this setup hook after quantizing.
        # Preserve its conversion while omitting live runner/kernel installation.
        def _setup_kernel(self, layer, w13, w2, w13_scale, w2_scale,
                          w13_input_scale, w2_input_scale):
            if w13_input_scale is not None or w2_input_scale is not None:
                raise RuntimeError("Expected dynamic per-tensor activation scaling")
            self.result = convert_to_fp8_moe_kernel_format(
                fp8_backend=Fp8MoeBackend.TRITON, layer=layer,
                w13=w13, w2=w2, w13_scale=w13_scale, w2_scale=w2_scale,
                w13_input_scale=None, w2_input_scale=None)

    capture = CaptureTritonWeights()
    native_method = Fp8PerTensorOnlineMoEMethod.process_weights_after_loading
    with torch.inference_mode():
        native_method(capture, layer)
        converted = dict(zip(("w1", "w2", "w1_scale", "w2_scale"), capture.result))
        for name in ("w1", "w2"):
            if converted[name].dtype != torch.float8_e4m3fn or converted[name].shape != tensors[name].shape:
                raise ValueError("Native online FP8 output does not match this Triton artifact contract")
            if not converted[name].is_contiguous() or not bool(torch.isfinite(converted[name].float()).all()):
                raise ValueError("Unexpected native FP8 weight layout or nonfinite payload")
            scale = converted[name + "_scale"]
            if tuple(scale.shape) != (layer.num_experts,) or scale.dtype != torch.float32 or not bool(
                    (torch.isfinite(scale) & (scale > 0)).all()):
                raise ValueError("Native scale is not one positive FP32 scalar per expert")
        result = {key: value.detach().cpu().contiguous() for key, value in converted.items()}
    torch.cuda.synchronize(device)
    evidence = {}
    for fn in (native_method, convert_to_fp8_moe_kernel_format, ops.scaled_fp8_quant):
        source = inspect.getsourcefile(fn)
        evidence[fn.__module__ + "." + fn.__qualname__] = {
            "source_file": source, "source_file_sha256": sha256(source),
            "function_source_sha256": hashlib.sha256(inspect.getsource(fn).encode()).hexdigest()}
    return result, {"vllm_version": version, "torch_version": str(torch.__version__),
        "native_conversion": evidence, "native_online_method_invoked": True,
        "runtime_runner_installed": False, "backend_layout_conversion": "Fp8MoeBackend.TRITON",
        "gate_up_quantization_rule": "concatenate BF16 gate/up first; quantize entire w13 per expert once",
        "weight_scale_contract": "FP32 [E] for merged gate/up and separately FP32 [E] for down",
        "activation_quantization": "dynamic per-tensor FP8; no static activation scales; no block scaling"}


def save_artifact(path, fmt, tensors, norm, report, extra):
    import torch
    if path.exists() or path.with_suffix(path.suffix + ".json").exists():
        raise FileExistsError(f"Keep prior artifacts intact; choose a new output directory: {path}")
    metadata = {**report, **extra, "schema": 1, "format": fmt, "layout": "vllm_triton",
        "base_checkpoint": report["base_checkpoint"], "base_checkpoint_revision": report["base_checkpoint_revision"],
        "exact_base_revision_known": True, "weight_source": "checkpoint", "quality_claim": False,
        "requested_backend": "triton", "auto_backend_selection": False,
        "calibration_method": (("AWQ-smoothed unquantized BF16 master" if report["checkpoint_kind"] == "local_derived"
                                else "not applicable: original BF16") if fmt == "bf16" else
                               "native vLLM online FP8 per-tensor conversion from content-verified BF16"),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "tensor_manifest": {key: {"shape": list(value.shape), "dtype": str(value.dtype),
                                   "bytes": value.numel() * value.element_size()} for key, value in tensors.items()},
        "input_adapter_implemented": False, "numeric_validation": "not_run; requires separate scheduled GPU check"}
    torch.save({"metadata": metadata, "tensors": tensors,
                "auxiliary_tensors": {"post_attention_layernorm_weight": norm}}, path)
    summary = {**metadata, "artifact": str(path.resolve()), "artifact_sha256": sha256(path)}
    path.with_suffix(path.suffix + ".json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return {"format": fmt, "artifact": str(path), "sha256": summary["artifact_sha256"],
            "activation_basis_id": report["activation_basis_id"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, help="Override manifest path after copying the pinned files")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--formats", nargs="+", choices=("bf16", "fp8"), default=["bf16", "fp8"])
    parser.add_argument("--device", default="cuda:0", help="Used only for native FP8 conversion")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--inspect-only", action="store_true", help="CPU config/index/header checks only; no torch or GPU")
    args = parser.parse_args()
    manifest, config, mapping, expected, report = inspect_checkpoint(
        args.manifest, args.checkpoint_dir, args.layer, allow_header_copies=args.inspect_only)
    if args.inspect_only:
        print(json.dumps(report, indent=2))
        return
    if args.out_dir is None:
        parser.error("--out-dir is required for export")
    formats = list(dict.fromkeys(args.formats))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {fmt: args.out_dir / f"layer{args.layer}_{fmt}_triton.pt" for fmt in formats}
    for path in outputs.values():
        if path.exists() or path.with_suffix(path.suffix + ".json").exists():
            raise FileExistsError(f"Export would overwrite existing result: {path}")
    tensors, norm = load_source(manifest, mapping, expected, report)
    results = []
    if "bf16" in formats:
        results.append(save_artifact(outputs["bf16"], "bf16", tensors, norm, report, {}))
    if "fp8" in formats:
        fp8, extra = quantize_online_fp8(tensors, args.device)
        results.append(save_artifact(outputs["fp8"], "fp8", fp8, norm, report, extra))
    print(json.dumps({"exports": results}, indent=2))


if __name__ == "__main__":
    main()
