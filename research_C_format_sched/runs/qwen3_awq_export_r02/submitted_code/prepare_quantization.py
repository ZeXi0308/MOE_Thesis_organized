#!/usr/bin/env python3
"""Prepare pinned calibrated checkpoints without installing dependencies.

Examples:
  python prepare_quantization.py catalog --model qwen3
  python prepare_quantization.py fetch --model qwen3 --format awq --output /data/qwen-awq --metadata-only
  python prepare_quantization.py register --model olmoe --format bf16 --path /data/olmoe --base-model allenai/OLMoE-1B-7B-0924 --output /data/olmoe-identity.json
  python prepare_quantization.py calibrate --source-manifest /data/olmoe-identity.json --format awq --calibration-jsonl train_text.jsonl --output /data/olmoe-awq

Calibrate uses llm-compressor 0.14.0 / compressed-tensors 0.19.0 /
transformers 5.17.0 in a separate calibration environment; the fixed vLLM
serving environment remains on compressed-tensors 0.17.0. It requires local
JSONL training texts and the GPU lock held by the launcher. The official 0.14.0
wheel registers OlmoeExperts linearization. Runtime export/load compatibility
still needs a smoke measurement in the serving environment.
NVFP4 also applies AWQ scaling; no data-free/RTN route is provided. Neither this
file nor eval_quality.py downloads anything or touches a GPU merely on import.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import shutil
import time

REGISTRY = Path(__file__).resolve().parents[1] / "configs" / "models.json"
CALIBRATION_PINS = {
    "llmcompressor": "0.14.0", "compressed-tensors": "0.19.0",
    "transformers": "5.17.0", "torch": "2.11.0", "triton": "3.6.0",
}
# This is the _moe_default_mappings pattern in the official 0.14.0 wheel.
# Ignoring the router for quantization does NOT exclude it from compensation:
# changing the norm scale must inversely change every consumer's input weights.
AWQ_MOE_MAPPINGS = [
    {"smooth_layer": "re:.*input_layernorm$", "balance_layers": ["re:.*q_proj$", "re:.*k_proj$", "re:.*v_proj$"]},
    {"smooth_layer": "re:.*v_proj$", "balance_layers": ["re:.*o_proj$"]},
    {"smooth_layer": "re:.*post_attention_layernorm$", "balance_layers": [
        "re:.*mlp.gate$", "re:.*mlp.experts.*.gate_proj$", "re:.*mlp.experts.*.up_proj$"]},
    {"smooth_layer": "re:.*up_proj$", "balance_layers": ["re:.*down_proj$"]},
]


def calibration_environment():
    actual = {name: importlib.metadata.version(name) for name in CALIBRATION_PINS}
    wrong = {name: actual[name] for name, version in CALIBRATION_PINS.items()
             if actual[name].split("+", 1)[0] != version}
    if wrong:
        raise RuntimeError(f"use the separate pinned calibration environment: {CALIBRATION_PINS}; mismatches: {wrong}")
    return actual


def effective_routers(model):
    """Small CPU snapshots of W_router diag(norm_weight), before softmax/top-k.

    For OLMoE/Qwen3 RMSNorm, this is the linear operator acting on the common
    normalized input. Comparing it checks the omitted-router-compensation bug,
    independent of changes elsewhere caused intentionally by quantization.
    """
    result = {}
    for name, norm in model.named_modules():
        if name.endswith(".post_attention_layernorm"):
            parent = name.rsplit(".", 1)[0]
            gate = model.get_submodule(parent + ".mlp.gate")
            # Copy to CPU first: this check is not a timed GPU workload.
            weight = gate.weight.detach().to("cpu").float()
            scale = norm.weight.detach().to("cpu").float()
            result[parent] = weight * scale.unsqueeze(0)
    return result


def check_router_compensation(before, after, storage_dtype):
    import torch
    if before.keys() != after.keys():
        raise RuntimeError("router set changed during calibration")
    tolerance = 4 * torch.finfo(storage_dtype).eps
    report = {}
    for name in before:
        relative = ((after[name] - before[name]).norm() /
                    before[name].norm().clamp_min(1e-30)).item()
        report[name] = {"relative_frobenius_error": relative,
                        "storage_roundoff_check_bound": tolerance}
        if not torch.isfinite(after[name]).all() or relative > tolerance:
            raise RuntimeError(f"router compensation changed {name}: relative error={relative}")
    return {"quantity": "W_router @ diag(post_attention_RMSNorm_weight)",
            "dtype": str(storage_dtype), "layers": report,
            "interpretation": "algebraic compensation within storage rounding; not a claim of identical top-k on near ties"}


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_config(config, model_key, fmt):
    expected = read_json(REGISTRY)["models"][model_key]
    shape = {"d": config.get("hidden_size"),
             "intermediate": config.get("moe_intermediate_size", config.get("intermediate_size")),
             "experts": config.get("num_experts", config.get("num_local_experts")),
             "top_k": config.get("num_experts_per_tok"), "layers": config.get("num_hidden_layers")}
    if config.get("model_type") != expected["model_type"] or shape != expected["shape"]:
        raise ValueError(f"checkpoint architecture differs: actual {shape}, expected {expected['shape']}")
    qc = config.get("quantization_config") or {}
    if fmt in ("bf16", "fp8"):
        if qc:
            raise ValueError("BF16/runtime-FP8 arms require the original unquantized checkpoint")
    elif fmt == "awq":
        if qc.get("quant_method") == "awq":
            if qc.get("bits") != 4:
                raise ValueError("AWQ checkpoint must have four-bit weights")
        else:
            groups = list(qc.get("config_groups", {}).values())
            if qc.get("quant_method") != "compressed-tensors" or not groups or not all(
                    g.get("weights", {}).get("num_bits") == 4 and
                    g.get("weights", {}).get("type") == "int" and
                    g.get("input_activations") is None for g in groups):
                raise ValueError("not a W4A16 compressed-tensors/AWQ checkpoint")
    elif fmt == "nvfp4":
        groups = list(qc.get("config_groups", {}).values())
        if qc.get("quant_algo") != "NVFP4" and not (groups and all(
                g.get("weights", {}).get("num_bits") == 4 and
                g.get("weights", {}).get("type") == "float" and
                g.get("weights", {}).get("group_size") == 16 for g in groups)):
            raise ValueError("config does not describe NVFP4 group16 weights")
    return shape, qc


def register(path, model_key, fmt, base_model, output, *, calibration=None,
             source_repo=None, revision=None):
    path = Path(path).resolve()
    config = read_json(path / "config.json")
    shape, qc = validate_config(config, model_key, fmt)
    if not base_model:
        raise ValueError("explicit base_model required; identical shape does not establish identity")
    registered_base = read_json(REGISTRY)["models"][model_key].get("base_model")
    if registered_base and base_model != registered_base:
        raise ValueError(f"this comparison requires base {registered_base}, not {base_model}")
    if fmt in ("awq", "nvfp4"):
        if not calibration or not calibration.get("method") or not calibration.get("evidence"):
            raise ValueError("four-bit checkpoints need calibration method and evidence")
        if "rtn" in calibration["method"].lower():
            raise ValueError("RTN is not permitted")
        if fmt == "awq" and "awq" not in calibration["method"].lower():
            raise ValueError("AWQ arm requires documented AWQ calibration")
    index = path / "model.safetensors.index.json"
    weight_names = sorted(set(read_json(index)["weight_map"].values())) if index.exists() else [
        p.name for p in path.glob("*.safetensors")]
    if not weight_names:
        raise ValueError("no safetensors weights found (metadata-only download is not a checkpoint)")
    files = []
    for name in weight_names:
        file = (path / name).resolve()
        if not file.is_relative_to(path) or not file.is_file():
            raise ValueError(f"missing or invalid weight shard: {name}")
        files.append({"path": name, "bytes": file.stat().st_size, "sha256": sha256(file)})
    identity = {"base_model": base_model, "repo_id": source_repo, "revision": revision,
                "config_sha256": sha256(path / "config.json"), "weight_files": files}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    result = {"schema_version": 1, "model_key": model_key, "format": fmt,
              "checkpoint_path": str(path), "shape": shape, "identity": identity,
              "checkpoint_fingerprint": fingerprint, "calibration": calibration,
              "quantization_config": qc,
              "runtime_quantization": "fp8" if fmt == "fp8" else qc.get("quant_method"),
              "created_unix_s": time.time(), "runtime_validation": "not_run"}
    write_json(output, result)
    return result


def fetch(args):
    from huggingface_hub import HfApi, snapshot_download
    spec = read_json(REGISTRY)["models"][args.model]["formats"][args.format]
    if not spec.get("repo_id") or not spec.get("revision"):
        raise ValueError("no pinned documented checkpoint in catalog; use local calibration")
    api = HfApi()
    info = api.model_info(spec["repo_id"], revision=spec["revision"], files_metadata=True)
    target = Path(args.output).resolve()
    target.mkdir(parents=True, exist_ok=True)
    suffixes = (".json", ".yaml", ".jinja", ".txt", ".model", ".md")
    selected = [s for s in info.siblings if s.rfilename.endswith(suffixes) or (
        not args.metadata_only and s.rfilename.endswith(".safetensors"))]
    if any(s.size is None for s in selected):
        raise RuntimeError("Hub did not supply file sizes; cannot safely plan this constrained disk")
    missing_bytes = sum(s.size for s in selected if not (target / s.rfilename).is_file()
                        or (target / s.rfilename).stat().st_size != s.size)
    free = shutil.disk_usage(target).free
    reserve = int(args.reserve_gib * 2**30)
    if free < missing_bytes + reserve:
        raise RuntimeError(f"need {missing_bytes / 2**30:.2f} GiB plus {args.reserve_gib} GiB reserve; free={free / 2**30:.2f} GiB")
    snapshot_download(spec["repo_id"], revision=info.sha, local_dir=str(target),
                      allow_patterns=[s.rfilename for s in selected], max_workers=2)
    shape, qc = validate_config(read_json(target / "config.json"), args.model, args.format)
    if args.metadata_only:
        return {"status": "metadata_only_no_weights", "repo_id": spec["repo_id"],
                "revision": info.sha, "shape": shape, "quantization_config": qc}
    return register(target, args.model, args.format, spec["base_model"],
                    target / "checkpoint_manifest.json", source_repo=spec["repo_id"], revision=info.sha,
                    calibration={"method": spec["calibration_method"],
                                 "details": spec["calibration"], "evidence": spec["sources"]})


def calibrate(args):
    # Explicit local input avoids silently downloading a 61 GB BF16 checkpoint.
    manifest = read_json(args.source_manifest)
    source = Path(manifest["checkpoint_path"])
    if manifest["format"] != "bf16":
        raise ValueError("calibration source must be registered BF16")
    if sha256(source / "config.json") != manifest["identity"]["config_sha256"]:
        raise ValueError("source config changed since registration")
    for record in manifest["identity"]["weight_files"]:
        if sha256(source / record["path"]) != record["sha256"]:
            raise ValueError("source weights changed since registration")
    validate_config(read_json(source / "config.json"), manifest["model_key"], "bf16")
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("output must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    estimate = read_json(REGISTRY)["models"][manifest["model_key"]]["storage_estimates_decimal_gb"][args.format] * 1e9
    if shutil.disk_usage(output).free < estimate + args.reserve_gib * 2**30:
        raise RuntimeError("insufficient free disk for compressed checkpoint and reserve")
    environment = calibration_environment()
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from llmcompressor import oneshot
    from llmcompressor.utils import load_context
    from llmcompressor.modifiers.transform.awq import AWQModifier
    from llmcompressor.modifiers.quantization import QuantizationModifier
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
    rows = [json.loads(line) for line in Path(args.calibration_jsonl).read_text().splitlines() if line.strip()]
    if len(rows) < args.samples or any(not isinstance(row.get("text"), str) for row in rows):
        raise ValueError("calibration JSONL needs at least --samples records with a text string")
    data = Dataset.from_list(rows).shuffle(seed=args.seed).select(range(args.samples))
    data = data.map(lambda row: tokenizer(row["text"], truncation=True, max_length=args.max_seq_length,
                                        add_special_tokens=False), remove_columns=data.column_names)
    with load_context():
        model = AutoModelForCausalLM.from_pretrained(source, local_files_only=True,
                    dtype=torch.bfloat16, device_map="auto")
    router_before = effective_routers(model)
    if len(router_before) != manifest["shape"]["layers"]:
        raise RuntimeError("router/norm mapping does not cover every model layer")
    recipe = [AWQModifier(mappings=AWQ_MOE_MAPPINGS), QuantizationModifier(targets="Linear",
              scheme="W4A16" if args.format == "awq" else "NVFP4",
              ignore=["lm_head", "re:.*mlp.gate$", "re:.*mlp.shared_expert_gate$"])]
    # Record whether every expert actually received data. A successful save alone
    # does not prove that a sparse model was calibrated rather than left unseen.
    counts, hooks = {}, []
    for name, module in model.named_modules():
        if re.search(r"\.experts\.\d+\.gate_proj$", name):
            counts[name] = 0
            def record(module, inputs, name=name):
                if inputs and inputs[0].numel():
                    counts[name] += inputs[0].numel() // inputs[0].shape[-1]
            hooks.append(module.register_forward_pre_hook(record))
    expected = manifest["shape"]["experts"] * manifest["shape"]["layers"]
    if len(counts) != expected:
        raise RuntimeError(f"expert calibration mapping unverified: observed {len(counts)} modules, expected {expected}; no RTN fallback")
    try:
        oneshot(model=model, dataset=data, recipe=recipe, max_seq_length=args.max_seq_length,
                num_calibration_samples=args.samples)
    finally:
        for hook in hooks:
            hook.remove()
        write_json(output / "calibration_coverage.json", counts)
    if any(count == 0 for count in counts.values()):
        raise RuntimeError("one or more experts received no calibration tokens; checkpoint not accepted")
    router_check = check_router_compensation(router_before, effective_routers(model), torch.bfloat16)
    write_json(output / "router_compensation.json", router_check)
    model.save_pretrained(output, save_compressed=True)
    tokenizer.save_pretrained(output)
    provenance = {"method": "AWQ" if args.format == "awq" else "AWQ plus NVFP4",
                  "evidence": {"calibration_jsonl_sha256": sha256(args.calibration_jsonl),
                               "samples": args.samples, "max_seq_length": args.max_seq_length,
                               "seed": args.seed, "expert_token_counts": counts,
                               "environment": environment, "awq_mappings": AWQ_MOE_MAPPINGS,
                               "router_compensation": router_check,
                               "source_checkpoint_fingerprint": manifest["checkpoint_fingerprint"]}}
    return register(output, manifest["model_key"], args.format, manifest["identity"]["base_model"],
                    output / "checkpoint_manifest.json", calibration=provenance)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    catalog = sub.add_parser("catalog")
    catalog.add_argument("--model", choices=("olmoe", "qwen3"), required=True)
    fetcher = sub.add_parser("fetch")
    fetcher.add_argument("--model", choices=("olmoe", "qwen3"), required=True)
    fetcher.add_argument("--format", choices=("awq", "nvfp4"), required=True)
    fetcher.add_argument("--output", required=True)
    fetcher.add_argument("--metadata-only", action="store_true")
    fetcher.add_argument("--reserve-gib", type=float, default=2.0)
    reg = sub.add_parser("register")
    reg.add_argument("--model", choices=("olmoe", "qwen3"), required=True)
    reg.add_argument("--format", choices=("bf16", "fp8", "awq", "nvfp4"), required=True)
    reg.add_argument("--path", required=True)
    reg.add_argument("--base-model", required=True)
    reg.add_argument("--output", required=True)
    reg.add_argument("--source-repo", help="Verified source repository, not inferred from shape")
    reg.add_argument("--revision", help="Verified source commit or immutable revision")
    reg.add_argument("--calibration-json", help="JSON with method and evidence for a local four-bit checkpoint")
    cal = sub.add_parser("calibrate")
    cal.add_argument("--source-manifest", required=True)
    cal.add_argument("--format", choices=("awq", "nvfp4"), required=True)
    cal.add_argument("--calibration-jsonl", required=True)
    cal.add_argument("--output", required=True)
    cal.add_argument("--samples", type=int, default=512)
    cal.add_argument("--max-seq-length", type=int, default=2048)
    cal.add_argument("--seed", type=int, default=42)
    cal.add_argument("--reserve-gib", type=float, default=2.0)
    args = parser.parse_args()
    if hasattr(args, "reserve_gib") and args.reserve_gib < 0:
        parser.error("--reserve-gib cannot be negative")
    if args.command == "calibrate" and (args.samples <= 0 or args.max_seq_length <= 0):
        parser.error("calibration sample count and sequence length must be positive")
    if args.command == "catalog":
        result = read_json(REGISTRY)["models"][args.model]
    elif args.command == "fetch":
        result = fetch(args)
    elif args.command == "register":
        result = register(args.path, args.model, args.format, args.base_model, args.output,
                          calibration=read_json(args.calibration_json) if args.calibration_json else None,
                          source_repo=args.source_repo, revision=args.revision)
    else:
        result = calibrate(args)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
