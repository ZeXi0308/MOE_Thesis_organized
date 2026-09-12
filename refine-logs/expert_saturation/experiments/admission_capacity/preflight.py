"""Offline resource preparation; never download or load model weights."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import re
import sys


def inspect_environment(model_id, revision, require_cuda=False):
    result = dict(model_id=str(model_id), requested_revision=revision, resolved_revision=None,
                  python=sys.version, require_cuda=require_cuda, blockers=[],
                  remaining_gpu_checks=["GPU process isolation", "full workload KV/activation headroom",
                                        "bounded pretrained CUDA execution"])
    blockers = result["blockers"]
    versions = {}
    for name in ("torch", "transformers", "tokenizers", "safetensors", "huggingface-hub", "numpy"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
            blockers.append(f"missing dependency: {name}")
    result["versions"] = versions
    result["version_validation"] = {
        "torch": dict(expected="2.8.*", status="CHECKED" if (versions["torch"] or "").startswith("2.8.") else "NOT_VALIDATED"),
        "transformers": dict(expected="4.57.6", status="CHECKED" if versions["transformers"] == "4.57.6" else "NOT_VALIDATED")}
    result["weights"] = dict(loaded=False, hashed=False, all_shards_present=False)
    try:
        from transformers import AutoConfig, AutoTokenizer, DynamicCache
        from transformers.utils.hub import cached_file
        result["dynamic_cache_from_legacy_cache"] = callable(getattr(DynamicCache, "from_legacy_cache", None))
        if not result["dynamic_cache_from_legacy_cache"]:
            blockers.append("DynamicCache.from_legacy_cache is unavailable")
        config_path = Path(cached_file(str(model_id), "config.json", revision=revision, local_files_only=True))
        config = AutoConfig.from_pretrained(str(model_id), revision=revision, local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(str(model_id), revision=revision, local_files_only=True)
        snapshot = config_path.parent
        resolved = getattr(config, "_commit_hash", None)
        if not resolved and snapshot.parent.name == "snapshots" and re.fullmatch(r"[0-9a-f]{40}", snapshot.name):
            resolved = snapshot.name
        result.update(resolved_revision=resolved, snapshot_path=str(snapshot), tokenizer_class=type(tokenizer).__name__)
        if resolved is None or (re.fullmatch(r"[0-9a-f]{40}", revision or "") and resolved != revision):
            blockers.append("requested model revision is unresolved or does not match the cached snapshot")
        fields = ("model_type", "num_hidden_layers", "num_experts", "num_experts_per_tok", "max_position_embeddings", "torch_dtype")
        result["model_config"] = {name: str(getattr(config, name, None)) if name == "torch_dtype" else getattr(config, name, None) for name in fields}
        if config.model_type != "olmoe" or any(not isinstance(getattr(config, name, None), int) or getattr(config, name) < 1 for name in fields[1:5]):
            blockers.append("cached config is not a valid OLMoE layer/expert/top-k/context configuration")
        elif config.num_experts_per_tok > config.num_experts:
            blockers.append("configured top-k exceeds the number of experts")
        index = json.loads((snapshot / "model.safetensors.index.json").read_text())
        names = sorted(set(index["weight_map"].values()))
        if not names or any(Path(name).is_absolute() or ".." in Path(name).parts for name in names):
            raise ValueError("empty or unsafe shard list")
        shards = [dict(name=name, exists=(snapshot / name).is_file(),
                       size_bytes=(snapshot / name).stat().st_size if (snapshot / name).is_file() else None) for name in names]
        result["weights"].update(shards=shards, total_shard_bytes=sum(row["size_bytes"] or 0 for row in shards),
                                 all_shards_present=all(row["exists"] and row["size_bytes"] > 0 for row in shards))
        if not result["weights"]["all_shards_present"]:
            blockers.append("missing or empty model shards: " + ", ".join(row["name"] for row in shards if not row["size_bytes"]))
    except Exception as exc:
        blockers.append(f"offline model/tokenizer/cache inspection: {type(exc).__name__}: {exc}")
    try:
        import torch
        result.update(cuda_available=torch.cuda.is_available(), torch_cuda=torch.version.cuda)
        if require_cuda:
            if not result["cuda_available"]:
                blockers.append("CUDA unavailable")
            elif torch.cuda.device_count() != 1:
                blockers.append("expose exactly one GPU with CUDA_VISIBLE_DEVICES")
            else:
                free, total = torch.cuda.mem_get_info(0)
                props = torch.cuda.get_device_properties(0)
                result["gpu"] = dict(name=props.name, capability=list(torch.cuda.get_device_capability(0)),
                                     bf16_supported=torch.cuda.is_bf16_supported(), memory_free_bytes=free,
                                     memory_total_bytes=total, weight_bytes_lower_bound=result["weights"].get("total_shard_bytes"))
                if not result["gpu"]["bf16_supported"]:
                    blockers.append("selected GPU does not support BF16")
                if free <= result["weights"].get("total_shard_bytes", 0):
                    blockers.append("free GPU memory does not exceed the serialized weight-size lower bound")
        else:
            result["remaining_gpu_checks"].insert(0, "single CUDA device, capability, BF16 and free-memory checks")
    except Exception as exc:
        blockers.append(f"torch/CUDA inspection: {type(exc).__name__}: {exc}")
    result["status"] = "BLOCKED" if blockers else ("GPU_PREFLIGHT_CHECKED" if require_cuda else "CPU_PREPARATION_CHECKED")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()
    report = inspect_environment(args.model_id, args.revision, args.require_cuda)
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    raise SystemExit(bool(report["blockers"]))
