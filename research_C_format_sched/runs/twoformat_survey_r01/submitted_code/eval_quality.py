#!/usr/bin/env python3
"""Run lm-eval against the exact checkpoint used for serving/performance.

Default is a CPU-only plan. Add --run inside the shared GPU-lock launcher.
  python eval_quality.py --checkpoint-manifest /data/olmoe-identity.json --format bf16 --output runs/quality/olmoe-bf16
  python eval_quality.py --checkpoint-manifest /data/qwen-awq/checkpoint_manifest.json --output runs/quality/qwen-awq --run

Uses the official lm_eval.simple_evaluate API, avoiding changes between the old
``lm_eval --model ...`` and new ``lm-eval run --model ...`` command interfaces.
Interface references are in configs/models.json. Requires lm-eval's vLLM backend
already installed; this runner does not install or download model weights.
Task datasets may be downloaded by lm-eval at execution time. ``--limit`` means
smoke-only and is explicitly recorded, not a complete quality-table result.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

from prepare_quantization import REGISTRY, read_json, sha256, validate_config, write_json


def verify_checkpoint(manifest, *, hash_weights):
    root = Path(manifest["checkpoint_path"]).resolve()
    if sha256(root / "config.json") != manifest["identity"]["config_sha256"]:
        raise ValueError("checkpoint config changed since registration")
    for record in manifest["identity"]["weight_files"]:
        path = (root / record["path"]).resolve()
        if not path.is_relative_to(root) or not path.is_file() or path.stat().st_size != record["bytes"]:
            raise ValueError(f"missing/changed checkpoint shard {record['path']}")
        if hash_weights and sha256(path) != record["sha256"]:
            raise ValueError(f"weight hash changed for {record['path']}; quality/performance identity is lost")
    validate_config(read_json(root / "config.json"), manifest["model_key"], manifest["format"])
    return root


def build_plan(args):
    manifest = read_json(args.checkpoint_manifest)
    fmt = args.format or manifest["format"]
    if fmt != manifest["format"] and not (manifest["format"] == "bf16" and fmt == "fp8"):
        raise ValueError("format must match checkpoint; only BF16 -> runtime FP8 is an allowed override")
    defaults = read_json(REGISTRY)["quality_defaults"]
    root = verify_checkpoint(manifest, hash_weights=args.run)
    quantization = "fp8" if fmt == "fp8" else manifest["runtime_quantization"]
    model_args = {"pretrained": str(root), "tokenizer": str(root),
                  "dtype": "bfloat16", "tensor_parallel_size": 1,
                  "gpu_memory_utilization": args.gpu_memory_utilization,
                  "max_model_len": args.max_model_len, "kv_cache_dtype": "auto",
                  "trust_remote_code": False, "seed": args.seed}
    if quantization:
        model_args["quantization"] = quantization
    tasks = args.tasks.split(",") if args.tasks else defaults["tasks"]
    return {"schema_version": 1, "status": "planned", "model_key": manifest["model_key"],
            "format": fmt, "checkpoint_fingerprint": manifest["checkpoint_fingerprint"],
            "checkpoint_manifest": str(Path(args.checkpoint_manifest).resolve()),
            "identity": manifest["identity"], "calibration": manifest["calibration"],
            "weights_hashes_verified": args.run, "smoke_only": args.limit is not None,
            "call": {"model": "vllm", "model_args": model_args, "tasks": tasks,
                     "num_fewshot": args.num_fewshot, "batch_size": args.batch_size,
                     "limit": args.limit, "log_samples": True,
                     "apply_chat_template": args.apply_chat_template,
                     "random_seed": args.seed, "numpy_random_seed": args.seed,
                     "torch_random_seed": args.seed, "fewshot_random_seed": args.seed,
                     "bootstrap_iters": args.bootstrap_iters},
            "scope": "single-format model quality; dual-format generation requires its actual runtime adapter",
            "comparison_note": "Use identical tasks, prompt/template, fewshot, KV dtype and seeds for all arms; never substitute a different base release."}


def json_default(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, (set, tuple)):
        return list(value)
    raise TypeError(f"cannot serialize {type(value)}")


def main():
    defaults = read_json(REGISTRY)["quality_defaults"]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint-manifest", required=True)
    parser.add_argument("--format", choices=("bf16", "fp8", "awq", "nvfp4"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--tasks", help="comma-separated lm-eval task names")
    parser.add_argument("--num-fewshot", type=int, default=defaults["num_fewshot"])
    parser.add_argument("--seed", type=int, default=defaults["seed"])
    parser.add_argument("--batch-size", default=defaults["batch_size"])
    parser.add_argument("--max-model-len", type=int, default=defaults["max_model_len"])
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--bootstrap-iters", type=int, default=1000)
    parser.add_argument("--apply-chat-template", action="store_true")
    parser.add_argument("--limit", type=float, help="smoke-only per-task count or fraction")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if not 0 < args.gpu_memory_utilization < 1:
        parser.error("--gpu-memory-utilization must be between zero and one")
    plan = build_plan(args)
    if not args.run:
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        return
    # Weights are local and already validated. Prevent an unexpected alternate
    # model download while leaving dataset access available to the harness.
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import lm_eval
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    plan["versions"] = {}
    for package in ("lm_eval", "vllm", "torch", "transformers", "compressed-tensors"):
        try:
            plan["versions"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            plan["versions"][package] = None
    plan.update(status="running", start_unix_s=time.time())
    write_json(output / "run.json", plan)
    try:
        results = lm_eval.simple_evaluate(**plan["call"])
        if results is None:
            raise RuntimeError("lm-eval returned no results")
        (output / "results.json").write_text(json.dumps(results, indent=2,
                           ensure_ascii=False, default=json_default, allow_nan=False) + "\n")
        plan.update(status="complete", end_unix_s=time.time())
    except BaseException as error:
        plan.update(status="failed", end_unix_s=time.time(), error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(output / "run.json", plan)
    print(json.dumps({"status": plan["status"], "output": str(output.resolve()),
                      "checkpoint_fingerprint": plan["checkpoint_fingerprint"],
                      "smoke_only": plan["smoke_only"]}, indent=2))


if __name__ == "__main__":
    main()
