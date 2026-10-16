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

--worker-config installs real per-layer sidecars with DualFormatWorker. It forces
eager execution and requires --lock-status for --run. The host checkpoint keeps
its own nonexpert precision; selecting FP8 sidecars does not quantize that host.
Single policies infer bf16/fp8/awq/nvfp4; dual policies infer dual_fp8_awq or
dual_fp8_nvfp4. Example JSONs in configs/quality_worker_*.example.json are layer
templates: supply real artifacts/basis IDs and enumerate EVERY MoE layer.
The default five tasks score prompt loglikelihood (prefill), not mixed-decode
generation quality. No model/worker is imported by the read-only plan path.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import time

from prepare_quantization import REGISTRY, read_json, sha256, validate_config, write_json


WORKER_CLASS = "dual_format_worker.DualFormatWorker"
MC_TASKS = {"hellaswag", "arc_easy", "arc_challenge", "winogrande", "piqa"}


def worker_plan(path, output, *, hash_weights):
    """Freeze resolved inputs without importing torch/vLLM or loading a model.

    Content hashes identify the actual sidecar files, not just their filenames or
    the host checkpoint. Basis/shape/source validation remains in the real worker;
    its installation report is retained even when model initialization fails.
    """
    path = Path(path).resolve()
    config = read_json(path)
    if config.get("schema_version") != 1:
        raise ValueError("worker config requires schema_version=1")
    if config.get("mechanism_only", False) is not False or config.get("allow_unverified_source", False) is not False:
        raise ValueError("quality evaluation rejects mechanism_only and unverified-source sidecars")
    policy = config.get("policy")
    if policy == "single":
        fmt = config.get("single_format")
        if fmt not in ("bf16", "fp8", "awq", "nvfp4"):
            raise ValueError("single worker requires single_format=bf16/fp8/awq/nvfp4")
        if any(key in config for key in ("threshold", "lowbit_format")):
            raise ValueError("single worker must not carry dual-policy fields")
        if config.get("release_original_weights", True) is not True:
            raise ValueError("single-residency worker must release the original expert weights")
        weights = {"weights": fmt}
        label = fmt
    elif policy in ("threshold", "phase", "expert_split"):
        fmt = config.get("lowbit_format")
        if fmt not in ("awq", "nvfp4") or (policy == "expert_split" and fmt != "awq"):
            raise ValueError("dual lowbit_format must be awq/nvfp4; expert_split supports awq only")
        if type(config.get("threshold")) is not int or config["threshold"] < 1:
            raise ValueError("dual worker requires a positive integer threshold")
        if "single_format" in config:
            raise ValueError("dual worker must not carry single_format")
        weights = {"fp8_weights": "fp8", "lowbit_weights": fmt}
        label = f"dual_fp8_{fmt}"
    else:
        raise ValueError("worker policy must be single, threshold, phase or expert_split")
    layers = config.get("layers")
    if not isinstance(layers, list) or not layers:
        raise ValueError("worker config must enumerate every actual MoE layer")
    records, seen = [], set()
    required = {"layer", "module", "worker_activation_basis_id", *weights}
    for layer in layers:
        if not isinstance(layer, dict) or set(layer) != required:
            raise ValueError(f"worker layer requires exactly {sorted(required)}")
        index = layer["layer"]
        if type(index) is not int or index < 0 or index in seen:
            raise ValueError("worker layer indices must be distinct nonnegative integers")
        seen.add(index)
        if layer["module"] != f"model.layers.{index}.mlp.experts":
            raise ValueError("unsupported worker module path")
        basis = layer["worker_activation_basis_id"]
        if not isinstance(basis, str) or not re.fullmatch(r"rmsnorm_input_basis_v1:sha256:[0-9a-f]{64}", basis):
            raise ValueError("worker layer requires its actual expected canonical RMSNorm basis ID")
        for key, artifact_format in weights.items():
            if not isinstance(layer[key], str) or not layer[key]:
                raise ValueError("sidecar path must be a nonempty string")
            artifact = (path.parent / layer[key]).resolve()
            if not artifact.is_file():
                raise FileNotFoundError(artifact)
            layer[key] = str(artifact)
            records.append({"layer": index, "role": key, "format": artifact_format,
                            "path": str(artifact), "bytes": artifact.stat().st_size,
                            "sha256": sha256(artifact) if hash_weights else None,
                            "worker_activation_basis_id": basis})
    output = Path(output).resolve()
    # Every run gets a new report, and relative sidecars retain source-JSON meaning.
    config["report_path"] = str(output / "worker_installation.json")
    return {"worker_class": WORKER_CLASS, "policy": policy, "format": label,
            "worker_model_kwargs": {"worker_cls": WORKER_CLASS, "enforce_eager": True,
                                    "compilation_config": {"mode": 0, "cudagraph_mode": "NONE"}},
            "source_config": str(path), "source_config_sha256": sha256(path),
            "effective_config_path": str(output / "worker_config.json"),
            "effective_config": config, "artifacts": records,
            "sidecar_hashes_verified": hash_weights,
            "basis_validation": "pending actual loaded model and independently checked sidecar norm tensors",
            "precision_scope": "expert sidecars only; router, attention and other nonexpert parameters follow the host checkpoint",
            "installation_verified": False}


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
    worker = worker_plan(args.worker_config, args.output, hash_weights=args.run) if args.worker_config else None
    fmt = args.format or (worker["format"] if worker else manifest["format"])
    if worker:
        if fmt != worker["format"]:
            raise ValueError(f"--format must be {worker['format']} for this worker config; dual is not ordinary FP8")
    elif fmt != manifest["format"] and not (manifest["format"] == "bf16" and fmt == "fp8"):
        raise ValueError("format must match checkpoint; only BF16 -> runtime FP8 is an allowed override")
    defaults = read_json(REGISTRY)["quality_defaults"]
    root = verify_checkpoint(manifest, hash_weights=args.run)
    quantization = "fp8" if fmt == "fp8" and not worker else manifest["runtime_quantization"]
    model_args = {"pretrained": str(root), "tokenizer": str(root),
                  "dtype": "bfloat16", "tensor_parallel_size": 1,
                  "gpu_memory_utilization": args.gpu_memory_utilization,
                  "max_model_len": args.max_model_len, "kv_cache_dtype": "auto",
                  "trust_remote_code": False, "seed": args.seed}
    if quantization:
        model_args["quantization"] = quantization
    if worker:
        model_args.update(worker["worker_model_kwargs"])
    if args.enforce_eager or worker:
        model_args["enforce_eager"] = True
        model_args["compilation_config"] = {"mode": 0, "cudagraph_mode": "NONE"}
    tasks = args.tasks.split(",") if args.tasks else defaults["tasks"]
    mc_only = bool(tasks) and set(tasks) <= MC_TASKS
    evaluated_identity = {"host_checkpoint_fingerprint": manifest["checkpoint_fingerprint"],
                          "host_runtime_quantization": quantization, "format": fmt}
    if worker:
        evaluated_identity["worker_policy"] = {key: value for key, value in worker["effective_config"].items()
                                               if key not in ("layers", "report_path")}
        evaluated_identity["sidecars"] = [{key: value for key, value in record.items() if key != "path"}
                                          for record in worker["artifacts"]]
    evaluated_fingerprint = (hashlib.sha256(json.dumps(evaluated_identity, sort_keys=True,
                              separators=(",", ":")).encode()).hexdigest() if args.run else None)
    return {"schema_version": 2, "status": "planned", "model_key": manifest["model_key"],
            "format": fmt, "checkpoint_fingerprint": manifest["checkpoint_fingerprint"],
            "host_checkpoint_format": manifest["format"], "worker": worker,
            "evaluated_model_identity": evaluated_identity,
            "evaluated_model_fingerprint": evaluated_fingerprint,
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
            "scope": ("multiple-choice prompt-loglikelihood quality (prefill); not mixed-decode generation quality"
                      if mc_only else "task-specific quality; inspect actual task output types before interpreting generation"),
            "evaluation_contract": {
                "default_five_tasks_unchanged": tasks == defaults["tasks"],
                "full_task_population_requested": args.limit is None,
                "known_multiple_choice_only": mc_only,
                "scoring_path": "lm-eval vLLM prompt_logprobs=1, max_tokens=1; extra sampled token is not the scored continuation" if mc_only else "task-dependent",
                "mixed_decode_quality_established": False,
                "phase_policy_limit": "default MC scoring exercises prompt/prefill choices, not a multi-step decode policy",
                "length_limit": "installed lm-eval truncates context+continuation to max_model_len-1; this is not the serving no-truncation protocol"},
            "comparison_note": "Use identical tasks, prompt/template, fewshot, KV dtype and seeds; sidecar arms must share the same nonexpert host checkpoint. Never substitute a different base release."}


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
    parser.add_argument("--format", choices=("bf16", "fp8", "awq", "nvfp4", "dual_fp8_awq", "dual_fp8_nvfp4"))
    parser.add_argument("--worker-config", help="Local DualFormatWorker JSON; infer format, force eager and retain sidecar identities")
    parser.add_argument("--lock-status", help="RUNNING run_locked status.json; required for --worker-config --run")
    parser.add_argument("--output", required=True)
    parser.add_argument("--tasks", help="comma-separated lm-eval task names")
    parser.add_argument("--num-fewshot", type=int, default=defaults["num_fewshot"])
    parser.add_argument("--seed", type=int, default=defaults["seed"])
    parser.add_argument("--batch-size", default=defaults["batch_size"])
    parser.add_argument("--max-model-len", type=int, default=defaults["max_model_len"])
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--bootstrap-iters", type=int, default=1000)
    parser.add_argument("--apply-chat-template", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true",
                        help="Disable compilation and CUDA graphs for bounded quality checks; recorded in model_args")
    parser.add_argument("--limit", type=float, help="smoke-only per-task count or fraction")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if not 0 < args.gpu_memory_utilization < 1:
        parser.error("--gpu-memory-utilization must be between zero and one")
    if args.run and args.worker_config:
        if not args.lock_status:
            parser.error("--worker-config --run requires --lock-status and a real run_locked ancestor")
        from bench_serving import verify_lock
        verify_lock(args.lock_status)
    plan = build_plan(args)
    if not args.run:
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        return
    # Weights are local and already validated. Prevent an unexpected alternate
    # model download while leaving dataset access available to the harness.
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    if plan["worker"]:
        worker = plan["worker"]
        write_json(worker["effective_config_path"], worker["effective_config"])
        worker["effective_config_sha256"] = sha256(worker["effective_config_path"])
        os.environ["C_FORMAT_DUAL_CONFIG"] = worker["effective_config_path"]
        code_path = str(Path(__file__).resolve().parent)
        os.environ["PYTHONPATH"] = os.pathsep.join(filter(None, (code_path, os.environ.get("PYTHONPATH"))))
        worker["environment"] = {"C_FORMAT_DUAL_CONFIG": os.environ["C_FORMAT_DUAL_CONFIG"],
                                 "PYTHONPATH_code_prefix": code_path}
        worker["implementation_sha256"] = {name: sha256(Path(code_path) / name) for name in
            ("dual_format_worker.py", "dual_format_moe.py", "format_backends.py", "export_bf16_fp8_artifact.py")}
    plan["versions"] = {}
    for package in ("lm_eval", "vllm", "torch", "transformers", "compressed-tensors"):
        try:
            plan["versions"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            plan["versions"][package] = None
    plan.update(status="running", start_unix_s=time.time())
    write_json(output / "run.json", plan)
    try:
        import lm_eval
        results = lm_eval.simple_evaluate(**plan["call"])
        if results is None:
            raise RuntimeError("lm-eval returned no results")
        (output / "results.json").write_text(json.dumps(results, indent=2,
                           ensure_ascii=False, default=json_default, allow_nan=False) + "\n")
        plan["actual_task_output_types"] = {name: config.get("output_type")
            for name, config in results.get("configs", {}).items() if isinstance(config, dict)}
        plan["actual_task_sample_counts"] = results.get("n-samples", {})
        if plan["worker"]:
            report = read_json(plan["worker"]["effective_config"]["report_path"])
            if report.get("status") != "installed_before_kv_profile" or report.get("config_sha256") != plan["worker"]["effective_config_sha256"]:
                raise RuntimeError("worker installation evidence is missing, failed or from a different config")
            plan["worker"]["installation_verified"] = True
            plan["worker"]["basis_validation"] = "passed real worker installation checks; see installation_report"
        plan.update(status="complete", end_unix_s=time.time())
    except BaseException as error:
        plan.update(status="failed", end_unix_s=time.time(), error=f"{type(error).__name__}: {error}")
        raise
    finally:
        if plan["worker"]:
            report_path = Path(plan["worker"]["effective_config"]["report_path"])
            if report_path.is_file():
                plan["worker"]["installation_report"] = str(report_path)
                plan["worker"]["installation_report_sha256"] = sha256(report_path)
            else:
                plan["worker"]["installation_report_missing"] = True
        write_json(output / "run.json", plan)
    print(json.dumps({"status": plan["status"], "output": str(output.resolve()),
                      "format": plan["format"],
                      "checkpoint_fingerprint": plan["checkpoint_fingerprint"],
                      "evaluated_model_fingerprint": plan["evaluated_model_fingerprint"],
                      "smoke_only": plan["smoke_only"]}, indent=2))


if __name__ == "__main__":
    main()
