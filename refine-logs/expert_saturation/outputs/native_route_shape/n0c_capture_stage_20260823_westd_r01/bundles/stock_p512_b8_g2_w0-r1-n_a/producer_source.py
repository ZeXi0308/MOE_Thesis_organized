#!/usr/bin/env python3
"""Replay one frozen N0b prefix for the N0c capture/export source triage."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA_CONFIG = "n0c-capture-stage-arm-config-v1"
SCHEMA_RESULT = "n0c-capture-stage-arm-result-v1"
CLAIM_CEILING = "FRESH_PROCESS_ASSOCIATIONAL_CAPTURE_TRIAGE_ONLY"
MODEL = "allenai/OLMoE-1B-7B-0924"
REVISION = "6d84c48581ece794365f2b8e9cfb043c68ade9c5"
SEED = 20260823
OUTPUT_TOKENS = 16
WARMUP_SHAPES = ((128, 4), (128, 8), (128, 16), (512, 4), (512, 8), (512, 16))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_sha256(value: Any) -> str:
    blob = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def _write_json_once(path: Path, value: Any) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _gpu_processes() -> list[str]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        timeout=120,
        check=False,
    )
    if result.returncode:
        raise RuntimeError("cannot verify exclusive GPU ownership")
    return [line for line in result.stdout.splitlines() if line.strip()]


def _load_workload(path: Path) -> list[str]:
    payload = json.loads(path.read_text())
    texts = [str(row["prompt"]) for row in payload.get("requests", []) if row.get("prompt")]
    if len(texts) < 16:
        raise RuntimeError("frozen workload has fewer than 16 prompts")
    return texts


def _build_prompts(tokenizer: Any, texts: list[str], length: int, count: int) -> list[list[int]]:
    separator = tokenizer.encode("\n", add_special_tokens=False) or [1]
    prompts: list[list[int]] = []
    for sample in range(count):
        ids: list[int] = []
        cursor = sample
        while len(ids) < length:
            encoded = tokenizer.encode(texts[cursor % len(texts)], add_special_tokens=False)
            if encoded:
                ids.extend(encoded)
                ids.extend(separator)
            cursor += 1
        prompts.append(ids[:length])
    return prompts


def _sampling_params(SamplingParams: Any, prompt_length: int, max_tokens: int, export: bool) -> Any:
    kwargs: dict[str, Any] = {
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "min_tokens": max_tokens,
        "ignore_eos": True,
        "seed": SEED,
    }
    if export:
        kwargs["routed_experts_prompt_start"] = prompt_length - 1
    return SamplingParams(**kwargs)


def _load_prefix(spec: dict[str, Any], input_root: Path) -> list[tuple[dict[str, Any], list[list[int]]]]:
    rows = spec.get("prefix_records")
    if not isinstance(rows, list) or not rows:
        raise RuntimeError("target spec has no prefix records")
    if [row.get("execution_order") for row in rows] != list(range(len(rows))):
        raise RuntimeError("prefix execution order is not contiguous from zero")
    if spec.get("target_record") != rows[-1]:
        raise RuntimeError("target record must be the final prefix record")
    expected_plan = spec.get("prefix_plan_sha256")
    if expected_plan != _json_sha256(rows):
        raise RuntimeError("prefix plan SHA-256 mismatch")
    loaded: list[tuple[dict[str, Any], list[list[int]]]] = []
    for row in rows:
        path = input_root / str(row["input_artifact"])
        if not path.is_file() or _sha256(path) != row.get("input_artifact_sha256"):
            raise RuntimeError(f"frozen input mismatch: {path}")
        with np.load(path, allow_pickle=False) as archive:
            prompts = archive["prompt_token_ids"].astype(np.int64).tolist()
        if len(prompts) != int(row["batch_size"]):
            raise RuntimeError(f"batch width mismatch: {path}")
        if any(len(item) != int(row["prompt_length"]) for item in prompts):
            raise RuntimeError(f"prompt length mismatch: {path}")
        if _json_sha256(prompts) != row.get("prompt_token_ids_sha256"):
            raise RuntimeError(f"prompt-token SHA-256 mismatch: {path}")
        loaded.append((row, prompts))
    return loaded


def _verify_runtime_import_root(module_file: str, expected_runtime_root: Path) -> Path:
    expected = expected_runtime_root.resolve()
    imported_file = Path(module_file).resolve()
    actual = imported_file.parent.parent
    if actual != expected:
        raise RuntimeError(
            f"vLLM import escaped expected runtime root: {actual} != {expected}"
        )
    return actual


def _expected_runtime_variant(target_runtime: str, capture_mode: str) -> str:
    suffix = "-device" if capture_mode == "device" else ""
    return f"{target_runtime}{suffix}"


def _runtime_identity(expected_runtime_root: Path, logical_runtime_variant: str) -> dict[str, Any]:
    import torch
    import vllm

    module_file = getattr(vllm, "__file__", None)
    if not isinstance(module_file, str) or not module_file:
        raise RuntimeError("vLLM import has no concrete module file")
    import_root = _verify_runtime_import_root(module_file, expected_runtime_root)
    if vllm.__version__ != "0.26.0":
        raise RuntimeError(f"wrong vLLM version in {logical_runtime_variant}: {vllm.__version__}")
    package = Path(vllm.__file__).resolve().parent
    source_files = (
        "model_executor/layers/fused_moe/routed_experts_capturer.py",
        "v1/worker/gpu_model_runner.py",
    )
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "vllm": vllm.__version__,
        "vllm_module_file": str(Path(module_file).resolve()),
        "vllm_package": str(package),
        "vllm_source_root": str(import_root),
        "expected_runtime_root": str(expected_runtime_root.resolve()),
        "logical_runtime_variant": logical_runtime_variant,
        "runtime_import_root_verified": True,
        "source_sha256": {name: _sha256(package / name) for name in source_files},
        "gpu": subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            text=True,
            timeout=120,
        ).strip(),
        "vllm_batch_invariant": os.environ.get("VLLM_BATCH_INVARIANT", "0"),
        "n0c_device_capture_only": os.environ.get("N0C_DEVICE_CAPTURE_ONLY", "0"),
    }


def run(args: argparse.Namespace) -> None:
    if args.require_exclusive_gpu and _gpu_processes():
        raise RuntimeError("GPU is not isolated before engine initialization")
    if args.capture_mode == "device" and os.environ.get("N0C_DEVICE_CAPTURE_ONLY") != "1":
        raise RuntimeError("device arm requires N0C_DEVICE_CAPTURE_ONLY=1")
    if args.capture_mode != "device" and os.environ.get("N0C_DEVICE_CAPTURE_ONLY") == "1":
        raise RuntimeError("device-capture patch leaked into a non-device arm")
    expected_variant = _expected_runtime_variant(args.target_runtime, args.capture_mode)
    if args.logical_runtime_variant != expected_variant:
        raise RuntimeError(
            f"logical runtime variant mismatch: {args.logical_runtime_variant} != {expected_variant}"
        )
    runtime = _runtime_identity(
        Path(args.expected_runtime_root), args.logical_runtime_variant
    )
    from vllm import LLM, SamplingParams

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    source_copy = output / "producer_source.py"
    source_copy.write_bytes(Path(__file__).read_bytes())
    spec_path = Path(args.target_spec)
    input_root = Path(args.input_root)
    workload = Path(args.workload_manifest)
    spec = json.loads(spec_path.read_text())
    if spec.get("schema") != "n0c-capture-target-spec-v1":
        raise RuntimeError("unsupported target spec schema")
    if spec.get("target_id") != args.target_id or spec.get("target_runtime") != args.target_runtime:
        raise RuntimeError("CLI target identity disagrees with frozen spec")
    prefix = _load_prefix(spec, input_root)
    target = spec["target_record"]
    capture_enabled = args.capture_mode != "off"
    export_enabled = args.capture_mode == "full_export"
    llm = LLM(
        model=MODEL,
        revision=REVISION,
        tokenizer_revision=REVISION,
        dtype="bfloat16",
        seed=SEED,
        enforce_eager=True,
        enable_return_routed_experts=capture_enabled,
        disable_log_stats=False,
        enable_prefix_caching=False,
        max_model_len=1024,
        max_num_seqs=32,
        max_num_batched_tokens=8192,
        gpu_memory_utilization=0.80,
    )
    tokenizer = llm.get_tokenizer()
    texts = _load_workload(workload)
    for prompt_length, batch_size in WARMUP_SHAPES:
        prompts = _build_prompts(tokenizer, texts, prompt_length, batch_size)
        params = [
            _sampling_params(SamplingParams, prompt_length, 4, export_enabled)
            for _ in prompts
        ]
        llm.generate([{"prompt_token_ids": ids} for ids in prompts], params, use_tqdm=False)

    target_outputs = None
    target_wall_ms = None
    for row, prompts in prefix:
        params = [
            _sampling_params(SamplingParams, int(row["prompt_length"]), OUTPUT_TOKENS, export_enabled)
            for _ in prompts
        ]
        started = time.perf_counter()
        generated = llm.generate(
            [{"prompt_token_ids": ids} for ids in prompts], params, use_tqdm=False
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if int(row["execution_order"]) == int(target["execution_order"]):
            target_outputs = generated
            target_wall_ms = elapsed_ms
    if target_outputs is None or len(target_outputs) != int(target["batch_size"]):
        raise RuntimeError("target output was not produced with the frozen width")

    token_ids: list[list[int]] = []
    for item in target_outputs:
        if len(item.outputs) != 1:
            raise RuntimeError("target request has multiple completions")
        completion = item.outputs[0]
        tokens = list(completion.token_ids)
        if len(tokens) != OUTPUT_TOKENS or completion.finish_reason != "length":
            raise RuntimeError("target completion denominator drifted")
        token_ids.append(tokens)

    target_spec_sha = _sha256(spec_path)
    config = {
        "schema": SCHEMA_CONFIG,
        "target_id": args.target_id,
        "target_runtime": args.target_runtime,
        "round": args.round,
        "arm": args.arm,
        "capture_mode": args.capture_mode,
        "logical_runtime_variant": args.logical_runtime_variant,
        "runtime_import_root_verified": True,
        "runtime_patch_id": args.runtime_patch_id,
        "claim_ceiling": CLAIM_CEILING,
        "target_spec_sha256": target_spec_sha,
        "prefix_plan_sha256": spec["prefix_plan_sha256"],
        "runtime_package_manifest_sha256": args.runtime_package_manifest_sha256,
        "target_input_artifact_sha256": target["input_artifact_sha256"],
        "target_prompt_token_ids_sha256": target["prompt_token_ids_sha256"],
        "workload_manifest_sha256": _sha256(workload),
        "producer_source_sha256": _sha256(source_copy),
        "runtime_identity": runtime,
    }
    _write_json_once(output / "config.json", config)
    result: dict[str, Any] = {
        **{
            key: config[key]
            for key in (
                "target_id",
                "target_runtime",
                "round",
                "arm",
                "capture_mode",
                "logical_runtime_variant",
                "runtime_import_root_verified",
                "runtime_patch_id",
                "claim_ceiling",
                "target_spec_sha256",
                "prefix_plan_sha256",
                "runtime_package_manifest_sha256",
                "target_input_artifact_sha256",
                "target_prompt_token_ids_sha256",
            )
        },
        "schema": SCHEMA_RESULT,
        "status": "COMPLETE",
        "output_token_ids": token_ids,
        "warmup_count": len(WARMUP_SHAPES),
        "prefix_cells_executed": len(prefix),
        "target_wall_ms": target_wall_ms,
    }
    route_sha: str | None = None
    if export_enabled:
        routes = np.stack(
            [np.asarray(item.outputs[0].routed_experts, dtype=np.int64) for item in target_outputs]
        )
        expected_shape = (int(target["batch_size"]), OUTPUT_TOKENS, 16, 8)
        if routes.shape != expected_shape:
            raise RuntimeError(f"full route shape mismatch: {routes.shape} != {expected_shape}")
        route_path = output / "routes.npz"
        np.savez_compressed(route_path, routes=routes)
        route_sha = _sha256(route_path)
        prompt_length = int(target["prompt_length"])
        result.update(
            {
                "full_export_includes_prompt_tail": True,
                "route_mapping": [
                    {
                        "route_row": row,
                        "input_position": prompt_length - 1 + row,
                        "produces_output_token_index": row,
                    }
                    for row in range(OUTPUT_TOKENS)
                ],
                "route_artifact": "routes.npz",
                "route_artifact_sha256": route_sha,
                "route_shape": list(routes.shape),
            }
        )
    _write_json_once(output / "result.json", result)
    _write_json_once(
        output / "RUN_COMPLETE.json",
        {
            "status": "RUN_COMPLETE",
            "config_sha256": _sha256(output / "config.json"),
            "result_sha256": _sha256(output / "result.json"),
            "route_sha256": route_sha,
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target-spec", required=True)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--workload-manifest", required=True)
    parser.add_argument("--target-id", required=True)
    parser.add_argument("--target-runtime", choices=("stock", "valid-window"), required=True)
    parser.add_argument("--round", type=int, choices=range(4), required=True)
    parser.add_argument("--arm", choices=("n_a", "capture_only", "full_export", "n_b"), required=True)
    parser.add_argument("--capture-mode", choices=("off", "device", "full_export"), required=True)
    parser.add_argument("--runtime-patch-id", required=True)
    parser.add_argument("--runtime-package-manifest-sha256", required=True)
    parser.add_argument("--logical-runtime-variant", choices=("stock", "stock-device", "valid-window", "valid-window-device"), required=True)
    parser.add_argument("--expected-runtime-root", required=True)
    parser.add_argument("--require-exclusive-gpu", action=argparse.BooleanOptionalAction, default=True)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
