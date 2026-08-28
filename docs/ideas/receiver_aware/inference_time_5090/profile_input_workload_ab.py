#!/usr/bin/env python3
"""Interleaved natural-text versus synthetic-token latency A/B on one GPU."""

from __future__ import annotations

import argparse
import json
import platform
import random
import sys
import time
from pathlib import Path
from typing import Any, Callable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import profile_multi_moe_inference as base
from profile_natural_moe_inference import ContiguousNaturalInputBuilder, file_sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="llm-jp/optimal-sparsity-math-d512-E32-k16-920M-A520M")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--prompt-file", required=True)
    parser.add_argument("--prompt-source-url", default=None)
    parser.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--batch-sizes", type=base.parse_int_list, default=base.parse_int_list("1,4,8"))
    parser.add_argument("--prompt-len", type=int, default=128)
    parser.add_argument("--decode-steps", type=int, default=16)
    parser.add_argument("--warmup-decode-steps", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--route-census-steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    for field in ("prompt_len", "decode_steps", "warmup_decode_steps", "repeats", "route_census_steps"):
        if getattr(args, field) <= 0:
            parser.error(f"--{field.replace('_', '-')} must be positive")
    return args


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing evidence directory: {output_dir}")
    output_dir.mkdir(parents=True)
    prompt_file = Path(args.prompt_file).resolve()
    if not prompt_file.is_file():
        raise FileNotFoundError(prompt_file)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=args.revision,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    natural_builder = ContiguousNaturalInputBuilder(tokenizer, prompt_file)
    synthetic_builder = base.build_input_ids
    builders: dict[str, Callable[..., tuple[torch.Tensor, torch.Tensor]]] = {
        "synthetic": synthetic_builder,
        "natural": natural_builder,
    }

    load_started = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.revision,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
        dtype=dtype,
        low_cpu_mem_usage=True,
    ).eval().to("cuda")
    torch.cuda.synchronize()
    load_seconds = time.time() - load_started
    blocks = base.discover_moe_blocks(model)
    if not blocks:
        raise RuntimeError("no MoE blocks discovered")
    top_k = int(getattr(model.config, "num_experts_per_tok", 0) or 0)
    if top_k <= 0:
        raise RuntimeError("model config does not expose a positive num_experts_per_tok")

    timing_rows: list[dict[str, Any]] = []
    route_rows: list[dict[str, Any]] = []
    try:
        with torch.inference_mode():
            for batch_size in args.batch_sizes:
                for workload, builder in builders.items():
                    base.build_input_ids = builder
                    base.run_sequence(
                        model,
                        tokenizer,
                        arm=f"warmup_{workload}",
                        batch_size=batch_size,
                        prompt_len=args.prompt_len,
                        decode_steps=args.warmup_decode_steps,
                        repeat=-1,
                        seed=args.seed + batch_size * 10_000,
                        timer=None,
                        record=False,
                    )
                for repeat in range(args.repeats):
                    workload_order = ["synthetic", "natural"] if repeat % 2 == 0 else ["natural", "synthetic"]
                    trial_seed = args.seed + batch_size * 10_000 + repeat * 101
                    for workload in workload_order:
                        base.build_input_ids = builders[workload]
                        current_timing, _ = base.run_sequence(
                            model,
                            tokenizer,
                            arm=workload,
                            batch_size=batch_size,
                            prompt_len=args.prompt_len,
                            decode_steps=args.decode_steps,
                            repeat=repeat,
                            seed=trial_seed,
                            timer=None,
                            record=True,
                        )
                        timing_rows.extend(current_timing)
                for workload, builder in builders.items():
                    base.build_input_ids = builder
                    current_routes = base.run_route_census(
                        model,
                        tokenizer,
                        blocks,
                        batch_size=batch_size,
                        prompt_len=args.prompt_len,
                        decode_steps=min(args.route_census_steps, args.decode_steps),
                        seed=args.seed + batch_size * 10_000,
                        top_k=top_k,
                    )
                    for row in current_routes:
                        row["workload"] = workload
                    route_rows.extend(current_routes)
    finally:
        base.build_input_ids = synthetic_builder

    base.write_csv(output_dir / "timings_raw.csv", timing_rows)
    base.write_csv(output_dir / "route_census_untimed.csv", route_rows)
    source_path = Path(__file__).resolve()
    base_path = Path(base.__file__).resolve()
    natural_path = Path(sys.modules[ContiguousNaturalInputBuilder.__module__].__file__).resolve()
    manifest = {
        "status": "SINGLE_GPU_INTERLEAVED_INPUT_AB_ONLY_NOT_RECEIVER_CONGESTION",
        "evidence_boundary": (
            "Interleaved AB/BA CUDA timing for frozen contiguous natural text and deterministic synthetic "
            "token sequences on one GPU. No continuous arrivals, EP, NCCL, or receiver traffic."
        ),
        "args": vars(args),
        "workload": {
            "natural_prompt_file": str(prompt_file),
            "natural_prompt_sha256": file_sha256(prompt_file),
            "natural_prompt_source_url": args.prompt_source_url,
            "natural_corpus_tokens": len(natural_builder.token_ids),
            "natural_sampling": "deterministic contiguous spans",
            "synthetic_sampling": "deterministic stride and wrap from frozen short corpus",
            "arm_order": "AB/BA alternating by repeat",
        },
        "model": {
            "name": args.model,
            "config_commit_hash": getattr(model.config, "_commit_hash", None),
            "num_hidden_layers": getattr(model.config, "num_hidden_layers", None),
            "num_experts_per_tok": top_k,
            "moe_block_count": len(blocks),
            "load_seconds": load_seconds,
        },
        "environment": {
            "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "nvidia_smi": base.nvidia_smi(),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        },
        "source": {
            "path": str(source_path),
            "sha256": base.sha256_file(source_path),
            "base_profiler_sha256": base.sha256_file(base_path),
            "natural_builder_sha256": base.sha256_file(natural_path),
        },
        "row_counts": {"timings_raw": len(timing_rows), "route_census_untimed": len(route_rows)},
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
