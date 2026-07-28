#!/usr/bin/env python3
"""Run the single-GPU MoE profiler on frozen contiguous natural-text spans."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import profile_multi_moe_inference as base


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
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--route-census-steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    for field in ("prompt_len", "decode_steps", "warmup_decode_steps", "repeats", "route_census_steps"):
        if getattr(args, field) <= 0:
            parser.error(f"--{field.replace('_', '-')} must be positive")
    return args


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ContiguousNaturalInputBuilder:
    def __init__(self, tokenizer: Any, prompt_file: Path) -> None:
        text = prompt_file.read_text(encoding="utf-8")
        token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if not token_ids:
            raise RuntimeError("natural prompt file tokenized to zero tokens")
        self.token_ids = token_ids

    def __call__(
        self,
        _tokenizer: Any,
        batch_size: int,
        prompt_len: int,
        seed: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if len(self.token_ids) < prompt_len:
            raise RuntimeError(
                f"natural corpus has {len(self.token_ids)} tokens, fewer than prompt_len={prompt_len}"
            )
        rng = random.Random(seed)
        max_start = len(self.token_ids) - prompt_len
        rows = []
        for _ in range(batch_size):
            start = rng.randint(0, max_start) if max_start else 0
            rows.append(self.token_ids[start : start + prompt_len])
        input_ids = torch.tensor(rows, dtype=torch.long, device=device)
        return input_ids, torch.ones_like(input_ids)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU timing is not accepted by this protocol")
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
    tokenizer.padding_side = "left"
    natural_builder = ContiguousNaturalInputBuilder(tokenizer, prompt_file)

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

    original_builder = base.build_input_ids
    base.build_input_ids = natural_builder
    timing_rows: list[dict[str, Any]] = []
    layer_rows: list[dict[str, Any]] = []
    route_rows: list[dict[str, Any]] = []
    try:
        with torch.inference_mode():
            for batch_size in args.batch_sizes:
                base.run_sequence(
                    model,
                    tokenizer,
                    arm="warmup",
                    batch_size=batch_size,
                    prompt_len=args.prompt_len,
                    decode_steps=args.warmup_decode_steps,
                    repeat=-1,
                    seed=args.seed + batch_size * 10_000,
                    timer=None,
                    record=False,
                )
                for repeat in range(args.repeats):
                    trial_seed = args.seed + batch_size * 10_000 + repeat * 101
                    arm_order = ["unprofiled", "profiled"] if repeat % 2 == 0 else ["profiled", "unprofiled"]
                    for arm in arm_order:
                        timer = base.LayerTimer(blocks) if arm == "profiled" else None
                        current_timing, current_layers = base.run_sequence(
                            model,
                            tokenizer,
                            arm=arm,
                            batch_size=batch_size,
                            prompt_len=args.prompt_len,
                            decode_steps=args.decode_steps,
                            repeat=repeat,
                            seed=trial_seed,
                            timer=timer,
                            record=True,
                        )
                        if timer is not None:
                            timer.close()
                        timing_rows.extend(current_timing)
                        layer_rows.extend(current_layers)
                route_rows.extend(
                    base.run_route_census(
                        model,
                        tokenizer,
                        blocks,
                        batch_size=batch_size,
                        prompt_len=args.prompt_len,
                        decode_steps=min(args.route_census_steps, args.decode_steps),
                        seed=args.seed + batch_size * 10_000,
                        top_k=top_k,
                    )
                )
    finally:
        base.build_input_ids = original_builder

    base.write_csv(output_dir / "timings_raw.csv", timing_rows)
    base.write_csv(output_dir / "moe_layers_raw.csv", layer_rows)
    base.write_csv(output_dir / "route_census_untimed.csv", route_rows)
    source_path = Path(__file__).resolve()
    base_path = Path(base.__file__).resolve()
    manifest = {
        "status": "SINGLE_GPU_NATURAL_TEXT_CHARACTERIZATION_ONLY_NOT_RECEIVER_CONGESTION",
        "evidence_boundary": (
            "Real CUDA timing of frozen contiguous natural-text spans and local MoE blocks on one RTX 5090. "
            "No continuous arrivals, EP ranks, NCCL, receiver queue, or return all-to-all traffic."
        ),
        "args": vars(args),
        "workload": {
            "kind": "contiguous_natural_text_spans",
            "prompt_file": str(prompt_file),
            "prompt_file_sha256": file_sha256(prompt_file),
            "prompt_source_url": args.prompt_source_url,
            "corpus_tokens": len(natural_builder.token_ids),
            "span_sampling": "deterministic uniform start offset without token reordering",
        },
        "model": {
            "name": args.model,
            "revision_requested": args.revision,
            "config_commit_hash": getattr(model.config, "_commit_hash", None),
            "architectures": getattr(model.config, "architectures", None),
            "num_hidden_layers": getattr(model.config, "num_hidden_layers", None),
            "num_experts": getattr(model.config, "num_experts", None),
            "num_experts_per_tok": top_k,
            "moe_block_count": len(blocks),
            "moe_block_names": [name for name, _ in blocks],
            "load_seconds": load_seconds,
        },
        "environment": {
            "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "transformers": base.package_version("transformers"),
            "huggingface_hub": base.package_version("huggingface_hub"),
            "nvidia_smi": base.nvidia_smi(),
            "cuda_device": torch.cuda.get_device_name(0),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        },
        "source": {
            "path": str(source_path),
            "sha256": base.sha256_file(source_path),
            "base_profiler_path": str(base_path),
            "base_profiler_sha256": base.sha256_file(base_path),
        },
        "row_counts": {
            "timings_raw": len(timing_rows),
            "moe_layers_raw": len(layer_rows),
            "route_census_untimed": len(route_rows),
        },
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
