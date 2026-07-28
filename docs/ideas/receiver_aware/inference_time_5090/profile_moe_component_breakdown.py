#!/usr/bin/env python3
"""Break down local Mixtral-style MoE-block CUDA time on one GPU.

This profiler temporarily replaces each discovered MoE block's forward method
with the same eager algorithm plus four coarse CUDA-event regions.  Coarse
regions avoid the prohibitive observer tax caused by a hook on every expert.
The expert-loop region still includes gather, expert compute, routing-weight
multiplication, and index_add; it is not pure GEMM time.  Nothing in this file
measures EP, NCCL, or receiver traffic.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import types
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

import profile_multi_moe_inference as base


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="llm-jp/optimal-sparsity-math-d512-E32-k16-920M-A520M")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--batch-sizes", type=base.parse_int_list, default=base.parse_int_list("1,4,8,16,32"))
    parser.add_argument("--prompt-len", type=int, default=128)
    parser.add_argument("--decode-steps", type=int, default=16)
    parser.add_argument("--warmup-decode-steps", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    for field in ("prompt_len", "decode_steps", "warmup_decode_steps", "repeats"):
        if getattr(args, field) <= 0:
            parser.error(f"--{field.replace('_', '-')} must be positive")
    return args


class CoarseMixtralTimer:
    """Patch Mixtral-style eager MoE forwards with four coarse timed regions."""

    def __init__(self, blocks: list[tuple[str, torch.nn.Module]]) -> None:
        self.context: base.Context | None = None
        self.pending: list[dict[str, Any]] = []
        self._original_forwards: list[tuple[torch.nn.Module, Callable[..., Any]]] = []
        for block_name, block in blocks:
            self._validate_block(block_name, block)
            original_forward = block.forward
            self._original_forwards.append((block, original_forward))
            block.forward = types.MethodType(self._make_forward(block_name), block)

    @staticmethod
    def _validate_block(block_name: str, block: torch.nn.Module) -> None:
        required = ("gate", "experts", "top_k", "num_experts")
        missing = [name for name in required if not hasattr(block, name)]
        if missing:
            raise RuntimeError(f"unsupported MoE block {block_name}; missing {missing}")
        if not isinstance(block.gate, torch.nn.Module):
            raise RuntimeError(f"unsupported MoE block {block_name}; gate is not a module")
        if not isinstance(block.experts, torch.nn.ModuleList):
            raise RuntimeError(f"unsupported MoE block {block_name}; experts are not a ModuleList")

    def _record_pair(
        self,
        *,
        block_name: str,
        component: str,
        start: torch.cuda.Event,
        end: torch.cuda.Event,
    ) -> None:
        if self.context is None:
            return
        self.pending.append(
            {
                "context": base.Context(**asdict(self.context)),
                "module_name": block_name,
                "component": component,
                "expert_index": -1,
                "start": start,
                "end": end,
            }
        )

    def _make_forward(self, block_name: str) -> Callable[..., Any]:
        timer = self

        def timed_forward(block: torch.nn.Module, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            total_start = torch.cuda.Event(enable_timing=True)
            total_end = torch.cuda.Event(enable_timing=True)
            gate_start = torch.cuda.Event(enable_timing=True)
            gate_end = torch.cuda.Event(enable_timing=True)
            routing_start = torch.cuda.Event(enable_timing=True)
            routing_end = torch.cuda.Event(enable_timing=True)
            loop_start = torch.cuda.Event(enable_timing=True)
            loop_end = torch.cuda.Event(enable_timing=True)

            total_start.record()
            batch_size, sequence_length, hidden_dim = hidden_states.shape
            if block.training and getattr(block, "jitter_noise", 0) > 0:
                jitter = float(block.jitter_noise)
                hidden_states *= torch.empty_like(hidden_states).uniform_(1.0 - jitter, 1.0 + jitter)
            hidden_states = hidden_states.view(-1, hidden_dim)

            gate_start.record()
            router_logits = block.gate(hidden_states)
            gate_end.record()

            routing_start.record()
            routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
            routing_weights, selected_experts = torch.topk(routing_weights, block.top_k, dim=-1)
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
            routing_weights = routing_weights.to(hidden_states.dtype)
            final_hidden_states = torch.zeros(
                (batch_size * sequence_length, hidden_dim),
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
            expert_mask = F.one_hot(selected_experts, num_classes=block.num_experts).permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
            routing_end.record()

            loop_start.record()
            for expert_idx in expert_hit:
                expert_layer = block.experts[expert_idx]
                idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))
                current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
                current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]
                final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
            loop_end.record()

            final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
            total_end.record()
            timer._record_pair(
                block_name=block_name,
                component="moe_total",
                start=total_start,
                end=total_end,
            )
            timer._record_pair(
                block_name=block_name,
                component="gate",
                start=gate_start,
                end=gate_end,
            )
            timer._record_pair(
                block_name=block_name,
                component="routing_setup",
                start=routing_start,
                end=routing_end,
            )
            timer._record_pair(
                block_name=block_name,
                component="expert_loop",
                start=loop_start,
                end=loop_end,
            )
            return final_hidden_states, router_logits

        return timed_forward

    def set_context(self, context: base.Context | None) -> None:
        self.context = context

    def drain(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for event in self.pending:
            row = asdict(event["context"])
            row.update(
                {
                    "module_name": event["module_name"],
                    "component": event["component"],
                    "expert_index": event["expert_index"],
                    "latency_ms": float(event["start"].elapsed_time(event["end"])),
                }
            )
            rows.append(row)
        self.pending.clear()
        return rows

    def close(self) -> None:
        for block, original_forward in self._original_forwards:
            block.forward = original_forward
        self._original_forwards.clear()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite evidence directory: {output_dir}")
    output_dir.mkdir(parents=True)

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

    timing_rows: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []
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
                arm_order = ["unprofiled", "breakdown"] if repeat % 2 == 0 else ["breakdown", "unprofiled"]
                for arm in arm_order:
                    timer = CoarseMixtralTimer(blocks) if arm == "breakdown" else None
                    current_timing, current_components = base.run_sequence(
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
                    component_rows.extend(current_components)

    base.write_csv(output_dir / "timings_raw.csv", timing_rows)
    base.write_csv(output_dir / "components_raw.csv", component_rows)
    source_path = Path(__file__).resolve()
    manifest = {
        "status": "SINGLE_GPU_LOCAL_MOE_COMPONENT_BREAKDOWN_ONLY",
        "evidence_boundary": (
            "Coarse CUDA-event regions inside eager local Mixtral-style MoE blocks on one GPU. "
            "The expert loop includes gather/weighting/index_add; there is no EP/NCCL/receiver traffic."
        ),
        "args": vars(args),
        "model": {
            "name": args.model,
            "config_commit_hash": getattr(model.config, "_commit_hash", None),
            "num_hidden_layers": getattr(model.config, "num_hidden_layers", None),
            "num_local_experts": getattr(model.config, "num_local_experts", None),
            "num_experts_per_tok": getattr(model.config, "num_experts_per_tok", None),
            "moe_block_count": len(blocks),
            "load_seconds": load_seconds,
        },
        "environment": {
            "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "python": sys.version,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "nvidia_smi": base.nvidia_smi(),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        },
        "row_counts": {"timings_raw": len(timing_rows), "components_raw": len(component_rows)},
        "source": {"path": str(source_path), "sha256": base.sha256_file(source_path)},
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
