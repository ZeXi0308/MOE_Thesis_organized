#!/usr/bin/env python3
"""Profile full inference time and cumulative MoE-block time on one CUDA GPU.

This experiment is deliberately single-GPU.  It measures prefill latency,
per-step KV-cache decode latency, and the cumulative time spent inside every
discovered MoE block.  A separate, untimed pass records router-load imbalance.
It does *not* emulate EP, NCCL, receiver queues, or return all-to-all traffic.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


PROMPT_CORPUS = " ".join(
    [
        "Mixture of Experts models route every token to a small subset of experts.",
        "The dispatch collective sends token activations to the selected expert owners.",
        "The return collective sends expert outputs back to the token owner for combine.",
        "A useful systems experiment separates measured observations from hypotheses.",
        "Continuous decoding repeatedly executes every sparse feed forward layer.",
        "Tail latency must be measured with the complete inference path in the denominator.",
        "Router imbalance is a local workload signal and is not itself network congestion.",
        "Strong overlap and fusion can hide communication from the exposed critical path.",
    ]
)


@dataclass
class Context:
    arm: str
    phase: str
    batch_size: int
    prompt_len: int
    repeat: int
    seed: int
    decode_step: int


@dataclass
class PendingLayerEvent:
    context: Context
    module_name: str
    start: torch.cuda.Event
    end: torch.cuda.Event


def parse_int_list(value: str) -> list[int]:
    values = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("expected a comma-separated list of positive integers")
    if len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("batch sizes must be unique")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="allenai/OLMoE-1B-7B-0924")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--batch-sizes", type=parse_int_list, default=parse_int_list("1,4,8"))
    parser.add_argument("--prompt-len", type=int, default=128)
    parser.add_argument("--decode-steps", type=int, default=32)
    parser.add_argument("--warmup-decode-steps", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--route-census-steps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    for field in (
        "prompt_len",
        "decode_steps",
        "warmup_decode_steps",
        "repeats",
        "route_census_steps",
    ):
        if getattr(args, field) <= 0:
            parser.error(f"--{field.replace('_', '-')} must be positive")
    return args


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def nvidia_smi() -> str:
    command = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,driver_version,temperature.gpu,power.limit",
        "--format=csv,noheader",
    ]
    try:
        return subprocess.check_output(command, text=True, timeout=10).strip()
    except (OSError, subprocess.SubprocessError):
        return "UNAVAILABLE"


def is_moe_block(module: torch.nn.Module) -> bool:
    class_name = module.__class__.__name__.lower()
    named = "sparsemoe" in class_name or "moeblock" in class_name or "sparse_moe" in class_name
    structural = hasattr(module, "experts") and (
        hasattr(module, "gate") or hasattr(module, "router")
    )
    return bool(named or structural)


def discover_moe_blocks(model: torch.nn.Module) -> list[tuple[str, torch.nn.Module]]:
    blocks = [(name, module) for name, module in model.named_modules() if name and is_moe_block(module)]
    # A structural parent can contain another explicitly named MoE block.  Keep only
    # the deepest candidates so the same GPU work is not double counted.
    names = {name for name, _ in blocks}
    kept: list[tuple[str, torch.nn.Module]] = []
    for name, module in blocks:
        if any(other.startswith(name + ".") for other in names):
            continue
        kept.append((name, module))
    return kept


class LayerTimer:
    def __init__(self, blocks: list[tuple[str, torch.nn.Module]]) -> None:
        self.context: Context | None = None
        self.pending: list[PendingLayerEvent] = []
        self._starts: dict[int, list[torch.cuda.Event]] = {}
        self._names = {id(module): name for name, module in blocks}
        self._handles: list[Any] = []
        for _, module in blocks:
            self._handles.append(module.register_forward_pre_hook(self._pre_hook))
            self._handles.append(module.register_forward_hook(self._post_hook))

    def _pre_hook(self, module: torch.nn.Module, _inputs: tuple[Any, ...]) -> None:
        if self.context is None:
            return
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        self._starts.setdefault(id(module), []).append(event)

    def _post_hook(self, module: torch.nn.Module, _inputs: tuple[Any, ...], _output: Any) -> None:
        if self.context is None:
            return
        stack = self._starts.get(id(module), [])
        if not stack:
            raise RuntimeError(f"missing MoE start event for {self._names[id(module)]}")
        start = stack.pop()
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        self.pending.append(
            PendingLayerEvent(
                context=Context(**asdict(self.context)),
                module_name=self._names[id(module)],
                start=start,
                end=end,
            )
        )

    def set_context(self, context: Context | None) -> None:
        self.context = context

    def drain(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for event in self.pending:
            row = asdict(event.context)
            row.update(
                {
                    "module_name": event.module_name,
                    "latency_ms": float(event.start.elapsed_time(event.end)),
                }
            )
            rows.append(row)
        self.pending.clear()
        if any(self._starts.values()):
            raise RuntimeError("unclosed MoE timing events")
        return rows

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


class RouterCensus:
    def __init__(
        self,
        blocks: list[tuple[str, torch.nn.Module]],
        default_top_k: int,
    ) -> None:
        self.context: Context | None = None
        self.rows: list[dict[str, Any]] = []
        self._handles: list[Any] = []
        self._names: dict[int, str] = {}
        self.default_top_k = default_top_k
        for block_name, block in blocks:
            router = getattr(block, "gate", None) or getattr(block, "router", None)
            if not isinstance(router, torch.nn.Module):
                continue
            self._names[id(router)] = block_name
            self._handles.append(router.register_forward_hook(self._hook))

    def _hook(self, module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
        if self.context is None:
            return
        logits = output[0] if isinstance(output, tuple) else output
        if not isinstance(logits, torch.Tensor) or logits.ndim < 2:
            return
        flat = logits.detach().float().reshape(-1, logits.shape[-1])
        num_experts = int(flat.shape[-1])
        top_k = min(self.default_top_k, num_experts)
        selected = torch.topk(flat, k=top_k, dim=-1, sorted=False).indices.reshape(-1)
        counts = torch.bincount(selected, minlength=num_experts).float().cpu()
        mean = float(counts.mean().item())
        std = float(counts.std(unbiased=False).item())
        total = float(counts.sum().item())
        probabilities = counts / max(total, 1.0)
        positive = probabilities[probabilities > 0]
        entropy = float((-(positive * positive.log()).sum()).item())
        normalized_entropy = entropy / math.log(num_experts) if num_experts > 1 else 1.0
        row = asdict(self.context)
        row.update(
            {
                "module_name": self._names[id(module)],
                "tokens": int(flat.shape[0]),
                "top_k": top_k,
                "num_experts": num_experts,
                "max_expert_load": int(counts.max().item()),
                "mean_expert_load": mean,
                "max_to_mean": float(counts.max().item() / mean) if mean > 0 else float("nan"),
                "load_cv": std / mean if mean > 0 else float("nan"),
                "active_expert_fraction": float((counts > 0).float().mean().item()),
                "normalized_entropy": normalized_entropy,
            }
        )
        self.rows.append(row)

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


def build_input_ids(
    tokenizer: Any,
    batch_size: int,
    prompt_len: int,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    corpus_ids = tokenizer(PROMPT_CORPUS, add_special_tokens=False)["input_ids"]
    if not corpus_ids:
        raise RuntimeError("tokenizer returned an empty prompt corpus")
    rng = random.Random(seed)
    rows: list[list[int]] = []
    for batch_index in range(batch_size):
        offset = rng.randrange(len(corpus_ids))
        stride = 1 + ((batch_index * 2 + seed) % 7)
        row = [corpus_ids[(offset + stride * index) % len(corpus_ids)] for index in range(prompt_len)]
        rows.append(row)
    input_ids = torch.tensor(rows, dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    return input_ids, attention_mask


def timed_forward(
    model: torch.nn.Module,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    past_key_values: Any,
    context: Context,
    timer: LayerTimer | None,
) -> tuple[Any, float, list[dict[str, Any]]]:
    if timer is not None:
        timer.set_context(context)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        use_cache=True,
        return_dict=True,
    )
    end.record()
    if timer is not None:
        timer.set_context(None)
    torch.cuda.synchronize()
    layer_rows = timer.drain() if timer is not None else []
    return outputs, float(start.elapsed_time(end)), layer_rows


def run_sequence(
    model: torch.nn.Module,
    tokenizer: Any,
    *,
    arm: str,
    batch_size: int,
    prompt_len: int,
    decode_steps: int,
    repeat: int,
    seed: int,
    timer: LayerTimer | None,
    record: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    device = next(model.parameters()).device
    input_ids, attention_mask = build_input_ids(tokenizer, batch_size, prompt_len, seed, device)
    timing_rows: list[dict[str, Any]] = []
    layer_rows: list[dict[str, Any]] = []

    prefill_context = Context(arm, "prefill", batch_size, prompt_len, repeat, seed, -1)
    outputs, latency_ms, current_layers = timed_forward(
        model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        past_key_values=None,
        context=prefill_context,
        timer=timer,
    )
    if record:
        row = asdict(prefill_context)
        row.update(
            {
                "latency_ms": latency_ms,
                "tokens_processed": batch_size * prompt_len,
                "goodput_tokens_per_s": batch_size * prompt_len * 1000.0 / latency_ms,
            }
        )
        timing_rows.append(row)
        layer_rows.extend(current_layers)

    past_key_values = outputs.past_key_values
    next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    for decode_step in range(decode_steps):
        attention_mask = torch.cat(
            [attention_mask, torch.ones((batch_size, 1), dtype=attention_mask.dtype, device=device)],
            dim=1,
        )
        context = Context(arm, "decode", batch_size, prompt_len, repeat, seed, decode_step)
        outputs, latency_ms, current_layers = timed_forward(
            model,
            input_ids=next_token,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            context=context,
            timer=timer,
        )
        if record:
            row = asdict(context)
            row.update(
                {
                    "latency_ms": latency_ms,
                    "tokens_processed": batch_size,
                    "goodput_tokens_per_s": batch_size * 1000.0 / latency_ms,
                }
            )
            timing_rows.append(row)
            layer_rows.extend(current_layers)
        past_key_values = outputs.past_key_values
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    return timing_rows, layer_rows


def run_route_census(
    model: torch.nn.Module,
    tokenizer: Any,
    blocks: list[tuple[str, torch.nn.Module]],
    *,
    batch_size: int,
    prompt_len: int,
    decode_steps: int,
    seed: int,
    top_k: int,
) -> list[dict[str, Any]]:
    census = RouterCensus(blocks, default_top_k=top_k)
    device = next(model.parameters()).device
    input_ids, attention_mask = build_input_ids(tokenizer, batch_size, prompt_len, seed, device)
    census.context = Context("route_census_untimed", "prefill", batch_size, prompt_len, 0, seed, -1)
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
        return_dict=True,
    )
    past_key_values = outputs.past_key_values
    next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    for decode_step in range(decode_steps):
        attention_mask = torch.cat(
            [attention_mask, torch.ones((batch_size, 1), dtype=attention_mask.dtype, device=device)],
            dim=1,
        )
        census.context = Context(
            "route_census_untimed", "decode", batch_size, prompt_len, 0, seed, decode_step
        )
        outputs = model(
            input_ids=next_token,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )
        past_key_values = outputs.past_key_values
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    torch.cuda.synchronize()
    census.context = None
    census.close()
    return census.rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU timing is not accepted by this protocol")
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing evidence directory: {output_dir}")
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

    blocks = discover_moe_blocks(model)
    if not blocks:
        raise RuntimeError("no MoE blocks discovered; refusing to emit misleading timing evidence")
    top_k = int(getattr(model.config, "num_experts_per_tok", 0) or 0)
    if top_k <= 0:
        raise RuntimeError("model config does not expose a positive num_experts_per_tok")

    timing_rows: list[dict[str, Any]] = []
    layer_rows: list[dict[str, Any]] = []
    route_rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for batch_size in args.batch_sizes:
            # Untimed shape warmup.  It includes KV-cache decode and is never
            # included in raw or summary statistics.
            run_sequence(
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
                # Alternate arm order to reduce monotonic thermal/order bias.
                arm_order = ["unprofiled", "profiled"] if repeat % 2 == 0 else ["profiled", "unprofiled"]
                for arm in arm_order:
                    timer = LayerTimer(blocks) if arm == "profiled" else None
                    current_timing, current_layers = run_sequence(
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
                run_route_census(
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

    write_csv(output_dir / "timings_raw.csv", timing_rows)
    write_csv(output_dir / "moe_layers_raw.csv", layer_rows)
    write_csv(output_dir / "route_census_untimed.csv", route_rows)
    source_path = Path(__file__).resolve()
    manifest = {
        "status": "SINGLE_GPU_CHARACTERIZATION_ONLY_NOT_RECEIVER_CONGESTION",
        "evidence_boundary": (
            "Real CUDA timing of full prefill, KV-cache decode, and local MoE blocks on one RTX 5090. "
            "No EP ranks, NCCL, NVLink/RDMA return traffic, receiver queue, continuous arrivals, TPOT/P99 serving claim, or RankLane comparison."
        ),
        "args": vars(args),
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
            "transformers": package_version("transformers"),
            "huggingface_hub": package_version("huggingface_hub"),
            "nvidia_smi": nvidia_smi(),
            "cuda_device": torch.cuda.get_device_name(0),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        },
        "source": {"path": str(source_path), "sha256": sha256_file(source_path)},
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
