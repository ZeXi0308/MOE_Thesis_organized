"""Local inference-time timing for Idea A combine policies.

This script measures wall-clock model forward time for the fake-quant policies
used in the Mac experiments. It is useful as an implementation sanity check:
does a policy add obvious local overhead, and what are P50/P95/P99 prefill or
single-token forward times on this machine?

It is not a real multi-GPU serving benchmark. On this code path, FP8/INT4 are
fake quant-dequant operations inside one process, so the measured time includes
Python/PyTorch quantization overhead and does not include NCCL/DeepEP all-to-all.
For paper-grade TBT/P99, run the same policies in an expert-parallel serving
stack on the 8xA100 machine and measure dispatch/combine communication directly.
"""
from __future__ import annotations

import argparse
import csv
import os
import statistics
import time
from pathlib import Path

import torch

from capture_moe import patch_mixtral_moe
from modeling import load_model, load_tokenizer
from prompts import get_prompts


DEFAULT_STRATEGIES = [
    "full",
    "uniform_fp8",
    "fp8top7_rest_int4",
    "fp8top6_rest_int4",
    "fp8top4_rest_int4",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="allenai/OLMoE-1B-7B-0924")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps", "auto"])
    p.add_argument("--offline", action="store_true", help="Use local HuggingFace cache only.")
    p.add_argument("--dataset", default="builtin", choices=["builtin", "wikitext2"])
    p.add_argument("--num-prompts", type=int, default=4)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--decode-steps", type=int, default=16)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--num-receiver-groups", type=int, default=4)
    p.add_argument("--receiver-mapping", default="contiguous", choices=["contiguous", "mod"])
    p.add_argument("--strategies", nargs="+", default=DEFAULT_STRATEGIES)
    p.add_argument("--output-dir", default="outputs/thesis_evidence/09_fp8_baseline")
    return p.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    if name == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        raise RuntimeError("MPS requested but torch.backends.mps.is_available() is false")
    return torch.device(name)


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "n": len(values),
        "mean_ms": statistics.fmean(values) if values else float("nan"),
        "p50_ms": percentile(values, 0.50),
        "p95_ms": percentile(values, 0.95),
        "p99_ms": percentile(values, 0.99),
        "min_ms": min(values) if values else float("nan"),
        "max_ms": max(values) if values else float("nan"),
    }


def timed_call(fn, device: torch.device) -> float:
    sync(device)
    start = time.perf_counter()
    fn()
    sync(device)
    return (time.perf_counter() - start) * 1000.0


def build_inputs(tokenizer, texts: list[str], seq_len: int, device: torch.device) -> list[dict[str, torch.Tensor]]:
    inputs = []
    for text in texts:
        # Repeat short built-in prompts so every prefill run has comparable length.
        expanded = text
        while len(tokenizer(expanded, add_special_tokens=True)["input_ids"]) < seq_len:
            expanded = f"{expanded} {text}"
        item = tokenizer(expanded, return_tensors="pt", truncation=True, max_length=seq_len)
        inputs.append({k: v.to(device) for k, v in item.items()})
    return inputs


def one_token_inputs(prefill_inputs: list[dict[str, torch.Tensor]]) -> list[dict[str, torch.Tensor]]:
    out = []
    for item in prefill_inputs:
        token = item["input_ids"][:, -1:].contiguous()
        one = {"input_ids": token}
        if "attention_mask" in item:
            one["attention_mask"] = torch.ones_like(token)
        out.append(one)
    return out


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    print(f"loading model={args.model} dtype={args.dtype} device={device}", flush=True)
    tokenizer = load_tokenizer(args.model, local_files_only=args.offline)
    model, load_s = load_model(args.model, dtype_name=args.dtype, local_files_only=args.offline)
    model.to(device)
    model.eval()
    print(f"model loaded in {load_s:.1f}s", flush=True)

    texts = get_prompts(args.dataset, args.num_prompts)
    prefill_batches = build_inputs(tokenizer, texts, args.seq_len, device)
    decode_batches = one_token_inputs(prefill_batches)

    rows: list[dict[str, object]] = []

    for strategy in args.strategies:
        print(f"\n=== strategy: {strategy} ===", flush=True)
        recorder = patch_mixtral_moe(
            model,
            strategy,
            num_receiver_groups=args.num_receiver_groups,
            receiver_mapping=args.receiver_mapping,
        )

        with torch.inference_mode():
            for _ in range(args.warmup):
                for item in prefill_batches[:1]:
                    _ = model(**item)
                for item in decode_batches[:1]:
                    _ = model(**item)

            prefill_ms: list[float] = []
            for item in prefill_batches:
                prefill_ms.append(timed_call(lambda item=item: model(**item), device))

            decode_ms: list[float] = []
            for step in range(args.decode_steps):
                item = decode_batches[step % len(decode_batches)]
                decode_ms.append(timed_call(lambda item=item: model(**item), device))

        byte_saving = recorder.total_byte_saving()
        for phase, values in [("prefill", prefill_ms), ("single_token_forward", decode_ms)]:
            row = {
                "strategy": strategy,
                "phase": phase,
                "model": args.model,
                "dtype": args.dtype,
                "device": str(device),
                "seq_len": args.seq_len if phase == "prefill" else 1,
                "byte_saving_observed": byte_saving,
            }
            row.update(summarize(values))
            rows.append(row)
            print(
                f"{phase:22s} n={row['n']:3d} "
                f"p50={row['p50_ms']:.2f}ms p95={row['p95_ms']:.2f}ms "
                f"p99={row['p99_ms']:.2f}ms mean={row['mean_ms']:.2f}ms "
                f"byte_saving={byte_saving*100:.1f}%",
                flush=True,
            )

    output_csv = out_dir / "inference_time_local.csv"
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nsaved to {output_csv}", flush=True)
    print(
        "\nNOTE: this is local fake-quant forward timing, not multi-GPU serving TBT. "
        "Use it as a sanity check only.",
        flush=True,
    )


if __name__ == "__main__":
    main()
