"""Rank-aware {BF16, FP8, INT4, drop} three-tier mixes for the >50% saving regime.

Tests whether rank-aware mixed precision can beat uniform_fp8 (50%, KL=0.293)
and uniform_int4 (75%, KL=27.15) in the 50-75% byte-saving window — the
reframed value proposition of Idea A after the FP8 baseline experiment.

Each strategy is a per-rank precision list (rank1 .. rank-k). A uniform LUT
(same assignment across all layers and receiver groups) is built and applied
via the existing patch_mixtral_moe(lut=...) path.
"""
from __future__ import annotations


# --- shared-lib bootstrap (auto) ---
import sys
from pathlib import Path as _Path

def _ensure_shared_on_path() -> None:
    here = _Path(__file__).resolve().parent
    for p in [here, *here.parents]:
        cand = p / "experiments" / "shared"
        if (cand / "capture_moe.py").exists():
            s = str(cand)
            if s not in sys.path:
                sys.path.insert(0, s)
            return
        if (p / "capture_moe.py").exists():
            s = str(p)
            if s not in sys.path:
                sys.path.insert(0, s)
            return

_ensure_shared_on_path()
del _ensure_shared_on_path, _Path
# --- end bootstrap ---

import argparse
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

from capture_moe import patch_mixtral_moe
from modeling import load_model, load_tokenizer
from prompts import get_prompts


# OLMoE top-8: per-rank precision list, rank1 (highest gate) .. rank8 (lowest).
# "x" marks the head/tail split. Byte budget per rank: bf16=2, fp8=1, int4=0.5, drop=0.
RANK_MIX_STRATEGIES = {
    # --- references (re-run in-sample for alignment) ---
    "uniform_fp8": ["fp8"] * 8,                       # 50.0%  (MegaScale baseline)
    "uniform_int4": ["int4"] * 8,                     # 75.0%
    # --- 50% saving: head-to-head with uniform_fp8 ---
    "mix_r12bf16_r34fp8_r58int4": ["bf16","bf16","fp8","fp8","int4","int4","int4","int4"],
    "mix_r1bf16_r26fp8_r78int4":  ["bf16","fp8","fp8","fp8","fp8","fp8","int4","int4"],
    "mix_r12bf16_r36fp8_r78drop": ["bf16","bf16","fp8","fp8","fp8","fp8","drop","drop"],
    # --- 56.25% saving ---
    "mix_r12bf16_r38int4":        ["bf16","bf16","int4","int4","int4","int4","int4","int4"],
    "mix_r1bf16_r24fp8_r58int4":  ["bf16","fp8","fp8","fp8","int4","int4","int4","int4"],
    # --- 65.625% saving ---
    "mix_r1bf16_r28int4":         ["bf16","int4","int4","int4","int4","int4","int4","int4"],
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="allenai/OLMoE-1B-7B-0924")
    p.add_argument("--num-samples", type=int, default=32)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--dataset", default="wikitext2")
    p.add_argument("--num-receiver-groups", type=int, default=4)
    p.add_argument("--receiver-mapping", default="contiguous")
    p.add_argument("--output-dir", default="outputs/main_experiments/olmoe_rank_mix")
    return p.parse_args()


def tokenized_inputs(tokenizer, texts, seq_len):
    for text in texts:
        yield tokenizer(text, return_tensors="pt", truncation=True, max_length=seq_len)


def compute_ppl(logits, input_ids):
    if logits.shape[1] < 2:
        return float("nan")
    sl = logits[:, :-1, :].contiguous().float()
    sy = input_ids[:, 1:].contiguous()
    loss = F.cross_entropy(sl.view(-1, sl.size(-1)), sy.view(-1))
    return float(torch.exp(loss).item())


def compute_kl(full_logits, approx_logits):
    full = full_logits[:, :-1, :].contiguous().float()
    approx = approx_logits[:, :-1, :].contiguous().float()
    p = F.softmax(full, dim=-1)
    log_q = F.log_softmax(approx, dim=-1)
    return float(F.kl_div(log_q, p, reduction="batchmean").item())


def build_lut(precisions, num_layers, num_groups, top_k):
    lut = {}
    for layer in range(num_layers):
        for group in range(num_groups):
            for rank in range(1, top_k + 1):
                lut[(layer, group, rank)] = precisions[rank - 1]
    return lut


def run_lut_strategy(model, tokenizer, name, precisions, texts, seq_len,
                     baseline_logits, num_layers, num_groups, top_k, receiver_mapping):
    lut = build_lut(precisions, num_layers, num_groups, top_k)
    recorder = patch_mixtral_moe(
        model, "lut",
        num_receiver_groups=num_groups,
        receiver_mapping=receiver_mapping,
        lut=lut,
    )
    ppls, kls = [], []
    for idx, inputs in enumerate(tokenized_inputs(tokenizer, texts, seq_len)):
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits.detach().cpu()
        ppls.append(compute_ppl(logits, inputs["input_ids"]))
        kls.append(compute_kl(baseline_logits[idx], logits))
    error_rows = recorder.error_rows()
    rel_mse = sum(r["sq_error"] for r in error_rows) / max(sum(r["sq_full"] for r in error_rows), 1e-12)
    byte_saving = recorder.total_byte_saving()
    return {
        "strategy": name,
        "samples": len(texts),
        "seq_len": seq_len,
        "top_k": top_k,
        "byte_saving": byte_saving,
        "mean_ppl": sum(ppls) / max(len(ppls), 1),
        "mean_kl_vs_full": sum(kls) / len(kls),
        "local_relative_mse": rel_mse,
    }


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    texts = get_prompts(args.dataset, args.num_samples)
    tokenizer = load_tokenizer(args.model)
    model, load_seconds = load_model(args.model, dtype_name=args.dtype)
    print(f"model loaded in {load_seconds:.1f}s", flush=True)

    num_layers = len(model.model.layers)
    top_k = int(getattr(model.config, "num_experts_per_tok", 8))
    num_groups = args.num_receiver_groups

    # 1. baseline full forward — cache logits
    print("running full baseline...", flush=True)
    patch_mixtral_moe(model, "full", num_receiver_groups=num_groups, receiver_mapping=args.receiver_mapping)
    baseline_logits = []
    ppls_full = []
    for idx, inputs in enumerate(tokenized_inputs(tokenizer, texts, args.seq_len)):
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits.detach().cpu()
        baseline_logits.append(logits)
        ppls_full.append(compute_ppl(logits, inputs["input_ids"]))
    full_ppl = sum(ppls_full) / len(ppls_full)
    rows = [{
        "strategy": "full", "samples": len(texts), "seq_len": args.seq_len, "top_k": top_k,
        "byte_saving": 0.0, "mean_ppl": full_ppl, "mean_kl_vs_full": 0.0, "local_relative_mse": 0.0,
    }]
    print(rows[0], flush=True)
    pd.DataFrame(rows).to_csv(out / "rank_mix_results.partial.csv", index=False)

    # 2. rank-mix + reference strategies
    for name, precisions in RANK_MIX_STRATEGIES.items():
        row = run_lut_strategy(
            model, tokenizer, name, precisions, texts, args.seq_len,
            baseline_logits, num_layers, num_groups, top_k, args.receiver_mapping,
        )
        rows.append(row)
        print(row, flush=True)
        pd.DataFrame(rows).to_csv(out / "rank_mix_results.partial.csv", index=False)

    df = pd.DataFrame(rows)
    full_ppl = float(df[df["strategy"] == "full"]["mean_ppl"].iloc[0])
    df["ppl_delta"] = df["mean_ppl"] - full_ppl
    df.to_csv(out / "rank_mix_results.csv", index=False)
    print(f"\nsaved to {out}/rank_mix_results.csv", flush=True)


if __name__ == "__main__":
    main()
