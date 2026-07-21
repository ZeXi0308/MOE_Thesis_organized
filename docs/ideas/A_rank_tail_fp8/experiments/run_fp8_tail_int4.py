"""FP8-foundation + tail-rank INT4 upgrade sweep.

Starts from uniform FP8 (50% saving, KL=0.293) and progressively upgrades tail
ranks from FP8 -> INT4, testing how far rank-awareness can push saving beyond
the 50% FP8 ceiling before cascading kicks in.

This is the natural "precision-tier-aware" strategy: uniform FP8 as the safe
foundation, tail ranks upgraded to INT4 because their contribution share is
small (rank8 = 4.91%). The question: is there a non-dominated point vs
uniform_fp8 at >50% saving?
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


# top-8: rank1..rank8. Progressively upgrade tail ranks FP8 -> INT4.
# "fp8_r{N}int4" = ranks 1..(8-N) at FP8, ranks (8-N+1)..8 at INT4.
TAIL_INT4_STRATEGIES = {
    "uniform_fp8":         ["fp8","fp8","fp8","fp8","fp8","fp8","fp8","fp8"],   # 50.0%
    "fp8_r8int4":          ["fp8","fp8","fp8","fp8","fp8","fp8","fp8","int4"],  # 53.125%
    "fp8_r78int4":         ["fp8","fp8","fp8","fp8","fp8","fp8","int4","int4"], # 56.25%
    "fp8_r678int4":        ["fp8","fp8","fp8","fp8","fp8","int4","int4","int4"],# 59.375%
    "fp8_r5678int4":       ["fp8","fp8","fp8","fp8","int4","int4","int4","int4"],# 62.5%
    "fp8_r345678int4":     ["fp8","fp8","int4","int4","int4","int4","int4","int4"],# 68.75%
    "uniform_int4":        ["int4","int4","int4","int4","int4","int4","int4","int4"], # 75.0%
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
    p.add_argument("--output-dir", default="outputs/main_experiments/olmoe_fp8_tail_int4")
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
        num_receiver_groups=num_groups, receiver_mapping=receiver_mapping, lut=lut,
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
    return {
        "strategy": name, "samples": len(texts), "seq_len": seq_len, "top_k": top_k,
        "byte_saving": recorder.total_byte_saving(),
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

    print("running full baseline...", flush=True)
    patch_mixtral_moe(model, "full", num_receiver_groups=num_groups, receiver_mapping=args.receiver_mapping)
    baseline_logits, ppls_full = [], []
    for inputs in tokenized_inputs(tokenizer, texts, args.seq_len):
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits.detach().cpu()
        baseline_logits.append(logits)
        ppls_full.append(compute_ppl(logits, inputs["input_ids"]))
    full_ppl = sum(ppls_full) / len(ppls_full)
    rows = [{"strategy": "full", "samples": len(texts), "seq_len": args.seq_len, "top_k": top_k,
             "byte_saving": 0.0, "mean_ppl": full_ppl, "mean_kl_vs_full": 0.0, "local_relative_mse": 0.0}]
    print(rows[0], flush=True)
    pd.DataFrame(rows).to_csv(out / "fp8_tail_int4_results.partial.csv", index=False)

    for name, precisions in TAIL_INT4_STRATEGIES.items():
        row = run_lut_strategy(model, tokenizer, name, precisions, texts, args.seq_len,
                               baseline_logits, num_layers, num_groups, top_k, args.receiver_mapping)
        rows.append(row)
        print(row, flush=True)
        pd.DataFrame(rows).to_csv(out / "fp8_tail_int4_results.partial.csv", index=False)

    df = pd.DataFrame(rows)
    full_ppl = float(df[df["strategy"] == "full"]["mean_ppl"].iloc[0])
    df["ppl_delta"] = df["mean_ppl"] - full_ppl
    df.to_csv(out / "fp8_tail_int4_results.csv", index=False)
    print(f"\nsaved to {out}/fp8_tail_int4_results.csv", flush=True)


if __name__ == "__main__":
    main()
