"""Dispatch gate-check: does tail-safety transfer from combine to dispatch?

The critical experiment for the round-trip direction. Combine error is scaled
by gate (small for tail → safe). Dispatch error goes through expert MLP
(nonlinear → possibly dangerous). Tests whether rank8 dispatch INT4 is also
safe, or only combine-side tail-INT4 works.

Two parts:
  1. Dispatch-only rank sweep (combine BF16): uniform/rank1/rank8 × FP8/INT8/INT4
  2. Round-trip matrix: dispatch_fp8 + combine variations
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

from capture_moe import patch_mixtral_moe
from modeling import load_model, load_tokenizer
from prompts import get_prompts


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="allenai/OLMoE-1B-7B-0924")
    p.add_argument("--num-samples", type=int, default=32)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--dataset", default="wikitext2")
    p.add_argument("--num-receiver-groups", type=int, default=4)
    p.add_argument("--output-dir", default="outputs/main_experiments/olmoe_dispatch_gatecheck")
    return p.parse_args()


def tokenized_inputs(tokenizer, texts, seq_len):
    for text in texts:
        yield tokenizer(text, return_tensors="pt", truncation=True, max_length=seq_len)


def compute_ppl(logits, input_ids):
    if logits.shape[1] < 2:
        return float("nan")
    sl = logits[:, :-1, :].contiguous().float()
    sy = input_ids[:, 1:].contiguous()
    return float(torch.exp(F.cross_entropy(sl.view(-1, sl.size(-1)), sy.view(-1))).item())


def compute_kl(full_logits, approx_logits):
    full = full_logits[:, :-1, :].contiguous().float()
    approx = approx_logits[:, :-1, :].contiguous().float()
    p = F.softmax(full, dim=-1)
    log_q = F.log_softmax(approx, dim=-1)
    return float(F.kl_div(log_q, p, reduction="batchmean").item())


def run_strategy(model, tokenizer, name, texts, seq_len, baseline_logits,
                 combine_policy="full", dispatch_policy=None, num_groups=4):
    recorder = patch_mixtral_moe(
        model, combine_policy, num_receiver_groups=num_groups,
        receiver_mapping="contiguous", dispatch_policy_name=dispatch_policy,
    )
    ppls, kls = [], []
    for idx, inputs in enumerate(tokenized_inputs(tokenizer, texts, seq_len)):
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits.detach().cpu()
        ppls.append(compute_ppl(logits, inputs["input_ids"]))
        kls.append(compute_kl(baseline_logits[idx], logits))
    return {"strategy": name, "mean_ppl": sum(ppls)/len(ppls),
            "mean_kl": sum(kls)/len(kls), "combine": combine_policy, "dispatch": dispatch_policy or "none"}


def main():
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    texts = get_prompts(args.dataset, args.num_samples)
    tokenizer = load_tokenizer(args.model)
    model, load_s = load_model(args.model, dtype_name=args.dtype)
    print(f"model loaded in {load_s:.1f}s", flush=True)

    # baseline
    patch_mixtral_moe(model, "full", num_receiver_groups=args.num_receiver_groups)
    baseline_logits, ppls_full = [], []
    for inputs in tokenized_inputs(tokenizer, texts, args.seq_len):
        with torch.no_grad():
            outputs = model(**inputs)
        baseline_logits.append(outputs.logits.detach().cpu())
        ppls_full.append(compute_ppl(baseline_logits[-1], inputs["input_ids"]))
    full_ppl = sum(ppls_full) / len(ppls_full)
    rows = [{"strategy": "full", "mean_ppl": full_ppl, "mean_kl": 0.0, "combine": "full", "dispatch": "none"}]
    print(f"  full: PPL={full_ppl:.4f}", flush=True)

    # 1. Dispatch-only rank sweep (combine = full/BF16)
    print("\n=== Dispatch-only sweep (combine BF16) ===", flush=True)
    dispatch_strategies = [
        ("dispatch_uniform_fp8", "full", "dispatch_uniform_fp8"),
        ("dispatch_uniform_int8", "full", "dispatch_uniform_int8"),
        ("dispatch_uniform_int4", "full", "dispatch_uniform_int4"),
        ("dispatch_rank8_int4", "full", "dispatch_rank8_int4"),
        ("dispatch_rank1_int4", "full", "dispatch_rank1_int4"),
    ]
    for name, comb, disp in dispatch_strategies:
        row = run_strategy(model, tokenizer, name, texts, args.seq_len, baseline_logits,
                           combine_policy=comb, dispatch_policy=disp, num_groups=args.num_receiver_groups)
        row["ppl_delta"] = row["mean_ppl"] - full_ppl
        rows.append(row)
        print(f"  {name:30s}  PPL={row['mean_ppl']:.4f}  KL={row['mean_kl']:.4f}  dPPL={row['ppl_delta']:+.4f}", flush=True)

    # 2. Round-trip matrix
    print("\n=== Round-trip matrix ===", flush=True)
    roundtrip_strategies = [
        ("rt_dispatch_fp8_combine_fp8", "uniform_fp8", "dispatch_uniform_fp8"),
        ("rt_dispatch_fp8_combine_tailint4", "fp8top4_rest_int4", "dispatch_uniform_fp8"),
        ("rt_dispatch_bf16_combine_tailint4", "fp8top4_rest_int4", None),
    ]
    for name, comb, disp in roundtrip_strategies:
        row = run_strategy(model, tokenizer, name, texts, args.seq_len, baseline_logits,
                           combine_policy=comb, dispatch_policy=disp, num_groups=args.num_receiver_groups)
        row["ppl_delta"] = row["mean_ppl"] - full_ppl
        rows.append(row)
        print(f"  {name:35s}  PPL={row['mean_ppl']:.4f}  KL={row['mean_kl']:.4f}  dPPL={row['ppl_delta']:+.4f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out / "dispatch_gatecheck_results.csv", index=False)
    print(f"\nsaved to {out}/dispatch_gatecheck_results.csv", flush=True)


if __name__ == "__main__":
    main()
