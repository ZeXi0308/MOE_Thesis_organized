"""Generalized FP8+tail-INT4 sweep + rank control for any top-k MoE model.

Builds strategies dynamically from the model's top_k, so it works on OLMoE
top-8, LLM-jp top-16, Mixtral top-2, etc. Runs:
  1. uniform_fp8 (50% baseline)
  2. tail-INT4 upgrade sweep (progressively upgrade tail ranks FP8->INT4)
  3. rank control at the ~62.5% saving point (tail vs head vs odd)

Usage:
  python run_fp8_tail_int4_general.py --model llm-jp/optimal-sparsity-math-d512-E32-k16-920M-A520M \
      --num-samples 64 --seq-len 128 --dtype bfloat16 --dataset wikitext2
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


def build_strategies(top_k: int) -> dict[str, list[str]]:
    """Build FP8+tail-INT4 sweep + rank control for a given top_k."""
    strat: dict[str, list[str]] = {}
    # uniform_fp8 baseline
    strat["uniform_fp8"] = ["fp8"] * top_k

    # tail-INT4 upgrade sweep: upgrade n_tail ranks to INT4.
    # Pick a few informative points: ~6.25%, ~25%, ~50%, ~62.5%, ~75% of ranks.
    n_tail_options = sorted(set([
        1, max(1, top_k // 8), max(1, top_k // 4), top_k // 2,
        max(1, 3 * top_k // 4), top_k - 1, top_k,
    ]))
    for n in n_tail_options:
        if n <= 0 or n > top_k:
            continue
        name = f"fp8_tail{n}int4"
        strat[name] = ["fp8"] * (top_k - n) + ["int4"] * n

    # uniform_int4 reference
    strat["uniform_int4"] = ["int4"] * top_k
    return strat


def build_rank_control(top_k: int) -> dict[str, list[str]]:
    """Build rank-selection control at ~62.5% saving (half FP8 + half INT4)."""
    half = top_k // 2
    if half == 0:
        half = 1
    return {
        f"fp8_tail{half}int4_rankaware": ["fp8"] * (top_k - half) + ["int4"] * half,
        f"fp8_head{half}int4_antirank":  ["int4"] * half + ["fp8"] * (top_k - half),
        "fp8_oddint4_norank":            ["int4" if i % 2 == 0 else "fp8" for i in range(top_k)],
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="llm-jp/optimal-sparsity-math-d512-E32-k16-920M-A520M")
    p.add_argument("--num-samples", type=int, default=64)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--dataset", default="wikitext2")
    p.add_argument("--num-receiver-groups", type=int, default=4)
    p.add_argument("--receiver-mapping", default="contiguous")
    p.add_argument("--output-dir", default=None)
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
    return {(l, g, r): precisions[r - 1]
            for l in range(num_layers) for g in range(num_groups) for r in range(1, top_k + 1)}


def run_lut(model, tokenizer, name, precisions, texts, seq_len, baseline_logits,
            num_layers, num_groups, top_k, receiver_mapping):
    lut = build_lut(precisions, num_layers, num_groups, top_k)
    recorder = patch_mixtral_moe(model, "lut", num_receiver_groups=num_groups,
                                 receiver_mapping=receiver_mapping, lut=lut)
    ppls, kls = [], []
    for idx, inputs in enumerate(tokenized_inputs(tokenizer, texts, seq_len)):
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits.detach().cpu()
        ppls.append(compute_ppl(logits, inputs["input_ids"]))
        kls.append(compute_kl(baseline_logits[idx], logits))
    err = recorder.error_rows()
    rel_mse = sum(r["sq_error"] for r in err) / max(sum(r["sq_full"] for r in err), 1e-12)
    return {"strategy": name, "samples": len(texts), "seq_len": seq_len, "top_k": top_k,
            "byte_saving": recorder.total_byte_saving(),
            "mean_ppl": sum(ppls) / max(len(ppls), 1),
            "mean_kl_vs_full": sum(kls) / len(kls), "local_relative_mse": rel_mse}


def main() -> None:
    args = parse_args()
    model_slug = args.model.replace("/", "--")
    out = Path(args.output_dir) if args.output_dir else Path(f"outputs/main_experiments/{model_slug}_fp8_tail_int4")
    out.mkdir(parents=True, exist_ok=True)
    texts = get_prompts(args.dataset, args.num_samples)
    tokenizer = load_tokenizer(args.model)
    model, load_seconds = load_model(args.model, dtype_name=args.dtype)
    print(f"model loaded in {load_seconds:.1f}s", flush=True)

    num_layers = len(model.model.layers)
    top_k = int(getattr(model.config, "num_experts_per_tok", 8))
    num_groups = args.num_receiver_groups
    print(f"top_k={top_k}, num_layers={num_layers}", flush=True)

    # baseline
    print("running full baseline...", flush=True)
    patch_mixtral_moe(model, "full", num_receiver_groups=num_groups, receiver_mapping=args.receiver_mapping)
    baseline_logits, ppls_full = [], []
    for inputs in tokenized_inputs(tokenizer, texts, args.seq_len):
        with torch.no_grad():
            outputs = model(**inputs)
        baseline_logits.append(outputs.logits.detach().cpu())
        ppls_full.append(compute_ppl(baseline_logits[-1], inputs["input_ids"]))
    full_ppl = sum(ppls_full) / len(ppls_full)
    rows = [{"strategy": "full", "samples": len(texts), "seq_len": args.seq_len, "top_k": top_k,
             "byte_saving": 0.0, "mean_ppl": full_ppl, "mean_kl_vs_full": 0.0, "local_relative_mse": 0.0}]
    print(rows[0], flush=True)
    pd.DataFrame(rows).to_csv(out / "results.partial.csv", index=False)

    # sweep + control
    all_strats = {**build_strategies(top_k), **build_rank_control(top_k)}
    for name, precisions in all_strats.items():
        row = run_lut(model, tokenizer, name, precisions, texts, args.seq_len,
                      baseline_logits, num_layers, num_groups, top_k, args.receiver_mapping)
        rows.append(row)
        print(row, flush=True)
        pd.DataFrame(rows).to_csv(out / "results.partial.csv", index=False)

    df = pd.DataFrame(rows)
    fp = float(df[df["strategy"] == "full"]["mean_ppl"].iloc[0])
    df["ppl_delta"] = df["mean_ppl"] - fp
    df.to_csv(out / "results.csv", index=False)
    print(f"\nsaved to {out}/results.csv", flush=True)


if __name__ == "__main__":
    main()
