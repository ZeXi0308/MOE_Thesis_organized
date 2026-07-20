"""Layer sensitivity experiment (Phase 6 / 实验 C).

Applies a given approximation strategy to ONE MoE layer at a time and measures
end-to-end KL / PPL / local MSE against the full-precision baseline.

Usage example (OLMoE top-8, rank8_int4):

    python run_layer_sensitivity.py \
        --model allenai/OLMoE-1B-7B-0924 \
        --strategy rank8_int4 \
        --num-samples 32 --seq-len 128 \
        --dtype bfloat16 --dataset wikitext2 \
        --num-receiver-groups 4
"""
from __future__ import annotations

import argparse

import pandas as pd
import torch
import torch.nn.functional as F

from capture_moe import patch_mixtral_moe
from modeling import DEFAULT_MODEL, load_model, load_tokenizer
from paths import resolve_output_dir
from prompts import get_prompts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--strategy", default="rankk_int4")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16", "auto"])
    parser.add_argument("--dataset", default="wikitext2", choices=["builtin", "wikitext2"])
    parser.add_argument("--num-receiver-groups", type=int, default=4)
    parser.add_argument("--receiver-mapping", default="contiguous", choices=["contiguous", "mod"])
    return parser.parse_args()


def tokenized_inputs(tokenizer, texts: list[str], seq_len: int):
    for text in texts:
        yield tokenizer(text, return_tensors="pt", truncation=True, max_length=seq_len)


def compute_ppl(logits: torch.Tensor, input_ids: torch.Tensor) -> float:
    if logits.shape[1] < 2:
        return float("nan")
    shift_logits = logits[:, :-1, :].contiguous().float()
    shift_labels = input_ids[:, 1:].contiguous()
    loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    return float(torch.exp(loss).item())


def compute_kl(full_logits: torch.Tensor, approx_logits: torch.Tensor) -> float:
    full = full_logits[:, :-1, :].contiguous().float()
    approx = approx_logits[:, :-1, :].contiguous().float()
    p = F.softmax(full, dim=-1)
    log_q = F.log_softmax(approx, dim=-1)
    return float(F.kl_div(log_q, p, reduction="batchmean").item())


def main() -> None:
    args = parse_args()
    out = resolve_output_dir(args.model, args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    texts = get_prompts(args.dataset, args.num_samples)
    tokenizer = load_tokenizer(args.model)
    model, load_seconds = load_model(args.model, dtype_name=args.dtype)
    print(f"model loaded in {load_seconds:.1f}s", flush=True)

    num_layers = len(model.model.layers)
    print(f"total layers: {num_layers}", flush=True)

    # ---- baseline (full) logits ----
    print("running full baseline...", flush=True)
    baseline_logits: list[torch.Tensor] = []
    baseline_ppls: list[float] = []
    for inputs in tokenized_inputs(tokenizer, texts, args.seq_len):
        patch_mixtral_moe(
            model, "full",
            num_receiver_groups=args.num_receiver_groups,
            receiver_mapping=args.receiver_mapping,
        )
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits.detach().cpu()
        baseline_logits.append(logits)
        baseline_ppls.append(compute_ppl(logits, inputs["input_ids"]))

    full_ppl = sum(baseline_ppls) / len(baseline_ppls)
    print(f"baseline PPL: {full_ppl:.4f}", flush=True)

    # ---- per-layer sweep ----
    rows: list[dict] = []
    for target_layer in range(num_layers):
        print(f"layer {target_layer}/{num_layers - 1}...", end=" ", flush=True)
        recorder = patch_mixtral_moe(
            model, args.strategy,
            num_receiver_groups=args.num_receiver_groups,
            receiver_mapping=args.receiver_mapping,
            target_layer=target_layer,
        )

        kls: list[float] = []
        ppls: list[float] = []
        for idx, inputs in enumerate(tokenized_inputs(tokenizer, texts, args.seq_len)):
            with torch.no_grad():
                outputs = model(**inputs)
            logits = outputs.logits.detach().cpu()
            kls.append(compute_kl(baseline_logits[idx], logits))
            ppls.append(compute_ppl(logits, inputs["input_ids"]))

        error_rows = recorder.error_rows()
        rel_mse = 0.0
        for r in error_rows:
            if r["layer"] == target_layer:
                rel_mse = r["relative_mse"]
                break

        mean_kl = sum(kls) / len(kls)
        mean_ppl = sum(ppls) / len(ppls)
        row = {
            "target_layer": target_layer,
            "mean_kl": mean_kl,
            "mean_ppl": mean_ppl,
            "ppl_delta": mean_ppl - full_ppl,
            "local_relative_mse": rel_mse,
        }
        rows.append(row)
        print(f"KL={mean_kl:.6f}  PPL_delta={mean_ppl - full_ppl:.6f}  MSE={rel_mse:.8f}", flush=True)

        pd.DataFrame(rows).to_csv(out / "layer_sensitivity.partial.csv", index=False)

    # ---- final output ----
    df = pd.DataFrame(rows)
    df.to_csv(out / "layer_sensitivity.csv", index=False)

    max_kl = float(df["mean_kl"].max())
    min_kl = float(df["mean_kl"].min())
    kl_ratio = max_kl / max(min_kl, 1e-12)

    max_mse = float(df["local_relative_mse"].max())
    min_mse = float(df["local_relative_mse"].min())
    mse_ratio = max_mse / max(min_mse, 1e-12)

    if kl_ratio >= 3.0:
        verdict = (
            f"**Layer sensitivity is significant** (KL ratio {kl_ratio:.2f}x >= 3x).\n\n"
            f"The `layer` dimension should stay in the LUT key: "
            f"`Rank-LUT[layer, receiver_group, rank] -> precision`."
        )
    elif kl_ratio >= 1.5:
        verdict = (
            f"**Layer sensitivity is moderate** (KL ratio {kl_ratio:.2f}x).\n\n"
            f"Consider keeping `layer` in the LUT, but the benefit may be marginal."
        )
    else:
        verdict = (
            f"**Layer sensitivity is weak** (KL ratio {kl_ratio:.2f}x < 1.5x).\n\n"
            f"The LUT can degenerate to `(receiver_group, rank) -> precision` without much loss."
        )

    top5 = df.nlargest(5, "mean_kl")
    bottom5 = df.nsmallest(5, "mean_kl")

    report = f"""# Layer Sensitivity Report

model: `{args.model}`
strategy: `{args.strategy}`
samples: `{args.num_samples}`
seq_len: `{args.seq_len}`
dtype: `{args.dtype}`
dataset: `{args.dataset}`

## Summary

| metric | value |
|---|---|
| num layers | {num_layers} |
| max KL | {max_kl:.6f} |
| min KL | {min_kl:.6f} |
| KL ratio (max/min) | {kl_ratio:.2f}x |
| max local MSE | {max_mse:.8f} |
| min local MSE | {min_mse:.8f} |
| MSE ratio (max/min) | {mse_ratio:.2f}x |

## Verdict

{verdict}

## Most Sensitive Layers (top 5)

| layer | KL | PPL delta | local MSE |
|---|---|---|---|
"""
    for _, r in top5.iterrows():
        report += f"| {int(r['target_layer'])} | {r['mean_kl']:.6f} | {r['ppl_delta']:.6f} | {r['local_relative_mse']:.8f} |\n"

    report += """
## Least Sensitive Layers (bottom 5)

| layer | KL | PPL delta | local MSE |
|---|---|---|---|
"""
    for _, r in bottom5.iterrows():
        report += f"| {int(r['target_layer'])} | {r['mean_kl']:.6f} | {r['ppl_delta']:.6f} | {r['local_relative_mse']:.8f} |\n"

    report += """
## Full Table

| layer | KL | PPL delta | local MSE |
|---|---|---|---|
"""
    for _, r in df.iterrows():
        report += f"| {int(r['target_layer'])} | {r['mean_kl']:.6f} | {r['ppl_delta']:.6f} | {r['local_relative_mse']:.8f} |\n"

    (out / "layer_sensitivity_report.md").write_text(report, encoding="utf-8")
    print(f"\nresults saved to {out}/", flush=True)
    print(f"KL ratio (max/min): {kl_ratio:.2f}x", flush=True)


if __name__ == "__main__":
    main()
