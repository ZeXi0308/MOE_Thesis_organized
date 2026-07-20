"""Direction 3 (error-feedback / residual compensation for combine quantization).

Idea: the currently-supported tail-rank policies (`fp8topN_rest_{int4,mxfp4,nvfp4}`)
quantize each token's tail-rank expert-output vector independently -- the rounding
error on token t is thrown away and has no effect on token t+1. This script tests
whether carrying the per-token quantization residual forward, CAUSALLY, along the
token sequence (same idea as error-feedback / residual accumulation in compressed
SGD, but applied to the combine-side vector stream instead of gradients) reduces
the corpus PPL / token-KL degradation at the SAME byte budget.

This is a purely single-node, single-forward-pass simulation: for a given layer,
tail rank slot `r`, and token order t=0..T-1 in a document, we maintain one
residual vector (hidden_dim,) and do:

    combined_t = raw_output_t + residual
    quantized_t = apply_precision(combined_t, prec)   # int4 / mxfp4 / nvfp4
    residual <- combined_t - quantized_t               # leftover rounding error
    output_t = quantized_t

Residual resets to zero at the start of each document (no cross-document state).
This requires ZERO extra bytes on the wire (residual is purely local sender-side
state); it only changes what value gets quantized before departure. If it works,
it strengthens the already-surviving "tail-INT4 is useful" claim with a free
accuracy improvement, without touching the receiver/aggregation side at all.

Boundary: this is still fake-quant PPL/KL evaluation, not a communication or
kernel benchmark. Error feedback here is applied entirely on the CPU/GPU compute
side before the (simulated) wire transfer; in a real system this would need to
sit in the sender-side pack path, adding O(hidden_dim) state and compute per
tail-rank slot per layer -- cheap, but a real implementation detail not modeled
here (no compute latency accounted for).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from types import MethodType

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from fake_quant import apply_precision
from metrics import MetricAccumulator
from modeling import load_model, load_tokenizer
from prompts import get_prompts


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="allenai/OLMoE-1B-7B-0924")
    p.add_argument("--dataset", default="wikitext2")
    p.add_argument("--dataset-split", default="validation")
    p.add_argument("--test-samples", type=int, default=32)
    p.add_argument("--test-offset", type=int, default=128)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--tail-precisions", default="int4,mxfp4,nvfp4")
    p.add_argument("--bootstrap", type=int, default=500)
    p.add_argument("--offline", action="store_true")
    p.add_argument(
        "--output-dir",
        default="experiments/idea_a_mac/outputs/paper_validation/error_feedback_tail",
    )
    return p.parse_args()


def dataframe_to_markdown(df: pd.DataFrame, columns: list[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for _, row in df[columns].iterrows():
        values = []
        for column in columns:
            value = row[column]
            values.append(f"{value:.6f}" if isinstance(value, (float, np.floating)) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _apply_tail_with_error_feedback(vec: torch.Tensor, prec: str) -> torch.Tensor:
    """vec: [T, H] tail-rank outputs for one document, one layer, in token order.
    Returns quantized [T, H] with causal residual carried forward, reset per call
    (i.e. per document x layer x rank slot)."""
    T, H = vec.shape
    out = torch.empty_like(vec)
    residual = torch.zeros(H, dtype=vec.dtype, device=vec.device)
    for t in range(T):
        combined = (vec[t] + residual).unsqueeze(0)
        quantized = apply_precision(combined, prec).squeeze(0)
        residual = (combined.squeeze(0) - quantized).detach()
        out[t] = quantized
    return out


def _patched_olmoe_forward_tail_variant(self, hidden_states: torch.Tensor):
    """Reimplementation of the OLMoE sparse-MoE forward that supports FOUR tail
    modes for the last `n_tail` rank slots (head ranks always get plain FP8):
      - "plain": apply_precision independently per token (existing behaviour)
      - "ef": causal error-feedback quantization across the token sequence
    Head ranks (rank 1..top_k-n_tail) always get plain FP8 in both modes, matching
    the existing `fp8topN_rest_{prec}` policy this is meant to be compared against.
    """
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    hidden_states = hidden_states.view(-1, hidden_dim)

    router_logits = self.gate(hidden_states)
    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
    routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
    if self.norm_topk_prob:
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
    routing_weights = routing_weights.to(hidden_states.dtype)

    total_tokens = batch_size * sequence_length
    raw_outputs = torch.zeros(
        (total_tokens, self.top_k, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device,
    )
    expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)
    expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
    for expert_idx_tensor in expert_hit:
        expert_idx = int(expert_idx_tensor.item())
        expert_layer = self.experts[expert_idx]
        idx, top_x = torch.where(expert_mask[expert_idx])
        current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
        raw_outputs[top_x, idx, :] = expert_layer(current_state)

    n_tail = self._ef_n_tail
    prec = self._ef_tail_precision
    mode = self._ef_mode  # "plain" or "ef"
    out = apply_precision(raw_outputs, "fp8")  # FP8 baseline for all ranks
    for rank_idx in range(self.top_k):
        if rank_idx < self.top_k - n_tail:
            continue  # head ranks stay FP8
        if mode == "plain":
            out[:, rank_idx, :] = apply_precision(raw_outputs[:, rank_idx, :], prec)
        elif mode == "ef":
            out[:, rank_idx, :] = _apply_tail_with_error_feedback(raw_outputs[:, rank_idx, :], prec)
        else:
            raise ValueError(f"unknown mode: {mode}")

    approx_final = (out * routing_weights[:, :, None]).sum(dim=1)
    return approx_final.reshape(batch_size, sequence_length, hidden_dim), router_logits


def _patched_olmoe_forward_full(self, hidden_states: torch.Tensor):
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    hidden_states = hidden_states.view(-1, hidden_dim)
    router_logits = self.gate(hidden_states)
    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
    routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
    if self.norm_topk_prob:
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
    routing_weights = routing_weights.to(hidden_states.dtype)
    total_tokens = batch_size * sequence_length
    raw_outputs = torch.zeros(
        (total_tokens, self.top_k, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device,
    )
    expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)
    expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
    for expert_idx_tensor in expert_hit:
        expert_idx = int(expert_idx_tensor.item())
        expert_layer = self.experts[expert_idx]
        idx, top_x = torch.where(expert_mask[expert_idx])
        current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
        raw_outputs[top_x, idx, :] = expert_layer(current_state)
    full_final = (raw_outputs * routing_weights[:, :, None]).sum(dim=1)
    return full_final.reshape(batch_size, sequence_length, hidden_dim), router_logits


def patch_model(model, mode: str | None, tail_precision: str | None, n_tail: int):
    """mode is None -> full precision baseline. Otherwise "plain" or "ef"."""
    for layer in model.model.layers:
        moe = layer.mlp
        if mode is None:
            moe.forward = MethodType(_patched_olmoe_forward_full, moe)
        else:
            moe._ef_n_tail = n_tail
            moe._ef_tail_precision = tail_precision
            moe._ef_mode = mode
            moe.forward = MethodType(_patched_olmoe_forward_tail_variant, moe)


def run_eval(model, tokenizer, texts, seq_len, mode, tail_precision, n_tail, baseline_logits=None):
    patch_model(model, mode, tail_precision, n_tail)
    metrics = MetricAccumulator()
    logits_out = []
    for local_idx, text in enumerate(texts):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=seq_len)
        with torch.no_grad():
            logits = model(**inputs).logits.detach().cpu()
        metrics.add(
            local_idx, logits, inputs["input_ids"],
            baseline_logits=baseline_logits[local_idx] if baseline_logits is not None else None,
            attention_mask=inputs.get("attention_mask"),
        )
        logits_out.append(logits)
    return metrics, logits_out


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    test_texts = get_prompts(
        args.dataset, args.test_samples, offset=args.test_offset, split=args.dataset_split,
    )
    tokenizer = load_tokenizer(args.model, local_files_only=args.offline)
    model, load_seconds = load_model(args.model, dtype_name=args.dtype, local_files_only=args.offline)
    top_k = int(getattr(model.config, "num_experts_per_tok", 8))
    n_tail = max(1, top_k // 2)
    print(f"model loaded in {load_seconds:.1f}s; top_k={top_k}, n_tail={n_tail}", flush=True)

    print("running full baseline...", flush=True)
    full_metrics, baseline_logits = run_eval(model, tokenizer, test_texts, args.seq_len, None, None, n_tail)

    tail_precisions = [v.strip() for v in args.tail_precisions.split(",") if v.strip()]
    summary_rows = []
    full_summary = full_metrics.bootstrap_summary(args.bootstrap)
    full_summary.update({"strategy": "full", "tail_precision": "-", "mode": "-", "ppl_delta_vs_full": 0.0})
    summary_rows.append(full_summary)

    for prec in tail_precisions:
        for mode, label in (("plain", "plain"), ("ef", "error_feedback")):
            print(f"running tail={prec} mode={label}...", flush=True)
            metrics, _ = run_eval(
                model, tokenizer, test_texts, args.seq_len, mode, prec, n_tail,
                baseline_logits=baseline_logits,
            )
            row = metrics.bootstrap_summary(args.bootstrap)
            row.update({
                "strategy": f"fp8top{top_k - n_tail}_rest_{prec}_{label}",
                "tail_precision": prec,
                "mode": label,
                "ppl_delta_vs_full": metrics.corpus_ppl - full_metrics.corpus_ppl,
            })
            summary_rows.append(row)
            pd.DataFrame(summary_rows).to_csv(out / "error_feedback.partial.csv", index=False)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out / "error_feedback_results.csv", index=False)

    # paired comparison: for each precision, is "ef" significantly better than "plain"?
    comparisons = []
    for prec in tail_precisions:
        plain_row = summary_df[(summary_df["tail_precision"] == prec) & (summary_df["mode"] == "plain")]
        ef_row = summary_df[(summary_df["tail_precision"] == prec) & (summary_df["mode"] == "error_feedback")]
        if plain_row.empty or ef_row.empty:
            continue
        plain_kl = float(plain_row["mean_token_kl"].iloc[0])
        ef_kl = float(ef_row["mean_token_kl"].iloc[0])
        comparisons.append({
            "tail_precision": prec,
            "plain_kl": plain_kl,
            "ef_kl": ef_kl,
            "kl_relative_change": (ef_kl - plain_kl) / max(plain_kl, 1e-12),
            "ef_better": ef_kl < plain_kl,
        })
    comparisons_df = pd.DataFrame(comparisons)
    comparisons_df.to_csv(out / "error_feedback_comparison.csv", index=False)

    columns = ["strategy", "tail_precision", "mode", "corpus_ppl", "ppl_delta_vs_full",
               "mean_token_kl", "mean_token_kl_ci_low", "mean_token_kl_ci_high"]
    table = dataframe_to_markdown(summary_df, columns)
    comp_table = dataframe_to_markdown(comparisons_df, list(comparisons_df.columns)) if not comparisons_df.empty else "(no data)"

    report = f"""# Error-Feedback (Residual Compensation) for Tail-Rank Combine Quantization

## 目的

检验方向 3：把 tail-rank INT4/MXFP4/NVFP4 量化的舍入残差，沿 token 序列因果地
向前传递（类似压缩 SGD 里的 error feedback，搬到 combine 输出向量流），能否在
**同样的字节预算**下降低 held-out PPL/KL 退化。这是一个纯 sender 侧本地状态，
不产生任何额外通信字节，只改变"量化前送进量化器的是什么值"。

## 方法

- baseline: `full`（无量化）
- 每种 tail precision（{', '.join(tail_precisions)}）分别跑两种模式：
  - `plain`：逐 token 独立量化（现有 `fp8topN_rest_{{prec}}` 策略的行为）
  - `error_feedback`：残差在同一文档内，沿 token 顺序、按 (layer, tail-rank-slot)
    因果累积并补偿到下一个 token 的量化输入上；每篇文档开头残差重置为 0。
- head ranks（前 `{top_k - n_tail}` 名）两种模式下都保持 FP8，不受影响。
- 测试集：`{args.dataset}:{args.dataset_split}` offset=`{args.test_offset}`，n=`{args.test_samples}`，
  seq_len=`{args.seq_len}`（与其余实验一致，避免引入新的评测口径差异）。

## 结果

{table}

## Plain vs Error-Feedback 直接对比

{comp_table}

## 解读边界

- 这仍是 fake-quant PPL/KL 实验，不是通信或 kernel benchmark；error feedback 在
  真实系统里需要在 sender-side pack 路径里维护每个 (layer, tail-rank-slot) 的
  hidden_dim 残差状态，本实验未建模这部分状态维护和计算的延迟开销。
- 残差按"文档内 token 顺序"因果累积，只使用过去信息，不使用未来 token，理论上
  可以在真实 autoregressive 解码中原生实现（每步只需要维护和更新一个 hidden_dim
  向量），不需要额外通信。
- 若 `error_feedback` 在多个 precision 下都稳定降低 `mean_token_kl`（且降低幅度
  超出 bootstrap CI 范围），说明这是一个几乎零成本、可以直接叠加在现有
  tail-rank two-lane 方案上的质量改进，可作为论文里"tail-INT4 有用"这条仅存
  claim 的进一步加固点。
- 若两者差异很小或方向不稳定，说明 combine 输出向量之间（不同 token、同一
  rank slot）的相关性不足以支撑 error feedback 起作用，应如实报告为负结果。
"""
    (out / "error_feedback_report.md").write_text(report, encoding="utf-8")
    print(table, flush=True)
    print(comp_table, flush=True)
    print(f"saved to {out}", flush=True)


if __name__ == "__main__":
    main()
