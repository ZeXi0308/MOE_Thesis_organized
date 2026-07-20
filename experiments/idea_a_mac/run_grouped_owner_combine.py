"""Offline grouped-wire-unit quality experiment for MoE combine.

Unlike the earlier routed-pair experiment, this driver first performs the
owner-local BF16 weighted reduction for every token and only then applies fake
FP8/MXFP4 to the partial vector used by a non-expanded/local-reduction combine
path. Expanded/LL EP paths can still transmit per-expert responses. It measures
quality and collision structure, not communication latency.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from capture_moe import patch_mixtral_moe
from grouped_owner_combine import GROUPED_POLICIES, grouped_wire_bytes
from metrics import MetricAccumulator
from modeling import load_model, load_tokenizer
from prompts import get_prompts


DEFAULT_STRATEGIES = [
    "grouped_bf16",
    "uniform_fp8",
    "uniform_mxfp4",
    "mixed_rank",
    "mixed_gate_mass",
    "mixed_inputnorm_gate",
    "mixed_pair_contribution",
    "mixed_contribution",
    "global_contribution",
    "token_contribution",
    "mixed_qerr",
    "mixed_oracle",
    "mixed_random",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="allenai/OLMoE-1B-7B-0924")
    p.add_argument("--dataset", default="wikitext2_docs")
    p.add_argument("--split", default="test")
    p.add_argument("--samples", type=int, default=16)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--dataset-seed", type=int, default=None)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--ep-sizes", type=int, nargs="+", default=[8])
    p.add_argument(
        "--mappings",
        nargs="+",
        choices=["contiguous", "round_robin"],
        default=["contiguous", "round_robin"],
    )
    p.add_argument("--tile-vectors", type=int, default=64)
    p.add_argument("--high-fraction", type=float, default=0.5)
    p.add_argument(
        "--strategies", nargs="+", choices=GROUPED_POLICIES, default=DEFAULT_STRATEGIES
    )
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--offline", action="store_true")
    p.add_argument(
        "--output-dir",
        default=(
            "experiments/idea_a_mac/outputs/paper_validation/"
            "grouped_owner_combine"
        ),
    )
    return p.parse_args()


def tokenized_inputs(tokenizer, texts: list[str], seq_len: int):
    for text in texts:
        yield tokenizer(text, return_tensors="pt", truncation=True, max_length=seq_len)


def moe_modules(model) -> list[object]:
    modules = []
    for layer in model.model.layers:
        if hasattr(layer, "block_sparse_moe"):
            modules.append(layer.block_sparse_moe)
        else:
            modules.append(layer.mlp)
    return modules


def restore_forwards(modules: list[object], forwards: list[object]) -> None:
    for module, forward in zip(modules, forwards):
        module.forward = forward


def validate_exact_paths(model, tokenizer, text: str, seq_len: int) -> dict[str, float | bool]:
    """Validate the existing full patch and the EP=1 grouped identity."""
    modules = moe_modules(model)
    original_forwards = [module.forward for module in modules]
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=seq_len)
    with torch.no_grad():
        original = model(**inputs).logits.detach().cpu()

    patch_mixtral_moe(model, "full")
    with torch.no_grad():
        patched_full = model(**inputs).logits.detach().cpu()
    restore_forwards(modules, original_forwards)

    patch_mixtral_moe(
        model,
        "full",
        grouped_owner_policy="grouped_bf16",
        grouped_ep_size=1,
        grouped_owner_mapping="contiguous",
    )
    with torch.no_grad():
        grouped_ep1 = model(**inputs).logits.detach().cpu()
    restore_forwards(modules, original_forwards)

    patched_diff = (patched_full.float() - original.float()).abs()
    ep1_diff = (grouped_ep1.float() - original.float()).abs()
    result: dict[str, float | bool] = {
        "patched_full_bitwise_equal": bool(torch.equal(patched_full, original)),
        "patched_full_max_abs_logit_diff": float(patched_diff.max().item()),
        "grouped_ep1_bitwise_equal": bool(torch.equal(grouped_ep1, original)),
        "grouped_ep1_max_abs_logit_diff": float(ep1_diff.max().item()),
    }
    if not result["patched_full_bitwise_equal"]:
        raise RuntimeError(f"legacy patched full path is not exact: {result}")
    if not result["grouped_ep1_bitwise_equal"]:
        raise RuntimeError(f"EP=1 grouped BF16 identity failed: {result}")
    return result


def run_original(
    model,
    tokenizer,
    texts: list[str],
    seq_len: int,
    sample_id_base: int,
) -> tuple[MetricAccumulator, list[torch.Tensor]]:
    metrics = MetricAccumulator()
    logits_by_sample: list[torch.Tensor] = []
    for local_idx, inputs in enumerate(tokenized_inputs(tokenizer, texts, seq_len)):
        sample_id = sample_id_base + local_idx
        with torch.no_grad():
            logits = model(**inputs).logits.detach().cpu()
        metrics.add(
            sample_id,
            logits,
            inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
        )
        logits_by_sample.append(logits)
        print(f"  original {local_idx + 1}/{len(texts)}", flush=True)
    return metrics, logits_by_sample


def run_grouped_strategy(
    model,
    tokenizer,
    texts: list[str],
    seq_len: int,
    strategy: str,
    ep_size: int,
    mapping: str,
    tile_vectors: int,
    high_fraction: float,
    original_logits: list[torch.Tensor],
    grouped_logits: list[torch.Tensor] | None,
    sample_id_base: int,
) -> tuple[
    MetricAccumulator,
    MetricAccumulator,
    list[torch.Tensor],
    object,
]:
    recorder = patch_mixtral_moe(
        model,
        "full",
        grouped_owner_policy=strategy,
        grouped_ep_size=ep_size,
        grouped_owner_mapping=mapping,
        grouped_tile_vectors=tile_vectors,
        grouped_high_fraction=high_fraction,
    )
    vs_original = MetricAccumulator()
    vs_grouped = MetricAccumulator()
    logits_by_sample: list[torch.Tensor] = []
    for local_idx, inputs in enumerate(tokenized_inputs(tokenizer, texts, seq_len)):
        sample_id = sample_id_base + local_idx
        recorder.set_sample_id(sample_id)
        with torch.no_grad():
            logits = model(**inputs).logits.detach().cpu()
        reference = logits if grouped_logits is None else grouped_logits[local_idx]
        vs_original.add(
            sample_id,
            logits,
            inputs["input_ids"],
            baseline_logits=original_logits[local_idx],
            attention_mask=inputs.get("attention_mask"),
        )
        vs_grouped.add(
            sample_id,
            logits,
            inputs["input_ids"],
            baseline_logits=reference,
            attention_mask=inputs.get("attention_mask"),
        )
        logits_by_sample.append(logits)
        print(
            f"  ep{ep_size}/{mapping}/{strategy} "
            f"{local_idx + 1}/{len(texts)}",
            flush=True,
        )
    return vs_original, vs_grouped, logits_by_sample, recorder


def paired_bootstrap_delta(
    candidate: MetricAccumulator,
    reference: MetricAccumulator,
    n_bootstrap: int,
    seed: int = 20260714,
) -> dict[str, float]:
    if len(candidate.samples) != len(reference.samples):
        raise ValueError("paired sample counts differ")
    n = len(candidate.samples)

    def corpus(rows, indices):
        tokens = sum(rows[i].token_count for i in indices)
        nll = sum(rows[i].nll_sum for i in indices) / max(tokens, 1)
        kl = sum(rows[i].kl_sum for i in indices) / max(tokens, 1)
        return math.exp(nll), kl

    all_indices = np.arange(n)
    cand_ppl, cand_kl = corpus(candidate.samples, all_indices)
    ref_ppl, ref_kl = corpus(reference.samples, all_indices)
    ppl_delta = cand_ppl - ref_ppl
    kl_delta = cand_kl - ref_kl
    if n < 2 or n_bootstrap <= 0:
        return {
            "paired_ppl_delta": ppl_delta,
            "paired_ppl_ci_low": ppl_delta,
            "paired_ppl_ci_high": ppl_delta,
            "paired_kl_delta": kl_delta,
            "paired_kl_ci_low": kl_delta,
            "paired_kl_ci_high": kl_delta,
        }
    rng = np.random.default_rng(seed)
    ppl_samples = np.empty(n_bootstrap)
    kl_samples = np.empty(n_bootstrap)
    for idx in range(n_bootstrap):
        chosen = rng.integers(0, n, size=n)
        cand_ppl, cand_kl = corpus(candidate.samples, chosen)
        ref_ppl, ref_kl = corpus(reference.samples, chosen)
        ppl_samples[idx] = cand_ppl - ref_ppl
        kl_samples[idx] = cand_kl - ref_kl
    return {
        "paired_ppl_delta": ppl_delta,
        "paired_ppl_ci_low": float(np.quantile(ppl_samples, 0.025)),
        "paired_ppl_ci_high": float(np.quantile(ppl_samples, 0.975)),
        "paired_kl_delta": kl_delta,
        "paired_kl_ci_low": float(np.quantile(kl_samples, 0.025)),
        "paired_kl_ci_high": float(np.quantile(kl_samples, 0.975)),
    }


def source_manifest() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    names = (
        "run_grouped_owner_combine.py",
        "grouped_owner_combine.py",
        "capture_moe.py",
        "fake_quant.py",
        "metrics.py",
        "modeling.py",
        "prompts.py",
    )
    return {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in names
    }


def data_manifest(
    tokenizer, texts: list[str], split: str, seq_len: int, offset: int
) -> list[dict[str, int | str]]:
    rows = []
    for local_idx, text in enumerate(texts):
        ids = tokenizer(text, add_special_tokens=True)["input_ids"]
        rows.append(
            {
                "sample_id": offset + local_idx,
                "split": split,
                "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "characters": len(text),
                "tokens_before_truncation": len(ids),
                "tokens_used": min(len(ids), seq_len),
            }
        )
    return rows


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for _, row in df[columns].iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, (float, np.floating)):
                values.append(f"{value:.6g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if "grouped_bf16" not in args.strategies:
        raise ValueError("--strategies must include grouped_bf16 as the reference")
    if args.tile_vectors < 1:
        raise ValueError("--tile-vectors must be positive")
    if not 0.0 <= args.high_fraction <= 1.0:
        raise ValueError("--high-fraction must be in [0, 1]")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    texts = get_prompts(
        args.dataset,
        args.samples,
        offset=args.offset,
        split=args.split,
        seed=args.dataset_seed,
    )
    tokenizer = load_tokenizer(args.model, local_files_only=args.offline)
    model, load_seconds = load_model(
        args.model, dtype_name=args.dtype, local_files_only=args.offline
    )
    modules = moe_modules(model)
    original_forwards = [module.forward for module in modules]
    num_experts = int(getattr(modules[0], "num_experts"))
    hidden_size = int(model.config.hidden_size)
    for ep_size in args.ep_sizes:
        if ep_size not in (2, 4, 8):
            raise ValueError("this experiment preregisters EP sizes 2, 4, or 8")
        if ep_size > num_experts:
            raise ValueError(f"ep_size={ep_size} exceeds num_experts={num_experts}")

    exactness = validate_exact_paths(model, tokenizer, texts[0], args.seq_len)
    restore_forwards(modules, original_forwards)
    print(
        f"loaded {args.model} in {load_seconds:.1f}s; experts={num_experts}; "
        f"exactness={exactness}",
        flush=True,
    )

    original_metrics, original_logits = run_original(
        model, tokenizer, texts, args.seq_len, args.offset
    )
    original_ppl = original_metrics.corpus_ppl

    summary_rows: list[dict[str, int | float | str]] = []
    layer_rows: list[dict[str, int | float | str]] = []
    tile_rows: list[dict[str, int | float | str]] = []
    sample_rows: list[dict[str, int | float | str]] = original_metrics.sample_rows(
        "original"
    )
    paired_rows: list[dict[str, int | float | str]] = []

    for ep_size in args.ep_sizes:
        for mapping in args.mappings:
            print(f"running EP={ep_size}, mapping={mapping}", flush=True)
            grouped_orig, grouped_self, grouped_logits, grouped_recorder = run_grouped_strategy(
                model,
                tokenizer,
                texts,
                args.seq_len,
                "grouped_bf16",
                ep_size,
                mapping,
                args.tile_vectors,
                args.high_fraction,
                original_logits,
                None,
                args.offset,
            )
            grouped_ppl = grouped_orig.corpus_ppl
            metrics_by_name: dict[str, MetricAccumulator] = {
                "grouped_bf16": grouped_self
            }
            strategy_results = [
                ("grouped_bf16", grouped_orig, grouped_self, grouped_recorder)
            ]

            for strategy in args.strategies:
                if strategy == "grouped_bf16":
                    continue
                vs_original, vs_grouped, _, recorder = run_grouped_strategy(
                    model,
                    tokenizer,
                    texts,
                    args.seq_len,
                    strategy,
                    ep_size,
                    mapping,
                    args.tile_vectors,
                    args.high_fraction,
                    original_logits,
                    grouped_logits,
                    args.offset,
                )
                metrics_by_name[strategy] = vs_grouped
                strategy_results.append(
                    (strategy, vs_original, vs_grouped, recorder)
                )

            for strategy, vs_original, vs_grouped, recorder in strategy_results:
                grouped_stats = recorder.grouped_owner_summary()
                local_error_rows = recorder.error_rows()
                local_sq_error = sum(float(row["sq_error"]) for row in local_error_rows)
                local_sq_full = sum(float(row["sq_full"]) for row in local_error_rows)
                wire_bytes = grouped_wire_bytes(
                    hidden_size,
                    int(grouped_stats["bf16_vectors"]),
                    int(grouped_stats["high_vectors"]),
                    int(grouped_stats["low_vectors"]),
                )
                grouped_bf16_bytes = (
                    int(grouped_stats["grouped_vectors_m"]) * 2 * hidden_size
                )
                pair_bf16_bytes = (
                    int(grouped_stats["routed_pairs_n"]) * 2 * hidden_size
                )
                orig_summary = vs_original.bootstrap_summary(args.bootstrap)
                grouped_summary = vs_grouped.bootstrap_summary(args.bootstrap)
                summary_rows.append(
                    {
                        "ep_size": ep_size,
                        "mapping": mapping,
                        "strategy": strategy,
                        "corpus_ppl": vs_original.corpus_ppl,
                        "ppl_delta_vs_original": vs_original.corpus_ppl - original_ppl,
                        "ppl_delta_vs_grouped_bf16": vs_original.corpus_ppl - grouped_ppl,
                        "mean_token_kl_vs_original": vs_original.mean_token_kl,
                        "mean_token_kl_vs_grouped_bf16": vs_grouped.mean_token_kl,
                        "kl_grouped_ci_low": grouped_summary["mean_token_kl_ci_low"],
                        "kl_grouped_ci_high": grouped_summary["mean_token_kl_ci_high"],
                        "ppl_ci_low": orig_summary["corpus_ppl_ci_low"],
                        "ppl_ci_high": orig_summary["corpus_ppl_ci_high"],
                        "local_relative_mse": local_sq_error / max(local_sq_full, 1e-12),
                        **grouped_stats,
                        "wire_bytes": wire_bytes,
                        "wire_saving_vs_grouped_bf16": 1.0
                        - wire_bytes / max(grouped_bf16_bytes, 1),
                        "wire_saving_vs_pair_bf16_counterfactual": 1.0
                        - wire_bytes / max(pair_bf16_bytes, 1),
                    }
                )
                sample_rows.extend(
                    {
                        **row,
                        "ep_size": ep_size,
                        "mapping": mapping,
                        "kl_reference": "grouped_bf16",
                    }
                    for row in vs_grouped.sample_rows(strategy)
                )
                for row in recorder.grouped_owner_rows():
                    layer_rows.append(
                        {**row, "ep_size": ep_size, "mapping": mapping, "strategy": strategy}
                    )
                for row in recorder.grouped_tile_rows:
                    tile_rows.append(
                        {**row, "ep_size": ep_size, "mapping": mapping, "strategy": strategy}
                    )

            for candidate in metrics_by_name:
                if candidate in ("grouped_bf16", "mixed_rank"):
                    continue
                for reference in ("mixed_rank", "mixed_gate_mass"):
                    if reference not in metrics_by_name or candidate == reference:
                        continue
                    paired = paired_bootstrap_delta(
                        metrics_by_name[candidate],
                        metrics_by_name[reference],
                        args.bootstrap,
                    )
                    paired_rows.append(
                        {
                            "ep_size": ep_size,
                            "mapping": mapping,
                            "candidate": candidate,
                            "reference": reference,
                            **paired,
                        }
                    )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out / "grouped_owner_summary.csv", index=False)
    pd.DataFrame(layer_rows).to_csv(out / "grouped_owner_by_layer.csv", index=False)
    pd.DataFrame(tile_rows).to_csv(out / "grouped_owner_tiles.csv", index=False)
    pd.DataFrame(sample_rows).to_csv(out / "sample_metrics.csv", index=False)
    pd.DataFrame(paired_rows).to_csv(out / "paired_comparisons.csv", index=False)
    pd.DataFrame(data_manifest(tokenizer, texts, args.split, args.seq_len, args.offset)).to_csv(
        out / "data_manifest.csv", index=False
    )
    (out / "exactness.json").write_text(
        json.dumps(exactness, indent=2), encoding="utf-8"
    )
    (out / "source_manifest.json").write_text(
        json.dumps(source_manifest(), indent=2), encoding="utf-8"
    )
    config = vars(args) | {
        "model_load_seconds": load_seconds,
        "num_experts": num_experts,
        "hidden_size": hidden_size,
        "original_corpus_ppl": original_ppl,
    }
    (out / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    columns = [
        "ep_size",
        "mapping",
        "strategy",
        "n_over_m",
        "collision_pair_fraction",
        "observed_high_fraction",
        "wire_saving_vs_grouped_bf16",
        "local_relative_mse",
        "mean_token_kl_vs_grouped_bf16",
        "ppl_delta_vs_grouped_bf16",
    ]
    table = markdown_table(summary_df, columns)
    report = f"""# Grouped-Owner Combine Offline Experiment

This experiment models a non-expanded/local-reduction EP combine wire unit:
selected expert outputs on the same owner are multiplied by router weights and
reduced in BF16 before fake FP8/MXFP4. DeepEP/NCCL-style LL or expanded paths
may instead transmit per-expert responses. This is a quality and trace-structure
experiment, not a kernel or network benchmark.

## Configuration

- model: `{args.model}`
- data: `{args.dataset}:{args.split}`, offset `{args.offset}`, n=`{args.samples}`
- sequence length: `{args.seq_len}`
- EP sizes: `{args.ep_sizes}`; mappings: `{args.mappings}`
- peer tile: `{args.tile_vectors}` present grouped vectors
- exact per-tile FP8 target: `{args.high_fraction}` (implemented as `floor(n*f)`)
- patched full exact: `{exactness['patched_full_bitwise_equal']}`
- grouped EP=1 exact: `{exactness['grouped_ep1_bitwise_equal']}`

## Results

{table}

## Metric definitions and boundaries

- `N` is the number of routed expert pairs; `M` is the number of nonempty
  token-owner vectors after owner-local reduction. `N/M` and the collision
  fractions characterize existing local combine aggregation; they are not a
  new compression saving.
- `grouped_bf16` is the precision reference for EP>1. Its logit drift from the
  original single-process path is reported because owner grouping changes BF16
  associativity. Quantized KL is therefore reported against both references.
- Mixed selectors get exactly the same `floor(n*f)` FP8 cardinality within each
  present-vector tile of each owner. Total counts can deviate slightly from
  `f*M` because final short tiles are handled independently. A real decode
  implementation must define fallback/carry behavior and must not wait to fill
  a tile on the critical path.
- `mixed_rank` uses a gate/output-free sum of reciprocal routed ranks.
  `mixed_gate_mass` uses the owner-local sum of router weights.
  `mixed_inputnorm_gate` multiplies that mass by the origin-available input norm.
  `mixed_pair_contribution` uses `sum(g*||o||)` before owner aggregation, while
  `mixed_contribution` uses `||sum(g*o)||` and therefore includes cancellation.
  `global_contribution` and `token_contribution` keep the same score/codec but
  remove the peer quota, isolating its quality cost; their per-peer lane counts
  are variable and they are not the proposed regular layout.
  `mixed_qerr` uses its fake-MXFP4 error energy.
- `mixed_oracle` is an exact *one-owner-group* local FP4-to-FP8 intervention
  score that includes same-token cross-owner terms. Multiple simultaneous
  upgrades still interact, so it is not a global combinatorial optimum.
- Wire accounting includes the fake formats' vector/block scale bytes but omits
  alignment, padding, membership masks, headers, packing, and GPU kernel cost.
- Upstream approximation can change later-layer routes. A formal run should add
  disjoint datasets, larger models, native FP4/FP8 kernels, and real EP traces.
"""
    (out / "grouped_owner_report.md").write_text(report, encoding="utf-8")
    print(summary_df[columns].to_string(index=False), flush=True)
    print(f"saved to {out}", flush=True)


if __name__ == "__main__":
    main()
