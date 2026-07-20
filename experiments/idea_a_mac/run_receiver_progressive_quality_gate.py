#!/usr/bin/env python3
"""Quality gate for receiver-credited progressive EP communication.

The candidate data path sends one 4-bit base vector for every routed pair and
optionally sends a second 4-bit residual for selected pairs.  At 50% refinement
the raw payload averages 6 bits/element.  The strongest equal-wire control sends
FP8 for some pairs and one 4-bit vector for the rest.

This experiment compares both representations on fresh documents with:
  * deployable gate-weight selection;
  * sender-output-aware quantization-error selection as an upper bound;
  * exact scale-metadata-aware wire accounting;
  * paired document bootstrap.

It evaluates representation quality only.  It does not claim that a receiver
can deliver credits in time or that two real communication lanes are fast.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from metrics import MetricAccumulator
from modeling import load_model, load_tokenizer
from policies import ApproxPolicy
from prompts import get_prompts
from run_layer_budget_experiment import run_logits


def vector_wire_bytes(precision: str, hidden_size: int) -> int:
    if precision in ("full", "bf16"):
        return 2 * hidden_size
    if precision == "fp8":
        return hidden_size + math.ceil(hidden_size / 128)
    if precision == "int4":
        return math.ceil(hidden_size / 2) + 4
    if precision == "mxfp4":
        return math.ceil(hidden_size / 2) + math.ceil(hidden_size / 32)
    raise ValueError(f"unsupported precision: {precision}")


def matched_direct_fraction(precision: str, hidden_size: int) -> float:
    """Fraction of direct vectors sent at 4-bit for residual 50% equal wire."""
    low = vector_wire_bytes(precision, hidden_size)
    high = vector_wire_bytes("fp8", hidden_size)
    progressive = 1.5 * low
    fraction = (high - progressive) / (high - low)
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(
            f"no direct FP8/{precision} mixture matches progressive wire bytes"
        )
    return fraction


def policy_wire_bytes_per_pair(
    policy_name: str,
    top_k: int,
    hidden_size: int,
) -> float:
    """Compute metadata-aware mean bytes from exact policy cardinality.

    A synthetic full block is sufficient because all compared policies enforce
    exact cardinality in each block and have no data-dependent byte count.
    """
    block_tokens = 16000
    selected = np.zeros((block_tokens, top_k), dtype=np.int64)
    weights = np.tile(np.linspace(1.0, 0.1, top_k), (block_tokens, 1))
    import torch

    byte_sizes = ApproxPolicy(policy_name).bytes_per_element_for_selected(
        torch.from_numpy(selected),
        num_experts=1,
        routing_weights=torch.from_numpy(weights).float(),
    )
    average_raw = float(byte_sizes.float().mean().item())
    if policy_name == "uniform_fp8":
        return float(vector_wire_bytes("fp8", hidden_size))
    if policy_name.startswith("uniform_"):
        precision = policy_name.removeprefix("uniform_")
        return float(vector_wire_bytes(precision, hidden_size))
    precision = policy_name.rsplit("_", 1)[-1]
    low = vector_wire_bytes(precision, hidden_size)
    if "_residual_" in policy_name:
        refined_fraction = (average_raw - 0.5) / 0.5
        return low * (1.0 + refined_fraction)
    high = vector_wire_bytes("fp8", hidden_size)
    low_fraction = (1.0 - average_raw) / 0.5
    return low_fraction * low + (1.0 - low_fraction) * high


def paired_bootstrap(
    sample_rows: pd.DataFrame,
    candidate: str,
    reference: str,
    repeats: int,
    seed: int,
) -> dict[str, float | str]:
    pivot_kl = sample_rows.pivot(
        index="sample_id", columns="strategy", values="kl_sum"
    )
    pivot_tokens = sample_rows.pivot(
        index="sample_id", columns="strategy", values="token_count"
    )
    if candidate not in pivot_kl or reference not in pivot_kl:
        raise ValueError(f"missing paired strategies: {candidate}, {reference}")
    candidate_kl = pivot_kl[candidate].to_numpy()
    reference_kl = pivot_kl[reference].to_numpy()
    candidate_tokens = pivot_tokens[candidate].to_numpy()
    reference_tokens = pivot_tokens[reference].to_numpy()
    indices = np.arange(len(pivot_kl))
    rng = np.random.default_rng(seed)
    differences = np.empty(repeats, dtype=np.float64)
    for bootstrap_index in range(repeats):
        chosen = rng.choice(indices, size=len(indices), replace=True)
        candidate_mean = candidate_kl[chosen].sum() / candidate_tokens[chosen].sum()
        reference_mean = reference_kl[chosen].sum() / reference_tokens[chosen].sum()
        differences[bootstrap_index] = reference_mean - candidate_mean
    point_candidate = candidate_kl.sum() / candidate_tokens.sum()
    point_reference = reference_kl.sum() / reference_tokens.sum()
    return {
        "candidate": candidate,
        "reference": reference,
        "reference_minus_candidate_kl": point_reference - point_candidate,
        "ci_low": float(np.quantile(differences, 0.025)),
        "ci_high": float(np.quantile(differences, 0.975)),
        "probability_candidate_better": float((differences > 0).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--dataset", default="wikitext103_docs")
    parser.add_argument("--split", default="train")
    parser.add_argument("--samples", type=int, default=24)
    parser.add_argument("--offset", type=int, default=160)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--block-tokens", type=int, default=16)
    parser.add_argument("--precisions", default="int4,mxfp4")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    texts = get_prompts(
        args.dataset,
        args.samples,
        offset=args.offset,
        split=args.split,
        seed=args.seed,
    )
    tokenizer = load_tokenizer(args.model, local_files_only=args.offline)
    model, load_seconds = load_model(
        args.model,
        dtype_name=args.dtype,
        local_files_only=args.offline,
    )
    hidden_size = int(model.config.hidden_size)
    top_k = int(model.config.num_experts_per_tok)
    precisions = [item.strip() for item in args.precisions.split(",") if item.strip()]

    full_metrics, full_logits, _ = run_logits(
        model,
        tokenizer,
        texts,
        args.seq_len,
        "full",
        1,
        args.offset,
    )
    strategies = ["uniform_fp8"]
    matched_fraction_by_precision: dict[str, float] = {}
    for precision in precisions:
        fraction = matched_direct_fraction(precision, hidden_size)
        matched_fraction_by_precision[precision] = fraction
        encoded_fraction = int(round(fraction * 1000))
        strategies.extend(
            [
                f"uniform_{precision}",
                f"block_gate{args.block_tokens}_f{encoded_fraction}_{precision}",
                f"block_gate{args.block_tokens}_residual_{precision}",
                f"block_qerr{args.block_tokens}_f{encoded_fraction}_{precision}",
                f"block_reserr{args.block_tokens}_residual_{precision}",
            ]
        )

    aggregate_rows: list[dict] = []
    sample_rows: list[dict] = []
    full_summary = full_metrics.bootstrap_summary(args.bootstrap)
    full_summary.update(
        {
            "strategy": "full",
            "wire_bytes_per_pair": vector_wire_bytes("full", hidden_size),
            "wire_saving_vs_bf16": 0.0,
            "ppl_delta_vs_full": 0.0,
        }
    )
    aggregate_rows.append(full_summary)
    sample_rows.extend(full_metrics.sample_rows("full"))

    for strategy in strategies:
        print(f"[{args.model_key}] running {strategy}", flush=True)
        metrics, _, _ = run_logits(
            model,
            tokenizer,
            texts,
            args.seq_len,
            strategy,
            1,
            args.offset,
            baseline_logits=full_logits,
        )
        wire_bytes = policy_wire_bytes_per_pair(strategy, top_k, hidden_size)
        summary = metrics.bootstrap_summary(args.bootstrap)
        summary.update(
            {
                "strategy": strategy,
                "wire_bytes_per_pair": wire_bytes,
                "wire_saving_vs_bf16": 1
                - wire_bytes / vector_wire_bytes("full", hidden_size),
                "ppl_delta_vs_full": metrics.corpus_ppl - full_metrics.corpus_ppl,
            }
        )
        aggregate_rows.append(summary)
        sample_rows.extend(metrics.sample_rows(strategy))
        pd.DataFrame(aggregate_rows).to_csv(
            output / "quality_gate.partial.csv", index=False
        )

    aggregates = pd.DataFrame(aggregate_rows)
    samples = pd.DataFrame(sample_rows)
    comparisons: list[dict] = []
    for precision in precisions:
        fraction = int(round(matched_fraction_by_precision[precision] * 1000))
        comparisons.append(
            paired_bootstrap(
                samples,
                f"block_gate{args.block_tokens}_residual_{precision}",
                f"block_gate{args.block_tokens}_f{fraction}_{precision}",
                args.bootstrap,
                args.seed + len(comparisons),
            )
        )
        comparisons.append(
            paired_bootstrap(
                samples,
                f"block_reserr{args.block_tokens}_residual_{precision}",
                f"block_qerr{args.block_tokens}_f{fraction}_{precision}",
                args.bootstrap,
                args.seed + len(comparisons),
            )
        )
    comparison_frame = pd.DataFrame(comparisons)
    aggregates.to_csv(output / "quality_gate.csv", index=False)
    samples.to_csv(output / "sample_metrics.csv", index=False)
    comparison_frame.to_csv(output / "paired_equal_wire.csv", index=False)
    metadata = {
        "model": args.model,
        "model_key": args.model_key,
        "dataset": args.dataset,
        "split": args.split,
        "samples": args.samples,
        "offset": args.offset,
        "seq_len": args.seq_len,
        "block_tokens": args.block_tokens,
        "hidden_size": hidden_size,
        "top_k": top_k,
        "precisions": precisions,
        "matched_direct_low_fraction": matched_fraction_by_precision,
        "load_seconds": load_seconds,
        "evidence_boundary": (
            "fake-quant representation quality with metadata-aware wire bytes; "
            "not a real two-lane communication kernel"
        ),
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    print("\nQuality gate:")
    print(
        aggregates[
            [
                "strategy",
                "wire_bytes_per_pair",
                "wire_saving_vs_bf16",
                "corpus_ppl",
                "ppl_delta_vs_full",
                "mean_token_kl",
                "mean_token_kl_ci_low",
                "mean_token_kl_ci_high",
            ]
        ].to_string(index=False)
    )
    print("\nEqual-wire paired comparisons:")
    print(comparison_frame.to_string(index=False))


if __name__ == "__main__":
    main()
