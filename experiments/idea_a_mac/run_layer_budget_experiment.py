"""Calibrate and test regular layer-wise tail-INT4 budgets.

The experiment asks whether a fixed per-layer tail count ``m_l`` can close some
of the quality gap between a globally fixed rank split and token-wise gate
selectors while preserving a regular two-lane layout.

Calibration profiles only one perturbation per layer: all layers use FP8 and
the target layer upgrades its last ``base_tail`` ranks to INT4.  The profile is
used only to rank layers.  Several frozen, equal-payload allocations are then
evaluated end-to-end on a disjoint test slice; profile KL is never treated as
an additive prediction of test KL.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from capture_moe import patch_mixtral_moe
from metrics import MetricAccumulator
from modeling import load_model, load_tokenizer
from prompts import get_prompts


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="allenai/OLMoE-1B-7B-0924")
    p.add_argument("--dataset", default="wikitext2")
    p.add_argument("--dataset-split", default="validation")
    p.add_argument("--calibration-samples", type=int, default=8)
    p.add_argument("--test-samples", type=int, default=32)
    p.add_argument("--calibration-offset", type=int, default=0)
    p.add_argument("--test-offset", type=int, default=128)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--num-receiver-groups", type=int, default=1)
    p.add_argument("--base-tail", type=int, default=None)
    p.add_argument(
        "--tail-precision",
        default="int4",
        choices=["int4", "mxfp4", "nvfp4"],
    )
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--offline", action="store_true")
    p.add_argument(
        "--output-dir",
        default="experiments/idea_a_mac/outputs/paper_validation/layer_budget",
    )
    return p.parse_args()


def tokenized_inputs(tokenizer, texts: list[str], seq_len: int, device=None):
    for text in texts:
        encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=seq_len)
        if device is not None:
            encoded = {k: v.to(device) for k, v in encoded.items()}
        yield encoded


def build_lut(
    tail_counts: list[int],
    top_k: int,
    num_groups: int,
    tail_precision: str = "int4",
) -> dict[tuple[int, int, int], str]:
    return {
        (layer, group, rank): tail_precision
        if rank > top_k - tail_counts[layer]
        else "fp8"
        for layer in range(len(tail_counts))
        for group in range(num_groups)
        for rank in range(1, top_k + 1)
    }


def run_logits(
    model,
    tokenizer,
    texts: list[str],
    seq_len: int,
    policy_name: str,
    num_groups: int,
    sample_id_base: int,
    baseline_logits: list[torch.Tensor] | None = None,
    lut: dict[tuple[int, int, int], str] | None = None,
    record_routes: bool = False,
) -> tuple[MetricAccumulator, list[torch.Tensor], object]:
    recorder = patch_mixtral_moe(
        model,
        policy_name,
        num_receiver_groups=num_groups,
        lut=lut,
        record_routes=record_routes,
    )
    metrics = MetricAccumulator()
    logits_by_sample: list[torch.Tensor] = []
    for local_idx, inputs in enumerate(tokenized_inputs(tokenizer, texts, seq_len, device=model.device)):
        recorder.set_sample_id(sample_id_base + local_idx)
        with torch.no_grad():
            logits = model(**inputs).logits.detach().cpu()
        metrics.add(
            sample_id_base + local_idx,
            logits,
            inputs["input_ids"].cpu(),
            baseline_logits=None if baseline_logits is None else baseline_logits[local_idx],
            attention_mask=inputs.get("attention_mask").cpu() if inputs.get("attention_mask") is not None else None,
        )
        logits_by_sample.append(logits)
    return metrics, logits_by_sample, recorder


def allocate_two_level(
    scores: dict[int, float],
    num_layers: int,
    base_tail: int,
    delta: int,
    high_count: int,
    reverse: bool = False,
) -> list[int]:
    if high_count * 2 > num_layers:
        raise ValueError("high_count must cover at most half of the layers")
    ordered = sorted(scores, key=scores.get, reverse=True)
    sensitive = ordered[:high_count]
    insensitive = ordered[-high_count:]
    counts = [base_tail] * num_layers
    for layer in sensitive:
        counts[layer] = base_tail + delta if reverse else base_tail - delta
    for layer in insensitive:
        counts[layer] = base_tail - delta if reverse else base_tail + delta
    return counts


def dataframe_to_markdown(df: pd.DataFrame, columns: list[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for _, row in df[columns].iterrows():
        values: list[str] = []
        for column in columns:
            value = row[column]
            values.append(f"{value:.6f}" if isinstance(value, (float, np.floating)) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def paired_bootstrap_vs_reference(
    sample_df: pd.DataFrame,
    reference: str,
    n_bootstrap: int,
    seed: int = 42,
) -> pd.DataFrame:
    """Paired sample bootstrap of reference KL minus candidate KL."""
    pivot = sample_df.pivot(
        index="sample_id", columns="strategy", values=["kl_sum", "token_count"]
    )
    if reference not in pivot["kl_sum"].columns:
        raise ValueError(f"missing paired-bootstrap reference: {reference}")
    sample_indices = np.arange(len(pivot))
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | str]] = []
    ref_kl = pivot[("kl_sum", reference)].to_numpy()
    ref_tokens = pivot[("token_count", reference)].to_numpy()
    for strategy in pivot["kl_sum"].columns:
        if strategy in ("full", reference):
            continue
        cand_kl = pivot[("kl_sum", strategy)].to_numpy()
        cand_tokens = pivot[("token_count", strategy)].to_numpy()
        differences = np.empty(n_bootstrap, dtype=np.float64)
        for bootstrap_idx in range(n_bootstrap):
            chosen = rng.choice(sample_indices, size=len(sample_indices), replace=True)
            reference_mean = ref_kl[chosen].sum() / max(ref_tokens[chosen].sum(), 1)
            candidate_mean = cand_kl[chosen].sum() / max(cand_tokens[chosen].sum(), 1)
            differences[bootstrap_idx] = reference_mean - candidate_mean
        point_reference = ref_kl.sum() / max(ref_tokens.sum(), 1)
        point_candidate = cand_kl.sum() / max(cand_tokens.sum(), 1)
        rows.append(
            {
                "reference": reference,
                "candidate": strategy,
                "reference_minus_candidate_kl": point_reference - point_candidate,
                "ci_low": float(np.quantile(differences, 0.025)),
                "ci_high": float(np.quantile(differences, 0.975)),
                "probability_candidate_better": float((differences > 0).mean()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    calibration_texts = get_prompts(
        args.dataset,
        args.calibration_samples,
        offset=args.calibration_offset,
        split=args.dataset_split,
    )
    test_texts = get_prompts(
        args.dataset,
        args.test_samples,
        offset=args.test_offset,
        split=args.dataset_split,
    )
    if set(calibration_texts) & set(test_texts):
        raise RuntimeError("calibration and test prompt text overlap")

    tokenizer = load_tokenizer(args.model, local_files_only=args.offline)
    model, load_seconds = load_model(
        args.model, dtype_name=args.dtype, local_files_only=args.offline
    )
    num_layers = len(model.model.layers)
    top_k = int(getattr(model.config, "num_experts_per_tok", 8))
    base_tail = args.base_tail if args.base_tail is not None else max(1, top_k // 2)
    if not 0 < base_tail < top_k:
        raise ValueError(f"base_tail must be in [1, {top_k - 1}], got {base_tail}")
    print(
        f"model loaded in {load_seconds:.1f}s; layers={num_layers}, top_k={top_k}, "
        f"base_tail={base_tail}",
        flush=True,
    )

    print("capturing full calibration routes...", flush=True)
    _, _, route_recorder = run_logits(
        model,
        tokenizer,
        calibration_texts,
        args.seq_len,
        "full",
        args.num_receiver_groups,
        args.calibration_offset,
        record_routes=True,
    )
    calibration_routes = pd.DataFrame(route_recorder.route_rows())
    calibration_routes.to_csv(out / "calibration_routes.csv", index=False)

    print("running uniform-FP8 calibration reference...", flush=True)
    _, calibration_fp8_logits, _ = run_logits(
        model,
        tokenizer,
        calibration_texts,
        args.seq_len,
        "uniform_fp8",
        args.num_receiver_groups,
        args.calibration_offset,
    )

    profile_rows: list[dict[str, float | int]] = []
    profile_scores: dict[int, float] = {}
    p95_scores: dict[int, float] = {}
    for layer in range(num_layers):
        counts = [0] * num_layers
        counts[layer] = base_tail
        metrics, _, recorder = run_logits(
            model,
            tokenizer,
            calibration_texts,
            args.seq_len,
            "lut",
            args.num_receiver_groups,
            args.calibration_offset,
            baseline_logits=calibration_fp8_logits,
            lut=build_lut(
                counts, top_k, args.num_receiver_groups, args.tail_precision
            ),
        )
        sample_kls = np.asarray([sample.mean_token_kl for sample in metrics.samples])
        mean_kl = metrics.mean_token_kl
        p95_kl = float(np.quantile(sample_kls, 0.95))
        profile_scores[layer] = mean_kl
        p95_scores[layer] = p95_kl
        profile_rows.append(
            {
                "layer": layer,
                "profile_tail_count": base_tail,
                "mean_token_kl_vs_uniform_fp8": mean_kl,
                "p95_sample_token_kl_vs_uniform_fp8": p95_kl,
                "payload_saving_vs_bf16": recorder.total_byte_saving(),
            }
        )
        print(f"  layer {layer + 1}/{num_layers}: KL={mean_kl:.7f}, p95={p95_kl:.7f}", flush=True)
    profile_df = pd.DataFrame(profile_rows)

    tail_start = top_k - base_tail + 1
    tail_routes = calibration_routes[calibration_routes["rank"] >= tail_start]
    gate_mass_scores = (
        tail_routes.groupby(["sample_id", "layer", "token_position"])["gate_weight"]
        .sum()
        .groupby("layer")
        .mean()
        .to_dict()
    )
    gate_mass_scores = {int(layer): float(value) for layer, value in gate_mass_scores.items()}
    profile_df["mean_tail_gate_mass"] = profile_df["layer"].map(gate_mass_scores)
    profile_df.to_csv(out / "layer_profile.csv", index=False)

    half = num_layers // 2
    quarter = max(1, num_layers // 4)
    narrow_name = f"kl_profile_{base_tail - 1}_{base_tail + 1}"
    p95_name = f"p95_profile_{base_tail - 1}_{base_tail + 1}"
    gate_name = f"gate_mass_profile_{base_tail - 1}_{base_tail + 1}"
    anti_name = f"anti_kl_profile_{base_tail - 1}_{base_tail + 1}"
    allocations: dict[str, list[int]] = {
        f"fixed_tail{base_tail}": [base_tail] * num_layers,
        narrow_name: allocate_two_level(
            profile_scores, num_layers, base_tail, delta=1, high_count=half
        ),
        p95_name: allocate_two_level(
            p95_scores, num_layers, base_tail, delta=1, high_count=half
        ),
        gate_name: allocate_two_level(
            gate_mass_scores, num_layers, base_tail, delta=1, high_count=half
        ),
        anti_name: allocate_two_level(
            profile_scores, num_layers, base_tail, delta=1, high_count=half, reverse=True
        ),
    }
    if base_tail >= 2 and base_tail + 2 <= top_k:
        wide_name = f"kl_profile_{base_tail - 2}_{base_tail}_{base_tail + 2}"
        allocations[wide_name] = allocate_two_level(
            profile_scores, num_layers, base_tail, delta=2, high_count=quarter
        )
    expected_total = num_layers * base_tail
    for name, counts in allocations.items():
        if sum(counts) != expected_total or min(counts) < 0 or max(counts) > top_k:
            raise RuntimeError(f"invalid allocation {name}: {counts}")
    (out / "allocations.json").write_text(
        json.dumps(
            {
                "model": args.model,
                "calibration_offset": args.calibration_offset,
                "calibration_samples": args.calibration_samples,
                "test_offset": args.test_offset,
                "test_samples": args.test_samples,
                "top_k": top_k,
                "base_tail": base_tail,
                "tail_precision": args.tail_precision,
                "allocations": allocations,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    for name, counts in allocations.items():
        print(f"{name}: {counts}", flush=True)

    print("running held-out full reference...", flush=True)
    full_metrics, full_test_logits, _ = run_logits(
        model,
        tokenizer,
        test_texts,
        args.seq_len,
        "full",
        args.num_receiver_groups,
        args.test_offset,
    )
    rows: list[dict[str, float | int | str]] = []
    sample_rows = full_metrics.sample_rows("full")
    full_row = full_metrics.bootstrap_summary(args.bootstrap)
    full_row.update(
        {
            "strategy": "full",
            "theoretical_payload_saving_vs_bf16": 0.0,
            "ppl_delta_vs_full": 0.0,
        }
    )
    rows.append(full_row)

    for name, counts in allocations.items():
        print(f"running held-out {name}...", flush=True)
        metrics, _, recorder = run_logits(
            model,
            tokenizer,
            test_texts,
            args.seq_len,
            "lut",
            args.num_receiver_groups,
            args.test_offset,
            baseline_logits=full_test_logits,
            lut=build_lut(
                counts, top_k, args.num_receiver_groups, args.tail_precision
            ),
        )
        row = metrics.bootstrap_summary(args.bootstrap)
        row.update(
            {
                "strategy": name,
                "theoretical_payload_saving_vs_bf16": recorder.total_byte_saving(),
                "ppl_delta_vs_full": metrics.corpus_ppl - full_metrics.corpus_ppl,
            }
        )
        rows.append(row)
        sample_rows.extend(metrics.sample_rows(name))
        pd.DataFrame(rows).to_csv(out / "layer_budget_results.partial.csv", index=False)

    results = pd.DataFrame(rows)
    results.to_csv(out / "layer_budget_results.csv", index=False)
    sample_df = pd.DataFrame(sample_rows)
    sample_df.to_csv(out / "sample_metrics.csv", index=False)
    paired_df = paired_bootstrap_vs_reference(
        sample_df, f"fixed_tail{base_tail}", args.bootstrap
    )
    paired_df.to_csv(out / "paired_bootstrap_vs_fixed.csv", index=False)
    columns = [
        "strategy",
        "theoretical_payload_saving_vs_bf16",
        "corpus_ppl",
        "ppl_delta_vs_full",
        "mean_token_kl",
        "mean_token_kl_ci_low",
        "mean_token_kl_ci_high",
    ]
    report = f"""# Layer-Wise Fixed Tail-Budget Experiment

## Boundary

This is a Mac fake-quant quality experiment.  Calibration profiles are used
only to rank layers; their KL values are not added to predict end-to-end KL.
Every frozen allocation is evaluated end-to-end on a disjoint held-out slice.

## Setup

- model: `{args.model}`
- calibration: `{args.dataset}:{args.dataset_split}` offset `{args.calibration_offset}`, n=`{args.calibration_samples}`
- test: `{args.dataset}:{args.dataset_split}` offset `{args.test_offset}`, n=`{args.test_samples}`
- top-k: `{top_k}`; base tail count: `{base_tail}`; total low-bit layer-rank slots: `{expected_total}`
- tail precision: `{args.tail_precision}`

## Results

{dataframe_to_markdown(results, columns)}

## Paired bootstrap versus fixed tail

Positive `reference_minus_candidate_kl` means the candidate is better.

{dataframe_to_markdown(paired_df, list(paired_df.columns))}

## Allocation interpretation

- `kl_profile_*`: protect layers with larger single-layer incremental KL and spend more tail ranks on less-sensitive layers.
- `p95_profile_*`: the same idea using the P95 per-sample KL risk score.
- `gate_mass_profile_*`: protect layers whose fixed tail carries more calibration gate mass.
- `anti_*`: equal-payload negative control that deliberately spends more INT4 ranks on sensitive layers.

These are regular per-layer fixed layouts.  A positive result still needs a
real two-lane kernel; a negative result means layer-wise rank budgeting does
not close the quality gap to token-wise gate selection.
"""
    (out / "layer_budget_report.md").write_text(report, encoding="utf-8")
    print(results[columns].to_string(index=False), flush=True)
    print(f"saved to {out}", flush=True)


if __name__ == "__main__":
    main()
