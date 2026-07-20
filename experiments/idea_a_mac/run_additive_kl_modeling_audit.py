#!/usr/bin/env python3
"""Additive-KL modeling audit: WHY does single-layer sensitivity fail to
predict multi-layer end-to-end KL, and is the failure fixable with a
lightweight correction term?

Background
----------
`MoE_唯一核心创新_严格研究报告_2026-07-14.md` KILLED "additive-KL MILP" on
[Observed] grounds: "单层sensitivity不可加和预测多层端到端KL". That verdict
is correct but was never decomposed into its two possible causes, which
matters a lot for whether a *lighter* additive-with-correction model could
still be useful (avoiding the expensive direct end-to-end sweep the MILP
approach needed):

  (1) Numerical-error non-additivity: even with routing FROZEN to the full
      model's choices at every layer (no routing drift possible), does
      perturbing multiple layers' precision simultaneously produce more/less
      KL than the sum of each layer's isolated (locked-routing) KL? If this
      fails, the quantization errors interact non-linearly through the
      residual stream even without any routing change -- a much more
      fundamental problem, not fixable by a routing-drift correction.

  (2) Routing-drift non-additivity: with routing left FREE (the real
      deployment setting), does perturbing multiple layers cause the router
      to change its expert selections in a way whose KL contribution does
      not add across layers either?

This script measures, for OLMoE-1B-7B (MXFP4 tail precision, same setup as
the layer_budget/PLTB experiments so results are directly comparable):
  - single-layer isolated KL_locked and KL_free for each of several
    individually-perturbed layers (calibration split, additive prediction
    inputs)
  - multi-layer combined KL_locked and KL_free when ALL of those layers are
    perturbed simultaneously (test split, the ground truth the additive
    model needs to predict)
  - three predictors of the true multi-layer KL_free:
      A. naive_additive       = sum(single-layer KL_free)          [the MILP's failed assumption]
      B. locked_additive      = sum(single-layer KL_locked)         [pure numerical additive prediction]
      C. locked_additive_plus_global_drift = locked_additive * (1 + mean_drift_fraction)
                                              [locked-additive corrected by ONE global scalar
                                               learned from the single-layer drift fractions --
                                               the cheapest possible fix, if it works at all]

Evidence tag: [Observed], real OLMoE-1B-7B forward passes, MXFP4 tail
precision matching the PLTB/layer_budget evidence base already in this
project (calibration/test split disjoint, same convention as elsewhere).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from capture_moe import patch_mixtral_moe
from modeling import DEFAULT_MODEL, load_model, load_tokenizer
from prompts import get_prompts


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--tail-precision", default="mxfp4")
    p.add_argument("--base-tail", type=int, default=4, help="how many low ranks to set to tail precision on a perturbed layer")
    p.add_argument("--target-layers", type=int, nargs="+", default=[0, 3, 6, 9, 12, 15],
                    help="which layers to perturb (individually, and jointly)")
    p.add_argument("--calibration-samples", type=int, default=16)
    p.add_argument("--calibration-offset", type=int, default=0)
    p.add_argument("--test-samples", type=int, default=32)
    p.add_argument("--test-offset", type=int, default=128)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--dataset", default="wikitext2_docs")
    p.add_argument("--dataset-split", default="validation")
    p.add_argument("--num-receiver-groups", type=int, default=4)
    p.add_argument("--offline", action="store_true")
    p.add_argument("--n-bootstrap", type=int, default=1000)
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


def compute_kl(full_logits: torch.Tensor, approx_logits: torch.Tensor) -> float:
    full = full_logits[:, :-1, :].contiguous().float()
    approx = approx_logits[:, :-1, :].contiguous().float()
    p = F.softmax(full, dim=-1)
    log_q = F.log_softmax(approx, dim=-1)
    return float(F.kl_div(log_q, p, reduction="batchmean").item())


def build_lut(perturbed_layers: list[int], num_layers: int, top_k: int, base_tail: int, precision: str) -> dict:
    lut = {}
    for layer in range(num_layers):
        for group in range(1):  # placeholder, expanded below per num_receiver_groups by caller
            pass
    return lut  # unused placeholder; real LUT built inline below for clarity


def run_forward_pair(model, tokenizer, inputs, lut, num_groups, lock_routing, routing_cache):
    patch_mixtral_moe(
        model, "lut",
        num_receiver_groups=num_groups,
        lut=lut,
        lock_routing=lock_routing,
        routing_cache=routing_cache if lock_routing else None,
        cache_routing=not lock_routing and routing_cache is not None,
    )
    with torch.no_grad():
        logits = model(**inputs).logits.detach().cpu()
    return logits


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    calib_texts = get_prompts(args.dataset, args.calibration_samples, offset=args.calibration_offset, split=args.dataset_split)
    test_texts = get_prompts(args.dataset, args.test_samples, offset=args.test_offset, split=args.dataset_split)
    if set(calib_texts) & set(test_texts):
        raise RuntimeError("calibration and test prompt text overlap")

    tokenizer = load_tokenizer(args.model, local_files_only=args.offline)
    model, load_seconds = load_model(args.model, dtype_name=args.dtype, local_files_only=args.offline)
    num_layers = len(model.model.layers)
    top_k = int(getattr(model.config, "num_experts_per_tok", 8))
    print(f"model loaded in {load_seconds:.1f}s; layers={num_layers}, top_k={top_k}", flush=True)

    def layer_lut(layers: list[int]) -> dict:
        lut = {}
        for layer in range(num_layers):
            for group in range(args.num_receiver_groups):
                for rank in range(1, top_k + 1):
                    if layer in layers and rank > top_k - args.base_tail:
                        lut[(layer, group, rank)] = args.tail_precision
                    else:
                        lut[(layer, group, rank)] = "fp8"
        return lut

    def eval_split(texts: list[str], offset: int) -> pd.DataFrame:
        rows = []
        for idx, text in enumerate(texts):
            sample_id = offset + idx
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=args.seq_len)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            recorder_full = patch_mixtral_moe(model, "full", num_receiver_groups=args.num_receiver_groups, cache_routing=True)
            with torch.no_grad():
                full_logits = model(**inputs).logits.detach().cpu()
            routing_cache = recorder_full.routing_cache

            # single-layer isolated perturbations
            for layer in args.target_layers:
                lut = layer_lut([layer])
                locked_logits = run_forward_pair(model, tokenizer, inputs, lut, args.num_receiver_groups, True, routing_cache)
                free_logits = run_forward_pair(model, tokenizer, inputs, lut, args.num_receiver_groups, False, None)
                rows.append({
                    "sample_id": sample_id, "config": f"single_L{layer}", "n_layers_perturbed": 1,
                    "kl_locked": compute_kl(full_logits, locked_logits),
                    "kl_free": compute_kl(full_logits, free_logits),
                })

            # joint (all target layers perturbed simultaneously)
            lut_all = layer_lut(args.target_layers)
            locked_logits = run_forward_pair(model, tokenizer, inputs, lut_all, args.num_receiver_groups, True, routing_cache)
            free_logits = run_forward_pair(model, tokenizer, inputs, lut_all, args.num_receiver_groups, False, None)
            rows.append({
                "sample_id": sample_id, "config": "joint_all", "n_layers_perturbed": len(args.target_layers),
                "kl_locked": compute_kl(full_logits, locked_logits),
                "kl_free": compute_kl(full_logits, free_logits),
            })
            print(f"  sample {sample_id} done", flush=True)
        return pd.DataFrame(rows)

    print("evaluating calibration split (learn per-layer drift fractions)...", flush=True)
    calib_df = eval_split(calib_texts, args.calibration_offset)
    calib_df.to_csv(out / "calibration_raw.csv", index=False)

    single_calib = calib_df[calib_df["config"].str.startswith("single_")]
    mean_drift_fraction = float(
        ((single_calib["kl_free"] - single_calib["kl_locked"]) / single_calib["kl_free"].clip(lower=1e-12)).mean()
    )
    print(f"learned mean_drift_fraction from calibration singles = {mean_drift_fraction:.4f}", flush=True)

    print("evaluating sealed test split (ground truth for joint effect)...", flush=True)
    test_df = eval_split(test_texts, args.test_offset)
    test_df.to_csv(out / "test_raw.csv", index=False)

    # Per-document additive predictions vs ground truth
    single_test = test_df[test_df["config"].str.startswith("single_")]
    joint_test = test_df[test_df["config"] == "joint_all"].set_index("sample_id")

    per_doc = single_test.groupby("sample_id").agg(
        naive_additive=("kl_free", "sum"),
        locked_additive=("kl_locked", "sum"),
    )
    per_doc["locked_additive_plus_global_drift"] = per_doc["locked_additive"] * (1 + mean_drift_fraction)
    per_doc["true_joint_kl_free"] = joint_test["kl_free"]
    per_doc["true_joint_kl_locked"] = joint_test["kl_locked"]
    per_doc.to_csv(out / "per_document_predictions.csv")

    def bootstrap_ratio_ci(pred_col: str, true_col: str, n_boot: int, seed: int) -> dict:
        pred = per_doc[pred_col].to_numpy()
        true = per_doc[true_col].to_numpy()
        rng = np.random.default_rng(seed)
        n = len(pred)
        ratios = np.empty(n_boot)
        for b in range(n_boot):
            idx = rng.integers(0, n, size=n)
            ratios[b] = pred[idx].sum() / max(true[idx].sum(), 1e-12)
        point = pred.sum() / max(true.sum(), 1e-12)
        rel_err = np.abs(pred - true) / np.clip(true, 1e-12, None)
        return {
            "predictor": pred_col, "true_col": true_col,
            "point_ratio_pred_over_true": point,
            "ratio_ci_low": float(np.quantile(ratios, 0.025)),
            "ratio_ci_high": float(np.quantile(ratios, 0.975)),
            "mean_relative_error": float(rel_err.mean()),
            "median_relative_error": float(np.median(rel_err)),
        }

    results = [
        bootstrap_ratio_ci("naive_additive", "true_joint_kl_free", args.n_bootstrap, 1),
        bootstrap_ratio_ci("locked_additive", "true_joint_kl_locked", args.n_bootstrap, 2),
        bootstrap_ratio_ci("locked_additive_plus_global_drift", "true_joint_kl_free", args.n_bootstrap, 3),
    ]
    result_df = pd.DataFrame(results)
    result_df.to_csv(out / "additivity_gate_results.csv", index=False)

    meta = {
        "model": args.model, "num_layers": num_layers, "top_k": top_k,
        "target_layers": args.target_layers, "tail_precision": args.tail_precision, "base_tail": args.base_tail,
        "mean_drift_fraction_from_calibration": mean_drift_fraction,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    lines = ["# Additive-KL Modeling Audit: Why Does It Fail, And Is It Fixable?", ""]
    lines.append(f"model={args.model}, target_layers={args.target_layers}, tail={args.tail_precision}, "
                 f"mean_drift_fraction(calibration)={mean_drift_fraction:.4f}")
    lines.append("")
    lines.append("Interpretation: point_ratio_pred_over_true near 1.0 with a narrow CI around 1.0 means the "
                 "predictor is accurate; ratio far from 1.0 (or CI excluding 1.0) means the additive assumption "
                 "at that stage fails.")
    lines.append("")
    cols = list(result_df.columns)
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    for _, row in result_df.iterrows():
        vals = [f"{v:.4f}" if isinstance(v, float) else str(v) for v in row]
        lines.append("| " + " | ".join(vals) + " |")
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nsaved to {out}")


if __name__ == "__main__":
    main()
