#!/usr/bin/env python3
"""Additive-KL modeling audit (corrected 2026-07-21).

Correct incremental test (aligns with PLTB profiling convention):

  KL_fp8      = KL(full, all_fp8)
  KL_single_i = KL(full, fp8 + layer_i_tail)
  KL_joint    = KL(full, fp8 + all_target_tails)

  pred_inc = Σ_i (KL_single_i − KL_fp8)
  true_inc = KL_joint − KL_fp8
  ratio    = pred_inc / true_inc   # near 1 supports additivity

The 2026-07-20 audit summed raw KL(full, single_i) without subtracting
KL_fp8, which double-counts the shared FP8 baseline ~n_layers times.
Those columns are retained as DEPRECATED_DOUBLE_COUNT for comparison.

See docs/archive/killed_ideas/errata/判死结论勘误_2026-07-21.md.
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
    p.add_argument("--base-tail", type=int, default=4,
                    help="how many low ranks to set to tail precision on a perturbed layer")
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


def run_forward_pair(model, inputs, lut, num_groups, lock_routing, routing_cache):
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

    calib_texts = get_prompts(
        args.dataset, args.calibration_samples,
        offset=args.calibration_offset, split=args.dataset_split,
    )
    test_texts = get_prompts(
        args.dataset, args.test_samples,
        offset=args.test_offset, split=args.dataset_split,
    )
    if set(calib_texts) & set(test_texts):
        raise RuntimeError("calibration and test prompt text overlap")

    tokenizer = load_tokenizer(args.model, local_files_only=args.offline)
    model, load_seconds = load_model(args.model, dtype_name=args.dtype, local_files_only=args.offline)
    num_layers = len(model.model.layers)
    top_k = int(getattr(model.config, "num_experts_per_tok", 8))
    print(f"model loaded in {load_seconds:.1f}s; layers={num_layers}, top_k={top_k}", flush=True)

    def layer_lut(layers: list[int] | None) -> dict:
        """If layers is None or empty: all ranks FP8 (all_fp8 baseline).
        Else: target layers get tail precision on bottom base_tail ranks; others FP8.
        """
        lut = {}
        perturbed = set(layers or [])
        for layer in range(num_layers):
            for group in range(args.num_receiver_groups):
                for rank in range(1, top_k + 1):
                    if layer in perturbed and rank > top_k - args.base_tail:
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

            recorder_full = patch_mixtral_moe(
                model, "full",
                num_receiver_groups=args.num_receiver_groups,
                cache_routing=True,
            )
            with torch.no_grad():
                full_logits = model(**inputs).logits.detach().cpu()
            routing_cache = recorder_full.routing_cache

            # all-FP8 baseline (no tail MXFP4/INT4 on any layer)
            lut_fp8 = layer_lut([])
            fp8_locked = run_forward_pair(
                model, inputs, lut_fp8, args.num_receiver_groups, True, routing_cache,
            )
            fp8_free = run_forward_pair(
                model, inputs, lut_fp8, args.num_receiver_groups, False, None,
            )
            rows.append({
                "sample_id": sample_id,
                "config": "all_fp8",
                "n_layers_perturbed": 0,
                "kl_locked": compute_kl(full_logits, fp8_locked),
                "kl_free": compute_kl(full_logits, fp8_free),
            })

            for layer in args.target_layers:
                lut = layer_lut([layer])
                locked_logits = run_forward_pair(
                    model, inputs, lut, args.num_receiver_groups, True, routing_cache,
                )
                free_logits = run_forward_pair(
                    model, inputs, lut, args.num_receiver_groups, False, None,
                )
                rows.append({
                    "sample_id": sample_id,
                    "config": f"single_L{layer}",
                    "n_layers_perturbed": 1,
                    "kl_locked": compute_kl(full_logits, locked_logits),
                    "kl_free": compute_kl(full_logits, free_logits),
                })

            lut_all = layer_lut(args.target_layers)
            locked_logits = run_forward_pair(
                model, inputs, lut_all, args.num_receiver_groups, True, routing_cache,
            )
            free_logits = run_forward_pair(
                model, inputs, lut_all, args.num_receiver_groups, False, None,
            )
            rows.append({
                "sample_id": sample_id,
                "config": "joint_all",
                "n_layers_perturbed": len(args.target_layers),
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
        ((single_calib["kl_free"] - single_calib["kl_locked"])
         / single_calib["kl_free"].clip(lower=1e-12)).mean()
    )
    print(f"learned mean_drift_fraction from calibration singles = {mean_drift_fraction:.4f}", flush=True)

    print("evaluating sealed test split (ground truth for joint effect)...", flush=True)
    test_df = eval_split(test_texts, args.test_offset)
    test_df.to_csv(out / "test_raw.csv", index=False)

    single_test = test_df[test_df["config"].str.startswith("single_")]
    joint_test = test_df[test_df["config"] == "joint_all"].set_index("sample_id")
    fp8_test = test_df[test_df["config"] == "all_fp8"].set_index("sample_id")

    per_doc = single_test.groupby("sample_id").agg(
        naive_additive_DEPRECATED_DOUBLE_COUNT=("kl_free", "sum"),
        locked_additive_DEPRECATED_DOUBLE_COUNT=("kl_locked", "sum"),
    )
    per_doc["kl_fp8_locked"] = fp8_test["kl_locked"]
    per_doc["kl_fp8_free"] = fp8_test["kl_free"]
    per_doc["true_joint_kl_free"] = joint_test["kl_free"]
    per_doc["true_joint_kl_locked"] = joint_test["kl_locked"]

    # Correct incremental predictors
    n_targets = len(args.target_layers)
    per_doc["locked_incremental_sum"] = (
        per_doc["locked_additive_DEPRECATED_DOUBLE_COUNT"]
        - n_targets * per_doc["kl_fp8_locked"]
    )
    per_doc["free_incremental_sum"] = (
        per_doc["naive_additive_DEPRECATED_DOUBLE_COUNT"]
        - n_targets * per_doc["kl_fp8_free"]
    )
    per_doc["true_joint_incremental_locked"] = (
        per_doc["true_joint_kl_locked"] - per_doc["kl_fp8_locked"]
    )
    per_doc["true_joint_incremental_free"] = (
        per_doc["true_joint_kl_free"] - per_doc["kl_fp8_free"]
    )
    per_doc["locked_incremental_plus_global_drift"] = (
        per_doc["locked_incremental_sum"] * (1 + mean_drift_fraction)
    )
    # Keep old names as aliases for the deprecated columns
    per_doc["locked_additive"] = per_doc["locked_additive_DEPRECATED_DOUBLE_COUNT"]
    per_doc["naive_additive"] = per_doc["naive_additive_DEPRECATED_DOUBLE_COUNT"]
    per_doc["locked_additive_plus_global_drift"] = (
        per_doc["locked_additive"] * (1 + mean_drift_fraction)
    )
    per_doc.to_csv(out / "per_document_predictions.csv")

    def bootstrap_ratio_ci(pred_col: str, true_col: str, n_boot: int, seed: int,
                           note: str = "") -> dict:
        pred = per_doc[pred_col].to_numpy(dtype=np.float64)
        true = per_doc[true_col].to_numpy(dtype=np.float64)
        rng = np.random.default_rng(seed)
        n = len(pred)
        ratios = np.empty(n_boot)
        for b in range(n_boot):
            idx = rng.integers(0, n, size=n)
            denom = true[idx].sum()
            ratios[b] = pred[idx].sum() / denom if abs(denom) > 1e-12 else float("nan")
        denom_all = true.sum()
        point = pred.sum() / denom_all if abs(denom_all) > 1e-12 else float("nan")
        rel_err = np.abs(pred - true) / np.clip(np.abs(true), 1e-12, None)
        ci_low = float(np.nanquantile(ratios, 0.025))
        ci_high = float(np.nanquantile(ratios, 0.975))
        return {
            "predictor": pred_col,
            "true_col": true_col,
            "note": note,
            "point_ratio_pred_over_true": float(point),
            "ratio_ci_low": ci_low,
            "ratio_ci_high": ci_high,
            "ci_contains_one": bool(ci_low <= 1.0 <= ci_high),
            "mean_relative_error": float(np.nanmean(rel_err)),
            "median_relative_error": float(np.nanmedian(rel_err)),
        }

    results = [
        bootstrap_ratio_ci(
            "locked_incremental_sum", "true_joint_incremental_locked",
            args.n_bootstrap, 10, note="CORRECT_INCREMENTAL_locked",
        ),
        bootstrap_ratio_ci(
            "free_incremental_sum", "true_joint_incremental_free",
            args.n_bootstrap, 11, note="CORRECT_INCREMENTAL_free",
        ),
        bootstrap_ratio_ci(
            "locked_incremental_plus_global_drift", "true_joint_incremental_free",
            args.n_bootstrap, 12, note="CORRECT_INCREMENTAL_locked_plus_drift_vs_free",
        ),
        bootstrap_ratio_ci(
            "locked_additive_DEPRECATED_DOUBLE_COUNT", "true_joint_kl_locked",
            args.n_bootstrap, 2, note="DEPRECATED_DOUBLE_COUNT",
        ),
        bootstrap_ratio_ci(
            "naive_additive_DEPRECATED_DOUBLE_COUNT", "true_joint_kl_free",
            args.n_bootstrap, 1, note="DEPRECATED_DOUBLE_COUNT",
        ),
        bootstrap_ratio_ci(
            "locked_additive_plus_global_drift", "true_joint_kl_free",
            args.n_bootstrap, 3, note="DEPRECATED_DOUBLE_COUNT",
        ),
    ]
    result_df = pd.DataFrame(results)
    result_df.to_csv(out / "additivity_gate_results.csv", index=False)

    inc_row = result_df[result_df["note"] == "CORRECT_INCREMENTAL_locked"].iloc[0]
    if bool(inc_row["ci_contains_one"]):
        verdict = (
            "WITHDRAW_ADDITIVITY_FALSIFIED: incremental locked ratio CI contains 1.0; "
            "prior 3.77x claim was an accounting artifact. Additivity unresolved/possible."
        )
    else:
        verdict = (
            "MAINTAIN_NONADDITIVITY_WITH_NEW_RATIO: incremental locked CI excludes 1.0; "
            f"replace 3.77x with {inc_row['point_ratio_pred_over_true']:.4f} "
            f"[{inc_row['ratio_ci_low']:.4f}, {inc_row['ratio_ci_high']:.4f}]. "
            "Do not cite residual-stream 3-5x narrative from the deprecated double-count."
        )

    meta = {
        "model": args.model,
        "num_layers": num_layers,
        "top_k": top_k,
        "target_layers": args.target_layers,
        "tail_precision": args.tail_precision,
        "base_tail": args.base_tail,
        "mean_drift_fraction_from_calibration": mean_drift_fraction,
        "correction": "2026-07-21 incremental KL vs all_fp8 baseline",
        "verdict": verdict,
        "locked_incremental_point_ratio": float(inc_row["point_ratio_pred_over_true"]),
        "locked_incremental_ci": [
            float(inc_row["ratio_ci_low"]),
            float(inc_row["ratio_ci_high"]),
        ],
        "ci_contains_one": bool(inc_row["ci_contains_one"]),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (out / "decision.json").write_text(json.dumps({
        "verdict": verdict,
        "ci_contains_one": bool(inc_row["ci_contains_one"]),
        "locked_incremental_point_ratio": float(inc_row["point_ratio_pred_over_true"]),
        "locked_incremental_ci": [
            float(inc_row["ratio_ci_low"]),
            float(inc_row["ratio_ci_high"]),
        ],
    }, indent=2), encoding="utf-8")

    lines = [
        "# Additive-KL Modeling Audit (Corrected 2026-07-21)",
        "",
        f"model={args.model}, target_layers={args.target_layers}, tail={args.tail_precision}, "
        f"mean_drift_fraction(calibration)={mean_drift_fraction:.4f}",
        "",
        "## Correct incremental test",
        "",
        "`pred_inc = Σ(KL_single_i − KL_fp8)` vs `true_inc = KL_joint − KL_fp8`.",
        "Deprecated double-count columns sum raw KL(full, ·) and must not be cited as science.",
        "",
        f"**Verdict:** {verdict}",
        "",
        "| " + " | ".join(result_df.columns) + " |",
        "|" + "|".join(["---"] * len(result_df.columns)) + "|",
    ]
    for _, row in result_df.iterrows():
        vals = []
        for v in row:
            if isinstance(v, float):
                vals.append(f"{v:.4f}")
            elif isinstance(v, (bool, np.bool_)):
                vals.append(str(bool(v)))
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nsaved to {out}")


if __name__ == "__main__":
    main()
