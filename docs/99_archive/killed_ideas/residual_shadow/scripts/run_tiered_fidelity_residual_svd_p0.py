#!/usr/bin/env python3
"""Tiered-Fidelity Shadow Experts + Residual-EP P0: neither candidate had ever
had a single line of code written (they require a distilled/fine-tuned local
surrogate, which was judged too expensive for this project's budget). This
script gets a real, GPU-measured quality/compression frontier for BOTH
candidates WITHOUT any training, by using truncated SVD (a training-free,
zero-gradient low-rank approximation) of the model's ALREADY-TRAINED expert
weight matrices as the surrogate -- the cheapest possible instantiation of
"local low-rank shadow" (Tiered-Fidelity) and "shared-base + residual"
(Residual-EP), and the natural first gate before spending any time on actual
distillation/fine-tuning.

Modes
-----
  shadow   (Tiered-Fidelity): each expert's 3 weight matrices
           (gate_proj/up_proj/down_proj) are independently replaced by their
           rank-r truncated-SVD reconstruction. This is the pure "run a
           compressed local copy of the SAME expert" case -- no remote
           communication at all, at the cost of approximation error.
  residual (Residual-EP): a single shared base B (the elementwise MEAN weight
           matrix across all experts of the same layer/matrix) is subtracted
           from every expert; the resulting per-expert residual
           Delta_e = W_e - B is then further compressed via rank-r truncated
           SVD. Reconstruction = B + lowrank(Delta_e). This directly tests the
           paper's premise that "the residual has smaller effective rank /
           dynamic range than the raw expert weight", which is the necessary
           condition for Residual-EP's wire-format savings claim to be worth
           pursuing at all.

Both modes sweep the same set of rank fractions and are scored with the exact
same downstream-KL methodology already used by PLTB/layer_budget (paired
bootstrap over documents, calibration/test split), so results sit on the same
evidence scale as every other candidate in the Approach Registry.

Evidence tag: [Observed], real forward passes, zero new training.
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
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from modeling import DEFAULT_MODEL, load_model, load_tokenizer
from prompts import get_prompts


def get_moe_layers(model):
    layers = model.model.layers
    out = []
    for layer in layers:
        if hasattr(layer, "mlp") and hasattr(layer.mlp, "experts"):
            out.append(layer.mlp)
        elif hasattr(layer, "block_sparse_moe"):
            out.append(layer.block_sparse_moe)
        else:
            raise TypeError(f"unsupported layer structure: {type(layer)}")
    return out


def truncated_svd_reconstruct(w: torch.Tensor, rank: int) -> torch.Tensor:
    """Rank-`rank` reconstruction of a 2D weight matrix via truncated SVD."""
    orig_dtype = w.dtype
    w32 = w.detach().float()
    u, s, vh = torch.linalg.svd(w32, full_matrices=False)
    rank = max(1, min(rank, s.shape[0]))
    recon = (u[:, :rank] * s[:rank]) @ vh[:rank, :]
    return recon.to(orig_dtype)


def apply_shadow(moe, rank_frac: float, target_layer_ids: set[int] | None, layer_id: int) -> dict | None:
    if target_layer_ids is not None and layer_id not in target_layer_ids:
        return None
    saved = {}
    for e_idx, expert in enumerate(moe.experts):
        for attr in ("gate_proj", "up_proj", "down_proj"):
            lin = getattr(expert, attr)
            w = lin.weight.data
            rank = max(1, int(round(rank_frac * min(w.shape))))
            saved[(e_idx, attr)] = w.clone()
            lin.weight.data = truncated_svd_reconstruct(w, rank)
    return saved


def apply_residual(moe, rank_frac: float, target_layer_ids: set[int] | None, layer_id: int) -> dict | None:
    if target_layer_ids is not None and layer_id not in target_layer_ids:
        return None
    saved = {}
    for attr in ("gate_proj", "up_proj", "down_proj"):
        stacked = torch.stack([getattr(e, attr).weight.data.float() for e in moe.experts], dim=0)
        base = stacked.mean(dim=0)  # shared base B, one per (layer, matrix)
        for e_idx, expert in enumerate(moe.experts):
            lin = getattr(expert, attr)
            w = lin.weight.data
            saved[(e_idx, attr)] = w.clone()
            delta = w.float() - base
            rank = max(1, int(round(rank_frac * min(delta.shape))))
            delta_lowrank = truncated_svd_reconstruct(delta, rank)
            lin.weight.data = (base + delta_lowrank).to(w.dtype)
    return saved


def restore(moe, saved: dict) -> None:
    for (e_idx, attr), w in saved.items():
        getattr(moe.experts[e_idx], attr).weight.data = w


def residual_effective_rank_report(moe, layer_id: int) -> list[dict]:
    """Diagnostic: how much smaller is the residual's effective (energy-90%)
    rank compared to the raw expert weight's effective rank? This is the
    necessary-condition check for Residual-EP -- if residual rank is NOT
    smaller than raw rank, the wire-format savings claim has no basis."""
    rows = []
    for attr in ("gate_proj", "up_proj", "down_proj"):
        stacked = torch.stack([getattr(e, attr).weight.data.float() for e in moe.experts], dim=0)
        base = stacked.mean(dim=0)
        for e_idx, expert in enumerate(moe.experts):
            w = getattr(expert, attr).weight.data.float()
            delta = w - base

            def eff_rank(m):
                s = torch.linalg.svdvals(m)
                energy = (s ** 2).cumsum(0) / (s ** 2).sum().clamp_min(1e-12)
                return int((energy < 0.90).sum().item()) + 1, int(s.shape[0])

            raw_rank, full_dim = eff_rank(w)
            delta_rank, _ = eff_rank(delta)
            rows.append({
                "layer": layer_id, "matrix": attr, "expert": e_idx,
                "full_dim": full_dim, "raw_effective_rank_90pct": raw_rank,
                "residual_effective_rank_90pct": delta_rank,
                "rank_reduction_frac": 1.0 - delta_rank / max(raw_rank, 1),
            })
    return rows


def compute_kl(full_logits: torch.Tensor, approx_logits: torch.Tensor) -> float:
    full = full_logits[:, :-1, :].contiguous().float()
    approx = approx_logits[:, :-1, :].contiguous().float()
    p = F.softmax(full, dim=-1)
    log_q = F.log_softmax(approx, dim=-1)
    log_p = F.log_softmax(full, dim=-1)
    kl = (p * (log_p - log_q)).sum(dim=-1)
    return float(kl.mean().item())


def paired_bootstrap_ci(diffs: np.ndarray, n_boot: int, seed: int, alpha: float = 0.05):
    rng = np.random.default_rng(seed)
    n = len(diffs)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot[b] = diffs[idx].mean()
    return float(np.quantile(boot, alpha / 2)), float(np.quantile(boot, 1 - alpha / 2)), float(diffs.mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--test-samples", type=int, default=32)
    ap.add_argument("--test-offset", type=int, default=20)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--dataset", default="wikitext2_docs")
    ap.add_argument("--dataset-split", default="validation")
    ap.add_argument("--target-layers", type=int, nargs="+", default=None,
                     help="if set, only apply the decomposition to these layers (default: all)")
    ap.add_argument("--rank-fracs", type=float, nargs="+", default=[0.05, 0.1, 0.25, 0.5])
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--n-bootstrap", type=int, default=1000)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("loading model...")
    model, _ = load_model(args.model, args.dtype, local_files_only=args.offline)
    tokenizer = load_tokenizer(args.model, local_files_only=args.offline)
    moe_layers = get_moe_layers(model)
    target_set = set(args.target_layers) if args.target_layers else None

    texts = get_prompts(args.dataset, args.test_samples, offset=args.test_offset, split=args.dataset_split)
    inputs_list = []
    for text in texts:
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=args.seq_len)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        inputs_list.append(enc)

    print(f"computing {len(inputs_list)} full-precision reference logits...")
    full_logits = []
    with torch.no_grad():
        for enc in inputs_list:
            full_logits.append(model(**enc).logits.detach().cpu())

    print("residual effective-rank diagnostic (necessary-condition check for Residual-EP)...")
    rank_rows = []
    for layer_id, moe in enumerate(moe_layers):
        if target_set is not None and layer_id not in target_set:
            continue
        rank_rows.extend(residual_effective_rank_report(moe, layer_id))
    rank_df = pd.DataFrame(rank_rows)
    rank_df.to_csv(out / "residual_effective_rank_diagnostic.csv", index=False)

    modes = {"shadow": apply_shadow, "residual": apply_residual}
    all_rows = []
    for mode_name, apply_fn in modes.items():
        for rank_frac in args.rank_fracs:
            print(f"[{mode_name}] rank_frac={rank_frac} ...")
            saved_by_layer = []
            for layer_id, moe in enumerate(moe_layers):
                saved = apply_fn(moe, rank_frac, target_set, layer_id)
                saved_by_layer.append((moe, saved))
            with torch.no_grad():
                for doc_idx, enc in enumerate(inputs_list):
                    approx_logits = model(**enc).logits.detach().cpu()
                    kl = compute_kl(full_logits[doc_idx], approx_logits)
                    all_rows.append({
                        "mode": mode_name, "rank_frac": rank_frac,
                        "sample_id": args.test_offset + doc_idx, "mean_token_kl": kl,
                    })
            for moe, saved in saved_by_layer:
                if saved is not None:
                    restore(moe, saved)

    df = pd.DataFrame(all_rows)
    df.to_csv(out / "per_document_kl.csv", index=False)

    summary_rows = []
    for mode_name in modes:
        for rank_frac in args.rank_fracs:
            sub = df[(df["mode"] == mode_name) & (df["rank_frac"] == rank_frac)]
            vals = sub["mean_token_kl"].to_numpy()
            lo, hi, mean_kl = paired_bootstrap_ci(vals, args.n_bootstrap, 20260720)
            summary_rows.append({
                "mode": mode_name, "rank_frac": rank_frac,
                "mean_kl": mean_kl, "ci_low": lo, "ci_high": hi, "n_docs": len(vals),
            })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out / "summary.csv", index=False)

    lines = ["# Tiered-Fidelity Shadow Experts + Residual-EP P0 (SVD surrogate, no training)", ""]
    lines.append("## KL vs rank fraction (lower is better; `full`=0 by construction)")
    lines.append("")
    cols = ["mode", "rank_frac", "mean_kl", "ci_low", "ci_high", "n_docs"]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    for _, row in summary.sort_values(["mode", "rank_frac"]).iterrows():
        vals = [f"{row[c]:.6f}" if isinstance(row[c], float) else str(row[c]) for c in cols]
        lines.append("| " + " | ".join(vals) + " |")

    lines.append("")
    lines.append("## Residual effective-rank diagnostic (mean across experts/matrices/layers)")
    lines.append("")
    agg = rank_df.groupby("matrix")[["raw_effective_rank_90pct", "residual_effective_rank_90pct",
                                      "rank_reduction_frac", "full_dim"]].mean().reset_index()
    agg_cols = agg.columns.tolist()
    lines.append("| " + " | ".join(agg_cols) + " |")
    lines.append("|" + "|".join(["---"] * len(agg_cols)) + "|")
    for _, row in agg.iterrows():
        vals = [f"{row[c]:.4f}" if isinstance(row[c], float) else str(row[c]) for c in agg_cols]
        lines.append("| " + " | ".join(vals) + " |")

    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nsaved to {out}")


if __name__ == "__main__":
    main()
