#!/usr/bin/env python3
"""Synergy check: does document-level MoE routing predictability (the signal
already computed for expert prefetch, P0/P0-C) correlate with document-level
quality risk (the signal Per-request Quality Isolation needs)?

If yes: the two candidates can share ONE per-document diagnostic instead of
computing two independent signals -- a genuine systems-level unification.
If no: they are orthogonal and must stay separate mechanisms.

Uses the EXACT same calibration_routes.csv/test_routes.csv pair as
Routing-Predictability P0 (so the transition table is unchanged), and a fresh
KL run on the identical wikitext2_docs:test offset=16 n=45 document set (so
sample_id directly indexes the same underlying documents).
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

import numpy as np
import pandas as pd
from scipy import stats


def build_transition_table(calib_top1: pd.DataFrame, num_layers: int) -> dict[int, dict[int, int]]:
    """Most-likely next-layer top-1 expert given current top-1 expert."""
    out = {}
    for layer in range(num_layers - 1):
        cur = calib_top1[calib_top1["layer"] == layer][["sample_id", "token_position", "top1_expert"]]
        nxt = calib_top1[calib_top1["layer"] == layer + 1][["sample_id", "token_position", "top1_expert"]]
        joined = cur.merge(nxt, on=["sample_id", "token_position"], suffixes=("_cur", "_nxt"))
        table = {}
        for prev_e, grp in joined.groupby("top1_expert_cur"):
            table[int(prev_e)] = int(grp["top1_expert_nxt"].value_counts().idxmax())
        out[layer] = table
    return out


def per_document_predictability(test_top1: pd.DataFrame, num_layers: int,
                                 trans_table: dict[int, dict[int, int]]) -> pd.DataFrame:
    rows = []
    for sample_id, doc in test_top1.groupby("sample_id"):
        by_layer = {layer: grp.set_index("token_position") for layer, grp in doc.groupby("layer")}
        hits, total = 0, 0
        entropies = []
        for layer in range(num_layers - 1):
            cur, nxt = by_layer.get(layer), by_layer.get(layer + 1)
            if cur is None or nxt is None:
                continue
            common = cur.index.intersection(nxt.index)
            if len(common) == 0:
                continue
            table = trans_table.get(layer, {})
            pred = cur.loc[common, "top1_expert"].map(table)
            true = nxt.loc[common, "top1_expert"]
            hits += int((pred == true).sum())
            total += len(common)
            counts = cur.loc[common, "top1_expert"].value_counts(normalize=True)
            entropies.append(float(-(counts * np.log(counts + 1e-12)).sum()))
        rows.append({
            "sample_id": sample_id,
            "top1_hit_rate": hits / max(total, 1),
            "mean_routing_entropy": float(np.mean(entropies)) if entropies else np.nan,
            "n_tokens": doc["token_position"].nunique(),
        })
    return pd.DataFrame(rows)


def main():
    calib = pd.read_csv("outputs/paper_validation/olmoe_r_layout_article_stage1_formal_2026-07-13/calibration_routes.csv")
    test = pd.read_csv("outputs/paper_validation/olmoe_r_layout_article_stage1_formal_2026-07-13/test_routes.csv")
    calib_top1 = calib[calib["rank"] == 1][["sample_id", "token_position", "layer", "expert_id"]].rename(
        columns={"expert_id": "top1_expert"})
    test_top1 = test[test["rank"] == 1][["sample_id", "token_position", "layer", "expert_id"]].rename(
        columns={"expert_id": "top1_expert"})
    num_layers = int(pd.concat([calib_top1["layer"], test_top1["layer"]]).max()) + 1

    trans_table = build_transition_table(calib_top1, num_layers)
    pred_df = per_document_predictability(test_top1, num_layers, trans_table)

    kl = pd.read_csv("outputs/quality_routing_synergy_test45_2026-07-20/sample_metrics.csv")
    kl_fixed = kl[kl["strategy"] == "fixed_tail4"][["sample_id", "mean_token_kl", "mean_nll", "token_count"]]

    merged = pred_df.merge(kl_fixed, on="sample_id", how="inner")
    print(f"merged n_docs={len(merged)}")

    rho_hit, p_hit = stats.spearmanr(merged["top1_hit_rate"], merged["mean_token_kl"])
    rho_ent, p_ent = stats.spearmanr(merged["mean_routing_entropy"], merged["mean_token_kl"])
    rho_len, p_len = stats.spearmanr(merged["token_count"], merged["mean_token_kl"])
    rho_hit_ent, p_hit_ent = stats.spearmanr(merged["top1_hit_rate"], merged["mean_routing_entropy"])

    merged.to_csv("outputs/quality_routing_synergy_test45_2026-07-20/merged_predictability_vs_kl.csv", index=False)

    report = [
        "# Synergy check: routing predictability vs quality-degradation risk (same 45 documents)",
        "",
        f"n_docs = {len(merged)}",
        "",
        f"Spearman(top1_hit_rate, mean_token_kl under fixed_tail4) = {rho_hit:.4f} (p={p_hit:.4g})",
        f"Spearman(mean_routing_entropy, mean_token_kl) = {rho_ent:.4f} (p={p_ent:.4g})",
        f"Spearman(token_count, mean_token_kl) = {rho_len:.4f} (p={p_len:.4g})  [length confound check]",
        f"Spearman(top1_hit_rate, mean_routing_entropy) = {rho_hit_ent:.4f} (p={p_hit_ent:.4g})  [sanity: hit_rate and entropy should be strongly anti-correlated]",
    ]
    print("\n".join(report))
    with open("outputs/quality_routing_synergy_test45_2026-07-20/synergy_report.md", "w") as f:
        f.write("\n".join(report))


if __name__ == "__main__":
    main()
