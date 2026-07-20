#!/usr/bin/env python3
"""Routing Predictability P0-B: how far ahead does the cross-layer signal
survive? (lookahead-distance decay curve)

P0 (run_routing_predictability_p0.py) established that a token's top-1
expert at layer L predicts its top-1 expert at layer L+1 far better than
the global frequency baseline (+13.9pp to +19.0pp across OLMoE/LLM-jp,
Holm-corrected, paired-bootstrap CI excludes 0). That is a necessary but not
sufficient condition for a routing-aware prefetch/draft mechanism: a
prefetch decision made at layer L needs enough LEAD TIME to actually issue
an HBM fetch or draft-model step before layer L+k is reached, so the
practically relevant question is how the signal decays as k grows
(predicting L+1, L+2, L+3, ... from information at L).

This script trains, for each lookahead distance k in {1,2,3,4}, a transition
table P(expert_{L+k} | expert_L) on the calibration split and evaluates
accuracy on the sealed test split, using the exact same paired-bootstrap +
Holm-correction methodology as P0.

Evidence tag: [Observed], real captured routes, same calibration/test split
as P0 (no new data collection needed).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm


def load_top1_by_layer(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    top1 = df[df["rank"] == 1][["sample_id", "token_position", "layer", "expert_id"]]
    return top1.rename(columns={"expert_id": "top1_expert"})


def build_freq_table(calib: pd.DataFrame, num_layers: int) -> dict[int, int]:
    out = {}
    for layer in range(num_layers):
        counts = calib[calib["layer"] == layer]["top1_expert"].value_counts()
        out[layer] = int(counts.idxmax()) if len(counts) else -1
    return out


def build_transition_k(calib: pd.DataFrame, num_layers: int, k: int) -> dict[int, dict[int, int]]:
    out: dict[int, dict[int, int]] = {}
    for layer in range(num_layers - k):
        cur = calib[calib["layer"] == layer][["sample_id", "token_position", "top1_expert"]]
        nxt = calib[calib["layer"] == layer + k][["sample_id", "token_position", "top1_expert"]]
        joined = cur.merge(nxt, on=["sample_id", "token_position"], suffixes=("_cur", "_nxt"))
        table: dict[int, int] = {}
        for prev_expert, grp in joined.groupby("top1_expert_cur"):
            table[int(prev_expert)] = int(grp["top1_expert_nxt"].value_counts().idxmax())
        out[layer] = table
    return out


def per_document_accuracy_k(test: pd.DataFrame, num_layers: int, k: int,
                             freq_table: dict[int, int], trans_k: dict[int, dict[int, int]]) -> pd.DataFrame:
    rows = []
    for sample_id, doc in test.groupby("sample_id"):
        by_layer = {layer: grp.set_index("token_position") for layer, grp in doc.groupby("layer")}
        for layer in range(num_layers - k):
            cur = by_layer.get(layer)
            nxt = by_layer.get(layer + k)
            if cur is None or nxt is None:
                continue
            common = cur.index.intersection(nxt.index)
            if len(common) == 0:
                continue
            true_next = nxt.loc[common, "top1_expert"].to_numpy()
            cur_top1 = cur.loc[common, "top1_expert"].to_numpy()
            table = trans_k.get(layer, {})
            pred_trans = np.array([table.get(int(e), freq_table.get(layer + k, -1)) for e in cur_top1])
            pred_freq = np.full(len(common), freq_table.get(layer + k, -1))
            rows.append({"sample_id": sample_id, "layer": layer, "k": k, "predictor": "top1_transition",
                         "accuracy": float(np.mean(pred_trans == true_next)), "n_tokens": len(common)})
            rows.append({"sample_id": sample_id, "layer": layer, "k": k, "predictor": "freq_baseline",
                         "accuracy": float(np.mean(pred_freq == true_next)), "n_tokens": len(common)})
    return pd.DataFrame(rows)


def paired_bootstrap_ci(diffs: np.ndarray, n_boot: int, seed: int, alpha: float = 0.05):
    rng = np.random.default_rng(seed)
    n = len(diffs)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot[b] = diffs[idx].mean()
    return float(np.quantile(boot, alpha / 2)), float(np.quantile(boot, 1 - alpha / 2)), float(diffs.mean())


def bootstrap_p(diffs: np.ndarray, n_boot: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    n = len(diffs)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot[b] = diffs[idx].mean()
    se = boot.std(ddof=1)
    if se == 0:
        return 0.0 if diffs.mean() != 0 else 1.0
    return float(2 * (1 - norm.cdf(abs(diffs.mean() / se))))


def holm_adjust(pvals: list[float]) -> list[float]:
    order = np.argsort(pvals)
    m = len(pvals)
    adj = [0.0] * m
    prev = 0.0
    for rank, idx in enumerate(order):
        v = min(1.0, pvals[idx] * (m - rank))
        v = max(v, prev)
        adj[idx] = v
        prev = v
    return adj


def run_model(model_key: str, calib_csv: Path, test_csv: Path, ks: list[int], n_boot: int, seed: int) -> pd.DataFrame:
    calib = load_top1_by_layer(calib_csv)
    test = load_top1_by_layer(test_csv)
    num_layers = int(pd.concat([calib["layer"], test["layer"]]).max()) + 1
    freq_table = build_freq_table(calib, num_layers)

    all_rows = []
    p_values = []
    pending = []
    for k in ks:
        trans_k = build_transition_k(calib, num_layers, k)
        acc_df = per_document_accuracy_k(test, num_layers, k, freq_table, trans_k)
        p_doc = acc_df[acc_df["predictor"] == "top1_transition"].groupby("sample_id")["accuracy"].mean()
        b_doc = acc_df[acc_df["predictor"] == "freq_baseline"].groupby("sample_id")["accuracy"].mean()
        common = p_doc.index.intersection(b_doc.index)
        diffs = (p_doc.loc[common] - b_doc.loc[common]).to_numpy()
        lo, hi, mean_diff = paired_bootstrap_ci(diffs, n_boot, seed)
        pval = bootstrap_p(diffs, n_boot, seed + 1)
        p_values.append(pval)
        pending.append({
            "model": model_key, "k": k,
            "mean_accuracy_diff_pp": mean_diff * 100, "ci_low_pp": lo * 100, "ci_high_pp": hi * 100,
            "predictor_mean_accuracy": p_doc.loc[common].mean(), "baseline_mean_accuracy": b_doc.loc[common].mean(),
            "n_docs": len(common), "p_value_raw": pval,
        })
    adj = holm_adjust(p_values)
    for row, p_adj in zip(pending, adj):
        row["p_value_holm"] = p_adj
        row["passes_5pp_gate"] = bool(row["mean_accuracy_diff_pp"] >= 5.0 and row["ci_low_pp"] > 0 and p_adj < 0.05)
        all_rows.append(row)
    return pd.DataFrame(all_rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--olmoe-calib", required=True)
    ap.add_argument("--olmoe-test", required=True)
    ap.add_argument("--llmjp-calib", required=True)
    ap.add_argument("--llmjp-test", required=True)
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 2, 3, 4, 6, 8])
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260720)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df_olmoe = run_model("olmoe", Path(args.olmoe_calib), Path(args.olmoe_test), args.ks, args.n_boot, args.seed)
    df_llmjp = run_model("llmjp", Path(args.llmjp_calib), Path(args.llmjp_test), args.ks, args.n_boot, args.seed + 1)
    df = pd.concat([df_olmoe, df_llmjp], ignore_index=True)
    df.to_csv(out / "lookahead_decay.csv", index=False)

    lines = ["# Routing Predictability P0-B: Lookahead-Distance Decay Curve", ""]
    lines.append("How much does top1_transition's advantage over freq_baseline decay as lookahead k increases?")
    lines.append("")
    cols = ["model", "k", "mean_accuracy_diff_pp", "ci_low_pp", "ci_high_pp",
            "predictor_mean_accuracy", "baseline_mean_accuracy", "n_docs", "p_value_holm", "passes_5pp_gate"]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    for _, row in df.sort_values(["model", "k"]).iterrows():
        vals = [f"{row[c]:.4f}" if isinstance(row[c], float) else str(row[c]) for c in cols]
        lines.append("| " + " | ".join(vals) + " |")
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nsaved to {out}")


if __name__ == "__main__":
    main()
