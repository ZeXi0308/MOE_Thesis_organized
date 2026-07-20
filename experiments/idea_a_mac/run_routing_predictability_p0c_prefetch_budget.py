#!/usr/bin/env python3
"""Routing Predictability P0-C: prefetch-budget hit-rate curve.

P0 established the signal exists (+13.9~+19.0pp top-1 accuracy over freq
baseline). P0-B established it survives up to 8 layers of lookahead. But a
real system does not need to guess the SINGLE most likely next expert -- an
expert-weight prefetcher or a speculative-decode draft step can afford to
speculate on a SET of N candidate experts per layer (bounded by HBM
prefetch bandwidth / draft budget). This script asks the system-relevant
question directly: if the prefetcher issues N candidate experts per token
per layer (chosen as the N experts with the highest calibration-learned
transition probability from the current top-1 expert), what fraction of
tokens have their TRUE next-layer top-1 expert already among the N
prefetched candidates ("hit rate")?

Compared against a size-N RANDOM/frequency-based budget baseline: the top-N
most frequent experts at layer L+1 (no per-token signal at all -- the
"floor" a real prefetcher must beat, matching the practical bar/evidence
discipline used throughout this project).

Evidence tag: [Observed], same calibration/test split and route traces as
P0/P0-B, zero new data collection.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def load_top1_by_layer(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    top1 = df[df["rank"] == 1][["sample_id", "token_position", "layer", "expert_id"]]
    return top1.rename(columns={"expert_id": "top1_expert"})


def build_topn_freq(calib: pd.DataFrame, num_layers: int, n: int) -> dict[int, list[int]]:
    out = {}
    for layer in range(num_layers):
        counts = calib[calib["layer"] == layer]["top1_expert"].value_counts()
        out[layer] = counts.index[:n].tolist()
    return out


def build_topn_transition(calib: pd.DataFrame, num_layers: int, n: int) -> dict[int, dict[int, list[int]]]:
    out: dict[int, dict[int, list[int]]] = {}
    for layer in range(num_layers - 1):
        cur = calib[calib["layer"] == layer][["sample_id", "token_position", "top1_expert"]]
        nxt = calib[calib["layer"] == layer + 1][["sample_id", "token_position", "top1_expert"]]
        joined = cur.merge(nxt, on=["sample_id", "token_position"], suffixes=("_cur", "_nxt"))
        table: dict[int, list[int]] = {}
        for prev_expert, grp in joined.groupby("top1_expert_cur"):
            table[int(prev_expert)] = grp["top1_expert_nxt"].value_counts().index[:n].tolist()
        out[layer] = table
    return out


def per_document_hit_rate(test: pd.DataFrame, num_layers: int, n: int,
                           freq_topn: dict[int, list[int]],
                           trans_topn: dict[int, dict[int, list[int]]]) -> pd.DataFrame:
    rows = []
    for sample_id, doc in test.groupby("sample_id"):
        by_layer = {layer: grp.set_index("token_position") for layer, grp in doc.groupby("layer")}
        for layer in range(num_layers - 1):
            cur = by_layer.get(layer)
            nxt = by_layer.get(layer + 1)
            if cur is None or nxt is None:
                continue
            common = cur.index.intersection(nxt.index)
            if len(common) == 0:
                continue
            true_next = nxt.loc[common, "top1_expert"].to_numpy()
            cur_top1 = cur.loc[common, "top1_expert"].to_numpy()
            table = trans_topn.get(layer, {})
            default_set = set(freq_topn.get(layer + 1, []))
            hits_trans = np.array([
                true_next[i] in set(table.get(int(cur_top1[i]), freq_topn.get(layer + 1, [])))
                for i in range(len(common))
            ])
            hits_freq = np.array([true_next[i] in default_set for i in range(len(common))])
            rows.append({"sample_id": sample_id, "layer": layer, "n": n, "predictor": "transition_topn",
                         "hit_rate": float(hits_trans.mean()), "n_tokens": len(common)})
            rows.append({"sample_id": sample_id, "layer": layer, "n": n, "predictor": "freq_topn",
                         "hit_rate": float(hits_freq.mean()), "n_tokens": len(common)})
    return pd.DataFrame(rows)


def paired_bootstrap_ci(diffs: np.ndarray, n_boot: int, seed: int, alpha: float = 0.05):
    rng = np.random.default_rng(seed)
    n = len(diffs)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot[b] = diffs[idx].mean()
    return float(np.quantile(boot, alpha / 2)), float(np.quantile(boot, 1 - alpha / 2)), float(diffs.mean())


def run_model(model_key: str, calib_csv: Path, test_csv: Path, ns: list[int], num_experts_hint: int,
              n_boot: int, seed: int) -> pd.DataFrame:
    calib = load_top1_by_layer(calib_csv)
    test = load_top1_by_layer(test_csv)
    num_layers = int(pd.concat([calib["layer"], test["layer"]]).max()) + 1

    rows = []
    for n in ns:
        freq_topn = build_topn_freq(calib, num_layers, n)
        trans_topn = build_topn_transition(calib, num_layers, n)
        hit_df = per_document_hit_rate(test, num_layers, n, freq_topn, trans_topn)
        p_doc = hit_df[hit_df["predictor"] == "transition_topn"].groupby("sample_id")["hit_rate"].mean()
        b_doc = hit_df[hit_df["predictor"] == "freq_topn"].groupby("sample_id")["hit_rate"].mean()
        common = p_doc.index.intersection(b_doc.index)
        diffs = (p_doc.loc[common] - b_doc.loc[common]).to_numpy()
        lo, hi, mean_diff = paired_bootstrap_ci(diffs, n_boot, seed)
        rows.append({
            "model": model_key, "budget_n": n, "budget_frac_of_experts": n / num_experts_hint,
            "transition_hit_rate": p_doc.loc[common].mean(), "freq_hit_rate": b_doc.loc[common].mean(),
            "mean_hit_rate_diff_pp": mean_diff * 100, "ci_low_pp": lo * 100, "ci_high_pp": hi * 100,
            "n_docs": len(common),
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--olmoe-calib", required=True)
    ap.add_argument("--olmoe-test", required=True)
    ap.add_argument("--olmoe-num-experts", type=int, default=64)
    ap.add_argument("--llmjp-calib", required=True)
    ap.add_argument("--llmjp-test", required=True)
    ap.add_argument("--llmjp-num-experts", type=int, default=32)
    ap.add_argument("--budgets", type=int, nargs="+", default=[1, 2, 3, 4, 6, 8])
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260720)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df_olmoe = run_model("olmoe", Path(args.olmoe_calib), Path(args.olmoe_test), args.budgets,
                          args.olmoe_num_experts, args.n_boot, args.seed)
    df_llmjp = run_model("llmjp", Path(args.llmjp_calib), Path(args.llmjp_test), args.budgets,
                          args.llmjp_num_experts, args.n_boot, args.seed + 1)
    df = pd.concat([df_olmoe, df_llmjp], ignore_index=True)
    df.to_csv(out / "prefetch_budget_hit_rate.csv", index=False)

    lines = ["# Routing Predictability P0-C: Prefetch-Budget Hit-Rate Curve", ""]
    lines.append("transition_hit_rate = P(true next-layer top-1 expert in top-N predicted candidates)")
    lines.append("freq_hit_rate = P(true next-layer top-1 expert in top-N most-frequent experts, no per-token signal)")
    lines.append("")
    cols = ["model", "budget_n", "budget_frac_of_experts", "transition_hit_rate", "freq_hit_rate",
            "mean_hit_rate_diff_pp", "ci_low_pp", "ci_high_pp", "n_docs"]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    for _, row in df.sort_values(["model", "budget_n"]).iterrows():
        vals = [f"{row[c]:.4f}" if isinstance(row[c], float) else str(row[c]) for c in cols]
        lines.append("| " + " | ".join(vals) + " |")
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nsaved to {out}")


if __name__ == "__main__":
    main()
