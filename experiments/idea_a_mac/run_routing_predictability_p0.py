#!/usr/bin/env python3
"""Routing Predictability P0: is a token's expert choice at layer L+1
predictable from information available at layer L, beyond what a trivial
global-frequency baseline already captures?

Why this experiment, and why now
---------------------------------
2026-07-20 research report (research_report_moe_ep_deeper_innovation_2026-07-20.md)
identified "speculative decoding / expert prefetch informed by MoE routing
predictability" as one of the few directions that (a) has a genuine gap in
the literature (MoESD, SP-MoE apply spec-decoding to MoE but do not exploit
cross-layer routing structure to design the draft/prefetch signal itself),
and (b) can be tested at ZERO additional compute cost using route traces
already captured by this project (OLMoE-1B-7B E64K8, LLM-jp E32K16).

This script is the GATE that must pass before any draft-model or expert
prefetch mechanism is worth designing: if a token's layer-L routing carries
no more information about its layer-(L+1) routing than the plain marginal
frequency of experts at layer L+1, there is no signal to build a prefetch/
draft mechanism on, and the direction should be killed here -- exactly the
same discipline this project applied to CreditReduce/RouteFidelity/
MassCover/TokenRace-EP (check whether the "oracle"/upper-bound signal exists
before designing algorithms around it).

Predictors compared (all learned on the CALIBRATION split, evaluated on the
SEALED TEST split -- same calibration/test separation already used
throughout this project, so this is directly comparable methodology, not a
new convention):

  A. freq_baseline       : argmax of the *global* marginal frequency of
                            top-1 experts at layer L+1 (no per-token signal
                            at all -- this is the floor any real predictor
                            must beat, analogous to `gate_mass`/`calib_static`
                            in prior KILLED verdicts).
  B. top1_transition      : P(expert_{L+1} | own top-1 expert at layer L),
                            a (num_experts x num_experts) transition table
                            per layer-pair, learned from calibration.
  C. topk_transition      : same idea but keyed by the *sorted tuple* of the
                            token's full top-k expert set at layer L (richer
                            state, sparser table -> falls back to A on
                            unseen states).
  D. neighbor_same_layer  : predicts token i's top-1 expert at layer L from
                            token (i-1)'s top-1 expert at the SAME layer L
                            (tests local-window / batch-level predictability,
                            independent of the cross-layer question -- useful
                            for judging whether coarse-grained prefetch by
                            token-neighborhood could work even if per-token
                            cross-layer prediction is weak).

Pre-registered practical bar (consistent with this project's established
5-10 percentage-point practical threshold used for CreditReduce/RouteFidelity
/MassCover): a predictor is "actionable" only if its accuracy improvement
over freq_baseline is >= 5 percentage points AND the Holm-corrected 95% CI
of (predictor_acc - baseline_acc), bootstrapped over TEST DOCUMENTS (the
correct resampling unit -- tokens within a document are not independent),
excludes 0.

Evidence tag: this whole script is [Observed] -- real captured routes from
real model forward passes, not synthetic/illustrative data.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm


def load_top1_by_layer(csv_path: Path) -> pd.DataFrame:
    """Return a frame indexed by (sample_id, token_position, layer) with the
    token's top-1 (rank==1, i.e. highest gate weight) expert id and full
    top-k expert set at that layer."""
    df = pd.read_csv(csv_path)
    top1 = df[df["rank"] == 1][["sample_id", "token_position", "layer", "expert_id"]]
    top1 = top1.rename(columns={"expert_id": "top1_expert"})
    full_sets = (
        df.sort_values(["sample_id", "token_position", "layer", "rank"])
        .groupby(["sample_id", "token_position", "layer"])["expert_id"]
        .apply(lambda s: tuple(sorted(s.tolist())))
        .reset_index(name="topk_set")
    )
    merged = top1.merge(full_sets, on=["sample_id", "token_position", "layer"])
    return merged


def build_freq_table(calib: pd.DataFrame, num_layers: int) -> dict[int, int]:
    """argmax expert at each layer, from calibration marginal frequency."""
    out = {}
    for layer in range(num_layers):
        counts = calib[calib["layer"] == layer]["top1_expert"].value_counts()
        out[layer] = int(counts.idxmax()) if len(counts) else -1
    return out


def build_top1_transition(calib: pd.DataFrame, num_layers: int) -> dict[int, dict[int, int]]:
    """For each layer L, table: prev_expert (at L) -> argmax next expert (at L+1)."""
    out: dict[int, dict[int, int]] = {}
    for layer in range(num_layers - 1):
        cur = calib[calib["layer"] == layer][["sample_id", "token_position", "top1_expert"]]
        nxt = calib[calib["layer"] == layer + 1][["sample_id", "token_position", "top1_expert"]]
        joined = cur.merge(nxt, on=["sample_id", "token_position"], suffixes=("_cur", "_nxt"))
        table: dict[int, int] = {}
        for prev_expert, grp in joined.groupby("top1_expert_cur"):
            table[int(prev_expert)] = int(grp["top1_expert_nxt"].value_counts().idxmax())
        out[layer] = table
    return out


def build_topk_transition(calib: pd.DataFrame, num_layers: int) -> dict[int, dict[tuple, int]]:
    out: dict[int, dict[tuple, int]] = {}
    for layer in range(num_layers - 1):
        cur = calib[calib["layer"] == layer][["sample_id", "token_position", "topk_set"]]
        nxt = calib[calib["layer"] == layer + 1][["sample_id", "token_position", "top1_expert"]]
        joined = cur.merge(nxt, on=["sample_id", "token_position"])
        table: dict[tuple, int] = {}
        for state, grp in joined.groupby("topk_set"):
            if len(grp) < 3:  # avoid overfitting to singleton states
                continue
            table[state] = int(grp["top1_expert"].value_counts().idxmax())
        out[layer] = table
    return out


def build_neighbor_transition(calib: pd.DataFrame, num_layers: int) -> dict[int, dict[int, int]]:
    """table: token (i-1)'s top-1 expert at layer L -> argmax token i's top-1
    expert at the SAME layer L."""
    out: dict[int, dict[int, int]] = {}
    for layer in range(num_layers):
        sub = calib[calib["layer"] == layer].sort_values(["sample_id", "token_position"])
        table_counts: dict[int, Counter] = defaultdict(Counter)
        for sample_id, grp in sub.groupby("sample_id"):
            experts = grp["top1_expert"].to_numpy()
            for i in range(1, len(experts)):
                table_counts[int(experts[i - 1])][int(experts[i])] += 1
        out[layer] = {k: v.most_common(1)[0][0] for k, v in table_counts.items()}
    return out


def per_document_accuracy(
    test: pd.DataFrame,
    num_layers: int,
    freq_table: dict[int, int],
    top1_trans: dict[int, dict[int, int]],
    topk_trans: dict[int, dict[tuple, int]],
    neighbor_trans: dict[int, dict[int, int]],
) -> pd.DataFrame:
    """Return one row per (sample_id, predictor, layer_bucket) with accuracy."""
    rows = []
    early_cutoff = num_layers // 2
    for sample_id, doc in test.groupby("sample_id"):
        by_layer = {layer: grp.set_index("token_position") for layer, grp in doc.groupby("layer")}
        for layer in range(num_layers - 1):
            cur = by_layer.get(layer)
            nxt = by_layer.get(layer + 1)
            if cur is None or nxt is None:
                continue
            common_tokens = cur.index.intersection(nxt.index)
            if len(common_tokens) == 0:
                continue
            true_next = nxt.loc[common_tokens, "top1_expert"].to_numpy()
            cur_top1 = cur.loc[common_tokens, "top1_expert"].to_numpy()
            cur_topk = cur.loc[common_tokens, "topk_set"].to_numpy()

            pred_freq = np.full(len(common_tokens), freq_table.get(layer + 1, -1))
            t1_table = top1_trans.get(layer, {})
            pred_top1 = np.array([t1_table.get(int(e), freq_table.get(layer + 1, -1)) for e in cur_top1])
            tk_table = topk_trans.get(layer, {})
            pred_topk = np.array([tk_table.get(tuple(s), freq_table.get(layer + 1, -1)) for s in cur_topk])

            bucket = "early" if layer < early_cutoff else "late"
            rows.append({"sample_id": sample_id, "layer": layer, "bucket": bucket,
                         "predictor": "freq_baseline", "accuracy": float(np.mean(pred_freq == true_next)),
                         "n_tokens": len(common_tokens)})
            rows.append({"sample_id": sample_id, "layer": layer, "bucket": bucket,
                         "predictor": "top1_transition", "accuracy": float(np.mean(pred_top1 == true_next)),
                         "n_tokens": len(common_tokens)})
            rows.append({"sample_id": sample_id, "layer": layer, "bucket": bucket,
                         "predictor": "topk_transition", "accuracy": float(np.mean(pred_topk == true_next)),
                         "n_tokens": len(common_tokens)})

        # neighbor_same_layer: separate question (same-layer, not cross-layer)
        for layer in range(num_layers):
            cur = by_layer.get(layer)
            if cur is None:
                continue
            cur_sorted = cur.sort_index()
            experts = cur_sorted["top1_expert"].to_numpy()
            if len(experts) < 2:
                continue
            true_cur = experts[1:]
            prev = experts[:-1]
            n_table = neighbor_trans.get(layer, {})
            pred_neighbor = np.array([n_table.get(int(p), freq_table.get(layer, -1)) for p in prev])
            pred_freq_same_layer = np.full(len(true_cur), freq_table.get(layer, -1))
            bucket = "early" if layer < early_cutoff else "late"
            rows.append({"sample_id": sample_id, "layer": layer, "bucket": bucket,
                         "predictor": "neighbor_same_layer", "accuracy": float(np.mean(pred_neighbor == true_cur)),
                         "n_tokens": len(true_cur)})
            rows.append({"sample_id": sample_id, "layer": layer, "bucket": bucket,
                         "predictor": "freq_baseline_same_layer", "accuracy": float(np.mean(pred_freq_same_layer == true_cur)),
                         "n_tokens": len(true_cur)})
    return pd.DataFrame(rows)


def paired_bootstrap_ci(diffs: np.ndarray, n_boot: int, seed: int, alpha: float = 0.05):
    rng = np.random.default_rng(seed)
    n = len(diffs)
    boot_means = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_means[b] = diffs[idx].mean()
    lo = float(np.quantile(boot_means, alpha / 2))
    hi = float(np.quantile(boot_means, 1 - alpha / 2))
    return lo, hi, float(diffs.mean())


def holm_adjust(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    m = len(p_values)
    adjusted = [0.0] * m
    prev = 0.0
    for rank, idx in enumerate(order):
        adj = min(1.0, p_values[idx] * (m - rank))
        adj = max(adj, prev)
        adjusted[idx] = adj
        prev = adj
    return adjusted


def two_sided_p_from_bootstrap(diffs: np.ndarray, n_boot: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    n = len(diffs)
    boot_means = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_means[b] = diffs[idx].mean()
    se = boot_means.std(ddof=1)
    if se == 0:
        return 0.0 if diffs.mean() != 0 else 1.0
    z = diffs.mean() / se
    return float(2 * (1 - norm.cdf(abs(z))))


def run_model(model_key: str, calib_csv: Path, test_csv: Path, n_boot: int, seed: int) -> dict:
    calib = load_top1_by_layer(calib_csv)
    test = load_top1_by_layer(test_csv)
    num_layers = int(pd.concat([calib["layer"], test["layer"]]).max()) + 1

    freq_table = build_freq_table(calib, num_layers)
    top1_trans = build_top1_transition(calib, num_layers)
    topk_trans = build_topk_transition(calib, num_layers)
    neighbor_trans = build_neighbor_transition(calib, num_layers)

    acc_df = per_document_accuracy(test, num_layers, freq_table, top1_trans, topk_trans, neighbor_trans)

    cross_layer_pairs = [
        ("top1_transition", "freq_baseline"),
        ("topk_transition", "freq_baseline"),
    ]
    same_layer_pairs = [("neighbor_same_layer", "freq_baseline_same_layer")]

    gate_rows = []
    p_values = []
    pending = []
    for bucket in ["early", "late", "all"]:
        sub = acc_df if bucket == "all" else acc_df[acc_df["bucket"] == bucket]
        for predictor, baseline in cross_layer_pairs + same_layer_pairs:
            p_doc = sub[sub["predictor"] == predictor].groupby("sample_id")["accuracy"].mean()
            b_doc = sub[sub["predictor"] == baseline].groupby("sample_id")["accuracy"].mean()
            common = p_doc.index.intersection(b_doc.index)
            diffs = (p_doc.loc[common] - b_doc.loc[common]).to_numpy()
            lo, hi, mean_diff = paired_bootstrap_ci(diffs, n_boot, seed)
            pval = two_sided_p_from_bootstrap(diffs, n_boot, seed + 1)
            p_values.append(pval)
            pending.append({
                "model": model_key, "bucket": bucket, "predictor": predictor, "baseline": baseline,
                "mean_accuracy_diff_pp": mean_diff * 100, "ci_low_pp": lo * 100, "ci_high_pp": hi * 100,
                "predictor_mean_accuracy": p_doc.loc[common].mean(), "baseline_mean_accuracy": b_doc.loc[common].mean(),
                "n_docs": len(common), "p_value_raw": pval,
            })
    adj = holm_adjust(p_values)
    for row, p_adj in zip(pending, adj):
        row["p_value_holm"] = p_adj
        row["passes_5pp_gate"] = bool(row["mean_accuracy_diff_pp"] >= 5.0 and row["ci_low_pp"] > 0 and p_adj < 0.05)
        gate_rows.append(row)

    return {
        "model": model_key,
        "num_layers": num_layers,
        "n_calib_docs": int(calib["sample_id"].nunique()),
        "n_test_docs": int(test["sample_id"].nunique()),
        "gate_rows": gate_rows,
        "raw_accuracy_by_layer": acc_df,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--olmoe-calib", required=True)
    ap.add_argument("--olmoe-test", required=True)
    ap.add_argument("--llmjp-calib", required=True)
    ap.add_argument("--llmjp-test", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260720)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    results = {}
    results["olmoe"] = run_model("olmoe", Path(args.olmoe_calib), Path(args.olmoe_test), args.n_boot, args.seed)
    results["llmjp"] = run_model("llmjp", Path(args.llmjp_calib), Path(args.llmjp_test), args.n_boot, args.seed + 1)

    all_gate_rows = []
    for m in results.values():
        all_gate_rows.extend(m["gate_rows"])
    gate_df = pd.DataFrame(all_gate_rows)
    gate_df.to_csv(out / "gate_results.csv", index=False)

    for model_key, m in results.items():
        m["raw_accuracy_by_layer"].to_csv(out / f"{model_key}_raw_accuracy_by_layer.csv", index=False)

    meta = {k: {kk: vv for kk, vv in v.items() if kk not in ("gate_rows", "raw_accuracy_by_layer")}
            for k, v in results.items()}
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    lines = ["# Routing Predictability P0: Gate Results", ""]
    lines.append("Predictor accuracy - baseline accuracy (percentage points), paired bootstrap over TEST documents, Holm-corrected across all cells.")
    lines.append("Practical bar: mean_diff_pp >= 5.0 AND ci_low_pp > 0 AND p_value_holm < 0.05.")
    lines.append("")
    cols = ["model", "bucket", "predictor", "baseline", "mean_accuracy_diff_pp", "ci_low_pp", "ci_high_pp",
            "predictor_mean_accuracy", "baseline_mean_accuracy", "n_docs", "p_value_holm", "passes_5pp_gate"]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    for _, row in gate_df.sort_values(["model", "bucket", "predictor"]).iterrows():
        vals = []
        for c in cols:
            v = row[c]
            vals.append(f"{v:.4f}" if isinstance(v, float) else str(v))
        lines.append("| " + " | ".join(vals) + " |")
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nsaved to {out}")


if __name__ == "__main__":
    main()
