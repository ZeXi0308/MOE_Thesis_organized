"""Interventional audit for fixed-quota MoE combine criticality scores.

For every routed pair, this script starts from an all-MXFP4 local combine and
measures the exact float32 local-error change when only that pair is upgraded to
fake FP8.  It then checks whether rank, gate, contribution, qerr, and qbenefit
predict that one-pair intervention.  This is deliberately narrower than an
end-to-end policy run: it diagnoses the score mechanism without claiming that
pairwise gains add across multiple upgrades or downstream layers.
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
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from capture_moe import patch_mixtral_moe
from modeling import load_model, load_tokenizer
from prompts import get_prompts
from run_signal_comparison import data_manifest, validate_exact_full_path


SCORES = ("rank_score", "gate", "contribution", "qerr", "qbenefit")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="allenai/OLMoE-1B-7B-0924")
    parser.add_argument("--dataset", default="wikitext2_docs")
    parser.add_argument("--split", default="test")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--num-receiver-groups", type=int, default=8)
    parser.add_argument("--tile-pairs", type=int, default=64)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def _num_experts(model) -> int:
    for layer in model.model.layers:
        moe = layer.block_sparse_moe if hasattr(layer, "block_sparse_moe") else layer.mlp
        if hasattr(moe, "num_experts"):
            return int(moe.num_experts)
        if hasattr(moe, "experts"):
            return int(len(moe.experts))
    raise RuntimeError("could not infer expert count")


def _pair_frame(recorder, num_experts: int) -> pd.DataFrame:
    frames = []
    for batch in recorder.pair_audit_batches:
        experts = batch["selected_experts"]
        assert isinstance(experts, torch.Tensor)
        tokens, top_k = experts.shape
        values: dict[str, np.ndarray] = {
            "sample_id": np.full(tokens * top_k, int(batch["sample_id"])),
            "layer": np.full(tokens * top_k, int(batch["layer"])),
            "token_position": np.repeat(np.arange(tokens), top_k),
            "rank": np.tile(np.arange(1, top_k + 1), tokens),
            "expert_id": experts.numpy().reshape(-1),
        }
        for name in (
            "gate",
            "contribution",
            "qerr",
            "qbenefit",
            "causal_local_gain",
            "output_energy",
            "low_error_energy",
            "high_error_energy",
        ):
            tensor = batch[name]
            assert isinstance(tensor, torch.Tensor)
            values[name] = tensor.numpy().reshape(-1)
        frame = pd.DataFrame(values)
        frame["rank_score"] = -frame["rank"].astype(float)
        frame["owner_group"] = np.minimum(
            frame["expert_id"].to_numpy(dtype=np.int64)
            * recorder.num_receiver_groups
            // num_experts,
            recorder.num_receiver_groups - 1,
        )
        frame["low_relative_error_coefficient"] = frame["low_error_energy"] / np.maximum(
            frame["output_energy"], 1e-30
        )
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _cross_frame(recorder) -> pd.DataFrame:
    frames = []
    for batch in recorder.cross_term_batches:
        diagonal = batch["diagonal_energy"]
        cross = batch["cross_energy"]
        total = batch["total_energy"]
        assert isinstance(diagonal, torch.Tensor)
        assert isinstance(cross, torch.Tensor)
        assert isinstance(total, torch.Tensor)
        frames.append(
            pd.DataFrame(
                {
                    "sample_id": int(batch["sample_id"]),
                    "layer": int(batch["layer"]),
                    "token_position": np.arange(len(diagonal)),
                    "diagonal_energy": diagonal.numpy(),
                    "cross_energy": cross.numpy(),
                    "total_energy": total.numpy(),
                }
            )
        )
    result = pd.concat(frames, ignore_index=True)
    result["cross_to_diagonal"] = result["cross_energy"] / np.maximum(
        result["diagonal_energy"], 1e-30
    )
    return result


def _correlations(pairs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for layer, frame in pairs.groupby("layer", sort=True):
        for score in SCORES:
            rows.append(
                {
                    "scope": "layer",
                    "layer": int(layer),
                    "score": score,
                    "pairs": int(len(frame)),
                    "spearman_vs_causal_gain": float(
                        frame[score].corr(frame["causal_local_gain"], method="spearman")
                    ),
                }
            )
    for score in SCORES:
        rows.append(
            {
                "scope": "pooled",
                "layer": -1,
                "score": score,
                "pairs": int(len(pairs)),
                "spearman_vs_causal_gain": float(
                    pairs[score].corr(pairs["causal_local_gain"], method="spearman")
                ),
            }
        )
    return pd.DataFrame(rows)


def _tile_recovery(pairs: pd.DataFrame, tile_pairs: int) -> pd.DataFrame:
    totals = {score: 0.0 for score in (*SCORES, "causal_local_gain")}
    tile_count = 0
    keys = ["sample_id", "layer", "owner_group"]
    ordered = pairs.sort_values(
        [*keys, "token_position", "rank"], kind="stable"
    )
    for _, stream in ordered.groupby(keys, sort=True):
        for start in range(0, len(stream), tile_pairs):
            tile = stream.iloc[start : start + tile_pairs]
            high_count = len(tile) - int(round(len(tile) * 0.5))
            if high_count <= 0:
                continue
            gains = tile["causal_local_gain"]
            for score in SCORES:
                chosen = tile.nlargest(high_count, score, keep="first").index
                totals[score] += float(gains.loc[chosen].sum())
            oracle = tile.nlargest(
                high_count, "causal_local_gain", keep="first"
            ).index
            totals["causal_local_gain"] += float(gains.loc[oracle].sum())
            tile_count += 1

    oracle_total = totals["causal_local_gain"]
    rank_total = totals["rank_score"]
    denominator = oracle_total - rank_total
    rows = []
    for score in SCORES:
        selected = totals[score]
        rows.append(
            {
                "score": score,
                "tiles": tile_count,
                "summed_one_pair_causal_gain": selected,
                "fraction_of_additive_oracle": selected / max(oracle_total, 1e-30),
                "rank_to_oracle_recovery": (
                    (selected - rank_total) / denominator if denominator > 0 else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows)


def _markdown(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for _, row in frame.iterrows():
        values = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                values.append(f"{value:.6g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = load_tokenizer(args.model, local_files_only=args.offline)
    model, load_seconds = load_model(
        args.model, dtype_name=args.dtype, local_files_only=args.offline
    )
    texts = get_prompts(
        args.dataset,
        args.samples,
        offset=args.offset,
        split=args.split,
    )
    exactness = validate_exact_full_path(model, tokenizer, texts[0], args.seq_len)
    recorder = patch_mixtral_moe(
        model,
        "full",
        num_receiver_groups=args.num_receiver_groups,
        audit_pair_scores=True,
    )
    for sample_id, text in enumerate(texts):
        recorder.set_sample_id(args.offset + sample_id)
        inputs = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=args.seq_len
        )
        with torch.no_grad():
            model(**inputs)
        print(f"audited {sample_id + 1}/{len(texts)}", flush=True)

    experts = _num_experts(model)
    pairs = _pair_frame(recorder, experts)
    cross = _cross_frame(recorder)
    correlations = _correlations(pairs)
    tile_recovery = _tile_recovery(pairs, args.tile_pairs)
    layer_summary = (
        correlations[correlations["scope"] == "layer"]
        .groupby("score")["spearman_vs_causal_gain"]
        .agg(["mean", "median", "min", "max"])
        .reset_index()
    )
    cross_summary = pd.DataFrame(
        [
            {
                "tokens": int(len(cross)),
                "median_cross_to_diagonal": float(cross["cross_to_diagonal"].median()),
                "median_abs_cross_to_diagonal": float(
                    cross["cross_to_diagonal"].abs().median()
                ),
                "p95_abs_cross_to_diagonal": float(
                    cross["cross_to_diagonal"].abs().quantile(0.95)
                ),
                "fraction_abs_cross_gt_0p3": float(
                    (cross["cross_to_diagonal"].abs() > 0.3).mean()
                ),
                "fraction_negative_cross": float((cross["cross_energy"] < 0).mean()),
                "pair_upgrade_positive_gain_fraction": float(
                    (pairs["causal_local_gain"] > 0).mean()
                ),
                "low_relative_error_coefficient_cv": float(
                    pairs["low_relative_error_coefficient"].std()
                    / max(pairs["low_relative_error_coefficient"].mean(), 1e-30)
                ),
            }
        ]
    )

    pairs.to_csv(output_dir / "pair_interventions.csv", index=False)
    cross.to_csv(output_dir / "cross_terms.csv", index=False)
    correlations.to_csv(output_dir / "score_causal_correlations.csv", index=False)
    layer_summary.to_csv(output_dir / "score_layer_summary.csv", index=False)
    tile_recovery.to_csv(output_dir / "tile_additive_oracle_recovery.csv", index=False)
    cross_summary.to_csv(output_dir / "cross_term_summary.csv", index=False)
    manifest = data_manifest(tokenizer, texts, args.split, args.seq_len)
    (output_dir / "data_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    sources = ("run_selector_causal_audit.py", "capture_moe.py", "fake_quant.py")
    config = {
        **vars(args),
        "model_load_seconds": load_seconds,
        "num_experts": experts,
        "baseline_equivalence": exactness,
        "source_sha256": {
            name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
            for name in sources
        },
        "boundary": (
            "fake MXFP4/FP8 one-pair local combine interventions on full-precision "
            "hidden states; not an exact multi-upgrade, downstream, native-format, "
            "communication, or latency oracle"
        ),
    }
    (output_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    pooled = correlations[correlations["scope"] == "pooled"]
    report = f"""# Selector Causal Audit

- model: `{args.model}`
- independent documents: {args.samples} (`{args.split}` offset {args.offset})
- sequence length: {args.seq_len}
- routed pairs: {len(pairs)}
- owner groups / fixed-rate tile: {args.num_receiver_groups} / {args.tile_pairs} pairs
- patched-full exactness: max/mean logit difference {exactness['max_abs_logit_diff']}/{exactness['mean_abs_logit_diff']}

## Pooled score correlation with one-pair local intervention

{_markdown(pooled)}

## Across-layer correlation stability

{_markdown(layer_summary)}

## Fixed-quota one-pair additive oracle recovery

{_markdown(tile_recovery)}

## Cross-term and relative-error diagnostics

{_markdown(cross_summary)}

## Interpretation boundary

`causal_local_gain` upgrades exactly one pair from fake MXFP4 to fake FP8 while
all other routed outputs for that token remain MXFP4.  It includes pair-vs-rest
error cancellation for that one intervention.  The tile table then sums these
one-pair gains, so its oracle is only an additive interventional oracle; it is
not the exact joint optimum after many simultaneous upgrades.  Cross terms,
BF16 accumulation order, downstream nonlinear propagation, routing drift,
native codec behavior, selector cost, network traffic, and latency remain
separate validation obligations.
"""
    (output_dir / "selector_causal_audit_report.md").write_text(
        report, encoding="utf-8"
    )
    print(report)


if __name__ == "__main__":
    main()
