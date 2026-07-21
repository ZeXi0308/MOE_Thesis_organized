#!/usr/bin/env python3
"""Strict-split GPU test of deployable Quality Isolation risk proxies.

The previous P0 selected the best proxy on the test set and used another
degradation mechanism's test-time KL. This experiment removes both problems.

For each fresh document it collects:
  * arrival-time lexical features (zero model forward);
  * early-prefix router/gate statistics from the normal full forward
    (zero *extra* forward; usable for later chunks/decode, not the first chunk);
  * full-prefill router statistics and NLL (usable for later decode);
  * fixed-tail degradation KL label from one offline evaluation forward.

Documents are deterministically shuffled and split into train/validation/test.
Ridge regularization and the feature group are selected only on validation.
The sealed test set is used once for Spearman, worst-decile AUROC/recall, and
token-budget-matched P95 isolation utility.

This is a proxy validity experiment, not a streaming serving implementation.
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
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats

from capture_moe import patch_mixtral_moe
from metrics import MetricAccumulator
from modeling import load_model, load_tokenizer
from prompts import get_prompts
from run_layer_budget_experiment import build_lut


def lexical_features(input_ids: torch.Tensor, text: str) -> dict[str, float]:
    ids = input_ids.detach().cpu().reshape(-1).numpy().astype(np.int64)
    counts = np.bincount(ids) if len(ids) else np.zeros(1)
    probs = counts[counts > 0] / max(len(ids), 1)
    entropy = float(-(probs * np.log(probs + 1e-12)).sum())
    repeat_rate = float((ids[1:] == ids[:-1]).mean()) if len(ids) > 1 else 0.0
    return {
        "char_count": float(len(text)),
        "token_count_input": float(len(ids)),
        "unique_token_ratio": float(len(np.unique(ids)) / max(len(ids), 1)),
        "token_id_entropy": entropy,
        "adjacent_repeat_rate": repeat_rate,
    }


def route_stats(
    experts_by_layer: list[np.ndarray],
    weights_by_layer: list[np.ndarray],
    num_experts: int,
    prefix: str,
) -> dict[str, float]:
    if not weights_by_layer:
        return {
            f"{prefix}_top1_weight_mean": 0.0,
            f"{prefix}_top1_weight_std": 0.0,
            f"{prefix}_top1_top2_margin_mean": 0.0,
            f"{prefix}_tail_mass_mean": 0.0,
            f"{prefix}_routing_entropy_mean": 0.0,
            f"{prefix}_rank1_hhi_mean": 0.0,
            f"{prefix}_active_expert_fraction_mean": 0.0,
            f"{prefix}_same_id_adjacent_layer_rate": 0.0,
        }

    weights = np.concatenate(weights_by_layer, axis=0)
    top1 = weights[:, 0]
    margin = top1 - weights[:, 1] if weights.shape[1] > 1 else top1
    tail_start = max(1, weights.shape[1] // 2)
    tail_mass = weights[:, tail_start:].sum(axis=1)
    normalized = weights / np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
    entropy = -(normalized * np.log(normalized + 1e-12)).sum(axis=1)
    if weights.shape[1] > 1:
        entropy = entropy / math.log(weights.shape[1])

    hhis: list[float] = []
    active_fracs: list[float] = []
    adjacent_same: list[float] = []
    for experts in experts_by_layer:
        rank1 = experts[:, 0].astype(int)
        counts = np.bincount(rank1, minlength=num_experts).astype(float)
        shares = counts / max(counts.sum(), 1.0)
        hhis.append(float((shares**2).sum()))
        active_fracs.append(float(len(np.unique(experts)) / num_experts))
    for current, nxt in zip(experts_by_layer[:-1], experts_by_layer[1:]):
        n = min(len(current), len(nxt))
        if n:
            adjacent_same.append(float((current[:n, 0] == nxt[:n, 0]).mean()))

    return {
        f"{prefix}_top1_weight_mean": float(top1.mean()),
        f"{prefix}_top1_weight_std": float(top1.std()),
        f"{prefix}_top1_top2_margin_mean": float(margin.mean()),
        f"{prefix}_tail_mass_mean": float(tail_mass.mean()),
        f"{prefix}_routing_entropy_mean": float(entropy.mean()),
        f"{prefix}_rank1_hhi_mean": float(np.mean(hhis)),
        f"{prefix}_active_expert_fraction_mean": float(np.mean(active_fracs)),
        f"{prefix}_same_id_adjacent_layer_rate": float(np.mean(adjacent_same))
        if adjacent_same
        else 0.0,
    }


def extract_router_features(
    route_batches: list[dict],
    num_experts: int,
    early_fraction: float,
) -> dict[str, float]:
    full_experts: list[np.ndarray] = []
    full_weights: list[np.ndarray] = []
    early_experts: list[np.ndarray] = []
    early_weights: list[np.ndarray] = []
    ordered = sorted(route_batches, key=lambda batch: int(batch["layer"]))
    for batch in ordered:
        experts = batch["selected_experts"].detach().cpu().numpy()
        weights = batch["routing_weights"].detach().float().cpu().numpy()
        full_experts.append(experts)
        full_weights.append(weights)
        prefix_tokens = max(1, int(math.ceil(len(experts) * early_fraction)))
        early_experts.append(experts[:prefix_tokens])
        early_weights.append(weights[:prefix_tokens])
    result = route_stats(early_experts, early_weights, num_experts, "early_route")
    result.update(route_stats(full_experts, full_weights, num_experts, "full_route"))
    return result


def spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    value = stats.spearmanr(y_true, y_pred).statistic
    return float(value) if np.isfinite(value) else 0.0


def standardize(
    train: np.ndarray, validation: np.ndarray, test: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std[std < 1e-12] = 1.0
    return (train - mean) / std, (validation - mean) / std, (test - mean) / std


def ridge_fit_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    test_x: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    train_x, validation_x, test_x = standardize(train_x, validation_x, test_x)
    train_design = np.column_stack([np.ones(len(train_x)), train_x])
    validation_design = np.column_stack([np.ones(len(validation_x)), validation_x])
    test_design = np.column_stack([np.ones(len(test_x)), test_x])
    penalty = np.eye(train_design.shape[1]) * alpha
    penalty[0, 0] = 0.0
    beta = np.linalg.pinv(train_design.T @ train_design + penalty) @ train_design.T @ train_y
    return validation_design @ beta, test_design @ beta


def worst_fraction_labels(values: np.ndarray, fraction: float) -> np.ndarray:
    count = max(1, int(math.ceil(len(values) * fraction)))
    order = np.argsort(-values)
    labels = np.zeros(len(values), dtype=int)
    labels[order[:count]] = 1
    return labels


def auc_score(labels: np.ndarray, scores: np.ndarray) -> float:
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return 0.5
    ranks = stats.rankdata(scores)
    rank_sum = float(ranks[labels == 1].sum())
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def recall_at_count(y_true: np.ndarray, y_pred: np.ndarray, fraction: float) -> float:
    count = max(1, int(math.ceil(len(y_true) * fraction)))
    true_set = set(np.argsort(-y_true)[:count].tolist())
    predicted_set = set(np.argsort(-y_pred)[:count].tolist())
    return len(true_set & predicted_set) / count


def bootstrap_spearman_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    repeats: int,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(repeats):
        indices = rng.integers(0, len(y_true), size=len(y_true))
        values.append(spearman(y_true[indices], y_pred[indices]))
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def select_under_token_budget(
    order: np.ndarray,
    token_counts: np.ndarray,
    budget_fraction: float,
) -> set[int]:
    budget = max(1, int(math.floor(token_counts.sum() * budget_fraction)))
    selected: set[int] = set()
    used = 0
    for index in order:
        cost = int(token_counts[index])
        if used + cost <= budget:
            selected.add(int(index))
            used += cost
    return selected


def allocation_utility(
    y_true: np.ndarray,
    scores: np.ndarray,
    token_counts: np.ndarray,
    quota_fractions: list[float],
    random_trials: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    baseline_p95 = float(np.quantile(y_true, 0.95))
    rows: list[dict] = []
    predicted_order = np.argsort(-scores)
    oracle_order = np.argsort(-y_true)
    for fraction in quota_fractions:
        predicted = select_under_token_budget(predicted_order, token_counts, fraction)
        oracle = select_under_token_budget(oracle_order, token_counts, fraction)

        def p95_after(selected: set[int]) -> float:
            values = y_true.copy()
            if selected:
                values[list(selected)] = 0.0
            return float(np.quantile(values, 0.95))

        random_p95 = []
        for _ in range(random_trials):
            order = rng.permutation(len(y_true))
            random_selected = select_under_token_budget(order, token_counts, fraction)
            random_p95.append(p95_after(random_selected))
        predicted_p95 = p95_after(predicted)
        oracle_p95 = p95_after(oracle)
        rows.append(
            {
                "quota_token_fraction": fraction,
                "baseline_p95_kl": baseline_p95,
                "predicted_selected_requests": len(predicted),
                "predicted_p95_kl": predicted_p95,
                "predicted_p95_reduction_pct": 100
                * (1 - predicted_p95 / max(baseline_p95, 1e-12)),
                "random_p95_kl_mean": float(np.mean(random_p95)),
                "random_p95_reduction_pct": 100
                * (1 - np.mean(random_p95) / max(baseline_p95, 1e-12)),
                "oracle_p95_kl": oracle_p95,
                "oracle_p95_reduction_pct": 100
                * (1 - oracle_p95 / max(baseline_p95, 1e-12)),
            }
        )
    return pd.DataFrame(rows)


def collect_samples(args) -> tuple[pd.DataFrame, dict]:
    texts = get_prompts(
        args.dataset,
        args.samples,
        offset=args.offset,
        split=args.split,
        seed=args.seed,
    )
    tokenizer = load_tokenizer(args.model, local_files_only=args.offline)
    model, load_seconds = load_model(
        args.model, dtype_name=args.dtype, local_files_only=args.offline
    )
    num_layers = len(model.model.layers)
    top_k = int(model.config.num_experts_per_tok)
    num_experts_value = getattr(model.config, "num_experts", None)
    if num_experts_value is None:
        num_experts_value = getattr(model.config, "num_local_experts")
    num_experts = int(num_experts_value)
    base_tail = args.base_tail if args.base_tail is not None else max(1, top_k // 2)
    lut = build_lut(
        [base_tail] * num_layers,
        top_k,
        1,
        tail_precision=args.tail_precision,
    )
    rows: list[dict] = []
    partial_path = Path(args.output_dir) / "sample_features_labels.partial.csv"

    for local_index, text in enumerate(texts):
        sample_id = args.offset + local_index
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=args.seq_len,
        )
        inputs = {key: value.to(model.device) for key, value in inputs.items()}
        row: dict[str, float | int | str] = {
            "sample_id": sample_id,
            **lexical_features(inputs["input_ids"], text),
        }

        full_recorder = patch_mixtral_moe(
            model,
            "full",
            num_receiver_groups=1,
            record_routes=True,
        )
        full_recorder.update_contrib = lambda *a, **k: None
        full_recorder.update_receiver = lambda *a, **k: None
        full_recorder.update_error = lambda *a, **k: None
        full_recorder.update_pair_audit = lambda *a, **k: None
        full_recorder.set_sample_id(sample_id)
        with torch.no_grad():
            full_logits = model(**inputs).logits.detach().cpu()
        row.update(
            extract_router_features(
                full_recorder.route_batches,
                num_experts,
                args.early_fraction,
            )
        )
        full_metrics = MetricAccumulator()
        full_sample = full_metrics.add(
            sample_id,
            full_logits,
            inputs["input_ids"].cpu(),
            attention_mask=inputs.get("attention_mask").cpu()
            if inputs.get("attention_mask") is not None
            else None,
        )
        row["full_mean_nll"] = full_sample.mean_nll
        row["full_sample_ppl"] = math.exp(full_sample.mean_nll)
        full_recorder.route_batches.clear()
        full_recorder.routing_weight_batches.clear()

        degraded_recorder = patch_mixtral_moe(
            model,
            "lut",
            num_receiver_groups=1,
            lut=lut,
            record_routes=False,
        )
        degraded_recorder.set_sample_id(sample_id)
        with torch.no_grad():
            degraded_logits = model(**inputs).logits.detach().cpu()
        degraded_metrics = MetricAccumulator()
        degraded_sample = degraded_metrics.add(
            sample_id,
            degraded_logits,
            inputs["input_ids"].cpu(),
            baseline_logits=full_logits,
            attention_mask=inputs.get("attention_mask").cpu()
            if inputs.get("attention_mask") is not None
            else None,
        )
        row["label_token_count"] = degraded_sample.token_count
        row["label_kl_sum"] = degraded_sample.kl_sum
        row["label_mean_token_kl"] = degraded_sample.mean_token_kl
        rows.append(row)
        pd.DataFrame(rows).to_csv(partial_path, index=False)
        print(
            f"[{args.model_key}] sample {local_index + 1}/{len(texts)} "
            f"KL={degraded_sample.mean_token_kl:.8f}",
            flush=True,
        )
        del full_logits, degraded_logits

    metadata = {
        "model": args.model,
        "model_key": args.model_key,
        "dataset": args.dataset,
        "split": args.split,
        "samples": args.samples,
        "offset": args.offset,
        "seq_len": args.seq_len,
        "seed": args.seed,
        "num_layers": num_layers,
        "num_experts": num_experts,
        "top_k": top_k,
        "base_tail": base_tail,
        "tail_precision": args.tail_precision,
        "early_fraction": args.early_fraction,
        "load_seconds": load_seconds,
    }
    return pd.DataFrame(rows), metadata


def evaluate_proxies(samples: pd.DataFrame, args) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    if args.train_samples + args.validation_samples >= len(samples):
        raise ValueError("train + validation must leave a non-empty test set")
    split = np.array(
        ["train"] * args.train_samples
        + ["validation"] * args.validation_samples
        + ["test"] * (len(samples) - args.train_samples - args.validation_samples)
    )
    samples = samples.copy()
    samples["split"] = split
    target = np.log10(samples["label_mean_token_kl"].to_numpy() + 1e-12)
    train_mask = split == "train"
    validation_mask = split == "validation"
    test_mask = split == "test"

    lexical = [
        "char_count",
        "token_count_input",
        "unique_token_ratio",
        "token_id_entropy",
        "adjacent_repeat_rate",
    ]
    early_route = [column for column in samples.columns if column.startswith("early_route_")]
    full_route = [column for column in samples.columns if column.startswith("full_route_")]
    groups = {
        "length_only": ["token_count_input"],
        "arrival_lexical": lexical,
        "early_router_plus_lexical": lexical + early_route,
        "full_router_plus_lexical": lexical + full_route,
        "prefill_nll_only": ["full_mean_nll"],
        "prefill_nll_plus_lexical": lexical + ["full_mean_nll"],
        "post_prefill_all": lexical + full_route + ["full_mean_nll"],
    }
    alphas = [0.0, 0.1, 1.0, 10.0, 100.0]
    proxy_rows: list[dict] = []
    predictions: dict[str, np.ndarray] = {}
    validation_scores: dict[str, float] = {}
    for group_name, columns in groups.items():
        matrix = samples[columns].to_numpy(dtype=float)
        train_x = matrix[train_mask]
        validation_x = matrix[validation_mask]
        test_x = matrix[test_mask]
        train_y = target[train_mask]
        validation_y = target[validation_mask]
        test_y = samples.loc[test_mask, "label_mean_token_kl"].to_numpy(dtype=float)
        best_alpha = None
        best_validation = -float("inf")
        for alpha in alphas:
            validation_prediction, _ = ridge_fit_predict(
                train_x,
                train_y,
                validation_x,
                test_x,
                alpha,
            )
            score = spearman(validation_y, validation_prediction)
            if score > best_validation:
                best_validation = score
                best_alpha = alpha
        assert best_alpha is not None
        combined_mask = train_mask | validation_mask
        combined_x = matrix[combined_mask]
        combined_y = target[combined_mask]
        _, best_test_prediction = ridge_fit_predict(
            combined_x,
            combined_y,
            combined_x[:1],
            test_x,
            best_alpha,
        )
        predictions[group_name] = best_test_prediction
        validation_scores[group_name] = best_validation
        worst_labels = worst_fraction_labels(test_y, 0.1)
        test_rho = spearman(test_y, best_test_prediction)
        ci_low, ci_high = bootstrap_spearman_ci(
            test_y,
            best_test_prediction,
            args.bootstrap,
            args.seed + len(proxy_rows),
        )
        proxy_rows.append(
            {
                "feature_group": group_name,
                "n_features": len(columns),
                "selected_alpha_on_validation": best_alpha,
                "validation_spearman": best_validation,
                "test_spearman": test_rho,
                "test_spearman_ci_low": ci_low,
                "test_spearman_ci_high": ci_high,
                "test_worst_decile_auc": auc_score(worst_labels, best_test_prediction),
                "test_worst_decile_recall_at_10pct": recall_at_count(
                    test_y, best_test_prediction, 0.1
                ),
                "availability": (
                    "arrival"
                    if group_name in {"length_only", "arrival_lexical"}
                    else "after_early_chunk"
                    if group_name == "early_router_plus_lexical"
                    else "after_prefill"
                ),
            }
        )

    selected_group = max(validation_scores, key=validation_scores.get)
    test_y = samples.loc[test_mask, "label_mean_token_kl"].to_numpy(dtype=float)
    test_tokens = samples.loc[test_mask, "label_token_count"].to_numpy(dtype=int)
    allocation = allocation_utility(
        test_y,
        predictions[selected_group],
        test_tokens,
        [0.1, 0.25, 0.5],
        args.random_trials,
        args.seed,
    )
    allocation.insert(0, "selected_feature_group", selected_group)
    selection = {
        "selected_feature_group": selected_group,
        "selection_rule": "highest validation Spearman only",
        "validation_spearman": validation_scores[selected_group],
        "test_samples": int(test_mask.sum()),
    }
    samples["split"] = split
    return pd.DataFrame(proxy_rows), allocation, selection


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--dataset", default="wikitext103_docs")
    parser.add_argument("--split", default="train")
    parser.add_argument("--samples", type=int, default=96)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--train-samples", type=int, default=48)
    parser.add_argument("--validation-samples", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--early-fraction", type=float, default=0.25)
    parser.add_argument("--base-tail", type=int, default=None)
    parser.add_argument(
        "--tail-precision",
        default="int4",
        choices=["int4", "mxfp4", "nvfp4"],
    )
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--random-trials", type=int, default=1000)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    samples, metadata = collect_samples(args)
    if args.collect_only:
        samples.to_csv(output / "sample_features_labels.csv", index=False)
        metadata["evidence_boundary"] = "feature/label collection only; no proxy selection"
        (output / "metadata.json").write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )
        print(f"\ncollected {len(samples)} samples to {output}")
        return
    proxy_results, allocation, selection = evaluate_proxies(samples, args)
    samples["split"] = (
        ["train"] * args.train_samples
        + ["validation"] * args.validation_samples
        + ["test"] * (len(samples) - args.train_samples - args.validation_samples)
    )
    samples.to_csv(output / "sample_features_labels.csv", index=False)
    proxy_results.to_csv(output / "proxy_results.csv", index=False)
    allocation.to_csv(output / "selected_proxy_allocation.csv", index=False)
    metadata.update(selection)
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("\nProxy results:")
    print(proxy_results.to_string(index=False))
    print("\nSelected proxy allocation:")
    print(allocation.to_string(index=False))
    print(f"\nsaved to {output}")


if __name__ == "__main__":
    main()
