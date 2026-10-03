#!/usr/bin/env python3
"""Strict single-GPU test of prefill-to-decode MoE fragility prediction.

Each document is split into a full-precision prompt and a teacher-forced decode
continuation.  Approximation is enabled only after prefill, so the experiment
does not corrupt the prompt KV cache and does not use same-prompt prefill KL as
an online proxy.

For every action, a fresh full-precision prefill cache is created and the same
ground-truth continuation tokens are decoded autoregressively.  Labels are
decode-logit KL against a full trajectory.  Arrival, router, and NLL feature
groups and ridge alpha are selected only on validation, then refit on
train+validation before opening test.

The drop policies are quality labels only in this P0: the Hugging Face expert
loop still computes all experts before outputs are masked.  A fused kernel that
skips dropped assignments is required before making a latency claim.
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
import torch.nn.functional as F

from capture_moe import patch_mixtral_moe
from metrics import MetricAccumulator
from modeling import load_model, load_tokenizer
from prompts import get_prompts
from run_quality_isolation_proxy_gpu_strict import (
    auc_score,
    bootstrap_spearman_ci,
    extract_router_features,
    lexical_features,
    recall_at_count,
    ridge_fit_predict,
    spearman,
    worst_fraction_labels,
)


def patch_policy(model, policy_name: str, record_routes: bool = False):
    recorder = patch_mixtral_moe(
        model,
        policy_name,
        num_receiver_groups=1,
        record_routes=record_routes,
    )
    recorder.update_contrib = lambda *a, **k: None
    recorder.update_receiver = lambda *a, **k: None
    recorder.update_error = lambda *a, **k: None
    recorder.update_pair_audit = lambda *a, **k: None
    return recorder


def prefill(model, prompt_ids: torch.Tensor, policy_name: str, record_routes: bool):
    recorder = patch_policy(model, policy_name, record_routes=record_routes)
    with torch.no_grad():
        output = model(input_ids=prompt_ids, use_cache=True)
    return output.logits.detach().cpu(), output.past_key_values, recorder


def decode_teacher_forced(
    model,
    cache,
    decode_input_ids: torch.Tensor,
    policy_name: str,
) -> torch.Tensor:
    patch_policy(model, policy_name, record_routes=False)
    logits: list[torch.Tensor] = []
    current_cache = cache
    for position in range(decode_input_ids.shape[1]):
        token = decode_input_ids[:, position : position + 1]
        with torch.no_grad():
            output = model(
                input_ids=token,
                past_key_values=current_cache,
                use_cache=True,
            )
        current_cache = output.past_key_values
        logits.append(output.logits[:, -1, :].detach().cpu())
    return torch.cat(logits, dim=0)


def decode_metrics(
    reference_logits: torch.Tensor,
    candidate_logits: torch.Tensor,
    target_ids: torch.Tensor,
) -> dict[str, float | int]:
    reference = reference_logits.float()
    candidate = candidate_logits.float()
    log_p = F.log_softmax(reference, dim=-1)
    log_q = F.log_softmax(candidate, dim=-1)
    p = log_p.exp()
    per_step_kl = (p * (log_p - log_q)).sum(dim=-1)
    reference_nll = F.cross_entropy(reference, target_ids, reduction="none")
    candidate_nll = F.cross_entropy(candidate, target_ids, reduction="none")
    return {
        "decode_steps": int(len(target_ids)),
        "decode_kl_sum": float(per_step_kl.sum().item()),
        "decode_mean_kl": float(per_step_kl.mean().item()),
        "decode_p95_step_kl": float(torch.quantile(per_step_kl, 0.95).item()),
        "decode_reference_nll": float(reference_nll.mean().item()),
        "decode_candidate_nll": float(candidate_nll.mean().item()),
        "decode_nll_delta": float((candidate_nll - reference_nll).mean().item()),
    }


def collect_documents(tokenizer, args) -> list[tuple[str, torch.Tensor]]:
    candidates = get_prompts(
        args.dataset,
        args.samples * 3,
        offset=args.offset,
        split=args.split,
        seed=args.seed,
    )
    required = args.prompt_len + args.decode_steps + 1
    selected: list[tuple[str, torch.Tensor]] = []
    for text in candidates:
        encoded = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=required,
        )["input_ids"]
        if encoded.shape[1] < required:
            continue
        selected.append((text, encoded[:, :required]))
        if len(selected) == args.samples:
            break
    if len(selected) < args.samples:
        raise RuntimeError(
            f"only {len(selected)} documents contain at least {required} tokens"
        )
    return selected


def collect_labels(args) -> tuple[pd.DataFrame, dict]:
    tokenizer = load_tokenizer(args.model, local_files_only=args.offline)
    model, load_seconds = load_model(
        args.model,
        dtype_name=args.dtype,
        local_files_only=args.offline,
    )
    top_k = int(model.config.num_experts_per_tok)
    num_experts_value = getattr(model.config, "num_experts", None)
    if num_experts_value is None:
        num_experts_value = getattr(model.config, "num_local_experts")
    num_experts = int(num_experts_value)
    half_k = max(1, top_k // 2)
    three_quarter_k = max(1, int(math.ceil(top_k * 0.75)))
    actions = [
        f"fp8top{half_k}_rest_int4",
        "rankk_drop_renorm",
        f"keep{three_quarter_k}_drop_renorm",
        f"keep{half_k}_drop_renorm",
    ]
    documents = collect_documents(tokenizer, args)
    rows: list[dict] = []
    output_dir = Path(args.output_dir)
    partial_path = output_dir / "decode_fragility.partial.csv"

    for local_index, (text, all_ids_cpu) in enumerate(documents):
        sample_id = args.offset + local_index
        prompt_ids = all_ids_cpu[:, : args.prompt_len].to(model.device)
        decode_inputs = all_ids_cpu[
            :, args.prompt_len : args.prompt_len + args.decode_steps
        ].to(model.device)
        decode_targets = all_ids_cpu[
            0,
            args.prompt_len + 1 : args.prompt_len + args.decode_steps + 1,
        ].cpu()
        row: dict[str, float | int | str] = {
            "sample_id": sample_id,
            **lexical_features(
                prompt_ids,
                tokenizer.decode(prompt_ids[0].detach().cpu()),
            ),
        }

        prefill_logits, reference_cache, route_recorder = prefill(
            model,
            prompt_ids,
            "full",
            record_routes=True,
        )
        route_recorder.set_sample_id(sample_id)
        row.update(
            extract_router_features(
                route_recorder.route_batches,
                num_experts,
                args.early_fraction,
            )
        )
        prefill_metric = MetricAccumulator().add(
            sample_id,
            prefill_logits,
            prompt_ids.cpu(),
        )
        row["full_mean_nll"] = prefill_metric.mean_nll
        row["full_sample_ppl"] = math.exp(prefill_metric.mean_nll)
        route_recorder.route_batches.clear()
        route_recorder.routing_weight_batches.clear()

        reference_decode_logits = decode_teacher_forced(
            model,
            reference_cache,
            decode_inputs,
            "full",
        )
        for action in actions:
            _, candidate_cache, _ = prefill(
                model,
                prompt_ids,
                "full",
                record_routes=False,
            )
            candidate_decode_logits = decode_teacher_forced(
                model,
                candidate_cache,
                decode_inputs,
                action,
            )
            metrics = decode_metrics(
                reference_decode_logits,
                candidate_decode_logits,
                decode_targets,
            )
            for key, value in metrics.items():
                row[f"{action}__{key}"] = value
            del candidate_decode_logits, candidate_cache
        rows.append(row)
        pd.DataFrame(rows).to_csv(partial_path, index=False)
        action_summary = ", ".join(
            f"{action}={row[f'{action}__decode_mean_kl']:.5f}"
            for action in actions
        )
        print(
            f"[{args.model_key}] {local_index + 1}/{len(documents)} {action_summary}",
            flush=True,
        )
        del reference_decode_logits, reference_cache, prefill_logits

    metadata = {
        "model": args.model,
        "model_key": args.model_key,
        "dataset": args.dataset,
        "split": args.split,
        "samples": args.samples,
        "offset": args.offset,
        "prompt_len": args.prompt_len,
        "decode_steps": args.decode_steps,
        "top_k": top_k,
        "num_experts": num_experts,
        "actions": actions,
        "load_seconds": load_seconds,
        "evidence_boundary": (
            "teacher-forced decode quality with real KV cache; drop policies "
            "mask outputs after expert computation and do not yet save latency"
        ),
    }
    return pd.DataFrame(rows), metadata


def analyze(
    samples: pd.DataFrame,
    actions: list[str],
    args,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    split = np.array(
        ["train"] * args.train_samples
        + ["validation"] * args.validation_samples
        + ["test"]
        * (len(samples) - args.train_samples - args.validation_samples)
    )
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
    full_route = [
        column for column in samples.columns if column.startswith("full_route_")
    ]
    groups = {
        "arrival_lexical": lexical,
        "full_router_plus_lexical": lexical + full_route,
        "prefill_nll_only": ["full_mean_nll"],
        "post_prefill_all": lexical + full_route + ["full_mean_nll"],
    }
    proxy_rows: list[dict] = []
    test_predictions: dict[str, np.ndarray] = {}
    selected_groups: dict[str, str] = {}
    for action_index, action in enumerate(actions):
        target_raw = samples[f"{action}__decode_mean_kl"].to_numpy(dtype=float)
        target = np.log10(target_raw + 1e-12)
        best_group = None
        best_alpha = None
        best_validation = -float("inf")
        best_columns = None
        for group_name, columns in groups.items():
            matrix = samples[columns].to_numpy(dtype=float)
            for alpha in [0.0, 0.1, 1.0, 10.0, 100.0]:
                validation_prediction, _ = ridge_fit_predict(
                    matrix[train_mask],
                    target[train_mask],
                    matrix[validation_mask],
                    matrix[test_mask],
                    alpha,
                )
                score = spearman(
                    target[validation_mask],
                    validation_prediction,
                )
                if score > best_validation:
                    best_validation = score
                    best_group = group_name
                    best_alpha = alpha
                    best_columns = columns
        assert best_group is not None and best_alpha is not None
        assert best_columns is not None
        selected_groups[action] = best_group
        matrix = samples[best_columns].to_numpy(dtype=float)
        combined_mask = train_mask | validation_mask
        _, test_prediction = ridge_fit_predict(
            matrix[combined_mask],
            target[combined_mask],
            matrix[combined_mask][:1],
            matrix[test_mask],
            best_alpha,
        )
        test_predictions[action] = test_prediction
        test_y = target_raw[test_mask]
        rho = spearman(test_y, test_prediction)
        ci_low, ci_high = bootstrap_spearman_ci(
            test_y,
            test_prediction,
            args.bootstrap,
            args.seed + action_index,
        )
        worst_labels = worst_fraction_labels(test_y, 0.1)
        proxy_rows.append(
            {
                "action": action,
                "selected_feature_group": best_group,
                "selected_alpha": best_alpha,
                "validation_spearman": best_validation,
                "test_spearman": rho,
                "test_spearman_ci_low": ci_low,
                "test_spearman_ci_high": ci_high,
                "test_worst_decile_auc": auc_score(
                    worst_labels, test_prediction
                ),
                "test_worst_decile_recall_at_10pct": recall_at_count(
                    test_y, test_prediction, 0.1
                ),
                "test_worst_quartile_recall_at_25pct": recall_at_count(
                    test_y, test_prediction, 0.25
                ),
            }
        )

    true_correlation_rows: list[dict] = []
    transfer_rows: list[dict] = []
    for source in actions:
        source_prediction = test_predictions[source]
        for target_action in actions:
            target_y = samples.loc[
                test_mask, f"{target_action}__decode_mean_kl"
            ].to_numpy(dtype=float)
            true_source = samples.loc[
                test_mask, f"{source}__decode_mean_kl"
            ].to_numpy(dtype=float)
            true_correlation_rows.append(
                {
                    "source_action": source,
                    "target_action": target_action,
                    "test_true_harm_spearman": spearman(
                        true_source, target_y
                    ),
                }
            )
            transfer_rows.append(
                {
                    "proxy_trained_for_action": source,
                    "evaluated_harm_action": target_action,
                    "test_transfer_spearman": spearman(
                        target_y, source_prediction
                    ),
                }
            )
    summary = {
        "selected_groups": selected_groups,
        "test_samples": int(test_mask.sum()),
        "selection_rule": (
            "feature group and ridge alpha selected on validation separately "
            "for each action; test opened after refit on train+validation"
        ),
    }
    samples = samples.copy()
    samples["split"] = split
    return (
        pd.DataFrame(proxy_rows),
        pd.DataFrame(true_correlation_rows),
        pd.DataFrame(transfer_rows),
        summary,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--dataset", default="wikitext103_docs")
    parser.add_argument("--split", default="train")
    parser.add_argument("--samples", type=int, default=48)
    parser.add_argument("--offset", type=int, default=184)
    parser.add_argument("--train-samples", type=int, default=24)
    parser.add_argument("--validation-samples", type=int, default=8)
    parser.add_argument("--prompt-len", type=int, default=64)
    parser.add_argument("--decode-steps", type=int, default=16)
    parser.add_argument("--early-fraction", type=float, default=0.25)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if args.train_samples + args.validation_samples >= args.samples:
        raise ValueError("train + validation must leave a non-empty test set")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    samples, metadata = collect_labels(args)
    actions = list(metadata["actions"])
    proxy_results, harm_correlations, transfer, analysis_summary = analyze(
        samples,
        actions,
        args,
    )
    split = (
        ["train"] * args.train_samples
        + ["validation"] * args.validation_samples
        + ["test"] * (len(samples) - args.train_samples - args.validation_samples)
    )
    samples["split"] = split
    samples.to_csv(output / "decode_fragility_samples.csv", index=False)
    proxy_results.to_csv(output / "decode_proxy_results.csv", index=False)
    harm_correlations.to_csv(output / "action_harm_correlations.csv", index=False)
    transfer.to_csv(output / "cross_action_proxy_transfer.csv", index=False)
    metadata.update(analysis_summary)
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    print("\nDecode proxy results:")
    print(proxy_results.to_string(index=False))
    print("\nTrue harm correlations:")
    print(harm_correlations.to_string(index=False))
    print("\nCross-action proxy transfer:")
    print(transfer.to_string(index=False))


if __name__ == "__main__":
    main()
