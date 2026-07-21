#!/usr/bin/env python3
"""GPU task-quality gate for receiver-aware online control.

This is the missing policy-level quality experiment. It executes the actual
lane decisions on real MoE expert outputs and measures MMLU accuracy, paired
correct-answer NLL harm, worst-request harm and CVaR. It does not substitute
``low_frac * KL_uniform_low`` for quality.

Transport semantics are explicit:
  * local expert-output pairs remain BF16;
  * remote pairs use ``--high-precision`` (default FP8);
  * remote pairs selected by a low lane use ``--low-precision`` (default INT4).

Five reward arms are evaluated: uniform_full (no low lane, remote FP8),
uniform_low, calibration-only static, causal previous-step threshold, and the
direct-benefit EWMA+hysteresis controller. A separate BF16 reference supplies
absolute task harm; λ uses incremental harm relative to uniform_full because
the communication saving is also incremental to that remote-FP8 baseline.
Calibration questions fit only the load scale and static profile; test questions
never tune thresholds.

Saving is recomputed from the exact same action trace using an analytic
bottleneck-wire model plus measured codec tax. It is action-consistent but
still not real NCCL/RDMA latency. This script closes the quality side of the
gate; real multi-GPU latency is a later, separate gate.
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
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import load_dataset

from capture_moe import patch_mixtral_moe
from modeling import load_model, load_tokenizer, resolve_device
from receiver_lane_policy import (
    REFERENCE_ARM,
    RECEIVER_ARMS,
    ReceiverLaneController,
    ReceiverPolicyConfig,
)


MODEL_SPECS = {
    "olmoe": {
        "model": "allenai/OLMoE-1B-7B-0924",
        "num_experts": 64,
        "hidden_size": 2048,
        "codec_pack_us": 26.12175941467285,
        "codec_unpack_us": 25.37951946258545,
    },
    "llmjp": {
        "model": "llm-jp/optimal-sparsity-math-d512-E32-k16-920M-A520M",
        "num_experts": 32,
        "hidden_size": 512,
        "codec_pack_us": 17.097280025482178,
        "codec_unpack_us": 16.322879791259766,
    },
}

DEFAULT_SUBJECTS = (
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "college_chemistry",
    "computer_security",
    "conceptual_physics",
    "high_school_geography",
    "high_school_mathematics",
    "high_school_us_history",
    "machine_learning",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=None,
                   help="defaults to the canonical checkpoint bound to --model-key")
    p.add_argument("--model-key", default="olmoe", choices=("olmoe", "llmjp"))
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--offline", action="store_true")
    p.add_argument("--subjects", default=",".join(DEFAULT_SUBJECTS))
    p.add_argument("--calib-per-subject", type=int, default=4)
    p.add_argument("--test-per-subject", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=8,
                   help="use >= ep-size so balanced/hotspot are meaningful within each microbatch")
    p.add_argument("--seq-len", type=int, default=320)
    p.add_argument("--seed", type=int, default=20260721)
    p.add_argument("--ep-size", type=int, default=8)
    p.add_argument("--gpus-per-node", type=int, default=4)
    p.add_argument("--placement", default="contiguous", choices=("contiguous", "round_robin"))
    p.add_argument("--high-precision", default="fp8")
    p.add_argument("--low-precision", default="int4")
    p.add_argument("--alpha", type=float, default=0.6)
    p.add_argument("--high-quantile", type=float, default=0.6)
    p.add_argument("--gap-ratio", type=float, default=0.5)
    p.add_argument("--dwell-min", type=int, default=1)
    p.add_argument("--origin-modes", default="balanced,hotspot")
    p.add_argument("--policies", default=",".join(RECEIVER_ARMS))
    p.add_argument("--cvar-fraction", type=float, default=0.10)
    p.add_argument("--n-bootstrap", type=int, default=2000)
    p.add_argument("--lambda-grid", default="0,0.1,0.2,0.5,1,2,5,10")
    p.add_argument("--inter-node-gbps", type=float, default=200.0)
    p.add_argument("--codec-pack-us", type=float, default=None,
                   help="defaults to the model-specific real GPU codec measurement")
    p.add_argument("--codec-unpack-us", type=float, default=None,
                   help="defaults to the model-specific real GPU codec measurement")
    p.add_argument("--high-scale-bytes", type=float, default=4.0)
    p.add_argument("--low-scale-bytes", type=float, default=4.0)
    p.add_argument("--codec-tile-rows", type=int, default=32)
    p.add_argument("--codec-tax-mode", default="once_per_step",
                   choices=("once_per_step", "serialized_tiles"))
    p.add_argument("--resume", action="store_true")
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()
    if args.batch_size < 1 or args.seq_len < 8:
        p.error("batch-size must be positive and seq-len must be >= 8")
    if args.batch_size < args.ep_size:
        p.error("batch-size must be >= ep-size for a meaningful per-microbatch receiver topology")
    if not 0 < args.cvar_fraction <= 1:
        p.error("cvar-fraction must be in (0, 1]")
    spec = MODEL_SPECS[args.model_key]
    if args.model is None:
        args.model = spec["model"]
    if args.codec_pack_us is None:
        args.codec_pack_us = spec["codec_pack_us"]
    if args.codec_unpack_us is None:
        args.codec_unpack_us = spec["codec_unpack_us"]
    return args


def format_prompt(question: str, choices: list[str]) -> str:
    lines = ["The following is a multiple choice question. Answer with the letter.", "", question, ""]
    for letter, choice in zip(("A", "B", "C", "D"), choices):
        lines.append(f"{letter}. {choice}")
    lines.extend(("", "Answer:"))
    return "\n".join(lines)


def resolve_choice_token_ids(tokenizer) -> tuple[list[int], str]:
    for prefix in (" ", ""):
        encoded = [tokenizer(letter if not prefix else prefix + letter, add_special_tokens=False)["input_ids"]
                   for letter in ("A", "B", "C", "D")]
        if all(len(ids) == 1 for ids in encoded):
            token_ids = [int(ids[0]) for ids in encoded]
            if len(set(token_ids)) == 4:
                return token_ids, prefix
    raise RuntimeError(
        "A/B/C/D are not four distinct one-token labels for this tokenizer. "
        "A sequence-level choice scorer is required before using this model."
    )


def load_mmlu_rows(
    split: str,
    subjects: list[str],
    per_subject: int,
    seed: int,
    offline: bool,
) -> list[dict[str, object]]:
    ds = load_dataset(
        "cais/mmlu",
        "all",
        split=split,
        download_mode="reuse_dataset_if_exists" if offline else None,
    )
    buckets = {subject: [] for subject in subjects}
    for row in ds:
        subject = str(row["subject"])
        if subject in buckets:
            buckets[subject].append({
                "question": str(row["question"]),
                "choices": list(row["choices"]),
                "answer": int(row["answer"]),
                "subject": subject,
            })
    rng = random.Random(seed)
    selected: list[dict[str, object]] = []
    for subject in subjects:
        rows = buckets[subject]
        if not rows:
            raise RuntimeError(f"MMLU split={split!r} has no rows for subject={subject!r}")
        rng.shuffle(rows)
        selected.extend(rows[: min(per_subject, len(rows))])
    rng.shuffle(selected)
    for request_id, row in enumerate(selected):
        row["request_id"] = request_id
    return selected


def filter_overlength_rows(tokenizer, rows: list[dict[str, object]], seq_len: int) -> tuple[list[dict[str, object]], list[int]]:
    kept: list[dict[str, object]] = []
    dropped: list[int] = []
    for row in rows:
        prompt = format_prompt(str(row["question"]), list(row["choices"]))
        token_count = len(tokenizer(prompt, add_special_tokens=True, truncation=False)["input_ids"])
        if token_count <= seq_len:
            copied = dict(row)
            copied["prompt_token_count"] = token_count
            kept.append(copied)
        else:
            dropped.append(int(row["request_id"]))
    for request_id, row in enumerate(kept):
        row["request_id"] = request_id
    return kept, dropped


def stable_run_fingerprint(args: argparse.Namespace, calibration_rows: list[dict[str, object]], test_rows: list[dict[str, object]]) -> dict[str, object]:
    sample_payload = [
        (split, row["subject"], row["question"], row["choices"], row["answer"])
        for split, rows in (("calibration", calibration_rows), ("test", test_rows))
        for row in rows
    ]
    relevant_config = {
        key: value for key, value in vars(args).items()
        if key not in {"resume", "output_dir"}
    }
    payload = json.dumps(
        {"config": relevant_config, "samples": sample_payload},
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )
    return {
        "sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        "config": relevant_config,
        "n_calibration": len(calibration_rows),
        "n_test": len(test_rows),
    }


def validate_model_spec(model, model_key: str) -> None:
    spec = MODEL_SPECS[model_key]
    num_experts = int(getattr(model.config, "num_local_experts", getattr(model.config, "num_experts", -1)))
    hidden_size = int(getattr(model.config, "hidden_size", -1))
    if num_experts != spec["num_experts"] or hidden_size != spec["hidden_size"]:
        raise RuntimeError(
            f"model-key={model_key!r} expects experts={spec['num_experts']}, hidden={spec['hidden_size']}, "
            f"but loaded model reports experts={num_experts}, hidden={hidden_size}; refusing to mix model quality and codec parameters"
        )


def receiver_assignment(request_ids: list[int], mode: str, ep_size: int) -> torch.Tensor:
    batch_size = len(request_ids)
    if mode == "balanced":
        return torch.tensor([index % ep_size for index in range(batch_size)], dtype=torch.long)
    if mode == "hotspot":
        hotspot_count = max(1, math.ceil(batch_size * 0.5))
        receivers = [0] * hotspot_count
        receivers.extend(1 + (i % max(ep_size - 1, 1)) for i in range(batch_size - hotspot_count))
        return torch.tensor(receivers, dtype=torch.long)
    raise ValueError(f"unknown origin mode: {mode}")


def score_batch(
    model,
    tokenizer,
    controller: ReceiverLaneController,
    rows: list[dict[str, object]],
    origin_mode: str,
    choice_token_ids: list[int],
    seq_len: int,
) -> tuple[np.ndarray, float]:
    prompts = [format_prompt(str(row["question"]), list(row["choices"])) for row in rows]
    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=False,
    )
    if int(encoded["input_ids"].shape[1]) > seq_len:
        raise RuntimeError(
            "an over-length MMLU prompt reached score_batch; filter rows before evaluation "
            "instead of truncating away choices or the Answer suffix"
        )
    request_ids = [int(row["request_id"]) for row in rows]
    receivers = receiver_assignment(request_ids, origin_mode, controller.config.ep_size)
    controller.begin_forward(receivers, encoded["attention_mask"], torch.tensor(request_ids, dtype=torch.long))
    inputs = {key: value.to(model.device) for key, value in encoded.items()}
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    try:
        with torch.inference_mode():
            logits = model(**inputs, use_cache=False).logits
    finally:
        controller.end_forward()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    last_positions = inputs["attention_mask"].sum(dim=1) - 1
    row_indices = torch.arange(len(rows), device=logits.device)
    next_logits = logits[row_indices, last_positions, :].float()
    log_probs = F.log_softmax(next_logits, dim=-1)
    choice_ids = torch.tensor(choice_token_ids, device=logits.device, dtype=torch.long)
    choice_ll = log_probs.index_select(dim=1, index=choice_ids)
    return choice_ll.detach().cpu().numpy(), elapsed


def run_rows(
    model,
    tokenizer,
    controller: ReceiverLaneController,
    rows: list[dict[str, object]],
    origin_mode: str,
    choice_token_ids: list[int],
    args: argparse.Namespace,
    partial_path: Path | None = None,
) -> pd.DataFrame:
    controller.reset_runtime(clear_observations=True)
    controller.install_on_model(model)
    results: list[dict[str, object]] = []
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start : start + args.batch_size]
        try:
            choice_ll, elapsed = score_batch(
                model, tokenizer, controller, batch, origin_mode, choice_token_ids, args.seq_len
            )
        except torch.OutOfMemoryError as exc:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise RuntimeError(
                f"CUDA OOM at batch_size={args.batch_size}, seq_len={args.seq_len}; "
                "reduce --batch-size or --seq-len and rerun all arms with the same setting"
            ) from exc
        for row, lls in zip(batch, choice_ll):
            answer = int(row["answer"])
            prediction = int(np.argmax(lls))
            incorrect = np.delete(lls, answer)
            results.append({
                "request_id": int(row["request_id"]),
                "subject": str(row["subject"]),
                "answer": answer,
                "prediction": prediction,
                "correct": int(prediction == answer),
                "correct_choice_nll": float(-lls[answer]),
                "correct_choice_margin": float(lls[answer] - incorrect.max()),
                "ll_A": float(lls[0]),
                "ll_B": float(lls[1]),
                "ll_C": float(lls[2]),
                "ll_D": float(lls[3]),
                "batch_forward_seconds": elapsed,
            })
        if partial_path is not None:
            pd.DataFrame(results).to_csv(partial_path, index=False)
        print(
            f"[{controller.config.arm}/{origin_mode}] {min(start + len(batch), len(rows))}/{len(rows)} "
            f"acc={np.mean([r['correct'] for r in results]):.4f}",
            flush=True,
        )
    frame = pd.DataFrame(results)
    exposure = controller.request_exposure()
    for column in ("remote_pairs", "low_pairs", "low_frac", "wire_bytes"):
        frame[column] = frame["request_id"].map(
            lambda request_id: exposure.get(int(request_id), {}).get(column, 0)
        )
    return frame


def calibration_pass(
    model,
    tokenizer,
    rows: list[dict[str, object]],
    choice_token_ids: list[int],
    args: argparse.Namespace,
) -> tuple[dict[tuple[int, int], float], float, float, float, ReceiverLaneController]:
    config = ReceiverPolicyConfig(
        arm="uniform_full",
        ep_size=args.ep_size,
        gpus_per_node=args.gpus_per_node,
        placement=args.placement,
        high_precision=args.high_precision,
        low_precision=args.low_precision,
        high_scale_bytes_per_vector=args.high_scale_bytes,
        low_scale_bytes_per_vector=args.low_scale_bytes,
        inter_node_gbps=args.inter_node_gbps,
        codec_pack_us=args.codec_pack_us,
        codec_unpack_us=args.codec_unpack_us,
        codec_tile_rows=args.codec_tile_rows,
        codec_tax_mode=args.codec_tax_mode,
    )
    collector = ReceiverLaneController(config)
    collector.reset_runtime(clear_observations=True)
    collector.install_on_model(model)
    for batch_index, start in enumerate(range(0, len(rows), args.batch_size)):
        batch = rows[start : start + args.batch_size]
        mode = "balanced" if batch_index % 2 == 0 else "hotspot"
        score_batch(model, tokenizer, collector, batch, mode, choice_token_ids, args.seq_len)
        print(f"[calibration/{mode}] {min(start + len(batch), len(rows))}/{len(rows)}", flush=True)
    profile = collector.static_profile()
    threshold_high, threshold_low, static_threshold = collector.fitted_thresholds(
        args.high_quantile, args.gap_ratio
    )
    return profile, threshold_high, threshold_low, static_threshold, collector


def bootstrap_mean_ci(values: np.ndarray, n_bootstrap: int, seed: int) -> tuple[float, float]:
    if len(values) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(n_bootstrap, len(values)))
    means = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def cvar(values: np.ndarray, fraction: float) -> float:
    if len(values) == 0:
        return float("nan")
    tail_count = max(1, int(math.ceil(len(values) * fraction)))
    return float(np.sort(values)[-tail_count:].mean())


def stratified_cvar_ci(
    values: np.ndarray,
    subjects: np.ndarray,
    fraction: float,
    n_bootstrap: int,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    subject_indices = [np.flatnonzero(subjects == subject) for subject in np.unique(subjects)]
    boot = []
    for _ in range(n_bootstrap):
        sampled = np.concatenate([
            indices[rng.integers(0, len(indices), size=len(indices))]
            for indices in subject_indices
        ])
        boot.append(cvar(values[sampled], fraction))
    return float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def action_consistent_saving(step_df: pd.DataFrame) -> dict[str, float]:
    baseline_us = float(step_df["baseline_step_us"].sum())
    wire_us = float((step_df["policy_step_us"] - step_df["codec_step_us"]).sum())
    optimistic_us = wire_us + float(step_df["codec_step_us_optimistic"].sum())
    serialized_tiles_us = wire_us + float(step_df["codec_step_us_serialized_tiles"].sum())
    selected_us = float(step_df["policy_step_us"].sum())

    def saving(policy_us: float) -> float:
        return 1.0 - policy_us / baseline_us if baseline_us > 0 else 0.0

    return {
        "baseline_us": baseline_us,
        "selected_policy_us": selected_us,
        "selected_saving": saving(selected_us),
        "optimistic_policy_us": optimistic_us,
        "optimistic_saving": saving(optimistic_us),
        "serialized_tiles_policy_us": serialized_tiles_us,
        "serialized_tiles_saving": saving(serialized_tiles_us),
    }


def summarize_policy(
    policy_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    uniform_full_df: pd.DataFrame,
    step_df: pd.DataFrame,
    arm: str,
    mode: str,
    args: argparse.Namespace,
) -> dict[str, object]:
    merged = policy_df.merge(
        reference_df[["request_id", "correct", "correct_choice_nll", "correct_choice_margin"]],
        on="request_id",
        suffixes=("", "_reference"),
        validate="one_to_one",
    )
    merged = merged.merge(
        uniform_full_df[["request_id", "correct", "correct_choice_nll", "correct_choice_margin"]],
        on="request_id",
        suffixes=("", "_uniform_full"),
        validate="one_to_one",
    )
    absolute_accuracy_delta = (
        merged["correct"].to_numpy(float) - merged["correct_reference"].to_numpy(float)
    )
    incremental_accuracy_delta = (
        merged["correct"].to_numpy(float) - merged["correct_uniform_full"].to_numpy(float)
    )
    absolute_nll_harm = (
        merged["correct_choice_nll"].to_numpy(float)
        - merged["correct_choice_nll_reference"].to_numpy(float)
    )
    incremental_nll_harm = (
        merged["correct_choice_nll"].to_numpy(float)
        - merged["correct_choice_nll_uniform_full"].to_numpy(float)
    )
    incremental_correctness_harm = -incremental_accuracy_delta
    positive_incremental_correctness_harm = np.clip(
        incremental_correctness_harm, 0.0, None
    )
    positive_incremental_nll_harm = np.clip(incremental_nll_harm, 0.0, None)
    margin_harm = merged["correct_choice_margin_reference"].to_numpy(float) - merged["correct_choice_margin"].to_numpy(float)
    absolute_acc_ci = bootstrap_mean_ci(absolute_accuracy_delta, args.n_bootstrap, args.seed + 11)
    incremental_acc_ci = bootstrap_mean_ci(incremental_accuracy_delta, args.n_bootstrap, args.seed + 13)
    absolute_nll_ci = bootstrap_mean_ci(absolute_nll_harm, args.n_bootstrap, args.seed + 17)
    incremental_nll_ci = bootstrap_mean_ci(incremental_nll_harm, args.n_bootstrap, args.seed + 19)
    subjects = merged["subject"].to_numpy(str)
    positive_accuracy_cvar = cvar(
        positive_incremental_correctness_harm, args.cvar_fraction
    )
    positive_accuracy_cvar_ci = stratified_cvar_ci(
        positive_incremental_correctness_harm,
        subjects,
        args.cvar_fraction,
        args.n_bootstrap,
        args.seed + 21,
    )
    incremental_cvar = cvar(incremental_nll_harm, args.cvar_fraction)
    positive_incremental_cvar = cvar(positive_incremental_nll_harm, args.cvar_fraction)
    incremental_cvar_ci = stratified_cvar_ci(
        incremental_nll_harm, subjects, args.cvar_fraction, args.n_bootstrap, args.seed + 23
    )
    positive_incremental_cvar_ci = stratified_cvar_ci(
        positive_incremental_nll_harm, subjects, args.cvar_fraction, args.n_bootstrap, args.seed + 29
    )
    saving = action_consistent_saving(step_df)
    return {
        "model": args.model_key,
        "origin_mode": mode,
        "arm": arm,
        "n_questions": len(merged),
        "bf16_reference_accuracy": float(merged["correct_reference"].mean()),
        "uniform_full_accuracy": float(merged["correct_uniform_full"].mean()),
        "accuracy": float(merged["correct"].mean()),
        "absolute_accuracy_delta_vs_bf16": float(absolute_accuracy_delta.mean()),
        "absolute_accuracy_delta_ci_low": absolute_acc_ci[0],
        "absolute_accuracy_delta_ci_high": absolute_acc_ci[1],
        "incremental_accuracy_delta_vs_uniform_full": float(incremental_accuracy_delta.mean()),
        "incremental_accuracy_harm": max(0.0, float(-incremental_accuracy_delta.mean())),
        "incremental_accuracy_harm_ucb95": max(0.0, float(-incremental_acc_ci[0])),
        "incremental_accuracy_delta_ci_low": incremental_acc_ci[0],
        "incremental_accuracy_delta_ci_high": incremental_acc_ci[1],
        "cvar_incremental_positive_accuracy_harm": positive_accuracy_cvar,
        "cvar_incremental_positive_accuracy_harm_ucb95": positive_accuracy_cvar_ci[1],
        "cvar_incremental_positive_accuracy_harm_ci_low": positive_accuracy_cvar_ci[0],
        "cvar_incremental_positive_accuracy_harm_ci_high": positive_accuracy_cvar_ci[1],
        "lost_correct_questions_vs_bf16": int(((merged["correct_reference"] == 1) & (merged["correct"] == 0)).sum()),
        "gained_correct_questions_vs_bf16": int(((merged["correct_reference"] == 0) & (merged["correct"] == 1)).sum()),
        "mean_absolute_nll_harm_vs_bf16": float(absolute_nll_harm.mean()),
        "mean_absolute_nll_harm_ci_low": absolute_nll_ci[0],
        "mean_absolute_nll_harm_ci_high": absolute_nll_ci[1],
        "mean_incremental_nll_harm_vs_uniform_full": float(incremental_nll_harm.mean()),
        "mean_incremental_nll_harm_ci_low": incremental_nll_ci[0],
        "mean_incremental_nll_harm_ci_high": incremental_nll_ci[1],
        "mean_incremental_positive_nll_harm": float(positive_incremental_nll_harm.mean()),
        "cvar_incremental_nll_harm": incremental_cvar,
        "cvar_incremental_nll_harm_ci_low": incremental_cvar_ci[0],
        "cvar_incremental_nll_harm_ci_high": incremental_cvar_ci[1],
        "cvar_incremental_positive_nll_harm": positive_incremental_cvar,
        "cvar_incremental_positive_nll_harm_ucb95": positive_incremental_cvar_ci[1],
        "cvar_incremental_positive_nll_harm_ci_low": positive_incremental_cvar_ci[0],
        "cvar_incremental_positive_nll_harm_ci_high": positive_incremental_cvar_ci[1],
        "worst_incremental_positive_nll_harm": float(positive_incremental_nll_harm.max(initial=0.0)),
        "mean_margin_harm": float(margin_harm.mean()),
        "remote_pairs": int(merged["remote_pairs"].sum()),
        "low_pairs": int(merged["low_pairs"].sum()),
        "measured_low_frac": float(merged["low_pairs"].sum() / max(merged["remote_pairs"].sum(), 1)),
        "simulated_saving_fraction": saving["selected_saving"],
        "simulated_saving_optimistic_codec": saving["optimistic_saving"],
        "simulated_saving_serialized_tiles_codec": saving["serialized_tiles_saving"],
        "simulated_baseline_us": saving["baseline_us"],
        "simulated_policy_us": saving["selected_policy_us"],
        "simulated_policy_us_optimistic_codec": saving["optimistic_policy_us"],
        "simulated_policy_us_serialized_tiles_codec": saving["serialized_tiles_policy_us"],
        "saving_evidence": "same-action analytic bottleneck wire time + measured codec sensitivity bounds; not real NCCL/RDMA latency",
    }


HARM_COLUMNS = (
    "incremental_accuracy_harm",
    "incremental_accuracy_harm_ucb95",
    "cvar_incremental_positive_accuracy_harm",
    "cvar_incremental_positive_accuracy_harm_ucb95",
    "mean_incremental_positive_nll_harm",
    "cvar_incremental_positive_nll_harm",
    "cvar_incremental_positive_nll_harm_ucb95",
)
SAVING_COLUMNS = (
    "simulated_saving_fraction",
    "simulated_saving_optimistic_codec",
    "simulated_saving_serialized_tiles_codec",
)


def build_lambda_winners(summary: pd.DataFrame, lambdas: list[float]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (model, mode), cell in summary.groupby(["model", "origin_mode"]):
        for saving_column in SAVING_COLUMNS:
            if cell[saving_column].isna().any():
                continue
            for harm_column in HARM_COLUMNS:
                for lam in lambdas:
                    reward = cell[saving_column] - lam * cell[harm_column]
                    best = float(reward.max())
                    winners = cell.loc[np.isclose(reward, best, atol=1e-12), "arm"].astype(str).tolist()
                    rows.append({
                        "model": model,
                        "origin_mode": mode,
                        "saving_basis": saving_column,
                        "harm_basis": harm_column,
                        "lambda": lam,
                        "winner": ";".join(winners),
                        "winner_reward": best,
                        **{
                            f"reward_{arm}": float(value)
                            for arm, value in zip(cell["arm"], reward)
                        },
                    })
    return pd.DataFrame(rows)


def build_exact_lambda_intervals(summary: pd.DataFrame) -> pd.DataFrame:
    """Return exact upper-envelope intervals for affine reward lines."""
    rows: list[dict[str, object]] = []
    for (model, mode), cell in summary.groupby(["model", "origin_mode"]):
        arms = cell["arm"].astype(str).to_numpy()
        for saving_column in SAVING_COLUMNS:
            savings = cell[saving_column].to_numpy(float)
            for harm_column in HARM_COLUMNS:
                harms = cell[harm_column].to_numpy(float)
                boundaries = {0.0}
                for i in range(len(arms)):
                    for j in range(i + 1, len(arms)):
                        denominator = harms[i] - harms[j]
                        if abs(denominator) <= 1e-15:
                            continue
                        crossing = (savings[i] - savings[j]) / denominator
                        if math.isfinite(crossing) and crossing > 0:
                            boundaries.add(float(crossing))
                finite = sorted(boundaries)
                intervals: list[tuple[float, float, tuple[str, ...]]] = []
                for index, lower in enumerate(finite):
                    upper = finite[index + 1] if index + 1 < len(finite) else float("inf")
                    probe = (lower + upper) / 2.0 if math.isfinite(upper) else max(1.0, lower * 2.0 + 1.0)
                    reward = savings - probe * harms
                    best = float(reward.max())
                    winners = tuple(sorted(arms[np.isclose(reward, best, atol=1e-10)].tolist()))
                    if intervals and intervals[-1][2] == winners:
                        intervals[-1] = (intervals[-1][0], upper, winners)
                    else:
                        intervals.append((lower, upper, winners))
                for lower, upper, winners in intervals:
                    rows.append({
                        "model": model,
                        "origin_mode": mode,
                        "saving_basis": saving_column,
                        "harm_basis": harm_column,
                        "lambda_low": lower,
                        "lambda_high": upper,
                        "winners": ";".join(winners),
                        "fine_online_wins": any(
                            arm in {"controller", "causal_no_hysteresis"} for arm in winners
                        ),
                        "coarse_only": all(
                            arm in {"uniform_full", "uniform_low", "calib_static"} for arm in winners
                        ),
                    })
    return pd.DataFrame(rows)


def build_paired_bootstrap_winner_probabilities(
    summary: pd.DataFrame,
    policy_frames: dict[tuple[str, str], pd.DataFrame],
    modes: list[str],
    policies: list[str],
    lambdas: list[float],
    cvar_fraction: float,
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    """Estimate winner probabilities using paired, subject-stratified resampling."""
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(seed)
    for mode in modes:
        aligned: dict[str, pd.DataFrame] = {}
        for arm in policies:
            aligned[arm] = policy_frames[(mode, arm)].sort_values("request_id").reset_index(drop=True)
        reference = aligned["uniform_full"]
        request_ids = reference["request_id"].to_numpy(int)
        subjects = reference["subject"].to_numpy(str)
        subject_indices = [np.flatnonzero(subjects == subject) for subject in np.unique(subjects)]
        for arm, frame in aligned.items():
            if not np.array_equal(frame["request_id"].to_numpy(int), request_ids):
                raise RuntimeError(f"paired bootstrap request mismatch for {mode}/{arm}")

        correctness_harm = {
            arm: reference["correct"].to_numpy(float) - frame["correct"].to_numpy(float)
            for arm, frame in aligned.items()
        }
        nll_harm = {
            arm: frame["correct_choice_nll"].to_numpy(float)
            - reference["correct_choice_nll"].to_numpy(float)
            for arm, frame in aligned.items()
        }
        cell = summary[summary["origin_mode"] == mode].set_index("arm")
        for saving_column in SAVING_COLUMNS:
            savings = np.asarray([cell.loc[arm, saving_column] for arm in policies], dtype=np.float64)
            for harm_basis in (
                "incremental_accuracy_harm",
                "cvar_incremental_positive_accuracy_harm",
                "mean_incremental_positive_nll_harm",
                "cvar_incremental_positive_nll_harm",
            ):
                counts = np.zeros((len(lambdas), len(policies)), dtype=np.float64)
                margins: list[list[float]] = [[] for _ in lambdas]
                for _ in range(n_bootstrap):
                    sampled = np.concatenate([
                        indices[rng.integers(0, len(indices), size=len(indices))]
                        for indices in subject_indices
                    ])
                    harms = []
                    for arm in policies:
                        if harm_basis == "incremental_accuracy_harm":
                            harm = max(0.0, float(correctness_harm[arm][sampled].mean()))
                        elif harm_basis == "cvar_incremental_positive_accuracy_harm":
                            harm = cvar(np.clip(correctness_harm[arm][sampled], 0.0, None), cvar_fraction)
                        elif harm_basis == "mean_incremental_positive_nll_harm":
                            harm = float(np.clip(nll_harm[arm][sampled], 0.0, None).mean())
                        else:
                            harm = cvar(np.clip(nll_harm[arm][sampled], 0.0, None), cvar_fraction)
                        harms.append(harm)
                    harms_array = np.asarray(harms, dtype=np.float64)
                    for lambda_index, lam in enumerate(lambdas):
                        reward = savings - lam * harms_array
                        order = np.argsort(reward)
                        best = reward[order[-1]]
                        winners = np.flatnonzero(np.isclose(reward, best, atol=1e-12))
                        counts[lambda_index, winners] += 1.0 / len(winners)
                        second = reward[order[-2]] if len(order) > 1 else best
                        margins[lambda_index].append(float(best - second))
                for lambda_index, lam in enumerate(lambdas):
                    for arm_index, arm in enumerate(policies):
                        rows.append({
                            "model": str(summary["model"].iloc[0]),
                            "origin_mode": mode,
                            "saving_basis": saving_column,
                            "harm_basis": harm_basis,
                            "lambda": lam,
                            "arm": arm,
                            "winner_probability": counts[lambda_index, arm_index] / n_bootstrap,
                            "winner_margin_ci_low": float(np.quantile(margins[lambda_index], 0.025)),
                            "winner_margin_ci_high": float(np.quantile(margins[lambda_index], 0.975)),
                        })
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    subjects = [value.strip() for value in args.subjects.split(",") if value.strip()]
    policies = [value.strip() for value in args.policies.split(",") if value.strip()]
    modes = [value.strip() for value in args.origin_modes.split(",") if value.strip()]
    unknown = set(policies) - set(RECEIVER_ARMS)
    if unknown:
        raise ValueError(f"unknown policies: {sorted(unknown)}")
    if "uniform_full" not in policies:
        raise ValueError("policies must include uniform_full as the incremental quality/reward baseline")
    if set(modes) - {"balanced", "hotspot"}:
        raise ValueError(f"unknown origin modes: {modes}")
    if "hotspot" in modes and args.batch_size < 4:
        print("WARNING: batch-size < 4 gives a weak balanced-vs-hotspot contrast")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    resolve_device()

    print("loading MMLU calibration/test splits...", flush=True)
    calibration_rows = load_mmlu_rows(
        "validation", subjects, args.calib_per_subject, args.seed, args.offline
    )
    test_rows = load_mmlu_rows(
        "test", subjects, args.test_per_subject, args.seed + 1, args.offline
    )
    tokenizer = load_tokenizer(args.model, local_files_only=args.offline)
    tokenizer.padding_side = "right"
    choice_token_ids, choice_prefix = resolve_choice_token_ids(tokenizer)
    calibration_rows, calibration_overlength = filter_overlength_rows(
        tokenizer, calibration_rows, args.seq_len
    )
    test_rows, test_overlength = filter_overlength_rows(tokenizer, test_rows, args.seq_len)
    calibration_tail = len(calibration_rows) % args.batch_size
    test_tail = len(test_rows) % args.batch_size
    calibration_incomplete = [int(row["request_id"]) for row in calibration_rows[-calibration_tail:]] if calibration_tail else []
    test_incomplete = [int(row["request_id"]) for row in test_rows[-test_tail:]] if test_tail else []
    if calibration_tail:
        calibration_rows = calibration_rows[:-calibration_tail]
    if test_tail:
        test_rows = test_rows[:-test_tail]
    if not calibration_rows or not test_rows:
        raise RuntimeError("no complete, non-truncated microbatch remains; increase samples or seq-len")

    fingerprint = stable_run_fingerprint(args, calibration_rows, test_rows)
    fingerprint_path = out / "run_fingerprint.json"
    if args.resume:
        if not fingerprint_path.exists():
            raise RuntimeError("--resume requires an existing run_fingerprint.json")
        previous = json.loads(fingerprint_path.read_text(encoding="utf-8"))
        if previous.get("sha256") != fingerprint["sha256"]:
            raise RuntimeError(
                "resume fingerprint mismatch; use a new output directory or rerun without --resume"
            )
    fingerprint_path.write_text(json.dumps(fingerprint, indent=2, default=str), encoding="utf-8")

    model, load_seconds = load_model(
        args.model, dtype_name=args.dtype, local_files_only=args.offline
    )
    validate_model_spec(model, args.model_key)
    patch_mixtral_moe(model, "full", record_diagnostics=False)
    print(
        f"model loaded in {load_seconds:.1f}s; choice tokens={choice_token_ids}; "
        f"kept calibration={len(calibration_rows)}, test={len(test_rows)}",
        flush=True,
    )

    profile, threshold_high, threshold_low, static_threshold, collector = calibration_pass(
        model, tokenizer, calibration_rows, choice_token_ids, args
    )
    calibration_metadata = {
        "profile": {f"{s}:{r}": value for (s, r), value in sorted(profile.items())},
        "threshold_high": threshold_high,
        "threshold_low": threshold_low,
        "static_threshold": static_threshold,
        "observed_steps": collector.observed_steps,
        "high_quantile": args.high_quantile,
        "gap_ratio": args.gap_ratio,
        "alpha": args.alpha,
        "dwell_min": args.dwell_min,
    }
    (out / "calibration.json").write_text(json.dumps(calibration_metadata, indent=2), encoding="utf-8")
    pd.DataFrame(collector.step_rows).to_csv(out / "calibration_steps.csv", index=False)

    reference_path = out / "bf16_reference_per_question.csv"
    if args.resume and reference_path.exists():
        reference_df = pd.read_csv(reference_path)
    else:
        reference = ReceiverLaneController(ReceiverPolicyConfig(
            arm=REFERENCE_ARM,
            ep_size=args.ep_size,
            gpus_per_node=args.gpus_per_node,
            placement=args.placement,
            high_precision=args.high_precision,
            low_precision=args.low_precision,
            high_scale_bytes_per_vector=args.high_scale_bytes,
            low_scale_bytes_per_vector=args.low_scale_bytes,
            inter_node_gbps=args.inter_node_gbps,
            codec_pack_us=args.codec_pack_us,
            codec_unpack_us=args.codec_unpack_us,
            codec_tile_rows=args.codec_tile_rows,
            codec_tax_mode=args.codec_tax_mode,
        ))
        reference_df = run_rows(
            model,
            tokenizer,
            reference,
            test_rows,
            "balanced",
            choice_token_ids,
            args,
            out / "bf16_reference.partial.csv",
        )
        reference_df.to_csv(reference_path, index=False)
        pd.DataFrame(reference.step_rows).to_csv(out / "bf16_reference_steps.csv", index=False)

    policy_frames: dict[tuple[str, str], pd.DataFrame] = {}
    step_frames: dict[tuple[str, str], pd.DataFrame] = {}
    for mode in modes:
        for arm in policies:
            policy_path = out / f"{arm}_{mode}_per_question.csv"
            steps_path = out / f"{arm}_{mode}_steps.csv"
            if args.resume and policy_path.exists():
                if not steps_path.exists():
                    raise RuntimeError(f"resume found {policy_path} but missing action trace {steps_path}")
                policy_df = pd.read_csv(policy_path)
                step_df = pd.read_csv(steps_path)
            else:
                config = ReceiverPolicyConfig(
                    arm=arm,
                    ep_size=args.ep_size,
                    gpus_per_node=args.gpus_per_node,
                    placement=args.placement,
                    high_precision=args.high_precision,
                    low_precision=args.low_precision,
                    alpha=args.alpha,
                    threshold_high=threshold_high,
                    threshold_low=threshold_low,
                    dwell_min=args.dwell_min,
                    static_profile=profile,
                    static_threshold=static_threshold,
                    high_scale_bytes_per_vector=args.high_scale_bytes,
                    low_scale_bytes_per_vector=args.low_scale_bytes,
                    inter_node_gbps=args.inter_node_gbps,
                    codec_pack_us=args.codec_pack_us,
                    codec_unpack_us=args.codec_unpack_us,
                    codec_tile_rows=args.codec_tile_rows,
                    codec_tax_mode=args.codec_tax_mode,
                )
                controller = ReceiverLaneController(config)
                policy_df = run_rows(
                    model,
                    tokenizer,
                    controller,
                    test_rows,
                    mode,
                    choice_token_ids,
                    args,
                    out / f"{arm}_{mode}.partial.csv",
                )
                policy_df.to_csv(policy_path, index=False)
                step_df = pd.DataFrame(controller.step_rows)
                step_df.to_csv(steps_path, index=False)
            policy_frames[(mode, arm)] = policy_df
            step_frames[(mode, arm)] = step_df
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    summaries: list[dict[str, object]] = []
    for mode in modes:
        uniform_full_df = policy_frames[(mode, "uniform_full")]
        for arm in policies:
            summaries.append(summarize_policy(
                policy_frames[(mode, arm)],
                reference_df,
                uniform_full_df,
                step_frames[(mode, arm)],
                arm,
                mode,
                args,
            ))

    summary = pd.DataFrame(summaries)
    summary.to_csv(out / "policy_task_quality_summary.csv", index=False)
    lambdas = [float(value) for value in args.lambda_grid.split(",") if value.strip()]
    lambda_winners = build_lambda_winners(summary, lambdas)
    lambda_winners.to_csv(out / "task_harm_lambda_winners.csv", index=False)
    lambda_intervals = build_exact_lambda_intervals(summary)
    lambda_intervals.to_csv(out / "task_harm_lambda_exact_intervals.csv", index=False)
    bootstrap_winners = build_paired_bootstrap_winner_probabilities(
        summary,
        policy_frames,
        modes,
        policies,
        lambdas,
        args.cvar_fraction,
        args.n_bootstrap,
        args.seed + 101,
    )
    bootstrap_winners.to_csv(
        out / "task_harm_lambda_bootstrap_winner_probabilities.csv", index=False
    )

    execution_device = str(next(model.parameters()).device)
    task_quality_hardware = (
        "cuda_gpu" if execution_device.startswith("cuda")
        else "mps_gpu" if execution_device.startswith("mps")
        else "cpu"
    )
    metadata = {
        "config": vars(args),
        "model_load_seconds": load_seconds,
        "execution_device": execution_device,
        "task_quality_hardware": task_quality_hardware,
        "choice_token_ids": choice_token_ids,
        "choice_prefix": choice_prefix,
        "calibration": calibration_metadata,
        "n_calibration_questions": len(calibration_rows),
        "n_test_questions": len(test_rows),
        "excluded_questions": {
            "calibration_overlength": calibration_overlength,
            "test_overlength": test_overlength,
            "calibration_incomplete_batch": calibration_incomplete,
            "test_incomplete_batch": test_incomplete,
        },
        "evidence_boundary": (
            f"Task accuracy/NLL/CVaR were measured on {task_quality_hardware} under actual "
            "policy-produced combine-output fake quantization. Receiver topology and request "
            "concurrency are emulated inside one process. Saving is computed from the exact "
            "same action trace with an analytic bottleneck-wire model plus codec tax measured "
            "separately on RTX 5090; it is not real NCCL/RDMA latency. Local pairs stay BF16; "
            "remote high/low pairs use the configured fake-quant formats."
        ),
        "causal_invariants": [
            "controller and causal threshold read only loads observed through the previous MoE step",
            "threshold quantiles use positive receiver loads, exactly matching the frozen direct-benefit controller calibration definition",
            "calibration profile and thresholds use validation questions only",
            "padding tokens do not contribute to lane load, exposure, or quantization",
            "each arm runs from a fresh controller state on the same frozen test order",
            "routing is not locked: downstream routing drift caused by each action is preserved",
        ],
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")

    report_table = summary.to_csv(index=False).strip()
    report = [
        f"# Receiver-Aware Task-Quality Gate ({args.model_key})",
        "",
        "```csv",
        report_table,
        "```",
        "",
        f"The task-quality columns were measured on {task_quality_hardware}. `simulated_saving_fraction` is computed from the same policy action trace with an analytic bottleneck-wire model and separately measured codec tax; it must not be cited as real NCCL/RDMA latency.",
    ]
    (out / "report.md").write_text("\n".join(report), encoding="utf-8")
    print(summary.to_string(index=False))
    if len(lambda_winners):
        print("\nTask-harm lambda winners (selected codec model):")
        selected = lambda_winners[
            lambda_winners["saving_basis"] == "simulated_saving_fraction"
        ]
        print(selected[["origin_mode", "harm_basis", "lambda", "winner", "winner_reward"]].to_string(index=False))
    if len(lambda_intervals):
        fine = lambda_intervals[
            (lambda_intervals["saving_basis"] == "simulated_saving_fraction")
            & lambda_intervals["fine_online_wins"]
        ]
        print("\nExact selected-codec intervals where a fine online arm wins:")
        print(fine.to_string(index=False) if len(fine) else "NONE")
    print(f"\nsaved to {out}")


if __name__ == "__main__":
    main()
