"""Held-out signal comparison for FP8-first selective low-bit combine.

This script closes three Mac-side evidence gaps:
  1. calibration and test use disjoint prompt slices;
  2. quality uses corpus PPL and per-token KL;
  3. fixed-rank is compared with calibrated gate-threshold, cumulative tail-mass,
     and contribution-oracle selectors at approximately matched payload.

It remains a fake-quant quality experiment, not a communication benchmark.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from capture_moe import patch_mixtral_moe
from metrics import MetricAccumulator
from modeling import load_model, load_tokenizer
from prompts import get_prompts


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="allenai/OLMoE-1B-7B-0924")
    p.add_argument("--dataset", default="wikitext2_docs")
    p.add_argument(
        "--dataset-split",
        default=None,
        help="Deprecated compatibility option: use one split for both calibration and test.",
    )
    p.add_argument("--calibration-split", default="validation")
    p.add_argument("--test-split", default="test")
    p.add_argument("--calibration-samples", type=int, default=16)
    p.add_argument("--test-samples", type=int, default=32)
    p.add_argument("--calibration-offset", type=int, default=0)
    p.add_argument("--test-offset", type=int, default=0)
    p.add_argument("--dataset-seed", type=int, default=None)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--num-receiver-groups", type=int, default=4)
    p.add_argument(
        "--tail-precision",
        default="int4",
        choices=["int4", "mxfp4", "nvfp4"],
    )
    p.add_argument("--bootstrap", type=int, default=5000)
    p.add_argument(
        "--minimal-baselines",
        action="store_true",
        help=(
            "Run only uniform FP8, uniform low-bit, fixed-rank, and calibrated "
            "gate baselines before requested block policies."
        ),
    )
    p.add_argument(
        "--block-sizes",
        type=int,
        nargs="*",
        default=[],
        help=(
            "Optional contiguous token-block sizes for fixed-rate gate allocation. "
            "Conditional on the routed-pair count, each block assigns exactly half "
            "to FP8 and half to tail precision."
        ),
    )
    p.add_argument(
        "--block-score-modes",
        nargs="*",
        default=[],
        choices=["contrib", "qenergy", "qerr", "qbenefit", "random", "reversegate"],
        help="Additional fixed-rate selectors evaluated at every --block-sizes value.",
    )
    p.add_argument(
        "--residual-block-sizes",
        type=int,
        nargs="*",
        default=[],
        help=(
            "Optional block sizes for progressive low-bit base plus fixed-budget "
            "low-bit residual refinement on the critical half."
        ),
    )
    p.add_argument(
        "--residual-score-modes",
        nargs="*",
        default=[],
        choices=["contrib", "resenergy", "reserr", "resbenefit", "random", "reversegate"],
        help="Additional residual-refinement selectors at every residual block size.",
    )
    p.add_argument(
        "--matched-direct-block-sizes",
        type=int,
        nargs="*",
        default=[],
        help=(
            "Optional direct FP8/low-bit block policies whose low-bit fraction "
            "is chosen to match the wire bytes of a 50-percent residual policy."
        ),
    )
    p.add_argument(
        "--matched-direct-score-modes",
        nargs="*",
        default=[],
        choices=["contrib", "qenergy", "qerr", "qbenefit", "random", "reversegate"],
        help="Additional direct selectors at the residual-matched physical byte rate.",
    )
    p.add_argument(
        "--peer-block-pairs",
        type=int,
        nargs="*",
        default=[],
        help="Optional fixed pair-tile sizes selected independently per receiver group.",
    )
    p.add_argument(
        "--peer-block-score-modes",
        nargs="*",
        default=[],
        choices=["contrib", "qenergy", "qerr", "qbenefit", "random", "reversegate"],
        help="Additional selectors for every owner-group pair tile size.",
    )
    p.add_argument(
        "--peer-residual-block-pairs",
        type=int,
        nargs="*",
        default=[],
        help="Peer-local pair-tile sizes for low-bit base plus residual refinement.",
    )
    p.add_argument(
        "--peer-residual-score-modes",
        nargs="*",
        default=[],
        choices=["contrib", "resenergy", "reserr", "resbenefit", "random", "reversegate"],
        help="Additional residual selectors for every owner-group pair tile size.",
    )
    p.add_argument(
        "--peer-matched-direct-block-pairs",
        type=int,
        nargs="*",
        default=[],
        help="Owner-group direct pair tiles at the residual-matched byte rate.",
    )
    p.add_argument(
        "--peer-matched-direct-score-modes",
        nargs="*",
        default=[],
        choices=["contrib", "qenergy", "qerr", "qbenefit", "random", "reversegate"],
        help="Additional owner-group selectors at the residual-matched byte rate.",
    )
    p.add_argument("--offline", action="store_true")
    p.add_argument(
        "--output-dir",
        default="experiments/idea_a_mac/outputs/paper_validation/signal_comparison",
    )
    return p.parse_args()


def tokenized_inputs(tokenizer, texts: list[str], seq_len: int):
    for text in texts:
        yield tokenizer(text, return_tensors="pt", truncation=True, max_length=seq_len)


def encode_policy_float(value: float) -> str:
    return f"{value:.6f}".replace(".", "p")


def vector_wire_bytes(precision: str, hidden_size: int) -> int:
    """Payload plus scale bytes for one expert-output vector."""
    if precision in ("full", "bf16"):
        return 2 * hidden_size
    if precision == "fp8":
        # ``fp8_e4m3_quant_dequant`` uses one FP32 absmax scale per expert-
        # output vector.  Keep wire accounting aligned with that numerical
        # experiment; a block-scaled FP8 variant must be implemented and named
        # separately rather than borrowing MX-style metadata here.
        return hidden_size + 4
    if precision == "int4":
        return math.ceil(hidden_size / 2) + 4
    if precision == "mxfp4":
        return math.ceil(hidden_size / 2) + math.ceil(hidden_size / 32)
    if precision == "nvfp4":
        return math.ceil(hidden_size / 2) + math.ceil(hidden_size / 16) + 4
    raise ValueError(f"unknown precision: {precision}")


def metadata_aware_saving(raw_saving: float, precision: str, hidden_size: int) -> float:
    """Convert recorder bit-only saving to a format-aware wire-byte estimate."""
    bf16_bytes = vector_wire_bytes("bf16", hidden_size)
    if precision == "full":
        return 0.0
    if precision == "fp8":
        return 1.0 - vector_wire_bytes("fp8", hidden_size) / bf16_bytes
    # Every selective policy starts from FP8 and moves an observed fraction to
    # a 4-bit format. Recorder accounting gives 0.50 + 0.25 * fraction.
    low_bit_fraction = min(1.0, max(0.0, (raw_saving - 0.5) / 0.25))
    actual = (
        (1.0 - low_bit_fraction) * vector_wire_bytes("fp8", hidden_size)
        + low_bit_fraction * vector_wire_bytes(precision, hidden_size)
    )
    return 1.0 - actual / bf16_bytes


def residual_metadata_aware_saving(precision: str, hidden_size: int) -> float:
    """Wire saving for one low-bit base on all pairs plus residual on half."""
    bf16_bytes = vector_wire_bytes("bf16", hidden_size)
    low_bit_vector = vector_wire_bytes(precision, hidden_size)
    actual = 1.5 * low_bit_vector
    return 1.0 - actual / bf16_bytes


def build_global_lut(
    precisions: list[str], num_layers: int, num_groups: int
) -> dict[tuple[int, int, int], str]:
    return {
        (layer, group, rank): precisions[rank - 1]
        for layer in range(num_layers)
        for group in range(num_groups)
        for rank in range(1, len(precisions) + 1)
    }


def dataframe_to_markdown(df: pd.DataFrame, columns: list[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for _, row in df[columns].iterrows():
        values = []
        for column in columns:
            value = row[column]
            values.append(f"{value:.6f}" if isinstance(value, (float, np.floating)) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _moe_modules(model) -> list[object]:
    modules = []
    for layer in model.model.layers:
        if hasattr(layer, "block_sparse_moe"):
            modules.append(layer.block_sparse_moe)
        else:
            modules.append(layer.mlp)
    return modules


def validate_exact_full_path(model, tokenizer, text: str, seq_len: int) -> dict[str, float]:
    """Fail fast unless the patched full path is exactly the pretrained path."""
    modules = _moe_modules(model)
    original_forwards = [module.forward for module in modules]
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=seq_len)
    with torch.no_grad():
        original = model(**inputs).logits.detach().cpu()
    patch_mixtral_moe(model, "full", num_receiver_groups=1)
    with torch.no_grad():
        patched = model(**inputs).logits.detach().cpu()
    for module, original_forward in zip(modules, original_forwards):
        module.forward = original_forward
    diff = (patched.float() - original.float()).abs()
    result = {
        "max_abs_logit_diff": float(diff.max().item()),
        "mean_abs_logit_diff": float(diff.mean().item()),
    }
    if not torch.equal(original, patched):
        raise RuntimeError(f"patched full path is not exact: {result}")
    return result


def data_manifest(tokenizer, texts: list[str], split: str, seq_len: int) -> list[dict]:
    rows = []
    for sample_id, text in enumerate(texts):
        token_count = len(tokenizer(text, add_special_tokens=True)["input_ids"])
        rows.append(
            {
                "sample_id": sample_id,
                "split": split,
                "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "title_prefix": text[:120],
                "characters": len(text),
                "tokens_before_truncation": token_count,
                "tokens_used": min(token_count, seq_len),
            }
        )
    return rows


def source_manifest() -> dict[str, str]:
    """Hash the exact local experiment sources when no repository commit exists."""
    root = Path(__file__).resolve().parent
    names = (
        "run_signal_comparison.py",
        "capture_moe.py",
        "policies.py",
        "fake_quant.py",
        "metrics.py",
        "modeling.py",
        "prompts.py",
    )
    return {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in names
    }


def paired_bootstrap_delta(
    candidate: MetricAccumulator,
    reference: MetricAccumulator,
    n_bootstrap: int,
    seed: int = 20260713,
) -> dict[str, float]:
    if len(candidate.samples) != len(reference.samples):
        raise ValueError("paired samples differ")
    n = len(candidate.samples)

    def corpus(rows, indices):
        tokens = sum(rows[i].token_count for i in indices)
        nll = sum(rows[i].nll_sum for i in indices) / max(tokens, 1)
        kl = sum(rows[i].kl_sum for i in indices) / max(tokens, 1)
        return math.exp(nll), kl

    all_idx = np.arange(n)
    cand_ppl, cand_kl = corpus(candidate.samples, all_idx)
    ref_ppl, ref_kl = corpus(reference.samples, all_idx)
    point_ppl = cand_ppl - ref_ppl
    point_kl = cand_kl - ref_kl
    if n < 2 or n_bootstrap <= 0:
        return {
            "paired_ppl_delta": point_ppl,
            "paired_ppl_ci_low": point_ppl,
            "paired_ppl_ci_high": point_ppl,
            "paired_kl_delta": point_kl,
            "paired_kl_ci_low": point_kl,
            "paired_kl_ci_high": point_kl,
        }
    rng = np.random.default_rng(seed)
    ppl_values = np.empty(n_bootstrap, dtype=np.float64)
    kl_values = np.empty(n_bootstrap, dtype=np.float64)
    for bootstrap_idx in range(n_bootstrap):
        chosen = rng.integers(0, n, size=n)
        cand_ppl, cand_kl = corpus(candidate.samples, chosen)
        ref_ppl, ref_kl = corpus(reference.samples, chosen)
        ppl_values[bootstrap_idx] = cand_ppl - ref_ppl
        kl_values[bootstrap_idx] = cand_kl - ref_kl
    return {
        "paired_ppl_delta": point_ppl,
        "paired_ppl_ci_low": float(np.quantile(ppl_values, 0.025)),
        "paired_ppl_ci_high": float(np.quantile(ppl_values, 0.975)),
        "paired_kl_delta": point_kl,
        "paired_kl_ci_low": float(np.quantile(kl_values, 0.025)),
        "paired_kl_ci_high": float(np.quantile(kl_values, 0.975)),
    }


def run_baseline(
    model,
    tokenizer,
    texts: list[str],
    seq_len: int,
    num_groups: int,
    record_routes: bool,
    sample_id_base: int,
) -> tuple[MetricAccumulator, list[torch.Tensor], object]:
    recorder = patch_mixtral_moe(
        model,
        "full",
        num_receiver_groups=num_groups,
        record_routes=record_routes,
    )
    metrics = MetricAccumulator()
    logits_by_sample: list[torch.Tensor] = []
    for local_idx, inputs in enumerate(tokenized_inputs(tokenizer, texts, seq_len)):
        sample_id = sample_id_base + local_idx
        recorder.set_sample_id(sample_id)
        with torch.no_grad():
            logits = model(**inputs).logits.detach().cpu()
        metrics.add(
            sample_id,
            logits,
            inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
        )
        logits_by_sample.append(logits)
        print(f"  full {local_idx + 1}/{len(texts)}", flush=True)
    return metrics, logits_by_sample, recorder


def run_strategy(
    model,
    tokenizer,
    texts: list[str],
    seq_len: int,
    baseline_logits: list[torch.Tensor],
    strategy_name: str,
    policy_name: str,
    num_groups: int,
    sample_id_base: int,
    lut: dict[tuple[int, int, int], str] | None = None,
) -> tuple[MetricAccumulator, object]:
    recorder = patch_mixtral_moe(
        model,
        policy_name,
        num_receiver_groups=num_groups,
        lut=lut,
    )
    metrics = MetricAccumulator()
    for local_idx, inputs in enumerate(tokenized_inputs(tokenizer, texts, seq_len)):
        sample_id = sample_id_base + local_idx
        recorder.set_sample_id(sample_id)
        with torch.no_grad():
            logits = model(**inputs).logits.detach().cpu()
        metrics.add(
            sample_id,
            logits,
            inputs["input_ids"],
            baseline_logits=baseline_logits[local_idx],
            attention_mask=inputs.get("attention_mask"),
        )
        print(f"  {strategy_name} {local_idx + 1}/{len(texts)}", flush=True)
    return metrics, recorder


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    calibration_split = args.dataset_split or args.calibration_split
    test_split = args.dataset_split or args.test_split
    calibration_texts = get_prompts(
        args.dataset,
        args.calibration_samples,
        offset=args.calibration_offset,
        split=calibration_split,
        seed=args.dataset_seed,
    )
    test_texts = get_prompts(
        args.dataset,
        args.test_samples,
        offset=args.test_offset,
        split=test_split,
        seed=args.dataset_seed,
    )
    if set(calibration_texts) & set(test_texts):
        raise RuntimeError("calibration and test prompt text overlap")

    tokenizer = load_tokenizer(args.model, local_files_only=args.offline)
    model, load_seconds = load_model(
        args.model,
        dtype_name=args.dtype,
        local_files_only=args.offline,
    )
    baseline_equivalence = validate_exact_full_path(
        model, tokenizer, calibration_texts[0], args.seq_len
    )
    num_layers = len(model.model.layers)
    top_k = int(getattr(model.config, "num_experts_per_tok", 8))
    hidden_size = int(model.config.hidden_size)
    fp8_vector_bytes = vector_wire_bytes("fp8", hidden_size)
    low_vector_bytes = vector_wire_bytes(args.tail_precision, hidden_size)
    matched_direct_low_fraction = (
        (fp8_vector_bytes - 1.5 * low_vector_bytes)
        / max(fp8_vector_bytes - low_vector_bytes, 1)
    )
    matched_direct_low_fraction = min(1.0, max(0.0, matched_direct_low_fraction))
    matched_direct_fraction_code = int(round(1000 * matched_direct_low_fraction))
    n_tail = max(1, top_k // 2)
    target_low_bit_fraction = n_tail / top_k
    print(
        f"model loaded in {load_seconds:.1f}s; layers={num_layers}, top_k={top_k}, "
        f"target_low_bit_fraction={target_low_bit_fraction:.3f}",
        flush=True,
    )

    print("calibrating routing thresholds...", flush=True)
    _, _, calibration_recorder = run_baseline(
        model,
        tokenizer,
        calibration_texts,
        args.seq_len,
        args.num_receiver_groups,
        record_routes=True,
        sample_id_base=args.calibration_offset,
    )
    calibration_routes = pd.DataFrame(calibration_recorder.route_rows())
    calibration_routes.to_csv(out / "calibration_routes.csv", index=False)
    weights = calibration_recorder.routing_weights_tensor().float()
    if weights.numel() == 0:
        raise RuntimeError("no routing weights captured during calibration")

    gate_threshold = float(torch.quantile(weights.flatten(), target_low_bit_fraction).item())
    matched_gate_threshold = float(
        torch.quantile(weights.flatten(), matched_direct_low_fraction).item()
    )
    suffix_mass = weights.flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])
    gate_tailmass = float(torch.quantile(suffix_mass.flatten(), target_low_bit_fraction).item())
    threshold_fraction = float((weights <= gate_threshold).float().mean().item())
    matched_threshold_fraction = float(
        (weights <= matched_gate_threshold).float().mean().item()
    )
    tailmass_fraction = float((suffix_mass <= gate_tailmass).float().mean().item())

    calibration = {
        "model": args.model,
        "dataset": args.dataset,
        "calibration_split": calibration_split,
        "test_split": test_split,
        "calibration_offset": args.calibration_offset,
        "calibration_samples": args.calibration_samples,
        "test_offset": args.test_offset,
        "test_samples": args.test_samples,
        "seq_len": args.seq_len,
        "top_k": top_k,
        "n_tail": n_tail,
        "target_low_bit_fraction": target_low_bit_fraction,
        "tail_precision": args.tail_precision,
        "gate_threshold": gate_threshold,
        "gate_threshold_calibration_fraction": threshold_fraction,
        "matched_gate_threshold": matched_gate_threshold,
        "matched_gate_threshold_calibration_fraction": matched_threshold_fraction,
        "gate_tailmass": gate_tailmass,
        "gate_tailmass_calibration_fraction": tailmass_fraction,
        "residual_matched_direct_low_fraction": matched_direct_low_fraction,
        "residual_matched_direct_fraction_code": matched_direct_fraction_code,
    }
    (out / "calibration.json").write_text(json.dumps(calibration, indent=2), encoding="utf-8")
    config = vars(args).copy()
    config.update(
        {
            "resolved_calibration_split": calibration_split,
            "resolved_test_split": test_split,
            "model_revision": getattr(model.config, "_commit_hash", None),
            "runtime_versions": {
                package: importlib.metadata.version(package)
                for package in ("torch", "transformers", "datasets", "pandas", "numpy")
            },
            "source_sha256": source_manifest(),
            "baseline_equivalence": baseline_equivalence,
            "boundary": (
                "article-level fake-quant quality and metadata-aware logical bytes; "
                "no native FP4 kernel, all-to-all, RDMA, TTFT, TBT, or P99"
            ),
        }
    )
    (out / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    manifest = {
        "calibration": data_manifest(
            tokenizer, calibration_texts, calibration_split, args.seq_len
        ),
        "test": data_manifest(tokenizer, test_texts, test_split, args.seq_len),
    }
    (out / "data_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(calibration, flush=True)

    print("running held-out full baseline...", flush=True)
    full_metrics, baseline_logits, test_recorder = run_baseline(
        model,
        tokenizer,
        test_texts,
        args.seq_len,
        args.num_receiver_groups,
        record_routes=True,
        sample_id_base=args.test_offset,
    )
    pd.DataFrame(test_recorder.route_rows()).to_csv(out / "test_routes.csv", index=False)

    head_count = top_k - n_tail
    gate_name = (
        f"gate_threshold_{encode_policy_float(gate_threshold)}_{args.tail_precision}"
    )
    matched_gate_name = (
        f"gate_threshold_{encode_policy_float(matched_gate_threshold)}_{args.tail_precision}"
    )
    mass_name = (
        f"gate_tailmass_{encode_policy_float(gate_tailmass)}_{args.tail_precision}"
    )
    head_lut = build_global_lut(
        [args.tail_precision] * n_tail + ["fp8"] * (top_k - n_tail),
        num_layers,
        args.num_receiver_groups,
    )
    interleaved_lut = build_global_lut(
        [args.tail_precision if rank % 2 == 0 else "fp8" for rank in range(top_k)],
        num_layers,
        args.num_receiver_groups,
    )
    strategies: list[tuple[str, str, dict | None]] = [
        ("uniform_fp8", "uniform_fp8", None),
        (f"uniform_{args.tail_precision}", f"uniform_{args.tail_precision}", None),
        (
            f"rank_tail{n_tail}_{args.tail_precision}",
            f"fp8top{head_count}_rest_{args.tail_precision}",
            None,
        ),
        (f"gate_threshold_{args.tail_precision}", gate_name, None),
    ]
    if not args.minimal_baselines:
        strategies.extend(
            [
                (f"gate_tailmass_{args.tail_precision}", mass_name, None),
                (
                    f"contribution_tail{n_tail}_{args.tail_precision}_oracle",
                    f"contrib_tail{n_tail}_{args.tail_precision}",
                    None,
                ),
                (f"head{n_tail}_{args.tail_precision}_control", "lut", head_lut),
                (
                    f"interleaved{n_tail}_{args.tail_precision}_control",
                    "lut",
                    interleaved_lut,
                ),
            ]
        )
    for block_size in args.block_sizes:
        if block_size < 1:
            raise ValueError(f"block size must be positive, got {block_size}")
        strategies.append(
            (
                f"block_gate{block_size}_{args.tail_precision}",
                f"block_gate{block_size}_{args.tail_precision}",
                None,
            )
        )
        for score_mode in args.block_score_modes:
            strategies.append(
                (
                    f"block_{score_mode}{block_size}_{args.tail_precision}",
                    f"block_{score_mode}{block_size}_{args.tail_precision}",
                    None,
                )
            )
    for block_size in args.residual_block_sizes:
        if block_size < 1:
            raise ValueError(f"residual block size must be positive, got {block_size}")
        strategies.append(
            (
                f"block_gate{block_size}_residual_{args.tail_precision}",
                f"block_gate{block_size}_residual_{args.tail_precision}",
                None,
            )
        )
        for score_mode in args.residual_score_modes:
            strategies.append(
                (
                    f"block_{score_mode}{block_size}_residual_{args.tail_precision}",
                    f"block_{score_mode}{block_size}_residual_{args.tail_precision}",
                    None,
                )
            )
    for block_size in args.matched_direct_block_sizes:
        if block_size < 1:
            raise ValueError(f"matched direct block size must be positive, got {block_size}")
        strategies.append(
            (
                f"block_gate{block_size}_f{matched_direct_fraction_code}_{args.tail_precision}",
                f"block_gate{block_size}_f{matched_direct_fraction_code}_{args.tail_precision}",
                None,
            )
        )
        for score_mode in args.matched_direct_score_modes:
            strategies.append(
                (
                    f"block_{score_mode}{block_size}_f{matched_direct_fraction_code}_{args.tail_precision}",
                    f"block_{score_mode}{block_size}_f{matched_direct_fraction_code}_{args.tail_precision}",
                    None,
                )
            )
    if args.matched_direct_block_sizes:
        strategies.append(
            (
                f"gate_threshold_matchedwire_{args.tail_precision}",
                matched_gate_name,
                None,
            )
        )
    for block_pairs in args.peer_block_pairs:
        if block_pairs < 1:
            raise ValueError(f"peer block pairs must be positive, got {block_pairs}")
        strategies.append(
            (
                f"peerblock_gate{block_pairs}_{args.tail_precision}",
                f"peerblock_gate{block_pairs}_{args.tail_precision}",
                None,
            )
        )
        for score_mode in args.peer_block_score_modes:
            strategies.append(
                (
                    f"peerblock_{score_mode}{block_pairs}_{args.tail_precision}",
                    f"peerblock_{score_mode}{block_pairs}_{args.tail_precision}",
                    None,
                )
            )
    for block_pairs in args.peer_residual_block_pairs:
        if block_pairs < 1:
            raise ValueError(f"peer residual block pairs must be positive, got {block_pairs}")
        strategies.append(
            (
                f"peerblock_gate{block_pairs}_residual_{args.tail_precision}",
                f"peerblock_gate{block_pairs}_residual_{args.tail_precision}",
                None,
            )
        )
        for score_mode in args.peer_residual_score_modes:
            strategies.append(
                (
                    f"peerblock_{score_mode}{block_pairs}_residual_{args.tail_precision}",
                    f"peerblock_{score_mode}{block_pairs}_residual_{args.tail_precision}",
                    None,
                )
            )
    for block_pairs in args.peer_matched_direct_block_pairs:
        if block_pairs < 1:
            raise ValueError(f"peer matched direct block pairs must be positive, got {block_pairs}")
        strategies.append(
            (
                f"peerblock_gate{block_pairs}_f{matched_direct_fraction_code}_{args.tail_precision}",
                f"peerblock_gate{block_pairs}_f{matched_direct_fraction_code}_{args.tail_precision}",
                None,
            )
        )
        for score_mode in args.peer_matched_direct_score_modes:
            strategies.append(
                (
                    f"peerblock_{score_mode}{block_pairs}_f{matched_direct_fraction_code}_{args.tail_precision}",
                    f"peerblock_{score_mode}{block_pairs}_f{matched_direct_fraction_code}_{args.tail_precision}",
                    None,
                )
            )

    summary_rows: list[dict[str, float | int | str]] = []
    layer_error_rows: list[dict[str, float | int | str]] = []
    metrics_by_name: dict[str, MetricAccumulator] = {"full": full_metrics}
    sample_rows = full_metrics.sample_rows("full")
    full_summary = full_metrics.bootstrap_summary(args.bootstrap)
    full_summary.update(
        {
            "strategy": "full",
            "policy_name": "full",
            "theoretical_payload_saving_vs_bf16": 0.0,
            "metadata_aware_wire_saving_vs_bf16": 0.0,
            "ppl_delta_vs_full": 0.0,
        }
    )
    summary_rows.append(full_summary)

    for strategy_name, policy_name, lut in strategies:
        print(f"running {strategy_name} ({policy_name})...", flush=True)
        metrics, recorder = run_strategy(
            model,
            tokenizer,
            test_texts,
            args.seq_len,
            baseline_logits,
            strategy_name,
            policy_name,
            args.num_receiver_groups,
            args.test_offset,
            lut=lut,
        )
        row = metrics.bootstrap_summary(args.bootstrap)
        raw_saving = recorder.total_byte_saving()
        format_name = "fp8" if strategy_name == "uniform_fp8" else args.tail_precision
        if "_residual_" in strategy_name:
            wire_saving = residual_metadata_aware_saving(format_name, hidden_size)
        else:
            wire_saving = metadata_aware_saving(raw_saving, format_name, hidden_size)
        row.update(
            {
                "strategy": strategy_name,
                "policy_name": policy_name,
                "theoretical_payload_saving_vs_bf16": raw_saving,
                "metadata_aware_wire_saving_vs_bf16": wire_saving,
                "ppl_delta_vs_full": metrics.corpus_ppl - full_metrics.corpus_ppl,
            }
        )
        summary_rows.append(row)
        metrics_by_name[strategy_name] = metrics
        sample_rows.extend(metrics.sample_rows(strategy_name))
        for error_row in recorder.error_rows():
            error_row["strategy"] = strategy_name
            layer_error_rows.append(error_row)
        pd.DataFrame(summary_rows).to_csv(out / "signal_comparison.partial.csv", index=False)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out / "signal_comparison.csv", index=False)
    pd.DataFrame(sample_rows).to_csv(out / "sample_metrics.csv", index=False)
    pd.DataFrame(layer_error_rows).to_csv(out / "layer_local_error.csv", index=False)

    fixed_name = f"rank_tail{n_tail}_{args.tail_precision}"
    paired_rows = []
    for candidate_name, candidate_metrics in metrics_by_name.items():
        if candidate_name == "full":
            continue
        references = ["full"] if candidate_name == "uniform_fp8" else ["uniform_fp8"]
        if candidate_name not in ("uniform_fp8", fixed_name):
            references.append(fixed_name)
        for reference_name in references:
            paired = paired_bootstrap_delta(
                candidate_metrics,
                metrics_by_name[reference_name],
                args.bootstrap,
            )
            paired.update(
                {"candidate": candidate_name, "reference": reference_name}
            )
            paired_rows.append(paired)
    pd.DataFrame(paired_rows).to_csv(out / "paired_comparisons.csv", index=False)

    columns = [
        "strategy",
        "theoretical_payload_saving_vs_bf16",
        "metadata_aware_wire_saving_vs_bf16",
        "corpus_ppl",
        "ppl_delta_vs_full",
        "mean_token_kl",
        "corpus_ppl_ci_low",
        "corpus_ppl_ci_high",
    ]
    table = dataframe_to_markdown(summary_df, columns)
    report = f"""# Held-Out Signal Comparison

This is a Mac fake-quant quality experiment. It does not measure all-to-all or TPOT.

## Split

- calibration: `{args.dataset}:{calibration_split}` offset `{args.calibration_offset}`, n=`{args.calibration_samples}`
- test: `{args.dataset}:{test_split}` offset `{args.test_offset}`, n=`{args.test_samples}`
- sequence length: `{args.seq_len}`
- target low-bit pair fraction: `{target_low_bit_fraction:.4f}`
- residual metadata-matched direct low-bit fraction: `{matched_direct_low_fraction:.4f}` (encoded `{matched_direct_fraction_code}`)
- tail precision: `{args.tail_precision}`
- calibrated gate threshold: `{gate_threshold:.6f}`
- metadata-matched calibrated gate threshold: `{matched_gate_threshold:.6f}`
- calibrated cumulative tail-mass budget: `{gate_tailmass:.6f}`

## Results

{table}

## Interpretation boundary

- `theoretical_payload_saving_vs_bf16` is bit-only. `metadata_aware_wire_saving_vs_bf16` includes format scale bytes, but still excludes padding, alignment, collective headers, and pack/unpack overhead.
- `contribution_*_oracle` uses expert-output norm after computation and is not a deployable early decision.
- `interleaved*_control` is a fixed rank-independent-pattern anti-control, not a per-token random policy.
- `block_*` is a quality-side fixed-rate proxy: conditional on routed-pair count it fixes the FP8/low-bit composition per contiguous token block, but it neither fixes total message volume nor implements peer-specific packing or a communication kernel.
- `qenergy`/`resenergy` use owner-local low-bit error energy without gate metadata. `qerr`/`reserr` multiply it by gate squared. `qbenefit`/`resbenefit` additionally quantize the alternative representation for all pairs and are expensive score upper bounds, not deployment-ready selectors.
- `random` is a deterministic rank-independent anti-control; `reversegate` intentionally gives scarce precision/refinement to lower-gate pairs.
- `peerblock_*` groups routed pairs by synthetic expert-owner group. Because each Mac forward has one implicit token origin, this is a one-origin `(owner -> origin)` quality proxy; it still lacks multi-origin traces, actual placement, padding, and a communication kernel.
- `block_gate*_residual_*` sends a low-bit base for every pair and a second low-bit residual for the critical half; its metadata accounting includes both sets of scales.
- Rank is supported only if its held-out quality is competitive at matched payload; real system superiority still requires a two-lane kernel.
"""
    (out / "signal_comparison_report.md").write_text(report, encoding="utf-8")
    print(summary_df[columns].to_string(index=False), flush=True)
    print(f"saved to {out}", flush=True)


if __name__ == "__main__":
    main()
