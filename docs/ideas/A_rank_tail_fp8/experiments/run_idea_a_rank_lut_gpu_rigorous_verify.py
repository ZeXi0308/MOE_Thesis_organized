#!/usr/bin/env python3
"""Idea A (combine-axis) foundational claim: rigorous real-GPU re-verification
with bootstrap CI, replacing the original Mac-CPU point-estimate evidence.

READ THIS BEFORE RUNNING OR CITING RESULTS.

Why this script exists
-----------------------
The ORIGINAL Idea A claim (docs/05_idea_a_legacy/IdeaA_主实验报告_论文版.md,
thesis_evidence/, dated 2026-06-24 through 2026-07-14) is the single largest,
most reproduced effect size in this entire project: at matched byte budget,
putting INT4 on the LOWEST-gate ("tail") routed expert output is 41x-108x less
damaging (KL) than putting it on the HIGHEST-gate ("head", i.e. rank1) output,
replicated across 3 different MoE architectures (OLMoE top-8, Mixtral-
TinyMistral top-2, LLM-jp top-16). It is also the taproot of every later
combine-axis candidate (FP8-first tail-INT4 escalation, R-layout, QuotaEP-H,
receiver-aware) -- if this does not hold up, nothing downstream does.

However, every one of those numbers was produced on a Mac M5 Pro CPU-only
machine with N=32-256 documents and NO confidence interval whatsoever -- only
single point estimates. This script re-runs the two most decisive comparisons
on the real RTX 5090 GPU this project now has access to, on a FRESH document
offset never touched by any prior experiment, with proper document-level
paired bootstrap CIs. This is a verification/tightening of an already very
strong claim, not a rescue of a weak one -- given the original effect sizes
(41x-108x), this is expected to replicate cleanly; the point is to finally
attach real statistical evidence to it, matching the rigor bar the rest of
this project's 2026-07-20 audits have been held to.

What is being re-verified (two claims, two frozen GO/NO-GO criteria)
----------------------------------------------------------------------------
Claim 1 (tail-vs-head asymmetry, matched byte budget -- the "smoking gun"):
  ``rank1_int4`` (INT4 only on the highest-gate rank) vs ``rankk_int4`` (INT4
  only on the lowest-gate/tail rank). Both change exactly one rank from BF16
  to INT4, so byte saving is IDENTICAL between them -- any KL difference is
  caused by WHICH rank was chosen, not by how much was compressed.
  GO iff the document-level paired-bootstrap 95% CI of
  (rank1_int4_kl - rankk_int4_kl) is entirely > 0 (tail strictly safer) AND
  the ratio rank1_int4_kl / rankk_int4_kl exceeds 5x (matching the original
  report's "strong" bar, well below its 41x-108x point estimates).

Claim 2 (FP8-first tail-INT4 Pareto frontier): starting from ``uniform_fp8``
(50% saving, near-zero KL), progressively moving MORE tail ranks from FP8 to
INT4 (``fp8top{n}_rest_int4`` for decreasing n) should raise byte saving
smoothly without KL blowing up, staying far below ``uniform_int4``'s KL at
its much higher saving.
  GO iff, at every point on the sweep, the mean KL is monotonically
  non-decreasing as saving increases (no instability) AND the point closest
  to the original report's ~62.5% saving target has a 95% CI upper bound
  still far below (>5x lower than) ``uniform_int4``'s CI lower bound.

This script makes NO new approximation mechanism -- it only calls the
EXISTING ``policies.py`` strategies (``full``, ``uniform_fp8``,
``uniform_int4``, ``rank1_int4``, ``rankk_int4``, ``fp8top{n}_rest_int4``)
through the EXISTING ``capture_moe.patch_mixtral_moe`` combine-degradation
patch, exactly as every prior Idea A experiment did. The only things genuinely
new here are: real GPU (not Mac CPU), a document offset never seen by any
prior experiment in this project, and document-level bootstrap CIs.

Evidence boundary (unchanged from the entire combine-axis lineage)
----------------------------------------------------------------------------
This is single-forward (not decode-loop) KL on the COMBINE output, exactly
like the original report. It says nothing about real communication latency,
real all-to-all, or real multi-GPU bottleneck bytes -- those claims belong to
the separate receiver-aware/direct-benefit-controller line and are NOT
re-verified here. This script only re-verifies the QUALITY side of the
foundational rank-tail claim.

Known confounds most likely to overturn a GO result
--------------------------------------------------------
  1. ``rank1_int4``/``rankk_int4`` use a per-token symmetric INT4 fake-quant
     proxy (``fake_quant.symmetric_quant_dequant``), the SAME proxy the
     original report used. The 2026-07-11 hardware-format audit found this
     proxy overstates damage relative to block-scaled MXFP4/NVFP4 (fixed-tail
     KL shrinks from 0.03032 to 0.00684/0.00571 under the more realistic
     format) -- the ABSOLUTE KL numbers here are proxy-dependent; only the
     RELATIVE tail-vs-head ordering is the load-bearing claim, and that
     ordering was already confirmed format-robust in the 2026-07-13 R-layout
     formal audit.
  2. Single-forward, not decode-loop: this measures a one-shot combine
     degradation on a truncated document, not autoregressive quality drift.
  3. ``fp8top{n}_rest_int4``'s exact saving-vs-KL curve depends on the
     model's top_k and gate-weight distribution; the sweep grid below is
     model-specific (computed from each model's real top_k), not copy-pasted
     from the OLMoE-specific ``fp8_r5678int4`` naming in the original report.

Usage
-----
  python run_idea_a_rank_lut_gpu_rigorous_verify.py \\
      --model allenai/OLMoE-1B-7B-0924 \\
      --model-revision 6d84c48581ece794365f2b8e9cfb043c68ade9c5 \\
      --model-key olmoe \\
      --output-dir outputs/idea_a_rank_lut_gpu_verify_2026-07-20_olmoe
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

import capture_moe as capture_moe_module
import fake_quant as fake_quant_module
import metrics as metrics_module
import modeling as modeling_module
import policies as policies_module
import prompts as prompts_module
from capture_moe import patch_mixtral_moe
from metrics import MetricAccumulator
from modeling import load_model, load_tokenizer
from policies import make_policy
from prompts import get_prompts


INT4_QUANTIZATION_CONTRACT = {
    "name": "per_row_symmetric_int_v1",
    "scale_dtype": "float32",
    "scale_formula": "clamp_min(absmax,1e-8)/qmax",
    "rounding": "nearest_ties_to_even",
    "zero_point": 0,
    "int4_qmin": -7,
    "int4_qmax": 7,
    "int4_storage": "two_signed_nibbles_per_uint8",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--model-revision", required=True,
                   help="immutable Hugging Face commit SHA")
    p.add_argument("--model-key", required=True)
    p.add_argument("--dataset", default="wikitext103_docs")
    p.add_argument("--split", default="train")
    p.add_argument("--samples", type=int, default=128)
    p.add_argument("--offset", type=int, default=600,
                    help="fresh offset, never touched by any prior experiment in this project")
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--seed", type=int, default=20260720)
    p.add_argument("--n-bootstrap", type=int, default=2000)
    p.add_argument("--head-tail-ratio-threshold", type=float, default=5.0)
    p.add_argument("--offline", action="store_true")
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


def forward_logits(model, inputs, policy_name: str) -> torch.Tensor:
    patch_mixtral_moe(model, policy_name, record_routes=False)
    with torch.no_grad():
        return model(**inputs).logits


def build_policy_grid(top_k: int) -> list[str]:
    grid = ["full", "uniform_fp8", "uniform_int4", "rank1_int4", "rankk_int4"]
    # FP8-first tail-INT4 escalation: move the tail-most n ranks from FP8 to
    # INT4, n = 1 .. top_k, evenly spaced at roughly 5 points across the range.
    n_points = sorted(set(
        max(1, round(top_k * frac)) for frac in (1.0, 0.75, 0.625, 0.5, 0.25)
    ))
    for n_int4 in n_points:
        n_fp8 = top_k - n_int4
        if n_fp8 <= 0:
            continue  # would equal uniform_int4, already in the grid
        grid.append(f"fp8top{n_fp8}_rest_int4")
    return grid


def collect(args) -> tuple[pd.DataFrame, dict]:
    tokenizer = load_tokenizer(
        args.model, local_files_only=args.offline, revision=args.model_revision
    )
    model, load_seconds = load_model(
        args.model,
        dtype_name=args.dtype,
        local_files_only=args.offline,
        revision=args.model_revision,
    )
    top_k = int(model.config.num_experts_per_tok)
    policies = build_policy_grid(top_k)
    byte_saving = {name: make_policy(name).byte_saving(top_k) for name in policies}

    texts = get_prompts(args.dataset, args.samples, offset=args.offset, split=args.split, seed=args.seed)
    rows = []
    partial_path = Path(args.output_dir) / "per_document.partial.csv"
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    for doc_index, text in enumerate(texts):
        sample_id = args.offset + doc_index
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=args.seq_len)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        ref_logits = forward_logits(model, inputs, "full")
        row: dict[str, float | int] = {"sample_id": sample_id, "token_count": int(inputs["input_ids"].shape[1])}
        for policy in policies:
            logits = forward_logits(model, inputs, policy) if policy != "full" else ref_logits
            acc = MetricAccumulator()
            sample = acc.add(sample_id, logits, inputs["input_ids"], baseline_logits=ref_logits)
            row[f"{policy}__kl"] = sample.mean_token_kl
            row[f"{policy}__nll"] = sample.mean_nll
        rows.append(row)
        pd.DataFrame(rows).to_csv(partial_path, index=False)
        print(f"[{args.model_key}] doc {doc_index + 1}/{len(texts)}: "
              f"rank1_int4={row['rank1_int4__kl']:.6f} rankk_int4={row['rankk_int4__kl']:.6f} "
              f"uniform_fp8={row['uniform_fp8__kl']:.6f} uniform_int4={row['uniform_int4__kl']:.6f}", flush=True)
        del ref_logits

    metadata = {
        "model": args.model, "model_revision": args.model_revision,
        "model_key": args.model_key, "top_k": top_k,
        "policies": policies, "byte_saving": byte_saving,
        "dataset": args.dataset, "split": args.split,
        "samples": args.samples, "offset": args.offset, "seq_len": args.seq_len,
        "dtype": args.dtype, "producer_seed": args.seed,
        "load_seconds": load_seconds,
        "evidence_boundary": (
            "single-forward combine-output KL on the COMBINE axis only, real GPU, "
            "fresh document offset, document-level paired bootstrap CIs. No decode-loop, "
            "no real communication latency claim."
        ),
    }
    return pd.DataFrame(rows), metadata


def document_bootstrap_ci(values: np.ndarray, n_bootstrap: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    means = np.array([values[rng.integers(0, n, size=n)].mean() for _ in range(n_bootstrap)])
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def analyze(df: pd.DataFrame, metadata: dict, args) -> tuple[pd.DataFrame, dict]:
    policies = metadata["policies"]
    byte_saving = metadata["byte_saving"]
    seed = args.seed

    summary_rows = []
    for policy in policies:
        vals = df[f"{policy}__kl"].to_numpy(dtype=float)
        ci_low, ci_high = document_bootstrap_ci(vals, args.n_bootstrap, seed)
        summary_rows.append({
            "policy": policy,
            "byte_saving": byte_saving[policy],
            "mean_kl": float(vals.mean()),
            "kl_ci_low": ci_low,
            "kl_ci_high": ci_high,
        })
    summary = pd.DataFrame(summary_rows).sort_values("byte_saving").reset_index(drop=True)

    # Claim 1: matched-byte-budget tail-vs-head asymmetry.
    head = df["rank1_int4__kl"].to_numpy(dtype=float)
    tail = df["rankk_int4__kl"].to_numpy(dtype=float)
    diff = head - tail
    rng = np.random.default_rng(seed + 1)
    n = len(diff)
    boot_diff = np.array([diff[rng.integers(0, n, size=n)].mean() for _ in range(args.n_bootstrap)])
    diff_ci_low, diff_ci_high = float(np.quantile(boot_diff, 0.025)), float(np.quantile(boot_diff, 0.975))
    ratio = float(head.mean() / max(tail.mean(), 1e-12))
    claim1_go = bool(diff_ci_low > 0.0 and ratio > args.head_tail_ratio_threshold)

    # Claim 2: FP8-first tail-INT4 Pareto frontier is monotone and stays far
    # below uniform_int4 at every point.
    frontier = summary[summary["policy"].isin(["uniform_fp8"] +
                                                [p for p in policies if p.startswith("fp8top")] +
                                                ["uniform_int4"])].sort_values("byte_saving")
    kl_series = frontier["mean_kl"].to_numpy()
    monotone = bool(np.all(np.diff(kl_series) >= -1e-9))
    uniform_int4_ci_low = float(summary.loc[summary.policy == "uniform_int4", "kl_ci_low"].iloc[0])
    fp8_rows = frontier[frontier["policy"].str.startswith("fp8top")]
    worst_fp8_tail_ci_high = float(fp8_rows["kl_ci_high"].max()) if len(fp8_rows) else float("nan")
    far_below_uniform_int4 = bool(
        len(fp8_rows) > 0 and worst_fp8_tail_ci_high > 0
        and (uniform_int4_ci_low / max(worst_fp8_tail_ci_high, 1e-12)) > 5.0
    )
    claim2_go = bool(monotone and far_below_uniform_int4)

    result = {
        "n_documents": int(len(df)),
        "claim1_tail_vs_head": {
            "head_mean_kl": float(head.mean()), "tail_mean_kl": float(tail.mean()),
            "diff_mean": float(diff.mean()), "diff_ci_low": diff_ci_low, "diff_ci_high": diff_ci_high,
            "ratio_head_over_tail": ratio, "go_no_go": "GO" if claim1_go else "NO-GO",
        },
        "claim2_fp8_first_pareto": {
            "monotone": monotone,
            "uniform_int4_kl_ci_low": uniform_int4_ci_low,
            "worst_fp8top_tail_kl_ci_high": worst_fp8_tail_ci_high,
            "safety_margin_x": (uniform_int4_ci_low / max(worst_fp8_tail_ci_high, 1e-12))
            if worst_fp8_tail_ci_high == worst_fp8_tail_ci_high else None,
            "go_no_go": "GO" if claim2_go else "NO-GO",
        },
    }
    return summary, result


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing to mix rank-quality evidence in non-empty {out}")
    out.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("formal rank-quality producer requires CUDA; CPU fallback is forbidden")
    gpu_name = torch.cuda.get_device_name(0)
    compute_capability = list(torch.cuda.get_device_capability(0))
    if "rtx 5090" not in gpu_name.lower() or compute_capability != [12, 0]:
        raise RuntimeError(
            f"formal rank-quality producer requires RTX 5090 sm_120, got "
            f"{gpu_name!r} capability={compute_capability}"
        )
    if not torch.__version__.startswith("2.8.0") or not str(torch.version.cuda).startswith("12.8"):
        raise RuntimeError(
            f"frozen runtime requires torch 2.8.0 / CUDA 12.8, got "
            f"torch={torch.__version__} cuda={torch.version.cuda}"
        )

    df, metadata = collect(args)
    df.to_csv(out / "per_document.csv", index=False)
    summary, result = analyze(df, metadata, args)
    summary.to_csv(out / "policy_summary_with_ci.csv", index=False)
    metadata["result"] = result
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    producer_path = Path(__file__).resolve()
    dependency_paths = {
        "fake_quant.py": Path(fake_quant_module.__file__).resolve(),
        "policies.py": Path(policies_module.__file__).resolve(),
        "capture_moe.py": Path(capture_moe_module.__file__).resolve(),
        "metrics.py": Path(metrics_module.__file__).resolve(),
        "modeling.py": Path(modeling_module.__file__).resolve(),
        "prompts.py": Path(prompts_module.__file__).resolve(),
    }
    provenance = {
        "schema_version": "rank-quality-int4-provenance-v1",
        "attestation": "PRODUCER_EMITTED_DURING_FORWARD_RUN",
        "producer_path": str(producer_path),
        "producer_sha256": sha256_file(producer_path),
        "runtime_environment": {
            "gpu": gpu_name,
            "compute_capability": compute_capability,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
        },
        "run_identity": {
            "model": args.model,
            "model_key": args.model_key,
            "model_revision": args.model_revision,
            "dataset": args.dataset,
            "split": args.split,
            "samples": args.samples,
            "offset": args.offset,
            "seq_len": args.seq_len,
            "dtype": args.dtype,
            "producer_seed": args.seed,
        },
        "dependency_sha256": {
            name: sha256_file(path) for name, path in sorted(dependency_paths.items())
        },
        "per_document_sha256": sha256_file(out / "per_document.csv"),
        "quantization_contract": INT4_QUANTIZATION_CONTRACT,
    }
    (out / "quantization_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8"
    )

    c1 = result["claim1_tail_vs_head"]
    c2 = result["claim2_fp8_first_pareto"]
    lines = [
        f"# Idea A Rank-LUT Foundational Claim -- Real GPU Rigorous Re-Verification ({metadata['model_key']})",
        "",
        f"- documents: {result['n_documents']} (fresh offset={args.offset}), top_k={metadata['top_k']}, seq_len={args.seq_len}",
        "",
        "## Policy summary (byte saving, mean KL, 95% CI)",
        summary.to_string(index=False),
        "",
        "## Claim 1: matched-byte-budget tail-vs-head asymmetry (the 'smoking gun')",
        f"- rank1_int4 (head) mean KL = {c1['head_mean_kl']:.6f}",
        f"- rankk_int4 (tail) mean KL = {c1['tail_mean_kl']:.6f}",
        f"- head - tail diff: mean={c1['diff_mean']:.6f}, 95% CI=[{c1['diff_ci_low']:.6f}, {c1['diff_ci_high']:.6f}]",
        f"- head/tail ratio = {c1['ratio_head_over_tail']:.2f}x  (GO threshold: CI_low > 0 and ratio > {args.head_tail_ratio_threshold}x)",
        f"- VERDICT: {c1['go_no_go']}",
        "",
        "## Claim 2: FP8-first tail-INT4 Pareto frontier",
        f"- monotone non-decreasing KL as saving increases: {c2['monotone']}",
        f"- uniform_int4 KL CI low = {c2['uniform_int4_kl_ci_low']:.6f}",
        f"- worst fp8top*_rest_int4 KL CI high = {c2['worst_fp8top_tail_kl_ci_high']:.6f}",
        f"- safety margin (uniform_int4 / worst fp8top*) = {c2['safety_margin_x']}",
        f"- VERDICT: {c2['go_no_go']}",
    ]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nsaved to {out}")


if __name__ == "__main__":
    main()
