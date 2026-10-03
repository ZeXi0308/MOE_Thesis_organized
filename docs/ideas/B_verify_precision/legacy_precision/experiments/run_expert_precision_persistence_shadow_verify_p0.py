#!/usr/bin/env python3
"""Idea B, Day-1 decisive experiment: does decode-time quantization risk on the
EXPERT WEIGHT/COMPUTE axis persist over a short horizon, and if so, is that
enough to build a causal shadow-verification controller?

READ THIS BEFORE RUNNING OR CITING RESULTS.

Why this script exists and what axis it is on
----------------------------------------------
This project's Quality Isolation line (``run_quality_isolation_proxy_gpu_strict.py``,
``run_decode_fragility_strict_gpu.py``) tried to predict FUTURE cross-document or
cross-time risk from PAST prefill statistics, on the COMBINE/DISPATCH axis
(quantizing ``raw_outputs`` before the weighted combine, via ``policies.py``).
Round 2 and Round 4 (2026-07-20) progressively falsified that predictive claim.

This script is deliberately on a DIFFERENT axis -- the EXPERT WEIGHT/COMPUTE
axis (the same axis as ``run_energy_slo_fp8_compute_quality_gate.py``, which
monkey-patches ``gate_proj/up_proj/down_proj`` (OLMoE/Qwen2-MoE) or
``w1/w2/w3`` (Mixtral) ``nn.Linear`` matmuls) -- and a DIFFERENT hypothesis: not
"can I predict a document's fragility from its prefill", but "does REALIZED,
WITHIN-DOCUMENT, SHORT-HORIZON risk persist enough that a reactive controller
which only ever looks at its OWN recent past can decide when to escalate
precision". This is a strictly weaker and more defensible claim than anything
the falsified predictor line required: it does not need to predict a NEW
document's risk from arrival-time features, and it does not need to transfer
across documents, models, or degradation mechanisms.

Why INT4 weight-only (not FP8) is the precision under test
------------------------------------------------------------
``run_energy_slo_fp8_compute_quality_gate.py`` already showed real FP8
tensor-core compute on expert FFNs is safe (GO, mean KL 0.0068-0.009, both
models) -- there is no meaningful risk left to protect against at FP8, so a
shadow-verification controller would have nothing to do. INT4 weight-only
fake-quantization (RTN, per-output-channel scale, via
``fake_quant.symmetric_quant_dequant`` -- reused verbatim, zero modification)
is the natural next precision level down: a real, standard quantization
scenario (matches GPTQ/AWQ-style W4A16 serving), with enough real quality risk
and variance to make "when should I escalate" a non-trivial, useful question.
This is a quant-dequant PROXY: it makes NO wall-clock speed claim (no real
INT4 tensor-core kernel is invoked). It only produces the QUALITY signal
needed to test the persistence hypothesis below.

The two hypotheses tested, and their PASS/FAIL thresholds (frozen before
looking at results)
----------------------------------------------------------------------------
H1 (primary, decisive): PAST realized per-step KL predicts NEAR-FUTURE
per-step KL within the same document, at lag 1..8 decode steps.
  GO for a given lag if the document-level-bootstrap 95% CI LOWER bound of
  the pooled Spearman correlation exceeds 0.20 on held-out TEST documents.
  This is deliberately a much weaker ask than the falsified predictor claims:
  it never needs to see a NEW document's arrival-time features, and it uses
  only the risk signal's own recent past.

H2 (secondary, operational): a CAUSAL periodic-shadow-verify + threshold
escalate controller, using ONLY the realized value at its own past verify
points, captures most of the achievable quality protection at a fraction of
the cost of always running high precision.
  GO for a given verify period if, on held-out TEST documents:
    (a) cumulative KL reduction vs. always-low-precision >= 50%, AND
    (b) the fraction of steps served at high precision <= 50%, AND
    (c) cumulative KL is within 2x of the NON-CAUSAL oracle upper bound at
        the same threshold (i.e., causal reactivity is not drastically worse
        than knowing the true per-step risk in advance).

Causality discipline (the most likely place for a silent bug)
----------------------------------------------------------------
Router/gate features recorded here come from the CANDIDATE (INT4-weight) path
itself, not the bf16 reference path -- a real system running INT4 weights only
ever observes ITS OWN internal state, never the reference model's. The router
linear layer itself is never patched (only expert FFN matmuls are, exactly as
in ``run_energy_slo_fp8_compute_quality_gate.py``), so layer-0 features are
identical to the reference path (nothing has diverged yet) but deeper-layer
features are NOT -- that divergence is a genuine downstream effect, not a
patching artifact, and is correctly reflected in what this script records.
Same-step router-feature correlations are reported ONLY as an exploratory
diagnostic (``same_step_diagnostic_correlations.csv``) and are explicitly NOT
part of either GO/NO-GO decision: deciding whether to run an ENTIRE decode
step at low or high precision cannot cite that same step's own router output
as an input, since the router only finishes computing partway through the
step. H1/H2 only ever use lagged (strictly past) information.

Prefill/decode boundary (matches this project's established convention)
----------------------------------------------------------------------------
Exactly as in ``run_decode_fragility_strict_gpu.py``: prefill is ALWAYS full
precision and freshly redone per path (reference vs. candidate each get their
own prefill call) to avoid any KV-cache aliasing between paths. The INT4
weight patch is active ONLY during the decode loop, never during prefill.

Statistical discipline: DOCUMENT-level block bootstrap, not point-level
----------------------------------------------------------------------------
Per-step KL values within one document are strongly serially correlated.
``run_quality_isolation_proxy_gpu_strict.bootstrap_spearman_ci`` resamples
individual POINTS and is NOT reused here -- doing so would silently understate
the true CI width and could turn a NO-GO into a false GO. This script
resamples DOCUMENTS with replacement instead (see
``document_block_bootstrap_spearman_ci``).

Known confounds most likely to overturn a GO result
--------------------------------------------------------
  1. Teacher-forcing on the real corpus continuation (not the model's own
     greedy/sampled generation) avoids the autoregressive-drift confound but
     means this measures "quality of continuing a real human-written
     sequence", not "quality of the model's own free-running generation" --
     the same evidence boundary already documented in
     ``run_decode_fragility_strict_gpu.py``.
  2. INT4 weight-only fake-quant (quant->dequant in bf16) is a quality PROXY,
     not a real low-bit kernel; a real INT4 GEMM could have different error
     characteristics (e.g. group-wise vs per-channel scaling) that change how
     much persistence exists.
  3. The escalate threshold tau is calibrated once on the calibration split
     and reused as-is on test; if the true risk distribution shifts between
     calibration and test documents (plausible with only ~12-20 documents per
     split), the GO/NO-GO could be sensitive to this specific split. Rerunning
     with more documents is the correct response to a borderline result, not
     re-tuning tau on test.
  4. Only expert FFN matmuls are patched (attention, router, LM head stay
     bf16), matching the Energy-SLO script's own scope -- a production system
     quantizing more of the model could show different persistence structure.
  5. Small ``--samples``/``--decode-steps`` is intentional for a decisive
     minimal-cost probe; a GO/NO-GO boundary result should be re-run with more
     documents/steps before being treated as final.

Usage
-----
  python run_expert_precision_persistence_shadow_verify_p0.py \\
      --model allenai/OLMoE-1B-7B-0924 --model-key olmoe \\
      --output-dir outputs/expert_precision_persistence_2026-07-20_olmoe
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
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from capture_moe import patch_mixtral_moe
from fake_quant import symmetric_quant_dequant
from modeling import load_model, load_tokenizer, resolve_device
from run_decode_fragility_strict_gpu import collect_documents
from run_energy_slo_fp8_compute_quality_gate import find_expert_linears
from run_quality_isolation_proxy_gpu_strict import extract_router_features, spearman


# ---------------------------------------------------------------------------
# INT4 weight-only fake-quant patch (W4A16): reuses fake_quant.py verbatim.
# ---------------------------------------------------------------------------

class _Int4WeightOnlyLinear:
    """Drop-in replacement for ``linear.forward``: weight-only INT4
    quant-dequant (per-output-channel symmetric scale), activations stay at
    native precision. Standard W4A16 weight-only quantization scenario. Makes
    NO wall-clock speed claim -- it exists only to produce a real, decision-
    relevant quality-risk signal for the persistence test below."""

    def __init__(self, linear: torch.nn.Linear, device: str):
        w = linear.weight.detach()
        self.weight = symmetric_quant_dequant(w, bits=4).to(device)
        self.bias = linear.bias

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class Int4ComputePatch:
    def __init__(self, model, scope: str, device: str):
        self.linears = find_expert_linears(model, scope)
        self.device = device

    def __enter__(self):
        for linear in self.linears:
            linear.forward = _Int4WeightOnlyLinear(linear, self.device)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for linear in self.linears:
            if "forward" in linear.__dict__:
                del linear.__dict__["forward"]


def self_test(device: str) -> None:
    if device != "cuda":
        raise RuntimeError("this experiment requires a CUDA GPU")
    x = torch.randn(8, 16, device=device, dtype=torch.bfloat16)
    lin = torch.nn.Linear(16, 32, bias=False, device=device, dtype=torch.bfloat16)
    wrapped = _Int4WeightOnlyLinear(lin, device)
    out = wrapped(x)
    if out.shape != (8, 32) or torch.isnan(out).any():
        raise RuntimeError("INT4 weight-only self-test produced an unexpected shape or NaNs")
    ref = x.float() @ lin.weight.detach().float().t()
    rel_err = (out.float() - ref).norm() / ref.norm().clamp_min(1e-12)
    print(f"self-test ok: INT4-weight-only vs bf16 relative L2 error on a random 8x16x32 linear = {rel_err.item():.4f}")


# ---------------------------------------------------------------------------
# Per-document trajectory collection.
# ---------------------------------------------------------------------------

def per_step_kl(reference_logits: torch.Tensor, candidate_logits: torch.Tensor) -> np.ndarray:
    ref = reference_logits.float()
    cand = candidate_logits.float()
    log_p = F.log_softmax(ref, dim=-1)
    log_q = F.log_softmax(cand, dim=-1)
    p = log_p.exp()
    return (p * (log_p - log_q)).sum(dim=-1).numpy()


def decode_reference(model, cache, decode_ids: torch.Tensor) -> torch.Tensor:
    patch_mixtral_moe(model, "full", record_routes=False)
    logits = []
    cur = cache
    for pos in range(decode_ids.shape[1]):
        token = decode_ids[:, pos: pos + 1]
        with torch.no_grad():
            out = model(input_ids=token, past_key_values=cur, use_cache=True)
        cur = out.past_key_values
        logits.append(out.logits[:, -1, :].detach().cpu())
    return torch.cat(logits, dim=0)


def decode_candidate_with_features(
    model, cache, decode_ids: torch.Tensor, num_experts: int, scope: str, device: str,
) -> tuple[torch.Tensor, list[dict]]:
    """Decode step-by-step under the INT4-weight patch, recording per-step
    router features from the CANDIDATE path itself (the only features a real
    online low-precision system could causally observe at that point)."""
    recorder = patch_mixtral_moe(model, "full", record_routes=True)
    logits = []
    feature_rows: list[dict] = []
    cur = cache
    with Int4ComputePatch(model, scope, device):
        for pos in range(decode_ids.shape[1]):
            recorder.route_batches.clear()
            recorder.routing_weight_batches.clear()
            token = decode_ids[:, pos: pos + 1]
            with torch.no_grad():
                out = model(input_ids=token, past_key_values=cur, use_cache=True)
            cur = out.past_key_values
            logits.append(out.logits[:, -1, :].detach().cpu())
            feature_rows.append(extract_router_features(recorder.route_batches, num_experts, 1.0))
    return torch.cat(logits, dim=0), feature_rows


def collect_document_trajectory(
    model, doc_id: int, all_ids_cpu: torch.Tensor, args, num_experts: int, device: str,
) -> pd.DataFrame:
    prompt_ids = all_ids_cpu[:, : args.prompt_len].to(model.device)
    decode_inputs = all_ids_cpu[:, args.prompt_len: args.prompt_len + args.decode_steps].to(model.device)

    # Reference path: fresh full-precision prefill + full-precision decode.
    patch_mixtral_moe(model, "full", record_routes=False)
    with torch.no_grad():
        ref_prefill = model(input_ids=prompt_ids, use_cache=True)
    ref_logits = decode_reference(model, ref_prefill.past_key_values, decode_inputs)

    # Candidate path: SEPARATE fresh full-precision prefill (approximation
    # only enabled after prefill, matching this project's convention), then
    # INT4-weight decode with causal per-step router feature recording.
    patch_mixtral_moe(model, "full", record_routes=False)
    with torch.no_grad():
        cand_prefill = model(input_ids=prompt_ids, use_cache=True)
    cand_logits, feature_rows = decode_candidate_with_features(
        model, cand_prefill.past_key_values, decode_inputs, num_experts, args.fp_scope, device,
    )

    kl = per_step_kl(ref_logits, cand_logits)
    rows = []
    for step, (kl_value, feats) in enumerate(zip(kl, feature_rows)):
        row = {"doc_id": doc_id, "step": step, "kl": float(kl_value)}
        row.update({k: v for k, v in feats.items() if k.startswith("full_route_")})
        rows.append(row)
    del ref_prefill, cand_prefill, ref_logits, cand_logits
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# H1: document-level block-bootstrap lag persistence.
# ---------------------------------------------------------------------------

def document_block_bootstrap_spearman_ci(
    per_doc_pairs: dict[int, tuple[np.ndarray, np.ndarray]],
    n_bootstrap: int,
    seed: int,
) -> tuple[float, float]:
    """Resample WHICH DOCUMENTS contribute pairs, not which individual
    (t, t+lag) pairs -- points within one document are serially correlated,
    so point-level resampling would understate the true CI width."""
    rng = np.random.default_rng(seed)
    doc_ids = list(per_doc_pairs.keys())
    values = []
    for _ in range(n_bootstrap):
        chosen = rng.choice(doc_ids, size=len(doc_ids), replace=True)
        a = np.concatenate([per_doc_pairs[d][0] for d in chosen])
        b = np.concatenate([per_doc_pairs[d][1] for d in chosen])
        values.append(spearman(a, b))
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def lag_persistence_analysis(
    df: pd.DataFrame, lags: list[int], test_docs: list[int], n_bootstrap: int, seed: int, ci_threshold: float,
) -> pd.DataFrame:
    rows = []
    for lag in lags:
        per_doc_pairs: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for doc_id in test_docs:
            sub = df[df.doc_id == doc_id].sort_values("step")
            kl = sub["kl"].to_numpy()
            if len(kl) <= lag:
                continue
            per_doc_pairs[doc_id] = (kl[:-lag], kl[lag:])
        if not per_doc_pairs:
            continue
        pooled_a = np.concatenate([a for a, _ in per_doc_pairs.values()])
        pooled_b = np.concatenate([b for _, b in per_doc_pairs.values()])
        point_rho = spearman(pooled_a, pooled_b)
        ci_low, ci_high = document_block_bootstrap_spearman_ci(per_doc_pairs, n_bootstrap, seed + lag)
        rows.append({
            "lag": lag,
            "spearman": point_rho,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "n_documents": len(per_doc_pairs),
            "n_pairs": int(len(pooled_a)),
            "go_no_go": "GO" if ci_low > ci_threshold else "NO-GO",
        })
    return pd.DataFrame(rows)


def same_step_diagnostics(df: pd.DataFrame, test_docs: list[int]) -> pd.DataFrame:
    """Exploratory only: same-step router-feature vs. same-step KL
    correlation. NOT causally actionable at whole-step decision granularity
    (the router only finishes partway through the step it is describing) and
    NOT part of any GO/NO-GO decision -- reported purely to help interpret
    WHERE risk concentrates."""
    sub = df[df.doc_id.isin(test_docs)]
    feature_cols = [c for c in sub.columns if c.startswith("full_route_")]
    rows = []
    for col in feature_cols:
        rows.append({
            "feature": col,
            "same_step_spearman_vs_kl": spearman(sub[col].to_numpy(), sub["kl"].to_numpy()),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# H2: causal periodic-shadow-verify + threshold-escalate controller simulation.
# ---------------------------------------------------------------------------

def simulate_policies(kl_trajectory: np.ndarray, threshold: float, period: int) -> dict[str, float]:
    T = len(kl_trajectory)

    always_low_kl = float(kl_trajectory.sum())

    verify_mask = np.zeros(T, dtype=bool)
    verify_mask[::period] = True
    no_escalate_kl = float(kl_trajectory[~verify_mask].sum())
    no_escalate_high_frac = float(verify_mask.mean())

    # Reactive escalate: CAUSAL -- decides the window AFTER a verify point
    # using ONLY that verify point's just-realized value.
    high_reactive = np.zeros(T, dtype=bool)
    t = 0
    while t < T:
        high_reactive[t] = True  # verify point itself is served high (bf16)
        verified_kl = kl_trajectory[t]
        end = min(t + period, T)
        if verified_kl > threshold:
            high_reactive[t:end] = True
        t = end
    reactive_kl = float(kl_trajectory[~high_reactive].sum())
    reactive_high_frac = float(high_reactive.mean())

    # Non-causal oracle: perfect per-step knowledge at the SAME threshold.
    high_oracle = kl_trajectory > threshold
    oracle_kl = float(kl_trajectory[~high_oracle].sum())
    oracle_high_frac = float(high_oracle.mean())

    return {
        "always_low_kl": always_low_kl, "always_low_high_frac": 0.0,
        "always_high_kl": 0.0, "always_high_high_frac": 1.0,
        "no_escalate_kl": no_escalate_kl, "no_escalate_high_frac": no_escalate_high_frac,
        "reactive_kl": reactive_kl, "reactive_high_frac": reactive_high_frac,
        "oracle_kl": oracle_kl, "oracle_high_frac": oracle_high_frac,
    }


def controller_simulation_analysis(
    df: pd.DataFrame, calib_docs: list[int], test_docs: list[int],
    periods: list[int], escalate_quantile: float,
    reduction_threshold: float, high_frac_threshold: float, oracle_ratio_threshold: float,
    n_bootstrap: int, seed: int,
) -> tuple[pd.DataFrame, float]:
    calib_kl = df[df.doc_id.isin(calib_docs)]["kl"].to_numpy()
    threshold = float(np.quantile(calib_kl, escalate_quantile))

    rows = []
    for period in periods:
        per_doc: dict[int, dict[str, float]] = {}
        for doc_id in test_docs:
            traj = df[df.doc_id == doc_id].sort_values("step")["kl"].to_numpy()
            if len(traj) < period:
                continue
            per_doc[doc_id] = simulate_policies(traj, threshold, period)
        if not per_doc:
            continue
        doc_ids = list(per_doc.keys())

        def aggregate(keys: list[int]) -> dict[str, float]:
            agg = {name: 0.0 for name in per_doc[doc_ids[0]]}
            for d in keys:
                for name, value in per_doc[d].items():
                    if name.endswith("_high_frac"):
                        agg[name] += value / len(keys)
                    else:
                        agg[name] += value
            return agg

        point = aggregate(doc_ids)
        reduction = 1.0 - point["reactive_kl"] / max(point["always_low_kl"], 1e-12)
        oracle_ratio = point["reactive_kl"] / max(point["oracle_kl"], 1e-9)

        rng = np.random.default_rng(seed + period)
        boot_reductions = []
        for _ in range(n_bootstrap):
            chosen = rng.choice(doc_ids, size=len(doc_ids), replace=True)
            agg_b = aggregate(list(chosen))
            boot_reductions.append(1.0 - agg_b["reactive_kl"] / max(agg_b["always_low_kl"], 1e-12))
        red_ci_low, red_ci_high = np.quantile(boot_reductions, [0.025, 0.975])

        go = bool(
            reduction >= reduction_threshold
            and point["reactive_high_frac"] <= high_frac_threshold
            and oracle_ratio <= oracle_ratio_threshold
            and red_ci_low > 0.0
        )
        rows.append({
            "period": period,
            "threshold_tau": threshold,
            "n_documents": len(doc_ids),
            "always_low_kl": point["always_low_kl"],
            "no_escalate_kl": point["no_escalate_kl"],
            "no_escalate_high_frac": point["no_escalate_high_frac"],
            "reactive_kl": point["reactive_kl"],
            "reactive_high_frac": point["reactive_high_frac"],
            "oracle_kl": point["oracle_kl"],
            "oracle_high_frac": point["oracle_high_frac"],
            "reactive_reduction_vs_always_low": reduction,
            "reactive_reduction_ci_low": float(red_ci_low),
            "reactive_reduction_ci_high": float(red_ci_high),
            "reactive_vs_oracle_kl_ratio": oracle_ratio,
            "go_no_go": "GO" if go else "NO-GO",
        })
    return pd.DataFrame(rows), threshold


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--model-key", required=True)
    p.add_argument("--dataset", default="wikitext103_docs")
    p.add_argument("--split", default="train")
    p.add_argument("--samples", type=int, default=32)
    p.add_argument("--offset", type=int, default=500,
                    help="chosen well beyond ranges already used by other experiments in this project")
    p.add_argument("--calib-samples", type=int, default=12)
    p.add_argument("--prompt-len", type=int, default=64)
    p.add_argument("--decode-steps", type=int, default=48)
    p.add_argument("--fp-scope", choices=["all", "gate_up_only", "down_only"], default="all")
    p.add_argument("--lags", default="1,2,3,4,6,8")
    p.add_argument("--persistence-ci-threshold", type=float, default=0.20)
    p.add_argument("--verify-periods", default="4,8,16")
    p.add_argument("--escalate-quantile", type=float, default=0.75)
    p.add_argument("--controller-reduction-threshold", type=float, default=0.50)
    p.add_argument("--controller-high-frac-threshold", type=float, default=0.50)
    p.add_argument("--controller-oracle-ratio-threshold", type=float, default=2.0)
    p.add_argument("--n-bootstrap", type=int, default=500)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--seed", type=int, default=20260720)
    p.add_argument("--offline", action="store_true")
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    device = resolve_device()
    self_test(device)

    tokenizer = load_tokenizer(args.model, local_files_only=args.offline)
    model, load_seconds = load_model(args.model, dtype_name=args.dtype, local_files_only=args.offline)
    top_k = int(model.config.num_experts_per_tok)
    num_experts_value = getattr(model.config, "num_experts", None)
    if num_experts_value is None:
        num_experts_value = getattr(model.config, "num_local_experts")
    num_experts = int(num_experts_value)

    documents = collect_documents(tokenizer, args)
    all_rows = []
    for local_index, (_text, all_ids_cpu) in enumerate(documents):
        doc_id = args.offset + local_index
        doc_df = collect_document_trajectory(model, doc_id, all_ids_cpu, args, num_experts, device)
        all_rows.append(doc_df)
        mean_kl = doc_df["kl"].mean()
        p95_kl = doc_df["kl"].quantile(0.95)
        print(f"[{args.model_key}] doc {local_index + 1}/{len(documents)} (id={doc_id}): "
              f"mean_kl={mean_kl:.6f} p95_kl={p95_kl:.6f}", flush=True)
        pd.concat(all_rows, ignore_index=True).to_csv(out / "per_step_samples.partial.csv", index=False)

    df = pd.concat(all_rows, ignore_index=True)
    df.to_csv(out / "per_step_samples.csv", index=False)

    doc_ids_ordered = args.offset + np.arange(len(documents))
    calib_docs = list(doc_ids_ordered[: args.calib_samples])
    test_docs = list(doc_ids_ordered[args.calib_samples:])
    if not test_docs:
        raise ValueError("--calib-samples leaves no test documents")

    lags = [int(x) for x in args.lags.split(",") if x.strip()]
    persistence = lag_persistence_analysis(
        df, lags, test_docs, args.n_bootstrap, args.seed, args.persistence_ci_threshold,
    )
    persistence.to_csv(out / "lag_persistence_results.csv", index=False)

    diagnostics = same_step_diagnostics(df, test_docs)
    diagnostics.to_csv(out / "same_step_diagnostic_correlations.csv", index=False)

    periods = [int(x) for x in args.verify_periods.split(",") if x.strip()]
    controller, tau = controller_simulation_analysis(
        df, calib_docs, test_docs, periods, args.escalate_quantile,
        args.controller_reduction_threshold, args.controller_high_frac_threshold,
        args.controller_oracle_ratio_threshold, args.n_bootstrap, args.seed,
    )
    controller.to_csv(out / "controller_simulation_results.csv", index=False)

    h1_go = bool((persistence["go_no_go"] == "GO").any()) if len(persistence) else False
    h2_go = bool((controller["go_no_go"] == "GO").any()) if len(controller) else False

    metadata = {
        "model": args.model,
        "model_key": args.model_key,
        "dataset": args.dataset,
        "samples": args.samples,
        "offset": args.offset,
        "calib_documents": calib_docs,
        "test_documents": test_docs,
        "prompt_len": args.prompt_len,
        "decode_steps": args.decode_steps,
        "fp_scope": args.fp_scope,
        "top_k": top_k,
        "num_experts": num_experts,
        "load_seconds": load_seconds,
        "escalate_threshold_tau": tau,
        "escalate_quantile": args.escalate_quantile,
        "h1_any_lag_go": h1_go,
        "h2_any_period_go": h2_go,
        "evidence_boundary": (
            "teacher-forced decode on real corpus continuations; INT4 weight-only "
            "fake-quant (quant-dequant proxy, no real low-bit kernel, no wall-clock "
            "claim); router/gate features taken from the CANDIDATE path only; "
            "document-level block bootstrap for all CIs."
        ),
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")

    lines = [
        f"# Expert-Precision Persistence & Shadow-Verify Controller ({args.model_key})", "",
        f"- documents: {len(documents)} (calib={len(calib_docs)}, test={len(test_docs)}), "
        f"decode_steps={args.decode_steps}, fp_scope={args.fp_scope}",
        f"- escalate threshold tau (calibrated at {args.escalate_quantile:.2f} quantile of calib KL): {tau:.6f}",
        "",
        "## H1: lag persistence (primary, decisive; GO iff CI_low > "
        f"{args.persistence_ci_threshold})",
        persistence.to_string(index=False) if len(persistence) else "(no lags evaluated)",
        "",
        "## H2: causal shadow-verify controller simulation (secondary, operational)",
        controller.to_string(index=False) if len(controller) else "(no periods evaluated)",
        "",
        "## Same-step router-feature diagnostics (exploratory ONLY, not part of any GO/NO-GO)",
        diagnostics.to_string(index=False) if len(diagnostics) else "(no features)",
        "",
        f"OVERALL: H1 {'GO' if h1_go else 'NO-GO'} (at least one lag) / "
        f"H2 {'GO' if h2_go else 'NO-GO'} (at least one verify period).",
    ]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nsaved to {out}")


if __name__ == "__main__":
    main()
