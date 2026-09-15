#!/usr/bin/env python3
"""Energy-SLO Precision EP: the missing quality confound gate for REAL FP8
tensor-core COMPUTE (not communication) on expert FFNs.

READ THIS BEFORE RUNNING OR CITING RESULTS.

Why this script exists
-----------------------
``run_energy_slo_power_probe.py`` already showed batching (17.4x energy/token
improvement, OLMoE, batch 1->64) and real FP8 tensor-core GEMM (34.3% lower
energy/matmul vs bf16) are both real, hardware-measured effects -- but the
FP8 measurement was a matched-shape MICRO-BENCHMARK (random tensors, one
isolated gate_proj-sized matmul, scale=1.0 i.e. no real dynamic range
handling). It never ran the actual model end-to-end with FP8-cast expert
compute and checked whether doing so hurts DOWNSTREAM QUALITY. This script
closes exactly that gap and nothing else: it is a quality confound GATE, not
a new energy measurement (energy is assumed from the existing power-probe
result; only the quality side is new here).

What is actually computed
----------------------------
For each document, a REFERENCE forward pass runs the untouched bf16 model.
Then every expert's ``gate_proj``/``up_proj``/``down_proj`` ``nn.Linear``
layers (SwiGLU-style expert MLP, shared by OLMoE/Mixtral/Qwen2-MoE) are
monkey-patched to run their matmul as a REAL ``torch._scaled_mm`` FP8 E4M3
tensor-core GEMM -- weights cast+scaled ONCE per patch (per-tensor absmax/448
scale, static since weights don't change), activations cast+scaled PER CALL
(per-tensor absmax/448, dynamic since activations vary by input) -- and a
CANDIDATE forward pass runs with the SAME input_ids and SAME routing gate
(only the expert MLP matmuls change dtype/kernel; the router itself is never
touched, so any routing divergence at deeper layers is a genuine downstream
effect of the earlier layers' altered numerics, not a patching artifact).
Token-level KL(reference || candidate) and NLL are computed exactly as in
every other quality experiment in this project (``metrics.py``).

Why this is NOT directly comparable to the project's existing "uniform_int4"
communication-degradation KL number
----------------------------------------------------------------------------
Prior rounds (receiver-aware, quality isolation) repeatedly found that
quantizing COMMUNICATION bytes and quantizing COMPUTE are not interchangeable
-- a format that looks fine on one axis can be catastrophic on the other
(the "microscopic MSE parity does not imply model-level KL parity" lesson
from the 2026-07-20 receiver-progressive audit). This script measures the
COMPUTE axis for the first time in this project; do not assume the known
uniform-INT4-communication KL (~0.257 on OLMoE, from the receiver-aware
codec-break-even round) transfers here. It is used below ONLY as a
qualitative "this is what clearly-too-much damage looks like" anchor, not a
statistically comparable baseline.

Go/No-Go thresholds (a judgment call made explicit, NOT a pre-registered
project constant -- no ``NLL_MARGIN`` or equivalent constant exists elsewhere
in this codebase to inherit)
----------------------------------------------------------------------------------
  - GO ("FP8 compute is a plausible free lunch"): 95% CI UPPER bound of mean
    token KL < ``--acceptable-kl`` (default 0.05, roughly an order of
    magnitude below the known-bad 0.257 communication-INT4 anchor).
  - NO-GO ("FP8 compute is not free"): 95% CI LOWER bound of mean token KL >
    ``--acceptable-kl``.
  - INCONCLUSIVE: CI straddles the threshold -- collect more documents before
    making a claim either way.

Known confounds most likely to overturn a GO result
--------------------------------------------------------
  1. Per-tensor (not per-channel/per-block) absmax scaling is the simplest
     real FP8 recipe and the one already used in the power-probe script; it
     is also the recipe most likely to lose accuracy on outlier-heavy
     activations. A GO here does not rule out a NO-GO under per-channel or
     microscaling (MXFP8) recipes -- those are a natural follow-up, not
     covered here.
  2. Only the FFN matmuls are patched; attention projections, the router's
     own gate matmul, and the embedding/LM head are left in bf16. A
     production system casting MORE of the model to FP8 could show a larger
     quality delta than this script measures.
  3. ``--num-docs`` is necessarily small (this is a decisive minimal-cost
     probe, not a paper-scale sweep); a GO/INCONCLUSIVE boundary result
     should be re-run with more documents before being treated as final.
  4. Routing divergence at deeper layers (a real, not spurious, effect of
     this patch -- see above) means KL is not purely "compute rounding
     error"; some of it is "the model now routes differently". A very high
     KL could in principle come mostly from routing flips rather than
     precision loss per se -- this script reports ``mean_token_kl`` only; a
     follow-up could additionally log routing-agreement-rate as a
     diagnostic if this matters for interpretation.

Usage
-----
  python run_energy_slo_fp8_compute_quality_gate.py \\
      --model allenai/OLMoE-1B-7B-0924 --model-key olmoe \\
      --output-dir outputs/energy_slo_fp8_compute_quality_2026-07-20
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

import torch

from metrics import MetricAccumulator
from modeling import load_model, load_tokenizer, resolve_device
from prompts import get_prompts

FP8_MAX = 448.0
KNOWN_BAD_COMMUNICATION_INT4_KL_ANCHOR = 0.257  # qualitative anchor only, see docstring


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="allenai/OLMoE-1B-7B-0924")
    p.add_argument("--model-key", default="olmoe")
    p.add_argument("--dataset", default="wikitext103_docs",
                    help="use wikitext103_docs for documents no prior experiment in this project has touched")
    p.add_argument("--num-docs", type=int, default=24)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--seed", type=int, default=20260720)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--fp8-scope", choices=["all", "gate_up_only", "down_only"], default="all")
    p.add_argument("--acceptable-kl", type=float, default=0.05)
    p.add_argument("--n-bootstrap", type=int, default=1000)
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


def _fp8_cast(x32: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    amax = x32.abs().amax().clamp_min(1e-12)
    scale = amax / FP8_MAX
    x_fp8 = (x32 / scale).to(torch.float8_e4m3fn)
    return x_fp8, scale


class _FP8CastLinear:
    """Drop-in replacement for `linear.forward`: real torch._scaled_mm FP8
    E4M3 GEMM. Weight cast+scaled once (static); activation cast+scaled per
    call (dynamic, since activations vary by input)."""

    def __init__(self, linear: torch.nn.Linear, device: str):
        w = linear.weight.detach().float()  # [out, in], row-major contiguous
        w_fp8_full, w_scale = _fp8_cast(w)  # still [out, in], row-major fp8
        # cuBLASLt's _scaled_mm requires a row-major "a" and a column-major
        # "b". Transposing a row-major-contiguous tensor gives a [in, out]
        # VIEW that is column-major with no copy -- do NOT call .contiguous()
        # here, that would silently flip it back to row-major and re-trigger
        # "Only multiplication of row-major and column-major matrices is
        # supported by cuBLASLt".
        self.w_fp8 = w_fp8_full.t().to(device)
        # Kept as 0-dim scalar tensors, matching the exact scale_a/scale_b shape
        # already proven to work with torch._scaled_mm on this hardware in
        # run_energy_slo_power_probe.py (`torch.tensor(1.0, device=device)`).
        self.w_scale = w_scale.to(device)
        self.bias = linear.bias
        self.out_dtype = linear.weight.dtype

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x2 = x.reshape(-1, orig_shape[-1]).float()
        if x2.shape[0] == 0:
            # A routed-but-unselected expert for this microbatch (real MoE
            # top-k routing routinely sends 0 tokens to some experts) --
            # amax()/torch._scaled_mm cannot run on an empty operand; the
            # correct output is simply an empty tensor of the right shape.
            out_features = self.w_fp8.shape[1]
            return x.new_zeros(*orig_shape[:-1], out_features)
        x_fp8, x_scale = _fp8_cast(x2)
        out = torch._scaled_mm(
            x_fp8.contiguous(), self.w_fp8,
            scale_a=x_scale, scale_b=self.w_scale,
            out_dtype=torch.bfloat16,
        )
        if self.bias is not None:
            out = out + self.bias
        return out.reshape(*orig_shape[:-1], -1).to(self.out_dtype)


def find_expert_linears(model, scope: str) -> list[torch.nn.Linear]:
    # Two SwiGLU-expert naming conventions exist across HF MoE archs:
    #   OLMoE/Qwen2-MoE style: gate_proj, up_proj, down_proj
    #   Mixtral style ("block_sparse_moe"): w1 (=gate), w3 (=up), w2 (=down)
    #     forward: w2(act_fn(w1(x)) * w3(x))
    gate_up_names = ["gate_proj", "up_proj", "w1", "w3"]
    down_names = ["down_proj", "w2"]
    targets = []
    if scope in ("all", "gate_up_only"):
        targets += gate_up_names
    if scope in ("all", "down_only"):
        targets += down_names
    linears: list[torch.nn.Linear] = []
    for layer in model.model.layers:
        if hasattr(layer, "block_sparse_moe"):
            moe = layer.block_sparse_moe
        elif hasattr(layer, "mlp") and hasattr(layer.mlp, "experts"):
            moe = layer.mlp
        else:
            continue
        for expert in moe.experts:
            for name in targets:
                if hasattr(expert, name) and isinstance(getattr(expert, name), torch.nn.Linear):
                    linears.append(getattr(expert, name))
    return linears


class Fp8ComputePatch:
    def __init__(self, model, scope: str, device: str):
        self.linears = find_expert_linears(model, scope)
        self.device = device
        self._applied = False

    def __enter__(self):
        for linear in self.linears:
            linear.forward = _FP8CastLinear(linear, self.device)
        self._applied = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for linear in self.linears:
            if "forward" in linear.__dict__:
                del linear.__dict__["forward"]
        self._applied = False


def self_test(device: str) -> None:
    if device != "cuda":
        raise RuntimeError("this experiment requires a CUDA GPU with real FP8 tensor-core support")
    if not hasattr(torch, "_scaled_mm"):
        raise RuntimeError("torch._scaled_mm is unavailable in this torch build")
    x = torch.randn(8, 16, device=device, dtype=torch.bfloat16)
    w = torch.nn.Linear(16, 32, bias=False, device=device, dtype=torch.bfloat16)
    wrapped = _FP8CastLinear(w, device)
    out = wrapped(x)
    if out.shape != (8, 32) or torch.isnan(out).any():
        raise RuntimeError("FP8 self-test produced an unexpected shape or NaNs; aborting before the main loop")
    ref = x.float() @ w.weight.detach().float().t()
    rel_err = (out.float() - ref).norm() / ref.norm().clamp_min(1e-12)
    print(f"self-test ok: FP8 vs bf16 relative L2 error on a random 8x16x32 linear = {rel_err.item():.4f}")


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = resolve_device()
    self_test(device)

    print(f"[{args.model_key}] loading model...")
    model, _elapsed = load_model(args.model, dtype_name="bfloat16", local_files_only=True)
    tokenizer = load_tokenizer(args.model, local_files_only=True)
    model.eval()

    docs = get_prompts(args.dataset, args.num_docs, offset=args.offset, seed=args.seed)

    acc_reference = MetricAccumulator()
    acc_candidate = MetricAccumulator()
    per_doc_rows = []

    for sample_id, text in enumerate(docs):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=args.max_length).to(device)
        with torch.no_grad():
            ref_logits = model(**inputs).logits
        with torch.no_grad(), Fp8ComputePatch(model, args.fp8_scope, device):
            cand_logits = model(**inputs).logits

        ref_row = acc_reference.add(sample_id, ref_logits, inputs["input_ids"])
        cand_row = acc_candidate.add(sample_id, cand_logits, inputs["input_ids"], baseline_logits=ref_logits)
        per_doc_rows.append({
            "sample_id": sample_id,
            "token_count": ref_row.token_count,
            "reference_mean_nll": ref_row.mean_nll,
            "candidate_mean_nll": cand_row.mean_nll,
            "mean_token_kl": cand_row.mean_token_kl,
        })
        print(f"  doc {sample_id}: tokens={ref_row.token_count}, "
              f"ref_nll={ref_row.mean_nll:.4f}, cand_nll={cand_row.mean_nll:.4f}, "
              f"kl={cand_row.mean_token_kl:.6f}")

    summary = acc_candidate.bootstrap_summary(n_bootstrap=args.n_bootstrap, seed=args.seed)
    kl_ci_low = summary["mean_token_kl_ci_low"]
    kl_ci_high = summary["mean_token_kl_ci_high"]
    if kl_ci_high < args.acceptable_kl:
        verdict = "GO"
    elif kl_ci_low > args.acceptable_kl:
        verdict = "NO-GO"
    else:
        verdict = "INCONCLUSIVE"

    result = {
        "model_key": args.model_key,
        "fp8_scope": args.fp8_scope,
        "num_docs": len(docs),
        "reference_corpus_ppl": acc_reference.corpus_ppl,
        "candidate_corpus_ppl": acc_candidate.corpus_ppl,
        "mean_token_kl": summary["mean_token_kl"],
        "mean_token_kl_ci_low": kl_ci_low,
        "mean_token_kl_ci_high": kl_ci_high,
        "acceptable_kl_threshold": args.acceptable_kl,
        "known_bad_communication_int4_kl_anchor": KNOWN_BAD_COMMUNICATION_INT4_KL_ANCHOR,
        "verdict": verdict,
    }

    pd_rows = per_doc_rows
    (out / "per_document.json").write_text(json.dumps(pd_rows, indent=2), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")

    lines = [
        f"# Energy-SLO FP8-Compute Quality Confound Gate ({args.model_key}, scope={args.fp8_scope})", "",
        f"- documents: {len(docs)} (dataset={args.dataset}, offset={args.offset})",
        f"- reference corpus PPL: {result['reference_corpus_ppl']:.4f}",
        f"- candidate (FP8-compute) corpus PPL: {result['candidate_corpus_ppl']:.4f}",
        f"- mean token KL: {result['mean_token_kl']:.6f}  (95% CI [{kl_ci_low:.6f}, {kl_ci_high:.6f}])",
        f"- acceptable KL threshold (this script's judgment call): {args.acceptable_kl}",
        f"- known-bad communication-INT4 KL anchor (qualitative only, NOT directly comparable): "
        f"{KNOWN_BAD_COMMUNICATION_INT4_KL_ANCHOR}",
        f"- VERDICT: {verdict}",
        "",
        "Evidence boundary: single-GPU real FP8 tensor-core compute on expert FFN matmuls only "
        "(attention/router/LM-head untouched); per-tensor absmax scaling; routing divergence at "
        "deeper layers is a real, expected downstream effect and is included in the KL number.",
    ]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nsaved to {out}")


if __name__ == "__main__":
    main()
