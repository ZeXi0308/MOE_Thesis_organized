#!/usr/bin/env python3
"""TokenRace-EP GPU P0: real-hardware kernel-launch / re-batch overhead
microbenchmark on RTX 5090, feeding measured constants back into the
Mac P0 simulation model (run_tokenrace_ep_p0.py) to recompute the
pre-registered 5% kill gate with real numbers instead of illustrative
Hypothesis-level constants.

This is the single decisive experiment identified in
TokenRace-EP_严格性审查与后续方向_2026-07-19.md: everything in the Mac P0/P0-B
results assumed "re-batching (dynamically pulling completed tokens out of a
batch and launching a smaller forward for them) is free". This script
measures whether that is true on a real GPU.

Three measurements, all [Observed] on real hardware (RTX 5090, CUDA 12.8):

  (A) Expert FFN forward wall-clock time as a function of token count, for
      both OLMoE (hidden=2048, intermediate=1024, SwiGLU) and LLM-jp
      (hidden=512, intermediate=1024, SwiGLU) real dims. Linear fit gives
      REAL base_launch_us and REAL per_token_us, replacing the Mac
      Hypothesis-level constants (BASE_LAUNCH_US=6.0, PER_TOKEN_US=0.35).

  (B) Dynamic gather (index_select) overhead: the real cost of extracting a
      subset of "already-finished" tokens out of a larger tensor -- this is
      the re-batching operation TokenRace-EP needs but full_barrier does
      not.

  (C) Re-run the SAME real-route-trace P0 simulation logic
      (identical to run_tokenrace_ep_p0.simulate_decode_step) with:
        - REAL base_launch_us / per_token_us from (A) instead of illustrative
          constants,
        - a best-case-minimal re-batch penalty added ONLY to token_race:
          exactly one extra kernel launch + one gather per layer (the
          cheapest possible implementation: split each layer's requests
          into just two release groups, "early" and "late"). full_barrier
          pays zero extra cost (it already launches once per layer for the
          whole batch).
      Recompute the pre-registered 5% P99 improvement kill gate under this
      best-case overhead assumption. If the gate still fails under the
      BEST case, the mechanism is dead. If it still passes, more overhead
      cases (>2 release groups) still need checking before any claim can be
      made -- this is a lower bound on the overhead penalty, not an upper
      bound.
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
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Model FFN dims, read from real HF configs (allenai/OLMoE-1B-7B-0924 and
# llm-jp/optimal-sparsity-math-d512-E32-k16-920M-A520M) on 2026-07-19.
# ---------------------------------------------------------------------------
MODEL_DIMS = {
    "olmoe": dict(hidden=2048, intermediate=1024, num_experts=64, top_k=8,
                  route_csv="olmoe_test_routes.csv", n_layers=16),
    "llmjp": dict(hidden=512, intermediate=1024, num_experts=32, top_k=16,
                  route_csv="llmjp_test_routes.csv", n_layers=16),
}

NOISE_REGIMES = {
    "none": dict(sigma=0.0, p_strag=0.0, strag_lo=1.0, strag_hi=1.0),
    "moderate": dict(sigma=0.07, p_strag=0.02, strag_lo=1.5, strag_hi=2.5),
    "severe": dict(sigma=0.15, p_strag=0.05, strag_lo=2.0, strag_hi=4.0),
}
MIN_IMPROVEMENT = 0.05


class SwiGLUExpert(nn.Module):
    """Matches OLMoE/Mixtral-style expert FFN: down(silu(gate(x)) * up(x))."""

    def __init__(self, hidden: int, intermediate: int, dtype=torch.bfloat16, device="cuda"):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False, dtype=dtype, device=device)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False, dtype=dtype, device=device)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False, dtype=dtype, device=device)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))


def timed_forward(module, x, n_repeat: int, warmup: int = 20) -> float:
    """Returns mean wall-clock time in microseconds per forward call,
    measured with CUDA events (device-side timing, not just CPU dispatch)."""
    for _ in range(warmup):
        _ = module(x)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times_ms = []
    for _ in range(n_repeat):
        start.record()
        _ = module(x)
        end.record()
        torch.cuda.synchronize()
        times_ms.append(start.elapsed_time(end))
    return float(np.median(times_ms)) * 1000.0  # ms -> us


def measure_ffn_scaling(hidden: int, intermediate: int, token_counts: list[int],
                         n_repeat: int, device="cuda") -> dict:
    """(A) Measure real forward time vs token count, fit base+per_token."""
    module = SwiGLUExpert(hidden, intermediate, device=device)
    module.eval()
    rows = []
    with torch.no_grad():
        for n in token_counts:
            x = torch.randn(n, hidden, dtype=torch.bfloat16, device=device)
            us = timed_forward(module, x, n_repeat)
            rows.append((n, us))
    ns = np.array([r[0] for r in rows], dtype=np.float64)
    us = np.array([r[1] for r in rows], dtype=np.float64)
    # linear fit: us = base + per_token * n
    A = np.vstack([np.ones_like(ns), ns]).T
    coef, *_ = np.linalg.lstsq(A, us, rcond=None)
    base_us, per_token_us = float(coef[0]), float(coef[1])
    resid = us - (base_us + per_token_us * ns)
    r2 = 1.0 - float(np.sum(resid ** 2) / np.sum((us - us.mean()) ** 2))
    return {
        "raw": [{"n_tokens": int(n), "us": float(u)} for n, u in rows],
        "fit_base_us": base_us,
        "fit_per_token_us": per_token_us,
        "fit_r2": r2,
    }


def measure_gather_overhead(hidden: int, batch_size: int, keep_fracs: list[float],
                             n_repeat: int, device="cuda") -> dict:
    """(B) Measure real cost of index_select-ing a dynamic subset of rows
    out of a batch tensor (the "pull completed tokens out" operation)."""
    x = torch.randn(batch_size, hidden, dtype=torch.bfloat16, device=device)
    rows = []
    for frac in keep_fracs:
        k = max(1, int(round(batch_size * frac)))
        idx = torch.randperm(batch_size, device=device)[:k]

        def op():
            return torch.index_select(x, 0, idx)

        for _ in range(20):
            _ = op()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        times_ms = []
        for _ in range(n_repeat):
            start.record()
            _ = op()
            end.record()
            torch.cuda.synchronize()
            times_ms.append(start.elapsed_time(end))
        us = float(np.median(times_ms)) * 1000.0
        rows.append({"batch_size": batch_size, "keep_frac": frac, "k": k, "us": us})
    return {"raw": rows, "mean_us": float(np.mean([r["us"] for r in rows]))}


def measure_bare_launch_overhead(n_repeat: int, device="cuda") -> float:
    """Pure kernel-launch overhead floor: a trivial 1x1 op, back-to-back,
    CUDA-event timed. Lower bound on 'one extra launch' cost."""
    x = torch.randn(1, 1, dtype=torch.bfloat16, device=device)
    for _ in range(50):
        _ = x + 1.0
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times_ms = []
    for _ in range(n_repeat):
        start.record()
        _ = x + 1.0
        end.record()
        torch.cuda.synchronize()
        times_ms.append(start.elapsed_time(end))
    return float(np.median(times_ms)) * 1000.0


# ---------------------------------------------------------------------------
# (C) Re-run the real-route-trace decode-step simulation with real constants
# plus best-case-minimal re-batch overhead for token_race.
# ---------------------------------------------------------------------------

def load_route_index(csv_path: Path):
    df = pd.read_csv(csv_path)
    index: dict[tuple[int, int], dict[int, list[int]]] = {}
    for (sample_id, token_position), grp in df.groupby(["sample_id", "token_position"], sort=False):
        by_layer: dict[int, list[int]] = {}
        for layer, sub in grp.groupby("layer", sort=False):
            by_layer[int(layer)] = sub["expert_id"].to_numpy().tolist()
        index[(int(sample_id), int(token_position))] = by_layer
    return index


def simulate_decode_step_real(
    batch_keys, route_index, n_layers, regime, rng,
    base_us: float, per_token_us: float,
    rebatch_overhead_us: float, attn_shared_us: float,
    barrier_graph_credit_us_per_layer: float = 0.0,
    rebatch_mode: str = "unconditional",
):
    """Simulate one decode step.

    barrier_graph_credit_us_per_layer
        Subtracted from the barrier path each layer (correct placement of
        CUDA-graph advantage). Do **not** add this into rebatch_overhead_us.

    rebatch_mode
        - ``unconditional``: legacy P0/P1 — every request pays own+rebatch
        - ``early_only``: early requests (own < barrier) pay own+rebatch;
          late requests pay barrier with no rebatch (aligned with P2 fallback)
    """
    n_req = len(batch_keys)
    barrier_total = attn_shared_us * n_layers
    race_totals = np.full(n_req, attn_shared_us * n_layers, dtype=np.float64)

    for layer in range(n_layers):
        expert_counts: dict[int, int] = {}
        req_experts: list[list[int]] = []
        for key in batch_keys:
            experts_this_layer = route_index[key].get(layer, [])
            req_experts.append(experts_this_layer)
            for e in experts_this_layer:
                expert_counts[e] = expert_counts.get(e, 0) + 1

        finish_time: dict[int, float] = {}
        for e, cnt in expert_counts.items():
            base = base_us + per_token_us * cnt
            noise = rng.normal(0.0, regime["sigma"]) if regime["sigma"] > 0 else 0.0
            mult = max(0.1, 1.0 + noise)
            if regime["p_strag"] > 0 and rng.random() < regime["p_strag"]:
                mult *= rng.uniform(regime["strag_lo"], regime["strag_hi"])
            finish_time[e] = base * mult

        layer_barrier = max(finish_time.values()) if finish_time else 0.0
        # full_barrier: one launch; optionally credit CUDA-graph speedup on barrier only
        barrier_total += max(0.0, layer_barrier - barrier_graph_credit_us_per_layer)

        for i, experts_this_layer in enumerate(req_experts):
            if not experts_this_layer:
                continue
            own = max(finish_time[e] for e in experts_this_layer)
            if rebatch_mode == "unconditional":
                # Legacy (pessimistic): every request pays rebatch every layer.
                race_totals[i] += own + rebatch_overhead_us
            elif rebatch_mode == "early_only":
                if own < layer_barrier - 1e-12:
                    race_totals[i] += own + rebatch_overhead_us
                else:
                    race_totals[i] += layer_barrier
            else:
                raise ValueError(f"unknown rebatch_mode={rebatch_mode}")

    return barrier_total, race_totals


def pctl(arr, q):
    return float(np.percentile(arr, q))


def run_gate_recompute(root: Path, real_constants: dict, n_decode_steps: int,
                        batch_sizes: list[int], seed: int) -> dict:
    rng = np.random.default_rng(seed)
    all_results = {}
    for model_key, spec in MODEL_DIMS.items():
        csv_path = root / spec["route_csv"]
        route_index = load_route_index(csv_path)
        all_keys = list(route_index.keys())
        base_us = real_constants[model_key]["fit_base_us"]
        per_token_us = real_constants[model_key]["fit_per_token_us"]
        rebatch_us = real_constants["rebatch_overhead_us"]
        attn_shared_us = real_constants[model_key]["fit_base_us"] * 0.5  # conservative proxy

        res = {"regimes": {}}
        for regime_name, regime in NOISE_REGIMES.items():
            regime_out = {}
            for B in batch_sizes:
                barrier_vals, race_vals = [], []
                for _ in range(n_decode_steps):
                    idx = rng.choice(len(all_keys), size=B, replace=False)
                    batch_keys = [all_keys[i] for i in idx]
                    b_total, r_totals = simulate_decode_step_real(
                        batch_keys, route_index, spec["n_layers"], regime, rng,
                        base_us, per_token_us, rebatch_us, attn_shared_us,
                    )
                    barrier_vals.append(b_total)
                    race_vals.extend(r_totals.tolist())
                b_arr, r_arr = np.array(barrier_vals), np.array(race_vals)

                def imp(fn):
                    b, r = fn(b_arr), fn(r_arr)
                    return (b - r) / b if b > 0 else 0.0

                regime_out[str(B)] = {
                    "improvement_p50": imp(lambda a: pctl(a, 50)),
                    "improvement_p99": imp(lambda a: pctl(a, 99)),
                    "improvement_mean": imp(np.mean),
                }
            res["regimes"][regime_name] = regime_out
        all_results[model_key] = res
    return all_results


def apply_gate(all_results: dict) -> dict:
    verdict = {"gate": "PASS", "reasons": []}
    for model_key, res in all_results.items():
        for B, stats in res["regimes"]["moderate"].items():
            if stats["improvement_p99"] < MIN_IMPROVEMENT:
                verdict["gate"] = "FAIL"
                verdict["reasons"].append(
                    f"{model_key} B={B}: moderate P99 improvement "
                    f"{stats['improvement_p99']*100:.2f}% < gate {MIN_IMPROVEMENT*100:.0f}% "
                    f"(WITH real-hardware best-case rebatch overhead included)"
                )
    if not verdict["reasons"]:
        verdict["reasons"].append("All gates passed even with real-hardware best-case rebatch overhead.")
    return verdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).resolve().parent))
    ap.add_argument("--n-decode-steps", type=int, default=1000)
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[32, 64, 128])
    ap.add_argument("--seed", type=int, default=20260719)
    ap.add_argument("--n-repeat", type=int, default=200)
    ap.add_argument("--output-dir", default="outputs/tokenrace_gpu_p0_2026-07-19")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA required for this script"
    device = "cuda"
    print(f"[gpu] {torch.cuda.get_device_name(0)}", file=sys.stderr)

    root = Path(args.root)
    out_dir = root / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    token_counts = [1, 2, 4, 8, 16, 32, 64, 128, 256]

    print("[A] measuring FFN scaling (OLMoE dims)...", file=sys.stderr)
    olmoe_fit = measure_ffn_scaling(MODEL_DIMS["olmoe"]["hidden"], MODEL_DIMS["olmoe"]["intermediate"],
                                     token_counts, args.n_repeat, device)
    print(f"    base_us={olmoe_fit['fit_base_us']:.3f} per_token_us={olmoe_fit['fit_per_token_us']:.4f} r2={olmoe_fit['fit_r2']:.4f}", file=sys.stderr)

    print("[A] measuring FFN scaling (LLM-jp dims)...", file=sys.stderr)
    llmjp_fit = measure_ffn_scaling(MODEL_DIMS["llmjp"]["hidden"], MODEL_DIMS["llmjp"]["intermediate"],
                                     token_counts, args.n_repeat, device)
    print(f"    base_us={llmjp_fit['fit_base_us']:.3f} per_token_us={llmjp_fit['fit_per_token_us']:.4f} r2={llmjp_fit['fit_r2']:.4f}", file=sys.stderr)

    print("[B] measuring gather/re-batch overhead...", file=sys.stderr)
    gather_olmoe = measure_gather_overhead(MODEL_DIMS["olmoe"]["hidden"], 128, [0.1, 0.25, 0.5, 0.75], args.n_repeat, device)
    gather_llmjp = measure_gather_overhead(MODEL_DIMS["llmjp"]["hidden"], 128, [0.1, 0.25, 0.5, 0.75], args.n_repeat, device)
    print(f"    olmoe gather mean_us={gather_olmoe['mean_us']:.3f}  llmjp gather mean_us={gather_llmjp['mean_us']:.3f}", file=sys.stderr)

    print("[bare] measuring pure kernel-launch floor...", file=sys.stderr)
    bare_launch_us = measure_bare_launch_overhead(args.n_repeat, device)
    print(f"    bare_launch_us={bare_launch_us:.3f}", file=sys.stderr)

    # Best-case-minimal rebatch overhead = 1 extra kernel launch (use the
    # REAL measured bare launch floor, which is <= any real compute kernel's
    # own launch component) + 1 gather op. This is deliberately the most
    # OPTIMISTIC (smallest) possible overhead estimate.
    rebatch_overhead_us = bare_launch_us + (gather_olmoe["mean_us"] + gather_llmjp["mean_us"]) / 2.0
    print(f"[C] best-case rebatch_overhead_us (per layer, per release event) = {rebatch_overhead_us:.3f}", file=sys.stderr)

    real_constants = {
        "olmoe": olmoe_fit,
        "llmjp": llmjp_fit,
        "gather_olmoe": gather_olmoe,
        "gather_llmjp": gather_llmjp,
        "bare_launch_us": bare_launch_us,
        "rebatch_overhead_us": rebatch_overhead_us,
    }

    print("[C] recomputing kill gate on real route traces with real constants...", file=sys.stderr)
    all_results = run_gate_recompute(root, real_constants, args.n_decode_steps, args.batch_sizes, args.seed)
    verdict = apply_gate(all_results)

    out = {
        "real_constants": real_constants,
        "gate_results": all_results,
        "verdict": verdict,
        "gpu_name": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }
    with open(out_dir / "gpu_p0_results.json", "w") as f:
        json.dump(out, f, indent=2)

    lines = ["# TokenRace-EP GPU P0：真实硬件重批开销 + 门槛重算\n"]
    lines.append(f"GPU: {torch.cuda.get_device_name(0)}, torch {torch.__version__}, cuda {torch.version.cuda}\n")
    lines.append(f"**GATE (real hardware, best-case rebatch overhead): {verdict['gate']}**\n")
    for r in verdict["reasons"]:
        lines.append(f"- {r}")
    lines.append("\n## (A) 真实FFN scaling拟合\n")
    lines.append(f"- OLMoE (hidden=2048,inter=1024): base_us={olmoe_fit['fit_base_us']:.3f}, per_token_us={olmoe_fit['fit_per_token_us']:.4f}, R2={olmoe_fit['fit_r2']:.4f}")
    lines.append(f"- LLM-jp (hidden=512,inter=1024): base_us={llmjp_fit['fit_base_us']:.3f}, per_token_us={llmjp_fit['fit_per_token_us']:.4f}, R2={llmjp_fit['fit_r2']:.4f}")
    lines.append(f"- (对比 Mac仿真illustrative值: BASE_LAUNCH_US=6.0, PER_TOKEN_US=0.35)\n")
    lines.append("## (B) 真实gather/重批开销\n")
    lines.append(f"- OLMoE dims gather mean_us={gather_olmoe['mean_us']:.3f}")
    lines.append(f"- LLM-jp dims gather mean_us={gather_llmjp['mean_us']:.3f}")
    lines.append(f"- bare kernel launch floor us={bare_launch_us:.3f}")
    lines.append(f"- **best-case rebatch_overhead_us(每层每次release) = {rebatch_overhead_us:.3f}**\n")
    lines.append("## (C) 门槛重算结果（moderate场景，含best-case重批开销）\n")
    lines.append("| model | B | P50改善 | P99改善 | mean改善 |")
    lines.append("|---|---|---|---|---|")
    for model_key, res in all_results.items():
        for B, s in res["regimes"]["moderate"].items():
            lines.append(f"| {model_key} | {B} | {s['improvement_p50']*100:.2f}% | {s['improvement_p99']*100:.2f}% | {s['improvement_mean']*100:.2f}% |")
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[done] gate={verdict['gate']} -> {out_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
