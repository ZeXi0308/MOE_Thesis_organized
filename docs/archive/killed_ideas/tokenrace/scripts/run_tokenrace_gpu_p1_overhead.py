#!/usr/bin/env python3
"""TokenRace-EP GPU P1: two remaining real-hardware overhead sources that
the P0 best-case estimate deliberately left out.

(D) CUDA Graph replay vs eager-mode cost. full_barrier launches ONE fixed
    shape per decode step (batch size is drawn from a small enumerable
    set, e.g. {32,64,128}) -> it CAN be served by a small library of
    pre-captured CUDA graphs, replayed at near-zero CPU dispatch cost.
    token_race needs a NEW, unpredictable sub-batch shape every release
    event -> capturing a graph for every possible shape on the fly is not
    practical (capture itself costs far more than one eager call), so in
    practice token_race is forced into eager mode. This measures the real
    gap between "eager" (what token_race is stuck with) and "graph replay"
    (what full_barrier can exploit), i.e. an EXTRA structural disadvantage
    for token_race that P0 did not charge it for.

(E) More realistic multi-group release overhead. P0 charged token_race the
    cheapest possible case: exactly 1 extra launch + 1 gather per layer
    (2 release groups: "early" + "late"). This re-measures the cost for
    3 and 4 release groups (representing a more realistic distribution of
    finish times within a batch) to see how fast the tax grows.

Both feed back into the same real-route-trace gate recomputation used in
run_tokenrace_gpu_microbench.py.
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
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_tokenrace_gpu_microbench import (  # noqa: E402
    SwiGLUExpert, MODEL_DIMS, load_route_index, simulate_decode_step_real,
    NOISE_REGIMES, MIN_IMPROVEMENT, pctl,
)


def measure_graph_vs_eager(hidden: int, intermediate: int, n_tokens: int,
                            n_repeat: int, device="cuda") -> dict:
    module = SwiGLUExpert(hidden, intermediate, device=device)
    module.eval()
    x = torch.randn(n_tokens, hidden, dtype=torch.bfloat16, device=device)

    # eager timing
    with torch.no_grad():
        for _ in range(30):
            _ = module(x)
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        eager_ms = []
        for _ in range(n_repeat):
            start.record(); _ = module(x); end.record()
            torch.cuda.synchronize()
            eager_ms.append(start.elapsed_time(end))
    eager_us = float(np.median(eager_ms)) * 1000.0

    # CUDA graph capture + replay timing
    static_x = x.clone()
    with torch.no_grad():
        for _ in range(5):
            _ = module(static_x)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                _ = module(static_x)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        with torch.cuda.graph(g):
            static_out = module(static_x)
        torch.cuda.synchronize()

        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        graph_ms = []
        for _ in range(n_repeat):
            start.record(); g.replay(); end.record()
            torch.cuda.synchronize()
            graph_ms.append(start.elapsed_time(end))
    graph_us = float(np.median(graph_ms)) * 1000.0

    return {"n_tokens": n_tokens, "eager_us": eager_us, "graph_replay_us": graph_us,
            "graph_advantage_us": eager_us - graph_us}


def measure_multigroup_rebatch(hidden: int, batch_size: int, n_groups_list: list[int],
                                n_repeat: int, device="cuda") -> dict:
    """Cost of splitting a batch into K groups: K index_select gathers +
    K bare-kernel-launch-equivalent dispatches (approximated by K trivial
    elementwise ops back-to-back, matching the earlier bare_launch floor
    methodology)."""
    x = torch.randn(batch_size, hidden, dtype=torch.bfloat16, device=device)
    out = {}
    for n_groups in n_groups_list:
        # pre-split indices for n_groups roughly equal partitions
        perm = torch.randperm(batch_size, device=device)
        splits = torch.chunk(perm, n_groups)

        def op():
            total = None
            for idx in splits:
                sub = torch.index_select(x, 0, idx)
                launch_marker = sub[:1, :1] + 1.0  # trivial op = 1 extra kernel launch marker
                total = launch_marker if total is None else total
            return total

        for _ in range(20):
            _ = op()
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        times_ms = []
        for _ in range(n_repeat):
            start.record(); _ = op(); end.record()
            torch.cuda.synchronize()
            times_ms.append(start.elapsed_time(end))
        us = float(np.median(times_ms)) * 1000.0
        out[str(n_groups)] = us
    return out


def recompute_gate_with_overhead(root: Path, rebatch_overhead_us: float,
                                  base_consts: dict, n_decode_steps: int,
                                  batch_sizes: list[int], seed: int,
                                  barrier_graph_credit_us_per_layer: float = 0.0,
                                  rebatch_mode: str = "unconditional") -> dict:
    rng = np.random.default_rng(seed)
    all_results = {}
    for model_key, spec in MODEL_DIMS.items():
        csv_path = root / spec["route_csv"]
        route_index = load_route_index(csv_path)
        all_keys = list(route_index.keys())
        base_us = base_consts[model_key]["fit_base_us"]
        per_token_us = base_consts[model_key]["fit_per_token_us"]
        attn_shared_us = base_us * 0.5

        res = {"regimes": {}}
        for regime_name, regime in NOISE_REGIMES.items():
            if regime_name != "moderate":
                continue
            regime_out = {}
            for B in batch_sizes:
                barrier_vals, race_vals = [], []
                for _ in range(n_decode_steps):
                    idx = rng.choice(len(all_keys), size=B, replace=False)
                    batch_keys = [all_keys[i] for i in idx]
                    b_total, r_totals = simulate_decode_step_real(
                        batch_keys, route_index, spec["n_layers"], regime, rng,
                        base_us, per_token_us, rebatch_overhead_us, attn_shared_us,
                        barrier_graph_credit_us_per_layer=barrier_graph_credit_us_per_layer,
                        rebatch_mode=rebatch_mode,
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
                }
            res["regimes"]["moderate"] = regime_out
        all_results[model_key] = res
    return all_results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).resolve().parent))
    ap.add_argument("--n-repeat", type=int, default=200)
    ap.add_argument("--n-decode-steps", type=int, default=1000)
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[32, 64, 128])
    ap.add_argument("--seed", type=int, default=20260719)
    ap.add_argument("--output-dir", default="outputs/tokenrace_gpu_p1_2026-07-19")
    args = ap.parse_args()

    device = "cuda"
    root = Path(args.root)
    out_dir = root / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[gpu] {torch.cuda.get_device_name(0)}", file=sys.stderr)

    # reuse base fits from P0 by re-measuring quickly (cheap, keeps this script standalone)
    from run_tokenrace_gpu_microbench import measure_ffn_scaling
    token_counts = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    print("[re-fit] base FFN constants...", file=sys.stderr)
    olmoe_fit = measure_ffn_scaling(MODEL_DIMS["olmoe"]["hidden"], MODEL_DIMS["olmoe"]["intermediate"], token_counts, args.n_repeat, device)
    llmjp_fit = measure_ffn_scaling(MODEL_DIMS["llmjp"]["hidden"], MODEL_DIMS["llmjp"]["intermediate"], token_counts, args.n_repeat, device)
    base_consts = {"olmoe": olmoe_fit, "llmjp": llmjp_fit}
    print(f"    olmoe base_us={olmoe_fit['fit_base_us']:.2f}  llmjp base_us={llmjp_fit['fit_base_us']:.2f}", file=sys.stderr)

    print("[D] measuring CUDA graph vs eager advantage...", file=sys.stderr)
    graph_results = {}
    for model_key, spec in MODEL_DIMS.items():
        rows = []
        for n in [8, 16, 32, 64]:
            r = measure_graph_vs_eager(spec["hidden"], spec["intermediate"], n, args.n_repeat, device)
            rows.append(r)
            print(f"    {model_key} n={n}: eager={r['eager_us']:.2f}us graph={r['graph_replay_us']:.2f}us advantage={r['graph_advantage_us']:.2f}us", file=sys.stderr)
        graph_results[model_key] = rows

    print("[E] measuring multi-group rebatch overhead...", file=sys.stderr)
    multigroup = {}
    for model_key, spec in MODEL_DIMS.items():
        mg = measure_multigroup_rebatch(spec["hidden"], 128, [2, 3, 4, 6], args.n_repeat, device)
        multigroup[model_key] = mg
        print(f"    {model_key} groups->us: {mg}", file=sys.stderr)

    # Recompute gate under corrected overhead scenarios:
    #  (i) 2-group rebatch, no CUDA graph asymmetry (eager-vs-eager)
    #  (ii_CORRECTED) barrier gets graph credit per layer; race pays 2-group
    #       rebatch only (do NOT add graph_advantage into rebatch_overhead)
    #  (iii) 4-group rebatch, no graph asymmetry
    # Legacy (ii) that added graph_advantage into race overhead is incorrect
    # for the improvement ratio denominator; kept only in offline errata recompute.
    print("[C'] recomputing gate under additional overhead scenarios...", file=sys.stderr)
    avg_graph_advantage = np.mean([
        np.mean([r["graph_advantage_us"] for r in graph_results["olmoe"]]),
        np.mean([r["graph_advantage_us"] for r in graph_results["llmjp"]]),
    ])
    rebatch_2 = float((multigroup["olmoe"]["2"] + multigroup["llmjp"]["2"]) / 2.0)
    scenario_iii_overhead = float((multigroup["olmoe"]["4"] + multigroup["llmjp"]["4"]) / 2.0)

    gate_ii_corrected = recompute_gate_with_overhead(
        root, rebatch_2, base_consts, args.n_decode_steps, args.batch_sizes, args.seed,
        barrier_graph_credit_us_per_layer=float(avg_graph_advantage),
        rebatch_mode="early_only",
    )
    gate_iii = recompute_gate_with_overhead(
        root, scenario_iii_overhead, base_consts, args.n_decode_steps, args.batch_sizes, args.seed,
        barrier_graph_credit_us_per_layer=0.0,
        rebatch_mode="early_only",
    )
    # Legacy incorrect placement retained for comparison only
    scenario_ii_legacy_overhead = rebatch_2 + float(avg_graph_advantage)
    gate_ii_legacy = recompute_gate_with_overhead(
        root, scenario_ii_legacy_overhead, base_consts, args.n_decode_steps, args.batch_sizes, args.seed,
        barrier_graph_credit_us_per_layer=0.0,
        rebatch_mode="unconditional",
    )

    out = {
        "base_consts": base_consts,
        "graph_results": graph_results,
        "multigroup_rebatch_us": multigroup,
        "avg_graph_advantage_us": float(avg_graph_advantage),
        "scenario_ii_overhead_us(2group+cudagraph_loss)_LEGACY_INCORRECT": scenario_ii_legacy_overhead,
        "scenario_iii_overhead_us(4group_no_graph_loss)": scenario_iii_overhead,
        "gate_scenario_ii_corrected_barrier_credit_early_only": gate_ii_corrected,
        "gate_scenario_ii_legacy_graph_on_race_unconditional": gate_ii_legacy,
        "gate_scenario_iii": gate_iii,
        "accounting_note": (
            "2026-07-21: graph advantage must credit barrier, not race overhead. "
            "See analyze_tokenrace_p1_corrected_recompute.py and 判死结论勘误."
        ),
    }
    with open(out_dir / "gpu_p1_results.json", "w") as f:
        json.dump(out, f, indent=2)

    lines = ["# TokenRace-EP GPU P1：CUDA Graph代价 + 多组释放代价\n"]
    lines.append(f"GPU: {torch.cuda.get_device_name(0)}\n")
    lines.append(f"平均CUDA graph replay相对eager的优势: {avg_graph_advantage:.2f}us/次调用\n")
    lines.append(
        "Accounting (2026-07-21): graph credit is applied to the **barrier** path; "
        "do not add it into race rebatch overhead.\n"
    )
    lines.append("## 多组重批开销 (us, batch=128)\n")
    for mk, mg in multigroup.items():
        lines.append(f"- {mk}: " + ", ".join(f"{k}组={v:.2f}us" for k, v in mg.items()))
    lines.append(
        f"\n## 场景(ii) CORRECTED: barrier graph credit + early-only 2-group rebatch "
        f"(rebatch={rebatch_2:.2f}us, graph_credit={avg_graph_advantage:.2f}us)\n"
    )
    lines.append("| model | B | P50改善 | P99改善 |")
    lines.append("|---|---|---|---|")
    for mk, res in gate_ii_corrected.items():
        for B, s in res["regimes"]["moderate"].items():
            lines.append(f"| {mk} | {B} | {s['improvement_p50']*100:.2f}% | {s['improvement_p99']*100:.2f}% |")
    lines.append(
        f"\n## 场景(ii) LEGACY (incorrect): graph added to race overhead={scenario_ii_legacy_overhead:.2f}us\n"
    )
    lines.append("| model | B | P50改善 | P99改善 |")
    lines.append("|---|---|---|---|")
    for mk, res in gate_ii_legacy.items():
        for B, s in res["regimes"]["moderate"].items():
            lines.append(f"| {mk} | {B} | {s['improvement_p50']*100:.2f}% | {s['improvement_p99']*100:.2f}% |")
    lines.append(f"\n## 场景(iii): 4组重批 early-only，无CUDA graph代价 (overhead={scenario_iii_overhead:.2f}us/层)\n")
    lines.append("| model | B | P50改善 | P99改善 |")
    lines.append("|---|---|---|---|")
    for mk, res in gate_iii.items():
        for B, s in res["regimes"]["moderate"].items():
            lines.append(f"| {mk} | {B} | {s['improvement_p50']*100:.2f}% | {s['improvement_p99']*100:.2f}% |")
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[done] -> {out_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
