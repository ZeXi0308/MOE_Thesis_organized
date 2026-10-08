#!/usr/bin/env python3
"""Can reclaimed expert residency supply the KV deficit?

This is the question the expert-union measurement exists to answer, expressed
in the currency of the main problem. The main problem is that long-context MoE
serving on one GPU cannot hold all admitted requests' KV, so the engine
preempts, and 96.3% of the resulting pause is waiting rather than recompute.
If idle expert weights could be reclaimed, the deficit would shrink and the
preemption would not be forced in the first place.

Two quantities decide it, and they differ by more than an order of magnitude:

  * INSTANTANEOUS idle  = 1 - U(layer, step), the per-step headroom. Recovering
    this requires paging experts in and out on the step timescale.
  * STABLE idle         = 1 - U_W(layer), the union over a W-step window. This
    is what a *static* residency decision can recover, because an expert that
    is needed anywhere in the window must stay resident.

Only measured inputs are used: the per-layer union statistics from the
collector, and the engine's own reported KV pool and weight footprint.

What this is NOT
----------------
- Not a paging design and not a transfer-cost model beyond a single bandwidth
  sanity bound. It answers a feasibility question, not an implementation one.
- The byte totals assume every idle expert's bytes are fully reclaimable, which
  no real allocator achieves. The figures are therefore optimistic ceilings.
- Structural: U comes from executed routing, but idle bytes being *unreferenced*
  is not the same as their being *evictable* by this engine's allocator.
"""

import argparse
import json
from pathlib import Path

GIB = 1024 ** 3


def expert_bytes_per_layer(num_experts, hidden, intermediate, bytes_per_param=2,
                           matrices_per_expert=3):
    """OLMoE expert = gate_proj + up_proj + down_proj, each hidden x intermediate."""
    return num_experts * matrices_per_expert * hidden * intermediate * bytes_per_param


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--union-summary", required=True)
    ap.add_argument("--output-dir", required=True)
    # Engine-reported values; defaults are what this environment printed.
    ap.add_argument("--kv-pool-tokens", type=int, default=122752,
                    help="engine: 'GPU KV cache size: N tokens'")
    ap.add_argument("--kv-pool-gib", type=float, default=14.98,
                    help="engine: 'Available KV cache memory: X GiB'")
    ap.add_argument("--max-num-seqs", type=int, default=32)
    ap.add_argument("--context-tokens", type=int, default=4096)
    ap.add_argument("--hidden", type=int, default=2048)
    ap.add_argument("--intermediate", type=int, default=1024)
    ap.add_argument("--stable-window", type=int, default=32)
    ap.add_argument("--h2d-bytes-per-s", type=float, default=25e9,
                    help="measured pinned H2D bandwidth; pass the real number")
    ap.add_argument("--step-time-ms", type=float, default=9.9,
                    help="measured pure-decode step time at this width")
    args = ap.parse_args()

    payload = json.load(open(args.union_summary))
    summary = payload["summary"]
    n_experts = payload["experts_total"]
    n_layers = len(summary)

    per_layer_bytes = expert_bytes_per_layer(n_experts, args.hidden, args.intermediate)
    expert_total_bytes = per_layer_bytes * n_layers

    # --- the deficit, in bytes ---
    kv_bytes_per_token = args.kv_pool_gib * GIB / args.kv_pool_tokens
    demand_tokens = args.max_num_seqs * args.context_tokens
    deficit_tokens = demand_tokens - args.kv_pool_tokens
    deficit_bytes = deficit_tokens * kv_bytes_per_token

    # --- the two idle budgets ---
    rows = []
    inst_sum = stable_sum = 0.0
    for key in sorted(summary, key=int):
        v = summary[key]
        horizons = v.get("horizon_union_p50") or {}
        u_w = horizons.get(str(args.stable_window))
        inst = v["idle_fraction_p50"]
        stable = None if u_w is None else max(0.0, 1.0 - u_w)
        inst_sum += inst
        stable_sum += stable or 0.0
        rows.append(dict(layer=int(key), median_width=v["median_batch_width"],
                         idle_instantaneous=inst, idle_stable=stable,
                         saturation_99=v.get("saturation_window_99"),
                         inst_bytes=inst * per_layer_bytes,
                         stable_bytes=(stable or 0.0) * per_layer_bytes))

    inst_bytes = inst_sum * per_layer_bytes
    stable_bytes = stable_sum * per_layer_bytes

    # --- what would it cost to actually harvest each budget? ---
    # An idle set with coherence window W must be re-staged every W steps, so the
    # amortised transfer time per step is bytes(W) / bandwidth / W. Comparing that
    # against the measured step time decides feasibility before any paging design.
    #
    # Two corrections that decide the verdict and were missing at first:
    #
    #  1. The step time must come from the SAME operating regime as the deficit.
    #     A 128-token-context step (9.9 ms) and a 4096-token-context step
    #     (19.7 ms) differ 2x, and using the wrong one halves or doubles every
    #     overhead ratio below.
    #  2. Paging needs a staging buffer, and that buffer is charged to the SAME
    #     budget the reclamation is trying to free. Transfers are per layer, so a
    #     double buffer of the largest per-layer moved set is the floor. Ignoring
    #     it would credit the mechanism with memory it must itself hold.
    windows = []
    for w in (1, 2, 4, 8, 16, 32):
        total = 0.0
        worst_layer = 0.0
        complete = True
        for key in sorted(summary, key=int):
            h = (summary[key].get("horizon_union_p50") or {}).get(str(w))
            if h is None:
                complete = False
                break
            idle = max(0.0, 1.0 - h)
            total += idle
            worst_layer = max(worst_layer, idle)
        if not complete:
            continue
        gross = total * per_layer_bytes
        # Floor on M_staging: double buffer of the largest per-layer moved set.
        staging = 2.0 * worst_layer * per_layer_bytes
        net = gross - staging
        secs = gross / args.h2d_bytes_per_s
        windows.append(dict(
            window=w, layer_equivalents=total,
            gross_gib=gross / GIB, staging_gib=staging / GIB, net_gib=net / GIB,
            gross_covers_deficit=gross / deficit_bytes,
            net_covers_deficit=net / deficit_bytes,
            transfer_ms=secs * 1000.0,
            amortised_ms_per_step=secs * 1000.0 / w,
            step_time_ms=args.step_time_ms,
            # Fraction of the per-step PCIe time budget consumed. Below 1.0 the
            # transfer could in principle hide behind compute with a dedicated
            # stream; this is the optimistic reading.
            pcie_duty_cycle=(secs * 1000.0 / w) / args.step_time_ms))

    result = dict(
        n_layers=n_layers, n_experts=n_experts,
        expert_bytes_per_layer=per_layer_bytes,
        expert_bytes_total=expert_total_bytes,
        expert_gib_total=expert_total_bytes / GIB,
        kv_bytes_per_token=kv_bytes_per_token,
        kv_pool_tokens=args.kv_pool_tokens,
        demand_tokens=demand_tokens, deficit_tokens=deficit_tokens,
        deficit_gib=deficit_bytes / GIB,
        deficit_requests=deficit_tokens / args.context_tokens,
        instantaneous=dict(
            layer_equivalents=inst_sum, mean_fraction=inst_sum / n_layers,
            gib=inst_bytes / GIB,
            covers_deficit_fraction=inst_bytes / deficit_bytes,
            tokens_bought=inst_bytes / kv_bytes_per_token,
            requests_bought=inst_bytes / kv_bytes_per_token / args.context_tokens),
        stable=dict(
            window=args.stable_window,
            layer_equivalents=stable_sum, mean_fraction=stable_sum / n_layers,
            gib=stable_bytes / GIB,
            covers_deficit_fraction=stable_bytes / deficit_bytes,
            tokens_bought=stable_bytes / kv_bytes_per_token,
            requests_bought=stable_bytes / kv_bytes_per_token / args.context_tokens),
        collapse_ratio=(inst_sum / stable_sum) if stable_sum else None,
        harvest_cost=dict(h2d_bytes_per_s=args.h2d_bytes_per_s,
                          step_time_ms=args.step_time_ms, windows=windows),
        per_layer=rows)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    json.dump(result, open(out / "budget.json", "w"), indent=1)

    lines = ["# Can reclaimed expert residency supply the KV deficit?", "",
             "Measured inputs only: per-layer unions from the collector, and the",
             "engine's own KV pool and weight footprint. Byte totals assume every",
             "idle expert is fully reclaimable, so they are optimistic ceilings.", "",
             "## The deficit", "",
             f'- KV pool: **{args.kv_pool_tokens:,} tokens** ({args.kv_pool_gib} GiB), '
             f'{kv_bytes_per_token/1024:.0f} KiB per token',
             f'- demand at max_num_seqs={args.max_num_seqs}, '
             f'context={args.context_tokens}: {demand_tokens:,} tokens',
             f'- **deficit: {deficit_tokens:,} tokens = {deficit_bytes/GIB:.4f} GiB '
             f'= {deficit_tokens/args.context_tokens:.2f} requests**', "",
             "## The two idle budgets", "",
             "| budget | layer-equivalents | mean fraction | GiB | covers deficit | "
             "KV tokens bought | requests bought |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name, d in (("instantaneous (W=1)", result["instantaneous"]),
                    (f'stable (W={args.stable_window})', result["stable"])):
        lines.append(
            f'| {name} | {d["layer_equivalents"]:.4f} | {d["mean_fraction"]*100:.2f}% | '
            f'**{d["gib"]:.4f}** | **{d["covers_deficit_fraction"]*100:.1f}%** | '
            f'{d["tokens_bought"]:,.0f} | {d["requests_bought"]:.3f} |')
    if result["collapse_ratio"]:
        lines += ["", f'Requiring the idle set to persist for {args.stable_window} steps '
                      f'shrinks it by **{result["collapse_ratio"]:.1f}x**.', ""]
    lines += ["## Cost of harvesting each coherence window", "",
              f'Measured H2D bandwidth {args.h2d_bytes_per_s/1e9:.2f} GB/s; '
              f'measured step time {args.step_time_ms:.3f} ms '
              f'(must be from the same regime as the deficit).',
              "An idle set that stays idle for only W steps must be re-staged every",
              "W steps. `staging` is a double buffer of the largest per-layer moved",
              "set, charged to the same budget the reclamation is freeing.",
              "`PCIe duty` below 1.0 means the transfer could in principle hide",
              "behind compute on a dedicated stream -- the optimistic reading.", "",
              "| W | gross GiB | staging GiB | net GiB | net covers deficit | "
              "transfer ms | amortised ms/step | PCIe duty |",
              "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for w in windows:
        lines.append(f'| {w["window"]} | {w["gross_gib"]:.4f} | {w["staging_gib"]:.4f} | '
                     f'{w["net_gib"]:.4f} | **{w["net_covers_deficit"]*100:.1f}%** | '
                     f'{w["transfer_ms"]:.1f} | {w["amortised_ms_per_step"]:.2f} | '
                     f'**{w["pcie_duty_cycle"]:.2f}x** |')
    feasible = [w for w in windows if w["pcie_duty_cycle"] < 1.0
                and w["net_covers_deficit"] >= 1.0]
    lines += ["", ("**No coherence window both covers the deficit net of staging and "
                   "fits inside one step time.**" if not feasible else
                   f'Feasible windows: {[w["window"] for w in feasible]}'), ""]
    lines += ["## Per layer", "",
              "| layer | width | idle W=1 | idle W=%d | sat99 | idle bytes W=1 (MiB) | "
              "idle bytes stable (MiB) |" % args.stable_window,
              "|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        st = "n/a" if r["idle_stable"] is None else f'{r["idle_stable"]:.4f}'
        lines.append(f'| {r["layer"]} | {r["median_width"]} | '
                     f'{r["idle_instantaneous"]:.4f} | {st} | {r["saturation_99"]} | '
                     f'{r["inst_bytes"]/1024/1024:.1f} | '
                     f'{r["stable_bytes"]/1024/1024:.1f} |')
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
