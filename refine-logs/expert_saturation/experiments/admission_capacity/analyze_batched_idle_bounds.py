"""Does a batched decode step leave any expert idle at all?

This checks the load-bearing arithmetic behind the "expert residency tax" claim in
`20260908_kv_pressure_probe_r01/REPORT.md`. That report computed idle expert bytes
as `expert_bytes * (1 - k/E)` = 10.50 GiB, using the per-token activation ratio
8/64. In batched serving that is the wrong unit: a decode step advances B requests
at once, and the layer must hold every expert selected by *any* of those B tokens.

Under an independent-uniform routing null, the chance a given expert is selected
by no token in the step is `(1 - k/E)^B`, so expected idle bytes shrink fast with
B. At the batch widths actually measured in this repository (16-32) the null
predicts almost nothing is idle.

Real MoE routing is not uniform: routers concentrate load on hot experts, so the
true idle count should sit somewhere between the per-token bound and the uniform
null. This script reports both bounds plus the gap that measurement must fill. It
deliberately does not claim a value for real routing -- that requires a route
trace, which is the experiment this analysis motivates.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

GIB = 2 ** 30


def uniform_null_idle_fraction(experts_total, experts_per_token, batch_width):
    """P(a given expert is selected by no token) under independent uniform routing."""
    if batch_width < 1:
        raise ValueError("batch width must be at least one")
    return (1.0 - experts_per_token / experts_total) ** batch_width


def concentrated_bound(experts_total, experts_per_token, batch_width):
    """Best case for reclamation: every token picks the SAME k experts.

    This is the opposite extreme from the uniform null and is the maximum idle
    fraction any routing distribution can produce at this batch width."""
    active = min(experts_total, experts_per_token)
    return 1.0 - active / experts_total


def worst_case_active(experts_total, experts_per_token, batch_width):
    """Minimum idle fraction: tokens pick disjoint experts until saturation."""
    active = min(experts_total, experts_per_token * batch_width)
    return 1.0 - active / experts_total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--expert-bytes", type=int, default=12884901888,
                        help="measured expert_w13_w2_storage_bytes")
    parser.add_argument("--experts-total", type=int, default=64)
    parser.add_argument("--experts-per-token", type=int, default=8)
    parser.add_argument("--kv-bytes", type=int, default=16101933056,
                        help="measured kv_storage_bytes for reference")
    parser.add_argument("--widths", default="1,2,4,8,12,16,24,32")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)

    widths = [int(v) for v in args.widths.split(",")]
    per_token_fraction = 1.0 - args.experts_per_token / args.experts_total
    rows = []
    for width in widths:
        uniform = uniform_null_idle_fraction(args.experts_total, args.experts_per_token, width)
        floor = worst_case_active(args.experts_total, args.experts_per_token, width)
        ceiling = concentrated_bound(args.experts_total, args.experts_per_token, width)
        rows.append(dict(
            batch_width=width,
            uniform_null_idle_fraction=uniform,
            uniform_null_idle_experts=uniform * args.experts_total,
            uniform_null_idle_gib=args.expert_bytes * uniform / GIB,
            disjoint_floor_idle_fraction=floor,
            disjoint_floor_idle_gib=args.expert_bytes * floor / GIB,
            fully_concentrated_ceiling_idle_fraction=ceiling,
            fully_concentrated_ceiling_idle_gib=args.expert_bytes * ceiling / GIB,
            uniform_share_of_per_token_claim=uniform / per_token_fraction,
            uniform_null_idle_as_share_of_kv=args.expert_bytes * uniform / args.kv_bytes))

    report = dict(
        evidence_type="ARITHMETIC_BOUND_ANALYSIS_NO_MEASUREMENT_NO_EXECUTION",
        purpose=("corrects the per-token idle-expert arithmetic used in "
                 "20260908_kv_pressure_probe_r01 and bounds what a route trace could show"),
        claim_boundary=(
            "the uniform column is a NULL MODEL, not a measurement of this router; "
            "the floor and ceiling columns bracket every possible routing distribution; "
            "real batched idle fraction must be measured from a route trace"),
        measured_inputs=dict(expert_bytes=args.expert_bytes, kv_bytes=args.kv_bytes,
                             expert_gib=args.expert_bytes / GIB, kv_gib=args.kv_bytes / GIB),
        model=dict(experts_total=args.experts_total, experts_per_token=args.experts_per_token,
                   per_token_idle_fraction=per_token_fraction,
                   per_token_idle_gib=args.expert_bytes * per_token_fraction / GIB),
        rows=rows,
        superseded_claim=dict(
            source="20260908_kv_pressure_probe_r01/REPORT.md",
            stated="idle expert 10.50 GiB = 70.0% of KV region",
            defect="used per-token 8/64 ratio; a decode step serves B tokens jointly",
            corrected_status="upper bound valid only at batch width 1"))
    (out / "bounds.json").write_text(json.dumps(report, indent=2) + "\n")

    lines = ["# Batched idle-expert bounds (arithmetic only)", "",
             f'measured expert weights {args.expert_bytes / GIB:.2f} GiB, '
             f'KV region {args.kv_bytes / GIB:.2f} GiB, top-k '
             f'{args.experts_per_token}/{args.experts_total}', "",
             "The `uniform null` column is a null model, **not** this router's behaviour.",
             "`disjoint floor` and `concentrated ceiling` bracket every possible routing.", "",
             "| batch width | uniform null idle | idle experts | idle GiB | vs per-token claim | "
             "floor GiB | ceiling GiB |", "|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(
            f'| {r["batch_width"]} | {r["uniform_null_idle_fraction"]:.4f} | '
            f'{r["uniform_null_idle_experts"]:.1f} | {r["uniform_null_idle_gib"]:.2f} | '
            f'{r["uniform_share_of_per_token_claim"]:.1%} | '
            f'{r["disjoint_floor_idle_gib"]:.2f} | '
            f'{r["fully_concentrated_ceiling_idle_gib"]:.2f} |')
    lines += ["", "## What this corrects", "",
              f'- `20260908_kv_pressure_probe_r01` reported idle expert '
              f'{args.expert_bytes * per_token_fraction / GIB:.2f} GiB using the per-token ratio.',
              "- That value is an upper bound valid only at batch width 1.",
              "- At the batch widths this repository actually measured (16-32) the uniform null",
              f'  predicts {rows[-3]["uniform_null_idle_gib"]:.2f}-{rows[-1]["uniform_null_idle_gib"]:.2f} GiB,',
              f'  i.e. {rows[-1]["uniform_share_of_per_token_claim"]:.1%}-'
              f'{rows[-3]["uniform_share_of_per_token_claim"]:.1%} of the reported figure.',
              "", "## What still has to be measured", "",
              "Real routers concentrate load, so the true idle fraction lies between the floor",
              "and ceiling columns and cannot be derived here. The decisive quantity is the",
              "per-step, per-layer **union** of experts selected across all tokens in the batch,",
              "which requires a route trace from the live engine."]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {out}")
    for r in rows:
        if r["batch_width"] in (1, 8, 16, 32):
            print(f'  B={r["batch_width"]:2d}: uniform-null idle {r["uniform_null_idle_gib"]:5.2f} GiB '
                  f'({r["uniform_share_of_per_token_claim"]:6.1%} of the per-token claim)')


if __name__ == "__main__":
    main()
