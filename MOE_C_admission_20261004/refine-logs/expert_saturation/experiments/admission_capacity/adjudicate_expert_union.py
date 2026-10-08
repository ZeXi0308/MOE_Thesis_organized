#!/usr/bin/env python3
"""Adjudicate the three frozen predictions of the expert-union measurement.

The predictions were fixed in `20260910_expert_union_r01/DECISIONS.md` before
any route data existed:

  E1  measured idle fraction EXCEEDS the uniform null (the router concentrates)
  E2  at batch width >= 16 the median idle fraction is < 0.10
  E3  the 99% saturation horizon is <= 4 steps

The expected outcome is `NO_RESIDENCY_HEADROOM` or
`HEADROOM_BUT_IMMEDIATE_REUSE`, which would close a mechanism family the
author had previously argued for, and would explain why FluxMoE v2 loads whole
layers rather than paging missing top-k experts.

E1 and E2 are in tension exactly at width 16, where the uniform null is
0.875^16 = 0.1181 -- just above E2's 0.10 threshold. One of them must lose
there. That is by design: the measurement cannot come out "both correct", so
it is forced to be informative.

This script only reads what the collector wrote. It fits nothing.

What this is NOT
----------------
- U is a STRUCTURAL signal recomputed from router logits. It is not measured
  HBM traffic, not a statement about what the fused backend loads, and not
  evidence that an idle expert's bytes are reclaimable in practice.
- A verdict of NO_RESIDENCY_HEADROOM closes only "top-k-miss paging in this
  operating regime". It does not refute whole-layer streaming, offload
  regimes, or the KV-to-expert direction taken by WiSP.
- Layers are adjudicated individually; a high idle fraction concentrated in a
  few layers must not be averaged across layers into an aggregate claim.
"""

import argparse
import json
import statistics as st
from pathlib import Path

WIDTH_THRESHOLD = 16
IDLE_THRESHOLD = 0.10
SATURATION_HORIZON = 4


def uniform_null_idle(experts_total, experts_per_token, width):
    return (1.0 - experts_per_token / experts_total) ** width


def adjudicate(payload):
    summary = payload["summary"]
    total = payload["experts_total"]
    per_tok = payload["experts_per_token"]

    rows = []
    for layer, v in sorted(summary.items(), key=lambda kv: int(kv[0])):
        width = v["median_batch_width"]
        null = uniform_null_idle(total, per_tok, width)
        rows.append(dict(
            layer=int(layer), n_steps=v["n_steps"], median_width=width,
            idle_p50=v["idle_fraction_p50"], idle_max=v["idle_fraction_max"],
            union_p50=v["union_fraction_p50"],
            uniform_null_idle=null,
            exceeds_null=v["idle_fraction_p50"] > null,
            saturation_99=v["saturation_window_99"],
            saturation_95=v["saturation_window_95"],
            horizon_union_p50=v.get("horizon_union_p50"),
            load_skew=(v.get("load_skew") or {}).get("max_over_mean"),
            experts_never_selected=(v.get("load_skew") or {}).get("experts_never_selected")))

    widths = [r["median_width"] for r in rows]
    wide = [r for r in rows if r["median_width"] >= WIDTH_THRESHOLD]
    scope = wide or rows

    # E1: does the real router concentrate more than an independent uniform one?
    n_exceed = sum(1 for r in rows if r["exceeds_null"])
    e1_holds = n_exceed > len(rows) / 2

    # E2: is there instantaneous headroom at realistic batch width?
    idle_scope = [r["idle_p50"] for r in scope]
    e2_holds = max(idle_scope) < IDLE_THRESHOLD if idle_scope else None

    # E3: is an evicted expert needed again almost immediately?
    sats = [r["saturation_99"] for r in rows]
    resolved = [s for s in sats if s is not None]
    e3_holds = (bool(resolved) and len(resolved) == len(sats)
                and max(resolved) <= SATURATION_HORIZON)

    return dict(
        width_threshold=WIDTH_THRESHOLD, idle_threshold=IDLE_THRESHOLD,
        saturation_horizon=SATURATION_HORIZON,
        experts_total=total, experts_per_token=per_tok,
        n_layers=len(rows), median_batch_width=st.median(widths) if widths else None,
        n_layers_at_or_above_width_threshold=len(wide),
        scope_note=("adjudicated on layers at or above the width threshold"
                    if wide else
                    "NO layer reached the width threshold; adjudicated on all "
                    "layers and E2 must be read as out of its intended scope"),
        E1=dict(prediction="measured idle fraction exceeds the uniform null",
                holds=e1_holds, n_layers_exceeding_null=n_exceed,
                falsified_if="measured <= null, i.e. routing is near uniform"),
        E2=dict(prediction=f"median idle fraction < {IDLE_THRESHOLD} at width "
                           f">= {WIDTH_THRESHOLD}",
                holds=e2_holds,
                max_layer_idle_p50_in_scope=max(idle_scope) if idle_scope else None,
                falsified_if="any layer holds idle fraction at or above the threshold"),
        E3=dict(prediction=f"99% saturation horizon <= {SATURATION_HORIZON} steps",
                holds=e3_holds, saturation_windows=sats,
                falsified_if="a layer does not saturate within the horizon"),
        collector_verdict=payload["verdict"]["verdict"],
        per_layer=rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--union-summary", action="append", required=True,
                    help="union_summary.json written by run_expert_union.py")
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    results = []
    for path in args.union_summary:
        payload = json.load(open(path))
        verdict = adjudicate(payload)
        verdict["source"] = str(path)
        disc = payload.get("collection_discipline") or {}
        verdict["collection_discipline"] = disc
        verdict["truncation"] = payload.get("truncation") or {}
        results.append(verdict)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    json.dump(dict(results=results), open(out / "adjudication.json", "w"), indent=1)

    lines = ["# Expert-union measurement: adjudication of the frozen predictions", "",
             "Thresholds were fixed before any route data existed. This script fits",
             "nothing; it only reads what the collector wrote.", ""]
    for r in results:
        lines += [f'## `{Path(r["source"]).parent.name}`', "",
                  f'- layers: {r["n_layers"]}, experts {r["experts_total"]}, '
                  f'top-k {r["experts_per_token"]}, median batch width '
                  f'{r["median_batch_width"]}',
                  f'- layers at or above width {r["width_threshold"]}: '
                  f'{r["n_layers_at_or_above_width_threshold"]}',
                  f'- collector verdict: **{r["collector_verdict"]}**', ""]
        disc = r["collection_discipline"]
        if disc:
            lines += [f'- pure-decode steps only: {disc.get("pure_decode_steps_only")}; '
                      f'skipped {disc.get("skipped_steps")}; hook mismatches '
                      f'{disc.get("hook_token_count_mismatches")}', ""]
        if r["truncation"].get("episode_truncated"):
            lines += ["- **episode truncated at max_route_steps**: no throughput or "
                      "latency claim may use this episode.", ""]
        lines += ["| prediction | holds | key number |", "|---|---|---|"]
        lines.append(f'| E1 router concentrates | **{r["E1"]["holds"]}** | '
                     f'{r["E1"]["n_layers_exceeding_null"]}/{r["n_layers"]} layers above null |')
        e2 = r["E2"]["max_layer_idle_p50_in_scope"]
        lines.append(f'| E2 idle < {r["idle_threshold"]} | **{r["E2"]["holds"]}** | '
                     f'max layer median idle {e2:.4f} |' if e2 is not None
                     else f'| E2 | n/a | no data |')
        lines.append(f'| E3 saturates <= {r["saturation_horizon"]} steps | '
                     f'**{r["E3"]["holds"]}** | windows {r["E3"]["saturation_windows"]} |')
        lines += ["", f'Scope: {r["scope_note"]}', "",
                  "| layer | steps | width | union p50 | idle p50 | uniform null | "
                  "above null | sat99 | sat95 | skew C | never selected |",
                  "|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|"]
        for p in r["per_layer"]:
            skew = "n/a" if p["load_skew"] is None else f'{p["load_skew"]:.3f}'
            lines.append(
                f'| {p["layer"]} | {p["n_steps"]} | {p["median_width"]} | '
                f'{p["union_p50"]:.4f} | {p["idle_p50"]:.4f} | {p["uniform_null_idle"]:.4f} | '
                f'{"yes" if p["exceeds_null"] else "no"} | {p["saturation_99"]} | '
                f'{p["saturation_95"]} | {skew} | {p["experts_never_selected"]} |')
        lines += ["", "U is a structural signal: not measured HBM traffic, not what the",
                  "fused backend loads, not proof that idle bytes are reclaimable.", ""]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
