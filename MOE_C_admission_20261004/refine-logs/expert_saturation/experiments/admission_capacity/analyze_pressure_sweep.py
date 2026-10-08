#!/usr/bin/env python3
"""Compare rotation against native across a sweep of structural KV deficits.

The metric this uses, and why it replaces the previous one
----------------------------------------------------------
Earlier rounds scored a policy by "tail waste": time spent at low batch width
beyond what the same tokens would have cost at high width. That quantity is
relative to the policy's own high-width phase, so it cannot be compared across
workloads, and it turned out to conflate two different things.

Decode is serial per request: a request that must emit L tokens occupies at
least L decode steps. With all requests admitted up front, the whole episode
therefore cannot finish in fewer than

    floor = max_i L_i

pure decode steps, whatever the scheduler does. Scoring a policy by

    step_efficiency = pure_decode_steps / floor

gives an absolute reference. A policy at 1.0 never wasted a step; 1.9 means it
spent 90% more steps than physically necessary, which happens when the batch
drains and the remaining requests are carried at low width.

This also corrected an attribution error: the 23.6% wall penalty measured for a
heterogeneous length mix was 106% accounted for by its higher `max(L)` (3072 vs
2048), not by width collapse. Both arms ran within 2.7-6.9% of their own floor.
Width collapse is a symptom of a high floor, not an independent cost.

What the sweep answers
----------------------
Whether rotation's advantage is a property of one operating point or a trend.
The deficit

    D = N * ceil((P + L) / block) - usable_blocks

is varied through the KV pool alone. For each (deficit, arm) the script reports
max ITL, throughput, step efficiency, preemption and recompute counts, and pairs
arms within a deficit so that nothing is compared across pools.

Limits
------
* Descriptive accounting of executed traces; no counterfactual.
* Per-cell n=1. The forward/reverse pair gives an observed repeat difference,
  which is reported as such and is not a significance test.
* `floor` assumes every request is admitted before the first completion. A cell
  whose arrival trace violates that would have a floor that is too low; the
  script reports the admission span so this stays visible.
* Steps carrying prefill or recompute are excluded from the pure-decode count
  and reported separately.
"""

import argparse, json, statistics as st
from pathlib import Path


def load_cell(run_dir):
    run_dir = Path(run_dir)
    raw = json.load(open(run_dir / "raw.json"))
    met = json.load(open(run_dir / "metrics.json"))
    cfg = json.load(open(run_dir / "config.json"))
    qual_path = run_dir / "safe-cap-qualification.json"
    qual = json.load(open(qual_path)) if qual_path.exists() else {}

    steps = raw["scheduler_steps"]
    dur = [steps[k + 1]["start_s"] - steps[k]["start_s"] for k in range(len(steps) - 1)]
    pure, recompute, prefill_steps = [], [], 0
    for k in range(len(steps) - 1):
        s = steps[k]
        if s.get("recompute_tokens"):
            recompute.append(k)
            continue
        if any(r.get("prefill_tokens") for r in s.get("scheduled", [])):
            prefill_steps += 1
            continue
        if s.get("decode_requests", 0) > 0:
            pure.append((s["decode_requests"], dur[k]))

    done = [r for r in met["per_request"] if r["status"] == "completed"]
    lengths = [r["n_output_tokens"] for r in done]
    floor = max(lengths) if lengths else None

    # Admission span: the floor argument assumes everyone is in before anyone
    # leaves. Surface the two timestamps so a violation is visible.
    adm = [r["admission_s"] for r in raw["requests"] if r.get("admission_s") is not None]
    comp = sorted(r["completion_s"] for r in raw["requests"] if r.get("completion_s"))

    med_pure = st.median([t for _, t in pure]) if pure else 0.0
    rc_extra = sum(dur[k] for k in recompute) - len(recompute) * med_pure

    # Cells are grouped by deficit, so a cell that was archived without its
    # qualification record must still land in the right group. The pool size is
    # what defines the group, and it is also recorded in config, so fall back to
    # deriving the block counts from the configured bytes. The ratio comes from
    # the sealed run (16,089,350,144 bytes over 7672 total blocks) and is only
    # used for grouping; a cell that carries its own qualification always uses
    # the engine-reported numbers instead.
    per_req = qual.get("per_request_reserved_blocks")
    usable = qual.get("usable_blocks")
    blocks_from_config = None
    if usable is None and cfg.get("fixed_kv_cache_memory_bytes"):
        bytes_per_block = 16089350144 / 7672
        total = round(cfg["fixed_kv_cache_memory_bytes"] / bytes_per_block)
        usable = total - 1
        blocks_from_config = usable
    if per_req is None:
        per_req = -(-(cfg["prompt_tokens"] + cfg["output_tokens"]) // 16)

    return dict(
        label=run_dir.name,
        policy=cfg.get("completion_policy"),
        kv_cache_bytes=cfg.get("fixed_kv_cache_memory_bytes"),
        usable_blocks=usable,
        usable_blocks_derived_from_config=blocks_from_config is not None,
        per_request_blocks=per_req,
        deficit_blocks=(per_req * cfg["requests"] - usable) if usable and per_req else None,
        wall_s=met["observation_duration_s"],
        throughput_rps=met["throughput_rps"],
        goodput_rps=met["goodput_rps"],
        n_slo_pass=met["n_slo_pass"],
        n_completed=met["n_completed"],
        preemptions=met.get("actual_preemption_count"),
        max_itl_s=max((max(r["itl_s"]) for r in done if r["itl_s"]), default=None),
        mean_max_itl_s=st.mean([max(r["itl_s"]) for r in done if r["itl_s"]]) if done else None,
        ttft_p95_s=sorted(r["ttft_s"] for r in done)[int(0.95 * len(done))] if done else None,
        mean_completion_s=st.mean([r["request_latency_s"] for r in done]) if done else None,
        completion_span_s=(comp[-1] - comp[0]) if comp else None,
        floor_steps=floor,
        pure_decode_steps=len(pure),
        step_efficiency=(len(pure) / floor) if floor else None,
        recompute_steps=len(recompute),
        recompute_tokens=sum(steps[k].get("recompute_tokens", 0) for k in recompute),
        marginal_recompute_s=rc_extra,
        prefill_steps=prefill_steps,
        admission_span_s=(max(adm) - min(adm)) if adm else None,
        first_completion_s=comp[0] if comp else None,
        admission_before_first_completion=(max(adm) < comp[0]) if adm and comp else None,
        mean_decode_width=(sum(w for w, _ in pure) / len(pure)) if pure else None,
        source_dir=str(run_dir),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cell", action="append", required=True, help="PATH to a completed cell")
    ap.add_argument("--baseline-policy", default="native")
    ap.add_argument("--intervention-policy", default="rotate")
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()

    cells = [load_cell(p) for p in args.cell]
    valid = [c for c in cells if c["n_completed"] and c["floor_steps"]]

    # Group by (deficit blocks, block index parsed from the label prefix).
    groups = {}
    for c in valid:
        block = c["label"].split("-", 1)[0]
        groups.setdefault((c["deficit_blocks"], block), {})[c["policy"]] = c

    pairs = []
    for (deficit, block), arms in sorted(groups.items(), key=lambda kv: (kv[0][0] or 0, kv[0][1])):
        b = arms.get(args.baseline_policy)
        i = arms.get(args.intervention_policy)
        if not b or not i:
            continue
        pairs.append(dict(
            deficit_blocks=deficit, block=block,
            deficit_requests=(deficit / b["per_request_blocks"]) if b["per_request_blocks"] else None,
            baseline=b["label"], intervention=i["label"],
            base_max_itl_s=b["max_itl_s"], interv_max_itl_s=i["max_itl_s"],
            d_max_itl_s=i["max_itl_s"] - b["max_itl_s"],
            base_throughput=b["throughput_rps"], interv_throughput=i["throughput_rps"],
            throughput_delta_pct=100 * (i["throughput_rps"] / b["throughput_rps"] - 1),
            d_wall_s=i["wall_s"] - b["wall_s"],
            base_step_eff=b["step_efficiency"], interv_step_eff=i["step_efficiency"],
            d_step_efficiency=i["step_efficiency"] - b["step_efficiency"],
            base_preemptions=b["preemptions"], interv_preemptions=i["preemptions"],
            base_recompute_tokens=b["recompute_tokens"], interv_recompute_tokens=i["recompute_tokens"],
            action_fired=bool(i["preemptions"]),
        ))

    # Observed repeat difference: same policy, same deficit, different block.
    repeats = {}
    by_key = {}
    for c in valid:
        by_key.setdefault((c["deficit_blocks"], c["policy"]), []).append(c)
    for key, cs in by_key.items():
        if len(cs) == 2:
            repeats[f"deficit{key[0]}/{key[1]}"] = dict(
                abs_d_wall_s=abs(cs[1]["wall_s"] - cs[0]["wall_s"]),
                abs_d_max_itl_s=abs(cs[1]["max_itl_s"] - cs[0]["max_itl_s"]),
                abs_d_step_eff=abs(cs[1]["step_efficiency"] - cs[0]["step_efficiency"]))
    floor_wall = max((v["abs_d_wall_s"] for v in repeats.values()), default=None)
    floor_itl = max((v["abs_d_max_itl_s"] for v in repeats.values()), default=None)

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    json.dump(dict(cells=cells, pairs=pairs, observed_repeat_differences=repeats,
                   largest_repeat_diff=dict(wall_s=floor_wall, max_itl_s=floor_itl),
                   evidence_ceiling="DESCRIPTIVE_ACCOUNTING_OF_EXECUTED_TRACES",
                   not_claimed=[
                       "observed repeat difference is not a noise bound or a significance test",
                       "per-cell n=1, forward/reverse confounded with order",
                       "step floor assumes admission completes before the first completion",
                       "no counterfactual wall time for any policy",
                   ]),
              open(out / "sweep.json", "w"), indent=1)

    L = ["# Rotation versus native across structural KV deficits\n"]
    L.append("| cell | policy | usable blk | deficit blk | wall s | thr | max ITL s | steps | floor | steps/floor | preempt | recomp tok |")
    L.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for c in sorted(cells, key=lambda x: (x["deficit_blocks"] or 0, x["label"])):
        if not c["floor_steps"]:
            L.append(f"| {c['label']} | {c['policy']} | - | - | - | - | - | - | - | INVALID | - | - |")
            continue
        L.append(f"| {c['label']} | {c['policy']} | {c['usable_blocks']} | {c['deficit_blocks']} | "
                 f"{c['wall_s']:.3f} | {c['throughput_rps']:.4f} | {c['max_itl_s']:.3f} | "
                 f"{c['pure_decode_steps']} | {c['floor_steps']} | {c['step_efficiency']:.3f} | "
                 f"{c['preemptions']} | {c['recompute_tokens']} |")
    if pairs:
        L.append(f"\n## {args.intervention_policy} against {args.baseline_policy}, within each deficit\n")
        L.append("| deficit req | block | d max ITL s | thr % | d steps/floor | action fired | base preempt |")
        L.append("|---:|---|---:|---:|---:|---|---:|")
        for p in pairs:
            dr = f"{p['deficit_requests']:.2f}" if p["deficit_requests"] is not None else "?"
            L.append(f"| {dr} | {p['block']} | {p['d_max_itl_s']:+.3f} | {p['throughput_delta_pct']:+.2f} | "
                     f"{p['d_step_efficiency']:+.3f} | {'yes' if p['action_fired'] else 'NO'} | "
                     f"{p['base_preemptions']} |")
    if repeats:
        L.append("\n## Observed repeat differences (same policy, same deficit, two blocks)\n")
        L.append("| key | abs d wall s | abs d max ITL s | abs d steps/floor |")
        L.append("|---|---:|---:|---:|")
        for k, v in sorted(repeats.items()):
            L.append(f"| {k} | {v['abs_d_wall_s']:.4f} | {v['abs_d_max_itl_s']:.4f} | {v['abs_d_step_eff']:.4f} |")
        L.append(f"\nLargest observed repeat difference: wall {floor_wall:.4f} s, "
                 f"max ITL {floor_itl:.4f} s. This describes repetition, not a noise bound.\n")
    open(out / "report.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
