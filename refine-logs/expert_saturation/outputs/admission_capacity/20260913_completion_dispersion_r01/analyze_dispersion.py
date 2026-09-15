#!/usr/bin/env python3
"""Width-cost accounting for admission policies under a structural KV deficit.

Question this answers
---------------------
Four policies share one fixed KV pool and one workload, yet their episode wall
times differ by up to 17.7%. Which *measurable* property of a policy predicts
that difference: the size of the pauses it inflicts, or something else?

The model
---------
Decode throughput is set by batch width, and the per-token cost g(w) = c(w)/w is
strongly super-linear as w shrinks (measured here: g(3)/g(32) = 3.6x). Total
decode work is fixed at N*L tokens, so wall time is decided by *how much of that
work runs at low width*.

Work at low width happens in the drain: the interval between the first and the
last completion. Therefore

    tail_waste = SUM over low-width steps of ( c(w) - w * g_high )

where g_high is the policy's own measured cost per token in its high-width phase.
This is a same-policy, same-hardware quantity -- no cross-arm calibration.

The falsifiable claim is that `tail_waste` tracks wall-time differences between
policies while `max ITL` does not. Both are computed here and both are reported,
including when the claim fails.

Deliberate limits
-----------------
* Descriptive accounting of executed traces. It does not simulate a policy, does
  not produce a counterfactual wall, and cannot say a policy *would* have been
  faster.
* g_high is an in-phase average, so it absorbs whatever prefill/bookkeeping
  overlapped the high-width steps. It is a reference level, not kernel time.
* `low width` is defined as w <= max_width/2 within the same episode. The
  threshold is reported and the sweep over thresholds is emitted so a reader can
  see whether the result depends on it.
* Steps carrying prefill or recompute are excluded from the cost table because
  their duration is not a function of decode width alone. They are counted
  separately and their wall share is reported.
"""

import argparse, json, statistics as st
from collections import Counter
from pathlib import Path


def step_durations(steps):
    """Wall duration of each step, from consecutive scheduler entry timestamps.

    `end_s` in these traces is when schedule() returned, not when the step
    finished executing, so it covers only a small fraction of wall time and
    cannot be used here. Successive `start_s` values are back-to-back in this
    synchronous in-process driver, so their difference is the step's wall span
    including execution, sampling and delivery. The final step has no successor
    and is dropped.
    """
    return [steps[k + 1]["start_s"] - steps[k]["start_s"] for k in range(len(steps) - 1)]


def classify(step):
    """Return 'decode', 'prefill' or 'recompute' for one scheduler step."""
    if step.get("recompute_tokens"):
        return "recompute"
    if any(r.get("prefill_tokens") for r in step.get("scheduled", [])):
        return "prefill"
    return "decode" if step.get("decode_requests", 0) > 0 else "idle"


def load_arm(run_dir):
    run_dir = Path(run_dir)
    raw = json.load(open(run_dir / "raw.json"))
    met = json.load(open(run_dir / "metrics.json"))
    cfg = json.load(open(run_dir / "config.json"))
    steps = raw["scheduler_steps"]
    dur = step_durations(steps)

    decode, other_s = [], 0.0
    for k in range(len(steps) - 1):
        kind = classify(steps[k])
        if kind == "decode":
            decode.append((steps[k]["decode_requests"], dur[k]))
        else:
            other_s += dur[k]

    done = [r for r in met["per_request"] if r["status"] == "completed"]
    comp = sorted(r["completion_s"] for r in raw["requests"] if r.get("completion_s"))
    return dict(
        label=run_dir.name,
        cap=cfg.get("cap"),
        policy=cfg.get("completion_policy"),
        wall_s=met["observation_duration_s"],
        throughput_rps=met["throughput_rps"],
        goodput_rps=met["goodput_rps"],
        n_slo_pass=met["n_slo_pass"],
        n_completed=met["n_completed"],
        preemptions=met.get("actual_preemption_count"),
        max_itl_s=max(max(r["itl_s"]) for r in done if r["itl_s"]),
        ttft_p95_s=sorted(r["ttft_s"] for r in done)[int(0.95 * len(done))],
        completion_span_s=comp[-1] - comp[0] if comp else None,
        decode=decode,
        non_decode_s=other_s,
    )


def tail_waste(decode, frac=0.5):
    """Seconds spent at low width beyond what the same tokens cost at high width."""
    if not decode:
        return None
    wmax = max(w for w, _ in decode)
    cut = wmax * frac
    hi = [(w, t) for w, t in decode if w > cut]
    lo = [(w, t) for w, t in decode if w <= cut]
    hi_tok = sum(w for w, _ in hi)
    if not hi_tok:
        return None
    g_hi = sum(t for _, t in hi) / hi_tok              # seconds per token, high width
    lo_s = sum(t for _, t in lo)
    lo_tok = sum(w for w, _ in lo)
    return dict(threshold_width=cut, g_high_s_per_token=g_hi,
                low_width_steps=len(lo), low_width_s=lo_s, low_width_tokens=lo_tok,
                waste_s=lo_s - lo_tok * g_hi,
                high_width_steps=len(hi), high_width_s=sum(t for _, t in hi))


def cost_table(decode, min_n=20):
    by = {}
    for w, t in decode:
        by.setdefault(w, []).append(t)
    return {w: dict(n=len(v), c_ms=st.median(v) * 1000,
                    g_ms_per_token=st.median(v) * 1000 / w)
            for w, v in sorted(by.items()) if len(v) >= min_n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", required=True,
                    help="NAME=PATH to a completed cell directory")
    ap.add_argument("--reference", required=True,
                    help="arm NAME used as the wall-time reference for deltas")
    ap.add_argument("--output-dir", required=True, type=Path)
    args = ap.parse_args()

    arms = {}
    for spec in args.arm:
        name, path = spec.split("=", 1)
        a = load_arm(path)
        a["tail"] = tail_waste(a["decode"])
        a["cost_table"] = cost_table(a["decode"])
        a["width_histogram"] = Counter(w for w, _ in a["decode"]).most_common(8)
        a["source_dir"] = path
        arms[name] = a

    ref = arms[args.reference]

    # Does tail waste track the wall-time delta better than max ITL does?
    rows = []
    for name, a in arms.items():
        if name == args.reference:
            continue
        d_wall = a["wall_s"] - ref["wall_s"]
        d_waste = a["tail"]["waste_s"] - ref["tail"]["waste_s"]
        d_span = a["completion_span_s"] - ref["completion_span_s"]
        d_itl = a["max_itl_s"] - ref["max_itl_s"]
        rows.append(dict(arm=name, d_wall_s=d_wall, d_tail_waste_s=d_waste,
                         d_completion_span_s=d_span, d_max_itl_s=d_itl,
                         explained_frac=(d_waste / d_wall) if d_wall else None,
                         wall_s=a["wall_s"], max_itl_s=a["max_itl_s"],
                         completion_span_s=a["completion_span_s"],
                         tail_waste_s=a["tail"]["waste_s"]))

    # Sign test: max ITL and wall move in OPPOSITE directions if the pause-cost
    # story is wrong; tail waste and wall should move together.
    agree_waste = sum(1 for r in rows if r["d_wall_s"] * r["d_tail_waste_s"] > 0)
    agree_itl = sum(1 for r in rows if r["d_wall_s"] * r["d_max_itl_s"] > 0)

    sweep = {}
    for f in (0.3, 0.4, 0.5, 0.6, 0.7):
        vals = {n: tail_waste(a["decode"], f)["waste_s"] for n, a in arms.items()}
        sweep[f] = vals

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    payload = dict(
        reference=args.reference,
        arms={n: {k: v for k, v in a.items() if k != "decode"} for n, a in arms.items()},
        comparisons=rows,
        sign_agreement=dict(tail_waste=f"{agree_waste}/{len(rows)}",
                            max_itl=f"{agree_itl}/{len(rows)}"),
        threshold_sweep=sweep,
        evidence_ceiling="DESCRIPTIVE_ACCOUNTING_OF_EXECUTED_TRACES",
        not_claimed=[
            "no counterfactual wall time for any policy",
            "no claim that a policy would be faster if its tail were shortened",
            "g_high absorbs overlapped prefill/bookkeeping; not kernel time",
            "single model, single GPU, one homogeneous 3072/1024 workload",
        ],
    )
    json.dump(payload, open(out / "dispersion.json", "w"), indent=1)

    L = []
    L.append("# Width-cost accounting across admission policies\n")
    L.append(f"Reference arm: `{args.reference}`. "
             f"Evidence ceiling: descriptive accounting of executed traces.\n")
    L.append("| arm | wall s | max ITL s | completion span s | tail waste s | preempt | SLO |")
    L.append("|---|---:|---:|---:|---:|---:|---:|")
    for n, a in arms.items():
        L.append(f"| {n} | {a['wall_s']:.3f} | {a['max_itl_s']:.3f} | "
                 f"{a['completion_span_s']:.3f} | {a['tail']['waste_s']:.3f} | "
                 f"{a['preemptions']} | {a['n_slo_pass']}/{a['n_completed']} |")
    L.append(f"\n## Deltas against `{args.reference}`\n")
    L.append("| arm | d wall s | d tail waste s | explained | d max ITL s | d span s |")
    L.append("|---|---:|---:|---:|---:|---:|")
    for r in rows:
        e = f"{100*r['explained_frac']:.1f}%" if r["explained_frac"] is not None else "n/a"
        L.append(f"| {r['arm']} | {r['d_wall_s']:+.3f} | {r['d_tail_waste_s']:+.3f} | {e} | "
                 f"{r['d_max_itl_s']:+.3f} | {r['d_completion_span_s']:+.3f} |")
    L.append(f"\nSign agreement with wall delta: tail waste {agree_waste}/{len(rows)}, "
             f"max ITL {agree_itl}/{len(rows)}.\n")
    L.append("## Measured per-width decode cost\n")
    for n, a in arms.items():
        cells = "  ".join(f"w{w}: {v['c_ms']:.2f}ms ({v['g_ms_per_token']:.3f} ms/tok, n={v['n']})"
                          for w, v in a["cost_table"].items())
        L.append(f"- **{n}** — {cells}")
    L.append("\n## Threshold sensitivity of `low width`\n")
    L.append("| frac | " + " | ".join(arms) + " |")
    L.append("|---|" + "---|" * len(arms))
    for f, vals in sweep.items():
        L.append(f"| {f} | " + " | ".join(f"{vals[n]:.3f}" for n in arms) + " |")
    open(out / "report.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
