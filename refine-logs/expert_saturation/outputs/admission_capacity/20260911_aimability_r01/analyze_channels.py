#!/usr/bin/env python3
"""Which channel does the admission cap actually act through?

Background
----------
The capture-ladder mechanism was designed to act through one channel: place the
decode batch on a CUDA-graph capture point and stop paying for padded width.
It failed with unstable sign (-44%/+119%/-20%/-57% on steady). The usual reading
is "the controller was mistuned".

The aimability analysis in this directory refuted the first alternative I
proposed -- the effect-time bucket is in fact highly predictable at the lag
where caps bind (0.85 of bucket entropy resolved at lag 1, cap binding median
0 steps), so lack of information is *not* the reason.

This analyser tests the remaining structural explanation:

    The cap moves goodput mainly through a channel other than the one the
    mechanism was designed around.

Two quantities per episode:
  * alignment channel  -- total time paid for padded graph width, as a fraction
    of episode duration. This is an arithmetic CEILING on everything the
    capture-alignment mechanism could ever recover.
  * admission channel  -- time requests spend between arriving and being
    admitted, as a fraction of episode duration.

If the observed goodput spread across caps is far larger than the alignment
ceiling, then the alignment mechanism cannot be what the lever is doing,
regardless of how the controller is tuned.

What this is NOT
----------------
- Not a policy result and not a counterfactual. Channels are decompositions of
  what was recorded, not predictions of what another policy would produce.
- The alignment ceiling assumes every padded slot could be reclaimed at zero
  cost, which no real action achieves. It is deliberately generous.
- Correlations are across a small cap sweep within one engine; they order the
  channels, they do not estimate an effect size.
"""

import argparse
import glob
import json
import statistics as st
from pathlib import Path

CAPTURE_SIZES = (1, 2, 4, 8, 16, 24, 32)


def capture_bucket(width):
    for size in CAPTURE_SIZES:
        if width <= size:
            return size
    return CAPTURE_SIZES[-1]


def spearman(xs, ys):
    """Rank correlation; ties get average ranks."""
    def rank(vals):
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        r = [0.0] * len(vals)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    if len(xs) < 3:
        return None
    rx, ry = rank(xs), rank(ys)
    mx, my = st.mean(rx), st.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else None


def step_execution_times(data):
    """Reconstruct per-step execution time from the engine's own timestamps.

    `end_s` in the trace is when `schedule()` returned, not when the step
    finished executing: it covers only ~1.6% of wall clock and using it
    understates every time-based channel by ~60x. With
    `async_scheduling=False` and `stream_interval=1` one schedule() call maps
    to exactly one output delivery, so

        exec_ms[k] = received_s[k] - scheduler_steps[k].start_s

    Both timestamps come from the same perf_counter origin recorded at run
    time. Returns None unless the one-to-one alignment actually holds, so a
    trace that violates the assumption is skipped rather than mis-measured.
    """
    steps = data.get("scheduler_steps") or []
    receipts = sorted({e["received_s"] for e in data.get("output_events", [])})
    if not steps or len(receipts) != len(steps):
        return None
    times = []
    for k, s in enumerate(steps):
        dur = receipts[k] - s["start_s"]
        if dur <= 0:
            return None
        if k + 1 < len(steps) and receipts[k] > steps[k + 1]["start_s"] + 1e-9:
            return None                   # receipt leaked past the next step
        times.append(dur)
    return times


def episode_channels(raw_path, duration_s):
    """Alignment ceiling and admission cost for one sealed episode."""
    data = json.load(open(raw_path))
    steps = data.get("scheduler_steps") or []
    if not steps or not duration_s:
        return None
    exec_times = step_execution_times(data)
    if exec_times is None:
        return None

    padded_time_s = 0.0
    paid_time_s = 0.0
    waste_fracs = []
    for k, s in enumerate(steps):
        width = s.get("decode_requests", 0)
        if width <= 0 or any(r.get("prefill_tokens") for r in s.get("scheduled", [])):
            continue                      # alignment claim is about pure decode
        dur = exec_times[k]
        bucket = capture_bucket(width)
        waste = 1 - width / bucket
        # Time attributable to slots that were paid for but carried no request.
        padded_time_s += dur * waste
        paid_time_s += dur
        waste_fracs.append(waste)

    admit_delays = [r["admission_s"] - r["arrival_s"]
                    for r in data.get("requests", [])
                    if r.get("admission_s") is not None and r.get("arrival_s") is not None]

    return dict(
        n_pure_decode_steps=len(waste_fracs),
        pure_decode_time_s=paid_time_s,
        # ALIGNMENT CHANNEL: generous ceiling on recoverable time.
        alignment_ceiling_s=padded_time_s,
        alignment_ceiling_frac_of_episode=padded_time_s / duration_s,
        mean_padding_waste=st.mean(waste_fracs) if waste_fracs else 0.0,
        # ADMISSION CHANNEL.
        mean_admission_delay_s=st.mean(admit_delays) if admit_delays else 0.0,
        total_admission_delay_s=sum(admit_delays) if admit_delays else 0.0,
        admission_frac_of_episode=(sum(admit_delays) / duration_s) if admit_delays else 0.0,
        mean_waiting=st.mean(s.get("waiting_before", 0) for s in steps))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paired-dir", action="append", required=True,
                    help="directory holding analysis/analysis.json and gpu_results/")
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    rows = []
    for base in args.paired_dir:
        base = Path(base)
        analysis = json.load(open(base / "analysis" / "analysis.json"))
        for cell in analysis["cells"]:
            if not cell.get("eligible"):
                continue
            metrics, plan = cell["metrics"], cell["plan"]
            duration = metrics.get("observation_duration_s")
            # The sealed analysis stores an absolute path from the run host;
            # relocate by filename inside this repository copy.
            name = Path(cell["path"]).name
            engine = Path(cell["path"]).parent.name
            candidates = glob.glob(str(base / "gpu_results" / engine / name))
            if not candidates:
                candidates = glob.glob(str(base / "gpu_results" / "**" / name), recursive=True)
            if not candidates:
                continue
            ch = episode_channels(candidates[0], duration)
            if ch is None:
                continue
            rows.append(dict(
                campaign=base.name, engine=engine, cell=name,
                regime=plan["regime"], cap=plan["cap"], policy=plan["policy"],
                goodput_rps=metrics["goodput_rps"],
                n_slo_pass=metrics["n_slo_pass"],
                duration_s=duration, **ch))

    if not rows:
        raise SystemExit("no rows built; check --paired-dir")

    # Group by (campaign, engine, regime) so nothing is pooled across engines.
    groups = {}
    for r in rows:
        groups.setdefault((r["campaign"], r["engine"], r["regime"]), []).append(r)

    summary = []
    for key, grp in sorted(groups.items()):
        statics = [r for r in grp if r["policy"] == "static"]
        if len(statics) < 3:
            continue
        gp = [r["goodput_rps"] for r in statics]
        summary.append(dict(
            campaign=key[0], engine=key[1], regime=key[2], n_static=len(statics),
            goodput_min=min(gp), goodput_max=max(gp),
            goodput_spread_frac=(max(gp) - min(gp)) / min(gp) if min(gp) else None,
            alignment_ceiling_frac_max=max(r["alignment_ceiling_frac_of_episode"]
                                           for r in statics),
            admission_frac_max=max(r["admission_frac_of_episode"] for r in statics),
            rho_goodput_vs_alignment=spearman(
                [r["alignment_ceiling_frac_of_episode"] for r in statics], gp),
            rho_goodput_vs_admission=spearman(
                [r["admission_frac_of_episode"] for r in statics], gp),
            rho_goodput_vs_padding_waste=spearman(
                [r["mean_padding_waste"] for r in statics], gp)))

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    json.dump(dict(capture_sizes=list(CAPTURE_SIZES), n_episodes=len(rows),
                   per_episode=rows, per_group=summary),
              open(out / "channels.json", "w"), indent=1)

    lines = ["# Which channel does the admission cap act through?", "",
             f"{len(rows)} sealed episodes, grouped by engine so nothing is pooled.",
             "The alignment ceiling assumes every padded graph slot is reclaimable at",
             "zero cost, so it over-states what the capture mechanism could ever win.",
             "", "## Static cap sweep per engine", "",
             "| campaign | engine | regime | n | goodput spread | alignment ceiling (max) | "
             "admission cost (max) | rho gp~alignment | rho gp~admission | rho gp~padding |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for s in summary:
        def f(v):
            return "n/a" if v is None else f"{v:+.3f}"
        lines.append(
            f'| {s["campaign"][:22]} | {s["engine"]} | {s["regime"]} | {s["n_static"]} | '
            f'{s["goodput_spread_frac"]*100:.1f}% | '
            f'{s["alignment_ceiling_frac_max"]*100:.2f}% | '
            f'{s["admission_frac_max"]*100:.1f}% | '
            f'{f(s["rho_goodput_vs_alignment"])} | {f(s["rho_goodput_vs_admission"])} | '
            f'{f(s["rho_goodput_vs_padding_waste"])} |')

    spreads = [s["goodput_spread_frac"] for s in summary if s["goodput_spread_frac"]]
    ceils = [s["alignment_ceiling_frac_max"] for s in summary]
    lines += ["", "## Magnitude bound", "",
              f"- median goodput spread across static caps: **{st.median(spreads)*100:.1f}%**",
              f"- median alignment ceiling: **{st.median(ceils)*100:.2f}%** of episode time",
              f"- ratio: the lever moves goodput about "
              f"**{st.median(spreads)/st.median(ceils):.0f}x** more than the entire",
              "  alignment channel could account for, even granting free reclamation.", ""]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
