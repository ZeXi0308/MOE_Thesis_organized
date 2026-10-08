#!/usr/bin/env python3
"""What tail could absence rotation buy, and what would it cost?

A bound, not a result. It is computed from the sealed long-context episode so
that the decision to spend GPU time is informed, and it deliberately stops
short of simulating KV, queue or completion dynamics -- those only come from a
real run.

The arithmetic rests on one measured invariant: the deficit is structural, so
the TOTAL absence is fixed and rotation can only redistribute it.

    absence_total  = sum over victims of (resume_step - preempt_step)
    binding_window = last_resume_step - first_preempt_step

Under concentration (what the engine does today) one request absorbs the
maximum. Under rotation with cooldown c over a window of W steps, at most
floor(W/c) swaps occur, and the mandatory absence is shared by the requests
that take turns, so the per-request maximum falls roughly to the cooldown.

Costs charged, all from measurement:
  * each swap pays one recompute of the resumed request's context;
  * each swap also evicts a fresh victim, whose own recompute is paid when it
    later returns -- so a swap costs TWO recomputes, not one. Charging only one
    would flatter the mechanism.

What this is NOT
----------------
- Not a policy result. No counterfactual execution is claimed; the real effect
  can only be measured by running the controller.
- Ignores the effect of holding blocks for a victim on the survivors' TPOT,
  which is a real cost this bound does not capture.
- The recompute cost is the measured wall span of the victim's recompute calls
  in the sealed run, which includes concurrent work, so it is not isolated GPU
  time.
"""

import argparse
import json
import statistics as st
from pathlib import Path


def load_dynamics(raw_path):
    """Extract preemption/resume dynamics and the MARGINAL recompute cost.

    The marginal cost matters and is easy to get wrong. During a victim's
    recompute the other ~30 requests keep decoding, so the price of a swap is
    how much those steps are SLOWED, not their whole wall duration. Charging
    the full duration (116 ms measured as the victim's recompute-call span)
    over-states the cost by roughly 3x and would wrongly kill the mechanism.

    The marginal figure is measured here rather than assumed: recompute steps
    are identified by a request appearing in `scheduled` with `decode_tokens`
    at zero (a chunk of prior context being replayed), and their excess over
    the pure-decode baseline is summed per victim.
    """
    data = json.load(open(raw_path))
    steps = data["scheduler_steps"]

    # Pure-decode baseline: one token per running request, no preemption.
    periods = []
    for k in range(len(steps) - 1):
        a, b = steps[k], steps[k + 1]
        if a["decode_requests"] < 29:
            continue
        if a["total_scheduled_tokens"] > 32:
            continue
        if a.get("preempted_request_ids"):
            continue
        periods.append((b["start_s"] - a["start_s"]) * 1000.0)
    base_ms = st.median(periods) if periods else None

    events = []
    for k, s in enumerate(steps):
        for rid in s.get("preempted_request_ids") or []:
            back = next((j for j in range(k + 1, len(steps))
                         if any(r["request_id"] == rid
                                for r in steps[j]["scheduled"])), None)
            marginal = None
            if back is not None and base_ms:
                window = [j for j in range(back, min(back + 12, len(steps) - 1))
                          if any(r["request_id"] == rid
                                 and not r.get("decode_tokens")
                                 for r in steps[j]["scheduled"])]
                if window:
                    actual = sum((steps[j + 1]["start_s"] - steps[j]["start_s"])
                                 * 1000.0 for j in window)
                    marginal = actual - len(window) * base_ms
            events.append(dict(request_id=rid, preempt_step=k, resume_step=back,
                               absence_steps=(back - k) if back else None,
                               recompute_marginal_ms=marginal))
    marginals = [e["recompute_marginal_ms"] for e in events
                 if e["recompute_marginal_ms"] is not None]
    completions = [r["completion_s"] for r in data.get("requests", [])
                   if r.get("completion_s") is not None]
    return dict(n_steps=len(steps), events=events,
                step_ms=base_ms, n_period_samples=len(periods),
                measured_span_s=max(completions) if completions else None,
                measured_recompute_marginal_ms=(
                    st.median(marginals) if marginals else None),
                n_marginal_samples=len(marginals))


def bound(dyn, cooldown_steps, recompute_ms, n_requests, output_tokens):
    events = [e for e in dyn["events"] if e["absence_steps"] is not None]
    if not events:
        return None
    step_ms = dyn["step_ms"]
    absence_total = sum(e["absence_steps"] for e in events)
    max_absence = max(e["absence_steps"] for e in events)
    window = (max(e["resume_step"] for e in events)
              - min(e["preempt_step"] for e in events))
    episode_steps = dyn["n_steps"]

    n_swaps = window // cooldown_steps if cooldown_steps > 0 else 0
    # Two recomputes per swap: the resumed request now, the new victim later.
    recompute_total_s = 2.0 * n_swaps * recompute_ms / 1000.0
    # Episode duration must be the MEASURED span, not n_steps x baseline: the
    # step count includes prefill and recompute steps that do not run at the
    # pure-decode rate, so the product over-states wall time by ~13% here and
    # would under-state every throughput cost below.
    episode_s = dyn.get("measured_span_s") or (episode_steps * step_ms / 1000.0)

    # Rotated per-request maximum: a turn lasts at most one cooldown, plus the
    # time before the first rotation becomes legal.
    rotated_max_steps = cooldown_steps
    return dict(
        cooldown_steps=cooldown_steps,
        absence_total_steps=absence_total,
        binding_window_steps=window,
        concentrated_max_absence_steps=max_absence,
        concentrated_max_absence_s=max_absence * step_ms / 1000.0,
        rotated_max_absence_steps=rotated_max_steps,
        rotated_max_absence_s=rotated_max_steps * step_ms / 1000.0,
        tail_reduction_x=max_absence / rotated_max_steps if rotated_max_steps else None,
        n_swaps=n_swaps,
        recompute_total_s=recompute_total_s,
        episode_s=episode_s,
        throughput_cost_fraction=recompute_total_s / episode_s if episode_s else None,
        absence_share_of_all_request_steps=absence_total / (n_requests * output_tokens))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", action="append", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--recompute-ms", type=float, default=None,
                    help="override the measured marginal recompute cost; by "
                         "default the value measured from the episode is used")
    ap.add_argument("--n-requests", type=int, default=32)
    ap.add_argument("--output-tokens", type=int, default=1024)
    ap.add_argument("--cooldowns", default="10,20,30,60,120")
    args = ap.parse_args()

    results = []
    for path in args.raw:
        dyn = load_dynamics(path)
        rc = args.recompute_ms if args.recompute_ms is not None \
            else dyn["measured_recompute_marginal_ms"]
        if rc is None:
            continue
        rows = [bound(dyn, c, rc, args.n_requests, args.output_tokens)
                for c in (int(x) for x in args.cooldowns.split(","))]
        results.append(dict(raw=str(path), dynamics=dyn, recompute_ms=rc,
                            bounds=[r for r in rows if r]))

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    json.dump(dict(recompute_ms_override=args.recompute_ms, results=results),
              open(out / "rotation_bound.json", "w"), indent=1)

    lines = ["# Bound on what absence rotation could buy", "",
             "Arithmetic on sealed dynamics. Not a policy result: no",
             "counterfactual execution is claimed, and the effect of holding",
             "blocks for a victim on the survivors' TPOT is NOT captured.", ""]
    for res in results:
        dyn = res["dynamics"]
        lines += [f'## `{Path(res["raw"]).parent.name}`', "",
                  f'- steps {dyn["n_steps"]}, pure-decode baseline '
                  f'**{dyn["step_ms"]:.3f} ms** ({dyn["n_period_samples"]} samples)',
                  f'- **measured marginal recompute cost '
                  f'{res["recompute_ms"]:.1f} ms** '
                  f'({dyn["n_marginal_samples"]} victims). This is the SLOWDOWN of '
                  f'the steps that carry a recompute chunk, not their whole '
                  f'duration: the other ~30 requests keep decoding through it.',
                  "", "| victim | preempt | resume | absence steps | marginal recompute ms |",
                  "|---|---:|---:|---:|---:|"]
        for e in dyn["events"]:
            mc = e.get("recompute_marginal_ms")
            lines.append(f'| {e["request_id"][-8:]} | {e["preempt_step"]} | '
                         f'{e["resume_step"]} | {e["absence_steps"]} | '
                         f'{"n/a" if mc is None else f"{mc:.1f}"} |')
        b0 = res["bounds"][0]
        lines += ["", f'- total absence **{b0["absence_total_steps"]} request-steps** '
                      f'= {b0["absence_share_of_all_request_steps"]*100:.2f}% of all '
                      f'request-steps (this is the volume rotation CANNOT reduce)',
                  f'- binding window {b0["binding_window_steps"]} steps', "",
                  "| cooldown | swaps | rotated max absence | vs concentrated | "
                  "recompute cost | throughput cost |",
                  "|---:|---:|---:|---:|---:|---:|"]
        for r in res["bounds"]:
            lines.append(
                f'| {r["cooldown_steps"]} | {r["n_swaps"]} | '
                f'{r["rotated_max_absence_s"]:.2f} s | '
                f'**{r["tail_reduction_x"]:.1f}x** lower than '
                f'{r["concentrated_max_absence_s"]:.2f} s | '
                f'{r["recompute_total_s"]:.2f} s | '
                f'**{r["throughput_cost_fraction"]*100:.1f}%** |')
        lines += ["", "Reference: the sealed comparison measured `safe29` (zero",
                  "preemption) at **-14.5% throughput** with TTFT p99 **18.0 s**.",
                  "A rotation point is only interesting if it beats that on both axes.",
                  ""]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
