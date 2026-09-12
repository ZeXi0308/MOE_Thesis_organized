"""CPU feasibility check for the CQA arm, on the real frozen arrival traces.

This does NOT predict performance. It answers one question that must be settled
before spending GPU time: on the actual arrival trace, does the coalescing rule
produce enough grouping to matter, and is the TTFT hold budget affordable?

Two quantities come out:

  * grouping: how many admission events remain versus one-per-arrival, and what
    the resulting release widths are. If the trace cannot be grouped, the action
    space does not exist and the GPU run is pointless.
  * charged hold delay: the exact added queueing delay per request. This is a
    pure function of arrival times and the release rule, so it is exactly
    computable on CPU and must be added to the measured TTFT of the baseline.

Everything about execution time, decode width evolution and goodput remains
GPU UNRUN. The hold delay computed here is a lower bound on the TTFT cost, since
it excludes any engine-side queueing that holding may shift.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from admission_coalescer import DEFAULT_CAPTURE_SIZES, CoalescingAdmission, ceil_capture


def simulate_release_schedule(arrivals, prompt_tokens, *, group_size, hold_budget_s,
                              snap, engine_max_seqs, token_budget, capture_sizes,
                              poll_interval_s=0.001):
    """Replay the release rule against the arrival trace on a fine poll grid.

    Running/waiting are unknown without execution, so `snap` is evaluated against
    the count of concurrently released-but-unretired requests under the explicit
    assumption that nothing completes during the arrival window. That assumption
    is stated, not hidden: with 128 output tokens and arrivals finishing inside
    ~1.6 s, no request can retire during the window, so within this window the
    assumption is exact rather than approximate.
    """
    policy = CoalescingAdmission(group_size=group_size, hold_budget_s=hold_budget_s,
                                 capture_sizes=capture_sizes, snap_to_capture_point=snap,
                                 engine_max_seqs=engine_max_seqs,
                                 max_prefill_tokens_per_release=token_budget)
    order = sorted(range(len(arrivals)), key=lambda i: (arrivals[i], i))
    releases, hold_delays = [], {}
    released_total = 0
    next_offer = 0
    clock = 0.0
    horizon = max(arrivals) + hold_budget_s + 1.0
    while (next_offer < len(order) or policy.pending()) and clock <= horizon:
        while next_offer < len(order) and arrivals[order[next_offer]] <= clock + 1e-12:
            index = order[next_offer]
            policy.offer(index=index, arrival_s=arrivals[index],
                         prompt_tokens=prompt_tokens[index], now_s=clock)
            next_offer += 1
        out, row = policy.poll(now_s=clock, running=released_total, waiting=0)
        if out:
            released_total += len(out)
            releases.append(dict(release_s=clock, count=len(out), reason=row["reason"],
                                 width_after=released_total,
                                 padded_width_after=ceil_capture(released_total, capture_sizes),
                                 request_indices=list(out)))
            for index in out:
                hold_delays[index] = clock - arrivals[index]
        clock += poll_interval_s
    return policy, releases, hold_delays


def evaluate(arrivals, prompt_tokens, *, label, capture_sizes, engine_max_seqs,
             token_budget, group_size, hold_budget_s, snap):
    policy, releases, hold_delays = simulate_release_schedule(
        arrivals, prompt_tokens, group_size=group_size, hold_budget_s=hold_budget_s,
        snap=snap, engine_max_seqs=engine_max_seqs, token_budget=token_budget,
        capture_sizes=capture_sizes)
    delays = sorted(hold_delays.values())
    widths = [r["count"] for r in releases]
    reasons = {}
    for r in releases:
        reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1
    landed = sum(1 for r in releases if r["width_after"] == r["padded_width_after"])
    return dict(
        arm=label, group_size=group_size, hold_budget_s=hold_budget_s, snap=snap,
        n_requests=len(arrivals), all_released=len(hold_delays) == len(arrivals),
        n_admission_events=len(releases),
        events_vs_baseline=f"{len(releases)}/{len(arrivals)}",
        admission_event_reduction=1.0 - len(releases) / len(arrivals) if arrivals else 0.0,
        release_widths=widths, median_release_width=st.median(widths) if widths else 0,
        release_reasons=reasons,
        releases_landing_on_capture_point=landed,
        hold_delay_s=dict(
            max=max(delays) if delays else 0.0, p50=st.median(delays) if delays else 0.0,
            p90=delays[min(len(delays) - 1, int(0.9 * len(delays)))] if delays else 0.0,
            mean=st.mean(delays) if delays else 0.0,
            total=sum(delays)),
        hold_delay_within_budget=all(d <= hold_budget_s + 2e-3 for d in delays),
        n_polls=len(policy.decisions))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--engine-max-seqs", type=int, default=32)
    parser.add_argument("--prefill-token-budget", type=int, default=1024)
    parser.add_argument("--ttft-slo-s", type=float, default=0.20)
    args = parser.parse_args()

    workload = json.loads((Path(args.prepared_dir) / "workload.json").read_text())
    prompts = workload["actual_prompt_token_ids"]
    tokens = [len(ids) for ids in prompts]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=False)

    results = []
    for regime, arrivals in sorted(workload["arrival_traces_s"].items()):
        base = dict(capture_sizes=DEFAULT_CAPTURE_SIZES, engine_max_seqs=args.engine_max_seqs,
                    token_budget=args.prefill_token_budget)
        # Frozen before any GPU run. hold budgets are fractions of the TTFT SLO,
        # not fitted values; group sizes are the capture-bucket gaps themselves.
        arms = [("immediate_baseline", 1, 0.0, False),
                ("cqa_g4_hold25ms", 4, 0.025, True),
                ("cqa_g4_hold50ms", 4, 0.050, True),
                ("cqa_g8_hold50ms", 8, 0.050, True),
                ("cqa_g8_hold100ms", 8, 0.100, True),
                ("coalesce_only_g8_hold50ms_nosnap", 8, 0.050, False)]
        for label, group, hold, snap in arms:
            row = evaluate(arrivals, tokens, label=label, group_size=group,
                           hold_budget_s=hold, snap=snap, **base)
            row["regime"] = regime
            row["ttft_slo_s"] = args.ttft_slo_s
            row["hold_share_of_ttft_slo"] = row["hold_delay_s"]["max"] / args.ttft_slo_s
            results.append(row)

    report = dict(
        evidence_type="CPU_ARRIVAL_TRACE_FEASIBILITY_ONLY_NO_EXECUTION",
        claim_boundary=("release schedule and charged hold delay are exact functions of the "
                        "frozen arrival trace; all execution time, decode width evolution, "
                        "TPOT and goodput remain GPU UNRUN"),
        prepared_dir=args.prepared_dir, engine_max_seqs=args.engine_max_seqs,
        prefill_token_budget=args.prefill_token_budget,
        capture_sizes=list(DEFAULT_CAPTURE_SIZES), arms=results)
    (out_dir / "feasibility.json").write_text(json.dumps(report, indent=2) + "\n")

    lines = ["# CQA arrival-trace feasibility (CPU only, no execution)", "",
             f"prepared dir `{args.prepared_dir}`; engine max seqs {args.engine_max_seqs}; "
             f"prefill token budget {args.prefill_token_budget}", "",
             "| regime | arm | admission events | reduction | median release width | "
             "landed on capture point | max hold ms | p90 hold ms | hold/TTFT SLO | all released |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in results:
        lines.append(
            f'| {r["regime"]} | {r["arm"]} | {r["events_vs_baseline"]} | '
            f'{r["admission_event_reduction"]:.3f} | {r["median_release_width"]:.0f} | '
            f'{r["releases_landing_on_capture_point"]} | {r["hold_delay_s"]["max"]*1000:.1f} | '
            f'{r["hold_delay_s"]["p90"]*1000:.1f} | {r["hold_share_of_ttft_slo"]:.3f} | '
            f'{r["all_released"]} |')
    lines += ["", "## Release reasons", ""]
    for r in results:
        lines.append(f'- `{r["regime"]}/{r["arm"]}`: {r["release_reasons"]}, '
                     f'widths {r["release_widths"]}')
    (out_dir / "report.md").write_text("\n".join(lines) + "\n")
    print(f"arms evaluated {len(results)}; wrote {out_dir}")
    for r in results:
        print(f'  {r["regime"]:7s} {r["arm"]:34s} events {r["events_vs_baseline"]:>6s} '
              f'maxhold {r["hold_delay_s"]["max"]*1000:6.1f}ms all_released {r["all_released"]}')


if __name__ == "__main__":
    main()
