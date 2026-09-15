"""Does a step-level width-nudging action space exist at all?

The paired experiment (20260908_capture_ladder_paired_r01) ruled the ladder-swap
action NO_GO, and localised why: a non-preemptive admission cap bounds the decode
width, it does not set it. The only remaining way to exploit the measured capture
staircase is an action that decides the executed width of a *single step*.

Before implementing any such action, this script tests whether the opportunity
exists, using only already-recorded steps. For every pure-decode step it asks:

  how many requests would have to be added or withheld to land the executed
  width exactly on a CUDA-graph capture point, and is that adjustment legal
  under non-preemption?

Legality is asymmetric and that asymmetry is the point:

  * WITHHOLDING a request is illegal here. A request already in the running set
    is decoding; removing it from the step means preemption or KV eviction, which
    the frozen invariants forbid. Downward nudges are therefore counted but
    marked illegal.
  * ADDING a request is legal only if a request is actually available to add at
    that instant -- i.e. the scheduler had `waiting_requests > 0` -- and if the
    engine's token budget admits it. Adding also forces a prefill interleave,
    whose measured cost is charged from the same reconstruction
    (tax = 3.853 + 0.008826 * prefill_tokens ms, widths >= 7).

So the action space is: steps sitting just *below* a capture point, with queued
requests available to fill the gap. The saving is the padding the step is already
paying for; the cost is the prefill interleave the fill would inject. Both are
measured, so this yields a bounded net estimate -- an upper bound on the benefit,
because it ignores the downstream effect of admitting earlier.

No new execution. No policy is run. This decides only whether to keep going.
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

CAPTURE_SIZES = (1, 2, 4, 8, 16, 24, 32)
# Measured interleave tax, from the sealed reconstruction (widths >= 7, R2=0.8498).
TAX_FIXED_MS = 3.8527
TAX_PER_PREFILL_TOKEN_MS = 0.008826


def ceil_capture(width):
    for size in CAPTURE_SIZES:
        if width <= size:
            return size
    return CAPTURE_SIZES[-1]


def next_capture_at_or_above(width):
    for size in CAPTURE_SIZES:
        if width <= size:
            return size
    return None


def load_cell(path):
    data = json.loads(Path(path).read_text())
    steps = data["scheduler_steps"]
    receipts = sorted({event["received_s"] for event in data["output_events"]})
    if len(steps) != len(receipts) or data["status"] != "COMPLETE":
        return None
    rows = []
    for k, step in enumerate(steps):
        scheduled = step["scheduled"]
        prefill = sum(r["prefill_tokens"] for r in scheduled)
        width = step["decode_requests"]
        if prefill or width <= 0:
            continue
        rows.append(dict(step=k, width=width, exec_ms=(receipts[k] - step["start_s"]) * 1000.0,
                         waiting=step["waiting_requests"], target=step["target_cap"],
                         actual_active=step["actual_active"]))
    if not rows:
        return None
    return dict(path=str(path), regime=data["regime"], cap=data["target_cap"],
                policy=data.get("policy", "static"), rows=rows,
                plan=data.get("plan", {}), wall_s=data["observation_end_s"])


def pure_cost_table(cells):
    """Measured median pure-decode cost per width; the saving comes from this."""
    pooled = defaultdict(list)
    for cell in cells:
        for row in cell["rows"]:
            pooled[row["width"]].append(row["exec_ms"])
    return {w: st.median(v) for w, v in pooled.items() if len(v) >= 20}


def classify(cells, cost_table, prompt_tokens, max_nudge):
    """Per-step classification of the nudging opportunity."""
    per_cell = []
    for cell in cells:
        buckets = defaultdict(int)
        legal_fill, illegal_withhold, already_aligned, no_queue, too_far = 0, 0, 0, 0, 0
        gross_saving_ms, net_saving_ms, fill_cost_ms = 0.0, 0.0, 0.0
        details = []
        for row in cell["rows"]:
            width = row["width"]
            if width in CAPTURE_SIZES:
                already_aligned += 1
                continue
            up_target = next_capture_at_or_above(width)
            gap_up = up_target - width if up_target else None
            down_target = max((s for s in CAPTURE_SIZES if s < width), default=None)
            gap_down = width - down_target if down_target else None
            # Downward nudge would require dropping a decoding request: illegal.
            if gap_down is not None and gap_down <= max_nudge:
                illegal_withhold += 1
            if gap_up is None or gap_up > max_nudge:
                too_far += 1
                continue
            if row["waiting"] < gap_up:
                no_queue += 1
                continue
            # Legal: enough queued requests exist to fill to the capture point.
            paid_now = cost_table.get(width)
            paid_filled = cost_table.get(up_target)
            if paid_now is None or paid_filled is None:
                too_far += 1
                continue
            # Filling costs one prefill interleave for gap_up requests.
            cost = TAX_FIXED_MS + TAX_PER_PREFILL_TOKEN_MS * prompt_tokens * gap_up
            # Saving: the same paid graph width now carries gap_up more requests,
            # so per-request cost drops. Charge per-request to stay comparable.
            saving = paid_now - paid_filled * width / up_target
            legal_fill += 1
            buckets[gap_up] += 1
            gross_saving_ms += max(0.0, saving)
            fill_cost_ms += cost
            net_saving_ms += saving - cost
            details.append(dict(step=row["step"], width=width, to=up_target, gap=gap_up,
                                waiting=row["waiting"], saving_ms=saving, cost_ms=cost))
        total = len(cell["rows"])
        per_cell.append(dict(
            cell=cell["path"], regime=cell["regime"], cap=cell["cap"], policy=cell["policy"],
            ladder=cell["plan"].get("ladder_name"), n_pure_steps=total,
            already_aligned=already_aligned, already_aligned_share=already_aligned / total,
            legal_fill=legal_fill, legal_fill_share=legal_fill / total,
            blocked_no_queue=no_queue, blocked_no_queue_share=no_queue / total,
            gap_too_far=too_far, illegal_withhold_candidates=illegal_withhold,
            gap_histogram=dict(sorted(buckets.items())),
            gross_saving_ms=gross_saving_ms, fill_cost_ms=fill_cost_ms,
            net_saving_ms=net_saving_ms,
            net_saving_share_of_episode=net_saving_ms / (cell["wall_s"] * 1000.0),
            examples=details[:5]))
    return per_cell


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-nudge", type=int, default=2)
    parser.add_argument("--prompt-tokens", type=int, default=128)
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)

    cells = []
    for root in args.results_dir:
        for path in sorted(glob.glob(str(Path(root) / "**" / "cell-*.json"), recursive=True)):
            cell = load_cell(path)
            if cell:
                cells.append(cell)
    cost_table = pure_cost_table(cells)
    per_cell = classify(cells, cost_table, args.prompt_tokens, args.max_nudge)

    by_regime = defaultdict(list)
    for row in per_cell:
        by_regime[row["regime"]].append(row)
    summary = {}
    for regime, rows in sorted(by_regime.items()):
        steps = sum(r["n_pure_steps"] for r in rows)
        summary[regime] = dict(
            n_episodes=len(rows), n_pure_steps=steps,
            already_aligned_share=sum(r["already_aligned"] for r in rows) / steps,
            legal_fill_share=sum(r["legal_fill"] for r in rows) / steps,
            blocked_no_queue_share=sum(r["blocked_no_queue"] for r in rows) / steps,
            illegal_withhold_share=sum(r["illegal_withhold_candidates"] for r in rows) / steps,
            median_net_saving_share=st.median([r["net_saving_share_of_episode"] for r in rows]),
            max_net_saving_share=max(r["net_saving_share_of_episode"] for r in rows),
            n_episodes_with_positive_net=sum(r["net_saving_ms"] > 0 for r in rows))

    verdict_lines = []
    for regime, v in summary.items():
        if v["legal_fill_share"] < 0.05:
            verdict_lines.append(f"{regime}: ACTION_SPACE_ABSENT "
                                 f"(legal fill steps {v['legal_fill_share']:.4f} < 0.05)")
        elif v["max_net_saving_share"] <= 0:
            verdict_lines.append(f"{regime}: ACTION_SPACE_PRESENT_BUT_NET_NEGATIVE "
                                 f"(best episode {v['max_net_saving_share']:.5f})")
        else:
            verdict_lines.append(f"{regime}: CANDIDATE (fill {v['legal_fill_share']:.4f}, "
                                 f"best net {v['max_net_saving_share']:.5f})")

    report = dict(
        evidence_type="OBSERVATIONAL_POSTHOC_ACTION_SPACE_EXISTENCE_CHECK_NO_EXECUTION",
        claim_boundary=("counts legal step-level fill opportunities in already-recorded steps "
                        "and bounds their net value with the measured interleave tax; it is an "
                        "upper bound on benefit and not a policy result"),
        legality_note=("withholding a decoding request would require preemption or KV eviction "
                       "and is counted only as an illegal candidate; filling requires "
                       "waiting_requests >= gap at that instant"),
        max_nudge=args.max_nudge, prompt_tokens=args.prompt_tokens,
        tax_model=dict(fixed_ms=TAX_FIXED_MS, per_prefill_token_ms=TAX_PER_PREFILL_TOKEN_MS),
        capture_sizes=list(CAPTURE_SIZES), n_cells=len(cells),
        measured_pure_cost_ms={str(k): v for k, v in sorted(cost_table.items())},
        summary=summary, mechanical_verdict=verdict_lines, per_cell=per_cell)
    (out / "action_space.json").write_text(json.dumps(report, indent=2) + "\n")

    lines = ["# Step-level width-nudge action space: does it exist?", "",
             f"cells {len(cells)}; max nudge +/-{args.max_nudge} requests; "
             f"prompt {args.prompt_tokens} tokens", "",
             "Withholding a decoding request is illegal under the frozen non-preemption",
             "invariant, so only *filling* upward to a capture point is a candidate action.", "",
             "| regime | episodes | pure steps | already aligned | **legal fill** | "
             "blocked: no queue | illegal withhold candidates | median net share | max net share | "
             "episodes net>0 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for regime, v in summary.items():
        lines.append(f'| {regime} | {v["n_episodes"]} | {v["n_pure_steps"]} | '
                     f'{v["already_aligned_share"]:.4f} | **{v["legal_fill_share"]:.4f}** | '
                     f'{v["blocked_no_queue_share"]:.4f} | {v["illegal_withhold_share"]:.4f} | '
                     f'{v["median_net_saving_share"]:.6f} | {v["max_net_saving_share"]:.6f} | '
                     f'{v["n_episodes_with_positive_net"]} |')
    lines += ["", "## Mechanical verdict", ""] + [f"- {line}" for line in verdict_lines]
    lines += ["", "## Measured pure-decode cost used for the saving term", "",
              "| width | median ms |", "|---:|---:|"]
    for width, ms in sorted(cost_table.items()):
        lines.append(f'| {width} | {ms:.3f} |')
    lines += ["", "## Per-episode detail", "",
              "| cell | regime | cap | policy | ladder | pure steps | aligned | legal fill | "
              "no queue | gaps | gross ms | fill cost ms | net ms |",
              "|---|---|---:|---|---|---:|---:|---:|---:|---|---:|---:|---:|"]
    for r in per_cell:
        lines.append(f'| {Path(r["cell"]).parent.name}/{Path(r["cell"]).name} | {r["regime"]} | '
                     f'{r["cap"]} | {r["policy"]} | {r["ladder"]} | {r["n_pure_steps"]} | '
                     f'{r["already_aligned"]} | {r["legal_fill"]} | {r["blocked_no_queue"]} | '
                     f'{r["gap_histogram"]} | {r["gross_saving_ms"]:.1f} | '
                     f'{r["fill_cost_ms"]:.1f} | {r["net_saving_ms"]:.1f} |')
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print(f"cells {len(cells)}; wrote {out}")
    for line in verdict_lines:
        print("  " + line)


if __name__ == "__main__":
    main()
