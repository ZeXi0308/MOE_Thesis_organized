#!/usr/bin/env python3
"""Check the precondition behind the step-floor metric on every sealed episode.

The metric
----------
Decode emits one token per request per step, so a request needing L tokens
occupies at least L decode steps. If every request is resident before any
request finishes, the episode cannot complete in fewer than

    floor = max_i L_i

pure decode steps, and `steps / floor` measures how far a policy is from what
the hardware forces. Unlike the earlier "tail waste" accounting, this has an
absolute zero and is comparable across workloads.

The precondition, and why it needs checking
-------------------------------------------
The bound assumes the batch is fully populated before it starts draining. If
arrivals were spread out far enough that some request is still queued when the
first one completes, that late request extends the makespan beyond `max(L)` for
reasons unrelated to scheduling, and `steps / floor` would flatter the policy.

The check is therefore `last_admission_s < first_completion_s`. A cell that
fails it is reported as `floor_not_applicable`, never silently scored.

Two further caveats are reported rather than corrected:

* The final prefill chunk step also emits a request's first decode token, so the
  true floor is one or two steps below `max(L)`. This shows up as ratios just
  under 1.0 on an unconstrained pool and does not affect the comparisons.
* Steps carrying prefill or recompute are excluded from the numerator. They are
  real time and are reported separately, but their duration is driven by the
  token budget, not by decode width.
"""

import argparse, json
from pathlib import Path


def check(run_dir):
    run_dir = Path(run_dir)
    raw = json.load(open(run_dir / "raw.json"))
    met = json.load(open(run_dir / "metrics.json"))
    cfg = json.load(open(run_dir / "config.json"))

    steps = raw["scheduler_steps"]
    pure = recompute = prefill = 0
    for s in steps[:-1]:
        if s.get("recompute_tokens"):
            recompute += 1
        elif any(r.get("prefill_tokens") for r in s.get("scheduled", [])):
            prefill += 1
        elif s.get("decode_requests", 0) > 0:
            pure += 1

    done = [r for r in met["per_request"] if r["status"] == "completed"]
    lengths = [r["n_output_tokens"] for r in done]
    adm = [r["admission_s"] for r in raw["requests"] if r.get("admission_s") is not None]
    comp = sorted(r["completion_s"] for r in raw["requests"] if r.get("completion_s"))

    if not lengths or not adm or not comp:
        return dict(label=run_dir.name, status="NO_COMPLETIONS")

    floor = max(lengths)
    last_admission, first_completion = max(adm), comp[0]
    applicable = last_admission < first_completion
    return dict(
        label=run_dir.name,
        policy=cfg.get("completion_policy"),
        kv_cache_bytes=cfg.get("fixed_kv_cache_memory_bytes"),
        n_completed=len(done),
        floor_steps=floor,
        distinct_output_lengths=sorted(set(lengths)),
        pure_decode_steps=pure,
        prefill_steps=prefill,
        recompute_steps=recompute,
        step_efficiency=pure / floor,
        last_admission_s=last_admission,
        first_completion_s=first_completion,
        admission_margin_s=first_completion - last_admission,
        floor_applicable=applicable,
        status="OK" if applicable else "FLOOR_NOT_APPLICABLE",
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cell", action="append", required=True, help="path to a completed cell")
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    rows = [check(p) for p in args.cell]
    ok = [r for r in rows if r.get("status") == "OK"]
    bad = [r for r in rows if r.get("status") != "OK"]

    print(f"{'cell':26s} {'policy':9s} {'floor':>6s} {'pure':>6s} {'steps/floor':>11s} "
          f"{'last adm s':>10s} {'1st comp s':>10s} {'ok':>4s}")
    for r in rows:
        if r.get("status") == "NO_COMPLETIONS":
            print(f"{r['label']:26s} {'-':9s} {'-':>6s} {'-':>6s} {'-':>11s} {'-':>10s} {'-':>10s} {'NO':>4s}")
            continue
        print(f"{r['label']:26s} {str(r['policy']):9s} {r['floor_steps']:6d} {r['pure_decode_steps']:6d} "
              f"{r['step_efficiency']:11.3f} {r['last_admission_s']:10.3f} "
              f"{r['first_completion_s']:10.3f} {'yes' if r['floor_applicable'] else 'NO':>4s}")
    print(f"\nprecondition holds on {len(ok)}/{len(rows)} cells")
    if bad:
        print("cells where the floor does not apply (not scored):")
        for r in bad:
            print(f"  {r['label']}: {r['status']}")
    if args.output:
        json.dump(dict(cells=rows, precondition_holds=f"{len(ok)}/{len(rows)}",
                       note="floor = max output length; valid only when the batch is "
                            "fully admitted before the first completion"),
                  open(args.output, "w"), indent=1)


if __name__ == "__main__":
    main()
