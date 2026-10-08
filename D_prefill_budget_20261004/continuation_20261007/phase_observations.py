"""Read-only native phase-transition reconstruction; no GPU or online policy."""
from collections import Counter
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "confirmation_v2/ablation_elapsed_v3"
REFERENCE = BASE / "block01/01_elapsed_replay/raw.json"
ANCHORS = (2, 291, 293, 825)
BUDGETS = (256, 512, 1024, 2048, 3840)


def encoded(value):
    return json.dumps(value, separators=(",", ":"), sort_keys=True).encode()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def allocation(remaining, cap):
    left, served, completed = cap, [], []
    for rid, need in remaining:
        if not left:
            break
        amount = min(left, need)
        served.append((rid, amount))
        left -= amount
        if amount == need:
            completed.append(rid)
    return served, completed, cap - left


def inspect(path):
    raw = json.loads(path.read_text())
    rows = {r["request_id"]: r for r in raw["requests"]}
    progress, order, seen = Counter(), [], set()
    counts, candidate_counts, timeline = Counter(), {p: Counter() for p in BUDGETS}, []
    prefix = hashlib.sha256()
    selected = {j for i in ANCHORS for j in range(max(0, i - 2), i + 3)}
    is_reference, mixed_index = path == REFERENCE, 0
    assert all(r["finished"] and len(r["output_token_ids"]) == r["max_tokens"] for r in rows.values())
    for i, step in enumerate(raw["steps"]):
        start = step["start_s"]
        active = [rid for rid in order if rows[rid]["completion_s"] > start]
        waiting = sorted((rid for rid, r in rows.items() if r["add_s"] <= start and rid not in seen),
                         key=lambda rid: rows[rid]["add_s"])
        remaining = [(rid, rows[rid]["prompt_tokens"] - progress[rid])
                     for rid in active + waiting if rows[rid]["prompt_tokens"] > progress[rid]]
        served, completed, actual_p = allocation(remaining, step["budget"])
        observed = [(r["request_id"], r["prefill_tokens"]) for r in step["requests"] if r["prefill_tokens"]]
        actual_c = [r["request_id"] for r in step["requests"] if r["prefill_tokens"] and
                    r["computed_start"] < rows[r["request_id"]]["prompt_tokens"] <= r["computed_start"] + r["prefill_tokens"]]
        assert (served, completed, actual_p) == (observed, actual_c, step["prefill_tokens"]), (path, i)
        assert actual_p == min(step["budget"], step["prefill_backlog_tokens_before"])
        assert not step["preempted"] and actual_p + step["decode_tokens"] <= 4096
        # E uses actual observed completions only: it is NOT an online input or forecast.
        exited = [rid for rid, r in rows.items() if r["completion_s"] == step["end_s"]]
        next_d = raw["steps"][i + 1]["decode_count_before"] if i + 1 < len(raw["steps"]) else 0
        assert next_d == step["decode_count_before"] + len(actual_c) - len(exited), (path, i)
        if remaining:
            counts[len(actual_c)] += 1
        if is_reference:
            candidates = {p: allocation(remaining, p) for p in BUDGETS}
            if remaining:
                for cap, (_, c, _) in candidates.items():
                    candidate_counts[cap][len(c)] += 1
            if i in selected:
                submitted = sorted((rid for rid, r in rows.items() if r["add_s"] <= start), key=lambda rid: rows[rid]["add_s"])
                output_prefix = {rid: r["output_token_ids"][:sum(t <= start for t in r["token_times_s"])] for rid, r in rows.items()}
                timeline.append(dict(step=i, anchor=i in ANCHORS, completed_mixed_steps_before=mixed_index,
                    raw_pointer=f"/steps/{i}", start_s=start, actual_elapsed_ms=1000 * (step["end_s"] - start),
                    budget=step["budget"], actual_prefill=actual_p, actual_decode=step["decode_tokens"],
                    D=step["decode_count_before"], context_sum=step["decode_context_sum"],
                    C_actual=len(actual_c), completed_prefill_ids=actual_c, E_posthoc=len(exited),
                    generation_completed_ids_posthoc=exited, observed_next_D=next_d,
                    remaining_prompt_head=[dict(request_id=rid, remaining=n, computed_prompt=progress[rid],
                        prompt_tokens=rows[rid]["prompt_tokens"]) for rid, n in remaining[:4]],
                    candidate_completion_counts={p: len(v[1]) for p, v in candidates.items()},
                    candidate_completed_ids={p: v[1] for p, v in candidates.items()},
                    candidate_latency_or_service_prediction=None,
                    waiting_after=step["waiting_count"], kv_fraction_after=step["kv_used_blocks"] / step["kv_total_blocks"],
                    prefix_schedule_sha256=prefix.hexdigest(), prefix_outputs_sha256=hashlib.sha256(encoded(output_prefix)).hexdigest(),
                    submitted_ids_before=submitted, running_order_before=active,
                    last_submission_delay_s=max((rows[r]["add_s"] - rows[r]["arrival_s"] for r in submitted), default=0)))
        prefix.update(encoded(dict(budget=step["budget"], requests=step["requests"])))
        for r in step["requests"]:
            rid = r["request_id"]
            if rid not in seen:
                seen.add(rid)
                order.append(rid)
            progress[rid] += r["prefill_tokens"]
        mixed_index += bool(actual_p and step["decode_tokens"])
    assert all(progress[rid] == r["prompt_tokens"] for rid, r in rows.items())
    summary = dict(raw=str(path.relative_to(ROOT)), raw_sha256=sha(path), steps=len(raw["steps"]),
        requests=len(rows), backlog_steps=sum(counts.values()), actual_completion_count_distribution=dict(counts),
        allocation_mismatches=0, completed_prefill_mismatches=0, next_decode_identity_mismatches=0,
        maximum_running=max(s["running_count"] for s in raw["steps"]))
    return summary, timeline, {p: dict(c) for p, c in candidate_counts.items()}


def main():
    files = sorted(BASE.glob("block[0-9][0-9]/[0-9][0-9]_*/raw.json"))
    assert len(files) == 12, "Use only the 12 complete retained A/R formal runs"
    summaries, timeline, distribution = [], None, None
    for path in files:
        summary, local_timeline, local_distribution = inspect(path)
        summaries.append(summary)
        if path == REFERENCE:
            timeline, distribution = local_timeline, local_distribution
    native = ROOT / "native_scheduler.py"
    lines = native.read_text().splitlines()
    needles = ("# First, schedule the RUNNING requests.", "# Next, schedule the WAITING requests.",
               "num_new_tokens = request.num_tokens - num_computed_tokens", "request.num_computed_tokens += num_scheduled_token")
    report = dict(schema="d-phase-observations-v1", raw_scope="Existing complete A/R runs only; failed and warmup runs excluded explicitly.",
        verification=summaries, total_verified_steps=sum(x["steps"] for x in summaries),
        reference=str(REFERENCE.relative_to(ROOT)), anchors=list(ANCHORS), neighborhood_radius_steps=2,
        timeline=timeline, reference_candidate_completion_distributions=distribution,
        formula="C(P)=number of whole remaining prompt prefixes covered by P in native running-then-waiting order; D_next=D+C-E holds on these observed runs.",
        E_warning="E is observed after execution and used only to validate the identity. Unknown generation exits are forbidden online inputs. D+C is an upper bound, not an exact forecast without E.",
        candidate_warning="Candidate C is allocation geometry at the recorded state. Unexecuted budgets have no latency, output, goodput, or future-state prediction here. Never subtract already-diverged A/R trajectories as same-state counterfactuals.",
        applicability="This resident OLMoE trace has no APC, speculative decoding, KV connector, preemption, multimodal/Mamba alignment, slot pressure or observed allocation restriction. Requalify before extending this arithmetic to other regimes.",
        sources=dict(native=dict(path=str(native.relative_to(ROOT)), sha256=sha(native),
            locations={needle: next(i + 1 for i, line in enumerate(lines) if needle in line) for needle in needles}),
            raw_metadata={"prompt_length": "/requests/*/prompt_tokens", "submitted_time": "/requests/*/add_s",
                "prior_computation": "/steps/*/requests/*/computed_start and cumulative earlier prefill_tokens",
                "actual_work": "/steps/*/requests/*/{prefill_tokens,decode_tokens}",
                "D_context": "/steps/*/{decode_count_before,decode_context_sum}",
                "E_posthoc": "/requests/*/completion_s equals /steps/*/end_s"}),
        prefix_probe_advice="Replay R independently to each anchor; before intervention compare cumulative scheduled identity/position/budget signatures, output-token prefixes, running/waiting order, prompt positions, submitted set and arrival-to-step assignment. Timing drift is measured, not silently retimed. Retain mismatched attempts; do not call them same-state branches. No KV bitwise equality is established by this script.")
    output = Path(__file__).with_suffix(".json")
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(output=str(output), runs=len(summaries), verified_steps=report["total_verified_steps"], anchors=list(ANCHORS))))


if __name__ == "__main__":
    main()
