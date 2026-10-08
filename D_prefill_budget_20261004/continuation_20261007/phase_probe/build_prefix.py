"""Freeze diagnostic submission-step replay evidence; reads old CPU JSON only."""
from bisect import bisect_left
from collections import Counter
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "confirmation_v2/ablation_elapsed_v3/block01/01_elapsed_replay/raw.json"
WORK = ROOT / "confirmation_v2/arrivals_01.json"
DRIVER = ROOT / "confirmation_v2/ablation_elapsed_v3/run_confirmation.py"
NATIVE = ROOT / "native_scheduler.py"
ANCHORS = (291, 293, 825)
RAW_SHA = "c71eca51fecc5c260a146633bd145c59c5b175a7e309b28a2b635d2062494b55"


def encoded(value):
    return json.dumps(value, separators=(",", ":"), sort_keys=True).encode()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source(path):
    return dict(path=str(path.relative_to(ROOT)), sha256=sha(path))


def main():
    raw, work = json.loads(RAW.read_text()), json.loads(WORK.read_text())
    protocol = json.loads((RAW.parent.parent / "protocol.json").read_text())
    assert sha(RAW) == RAW_SHA and raw["policy"] == "elapsed_replay"
    assert sha(WORK) == protocol["workload_sha256"]
    assert sha(DRIVER) == protocol["source_sha256"][DRIVER.name]
    rows, steps = {r["request_id"]: r for r in raw["requests"]}, raw["steps"]
    assert len(rows) == len(work) == 160
    assert all(all(rows[w["request_id"]][k] == v for k, v in w.items()) for w in work)
    starts = [s["start_s"] for s in steps]
    assert all(a["end_s"] < b["start_s"] for a, b in zip(steps, steps[1:]))
    submitted = sorted(rows.values(), key=lambda r: r["add_s"])
    assert len({r["add_s"] for r in submitted}) == len(submitted)
    assert [r["request_id"] for r in submitted] == [w["request_id"] for w in sorted(work, key=lambda w: w["arrival_s"])]
    work_index = {w["request_id"]: i for i, w in enumerate(work)}
    ledger, batch_counts, add_to_start, end_to_add = [], Counter(), [], []
    for r in submitted:
        i = bisect_left(starts, r["add_s"])
        assert i < len(steps) and r["arrival_s"] <= r["add_s"] < starts[i]
        if i:
            assert steps[i - 1]["end_s"] < r["add_s"]
            end_to_add.append(r["add_s"] - steps[i - 1]["end_s"])
        add_to_start.append(starts[i] - r["add_s"])
        ledger.append(dict(request_id=r["request_id"], arrival_s=r["arrival_s"],
            reference_add_s=r["add_s"], submit_step=i, submit_order=batch_counts[i],
            reference_workload_index=work_index[r["request_id"]]))
        batch_counts[i] += 1
    by_id = {r["request_id"]: r for r in ledger}
    seen, first_schedule, computed, checkpoints = [], {}, Counter(), {}
    schedule_hash, mixed = hashlib.sha256(), 0
    for i, step in enumerate(steps):
        active = [rid for rid in seen if rows[rid]["completion_s"] > step["start_s"]]
        waiting = [r["request_id"] for r in ledger if r["submit_step"] <= i and r["request_id"] not in first_schedule]
        d_ids = [rid for rid in active if computed[rid] >= rows[rid]["prompt_tokens"]]
        backlog = sum(max(0, rows[rid]["prompt_tokens"] - computed[rid]) for rid in active + waiting)
        assert len(d_ids) == step["decode_count_before"]
        assert sum(computed[rid] for rid in d_ids) == step["decode_context_sum"]
        assert backlog == step["prefill_backlog_tokens_before"] and not step["preempted"]
        if i in ANCHORS:
            outputs = {rid: r["output_token_ids"][:sum(t < step["start_s"] for t in r["token_times_s"])] for rid, r in rows.items()}
            state = dict(running_ids=active, waiting_ids=waiting,
                submitted_ids=[r["request_id"] for r in ledger if r["submit_step"] <= i],
                request_states=[dict(request_id=rid, computed_tokens=computed[rid],
                    prompt_tokens=rows[rid]["prompt_tokens"], output_tokens=len(outputs[rid])) for rid in active + waiting],
                completed_mixed_steps_before=mixed, decode_count_before=len(d_ids),
                decode_context_sum=step["decode_context_sum"], prefill_backlog_tokens_before=backlog)
            checkpoints[str(i)] = dict(before_step=i, reference_start_s=step["start_s"],
                reference_budget=step["budget"], schedule_prefix_sha256=schedule_hash.hexdigest(),
                output_prefix_sha256=hashlib.sha256(encoded(outputs)).hexdigest(),
                inferred_online_state=state, inferred_online_state_sha256=hashlib.sha256(encoded(state)).hexdigest())
        schedule_hash.update(encoded(dict(budget=step["budget"], requests=step["requests"])))
        for q in step["requests"]:
            rid = q["request_id"]
            assert by_id[rid]["submit_step"] <= i and q["computed_start"] == computed[rid]
            if rid not in first_schedule:
                first_schedule[rid] = i
                seen.append(rid)
            computed[rid] += q["prefill_tokens"] + q["decode_tokens"]
        assert step["running_count"] == len(active) + sum(first_schedule[q["request_id"]] == i for q in step["requests"])
        assert step["waiting_count"] == sum(rid not in first_schedule for rid in waiting)
        mixed += bool(step["prefill_tokens"] and step["decode_tokens"])
    assert len(first_schedule) == 160 and max(r["submit_step"] for r in ledger) < min(ANCHORS)
    lines = DRIVER.read_text().splitlines()
    needles = ("rows[item['request_id']]['add_s']=now()", "start = now()", "outputs = engine.step()", "step.update(start_s=start,end_s=end)")
    report = dict(schema="d-phase-prefix-v1", path_base="D_prefill_budget_20261004 root; all source paths are relative to it",
        purpose="Diagnostic submission-engine-step replay only; never a normal wall-clock-arrival E2E workload.",
        reference_raw=source(RAW), workload=source(WORK), reference_driver=source(DRIVER), native_scheduler=source(NATIVE),
        anchors=list(ANCHORS), requests=ledger, checkpoints=checkpoints,
        inference=dict(rule="submit_step = min i with steps[i].start_s >= request.add_s; submit_order is increasing add_s within that step (zero based)",
            evidence="Serial driver records add_s after add_request, then outer start before engine.step; outer start replaces schedule-entry start in raw. No concurrent injection.",
            boundary="This source has strict previous end_s < add_s < next start_s (step 0 has no previous boundary), no equal add_s/start_s, and no tied add_s. Ambiguities fail closed; no rounding tolerance.",
            first_schedule_warning="Submission is not first scheduling: requests may remain in native waiting queue after add_request.",
            driver_lines={n: next(j + 1 for j, line in enumerate(lines) if n in line) for n in needles}),
        verification=dict(requests=len(ledger), reference_steps=len(steps), submission_batches=len(batch_counts),
            latest_submit_step=max(batch_counts), ambiguous_boundaries=0, tied_submission_times=0,
            min_add_to_start_s=min(add_to_start), min_previous_end_to_add_s=min(end_to_add),
            submit_not_first_schedule=sum(r["submit_step"] != first_schedule[r["request_id"]] for r in ledger),
            reconstructed_backlog_decode_context_and_computed_positions_verified=True),
        hash_definition=dict(encoding="UTF-8 json.dumps(value, separators=(',', ':'), sort_keys=True)",
            schedule="SHA256 concatenation, in order, of encoded {budget,requests} for steps [0,anchor); requests retains native list order and full P/D/computed_start details",
            output="SHA256 encoded mapping of every request ID to emitted token IDs with token_times_s < reference anchor start_s, including empty prefixes",
            state="SHA256 encoded inferred_online_state; these are reconstructed visible fields, not a serialized engine or KV snapshot"),
        qualification=dict(required="Before branching, verify all prior schedule signatures and output prefixes, running/waiting order, submitted IDs, per-request computed/output counts and R mixed-step counter. Treat any mismatch as ineligible and retain the attempt.",
            state_scope="Reconstruction assumes this trace's no preemption/APC/speculation and FIFO native running-append/waiting order; it was checked against every step's logged counts/positions/backlog/context. KV tensor equality is not established.",
            arrival_logging="Keep original arrival_s immutable. Record reference_add_s and actual_add_s plus signed actual_add_s-arrival_s and actual_add_s-reference_add_s. Diagnostic injection may precede wall-clock arrival; disclose it.",
            after_branch="All requests are submitted by step 260, before all three anchors. No future arrival retiming is needed after a qualified branch. Main E2E must continue using normal wall-clock arrivals."))
    output = Path(__file__).with_name("frozen_prefix.json")
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(output=str(output), requests=len(ledger), batches=len(batch_counts), anchors=list(ANCHORS))))


if __name__ == "__main__":
    main()
