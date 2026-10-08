"""Freeze M/N from retained high-pressure development traces, never new runs."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
GROUPS = ("high-normal-r01", "extra-controls-high", "extra-controls-high384")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def number_rule(decode_count):
    return 2048 if decode_count < 24 else 1024 if decode_count < 35 else 256


def analyze(path):
    raw = json.loads(path.read_text())
    requests = {r["request_id"]: r for r in raw["requests"]}
    progress = Counter()
    samples, mismatch, pzero, max_total = [], 0, 0, 0
    assert all(r["finished"] and len(r["output_token_ids"]) == r["max_tokens"]
               for r in requests.values())
    for index, step in enumerate(raw["steps"]):
        started = step["start_s"]
        arrived = {rid for rid, r in requests.items()
                   if r["arrival_s"] <= started and progress[rid] < r["prompt_tokens"]}
        submitted = {rid for rid, r in requests.items()
                     if r["add_s"] <= started and progress[rid] < r["prompt_tokens"]}
        mismatch += arrived != submitted
        pt = sum(r["prefill_tokens"] for r in step["requests"])
        dt = sum(r["decode_tokens"] for r in step["requests"])
        assert pt == step["prefill_tokens"] and dt == step["decode_tokens"]
        assert 0 <= pt <= step["budget"] and pt + dt <= 4096
        assert not step["preempted"]
        max_total = max(max_total, pt + dt)
        if arrived or submitted:
            # Include every backlog step, explicitly retaining pt == 0.
            pzero += pt == 0
            samples.append(dict(step=index, arrived=bool(arrived), submitted=bool(submitted),
                arrived_requests=len(arrived), submitted_requests=len(submitted),
                arrived_prompt_tokens=sum(requests[r]["prompt_tokens"] - progress[r] for r in arrived),
                submitted_prompt_tokens=sum(requests[r]["prompt_tokens"] - progress[r] for r in submitted),
                actual_prefill=pt, cap=step["budget"], decode=step["decode_count_before"],
                context_sum=step["decode_context_sum"],
                duration_ms=1000 * (step["end_s"] - started)))
        assert pt == 0 or submitted
        for r in step["requests"]:
            rid = r["request_id"]
            if r["prefill_tokens"]:
                assert r["computed_start"] == progress[rid]
            progress[rid] += r["prefill_tokens"]
            assert progress[rid] <= requests[rid]["prompt_tokens"]
    assert all(progress[rid] == r["prompt_tokens"] for rid, r in requests.items())
    measures = {}
    for kind in ("arrived", "submitted"):
        selected = [s for s in samples if s[kind]]
        measures[kind] = dict(steps=len(selected), zero_prefill_steps=sum(s["actual_prefill"] == 0 for s in selected),
            sum_actual_prefill=sum(s["actual_prefill"] for s in selected),
            mean_actual_prefill=mean(s["actual_prefill"] for s in selected),
            mean_chosen_cap=mean(s["cap"] for s in selected))
    return dict(policy=raw["policy"], request_count=len(requests),
        prompt_tokens=sum(progress.values()), output_tokens=sum(len(r["output_token_ids"]) for r in requests.values()),
        total_steps=len(raw["steps"]), arrived_submitted_set_mismatches=mismatch,
        max_scheduled_total_tokens=max_total, backlog=measures,
        workload_sha256=json.loads((path.parent.parent / "protocol.json").read_text())["workload_sha256"],
        raw_sha256=digest(path)), samples


def main():
    cells, feedback_samples, hashes = {}, [], {}
    for group in GROUPS:
        for path in sorted((ROOT / group).glob("[0-9]*/raw.json")):
            name = str(path.parent.relative_to(ROOT))
            result, samples = analyze(path)
            cells[name] = result
            hashes[str(path.relative_to(ROOT))] = result["raw_sha256"]
            if result["policy"] == "feedback":
                feedback_samples.extend(dict(s, run=name) for s in samples if s["submitted"])
    assert len(feedback_samples) == 6213
    assert len({r["workload_sha256"] for r in cells.values()}) == 1
    assert all(r["arrived_submitted_set_mismatches"] == 0 for r in cells.values())
    target = mean(s["actual_prefill"] for s in feedback_samples)
    fixed = {}
    for cap in (304, 384):
        selected = [r for r in cells.values() if r["policy"] == "fixed" + str(cap)]
        numerator = sum(r["backlog"]["submitted"]["sum_actual_prefill"] for r in selected)
        denominator = sum(r["backlog"]["submitted"]["steps"] for r in selected)
        actual = numerator / denominator
        fixed[str(cap)] = dict(runs=len(selected), backlog_steps=denominator,
            mean_actual_prefill=actual, relative_difference_from_feedback=actual / target - 1)
    bands = []
    for low, high in ((0, 23), (24, 34), (35, 42), (43, 48), (49, None)):
        ss = [s for s in feedback_samples if s["decode"] >= low and (high is None or s["decode"] <= high)]
        per_cap = {}
        for cap in sorted({s["cap"] for s in ss}):
            z = [s for s in ss if s["cap"] == cap]
            per_cap[str(cap)] = dict(steps=len(z), mean_duration_ms=mean(s["duration_ms"] for s in z),
                mean_decode_context_length=mean(s["context_sum"] / max(1, s["decode"]) for s in z))
        bands.append(dict(decode_min=low, decode_max=high, steps=len(ss), by_cap=per_cap))
    agreement = sum(number_rule(s["decode"]) == s["cap"] for s in feedback_samples)
    report = dict(schema="d-confirmation-v2-controls-development-freeze-v1",
        data_scope="Only retained high-normal-r01, extra-controls-high, extra-controls-high384; no confirmation results used.",
        accounting="Before each engine.step: arrived/submitted requests with cumulative actually scheduled prompt tokens below prompt length. Include P=0 backlog steps. Prefix caching and preemption are absent.",
        semantics=dict(total_scheduled_token_limit=4096, total_limit_constant=True,
            selected_prefill_cap="aggregate prompt-token cap across all requests; not total batched-token cap or per-request limit",
            source="prefill_policy.install adds one shared d_prefill_left, clips running/waiting allocations and debits successful prompt portions only",
            engine_max_num_seqs=192, maximum_possible_total_for_largest_tested_cap=3840 + 192,
            source_sha256={p: digest(ROOT / p) for p in ("prefill_policy.py", "run_normal.py", "fixed-normal-r01/patched_schedule.py", "fixed-normal-r01/engine_args.json")}),
        M=dict(policy="fixed304", cap=304, feedback_backlog_steps=len(feedback_samples),
            feedback_backlog_actual_prefill_mean=target,
            feedback_zero_prefill_backlog_steps=sum(s["actual_prefill"] == 0 for s in feedback_samples),
            fixed_comparisons=fixed,
            rationale="304 matches pooled actual prompt work per backlog step within 0.02%; 384 is a separate competitive fixed baseline, not work-matched."),
        N=dict(policy="decode_v2", formula="2048 if D < 24 else 1024 if D < 35 else 256",
            signal="D = current visible running requests whose computed tokens have reached prompt length",
            levels=[2048, 1024, 256], thresholds=[24, 35], memory=False,
            rationale="Three coarse bands of the existing feedback action distribution. D=24 is the first sustained move from 2048 to 1024; D>=35 is dominated by 256 in the observed long-context backlog. Replaces the untuned D<8 rule.",
            old_feedback_action_agreement_count=agreement,
            old_feedback_action_agreement_fraction=agreement / len(feedback_samples),
            old_feedback_state_shadow_mean_cap=mean(number_rule(s["decode"]) for s in feedback_samples),
            restriction="Shadow action agreement is diagnostic only. N must independently execute all requests; no replay service or goodput claim.",
            development_bands=bands),
        limitation="Identical D has different contexts/costs and feedback actions across trajectory phases; the coarse count rule intentionally tests whether recent execution time has independent value. No claim of optimal count rule or new predictive model.",
        cells=cells, source_hashes=hashes)
    output = Path(__file__).with_name("frozen_controls.json")
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(output), "M": report["M"], "N": {k: report["N"][k] for k in ("formula", "old_feedback_action_agreement_fraction", "old_feedback_state_shadow_mean_cap")}}, indent=2))


if __name__ == "__main__":
    main()
