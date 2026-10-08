#!/usr/bin/env python3
"""Independent CPU verifier; python3 -B verify_viability.py RAW_OR_DIRECTORY [...].

Writes JSON to stdout only. No production policy imports or online controller.
"""
import argparse
from bisect import bisect_right
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

SLO = dict(ttft_s=4, gap_s=.1, completion_s=20)
CAPS = (256, 512, 1024, 2048)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify(path):
    result = dict(raw=str(path), passed=False, checked_steps=0, errors=[])
    index = None
    try:
        data = path.read_bytes()
        raw = json.loads(data)
        result.update(raw_sha256=hashlib.sha256(data).hexdigest(), policy=raw["policy"])
        protocol_path = next((p/"protocol.json" for p in path.parents if (p/"protocol.json").is_file()), None)
        require(protocol_path is not None, "Missing protocol: cannot validate SLO")
        protocol = json.loads(protocol_path.read_text())
        require(protocol["slo"] == SLO, "Primary SLO differs from 4/.1/20")
        require(protocol["legal_prefill_caps"] == list(CAPS), "Legal cap set changed")
        require(raw["policy"] in ("baseline_r", "viability_escape", "timer19"), "Unknown policy")
        if raw["policy"] == "timer19":
            require(isinstance(protocol.get("timer_control"), str) and protocol["timer_control"], "Missing timer_control protocol")
        rows = {r["request_id"]: r for r in raw["requests"]}
        require(rows and len(rows) == len(raw["requests"]), "Empty/duplicate requests")
        require("failed_decisions" in raw, "Missing failed-decision record")
        result["failed_decisions"] = raw["failed_decisions"]
        if raw["failed_decisions"]:
            result["errors"].append("Failed native decisions retained; never treated as executed work")
        require(raw["steps"], "No actual steps")
        # Static prompt metadata plus event times; future completion events are not consumed early.
        adds, finishes = [], []
        for rid, r in rows.items():
            require(r["prompt_tokens"] == len(r["prompt_token_ids"]), f"Prompt length mismatch: {rid}")
            ts = r["token_times_s"]
            require(len(ts) == len(r["output_token_ids"]) and all(math.isfinite(t) for t in ts), f"Invalid output times: {rid}")
            require(all(a <= b for a, b in zip(ts, ts[1:])), f"Nonmonotone output times: {rid}")
            if r.get("add_s") is not None:
                require(math.isfinite(r["add_s"]) and r["add_s"] >= r["arrival_s"], f"Early/invalid add: {rid}")
                adds.append((r["add_s"], rid))
            if r.get("completion_s") is not None:
                finishes.append((r["completion_s"], rid))
        adds.sort(); finishes.sort()
        submitted, completed, admitted = set(), set(), set()
        running, computed, mixed, apos, fpos = [], Counter(), 0, 0, 0
        counts, previous_end = Counter(), -1.0
        for index, step in enumerate(raw["steps"]):
            t, start, end = step["decision_time_s"], step["start_s"], step["end_s"]
            require(end is not None and all(math.isfinite(v) for v in (t, start, end)), "Incomplete/invalid step timing")
            require(previous_end <= start <= t < end, "Decision time outside its real engine.step")
            while apos < len(adds) and adds[apos][0] <= t:
                submitted.add(adds[apos][1]); apos += 1
            while fpos < len(finishes) and finishes[fpos][0] <= t:
                finish, rid = finishes[fpos]
                require(finish <= previous_end and rid in admitted, "Completion not in prior execution history")
                completed.add(rid); fpos += 1
            active = [rid for rid in running if rid not in completed]
            decoders = [rid for rid in active if computed[rid] >= rows[rid]["prompt_tokens"]]
            remaining = {rid: max(0, rows[rid]["prompt_tokens"]-computed[rid]) for rid in submitted-completed}
            backlog = sum(remaining.values())
            require(step["D"] == step["decode_count_before"] == len(decoders), "Predecision D mismatch")
            require(step["decode_context_sum"] == sum(computed[rid] for rid in decoders), "Predecision context mismatch")
            require(step["prefill_backlog_tokens_before"] == backlog, "Predecision backlog token mismatch")
            require(step["prefill_backlog_requests_before"] == sum(v > 0 for v in remaining.values()), "Predecision backlog request mismatch")
            reasons = dict(ttft=0, observed_gap=0, open_gap=0, age=0, any=0)
            for rid in decoders:
                r = rows[rid]
                ts = r["token_times_s"][:bisect_right(r["token_times_s"], t)]
                require(ts and r["arrival_s"] <= ts[0] <= ts[-1] <= previous_end, f"Decoder lacks prior observed output: {rid}")
                failures = (ts[0]-r["arrival_s"] > 4,
                    max((b-a for a, b in zip(ts, ts[1:])), default=0) > .1,
                    t-ts[-1] > .1, t-r["arrival_s"] > 20)
                for key, failed in zip(("ttft", "observed_gap", "open_gap", "age"), failures):
                    reasons[key] += int(failed)
                reasons["any"] += int(any(failures))
            k = len(decoders)-reasons["any"]
            require(step["K"] == k and step["viability_failure_reason_counts"] == reasons, "K/failure-reason mismatch")
            # Check the frozen R recurrence on this run's prior actual mixed steps; no imported controller.
            baseline = 2048 if not decoders else (1024, 2048, 1024, 512, 256)[sum(mixed >= b for b in (1, 9, 73, 113))]
            condition = bool(decoders) and k == 0 and backlog > baseline
            candidate = 2048 if condition else baseline
            requested = candidate if raw["policy"] == "viability_escape" else baseline
            timer_condition = bool(decoders) and t >= 19.0 and backlog > baseline
            if raw["policy"] == "timer19":
                require("timer_condition" in step, "Missing T timer_condition")
                requested = 2048 if timer_condition else baseline
            if "timer_condition" in step:
                require(step["timer_condition"] == timer_condition, "Timer eligibility mismatch")
            require(step["baseline_cap"] == baseline and step["candidate_cap"] == candidate, "Baseline/candidate cap mismatch")
            require(step["escape_condition"] == condition, "Escape eligibility mismatch")
            require(step["requested_cap"] == step["executed_cap"] == step["budget"] == requested and requested in CAPS, "Requested/executed cap mismatch")
            require(not step["preempted"] and step["cancellation_reason"] is None, "Preemption/cancellation in completed step")
            detail = step["requests"]
            require(len({q["request_id"] for q in detail}) == len(detail), "Duplicate scheduled request")
            require([q["request_id"] for q in detail if q["decode_tokens"]] == decoders, "Decode identity/order skipped or altered")
            for q in detail:
                rid, p, d = q["request_id"], q["prefill_tokens"], q["decode_tokens"]
                require(rid in submitted-completed, f"Scheduled unsubmitted/completed request: {rid}")
                require(isinstance(p, int) and isinstance(d, int) and p >= 0 and d >= 0 and p+d > 0, "Invalid actual work")
                require(q["computed_start"] == computed[rid], f"Computed position mismatch: {rid}")
                require(p == min(p+d, max(0, rows[rid]["prompt_tokens"]-computed[rid])), "Actual prefill/decode split mismatch")
                if rid in decoders:
                    require(p == 0 and d == 1, f"Decoder did not execute exactly one token: {rid}")
                if rid not in admitted:
                    admitted.add(rid); running.append(rid)
                computed[rid] += p+d
            p, d = sum(q["prefill_tokens"] for q in detail), sum(q["decode_tokens"] for q in detail)
            require(p == step["prefill_tokens"] <= requested and d == step["decode_tokens"] == len(decoders), "Actual aggregate work mismatch")
            require(p+d <= 4096, "Total step token cap exceeded")
            mixed += bool(p and d); previous_end = end
            counts.update(steps=1, D_positive=bool(decoders), K_zero_with_D=bool(decoders) and k == 0,
                          escape_condition=condition, requested_above_R=requested > baseline, actual_P_above_R=p > baseline)
            result["checked_steps"] += 1
        # Fixed-length/full-drain contract is checked here only; max_tokens never enters K.
        require(all(r["finished"] and r.get("completion_s") is not None and r["token_times_s"]
            and r["completion_s"] == r["token_times_s"][-1] <= previous_end <= raw["elapsed_s"]
            and len(r["output_token_ids"]) == r["max_tokens"] for r in rows.values()), "Incomplete full-drain/output contract")
        require(all(rid in admitted and computed[rid] == r["prompt_tokens"]+len(r["output_token_ids"])-1
                    for rid, r in rows.items()), "Final prompt/decode computation does not cover all outputs")
        result.update(requests=len(rows), counts=dict(counts), passed=not result["errors"])
    except (OSError, ValueError, KeyError, TypeError, StopIteration) as exc:
        result["errors"].append(dict(step=index, error=type(exc).__name__+": "+str(exc)))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    paths = sorted({p for item in args.paths for p in ([item] if item.is_file() else item.rglob("raw.json"))})
    results = [verify(p) for p in paths]
    passed = bool(results) and all(r["passed"] for r in results)
    print(json.dumps(dict(schema="d-independent-viability-verifier-v1", all_supplied_raws_verified=passed,
        scope="Only supplied raw files, including warmups; not campaign completeness or performance. Prestate derives from prior work and past events, never post-native running/KV counters.",
        K_information_boundary="Only current reconstructed decoders and observed output prefixes at decision_time_s; no future output IDs, EOS or max_tokens for K.",
        runs=results, status="PASS_SUPPLIED_RAWS_ONLY" if passed else "FAILED_OR_NO_RAW"), indent=2, ensure_ascii=False, allow_nan=False))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
