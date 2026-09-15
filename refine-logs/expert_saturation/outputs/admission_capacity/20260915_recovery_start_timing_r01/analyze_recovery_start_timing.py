"""Six fixed start-time cells: descriptive performance and separate recovery diagnostics."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from statistics import mean


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def performance(raw):
    rows = []
    for r in raw["requests"]:
        ts, tokens = r["token_times_s"], r["output_token_ids"]
        if len(ts) != len(tokens) or any(b < a for a, b in zip(ts, ts[1:])):
            raise ValueError("token/time alignment")
        unique = sorted(set(ts))  # A multi-token chunk has no resolved intra-chunk gap.
        rows.append(dict(request_id=r["request_id"], document_id=r["document_id"], status=r["status"],
            arrival_s=r["arrival_s"], completion_s=r.get("completion_s"), stop_reason=r.get("stop_reason"),
            outputs=len(tokens), output_sha256=digest(tokens), prompt_sha256=r["prompt_token_ids_sha256"],
            max_output=r.get("max_output_tokens"), last_output_s=ts[-1] if ts else None,
            completion_latency_s=None if r.get("completion_s") is None else r["completion_s"]-r["arrival_s"],
            ttft_s=ts[0]-r["arrival_s"] if ts else None,
            max_engine_return_gap_s=max((b-a for a, b in zip(unique, unique[1:])), default=None)))
    values = lambda key: [r[key] for r in rows if r[key] is not None]
    if len({r["request_id"] for r in rows}) != len(rows):
        raise ValueError("duplicate request identity")
    wall = raw["observation_end_s"]
    complete = raw.get("status") == "COMPLETE" and bool(rows) and all(r["status"] == "completed" and r["completion_s"] is not None for r in rows)
    return dict(complete=complete, requests=rows, request_count=len(rows), full_wall_s=wall, requests_completed=sum(r["status"] == "completed" for r in rows),
        request_rate_s=len(rows)/wall if complete and wall > 0 else None, total_output_tokens=sum(r["outputs"] for r in rows),
        output_token_rate_s=sum(r["outputs"] for r in rows)/wall if complete and wall > 0 else None,
        mean_completion_s=mean(values("completion_latency_s")) if values("completion_latency_s") else None,
        mean_ttft_s=mean(values("ttft_s")) if values("ttft_s") else None,
        max_engine_return_gap_s=max(values("max_engine_return_gap_s"), default=None),
        ttft_request_count=len(values("ttft_s")), gap_request_count=len(values("max_engine_return_gap_s")),
        actual_stop_counts=dict(Counter(r["stop_reason"] for r in rows)), chunk_diagnostics=raw.get("host_chunk_diagnostics"))


def eligibility(s, rid, origin, dispatch, completions, block_size=16):
    r = s["requests"].get(rid)
    if r is None or r["status"].split(".")[-1] != "PREEMPTED":
        return None
    held = lambda r: r["held_blocks"] if isinstance(r["held_blocks"], int) else sum(r["held_blocks"])
    need = max(0, (r["prompt"]+r["output"]+block_size-1)//block_size-held(r))
    reserve = s["protected_reserve"] if s["protected_id"] not in (None, rid) else 0
    free, tracker = s["free_blocks"]-reserve, s["tracker"]
    cfg, step = tracker["config"], s["step"]
    candidates = [v for v in s["running_ids"] if v != rid
        and max(0, s["requests"][v]["computed"]-s["requests"][v]["prompt"])/max(1, s["requests"][v]["max_output"]) < cfg["protect_progress_fraction"]
        and tracker["absence_count"].get(v, 0) < cfg["max_absences_per_request"]
        and step-tracker["resident_since"].get(v, -10**9) >= cfg["min_residency_steps"]]
    victim = min(candidates, key=lambda v: (-s["requests"][v]["output"], v)) if candidates else None
    v = s["requests"].get(victim, {})
    preparable = bool(v) and v["status"].split(".")[-1] == "RUNNING" and v["output"] > 0 and v["computed"] == v["prompt"]+v["output"]-1 and v["output"]+1 < v["max_output"]
    pending = [d["job_id"] for d in dispatch if d["accepted"] and d["is_store"] and d["request"] in (rid, victim)
        and d["before_perf_s"] <= s["host_perf_counter_s"] < completions.get(str(d["job_id"]), float("inf"))]
    return dict(step=step, time_s=s["host_perf_counter_s"]-origin, target_status=r["status"], need_blocks=need,
        effective_free_blocks=free, other_protected_reserve=reserve, direct=free >= need,
        fixed_most_funded=preparable and free+held(v) >= need, victim=victim, victim_held_blocks=held(v) if v else None,
        victim_prepare_growth_blocks=max(0, (v["prompt"]+v["output"]+block_size-1)//block_size-held(v)) if v else None,
        pending_store_jobs=pending, cohort_active=s["cohort_active"], plan_victim=s["plan_victim"], plan_target=s["plan_target"],
        absence_steps=step-tracker["absent_since"].get(rid, step), steps_since_swap=step-tracker["last_swap_step"])


def diagnostic(raw, selective, offload):
    origin = raw["measurement_origin_perf_counter_s"]
    dispatch = offload["dispatch"]
    completed_jobs = {str(j["job_id"]): e["time_s"] for e in offload["completed_jobs"] for j in e["jobs"]}
    calls = {step: c for c in raw["engine_steps"] if c["completed"]
        for step in range(c["scheduler_step_start"], c["scheduler_step_end"])}
    by_request, mapped = {}, raw["internal_to_source"]
    for p in raw["preemption_events"]:
        if p.get("original_preemption_returned", True):
            by_request.setdefault(p["victim_internal_request_id"], []).append(p)
    rows, requests = [], {r["request_id"]: r for r in raw["requests"]}
    for rid, preempts in by_request.items():
        r = requests[mapped[rid]]
        for index, p in enumerate(preempts):
            nxt = preempts[index+1] if index+1 < len(preempts) else None
            start = p["host_perf_counter_s"]-origin
            end = nxt["host_perf_counter_s"]-origin if nxt else raw["observation_end_s"]
            count = p["victim_state"]["output_tokens"]
            stop_count = nxt["victim_state"]["output_tokens"] if nxt else len(r["token_times_s"])
            outputs = r["token_times_s"][count:stop_count]
            executed = [(s, x, calls[s["step"]]) for s in raw["scheduler_steps"] if s["step"] in calls
                and s["step"] >= p["attempted_step"] and (nxt is None or s["step"] < nxt["attempted_step"])
                for x in s["scheduled"] if x["internal_request_id"] == rid]
            loads = [d for d in dispatch if d["accepted"] and not d["is_store"] and d["request"] == rid
                and start <= d["before_perf_s"]-origin < end]
            starts = [dict(time_s=d["before_perf_s"]-origin, boundary="host_submit_load", job_id=d["job_id"],
                returned_s=d["after_perf_s"]-origin) for d in loads]
            if executed:
                s, x, call = executed[0]
                starts.append(dict(time_s=call["start_s"], boundary="host_engine_call", step=s["step"],
                    returned_s=call["returned_s"], preemption_inside_call=call["start_s"] < start))
            host_calls = [executed[0][2]] if executed else []
            if loads:
                submitted = min(d["before_perf_s"] for d in loads)-origin
                host_calls += [c for c in raw["engine_steps"] if c["completed"] and c["start_s"] <= submitted <= c["returned_s"]]
            first_call = min(host_calls, key=lambda c: c["start_s"]) if host_calls else None
            states = [eligibility(s, rid, origin, dispatch, completed_jobs) for s in selective["eligibility_snapshots"]
                if start <= s["host_perf_counter_s"]-origin < end]
            states = [s for s in states if s is not None]
            first = lambda key: next((s for s in states if s[key]), None)
            terminal = r.get("completion_s") if nxt is None else None
            to_first_output = [(s, x, c) for s, x, c in executed if not outputs or c["returned_s"] <= outputs[0]]
            rows.append(dict(request_id=mapped[rid], internal_request_id=rid, preempt_step=p["attempted_step"], preempt_s=start,
                preempt_state_before=p["victim_state"], preempt_state_after=p.get("victim_state_after"),
                L_s=r["token_times_s"][count-1] if count else None,
                E_direct=first("direct"), E_fixed_most_funded=first("fixed_most_funded"), eligible_observations=len(states),
                S=min(starts, key=lambda x: x["time_s"]) if starts else None, recovery_starts=starts,
                S_host_engine_call=None if first_call is None else dict(time_s=first_call["start_s"],
                    returned_s=first_call["returned_s"], call_index=first_call.get("call_index"), boundary="host_engine_call"),
                F_s=outputs[0] if outputs else None, useful_outputs=len(outputs),
                next_output_after_preempt_s=r["token_times_s"][count] if count < len(r["token_times_s"]) else None,
                end="repreempted" if nxt else ("completed" if terminal is not None else "unfinished"), completion_s=terminal,
                next_preempt_step=nxt["attempted_step"] if nxt else None,
                confirmed_recompute_tokens=sum(x["recompute_tokens"] for _, x, _ in executed),
                recompute_tokens_to_first_output=sum(x["recompute_tokens"] for _, x, _ in to_first_output),
                confirmed_scheduled_tokens=sum(x["scheduled_tokens"] for _, x, _ in executed),
                positive_computed_adjustments=sum(max(0, x["computed_adjustment"]) for _, x, _ in executed),
                actual_load_dispatches=loads, executed_steps=[dict(step=s["step"], **x) for s, x, _ in executed],
                discarded_computed_tokens=nxt["victim_state"]["computed_tokens"] if nxt else None,
                discarded_block_counts=nxt["victim_state"].get("block_counts") if nxt else None))
    resumed = [r for r in rows if r["S"] is not None]
    return dict(segments=rows, counts=dict(preemptions=len(rows), actual_recoveries=len(resumed),
        zero_output_reinvalidated=sum(r["end"] == "repreempted" and r["useful_outputs"] == 0 for r in resumed),
        one_two_output_reinvalidated=sum(r["end"] == "repreempted" and 1 <= r["useful_outputs"] <= 2 for r in resumed),
        longer_reinvalidated=sum(r["end"] == "repreempted" and r["useful_outputs"] > 2 for r in resumed),
        completed_recoveries=sum(r["end"] == "completed" for r in resumed)),
        selective_events=selective.get("events", []), selector_decisions=selective.get("selector_decisions", []), offload=offload)


def cell(folder, diagnostic_cell):
    out = dict(cell=folder.name, status="UNRUN", comparable=False, sources={}, errors=[])
    data = {}
    for name in ("status", "raw", "config", "selective-store", "offload-events"):
        path = folder/(name+".json")
        if path.exists():
            out["sources"][name] = dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            try:
                data[name] = json.loads(path.read_text())
            except Exception as exc:
                out["errors"].append(f"{name}: {exc}")
    out["status"] = data.get("status", {}).get("status", "UNRUN")
    out["runtime_status"] = data.get("status")
    out["config"] = data.get("config")
    out["host_snapshots"] = {p.name: json.loads(p.read_text()) for p in sorted(folder.glob("*host*.json"))}
    if "raw" in data:
        try:
            out["requests"] = performance(data["raw"])
            out["comparable"] = not diagnostic_cell and out["status"] == "COMPLETE" and out["requests"]["complete"]
            if data.get("config", {}).get("requests", out["requests"]["request_count"]) != out["requests"]["request_count"]:
                raise ValueError("planned request count mismatch")
            if diagnostic_cell:
                out["diagnostic"] = diagnostic(data["raw"], data["selective-store"], data["offload-events"])
        except Exception as exc:
            out["errors"].append(f"analysis: {type(exc).__name__}: {exc}")
            out["comparable"] = False
    return out


def analyze(root):
    names = ("diag-default", "diag-early", "block0-default", "block0-early", "block1-early", "block1-default")
    cells, comparisons = {}, []
    for name in names:
        try:
            cells[name] = cell(root/name, name.startswith("diag"))
        except Exception as exc:
            cells[name] = dict(cell=name, status="ANALYSIS_FAILED", comparable=False, errors=[repr(exc)])
    for block in (0, 1):
        off, on = (cells[f"block{block}-{arm}"] for arm in ("default", "early"))
        row = dict(block=block, status="NOT_COMPARABLE")
        if off["comparable"] and on["comparable"]:
            a, b = off["requests"], on["requests"]
            identity = lambda x: sorted((r["request_id"], r["document_id"], r["prompt_sha256"], r["arrival_s"], r["max_output"]) for r in x["requests"])
            if identity(a) == identity(b):
                row.update(status="COMPLETE", early_minus_default={k: b[k]-a[k] if a[k] is not None and b[k] is not None else None
                    for k in ("full_wall_s", "request_rate_s", "total_output_tokens", "output_token_rate_s", "mean_completion_s", "mean_ttft_s", "max_engine_return_gap_s")})
                a_by_id = {r["request_id"]: r for r in a["requests"]}
                row["request_deltas"] = [dict(request_id=r["request_id"], output_identical=r["output_sha256"] == a_by_id[r["request_id"]]["output_sha256"],
                    completion_latency_early_minus_default_s=r["completion_latency_s"]-a_by_id[r["request_id"]]["completion_latency_s"]) for r in b["requests"]]
            else:
                row["reason"] = "request/input/arrival/max-output identity mismatch"
        comparisons.append(row)
    return dict(cells=cells, performance_comparisons=comparisons, semantics=[
        "L/F are actual engine-return output times; F is within this residency, next_output_after_preempt_s can cross another zero-output preemption. Terminal completion is separate. Episode wall excludes initialization/warmup and later drain.",
        "E is first OBSERVED PREEMPTED full-history GPU funding, with recompute fallback; no native lookup or guarantee of immediate dispatch. Fixed most uses current guards, never alternative-victim search.",
        "E ignores target-priority/cooldown/plan-busy gates and joint prepare-batch growth; context and victim preparation growth are retained. Pending store jobs are not valid host cache. No host-cache hit is inferred from E.",
        "S is accepted load host submission or completed recovery engine-call start, not physical transfer/kernel start. A call can enclose E/preemption; do not interpret sub-call ordering as negative waiting.",
        "S_host_engine_call uses the earliest completed call carrying first accepted load dispatch or recovery computation, providing the same host-call entry boundary in both arms; original S is retained.",
        "Recompute counts are confirmed executed positions, not time savings. Positive computed adjustments are separate; discarded GPU state may still be reusable on host.",
        "Diagnostic timings include instrumentation and never enter performance comparisons. Incomplete/failed cells are retained, never compared."])


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.write_text(json.dumps(analyze(args.results), indent=2, ensure_ascii=False)+"\n")
