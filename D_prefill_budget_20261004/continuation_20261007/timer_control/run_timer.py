"""Normal-arrival action-value probe: only the aggregate prefill cap changes."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
import run_confirmation as base
from prefill_policy import Policy

NAMES = ("timer19", "viability_escape")
LEGAL_CAPS = (256, 512, 1024, 2048)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def viability(decoder_rows, decision_time_s):
    """Observed irreversible SLO failures only; no output budget/EOS lookahead."""
    counts = dict(ttft=0, observed_gap=0, open_gap=0, age=0, any=0)
    for row in decoder_rows:
        first, last = row["first_output_s"], row["last_output_s"]
        assert first is not None and last is not None, "Decoder lacks observed output"
        assert all(math.isfinite(v) for v in (decision_time_s, row["arrival_s"], first, last,
                                              row["max_observed_generation_gap_s"]))
        assert row["arrival_s"] <= first <= last <= decision_time_s
        assert row["max_observed_generation_gap_s"] >= 0
        reasons = dict(ttft=first-row["arrival_s"] > 4,
            observed_gap=row["max_observed_generation_gap_s"] > .1,
            open_gap=decision_time_s-last > .1, age=decision_time_s-row["arrival_s"] > 20)
        for reason, failed in reasons.items():
            counts[reason] += int(failed)
        counts["any"] += int(any(reasons.values()))
    return len(decoder_rows)-counts["any"], counts


def select_cap(name, baseline, D, K, backlog, decision_time_s):
    assert name in NAMES and baseline in LEGAL_CAPS and 0 <= K <= D
    condition = D > 0 and K == 0 and backlog > baseline
    candidate = 2048 if condition else baseline
    timer_condition = D > 0 and decision_time_s >= 19.0 and backlog > baseline
    requested = candidate if name == "viability_escape" else 2048 if timer_condition else baseline
    reason = (("timer19_elapsed" if timer_condition else "timer19_baseline") if name == "timer19"
              else "escape_K_zero" if condition and candidate != baseline else "baseline_already_2048"
              if condition else "no_running_decode" if D == 0 else "viable_decode_present" if K
              else "backlog_within_baseline")
    return candidate, requested, condition, reason


def episode(engine, work, name, path, target_ms):
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind
    assert name in NAMES and target_ms == 24
    path.mkdir()
    design_path = ROOT / "design.json"
    design = json.loads(design_path.read_text())
    design_sha = sha(design_path)
    workload_path = (ROOT/design["workload"]).resolve()
    assert sha(workload_path) == design["workload_sha256"]
    assert work == json.loads(workload_path.read_text())
    protocol_path = path.parent / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    assert protocol["slo"] == dict(ttft_s=4, gap_s=.1, completion_s=20)
    protocol["source_sha256"].update({"timer_control/"+p: sha(ROOT/p)
        for p in ("run_timer.py", "launch.sh", "design.json")})
    protocol.update(experiment_kind="NORMAL_ARRIVAL_TIMER_ABLATION_NOT_CONFIRMATION",
        arrival_mode="Original wall-clock injection loop; no ledger replay or custom waits.",
        warmup="Both policies replay and fully drain the same workload before formal runs.",
        timing="Host engine.step includes both arms' viability diagnostics and native scheduling; outer output bookkeeping remains in full elapsed and output gaps.",
        legal_prefill_caps=list(LEGAL_CAPS), viability_design=design,
        timer_control="T ignores K and uses decision_time_s>=19s with the same D/backlog guards; candidate_cap/escape_condition remain shadow V recommendations in both arms.",
        viability_signal="K counts running decoders with no observed TTFT/gap/open-gap/age failure; reason counts overlap, any counts their union. No future outputs or waiting/unarrived requests enter K.")
    base.dump(protocol_path, protocol)
    scheduler = engine.engine_core.engine_core.scheduler
    assert not scheduler.requests and not engine.has_unfinished_requests()
    policy, native = Policy("elapsed_replay", 24), scheduler.schedule
    rows = {x["request_id"]: dict(x, prompt_tokens=len(x["prompt_token_ids"]),
        add_s=None, first_scheduled_s=None, completion_s=None, finished=False,
        token_times_s=[], output_token_ids=[], first_output_s=None, last_output_s=None,
        max_observed_generation_gap_s=0.0) for x in work}
    assert len(rows) == len(work)
    steps, failed_decisions = [], []
    origin, epoch = time.perf_counter(), time.time()
    def now(): return time.perf_counter()-origin
    def schedule(*args, **kwargs):
        started, tick = now(), time.perf_counter()
        decoding = [r for r in scheduler.running if r.num_computed_tokens >= r.num_prompt_tokens]
        decode_ids = {r.request_id for r in decoding}
        contexts = sum(r.num_computed_tokens for r in decoding)
        before = {rid: (r.num_computed_tokens, r.num_prompt_tokens,
            engine.output_processor.request_states[rid].external_req_id) for rid, r in scheduler.requests.items()}
        backlog = sum(max(0, p-c) for c, p, eid in before.values())
        decision_time_s = now()
        K, failures = viability([rows[before[r.request_id][2]] for r in decoding], decision_time_s)
        baseline = policy.choose(len(decoding))
        candidate, requested, condition, reason = select_cap(name, baseline, len(decoding), K, backlog, decision_time_s)
        scheduler.d_prefill_budget = requested
        executed = scheduler.d_prefill_budget
        assert executed in LEGAL_CAPS
        overhead = (time.perf_counter()-tick)*1e6
        out = None
        try:
            out = native(*args, **kwargs)
            detail = []
            for rid, amount in out.num_scheduled_tokens.items():
                previous, prompt, eid = before[rid]
                start = scheduler.requests[rid].num_computed_tokens-amount
                prefill = min(amount, max(0, prompt-start))
                if rows[eid]["first_scheduled_s"] is None:
                    rows[eid]["first_scheduled_s"] = started
                detail.append(dict(request_id=eid, prefill_tokens=prefill,
                                   decode_tokens=amount-prefill, computed_start=start))
            pt, dt = sum(r["prefill_tokens"] for r in detail), sum(r["decode_tokens"] for r in detail)
            assert pt <= executed and not out.preempted_req_ids
            assert decode_ids <= set(out.num_scheduled_tokens), "Native decoder unexpectedly skipped"
            assert all(out.num_scheduled_tokens[rid] == 1 for rid in decode_ids)
        except Exception as exc:
            failed_decisions.append(dict(step_index=len(steps), decision_time_s=decision_time_s,
                D=len(decoding), K=K, prefill_backlog_tokens_before=backlog,
                viability_failure_reason_counts=failures, baseline_cap=baseline, candidate_cap=candidate,
                requested_cap=requested, executed_cap=executed, escape_condition=condition,
                chosen_reason=reason, cancellation_reason=type(exc).__name__+": "+str(exc),
                native_schedule_returned=out is not None, forward_completed=False,
                actual_prefill_tokens=None, decision_us=overhead))
            raise
        unused = ("timer_did_not_apply_K_candidate" if name == "timer19" and executed != candidate
                  else "native_work_did_not_exceed_baseline_cap" if executed > baseline and pt <= baseline
                  else "known_backlog_below_cap" if pt < executed and pt == backlog
                  else "native_allocation_underfilled_cap" if pt < executed else None)
        steps.append(dict(start_s=started, end_s=None, budget=executed, prefill_tokens=pt,
            decode_tokens=dt, decode_count_before=len(decoding), decode_context_sum=contexts,
            prefill_backlog_tokens_before=backlog,
            prefill_backlog_requests_before=sum(c < p for c, p, eid in before.values()),
            decision_time_s=decision_time_s, D=len(decoding), K=K,
            timer_condition=len(decoding) > 0 and decision_time_s >= 19.0 and backlog > baseline,
            viability_failure_reason_counts=failures, baseline_cap=baseline, candidate_cap=candidate,
            requested_cap=requested, executed_cap=executed, escape_condition=condition,
            chosen_reason=reason, cancellation_reason=None, unused_reason=unused,
            schedule_s=now()-started, decision_us=overhead, requests=detail,
            running_count=len(scheduler.running), waiting_count=len(scheduler.waiting),
            kv_used_blocks=scheduler.kv_cache_manager.block_pool.num_gpu_blocks-1-scheduler.kv_cache_manager.block_pool.get_num_free_blocks(),
            kv_total_blocks=scheduler.kv_cache_manager.block_pool.num_gpu_blocks-1,
            preempted=list(out.preempted_req_ids or [])))
        return out
    scheduler.schedule = schedule
    pending, pos = sorted(work, key=lambda x: x["arrival_s"]), 0
    try:
        while pos < len(pending) or engine.has_unfinished_requests():
            if now() > 180: raise TimeoutError("Complete-service bound exceeded; retain partial results")
            while pos < len(pending) and pending[pos]["arrival_s"] <= now():
                item = pending[pos]
                p = SamplingParams(temperature=0, max_tokens=item["max_tokens"],
                    min_tokens=item["max_tokens"], ignore_eos=True, detokenize=False,
                    output_kind=RequestOutputKind.CUMULATIVE)
                engine.add_request(item["request_id"], {"prompt_token_ids": item["prompt_token_ids"]},
                                   p, arrival_time=epoch+item["arrival_s"])
                rows[item["request_id"]]["add_s"] = now()
                pos += 1
            if not engine.has_unfinished_requests():
                time.sleep(min(.002, max(0, pending[pos]["arrival_s"]-now())))
                continue
            start, before_count = now(), len(steps)
            outputs = engine.step()
            end = now()
            assert len(steps) == before_count+1
            step = steps[-1]
            step.update(start_s=start, end_s=end)
            policy.observe(end-start, step["prefill_tokens"], step["decode_tokens"])
            for output in outputs:
                row = rows[output.request_id]
                tokens = list(output.outputs[0].token_ids)
                old = len(row["output_token_ids"])
                assert tokens[:old] == row["output_token_ids"]
                if len(tokens) > old:
                    if row["first_output_s"] is None: row["first_output_s"] = end
                    if row["last_output_s"] is not None:
                        row["max_observed_generation_gap_s"] = max(row["max_observed_generation_gap_s"], end-row["last_output_s"])
                    row["last_output_s"] = end
                row["token_times_s"].extend([end]*(len(tokens)-old))
                row["output_token_ids"] = tokens
                if output.finished:
                    row.update(finished=True, completion_s=end, finish_reason=output.outputs[0].finish_reason)
        assert all(r["finished"] and len(r["output_token_ids"]) == r["max_tokens"] for r in rows.values())
        assert not scheduler.requests
    finally:
        scheduler.schedule = native
        data = dict(policy=name, elapsed_s=now(), requests=list(rows.values()), steps=steps,
            failed_decisions=failed_decisions,
            experiment_kind="normal_arrival_timer_ablation", frozen_design_sha256=design_sha)
        base.dump(path/"raw.json", data)
    return dict(policy=name, elapsed_s=data["elapsed_s"], requests=len(rows),
        output_tokens=sum(len(r["output_token_ids"]) for r in rows.values()),
        mixed_steps=sum(s["prefill_tokens"] > 0 and s["decode_tokens"] > 0 for s in steps))


if __name__ == "__main__":
    design = json.loads((ROOT/"design.json").read_text())
    assert design["status_at_freeze"] == "PRE_GPU_FROZEN"
    for relative, digest in design["source_sha256"].items():
        assert sha(ROOT/relative) == digest, relative
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--workload", default=design["workload"])
    parser.add_argument("--policies", default=",".join(design["policies"]))
    parser.add_argument("--warm-policies", default=",".join(design["warm_policies"]))
    parser.add_argument("--target-ms", type=float, default=24)
    args, rest = parser.parse_known_args()
    assert args.policies.split(",") == design["policies"] and all(p in NAMES for p in design["policies"])
    assert args.warm_policies.split(",") == design["warm_policies"] and set(design["warm_policies"]) == set(NAMES)
    assert args.workload == design["workload"] and args.target_ms == 24 and "--command-loop" not in rest
    workload_path = (ROOT/args.workload).resolve()
    assert sha(workload_path) == design["workload_sha256"]
    sys.argv = [__file__, *rest, "--workload", str(workload_path), "--policies", args.policies,
                "--warm-policies", args.warm_policies, "--target-ms", "24"]
    base.episode = episode  # Keep parent ROOT, shared lock, engine, bootstrap and native install.
    raise SystemExit(base.main())
