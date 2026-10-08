"""Diagnostic submission-step replay, NOT normal-arrival serving evidence.

Only the two prompt caps at a frozen anchor differ; native FCFS is unchanged.
This draft refuses to run without a separate frozen design and source hashes.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
import run_confirmation as base
from prefill_policy import Policy


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def episode(engine, work, name, path, target_ms):
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind
    match = re.fullmatch(r"s(291|293)_(hl|lh)_(warm|r[12])", name)
    assert match, name
    anchor, order = int(match[1]), match[2]
    actions = [512, 256] if order == "hl" else [256, 512]
    ledger = json.loads((ROOT / "frozen_prefix.json").read_text())
    assert sha(ROOT / "workload.json") == ledger["workload"]["sha256"]
    assert work == json.loads((ROOT / "workload.json").read_text())
    by_id = {r["request_id"]: r for r in work}
    pending = ledger["requests"]
    assert set(by_id) == {r["request_id"] for r in pending}
    path.mkdir()
    protocol_path = path.parent / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    protocol.update(experiment_kind="DIAGNOSTIC_SUBMISSION_STEP_REPLAY_NOT_NORMAL_ARRIVAL_E2E",
        arrival_mode="Frozen submit step/order; wait for original arrival when necessary, never inject early.",
        warmup="All four anchor/action sequences replay and fully drain before formal measurements.",
        qualification="Post-run exact schedule/output/logical-state prefix comparison; no bitwise KV claim.",
        frozen_diagnostic_design=json.loads((ROOT/"design.json").read_text()))
    base.dump(protocol_path, protocol)
    scheduler = engine.engine_core.engine_core.scheduler
    assert not scheduler.requests and not engine.has_unfinished_requests()
    policy, native = Policy("elapsed_replay", target_ms), scheduler.schedule
    rows = {x["request_id"]: dict(x, prompt_tokens=len(x["prompt_token_ids"]),
        add_s=None, reference_add_s=None, submit_step=None, first_scheduled_s=None,
        completion_s=None, finished=False, token_times_s=[], output_token_ids=[])
        for x in work}
    steps, injected, waits = [], [], []
    prefix_state, prefix_capture_s = None, None
    origin, epoch = time.perf_counter(), time.time()
    def now(): return time.perf_counter() - origin
    def external(rid):
        return engine.output_processor.request_states[rid].external_req_id
    def snapshot():
        running = [external(r.request_id) for r in scheduler.running]
        waiting = [external(r.request_id) for r in scheduler.waiting]
        requests = {external(rid): r for rid, r in scheduler.requests.items()}
        decoding = [requests[r] for r in running
                    if requests[r].num_computed_tokens >= requests[r].num_prompt_tokens]
        return dict(running_ids=running, waiting_ids=waiting, submitted_ids=list(injected),
            request_states=[dict(request_id=rid, computed_tokens=requests[rid].num_computed_tokens,
                prompt_tokens=requests[rid].num_prompt_tokens, output_tokens=len(rows[rid]["output_token_ids"]))
                for rid in running + waiting],
            completed_mixed_steps_before=policy.mixed_steps,
            decode_count_before=len(decoding),
            decode_context_sum=sum(r.num_computed_tokens for r in decoding),
            prefill_backlog_tokens_before=sum(max(0, r.num_prompt_tokens-r.num_computed_tokens)
                for r in requests.values()),
            output_prefix_lengths={rid: len(r["output_token_ids"]) for rid, r in rows.items()})
    def schedule(*args, **kwargs):
        started = now()
        decoding = [r for r in scheduler.running if r.num_computed_tokens >= r.num_prompt_tokens]
        decode_ids = {r.request_id for r in decoding}
        contexts = sum(r.num_computed_tokens for r in decoding)
        tick = time.perf_counter()
        common_cap = policy.choose(len(decoding))
        i = len(steps)
        cap = actions[i-anchor] if anchor <= i < anchor+2 else common_cap
        scheduler.d_prefill_budget = cap
        overhead = (time.perf_counter()-tick)*1e6
        before = {rid: (r.num_computed_tokens, r.num_prompt_tokens, external(rid))
                  for rid, r in scheduler.requests.items()}
        out = native(*args, **kwargs)
        detail, completed = [], []
        for rid, amount in out.num_scheduled_tokens.items():
            previous, prompt, eid = before[rid]
            start = scheduler.requests[rid].num_computed_tokens-amount
            p = min(amount, max(0, prompt-start))
            if rows[eid]["first_scheduled_s"] is None:
                rows[eid]["first_scheduled_s"] = started
            detail.append(dict(request_id=eid, prefill_tokens=p,
                               decode_tokens=amount-p, computed_start=start))
            if start < prompt <= start+p: completed.append(eid)
        pt, dt = sum(r["prefill_tokens"] for r in detail), sum(r["decode_tokens"] for r in detail)
        assert pt <= cap and not out.preempted_req_ids
        assert decode_ids <= set(out.num_scheduled_tokens)
        assert all(out.num_scheduled_tokens[rid] == 1 for rid in decode_ids)
        if anchor <= i < anchor+2: assert pt == cap, "Work matching failed"
        steps.append(dict(start_s=started, end_s=None, budget=cap, common_budget=common_cap,
            prefill_tokens=pt, decode_tokens=dt, decode_count_before=len(decoding),
            decode_context_sum=contexts, completed_prefill_ids=completed,
            prefill_backlog_tokens_before=sum(max(0,p-c) for c,p,e in before.values()),
            prefill_backlog_requests_before=sum(c<p for c,p,e in before.values()),
            schedule_s=now()-started, decision_us=overhead, requests=detail,
            running_count=len(scheduler.running), waiting_count=len(scheduler.waiting),
            kv_used_blocks=scheduler.kv_cache_manager.block_pool.num_gpu_blocks-1-scheduler.kv_cache_manager.block_pool.get_num_free_blocks(),
            kv_total_blocks=scheduler.kv_cache_manager.block_pool.num_gpu_blocks-1,
            preempted=list(out.preempted_req_ids or [])))
        return out
    scheduler.schedule = schedule
    pos = 0
    try:
        while pos < len(pending) or engine.has_unfinished_requests():
            if now() > 180: raise TimeoutError("Full diagnostic drain exceeded 180s")
            i = len(steps)
            batch = []
            while pos < len(pending) and pending[pos]["submit_step"] == i:
                batch.append(pending[pos]); pos += 1
            assert pos == len(pending) or pending[pos]["submit_step"] > i
            if batch:
                earliest_allowed = max(x["arrival_s"] for x in batch)
                wait_start = now()
                did_wait = False
                while now() < earliest_allowed:
                    did_wait = True
                    time.sleep(min(.002, max(0, earliest_allowed-now())))
                if did_wait:
                    waits.append(dict(before_step=i, elapsed_s=now()-wait_start,
                                      required_arrival_s=earliest_allowed))
            for entry in batch:
                item = by_id[entry["request_id"]]
                p = SamplingParams(temperature=0, max_tokens=item["max_tokens"],
                    min_tokens=item["max_tokens"], ignore_eos=True, detokenize=False,
                    output_kind=RequestOutputKind.CUMULATIVE)
                engine.add_request(item["request_id"], {"prompt_token_ids":item["prompt_token_ids"]},
                                   p, arrival_time=epoch+item["arrival_s"])
                added = now()
                rows[item["request_id"]].update(add_s=added, reference_add_s=entry["reference_add_s"],
                    submit_step=i, arrival_delay_s=added-item["arrival_s"],
                    reference_add_drift_s=added-entry["reference_add_s"])
                injected.append(item["request_id"])
            assert engine.has_unfinished_requests(), "Reference submission ledger cannot advance"
            if i == anchor:
                tick = now(); prefix_state = snapshot(); prefix_capture_s = now()-tick
            start = now()
            outputs = engine.step()
            end = now()
            assert len(steps) == i+1
            step = steps[-1]
            step.update(start_s=start, end_s=end)
            policy.observe(end-start, step["prefill_tokens"], step["decode_tokens"])
            for output in outputs:
                row = rows[output.request_id]
                tokens = list(output.outputs[0].token_ids)
                old = len(row["output_token_ids"])
                assert tokens[:old] == row["output_token_ids"]
                row["token_times_s"].extend([end]*(len(tokens)-old))
                row["output_token_ids"] = tokens
                if output.finished:
                    row.update(finished=True, completion_s=end, finish_reason=output.outputs[0].finish_reason)
        assert prefix_state is not None and pos == len(pending) and not scheduler.requests
        assert all(r["finished"] and len(r["output_token_ids"]) == r["max_tokens"] for r in rows.values())
    finally:
        scheduler.schedule = native
        data = dict(policy=name, elapsed_s=now(), requests=list(rows.values()), steps=steps,
            diagnostic_only=True, anchor=anchor, actions=actions, prefix_state=prefix_state,
            prefix_capture_s=prefix_capture_s, explicit_arrival_waits=waits,
            prefix_ledger_sha256=sha(ROOT/"frozen_prefix.json"),
            frozen_design_sha256=sha(ROOT/"design.json"))
        base.dump(path/"raw.json", data)
    return dict(policy=name, elapsed_s=data["elapsed_s"], requests=len(rows),
                output_tokens=sum(len(r["output_token_ids"]) for r in rows.values()),
                diagnostic_only=True, anchor=anchor, actions=actions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--wait-lock", type=float, default=0)
    args = parser.parse_args()
    design = json.loads((ROOT/"design.json").read_text())
    assert design["status_at_freeze"] == "PRE_GPU_FROZEN"
    for relative, digest in design["source_sha256"].items():
        assert sha(ROOT/relative) == digest, relative
    sys.argv = [__file__, "--output", args.output, "--wait-lock", str(args.wait_lock),
        "--workload", "workload.json", "--target-ms", "24",
        "--policies", ",".join(design["policies"]),
        "--warm-policies", ",".join(design["warm_policies"])]
    base.ROOT, base.episode = ROOT, episode
    raise SystemExit(base.main())
