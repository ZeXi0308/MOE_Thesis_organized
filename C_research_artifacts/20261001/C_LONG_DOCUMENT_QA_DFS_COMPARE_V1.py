#!/usr/bin/env python3
"""Compare one full-150 native source-order cell with one author DFS-order cell."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics


N = 150
CAP = 64
WORKLOAD_SHA = "3e34bc8a46abc1329744e22b6c97b18582a540f0abd98de55f012f560aac92fc"
RESOURCE_KEYS = ("max_model_len", "max_num_seqs", "max_num_batched_tokens",
                 "kv_cache_memory_bytes", "enable_prefix_caching",
                 "enable_chunked_prefill", "scheduler_reserve_full_isl",
                 "scheduling_policy", "seed", "dtype", "async_scheduling")


def load(path: Path):
    raw = path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def summary(values: list[float | int]) -> dict:
    ordered = sorted(values)
    return dict(count=len(ordered), minimum=ordered[0], mean=statistics.mean(ordered),
                median=statistics.median(ordered),
                p95_nearest_rank=ordered[math.ceil(.95 * len(ordered)) - 1],
                maximum=ordered[-1])


def directions(deltas: list[float], *, smaller_is_better: bool) -> dict:
    return dict(improved=sum(d < 0 if smaller_is_better else d > 0 for d in deltas),
                worsened=sum(d > 0 if smaller_is_better else d < 0 for d in deltas),
                same=sum(d == 0 for d in deltas))


def read_cell(run: Path, quality_path: Path) -> dict:
    names = ("config.json", "engine_args.json", "status.json",
             "measured-outputs.json", "measured-steps.json", "resolved-eos.json")
    data, hashes = {}, {}
    for name in names:
        data[name], hashes[name] = load(run / name)
    quality, quality_sha = load(quality_path)
    config, args, status = (data[name] for name in
                            ("config.json", "engine_args.json", "status.json"))
    outputs, trace, eos = (data[name] for name in
                            ("measured-outputs.json", "measured-steps.json", "resolved-eos.json"))
    if (status.get("status") != "COMPLETE" or status.get("request_count") != N
            or quality.get("status") != "COMPLETE"
            or quality.get("requests_completed") != N
            or quality.get("requests_planned") != N or quality.get("issues")
            or config.get("workload_sha256") != WORKLOAD_SHA
            or config.get("output_tokens") != CAP
            or eos.get("qualification_status") != "QUALIFIED"
            or eos.get("qualified_eos_token_ids") != [50279]
            or not isinstance(outputs, list) or len(outputs) != N):
        raise ValueError("requires complete frozen full-150 cell and qualified EOS")
    for name in ("status.json", "measured-outputs.json", "measured-steps.json"):
        if quality.get("original_run_sha256", {}).get(name) != hashes[name]:
            raise ValueError("quality does not match original " + name)
    if quality.get("frozen_input_sha256", {}).get("workload.json") != WORKLOAD_SHA:
        raise ValueError("quality input identity differs")
    qrows = {row["request_id"]: row for row in quality["per_request"]}
    rows = {}
    for out in outputs:
        rid = out["request_id"]
        ids, times = out["output_token_ids"], out["token_times_s"]
        arrival, finish = out["arrival_s"], out["host_elapsed_s"]
        reason = out["finish_reason"]
        q = qrows.get(rid)
        if (rid in rows or q is None or q.get("status") != "COMPLETE"
                or out.get("finished") is not True
                or not isinstance(ids, list) or not ids or len(ids) > CAP
                or not isinstance(times, list) or len(ids) != len(times)
                or any(type(token) is not int or token < 0 for token in ids)
                or any(type(t) not in (int, float) or not math.isfinite(t) for t in times)
                or any(b < a for a, b in zip(times, times[1:]))
                or arrival != 0 or not 0 <= times[0] <= times[-1] <= finish
                or reason not in ("stop", "length") or out.get("stop_reason") is not None
                or (reason == "stop" and ids[-1] != 50279)
                or (reason == "length" and (len(ids) != CAP or 50279 in ids))
                or q.get("output_tokens") != len(ids)
                or q.get("output_text") != out.get("output_text")
                or q.get("finish_reason") != reason
                or not isinstance(q.get("f1"), (int, float))
                or not 0 <= q["f1"] <= 1):
            raise ValueError("original request, quality, timing, or EOS differs: " + rid)
        distinct = sorted(set(times))
        rows[rid] = dict(source_index=out["source_index"],
            source_id=out["source_id"], prompt_ids=out["prompt_token_ids"],
            question=out["question"], answers=out["answers"],
            arrival_s=arrival, output_ids=ids, output_text=out["output_text"],
            output_tokens=len(ids), finish_reason=reason, f1=q["f1"],
            ttft_s=times[0] - arrival, flow_s=finish - arrival,
            max_distinct_return_gap_s=max(
                (b - a for a, b in zip(distinct, distinct[1:])), default=0.0))
    if (len(rows) != N or sorted(r["source_index"] for r in rows.values()) != list(range(N))
            or sum(r["output_tokens"] for r in rows.values()) != status["output_tokens"]
            or dict(Counter(r["finish_reason"] for r in rows.values()))
               != status["finish_reason_counts"]):
        raise ValueError("output inventory or status count differs")
    calls = trace["scheduler_calls"]
    if (not calls or len(calls) != status["schedule_calls"]
            or any(call.get("call") != i
                   or type(call.get("scheduled_tokens_total")) is not int
                   or call["scheduled_tokens_total"] < 0
                   for i, call in enumerate(calls))):
        raise ValueError("schedule inventory differs")
    preempted = [rid for call in calls for rid in call["preempted_request_ids"]]
    if preempted != trace["preempted_request_ids"] or len(preempted) != status["preemptions"]:
        raise ValueError("preemption receipt differs")
    scheduled = sum(call["scheduled_tokens_total"] for call in calls)
    decode = sum(row["output_tokens"] - 1 for row in rows.values())
    return dict(rows=rows, config=config, args=args, status=status, trace=trace,
                hashes=hashes, quality_sha256=quality_sha,
                scheduled_tokens=scheduled, decode_scheduled_tokens=decode,
                deduced_prompt_scheduled_tokens=scheduled - decode if not preempted else None,
                prompt_deduction_scope=("P+O-1 one-pass accounting with EOS in output IDs"
                    if not preempted else "NULL: preemption/recomputation prevents one-pass prompt deduction"))


def compare(base: dict, dfs: dict) -> dict:
    if (base["config"]["model"] != dfs["config"]["model"]
            or any(base["args"].get(key) != dfs["args"].get(key)
                   for key in RESOURCE_KEYS)
            or set(base["rows"]) != set(dfs["rows"])):
        raise ValueError("model, resources, or request inventory differ")
    per_request = []
    for rid in sorted(base["rows"], key=lambda rid: base["rows"][rid]["source_index"]):
        a, b = base["rows"][rid], dfs["rows"][rid]
        if any(a[key] != b[key] for key in
               ("source_index", "source_id", "prompt_ids", "question", "answers", "arrival_s")):
            raise ValueError("prompt, reference, or arrival differs: " + rid)
        per_request.append(dict(request_id=rid, source_index=a["source_index"],
            output_ids_changed=a["output_ids"] != b["output_ids"],
            output_text_changed=a["output_text"] != b["output_text"],
            output_length_delta=b["output_tokens"] - a["output_tokens"],
            finish_reason_before=a["finish_reason"], finish_reason_after=b["finish_reason"],
            f1_before=a["f1"], f1_after=b["f1"], f1_delta=b["f1"] - a["f1"],
            ttft_delta_s=b["ttft_s"] - a["ttft_s"],
            flow_delta_s=b["flow_s"] - a["flow_s"],
            max_distinct_return_gap_delta_s=(
                b["max_distinct_return_gap_s"] - a["max_distinct_return_gap_s"])))
    def metric(key: str, delta_key: str, smaller: bool) -> dict:
        differences = [row[delta_key] for row in per_request]
        return dict(baseline=summary([row[key] for row in base["rows"].values()]),
                    dfs=summary([row[key] for row in dfs["rows"].values()]),
                    deltas=summary(differences),
                    direction=directions(differences, smaller_is_better=smaller))
    counts = dict(ids_changed=sum(r["output_ids_changed"] for r in per_request),
                  text_changed=sum(r["output_text_changed"] for r in per_request),
                  length_changed=sum(r["output_length_delta"] != 0 for r in per_request),
                  finish_changed=sum(r["finish_reason_before"] != r["finish_reason_after"]
                                     for r in per_request),
                  f1_changed=sum(r["f1_delta"] != 0 for r in per_request))
    reorder = dfs["status"].get("author_default_dfs_reorder_s")
    if (type(reorder) not in (int, float) or not math.isfinite(reorder)
            or reorder < 0 or reorder != dfs["trace"].get("author_default_dfs_reorder_s")):
        raise ValueError("DFS reorder overhead receipt differs")
    def arm(cell: dict) -> dict:
        status = cell["status"]
        return dict(episode_s=status["observation_end_s"],
                    output_tokens_total=status["output_tokens"],
                    finish_reason_counts=status["finish_reason_counts"],
                    preemption_events=status["preemptions"],
                    schedule_calls=status["schedule_calls"],
                    scheduled_tokens_total=cell["scheduled_tokens"],
                    decode_scheduled_tokens_one_pass=cell["decode_scheduled_tokens"],
                    deduced_prompt_scheduled_tokens=cell["deduced_prompt_scheduled_tokens"],
                    prompt_deduction_scope=cell["prompt_deduction_scope"])
    return dict(schema="c-longbench-multifieldqa-en-author-dfs-compare-v1",
        scope="One later author-default DFS enqueue-order cell versus earlier source-order native cell; descriptive policy-level comparison, not a matched repetition or full PEEK reproduction",
        requests=N, output_cap=CAP, common_workload_sha256=WORKLOAD_SHA,
        baseline=arm(base), dfs=arm(dfs), dfs_reorder_s_included_in_host_origin=reorder,
        episode_difference_s=(dfs["status"]["observation_end_s"]
                              - base["status"]["observation_end_s"]),
        output_change_counts=counts,
        output_length_direction=dict(longer=sum(r["output_length_delta"] > 0 for r in per_request),
                                     shorter=sum(r["output_length_delta"] < 0 for r in per_request),
                                     same=sum(r["output_length_delta"] == 0 for r in per_request)),
        finish_transition_counts=dict(Counter(
            r["finish_reason_before"] + "->" + r["finish_reason_after"] for r in per_request)),
        f1=metric("f1", "f1_delta", False),
        host_ttft_s=metric("ttft_s", "ttft_delta_s", True),
        host_flow_s=metric("flow_s", "flow_delta_s", True),
        maximum_distinct_host_return_gap_s=metric(
            "max_distinct_return_gap_s", "max_distinct_return_gap_delta_s", True),
        per_request=per_request,
        original_sha256=dict(baseline=base["hashes"], dfs=dfs["hashes"],
                             baseline_quality=base["quality_sha256"],
                             dfs_quality=dfs["quality_sha256"]),
        limitations=[
            "All 150 requests remain paired by ID. Different output IDs and lengths mean different generated work; timing deltas do not prove equal-work speedup.",
            "DFS reorder CPU time is inside its host observation origin; this is one later cell, not a matched repeated experiment.",
            "Only the author's guarded offline DFS enqueue order is compared, not online PEEK lanes, eviction policy, or full system.",
            "Maximum distinct host-return gap omits within-return token timing; no new SLO threshold is applied.",
            "Deduced prompt scheduled work subtracts O-minus-one only with zero preemption and qualified EOS. It is not a direct cache-hit or eviction counter.",
        ])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-run-dir", type=Path, required=True)
    parser.add_argument("--baseline-quality", type=Path, required=True)
    parser.add_argument("--dfs-run-dir", type=Path, required=True)
    parser.add_argument("--dfs-quality", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(read_cell(args.baseline_run_dir, args.baseline_quality),
                     read_cell(args.dfs_run_dir, args.dfs_quality))
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, ensure_ascii=False)
        stream.write("\n")
    print(json.dumps(dict(requests=result["requests"],
        output_change_counts=result["output_change_counts"],
        episode_difference_s=result["episode_difference_s"])))


if __name__ == "__main__":
    main()
