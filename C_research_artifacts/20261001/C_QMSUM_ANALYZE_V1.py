#!/usr/bin/env python3
"""Analyze one full QMSum200 native cell with pinned LongBench Rouge-L.

Scores complete output text, retains every source row in the denominator, and
reports native host observations. This local 4k/OLMoE adaptation is not an
official LongBench model score or a policy comparison.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
from statistics import mean, median

N, CAP, EOS = 200, 512, 50279
TASK = "qmsum"
CODE = Path(__file__).resolve().parent / "20261001_c_longbench_official_code_v1"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def distribution(values):
    if not values:
        return None
    values = sorted(values)
    return dict(count=len(values), minimum=values[0], mean=mean(values),
                median=median(values), p95_nearest_rank=values[math.ceil(.95 * len(values))-1],
                maximum=values[-1])


def longest_repeat(ids):
    """Longest exact, nonoverlapping repeated span; descriptive, no cutoff."""
    following, best = [0] * (len(ids) + 1), (0, None, None)
    for i in range(len(ids) - 1, -1, -1):
        current = [0] * (len(ids) + 1)
        for j in range(len(ids) - 1, i, -1):
            if ids[i] == ids[j]:
                current[j] = 1 + following[j + 1]
                length = min(current[j], j - i)
                if length > best[0]:
                    best = (length, i, j)
        following = current
    return dict(tokens=best[0], first_start=best[1], second_start=best[2])


def frozen_inputs(directory, metadata_dir, code_dir):
    names = ("config.json", "workload.json", "geometry.json", "SOURCE_RECEIPT.json")
    hashes = {name: sha(directory / name) for name in names}
    config, workload, receipt = (load(directory / name) for name in
                                 ("config.json", "workload.json", "SOURCE_RECEIPT.json"))
    if (config.get("schema") != "c-longbench-qmsum-full-config-v1"
            or workload.get("schema") != "c-longbench-qmsum-full-workload-v1"
            or receipt.get("schema") != "c-longbench-qmsum-full-source-receipt-v1"
            or config.get("workload_sha256") != hashes["workload.json"]
            or receipt.get("workload_sha256") != hashes["workload.json"]
            or receipt.get("config_sha256") != hashes["config.json"]
            or receipt.get("geometry_sha256") != hashes["geometry.json"]
            or config.get("requests") != N or config.get("output_tokens") != CAP
            or config.get("max_model_len") != 4096
            or config.get("task") != TASK or workload.get("task") != TASK
            or workload.get("arrival_traces_s") != [0.0] * N
            or workload.get("sampling") != dict(temperature=0.0, max_tokens=CAP,
                                                 min_tokens=0, ignore_eos=False, stop=[])):
        raise ValueError("frozen QMSum input/schema/hash/sampling contract differs")
    rows = workload.get("requests")
    if (not isinstance(rows, list) or len(rows) != N
            or [r.get("source_index") for r in rows] != list(range(N))
            or [r.get("example_index") for r in rows] != list(range(N))
            or len({r.get("request_id") for r in rows}) != N):
        raise ValueError("QMSum source-order request inventory differs")
    for row in rows:
        ids = row.get("prompt_token_ids")
        if (not isinstance(ids, list) or not ids or len(ids) + CAP > 4096
                or not isinstance(row.get("context_sha256"), str)
                or len(row["context_sha256"]) != 64
                or not isinstance(row.get("answers"), list) or not row["answers"]
                or any(not isinstance(a, str) for a in row["answers"])):
            raise ValueError("frozen QMSum request geometry/reference differs")
    tokenizer_path = metadata_dir / "tokenizer.json"
    if (config.get("metadata_sha256", {}).get("tokenizer.json") != sha(tokenizer_path)
            or receipt.get("metadata_sha256", {}).get("tokenizer.json") != sha(tokenizer_path)):
        raise ValueError("frozen QMSum tokenizer differs")
    pinned = receipt.get("official_code", {}).get("files_sha256", {})
    for name in ("LongBench/eval.py", "LongBench/metrics.py"):
        if pinned.get(name) != sha(code_dir / name):
            raise ValueError("official LongBench metric/eval source differs: " + name)
    if ("\"qmsum\": rouge_score" not in (code_dir / "LongBench/eval.py").read_text()
            or 'return scores["rouge-l"]["f"]' not in
            (code_dir / "LongBench/metrics.py").read_text()):
        raise ValueError("official QMSum Rouge-L mapping differs")
    return config, rows, hashes, dict(metric=sha(code_dir / "LongBench/metrics.py"),
                                      eval=sha(code_dir / "LongBench/eval.py"),
                                      tokenizer=sha(tokenizer_path))


def service_mix(mix, trace, status, expected_ids):
    if mix.get("schema") != "c-qmsum-native-service-mix-v1":
        raise ValueError("native service-mix schema differs")
    calls, native = mix.get("scheduler_calls"), trace.get("scheduler_calls")
    if (not isinstance(calls, list) or not isinstance(native, list)
            or len(calls) != status.get("schedule_calls") or len(calls) != len(native)):
        raise ValueError("native service-mix schedule-call inventory differs")
    totals = Counter()
    types = Counter()
    per_step = []
    for index, (record, original) in enumerate(zip(calls, native)):
        if record.get("call") != index or original.get("call") != index:
            raise ValueError("service-mix call index differs")
        served = record.get("requests")
        if not isinstance(served, list):
            raise ValueError("service-mix requests missing")
        if len({r.get("external_request_id") for r in served}) != len(served):
            raise ValueError("duplicate scheduled request in service-mix call")
        for row in served:
            if (row.get("external_request_id") not in expected_ids
                    or any(type(row.get(k)) is not int or row[k] < 0 for k in
                           ("scheduled_tokens", "scheduled_prefill_tokens",
                            "scheduled_decode_tokens", "previous_computed_tokens",
                            "new_prefix_cached_tokens", "prompt_length_tokens"))
                    or row["scheduled_prefill_tokens"] + row["scheduled_decode_tokens"]
                       != row["scheduled_tokens"]):
                raise ValueError("service-mix request token accounting differs")
        prefill = sum(r["scheduled_prefill_tokens"] for r in served)
        decode = sum(r["scheduled_decode_tokens"] for r in served)
        scheduled = sum(r["scheduled_tokens"] for r in served)
        if (record.get("prefill_tokens") != prefill
                or record.get("decode_tokens") != decode
                or record.get("scheduled_tokens") != scheduled
                or original.get("scheduled_tokens_total") != scheduled):
            raise ValueError("service-mix and native schedule totals differ")
        totals.update(prefill_tokens=prefill, decode_tokens=decode,
                      scheduled_tokens=scheduled, scheduled_requests=len(served))
        types["mixed" if prefill and decode else "prefill_only" if prefill else
              "decode_only" if decode else "idle"] += 1
        per_step.append(dict(call=index, scheduled_requests=len(served),
                             scheduled_tokens=scheduled, prefill_tokens=prefill,
                             decode_tokens=decode))
    first = mix.get("first_successful_allocation")
    if (not isinstance(first, list) or len(first) != N
            or len({r.get("external_request_id") for r in first}) != N
            or {r.get("external_request_id") for r in first} != expected_ids):
        raise ValueError("first successful allocation must cover all 200 requests exactly once")
    return dict(classified_scheduled_prefill_tokens=totals["prefill_tokens"],
                classified_scheduled_decode_tokens=totals["decode_tokens"],
                scheduled_tokens_total=totals["scheduled_tokens"],
                scheduled_request_actions=totals["scheduled_requests"],
                schedule_call_types=dict(types), schedule_calls=len(calls),
                per_step=per_step,
                first_successful_allocations=len(first),
                first_allocation_prefix_cached_tokens=sum(
                    r["new_prefix_cached_tokens"] for r in first),
                preemptions=status.get("preemptions"),
                interpretation="Direct native prompt-boundary classification; includes any preemption/recomputation. No Σ(O−1) one-pass deduction or cache-eviction inference.")


def analyze(input_dir, run_dir, metadata_dir, code_dir):
    from rouge import Rouge
    from tokenizers import Tokenizer

    config, inputs, input_hashes, source_hashes = frozen_inputs(
        input_dir, metadata_dir, code_dir)
    if not (run_dir / "status.json").is_file() and (run_dir / "native/status.json").is_file():
        run_dir = run_dir / "native"
    tokenizer = Tokenizer.from_file(str(metadata_dir / "tokenizer.json"))
    scorer = Rouge()
    run_hashes = {p.name: sha(p) for p in sorted(run_dir.glob("*.json"))}
    required = ("status.json", "measured-outputs.json", "measured-steps.json",
                "measured-service-mix.json", "resolved-eos.json", "environment.json",
                "input-tokenizer-check.json", "native-drain.json")
    if any(name not in run_hashes for name in required):
        raise ValueError("required QMSum raw file missing")
    status, outputs, trace, mix, eos, environment, tokenizer_check, drain = (
        load(run_dir / name) for name in required)
    issues = []
    if status.get("status") != "COMPLETE" or status.get("request_count") != N:
        issues.append("native_cell_not_complete200")
    if (eos.get("qualification_status") != "QUALIFIED"
            or eos.get("qualified_eos_token_ids") != [EOS]
            or status.get("eos_qualification_status") != "QUALIFIED"):
        issues.append("runtime_eos_not_qualified")
    if tokenizer_check.get("status") != "QUALIFIED":
        issues.append("runtime_tokenizer_check_not_qualified")
    if drain.get("status") != "QUALIFIED":
        issues.append("native_drain_not_qualified")
    if any(environment.get("input_sha256", {}).get(name) != input_hashes[name]
           for name in ("config.json", "workload.json", "SOURCE_RECEIPT.json")):
        issues.append("environment_input_identity_differs")
    if not isinstance(outputs, list) or len(outputs) != N:
        raise ValueError("measured QMSum output inventory differs")
    by_id = {}
    for row in outputs:
        rid = row.get("request_id")
        if rid in by_id:
            raise ValueError("duplicate measured QMSum request ID")
        by_id[rid] = row
    if set(by_id) != {r["request_id"] for r in inputs}:
        raise ValueError("measured QMSum IDs differ from frozen inputs")

    per_request, context_groups = [], defaultdict(list)
    exact_output_groups = defaultdict(list)
    for source in inputs:
        rid = source["request_id"]
        row = by_id[rid]
        row_issues = ["input_" + key + "_differs" for key in source
                      if row.get(key) != source[key]]
        ids, times = row.get("output_token_ids"), row.get("token_times_s")
        if (not isinstance(ids, list) or not ids or len(ids) > CAP
                or any(type(t) is not int or t < 0 for t in ids)
                or not isinstance(times, list) or len(times) != len(ids)
                or any(type(t) not in (int, float) or not math.isfinite(t) or t < 0
                       for t in times)
                or any(b < a for a, b in zip(times, times[1:]))):
            row_issues.append("invalid_output_ids_or_host_times")
        else:
            finish, arrival = row.get("host_elapsed_s"), row.get("arrival_s")
            if (type(finish) not in (int, float) or not math.isfinite(finish)
                    or arrival != 0.0 or not 0 <= times[0] <= times[-1] <= finish):
                row_issues.append("invalid_host_completion_or_arrival")
        reason = row.get("finish_reason")
        if (row.get("finished") is not True or reason not in ("stop", "length")
                or (reason == "stop" and (not ids or ids[-1] != EOS))
                or (reason == "length" and (not ids or len(ids) != CAP or EOS in ids))
                or row.get("stop_reason") is not None):
            row_issues.append("invalid_finish_or_eos_cap")
        text = row.get("output_text")
        if not isinstance(text, str) or tokenizer.decode(ids if isinstance(ids, list) else [],
                                                         skip_special_tokens=True) != text:
            row_issues.append("output_text_decode_differs")
        valid = not row_issues
        if valid:
            scores = []
            for answer in source["answers"]:
                try:
                    scores.append(scorer.get_scores([text], [answer], avg=True)["rouge-l"]["f"])
                except Exception:
                    scores.append(0.0)  # Exact official metrics.py exception rule.
            rouge_l = max(scores)
            distinct = sorted(set(times))
            ttft, flow = times[0], row["host_elapsed_s"]
            gap = max((b-a for a, b in zip(distinct, distinct[1:])), default=0.0)
            repeated = longest_repeat(ids)
            exact_output_groups[text].append(rid)
            context_groups[source["context_sha256"]].append((source["source_index"], rid, flow))
        else:
            rouge_l, ttft, flow, gap = 0.0, None, None, None
            repeated = None
            issues.extend(rid + ":" + issue for issue in row_issues)
        per_request.append(dict(request_id=rid, source_index=source["source_index"],
            source_id=source["source_id"], context_sha256=source["context_sha256"],
            answers=source["answers"], output_text=text, output_token_ids=ids,
            output_tokens=len(ids) if isinstance(ids, list) else None,
            finish_reason=reason, natural_eos=valid and reason == "stop",
            length_cap=valid and reason == "length", empty_output=valid and not text.strip(),
            rouge_l_f1=rouge_l, ttft_s=ttft, flow_s=flow,
            max_distinct_host_return_gap_s=gap, longest_exact_nonoverlap_repeat=repeated,
            status="COMPLETE" if valid else "INCOMPLETE", issues=row_issues))
    completed = [r for r in per_request if r["status"] == "COMPLETE"]
    lengths = [r["output_tokens"] for r in completed]
    if sum(lengths) != status.get("output_tokens"):
        issues.append("status_output_token_total_differs")
    if dict(Counter(r["finish_reason"] for r in completed)) != status.get("finish_reason_counts"):
        issues.append("status_finish_reason_counts_differ")
    expected_external = {"measured/" + r["request_id"] for r in inputs}
    service = service_mix(mix, trace, status, expected_external)
    groups = [dict(context_sha256=context,
                   source_indices=[r[0] for r in sorted(members)],
                   request_ids=[r[1] for r in sorted(members)],
                   request_count=len(members), completion_host_s=max(r[2] for r in members))
              for context, members in context_groups.items()]
    groups.sort(key=lambda group: min(group["source_indices"]))
    multi = [g for g in groups if g["request_count"] > 1]
    duplicates = [dict(request_ids=rids, count=len(rids))
                  for rids in exact_output_groups.values() if len(rids) > 1]
    try:
        rouge_version = version("rouge")
    except Exception:
        rouge_version = "unavailable-distribution-metadata"
    return dict(schema="c-qmsum-native-analysis-v1",
        status="COMPLETE" if not issues and len(completed) == N else "INCOMPLETE",
        issues=issues, task=TASK, model=config["model"], requests_planned=N,
        requests_completed=len(completed), natural_eos_count=sum(r["natural_eos"] for r in per_request),
        length_cap_count=sum(r["length_cap"] for r in per_request),
        empty_output_count=sum(r["empty_output"] for r in per_request),
        output_tokens_total=sum(lengths), output_token_length=distribution(lengths),
        rouge_l_f1_mean=sum(r["rouge_l_f1"] for r in per_request) / N,
        rouge_l_f1_percent=100 * sum(r["rouge_l_f1"] for r in per_request) / N,
        official_eval_rounded_percent=round(100 * sum(r["rouge_l_f1"] for r in per_request) / N, 2),
        metric="Pinned LongBench qmsum: Rouge().get_scores([full_output], [reference], avg=True)['rouge-l']['f']; max over references; zero on scorer exception",
        rouge_package_version=rouge_version,
        ttft_s=distribution([r["ttft_s"] for r in completed]),
        flow_s=distribution([r["flow_s"] for r in completed]),
        max_distinct_host_return_gap_s=distribution(
            [r["max_distinct_host_return_gap_s"] for r in completed]),
        document_groups=dict(group_count=len(groups), multiquestion_group_count=len(multi),
            group_size_histogram=dict(Counter(g["request_count"] for g in groups)),
            all_completion_s=distribution([g["completion_host_s"] for g in groups]),
            multiquestion_completion_s=distribution([g["completion_host_s"] for g in multi]),
            definition="For each identical original context SHA, latest member host completion; all arrival times zero; no user-joint-completion assumption",
            per_document=groups),
        repetition_diagnostic=dict(
            definition="Longest exact nonoverlapping repeated output-token span per request, no threshold or quality gate; exact duplicate full decoded outputs across requests",
            longest_span_tokens=distribution([r["longest_exact_nonoverlap_repeat"]["tokens"]
                                              for r in completed]),
            exact_duplicate_output_groups=duplicates),
        service_mix=service, peak_running_after_schedule=status.get("peak_running_after_schedule"),
        peak_waiting_after_schedule=status.get("peak_waiting_after_schedule"),
        peak_used_blocks_after_schedule=status.get("peak_used_blocks_after_schedule"),
        allocation_failure_count=status.get("allocation_failure_count"),
        observation_end_s=status.get("observation_end_s"),
        per_request=per_request, frozen_input_sha256=input_hashes,
        pinned_source_sha256=source_hashes, original_run_sha256=run_hashes,
        analyzer_sha256=sha(Path(__file__)),
        limitations=[
            "All 200 official source-order QMSum rows; local OLMoE chat and 3500-token head/tail truncation adaptation, not an official LongBench model score.",
            "Rouge-L scores full decoded outputs; capped or missing rows remain in denominator.",
            "Prefill/decode totals use native classified scheduler actions, including any recomputation; do not infer one-pass prompt work after preemption.",
            "Host token returns and completion are observer timestamps, not GPU kernel times.",
        ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--metadata-dir", type=Path, required=True)
    parser.add_argument("--official-code-dir", type=Path, default=CODE)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.input_dir, args.run_dir, args.metadata_dir, args.official_code_dir)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({k: report[k] for k in
                      ("status", "requests_completed", "natural_eos_count",
                       "length_cap_count", "rouge_l_f1_percent")}))
    if report["status"] != "COMPLETE":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
