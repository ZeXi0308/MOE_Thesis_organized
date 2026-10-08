#!/usr/bin/env python3
"""Exact host-time decomposition and prior-1024 signature check for QMSum."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean

from C_QMSUM_ANALYZE_V1 import load, sha


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"))
                          .encode()).hexdigest()


def cell(directory: Path):
    names = ("measured-outputs.json", "measured-service-mix.json", "measured-steps.json",
             "status.json", "shape-warmup-source.json", "resolved-scheduler.json")
    docs = {name: load(directory / name) for name in names}
    outputs, mix, steps, status, shape, resolved = (docs[name] for name in names)
    by_index = {r["source_index"]: r for r in outputs}
    if (status.get("status") != "COMPLETE" or len(outputs) != 200
            or len(by_index) != 200 or set(by_index) != set(range(200))
            or len(mix["first_successful_allocation"]) != 200
            or len(mix["scheduler_calls"]) != status["schedule_calls"]):
        raise ValueError(f"incomplete 200-request cell: {directory}")
    for row in outputs:
        times = row["token_times_s"]
        if (not row["finished"] or not times or len(times) != len(row["output_token_ids"])
                or row["arrival_s"] != 0.0 or not times[0] <= row["host_elapsed_s"]):
            raise ValueError("output host timestamp contract differs")
    first = [r["external_request_id"] for r in mix["first_successful_allocation"]]
    if len(set(first)) != 200:
        raise ValueError("first-allocation inventory differs")
    signature = [
        [(r["external_request_id"], r["scheduled_tokens"],
          r["scheduled_prefill_tokens"], r["scheduled_decode_tokens"],
          r["previous_computed_tokens"], r["new_prefix_cached_tokens"])
         for r in call["requests"]]
        for call in mix["scheduler_calls"]]
    return dict(outputs=by_index, first=first, signature=signature,
                first_sha256=canonical_hash(first),
                schedule_sha256=canonical_hash(signature),
                output_ids_sha256=canonical_hash([by_index[i]["output_token_ids"]
                                                  for i in range(200)]),
                status=status, shape=shape, resolved=resolved,
                raw_sha256={name: sha(directory / name) for name in names},
                submission=steps["source_indices_in_submission_order"])


def parts(row):
    arrival = row["arrival_s"]
    first = row["token_times_s"][0]
    completion = row["host_elapsed_s"]
    return dict(completion_s=completion-arrival, ttft_s=first-arrival,
                post_first_token_span_s=completion-first,
                output_tokens=len(row["output_token_ids"]),
                finish_reason=row["finish_reason"],
                max_distinct_host_gap_s=max((b-a for a, b in zip(
                    row["token_times_s"], row["token_times_s"][1:]) if b > a), default=0.0))


def avg(rows, key):
    return mean(row[key] for row in rows)


def document_context(a, b, index):
    context = a["outputs"][index]["context_sha256"]
    members = sorted(i for i, row in a["outputs"].items()
                     if row["context_sha256"] == context)
    for i in members:
        if b["outputs"][i]["context_sha256"] != context:
            raise ValueError("document context differs across pair")
    return dict(context_sha256=context, member_source_indices=members,
        completion_512_s=max((a["outputs"][i]["host_elapsed_s"], i) for i in members),
        completion_1024_s=max((b["outputs"][i]["host_elapsed_s"], i) for i in members))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--small-dir", type=Path, required=True)
    ap.add_argument("--large-dir", type=Path, required=True)
    ap.add_argument("--backfill-control-dir", type=Path, required=True)
    ap.add_argument("--whole-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    runs = {"512": cell(args.small_dir), "1024": cell(args.large_dir),
            "earlier_backfill_control_1024": cell(args.backfill_control_dir),
            "earlier_whole_1024": cell(args.whole_dir)}
    small, large = runs["512"], runs["1024"]
    if (small["submission"] != large["submission"]
            or small["resolved"]["max_num_batched_tokens"] != 512
            or any(runs[k]["resolved"]["max_num_batched_tokens"] != 1024
                   for k in runs if k != "512")):
        raise ValueError("native budget or fixed submission differs")
    rows = []
    for i in range(200):
        a, b = small["outputs"][i], large["outputs"][i]
        if a["prompt_token_ids"] != b["prompt_token_ids"] or a["request_id"] != b["request_id"]:
            raise ValueError(f"input request differs at source index {i}")
        aa, bb = parts(a), parts(b)
        if (abs(aa["completion_s"]-aa["ttft_s"]-aa["post_first_token_span_s"]) > 1e-9
                or abs(bb["completion_s"]-bb["ttft_s"]-bb["post_first_token_span_s"]) > 1e-9):
            raise ValueError("completion identity differs")
        rows.append(dict(source_index=i, request_id=a["request_id"],
            budget_512=aa, budget_1024=bb,
            completion_delta_s=aa["completion_s"]-bb["completion_s"],
            ttft_delta_s=aa["ttft_s"]-bb["ttft_s"],
            post_first_span_delta_s=(aa["post_first_token_span_s"]-
                                     bb["post_first_token_span_s"]),
            output_ids_equal=a["output_token_ids"] == b["output_token_ids"],
            output_length_delta=aa["output_tokens"]-bb["output_tokens"],
            finish_reason_changed=aa["finish_reason"] != bb["finish_reason"]))
    equal = [r for r in rows if r["output_ids_equal"]]
    keys = ("completion_delta_s", "ttft_delta_s", "post_first_span_delta_s")
    summary = {"all_200_equal_weight_mean_delta_512_minus_1024_s":
               {k: avg(rows, k) for k in keys},
               "changed_output_ids": 200-len(equal),
               "changed_output_lengths": sum(r["output_length_delta"] != 0 for r in rows),
               "changed_finish_reasons": sum(r["finish_reason_changed"] for r in rows),
               "unchanged_output_ids_subset_diagnostic_only": dict(count=len(equal),
                   mean_delta_512_minus_1024_s={k: avg(equal, k) for k in keys}),
               "completion_identity_mean_residual_s":
                   avg(rows, "completion_delta_s")-avg(rows, "ttft_delta_s")
                   -avg(rows, "post_first_span_delta_s")}
    tails = []
    for i in (41, 49):
        row = rows[i]
        tails.append(dict(**row, document=document_context(small, large, i)))
    reference = {}
    for label in ("1024", "earlier_backfill_control_1024", "earlier_whole_1024"):
        run = runs[label]
        reference[label] = dict(first_allocation_sha256=run["first_sha256"],
            scheduled_batch_signature_sha256=run["schedule_sha256"],
            output_ids_sha256=run["output_ids_sha256"],
            schedule_calls=run["status"]["schedule_calls"],
            output_tokens=run["status"]["output_tokens"],
            observation_end_s=run["status"]["observation_end_s"],
            mean_completion_s=mean(r["host_elapsed_s"]-r["arrival_s"]
                                   for r in run["outputs"].values()),
            shape_warmup=run["shape"].get("shapes", run["shape"].get("prompt_tokens")),
            first_allocation_identical_to_current_1024=run["first"] == large["first"],
            scheduled_batch_identical_to_current_1024=run["signature"] == large["signature"],
            changed_output_ids_vs_current_1024=sum(
                run["outputs"][i]["output_token_ids"]
                != large["outputs"][i]["output_token_ids"] for i in range(200)))
    result = dict(schema="c-qmsum-chunk-host-decomposition-v1",
        scope="One 512→1024 same-host development pair, with two completed prior 1024 observations; no new GPU run or equal-work inference",
        raw_sha256={label: run["raw_sha256"] for label, run in runs.items()},
        summary=summary, tail_requests_41_49=tails,
        finish_reason_changed_indices=[r["source_index"] for r in rows
                                       if r["finish_reason_changed"]],
        per_request=rows, completed_1024_reference_runs=reference,
        interpretation_limits=[
            "Completion is arrival to final host return; TTFT ends at first host token return; post-first span ends at final return. Each interval includes scheduling/observer and actual step effects, not isolated GPU kernels.",
            "The 23 unchanged-output requests are a post-run diagnostic subset, not a primary comparison or randomized control.",
            "Different generated lengths and finish reasons alter work; the decomposition is an exact timing identity, not a causal partition of budget effects.",
            "The current chunk pair used both 511 and 1023-token shape warmups; earlier 1024 runs used one 1023-token shape warmup. Exact output/schedule signatures can be compared, but host times are separate executions."])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(summary=summary,
        identical_1024_signatures=all(v["scheduled_batch_identical_to_current_1024"]
                                      and v["changed_output_ids_vs_current_1024"] == 0
                                      for v in reference.values())), sort_keys=True))


if __name__ == "__main__":
    main()
