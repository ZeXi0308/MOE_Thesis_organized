#!/usr/bin/env python3
"""Compare two complete QMSum200 analyses without dropping changed outputs."""
import argparse
import json
from pathlib import Path

from C_QMSUM_ANALYZE_V1 import distribution, sha


def compare(control_path, treatment_path):
    a, b = (json.loads(p.read_text()) for p in (control_path, treatment_path))
    assert a["status"] == b["status"] == "COMPLETE"
    assert a["requests_completed"] == b["requests_completed"] == 200
    assert a["frozen_input_sha256"] == b["frozen_input_sha256"]
    assert a["model"] == b["model"]
    left = {r["request_id"]: r for r in a["per_request"]}
    right = {r["request_id"]: r for r in b["per_request"]}
    assert left.keys() == right.keys()
    per = []
    for rid, x in left.items():
        y = right[rid]
        per.append(dict(request_id=rid, source_index=x["source_index"],
            flow_delta_s=y["flow_s"]-x["flow_s"],
            ttft_delta_s=y["ttft_s"]-x["ttft_s"],
            gap_delta_s=y["max_distinct_host_return_gap_s"]-x["max_distinct_host_return_gap_s"],
            output_ids_changed=x["output_token_ids"] != y["output_token_ids"],
            output_length_delta=y["output_tokens"]-x["output_tokens"],
            finish_reason_changed=x["finish_reason"] != y["finish_reason"],
            rouge_l_delta=y["rouge_l_f1"]-x["rouge_l_f1"]))
    fields = ("flow_s", "ttft_s", "max_distinct_host_return_gap_s")
    metrics = {key: dict(control=a[key], treatment=b[key],
        relative_mean_change=b[key]["mean"]/a[key]["mean"]-1) for key in fields}
    docs_a = {d["context_sha256"]: d for d in a["document_groups"]["per_document"]}
    docs_b = {d["context_sha256"]: d for d in b["document_groups"]["per_document"]}
    assert docs_a.keys() == docs_b.keys()
    docs = [dict(context_sha256=k, request_count=v["request_count"],
                 completion_delta_s=docs_b[k]["completion_host_s"]-v["completion_host_s"])
            for k, v in docs_a.items()]
    slower = [r for r in per if r["flow_delta_s"] > 0]
    return dict(schema="c-qmsum-pair-comparison-v1", status="COMPLETE",
        analysis_sha256=dict(control=sha(control_path), treatment=sha(treatment_path)),
        metrics=metrics,
        episode_s=dict(control=a["observation_end_s"], treatment=b["observation_end_s"]),
        output_tokens=dict(control=a["output_tokens_total"], treatment=b["output_tokens_total"]),
        rouge_l_percent=dict(control=a["rouge_l_f1_percent"], treatment=b["rouge_l_f1_percent"]),
        changed_output_ids=sum(r["output_ids_changed"] for r in per),
        changed_output_lengths=sum(r["output_length_delta"] != 0 for r in per),
        changed_finish_reasons=sum(r["finish_reason_changed"] for r in per),
        earlier_completions=sum(r["flow_delta_s"] < 0 for r in per),
        later_completions=len(slower),
        later_added_wait_s=distribution([r["flow_delta_s"] for r in slower]),
        document_completion_delta_s=distribution([d["completion_delta_s"] for d in docs]),
        per_request=per, per_document=docs,
        service={k: {key: value for key, value in doc["service_mix"].items()
                     if key != "per_step"} for k, doc in (("control", a), ("treatment", b))},
        limitation="One viewed development pair; no confidence claim, equal-output speedup assumption, or unseen confirmation. Host return gaps are not GPU-kernel timings.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("control", "treatment", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.control, args.treatment)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps({k: result[k] for k in ("status", "earlier_completions",
        "later_completions", "changed_output_ids", "episode_s")}))
