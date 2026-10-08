#!/usr/bin/env python3
"""Summarize pinned LongBench QMSum source and frozen input geometry, CPU only."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics

from tokenizers import Tokenizer

SOURCE_SHA = "e992f5157679b0c1ca281d0da19d1a8b3496117630ae639c9683ed3dab029113"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dist(values: list[int]) -> dict:
    values = sorted(values)
    return dict(count=len(values), min=values[0], median=statistics.median(values),
                p95_nearest_rank=values[math.ceil(.95 * len(values)) - 1], max=values[-1],
                mean=sum(values) / len(values))


def lcp(sequences: list[list[int]]) -> int:
    length = 0
    for column in zip(*sequences):
        if len(set(column)) != 1:
            break
        length += 1
    return length


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-jsonl", type=Path, required=True)
    p.add_argument("--inputs-dir", type=Path, required=True)
    p.add_argument("--metadata-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if sha(a.source_jsonl) != SOURCE_SHA:
        raise ValueError("QMSum source member SHA differs")
    source = [json.loads(line) for line in a.source_jsonl.open(encoding="utf-8")]
    workload_path = a.inputs_dir / "workload.json"
    workload = json.loads(workload_path.read_text(encoding="utf-8"))
    geometry = json.loads((a.inputs_dir / "geometry.json").read_text(encoding="utf-8"))
    config = json.loads((a.inputs_dir / "config.json").read_text(encoding="utf-8"))
    requests = workload["requests"]
    if (len(source) != 200 or len(requests) != 200
            or workload["schema"] != "c-longbench-qmsum-full-workload-v1"
            or config["workload_sha256"] != sha(workload_path)
            or config["output_tokens"] != 512):
        raise ValueError("frozen QMSum inputs differ")
    tokenizer = Tokenizer.from_file(str(a.metadata_dir / "tokenizer.json"))
    groups = defaultdict(list)
    context_tokens, question_tokens, reference_tokens, reference_words = [], [], [], []
    for i, (row, request) in enumerate(zip(source, requests)):
        context_sha = hashlib.sha256(row["context"].encode()).hexdigest()
        if (request["source_index"] != i or request["context_sha256"] != context_sha
                or request["question"] != row["input"]
                or request["answers"] != row["answers"] or len(row["answers"]) != 1):
            raise ValueError(f"source/input identity differs at row {i}")
        groups[context_sha].append(i)
        context_tokens.append(len(tokenizer.encode(row["context"], add_special_tokens=False).ids))
        question_tokens.append(len(tokenizer.encode(row["input"], add_special_tokens=False).ids))
        reference_tokens.append(len(tokenizer.encode(row["answers"][0], add_special_tokens=False).ids))
        reference_words.append(len(row["answers"][0].split()))
    if len(groups) != geometry["distinct_full_meeting_contexts"]:
        raise ValueError("meeting group count differs")
    group_rows = []
    for context_sha, indices in groups.items():
        common = lcp([requests[i]["prompt_token_ids"] for i in indices])
        group_rows.append(dict(context_sha256=context_sha, source_indices=indices,
                               request_count=len(indices),
                               context_tokens=context_tokens[indices[0]],
                               rendered_prompt_common_prefix_tokens=common))
    multi = [r for r in group_rows if r["request_count"] >= 2]
    source_runs = 1 + sum(requests[i - 1]["context_sha256"] != requests[i]["context_sha256"]
                          for i in range(1, len(requests)))
    report = dict(schema="c-qmsum-structure-v1", source_sha256=SOURCE_SHA,
                  workload_sha256=sha(workload_path),
                  tokenizer_sha256=sha(a.metadata_dir / "tokenizer.json"),
                  official_task="LongBench QMSum test, source order; 512-token generation cap",
                  source_rows=len(source), distinct_meeting_contexts=len(groups),
                  meeting_size_histogram=dict(sorted(Counter(map(len, groups.values())).items())),
                  source_order_context_runs=source_runs,
                  context_tokens_per_request=dist(context_tokens),
                  question_tokens=dist(question_tokens),
                  reference_tokens=dist(reference_tokens),
                  reference_words=dist(reference_words),
                  reference_gt_64_tokens=sum(n > 64 for n in reference_tokens),
                  reference_gt_512_tokens=sum(n > 512 for n in reference_tokens),
                  rendered_prompt_plus_output_cap_fits=sum(
                      p + 512 <= 4096 for p in geometry["prompt_tokens"]),
                  truncated_requests=geometry["truncated_count"],
                  rendered_prompt_global_lcp_tokens=geometry["global_rendered_prompt_common_prefix_tokens"],
                  first32_group_count=geometry["first32_token_prefix_group_count"],
                  first32_group_size_histogram=geometry["first32_token_prefix_group_size_histogram"],
                  first32_group_distinct_meeting_counts=geometry["first32_token_prefix_distinct_contexts"],
                  multiquestion_meetings=len(multi),
                  multiquestion_requests=sum(r["request_count"] for r in multi),
                  within_meeting_rendered_lcp_tokens=dist([
                      r["rendered_prompt_common_prefix_tokens"] for r in multi]),
                  groups=group_rows,
                  scope="Offline source/input structure, no model outputs, EOS, cache allocation, or service claim")
    with a.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({k: report[k] for k in ("source_rows", "distinct_meeting_contexts",
        "reference_tokens", "rendered_prompt_global_lcp_tokens", "first32_group_count")},
        sort_keys=True))


if __name__ == "__main__":
    main()
