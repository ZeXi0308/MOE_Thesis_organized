#!/usr/bin/env python3
"""Descriptive LongBench document-group completion from completed native runs.

Grouping uses the frozen input's raw-context SHA. All questions arrived at t=0;
the group's completion is its latest member's host_elapsed_s. This is a service
granularity diagnostic, not a claim that actual users require joint completion.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def distribution(values):
    if not values:
        return None
    return dict(count=len(values), mean_s=mean(values), median_s=median(values),
                p95_s=percentile(values, 0.95), max_s=max(values))


def parse_run(text):
    if "=" not in text:
        raise ValueError("--run must be NAME=DIR")
    name, directory = text.split("=", 1)
    if not name or not directory or not name.replace("_", "").isalnum():
        raise ValueError("--run must have a simple nonempty NAME and DIR")
    root = Path(directory)
    path = root / "measured-outputs.json"
    if not path.exists():
        path = root / "native" / "measured-outputs.json"
    if not path.is_file():
        raise FileNotFoundError(f"measured-outputs.json not found under {root}")
    return name, path


def load_inputs(path):
    data = json.loads(path.read_text())
    rows = data["requests"]
    ids = [r["request_id"] for r in rows]
    indices = [r["source_index"] for r in rows]
    if len(ids) != len(set(ids)) or sorted(indices) != list(range(len(rows))):
        raise ValueError("input IDs or source indices are invalid")
    if len(data["arrival_traces_s"]) != len(rows) or any(
        x != 0 for x in data["arrival_traces_s"]
    ):
        raise ValueError("this diagnostic requires every input arrival at t=0")
    by_id = {r["request_id"]: r for r in rows}
    groups = defaultdict(list)
    for row in rows:
        key = row["context_sha256"]
        if not isinstance(key, str) or len(key) != 64:
            raise ValueError("missing context SHA-256")
        groups[key].append(row)
    return rows, by_id, groups


def load_run(path, inputs):
    outputs = json.loads(path.read_text())
    if not isinstance(outputs, list) or len(outputs) != len(inputs):
        raise ValueError(f"wrong completed request count in {path}")
    by_id = {}
    for row in outputs:
        rid = row["request_id"]
        if rid in by_id or rid not in inputs:
            raise ValueError(f"duplicate or unknown request {rid} in {path}")
        source = inputs[rid]
        if (row["source_index"] != source["source_index"]
                or row["context_sha256"] != source["context_sha256"]
                or row["prompt_token_ids_sha256"] != source["prompt_token_ids_sha256"]):
            raise ValueError(f"input identity differs for {rid} in {path}")
        completion = row["host_elapsed_s"]
        if (row["arrival_s"] != 0 or row["finished"] is not True
                or not isinstance(completion, (int, float))
                or not math.isfinite(completion) or completion < 0):
            raise ValueError(f"incomplete or invalid timing for {rid} in {path}")
        by_id[rid] = float(completion)
    if set(by_id) != set(inputs):
        raise ValueError(f"missing requests in {path}")
    return by_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-workload", type=Path, required=True)
    parser.add_argument("--run", action="append", required=True,
                        help="NAME=run root or native directory; repeat for each arm")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.run) < 2:
        parser.error("provide at least two named runs")
    input_rows, input_by_id, groups = load_inputs(args.input_workload)
    run_files = dict(parse_run(arg) for arg in args.run)
    if len(run_files) != len(args.run):
        parser.error("run names must be unique")
    times = {name: load_run(path, input_by_id) for name, path in run_files.items()}
    names = list(run_files)
    pairs = [(before, after) for i, before in enumerate(names)
             for after in names[i + 1:]]

    group_rows = []
    for context, members in sorted(groups.items(),
                                   key=lambda item: min(r["source_index"] for r in item[1])):
        ids = [r["request_id"] for r in members]
        completions = {name: max(run[rid] for rid in ids)
                       for name, run in times.items()}
        group_rows.append(dict(context_sha256=context, request_count=len(ids),
                               member_indices=sorted(r["source_index"] for r in members),
                               member_request_ids=ids, completion_host_s=completions,
                               pair_delta_s={f"{after}_minus_{before}":
                                             completions[after] - completions[before]
                                             for before, after in pairs}))

    scopes = {"all_documents": group_rows,
              "multiquestion_documents": [g for g in group_rows if g["request_count"] > 1]}
    summaries = {}
    for scope, selected in scopes.items():
        summaries[scope] = dict(
            group_count=len(selected), question_count=sum(g["request_count"] for g in selected),
            runs={name: distribution([g["completion_host_s"][name] for g in selected])
                  for name in names},
            comparisons={f"{after}_minus_{before}": dict(
                mean_delta_s=mean(deltas), median_delta_s=median(deltas),
                improved_groups=sum(x < 0 for x in deltas),
                worsened_groups=sum(x > 0 for x in deltas),
                tied_groups=sum(x == 0 for x in deltas),
                relative_mean_change_pct=100 * mean(deltas) /
                mean(g["completion_host_s"][before] for g in selected))
                for before, after in pairs
                for deltas in [[g["completion_host_s"][after]
                                - g["completion_host_s"][before] for g in selected]]})

    result = dict(schema="c-long-document-qa-document-completion-v1",
                  scope="Descriptive host completion; equal weight per context SHA group; "
                        "no assertion that users require all questions in a group",
                  definition="max(host_elapsed_s) among a group's questions; all arrivals t=0",
                  input_workload=str(args.input_workload),
                  input_workload_sha256=sha256(args.input_workload),
                  run_files={name: dict(path=str(path), sha256=sha256(path))
                             for name, path in run_files.items()},
                  question_level_mean_host_completion_s={name: mean(run.values())
                                                         for name, run in times.items()},
                  group_size_histogram=dict(sorted(Counter(len(m) for m in groups.values()).items())),
                  summaries=summaries, per_document=group_rows)
    with args.output.open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "summaries": summaries}, indent=2))


if __name__ == "__main__":
    main()
