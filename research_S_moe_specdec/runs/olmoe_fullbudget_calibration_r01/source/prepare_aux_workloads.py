#!/usr/bin/env python3
"""Prepare full LongBench-derived code/summary serving prompts on CPU.

Reads an existing pinned data.zip; never downloads data or loads model weights.
This is a study-specific workload, not the official LongBench evaluation.

  python code/prepare_aux_workloads.py --task lcc --tokenizer /local/target \
      --source ../research_C_format_sched/datasets/serving_sources/longbench/data.zip \
      --output datasets/lcc_target_160.json

The default requires 160 complete rendered prompts of at most 2048 tokens.
Insufficient eligible records causes an explicit error and selection report;
no output dataset, duplicated records, truncated context, or silent fallback.
"""
from __future__ import annotations

import argparse
from collections import Counter
import datetime as dt
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import zipfile

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_new(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def check_source(path, source):
    if path.stat().st_size != source["bytes"]:
        raise ValueError("LongBench source byte count differs from the pinned data.zip (or download is incomplete)")
    digest = sha256(path)
    if digest != source["sha256"]:
        raise ValueError("LongBench source SHA256 differs from the pinned data.zip")
    return digest


def validate_row(row, task, index):
    if not isinstance(row, dict):
        raise ValueError(f"{task} row {index}: expected object")
    if row.get("dataset") != task:
        raise ValueError(f"{task} row {index}: dataset field mismatch")
    if not isinstance(row.get("context"), str) or not row["context"].strip():
        raise ValueError(f"{task} row {index}: missing context")
    if row.get("input") != "":
        raise ValueError(f"{task} row {index}: nonempty input requires a separately specified template")
    if not isinstance(row.get("answers"), list) or not row["answers"] or not all(isinstance(x, str) for x in row["answers"]):
        raise ValueError(f"{task} row {index}: invalid reference answers")
    if not isinstance(row.get("_id"), str):
        raise ValueError(f"{task} row {index}: missing source id")


def select(archive_path, task, task_spec, tokenizer, count, min_tokens, max_tokens):
    """Full population scan with deterministic first-eligible selection."""
    records, population = [], []
    counters = Counter()
    seen_ids = set()
    with zipfile.ZipFile(archive_path) as archive:
        members = [n for n in archive.namelist() if Path(n).name == task + ".jsonl"]
        if members != [task_spec["member"]]:
            raise ValueError(f"expected one exact {task_spec['member']} member, got {members}")
        with archive.open(task_spec["member"]) as stream:
            for index, line in enumerate(stream):
                if not line.strip():
                    counters["blank_lines"] += 1
                    continue
                row = json.loads(line)
                validate_row(row, task, index)
                if row["_id"] in seen_ids:
                    raise ValueError(f"{task} row {index}: duplicate source id")
                seen_ids.add(row["_id"])
                counters["source_records"] += 1
                prompt = task_spec["prompt_template"].format(context=row["context"])
                rendered = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
                ids = tokenizer.encode(rendered, add_special_tokens=False, truncation=False)
                if not isinstance(ids, list) or not all(isinstance(x, int) for x in ids):
                    raise ValueError("tokenizer returned invalid token IDs")
                reason = "eligible" if min_tokens <= len(ids) <= max_tokens else (
                    "below_min_prompt" if len(ids) < min_tokens else "above_max_prompt")
                counters[reason] += 1
                chosen = reason == "eligible" and len(records) < count
                population.append({"source_index": index, "source_id": row["_id"],
                                   "prompt_tokens": len(ids), "eligibility": reason, "selected": chosen})
                if chosen:
                    records.append({"source_index": index, "source_id": row["_id"],
                                    "task": task, "source_member": task_spec["member"],
                                    "prompt_token_ids": ids, "prompt": prompt,
                                    "rendered_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
                                    "context_sha256": hashlib.sha256(row["context"].encode()).hexdigest(),
                                    "reference": row["answers"][0], "references": row["answers"],
                                    "language": row.get("language"),
                                    "source_length_statistic": row.get("length")})
    if counters["source_records"] != task_spec["expected_source_rows"]:
        raise ValueError(f"unexpected {task} source row count: {counters['source_records']}")
    return records, {"counts": dict(counters), "requested_count": count,
                     "selected_count": len(records), "population": population}


def tokenizer_fingerprints(path):
    names = ("config.json", "tokenizer.json", "tokenizer_config.json", "tokenizer.model",
             "vocab.json", "vocab.txt", "merges.txt", "added_tokens.json",
             "special_tokens_map.json", "chat_template.jinja")
    files = [path / name for name in names if (path / name).is_file()]
    if (path / "chat_templates").is_dir():
        files.extend(sorted((path / "chat_templates").rglob("*.jinja")))
    return {str(p.relative_to(path)): sha256(p) for p in files}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/aux_workloads.json")
    parser.add_argument("--source", type=Path, default=ROOT.parent / "research_C_format_sched/datasets/serving_sources/longbench/data.zip")
    parser.add_argument("--tokenizer", "--model", dest="tokenizer", type=Path, required=True)
    parser.add_argument("--task", choices=("lcc", "multi_news"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int)
    parser.add_argument("--min-prompt", type=int)
    parser.add_argument("--max-prompt", type=int)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    count = args.count if args.count is not None else config["defaults"]["count"]
    lower = args.min_prompt if args.min_prompt is not None else config["defaults"]["min_prompt_tokens"]
    upper = args.max_prompt if args.max_prompt is not None else config["defaults"]["max_prompt_tokens"]
    if count < 1 or not 1 <= lower <= upper <= 2048:
        raise ValueError("require count >= 1 and 1 <= min-prompt <= max-prompt <= 2048")
    output, source_path, tokenizer_path = args.output.resolve(), args.source.resolve(), args.tokenizer.resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError("outputs must be inside this research project; shared inputs are read-only")
    if output.exists():
        raise FileExistsError(output)
    if not tokenizer_path.is_dir():
        raise ValueError("tokenizer must be an existing local directory")
    digest = check_source(source_path, config["source"])
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True, trust_remote_code=False)
    if not tokenizer.chat_template:
        raise ValueError("specified local tokenizer has no chat template; no fallback is assumed")
    records, report = select(source_path, args.task, config["tasks"][args.task], tokenizer, count, lower, upper)
    metadata = {"dataset": f"LongBench-derived/{args.task}", "source": str(source_path),
                "source_repo": config["source"]["repo_id"], "source_revision": config["source"]["revision"],
                "source_sha256": digest, "source_member": config["tasks"][args.task]["member"],
                "model": str(tokenizer_path), "tokenizer": str(tokenizer_path),
                "tokenizer_sha256": tokenizer_fingerprints(tokenizer_path),
                "tokenizer_class": type(tokenizer).__name__,
                "transformers_version": importlib.metadata.version("transformers"),
                "selection": config["selection"], "prompt_range": [lower, upper],
                "prompt_template": config["tasks"][args.task]["prompt_template"],
                "chat_template": tokenizer.chat_template, "rendering": config["rendering"],
                "scope": config["scope"], "reference_semantics": config["tasks"][args.task]["reference_semantics"],
                "config_sha256": sha256(args.config), "preparer_sha256": sha256(__file__),
                "cpu_only": True, "generation_performed": False, "selection_report": report}
    if len(records) < count:
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        report_path = output.with_name(output.name + f".insufficient.{stamp}.json")
        write_new(report_path, {**metadata, "status": "insufficient_eligible_records"})
        raise ValueError(f"{args.task}: only {len(records)} eligible complete prompts; requested {count}; "
                         f"no dataset written, selection report: {report_path}")
    write_new(output, {**metadata, "records": records})
    print(json.dumps({"output": str(output), "requests": len(records),
                      "eligible_population": report["counts"].get("eligible", 0), "source_sha256": digest}))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        sys.exit(1)
