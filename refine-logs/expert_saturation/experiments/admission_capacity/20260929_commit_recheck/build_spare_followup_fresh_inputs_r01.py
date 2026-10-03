#!/usr/bin/env python3
"""Prepare source-ordered, complete train articles for the fixed fresh-input pair.

The missing historical 224-document list is handled by quarantining its entire
confirmed train-row prefix. No GPU outputs or latency data enter selection.
"""

import argparse
from bisect import bisect_right
from collections import Counter
import json
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
OLD_BUILDER_DIR = HERE.parents[2] / "outputs/admission_capacity/20260915_natural_cadence_holdout_r02"
sys.path.insert(0, str(OLD_BUILDER_DIR))
from input_builder_base import (  # noqa: E402
    DATASET_REVISION, REVISION, closed_articles, describe, digest, file_sha,
    token_sha,
)

PARQUET_SHA256 = "74da360f23826045b3e6ac6375411fdb15f003030aa74f2596ed08b857cb9212"
INVENTORY_SHA256 = "f92a73a93c906e0fbe05c281991a65b29905d20063901cafd0d40f8c54af1555"
MODEL_CONFIG_SHA256 = "3643aa880d2f1c9b418156269ae791c73e5612d6b6b6fde0724d927cf89b6335"


class ParquetRows:
    """Source-order row view with the indexing contract of closed_articles."""

    def __init__(self, path):
        import pyarrow.parquet as pq

        self.parquet = pq.ParquetFile(path)
        if self.parquet.schema.names != ["text"]:
            raise ValueError("Expected the official one-column train text shard")
        sizes = [self.parquet.metadata.row_group(i).num_rows
                 for i in range(self.parquet.num_row_groups)]
        self.offsets = [0]
        for size in sizes:
            self.offsets.append(self.offsets[-1] + size)
        self.cache = {}

    def __len__(self):
        return self.offsets[-1]

    def __getitem__(self, index):
        if index < 0 or index >= len(self):
            raise IndexError(index)
        group = bisect_right(self.offsets, index) - 1
        if group not in self.cache:
            if len(self.cache) >= 2:
                self.cache.pop(next(iter(self.cache)))
            self.cache[group] = self.parquet.read_row_group(
                group, columns=["text"]).column("text").to_pylist()
        return {"text": self.cache[group][index - self.offsets[group]]}

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-parquet", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--inventory", type=Path,
                        default=HERE / "spare_followup_fresh_exclusion_inventory_r01.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit("Preserve prepared inputs: output directory already exists")
    if args.dataset_parquet.name != "train-00000-of-00002.parquet":
        raise SystemExit("Unexpected Parquet shard filename")
    if file_sha(args.dataset_parquet) != PARQUET_SHA256:
        raise SystemExit("Pinned official Parquet SHA-256 mismatch")
    if file_sha(args.inventory) != INVENTORY_SHA256:
        raise SystemExit("Local exclusion inventory SHA-256 mismatch")

    inventory = json.loads(args.inventory.read_text())
    prefix_end = inventory["conservative_prior_train_prefix"]["exclude_article_if_start_row_less_than"]
    if prefix_end != 21131 or inventory["old_224_summary"]["count"] != 224:
        raise SystemExit("Prior train prefix or historical summary changed")
    known = inventory["known_records"]
    by_start = {r["row_start"]: r for r in known}
    if len(known) != 256 or len(by_start) != 256:
        raise SystemExit("Materialized source inventory is incomplete or duplicated")
    known_ids = {r["document_id"] for r in known}
    known_text_hashes = {r["document_sha256"] for r in known}
    known_token_hashes = {r["prompt_token_ids_sha256"] for r in known}

    base_path = HERE / "candidate_spare_followup_r01/pkg/inputs/config.json"
    config = json.loads(base_path.read_text())
    model, source = config["model"], config["source"]
    if model["revision"] != REVISION or model["tokenizer_revision"] != REVISION:
        raise SystemExit("Model/tokenizer revision mismatch")
    if source["dataset_revision"] != DATASET_REVISION:
        raise SystemExit("Dataset revision mismatch")
    required = dict(source["tokenizer_files_sha256"], **{"config.json": MODEL_CONFIG_SHA256})
    assets = {}
    for name, expected in required.items():
        path = args.tokenizer_dir / name
        actual = file_sha(path)
        if actual != expected:
            raise SystemExit(f"Pinned tokenizer/model asset mismatch: {name}")
        assets[name] = {"path": str(path), "sha256": actual}
    tokenizer_config = json.loads((args.tokenizer_dir / "tokenizer_config.json").read_text())
    if (tokenizer_config.get("tokenizer_class") != "GPTNeoXTokenizer" or
            tokenizer_config.get("add_bos_token") is not False or
            tokenizer_config.get("add_eos_token") is not False):
        raise SystemExit("Unexpected tokenizer special-token settings")
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(args.tokenizer_dir / "tokenizer.json"))

    dataset = ParquetRows(args.dataset_parquet)
    selected, token_lists, scan = [], [], []
    chosen_text_hashes, chosen_token_hashes, checked_full = set(), set(), set()
    counts = Counter()
    for start, end, text in closed_articles(dataset):
        rid = f"memory-train-article-{start:07d}"
        text_sha = digest(text.encode())
        prior = by_start.get(start)
        if prior is not None and prior["has_full_token_ids"]:
            if (rid != prior["document_id"] or end != prior["row_end_exclusive"] or
                    text_sha != prior["document_sha256"] or
                    text_sha != prior["prompt_sha256"]):
                raise SystemExit(f"Known source row/text differs at {start}")
            ids = tokenizer.encode(text, add_special_tokens=True).ids
            if (len(ids) != prior["prompt_token_count"] or
                    token_sha(ids) != prior["prompt_token_ids_sha256"]):
                raise SystemExit(f"Known full token sequence differs at {start}")
            checked_full.add(start)

        if start < prefix_end:
            reason = "conservative_old_train_prefix"
        elif rid in known_ids or text_sha in known_text_hashes:
            reason = "known_identity_or_text"
        else:
            ids = tokenizer.encode(text, add_special_tokens=True).ids
            ids_sha = token_sha(ids)
            reason = ("outside_fixed_256_3072" if not 256 <= len(ids) <= 3072 else
                      "known_token_content" if ids_sha in known_token_hashes else
                      "duplicate_new_content" if text_sha in chosen_text_hashes or
                      ids_sha in chosen_token_hashes else "selected")
        counts[reason] += 1
        scan.append({"document_id": rid, "row_start": start,
                     "row_end_exclusive": end, "reason": reason})
        if reason != "selected":
            continue
        chosen_text_hashes.add(text_sha)
        chosen_token_hashes.add(ids_sha)
        selected.append({
            "request_id": rid, "document_id": rid, "dataset_row_index": start,
            "dataset_row_end_exclusive": end, "article_title": text.splitlines()[0].strip(),
            "source_index": len(selected), "prompt": text,
            "document_sha256": text_sha, "prompt_sha256": text_sha,
            "prompt_token_count": len(ids), "original_document_token_count": len(ids),
            "prompt_token_ids_sha256": ids_sha, "max_output_tokens": 1024,
        })
        token_lists.append(ids)
        if len(selected) == 128:
            break
    expected_full = {r["row_start"] for r in known if r["has_full_token_ids"]}
    if len(selected) != 128 or checked_full != expected_full:
        raise SystemExit("Could not select 128 articles after validating known full source rows")
    if selected[0]["dataset_row_index"] < prefix_end:
        raise SystemExit("Selected article enters prior train prefix")
    lengths = [len(ids) for ids in token_lists]
    workload = {
        "schema": "olmoe-natural-cadence-holdout-v1", "source_requests": selected,
        "actual_prompt_token_ids": token_lists,
        "arrival_traces_s": {"steady": [i * .2 for i in range(128)]},
        "arrival_rule": "Source order,0.2s spacing,128requests,25.4s finite arrival span",
        "output_contract": {"max_output_tokens": 1024, "ignore_eos": False,
                            "min_tokens": 0, "actual_output_lengths": "UNKNOWN_UNTIL_EXECUTION"},
    }
    for key in ("candidate_source", "inherited_freeze_contract", "freeze_contract"):
        config.pop(key, None)
    config.update({
        "status": "FRESH_TRAIN_INPUT_CPU_PREPARED_GPU_UNRUN", "requests": 128,
        "prompt_tokens": max(lengths), "prompt_tokens_by_request": lengths,
        "arrival_span_s": 127 * .2,
        "workload_sha256": digest(json.dumps(workload, sort_keys=True).encode()),
        "prompt_tokens_semantics": f"Upper bound {max(lengths)}; full lengths {min(lengths)}..{max(lengths)}, no padding/truncation.",
        "input_preparation": "First128 source-ordered complete train articles after conservative old train-prefix and ROOT-local known-content exclusions; fixed 256..3072 full-token gate, 0.2s arrivals and 1024 natural-EOS cap.",
        "input_independence": "Fresh documents relative to the known local train inputs; not a global unseen claim or an independent model/runtime/host measurement.",
        "exclusion_inventory_sha256": INVENTORY_SHA256,
    })
    source = dict(source)
    source.pop("arrow_file", None)
    source.pop("arrow_sha256", None)
    source.update({
        "source_format": "official_parquet_not_historical_arrow",
        "parquet_file": args.dataset_parquet.name, "parquet_sha256": PARQUET_SHA256,
        "selection": "First128 source-ordered complete articles after quarantining prior train prefix ending row21131 and excluding known local identities/text/token content; full token length256..3072. No outcomes used.",
    })
    config["source"] = source
    report = {
        "status": config["status"], "builder_base_sha256": file_sha(OLD_BUILDER_DIR / "input_builder_base.py"),
        "source_base_config_sha256": file_sha(base_path),
        "exclusion_inventory_sha256": INVENTORY_SHA256,
        "old_train_articles_summarized": 224,
        "old_train_prefix_quarantine_end_exclusive": prefix_end,
        "known_local_full_articles_validated_row_text_and_tokens": len(checked_full),
        "known_local_partial_warmup_records_excluded_by_prefix": len(known) - len(checked_full),
        "local_assets": assets, "source_parquet_sha256": PARQUET_SHA256,
        "source_format_note": "Official Parquet closed-article row/text/token identity checked against 192 complete G/H articles; 64 historical warmup snippets remain inside quarantined prefix. Historical Arrow binary hash is not claimed for this Parquet.",
        "scanned_interval": {"first_row": scan[0]["row_start"],
                             "end_exclusive": scan[-1]["row_end_exclusive"]},
        "examined_article_counts": dict(counts), "examined_articles": scan,
        "selected_full_prompt_lengths": describe(lengths),
        "selected_row_first": selected[0]["dataset_row_index"],
        "selected_row_end_exclusive": selected[-1]["dataset_row_end_exclusive"],
        "arrival_gap_s": .2, "arrival_span_s": 127 * .2,
        "measured_requests_per_cell": 128, "planned_full_group_cells": 3,
        "measured_request_budget": 384, "gpu_executions": 0,
        "network_downloads_by_builder": 0,
    }
    args.output_dir.mkdir(parents=True)
    for name, obj in (("workload.json", workload), ("config.json", config),
                      ("INPUT_STATS.json", report)):
        (args.output_dir / name).write_text(
            json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"],
                      "selected_row_first": report["selected_row_first"],
                      "selected_row_end_exclusive": report["selected_row_end_exclusive"],
                      "examined_article_counts": report["examined_article_counts"],
                      "validated_known_complete": len(checked_full),
                      "lengths": report["selected_full_prompt_lengths"]},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
