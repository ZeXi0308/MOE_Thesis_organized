#!/usr/bin/env python3
"""Prepare disjoint, grouped WikiText continuation prompts on CPU only.

Uses existing article JSONL and a local tokenizer. No model weights are loaded,
and no downloads or GPU execution are permitted. Window records from the same
article remain grouped; 200 prompts do not imply 200 independent articles.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path


def sha(data):
    return hashlib.sha256(data).hexdigest()


def token_digest(ids):
    # Same serialization as free_generate.digest for the eventual actual prefix.
    return sha(json.dumps(ids, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":")).encode())


def load_articles(path):
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        text = row["text"]
        actual = sha(text.encode())
        if row.get("text_sha256", actual) != actual:
            raise ValueError(f"Article hash mismatch: {path}:{line_number}")
        article_id = str(row.get("source_article_id", row["id"]))
        rows.append({**row, "source_article_id": article_id,
                     "text_sha256": actual, "source_line": line_number})
    if len({r["source_article_id"] for r in rows}) != len(rows):
        raise ValueError(f"Input must have one row per source article: {path}")
    return rows


def file_info(path):
    return {"path": str(path.resolve()), "sha256": sha(path.read_bytes())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--articles", type=Path, required=True)
    parser.add_argument("--exclude-articles", type=Path, nargs="+", required=True)
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--window-words", type=int, default=256)
    parser.add_argument("--prompt-tokens", type=int, default=128)
    args = parser.parse_args()
    if min(args.count, args.window_words, args.prompt_tokens) <= 0:
        parser.error("Counts must be positive")
    if not args.tokenizer.is_dir():
        parser.error("--tokenizer must be an existing local model/tokenizer directory")
    if args.output_dir.exists():
        raise FileExistsError("Choose a fresh output directory; prepared inputs are immutable")

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)

    articles = load_articles(args.articles)
    excluded = [r for path in args.exclude_articles for r in load_articles(path)]
    excluded_ids = {r["source_article_id"] for r in excluded}
    excluded_hashes = {r["text_sha256"] for r in excluded}
    eligible = [r for r in articles if r["source_article_id"] not in excluded_ids
                and r["text_sha256"] not in excluded_hashes]
    words = {r["source_article_id"]: r["text"].split() for r in eligible}
    maximum_windows = max((len(w) // args.window_words for w in words.values()), default=0)
    selected, seen_prefix, seen_text = [], set(), set()
    rejection = Counter()
    for index in range(maximum_windows):
        for article in eligible:
            start = index * args.window_words
            end = start + args.window_words
            article_words = words[article["source_article_id"]]
            if end > len(article_words):
                continue
            text = " ".join(article_words[start:end])
            text_sha = sha(text.encode())
            tokens = tokenizer.encode(text, add_special_tokens=True)
            if len(tokens) < args.prompt_tokens:
                rejection["short_token_prefix"] += 1
                continue
            prefix_sha = token_digest(tokens[:args.prompt_tokens])
            if text_sha in seen_text or prefix_sha in seen_prefix:
                rejection["duplicate_text_or_token_prefix"] += 1
                continue
            seen_text.add(text_sha)
            seen_prefix.add(prefix_sha)
            selected.append({
                "id": f"{article['source_article_id']}_free_w{index:03d}",
                "text": text, "text_sha256": text_sha,
                "split": article.get("split", "unspecified"),
                "source_split": article.get("source_split", article.get("split", "unspecified")),
                "source_article_id": article["source_article_id"],
                "source_article_text_sha256": article["text_sha256"],
                "source_article_line": article["source_line"],
                "window_index": index, "window_start_word": start,
                "window_end_word": end, "window_words": args.window_words,
                "unit": "nonoverlapping_word_window_from_grouped_article",
                "source_window_token_count": len(tokens),
                "prepared_prefix_tokens": args.prompt_tokens,
                "prepared_prefix_token_sha256": prefix_sha,
                "purpose": "exploratory natural-termination and repetition description; no task-accuracy target",
            })
            if len(selected) == args.count:
                break
        if len(selected) == args.count:
            break
    if len(selected) != args.count:
        raise ValueError(f"Only {len(selected)} eligible distinct prefixes; requested {args.count}")
    counts = Counter(r["source_article_id"] for r in selected)
    assert not (set(counts) & excluded_ids)
    assert not ({r["source_article_text_sha256"] for r in selected} & excluded_hashes)
    for article_id in counts:
        intervals = sorted((r["window_start_word"], r["window_end_word"])
                           for r in selected if r["source_article_id"] == article_id)
        assert all(a[1] <= b[0] for a, b in zip(intervals, intervals[1:]))

    args.output_dir.mkdir(parents=True)
    destination = args.output_dir / "prompts.jsonl"
    with destination.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    tokenizer_files = [args.tokenizer / name for name in
                       ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")]
    manifest = {
        "status": "READY", "source_articles": file_info(args.articles),
        "exclusions": [file_info(p) for p in args.exclude_articles],
        "source_manifest": file_info(args.source_manifest) if args.source_manifest else None,
        "preparation_script": file_info(Path(__file__)),
        "output": {**file_info(destination), "records": len(selected)},
        "source_article_count_before_exclusion": len(articles),
        "eligible_source_articles": len(eligible), "selected_source_articles": len(counts),
        "selected_source_splits": sorted({r["source_split"] for r in selected}),
        "records_by_source_article": dict(counts),
        "eligible_nonoverlapping_full_windows": sum(len(w) // args.window_words for w in words.values()),
        "window_words": args.window_words, "prompt_tokens": args.prompt_tokens,
        "unique_actual_prefixes": len(seen_prefix), "rejected": dict(rejection),
        "tokenizer": {"path": str(args.tokenizer.resolve()), "local_files_only": True,
                      "add_special_tokens": True,
                      "files": [file_info(p) for p in tokenizer_files if p.is_file()]},
        "selection": "Deterministic round-robin by nonoverlapping window index, then original article order; skip duplicate actual token prefixes.",
        "scope": "Exploratory natural continuation, EOS/limit/failure/length and repetition description. No reference-answer accuracy score.",
        "dependence": "Records sharing source_article_id are not independent articles. Generation may naturally repeat despite distinct prompts.",
        "exclusion_check": "Selected article IDs and full-text SHA256 are disjoint from every supplied exclusion file.",
        "recommended_generation": {"limit": args.count, "prompt_tokens": args.prompt_tokens,
                                   "max_tokens": 256, "ignore_eos": False},
    }
    temporary = args.output_dir / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(args.output_dir / "manifest.json")
    print(json.dumps({"status": "READY", "path": str(destination), "records": len(selected),
                      "source_articles": len(counts), "unique_actual_prefixes": len(seen_prefix),
                      "rejected": dict(rejection)}))


if __name__ == "__main__":
    main()
