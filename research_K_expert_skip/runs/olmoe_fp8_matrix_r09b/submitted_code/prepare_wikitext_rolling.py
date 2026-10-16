"""Build a complete local WikiText split into pretokenized decode-quality windows.

Rows are joined in original parquet order with exactly two newlines, including
empty rows; the resulting text is tokenized once, without special tokens.
For prefix P and score length S, score [P,N) exactly once in adjacent spans of
at most S tokens. Each window resets its history to the preceding P tokens.
The engine receives P-1 prompt tokens and emits the final context token from
prefill (excluded); all S scored predictions therefore occur in real decode.
These windows are not independent articles, and this declared rolling-context
PPL is not asserted equivalent to other WikiText evaluation protocols.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

SCHEMA = "wikitext_decode_rolling_v1"


def default_source_manifest():
    """Use checkout data only when present; installed CLI needs explicit input."""
    module_dir = Path(__file__).resolve().parent
    candidate = module_dir.parent / "datasets/wikitext2_serving_r01/manifest.json"
    if (module_dir.name == "code" and (module_dir.parent / "pyproject.toml").is_file()
            and candidate.is_file()):
        return candidate
    return None


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for data in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(data)
    return digest.hexdigest()


def tokenizer_identity(tokenizer):
    """Path-independent identity of the actual fast tokenizer, not its name.

    Runtime padding/truncation settings are excluded because windows are passed
    as token IDs. Vocabulary, normalization, pre/post-processing, decoder and
    added-token rules remain in the backend hash. No encode/decode is performed.
    """
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is None or not hasattr(backend, "to_str"):
        raise ValueError("Rolling input requires a fast tokenizer with serializable backend identity")
    payload = json.loads(backend.to_str())
    payload.pop("padding", None)
    payload.pop("truncation", None)
    semantic = {
        "backend_sha256": json_hash(payload),
        "vocabulary_sha256": json_hash(tokenizer.get_vocab()),
        "special_token_ids": {name: getattr(tokenizer, name, None) for name in
                              ("bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id")},
        "all_special_ids": sorted(set(tokenizer.all_special_ids)),
    }
    return {**semantic, "sha256": json_hash(semantic),
            "class": type(tokenizer).__name__, "name_or_path": str(getattr(tokenizer, "name_or_path", ""))}


def validate_tokenizer_identity(expected, tokenizer):
    actual = tokenizer_identity(tokenizer)
    if actual["sha256"] != expected["sha256"]:
        differences = [key for key in ("backend_sha256", "vocabulary_sha256", "special_token_ids", "all_special_ids")
                       if actual[key] != expected.get(key)]
        raise ValueError(f"Pretokenized window tokenizer identity mismatch: {differences}")
    return actual


def rolling_windows(token_ids, prefix=128, scored=128, split="validation"):
    if type(prefix) is not int or prefix < 2 or type(scored) is not int or scored < 1:
        raise ValueError("prefix must be >=2 (nonempty engine prompt), scored must be >=1")
    if any(type(token) is not int or token < 0 for token in token_ids):
        raise ValueError("Token stream must contain nonnegative integer IDs")
    if len(token_ids) <= prefix:
        raise ValueError("Split has no scoreable token after the excluded initial context")
    stream_sha = json_hash(token_ids)
    for index, start in enumerate(range(prefix, len(token_ids), scored)):
        stop = min(start + scored, len(token_ids))
        yield {"id": f"{split}:{start}:{stop}", "window_index": index, "split": split,
               "doc_key": f"{stream_sha}:{start}:{stop}", "prefix_tokens": prefix,
               "scored_tokens": stop - start, "context_start_token": start - prefix,
               "prefill_bridge_token_offset": start - 1,
               "score_start_token": start, "score_stop_token": stop,
               "token_ids": token_ids[start - prefix:stop]}


def load_rolling_documents(manifest_path, limit=None):
    """Validate complete coverage before optionally selecting a prefix of windows.

    Runtime tokenizer identity is checked separately against llm.get_tokenizer().
    Keeping every incomplete batch and the short tail preserves corpus coverage.
    """
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != SCHEMA or manifest.get("status") != "READY":
        raise ValueError("Expected a READY rolling-window manifest")
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("limit-windows must be positive when supplied")
    relative = Path(manifest["windows_file"]["path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("windows_file must stay inside the rolling dataset directory")
    path = manifest_path.parent / relative
    if file_hash(path) != manifest["windows_file"]["sha256"]:
        raise ValueError("Rolling windows file SHA-256 mismatch")
    prefix, scored = manifest["context_prefix_tokens"], manifest["max_scored_tokens"]
    total, expected_start = manifest["token_stream"]["tokens"], prefix
    if type(prefix) is not int or prefix < 2 or type(scored) is not int or scored < 1 or total <= prefix:
        raise ValueError("Invalid rolling context/scoring dimensions")
    if manifest["engine_prompt_tokens"] != prefix - 1:
        raise ValueError("Rolling engine prompt must leave exactly one prefill bridge token")
    documents, stream = [], []
    for index, line in enumerate(path.read_text().splitlines()):
        row = json.loads(line)
        start, stop = row["score_start_token"], row["score_stop_token"]
        ids = row["token_ids"]
        if (row["window_index"] != index or start != expected_start or start >= total or stop <= start
                or stop != min(start + scored, total)
                or row["context_start_token"] != start - prefix or row["prefill_bridge_token_offset"] != start - 1
                or row["prefix_tokens"] != prefix or row["scored_tokens"] != stop - start
                or row["split"] != manifest["split"] or len(ids) != prefix + stop - start
                or any(type(token) is not int or token < 0 for token in ids)):
            raise ValueError(f"Invalid rolling coverage/offsets at window {index}")
        if index == 0:
            stream.extend(ids[:prefix])
        if ids[:prefix] != stream[start-prefix:start]:
            raise ValueError(f"Rolling context disagrees with preceding source tokens at window {index}")
        stream.extend(ids[prefix:])
        expected_key = f'{manifest["token_stream"]["sha256"]}:{start}:{stop}'
        if row["doc_key"] != expected_key:
            raise ValueError(f"Rolling key does not match source offsets at window {index}")
        documents.append({**{key: value for key, value in row.items() if key != "token_ids"},
                          "source_line": index + 1, "source_token_count": total,
                          "window_start_token": start - prefix,
                          "prompt_token_ids": ids[:prefix-1], "forced_tokens": ids[prefix-1:],
                          "statistical_unit": "rolling_window_not_independent_article"})
        expected_start = stop
    if (expected_start != total or len(documents) != manifest["windows_file"]["rows"]
            or json_hash(stream) != manifest["token_stream"]["sha256"]
            or manifest["coverage"]["scored_tokens"] != total - prefix):
        raise ValueError("Rolling token stream/count/coverage mismatch")
    all_count = len(documents)
    documents = documents if limit is None else documents[:limit]
    selected_stop = documents[-1]["score_stop_token"]
    selection = {"input_kind": SCHEMA, "source_manifest": str(manifest_path.resolve()),
                 "source_manifest_sha256": file_hash(manifest_path), "available_windows": all_count,
                 "selected_windows": len(documents), "selected_scored_span": [prefix, selected_stop],
                 "selected_scored_tokens": selected_stop - prefix,
                 "unselected_suffix_tokens": total - selected_stop,
                 "complete_split_after_initial_context": selected_stop == total,
                 "dropped_incomplete_batch_windows": 0, "statistical_unit": "rolling window, not independent article"}
    return documents, selection, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Local model/tokenizer directory; no weights are loaded")
    parser.add_argument("--source-manifest", type=Path,
                        default=default_source_manifest(),
                        help="Original split provenance manifest; required outside a checkout with bundled local dataset provenance")
    parser.add_argument("--parquet", type=Path, help="Local original parquet; must match the split SHA in source-manifest")
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--prefix-tokens", type=int, default=128,
                        help="Context at first scored token, including one excluded prefill bridge; engine prompt is P-1")
    parser.add_argument("--scored-tokens", type=int, default=128)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()
    if args.source_manifest is None:
        parser.error("--source-manifest is required for an installed CLI; dataset inputs are not bundled in the wheel")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    source_manifest = json.loads(args.source_manifest.read_text())
    source = source_manifest["sources"][args.split]
    parquet = args.parquet or Path(source["path"])
    if not parquet.is_file():
        raise FileNotFoundError(f"Pass --parquet pointing to the original local {args.split} parquet: {parquet}")
    if file_hash(parquet) != source["sha256"]:
        raise ValueError("Original parquet does not match source manifest SHA-256")
    if not args.model.is_dir():
        raise ValueError("--model must be a local tokenizer directory")
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    table = pq.read_table(parquet, columns=["text"])
    rows = table.column("text").to_pylist()
    if any(not isinstance(text, str) for text in rows):
        raise ValueError("Every original parquet text row must be a string; null rows are not silently skipped")
    if len(rows) != source["parquet_rows"]:
        raise ValueError("Original parquet row count disagrees with source manifest")
    text = "\n\n".join(rows)
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True,
                                               trust_remote_code=args.trust_remote_code, use_fast=True)
    token_ids = list(tokenizer.encode(text, add_special_tokens=False, truncation=False))
    identity = tokenizer_identity(tokenizer)
    windows = list(rolling_windows(token_ids, args.prefix_tokens, args.scored_tokens, args.split))
    args.output_dir.mkdir(parents=True)
    output = args.output_dir / "windows.jsonl"
    with output.open("w", encoding="utf-8") as handle:
        for row in windows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    manifest = {"status": "READY", "schema": SCHEMA, "created_unix": time.time(), "split": args.split,
                "source_manifest": {"path": str(args.source_manifest.resolve()), "sha256": file_hash(args.source_manifest)},
                "source_parquet": {"path": str(parquet.resolve()), "sha256": source["sha256"], "rows": len(rows)},
                "text_construction": {"order": "original parquet order, including empty rows", "separator": "\n\n",
                                      "characters": len(text), "sha256": hashlib.sha256(text.encode()).hexdigest()},
                "tokenizer": identity, "tokenization": {"add_special_tokens": False, "truncation": False,
                    "calls": 1, "packages": {name: importlib.metadata.version(name) for name in ("transformers", "tokenizers", "pyarrow")}},
                "token_stream": {"tokens": len(token_ids), "sha256": json_hash(token_ids), "hash_encoding": "canonical JSON integer array"},
                "context_prefix_tokens": args.prefix_tokens, "engine_prompt_tokens": args.prefix_tokens-1,
                "max_scored_tokens": args.scored_tokens,
                "windows_file": {"path": output.name, "rows": len(windows), "sha256": file_hash(output)},
                "coverage": {"excluded_initial_span": [0, args.prefix_tokens], "scored_span": [args.prefix_tokens, len(token_ids)],
                             "scored_tokens": len(token_ids)-args.prefix_tokens, "scored_spans_overlap": False,
                             "scored_spans_have_gaps": False, "last_window_scored_tokens": windows[-1]["scored_tokens"],
                             "tail_policy": "score the final short window without padding; never drop incomplete batches"},
                "scope": "rolling teacher-forced decode PPL/KL with reset context P..P+S-1; all scored positions are decode",
                "statistical_unit": "windows of one ordered split token stream; not independent articles or system repetitions",
                "prefill_bridge": "token at score_start-1 is emitted by prefill and excluded; every score_start:score_stop prediction uses decode",
                "standard_protocol_equivalence_claimed": False}
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    load_rolling_documents(manifest_path)
    print(json.dumps({"status": "READY", "manifest": str(manifest_path), "windows": len(windows),
                      "source_tokens": len(token_ids), "scored_tokens": len(token_ids)-args.prefix_tokens}))


if __name__ == "__main__":
    main()
