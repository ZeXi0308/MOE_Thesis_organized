#!/usr/bin/env python3
"""Download the pinned OLMo dense draft using the existing resumable downloader.

CPU only; no torch, model loading, global HF cache, or writes to the target.
All outputs stay in this project's models/olmo_1b_0724_draft/<revision>/.

  python code/download_olmo_draft.py --resolve-only --endpoint https://hf-mirror.com
  python code/download_olmo_draft.py --endpoint https://hf-mirror.com --target-path /model
  python code/download_olmo_draft.py --check-only --target-path /model

--check-only refreshes config/tokenizer compatibility from local files, verifies
the small draft files, and does not rehash weights or contact the network.
Missing local target files are pending_local_target, not a failed draft.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import re
import sys
from urllib.parse import quote

import download_draft as common


MAX_BUNDLE_BYTES = 6 * 1024**3


def manifest_entries(source, endpoint, draft_path):
    if source.get("method") != "draft_model":
        raise ValueError("this entry point requires method=draft_model")
    if not re.fullmatch(r"[0-9a-f]{40}", source["revision"]):
        raise ValueError("a full immutable draft revision is required")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", source["repo_id"]):
        raise ValueError("invalid HF repository id")
    entries, seen = [], set()
    for item in source["files"]:
        name, size = item["path"], item["bytes"]
        common.file_path(draft_path, name)
        if name in seen or not isinstance(size, int) or size < 0:
            raise ValueError("duplicate file or invalid file size in source manifest")
        seen.add(name)
        sha = item.get("sha256")
        blob = None if sha else item.get("git_blob_sha1")
        if not (re.fullmatch(r"[0-9a-f]{64}", sha or "") or re.fullmatch(r"[0-9a-f]{40}", blob or "")):
            raise ValueError(f"missing content checksum for {name}")
        entries.append({"name": name, "size": size, "sha256": sha, "git_blob_sha1": blob,
                        "repo_id": source["repo_id"], "revision": source["revision"],
                        "url": f"{endpoint}/{source['repo_id']}/resolve/{source['revision']}/{quote(name, safe='/')}"})
    total = sum(e["size"] for e in entries)
    if total != source["total_repository_bytes"] or total > MAX_BUNDLE_BYTES:
        raise ValueError("source byte total disagrees with manifest or exceeds 6 GiB")
    if not {"config.json", "tokenizer.json", "model.safetensors.index.json"}.issubset(seen):
        raise ValueError("required draft files missing from source manifest")
    return entries


def check_inventory(expected, actual):
    fields = ("size", "sha256", "git_blob_sha1", "repo_id", "revision")
    by_name = {entry["name"]: entry for entry in actual}
    if len(by_name) != len(actual) or set(by_name) != {entry["name"] for entry in expected}:
        raise ValueError("upstream file inventory differs from the pinned source manifest")
    for entry in expected:
        if any(entry.get(key) != by_name[entry["name"]].get(key) for key in fields):
            raise ValueError(f"upstream size/hash/revision differs for {entry['name']}")


def normalized_merges(tokenizer):
    return [tuple(pair.split(" ")) if isinstance(pair, str) else tuple(pair)
            for pair in tokenizer["model"]["merges"]]


def compare_tokenizers(draft, target, source):
    spec = source["tokenizer_verification"]
    checks = {
        "model_type_equal": draft["model"]["type"] == target["model"]["type"] == "BPE",
        "base_vocab_count_matches": len(draft["model"]["vocab"]) == len(target["model"]["vocab"]) == spec["base_vocab_entries"],
        "base_token_to_id_equal": draft["model"]["vocab"] == target["model"]["vocab"],
        "bpe_merges_equal": normalized_merges(draft) == normalized_merges(target),
    }
    for key in ("normalizer", "pre_tokenizer", "decoder", "post_processor"):
        checks[key + "_equal"] = draft.get(key) == target.get(key)
    draft_added, target_added = draft.get("added_tokens", []), target.get("added_tokens", [])
    checks["all_draft_added_tokens_match_target"] = all(t in target_added for t in draft_added)
    target_only = [t for t in target_added if t not in draft_added]
    pad = spec["target_only_added_token"]
    checks["only_expected_target_pad_is_extra"] = all(
        t.get("id") == pad["id"] and t.get("content") == pad["content"] and t.get("special") is True
        for t in target_only) and len(target_only) <= 1
    return {"checks": checks, "target_only_added_tokens": target_only,
            "status": "matched_with_target_pad_caveat" if all(checks.values()) else "mismatch",
            "caveat": spec["boundary"]}


def file_check(path, expected_sha256=None):
    exists = path.is_file()
    actual = hashlib.sha256(path.read_bytes()).hexdigest() if exists else None
    return {"path": str(path), "exists": exists, "expected_sha256": expected_sha256,
            "actual_sha256": actual,
            "hash_matches": actual == expected_sha256 if exists and expected_sha256 else None}


def compatibility(source, draft_path, target_path):
    dc = json.loads((draft_path / "config.json").read_text())
    dt = json.loads((draft_path / "tokenizer.json").read_text())
    expected = source["expected_config"]
    draft_mismatches = [key for key, value in expected.items() if dc.get(key) != value]
    draft_tokenizer = file_check(draft_path / "tokenizer.json", source["tokenizer_verification"]["draft_tokenizer_sha256"])
    errors = []
    if draft_mismatches or not draft_tokenizer["hash_matches"]:
        errors.append("downloaded draft config/tokenizer differs from the pinned candidate")
    local = {"status": "pending_local_target", "path": str(target_path) if target_path else None,
             "missing_files": ["config.json", "tokenizer.json"], "files": {}}
    if target_path is not None:
        local["files"] = {
            "config.json": file_check(target_path / "config.json"),
            "tokenizer.json": file_check(target_path / "tokenizer.json", source["tokenizer_verification"]["target_tokenizer_sha256"]),
        }
        local["missing_files"] = [name for name, state in local["files"].items() if not state["exists"]]
        mismatches = []
        if local["files"]["config.json"]["exists"]:
            tc = json.loads((target_path / "config.json").read_text())
            target_expected = source["expected_target_config"]
            mismatches = [key for key, value in target_expected.items() if tc.get(key) != value]
        local["config_mismatches"] = mismatches
        token_mismatch = False
        if local["files"]["tokenizer.json"]["exists"]:
            tt = json.loads((target_path / "tokenizer.json").read_text())
            local["tokenizer_comparison"] = compare_tokenizers(dt, tt, source)
            token_mismatch = local["tokenizer_comparison"]["status"] == "mismatch"
            local["snapshot_identity"] = ("pinned_tokenizer_bytes_match" if local["files"]["tokenizer.json"]["hash_matches"]
                                          else "tokenizer_bytes_differ; semantic comparison recorded separately")
        local["status"] = ("mismatch" if mismatches or token_mismatch else
                           "pending_local_target" if local["missing_files"] else "matched_with_target_pad_caveat")
        if mismatches or token_mismatch:
            errors.append("existing target config or token-ID mapping differs from the candidate's supported target")
    return {"checked_at": common.timestamp(),
            "status": "failed" if errors else "pending_local_target" if local["status"] == "pending_local_target" else "structural_checks_passed_with_pad_caveat",
            "draft_config_mismatches": draft_mismatches, "draft_tokenizer": draft_tokenizer,
            "local_target": local, "errors": errors,
            "target_revision_expected": source["target_revision"],
            "runtime_status": "not tested; native load, actual prompt IDs, acceptance, draft KV allocation and complete cost still require validation",
            "weight_validation": "This report checks config/tokenizers only; download_manifest.json records per-file content verification. --check-only never verifies weights.",
            "runtime_dtype": source["runtime_dtype"],
            "caveat": source["tokenizer_verification"]["boundary"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=common.PROJECT / "configs/olmo_dense_draft.json")
    parser.add_argument("--endpoint", choices=common.ENDPOINTS, default=common.ENDPOINTS[0])
    parser.add_argument("--target-path", type=Path, help="read-only target model directory; uses its config.json and tokenizer.json")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--resolve-only", action="store_true")
    modes.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    source = json.loads(args.config.read_text())
    base = common.confined(common.MODELS / "olmo_1b_0724_draft" / source["revision"])
    draft_path = base / "draft"
    entries = manifest_entries(source, args.endpoint, draft_path)
    base.mkdir(parents=True, exist_ok=True)
    target_path = args.target_path.resolve() if args.target_path else None
    with common.confined(base / ".download.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.check_only:
            for entry in entries:
                if entry["name"] in ("config.json", "tokenizer.json"):
                    common.verify(common.file_path(draft_path, entry["name"]), entry)
            report = compatibility(source, draft_path, target_path)
            common.save_json(base / "compatibility.json", report)
            print(json.dumps({"mode": "local_compatibility_refresh", "compatibility_status": report["status"],
                              "draft_path": str(draft_path), "report": str(base / "compatibility.json")}))
            return 1 if report["errors"] else 0
        upstream = common.inventory(source["repo_id"], source["revision"], args.endpoint, base / "upstream_metadata.json")
        check_inventory(entries, upstream)
        manifest = {"resolved_at": common.timestamp(), "source": source,
                    "source_manifest_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
                    "endpoint": args.endpoint, "expected_total_bytes": sum(e["size"] for e in entries),
                    "draft_path": str(draft_path), "files": entries, "status": "resolved"}
        common.save_json(base / "download_manifest.json", manifest)
        print(json.dumps({"status": "resolved", "expected_total_bytes": manifest["expected_total_bytes"],
                          "draft_path": str(draft_path)}), flush=True)
        if args.resolve_only:
            return 0
        for entry in entries:
            print(f"verifying/downloading {entry['name']} ({entry['size']} bytes)", flush=True)
            entry["verified_sha256"] = common.download(entry, draft_path)
            entry["status"] = "verified"
            common.save_json(base / "download_manifest.json", manifest)
        report = compatibility(source, draft_path, target_path)
        common.save_json(base / "compatibility.json", report)
        manifest.update(status="complete", compatibility_status=report["status"])
        common.save_json(base / "download_manifest.json", manifest)
        print(json.dumps({"status": "complete", "compatibility_status": report["status"],
                          "draft_path": str(draft_path), "target_tokenizer_path": str(target_path) if target_path else None}), flush=True)
        return 1 if report["errors"] else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        sys.exit(1)
