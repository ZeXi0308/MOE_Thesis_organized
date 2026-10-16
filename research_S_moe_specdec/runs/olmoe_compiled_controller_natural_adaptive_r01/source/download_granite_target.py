#!/usr/bin/env python3
"""CPU-only pinned Granite3B target assembly; B assets are always read-only.

python code/download_granite_target.py --endpoint https://hf-mirror.com
python code/download_granite_target.py --resolve-only --endpoint https://hf-mirror.com
python code/download_granite_target.py --check-only

All outputs, metadata, locks and partial files stay under
S/models/granite3b/<fixed revision>/. Complete B files are verified before any
copy, copied to new regular files, verified again, then atomically published.
Only the missing first shard and .gitattributes may be fetched. A missing or
inconsistent expected reusable file is an error, not permission to redownload it.
The existing common downloader supplies Range resume and final content checks.
No torch, transformers, GPU initialization, global HF cache or B writes.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import sys
from urllib.parse import quote

import download_draft as common


REPO = "ibm-granite/granite-3.1-3b-a800m-base"
REVISION = "d4dd87aa3a6c201bc374851d7d7ff4cf39a0b82a"
REUSE = Path("/root/autodl-tmp/research_B_tiered_combine/models/granite3b")
DRAFT = Path("/root/autodl-tmp/research_B_tiered_combine/models/granite1b")
DRAFT_REVISION = "408b6e90baab8cf24f4aa9f8e19703ffa0a53b29"
NETWORK_FILES = {".gitattributes", "model-00001-of-00002.safetensors"}
TOTAL_BYTES = 6599724398


def regular_path(path, *, output=False):
    """Reject symlinked paths/ancestors, including dangling symlinks."""
    path = Path(path).absolute()
    if path.resolve() != path or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError(f"symlinked path is not allowed: {path}")
    if path.exists() and not path.is_file():
        raise ValueError(f"expected a regular file: {path}")
    if output:
        common.confined(path)
        if path.exists() and path.stat().st_nlink != 1:
            raise ValueError(f"multiply linked output is not allowed: {path}")
    return path


def source_file(root, name):
    name = PurePosixPath(name)
    if name.is_absolute() or ".." in name.parts or "\\" in str(name) or not name.parts:
        raise ValueError("unsafe source repository path")
    root = Path(root).absolute()
    path = regular_path(root / name)
    if not path.is_relative_to(root):
        raise ValueError("source path escaped its declared directory")
    return path


def target_file(root, name):
    path = common.file_path(root, name)
    # Check the lexical path too: common.confined resolves existing symlinks.
    regular_path(Path(root) / name, output=True)
    return path


def write_json(path, data):
    regular_path(path, output=True)
    regular_path(path.with_name(path.name + ".tmp"), output=True)
    common.save_json(path, data)


def entries_from_source(source, endpoint, target):
    if (source["repo_id"] != REPO or source["revision"] != REVISION
            or source["target_directory"] != f"models/granite3b/{REVISION}/target"
            or source["reuse_directory"] != str(REUSE)
            or source["draft"]["model_path"] != str(DRAFT)
            or source["draft"]["source_manifest"] != str(DRAFT.parents[1] / "runs/model_source_granite1b/source.json")
            or source["draft"]["repo_id"] != "ibm-granite/granite-3.1-1b-a400m-base"
            or source["draft"]["revision"] != DRAFT_REVISION
            or source["workload"]["path"] != "configs/sharegpt_granite1b.json"):
        raise ValueError("source must retain the pinned target/draft and fixed project paths")
    if set(source["network_allowed_files"]) != NETWORK_FILES:
        raise ValueError("network file allowlist differs from the authorized missing assets")
    entries, names = [], set()
    for item in source["files"]:
        name, size = item["name"], item["bytes"]
        target_file(target, name)
        if name in names or type(size) is not int or size < 0:
            raise ValueError("duplicate file name or invalid expected size")
        names.add(name)
        sha, git = item.get("sha256"), item.get("git_blob_sha1")
        if not (re.fullmatch(r"[0-9a-f]{64}", sha or "") or re.fullmatch(r"[0-9a-f]{40}", git or "")):
            raise ValueError(f"missing fixed content checksum: {name}")
        entries.append({"name": name, "size": size, "sha256": sha, "git_blob_sha1": git,
                        "repo_id": REPO, "revision": REVISION,
                        "url": f"{endpoint}/{REPO}/resolve/{REVISION}/{quote(name, safe='/')}"})
    if (sum(e["size"] for e in entries) != source["expected_total_bytes"]
            or source["expected_total_bytes"] != TOTAL_BYTES or len(entries) != 10
            or not NETWORK_FILES.issubset(names)):
        raise ValueError("pinned inventory byte total or file set differs")
    return entries


def check_inventory(expected, upstream):
    actual = {e["name"]: e for e in upstream}
    if len(actual) != len(upstream) or set(actual) != {e["name"] for e in expected}:
        raise ValueError("upstream inventory differs from the pinned full repository")
    for entry in expected:
        other = actual[entry["name"]]
        keys = ("size", "repo_id", "revision")
        if any(entry[k] != other[k] for k in keys):
            raise ValueError(f"upstream size/revision mismatch: {entry['name']}")
        # Small Git files additionally carry our observed SHA256, which HF's
        # metadata need not expose. Its Git blob ID is the upstream identity.
        key = "sha256" if other.get("sha256") else "git_blob_sha1"
        if not other.get(key) or entry.get(key) != other[key]:
            raise ValueError(f"upstream content hash mismatch: {entry['name']}")


def plan_transfer(entries, target, reuse, network_files):
    """Verify every present destination/source before copying anything."""
    jobs = []
    for entry in entries:
        dest = target_file(target, entry["name"])
        for suffix in (".part", ".copying"):
            regular_path(dest.with_name(dest.name + suffix), output=True)
        if dest.exists():
            jobs.append({"entry": entry, "kind": "existing_s_file", "sha256": common.verify(dest, entry)})
            continue
        src = source_file(reuse, entry["name"])
        if src.is_file():
            jobs.append({"entry": entry, "kind": "verified_b_copy", "source": src,
                         "sha256": common.verify(src, entry)})
        elif entry["name"] in network_files:
            jobs.append({"entry": entry, "kind": "pinned_network_download"})
        else:
            raise ValueError(f"required reusable B asset is missing: {entry['name']}")
    return jobs


def copy_verified(src, target, entry):
    src = regular_path(src)
    dest = target_file(target, entry["name"])
    if dest.exists():
        return common.verify(dest, entry)
    temp = regular_path(dest.with_name(dest.name + ".copying"), output=True)
    temp.parent.mkdir(parents=True, exist_ok=True)
    # Never create a symlink copy if the source changes after preflight.
    with os.fdopen(os.open(src, os.O_RDONLY | os.O_NOFOLLOW), "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError("copy source is no longer a regular file")
        with temp.open("wb") as output:
            shutil.copyfileobj(stream, output, length=8 * 1024 * 1024)
    regular_path(temp, output=True)
    digest = common.verify(temp, entry)
    temp.replace(dest)
    return digest


def compatibility(source, target):
    target_config = json.loads(target_file(target, "config.json").read_text())
    if any(target_config.get(k) != v for k, v in source["expected_config"].items()):
        raise ValueError("downloaded target architecture differs from the pinned E40/k8 model")
    index = json.loads(target_file(target, "model.safetensors.index.json").read_text())
    shards = set(index["weight_map"].values())
    expected = {e["name"] for e in source["files"] if e["name"].endswith(".safetensors")}
    if shards != expected or any(not target_file(target, name).is_file() for name in shards):
        raise ValueError("target index references missing or unexpected weight shards")
    draft = source["draft"]
    manifest_path = regular_path(Path(draft["source_manifest"]))
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != draft["source_manifest_sha256"]:
        raise ValueError("read-only Granite1B source manifest changed")
    original = json.loads(manifest_path.read_text())
    if original.get("repo") != draft["repo_id"] or original.get("revision") != DRAFT_REVISION or original.get("status") != "COMPLETE":
        raise ValueError("read-only Granite1B snapshot is not the pinned complete model")
    checks = {}
    for item in draft["files"]:
        path = source_file(DRAFT, item["name"])
        entry = {"name": item["name"], "size": item["bytes"], "sha256": item["sha256"]}
        checks[item["name"]] = {"bytes": item["bytes"], "verified_sha256": common.verify(path, entry)}
    dc = json.loads(source_file(DRAFT, "config.json").read_text())
    if any(dc.get(k) != v for k, v in draft["expected_config"].items()):
        raise ValueError("read-only draft architecture differs from the pinned E32/k8 model")
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "generation_config.json"):
        if target_file(target, name).read_bytes() != source_file(DRAFT, name).read_bytes():
            raise ValueError(f"target/draft tokenization or stopping metadata differs: {name}")
    tc = json.loads(target_file(target, "tokenizer_config.json").read_text())
    if tc.get("chat_template"):
        raise ValueError("raw-continuation tokenizer unexpectedly has a chat template")
    workload_path = common.PROJECT / source["workload"]["path"]
    if hashlib.sha256(workload_path.read_bytes()).hexdigest() != source["workload"]["sha256"]:
        raise ValueError("fixed raw-continuation workload changed")
    return {"checked_at": common.timestamp(), "status": "structural_checks_passed_runtime_unverified",
            "target_repo": REPO, "target_revision": REVISION, "target_experts": 40,
            "draft_repo": draft["repo_id"], "draft_revision": DRAFT_REVISION, "draft_experts": 32,
            "vocab_size": target_config["vocab_size"], "tokenizer_files_byte_identical": True,
            "raw_continuation": True, "draft_verified_files": checks,
            "workload_sha256": source["workload"]["sha256"], "runtime_status": source["runtime_status"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=common.PROJECT / "configs/granite3b_target_source.json")
    parser.add_argument("--endpoint", choices=common.ENDPOINTS, default=common.ENDPOINTS[0])
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--resolve-only", action="store_true")
    modes.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    source = json.loads(args.config.read_text())
    raw_base = common.MODELS / "granite3b" / REVISION
    if raw_base.resolve() != raw_base.absolute():
        raise ValueError("target output directory must not contain symlinks")
    base = common.confined(raw_base)
    target = base / "target"
    entries = entries_from_source(source, args.endpoint, target)
    base.mkdir(parents=True, exist_ok=True)
    with regular_path(base / ".download.lock", output=True).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.check_only:
            verified = {e["name"]: common.verify(target_file(target, e["name"]), e) for e in entries}
            report = compatibility(source, target)
            report["target_verified_files"] = verified
            write_json(base / "compatibility.json", report)
            print(json.dumps({"status": report["status"], "target_path": str(target)}))
            return
        cache = regular_path(base / "upstream_metadata.json", output=True)
        regular_path(cache.with_name(cache.name + ".tmp"), output=True)
        upstream = common.inventory(REPO, REVISION, args.endpoint, cache)
        check_inventory(entries, upstream)
        manifest = {"status": "resolved", "resolved_at": common.timestamp(), "repo": REPO,
                    "revision": REVISION, "source_manifest_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
                    "expected_total_bytes": TOTAL_BYTES, "target_path": str(target), "files": entries,
                    "source": source, "endpoint": args.endpoint, "verified_files": {}}
        receipt = base / "download_manifest.json"
        write_json(receipt, manifest)
        print(json.dumps({"status": "resolved", "expected_total_bytes": TOTAL_BYTES,
                          "target_path": str(target)}), flush=True)
        if args.resolve_only:
            return
        try:
            jobs = plan_transfer(entries, target, REUSE, NETWORK_FILES)
            # Finish all validated copies before starting any network transfer.
            jobs.sort(key=lambda j: j["kind"] == "pinned_network_download")
            for job in jobs:
                entry = job["entry"]
                print(f"{job['kind']} {entry['name']} ({entry['size']} bytes)", flush=True)
                if job["kind"] == "verified_b_copy":
                    digest = copy_verified(job["source"], target, entry)
                elif job["kind"] == "pinned_network_download":
                    digest = common.download(entry, target)
                else:
                    digest = job["sha256"]
                manifest["verified_files"][entry["name"]] = {
                    "bytes": entry["size"], "sha256": digest, "method": job["kind"],
                    "source_path": str(job["source"]) if "source" in job else None}
                manifest["status"] = "assembling"
                write_json(receipt, manifest)
            report = compatibility(source, target)
            write_json(base / "compatibility.json", report)
            manifest.update(status="complete", completed_at=common.timestamp(), compatibility_status=report["status"])
            write_json(receipt, manifest)
            print(json.dumps({"status": "complete", "target_path": str(target),
                              "compatibility_status": report["status"]}), flush=True)
        except Exception as error:
            manifest.update(status="incomplete", last_error_type=type(error).__name__)
            write_json(receipt, manifest)
            raise


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        # Never print signed redirect URLs or arbitrary network exception text.
        message = str(error) if isinstance(error, (ValueError, RuntimeError)) else "see local manifest; files retained"
        print(f"{type(error).__name__}: {message}", file=sys.stderr)
        sys.exit(1)
