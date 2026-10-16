#!/usr/bin/env python3
"""Resume pinned HF-identical DeepSeek weights from official ModelScope.

Default is a metadata/space plan. --download is required for full transfers.
Never stops the original downloader: download mode takes its existing lock
nonblockingly and refuses while it is held. No GPU is used. --adopt-hf-partials
migrates eligible HF partials under that lock. --fetch-small-assets fills missing
nonweight files from the fixed HF revision, capped at 32 MiB. Source manifests and
existing small assets are untouched. WEIGHTS_COMPLETE is not model-load success.

Example after the original downloader has exited:
  python code/prepare_deepseek_mirror.py --root /root/autodl-tmp/research_K_expert_skip \
    --source-manifest runs/deepseek_source_r02/source.json --download \
    --adopt-hf-partials --fetch-small-assets

Adapted from research_A_horizon/code/fetch_qwen_mirror.py: pinned per-file
ModelScope revision, strict HTTP ranges, resumable partials, final SHA256 checks.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import threading
import time
import urllib.error
import urllib.parse
import urllib.request


REPO = "deepseek-ai/DeepSeek-V2-Lite"
HF_REVISION = "604d5664dddd88a0433dbae533b7fe9472482de0"
METADATA_URL = f"https://www.modelscope.cn/api/v1/models/{REPO}/repo/files?Revision=master&Recursive=true"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 2**20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, data):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def allocated(path):
    total = 0
    seen = set()
    for file in Path(path).rglob("*"):
        if file.is_file():
            stat = file.stat()
            key = (stat.st_dev, stat.st_ino)
            if key not in seen:
                total += stat.st_blocks * 512
                seen.add(key)
    return total


def other_reservations(source, root):
    """Refresh known shared downloads; unknown entries retain conservative debt."""
    shared = root.parent
    rows = []
    for old in source.get("other_download_reservations", []):
        name = old["name"]
        if name.startswith("qwen3_"):
            path = shared / "research_C_format_sched/checkpoints" / name
        elif name == "Qwen3-8B":
            path = shared / "research_A_horizon/models/hf"
        else:
            path = shared / "research_B_tiered_combine/models" / name
        if path.exists():
            present = allocated(path)
            rows.append({**old, "path": str(path), "allocated_bytes": present,
                         "remaining_bytes": max(0, old["expected_bytes"] - present),
                         "reservation_source": "current_allocated_blocks"})
        else:
            rows.append({**old, "reservation_source": "original_manifest_unresolved_path"})
    return rows


def load_metadata():
    request = urllib.request.Request(METADATA_URL, headers={"Accept-Encoding": "identity"})
    with urllib.request.urlopen(request, timeout=30) as response:
        raw = response.read(2 * 2**20 + 1)
    if len(raw) > 2 * 2**20:
        raise RuntimeError("ModelScope metadata exceeds 2 MiB bound")
    metadata = json.loads(raw)
    if metadata.get("Code") != 200 or metadata.get("Success") is not True:
        raise RuntimeError("ModelScope metadata reports failure")
    return metadata


def match_shards(source, metadata):
    if source.get("repo") != REPO or source.get("revision") != HF_REVISION:
        raise ValueError("source must be the original pinned DeepSeek-V2-Lite HF revision")
    mirror = {f["Path"]: f for f in metadata["Data"]["Files"] if f.get("Type") == "blob"}
    shards = []
    for row in source["files"]:
        name = row["path"]
        if not name.endswith(".safetensors"):
            continue
        if PurePosixPath(name).name != name or not re.fullmatch(r"[0-9a-f]{64}", row.get("lfs_sha256") or ""):
            raise ValueError("invalid pinned weight filename or SHA256")
        other = mirror.get(name, {})
        if other.get("Sha256") != row["lfs_sha256"] or other.get("Size") != row["size"]:
            raise ValueError("mirror/HF size or SHA256 mismatch: " + name)
        revision = other.get("Revision", "")
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("missing pinned ModelScope file revision: " + name)
        url = f"https://www.modelscope.cn/models/{REPO}/resolve/{revision}/{urllib.parse.quote(name)}"
        shards.append({**row, "modelscope_revision": revision, "url": url})
    if len(shards) != 4 or not 20 * 2**30 < sum(r["size"] for r in shards) < 40 * 2**30:
        raise ValueError("unexpected shard count or checkpoint size")
    return shards


def small_asset_status(source, destination):
    rows = []
    for row in source["files"]:
        name = row["path"]
        if name.endswith(".safetensors"):
            continue
        if PurePosixPath(name).name != name or any(c in name for c in "*?["):
            raise ValueError("unsafe nonweight asset name")
        path = destination / name
        present = path.is_file()
        size = path.stat().st_size if present else None
        rows.append({"path": name, "expected_size": row["size"], "actual_size": size,
                     "status": "missing" if not present else "present_at_expected_size" if size == row["size"] else "size_mismatch"})
    return {"nonweight_files": rows, "missing_assets": [r["path"] for r in rows if r["status"] == "missing"],
            "size_mismatch_assets": [r["path"] for r in rows if r["status"] == "size_mismatch"],
            "asset_validation_scope": "existence and original HF-manifest size; not model-load validation"}


def plan_partial_adoptions(shards, destination):
    """No writes; only unique hash-named regular HF partials are eligible."""
    cache = destination / ".cache/huggingface/download"
    candidates = list(cache.rglob("*.incomplete")) if cache.is_dir() else []
    plans = []
    for row in shards:
        target = destination / row["path"]
        partial = destination / ".modelscope-partials" / (row["lfs_sha256"] + ".partial")
        if target.exists() or partial.exists():
            continue
        pattern = re.compile(r"(?:^|\.)" + re.escape(row["lfs_sha256"]) + r"(?:\.|$)")
        matches = [p for p in candidates if pattern.search(p.name)]
        if not matches:
            continue
        if len(matches) != 1:
            raise ValueError("ambiguous HF partials for " + row["path"])
        old = matches[0]
        size = old.stat().st_size
        if old.is_symlink() or not old.is_file() or not 0 < size <= row["size"]:
            raise ValueError("unsafe or oversized HF partial for " + row["path"])
        plans.append({"file": row["path"], "sha256": row["lfs_sha256"], "from": str(old),
                      "to": str(partial), "size": size, "status": "PLANNED"})
    return plans


def adopt_partial(plan):
    """Move one inode without overwriting an existing destination; caller holds lock."""
    old, new = Path(plan["from"]), Path(plan["to"])
    if old.is_symlink() or not old.is_file() or old.stat().st_size != plan["size"]:
        raise RuntimeError("HF partial changed since planning; refusing adoption")
    new.parent.mkdir(parents=True, exist_ok=True)
    # link() fails atomically if new exists, unlike POSIX rename() overwrite.
    # Unlinking the old name then completes a same-filesystem inode migration.
    os.link(old, new, follow_symlinks=False)
    old.unlink()
    return {**plan, "status": "ADOPTED_UNVERIFIED", "method": "link_then_unlink_no_clobber",
            "verification": "full original HF SHA256 required before weight installation"}


def fetch_missing_small_assets(source, destination):
    status = small_asset_status(source, destination)
    missing = status["missing_assets"]
    total = sum(r["size"] for r in source["files"] if r["path"] in missing)
    if total > 32 * 2**20:
        raise ValueError("missing nonweight assets exceed the 32 MiB download bound")
    if missing:
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
        from huggingface_hub import snapshot_download
        snapshot_download(REPO, revision=HF_REVISION, endpoint=source["endpoint"], token=False,
                          local_dir=destination, allow_patterns=missing, max_workers=1)
    return {"endpoint": source["endpoint"], "revision": HF_REVISION,
            "allow_patterns": missing, "expected_new_bytes": total,
            "semantics": "only previously missing nonweight files; no weight fallback"}


def download_shard(row, destination, chunk_bytes, retries, stop=None):
    target = destination / row["path"]
    partial = destination / ".modelscope-partials" / (row["lfs_sha256"] + ".partial")
    started = time.time()
    if target.exists():
        if target.stat().st_size != row["size"] or digest(target) != row["lfs_sha256"]:
            raise RuntimeError("existing destination fails pinned SHA256; left untouched: " + str(target))
        return {"file": row["path"], "sha256": row["lfs_sha256"], "source": "existing_verified_file"}
    partial.parent.mkdir(parents=True, exist_ok=True)
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > row["size"]:
        raise RuntimeError("partial exceeds expected shard size; left untouched")
    resumed_bytes = offset
    failures = 0
    while offset < row["size"]:
        if stop is not None and stop.is_set():
            raise RuntimeError("mirror transfer cancelled; partial preserved")
        end = min(row["size"] - 1, offset + chunk_bytes - 1)
        request = urllib.request.Request(row["url"], headers={
            "Range": f"bytes={offset}-{end}", "Accept-Encoding": "identity"})
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                expected = f"bytes {offset}-{end}/{row['size']}"
                if response.status != 206 or response.headers.get("Content-Range") != expected:
                    raise ValueError("unexpected range response; no response body saved: " + row["path"])
                if response.headers.get("Content-Encoding", "identity") != "identity":
                    raise ValueError("compressed range response rejected")
                length = response.headers.get("Content-Length")
                if length is not None and int(length) != end - offset + 1:
                    raise ValueError("unexpected range Content-Length")
                with partial.open("ab") as output:
                    while offset <= end:
                        if stop is not None and stop.is_set():
                            raise RuntimeError("mirror transfer cancelled; partial preserved")
                        block = response.read(min(2**20, end + 1 - offset))
                        if not block:
                            raise IOError("truncated range")
                        output.write(block)
                        offset += len(block)
                    output.flush()
                    os.fsync(output.fileno())
            failures = 0
        except (urllib.error.URLError, TimeoutError, OSError):
            failures += 1
            if failures > retries:
                raise
            offset = partial.stat().st_size if partial.exists() else 0
            delay = min(2**failures, 30)
            if stop is None:
                time.sleep(delay)
            else:
                stop.wait(delay)
        print(json.dumps({"file": row["path"], "bytes": offset, "total": row["size"]}), flush=True)
    if digest(partial) != row["lfs_sha256"]:
        raise RuntimeError("completed partial fails original HF SHA256; preserved: " + str(partial))
    if target.exists():
        raise RuntimeError("destination appeared during download; refusing overwrite")
    partial.rename(target)
    return {"file": row["path"], "sha256": row["lfs_sha256"], "size": row["size"],
            "resumed_bytes": resumed_bytes, "source": row["url"], "elapsed_s": time.time() - started}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/root/autodl-tmp/research_K_expert_skip"))
    parser.add_argument("--source-manifest", type=Path, default=Path("runs/deepseek_source_r02/source.json"))
    parser.add_argument("--download", action="store_true", help="perform full download after acquiring original downloader lock")
    parser.add_argument("--adopt-hf-partials", action="store_true", help="migrate uniquely hash-matched HF partials only under download lock")
    parser.add_argument("--fetch-small-assets", action="store_true", help="fill only missing nonweight assets from pinned HF revision, <=32 MiB")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--chunk-mib", type=int, default=64)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--reserve-gib", type=float, default=20)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4 or not 1 <= args.chunk_mib <= 128 or args.retries < 0 or args.reserve_gib < 20:
        parser.error("workers 1..4, chunk-mib 1..128, retries >= 0, reserve-gib >= 20 required")
    root = args.root.resolve()
    source_path = args.source_manifest if args.source_manifest.is_absolute() else root / args.source_manifest
    source = json.loads(source_path.read_text())
    destination = Path(source["local_dir"]).resolve()
    if not destination.is_relative_to(root / "models"):
        raise ValueError("source destination must be inside this research root/models")
    lock = None
    if args.download:
        lock = (root / "deepseek_download.lock").open("a")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("original DeepSeek downloader holds lock; no second download started") from None
    metadata = load_metadata()
    shards = match_shards(source, metadata)
    adoptions = plan_partial_adoptions(shards, destination) if args.adopt_hf_partials else []
    adopt_sizes = {r["file"]: r["size"] for r in adoptions}
    assets = small_asset_status(source, destination)
    missing = 0
    for row in shards:
        target = destination / row["path"]
        partial = destination / ".modelscope-partials" / (row["lfs_sha256"] + ".partial")
        present = target.stat().st_size if target.exists() else partial.stat().st_size if partial.exists() else adopt_sizes.get(row["path"], 0)
        if present > row["size"]:
            raise ValueError("existing weight or partial exceeds expected size")
        missing += row["size"] - present
    reservations = other_reservations(source, root)
    remaining_other = sum(r["remaining_bytes"] for r in reservations)
    missing_assets_bytes = sum(r["size"] for r in source["files"] if r["path"] in assets["missing_assets"])
    if args.fetch_small_assets and missing_assets_bytes > 32 * 2**20:
        raise ValueError("missing nonweight assets exceed 32 MiB")
    free = shutil.disk_usage(root).free
    receipt = {"status": "PLAN", "pid": os.getpid(), "start_unix": time.time(), "repo": REPO,
               "hf_revision": HF_REVISION, "source_manifest": str(source_path),
               "source_manifest_sha256": digest(source_path), "destination": str(destination),
               "metadata_url": METADATA_URL, "modelscope_metadata": metadata, "shards": shards,
               "workers": args.workers, "chunk_bytes": args.chunk_mib * 2**20,
               "missing_weight_bytes": missing, "free_bytes": free,
               "other_download_reservations": reservations, "remaining_other_bytes": remaining_other,
               "reserve_bytes": int(args.reserve_gib * 2**30), "verified": [], "partial_adoptions": adoptions,
               "missing_assets_bytes": missing_assets_bytes, "assets": assets}
    if free - missing - missing_assets_bytes - remaining_other < args.reserve_gib * 2**30:
        raise RuntimeError("insufficient space after other downloads and free-space reserve")
    if not args.download:
        print(json.dumps(receipt, indent=2))
        return
    run_dir = root / "runs" / ("deepseek_mirror_" + time.strftime("%Y%m%d_%H%M%S") + f"_{os.getpid()}")
    run_dir.mkdir(parents=True, exist_ok=False)
    receipt_path = run_dir / "source.json"
    receipt["status"] = "DOWNLOADING"
    atomic_json(receipt_path, receipt)
    stop = threading.Event()
    try:
        destination.mkdir(parents=True, exist_ok=True)
        for index, plan in enumerate(adoptions):
            receipt["partial_adoptions"][index] = adopt_partial(plan)
            atomic_json(receipt_path, receipt)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(download_shard, row, destination, args.chunk_mib * 2**20, args.retries, stop)
                       for row in shards]
            try:
                for future in as_completed(futures):
                    receipt["verified"].append(future.result())
                    atomic_json(receipt_path, receipt)
            except BaseException:
                stop.set()
                for future in futures:
                    future.cancel()
                raise
        receipt["status"] = "WEIGHTS_COMPLETE"
        if args.fetch_small_assets:
            receipt["small_asset_download"] = fetch_missing_small_assets(source, destination)
        receipt["assets"] = small_asset_status(source, destination)
    except BaseException as error:
        receipt.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        receipt["ended_unix"] = time.time()
        atomic_json(receipt_path, receipt)
    print(json.dumps({"status": receipt["status"], "receipt": str(receipt_path),
                      "verified_shards": len(receipt["verified"]),
                      "missing_assets": receipt["assets"]["missing_assets"],
                      "size_mismatch_assets": receipt["assets"]["size_mismatch_assets"]}), flush=True)


if __name__ == "__main__":
    main()
