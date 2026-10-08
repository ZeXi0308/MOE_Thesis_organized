#!/usr/bin/env python3
"""One bounded CPU-only shard preparation under C's shared nonblocking lock.

Invoke once with one chosen shard and a new immutable output root. Already
verified cache files are retained. No model, CUDA, or GPU worker is started.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import sys
import time

import C_SPARE_RESOURCE_V1 as resource
from C_INSTRUCT_CACHED_MODEL_V1 import (
    CACHE, MARKER_NAME, WEIGHT_NAMES, initialize_cache_marker,
    manifest_info, require_private_directory, stamp,
    validate_cache_marker, verify_pinned_file,
)
from C_INSTRUCT_DOWNLOAD_V2 import (
    GIB, MAX_SECONDS, PROGRESS_BYTES, RANGE_BYTES, MAX_WORKERS,
    download_ranged, memory_headroom, write_receipt,
)

DEFAULT_MANIFEST = resource.BASE / "C_INSTRUCT_MODEL_MANIFEST_V1.json"


def _cache_bytes(cache_dir, by_name):
    total = 0
    for entry in cache_dir.iterdir():
        if entry.name == MARKER_NAME:
            continue
        if entry.name not in WEIGHT_NAMES:
            raise RuntimeError("unknown or partial file in persistent cache: " + entry.name)
        verify_pinned_file(entry, by_name[entry.name])
        total += by_name[entry.name]["size"]
    return total


def _lock_shared_file():
    info = resource.LOCK.lstat()
    if not stat.S_ISREG(info.st_mode) or f"{info.st_dev}:{info.st_ino}" != resource.LOCK_INODE:
        raise RuntimeError("shared lock inode changed")
    fd = os.open(resource.LOCK, os.O_RDWR | os.O_NOFOLLOW)
    opened = os.fstat(fd)
    if not stat.S_ISREG(opened.st_mode) or f"{opened.st_dev}:{opened.st_ino}" != resource.LOCK_INODE:
        os.close(fd)
        raise RuntimeError("opened shared lock inode changed")
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BaseException:
        os.close(fd)
        raise
    return fd


def prepare_one_shard(shard, output, model_manifest=DEFAULT_MANIFEST):
    if shard not in WEIGHT_NAMES:
        raise ValueError("choose exactly one official weight shard")
    output, model_manifest = Path(output), Path(model_manifest)
    manifest, manifest_sha = manifest_info(model_manifest)
    by_name = {item["filename"]: item for item in manifest["files"]}
    item = by_name[shard]
    if output.exists() or output.is_symlink():
        raise RuntimeError("immutable shard preparation output root already exists")
    try:
        lock_fd = _lock_shared_file()
    except BlockingIOError:
        print(json.dumps(dict(status="GPU_DEFERRED", reason="shared lock busy")), flush=True)
        return 75
    try:
        output.mkdir(mode=0o700, exist_ok=False)
        receipt_path = output / "cache-shard-prepare.json"
        start = time.monotonic()
        deadline = start + MAX_SECONDS
        receipt = dict(status="PREPARING", started_utc=stamp(), shard=shard,
                       model_id=manifest["model_id"], revision=manifest["revision"],
                       model_manifest_sha256=manifest_sha, cache_path=str(CACHE),
                       expected_bytes=item["size"], expected_sha256=item["sha256"],
                       public_url=item["url"], deadline_seconds=MAX_SECONDS,
                       received_bytes=0, range_bytes=RANGE_BYTES,
                       workers=MAX_WORKERS, max_attempts_per_range=3,
                       range_attempts_total=0, range_retries_started=0,
                       range_attempts={}, downloaded_new_bytes=0,
                       downloader_sha256=hashlib.sha256(Path(download_ranged.__code__.co_filename).read_bytes()).hexdigest(),
                       cache_preparer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        write_receipt(receipt_path, receipt)
        partial = CACHE / (shard + ".part")
        owns_partial = False
        try:
            before = resource.probe()  # also enforces the pinned helper GPU UUID
            receipt["gpu_before"] = before
            if before.get("used_mib", 65) > 64 or before.get("compute_processes") != []:
                raise RuntimeError("GPU is not idle under the shared lock")
            write_receipt(receipt_path, receipt)
            if not CACHE.exists() and not CACHE.is_symlink():
                CACHE.mkdir(mode=0o700, exist_ok=False)
            require_private_directory(CACHE)
            # A new marker may adopt only already pinned and verified shards.
            initialize_cache_marker(CACHE, manifest, manifest_sha)
            receipt["cache_identity_marker_sha256"] = validate_cache_marker(CACHE, manifest_sha)
            receipt["cache_physical_bytes_before"] = _cache_bytes(CACHE, by_name)
            target = CACHE / shard
            if target.exists() or target.is_symlink():
                verified = verify_pinned_file(target, item)
                receipt.update(status="ALREADY_VERIFIED", downloaded_new_bytes=0,
                               sha256=verified["sha256"],
                               cache_physical_bytes_after=receipt["cache_physical_bytes_before"])
            else:
                if partial.exists() or partial.is_symlink():
                    raise RuntimeError("selected shard has an existing partial file")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("shard preparation exceeded 1200-second deadline")
                fs = os.statvfs(CACHE)
                if fs.f_bavail * fs.f_frsize < item["size"] + GIB:
                    raise RuntimeError("insufficient /dev/shm space for chosen shard and reserve")
                if memory_headroom() < item["size"] + 26 * GIB:
                    raise RuntimeError("insufficient container headroom for chosen shard and later engine")
                next_progress = PROGRESS_BYTES

                def on_chunk(count):
                    nonlocal next_progress
                    receipt["received_bytes"] += count
                    if receipt["received_bytes"] >= next_progress:
                        while receipt["received_bytes"] >= next_progress:
                            next_progress += PROGRESS_BYTES
                        if memory_headroom() < 26 * GIB:
                            raise RuntimeError("container headroom fell below 26 GiB during cache fill")
                        print(json.dumps(dict(stage="shard_download", shard=shard,
                                              received_bytes=receipt["received_bytes"],
                                              expected_bytes=item["size"])), flush=True)
                        write_receipt(receipt_path, receipt)

                def on_attempt(range_start, _range_end, attempt):
                    receipt["range_attempts_total"] += 1
                    if attempt > 1:
                        receipt["range_retries_started"] += 1
                    receipt["range_attempts"][str(range_start)] = attempt

                owns_partial = True
                result = download_ranged(item["url"], partial, item["size"],
                                         item["sha256"], deadline, on_chunk,
                                         attempt_callback=on_attempt)
                # link() cannot overwrite an existing verified cache shard. Both
                # names briefly share one inode, so this adds no second data copy.
                os.link(partial, target, follow_symlinks=False)
                pinfo, tinfo = partial.lstat(), target.lstat()
                if (pinfo.st_dev, pinfo.st_ino) != (tinfo.st_dev, tinfo.st_ino):
                    raise RuntimeError("published shard differs from verified partial inode")
                partial.unlink()
                owns_partial = False
                receipt.update(status="VERIFIED", downloaded_new_bytes=item["size"],
                               sha256=result["sha256"], range_count=result["range_count"],
                               cache_physical_bytes_after=receipt["cache_physical_bytes_before"] + item["size"])
            after = resource.probe()
            receipt["gpu_after"] = after
            if after.get("used_mib", 65) > 64 or after.get("compute_processes") != []:
                raise RuntimeError("GPU became busy during CPU cache preparation")
            receipt.update(ended_utc=stamp(), elapsed_s=time.monotonic() - start,
                           memory_headroom_after=memory_headroom())
            write_receipt(receipt_path, receipt)
            print(json.dumps(dict(status=receipt["status"], shard=shard,
                                  cache_physical_bytes=receipt["cache_physical_bytes_after"])), flush=True)
            return 0
        except BaseException as exc:
            # download_ranged has reaped all workers before raising. Discard only
            # this invocation's incomplete bytes; retain every verified shard.
            if owns_partial and partial.exists() and not partial.is_symlink():
                partial.unlink()
                receipt["owned_partial_cleanup"] = "REMOVED"
            if "gpu_before" in receipt and "gpu_after" not in receipt:
                try:
                    receipt["gpu_after"] = resource.probe()
                except Exception as probe_exc:
                    receipt["gpu_after_error"] = f"{type(probe_exc).__name__}: {str(probe_exc)[:200]}"
            receipt.update(status="FAILED", ended_utc=stamp(), elapsed_s=time.monotonic() - start,
                           error=f"{type(exc).__name__}: {str(exc)[:500]}")
            write_receipt(receipt_path, receipt)
            raise
    finally:
        os.close(lock_fd)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", required=True, choices=WEIGHT_NAMES)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    return prepare_one_shard(args.shard, args.output, args.manifest)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"CACHE_SHARD_PREP_ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(70)
