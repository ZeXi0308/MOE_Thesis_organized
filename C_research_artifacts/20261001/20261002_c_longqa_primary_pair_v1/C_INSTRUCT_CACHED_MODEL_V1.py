"""Verify a pinned persistent weight cache and hardlink it into one fresh stage.

The caller holds the existing nonblocking shared GPU lock for the entire call.
Weights remain charged once to the private /dev/shm cache. Metadata is copied
from fixed-revision Git files retained next to the manifest.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import time

from C_INSTRUCT_DOWNLOAD_V2 import GIB, MODEL_ID, REVISION, memory_headroom, write_receipt

CACHE = Path("/dev/shm/c-instruct-verified-cache-20261001")
MARKER_NAME = "cache-identity.json"
METADATA_DIR_NAME = "20261001_c_instruct_model_metadata_v1"
WEIGHT_NAMES = tuple(f"model-{i:05d}-of-00003.safetensors" for i in (1, 2, 3))
EXPECTED_NAMES = set(WEIGHT_NAMES) | {
    "config.json", "generation_config.json", "model.safetensors.index.json",
    "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json",
}
IO_BYTES = 4 * 1024**2


def stamp():
    return datetime.now(timezone.utc).isoformat()


def manifest_info(model_manifest):
    model_manifest = Path(model_manifest)
    manifest_bytes = model_manifest.read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get("model_id") != MODEL_ID or manifest.get("revision") != REVISION:
        raise RuntimeError("model identity differs from frozen Instruct checkpoint")
    files = manifest.get("files")
    if not isinstance(files, list) or len(files) != len(EXPECTED_NAMES):
        raise RuntimeError("model manifest file set differs from frozen checkpoint")
    names = [item.get("filename") for item in files]
    if len(set(names)) != len(names) or set(names) != EXPECTED_NAMES:
        raise RuntimeError("model manifest has missing, extra, or duplicate files")
    prefix = f"https://hf-mirror.com/{MODEL_ID}/resolve/{REVISION}/"
    for item in files:
        name = item["filename"]
        if Path(name).name != name or name in (".", ".."):
            raise RuntimeError("model manifest contains unsafe filename")
        if item.get("url") != prefix + name:
            raise RuntimeError("model manifest URL differs from frozen public path")
        if type(item.get("size")) is not int or item["size"] <= 0:
            raise RuntimeError("model manifest has incomplete byte count")
        if not isinstance(item.get("sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", item["sha256"]):
            raise RuntimeError("model manifest has incomplete SHA256")
        if name in WEIGHT_NAMES and item.get("source") is not None:
            raise RuntimeError("weight shard must come from pinned HTTPS cache")
        if name not in WEIGHT_NAMES and item.get("source") != "frozen_metadata":
            raise RuntimeError("metadata must come from fixed-revision Git files")
    return manifest, hashlib.sha256(manifest_bytes).hexdigest()


def require_private_directory(path):
    path = Path(path)
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode) or path.is_symlink():
        raise RuntimeError("expected a real cache/stage directory")
    if stat.S_IMODE(info.st_mode) & 0o077:
        raise RuntimeError("cache/stage directory must be private (0700)")
    return info


def verify_pinned_file(path, item):
    """Hash a regular, non-symlink file and return its verified inode identity."""
    path = Path(path)
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_size != item["size"]:
        raise RuntimeError("pinned file is missing, unsafe, or wrong size: " + item["filename"])
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        opened = os.fstat(fd)
        if (not stat.S_ISREG(opened.st_mode) or opened.st_size != item["size"] or
                (opened.st_dev, opened.st_ino) != (info.st_dev, info.st_ino)):
            raise RuntimeError("pinned file changed while opening: " + item["filename"])
        digest = hashlib.sha256()
        while True:
            chunk = os.read(fd, IO_BYTES)
            if not chunk:
                break
            digest.update(chunk)
        if digest.hexdigest() != item["sha256"]:
            raise RuntimeError("pinned file SHA256 differs: " + item["filename"])
        after = os.fstat(fd)
        if after.st_size != item["size"] or (after.st_dev, after.st_ino) != (info.st_dev, info.st_ino):
            raise RuntimeError("pinned file changed while hashing: " + item["filename"])
        return dict(device=opened.st_dev, inode=opened.st_ino, size=opened.st_size,
                    sha256=digest.hexdigest())
    finally:
        os.close(fd)


def _marker_payload(manifest_sha256):
    return dict(model_id=MODEL_ID, revision=REVISION,
                model_manifest_sha256=manifest_sha256, owner="C",
                persistent_cache=True)


def validate_cache_marker(cache_dir, manifest_sha256):
    require_private_directory(cache_dir)
    path = Path(cache_dir) / MARKER_NAME
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_size > 4096:
        raise RuntimeError("cache identity marker is not a regular file")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        opened = os.fstat(fd)
        if not stat.S_ISREG(opened.st_mode) or opened.st_ino != info.st_ino or opened.st_dev != info.st_dev:
            raise RuntimeError("cache identity marker changed while opening")
        payload = json.loads(os.read(fd, 4097))
    finally:
        os.close(fd)
    if payload != _marker_payload(manifest_sha256):
        raise RuntimeError("cache identity marker differs from model/revision/manifest pin")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def initialize_cache_marker(cache_dir, manifest, manifest_sha256):
    """Create the identity marker after validating every preexisting cache shard.

    This permits a previously verified first shard to be retained when the
    transport plan changes. It never adopts an unknown file or partial.
    """
    require_private_directory(cache_dir)
    cache_dir = Path(cache_dir)
    marker = cache_dir / MARKER_NAME
    if marker.exists() or marker.is_symlink():
        return validate_cache_marker(cache_dir, manifest_sha256)
    by_name = {item["filename"]: item for item in manifest["files"]}
    for entry in cache_dir.iterdir():
        if entry.name not in WEIGHT_NAMES:
            raise RuntimeError("unknown or partial file in unmarked cache: " + entry.name)
        verify_pinned_file(entry, by_name[entry.name])
    payload = _marker_payload(manifest_sha256)
    fd = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        body = (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode()
        if os.write(fd, body) != len(body):
            raise OSError("short cache marker write")
        os.fsync(fd)
    finally:
        os.close(fd)
    return validate_cache_marker(cache_dir, manifest_sha256)


def prepare_model(model_dir, model_manifest, output, *, cache_dir=CACHE):
    """Same interface as V1/V2, with zero network and zero weight data copies."""
    model_dir, model_manifest, output = map(Path, (model_dir, model_manifest, output))
    cache_dir = Path(cache_dir)
    manifest, manifest_sha = manifest_info(model_manifest)
    require_private_directory(model_dir)
    if any(model_dir.iterdir()):
        raise RuntimeError("model stage must be launcher-owned and empty")
    if not output.is_dir():
        raise RuntimeError("receipt output directory must exist")
    marker_sha = validate_cache_marker(cache_dir, manifest_sha)
    if memory_headroom() < 26 * GIB:
        raise RuntimeError("less than 26 GiB container headroom for cached model and engine")
    metadata_bytes = sum(item["size"] for item in manifest["files"]
                         if item["filename"] not in WEIGHT_NAMES)
    fs = os.statvfs(model_dir)
    if fs.f_bavail * fs.f_frsize < metadata_bytes + GIB:
        raise RuntimeError("insufficient RAM filesystem space for metadata and reserve")
    receipt_path = output / "model-download.json"
    start = time.monotonic()
    receipt = dict(status="PREPARING", started_utc=stamp(), model_id=MODEL_ID,
                   revision=REVISION, model_manifest_sha256=manifest_sha,
                   cache_identity_marker_sha256=marker_sha, cache_path=str(cache_dir),
                   cache_physical_weight_bytes=0, stage_additional_weight_bytes=0,
                   stage_metadata_copied_bytes=0, files=[],
                   memory_headroom_before=memory_headroom(),
                   transport="verified_private_cache_hardlinks_v1")
    current = None
    try:
        write_receipt(receipt_path, receipt)
        for item in manifest["files"]:
            name = item["filename"]
            target = model_dir / name
            current = dict(filename=name, expected_bytes=item["size"],
                           expected_sha256=item["sha256"], status="PREPARING")
            receipt["files"].append(current)
            if name in WEIGHT_NAMES:
                source = cache_dir / name
                verified = verify_pinned_file(source, item)
                os.link(source, target, follow_symlinks=False)
                linked = target.lstat()
                if not stat.S_ISREG(linked.st_mode) or (
                    linked.st_dev, linked.st_ino
                ) != (verified["device"], verified["inode"]):
                    raise RuntimeError("stage hardlink does not share verified cache inode")
                current.update(source="verified_cache_hardlink", cache_device=verified["device"],
                               cache_inode=verified["inode"], stage_inode=linked.st_ino,
                               sha256=verified["sha256"], status="VERIFIED")
                receipt["cache_physical_weight_bytes"] += item["size"]
            else:
                source = model_manifest.parent / METADATA_DIR_NAME / name
                verify_pinned_file(source, item)
                fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
                try:
                    with source.open("rb") as input_stream:
                        while True:
                            chunk = input_stream.read(IO_BYTES)
                            if not chunk:
                                break
                            view = memoryview(chunk)
                            while view:
                                written = os.write(fd, view)
                                if written <= 0:
                                    raise OSError("short metadata write")
                                view = view[written:]
                    os.fsync(fd)
                finally:
                    os.close(fd)
                verify_pinned_file(target, item)
                current.update(source="fixed_revision_git_metadata_copy", sha256=item["sha256"],
                               status="VERIFIED")
                receipt["stage_metadata_copied_bytes"] += item["size"]
            write_receipt(receipt_path, receipt)
        receipt.update(status="VERIFIED", ended_utc=stamp(), elapsed_s=time.monotonic() - start,
                       memory_headroom_after=memory_headroom())
        write_receipt(receipt_path, receipt)
        return receipt
    except BaseException as exc:
        if current is not None and current["status"] == "PREPARING":
            current["status"] = "FAILED"
        receipt.update(status="FAILED", ended_utc=stamp(), elapsed_s=time.monotonic() - start,
                       error=f"{type(exc).__name__}: {str(exc)[:500]}")
        write_receipt(receipt_path, receipt)
        raise
