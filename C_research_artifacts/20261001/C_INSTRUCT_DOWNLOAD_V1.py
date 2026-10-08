"""Bounded HTTPS model staging into a launcher-owned temporary RAM directory."""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import os
import time
import urllib.request

MODEL_ID = "allenai/OLMoE-1B-7B-0924-Instruct"
REVISION = "7f1c97f440f06ce36705e4f2b843edb5925f4498"
MAX_SECONDS = 1200
GIB = 1024**3


def stamp():
    return datetime.now(timezone.utc).isoformat()


def memory_headroom():
    root = Path("/sys/fs/cgroup")
    limit = (root / "memory.max").read_text().strip()
    current = int((root / "memory.current").read_text())
    if limit == "max":
        raise RuntimeError("expected the explicit container memory limit")
    return int(limit) - current


def write_receipt(path, receipt):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    os.replace(temp, path)


def prepare_model(model_dir, model_manifest, output):
    model_dir, model_manifest, output = map(Path, (model_dir, model_manifest, output))
    manifest = json.loads(model_manifest.read_text())
    if manifest["model_id"] != MODEL_ID or manifest["revision"] != REVISION:
        raise RuntimeError("model identity differs from the frozen Instruct checkpoint")
    if model_dir.is_symlink() or not model_dir.is_dir() or any(model_dir.iterdir()):
        raise RuntimeError("staging directory must be launcher-owned, empty and not a symlink")
    files = manifest["files"]
    names = [item["filename"] for item in files]
    if len(names) != len(set(names)) or any(Path(n).name != n or n in (".", "..") for n in names):
        raise RuntimeError("model manifest contains invalid or duplicate filenames")
    total = sum(item["size"] for item in files)
    if not 10 * GIB < total < 20 * GIB:
        raise RuntimeError("unexpected total checkpoint size")
    stat = os.statvfs(model_dir)
    if stat.f_bavail * stat.f_frsize < total + GIB or memory_headroom() < total + 26 * GIB:
        raise RuntimeError("insufficient RAM staging or container headroom")
    receipt = dict(status="DOWNLOADING", started_utc=stamp(), model_id=MODEL_ID,
        revision=REVISION, total_expected_bytes=total, files=[],
        model_manifest_sha256=hashlib.sha256(model_manifest.read_bytes()).hexdigest(),
        memory_headroom_before=memory_headroom(), deadline_seconds=MAX_SECONDS)
    path = output / "model-download.json"
    start = time.monotonic()
    try:
        for item in files:
            if time.monotonic() - start >= MAX_SECONDS:
                raise TimeoutError("model download exceeded its fixed deadline")
            name, expected, url = item["filename"], item["size"], item["url"]
            prefix = f"https://hf-mirror.com/{MODEL_ID}/resolve/{REVISION}/"
            if url != prefix + name:
                raise RuntimeError("only the frozen public model file URLs are accepted")
            target, partial = model_dir / name, model_dir / (name + ".part")
            digest, received, checkpoint = hashlib.sha256(), 0, 0
            record = dict(filename=name, expected_bytes=expected, expected_sha256=item["sha256"],
                          public_url=url, status="DOWNLOADING", received_bytes=0)
            receipt["files"].append(record)
            write_receipt(path, receipt)
            is_metadata = item.get("source") == "frozen_metadata"
            if is_metadata:
                if name.endswith(".safetensors"):
                    raise RuntimeError("weights must use the pinned HTTPS source")
                source = model_manifest.parent / "20261001_c_instruct_model_metadata_v1" / name
                response_context = source.open("rb")
                record["source"] = "fixed-revision Git metadata, copied from frozen local files"
            else:
                request = urllib.request.Request(url, headers={"User-Agent": "C-research-model-qualification/1"})
                response_context = urllib.request.urlopen(request, timeout=30)
            with response_context as response, partial.open("xb") as stream:
                if not is_metadata and not response.geturl().startswith("https://"):
                    raise RuntimeError("model download redirected away from HTTPS")
                while True:
                    if time.monotonic() - start >= MAX_SECONDS:
                        raise TimeoutError("model download exceeded its fixed deadline")
                    chunk = response.read(4 * 1024**2)
                    if not chunk:
                        break
                    received += len(chunk)
                    if received > expected:
                        raise RuntimeError("model file exceeds its frozen byte count")
                    stream.write(chunk)
                    digest.update(chunk)
                    record["received_bytes"] = received
                    if received - checkpoint >= 256 * 1024**2:
                        checkpoint = received
                        if memory_headroom() < 20 * GIB:
                            raise RuntimeError("container RAM reserve fell below20GiB during staging")
                        print(json.dumps(dict(stage="download", filename=name,
                            received_bytes=received, expected_bytes=expected)), flush=True)
                        write_receipt(path, receipt)
            record["sha256"] = digest.hexdigest()
            if received != expected or record["sha256"] != item["sha256"]:
                raise RuntimeError("downloaded model file failed size/SHA validation: " + name)
            os.replace(partial, target)
            record["status"] = "VERIFIED"
            write_receipt(path, receipt)
            print(json.dumps(dict(stage="file_verified", filename=name, bytes=received)), flush=True)
        receipt.update(status="VERIFIED", ended_utc=stamp(),
            elapsed_s=time.monotonic() - start, memory_headroom_after=memory_headroom())
        write_receipt(path, receipt)
        return receipt
    except BaseException as exc:
        receipt.update(status="FAILED", ended_utc=stamp(), elapsed_s=time.monotonic() - start,
            error=f"{type(exc).__name__}: {str(exc)[:500]}")
        write_receipt(path, receipt)
        raise
