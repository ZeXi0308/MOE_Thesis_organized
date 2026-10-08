"""Bounded, pinned model staging with parallel HTTP ranges for weight shards.

The launcher owns an empty destination directory. This module never starts a
GPU process. A range can be retried twice only for transient transport errors.
"""
from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
import hashlib
import http.client
import json
import os
from pathlib import Path
import re
import socket
import ssl
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

MODEL_ID = "allenai/OLMoE-1B-7B-0924-Instruct"
REVISION = "7f1c97f440f06ce36705e4f2b843edb5925f4498"
MAX_SECONDS = 1200
MIB = 1024**2
GIB = 1024**3
RANGE_BYTES = 32 * MIB
PROBE_BYTES = 8 * MIB
MAX_WORKERS = 8
READ_TIMEOUT = 30
MAX_ATTEMPTS = 3
IO_BYTES = 4 * MIB
PROGRESS_BYTES = 256 * MIB
_CONTENT_RANGE = re.compile(r"bytes (\d+)-(\d+)/(\d+)\Z")


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


def _remaining(deadline_monotonic):
    left = deadline_monotonic - time.monotonic()
    if left <= 0:
        raise TimeoutError("model download exceeded its fixed deadline")
    return left


def _checked_range(url, expected_size, start, end, on_chunk, deadline_monotonic,
                   cancelled=None, *, https_only=True):
    """Read one exact range, forwarding each verified-position chunk to on_chunk.

    The callback is called only after status and response headers pass.  Body
    byte count is checked before returning; a caller must hash the final file.
    """
    if not 0 <= start <= end < expected_size:
        raise ValueError("invalid requested byte range")
    if https_only and urllib.parse.urlsplit(url).scheme != "https":
        raise RuntimeError("model range URL must use HTTPS")
    if cancelled is not None and cancelled.is_set():
        raise RuntimeError("range cancelled before request")
    request = urllib.request.Request(url, headers={
        "User-Agent": "C-research-model-qualification/2",
        "Range": f"bytes={start}-{end}",
        "Accept-Encoding": "identity",
    })
    with urllib.request.urlopen(request, timeout=min(READ_TIMEOUT, _remaining(deadline_monotonic))) as response:
        if https_only and urllib.parse.urlsplit(response.geturl()).scheme != "https":
            raise RuntimeError("model range redirected away from HTTPS")
        if response.status != 206:
            raise RuntimeError(f"range response status {response.status}, expected 206")
        match = _CONTENT_RANGE.fullmatch(response.headers.get("Content-Range", "").strip())
        if match is None or tuple(map(int, match.groups())) != (start, end, expected_size):
            raise RuntimeError("range response Content-Range differs from requested frozen bytes")
        content_length = response.headers.get("Content-Length")
        if content_length is not None and content_length.strip() != str(end - start + 1):
            raise RuntimeError("range response Content-Length differs from requested byte count")
        encoding = response.headers.get("Content-Encoding", "identity").strip().lower()
        if encoding != "identity":
            raise RuntimeError("range response used unexpected content encoding")
        cursor = start
        while cursor <= end:
            if cancelled is not None and cancelled.is_set():
                raise RuntimeError("range cancelled")
            _remaining(deadline_monotonic)
            chunk = response.read(min(IO_BYTES, end - cursor + 1))
            if not chunk:
                raise RuntimeError("range response ended before requested byte count")
            if len(chunk) > end - cursor + 1:
                raise RuntimeError("range response exceeded requested byte count")
            on_chunk(cursor, chunk)
            cursor += len(chunk)
        _remaining(deadline_monotonic)
        if response.read(1):
            raise RuntimeError("range response has extra body bytes")
    return cursor - start


def resolve_https_range_url(url, expected_size, deadline_monotonic):
    """Follow one pinned public HEAD to its signed HTTPS object URL.

    The resulting URL is an ephemeral transport detail and must not be written
    to a public receipt.  Its HEAD-reported object size must match the pin.
    """
    if urllib.parse.urlsplit(url).scheme != "https":
        raise RuntimeError("model HEAD URL must use HTTPS")
    request = urllib.request.Request(url, method="HEAD", headers={
        "User-Agent": "C-research-model-qualification/2",
        "Accept-Encoding": "identity",
    })
    with urllib.request.urlopen(request, timeout=min(READ_TIMEOUT, _remaining(deadline_monotonic))) as response:
        resolved = response.geturl()
        if urllib.parse.urlsplit(resolved).scheme != "https":
            raise RuntimeError("model HEAD redirected away from HTTPS")
        if response.status != 200:
            raise RuntimeError(f"model HEAD status {response.status}, expected 200")
        if response.headers.get("Content-Length", "").strip() != str(expected_size):
            raise RuntimeError("model HEAD Content-Length differs from frozen size")
        return resolved


def _retryable_range_error(exc):
    if isinstance(exc, urllib.error.HTTPError):
        return 500 <= exc.code <= 599
    if isinstance(exc, (TimeoutError, socket.timeout, ConnectionError,
                        http.client.IncompleteRead)):
        return True
    if isinstance(exc, urllib.error.URLError):
        return not isinstance(exc.reason, ssl.SSLError)
    return False


def probe_range(url, expected_size, *, start=0, length=PROBE_BYTES,
                timeout=READ_TIMEOUT, https_only=True):
    """Bounded, read-only transport probe for the exact-range server contract.

    The result is the SHA256 of this range, not a whole-file integrity claim.
    An explicit ``https_only=False`` exists solely for local HTTP server tests.
    """
    if expected_size <= 0 or length <= 0 or start < 0 or start + length > expected_size:
        raise ValueError("invalid probe range")
    if timeout <= 0 or timeout > READ_TIMEOUT:
        raise ValueError("probe timeout must be in (0, 30] seconds")
    digest = hashlib.sha256()
    received = 0

    def accept(_position, chunk):
        nonlocal received
        digest.update(chunk)
        received += len(chunk)

    _checked_range(url, expected_size, start, start + length - 1, accept,
                   time.monotonic() + timeout, https_only=https_only)
    return dict(start=start, end=start + length - 1, total_bytes=expected_size,
                received_bytes=received, sha256=digest.hexdigest(), status="RANGE_VERIFIED")


def _pwrite_all(fd, position, data):
    view = memoryview(data)
    done = 0
    while done < len(view):
        written = os.pwrite(fd, view[done:], position + done)
        if written <= 0:
            raise OSError("pwrite made no progress")
        done += written


def _sha256_file(path, deadline_monotonic):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while True:
            _remaining(deadline_monotonic)
            chunk = stream.read(IO_BYTES)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def download_ranged(url, partial, expected_size, expected_sha256,
                    deadline_monotonic, progress_callback=None, *,
                    range_bytes=RANGE_BYTES, workers=MAX_WORKERS, https_only=True,
                    resolve_head=True, attempt_callback=None):
    """Download one frozen file into a new .part path and verify its whole SHA.

    All scheduled ranges are disjoint. Each failed range gets at most two
    retries for transient transport errors (three attempts total); bad status,
    range metadata, or final hash fails immediately. On failure, pending
    futures are cancelled and active readers see a cancellation flag between
    reads. Each network operation has a 30 s socket timeout. Executor shutdown
    waits for them before the descriptor is closed; no worker survives a return.
    """
    partial = Path(partial)
    if expected_size <= 0 or range_bytes <= 0 or not 1 <= workers <= MAX_WORKERS:
        raise ValueError("invalid range download geometry")
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ValueError("invalid frozen SHA256")
    _remaining(deadline_monotonic)
    if resolve_head:
        if not https_only:
            raise ValueError("HEAD resolution requires HTTPS")
        range_url = resolve_https_range_url(url, expected_size, deadline_monotonic)
    else:
        range_url = url
    ranges = [(start, min(start + range_bytes, expected_size) - 1)
              for start in range(0, expected_size, range_bytes)]
    cancelled = threading.Event()
    fd = os.open(partial, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
    received = 0
    attempts_total = 0
    retries_started = 0
    received_lock = threading.Lock()

    def run_one(start, end):
        def accept(position, chunk):
            _pwrite_all(fd, position, chunk)

        nonlocal received, attempts_total, retries_started
        for attempt in range(1, MAX_ATTEMPTS + 1):
            if cancelled.is_set():
                raise RuntimeError("range cancelled")
            _remaining(deadline_monotonic)
            with received_lock:
                attempts_total += 1
                if attempt > 1:
                    retries_started += 1
                if attempt_callback is not None:
                    attempt_callback(start, end, attempt)
            try:
                count = _checked_range(range_url, expected_size, start, end, accept,
                                       deadline_monotonic, cancelled, https_only=https_only)
            except BaseException as exc:
                if attempt == MAX_ATTEMPTS or not _retryable_range_error(exc):
                    raise
                continue
            # A partially written failed attempt is overwritten by its retry.
            # Credit progress only when the complete range has passed all gates.
            with received_lock:
                received += count
                if progress_callback is not None:
                    progress_callback(count)
            return count
        raise AssertionError("unreachable range attempt state")

    try:
        os.ftruncate(fd, expected_size)
        executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="model-range")
        pending = set()
        try:
            pending = {executor.submit(run_one, start, end) for start, end in ranges}
            while pending:
                done, pending = wait(pending, timeout=min(READ_TIMEOUT, _remaining(deadline_monotonic)),
                                     return_when=FIRST_COMPLETED)
                for future in done:
                    future.result()
        except BaseException:
            cancelled.set()
            for future in pending:
                future.cancel()
            raise
        finally:
            # A reader may still be inside a socket read; that read is bounded
            # by READ_TIMEOUT.  Waiting here prevents leaked background writers.
            executor.shutdown(wait=True, cancel_futures=True)
    finally:
        os.close(fd)
    if received != expected_size or partial.stat().st_size != expected_size:
        raise RuntimeError("range assembly failed frozen byte-count validation")
    actual_sha256 = _sha256_file(partial, deadline_monotonic)
    if actual_sha256 != expected_sha256:
        raise RuntimeError("range assembly failed frozen whole-file SHA256 validation")
    return dict(received_bytes=received, sha256=actual_sha256, range_count=len(ranges),
                range_bytes=range_bytes, workers=workers, attempts_total=attempts_total,
                retries_started=retries_started)


def _copy_frozen_metadata(source, partial, expected, deadline_monotonic, on_chunk):
    digest = hashlib.sha256()
    received = 0
    with source.open("rb") as input_stream, partial.open("xb") as output_stream:
        while True:
            _remaining(deadline_monotonic)
            chunk = input_stream.read(IO_BYTES)
            if not chunk:
                break
            received += len(chunk)
            if received > expected:
                raise RuntimeError("frozen metadata exceeds pinned byte count")
            output_stream.write(chunk)
            digest.update(chunk)
            on_chunk(len(chunk))
    return received, digest.hexdigest()


def prepare_model(model_dir, model_manifest, output):
    model_dir, model_manifest, output = map(Path, (model_dir, model_manifest, output))
    manifest = json.loads(model_manifest.read_text())
    if manifest["model_id"] != MODEL_ID or manifest["revision"] != REVISION:
        raise RuntimeError("model identity differs from the frozen Instruct checkpoint")
    if model_dir.is_symlink() or not model_dir.is_dir() or any(model_dir.iterdir()):
        raise RuntimeError("staging directory must be launcher-owned, empty and not a symlink")
    if not output.is_dir():
        raise RuntimeError("receipt output directory must exist")
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
                   revision=REVISION, total_expected_bytes=total, total_received_bytes=0,
                   files=[], model_manifest_sha256=hashlib.sha256(model_manifest.read_bytes()).hexdigest(),
                   memory_headroom_before=memory_headroom(), deadline_seconds=MAX_SECONDS,
                   weight_transport="parallel_exact_http_range_v2")
    path = output / "model-download.json"
    start = time.monotonic()
    deadline = start + MAX_SECONDS
    next_progress = PROGRESS_BYTES
    current_record = None

    def on_chunk(count):
        nonlocal next_progress
        current_record["received_bytes"] += count
        receipt["total_received_bytes"] += count
        if receipt["total_received_bytes"] >= next_progress:
            while receipt["total_received_bytes"] >= next_progress:
                next_progress += PROGRESS_BYTES
            if memory_headroom() < 20 * GIB:
                raise RuntimeError("container RAM reserve fell below 20 GiB during staging")
            print(json.dumps(dict(stage="download", filename=current_record["filename"],
                                  received_bytes=receipt["total_received_bytes"],
                                  expected_bytes=total)), flush=True)
            write_receipt(path, receipt)

    try:
        write_receipt(path, receipt)
        for item in files:
            _remaining(deadline)
            name, expected, url = item["filename"], item["size"], item["url"]
            prefix = f"https://hf-mirror.com/{MODEL_ID}/resolve/{REVISION}/"
            if url != prefix + name:
                raise RuntimeError("only the frozen public model file URLs are accepted")
            if not isinstance(expected, int) or expected <= 0 or not re.fullmatch(r"[0-9a-f]{64}", item["sha256"]):
                raise RuntimeError("model manifest has incomplete frozen size or SHA256")
            target, partial = model_dir / name, model_dir / (name + ".part")
            record = dict(filename=name, expected_bytes=expected, expected_sha256=item["sha256"],
                          public_url=url, status="DOWNLOADING", received_bytes=0)
            receipt["files"].append(record)
            current_record = record
            write_receipt(path, receipt)
            if item.get("source") == "frozen_metadata":
                if name.endswith(".safetensors"):
                    raise RuntimeError("weights must use the pinned HTTPS source")
                source = model_manifest.parent / "20261001_c_instruct_model_metadata_v1" / name
                record["source"] = "fixed-revision Git metadata, copied from frozen local files"
                received, actual_sha256 = _copy_frozen_metadata(source, partial, expected,
                                                                  deadline, on_chunk)
            else:
                if not name.endswith(".safetensors"):
                    raise RuntimeError("non-weight files must come from frozen local metadata")
                record.update(source="frozen HTTPS exact ranges", range_bytes=RANGE_BYTES,
                              worker_count=MAX_WORKERS, max_attempts_per_range=MAX_ATTEMPTS,
                              range_attempts_total=0, range_retries_started=0,
                              range_attempts={})

                def on_attempt(range_start, _range_end, attempt):
                    record["range_attempts_total"] += 1
                    if attempt > 1:
                        record["range_retries_started"] += 1
                    record["range_attempts"][str(range_start)] = attempt

                result = download_ranged(url, partial, expected, item["sha256"], deadline,
                                         on_chunk, attempt_callback=on_attempt)
                received, actual_sha256 = result["received_bytes"], result["sha256"]
                record["range_count"] = result["range_count"]
            record["sha256"] = actual_sha256
            if received != expected or actual_sha256 != item["sha256"]:
                raise RuntimeError("downloaded model file failed size/SHA validation: " + name)
            os.replace(partial, target)
            record["status"] = "VERIFIED"
            write_receipt(path, receipt)
            print(json.dumps(dict(stage="file_verified", filename=name, bytes=received)), flush=True)
        receipt.update(status="VERIFIED", ended_utc=stamp(), elapsed_s=time.monotonic() - start,
                       memory_headroom_after=memory_headroom())
        write_receipt(path, receipt)
        return receipt
    except BaseException as exc:
        if current_record is not None and current_record["status"] == "DOWNLOADING":
            current_record["status"] = "FAILED"
        receipt.update(status="FAILED", ended_utc=stamp(), elapsed_s=time.monotonic() - start,
                       error=f"{type(exc).__name__}: {str(exc)[:500]}")
        write_receipt(path, receipt)
        raise
