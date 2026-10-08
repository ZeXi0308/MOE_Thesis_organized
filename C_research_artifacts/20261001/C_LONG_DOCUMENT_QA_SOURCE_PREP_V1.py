#!/usr/bin/env python3
"""One bounded official LongBench data fetch under the spare's existing lock."""
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time
import urllib.request
import zipfile

import C_SPARE_RESOURCE_V1 as resource
from C_INSTRUCT_SPARE_SHARD_PREP_V1 import _lock_shared_file

REV = "5e628be450b7e67fb7ae6e201bd6d8f7056f7672"
URL = f"https://huggingface.co/datasets/zai-org/LongBench/resolve/{REV}/data.zip?download=true"
ZIP_SHA = "cb45b11a4133c6bc1d6a44b0f8e701335ff1e543195db1103472e575857f7f64"
MEMBER = "data/multifieldqa_en.jsonl"
ROOT = resource.BASE / "c-longbench-source-20261001-v1"
LIMIT = 512 * 1024**2


def deadline(_signum, _frame):
    raise TimeoutError("180-second source preparation deadline")


def interrupted(signum, _frame):
    raise InterruptedError(f"source preparation interrupted by signal {signum}")


def main():
    if ROOT.exists():
        raise RuntimeError("source preparation root already exists; inspect rather than rerun")
    try:
        fd = _lock_shared_file()
    except BlockingIOError:
        print(json.dumps(dict(status="GPU_DEFERRED", reason="shared lock busy")), flush=True)
        return 75
    start = time.monotonic()
    report = dict(status="PREPARING", url=URL, revision=REV, expected_zip_sha256=ZIP_SHA,
                  member=MEMBER, started_unix_s=time.time(), deadline_seconds=180,
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    try:
        ROOT.mkdir(mode=0o700)
        def save():
            (ROOT / "source-prepare.json").write_text(json.dumps(report, indent=2) + "\n")
        save()
        signal.signal(signal.SIGALRM, deadline)
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(sig, interrupted)
        signal.alarm(180)
        report["gpu_before"] = resource.probe()
        if report["gpu_before"]["used_mib"] > 64 or report["gpu_before"]["compute_processes"]:
            raise RuntimeError("GPU not idle under shared lock")
        req = urllib.request.Request(URL, headers={"Accept-Encoding": "identity"})
        total = 0
        digest = hashlib.sha256()
        with urllib.request.urlopen(req, timeout=20) as response:
            if response.status != 200 or response.headers.get("Content-Encoding", "identity") != "identity":
                raise RuntimeError("unexpected data response")
            declared = response.headers.get("Content-Length")
            declared = int(declared) if declared is not None else None
            if declared is None or not 0 < declared <= LIMIT:
                raise RuntimeError("missing or excessive archive size")
            fs = os.statvfs(ROOT)
            if fs.f_bavail * fs.f_frsize < declared + 100 * 1024**2:
                raise RuntimeError("insufficient private source disk headroom")
            report["declared_bytes"] = declared
            save()
            with (ROOT / "data.zip.part").open("xb") as stream:
                while True:
                    chunk = response.read(1024**2)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > declared:
                        raise RuntimeError("archive exceeds declared length")
                    stream.write(chunk)
                    digest.update(chunk)
        report.update(received_bytes=total, zip_sha256=digest.hexdigest())
        if total != declared or digest.hexdigest() != ZIP_SHA:
            raise RuntimeError("official pinned archive size/hash mismatch")
        (ROOT / "data.zip.part").rename(ROOT / "data.zip")
        with zipfile.ZipFile(ROOT / "data.zip") as archive:
            info = archive.getinfo(MEMBER)
            if info.file_size > 32 * 1024**2:
                raise RuntimeError("unexpected task-member size")
            content = archive.read(info)
        rows = [json.loads(line) for line in content.splitlines()]
        if len(rows) != 150 or any(not all(k in r for k in ("context", "input", "answers", "_id")) for r in rows):
            raise RuntimeError("official task inventory/schema differs")
        (ROOT / "multifieldqa_en.jsonl").write_bytes(content)
        report.update(status="COMPLETE", rows=len(rows), task_bytes=len(content),
                      task_sha256=hashlib.sha256(content).hexdigest())
        returncode = 0
    except BaseException as exc:
        report.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
        part = ROOT / "data.zip.part"
        if part.is_file():
            part.unlink()
            report["owned_partial_cleanup"] = "REMOVED"
        returncode = 70
    finally:
        signal.alarm(0)
        try:
            report["gpu_after"] = resource.probe()
            if report["gpu_after"]["used_mib"] > 64 or report["gpu_after"]["compute_processes"]:
                report.update(status="FAILED", error="GPU not idle at source preparation end")
                returncode = 70
        except Exception as exc:
            report["gpu_after_error"] = str(exc)
            report["status"] = "FAILED"
            returncode = 70
        report.update(ended_unix_s=time.time(), elapsed_s=time.monotonic() - start)
        if ROOT.is_dir():
            (ROOT / "source-prepare.json").write_text(json.dumps(report, indent=2) + "\n")
        os.close(fd)
    print(json.dumps(report), flush=True)
    return returncode


if __name__ == "__main__":
    sys.exit(main())
