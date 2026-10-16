#!/usr/bin/env python3
"""Losslessly archive only r07 performance/energy JSON; never alter its results."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shlex
import shutil
import tarfile
import time

RUN = "olmoe_bf16_matrix_r07"
MANIFEST_SHA = "9d4c60a13d61a452071c3c11f2eb14fa603146c2d3aa2299d40630ae229b041a"
ARCHIVE = "archived_performance.tgz"


def digest(stream):
    h = hashlib.sha256()
    for part in iter(lambda: stream.read(8 * 2**20), b""):
        h.update(part)
    return h.hexdigest()


def file_sha(path):
    with path.open("rb") as stream:
        return digest(stream)


def stamp(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Expected a regular non-symlink file: {path}")
    st = path.stat()
    return (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns)


def expected_members(run):
    if file_sha(run / "SHA256.json") != MANIFEST_SHA:
        raise ValueError("r07 original SHA256.json changed")
    hashes = json.loads((run / "SHA256.json").read_text())
    for name in ("status.json", "bench/status.json"):
        if file_sha(run / name) != hashes[name]:
            raise ValueError(f"r07 terminal state changed: {name}")
    status = json.loads((run / "status.json").read_text())
    bench = json.loads((run / "bench/status.json").read_text())
    if status["state"] != "COMPLETE" or status["exit_code"] != 0 or bench["status"] != "COMPLETE" or bench["completed_cells"] != 63:
        raise ValueError("r07 must remain the completed 63-cell run")
    members = {name: sha for name, sha in hashes.items()
               if PurePosixPath(name).name in ("performance.json", "energy.json")}
    if len(members) != 189 or any(PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts or not name.startswith("bench/") for name in members):
        raise ValueError("Unexpected r07 archive member set")
    return dict(sorted(members.items())), status


def verify_archive(path, expected):
    before = stamp(path)
    rows, seen = [], set()
    with tarfile.open(path, "r:gz") as archive:
        for member in archive:
            if not member.isfile() or member.name not in expected or member.name in seen:
                raise ValueError(f"Unexpected, non-regular or duplicate archive member: {member.name}")
            with archive.extractfile(member) as stream:
                sha = digest(stream)
            if sha != expected[member.name]:
                raise ValueError(f"Archive member SHA256 mismatch: {member.name}")
            rows.append({"path": member.name, "sha256": sha, "size": member.size})
            seen.add(member.name)
    if seen != set(expected) or stamp(path) != before:
        raise ValueError("Archive is incomplete or changed during verification")
    return rows


def verify_originals(run, expected, allow_missing=False):
    stamps = {}
    for name, sha in expected.items():
        path = run / name
        if allow_missing and not path.exists() and not path.is_symlink():
            continue  # A prior interrupted compact may already have removed it.
        before = stamp(path)
        if file_sha(path) != sha or stamp(path) != before:
            raise ValueError(f"Original differs from archived pinned bytes: {name}")
        stamps[name] = before
    return stamps


def require_idle(root, status):
    if not Path("/proc/self/stat").exists():
        raise RuntimeError("--compact requires Linux /proc; --prepare is portable")
    for key in ("controller_pid", "worker_pid"):
        if Path(f"/proc/{int(status[key])}").exists():
            raise RuntimeError(f"r07 {key} still exists; PID reuse is conservatively refused")
    boot = next(int(line.split()[1]) for line in Path("/proc/stat").read_text().splitlines() if line.startswith("btime "))
    ticks = os.sysconf("SC_CLK_TCK")
    for path in (root / "runs").glob("*/status.json"):
        row = json.loads(path.read_text())
        if row.get("state") in ("PREPARING", "WAITING", "RUNNING") or row.get("status") in ("PREPARING", "WAITING", "RUNNING"):
            raise RuntimeError(f"K still has an active run: {path.parent.name}")
        for key in ("controller_pid", "worker_pid"):
            if not row.get(key):
                continue
            try:
                parts = Path(f"/proc/{int(row[key])}/stat").read_text().rsplit(")", 1)[1].split()
            except FileNotFoundError:
                continue
            started = boot + int(parts[19]) / ticks
            if started <= row.get("ended_unix", float("inf")) + 1:
                raise RuntimeError(f"K process has not exited: {path.parent.name} {key}")


def prepare(run, archive, expected):
    if archive.exists() or archive.is_symlink():
        raise FileExistsError(archive)
    stamps = verify_originals(run, expected)
    archive.parent.mkdir(parents=True, exist_ok=True)
    temporary = archive.with_name(archive.name + ".tmp")
    with temporary.open("xb") as output:
        with tarfile.open(fileobj=output, mode="w:gz", compresslevel=6) as bundle:
            for name in expected:
                if stamp(run / name) != stamps[name]:
                    raise RuntimeError(f"Original changed: {name}")
                bundle.add(run / name, arcname=name, recursive=False)
        output.flush()
        os.fsync(output.fileno())
    rows = verify_archive(temporary, expected)
    temporary.rename(archive)
    print(json.dumps({"status": "PREPARED", "archive": str(archive), "archive_sha256": file_sha(archive),
                      "members": len(rows), "original_bytes": sum(row["size"] for row in rows),
                      "archive_bytes": archive.stat().st_size,
                      "original_manifest_sha256": MANIFEST_SHA}))


def compact(root, run, uploaded, expected, status):
    require_idle(root, status)
    rows = verify_archive(uploaded, expected)
    uploaded_sha = file_sha(uploaded)
    destination = run / ARCHIVE
    storage = run / "storage.json"
    old = json.loads(storage.read_text()) if storage.exists() else None
    if old and (old.get("archive_sha256") != uploaded_sha or old.get("original_manifest_sha256") != MANIFEST_SHA):
        raise ValueError("Existing storage receipt belongs to another archive")
    originals = verify_originals(run, expected, allow_missing=old is not None)
    if destination.exists() or destination.is_symlink():
        stamp(destination)
        if file_sha(destination) != uploaded_sha:
            raise ValueError("Existing r07 archive differs; left untouched")
    else:
        temporary = destination.with_name(destination.name + ".tmp")
        with uploaded.open("rb") as source, temporary.open("xb") as target:
            shutil.copyfileobj(source, target, 8 * 2**20)
            target.flush()
            os.fsync(target.fileno())
        if file_sha(temporary) != uploaded_sha:
            raise ValueError("Archive transfer changed bytes")
        verify_archive(temporary, expected)
        temporary.rename(destination)
    require_idle(root, status)
    receipt = {"status": "ARCHIVE_VERIFIED", "archive": ARCHIVE, "archive_sha256": uploaded_sha,
               "original_manifest_sha256": MANIFEST_SHA, "members": rows,
               "original_bytes": sum(row["size"] for row in rows), "archive_bytes": destination.stat().st_size,
               "restore_command": f"tar -xzf {shlex.quote(str(destination))} -C {shlex.quote(str(run))}",
               "scope": "Lossless storage compaction only; original SHA256.json and all other results retained",
               "started_unix": time.time()}
    def save():
        temporary = storage.with_name(storage.name + ".tmp")
        with temporary.open("w") as handle:
            handle.write(json.dumps(receipt, indent=2) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(storage)
        descriptor = os.open(run, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    save()  # Archive + receipt durable before the first unlink.
    try:
        for name, before in originals.items():
            path = run / name
            if stamp(path) != before:
                raise RuntimeError(f"Original changed before removal: {name}")
            path.unlink()
        receipt.update(status="COMPACTED", ended_unix=time.time(),
                       removed_expanded_bytes=sum(value[2] for value in originals.values()))
    except BaseException as error:
        receipt.update(status="PARTIAL", error=f"{type(error).__name__}: {error}", ended_unix=time.time())
        raise
    finally:
        save()
    print(json.dumps({key: receipt[key] for key in ("status", "archive_sha256", "original_bytes", "archive_bytes", "removed_expanded_bytes")}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--compact", action="store_true")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--archive", type=Path, default=Path(__file__).resolve().parent / ARCHIVE)
    args = parser.parse_args()
    root, archive = args.root.resolve(), args.archive.resolve()
    run = root / "runs" / RUN
    expected, status = expected_members(run)
    if args.prepare:
        prepare(run, archive, expected)
    else:
        compact(root, run, archive, expected, status)


if __name__ == "__main__":
    main()
