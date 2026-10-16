#!/usr/bin/env python3
"""Copy the completed 16-layer integration sidecars/configs to this workstation.

Default prints a local plan. --execute waits for the existing export run's
VERIFIED_LOCAL receipt and COMPLETE status, then uses the existing SSH master.
No GPU work, remote deletion, automatic retry, or manifest path rewriting.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import time


RUN = "olmoe_integration_export_r01"
DIRECTORIES = ("artifacts/olmoe_awq_integration_r01", "configs/olmoe_integration_r01")
REMOTE_ROOT = "/root/autodl-tmp/research_C_format_sched"

# Existing exporter/config SHA values cover all large payloads. Hash only other
# files on the remote CPU; no torch import and no GPU/NVML access.
REMOTE_INVENTORY = r'''
import hashlib,json,stat,sys
from pathlib import Path
root=Path(sys.argv[1]); dirs=json.loads(sys.argv[2]); expected={}; sources={}
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''): h.update(b)
    return h.hexdigest()
def relative(value, directory):
    p=Path(value)
    if not p.is_absolute(): p=root/p
    p=p.resolve(); p.relative_to((root/directory).resolve())
    return str(p.relative_to(root.resolve()))
def add(name, value, source):
    if not isinstance(value,str) or len(value)!=64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('invalid existing SHA256')
    if name in expected and expected[name]!=value: raise ValueError('conflicting SHA256')
    expected[name]=value; sources[name]=source
side=root/dirs[0]/'sidecar_manifest.json'; config=root/dirs[1]/'integration_manifest.json'
s=json.loads(side.read_text()); c=json.loads(config.read_text())
rows=s['artifacts']
if s.get('state')!='complete' or len(rows)!=32 or {(r['layer'],r['format']) for r in rows}!={(i,f) for i in range(16) for f in ('fp8','awq')}:
    raise ValueError('expected complete 16-layer FP8/AWQ export')
for row in rows:
    name=relative(row['artifact'],dirs[0])
    add(name,row['artifact_sha256'],'sidecar_manifest payload')
    add(name+'.json',row['metadata_sha256'],'sidecar_manifest metadata')
if c['sidecar_manifest_sha256']!=digest(side) or relative(c['sidecar_manifest'],dirs[0])!=str(side.relative_to(root)):
    raise ValueError('config manifest does not identify this sidecar manifest')
if set(c['configs'])!={'single_fp8','single_awq','dual_phase'}:
    raise ValueError('expected exactly three integration worker configs')
for row in c['configs'].values():
    name=relative(row['path'],dirs[1])
    add(name,row.get('sha256') or digest(root/name),
        'integration_manifest config' if row.get('sha256') else 'remote file bytes')
files={}
for name in dirs:
    directory=root/name
    if not directory.is_dir() or directory.is_symlink(): raise ValueError('missing/symlink source directory')
    for p in sorted(directory.rglob('*')):
        if p.is_symlink(): raise ValueError('source symlinks unsupported')
        if p.is_dir(): continue
        st=p.stat(); rel=str(p.relative_to(root))
        if not stat.S_ISREG(st.st_mode) or any(x in ('.rsync-partial','.rsync-backups') for x in p.parts):
            raise ValueError('nonregular/reserved source file')
        value=expected.get(rel) or digest(p)
        if (st.st_size,st.st_mtime_ns)!=(p.stat().st_size,p.stat().st_mtime_ns):
            raise ValueError('source changed during inventory')
        files[rel]={'bytes':st.st_size,'sha256':value,'source':sources.get(rel,'remote file bytes')}
if not set(expected)<=set(files): raise ValueError('manifest-referenced source file missing')
print(json.dumps({'payloads':32,'worker_configs':3,'files':files}))
'''


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def file_sha(path, deadline=None):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("copy/verification deadline reached")
            h.update(chunk)
    return h.hexdigest()


def local_gate(root):
    receipt = root / "runs" / (RUN + ".local_sync.json")
    status = root / "runs" / RUN / "status.json"
    saved = json.loads(receipt.read_text()) if receipt.exists() else {}
    run = json.loads(status.read_text()) if status.exists() else {}
    if run.get("state") == "FAILED" or saved.get("state") in ("VERIFICATION_FAILED", "WAIT_TIMEOUT", "COPY_OR_VERIFICATION_FAILED"):
        raise RuntimeError("export run or its existing local backup failed; no sidecar transfer")
    if saved.get("state") != "VERIFIED_LOCAL":
        return None
    if run.get("state") != "COMPLETE":
        raise RuntimeError("VERIFIED_LOCAL export receipt lacks a COMPLETE local run")
    archived = json.loads((status.parent / "SHA256.json").read_text())
    if archived.get("status.json") != file_sha(status):
        raise ValueError("local COMPLETE status differs from its verified run archive")
    return {"run_receipt_sha256": file_sha(receipt), "run_status_sha256": file_sha(status)}


def main():
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--wait-seconds", type=int, default=7200)
    p.add_argument("--copy-timeout", type=int, default=7200, help="total seconds for remote inventory, both copies and local SHA checks")
    p.add_argument("--poll-seconds", type=int, default=30)
    p.add_argument("--reserve-gib", type=float, default=20)
    p.add_argument("--control-path", default="/tmp/c-format-ssh.sock")
    args = p.parse_args()
    if args.wait_seconds < 1 or args.copy_timeout < 1 or not 1 <= args.poll_seconds <= 60 or not 20 <= args.reserve_gib < 10000:
        p.error("positive timeouts, 1..60 second polling, and at least 20 GiB reserve required")
    plan = {"state": "PLAN_NO_COPY", "run": RUN, "local_root": str(root), "remote_root": REMOTE_ROOT,
            "directories": list(DIRECTORIES), "wait_seconds": args.wait_seconds,
            "copy_and_verification_timeout_s": args.copy_timeout, "reserve_gib": args.reserve_gib,
            "gate": "existing local run receipt VERIFIED_LOCAL and archived status COMPLETE",
            "remote_deletion": False, "gpu_lock_or_gpu_work": False, "manifest_paths_rewritten": False}
    if not args.execute:
        print(json.dumps(plan, indent=2)); return
    def interrupted(signum, frame):
        raise InterruptedError(f"received signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    receipt = root / "runs" / (RUN + ".sidecars.local_sync.json")
    status = {**plan, "state": "WAITING_FOR_VERIFIED_LOCAL_EXPORT", "started_unix_s": time.time()}
    # Prevent two independently launched copies of this same derived payload.
    lock = receipt.with_suffix(".lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    write_json(receipt, status)
    try:
        deadline = time.monotonic() + args.wait_seconds
        while True:
            gate = local_gate(root)
            if gate: break
            if time.monotonic() >= deadline:
                raise TimeoutError("local export did not become verified and COMPLETE within wait limit")
            time.sleep(min(args.poll_seconds, max(0, deadline - time.monotonic())))
        status.update(gate=gate, state="READING_REMOTE_INVENTORY"); write_json(receipt, status)
        if shutil.which("rsync") is None or not Path(args.control_path).is_socket():
            raise RuntimeError("rsync and the existing authenticated SSH master socket are required")
        ssh = ["ssh", "-S", args.control_path, "-p", "50328", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]
        host = "root@connect.westb.seetacloud.com"
        subprocess.run([*ssh, "-O", "check", host], check=True, timeout=15)
        deadline = time.monotonic() + args.copy_timeout
        command = shlex.join(["/root/miniconda3/bin/python", "-", REMOTE_ROOT, json.dumps(DIRECTORIES)])
        result = subprocess.run([*ssh, host, command], input=REMOTE_INVENTORY, text=True,
                                capture_output=True, check=True, timeout=min(120, args.copy_timeout))
        inventory = json.loads(result.stdout)
        required = sum(row["bytes"] for row in inventory["files"].values())
        reserve = int(args.reserve_gib * 2**30)
        if shutil.disk_usage(root).free < required + reserve:
            raise RuntimeError("insufficient local space for the full copy/backup allowance plus 20 GiB reserve")
        status.update(files=inventory["files"], payloads=inventory["payloads"], worker_configs=inventory["worker_configs"],
                      source_bytes=required, state="COPYING"); write_json(receipt, status)
        attempt = str(time.time_ns())
        for directory in DIRECTORIES:
            target = root / directory
            if target.is_symlink() or not target.resolve().is_relative_to(root.resolve()):
                raise ValueError("destination directory escapes local project")
            target.mkdir(parents=True, exist_ok=True)
            if any(f.is_symlink() for f in target.rglob("*")):
                raise ValueError("destination contains symlinks")
            copy = ["rsync", "-a", "--checksum", "--partial", "--partial-dir=.rsync-partial", "--backup",
                    "--backup-dir=" + str(target / ".rsync-backups" / attempt), "-e", shlex.join(ssh),
                    host + ":" + REMOTE_ROOT + "/" + directory + "/", str(target) + "/"]
            child = subprocess.Popen(copy, start_new_session=True)
            try:
                while child.poll() is None:
                    if time.monotonic() >= deadline:
                        raise TimeoutError("copy/verification deadline reached")
                    if shutil.disk_usage(root).free < reserve:
                        raise RuntimeError("local free space fell below reserve; stopping this rsync only")
                    time.sleep(1)
                if child.returncode:
                    raise subprocess.CalledProcessError(child.returncode, copy)
            finally:
                if child.poll() is None:
                    # This process group was created above; the shared SSH
                    # master belongs to another group and is never signalled.
                    os.killpg(child.pid, signal.SIGTERM)
                    try: child.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL); child.wait(timeout=5)
        status["state"] = "VERIFYING_LOCAL"; write_json(receipt, status)
        actual = set()
        for directory in DIRECTORIES:
            for path in (root / directory).rglob("*"):
                if any(x in (".rsync-partial", ".rsync-backups") for x in path.parts): continue
                if path.is_symlink(): raise ValueError("unexpected copied symlink")
                if path.is_file(): actual.add(str(path.relative_to(root)))
        if actual != set(inventory["files"]):
            raise ValueError("copied file inventory differs; existing extra files are retained")
        for name, row in inventory["files"].items():
            path = root / name
            if path.stat().st_size != row["bytes"] or file_sha(path, deadline) != row["sha256"]:
                raise ValueError("local size/SHA256 mismatch: " + name)
        if shutil.disk_usage(root).free < reserve:
            raise RuntimeError("local free space below reserve after verification")
        status.update(state="VERIFIED_LOCAL", files_verified=len(actual), completed_unix_s=time.time(),
                      local_free_bytes=shutil.disk_usage(root).free)
        write_json(receipt, status)
        print(json.dumps({k: v for k, v in status.items() if k != "files"}, indent=2))
    except BaseException as error:
        status.update(state="COPY_OR_VERIFICATION_FAILED", ended_unix_s=time.time(),
                      error=f"{type(error).__name__}: {error}", partial_files_retained=True)
        write_json(receipt, status)
        raise
    finally:
        lock.close()


if __name__ == "__main__":
    main()
