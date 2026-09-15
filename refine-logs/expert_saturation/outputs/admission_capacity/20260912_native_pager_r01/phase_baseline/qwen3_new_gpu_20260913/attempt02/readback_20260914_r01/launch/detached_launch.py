"""Detach an unchanged, guarded run_remote.py; never restart or kill a campaign."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time


def absolute(value):
    path = Path(value)
    if not path.is_absolute():
        raise ValueError("absolute paths are required")
    return path.resolve(strict=True)


def guard(context):
    package = absolute(context["package_dir"])
    absolute(context["python"])
    interpreter = Path(context["python"])  # Preserve a venv interpreter symlink when executing.
    if not interpreter.is_file() or not os.access(interpreter, os.X_OK):
        raise ValueError("Python interpreter is not executable")
    for name, expected in (("run_remote.py", context["entry_sha256"]), ("protocol.json", context["protocol_sha256"])):
        path = absolute(package / name)
        if not path.is_relative_to(package) or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("entry/protocol guard failed: " + name)
    protocol = json.loads((package / "protocol.json").read_text())
    for name, expected in protocol["package_files"].items():
        path = absolute(package / name)
        if not path.is_relative_to(package) or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("package guard failed: " + name)
    return package, interpreter


def monitor(directory):
    directory = absolute(directory)
    context = json.loads((directory / "context.json").read_text())
    with (directory / "receipt.jsonl").open("x") as receipt:
        def event(status, **fields):
            receipt.write(json.dumps(dict(status=status, unix_s=time.time(), **fields)) + "\n")
            receipt.flush(); os.fsync(receipt.fileno())
        event("MONITOR_STARTED", monitor_pid=os.getpid(), monitor_session_id=os.getsid(0))
        try:
            package, interpreter = guard(context)
            child = subprocess.Popen([str(interpreter), "-u", str(package / "run_remote.py")], cwd=package,
                stdin=subprocess.DEVNULL, stdout=sys.stdout, stderr=subprocess.STDOUT, close_fds=True, start_new_session=True)
            event("CHILD_STARTED", child_pid=child.pid, command_entry=str(package / "run_remote.py"))
            result = child.wait()
            event("CHILD_EXITED", child_pid=child.pid, exit_code=result)
            return 0 if result == 0 else 1
        except Exception as exc:
            event("MONITOR_ERROR", error=f"{type(exc).__name__}: {exc}")
            return 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--monitor", help=argparse.SUPPRESS)
    parser.add_argument("--package-dir"); parser.add_argument("--python")
    parser.add_argument("--entry-sha256"); parser.add_argument("--protocol-sha256")
    parser.add_argument("--receipt-root")
    args = parser.parse_args()
    if args.monitor:
        return monitor(args.monitor)
    if any(getattr(args, key) is None for key in ("package_dir", "python", "entry_sha256", "protocol_sha256", "receipt_root")):
        parser.error("package, interpreter, both SHA256 values and receipt root are required")
    context = {key: getattr(args, key) for key in ("package_dir", "python", "entry_sha256", "protocol_sha256")}
    package, interpreter = guard(context)
    receipt_root = absolute(args.receipt_root)
    directory = Path(tempfile.mkdtemp(prefix="detached-", dir=receipt_root))
    context.update(package_dir=str(package), python=str(interpreter), cwd=str(package), launcher_pid=os.getpid(), created_unix_s=time.time())
    with (directory / "context.json").open("x") as stream:
        json.dump(context, stream, indent=2); stream.write("\n")
    with (directory / "monitor.log").open("xb") as log:
        child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--monitor", str(directory)],
            cwd=package, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, close_fds=True, start_new_session=True)
    with (directory / "launcher_receipt.json").open("x") as stream:
        json.dump(dict(status="DETACHED_STARTED", monitor_pid=child.pid, directory=str(directory), unix_s=time.time()), stream)
    print(json.dumps(dict(monitor_pid=child.pid, directory=str(directory))), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
