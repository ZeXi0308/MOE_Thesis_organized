#!/usr/bin/env python3
"""Build a private RECORD-whitelisted torch overlay; never edit its source.

The host's torch 2.11 installation contains stale torch 2.x Python files. Its
kernel directory importer loads these as well as current files, registering the
flex_attention Triton template twice. Symlinking only RECORD-listed files gives
this experiment a clean import view without reinstalling the shared environment.
This is a CPU import check, not a CUDA or serving validation.
"""
import argparse
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time


PROBE = """import json
import torch
import torch._inductor.lowering
import vllm
import transformers
print(json.dumps({"torch": torch.__version__, "vllm": vllm.__version__,
                  "transformers": transformers.__version__,
                  "torch_file": torch.__file__,
                  "cuda_initialized": torch.cuda.is_initialized()}))
"""


def check_import(pythonpath):
    env = dict(os.environ, PYTHONPATH=pythonpath, PYTHONDONTWRITEBYTECODE="1")
    started = time.monotonic()
    try:
        result = subprocess.run([sys.executable, "-c", PROBE], env=env,
                                text=True, capture_output=True, timeout=90)
        return {"argv": [sys.executable, "-c", PROBE],
                "PYTHONPATH": pythonpath, "returncode": result.returncode,
                "seconds": time.monotonic() - started,
                "stdout": result.stdout, "stderr": result.stderr}
    except subprocess.TimeoutExpired as exc:
        return {"PYTHONPATH": pythonpath, "returncode": None,
                "seconds": time.monotonic() - started, "error": str(exc)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.absolute()
    # A rerun must not overwrite an existing environment or diagnostic result.
    output.mkdir(parents=True, exist_ok=False)
    dist = metadata.distribution("torch")
    base = Path(dist.locate_file("")).absolute()
    source = base / "torch"
    record = Path(dist._path) / "RECORD"
    listed = {str(p) for p in dist.files or []}
    actual = {str(p.relative_to(base)) for p in source.rglob("*")
              if p.is_file() and "__pycache__" not in p.parts}
    included = sorted(p for p in listed if p.startswith("torch/")
                      and "__pycache__" not in Path(p).parts)
    report = {
        "reason": "Untracked stale torch kernel modules cause duplicate template registration",
        "source": str(source), "overlay": str(output),
        "python": sys.executable, "python_version": sys.version,
        "versions": {p: metadata.version(p) for p in
                     ("torch", "vllm", "transformers", "triton")},
        "record_path": str(record),
        "record_sha256": hashlib.sha256(record.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH", ""),
        "excluded_untracked_files": sorted(actual - listed),
        "excluded_bytecode": True,
        "source_modified": False,
        "check_scope": "CPU imports only; does not initialize CUDA or run inference",
    }
    try:
        report["original_import"] = check_import("")
        for relative in included:
            target = base / relative
            if not target.is_file():
                raise FileNotFoundError(f"RECORD-listed source file missing: {target}")
            link = output / relative
            link.parent.mkdir(parents=True, exist_ok=True)
            link.symlink_to(target)
        (output / Path(dist._path).name).symlink_to(Path(dist._path), target_is_directory=True)
        report["symlinked_torch_files"] = len(included)
        report["overlay_import"] = check_import(str(output))
        report["status"] = ("CPU_IMPORT_OK" if report["overlay_import"]["returncode"] == 0
                            else "CPU_IMPORT_FAILED")
    except Exception as exc:
        report["status"] = "BUILD_FAILED"
        report["error"] = repr(exc)
        raise
    finally:
        (output / "repair.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({key: report.get(key) for key in
                          ("status", "source", "overlay", "symlinked_torch_files")}, indent=2))
    return 0 if report["status"] == "CPU_IMPORT_OK" else 1


if __name__ == "__main__":
    raise SystemExit(main())
