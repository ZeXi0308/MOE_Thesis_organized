#!/usr/bin/env python3
"""Hash-close a completed Longrun-A bundle after its tee stream has exited."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time


REQUIRED_MAIN = {
    "RUN_STARTED.json",
    "RUN_COMPLETE.json",
    "arm_metrics.json",
    "commands.sh",
    "config.json",
    "first_divergence.json",
    "matched_alternatives.json",
    "propagation.json",
    "report.md",
    "run.log",
    "selected_events.json",
}
REQUIRED_PREVALENCE = {
    "RUN_STARTED.json",
    "RUN_COMPLETE.json",
    "commands.sh",
    "config.json",
    "report.md",
    "results.json",
    "run.log",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{path} is not an object")
    return value


def current_files(output_dir: Path) -> dict[str, dict[str, int | str]]:
    paths = sorted(
        path
        for path in output_dir.rglob("*")
        if path.is_file() and path.name != "POST_RUN_SEAL.json"
    )
    return {
        str(path.relative_to(output_dir)): {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for path in paths
    }


def verify_complete_manifest(output_dir: Path, complete: dict) -> None:
    if complete.get("status") not in {
        "MAIN_EXPERIMENT_COMPLETE",
        "PREVALENCE_CHECK_COMPLETE",
    }:
        raise RuntimeError("RUN_COMPLETE status is not complete")
    for relative, expected in complete.get("files", {}).items():
        path = output_dir / relative
        if not path.is_file() or sha256_file(path) != expected.get("sha256"):
            raise RuntimeError(f"RUN_COMPLETE hash mismatch: {relative}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    seal_path = output_dir / "POST_RUN_SEAL.json"
    if args.verify:
        seal = read_json(seal_path)
        observed = current_files(output_dir)
        if seal.get("status") != "POST_RUN_SEALED" or observed != seal.get("files"):
            raise RuntimeError("post-run seal verification failed")
        print(json.dumps({"status": "POST_RUN_SEAL_VERIFIED", "files": len(observed)}))
        return
    if seal_path.exists():
        raise RuntimeError(f"refusing to overwrite {seal_path}")
    complete = read_json(output_dir / "RUN_COMPLETE.json")
    required = (
        REQUIRED_MAIN
        if complete.get("status") == "MAIN_EXPERIMENT_COMPLETE"
        else REQUIRED_PREVALENCE
    )
    missing = sorted(name for name in required if not (output_dir / name).is_file())
    if missing:
        raise RuntimeError(f"required artifact files are missing: {missing}")
    if (output_dir / "RUN_FAILED.json").exists():
        raise RuntimeError("cannot seal a failed run")
    verify_complete_manifest(output_dir, complete)
    files = current_files(output_dir)
    payload = {
        "schema": "longrun-a-post-run-seal-v1",
        "status": "POST_RUN_SEALED",
        "sealed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seal_tool": str(Path(__file__).resolve()),
        "seal_tool_sha256": sha256_file(Path(__file__).resolve()),
        "required_top_level_files": sorted(required),
        "files": files,
    }
    with seal_path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"status": "POST_RUN_SEALED", "files": len(files)}))


if __name__ == "__main__":
    main()
