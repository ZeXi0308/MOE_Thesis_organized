"""CPU-only gate for comparing two native recovery package outputs.

Each argument is a package directory containing manifest.json and pkg/.
The gate checks payload integrity, common backend bytes, physical/input
configuration, and exact workload identity before any performance contrast.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


COMMON_BACKEND = (
    "native_capture.py",
    "request_measurement.py",
    "rotation_native.py",
    "staged_save_contract.py",
    "native_offload_observer.py",
    "runtime_source_hashes.json",
)
COMMON_INPUTS = (
    "warmups/short/config.json",
    "warmups/short/workload.json",
    "warmups/long/config.json",
    "warmups/long/workload.json",
)
RESOURCE_AND_MODEL = (
    "model", "seed", "max_model_len", "engine_max_num_seqs",
    "max_num_batched_tokens", "fixed_kv_cache_memory_bytes",
    "target_usable_kv_blocks", "kv_block_bytes", "null_blocks",
    "offload_gib", "ignore_eos", "min_tokens", "max_output_tokens",
    "arrival_gap_s", "requests",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect(package: Path) -> dict:
    manifest = json.loads((package / "manifest.json").read_text())
    for relative, expected in manifest.items():
        path = package / relative
        if not path.is_file() or path.is_symlink() or digest(path) != expected:
            raise ValueError(f"payload drift: {package}/{relative}")
    input_dir = package / "pkg" / "inputs"
    config = json.loads((input_dir / "config.json").read_text())
    workload_path = input_dir / "workload.json"
    workload = json.loads(workload_path.read_text())
    canonical = hashlib.sha256(json.dumps(workload, sort_keys=True).encode()).hexdigest()
    if config["workload_sha256"] != canonical:
        raise ValueError(f"config/workload canonical hash mismatch: {package}")
    return {
        "package": str(package),
        "payload_files_checked": len(manifest),
        "config": config,
        "workload_file_sha256": digest(workload_path),
        "workload_canonical_sha256": canonical,
        "common_files": {
            name: digest(package / "pkg" / name)
            for name in COMMON_BACKEND + COMMON_INPUTS
        },
    }


def compare(left: Path, right: Path) -> dict:
    a, b = inspect(left), inspect(right)
    differences = []
    if a["workload_file_sha256"] != b["workload_file_sha256"]:
        differences.append("workload_file_sha256")
    for name in RESOURCE_AND_MODEL:
        if a["config"].get(name) != b["config"].get(name):
            differences.append(f"config.{name}")
    for name in a["common_files"]:
        if a["common_files"][name] != b["common_files"][name]:
            differences.append(f"pkg/{name}")
    return {
        "status": "COMPATIBLE_INPUT_AND_BACKEND" if not differences else
                  "INCOMPATIBLE_FOR_DIRECT_PERFORMANCE_CONTRAST",
        "differences": differences,
        "left": {k: a[k] for k in ("package", "payload_files_checked",
                                  "workload_file_sha256", "workload_canonical_sha256")},
        "right": {k: b[k] for k in ("package", "payload_files_checked",
                                   "workload_file_sha256", "workload_canonical_sha256")},
        "scope": "Static input/backend gate only; policy action, runtime resources and native lifecycle require live receipts.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    args = parser.parse_args()
    result = compare(args.left, args.right)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if result["differences"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
