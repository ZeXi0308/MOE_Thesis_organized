#!/usr/bin/env python3
"""Validate the exact vLLM 0.26 source state for the valid-window Gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


FILES = {
    "vllm/model_executor/layers/fused_moe/routed_experts_capturer.py": {
        "original": "a75c29de4425efcb97d0e215b988fec38cd16ee87e69c0c16d985e848c32021d",
        "patched": "690de10ebd1ccb4ce156f8432ec351513d59702a1eacd9d1bf86dabb2b54226e",
    },
    "vllm/v1/worker/gpu_model_runner.py": {
        "original": "81b7627fbe81f7aaa2f77b4bf085faa353c69d03662ebfe369536a9773bb70d0",
        "patched": "1253bec5fafbc8e4ad0ea2735b50d2b6e6f4f97f02300fd489f4d29b2a2ee8ac",
    },
}
PATCH_SHA256 = "862b3ff7732fd4ccac4ffeba923174ab3d662e57834a981eb329aba893e0d87b"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(source_root: Path, patch_path: Path) -> dict[str, object]:
    patch_actual = sha256(patch_path)
    errors: list[str] = []
    states: list[str] = []
    sources: dict[str, dict[str, object]] = {}
    if patch_actual != PATCH_SHA256:
        errors.append("patch_sha256_mismatch")

    for relative, expected in FILES.items():
        path = source_root / relative
        if not path.is_file():
            errors.append(f"missing:{relative}")
            continue
        actual = sha256(path)
        state = next((name for name, digest in expected.items() if digest == actual), "unknown")
        states.append(state)
        sources[relative] = {
            "sha256": actual,
            "state": state,
            "size_bytes": path.stat().st_size,
        }
        if state == "unknown":
            errors.append(f"unexpected_source_hash:{relative}")
        try:
            compile(path.read_text(), str(path), "exec")
        except SyntaxError as exc:
            errors.append(f"syntax:{relative}:{exc}")

    source_state = states[0] if states and len(states) == len(FILES) and len(set(states)) == 1 else "mixed_or_unknown"
    if source_state == "mixed_or_unknown":
        errors.append("source_files_not_in_one_known_state")
    return {
        "schema": "vllm-valid-window-source-validation-v1",
        "valid": not errors,
        "source_state": source_state,
        "patch_sha256": patch_actual,
        "sources": sources,
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_root", type=Path)
    parser.add_argument(
        "--patch",
        type=Path,
        default=Path(__file__).with_name("vllm-0.26-valid-route-window.patch"),
    )
    args = parser.parse_args()
    report = validate(args.source_root.resolve(), args.patch.resolve())
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["valid"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
