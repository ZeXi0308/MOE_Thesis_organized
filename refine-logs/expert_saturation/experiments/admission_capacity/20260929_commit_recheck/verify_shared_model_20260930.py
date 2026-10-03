#!/usr/bin/env python3
"""Read-only, exact-revision model check before A's native lifecycle cell."""

import hashlib
import json
from pathlib import Path
import sys


CACHE = Path("/root/autodl-tmp/c-research-20260930/hf-cache")
REVISION = "6d84c48581ece794365f2b8e9cfb043c68ade9c5"
MODEL_ROOT = CACHE / "hub/models--allenai--OLMoE-1B-7B-0924"
SNAPSHOT = MODEL_ROOT / "snapshots" / REVISION
# Config and tokenizer hashes were frozen in the earlier G/H input preflight.
# Generation and index hashes were frozen from this exact revision before A GPU use.
EXPECTED_METADATA = {
    "config.json": "3643aa880d2f1c9b418156269ae791c73e5612d6b6b6fde0724d927cf89b6335",
    "generation_config.json": "d77272ffaa7e62a904e8e130bb25ab11585bd4a5026e388d6d682e4b82892ce2",
    "model.safetensors.index.json": "0e2e1e0d8d357ac7af817cff28410c3dbad398f060c517a433e4076b2aae5579",
    "special_tokens_map.json": "b77491e270c6fcc5b2ecf22370f7318a6a18d3cabea09ba7bab92e9bf12656c2",
    "tokenizer.json": "a094266ac6c4982efba277bc251349a5a6d6ad37efb39a2a90f53d8be2a40a40",
    "tokenizer_config.json": "78a839c7851f14f9fb30e664c2b46166dc0628f2900679e5ec160656f702edff",
}
SHARDS = {
    "model-00001-of-00003.safetensors": (
        4997744872, "5e3cff7e367794685c241169072c940d200918617d5e2813f1c387dff52d845e"),
    "model-00002-of-00003.safetensors": (
        4997235176, "15ef5c730ee3cfed7199498788cd2faf337203fc74b529625e7502cdd759f4a7"),
    "model-00003-of-00003.safetensors": (
        3843741912, "a9abac4ac1b55c9adabac721a02fa39971f103eea9a65c310972b1246de76e04"),
}


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def checked_file(name: str) -> Path:
    path = SNAPSHOT / name
    if not path.is_symlink() or not path.is_file():
        raise ValueError(f"missing or non-cache snapshot file: {name}")
    if not path.resolve().is_relative_to((MODEL_ROOT / "blobs").resolve()):
        raise ValueError(f"snapshot file points outside model blobs: {name}")
    return path


def main() -> dict:
    if not SNAPSHOT.is_dir() or SNAPSHOT.is_symlink():
        raise ValueError("exact model revision snapshot is missing or a symlink")
    metadata = {name: digest(checked_file(name)) for name in EXPECTED_METADATA}
    for name, expected in EXPECTED_METADATA.items():
        if metadata[name] != expected:
            raise ValueError(f"fixed model metadata SHA-256 differs: {name}")
    index = json.loads(checked_file("model.safetensors.index.json").read_text())
    names = set(index["weight_map"].values())
    if names != set(SHARDS):
        raise ValueError(f"fixed model shard names differ: {sorted(names)}")
    checked = {}
    for name, (size, sha) in SHARDS.items():
        path = checked_file(name)
        if path.stat().st_size != size:
            raise ValueError(f"fixed model shard byte size differs: {name}")
        actual = digest(path)
        if actual != sha:
            raise ValueError(f"fixed model shard SHA-256 differs: {name}")
        checked[name] = {"bytes": size, "sha256": actual}
    return {
        "status": "VERIFIED_READ_ONLY",
        "cache": str(CACHE),
        "revision": REVISION,
        "metadata_sha256": metadata,
        "shards": checked,
        "scope": "Exact revision and shard bytes only; native GPU lifecycle remains unrun.",
    }


if __name__ == "__main__":
    try:
        result = main()
    except Exception as error:
        result = {"status": "INCOMPLETE", "error": f"{type(error).__name__}: {error}"}
        print(json.dumps(result, indent=2, sort_keys=True))
        sys.exit(1)
    print(json.dumps(result, indent=2, sort_keys=True))
