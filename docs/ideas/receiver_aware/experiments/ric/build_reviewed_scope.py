#!/usr/bin/env python3
"""Build the canonical current-byte RIC Phase-4 reviewed-scope artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
IDEA_ROOT = HERE.parents[1]
REPO_ROOT = HERE.parents[4]

try:
    from .formal_provenance import add_self_hash, sha256_file
except ImportError:
    from formal_provenance import add_self_hash, sha256_file  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--git-head-artifact", type=Path, required=True)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=IDEA_ROOT / "RIC_Phase2_冻结实验协议_2026-07-22.md",
    )
    parser.add_argument(
        "--config", type=Path, default=IDEA_ROOT / "configs" / "ric_v1.json"
    )
    parser.add_argument("--reviewed-extra", type=Path, action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite {args.output}")
    head_text = args.git_head_artifact.read_text(encoding="ascii")
    git_head = head_text.removesuffix("\n")
    if head_text != f"{git_head}\n" or len(git_head) != 40:
        raise RuntimeError("invalid reviewed git-head artifact")
    paths = sorted(
        [*HERE.glob("*.py"), args.protocol, args.config, *args.reviewed_extra],
        key=lambda path: path.resolve().relative_to(REPO_ROOT.resolve()).as_posix(),
    )
    rows = [
        {
            "path": path.resolve().relative_to(REPO_ROOT.resolve()).as_posix(),
            "sha256": sha256_file(path.resolve(strict=True)),
        }
        for path in paths
    ]
    payload = add_self_hash(
        {
            "schema_version": "ric-reviewed-worktree-v1",
            "status": "REVIEWED",
            "git_head": git_head,
            "protocol_sha256": sha256_file(args.protocol.resolve(strict=True)),
            "config_sha256": sha256_file(args.config.resolve(strict=True)),
            "sources": rows,
        },
        field="scope_sha256",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        errors="strict",
    )
    print(json.dumps({"output": str(args.output), "scope_sha256": payload["scope_sha256"]}))


if __name__ == "__main__":
    main()
