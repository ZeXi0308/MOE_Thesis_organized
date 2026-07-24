#!/usr/bin/env python3
"""Build a reproducible, stage-specific RIC Phase-4 source manifest/signoff."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[4]

try:
    from .formal_provenance import (
        FormalProvenanceError,
        build_phase4_signoff_payload,
        build_source_manifest_payload,
        canonical_reviewed_scope_paths,
        load_json_mapping_strict,
        sha256_file,
        validate_reviewed_scope,
    )
except ImportError:
    from formal_provenance import (  # type: ignore
        FormalProvenanceError,
        build_phase4_signoff_payload,
        build_source_manifest_payload,
        canonical_reviewed_scope_paths,
        load_json_mapping_strict,
        sha256_file,
        validate_reviewed_scope,
    )


class SignoffBuildError(RuntimeError):
    """The requested evidence set is incomplete or not canonical."""


def _strict_mapping(path: Path, *, label: str) -> Mapping[str, Any]:
    try:
        value = load_json_mapping_strict(path, label=label)
    except FormalProvenanceError as exc:
        raise SignoffBuildError(str(exc)) from exc
    if not isinstance(value, Mapping):
        raise SignoffBuildError(f"{label} must be a JSON object")
    return value


def _git_head(path: Path) -> str:
    try:
        value = path.read_text(encoding="ascii")
    except (OSError, UnicodeError) as exc:
        raise SignoffBuildError("cannot read git-head artifact") from exc
    head = value.removesuffix("\n")
    if value != f"{head}\n" or len(head) != 40:
        raise SignoffBuildError("git-head artifact must contain one full SHA-1")
    return head


def _verify_reviewed_scope(
    path: Path,
    *,
    source_manifest: Mapping[str, Any],
    git_head: str,
    expected_fields: Mapping[str, Any],
    stage_source_paths: Sequence[Path],
    reviewed_extra_paths: Sequence[Path] = (),
) -> None:
    """Builder-side preflight; consumers repeat the same shared check."""
    try:
        validate_reviewed_scope(
            path,
            repo_root=REPO_ROOT,
            source_manifest=source_manifest,
            required_reviewed_paths=tuple(
                dict.fromkeys(
                    (
                        *canonical_reviewed_scope_paths(
                            REPO_ROOT, stage_source_paths
                        ),
                        *(path.resolve(strict=True) for path in reviewed_extra_paths),
                    )
                )
            ),
            git_head=git_head,
            protocol_sha256=str(expected_fields.get("protocol_sha256")),
            config_sha256=str(expected_fields.get("config_sha256")),
        )
    except FormalProvenanceError as exc:
        raise SignoffBuildError(str(exc)) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--expected-fields", type=Path, required=True)
    parser.add_argument("--test-summary", type=Path, required=True)
    parser.add_argument("--review-report", type=Path, required=True)
    parser.add_argument("--test-report", type=Path, required=True)
    parser.add_argument("--reviewed-patch", type=Path, required=True)
    parser.add_argument("--git-head-artifact", type=Path, required=True)
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--reviewed-extra", type=Path, action="append", default=[])
    parser.add_argument("--source-manifest-output", type=Path, required=True)
    parser.add_argument("--signoff-output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for output in (args.source_manifest_output, args.signoff_output):
        if output.exists():
            raise SignoffBuildError(f"refusing to overwrite {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
    expected_fields = _strict_mapping(args.expected_fields, label="expected fields")
    test_summary = _strict_mapping(args.test_summary, label="test summary")
    git_head = _git_head(args.git_head_artifact)
    try:
        source_manifest = build_source_manifest_payload(
            repo_root=REPO_ROOT,
            source_paths=args.source,
            git_head=git_head,
            worktree_patch_sha256=sha256_file(args.reviewed_patch),
        )
        _verify_reviewed_scope(
            args.reviewed_patch,
            source_manifest=source_manifest,
            git_head=git_head,
            expected_fields=expected_fields,
        stage_source_paths=args.source,
        reviewed_extra_paths=args.reviewed_extra,
    )
        args.source_manifest_output.write_text(
            json.dumps(source_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            errors="strict",
        )
        signoff = build_phase4_signoff_payload(
            repo_root=REPO_ROOT,
            stage=args.stage,
            expected_fields=expected_fields,
            artifact_paths={
                "review_report": args.review_report,
                "source_manifest": args.source_manifest_output,
                "test_report": args.test_report,
                "reviewed_patch": args.reviewed_patch,
                "git_head_artifact": args.git_head_artifact,
            },
            git_head=git_head,
            test_summary=test_summary,
        )
    except FormalProvenanceError as exc:
        raise SignoffBuildError(str(exc)) from exc
    args.signoff_output.write_text(
        json.dumps(signoff, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        errors="strict",
    )
    print(
        json.dumps(
            {
                "stage": args.stage,
                "source_manifest_sha256": source_manifest["manifest_sha256"],
                "signoff_sha256": signoff["signoff_sha256"],
                "signoff_output": str(args.signoff_output),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
