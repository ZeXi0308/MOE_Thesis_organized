#!/usr/bin/env python3
"""Create a non-approved formal binding preview for human review."""

from __future__ import annotations

import argparse
from pathlib import Path

from r0a_artifacts import (
    build_frozen_bindings,
    canonical_json_sha256,
    environment_snapshot,
    load_config,
    write_json_no_overwrite,
)
from run_r0a_5090 import (
    SOURCE_NAMES,
    _core_source_binding,
    _load_qualification,
    qualification_artifact_paths,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-provenance", type=Path, required=True)
    parser.add_argument("--historical-registry", type=Path, required=True)
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--review-dir", type=Path, required=True)
    args = parser.parse_args()
    experiment_dir = Path(__file__).resolve().parent
    config = load_config(args.config)
    _load_qualification(
        args.qualification,
        expected_phase="calibration",
        expected_core_binding=_core_source_binding(experiment_dir, config, args.repo_root),
        current_environment=environment_snapshot(),
    )
    bindings = build_frozen_bindings(
        repo_root=args.repo_root,
        config_path=args.config,
        manifest_path=args.manifest,
        source_paths=[
            *(experiment_dir / name for name in SOURCE_NAMES),
            args.data_provenance,
            args.historical_registry,
            *qualification_artifact_paths(args.qualification),
        ],
    )
    digest = canonical_json_sha256(bindings)
    args.review_dir.mkdir(parents=True, exist_ok=False)
    write_json_no_overwrite(args.review_dir / "formal_bindings_preview.json", bindings, mode=0o600)
    write_json_no_overwrite(
        args.review_dir / "approval.template.json",
        {
            "schema_version": "routeguard-kv-gpu-approval-v1",
            "approval": "NOT_APPROVED",
            "required_literal_after_human_review": config["approval"]["required_literal"],
            "frozen_bindings_sha256": digest,
            "scope": "R0A_MECHANISM_PROBE_ONLY",
            "note": "Copy to approval.json and change approval only after smoke/calibration Code Review.",
        },
        mode=0o600,
    )
    print(f"APPROVAL_TEMPLATE_CREATED bindings_sha256={digest} status=NOT_APPROVED")


if __name__ == "__main__":
    main()
