#!/usr/bin/env python3
"""Capture a path census proving Amendment-Q preceded sealed outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
from typing import Any, Mapping


HERE = Path(__file__).resolve().parent
IDEA_ROOT = HERE.parents[1]

try:
    from .build_scenarios import (
        FORMAL_AUTHORITATIVE_BUNDLE_ROOT,
        FORMAL_CENSUS_RELATIVE_ROOTS,
        FORMAL_PREOUTCOME_ATTESTATION_PATH,
    )
    from .formal_provenance import (
        FormalProvenanceError,
        add_self_hash,
        canonical_reviewed_scope_paths,
        sha256_file,
        verify_phase4_signoff,
    )
except ImportError:
    from build_scenarios import (  # type: ignore
        FORMAL_AUTHORITATIVE_BUNDLE_ROOT,
        FORMAL_CENSUS_RELATIVE_ROOTS,
        FORMAL_PREOUTCOME_ATTESTATION_PATH,
    )
    from formal_provenance import (  # type: ignore
        FormalProvenanceError,
        add_self_hash,
        canonical_reviewed_scope_paths,
        sha256_file,
        verify_phase4_signoff,
    )


DEFAULT_CONFIG = IDEA_ROOT / "configs" / "ric_v1.json"
DEFAULT_PROTOCOL = IDEA_ROOT / "RIC_Phase2_冻结实验协议_2026-07-22.md"
DEFAULT_AMENDMENT = IDEA_ROOT / "RIC_AmendmentQ_ConsumerMigration_2026-07-22.md"
REPO_ROOT = next(candidate for candidate in HERE.parents if (candidate / "experiments/shared").is_dir())
HISTORICAL_REVIEWED_SOURCE_SNAPSHOT_SHA256 = (
    "15db8b79ea590fa4c4354835c8ba472928433a685c4df82f8ff7c9d2e155a9b8"
)
SEALED_OUTCOME_SCHEMAS = {
    "ric-scenario-tree-v1",
    "ric-paired-bootstrap-v1",
    "ric-gate-matrix-v1",
    "ric-decision-v1",
    "ric-run-status-v1",
    "ric-sealed-evaluation-consumption-v1",
}


class PreOutcomeAttestationError(RuntimeError):
    """The bundle already contains a sealed scenario or result."""


def _load_json_strict_any(path: Path, *, label: str) -> Any:
    """Parse any JSON value while rejecting duplicates and non-finite tokens."""

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise PreOutcomeAttestationError(
                    f"duplicate JSON key in {label}: {key}"
                )
            result[key] = value
        return result

    def reject_nonfinite(token: str) -> None:
        raise PreOutcomeAttestationError(
            f"non-finite JSON constant in {label}: {token}"
        )

    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_nonfinite,
        )
    except PreOutcomeAttestationError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PreOutcomeAttestationError(
            f"invalid JSON in authoritative bundle: {label}"
        ) from exc


def _source_sha256() -> str:
    digest = hashlib.sha256()
    for path in (
        Path(__file__),
        HERE / "build_scenarios.py",
        HERE / "formal_provenance.py",
    ):
        digest.update(str(path.resolve().relative_to(REPO_ROOT.resolve())).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _canonical_relative(root: Path, path: Path) -> str:
    relative = path.relative_to(root).as_posix()
    parsed = PurePosixPath(relative)
    if (
        parsed.is_absolute()
        or not parsed.parts
        or any(part in {"", ".", ".."} for part in parsed.parts)
        or parsed.as_posix() != relative
    ):
        raise PreOutcomeAttestationError("non-canonical census path")
    return relative


def _sealed_outcome_reason(path: Path, value: Mapping[str, Any]) -> str | None:
    schema = value.get("schema_version")
    if schema == "ric-scenario-tree-v1" and value.get("role") == "sealed":
        return "sealed scenario tree already exists"
    if (
        schema == "ric-sealed-evaluation-consumption-v1"
        and value.get("role") == "sealed"
    ):
        return "sealed evaluation was already consumed"
    if (
        schema == "ric-sealed-consumption-v1"
        and value.get("role") == "sealed"
        and {
            "scenario_tree_sha256",
            "oracle_status_sha256",
            "run_experiment_source_sha256",
        }.issubset(value)
    ):
        return "legacy sealed evaluation was already consumed"
    if schema in SEALED_OUTCOME_SCHEMAS and value.get("stage") == "sealed":
        return "sealed replay/gate/decision already exists"
    return None


def _is_atomic_partial_path(root: Path, path: Path) -> bool:
    """Recognize crash remnants created by atomic_output_directory."""

    relative = path.relative_to(root)
    return any(
        part.startswith(".") and ".partial-" in part
        for part in relative.parts
    )


def capture(
    *,
    scanned_root: Path,
    output_path: Path,
    config_path: Path,
    protocol_path: Path,
    amendment_path: Path,
    snapshot_path: Path,
    required_inputs: Mapping[str, Path],
    signoff_path: Path,
) -> Mapping[str, Any]:
    root = scanned_root.resolve(strict=True)
    if root != FORMAL_AUTHORITATIVE_BUNDLE_ROOT:
        raise PreOutcomeAttestationError(
            "scanned root differs from reviewed formal bundle root"
        )
    if output_path.resolve(strict=False) != FORMAL_PREOUTCOME_ATTESTATION_PATH:
        raise PreOutcomeAttestationError(
            "output differs from reviewed write-once attestation path"
        )
    if output_path.exists():
        raise PreOutcomeAttestationError("pre-outcome attestation is write-once")
    if sha256_file(snapshot_path) != HISTORICAL_REVIEWED_SOURCE_SNAPSHOT_SHA256:
        raise PreOutcomeAttestationError("historical snapshot hash mismatch")
    expected_signoff = {
        "stage": "capture_preoutcome_attestation",
        "protocol_sha256": sha256_file(protocol_path),
        "config_sha256": sha256_file(config_path),
        "capture_preoutcome_attestation_source_sha256": _source_sha256(),
        "consumer_amendment_sha256": sha256_file(amendment_path),
        "historical_reviewed_source_snapshot_sha256": (
            HISTORICAL_REVIEWED_SOURCE_SNAPSHOT_SHA256
        ),
        "authoritative_bundle_root": str(root),
        "pre_outcome_attestation_path": str(FORMAL_PREOUTCOME_ATTESTATION_PATH),
    }
    try:
        signoff = verify_phase4_signoff(
            signoff_path,
            repo_root=REPO_ROOT,
            expected_fields=expected_signoff,
            required_source_paths=(
                Path(__file__),
                HERE / "build_scenarios.py",
                HERE / "formal_provenance.py",
            ),
            required_reviewed_scope_paths=(
                *canonical_reviewed_scope_paths(
                    REPO_ROOT,
                    (
                        Path(__file__),
                        HERE / "build_scenarios.py",
                        HERE / "formal_provenance.py",
                    ),
                ),
                amendment_path,
            ),
        )
    except FormalProvenanceError as exc:
        raise PreOutcomeAttestationError(
            "pre-outcome producer Phase-4 signoff is invalid"
        ) from exc
    if not required_inputs:
        raise PreOutcomeAttestationError("required input registry is empty")
    rows: list[dict[str, Any]] = []
    forbidden: list[dict[str, str]] = []
    for relative_root in FORMAL_CENSUS_RELATIVE_ROOTS:
        census_root = root / relative_root
        if census_root.is_symlink() or not census_root.is_dir():
            raise PreOutcomeAttestationError(
                f"authoritative census root is invalid: {relative_root}"
            )
    census_paths = sorted(
        path
        for relative_root in FORMAL_CENSUS_RELATIVE_ROOTS
        for path in (root / relative_root).rglob("*")
    )
    for path in census_paths:
        if path.is_symlink():
            raise PreOutcomeAttestationError(
                f"symlink is forbidden in authoritative bundle: {path}"
            )
        if _is_atomic_partial_path(root, path):
            raise PreOutcomeAttestationError(
                f"crash-partial formal output exists: {path}"
            )
        if not path.is_file():
            continue
        try:
            if path.resolve() == output_path.resolve():
                continue
        except FileNotFoundError:
            pass
        relative = _canonical_relative(root, path)
        rows.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
        if path.suffix != ".json":
            continue
        value = _load_json_strict_any(path, label=relative)
        reason = (
            _sealed_outcome_reason(path, value)
            if isinstance(value, Mapping)
            else None
        )
        if reason is not None:
            forbidden.append({"path": relative, "reason": reason})
    if not rows:
        raise PreOutcomeAttestationError("cannot attest an empty bundle")
    if forbidden:
        raise PreOutcomeAttestationError(
            f"sealed outcome already exists: {[row['path'] for row in forbidden]}"
        )
    census = {row["path"]: row for row in rows}
    registered: dict[str, Mapping[str, Any]] = {}
    for name, input_path in sorted(required_inputs.items()):
        if not name or name in registered:
            raise PreOutcomeAttestationError("invalid/duplicate required input name")
        resolved = input_path.resolve(strict=True)
        if resolved.is_symlink() or not resolved.is_file():
            raise PreOutcomeAttestationError("required input is not a regular file")
        try:
            relative = _canonical_relative(root, resolved)
        except ValueError as exc:
            raise PreOutcomeAttestationError(
                "required input is outside authoritative bundle"
            ) from exc
        row = census.get(relative)
        if row is None or row["sha256"] != sha256_file(resolved):
            raise PreOutcomeAttestationError("required input is absent from census")
        registered[name] = dict(row)
    payload = add_self_hash(
        {
            "schema_version": "ric-pre-outcome-attestation-v1",
            "status": "PRE_OUTCOME_CONFIRMED",
            "scientific_result": False,
            "protocol_sha256": sha256_file(protocol_path),
            "config_sha256": sha256_file(config_path),
            "consumer_amendment_sha256": sha256_file(amendment_path),
            "historical_reviewed_source_snapshot_sha256": (
                HISTORICAL_REVIEWED_SOURCE_SNAPSHOT_SHA256
            ),
            "capture_preoutcome_attestation_source_sha256": _source_sha256(),
            "producer_signoff_file_sha256": sha256_file(signoff_path),
            "producer_signoff_self_hash": signoff["signoff_sha256"],
            "scanned_root": str(root),
            "census_roots": list(FORMAL_CENSUS_RELATIVE_ROOTS),
            "path_census": rows,
            "required_inputs": registered,
            "forbidden_hits": [],
        },
        field="attestation_sha256",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor = os.open(output_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        offset = 0
        while offset < len(encoded):
            written = os.write(descriptor, encoded[offset:])
            if written <= 0:
                raise PreOutcomeAttestationError("short write to pre-outcome ledger")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    parent_descriptor = os.open(output_path.parent, directory_flags)
    try:
        os.fsync(parent_descriptor)
    finally:
        os.close(parent_descriptor)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scanned-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--consumer-amendment", type=Path, default=DEFAULT_AMENDMENT)
    parser.add_argument("--historical-reviewed-source-snapshot", type=Path, required=True)
    parser.add_argument("--signoff", type=Path, required=True)
    parser.add_argument(
        "--required-input",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="one exact immutable upstream file; repeat for the complete bundle",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    required_inputs: dict[str, Path] = {}
    for raw in args.required_input:
        name, separator, path = raw.partition("=")
        if not separator or not name or not path or name in required_inputs:
            raise PreOutcomeAttestationError("--required-input must be unique NAME=PATH")
        required_inputs[name] = Path(path)
    payload = capture(
        scanned_root=args.scanned_root,
        output_path=args.output,
        config_path=args.config,
        protocol_path=args.protocol,
        amendment_path=args.consumer_amendment,
        snapshot_path=args.historical_reviewed_source_snapshot,
        required_inputs=required_inputs,
        signoff_path=args.signoff,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
