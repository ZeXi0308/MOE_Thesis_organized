#!/usr/bin/env python3
"""Fail-closed formal provenance primitives for the frozen RIC-v1 pipeline.

This module contains no experiment policy.  It verifies that a Phase-4
signoff refers to the exact reviewed repository files and provides the strict
data-manifest validation shared by downstream GPU producers.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import subprocess
from typing import Any, Mapping, Sequence


SIGNOFF_SCHEMA = "ric-phase4-signoff-v1"
SOURCE_MANIFEST_SCHEMA = "ric-source-manifest-v1"
DATA_MANIFEST_SCHEMA = "ric-data-manifest-v1"
CALIBRATION_LOCK_SCHEMA = "ric-calibration-lock-v1"
EMBEDDED_PRODUCER_SIGNOFF = "producer_signoff.json"
SIGNOFF_ARTIFACT_FIELDS = (
    "review_report",
    "source_manifest",
    "test_report",
    "reviewed_patch",
    "git_head_artifact",
)
TEXT_HASH_FIELDS = frozenset(
    {"text_sha256", "canonical_text_sha256", "request_text_sha256"}
)


class FormalProvenanceError(RuntimeError):
    """A formal provenance or frozen-manifest invariant failed."""


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def materialize_verified_signoff(path: Path | None, output_dir: Path) -> str:
    """Copy one already-verified formal signoff into an atomic output tree.

    Downstream consumers must verify this embedded copy again against their
    current source closure.  Recording only an arbitrary 64-hex value in a
    producer manifest is not a provenance chain.
    """

    if path is None:
        raise FormalProvenanceError("formal output requires a producer signoff")
    value = load_json_mapping_strict(path, label="producer Phase-4 signoff")
    if value.get("schema_version") != SIGNOFF_SCHEMA:
        raise FormalProvenanceError("producer signoff schema mismatch")
    validate_self_hash(value, field="signoff_sha256")
    if value.get("status") != "SIGNED-OFF" or value.get("open_p0") != 0:
        raise FormalProvenanceError("producer signoff is not signed off")
    target = output_dir / EMBEDDED_PRODUCER_SIGNOFF
    if target.exists():
        raise FormalProvenanceError("embedded producer signoff already exists")
    target.write_bytes(path.read_bytes())
    return sha256_file(target)


def is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _strict_self_hash_json_object(
    pairs: Sequence[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise FormalProvenanceError(
                f"self-hashed payload has duplicate JSON object key {key!r}"
            )
        result[key] = value
    return result


def add_self_hash(
    payload: Mapping[str, Any], *, field: str = "manifest_sha256"
) -> dict[str, Any]:
    if field in payload:
        raise FormalProvenanceError(f"refusing to replace existing {field}")
    try:
        normalized = json.loads(
            json.dumps(dict(payload), ensure_ascii=False, allow_nan=False),
            object_pairs_hook=_strict_self_hash_json_object,
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise FormalProvenanceError("self-hashed payload is not strict JSON") from exc
    if not isinstance(normalized, dict):
        raise FormalProvenanceError("self-hashed payload must be a JSON object")
    result = normalized
    result[field] = sha256_bytes(canonical_json_bytes(result))
    return result


def validate_self_hash(
    payload: Mapping[str, Any], *, field: str = "manifest_sha256"
) -> str:
    try:
        normalized = json.loads(
            json.dumps(dict(payload), ensure_ascii=False, allow_nan=False),
            object_pairs_hook=_strict_self_hash_json_object,
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise FormalProvenanceError("self-hashed payload is not strict JSON") from exc
    if not isinstance(normalized, dict):
        raise FormalProvenanceError("self-hashed payload must be a JSON object")
    supplied = normalized.get(field)
    unhashed = dict(normalized)
    unhashed.pop(field, None)
    actual = sha256_bytes(canonical_json_bytes(unhashed))
    if supplied != actual:
        raise FormalProvenanceError(f"{field} mismatch")
    return actual


def _canonical_relative_path(raw: object) -> PurePosixPath:
    if not isinstance(raw, str) or not raw:
        raise FormalProvenanceError("artifact path must be a non-empty string")
    if "\\" in raw:
        raise FormalProvenanceError("artifact path must use canonical POSIX separators")
    relative = PurePosixPath(raw)
    if relative.is_absolute() or str(relative) != raw:
        raise FormalProvenanceError("artifact path must be canonical and repo-relative")
    if any(part in {"", ".", ".."} for part in relative.parts):
        raise FormalProvenanceError("artifact path contains a forbidden component")
    return relative


def _relocated_ric_v1_path(relative: PurePosixPath) -> PurePosixPath:
    """Locate immutable RIC-v1 files after the documentation-only archive move.

    Historical signoffs keep their original repo-relative path strings.  This
    exact-prefix relocation changes only file lookup; the existing content
    hashes and signoff checks remain authoritative.
    """

    old_root = PurePosixPath("docs/ideas/receiver_aware")
    try:
        tail = relative.relative_to(old_root)
    except ValueError:
        return relative
    if not tail.parts:
        return relative
    if tail.parts[0] == "formal_signoff":
        return PurePosixPath("docs/archive/receiver_aware/ric_v1") / tail
    if tail.parts[:2] == ("experiments", "ric"):
        return PurePosixPath("docs/archive/receiver_aware/ric_v1") / tail
    if tail == PurePosixPath("configs/ric_v1.json"):
        return PurePosixPath("docs/archive/receiver_aware/ric_v1") / tail
    if tail.name.startswith("RIC_") and not tail.name.startswith("RIC_Clean"):
        return PurePosixPath("docs/archive/receiver_aware/ric_v1") / tail
    return relative


def resolve_repo_file(repo_root: Path, raw_relative_path: object) -> Path:
    """Resolve one canonical repo-relative regular file without symlinks."""

    root = repo_root.resolve(strict=True)
    stored_relative = _canonical_relative_path(raw_relative_path)
    relocated = _relocated_ric_v1_path(stored_relative)
    relative = (
        relocated
        if root.joinpath(*relocated.parts).exists()
        else stored_relative
    )
    candidate = root.joinpath(*relative.parts)
    cursor = root
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise FormalProvenanceError(f"symlink substitution is forbidden: {relative}")
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise FormalProvenanceError(
            f"artifact path is missing or outside repository: {relative}"
        ) from exc
    if resolved != candidate or not resolved.is_file():
        raise FormalProvenanceError(
            f"artifact path is not a canonical regular file: {relative}"
        )
    return resolved


def repo_relative_file(repo_root: Path, path: Path) -> str:
    """Return a canonical repo-relative path and reject any symlink component."""

    root = repo_root.resolve(strict=True)
    if path.is_symlink():
        raise FormalProvenanceError("required source cannot be a symlink")
    try:
        raw_relative = path.resolve(strict=True).relative_to(root).as_posix()
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise FormalProvenanceError("required source is outside repository") from exc
    resolved = resolve_repo_file(root, raw_relative)
    if resolved != path.resolve(strict=True):
        raise FormalProvenanceError("required source path is not canonical")
    return raw_relative


def loads_json_mapping_strict(raw: str, *, label: str) -> dict[str, Any]:
    """Parse one JSON object while rejecting duplicate keys at every depth."""

    def reject_duplicates(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise FormalProvenanceError(f"duplicate JSON key in {label}: {key}")
            result[key] = item
        return result

    def reject_nonfinite(token: str) -> None:
        raise FormalProvenanceError(
            f"non-finite JSON constant in {label}: {token}"
        )

    try:
        value = json.loads(
            raw,
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_nonfinite,
        )
    except FormalProvenanceError:
        raise
    except json.JSONDecodeError as exc:
        raise FormalProvenanceError(f"cannot read {label}") from exc
    if not isinstance(value, dict):
        raise FormalProvenanceError(f"{label} must be a JSON object")
    return value


def load_json_mapping_strict(path: Path, *, label: str) -> dict[str, Any]:
    """Load one JSON object while rejecting duplicate keys at every depth."""

    try:
        raw = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise FormalProvenanceError(f"cannot read {label}") from exc
    return loads_json_mapping_strict(raw, label=label)


# Backward-compatible private spelling for already-reviewed callers in this
# module.  New producers should import the explicit strict spelling.
_load_json_mapping = load_json_mapping_strict


def validate_calibration_lock_fields(
    value: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    protocol_sha256: str,
    config_sha256: str,
    expected_run_experiment_source_sha256: str,
) -> Mapping[str, Any]:
    """Validate the formal, G1-passing calibration lock before sealed access.

    The lock is an authorization boundary, not merely metadata.  In
    particular, ``bool(1)``-style coercions are forbidden and every frozen
    model must have a complete main-cell budget before the lock can authorize
    sealed preparation or the matched-world oracle.
    """

    if not all(
        is_sha256(item)
        for item in (
            protocol_sha256,
            config_sha256,
            expected_run_experiment_source_sha256,
        )
    ):
        raise FormalProvenanceError("calibration lock expected binding is invalid")
    if value.get("schema_version") != CALIBRATION_LOCK_SCHEMA:
        raise FormalProvenanceError("calibration lock schema mismatch")
    validate_self_hash(value)
    frozen = {
        "status": "CALIBRATION_LOCKED",
        "scientific_result": False,
        "mode": "formal",
        "role": "calibration",
        "protocol_sha256": protocol_sha256,
        "config_sha256": config_sha256,
        "run_experiment_source_sha256": expected_run_experiment_source_sha256,
    }
    for field, wanted in frozen.items():
        if value.get(field) != wanted or type(value.get(field)) is not type(wanted):
            raise FormalProvenanceError(
                f"calibration lock frozen binding mismatch: {field}"
            )
    go_no_go = config.get("go_no_go")
    if not isinstance(go_no_go, Mapping):
        raise FormalProvenanceError("config lacks calibration-lock gate identity")
    required_models_raw = go_no_go.get("required_models")
    required_cells_raw = go_no_go.get("required_main_cells")
    if (
        not isinstance(required_models_raw, list)
        or not required_models_raw
        or not all(isinstance(item, str) and item for item in required_models_raw)
        or len(set(required_models_raw)) != len(required_models_raw)
        or not isinstance(required_cells_raw, list)
        or not required_cells_raw
        or not all(isinstance(item, str) and item for item in required_cells_raw)
        or len(set(required_cells_raw)) != len(required_cells_raw)
    ):
        raise FormalProvenanceError("config calibration-lock model/cell grid is invalid")
    required_models = set(required_models_raw)
    required_cells = set(required_cells_raw)
    for field in (
        "scenario_tree_sha256",
        "service_lut_metadata_sha256",
        "capability_probe_sha256",
        "scenario_producer_signoff_sha256",
        "capability_producer_signoff_sha256",
    ):
        grid = value.get(field)
        if not isinstance(grid, Mapping) or set(grid) != required_models:
            raise FormalProvenanceError(
                f"calibration lock incomplete model binding: {field}"
            )
        if any(not is_sha256(grid[model]) for model in required_models):
            raise FormalProvenanceError(
                f"calibration lock invalid model hash binding: {field}"
            )
    g1_by_model = value.get("g1_by_model")
    if not isinstance(g1_by_model, Mapping) or set(g1_by_model) != required_models:
        raise FormalProvenanceError("calibration lock G1 model grid is incomplete")
    if any(type(g1_by_model[model]) is not bool for model in required_models):
        raise FormalProvenanceError("calibration lock G1 decisions must be exact bools")
    if value.get("g1_pass") is not True or not all(g1_by_model.values()):
        raise FormalProvenanceError("calibration lock G1 did not pass")
    models = value.get("models")
    if not isinstance(models, Mapping) or set(models) != required_models:
        raise FormalProvenanceError("calibration lock model payload grid is incomplete")
    for model in required_models:
        model_payload = models[model]
        if not isinstance(model_payload, Mapping):
            raise FormalProvenanceError("calibration lock model payload is invalid")
        cells = model_payload.get("cells")
        if not isinstance(cells, Mapping) or not required_cells.issubset(cells):
            raise FormalProvenanceError(
                f"calibration lock main-cell grid is incomplete: {model}"
            )
        for cell in required_cells:
            row = cells[cell]
            if not isinstance(row, Mapping):
                raise FormalProvenanceError("calibration lock cell payload is invalid")
            budget = row.get("closure_budget_us")
            if (
                isinstance(budget, bool)
                or not isinstance(budget, (int, float))
                or not math.isfinite(float(budget))
                or float(budget) <= 0
            ):
                raise FormalProvenanceError(
                    f"calibration lock closure budget is invalid: {model}/{cell}"
                )
    if not is_sha256(value.get("policy_semantics_sha256")):
        raise FormalProvenanceError("calibration lock lacks policy-semantics binding")
    if not is_sha256(value.get("signoff_sha256")):
        raise FormalProvenanceError("calibration lock lacks producer signoff")
    return value


def verify_calibration_lock_producer_signoff(
    lock_path: Path,
    lock: Mapping[str, Any],
    *,
    repo_root: Path,
    required_source_paths: Sequence[Path],
) -> Mapping[str, Any]:
    """Verify the runner attestation embedded beside a calibration lock."""

    signoff_path = lock_path.parent / EMBEDDED_PRODUCER_SIGNOFF
    if (
        not signoff_path.is_file()
        or sha256_file(signoff_path) != lock.get("signoff_sha256")
    ):
        raise FormalProvenanceError(
            "calibration lock embedded producer signoff mismatch"
        )
    expected = {
        "stage": "calibration",
        "config_sha256": lock.get("config_sha256"),
        "protocol_sha256": lock.get("protocol_sha256"),
        "run_experiment_source_sha256": lock.get(
            "run_experiment_source_sha256"
        ),
        "scenario_tree_sha256": lock.get("scenario_tree_sha256"),
        "scenario_producer_signoff_sha256": lock.get(
            "scenario_producer_signoff_sha256"
        ),
        "capability_probe_sha256": lock.get("capability_probe_sha256"),
        "capability_producer_signoff_sha256": lock.get(
            "capability_producer_signoff_sha256"
        ),
    }
    return verify_phase4_signoff(
        signoff_path,
        repo_root=repo_root,
        expected_fields=expected,
        required_source_paths=required_source_paths,
    )


def current_git_head(repo_root: Path) -> str:
    # Use the OS git directly.  Developer PATH shims can depend on the parent
    # interpreter's locale and are not part of the reviewed provenance chain.
    git_executable = Path("/usr/bin/git")
    if not git_executable.is_file():
        raise FormalProvenanceError("system git executable is unavailable")
    try:
        result = subprocess.run(
            [str(git_executable), "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise FormalProvenanceError("cannot resolve current git HEAD") from exc
    if len(result) != 40 or any(character not in "0123456789abcdef" for character in result):
        raise FormalProvenanceError("current git HEAD is not a full SHA-1")
    return result


def _verify_git_head_binding(
    *, repo_root: Path, signed_git_head: object, artifact_path: Path
) -> str:
    if (
        not isinstance(signed_git_head, str)
        or len(signed_git_head) != 40
        or any(character not in "0123456789abcdef" for character in signed_git_head)
    ):
        raise FormalProvenanceError("Phase-4 signoff git_head is not a full SHA-1")
    try:
        artifact_value = artifact_path.read_text(encoding="ascii")
    except (OSError, UnicodeError) as exc:
        raise FormalProvenanceError("cannot read reviewed git-head artifact") from exc
    if artifact_value != f"{signed_git_head}\n":
        raise FormalProvenanceError("reviewed git-head artifact content mismatch")
    # A full checkout must agree with its actual HEAD.  A deliberately exported
    # formal bundle may omit .git; in that case the separately reviewed,
    # content-hashed git-head artifact remains mandatory and source files are
    # still verified byte-for-byte.
    if (repo_root / ".git").exists() and current_git_head(repo_root) != signed_git_head:
        raise FormalProvenanceError("Phase-4 signoff git_head differs from current HEAD")
    return signed_git_head


def _verify_artifact_reference(
    value: Mapping[str, Any], *, field: str, repo_root: Path
) -> tuple[Path, str]:
    reference = value.get(field)
    if not isinstance(reference, Mapping) or set(reference) != {"path", "sha256"}:
        raise FormalProvenanceError(
            f"{field} must contain exactly repo-relative path and sha256"
        )
    if not is_sha256(reference.get("sha256")):
        raise FormalProvenanceError(f"{field} has an invalid sha256")
    path = resolve_repo_file(repo_root, reference.get("path"))
    actual = sha256_file(path)
    if actual != reference["sha256"]:
        raise FormalProvenanceError(f"{field} file hash mismatch")
    return path, actual


def verify_source_manifest(
    path: Path,
    *,
    repo_root: Path,
    required_source_paths: Sequence[Path],
    expected_git_head: str,
    expected_worktree_patch_sha256: str,
) -> Mapping[str, Any]:
    """Verify all listed source files and require the exact stage source set."""

    value = _load_json_mapping(path, label="source manifest")
    if value.get("schema_version") != SOURCE_MANIFEST_SCHEMA:
        raise FormalProvenanceError("source manifest schema mismatch")
    validate_self_hash(value)
    if value.get("git_head") != expected_git_head:
        raise FormalProvenanceError("source manifest git_head mismatch")
    if value.get("worktree_patch_sha256") != expected_worktree_patch_sha256:
        raise FormalProvenanceError("source manifest worktree patch mismatch")
    rows = value.get("sources")
    if not isinstance(rows, list) or not rows:
        raise FormalProvenanceError("source manifest must contain source rows")
    observed: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != {"path", "sha256"}:
            raise FormalProvenanceError("source row schema mismatch")
        relative = _canonical_relative_path(row.get("path")).as_posix()
        if relative in observed:
            raise FormalProvenanceError("duplicate source path in manifest")
        if not is_sha256(row.get("sha256")):
            raise FormalProvenanceError("source row has invalid sha256")
        current_path = resolve_repo_file(repo_root, relative)
        current_hash = sha256_file(current_path)
        if current_hash != row["sha256"]:
            raise FormalProvenanceError(f"current source hash mismatch: {relative}")
        observed[relative] = current_hash
    expected = {
        repo_relative_file(repo_root, source): sha256_file(source.resolve(strict=True))
        for source in required_source_paths
    }
    if observed != expected:
        missing = sorted(set(expected) - set(observed))
        extra = sorted(set(observed) - set(expected))
        raise FormalProvenanceError(
            f"source manifest is not the exact reviewed source set; missing={missing}, extra={extra}"
        )
    return value


def validate_reviewed_scope(
    path: Path,
    *,
    repo_root: Path,
    source_manifest: Mapping[str, Any],
    required_reviewed_paths: Sequence[Path],
    git_head: str,
    protocol_sha256: str,
    config_sha256: str,
) -> None:
    """Require the reviewed scope to cover every current signed source byte."""

    scope = load_json_mapping_strict(path, label="reviewed scope")
    required = {
        "schema_version",
        "status",
        "git_head",
        "protocol_sha256",
        "config_sha256",
        "sources",
        "scope_sha256",
    }
    if set(scope) != required:
        raise FormalProvenanceError("reviewed scope exact schema mismatch")
    validate_self_hash(scope, field="scope_sha256")
    if (
        scope.get("schema_version") != "ric-reviewed-worktree-v1"
        or scope.get("status") != "REVIEWED"
        or scope.get("git_head") != git_head
        or scope.get("protocol_sha256") != protocol_sha256
        or scope.get("config_sha256") != config_sha256
    ):
        raise FormalProvenanceError("reviewed scope identity mismatch")
    rows = scope.get("sources")
    if not isinstance(rows, list):
        raise FormalProvenanceError("reviewed scope sources must be a list")
    reviewed: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != {"path", "sha256"}:
            raise FormalProvenanceError("reviewed scope source row mismatch")
        source_path = row.get("path")
        source_sha = row.get("sha256")
        if (
            not isinstance(source_path, str)
            or not is_sha256(source_sha)
            or source_path in reviewed
        ):
            raise FormalProvenanceError("reviewed scope source identity mismatch")
        canonical = _canonical_relative_path(source_path).as_posix()
        if canonical != source_path:
            raise FormalProvenanceError("reviewed scope path is not canonical")
        current_path = resolve_repo_file(repo_root, canonical)
        current_sha = sha256_file(current_path)
        if current_sha != source_sha:
            raise FormalProvenanceError(
                f"reviewed scope source is stale: {canonical}"
            )
        reviewed[canonical] = source_sha
    manifest_rows = source_manifest.get("sources")
    if not isinstance(manifest_rows, list):
        raise FormalProvenanceError("source manifest sources are missing")
    for row in manifest_rows:
        if reviewed.get(str(row["path"])) != row["sha256"]:
            raise FormalProvenanceError(
                f"source was not reviewed at current hash: {row['path']}"
            )
    expected_reviewed = {
        repo_relative_file(repo_root, source): sha256_file(source.resolve(strict=True))
        for source in required_reviewed_paths
    }
    if reviewed != expected_reviewed:
        missing = sorted(set(expected_reviewed) - set(reviewed))
        extra = sorted(set(reviewed) - set(expected_reviewed))
        raise FormalProvenanceError(
            "reviewed scope is not the exact code/test/protocol/config universe; "
            f"missing={missing}, extra={extra}"
        )


def canonical_reviewed_scope_paths(
    repo_root: Path, stage_source_paths: Sequence[Path]
) -> tuple[Path, ...]:
    """Expand any RIC runtime stage to the exact Phase-4 review universe."""

    resolved_stage = tuple(path.resolve(strict=True) for path in stage_source_paths)
    this_dir = Path(__file__).resolve().parent
    if any(path.parent == this_dir for path in resolved_stage):
        idea_root = this_dir.parents[1]
        paths = [
            *this_dir.glob("*.py"),
            idea_root / "RIC_Phase2_冻结实验协议_2026-07-22.md",
            idea_root / "configs" / "ric_v1.json",
        ]
    else:
        paths = list(resolved_stage)
    unique = {repo_relative_file(repo_root, path): path.resolve(strict=True) for path in paths}
    return tuple(unique[key] for key in sorted(unique))


def _verify_test_summary(value: object) -> None:
    if not isinstance(value, Mapping):
        raise FormalProvenanceError("test_summary must be an object")
    required = {"status", "total", "passed", "failed", "errors", "skipped"}
    if set(value) != required or value.get("status") != "PASS":
        raise FormalProvenanceError("test_summary schema/status mismatch")
    counts = {}
    for field in required - {"status"}:
        count = value.get(field)
        if type(count) is not int or count < 0:
            raise FormalProvenanceError(f"test_summary {field} must be a non-negative int")
        counts[field] = count
    if counts["failed"] != 0 or counts["errors"] != 0 or counts["passed"] <= 0:
        raise FormalProvenanceError("test_summary is not a passing run")
    if counts["total"] != (
        counts["passed"]
        + counts["failed"]
        + counts["errors"]
        + counts["skipped"]
    ):
        raise FormalProvenanceError("test_summary total is inconsistent")


def _machine_readable_report_fields(path: Path, *, label: str) -> Mapping[str, str]:
    """Parse canonical ``UPPER_KEY: value`` report lines fail-closed."""

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise FormalProvenanceError(f"cannot read {label}") from exc
    fields: dict[str, str] = {}
    for line in lines:
        if ": " not in line:
            continue
        key, value = line.split(": ", 1)
        if (
            not key
            or any(character not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ_0123456789" for character in key)
            or not value
        ):
            continue
        if key in fields:
            raise FormalProvenanceError(f"duplicate machine-readable field in {label}: {key}")
        fields[key] = value
    return fields


def _verify_review_and_test_reports(
    *,
    review_report_path: Path,
    test_report_path: Path,
    test_summary: Mapping[str, Any],
    reviewed_scope_sha256: str,
) -> None:
    review = _machine_readable_report_fields(
        review_report_path, label="review report"
    )
    if review.get("STATUS") != "SIGNED-OFF" or review.get("OPEN_P0") != "0":
        raise FormalProvenanceError(
            "review report lacks STATUS: SIGNED-OFF / OPEN_P0: 0"
        )
    report = _machine_readable_report_fields(test_report_path, label="test report")
    if report.get("STATUS") != "PASS":
        raise FormalProvenanceError("test report lacks STATUS: PASS")
    if report.get("REVIEWED_SCOPE_SHA256") != reviewed_scope_sha256:
        raise FormalProvenanceError("test report/reviewed scope hash mismatch")
    report_names = {
        "TOTAL": "total",
        "PASSED": "passed",
        "FAILED": "failed",
        "ERRORS": "errors",
        "SKIPPED": "skipped",
    }
    for report_field, summary_field in report_names.items():
        supplied = report.get(report_field)
        if supplied is None or not supplied.isascii() or not supplied.isdecimal():
            raise FormalProvenanceError(
                f"test report lacks canonical count: {report_field}"
            )
        # Canonical decimal spelling forbids +0, whitespace and leading zeroes.
        if str(int(supplied)) != supplied:
            raise FormalProvenanceError(
                f"test report has non-canonical count: {report_field}"
            )
        if int(supplied) != test_summary.get(summary_field):
            raise FormalProvenanceError(
                f"test report/test_summary count mismatch: {report_field}"
            )


def build_source_manifest_payload(
    *,
    repo_root: Path,
    source_paths: Sequence[Path],
    git_head: str,
    worktree_patch_sha256: str,
) -> dict[str, Any]:
    """Build, but do not write, one canonical source-manifest payload."""

    if not source_paths:
        raise FormalProvenanceError("cannot build an empty source manifest")
    if not is_sha256(worktree_patch_sha256):
        raise FormalProvenanceError("reviewed patch hash is not SHA-256")
    if (
        not isinstance(git_head, str)
        or len(git_head) != 40
        or any(character not in "0123456789abcdef" for character in git_head)
    ):
        raise FormalProvenanceError("git_head is not a full SHA-1")
    rows = sorted(
        (
            {
                "path": repo_relative_file(repo_root, source),
                "sha256": sha256_file(source.resolve(strict=True)),
            }
            for source in source_paths
        ),
        key=lambda row: row["path"],
    )
    if len({row["path"] for row in rows}) != len(rows):
        raise FormalProvenanceError("duplicate source path requested")
    return add_self_hash(
        {
            "schema_version": SOURCE_MANIFEST_SCHEMA,
            "git_head": git_head,
            "worktree_patch_sha256": worktree_patch_sha256,
            "sources": rows,
        }
    )


def build_phase4_signoff_payload(
    *,
    repo_root: Path,
    stage: str,
    expected_fields: Mapping[str, Any],
    artifact_paths: Mapping[str, Path],
    git_head: str,
    test_summary: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a canonical self-hashed signoff after its evidence files exist."""

    if not isinstance(stage, str) or not stage:
        raise FormalProvenanceError("signoff stage must be non-empty")
    if set(artifact_paths) != set(SIGNOFF_ARTIFACT_FIELDS):
        raise FormalProvenanceError("signoff builder requires the exact evidence artifact set")
    _verify_test_summary(test_summary)
    _verify_git_head_binding(
        repo_root=repo_root.resolve(strict=True),
        signed_git_head=git_head,
        artifact_path=artifact_paths["git_head_artifact"].resolve(strict=True),
    )
    _verify_review_and_test_reports(
        review_report_path=artifact_paths["review_report"].resolve(strict=True),
        test_report_path=artifact_paths["test_report"].resolve(strict=True),
        test_summary=test_summary,
        reviewed_scope_sha256=sha256_file(
            artifact_paths["reviewed_patch"].resolve(strict=True)
        ),
    )
    references = {
        field: {
            "path": repo_relative_file(repo_root, artifact_paths[field]),
            "sha256": sha256_file(artifact_paths[field].resolve(strict=True)),
        }
        for field in SIGNOFF_ARTIFACT_FIELDS
    }
    patch_hash = references["reviewed_patch"]["sha256"]
    reserved = {
        "schema_version",
        "status",
        "open_p0",
        "stage",
        "git_head",
        "worktree_patch_sha256",
        "test_summary",
        "signoff_sha256",
        *SIGNOFF_ARTIFACT_FIELDS,
    }
    overlap = reserved & set(expected_fields)
    if overlap:
        raise FormalProvenanceError(
            f"stage fields replace reserved signoff fields: {sorted(overlap)}"
        )
    payload: dict[str, Any] = {
        "schema_version": SIGNOFF_SCHEMA,
        "status": "SIGNED-OFF",
        "open_p0": 0,
        "stage": stage,
        "git_head": git_head,
        "worktree_patch_sha256": patch_hash,
        "test_summary": dict(test_summary),
        **references,
        **dict(expected_fields),
    }
    return add_self_hash(payload, field="signoff_sha256")


def verify_phase4_signoff(
    path: Path | None,
    *,
    repo_root: Path,
    expected_fields: Mapping[str, Any],
    required_source_paths: Sequence[Path],
    required_reviewed_scope_paths: Sequence[Path] | None = None,
) -> Mapping[str, Any]:
    """Verify one self-hashed Phase-4 signoff and its complete evidence chain."""

    if path is None:
        raise FormalProvenanceError("formal execution requires Phase-4 signoff")
    root = repo_root.resolve(strict=True)
    try:
        resolved_signoff = path.resolve(strict=True)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise FormalProvenanceError("signoff file is missing") from exc
    try:
        relative_signoff = resolved_signoff.relative_to(root).as_posix()
    except ValueError:
        if (
            path.name != EMBEDDED_PRODUCER_SIGNOFF
            or path.is_symlink()
            or not resolved_signoff.is_file()
        ):
            raise FormalProvenanceError(
                "external signoff must be a regular embedded producer copy"
            )
        signoff_path = resolved_signoff
    else:
        signoff_path = resolve_repo_file(root, relative_signoff)
    value = _load_json_mapping(signoff_path, label="Phase-4 signoff")
    if value.get("schema_version") != SIGNOFF_SCHEMA:
        raise FormalProvenanceError("Phase-4 signoff schema mismatch")
    validate_self_hash(value, field="signoff_sha256")
    if value.get("status") != "SIGNED-OFF" or type(value.get("open_p0")) is not int:
        raise FormalProvenanceError("Phase-4 signoff status/open_p0 mismatch")
    if value["open_p0"] != 0:
        raise FormalProvenanceError("Phase-4 signoff has open P0 defects")
    if not isinstance(value.get("stage"), str) or not value["stage"]:
        raise FormalProvenanceError("Phase-4 signoff lacks stage identity")
    for field, wanted in expected_fields.items():
        if value.get(field) != wanted:
            raise FormalProvenanceError(f"Phase-4 signoff mismatch for {field}")
    references = {
        field: _verify_artifact_reference(value, field=field, repo_root=root)
        for field in SIGNOFF_ARTIFACT_FIELDS
    }
    git_head = _verify_git_head_binding(
        repo_root=root,
        signed_git_head=value.get("git_head"),
        artifact_path=references["git_head_artifact"][0],
    )
    patch_hash = references["reviewed_patch"][1]
    if value.get("worktree_patch_sha256") != patch_hash:
        raise FormalProvenanceError("reviewed patch/worktree hash mismatch")
    _verify_test_summary(value.get("test_summary"))
    _verify_review_and_test_reports(
        review_report_path=references["review_report"][0],
        test_report_path=references["test_report"][0],
        test_summary=value["test_summary"],
        reviewed_scope_sha256=references["reviewed_patch"][1],
    )
    source_manifest = verify_source_manifest(
        references["source_manifest"][0],
        repo_root=root,
        required_source_paths=required_source_paths,
        expected_git_head=git_head,
        expected_worktree_patch_sha256=patch_hash,
    )
    validate_reviewed_scope(
        references["reviewed_patch"][0],
        repo_root=root,
        source_manifest=source_manifest,
        required_reviewed_paths=(
            tuple(required_reviewed_scope_paths)
            if required_reviewed_scope_paths is not None
            else canonical_reviewed_scope_paths(root, required_source_paths)
        ),
        git_head=git_head,
        protocol_sha256=str(value.get("protocol_sha256")),
        config_sha256=str(value.get("config_sha256")),
    )
    return value


def _model_revision_string(spec: Mapping[str, Any]) -> str:
    return f"{spec['repo_id']}@{spec['revision']}"


def expected_formal_dataset_identity(
    config: Mapping[str, Any], *, role: str
) -> Mapping[str, str]:
    """Parse the exact Phase-2-frozen data-preparation identity.

    Fingerprints are useful cache identities but are not sufficient evidence on
    their own.  Formal preparation and every downstream manifest consumer bind
    the producer interpreter, datasets version, immutable dataset revision, and
    the ordered full-window content SHA as one indivisible identity.
    """

    if role not in {"calibration", "sealed"}:
        raise FormalProvenanceError("invalid formal dataset identity role")
    data = config.get("data")
    if not isinstance(data, Mapping):
        raise FormalProvenanceError("config lacks data mapping")
    identity = data.get("formal_dataset_identity")
    required_identity_fields = {
        "producer",
        "python_environment",
        "datasets_library_version",
        "dataset_repo_id",
        "dataset_revision",
        "calibration",
        "sealed",
    }
    if not isinstance(identity, Mapping) or set(identity) != required_identity_fields:
        raise FormalProvenanceError(
            "config formal_dataset_identity exact schema mismatch"
        )
    if identity.get("producer") != "prepare_data.py":
        raise FormalProvenanceError("formal dataset producer identity mismatch")
    python_environment = identity.get("python_environment")
    if (
        not isinstance(python_environment, str)
        or not python_environment
        or Path(python_environment).is_absolute()
        or "\\" in python_environment
        or any(part in {"", ".", ".."} for part in PurePosixPath(python_environment).parts)
    ):
        raise FormalProvenanceError(
            "formal dataset python_environment must be canonical repo-relative"
        )
    for field in ("datasets_library_version", "dataset_repo_id"):
        if not isinstance(identity.get(field), str) or not identity[field]:
            raise FormalProvenanceError(f"formal dataset identity lacks {field}")
    revision = identity.get("dataset_revision")
    if (
        not isinstance(revision, str)
        or len(revision) != 40
        or any(character not in "0123456789abcdef" for character in revision)
    ):
        raise FormalProvenanceError("formal dataset revision is not a full SHA-1")
    role_identity = identity.get(role)
    required_role_fields = {
        "dataset_slice_fingerprint",
        "dataset_slice_canonical_content_sha256",
    }
    if not isinstance(role_identity, Mapping) or set(role_identity) != required_role_fields:
        raise FormalProvenanceError(
            f"formal dataset identity {role} exact schema mismatch"
        )
    fingerprint = role_identity.get("dataset_slice_fingerprint")
    if (
        not isinstance(fingerprint, str)
        or not fingerprint
        or any(character not in "0123456789abcdef" for character in fingerprint)
    ):
        raise FormalProvenanceError(
            f"formal dataset identity {role} fingerprint is invalid"
        )
    content_sha256 = role_identity.get(
        "dataset_slice_canonical_content_sha256"
    )
    if not is_sha256(content_sha256):
        raise FormalProvenanceError(
            f"formal dataset identity {role} canonical content SHA is invalid"
        )
    return {
        "producer": str(identity["producer"]),
        "python_environment": str(python_environment),
        "datasets_library_version": str(identity["datasets_library_version"]),
        "dataset_repo_id": str(identity["dataset_repo_id"]),
        "dataset_revision": str(revision),
        "dataset_slice_fingerprint": str(fingerprint),
        "dataset_slice_canonical_content_sha256": str(content_sha256),
    }


def _canonical_text(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def frozen_concat_sha256(*parts: object) -> str:
    """Frozen literal UTF-8 concatenation with no implicit delimiter."""

    if not parts:
        raise FormalProvenanceError("frozen hash requires at least one part")
    if any(isinstance(part, (bytes, bytearray)) for part in parts):
        raise FormalProvenanceError("frozen hash parts must use their canonical text form")
    return sha256_bytes("".join(str(part) for part in parts).encode("utf-8"))


def canonical_text_sequence_sha256(rows: Sequence[str]) -> str:
    """Hash an ordered full dataset slice after newline-only canonicalization."""

    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes, bytearray)):
        raise FormalProvenanceError("dataset slice rows must be a text sequence")
    canonical_rows = []
    for row in rows:
        if not isinstance(row, str):
            raise FormalProvenanceError("dataset slice contains a non-text row")
        canonical_rows.append(_canonical_text(row))
    return sha256_bytes(canonical_json_bytes(canonical_rows))


def validate_data_manifest_fields(
    value: Mapping[str, Any],
    *,
    mode: str,
    role: str,
    config: Mapping[str, Any],
    protocol_sha256: str,
    config_sha256: str,
    expected_prepare_data_source_sha256: str,
    expected_historical_registry_sha256: str | None = None,
    expected_dataset_slice_fingerprint: str | None = None,
    expected_dataset_slice_canonical_content_sha256: str | None = None,
    expected_selected_text_sha256: Sequence[str] | None = None,
    expected_calibration_lock_self_hash: str | None = None,
    expected_calibration_lock_file_sha256: str | None = None,
) -> Mapping[str, Any]:
    """Strictly validate the frozen data manifest for downstream producers."""

    if not is_sha256(expected_prepare_data_source_sha256):
        raise FormalProvenanceError("expected prepare-data source digest is invalid")
    if mode not in {"dev", "formal"} or role not in {"calibration", "sealed"}:
        raise FormalProvenanceError("invalid data manifest mode/role request")
    if mode == "dev" and role == "sealed":
        raise FormalProvenanceError("dev mode is forbidden from reading sealed data")
    formal_dataset_identity = (
        expected_formal_dataset_identity(config, role=role)
        if mode == "formal"
        else None
    )
    if value.get("schema_version") != DATA_MANIFEST_SCHEMA:
        raise FormalProvenanceError("data manifest schema mismatch")
    validate_self_hash(value)
    expected_status = "INPUT_ONLY" if mode == "formal" else "NOT_TESTED"
    frozen = {
        "status": expected_status,
        "scientific_result": False,
        "mode": mode,
        "role": role,
        "dataset_loader": "wikitext",
        "dataset_config": config["data"]["config"],
        "dataset_split": config["data"]["split"],
        "selection_seed": int(config["data"]["selection_seed"]),
        "selection_method": config["data"]["selection_method"],
        "sequence_tokens": int(config["data"]["sequence_length"]),
        "batch_size": int(config["data"]["batch_size"]),
        "padding_allowed": bool(config["data"]["padding_allowed"]),
        "protocol_sha256": protocol_sha256,
        "config_sha256": config_sha256,
        "prepare_data_source_sha256": expected_prepare_data_source_sha256,
        "model_revisions": {
            key: _model_revision_string(spec) for key, spec in config["models"].items()
        },
        "tokenizer_revisions": {
            key: _model_revision_string(spec) for key, spec in config["models"].items()
        },
    }
    for field, wanted in frozen.items():
        if value.get(field) != wanted:
            raise FormalProvenanceError(f"data manifest frozen field mismatch: {field}")
    if mode == "formal":
        if not is_sha256(value.get("signoff_sha256")):
            raise FormalProvenanceError("formal data manifest lacks producer signoff")
    elif value.get("signoff_sha256") is not None:
        raise FormalProvenanceError("development data cannot claim a producer signoff")
    for field in ("selection_seed", "sequence_tokens", "batch_size"):
        if type(value.get(field)) is not int:
            raise FormalProvenanceError(
                f"data manifest frozen integer field has wrong type: {field}"
            )
    if type(value.get("scientific_result")) is not bool:
        raise FormalProvenanceError("data manifest scientific_result must be exact bool")
    if type(value.get("padding_allowed")) is not bool:
        raise FormalProvenanceError("data manifest padding_allowed must be exact bool")
    role_cfg = config["data"][role]
    start = int(role_cfg["candidate_row_start_inclusive"])
    stop = int(role_cfg["candidate_row_end_exclusive"])
    if value.get("candidate_window") != [start, stop]:
        raise FormalProvenanceError("data manifest candidate window mismatch")
    if value.get("dataset_slice_row_count") != stop - start:
        raise FormalProvenanceError("data manifest slice row count mismatch")
    for field in (
        "dataset_slice_fingerprint",
        "datasets_library_version",
    ):
        if not isinstance(value.get(field), str) or not value[field]:
            raise FormalProvenanceError(f"data manifest lacks {field}")
    if not is_sha256(value.get("dataset_slice_canonical_content_sha256")):
        raise FormalProvenanceError("data manifest lacks canonical slice content hash")
    if not is_sha256(value.get("dataset_source_urls_sha256")):
        raise FormalProvenanceError("data manifest lacks dataset source-URL hash")
    if formal_dataset_identity is not None:
        formal_manifest_fields = {
            "data_preparation_producer": formal_dataset_identity["producer"],
            "data_preparation_python_environment": formal_dataset_identity[
                "python_environment"
            ],
            "datasets_library_version": formal_dataset_identity[
                "datasets_library_version"
            ],
            "dataset_repo_id": formal_dataset_identity["dataset_repo_id"],
            "dataset_revision": formal_dataset_identity["dataset_revision"],
            "dataset_slice_fingerprint": formal_dataset_identity[
                "dataset_slice_fingerprint"
            ],
            "dataset_slice_canonical_content_sha256": formal_dataset_identity[
                "dataset_slice_canonical_content_sha256"
            ],
        }
        for field, wanted in formal_manifest_fields.items():
            if value.get(field) != wanted:
                raise FormalProvenanceError(
                    f"data manifest formal dataset identity mismatch: {field}"
                )
    if (
        expected_dataset_slice_fingerprint is not None
        and value.get("dataset_slice_fingerprint")
        != expected_dataset_slice_fingerprint
    ):
        raise FormalProvenanceError("data manifest dataset fingerprint mismatch")
    if (
        expected_dataset_slice_canonical_content_sha256 is not None
        and value.get("dataset_slice_canonical_content_sha256")
        != expected_dataset_slice_canonical_content_sha256
    ):
        raise FormalProvenanceError("data manifest canonical slice hash mismatch")
    registry_sha = value.get("historical_exclusion_registry_sha256")
    if not is_sha256(registry_sha):
        raise FormalProvenanceError("data manifest lacks historical registry hash")
    if (
        expected_historical_registry_sha256 is not None
        and registry_sha != expected_historical_registry_sha256
    ):
        raise FormalProvenanceError("data manifest historical registry binding mismatch")
    requests = value.get("requests")
    expected_count = int(role_cfg["document_count"])
    if not isinstance(requests, list) or len(requests) != expected_count:
        raise FormalProvenanceError("data manifest request count mismatch")
    expected_models = set(config["models"])
    seed = int(config["data"]["selection_seed"])
    min_tokens = int(config["data"]["min_tokens_both_frozen_tokenizers"])
    identities: set[str] = set()
    hashes: set[str] = set()
    source_rows: set[int] = set()
    ordering: list[tuple[str, int]] = []
    selected_hashes: list[str] = []
    for request_index, row in enumerate(requests):
        if not isinstance(row, Mapping):
            raise FormalProvenanceError("data manifest request row is not an object")
        request_id = row.get("request_id")
        text = row.get("text")
        text_hash = row.get("text_sha256")
        source_row = row.get("source_row")
        rank_hash = row.get("rank_sha256")
        lengths = row.get("token_lengths")
        if not isinstance(request_id, str):
            raise FormalProvenanceError("data manifest request identity mismatch")
        if request_id in identities:
            raise FormalProvenanceError("data manifest request identity duplicated")
        identities.add(request_id)
        if (
            not isinstance(text, str)
            or sha256_bytes(_canonical_text(text).encode("utf-8")) != text_hash
        ):
            raise FormalProvenanceError("data manifest request text hash mismatch")
        if not is_sha256(text_hash) or text_hash in hashes:
            raise FormalProvenanceError("data manifest request text hash invalid/duplicated")
        hashes.add(text_hash)
        if type(source_row) is not int or not start <= source_row < stop:
            raise FormalProvenanceError("data manifest source row outside frozen window")
        if source_row in source_rows:
            raise FormalProvenanceError("data manifest source row duplicated")
        source_rows.add(source_row)
        if rank_hash != frozen_concat_sha256(seed, text_hash):
            raise FormalProvenanceError("data manifest rank hash mismatch")
        if not isinstance(lengths, Mapping) or set(lengths) != expected_models:
            raise FormalProvenanceError("data manifest tokenizer-length keys mismatch")
        if any(
            type(lengths[key]) is not int or lengths[key] < min_tokens
            for key in expected_models
        ):
            raise FormalProvenanceError("data manifest tokenizer length below frozen minimum")
        expected_request_id = f"ric:{role}:{request_index:04d}:{text_hash[:12]}"
        if request_id != expected_request_id:
            raise FormalProvenanceError("data manifest request identity mismatch")
        ordering.append((rank_hash, source_row))
        selected_hashes.append(text_hash)
    if ordering != sorted(ordering):
        raise FormalProvenanceError("data manifest requests are not in frozen rank order")
    if value.get("selected_text_sha256") != selected_hashes:
        raise FormalProvenanceError("data manifest selected-hash binding mismatch")
    if (
        expected_selected_text_sha256 is not None
        and selected_hashes != list(expected_selected_text_sha256)
    ):
        raise FormalProvenanceError("data manifest selected hashes differ from expected")
    if mode == "formal" and role == "sealed":
        required_sealed_bindings = {
            "sealed_reservation_sha256": "reservation",
            "sealed_nonce_sha256": "nonce",
            "calibration_manifest_self_hash": "calibration manifest self-hash",
            "calibration_manifest_file_sha256": "calibration manifest file hash",
            "calibration_selected_list_sha256": "calibration selected-text-list hash",
            "calibration_lock_self_hash": "calibration lock self-hash",
            "calibration_lock_file_sha256": "calibration lock file hash",
        }
        for field, label in required_sealed_bindings.items():
            if not is_sha256(value.get(field)):
                raise FormalProvenanceError(
                    f"sealed formal manifest lacks {label} binding"
                )
        for field, wanted in (
            ("calibration_lock_self_hash", expected_calibration_lock_self_hash),
            ("calibration_lock_file_sha256", expected_calibration_lock_file_sha256),
        ):
            if wanted is not None and value.get(field) != wanted:
                raise FormalProvenanceError(
                    f"sealed formal manifest calibration-lock mismatch: {field}"
                )
    return value
