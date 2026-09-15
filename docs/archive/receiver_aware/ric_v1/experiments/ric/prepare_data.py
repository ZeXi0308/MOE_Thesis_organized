#!/usr/bin/env python3
"""Prepare fail-closed RIC-v1 calibration or sealed WikiText manifests.

The producer performs the frozen window selection with both frozen model
tokenizers and writes the historical-exclusion registry used for the check.
It does not capture routes or compute any scientific metric.  A dev process
is intentionally unable to request the sealed role.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import secrets
import sys
import tempfile
from typing import Any, Callable, Iterable, Mapping, Sequence

try:  # Package import.
    from .formal_provenance import (
        EMBEDDED_PRODUCER_SIGNOFF,
        FormalProvenanceError,
        canonical_text_sequence_sha256,
        expected_formal_dataset_identity,
        frozen_concat_sha256 as _frozen_concat_sha256,
        is_sha256,
        load_json_mapping_strict,
        materialize_verified_signoff,
        validate_calibration_lock_fields,
        validate_data_manifest_fields,
        verify_calibration_lock_producer_signoff,
        verify_phase4_signoff,
    )
except ImportError:  # Direct entrypoint/tests from this directory.
    from formal_provenance import (  # type: ignore
        EMBEDDED_PRODUCER_SIGNOFF,
        FormalProvenanceError,
        canonical_text_sequence_sha256,
        expected_formal_dataset_identity,
        frozen_concat_sha256 as _frozen_concat_sha256,
        is_sha256,
        load_json_mapping_strict,
        materialize_verified_signoff,
        validate_calibration_lock_fields,
        validate_data_manifest_fields,
        verify_calibration_lock_producer_signoff,
        verify_phase4_signoff,
    )


HERE = Path(__file__).resolve().parent
REPO_ROOT = next(candidate for candidate in HERE.parents if (candidate / "experiments/shared").is_dir())
IDEA_ROOT = HERE.parents[1]
DEFAULT_CONFIG = IDEA_ROOT / "configs" / "ric_v1.json"
DEFAULT_PROTOCOL = IDEA_ROOT / "RIC_Phase2_冻结实验协议_2026-07-22.md"
DEFAULT_SEALED_LEDGER_DIR = IDEA_ROOT / ".formal_state" / "ric_v1_sealed"

TEXT_HASH_FIELDS = frozenset(
    {
        "text_sha256",
        "canonical_text_sha256",
        "request_text_sha256",
    }
)


class DataPreparationError(RuntimeError):
    """A frozen data/provenance invariant failed."""


@dataclass(frozen=True)
class LoadedDatasetSlice:
    rows: tuple[str, ...]
    dataset_repo_id: str
    dataset_revision: str
    dataset_source_urls_sha256: str
    datasets_library_version: str
    dataset_slice_fingerprint: str
    canonical_content_sha256: str


@dataclass(frozen=True)
class CalibrationManifestBinding:
    manifest_sha256: str
    file_sha256: str
    selected_text_sha256: tuple[str, ...]
    request_ids: tuple[str, ...]


@dataclass(frozen=True)
class CalibrationLockBinding:
    manifest_sha256: str
    file_sha256: str
    run_experiment_source_sha256: str
    scenario_tree_sha256: Mapping[str, str]


RUN_EXPERIMENT_SOURCE_PATHS = (
    HERE / "__init__.py",
    HERE / "run_experiment.py",
    HERE / "run_oracle.py",
    HERE / "oracle.py",
    HERE / "build_scenarios.py",
    HERE / "scenario.py",
    HERE / "replay.py",
    HERE / "accounting.py",
    HERE / "wire.py",
    HERE / "policy_views.py",
    HERE / "schema.py",
    HERE / "measure_capability_gpu.py",
    HERE / "measure_service_lut_gpu.py",
    HERE / "capture_routes_gpu.py",
    HERE / "prepare_data.py",
    HERE / "formal_provenance.py",
    HERE / "capability_contract.py",
)
RUN_EXPERIMENT_SIGNOFF_SOURCE_PATHS = RUN_EXPERIMENT_SOURCE_PATHS


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


def frozen_concat_sha256(*parts: object) -> str:
    """Expose the protocol's literal, delimiter-free UTF-8 hash preimage."""

    try:
        return _frozen_concat_sha256(*parts)
    except FormalProvenanceError as exc:
        raise DataPreparationError(str(exc)) from exc


def canonical_text(text: str) -> str:
    """Canonicalize transport newlines only; do not silently strip content."""

    return text.replace("\r\n", "\n").replace("\r", "\n")


def _strict_self_hash_json_object(
    pairs: Sequence[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DataPreparationError(
                f"self-hashed payload has duplicate JSON object key {key!r}"
            )
        result[key] = value
    return result


def add_self_hash(payload: Mapping[str, Any], field: str = "manifest_sha256") -> dict[str, Any]:
    if field in payload:
        raise DataPreparationError(f"refusing to replace existing {field}")
    try:
        normalized = json.loads(
            json.dumps(dict(payload), ensure_ascii=False, allow_nan=False),
            object_pairs_hook=_strict_self_hash_json_object,
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise DataPreparationError("self-hashed payload is not strict JSON") from exc
    if not isinstance(normalized, dict):
        raise DataPreparationError("self-hashed payload must be a JSON object")
    result = normalized
    result[field] = sha256_bytes(canonical_json_bytes(result))
    return result


def validate_self_hash(payload: Mapping[str, Any], field: str = "manifest_sha256") -> str:
    try:
        normalized = json.loads(
            json.dumps(dict(payload), ensure_ascii=False, allow_nan=False),
            object_pairs_hook=_strict_self_hash_json_object,
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise DataPreparationError("self-hashed payload is not strict JSON") from exc
    if not isinstance(normalized, dict):
        raise DataPreparationError("self-hashed payload must be a JSON object")
    supplied = normalized.get(field)
    unhashed = dict(normalized)
    unhashed.pop(field, None)
    actual = sha256_bytes(canonical_json_bytes(unhashed))
    if supplied != actual:
        raise DataPreparationError(f"{field} mismatch")
    return actual


def _iter_text_hashes(value: object) -> Iterable[str]:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key in TEXT_HASH_FIELDS and isinstance(item, str):
                if len(item) == 64 and all(char in "0123456789abcdef" for char in item):
                    yield item
            else:
                yield from _iter_text_hashes(item)
    elif isinstance(value, list):
        for item in value:
            yield from _iter_text_hashes(item)


def _looks_like_explicit_data_manifest(path: Path, value: object | None) -> bool:
    name = path.name.lower().replace("-", "_")
    if "data_manifest" in name:
        return True
    if not isinstance(value, Mapping):
        return False
    schema = str(value.get("schema_version", "")).lower().replace("_", "-")
    return "data-manifest" in schema


def validate_formal_historical_scan_root(scan_root: Path) -> Path:
    """Require the one frozen repository-wide historical evidence root."""

    expected = (REPO_ROOT / "docs").resolve(strict=True)
    if scan_root.is_symlink():
        raise DataPreparationError("formal historical scan root cannot be a symlink")
    try:
        resolved = scan_root.resolve(strict=True)
    except (FileNotFoundError, RuntimeError) as exc:
        raise DataPreparationError("formal historical scan root does not exist") from exc
    if resolved != expected or not resolved.is_dir():
        raise DataPreparationError(
            "formal historical scan root must be the canonical repository docs directory"
        )
    return resolved


def build_historical_registry(
    scan_root: Path,
    *,
    excluded_paths: Sequence[Path] = (),
    formal: bool = False,
) -> dict[str, Any]:
    """Scan existing JSON evidence for canonical request/text hashes.

    Only explicit text-hash fields count.  Generic SHA fields (source hashes,
    code hashes, output hashes) must not accidentally exclude a prompt.
    Unreadable JSON is ignored but recorded; a formal caller rejects any such
    parse failure rather than treating the scan as complete.
    """

    scan_root = validate_formal_historical_scan_root(scan_root) if formal else scan_root
    if not scan_root.exists() or not scan_root.is_dir():
        raise DataPreparationError("historical scan root is missing or not a directory")
    excluded = {path.resolve() for path in excluded_paths}

    def path_label(path: Path) -> str:
        if formal:
            return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
        return str(path)

    hashes: set[str] = set()
    sources: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for path in sorted(scan_root.rglob("*.json")):
        resolved = path.resolve()
        if any(resolved == root or root in resolved.parents for root in excluded):
            continue
        if ".git" in path.parts:
            continue
        explicit_by_name = _looks_like_explicit_data_manifest(path, None)
        if formal:
            try:
                resolved.relative_to(scan_root)
            except ValueError as exc:
                raise DataPreparationError(
                    "historical manifest resolves outside canonical scan root"
                ) from exc
            if path.is_symlink() or resolved != path:
                failures.append(
                    {
                        "path": path_label(path),
                        "error": "symlink/non-canonical manifest path",
                    }
                )
                continue
        try:
            value = load_json_mapping_strict(path, label=path_label(path))
        except (OSError, UnicodeError, json.JSONDecodeError, FormalProvenanceError) as exc:
            if explicit_by_name:
                failures.append(
                    {
                        "path": path_label(path),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
            continue
        found = sorted(set(_iter_text_hashes(value)))
        explicit_manifest = _looks_like_explicit_data_manifest(path, value)
        if explicit_manifest and not found:
            failures.append(
                {
                    "path": path_label(path),
                    "error": "explicit data manifest has no recognized canonical text hash",
                }
            )
            continue
        if not found:
            continue
        hashes.update(found)
        try:
            relative = str(path.resolve().relative_to(REPO_ROOT.resolve()))
        except ValueError:
            relative = str(path.resolve())
        sources.append(
            {
                "path": relative,
                "file_sha256": sha256_file(path),
                "text_hash_count": len(found),
            }
        )
    payload = {
        "schema_version": "ric-historical-exclusion-v1",
        "scan_root": (
            scan_root.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
            if formal
            else str(scan_root.resolve())
        ),
        "scan_policy": "canonical_repo_docs_explicit_data_manifests_and_recognized_text_hashes",
        "recognized_text_hash_fields": sorted(TEXT_HASH_FIELDS),
        "source_files": sources,
        "source_file_count": len(sources),
        "text_hashes": sorted(hashes),
        "text_hash_count": len(hashes),
        "parse_failures": failures,
        "complete": not failures,
    }
    return add_self_hash(payload, "registry_sha256")


def _model_revision_string(spec: Mapping[str, Any]) -> str:
    return f"{spec['repo_id']}@{spec['revision']}"


def select_requests(
    rows: Sequence[str],
    *,
    source_row_start: int,
    required_count: int,
    selection_seed: int,
    min_tokens: int,
    token_lengths: Callable[[str], Mapping[str, int]],
    historical_hashes: set[str],
    role: str,
) -> list[dict[str, Any]]:
    """Apply the exact frozen selection; never skip a selected collision."""

    candidates: list[dict[str, Any]] = []
    for offset, raw_text in enumerate(rows):
        text = canonical_text(str(raw_text))
        text_hash = sha256_bytes(text.encode("utf-8"))
        lengths = {str(key): int(value) for key, value in token_lengths(text).items()}
        if not lengths or min(lengths.values()) < min_tokens:
            continue
        rank_hash = frozen_concat_sha256(selection_seed, text_hash)
        candidates.append(
            {
                "rank_sha256": rank_hash,
                "source_row": source_row_start + offset,
                "text_sha256": text_hash,
                "token_lengths": lengths,
                "text": text,
            }
        )
    selected = sorted(
        candidates,
        key=lambda row: (str(row["rank_sha256"]), int(row["source_row"])),
    )[:required_count]
    if len(selected) != required_count:
        raise DataPreparationError(
            f"frozen window has {len(selected)} dual-tokenizer-valid rows; "
            f"required {required_count}; automatic window fallback is forbidden"
        )
    selected_hashes = [str(row["text_sha256"]) for row in selected]
    duplicates = sorted(
        value for value in set(selected_hashes) if selected_hashes.count(value) > 1
    )
    if duplicates:
        raise DataPreparationError(
            f"selected manifest has duplicate canonical texts: {duplicates[:3]}"
        )
    collisions = sorted(set(selected_hashes) & historical_hashes)
    if collisions:
        raise DataPreparationError(
            "frozen selected rows collide with historical evidence; Phase-2 amendment "
            f"is required before any automatic fallback: {collisions[:3]}"
        )
    requests: list[dict[str, Any]] = []
    for index, row in enumerate(selected):
        requests.append(
            {
                **row,
                "request_id": (
                    f"ric:{role}:{index:04d}:{str(row['text_sha256'])[:12]}"
                ),
            }
        )
    return requests


def _validate_mode_role(mode: str, role: str) -> None:
    if mode == "dev" and role == "sealed":
        raise DataPreparationError("dev mode is forbidden from opening sealed data")


def _validate_calibration_manifest_argument(
    *, mode: str, role: str, path: Path | None
) -> None:
    is_formal_sealed = mode == "formal" and role == "sealed"
    if is_formal_sealed and path is None:
        raise DataPreparationError(
            "formal sealed preparation requires --calibration-manifest"
        )
    if not is_formal_sealed and path is not None:
        raise DataPreparationError(
            "--calibration-manifest is accepted only for formal sealed preparation"
        )


def _validate_calibration_lock_argument(
    *, mode: str, role: str, path: Path | None
) -> None:
    is_formal_sealed = mode == "formal" and role == "sealed"
    if is_formal_sealed and path is None:
        raise DataPreparationError(
            "formal sealed preparation requires --calibration-lock"
        )
    if not is_formal_sealed and path is not None:
        raise DataPreparationError(
            "--calibration-lock is accepted only for formal sealed preparation"
        )


def _load_config(path: Path) -> Mapping[str, Any]:
    try:
        value = load_json_mapping_strict(path, label="RIC config")
    except FormalProvenanceError as exc:
        raise DataPreparationError(str(exc)) from exc
    if value.get("schema_version") != "ric-config-v1":
        raise DataPreparationError("not a RIC-v1 config")
    if value.get("status") != "PHASE2_FROZEN_NO_SCIENTIFIC_RESULT":
        raise DataPreparationError("RIC config is not frozen")
    return value


def _formal_dataset_identity(
    config: Mapping[str, Any], *, role: str
) -> Mapping[str, str]:
    try:
        return expected_formal_dataset_identity(config, role=role)
    except FormalProvenanceError as exc:
        raise DataPreparationError(str(exc)) from exc


def validate_formal_data_preparation_environment(
    config: Mapping[str, Any], *, role: str
) -> Mapping[str, str]:
    """Require formal preparation to use the Phase-2-frozen interpreter."""

    expected = _formal_dataset_identity(config, role=role)
    expected_interpreter = REPO_ROOT / expected["python_environment"]
    expected_absolute = expected_interpreter.absolute()
    current_absolute = Path(sys.executable).absolute()
    if not expected_interpreter.exists() or not expected_interpreter.is_file():
        raise DataPreparationError(
            "formal data-preparation interpreter is missing from repository"
        )
    # Do not resolve the venv symlink: resolving it would incorrectly allow the
    # underlying system interpreter to impersonate the frozen venv entrypoint.
    if current_absolute != expected_absolute:
        raise DataPreparationError(
            "formal data preparation is running under the wrong interpreter"
        )
    return expected


def validate_loaded_dataset_identity(
    dataset_slice: LoadedDatasetSlice,
    *,
    config: Mapping[str, Any],
    role: str,
) -> None:
    """Compare all observed data identities with the Phase-2 frozen values."""

    expected = _formal_dataset_identity(config, role=role)
    observed = {
        "dataset_repo_id": dataset_slice.dataset_repo_id,
        "dataset_revision": dataset_slice.dataset_revision,
        "datasets_library_version": dataset_slice.datasets_library_version,
        "dataset_slice_fingerprint": dataset_slice.dataset_slice_fingerprint,
        "dataset_slice_canonical_content_sha256": (
            dataset_slice.canonical_content_sha256
        ),
    }
    for field, actual in observed.items():
        if actual != expected[field]:
            raise DataPreparationError(
                f"formal loaded dataset identity mismatch: {field}"
            )


def _source_closure_sha256(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for source in paths:
        digest.update(
            str(source.resolve().relative_to(REPO_ROOT.resolve())).encode("utf-8")
        )
        digest.update(b"\0")
        digest.update(source.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _run_experiment_source_sha256() -> str:
    """Reproduce the reviewed calibration producer's source-closure digest."""

    return _source_closure_sha256(RUN_EXPERIMENT_SOURCE_PATHS)


def load_formal_calibration_lock(
    path: Path | None,
    *,
    config: Mapping[str, Any],
    protocol_sha256: str,
    config_sha256: str,
) -> CalibrationLockBinding:
    """Validate the exact G1-passing formal lock before any sealed access."""

    if path is None:
        raise DataPreparationError(
            "formal sealed preparation requires --calibration-lock"
        )
    if path.is_symlink():
        raise DataPreparationError("calibration lock cannot be a symlink")
    try:
        resolved = path.resolve(strict=True)
    except (FileNotFoundError, RuntimeError) as exc:
        raise DataPreparationError("calibration lock is missing") from exc
    if not resolved.is_file():
        raise DataPreparationError("calibration lock must be a regular file")
    try:
        lock = load_json_mapping_strict(resolved, label="calibration lock")
        validate_calibration_lock_fields(
            lock,
            config=config,
            protocol_sha256=protocol_sha256,
            config_sha256=config_sha256,
            expected_run_experiment_source_sha256=(
                _run_experiment_source_sha256()
            ),
        )
        verify_calibration_lock_producer_signoff(
            resolved,
            lock,
            repo_root=REPO_ROOT,
            required_source_paths=RUN_EXPERIMENT_SOURCE_PATHS,
        )
    except FormalProvenanceError as exc:
        raise DataPreparationError(
            f"invalid formal calibration lock: {exc}"
        ) from exc
    scenario_hashes = lock.get("scenario_tree_sha256")
    assert isinstance(scenario_hashes, Mapping)
    return CalibrationLockBinding(
        manifest_sha256=str(lock["manifest_sha256"]),
        file_sha256=sha256_file(resolved),
        run_experiment_source_sha256=str(lock["run_experiment_source_sha256"]),
        scenario_tree_sha256={
            str(model): str(digest) for model, digest in scenario_hashes.items()
        },
    )


def load_formal_calibration_manifest(
    path: Path | None,
    *,
    config: Mapping[str, Any],
    protocol_sha256: str,
    config_sha256: str,
    producer_source_sha256: str,
) -> CalibrationManifestBinding:
    """Validate and bind this run's calibration selection before sealed access."""

    if path is None:
        raise DataPreparationError(
            "formal sealed preparation requires --calibration-manifest"
        )
    if path.is_symlink():
        raise DataPreparationError("calibration manifest cannot be a symlink")
    try:
        resolved = path.resolve(strict=True)
    except (FileNotFoundError, RuntimeError) as exc:
        raise DataPreparationError("calibration manifest is missing") from exc
    if not resolved.is_file():
        raise DataPreparationError("calibration manifest must be a regular file")
    try:
        manifest = load_json_mapping_strict(
            resolved, label="calibration manifest"
        )
    except FormalProvenanceError as exc:
        raise DataPreparationError(f"cannot read calibration manifest: {exc}") from exc
    try:
        validate_data_manifest_fields(
            manifest,
            mode="formal",
            role="calibration",
            config=config,
            protocol_sha256=protocol_sha256,
            config_sha256=config_sha256,
            expected_prepare_data_source_sha256=producer_source_sha256,
        )
    except FormalProvenanceError as exc:
        raise DataPreparationError(
            f"invalid formal calibration manifest: {exc}"
        ) from exc
    verify_embedded_formal_signoff(
        resolved,
        manifest,
        protocol_sha256=protocol_sha256,
        config_sha256=config_sha256,
    )
    selected = manifest.get("selected_text_sha256")
    requests = manifest.get("requests")
    assert isinstance(selected, list)  # Enforced by the shared strict validator.
    assert isinstance(requests, list)  # Enforced by the shared strict validator.
    return CalibrationManifestBinding(
        manifest_sha256=str(manifest["manifest_sha256"]),
        file_sha256=sha256_file(resolved),
        selected_text_sha256=tuple(str(value) for value in selected),
        request_ids=tuple(str(row["request_id"]) for row in requests),
    )


def _producer_source_sha256() -> str:
    digest = hashlib.sha256()
    for path in (
        Path(__file__),
        HERE / "formal_provenance.py",
        HERE / "test_provenance.py",
    ):
        digest.update(str(path.resolve().relative_to(REPO_ROOT.resolve())).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def require_formal_signoff(
    path: Path | None,
    *,
    protocol_sha256: str,
    config_sha256: str,
    producer_source_sha256: str,
    data_role: str,
    calibration_lock_binding: CalibrationLockBinding | None = None,
) -> Mapping[str, Any]:
    expected_fields: dict[str, Any] = {
        "stage": "prepare_data",
        "protocol_sha256": protocol_sha256,
        "config_sha256": config_sha256,
        "prepare_data_source_sha256": producer_source_sha256,
        "data_role": data_role,
    }
    required_sources = [
        Path(__file__),
        HERE / "formal_provenance.py",
        HERE / "test_provenance.py",
    ]
    if data_role == "sealed":
        if calibration_lock_binding is None:
            raise DataPreparationError(
                "sealed formal signoff requires validated calibration lock"
            )
        expected_fields.update(
            {
                "calibration_lock_sha256": (
                    calibration_lock_binding.manifest_sha256
                ),
                "calibration_lock_file_sha256": (
                    calibration_lock_binding.file_sha256
                ),
                "run_experiment_source_sha256": (
                    calibration_lock_binding.run_experiment_source_sha256
                ),
                "scenario_tree_sha256": dict(
                    calibration_lock_binding.scenario_tree_sha256
                ),
            }
        )
        required_sources.extend(RUN_EXPERIMENT_SIGNOFF_SOURCE_PATHS)
    try:
        return verify_phase4_signoff(
            path,
            repo_root=REPO_ROOT,
            expected_fields=expected_fields,
            required_source_paths=tuple(dict.fromkeys(required_sources)),
        )
    except FormalProvenanceError as exc:
        raise DataPreparationError(str(exc)) from exc


def verify_embedded_formal_signoff(
    manifest_path: Path,
    manifest: Mapping[str, Any],
    *,
    protocol_sha256: str,
    config_sha256: str,
) -> Mapping[str, Any]:
    """Re-verify the prepare-data producer attestation from its output tree."""

    role = str(manifest.get("role"))
    if manifest.get("mode") != "formal" or role not in {"calibration", "sealed"}:
        raise DataPreparationError("embedded data signoff requires formal data")
    signoff_path = manifest_path.parent / EMBEDDED_PRODUCER_SIGNOFF
    expected_file_sha256 = manifest.get("signoff_sha256")
    if not is_sha256(expected_file_sha256) or not signoff_path.is_file():
        raise DataPreparationError("formal data lacks embedded producer signoff")
    if sha256_file(signoff_path) != expected_file_sha256:
        raise DataPreparationError("embedded data producer signoff hash mismatch")
    expected_fields: dict[str, Any] = {
        "stage": "prepare_data",
        "protocol_sha256": protocol_sha256,
        "config_sha256": config_sha256,
        "prepare_data_source_sha256": _producer_source_sha256(),
        "data_role": role,
    }
    required_sources = [Path(__file__), HERE / "formal_provenance.py", HERE / "test_provenance.py"]
    if role == "sealed":
        expected_fields.update(
            {
                "calibration_lock_sha256": manifest.get("calibration_lock_self_hash"),
                "calibration_lock_file_sha256": manifest.get("calibration_lock_file_sha256"),
                "run_experiment_source_sha256": _run_experiment_source_sha256(),
            }
        )
        required_sources.extend(RUN_EXPERIMENT_SIGNOFF_SOURCE_PATHS)
    try:
        return verify_phase4_signoff(
            signoff_path,
            repo_root=REPO_ROOT,
            expected_fields=expected_fields,
            required_source_paths=tuple(dict.fromkeys(required_sources)),
        )
    except FormalProvenanceError as exc:
        raise DataPreparationError(str(exc)) from exc


def _load_rows(
    *,
    config: Mapping[str, Any],
    role: str,
    cache_dir: Path | None,
    allow_download: bool,
) -> LoadedDatasetSlice:
    try:
        import datasets
        from datasets import DownloadConfig, load_dataset
    except ImportError as exc:  # pragma: no cover - environment capability
        raise DataPreparationError("datasets is required for WikiText preparation") from exc
    data = config["data"]
    identity = _formal_dataset_identity(config, role=role)
    split_cfg = data[role]
    start = int(split_cfg["candidate_row_start_inclusive"])
    stop = int(split_cfg["candidate_row_end_exclusive"])
    download = DownloadConfig(local_files_only=not allow_download)
    dataset = load_dataset(
        identity["dataset_repo_id"],
        str(data["config"]),
        split=f"{data['split']}[{start}:{stop}]",
        revision=identity["dataset_revision"],
        cache_dir=str(cache_dir) if cache_dir else None,
        download_config=download,
    )
    rows = tuple(canonical_text(str(row["text"])) for row in dataset)
    if len(rows) != stop - start:
        raise DataPreparationError("dataset slice length drifted from frozen window")
    fingerprint = getattr(dataset, "_fingerprint", None)
    if not isinstance(fingerprint, str) or not fingerprint:
        raise DataPreparationError("dataset slice has no reproducible fingerprint")
    library_version = getattr(datasets, "__version__", None)
    if not isinstance(library_version, str) or not library_version:
        raise DataPreparationError("datasets library has no version identity")
    download_checksums = getattr(getattr(dataset, "info", None), "download_checksums", None)
    if not isinstance(download_checksums, Mapping) or not download_checksums:
        raise DataPreparationError("dataset slice has no immutable source URL identity")
    source_urls = sorted(str(value) for value in download_checksums)
    expected_source_prefix = (
        f"hf://datasets/{identity['dataset_repo_id']}@"
        f"{identity['dataset_revision']}/"
    )
    if any(not value.startswith(expected_source_prefix) for value in source_urls):
        raise DataPreparationError("dataset source URL is not bound to frozen revision")
    return LoadedDatasetSlice(
        rows=rows,
        dataset_repo_id=identity["dataset_repo_id"],
        dataset_revision=identity["dataset_revision"],
        dataset_source_urls_sha256=sha256_bytes(canonical_json_bytes(source_urls)),
        datasets_library_version=library_version,
        dataset_slice_fingerprint=fingerprint,
        canonical_content_sha256=canonical_text_sequence_sha256(rows),
    )


def _load_tokenizers(
    config: Mapping[str, Any],
    *,
    cache_dir: Path | None,
    allow_download: bool,
) -> Mapping[str, Any]:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - environment capability
        raise DataPreparationError("transformers is required for tokenizer checks") from exc
    result = {}
    for key, spec in config["models"].items():
        result[key] = AutoTokenizer.from_pretrained(
            spec["repo_id"],
            revision=spec["revision"],
            cache_dir=str(cache_dir) if cache_dir else None,
            local_files_only=not allow_download,
        )
    return result


def _exclusive_immutable_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Create one immutable ledger file; any partial create remains fail-closed."""

    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o400)
    except FileExistsError as exc:
        raise DataPreparationError(
            "sealed data already reserved or consumed; one-shot retry is forbidden"
        ) from exc
    except OSError as exc:
        raise DataPreparationError("cannot create sealed one-shot ledger") from exc
    try:
        encoded = (
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except BaseException:
        # Never unlink a partially created reservation: a crash or short write
        # must consume the one-shot opportunity rather than allow a retry.
        raise


def _prepare_ledger_directory(ledger_dir: Path) -> Path:
    if ledger_dir.is_symlink():
        raise DataPreparationError("sealed ledger directory cannot be a symlink")
    ledger_dir.parent.mkdir(parents=True, exist_ok=True)
    if ledger_dir.parent.is_symlink():
        raise DataPreparationError("sealed ledger parent cannot be a symlink")
    try:
        ledger_dir.mkdir(mode=0o700, exist_ok=True)
    except OSError as exc:
        raise DataPreparationError("cannot create sealed ledger directory") from exc
    if ledger_dir.is_symlink() or not ledger_dir.is_dir():
        raise DataPreparationError("sealed ledger is not a canonical directory")
    return ledger_dir.resolve(strict=True)


def reserve_sealed_consumption(
    ledger_dir: Path,
    *,
    signoff_path: Path,
    signoff: Mapping[str, Any],
    protocol_sha256: str,
    config_sha256: str,
    producer_source_sha256: str,
    calibration_lock_binding: CalibrationLockBinding,
    output_dir: Path,
) -> Mapping[str, Any]:
    """O_EXCL-reserve the global sealed nonce before any sealed data opens."""

    directory = _prepare_ledger_directory(ledger_dir)
    nonce = secrets.token_hex(32)
    reservation = add_self_hash(
        {
            "schema_version": "ric-sealed-reservation-v1",
            "state": "RESERVED_FAIL_CLOSED",
            "role": "sealed",
            "mode": "formal",
            "nonce": nonce,
            "nonce_sha256": sha256_bytes(nonce.encode("ascii")),
            "signoff_manifest_sha256": signoff.get("signoff_sha256"),
            "signoff_file_sha256": sha256_file(signoff_path),
            "protocol_sha256": protocol_sha256,
            "config_sha256": config_sha256,
            "producer_source_sha256": producer_source_sha256,
            "calibration_lock_self_hash": (
                calibration_lock_binding.manifest_sha256
            ),
            "calibration_lock_file_sha256": calibration_lock_binding.file_sha256,
            "output_identity_sha256": sha256_bytes(
                str(output_dir.resolve()).encode("utf-8")
            ),
        },
        "record_sha256",
    )
    _exclusive_immutable_json(directory / "reservation.json", reservation)
    return reservation


def finalize_sealed_consumption(
    ledger_dir: Path,
    *,
    reservation: Mapping[str, Any],
    manifest_sha256: str,
    dataset_slice_canonical_content_sha256: str,
) -> Mapping[str, Any]:
    """Append the immutable external consumption record before output commit."""

    required_reservation_fields = {
        "schema_version",
        "state",
        "role",
        "mode",
        "nonce",
        "nonce_sha256",
        "signoff_manifest_sha256",
        "signoff_file_sha256",
        "protocol_sha256",
        "config_sha256",
        "producer_source_sha256",
        "calibration_lock_self_hash",
        "calibration_lock_file_sha256",
        "output_identity_sha256",
        "record_sha256",
    }
    if set(reservation) != required_reservation_fields:
        raise DataPreparationError("sealed reservation exact schema mismatch")
    validate_self_hash(reservation, "record_sha256")
    if (
        reservation.get("schema_version") != "ric-sealed-reservation-v1"
        or reservation.get("state") != "RESERVED_FAIL_CLOSED"
        or reservation.get("role") != "sealed"
        or reservation.get("mode") != "formal"
    ):
        raise DataPreparationError("sealed reservation state mismatch")
    nonce = reservation.get("nonce")
    if (
        not isinstance(nonce, str)
        or len(nonce) != 64
        or any(character not in "0123456789abcdef" for character in nonce)
        or sha256_bytes(nonce.encode("ascii")) != reservation.get("nonce_sha256")
    ):
        raise DataPreparationError("sealed reservation nonce binding mismatch")
    for field in (
        "nonce_sha256",
        "signoff_manifest_sha256",
        "signoff_file_sha256",
        "protocol_sha256",
        "config_sha256",
        "producer_source_sha256",
        "calibration_lock_self_hash",
        "calibration_lock_file_sha256",
        "output_identity_sha256",
        "record_sha256",
    ):
        if not is_sha256(reservation.get(field)):
            raise DataPreparationError(f"sealed reservation invalid hash: {field}")
    if not is_sha256(manifest_sha256) or not is_sha256(
        dataset_slice_canonical_content_sha256
    ):
        raise DataPreparationError("sealed finalization identity hash is invalid")
    directory = _prepare_ledger_directory(ledger_dir)
    reservation_path = directory / "reservation.json"
    if not reservation_path.is_file() or reservation_path.is_symlink():
        raise DataPreparationError("sealed reservation disappeared or was substituted")
    if sha256_file(reservation_path) != sha256_bytes(
        (
            json.dumps(reservation, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
    ):
        raise DataPreparationError("sealed reservation file changed before finalization")
    record = add_self_hash(
        {
            "schema_version": "ric-sealed-consumption-v1",
            "state": "CONSUMED",
            "role": "sealed",
            "mode": "formal",
            "reservation_sha256": reservation.get("record_sha256"),
            "nonce_sha256": reservation.get("nonce_sha256"),
            "signoff_manifest_sha256": reservation.get("signoff_manifest_sha256"),
            "signoff_file_sha256": reservation.get("signoff_file_sha256"),
            "calibration_lock_self_hash": reservation.get(
                "calibration_lock_self_hash"
            ),
            "calibration_lock_file_sha256": reservation.get(
                "calibration_lock_file_sha256"
            ),
            "manifest_sha256": manifest_sha256,
            "dataset_slice_canonical_content_sha256": (
                dataset_slice_canonical_content_sha256
            ),
        },
        "record_sha256",
    )
    _exclusive_immutable_json(directory / "consumption.json", record)
    return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("calibration", "sealed"), required=True)
    parser.add_argument("--mode", choices=("dev", "formal"), default="dev")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--signoff", type=Path)
    parser.add_argument(
        "--calibration-manifest",
        type=Path,
        help="required exact formal calibration manifest for formal sealed preparation",
    )
    parser.add_argument(
        "--calibration-lock",
        type=Path,
        help="required G1-passing formal calibration lock for sealed preparation",
    )
    parser.add_argument("--historical-scan-root", type=Path, default=REPO_ROOT / "docs")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--allow-download", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _validate_mode_role(args.mode, args.role)
    _validate_calibration_manifest_argument(
        mode=args.mode,
        role=args.role,
        path=args.calibration_manifest,
    )
    _validate_calibration_lock_argument(
        mode=args.mode,
        role=args.role,
        path=args.calibration_lock,
    )
    if args.output_dir.exists():
        raise DataPreparationError("refusing to overwrite data output directory")
    config = _load_config(args.config)
    if args.mode == "formal":
        validate_formal_data_preparation_environment(config, role=args.role)
    protocol_sha = sha256_file(args.protocol)
    config_sha = sha256_file(args.config)
    source_sha = _producer_source_sha256()
    calibration_lock_binding = None
    if args.mode == "formal" and args.role == "sealed":
        # Authorization is checked before signoff verification, historical
        # scanning, reservation, calibration-manifest loading, or dataset I/O.
        calibration_lock_binding = load_formal_calibration_lock(
            args.calibration_lock,
            config=config,
            protocol_sha256=protocol_sha,
            config_sha256=config_sha,
        )
    signoff = None
    if args.mode == "formal":
        signoff = require_formal_signoff(
            args.signoff,
            protocol_sha256=protocol_sha,
            config_sha256=config_sha,
            producer_source_sha256=source_sha,
            data_role=args.role,
            calibration_lock_binding=calibration_lock_binding,
        )

    registry = build_historical_registry(
        args.historical_scan_root,
        excluded_paths=(() if args.mode == "formal" else (args.output_dir,)),
        formal=args.mode == "formal",
    )
    if args.mode == "formal" and not registry["complete"]:
        raise DataPreparationError("formal historical registry scan has parse failures")
    calibration_binding = None
    if args.mode == "formal" and args.role == "sealed":
        calibration_binding = load_formal_calibration_manifest(
            args.calibration_manifest,
            config=config,
            protocol_sha256=protocol_sha,
            config_sha256=config_sha,
            producer_source_sha256=source_sha,
        )
    sealed_reservation = None
    if args.mode == "formal" and args.role == "sealed":
        assert (
            signoff is not None
            and args.signoff is not None
            and calibration_lock_binding is not None
        )
        sealed_reservation = reserve_sealed_consumption(
            DEFAULT_SEALED_LEDGER_DIR,
            signoff_path=args.signoff,
            signoff=signoff,
            protocol_sha256=protocol_sha,
            config_sha256=config_sha,
            producer_source_sha256=source_sha,
            calibration_lock_binding=calibration_lock_binding,
            output_dir=args.output_dir,
        )
    dataset_slice = _load_rows(
        config=config,
        role=args.role,
        cache_dir=args.cache_dir,
        allow_download=args.allow_download,
    )
    if args.mode == "formal":
        validate_loaded_dataset_identity(
            dataset_slice,
            config=config,
            role=args.role,
        )
    tokenizers = _load_tokenizers(
        config,
        cache_dir=args.cache_dir,
        allow_download=args.allow_download,
    )

    def token_lengths(text: str) -> Mapping[str, int]:
        return {
            key: len(tokenizer(text, add_special_tokens=False)["input_ids"])
            for key, tokenizer in tokenizers.items()
        }

    data_cfg = config["data"]
    role_cfg = data_cfg[args.role]
    exclusion_hashes = set(registry["text_hashes"])
    if calibration_binding is not None:
        exclusion_hashes.update(calibration_binding.selected_text_sha256)
    requests = select_requests(
        dataset_slice.rows,
        source_row_start=int(role_cfg["candidate_row_start_inclusive"]),
        required_count=int(role_cfg["document_count"]),
        selection_seed=int(data_cfg["selection_seed"]),
        min_tokens=int(data_cfg["min_tokens_both_frozen_tokenizers"]),
        token_lengths=token_lengths,
        historical_hashes=exclusion_hashes,
        role=args.role,
    )
    if calibration_binding is not None:
        request_id_collisions = sorted(
            {str(row["request_id"]) for row in requests}
            & set(calibration_binding.request_ids)
        )
        if request_id_collisions:
            raise DataPreparationError(
                "sealed request identities collide with formal calibration manifest: "
                f"{request_id_collisions[:3]}"
            )
    manifest_payload = {
        "schema_version": "ric-data-manifest-v1",
        "status": "INPUT_ONLY" if args.mode == "formal" else "NOT_TESTED",
        "scientific_result": False,
        "mode": args.mode,
        "role": args.role,
        "dataset_loader": "wikitext",
        "dataset_config": data_cfg["config"],
        "dataset_split": data_cfg["split"],
        "data_preparation_producer": data_cfg["formal_dataset_identity"]["producer"],
        "data_preparation_python_environment": data_cfg["formal_dataset_identity"][
            "python_environment"
        ],
        "dataset_repo_id": dataset_slice.dataset_repo_id,
        "dataset_revision": dataset_slice.dataset_revision,
        "dataset_source_urls_sha256": dataset_slice.dataset_source_urls_sha256,
        "datasets_library_version": dataset_slice.datasets_library_version,
        "dataset_slice_fingerprint": dataset_slice.dataset_slice_fingerprint,
        "dataset_slice_row_count": len(dataset_slice.rows),
        "dataset_slice_canonical_content_sha256": (
            dataset_slice.canonical_content_sha256
        ),
        "candidate_window": [
            int(role_cfg["candidate_row_start_inclusive"]),
            int(role_cfg["candidate_row_end_exclusive"]),
        ],
        "selection_seed": int(data_cfg["selection_seed"]),
        "selection_method": data_cfg["selection_method"],
        "sequence_tokens": int(data_cfg["sequence_length"]),
        "batch_size": int(data_cfg["batch_size"]),
        "padding_allowed": bool(data_cfg["padding_allowed"]),
        "model_revisions": {
            key: _model_revision_string(spec) for key, spec in config["models"].items()
        },
        "tokenizer_revisions": {
            key: _model_revision_string(spec) for key, spec in config["models"].items()
        },
        "historical_exclusion_registry_sha256": registry["registry_sha256"],
        "protocol_sha256": protocol_sha,
        "config_sha256": config_sha,
        "prepare_data_source_sha256": source_sha,
        "signoff_sha256": sha256_file(args.signoff) if signoff is not None else None,
        "selected_text_sha256": [str(row["text_sha256"]) for row in requests],
        "requests": requests,
    }
    if sealed_reservation is not None:
        assert calibration_binding is not None and calibration_lock_binding is not None
        manifest_payload.update(
            {
                "sealed_reservation_sha256": sealed_reservation["record_sha256"],
                "sealed_nonce_sha256": sealed_reservation["nonce_sha256"],
                "calibration_manifest_self_hash": calibration_binding.manifest_sha256,
                "calibration_manifest_file_sha256": calibration_binding.file_sha256,
                "calibration_selected_list_sha256": sha256_bytes(
                    canonical_json_bytes(calibration_binding.selected_text_sha256)
                ),
                "calibration_lock_self_hash": (
                    calibration_lock_binding.manifest_sha256
                ),
                "calibration_lock_file_sha256": (
                    calibration_lock_binding.file_sha256
                ),
            }
        )
    manifest = add_self_hash(manifest_payload)
    try:
        validate_data_manifest_fields(
            manifest,
            mode=args.mode,
            role=args.role,
            config=config,
            protocol_sha256=protocol_sha,
            config_sha256=config_sha,
            expected_prepare_data_source_sha256=source_sha,
            expected_historical_registry_sha256=registry["registry_sha256"],
            expected_calibration_lock_self_hash=(
                calibration_lock_binding.manifest_sha256
                if calibration_lock_binding is not None
                else None
            ),
            expected_calibration_lock_file_sha256=(
                calibration_lock_binding.file_sha256
                if calibration_lock_binding is not None
                else None
            ),
        )
    except FormalProvenanceError as exc:
        raise DataPreparationError(str(exc)) from exc

    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{args.output_dir.name}.partial-", dir=args.output_dir.parent
    ) as temporary_directory:
        temporary = Path(temporary_directory)
        if signoff is not None:
            embedded_signoff_sha256 = materialize_verified_signoff(
                args.signoff, temporary
            )
            if embedded_signoff_sha256 != manifest["signoff_sha256"]:
                raise DataPreparationError("embedded data signoff copy changed")
        (temporary / "historical_exclusion_registry.json").write_text(
            json.dumps(registry, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (temporary / f"data_manifest_{args.role}.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if sealed_reservation is not None:
            record = finalize_sealed_consumption(
                DEFAULT_SEALED_LEDGER_DIR,
                reservation=sealed_reservation,
                manifest_sha256=manifest["manifest_sha256"],
                dataset_slice_canonical_content_sha256=(
                    dataset_slice.canonical_content_sha256
                ),
            )
            (temporary / "sealed_consumption_record.json").write_text(
                json.dumps(record, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        temporary.rename(args.output_dir)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "role": args.role,
                "request_count": len(requests),
                "manifest_sha256": manifest["manifest_sha256"],
                "output_dir": str(args.output_dir),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
