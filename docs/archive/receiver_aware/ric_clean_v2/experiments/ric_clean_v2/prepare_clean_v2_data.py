#!/usr/bin/env python3
"""Prepare the clean RIC-v2 calibration or sealed data manifest.

The sealed entry point is fail-closed: Phase-4 signoff and an N1-GO lock are
validated before dataset I/O, and a fixed O_EXCL ledger is reserved inside the
API rather than by a CLI wrapper.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence


HERE = Path(__file__).resolve().parent
IDEA_ROOT = HERE.parents[1]
REPO_ROOT = next(candidate for candidate in HERE.parents if (candidate / "experiments/shared").is_dir())
DEFAULT_CONFIG = IDEA_ROOT / "configs/ric_clean_v2.json"
DEFAULT_PROTOCOL = IDEA_ROOT / "RIC_Clean_v2_Phase2_冻结实验协议_2026-07-23.md"
CLEAN_BUNDLE_ROOT = Path("/root/autodl-tmp/ric_clean_v2_20260723")
CLEAN_STATE_DIR = CLEAN_BUNDLE_ROOT / "state"
SEALED_LEDGER = CLEAN_STATE_DIR / "sealed_data_consumption.json"
OLD_BUNDLE_ROOT = Path("/root/autodl-tmp/ric_formal_v1_20260722")
FROZEN_EXCLUSION_ROOTS = (
    OLD_BUNDLE_ROOT / "docs",
    OLD_BUNDLE_ROOT / "formal_outputs",
)
CLEAN_REVIEW_DIR = CLEAN_BUNDLE_ROOT / "review"
REVIEW_REPORT = CLEAN_REVIEW_DIR / "RIC_Clean_v2_CodeReview.md"
TEST_REPORT = CLEAN_REVIEW_DIR / "RIC_Clean_v2_TestReport.json"
REVIEWED_SOURCE_MANIFEST = CLEAN_REVIEW_DIR / "reviewed_source_manifest.json"
CALIBRATION_MANIFEST = CLEAN_BUNDLE_ROOT / "clean_v2/data/calibration/manifest.json"
N1_LOCK = CLEAN_BUNDLE_ROOT / "clean_v2/n1_lock/lock.json"
CALIBRATION_LOCK = CLEAN_BUNDLE_ROOT / "clean_v2/calibration_lock/lock.json"
N1_PRODUCER = HERE / "run_clean_v2_n1.py"
CALIBRATION_PRODUCER = HERE / "run_clean_v2_calibration.py"
MODEL_DIRS = {
    "olmoe": Path("/root/autodl-tmp/models/olmoe"),
    "llmjp": Path("/root/autodl-tmp/models/llmjp"),
}
TEXT_HASH_FIELDS = {
    "text_sha256",
    "canonical_text_sha256",
    "selected_text_sha256",
    "selected_canonical_text_sha256",
}


class CleanDataError(RuntimeError):
    pass


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def object_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def add_self_hash(value: Mapping[str, Any], field: str = "manifest_sha256") -> dict[str, Any]:
    payload = dict(value)
    if field in payload:
        raise CleanDataError(f"self-hash field already exists: {field}")
    payload[field] = object_sha256(payload)
    return payload


def load_mapping(path: Path, *, label: str) -> dict[str, Any]:
    def reject_duplicates(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise CleanDataError(f"duplicate JSON key in {label}: {key}")
            result[key] = item
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(
                CleanDataError(f"non-finite JSON token in {label}: {token}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CleanDataError(f"cannot load {label}") from exc
    if not isinstance(value, dict):
        raise CleanDataError(f"{label} must be a JSON object")
    return value


def decode_json_bytes(raw: bytes, *, label: str) -> Any:
    def reject_duplicates(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise CleanDataError(f"duplicate JSON key in {label}: {key}")
            result[key] = item
        return result

    try:
        return json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(
                CleanDataError(f"non-finite JSON token in {label}: {token}")
            ),
        )
    except CleanDataError:
        raise
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise CleanDataError(f"cannot load historical JSON: {label}") from exc


def load_json_any(path: Path, *, label: str) -> Any:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise CleanDataError(f"cannot load historical JSON: {label}") from exc
    return decode_json_bytes(raw, label=label)


def validate_self_hash(value: Mapping[str, Any], field: str = "manifest_sha256") -> None:
    supplied = value.get(field)
    unhashed = dict(value)
    unhashed.pop(field, None)
    if not isinstance(supplied, str) or supplied != object_sha256(unhashed):
        raise CleanDataError(f"invalid {field}")


def source_sha256() -> str:
    return object_sha256(
        {
            "path": Path(__file__).resolve().relative_to(REPO_ROOT).as_posix(),
            "bytes_sha256": file_sha256(Path(__file__)),
        }
    )


def validate_phase4_signoff(path: Path, *, role: str) -> Mapping[str, Any]:
    expected_path = CLEAN_REVIEW_DIR / f"signoff_data_{role}.json"
    if path.resolve(strict=True) != expected_path:
        raise CleanDataError("Phase-4 signoff path is not the fixed reviewed path")
    value = load_mapping(path, label="clean-v2 Phase-4 signoff")
    validate_self_hash(value, "signoff_sha256")
    if REVIEW_REPORT.is_symlink() or TEST_REPORT.is_symlink() or REVIEWED_SOURCE_MANIFEST.is_symlink():
        raise CleanDataError("Phase-4 evidence may not be a symlink")
    source_manifest = load_mapping(
        REVIEWED_SOURCE_MANIFEST, label="clean-v2 reviewed source manifest"
    )
    validate_self_hash(source_manifest)
    review_lines = REVIEW_REPORT.read_text(encoding="utf-8").splitlines()
    if (
        "STATUS: SIGNED-OFF" not in review_lines
        or "OPEN_P0: 0" not in review_lines
        or f"REVIEWED_SOURCE_MANIFEST_SHA256: {source_manifest['manifest_sha256']}"
        not in review_lines
    ):
        raise CleanDataError("Phase-4 review report is not bound and signed off")
    expected_sources = {
        "config": {
            "path": DEFAULT_CONFIG.resolve(strict=True).relative_to(REPO_ROOT).as_posix(),
            "file_sha256": file_sha256(DEFAULT_CONFIG),
        },
        "protocol": {
            "path": DEFAULT_PROTOCOL.resolve(strict=True).relative_to(REPO_ROOT).as_posix(),
            "file_sha256": file_sha256(DEFAULT_PROTOCOL),
        },
        "producer": {
            "path": Path(__file__).resolve(strict=True).relative_to(REPO_ROOT).as_posix(),
            "file_sha256": file_sha256(Path(__file__)),
            "source_sha256": source_sha256(),
        },
        "tests": {
            "path": (HERE / "test_prepare_clean_v2_data.py").resolve(strict=True).relative_to(REPO_ROOT).as_posix(),
            "file_sha256": file_sha256(HERE / "test_prepare_clean_v2_data.py"),
        },
    }
    if (
        source_manifest.get("schema_version") != "ric-clean-v2-reviewed-source-manifest-v1"
        or source_manifest.get("status") != "REVIEWED"
        or source_manifest.get("sources") != expected_sources
    ):
        raise CleanDataError("reviewed source manifest does not bind exact sources")
    test_report = load_mapping(TEST_REPORT, label="clean-v2 test report")
    validate_self_hash(test_report)
    if (
        test_report.get("schema_version") != "ric-clean-v2-test-report-v1"
        or test_report.get("status") != "PASS"
        or test_report.get("errors") != 0
        or test_report.get("failures") != 0
        or not isinstance(test_report.get("tests_run"), int)
        or isinstance(test_report.get("tests_run"), bool)
        or test_report["tests_run"] <= 0
        or test_report.get("reviewed_source_manifest_sha256")
        != source_manifest["manifest_sha256"]
        or test_report.get("reviewed_source_manifest_file_sha256")
        != file_sha256(REVIEWED_SOURCE_MANIFEST)
    ):
        raise CleanDataError("Phase-4 test report does not bind reviewed sources")
    expected = {
        "schema_version": "ric-clean-v2-phase4-signoff-v1",
        "status": "SIGNED-OFF",
        "open_p0": 0,
        "role": role,
        "config_sha256": file_sha256(DEFAULT_CONFIG),
        "protocol_sha256": file_sha256(DEFAULT_PROTOCOL),
        "prepare_clean_v2_data_source_sha256": source_sha256(),
        "review_report_sha256": file_sha256(REVIEW_REPORT),
        "test_report_sha256": file_sha256(TEST_REPORT),
        "reviewed_source_manifest_sha256": source_manifest["manifest_sha256"],
        "reviewed_source_manifest_file_sha256": file_sha256(REVIEWED_SOURCE_MANIFEST),
    }
    for field, wanted in expected.items():
        if value.get(field) != wanted or type(value.get(field)) is not type(wanted):
            raise CleanDataError(f"Phase-4 signoff mismatch: {field}")
    return value


def validate_n1_lock(path: Path) -> Mapping[str, Any]:
    if path.resolve(strict=True) != N1_LOCK:
        raise CleanDataError("N1 lock path is not the fixed producer path")
    value = load_mapping(path, label="clean-v2 N1 lock")
    validate_self_hash(value)
    required_hashes = (
        "milp_instances_file_sha256",
        "milp_solutions_file_sha256",
        "oracle_status_file_sha256",
        "calibration_manifest_sha256",
        "calibration_lock_sha256",
    )
    if (
        value.get("schema_version") != "ric-clean-v2-n1-lock-v1"
        or value.get("status") != "N1_GO_LOCKED"
        or value.get("scientific_result") is not False
        or value.get("config_sha256") != file_sha256(DEFAULT_CONFIG)
        or value.get("protocol_sha256") != file_sha256(DEFAULT_PROTOCOL)
        or value.get("n1_gate_pass") is not True
        or not N1_PRODUCER.is_file()
        or value.get("producer_source_sha256") != file_sha256(N1_PRODUCER)
        or any(
            not isinstance(value.get(field), str) or len(value[field]) != 64
            for field in required_hashes
        )
        or not isinstance(value.get("model_summaries"), dict)
        or set(value["model_summaries"]) != {"olmoe", "llmjp"}
    ):
        raise CleanDataError("sealed preparation requires an exact N1-GO lock")
    return value


def validate_calibration_manifest(path: Path) -> Mapping[str, Any]:
    if path.resolve(strict=True) != CALIBRATION_MANIFEST:
        raise CleanDataError("calibration manifest path is not fixed")
    value = load_mapping(path, label="clean-v2 calibration manifest")
    validate_self_hash(value)
    config = load_mapping(DEFAULT_CONFIG, label="clean-v2 config")
    selected_requests = value.get("selected_request_ids")
    if (
        value.get("schema_version") != "ric-clean-v2-data-manifest-v1"
        or value.get("status") != "INPUT_ONLY"
        or value.get("scientific_result") is not False
        or value.get("role") != "calibration"
        or value.get("config_sha256") != file_sha256(DEFAULT_CONFIG)
        or value.get("protocol_sha256") != file_sha256(DEFAULT_PROTOCOL)
        or value.get("producer_source_sha256") != source_sha256()
        or value.get("formal_dataset_identity")
        != config["data"]["formal_dataset_identity"]
        or not isinstance(value.get("selected_text_sha256"), list)
        or not value["selected_text_sha256"]
        or not isinstance(selected_requests, list)
        or len(selected_requests) != len(value["selected_text_sha256"])
        or len(set(selected_requests)) != len(selected_requests)
        or any(
            not isinstance(request_id, str)
            or not request_id.startswith("ric-clean-v2:calibration:")
            for request_id in selected_requests
        )
    ):
        raise CleanDataError("calibration manifest identity mismatch")
    return value


def validate_calibration_lock(path: Path) -> Mapping[str, Any]:
    if path.resolve(strict=True) != CALIBRATION_LOCK:
        raise CleanDataError("calibration lock path is not fixed")
    value = load_mapping(path, label="clean-v2 calibration lock")
    validate_self_hash(value)
    required_maps = (
        "calib_best_joinblind_by_model_cell",
        "closure_budget_us_by_model_cell",
        "service_lut_metadata_sha256_by_model",
        "scenario_tree_sha256_by_model",
    )
    if (
        value.get("schema_version") != "ric-clean-v2-calibration-lock-v1"
        or value.get("status") != "CALIBRATION_LOCKED"
        or value.get("scientific_result") is not False
        or value.get("config_sha256") != file_sha256(DEFAULT_CONFIG)
        or value.get("protocol_sha256") != file_sha256(DEFAULT_PROTOCOL)
        or not CALIBRATION_PRODUCER.is_file()
        or value.get("producer_source_sha256") != file_sha256(CALIBRATION_PRODUCER)
        or any(not isinstance(value.get(field), dict) or not value[field] for field in required_maps)
    ):
        raise CleanDataError("calibration lock identity/content mismatch")
    return value


def validate_clean_bundle_precondition(*, role: str, output_dir: Path) -> None:
    if role == "calibration":
        forbidden_roots = (
            CLEAN_BUNDLE_ROOT / "clean_v2/data",
            CLEAN_BUNDLE_ROOT / "clean_v2/routes",
            CLEAN_BUNDLE_ROOT / "clean_v2/scenarios",
            CLEAN_BUNDLE_ROOT / "clean_v2/oracle",
            CLEAN_BUNDLE_ROOT / "clean_v2/results",
            CLEAN_BUNDLE_ROOT / "clean_v2/n1_lock",
            CLEAN_BUNDLE_ROOT / "clean_v2/calibration_lock",
        )
    else:
        forbidden_roots = (
            CLEAN_BUNDLE_ROOT / "clean_v2/data/sealed",
            CLEAN_BUNDLE_ROOT / "clean_v2/routes/sealed",
            CLEAN_BUNDLE_ROOT / "clean_v2/scenarios/sealed",
            CLEAN_BUNDLE_ROOT / "clean_v2/results/sealed",
        )
    existing = [str(path) for path in forbidden_roots if path.exists()]
    partials = [
        str(path)
        for path in CLEAN_BUNDLE_ROOT.rglob("*")
        if any(part.startswith(".") and ".partial-" in part for part in path.parts)
    ]
    if existing or partials:
        raise CleanDataError(
            f"clean bundle precondition failed: existing={existing}, partials={partials[:3]}"
        )
    if output_dir.exists():
        raise CleanDataError("clean output already exists")


def validate_fixed_output_ancestry(output_dir: Path) -> None:
    for candidate in (CLEAN_BUNDLE_ROOT / "clean_v2", CLEAN_BUNDLE_ROOT / "clean_v2/data"):
        if candidate.exists() and (
            candidate.is_symlink() or candidate.resolve(strict=True) != candidate
        ):
            raise CleanDataError(f"clean output ancestor identity mismatch: {candidate}")


def iter_text_hashes(value: object, *, parent: str = "") -> Iterable[str]:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key in TEXT_HASH_FIELDS:
                if isinstance(item, str) and len(item) == 64:
                    yield item
                elif isinstance(item, list):
                    yield from (
                        entry for entry in item if isinstance(entry, str) and len(entry) == 64
                    )
            yield from iter_text_hashes(item, parent=str(key))
    elif isinstance(value, list):
        for item in value:
            yield from iter_text_hashes(item, parent=parent)


def historical_exclusion_hashes(roots: Sequence[Path]) -> tuple[set[str], list[dict[str, Any]]]:
    hashes: set[str] = set()
    sources: list[dict[str, Any]] = []
    for root in roots:
        if (
            root.is_symlink()
            or Path(os.path.abspath(root)) != root
            or root.resolve(strict=True) != root
        ):
            raise CleanDataError(f"historical root identity mismatch: {root}")
        resolved_root = root.resolve(strict=True)
        for entry in resolved_root.rglob("*"):
            if entry.is_symlink():
                raise CleanDataError(f"historical tree contains symlink: {entry}")
        for path in sorted(resolved_root.rglob("*.json")):
            if path.is_symlink() or not path.is_file():
                raise CleanDataError(f"non-regular historical JSON: {path}")
            before = path.stat()
            raw = path.read_bytes()
            after = path.stat()
            if (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
            ) != (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
            ):
                raise CleanDataError(f"historical JSON changed while reading: {path}")
            value = decode_json_bytes(raw, label=str(path))
            found = sorted(set(iter_text_hashes(value)))
            hashes.update(found)
            sources.append(
                {
                    "path": str(path.resolve()),
                    "file_sha256": hashlib.sha256(raw).hexdigest(),
                    "text_hash_count": len(found),
                }
            )
    return hashes, sources


def model_tree_sha256(root: Path) -> str:
    resolved = root.resolve(strict=True)
    if root.is_symlink() or not resolved.is_dir():
        raise CleanDataError(f"model root is not a real directory: {root}")
    rows: list[dict[str, Any]] = []
    for path in sorted(candidate for candidate in resolved.rglob("*") if candidate.is_file()):
        if path.is_symlink():
            raise CleanDataError(f"model tree contains symlink: {path}")
        before = path.stat()
        digest = file_sha256(path)
        after = path.stat()
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise CleanDataError(f"model file changed while hashing: {path}")
        rows.append(
            {
                "path": path.relative_to(resolved).as_posix(),
                "size_bytes": after.st_size,
                "sha256": digest,
            }
        )
    if not rows:
        raise CleanDataError("model tree is empty")
    return object_sha256(rows)


def validate_historical_inventory(
    sources: Sequence[Mapping[str, Any]], data_config: Mapping[str, Any]
) -> None:
    rows = [
        {"path": row["path"], "file_sha256": row["file_sha256"]}
        for row in sources
    ]
    if (
        len(rows) != data_config["historical_exclusion_expected_json_file_count"]
        or object_sha256(rows)
        != data_config["historical_exclusion_expected_inventory_sha256"]
    ):
        raise CleanDataError("historical exclusion inventory differs from frozen config")


def validate_runtime_identity(
    identity: Mapping[str, Any],
    *,
    datasets_version: str,
    transformers_version: str,
    tokenizers_version: str,
    executable: Path | None = None,
) -> None:
    actual_python = Path(sys.executable) if executable is None else executable
    expected_python = Path(str(identity["python_environment"]))
    if (
        not expected_python.is_file()
        or Path(os.path.abspath(actual_python)) != expected_python
        or actual_python.resolve(strict=True) != expected_python.resolve(strict=True)
    ):
        raise CleanDataError("formal Python environment differs from frozen config")
    expected_versions = {
        "datasets": identity["datasets_library_version"],
        "transformers": identity["transformers_library_version"],
        "tokenizers": identity["tokenizers_library_version"],
    }
    actual_versions = {
        "datasets": datasets_version,
        "transformers": transformers_version,
        "tokenizers": tokenizers_version,
    }
    for library, expected in expected_versions.items():
        if actual_versions[library] != expected:
            raise CleanDataError(f"{library} library version differs from frozen config")


def select_requests(
    rows: Sequence[str],
    *,
    role: str,
    source_start: int,
    required: int,
    seed: int,
    min_tokens: int,
    tokenizers: Mapping[str, Any],
    excluded: set[str],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for offset, raw in enumerate(rows):
        text = str(raw).replace("\r\n", "\n").replace("\r", "\n")
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        lengths = {
            key: len(tokenizer(text, add_special_tokens=False)["input_ids"])
            for key, tokenizer in tokenizers.items()
        }
        if not lengths or min(lengths.values()) < min_tokens:
            continue
        rank = hashlib.sha256(f"{seed}{text_hash}".encode()).hexdigest()
        candidates.append(
            {
                "source_row": source_start + offset,
                "rank_sha256": rank,
                "text_sha256": text_hash,
                "token_lengths": lengths,
                "text": text,
            }
        )
    selected = sorted(candidates, key=lambda row: (row["rank_sha256"], row["source_row"]))[
        :required
    ]
    if len(selected) != required:
        raise CleanDataError("frozen window lacks enough dual-tokenizer-valid rows")
    selected_hashes = [str(row["text_sha256"]) for row in selected]
    if len(set(selected_hashes)) != len(selected_hashes):
        raise CleanDataError("frozen selection contains duplicate canonical text")
    collisions = sorted(set(selected_hashes) & excluded)
    if collisions:
        raise CleanDataError(
            f"BLOCKED_DATA_SPLIT: frozen selection collides with history: {collisions[:3]}"
        )
    return [
        {**row, "request_id": f"ric-clean-v2:{role}:{index:04d}:{row['text_sha256'][:12]}"}
        for index, row in enumerate(selected)
    ]


def reserve_sealed(*, output_dir: Path, signoff: Mapping[str, Any], n1_lock: Mapping[str, Any]) -> Mapping[str, Any]:
    CLEAN_STATE_DIR.mkdir(parents=True, exist_ok=True)
    if CLEAN_STATE_DIR.is_symlink() or SEALED_LEDGER.is_symlink():
        raise CleanDataError("clean-v2 state path may not be a symlink")
    record = add_self_hash(
        {
            "schema_version": "ric-clean-v2-sealed-data-consumption-v1",
            "state": "RESERVED_FAIL_CLOSED",
            "output_dir": str(output_dir.resolve(strict=False)),
            "phase4_signoff_sha256": signoff["signoff_sha256"],
            "n1_lock_sha256": n1_lock["manifest_sha256"],
            "config_sha256": file_sha256(DEFAULT_CONFIG),
            "protocol_sha256": file_sha256(DEFAULT_PROTOCOL),
        }
    )
    encoded = json.dumps(record, indent=2, sort_keys=True).encode() + b"\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(SEALED_LEDGER, flags, 0o400)
    except FileExistsError as exc:
        raise CleanDataError("clean-v2 sealed data was already consumed") from exc
    try:
        offset = 0
        while offset < len(encoded):
            written = os.write(descriptor, encoded[offset:])
            if written <= 0:
                raise CleanDataError("sealed ledger write made no progress")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    if SEALED_LEDGER.read_bytes() != encoded:
        raise CleanDataError("sealed ledger byte verification failed")
    return record


def prepare(
    *,
    role: str,
    output_dir: Path,
    phase4_signoff_path: Path,
    cache_dir: Path,
) -> Mapping[str, Any]:
    if role not in {"calibration", "sealed"}:
        raise CleanDataError("invalid role")
    if CLEAN_BUNDLE_ROOT.resolve(strict=True) != CLEAN_BUNDLE_ROOT:
        raise CleanDataError("clean authoritative root identity mismatch")
    expected_output = CLEAN_BUNDLE_ROOT / f"clean_v2/data/{role}"
    if Path(os.path.abspath(output_dir)) != expected_output:
        raise CleanDataError("output path is not the fixed role path")
    validate_fixed_output_ancestry(output_dir)
    normalized_cache = Path(os.path.abspath(cache_dir))
    resolved_cache = cache_dir.resolve(strict=False)
    if (
        cache_dir.is_symlink()
        or normalized_cache == CLEAN_BUNDLE_ROOT
        or CLEAN_BUNDLE_ROOT in normalized_cache.parents
        or resolved_cache == CLEAN_BUNDLE_ROOT
        or CLEAN_BUNDLE_ROOT in resolved_cache.parents
    ):
        raise CleanDataError("dataset cache must be outside the clean bundle")
    validate_clean_bundle_precondition(role=role, output_dir=output_dir)
    signoff = validate_phase4_signoff(phase4_signoff_path, role=role)
    n1_lock = None
    calibration_manifest = None
    calibration_lock = None
    if role == "sealed":
        calibration_manifest = validate_calibration_manifest(CALIBRATION_MANIFEST)
        calibration_lock = validate_calibration_lock(CALIBRATION_LOCK)
        n1_lock = validate_n1_lock(N1_LOCK)
        if (
            n1_lock.get("calibration_lock_sha256")
            != calibration_lock["manifest_sha256"]
            or n1_lock.get("calibration_manifest_sha256")
            != calibration_manifest["manifest_sha256"]
        ):
            raise CleanDataError("N1 lock is not bound to calibration inputs")

    config = load_mapping(DEFAULT_CONFIG, label="clean-v2 config")
    if config.get("schema_version") != "ric-clean-config-v2":
        raise CleanDataError("wrong clean-v2 config")
    data = config["data"]
    split = data[role]
    start = int(split["candidate_row_start_inclusive"])
    stop = int(split["candidate_row_end_exclusive"])

    try:
        import datasets
        from datasets import DownloadConfig, load_dataset
        import tokenizers as tokenizers_library
        import transformers
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise CleanDataError("datasets and transformers are required") from exc

    identity = data["formal_dataset_identity"]
    validate_runtime_identity(
        identity,
        datasets_version=datasets.__version__,
        transformers_version=transformers.__version__,
        tokenizers_version=tokenizers_library.__version__,
    )

    if role == "sealed":
        assert n1_lock is not None
        reservation = reserve_sealed(output_dir=output_dir, signoff=signoff, n1_lock=n1_lock)
    else:
        reservation = None

    dataset = load_dataset(
        identity["dataset_repo_id"],
        data["config"],
        split=f"{data['split']}[{start}:{stop}]",
        revision=identity["dataset_revision"],
        cache_dir=str(cache_dir),
        download_config=DownloadConfig(local_files_only=True),
    )
    rows = tuple(str(row["text"]) for row in dataset)
    if len(rows) != stop - start:
        raise CleanDataError("dataset slice length drift")
    model_tree_hashes = {
        key: model_tree_sha256(path) for key, path in MODEL_DIRS.items()
    }
    for key, actual in model_tree_hashes.items():
        if actual != config["models"][key]["expected_local_model_tree_manifest_sha256"]:
            raise CleanDataError(f"frozen model/tokenizer tree mismatch: {key}")
    tokenizer_instances = {
        key: AutoTokenizer.from_pretrained(path, local_files_only=True)
        for key, path in MODEL_DIRS.items()
    }
    exclusions, exclusion_sources = historical_exclusion_hashes(
        FROZEN_EXCLUSION_ROOTS
    )
    if calibration_manifest is not None:
        exclusions.update(str(value) for value in calibration_manifest["selected_text_sha256"])
    exclusion_registry = add_self_hash(
        {
            "schema_version": "ric-clean-v2-historical-exclusion-v1",
            "frozen_roots": [str(path) for path in FROZEN_EXCLUSION_ROOTS],
            "source_files": exclusion_sources,
            "text_hashes": sorted(exclusions),
            "source_file_count": len(exclusion_sources),
            "text_hash_count": len(exclusions),
        },
        "registry_sha256",
    )
    validate_historical_inventory(exclusion_sources, data)
    requests = select_requests(
        rows,
        role=role,
        source_start=start,
        required=int(split["document_count"]),
        seed=int(data["selection_seed"]),
        min_tokens=int(data["min_tokens_both_frozen_tokenizers"]),
        tokenizers=tokenizer_instances,
        excluded=exclusions,
    )
    manifest = add_self_hash(
        {
            "schema_version": "ric-clean-v2-data-manifest-v1",
            "status": "INPUT_ONLY",
            "scientific_result": False,
            "role": role,
            "config_sha256": file_sha256(DEFAULT_CONFIG),
            "protocol_sha256": file_sha256(DEFAULT_PROTOCOL),
            "producer_source_sha256": source_sha256(),
            "phase4_signoff_sha256": signoff["signoff_sha256"],
            "n1_lock_sha256": n1_lock["manifest_sha256"] if n1_lock else None,
            "calibration_manifest_sha256": (
                calibration_manifest["manifest_sha256"]
                if calibration_manifest is not None
                else None
            ),
            "calibration_manifest_file_sha256": (
                file_sha256(CALIBRATION_MANIFEST)
                if calibration_manifest is not None
                else None
            ),
            "calibration_selected_list_sha256": (
                object_sha256(calibration_manifest["selected_text_sha256"])
                if calibration_manifest is not None
                else None
            ),
            "calibration_lock_sha256": (
                calibration_lock["manifest_sha256"]
                if calibration_lock is not None
                else None
            ),
            "sealed_reservation_sha256": reservation["manifest_sha256"] if reservation else None,
            "dataset_repo_id": identity["dataset_repo_id"],
            "dataset_revision": identity["dataset_revision"],
            "formal_dataset_identity": identity,
            "datasets_library_version": datasets.__version__,
            "transformers_library_version": transformers.__version__,
            "tokenizers_library_version": tokenizers_library.__version__,
            "python_executable": str(Path(os.path.abspath(sys.executable))),
            "model_tree_manifest_sha256": model_tree_hashes,
            "dataset_slice_fingerprint": str(dataset._fingerprint),
            "dataset_slice_canonical_content_sha256": object_sha256(rows),
            "candidate_window": [start, stop],
            "selection_seed": int(data["selection_seed"]),
            "historical_exclusion_text_hash_count": len(exclusions),
            "historical_exclusion_registry_sha256": exclusion_registry[
                "registry_sha256"
            ],
            "selected_text_sha256": [row["text_sha256"] for row in requests],
            "selected_request_ids": [row["request_id"] for row in requests],
            "requests": requests,
        }
    )
    if (
        file_sha256(DEFAULT_CONFIG) != signoff["config_sha256"]
        or file_sha256(DEFAULT_PROTOCOL) != signoff["protocol_sha256"]
        or source_sha256() != signoff["prepare_clean_v2_data_source_sha256"]
        or validate_phase4_signoff(phase4_signoff_path, role=role)["signoff_sha256"]
        != signoff["signoff_sha256"]
    ):
        raise CleanDataError("reviewed source changed during preparation")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output_dir.name}.partial-", dir=output_dir.parent) as raw:
        temporary = Path(raw)
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (temporary / "selected_hashes.json").write_text(
            json.dumps(manifest["selected_text_sha256"], indent=2) + "\n", encoding="utf-8"
        )
        (temporary / "historical_exclusion_registry.json").write_text(
            json.dumps(exclusion_registry, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.rename(output_dir)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("calibration", "sealed"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--phase4-signoff", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = prepare(
        role=args.role,
        output_dir=args.output_dir,
        phase4_signoff_path=args.phase4_signoff,
        cache_dir=args.cache_dir,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
