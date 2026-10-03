#!/usr/bin/env python3
"""Fail-closed loader for the sealed N0d continuous-decode capture.

This module deliberately uses only the Python standard library.  The evaluator
can therefore bind its reference tokens to retained capture bytes without
loading PyTorch, Transformers, or any model code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence


SCHEMA = "n0d-sealed-capture-contract-v1"
EXPECTED_WORKLOAD_SHA256 = (
    "47babe9d8f875fda3457a68ca83ee7d1274866ebc47013622691d1fc1b556a6d"
)
EXPECTED_REQUEST_IDS = tuple(
    "olmoe-dev-steady-{index:03d}".format(index=index) for index in range(4)
)
DECODE_STEPS = 8
CAPTURE_FILES = frozenset(
    {
        "routes.csv",
        "decode_batches.jsonl",
        "request_ledger.jsonl",
        "workload_manifest.json",
        "preregistration.json",
        "environment.json",
        "serial_audit.json",
    }
)


class CaptureContractError(RuntimeError):
    """The retained capture does not satisfy the frozen N0d contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise CaptureContractError("cannot hash {0}: {1}".format(path, exc)) from exc
    return digest.hexdigest()


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CaptureContractError("{0} must be an object".format(label))
    return value


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CaptureContractError("cannot read {0}: {1}".format(label, exc)) from exc
    return dict(_mapping(value, label))


def _require_sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise CaptureContractError("{0} is not a lowercase SHA-256".format(label))
    return value


def _require_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CaptureContractError("{0} must be a non-negative integer".format(label))
    return value


def _sealed_paths(
    capture_dir: Path, complete: Mapping[str, Any]
) -> dict[str, Path]:
    files = _mapping(complete.get("files"), "CAPTURE_COMPLETE.files")
    if set(files) != set(CAPTURE_FILES):
        raise CaptureContractError("capture does not satisfy the exact seven-file set")
    paths: dict[str, Path] = {}
    for name in sorted(CAPTURE_FILES):
        expected = _require_sha(files.get(name), "CAPTURE_COMPLETE.files.{0}".format(name))
        path = capture_dir / name
        if not path.is_file():
            raise CaptureContractError("sealed capture file is missing: {0}".format(name))
        if sha256_file(path) != expected:
            raise CaptureContractError("sealed capture file hash mismatch: {0}".format(name))
        paths[name] = path
    return paths


def _load_reference_tokens(
    ledger_path: Path,
    request_ids: Sequence[str],
    decode_steps: int,
) -> list[dict[str, Any]]:
    rows: dict[str, Mapping[str, Any]] = {}
    try:
        with ledger_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                raw = json.loads(line)
                row = _mapping(raw, "request ledger line {0}".format(line_number))
                request_id = row.get("request_id")
                if not isinstance(request_id, str) or not request_id or request_id in rows:
                    raise CaptureContractError(
                        "request ledger IDs are empty, non-string, or duplicated"
                    )
                rows[request_id] = row
    except (OSError, json.JSONDecodeError) as exc:
        raise CaptureContractError("cannot parse sealed request ledger: {0}".format(exc)) from exc

    reference: list[dict[str, Any]] = []
    for request_id in request_ids:
        row = rows.get(request_id)
        if row is None:
            raise CaptureContractError(
                "request ledger is missing selected request {0}".format(request_id)
            )
        steps = row.get("steps")
        if not isinstance(steps, list) or len(steps) < decode_steps:
            raise CaptureContractError(
                "captured request {0} has fewer than {1} steps".format(
                    request_id, decode_steps
                )
            )
        for expected_step, raw_step in enumerate(steps[:decode_steps]):
            step = _mapping(
                raw_step,
                "ledger {0} step {1}".format(request_id, expected_step),
            )
            observed_step = _require_int(
                step.get("decode_step"),
                "ledger {0} decode_step".format(request_id),
            )
            if observed_step != expected_step:
                raise CaptureContractError("request ledger decode-step identity drifted")
            reference.append(
                {
                    "request_id": request_id,
                    "decode_step": expected_step,
                    "input_token_id": _require_int(
                        step.get("input_token_id"),
                        "ledger {0} input token".format(request_id),
                    ),
                    "predicted_next_token_id": _require_int(
                        step.get("predicted_next_token_id"),
                        "ledger {0} predicted token".format(request_id),
                    ),
                }
            )
    if len(reference) != len(request_ids) * decode_steps:
        raise CaptureContractError("reference-token cardinality did not close")
    return reference


def load_capture_contract(
    capture_dir: Path,
    *,
    expected_workload_sha256: str,
    expected_request_ids: Sequence[str],
    decode_steps: int,
    require_batch_dependence: bool = True,
) -> dict[str, Any]:
    """Load one sealed capture and independently reconstruct token reference rows."""

    root = Path(capture_dir).resolve()
    if not root.is_dir():
        raise CaptureContractError("capture directory does not exist: {0}".format(root))
    expected_workload = _require_sha(
        expected_workload_sha256, "expected workload SHA-256"
    )
    request_ids = tuple(str(value) for value in expected_request_ids)
    if (
        not request_ids
        or len(set(request_ids)) != len(request_ids)
        or decode_steps <= 0
    ):
        raise CaptureContractError("expected request/decode identity is invalid")

    run_status_path = root / "RUN_STATUS.json"
    run_status = _read_json(run_status_path, "RUN_STATUS.json")
    if run_status != {
        "status": "COMPLETE",
        "required_sentinel": "CAPTURE_COMPLETE.json",
    }:
        raise CaptureContractError("capture RUN_STATUS is not the closed COMPLETE marker")

    complete_path = root / "CAPTURE_COMPLETE.json"
    complete = _read_json(complete_path, "CAPTURE_COMPLETE.json")
    if (
        complete.get("schema") != "bcrd-continuous-capture-complete-v1"
        or complete.get("status") != "CAPTURE_COMPLETE"
        or complete.get("run_class") != "development"
    ):
        raise CaptureContractError("capture is not a completed development capture")
    paths = _sealed_paths(root, complete)

    workload_hash = sha256_file(paths["workload_manifest.json"])
    if workload_hash != expected_workload:
        raise CaptureContractError("sealed workload differs from the frozen N0d SHA-256")
    if complete.get("workload_manifest_sha256") != workload_hash:
        raise CaptureContractError("capture sentinel workload hash does not close")
    workload = _read_json(paths["workload_manifest.json"], "workload_manifest.json")
    if (
        workload.get("schema") != "bcrd-continuous-workload-v1"
        or workload.get("run_class") != "development"
        or tuple(workload.get("serial_audit_request_ids", ()))[: len(request_ids)]
        != request_ids
    ):
        raise CaptureContractError("sealed workload request identity does not match N0d")
    known_ids = {
        str(row.get("request_id"))
        for row in workload.get("requests", ())
        if isinstance(row, Mapping)
    }
    if not set(request_ids).issubset(known_ids):
        raise CaptureContractError("selected N0d request is absent from sealed workload")
    marker = _mapping(
        workload.get("route_capacity_envelope"), "workload.route_capacity_envelope"
    )
    if (
        marker.get("episode_id") != "olmoe-dev-steady"
        or marker.get("arrival_regime") != "steady"
        or marker.get("serial_route_identity_semantics")
        != "per_layer_expert_assignment_multiset"
    ):
        raise CaptureContractError("sealed workload is not the frozen steady episode")

    serial_audit = _read_json(paths["serial_audit.json"], "serial_audit.json")
    if complete.get("serial_audit") != serial_audit:
        raise CaptureContractError(
            "capture sentinel serial audit differs from sealed serial_audit.json"
        )
    source_batch_dependence = serial_audit.get("batch_dependent_route_observed")
    if not isinstance(source_batch_dependence, bool):
        raise CaptureContractError("serial audit batch-dependence field is not boolean")
    if require_batch_dependence and (
        source_batch_dependence is not True
        or serial_audit.get("status") != "PASS_TOKEN_PARITY_ROUTE_BATCH_DEPENDENT"
        or serial_audit.get("token_match_fraction") != 1.0
        or serial_audit.get("route_identity_semantics")
        != "per_layer_expert_assignment_multiset"
        or serial_audit.get("scientific_ground_truth") is not False
    ):
        raise CaptureContractError("sealed serial audit does not qualify N0d")

    reference_tokens = _load_reference_tokens(
        paths["request_ledger.jsonl"], request_ids, decode_steps
    )
    encoded_reference = json.dumps(
        reference_tokens, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return {
        "schema": SCHEMA,
        "capture_dir": str(root),
        "capture_complete_sha256": sha256_file(complete_path),
        "run_status_sha256": sha256_file(run_status_path),
        "files_sha256": {
            name: sha256_file(path) for name, path in sorted(paths.items())
        },
        "workload_manifest_sha256": workload_hash,
        "serial_audit_sha256": sha256_file(paths["serial_audit.json"]),
        "request_ledger_sha256": sha256_file(paths["request_ledger.jsonl"]),
        "serial_audit": serial_audit,
        "source_batch_dependence": source_batch_dependence,
        "request_ids": list(request_ids),
        "decode_steps": decode_steps,
        "reference_tokens": reference_tokens,
        "reference_tokens_sha256": hashlib.sha256(encoded_reference).hexdigest(),
    }


def load_n0d_capture_contract(capture_dir: Path) -> dict[str, Any]:
    """Load the exact frozen N0d capture contract."""

    return load_capture_contract(
        capture_dir,
        expected_workload_sha256=EXPECTED_WORKLOAD_SHA256,
        expected_request_ids=EXPECTED_REQUEST_IDS,
        decode_steps=DECODE_STEPS,
        require_batch_dependence=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", required=True)
    args = parser.parse_args()
    contract = load_n0d_capture_contract(Path(args.capture_dir))
    print(
        json.dumps(
            {
                "status": "N0D_CAPTURE_CONTRACT_VALID",
                "capture_complete_sha256": contract["capture_complete_sha256"],
                "workload_manifest_sha256": contract["workload_manifest_sha256"],
                "request_ledger_sha256": contract["request_ledger_sha256"],
                "reference_tokens_sha256": contract["reference_tokens_sha256"],
                "source_batch_dependence": contract["source_batch_dependence"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
