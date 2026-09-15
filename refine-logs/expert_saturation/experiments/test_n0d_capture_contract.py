#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


MODULE_PATH = Path(__file__).with_name("n0d_capture_contract.py")
SPEC = importlib.util.spec_from_file_location("n0d_capture_contract_under_test", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot import n0d capture-contract helper")
CONTRACT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CONTRACT)


REQUEST_IDS = tuple("req-{0}".format(index) for index in range(4))
DECODE_STEPS = 8


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(root: Path) -> str:
    workload = {
        "schema": "bcrd-continuous-workload-v1",
        "run_class": "development",
        "serial_audit_request_ids": list(REQUEST_IDS),
        "requests": [{"request_id": request_id} for request_id in REQUEST_IDS],
        "route_capacity_envelope": {
            "episode_id": "olmoe-dev-steady",
            "arrival_regime": "steady",
            "serial_route_identity_semantics": "per_layer_expert_assignment_multiset",
        },
    }
    audit = {
        "status": "PASS_TOKEN_PARITY_ROUTE_BATCH_DEPENDENT",
        "token_match_fraction": 1.0,
        "route_identity_semantics": "per_layer_expert_assignment_multiset",
        "batch_dependent_route_observed": True,
        "scientific_ground_truth": False,
    }
    ledger = []
    for request_index, request_id in enumerate(REQUEST_IDS):
        ledger.append(
            {
                "request_id": request_id,
                "steps": [
                    {
                        "decode_step": step,
                        "input_token_id": request_index * 100 + step,
                        "predicted_next_token_id": request_index * 100 + step + 1,
                    }
                    for step in range(DECODE_STEPS)
                ],
            }
        )
    _write_json(root / "workload_manifest.json", workload)
    _write_json(root / "serial_audit.json", audit)
    (root / "request_ledger.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in ledger),
        encoding="utf-8",
    )
    (root / "routes.csv").write_text("header\n", encoding="utf-8")
    (root / "decode_batches.jsonl").write_text("{}\n", encoding="utf-8")
    _write_json(root / "preregistration.json", {})
    _write_json(root / "environment.json", {})
    files = {
        name: _sha(root / name) for name in sorted(CONTRACT.CAPTURE_FILES)
    }
    complete = {
        "schema": "bcrd-continuous-capture-complete-v1",
        "status": "CAPTURE_COMPLETE",
        "run_class": "development",
        "workload_manifest_sha256": files["workload_manifest.json"],
        "serial_audit": audit,
        "files": files,
    }
    _write_json(root / "RUN_STATUS.json", {
        "status": "COMPLETE",
        "required_sentinel": "CAPTURE_COMPLETE.json",
    })
    _write_json(root / "CAPTURE_COMPLETE.json", complete)
    return files["workload_manifest.json"]


class CaptureContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.workload_sha = _fixture(self.root)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def load(self) -> dict:
        return CONTRACT.load_capture_contract(
            self.root,
            expected_workload_sha256=self.workload_sha,
            expected_request_ids=REQUEST_IDS,
            decode_steps=DECODE_STEPS,
        )

    def test_valid_contract_rebuilds_exact_reference_tokens(self) -> None:
        contract = self.load()
        self.assertEqual(contract["request_ids"], list(REQUEST_IDS))
        self.assertEqual(len(contract["reference_tokens"]), 32)
        self.assertEqual(
            contract["reference_tokens"][17],
            {
                "request_id": "req-2",
                "decode_step": 1,
                "input_token_id": 201,
                "predicted_next_token_id": 202,
            },
        )
        self.assertTrue(contract["source_batch_dependence"])

    def test_tampered_ledger_fails_seal(self) -> None:
        with (self.root / "request_ledger.jsonl").open("a", encoding="utf-8") as handle:
            handle.write("{}\n")
        with self.assertRaisesRegex(CONTRACT.CaptureContractError, "hash mismatch"):
            self.load()

    def test_serial_audit_must_equal_sentinel_copy(self) -> None:
        complete_path = self.root / "CAPTURE_COMPLETE.json"
        complete = json.loads(complete_path.read_text(encoding="utf-8"))
        complete["serial_audit"]["batch_dependent_route_observed"] = False
        _write_json(complete_path, complete)
        with self.assertRaisesRegex(CONTRACT.CaptureContractError, "serial audit differs"):
            self.load()

    def test_workload_must_match_expected_hash(self) -> None:
        with self.assertRaisesRegex(CONTRACT.CaptureContractError, "frozen N0d"):
            CONTRACT.load_capture_contract(
                self.root,
                expected_workload_sha256="0" * 64,
                expected_request_ids=REQUEST_IDS,
                decode_steps=DECODE_STEPS,
            )

    def test_exact_seven_file_set_is_required(self) -> None:
        complete_path = self.root / "CAPTURE_COMPLETE.json"
        complete = json.loads(complete_path.read_text(encoding="utf-8"))
        complete["files"]["extra.json"] = "0" * 64
        _write_json(complete_path, complete)
        with self.assertRaisesRegex(CONTRACT.CaptureContractError, "seven-file"):
            self.load()


if __name__ == "__main__":
    unittest.main()

