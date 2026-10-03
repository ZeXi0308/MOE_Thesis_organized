#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolves forward references through sys.modules while the
    # module body is executing.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


prepare = load_module(HERE / "prepare_dev_workloads.py", "prepare_dev_workloads")
BCRD_EXPERIMENTS = HERE.parents[2] / "bcrd" / "experiments"
sys.path.insert(0, str(BCRD_EXPERIMENTS))
producer = load_module(
    BCRD_EXPERIMENTS / "capture_continuous_decode.py",
    "capture_continuous_decode",
)


class DevWorkloadTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.workloads = prepare.materialize(HERE / "olmoe_dev_workload.json")

    def test_materialized_workloads_pass_producer_contract(self) -> None:
        self.assertEqual(set(self.workloads), {"olmoe-dev-steady", "olmoe-dev-bursty"})
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for episode_id, workload in self.workloads.items():
                self.assertEqual(workload["run_class"], "development")
                self.assertEqual(len(workload["requests"]), 16)
                self.assertEqual(workload["generation"]["max_decode_steps"], 8)
                self.assertEqual(workload["scheduler"]["max_batch_size"], 4)
                self.assertEqual(
                    workload["scheduler"]["arrival_trace_sha256"],
                    prepare.arrival_trace_sha256(workload["requests"]),
                )
                path = root / f"{episode_id}.json"
                path.write_text(json.dumps(workload), encoding="utf-8")
                loaded = producer.load_workload_manifest(path)
                self.assertEqual(loaded["expected_requests"], 16)

    def test_episode_request_and_document_identities_are_disjoint(self) -> None:
        steady = self.workloads["olmoe-dev-steady"]["requests"]
        bursty = self.workloads["olmoe-dev-bursty"]["requests"]
        self.assertFalse(
            {row["request_id"] for row in steady}
            & {row["request_id"] for row in bursty}
        )
        self.assertFalse(
            {row["document_id"] for row in steady}
            & {row["document_id"] for row in bursty}
        )
        self.assertFalse(
            {row["sample_id"] for row in steady}
            & {row["sample_id"] for row in bursty}
        )

if __name__ == "__main__":
    unittest.main()
