"""Synthetic corrected T200 session checks for integrity and paired cohort gates."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import audit_g64_guarded_t200_performance_r01 as target


def put(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


def fixture(root: Path) -> tuple[Path, str]:
    session = root / target.SESSION_NAME
    session.mkdir()
    package = Path(target.__file__).parent / "candidate_g64_perf_guard_t200_r01/pkg"
    input_config = json.loads((package / "inputs/config.json").read_text())
    workload = json.loads((package / "inputs/workload.json").read_text())
    manifest = json.loads((package.parent / "manifest.json").read_text())
    plan = {"schema_version": 1, "session_dir": "/remote/session", "approved_host_bytes": 96636764160,
            "expected_lock_device_inode": "2304:25841682495",
            "authorized_gpu_uuid": "GPU-3fc910c2-bf65-5273-e6b5-6c0d8b6ce03e",
            "model_revision": target.REVISION, "model_verifier_sha256": "a" * 64,
            "shared_model_source_cache": "/remote/shared", "hf_cache_dir": "/remote/private",
            "cells": [{"arm": arm, "package_dir": "/remote/candidate_g64_perf_guard_t200_r01",
                       "output_dir": f"/remote/output-{arm}", "max_wall_seconds": 900}
                      for arm in target.ARMS]}
    put(session / "plan.json", plan)
    plan_sha = target.sha_file(session / "plan.json")
    receipt = {"plan_sha256": plan_sha, "status": "CELLS_COMPLETE",
               "held_lock_device_inode": plan["expected_lock_device_inode"],
               "model_identity": {"status": "VERIFIED_PRIVATE_MODEL_VIEW",
                                  "shared_source_receipt": "model-identity-receipt.json",
                                  "private_view_receipt": "model-private-view-receipt.json",
                                  "offline_resolution_receipt": "model-offline-resolution-receipt.json",
                                  "verifier_sha256": "a" * 64}, "cells": []}
    model_hashes = target.MODEL_METADATA_SHA
    shards = target.MODEL_SHARDS
    put(session / "model-identity-receipt.json",
        {"status": "VERIFIED_READ_ONLY", "cache": "/remote/shared", "revision": target.REVISION,
         "metadata_sha256": model_hashes, "shards": shards})
    put(session / "model-private-view-receipt.json",
        {"status": "VERIFIED_PRIVATE_MODEL_VIEW", "private_cache": "/remote/private",
         "source_cache": "/remote/shared", "revision": target.REVISION,
         "metadata_sha256": model_hashes, "shards": shards})
    put(session / "model-offline-resolution-receipt.json",
        {"status": "RESOLVED_OFFLINE", "hf_home": "/remote/private",
         "path": "/remote/private/hub/" + target.REVISION + "/config.json"})
    engine = {"model": target.MODEL_ID, "revision": target.REVISION,
              "tokenizer_revision": target.REVISION, "dtype": "bfloat16", "seed": 20260905,
              "max_model_len": 4096, "max_num_seqs": 32, "max_num_batched_tokens": 1024,
              "kv_cache_memory_bytes": target.KV_BYTES, "kv_offloading_size": 16,
              "kv_offloading_backend": "native", "enable_prefix_caching": False,
              "scheduler_reserve_full_isl": True, "scheduling_policy": "fcfs",
              "async_scheduling": False, "stream_interval": 1}
    for index, arm in enumerate(target.ARMS):
        cell = plan["cells"][index]
        receipt["cells"].append({"arm": arm, "output_dir": cell["output_dir"],
            "argv": [cell["package_dir"] + "/pkg/run.sh", arm, cell["output_dir"]],
            "launch_status": "FINISHED", "exit_code": 0, "timed_out": False,
            "archive_status": "VERIFIED", "gpu_process_state_after": "EMPTY",
            "gpu_process_rows_after": []})
        archive = session / f"cell-{index:02d}-{arm}" / "archive"
        archive.mkdir(parents=True)
        config = copy.deepcopy(input_config)
        config.update({"requests": 64, "seed": 20260905, "output_tokens": 1024,
            "output_mode": "eos", "ignore_eos": False, "min_tokens": 0, "cap": 32,
            "engine_max_num_seqs": 32, "max_num_batched_tokens": 1024,
            "max_seconds": 180, "fixed_kv_cache_memory_bytes": target.KV_BYTES,
            "offload_gib": 16, "measurement_mode": "performance_sparse_preemptions",
            "variant": "ltr_style_selected" if arm.startswith("ltr_") else arm,
            "store_scope": "selected"})
        if arm.startswith("ltr_"):
            config["ltr_config"] = {"threshold": 200, "quantum": 1 if arm.endswith("q1") else 10}
        else:
            config["ltr_config"] = None
            config["commit_recheck"] = False
        rows = []
        for i, (source, prompt) in enumerate(zip(workload["source_requests"],
                                                 workload["actual_prompt_token_ids"])):
            arrival = workload["arrival_traces_s"]["steady"][i]
            rows.append({"request_id": source["request_id"], "document_id": source["document_id"],
                "prompt_token_ids_sha256": source["prompt_token_ids_sha256"],
                "arrival_s": arrival, "prompt_tokens": len(prompt), "max_output_tokens": 1024,
                "status": "completed", "completion_s": arrival + 2,
                "stop_reason": "stop", "output_token_ids": [index + 1, 42],
                "token_times_s": [arrival + 1, arrival + 2]})
        raw = {"status": "COMPLETE", "error": None, "regime": "steady",
               "arrival_scale": 1.0, "requests": rows, "observation_end_s": 16.0}
        runner = "run_ltr_style.py" if arm.startswith("ltr_") else "run_recovery_cadence.py"
        commands = f"/remote/python {runner} --measurement-mode performance --output-dir {cell['output_dir']}\n"
        (archive / "commands.txt").write_text(commands)
        objects = {
            "config.json": config, "engine_args.json": engine,
            "environment.json": {"vllm": "0.26.0", "python": "3.12", "torch": "2.11",
                "cuda": "13", "transformers": "5.15", "gpu_before": {
                    "device": plan["authorized_gpu_uuid"], "compute_processes": ""},
                "source_sha256": {name: manifest[f"pkg/{name}"] for name in target.ARM_SOURCES[arm]},
                "vllm_source_sha256": target.PINNED_RUNTIME_SHA},
            "status.json": {"status": "COMPLETE", "capture_status": "COMPLETE",
                            "requests_completed": 64, "error": None, "forced_rotations": index,
                            "finish_reason_counts": {"stop": 64}},
            "raw.json": raw,
            "safe-cap-qualification.json": {"status": "QUALIFIED", "usable_blocks": 4096,
                "total_blocks": 4097, "block_size": 16, "engine_max_num_seqs": 32,
                "observed_scheduler_reserve_full_isl": True},
            "memory-after-init.json": {"kv_storage_bytes": target.KV_BYTES},
            "warmup-cache-reset.json": {"success": True},
            "selective-store.json": {"store_scope": "selected",
                "native_calc_overridden": True,
                "status": "DRAINED",
                "ltr_config": {"threshold": 200, "quantum": 1 if arm.endswith("q1") else 10}
                              if arm.startswith("ltr_") else None,
                "direct_guard_probe_calls": 1 if arm.startswith("ltr_") else None,
                "direct_guard_checked_step_targets": 1 if arm.startswith("ltr_") else None,
                "direct_guard_rejected_step_targets": 1 if arm == "ltr_t200_q1" else 0,
                "first_direct_guard_rejection": {"reason": "synthetic CPU-only"}
                                                if arm == "ltr_t200_q1" else None},
        }
        host = {"cpu_kv": {"unique_storage_bytes": target.HOST_BYTES},
                "manager": {"capacity_blocks": 8192},
                "parent_cgroup": {"values": {"memory.max": plan["approved_host_bytes"]}}}
        for name in ("host-after-init.json", "host-before.json", "host-request-end.json",
                     "host-after.json"):
            objects[name] = host
        objects["host-after.json"] = {**host,
            "pending": {"scheduler_store_jobs": 0, "scheduler_load_jobs": 0,
                        "pending_worker_acknowledgements": 0},
            "manager": {"capacity_blocks": 8192, "pending_store_entries": 0,
                        "active_load_references": 0}}
        for n, count in enumerate((32, 32, 2)):
            objects[f"warmup-{n}.json"] = {"status": "COMPLETE",
                "requests": [{"status": "completed"} for _ in range(count)]}
        for name in ("warmup-offload-drain.json", "post-request-drain.json"):
            objects[name] = {"calls": 0, "seconds": 0.0}
        for name in ("offload-events.json", "timing.json", "resolved-scheduler-config.json"):
            objects[name] = {}
        for name, value in objects.items():
            put(archive / name, value)
        put(archive.parent / "output_sha256.json", target.archive_hashes(archive))
    put(session / "receipt.json", receipt)
    return session, plan_sha


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.session, self.plan_sha = fixture(Path(self.tmp.name))

    def test_complete_t200_block_requires_t30_for_selection(self):
        result = target.audit(self.session, self.plan_sha)
        self.assertEqual(result["status"], "T200_BLOCK_COMPLETE_SELECTION_REQUIRES_T30")
        self.assertEqual(result["calibration_status"], "REQUIRES_CROSS_BLOCK_G64_SELECTION")
        self.assertEqual(result["metrics"]["eager"]["completed"], 64)
        self.assertEqual(result["comparisons"]["ltr_t200_q10_vs_eager"]
                         ["output_differences"]["different_output_sequence_count"], 64)
        self.assertEqual(result["comparisons"]["ltr_t200_q1_vs_eager"]
                         ["output_differences"]["different_output_sequence_count"], 64)

    def test_wrong_external_plan_sha_rejected(self):
        with self.assertRaisesRegex(target.AuditError, "external SHA"):
            target.audit(self.session, "0" * 64)

    def test_archive_mutation_rejected(self):
        archive = self.session / "cell-01-eager/archive"
        (archive / "raw.json").write_text("{}\n")
        with self.assertRaisesRegex(target.AuditError, "archive hash"):
            target.audit(self.session, self.plan_sha)

    def test_receipt_failed_cell_rejected(self):
        path = self.session / "receipt.json"
        receipt = json.loads(path.read_text())
        receipt["cells"][1]["exit_code"] = 1
        put(path, receipt)
        with self.assertRaisesRegex(target.AuditError, "controller cell"):
            target.audit(self.session, self.plan_sha)

    def test_resource_drift_with_valid_new_hash_rejected(self):
        cell = self.session / "cell-02-ltr_t200_q1"
        path = cell / "archive/memory-after-init.json"
        put(path, {"kv_storage_bytes": target.KV_BYTES - 1})
        put(cell / "output_sha256.json", target.archive_hashes(cell / "archive"))
        with self.assertRaisesRegex(target.AuditError, "physical KV storage"):
            target.audit(self.session, self.plan_sha)

    def test_cohort_drift_with_valid_new_hash_rejected(self):
        cell = self.session / "cell-01-eager"
        path = cell / "archive/raw.json"
        raw = json.loads(path.read_text())
        raw["requests"][0]["prompt_token_ids_sha256"] = "f" * 64
        put(path, raw)
        put(cell / "output_sha256.json", target.archive_hashes(cell / "archive"))
        with self.assertRaisesRegex(target.AuditError, "capture cohort"):
            target.audit(self.session, self.plan_sha)


if __name__ == "__main__":
    unittest.main()
