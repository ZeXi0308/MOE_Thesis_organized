"""Synthetic CPU checks for the H1 diagnostic archive and NO_ACTION gate."""
from __future__ import annotations

import json
import importlib.util
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import audit_h1_guard_qualification_r02 as target
import test_audit_h128_guarded_transfer_r02 as h128_fixture


def put(path: Path, value) -> None:
    path.write_text(json.dumps(value) + "\n")


def fixture(root: Path) -> tuple[Path, str]:
    old_session, _ = h128_fixture.fixture(root)
    session = old_session.rename(root / target.SESSION_NAME)
    for cell in (session / "cell-00-native_full_native", session / "cell-02-ltr_t200_q1"):
        shutil.rmtree(cell)
    old_cell = session / "cell-01-eager"
    cell_dir = old_cell.rename(session / f"cell-00-{target.ARM}")
    archive = cell_dir / "archive"

    plan = json.loads((session / "plan.json").read_text())
    receipt = json.loads((session / "receipt.json").read_text())
    plan.update(authorized_gpu_uuid=target.EXPECTED_GPU_UUID,
        approved_host_bytes=96636764160, approved_total_wall_seconds=2100,
        lock_path="/root/autodl-tmp/moe-research-gpu.lock",
        expected_lock_device_inode=target.EXPECTED_LOCK_INODE,
        session_dir=target.EXPECTED_SESSION, python=target.EXPECTED_PYTHON,
        hf_cache_dir=target.EXPECTED_MODEL_CACHE,
        shared_model_source_cache=target.EXPECTED_SOURCE_CACHE,
        model_verifier_path=target.EXPECTED_MODEL_VERIFIER,
        model_verifier_sha256=target.EXPECTED_MODEL_VERIFIER_SHA,
        cgroup_memory_max_file="/sys/fs/cgroup/memory.max")
    plan["cells"] = [{"arm": target.ARM, "package_dir": target.EXPECTED_PACKAGE,
        "output_dir": target.EXPECTED_OUTPUT, "max_wall_seconds": 900}]
    put(session / "plan.json", plan)
    plan_sha = target.sha_file(session / "plan.json")
    receipt["plan_sha256"] = plan_sha
    receipt["held_lock_device_inode"] = target.EXPECTED_LOCK_INODE
    receipt["model_identity"]["verifier_sha256"] = target.EXPECTED_MODEL_VERIFIER_SHA
    cell = receipt["cells"][1]
    cell.update(arm=target.ARM, output_dir=target.EXPECTED_OUTPUT,
        argv=[target.EXPECTED_PACKAGE + "/pkg/run.sh", "eager", "diagnostic", "on",
              target.EXPECTED_OUTPUT])
    receipt["cells"] = [cell]
    put(session / "receipt.json", receipt)
    shared = json.loads((session / "model-identity-receipt.json").read_text())
    shared["cache"] = target.EXPECTED_SOURCE_CACHE
    put(session / "model-identity-receipt.json", shared)
    private = json.loads((session / "model-private-view-receipt.json").read_text())
    private["private_cache"] = target.EXPECTED_MODEL_CACHE
    private["source_cache"] = target.EXPECTED_SOURCE_CACHE
    put(session / "model-private-view-receipt.json", private)
    offline = json.loads((session / "model-offline-resolution-receipt.json").read_text())
    offline["hf_home"] = target.EXPECTED_MODEL_CACHE
    offline["path"] = target.EXPECTED_MODEL_CACHE + "/hub/" + target.common.REVISION + "/config.json"
    put(session / "model-offline-resolution-receipt.json", offline)

    manifest = target.verify_local_package()
    environment = json.loads((archive / "environment.json").read_text())
    environment["source_sha256"] = {name: manifest[f"pkg/{name}"] for name in target.SOURCE_NAMES}
    put(archive / "environment.json", environment)
    config = json.loads((archive / "config.json").read_text())
    config.update(measurement_mode="diagnostic", variant="eager", store_scope="selected",
        commit_recheck=True, population_mode="open", global_cooldown_steps=0,
        rotation_victim_order="most_output", rotation_config={"min_steps_between_swaps": 0})
    put(archive / "config.json", config)
    raw = json.loads((archive / "raw.json").read_text())
    raw.update(measurement_origin_perf_counter_s=1.0,
        scheduler_steps=[{"step": 0, "start_s": 0.0, "scheduled": [], "preempted_request_ids": []}],
        engine_steps=[{"call_index": 0}], memory_trace=[{"attempted_step": 0}])
    put(archive / "raw.json", raw)
    installed = json.loads((archive / "selective-store.json").read_text())
    installed.update(status="DRAINED", store_scope="selected", native_calc_overridden=True,
        commit_recheck=True, diagnostic=True, population_mode="open",
        rotation_config=config["rotation_config"], events=[], direct_commits=0,
        native_reservation_gate={"checked": 0, "zero": 0,
                                 "positive_keep": 0, "unknown_keep": 0})
    put(archive / "selective-store.json", installed)
    put(archive / "offload-events.json", {"diagnostic": True,
        "detailed_logging": "ENABLED", "worker_wrapped": True,
        "lookup": [], "transfers": [], "completed_jobs": [], "dispatch": [],
        "host_snapshots": []})
    status = json.loads((archive / "status.json").read_text())
    status["direct_commits"] = 0
    put(archive / "status.json", status)
    put(archive / "metrics.json", {"direct_commits": 0,
        "complete_episode_comparison_eligible": True})
    for name in ("resolved-eos.json", "memory-before.json", "memory-after.json",
                 "gpu-after.json"):
        put(archive / name, {})
    (archive / "commands.txt").write_text(
        f"/remote/python run_recovery_cadence.py --variant eager --measurement-mode diagnostic "
        f"--output-dir {target.EXPECTED_OUTPUT} --commit-recheck\n")
    (cell_dir / "launch.log").write_text("Candidate payload verified: 25 files\n")
    put(cell_dir / "output_sha256.json", target.archive_hashes(archive))
    return session, plan_sha


class H1AuditTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.session, self.plan_sha = fixture(Path(temporary.name))
        self.patch = patch.object(target, "PLAN_SHA", self.plan_sha)
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def rehash(self):
        cell = self.session / f"cell-00-{target.ARM}"
        put(cell / "output_sha256.json", target.archive_hashes(cell / "archive"))

    def test_complete_diagnostic_with_no_direct_action_stops_extension(self):
        result = target.audit(self.session, self.plan_sha)
        self.assertEqual(result["status"], "NO_ACTION")
        self.assertEqual(result["qualification"], "NO_ACTION_STOP")
        self.assertEqual(result["requests_completed"], 128)
        self.assertEqual(result["archive_file_count"], 27)

    @unittest.skipUnless(importlib.util.find_spec("ijson"), "ijson parser unavailable")
    def test_stream_projection_matches_small_original_fields_and_counts(self):
        path = self.session / f"cell-00-{target.ARM}/archive/raw.json"
        original = json.loads(path.read_text())
        original["output_events"] = [{"request_id": "a", "cumulative": [1, 2, 3]}]
        put(path, original)
        projected, provenance = target.read_raw_projection(path, force_stream=True)
        for key in target.RAW_SCALARS + target.RAW_RETAINED_ARRAYS:
            self.assertEqual(projected[key], original[key])
        self.assertNotIn("output_events", projected)
        self.assertNotIn("memory_trace", projected)
        self.assertEqual(provenance["reader"], "ijson_single_pass")
        self.assertEqual(provenance["counts"], {
            key: len(original[key]) for key in target.RAW_RETAINED_ARRAYS +
            target.RAW_COUNTED_ARRAYS
        })

    def test_changed_raw_without_hash_map_update_is_rejected(self):
        raw_path = self.session / f"cell-00-{target.ARM}/archive/raw.json"
        raw_path.write_text("{}\n")
        with self.assertRaisesRegex(target.AuditError, "archive file hash"):
            target.audit(self.session, self.plan_sha)

    def test_resource_drift_with_valid_new_hash_is_rejected(self):
        cell = self.session / f"cell-00-{target.ARM}"
        put(cell / "archive/memory-after-init.json", {"kv_storage_bytes": target.common.KV_BYTES - 1})
        self.rehash()
        with self.assertRaisesRegex(target.AuditError, "physical GPU KV"):
            target.audit(self.session, self.plan_sha)

    def test_cohort_drift_with_valid_new_hash_is_rejected(self):
        cell = self.session / f"cell-00-{target.ARM}"
        path = cell / "archive/raw.json"
        raw = json.loads(path.read_text())
        raw["requests"][0]["prompt_token_ids_sha256"] = "f" * 64
        put(path, raw)
        self.rehash()
        with self.assertRaisesRegex(target.AuditError, "capture cohort"):
            target.audit(self.session, self.plan_sha)

    def test_diagnostic_mode_drift_with_valid_new_hash_is_rejected(self):
        cell = self.session / f"cell-00-{target.ARM}"
        path = cell / "archive/config.json"
        config = json.loads(path.read_text())
        config["measurement_mode"] = "performance_sparse_preemptions"
        put(path, config)
        self.rehash()
        with self.assertRaisesRegex(target.AuditError, "diagnostic config differs"):
            target.audit(self.session, self.plan_sha)


if __name__ == "__main__":
    unittest.main()
