"""Synthetic runtime-limit archive and fail-closed continuation checks."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile
import unittest

import audit_g64_perf_three_arm as base
import audit_g64_perf_r02_partial as target
from test_audit_g64_perf_three_arm import fixture, put


def partial_fixture(root: Path):
    pilot_session, pilot_plan_sha = fixture(root)
    pilot_receipt_path = pilot_session / "receipt.json"
    pilot_receipt = json.loads(pilot_receipt_path.read_text())
    pilot_receipt["finished_unix_s"] = 10
    put(pilot_receipt_path, pilot_receipt)
    pilot_audit_path = root / "pilot-audit.json"
    put(pilot_audit_path, base.audit(pilot_session, pilot_plan_sha))
    pilot_audit_sha = base.sha_file(pilot_audit_path)
    r02 = root / "r02"
    r02.mkdir()
    pplan = json.loads((pilot_session / "plan.json").read_text())
    plan = dict(pplan)
    plan["session_dir"] = "/remote/r02"
    plan["cells"] = [{"arm": arm, "package_dir": "/remote/candidate_g64_perf_r01",
                      "output_dir": f"/remote/r02-output-{arm}", "max_wall_seconds": 900}
                     for arm in target.PLANNED_ARMS]
    put(r02 / "plan.json", plan)
    r02_plan_sha = base.sha_file(r02 / "plan.json")
    for name in ("model-identity-receipt.json", "model-private-view-receipt.json",
                 "model-offline-resolution-receipt.json"):
        shutil.copy2(pilot_session / name, r02 / name)
    src = pilot_session / "cell-02-ltr_t30_q10/archive"
    cell_dir = r02 / "cell-00-ltr_t30_q1"
    shutil.copytree(src, cell_dir / "archive")
    archive = cell_dir / "archive"
    config_path = archive / "config.json"
    config = json.loads(config_path.read_text())
    config["ltr_config"] = {"threshold": 30, "quantum": 1}
    put(config_path, config)
    installed_path = archive / "selective-store.json"
    installed = json.loads(installed_path.read_text())
    installed["ltr_config"] = {"threshold": 30, "quantum": 1}
    target_id = "measured/memory-train-article-0000464-test"
    installed.update(status="UNINSTALLED_WITH_PENDING_REQUESTS", schedule_calls=2,
                     applied_rotations=0, events=[])
    for step in range(2):
        installed["events"].extend([
            {"event": "accept", "step": step, "host_perf_counter_s": float(step),
             "action": "PRIORITIZE_WAITING", "target_id": target_id},
            {"event": "backend_censored", "step": step,
             "host_perf_counter_s": float(step) + .1,
             "target": target_id, "reason": "PREEMPTED"},
            {"event": "release", "step": step,
             "host_perf_counter_s": float(step) + .2, "target": target_id,
             "reason": "native admission censored; rescan without resetting counters"},
        ])
    put(installed_path, installed)
    commands_path = archive / "commands.txt"
    commands_path.write_text("/remote/python run_ltr_style.py --ltr-threshold 30 --ltr-quantum 1 "
                             "--measurement-mode performance --output-dir /remote/r02-output-ltr_t30_q1\n")
    raw_path = archive / "raw.json"
    raw = json.loads(raw_path.read_text())
    raw["status"] = "INCOMPLETE"
    raw["error"] = "runtime_limit"
    raw["observation_end_s"] = 180.01
    raw["engine_call_count"] = raw["engine_return_count"] = 2
    raw["internal_to_source"] = {target_id: "memory-train-article-0000464"}
    raw["output_events"] = [
        {"received_s": at, "new_token_ids": [token]}
        for row in raw["requests"]
        for at, token in zip(row["token_times_s"], row["output_token_ids"])
    ]
    for row in raw["requests"]:
        row["arrived_at_observation_end"] = True
    raw["requests"][0]["status"] = "unfinished"
    raw["requests"][0]["completion_s"] = None
    raw["requests"][0].pop("stop_reason", None)
    put(raw_path, raw)
    status_path = archive / "status.json"
    status = json.loads(status_path.read_text())
    status.update(status="INCOMPLETE", capture_status="INCOMPLETE",
                  requests_completed=63, finish_reason_counts={"stop": 63, "unfinished": 1},
                  error="RuntimeError: runtime_limit")
    put(status_path, status)
    put(cell_dir / "output_sha256.json", base.archive_hashes(archive))
    preceipt = json.loads(pilot_receipt_path.read_text())
    preceipt.update(status="ABORTED", plan_sha256=r02_plan_sha, started_unix_s=20,
                    finished_unix_s=22, error="RuntimeError: cell 0 failed")
    preceipt["cells"] = [{"arm": "ltr_t30_q1", "output_dir": plan["cells"][0]["output_dir"],
                          "argv": [plan["cells"][0]["package_dir"] + "/pkg/run.sh", "ltr_t30_q1",
                                   plan["cells"][0]["output_dir"]],
                          "launch_status": "FINISHED", "exit_code": 1, "timed_out": False,
                          "archive_status": "VERIFIED", "gpu_process_state_after": "EMPTY",
                          "gpu_process_rows_after": []}]
    put(r02 / "receipt.json", preceipt)
    return pilot_session, pilot_audit_path, pilot_plan_sha, pilot_audit_sha, r02, r02_plan_sha


class PartialTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.args = partial_fixture(Path(tmp.name))

    def audit(self):
        return target.audit(*self.args)

    def test_runtime_limit_preserves_incomplete_cohort_and_no_selection(self):
        result = self.audit()
        self.assertEqual(result["calibration_status"], "INCOMPLETE_GRID_NO_SELECTION")
        self.assertIsNone(result["selected_ltr_arm"])
        self.assertEqual(result["t30_q1_partial"]["completed"], 63)
        self.assertEqual(result["t30_q1_partial"]["unfinished"], 1)
        self.assertGreater(result["t30_q1_partial"]["mean_flow_with_incomplete_penalty_s"],
                           result["t30_q1_partial"]["mean_completed_flow_s"])
        self.assertEqual(result["liveness_signature"]["terminal_repeated_triad"]
                         ["consecutive_steps"], 2)
        self.assertEqual(result["source_localization"]["native_scheduler_sha256"],
                         base.PINNED_RUNTIME_SHA["v1/core/sched/scheduler.py"])

    def test_changed_archive_byte_rejected(self):
        archive = self.args[4] / "cell-00-ltr_t30_q1/archive"
        (archive / "raw.json").write_text("{}\n")
        with self.assertRaisesRegex(base.AuditError, "archive file set"):
            self.audit()

    def test_missing_request_even_with_rehashed_archive_rejected(self):
        cell = self.args[4] / "cell-00-ltr_t30_q1"
        raw_path = cell / "archive/raw.json"
        raw = json.loads(raw_path.read_text())
        raw["requests"].pop()
        put(raw_path, raw)
        put(cell / "output_sha256.json", base.archive_hashes(cell / "archive"))
        with self.assertRaisesRegex(base.AuditError, "capture clock or cohort size"):
            self.audit()

    def test_invalid_physical_kv_even_with_rehashed_archive_rejected(self):
        cell = self.args[4] / "cell-00-ltr_t30_q1"
        put(cell / "archive/memory-after-init.json", {"kv_storage_bytes": 1})
        put(cell / "output_sha256.json", base.archive_hashes(cell / "archive"))
        with self.assertRaisesRegex(base.AuditError, "physical GPU KV"):
            self.audit()

    def test_second_cell_receipt_rejected_after_first_failure(self):
        path = self.args[4] / "receipt.json"
        rec = json.loads(path.read_text())
        rec["cells"].append(dict(rec["cells"][0], arm="ltr_t200_q1"))
        put(path, rec)
        with self.assertRaisesRegex(base.AuditError, "stop after its first cell"):
            self.audit()

    def test_scheduler_call_mismatch_with_rehashed_archive_rejected(self):
        cell = self.args[4] / "cell-00-ltr_t30_q1"
        path = cell / "archive/selective-store.json"
        installed = json.loads(path.read_text())
        installed["schedule_calls"] += 1
        put(path, installed)
        put(cell / "output_sha256.json", base.archive_hashes(cell / "archive"))
        with self.assertRaisesRegex(base.AuditError, "scheduler event clock"):
            self.audit()

    def test_unarrived_request_with_rehashed_archive_rejected(self):
        cell = self.args[4] / "cell-00-ltr_t30_q1"
        path = cell / "archive/raw.json"
        raw = json.loads(path.read_text())
        raw["requests"][0]["arrived_at_observation_end"] = False
        put(path, raw)
        put(cell / "output_sha256.json", base.archive_hashes(cell / "archive"))
        with self.assertRaisesRegex(base.AuditError, "capture cohort or status"):
            self.audit()


if __name__ == "__main__":
    unittest.main()
