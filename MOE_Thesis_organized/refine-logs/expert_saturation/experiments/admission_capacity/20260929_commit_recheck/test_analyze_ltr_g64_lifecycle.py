"""Synthetic receipt tests for the read-only G64 native lifecycle auditor."""

from __future__ import annotations

import importlib.util
import ast
import json
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("ltr_audit", HERE / "analyze_ltr_g64_lifecycle.py")
audit_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit_module)


def fixture(folder: Path, *, action: bool = True) -> None:
    pkg = HERE / "candidate_ltr_r02_env"
    manifest = json.loads((pkg / "manifest.json").read_text())
    config = json.loads((pkg / "pkg/inputs/config.json").read_text())
    config.update(output_mode="eos", requests=64, output_tokens=1024,
                  measurement_mode="diagnostic", variant="ltr_style_selected",
                  store_scope="selected", selective_save="on", offload_gib=16,
                  fixed_kv_cache_memory_bytes=audit_module.KV_BYTES,
                  ltr_config=dict(threshold=30, quantum=10))
    source_names = ("run_ltr_style.py", "ltr_style_native.py", "ltr_style_selected.py",
                    "recovery_service_components.py", "native_capture.py",
                    "native_offload_observer.py", "native_store_delta.py")
    sources = {name: manifest[f"pkg/{name}"] for name in source_names}
    runner_tree = ast.parse((pkg / "pkg/run_ltr_style.py").read_text())
    runner_sources = next(ast.literal_eval(node.value) for node in runner_tree.body
                          if isinstance(node, ast.Assign) and any(
                              isinstance(target, ast.Name) and target.id == "RUNTIME_SOURCES"
                              for target in node.targets))
    runtime_sources = {name: "0" * 64 for name in runner_sources}
    runtime_sources.update({
        "v1/core/sched/scheduler.py": "2ed2a550b6558b2495eda845a97ae38bcf0225027b9e25fbf00fc3880c1d3941",
        "v1/core/kv_cache_manager.py": "3f4af8d247f3fe9570b0132818b832b66ae6a2ac12942588828f899f6ff77ccf",
    })
    requests = []
    outputs = []
    mapping = {}
    for n in range(64):
        source, internal = f"s{n}", f"measured/s{n}"
        when = 3.0 if n == 0 else 1.0
        mapping[internal] = source
        requests.append(dict(request_id=source, internal_request_id=internal,
                             status="completed", stop_reason="stop",
                             output_token_ids=[n], token_times_s=[when]))
        outputs.append(dict(request_id=source, received_s=when, chunk_size=1,
                            cumulative_tokens=1,
                            prefix_valid=True, cumulative_token_ids=[n]))
    steps = [dict(step=i, preempted_request_ids=[], scheduled=[]) for i in range(6)]
    engine_steps = [dict(call_index=i, completed=True, scheduler_step_start=i,
                         scheduler_step_end=i + 1,
                         output_request_ids=["measured/s0"] if i == 4 else [])
                    for i in range(6)]
    events = []
    offload = dict(diagnostic=True, detailed_logging="ENABLED", worker_wrapped=True,
                   lookup=[], dispatch=[], completed_jobs=[], transfers=[])
    preemptions = []
    if action:
        target, victim = "measured/s0", "measured/s1"
        def event(kind, step, clock, **values):
            events.append(dict(event=kind, step=step, host_perf_counter_s=clock, **values))
        def alloc(step, clock, amount, remaining):
            event("allocation", step, clock, scheduled={target: amount} if amount else {},
                  active_target=target,
                  boosted={target: dict(priority=-1, quantum_remaining=remaining)})
        event("prepare", 0, 101.0, victim=victim, target=target, saved_tokens=16)
        event("accept", 0, 101.01, action="PREPARE_SELECTED", target_id=target,
              victim_id=victim, quantum_remaining=10)
        event("store_delta", 0, 101.1, status="NEW_STORES_VALID",
              jobs=[dict(job=1, logical_indices=[0], blocks=[10])],
              new_store_blocks=1, prepared_prefix_blocks=1)
        event("metadata", 0, 101.11, stores=[1], loads=[], flush=[],
              jobs=[dict(job_id=1, request=victim, is_store=True)])
        alloc(0, 101.12, 0, 10)
        event("commit_check", 1, 102.0, reason="READY", victim=victim, target=target)
        event("metadata", 1, 102.01, stores=[], loads=[], flush=[1],
              jobs=[dict(job_id=1, request=victim, is_store=True)])
        alloc(1, 102.02, 0, 10)
        event("metadata", 2, 102.25, stores=[], loads=[2], flush=[],
              jobs=[dict(job_id=2, request=target, is_store=False)])
        alloc(2, 102.3, 1, 9)
        event("target_new_output", 3, 103.1, target=target,
              output_tokens=1, quantum_remaining=9)
        alloc(3, 103.2, 0, 9)
        alloc(4, 104.0, 1, 8)
        event("release", 5, 105.0, target=target, reason="terminal")
        steps[1]["preempted_request_ids"] = ["s1"]
        steps[2]["scheduled"] = [dict(internal_request_id=target, scheduled_tokens=1)]
        steps[4]["scheduled"] = [dict(internal_request_id=target, scheduled_tokens=1)]
        preemptions = [dict(original_preemption_returned=True)]
        offload["lookup"] = [dict(request=target, matched=1, asynchronous=True)]
        offload["dispatch"] = [
            dict(job_id=1, request=victim, is_store=True, accepted=True, after_perf_s=101.2),
            dict(job_id=2, request=target, is_store=False, accepted=True, after_perf_s=102.1),
        ]
        offload["completed_jobs"] = [
            dict(time_s=101.4, jobs=[dict(job_id=1, request=victim, is_store=True, count=1)]),
            dict(time_s=102.2, jobs=[dict(job_id=2, request=target, is_store=False, count=1)]),
        ]
    host = dict(status="PARTIAL",
                errors={"memory.peak": "FileNotFoundError: /sys/fs/cgroup/memory.peak"},
                parent_cgroup=dict(values={"memory.current": 50 * 1024**3,
                                           "memory.max": 96636764160,
                                           "memory.peak": None, "memory.swap.max": 0}),
                cpu_kv=dict(unique_storage_bytes=audit_module.HOST_BYTES,
                worker_store_pending_events=0, worker_load_pending_events=0,
                worker_unsubmitted_store_jobs=0, worker_load_jobs=0),
                manager=dict(capacity_blocks=8192, pending_store_entries=0,
                             recorded_pending_store_blocks=0),
                pending=dict(scheduler_store_jobs=0, scheduler_load_jobs=0,
                             pending_worker_acknowledgements=0))
    docs = {
        "config.json": config,
        "status.json": dict(status="COMPLETE", capture_status="COMPLETE", requests_completed=64),
        "raw.json": dict(status="COMPLETE", error=None, requests=requests,
                         output_events=outputs, scheduler_steps=steps,
                         engine_steps=engine_steps,
                         memory_trace=[], preemption_events=preemptions,
                         actual_preemption_count=len(preemptions),
                         internal_to_source=mapping, measurement_origin_perf_counter_s=100.0),
        "selective-store.json": dict(status="DRAINED", store_scope="selected",
                                     native_calc_overridden=True,
                                     ltr_config=dict(threshold=30, quantum=10),
                                     active_target=None, pending_plan=None,
                                     schedule_calls=6, applied_rotations=int(action), events=events),
        "offload-events.json": offload,
        "environment.json": dict(vllm="0.26.0", source_sha256=sources,
                                 vllm_source_sha256=runtime_sources),
        "engine_args.json": dict(model=config["model"]["id"],
                                 revision=config["model"]["revision"],
                                 tokenizer_revision=config["model"]["tokenizer_revision"],
                                 max_num_seqs=32, max_num_batched_tokens=1024,
                                 kv_cache_memory_bytes=audit_module.KV_BYTES,
                                 kv_offloading_size=16, enable_prefix_caching=False,
                                 async_scheduling=False),
        "safe-cap-qualification.json": dict(status="QUALIFIED", usable_blocks=4096,
                                            block_size=16, observed_scheduler_reserve_full_isl=True),
        "memory-after-init.json": dict(kv_storage_bytes=audit_module.KV_BYTES),
        "host-after-init.json": host,
        "host-before.json": host,
        "host-request-end.json": host,
        "host-after.json": host,
        "post-request-drain.json": dict(seconds=0.001, calls=0),
        "warmup-cache-reset.json": dict(success=True),
        "resolved-eos.json": dict(status="READ"),
        "resolved-scheduler-config.json": dict(max_num_running_reqs=32),
        "timing.json": dict(measurement_start_perf_s=100.0,
                            measurement_return_perf_s=105.0,
                            post_request_drain_end_perf_s=106.0,
                            process_end_perf_s=107.0),
    }
    for name in audit_module.REQUIRED_JSON:
        (folder / name).write_text(json.dumps(docs.get(name, dict(status="COMPLETE"))))
    (folder / "commands.txt").write_text("synthetic fixture\n")


class LifecycleAuditTests(unittest.TestCase):
    def test_post_output_service_requires_native_return(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            fixture(path)
            file = path / "raw.json"
            data = json.loads(file.read_text())
            data["engine_steps"][4]["output_request_ids"] = []
            file.write_text(json.dumps(data))
            result = audit_module.audit(path)
            self.assertEqual(result["verdict"], "INCOMPLETE")
            self.assertEqual(result["counts"]["observed_chains"], 0)

    def test_load_cannot_finish_before_selected_store_finishes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            fixture(path)
            file = path / "offload-events.json"
            data = json.loads(file.read_text())
            data["completed_jobs"][0]["time_s"] = 102.25
            file.write_text(json.dumps(data))
            result = audit_module.audit(path)
            self.assertEqual(result["verdict"], "INCOMPLETE")
            self.assertEqual(result["counts"]["observed_chains"], 0)

    def test_partial_host_snapshot_only_missing_cgroup_peak_is_usable(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            fixture(path)
            result = audit_module.audit(path)
            self.assertTrue(result["checks"]["physical_resources"], result["issues"])
            self.assertTrue(any("memory.peak" in value for value in result["unknown"]))
            file = path / "host-before.json"
            data = json.loads(file.read_text())
            data["errors"]["cpu_kv"] = "unavailable"
            file.write_text(json.dumps(data))
            result = audit_module.audit(path)
            self.assertEqual(result["verdict"], "INCOMPLETE")
            self.assertTrue(any("host KV snapshot" in value for value in result["issues"]))

    def test_wait_load_charges_only_actual_positive_schedule(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            fixture(path)
            file = path / "selective-store.json"
            data = json.loads(file.read_text())
            for step in (1, 2):
                data["events"].append(dict(event="intent", action="WAIT_LOAD", step=step,
                                           target_id="measured/s0", quantum_remaining=10,
                                           host_perf_counter_s=102 + step * 0.1))
            file.write_text(json.dumps(data))
            result = audit_module.audit(path)
            self.assertEqual(result["verdict"], "OBSERVED_CHAIN", result["issues"])
            self.assertEqual(result["coverage"]["pending_load_zero_quantum"], "OBSERVED")
            self.assertEqual(result["counts"]["wait_load_zero_scheduled"], 1)
            self.assertEqual(result["counts"]["wait_load_positive_scheduled"], 1)

    def test_preflight_source_map_is_not_runner_source_map(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            fixture(path)
            file = path / "environment.json"
            data = json.loads(file.read_text())
            data["vllm_source_sha256"] = json.loads((HERE / "candidate_ltr_r02_env/pkg/runtime_source_hashes.json").read_text())
            file.write_text(json.dumps(data))
            result = audit_module.audit(path)
            self.assertEqual(result["verdict"], "INCOMPLETE")
            self.assertFalse(result["checks"]["runtime_source"])

    def test_deferred_lookup_with_none_does_not_hide_completed_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            fixture(path)
            file = path / "offload-events.json"
            data = json.loads(file.read_text())
            data["lookup"].insert(0, dict(request="measured/s0", matched=None, asynchronous=False))
            file.write_text(json.dumps(data))
            result = audit_module.audit(path)
            self.assertEqual(result["verdict"], "OBSERVED_CHAIN", result["issues"])

    def test_joined_chain(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture(Path(tmp))
            result = audit_module.audit(Path(tmp))
            self.assertEqual(result["verdict"], "OBSERVED_CHAIN", result["issues"])
            self.assertEqual(result["counts"]["observed_chains"], 1)
            self.assertEqual(result["observed_chains"][0]["post_output_positive_steps"], [4])

    def test_no_action_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture(Path(tmp), action=False)
            result = audit_module.audit(Path(tmp))
            self.assertEqual(result["verdict"], "NO_ACTION", result["issues"])

    def test_missing_worker_completion_cannot_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            fixture(path)
            file = path / "offload-events.json"
            data = json.loads(file.read_text())
            data["completed_jobs"] = data["completed_jobs"][:1]
            file.write_text(json.dumps(data))
            result = audit_module.audit(path)
            self.assertEqual(result["verdict"], "INCOMPLETE")
            self.assertTrue(any("load 2 lacks" in issue for issue in result["issues"]))

    def test_missing_output_file_cannot_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            fixture(path)
            (path / "raw.json").unlink()
            result = audit_module.audit(path)
            self.assertEqual(result["verdict"], "INCOMPLETE")
            self.assertTrue(any("missing result file: raw.json" == issue for issue in result["issues"]))

    def test_output_before_load_completion_cannot_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            fixture(path)
            file = path / "raw.json"
            data = json.loads(file.read_text())
            data["requests"][0]["token_times_s"] = [2.15]
            data["output_events"][0]["received_s"] = 2.15
            file.write_text(json.dumps(data))
            result = audit_module.audit(path)
            self.assertEqual(result["verdict"], "INCOMPLETE")
            self.assertEqual(result["counts"]["observed_chains"], 0)

    def test_first_output_without_later_service_cannot_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            fixture(path)
            file = path / "selective-store.json"
            data = json.loads(file.read_text())
            data["events"] = [e for e in data["events"] if not (
                e["event"] == "allocation" and e["step"] == 4)]
            file.write_text(json.dumps(data))
            result = audit_module.audit(path)
            self.assertEqual(result["verdict"], "INCOMPLETE")
            self.assertEqual(result["counts"]["observed_chains"], 0)


if __name__ == "__main__":
    unittest.main()
