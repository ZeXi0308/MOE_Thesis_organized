#!/usr/bin/env python3
"""One locked vLLM server lifecycle and an explicit sequence of serving traces.

Default: CPU/read-only plan, with no server, NVML initialization or output writes.
  python code/run_serving_group.py --config smoke.json --output runs/smoke/group \
      --lock-status runs/smoke/status.json
Execute ONLY as a descendant of the real lock controller:
  python code/run_locked.py --name smoke --timeout 300 -- \
      python code/run_serving_group.py --config smoke.json --output runs/smoke/group \
      --lock-status runs/smoke/status.json --run

Minimal short BF16 smoke JSON (paths relative to this JSON):
  {"schema_version":1, "model":"/local/olmoe", "arm":"bf16",
   "config_id":"olmoe_bf16_eager_smoke", "served_model_name":"olmoe-bf16",
   "max_model_len":512, "max_num_seqs":2, "max_num_batched_tokens":512,
   "gpu_memory_utilization":0.4, "startup_timeout_s":120,
   "client":{"request_timeout_s":20,"drain_s":20,"idle_seconds":3},
   "runs":[{"id":"rate1_r1","trace":"smoke_requests.jsonl",
            "request_rate":1,"repeat_id":1,"seed":42,"timeout_s":45}]}
Use a small real-prompt trace with e.g. 2 requests and max_tokens=8 for a smoke.
No trace is synthesized or shortened here. Add explicit runs for other rates
or repeats; arrival_s traces omit request_rate. Optional warmup_trace is kept
separate from measurement. Smoke results are not a full quality/performance run.

Server defaults: BF16 input, KV auto, eager/no graph or compile, seed 42, TP=1,
prefix caching off, chunked prefill on, generation-config=vllm (no checkpoint
generation defaults). Explicit quantization, attention_backend, moe_backend,
compilation_config, worker_cls/worker_config and extra_vllm_args are supported.
extra_vllm_args must be a JSON list of strings; managed fields cannot be
overridden through it. A dual worker config is passed via C_FORMAT_DUAL_CONFIG.

Server/client process groups are created by this program, tracked using Linux
PID start times, and are the ONLY groups it signals. On failure/SIGTERM it
preserves logs, terminates its clients/server, escalates its own group to KILL,
and waits for group exit and NVML GPU-process release. Cleanup failure is an
explicit failed run, never success; unrelated processes are recorded, not killed.
The lock controller's 30-second TERM-to-KILL interval bounds cleanup. Startup
wall time and NVML residency are separate from bench_serving's steady window.

Every client summary must confirm a complete, known-token population before
another client starts. Missing/failed/timeout/incomplete summaries stop this
server group; later runs are retained as not_run, not silently omitted.
DualFormatWorker enables localhost development collective_rpc for F7 evidence:
summary/take_records/reset before and after each fully drained client. These
calls and their wall times are outside client measurement. Stock workers force
development mode off. collect_dual_statistics=false disables this collection;
worker_rpc_timeout_s defaults to 5. F7 collection rejects client warmup_trace
because its internal warmup cannot be separated by a launcher-side reset. Use
a separate, explicitly excluded client run for any extra load warmup.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import traceback
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from bench_serving import load_trace, verify_lock


CODE = Path(__file__).resolve().parent
STOP = {"signal": None}
DUAL_WORKER = "dual_format_worker.DualFormatWorker"
RPC_METHODS = ("dual_format_summary", "dual_format_take_records", "dual_format_reset_statistics")
MANAGED = {"--model", "--host", "--port", "--served-model-name", "--dtype", "--quantization", "-q",
           "--kv-cache-dtype", "--seed", "--max-model-len", "--gpu-memory-utilization", "--max-num-seqs",
           "--max-num-batched-tokens", "--tensor-parallel-size", "-tp", "--pipeline-parallel-size", "-pp",
           "--data-parallel-size", "-dp", "--enforce-eager", "--no-enforce-eager", "--compilation-config", "-cc", "-O",
           "--enable-prefix-caching", "--no-enable-prefix-caching", "--enable-chunked-prefill", "--no-enable-chunked-prefill",
           "--generation-config", "--override-generation-config", "--attention-backend", "--moe-backend", "--worker-cls",
           "--shutdown-timeout", "--api-key", "--config", "--revision", "--tokenizer", "--tokenizer-revision"}
MANAGED.update({"--distributed-executor-backend", "--data-parallel-backend", "--attention-config", "-ac",
                "--kernel-config", "--optimization-level", "--enable-expert-parallel"})


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    tmp.replace(path)


def positive(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return value


def resolve(root, value):
    if not isinstance(value, str) or not value:
        raise ValueError("path must be a nonempty string")
    return (root / value).resolve()


def build_plan(args):
    source = Path(args.config).resolve()
    output = Path(args.output).resolve()
    config = json.loads(source.read_text())
    allowed = {"schema_version", "model", "checkpoint_manifest", "arm", "config_id", "served_model_name", "dtype",
               "quantization", "kv_cache_dtype", "seed", "max_model_len", "gpu_memory_utilization", "max_num_seqs",
               "max_num_batched_tokens", "enforce_eager", "compilation_config", "enable_prefix_caching", "enable_chunked_prefill",
               "attention_backend", "moe_backend", "worker_cls", "worker_config", "extra_vllm_args", "runs", "client", "port",
               "startup_timeout_s", "shutdown_grace_s", "release_timeout_s", "residency_interval_s",
               "collect_dual_statistics", "worker_rpc_timeout_s"}
    if set(config) - allowed:
        raise ValueError(f"unknown group config fields: {sorted(set(config) - allowed)}")
    if config.get("schema_version") != 1:
        raise ValueError("schema_version=1 required")
    for key in ("arm", "config_id", "served_model_name"):
        if not isinstance(config.get(key), str) or not config[key]:
            raise ValueError(f"{key} must be specified")
    defaults = {"dtype": "bfloat16", "quantization": None, "kv_cache_dtype": "auto", "seed": 42,
                "max_model_len": 4096, "gpu_memory_utilization": .8, "max_num_seqs": 16,
                "max_num_batched_tokens": 4096, "enforce_eager": True,
                "compilation_config": {"mode": 0, "cudagraph_mode": "NONE"},
                "enable_prefix_caching": False, "enable_chunked_prefill": True,
                "attention_backend": None, "moe_backend": "auto", "worker_cls": None, "worker_config": None,
                "extra_vllm_args": [], "port": 8000, "startup_timeout_s": 180,
                "shutdown_grace_s": 8, "release_timeout_s": 10, "residency_interval_s": .5,
                "worker_rpc_timeout_s": 5}
    for key, value in defaults.items():
        config.setdefault(key, value)
    config.setdefault("collect_dual_statistics", config["worker_cls"] == DUAL_WORKER)
    for key in ("enforce_eager", "enable_prefix_caching", "enable_chunked_prefill", "collect_dual_statistics"):
        if type(config[key]) is not bool:
            raise ValueError(f"{key} must be boolean")
    for key in ("max_model_len", "max_num_seqs", "max_num_batched_tokens", "port", "seed"):
        if type(config[key]) is not int or config[key] < (0 if key == "seed" else 1):
            raise ValueError(f"invalid integer {key}")
    if config["port"] > 65535 or not 0 < config["gpu_memory_utilization"] <= 1:
        raise ValueError("invalid port or gpu_memory_utilization")
    if not isinstance(config["compilation_config"], dict):
        raise ValueError("compilation_config must be an object")
    for key in ("startup_timeout_s", "shutdown_grace_s", "release_timeout_s", "residency_interval_s", "worker_rpc_timeout_s"):
        positive(config[key], key)
    if config["collect_dual_statistics"] and (config["worker_cls"] != DUAL_WORKER or not config["worker_config"]):
        raise ValueError("F7 RPC collection requires dual_format_worker.DualFormatWorker and worker_config")
    if config["shutdown_grace_s"] + config["release_timeout_s"] > 18:
        raise ValueError("server cleanup must leave room within run_locked's 30-second TERM-to-KILL period")
    model = resolve(source.parent, config["model"])
    if not (model / "config.json").is_file():
        raise ValueError("model must be a local checkpoint directory; no model download is performed")
    config["model"] = str(model)
    model_config = json.loads((model / "config.json").read_text())
    checkpoint_quant = model_config.get("quantization_config") or {}
    effective_quant = config["quantization"] or checkpoint_quant.get("quant_method")
    expert_worker = None
    if config["worker_cls"] == DUAL_WORKER:
        if not config["worker_config"]:
            raise ValueError("the expert worker requires worker_config")
        from eval_quality import worker_plan
        expert_worker = worker_plan(resolve(source.parent, config["worker_config"]), output,
                                    hash_weights=bool(getattr(args, "run", False)))
        if config["arm"] in ("bf16", "fp8", "awq", "nvfp4") and config["arm"] != expert_worker["format"]:
            raise ValueError("single-format arm label differs from the expert worker precision")
    if config["arm"] == "bf16" and (effective_quant is not None or checkpoint_quant):
        raise ValueError("a BF16 arm must use an unquantized checkpoint and no quantization override")
    if config["arm"] == "fp8" and expert_worker is None:
        groups = list(checkpoint_quant.get("config_groups", {}).values())
        serialized_fp8 = bool(groups) and all(g.get("weights", {}).get("num_bits") == 8 and
            g.get("weights", {}).get("type") == "float" for g in groups)
        if effective_quant != "fp8" and not serialized_fp8:
            raise ValueError("an FP8 arm requires quantization='fp8' or an explicitly FP8 checkpoint")
    extra = config["extra_vllm_args"]
    if not isinstance(extra, list) or any(not isinstance(x, str) or not x or "\0" in x for x in extra):
        raise ValueError("extra_vllm_args must be a JSON list of nonempty argument strings")
    for token in extra:
        option = token.split("=", 1)[0].split(".", 1)[0].replace("_", "-")
        if option in MANAGED or option.startswith(("-O", "--data-parallel-", "--pipeline-parallel-", "--tensor-parallel-")):
            raise ValueError(f"use the structured config field rather than overriding {token}")
    metadata = {"vllm_version_required": "0.26.0", "model_path": str(model),
                "model_config_sha256": sha256(model / "config.json"), "arm": config["arm"],
                "generation_config": "vllm", "tokenizer": "same local checkpoint", "tensor_parallel_size": 1,
                "distributed_executor_backend": "uni", "checkpoint_quantization_config": checkpoint_quant,
                "effective_quantization_method": effective_quant}
    if config.get("checkpoint_manifest"):
        path = resolve(source.parent, config["checkpoint_manifest"])
        manifest = json.loads(path.read_text())
        if Path(manifest["checkpoint_path"]).resolve() != model:
            raise ValueError("checkpoint manifest does not identify the model directory")
        if manifest["identity"]["config_sha256"] != metadata["model_config_sha256"]:
            raise ValueError("model config differs from checkpoint identity")
        metadata["checkpoint_identity"] = manifest["identity"]
        metadata["checkpoint_fingerprint"] = manifest["checkpoint_fingerprint"]
        config["checkpoint_manifest"] = str(path)
    metadata["checkpoint_identity_status"] = "manifest_provided_weights_not_rehashed_here" if "checkpoint_identity" in metadata else "local_path_and_config_only"
    if config["worker_config"]:
        if not config["worker_cls"]:
            raise ValueError("worker_config requires worker_cls")
        if expert_worker is not None:
            worker = expert_worker["effective_config"]
            config["worker_config"] = expert_worker["effective_config_path"]
            metadata.update(expert_precision=expert_worker["format"],
                expert_precision_scope=expert_worker["precision_scope"],
                worker_artifacts=expert_worker["artifacts"],
                sidecar_hashes_verified=expert_worker["sidecar_hashes_verified"],
                worker_source_config_sha256=expert_worker["source_config_sha256"])
        else:
            path = resolve(source.parent, config["worker_config"])
            worker = json.loads(path.read_text())
            config["worker_config"] = str(path)
        # report_path is an output location, not a comparable model option.
        metadata["worker_configuration"] = {k: v for k, v in worker.items() if k != "report_path"}
    argv = [sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", str(model),
            "--host", "127.0.0.1", "--port", str(config["port"]), "--served-model-name", config["served_model_name"],
            "--tensor-parallel-size", "1", "--distributed-executor-backend", "uni",
            "--generation-config", "vllm", "--shutdown-timeout", "5"]
    for key in ("dtype", "kv_cache_dtype", "seed", "max_model_len", "gpu_memory_utilization", "max_num_seqs",
                "max_num_batched_tokens", "quantization", "attention_backend", "moe_backend", "worker_cls"):
        metadata[key] = config[key]
        if config[key] is not None:
            argv += ["--" + key.replace("_", "-"), str(config[key])]
    for key in ("enable_prefix_caching", "enable_chunked_prefill"):
        argv += [("--" if config[key] else "--no-") + key.replace("_", "-")]
        metadata[key] = config[key]
    if config["enforce_eager"]:
        argv.append("--enforce-eager")
    argv += ["--compilation-config", json.dumps(config["compilation_config"], separators=(",", ":")), *extra]
    metadata.update(enforce_eager=config["enforce_eager"], compilation_config=config["compilation_config"], extra_vllm_args=extra,
                    launch_argv=argv, served_model_name=config["served_model_name"],
                    backend_selection="requested backends; actual selection must be read from server.log")
    metadata["worker_statistics_rpc"] = {
        "enabled": config["collect_dual_statistics"], "development_api_loopback_only": True,
        "route": "/collective_rpc" if config["collect_dual_statistics"] else None,
        "methods": list(RPC_METHODS) if config["collect_dual_statistics"] else [],
        "timeout_s": config["worker_rpc_timeout_s"],
        "scope": "before/after each complete client, excluded from client energy/latency; bounded records may omit earlier calls",
        "serving_integration_validation": "not established by this launcher or CPU checks"}
    client = {"request_timeout_s": 180, "drain_s": 180, "idle_seconds": 3, **config.get("client", {})}
    if set(client) != {"request_timeout_s", "drain_s", "idle_seconds"}:
        raise ValueError("unknown client configuration")
    for key, value in client.items():
        positive(value, key)
    endpoint = f"http://127.0.0.1:{config['port']}"
    planned, ids = [], set()
    if not isinstance(config.get("runs"), list) or not config["runs"]:
        raise ValueError("runs must explicitly list each trace/rate/repetition")
    for run in config["runs"]:
        allowed_run = {"id", "trace", "request_rate", "repeat_id", "seed", "arrival_window_s", "warmup_trace", "timeout_s"}
        if not isinstance(run, dict) or set(run) - allowed_run:
            raise ValueError("unknown run fields")
        rid = run.get("id")
        if not isinstance(rid, str) or not rid or rid in ids or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in rid):
            raise ValueError("run id must be distinct and contain only letters/digits/_/-")
        ids.add(rid)
        if "repeat_id" not in run:
            raise ValueError("each run must explicitly identify its repeat")
        trace = resolve(source.parent, run["trace"])
        seed = run.get("seed", config["seed"])
        if type(seed) is not int:
            raise ValueError("schedule seed must be an integer")
        rate = run.get("request_rate")
        rows = load_trace(trace, rate, seed)
        window = run.get("arrival_window_s", max(rows[-1]["arrival_s"], 1 / rate if rate else .001))
        if not math.isfinite(window) or window < rows[-1]["arrival_s"]:
            raise ValueError("arrival window must cover all trace arrivals")
        cmd = [sys.executable, str(CODE / "bench_serving.py"), "--trace", str(trace), "--endpoint", endpoint + "/v1/completions",
               "--model", config["served_model_name"], "--arm", config["arm"], "--config-id", config["config_id"],
               "--repeat-id", str(run["repeat_id"]), "--server-metadata", str(output / "server_metadata.json"),
               "--lock-status", str(Path(args.lock_status).resolve()), "--output", str(output / rid),
               "--seed", str(seed), "--arrival-window-s", str(window)]
        for key, value in client.items():
            cmd += ["--" + key.replace("_", "-"), str(value)]
        if rate is not None:
            cmd += ["--request-rate", str(rate)]
        warmup_bound = 0
        warmup = None
        if run.get("warmup_trace"):
            if config["collect_dual_statistics"]:
                raise ValueError("F7 RPC collection cannot isolate a client-internal warmup_trace; use a separate excluded client run")
            warmup = resolve(source.parent, run["warmup_trace"])
            warmup_bound = len(load_trace(warmup, 1, seed)) * client["request_timeout_s"]
            cmd += ["--warmup-trace", str(warmup)]
        timeout = positive(run.get("timeout_s", window + client["drain_s"] + client["idle_seconds"] + warmup_bound + 15), "run timeout")
        planned.append({"id": rid, "argv": cmd, "trace": str(trace), "trace_sha256": sha256(trace),
                        "requests": len(rows), "request_rate": rate, "seed": seed, "repeat_id": run["repeat_id"],
                        "arrival_window_s": window, "warmup_trace": str(warmup) if warmup else None,
                        "warmup_trace_sha256": sha256(warmup) if warmup else None, "timeout_s": timeout})
    return {"schema_version": 1, "mode": "plan", "config_path": str(source), "config_sha256": sha256(source),
            "config": config, "server_metadata": metadata, "server_argv": argv, "endpoint": endpoint,
            "expert_worker": expert_worker,
            "runs": planned, "output": str(output), "lock_status": str(Path(args.lock_status).resolve()),
            "worst_case_seconds_before_cleanup": config["startup_timeout_s"] + sum(r["timeout_s"] for r in planned)
                + (2 * len(RPC_METHODS) * config["worker_rpc_timeout_s"] * len(planned) if config["collect_dual_statistics"] else 0),
            "cleanup_reserve_s": config["shutdown_grace_s"] + config["release_timeout_s"] + 8}


def client_population_check(summary_path, planned_requests):
    """Fail closed on cancellation/unknown population; retain the raw summary."""
    path = Path(summary_path)
    result = {"summary_path": str(path), "safe_to_reuse_server": False, "reasons": []}
    try:
        summary = json.loads(path.read_text())
        result["summary_sha256"] = sha256(path)
        population, metadata = summary["population"], summary["metadata"]
        result.update(population=population, client_state=metadata.get("state"), aborted=metadata.get("aborted"))
        expected = {"planned": planned_requests, "arrived": planned_requests, "sent": planned_requests,
                    "completed": planned_requests, "noncompleted": 0, "unknown_output_token_count": 0}
        for key, value in expected.items():
            if type(population.get(key)) is not int or population[key] != value:
                result["reasons"].append(f"population.{key}={population.get(key)!r}, expected {value}")
        counts = population.get("status_counts")
        if (not isinstance(counts, dict) or any(type(v) is not int or v < 0 for v in counts.values()) or
                counts.get("completed") != planned_requests or
                any(value != 0 for status, value in counts.items() if status != "completed")):
            result["reasons"].append("status_counts does not describe only completed requests")
        if metadata.get("state") != "COMPLETE" or metadata.get("aborted") is not False:
            result["reasons"].append("client metadata is not COMPLETE with aborted=false")
        if planned_requests <= 0:
            result["reasons"].append("empty planned population")
        result["safe_to_reuse_server"] = not result["reasons"]
    except Exception as error:
        result["reasons"].append(f"cannot validate terminal summary: {type(error).__name__}: {error}")
    result["scope"] = "all planned client requests completed with known output counts; cancellation is never treated as GPU drain"
    return result


def collect_worker_statistics(endpoint, output_path, boundary, timeout_s, deadline, tick):
    """Use the vLLM 0.26 localhost development API, outside a client interval.

    Empty kwargs deliberately preserve take_records(synchronize=False); the
    development API's serialized-string argument contract must not turn the
    string 'false' into a truthy Python value. No arbitrary RPC is configurable.
    """
    if not endpoint.startswith("http://127.0.0.1:") or "/" in endpoint.removeprefix("http://127.0.0.1:"):
        raise ValueError("worker statistics RPC requires the owned loopback server endpoint")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = {"schema_version": 1, "boundary": boundary, "status": "running", "calls": [],
              "started_monotonic_s": time.monotonic(), "started_unix_s": time.time(),
              "outside_client_measurement": True, "previous_population_complete": True,
              "records_synchronize": False, "gpu_serving_validation_claim": False}
    write_json(output_path, report)
    try:
        for method in RPC_METHODS:
            check_continue(deadline)
            tick()
            timeout = min(timeout_s, deadline - time.monotonic())
            if timeout <= 0:
                raise TimeoutError("no time remains for worker RPC")
            payload = {"method": method, "args": [], "kwargs": {}, "timeout": timeout}
            call = {"method": method, "request": payload, "started_monotonic_s": time.monotonic(), "started_unix_s": time.time()}
            report["calls"].append(call)
            try:
                request = Request(endpoint + "/collective_rpc", data=json.dumps(payload).encode(),
                                  headers={"Content-Type": "application/json"}, method="POST")
                with urlopen(request, timeout=timeout) as response:
                    call["http_status"] = response.status
                    body = response.read().decode("utf-8")
                # Keep undecodable/error responses as evidence too.
                call["response_text"] = body
                parsed = json.loads(body)
                call["response"] = parsed
                del call["response_text"]
                results = parsed.get("results")
                if call["http_status"] != 200 or not isinstance(results, list) or len(results) != 1 or not isinstance(results[0], dict):
                    raise ValueError("expected one dictionary result from the single-rank dual worker")
                call["status"] = "complete"
            except HTTPError as error:
                call.update(status="failed", http_status=error.code, response_text=error.read().decode("utf-8", errors="replace"))
                raise
            except BaseException as error:
                call.update(status="failed", error=f"{type(error).__name__}: {error}")
                raise
            finally:
                call.update(ended_monotonic_s=time.monotonic(), ended_unix_s=time.time(),
                            wall_s=time.monotonic() - call["started_monotonic_s"])
                write_json(output_path, report)
        report["status"] = "complete"
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report.update(ended_monotonic_s=time.monotonic(), ended_unix_s=time.time(),
                      wall_s=time.monotonic() - report["started_monotonic_s"])
        write_json(output_path, report)
    return {"path": str(output_path), "sha256": sha256(output_path), "status": report["status"],
            "wall_s": report["wall_s"], "started_monotonic_s": report["started_monotonic_s"],
            "ended_monotonic_s": report["ended_monotonic_s"], "outside_client_measurement": True}


def proc_stat(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return {"pid": pid, "state": fields[0], "ppid": int(fields[1]), "pgid": int(fields[2]),
                "sid": int(fields[3]), "start_ticks": int(fields[19])}
    except (FileNotFoundError, ProcessLookupError):
        return None


class OwnedGroup:
    """Only a new-session child created here may be signalled through this class."""
    def __init__(self, argv, log, env):
        self.spawn_after_ticks = int(time.clock_gettime(time.CLOCK_BOOTTIME) * os.sysconf("SC_CLK_TCK"))
        self.process = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        self.pid = self.process.pid
        self.birth = proc_stat(self.pid)
        self.gone = False
        self.seen = {}
        if self.birth is not None and (self.birth["pgid"], self.birth["sid"]) != (self.pid, self.pid):
            raise RuntimeError("child did not enter its own process group/session")

    def members(self):
        if self.gone:
            return []
        found = []
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            stat = proc_stat(int(entry.name))
            if stat and stat["pgid"] == self.pid and stat["sid"] == self.pid:
                if stat["start_ticks"] < self.spawn_after_ticks:
                    raise RuntimeError("cannot verify owned process group birth")
                self.seen[stat["pid"]] = stat["start_ticks"]
                if stat["state"] != "Z":
                    found.append(stat)
        if not found and self.process.poll() is not None:
            self.gone = True  # Never signal a later group that might reuse this ID.
        return found

    def send(self, sig):
        members = self.members()
        if members:
            if self.pid == os.getpgrp():
                raise RuntimeError("refusing to signal the controller process group")
            try:
                os.killpg(self.pid, sig)
            except ProcessLookupError:
                pass
        return members


class Residency:
    def __init__(self, uuid, output, interval):
        import pynvml as nvml
        self.n = nvml
        nvml.nvmlInit()
        self.handle = nvml.nvmlDeviceGetHandleByUUID(uuid)
        self.log = (output / "gpu_residency.jsonl").open("w")
        self.interval, self.last, self.last_stamp = interval, None, 0.
        self.peak = self.process_peak = 0
        self.byte_seconds = {}

    def sample(self, phase, force=False):
        now = time.monotonic()
        if not force and now - self.last_stamp < self.interval:
            return self.last
        memory = self.n.nvmlDeviceGetMemoryInfo(self.handle)
        processes = {}
        for proc in self.n.nvmlDeviceGetComputeRunningProcesses(self.handle) + self.n.nvmlDeviceGetGraphicsRunningProcesses(self.handle):
            used = getattr(proc, "usedGpuMemory", None)
            if not isinstance(used, int) or used < 0 or used > memory.total:
                used = None
            processes[proc.pid] = {"pid": proc.pid, "used_gpu_memory_bytes": used, "proc": proc_stat(proc.pid)}
        row = {"monotonic_s": now, "unix_s": time.time(), "phase": phase,
               "used_bytes": int(memory.used), "free_bytes": int(memory.free), "total_bytes": int(memory.total),
               "processes": list(processes.values())}
        if self.last:
            dt = now - self.last_stamp
            key = self.last["phase"]
            self.byte_seconds[key] = self.byte_seconds.get(key, 0.) + dt * (self.last["used_bytes"] + row["used_bytes"]) / 2
        self.last, self.last_stamp = row, now
        self.peak = max(self.peak, row["used_bytes"])
        self.process_peak = max(self.process_peak, sum(p["used_gpu_memory_bytes"] or 0 for p in processes.values()))
        self.log.write(json.dumps(row) + "\n")
        self.log.flush()
        return row

    def close(self):
        self.log.close()
        self.n.nvmlShutdown()


def stop_group(group, grace, tick):
    if group is None:
        return {"started": False, "released": True}
    monitoring_errors = []
    def sample_safely():
        try:
            tick()
        except Exception as error:
            monitoring_errors.append(f"{type(error).__name__}: {error}")
    initial = group.send(signal.SIGTERM)
    end = time.monotonic() + grace
    while group.members() and time.monotonic() < end:
        sample_safely()
        time.sleep(.1)
    killed = bool(group.members())
    if killed:
        group.send(signal.SIGKILL)
    end = time.monotonic() + 2
    while group.members() and time.monotonic() < end:
        sample_safely()
        time.sleep(.1)
    group.process.poll()
    remaining = group.members()
    return {"started": True, "pid": group.pid, "initial_members": initial, "sent_kill": killed,
            "exit_code": group.process.returncode, "released": not remaining, "remaining": remaining,
            "observed_pid_start_ticks": group.seen, "monitoring_errors": monitoring_errors}


def check_continue(deadline):
    if STOP["signal"] is not None:
        raise InterruptedError(f"received signal {STOP['signal']}")
    if time.monotonic() >= deadline:
        raise TimeoutError("group deadline reached before lock-controller timeout")


def execute(plan):
    lock = verify_lock(plan["lock_status"])
    if importlib.metadata.version("vllm") != "0.26.0":
        raise RuntimeError("vLLM 0.26.0 is required")
    controller = json.loads(Path(plan["lock_status"]).read_text())
    remaining = controller["started_utc_s"] + controller["timeout_s"] - time.time()
    reserve = plan["cleanup_reserve_s"]
    if remaining <= reserve or plan["worst_case_seconds_before_cleanup"] > remaining - reserve:
        raise ValueError("run_locked timeout must cover configured startup + all client deadlines + cleanup reserve")
    deadline = time.monotonic() + remaining - reserve
    output = Path(plan["output"])
    if not output.is_relative_to(Path(plan["lock_status"]).parent):
        raise ValueError("group output must be inside the current run_locked run directory for archival")
    output.mkdir(parents=True, exist_ok=False)
    if plan.get("expert_worker") is not None:
        write_json(Path(plan["expert_worker"]["effective_config_path"]),
                   plan["expert_worker"]["effective_config"])
    write_json(output / "plan.json", plan)
    record = {"status": "preparing", "started_unix_s": time.time(), "lock": lock, "runs": [],
              "startup_excluded_from_steady_metrics": True, "server_log": str(output / "server.log")}
    write_json(output / "group.json", record)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = lock["gpu_uuid"]
    env["PYTHONPATH"] = str(CODE) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    config = plan["config"]
    # Never inherit development mode into stock/reference server arms.
    env["VLLM_SERVER_DEV_MODE"] = "1" if config["collect_dual_statistics"] else "0"
    if config["worker_config"]:
        env["C_FORMAT_DUAL_CONFIG"] = config["worker_config"]
    else:
        env.pop("C_FORMAT_DUAL_CONFIG", None)
    metadata = dict(plan["server_metadata"])
    metadata["versions"] = {}
    for package in ("vllm", "torch", "triton", "transformers", "compressed-tensors"):
        metadata["versions"][package] = importlib.metadata.version(package)
    if config["collect_dual_statistics"]:
        vllm_root = Path(importlib.metadata.distribution("vllm").locate_file("vllm"))
        metadata["worker_rpc_interface_source_sha256"] = {name: sha256(vllm_root / name) for name in (
            "entrypoints/serve/dev/rpc/api_router.py", "entrypoints/openai/api_server.py")}
    metadata["environment"] = {key: value for key, value in env.items()
        if key in ("CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH", "HF_ENDPOINT", "HF_HUB_DISABLE_XET") or
        (key.startswith("VLLM_") and not any(secret in key for secret in ("TOKEN", "SECRET", "KEY")))}
    write_json(output / "server_metadata.json", metadata)
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda number, frame: STOP.update(signal=number))
    server = client = meter = None
    server_log = None
    launched = None
    cleanup_errors = []
    try:
        # Refuse an occupied loopback port before creating any service process.
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", config["port"]))
        meter = Residency(lock["gpu_uuid"], output, config["residency_interval_s"])
        baseline = meter.sample("before_server", force=True)
        record["gpu_before_server"] = baseline
        if baseline["processes"]:
            raise RuntimeError("GPU became occupied after lock acquisition; refusing to start a second server")
        server_log = (output / "server.log").open("w")
        launched = time.monotonic()
        record.update(status="starting_server", server_start_monotonic_s=launched, server_start_unix_s=time.time())
        server = OwnedGroup(plan["server_argv"], server_log, env)
        meter.sample("startup", force=True)
        record["server_pid"] = server.pid
        write_json(output / "group.json", record)
        ready_deadline = min(deadline, launched + config["startup_timeout_s"])
        last_error = None
        while True:
            check_continue(ready_deadline)
            meter.sample("startup")
            server.members()
            if server.process.poll() is not None:
                raise RuntimeError(f"server exited before readiness: {server.process.returncode}")
            try:
                with urlopen(plan["endpoint"] + "/health", timeout=1) as response:
                    if response.status != 200:
                        raise RuntimeError(f"health status {response.status}")
                with urlopen(plan["endpoint"] + "/v1/models", timeout=1) as response:
                    models = json.load(response)
                if config["served_model_name"] not in [m["id"] for m in models["data"]]:
                    raise RuntimeError("ready server does not expose the configured served model")
                record["models_response"] = models
                break
            except (URLError, HTTPError, TimeoutError, ConnectionError, OSError) as error:
                last_error = str(error)
                record["last_health_error"] = last_error
                time.sleep(.2)
        ready = time.monotonic()
        record.update(status="running_clients", ready_monotonic_s=ready, startup_wall_s=ready - launched,
                      gpu_at_ready=meter.sample("ready", force=True))
        write_json(output / "group.json", record)
        for run in plan["runs"]:
            check_continue(deadline)
            verify_lock(plan["lock_status"])
            # Detect changed inputs between read-only planning and execution.
            if sha256(run["trace"]) != run["trace_sha256"] or (run["warmup_trace"] and sha256(run["warmup_trace"]) != run["warmup_trace_sha256"]):
                raise ValueError("trace changed since group plan")
            item = {"id": run["id"], "status": "running", "started_unix_s": time.time(),
                    "start_monotonic_s": time.monotonic(), "argv": run["argv"]}
            record["runs"].append(item)
            write_json(output / "group.json", record)
            if config["collect_dual_statistics"]:
                before_path = output / "f7" / (run["id"] + ".before.json")
                item["worker_statistics"] = {"before_path": str(before_path),
                    "scope": "launcher boundaries outside client measurement; no client-internal warmup"}
                item["worker_statistics"]["before"] = collect_worker_statistics(
                    plan["endpoint"], before_path, "before_client", config["worker_rpc_timeout_s"], deadline,
                    lambda: meter.sample("worker_rpc_before:" + run["id"], force=True))
                write_json(output / "group.json", record)
            with (output / (run["id"] + ".client.log")).open("w") as log:
                meter.sample("client:" + run["id"], force=True)
                client = OwnedGroup([*run["argv"], "--nvml-device", lock["gpu_uuid"]], log, env)
                item.update(client_pid=client.pid, client_launched_monotonic_s=time.monotonic())
                client_deadline = min(deadline, time.monotonic() + run["timeout_s"])
                while client.process.poll() is None:
                    check_continue(client_deadline)
                    meter.sample("client:" + run["id"])
                    server.members()
                    if server.process.poll() is not None:
                        raise RuntimeError("server exited while benchmark client was running")
                    time.sleep(.1)
                item.update(status="complete" if client.process.returncode == 0 else "failed", exit_code=client.process.returncode,
                            ended_unix_s=time.time(), client_terminal_monotonic_s=time.monotonic(),
                            client_process_wall_s=time.monotonic() - item["client_launched_monotonic_s"],
                            wall_s=time.monotonic() - item["start_monotonic_s"])
                residue = stop_group(client, 1, lambda: meter.sample("client_cleanup"))
                item["client_cleanup"] = residue
                client = None
            item["population_check"] = client_population_check(output / run["id"] / "summary.json", run["requests"])
            if not item["population_check"]["safe_to_reuse_server"]:
                item["status"] = "failed_population"
            write_json(output / "group.json", record)
            if item["status"] != "complete" or not residue["released"]:
                if config["collect_dual_statistics"]:
                    item["worker_statistics"]["after"] = {"status": "not_run",
                        "reason": "client/process population did not fully drain; no RPC reset or subsequent load is allowed"}
                raise RuntimeError(f"benchmark {run['id']} failed; preserving trace/results and stopping this group")
            if config["collect_dual_statistics"]:
                after_path = output / "f7" / (run["id"] + ".after.json")
                item["worker_statistics"]["after_path"] = str(after_path)
                item["worker_statistics"]["after"] = collect_worker_statistics(
                    plan["endpoint"], after_path, "after_complete_client", config["worker_rpc_timeout_s"], deadline,
                    lambda: meter.sample("worker_rpc_after:" + run["id"], force=True))
                write_json(output / "group.json", record)
        record["status"] = "clients_complete"
    except BaseException as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        if launched is not None and "startup_wall_s" not in record:
            record["failed_startup_wall_s"] = time.monotonic() - launched
    finally:
        tick = lambda: meter.sample("shutdown") if meter else None
        if client:
            try:
                record["active_client_cleanup"] = stop_group(client, 2, tick)
            except Exception as error:
                cleanup_errors.append(f"client cleanup: {error}")
        # Abrupt client timeout/signal may write its terminal summary while the
        # owned process group is being stopped. Read it now, before server exit.
        run_by_id = {run["id"]: run for run in plan["runs"]}
        for item in record["runs"]:
            if "client_pid" in item and "population_check" not in item:
                run = run_by_id[item["id"]]
                item["population_check"] = client_population_check(output / run["id"] / "summary.json", run["requests"])
            if "worker_statistics" in item and "after" not in item["worker_statistics"]:
                item["worker_statistics"]["after"] = {"status": "not_run_or_failed",
                    "reason": "group interrupted before a successful after-client RPC boundary; preserve any partial after_path report"}
            statistics = item.get("worker_statistics", {})
            for phase in ("before", "after"):
                path = statistics.get(phase + "_path")
                if path and Path(path).is_file():
                    try:
                        receipt = json.loads(Path(path).read_text())
                        statistics[phase] = {"path": path, "sha256": sha256(path), "status": receipt["status"],
                            "wall_s": receipt["wall_s"], "outside_client_measurement": True,
                            "started_monotonic_s": receipt["started_monotonic_s"],
                            "ended_monotonic_s": receipt["ended_monotonic_s"]}
                    except Exception as error:
                        statistics[phase + "_receipt_error"] = f"{type(error).__name__}: {error}"
        try:
            record["server_cleanup"] = stop_group(server, config["shutdown_grace_s"], tick)
            if not record["server_cleanup"]["released"]:
                cleanup_errors.append("server process group still has live members")
        except Exception as error:
            cleanup_errors.append(f"server cleanup: {error}")
        if meter:
            try:
                end = time.monotonic() + config["release_timeout_s"]
                sample = meter.sample("gpu_release", force=True)
                while sample["processes"] and time.monotonic() < end:
                    time.sleep(.2)
                    sample = meter.sample("gpu_release", force=True)
                record["gpu_after_cleanup"] = sample
                record["gpu_process_release_verified"] = not sample["processes"]
                if sample["processes"]:
                    cleanup_errors.append("GPU processes remain after cleanup; no unrelated process was signalled")
                record["gpu_residency"] = {"board_peak_used_bytes": meter.peak, "all_gpu_processes_peak_bytes": meter.process_peak,
                    "board_used_byte_seconds_by_phase": meter.byte_seconds,
                    "scope": "sampled NVML board/process residency; startup and serving recorded separately; not allocated KV bytes"}
            except Exception as error:
                cleanup_errors.append(f"GPU release verification: {error}")
            finally:
                try:
                    meter.close()
                except Exception as error:
                    cleanup_errors.append(f"NVML/log close: {error}")
        if server_log:
            server_log.close()
        record.update(cleanup_errors=cleanup_errors, ended_unix_s=time.time(), received_signal=STOP["signal"])
        record["nonmeasurement_worker_rpc_wall_s"] = sum(
            item.get("worker_statistics", {}).get(phase, {}).get("wall_s", 0.)
            for item in record["runs"] for phase in ("before", "after"))
        if launched is not None:
            record["server_lifecycle_wall_s"] = time.monotonic() - launched
        if cleanup_errors:
            record["status"] = "failed_cleanup"
        elif record["status"] == "clients_complete":
            record["status"] = "complete"
        for item in record["runs"]:
            if item["status"] == "running":
                item.update(status="interrupted_or_failed" if "client_pid" in item else "not_run",
                            ended_unix_s=time.time(), reason=record.get("error", "group interrupted before client terminal state"))
        attempted = {item["id"] for item in record["runs"]}
        for run in plan["runs"]:
            if run["id"] not in attempted:
                record["runs"].append({"id": run["id"], "status": "not_run", "repeat_id": run["repeat_id"],
                    "request_rate": run["request_rate"], "trace": run["trace"], "requests": run["requests"],
                    "reason": record.get("error", "server group stopped before this client could start")})
        write_json(output / "group.json", record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--lock-status", required=True)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    plan = build_plan(args)
    if not args.run:
        print(json.dumps(plan, indent=2, ensure_ascii=False, allow_nan=False))
        return
    record = execute(plan)
    print(json.dumps({"status": record["status"], "output": plan["output"], "cleanup_errors": record["cleanup_errors"]}))
    raise SystemExit(0 if record["status"] == "complete" else 1)


if __name__ == "__main__":
    main()
