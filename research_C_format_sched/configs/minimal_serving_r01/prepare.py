#!/usr/bin/env python3
"""Freeze the minimal three-arm serving inputs. CPU only; never starts a server.

Run from any directory. Existing measured output is never touched. The inputs
are derived from the already frozen OLMoE serving pool, with unchanged prompts.
"""
import hashlib
import json
from pathlib import Path
import random
import shlex
import sys


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "code"))
from bench_serving import digest, load_trace


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True,
                                    allow_nan=False) + "\n" for row in rows))


def main():
    source = ROOT / "datasets/serving_olmoe_r01/serving_natural.jsonl"
    source_manifest = source.parent / "manifest.json"
    original = json.loads(source_manifest.read_text())
    expected = original["output_files"][source.name]["sha256"]
    if sha256(source) != expected:
        raise ValueError("source trace SHA differs from its frozen manifest")
    population = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    rng = random.Random(42)
    selected = []
    for source_class in ("sharegpt", "longbench"):
        pool = sorted((r for r in population if r["source"] == source_class),
                      key=lambda r: r["request_id"])
        if len(pool) != 128:
            raise ValueError(f"expected 128 source records for {source_class}")
        selected.extend(rng.sample(pool, 8))
    rng.shuffle(selected)
    if len({row["request_id"] for row in selected}) != 16:
        raise ValueError("duplicate selected request ID")
    rows = []
    for item in selected:
        if hashlib.sha256(item["prompt"].encode()).hexdigest() != item["prompt_sha256"]:
            raise ValueError("source prompt hash mismatch")
        if item["prompt_tokens"] + 32 > 4096:
            raise ValueError("prompt plus cap exceeds the shared model limit")
        row = dict(item)
        row["max_tokens"] = 32
        row["sampling"] = {"temperature": 0.0, "ignore_eos": False, "min_tokens": 0, "seed": 42}
        rows.append(row)
    warmup, seen = [], set()
    for row in rows:
        if row["source"] not in seen:
            warmup.append(row)
            seen.add(row["source"])
    measurement_trace = HERE / "requests_natural_16.jsonl"
    warmup_trace = HERE / "warmup_natural_2.jsonl"
    write_rows(measurement_trace, rows)
    write_rows(warmup_trace, warmup)
    scheduled = load_trace(measurement_trace, 2.0, 42)
    warmup_scheduled = load_trace(warmup_trace, 2.0, 42)
    if len(scheduled) != 16 or len(warmup_scheduled) != 2:
        raise ValueError("bench_serving did not load the expected population")
    runs = [
        {"id": "warmup_excluded", "trace": warmup_trace.name, "repeat_id": "warmup_excluded",
         "request_rate": 2.0, "seed": 42, "timeout_s": 35},
        *[{"id": f"measure_r{i:02d}", "trace": measurement_trace.name, "repeat_id": i,
           "request_rate": 2.0, "seed": 42, "timeout_s": 45} for i in (1, 2)],
    ]
    configs = {}
    for name, arm in (("single_fp8", "fp8"), ("single_awq", "awq"), ("dual_phase", "dual_fp8_awq")):
        config = {
            "schema_version": 1,
            "model": "../../checkpoints/olmoe_awq_integration_bf16_r01",
            "checkpoint_manifest": "../../checkpoints/olmoe_awq_integration_bf16_r01/checkpoint_manifest.json",
            "arm": arm, "config_id": f"minimal_serving_r01_{name}",
            "served_model_name": "olmoe-integration",
            "dtype": "bfloat16", "quantization": None, "kv_cache_dtype": "auto", "seed": 42,
            "max_model_len": 4096, "gpu_memory_utilization": 0.8,
            "max_num_seqs": 16, "max_num_batched_tokens": 4096,
            "enforce_eager": True, "compilation_config": {"mode": 0, "cudagraph_mode": "NONE"},
            "enable_prefix_caching": False, "enable_chunked_prefill": True,
            "attention_backend": None, "moe_backend": "auto",
            "worker_cls": "dual_format_worker.DualFormatWorker",
            "worker_config": f"../olmoe_integration_r01/{name}.json",
            "extra_vllm_args": [], "port": 8000, "startup_timeout_s": 120,
            "shutdown_grace_s": 8, "release_timeout_s": 10, "residency_interval_s": 0.5,
            "collect_dual_statistics": True, "worker_rpc_timeout_s": 2,
            "client": {"request_timeout_s": 25, "drain_s": 25, "idle_seconds": 3},
            "runs": runs,
        }
        target = HERE / f"{name}.json"
        write_json(target, config)
        configs[target.name] = {"sha256": sha256(target), "arm": arm,
                                "worker_config": config["worker_config"]}
    schedule = {
        "request_rate_per_s": 2.0, "arrival_seed": 42, "generated_by": "bench_serving.load_trace",
        "measurement_schedule_sha256": digest(scheduled),
        "measurement_arrivals": [{"request_id": r["request_id"], "arrival_s": r["arrival_s"]} for r in scheduled],
        "warmup_schedule_sha256": digest(warmup_scheduled),
        "warmup_arrivals": [{"request_id": r["request_id"], "arrival_s": r["arrival_s"]} for r in warmup_scheduled],
    }
    write_json(HERE / "schedule.json", schedule)
    commands = {}
    python = "/root/miniconda3/bin/python"
    for name in ("single_fp8", "single_awq", "dual_phase"):
        run_name = f"minimal_serving_r01_{name}"
        group = [python, "code/run_serving_group.py", "--config", f"configs/minimal_serving_r01/{name}.json",
                 "--output", f"runs/{run_name}/group", "--lock-status", f"runs/{run_name}/status.json"]
        execute = [python, "code/run_locked.py", "--name", run_name, "--timeout", "360", "--", *group, "--run"]
        commands[name] = {"plan_argv": group, "run_argv": execute,
                          "plan_shell": shlex.join(group), "run_shell": shlex.join(execute)}
    write_json(HERE / "commands.json", {
        "cwd_remote": "/root/autodl-tmp/research_C_format_sched",
        "environment": "use the existing main vLLM 0.26.0 environment and CUDA13 LD_LIBRARY_PATH unchanged",
        "execution": "one arm at a time; no automatic retry; backup each complete or failed runs directory locally and verify its SHA256.json before the next group",
        "precondition": "real integration checkpoint and all32 sidecars plus three worker configs must exist; plan commands are CPU-only and require these actual assets",
        "arms": commands,
    })
    dependency_paths = ["checkpoints/olmoe_awq_integration_bf16_r01/config.json",
                        "checkpoints/olmoe_awq_integration_bf16_r01/checkpoint_manifest.json",
                        *[f"configs/olmoe_integration_r01/{name}.json" for name in ("single_fp8", "single_awq", "dual_phase")]]
    manifest = {
        "schema_version": 1, "scope": "minimal actual serving integration smoke, not an SLO sweep or final performance conclusion",
        "state": "INPUTS_FROZEN_NOT_RUN", "cpu_only_preparation": True,
        "source_trace": str(source.relative_to(ROOT)), "source_trace_sha256": expected,
        "source_manifest_sha256": sha256(source_manifest), "prepare_script_sha256": sha256(Path(__file__)),
        "selection": {"seed": 42, "algorithm": "sort each class by request_id; Random(42).sample 8 ShareGPT then 8 LongBench without replacement; shuffle the combined 16 using the same RNG",
                      "source_classes": {"sharegpt": 8, "longbench": 8}, "prompt_text_changed": False,
                      "source_cap": 128, "new_cap": 32, "truncation": False,
                      "source_ordered_request_ids": [r["request_id"] for r in rows]},
        "source_tokenizer": {"tokenizer_path_at_preparation": original["tokenizer_path"],
                             "class": original["tokenizer_class"], "add_special_tokens": original["add_special_tokens"],
                             "files": {k: v for k, v in original["output_files"].items() if k.startswith("tokenizer/")}},
        "prompt_tokens": {"total": sum(r["prompt_tokens"] for r in rows),
                          "min": min(r["prompt_tokens"] for r in rows), "max": max(r["prompt_tokens"] for r in rows),
                          "by_source": {s: sum(r["prompt_tokens"] for r in rows if r["source"] == s) for s in ("sharegpt", "longbench")},
                          "scope": "frozen source-tokenizer counts; retain actual server prompt usage as well"},
        "natural_generation": rows[0]["sampling"], "max_tokens": 32,
        "measurement_trace": {"path": measurement_trace.name, "sha256": sha256(measurement_trace), "rows": 16,
                              "schedule_sha256": digest(scheduled), "last_arrival_s": scheduled[-1]["arrival_s"]},
        "warmup": {"trace": warmup_trace.name, "sha256": sha256(warmup_trace), "rows": 2,
                   "request_ids": [r["request_id"] for r in warmup],
                   "rule": "first occurrence of each source class in the fixed measurement order",
                   "excluded_from_main_measurements": True, "retained_run_id": "warmup_excluded",
                   "scope": "two real requests exercise prefill/decode before measured clients; does not establish complete shape warmup or continuous-power steady state"},
        "measured_run_ids": ["measure_r01", "measure_r02"], "measured_repetitions_per_arm": 2,
        "all_arms_identical_prompt_order_and_arrivals": True, "group_configs": configs,
        "commands": {"path": "commands.json", "sha256": sha256(HERE / "commands.json")},
        "precision_scope": "expert-only FP8 or calibrated AWQ sidecars; common BF16 host attention/router and verified common activation basis",
        "phase_rule": "real scheduler pure-prefill FP8, pure-decode AWQ; mixed threshold32 is an untuned mechanism setting; actual branch coverage must be checked from F7",
        "budget": {"outer_timeout_s_per_arm": 360, "three_arm_timeout_sum_s": 1080,
                   "per_arm_precleanup_bound_s": 281, "per_arm_cleanup_reserve_s": 26,
                   "formula": "120 startup + (35 + 45 + 45) clients + 2 boundaries * 3 RPC methods * 2 seconds * 3 clients = 281; cleanup reserve 8 + 10 + 8 = 26",
                   "outer_TERM_to_KILL_max_extra_s_per_arm": 30, "three_arm_timeout_plus_KILL_sum_s": 1170,
                   "scope": "execution ceilings exclude lock waiting, CPU preflight/archive/hash/transfer; no automatic retry"},
        "measurement_accounting": {
            "startup_and_residency": "group.json and gpu residency samples; startup and warmup excluded from per-client steady serving summaries",
            "client_summaries": "TTFT/TPOT P50/P95, successful throughput, finish reasons, all failures and unknown tokens; primary/counter/sampled gross/net J per generated token",
            "energy_boundary": "whole-device client arrival plus drain, after a separate 3 s idle baseline; includes Poisson gaps/load onset, not a continuous-load energy microbenchmark",
            "F7": "collective_rpc summary/take_records/reset before and after fully drained client; RPC time is outside the client latency/energy window",
            "failure_rule": "any incomplete, failed, timeout, or unknown-token population stops the server group; later runs are not_run; keep failure results",
            "analysis_population": "only measure_r01 and measure_r02; never glob warmup_excluded into measured repeats",
            "limits": "natural EOS can change output length/content across formats; report actual work and stop/cap/failure counts; two runs give a range, not a significance claim"},
        "validation": {"bench_serving_load_trace": "passed both traces without importing torch or vLLM",
                       "run_serving_group_full_plan": "requires actual generated checkpoint/worker sidecars; no placeholders or mock artifacts used",
                       "dependencies_present_locally": {p: (ROOT / p).is_file() for p in dependency_paths}},
    }
    write_json(HERE / "manifest.json", manifest)
    print(json.dumps({"state": manifest["state"], "requests": len(rows), "warmup_requests": len(warmup),
                      "prompt_tokens": manifest["prompt_tokens"], "trace_sha256": sha256(measurement_trace),
                      "schedule_sha256": digest(scheduled), "last_arrival_s": scheduled[-1]["arrival_s"],
                      "outer_timeout_s_per_arm": 360, "three_arm_timeout_sum_s": 1080}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
