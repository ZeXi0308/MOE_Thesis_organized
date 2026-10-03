#!/usr/bin/env python3
"""Small real OLMoE cap scan. No CPU performance fallback and no synthetic curves."""
from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

from metrics import summarize_episode_requests
from runtime import ROOT, kv, run_episode


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def checkout_state():
    """A minimal source export need not contain Git; source hashes still apply."""
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                          text=True, capture_output=True)
    if head.returncode:
        return dict(git_head=None, git_status=None, execution_checkout="SOURCE_EXPORT")
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
    return dict(git_head=head.stdout.strip(), git_status=status, execution_checkout="GIT_CHECKOUT")


def gpu_state():
    gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=name,uuid,memory.total,memory.used,temperature.gpu,power.draw,clocks.sm",
                                   "--format=csv,noheader"], text=True).strip()
    processes = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                                         "--format=csv,noheader"], text=True).strip()
    foreign = [r for r in csv.reader(processes.splitlines()) if r and int(r[0]) != os.getpid()]
    if foreign:
        raise RuntimeError(f"GPU has other compute processes: {foreign}")
    return dict(gpu=gpu, compute_processes=processes)


def select_source_requests(source, prompt_tokens, requests, request_offset=0):
    """Filter by length, then take one deterministic contiguous cohort."""
    if type(request_offset) is not int or request_offset < 0:
        raise ValueError("request-offset must be a nonnegative integer")
    eligible = [r for r in source["requests"] if r["prompt_token_count"] >= prompt_tokens]
    selected = eligible[request_offset:request_offset + requests]
    if len(selected) != requests:
        raise ValueError(f"not enough eligible requests for offset={request_offset}, count={requests}; available={len(eligible)}")
    return selected


def make_workload(tokenizer, args, source, device):
    selected = select_source_requests(source, args.prompt_tokens, args.requests,
                                      getattr(args, "request_offset", 0))
    requests = []
    for r in selected:
        if hashlib.sha256(r["prompt"].encode()).hexdigest() != r["prompt_sha256"]:
            raise ValueError("source prompt hash mismatch")
        encoded = tokenizer(r["prompt"], return_tensors="pt", truncation=True, max_length=args.prompt_tokens)
        ids = encoded["input_ids"].to(device)
        if ids.shape[1] != args.prompt_tokens:
            raise ValueError("prompt length differs from the requested matched-length cohort")
        if args.prompt_tokens == source["max_prompt_tokens"] and kv._prompt_token_ids_sha256(ids) != r["prompt_token_ids_sha256"]:
            raise ValueError("tokenized prompt differs from the pinned source manifest")
        requests.append(kv.ContinuousRequest(r["request_id"], r["sample_id"], r["document_id"],
            0, 0, ids, encoded["attention_mask"].to(device)))
    return selected, requests


def arrival_traces(count, gap, burst_size):
    last_group = (count - 1) // burst_size
    if last_group == 0:
        raise ValueError("burst size must be smaller than the cohort")
    return {"steady": [i * gap for i in range(count)],
            "bursty": [(i // burst_size) / last_group * (count - 1) * gap for i in range(count)]}


def workload_payload(selected, requests, args):
    return dict(schema="olmoe-admission-inputs-v1", source_requests=selected,
        actual_prompt_token_ids=[r.input_ids.flatten().cpu().tolist() for r in requests],
        arrival_traces_s=arrival_traces(len(requests), args.arrival_gap_s, args.burst_size),
        arrival_rule="steady evenly spaced; grouped burst arrivals with identical first/last arrival",
        document_identity_scope="source WikiText rows, not certified article-disjoint holdout")


def validate_prepared(workload, config, source):
    selected = select_source_requests(source, config["prompt_tokens"], config["requests"],
                                      config.get("request_offset", 0))
    if workload["source_requests"] != selected or len(selected) != config["requests"]:
        raise ValueError("prepared cohort differs from deterministic source selection")
    ids = workload["actual_prompt_token_ids"]
    if len(ids) != len(selected):
        raise ValueError("prepared token/cohort count mismatch")
    for row, tokens in zip(selected, ids):
        if len(tokens) != config["prompt_tokens"] or any(type(t) is not int or t < 0 for t in tokens):
            raise ValueError("invalid prepared prompt token IDs")
        digest = hashlib.sha256(json.dumps(tokens, separators=(",", ":")).encode()).hexdigest()
        if config["prompt_tokens"] == source["max_prompt_tokens"] and digest != row["prompt_token_ids_sha256"]:
            raise ValueError("prepared prompt token identity drift")
    if workload["arrival_traces_s"] != arrival_traces(len(selected), config["arrival_gap_s"], config["burst_size"]):
        raise ValueError("prepared arrival trace drift")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true", help="freeze exact inputs and run CPU/cache preflight; no GPU model load")
    parser.add_argument("--prepared-dir", type=Path, help="execute an existing prepared config/cohort without respecifying workload knobs")
    parser.add_argument("--caps", default="2,4,8")
    parser.add_argument("--requests", type=int, default=16)
    parser.add_argument("--request-offset", type=int, default=0, help="offset after filtering source requests by prompt length")
    parser.add_argument("--prompt-tokens", type=int, default=128)
    parser.add_argument("--output-tokens", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--arrival-gap-s", type=float)
    parser.add_argument("--burst-size", type=int, default=4)
    parser.add_argument("--ttft-slo-s", type=float)
    parser.add_argument("--tpot-slo-s", type=float)
    parser.add_argument("--max-cell-seconds", type=float, default=600)
    parser.add_argument("--max-run-seconds", type=float, default=1200, help="total execution budget including model loading and warmup")
    parser.add_argument("--natural-stop", action="store_true")
    parser.add_argument("--model-path", help="optional local snapshot of the pinned model revision")
    parser.add_argument("--cap-schedule", help="optional JSON [[seconds, target], ...]; replaces the static cap scan")
    parser.add_argument("--step-action", help="one [completed_decode_steps, target_cap] action after the common prefix")
    parser.add_argument("--warmup-caps", help="optional common warmup widths, e.g. 6,8, for both hold and up arms")
    args = parser.parse_args()
    prepared_config = prepared_workload = None
    if args.prepared_dir:
        allowed = {"--output-dir", "--prepared-dir", "--model-path", "--prepare-only"}
        if any(value.split("=")[0] not in allowed for value in sys.argv[1:] if value.startswith("--")):
            parser.error("a prepared plan freezes workload knobs; create a new preparation to change them")
        prepared_config = json.loads((args.prepared_dir / "config.json").read_text())
        prepared_workload = json.loads((args.prepared_dir / "workload.json").read_text())
        for name in ("caps", "requests", "prompt_tokens", "output_tokens", "repeats", "arrival_gap_s",
                     "burst_size", "ttft_slo_s", "tpot_slo_s", "max_cell_seconds", "max_run_seconds", "natural_stop", "cap_schedule"):
            setattr(args, name, prepared_config[name])
        args.request_offset = prepared_config.get("request_offset", 0)
        args.step_action = prepared_config.get("step_action")
        args.warmup_caps = prepared_config.get("warmup_caps")
    if any(getattr(args, name) is None for name in ("arrival_gap_s", "ttft_slo_s", "tpot_slo_s")):
        parser.error("supply arrival gap and TTFT/TPOT SLO, or --prepared-dir")
    caps = [int(x) for x in args.caps.split(",")]
    if not caps or min(caps) < 1 or len(set(caps)) != len(caps):
        parser.error("caps must be unique positive integers")
    if min(args.requests, args.prompt_tokens, args.repeats, args.burst_size) < 1 or args.output_tokens < 2:
        parser.error("invalid workload size")
    if type(args.request_offset) is not int or args.request_offset < 0:
        parser.error("request-offset must be a nonnegative integer")
    import math
    if any(not math.isfinite(v) or v <= 0 for v in
           (args.arrival_gap_s, args.ttft_slo_s, args.tpot_slo_s, args.max_cell_seconds, args.max_run_seconds)):
        parser.error("arrival, SLO and runtime values must be finite and positive")
    if args.max_cell_seconds <= (args.requests - 1) * args.arrival_gap_s:
        parser.error("runtime limit must extend beyond all arrivals")
    schedule = json.loads(args.cap_schedule) if args.cap_schedule else None
    if args.cap_schedule is not None:
        from runtime import validate_schedule
        validate_schedule(schedule)
        caps = [schedule[0][1]]
    step_action = json.loads(args.step_action) if args.step_action else None
    if step_action is not None:
        if (len(caps) != 1 or (schedule is not None and len(schedule) != 1)
                or not isinstance(step_action, list) or len(step_action) != 2
                or any(type(x) is not int or x < 1 for x in step_action)):
            parser.error("step-action requires one initial cap and [positive integer completed steps, positive integer target]")
    all_caps = [int(s[1]) for s in schedule] if schedule else caps
    if step_action:
        all_caps.append(step_action[1])
    warmup_caps = [int(x) for x in args.warmup_caps.split(",")] if args.warmup_caps else all_caps
    if not warmup_caps or min(warmup_caps) < 1 or max(warmup_caps) > args.requests:
        parser.error("warmup caps must be positive and fit the cohort")
    warmup_caps = sorted(set(all_caps + warmup_caps))
    if max(all_caps) > args.requests or args.burst_size >= args.requests:
        parser.error("caps must fit cohort and burst-size must be smaller than requests")
    if not args.natural_stop and args.output_tokens <= max(all_caps):
        parser.error("one-prefill-per-iteration needs output-tokens > cap for the requested cap to be reachable")
    output = args.output_dir
    source_path = ROOT / "docs/ideas/bcrd/experiments/configs/workloads/olmoe.formal.json"
    source = json.loads(source_path.read_text())
    plans = []
    for repeat in range(args.repeats):
        order = caps if repeat % 2 == 0 else list(reversed(caps))
        for regime in ("steady", "bursty"):
            for cap in order:
                for telemetry in ([False, True] if repeat % 2 == 0 else [True, False]):
                    plans.append(dict(repeat=repeat, regime=regime, cap=cap, telemetry=telemetry))
    config = vars(args).copy()
    config["output_dir"] = str(output)
    config["prepared_dir"] = str(args.prepared_dir) if args.prepared_dir else None
    config.update(model=source["model"], run_order=plans, all_runs_retained=True, cell_checks_required=True,
        source_manifest_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
        evidence_ceiling="CUSTOM_CONTINUOUS_RUNTIME_EXPLORATORY",
        slo_kind="user_supplied_exploratory_thresholds", seed=20260905,
        measurement_rule="OFF and ON independent real reruns; never attach ON pressure to OFF future state")
    if prepared_config:
        if (prepared_config["source_manifest_sha256"] != config["source_manifest_sha256"]
                or prepared_config["model"] != config["model"] or prepared_config["run_order"] != plans):
            raise ValueError("prepared plan/source identity drift")
        validate_prepared(prepared_workload, config, source)
        if hashlib.sha256(json.dumps(prepared_workload, sort_keys=True).encode()).hexdigest() != prepared_config["workload_sha256"]:
            raise ValueError("prepared workload bytes/values changed")
        config["prepared_config_sha256"] = hashlib.sha256((args.prepared_dir / "config.json").read_bytes()).hexdigest()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_json(output / "config.json", config)
    (output / "commands.txt").write_text(shlex.join([sys.executable, *sys.argv]) + "\n")
    write_json(output / "status.json", dict(status="STARTED", cells_completed=0))
    curves = []
    cells_executed = 0
    try:
        import torch
        import transformers
        from preflight import inspect_environment
        env = dict(python=sys.version, torch=torch.__version__, transformers=transformers.__version__,
                   cuda=torch.version.cuda, cuda_available=torch.cuda.is_available(), **checkout_state())
        env.update(cpu_threads=torch.get_num_threads(), cpu_interop_threads=torch.get_num_interop_threads(),
                   execution_env={k: os.environ.get(k) for k in
                                  ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "CUDA_LAUNCH_BLOCKING")})
        env["gpu_isolation_observation"] = "before_load_and_cell_boundaries_only_not_continuous"
        # Hash the actual helper bytes too, because a dirty checkout is permitted.
        env["source_sha256"] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in [Path(__file__), Path(__file__).with_name("runtime.py"),
                                          Path(__file__).with_name("metrics.py"),
                                          Path(__file__).with_name("preflight.py"), Path(kv.__file__),
                                          Path(kv.__file__).with_name("core.py")]}
        write_json(output / "environment.json", env)
        model_id = args.model_path or source["model"]["id"]
        revision = source["model"]["revision"]
        preflight = inspect_environment(model_id, revision, require_cuda=False)
        write_json(output / "preflight.json", preflight)
        if preflight["blockers"]:
            raise RuntimeError("preflight blocked: " + "; ".join(preflight["blockers"]))
        if args.prompt_tokens + args.output_tokens > preflight["model_config"]["max_position_embeddings"]:
            raise ValueError("prompt + output exceeds the cached model context limit")
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, revision=revision, local_files_only=True)
        if prepared_workload is None:
            selected, requests = make_workload(tokenizer, args, source, "cpu")
            workload = workload_payload(selected, requests, args)
        else:
            workload = prepared_workload
            requests = [kv.ContinuousRequest(r["request_id"], r["sample_id"], r["document_id"], 0, 0,
                        torch.tensor([ids], dtype=torch.long), torch.ones((1, len(ids)), dtype=torch.long))
                        for r, ids in zip(workload["source_requests"], workload["actual_prompt_token_ids"])]
        validate_prepared(workload, config, source)
        write_json(output / "workload.json", workload)
        config["workload_sha256"] = hashlib.sha256(json.dumps(workload, sort_keys=True).encode()).hexdigest()
        write_json(output / "config.json", config)
        if args.prepare_only:
            write_json(output / "status.json", dict(status="PREPARED_CPU_CHECKED_GPU_UNRUN", cells_executed=0,
                       scientific_verdict="UNRUN", planned_cells=len(plans), exact_prompt_count=len(requests)))
            print(f"Prepared {len(requests)} exact prompts and {len(plans)} cells; GPU experiment UNRUN.")
            return 0
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; pretrained GPU capacity experiment is UNRUN")
        if torch.cuda.device_count() != 1:
            raise RuntimeError("expose exactly one GPU with CUDA_VISIBLE_DEVICES")
        execution_started = time.perf_counter()

        def remaining_budget():
            remaining = min(args.max_cell_seconds, args.max_run_seconds - (time.perf_counter() - execution_started))
            if remaining <= max(workload["arrival_traces_s"]["steady"]):
                raise RuntimeError("total execution budget exhausted; all prior cells retained")
            return remaining

        env["gpu_before_load"] = gpu_state()
        write_json(output / "environment.json", env)
        preflight = inspect_environment(model_id, revision, require_cuda=True)
        write_json(output / "gpu_preflight.json", preflight)
        if preflight["blockers"]:
            raise RuntimeError("GPU preflight blocked: " + "; ".join(preflight["blockers"]))
        torch.manual_seed(20260905)
        torch.cuda.manual_seed_all(20260905)
        model = transformers.AutoModelForCausalLM.from_pretrained(model_id, revision=revision,
            torch_dtype=torch.bfloat16, local_files_only=True, attn_implementation="eager").to("cuda").eval()
        if model.config.model_type != "olmoe":
            raise ValueError("this probe only supports the OLMoE routing implementation")
        env.update(resolved_model_commit=getattr(model.config, "_commit_hash", None),
                   model_path_revision_verified=args.model_path is None,
                   attention_implementation=model.config._attn_implementation)
        write_json(output / "environment.json", env)
        if args.prompt_tokens + args.output_tokens > model.config.max_position_embeddings:
            raise ValueError("prompt + output exceeds the model context limit")
        requests = [replace(r, input_ids=r.input_ids.to(model.device), attention_mask=r.attention_mask.to(model.device)) for r in requests]
        # Warm every width once before measured episodes; fresh KV each time.
        for cap in warmup_caps:
            warm = run_episode(model, requests[:min(cap, len(requests))], cap_schedule=[(0, cap)],
                               output_tokens=max(3, cap + 1), telemetry=True, max_seconds=remaining_budget())
            if warm["status"] != "COMPLETE":
                raise RuntimeError(f"warmup failed: {warm['error']}")
            del warm
        for index, plan in enumerate(plans):
            before = gpu_state()
            current = [replace(r, arrival_us=workload["arrival_traces_s"][plan["regime"]][i] * 1e6)
                       for i, r in enumerate(requests)]
            result = run_episode(model, current, cap_schedule=schedule or [(0, plan["cap"])],
                step_action=step_action,
                output_tokens=args.output_tokens, telemetry=plan["telemetry"],
                eos_token_id=tokenizer.eos_token_id if args.natural_stop else None,
                max_seconds=remaining_budget())
            result.update(plan=plan, gpu_before=before)
            # Persist raw results before post-run checks or metric computation.
            write_json(output / f"cell-{index:03d}.json", result)
            cells_executed += 1
            # Keep failed isolation checks beside immutable raw; the analyzer must
            # not select a complete-looking cell whose post-run check failed.
            try:
                after = gpu_state()
            except Exception as exc:
                write_json(output / f"checks-{index:03d}.json",
                           dict(status="FAILED", reason=f"{type(exc).__name__}: {exc}"))
                raise
            write_json(output / f"checks-{index:03d}.json", dict(status="PASS", gpu_after=after))
            result["gpu_after"] = after
            metrics = summarize_episode_requests(result["requests"], observation_end_s=result["observation_end_s"],
                ttft_slo_s=args.ttft_slo_s, tpot_slo_s=args.tpot_slo_s)
            write_json(output / f"metrics-{index:03d}.json", dict(metrics=metrics, gpu_after=after))
            pressures = [p for s in result["steps"] for p in s["pressure"] if p["U"] is not None]
            curve = dict(cell=index, **plan, status=result["status"],
                goodput_rps=metrics["goodput_rps"], slo_attainment=metrics["slo_attainment"],
                throughput_rps=metrics["throughput_rps"], n_completed=metrics["n_completed"],
                ttft_p50_s=metrics["latency_s"]["ttft"]["p50"],
                tpot_p50_s=metrics["latency_s"]["tpot"]["p50"],
                U_mean=sum(p["U"] for p in pressures)/len(pressures) if pressures else None,
                C_mean=sum(p["C"] for p in pressures)/len(pressures) if pressures else None)
            curves.append(curve)
            with (output / "curves.csv").open("w") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(curve))
                writer.writeheader()
                writer.writerows(curves)
            print(json.dumps(curve), flush=True)
            write_json(output / "status.json", dict(status="RUNNING", cells_completed=len(curves)))
            if result["status"] != "COMPLETE":
                raise RuntimeError(f"cell incomplete; retained all requests: {result['error']}")
        write_json(output / "status.json", dict(status="COMPLETE", cells_completed=len(curves),
                   scientific_verdict="MEASUREMENT_ONLY_PENDING_RESPONSE_ANALYSIS"))
    except Exception as exc:
        write_json(output / "status.json", dict(status="BLOCKED" if not cells_executed else "INCOMPLETE",
                   reason=f"{type(exc).__name__}: {exc}", cells_executed=cells_executed,
                   cells_completed=sum(c["status"] == "COMPLETE" for c in curves),
                   scientific_verdict="UNRUN" if not cells_executed else "PARTIAL_MEASUREMENT"))
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
