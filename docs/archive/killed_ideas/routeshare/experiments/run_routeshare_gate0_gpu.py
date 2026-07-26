#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parent))
from routeshare_core import make_scenario, scenario_features  # noqa: E402
from routeshare_executor import (  # noqa: E402
    closure_error,
    execute_expert_stage,
    execute_tenants_separately,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise RuntimeError("unsupported config schema")
    return config


def gpu_environment() -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; GPU experiment is blocked")
    device = torch.device("cuda:0")
    name = torch.cuda.get_device_name(device)
    props = torch.cuda.get_device_properties(device)
    try:
        smi = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=uuid,driver_version,power.limit,clocks.sm,memory.total,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        compute_apps = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip().splitlines()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("nvidia-smi environment query failed") from exc
    foreign = []
    for row in compute_apps:
        if not row.strip():
            continue
        try:
            pid = int(row.split(",", 1)[0].strip())
        except ValueError:
            foreign.append(row)
            continue
        if pid != os.getpid():
            foreign.append(row)
    if foreign:
        raise RuntimeError(f"foreign GPU compute process present: {foreign}")
    return {
        "gpu_name": name,
        "compute_capability": f"{props.major}.{props.minor}",
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "nvidia_smi": smi,
        "compute_apps": compute_apps,
    }


def extract_experts(model: torch.nn.Module, layer_id: int):
    layer = model.model.layers[layer_id]
    if hasattr(layer, "block_sparse_moe") and hasattr(layer.block_sparse_moe, "experts"):
        return layer.block_sparse_moe.experts
    if hasattr(layer, "mlp") and hasattr(layer.mlp, "experts"):
        return layer.mlp.experts
    raise RuntimeError(f"cannot locate experts for layer {layer_id}")


def calibrate_repeats(fn, minimum_ms: float, maximum: int) -> int:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    fn()
    end.record()
    torch.cuda.synchronize()
    one_ms = max(float(start.elapsed_time(end)), 1e-4)
    return max(1, min(maximum, int(math.ceil(minimum_ms / one_ms))))


def measure_arm(fn, repeats: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        fn()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end)) / repeats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-key", choices=["olmoe", "llm_jp"], required=True)
    parser.add_argument("--split", choices=["calibration", "sealed"], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)
    environment_before = gpu_environment()
    if environment_before["gpu_name"] != config["gpu_exact_name"]:
        raise RuntimeError(
            f"GPU mismatch: {environment_before['gpu_name']} != {config['gpu_exact_name']}"
        )

    spec = config["models"][args.model_key]
    model = AutoModelForCausalLM.from_pretrained(
        spec["model_id"],
        revision=spec["revision"],
        torch_dtype=torch.bfloat16,
        local_files_only=args.local_files_only,
        low_cpu_mem_usage=True,
    ).to("cuda:0")
    model.eval()
    if len(model.model.layers) != spec["num_layers"]:
        raise RuntimeError("model layer count mismatch")
    if int(model.config.num_experts_per_tok) != spec["top_k"]:
        raise RuntimeError("model top_k mismatch")
    num_experts = getattr(model.config, "num_experts", None)
    if num_experts is None:
        num_experts = getattr(model.config, "num_local_experts")
    if int(num_experts) != spec["num_experts"]:
        raise RuntimeError("model expert count mismatch")
    if int(model.config.hidden_size) != spec["hidden_size"]:
        raise RuntimeError("model hidden size mismatch")

    seeds = config[f"{args.split}_seeds"]
    raw_rows = []
    summary_rows = []
    torch.set_grad_enabled(False)
    for layer_id in config["layers"]:
        experts = extract_experts(model, layer_id)
        if len(experts) != spec["num_experts"]:
            raise RuntimeError("expert module count mismatch")
        for tokens in config["tokens_per_tenant"]:
            for overlap in config["overlap_fractions"]:
                for regime in config["histogram_regimes"]:
                    for seed in seeds:
                        scenario = make_scenario(
                            split=args.split,
                            tokens_per_tenant=tokens,
                            top_k=spec["top_k"],
                            num_experts=spec["num_experts"],
                            overlap_fraction=overlap,
                            histogram_regime=regime,
                            seed=seed,
                        )
                        generator = torch.Generator(device="cuda:0")
                        generator.manual_seed(seed + layer_id * 100000 + tokens * 1000)
                        hidden = torch.randn(
                            2 * tokens,
                            spec["hidden_size"],
                            dtype=torch.bfloat16,
                            device="cuda:0",
                            generator=generator,
                        )
                        selected = torch.as_tensor(
                            scenario.selected_experts, dtype=torch.long, device="cuda:0"
                        )
                        weights = torch.full(
                            selected.shape,
                            1.0 / spec["top_k"],
                            dtype=torch.bfloat16,
                            device="cuda:0",
                        )
                        tenants = torch.as_tensor(
                            scenario.tenant_ids, dtype=torch.long, device="cuda:0"
                        )

                        coalition_fn = lambda: execute_expert_stage(
                            experts, hidden, selected, weights
                        )
                        separate_fn = lambda: execute_tenants_separately(
                            experts, hidden, selected, weights, tenants
                        )
                        # Identity relabel is intentionally executor-invisible.
                        relabeled = 1 - tenants
                        sham_fn = lambda: execute_tenants_separately(
                            experts, hidden, selected, weights, relabeled
                        )
                        for _ in range(config["warmup_calls"]):
                            coalition_fn()
                            separate_fn()
                            sham_fn()
                        coalition_output = coalition_fn()
                        separate_output = separate_fn()
                        max_abs, max_rel = closure_error(coalition_output, separate_output)
                        if max_abs > config["gate"]["output_max_abs"] or max_rel > config["gate"]["output_max_rel"]:
                            raise RuntimeError(
                                f"executor closure failed: abs={max_abs}, rel={max_rel}"
                            )
                        arms = {
                            "coalition": coalition_fn,
                            "tenant_separate": separate_fn,
                            "identity_relabel_sham": sham_fn,
                        }
                        repeats = {
                            name: calibrate_repeats(
                                fn,
                                config["minimum_cuda_ms_per_trial"],
                                config["maximum_inner_repeats"],
                            )
                            for name, fn in arms.items()
                        }
                        scenario_trials = {name: [] for name in arms}
                        base_order = list(arms)
                        for trial in range(config["paired_trials"]):
                            order = base_order if (trial + int(scenario.scenario_id[:2], 16)) % 2 == 0 else list(reversed(base_order))
                            for arm in order:
                                latency = measure_arm(arms[arm], repeats[arm])
                                scenario_trials[arm].append(latency)
                                raw_rows.append(
                                    {
                                        "model_key": args.model_key,
                                        "layer_id": layer_id,
                                        "scenario_id": scenario.scenario_id,
                                        "split": args.split,
                                        "arm": arm,
                                        "trial": trial,
                                        "inner_repeats": repeats[arm],
                                        "latency_ms": latency,
                                    }
                                )
                        features = scenario_features(scenario)
                        summary_rows.append(
                            {
                                "model_key": args.model_key,
                                "layer_id": layer_id,
                                **features,
                                "coalition_latency_ms": float(np.median(scenario_trials["coalition"])),
                                "tenant_separate_latency_ms": float(np.median(scenario_trials["tenant_separate"])),
                                "sham_latency_ms": float(np.median(scenario_trials["identity_relabel_sham"])),
                                "coalition_over_separate": float(
                                    np.median(scenario_trials["coalition"])
                                    / np.median(scenario_trials["tenant_separate"])
                                ),
                                "closure_max_abs": max_abs,
                                "closure_max_rel": max_rel,
                            }
                        )
                        print(
                            f"{args.model_key} L{layer_id} {args.split} tokens={tokens} "
                            f"overlap={overlap} {regime} seed={seed} done",
                            flush=True,
                        )

    environment_after = gpu_environment()
    pd.DataFrame(raw_rows).to_csv(args.output_dir / "trial_measurements.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(args.output_dir / "scenario_summary.csv", index=False)
    metadata = {
        "model_key": args.model_key,
        "split": args.split,
        "config_sha256": sha256_file(args.config),
        "runner_sha256": sha256_file(Path(__file__)),
        "core_sha256": sha256_file(Path(__file__).with_name("routeshare_core.py")),
        "executor_sha256": sha256_file(Path(__file__).with_name("routeshare_executor.py")),
        "environment_before": environment_before,
        "environment_after": environment_after,
        "evidence_boundary": "single-GPU single-layer BF16 executable oracle; not serving/network",
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
