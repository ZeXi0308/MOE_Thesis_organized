#!/usr/bin/env python3
"""Shared large odd prefill warmup for DFS and fixed-grain LongBench cells."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import C_LONG_DOCUMENT_QA_DFS_CELL_V1 as dfs
import C_LONG_DOCUMENT_QA_DENSITY_CELL_V1 as density
import C_LONG_DOCUMENT_QA_FIXED_GRAIN_CELL_V2 as grain
from C_LONG_DOCUMENT_QA_FIXED_GRAIN_ORDER_V1 import WORKLOAD_SHA
from C_SPARE_CELL_HELPERS_V1 import dump, sha


def run(inputs: Path, output: Path, parent: Path, model_dir: Path,
        model_manifest: Path, policy: str) -> None:
    base = dfs if policy == "dfs" else density
    config, _, requests = base.validate_inputs(inputs)
    source = requests[0]
    batch_budget = config["proposed_runtime"]["max_num_batched_tokens"]
    prompt_tokens = batch_budget - 1
    if (sha(inputs / "workload.json") != WORKLOAD_SHA
            or batch_budget != 1024 or source["source_index"] != 0
            or len(source["prompt_token_ids"]) < prompt_tokens):
        raise RuntimeError("frozen source-zero 1023-token warmup contract differs")
    ids = source["prompt_token_ids"][:prompt_tokens]
    synthetic = dict(request_id="synthetic-source0-prefix1023", source_index=0,
                     prompt_token_ids=ids)
    original_reset = base.reset_warmup_cache

    def reset_with_shape_warmup(engine):
        # Called by the base cell exactly after warmup-last and before its
        # measured run. The extra pre-reset removes source-zero warmup-first
        # APC hits; the original reset contract is returned after the shape.
        started = time.perf_counter()
        reset_started = time.perf_counter()
        pre_reset = original_reset(engine)
        reset_seconds = time.perf_counter() - reset_started
        dump(output / "warmup-large-odd-pre-reset.json", pre_reset)
        generate_started = time.perf_counter()
        summary = base.generate(engine, [synthetic], 1,
            run_id="warmup-large-odd", output_dir=output)
        generate_seconds = time.perf_counter() - generate_started
        steps = json.loads((output / "warmup-large-odd-steps.json").read_text())
        scheduled = [call["scheduled_tokens_total"]
                     for call in steps["scheduler_calls"]]
        if scheduled.count(prompt_tokens) != 1:
            raise RuntimeError("synthetic warmup lacked exactly one 1023-token schedule call")
        second_reset_started = time.perf_counter()
        post_reset = original_reset(engine)
        second_reset_seconds = time.perf_counter() - second_reset_started
        dump(output / "shape-warmup-source.json", dict(
            policy=policy, source_index=0, source_request_id=source["request_id"],
            source_prompt_tokens=len(source["prompt_token_ids"]),
            prompt_tokens=prompt_tokens, max_tokens=1,
            prompt_token_ids_sha256=hashlib.sha256(json.dumps(ids,
                separators=(",", ":")).encode()).hexdigest(),
            input_workload_sha256=WORKLOAD_SHA,
            max_num_batched_tokens=batch_budget,
            pre_synthetic_prefix_reset_file="warmup-large-odd-pre-reset.json",
            measurement_prefix_reset_file="prefix-cache-reset.json",
            scheduled_tokens_per_call=scheduled,
            schedule_calls_with_1023_tokens=scheduled.count(prompt_tokens),
            pre_synthetic_reset_wall_s=reset_seconds,
            synthetic_generate_wall_s=generate_seconds,
            measurement_reset_wall_s=second_reset_seconds,
            total_shape_control_wall_s=time.perf_counter() - started,
            synthetic_summary=summary,
            adapter_sha256=sha(Path(__file__)),
            scope="Extra source-zero 1023-token prompt prefix and one generated token; "
                  "first/last warmups and measured 150 requests unchanged; compile/runtime "
                  "warmup cost is separate from measured request latency"))
        return post_reset

    base.reset_warmup_cache = reset_with_shape_warmup
    try:
        if policy == "dfs":
            base.run(inputs, output, parent, model_dir, model_manifest)
        else:
            grain.run(inputs, output, parent, model_dir, model_manifest, policy)
    finally:
        base.reset_warmup_cache = original_reset
        if output.is_dir():
            environment_path = output / "environment.json"
            if environment_path.exists():
                environment = json.loads(environment_path.read_text())
                environment["shape_warmup_adapter_sha256"] = sha(Path(__file__))
                dump(environment_path, environment)
            status_path = output / "status.json"
            if status_path.exists():
                status = json.loads(status_path.read_text())
                status["shape_warmup_policy"] = policy
                status["shape_warmup_source_file"] = "shape-warmup-source.json"
                status["scientific_scope"] = (
                    status.get("scientific_scope", "") + "; shared extra 1023-token "
                    "prefill plus one generated token before the original measurement "
                    "cache reset; additional warmup time is recorded separately")
                dump(status_path, status)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--policy", choices=("dfs", "whole", "1", "2"), required=True)
    args = parser.parse_args()
    run(args.inputs, args.output, args.parent, args.model_dir,
        args.model_manifest, args.policy)
