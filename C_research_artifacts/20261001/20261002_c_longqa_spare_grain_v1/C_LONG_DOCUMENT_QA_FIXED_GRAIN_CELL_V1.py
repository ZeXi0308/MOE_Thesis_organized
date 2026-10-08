#!/usr/bin/env python3
"""Fixed-grain new-request enqueue order on the unchanged density GPU cell.

This adapter changes only the prompt-only enqueue permutation. It records the
first successful KV allocation as an admission proxy, not GPU execution time.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import C_LONG_DOCUMENT_QA_DENSITY_CELL_V1 as density
from C_LONG_DOCUMENT_QA_FIXED_GRAIN_ORDER_V1 import WORKLOAD_SHA, fixed_grain_order
from C_SPARE_CELL_HELPERS_V1 import dump, sha

RECEIPT_NAME = "long_document_qa_fixed_grain_order_v1.json"
ORDER_SOURCE = "C_LONG_DOCUMENT_QA_FIXED_GRAIN_ORDER_V1.py"


def run(inputs: Path, output: Path, parent: Path, model_dir: Path,
        model_manifest: Path, quantum: str) -> None:
    frozen_path = Path(__file__).parent / RECEIPT_NAME
    frozen = json.loads(frozen_path.read_text())
    if (sha(inputs / "workload.json") != WORKLOAD_SHA
            or frozen.get("schema") != "c-longqa-fixed-grain-order-v1"
            or frozen.get("workload_sha256") != WORKLOAD_SHA
            or frozen.get("request_count") != 150
            or quantum not in frozen.get("arms", {})):
        raise RuntimeError("fixed-grain workload or frozen CPU receipt differs")
    expected_receipt = frozen["arms"][quantum]
    expected_order = expected_receipt["source_indices_in_submission_order"]
    q = quantum if quantum == "whole" else int(quantum)
    original_generate = density.generate

    def generate(engine, requests, max_tokens, *, run_id, output_dir,
                 max_seconds=900):
        scheduler = engine.engine_core.engine_core.scheduler
        manager = scheduler.kv_cache_manager
        raw_schedule, raw_allocate = scheduler.schedule, manager.allocate_slots
        schedule_call = -1
        first_allocation, seen = [], set()
        group_receipt = None

        def reorder(sequences):
            nonlocal group_receipt
            order, group_receipt = fixed_grain_order(sequences, q)
            if run_id == "measured" and group_receipt != expected_receipt:
                raise RuntimeError("measured fixed-grain order/group receipt differs from frozen CPU result")
            return order

        def schedule(*args, **kwargs):
            nonlocal schedule_call
            schedule_call += 1
            return raw_schedule(*args, **kwargs)

        def allocate(request, num_new_tokens, *args, **kwargs):
            request_id = request.request_id
            status_before = request.status.name
            previous_computed = int(request.num_computed_tokens)
            result = raw_allocate(request, num_new_tokens, *args, **kwargs)
            if result is not None and request_id not in seen:
                seen.add(request_id)
                first_allocation.append(dict(
                    schedule_call=schedule_call, request_id=request_id,
                    request_status_before=status_before,
                    prompt_length_tokens=len(request.prompt_token_ids),
                    previous_computed_tokens=previous_computed,
                    new_prefix_cached_tokens=int(kwargs.get("num_new_computed_tokens", 0)),
                    new_external_computed_tokens=int(kwargs.get("num_external_computed_tokens", 0)),
                    scheduled_compute_tokens=int(num_new_tokens)))
            return result

        density.DENSITY_REORDER = reorder
        density.EXPECTED_ORDER = expected_order
        scheduler.schedule, manager.allocate_slots = schedule, allocate
        try:
            summary = original_generate(engine, requests, max_tokens,
                run_id=run_id, output_dir=output_dir, max_seconds=max_seconds)
            expected_ids = {run_id + "/" + row["request_id"] for row in requests}
            if seen != expected_ids:
                raise RuntimeError("completed requests lack exactly one first successful allocation")
            return summary
        finally:
            scheduler.schedule, manager.allocate_slots = raw_schedule, raw_allocate
            if group_receipt is not None:
                dump(output_dir / (run_id + "-group-receipt.json"), dict(
                    group_receipt, run_id=run_id,
                    request_ids_in_submission_order=[
                        requests[i]["request_id"] for i in
                        group_receipt["source_indices_in_submission_order"]]))
            steps_path = output_dir / (run_id + "-steps.json")
            if steps_path.exists():
                steps = json.loads(steps_path.read_text())
                steps["first_successful_allocation"] = first_allocation
                steps["first_successful_allocation_scope"] = (
                    "First successful KV allocate_slots call for each request, in call order; "
                    "a new-request admission proxy, not GPU execution or an active-count gate")
                steps["quantum"] = quantum
                dump(steps_path, steps)

    density.generate = generate
    try:
        density.run(inputs, output, parent, model_dir, model_manifest)
    finally:
        density.generate = original_generate
        if output.is_dir():
            dump(output / "fixed-grain-source.json", dict(
                quantum=quantum, frozen_receipt_sha256=sha(frozen_path),
                fixed_grain_order_sha256=sha(Path(__file__).parent / ORDER_SOURCE),
                density_order_sha256=sha(Path(__file__).parent /
                    "C_LONG_DOCUMENT_QA_DENSITY_ORDER_V1.py"),
                density_cell_sha256=sha(Path(density.__file__)),
                adapter_cell_sha256=sha(Path(__file__)),
                scope="Fixed first-32-token group release order over unchanged density ranking; "
                      "native FCFS, zero external arrivals; no active-count gate or decode preemption"))
            density_source_path = output / "density-reorder-source.json"
            if density_source_path.exists():
                density_source = json.loads(density_source_path.read_text())
                density_source["scope"] = (
                    "Base density ranking source only; actual fixed-grain submission order "
                    "is in measured-group-receipt.json")
                dump(density_source_path, density_source)
            environment_path = output / "environment.json"
            if environment_path.exists():
                environment = json.loads(environment_path.read_text())
                environment["fixed_grain_adapter_cell_sha256"] = sha(Path(__file__))
                dump(environment_path, environment)
            status_path = output / "status.json"
            if status_path.exists():
                status = json.loads(status_path.read_text())
                status["quantum"] = quantum
                status["group_receipt_file"] = "measured-group-receipt.json"
                status["ordering_cpu_seconds_field"] = "subtree_density_reorder_s"
                status["scientific_scope"] = (
                    "Fixed operational prefix-group new-request enqueue order over the same "
                    "density rank; native FCFS and unchanged model, warmups, sampling and resources; "
                    "development comparison only, no novelty or full-prior reproduction claim")
                dump(status_path, status)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--quantum", choices=("whole", "1", "2"), required=True)
    args = parser.parse_args()
    run(args.inputs, args.output, args.parent, args.model_dir,
        args.model_manifest, args.quantum)
