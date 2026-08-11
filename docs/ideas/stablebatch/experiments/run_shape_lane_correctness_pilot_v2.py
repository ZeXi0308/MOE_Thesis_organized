#!/usr/bin/env python3
"""D10 C=8 correctness Gate v2 with document-disjoint companion donors.

Run01 was preserved as INVALID because the 16-token target windows did not
contain 21 distinct companion rows for every target expert.  V2 does not
change C, contexts, slots, repetitions, coverage threshold, or decisions.  It
adds docs 16..31 at offset 0, width 512 as companion-pool donors only; donor
rows never choose target cells and their outputs are never scored.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_shape_lane_correctness_pilot as v1  # noqa: E402
import run_single_contribution_pilot as base  # noqa: E402


ProtocolError = base.ProtocolError


def load_donor_workloads(
    config: Mapping[str, Any], repo_root: Path, tokenizer: Any
) -> list[dict[str, Any]]:
    donor = config["companion_donors"]
    manifest_path = (repo_root / str(donor["manifest"])).resolve()
    if not manifest_path.is_file():
        raise ProtocolError("donor manifest is absent")
    if base.sha256_file(manifest_path) != str(donor["manifest_sha256"]):
        raise ProtocolError("donor manifest hash mismatch")
    donor_documents = set(map(int, donor["document_indices"]))
    target_documents = set(map(int, donor["target_document_indices"]))
    if donor_documents.intersection(target_documents):
        raise ProtocolError("donor and target documents overlap")
    documents = {
        int(row["document_index"]): row for row in base.load_jsonl(manifest_path)
    }
    workloads: list[dict[str, Any]] = []
    offset = int(donor["token_offset"])
    width = int(donor["window_tokens"])
    for document_index in map(int, donor["document_indices"]):
        if document_index not in documents:
            raise ProtocolError(f"donor document is absent: {document_index}")
        document = documents[document_index]
        text = str(document["text"])
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if text_hash != str(document["text_sha256"]):
            raise ProtocolError(f"donor text hash mismatch: {document_index}")
        token_ids = tokenizer(
            text, add_special_tokens=bool(donor["add_special_tokens"])
        )["input_ids"]
        window = list(map(int, token_ids[offset : offset + width]))
        if len(window) != width:
            raise ProtocolError(f"donor window is incomplete: {document_index}")
        workloads.append(
            {
                "victim_id": f"donor-doc{document_index:03d}-offset{offset:04d}",
                "document_index": document_index,
                "text_sha256": text_hash,
                "window_token_ids": window,
                "window_token_ids_sha256": hashlib.sha256(
                    base.canonical_json_bytes(window)
                ).hexdigest(),
                "companion_pool_only": True,
            }
        )
    if len(workloads) != len(donor_documents):
        raise ProtocolError("donor workload count mismatch")
    return workloads


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-wall-seconds", type=int, default=3600)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = args.repo_root.resolve()
    config_path = args.config.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise ProtocolError(f"refusing to reuse output directory: {output_dir}")
    config = base.load_json(config_path)
    if config.get("schema_version") != "stablebatch-shape-lane-correctness-pilot-v2":
        raise ProtocolError("unexpected V2 config schema")
    if config.get("status") != "FROZEN_PRE_RUN":
        raise ProtocolError("V2 config is not frozen")
    base_config_path = v1.load_bound_file(repo_root, config["base_config"])
    target_path = v1.load_bound_file(repo_root, config["selected_targets"])
    base_config = base.load_json(base_config_path)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir()
    started = time.time()
    try:
        base.write_json_new(
            output_dir / "run_request.json",
            {
                "schema_version": "stablebatch-shape-lane-run-request-v2",
                "started_at": base.utc_now(),
                "argv": sys.argv,
                "runner_sha256": base.sha256_file(Path(__file__).resolve()),
                "v1_runner_sha256": base.sha256_file(HERE / "run_shape_lane_correctness_pilot.py"),
                "config_sha256": base.sha256_file(config_path),
                "base_config_sha256": base.sha256_file(base_config_path),
                "selected_targets_sha256": base.sha256_file(target_path),
                "max_wall_seconds": args.max_wall_seconds,
                "run01_boundary": "preserved_invalid_no_threshold_or_lane_change",
            },
        )
        pre_import_gpu = base.gpu_snapshot()
        environment = base.verify_environment(base_config, pre_import_gpu)
        base.write_json_new(output_dir / "environment.json", environment)
        base.write_json_new(output_dir / "config_snapshot.json", config)
        targets, cells = v1.load_cells(target_path, config)
        target_workloads = v1.workload_rows(cells)
        model, tokenizer = base.load_model(base_config)
        donor_workloads = load_donor_workloads(config, repo_root, tokenizer)
        if {row["document_index"] for row in target_workloads}.intersection(
            {row["document_index"] for row in donor_workloads}
        ):
            raise ProtocolError("captured target and donor documents overlap")
        base.write_jsonl_new(output_dir / "target_workloads.jsonl", target_workloads)
        base.write_jsonl_new(output_dir / "companion_donor_workloads.jsonl", donor_workloads)
        base.write_jsonl_new(output_dir / "source_targets.jsonl", targets)
        first_ids = __import__("torch").tensor(
            [target_workloads[0]["window_token_ids"]],
            dtype=__import__("torch").long,
            device="cuda",
        )
        v1.observable.warmup_native_only(model, first_ids, base_config)
        captures, pools = v1.build_capture_and_pools(
            model,
            [*target_workloads, *donor_workloads],
            cells,
            base_config,
        )
        eligible, rejected = v1.prepare_cells(
            cells, captures, pools, config, base_config
        )
        base.write_jsonl_new(output_dir / "rejected_cells.jsonl", rejected)
        base.write_jsonl_new(
            output_dir / "eligible_cells.jsonl",
            (
                {
                    key: value
                    for key, value in row.items()
                    if not key.startswith("_") and key != "prior_sensitive_targets"
                }
                for row in eligible
            ),
        )
        result_rows: list[dict[str, Any]] = []
        result_path = output_dir / "cell_results.jsonl"
        with result_path.open("x", encoding="utf-8") as stream:
            for index, cell in enumerate(eligible):
                if time.time() - started > args.max_wall_seconds:
                    raise TimeoutError("V2 shape-lane pilot exceeded wall-time cap")
                row = v1.run_cell(model, index, cell, config, base_config)
                result_rows.append(row)
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                stream.flush()
        summary = v1.classify_results(result_rows, rejected, config)
        summary["completed_at"] = base.utc_now()
        summary["wall_seconds"] = time.time() - started
        summary["run01_invalid_reason_resolved_by"] = (
            "document_disjoint_512_token_companion_donors_only"
        )
        base.write_json_new(output_dir / "summary.json", summary)
        base.write_json_new(
            output_dir / "runtime_final.json", base.verify_final_runtime(base_config)
        )
        base.write_json_new(
            output_dir / "RUN_STATUS.json",
            {
                "status": "COMPLETE",
                "scientific_result_eligible": True,
                "paper_result": False,
                "verdict": summary["verdict"],
                "completed_at": base.utc_now(),
                "wall_seconds": time.time() - started,
            },
        )
        return 0
    except Exception as error:
        if not (output_dir / "RUN_STATUS.json").exists():
            base.write_json_new(
                output_dir / "RUN_STATUS.json",
                {
                    "status": "FAILED",
                    "scientific_result_eligible": False,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "failed_at": base.utc_now(),
                },
            )
        raise


if __name__ == "__main__":
    raise SystemExit(main())

