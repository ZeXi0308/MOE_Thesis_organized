#!/usr/bin/env python3
"""Run the frozen fresh-request StableBatch Selectability Decomposition Gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_observable_selector_pilot as observable  # noqa: E402
import run_oracle_action_sweep as oracle  # noqa: E402
import run_single_contribution_pilot as base  # noqa: E402
import selectability_policy as policy  # noqa: E402


ProtocolError = base.ProtocolError
RUNNER_RELATIVE = (
    "docs/ideas/stablebatch/experiments/run_selectability_decomposition_gate.py"
)
POLICY_RELATIVE = "docs/ideas/stablebatch/experiments/selectability_policy.py"
CONFIG_RELATIVE = (
    "docs/ideas/stablebatch/experiments/configs/"
    "selectability_decomposition_gate_v1.json"
)
LOCK_RELATIVE = (
    "docs/ideas/stablebatch/experiments/configs/"
    "FROZEN_SELECTABILITY_DECOMPOSITION_LOCK_V1.json"
)
LOCK_SCHEMA = "stablebatch-selectability-decomposition-frozen-lock-v1"


def expected_lock_files(config: Mapping[str, Any]) -> set[str]:
    return {
        RUNNER_RELATIVE,
        POLICY_RELATIVE,
        "docs/ideas/stablebatch/experiments/recompute_selectability_decomposition_gate.py",
        "docs/ideas/stablebatch/experiments/test_selectability_decomposition_gate.py",
        "docs/ideas/stablebatch/experiments/prepare_selectability_eval_manifest.py",
        "docs/ideas/stablebatch/experiments/run_single_contribution_pilot.py",
        "docs/ideas/stablebatch/experiments/run_observable_selector_pilot.py",
        "docs/ideas/stablebatch/experiments/run_oracle_action_sweep.py",
        "docs/ideas/semanticfence/experiments/prepare_eval_manifest.py",
        CONFIG_RELATIVE,
        str(config["data"]["manifest"]),
        str(config["data"]["provenance"]),
        "docs/ideas/stablebatch/experiments/data/selectability_fresh_eval_20260810_v1/artifact_hashes.json",
        str(config["calibration"]["cell_results"]),
        str(config["calibration"]["summary"]),
        str(config["calibration"]["manifest"]),
        str(config["calibration"]["document_manifest"]),
        "docs/ideas/stablebatch/experiments/configs/oracle_action_sweep_v1.json",
        "docs/ideas/stablebatch/experiments/configs/FROZEN_ORACLE_ACTION_SWEEP_LOCK_V1.json",
    }


def frozen_semantics(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        field: config[field]
        for field in ("selection", "action_space", "selectors", "gate")
    }


def validate_frozen_lock(lock: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    if lock.get("schema_version") != LOCK_SCHEMA:
        raise ProtocolError("wrong Selectability frozen-lock schema")
    if lock.get("status") != "FROZEN_PRE_RUN":
        raise ProtocolError("Selectability frozen lock is not pre-run")
    if lock.get("hypothesis_sha256") != hashlib.sha256(
        str(config["hypothesis"]).encode("utf-8")
    ).hexdigest():
        raise ProtocolError("Selectability hypothesis binding mismatch")
    if lock.get("frozen_semantics_sha256") != policy.content_sha256(
        frozen_semantics(config)
    ):
        raise ProtocolError("Selectability lock semantics binding mismatch")
    files = lock.get("files")
    if not isinstance(files, dict) or set(map(str, files)) != expected_lock_files(config):
        raise ProtocolError("Selectability lock does not bind the exact required file set")


def verify_calibration_and_freshness(
    config: Mapping[str, Any], repo_root: Path
) -> dict[str, Any]:
    calibration = config["calibration"]
    observed_calibration: dict[str, str] = {}
    for field in ("cell_results", "summary", "manifest", "document_manifest"):
        path = repo_root / str(calibration[field])
        observed = base.sha256_file(path)
        expected = str(calibration[f"{field}_sha256"])
        if observed != expected:
            raise ProtocolError(f"calibration {field} hash {observed} != {expected}")
        observed_calibration[field] = observed

    data = config["data"]
    provenance_path = repo_root / str(data["provenance"])
    if base.sha256_file(provenance_path) != str(data["provenance_sha256"]):
        raise ProtocolError("fresh data provenance hash mismatch")
    provenance = base.load_json(provenance_path)
    if provenance.get("status") != "PREPARED_NOT_EXECUTED":
        raise ProtocolError("fresh data provenance was not prepared pre-execution")
    if int(provenance["exclusions"]["selected_overlap_count"]) != 0:
        raise ProtocolError("fresh manifest overlaps its exclusion union")
    if int(provenance["exclusions"]["union_unique_hashes"]) != 1169:
        raise ProtocolError("fresh exclusion union cardinality drifted")
    if provenance["window"]["ordered_window_hash_digest"] != data[
        "ordered_window_hash_digest"
    ]:
        raise ProtocolError("fresh provenance window digest differs from config")

    heldout_manifest = base.load_jsonl(repo_root / str(data["manifest"]))
    calibration_manifest = base.load_jsonl(
        repo_root / str(calibration["document_manifest"])
    )
    heldout_text = {str(row["text_sha256"]) for row in heldout_manifest}
    calibration_text = {str(row["text_sha256"]) for row in calibration_manifest}
    if len(heldout_manifest) != 16 or len(heldout_text) != 16:
        raise ProtocolError("fresh held-out manifest is not 16 unique documents")
    overlap = heldout_text & calibration_text
    if overlap:
        raise ProtocolError(f"held-out/calibration document overlap: {sorted(overlap)}")

    calibration_rows = base.load_jsonl(
        repo_root / str(calibration["cell_results"])
    )
    if len(calibration_rows) != int(calibration["expected_cells"]):
        raise ProtocolError("calibration cell count drifted")
    calibration_windows = {
        str(row["window_token_ids_sha256"]) for row in calibration_rows
    }
    heldout_windows = {
        str(row["offset512_window_token_ids_sha256"])
        for row in heldout_manifest
    }
    if heldout_windows & calibration_windows:
        raise ProtocolError("held-out/calibration exact window overlap")
    return {
        "calibration_file_sha256": observed_calibration,
        "calibration_cells": len(calibration_rows),
        "heldout_documents": len(heldout_manifest),
        "heldout_unique_text_sha256": len(heldout_text),
        "heldout_vs_calibration_text_overlap": 0,
        "heldout_vs_calibration_window_overlap": 0,
        "global_exclusion_union_unique_hashes": int(
            provenance["exclusions"]["union_unique_hashes"]
        ),
        "fresh_provenance_sha256": base.sha256_file(provenance_path),
    }


def namespace_workloads(
    workloads: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> list[dict[str, Any]]:
    manifest_hash = str(config["data"]["manifest_sha256"])
    offset = int(config["data"]["token_offset"])
    result: list[dict[str, Any]] = []
    for row in workloads:
        value = dict(row)
        original = str(value["victim_id"])
        value["victim_id"] = f"sbsel-{manifest_hash[:8]}-{original}"
        value["document_text_sha256"] = str(value["text_sha256"])
        value["token_offset"] = offset
        value["source_manifest_sha256"] = manifest_hash
        result.append(value)
    return result


def augment_cells(
    cells: Sequence[Mapping[str, Any]],
    workloads: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    by_victim = {str(row["victim_id"]): row for row in workloads}
    manifest_hash = str(config["data"]["manifest_sha256"])
    result: list[dict[str, Any]] = []
    for raw in cells:
        row = dict(raw)
        workload = by_victim[str(row["victim_id"])]
        row["document_text_sha256"] = str(workload["document_text_sha256"])
        row["token_offset"] = int(config["data"]["token_offset"])
        row["window_tokens"] = int(config["data"]["window_tokens"])
        row["source_manifest_sha256"] = manifest_hash
        row["cell_identity"] = policy.selectability_cell_identity(row, manifest_hash)
        result.append(row)
    if len({row["cell_identity"] for row in result}) != len(result):
        raise ProtocolError("Selectability cell identities are not unique")
    return result


def sidecall_assignment(
    cell: Mapping[str, Any], policy_lock: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, Any]:
    identity = str(cell["cell_identity"])
    m1 = int(config["intervention"]["baseline_m"])
    m64 = int(config["intervention"]["treatment_m"])
    repeats = int(config["intervention"]["repeats_per_arm"])
    if repeats != 3:
        raise ProtocolError("Selectability v1 requires exactly three side-call repeats")
    schedule = [m1, m64, m64, m1, m1, m64]
    schedule_seed = str(config["intervention"]["sidecall_schedule_seed"])
    if int(hashlib.sha256(f"{schedule_seed}|{identity}".encode()).hexdigest()[-1], 16) % 2:
        schedule.reverse()

    selected_rank: dict[str, int | None] = {}
    for name in ("static", "online", "shuffle"):
        plan = policy_lock[f"{name}_plan"]
        selected = {
            str(row["cell_identity"]): int(row["rank"])
            for row in plan["selected"]
        }
        selected_rank[name] = selected.get(identity)
    return {
        "sidecall_m_order_per_rank": schedule,
        "shuffled_rank": selected_rank["shuffle"] if selected_rank["shuffle"] is not None else 0,
        "observable_rank": selected_rank["online"] if selected_rank["online"] is not None else 0,
        "preoutcome_policy_rank": selected_rank,
    }


def build_manifest(
    output_dir: Path, pending_status: Mapping[str, Any]
) -> dict[str, Any]:
    files: dict[str, dict[str, Any]] = {}
    for path in sorted(output_dir.iterdir()):
        if not path.is_file() or path.name in {"MANIFEST.json", "RUN_STATUS.json"}:
            continue
        files[path.name] = {
            "size_bytes": path.stat().st_size,
            "sha256": base.sha256_file(path),
        }
    status_bytes = oracle.json_artifact_bytes(pending_status)
    files["RUN_STATUS.json"] = {
        "size_bytes": len(status_bytes),
        "sha256": hashlib.sha256(status_bytes).hexdigest(),
    }
    return {
        "schema_version": "stablebatch-selectability-decomposition-manifest-v1",
        "created_at": base.utc_now(),
        "files": files,
    }


def verify_manifest(output_dir: Path) -> None:
    manifest = base.load_json(output_dir / "MANIFEST.json")
    if manifest.get("schema_version") != "stablebatch-selectability-decomposition-manifest-v1":
        raise ProtocolError("wrong Selectability manifest schema")
    expected = manifest.get("files")
    actual = {
        path.name
        for path in output_dir.iterdir()
        if path.is_file() and path.name != "MANIFEST.json"
    }
    if not isinstance(expected, dict) or actual != set(map(str, expected)):
        raise ProtocolError("Selectability manifest file set mismatch")
    for name, binding in expected.items():
        path = output_dir / str(name)
        if path.stat().st_size != int(binding["size_bytes"]):
            raise ProtocolError(f"Selectability manifest size mismatch for {name}")
        if base.sha256_file(path) != str(binding["sha256"]):
            raise ProtocolError(f"Selectability manifest hash mismatch for {name}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frozen-lock", type=Path, required=True)
    parser.add_argument("--max-wall-seconds", type=int, default=7200)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = args.repo_root.resolve()
    runner_path = Path(__file__).resolve()
    config_path = args.config.resolve()
    lock_path = args.frozen_lock.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise ProtocolError(f"refusing to reuse output directory {output_dir}")
    if str(runner_path.relative_to(repo_root)) != RUNNER_RELATIVE:
        raise ProtocolError("Selectability runner path differs from frozen path")
    if str(config_path.relative_to(repo_root)) != CONFIG_RELATIVE:
        raise ProtocolError("Selectability config path differs from frozen path")
    if str(lock_path.relative_to(repo_root)) != LOCK_RELATIVE:
        raise ProtocolError("Selectability lock path differs from frozen path")
    config = base.load_json(config_path)
    lock = base.load_json(lock_path)
    validate_frozen_lock(lock, config)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir()
    started = time.time()
    try:
        base.write_json_new(
            output_dir / "run_request.json",
            {
                "schema_version": "stablebatch-selectability-run-request-v1",
                "started_at": base.utc_now(),
                "argv": sys.argv,
                "pid": os.getpid(),
                "runner_sha256": base.sha256_file(runner_path),
                "policy_sha256": base.sha256_file(repo_root / POLICY_RELATIVE),
                "config_sha256": base.sha256_file(config_path),
                "lock_sha256": base.sha256_file(lock_path),
                "max_wall_seconds": int(args.max_wall_seconds),
                "git_head": base.command_output(["git", "-C", str(repo_root), "rev-parse", "HEAD"]),
                "git_status_short": base.command_output(["git", "-C", str(repo_root), "status", "--short"]),
            },
        )
        pre_import_gpu = base.gpu_snapshot()
        environment = base.verify_environment(config, pre_import_gpu)
        static = base.verify_static_inputs(
            config, repo_root, runner_path, config_path, lock_path
        )
        freshness = verify_calibration_and_freshness(config, repo_root)
        base.write_json_new(output_dir / "environment.json", environment)
        base.write_json_new(output_dir / "static_bindings.json", static)
        base.write_json_new(output_dir / "freshness_closure.json", freshness)
        base.write_json_new(output_dir / "config_snapshot.json", config)

        model, tokenizer = base.load_model(config)
        workloads = namespace_workloads(
            base.load_workloads(config, repo_root, tokenizer), config
        )
        workload_digest = observable.verify_workload_digest(workloads, config)
        base.write_jsonl_new(output_dir / "workloads.jsonl", workloads)
        torch = __import__("torch")
        first_ids = torch.tensor(
            [workloads[0]["window_token_ids"]], dtype=torch.long, device="cuda"
        )
        observable.warmup_native_only(model, first_ids, config)
        cells = augment_cells(
            observable.scan_observable_cells(model, workloads, config),
            workloads,
            config,
        )
        sorted_cells = sorted(cells, key=lambda row: str(row["cell_identity"]))
        base.write_jsonl_new(
            output_dir / "observable_cells.jsonl",
            (observable.public_cell(row) for row in sorted_cells),
        )

        result_path = output_dir / "cell_results.jsonl"
        if result_path.exists():
            raise ProtocolError("held-out outcome rows existed before selector seal")
        calibration_rows = base.load_jsonl(
            repo_root / str(config["calibration"]["cell_results"])
        )
        frozen_policy = policy.build_preoutcome_policy_lock(
            calibration_rows, sorted_cells, config
        )
        selector_lock = {
            **frozen_policy,
            "sealed_at": base.utc_now(),
            "hypothesis_sha256": hashlib.sha256(
                str(config["hypothesis"]).encode("utf-8")
            ).hexdigest(),
            "config_sha256": base.sha256_file(config_path),
            "frozen_run_lock_sha256": base.sha256_file(lock_path),
            "calibration_cell_results_sha256": base.sha256_file(
                repo_root / str(config["calibration"]["cell_results"])
            ),
            "heldout_manifest_sha256": str(config["data"]["manifest_sha256"]),
            "heldout_ordered_window_hash_digest": workload_digest,
            "observable_cells_sha256": base.sha256_file(
                output_dir / "observable_cells.jsonl"
            ),
            "result_path_existed_at_seal": result_path.exists(),
        }
        base.write_json_new(output_dir / "SELECTOR_LOCK.json", selector_lock)
        selector_lock_sha = base.sha256_file(output_dir / "SELECTOR_LOCK.json")
        if result_path.exists() or selector_lock["result_path_existed_at_seal"]:
            raise ProtocolError("selector seal ordering failed")

        rows: list[dict[str, Any]] = []
        with result_path.open("x", encoding="utf-8") as stream:
            for cell_index, cell in enumerate(sorted_cells):
                if time.time() - started > int(args.max_wall_seconds):
                    raise TimeoutError("Selectability Gate exceeded max wall time")
                assignment = sidecall_assignment(cell, frozen_policy, config)
                row = oracle.run_oracle_cell(
                    model, cell_index, cell, assignment, config, config
                )
                row["cell_identity"] = str(cell["cell_identity"])
                row["source_manifest_sha256"] = str(config["data"]["manifest_sha256"])
                row["document_text_sha256"] = str(cell["document_text_sha256"])
                row["token_offset"] = int(config["data"]["token_offset"])
                row["preoutcome_policy_rank"] = assignment["preoutcome_policy_rank"]
                rows.append(row)
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())

        classification = policy.classify_selectability(rows, frozen_policy, config)
        summary = {
            "schema_version": "stablebatch-selectability-decomposition-summary-v1",
            "status": "COMPLETE",
            "hypothesis": config["hypothesis"],
            "evaluation_type": config["evaluation_type"],
            "research_boundary": config["research_boundary"],
            "selector_lock_sha256": selector_lock_sha,
            **classification,
            "wall_seconds": time.time() - started,
            "completed_at": base.utc_now(),
        }
        base.write_json_new(output_dir / "summary.json", summary)
        base.write_json_new(
            output_dir / "runtime_final.json", base.verify_final_runtime(config)
        )
        status = {
            "status": "COMPLETE",
            "scientific_result_eligible": True,
            "verdict": summary["verdict"],
            "completed_at": base.utc_now(),
            "wall_seconds": time.time() - started,
        }
        base.write_json_new(output_dir / "MANIFEST.json", build_manifest(output_dir, status))
        oracle.write_bound_status(output_dir, status)
        verify_manifest(output_dir)
        return 0
    except BaseException as error:
        failure = {
            "status": "FAILED",
            "scientific_result_eligible": False,
            "error_type": type(error).__name__,
            "error": str(error),
            "failed_at": base.utc_now(),
            "wall_seconds": time.time() - started,
        }
        if not (output_dir / "FAILURE.json").exists():
            base.write_json_new(output_dir / "FAILURE.json", failure)
        if not (output_dir / "RUN_STATUS.json").exists():
            base.write_json_new(output_dir / "RUN_STATUS.json", failure)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
