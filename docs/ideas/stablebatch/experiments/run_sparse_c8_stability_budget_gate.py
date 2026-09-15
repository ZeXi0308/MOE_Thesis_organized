#!/usr/bin/env python3
"""Run the corrected global sparse-C8 StabilityBudget Gate.

The three commands intentionally separate information flow:

* ``broad`` completes C8 outcomes for every cell/rank in the old frozen cohort;
* ``seal`` scans the fresh cohort, fits the frozen ridge on broad C8 utility,
  selects one rank per cell and global exact-B cells, then writes a seal;
* ``fresh`` verifies that seal before producing any fresh C8 outcome.

This is a single-GPU research probe.  It is not a serving scheduler or a model
quality evaluation.
"""

from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Iterable, Mapping, Sequence


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_c8_action_transfer as c8  # noqa: E402
import run_observable_selector_pilot as observable  # noqa: E402
import run_oracle_action_sweep as oracle  # noqa: E402
import run_single_contribution_pilot as base  # noqa: E402
import sparse_c8_stability_budget_policy as policy  # noqa: E402


ProtocolError = base.ProtocolError
RUNNER_RELATIVE = (
    "docs/ideas/stablebatch/experiments/"
    "run_sparse_c8_stability_budget_gate.py"
)
POLICY_RELATIVE = (
    "docs/ideas/stablebatch/experiments/"
    "sparse_c8_stability_budget_policy.py"
)
TEST_RELATIVE = (
    "docs/ideas/stablebatch/experiments/"
    "test_sparse_c8_stability_budget_policy.py"
)
CONFIG_RELATIVE = (
    "docs/ideas/stablebatch/experiments/configs/"
    "sparse_c8_stability_budget_gate_v1.json"
)
LOCK_RELATIVE = (
    "docs/ideas/stablebatch/experiments/configs/"
    "FROZEN_SPARSE_C8_STABILITY_BUDGET_LOCK_V1.json"
)
LOCK_SCHEMA = "stablebatch-sparse-c8-stability-budget-lock-v1"


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def fraction_payload(value: Fraction | int) -> dict[str, Any]:
    item = value if isinstance(value, Fraction) else Fraction(value, 1)
    return {
        "numerator": int(item.numerator),
        "denominator": int(item.denominator),
        "value": float(item),
    }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return base.load_jsonl(path)


def write_json_new(path: Path, value: Any) -> None:
    base.write_json_new(path, value)


def write_jsonl_new(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())


def verify_bound_file(repo_root: Path, binding: Mapping[str, Any]) -> Path:
    path = (repo_root / str(binding["path"])).resolve()
    if not path.is_file():
        raise ProtocolError(f"bound input is absent: {path}")
    observed = base.sha256_file(path)
    if observed != str(binding["sha256"]):
        raise ProtocolError(
            f"bound input hash mismatch for {path}: {observed} != {binding['sha256']}"
        )
    return path


def verify_lock(
    repo_root: Path,
    runner_path: Path,
    config_path: Path,
    lock_path: Path,
    config: Mapping[str, Any],
) -> dict[str, str]:
    expected_paths = {
        "runner": RUNNER_RELATIVE,
        "policy": POLICY_RELATIVE,
        "test": TEST_RELATIVE,
        "config": CONFIG_RELATIVE,
        "lock": LOCK_RELATIVE,
    }
    actual_paths = {
        "runner": runner_path,
        "policy": HERE / Path(POLICY_RELATIVE).name,
        "test": HERE / Path(TEST_RELATIVE).name,
        "config": config_path,
        "lock": lock_path,
    }
    for role, path in actual_paths.items():
        if str(path.resolve().relative_to(repo_root)) != expected_paths[role]:
            raise ProtocolError(f"{role} path differs from frozen path")
    lock = base.load_json(lock_path)
    if lock.get("schema_version") != LOCK_SCHEMA:
        raise ProtocolError("wrong sparse-C8 lock schema")
    if lock.get("status") != "FROZEN_PRE_RUN":
        raise ProtocolError("sparse-C8 lock is not FROZEN_PRE_RUN")
    expected_semantics_hash = hashlib.sha256(
        base.canonical_json_bytes(frozen_semantics(config))
    ).hexdigest()
    if lock.get("frozen_semantics_sha256") != expected_semantics_hash:
        raise ProtocolError("sparse-C8 frozen semantics drifted")
    files = lock.get("files")
    if not isinstance(files, Mapping) or not files:
        raise ProtocolError("sparse-C8 lock has no file bindings")
    observed: dict[str, str] = {}
    for relative, expected in files.items():
        path = (repo_root / str(relative)).resolve()
        if not path.is_file():
            raise ProtocolError(f"locked file is absent: {relative}")
        digest = base.sha256_file(path)
        if digest != str(expected):
            raise ProtocolError(f"locked file hash mismatch: {relative}")
        observed[str(relative)] = digest
    return observed


def frozen_semantics(config: Mapping[str, Any]) -> dict[str, Any]:
    selection = config["selection"]
    features = config["features"]
    return {
        "gate_type": "GLOBAL_CELL_RANK_SELECTOR_GATE",
        "eligibility": config["shared_action_pre_eligibility"],
        "old_cell_count": int(config["cohorts"]["old_training"]["cell_count"]),
        "fresh_cell_count": int(config["cohorts"]["fresh_evaluation"]["cell_count"]),
        "candidate_ranks": list(map(int, config["action_space"]["candidate_ranks"])),
        "label": "c8_route_recovered_minus_route_harmed",
        "features": list(features["categorical_one_hot"])
        + list(features["continuous"]),
        "historical_outcome_derived_feature": False,
        "ridge_alpha": float(config["ridge"]["alpha"]),
        "budget": int(selection["budget_B"]),
        "max_actions_per_cell": int(selection["max_actions_per_cell"]),
        "within_cell_tie_break": str(
            selection["per_cell_rank_selection"]["tie_break"]
        ),
        "global_tie_break": list(
            selection["global_cell_selection"]["tie_break"]
        ),
        "random_baselines": config["random_baselines"],
        "oracles": config["oracles"],
        "headroom_formula": config["primary_metrics"]["rank_headroom_capture"],
    }


def old_bindings(config: Mapping[str, Any]) -> Mapping[str, Any]:
    return config["cohorts"]["old_training"]["bindings"]


def fresh_bindings(config: Mapping[str, Any]) -> Mapping[str, Any]:
    return config["cohorts"]["fresh_evaluation"]["bindings"]


def source_config(
    repo_root: Path, config: Mapping[str, Any]
) -> dict[str, Any]:
    path = verify_bound_file(repo_root, old_bindings(config)["source_config"])
    return base.load_json(path)


def phase_config(
    config: Mapping[str, Any], split: str, repo_root: Path
) -> dict[str, Any]:
    if split not in {"old", "fresh"}:
        raise ValueError(split)
    source = source_config(repo_root, config)
    eligibility = config["shared_action_pre_eligibility"]
    if split == "old":
        data = dict(source["data"])
    else:
        cohort = config["cohorts"]["fresh_evaluation"]
        bindings = fresh_bindings(config)
        data = {
            "manifest": str(bindings["manifest"]["path"]),
            "manifest_sha256": str(bindings["manifest"]["sha256"]),
            "document_indices": list(map(int, cohort["document_indices"])),
            "token_offset": int(eligibility["token_offset"]),
            "window_tokens": int(eligibility["window_tokens"]),
            "victim_position": int(eligibility["victim_position"]),
            "add_special_tokens": False,
            "ordered_window_hash_digest_method": (
                "sha256_of_concatenated_window_sha256_hex_in_document_indices_order"
            ),
            "ordered_window_hash_digest": str(
                bindings["ordered_window_hash_digest"]
            ),
        }
    return {
        "environment": source["environment"],
        "model": source["model"],
        "data": data,
        "selection": {
            "target_layers": eligibility["target_layers"],
            "cell_count": int(config["cohorts"][
                "old_training" if split == "old" else "fresh_evaluation"
            ]["cell_count"]),
        },
        "intervention": {
            "baseline_m": int(config["action_space"]["m_reference"]),
            "treatment_m": int(config["action_space"]["m_unprotected"]),
            "repeats_per_arm": int(
                config["execution"]["m1_m64_sidecall_repeats"]
            ),
            "sidecall_schedule_seed": str(
                config["execution"]["fresh_sidecall_schedule_seed"]
            ),
        },
    }


def validate_config(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != "stablebatch-sparse-c8-stability-budget-gate-v1":
        raise ProtocolError("wrong sparse-C8 config schema")
    if config.get("status") != "FROZEN_BEFORE_BROAD_TRAINING_C8_AND_FRESH_C8_OUTCOMES":
        raise ProtocolError("sparse-C8 config is not frozen")
    expected_layers = list(range(15))
    eligibility = config["shared_action_pre_eligibility"]
    if list(map(int, eligibility["target_layers"])) != expected_layers:
        raise ProtocolError("eligibility must contain all layers 0..14")
    if eligibility["filters"] or eligibility["cell_count"] != 240:
        raise ProtocolError("wrong action-pre eligibility predicate")
    if int(config["cohorts"]["old_training"]["cell_count"]) != 240 or int(
        config["cohorts"]["fresh_evaluation"]["cell_count"]
    ) != 240:
        raise ProtocolError("both cohorts must contain exactly 240 cells")
    if list(map(int, config["action_space"]["candidate_ranks"])) != list(range(8)):
        raise ProtocolError("candidate rank surface must be 0..7")
    if float(config["ridge"]["alpha"]) != 1.0:
        raise ProtocolError("ridge alpha is frozen to 1.0")
    if int(config["selection"]["budget_B"]) != 33:
        raise ProtocolError("selection budget is frozen to 33")
    if int(config["selection"]["max_actions_per_cell"]) != 1:
        raise ProtocolError("one-action-per-cell constraint is required")
    if bool(
        config["features"][
            "historical_outcome_derived_sensitivity_in_primary_features"
        ]
    ):
        raise ProtocolError("primary feature set must be outcome-naive")


def verify_cohort_disjointness(
    repo_root: Path, config: Mapping[str, Any]
) -> dict[str, Any]:
    old_manifest = verify_bound_file(
        repo_root, old_bindings(config)["source_manifest"]
    )
    fresh_manifest = verify_bound_file(
        repo_root, fresh_bindings(config)["manifest"]
    )
    old_rows = load_jsonl(old_manifest)
    fresh_rows = load_jsonl(fresh_manifest)
    old_indices = set(
        map(int, config["cohorts"]["old_training"]["document_indices"])
    )
    fresh_indices = set(
        map(int, config["cohorts"]["fresh_evaluation"]["document_indices"])
    )
    old_hashes = {
        str(row["text_sha256"])
        for row in old_rows
        if int(row["document_index"]) in old_indices
    }
    fresh_hashes = {
        str(row["text_sha256"])
        for row in fresh_rows
        if int(row["document_index"]) in fresh_indices
    }
    if len(old_hashes) != 16 or len(fresh_hashes) != 16:
        raise ProtocolError("both cohorts must contain 16 unique documents")
    overlap = sorted(old_hashes & fresh_hashes)
    if overlap:
        raise ProtocolError("fresh documents overlap old broad-training documents")
    return {
        "old_unique_document_hashes": len(old_hashes),
        "fresh_unique_document_hashes": len(fresh_hashes),
        "document_hash_overlap_count": len(overlap),
    }


def verify_model_and_data(
    repo_root: Path, config: Mapping[str, Any], split: str
) -> dict[str, Any]:
    runtime = phase_config(config, split, repo_root)
    model_root = Path(str(runtime["model"]["local_path"])).resolve()
    model_hashes: dict[str, str] = {}
    for relative, expected in runtime["model"]["file_sha256"].items():
        path = model_root / str(relative)
        if not path.is_file():
            raise ProtocolError(f"model file is absent: {path}")
        observed = base.sha256_file(path)
        if observed != str(expected):
            raise ProtocolError(f"model file drifted: {relative}")
        model_hashes[str(relative)] = observed
    data = runtime["data"]
    manifest = (repo_root / str(data["manifest"])).resolve()
    if base.sha256_file(manifest) != str(data["manifest_sha256"]):
        raise ProtocolError(f"{split} manifest hash mismatch")
    return {
        "model_path": str(model_root),
        "model_file_sha256": model_hashes,
        "manifest": str(manifest),
        "manifest_sha256": base.sha256_file(manifest),
    }


def cell_identity(row: Mapping[str, Any], split: str) -> str:
    return (
        f"{split}|text={row['document_text_sha256']}"
        f"|window={row['window_token_ids_sha256']}"
        f"|layer={int(row['layer']):02d}"
    )


def enrich_cells(
    cells: Sequence[Mapping[str, Any]],
    workloads: Sequence[Mapping[str, Any]],
    split: str,
) -> list[dict[str, Any]]:
    workload_by_victim = {str(row["victim_id"]): row for row in workloads}
    enriched: list[dict[str, Any]] = []
    for raw in cells:
        row = dict(raw)
        workload = workload_by_victim[str(row["victim_id"])]
        row["document_text_sha256"] = str(workload["text_sha256"])
        row["cell_identity"] = cell_identity(row, split)
        if len(row["expert_ids"]) != 8 or len(set(map(int, row["expert_ids"]))) != 8:
            raise ProtocolError("eligibility scan did not emit eight unique experts")
        if len(row["gate_weights"]) != 8 or any(
            not math.isfinite(float(value)) for value in row["gate_weights"]
        ):
            raise ProtocolError("eligibility scan emitted invalid gate weights")
        enriched.append(row)
    if len({str(row["cell_identity"]) for row in enriched}) != len(enriched):
        raise ProtocolError("cell identities are not unique")
    return sorted(enriched, key=lambda row: str(row["cell_identity"]))


def scan_split(
    model: Any,
    tokenizer: Any,
    repo_root: Path,
    config: Mapping[str, Any],
    split: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    runtime = phase_config(config, split, repo_root)
    workloads = base.load_workloads(runtime, repo_root, tokenizer)
    observable.verify_workload_digest(workloads, runtime)
    first_ids = __import__("torch").tensor(
        [workloads[0]["window_token_ids"]],
        dtype=__import__("torch").long,
        device="cuda",
    )
    observable.warmup_native_only(model, first_ids, runtime)
    cells = observable.scan_observable_cells(model, workloads, runtime)
    cells = enrich_cells(cells, workloads, split)
    cohort_key = "old_training" if split == "old" else "fresh_evaluation"
    if len(cells) != int(config["cohorts"][cohort_key]["cell_count"]):
        raise ProtocolError(f"{split} eligibility cell count mismatch")
    return workloads, cells


def deterministic_sidecall_assignment(
    identity: str, config: Mapping[str, Any]
) -> dict[str, Any]:
    m1 = int(config["action_space"]["m_reference"])
    m64 = int(config["action_space"]["m_unprotected"])
    schedule = [m1, m64, m64, m1, m1, m64]
    seed = str(config["execution"]["fresh_sidecall_schedule_seed"])
    if int(sha256_text(f"{seed}|{identity}")[-1], 16) % 2:
        schedule.reverse()
    return {"sidecall_m_order_per_rank": schedule}


def run_surface_cell(
    model: Any,
    index: int,
    cell: Mapping[str, Any],
    runtime_config: Mapping[str, Any],
    config: Mapping[str, Any],
    source_row: Mapping[str, Any] | None,
) -> dict[str, Any]:
    import torch

    ranks = list(range(8))
    m1 = int(config["action_space"]["m_reference"])
    m64 = int(config["action_space"]["m_unprotected"])
    assignment = (
        {"sidecall_m_order_per_rank": source_row["sidecall_m_order_per_rank"]}
        if source_row is not None
        else deterministic_sidecall_assignment(str(cell["cell_identity"]), config)
    )
    m_replacements, m_metadata = observable.precompute_cell_replacements(
        model, cell, assignment, runtime_config
    )
    c8_replacements, c8_metadata = c8.precompute_c8_replacements(
        model,
        cell,
        int(config["action_space"]["canonical_c"]),
        config["action_space"]["c8_context"],
        int(config["execution"]["c8_sidecall_repeats"]),
    )
    if source_row is not None:
        for rank in ranks:
            key = str(rank)
            previous = source_row["local_side_calls"]["ranks"][key]
            if m_metadata["ranks"][key]["m1_sha256"] != previous["m1_sha256"]:
                raise ProtocolError("old-cohort M1 side-call closure failed")
            if m_metadata["ranks"][key]["m64_sha256"] != previous["m64_sha256"]:
                raise ProtocolError("old-cohort M64 side-call closure failed")

    representative = base.PairIdentity(
        layer=int(cell["layer"]),
        flat_token_idx=int(cell["flat_token_idx"]),
        topk_rank=0,
        expert_id=int(cell["expert_ids"][0]),
    )
    input_ids = torch.tensor(
        [cell["window_token_ids"]], dtype=torch.long, device="cuda"
    )
    native = base.run_observation(model, input_ids, runtime_config, representative)
    with observable.patched_topk_contributions(model, cell, None, "self") as noop_trace:
        noop = base.run_observation(model, input_ids, runtime_config, representative)
    noop_checks = oracle.validate_noop(native, noop, noop_trace, cell, 8)

    r_map = {rank: m_replacements[rank][m1] for rank in ranks}
    u_map = {rank: m_replacements[rank][m64] for rank in ranks}
    r_hash = {rank: m_metadata["ranks"][str(rank)]["m1_sha256"] for rank in ranks}
    u_hash = {rank: m_metadata["ranks"][str(rank)]["m64_sha256"] for rank in ranks}
    surfaces: dict[str, tuple[dict[int, Any], dict[int, str], dict[int, int]]] = {
        "R": (r_map, r_hash, {rank: m1 for rank in ranks}),
        "U": (u_map, u_hash, {rank: m64 for rank in ranks}),
    }
    for rank in ranks:
        replacements = dict(u_map)
        hashes = dict(u_hash)
        replacements[rank] = c8_replacements[rank]
        hashes[rank] = c8_metadata["ranks"][str(rank)]["c8_sha256"]
        surfaces[f"C8-{rank}"] = (
            replacements,
            hashes,
            {item: (8 if item == rank else m64) for item in ranks},
        )
    labels = list(surfaces)
    order = oracle.deterministic_arm_order(
        str(cell["cell_identity"]),
        labels,
        str(config["execution"]["full_forward_arm_order_seed"]),
    )
    raw: dict[str, dict[str, Any]] = {}
    for label in order:
        replacements, hashes, descriptor = surfaces[label]
        raw[label] = c8.execute_replacement_arm(
            model,
            input_ids,
            cell,
            representative,
            runtime_config,
            replacements,
            hashes,
            descriptor,
            native,
            noop_trace,
        )
    if source_row is not None:
        c8.assert_source_arm_closure(raw["R"], source_row["reference_arm"], "R")
        c8.assert_source_arm_closure(raw["U"], source_row["unprotected_arm"], "U")

    reference_routes = raw["R"]["observation"]["topk_experts_by_layer"]
    reference_logits = raw["R"]["observation"]["_final_logits_cpu"]
    start_layer = int(cell["layer"]) + 1
    unprotected_changed = base.changed_membership_layers(
        reference_routes,
        raw["U"]["observation"]["topk_experts_by_layer"],
        start_layer,
    )
    if source_row is not None and unprotected_changed != list(
        source_row["unprotected_changed_layers_vs_R"]
    ):
        raise ProtocolError("old-cohort U route closure failed")
    reference = c8.public_arm(
        raw["R"], reference_logits, reference_routes, unprotected_changed, start_layer
    )
    unprotected = c8.public_arm(
        raw["U"], reference_logits, reference_routes, unprotected_changed, start_layer
    )
    actions: dict[str, dict[str, Any]] = {}
    for rank in ranks:
        action = c8.public_arm(
            raw[f"C8-{rank}"],
            reference_logits,
            reference_routes,
            unprotected_changed,
            start_layer,
        )
        action = c8.add_final_decomposition(unprotected, action)
        action.update(
            {
                "rank": rank,
                "expert_id": int(cell["expert_ids"][rank]),
                "utility": int(action["route_net_reward"]),
            }
        )
        actions[str(rank)] = action
    return {
        "schema_version": "stablebatch-sparse-c8-surface-cell-v1",
        "cell_index": index,
        **observable.public_cell(cell),
        "arm_order": order,
        "m1_m64_side_calls": m_metadata,
        "c8_side_calls": c8_metadata,
        "native_noop_checks": noop_checks,
        "reference_arm": reference,
        "unprotected_arm": unprotected,
        "c8_actions": actions,
        "integrity_status": "PASS",
    }


def surface_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(rows) != 240 or len({str(row["cell_identity"]) for row in rows}) != 240:
        raise ProtocolError("surface summary requires 240 unique cells")
    actions = [row["c8_actions"][str(rank)] for row in rows for rank in range(8)]
    best = [
        min(
            row["c8_actions"].values(),
            key=lambda item: (
                -int(item["route_net_reward"]),
                int(item["route_harmed_count"]),
                int(item["rank"]),
            ),
        )
        for row in rows
    ]
    return {
        "cell_count": len(rows),
        "candidate_action_count": len(actions),
        "document_count": len({str(row["document_text_sha256"]) for row in rows}),
        "opportunity_cell_count": sum(
            int(int(row["unprotected_arm"]["distance_vs_R"]) > 0) for row in rows
        ),
        "no_opportunity_cell_count": sum(
            int(int(row["unprotected_arm"]["distance_vs_R"]) == 0) for row in rows
        ),
        "positive_action_count": sum(int(int(row["route_net_reward"]) > 0) for row in actions),
        "zero_action_count": sum(int(int(row["route_net_reward"]) == 0) for row in actions),
        "negative_action_count": sum(int(int(row["route_net_reward"]) < 0) for row in actions),
        "positive_best_cell_count": sum(int(int(row["route_net_reward"]) > 0) for row in best),
        "zero_best_cell_count": sum(int(int(row["route_net_reward"]) == 0) for row in best),
        "negative_best_cell_count": sum(int(int(row["route_net_reward"]) < 0) for row in best),
        "total_recovered": sum(int(row["route_recovered_count"]) for row in actions),
        "total_harmed": sum(int(row["route_harmed_count"]) for row in actions),
        "total_net": sum(int(row["route_net_reward"]) for row in actions),
    }


def build_manifest(directory: Path, names: Sequence[str], schema: str) -> dict[str, Any]:
    return {
        "schema_version": schema,
        "files": {
            name: {
                "size_bytes": (directory / name).stat().st_size,
                "sha256": base.sha256_file(directory / name),
            }
            for name in names
        },
    }


def verify_manifest(directory: Path, name: str = "MANIFEST.json") -> dict[str, Any]:
    manifest = base.load_json(directory / name)
    files = manifest.get("files")
    if not isinstance(files, Mapping) or not files:
        raise ProtocolError("output manifest has no files")
    for relative, binding in files.items():
        path = directory / str(relative)
        if not path.is_file():
            raise ProtocolError(f"manifest file absent: {relative}")
        if path.stat().st_size != int(binding["size_bytes"]):
            raise ProtocolError(f"manifest size mismatch: {relative}")
        if base.sha256_file(path) != str(binding["sha256"]):
            raise ProtocolError(f"manifest hash mismatch: {relative}")
    return manifest


def run_broad(
    repo_root: Path,
    output_root: Path,
    config: Mapping[str, Any],
    lock_hashes: Mapping[str, str],
    max_wall_seconds: int,
) -> None:
    broad_dir = output_root / "broad"
    if broad_dir.exists():
        raise ProtocolError(f"refusing to reuse broad output {broad_dir}")
    broad_dir.mkdir(parents=True)
    started = time.time()
    runtime = phase_config(config, "old", repo_root)
    source_ledger_path = verify_bound_file(
        repo_root,
        old_bindings(config)["m1_action_ledger_for_R_U_execution_closure_only"],
    )
    source_observable_path = verify_bound_file(
        repo_root, old_bindings(config)["action_pre_cells"]
    )
    source_rows = load_jsonl(source_ledger_path)
    if len(source_rows) != 240:
        raise ProtocolError("old oracle ledger must have all 240 cells")
    source_by_key = {c8.cell_key(row): row for row in source_rows}
    write_json_new(
        broad_dir / "run_request.json",
        {
            "schema_version": "stablebatch-sparse-c8-broad-request-v1",
            "started_at": base.utc_now(),
            "mode": "broad",
            "locked_file_sha256": dict(lock_hashes),
            "result_rows_existed_at_start": False,
        },
    )
    write_json_new(broad_dir / "environment.json", base.verify_environment(runtime, base.gpu_snapshot()))
    write_json_new(broad_dir / "static_bindings.json", verify_model_and_data(repo_root, config, "old"))
    model, tokenizer = base.load_model(runtime)
    workloads, cells = scan_split(model, tokenizer, repo_root, config, "old")
    write_jsonl_new(broad_dir / "workloads.jsonl", workloads)
    public_cells = [observable.public_cell(row) for row in cells]
    write_jsonl_new(broad_dir / "observable_cells.jsonl", public_cells)
    # The old scan is bound byte-for-byte except for the new identity/text fields.
    old_public = load_jsonl(source_observable_path)
    old_lookup = {c8.cell_key(row): row for row in old_public}
    for row in public_cells:
        previous = old_lookup.get(c8.cell_key(row))
        if previous is None:
            raise ProtocolError("old observable scan omitted a frozen cell")
        for field in (
            "victim_id",
            "document_index",
            "window_token_ids",
            "window_token_ids_sha256",
            "layer",
            "flat_token_idx",
            "target_hidden_sha256",
            "target_router_logits_sha256",
            "current_layer_topk_cutoff_margin",
            "gate_weights",
            "expert_ids",
        ):
            if row[field] != previous[field]:
                raise ProtocolError(f"old observable closure failed for {field}")
    result_path = broad_dir / "cell_results.jsonl"
    write_json_new(
        broad_dir / "BROAD_COHORT_LOCK.json",
        {
            "schema_version": "stablebatch-sparse-c8-broad-cohort-lock-v1",
            "status": "SEALED_BEFORE_BROAD_C8_OUTCOMES",
            "sealed_at": base.utc_now(),
            "eligibility": config["shared_action_pre_eligibility"],
            "observable_cells_sha256": base.sha256_file(broad_dir / "observable_cells.jsonl"),
            "ordered_cell_identities_sha256": sha256_text(
                "".join(str(row["cell_identity"]) for row in public_cells)
            ),
            "cell_count": len(cells),
            "candidate_action_count": len(cells) * 8,
            "result_rows_existed_at_seal": result_path.exists(),
        },
    )
    if result_path.exists():
        raise ProtocolError("broad outcomes existed before cohort seal")
    rows: list[dict[str, Any]] = []
    with result_path.open("x", encoding="utf-8") as stream:
        for index, cell in enumerate(cells):
            if time.time() - started > max_wall_seconds:
                raise TimeoutError("broad sparse-C8 run exceeded wall limit")
            source = source_by_key.get(c8.cell_key(cell))
            if source is None:
                raise ProtocolError("old source ledger omitted a cell")
            row = run_surface_cell(model, index, cell, runtime, config, source)
            rows.append(row)
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
    summary = {
        "schema_version": "stablebatch-sparse-c8-broad-summary-v1",
        "status": "COMPLETE",
        "eligibility": config["shared_action_pre_eligibility"],
        "metrics": surface_summary(rows),
        "wall_seconds": time.time() - started,
        "completed_at": base.utc_now(),
    }
    write_json_new(broad_dir / "summary.json", summary)
    write_json_new(broad_dir / "runtime_final.json", base.verify_final_runtime(runtime))
    names = [
        "run_request.json",
        "environment.json",
        "static_bindings.json",
        "workloads.jsonl",
        "observable_cells.jsonl",
        "BROAD_COHORT_LOCK.json",
        "cell_results.jsonl",
        "summary.json",
        "runtime_final.json",
    ]
    write_json_new(
        broad_dir / "MANIFEST.json",
        build_manifest(broad_dir, names, "stablebatch-sparse-c8-broad-manifest-v1"),
    )
    write_json_new(
        broad_dir / "RUN_STATUS.json",
        {
            "status": "COMPLETE",
            "scientific_result_eligible_as_training_ledger": True,
            "completed_at": base.utc_now(),
        },
    )
    verify_manifest(broad_dir)


def policy_candidates_from_cells(cells: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return policy.flatten_preaction_cells(cells, top_k=8)


def training_candidates(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return policy.flatten_outcome_cells(policy_outcome_cells(rows), top_k=8)


def policy_outcome_cells(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for row in rows:
        actions: dict[str, Any] = {}
        for rank in range(8):
            source = row["c8_actions"][str(rank)]
            recovered = int(source["route_recovered_count"])
            harmed = int(source["route_harmed_count"])
            actions[str(rank)] = {
                "recovered": recovered,
                "harmed": harmed,
                "net": recovered - harmed,
            }
        converted.append(
            {
                "cell_identity": str(row["cell_identity"]),
                "document_id": str(row["document_text_sha256"]),
                "victim_id": str(row["victim_id"]),
                "document_index": int(row["document_index"]),
                "document_text_sha256": str(row["document_text_sha256"]),
                "layer": int(row["layer"]),
                "expert_ids": list(map(int, row["expert_ids"])),
                "gate_weights": list(map(float, row["gate_weights"])),
                "current_layer_topk_cutoff_margin": float(
                    row["current_layer_topk_cutoff_margin"]
                ),
                "actions": actions,
            }
        )
    return converted


def run_seal(
    repo_root: Path,
    output_root: Path,
    config: Mapping[str, Any],
    lock_hashes: Mapping[str, str],
) -> None:
    broad_dir = output_root / "broad"
    verify_manifest(broad_dir)
    broad_rows = load_jsonl(broad_dir / "cell_results.jsonl")
    if surface_summary(broad_rows) != base.load_json(broad_dir / "summary.json")["metrics"]:
        raise ProtocolError("broad summary does not recompute")
    fresh_dir = output_root / "fresh"
    if fresh_dir.exists():
        raise ProtocolError(f"refusing to reuse fresh seal directory {fresh_dir}")
    fresh_dir.mkdir(parents=True)
    runtime = phase_config(config, "fresh", repo_root)
    write_json_new(fresh_dir / "environment.json", base.verify_environment(runtime, base.gpu_snapshot()))
    write_json_new(fresh_dir / "static_bindings.json", verify_model_and_data(repo_root, config, "fresh"))
    model, tokenizer = base.load_model(runtime)
    workloads, cells = scan_split(model, tokenizer, repo_root, config, "fresh")
    write_jsonl_new(fresh_dir / "workloads.jsonl", workloads)
    write_jsonl_new(
        fresh_dir / "observable_cells.jsonl",
        (observable.public_cell(row) for row in cells),
    )
    train = training_candidates(broad_rows)
    fresh = policy_candidates_from_cells(cells)
    ridge_model = policy.fit_action_pre_ridge(
        train,
        num_layers=15,
        num_experts=64,
        top_k=8,
    )
    if float(ridge_model["alpha"]) != float(config["ridge"]["alpha"]):
        raise ProtocolError("fitted ridge alpha differs from frozen config")
    scores = policy.predict_action_scores(fresh, ridge_model)
    plan = policy.select_global_exact_b(
        scores,
        budget=int(config["selection"]["budget_B"]),
        top_k=8,
    )
    write_json_new(fresh_dir / "ridge_model.json", ridge_model)
    write_json_new(fresh_dir / "selection_plan.json", plan)
    outcome_paths = [fresh_dir / "cell_results.jsonl", fresh_dir / "summary.json"]
    if any(path.exists() for path in outcome_paths):
        raise ProtocolError("fresh outcomes existed before selection seal")
    seal_body = {
        "schema_version": "stablebatch-sparse-c8-selection-seal-v1",
        "status": "SEALED_BEFORE_FRESH_C8_OUTCOMES",
        "sealed_at": base.utc_now(),
        "gate_type": "GLOBAL_CELL_RANK_SELECTOR_GATE",
        "eligibility": config["shared_action_pre_eligibility"],
        "feature_families": list(config["features"]["categorical_one_hot"])
        + list(config["features"]["continuous"]),
        "historical_outcome_derived_feature": False,
        "claim_term": "outcome_naive_action_pre_rank_utility_selection",
        "ridge_alpha": float(config["ridge"]["alpha"]),
        "budget": int(config["selection"]["budget_B"]),
        "max_actions_per_cell": 1,
        "broad_cell_results_sha256": base.sha256_file(broad_dir / "cell_results.jsonl"),
        "broad_manifest_sha256": base.sha256_file(broad_dir / "MANIFEST.json"),
        "fresh_manifest_sha256": fresh_bindings(config)["manifest"]["sha256"],
        "fresh_observable_cells_sha256": base.sha256_file(fresh_dir / "observable_cells.jsonl"),
        "ridge_model_sha256": base.sha256_file(fresh_dir / "ridge_model.json"),
        "selection_plan_sha256": base.sha256_file(fresh_dir / "selection_plan.json"),
        "training_cell_count": len(broad_rows),
        "training_action_count": len(train),
        "fresh_cell_count": len(cells),
        "fresh_candidate_action_count": len(fresh),
        "selected_action_count": len(plan["selected"]),
        "selected_unique_cell_count": len(
            {str(row["cell_identity"]) for row in plan["selected"]}
        ),
        "fresh_outcome_paths_existed_at_seal": {
            path.name: path.exists() for path in outcome_paths
        },
        "locked_file_sha256": dict(lock_hashes),
    }
    seal_body["deterministic_content_sha256"] = hashlib.sha256(
        base.canonical_json_bytes(
            {key: value for key, value in seal_body.items() if key != "sealed_at"}
        )
    ).hexdigest()
    write_json_new(fresh_dir / "SELECTION_SEAL.json", seal_body)
    names = [
        "environment.json",
        "static_bindings.json",
        "workloads.jsonl",
        "observable_cells.jsonl",
        "ridge_model.json",
        "selection_plan.json",
        "SELECTION_SEAL.json",
    ]
    write_json_new(
        fresh_dir / "PREOUTCOME_MANIFEST.json",
        build_manifest(
            fresh_dir, names, "stablebatch-sparse-c8-preoutcome-manifest-v1"
        ),
    )
    write_json_new(
        fresh_dir / "SEAL_STATUS.json",
        {
            "status": "COMPLETE_PREOUTCOME_SEAL",
            "scientific_result_eligible": False,
            "fresh_outcomes_opened": False,
            "completed_at": base.utc_now(),
        },
    )
    verify_manifest(fresh_dir, "PREOUTCOME_MANIFEST.json")


def verify_selection_seal(
    output_root: Path,
    config: Mapping[str, Any],
    current_lock_hashes: Mapping[str, str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    broad_dir = output_root / "broad"
    fresh_dir = output_root / "fresh"
    verify_manifest(broad_dir)
    verify_manifest(fresh_dir, "PREOUTCOME_MANIFEST.json")
    seal = base.load_json(fresh_dir / "SELECTION_SEAL.json")
    if seal.get("status") != "SEALED_BEFORE_FRESH_C8_OUTCOMES":
        raise ProtocolError("fresh selection is not sealed")
    if any(bool(value) for value in seal["fresh_outcome_paths_existed_at_seal"].values()):
        raise ProtocolError("selection seal records pre-existing fresh outcomes")
    checks = {
        "broad_cell_results_sha256": broad_dir / "cell_results.jsonl",
        "broad_manifest_sha256": broad_dir / "MANIFEST.json",
        "fresh_observable_cells_sha256": fresh_dir / "observable_cells.jsonl",
        "ridge_model_sha256": fresh_dir / "ridge_model.json",
        "selection_plan_sha256": fresh_dir / "selection_plan.json",
    }
    for field, path in checks.items():
        if base.sha256_file(path) != str(seal[field]):
            raise ProtocolError(f"selection seal hash mismatch: {field}")
    if int(seal["budget"]) != int(config["selection"]["budget_B"]):
        raise ProtocolError("selection seal budget drifted")
    if seal.get("locked_file_sha256") != dict(current_lock_hashes):
        raise ProtocolError("locked execution identity differs from selection seal")
    plan = base.load_json(fresh_dir / "selection_plan.json")
    if len(plan["selected"]) != 33 or len(
        {str(row["cell_identity"]) for row in plan["selected"]}
    ) != 33:
        raise ProtocolError("selection plan violates exact unique-cell budget")
    return seal, plan


def classify(
    metrics: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, Any]:
    exact_oracle = metrics["oracle_exact_B"]
    at_most_oracle = metrics["oracle_at_most_B"]
    selector = metrics["selector"]
    global_random = metrics["global_matched_random"]
    thresholds = config["classification"]["fixed_thresholds"]
    rank_gain_payload = metrics["decomposition"]["rank_selection_gain"]
    rank_gain = Fraction(
        int(rank_gain_payload["numerator"]),
        int(rank_gain_payload["denominator"]),
    )
    action_generalizes = (
        int(exact_oracle["net"]) > 0
        and Fraction(int(exact_oracle["net"]), 1)
        > Fraction(
            int(global_random["net"]["numerator"]),
            int(global_random["net"]["denominator"]),
        )
        and int(exact_oracle["recovered"]) > int(exact_oracle["harmed"])
        and int(at_most_oracle["net"]) > 0
        and int(metrics["oracle_positive_net_documents"])
        >= int(thresholds["min_oracle_positive_net_documents_for_action_generalization"])
    )
    selector_net = Fraction(int(selector["net"]), 1)
    selector_recovered = Fraction(int(selector["recovered"]), 1)
    selector_harmed = Fraction(int(selector["harmed"]), 1)
    selector_beats_global = selector_net > _payload_fraction(
        global_random["net"]
    )
    capture_payload = metrics["decomposition"]["rank_headroom_capture"]
    capture = (
        _payload_fraction(capture_payload) if capture_payload is not None else None
    )
    strong_rank_signal = (
        rank_gain > 0
        and capture is not None
        and capture
        >= Fraction(
            str(thresholds["min_rank_headroom_capture_for_strong_selector"])
        )
    )
    selector_cross_document = (
        int(metrics["selector_positive_documents"])
        >= int(thresholds["min_selector_positive_net_documents_for_strong_selector"])
        and int(metrics["selector_gt_cell_matched_documents"])
        >= int(
            thresholds[
                "min_selector_above_cell_matched_random_documents_for_strong_selector"
            ]
        )
    )
    harm_dominates = (
        int(exact_oracle["recovered"]) > 0 and int(exact_oracle["net"]) <= 0
    ) or (selector_recovered > 0 and selector_net <= 0)
    if (
        harm_dominates
    ):
        case = "HARM_DOMINATES"
    elif not action_generalizes:
        case = "ACTION_GENERALIZATION_FAIL"
    elif (
        strong_rank_signal
        and selector_beats_global
        and selector_net > 0
        and selector_recovered > selector_harmed
        and selector_cross_document
    ):
        case = "SELECTOR_AND_ACTION_GENERALIZE"
    elif rank_gain > 0:
        case = "SELECTOR_WEAKLY_ABOVE_RANDOM"
    else:
        case = "ACTION_GENERALIZES_SELECTOR_WEAK"
    implication = {
        "SELECTOR_AND_ACTION_GENERALIZE": "GO_STABILITYBUDGET_PROTOTYPE",
        "ACTION_GENERALIZES_SELECTOR_WEAK": "KEEP_SHAPELANE_CHANGE_POLICY_ABSTRACTION",
        "SELECTOR_WEAKLY_ABOVE_RANDOM": "PROFILE_ONCE_SERVE_MANY",
        "ACTION_GENERALIZATION_FAIL": "WEAKEN_FIXED_C8_PRIMITIVE",
        "HARM_DOMINATES": "REDEFINE_RISK_AWARE_OBJECTIVE_ON_NEW_DATA",
    }[case]
    return {
        "gate_classification": case,
        "system_implication": implication,
        "action_generalizes": action_generalizes,
        "rank_specific_selector_signal": bool(rank_gain > 0),
        "strong_rank_signal": strong_rank_signal,
        "selector_harm_safe": bool(
            selector_net > 0 and selector_recovered > selector_harmed
        ),
        "selector_beats_global_random": selector_beats_global,
        "selector_cross_document": selector_cross_document,
        "harm_dominates": harm_dominates,
    }


def _payload_fraction(value: Mapping[str, Any]) -> Fraction:
    return Fraction(int(value["numerator"]), int(value["denominator"]))


def _median_fraction(values: Sequence[Fraction]) -> Fraction:
    if not values:
        return Fraction(0)
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def augment_metrics(
    metrics: dict[str, Any],
    outcome_actions: Sequence[Mapping[str, Any]],
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    lookup = {
        (str(row["cell_identity"]), int(row["rank"])): row
        for row in outcome_actions
    }
    selected = [
        lookup[(str(row["cell_identity"]), int(row["rank"]))]
        for row in plan["selected"]
    ]
    rank_counts = {
        str(rank): sum(int(int(row["rank"]) == rank) for row in selected)
        for rank in range(8)
    }
    selected_documents = sorted({str(row["document_id"]) for row in selected})
    per_document = metrics["per_document"]
    gaps: list[Fraction] = []
    selector_positive = 0
    selector_above_cell = 0
    selector_above_global = 0
    oracle_positive = 0
    for item in per_document.values():
        selector_net = Fraction(int(item["selector"]["net"]), 1)
        cell_net = _payload_fraction(item["cell_matched_random_rank"]["net"])
        global_net = _payload_fraction(item["global_matched_random"]["net"])
        gaps.append(selector_net - cell_net)
        selector_positive += int(selector_net > 0)
        selector_above_cell += int(selector_net > cell_net)
        selector_above_global += int(selector_net > global_net)
        oracle_positive += int(int(item["oracle_at_most_B"]["net"]) > 0)
    oracle_gap = Fraction(
        int(metrics["oracle_exact_B"]["net"]) - int(metrics["selector"]["net"]),
        1,
    )
    metrics.update(
        {
            "selected_positive_action_count": sum(
                int(int(row["net"]) > 0) for row in selected
            ),
            "selected_zero_action_count": sum(
                int(int(row["net"]) == 0) for row in selected
            ),
            "selected_negative_action_count": sum(
                int(int(row["net"]) < 0) for row in selected
            ),
            "selected_document_count": len(selected_documents),
            "selected_documents": selected_documents,
            "selected_rank_distribution": rank_counts,
            "selector_positive_documents": selector_positive,
            "selector_gt_cell_matched_documents": selector_above_cell,
            "selector_gt_global_random_documents": selector_above_global,
            "oracle_positive_net_documents": oracle_positive,
            "median_document_rank_gap": fraction_payload(_median_fraction(gaps)),
            "oracle_gap": fraction_payload(oracle_gap),
        }
    )
    return metrics


def run_fresh(
    repo_root: Path,
    output_root: Path,
    config: Mapping[str, Any],
    lock_hashes: Mapping[str, str],
    max_wall_seconds: int,
) -> None:
    seal, plan = verify_selection_seal(output_root, config, lock_hashes)
    fresh_dir = output_root / "fresh"
    result_path = fresh_dir / "cell_results.jsonl"
    summary_path = fresh_dir / "summary.json"
    if result_path.exists() or summary_path.exists():
        raise ProtocolError("refusing to reuse fresh outcome files")
    started = time.time()
    runtime = phase_config(config, "fresh", repo_root)
    base.verify_environment(runtime, base.gpu_snapshot())
    verify_model_and_data(repo_root, config, "fresh")
    model, tokenizer = base.load_model(runtime)
    workloads, cells = scan_split(model, tokenizer, repo_root, config, "fresh")
    scanned_public = [observable.public_cell(row) for row in cells]
    sealed_public = load_jsonl(fresh_dir / "observable_cells.jsonl")
    if scanned_public != sealed_public:
        raise ProtocolError("fresh action-pre cells differ from selection seal")
    rows: list[dict[str, Any]] = []
    with result_path.open("x", encoding="utf-8") as stream:
        for index, cell in enumerate(cells):
            if time.time() - started > max_wall_seconds:
                raise TimeoutError("fresh sparse-C8 run exceeded wall limit")
            row = run_surface_cell(model, index, cell, runtime, config, None)
            rows.append(row)
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
    outcome_actions = policy.flatten_outcome_cells(
        policy_outcome_cells(rows), top_k=8
    )
    metrics = policy.evaluate_budget_decomposition(
        outcome_actions,
        plan,
        budget=int(config["selection"]["budget_B"]),
        top_k=8,
    )
    metrics = augment_metrics(metrics, outcome_actions, plan)
    decision = classify(metrics, config)
    summary = {
        "schema_version": "stablebatch-sparse-c8-stability-budget-summary-v1",
        "status": "COMPLETE",
        "evaluation_type": "global_outcome_naive_sparse_c8_cell_rank_selection",
        "research_boundary": config["research_boundary"],
        "selection_seal_sha256": base.sha256_file(fresh_dir / "SELECTION_SEAL.json"),
        "selection_sealed_before_outcomes": True,
        "training_surface": surface_summary(
            load_jsonl(output_root / "broad" / "cell_results.jsonl")
        ),
        "fresh_surface": surface_summary(rows),
        "metrics": metrics,
        "decision": decision,
        "wall_seconds": time.time() - started,
        "completed_at": base.utc_now(),
    }
    write_json_new(summary_path, summary)
    write_json_new(fresh_dir / "runtime_final.json", base.verify_final_runtime(runtime))
    write_json_new(
        fresh_dir / "RUN_STATUS.json",
        {
            "status": "COMPLETE",
            "scientific_result_eligible": True,
            **decision,
            "completed_at": base.utc_now(),
        },
    )
    names = [
        "PREOUTCOME_MANIFEST.json",
        "SELECTION_SEAL.json",
        "ridge_model.json",
        "selection_plan.json",
        "observable_cells.jsonl",
        "workloads.jsonl",
        "cell_results.jsonl",
        "summary.json",
        "runtime_final.json",
        "RUN_STATUS.json",
    ]
    write_json_new(
        fresh_dir / "MANIFEST.json",
        build_manifest(fresh_dir, names, "stablebatch-sparse-c8-final-manifest-v1"),
    )
    verify_manifest(fresh_dir)
    if base.sha256_file(fresh_dir / "SELECTION_SEAL.json") != str(
        summary["selection_seal_sha256"]
    ):
        raise ProtocolError("selection seal changed after outcomes")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("broad", "seal", "fresh"))
    parser.add_argument("--repo-root", type=Path, default=HERE.parents[3])
    parser.add_argument(
        "--config",
        type=Path,
        default=HERE / "configs/sparse_c8_stability_budget_gate_v1.json",
    )
    parser.add_argument(
        "--frozen-lock",
        type=Path,
        default=HERE / "configs/FROZEN_SPARSE_C8_STABILITY_BUDGET_LOCK_V1.json",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-wall-seconds", type=int, default=5400)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = args.repo_root.resolve()
    runner_path = Path(__file__).resolve()
    config_path = args.config.resolve()
    lock_path = args.frozen_lock.resolve()
    output_root = args.output_root.resolve()
    config = base.load_json(config_path)
    validate_config(config)
    lock_hashes = verify_lock(
        repo_root, runner_path, config_path, lock_path, config
    )
    verify_bound_file(repo_root, old_bindings(config)["source_config"])
    verify_bound_file(repo_root, old_bindings(config)["action_pre_cells"])
    verify_bound_file(
        repo_root,
        old_bindings(config)["m1_action_ledger_for_R_U_execution_closure_only"],
    )
    verify_bound_file(repo_root, fresh_bindings(config)["provenance"])
    verify_cohort_disjointness(repo_root, config)
    output_root.mkdir(parents=True, exist_ok=True)
    if args.mode == "broad":
        run_broad(
            repo_root, output_root, config, lock_hashes, int(args.max_wall_seconds)
        )
    elif args.mode == "seal":
        run_seal(repo_root, output_root, config, lock_hashes)
    else:
        run_fresh(
            repo_root,
            output_root,
            config,
            lock_hashes,
            int(args.max_wall_seconds),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
