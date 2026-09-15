#!/usr/bin/env python3
"""Replay the Stage-1 focal cohort with each original M=2 pair slot-swapped.

This is a calibration-only causal probe.  Pair identity is fixed to run03;
only row order changes.  ``COMPLETE.json`` is written last.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Mapping, Sequence


SCHEMA = "semanticfence-slot-permutation-config-v1"
SCHEDULE_SCHEMA = "semanticfence-slot-permutation-schedule-v1"
PLAN_SCHEMA = "semanticfence-slot-permutation-plan-v1"
LOCK_SCHEMA = "semanticfence-slot-permutation-lock-v1"
NUMERIC_SCHEMA = "semanticfence-slot-permutation-numeric-v1"
RESULT_SCHEMA = "semanticfence-slot-permutation-result-v1"
EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT_DEFAULT = EXPERIMENT_DIR.parents[3]
PARTNER_RUNNER_PATH = EXPERIMENT_DIR / "run_partner_permutation_5090.py"
TEST_PATH = EXPERIMENT_DIR / "test_run_slot_permutation.py"


class ProtocolError(RuntimeError):
    pass


def _load_partner() -> Any:
    name = "semanticfence_slot_partner_runner"
    spec = importlib.util.spec_from_file_location(name, PARTNER_RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise ProtocolError(f"cannot import {PARTNER_RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


P = _load_partner()
CONTRACT = P.CONTRACT


def validate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    if set(config) != {
        "schema_version",
        "status",
        "evidence_boundary",
        "inputs",
        "cohort",
        "execution",
        "decision",
    }:
        raise ProtocolError("slot config fields changed")
    if config["schema_version"] != SCHEMA or config["status"] != "FROZEN_PRE_RUN":
        raise ProtocolError("slot config schema/status changed")
    expected_boundary = (
        "single_rtx5090_reused_run03_calibration_rows_stage1_focal_cohort_"
        "expert_stage_only_row_plus_original_pair_plus_m2_slot_invariance_"
        "not_fresh_evaluation_not_full_layer_not_serving_not_ep_not_latency"
    )
    if config["evidence_boundary"] != expected_boundary:
        raise ProtocolError("slot evidence boundary changed")
    inputs = config["inputs"]
    if set(inputs) != {
        "partner_config_sha256",
        "partner_schedule_sha256",
        "source_calibration_numeric_sha256",
        "source_calibration_captures_sha256",
        "source_calibration_reference_rows_sha256",
        "source_stack_digest",
    } or any(not P.is_sha256(value) for value in inputs.values()):
        raise ProtocolError("slot input bindings changed")
    cohort = config["cohort"]
    expected_cohort = {
        "expected_focals": 512,
        "expected_safe_focals": 256,
        "expected_unsafe_focals": 256,
        "expected_original_pairs": 508,
        "expected_single_focal_pairs": 504,
        "expected_dual_focal_pairs": 4,
        "expected_unique_rows": 1016,
        "require_same_stage1_focal_ids": True,
        "require_original_partner": True,
        "require_actual_slot_swap": True,
    }
    if cohort != expected_cohort:
        raise ProtocolError("slot cohort constants changed")
    if config["execution"] != {
        "m": 2,
        "dtype": "bfloat16",
        "warmups": 3,
        "repeats": 10,
        "max_gpu_seconds": 300,
        "require_old_m1_reference_match": True,
    }:
        raise ProtocolError("slot execution constants changed")
    if config["decision"] != {
        "stable_opposite_outcome_falsifies": True,
        "mixed_outcome_weakens_stability": True,
        "claim": "calibration_only_row_plus_original_pair_plus_m2_slot_invariance",
    }:
        raise ProtocolError("slot decision constants changed")
    return dict(config)


def load_stage1(
    *,
    config: Mapping[str, Any],
    partner_config_path: Path,
    source_dir: Path,
    partner_plan_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    if P.sha256_file(partner_config_path) != config["inputs"][
        "partner_config_sha256"
    ]:
        raise ProtocolError("Stage-1 config hash changed")
    partner_config = P.validate_config(P.load_json(partner_config_path))
    if partner_config["source"]["calibration_numeric_sha256"] != config["inputs"][
        "source_calibration_numeric_sha256"
    ]:
        raise ProtocolError("Stage-1 source numeric binding changed")
    if partner_config["source"]["calibration_captures_sha256"] != config[
        "inputs"
    ]["source_calibration_captures_sha256"]:
        raise ProtocolError("Stage-1 capture binding changed")
    P.verify_source_artifacts(partner_config, source_dir)
    evidence = P.load_m2_evidence(partner_config, source_dir)
    references = P.load_reference_hashes(partner_config, source_dir)
    partner_schedule, _ = P.load_and_verify_plan(
        config=partner_config,
        config_path=partner_config_path,
        plan_dir=partner_plan_dir,
        evidence=evidence,
    )
    if P.sha256_file(Path(partner_plan_dir) / "schedule.jsonl") != config[
        "inputs"
    ]["partner_schedule_sha256"]:
        raise ProtocolError("Stage-1 schedule hash changed")
    return partner_config, evidence, references, partner_schedule


def build_slot_schedule(
    config: Mapping[str, Any],
    partner_schedule: Sequence[Mapping[str, Any]],
    evidence: Mapping[str, Any],
    references: Mapping[str, str],
) -> list[dict[str, Any]]:
    focal_labels: dict[str, str] = {}
    for call in partner_schedule:
        row_id = str(call["focal_row_id"])
        label = str(call["focal_baseline_label"])
        if row_id in focal_labels and focal_labels[row_id] != label:
            raise ProtocolError("Stage-1 focal label changed across calls")
        focal_labels[row_id] = label
    labels = Counter(focal_labels.values())
    if len(focal_labels) != 512 or labels != Counter({"safe": 256, "unsafe": 256}):
        raise ProtocolError("Stage-1 focal cohort/labels changed")

    calls_by_pack: dict[str, dict[str, Any]] = {}
    for focal_id, baseline_label in sorted(focal_labels.items()):
        focal = evidence.get(focal_id)
        if focal is None or focal.baseline_label != baseline_label:
            raise ProtocolError("Stage-1 focal is absent/mislabeled in run03")
        partner = evidence.get(focal.original_partner_row_id)
        if (
            partner is None
            or partner.original_partner_row_id != focal_id
            or partner.original_pack_id != focal.original_pack_id
        ):
            raise ProtocolError("run03 original pair is not symmetric")
        original = (
            (focal, partner) if focal.original_slot == 0 else (partner, focal)
        )
        original_pack = CONTRACT.Pack(
            layer=focal.record.layer,
            expert_id=focal.record.expert_id,
            rows=tuple(row.record for row in original),
        )
        if original_pack.pack_id != focal.original_pack_id:
            raise ProtocolError("run03 original pack order/identity changed")
        swapped = tuple(reversed(original))
        swapped_pack = CONTRACT.Pack(
            layer=focal.record.layer,
            expert_id=focal.record.expert_id,
            rows=tuple(row.record for row in swapped),
        )
        if swapped[focal.original_slot].row_id == focal_id:
            raise ProtocolError("slot intervention did not move the focal")
        descriptor = {
            "focal_row_id": focal_id,
            "baseline_label": baseline_label,
            "original_slot": int(focal.original_slot),
            "swapped_slot": 1 - int(focal.original_slot),
            "old_m1_reference_sha256": references[focal_id],
        }
        current = calls_by_pack.get(original_pack.pack_id)
        if current is None:
            calls_by_pack[original_pack.pack_id] = {
                "schema_version": SCHEDULE_SCHEMA,
                "call_index": -1,
                "layer": int(focal.record.layer),
                "expert_id": int(focal.record.expert_id),
                "m": 2,
                "original_pack_id": original_pack.pack_id,
                "original_row_ids": [row.row_id for row in original],
                "swapped_pack_id": swapped_pack.pack_id,
                "swapped_row_ids": [row.row_id for row in swapped],
                "swapped_row_records": [
                    row.record.identity_payload() for row in swapped
                ],
                "focals": [descriptor],
            }
        else:
            if (
                current["original_row_ids"] != [row.row_id for row in original]
                or current["swapped_row_ids"] != [row.row_id for row in swapped]
            ):
                raise ProtocolError("deduplicated original pair has inconsistent order")
            current["focals"].append(descriptor)

    schedule = [calls_by_pack[key] for key in sorted(calls_by_pack)]
    for index, call in enumerate(schedule):
        call["call_index"] = index
        call["focals"].sort(key=lambda value: value["focal_row_id"])
    validate_slot_schedule(config, schedule, expected_focal_ids=set(focal_labels))
    return schedule


def validate_slot_schedule(
    config: Mapping[str, Any],
    schedule: Sequence[Mapping[str, Any]],
    *,
    expected_focal_ids: set[str] | None = None,
) -> dict[str, int]:
    if len(schedule) != 508:
        raise ProtocolError("slot schedule physical call denominator changed")
    focal_ids: set[str] = set()
    label_counts: Counter[str] = Counter()
    multiplicities: Counter[int] = Counter()
    unique_rows: set[str] = set()
    for index, call in enumerate(schedule):
        if (
            call.get("schema_version") != SCHEDULE_SCHEMA
            or int(call.get("call_index", -1)) != index
            or int(call.get("m", -1)) != 2
            or len(call.get("original_row_ids", [])) != 2
            or len(call.get("swapped_row_ids", [])) != 2
            or call["swapped_row_ids"] != list(reversed(call["original_row_ids"]))
            or set(call["swapped_row_ids"]) != set(call["original_row_ids"])
            or len(call.get("swapped_row_records", [])) != 2
            or len(call.get("focals", [])) not in {1, 2}
        ):
            raise ProtocolError("slot schedule call is invalid/not an actual swap")
        records = tuple(
            P.PILOT.row_record_from_mapping(value)
            for value in call["swapped_row_records"]
        )
        pack = CONTRACT.Pack(
            layer=int(call["layer"]),
            expert_id=int(call["expert_id"]),
            rows=records,
        )
        if (
            [row.row_id for row in records] != call["swapped_row_ids"]
            or pack.pack_id != call["swapped_pack_id"]
        ):
            raise ProtocolError("slot schedule row/pack identity changed")
        multiplicities[len(call["focals"])] += 1
        unique_rows.update(call["swapped_row_ids"])
        for focal in call["focals"]:
            row_id = focal["focal_row_id"]
            old_slot = int(focal["original_slot"])
            new_slot = int(focal["swapped_slot"])
            if (
                row_id in focal_ids
                or old_slot not in {0, 1}
                or new_slot != 1 - old_slot
                or call["original_row_ids"][old_slot] != row_id
                or call["swapped_row_ids"][new_slot] != row_id
                or not P.is_sha256(focal["old_m1_reference_sha256"])
                or focal["baseline_label"] not in {"safe", "unsafe"}
            ):
                raise ProtocolError("slot focal cohort/slot/label is invalid")
            focal_ids.add(row_id)
            label_counts[focal["baseline_label"]] += 1
    if expected_focal_ids is not None and focal_ids != expected_focal_ids:
        raise ProtocolError("slot schedule does not preserve Stage-1 focal IDs")
    if (
        len(focal_ids) != 512
        or label_counts != Counter({"safe": 256, "unsafe": 256})
        or multiplicities != Counter({1: 504, 2: 4})
        or len(unique_rows) != 1016
    ):
        raise ProtocolError("slot schedule frozen denominators changed")
    return {
        "physical_call_count": len(schedule),
        "focal_count": len(focal_ids),
        "single_focal_call_count": multiplicities[1],
        "dual_focal_call_count": multiplicities[2],
        "unique_row_count": len(unique_rows),
    }


def _plan_hashes(plan_dir: Path) -> dict[str, str]:
    return {
        name: P.sha256_file(Path(plan_dir) / name)
        for name in ("config.json", "slot_schedule.jsonl", "SLOT_PLAN.json")
    }


def write_plan(config_path: Path, plan_dir: Path, schedule: Sequence[Mapping[str, Any]]) -> None:
    if plan_dir.exists():
        raise ProtocolError(f"slot plan directory exists: {plan_dir}")
    plan_dir.mkdir(parents=True)
    shutil.copyfile(config_path, plan_dir / "config.json")
    P.write_jsonl_no_overwrite(plan_dir / "slot_schedule.jsonl", schedule)
    validation = validate_slot_schedule(P.load_json(config_path), schedule)
    P.write_json_no_overwrite(
        plan_dir / "SLOT_PLAN.json",
        {
            "schema_version": PLAN_SCHEMA,
            "status": "PLANNED_NOT_EXECUTED",
            "config_sha256": P.sha256_file(config_path),
            "slot_schedule_sha256": P.sha256_file(plan_dir / "slot_schedule.jsonl"),
            "validation": validation,
            "gpu_executed": False,
        },
    )
    P.write_json_no_overwrite(
        plan_dir / "SLOT_PLAN_COMPLETE.json",
        {"status": "SUCCESS_COMPLETE", "artifact_sha256": _plan_hashes(plan_dir)},
    )


def load_plan(config_path: Path, plan_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    complete = P.load_json(plan_dir / "SLOT_PLAN_COMPLETE.json")
    if complete.get("status") != "SUCCESS_COMPLETE" or complete.get(
        "artifact_sha256"
    ) != _plan_hashes(plan_dir):
        raise ProtocolError("slot plan completion/hash is invalid")
    plan = P.load_json(plan_dir / "SLOT_PLAN.json")
    schedule = P.load_jsonl(plan_dir / "slot_schedule.jsonl")
    if (
        plan.get("schema_version") != PLAN_SCHEMA
        or plan.get("status") != "PLANNED_NOT_EXECUTED"
        or plan.get("config_sha256") != P.sha256_file(config_path)
        or plan.get("slot_schedule_sha256")
        != P.sha256_file(plan_dir / "slot_schedule.jsonl")
        or plan.get("gpu_executed") is not False
    ):
        raise ProtocolError("slot plan metadata changed")
    validate_slot_schedule(P.load_json(config_path), schedule)
    return schedule, plan


def _source_bindings(repo_root: Path, config_path: Path) -> dict[str, str]:
    paths = (
        Path(__file__),
        TEST_PATH,
        config_path,
        PARTNER_RUNNER_PATH,
        P.BASE_RUNNER_PATH,
        P.GPU_EXECUTION_PATH,
        P.CONTRACT_PATH,
    )
    result = {}
    for path in paths:
        resolved = path.resolve()
        result[str(resolved.relative_to(repo_root.resolve()))] = P.sha256_file(resolved)
    return dict(sorted(result.items()))


def create_lock(args: argparse.Namespace) -> dict[str, Any]:
    config_path = Path(args.config).resolve()
    partner_config_path = Path(args.partner_config).resolve()
    source_dir = Path(args.source_dir).resolve()
    partner_plan_dir = Path(args.partner_plan_dir).resolve()
    plan_dir = Path(args.plan_dir).resolve()
    config = validate_config(P.load_json(config_path))
    partner_config, evidence, references, partner_schedule = load_stage1(
        config=config,
        partner_config_path=partner_config_path,
        source_dir=source_dir,
        partner_plan_dir=partner_plan_dir,
    )
    schedule, plan = load_plan(config_path, plan_dir)
    expected = build_slot_schedule(config, partner_schedule, evidence, references)
    if schedule != expected:
        raise ProtocolError("slot plan differs from deterministic Stage-1 cohort plan")
    acceptance = P._verify_acceptance(partner_config, Path(args.acceptance_artifact))
    payload = {
        "schema_version": LOCK_SCHEMA,
        "status": "SEALED_BEFORE_GPU_EXECUTION",
        "config_sha256": P.sha256_file(config_path),
        "partner_config_sha256": P.sha256_file(partner_config_path),
        "partner_schedule_sha256": config["inputs"]["partner_schedule_sha256"],
        "slot_schedule_sha256": plan["slot_schedule_sha256"],
        "slot_plan_complete_sha256": P.sha256_file(
            plan_dir / "SLOT_PLAN_COMPLETE.json"
        ),
        "acceptance_sha256": P.sha256_file(Path(args.acceptance_artifact)),
        "acceptance_complete_sha256": P.sha256_file(
            Path(args.acceptance_artifact).parent / "ACCEPTANCE_COMPLETE.json"
        ),
        "stack_digest": acceptance["stack"]["stack_digest"],
        "source_bindings": _source_bindings(Path(args.repo_root), config_path),
    }
    return payload | {"lock_sha256": P.canonical_sha256(payload)}


def verify_lock(args: argparse.Namespace) -> dict[str, Any]:
    observed = P.load_json(Path(args.frozen_lock))
    payload = dict(observed)
    digest = payload.pop("lock_sha256", None)
    if not P.is_sha256(digest) or P.canonical_sha256(payload) != digest:
        raise ProtocolError("slot lock content hash changed")
    expected = create_lock(args)
    if observed != expected:
        raise ProtocolError("slot lock inputs/code changed after seal")
    return observed


def worker_run(args: argparse.Namespace) -> int:
    P.PILOT.assert_numeric_logging_disabled()
    import torch

    config = validate_config(P.load_json(Path(args.config)))
    verify_lock(args)
    partner_config, evidence, references, _ = load_stage1(
        config=config,
        partner_config_path=Path(args.partner_config),
        source_dir=Path(args.source_dir),
        partner_plan_dir=Path(args.partner_plan_dir),
    )
    schedule, _ = load_plan(Path(args.config), Path(args.plan_dir))
    _, acceptance, model = P._load_live_model(
        config=partner_config,
        source_dir=Path(args.source_dir),
        acceptance_path=Path(args.acceptance_artifact),
        model_path_override=args.model_path,
    )
    gpu = P.PILOT._gpu()
    captures = torch.load(
        Path(args.source_dir) / "calibration_captures.pt",
        map_location="cpu",
        weights_only=False,
    )
    materialized = {row.row_id: row for row in gpu.materialize_routed_rows(captures)}
    row_ids = sorted({row_id for call in schedule for row_id in call["swapped_row_ids"]})
    if len(row_ids) != 1016 or any(row_id not in materialized for row_id in row_ids):
        raise ProtocolError("slot scheduled rows do not match frozen captures")
    packs = []
    for call in schedule:
        records = tuple(materialized[row_id].record for row_id in call["swapped_row_ids"])
        if [row.identity_payload() for row in records] != call["swapped_row_records"]:
            raise ProtocolError("slot materialized rows differ from sealed plan")
        packs.append(CONTRACT.Pack(call["layer"], call["expert_id"], records))
    execution = gpu.execute_calibration(
        model=model,
        packs=packs,
        rows=tuple(materialized[row_id] for row_id in row_ids),
        repeats=10,
    )
    raw_path = Path(args.output_dir) / "slot_raw_outputs.pt"
    torch.save(execution, raw_path)
    new_refs = {row.row_id: row for row in execution.reference.rows}
    for row_id in row_ids:
        row = new_refs[row_id]
        if (
            not row.bitwise_stable
            or not row.all_exact_to_reference
            or row.reference_sha256 != references[row_id]
        ):
            raise ProtocolError("slot probe M1 reference is unstable/differs from run03")
    P.write_json_no_overwrite(
        Path(args.output_dir) / "slot_reference_verification.json",
        {
            "status": "ALL_MATCH",
            "unique_row_count": len(row_ids),
            "old_reference_mismatch_count": 0,
        },
    )
    numeric = []
    for call, observed in zip(schedule, execution.packs):
        numeric.append(
            {
                "schema_version": NUMERIC_SCHEMA,
                "call_index": call["call_index"],
                "swapped_pack_id": observed.pack.pack_id,
                "swapped_row_ids": call["swapped_row_ids"],
                "focals": [
                    descriptor
                    | {
                        "repeat_exact": [
                            bool(repeat[int(descriptor["swapped_slot"])])
                            for repeat in observed.repeat_row_exact
                        ],
                        "repeat_sha256": [
                            repeat[int(descriptor["swapped_slot"])]
                            for repeat in observed.repeat_row_sha256
                        ],
                    }
                    for descriptor in call["focals"]
                ],
            }
        )
    P.write_jsonl_no_overwrite(Path(args.output_dir) / "slot_numeric.jsonl", numeric)
    P.write_json_no_overwrite(
        Path(args.output_dir) / "worker_status.json",
        {
            "status": "COMPLETE",
            "stack_digest": acceptance["stack"]["stack_digest"],
            "physical_call_count": len(schedule),
            "focal_count": sum(len(call["focals"]) for call in schedule),
            "slot_numeric_sha256": P.sha256_file(
                Path(args.output_dir) / "slot_numeric.jsonl"
            ),
            "slot_raw_outputs_sha256": P.sha256_file(raw_path),
        },
    )
    return 0


def _status(flags: Sequence[bool]) -> str:
    if len(flags) != 10 or any(type(value) is not bool for value in flags):
        raise ProtocolError("slot focal does not have 10 boolean repeats")
    return "safe" if all(flags) else "unsafe" if not any(flags) else "mixed"


def summarize(
    config: Mapping[str, Any],
    schedule: Sequence[Mapping[str, Any]],
    numeric: Sequence[Mapping[str, Any]],
    reference: Mapping[str, Any],
) -> dict[str, Any]:
    validate_slot_schedule(config, schedule)
    if (
        len(numeric) != len(schedule)
        or reference.get("status") != "ALL_MATCH"
        or int(reference.get("old_reference_mismatch_count", -1)) != 0
    ):
        raise ProtocolError("slot numeric/reference evidence is incomplete")
    flips, mixed = [], []
    seen: set[str] = set()
    for call, value in zip(schedule, numeric):
        if (
            value.get("schema_version") != NUMERIC_SCHEMA
            or int(value.get("call_index", -1)) != int(call["call_index"])
            or value.get("swapped_row_ids") != call["swapped_row_ids"]
            or value.get("swapped_pack_id") != call["swapped_pack_id"]
            or len(value.get("focals", [])) != len(call["focals"])
        ):
            raise ProtocolError("slot numeric differs from sealed schedule")
        expected = {item["focal_row_id"]: item for item in call["focals"]}
        for observed in value["focals"]:
            focal_id = observed["focal_row_id"]
            descriptor = expected.get(focal_id)
            if (
                descriptor is None
                or focal_id in seen
                or any(observed.get(key) != descriptor[key] for key in descriptor)
                or len(observed.get("repeat_sha256", [])) != 10
                or any(not P.is_sha256(item) for item in observed["repeat_sha256"])
            ):
                raise ProtocolError("slot focal numeric identity/metric changed")
            seen.add(focal_id)
            status = _status(observed["repeat_exact"])
            if status == "mixed":
                mixed.append(focal_id)
            elif status != descriptor["baseline_label"]:
                flips.append(focal_id)
    if len(seen) != 512:
        raise ProtocolError("slot numeric focal denominator changed")
    decision = (
        "FALSIFY_SLOT_INVARIANCE"
        if flips
        else "WEAKEN_SLOT_STABILITY"
        if mixed
        else "SUPPORT_CALIBRATION_ONLY"
    )
    return {
        "schema_version": RESULT_SCHEMA,
        "decision": decision,
        "paper_result": False,
        "claim": config["decision"]["claim"],
        "evidence_boundary": config["evidence_boundary"],
        "physical_call_count": 508,
        "focal_count": 512,
        "stable_slot_flip_count": len(flips),
        "mixed_focal_count": len(mixed),
        "stable_slot_flip_focal_ids": sorted(flips),
        "mixed_focal_ids": sorted(mixed),
        "reference_verification": dict(reference),
        "interpretation": (
            "Reused run03 calibration rows; original pair identity and M=2 are fixed, "
            "only row slot is swapped. Not fresh evaluation, serving, latency, or paper evidence."
        ),
    }


def run_plan(args: argparse.Namespace) -> int:
    config = validate_config(P.load_json(Path(args.config)))
    _, evidence, references, partner_schedule = load_stage1(
        config=config,
        partner_config_path=Path(args.partner_config),
        source_dir=Path(args.source_dir),
        partner_plan_dir=Path(args.partner_plan_dir),
    )
    schedule = build_slot_schedule(config, partner_schedule, evidence, references)
    write_plan(Path(args.config), Path(args.plan_dir), schedule)
    print(json.dumps(validate_slot_schedule(config, schedule), sort_keys=True))
    return 0


def run_seal(args: argparse.Namespace) -> int:
    if Path(args.output).exists():
        raise ProtocolError("slot lock output exists")
    P.write_json_no_overwrite(Path(args.output), create_lock(args))
    return 0


def _worker_command(args: argparse.Namespace, deadline: float) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "_worker",
        "--config", str(Path(args.config).resolve()),
        "--partner-config", str(Path(args.partner_config).resolve()),
        "--repo-root", str(Path(args.repo_root).resolve()),
        "--source-dir", str(Path(args.source_dir).resolve()),
        "--partner-plan-dir", str(Path(args.partner_plan_dir).resolve()),
        "--plan-dir", str(Path(args.plan_dir).resolve()),
        "--acceptance-artifact", str(Path(args.acceptance_artifact).resolve()),
        "--frozen-lock", str(Path(args.frozen_lock).resolve()),
        "--output-dir", str(Path(args.output_dir).resolve()),
        "--deadline-epoch", str(deadline),
    ]
    if args.model_path:
        command.extend(("--model-path", str(Path(args.model_path).resolve())))
    return command


def run_parent(args: argparse.Namespace) -> int:
    if Path(args.output_dir).exists():
        raise ProtocolError("slot run output directory exists")
    config = validate_config(P.load_json(Path(args.config)))
    lock = verify_lock(args)
    schedule, _ = load_plan(Path(args.config), Path(args.plan_dir))
    partner_config = P.validate_config(P.load_json(Path(args.partner_config)))
    acceptance = P._verify_acceptance(partner_config, Path(args.acceptance_artifact))
    output = Path(args.output_dir)
    output.mkdir(parents=True)
    deadline = time.time() + int(config["execution"]["max_gpu_seconds"])
    P.write_json_no_overwrite(
        output / "run_request.json",
        {
            "started_at_epoch": time.time(),
            "deadline_epoch": deadline,
            "lock_sha256": lock["lock_sha256"],
            "slot_schedule_sha256": P.sha256_file(Path(args.plan_dir) / "slot_schedule.jsonl"),
            "evidence_boundary": config["evidence_boundary"],
        },
    )
    P.PILOT.run_worker_monitored(
        command=_worker_command(args, deadline),
        log_path=output / "worker.log",
        expected_gpu_uuid=acceptance["stack"]["gpu"]["uuid"],
        deadline_epoch=deadline,
    )
    status = P.load_json(output / "worker_status.json")
    if status.get("status") != "COMPLETE" or status.get(
        "slot_numeric_sha256"
    ) != P.sha256_file(output / "slot_numeric.jsonl"):
        raise ProtocolError("slot worker output is incomplete")
    result = summarize(
        config,
        schedule,
        P.load_jsonl(output / "slot_numeric.jsonl"),
        P.load_json(output / "slot_reference_verification.json"),
    ) | {
        "stack_digest": acceptance["stack"]["stack_digest"],
        "within_gpu_budget": time.time() <= deadline,
    }
    if result["within_gpu_budget"] is not True:
        raise ProtocolError("slot run exceeded GPU deadline")
    P.finalize_complete(output, result)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subs = parser.add_subparsers(dest="command", required=True)
    plan = subs.add_parser("plan")
    for name in ("config", "partner-config", "source-dir", "partner-plan-dir", "plan-dir"):
        plan.add_argument(f"--{name}", required=True)
    seal = subs.add_parser("seal")
    run = subs.add_parser("run")
    worker = subs.add_parser("_worker", help=argparse.SUPPRESS)
    for sub in (seal, run, worker):
        sub.add_argument("--config", required=True)
        sub.add_argument("--partner-config", required=True)
        sub.add_argument("--repo-root", default=str(REPO_ROOT_DEFAULT))
        sub.add_argument("--source-dir", required=True)
        sub.add_argument("--partner-plan-dir", required=True)
        sub.add_argument("--plan-dir", required=True)
        sub.add_argument("--acceptance-artifact", required=True)
    seal.add_argument("--output", required=True)
    for sub in (run, worker):
        sub.add_argument("--frozen-lock", required=True)
        sub.add_argument("--output-dir", required=True)
        sub.add_argument("--model-path")
    worker.add_argument("--deadline-epoch", required=True, type=float)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "plan":
        return run_plan(args)
    if args.command == "seal":
        return run_seal(args)
    if args.command == "_worker":
        return worker_run(args)
    return run_parent(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ProtocolError, P.ProtocolError, P.PILOT.ProtocolError) as error:
        print(f"INVALID: {error}", file=sys.stderr)
        raise SystemExit(2)
