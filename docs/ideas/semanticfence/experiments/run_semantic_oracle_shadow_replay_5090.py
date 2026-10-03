#!/usr/bin/env python3
"""Replay M=2 expert outputs through their original calibration windows.

The CPU-only ``plan`` command selects one safe and one unsafe focal per layer
from the completed partner-permutation cohort.  Each focal is paired with its
original companion and with one opposite-label rank-0 companion (32 targets,
64 interventions).  ``run`` computes the M=2 target output, injects exactly
that one raw expert contribution into the focal's original 16-token window,
and observes downstream target-token routes and final logits.

This is a reused-calibration, single-GPU semantic shadow replay.  It is not a
fresh evaluation, a serving result, an EP result, or a paper result.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import sys
from typing import Any, Iterable, Mapping, Sequence


EXPERIMENT_DIR = Path(__file__).resolve().parent
STABLEBATCH_RUNNER = (
    EXPERIMENT_DIR.parents[1]
    / "stablebatch"
    / "experiments"
    / "run_observable_selector_pilot.py"
)
DEFAULT_PARTNER_DIR = (
    EXPERIMENT_DIR / "outputs" / "partner_permutation_20260810_run01"
)
DEFAULT_SOURCE_DIR = (
    EXPERIMENT_DIR / "outputs" / "semanticfence_pilot_20260810_run03"
)
SCHEDULE_SCHEMA = "semanticfence-semantic-oracle-shadow-schedule-v1"
RESULT_SCHEMA = "semanticfence-semantic-oracle-shadow-result-v1"
COMPLETE_SCHEMA = "semanticfence-semantic-oracle-shadow-complete-v1"
SELECTION_SEED = "semanticfence-m2-semantic-oracle-shadow-v1"
EXPECTED_LAYERS = 16
EXPECTED_TARGETS = 32
EXPECTED_INTERVENTIONS = 64
SIDE_REPEATS = 10
FULL_REPEATS = 2


class ShadowReplayError(RuntimeError):
    """The shadow replay cannot be interpreted."""


def semantic_surface_contract() -> dict[str, Any]:
    return {
        "baseline": "fresh_M1_target_output_injected",
        "treatment": "paired_M2_target_output_injected",
        "other_contributions": "native",
        "side_call_repeats": SIDE_REPEATS,
        "full_forward_repeats_per_surface": FULL_REPEATS,
    }


def _load_module(name: str, path: Path) -> Any:
    cached = sys.modules.get(name)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ShadowReplayError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


import run_cross_companion_metric_replay_5090 as cross  # noqa: E402


observable = _load_module(
    "semanticfence_shadow_observable_selector", STABLEBATCH_RUNNER
)
stable = observable.base


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    return cross.sha256_file(Path(path))


def write_json_no_overwrite(path: Path, value: Any) -> None:
    cross.write_json_no_overwrite(Path(path), value)


def write_jsonl_no_overwrite(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    cross.write_jsonl_no_overwrite(Path(path), rows)


def _selection_key(namespace: str, *values: Any) -> str:
    payload = "|".join([SELECTION_SEED, namespace, *(str(value) for value in values)])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def extract_focal_records(
    partner_rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for call in partner_rows:
        focal_id = str(call["focal_row_id"])
        slot = int(call["focal_original_slot"])
        row_ids = list(call["row_ids"])
        row_records = list(call["row_records"])
        if slot not in {0, 1} or row_ids[slot] != focal_id or len(row_records) != 2:
            raise ShadowReplayError("partner focal record/slot is malformed")
        record = dict(row_records[slot])
        existing = records.setdefault(focal_id, record)
        if existing != record:
            raise ShadowReplayError("partner schedule changes a focal row record")
    return records


def load_m2_row_info(path: Path) -> dict[str, dict[str, Any]]:
    """Load the frozen M=2 record and stable label for every endpoint row."""

    result: dict[str, dict[str, Any]] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            value = json.loads(line)
            if int(value["m"]) != 2:
                continue
            row_ids = list(value["row_ids"])
            records = list(value["row_records"])
            repeats = list(value["repeat_row_exact"])
            if len(row_ids) != 2 or len(records) != 2 or len(repeats) != SIDE_REPEATS:
                raise ShadowReplayError("frozen M2 endpoint evidence is malformed")
            for index, row_id in enumerate(row_ids):
                flags = [bool(repeat[index]) for repeat in repeats]
                if len(set(flags)) != 1:
                    raise ShadowReplayError("frozen M2 endpoint label is unstable")
                info = {
                    "record": dict(records[index]),
                    "baseline_label": "safe" if flags[0] else "unsafe",
                }
                existing = result.setdefault(str(row_id), info)
                if existing != info:
                    raise ShadowReplayError("frozen M2 endpoint identity is duplicated")
    if len(result) != 32234:
        raise ShadowReplayError(
            f"expected 32,234 frozen M2 endpoint rows, observed {len(result)}"
        )
    return result


def load_capture_windows(path: Path) -> dict[tuple[int, str, int], dict[str, Any]]:
    windows: dict[tuple[int, str, int], dict[str, Any]] = {}
    for row in cross.load_jsonl(Path(path)):
        key = (
            int(row["document_index"]),
            str(row["document_sha256"]),
            int(row["offset"]),
        )
        if key in windows:
            raise ShadowReplayError("capture manifest repeats a document window")
        token_ids = list(map(int, row.get("window_token_ids", [])))
        if len(token_ids) != 16 or list(row.get("selected_positions", [])) != list(range(16)):
            raise ShadowReplayError("capture manifest lacks a full replayable 16-token window")
        router_hashes = list(row.get("router_logits_sha256_by_layer", []))
        if len(router_hashes) != EXPECTED_LAYERS:
            raise ShadowReplayError("capture manifest lacks all layer router hashes")
        windows[key] = dict(row)
    if len(windows) != 16:
        raise ShadowReplayError(f"expected 16 calibration windows, observed {len(windows)}")
    return windows


def build_shadow_schedule(
    cross_calls: Sequence[Mapping[str, Any]],
    *,
    row_info: Mapping[str, Mapping[str, Any]],
    windows: Mapping[tuple[int, str, int], Mapping[str, Any]],
    source_schedule_sha256: str,
    expected_layers: int = EXPECTED_LAYERS,
) -> list[dict[str, Any]]:
    """Select 32 targets and bind each to original/opposite companions."""

    by_focal: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for call in cross_calls:
        by_focal[str(call["focal_row_id"])].append(call)
    by_layer_label: dict[tuple[int, str], list[str]] = defaultdict(list)
    for focal_id, calls in by_focal.items():
        if len(calls) != 4:
            raise ShadowReplayError("cross-companion focal lacks four calls")
        first = calls[0]
        by_layer_label[(int(first["layer"]), str(first["focal_baseline_label"]))].append(focal_id)

    selected: list[str] = []
    for layer in range(expected_layers):
        for label in ("safe", "unsafe"):
            candidates = sorted(
                by_layer_label[(layer, label)],
                key=lambda focal_id: _selection_key(
                    "target", source_schedule_sha256, layer, label, focal_id
                ),
            )
            if not candidates:
                raise ShadowReplayError(f"layer {layer} lacks a {label} focal")
            selected.append(candidates[0])

    schedule: list[dict[str, Any]] = []
    for focal_id in selected:
        calls = by_focal[focal_id]
        first = calls[0]
        label = str(first["focal_baseline_label"])
        by_kind = {str(call["companion_kind"]): call for call in calls}
        opposite_kind = "unsafe_rank0" if label == "safe" else "safe_rank0"
        if "original" not in by_kind or opposite_kind not in by_kind:
            raise ShadowReplayError("cross cohort lacks original/opposite companion")
        record = dict(row_info[focal_id]["record"])
        key = (
            int(record["document_index"]),
            str(record["document_sha256"]),
            int(record["offset"]),
        )
        if key not in windows:
            raise ShadowReplayError("focal has no replayable capture window")
        window = windows[key]
        token_position = int(record["token_position"])
        route_rank = int(record["route_rank"])
        if not 0 <= token_position < 16 or not 1 <= route_rank <= 8:
            raise ShadowReplayError("focal token/rank is outside the captured surface")
        for intervention_kind, source_call in (
            ("original_companion", by_kind["original"]),
            ("opposite_label_rank0_companion", by_kind[opposite_kind]),
        ):
            endpoints: list[dict[str, Any]] = []
            for endpoint_index, row_id in enumerate(source_call["row_ids"]):
                info = row_info[str(row_id)]
                endpoint_record = dict(info["record"])
                endpoint_key = (
                    int(endpoint_record["document_index"]),
                    str(endpoint_record["document_sha256"]),
                    int(endpoint_record["offset"]),
                )
                if endpoint_key not in windows:
                    raise ShadowReplayError("pair endpoint has no replayable window")
                endpoint_window = windows[endpoint_key]
                endpoints.append(
                    {
                        "endpoint_index": endpoint_index,
                        "row_id": str(row_id),
                        "baseline_label": str(info["baseline_label"]),
                        "row_record": endpoint_record,
                        "window_id": str(endpoint_window["window_id"]),
                        "window_token_ids": list(
                            map(int, endpoint_window["window_token_ids"])
                        ),
                        "capture_full_hidden_states_sha256": str(
                            endpoint_window["full_hidden_states_sha256"]
                        ),
                    }
                )
            schedule.append(
                {
                    "schema_version": SCHEDULE_SCHEMA,
                    "source_schedule_sha256": source_schedule_sha256,
                    "layer": int(source_call["layer"]),
                    "expert_id": int(source_call["expert_id"]),
                    "focal_row_id": focal_id,
                    "focal_baseline_label": label,
                    "focal_original_slot": int(source_call["focal_original_slot"]),
                    "focal_old_m1_reference_sha256": str(
                        source_call["focal_old_m1_reference_sha256"]
                    ),
                    "companion_row_id": str(source_call["companion_row_id"]),
                    "companion_kind": str(source_call["companion_kind"]),
                    "companion_baseline_label": source_call[
                        "companion_baseline_label"
                    ],
                    "intervention_kind": intervention_kind,
                    "row_ids": list(source_call["row_ids"]),
                    "endpoints": endpoints,
                    "target_row_record": record,
                    "window_id": str(window["window_id"]),
                    "window_token_ids": list(map(int, window["window_token_ids"])),
                    "capture_full_hidden_states_sha256": str(
                        window["full_hidden_states_sha256"]
                    ),
                }
            )
    schedule.sort(
        key=lambda row: (
            int(row["layer"]),
            str(row["focal_baseline_label"]),
            str(row["focal_row_id"]),
            str(row["intervention_kind"]),
        )
    )
    for index, row in enumerate(schedule):
        row["call_index"] = index
        row["call_sha256"] = canonical_sha256(
            {key: value for key, value in row.items() if key != "call_sha256"}
        )

    expected_targets = expected_layers * 2
    if len({row["focal_row_id"] for row in schedule}) != expected_targets:
        raise ShadowReplayError("semantic shadow target count changed")
    if len(schedule) != expected_targets * 2:
        raise ShadowReplayError("semantic shadow intervention count changed")
    counts = Counter(
        (int(row["layer"]), str(row["focal_baseline_label"])) for row in schedule
    )
    if any(counts[(layer, label)] != 2 for layer in range(expected_layers) for label in ("safe", "unsafe")):
        raise ShadowReplayError("semantic shadow layer/label coverage is not exact")
    return schedule


def prepare_plan(partner_dir: Path, source_dir: Path) -> tuple[list[dict[str, Any]], str]:
    _path, partner_rows, source_digest = cross._source_schedule(partner_dir)
    cross_calls = cross.build_cross_companion_schedule(
        partner_rows, source_schedule_sha256=source_digest
    )
    row_info = load_m2_row_info(source_dir / "calibration_numeric.jsonl")
    windows = load_capture_windows(source_dir / "calibration_capture_manifest.jsonl")
    schedule = build_shadow_schedule(
        cross_calls,
        row_info=row_info,
        windows=windows,
        source_schedule_sha256=source_digest,
    )
    return schedule, source_digest


def run_plan(args: argparse.Namespace) -> int:
    schedule, source_digest = prepare_plan(
        Path(args.partner_dir).resolve(), Path(args.source_dir).resolve()
    )
    output = Path(args.output).resolve()
    write_jsonl_no_overwrite(output, schedule)
    print(
        json.dumps(
            {
                "source_schedule_sha256": source_digest,
                "schedule_sha256": sha256_file(output),
                "targets": len({row["focal_row_id"] for row in schedule}),
                "unique_endpoint_rows": len(
                    {
                        endpoint["row_id"]
                        for row in schedule
                        for endpoint in row["endpoints"]
                    }
                ),
                "interventions": len(schedule),
                "layers": len({int(row["layer"]) for row in schedule}),
                "replayable_windows": len(
                    {
                        endpoint["window_id"]
                        for row in schedule
                        for endpoint in row["endpoints"]
                    }
                ),
                "gpu_executed": False,
            },
            sort_keys=True,
        )
    )
    return 0


def public_observation(observation: Mapping[str, Any]) -> dict[str, Any]:
    return stable.public_observation(observation)


def observation_signature(observation: Mapping[str, Any]) -> str:
    return canonical_sha256(public_observation(observation))


def require_stable_hashes(
    label: str, hashes: Sequence[str], *, expected_repeats: int
) -> str:
    if len(hashes) != expected_repeats or len(set(hashes)) != 1:
        raise ShadowReplayError(
            f"{label} is not {expected_repeats}/{expected_repeats} bitwise stable"
        )
    return str(hashes[0])


def validate_native_noop_observations(
    native_rows: Sequence[Mapping[str, Any]],
    noop_rows: Sequence[Mapping[str, Any]],
    *,
    expected_repeats: int = FULL_REPEATS,
) -> str:
    if len(native_rows) != expected_repeats or len(noop_rows) != expected_repeats:
        raise ShadowReplayError("native/no-op repeat denominator changed")
    signatures = [observation_signature(row) for row in native_rows]
    native_signature = require_stable_hashes(
        "native full-forward", signatures, expected_repeats=expected_repeats
    )
    noop_signatures = [observation_signature(row) for row in noop_rows]
    if any(value != native_signature for value in noop_signatures):
        raise ShadowReplayError("patched native no-op differs from unmodified native")
    return native_signature


def compare_route_traces(
    native_routes: Sequence[Sequence[int]],
    observed_routes: Sequence[Sequence[int]],
    *,
    start_layer: int,
) -> dict[str, Any]:
    if len(native_routes) != len(observed_routes):
        raise ShadowReplayError("route traces have different layer counts")
    ordered_changed: list[int] = []
    membership_changed: list[int] = []
    for layer in range(start_layer, len(native_routes)):
        left = list(map(int, native_routes[layer]))
        right = list(map(int, observed_routes[layer]))
        if left != right:
            ordered_changed.append(layer)
        if set(left) != set(right):
            membership_changed.append(layer)
    return {
        "downstream_start_layer": start_layer,
        "ordered_topk_changed_layers": ordered_changed,
        "membership_changed_layers": membership_changed,
        "any_ordered_topk_change": bool(ordered_changed),
        "any_membership_change": bool(membership_changed),
    }


def endpoint_route_topk_safe(endpoint_result: Mapping[str, Any]) -> bool:
    """Return the frozen Oracle-B predicate; greedy changes stay diagnostic."""

    return not bool(endpoint_result["route_delta"]["any_ordered_topk_change"])


def _tensor_difference_metrics(reference: Any, observed: Any) -> dict[str, Any]:
    import torch

    if reference.dtype != observed.dtype or tuple(reference.shape) != tuple(observed.shape):
        raise ShadowReplayError("final-logit tensors differ in dtype/shape")
    delta = observed.float() - reference.float()
    return {
        "differing_count": stable.bitwise_changed_elements(reference, observed),
        "max_abs": float(delta.abs().max().item()),
        "l1": float(delta.abs().sum().item()),
        "l2": float(torch.linalg.vector_norm(delta).item()),
    }


def _runtime_config(model: Any, token_position: int) -> dict[str, Any]:
    return {
        "data": {"victim_position": int(token_position)},
        "model": {
            "num_experts_per_tok": int(model.config.num_experts_per_tok),
        },
    }


def _resolve_schedule(
    schedule: list[dict[str, Any]],
    *,
    evidence: Mapping[str, Any],
    rows_by_id: Mapping[str, Any],
    captures_by_window: Mapping[str, Any],
) -> None:
    for call in schedule:
        focal = evidence.get(call["focal_row_id"])
        companion = evidence.get(call["companion_row_id"])
        if focal is None or companion is None:
            raise ShadowReplayError("scheduled row lacks frozen M2 evidence")
        if focal.baseline_label != call["focal_baseline_label"]:
            raise ShadowReplayError("focal label differs from M2 evidence")
        if focal.original_slot != int(call["focal_original_slot"]):
            raise ShadowReplayError("focal slot differs from M2 evidence")
        if call["intervention_kind"] == "opposite_label_rank0_companion":
            expected = "unsafe" if focal.baseline_label == "safe" else "safe"
            if companion.baseline_label != expected:
                raise ShadowReplayError("opposite-label companion is not opposite")
        call["resolved_companion_baseline_label"] = companion.baseline_label
        for endpoint_index, endpoint in enumerate(call["endpoints"]):
            row_id = str(endpoint["row_id"])
            try:
                row = rows_by_id[row_id]
                capture = captures_by_window[endpoint["window_id"]]
            except KeyError as exc:
                raise ShadowReplayError(
                    "scheduled endpoint/window is absent from captures"
                ) from exc
            if row.record.identity_payload() != endpoint["row_record"]:
                raise ShadowReplayError("materialized endpoint differs from frozen record")
            if (
                row.record.layer != int(call["layer"])
                or row.record.expert_id != int(call["expert_id"])
            ):
                raise ShadowReplayError("M2 side-call crosses layer/expert")
            endpoint_evidence = evidence[row_id]
            if endpoint_evidence.baseline_label != endpoint["baseline_label"]:
                raise ShadowReplayError("endpoint label differs from frozen evidence")
            if tuple(map(int, capture.window_token_ids)) != tuple(
                endpoint["window_token_ids"]
            ):
                raise ShadowReplayError("endpoint capture token IDs differ from plan")
            record = row.record
            rank_index = int(record.route_rank) - 1
            observed_expert = int(
                capture.selected_experts[
                    int(record.layer), int(record.token_position), rank_index
                ].item()
            )
            if observed_expert != int(record.expert_id):
                raise ShadowReplayError("endpoint capture expert/rank alignment failed")
            endpoint["topk_rank_zero_based"] = rank_index
            endpoint["endpoint_index"] = endpoint_index
        focal_endpoint = call["endpoints"][int(call["focal_original_slot"])]
        call["target_topk_rank_zero_based"] = int(
            focal_endpoint["topk_rank_zero_based"]
        )


def _side_calls(
    model: Any,
    schedule: Sequence[Mapping[str, Any]],
    rows_by_id: Mapping[str, Any],
    old_references: Mapping[str, str],
) -> tuple[dict[str, dict[Any, Any]], list[dict[str, Any]]]:
    import torch

    m1_by_row: dict[str, Any] = {}
    m2_by_call: dict[int, tuple[Any, Any]] = {}
    ledger: list[dict[str, Any]] = []
    first_by_row: dict[str, tuple[Mapping[str, Any], Mapping[str, Any]]] = {}
    for call in schedule:
        for endpoint in call["endpoints"]:
            first_by_row.setdefault(str(endpoint["row_id"]), (call, endpoint))

    for row_id, (call, endpoint) in sorted(first_by_row.items()):
        row = rows_by_id[row_id]
        hidden = row.tensor.to(device="cuda", dtype=torch.bfloat16).reshape(1, -1)
        expert = model.model.layers[int(call["layer"])].mlp.experts[int(call["expert_id"])]
        outputs: list[Any] = []
        hashes: list[str] = []
        with torch.inference_mode():
            for _ in range(SIDE_REPEATS):
                output = expert(hidden).detach().cpu().clone()
                if output.dtype != torch.bfloat16 or not bool(torch.isfinite(output).all().item()):
                    raise ShadowReplayError("M1 side-call output is invalid")
                outputs.append(output[0].detach().clone())
                hashes.append(stable.tensor_sha256(output[0]))
        observed_hash = require_stable_hashes(
            "M1 side-call", hashes, expected_repeats=SIDE_REPEATS
        )
        if observed_hash != old_references[row_id]:
            raise ShadowReplayError("fresh M1 side-call differs from frozen M1 reference")
        m1_by_row[row_id] = outputs[0].to(device="cuda", dtype=torch.bfloat16)
        ledger.append(
            {
                "surface": "M1_injected_baseline",
                "row_id": row_id,
                "endpoint_baseline_label": endpoint["baseline_label"],
                "m": 1,
                "target_sha256_by_repeat": hashes,
                "target_output_bitwise_stable_10_of_10": True,
                "matches_old_m1_reference": True,
            }
        )

    for call in schedule:
        materialized = [rows_by_id[row_id] for row_id in call["row_ids"]]
        batch = torch.stack(
            [row.tensor.to(device="cuda", dtype=torch.bfloat16) for row in materialized]
        )
        expert = model.model.layers[int(call["layer"])].mlp.experts[int(call["expert_id"])]
        endpoint_outputs: list[list[Any]] = [[], []]
        endpoint_hashes: list[list[str]] = [[], []]
        full_hashes: list[str] = []
        with torch.inference_mode():
            for _ in range(SIDE_REPEATS):
                output = expert(batch).detach().cpu().clone()
                if output.dtype != torch.bfloat16 or not bool(torch.isfinite(output).all().item()):
                    raise ShadowReplayError("M2 side-call output is invalid")
                for endpoint_index in (0, 1):
                    endpoint_output = output[endpoint_index].detach().clone()
                    endpoint_outputs[endpoint_index].append(endpoint_output)
                    endpoint_hashes[endpoint_index].append(
                        stable.tensor_sha256(endpoint_output)
                    )
                full_hashes.append(stable.tensor_sha256(output))
        require_stable_hashes(
            "M2 full side-call", full_hashes, expected_repeats=SIDE_REPEATS
        )
        endpoint_ledger: list[dict[str, Any]] = []
        replacement_pair: list[Any] = []
        for endpoint_index, endpoint in enumerate(call["endpoints"]):
            endpoint_hash = require_stable_hashes(
                f"M2 endpoint {endpoint_index} side-call",
                endpoint_hashes[endpoint_index],
                expected_repeats=SIDE_REPEATS,
            )
            exact_to_m1 = endpoint_hash == old_references[endpoint["row_id"]]
            if exact_to_m1 != (endpoint["baseline_label"] == "safe"):
                raise ShadowReplayError(
                    "fresh M2 endpoint label differs from frozen baseline"
                )
            replacement_pair.append(
                endpoint_outputs[endpoint_index][0].to(
                    device="cuda", dtype=torch.bfloat16
                )
            )
            endpoint_ledger.append(
                {
                    "endpoint_index": endpoint_index,
                    "row_id": endpoint["row_id"],
                    "baseline_label": endpoint["baseline_label"],
                    "sha256_by_repeat": endpoint_hashes[endpoint_index],
                    "exact_to_old_m1": exact_to_m1,
                }
            )
        m2_by_call[int(call["call_index"])] = tuple(replacement_pair)  # type: ignore[assignment]
        ledger.append(
            {
                "surface": "M2_paired_treatment",
                "call_index": int(call["call_index"]),
                "focal_row_id": call["focal_row_id"],
                "companion_row_id": call["companion_row_id"],
                "intervention_kind": call["intervention_kind"],
                "full_m2_sha256_by_repeat": full_hashes,
                "endpoints": endpoint_ledger,
                "both_endpoint_outputs_bitwise_stable_10_of_10": True,
                "full_output_bitwise_stable_10_of_10": True,
            }
        )
    return {"m1_by_row": m1_by_row, "m2_by_call": m2_by_call}, ledger


def _endpoint_call(
    call: Mapping[str, Any], endpoint_index: int, capture: Any
) -> dict[str, Any]:
    endpoint = call["endpoints"][endpoint_index]
    other = call["endpoints"][1 - endpoint_index]
    return {
        "call_index": int(call["call_index"]),
        "pair_call_index": int(call["call_index"]),
        "endpoint_index": endpoint_index,
        "layer": int(call["layer"]),
        "expert_id": int(call["expert_id"]),
        "focal_row_id": endpoint["row_id"],
        "focal_baseline_label": endpoint["baseline_label"],
        "companion_row_id": other["row_id"],
        "resolved_companion_baseline_label": other["baseline_label"],
        "intervention_kind": call["intervention_kind"],
        "target_row_record": endpoint["row_record"],
        "target_topk_rank_zero_based": int(endpoint["topk_rank_zero_based"]),
        "window_token_ids": endpoint["window_token_ids"],
        "window_id": endpoint["window_id"],
        "_capture": capture,
    }


def _native_and_noop(
    model: Any,
    call: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch

    record = call["target_row_record"]
    identity = stable.PairIdentity(
        layer=int(call["layer"]),
        flat_token_idx=int(record["token_position"]),
        topk_rank=int(call["target_topk_rank_zero_based"]),
        expert_id=int(call["expert_id"]),
    )
    input_ids = torch.tensor([call["window_token_ids"]], dtype=torch.long, device="cuda")
    config = _runtime_config(model, int(record["token_position"]))
    native_rows = [
        stable.run_observation(model, input_ids, config, identity)
        for _ in range(FULL_REPEATS)
    ]
    native = native_rows[0]
    if native["target_input_sha256"] != record["hidden_sha256"]:
        raise ShadowReplayError("native target hidden differs from captured focal")
    expected_topk = list(
        map(
            int,
            call["_capture"].selected_experts[
                int(call["layer"]), int(record["token_position"])
            ].tolist(),
        )
    )
    if native["topk_experts_by_layer"][int(call["layer"])] != expected_topk:
        raise ShadowReplayError("native target top-k differs from capture")
    if expected_topk[int(call["target_topk_rank_zero_based"])] != int(call["expert_id"]):
        raise ShadowReplayError("native expert/rank identity differs from focal")

    noop_rows: list[dict[str, Any]] = []
    noop_traces: list[dict[str, Any]] = []
    for _ in range(FULL_REPEATS):
        with stable.patched_single_contribution(
            model, identity, None, "self"
        ) as trace:
            observation = stable.run_observation(model, input_ids, config, identity)
        noop_rows.append(observation)
        noop_traces.append(dict(trace))
    validate_native_noop_observations(native_rows, noop_rows)
    trace_signatures = {canonical_sha256(trace) for trace in noop_traces}
    if len(trace_signatures) != 1:
        raise ShadowReplayError("native no-op trace is not 2/2 stable")
    trace = noop_traces[0]
    if (
        int(trace["pair_match_count"]) != 1
        or trace["target_native_raw_sha256"] != trace["target_applied_raw_sha256"]
        or trace["target_selected_experts"] != expected_topk
    ):
        raise ShadowReplayError("native no-op target pair alignment failed")
    baseline = {
        "focal_row_id": call["focal_row_id"],
        "native_observation": public_observation(native),
        "noop_observation": public_observation(noop_rows[0]),
        "noop_trace": trace,
        "native_full_forward_stable_2_of_2": True,
        "noop_full_forward_stable_2_of_2": True,
        "native_noop_exact": True,
    }
    return {"observation": native, "trace": trace, "identity": identity, "input_ids": input_ids, "config": config}, baseline


def _run_intervention(
    model: Any,
    call: Mapping[str, Any],
    m1_replacement: Any,
    m2_replacement: Any,
    baseline: Mapping[str, Any],
) -> dict[str, Any]:
    native = baseline["observation"]
    noop_trace = baseline["trace"]
    identity = baseline["identity"]

    def run_surface(surface: str, replacement: Any) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        replacement_hash = stable.tensor_sha256(replacement)
        for repeat in range(FULL_REPEATS):
            with stable.patched_single_contribution(
                model, identity, replacement, "replacement"
            ) as trace:
                observation = stable.run_observation(
                    model, baseline["input_ids"], baseline["config"], identity
                )
            if (
                int(trace["pair_match_count"]) != 1
                or trace["target_input_sha256"]
                != call["target_row_record"]["hidden_sha256"]
                or trace["target_applied_raw_sha256"] != replacement_hash
                or trace["target_native_raw_sha256"]
                != noop_trace["target_native_raw_sha256"]
                or trace["non_target_contributions_sha256"]
                != noop_trace["non_target_contributions_sha256"]
            ):
                raise ShadowReplayError(
                    f"{surface} did not isolate one target contribution"
                )
            for layer in range(int(call["layer"]) + 1):
                if observation["router_logits_sha256_by_layer"][layer] != native[
                    "router_logits_sha256_by_layer"
                ][layer]:
                    raise ShadowReplayError(
                        f"{surface} changed a route before/at intervention layer"
                    )
            rows.append(
                {
                    "observation": observation,
                    "trace": dict(trace),
                    "final_logits_vs_native": _tensor_difference_metrics(
                        native["_final_logits_cpu"], observation["_final_logits_cpu"]
                    ),
                }
            )
        signatures = {
            canonical_sha256(
                {
                    "observation": public_observation(row["observation"]),
                    "trace": row["trace"],
                    "final_logits_vs_native": row["final_logits_vs_native"],
                }
            )
            for row in rows
        }
        if len(signatures) != 1:
            raise ShadowReplayError(f"{surface} full-forward is not 2/2 stable")
        return rows

    m1_rows = run_surface("M1 injected baseline", m1_replacement)
    m2_rows = run_surface("M2 paired treatment", m2_replacement)
    m1 = m1_rows[0]
    m2 = m2_rows[0]
    m1_observation = m1["observation"]
    m2_observation = m2["observation"]
    route_delta = compare_route_traces(
        m1_observation["topk_experts_by_layer"],
        m2_observation["topk_experts_by_layer"],
        start_layer=int(call["layer"]) + 1,
    )
    router_hash_changed = [
        layer
        for layer in range(
            int(call["layer"]) + 1,
            len(m1_observation["router_logits_sha256_by_layer"]),
        )
        if m1_observation["router_logits_sha256_by_layer"][layer]
        != m2_observation["router_logits_sha256_by_layer"][layer]
    ]
    logits = _tensor_difference_metrics(
        m1_observation["_final_logits_cpu"], m2_observation["_final_logits_cpu"]
    )
    greedy_changed = (
        m1_observation["greedy_token_id"] != m2_observation["greedy_token_id"]
    )
    return {
        "call_index": int(call["call_index"]),
        "pair_call_index": int(call["pair_call_index"]),
        "endpoint_index": int(call["endpoint_index"]),
        "focal_row_id": call["focal_row_id"],
        "focal_baseline_label": call["focal_baseline_label"],
        "companion_row_id": call["companion_row_id"],
        "resolved_companion_baseline_label": call[
            "resolved_companion_baseline_label"
        ],
        "intervention_kind": call["intervention_kind"],
        "layer": int(call["layer"]),
        "expert_id": int(call["expert_id"]),
        "target_topk_rank_zero_based": int(call["target_topk_rank_zero_based"]),
        "m1_injected_full_forward_stable_2_of_2": True,
        "m2_injected_full_forward_stable_2_of_2": True,
        "m1_injected_baseline": {
            "observation": public_observation(m1_observation),
            "trace": m1["trace"],
            "final_logits_vs_native": m1["final_logits_vs_native"],
        },
        "m2_injected_treatment": {
            "observation": public_observation(m2_observation),
            "trace": m2["trace"],
            "final_logits_vs_native": m2["final_logits_vs_native"],
        },
        "route_delta": route_delta,
        "router_logits_changed_layers": router_hash_changed,
        "final_logits_m2_vs_m1": logits,
        "greedy_changed": greedy_changed,
        "semantic_edge": route_delta["any_ordered_topk_change"],
        "route_topk_semantic_safe": not route_delta["any_ordered_topk_change"],
    }


def maximum_safe_matching(pair_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Compute an exact maximum matching on the observed bipartite edge sample."""

    graph: dict[str, set[str]] = defaultdict(set)
    edge_index: dict[frozenset[str], int] = {}
    for row in pair_rows:
        left, right = map(str, row["row_ids"])
        if left == right or frozenset((left, right)) in edge_index:
            raise ShadowReplayError("observed pair graph has a loop/duplicate edge")
        graph[left].add(right)
        graph[right].add(left)
        edge_index[frozenset((left, right))] = int(row["call_index"])

    color: dict[str, int] = {}
    for start in sorted(graph):
        if start in color:
            continue
        color[start] = 0
        queue = [start]
        while queue:
            node = queue.pop(0)
            for neighbor in sorted(graph[node]):
                if neighbor not in color:
                    color[neighbor] = 1 - color[node]
                    queue.append(neighbor)
                elif color[neighbor] == color[node]:
                    raise ShadowReplayError("observed pair graph is not bipartite")

    safe_adjacency: dict[str, list[str]] = defaultdict(list)
    for row in pair_rows:
        if not bool(row["semantic_safe"]):
            continue
        first, second = map(str, row["row_ids"])
        left, right = (first, second) if color[first] == 0 else (second, first)
        safe_adjacency[left].append(right)
    for left in safe_adjacency:
        safe_adjacency[left].sort()

    matched_right: dict[str, str] = {}

    def augment(left: str, seen: set[str]) -> bool:
        for right in safe_adjacency.get(left, []):
            if right in seen:
                continue
            seen.add(right)
            if right not in matched_right or augment(matched_right[right], seen):
                matched_right[right] = left
                return True
        return False

    for left in sorted(safe_adjacency):
        augment(left, set())
    pairs = sorted((left, right) for right, left in matched_right.items())
    matched_edges = [
        {
            "row_ids": [left, right],
            "source_call_index": edge_index[frozenset((left, right))],
        }
        for left, right in pairs
    ]
    vertices = len(graph)
    return {
        "algorithm": "exact_bipartite_augmenting_path_on_observed_64_edge_graph",
        "observed_vertices": vertices,
        "observed_edges": len(pair_rows),
        "safe_edges": sum(bool(row["semantic_safe"]) for row in pair_rows),
        "matching_edges": len(matched_edges),
        "covered_vertices": 2 * len(matched_edges),
        "vertex_coverage": (2 * len(matched_edges) / vertices) if vertices else 0.0,
        "matching": matched_edges,
    }


def aggregate_results(
    schedule: Sequence[Mapping[str, Any]],
    pair_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if len(pair_rows) != len(schedule):
        raise ShadowReplayError("pair result denominator differs from schedule")
    endpoints = [endpoint for pair in pair_rows for endpoint in pair["endpoints"]]
    endpoint_edges = sum(bool(row["semantic_edge"]) for row in endpoints)
    route_edges = sum(
        bool(row["route_delta"]["any_ordered_topk_change"]) for row in endpoints
    )
    greedy_edges = sum(bool(row["greedy_changed"]) for row in endpoints)
    logit_edges = sum(
        int(row["final_logits_m2_vs_m1"]["differing_count"] > 0)
        for row in endpoints
    )
    safe_pairs = sum(bool(row["semantic_safe"]) for row in pair_rows)
    strata: dict[str, dict[str, int]] = defaultdict(
        lambda: {"pairs": 0, "semantic_safe_pairs": 0}
    )
    for pair in pair_rows:
        labels = sorted(
            str(endpoint["focal_baseline_label"]) for endpoint in pair["endpoints"]
        )
        key = f"{'+'.join(labels)}__{pair['intervention_kind']}"
        strata[key]["pairs"] += 1
        strata[key]["semantic_safe_pairs"] += int(bool(pair["semantic_safe"]))
    unique_records = {
        endpoint["row_id"]: endpoint
        for call in schedule
        for endpoint in call["endpoints"]
    }
    coverage = {
        "observed_pairs": len(pair_rows),
        "endpoint_observations": len(endpoints),
        "unique_endpoint_rows": len(unique_records),
        "layers": len({int(row["layer"]) for row in schedule}),
        "documents": len(
            {
                endpoint["row_record"]["document_sha256"]
                for endpoint in unique_records.values()
            }
        ),
        "windows": len({endpoint["window_id"] for endpoint in unique_records.values()}),
        "layer_expert_cells": len(
            {(int(row["layer"]), int(row["expert_id"])) for row in schedule}
        ),
        "topk_ranks": sorted(
            {
                int(endpoint["topk_rank_zero_based"])
                if "topk_rank_zero_based" in endpoint
                else int(endpoint["row_record"]["route_rank"]) - 1
                for endpoint in unique_records.values()
            }
        ),
    }
    return {
        "target_coverage": coverage,
        "semantic_endpoint_safe_definition": (
            "M2-injected treatment versus fresh M1-injected baseline has no "
            "downstream target-token ordered-top-k change; greedy-token change is "
            "reported separately and does not tighten the frozen Oracle-B contract"
        ),
        "semantic_pair_safe_definition": "both endpoint semantic-safe flags are true",
        "endpoint_semantic_edges": endpoint_edges,
        "endpoint_semantic_edge_rate": endpoint_edges / len(endpoints),
        "route_edges": route_edges,
        "greedy_edges": greedy_edges,
        "final_logit_bitwise_edges": logit_edges,
        "sample_semantic_safe_edges": safe_pairs,
        "sample_safe_edge_density": safe_pairs / len(pair_rows),
        "maximum_safe_matching": maximum_safe_matching(pair_rows),
        "strata": {
            key: {
                **value,
                "safe_edge_density": value["semantic_safe_pairs"] / value["pairs"],
            }
            for key, value in sorted(strata.items())
        },
    }


def _measure_microcost(
    model: Any,
    schedule: Sequence[Mapping[str, Any]],
    rows_by_id: Mapping[str, Any],
) -> dict[str, Any]:
    """Measure the same 64 edges as two M1 calls versus one M2 call."""

    import torch

    prepared: list[tuple[Any, Any, tuple[Any, Any]]] = []
    for call in schedule:
        rows = [rows_by_id[row_id] for row_id in call["row_ids"]]
        batch = torch.stack(
            [row.tensor.to(device="cuda", dtype=torch.bfloat16) for row in rows]
        )
        expert = model.model.layers[int(call["layer"])].mlp.experts[
            int(call["expert_id"])
        ]
        prepared.append((expert, batch, (batch[0:1], batch[1:2])))

    def execute(surface: str) -> None:
        for expert, batch, singles in prepared:
            if surface == "m1":
                expert(singles[0])
                expert(singles[1])
            else:
                expert(batch)

    with torch.inference_mode():
        for _ in range(3):
            execute("m1")
            execute("m2")
        torch.cuda.synchronize()
    timings: dict[str, list[float]] = {"m1": [], "m2": []}
    with torch.inference_mode():
        for repeat in range(10):
            order = ("m1", "m2") if repeat % 2 == 0 else ("m2", "m1")
            for surface in order:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                execute(surface)
                end.record()
                end.synchronize()
                timings[surface].append(float(start.elapsed_time(end)))
    if any(value <= 0 for values in timings.values() for value in values):
        raise ShadowReplayError("microcost timing contains a non-positive value")
    return {
        "scope": "paired aggregate over the same 64 observed edges",
        "warmups": 3,
        "repeats": 10,
        "m1_two_single_calls_ms": timings["m1"],
        "m2_one_pair_call_ms": timings["m2"],
        "m1_median_ms": statistics.median(timings["m1"]),
        "m2_median_ms": statistics.median(timings["m2"]),
        "ratio_of_medians_m1_over_m2": statistics.median(timings["m1"])
        / statistics.median(timings["m2"]),
    }


def project_matching_microcost(
    matching: Mapping[str, Any], microcost: Mapping[str, Any]
) -> dict[str, Any]:
    observed_edges = int(matching["observed_edges"])
    vertices = int(matching["observed_vertices"])
    matched = int(matching["matching_edges"])
    if observed_edges <= 0 or vertices < 2 * matched:
        raise ShadowReplayError("matching projection denominator is invalid")
    m1_single_ms = float(microcost["m1_median_ms"]) / (2 * observed_edges)
    m2_pair_ms = float(microcost["m2_median_ms"]) / observed_edges
    all_m1_ms = vertices * m1_single_ms
    matched_ms = matched * m2_pair_ms + (vertices - 2 * matched) * m1_single_ms
    return {
        "boundary": (
            "linear microcost projection from the sampled 64-edge aggregate; "
            "not observed scheduler, serving, or end-to-end latency"
        ),
        "estimated_m1_single_call_ms": m1_single_ms,
        "estimated_m2_pair_call_ms": m2_pair_ms,
        "all_vertices_isolated_projection_ms": all_m1_ms,
        "maximum_safe_matching_projection_ms": matched_ms,
        "projected_saved_ms": all_m1_ms - matched_ms,
        "projected_saved_fraction": (
            (all_m1_ms - matched_ms) / all_m1_ms if all_m1_ms else 0.0
        ),
        "matched_pair_calls": matched,
        "unmatched_m1_calls": vertices - 2 * matched,
    }


def run_gpu(args: argparse.Namespace) -> int:
    import torch

    partner_dir = Path(args.partner_dir).resolve()
    source_dir = Path(args.source_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise ShadowReplayError(f"refusing existing output directory: {output_dir}")
    output_dir.mkdir(parents=True)
    schedule, source_digest = prepare_plan(partner_dir, source_dir)
    write_jsonl_no_overwrite(output_dir / "shadow_schedule.jsonl", schedule)

    config_path = partner_dir / "frozen_inputs" / "config.json"
    config = cross.PARTNER.validate_config(cross.PARTNER.load_json(config_path))
    cross.PARTNER.verify_source_artifacts(config, source_dir)
    evidence = cross.PARTNER.load_m2_evidence(config, source_dir)
    cross.PARTNER.load_and_verify_plan(
        config=config,
        config_path=config_path,
        plan_dir=partner_dir / "frozen_inputs",
        evidence=evidence,
    )
    old_references = cross.PARTNER.load_reference_hashes(config, source_dir)
    original_config, acceptance, model = cross.PARTNER._load_live_model(
        config=config,
        source_dir=source_dir,
        acceptance_path=partner_dir / "frozen_inputs" / "ACCEPTANCE.json",
        model_path_override=args.model_path,
    )
    gpu = cross.PARTNER.PILOT._gpu()
    capture_path = source_dir / "calibration_captures.pt"
    if sha256_file(capture_path) != config["source"]["calibration_captures_sha256"]:
        raise ShadowReplayError("calibration capture hash differs from frozen source")
    captures = torch.load(capture_path, map_location="cpu", weights_only=False)
    rows = gpu.materialize_routed_rows(captures)
    rows_by_id = {row.row_id: row for row in rows}
    captures_by_window = {capture.window_id: capture for capture in captures}
    _resolve_schedule(
        schedule,
        evidence=evidence,
        rows_by_id=rows_by_id,
        captures_by_window=captures_by_window,
    )
    for call in schedule:
        call["_endpoint_calls"] = [
            _endpoint_call(
                call,
                endpoint_index,
                captures_by_window[endpoint["window_id"]],
            )
            for endpoint_index, endpoint in enumerate(call["endpoints"])
        ]

    resolved_public = [
        {key: value for key, value in call.items() if not key.startswith("_")}
        for call in schedule
    ]
    write_jsonl_no_overwrite(output_dir / "resolved_shadow_schedule.jsonl", resolved_public)
    replacements, side_ledger = _side_calls(
        model, schedule, rows_by_id, old_references
    )
    write_jsonl_no_overwrite(output_dir / "side_call_ledger.jsonl", side_ledger)

    first_by_endpoint: dict[str, Mapping[str, Any]] = {}
    for call in schedule:
        for endpoint_call in call["_endpoint_calls"]:
            first_by_endpoint.setdefault(
                str(endpoint_call["focal_row_id"]), endpoint_call
            )
    runtime_baselines: dict[str, dict[str, Any]] = {}
    baseline_public: list[dict[str, Any]] = []
    for row_id, endpoint_call in sorted(first_by_endpoint.items()):
        runtime, public = _native_and_noop(model, endpoint_call)
        runtime_baselines[row_id] = runtime
        baseline_public.append(public)
    write_jsonl_no_overwrite(output_dir / "target_baselines.jsonl", baseline_public)

    pair_rows: list[dict[str, Any]] = []
    for call in schedule:
        endpoint_rows: list[dict[str, Any]] = []
        m2_pair = replacements["m2_by_call"][int(call["call_index"])]
        for endpoint_index, endpoint_call in enumerate(call["_endpoint_calls"]):
            row_id = str(endpoint_call["focal_row_id"])
            endpoint_rows.append(
                _run_intervention(
                    model,
                    endpoint_call,
                    replacements["m1_by_row"][row_id],
                    m2_pair[endpoint_index],
                    runtime_baselines[row_id],
                )
            )
        pair_rows.append(
            {
                "call_index": int(call["call_index"]),
                "row_ids": list(call["row_ids"]),
                "intervention_kind": call["intervention_kind"],
                "endpoints": endpoint_rows,
                "semantic_safe": all(
                    endpoint_route_topk_safe(row) for row in endpoint_rows
                ),
            }
        )
    write_jsonl_no_overwrite(output_dir / "pair_results.jsonl", pair_rows)
    aggregate = aggregate_results(schedule, pair_rows)
    microcost = _measure_microcost(model, schedule, rows_by_id)
    projection = project_matching_microcost(
        aggregate["maximum_safe_matching"], microcost
    )
    result = {
        "schema_version": RESULT_SCHEMA,
        "status": "COMPLETE_EXPLORATORY_SHADOW_REPLAY",
        "decision": "REPORT_SEMANTIC_ORACLE_SHADOW_EDGES",
        "paper_result": False,
        "evidence_boundary": (
            "single_rtx5090_reused_run03_calibration_windows_expert_contribution_"
            "m2_semantic_shadow_replay_not_fresh_evaluation_not_serving_not_ep"
        ),
        "source": {
            "partner_schedule_sha256": source_digest,
            "calibration_captures_sha256": sha256_file(capture_path),
            "stack_digest": acceptance["stack"]["stack_digest"],
            "model_config_sha256": canonical_sha256(original_config),
        },
        "execution": {
            **semantic_surface_contract(),
            "native_noop_required": True,
            "single_target_contribution_replacement": True,
        },
        "sample_microcost_latency": microcost,
        "maximum_matching_microcost_projection": projection,
        **aggregate,
    }
    write_json_no_overwrite(output_dir / "SEMANTIC_ORACLE_RESULT.json", result)
    complete = {
        "schema_version": COMPLETE_SCHEMA,
        "status": "SUCCESS_COMPLETE",
        "artifact_sha256": {
            name: sha256_file(output_dir / name)
            for name in (
                "shadow_schedule.jsonl",
                "resolved_shadow_schedule.jsonl",
                "side_call_ledger.jsonl",
                "target_baselines.jsonl",
                "pair_results.jsonl",
                "SEMANTIC_ORACLE_RESULT.json",
            )
        },
    }
    write_json_no_overwrite(output_dir / "COMPLETE.json", complete)
    cross.PARTNER.PILOT.assert_clean_gpu(
        acceptance["stack"]["gpu"]["uuid"], allowed_pids={os.getpid()}
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan = subparsers.add_parser("plan", help="write the CPU-only replay plan")
    plan.add_argument("--partner-dir", default=str(DEFAULT_PARTNER_DIR))
    plan.add_argument("--source-dir", default=str(DEFAULT_SOURCE_DIR))
    plan.add_argument("--output", required=True)
    plan.set_defaults(func=run_plan)
    run = subparsers.add_parser("run", help="run the accepted RTX 5090 shadow replay")
    run.add_argument("--partner-dir", default=str(DEFAULT_PARTNER_DIR))
    run.add_argument("--source-dir", default=str(DEFAULT_SOURCE_DIR))
    run.add_argument("--output-dir", required=True)
    run.add_argument("--model-path")
    run.set_defaults(func=run_gpu)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
