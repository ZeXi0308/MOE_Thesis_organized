#!/usr/bin/env python3
"""Exploratory RouteSnap oracle curve on the frozen SemanticFence shadow cohort.

The experiment reuses the 64 single-contribution M1/M2 interventions from the
completed SemanticFence semantic shadow replay.  It changes only downstream
router semantics: router logits are snapped to a fixed absolute lattice with a
deterministic expert-id tie break.  Expert execution and all non-target
contributions remain native.

This is a reused-cohort mechanism-feasibility probe.  It is not fresh held-out
evidence, a serving result, a quality result, or a proof of batch invariance.
"""

from __future__ import annotations

import argparse
import contextlib
from collections import defaultdict
import json
import os
from pathlib import Path
import sys
import types
from typing import Any, Iterable, Mapping, Sequence


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
SEMANTIC_DIR = REPO_ROOT / "docs" / "ideas" / "semanticfence" / "experiments"
if str(SEMANTIC_DIR) not in sys.path:
    sys.path.insert(0, str(SEMANTIC_DIR))

import run_semantic_oracle_shadow_replay_5090 as shadow  # noqa: E402


DEFAULT_PARTNER_DIR = (
    SEMANTIC_DIR / "outputs" / "partner_permutation_20260810_run01"
)
DEFAULT_SOURCE_DIR = (
    SEMANTIC_DIR / "outputs" / "semanticfence_pilot_20260810_run03"
)
DEFAULT_SHADOW_DIR = (
    SEMANTIC_DIR / "outputs" / "semantic_oracle_shadow_20260810_run01"
)
DEFAULT_STEPS = (
    2.0**-12,
    2.0**-11,
    2.0**-10,
    2.0**-9,
    2.0**-8,
    2.0**-7,
    2.0**-6,
    2.0**-5,
    2.0**-4,
)
RESULT_SCHEMA = "routesnap-shadow-result-v1"
ENDPOINT_SCHEMA = "routesnap-shadow-endpoint-v1"


class RouteSnapError(RuntimeError):
    """The exploratory RouteSnap result cannot be interpreted."""


def parse_steps(value: str) -> tuple[float, ...]:
    steps = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not steps or any(step <= 0.0 for step in steps):
        raise argparse.ArgumentTypeError("steps must be a non-empty positive list")
    if len(set(steps)) != len(steps):
        raise argparse.ArgumentTypeError("steps must be unique")
    return tuple(sorted(steps))


def snap_logits(logits: Any, step: float) -> Any:
    """Return float32 lattice logits with an explicit expert-id tie break.

    The maximum tie offset is below one quarter of a lattice step, so it cannot
    reverse two values that landed in different lattice bins.  Smaller expert
    ids win exact-bin ties.
    """

    import torch

    if float(step) <= 0.0:
        raise ValueError("step must be positive")
    values = logits.float()
    snapped = torch.round(values / float(step)) * float(step)
    expert_count = int(snapped.shape[-1])
    offsets = -torch.arange(
        expert_count, device=snapped.device, dtype=torch.float32
    ) * (float(step) / (4.0 * max(expert_count, 1)))
    return snapped + offsets


@contextlib.contextmanager
def patched_downstream_routers(model: Any, *, after_layer: int, step: float):
    """Snap every router strictly downstream of the injected contribution."""

    originals: list[tuple[Any, Any]] = []
    trace: dict[str, Any] = {
        "after_layer": int(after_layer),
        "step": float(step),
        "patched_layers": [],
        "calls_by_layer": defaultdict(int),
    }
    try:
        for layer_index in range(int(after_layer) + 1, len(model.model.layers)):
            gate = model.model.layers[layer_index].mlp.gate
            original = gate.forward

            def snapped_forward(
                this: Any,
                hidden: Any,
                *args: Any,
                _original: Any = original,
                _layer: int = layer_index,
                **kwargs: Any,
            ) -> Any:
                raw = _original(hidden, *args, **kwargs)
                trace["calls_by_layer"][_layer] += 1
                return snap_logits(raw, float(step))

            originals.append((gate, original))
            gate.forward = types.MethodType(snapped_forward, gate)
            trace["patched_layers"].append(layer_index)
        yield trace
    finally:
        for gate, original in reversed(originals):
            gate.forward = original
        trace["calls_by_layer"] = {
            str(key): int(value)
            for key, value in sorted(trace["calls_by_layer"].items())
        }


def route_distance(
    left: Sequence[Sequence[int]],
    right: Sequence[Sequence[int]],
    *,
    start_layer: int,
) -> int:
    if len(left) != len(right):
        raise RouteSnapError("route traces have different layer counts")
    return sum(
        int(list(map(int, left[layer])) != list(map(int, right[layer])))
        for layer in range(int(start_layer), len(left))
    )


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def raw_edge_index(shadow_dir: Path) -> dict[tuple[int, int], bool]:
    result: dict[tuple[int, int], bool] = {}
    for pair in load_jsonl(Path(shadow_dir) / "pair_results.jsonl"):
        call_index = int(pair["call_index"])
        for endpoint in pair["endpoints"]:
            key = (call_index, int(endpoint["endpoint_index"]))
            if key in result:
                raise RouteSnapError(f"duplicate historical endpoint {key}")
            result[key] = bool(endpoint["route_delta"]["any_ordered_topk_change"])
    if len(result) != 64:
        raise RouteSnapError(f"historical shadow has {len(result)} endpoints, expected 64")
    return result


def run_snapped_surface(
    model: Any,
    call: Mapping[str, Any],
    replacement: Any,
    baseline: Mapping[str, Any],
    *,
    step: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    native = baseline["observation"]
    noop_trace = baseline["trace"]
    replacement_hash = shadow.stable.tensor_sha256(replacement)
    with patched_downstream_routers(
        model, after_layer=int(call["layer"]), step=float(step)
    ) as snap_trace:
        with shadow.stable.patched_single_contribution(
            model, baseline["identity"], replacement, "replacement"
        ) as intervention_trace:
            observation = shadow.stable.run_observation(
                model,
                baseline["input_ids"],
                baseline["config"],
                baseline["identity"],
            )
    if (
        int(intervention_trace["pair_match_count"]) != 1
        or intervention_trace["target_input_sha256"]
        != call["target_row_record"]["hidden_sha256"]
        or intervention_trace["target_applied_raw_sha256"] != replacement_hash
        or intervention_trace["target_native_raw_sha256"]
        != noop_trace["target_native_raw_sha256"]
        or intervention_trace["non_target_contributions_sha256"]
        != noop_trace["non_target_contributions_sha256"]
    ):
        raise RouteSnapError("snapped surface did not isolate one contribution")
    for layer in range(int(call["layer"]) + 1):
        if (
            observation["router_logits_sha256_by_layer"][layer]
            != native["router_logits_sha256_by_layer"][layer]
        ):
            raise RouteSnapError("RouteSnap changed a pre-intervention router")
    return observation, {
        "snap": dict(snap_trace),
        "intervention": dict(intervention_trace),
    }


def endpoint_result(
    model: Any,
    call: Mapping[str, Any],
    m1_replacement: Any,
    m2_replacement: Any,
    baseline: Mapping[str, Any],
    *,
    step: float,
    raw_edge: bool,
) -> dict[str, Any]:
    m1, m1_trace = run_snapped_surface(
        model, call, m1_replacement, baseline, step=step
    )
    m2, m2_trace = run_snapped_surface(
        model, call, m2_replacement, baseline, step=step
    )
    start_layer = int(call["layer"]) + 1
    snapped_distance = route_distance(
        m1["topk_experts_by_layer"],
        m2["topk_experts_by_layer"],
        start_layer=start_layer,
    )
    m1_native_distance = route_distance(
        baseline["observation"]["topk_experts_by_layer"],
        m1["topk_experts_by_layer"],
        start_layer=start_layer,
    )
    snapped_edge = bool(snapped_distance)
    return {
        "schema_version": ENDPOINT_SCHEMA,
        "call_index": int(call["call_index"]),
        "endpoint_index": int(call["endpoint_index"]),
        "row_id": str(call["focal_row_id"]),
        "layer": int(call["layer"]),
        "expert_id": int(call["expert_id"]),
        "step": float(step),
        "raw_edge": bool(raw_edge),
        "snapped_edge": snapped_edge,
        "healed": bool(raw_edge and not snapped_edge),
        "induced": bool(not raw_edge and snapped_edge),
        "snapped_route_distance": int(snapped_distance),
        "m1_route_drift_vs_native": int(m1_native_distance),
        "m1_greedy_drift_vs_native": bool(
            int(m1["greedy_token_id"])
            != int(baseline["observation"]["greedy_token_id"])
        ),
        "m1_m2_greedy_changed": bool(
            int(m1["greedy_token_id"]) != int(m2["greedy_token_id"])
        ),
        "m1_final_logits_vs_native": shadow._tensor_difference_metrics(
            baseline["observation"]["_final_logits_cpu"], m1["_final_logits_cpu"]
        ),
        "m2_final_logits_vs_m1": shadow._tensor_difference_metrics(
            m1["_final_logits_cpu"], m2["_final_logits_cpu"]
        ),
        "m1_trace": m1_trace,
        "m2_trace": m2_trace,
    }


def aggregate(rows: Sequence[Mapping[str, Any]], steps: Sequence[float]) -> dict[str, Any]:
    by_step: dict[float, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_step[float(row["step"])].append(row)
    summaries: list[dict[str, Any]] = []
    for step in steps:
        cohort = by_step[float(step)]
        if len(cohort) != 64:
            raise RouteSnapError(f"step {step} has {len(cohort)} endpoints, expected 64")
        raw_edges = sum(int(row["raw_edge"]) for row in cohort)
        healed = sum(int(row["healed"]) for row in cohort)
        induced = sum(int(row["induced"]) for row in cohort)
        m1_route_drift = sum(int(int(row["m1_route_drift_vs_native"]) > 0) for row in cohort)
        m1_greedy_drift = sum(int(row["m1_greedy_drift_vs_native"]) for row in cohort)
        summaries.append(
            {
                "step": float(step),
                "endpoints": len(cohort),
                "raw_edges": raw_edges,
                "snapped_edges": sum(int(row["snapped_edge"]) for row in cohort),
                "healed": healed,
                "induced": induced,
                "net_healed": healed - induced,
                "healed_fraction_of_raw_edges": healed / raw_edges if raw_edges else 0.0,
                "m1_route_drift_endpoints_vs_native": m1_route_drift,
                "m1_greedy_drift_endpoints_vs_native": m1_greedy_drift,
                "m1_m2_greedy_changed_endpoints": sum(
                    int(row["m1_m2_greedy_changed"]) for row in cohort
                ),
                "exploratory_go": bool(
                    raw_edges > 0
                    and healed / raw_edges >= 0.50
                    and healed > induced
                    and m1_greedy_drift == 0
                ),
            }
        )
    go_steps = [row["step"] for row in summaries if row["exploratory_go"]]
    return {
        "steps": summaries,
        "decision": (
            "ROUTESNAP_FEASIBILITY_SIGNAL" if go_steps else "STOP_SIMPLE_ROUTESNAP"
        ),
        "exploratory_go_steps": go_steps,
        "gate": {
            "healed_fraction_of_raw_edges_min": 0.50,
            "net_healed_strictly_positive": True,
            "m1_greedy_drift_endpoints_vs_native_max": 0,
            "quality_and_serving_followup_required": True,
        },
    }


def run_gpu(args: argparse.Namespace) -> int:
    import torch

    partner_dir = Path(args.partner_dir).resolve()
    source_dir = Path(args.source_dir).resolve()
    shadow_dir = Path(args.shadow_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise RouteSnapError(f"refusing existing output directory: {output_dir}")
    output_dir.mkdir(parents=True)

    schedule = load_jsonl(shadow_dir / "resolved_shadow_schedule.jsonl")
    historical = raw_edge_index(shadow_dir)
    config_path = partner_dir / "frozen_inputs" / "config.json"
    config = shadow.cross.PARTNER.validate_config(
        shadow.cross.PARTNER.load_json(config_path)
    )
    shadow.cross.PARTNER.verify_source_artifacts(config, source_dir)
    evidence = shadow.cross.PARTNER.load_m2_evidence(config, source_dir)
    shadow.cross.PARTNER.load_and_verify_plan(
        config=config,
        config_path=config_path,
        plan_dir=partner_dir / "frozen_inputs",
        evidence=evidence,
    )
    old_references = shadow.cross.PARTNER.load_reference_hashes(config, source_dir)
    original_config, acceptance, model = shadow.cross.PARTNER._load_live_model(
        config=config,
        source_dir=source_dir,
        acceptance_path=partner_dir / "frozen_inputs" / "ACCEPTANCE.json",
        model_path_override=args.model_path,
    )
    gpu = shadow.cross.PARTNER.PILOT._gpu()
    capture_path = source_dir / "calibration_captures.pt"
    if shadow.sha256_file(capture_path) != config["source"]["calibration_captures_sha256"]:
        raise RouteSnapError("calibration capture hash differs from frozen source")
    captures = torch.load(capture_path, map_location="cpu", weights_only=False)
    routed_rows = gpu.materialize_routed_rows(captures)
    rows_by_id = {row.row_id: row for row in routed_rows}
    captures_by_window = {capture.window_id: capture for capture in captures}
    shadow._resolve_schedule(
        schedule,
        evidence=evidence,
        rows_by_id=rows_by_id,
        captures_by_window=captures_by_window,
    )
    for call in schedule:
        call["_endpoint_calls"] = [
            shadow._endpoint_call(
                call,
                endpoint_index,
                captures_by_window[endpoint["window_id"]],
            )
            for endpoint_index, endpoint in enumerate(call["endpoints"])
        ]

    replacements, side_ledger = shadow._side_calls(
        model, schedule, rows_by_id, old_references
    )
    shadow.write_jsonl_no_overwrite(output_dir / "side_call_ledger.jsonl", side_ledger)

    baselines: dict[tuple[int, int], dict[str, Any]] = {}
    baseline_public: list[dict[str, Any]] = []
    for call in schedule:
        for endpoint_call in call["_endpoint_calls"]:
            key = (int(endpoint_call["call_index"]), int(endpoint_call["endpoint_index"]))
            runtime, public = shadow._native_and_noop(model, endpoint_call)
            baselines[key] = runtime
            baseline_public.append(public)
    if set(baselines) != set(historical):
        raise RouteSnapError("runtime endpoints differ from historical shadow")
    shadow.write_jsonl_no_overwrite(output_dir / "target_baselines.jsonl", baseline_public)

    endpoint_rows: list[dict[str, Any]] = []
    for step in args.steps:
        for call in schedule:
            m2_pair = replacements["m2_by_call"][int(call["call_index"])]
            for endpoint_index, endpoint_call in enumerate(call["_endpoint_calls"]):
                key = (int(call["call_index"]), endpoint_index)
                row_id = str(endpoint_call["focal_row_id"])
                endpoint_rows.append(
                    endpoint_result(
                        model,
                        endpoint_call,
                        replacements["m1_by_row"][row_id],
                        m2_pair[endpoint_index],
                        baselines[key],
                        step=float(step),
                        raw_edge=historical[key],
                    )
                )
    shadow.write_jsonl_no_overwrite(output_dir / "endpoint_results.jsonl", endpoint_rows)

    result = {
        "schema_version": RESULT_SCHEMA,
        "status": "COMPLETE_EXPLORATORY_REUSED_COHORT",
        "paper_result": False,
        "evidence_boundary": (
            "single_rtx5090_reused_semanticfence_calibration_shadow_"
            "single_contribution_not_fresh_not_serving_not_quality"
        ),
        "mechanism": {
            "selection_logits": "absolute_lattice_round_plus_expert_id_tie_break",
            "mixing_weights": "snapped_logits_in_this_pilot",
            "patched_scope": "all_tokens_all_routers_strictly_after_intervention_layer",
            "steps": list(map(float, args.steps)),
        },
        "source": {
            "calibration_captures_sha256": shadow.sha256_file(capture_path),
            "historical_pair_results_sha256": shadow.sha256_file(
                shadow_dir / "pair_results.jsonl"
            ),
            "stack_digest": acceptance["stack"]["stack_digest"],
            "model_config_sha256": shadow.canonical_sha256(original_config),
        },
        **aggregate(endpoint_rows, args.steps),
    }
    shadow.write_json_no_overwrite(output_dir / "ROUTESNAP_RESULT.json", result)
    shadow.cross.PARTNER.PILOT.assert_clean_gpu(
        acceptance["stack"]["gpu"]["uuid"], allowed_pids={os.getpid()}
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partner-dir", default=str(DEFAULT_PARTNER_DIR))
    parser.add_argument("--source-dir", default=str(DEFAULT_SOURCE_DIR))
    parser.add_argument("--shadow-dir", default=str(DEFAULT_SHADOW_DIR))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-path")
    parser.add_argument(
        "--steps",
        type=parse_steps,
        default=DEFAULT_STEPS,
        help="comma-separated positive absolute lattice steps",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(run_gpu(args))


if __name__ == "__main__":
    raise SystemExit(main())
