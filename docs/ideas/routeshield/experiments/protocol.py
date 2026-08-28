from __future__ import annotations

"""Readiness and result-contract evaluation for RouteShield Gate-0."""

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

try:
    from .schema import ProtocolError
except ImportError:
    from schema import ProtocolError


# Hash-bound raw paired-block recomputation exists only as a development harness.
# Formal readiness requires a reviewed code change after full-DAG, executed-dispatch,
# tensor-exactness, and exact-Oracle certificate validators close.
FORMAL_RAW_EVALUATOR_IMPLEMENTED = False
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
UNRESOLVED = re.compile(r"^UNRESOLVED_[A-Z0-9_]+$")

FROZEN_MODELS = (
    (
        "olmoe",
        "allenai/OLMoE-1B-7B-0924",
        "6d84c48581ece794365f2b8e9cfb043c68ade9c5",
        "bfloat16",
        64,
        8,
    ),
    (
        "llmjp",
        "llm-jp/optimal-sparsity-math-d512-E32-k16-920M-A520M",
        "1d5983076dfc67aee4a77ec06a27027f5bab6055",
        "bfloat16",
        32,
        16,
    ),
)
FROZEN_THRESHOLDS = {
    "harm_point_min": 0.2,
    "harm_lcb_strict_min": 0.1,
    "oracle_gain_point_min": 0.1,
    "oracle_gain_lcb_strict_min": 0.05,
    "oracle_recovery_lcb_min": 0.5,
    "simple_capture_ucb_strict_max": 0.9,
    "benign_goodput_loss_ucb_strict_max": 0.05,
    "future_policy_residual_lcb_strict_min": 0.05,
}
CONFIG_TOP_LEVEL_FIELDS = {
    "schema",
    "frozen_at",
    "status",
    "formal_execution_authorized",
    "authority_boundary",
    "claim_scope",
    "models",
    "threat_model",
    "workloads",
    "target_system",
    "required_evidence",
    "causal_observation",
    "baselines",
    "baseline_selection",
    "counterfactuals",
    "statistics",
    "exactness",
    "allowed_statuses",
}
WORKLOAD_FIELDS = {
    "traffic_classes",
    "prompt_lengths",
    "route_census_min_independent_prompts_per_cell",
    "formal_p99_min_paired_blocks_per_cell",
    "formal_p99_min_completed_victim_requests_per_cell",
    "calibration_manifest_sha256",
    "sealed_evaluation_manifest_sha256",
    "historical_prompt_registry_sha256",
    "natural_dataset_revision",
    "structured_benign_dataset_revision",
    "attack_generator_source_sha256",
    "sealed_evaluation_opened",
}
TARGET_SYSTEM_FIELDS = {
    "primary_load_fraction",
    "negative_control_load_fraction",
    "queue_must_be_stable",
    "engine",
    "expert_backend",
    "ep_size",
    "placement_id",
    "placement_snapshot_path",
    "placement_snapshot_sha256",
    "arrival_trace_sha256",
    "chunked_prefill_config_sha256",
}
REQUIRED_EVIDENCE_FIELDS = {
    "raw_request_ledger_sha256",
    "raw_block_ledger_sha256",
    "tenant_qualified_route_producer_sha256",
    "tenant_route_ledger_sha256",
    "route_census_summary_sha256",
    "expected_route_event_manifest_path",
    "expected_route_event_manifest_sha256",
    "prompt_bytes_token_ids_manifest_sha256",
    "native_continuous_prefill",
    "full_request_dag_replay_sha256",
    "exact_legal_oracle_sha256",
    "service_surface_manifest_sha256",
    "service_weighted_causal_ledger_sha256",
    "full_path_denominator_manifest_sha256",
    "exactness_manifest_sha256",
    "baseline_bundle_sha256",
}
STATISTICS_FIELDS = {
    "bootstrap",
    "bootstrap_resamples",
    "bootstrap_seed",
    "empirical_p99",
    "bootstrap_interval",
    "confidence_level",
    "no_cross_model_pooling",
    "queue_growth_max_fraction",
    "goodput_definition",
    "goodput_window",
    "goodput_aggregation",
    "thresholds",
}
FROZEN_BASELINES = (
    "production_default_fcfs_chunked_prefill",
    "per_tenant_request_concurrency_quota",
    "per_tenant_input_token_quota",
    "exact_repetition_detector",
    "ppl_filter_with_structured_benign_false_positive",
    "vtc_token_cost_fairness",
    "fairserve_weighted_service_counter",
    "expert_drr_drf",
    "physical_rank_drr_drf",
    "fixed_tenant_capacity_partition",
    "vulnerability_aware_placement",
    "dynamic_eplb_snapshot",
)
FROZEN_ALLOWED_STATUSES = (
    "BLOCKED_PROTOCOL_NOT_AUTHORIZED",
    "BLOCKED_MISSING_FORMAL_EVIDENCE",
    "INVALID_CONFIG",
    "INVALID_ARTIFACT",
    "INVALID_REQUEST_DAG",
    "UNSOLVED_EXACT_STATE_LIMIT",
    "NO_GO_PHENOMENON",
    "NO_GO_ORACLE",
    "SIMPLE_BASELINE_WINS",
    "NO_GO_BATCHING_TAX",
    "MODEL_SPECIFIC_OBSERVATION",
    "UNTRUSTED_AGGREGATE_SHAPE_ONLY",
    "RAW_RECOMPUTE_DIAGNOSTIC_ONLY",
    "RAW_RECOMPUTE_SMOKE_ONLY",
    "BLOCKED_INSUFFICIENT_RAW_SAMPLE",
    "BLOCKED_FORMAL_RAW_EVALUATOR_NOT_APPROVED",
    "QUALIFIED_FOR_8XA100_EXISTENCE_GATE",
    "SMOKE_ONLY",
)


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            raise ProtocolError(f"duplicate config JSON key: {key}")
        output[key] = value
    return output


def _reject_json_constant(value: str) -> None:
    raise ProtocolError(f"non-finite config JSON value is forbidden: {value}")


def _walk_items(value: object, *, path: str = "") -> Iterable[tuple[str, str, object]]:
    if isinstance(value, Mapping):
        for key, item in value.items():
            child = f"{path}.{key}" if path else str(key)
            yield child, str(key), item
            yield from _walk_items(item, path=child)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _walk_items(item, path=f"{path}[{index}]")


def _mapping_with_fields(
    config: Mapping[str, Any], key: str, fields: set[str]
) -> Mapping[str, Any]:
    value = config.get(key)
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ProtocolError(f"{key} fields changed; schema version bump is required")
    return value


def validate_config(config: Mapping[str, Any]) -> None:
    if set(config) != CONFIG_TOP_LEVEL_FIELDS:
        raise ProtocolError(
            "config top-level fields changed; schema version bump is required"
        )
    if config.get("schema") != "routeshield-gate0-v1":
        raise ProtocolError("unsupported RouteShield config schema")
    if config.get("frozen_at") != "2026-07-29":
        raise ProtocolError("formal v1 freeze date changed")
    if config.get("authority_boundary") != (
        "Candidate qualification protocol only; does not replace docs/current/README.md"
    ):
        raise ProtocolError("formal v1 authority boundary changed")

    claim_scope = _mapping_with_fields(
        config,
        "claim_scope",
        {
            "primary_phase",
            "primary_metric",
            "single_gpu_metric_name",
            "secondary_metrics",
            "maximum_gate0_verdict",
            "forbidden_claims",
        },
    )
    if claim_scope != {
        "primary_phase": "prefill",
        "primary_metric": "victim_request_ttft_p99",
        "single_gpu_metric_name": "REPLAYED_TTFT_P99",
        "secondary_metrics": [
            "victim_request_tpot_p99_exploratory_only",
            "per_tenant_goodput",
            "total_goodput",
            "queue_stability",
        ],
        "maximum_gate0_verdict": "QUALIFIED_FOR_8XA100_EXISTENCE_GATE",
        "forbidden_claims": [
            "RTX5090_SYSTEM_P99",
            "RTX5090_EP_ISOLATION",
            "PROVABLE_WORST_CASE_BOUND",
            "PRODUCTION_GO",
        ],
    }:
        raise ProtocolError("formal v1 claim scope changed")

    threat_model = _mapping_with_fields(
        config,
        "threat_model",
        {
            "attacker_identity",
            "sybil_accounts",
            "capabilities",
            "forbidden_capabilities",
            "matched_budget",
            "policy_constraints",
        },
    )
    if threat_model != {
        "attacker_identity": "single_authenticated_tenant",
        "sybil_accounts": "out_of_scope",
        "capabilities": ["black_box_text_api"],
        "forbidden_capabilities": [
            "direct_route_injection",
            "router_weight_access",
            "placement_knowledge",
            "privileged_backend_state",
        ],
        "matched_budget": {
            "same_request_count": True,
            "same_arrivals": True,
            "same_input_token_count": True,
            "max_new_tokens": 1,
        },
        "policy_constraints": {
            "drop_attacker": False,
            "unbounded_starvation": False,
            "change_router_or_topk": False,
            "change_expert_identity_or_weights": False,
            "change_output_tokens": False,
        },
    }:
        raise ProtocolError("formal v1 threat model changed")
    matched_budget = threat_model["matched_budget"]
    policy_constraints = threat_model["policy_constraints"]
    if (
        not isinstance(matched_budget, Mapping)
        or type(matched_budget.get("same_request_count")) is not bool
        or type(matched_budget.get("same_arrivals")) is not bool
        or type(matched_budget.get("same_input_token_count")) is not bool
        or type(matched_budget.get("max_new_tokens")) is not int
        or not isinstance(policy_constraints, Mapping)
        or any(type(value) is not bool for value in policy_constraints.values())
    ):
        raise ProtocolError("formal v1 threat-model JSON types changed")
    models = config.get("models")
    if not isinstance(models, list) or len(models) != 2:
        raise ProtocolError("Gate-0 requires exactly two frozen model entries")
    observed_models = []
    for row in models:
        if not isinstance(row, Mapping):
            raise ProtocolError("model entries must be objects")
        if set(row) != {
            "key",
            "model_id",
            "revision",
            "tokenizer_sha256",
            "dtype",
            "num_experts",
            "top_k",
        }:
            raise ProtocolError("frozen model entry fields changed")
        if (
            type(row.get("num_experts")) is not int
            or type(row.get("top_k")) is not int
        ):
            raise ProtocolError("model expert/top-k counts must be JSON integers")
        observed_models.append(
            (
                row.get("key"),
                row.get("model_id"),
                row.get("revision"),
                row.get("dtype"),
                row.get("num_experts"),
                row.get("top_k"),
            )
        )
    if tuple(observed_models) != FROZEN_MODELS:
        raise ProtocolError("routeshield-gate0-v1 requires the two frozen model tuples")
    if any(not HEX40.fullmatch(str(row[2])) for row in observed_models):
        raise ProtocolError("model revisions must be lowercase 40-hex commits")

    workloads = _mapping_with_fields(config, "workloads", WORKLOAD_FIELDS)
    if workloads.get("traffic_classes") != [
        "NAT_BENIGN",
        "NAT_PATHOLOGICAL",
        "ADV_TEXT",
        "SYN_ROUTE",
    ]:
        raise ProtocolError("formal v1 traffic classes changed")
    if workloads.get("prompt_lengths") != [512, 2048]:
        raise ProtocolError("formal v1 prompt lengths changed")
    if (
        type(workloads.get("route_census_min_independent_prompts_per_cell"))
        is not int
        or workloads.get("route_census_min_independent_prompts_per_cell") != 128
    ):
        raise ProtocolError("formal v1 route-census sample floor changed")
    if (
        type(workloads.get("formal_p99_min_paired_blocks_per_cell")) is not int
        or workloads.get("formal_p99_min_paired_blocks_per_cell") != 30
    ):
        raise ProtocolError("formal v1 paired-block sample floor changed")
    if (
        type(workloads.get("formal_p99_min_completed_victim_requests_per_cell"))
        is not int
        or workloads.get("formal_p99_min_completed_victim_requests_per_cell")
        != 10000
    ):
        raise ProtocolError("formal v1 victim-request sample floor changed")
    if type(workloads.get("sealed_evaluation_opened")) is not bool:
        raise ProtocolError("sealed_evaluation_opened must be a JSON boolean")

    target = _mapping_with_fields(config, "target_system", TARGET_SYSTEM_FIELDS)
    if (
        target.get("primary_load_fraction") != 0.7
        or target.get("negative_control_load_fraction") != 0.3
        or type(target.get("ep_size")) is not int
        or target.get("ep_size") != 8
        or target.get("queue_must_be_stable") is not True
    ):
        raise ProtocolError("formal v1 target load/EP/queue contract changed")

    required_evidence = _mapping_with_fields(
        config, "required_evidence", REQUIRED_EVIDENCE_FIELDS
    )
    if type(required_evidence.get("native_continuous_prefill")) is not bool:
        raise ProtocolError("native_continuous_prefill must be a JSON boolean")

    causal = _mapping_with_fields(
        config,
        "causal_observation",
        {
            "prefix_contribution_fraction",
            "minimum_remaining_service_work_fraction_at_action",
            "future_routes_visible_to_online_policy",
            "future_arrivals_visible_to_online_policy",
            "future_service_visible_to_online_policy",
        },
    )
    if causal != {
        "prefix_contribution_fraction": 0.25,
        "minimum_remaining_service_work_fraction_at_action": 0.5,
        "future_routes_visible_to_online_policy": False,
        "future_arrivals_visible_to_online_policy": False,
        "future_service_visible_to_online_policy": False,
    }:
        raise ProtocolError("formal v1 causal-observation contract changed")
    if any(
        type(causal[key]) is not bool
        for key in (
            "future_routes_visible_to_online_policy",
            "future_arrivals_visible_to_online_policy",
            "future_service_visible_to_online_policy",
        )
    ):
        raise ProtocolError("formal v1 causal-observation JSON types changed")

    baselines = config.get("baselines")
    if not isinstance(baselines, list) or tuple(baselines) != FROZEN_BASELINES:
        raise ProtocolError("formal v1 baseline registry changed")
    selection = _mapping_with_fields(
        config,
        "baseline_selection",
        {
            "selection_split",
            "selection_manifest_sha256",
            "frozen_strongest_simple_by_model",
        },
    )
    selected = selection.get("frozen_strongest_simple_by_model")
    if selection.get("selection_split") != "calibration" or not isinstance(
        selected, Mapping
    ) or set(selected) != {"olmoe", "llmjp"}:
        raise ProtocolError("formal v1 strongest-simple selection contract changed")
    for model, policy in selected.items():
        if not isinstance(policy, str) or not (
            UNRESOLVED.fullmatch(policy) or policy in FROZEN_BASELINES
        ):
            raise ProtocolError(f"invalid frozen strongest-simple policy for {model}")

    counterfactuals = _mapping_with_fields(
        config,
        "counterfactuals",
        {
            "delete_attacker",
            "legal_oracle",
            "oracle_state_limit_behavior",
            "legal_oracle_objective",
            "legal_oracle_scope",
            "delete_attacker_used_for_policy_capture",
        },
    )
    if counterfactuals != {
        "delete_attacker": "attribution_only_not_policy_oracle",
        "legal_oracle": "future_known_exact_all_work_retained_real_action_space",
        "oracle_state_limit_behavior": "UNSOLVED_EXACT_STATE_LIMIT",
        "legal_oracle_objective": "victim_request_ttft_p99",
        "legal_oracle_scope": "full_frozen_cell",
        "delete_attacker_used_for_policy_capture": False,
    }:
        raise ProtocolError("formal v1 counterfactual contract changed")
    if type(counterfactuals.get("delete_attacker_used_for_policy_capture")) is not bool:
        raise ProtocolError("formal v1 counterfactual JSON types changed")

    for path, key, value in _walk_items(config):
        if key.endswith("_sha256"):
            if not isinstance(value, str) or not (
                HEX64.fullmatch(value) or UNRESOLVED.fullmatch(value)
            ):
                raise ProtocolError(
                    f"{path} must be lowercase SHA-256 or an UNRESOLVED_* placeholder"
                )
        elif isinstance(value, str) and not value:
            raise ProtocolError(f"{path} must not be empty")

    statistics = _mapping_with_fields(config, "statistics", STATISTICS_FIELDS)
    if statistics.get("bootstrap") != "paired_arrival_block_clustered_by_request_document":
        raise ProtocolError("bootstrap estimator is not the frozen paired-block estimator")
    if statistics.get("bootstrap_resamples") != 10000:
        raise ProtocolError("formal v1 requires exactly 10000 bootstrap resamples")
    if statistics.get("bootstrap_seed") != 20260729:
        raise ProtocolError("formal v1 requires bootstrap_seed=20260729")
    if statistics.get("empirical_p99") != "nearest_rank_v1":
        raise ProtocolError("formal v1 requires empirical_p99=nearest_rank_v1")
    if statistics.get("bootstrap_interval") != "percentile_two_sided_type7_95":
        raise ProtocolError(
            "formal v1 requires percentile_two_sided_type7_95 intervals"
        )
    if statistics.get("confidence_level") != 0.95:
        raise ProtocolError("formal v1 requires confidence_level=0.95")
    if statistics.get("goodput_definition") != "completed_input_tokens_per_wall_clock_us":
        raise ProtocolError("formal v1 goodput definition changed")
    if statistics.get("goodput_window") != "last_completion_minus_first_arrival_per_block":
        raise ProtocolError("formal v1 goodput window changed")
    if statistics.get("goodput_aggregation") != "paired_block_ratio_of_sums":
        raise ProtocolError("formal v1 goodput aggregation changed")
    if statistics.get("queue_growth_max_fraction") != 0.02:
        raise ProtocolError("formal v1 queue-growth tolerance changed")
    if statistics.get("thresholds") != FROZEN_THRESHOLDS:
        raise ProtocolError("formal v1 threshold table changed")
    if statistics.get("no_cross_model_pooling") is not True:
        raise ProtocolError("formal v1 forbids cross-model pooling")

    exactness = _mapping_with_fields(
        config,
        "exactness",
        {
            "output_token_ids_exact",
            "argmax_exact",
            "hidden_max_abs",
            "logit_max_abs",
            "logit_max_rel",
        },
    )
    if (
        exactness.get("output_token_ids_exact") is not True
        or exactness.get("argmax_exact") is not True
    ):
        raise ProtocolError("formal v1 output/token exactness contract changed")
    for key in ("hidden_max_abs", "logit_max_abs", "logit_max_rel"):
        value = exactness.get(key)
        if isinstance(value, str) and UNRESOLVED.fullmatch(value):
            continue
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0
        ):
            raise ProtocolError(f"exactness.{key} must be unresolved or non-negative")

    authorized = config.get("formal_execution_authorized")
    if type(authorized) is not bool:
        raise ProtocolError("formal_execution_authorized must be a JSON boolean")
    expected_status = (
        "FORMAL_EXECUTION_AUTHORIZED" if authorized else "PROTOCOL_ONLY_NOT_AUTHORIZED"
    )
    if config.get("status") != expected_status:
        raise ProtocolError("status and formal_execution_authorized disagree")
    if tuple(config.get("allowed_statuses", ())) != FROZEN_ALLOWED_STATUSES:
        raise ProtocolError("formal v1 status registry changed")


def load_config(path: str | Path) -> dict[str, Any]:
    config = json.loads(
        Path(path).read_text(encoding="utf-8"),
        object_pairs_hook=_unique_json_object,
        parse_constant=_reject_json_constant,
    )
    if not isinstance(config, dict):
        raise ProtocolError("RouteShield config root must be an object")
    validate_config(config)
    return config


def unresolved_fields(value: object, *, path: str = "") -> list[str]:
    unresolved: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            child = f"{path}.{key}" if path else str(key)
            unresolved.extend(unresolved_fields(item, path=child))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            unresolved.extend(unresolved_fields(item, path=f"{path}[{index}]"))
    elif isinstance(value, str) and value.startswith("UNRESOLVED_"):
        unresolved.append(path)
    return unresolved


def readiness_report(config: Mapping[str, Any]) -> dict[str, object]:
    try:
        validate_config(config)
    except (KeyError, TypeError, ProtocolError) as exc:
        return {
            "schema": "routeshield-gate0-readiness-v1",
            "status": "INVALID_CONFIG",
            "formal_result": False,
            "unresolved_count": 1,
            "blockers": [str(exc)],
            "evidence_boundary": "Invalid config; no artifact was evaluated",
        }
    blockers = unresolved_fields(config)
    required = config["required_evidence"]
    if not FORMAL_RAW_EVALUATOR_IMPLEMENTED:
        blockers.append("implementation.formal_raw_evaluator_approved=false")
    if not bool(config.get("formal_execution_authorized")):
        blockers.append("formal_execution_authorized=false")
    if not bool(required.get("native_continuous_prefill")):
        blockers.append("required_evidence.native_continuous_prefill=false")
    if bool(config["workloads"].get("sealed_evaluation_opened")):
        blockers.append("sealed_evaluation_opened=true_before_authorization")
    blockers = sorted(set(blockers))
    return {
        "schema": "routeshield-gate0-readiness-v1",
        "status": (
            "BLOCKED_PROTOCOL_NOT_AUTHORIZED"
            if not bool(config.get("formal_execution_authorized"))
            else (
                "BLOCKED_MISSING_FORMAL_EVIDENCE" if blockers else "READY_FOR_FORMAL_GATE0"
            )
        ),
        "formal_result": False,
        "unresolved_count": len(blockers),
        "blockers": blockers,
        "evidence_boundary": (
            "Readiness only; no route, replay, P99, EP, or policy result was evaluated"
        ),
    }


def _positive_finite(name: str, value: float) -> float:
    if not math.isfinite(value) or value <= 0:
        raise ProtocolError(f"{name} must be finite and positive")
    return value


def route_specific_harm(*, attack_p99: float, matched_benign_p99: float) -> float:
    attack = _positive_finite("attack_p99", attack_p99)
    benign = _positive_finite("matched_benign_p99", matched_benign_p99)
    return attack / benign - 1.0


def oracle_gain(*, attack_p99: float, oracle_p99: float) -> float:
    attack = _positive_finite("attack_p99", attack_p99)
    oracle = _positive_finite("oracle_p99", oracle_p99)
    if oracle > attack:
        raise ProtocolError("legal Oracle cannot be slower than the attacked baseline")
    return (attack - oracle) / attack


def oracle_recovery(
    *, attack_p99: float, matched_benign_p99: float, oracle_p99: float
) -> float:
    attack = _positive_finite("attack_p99", attack_p99)
    benign = _positive_finite("matched_benign_p99", matched_benign_p99)
    oracle = _positive_finite("oracle_p99", oracle_p99)
    denominator = attack - benign
    if denominator <= 0:
        raise ProtocolError("oracle recovery is undefined without positive attack harm")
    if oracle > attack:
        raise ProtocolError("legal Oracle cannot be slower than the attacked baseline")
    return (attack - oracle) / denominator


def simple_capture(*, attack_p99: float, oracle_p99: float, simple_p99: float) -> float:
    attack = _positive_finite("attack_p99", attack_p99)
    oracle = _positive_finite("oracle_p99", oracle_p99)
    simple = _positive_finite("simple_p99", simple_p99)
    denominator = attack - oracle
    if denominator <= 0:
        raise ProtocolError("simple capture is undefined without positive Oracle headroom")
    if simple < oracle - 1e-12:
        raise ProtocolError("a legal simple policy cannot outperform the exact legal Oracle")
    return (attack - simple) / denominator


def proposed_residual(*, simple_p99: float, proposed_p99: float, oracle_p99: float) -> float:
    simple = _positive_finite("simple_p99", simple_p99)
    proposed = _positive_finite("proposed_p99", proposed_p99)
    oracle = _positive_finite("oracle_p99", oracle_p99)
    if proposed < oracle - 1e-12:
        raise ProtocolError("a legal proposed policy cannot outperform the exact legal Oracle")
    return (simple - proposed) / simple


@dataclass(frozen=True)
class MetricCell:
    model: str
    load_cell: str
    traffic_class: str
    metric_name: str
    harm_point: float
    harm_lcb: float
    oracle_gain_point: float
    oracle_gain_lcb: float
    oracle_recovery_lcb: float
    simple_capture_ucb: float
    benign_goodput_loss_ucb: float
    exactness_pass: bool
    queue_stable: bool
    no_drop_or_starvation: bool
    full_request_dag_exact: bool
    legal_action_space: bool
    oracle_exact: bool

    @classmethod
    def from_mapping(cls, row: Mapping[str, object]) -> "MetricCell":
        numeric = (
            "harm_point",
            "harm_lcb",
            "oracle_gain_point",
            "oracle_gain_lcb",
            "oracle_recovery_lcb",
            "simple_capture_ucb",
            "benign_goodput_loss_ucb",
        )
        converted: dict[str, float] = {}
        for key in numeric:
            try:
                value = float(row[key])
            except (KeyError, TypeError, ValueError) as exc:
                raise ProtocolError(f"metric cell missing numeric {key}") from exc
            if not math.isfinite(value):
                raise ProtocolError(f"metric {key} must be finite")
            converted[key] = value
        booleans = (
            "exactness_pass",
            "queue_stable",
            "no_drop_or_starvation",
            "full_request_dag_exact",
            "legal_action_space",
            "oracle_exact",
        )
        for key in booleans:
            if type(row.get(key)) is not bool:
                raise ProtocolError(f"metric cell {key} must be a JSON boolean")
        return cls(
            model=str(row.get("model", "")),
            load_cell=str(row.get("load_cell", "")),
            traffic_class=str(row.get("traffic_class", "")),
            metric_name=str(row.get("metric_name", "")),
            **converted,
            **{key: bool(row[key]) for key in booleans},
        )


def evaluate_metric_cells(
    config: Mapping[str, Any], cells: Iterable[MetricCell]
) -> dict[str, object]:
    materialized = list(cells)
    expected_models = {str(row["key"]) for row in config["models"]}
    primary = [
        cell
        for cell in materialized
        if cell.load_cell == "70pct" and cell.traffic_class == "ADV_TEXT"
    ]
    low_load_controls = [
        cell
        for cell in materialized
        if cell.load_cell == "30pct" and cell.traffic_class == "NAT_BENIGN"
    ]
    structured_controls = [
        cell
        for cell in materialized
        if cell.load_cell == "70pct" and cell.traffic_class == "NAT_PATHOLOGICAL"
    ]
    observed_models = {cell.model for cell in primary}
    low_load_models = {cell.model for cell in low_load_controls}
    structured_models = {cell.model for cell in structured_controls}
    if (
        len(materialized) != 3 * len(expected_models)
        or len(primary) + len(low_load_controls) + len(structured_controls)
        != len(materialized)
        or observed_models != expected_models
        or low_load_models != expected_models
        or structured_models != expected_models
        or len(primary) != len(expected_models)
        or len(low_load_controls) != len(expected_models)
        or len(structured_controls) != len(expected_models)
    ):
        return {
            "status": "INVALID_ARTIFACT",
            "formal_result": False,
            "reason": (
                "need exactly one 70pct ADV_TEXT primary, one 30pct NAT_BENIGN "
                "low-load control, and one 70pct NAT_PATHOLOGICAL false-positive "
                "control per frozen model"
            ),
        }

    evaluated = [*primary, *low_load_controls, *structured_controls]
    if any(cell.metric_name != "REPLAYED_TTFT_P99" for cell in evaluated):
        return {
            "status": "INVALID_ARTIFACT",
            "formal_result": False,
            "reason": "single-GPU metric must be named REPLAYED_TTFT_P99",
        }
    if any(
        not (
            cell.exactness_pass
            and cell.queue_stable
            and cell.no_drop_or_starvation
            and cell.full_request_dag_exact
            and cell.legal_action_space
            and cell.oracle_exact
        )
        for cell in evaluated
    ):
        return {
            "status": "INVALID_REQUEST_DAG",
            "formal_result": False,
            "reason": "exactness, stable queue, legal action space, or exact full-DAG Oracle is missing",
        }

    thresholds = config["statistics"]["thresholds"]
    if any(
        cell.harm_point < float(thresholds["harm_point_min"])
        or cell.harm_lcb <= float(thresholds["harm_lcb_strict_min"])
        for cell in primary
    ):
        threshold_branch = "PHENOMENON_THRESHOLD_FAIL"
    elif any(
        cell.oracle_gain_point < float(thresholds["oracle_gain_point_min"])
        or cell.oracle_gain_lcb <= float(thresholds["oracle_gain_lcb_strict_min"])
        or cell.oracle_recovery_lcb
        < float(thresholds["oracle_recovery_lcb_min"])
        for cell in primary
    ):
        threshold_branch = "ORACLE_THRESHOLD_FAIL"
    elif any(
        cell.simple_capture_ucb
        >= float(thresholds["simple_capture_ucb_strict_max"])
        for cell in primary
    ):
        threshold_branch = "SIMPLE_CAPTURE_THRESHOLD_FAIL"
    elif any(
        cell.benign_goodput_loss_ucb
        >= float(thresholds["benign_goodput_loss_ucb_strict_max"])
        for cell in [*low_load_controls, *structured_controls]
    ):
        threshold_branch = "CONTROL_TAX_THRESHOLD_FAIL"
    else:
        threshold_branch = "ALL_THRESHOLDS_PASS"

    return {
        "status": "UNTRUSTED_AGGREGATE_SHAPE_ONLY",
        "threshold_branch": threshold_branch,
        "formal_result": False,
        "models_evaluated_separately": sorted(observed_models),
        "evidence_boundary": (
            "Untrusted self-reported aggregate shape check only; raw paired blocks, sample "
            "counts, hashes, estimates, and bootstrap intervals were not recomputed"
        ),
    }
