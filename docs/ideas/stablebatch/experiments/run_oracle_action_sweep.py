#!/usr/bin/env python3
"""Oracle-first StableBatch action-space sweep.

For every frozen victim/layer cell, execute the same all-M1 reference, the
all-M64 unprotected arm, and all eight possible one-M1-plus-seven-M64 rank
actions.  The oracle reads outcomes only after every action has executed and is
therefore an upper bound, not an online selector.  A positive selected action
is rerun once to confirm the hindsight choice is stable.
"""

from __future__ import annotations

import argparse
from fractions import Fraction
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
import run_single_contribution_pilot as base  # noqa: E402


ProtocolError = base.ProtocolError
RUNNER_RELATIVE = "docs/ideas/stablebatch/experiments/run_oracle_action_sweep.py"
TEST_RELATIVE = "docs/ideas/stablebatch/experiments/test_oracle_action_sweep.py"
CONFIG_RELATIVE = (
    "docs/ideas/stablebatch/experiments/configs/oracle_action_sweep_v1.json"
)
LOCK_RELATIVE = (
    "docs/ideas/stablebatch/experiments/configs/"
    "FROZEN_ORACLE_ACTION_SWEEP_LOCK_V1.json"
)
LOCK_SCHEMA = "stablebatch-oracle-action-sweep-frozen-lock-v1"


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def candidate_surface(
    protected_rank: int,
    top_k: int,
    baseline_m: int = 1,
    treatment_m: int = 64,
) -> dict[int, int]:
    if protected_rank not in range(top_k):
        raise ProtocolError(f"protected rank {protected_rank} outside top-k {top_k}")
    return {
        rank: (baseline_m if rank == protected_rank else treatment_m)
        for rank in range(top_k)
    }


def all_surfaces(
    top_k: int, baseline_m: int = 1, treatment_m: int = 64
) -> dict[str, dict[int, int]]:
    surfaces = {
        "R": {rank: baseline_m for rank in range(top_k)},
        "U": {rank: treatment_m for rank in range(top_k)},
    }
    surfaces.update(
        {
            f"A{rank}": candidate_surface(
                rank, top_k, baseline_m=baseline_m, treatment_m=treatment_m
            )
            for rank in range(top_k)
        }
    )
    return surfaces


def deterministic_arm_order(
    cell_key: str, labels: Sequence[str], seed: str
) -> list[str]:
    return sorted(labels, key=lambda label: (sha256_text(f"{seed}|{cell_key}|{label}"), label))


def compact_arm(row: Mapping[str, Any]) -> dict[str, Any]:
    observation = row["observation"]
    trace = row["intervention_trace"]
    return {
        "surface_m_by_rank": row["surface_m_by_rank"],
        "target_moe_output_sha256": observation["target_moe_output_sha256"],
        "topk_experts_by_layer": observation["topk_experts_by_layer"],
        "final_logits_sha256": observation["final_logits_sha256"],
        "greedy_token_id": observation["greedy_token_id"],
        "target_applied_raw_sha256_by_rank": trace[
            "target_applied_raw_sha256_by_rank"
        ],
        "target_native_raw_sha256_by_rank": trace[
            "target_native_raw_sha256_by_rank"
        ],
        "target_gate_weight_sha256_by_rank": trace[
            "target_gate_weight_sha256_by_rank"
        ],
        "non_target_contributions_sha256": trace[
            "non_target_contributions_sha256"
        ],
    }


def validate_noop(
    native: Mapping[str, Any],
    noop: Mapping[str, Any],
    noop_trace: Mapping[str, Any],
    cell: Mapping[str, Any],
    top_k: int,
) -> dict[str, bool]:
    checks = {
        "input_equal": native["input_ids_sha256"] == noop["input_ids_sha256"],
        "attention_mask_equal": native["attention_mask_sha256"]
        == noop["attention_mask_sha256"],
        "target_input_equal": native["target_input_sha256"]
        == noop["target_input_sha256"],
        "target_router_equal": native["target_router_logits_sha256"]
        == noop["target_router_logits_sha256"],
        "target_moe_output_equal": native["target_moe_output_sha256"]
        == noop["target_moe_output_sha256"],
        "all_routes_equal": native["topk_experts_by_layer"]
        == noop["topk_experts_by_layer"],
        "final_logits_equal": native["final_logits_sha256"]
        == noop["final_logits_sha256"],
    }
    if not all(checks.values()):
        raise ProtocolError(f"oracle native no-op failed: {checks}")
    if noop_trace["target_moe_output_sha256"] != noop["target_moe_output_sha256"]:
        raise ProtocolError("oracle no-op trace and observation disagree")
    if noop_trace["target_selected_experts"] != list(map(int, cell["expert_ids"])):
        raise ProtocolError("oracle no-op target experts differ from sealed cell")
    for rank in range(top_k):
        key = str(rank)
        if noop_trace["pair_match_count_by_rank"][key] != 1:
            raise ProtocolError("oracle no-op missed a target rank")
        if noop_trace["target_native_raw_sha256_by_rank"][key] != noop_trace[
            "target_applied_raw_sha256_by_rank"
        ][key]:
            raise ProtocolError("oracle no-op changed a target raw output")
    return checks


def execute_surface(
    model: Any,
    input_ids: Any,
    cell: Mapping[str, Any],
    representative: Any,
    source_config: Mapping[str, Any],
    surface: Mapping[int, int],
    replacements: Mapping[int, Mapping[int, Any]],
    local: Mapping[str, Any],
    native: Mapping[str, Any],
    noop_trace: Mapping[str, Any],
) -> dict[str, Any]:
    replacement_map = {
        rank: replacements[rank][int(m_value)] for rank, m_value in surface.items()
    }
    with observable.patched_topk_contributions(
        model, cell, replacement_map, "replacement"
    ) as trace:
        observation = base.run_observation(
            model, input_ids, source_config, representative
        )
    if trace["target_input_sha256"] != cell["target_hidden_sha256"]:
        raise ProtocolError("oracle arm target input differs from sealed cell")
    if trace["target_router_logits_sha256"] != cell["target_router_logits_sha256"]:
        raise ProtocolError("oracle arm target router differs from sealed cell")
    if trace["target_selected_experts"] != list(map(int, cell["expert_ids"])):
        raise ProtocolError("oracle arm target experts differ from sealed cell")
    for rank, m_value in surface.items():
        key = str(rank)
        if trace["pair_match_count_by_rank"][key] != 1:
            raise ProtocolError("oracle arm missed a target rank")
        if trace["routing_weight_apply_count_by_rank"][key] != 1:
            raise ProtocolError("oracle arm did not apply a gate weight exactly once")
        expected_key = "m1_sha256" if int(m_value) == 1 else "m64_sha256"
        if trace["target_applied_raw_sha256_by_rank"][key] != local["ranks"][key][
            expected_key
        ]:
            raise ProtocolError("oracle arm applied raw output differs from side-call")
        if trace["target_native_raw_sha256_by_rank"][key] != noop_trace[
            "target_native_raw_sha256_by_rank"
        ][key]:
            raise ProtocolError("oracle arm native raw output differs from no-op")
        if trace["target_gate_weight_sha256_by_rank"][key] != noop_trace[
            "target_gate_weight_sha256_by_rank"
        ][key]:
            raise ProtocolError("oracle arm gate weight differs from no-op")
    if trace["target_routing_weights_sha256"] != noop_trace[
        "target_routing_weights_sha256"
    ]:
        raise ProtocolError("oracle arm routing weights differ from no-op")
    if trace["non_target_contributions_sha256"] != noop_trace[
        "non_target_contributions_sha256"
    ]:
        raise ProtocolError("oracle arm changed non-target contributions")
    if trace["target_moe_output_sha256"] != observation["target_moe_output_sha256"]:
        raise ProtocolError("oracle arm trace and observation disagree")
    if observation["input_ids_sha256"] != native["input_ids_sha256"]:
        raise ProtocolError("oracle arm input IDs differ from native")
    if observation["attention_mask_sha256"] != native["attention_mask_sha256"]:
        raise ProtocolError("oracle arm attention mask differs from native")
    for layer_idx in range(int(cell["layer"]) + 1):
        if observation["router_logits_sha256_by_layer"][layer_idx] != native[
            "router_logits_sha256_by_layer"
        ][layer_idx]:
            raise ProtocolError(
                f"oracle arm differs before intervention at layer {layer_idx}"
            )
    return {
        "surface_m_by_rank": {str(rank): int(value) for rank, value in surface.items()},
        "intervention_trace": dict(trace),
        "observation": observation,
    }


def run_oracle_cell(
    model: Any,
    cell_index: int,
    cell: Mapping[str, Any],
    assignment: Mapping[str, Any],
    source_config: Mapping[str, Any],
    oracle_config: Mapping[str, Any],
) -> dict[str, Any]:
    import torch

    top_k = int(source_config["model"]["num_experts_per_tok"])
    m1 = int(source_config["intervention"]["baseline_m"])
    m64 = int(source_config["intervention"]["treatment_m"])
    configured_ranks = list(map(int, oracle_config["action_space"]["candidate_ranks"]))
    if configured_ranks != list(range(top_k)):
        raise ProtocolError("oracle action space must enumerate every top-k rank")
    representative = base.PairIdentity(
        layer=int(cell["layer"]),
        flat_token_idx=int(cell["flat_token_idx"]),
        topk_rank=0,
        expert_id=int(cell["expert_ids"][0]),
    )
    input_ids = torch.tensor(
        [cell["window_token_ids"]], dtype=torch.long, device="cuda"
    )
    replacements, local = observable.precompute_cell_replacements(
        model, cell, assignment, source_config
    )
    native = base.run_observation(model, input_ids, source_config, representative)
    if native["target_input_sha256"] != cell["target_hidden_sha256"]:
        raise ProtocolError("oracle native target input differs from sealed scan")
    if native["target_router_logits_sha256"] != cell["target_router_logits_sha256"]:
        raise ProtocolError("oracle native router differs from sealed scan")
    with observable.patched_topk_contributions(model, cell, None, "self") as noop_trace:
        noop = base.run_observation(model, input_ids, source_config, representative)
    noop_checks = validate_noop(native, noop, noop_trace, cell, top_k)

    surfaces = all_surfaces(top_k, m1, m64)
    labels = list(surfaces)
    order = deterministic_arm_order(
        observable.cell_key(cell),
        labels,
        str(oracle_config["action_space"]["arm_order_seed"]),
    )
    raw_arms: dict[str, dict[str, Any]] = {}
    for label in order:
        raw_arms[label] = execute_surface(
            model,
            input_ids,
            cell,
            representative,
            source_config,
            surfaces[label],
            replacements,
            local,
            native,
            noop_trace,
        )

    reference_routes = raw_arms["R"]["observation"]["topk_experts_by_layer"]
    start_layer = int(cell["layer"]) + 1
    d_u_layers = base.changed_membership_layers(
        reference_routes,
        raw_arms["U"]["observation"]["topk_experts_by_layer"],
        start_layer,
    )
    d_u = len(d_u_layers)
    action_rows: dict[str, dict[str, Any]] = {}
    for rank in configured_ranks:
        label = f"A{rank}"
        changed = base.changed_membership_layers(
            reference_routes,
            raw_arms[label]["observation"]["topk_experts_by_layer"],
            start_layer,
        )
        if changed and raw_arms[label]["observation"][
            "target_moe_output_sha256"
        ] == raw_arms["R"]["observation"]["target_moe_output_sha256"]:
            raise ProtocolError("oracle action changed routes without changing target combine")
        action_rows[str(rank)] = {
            "rank": rank,
            "expert_id": int(cell["expert_ids"][rank]),
            "changed_layers_vs_R": changed,
            "distance_vs_R": len(changed),
            "reward": d_u - len(changed),
            "full_restoration": bool(d_u > 0 and not changed),
            "arm": compact_arm(raw_arms[label]),
        }
    best_rank = min(
        configured_ranks,
        key=lambda rank: (-int(action_rows[str(rank)]["reward"]), rank),
    )
    best_reward = int(action_rows[str(best_rank)]["reward"])
    confirmation: dict[str, Any] | None = None
    if best_reward > 0 and bool(
        oracle_config["action_space"]["confirm_selected_positive_oracle_action"]
    ):
        label = f"A{best_rank}"
        repeated = execute_surface(
            model,
            input_ids,
            cell,
            representative,
            source_config,
            surfaces[label],
            replacements,
            local,
            native,
            noop_trace,
        )
        first_signature = observable.multi_arm_signature(raw_arms[label])
        repeated_signature = observable.multi_arm_signature(repeated)
        if first_signature != repeated_signature:
            raise ProtocolError("selected positive oracle action was not repeat-stable")
        repeated_changed = base.changed_membership_layers(
            reference_routes,
            repeated["observation"]["topk_experts_by_layer"],
            start_layer,
        )
        if repeated_changed != action_rows[str(best_rank)]["changed_layers_vs_R"]:
            raise ProtocolError("selected oracle action route outcome was not stable")
        confirmation = {
            "rank": best_rank,
            "signature_sha256": hashlib.sha256(first_signature).hexdigest(),
            "changed_layers_vs_R": repeated_changed,
            "status": "PASS",
        }

    return {
        "schema_version": "stablebatch-oracle-action-cell-v1",
        "cell_index": cell_index,
        "cell_id": f"cell-{cell_index:03d}",
        **observable.public_cell(cell),
        "source_shuffled_rank": int(assignment["shuffled_rank"]),
        "source_maxgate_rank": int(assignment["observable_rank"]),
        "arm_order": order,
        "sidecall_m_order_per_rank": assignment["sidecall_m_order_per_rank"],
        "local_side_calls": local,
        "native_noop_checks": noop_checks,
        "reference_arm": compact_arm(raw_arms["R"]),
        "unprotected_arm": compact_arm(raw_arms["U"]),
        "unprotected_changed_layers_vs_R": d_u_layers,
        "unprotected_distance_vs_R": d_u,
        "actions": action_rows,
        "forced_oracle_rank": best_rank,
        "forced_oracle_reward": best_reward,
        "abstaining_oracle_action": (
            {"action": "protect_rank", "rank": best_rank, "reward": best_reward}
            if best_reward > 0
            else {"action": "abstain", "rank": None, "reward": 0}
        ),
        "selected_positive_action_confirmation": confirmation,
        "integrity_status": "PASS",
    }


def classify_results(
    rows: Sequence[Mapping[str, Any]], oracle_config: Mapping[str, Any]
) -> dict[str, Any]:
    expected = oracle_config["source"]["expected_closure"]
    if len(rows) != int(expected["cell_count"]):
        raise ProtocolError("oracle summary received wrong cell count")
    if any(row["integrity_status"] != "PASS" for row in rows):
        raise ProtocolError("oracle summary received a failed cell")
    ranks = list(map(int, oracle_config["action_space"]["candidate_ranks"]))
    if any(set(map(int, row["actions"])) != set(ranks) for row in rows):
        raise ProtocolError("oracle cell action space is incomplete")

    total_u = sum(int(row["unprotected_distance_vs_R"]) for row in rows)
    all_rewards = [
        int(row["actions"][str(rank)]["reward"])
        for row in rows
        for rank in ranks
    ]
    random_all_cells_expected = Fraction(sum(all_rewards), len(ranks))
    forced_oracle_total = sum(int(row["forced_oracle_reward"]) for row in rows)
    abstaining_oracle_total = sum(
        max(0, int(row["forced_oracle_reward"])) for row in rows
    )
    maxgate_rank = int(expected["maxgate_rank"])
    maxgate_total = sum(
        int(row["actions"][str(maxgate_rank)]["reward"]) for row in rows
    )
    shuffle_total = sum(
        int(row["actions"][str(int(row["source_shuffled_rank"]))]["reward"])
        for row in rows
    )
    if maxgate_total != int(expected["maxgate_total_reward"]):
        raise ProtocolError(
            f"oracle sweep MaxGate closure {maxgate_total} != {expected['maxgate_total_reward']}"
        )
    if shuffle_total != int(expected["frozen_shuffle_total_reward"]):
        raise ProtocolError(
            f"oracle sweep shuffle closure {shuffle_total} != {expected['frozen_shuffle_total_reward']}"
        )

    positive_rows = [row for row in rows if int(row["forced_oracle_reward"]) > 0]
    positive_victims = sorted({str(row["victim_id"]) for row in positive_rows})
    if any(row["selected_positive_action_confirmation"] is None for row in positive_rows):
        raise ProtocolError("positive oracle action lacks confirmation")
    recovery_fraction = (
        float(Fraction(abstaining_oracle_total, total_u)) if total_u else 0.0
    )
    positive_action_budget = len(positive_rows)
    budget_matched_global_random = Fraction(
        positive_action_budget * sum(all_rewards), len(rows) * len(ranks)
    )
    budget_matched_conditional_random = Fraction(
        sum(
            int(row["actions"][str(rank)]["reward"])
            for row in positive_rows
            for rank in ranks
        ),
        len(ranks),
    )
    advantage_global = float(
        Fraction(abstaining_oracle_total, 1) - budget_matched_global_random
    )
    advantage_conditional = float(
        Fraction(abstaining_oracle_total, 1) - budget_matched_conditional_random
    )
    threshold = oracle_config["signal"]["strong_if"]
    min_matched_advantage = float(
        threshold[
            "min_abstaining_oracle_advantage_over_each_budget_matched_random_expected_reward"
        ]
    )
    strong_checks = {
        "oracle_recovery_fraction": recovery_fraction
        >= float(threshold["min_abstaining_oracle_recovery_fraction"]),
        "oracle_advantage_over_budget_matched_global_random": advantage_global
        >= min_matched_advantage,
        "oracle_advantage_over_budget_matched_conditional_random": advantage_conditional
        >= min_matched_advantage,
        "positive_victim_coverage": len(positive_victims)
        >= int(threshold["min_distinct_victims_with_positive_oracle_reward"]),
    }
    if all(strong_checks.values()):
        verdict = "STRONG_ORACLE_ACTION_VALUE_SIGNAL"
    elif (
        abstaining_oracle_total > float(budget_matched_global_random)
        and abstaining_oracle_total > float(budget_matched_conditional_random)
        and positive_victims
    ):
        verdict = "WEAK_ORACLE_ACTION_VALUE_SIGNAL"
    else:
        verdict = "WEAKENS_SINGLE_CONTRIBUTION_ACTION_SPACE"

    positive_best = sorted(
        (
            {
                "cell_key": observable.cell_key(row),
                "victim_id": str(row["victim_id"]),
                "layer": int(row["layer"]),
                "rank": int(row["forced_oracle_rank"]),
                "reward": int(row["forced_oracle_reward"]),
            }
            for row in positive_rows
        ),
        key=lambda item: (-int(item["reward"]), str(item["cell_key"])),
    )
    row_by_key = {observable.cell_key(row): row for row in rows}
    budget_curve: list[dict[str, Any]] = []
    for budget in map(int, oracle_config["budget_curve"]["budgets_in_protected_cells"]):
        selected = positive_best[:budget]
        selected_rows = [row_by_key[str(item["cell_key"])] for item in selected]
        actions_used = len(selected)
        global_random = Fraction(
            actions_used * sum(all_rewards), len(rows) * len(ranks)
        )
        conditional_random = Fraction(
            sum(
                int(row["actions"][str(rank)]["reward"])
                for row in selected_rows
                for rank in ranks
            ),
            len(ranks),
        )
        oracle_reward = sum(int(item["reward"]) for item in selected)
        budget_curve.append(
            {
                "budget": budget,
                "actions_used": actions_used,
                "oracle_total_reward": oracle_reward,
                "global_uniform_random_expected_reward": float(global_random),
                "global_uniform_random_expected_reward_exact": {
                    "numerator": global_random.numerator,
                    "denominator": global_random.denominator,
                },
                "conditional_uniform_random_rank_expected_reward": float(
                    conditional_random
                ),
                "conditional_uniform_random_rank_expected_reward_exact": {
                    "numerator": conditional_random.numerator,
                    "denominator": conditional_random.denominator,
                },
                "oracle_advantage_over_global_random": float(
                    Fraction(oracle_reward, 1) - global_random
                ),
                "oracle_advantage_over_conditional_random": float(
                    Fraction(oracle_reward, 1) - conditional_random
                ),
                "distinct_victims": len({str(item["victim_id"]) for item in selected}),
                "selected_cells": [str(item["cell_key"]) for item in selected],
            }
        )

    per_victim: dict[str, dict[str, Any]] = {}
    for victim in sorted({str(row["victim_id"]) for row in rows}):
        victim_rows = [row for row in rows if str(row["victim_id"]) == victim]
        victim_random = Fraction(
            sum(
                int(row["actions"][str(rank)]["reward"])
                for row in victim_rows
                for rank in ranks
            ),
            len(ranks),
        )
        per_victim[victim] = {
            "unprotected_distance": sum(
                int(row["unprotected_distance_vs_R"]) for row in victim_rows
            ),
            "uniform_random_expected_reward": float(victim_random),
            "abstaining_oracle_reward": sum(
                max(0, int(row["forced_oracle_reward"])) for row in victim_rows
            ),
            "positive_cells": sum(
                int(int(row["forced_oracle_reward"]) > 0) for row in victim_rows
            ),
        }

    positive_rank_counts = {
        str(rank): sum(
            int(int(row["forced_oracle_rank"]) == rank)
            for row in positive_rows
        )
        for rank in ranks
    }
    return {
        "cell_count": len(rows),
        "independent_document_count": len({str(row["victim_id"]) for row in rows}),
        "candidate_action_count": len(rows) * len(ranks),
        "unprotected_total_downstream_route_distance": total_u,
        "no_intervention_total_reward": 0,
        "uniform_random_one_action_per_cell_expected_total_reward": float(
            random_all_cells_expected
        ),
        "uniform_random_one_action_per_cell_expected_total_reward_exact": {
            "numerator": random_all_cells_expected.numerator,
            "denominator": random_all_cells_expected.denominator,
        },
        "frozen_shuffle_total_reward": shuffle_total,
        "maxgate_v1_total_reward": maxgate_total,
        "forced_oracle_total_reward": forced_oracle_total,
        "abstaining_oracle_total_reward": abstaining_oracle_total,
        "abstaining_oracle_remaining_route_distance": total_u
        - abstaining_oracle_total,
        "abstaining_oracle_recovery_fraction": recovery_fraction,
        "abstaining_oracle_action_budget": positive_action_budget,
        "budget_matched_global_random_expected_reward": float(
            budget_matched_global_random
        ),
        "budget_matched_global_random_expected_reward_exact": {
            "numerator": budget_matched_global_random.numerator,
            "denominator": budget_matched_global_random.denominator,
        },
        "budget_matched_conditional_random_expected_reward": float(
            budget_matched_conditional_random
        ),
        "budget_matched_conditional_random_expected_reward_exact": {
            "numerator": budget_matched_conditional_random.numerator,
            "denominator": budget_matched_conditional_random.denominator,
        },
        "abstaining_oracle_advantage_over_budget_matched_global_random": advantage_global,
        "abstaining_oracle_advantage_over_budget_matched_conditional_random": advantage_conditional,
        "positive_oracle_cell_count": len(positive_rows),
        "positive_oracle_distinct_victim_count": len(positive_victims),
        "positive_oracle_victims": positive_victims,
        "oracle_full_restoration_cell_count": sum(
            int(
                row["actions"][str(int(row["forced_oracle_rank"]))][
                    "full_restoration"
                ]
                and int(row["forced_oracle_reward"]) > 0
            )
            for row in rows
        ),
        "positive_oracle_rank_counts": positive_rank_counts,
        "budget_curve": budget_curve,
        "per_victim": per_victim,
        "strong_signal_checks": strong_checks,
        "verdict": verdict,
        "scope_of_failure": oracle_config["scope_of_failure"],
    }


def json_artifact_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    ).encode("utf-8")


def build_manifest(
    output_dir: Path, pending_run_status: Mapping[str, Any]
) -> dict[str, Any]:
    files: dict[str, dict[str, Any]] = {}
    for path in sorted(output_dir.iterdir()):
        if not path.is_file() or path.name in {"MANIFEST.json", "RUN_STATUS.json"}:
            continue
        files[path.name] = {
            "size_bytes": path.stat().st_size,
            "sha256": base.sha256_file(path),
        }
    status_bytes = json_artifact_bytes(pending_run_status)
    files["RUN_STATUS.json"] = {
        "size_bytes": len(status_bytes),
        "sha256": hashlib.sha256(status_bytes).hexdigest(),
    }
    return {
        "schema_version": "stablebatch-oracle-action-sweep-manifest-v1",
        "created_at": base.utc_now(),
        "files": files,
    }


def write_bound_status(output_dir: Path, status: Mapping[str, Any]) -> None:
    pending = output_dir / ".RUN_STATUS.json.pending"
    final = output_dir / "RUN_STATUS.json"
    base.write_json_new(pending, status)
    expected = hashlib.sha256(json_artifact_bytes(status)).hexdigest()
    if base.sha256_file(pending) != expected:
        raise ProtocolError("oracle pending status differs from manifest binding")
    os.rename(pending, final)


def verify_manifest(output_dir: Path) -> None:
    manifest = base.load_json(output_dir / "MANIFEST.json")
    if manifest.get("schema_version") != "stablebatch-oracle-action-sweep-manifest-v1":
        raise ProtocolError("wrong oracle manifest schema")
    expected = manifest.get("files")
    if not isinstance(expected, dict):
        raise ProtocolError("oracle manifest has no files map")
    actual = {
        path.name
        for path in output_dir.iterdir()
        if path.is_file() and path.name != "MANIFEST.json"
    }
    if actual != set(map(str, expected)):
        raise ProtocolError("oracle manifest file set mismatch")
    for name, binding in expected.items():
        path = output_dir / str(name)
        if path.stat().st_size != int(binding["size_bytes"]):
            raise ProtocolError(f"oracle manifest size mismatch for {name}")
        if base.sha256_file(path) != str(binding["sha256"]):
            raise ProtocolError(f"oracle manifest hash mismatch for {name}")


def validate_lock(
    lock: Mapping[str, Any], oracle_config: Mapping[str, Any]
) -> None:
    if lock.get("schema_version") != LOCK_SCHEMA:
        raise ProtocolError("wrong oracle frozen-lock schema")
    if lock.get("status") != "FROZEN_PRE_RUN":
        raise ProtocolError("oracle lock is not frozen")
    if lock.get("hypothesis_sha256") != sha256_text(str(oracle_config["hypothesis"])):
        raise ProtocolError("oracle lock hypothesis binding mismatch")
    if lock.get("action_space") != oracle_config["action_space"]:
        raise ProtocolError("oracle lock action-space binding mismatch")
    if lock.get("signal") != oracle_config["signal"]:
        raise ProtocolError("oracle lock signal binding mismatch")


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
    oracle_config = base.load_json(config_path)
    if oracle_config.get("status") != "FROZEN_PRE_RUN":
        raise ProtocolError("oracle config is not FROZEN_PRE_RUN")
    if str(runner_path.relative_to(repo_root)) != RUNNER_RELATIVE:
        raise ProtocolError("oracle runner path differs from frozen path")
    if str(config_path.relative_to(repo_root)) != CONFIG_RELATIVE:
        raise ProtocolError("oracle config path differs from frozen path")
    if str(lock_path.relative_to(repo_root)) != LOCK_RELATIVE:
        raise ProtocolError("oracle lock path differs from frozen path")
    source_config_path = repo_root / str(oracle_config["source"]["config"])
    source_config = base.load_json(source_config_path)
    lock = base.load_json(lock_path)
    validate_lock(lock, oracle_config)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir()
    started = time.time()
    try:
        base.write_json_new(
            output_dir / "run_request.json",
            {
                "schema_version": "stablebatch-oracle-action-sweep-request-v1",
                "started_at": base.utc_now(),
                "argv": sys.argv,
                "pid": os.getpid(),
                "runner_sha256": base.sha256_file(runner_path),
                "config_sha256": base.sha256_file(config_path),
                "lock_sha256": base.sha256_file(lock_path),
                "source_config_sha256": base.sha256_file(source_config_path),
                "repo_root": str(repo_root),
                "max_wall_seconds": int(args.max_wall_seconds),
                "git_head": base.command_output(
                    ["git", "-C", str(repo_root), "rev-parse", "HEAD"]
                ),
                "git_status_short": base.command_output(
                    ["git", "-C", str(repo_root), "status", "--short"]
                ),
            },
        )
        pre_import_gpu = base.gpu_snapshot()
        environment = base.verify_environment(source_config, pre_import_gpu)
        static = base.verify_static_inputs(
            source_config, repo_root, runner_path, config_path, lock_path
        )
        for key in ("config", "observable_cells", "assignment_ledger", "summary"):
            path = repo_root / str(oracle_config["source"][key])
            expected_hash = str(oracle_config["source"][f"{key}_sha256"])
            if base.sha256_file(path) != expected_hash:
                raise ProtocolError(f"oracle source binding mismatch for {key}")
        base.write_json_new(output_dir / "environment.json", environment)
        base.write_json_new(output_dir / "static_bindings.json", static)
        base.write_json_new(output_dir / "config_snapshot.json", oracle_config)
        base.write_json_new(output_dir / "source_config_snapshot.json", source_config)

        model, tokenizer = base.load_model(source_config)
        workloads = base.load_workloads(source_config, repo_root, tokenizer)
        workload_digest = observable.verify_workload_digest(workloads, source_config)
        base.write_jsonl_new(output_dir / "workloads.jsonl", workloads)
        first_ids = __import__("torch").tensor(
            [workloads[0]["window_token_ids"]],
            dtype=__import__("torch").long,
            device="cuda",
        )
        observable.warmup_native_only(model, first_ids, source_config)
        cells = observable.scan_observable_cells(model, workloads, source_config)
        cell_path = output_dir / "observable_cells.jsonl"
        base.write_jsonl_new(
            cell_path, (observable.public_cell(row) for row in cells)
        )
        if base.sha256_file(cell_path) != str(
            oracle_config["source"]["observable_cells_sha256"]
        ):
            raise ProtocolError("oracle scan differs from frozen source cells")

        ledger = base.load_json(
            repo_root / str(oracle_config["source"]["assignment_ledger"])
        )
        assignment_by_key = {
            str(row["cell_key"]): row for row in ledger["cells"]
        }
        sorted_cells = sorted(
            cells, key=lambda row: (str(row["victim_id"]), int(row["layer"]))
        )
        action_plan = {
            "schema_version": "stablebatch-oracle-action-plan-v1",
            "status": "SEALED_BEFORE_ORACLE_ACTION_OUTCOMES",
            "sealed_at": base.utc_now(),
            "hypothesis_sha256": sha256_text(str(oracle_config["hypothesis"])),
            "ordered_window_hash_digest": workload_digest,
            "candidate_ranks": oracle_config["action_space"]["candidate_ranks"],
            "action_budget_per_cell": oracle_config["action_space"][
                "action_budget_per_cell"
            ],
            "arm_order_seed": oracle_config["action_space"]["arm_order_seed"],
            "cells": [
                {
                    "cell_key": observable.cell_key(cell),
                    "source_shuffled_rank": int(
                        assignment_by_key[observable.cell_key(cell)]["shuffled_rank"]
                    ),
                    "source_maxgate_rank": int(
                        assignment_by_key[observable.cell_key(cell)]["observable_rank"]
                    ),
                    "sidecall_m_order_per_rank": assignment_by_key[
                        observable.cell_key(cell)
                    ]["sidecall_m_order_per_rank"],
                    "arm_order": deterministic_arm_order(
                        observable.cell_key(cell),
                        list(
                            all_surfaces(
                                int(source_config["model"]["num_experts_per_tok"])
                            )
                        ),
                        str(oracle_config["action_space"]["arm_order_seed"]),
                    ),
                }
                for cell in sorted_cells
            ],
            "result_rows_existed_at_seal": False,
        }
        base.write_json_new(output_dir / "ACTION_SWEEP_LOCK.json", action_plan)

        rows: list[dict[str, Any]] = []
        result_path = output_dir / "cell_results.jsonl"
        with result_path.open("x", encoding="utf-8") as stream:
            for cell_index, cell in enumerate(sorted_cells):
                if time.time() - started > int(args.max_wall_seconds):
                    raise TimeoutError("oracle action sweep exceeded max wall time")
                row = run_oracle_cell(
                    model,
                    cell_index,
                    cell,
                    assignment_by_key[observable.cell_key(cell)],
                    source_config,
                    oracle_config,
                )
                rows.append(row)
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())

        summary = {
            "schema_version": "stablebatch-oracle-action-sweep-summary-v1",
            "status": "COMPLETE",
            "hypothesis": oracle_config["hypothesis"],
            "evaluation_type": oracle_config["evaluation_type"],
            "research_boundary": oracle_config["research_boundary"],
            "action_sweep_lock_sha256": base.sha256_file(
                output_dir / "ACTION_SWEEP_LOCK.json"
            ),
            **classify_results(rows, oracle_config),
            "wall_seconds": time.time() - started,
            "completed_at": base.utc_now(),
        }
        base.write_json_new(output_dir / "summary.json", summary)
        base.write_json_new(
            output_dir / "runtime_final.json", base.verify_final_runtime(source_config)
        )
        status = {
            "status": "COMPLETE",
            "scientific_result_eligible": True,
            "verdict": summary["verdict"],
            "completed_at": base.utc_now(),
            "wall_seconds": time.time() - started,
        }
        base.write_json_new(
            output_dir / "MANIFEST.json", build_manifest(output_dir, status)
        )
        write_bound_status(output_dir, status)
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
