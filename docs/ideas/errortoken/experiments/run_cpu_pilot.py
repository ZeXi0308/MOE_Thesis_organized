#!/usr/bin/env python3
"""Retrospective cross-artifact selector pilot for ErrorToken-MoE.

This program is deliberately CPU-only and development-only.  It freezes all
three selectors and writes ``SELECTION_FROZEN.json`` before it opens the
StableBatch outcome file.  Its action artifact is a plan; it never executes a
replay, changes a model, or touches a GPU.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


Key = tuple[int, int, int]

FORBIDDEN_SELECTOR_FIELDS = frozenset(
    {
        "next_layer_topk_margin",
        "selection_score",
        "earliest_changed_downstream_layer",
    }
)
REQUIRED_SELECTOR_FIELDS = frozenset(
    {"layer", "expert_id", "topk_rank_plus_one", "mismatch_onset_risk"}
)
OUTCOME_FIELD = "reproducible_route_propagation"


@dataclass(frozen=True)
class RiskStats:
    mismatch_onset_risk: float
    calibration_row_count: int
    rows_with_observed_mismatch: int
    onset_counts: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class Candidate:
    target_id: str
    victim_id: str
    layer: int
    expert_id: int
    topk_rank_zero_based: int
    route_rank_one_based: int
    gate_weight: float
    mismatch_onset_risk: float | None
    calibration_row_count: int

    @property
    def calibration_key(self) -> Key:
        return (self.layer, self.expert_id, self.route_rank_one_based)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_manifest_hash(path: Path, expected_sha256: str, label: str) -> str:
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ValueError(f"sealed hash mismatch for {label}: expected {expected_sha256}, got {actual}")
    return actual


def verify_cross_artifact_seals(inputs: Mapping[str, Path]) -> dict[str, Any]:
    semantic_complete = json.loads(inputs["semanticfence_complete"].read_text(encoding="utf-8"))
    if semantic_complete.get("status") != "SUCCESS_COMPLETE":
        raise ValueError("SemanticFence COMPLETE status is not SUCCESS_COMPLETE")
    semantic_hashes = semantic_complete["artifact_sha256"]
    calibration_hash = require_manifest_hash(
        inputs["calibration_numeric"],
        str(semantic_hashes["calibration_numeric.jsonl"]),
        "SemanticFence calibration_numeric.jsonl",
    )
    semantic_summary_hash = require_manifest_hash(
        inputs["semanticfence_summary"],
        str(semantic_complete["summary_sha256"]),
        "SemanticFence summary.json",
    )

    stable_manifest = json.loads(inputs["stablebatch_manifest"].read_text(encoding="utf-8"))
    stable_files = stable_manifest["files"]
    selected_hash = require_manifest_hash(
        inputs["selected_targets"],
        str(stable_files["selected_targets.jsonl"]["sha256"]),
        "StableBatch selected_targets.jsonl",
    )
    # This is a byte hash only.  No outcome field is decoded until after
    # SELECTION_FROZEN.json has been written.
    target_results_hash = require_manifest_hash(
        inputs["target_results"],
        str(stable_files["target_results.jsonl"]["sha256"]),
        "StableBatch target_results.jsonl",
    )
    stable_summary_hash = require_manifest_hash(
        inputs["stablebatch_summary"],
        str(stable_files["summary.json"]["sha256"]),
        "StableBatch summary.json",
    )
    run_status = json.loads(inputs["stablebatch_run_status"].read_text(encoding="utf-8"))
    stable_summary = json.loads(inputs["stablebatch_summary"].read_text(encoding="utf-8"))
    if (
        run_status.get("status") != "COMPLETE"
        or run_status.get("verdict") != "SUPPORT"
        or not bool(run_status.get("scientific_result_eligible"))
    ):
        raise ValueError("StableBatch RUN_STATUS is not COMPLETE/SUPPORT/eligible")
    if stable_summary.get("status") != "COMPLETE" or stable_summary.get("verdict") != "SUPPORT":
        raise ValueError("StableBatch summary is not COMPLETE/SUPPORT")
    return {
        "semanticfence_complete_status": semantic_complete["status"],
        "semanticfence_calibration_numeric_sha256": calibration_hash,
        "semanticfence_summary_sha256": semantic_summary_hash,
        "stablebatch_status": run_status["status"],
        "stablebatch_verdict": run_status["verdict"],
        "stablebatch_selected_targets_sha256": selected_hash,
        "stablebatch_target_results_sha256_byte_verified_before_freeze": target_results_hash,
        "stablebatch_summary_sha256": stable_summary_hash,
    }


def load_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            yield row


def validate_config(config: Mapping[str, Any]) -> None:
    if config.get("status") != "RETROSPECTIVE_DEVELOPMENT_ONLY":
        raise ValueError("config must remain RETROSPECTIVE_DEVELOPMENT_ONLY")
    constraints = config["constraints"]
    if int(constraints["budget_per_victim"]) != 1:
        raise ValueError("this pilot is frozen to B=1")
    if int(constraints["candidates_per_victim"]) != 2:
        raise ValueError("this pilot requires exactly two candidates per victim")
    calibration = config["calibration"]
    if tuple(calibration["key_fields"]) != ("layer", "expert_id", "route_rank"):
        raise ValueError("calibration key must be (layer, expert_id, route_rank)")
    if int(calibration["stablebatch_topk_rank_offset"]) != 1:
        raise ValueError("StableBatch topk_rank must be converted with +1")
    if calibration["row_mismatch_rule"] != "any_repeat_nonexact":
        raise ValueError("row mismatch rule drifted")
    if calibration["onset_risk_formula"] != "mean(min_m/first_mismatch_m);never=0":
        raise ValueError("mismatch-onset risk formula drifted")

    selector_fields = frozenset(config["selector_contract"]["selector_fields"])
    declared_forbidden = frozenset(config["selector_contract"]["forbidden_fields"])
    if selector_fields != REQUIRED_SELECTOR_FIELDS:
        raise ValueError(f"selector fields drifted: {sorted(selector_fields)}")
    if not FORBIDDEN_SELECTOR_FIELDS.issubset(declared_forbidden):
        raise ValueError("config does not statically forbid all leakage fields")
    overlap = selector_fields.intersection(FORBIDDEN_SELECTOR_FIELDS)
    if overlap:
        raise ValueError(f"forbidden selector fields requested: {sorted(overlap)}")
    if config["evaluation"]["outcome_field"] != OUTCOME_FIELD:
        raise ValueError("outcome field drifted")
    if not bool(config["evaluation"]["enumerate_all_matched_assignments"]):
        raise ValueError("exact matched-null enumeration must remain enabled")
    if config["action_plan"]["execution_status"] != "NOT_EXECUTED_PLAN_ONLY":
        raise ValueError("action artifact must remain a non-executed plan")
    policies = config["policies"]
    if set(policies) != {
        "errortoken_mismatch_onset",
        "gate_weight_first",
        "topk_rank_first",
    }:
        raise ValueError("policy set drifted")
    if any(spec.get("selection") != "causal_first_eligible_B1" for spec in policies.values()):
        raise ValueError("all policies must use causal_first_eligible_B1")


def row_is_mismatch(repeat_row_exact: Sequence[Sequence[Any]], row_index: int) -> bool:
    if not repeat_row_exact:
        raise ValueError("calibration call has no repeats")
    values: list[bool] = []
    for repeat in repeat_row_exact:
        if row_index >= len(repeat):
            raise ValueError("repeat row width does not match row_records")
        values.append(bool(repeat[row_index]))
    return not all(values)


def build_mismatch_onset_risks(
    calibration_numeric_path: Path, m_grid: Sequence[int]
) -> dict[Key, RiskStats]:
    grid = tuple(sorted(int(value) for value in m_grid))
    if not grid or grid[0] <= 1 or any(value <= 1 for value in grid):
        raise ValueError("m_grid must contain multi-row M values only")
    if len(set(grid)) != len(grid):
        raise ValueError("m_grid contains duplicates")
    minimum_m = grid[0]

    # Each calibration row is followed across the available M values.  The
    # only numerical signal used here is whether each repeated row was exact.
    observations: dict[Key, dict[str, dict[int, bool]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for call in load_jsonl(calibration_numeric_path):
        if call.get("arm") != "calibration":
            raise ValueError("non-calibration row in calibration_numeric")
        m = int(call["m"])
        if m not in grid:
            raise ValueError(f"unexpected calibration M={m}")
        layer = int(call["layer"])
        expert_id = int(call["expert_id"])
        row_ids = [str(value) for value in call["row_ids"]]
        row_records = call["row_records"]
        if len(row_ids) != m or len(row_records) != m:
            raise ValueError(f"call {call.get('call_index')} has inconsistent M/row counts")
        for row_index, (row_id, record) in enumerate(zip(row_ids, row_records)):
            if int(record["layer"]) != layer or int(record["expert_id"]) != expert_id:
                raise ValueError("row record disagrees with call layer/expert")
            route_rank = int(record["route_rank"])
            if route_rank < 1:
                raise ValueError("SemanticFence route_rank must be one-based")
            key = (layer, expert_id, route_rank)
            mismatch = row_is_mismatch(call["repeat_row_exact"], row_index)
            previous = observations[key][row_id].get(m)
            if previous is not None and previous != mismatch:
                raise ValueError(f"conflicting duplicate calibration observation for {key}/{row_id}/M{m}")
            observations[key][row_id][m] = mismatch

    risks: dict[Key, RiskStats] = {}
    for key, row_map in observations.items():
        onsets: list[int | None] = []
        for by_m in row_map.values():
            onset = min((m for m, mismatch in by_m.items() if mismatch), default=None)
            onsets.append(onset)
        score = sum(0.0 if onset is None else minimum_m / onset for onset in onsets) / len(onsets)
        counts = Counter("never" if onset is None else str(onset) for onset in onsets)
        ordered_counts = tuple(
            (label, counts[label])
            for label in [*(str(value) for value in grid), "never"]
            if counts[label]
        )
        risks[key] = RiskStats(
            mismatch_onset_risk=score,
            calibration_row_count=len(onsets),
            rows_with_observed_mismatch=sum(onset is not None for onset in onsets),
            onset_counts=ordered_counts,
        )
    return risks


def load_candidates(
    selected_targets_path: Path,
    risks: Mapping[Key, RiskStats],
    topk_rank_offset: int,
) -> list[Candidate]:
    candidates: list[Candidate] = []
    for target_index, row in enumerate(load_jsonl(selected_targets_path)):
        # This is an explicit allowlist copy.  In particular, none of the
        # outcome-informed StableBatch enrichment fields are copied.
        layer = int(row["layer"])
        expert_id = int(row["expert_id"])
        topk_rank = int(row["topk_rank"])
        route_rank = topk_rank + topk_rank_offset
        if topk_rank < 0 or route_rank < 1:
            raise ValueError("invalid StableBatch topk rank")
        key = (layer, expert_id, route_rank)
        stats = risks.get(key)
        candidates.append(
            Candidate(
                target_id=f"target-{target_index:02d}",
                victim_id=str(row["victim_id"]),
                layer=layer,
                expert_id=expert_id,
                topk_rank_zero_based=topk_rank,
                route_rank_one_based=route_rank,
                gate_weight=float(row["gate_weight"]),
                mismatch_onset_risk=(stats.mismatch_onset_risk if stats else None),
                calibration_row_count=(stats.calibration_row_count if stats else 0),
            )
        )
    return candidates


def group_candidate_pairs(
    candidates: Sequence[Candidate], expected_victims: int, candidates_per_victim: int
) -> dict[str, tuple[Candidate, ...]]:
    groups: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        groups[candidate.victim_id].append(candidate)
    if len(groups) != expected_victims:
        raise ValueError(f"expected {expected_victims} victims, found {len(groups)}")
    for victim_id, values in groups.items():
        if len(values) != candidates_per_victim:
            raise ValueError(
                f"victim {victim_id} has {len(values)} candidates, expected {candidates_per_victim}"
            )
    return {victim: tuple(values) for victim, values in sorted(groups.items())}


def policy_eligible(candidate: Candidate, policy: str, spec: Mapping[str, Any]) -> bool:
    threshold = float(spec["threshold"])
    if policy == "errortoken_mismatch_onset":
        return (
            candidate.mismatch_onset_risk is not None
            and candidate.mismatch_onset_risk >= threshold
        )
    if policy == "gate_weight_first":
        return candidate.gate_weight >= threshold
    if policy == "topk_rank_first":
        return candidate.topk_rank_zero_based <= int(spec["threshold"])
    raise ValueError(f"unknown policy: {policy}")


def freeze_policy_selections(
    pairs: Mapping[str, Sequence[Candidate]], policy_specs: Mapping[str, Mapping[str, Any]]
) -> dict[str, tuple[Candidate, ...]]:
    frozen: dict[str, tuple[Candidate, ...]] = {}
    for policy, spec in policy_specs.items():
        chosen: list[Candidate] = []
        for victim_id, pair in pairs.items():
            # A runtime-feasible B=1 policy cannot inspect a later layer and
            # retroactively replace an earlier action.  Scan in causal order;
            # the first pre-registered eligible candidate consumes the token.
            causal_pair = sorted(pair, key=lambda item: (item.layer, item.target_id))
            selected = next(
                (item for item in causal_pair if policy_eligible(item, policy, spec)), None
            )
            if selected is None:
                raise ValueError(f"policy {policy} has no eligible candidate for {victim_id}")
            chosen.append(selected)
        frozen[policy] = tuple(sorted(chosen, key=lambda item: (item.layer, item.victim_id, item.target_id)))
    return frozen


def candidate_record(candidate: Candidate) -> dict[str, Any]:
    value = asdict(candidate)
    value["calibration_key"] = list(candidate.calibration_key)
    return value


def make_selection_bundle(
    selections: Mapping[str, Sequence[Candidate]],
    policy_specs: Mapping[str, Mapping[str, Any]],
    selector_input_hashes: Mapping[str, str],
    seal_validation: Mapping[str, Any],
    action_config: Mapping[str, Any],
) -> dict[str, Any]:
    policies: dict[str, Any] = {}
    for policy, chosen in selections.items():
        records = [candidate_record(candidate) for candidate in chosen]
        policies[policy] = {
            "selected_count": len(records),
            "selection_rule": dict(policy_specs[policy]),
            "selected": records,
        }
        if policy == "errortoken_mismatch_onset":
            policies[policy]["action_plan"] = [
                {
                    "sequence": sequence,
                    "victim_id": candidate.victim_id,
                    "target_id": candidate.target_id,
                    "layer": candidate.layer,
                    "expert_id": candidate.expert_id,
                    "route_rank_one_based": candidate.route_rank_one_based,
                    "planned_action": action_config["planned_action"],
                    "execution_status": action_config["execution_status"],
                }
                for sequence, candidate in enumerate(chosen, start=1)
            ]
    payload = {
        "schema_version": "errortoken-cross-artifact-selection-v1",
        "status": "FROZEN_BEFORE_OUTCOME_LOAD",
        "selector_input_hashes": dict(selector_input_hashes),
        "pre_outcome_seal_validation": dict(seal_validation),
        "forbidden_selector_fields": sorted(FORBIDDEN_SELECTOR_FIELDS),
        "policies": policies,
    }
    payload["selection_sha256"] = sha256_bytes(canonical_bytes(payload))
    return payload


def load_outcomes_after_freeze(
    target_results_path: Path,
    candidates: Sequence[Candidate],
    selection_bundle: Mapping[str, Any],
) -> dict[str, bool]:
    if selection_bundle.get("status") != "FROZEN_BEFORE_OUTCOME_LOAD":
        raise ValueError("selection must be frozen before outcomes are opened")
    by_id = {candidate.target_id: candidate for candidate in candidates}
    outcomes: dict[str, bool] = {}
    for row in load_jsonl(target_results_path):
        target_id = str(row["target_id"])
        candidate = by_id.get(target_id)
        if candidate is None:
            raise ValueError(f"outcome has unknown target_id: {target_id}")
        identity = (
            str(row["victim_id"]),
            int(row["layer"]),
            int(row["expert_id"]),
            int(row["topk_rank"]),
        )
        expected = (
            candidate.victim_id,
            candidate.layer,
            candidate.expert_id,
            candidate.topk_rank_zero_based,
        )
        if identity != expected:
            raise ValueError(f"selected-target/result identity mismatch for {target_id}")
        if target_id in outcomes:
            raise ValueError(f"duplicate outcome: {target_id}")
        label = row[OUTCOME_FIELD]
        if not isinstance(label, bool):
            raise ValueError(f"non-boolean {OUTCOME_FIELD} for {target_id}")
        outcomes[target_id] = label
    if set(outcomes) != set(by_id):
        raise ValueError("outcome target set does not match candidate target set")
    return outcomes


def enumerate_exact_matched_null(
    pairs: Mapping[str, Sequence[Candidate]], outcomes: Mapping[str, bool]
) -> dict[str, Any]:
    ordered_pairs = [pairs[victim] for victim in sorted(pairs)]
    if any(len(pair) != 2 for pair in ordered_pairs):
        raise ValueError("exact bitmask null requires two candidates per victim")
    assignment_count = 1 << len(ordered_pairs)
    histogram: Counter[int] = Counter()
    for mask in range(assignment_count):
        hits = 0
        for pair_index, pair in enumerate(ordered_pairs):
            chosen = pair[(mask >> pair_index) & 1]
            hits += int(outcomes[chosen.target_id])
        histogram[hits] += 1
    mean_hits = sum(score * count for score, count in histogram.items()) / assignment_count
    return {
        "victim_pair_count": len(ordered_pairs),
        "enumerated_assignments": assignment_count,
        "expected_assignments": 2 ** len(ordered_pairs),
        "hit_count_mean": mean_hits,
        "hit_count_histogram": {str(score): histogram[score] for score in sorted(histogram)},
    }


def evaluate_policy(
    chosen: Sequence[Candidate],
    all_candidates: Sequence[Candidate],
    outcomes: Mapping[str, bool],
    matched_null: Mapping[str, Any],
) -> dict[str, Any]:
    selected_ids = {candidate.target_id for candidate in chosen}
    selected_hits = sum(int(outcomes[target_id]) for target_id in selected_ids)
    unselected = [candidate for candidate in all_candidates if candidate.target_id not in selected_ids]
    unselected_hits = sum(int(outcomes[candidate.target_id]) for candidate in unselected)
    histogram = {int(score): int(count) for score, count in matched_null["hit_count_histogram"].items()}
    assignments = int(matched_null["enumerated_assignments"])
    p_ge = sum(count for score, count in histogram.items() if score >= selected_hits) / assignments
    p_le = sum(count for score, count in histogram.items() if score <= selected_hits) / assignments
    return {
        "selected_target_ids": [candidate.target_id for candidate in chosen],
        "selected_hits": selected_hits,
        "selected_count": len(chosen),
        "selected_hit_rate": selected_hits / len(chosen) if chosen else None,
        "unselected_hits": unselected_hits,
        "unselected_count": len(unselected),
        "unselected_hit_rate": unselected_hits / len(unselected) if unselected else None,
        "selected_minus_unselected_hits": selected_hits - unselected_hits,
        "exact_one_sided_p_ge": p_ge,
        "exact_one_sided_p_le": p_le,
    }


def retrospective_verdict(primary: Mapping[str, Any], null_mean: float, alpha: float) -> tuple[str, str]:
    hits = int(primary["selected_hits"])
    if hits > null_mean and float(primary["exact_one_sided_p_ge"]) <= alpha:
        return (
            "PROMISING_RETROSPECTIVE_DEVELOPMENT_SIGNAL_ONLY",
            "the frozen selector exceeds the exact matched-null mean at the configured exploratory alpha",
        )
    if hits <= null_mean:
        return (
            "NO_RETROSPECTIVE_ENRICHMENT",
            "the frozen selector does not exceed the exact matched-null mean",
        )
    return (
        "INCONCLUSIVE_RETROSPECTIVE_ENRICHMENT",
        "the frozen selector is above the matched-null mean but not beyond the exploratory exact-tail threshold",
    )


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_pilot(config_path: Path, repo_root: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_config(config)
    output_dir = repo_root / config["output_dir"]
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {output_dir}")

    inputs = {name: repo_root / value for name, value in config["inputs"].items()}
    seal_validation = verify_cross_artifact_seals(inputs)
    selector_input_hashes = {
        "config_sha256": sha256_file(config_path),
        "calibration_numeric_sha256": sha256_file(inputs["calibration_numeric"]),
        "selected_targets_sha256": sha256_file(inputs["selected_targets"]),
        "runner_sha256": sha256_file(Path(__file__)),
    }
    risks = build_mismatch_onset_risks(
        inputs["calibration_numeric"], config["calibration"]["m_grid"]
    )
    candidates = load_candidates(
        inputs["selected_targets"],
        risks,
        int(config["calibration"]["stablebatch_topk_rank_offset"]),
    )
    constraints = config["constraints"]
    if len(candidates) != int(constraints["expected_target_count"]):
        raise ValueError("unexpected StableBatch target count")
    pairs = group_candidate_pairs(
        candidates,
        int(constraints["expected_victim_count"]),
        int(constraints["candidates_per_victim"]),
    )
    policy_specs = config["policies"]
    policies = tuple(policy_specs)
    selections = freeze_policy_selections(pairs, policy_specs)
    selection_bundle = make_selection_bundle(
        selections, policy_specs, selector_input_hashes, seal_validation, config["action_plan"]
    )

    # The exclusive directory creation and frozen artifact both happen before
    # target_results is opened.  A failed later phase remains a non-overwritten
    # partial evidence directory instead of silently changing the selection.
    output_dir.mkdir(parents=True, exist_ok=False)
    write_json(output_dir / "SELECTION_FROZEN.json", selection_bundle)
    with (output_dir / "selection_plan.jsonl").open("w", encoding="utf-8") as handle:
        for action in selection_bundle["policies"]["errortoken_mismatch_onset"]["action_plan"]:
            handle.write(json.dumps(action, sort_keys=True) + "\n")
    with (output_dir / "calibration_key_risks.jsonl").open("w", encoding="utf-8") as handle:
        for key in sorted(risks):
            stats = risks[key]
            handle.write(
                json.dumps(
                    {
                        "layer": key[0],
                        "expert_id": key[1],
                        "route_rank": key[2],
                        **asdict(stats),
                    },
                    sort_keys=True,
                )
                + "\n"
            )

    outcomes = load_outcomes_after_freeze(inputs["target_results"], candidates, selection_bundle)
    outcome_hash = sha256_file(inputs["target_results"])
    matched_null = enumerate_exact_matched_null(pairs, outcomes)
    evaluations = {
        policy: evaluate_policy(selections[policy], candidates, outcomes, matched_null)
        for policy in policies
    }
    primary = evaluations["errortoken_mismatch_onset"]
    alpha = float(config["evaluation"]["exploratory_exact_tail_alpha"])
    verdict, reason = retrospective_verdict(
        primary, float(matched_null["hit_count_mean"]), alpha
    )
    baseline_comparison = {
        policy: {
            "same_selected_targets": len(
                set(primary["selected_target_ids"]).intersection(
                    evaluations[policy]["selected_target_ids"]
                )
            ),
            "primary_minus_baseline_hits": primary["selected_hits"]
            - evaluations[policy]["selected_hits"],
        }
        for policy in policies
        if policy != "errortoken_mismatch_onset"
    }
    matched_candidates = sum(candidate.mismatch_onset_risk is not None for candidate in candidates)
    summary = {
        "schema_version": "errortoken-cross-artifact-cpu-pilot-summary-v1",
        "status": "COMPLETE_RETROSPECTIVE_DEVELOPMENT_ONLY",
        "verdict": verdict,
        "reason": reason,
        "claim_boundary": config["claim_boundary"],
        "selection_frozen_before_outcome_load": True,
        "selection_sha256": selection_bundle["selection_sha256"],
        "pre_outcome_seal_validation": seal_validation,
        "inputs": {
            **selector_input_hashes,
            "target_results_sha256_loaded_only_after_freeze": outcome_hash,
        },
        "denominators": {
            "calibration_key_count": len(risks),
            "stablebatch_target_count": len(candidates),
            "victim_count": len(pairs),
            "candidates_per_victim": int(constraints["candidates_per_victim"]),
            "budget_per_victim": int(constraints["budget_per_victim"]),
            "targets_with_calibration_key": matched_candidates,
            "target_key_coverage": matched_candidates / len(candidates),
        },
        "rank_conversion": {
            "stablebatch_topk_rank_basis": "zero_based",
            "semanticfence_route_rank_basis": "one_based",
            "applied_offset": int(config["calibration"]["stablebatch_topk_rank_offset"]),
        },
        "matched_null": matched_null,
        "policy_results": evaluations,
        "baseline_comparison": baseline_comparison,
        "action_execution_status": config["action_plan"]["execution_status"],
        "limitations": config["limitations"],
    }
    write_json(output_dir / "summary.json", summary)
    complete = {
        "schema_version": "errortoken-cross-artifact-complete-v1",
        "status": "COMPLETE",
        "summary_sha256": sha256_file(output_dir / "summary.json"),
        "selection_frozen_sha256": sha256_file(output_dir / "SELECTION_FROZEN.json"),
        "action_execution_status": "NOT_EXECUTED_PLAN_ONLY",
    }
    write_json(output_dir / "COMPLETE.json", complete)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    summary = run_pilot(args.config.resolve(), args.repo_root.resolve())
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
