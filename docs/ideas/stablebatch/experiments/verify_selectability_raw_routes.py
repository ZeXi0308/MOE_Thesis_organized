#!/usr/bin/env python3
"""Rebuild the Selectability verdict from raw per-arm route lists.

This post-run verifier is intentionally independent of the primary policy and
recompute modules.  Its first responsibility is to derive every changed-layer
set from ``reference_arm`` and each raw U/A arm, rather than trusting the
changed-layer fields emitted by the GPU runner.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from fractions import Fraction
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


class VerificationError(RuntimeError):
    pass


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_manifest(run_dir: Path) -> str:
    manifest_path = run_dir / "MANIFEST.json"
    manifest = load_json(manifest_path)
    expected = manifest["files"]
    actual = {
        path.name for path in run_dir.iterdir() if path.is_file() and path.name != "MANIFEST.json"
    }
    if actual != set(expected):
        raise VerificationError("run manifest file set mismatch")
    for name, binding in expected.items():
        path = run_dir / name
        if path.stat().st_size != int(binding["size_bytes"]):
            raise VerificationError(f"size mismatch: {name}")
        if sha256_file(path) != str(binding["sha256"]):
            raise VerificationError(f"hash mismatch: {name}")
    return sha256_file(manifest_path)


def changed_membership_layers(
    reference: Sequence[Sequence[int]], arm: Sequence[Sequence[int]], start: int
) -> list[int]:
    if len(reference) != len(arm):
        raise VerificationError("raw route layer counts differ")
    return [
        layer
        for layer in range(start, len(reference))
        if set(map(int, reference[layer])) != set(map(int, arm[layer]))
    ]


def decomposition(unprotected: Sequence[int], action: Sequence[int]) -> dict[str, int]:
    u = set(map(int, unprotected))
    a = set(map(int, action))
    return {
        "recovered": len(u - a),
        "harmed": len(a - u),
        "persistent": len(u & a),
        "reward": len(u - a) - len(a - u),
    }


def raw_outcomes(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[tuple[str, int], dict[str, Any]], list[str], int]:
    lookup: dict[tuple[str, int], dict[str, Any]] = {}
    mismatches: list[str] = []
    total_unprotected = 0
    for row in rows:
        identity = str(row["cell_identity"])
        start = int(row["layer"]) + 1
        reference = row["reference_arm"]["topk_experts_by_layer"]
        unprotected = changed_membership_layers(
            reference, row["unprotected_arm"]["topk_experts_by_layer"], start
        )
        total_unprotected += len(unprotected)
        if list(map(int, row["unprotected_changed_layers_vs_R"])) != unprotected:
            mismatches.append(f"{identity}:stored_unprotected_layers")
        if int(row["unprotected_distance_vs_R"]) != len(unprotected):
            mismatches.append(f"{identity}:stored_unprotected_distance")
        for rank in range(8):
            stored = row["actions"][str(rank)]
            action = changed_membership_layers(
                reference, stored["arm"]["topk_experts_by_layer"], start
            )
            values = decomposition(unprotected, action)
            if list(map(int, stored["changed_layers_vs_R"])) != action:
                mismatches.append(f"{identity}:rank={rank}:stored_action_layers")
            if int(stored["distance_vs_R"]) != len(action):
                mismatches.append(f"{identity}:rank={rank}:stored_action_distance")
            if int(stored["reward"]) != values["reward"]:
                mismatches.append(f"{identity}:rank={rank}:stored_action_reward")
            lookup[(identity, rank)] = {
                **values,
                "cell_identity": identity,
                "victim_id": str(row["victim_id"]),
                "rank": rank,
            }
    return lookup, mismatches, total_unprotected


def oracle_ranking(
    rows: Sequence[Mapping[str, Any]],
    lookup: Mapping[tuple[str, int], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    winners: list[dict[str, Any]] = []
    for row in rows:
        identity = str(row["cell_identity"])
        winner = min(
            (lookup[(identity, rank)] for rank in range(8)),
            key=lambda value: (
                -int(value["reward"]), int(value["harmed"]), int(value["rank"])
            ),
        )
        winners.append(dict(winner))
    return sorted(
        winners,
        key=lambda value: (
            -int(value["reward"]),
            int(value["harmed"]),
            int(value["rank"]),
            str(value["cell_identity"]),
        ),
    )


def select(
    ranking: Sequence[Mapping[str, Any]], budget: int, excluded_victim: str | None = None
) -> list[Mapping[str, Any]]:
    selected = [
        row
        for row in ranking
        if excluded_victim is None or str(row["victim_id"]) != excluded_victim
    ][:budget]
    if len(selected) != budget:
        raise VerificationError("cannot refill exact action budget")
    if len({str(row["cell_identity"]) for row in selected}) != budget:
        raise VerificationError("selected plan repeats a cell")
    return selected


def aggregate(
    plan: Sequence[Mapping[str, Any]],
    lookup: Mapping[tuple[str, int], Mapping[str, Any]],
) -> dict[str, Any]:
    outcomes = [lookup[(str(row["cell_identity"]), int(row["rank"]))] for row in plan]
    per_victim: dict[str, dict[str, int]] = defaultdict(
        lambda: {"reward": 0, "recovered": 0, "harmed": 0, "actions": 0}
    )
    for outcome in outcomes:
        victim = str(outcome["victim_id"])
        for field in ("reward", "recovered", "harmed"):
            per_victim[victim][field] += int(outcome[field])
        per_victim[victim]["actions"] += 1
    return {
        "actions": len(outcomes),
        "reward": sum(int(row["reward"]) for row in outcomes),
        "recovered": sum(int(row["recovered"]) for row in outcomes),
        "harmed": sum(int(row["harmed"]) for row in outcomes),
        "positive_net_victims": sorted(
            victim for victim, value in per_victim.items() if value["reward"] > 0
        ),
        "per_victim": {key: dict(value) for key, value in sorted(per_victim.items())},
    }


def classify(
    rows: Sequence[Mapping[str, Any]],
    lock: Mapping[str, Any],
    config: Mapping[str, Any],
    lookup: Mapping[tuple[str, int], Mapping[str, Any]],
    total_unprotected: int,
) -> dict[str, Any]:
    budget = int(config["selection"]["action_budget_cells"])
    expected_cells = int(config["selection"]["cell_count"])
    if len(rows) != expected_cells:
        raise VerificationError("wrong cell count")
    rankings = {
        "oracle": oracle_ranking(rows, lookup),
        "static": lock["static_plan"]["ranking"],
        "online": lock["online_plan"]["ranking"],
        "shuffle": lock["shuffle_plan"]["ranking"],
    }
    aggregates = {
        name: aggregate(select(ranking, budget), lookup)
        for name, ranking in rankings.items()
    }
    denominator = aggregates["oracle"]["reward"] - aggregates["shuffle"]["reward"]
    gaps = {
        name: (
            (aggregates[name]["reward"] - aggregates["shuffle"]["reward"]) / denominator
            if denominator > 0
            else None
        )
        for name in ("static", "online")
    }
    oracle_selected = select(rankings["oracle"], budget)
    positive_oracle_victims = sorted(
        {str(row["victim_id"]) for row in oracle_selected if int(row["reward"]) > 0}
    )
    oracle_recovery_fraction = (
        aggregates["oracle"]["recovered"] / total_unprotected if total_unprotected else 0.0
    )
    oracle_cfg = config["gate"]["oracle_opportunity"]
    oracle_checks = {
        "reward_positive": aggregates["oracle"]["reward"] > 0,
        "above_shuffle": denominator > 0,
        "min_recovery_fraction": oracle_recovery_fraction
        >= float(oracle_cfg["min_recovery_fraction"]),
        "min_positive_victims": len(positive_oracle_victims)
        >= int(oracle_cfg["min_positive_victims"]),
    }
    victims = sorted({str(row["victim_id"]) for row in rows})
    lodo: dict[str, Any] = {}
    for victim in victims:
        values = {
            name: aggregate(select(ranking, budget, victim), lookup)
            for name, ranking in rankings.items()
        }
        fold_denominator = values["oracle"]["reward"] - values["shuffle"]["reward"]
        lodo[victim] = {
            "aggregates": values,
            "oracle_minus_shuffle": fold_denominator,
        }
        for name in ("static", "online"):
            lodo[victim][f"{name}_recovered_oracle_gap"] = (
                (values[name]["reward"] - values["shuffle"]["reward"])
                / fold_denominator
                if fold_denominator > 0
                else None
            )
    min_gap = float(config["gate"]["selector"]["min_recovered_oracle_gap"])
    min_victims = int(config["gate"]["selector"]["min_positive_net_victims"])
    selector_results: dict[str, Any] = {}
    for name in ("static", "online"):
        full_checks = {
            "reward_positive": aggregates[name]["reward"] > 0,
            "above_shuffle": aggregates[name]["reward"] > aggregates["shuffle"]["reward"],
            "min_recovered_oracle_gap": gaps[name] is not None and gaps[name] >= min_gap,
            "min_positive_net_victims": len(aggregates[name]["positive_net_victims"])
            >= min_victims,
        }
        fold_checks = []
        for victim in victims:
            values = lodo[victim]["aggregates"]
            gap = lodo[victim][f"{name}_recovered_oracle_gap"]
            fold_checks.append(
                gap is not None
                and values[name]["reward"] > 0
                and values[name]["reward"] > values["shuffle"]["reward"]
                and gap >= min_gap
            )
        selector_results[name] = {
            "recovered_oracle_gap": gaps[name],
            "full_checks": full_checks,
            "lodo_pass_count": sum(fold_checks),
            "lodo_total": len(fold_checks),
            "go": all(oracle_checks.values())
            and all(full_checks.values())
            and all(fold_checks),
        }
    if not all(oracle_checks.values()):
        verdict = "STOP_NO_FRESH_ORACLE_OPPORTUNITY"
    elif selector_results["static"]["go"]:
        verdict = "GO_STATIC_COMPATIBILITY"
    elif selector_results["online"]["go"]:
        verdict = "GO_ROW_CONDITIONED"
    else:
        verdict = "STOP_PREACTION_STABLEBATCH"
    uniform = Fraction(
        budget * sum(int(value["reward"]) for value in lookup.values()),
        expected_cells * 8,
    )
    return {
        "cell_count": len(rows),
        "action_budget_cells": budget,
        "total_unprotected_route_distance": total_unprotected,
        "aggregates": aggregates,
        "oracle_minus_shuffle": denominator,
        "oracle_recovery_fraction": oracle_recovery_fraction,
        "positive_oracle_victims": positive_oracle_victims,
        "oracle_opportunity_checks": oracle_checks,
        "selector_results": selector_results,
        "uniform_random_expected_reward": {
            "numerator": uniform.numerator,
            "denominator": uniform.denominator,
            "float": float(uniform),
        },
        "leave_one_victim_out": lodo,
        "verdict": verdict,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    manifest_sha256 = verify_manifest(run_dir)
    status = load_json(run_dir / "RUN_STATUS.json")
    if status.get("status") != "COMPLETE" or not status.get("scientific_result_eligible"):
        raise VerificationError("run is not scientifically eligible")
    rows = load_jsonl(run_dir / "cell_results.jsonl")
    lock = load_json(run_dir / "SELECTOR_LOCK.json")
    config = load_json(run_dir / "config_snapshot.json")
    summary = load_json(run_dir / "summary.json")
    lookup, raw_mismatches, total_unprotected = raw_outcomes(rows)
    raw_classification = classify(rows, lock, config, lookup, total_unprotected)
    compared_fields = tuple(raw_classification)
    summary_mismatches = [
        field for field in compared_fields if raw_classification[field] != summary[field]
    ]
    result = {
        "schema_version": "stablebatch-selectability-raw-route-verifier-v1",
        "status": "PASS" if not raw_mismatches and not summary_mismatches else "FAIL",
        "run_dir": str(run_dir),
        "run_manifest_sha256": manifest_sha256,
        "selector_lock_sha256": sha256_file(run_dir / "SELECTOR_LOCK.json"),
        "cell_results_sha256": sha256_file(run_dir / "cell_results.jsonl"),
        "raw_route_checks": {
            "cells": len(rows),
            "unprotected_arms_rederived": len(rows),
            "action_arms_rederived": len(rows) * 8,
            "mismatch_count": len(raw_mismatches),
            "mismatches": raw_mismatches,
        },
        "summary_mismatch_fields": summary_mismatches,
        "raw_classification": raw_classification,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    print(
        json.dumps(
            {
                "status": result["status"],
                "raw_route_mismatch_count": len(raw_mismatches),
                "summary_mismatch_fields": summary_mismatches,
                "verdict": raw_classification["verdict"],
            },
            sort_keys=True,
        )
    )
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
