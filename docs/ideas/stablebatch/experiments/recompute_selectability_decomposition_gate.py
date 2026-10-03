#!/usr/bin/env python3
"""Independent raw-ledger recompute for the StableBatch Selectability Gate."""

from __future__ import annotations

import argparse
from collections import defaultdict
from fractions import Fraction
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


CLASSIFICATION_FIELDS = (
    "cell_count",
    "action_budget_cells",
    "total_unprotected_route_distance",
    "aggregates",
    "oracle_minus_shuffle",
    "oracle_recovery_fraction",
    "positive_oracle_victims",
    "oracle_opportunity_checks",
    "selector_results",
    "uniform_random_expected_reward",
    "leave_one_victim_out",
    "verdict",
)


class RecomputeError(RuntimeError):
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


def verify_manifest(run_dir: Path) -> None:
    manifest = load_json(run_dir / "MANIFEST.json")
    if manifest.get("schema_version") != "stablebatch-selectability-decomposition-manifest-v1":
        raise RecomputeError("wrong run manifest schema")
    expected = manifest["files"]
    actual = {
        path.name for path in run_dir.iterdir() if path.is_file() and path.name != "MANIFEST.json"
    }
    if actual != set(expected):
        raise RecomputeError("manifest file set mismatch")
    for name, binding in expected.items():
        path = run_dir / name
        if path.stat().st_size != int(binding["size_bytes"]):
            raise RecomputeError(f"size mismatch for {name}")
        if sha256_file(path) != str(binding["sha256"]):
            raise RecomputeError(f"hash mismatch for {name}")


def decompose(unprotected: Sequence[int], action: Sequence[int]) -> dict[str, int]:
    u = set(map(int, unprotected))
    a = set(map(int, action))
    recovered = len(u - a)
    harmed = len(a - u)
    return {
        "recovered": recovered,
        "harmed": harmed,
        "persistent": len(u & a),
        "reward": recovered - harmed,
    }


def outcome_lookup(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
    values: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        identity = str(row["cell_identity"])
        for rank in range(8):
            action = row["actions"][str(rank)]
            result = decompose(
                row["unprotected_changed_layers_vs_R"], action["changed_layers_vs_R"]
            )
            if int(action["reward"]) != result["reward"]:
                raise RecomputeError("stored action reward differs from raw route sets")
            values[(identity, rank)] = {
                **result,
                "cell_identity": identity,
                "victim_id": str(row["victim_id"]),
                "rank": rank,
            }
    return values


def build_oracle(rows: Sequence[Mapping[str, Any]], lookup: Mapping[tuple[str, int], Mapping[str, Any]]) -> list[dict[str, Any]]:
    winners: list[dict[str, Any]] = []
    for row in rows:
        identity = str(row["cell_identity"])
        choices = [lookup[(identity, rank)] for rank in range(8)]
        best = min(
            choices,
            key=lambda value: (
                -int(value["reward"]),
                int(value["harmed"]),
                int(value["rank"]),
            ),
        )
        winners.append(dict(best))
    return sorted(
        winners,
        key=lambda value: (
            -int(value["reward"]),
            int(value["harmed"]),
            int(value["rank"]),
            str(value["cell_identity"]),
        ),
    )


def select(ranking: Sequence[Mapping[str, Any]], budget: int, excluded: str | None = None) -> list[Mapping[str, Any]]:
    values = [row for row in ranking if excluded is None or str(row["victim_id"]) != excluded][:budget]
    if len(values) != budget:
        raise RecomputeError("cannot refill exact action budget")
    return values


def aggregate(plan: Sequence[Mapping[str, Any]], lookup: Mapping[tuple[str, int], Mapping[str, Any]]) -> dict[str, Any]:
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


def classify(rows: Sequence[Mapping[str, Any]], lock: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    expected_cells = int(config["selection"]["cell_count"])
    budget = int(config["selection"]["action_budget_cells"])
    if len(rows) != expected_cells or any(row.get("integrity_status") != "PASS" for row in rows):
        raise RecomputeError("row cardinality/integrity failure")
    lookup = outcome_lookup(rows)
    rankings = {
        "oracle": build_oracle(rows, lookup),
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
    all_rewards = [int(value["reward"]) for value in lookup.values()]
    uniform = Fraction(budget * sum(all_rewards), expected_cells * 8)
    total_unprotected = sum(int(row["unprotected_distance_vs_R"]) for row in rows)
    oracle_selected = select(rankings["oracle"], budget)
    positive_oracle_victims = sorted(
        {str(row["victim_id"]) for row in oracle_selected if int(row["reward"]) > 0}
    )
    oracle_recovery_fraction = (
        float(aggregates["oracle"]["recovered"] / total_unprotected)
        if total_unprotected
        else 0.0
    )
    oracle_cfg = config["gate"]["oracle_opportunity"]
    oracle_checks = {
        "reward_positive": aggregates["oracle"]["reward"] > 0,
        "above_shuffle": denominator > 0,
        "min_recovery_fraction": oracle_recovery_fraction >= float(oracle_cfg["min_recovery_fraction"]),
        "min_positive_victims": len(positive_oracle_victims) >= int(oracle_cfg["min_positive_victims"]),
    }
    victims = sorted({str(row["victim_id"]) for row in rows})
    lodo: dict[str, Any] = {}
    for victim in victims:
        values = {
            name: aggregate(select(ranking, budget, victim), lookup)
            for name, ranking in rankings.items()
        }
        denom = values["oracle"]["reward"] - values["shuffle"]["reward"]
        lodo[victim] = {"aggregates": values, "oracle_minus_shuffle": denom}
        for name in ("static", "online"):
            lodo[victim][f"{name}_recovered_oracle_gap"] = (
                (values[name]["reward"] - values["shuffle"]["reward"]) / denom
                if denom > 0
                else None
            )
    min_gap = float(config["gate"]["selector"]["min_recovered_oracle_gap"])
    min_victims = int(config["gate"]["selector"]["min_positive_net_victims"])
    selector_results: dict[str, Any] = {}
    for name in ("static", "online"):
        checks = {
            "reward_positive": aggregates[name]["reward"] > 0,
            "above_shuffle": aggregates[name]["reward"] > aggregates["shuffle"]["reward"],
            "min_recovered_oracle_gap": gaps[name] is not None and gaps[name] >= min_gap,
            "min_positive_net_victims": len(aggregates[name]["positive_net_victims"]) >= min_victims,
        }
        lodo_checks = []
        for victim in victims:
            values = lodo[victim]["aggregates"]
            gap = lodo[victim][f"{name}_recovered_oracle_gap"]
            lodo_checks.append(
                bool(
                    gap is not None
                    and values[name]["reward"] > 0
                    and values[name]["reward"] > values["shuffle"]["reward"]
                    and gap >= min_gap
                )
            )
        selector_results[name] = {
            "recovered_oracle_gap": gaps[name],
            "full_checks": checks,
            "lodo_pass_count": sum(lodo_checks),
            "lodo_total": len(lodo_checks),
            "go": all(oracle_checks.values()) and all(checks.values()) and all(lodo_checks),
        }
    if not all(oracle_checks.values()):
        verdict = "STOP_NO_FRESH_ORACLE_OPPORTUNITY"
    elif selector_results["static"]["go"]:
        verdict = "GO_STATIC_COMPATIBILITY"
    elif selector_results["online"]["go"]:
        verdict = "GO_ROW_CONDITIONED"
    else:
        verdict = "STOP_PREACTION_STABLEBATCH"
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
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    verify_manifest(run_dir)
    status = load_json(run_dir / "RUN_STATUS.json")
    if status.get("status") != "COMPLETE" or not status.get("scientific_result_eligible"):
        raise RecomputeError("run is not COMPLETE and scientifically eligible")
    config = load_json(args.config.resolve())
    lock = load_json(run_dir / "SELECTOR_LOCK.json")
    summary = load_json(run_dir / "summary.json")
    if sha256_file(run_dir / "SELECTOR_LOCK.json") != summary["selector_lock_sha256"]:
        raise RecomputeError("summary selector-lock binding mismatch")
    if lock.get("outcome_rows_existed_at_seal") or lock.get("result_path_existed_at_seal"):
        raise RecomputeError("selection lock admits pre-seal outcome rows")
    rows = load_jsonl(run_dir / "cell_results.jsonl")
    independent = classify(rows, lock, config)
    expected = {field: summary[field] for field in CLASSIFICATION_FIELDS}
    mismatches = [field for field in CLASSIFICATION_FIELDS if independent[field] != expected[field]]
    result = {
        "schema_version": "stablebatch-selectability-independent-recompute-v1",
        "status": "PASS" if not mismatches else "FAIL",
        "run_dir": str(run_dir),
        "run_manifest_sha256": sha256_file(run_dir / "MANIFEST.json"),
        "selector_lock_sha256": sha256_file(run_dir / "SELECTOR_LOCK.json"),
        "cell_results_sha256": sha256_file(run_dir / "cell_results.jsonl"),
        "mismatch_fields": mismatches,
        "independent": independent,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, sort_keys=True)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "mismatch_fields": mismatches, "verdict": independent["verdict"]}, sort_keys=True))
    return 0 if not mismatches else 2


if __name__ == "__main__":
    raise SystemExit(main())
