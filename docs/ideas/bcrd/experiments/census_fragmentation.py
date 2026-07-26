from __future__ import annotations

"""Gate 1: measure assignment-induced expert work fragmentation on native routes."""

import argparse
import csv
import json
from pathlib import Path
from typing import Sequence

try:
    from .core import ServiceCatalog, clustered_bootstrap_mean_ci, load_routes, sha256_file, write_json
    from .policies import assign_online, make_policy
except ImportError:
    from core import ServiceCatalog, clustered_bootstrap_mean_ci, load_routes, sha256_file, write_json
    from policies import assign_online, make_policy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", action="append", required=True)
    parser.add_argument("--service-curve", required=True)
    parser.add_argument("--replicas", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 16, 64])
    parser.add_argument("--policies", nargs="+", default=["current_hash", "current_least_load"])
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--min-requests-per-model-phase", type=int, default=128)
    parser.add_argument("--exposure-csv", help="required for a formal E2E-equivalent decision")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def _waves(rows, concurrency: int):
    grouped = {}
    for row in rows:
        grouped.setdefault((row.model, row.phase, row.layer), []).append(row)
    for (model, phase, layer), layer_rows in sorted(grouped.items()):
        request_order = sorted(
            {row.request_id: row.arrival_us for row in layer_rows}.items(), key=lambda value: (value[1], value[0])
        )
        by_request = {}
        for row in layer_rows:
            by_request.setdefault(row.request_id, []).append(row)
        for start in range(0, len(request_order), concurrency):
            request_ids = [value[0] for value in request_order[start : start + concurrency]]
            if len(request_ids) < concurrency:
                continue
            wave = [item for request_id in request_ids for item in by_request[request_id]]
            yield model, phase, layer, start // concurrency, "|".join(sorted(request_ids)), wave


def _load_exposure(path: str | None) -> dict[tuple[str, str, int, int], float]:
    if path is None:
        return {}
    output = {}
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"model", "phase", "layer", "concurrency", "total_path_us"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"exposure CSV missing {sorted(missing)}")
        for row in reader:
            key = (str(row["model"]), str(row["phase"]), int(row["layer"]), int(row["concurrency"]))
            value = float(row["total_path_us"])
            if value <= 0:
                raise ValueError("total_path_us must be positive")
            output[key] = value
    return output


def _work(catalog: ServiceCatalog, wave, assignments: Sequence[int]) -> dict[str, object]:
    by_expert = {}
    for item, replica in zip(wave, assignments):
        by_expert.setdefault(item.expert_id, {}).setdefault(replica, 0)
        by_expert[item.expert_id][replica] += 1
    model, layer = wave[0].model, wave[0].layer
    fragmented = 0.0
    consolidated = 0.0
    actionable = False
    fragmented_launches = 0
    partition_rows = {}
    for expert, replica_rows in by_expert.items():
        total = sum(replica_rows.values())
        consolidated += catalog.estimate_us(model, layer, total)
        fragmented += sum(catalog.estimate_us(model, layer, count) for count in replica_rows.values())
        fragmented_launches += len(replica_rows)
        actionable = actionable or len(replica_rows) >= 2
        partition_rows[str(expert)] = sorted(replica_rows.values(), reverse=True)
    consolidated_launches = len(by_expert)
    return {
        "fragmented_work_us": fragmented,
        "consolidated_work_us": consolidated,
        "actionable": actionable,
        "fragmented_launches": fragmented_launches,
        "consolidated_launches": consolidated_launches,
        "extra_launches": fragmented_launches - consolidated_launches,
        # One event means one additional replica has to touch that expert's
        # weights. Actual bytes remain backend/cache dependent.
        "extra_weight_read_proxy_events": fragmented_launches - consolidated_launches,
        "partition_rows_json": json.dumps(partition_rows, sort_keys=True),
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    rows = load_routes(args.trace)
    catalog = ServiceCatalog.from_csv(args.service_curve)
    exposure = _load_exposure(args.exposure_csv)
    raw = []
    for concurrency in args.concurrency:
        for model, phase, layer, wave_id, cluster_id, wave in _waves(rows, concurrency):
            for replicas in args.replicas:
                for policy_name in args.policies:
                    policy = make_policy(policy_name, seed=args.seed + wave_id, remote_latency_us=0.0)
                    assignments = assign_online(wave, policy, catalog, replicas)
                    work = _work(catalog, wave, assignments)
                    fragmented = float(work["fragmented_work_us"])
                    consolidated = float(work["consolidated_work_us"])
                    removable = max(0.0, fragmented - consolidated)
                    denominator = exposure.get((model, phase, layer, concurrency))
                    raw.append(
                        {
                            "model": model,
                            "phase": phase,
                            "layer": layer,
                            "wave_id": wave_id,
                            "cluster_id": cluster_id,
                            "concurrency": concurrency,
                            "replicas": replicas,
                            "policy": policy_name,
                            "contributions": len(wave),
                            **work,
                            "expert_work_removable_fraction": removable / fragmented if fragmented else 0.0,
                            "exposed_penalty_fraction": removable / denominator if denominator else None,
                        }
                    )

    cell_rows = {}
    for row in raw:
        key = (row["model"], row["phase"], row["concurrency"], row["replicas"], row["policy"])
        cell_rows.setdefault(key, []).append(row)
    summaries = []
    metric_name = "exposed_penalty_fraction" if exposure else "expert_work_removable_fraction"
    for key, values in sorted(cell_rows.items()):
        metrics = [float(row[metric_name]) for row in values if row[metric_name] is not None]
        clusters = [str(row["cluster_id"]) for row in values if row[metric_name] is not None]
        point, low, high = clustered_bootstrap_mean_ci(
            metrics, clusters, replicates=args.bootstrap, seed=args.seed
        )
        summaries.append(
            {
                "model": key[0],
                "phase": key[1],
                "concurrency": key[2],
                "replicas": key[3],
                "policy": key[4],
                "metric": metric_name,
                "point": point,
                "ci_low": low,
                "ci_high": high,
                "actionable_rate": sum(bool(row["actionable"]) for row in values) / len(values),
                "waves": len(values),
                "request_clusters": len(set(clusters)),
            }
        )

    models = sorted({str(row["model"]) for row in summaries})
    request_counts = {}
    for row in rows:
        request_counts.setdefault((row.model, row.phase), set()).add(row.request_id)
    undersized = {
        f"{model}:{phase}": len(request_ids)
        for (model, phase), request_ids in request_counts.items()
        if len(request_ids) < args.min_requests_per_model_phase
    }
    common = {}
    for row in summaries:
        key = (row["phase"], row["concurrency"], row["replicas"], row["policy"])
        common.setdefault(key, {})[row["model"]] = row
    passing10 = [
        key for key, by_model in common.items()
        if len(by_model) == len(models) >= 2
        and all(float(row["point"]) >= 0.10 and float(row["ci_low"]) > 0.05 and float(row["actionable_rate"]) >= 0.20 for row in by_model.values())
    ]
    passing15 = [
        key for key, by_model in common.items()
        if len(by_model) == len(models) >= 2 and all(float(row["point"]) >= 0.15 for row in by_model.values())
    ]
    if args.smoke:
        status = "SMOKE_ONLY"
    elif undersized:
        status = "INVALID_INSUFFICIENT_NATURAL_REQUESTS"
    elif not exposure:
        status = "INVALID_MISSING_EXPOSED_PATH_DENOMINATOR"
    elif passing10 and passing15:
        status = "PASS_GATE1"
    elif any(len(by_model) == len(models) >= 2 and all(float(row["point"]) < 0.05 for row in by_model.values()) for by_model in common.values()):
        status = "KILL_BCRD"
    else:
        status = "NO_GO_GATE1"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if raw:
        with (output_dir / "fragmentation_raw.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(raw[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(raw)
    payload = {
        "schema": "bcrd-gate1-v1",
        "status": status,
        "smoke": bool(args.smoke),
        "formal_metric": metric_name,
        "models": models,
        "request_counts": {
            f"{model}:{phase}": len(request_ids)
            for (model, phase), request_ids in sorted(request_counts.items())
        },
        "undersized_model_phases": undersized,
        "passing_common_10pct_cells": [list(key) for key in passing10],
        "passing_common_15pct_cells": [list(key) for key in passing15],
        "cells": summaries,
        "inputs": {
            "trace_sha256": {str(path): sha256_file(path) for path in args.trace},
            "service_curve_sha256": sha256_file(args.service_curve),
            "exposure_sha256": sha256_file(args.exposure_csv) if args.exposure_csv else None,
        },
        "evidence_boundary": (
            "SMOKE_ONLY code-path validation" if args.smoke else
            "single-GPU route/service accounting; no multi-GPU A2A or serving P99 claim"
        ),
    }
    write_json(output_dir / "gate1_summary.json", payload)
    return payload


def main() -> None:
    args = parse_args()
    payload = run(args)
    print(payload["status"])


if __name__ == "__main__":
    main()
