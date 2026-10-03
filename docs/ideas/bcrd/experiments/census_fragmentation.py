from __future__ import annotations

"""Gate 1: measure assignment-induced expert work fragmentation on native routes."""

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

try:
    from .core import ProtocolError, ServiceCatalog, clustered_bootstrap_mean_ci, load_routes, read_json, sha256_file, write_json
    from .policies import assign_online, make_policy
except ImportError:
    from core import ProtocolError, ServiceCatalog, clustered_bootstrap_mean_ci, load_routes, read_json, sha256_file, write_json
    from policies import assign_online, make_policy


FORMAL_REPLICAS = (2, 4, 8)
FORMAL_CONCURRENCY = (1, 4, 16, 64)
FORMAL_POLICIES = ("current_hash", "current_least_load")
FORMAL_SERVICE_SURFACE_CONSUMER_READY = False


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
    parser.add_argument("--min-documents-per-model-phase", type=int)
    parser.add_argument("--min-independent-clusters-per-cell", type=int, default=2)
    parser.add_argument("--exposure-csv", help="required for a formal E2E-equivalent decision")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def _event_id(row: object) -> str:
    value = str(getattr(row, "input_event_id", "")).strip()
    if value:
        return value
    return (
        f"{getattr(row, 'request_id')}:{getattr(row, 'phase')}:"
        f"{int(getattr(row, 'decode_step', -1))}:"
        f"{int(getattr(row, 'token_position'))}"
    )


def _dispatch_ready_us(row: object) -> float:
    value = getattr(row, "dispatch_ready_us", None)
    ready = float(getattr(row, "arrival_us") if value is None else value)
    if not math.isfinite(ready) or ready < 0:
        raise ProtocolError("dispatch-ready timestamp must be finite and non-negative")
    return ready


def _cluster_id(row: object) -> str:
    document_id = str(getattr(row, "document_id", "")).strip()
    if document_id:
        return f"document:{document_id}"
    return f"request:{getattr(row, 'request_id')}"


def _waves(rows, concurrency: int):
    """Yield causal input-event waves with at most one event per request.

    Route-v2 rows do not carry a physical dispatch-ready timestamp, so their
    request arrival remains a development-only fallback.  Even on that legacy
    input, sequential decode events are never pooled into one artificial batch.
    """

    if concurrency <= 0:
        raise ValueError("concurrency must be positive")
    grouped = {}
    for row in rows:
        grouped.setdefault((row.model, row.phase, row.layer), {}).setdefault(
            (row.request_id, _event_id(row)), []
        ).append(row)
    for (model, phase, layer), raw_events in sorted(grouped.items()):
        events = []
        for (request_id, input_event_id), event_rows in raw_events.items():
            ready_values = {_dispatch_ready_us(row) for row in event_rows}
            decode_steps = {int(getattr(row, "decode_step", -1)) for row in event_rows}
            cluster_ids = {_cluster_id(row) for row in event_rows}
            if len(ready_values) != 1 or len(decode_steps) != 1 or len(cluster_ids) != 1:
                raise ProtocolError(
                    f"inconsistent causal identity inside event {input_event_id!r}"
                )
            events.append(
                {
                    "request_id": request_id,
                    "input_event_id": input_event_id,
                    "ready_us": next(iter(ready_values)),
                    "decode_step": next(iter(decode_steps)),
                    "cluster_id": next(iter(cluster_ids)),
                    "rows": event_rows,
                    "end_us": max(
                        float(getattr(row, "combine_end_us", -1.0))
                        for row in event_rows
                    ),
                }
            )
        pending = sorted(
            events,
            key=lambda event: (
                float(event["ready_us"]),
                int(event["decode_step"]),
                str(event["request_id"]),
                str(event["input_event_id"]),
            ),
        )
        wave_id = 0
        has_observed_intervals = all(float(event["end_us"]) >= 0 for event in pending)
        while pending:
            anchor = pending[0]
            chosen_indices: list[int] = []
            if has_observed_intervals:
                # Interval cliques have a witness at the maximum start time of
                # their members. Search those witnesses while requiring the
                # oldest pending anchor; a greedy early overlap can otherwise
                # hide a later valid size-C active set.
                witness_times = sorted(
                    {
                        float(anchor["ready_us"]),
                        *(
                            float(event["ready_us"])
                            for event in pending[1:]
                            if float(anchor["ready_us"]) - 1e-12
                            <= float(event["ready_us"])
                            < float(anchor["end_us"]) - 1e-12
                        ),
                    }
                )
                for witness in witness_times:
                    if witness >= float(anchor["end_us"]) - 1e-12:
                        continue
                    candidate_indices = [0]
                    candidate_requests = {str(anchor["request_id"])}
                    for index, event in enumerate(pending[1:], start=1):
                        request_id = str(event["request_id"])
                        if request_id in candidate_requests:
                            continue
                        if not (
                            float(event["ready_us"]) <= witness + 1e-12
                            and float(event["end_us"]) > witness + 1e-12
                        ):
                            continue
                        candidate_indices.append(index)
                        candidate_requests.add(request_id)
                        if len(candidate_indices) == concurrency:
                            chosen_indices = candidate_indices
                            break
                    if chosen_indices:
                        break
            else:
                chosen_requests = set()
                for index, event in enumerate(pending):
                    request_id = str(event["request_id"])
                    if request_id in chosen_requests:
                        continue
                    chosen_indices.append(index)
                    chosen_requests.add(request_id)
                    if len(chosen_indices) == concurrency:
                        break
            if len(chosen_indices) < concurrency:
                # The anchor cannot form an observed active-set cell at this
                # concurrency. Drop it; never pool it with a distant event.
                pending.pop(0)
                continue
            chosen = [pending[index] for index in chosen_indices]
            for index in reversed(chosen_indices):
                pending.pop(index)
            wave = [row for event in chosen for row in event["rows"]]
            event_keys = {
                (str(event["request_id"]), str(event["input_event_id"]))
                for event in chosen
            }
            chosen_requests = {str(event["request_id"]) for event in chosen}
            if len(event_keys) != len(chosen_requests):
                raise AssertionError("one wave mixed multiple input events from one request")
            cluster_members = tuple(sorted(str(event["cluster_id"]) for event in chosen))
            cluster_id = "|".join(cluster_members)
            yield model, phase, layer, wave_id, cluster_id, cluster_members, wave
            wave_id += 1


def _connected_component_labels(member_sets: Sequence[Sequence[str]]) -> list[str]:
    """Give overlapping document sets one conservative independent-unit label."""

    parent: dict[str, str] = {}

    def find(value: str) -> str:
        parent.setdefault(value, value)
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: str, right: str) -> None:
        a, b = find(left), find(right)
        if a != b:
            parent[max(a, b)] = min(a, b)

    for members in member_sets:
        ordered = tuple(sorted(set(members)))
        if not ordered:
            raise ProtocolError("independent-unit member set is empty")
        find(ordered[0])
        for member in ordered[1:]:
            union(ordered[0], member)
    components: dict[str, list[str]] = {}
    for member in parent:
        components.setdefault(find(member), []).append(member)
    labels = {
        member: f"component:{min(components[find(member)])}"
        for member in parent
    }
    return [labels[next(iter(members))] for members in member_sets]


def evaluate_gate1_predicate(
    common: Mapping[tuple[object, ...], Mapping[str, Mapping[str, object]]],
    models: Sequence[str],
    preregistered_keys: Sequence[tuple[object, ...]],
) -> dict[str, object]:
    """Evaluate the frozen Gate-1 cell predicates without swapping quantifiers."""

    model_set = set(models)
    ordered_keys = tuple(
        sorted(set(preregistered_keys), key=lambda key: tuple(str(value) for value in key))
    )
    complete = {
        key: common[key]
        for key in ordered_keys
        if key in common and set(common[key]) == model_set and len(model_set) >= 2
    }
    missing = tuple(key for key in ordered_keys if key not in complete)
    passing10 = tuple(
        key
        for key in ordered_keys
        if key in complete
        and all(
            float(row["point"]) >= 0.10
            and float(row["ci_low"]) > 0.05
            and float(row.get("actionable_expert_work_mass", row["actionable_rate"])) >= 0.20
            for row in complete[key].values()
        )
    )
    # The 15% witness must itself satisfy the LCB/actionability contract.
    passing15 = tuple(
        key
        for key in passing10
        if all(float(row["point"]) >= 0.15 for row in complete[key].values())
    )
    all_low = bool(ordered_keys) and not missing and len(model_set) >= 2 and all(
        all(float(row["point"]) < 0.05 for row in complete[key].values())
        for key in ordered_keys
    )
    return {
        "passing10": passing10,
        "passing15": passing15,
        "missing_preregistered_cells": missing,
        "all_preregistered_common_cells_low": all_low,
    }


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
            if key in output:
                raise ProtocolError(f"duplicate exposure coordinate {key}")
            if not math.isfinite(value) or value <= 0:
                raise ValueError("total_path_us must be finite and positive")
            output[key] = value
    return output


def _validate_exposure_coverage(
    rows: Sequence[object],
    concurrencies: Sequence[int],
    exposure: Mapping[tuple[str, str, int, int], float],
) -> None:
    """Require one exact denominator for every observed route cell."""

    expected = {
        (
            str(getattr(row, "model")),
            str(getattr(row, "phase")),
            int(getattr(row, "layer")),
            int(concurrency),
        )
        for row in rows
        for concurrency in concurrencies
    }
    observed = set(exposure)
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    if missing or extra:
        raise ProtocolError(
            "exposure denominator coordinates do not exactly cover the route matrix: "
            f"missing={missing[:3]}, extra={extra[:3]}"
        )


def _require_routeslack_formal_provenance(args: argparse.Namespace) -> None:
    """Prevent development captures/curves from being promoted by this script."""

    if args.smoke:
        return
    for trace in args.trace:
        meta_path = Path(trace).with_suffix(".meta.json")
        if not meta_path.is_file():
            raise ProtocolError(f"formal trace metadata is missing: {meta_path}")
        meta = read_json(meta_path)
        if not isinstance(meta, dict) or meta.get("formal_eligible") is not True:
            raise ProtocolError(f"trace is not formally eligible: {meta_path}")
        if meta.get("smoke") is not False:
            raise ProtocolError(f"formal trace metadata is smoke or ambiguous: {meta_path}")
        if meta.get("schema") != "bcrd-route-v3":
            raise ProtocolError(f"formal trace must use explicit bcrd-route-v3: {meta_path}")
        if meta.get("temporal_ledger_eligible") is not True:
            raise ProtocolError(f"formal trace lacks a causal stage ledger: {meta_path}")
        if meta.get("output_sha256") != sha256_file(trace):
            raise ProtocolError(f"formal trace differs from its metadata hash: {meta_path}")
    curve_meta_path = Path(args.service_curve).with_suffix(".meta.json")
    if not curve_meta_path.is_file():
        raise ProtocolError(f"formal service/energy metadata is missing: {curve_meta_path}")
    curve_meta = read_json(curve_meta_path)
    if (
        not isinstance(curve_meta, dict)
        or curve_meta.get("routeslack_gate1_eligible") is not True
    ):
        raise ProtocolError(
            f"service curve is not RouteSlack Gate-1 eligible: {curve_meta_path}"
        )
    if curve_meta.get("smoke") is not False:
        raise ProtocolError(
            f"formal service curve metadata is smoke or ambiguous: {curve_meta_path}"
        )
    if curve_meta.get("output_sha256") != sha256_file(args.service_curve):
        raise ProtocolError(
            f"formal service curve differs from its metadata hash: {curve_meta_path}"
        )


def _require_formal_service_surface_consumer(*, smoke: bool) -> None:
    """Do not promote the current layer-level latency proxy to formal Gate 1."""

    if not smoke and not FORMAL_SERVICE_SURFACE_CONSUMER_READY:
        raise ProtocolError(
            "formal Gate 1 expert/dtype service-surface consumer is not implemented"
        )


def _continuous_policy_assignments(
    rows: Sequence[object],
    catalog: ServiceCatalog,
    replicas: int,
    policy_name: str,
    *,
    seed: int,
) -> dict[str, int]:
    """Replay each model/phase/layer stream once, never once per census wave."""

    grouped: dict[tuple[str, str, int], list[object]] = {}
    for row in rows:
        grouped.setdefault(
            (str(getattr(row, "model")), str(getattr(row, "phase")), int(getattr(row, "layer"))),
            [],
        ).append(row)
    output: dict[str, int] = {}
    for group_rows in grouped.values():
        policy = make_policy(policy_name, seed=seed, remote_latency_us=0.0)
        assignments = assign_online(group_rows, policy, catalog, replicas)
        for row, replica in zip(group_rows, assignments):
            identity = str(getattr(row, "route_semantic_id"))
            if identity in output:
                raise ProtocolError(f"duplicate route semantic identity {identity!r}")
            output[identity] = replica
    return output


def _work(
    catalog: ServiceCatalog,
    wave,
    assignments: Sequence[int],
    replica_count: int,
) -> dict[str, object]:
    by_expert = {}
    expert_items = {}
    for item, replica in zip(wave, assignments):
        by_expert.setdefault(item.expert_id, {}).setdefault(replica, 0)
        by_expert[item.expert_id][replica] += 1
        expert_items.setdefault(item.expert_id, []).append(item)
    model, layer = wave[0].model, wave[0].layer
    fragmented = 0.0
    consolidated = 0.0
    actionable = False
    actionable_fragmented_work_us = 0.0
    fragmented_launches = 0
    consolidated_launches = 0
    partition_rows = {}
    for expert, replica_rows in by_expert.items():
        total = sum(replica_rows.values())
        expert_fragmented = sum(
            catalog.estimate_us(model, layer, count)
            for count in replica_rows.values()
        )
        common_legal = set.intersection(
            *(set(item.legal_replicas(replica_count)) for item in expert_items[expert])
        )
        can_consolidate = bool(common_legal)
        expert_consolidated = (
            catalog.estimate_us(model, layer, total)
            if can_consolidate
            else expert_fragmented
        )
        fragmented += expert_fragmented
        consolidated += expert_consolidated
        fragmented_launches += len(replica_rows)
        expert_actionable = can_consolidate and len(replica_rows) >= 2
        actionable = actionable or expert_actionable
        if expert_actionable:
            actionable_fragmented_work_us += expert_fragmented
        partition_rows[str(expert)] = sorted(replica_rows.values(), reverse=True)
        consolidated_launches += 1 if can_consolidate else len(replica_rows)
    return {
        "fragmented_work_us": fragmented,
        "consolidated_work_us": consolidated,
        "actionable": actionable,
        "actionable_fragmented_work_us": actionable_fragmented_work_us,
        "fragmented_launches": fragmented_launches,
        "consolidated_launches": consolidated_launches,
        "extra_launches": fragmented_launches - consolidated_launches,
        # One event means one additional replica has to touch that expert's
        # weights. Actual bytes remain backend/cache dependent.
        "extra_weight_read_proxy_events": fragmented_launches - consolidated_launches,
        "partition_rows_json": json.dumps(partition_rows, sort_keys=True),
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    if not args.smoke:
        if tuple(sorted(args.replicas)) != FORMAL_REPLICAS or len(args.replicas) != len(FORMAL_REPLICAS):
            raise ProtocolError(f"formal Gate 1 replicas must be {FORMAL_REPLICAS}")
        if tuple(sorted(args.concurrency)) != FORMAL_CONCURRENCY or len(args.concurrency) != len(FORMAL_CONCURRENCY):
            raise ProtocolError(f"formal Gate 1 concurrency must be {FORMAL_CONCURRENCY}")
        if tuple(args.policies) != FORMAL_POLICIES:
            raise ProtocolError(f"formal Gate 1 policies must be {FORMAL_POLICIES}")
        if args.bootstrap != 2000 or args.seed != 20260725:
            raise ProtocolError("formal Gate 1 statistics require 2000 bootstraps and seed 20260725")
    if args.min_independent_clusters_per_cell < 2:
        raise ProtocolError("at least two independent clusters per formal cell are required")
    _require_routeslack_formal_provenance(args)
    _require_formal_service_surface_consumer(smoke=bool(args.smoke))
    rows = load_routes(args.trace, require_explicit_v3=not args.smoke)
    catalog = ServiceCatalog.from_csv(args.service_curve)
    exposure = _load_exposure(args.exposure_csv)
    if exposure:
        _validate_exposure_coverage(rows, args.concurrency, exposure)
    continuous_assignments = {
        (replicas, policy_name): _continuous_policy_assignments(
            rows,
            catalog,
            replicas,
            policy_name,
            seed=args.seed,
        )
        for replicas in args.replicas
        for policy_name in args.policies
    }
    raw = []
    for concurrency in args.concurrency:
        for model, phase, layer, wave_id, cluster_id, cluster_members, wave in _waves(rows, concurrency):
            for replicas in args.replicas:
                for policy_name in args.policies:
                    assignment_map = continuous_assignments[(replicas, policy_name)]
                    assignments = [
                        assignment_map[str(row.route_semantic_id)] for row in wave
                    ]
                    work = _work(catalog, wave, assignments, replicas)
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
                            "cluster_members_json": json.dumps(cluster_members),
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
        component_labels = _connected_component_labels(
            [json.loads(str(row["cluster_members_json"])) for row in values]
        )
        for row, label in zip(values, component_labels):
            row["independent_cluster_id"] = label
        metrics = [float(row[metric_name]) for row in values if row[metric_name] is not None]
        clusters = [str(row["independent_cluster_id"]) for row in values if row[metric_name] is not None]
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
                "actionable_expert_work_mass": (
                    sum(float(row["actionable_fragmented_work_us"]) for row in values)
                    / max(
                        sum(float(row["fragmented_work_us"]) for row in values),
                        1e-12,
                    )
                ),
                "waves": len(values),
                "request_clusters": len(set(clusters)),
            }
        )

    models = sorted({str(row["model"]) for row in summaries})
    request_counts = {}
    document_counts = {}
    for row in rows:
        request_counts.setdefault((row.model, row.phase), set()).add(row.request_id)
        document_counts.setdefault((row.model, row.phase), set()).add(row.document_id)
    minimum_documents = 128 if not args.smoke else (
        args.min_requests_per_model_phase
        if args.min_documents_per_model_phase is None
        else args.min_documents_per_model_phase
    )
    if not args.smoke and (
        args.min_requests_per_model_phase != 128
        or args.min_documents_per_model_phase not in (None, 128)
    ):
        raise ProtocolError(
            "formal Gate 1 requires the frozen 128-document minimum without override"
        )
    undersized = {
        f"{model}:{phase}": len(document_ids)
        for (model, phase), document_ids in document_counts.items()
        if len(document_ids) < minimum_documents
    }
    undersized_cells = [
        (
            row["model"], row["phase"], row["concurrency"],
            row["replicas"], row["policy"], row["request_clusters"],
        )
        for row in summaries
        if int(row["request_clusters"]) < args.min_independent_clusters_per_cell
    ]
    common = {}
    for row in summaries:
        key = (row["phase"], row["concurrency"], row["replicas"], row["policy"])
        common.setdefault(key, {})[row["model"]] = row
    preregistered_keys = tuple(
        (phase, concurrency, replicas, policy)
        for phase in sorted({row.phase for row in rows})
        for concurrency in sorted(set(args.concurrency))
        for replicas in sorted(set(args.replicas))
        for policy in sorted(set(args.policies))
    )
    gate1 = evaluate_gate1_predicate(common, models, preregistered_keys)
    passing10 = list(gate1["passing10"])
    passing15 = list(gate1["passing15"])
    if args.smoke:
        status = "SMOKE_ONLY"
    elif undersized:
        status = "INVALID_INSUFFICIENT_NATURAL_DOCUMENTS"
    elif undersized_cells:
        status = "INVALID_INSUFFICIENT_INDEPENDENT_UNITS"
    elif not exposure:
        status = "INVALID_MISSING_EXPOSED_PATH_DENOMINATOR"
    elif len(models) < 2:
        status = "INVALID_NEED_TWO_MODELS"
    elif gate1["missing_preregistered_cells"]:
        status = "INVALID_INCOMPLETE_PREREGISTERED_COMMON_CELLS"
    elif passing10 and passing15:
        status = "PASS_GATE1"
    elif gate1["all_preregistered_common_cells_low"]:
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
        "schema": "bcrd-gate1-v2",
        "status": status,
        "smoke": bool(args.smoke),
        "formal_metric": metric_name,
        "models": models,
        "request_counts": {
            f"{model}:{phase}": len(request_ids)
            for (model, phase), request_ids in sorted(request_counts.items())
        },
        "document_counts": {
            f"{model}:{phase}": len(document_ids)
            for (model, phase), document_ids in sorted(document_counts.items())
        },
        "minimum_documents_per_model_phase": minimum_documents,
        "undersized_model_phases": undersized,
        "minimum_independent_clusters_per_cell": args.min_independent_clusters_per_cell,
        "undersized_independent_cells": [list(key) for key in undersized_cells],
        "preregistered_common_cells": [list(key) for key in preregistered_keys],
        "missing_preregistered_common_cells": [
            list(key) for key in gate1["missing_preregistered_cells"]
        ],
        "passing_common_10pct_cells": [list(key) for key in passing10],
        "passing_common_15pct_cells": [list(key) for key in passing15],
        "all_preregistered_common_cells_below_5pct": bool(
            gate1["all_preregistered_common_cells_low"]
        ),
        "cells": summaries,
        "inputs": {
            "trace_sha256": {str(path): sha256_file(path) for path in args.trace},
            "service_curve_sha256": sha256_file(args.service_curve),
            "exposure_sha256": sha256_file(args.exposure_csv) if args.exposure_csv else None,
        },
        "online_baseline_replay": "one continuous causal stream per model/phase/layer",
        "hash_salt_scope": "one fixed run seed; wave/instance ids are excluded",
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
