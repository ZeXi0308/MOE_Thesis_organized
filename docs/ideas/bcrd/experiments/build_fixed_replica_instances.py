from __future__ import annotations

"""Freeze native route rows into bounded fixed-replica Oracle instances."""

import argparse
import math
from pathlib import Path
from typing import Mapping, Sequence

try:
    from .core import Contribution, ProtocolError, load_routes, read_json, sha256_file, stable_index, validate_causal_route_v3, validate_identity_conservation, write_json, write_jsonl
except ImportError:
    from core import Contribution, ProtocolError, load_routes, read_json, sha256_file, stable_index, validate_causal_route_v3, validate_identity_conservation, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", action="append", required=True)
    parser.add_argument("--gate1-summary", required=True)
    parser.add_argument("--replicas", type=int, default=2)
    parser.add_argument("--tokens-per-instance", type=int, default=2)
    parser.add_argument("--phase", choices=("prefill", "decode"))
    parser.add_argument("--gate1-concurrency", type=int)
    parser.add_argument(
        "--gate1-policy",
        choices=("current_hash", "current_least_load"),
    )
    parser.add_argument("--calibration-fraction", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _document_id(row: object) -> str | None:
    raw_value = getattr(row, "document_id", "")
    value = "" if raw_value is None else str(raw_value).strip()
    return value or None


def _split_cluster_id(row: object) -> str:
    document_id = _document_id(row)
    if document_id is not None:
        return f"document:{document_id}"
    request_id = str(getattr(row, "request_id", "")).strip()
    if not request_id:
        raise ProtocolError("route row has neither document_id nor request_id")
    return f"request:{request_id}"


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


def assign_cluster_splits(
    rows: Sequence[object], *, calibration_fraction: float, seed: int
) -> dict[str, str]:
    """Assign each document/request cluster exactly once for the whole corpus."""

    if not 0 < calibration_fraction < 1:
        raise ValueError("calibration-fraction must lie in (0,1)")
    clusters = sorted({_split_cluster_id(row) for row in rows})
    if len(clusters) < 2:
        raise ProtocolError("insufficient request/document clusters for a disjoint split")
    threshold = int(calibration_fraction * 10_000)
    assignments = {
        cluster: (
            "calibration"
            if stable_index(cluster, 10_000, seed=seed) < threshold
            else "evaluation"
        )
        for cluster in clusters
    }
    if set(assignments.values()) != {"calibration", "evaluation"}:
        raise ProtocolError(
            "deterministic request/document split did not populate both partitions; "
            "change the preregistered seed or provide more clusters"
        )
    return assignments


def _unique_request_chunks(events: Sequence[dict[str, object]], size: int):
    pending = list(events)
    while pending:
        chosen_indices = []
        request_ids = set()
        for index, event in enumerate(pending):
            request_id = str(event["request_id"])
            if request_id in request_ids:
                continue
            chosen_indices.append(index)
            request_ids.add(request_id)
            if len(chosen_indices) == size:
                break
        if len(chosen_indices) < size:
            return
        chosen = [pending[index] for index in chosen_indices]
        for index in reversed(chosen_indices):
            pending.pop(index)
        yield chosen


def _assign_independent_cluster_ids(instances: Sequence[dict[str, object]]) -> None:
    """Collapse overlapping document blocks before any bootstrap CI."""

    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for instance in instances:
        key = (str(instance["phase"]), str(instance["split"]))
        grouped.setdefault(key, []).append(instance)
    for key, values in grouped.items():
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

        for instance in values:
            members = tuple(str(value) for value in instance["split_cluster_ids"])
            find(members[0])
            for member in members[1:]:
                union(members[0], member)
        components: dict[str, list[str]] = {}
        for member in parent:
            components.setdefault(find(member), []).append(member)
        labels = {
            member: f"{key[0]}:{key[1]}:component:{min(components[find(member)])}"
            for member in parent
        }
        for instance in values:
            members = tuple(str(value) for value in instance["split_cluster_ids"])
            instance["independent_cluster_id"] = labels[members[0]]


def build_instances(
    rows: Sequence[object],
    *,
    replicas: int,
    tokens_per_instance: int,
    phase: str | None,
    calibration_fraction: float,
    seed: int,
) -> tuple[list[dict[str, object]], dict[str, str]]:
    selected = [row for row in rows if phase is None or getattr(row, "phase") == phase]
    if not selected:
        raise ProtocolError("no route rows match the requested phase")
    cluster_splits = assign_cluster_splits(
        selected, calibration_fraction=calibration_fraction, seed=seed
    )
    by_layer: dict[
        tuple[str, str, int, str],
        dict[tuple[str, str], list[object]],
    ] = {}
    for row in selected:
        split = cluster_splits[_split_cluster_id(row)]
        key = (str(getattr(row, "model")), str(getattr(row, "phase")), int(getattr(row, "layer")), split)
        event_key = (str(getattr(row, "request_id")), _event_id(row))
        by_layer.setdefault(key, {}).setdefault(event_key, []).append(row)

    instances: list[dict[str, object]] = []
    for (model, row_phase, layer, split), raw_events in sorted(by_layer.items()):
        events: list[dict[str, object]] = []
        for (request_id, input_event_id), event_rows in raw_events.items():
            clusters = {_split_cluster_id(row) for row in event_rows}
            ready_values = {_dispatch_ready_us(row) for row in event_rows}
            decode_steps = {int(getattr(row, "decode_step", -1)) for row in event_rows}
            document_ids = {
                document_id
                for row in event_rows
                if (document_id := _document_id(row)) is not None
            }
            if len(clusters) != 1 or len(ready_values) != 1 or len(decode_steps) != 1:
                raise ProtocolError(
                    f"inconsistent event identity for {request_id}/{input_event_id}"
                )
            cluster = next(iter(clusters))
            if cluster_splits[cluster] != split:
                raise AssertionError("cluster split changed while building instances")
            events.append(
                {
                    "request_id": request_id,
                    "input_event_id": input_event_id,
                    "cluster_id": cluster,
                    "document_ids": tuple(sorted(document_ids)),
                    "ready_us": next(iter(ready_values)),
                    "decode_step": next(iter(decode_steps)),
                    "rows": event_rows,
                }
            )
        ordered = sorted(
            events,
            key=lambda event: (
                float(event["ready_us"]),
                int(event["decode_step"]),
                str(event["request_id"]),
                str(event["input_event_id"]),
            ),
        )
        for window, chosen in enumerate(
            _unique_request_chunks(ordered, tokens_per_instance)
        ):
            contributions = [
                item
                for event in chosen
                for item in sorted(
                    event["rows"],
                    key=lambda row: (
                        str(getattr(row, "request_id")),
                        _event_id(row),
                        int(getattr(row, "rank")),
                    ),
                )
            ]
            request_ids = sorted({str(getattr(item, "request_id")) for item in contributions})
            document_ids = sorted(
                {
                    document_id
                    for item in contributions
                    if (document_id := _document_id(item)) is not None
                }
            )
            split_cluster_ids = sorted({_split_cluster_id(item) for item in contributions})
            if any(cluster_splits[cluster] != split for cluster in split_cluster_ids):
                raise AssertionError("one instance crossed the frozen cluster split")
            instance_id = f"{model}:{row_phase}:l{layer}:{split}:w{window:06d}"
            instances.append(
                {
                    "schema": "bcrd-instance-v2",
                    "instance_id": instance_id,
                    "model": model,
                    "phase": row_phase,
                    "layer": layer,
                    "replica_count": replicas,
                    "replica_ids": list(range(replicas)),
                    "queue_discipline": "EDF_AFTER_CAUSAL_PER_EXPERT_SEAL",
                    "split": split,
                    "token_count": len(chosen),
                    "request_count": len(request_ids),
                    "request_ids": request_ids,
                    "document_ids": document_ids,
                    "split_cluster_ids": split_cluster_ids,
                    "cluster_id": "|".join(split_cluster_ids),
                    "contribution_count": len(contributions),
                    "legal_targets_by_expert": {
                        str(expert): sorted(
                            set.intersection(
                                *(
                                    set(item.legal_replicas(replicas))
                                    for item in contributions
                                    if int(getattr(item, "expert_id")) == expert
                                )
                            )
                        )
                        for expert in sorted(
                            {int(getattr(item, "expert_id")) for item in contributions}
                        )
                    },
                    "contributions": [item.to_json() for item in contributions],
                }
            )
    if not instances:
        raise ProtocolError("no complete instances could be built")
    _assign_independent_cluster_ids(instances)
    validate_instance_contracts(instances, require_formal_v3=False)
    validate_instance_split_disjointness(instances)
    return instances, cluster_splits


def validate_instance_contracts(
    instances: Sequence[Mapping[str, object]], *, require_formal_v3: bool
) -> dict[str, int]:
    """Validate the complete v2 action-space contract before Oracle/replay."""

    if not instances:
        raise ProtocolError("instance set is empty")
    seen_ids: set[str] = set()
    independent_owners: dict[tuple[str, str, str], str] = {}
    total_contributions = 0
    for instance in instances:
        if instance.get("schema") != "bcrd-instance-v2":
            raise ProtocolError("Gate 2/3 requires bcrd-instance-v2")
        instance_id = str(instance.get("instance_id", "")).strip()
        if not instance_id or instance_id in seen_ids:
            raise ProtocolError("instance_id must be non-empty and unique")
        seen_ids.add(instance_id)
        replicas = int(instance.get("replica_count", 0))
        if replicas < 2 or instance.get("replica_ids") != list(range(replicas)):
            raise ProtocolError(f"{instance_id}: replica identity contract is invalid")
        if instance.get("queue_discipline") != "EDF_AFTER_CAUSAL_PER_EXPERT_SEAL":
            raise ProtocolError(f"{instance_id}: queue discipline is not frozen")
        raw = instance.get("contributions")
        if not isinstance(raw, list) or len(raw) != int(instance.get("contribution_count", -1)):
            raise ProtocolError(f"{instance_id}: contribution count mismatch")
        contributions = [
            Contribution.from_mapping(value)
            for value in raw
            if isinstance(value, Mapping)
        ]
        if len(contributions) != len(raw):
            raise ProtocolError(f"{instance_id}: contribution payload is not an object")
        validate_identity_conservation(contributions)
        if require_formal_v3:
            validate_causal_route_v3(contributions, require_observed_stages=True)
        model = str(instance.get("model", ""))
        phase = str(instance.get("phase", ""))
        layer = int(instance.get("layer", -1))
        if any(
            item.model != model or item.phase != phase or item.layer != layer
            for item in contributions
        ):
            raise ProtocolError(f"{instance_id}: contribution scope differs from instance")
        if any(item.source_rank >= replicas for item in contributions):
            raise ProtocolError(f"{instance_id}: source rank is outside frozen replicas")
        request_ids = sorted({item.request_id for item in contributions})
        events_by_request: dict[str, set[str]] = {}
        for item in contributions:
            events_by_request.setdefault(item.request_id, set()).add(item.input_event_id)
        if any(len(event_ids) != 1 for event_ids in events_by_request.values()):
            raise ProtocolError(f"{instance_id}: instance contains sequential events from one request")
        document_ids = sorted({item.document_id for item in contributions})
        split_clusters = sorted({_split_cluster_id(item) for item in contributions})
        if instance.get("request_ids") != request_ids:
            raise ProtocolError(f"{instance_id}: request identity manifest mismatch")
        if instance.get("document_ids") != document_ids:
            raise ProtocolError(f"{instance_id}: document identity manifest mismatch")
        if instance.get("split_cluster_ids") != split_clusters:
            raise ProtocolError(f"{instance_id}: split-cluster manifest mismatch")
        if instance.get("cluster_id") != "|".join(split_clusters):
            raise ProtocolError(f"{instance_id}: cluster_id is not canonical")
        independent_cluster = str(instance.get("independent_cluster_id", "")).strip()
        if not independent_cluster:
            raise ProtocolError(f"{instance_id}: independent_cluster_id is missing")
        split = str(instance.get("split", ""))
        for cluster in split_clusters:
            owner_key = (phase, split, cluster)
            prior = independent_owners.setdefault(owner_key, independent_cluster)
            if prior != independent_cluster:
                raise ProtocolError(
                    f"{instance_id}: one document belongs to multiple independent clusters"
                )
        expected_legal = {
            str(expert): sorted(
                set.intersection(
                    *(
                        set(item.legal_replicas(replicas))
                        for item in contributions
                        if item.expert_id == expert
                    )
                )
            )
            for expert in sorted({item.expert_id for item in contributions})
        }
        if any(not replicas_for_expert for replicas_for_expert in expected_legal.values()):
            raise ProtocolError(f"{instance_id}: an expert has no common legal target")
        if instance.get("legal_targets_by_expert") != expected_legal:
            raise ProtocolError(f"{instance_id}: legal target manifest mismatch")
        total_contributions += len(contributions)
    return {"instances": len(instances), "contributions": total_contributions}


def validate_instance_metadata(
    instance_path: str | Path,
    *,
    service_curve_path: str | Path,
    expected_smoke: bool,
) -> Mapping[str, object]:
    """Bind Gate 2/3 to the exact Gate-1 traces and service curve."""

    path = Path(instance_path)
    meta_path = path.with_suffix(".meta.json")
    if not meta_path.is_file():
        raise ProtocolError(f"instance metadata is missing: {meta_path}")
    meta = read_json(meta_path)
    if not isinstance(meta, Mapping) or meta.get("schema") != "bcrd-instances-meta-v2":
        raise ProtocolError("Gate 2/3 requires bcrd-instances-meta-v2")
    if bool(meta.get("smoke")) != expected_smoke:
        raise ProtocolError("instance metadata smoke/formal mode mismatch")
    if meta.get("output_sha256") != sha256_file(path):
        raise ProtocolError("instance JSONL hash differs from its metadata")
    if (
        meta.get("replay_scope") != "single_layer_window"
        or meta.get("counterfactual_request_dag_propagation") is not False
    ):
        raise ProtocolError("instance replay scope is missing or misrepresented")
    if meta.get("gate1_service_curve_sha256") != sha256_file(service_curve_path):
        raise ProtocolError("service curve differs from the Gate-1 frozen curve")
    return meta


def validate_instance_split_disjointness(
    instances: Sequence[Mapping[str, object]],
) -> dict[str, int]:
    """Fail closed if any request/document cluster crosses the frozen split."""

    requests = {"calibration": set(), "evaluation": set()}
    documents = {"calibration": set(), "evaluation": set()}
    clusters = {"calibration": set(), "evaluation": set()}
    cluster_owner: dict[str, str] = {}
    for instance in instances:
        split = str(instance.get("split", ""))
        if split not in requests:
            raise ProtocolError(f"invalid instance split {split!r}")
        request_ids = {str(value) for value in instance.get("request_ids", ())}
        document_ids = {str(value) for value in instance.get("document_ids", ())}
        split_clusters = {str(value) for value in instance.get("split_cluster_ids", ())}
        if not split_clusters:
            split_clusters = (
                {f"document:{value}" for value in document_ids}
                if document_ids
                else {f"request:{value}" for value in request_ids}
            )
        if not request_ids or not split_clusters:
            raise ProtocolError("instance lacks request/document split identities")
        for cluster in split_clusters:
            previous = cluster_owner.setdefault(cluster, split)
            if previous != split:
                raise ProtocolError(
                    f"request/document cluster {cluster!r} crosses calibration/evaluation"
                )
        requests[split].update(request_ids)
        documents[split].update(document_ids)
        clusters[split].update(split_clusters)
    if not clusters["calibration"] or not clusters["evaluation"]:
        raise ProtocolError("insufficient request/document clusters for both partitions")
    if overlap := requests["calibration"] & requests["evaluation"]:
        raise ProtocolError(f"requests cross calibration/evaluation: {sorted(overlap)}")
    if overlap := documents["calibration"] & documents["evaluation"]:
        raise ProtocolError(f"documents cross calibration/evaluation: {sorted(overlap)}")
    if overlap := clusters["calibration"] & clusters["evaluation"]:
        raise ProtocolError(f"split clusters overlap: {sorted(overlap)}")
    return {
        "calibration_requests": len(requests["calibration"]),
        "evaluation_requests": len(requests["evaluation"]),
        "calibration_documents": len(documents["calibration"]),
        "evaluation_documents": len(documents["evaluation"]),
        "calibration_clusters": len(clusters["calibration"]),
        "evaluation_clusters": len(clusters["evaluation"]),
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    gate1 = read_json(args.gate1_summary)
    if not isinstance(gate1, dict):
        raise ProtocolError("Gate 1 summary must be an object")
    if gate1.get("schema") != "bcrd-gate1-v2":
        raise ProtocolError("instance construction requires bcrd-gate1-v2")
    if bool(gate1.get("smoke")) != bool(args.smoke):
        raise ProtocolError("Gate 1 smoke/formal mode mismatch")
    accepted = {"PASS_GATE1"} | ({"SMOKE_ONLY"} if args.smoke else set())
    if gate1.get("status") not in accepted:
        raise ProtocolError(f"Gate 1 status {gate1.get('status')!r} does not authorize Gate 2")
    if args.replicas < 2 or args.tokens_per_instance <= 0:
        raise ValueError("replicas>=2 and tokens-per-instance>0 are required")
    if not 0 < args.calibration_fraction < 1:
        raise ValueError("calibration-fraction must lie in (0,1)")
    if not args.smoke and (
        args.tokens_per_instance != 2
        or args.calibration_fraction != 0.3
        or args.seed != 20260725
    ):
        raise ProtocolError(
            "formal Gate 2 instances require tokens-per-instance=2, calibration-fraction=0.3 and seed=20260725"
        )

    gate1_inputs = gate1.get("inputs")
    if not isinstance(gate1_inputs, Mapping):
        raise ProtocolError("Gate 1 input hash manifest is missing")
    gate1_trace_hashes = gate1_inputs.get("trace_sha256")
    if not isinstance(gate1_trace_hashes, Mapping) or sorted(gate1_trace_hashes.values()) != sorted(
        sha256_file(path) for path in args.trace
    ):
        raise ProtocolError("Gate 1 trace hashes do not match instance inputs")
    gate1_curve_hash = gate1_inputs.get("service_curve_sha256")
    if not isinstance(gate1_curve_hash, str) or len(gate1_curve_hash) != 64:
        raise ProtocolError("Gate 1 service-curve hash is missing")
    frozen_gate1_cell = None
    if not args.smoke:
        if args.phase is None or args.gate1_concurrency is None or args.gate1_policy is None:
            raise ProtocolError(
                "formal instance construction requires --phase, --gate1-concurrency and --gate1-policy"
            )
        frozen_gate1_cell = (
            args.phase,
            args.gate1_concurrency,
            args.replicas,
            args.gate1_policy,
        )
        passing10 = {tuple(value) for value in gate1.get("passing_common_10pct_cells", ())}
        passing15 = {tuple(value) for value in gate1.get("passing_common_15pct_cells", ())}
        if frozen_gate1_cell not in passing10 or frozen_gate1_cell not in passing15:
            raise ProtocolError("requested Gate-2 cell did not pass the frozen Gate-1 predicates")

    rows = load_routes(args.trace, require_explicit_v3=not args.smoke)
    instances, cluster_splits = build_instances(
        rows,
        replicas=args.replicas,
        tokens_per_instance=args.tokens_per_instance,
        phase=args.phase,
        calibration_fraction=args.calibration_fraction,
        seed=args.seed,
    )
    split_audit = validate_instance_split_disjointness(instances)

    write_jsonl(args.output, instances)
    meta = {
        "schema": "bcrd-instances-meta-v2",
        "smoke": bool(args.smoke),
        "instances": len(instances),
        "models": sorted({str(row["model"]) for row in instances}),
        "replicas": args.replicas,
        "replica_ids": list(range(args.replicas)),
        "queue_discipline": "per-replica EDF after per-(replica,expert) causal seal",
        "replay_scope": "single_layer_window",
        "counterfactual_request_dag_propagation": False,
        "tokens_per_instance": args.tokens_per_instance,
        "split_unit": "document_id_else_request_id",
        "split_clusters": len(cluster_splits),
        "split_audit": split_audit,
        "frozen_gate1_cell": list(frozen_gate1_cell) if frozen_gate1_cell else None,
        "max_contributions": max(int(row["contribution_count"]) for row in instances),
        "trace_sha256": {str(path): sha256_file(path) for path in args.trace},
        "gate1_summary_sha256": sha256_file(args.gate1_summary),
        "gate1_service_curve_sha256": gate1_curve_hash,
        "gate1_exposure_sha256": gate1_inputs.get("exposure_sha256"),
        "output_sha256": sha256_file(args.output),
        "evidence_boundary": (
            "SMOKE_ONLY fixed-replica instances" if args.smoke else
            "frozen logical assignment instances; no physical EP execution"
        ),
    }
    write_json(Path(args.output).with_suffix(".meta.json"), meta)
    return meta


def main() -> None:
    args = parse_args()
    result = run(args)
    print(f"wrote {result['instances']} instances")


if __name__ == "__main__":
    main()
