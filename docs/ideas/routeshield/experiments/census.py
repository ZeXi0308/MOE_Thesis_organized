from __future__ import annotations

"""Count-based expert/rank footprint census for RouteShield Gate-0."""

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Iterable, Mapping, Sequence

try:
    from .protocol import load_config
    from .schema import (
        ProtocolError,
        RouteContribution,
        load_expected_events_jsonl,
        load_route_csv,
    )
except ImportError:
    from protocol import load_config
    from schema import (
        ProtocolError,
        RouteContribution,
        load_expected_events_jsonl,
        load_route_csv,
    )


def _normalized_entropy(counts: Mapping[int, int], total_bins: int) -> float:
    if total_bins <= 0:
        raise ProtocolError("entropy bin count must be positive")
    total = sum(counts.values())
    if total <= 0 or total_bins == 1:
        return 0.0
    entropy = 0.0
    for count in counts.values():
        if count <= 0:
            continue
        probability = count / total
        entropy -= probability * math.log(probability)
    return entropy / math.log(total_bins)


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            raise ProtocolError(f"duplicate placement JSON key: {key}")
        output[key] = value
    return output


def _reject_json_constant(value: str) -> None:
    raise ProtocolError(f"non-finite placement JSON value is forbidden: {value}")


def _maximum_share(counts: Mapping[int, int]) -> float:
    total = sum(counts.values())
    return max(counts.values()) / total if total else 0.0


def _distribution(counts: Mapping[int, int], bins: Sequence[int]) -> list[float]:
    total = sum(counts.values())
    if total <= 0:
        return [0.0 for _ in bins]
    return [counts.get(item, 0) / total for item in bins]


def _prefix_future_persistence(
    rows: Sequence[RouteContribution],
    *,
    prefix_fraction: float,
    num_ranks: int,
) -> tuple[float | None, bool | None, float | None, float | None]:
    if not 0.0 < prefix_fraction < 1.0:
        raise ProtocolError("prefix_fraction must be in (0, 1)")
    chunk_counts = Counter(row.chunk_id for row in rows)
    chunk_observed_us: dict[int, float] = {}
    for row in rows:
        chunk_observed_us[row.chunk_id] = max(
            chunk_observed_us.get(row.chunk_id, 0.0), row.route_observed_us
        )
    chunks_by_observation: dict[float, list[int]] = {}
    for chunk_id, observed_us in chunk_observed_us.items():
        chunks_by_observation.setdefault(observed_us, []).append(chunk_id)
    observation_times = sorted(chunks_by_observation)
    target_contributions = math.ceil(len(rows) * prefix_fraction)
    prefix_chunks: set[int] = set()
    observed_contributions = 0
    causal_observation_us: float | None = None
    for observed_us in observation_times[:-1]:
        for chunk_id in chunks_by_observation[observed_us]:
            prefix_chunks.add(chunk_id)
            observed_contributions += chunk_counts[chunk_id]
        causal_observation_us = observed_us
        if observed_contributions >= target_contributions:
            break
    if not prefix_chunks or observed_contributions < target_contributions:
        return None, None, None, None
    prefix = Counter(row.target_rank for row in rows if row.chunk_id in prefix_chunks)
    future = Counter(row.target_rank for row in rows if row.chunk_id not in prefix_chunks)
    bins = list(range(num_ranks))
    prefix_distribution = _distribution(prefix, bins)
    future_distribution = _distribution(future, bins)
    total_variation = 0.5 * sum(
        abs(left - right)
        for left, right in zip(prefix_distribution, future_distribution)
    )
    prefix_dominant = min(
        rank for rank, count in prefix.items() if count == max(prefix.values())
    )
    future_dominant = min(
        rank for rank, count in future.items() if count == max(future.values())
    )
    prefix_contribution_fraction = sum(prefix.values()) / (
        sum(prefix.values()) + sum(future.values())
    )
    return (
        1.0 - total_variation,
        prefix_dominant == future_dominant,
        prefix_contribution_fraction,
        causal_observation_us,
    )


@dataclass(frozen=True)
class RequestCensus:
    model: str
    model_revision: str
    tenant_id: str
    request_id: str
    document_id: str
    prompt_hash: str
    split: str
    role: str
    traffic_class: str
    phase: str
    prompt_tokens: int
    placement_id: str
    contribution_count: int
    token_event_count: int
    layer_count: int
    chunk_count: int
    activated_expert_count: int
    activated_rank_count: int
    expert_max_share: float
    rank_max_share: float | None
    expert_normalized_entropy: float
    rank_normalized_entropy: float | None
    rank_imbalance_factor: float | None
    prefix_future_rank_persistence: float | None
    prefix_future_dominant_rank_match: bool | None
    observed_prefix_contribution_fraction: float | None
    causal_observation_us: float | None
    remaining_service_work_fraction: float | None
    causal_action_eligible: bool | None


def build_request_census(
    rows: Iterable[RouteContribution],
    *,
    num_experts: Mapping[str, int],
    num_ranks: int,
    prefix_fraction: float,
    require_rank_binding: bool = True,
) -> list[RequestCensus]:
    if num_ranks <= 0:
        raise ProtocolError("num_ranks must be positive")
    grouped: dict[tuple[str, str], list[RouteContribution]] = {}
    for row in rows:
        grouped.setdefault(row.request_key, []).append(row)
    if not grouped:
        raise ProtocolError("cannot census an empty route ledger")

    output: list[RequestCensus] = []
    for request_rows in grouped.values():
        first = request_rows[0]
        expert_counts = Counter(row.expert_id for row in request_rows)
        if require_rank_binding:
            if any(row.target_rank < 0 or row.target_rank >= num_ranks for row in request_rows):
                raise ProtocolError(
                    f"request {first.request_id} contains a target rank outside [0, {num_ranks})"
                )
            rank_counts = Counter(row.target_rank for row in request_rows)
            (
                persistence,
                dominant_match,
                observed_fraction,
                causal_observation_us,
            ) = _prefix_future_persistence(
                request_rows,
                prefix_fraction=prefix_fraction,
                num_ranks=num_ranks,
            )
            rank_max_share = _maximum_share(rank_counts)
            rank_entropy = _normalized_entropy(rank_counts, num_ranks)
            rank_imbalance = rank_max_share * num_ranks
            activated_ranks = len(rank_counts)
        else:
            persistence = dominant_match = observed_fraction = causal_observation_us = None
            rank_max_share = rank_entropy = rank_imbalance = None
            activated_ranks = 0

        output.append(
            RequestCensus(
                model=first.model,
                model_revision=first.model_revision,
                tenant_id=first.tenant_id,
                request_id=first.request_id,
                document_id=first.document_id,
                prompt_hash=first.prompt_hash,
                split=first.split,
                role=first.role,
                traffic_class=first.traffic_class,
                phase=first.phase,
                prompt_tokens=first.prompt_tokens,
                placement_id=first.placement_id,
                contribution_count=len(request_rows),
                token_event_count=len({row.token_event_key for row in request_rows}),
                layer_count=len({row.layer_id for row in request_rows}),
                chunk_count=len({row.chunk_id for row in request_rows}),
                activated_expert_count=len(expert_counts),
                activated_rank_count=activated_ranks,
                expert_max_share=_maximum_share(expert_counts),
                rank_max_share=rank_max_share,
                expert_normalized_entropy=_normalized_entropy(
                    expert_counts, num_experts[first.model]
                ),
                rank_normalized_entropy=rank_entropy,
                rank_imbalance_factor=rank_imbalance,
                prefix_future_rank_persistence=persistence,
                prefix_future_dominant_rank_match=dominant_match,
                observed_prefix_contribution_fraction=observed_fraction,
                causal_observation_us=causal_observation_us,
                remaining_service_work_fraction=None,
                causal_action_eligible=None,
            )
        )
    return sorted(output, key=lambda row: (row.model, row.tenant_id, row.request_id))


def summarize_census(rows: Iterable[RequestCensus]) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[RequestCensus]] = {}
    for row in rows:
        key = (row.model, row.split, row.role, row.traffic_class, row.phase, row.prompt_tokens)
        grouped.setdefault(key, []).append(row)
    summaries: list[dict[str, object]] = []
    for key, group in sorted(grouped.items()):
        persistence = [
            row.prefix_future_rank_persistence
            for row in group
            if row.prefix_future_rank_persistence is not None
        ]
        dominant_matches = [
            row.prefix_future_dominant_rank_match
            for row in group
            if row.prefix_future_dominant_rank_match is not None
        ]
        rank_shares = [row.rank_max_share for row in group if row.rank_max_share is not None]
        summaries.append(
            {
                "model": key[0],
                "split": key[1],
                "role": key[2],
                "traffic_class": key[3],
                "phase": key[4],
                "prompt_tokens": key[5],
                "request_count": len(group),
                "unique_document_count": len({row.document_id for row in group}),
                "unique_prompt_hash_count": len({row.prompt_hash for row in group}),
                "independent_document_prompt_cluster_count": len(
                    {(row.document_id, row.prompt_hash) for row in group}
                ),
                "median_expert_max_share": statistics.median(
                    row.expert_max_share for row in group
                ),
                "median_rank_max_share": (
                    statistics.median(rank_shares) if rank_shares else None
                ),
                "median_prefix_future_rank_persistence": (
                    statistics.median(persistence) if persistence else None
                ),
                "dominant_rank_match_rate": (
                    sum(bool(value) for value in dominant_matches) / len(dominant_matches)
                    if dominant_matches
                    else None
                ),
            }
        )
    return summaries


def _load_dispatch_bindings(
    config: Mapping[str, object], config_path: Path
) -> dict[str, dict[int, frozenset[tuple[int, str, str]]]]:
    target = config["target_system"]
    snapshot_value = str(target["placement_snapshot_path"])
    expected_hash = str(target["placement_snapshot_sha256"])
    if snapshot_value.startswith("UNRESOLVED_") or expected_hash.startswith(
        "UNRESOLVED_"
    ):
        raise ProtocolError(
            "formal rank census requires a resolved placement snapshot path and hash"
        )
    snapshot_path = Path(snapshot_value)
    if not snapshot_path.is_absolute():
        snapshot_path = config_path.parent / snapshot_path
    payload_bytes = snapshot_path.read_bytes()
    actual_hash = hashlib.sha256(payload_bytes).hexdigest()
    if actual_hash != expected_hash:
        raise ProtocolError(
            f"placement snapshot hash mismatch: {actual_hash} != {expected_hash}"
        )
    try:
        payload = json.loads(
            payload_bytes,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except json.JSONDecodeError as exc:
        raise ProtocolError("placement snapshot is invalid JSON") from exc
    if not isinstance(payload, Mapping) or set(payload) != {
        "schema",
        "placement_id",
        "ep_size",
        "models",
    }:
        raise ProtocolError("placement snapshot top-level fields changed")
    if payload.get("schema") != "routeshield-placement-v1":
        raise ProtocolError("unsupported placement snapshot schema")
    if payload.get("placement_id") != target["placement_id"]:
        raise ProtocolError("placement snapshot ID does not match frozen config")
    ep_size = int(target["ep_size"])
    if int(payload.get("ep_size", -1)) != ep_size:
        raise ProtocolError("placement snapshot EP size does not match frozen config")

    bindings: dict[str, dict[int, frozenset[tuple[int, str, str]]]] = {}
    models = payload.get("models")
    if not isinstance(models, Mapping):
        raise ProtocolError("placement snapshot models must be an object")
    expected_model_keys = {str(model["key"]) for model in config["models"]}
    if set(models) != expected_model_keys:
        raise ProtocolError("placement snapshot model set differs from frozen config")
    for model in config["models"]:
        model_key = str(model["key"])
        raw_model = models.get(model_key)
        if not isinstance(raw_model, Mapping):
            raise ProtocolError(f"placement snapshot missing model {model_key}")
        expected_experts = set(range(int(model["num_experts"])))
        if set(raw_model) != {str(expert_id) for expert_id in expected_experts}:
            raise ProtocolError(
                f"placement snapshot does not close all experts for {model_key}"
            )
        model_bindings: dict[int, frozenset[tuple[int, str, str]]] = {}
        for expert_id in expected_experts:
            raw_replicas = raw_model[str(expert_id)]
            if not isinstance(raw_replicas, list) or not raw_replicas:
                raise ProtocolError("each expert must bind to at least one replica")
            parsed: set[tuple[int, str, str]] = set()
            for replica in raw_replicas:
                if not isinstance(replica, Mapping) or set(replica) != {
                    "target_rank",
                    "replica_instance_id",
                    "device_uuid",
                }:
                    raise ProtocolError("placement replica fields changed")
                raw_rank = replica["target_rank"]
                replica_id = replica["replica_instance_id"]
                device_uuid = replica["device_uuid"]
                if isinstance(raw_rank, bool) or not isinstance(raw_rank, int):
                    raise ProtocolError("placement target_rank must be an integer")
                if (
                    not isinstance(replica_id, str)
                    or not replica_id
                    or replica_id.startswith(("UNBOUND", "UNRESOLVED_"))
                    or not isinstance(device_uuid, str)
                    or not device_uuid
                    or device_uuid.startswith(("UNBOUND", "UNRESOLVED_"))
                ):
                    raise ProtocolError("placement replica identity must be non-empty")
                parsed.add((raw_rank, replica_id, device_uuid))
            if len(parsed) != len(raw_replicas):
                raise ProtocolError("placement snapshot contains duplicate replicas")
            if any(rank < 0 or rank >= ep_size for rank, _, _ in parsed):
                raise ProtocolError("placement snapshot contains an out-of-range rank")
            model_bindings[expert_id] = frozenset(parsed)
        bindings[model_key] = model_bindings
    return bindings


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--routes", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--development-expert-only",
        action="store_true",
        help="allow target_rank=-1 and suppress all rank metrics",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = load_config(args.config)
    models = {row["key"]: row for row in config["models"]}
    expected_topk = {key: int(row["top_k"]) for key, row in models.items()}
    expected_revisions = {key: str(row["revision"]) for key, row in models.items()}
    expected_tokenizers = {
        key: str(row["tokenizer_sha256"]) for key, row in models.items()
    }
    num_experts = {key: int(row["num_experts"]) for key, row in models.items()}
    require_rank = not args.development_expert_only
    dispatch_bindings = None
    expected_events = None
    if require_rank:
        expected_placement = str(config["target_system"]["placement_id"])
        if expected_placement.startswith("UNRESOLVED_"):
            raise ProtocolError("formal rank census requires a resolved config placement_id")
        config_path = Path(args.config).resolve()
        route_hash = str(config["required_evidence"]["tenant_route_ledger_sha256"])
        if route_hash.startswith("UNRESOLVED_"):
            raise ProtocolError("formal census requires a frozen tenant route ledger hash")
        if hashlib.sha256(Path(args.routes).read_bytes()).hexdigest() != route_hash:
            raise ProtocolError("tenant route ledger hash mismatch")
        dispatch_bindings = _load_dispatch_bindings(config, config_path)
        event_value = str(
            config["required_evidence"]["expected_route_event_manifest_path"]
        )
        event_hash = str(
            config["required_evidence"]["expected_route_event_manifest_sha256"]
        )
        if event_value.startswith("UNRESOLVED_") or event_hash.startswith(
            "UNRESOLVED_"
        ):
            raise ProtocolError("formal census requires a frozen route-event manifest")
        event_path = Path(event_value)
        if not event_path.is_absolute():
            event_path = config_path.parent / event_path
        if hashlib.sha256(event_path.read_bytes()).hexdigest() != event_hash:
            raise ProtocolError("route-event manifest hash mismatch")
        expected_events = load_expected_events_jsonl(event_path)
    routes = load_route_csv(
        args.routes,
        expected_topk=expected_topk,
        expected_revisions=expected_revisions,
        expected_tokenizers=expected_tokenizers,
        num_experts=num_experts,
        expected_dispatch_bindings=dispatch_bindings,
        expected_events=expected_events,
        require_rank_binding=require_rank,
    )
    if require_rank:
        observed_placements = {row.placement_id for row in routes}
        if observed_placements != {expected_placement}:
            raise ProtocolError(
                f"route placement {sorted(observed_placements)} != frozen {expected_placement}"
            )
    census = build_request_census(
        routes,
        num_experts=num_experts,
        num_ranks=int(config["target_system"]["ep_size"]),
        prefix_fraction=float(
            config["causal_observation"]["prefix_contribution_fraction"]
        ),
        require_rank_binding=require_rank,
    )
    payload = {
        "schema": "routeshield-route-census-v1",
        "evidence_boundary": (
            "DEVELOPMENT_EXPERT_ROUTE_CENSUS_ONLY"
            if not require_rank
            else (
                "HASH_BOUND_PRODUCER_ASSERTED_EXECUTED_DISPATCH_CENSUS_"
                "NOT_INDEPENDENT_PHYSICAL_EP_TELEMETRY"
            )
        ),
        "formal_gate_result": False,
        "service_work_gate": {
            "status": "BLOCKED_MISSING_SERVICE_WEIGHTED_CAUSAL_LEDGER",
            "remaining_service_work_fraction": None,
            "causal_action_eligible": None,
        },
        "request_rows": [asdict(row) for row in census],
        "summaries": summarize_census(census),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {len(census)} request census rows to {output}")


if __name__ == "__main__":
    main()
