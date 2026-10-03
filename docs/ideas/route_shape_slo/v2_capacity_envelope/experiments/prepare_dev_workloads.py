#!/usr/bin/env python3
"""Materialize small producer-compatible development workloads.

The checked-in specification selects disjoint requests from the frozen OLMoE
manifest and replaces only request IDs, arrivals, deadlines, and development
scale.  The source formal manifest is never edited.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


class ConfigError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def arrival_trace_sha256(requests: Sequence[Mapping[str, Any]]) -> str:
    ordered = sorted(
        requests,
        key=lambda request: (float(request["arrival_us"]), int(request["sample_id"])),
    )
    payload = [
        {
            "sample_id": int(request["sample_id"]),
            "arrival_us": format(float(request["arrival_us"]), ".17g"),
            "deadline_us": format(float(request["deadline_us"]), ".17g"),
        }
        for request in ordered
    ]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _arrival_times(
    episode: Mapping[str, Any],
    source_requests: Sequence[Mapping[str, Any]],
) -> list[int]:
    count = int(episode["request_count"])
    arrival = episode.get("arrival")
    if not isinstance(arrival, Mapping):
        raise ConfigError("episode.arrival must be an object")
    kind = arrival.get("kind")
    if kind == "preserve_source_relative":
        origin = int(source_requests[0]["arrival_us"])
        values = [int(request["arrival_us"]) - origin for request in source_requests]
        if any(value < 0 for value in values):
            raise ConfigError("source-relative arrivals regress before the origin")
        return values
    if kind == "fixed_interarrival":
        spacing = int(arrival.get("interarrival_us", 0))
        if spacing <= 0:
            raise ConfigError("steady interarrival_us must be positive")
        return [index * spacing for index in range(count)]
    if kind == "fixed_bursts":
        burst_size = int(arrival.get("burst_size", 0))
        gap = int(arrival.get("burst_gap_us", 0))
        if burst_size <= 0 or gap <= 0:
            raise ConfigError("burst_size and burst_gap_us must be positive")
        return [(index // burst_size) * gap for index in range(count)]
    raise ConfigError(f"unsupported arrival kind: {kind!r}")


def materialize(spec_path: Path) -> dict[str, dict[str, Any]]:
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if spec.get("schema") != "route-capacity-envelope-dev-workload-v1":
        raise ConfigError("unexpected workload-spec schema")
    repo_root = next(parent for parent in spec_path.parents if (parent / ".git").exists())
    source_path = repo_root / str(spec["source_manifest"])
    if sha256_file(source_path) != str(spec["source_manifest_sha256"]):
        raise ConfigError("frozen source workload hash mismatch")
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if source.get("run_class") != "formal":
        raise ConfigError("source workload is not the frozen formal manifest")
    if source.get("model") != spec.get("model"):
        raise ConfigError("model identity differs from the frozen OLMoE source")

    source_requests = source.get("requests")
    episodes = spec.get("episodes")
    if not isinstance(source_requests, list) or not isinstance(episodes, list) or not episodes:
        raise ConfigError("source requests and episodes must be non-empty lists")

    used_indices: set[int] = set()
    used_documents: set[str] = set()
    output: dict[str, dict[str, Any]] = {}
    for raw_episode in episodes:
        if not isinstance(raw_episode, Mapping):
            raise ConfigError("episode must be an object")
        episode = dict(raw_episode)
        episode_id = str(episode.get("episode_id", ""))
        regime = str(episode.get("arrival_regime", ""))
        start = int(episode.get("source_request_start", -1))
        count = int(episode.get("request_count", 0))
        if not episode_id or regime not in {"steady", "bursty"}:
            raise ConfigError("episode identity/regime is invalid")
        indices = list(range(start, start + count))
        if count < 16 or count > 32 or start < 0 or indices[-1] >= len(source_requests):
            raise ConfigError("each episode must select 16-32 in-range source requests")
        if used_indices.intersection(indices):
            raise ConfigError("episodes reuse source requests")
        used_indices.update(indices)
        selected = [source_requests[index] for index in indices]
        arrivals = _arrival_times(episode, selected)
        requests: list[dict[str, Any]] = []
        for local_index, (source_index, arrival_us) in enumerate(zip(indices, arrivals)):
            request = copy.deepcopy(source_requests[source_index])
            request["request_id"] = f"{episode_id}-{local_index:03d}"
            request["arrival_us"] = arrival_us
            request["deadline_us"] = arrival_us + int(spec["deadline_offset_us"])
            document_id = str(request["document_id"])
            if document_id in used_documents:
                raise ConfigError("episodes reuse document identities")
            used_documents.add(document_id)
            requests.append(request)

        workload = copy.deepcopy(source)
        workload["run_class"] = "development"
        workload["expected_requests"] = count
        workload["generation"] = {
            "mode": "greedy",
            "do_sample": False,
            "max_decode_steps": int(spec["max_decode_steps"]),
        }
        workload["scheduler"] = {
            "max_batch_size": int(spec["max_batch_size"]),
            "arrival_trace_sha256": arrival_trace_sha256(requests),
            "arrival_source": {
                "kind": "deterministic_development_transform",
                "source_manifest_sha256": str(spec["source_manifest_sha256"]),
                "episode_id": episode_id,
                "arrival_regime": regime,
                "rule": copy.deepcopy(episode["arrival"]),
            },
        }
        workload["seed"] = int(spec["seed"])
        workload["max_prompt_tokens"] = int(spec["max_prompt_tokens"])
        workload["requests"] = requests
        audit_count = int(spec["serial_audit_requests_per_episode"])
        if audit_count <= 0 or audit_count > count:
            raise ConfigError("serial audit count is outside the episode")
        workload["serial_audit_request_ids"] = [
            request["request_id"] for request in requests[:audit_count]
        ]
        workload["route_capacity_envelope"] = {
            "episode_id": episode_id,
            "arrival_regime": regime,
            "request_document_disjoint_across_episodes": True,
            "evidence_scope": "development_custom_continuous_runtime",
            "serial_route_identity_semantics": (
                "per_layer_expert_assignment_multiset"
            ),
        }
        output[episode_id] = workload
    return output


def write_workloads(output_dir: Path, workloads: Mapping[str, Mapping[str, Any]]) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ConfigError(f"refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    for episode_id, workload in sorted(workloads.items()):
        path = output_dir / f"{episode_id}.json"
        path.write_text(
            json.dumps(workload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--spec",
        default=str(Path(__file__).with_name("olmoe_dev_workload.json")),
    )
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workloads = materialize(Path(args.spec).resolve())
    write_workloads(Path(args.output_dir).resolve(), workloads)
    print(
        json.dumps(
            {
                "status": "DEV_WORKLOADS_MATERIALIZED",
                "episodes": sorted(workloads),
                "request_counts": {
                    key: len(value["requests"]) for key, value in workloads.items()
                },
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
